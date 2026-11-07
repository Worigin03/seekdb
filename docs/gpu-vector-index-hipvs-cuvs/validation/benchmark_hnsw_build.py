"""Compare seekdb HNSW DDL backends on the same running Radeon instance."""

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import tempfile
import time
import uuid
from pathlib import Path


def log_identity(stat):
    return stat.st_dev, stat.st_ino


def log_events_since(path, checkpoint, markers):
    """Read matching lines from the checkpoint through any retained rotations."""
    origin = log_identity(checkpoint)
    rotated_name = re.compile(re.escape(path.name) + r"\.\d{17}")
    for _ in range(3):
        started = False
        events = []
        try:
            active = log_identity(path.stat())
            files = []
            if active != origin:
                # Seekdb renames the old file to <name>.YYYYMMDDhhmmssmmm.
                files = sorted(p for p in path.parent.iterdir() if rotated_name.fullmatch(p.name))
            for logfile in (*files, path):
                with logfile.open("rb") as log:
                    if not started:
                        info = os.fstat(log.fileno())
                        if log_identity(info) != origin:
                            continue
                        if info.st_size < checkpoint.st_size:
                            raise RuntimeError(f"Observer log was truncated during the trial: {path}")
                        log.seek(checkpoint.st_size)
                        started = True
                    events.extend(
                        line.decode("utf-8", errors="replace").strip()
                        for line in log if any(marker in line for marker in markers)
                    )
            if log_identity(path.stat()) != active:
                continue
        except FileNotFoundError:
            # The active name briefly disappears between rename and open.
            time.sleep(0.05)
            continue
        if not started:
            raise RuntimeError(f"Observer log checkpoint was recycled during the trial: {path}")
        return events
    raise RuntimeError(f"Observer log kept rotating while reading the trial: {path}")


def benchmark(args):
    database = "hipvs_build_" + uuid.uuid4().hex[:12]
    client = [
        args.mysql, "--protocol=TCP", "-h", args.host, "-P", str(args.port),
        "-u", args.user, "--local-infile=1", "--batch", "--skip-column-names",
        "--raw", "--connect-timeout=10",
    ]

    def sql(statement):
        result = subprocess.run(
            client, input="set ob_query_timeout=1800000000;\n" + statement + "\n",
            capture_output=True, text=True, check=False, timeout=1830,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        return result.stdout.splitlines()

    def vector(values):
        return "[" + ",".join(format(value, ".8g") for value in values) + "]"

    rng = random.Random(args.seed)
    centers = [[rng.gauss(0, 1) for _ in range(args.dim)] for _ in range(64)]

    def sample():
        return vector(value + rng.gauss(0, 0.25) for value in rng.choice(centers))

    version = sql("select version();")[0]
    sql(f"create database {database};")
    print(f"Database {database}, seekdb {version}", flush=True)
    base = f"{database}.base_vectors"
    table_schema = f"(id bigint primary key, v vector({args.dim}) not null)"
    sql(f"create table {base}{table_schema};")
    with tempfile.TemporaryDirectory(prefix="seekdb-hipvs-") as directory:
        data = Path(directory) / "base.tsv"
        with data.open("w", encoding="ascii", newline="\n") as output:
            for row in range(args.rows):
                output.write(f"{row}\t{sample()}\n")
        data_path = data.as_posix().replace("'", "''")
        sql(
            f"load data local infile '{data_path}' into table {base} "
            "fields terminated by '\\t' lines terminated by '\\n' (id,v);"
        )
    if sql(f"select count(*) from {base};") != [str(args.rows)]:
        raise RuntimeError("source row count differs from the requested row count")

    recall_probes = [sample() for _ in range(args.probes)]
    query_probes = [sample() for _ in range(args.queries)]

    def neighbors(table, mode, topk, probe_values):
        rows = sql("\n".join(
            f"select id from {table} order by l2_distance(v,'{probe}') {mode} limit {topk};"
            for probe in probe_values
        ))
        try:
            rows = [int(row) for row in rows]
        except ValueError as error:
            raise RuntimeError(f"non-integer neighbor ID returned by {table}") from error
        if any(row < 0 or row >= args.rows for row in rows):
            raise RuntimeError(f"out-of-range neighbor ID returned by {table}")
        groups = [rows[i:i + topk] for i in range(0, len(rows), topk)]
        if (len(rows) != len(probe_values) * topk
                or any(len(group) != topk or len(set(group)) != topk for group in groups)):
            raise RuntimeError(f"expected {topk} unique results per probe from {table}")
        return groups

    # One exact Top-100 scan supplies ordered ground truth for both recall levels.
    exact = neighbors(base, "", 100, recall_probes)
    samples = {"vsag": [], "cuvs": []}
    for run in range(args.repeats):
        order = ("vsag", "cuvs") if run % 2 == 0 else ("cuvs", "vsag")
        for backend in order:
            table = f"{database}.{backend}_{run}"
            sql(f"create table {table}{table_schema}; insert into {table} select * from {base};")
            if sql(f"select count(*) from {table};") != [str(args.rows)]:
                raise RuntimeError(f"wrong pre-DDL row count for {table}")
            log_start = args.log.stat()
            start = time.perf_counter()
            sql(
                f"create vector index idx on {table}(v) with "
                f"(distance=l2,type=hnsw,lib={backend},"
                "m=16,ef_construction=200,ef_search=64);"
            )
            ddl_seconds = time.perf_counter() - start
            # The contest build interval ends after, not before, the first query.
            neighbors(table, "approximate", 10, recall_probes[:1])
            seconds = time.perf_counter() - start
            snapshot_events = log_events_since(
                args.log, log_start,
                (b" serialized GPU CAGRA snapshot", b"GPU CAGRA build failed during snapshot serialize"),
            )
            snapshots = [
                line for line in snapshot_events
                if " serialized GPU CAGRA snapshot" in line
                and re.search(rf"\bn={args.rows}\b", line)
                and re.search(rf"\bdim32={args.dim}\b", line)
                and re.search(r"\bgraph_bytes=[1-9]\d*\b", line)
            ]
            if backend == "cuvs" and not any("built=true" in line for line in snapshots):
                events = "\n".join(snapshot_events[-5:]) or "(no CAGRA snapshot events)"
                raise RuntimeError(
                    f"{table}: DDL returned in {seconds:.3f}s, but the log did not verify "
                    f"GPU construction of {args.rows} vectors with dimension {args.dim}.\n"
                    f"Log: {args.log}\n{events}"
                )
            if backend == "vsag" and snapshots:
                raise RuntimeError(f"unexpected GPU snapshot during CPU DDL for {table}")

            recalls = {}
            query_qps = {}
            for topk in (10, 100):
                actual = neighbors(table, "approximate", topk, recall_probes)
                recalls[topk] = sum(
                    len(set(want[:topk]) & set(got))
                    for want, got in zip(exact, actual)
                ) / (topk * args.probes)
                if args.queries:
                    query_start = time.perf_counter()
                    neighbors(table, "approximate", topk, query_probes)
                    query_qps[topk] = args.queries / (time.perf_counter() - query_start)
            if backend == "cuvs" and log_events_since(args.log, log_start, (b"lazily built VSAG fallback",)):
                raise RuntimeError(f"{table}: VSAG fallback observed during the GPU trial")
            result = {
                "ddl_seconds": ddl_seconds,
                "build_seconds_with_first_query": seconds,
                "recall_at_10": recalls[10], "recall_at_100": recalls[100],
                "query_qps": query_qps, "gpu_snapshots": snapshots,
            }
            samples[backend].append(result)
            print(
                f"{table}: DDL+first query {seconds:.3f}s, "
                f"Recall@10 {recalls[10]:.4f}, Recall@100 {recalls[100]:.4f}"
                + (f", QPS@10 {query_qps[10]:.2f}, QPS@100 {query_qps[100]:.2f}"
                   if args.queries else ""),
                flush=True,
            )
            # Keep the unindexed source for inspection without retaining every
            # trial's CPU/GPU index and letting memory accumulate across rounds.
            sql(f"set recyclebin='off'; drop table {table};")

    median = {
        backend: statistics.median(run["build_seconds_with_first_query"] for run in runs)
        for backend, runs in samples.items()
    }
    summary = {
        "seekdb_version": version, "database": database, "rows": args.rows,
        "dim": args.dim, "probes": args.probes, "queries": args.queries,
        "seed": args.seed, "runs": samples,
        "median_build_seconds_with_first_query": median,
        "speedup": median["vsag"] / median["cuvs"],
    }
    if args.queries:
        summary["median_query_qps"] = {
            backend: {
                str(topk): statistics.median(run["query_qps"][topk] for run in runs)
                for topk in (10, 100)
            }
            for backend, runs in samples.items()
        }
        summary["query_speedup"] = {
            str(topk): summary["median_query_qps"]["cuvs"][str(topk)]
            / summary["median_query_qps"]["vsag"][str(topk)]
            for topk in (10, 100)
        }
    return summary


def find_observer_log(port):
    """Resolve the live Linux process log from its listening SQL socket."""
    proc = Path("/proc")
    listeners = set()
    for table in (proc / "net/tcp", proc / "net/tcp6"):
        try:
            rows = table.read_text(encoding="ascii").splitlines()[1:]
        except FileNotFoundError:
            continue
        for row in rows:
            fields = row.split()
            if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                listeners.add(f"socket:[{fields[9]}]")

    logs = set()
    if listeners:
        for process in proc.iterdir():
            if not process.name.isdecimal():
                continue
            try:
                descriptors = list((process / "fd").iterdir())
            except (FileNotFoundError, PermissionError):
                continue
            targets = set()
            for descriptor in descriptors:
                try:
                    targets.add(str(descriptor.readlink()))
                except (FileNotFoundError, PermissionError):
                    continue
            if targets & listeners:
                for target in targets:
                    path = Path(target)
                    if path.name in {"seekdb.log", "observer.log"} and path.is_file():
                        logs.add(path.resolve())
    if len(logs) != 1:
        found = ", ".join(map(str, sorted(logs))) or "none"
        raise ValueError(f"cannot identify a unique observer log for SQL port {port} (found {found})")
    return logs.pop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, help="observer log (default: discover by local SQL port)")
    parser.add_argument("--output", type=Path, help="new JSON result file (default: generated name)")
    parser.add_argument("--mysql", default="mysql")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2881)
    parser.add_argument("--user", default="root")
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--probes", type=int, default=100, help="exact-ground-truth recall queries")
    parser.add_argument("--queries", type=int, default=4096, help="single-row SQL queries per Top-k QPS run; 0 disables")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.rows < 1000 or min(args.dim, args.repeats, args.probes) < 1 or args.queries < 0:
        parser.error("at least 1000 rows, positive dimensions/repeats/probes, and nonnegative queries are required")
    if args.log is None:
        if args.host not in {"127.0.0.1", "::1", "localhost"}:
            parser.error("--log is required when the SQL host is remote")
        try:
            args.log = find_observer_log(args.port)
        except (OSError, ValueError) as error:
            parser.error(f"{error}; check that seekdb is running or pass --log with its live log path")
    if not args.log.is_file():
        parser.error(f"observer log does not exist or is not a regular file: {args.log}")
    if args.output is None:
        args.output = Path(f"hnsw-build-results-{uuid.uuid4().hex[:12]}.json")
    if args.output.exists():
        parser.error(f"result file already exists: {args.output}; choose another --output or omit it")
    print(f"Observer log {args.log}; result file {args.output}", flush=True)
    result = json.dumps(benchmark(args), indent=2) + "\n"
    with args.output.open("x", encoding="utf-8") as output:
        output.write(result)
    print(result)
