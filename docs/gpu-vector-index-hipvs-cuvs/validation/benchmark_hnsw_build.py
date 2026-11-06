"""Compare seekdb HNSW DDL backends on the same running Radeon instance."""

import argparse
import json
import random
import re
import statistics
import subprocess
import tempfile
import time
import uuid
from pathlib import Path


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

    probes = [sample() for _ in range(args.probes)]

    def neighbors(table, mode):
        rows = sql("\n".join(
            f"select id from {table} order by l2_distance(v,'{probe}') {mode} limit 10;"
            for probe in probes
        ))
        groups = [set(rows[i:i + 10]) for i in range(0, len(rows), 10)]
        if len(rows) != args.probes * 10 or any(len(group) != 10 for group in groups):
            raise RuntimeError(f"expected 10 unique results per probe from {table}")
        return groups

    exact = neighbors(base, "")
    samples = {"vsag": [], "cuvs": []}
    for run in range(args.repeats):
        order = ("vsag", "cuvs") if run % 2 == 0 else ("cuvs", "vsag")
        for backend in order:
            table = f"{database}.{backend}_{run}"
            sql(f"create table {table}{table_schema}; insert into {table} select * from {base};")
            if sql(f"select count(*) from {table};") != [str(args.rows)]:
                raise RuntimeError(f"wrong pre-DDL row count for {table}")
            with args.log.open("r", encoding="utf-8", errors="replace") as log:
                log.seek(0, 2)
                start = time.perf_counter()
                sql(
                    f"create vector index idx on {table}(v) with "
                    f"(distance=l2,type=hnsw,lib={backend},"
                    "m=16,ef_construction=200,ef_search=200);"
                )
                seconds = time.perf_counter() - start
                snapshot_events = [
                    line.strip() for line in log
                    if " serialized GPU CAGRA snapshot" in line
                    or "GPU CAGRA build failed during snapshot serialize" in line
                ]
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
            actual = neighbors(table, "approximate")
            recall = sum(len(want & got) for want, got in zip(exact, actual)) / (10 * args.probes)
            result = {"ddl_seconds": seconds, "recall_at_10": recall, "gpu_snapshots": snapshots}
            samples[backend].append(result)
            print(f"{table}: DDL {seconds:.3f}s, Recall@10 {recall:.4f}", flush=True)

    median = {
        backend: statistics.median(run["ddl_seconds"] for run in runs)
        for backend, runs in samples.items()
    }
    return {
        "seekdb_version": version, "database": database, "rows": args.rows,
        "dim": args.dim, "probes": args.probes, "seed": args.seed, "runs": samples,
        "median_ddl_seconds": median, "speedup": median["vsag"] / median["cuvs"],
    }


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
    parser.add_argument("--probes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.rows < 1000 or min(args.dim, args.repeats, args.probes) < 1:
        parser.error("at least 1000 rows and positive dimensions, repeats, and probes are required")
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
