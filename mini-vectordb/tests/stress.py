#!/usr/bin/env python3
"""Large-data / memory-pressure check.

Loads more vector data than a small container can hold, then verifies that the
data is still queryable and survives a restart. Run it under a cgroup limit,
e.g.  docker run --memory=256m ... python3 tests/stress.py

The report includes the container's cgroup memory.peak, which is what the
contest measures, so it also acts as a peak-memory probe.
"""
import os
import random
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXE = os.environ.get("MVDB_BIN", os.path.join(ROOT, "minivectordb.exe" if os.name == "nt" else "minivectordb"))
N = int(os.environ.get("N", "150000"))
DIM = int(os.environ.get("DIM", "128"))
BATCH = int(os.environ.get("BATCH", "2000"))
GROUPS = 8


def cgroup_mem():
    for path, label in (("/sys/fs/cgroup/memory.peak", "memory.peak"),
                        ("/sys/fs/cgroup/memory/memory.max_usage_in_bytes", "max_usage_in_bytes")):
        try:
            with open(path) as f:
                return label, int(f.read().strip())
        except Exception:
            continue
    return None, None


class Session:
    def __init__(self, datadir, fresh=False):
        if fresh and os.path.isdir(datadir):
            shutil.rmtree(datadir)
        os.makedirs(datadir, exist_ok=True)
        self.p = subprocess.Popen([EXE, "--data-dir=" + datadir], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        line = self.p.stdout.readline().strip()
        assert line == "READY", f"banner: {line!r}"

    def sql(self, stmt):
        self.p.stdin.write(stmt.rstrip(";") + ";\n")
        self.p.stdin.flush()
        rows, headers = [], None
        while True:
            line = self.p.stdout.readline()
            if line == "":
                raise RuntimeError("closed: " + self.p.stderr.read()[-400:])
            line = line.rstrip("\n")
            if line == "OK" or line.startswith("OK "):
                return headers, rows
            if line.startswith("ERROR "):
                raise RuntimeError(line)
            if headers is None:
                headers = line
            else:
                rows.append(line)

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=120)


def main():
    base = os.path.join(os.path.dirname(ROOT), "mvdb-stress")
    random.seed(17)
    pool = []
    for _ in range(4096):
        v = [random.uniform(-1, 1) for _ in range(DIM)]
        pool.append("[" + ",".join("%.4f" % x for x in v) + "]")

    label, mem0 = cgroup_mem()
    print(f"rows={N} dim={DIM} batch={BATCH} data={N * DIM * 4 / 1048576:.0f} MiB vectors")

    s = Session(base, fresh=True)
    s.sql(f"CREATE TABLE t (id BIGINT PRIMARY KEY, g VARCHAR(8), v VECTOR({DIM}))")
    t0 = time.time()
    for start in range(0, N, BATCH):
        vals = ",".join("(%d,'g%d','%s')" % (i, i % GROUPS, pool[i % len(pool)])
                        for i in range(start, min(start + BATCH, N)))
        s.sql("INSERT INTO t VALUES " + vals)
    t1 = time.time()
    print(f"insert : {t1 - t0:.2f}s ({N / (t1 - t0):.0f} rows/s)")

    q = pool[0]
    t2 = time.time()
    _, r = s.sql(f"SELECT id FROM t ORDER BY L2_DISTANCE(v,'{q}') LIMIT 10")
    t3 = time.time()
    print(f"topk   : {len(r)} rows in {(t3 - t2) * 1000:.0f} ms (first query, cold cache)")
    _, r2 = s.sql(f"SELECT id FROM t WHERE g = 'g3' ORDER BY L2_DISTANCE(v,'{q}') LIMIT 10")
    print(f"filter : {len(r2)} rows")
    s.close()

    label, mem1 = cgroup_mem()
    print(f"{label}: {mem1 / 1048576:.0f} MiB" if mem1 else "cgroup memory: n/a")

    t4 = time.time()
    s = Session(base)
    t5 = time.time()
    _, r = s.sql(f"SELECT id FROM t ORDER BY L2_DISTANCE(v,'{q}') LIMIT 10")
    print(f"restart: {t5 - t4:.2f}s, topk after restart {len(r)} rows")
    _, r3 = s.sql("SELECT id FROM t WHERE g = 'g0' ORDER BY L2_DISTANCE(v,'" + q + "') LIMIT 5")
    print(f"filter after restart: {len(r3)} rows")
    s.close()

    dbsize = sum(os.path.getsize(os.path.join(base, f)) for f in os.listdir(base))
    print(f"/data  : {dbsize / 1048576:.0f} MiB")

    ok = len(r) == 10 and len(r3) == 5
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
