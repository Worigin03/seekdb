#!/usr/bin/env python3
"""Small benchmark mirroring the contest workload: 10k x 128d vectors,
100-row commit batches, then 1000 TopK queries (half filtered)."""
import os
import random
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXE = os.environ.get("MVDB_BIN", os.path.join(ROOT, "minivectordb.exe" if os.name == "nt" else "minivectordb"))

N = int(os.environ.get("N", "10000"))
DIM = int(os.environ.get("DIM", "128"))
Q = int(os.environ.get("Q", "1000"))
K = int(os.environ.get("K", "10"))
MODE = os.environ.get("MODE", "exact")


class Session:
    def __init__(self, datadir, fresh=False):
        if fresh and os.path.isdir(datadir):
            shutil.rmtree(datadir)
        os.makedirs(datadir, exist_ok=True)
        self.p = subprocess.Popen([EXE, "--data-dir=" + datadir], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        assert self.p.stdout.readline().strip() == "READY"

    def sql(self, stmt):
        self.p.stdin.write(stmt.rstrip(";") + ";\n")
        self.p.stdin.flush()
        headers = None
        rows = 0
        while True:
            line = self.p.stdout.readline()
            if line == "":
                raise RuntimeError("closed: " + self.p.stderr.read())
            line = line.rstrip("\n")
            if line == "OK" or line.startswith("OK "):
                return rows
            if line.startswith("ERROR "):
                raise RuntimeError(line)
            if headers is None:
                headers = line
            else:
                rows += 1

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=60)


def main():
    base = os.path.join(os.path.dirname(ROOT), "mvdb-benchdata")
    random.seed(42)
    data = [[random.uniform(-1, 1) for _ in range(DIM)] for _ in range(N)]
    groups = ["g0", "g1", "g2", "g3"]

    s = Session(base, fresh=True)
    s.sql(f"CREATE TABLE t (id BIGINT PRIMARY KEY, g VARCHAR(8), v VECTOR({DIM}))")
    t0 = time.time()
    for start in range(0, N, 100):
        vals = []
        for i in range(start, min(start + 100, N)):
            vec = "[" + ",".join("%.5f" % x for x in data[i]) + "]"
            vals.append(f"({i},'{groups[i % 4]}','{vec}')")
        s.sql("INSERT INTO t VALUES " + ",".join(vals))
    t1 = time.time()

    if MODE == "ann":
        t2 = time.time()
        s.sql("CREATE VECTOR INDEX idx ON t(v)")
        s.sql("SET vector_mode = 'ann'")
        t3 = time.time()
    else:
        t2 = t3 = t1

    queries = []
    for i in range(Q):
        q = [random.uniform(-1, 1) for _ in range(DIM)]
        qs = "[" + ",".join("%.5f" % x for x in q) + "]"
        if i % 2 == 0:
            queries.append(f"SELECT id, L2_DISTANCE(v,'{qs}') AS d FROM t WHERE g = '{groups[i % 4]}' ORDER BY d LIMIT {K}")
        else:
            queries.append(f"SELECT id, L2_DISTANCE(v,'{qs}') AS d FROM t ORDER BY d LIMIT {K}")
    t4 = time.time()
    rows = 0
    for sql in queries:
        rows += s.sql(sql)
    t5 = time.time()
    s.close()

    wal = os.path.join(base, "wal.log")
    print(f"rows={N} dim={DIM} queries={Q} k={K} mode={MODE}")
    print(f"write   : {t1-t0:8.3f}s  ({(t1-t0)/N*1000:.3f} ms/row)")
    if MODE == "ann":
        print(f"index   : {t3-t2:8.3f}s")
    print(f"query   : {t5-t4:8.3f}s  ({(t5-t4)/Q*1000:.3f} ms/query, {rows} rows)")
    print(f"wal size: {os.path.getsize(wal)/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
