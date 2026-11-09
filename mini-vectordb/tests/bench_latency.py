#!/usr/bin/env python3
"""Server-side latency probe.

Runs the contest workload and parses the per-statement timings the server
writes to stderr (MVDB_DEBUG=1), separating filtered from unfiltered TopK.
"""
import os
import random
import shutil
import statistics
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
GROUPS = int(os.environ.get("GROUPS", "4"))
DBG = os.path.join(os.path.dirname(ROOT), "mvdb-latency.log")


def vec(v):
    return "[" + ",".join("%.5f" % x for x in v) + "]"


def main():
    base = os.path.join(os.path.dirname(ROOT), "mvdb-latency")
    if os.path.isdir(base):
        shutil.rmtree(base)
    os.makedirs(base)
    env = dict(os.environ)
    env["MVDB_DEBUG"] = "1"
    log = open(DBG, "w")
    p = subprocess.Popen([EXE, "--data-dir=" + base], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=log, text=True, bufsize=1, env=env)
    assert p.stdout.readline().strip() == "READY"

    def sql(stmt, ret=False):
        p.stdin.write(stmt.rstrip(";") + ";\n")
        p.stdin.flush()
        rows, headers = [], None
        while True:
            line = p.stdout.readline()
            if line == "":
                raise RuntimeError("closed")
            line = line.rstrip("\n")
            if line == "OK" or line.startswith("OK "):
                return headers, rows
            if line.startswith("ERROR"):
                raise RuntimeError(line)
            if headers is None:
                headers = line
            else:
                rows.append(line)

    random.seed(11)
    data = [[random.uniform(-1, 1) for _ in range(DIM)] for _ in range(N)]
    sql(f"CREATE TABLE t (id BIGINT PRIMARY KEY, g VARCHAR(8), v VECTOR({DIM}))")
    t0 = time.time()
    for i in range(0, N, 100):
        vals = ",".join("(%d,'g%d','%s')" % (j, j % GROUPS, vec(data[j])) for j in range(i, min(i + 100, N)))
        sql("INSERT INTO t VALUES " + vals)
    t1 = time.time()
    sql("CREATE VECTOR INDEX idx ON t(v)")
    t2 = time.time()
    sql("SET vector_mode = 'ann'")
    qs = [[random.uniform(-1, 1) for _ in range(DIM)] for _ in range(Q)]
    t3 = time.time()
    for i, q in enumerate(qs):
        if i % 2 == 0:
            sql("SELECT id, L2_DISTANCE(v,'%s') AS d FROM t WHERE g = 'g%d' ORDER BY d LIMIT %d" % (vec(q), i % GROUPS, K))
        else:
            sql("SELECT id, L2_DISTANCE(v,'%s') AS d FROM t ORDER BY d LIMIT %d" % (vec(q), K))
    t4 = time.time()
    p.stdin.close()
    p.wait()
    log.close()

    filt, plain = [], []
    for line in open(DBG):
        if not line.startswith("["):
            continue
        us = int(line.split("]")[0].strip("[").split()[0])
        body = line.split("] ", 1)[-1]
        if "L2_DISTANCE" not in body:
            continue
        (filt if "WHERE" in body else plain).append(us)

    def pct(v, q):
        v = sorted(v)
        return v[min(len(v) - 1, int(len(v) * q))]

    print("rows=%d dim=%d queries=%d k=%d" % (N, DIM, Q, K))
    print("write   : %.3fs (%.4f ms/row, %d batches with fsync)" % (t1 - t0, (t1 - t0) / N * 1000, (N + 99) // 100))
    print("index   : %.3fs" % (t2 - t1))
    print("query   : %.3fs total (%.1f us/query end-to-end incl. driver IPC)" % (t4 - t3, (t4 - t3) / Q * 1e6))
    for name, v in (("plain", plain), ("filtered", filt)):
        if v:
            print("%-8s n=%d  median %d us  p90 %d us  mean %.1f us" % (name, len(v), pct(v, 0.5), pct(v, 0.9),
                                                                       statistics.mean(v)))
    wal = os.path.join(base, "wal.log")
    if os.path.exists(wal):
        print("wal size: %.2f MB" % (os.path.getsize(wal) / 1048576.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
