#!/usr/bin/env python3
"""Functional test-suite for MiniVectorDB (stdin/stdout protocol)."""
import os
import random
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXE = os.environ.get("MVDB_BIN", os.path.join(ROOT, "minivectordb.exe" if os.name == "nt" else "minivectordb"))


class Session:
    def __init__(self, datadir, fresh=False):
        self.datadir = datadir
        if fresh and os.path.isdir(datadir):
            shutil.rmtree(datadir)
        os.makedirs(datadir, exist_ok=True)
        self.p = subprocess.Popen(
            [EXE, "--data-dir=" + datadir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        line = self.p.stdout.readline().strip()
        assert line == "READY", f"bad banner: {line!r}"

    def sql(self, stmt):
        """Returns (headers, rows) or raises RuntimeError with error text."""
        if not stmt.rstrip().endswith(";"):
            stmt = stmt.rstrip() + ";"
        self.p.stdin.write(stmt + "\n")
        self.p.stdin.flush()
        rows = []
        headers = None
        while True:
            line = self.p.stdout.readline()
            if line == "":
                raise RuntimeError("server closed; stderr=" + self.p.stderr.read())
            line = line.rstrip("\n")
            if line == "OK":
                return headers, rows
            if line.startswith("ERROR "):
                raise RuntimeError(line)
            if headers is None:
                headers = line.split("\t") if line != "" else []
            else:
                rows.append(line.split("\t") if line != "" else [])

    def close(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        self.p.wait(timeout=30)

    def kill(self):
        self.p.kill()
        self.p.wait(timeout=30)


def approx(a, b, eps=1e-4):
    return abs(float(a) - float(b)) <= eps


PASS = 0
FAIL = 0
FAILURES = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL {name} {detail}")


def expect_error(name, fn, needle=""):
    try:
        fn()
    except RuntimeError as e:
        check(name, needle in str(e), f"got {e}")
        return
    check(name, False, "expected an error")


# --------------------------------------------------------------------------- tests

def test_basic(d):
    s = Session(d, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, name VARCHAR(10), v VECTOR(3))")
    s.sql("INSERT INTO t VALUES (1,'a','[1,2,3]'),(2,'b','[4,5,6]'),(3,'c','[0,0,0]')")
    h, r = s.sql("SELECT * FROM t")
    check("basic.header", h == ["id", "name", "v"], str(h))
    check("basic.rows", len(r) == 3 and r[0] == ["1", "a", "[1,2,3]"], str(r))
    h, r = s.sql("SELECT id, name FROM t WHERE id >= 2 ORDER BY id DESC")
    check("basic.order", r == [["3", "c"], ["2", "b"]], str(r))
    h, r = s.sql("SELECT id FROM t WHERE name = 'b'")
    check("basic.varchar-eq", r == [["2"]], str(r))
    h, r = s.sql("SELECT id FROM t ORDER BY id LIMIT 2")
    check("basic.limit", r == [["1"], ["2"]], str(r))
    h, r = s.sql("SELECT id, L2_DISTANCE(v, '[1,2,4]') AS d FROM t ORDER BY d LIMIT 1")
    check("basic.topk", r[0][0] == "1" and approx(r[0][1], 1.0), str(r))
    s.close()
    # persistence
    s = Session(d)
    h, r = s.sql("SELECT id, name FROM t ORDER BY id")
    check("basic.restart", len(r) == 3 and r[2] == ["3", "c"], str(r))
    s.close()


def test_types(d):
    s = Session(d, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(3), v VECTOR(2))")
    s.sql("INSERT INTO t VALUES (1,'中文', '[1,2]')")
    h, r = s.sql("SELECT s FROM t")
    check("types.utf8", r == [["中文"]], str(r))
    expect_error("types.varchar-too-long", lambda: s.sql("INSERT INTO t VALUES (2,'abcd','[1,2]')"), "1406")
    expect_error("types.vec-dim", lambda: s.sql("INSERT INTO t VALUES (3,'a','[1,2,3]')"), "1366")
    expect_error("types.vec-nan", lambda: s.sql("INSERT INTO t VALUES (4,'a','[nan,1]')"), "1366")
    expect_error("types.vec-inf", lambda: s.sql("INSERT INTO t VALUES (5,'a','[inf,1]')"), "1366")
    expect_error("types.dup-pk", lambda: s.sql("INSERT INTO t VALUES (1,'a','[1,2]')"), "1062")
    expect_error("types.dup-pk-multi",
                 lambda: s.sql("INSERT INTO t VALUES (10,'a','[1,2]'),(10,'b','[1,2]')"), "1062")
    h, r = s.sql("SELECT id FROM t")
    check("types.partial-batch-rejected", len(r) == 1, str(r))
    expect_error("types.unknown-col", lambda: s.sql("SELECT nope FROM t"), "1054")
    expect_error("types.missing-table", lambda: s.sql("SELECT * FROM nope"), "1146")
    expect_error("types.bad-syntax", lambda: s.sql("SELEC * FROM t"), "1064")
    s.close()


def test_update_delete(d):
    s = Session(d, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, g VARCHAR(5), v VECTOR(2))")
    s.sql("INSERT INTO t VALUES (1,'x','[0,0]'),(2,'x','[1,1]'),(3,'y','[2,2]')")
    s.sql("UPDATE t SET g = 'z' WHERE id = 2")
    h, r = s.sql("SELECT id,g FROM t ORDER BY id")
    check("dml.update", r == [["1", "x"], ["2", "z"], ["3", "y"]], str(r))
    s.sql("UPDATE t SET v = '[9,9]' WHERE id = 1")
    h, r = s.sql("SELECT id, L2_DISTANCE(v,'[9,9]') AS d FROM t ORDER BY d LIMIT 1")
    check("dml.update-vector", r[0][0] == "1" and approx(r[0][1], 0.0), str(r))
    s.sql("DELETE FROM t WHERE g = 'z'")
    h, r = s.sql("SELECT id FROM t ORDER BY id")
    check("dml.delete", r == [["1"], ["3"]], str(r))
    s.sql("INSERT INTO t VALUES (2,'q','[1,1]')")
    h, r = s.sql("SELECT g FROM t WHERE id = 2")
    check("dml.reinsert", r == [["q"]], str(r))
    h, r = s.sql("SELECT id FROM t WHERE g = 'x'")
    check("dml.cache-after-delete", r == [["1"]], str(r))
    s.close()


def test_filtered_topk(d, n=400, dim=16):
    random.seed(7)
    s = Session(d, fresh=True)
    s.sql(f"CREATE TABLE t (id BIGINT PRIMARY KEY, g VARCHAR(8), v VECTOR({dim}))")
    data = []
    groups = ["a", "b", "c"]
    for i in range(n):
        vec = [round(random.uniform(-1, 1), 4) for _ in range(dim)]
        g = groups[i % 3]
        data.append((i, g, vec))
    batch = []
    for i, g, vec in data:
        batch.append(f"({i},'{g}','[{','.join(map(str, vec))}]')")
        if len(batch) == 100:
            s.sql("INSERT INTO t VALUES " + ",".join(batch))
            batch = []
    if batch:
        s.sql("INSERT INTO t VALUES " + ",".join(batch))

    q = [round(random.uniform(-1, 1), 4) for _ in range(dim)]
    qs = "[" + ",".join(map(str, q)) + "]"
    h, r = s.sql(f"SELECT id, L2_DISTANCE(v,'{qs}') AS d FROM t ORDER BY d LIMIT 10")
    truth = sorted(data, key=lambda x: (sum((a - b) ** 2 for a, b in zip(x[2], q)), x[0]))[:10]
    got = [(int(x[0]), float(x[1])) for x in r]
    ok = [g[0] for g in got] == [t[0] for t in truth]
    check("topk.exact-ids", ok, f"got={[g[0] for g in got]} want={[t[0] for t in truth]}")
    check("topk.dist", all(approx(g[1], sum((a - b) ** 2 for a, b in zip(t[2], q)) ** 0.5)
                       for g, t in zip(got, truth)), str(got[:2]))

    h, r = s.sql(f"SELECT id FROM t WHERE g = 'b' ORDER BY L2_DISTANCE(v,'{qs}') LIMIT 5")
    truth = sorted([x for x in data if x[1] == "b"],
                   key=lambda x: (sum((a - b) ** 2 for a, b in zip(x[2], q)), x[0]))[:5]
    check("topk.filtered", [int(x[0]) for x in r] == [t[0] for t in truth],
          str([x[0] for x in r]) + " vs " + str([t[0] for t in truth]))

    # tie-break on identical vectors -> primary key ascending
    s.sql("CREATE TABLE tie (id BIGINT PRIMARY KEY, v VECTOR(2))")
    s.sql("INSERT INTO tie VALUES (5,'[1,1]'),(3,'[1,1]'),(9,'[1,1]')")
    h, r = s.sql("SELECT id FROM tie ORDER BY L2_DISTANCE(v,'[1,1]') LIMIT 3")
    check("topk.tie", [x[0] for x in r] == ["3", "5", "9"], str(r))
    # fewer than K
    zeros = "[" + ",".join(["0"] * dim) + "]"
    h, r = s.sql(f"SELECT id FROM t WHERE g = 'zzz' ORDER BY L2_DISTANCE(v,'{zeros}') LIMIT 5")
    check("topk.empty", r == [], str(r))
    s.close()


def test_transactions(d):
    s = Session(d, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2))")
    s.sql("INSERT INTO t VALUES (1,'[0,0]')")
    s.sql("BEGIN")
    s.sql("INSERT INTO t VALUES (2,'[1,1]')")
    h, r = s.sql("SELECT id FROM t ORDER BY id")
    check("txn.read-own-write", r == [["1"], ["2"]], str(r))
    h, r = s.sql("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[1,1]') LIMIT 2")
    check("txn.vector-read-own-write", [x[0] for x in r] == ["2", "1"], str(r))
    s.sql("ROLLBACK")
    h, r = s.sql("SELECT id FROM t")
    check("txn.rollback", r == [["1"]], str(r))
    h, r = s.sql("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[1,1]') LIMIT 2")
    check("txn.rollback-vector", [x[0] for x in r] == ["1"], str(r))

    s.sql("BEGIN")
    s.sql("INSERT INTO t VALUES (3,'[2,2]')")
    s.sql("COMMIT")
    h, r = s.sql("SELECT id FROM t ORDER BY id")
    check("txn.commit", r == [["1"], ["3"]], str(r))

    # statement failure inside a transaction keeps earlier statements usable
    s.sql("BEGIN")
    s.sql("INSERT INTO t VALUES (4,'[4,4]')")
    expect_error("txn.stmt-fail", lambda: s.sql("INSERT INTO t VALUES (1,'[5,5]')"), "1062")
    s.sql("INSERT INTO t VALUES (5,'[5,5]')")
    s.sql("COMMIT")
    h, r = s.sql("SELECT id FROM t ORDER BY id")
    check("txn.after-stmt-fail", r == [["1"], ["3"], ["4"], ["5"]], str(r))
    s.close()


def test_crash_recovery(d):
    s = Session(d, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2))")
    s.sql("INSERT INTO t VALUES (1,'[0,0]')")
    s.sql("BEGIN")
    s.sql("INSERT INTO t VALUES (2,'[1,1]')")
    s.kill()
    s = Session(d)
    h, r = s.sql("SELECT id FROM t ORDER BY id")
    check("crash.rollback-uncommitted", r == [["1"]], str(r))
    s.sql("INSERT INTO t VALUES (3,'[3,3]')")
    s.kill()
    s = Session(d)
    h, r = s.sql("SELECT id FROM t ORDER BY id")
    check("crash.keep-committed", r == [["1"], ["3"]], str(r))
    s.close()


def test_index(d, n=1000, dim=8):
    random.seed(11)
    s = Session(d, fresh=True)
    s.sql(f"CREATE TABLE t (id BIGINT PRIMARY KEY, g VARCHAR(4), v VECTOR({dim}))")
    data = {}
    batch = []
    for i in range(n):
        vec = [round(random.uniform(-1, 1), 4) for _ in range(dim)]
        data[i] = vec
        batch.append(f"({i},'{'ab'[i % 2]}','[{','.join(map(str, vec))}]')")
        if len(batch) == 200:
            s.sql("INSERT INTO t VALUES " + ",".join(batch))
            batch = []
    if batch:
        s.sql("INSERT INTO t VALUES " + ",".join(batch))
    s.sql("CREATE VECTOR INDEX idx ON t(v)")
    # mutate after index creation
    s.sql("DELETE FROM t WHERE id < 50")
    for i in range(50, 60):
        del data[i]
    s.sql("INSERT INTO t VALUES (5000,'a','[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]')")
    data[5000] = [0.5] * dim
    s.sql("UPDATE t SET v = '[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]' WHERE id = 900")
    data[900] = [0.1] * dim
    s.sql("SET vector_mode = 'ann'")
    hits = 0
    total = 0
    for _ in range(30):
        q = [round(random.uniform(-1, 1), 4) for _ in range(dim)]
        qs = "[" + ",".join(map(str, q)) + "]"
        h, r = s.sql(f"SELECT id FROM t ORDER BY L2_DISTANCE(v,'{qs}') LIMIT 10")
        got = [int(x[0]) for x in r]
        truth = sorted(data.items(), key=lambda kv: (sum((a - b) ** 2 for a, b in zip(kv[1], q)), kv[0]))[:10]
        want = [k for k, _ in truth]
        hits += len(set(got) & set(want))
        total += 10
        check("index.no-dup", len(got) == len(set(got)))
    recall = hits / total
    check("index.recall", recall >= 0.95, f"recall={recall}")
    # filtered ann query must respect the filter
    h, r = s.sql(f"SELECT id FROM t WHERE g = 'a' ORDER BY L2_DISTANCE(v,'[0,0,0,0,0,0,0,0]') LIMIT 5")
    ids = [int(x[0]) for x in r]
    check("index.filtered-ann", all(i % 2 == 0 for i in ids), str(ids))
    s.close()


def main():
    base = os.path.join(os.path.dirname(ROOT), "mvdb-testdata")
    if os.path.isdir(base):
        shutil.rmtree(base)
    os.makedirs(base)
    tests = [
        ("basic", test_basic),
        ("types", test_types),
        ("update_delete", test_update_delete),
        ("filtered_topk", test_filtered_topk),
        ("transactions", test_transactions),
        ("crash_recovery", test_crash_recovery),
        ("index", test_index),
    ]
    for name, fn in tests:
        d = os.path.join(base, name)
        t0 = time.time()
        print(f"[{name}]")
        try:
            fn(d)
        except Exception as e:
            check(name + ".exception", False, repr(e))
        print(f"  ({time.time()-t0:.2f}s)")
    print(f"\npassed={PASS} failed={FAIL}")
    if FAILURES:
        for f in FAILURES:
            print(" -", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
