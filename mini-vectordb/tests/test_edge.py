#!/usr/bin/env python3
"""Boundary and contract tests: value limits, validation, atomicity, recovery."""
import math
import os
import random
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXE = os.environ.get("MVDB_BIN", os.path.join(ROOT, "minivectordb.exe" if os.name == "nt" else "minivectordb"))

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


class Session:
    def __init__(self, datadir, fresh=False):
        self.datadir = datadir
        if fresh and os.path.isdir(datadir):
            shutil.rmtree(datadir)
        os.makedirs(datadir, exist_ok=True)
        self.p = subprocess.Popen([EXE, "--data-dir=" + datadir], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        line = self.p.stdout.readline().strip()
        assert line == "READY", f"bad banner: {line!r}"

    def raw(self, stmt):
        """Returns (headers, rows, error-or-None)."""
        if not stmt.rstrip().endswith(";"):
            stmt = stmt.rstrip() + ";"
        self.p.stdin.write(stmt + "\n")
        self.p.stdin.flush()
        rows, headers = [], None
        while True:
            line = self.p.stdout.readline()
            if line == "":
                raise RuntimeError("server closed: " + self.p.stderr.read())
            line = line.rstrip("\n")
            if line == "OK" or line.startswith("OK "):
                return headers, rows, None
            if line.startswith("ERROR "):
                return headers, rows, line
            if headers is None:
                headers = line.split("\t") if line else []
            else:
                rows.append(line.split("\t") if line else [])

    def sql(self, stmt):
        h, r, e = self.raw(stmt)
        if e:
            raise RuntimeError(e)
        return h, r

    def err(self, stmt):
        _, _, e = self.raw(stmt)
        return e or ""

    def kill(self):
        self.p.kill()
        self.p.wait()

    def close(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        self.p.wait(timeout=60)


def vec(v):
    return "[" + ",".join("%.6f" % x for x in v) + "]"


def test_pk_limits(base):
    s = Session(base, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, name VARCHAR(16))")
    lo, hi = -(2 ** 63), 2 ** 63 - 1
    s.sql(f"INSERT INTO t VALUES ({lo},'min'),(0,'zero'),({hi},'max')")
    _, r = s.sql("SELECT id,name FROM t ORDER BY id")
    check("pk.range", r == [[str(lo), "min"], ["0", "zero"], [str(hi), "max"]], str(r))
    # values outside int64 must be rejected without inserting anything
    e = s.err(f"INSERT INTO t VALUES ({hi + 1},'overflow')")
    check("pk.overflow", e.startswith("ERROR "), e)
    _, r = s.sql("SELECT id FROM t ORDER BY id")
    check("pk.overflow-atomic", len(r) == 3, str(r))
    s.close()


def test_unicode(base):
    s = Session(base, fresh=True)
    s.sql("CREATE TABLE u (id BIGINT PRIMARY KEY, name VARCHAR(4))")
    s.sql("INSERT INTO u VALUES (1,'中文测试'),(2,'abcd'),(3,'az')")
    _, r = s.sql("SELECT id,name FROM u ORDER BY name")
    # code point order: 'az' < 'abcd' < '中文测试'  (b < z < 0x4e2d)
    check("utf8.order", [x[0] for x in r] == ["2", "3", "1"], str(r))
    e = s.err("INSERT INTO u VALUES (4,'五个字符啊')")
    check("utf8.toolong", e.startswith("ERROR 1406") or "too long" in e, e)
    s.sql("INSERT INTO u VALUES (5,'四个字符')")
    _, r = s.sql("SELECT name FROM u WHERE id = 5")
    check("utf8.exact-len", r == [["四个字符"]], str(r))
    s.sql("INSERT INTO u VALUES (6,'it''s')")
    _, r = s.sql("SELECT name FROM u WHERE id = 6")
    check("utf8.quote", r == [["it's"]], str(r))
    s.close()


def test_vector_validation(base):
    s = Session(base, fresh=True)
    s.sql("CREATE TABLE v (id BIGINT PRIMARY KEY, e VECTOR(3))")
    s.sql("INSERT INTO v VALUES (1,'[1,2,3]'),(2,'[4,5,6]')")
    for bad, tag in [("'[1,2]'", "dim-short"), ("'[1,2,3,4]'", "dim-long"),
                     ("'[1,2,nan]'", "nan"), ("'[1,2,inf]'", "inf"),
                     ("'[1,2,1e40]'", "overflow"), ("'[1,2,x]'", "not-number")]:
        e = s.err(f"INSERT INTO v VALUES (9,{bad})")
        check(f"vec.{tag}", e.startswith("ERROR "), e)
    _, r = s.sql("SELECT id FROM v ORDER BY id")
    check("vec.rejected-atomic", r == [["1"], ["2"]], str(r))
    # stored value: inserting more decimals than float32 keeps must round
    s.sql("INSERT INTO v VALUES (3,'[0.1,0.2,0.3]')")
    _, r = s.sql("SELECT L2_DISTANCE(e,'[0,0,0]') FROM v WHERE id = 3")
    want = math.sqrt(sum(float(__import__("struct").unpack("f", __import__("struct").pack("f", x))[0]) ** 2
                         for x in (0.1, 0.2, 0.3)))
    check("vec.float32-round", abs(float(r[0][0]) - want) < 1e-6, f"{r} vs {want}")
    s.close()


def test_insert_atomicity(base):
    s = Session(base, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, g VARCHAR(8))")
    s.sql("INSERT INTO t VALUES (1,'a')")
    e = s.err("INSERT INTO t VALUES (2,'b'),(1,'dup'),(3,'c')")
    check("insert.dup-in-batch", e.startswith("ERROR 1062"), e)
    _, r = s.sql("SELECT id FROM t ORDER BY id")
    check("insert.dup-atomic", r == [["1"]], str(r))
    e = s.err("INSERT INTO t VALUES (4,'x'),(5)")
    check("insert.arity", e.startswith("ERROR 1136"), e)
    _, r = s.sql("SELECT id FROM t ORDER BY id")
    check("insert.arity-atomic", r == [["1"]], str(r))
    s.sql("DELETE FROM t WHERE id = 1")
    s.sql("INSERT INTO t VALUES (1,'again')")
    _, r = s.sql("SELECT id,g FROM t ORDER BY id")
    check("insert.pk-reuse", r == [["1", "again"]], str(r))
    s.close()


def test_rollback_vector(base):
    s = Session(base, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, e VECTOR(2))")
    s.sql("INSERT INTO t VALUES (1,'[0,0]'),(2,'[10,10]')")
    s.sql("BEGIN")
    s.sql("UPDATE t SET e = '[1,1]' WHERE id = 1")
    _, r = s.sql("SELECT L2_DISTANCE(e,'[0,0]') FROM t WHERE id = 1")
    check("txn.read-own-write", abs(float(r[0][0]) - math.sqrt(2)) < 1e-6, str(r))
    s.sql("DELETE FROM t WHERE id = 2")
    s.sql("ROLLBACK")
    _, r = s.sql("SELECT id FROM t ORDER BY L2_DISTANCE(e,'[0,0]')")
    check("txn.rollback-vector", [x[0] for x in r] == ["1", "2"], str(r))
    _, r = s.sql("SELECT L2_DISTANCE(e,'[0,0]') FROM t WHERE id = 1")
    check("txn.rollback-value", float(r[0][0]) == 0.0, str(r))
    s.close()


def test_limit_and_empty(base):
    s = Session(base, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, e VECTOR(2), g VARCHAR(4))")
    _, r = s.sql("SELECT id FROM t ORDER BY L2_DISTANCE(e,'[0,0]') LIMIT 5")
    check("empty.topk", r == [], str(r))
    _, r = s.sql("SELECT id FROM t")
    check("empty.scan", r == [], str(r))
    s.sql("INSERT INTO t VALUES (1,'[0,0]','a'),(2,'[1,1]','b'),(3,'[2,2]','a')")
    _, r = s.sql("SELECT id FROM t ORDER BY L2_DISTANCE(e,'[0,0]') LIMIT 100")
    check("limit.over-rows", [x[0] for x in r] == ["1", "2", "3"], str(r))
    _, r = s.sql("SELECT id FROM t WHERE g = 'zzz' ORDER BY L2_DISTANCE(e,'[0,0]') LIMIT 3")
    check("filter.empty", r == [], str(r))
    _, r = s.sql("SELECT id FROM t ORDER BY L2_DISTANCE(e,'[0,0]') DESC LIMIT 2")
    check("topk.desc", [x[0] for x in r] == ["3", "2"], str(r))
    _, r = s.sql("SELECT id,L2_DISTANCE(e,'[0,0]') AS d FROM t ORDER BY d LIMIT 0")
    check("limit.zero", r == [], str(r))
    _, r = s.sql("SELECT id FROM t ORDER BY L2_DISTANCE(e,'[0,0]') LIMIT 9223372036854775807 OFFSET 1")
    check("limit.offset-overflow", r == [["2"], ["3"]], str(r))
    s.sql("CREATE TABLE wide (id BIGINT PRIMARY KEY, e VECTOR(4))")
    s.sql("INSERT INTO wide VALUES (1,'[3e30,0,0,0]'),(2,'[2e30,0,0,0]')")
    _, r = s.sql("SELECT id FROM wide ORDER BY L2_DISTANCE(e,'[0,0,0,0]') LIMIT 1")
    check("topk.finite-large-distances", r == [["2"]], str(r))
    s.sql("CREATE TABLE cutoff (id BIGINT PRIMARY KEY, label VARCHAR(8), e VECTOR(32))")
    def first_component(value):
        return "'[" + value + ",0" * 31 + "]'"
    samples = [(9, "keep", "1.0000001"), (8, "keep", "1"), (7, "keep", "-1"),
               (6, "keep", "1e-25"), (5, "keep", "2e-25"), (4, "keep", "3e30"),
               (3, "skip", "0"), (2, "keep", "1"), (1, "keep", "4")]
    values = ",".join(f"({ident},'{label}',{first_component(value)})"
                      for ident, label, value in samples)
    s.sql("INSERT INTO cutoff VALUES " + values)
    order = f"ORDER BY L2_DISTANCE(e,{first_component('0')})"
    _, r = s.sql(f"SELECT id FROM cutoff WHERE label = 'keep' {order} LIMIT 4")
    check("topk.cutoff-near-ties-and-underflow", r == [["6"], ["5"], ["2"], ["7"]], str(r))
    _, r = s.sql(f"SELECT id FROM cutoff {order} LIMIT 4")
    check("topk.cutoff-unfiltered", r == [["3"], ["6"], ["5"], ["2"]], str(r))
    s.close()


def test_index_maintenance(base):
    s = Session(base, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, e VECTOR(2), g VARCHAR(4))")
    s.sql("CREATE VECTOR INDEX idx ON t(e)")
    for i in range(1, 21):
        s.sql(f"INSERT INTO t VALUES ({i},'[{i},{i}]','g{i % 3}')")
    s.sql("SET vector_mode = 'ann'")
    _, r = s.sql("SELECT id FROM t ORDER BY L2_DISTANCE(e,'[1,1]') LIMIT 3")
    check("index.after-insert", [x[0] for x in r] == ["1", "2", "3"], str(r))
    s.sql("UPDATE t SET e = '[0,0]' WHERE id = 10")
    _, r = s.sql("SELECT id FROM t ORDER BY L2_DISTANCE(e,'[0,0]') LIMIT 2")
    check("index.after-update", [x[0] for x in r] == ["10", "1"], str(r))
    s.sql("DELETE FROM t WHERE id = 10")
    _, r = s.sql("SELECT id FROM t ORDER BY L2_DISTANCE(e,'[0,0]') LIMIT 2")
    check("index.after-delete", [x[0] for x in r] == ["1", "2"], str(r))
    _, r = s.sql("SELECT id FROM t WHERE g = 'g0' ORDER BY L2_DISTANCE(e,'[0,0]') LIMIT 3")
    check("index.filtered", [x[0] for x in r] == ["3", "6", "9"], str(r))
    s.close()


def test_recovery_volume(base):
    """5000 rows, a rolled back transaction and a hard kill."""
    s = Session(base, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, e VECTOR(8), g VARCHAR(4))")
    random.seed(7)
    rows = []
    for i in range(5000):
        v = [random.uniform(-1, 1) for _ in range(8)]
        rows.append(f"({i},{vec(v)},'g{i % 7}')")
    for i in range(0, 5000, 500):
        s.sql("INSERT INTO t VALUES " + ",".join(rows[i:i + 500]))
    s.sql("BEGIN")
    s.sql("DELETE FROM t WHERE g = 'g1'")
    s.sql("ROLLBACK")
    s.sql("BEGIN")
    s.sql("INSERT INTO t VALUES (9001,'[0,0,0,0,0,0,0,0]','gg')")
    s.kill()  # uncommitted: must not survive
    s = Session(base)
    _, r = s.sql("SELECT id FROM t WHERE g = 'g1' LIMIT 1")
    check("recover.rollback-kept", len(r) == 1, str(r))
    _, r = s.sql("SELECT id FROM t WHERE id = 9001")
    check("recover.uncommitted", r == [], str(r))
    _, r = s.sql("SELECT id FROM t ORDER BY L2_DISTANCE(e,'[0,0,0,0,0,0,0,0]') LIMIT 1")
    check("recover.vector-usable", len(r) == 1, str(r))
    s.close()


def test_mixed_dml(base):
    s = Session(base, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, n BIGINT, g VARCHAR(4), e VECTOR(2))")
    s.sql("INSERT INTO t VALUES (1,10,'a','[0,0]'),(2,20,'b','[1,1]'),(3,30,'a','[2,2]')")
    # expressions in SET are outside the required subset: must be a clean error
    e = s.err("UPDATE t SET n = n + 1 WHERE id = 1")
    check("dml.update-expr", e.startswith("ERROR "), e)
    _, r = s.sql("SELECT id,n FROM t ORDER BY id")
    check("dml.update-expr-atomic", r == [["1", "10"], ["2", "20"], ["3", "30"]], str(r))
    e = s.err("UPDATE t SET n = 5, id = 99 WHERE id = 1")
    check("dml.pk-update", e.startswith("ERROR ") and "primary key" in e, e)
    _, r = s.sql("SELECT id,n FROM t ORDER BY id")
    check("dml.pk-update-atomic", r == [["1", "10"], ["2", "20"], ["3", "30"]], str(r))
    s.sql("DELETE FROM t WHERE n >= 20")
    _, r = s.sql("SELECT id FROM t")
    check("dml.delete-range", r == [["1"]], str(r))
    # errors leave the connection usable
    _, r = s.sql("SELECT id FROM t")
    check("dml.usable", r == [["1"]], str(r))
    s.close()


def main():
    base = os.path.join(os.path.dirname(ROOT), "mvdb-edge")
    os.makedirs(base, exist_ok=True)
    tests = [
        ("pk_limits", test_pk_limits),
        ("unicode", test_unicode),
        ("vector_validation", test_vector_validation),
        ("insert_atomicity", test_insert_atomicity),
        ("rollback_vector", test_rollback_vector),
        ("limit_and_empty", test_limit_and_empty),
        ("index_maintenance", test_index_maintenance),
        ("recovery_volume", test_recovery_volume),
        ("mixed_dml", test_mixed_dml),
    ]
    for name, fn in tests:
        print(f"[{name}]")
        try:
            fn(os.path.join(base, name))
        except Exception as e:
            check(name + ".exception", False, repr(e))
    print(f"\npassed={PASS} failed={FAIL}")
    for f in FAILURES:
        print(" -", f)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
