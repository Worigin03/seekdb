#!/usr/bin/env python3
"""Extra checks modelled on the formal acceptance table (section 3.2).

Covers the categories the public case package does not exercise directly:
crash recovery (SIGKILL, repeated), graceful shutdown rollback, invalid vector
input, VARCHAR character boundaries, filtered ANN recall, statement framing and
transaction visibility.

    python3 tests/test_formal.py
"""

import math
import os
import random
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_EXE = os.path.join(ROOT, "minivectordb.exe" if os.name == "nt" else "minivectordb")
EXE = os.environ.get("MVDB_BIN") or DEFAULT_EXE
WORK = os.environ.get("MVDB_WORK") or os.path.join(ROOT, "..", "mvdb-formal")

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
        print("FAIL %s %s" % (name, detail))


class Srv:
    def __init__(self, data_dir, fresh=False):
        self.dir = data_dir
        if fresh:
            import shutil
            shutil.rmtree(data_dir, ignore_errors=True)
        os.makedirs(data_dir, exist_ok=True)
        self.p = subprocess.Popen(
            [EXE, "--data-dir=" + self.dir], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1, encoding="utf-8", errors="replace")
        # Consume the readiness marker when the server announces one.
        line = self.p.stdout.readline()
        if line.strip() != "READY":
            self._pending = line
        else:
            self._pending = None

    def sql(self, stmt):
        """Returns (columns, rows, affected, error)."""
        if self._pending is not None:
            line, self._pending = self._pending, None
        else:
            self.p.stdin.write(stmt.rstrip().rstrip(";") + "\n")
            self.p.stdin.flush()
            line = self.p.stdout.readline()
        columns, rows = None, []
        while True:
            if line == "":
                raise RuntimeError("server closed: " + stmt[:60])
            if line.startswith("ERROR"):
                return None, None, None, line.rstrip("\n")
            if line == "OK" or line.startswith("OK "):
                parts = line.split()
                return columns, rows, (int(parts[1]) if len(parts) > 1 else None), None
            if columns is None:
                columns = line.rstrip("\n").split("\t")
            else:
                rows.append(line.rstrip("\n").split("\t"))
            line = self.p.stdout.readline()

    def err(self, stmt):
        _, _, _, e = self.sql(stmt)
        return e or ""

    def kill9(self):
        self.p.kill()
        self.p.wait()

    def sigterm(self):
        self.p.send_signal(signal.SIGTERM)
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()

    def close(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()


def vec(v):
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


def d(name):
    return os.path.join(WORK, name)


# ------------------------------------------------------------------ recovery ---

def test_durability(name, shutdown):
    dir_ = d(name)
    s = Srv(dir_, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2), tag VARCHAR(8))")
    s.sql("INSERT INTO t VALUES (1,'[1,0]','a'),(2,'[0,1]','b')")
    s.sql("BEGIN")
    s.sql("UPDATE t SET tag = 'uncommitted' WHERE id = 1")
    s.sql("INSERT INTO t VALUES (3,'[0,0]','c')")
    shutdown(s)                      # hard kill or graceful stop, both must roll back
    s = Srv(dir_)
    _, rows, _, e = s.sql("SELECT id, tag FROM t ORDER BY id")
    check(name + ".visible", e is None and rows == [["1", "a"], ["2", "b"]], str(rows) + str(e))
    _, rows, _, e = s.sql("SELECT id FROM t WHERE tag = 'uncommitted'")
    check(name + ".rolled-back", rows == [], str(rows))
    s.close()


def test_commit_survives_kill():
    dir_ = d("kill_commit")
    s = Srv(dir_, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, n BIGINT)")
    s.sql("INSERT INTO t VALUES (1, 10),(2, 20)")
    s.sql("BEGIN")
    s.sql("UPDATE t SET n = 99 WHERE id = 2")
    s.sql("COMMIT")                   # COMMIT returned OK: must be durable
    s.kill9()
    s = Srv(dir_)
    _, rows, _, e = s.sql("SELECT id, n FROM t ORDER BY id")
    check("kill.commit-survives", rows == [["1", "10"], ["2", "99"]], str(rows) + str(e))
    s.close()


def test_kill_during_recovery():
    """Kill the process while it is replaying the log, twice, then read."""
    dir_ = d("kill_recovery")
    s = Srv(dir_, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(4))")
    for i in range(4000):
        s.sql("INSERT INTO t VALUES (%d,'[%d,1,2,3]')" % (i, i % 97))
    s.close()
    for round_ in range(3):
        p = subprocess.Popen([EXE, "--data-dir=" + dir_], stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.02 * (round_ + 1))   # somewhere in the middle of recovery
        p.kill()
        p.wait()
    s = Srv(dir_)
    _, rows, _, e = s.sql("SELECT id FROM t ORDER BY id LIMIT 3")
    check("recovery.repeat", rows == [["0"], ["1"], ["2"]], str(rows) + str(e))
    _, rows, _, e = s.sql("SELECT id FROM t ORDER BY id DESC LIMIT 1")
    check("recovery.repeat-full", rows == [["3999"]], str(rows) + str(e))
    s.close()


# -------------------------------------------------------------------- vectors ---

def test_invalid_vectors():
    dir_ = d("badvec")
    s = Srv(dir_, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2))")
    s.sql("INSERT INTO t VALUES (1,'[1,2]')")
    for label, lit in [("nan", "[nan,0]"), ("inf", "[inf,0]"), ("ninf", "[-inf,0]"),
                       ("dim-short", "[1]"), ("dim-long", "[1,2,3]"), ("overflow", "[1e39,0]"),
                       ("junk", "[1,2x]"), ("empty", "[]")]:
        e = s.err("INSERT INTO t VALUES (9,'%s')" % lit)
        check("badvec." + label, e.startswith("ERROR"), "%r -> %r" % (lit, e))
    _, rows, _, e = s.sql("SELECT id FROM t ORDER BY id")
    check("badvec.no-partial", rows == [["1"]], str(rows) + str(e))
    s.close()


def test_varchar_boundary():
    dir_ = d("varchar")
    s = Srv(dir_, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(4))")
    e = s.err("INSERT INTO t VALUES (1,'中文字符')")      # 4 characters, 12 bytes
    check("varchar.four-chars", e == "", e)
    e = s.err("INSERT INTO t VALUES (2,'中文字符x')")     # 5 characters
    check("varchar.five-chars-rejected", e.startswith("ERROR"), e)
    _, rows, _, _ = s.sql("SELECT s FROM t")
    check("varchar.value", rows == [["中文字符"]], str(rows))
    s.close()


def test_ann_recall():
    """Filtered top-k must match brute force, also in ann mode."""
    dir_ = d("ann")
    n, dim, k, qn = 1500, 8, 10, 12
    rnd = random.Random(7)
    data = [[rnd.uniform(-1, 1) for _ in range(dim)] for _ in range(n)]
    groups = ["g%d" % (i % 3) for i in range(n)]
    s = Srv(dir_, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, g VARCHAR(2), v VECTOR(%d))" % dim)
    for start in range(0, n, 100):
        vals = ",".join("(%d,'%s','%s')" % (i, groups[i], vec(data[i]))
                        for i in range(start, min(start + 100, n)))
        s.sql("INSERT INTO t VALUES " + vals)
    s.sql("CREATE VECTOR INDEX idx ON t(v)")
    s.sql("SET vector_mode = 'ann'")
    hits = total = 0
    for _ in range(qn):
        q = [rnd.uniform(-1, 1) for _ in range(dim)]
        g = rnd.choice(["g0", "g1", "g2"])
        want = sorted((sum((a - b) ** 2 for a, b in zip(data[i], q)), i)
                      for i in range(n) if groups[i] == g)[:k]
        _, rows, _, e = s.sql("SELECT id FROM t WHERE g = '%s' ORDER BY L2_DISTANCE(v,'%s') LIMIT %d"
                              % (g, vec(q), k))
        got = [int(r[0]) for r in rows]
        check_ids = all(groups[i] == g for i in got) and len(set(got)) == len(got)
        check("ann.valid", check_ids and e is None, "%r %r" % (got, e))
        hits += len(set(got) & set(w[1] for w in want))
        total += min(k, len(want))
    check("ann.recall", hits == total, "%d/%d" % (hits, total))
    s.close()


# --------------------------------------------------------------- sql framing ---

def test_framing():
    dir_ = d("framing")
    s = Srv(dir_, fresh=True)
    s.p.stdin.write("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(8)); INSERT INTO t VALUES (1,'a')\n")
    s.p.stdin.flush()
    self_line = s.p.stdout.readline()
    # first statement's status, then the second's
    got = [self_line.strip(), s.p.stdout.readline().strip()]
    check("framing.multi-on-one-line", got == ["OK 0", "OK 1"], str(got))

    # a statement that ends on a word which is legal SQL but also a keyword
    _, rows, _, e = s.sql("SELECT id FROM t ORDER BY id")
    check("framing.trailing-keyword", rows == [["1"]], str(rows) + str(e))

    # a semicolon-free line still executes (the package sends one per line)
    s.p.stdin.write("INSERT INTO t VALUES (2,'b')\n")
    s.p.stdin.flush()
    line = s.p.stdout.readline().strip()
    check("framing.no-semicolon", line == "OK 1", line)

    _, rows, _, _ = s.sql("SELECT id FROM t ORDER BY id LIMIT 0")
    check("framing.limit-zero", rows == [], str(rows))
    _, rows, _, _ = s.sql("SELECT id FROM t ORDER BY id LIMIT 9")
    check("framing.limit-large", rows == [["1"], ["2"]], str(rows))
    s.close()


def test_transaction_visibility():
    dir_ = d("txnvis")
    s = Srv(dir_, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2))")
    s.sql("INSERT INTO t VALUES (1,'[0,0]')")
    s.sql("BEGIN")
    s.sql("INSERT INTO t VALUES (2,'[1,0]')")
    _, rows, _, _ = s.sql("SELECT id FROM t ORDER BY id")
    check("txn.sees-own-writes", rows == [["1"], ["2"]], str(rows))
    _, rows, _, _ = s.sql("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[1,0]') LIMIT 1")
    check("txn.vector-sees-own-writes", rows == [["2"]], str(rows))
    s.sql("ROLLBACK")
    _, rows, _, _ = s.sql("SELECT id FROM t ORDER BY id")
    check("txn.rollback", rows == [["1"]], str(rows))
    # statement failure inside a transaction keeps the transaction usable
    s.sql("BEGIN")
    s.sql("INSERT INTO t VALUES (3,'[0,0]')")
    e = s.err("INSERT INTO t VALUES (3,'[0,0]')")
    check("txn.failed-stmt", e.startswith("ERROR"), e)
    s.sql("COMMIT")
    _, rows, _, _ = s.sql("SELECT id FROM t ORDER BY id")
    check("txn.commit-after-failure", rows == [["1"], ["3"]], str(rows))
    s.close()


def main():
    os.makedirs(WORK, exist_ok=True)
    test_commit_survives_kill()
    test_durability("kill_txn", lambda s: s.kill9())
    if os.name != "nt":
        test_durability("sigterm_txn", lambda s: s.sigterm())
    test_kill_during_recovery()
    test_invalid_vectors()
    test_varchar_boundary()
    test_ann_recall()
    test_framing()
    test_transaction_visibility()
    print("\npassed=%d failed=%d" % (passed, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
