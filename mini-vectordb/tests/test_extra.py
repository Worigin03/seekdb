#!/usr/bin/env python3
"""Durability / edge-case tests: disk mode, compaction, WAL tail damage, DDL."""
import os
import shutil
import struct
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_minivectordb import Session, check, expect_error, approx, PASS, FAIL, FAILURES
import test_minivectordb as T


def test_disk_mode(base):
    os.environ["MVDB_VEC_MEM_MB"] = "1"  # force the mmap path
    try:
        s = Session(base, fresh=True)
        s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, g VARCHAR(4), v VECTOR(8))")
        vals = []
        for i in range(300):
            vec = "[" + ",".join(str((i + k) % 17 * 0.5) for k in range(8)) + "]"
            vals.append(f"({i},'{'ab'[i%2]}','{vec}')")
        s.sql("INSERT INTO t VALUES " + ",".join(vals))
        h, r = s.sql("SELECT id FROM t WHERE g = 'b' ORDER BY L2_DISTANCE(v,'[0,0,0,0,0,0,0,0]') LIMIT 3")
        check("disk.rows", len(r) == 3, str(r))
        s.close()
        s = Session(base)
        h, r = s.sql("SELECT id FROM t ORDER BY id LIMIT 3")
        check("disk.restart", [x[0] for x in r] == ["0", "1", "2"], str(r))
        h, r = s.sql("SELECT id FROM t WHERE id >= 297 ORDER BY id")
        check("disk.tail", [x[0] for x in r] == ["297", "298", "299"], str(r))
        s.close()
    finally:
        os.environ.pop("MVDB_VEC_MEM_MB", None)


def test_compact(base):
    s = Session(base, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(4))")
    s.sql("INSERT INTO t VALUES (1,'[1,1,1,1]'),(2,'[2,2,2,2]')")
    s.sql("DELETE FROM t WHERE id = 1")
    s.sql("INSERT INTO t VALUES (3,'[3,3,3,3]')")
    size_before = os.path.getsize(os.path.join(base, "wal.log"))
    s.sql("COMPACT")
    size_after = os.path.getsize(os.path.join(base, "wal.log"))
    h, r = s.sql("SELECT id FROM t ORDER BY id")
    check("compact.data", r == [["2"], ["3"]], str(r))
    check("compact.shrink", size_after < size_before, f"{size_before} -> {size_after}")
    h, r = s.sql("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[3,3,3,3]') LIMIT 1")
    check("compact.vector", r[0][0] == "3", str(r))
    s.sql("INSERT INTO t VALUES (4,'[4,4,4,4]')")
    s.close()
    s = Session(base)
    h, r = s.sql("SELECT id FROM t ORDER BY id")
    check("compact.restart", r == [["2"], ["3"], ["4"]], str(r))
    s.close()


def test_wal_tail_damage(base):
    s = Session(base, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY)")
    s.sql("INSERT INTO t VALUES (1),(2)")
    s.close()
    # simulate a torn write: append garbage that looks like a record header
    with open(os.path.join(base, "wal.log"), "ab") as f:
        f.write(struct.pack("<II", 64, 0xDEADBEEF) + b"\x03garbage")
    s = Session(base)
    h, r = s.sql("SELECT id FROM t ORDER BY id")
    check("wal.garbage-ignored", r == [["1"], ["2"]], str(r))
    s.sql("INSERT INTO t VALUES (3)")
    h, r = s.sql("SELECT id FROM t ORDER BY id")
    check("wal.append-after-repair", r == [["1"], ["2"], ["3"]], str(r))
    s.close()
    s = Session(base)
    h, r = s.sql("SELECT id FROM t ORDER BY id")
    check("wal.restart-after-repair", r == [["1"], ["2"], ["3"]], str(r))
    s.close()


def test_ddl(base):
    s = Session(base, fresh=True)
    expect_error("ddl.drop-missing", lambda: s.sql("DROP TABLE nope"), "1051")
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, name VARCHAR(4), v VECTOR(2))")
    s.sql("CREATE TABLE IF NOT EXISTS t (id BIGINT PRIMARY KEY)")
    expect_error("ddl.duplicate", lambda: s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY)"), "1050")
    expect_error("ddl.no-pk", lambda: s.sql("CREATE TABLE bad (id BIGINT, v VECTOR(2))"), "1235")
    expect_error("ddl.two-vectors",
                 lambda: s.sql("CREATE TABLE bad2 (id BIGINT PRIMARY KEY, a VECTOR(2), b VECTOR(2))"), "1235")
    s.sql("INSERT INTO t VALUES (1,'a','[1,2]')")
    h, r = s.sql("SHOW TABLES")
    check("ddl.show", [x[0] for x in r] == ["t"], str(r))
    h, r = s.sql("DESC t")
    check("ddl.desc", r[0][0] == "id" and r[2][0] == "v", str(r))
    s.sql("DROP TABLE t")
    expect_error("ddl.after-drop", lambda: s.sql("SELECT * FROM t"), "1146")
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY)")
    s.sql("INSERT INTO t VALUES (7)")
    h, r = s.sql("SELECT id FROM t")
    check("ddl.recreate", r == [["7"]], str(r))
    s.close()


def test_txn_mixed(base):
    s = Session(base, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, g VARCHAR(4), v VECTOR(2))")
    s.sql("INSERT INTO t VALUES (1,'a','[0,0]'),(2,'a','[1,1]'),(3,'b','[2,2]')")
    s.sql("BEGIN")
    s.sql("UPDATE t SET g = 'c' WHERE id = 1")
    s.sql("DELETE FROM t WHERE id = 2")
    s.sql("INSERT INTO t VALUES (4,'d','[4,4]')")
    h, r = s.sql("SELECT id,g FROM t ORDER BY id")
    check("txn.mixed-visible", r == [["1", "c"], ["3", "b"], ["4", "d"]], str(r))
    s.sql("ROLLBACK")
    h, r = s.sql("SELECT id,g FROM t ORDER BY id")
    check("txn.mixed-rollback", r == [["1", "a"], ["2", "a"], ["3", "b"]], str(r))
    h, r = s.sql("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[1,1]') LIMIT 3")
    check("txn.mixed-rollback-vector", [x[0] for x in r] == ["2", "1", "3"], str(r))
    s.sql("BEGIN")
    s.sql("CREATE TABLE t2 (id BIGINT PRIMARY KEY)")
    s.sql("INSERT INTO t2 VALUES (1)")
    s.sql("COMMIT")
    s.close()
    s = Session(base)
    h, r = s.sql("SELECT id FROM t2")
    check("txn.multitable", r == [["1"]], str(r))
    h, r = s.sql("SELECT id,g FROM t ORDER BY id")
    check("txn.multitable-other", r == [["1", "a"], ["2", "a"], ["3", "b"]], str(r))
    s.close()


def main():
    base = os.path.join(os.path.dirname(T.ROOT), "mvdb-extra")
    if os.path.isdir(base):
        shutil.rmtree(base)
    os.makedirs(base)
    for name, fn in [
        ("disk_mode", test_disk_mode),
        ("compact", test_compact),
        ("wal_tail", test_wal_tail_damage),
        ("ddl", test_ddl),
        ("txn_mixed", test_txn_mixed),
    ]:
        print(f"[{name}]")
        try:
            fn(os.path.join(base, name))
        except Exception as e:
            check(name + ".exception", False, repr(e))
    print(f"\npassed={T.PASS} failed={T.FAIL}")
    for f in T.FAILURES:
        print(" -", f)
    return 0 if T.FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
