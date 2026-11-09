#!/usr/bin/env python3
"""Spec conformance checklist.

Each check maps to one bullet of the round-2 statement (section 2.x). The
script prints one line per requirement so the result can be pasted into the
submission notes.
"""
import math
import os
import random
import shutil
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXE = os.environ.get("MVDB_BIN", os.path.join(ROOT, "minivectordb.exe" if os.name == "nt" else "minivectordb"))

RESULTS = []


def check(req, name, ok, detail=""):
    RESULTS.append((ok, req, name, detail))
    print(("  PASS " if ok else "  FAIL ") + f"[{req}] {name}" + ("" if ok else f"  <- {detail}"))


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
        return self.raw(stmt)[2] or ""

    def kill(self):
        self.p.kill()
        self.p.wait()

    def close(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        self.p.wait(timeout=60)


def vec(v, nd=6):
    out = []
    for x in v:
        t = "%.*f" % (nd, x)
        if "." in t:
            t = t.rstrip("0").rstrip(".")
        out.append(t if t not in ("", "-") else "0")
    return "[" + ",".join(out) + "]"


def l2(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def main():
    base = os.path.join(os.path.dirname(ROOT), "mvdb-spec")
    os.makedirs(base, exist_ok=True)
    print("== 2.1 表结构与数据操作 ==")
    d = os.path.join(base, "schema")
    s = Session(d, fresh=True)
    s.sql("CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(32), content VARCHAR(1024), "
          "embedding VECTOR(4))")
    s.sql("INSERT INTO docs VALUES (1,'database','事务与恢复','[1,0,0,0]'),"
          "(2,'database','向量索引','[0,1,0,0]'),(3,'travel','旅行攻略','[0,0,1,0]')")
    _, r = s.sql("SELECT id, content FROM docs WHERE category = 'database' ORDER BY id")
    check("2.1", "BIGINT 主键 + VARCHAR + VECTOR(n) 建表/多行插入/条件投影",
          r == [["1", "事务与恢复"], ["2", "向量索引"]], str(r))
    s.sql("UPDATE docs SET category = 'db' WHERE id = 1")
    s.sql("DELETE FROM docs WHERE id = 3")
    _, r = s.sql("SELECT id, category FROM docs ORDER BY id")
    check("2.1", "按主键/条件 UPDATE 与 DELETE", r == [["1", "db"], ["2", "database"]], str(r))

    print("== 2.1 条件查询 ==")
    _, r = s.sql("SELECT id FROM docs WHERE id >= 1 AND category = 'database'")
    check("2.1", "比较运算符 + AND", r == [["2"]], str(r))
    _, r = s.sql("SELECT id FROM docs WHERE id = 2 OR id = 1 ORDER BY id DESC")
    check("2.1", "OR + ORDER BY DESC", [x[0] for x in r] == ["2", "1"], str(r))
    _, r = s.sql("SELECT * FROM docs ORDER BY id LIMIT 1")
    check("2.1", "SELECT * 与 LIMIT", len(r) == 1 and len(r[0]) == 4, str(r))
    s.close()

    print("== 2.2 向量检索 ==")
    d = os.path.join(base, "vector")
    s = Session(d, fresh=True)
    s.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, g VARCHAR(4), v VECTOR(4))")
    # 两条记录与查询点等距，用于验证并列时的主键规则
    s.sql("INSERT INTO t VALUES (10,'a','[0,0,0,0]'),(2,'a','[1,0,0,0]'),(7,'b','[0,1,0,0]'),"
          "(5,'b','[0,0,2,0]')")
    _, r = s.sql("SELECT id, L2_DISTANCE(v,'[0,0,0,0]') AS d FROM t ORDER BY d LIMIT 4")
    check("2.2", "L2_DISTANCE 返回欧氏距离",
          abs(float(r[0][1]) - 0.0) < 1e-9 and abs(float(r[1][1]) - 1.0) < 1e-9, str(r))
    check("2.2", "距离并列时按主键升序", [x[0] for x in r] == ["10", "2", "7", "5"], str(r))
    _, r = s.sql("SELECT id FROM t WHERE g = 'b' ORDER BY L2_DISTANCE(v,'[0,0,0,0]') LIMIT 1")
    check("2.2", "过滤后 TopK 在满足条件的集合内取", [x[0] for x in r] == ["7"], str(r))
    _, r = s.sql("SELECT id FROM t WHERE g = 'zzz' ORDER BY L2_DISTANCE(v,'[0,0,0,0]') LIMIT 3")
    check("2.2", "结果不足 K 条时返回全部（空集合）", r == [], str(r))
    _, r = s.sql("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[0,0,0,0]') LIMIT 99")
    check("2.2", "LIMIT 大于行数返回全部", len(r) == 4, str(r))
    # 普通 ORDER BY 并列按主键
    s.sql("INSERT INTO t VALUES (3,'c','[0,0,0,1]'),(1,'c','[0,0,1,0]')")
    _, r = s.sql("SELECT id FROM t WHERE g = 'c' ORDER BY g LIMIT 2")
    check("2.2", "普通 ORDER BY 并列按主键升序", [x[0] for x in r] == ["1", "3"], str(r))
    # 暴力真值对比
    random.seed(3)
    s.sql("CREATE TABLE big (id BIGINT PRIMARY KEY, g VARCHAR(4), v VECTOR(8))")
    data = {}
    rows = []
    for i in range(200):
        v = [random.uniform(-1, 1) for _ in range(8)]
        data[i] = v
        rows.append(f"({i},'g{i % 4}','{vec(v)}')")
    for i in range(0, 200, 50):
        s.sql("INSERT INTO big VALUES " + ",".join(rows[i:i + 50]))
    ok = True
    for _ in range(5):
        q = [random.uniform(-1, 1) for _ in range(8)]
        truth = sorted(data.items(), key=lambda kv: (l2(kv[1], q), kv[0]))[:10]
        _, r = s.sql(f"SELECT id FROM big ORDER BY L2_DISTANCE(v,'{vec(q)}') LIMIT 10")
        ok = ok and [int(x[0]) for x in r] == [k for k, _ in truth]
        g = random.randrange(4)
        truth = sorted([kv for kv in data.items() if kv[0] % 4 == g],
                       key=lambda kv: (l2(kv[1], q), kv[0]))[:10]
        _, r = s.sql(f"SELECT id FROM big WHERE g = 'g{g}' ORDER BY L2_DISTANCE(v,'{vec(q)}') LIMIT 10")
        ok = ok and [int(x[0]) for x in r] == [k for k, _ in truth]
    check("2.2", "精确 TopK 与暴力真值一致（有/无过滤各 5 组）", ok)

    print("== 2.2 索引维护 ==")
    s.sql("CREATE VECTOR INDEX idx ON t(v)")
    s.sql("INSERT INTO t VALUES (20,'d','[0,0,0,0]')")
    s.sql("UPDATE t SET v = '[9,9,9,9]' WHERE id = 10")
    s.sql("DELETE FROM t WHERE id = 5")
    _, r = s.sql("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[0,0,0,0]') LIMIT 3")
    check("2.2", "建索引后插入/更新/删除均反映到检索", [x[0] for x in r] == ["20", "1", "2"], str(r))
    s.sql("SET vector_mode = 'ann'")
    _, r = s.sql("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[0,0,0,0]') LIMIT 3")
    check("2.2", "ann 模式下结果与 exact 一致（Recall=1）", [x[0] for x in r] == ["20", "1", "2"], str(r))

    print("== 2.3 存储与事务 ==")
    s.sql("BEGIN")
    s.sql("INSERT INTO t VALUES (30,'e','[5,5,5,5]')")
    _, r = s.sql("SELECT id FROM t WHERE id = 30")
    check("2.3", "事务内读到自己的修改", r == [["30"]], str(r))
    s.sql("ROLLBACK")
    _, r = s.sql("SELECT id FROM t WHERE id = 30")
    check("2.3", "ROLLBACK 后记录不可见", r == [], str(r))
    s.sql("BEGIN")
    s.sql("INSERT INTO t VALUES (31,'e','[5,5,5,5]')")
    s.sql("COMMIT")
    s.close()
    s = Session(d)
    _, r = s.sql("SELECT id FROM t WHERE id = 31")
    check("2.3", "COMMIT 后重启仍在", r == [["31"]], str(r))
    e = s.err("INSERT INTO t VALUES (31,'x','[0,0,0,0]')")
    check("2.3", "主键重复被拒绝", e.startswith("ERROR 1062"), e)
    _, r = s.sql("SELECT id, g FROM t WHERE id = 31")
    check("2.3", "失败语句不留部分修改", r == [["31", "e"]], str(r))
    s.close()

    print("== 2.3 崩溃恢复 ==")
    s = Session(d)
    s.sql("BEGIN")
    s.sql("DELETE FROM t")
    s.sql("INSERT INTO t VALUES (99,'z','[0,0,0,0]')")
    s.kill()  # 未提交
    s = Session(d)
    _, r = s.sql("SELECT id FROM t WHERE id = 99")
    check("2.3", "未提交事务不生效", r == [], str(r))
    _, r = s.sql("SELECT id FROM t ORDER BY id LIMIT 1")
    check("2.3", "事务前的数据仍在", len(r) == 1, str(r))
    _, r = s.sql("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[0,0,0,0]') LIMIT 1")
    check("2.3", "崩溃后向量检索可用", len(r) == 1, str(r))

    print("== 2.1 输入校验 ==")
    s.sql("CREATE TABLE chk (id BIGINT PRIMARY KEY, s VARCHAR(2), v VECTOR(2))")
    cases = [("INSERT INTO chk VALUES (1,'abc','[0,0]')", "1406", "VARCHAR 按码点计长"),
             ("INSERT INTO chk VALUES (1,'a','[0,0,0]')", "1366", "向量维度不符"),
             ("INSERT INTO chk VALUES (1,'a','[nan,0]')", "1366", "NaN 被拒绝"),
             ("INSERT INTO chk VALUES (1,'a','[1e400,0]')", "1366", "float32 溢出被拒绝"),
             ("INSERT INTO chk VALUES (1,'a',[0,0])", "OK", "裸向量字面量可用")]
    for stmt, want, name in cases:
        e = s.err(stmt)
        ok = (want == "OK" and e == "") or e.startswith("ERROR " + want)
        check("2.1", name, ok, e)
        if want == "OK":
            s.sql("DELETE FROM chk")
    s.sql("INSERT INTO chk VALUES (1,'中文','[0,0]')")
    _, r = s.sql("SELECT s FROM chk WHERE id = 1")
    check("2.1", "VARCHAR 长度按 Unicode 字符数", r == [["中文"]], str(r))

    print("== 2.1 不支持的语句必须报错 ==")
    s.sql("CREATE TABLE p (id BIGINT PRIMARY KEY, n BIGINT)")
    s.sql("INSERT INTO p VALUES (1,1)")
    for stmt, name in [("ALTER TABLE p ADD COLUMN x BIGINT", "ALTER TABLE"),
                       ("SELECT COUNT(*) FROM p", "聚合"),
                       ("SELECT * FROM p, p AS q", "JOIN"),
                       ("SELECT * FROM p WHERE id IN (SELECT id FROM p)", "子查询"),
                       ("UPDATE p SET n = n + 1", "SET 表达式"),
                       ("SELECT id FROM p WHERE n = NULL", "NULL 比较")]:
        e = s.err(stmt)
        check("2.1", f"{name} 返回错误而不是静默执行", e.startswith("ERROR "), e)
    s.close()

    print("== 2.1 大小写与标识符 ==")
    d = os.path.join(base, "case")
    s = Session(d, fresh=True)
    s.sql("create table T (ID bigint primary key, Name varchar(8))")
    s.sql("insert into t values (1,'x')")
    _, r = s.sql("SELECT id, name FROM T")
    check("2.1", "表名/列名大小写不敏感", r == [["1", "x"]], str(r))
    s.close()

    total = len(RESULTS)
    failed = [r for r in RESULTS if not r[0]]
    print(f"\n共 {total} 项，通过 {total - len(failed)}，失败 {len(failed)}")
    for _, req, name, detail in failed:
        print(f"  - [{req}] {name}: {detail}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
