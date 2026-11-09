#!/usr/bin/env python3
"""Randomised differential test against a Python reference model."""
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_minivectordb import Session, check, approx
import test_minivectordb as T

DIM = 4


def dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def vec_lit(v):
    return "[" + ",".join("%.4f" % x for x in v) + "]"


def run(seed, steps=200):
    rnd = random.Random(seed)
    base = os.path.join(os.path.dirname(T.ROOT), "mvdb-fuzz", str(seed))
    s = Session(base, fresh=True)
    s.sql(f"CREATE TABLE t (id BIGINT PRIMARY KEY, g VARCHAR(4), v VECTOR({DIM}))")
    model = {}
    next_id = 0
    for step in range(steps):
        op = rnd.random()
        if op < 0.35 or not model:
            rows = []
            for _ in range(rnd.randint(1, 3)):
                g = rnd.choice("abc")
                v = [round(rnd.uniform(-2, 2), 4) for _ in range(DIM)]
                rows.append((next_id, g, v))
                model[next_id] = (g, v)
                next_id += 1
            s.sql("INSERT INTO t VALUES " + ",".join(
                f"({i},'{g}','{vec_lit(v)}')" for i, g, v in rows))
        elif op < 0.45:
            i = rnd.choice(list(model))
            g = rnd.choice("abc")
            s.sql(f"UPDATE t SET g = '{g}' WHERE id = {i}")
            model[i] = (g, model[i][1])
        elif op < 0.55:
            i = rnd.choice(list(model))
            v = [round(rnd.uniform(-2, 2), 4) for _ in range(DIM)]
            s.sql(f"UPDATE t SET v = '{vec_lit(v)}' WHERE id = {i}")
            model[i] = (model[i][0], v)
        elif op < 0.65:
            i = rnd.choice(list(model))
            s.sql(f"DELETE FROM t WHERE id = {i}")
            del model[i]
        elif op < 0.72:
            g = rnd.choice("abc")
            s.sql(f"DELETE FROM t WHERE g = '{g}'")
            for k in [k for k, val in model.items() if val[0] == g]:
                del model[k]
        elif op < 0.85:
            _, rows = s.sql("SELECT id,g FROM t ORDER BY id")
            want = [[str(k), model[k][0]] for k in sorted(model)]
            check(f"fuzz{seed}.select", rows == want, f"{rows[:3]} vs {want[:3]}")
        else:
            q = [round(rnd.uniform(-2, 2), 4) for _ in range(DIM)]
            k = rnd.randint(1, 5)
            if rnd.random() < 0.5:
                g = rnd.choice("abc")
                _, rows = s.sql(f"SELECT id FROM t WHERE g = '{g}' ORDER BY "
                             f"L2_DISTANCE(v,'{vec_lit(q)}') LIMIT {k}")
                pool = {i: v for i, (gg, v) in model.items() if gg == g}
            else:
                _, rows = s.sql(f"SELECT id FROM t ORDER BY L2_DISTANCE(v,'{vec_lit(q)}') LIMIT {k}")
                pool = {i: v for i, (gg, v) in model.items()}
            want = [i for i, _ in sorted(pool.items(), key=lambda kv: (dist(kv[1], q), kv[0]))][:k]
            got = [int(r[0]) for r in rows]
            check(f"fuzz{seed}.topk", got == want, f"{got} vs {want}")

    # durability after the storm
    s.close()
    s = Session(base)
    _, rows = s.sql("SELECT id,g FROM t ORDER BY id")
    want = [[str(k), model[k][0]] for k in sorted(model)]
    check(f"fuzz{seed}.restart", rows == want, f"{rows} vs {want}")
    s.close()
    shutil.rmtree(base, ignore_errors=True)


def main():
    for seed in range(int(os.environ.get("SEEDS", "4"))):
        run(seed, steps=int(os.environ.get("STEPS", "150")))
    print(f"passed={T.PASS} failed={T.FAIL}")
    for f in T.FAILURES:
        print(" -", f)
    return 0 if T.FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
