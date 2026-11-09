#!/usr/bin/env python3
"""Runs the official public cases from mysqltest/ against this build.

The organizers keep their driver private; this script re-implements only the
documented semantics (one statement per line, JSON assertions in .result) so the
SQL behaviour can be checked locally:

    python3 tests/run_public_cases.py --case sql_crud
    python3 tests/run_public_cases.py            # all cases, manifest order

CASE_DIR defaults to ../mysqltest.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXE = os.environ.get("MVDB_BIN", os.path.join(ROOT, "minivectordb.exe" if os.name == "nt" else "minivectordb"))
CASE_DIR = os.environ.get("CASE_DIR", os.path.join(os.path.dirname(ROOT), "mysqltest"))
REL_TOL = 1e-5
ABS_TOL = 1e-5


class Proc:
    def __init__(self, datadir):
        os.makedirs(datadir, exist_ok=True)
        self.p = subprocess.Popen([EXE, "--data-dir=" + datadir], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        banner = self.p.stdout.readline().strip()
        if banner != "READY":
            raise RuntimeError("bad banner: %r" % banner)

    def run(self, sql):
        self.p.stdin.write(sql + "\n")
        self.p.stdin.flush()
        rows, columns, status = [], None, None
        while True:
            line = self.p.stdout.readline()
            if line == "":
                raise RuntimeError("process closed; stderr=%s" % self.p.stderr.read()[-400:])
            line = line.rstrip("\n")
            if line.startswith("OK"):
                status = line
                break
            if line.startswith("ERROR"):
                return None, None, None, line
            if columns is None:
                columns = line.split("\t") if line else []
            else:
                rows.append(line.split("\t") if line else [])
        n = 0
        parts = status.split()
        if len(parts) > 1:
            n = int(parts[1])
        return columns, rows, n, None

    def close(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        self.p.wait(timeout=60)


def match_value(expect, got):
    if isinstance(expect, bool):
        return str(expect).lower() == got.lower()
    if isinstance(expect, int):
        try:
            return int(got) == expect
        except ValueError:
            return False
    if isinstance(expect, float):
        try:
            g = float(got)
        except ValueError:
            return False
        return abs(g - expect) <= ABS_TOL + REL_TOL * abs(expect)
    return str(expect) == got


def check(expect, columns, rows, affected, error):
    """Returns (ok, reason)."""
    if error:
        return False, "unexpected error: " + error
    if "affected_rows" in expect:
        if affected != expect["affected_rows"]:
            return False, "affected_rows %r != %r" % (affected, expect["affected_rows"])
    if "columns" in expect:
        if columns is None:
            return False, "no result set"
        if columns != expect["columns"]:
            return False, "columns %r != %r" % (columns, expect["columns"])
    if "rows" in expect:
        want, got = expect["rows"], rows or []
        if len(want) != len(got):
            return False, "row count %d != %d (%r)" % (len(got), len(want), got[:3])
        if expect.get("unordered"):
            pool = [list(g) for g in got]
            for w in want:
                for i, g in enumerate(pool):
                    if len(w) == len(g) and all(match_value(a, b) for a, b in zip(w, g)):
                        pool.pop(i)
                        break
                else:
                    return False, "missing row %r in %r" % (w, got)
        else:
            for i, (w, g) in enumerate(zip(want, got)):
                if len(w) != len(g):
                    return False, "row %d width %d != %d" % (i, len(g), len(w))
                for a, b in zip(w, g):
                    if not match_value(a, b):
                        return False, "row %d value %r != %r" % (i, b, a)
    return True, ""


def run_case(case_id, base):
    test_path = os.path.join(CASE_DIR, "cases", case_id + ".test")
    result_path = os.path.join(CASE_DIR, "cases", case_id + ".result")
    with open(test_path, encoding="utf-8") as f:
        sqls = [ln.rstrip("\n") for ln in f if ln.strip()]
    with open(result_path, encoding="utf-8") as f:
        expects = json.load(f)
    if len(sqls) != len(expects):
        return False, ["case file mismatch: %d statements, %d assertions" % (len(sqls), len(expects))]

    datadir = os.path.join(base, case_id)
    if os.path.isdir(datadir):
        shutil.rmtree(datadir)
    p = Proc(datadir)
    failures = []
    try:
        for i, (sql, expect) in enumerate(zip(sqls, expects)):
            columns, rows, affected, error = p.run(sql)
            ok, reason = check(expect, columns, rows, affected, error)
            if not ok:
                failures.append("  #%d %s\n      -> %s" % (i + 1, sql, reason))
    finally:
        p.close()
    return not failures, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    ap.add_argument("--dir", default=None, help="data directory root")
    args = ap.parse_args()

    if not os.path.isdir(CASE_DIR):
        print("case dir not found: " + CASE_DIR)
        return 2
    with open(os.path.join(CASE_DIR, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    ids = [c["id"] for c in manifest["cases"]]
    if args.case:
        ids = [args.case]
    base = args.dir or os.path.join(os.path.dirname(ROOT), "mvdb-public")
    total = 0
    bad = 0
    for cid in ids:
        ok, failures = run_case(cid, base)
        total += 1
        if not ok:
            bad += 1
            print("FAIL %s" % cid)
            print("\n".join(failures))
        else:
            print("PASS %s" % cid)
    print("\n%d/%d cases passed" % (total - bad, total))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
