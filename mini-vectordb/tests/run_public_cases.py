#!/usr/bin/env python3
"""Run the official public case package against MiniVectorDB.

The package (mysqltest/) is a mysqltest-style set of ``.test`` SQL files plus
``.result`` JSON assertions.  Each non-empty line of a ``.test`` file is either
one complete SQL statement or a driver directive:

  ``--restart``    shut the server down gracefully and start it again on the
                   same data directory (uncommitted work must be rolled back)
  ``--mode ann``   switch subsequent vector searches to ann mode

Directives do not consume an assertion slot; assertions are matched by position
against the SQL statements only.

Usage:
  CASE_DIR=/path/to/mysqltest python3 tests/run_public_cases.py [case ...]
"""

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_EXE = os.path.join(ROOT, "minivectordb.exe" if os.name == "nt" else "minivectordb")
EXE = os.environ.get("MVDB_BIN") or DEFAULT_EXE
CASE_DIR = os.environ.get("CASE_DIR") or os.path.join(ROOT, "..", "mysqltest")
WORK = os.environ.get("MVDB_WORK") or os.path.join(ROOT, "..", "mvdb-public")


class Session:
    """One server process plus its stdout reader thread."""

    def __init__(self, data_dir, fresh=True):
        self.dir = data_dir
        if fresh and os.path.isdir(data_dir):
            shutil.rmtree(data_dir)
        os.makedirs(data_dir, exist_ok=True)
        self.p = None
        self.q = queue.Queue()
        self.err = ""
        self.start()

    def start(self):
        self.p = subprocess.Popen(
            [EXE, "--data-dir=" + self.dir],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, encoding="utf-8", errors="replace")
        self.q = queue.Queue()
        threading.Thread(target=self._pump, args=(self.p, self.q), daemon=True).start()
        # The server may announce readiness; a driver that starts writing at
        # once gets no announcement.  Either way the first statement works.
        try:
            first = self.q.get(timeout=float(os.environ.get("MVDB_READY_WAIT", "1.5")))
            if first.strip() != "READY":
                self.q.put(first)
        except queue.Empty:
            pass

    @staticmethod
    def _pump(proc, q):
        try:
            for line in proc.stdout:
                q.put(line.rstrip("\n"))
        except Exception:
            pass

    def sql(self, text):
        """Sends one statement; returns (columns, rows, affected, error)."""
        self.p.stdin.write(text.rstrip().rstrip(";") + "\n")
        self.p.stdin.flush()
        columns, rows = None, []
        while True:
            try:
                line = self.q.get(timeout=120)
            except queue.Empty:
                raise RuntimeError("timeout waiting for response to: " + text[:80])
            if line.startswith("ERROR"):
                return None, None, None, line
            if line == "OK" or line.startswith("OK "):
                parts = line.split()
                affected = int(parts[1]) if len(parts) > 1 else None
                return columns, rows, affected, None
            if columns is None:
                columns = line.split("\t") if line else []
            else:
                rows.append(line.split("\t") if line else [])

    def restart(self):
        """Graceful restart: ask to exit, wait, then start again."""
        try:
            self.p.stdin.write("exit\n")
            self.p.stdin.flush()
        except Exception:
            pass
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.p.terminate()
            try:
                self.p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait(timeout=10)
        self.err += self.p.stderr.read() or ""
        self.start()

    def close(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait(timeout=10)
        self.err += self.p.stderr.read() or ""


def load_cases():
    with open(os.path.join(CASE_DIR, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)["cases"]


def num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def same_value(expected, got):
    if isinstance(expected, bool):
        return False
    if isinstance(expected, (int, float)):
        g = num(got)
        if g is None:
            return False
        return abs(g - expected) <= 1e-5 + 1e-5 * abs(expected)
    return got == expected


def compare(expect, columns, rows, affected, error, tag):
    """Returns a list of failure strings (empty means the assertion passed)."""
    fails = []
    if "error" in expect:
        if error is None:
            fails.append("%s: expected error %r, got success" % (tag, expect["error"]))
        return fails
    if error is not None:
        return ["%s: unexpected %s" % (tag, error)]
    if "affected_rows" in expect:
        if affected != expect["affected_rows"]:
            fails.append("%s: affected_rows %r != %r" % (tag, affected, expect["affected_rows"]))
    if "columns" in expect and columns != expect["columns"]:
        fails.append("%s: columns %r != %r" % (tag, columns, expect["columns"]))
    if "rows" in expect:
        want, got = expect["rows"], rows
        if expect.get("unordered"):
            norm = lambda rs: sorted(json.dumps(r, ensure_ascii=False) for r in rs)
            ok = norm(got) == norm(want)
        else:
            ok = len(got) == len(want) and all(
                len(g) == len(w) and all(same_value(v, gv) for v, gv in zip(w, g))
                for w, g in zip(want, got))
        if not ok:
            fails.append("%s: rows %r != %r" % (tag, got, want))
    return fails


def run_case(case, work_root):
    data_dir = os.path.join(work_root, case["id"])
    test = os.path.join(CASE_DIR, case["test"])
    with open(os.path.join(CASE_DIR, case["result"]), encoding="utf-8") as f:
        results = json.load(f)

    lines = [ln.strip() for ln in open(test, encoding="utf-8") if ln.strip()]
    sqls = [s for s in lines if not s.startswith("--")]
    if len(sqls) != len(results):
        return ["%s: %d statements but %d assertions" % (case["id"], len(sqls), len(results))]

    fails, idx = [], 0
    s = Session(data_dir)
    try:
        for line in lines:
            if line == "--restart":
                s.restart()
                continue
            if line.startswith("--mode"):
                s.sql("SET vector_mode = '%s'" % line.split()[1])
                continue
            columns, rows, affected, error = s.sql(line)
            fails += compare(results[idx], columns, rows, affected, error,
                             "%s#%d(%s)" % (case["id"], idx, line[:60]))
            idx += 1
    except Exception as exc:  # noqa: BLE001 - report, do not abort the run
        fails.append("%s: %s" % (case["id"], exc))
    finally:
        try:
            s.close()
        except Exception:
            pass
    return fails


def main():
    only = set(sys.argv[1:])
    cases = [c for c in load_cases() if not only or c["id"] in only]
    os.makedirs(WORK, exist_ok=True)
    failed = 0
    for c in cases:
        fails = run_case(c, WORK)
        if fails:
            failed += 1
            print("FAIL %s" % c["id"])
            for f in fails[:6]:
                print("     " + f)
        else:
            print("PASS %s" % c["id"])
    print("\n%d/%d cases passed" % (len(cases) - failed, len(cases)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
