#!/usr/bin/env python3
"""Run file SQL integration checks against an isolated seekdb instance.

Usage: python3 test_file_sql_integration.py --client /path/to/obclient
       --port 2888 --root /path/allowed/by/secure_file_priv
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import uuid


def literal(value):
    return "'" + str(value).replace("\\", "\\\\").replace("'", "''") + "'"


class FileSQLIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "OPTIONS" not in globals():
            raise unittest.SkipTest("requires an isolated seekdb instance; run this file with --help")
        cls.temporary = tempfile.TemporaryDirectory(prefix="file-sql-regression-", dir=OPTIONS.root)
        cls.root = Path(cls.temporary.name)
        cls.csv = cls.root / "rows.csv"
        cls.csv.write_bytes(
            'id,name,kind,amount,nullable\r\n1,alpha,a,10,NULL\r\n'
            '2,"",b,20,"NULL"\r\n3,"中,文",a,30,\r\n'.encode("utf-8")
        )
        cls.json = cls.root / "rows.jsonl"
        cls.json.write_text("\n".join(json.dumps(row) for row in [
            {"id": 1, "label": "one"}, {"id": 3, "label": "three"},
            {"id": 4, "label": "four", "optional": True},
        ]) + "\n", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def query(self, sql, enabled=True, succeeds=True):
        command = [OPTIONS.client, "--no-defaults", "-h127.0.0.1", "-P" + str(OPTIONS.port),
                   "-u" + OPTIONS.user, "--database=test", "--batch", "--raw",
                   "--skip-column-names", "--default-character-set=utf8mb4"]
        source = "SET SESSION enable_file_sql=" + ("ON;" if enabled else "OFF;") + sql + ";"
        result = subprocess.run(command, input=source.encode("utf-8"), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=120)
        if succeeds:
            self.assertEqual(0, result.returncode, result.stderr.decode("utf-8", errors="replace"))
        else:
            self.assertNotEqual(0, result.returncode, result.stdout.decode("utf-8", errors="replace"))
        return result.stdout.decode("utf-8").splitlines()

    def test_disabled_and_regular_sql(self):
        self.query("SELECT * FROM read_csv(" + literal(self.csv) + ")", enabled=False, succeeds=False)
        self.assertEqual(["42"], self.query("SELECT 6*7", enabled=False))

    def test_projection_filter_group_and_having(self):
        source = "read_csv(" + literal(self.csv) + ")"
        self.assertEqual(["2\t40"], self.query(
            "SELECT id,amount*2 AS doubled FROM " + source +
            " WHERE amount>=20 ORDER BY amount DESC LIMIT 1 OFFSET 1"))
        self.assertEqual(["a\t2\t40", "b\t1\t20"], self.query(
            "SELECT kind,COUNT(*),SUM(amount) FROM " + source +
            " GROUP BY kind HAVING SUM(amount)>15 ORDER BY kind"))

    def test_cross_format_inner_and_left_join(self):
        left = "read_csv(" + literal(self.csv) + ") a"
        right = "read_ndjson(" + literal(self.json) + ") b"
        self.assertEqual(["1\tone", "3\tthree"], self.query(
            "SELECT a.id,b.label FROM " + left + " INNER JOIN " + right + " ON a.id=b.id ORDER BY a.id"))
        self.assertEqual(["1\tone", "2\tNULL", "3\tthree"], self.query(
            "SELECT a.id,b.label FROM " + left + " LEFT JOIN " + right + " ON a.id=b.id ORDER BY a.id"))
        self.query("SELECT id FROM " + left + " JOIN " + right + " ON a.id=b.id", succeeds=False)

    def test_describe_glob_and_schema_refresh(self):
        columns = self.query("DESCRIBE SELECT * FROM read_csv(" + literal(self.csv) + ")")
        self.assertEqual(5, len(columns))
        self.assertEqual("id\tBIGINT\tNO\tNULL\tNULL\tNULL", columns[0])
        files = self.query("SELECT SUBSTRING_INDEX(file,'/',-1) FROM glob(" +
                           literal(str(self.root / "*.jsonl")) + ") ORDER BY file")
        self.assertEqual(["rows.jsonl"], files)
        path = self.root / "refresh.csv"
        path.write_text("id\n1\n", encoding="utf-8")
        sql = "SELECT * FROM read_csv(" + literal(path) + ")"
        self.assertEqual(["1"], self.query(sql))
        path.write_text("id,label\n2,new\n", encoding="utf-8")
        self.assertEqual(["2\tnew"], self.query(sql))

    def test_null_and_csv_export_round_trip(self):
        path = self.root / "export.csv"
        source = "read_csv(" + literal(self.csv) + ")"
        statement = "COPY (SELECT * FROM " + source + " ORDER BY id) TO " + literal(path)
        statement += " (FORMAT csv, HEADER, DELIMITER ';', NULLSTR 'NULL')"
        self.query(statement)
        restored = "read_csv(" + literal(path) + ",delimiter=';')"
        self.assertEqual(["1\talpha\t<null>", "2\t\tNULL", "3\t中,文\t<null>"], self.query(
            "SELECT id,name,IF(nullable IS NULL,'<null>',nullable) FROM " + restored + " ORDER BY id"))
        original = path.read_bytes()
        self.query(statement, succeeds=False)
        self.assertEqual(original, path.read_bytes())

    def test_export_execution_error_cleans_temporary_file(self):
        source = self.root / "overflow.csv"
        source.write_text("id\n1\n-9223372036854775808\n", encoding="utf-8")
        target = self.root / "failed-export.csv"
        self.query("COPY (SELECT ABS(id) FROM read_csv(" + literal(source) + ")) TO " +
                   literal(target) + " (HEADER)", succeeds=False)
        self.assertFalse(target.exists())
        self.assertEqual([], list(self.root.glob(".file_sql_*")))

    def test_copy_table_and_scalar_types(self):
        table = "test.file_sql_copy_" + uuid.uuid4().hex[:12]
        path = self.root / "table-export.csv"
        self.query("CREATE TABLE " + table + "(id BIGINT PRIMARY KEY,text VARCHAR(100))")
        try:
            self.query("INSERT INTO " + table + " VALUES(1,'first'),(2,NULL)")
            self.query("COPY " + table + " TO " + literal(path) + " (HEADER)")
            self.assertEqual(["1\tfirst", "2\tNULL"], self.query(
                "SELECT id,text FROM read_csv(" + literal(path) + ") ORDER BY id"))
        finally:
            self.query("DROP TABLE " + table)
        boundaries = self.root / "types.csv"
        boundaries.write_text("id,flag,huge,empty,day\n"
                              "1,true,9223372036854775807,NULL,2024-02-29\n"
                              "2,false,9223372036854775808,,2025-02-29\n", encoding="utf-8")
        source = "read_csv(" + literal(boundaries) + ")"
        self.assertEqual(["2\t1\t0"], self.query(
            "SELECT COUNT(*),SUM(flag),COUNT(empty) FROM " + source))
        self.assertEqual(["NULL\tNULL"], self.query("SELECT SUM(empty),MIN(empty) FROM " + source))
        self.assertIn("empty\tNULL\tYES\tNULL\tNULL\tNULL", self.query("DESCRIBE SELECT * FROM " + source))
        self.assertEqual(["9223372036854775808\t2025-02-29"], self.query(
            "SELECT huge,day FROM " + source + " WHERE id=2"))

    def test_security_and_malformed_files(self):
        self.query("SELECT * FROM read_csv('/etc/passwd')", succeeds=False)
        link = self.root / "escape.csv"
        os.symlink("/etc/passwd", str(link))
        self.query("SELECT * FROM read_csv(" + literal(link) + ")", succeeds=False)
        invalid = self.root / "invalid.csv"
        invalid.write_bytes(b"a,b\n1,\"unterminated\n")
        self.query("SELECT * FROM read_csv(" + literal(invalid) + ")", succeeds=False)
        nested = self.root / "nested.jsonl"
        nested.write_text('{"a":{"nested":1}}\n', encoding="utf-8")
        self.query("SELECT * FROM read_json(" + literal(nested) + ")", succeeds=False)
        ambiguous = self.root / "ambiguous.jsonl"
        ambiguous.write_text('{"A":1,"a":2}\n', encoding="utf-8")
        self.query("SELECT * FROM read_json(" + literal(ambiguous) + ")", succeeds=False)
        self.assertEqual(["42"], self.query("SELECT 42"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", required=True)
    parser.add_argument("--port", type=int, default=2888)
    parser.add_argument("--root", required=True)
    parser.add_argument("--user", default="root")
    OPTIONS, remaining = parser.parse_known_args()
    unittest.main(argv=[__file__] + remaining)
