#!/usr/bin/env python3
"""Additional AI column invariants against an isolated local seekdb server.

Run with --client /path/to/obclient --mock /path/to/ai_column_mock.py.
The supplied mock is local; no real model credentials are used.
"""
import argparse
import subprocess
import sys
import time
import unittest
import uuid


class AIColumnIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "OPTIONS" not in globals():
            raise unittest.SkipTest("requires an isolated seekdb instance and the contest mock")
        suffix = uuid.uuid4().hex[:12]
        cls.database = "ai_column_regression_" + suffix
        cls.model = "ai_column_model_" + suffix
        cls.endpoint = "ai_column_endpoint_" + suffix
        subprocess.run([sys.executable, OPTIONS.mock, "start", str(OPTIONS.mock_port)], check=True)
        cls.query("CREATE DATABASE " + cls.database)
        cls.query("CALL DBMS_AI_SERVICE.CREATE_AI_MODEL('" + cls.model +
                  "','{\"type\":\"completion\",\"model_name\":\"mock-complete\"}')")
        cls.query("CALL DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT('" + cls.endpoint +
                  "','{\"ai_model_name\":\"" + cls.model +
                  "\",\"url\":\"http://127.0.0.1:" + str(OPTIONS.mock_port) +
                  "/v1/chat/completions\",\"access_key\":\"local-mock-key\",\"provider\":\"openai\"}')")

    @classmethod
    def tearDownClass(cls):
        try:
            cls.query("DROP DATABASE " + cls.database)
            cls.query("CALL DBMS_AI_SERVICE.DROP_AI_MODEL_ENDPOINT('" + cls.endpoint + "')")
            cls.query("CALL DBMS_AI_SERVICE.DROP_AI_MODEL('" + cls.model + "')")
        finally:
            subprocess.run([sys.executable, OPTIONS.mock, "stop", str(OPTIONS.mock_port)], check=True)

    @classmethod
    def query(cls, sql, succeeds=True):
        command = [OPTIONS.client, "--no-defaults", "-h127.0.0.1", "-P" + str(OPTIONS.port),
                   "-uroot", "--database=test", "--batch", "--raw", "--skip-column-names",
                   "--default-character-set=utf8mb4"]
        result = subprocess.run(command, input=(sql + ";").encode("utf-8"),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if (result.returncode == 0) != succeeds:
            raise AssertionError(result.stderr.decode("utf-8", errors="replace") +
                                 result.stdout.decode("utf-8", errors="replace"))
        return result.stdout.decode("utf-8").splitlines()

    def wait_for(self, sql, expected):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.query(sql) == expected:
                return
            time.sleep(0.1)
        self.assertEqual(expected, self.query(sql))

    def test_null_result_readiness_and_business_write_protection(self):
        target = self.database + ".docs"
        self.query("CREATE TABLE " + target + "(id BIGINT PRIMARY KEY,title VARCHAR(100),"
                   "content TEXT,result VARCHAR(400) AI COLUMN(AI_COMPLETE('" + self.model +
                   "',content)) WITH(sync_mode='async',max_retries=0))")
        try:
            self.query("INSERT INTO " + target +
                       "(id,title,content) VALUES(1,'null input',NULL),(2,'text input','original')")
            self.wait_for("SELECT result FROM " + target + " WHERE id=2", ["mock:original"])
            self.wait_for("SELECT COUNT(*) FROM oceanbase.__all_ai_column_task task "
                          "JOIN oceanbase.__all_table t ON t.table_id=task.table_id "
                          "JOIN oceanbase.__all_database d ON d.database_id=t.database_id "
                          "WHERE d.database_name='" + self.database + "'", ["0"])
            self.assertEqual(["1\tNULL", "2\tmock:original"], self.query(
                "SET SESSION ai_column_strict_read=ON; SELECT id,result FROM " + target + " ORDER BY id"))
            definition = "\n".join(self.query("SHOW CREATE TABLE " + target))
            self.assertIn("AI COLUMN", definition)
            self.assertIn("max_retries=0", definition)
            for assignment in ["NULL", "DEFAULT", "'forged'", "result"]:
                self.query("UPDATE " + target + " SET result=" + assignment + " WHERE id=2", succeeds=False)
            self.query("INSERT INTO " + target +
                       "(id,title,content,result) VALUES(3,'title','body',DEFAULT)", succeeds=False)
            self.query("REPLACE INTO " + target +
                       "(id,title,content,result) VALUES(2,'title','body','forged')", succeeds=False)
            self.assertEqual(["2\toriginal\tmock:original"], self.query(
                "SELECT id,content,result FROM " + target + " WHERE id=2"))
            self.query("INSERT INTO " + target + " VALUES(3,'implicit columns','implicit input')")
            self.wait_for("SELECT result FROM " + target + " WHERE id=3", ["mock:implicit input"])
        finally:
            self.query("DROP TABLE " + target)

    def test_out_of_row_input_and_result(self):
        target = self.database + ".long_docs"
        self.query("CREATE TABLE " + target + "(id BIGINT PRIMARY KEY,content LONGTEXT,"
                   "result LONGTEXT AI COLUMN(AI_COMPLETE('" + self.model +
                   "',content)) WITH(sync_mode='async',max_retries=0))")
        try:
            self.query("INSERT INTO " + target + "(id,content) VALUES(1,REPEAT('abc',3000))")
            self.wait_for("SELECT LENGTH(result),SHA2(SUBSTRING(result,6),256)=SHA2(content,256) "
                          "FROM " + target + " WHERE id=1", ["9005\t1"])
        finally:
            self.query("DROP TABLE " + target)

    def test_config_cannot_call_a_model_inside_business_transaction(self):
        self.query("CREATE TABLE " + self.database + ".bad_config(id BIGINT PRIMARY KEY,"
                   "content TEXT,result LONGTEXT AI COLUMN(AI_COMPLETE('" + self.model +
                   "',content,AI_COMPLETE('" + self.model + "',content))))", succeeds=False)

    def test_partial_group_invalidation_releases_unaffected_column(self):
        target = self.database + ".split_docs"
        self.query("CREATE TABLE " + target + "(id BIGINT PRIMARY KEY,content TEXT,other TEXT,"
                   "result VARCHAR(400) AI COLUMN(AI_COMPLETE('" + self.model + "',content)),"
                   "label VARCHAR(400) AI COLUMN(AI_COMPLETE('" + self.model + "',other)))")
        try:
            self.query("INSERT INTO " + target +
                       "(id,content,other) VALUES(1,'slow-old','unchanged')")
            self.wait_for("SELECT COUNT(*) FROM oceanbase.__all_ai_column_task task "
                          "JOIN oceanbase.__all_table t ON t.table_id=task.table_id "
                          "JOIN oceanbase.__all_database d ON d.database_id=t.database_id "
                          "WHERE d.database_name='" + self.database +
                          "' AND t.table_name='split_docs' AND task.state=1", ["2"])
            self.query("UPDATE " + target + " SET content='fast-new' WHERE id=1")
            self.wait_for("SELECT result,label FROM " + target + " WHERE id=1",
                          ["mock:fast-new\tmock:unchanged"])
        finally:
            self.query("DROP TABLE " + target)

    def test_modified_definition_regenerates_existing_row(self):
        target = self.database + ".modify_docs"
        self.query("CREATE TABLE " + target + "(id BIGINT PRIMARY KEY,content TEXT,"
                   "result VARCHAR(400) AI COLUMN(AI_COMPLETE('" + self.model + "',content)))")
        try:
            self.query("INSERT INTO " + target + "(id,content) VALUES(1,'first')")
            self.wait_for("SELECT result FROM " + target, ["mock:first"])
            self.query("ALTER TABLE " + target + " MODIFY result VARCHAR(400) AI COLUMN("
                       "AI_COMPLETE('" + self.model + "',CONCAT('new:',content))) "
                       "WITH(sync_mode='async',max_retries=0)")
            self.wait_for("SELECT result FROM " + target, ["mock:new:first"])
            self.assertIn("new:", "\n".join(self.query("SHOW CREATE TABLE " + target)))
        finally:
            self.query("DROP TABLE " + target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", required=True)
    parser.add_argument("--mock", required=True)
    parser.add_argument("--port", type=int, default=2888)
    parser.add_argument("--mock-port", type=int, default=19125)
    OPTIONS, remaining = parser.parse_known_args()
    unittest.main(argv=[__file__] + remaining)
