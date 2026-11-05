/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <gtest/gtest.h>
#include <fstream>
#include <unistd.h>
#include "sql/engine/basic/file_scan_utils.h"
#include "sql/engine/basic/parquet_reader.h"

using namespace oceanbase::sql::file_sql;

class FileScan : public ::testing::Test {
protected:
  void SetUp() override {
    char path[] = "/tmp/file-scan-test-XXXXXX";
    char *result = ::mkdtemp(path); ASSERT_NE(nullptr, result); root_ = result;
  }
  void TearDown() override { for (const auto &p : files_) ::unlink(p.c_str()); ::rmdir(root_.c_str()); }
  std::string write(const std::string &name, const std::string &contents) {
    std::string path = root_ + "/" + name;
    std::ofstream file(path,std::ios::binary); file << contents; file.close(); files_.push_back(path); return path;
  }
  Definition infer(const std::string &name, const std::string &contents, Format format = Format::CSV) {
    Definition d; d.path = write(name,contents); d.format = format; return FileReader::infer(d,root_);
  }
  std::string root_;
  std::vector<std::string> files_;
};

TEST_F(FileScan, csv_multiline_nulls_duplicate_headers_and_rescan) {
  Definition d = infer("basic.csv", "id,text,id,empty\r\n1,\"hello,\nworld\",2,\r\n3,\"say \"\"hi\"\"\",4,\"\"\n");
  ASSERT_EQ(4,d.columns.size()); EXPECT_EQ("id__2",d.columns[2].name);
  EXPECT_EQ(Type::BIGINT,d.columns[0].type); EXPECT_TRUE(d.columns[3].nullable);
  FileReader reader(d); reader.open(root_); std::vector<Cell> row;
  ASSERT_TRUE(reader.next(row)); EXPECT_EQ("hello,\nworld",row[1].text); EXPECT_TRUE(row[3].is_null);
  ASSERT_TRUE(reader.next(row)); EXPECT_EQ("say \"hi\"",row[1].text); EXPECT_FALSE(row[3].is_null);
  EXPECT_FALSE(reader.next(row)); reader.rescan(); ASSERT_TRUE(reader.next(row)); EXPECT_EQ("1",row[0].text);
}

TEST_F(FileScan, json_union_missing_fields_and_logical_types) {
  Definition d = infer("rows.jsonl", "{\"id\":1,\"value\":true}\n{\"id\":2,\"other\":\"001\"}\n{\"id\":3,\"value\":null,\"other\":\"002\"}\n",Format::JSONL);
  ASSERT_EQ(3,d.columns.size()); EXPECT_EQ(Type::BOOLEAN,d.columns[1].type);
  EXPECT_EQ(Type::VARCHAR,d.columns[2].type); EXPECT_TRUE(d.columns[1].nullable); EXPECT_TRUE(d.columns[2].nullable);
  FileReader reader(d); reader.open(root_); std::vector<Cell> row;
  ASSERT_TRUE(reader.next(row)); ASSERT_EQ(3,row.size()); EXPECT_TRUE(row[2].is_null);
  ASSERT_TRUE(reader.next(row)); EXPECT_TRUE(row[1].is_null); EXPECT_EQ("001",row[2].text);
}

TEST_F(FileScan, csv_distinct_escape_and_embedded_newline) {
  Definition d;d.path=write("escape.csv","id;text\r\n1;\"say \\\"hi\\\"\\\nnext\"\r\n");
  d.options.delimiter=';';d.options.escape='\\';d=FileReader::infer(d,root_);
  ASSERT_EQ(2,d.columns.size());
  FileReader reader(d);reader.open(root_);std::vector<Cell> row;
  ASSERT_TRUE(reader.next(row));EXPECT_EQ("say \"hi\"\nnext",row[1].text);
  EXPECT_FALSE(reader.next(row));reader.rescan();ASSERT_TRUE(reader.next(row));
}

TEST_F(FileScan, reject_nested_and_malformed_records) {
  EXPECT_THROW(infer("nested.jsonl","{\"a\":[1]}\n",Format::JSONL),FileError);
  EXPECT_THROW(infer("quote.csv","a,b\n1,\"unterminated\n"),FileError);
  EXPECT_THROW(infer("width.csv","a,b\n1,2,3\n"),FileError);
  EXPECT_THROW(infer("bad.jsonl","{not-json}\n",Format::JSONL),FileError);
  EXPECT_THROW(infer("long-key.jsonl", "{\"" + std::string(257, 'a') + "\":1}\n", Format::JSONL),FileError);
  try {infer("location.jsonl","\n\n{invalid}\n",Format::JSONL);FAIL()<<"invalid JSON accepted";}
  catch(const FileError &error) {EXPECT_NE(std::string::npos,std::string(error.what()).find("line 3"));}
  try {infer("location.csv","a,b\n1,2,3\n");FAIL()<<"invalid row width accepted";}
  catch(const FileError &error) {EXPECT_NE(std::string::npos,std::string(error.what()).find("line 2"));}
}

TEST_F(FileScan, deterministic_boundaries_and_calendar) {
  EXPECT_EQ(Type::BIGINT,FileReader::infer_type("9223372036854775807"));
  EXPECT_EQ(Type::BIGINT,FileReader::infer_type("-9223372036854775808"));
  EXPECT_EQ(Type::VARCHAR,FileReader::infer_type("9223372036854775808"));
  EXPECT_EQ(Type::VARCHAR,FileReader::infer_type("1e999"));
  EXPECT_EQ(Type::VARCHAR,FileReader::infer_type("nan"));
  EXPECT_EQ(Type::VARCHAR,FileReader::infer_type("2025-02-29"));
  EXPECT_EQ(Type::DATE,FileReader::infer_type("2024-02-29"));
  EXPECT_EQ(Type::DATETIME,FileReader::infer_type("2026-08-03T12:13:14"));
  EXPECT_EQ(Type::DOUBLE,FileReader::widen(Type::DOUBLE,Type::BIGINT));
  int64_t value = 1; bool time = false;
  ASSERT_TRUE(FileReader::date_value("1970-01-01 00:00:00",value,time)); EXPECT_EQ(0,value);
}

TEST_F(FileScan, all_null_columns_keep_null_type) {
  Definition csv=infer("nulls.csv","id,value\n1,NULL\n2,\n");
  ASSERT_EQ(2,csv.columns.size());EXPECT_EQ(Type::NIL,csv.columns[1].type);
  EXPECT_TRUE(csv.columns[1].nullable);
  Definition json=infer("nulls.jsonl","{\"value\":null,\"id\":1}\n{\"id\":2}\n",Format::JSONL);
  EXPECT_EQ(Type::NIL,json.columns[0].type);EXPECT_TRUE(json.columns[0].nullable);
  EXPECT_EQ(Type::DOUBLE,FileReader::widen(Type::NIL,Type::DOUBLE));
}

TEST_F(FileScan, detect_replacement_and_content_changes) {
  Definition d = infer("change.csv","id\n1\n");
  FileReader reader(d); reader.open(root_);
  { std::ofstream file(d.path,std::ios::app); file << "2\n"; }
  std::vector<Cell> row; EXPECT_THROW(reader.next(row),FileError);
  EXPECT_THROW(reader.rescan(),FileError);
  EXPECT_THROW(FileReader(d).open(root_),FileError);
}

TEST_F(FileScan, reject_path_escape_and_symlink_escape) {
  Definition d; d.path = "/etc/passwd";
  EXPECT_THROW(FileReader::infer(d,root_),FileError);
  std::string link = root_ + "/escape"; files_.push_back(link);
  ASSERT_EQ(0,::symlink("/etc/passwd",link.c_str())); d.path=link;
  EXPECT_THROW(FileReader::infer(d,root_),FileError);
}

TEST_F(FileScan, schema_descriptor_round_trip) {
  Definition d = infer("schema.csv", "id,date,whole,fractional\n"
      "1,2026-08-01,2026-08-01 12:13:14,2026-08-01 12:13:14.123\n"
      "2,2026-08-02,2026-08-02 12:13:14,2026-08-02 12:13:14.123456\n");
  Definition copy = FileReader::decode(FileReader::encode(d));
  EXPECT_EQ(d.path,copy.path); EXPECT_TRUE(d.snapshot==copy.snapshot);
  ASSERT_EQ(4,copy.columns.size()); EXPECT_EQ(Type::DATE,copy.columns[1].type);
  EXPECT_EQ(Type::DATETIME, copy.columns[2].type);
  EXPECT_EQ(0, copy.columns[2].datetime_scale);
  EXPECT_EQ(Type::DATETIME, copy.columns[3].type);
  EXPECT_EQ(6, copy.columns[3].datetime_scale);
  EXPECT_THROW(FileReader::decode("{}"),FileError);
}

TEST_F(FileScan, glob_reports_only_permitted_files) {
  write("one.csv","id\n1\n"); write("two.jsonl","{}\n");
  Definition d; d.path=root_+"/*"; d.format=Format::GLOB;
  d=FileReader::infer(d,root_); ASSERT_EQ(1,d.columns.size()); EXPECT_EQ("file",d.columns[0].name);
  FileReader reader(d); reader.open(root_); std::vector<Cell> row; size_t count=0;
  while(reader.next(row)) { EXPECT_EQ(0,row[0].text.find(root_+"/")); ++count; }
  EXPECT_EQ(2,count);
}

TEST_F(FileScan, snappy_literals_overlapping_copies_and_corruption) {
  EXPECT_EQ("hello",ParquetReader::decode_snappy(std::string("\x05\x10" "hello",7),5));
  EXPECT_EQ(std::string(9,'a'),ParquetReader::decode_snappy(std::string("\x09\x00" "a\x1e\x01\x00",6),9));
  EXPECT_THROW(ParquetReader::decode_snappy(std::string("\x05\x01\x00",3),5),FileError);
  EXPECT_THROW(ParquetReader::decode_snappy(std::string("\x05\x10" "he",4),5),FileError);
}

TEST_F(FileScan, parquet_v2_snappy_fixture_and_projection) {
  std::ifstream input("tools/deploy/mysql_test/test_suite/file_sql/data/simple.parquet.b64");
  ASSERT_TRUE(input.good());
  const std::string alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string binary; uint32_t carry=0; int bits=0; char ch;
  while(input.get(ch)) {
    auto value=alphabet.find(ch); if(value==std::string::npos)continue;
    carry=(carry<<6)|static_cast<uint32_t>(value);bits+=6;
    if(bits>=8) {bits-=8;binary.push_back(static_cast<char>(carry>>bits));}
  }
  Definition d=infer("simple.parquet",binary,Format::PARQUET);
  ASSERT_EQ(1,d.columns.size());EXPECT_EQ(Type::BIGINT,d.columns[0].type);
  FileReader reader(d);reader.open(root_);std::vector<Cell> row;
  for(int i=0;i<7;++i) {ASSERT_TRUE(reader.next(row));EXPECT_EQ(std::to_string(i),row[0].text);}
  EXPECT_FALSE(reader.next(row)); reader.rescan();ASSERT_TRUE(reader.next(row));EXPECT_EQ("0",row[0].text);
  reader.close();reader.project({});reader.open(root_);int count=0;
  while(reader.next(row)) {EXPECT_TRUE(row[0].is_null);++count;}EXPECT_EQ(7,count);
  // Keep schema and chunk paths consistent while corrupting only the UTF-8
  // bytes of the column name in the footer.
  uint32_t footer_size=0;
  for(int i=0;i<4;++i)footer_size|=uint32_t(static_cast<unsigned char>(binary[binary.size()-8+i]))<<(8*i);
  ASSERT_LE(footer_size,binary.size()-8);
  size_t position=binary.size()-8-footer_size;
  while((position=binary.find("c1",position))!=std::string::npos&&position<binary.size()-8) {
    binary[position]=static_cast<char>(0xff);position+=2;
  }
  try {infer("bad-name.parquet",binary,Format::PARQUET);FAIL()<<"invalid column name accepted";}
  catch(const FileError &error) {EXPECT_NE(std::string::npos,std::string(error.what()).find("UTF-8"));}
}

TEST_F(FileScan, csv_export_round_trip_and_no_overwrite) {
  Definition d;d.path=root_+"/export.csv";files_.push_back(d.path);
  Column first;first.name="text";Column second;second.name="empty";d.columns={first,second};
  Cell text;text.text="line1\n\"line2\", end";text.is_null=false;Cell null;
  { CsvExporter writer;writer.open(d,root_);writer.write({text,null});writer.commit(); }
  d=FileReader::infer(d,root_);FileReader reader(d);reader.open(root_);std::vector<Cell> row;
  ASSERT_TRUE(reader.next(row));EXPECT_EQ(text.text,row[0].text);EXPECT_TRUE(row[1].is_null);
  EXPECT_FALSE(reader.next(row));EXPECT_THROW(CsvExporter().open(d,root_),FileError);
}

TEST_F(FileScan, csv_export_aborts_and_cleans_temporary_file) {
  Definition d;d.path=root_+"/aborted.csv";Column column;column.name="value";d.columns.push_back(column);
  { CsvExporter writer;writer.open(d,root_);Cell cell;cell.text="partial";cell.is_null=false;writer.write({cell}); }
  EXPECT_NE(0,::access(d.path.c_str(),F_OK));
  Definition glob;glob.path=root_+"/.*";glob.format=Format::GLOB;FileReader reader(glob);reader.open(root_);std::vector<Cell> row;
  EXPECT_FALSE(reader.next(row));
}

TEST_F(FileScan, parquet_multiple_row_groups_dictionary_and_plain) {
  for (const char *name : {"v1.0_False", "v1.0_True", "v2.0_False", "v2.0_True"}) {
    std::ifstream input(std::string("unittest/sql/engine/basic/data/") + name + ".parquet.b64");
    ASSERT_TRUE(input.good());
    const std::string alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string binary; uint32_t carry=0; int bits=0; char ch;
    while(input.get(ch)) {
      auto value=alphabet.find(ch); if(value==std::string::npos)continue;
      carry=(carry<<6)|static_cast<uint32_t>(value);bits+=6;
      if(bits>=8) {bits-=8;binary.push_back(static_cast<char>(carry>>bits));}
    }
    Definition d=infer(std::string(name)+".parquet",binary,Format::PARQUET);
    ASSERT_EQ(4,d.columns.size());EXPECT_EQ(Type::BOOLEAN,d.columns[3].type);
    FileReader reader(d);reader.project({0,1,3});reader.open(root_);std::vector<Cell> row;
    int count=0;
    const int64_t ids[]={1,0,3,-4,5,6,7,8,9};
    while(reader.next(row)) {
      ASSERT_LT(count,9);ASSERT_EQ(4,row.size());EXPECT_TRUE(row[2].is_null);
      if(count==1) {EXPECT_TRUE(row[0].is_null);EXPECT_TRUE(row[3].is_null);}
      else EXPECT_EQ(std::to_string(ids[count]),row[0].text);
      if(count==2)EXPECT_TRUE(row[1].is_null);
      ++count;
    }
    EXPECT_EQ(9,count);reader.rescan();ASSERT_TRUE(reader.next(row));EXPECT_EQ("1",row[0].text);
  }
}
