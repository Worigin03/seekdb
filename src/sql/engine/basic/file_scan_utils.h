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
#pragma once

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace oceanbase { namespace sql { namespace file_sql {

enum class Type { NIL, BOOLEAN, BIGINT, DOUBLE, VARCHAR, DATE, DATETIME };
enum class Format { CSV, JSONL, PARQUET, GLOB };

struct Cell {
  std::string text;
  bool is_null = true;
  Type type = Type::NIL;
};
struct Column {
  std::string name;
  Type type = Type::NIL;
  bool nullable = false;
  int datetime_scale = 0;
};
struct Options {
  bool header = true;
  char delimiter = ',';
  char quote = '"';
  char escape = '"';
  std::string nullstr = "NULL";
};
struct Snapshot {
  int64_t device = 0, inode = 0, size = 0, mtime = 0, ctime = 0;
  bool operator==(const Snapshot &other) const;
};
struct Definition {
  Format format = Format::CSV;
  std::string path;
  Options options;
  Snapshot snapshot;
  std::vector<Column> columns;
};
class ParquetReader;

class FileError : public std::runtime_error {
public:
  enum Kind { INVALID, UNSUPPORTED, SECURITY, CHANGED, IO };
  FileError(Kind kind, const std::string &message) : std::runtime_error(message), kind_(kind) {}
  Kind kind_;
};

// Owns one open file and at most one decoded record. Schema discovery and execution
// use separate passes; both pin and validate the same file identity.
class FileReader {
public:
  explicit FileReader(const Definition &definition);
  ~FileReader();
  FileReader(const FileReader &) = delete;
  FileReader &operator=(const FileReader &) = delete;
  void open(const std::string &secure_root);
  bool next(std::vector<Cell> &row);
  void rescan();
  void close();
  void verify() const;
  void project(const std::vector<int64_t> &columns);
  const Definition &definition() const { return definition_; }
  static Definition infer(Definition definition, const std::string &secure_root);
  static Type infer_type(const std::string &value);
  static Type widen(Type a, Type b);
  static bool date_value(const std::string &value, int64_t &microseconds, bool &has_time);
  static std::string encode(const Definition &definition);
  static Definition decode(const std::string &encoded);
private:
  bool csv_record(std::vector<Cell> &row);
  bool json_record(std::vector<Cell> &row, bool infer);
  bool read_line(std::string &line);
  void discover_glob(const std::string &secure_root);
  void initialize_header();
  Definition definition_;
  FILE *file_ = nullptr;
  int64_t line_ = 1;
  int64_t row_number_ = 0;
  std::vector<std::string> files_;
  size_t file_index_ = 0;
  bool inferring_ = false;
  ParquetReader *parquet_ = nullptr;
  bool has_projection_ = false;
  std::vector<int64_t> projection_;
};

class CsvExporter {
public:
  CsvExporter() = default;
  ~CsvExporter();
  CsvExporter(const CsvExporter &) = delete;
  CsvExporter &operator=(const CsvExporter &) = delete;
  void open(const Definition &definition, const std::string &secure_root);
  void write(const std::vector<Cell> &row);
  void commit();
private:
  void field(const Cell &cell);
  void bytes(const char *data, size_t size);
  FILE *file_ = nullptr;
  int directory_ = -1;
  bool committed_ = false;
  std::string temporary_, target_;
  Options options_;
  size_t columns_ = 0;
};

}}} // namespace oceanbase::sql::file_sql
