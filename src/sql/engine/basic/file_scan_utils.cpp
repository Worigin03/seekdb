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
#include "file_scan_utils.h"
#include "parquet_reader.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <set>
#include <atomic>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#if defined(__linux__)
#include <fcntl.h>
#include <glob.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace oceanbase { namespace sql { namespace file_sql {
namespace {
constexpr size_t MAX_FIELD = 16 * 1024 * 1024;
constexpr size_t MAX_RECORD = 64 * 1024 * 1024;
constexpr size_t MAX_COLUMNS = 4096;
constexpr size_t MAX_FILES = 100000;

void require(bool valid, FileError::Kind kind, const std::string &message)
{
  if (!valid) { throw FileError(kind, "file_sql: " + message); }
}

bool utf8(const std::string &s)
{
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = s[i++];
    if (c < 0x80) { if (c == 0) return false; continue; }
    int bytes = c >= 0xc2 && c <= 0xdf ? 1 : c >= 0xe0 && c <= 0xef ? 2 : c >= 0xf0 && c <= 0xf4 ? 3 : -1;
    if (bytes < 0 || s.size() - i < static_cast<size_t>(bytes)) return false;
    const unsigned char first = s[i];
    if ((c == 0xe0 && first < 0xa0) || (c == 0xed && first >= 0xa0)
        || (c == 0xf0 && first < 0x90) || (c == 0xf4 && first >= 0x90)) return false;
    for (int j = 0; j < bytes; ++j) { if ((static_cast<unsigned char>(s[i++]) & 0xc0) != 0x80) return false; }
  }
  return true;
}

#if defined(__linux__)
std::string canonical(const std::string &path)
{
  require(path.find('\0') == std::string::npos && !path.empty(), FileError::SECURITY, "invalid path");
  std::unique_ptr<char, decltype(&::free)> resolved(::realpath(path.c_str(), nullptr), &::free);
  require(resolved != nullptr, FileError::IO, "cannot resolve path: " + path);
  return std::string(resolved.get());
}

void check_root(const std::string &path, const std::string &root)
{
  // An empty secure_file_priv permits local paths, matching seekdb's existing contract.
  require(root != "NULL" && root != "null", FileError::SECURITY, "secure_file_priv disables file access");
  if (!root.empty()) {
    std::string allowed = canonical(root);
    require(allowed == "/" || path == allowed ||
            (path.size() > allowed.size() && path.compare(0, allowed.size(), allowed) == 0
             && path[allowed.size()] == '/'), FileError::SECURITY, "path outside secure_file_priv");
  }
}

Snapshot snapshot(const struct stat &s)
{
  Snapshot r;
  r.device = s.st_dev; r.inode = s.st_ino; r.size = s.st_size;
  r.mtime = s.st_mtim.tv_sec * INT64_C(1000000000) + s.st_mtim.tv_nsec;
  r.ctime = s.st_ctim.tv_sec * INT64_C(1000000000) + s.st_ctim.tv_nsec;
  return r;
}

// Open every component relative to an already-open directory; no symlinks are
// followed after canonicalization, including in parent components.
int open_regular(const std::string &path)
{
  int directory = ::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (directory < 0) throw FileError(FileError::IO, "file_sql: cannot open filesystem root");
  if (path == "/") return directory;
  try {
    size_t begin = 1;
    while (begin < path.size()) {
      size_t end = path.find('/', begin);
      const bool last = end == std::string::npos;
      std::string component = path.substr(begin, last ? std::string::npos : end - begin);
      int next = ::openat(directory, component.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | (last ? 0 : O_DIRECTORY));
      ::close(directory); directory = next;
      require(directory >= 0, FileError::IO, "cannot open file component: " + path);
      if (last) return directory;
      begin = end + 1;
    }
  } catch (...) { if (directory >= 0) ::close(directory); throw; }
  ::close(directory);
  throw FileError(FileError::IO, "file_sql: expected a regular file");
}
#endif

int digits(const std::string &s, size_t p, size_t n)
{
  int value = 0;
  if (p + n > s.size()) return -1;
  for (size_t i = 0; i < n; ++i) {
    if (s[p+i] < '0' || s[p+i] > '9') return -1;
    value = value * 10 + s[p+i] - '0';
  }
  return value;
}

std::string unique_name(const std::string &name, size_t index, const std::vector<Column> &columns)
{
  std::string base = name.empty() ? "column" + std::to_string(index + 1) : name;
  std::string candidate = base;
  size_t suffix = 1;
  auto lower = [](std::string s) { for (char &c : s) if (c >= 'A' && c <= 'Z') c += 'a' - 'A'; return s; };
  while (std::any_of(columns.begin(), columns.end(), [&](const Column &c) { return lower(c.name) == lower(candidate); })) {
    candidate = base + "__" + std::to_string(++suffix);
  }
  require(candidate.size() <= 256, FileError::INVALID, "column name exceeds 256 bytes");
  return candidate;
}
} // namespace

bool Snapshot::operator==(const Snapshot &o) const
{ return device == o.device && inode == o.inode && size == o.size && mtime == o.mtime && ctime == o.ctime; }

FileReader::FileReader(const Definition &definition) : definition_(definition) {}
FileReader::~FileReader() { close(); }
void FileReader::close() {
  if (parquet_ != nullptr) { delete parquet_; parquet_ = nullptr; }
  if (file_ != nullptr) { ::fclose(file_); file_ = nullptr; }
}

void FileReader::project(const std::vector<int64_t> &columns) {
  projection_ = columns; has_projection_ = true;
  if (parquet_ != nullptr) parquet_->project(columns);
}

void FileReader::open(const std::string &secure_root)
{
  close(); line_ = 1; row_number_ = 0; file_index_ = 0;
#if defined(__linux__)
  require(definition_.options.delimiter != '\0' && definition_.options.quote != '\0'
          && definition_.options.delimiter != definition_.options.quote
          && definition_.options.delimiter != '\r' && definition_.options.delimiter != '\n',
          FileError::INVALID, "invalid CSV delimiter or quote");
  if (definition_.format == Format::GLOB) { discover_glob(secure_root); return; }
  std::string path = canonical(definition_.path);
  check_root(path, secure_root);
  definition_.path = path;
  int fd = open_regular(path);
  struct stat st;
  if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
    ::close(fd); throw FileError(FileError::IO, "file_sql: expected a regular file: " + path);
  }
  const Snapshot current = snapshot(st);
  if (definition_.snapshot.inode != 0 && !(current == definition_.snapshot)) {
    ::close(fd); throw FileError(FileError::CHANGED, "file_sql: file changed since schema inference: " + path);
  }
  definition_.snapshot = current;
  file_ = ::fdopen(fd, "rb");
  if (file_ == nullptr) { ::close(fd); throw FileError(FileError::IO, "file_sql: fdopen failed"); }
  ::setvbuf(file_, nullptr, _IOFBF, 64 * 1024);
  if (definition_.format == Format::PARQUET) {
    parquet_ = new ParquetReader(file_,definition_.snapshot.size);
    definition_.columns = parquet_->columns();
    for (const Column &column : definition_.columns)
      require(utf8(column.name), FileError::INVALID, "invalid UTF-8 in Parquet column name");
    if (has_projection_) parquet_->project(projection_);
  }
  if (definition_.format == Format::CSV && definition_.options.header) initialize_header();
#else
  throw FileError(FileError::UNSUPPORTED, "file_sql: local scans require Linux");
#endif
}

void FileReader::verify() const
{
#if defined(__linux__)
  if (file_ != nullptr) {
    struct stat opened, named;
    require(::fstat(::fileno(file_), &opened) == 0 && ::lstat(definition_.path.c_str(), &named) == 0
            && snapshot(opened) == definition_.snapshot && snapshot(named) == definition_.snapshot,
            FileError::CHANGED, "file changed while scanning: " + definition_.path);
  }
#endif
}

void FileReader::rescan()
{
  verify();
  if (definition_.format == Format::GLOB) { file_index_ = 0; return; }
  if (parquet_ != nullptr) { parquet_->rescan(); row_number_ = 0; return; }
  require(file_ != nullptr && ::fseek(file_, 0, SEEK_SET) == 0, FileError::IO, "cannot rewind file");
  ::clearerr(file_); line_ = 1; row_number_ = 0;
  if (definition_.format == Format::CSV && definition_.options.header) initialize_header();
}

void FileReader::initialize_header()
{
  std::vector<Cell> header;
  require(csv_record(header), FileError::INVALID, "CSV header is missing");
  if (definition_.columns.empty()) {
    for (size_t i = 0; i < header.size(); ++i) {
      if (i == 0 && header[i].text.compare(0, 3, "\xef\xbb\xbf") == 0) header[i].text.erase(0, 3);
      Column c; c.name = unique_name(header[i].text, i, definition_.columns);
      definition_.columns.push_back(c);
    }
  } else {
    require(header.size() == definition_.columns.size(), FileError::CHANGED, "CSV header width changed");
  }
}

bool FileReader::csv_record(std::vector<Cell> &row)
{
  row.clear();
  bool any = false, quoted = false, inside = false, after_quote = false;
  std::string value;
  size_t record_bytes = 0;
  auto finish = [&]() {
    require(utf8(value), FileError::INVALID, "invalid UTF-8 at line " + std::to_string(line_));
    Cell cell; cell.text.swap(value);
    cell.is_null = !quoted && (cell.text.empty() || cell.text == definition_.options.nullstr || cell.text == "\\N");
    row.push_back(std::move(cell)); quoted = false; after_quote = false;
    require(row.size() <= MAX_COLUMNS, FileError::INVALID, "too many columns");
  };
  for (;;) {
    int ch = ::fgetc(file_);
    if (ch == EOF) {
      require(!::ferror(file_), FileError::IO, "error reading CSV");
      require(!inside, FileError::INVALID, "unterminated CSV quote at line " + std::to_string(line_));
      if (!any) return false;
      finish(); return true;
    }
    any = true;
    require(++record_bytes <= MAX_RECORD, FileError::INVALID, "record exceeds 64 MiB");
    if (inside) {
      if (ch == definition_.options.escape && definition_.options.escape != definition_.options.quote) {
        int escaped = ::fgetc(file_);
        require(escaped != EOF, FileError::INVALID, "unterminated CSV escape");
        require(++record_bytes <= MAX_RECORD, FileError::INVALID, "record exceeds 64 MiB");
        if (escaped == '\n') ++line_;
        value.push_back(static_cast<char>(escaped));
      } else if (ch == definition_.options.quote) {
        int following = ::fgetc(file_);
        if (following == definition_.options.quote && definition_.options.escape == definition_.options.quote) {
          require(++record_bytes <= MAX_RECORD, FileError::INVALID, "record exceeds 64 MiB");
          value.push_back(static_cast<char>(ch));
        } else {
          inside = false; after_quote = true;
          if (following != EOF) ::ungetc(following, file_);
        }
      } else {
        if (ch == '\n') ++line_;
        value.push_back(static_cast<char>(ch));
      }
    } else if (ch == definition_.options.delimiter) {
      finish();
    } else if (ch == '\n' || ch == '\r') {
      if (ch == '\r') { int next = ::fgetc(file_); if (next != '\n' && next != EOF) ::ungetc(next, file_); }
      finish(); ++line_; return true;
    } else if (ch == definition_.options.quote && value.empty() && !after_quote) {
      quoted = true; inside = true;
    } else {
      require(!after_quote && ch != definition_.options.quote, FileError::INVALID,
              "unexpected character after CSV quote at line " + std::to_string(line_));
      value.push_back(static_cast<char>(ch));
    }
    require(value.size() <= MAX_FIELD, FileError::INVALID, "field exceeds 16 MiB");
  }
}

bool FileReader::read_line(std::string &line)
{
  line.clear();
  for (;;) {
    int c = ::fgetc(file_);
    if (c == EOF) { require(!::ferror(file_), FileError::IO, "error reading JSONL"); return !line.empty(); }
    if (c == '\n') { ++line_; if (!line.empty() && line.back() == '\r') line.pop_back(); return true; }
    line.push_back(static_cast<char>(c));
    require(line.size() <= MAX_RECORD, FileError::INVALID, "JSONL record exceeds 64 MiB");
  }
}

bool FileReader::json_record(std::vector<Cell> &row, bool infer)
{
  std::string line;
  int64_t record_line = line_;
  while (read_line(line)) {
    if (row_number_ == 0 && line.compare(0, 3, "\xef\xbb\xbf") == 0) line.erase(0, 3);
    if (line.find_first_not_of(" \t\r") == std::string::npos) { record_line = line_; continue; }
    require(utf8(line), FileError::INVALID, "invalid UTF-8 in JSONL at line " + std::to_string(record_line));
    rapidjson::Document doc;
    doc.Parse(line.c_str(), line.size());
    require(!doc.HasParseError() && doc.IsObject(), FileError::INVALID,
            "expected flat JSON object near line " + std::to_string(record_line));
    row.assign(definition_.columns.size(), Cell());
    std::set<std::string> seen;
    for (auto it = doc.MemberBegin(); it != doc.MemberEnd(); ++it) {
      std::string name(it->name.GetString(), it->name.GetStringLength());
      require(!name.empty() && utf8(name) && seen.insert(name).second, FileError::INVALID,
              "invalid or duplicate JSON field name at line " + std::to_string(record_line));
      require(name.size() <= 256, FileError::INVALID, "JSON field name exceeds 256 bytes");
      require(!it->value.IsObject() && !it->value.IsArray(), FileError::INVALID,
              "nested JSON values are not supported at line " + std::to_string(record_line));
      size_t index = 0;
      while (index < definition_.columns.size() && definition_.columns[index].name != name) ++index;
      if (index == definition_.columns.size()) {
        require(infer, FileError::CHANGED, "new JSON field after schema inference");
        require(index < MAX_COLUMNS, FileError::INVALID, "too many JSON fields");
        Column c; c.name = name; c.nullable = row_number_ > 0;
        definition_.columns.push_back(c); row.push_back(Cell());
      }
      Cell &cell = row[index];
      cell.is_null = it->value.IsNull();
      if (it->value.IsString()) {
        cell.text.assign(it->value.GetString(), it->value.GetStringLength());
        require(utf8(cell.text), FileError::INVALID, "invalid UTF-8 in JSON string");
        int64_t timestamp = 0; bool has_time = false;
        cell.type = date_value(cell.text, timestamp, has_time) ? (has_time ? Type::DATETIME : Type::DATE) : Type::VARCHAR;
      } else if (!cell.is_null) {
        rapidjson::StringBuffer buffer; rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        it->value.Accept(writer); cell.text.assign(buffer.GetString(), buffer.GetSize());
        cell.type = it->value.IsBool() ? Type::BOOLEAN : it->value.IsInt64() ? Type::BIGINT
                    : it->value.IsUint64() ? Type::VARCHAR : Type::DOUBLE;
      }
      require(cell.text.size() <= MAX_FIELD, FileError::INVALID, "JSON field exceeds 16 MiB");
    }
    return true;
  }
  return false;
}

bool FileReader::next(std::vector<Cell> &row)
{
  if (definition_.format == Format::GLOB) {
    row.clear(); if (file_index_ >= files_.size()) return false;
    Cell cell; cell.text = files_[file_index_++]; cell.is_null = false; row.push_back(std::move(cell)); return true;
  }
  require(file_ != nullptr, FileError::IO, "reader is not open");
  verify();
  const int64_t record_line = line_;
  bool found = parquet_ != nullptr ? parquet_->next(row)
               : definition_.format == Format::CSV ? csv_record(row) : json_record(row, inferring_);
  if (found) {
    if (definition_.columns.empty() && inferring_) {
      for (size_t i = 0; i < row.size(); ++i) { Column c; c.name = "column" + std::to_string(i+1); definition_.columns.push_back(c); }
    }
    require(row.size() == definition_.columns.size(), FileError::INVALID,
            "wrong number of fields near line " + std::to_string(record_line));
    for (const Cell &cell : row) require(cell.is_null || utf8(cell.text), FileError::INVALID, "invalid UTF-8 value");
    ++row_number_;
    verify();
  } else { verify(); }
  return found;
}

Definition FileReader::infer(Definition definition, const std::string &secure_root)
{
  FileReader reader(definition); reader.inferring_ = true; reader.open(secure_root);
  if (definition.format == Format::PARQUET) { reader.verify(); return reader.definition_; }
  std::vector<Cell> row;
  while (reader.next(row)) {
    for (size_t i = 0; i < row.size(); ++i) {
      Column &column = reader.definition_.columns[i];
      column.nullable |= row[i].is_null;
      const Type type = reader.definition_.format == Format::JSONL ? row[i].type : infer_type(row[i].text);
      column.type = widen(column.type, row[i].is_null ? Type::NIL : type);
      if (type == Type::DATETIME && !row[i].is_null && row[i].text.size() > 20) {
        column.datetime_scale = std::max(column.datetime_scale, static_cast<int>(row[i].text.size() - 20));
      }
    }
  }
  require(!reader.definition_.columns.empty(), FileError::INVALID, "cannot infer columns from an empty file");
  for (Column &column : reader.definition_.columns) if (column.type == Type::NIL) column.nullable = true;
  return reader.definition_;
}

Type FileReader::widen(Type a, Type b)
{
  if (a == Type::NIL) return b;
  if (b == Type::NIL || a == b) return a;
  if ((a == Type::BIGINT && b == Type::DOUBLE) || (a == Type::DOUBLE && b == Type::BIGINT)) return Type::DOUBLE;
  if ((a == Type::DATE && b == Type::DATETIME) || (a == Type::DATETIME && b == Type::DATE)) return Type::DATETIME;
  return Type::VARCHAR;
}

bool FileReader::date_value(const std::string &s, int64_t &micros, bool &has_time)
{
  if (s.size() != 10 && s.size() != 19 && (s.size() < 21 || s.size() > 26)) return false;
  int year = digits(s,0,4), month = digits(s,5,2), day = digits(s,8,2);
  if (s[4] != '-' || s[7] != '-' || year < 1 || month < 1 || month > 12 || day < 1) return false;
  const int lengths[] = {31,28,31,30,31,30,31,31,30,31,30,31};
  bool leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
  if (day > lengths[month-1] + (month == 2 && leap)) return false;
  int hour = 0, minute = 0, second = 0, fraction = 0;
  has_time = s.size() > 10;
  if (has_time) {
    if ((s[10] != ' ' && s[10] != 'T') || s[13] != ':' || s[16] != ':') return false;
    hour = digits(s,11,2); minute = digits(s,14,2); second = digits(s,17,2);
    if (hour < 0 || hour > 23 || minute < 0 || minute > 59 || second < 0 || second > 59) return false;
    if (s.size() > 19) {
      if (s[19] != '.') return false;
      fraction = digits(s,20,s.size()-20); if (fraction < 0) return false;
      for (size_t i = s.size()-20; i < 6; ++i) fraction *= 10;
    }
  }
  // Gregorian civil date to days since the Unix epoch.
  year -= month <= 2;
  const int era = year / 400, yoe = year - era * 400;
  const int doy = (153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;
  const int64_t days = era * INT64_C(146097) + yoe * 365 + yoe/4 - yoe/100 + doy - 719468;
  micros = ((days * 24 + hour) * 3600 + minute * 60 + second) * INT64_C(1000000) + fraction;
  return true;
}

Type FileReader::infer_type(const std::string &s)
{
  if (s == "true" || s == "false" || s == "TRUE" || s == "FALSE") return Type::BOOLEAN;
  int64_t date = 0; bool time = false;
  if (date_value(s,date,time)) return time ? Type::DATETIME : Type::DATE;
  if (s.empty()) return Type::VARCHAR;
  size_t pos = (s[0] == '+' || s[0] == '-') ? 1 : 0;
  bool integer = pos < s.size();
  for (size_t i = pos; integer && i < s.size(); ++i) integer = s[i] >= '0' && s[i] <= '9';
  if (integer) {
    errno = 0; char *end = nullptr; ::strtoll(s.c_str(), &end, 10);
    return errno == 0 && end == s.c_str() + s.size() ? Type::BIGINT : Type::VARCHAR;
  }
  // strtod accepts whitespace, hexadecimal and infinities; exclude them explicitly.
  for (char c : s) if (!(c >= '0' && c <= '9') && c != '+' && c != '-' && c != '.' && c != 'e' && c != 'E') return Type::VARCHAR;
  errno = 0; char *end = nullptr; double value = ::strtod(s.c_str(), &end);
  return errno == 0 && end != s.c_str() && end == s.c_str() + s.size() && std::isfinite(value) ? Type::DOUBLE : Type::VARCHAR;
}

void FileReader::discover_glob(const std::string &secure_root)
{
#if defined(__linux__)
  files_.clear();
  require(definition_.path.find('\0') == std::string::npos, FileError::SECURITY, "invalid glob pattern");
  glob_t matches{};
  const int code = ::glob(definition_.path.c_str(), GLOB_NOSORT, nullptr, &matches);
  if (code != 0 && code != GLOB_NOMATCH) { ::globfree(&matches); throw FileError(FileError::IO, "file_sql: glob failed"); }
  try {
    require(matches.gl_pathc <= MAX_FILES, FileError::INVALID, "glob exceeds 100000 files");
    for (size_t i = 0; i < matches.gl_pathc; ++i) {
      struct stat candidate;
      if (::lstat(matches.gl_pathv[i], &candidate) == 0 && S_ISDIR(candidate.st_mode)) continue;
      std::string path = canonical(matches.gl_pathv[i]); check_root(path, secure_root);
      struct stat st;
      require(::stat(path.c_str(), &st) == 0, FileError::IO, "cannot stat discovered file");
      if (S_ISREG(st.st_mode)) files_.push_back(path);
    }
  } catch (...) { ::globfree(&matches); throw; }
  ::globfree(&matches); std::sort(files_.begin(),files_.end());
  if (definition_.columns.empty()) { Column c; c.name = "file"; c.type = Type::VARCHAR; definition_.columns.push_back(c); }
#else
  throw FileError(FileError::UNSUPPORTED, "file_sql: glob requires Linux");
#endif
}

std::string FileReader::encode(const Definition &d)
{
  rapidjson::StringBuffer buffer; rapidjson::Writer<rapidjson::StringBuffer> w(buffer);
  w.StartObject();
  w.Key("file_sql"); w.Int(1); w.Key("format"); w.Int(static_cast<int>(d.format));
  w.Key("path"); w.String(d.path.data(),d.path.size());
  w.Key("options"); w.StartArray(); w.Bool(d.options.header); w.Int(d.options.delimiter); w.Int(d.options.quote); w.Int(d.options.escape);
  w.String(d.options.nullstr.data(),d.options.nullstr.size()); w.EndArray();
  w.Key("snapshot"); w.StartArray(); w.Int64(d.snapshot.device); w.Int64(d.snapshot.inode); w.Int64(d.snapshot.size);
  w.Int64(d.snapshot.mtime); w.Int64(d.snapshot.ctime); w.EndArray();
  w.Key("columns"); w.StartArray();
  for (const Column &c : d.columns) { w.StartArray(); w.String(c.name.data(),c.name.size()); w.Int(static_cast<int>(c.type)); w.Bool(c.nullable); w.Int(c.datetime_scale); w.EndArray(); }
  w.EndArray(); w.EndObject();
  return std::string(buffer.GetString(),buffer.GetSize());
}

Definition FileReader::decode(const std::string &encoded)
{
  require(encoded.size() <= MAX_RECORD, FileError::INVALID, "file descriptor too large");
  rapidjson::Document doc; doc.Parse(encoded.c_str(),encoded.size());
  require(!doc.HasParseError() && doc.IsObject() && doc.HasMember("file_sql") && doc["file_sql"].IsInt() && doc["file_sql"].GetInt() == 1
          && doc.HasMember("format") && doc["format"].IsInt() && doc["format"].GetInt() >= 0 && doc["format"].GetInt() <= 3
          && doc.HasMember("path") && doc["path"].IsString() && doc.HasMember("options") && doc["options"].IsArray()
          && doc.HasMember("snapshot") && doc["snapshot"].IsArray() && doc.HasMember("columns") && doc["columns"].IsArray(),
          FileError::INVALID, "invalid file scan descriptor");
  Definition d; d.format = static_cast<Format>(doc["format"].GetInt()); d.path.assign(doc["path"].GetString(),doc["path"].GetStringLength());
  const auto &opts = doc["options"];
  require(opts.Size() == 5 && opts[0].IsBool() && opts[1].IsInt() && opts[2].IsInt() && opts[3].IsInt() && opts[4].IsString(), FileError::INVALID, "invalid scan options");
  d.options.header=opts[0].GetBool(); d.options.delimiter=opts[1].GetInt(); d.options.quote=opts[2].GetInt(); d.options.escape=opts[3].GetInt();
  d.options.nullstr.assign(opts[4].GetString(),opts[4].GetStringLength());
  const auto &snap = doc["snapshot"];
  require(snap.Size() == 5, FileError::INVALID, "invalid file snapshot");
  for (rapidjson::SizeType i=0;i<5;++i) require(snap[i].IsInt64(), FileError::INVALID, "invalid file snapshot value");
  d.snapshot.device=snap[0].GetInt64(); d.snapshot.inode=snap[1].GetInt64(); d.snapshot.size=snap[2].GetInt64(); d.snapshot.mtime=snap[3].GetInt64(); d.snapshot.ctime=snap[4].GetInt64();
  const auto &cols = doc["columns"];
  require(cols.Size() > 0 && cols.Size() <= MAX_COLUMNS, FileError::INVALID, "invalid schema width");
  for (rapidjson::SizeType i=0;i<cols.Size();++i) {
    const auto &col=cols[i];
    require(col.IsArray() && (col.Size()==3 || col.Size()==4) && col[0].IsString() && col[1].IsInt()
            && col[1].GetInt()>=0 && col[1].GetInt()<=6 && col[2].IsBool(), FileError::INVALID, "invalid schema column");
    Column c; c.name.assign(col[0].GetString(),col[0].GetStringLength()); c.type=static_cast<Type>(col[1].GetInt()); c.nullable=col[2].GetBool();
    if (col.Size() == 4) {
      require(col[3].IsInt() && col[3].GetInt() >= 0 && col[3].GetInt() <= 6,
              FileError::INVALID, "invalid datetime scale");
      c.datetime_scale = col[3].GetInt();
    }
    d.columns.push_back(std::move(c));
  }
  return d;
}

CsvExporter::~CsvExporter()
{
  if (file_ != nullptr) ::fclose(file_);
#if defined(__linux__)
  if (directory_ >= 0) {
    if (!temporary_.empty()) ::unlinkat(directory_,temporary_.c_str(),0);
    ::close(directory_);
  }
#endif
}

void CsvExporter::open(const Definition &definition,const std::string &secure_root)
{
#if defined(__linux__)
  require(file_ == nullptr && directory_ < 0,FileError::INVALID,"exporter already opened");
  options_=definition.options;columns_=definition.columns.size();
  require(columns_ > 0 && columns_ <= MAX_COLUMNS,FileError::INVALID,"invalid export schema");
  require(options_.delimiter && options_.quote && options_.escape && options_.delimiter != options_.quote
          && options_.delimiter != '\n' && options_.delimiter != '\r',FileError::INVALID,"invalid CSV export options");
  for (char c : options_.nullstr) require(c != options_.delimiter && c != options_.quote && c != '\r' && c != '\n' && c != '\0',
                                        FileError::INVALID,"NULL text cannot contain CSV syntax characters");
  require(!definition.path.empty() && definition.path.find('\0') == std::string::npos,FileError::SECURITY,"invalid export path");
  size_t slash=definition.path.find_last_of('/');
  std::string parent=canonical(slash == std::string::npos ? "." : slash == 0 ? "/" : definition.path.substr(0,slash));
  target_=definition.path.substr(slash == std::string::npos ? 0 : slash+1);
  require(!target_.empty() && target_ != "." && target_ != "..",FileError::SECURITY,"invalid export filename");
  check_root(parent+"/"+target_,secure_root);
  directory_=open_regular(parent);
  struct stat st;
  require(::fstat(directory_,&st) == 0 && S_ISDIR(st.st_mode),FileError::IO,"invalid export directory");
  require(::fstatat(directory_,target_.c_str(),&st,AT_SYMLINK_NOFOLLOW) != 0 && errno == ENOENT,
          FileError::IO,"export target already exists");
  static std::atomic<uint64_t> sequence{0};
  int fd=-1;
  for (int attempt=0;attempt<100 && fd<0;++attempt) {
    std::string candidate=".file_sql_"+std::to_string(::getpid())+"_"+std::to_string(sequence.fetch_add(1));
    fd=::openat(directory_,candidate.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
    if (fd>=0) temporary_=std::move(candidate);
    if (fd<0 && errno != EEXIST) break;
  }
  if (fd<0) throw FileError(FileError::IO,"file_sql: cannot create temporary export file");
  file_=::fdopen(fd,"wb");
  if (!file_) {::close(fd);throw FileError(FileError::IO,"file_sql: cannot open export stream");}
  if (options_.header) {
    std::vector<Cell> header;
    for (const Column &column:definition.columns) {Cell cell;cell.text=column.name;cell.is_null=false;header.push_back(std::move(cell));}
    write(header);
  }
#else
  throw FileError(FileError::UNSUPPORTED,"file_sql: CSV export requires Linux");
#endif
}

void CsvExporter::bytes(const char *data,size_t size)
{ require(file_ != nullptr && ::fwrite(data,1,size,file_) == size,FileError::IO,"cannot write CSV export"); }

void CsvExporter::field(const Cell &cell)
{
  if (cell.is_null) {bytes(options_.nullstr.data(),options_.nullstr.size());return;}
  require(cell.text.size() <= MAX_FIELD && utf8(cell.text),FileError::INVALID,"invalid export value");
  bool quote=cell.text.empty() || cell.text == options_.nullstr || cell.text == "\\N";
  for (char c:cell.text) quote |= c == options_.delimiter || c == options_.quote || c == '\r' || c == '\n';
  if (quote) bytes(&options_.quote,1);
  size_t start=0;
  for (size_t i=0;quote && i<cell.text.size();++i) {
    if (cell.text[i] == options_.quote || (options_.escape != options_.quote && cell.text[i] == options_.escape)) {
      bytes(cell.text.data()+start,i-start);bytes(&options_.escape,1);bytes(cell.text.data()+i,1);start=i+1;
    }
  }
  bytes(cell.text.data()+start,cell.text.size()-start);
  if (quote) bytes(&options_.quote,1);
}

void CsvExporter::write(const std::vector<Cell> &row)
{
  require(!committed_ && row.size() == columns_,FileError::INVALID,"invalid export row width");
  for (size_t i=0;i<row.size();++i) {if(i)bytes(&options_.delimiter,1);field(row[i]);}
  bytes("\n",1);
}

void CsvExporter::commit()
{
#if defined(__linux__)
  require(file_ != nullptr && !committed_,FileError::INVALID,"exporter is not active");
  require(::fflush(file_) == 0 && ::fsync(::fileno(file_)) == 0,FileError::IO,"cannot flush CSV export");
  // linkat is atomic and fails if another writer creates the destination first.
  if (::linkat(directory_,temporary_.c_str(),directory_,target_.c_str(),0) != 0)
    throw FileError(FileError::IO,"file_sql: cannot publish CSV export (target may already exist)");
  committed_=true;
  ::unlinkat(directory_,temporary_.c_str(),0);temporary_.clear();
#else
  throw FileError(FileError::UNSUPPORTED,"file_sql: CSV export requires Linux");
#endif
}

}}} // namespace oceanbase::sql::file_sql
