// Schema, values and row encoding.
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

#include "platform.h"

namespace mvdb {

enum class Type : uint8_t { BIGINT = 1, VARCHAR = 2, VECTOR = 3 };

struct Column {
  std::string name;
  Type type = Type::BIGINT;
  int len = 0;  // VARCHAR: max characters; VECTOR: dimension
};

struct Schema {
  std::string name;
  std::vector<Column> cols;
  int pk = -1;
  int vec = -1;

  int colIndex(const std::string& n) const {
    for (size_t i = 0; i < cols.size(); i++)
      if (iequals(cols[i].name, n)) return (int)i;
    return -1;
  }
};

struct Value {
  Type type = Type::BIGINT;
  int64_t i = 0;
  std::string s;
  std::vector<float> v;

  Value() = default;
  static Value ofInt(int64_t x) {
    Value r;
    r.type = Type::BIGINT;
    r.i = x;
    return r;
  }
  static Value ofStr(std::string x) {
    Value r;
    r.type = Type::VARCHAR;
    r.s = std::move(x);
    return r;
  }
  static Value ofVec(std::vector<float> x) {
    Value r;
    r.type = Type::VECTOR;
    r.v = std::move(x);
    return r;
  }
};

// Number of Unicode code points in a UTF-8 string (invalid bytes count as one).
size_t utf8Length(const std::string& s);

// Parses a vector literal such as "[1, 2.5, -3e2]". Returns false and fills err
// on syntax errors, non-finite values or values outside float32 range.
bool parseVectorLiteral(const std::string& text, int expectDim, std::vector<float>& out,
                        std::string& err);

// Row payload encoding. The primary-key column is stored separately in the WAL
// record, so it is skipped here.
void encodeRow(const Schema& sc, const std::vector<Value>& row, std::string& out);
bool decodeRow(const Schema& sc, const char* p, size_t n, int64_t pk, std::vector<Value>& out);

std::string formatValue(const Value& v, bool isDistance = false);
std::string formatDouble(double d);

}  // namespace mvdb
