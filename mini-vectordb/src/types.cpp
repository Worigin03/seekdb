#include "types.h"

#include <cfloat>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <algorithm>

namespace mvdb {

size_t utf8Length(const std::string& s) {
  size_t n = 0;
  for (unsigned char c : s)
    if ((c & 0xC0) != 0x80) n++;
  return n;
}

// Fast decimal parser for vector literals. Avoids the very slow CRT strtod on
// Windows while staying within float32 accuracy.
static bool parseDoubleFast(const char*& p, const char* end, double& out) {
  while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
  bool neg = false;
  if (p < end && (*p == '-' || *p == '+')) {
    neg = (*p == '-');
    p++;
  }
  uint64_t mant = 0;
  int digits = 0;
  int exp10 = 0;
  while (p < end && *p >= '0' && *p <= '9') {
    if (digits < 19) {
      mant = mant * 10 + (uint64_t)(*p - '0');
      digits++;
    } else {
      exp10++;
    }
    p++;
  }
  if (p < end && *p == '.') {
    p++;
    while (p < end && *p >= '0' && *p <= '9') {
      if (digits < 19) {
        mant = mant * 10 + (uint64_t)(*p - '0');
        digits++;
        exp10--;
      }
      p++;
    }
  }
  if (digits == 0) return false;
  if (p < end && (*p == 'e' || *p == 'E')) {
    p++;
    bool eneg = false;
    if (p < end && (*p == '+' || *p == '-')) {
      eneg = (*p == '-');
      p++;
    }
    if (p >= end || *p < '0' || *p > '9') return false;
    int e = 0;
    while (p < end && *p >= '0' && *p <= '9') {
      if (e < 10000) e = e * 10 + (*p - '0');
      p++;
    }
    exp10 += eneg ? -e : e;
  }
  double v = (double)mant;
  if (exp10 != 0) {
    static const double kPow[] = {1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,
                                  1e8,  1e9,  1e10, 1e11, 1e12, 1e13, 1e14, 1e15};
    int e = exp10 < 0 ? -exp10 : exp10;
    double scale = 1.0;
    if (e <= 15) {
      scale = kPow[e];
    } else {
      scale = std::pow(10.0, (double)e);
    }
    v = exp10 > 0 ? v * scale : v / scale;
  }
  out = neg ? -v : v;
  return true;
}

bool parseVectorLiteral(const std::string& text, int expectDim, std::vector<float>& out,
                        std::string& err) {
  const char* p = text.c_str();
  const char* end = p + text.size();
  while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
  bool bracketed = false;
  if (p < end && *p == '[') {
    bracketed = true;
    p++;
  }
  out.clear();
  while (true) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    if (bracketed && p < end && *p == ']' && out.empty()) {
      p++;
      break;
    }
    if (p >= end) {
      if (bracketed) {
        err = "vector literal is not terminated";
        return false;
      }
      break;
    }
    double d = 0;
    if (!parseDoubleFast(p, end, d)) {
      err = "invalid number in vector literal";
      return false;
    }
    if (!std::isfinite(d)) {
      err = "vector element must be finite";
      return false;
    }
    if (d > (double)FLT_MAX || d < -(double)FLT_MAX) {
      err = "vector element out of float32 range";
      return false;
    }
    out.push_back((float)d);
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    if (p < end && *p == ',') {
      p++;
      continue;
    }
    if (bracketed && p < end && *p == ']') {
      p++;
      break;
    }
    if (!bracketed && p >= end) break;
    err = "invalid vector literal";
    return false;
  }
  while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
  if (p != end) {
    err = "trailing characters in vector literal";
    return false;
  }
  if (expectDim > 0 && (int)out.size() != expectDim) {
    err = "vector dimension mismatch";
    return false;
  }
  return true;
}

void encodeRow(const Schema& sc, const std::vector<Value>& row, std::string& out) {
  for (size_t i = 0; i < sc.cols.size(); i++) {
    if ((int)i == sc.pk) continue;
    const Value& v = row[i];
    switch (sc.cols[i].type) {
      case Type::BIGINT: {
        uint64_t u = (uint64_t)v.i;
        char b[8];
        for (int k = 0; k < 8; k++) b[k] = (char)((u >> (8 * k)) & 0xff);
        out.append(b, 8);
        break;
      }
      case Type::VARCHAR: {
        putVarint(out, v.s.size());
        out.append(v.s);
        break;
      }
      case Type::VECTOR: {
        size_t dim = (size_t)sc.cols[i].len;
        if (v.v.size() != dim) throw std::runtime_error("vector size does not match schema");
        out.append((const char*)v.v.data(), dim * sizeof(float));
        break;
      }
    }
  }
}

bool decodeRow(const Schema& sc, const char* p, size_t n, int64_t pk, std::vector<Value>& out) {
  const char* end = p + n;
  // Keeps the previous buffers alive so a caller that decodes row after row
  // (result materialisation, filter caches) does not allocate per row.
  if (out.size() != sc.cols.size()) out.resize(sc.cols.size());
  for (size_t i = 0; i < sc.cols.size(); i++) {
    Value& v = out[i];
    v.type = sc.cols[i].type;
    if ((int)i == sc.pk) {
      v.i = pk;
      continue;
    }
    switch (sc.cols[i].type) {
      case Type::BIGINT: {
        if (p + 8 > end) return false;
        uint64_t u = 0;
        for (int k = 0; k < 8; k++) u |= (uint64_t)(uint8_t)p[k] << (8 * k);
        v.i = (int64_t)u;
        p += 8;
        break;
      }
      case Type::VARCHAR: {
        uint64_t len = 0;
        if (!getVarint(p, end, len)) return false;
        if (p + len > end) return false;
        v.s.assign(p, (size_t)len);
        p += len;
        break;
      }
      case Type::VECTOR: {
        int dim = sc.cols[i].len;
        if (p + (size_t)dim * 4 > end) return false;
        v.v.resize((size_t)dim);
        memcpy(v.v.data(), p, (size_t)dim * 4);
        p += (size_t)dim * 4;
        break;
      }
    }
  }
  return p == end;
}

std::string formatDouble(double d) {
  if (d == 0) d = 0;  // normalise -0
  char buf[64];
  snprintf(buf, sizeof(buf), "%.17g", d);
  // Drop trailing digits while the shorter form still parses back to the same
  // double: this yields the shortest exact representation of the value.
  size_t n = strlen(buf);
  char tmp[64];
  while (n > 1) {
    char last = buf[n - 1];
    if (last < '0' || last > '9') break;
    memcpy(tmp, buf, n - 1);
    tmp[n - 1] = 0;
    if (strtod(tmp, nullptr) != d) break;
    n--;
    buf[n] = 0;
  }
  return std::string(buf);
}

std::string formatValue(const Value& v, bool isDistance) {
  switch (v.type) {
    case Type::BIGINT:
      return std::to_string(v.i);
    case Type::VARCHAR:
      return v.s;
    case Type::VECTOR: {
      std::string r = "[";
      char buf[32];
      for (size_t i = 0; i < v.v.size(); i++) {
        if (i) r.push_back(',');
        snprintf(buf, sizeof(buf), "%g", (double)v.v[i]);
        r += buf;
      }
      r.push_back(']');
      return r;
    }
  }
  (void)isDistance;
  return std::string();
}

}  // namespace mvdb
