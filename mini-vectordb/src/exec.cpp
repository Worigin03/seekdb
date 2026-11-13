#include "engine.h"
#include "simd.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>
#include <cstdio>
#include <set>
#include <unordered_set>

namespace mvdb {

// ----------------------------------------------------------------- helpers ---

static uint64_t hashStr(const std::string& s) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

static inline bool cmpIntOp(int64_t a, int64_t b, Cond::Op op) {
  switch (op) {
    case Cond::EQ: return a == b;
    case Cond::NE: return a != b;
    case Cond::LT: return a < b;
    case Cond::LE: return a <= b;
    case Cond::GT: return a > b;
    case Cond::GE: return a >= b;
  }
  return false;
}

// Byte-wise compare == Unicode code point order for UTF-8.
static inline bool cmpStrOp(const char* p, size_t n, const std::string& b, Cond::Op op) {
  size_t m = b.size();
  size_t k = n < m ? n : m;
  int r = k ? memcmp(p, b.data(), k) : 0;
  if (r == 0) r = (n < m) ? -1 : (n > m ? 1 : 0);
  switch (op) {
    case Cond::EQ: return r == 0;
    case Cond::NE: return r != 0;
    case Cond::LT: return r < 0;
    case Cond::LE: return r <= 0;
    case Cond::GT: return r > 0;
    case Cond::GE: return r >= 0;
  }
  return false;
}

Table* Database::findTable(const std::string& name) {
  for (auto& t : tables_)
    if (iequals(t->schema.name, name)) return t.get();
  return nullptr;
}

Table* Database::getTableOrThrow(const std::string& name) {
  Table* t = findTable(name);
  if (!t) throwErr(1146, "42S02", "Table 'minivectordb." + name + "' doesn't exist");
  return t;
}

void Database::decodeRowAt(Table& t, uint64_t rowid, std::vector<Value>& out) {
  const char* p = payloadPtr(t, rowid);
  decodeRow(t.schema, p, t.len[rowid], t.pkOf[rowid], out);
}

void Database::ensureCaches(Table& t, int col) {
  if (col < 0 || col >= (int)t.cache.size()) return;
  ColCache& cc = t.cache[col];
  if (cc.ready) return;
  if (t.nextRowid > 4000000ull) return;  // keep memory bounded on huge tables
  cc.ready = true;
  std::vector<Value> row;
  if (t.schema.cols[col].type == Type::BIGINT) {
    cc.ints.assign(t.nextRowid, 0);
    for (uint64_t r = 0; r < t.nextRowid; r++) {
      if (!t.live[r]) continue;
      decodeRowAt(t, r, row);
      cc.ints[r] = row[col].i;
    }
  } else if (t.schema.cols[col].type == Type::VARCHAR) {
    cc.hashes.assign(t.nextRowid, 0);
    cc.lens.assign(t.nextRowid, 0);
    cc.offs.assign(t.nextRowid, 0);
    size_t total = 0;
    for (uint64_t r = 0; r < t.nextRowid; r++) {
      if (!t.live[r]) continue;
      decodeRowAt(t, r, row);
      total += row[col].s.size();
    }
    cc.arena.reserve(total + 16);
    for (uint64_t r = 0; r < t.nextRowid; r++) {
      if (!t.live[r]) continue;
      decodeRowAt(t, r, row);
      cc.offs[r] = (uint32_t)cc.arena.size();
      cc.lens[r] = (uint32_t)row[col].s.size();
      cc.hashes[r] = hashStr(row[col].s);
      cc.arena += row[col].s;
    }
  }
}

std::unique_ptr<CompiledCond> Database::compileCond(Table& t, const Cond& c) {
  auto out = std::make_unique<CompiledCond>();
  out->kind = (CompiledCond::Kind)c.kind;
  if (c.kind != Cond::CMP) {
    for (auto& k : c.kids) out->kids.push_back(compileCond(t, *k));
    return out;
  }
  int ci = t.schema.colIndex(c.col);
  if (ci < 0) throwErr(1054, "42S22", "Unknown column '" + c.col + "' in 'where clause'");
  const Column& col = t.schema.cols[ci];
  if (col.type == Type::VECTOR)
    throwErr(1235, "42000", "comparison on VECTOR column is not supported");
  if (col.type == Type::BIGINT && c.val.type != Type::BIGINT)
    throwErr(1366, "HY000", "Incorrect integer value for column '" + c.col + "'");
  if (col.type == Type::VARCHAR && c.val.type != Type::VARCHAR)
    throwErr(1366, "HY000", "Incorrect string value for column '" + c.col + "'");
  out->col = ci;
  out->op = c.op;
  out->type = col.type;
  if (col.type == Type::BIGINT) {
    out->ival = c.val.i;
  } else {
    out->sval = c.val.s;
    out->shash = hashStr(c.val.s);
  }
  ensureCaches(t, ci);
  return out;
}

bool Database::evalCond(Table& t, const CompiledCond& c, uint64_t rowid) {
  switch (c.kind) {
    case CompiledCond::AND:
      for (auto& k : c.kids)
        if (!evalCond(t, *k, rowid)) return false;
      return true;
    case CompiledCond::OR:
      for (auto& k : c.kids)
        if (evalCond(t, *k, rowid)) return true;
      return false;
    case CompiledCond::NOT:
      return !evalCond(t, *c.kids[0], rowid);
    case CompiledCond::CMP:
      break;
  }
  const ColCache& cc = t.cache[c.col];
  if (c.type == Type::BIGINT) {
    if (cc.ready && cc.ints.size() > rowid) return cmpIntOp(cc.ints[rowid], c.ival, c.op);
    std::vector<Value> row;
    decodeRowAt(t, rowid, row);
    return cmpIntOp(row[c.col].i, c.ival, c.op);
  }
  if (cc.ready && cc.lens.size() > rowid) {
    const char* p = cc.arena.data() + cc.offs[rowid];
    size_t n = cc.lens[rowid];
    if (c.op == Cond::EQ || c.op == Cond::NE) {
      bool eq = cc.hashes[rowid] == c.shash && n == c.sval.size() &&
                (n == 0 || memcmp(p, c.sval.data(), n) == 0);
      return c.op == Cond::EQ ? eq : !eq;
    }
    return cmpStrOp(p, n, c.sval, c.op);
  }
  std::vector<Value> row;
  decodeRowAt(t, rowid, row);
  return cmpStrOp(row[c.col].s.data(), row[c.col].s.size(), c.sval, c.op);
}

const std::vector<uint32_t>* Database::topkFilterRows(Table& t, const CompiledCond& c) {
  constexpr size_t maxFilters = 16;
  constexpr size_t maxRowsPerFilter = 65536;
  if (c.kind != CompiledCond::CMP || c.op != Cond::EQ || c.type != Type::VARCHAR ||
      c.sval.size() > 256 || t.nextRowid > UINT32_MAX)
    return nullptr;
  const auto& hashes = t.cache[c.col].hashes;
  if (hashes.size() < t.nextRowid) return nullptr;
  for (const auto& filter : t.topkFilters) {
    if (filter.column == c.col && filter.value == c.sval)
      return filter.usable ? &filter.rowids : nullptr;
  }

  // Materialize selective matches once per predicate. A dense predicate
  // uses the normal scan, and mutations invalidate the cached row IDs.
  TopKFilter filter{c.col, c.sval, true, {}};
  for (uint64_t rowid = 0; rowid < t.nextRowid; ++rowid) {
    if (hashes[rowid] != c.shash || !t.live[rowid] || !evalCond(t, c, rowid))
      continue;
    if (filter.rowids.size() == maxRowsPerFilter) {
      filter.usable = false;
      filter.rowids = std::vector<uint32_t>{};
      break;
    }
    filter.rowids.push_back((uint32_t)rowid);
  }
  if (t.topkFilters.size() == maxFilters) t.topkFilters.erase(t.topkFilters.begin());
  t.topkFilters.push_back(std::move(filter));
  const auto& entry = t.topkFilters.back();
  return entry.usable ? &entry.rowids : nullptr;
}

// --------------------------------------------------------------------- DDL ---

void Database::execCreateTable(const Stmt& st, ResultSet& rs) {
  (void)rs;
  const CreateTableStmt& ct = st.createTable;
  if (findTable(ct.name)) {
    if (ct.ifNotExists) return;
    throwErr(1050, "42S01", "Table '" + ct.name + "' already exists");
  }
  if (ct.cols.empty()) throwErr(1064, "42000", "a table needs at least one column");
  auto t = std::make_unique<Table>();
  t->id = nextTableId_++;
  t->schema.name = ct.name;
  t->schema.cols = ct.cols;
  for (size_t i = 0; i < t->schema.cols.size(); i++)
    for (size_t j = i + 1; j < t->schema.cols.size(); j++)
      if (iequals(t->schema.cols[i].name, t->schema.cols[j].name))
        throwErr(1060, "42S21", "Duplicate column name '" + t->schema.cols[i].name + "'");

  int pk = -1;
  if (!ct.pkCol.empty()) pk = t->schema.colIndex(ct.pkCol);
  if (pk < 0) throwErr(1235, "42000", "a BIGINT PRIMARY KEY column is required");
  if (t->schema.cols[pk].type != Type::BIGINT)
    throwErr(1235, "42000", "PRIMARY KEY must be BIGINT");
  t->schema.pk = pk;

  int vc = -1, vcount = 0;
  for (size_t i = 0; i < t->schema.cols.size(); i++)
    if (t->schema.cols[i].type == Type::VECTOR) {
      vc = (int)i;
      vcount++;
    }
  if (vcount > 1) throwErr(1235, "42000", "at most one VECTOR column per table");
  t->schema.vec = vc;

  t->pk.init(1024);
  t->cache.resize(t->schema.cols.size());
  if (vc >= 0) {
    t->vecs = std::make_unique<VecStore>();
    t->vecs->init(t->schema.cols[vc].len, dir_ + "/t" + std::to_string(t->id) + ".vec",
                  vecMemBudget_);
  }
  Table* raw = t.get();
  tables_.push_back(std::move(t));
  std::string body;
  encodeCreateTable(*raw, body);
  appendRecord(body);
}

void Database::execDropTable(const Stmt& st, ResultSet& rs) {
  (void)rs;
  Table* t = findTable(st.dropTable.name);
  if (!t) {
    if (st.dropTable.ifExists) return;
    throwErr(1051, "42S02", "Unknown table '" + st.dropTable.name + "'");
  }
  std::string body;
  body.push_back((char)2);
  putVarint(body, t->id);
  appendRecord(body);
  applyDropTable(t->id);
}

void Database::execCreateIndex(const Stmt& st, ResultSet& rs) {
  (void)rs;
  Table* t = getTableOrThrow(st.createIndex.table);
  int ci = t->schema.colIndex(st.createIndex.column);
  if (ci < 0)
    throwErr(1054, "42S22", "Unknown column '" + st.createIndex.column + "' in 'field list'");
  if (ci != t->schema.vec)
    throwErr(1235, "42000", "vector index must be created on the VECTOR column");
  // The default index is a flat (exact) index: the vector column is already
  // stored contiguously, so a filtered/unfiltered TopK is a single SIMD scan.
  // A graph index is available for experiments via MVDB_HNSW=1.
  const char* useHnsw = getenv("MVDB_HNSW");
  if (!useHnsw || atoi(useHnsw) == 0 || st.createIndex.algorithm == "flat" ||
      st.createIndex.algorithm == "none") {
    t->hnsw.reset();
    return;
  }
  if (!t->hnsw) t->hnsw = std::make_unique<Hnsw>();
  t->hnsw->build(t);
}

// ------------------------------------------------------------------- DML ---

static void checkVarchar(const Column& c, const std::string& s, int rowNo) {
  if ((int)utf8Length(s) > c.len)
    throwErr(1406, "22001", "Data too long for column '" + c.name + "' at row " +
                                 std::to_string(rowNo));
}

static void coerceValue(const Column& c, const Expr& e, Value& out, int rowNo) {
  switch (c.type) {
    case Type::BIGINT: {
      if (e.kind != Expr::INT_LIT)
        throwErr(1366, "HY000", "Incorrect integer value for column '" + c.name + "' at row " +
                                     std::to_string(rowNo));
      out.i = e.i;
      break;
    }
    case Type::VARCHAR: {
      if (e.kind == Expr::STR_LIT) {
        out.s.assign(e.s);
      } else if (e.kind == Expr::INT_LIT) {
        out.s.assign(std::to_string(e.i));
      } else {
        throwErr(1366, "HY000", "Incorrect string value for column '" + c.name + "' at row " +
                                     std::to_string(rowNo));
      }
      checkVarchar(c, out.s, rowNo);
      break;
    }
    case Type::VECTOR: {
      if (e.kind != Expr::VEC_LIT)
        throwErr(1366, "HY000", "Incorrect vector value for column '" + c.name + "' at row " +
                                     std::to_string(rowNo));
      if (e.vecOk) {
        // Already parsed when the statement was parsed: only the dimension is
        // schema-dependent.
        if ((int)e.vec.size() != c.len)
          throwErr(1366, "HY000", "Incorrect vector value for column '" + c.name + "' at row " +
                                       std::to_string(rowNo) + ": vector dimension mismatch");
        out.v.assign(e.vec.begin(), e.vec.end());
      } else {
        // Parses into the caller's buffer, which is reused by the next row.
        std::string err;
        if (!parseVectorLiteral(e.vecText, c.len, out.v, err))
          throwErr(1366, "HY000", "Incorrect vector value for column '" + c.name + "' at row " +
                                       std::to_string(rowNo) + ": " + err);
      }
      break;
    }
  }
  out.type = c.type;
}

static int64_t g_tCoerce = 0, g_tEncode = 0, g_tWal = 0, g_tApply = 0;
static bool g_insTiming = getenv("MVDB_INS_TIMING") != nullptr;

void Database::execInsert(const Stmt& st, ResultSet& rs) {
  int64_t tt0 = g_insTiming ? nowMicros() : 0;
  Table* t = getTableOrThrow(st.insert.table);
  const Schema& sc = t->schema;
  std::vector<int> colMap;
  if (st.insert.cols.empty()) {
    for (size_t i = 0; i < sc.cols.size(); i++) colMap.push_back((int)i);
  } else {
    for (auto& name : st.insert.cols) {
      int ci = sc.colIndex(name);
      if (ci < 0) throwErr(1054, "42S22", "Unknown column '" + name + "' in 'field list'");
      colMap.push_back(ci);
    }
  }
  // Source column index per schema column (-1 when the column is not listed).
  std::vector<int> srcOf(sc.cols.size(), -1);
  for (size_t i = 0; i < colMap.size(); i++) srcOf[colMap[i]] = (int)i;

  // Coerce and encode in one pass. The row buffer and the payload strings are
  // reused across rows so a bulk INSERT does not allocate per value.
  std::vector<std::string> payloads;
  std::vector<int64_t> pks;
  payloads.reserve(st.insert.rows.size());
  pks.reserve(st.insert.rows.size());
  std::vector<Value> row(sc.cols.size());
  std::unordered_set<int64_t> seen;
  seen.reserve(st.insert.rows.size() * 2 + 8);
  for (size_t r = 0; r < st.insert.rows.size(); r++) {
    const auto& src = st.insert.rows[r];
    if (src.size() != colMap.size())
      throwErr(1136, "21S01", "Column count doesn't match value count at row " +
                                   std::to_string(r + 1));
    for (size_t i = 0; i < sc.cols.size(); i++) {
      Value& v = row[i];
      v.type = sc.cols[i].type;
      int si = srcOf[i];
      if (si >= 0) {
        coerceValue(sc.cols[i], *src[si], v, (int)r + 1);
        continue;
      }
      // Columns the statement does not mention: VARCHAR defaults to '', the
      // primary key keeps its default (0), anything else is an error.
      if ((int)i == sc.pk) {
        v.i = 0;
        continue;
      }
      if (sc.cols[i].type != Type::VARCHAR)
        throwErr(1364, "HY000", "Field '" + sc.cols[i].name + "' doesn't have a default value");
      v.s.clear();
    }
    int64_t pk = row[sc.pk].i;
    uint64_t existing = 0;
    if (t->pk.get(pk, existing))
      throwErr(1062, "23000", "Duplicate entry '" + std::to_string(pk) + "' for key 'PRIMARY'");
    if (!seen.insert(pk).second)
      throwErr(1062, "23000", "Duplicate entry '" + std::to_string(pk) + "' for key 'PRIMARY'");
    payloads.emplace_back();
    encodeRow(sc, row, payloads.back());
    pks.push_back(pk);
  }
  int64_t tt1 = g_insTiming ? nowMicros() : 0;

  std::vector<std::string> bodies(payloads.size());
  std::vector<uint64_t> rowids(payloads.size());
  bodies.reserve(payloads.size());
  for (size_t i = 0; i < payloads.size(); i++) {
    uint64_t rowid = t->nextRowid++;
    std::string& body = bodies[i];
    body.reserve(payloads[i].size() + 16);
    body.push_back((char)3);
    putVarint(body, t->id);
    putVarint(body, rowid);
    for (int k = 0; k < 8; k++) body.push_back((char)((uint64_t)pks[i] >> (8 * k)));
    body.append(payloads[i]);
    rowids[i] = rowid;
  }
  int64_t tt2 = g_insTiming ? nowMicros() : 0;
  std::vector<uint64_t> offs;
  appendRecords(bodies, offs);
  int64_t tt3 = g_insTiming ? nowMicros() : 0;
  rs.affected = (int64_t)rowids.size();
  for (size_t i = 0; i < rowids.size(); i++) {
    // payload sits at the tail of the record body
    uint64_t payloadOff = offs[i] + (bodies[i].size() - payloads[i].size());
    applyPendingInsert(*t, rowids[i], pks[i], payloads[i].data(), (uint32_t)payloads[i].size(),
                       payloadOff, true);
  }
  if (g_insTiming) {
    int64_t tt4 = nowMicros();
    g_tCoerce += tt1 - tt0;
    g_tEncode += tt2 - tt1;
    g_tWal += tt3 - tt2;
    g_tApply += tt4 - tt3;
  }
}

void Database::dumpInsertTiming() {
  if (!g_insTiming) return;
  fprintf(stderr, "INSERT coerce=%.1fms encode=%.1fms wal=%.1fms apply=%.1fms\n",
          g_tCoerce / 1000.0, g_tEncode / 1000.0, g_tWal / 1000.0, g_tApply / 1000.0);
}

void Database::execDelete(const Stmt& st, ResultSet& rs) {
  Table* t = getTableOrThrow(st.del.table);
  std::unique_ptr<CompiledCond> where;
  if (st.del.where) where = compileCond(*t, *st.del.where);
  std::vector<uint64_t> hits;
  uint64_t limit = t->nextRowid;
  for (uint64_t r = 0; r < limit; r++) {
    if (!t->live[r]) continue;
    if (where && !evalCond(*t, *where, r)) continue;
    hits.push_back(r);
  }
  if (hits.empty()) return;
  std::vector<std::string> bodies;
  for (uint64_t r : hits) {
    std::string body;
    body.push_back((char)4);
    putVarint(body, t->id);
    putVarint(body, r);
    int64_t pk = t->pkOf[r];
    for (int k = 0; k < 8; k++) body.push_back((char)((uint64_t)pk >> (8 * k)));
    bodies.push_back(std::move(body));
  }
  std::vector<uint64_t> offs;
  appendRecords(bodies, offs);
  rs.affected = (int64_t)hits.size();
  for (uint64_t r : hits) applyPendingDelete(*t, r, t->pkOf[r], true);
}

void Database::execUpdate(const Stmt& st, ResultSet& rs) {
  Table* t = getTableOrThrow(st.update.table);
  const Schema& sc = t->schema;
  std::vector<int> setCols;
  std::vector<Value> setVals;
  for (auto& kv : st.update.sets) {
    int ci = sc.colIndex(kv.first);
    if (ci < 0) throwErr(1054, "42S22", "Unknown column '" + kv.first + "' in 'field list'");
    Value v;
    coerceValue(sc.cols[ci], *kv.second, v, 1);
    setCols.push_back(ci);
    setVals.push_back(std::move(v));
  }
  std::unique_ptr<CompiledCond> where;
  if (st.update.where) where = compileCond(*t, *st.update.where);
  std::vector<uint64_t> hits;
  uint64_t limit = t->nextRowid;
  for (uint64_t r = 0; r < limit; r++) {
    if (!t->live[r]) continue;
    if (where && !evalCond(*t, *where, r)) continue;
    hits.push_back(r);
  }
  if (hits.empty()) return;

  std::vector<std::string> bodies;
  std::vector<std::string> payloads;
  std::vector<uint64_t> oldIds, newIds;
  std::vector<int64_t> pks;
  std::vector<Value> row;
  for (uint64_t r : hits) {
    decodeRowAt(*t, r, row);
    int64_t pk = row[sc.pk].i;
    for (size_t i = 0; i < setCols.size(); i++) {
      if (setCols[i] == sc.pk && setVals[i].i != pk)
        throwErr(1235, "42000", "updating the primary key is not supported");
      row[setCols[i]] = setVals[i];
    }
    std::string payload;
    encodeRow(sc, row, payload);
    uint64_t newId = t->nextRowid++;
    std::string body;
    body.push_back((char)5);
    putVarint(body, t->id);
    putVarint(body, r);
    putVarint(body, newId);
    for (int k = 0; k < 8; k++) body.push_back((char)((uint64_t)pk >> (8 * k)));
    body.append(payload);
    bodies.push_back(std::move(body));
    payloads.push_back(std::move(payload));
    oldIds.push_back(r);
    newIds.push_back(newId);
    pks.push_back(pk);
  }
  std::vector<uint64_t> offs;
  appendRecords(bodies, offs);
  rs.affected = (int64_t)oldIds.size();
  for (size_t i = 0; i < oldIds.size(); i++) {
    uint64_t payloadOff = offs[i] + (bodies[i].size() - payloads[i].size());
    applyUpdate(t->id, oldIds[i], newIds[i], pks[i], payloads[i].data(),
                (uint32_t)payloads[i].size(), payloadOff);
    if (t->hnsw) t->hnsw->addRow(newIds[i]);
    undo_.push_back(Undo{0, t->id, newIds[i], pks[i]});
    undo_.push_back(Undo{1, t->id, oldIds[i], pks[i]});
  }
}

// ----------------------------------------------------------------- SELECT ---

namespace {
struct CompiledItem {
  int col = -1;
  bool dist = false;
  bool isConst = false;
  bool squared = false;
  Value cval;
  int distCol = -1;
  std::vector<float> q;
  std::string header;
};

struct Cand {
  double d;
  int64_t pk;
  uint64_t rowid;
};
}  // namespace

static Database::ResultCell resultCell(const Value& value) {
  if (value.type == Type::BIGINT) return value.i;
  if (value.type == Type::VARCHAR) return value.s;
  return value.v;
}

void Database::execSelect(const Stmt& st, ResultSet& rs, const std::string& searchMode) {
  Table* t = getTableOrThrow(st.select.table);
  const Schema& sc = t->schema;

  std::vector<CompiledItem> items;
  if (st.select.star) {
    for (size_t i = 0; i < sc.cols.size(); i++) {
      CompiledItem it;
      it.col = (int)i;
      it.header = sc.cols[i].name;
      items.push_back(std::move(it));
    }
  } else {
    for (auto& e : st.select.items) {
      CompiledItem it;
      if (e->kind == Expr::DIST) {
        it.dist = true;
        it.squared = (e->s == "sq");
        it.distCol = sc.colIndex(e->distCol);
        if (it.distCol < 0)
          throwErr(1054, "42S22", "Unknown column '" + e->distCol + "' in 'field list'");
        if (it.distCol != sc.vec)
          throwErr(1235, "42000", "L2_DISTANCE requires the VECTOR column");
        int dim = sc.cols[it.distCol].len;
        std::string err;
        if (e->vecOk) {
          if ((int)e->vec.size() != dim)
            throwErr(1366, "HY000", "Incorrect vector value: vector dimension mismatch");
          it.q = e->vec;
        } else if (!parseVectorLiteral(e->vecText, dim, it.q, err)) {
          throwErr(1366, "HY000", "Incorrect vector value: " + err);
        }
        it.header = e->hasAlias ? e->alias : e->raw;
      } else if (e->kind == Expr::COLUMN) {
        it.col = sc.colIndex(e->name);
        if (it.col < 0)
          throwErr(1054, "42S22", "Unknown column '" + e->name + "' in 'field list'");
        it.header = e->hasAlias ? e->alias : e->name;
      } else if (e->kind == Expr::INT_LIT || e->kind == Expr::STR_LIT) {
        it.isConst = true;
        it.cval = e->kind == Expr::INT_LIT ? Value::ofInt(e->i) : Value::ofStr(e->s);
        it.header = e->hasAlias ? e->alias : e->raw;
      } else {
        throwErr(1235, "42000", "unsupported select expression");
      }
      items.push_back(std::move(it));
    }
  }

  // ORDER BY resolution
  int orderCol = -1;
  bool orderDist = false, orderDesc = st.select.orderDesc;
  int orderDistCol = -1;
  bool orderSquared = false;
  std::vector<float> orderQ;
  if (st.select.hasOrder) {
    ExprPtr oe = st.select.orderExpr;
    if (oe && oe->kind == Expr::INT_LIT) {
      int idx = (int)oe->i - 1;
      if (idx < 0 || idx >= (int)items.size())
        throwErr(1054, "42S22", "Unknown column '" + std::to_string(oe->i) + "' in 'order clause'");
      if (items[idx].dist) {
        orderDist = true;
        orderDistCol = items[idx].distCol;
        orderQ = items[idx].q;
        orderSquared = items[idx].squared;
      } else if (!items[idx].isConst) {
        orderCol = items[idx].col;
      } else {
        throwErr(1235, "42000", "cannot ORDER BY a constant");
      }
    } else if (oe && oe->kind == Expr::DIST) {
      orderDist = true;
      orderSquared = (oe->s == "sq");
      orderDistCol = sc.colIndex(oe->distCol);
      if (orderDistCol < 0)
        throwErr(1054, "42S22", "Unknown column '" + oe->distCol + "' in 'order clause'");
      if (orderDistCol != sc.vec)
        throwErr(1235, "42000", "L2_DISTANCE requires the VECTOR column");
      std::string err;
      if (oe->vecOk) {
        if ((int)oe->vec.size() != sc.cols[orderDistCol].len)
          throwErr(1366, "HY000", "Incorrect vector value: vector dimension mismatch");
        orderQ = oe->vec;
      } else if (!parseVectorLiteral(oe->vecText, sc.cols[orderDistCol].len, orderQ, err)) {
        throwErr(1366, "HY000", "Incorrect vector value: " + err);
      }
    } else if (oe && oe->kind == Expr::COLUMN) {
      bool matched = false;
      for (size_t i = 0; i < items.size(); i++) {
        if (iequals(items[i].header, oe->name)) {
          if (items[i].dist) {
            orderDist = true;
            orderDistCol = items[i].distCol;
            orderQ = items[i].q;
            orderSquared = items[i].squared;
          } else {
            orderCol = items[i].col;
          }
          matched = true;
          break;
        }
      }
      if (!matched) {
        orderCol = sc.colIndex(oe->name);
        if (orderCol < 0)
          throwErr(1054, "42S22", "Unknown column '" + oe->name + "' in 'order clause'");
      }
    } else {
      throwErr(1235, "42000", "unsupported ORDER BY expression");
    }
  }

  rs.hasResult = true;
  rs.affected = 0;
  for (auto& it : items) rs.headers.push_back(it.header);

  std::unique_ptr<CompiledCond> where;
  if (st.select.where) where = compileCond(*t, *st.select.where);
  auto passes = [&](uint64_t r) -> bool { return !where || evalCond(*t, *where, r); };
  int dim = sc.vec >= 0 ? sc.cols[sc.vec].len : 0;
  const float* vbase = (dim > 0 && t->vecs) ? t->vecs->base() : nullptr;
  // Stored components are float32, but squared distances need float64 range.
  auto distOf = [&](uint64_t r, const std::vector<float>& q) -> double {
    return l2sqRef(vbase + r * (size_t)dim, q.data(), dim);
  };

  int64_t lim = st.select.hasLimit ? st.select.limit : -1;
  int64_t off = st.select.hasLimit ? st.select.offset : 0;
  if (lim == 0) return;  // LIMIT 0 never returns rows

  // --- vector top-k fast path
  if (orderDist && lim > 0 && dim > 0) {
    // Never ask the scan kernel for more slots than there are rows: LIMIT may
    // legally be far larger than the table.
    const int64_t rowCount = (int64_t)t->nextRowid;
    if (off >= rowCount) return;
    const int64_t k = off + std::min(lim, rowCount - off);
    std::vector<Cand> results;
    bool usedIndex = false;
    if (searchMode == "ann" && t->hnsw && !orderDesc) {
      auto accept = [&](uint64_t r) { return passes(r); };
      std::vector<std::pair<float, uint64_t>> hits;
      static const int efOverride = getenv("MVDB_HNSW_EF") ? atoi(getenv("MVDB_HNSW_EF")) : 0;
      int ef = (int)std::max<int64_t>(efOverride > 0 ? efOverride : 64, k * 8);
      t->hnsw->search(orderQ.data(), (int)k, ef, accept, hits);
      if ((int64_t)hits.size() >= k) {
        usedIndex = true;
        for (auto& h : hits)
          results.push_back(Cand{(double)h.first, t->pkOf[h.second], h.second});
      }
    }
    // A LIMIT in the hundreds would make the k-wide insertion buffer quadratic
    // (n * k). Collect every qualifying distance and partial-sort instead.
    if (!usedIndex && k > 256) {
      std::vector<Cand> all;
      all.reserve((size_t)t->nextRowid);
      for (uint64_t r = 0; r < t->nextRowid; r++) {
        if (!t->live[r]) continue;
        if (where && !evalCond(*t, *where, r)) continue;
        all.push_back(Cand{distOf(r, orderQ), t->pkOf[r], r});
      }
      auto cmp = [&](const Cand& a, const Cand& b) {
        if (a.d != b.d) return orderDesc ? (a.d > b.d) : (a.d < b.d);
        return a.pk < b.pk;
      };
      if ((int64_t)all.size() > k) {
        std::partial_sort(all.begin(), all.begin() + (size_t)k, all.end(), cmp);
        all.resize((size_t)k);
      } else {
        std::sort(all.begin(), all.end(), cmp);
      }
      results = std::move(all);
      usedIndex = true;  // results already collected
    }
    if (!usedIndex) {
      const auto* cachedRows = where ? topkFilterRows(*t, *where) : nullptr;
      const uint64_t nrows = cachedRows ? cachedRows->size() : t->nextRowid;
      const int kd = orderDesc ? 1 : 0;
      std::vector<uint8_t> qsketch;
      const bool useSketch =
          !orderDesc && t->vecs && t->vecs->sketchReady(t->nextRowid) &&
          t->vecs->sketchQuery(orderQ.data(), qsketch);
      const uint8_t* skbase = useSketch ? t->vecs->sketchBase() : nullptr;
      const uint8_t* sksafe = useSketch ? t->vecs->sketchSafe() : nullptr;
      // One chunk of the scan: rows [lo, hi) -> best k, ordered best first.
      auto runChunk = [&](uint64_t lo, uint64_t hi, CandOut* out) -> int {
        if (lo >= hi) return 0;
        if (cachedRows) {
          if (useSketch)
            return scanTopKGatherQuant(vbase, dim, skbase, sksafe, cachedRows->data() + lo,
                                       hi - lo, t->pkOf.data(), orderQ.data(),
                                       qsketch.data(), (int)k, kd, out);
          return scanTopKGather(vbase, dim, cachedRows->data() + lo, hi - lo,
                                t->pkOf.data(), orderQ.data(), (int)k, kd, out);
        }
        if (where) {
          std::vector<uint32_t> hits;
          if (where->kind == CompiledCond::CMP && where->op == Cond::EQ &&
              where->type == Type::VARCHAR && t->cache[where->col].hashes.size() >= hi) {
            const auto& hashes = t->cache[where->col].hashes;
            for (uint64_t r = lo; r < hi; r++) {
              if (hashes[r] == where->shash && t->live[r] && evalCond(*t, *where, r))
                hits.push_back((uint32_t)r);
            }
          } else {
            for (uint64_t r = lo; r < hi; r++) {
              if (t->live[r] && evalCond(*t, *where, r)) hits.push_back((uint32_t)r);
            }
          }
          if (useSketch)
            return scanTopKGatherQuant(vbase, dim, skbase, sksafe, hits.data(), hits.size(),
                                       t->pkOf.data(), orderQ.data(), qsketch.data(), (int)k,
                                       kd, out);
          return scanTopKGather(vbase, dim, hits.data(), hits.size(), t->pkOf.data(),
                                orderQ.data(), (int)k, kd, out);
        }
        int got = useSketch
                      ? scanTopKRangeQuant(vbase + lo * (size_t)dim, dim,
                                           skbase + lo * (size_t)dim, sksafe + lo,
                                           t->live.data() + lo, t->pkOf.data() + lo, hi - lo,
                                           orderQ.data(), qsketch.data(), (int)k, kd, out)
                      : scanTopKRange(vbase + lo * (size_t)dim, dim, t->live.data() + lo,
                                      t->pkOf.data() + lo, hi - lo, orderQ.data(), (int)k,
                                      kd, out);
        for (int i = 0; i < got; i++) out[i].rowid += (uint32_t)lo;
        return got;
      };

      // Parallel scan for tables large enough to pay for the fan-out. Each
      // worker keeps its own top-k and the caller merges them, so the result
      // set is identical to the single-threaded scan.
      double work = (double)t->nextRowid * (double)std::max(dim, 1);
      static const double parMin = getenv("MVDB_PAR_MIN") ? atof(getenv("MVDB_PAR_MIN")) : 5.0e5;
      int threads = 1;
      if (work >= parMin && k <= 256) {
        threads = ThreadPool::get().size();
        int maxChunks = (int)(work / 3.0e5);
        if (maxChunks < 1) maxChunks = 1;
        if (threads > maxChunks) threads = maxChunks;
        if (nrows < (uint64_t)threads) threads = (int)nrows;
      }
      if (threads <= 1) {
        std::vector<CandOut> buf((size_t)k);
        int got = runChunk(0, nrows, buf.data());
        for (int i = 0; i < got; i++)
          results.push_back(Cand{(double)buf[i].d, buf[i].pk, buf[i].rowid});
      } else {
        std::vector<std::vector<CandOut>> parts((size_t)threads);
        std::vector<int> counts((size_t)threads, 0);
        std::vector<const CandOut*> ptrs((size_t)threads, nullptr);
        uint64_t chunk = (nrows + (uint64_t)threads - 1) / (uint64_t)threads;
        ThreadPool::get().parallelFor(threads, [&](int ci) {
          parts[ci].resize((size_t)k);
          uint64_t lo = (uint64_t)ci * chunk;
          uint64_t hi = std::min<uint64_t>(nrows, lo + chunk);
          counts[ci] = runChunk(lo, hi, parts[ci].data());
        });
        for (int i = 0; i < threads; i++) ptrs[i] = parts[i].data();
        std::vector<CandOut> merged((size_t)k);
        int got = mergeTopK(ptrs.data(), counts.data(), threads, (int)k, kd, merged.data());
        for (int i = 0; i < got; i++)
          results.push_back(Cand{(double)merged[i].d, merged[i].pk, merged[i].rowid});
      }
    }
    std::sort(results.begin(), results.end(), [&](const Cand& a, const Cand& b) {
      if (a.d != b.d) return orderDesc ? (a.d > b.d) : (a.d < b.d);
      return a.pk < b.pk;
    });
    if ((int64_t)results.size() > k) results.resize((size_t)k);
    size_t skip = (size_t)std::min<int64_t>(off, (int64_t)results.size());
    int64_t emitted = 0;
    std::vector<bool> reuseOrderDist(items.size(), false);
    bool needRow = false;
    for (size_t i = 0; i < items.size(); i++) {
      const auto& it = items[i];
      if (it.dist && searchMode != "ann" && it.distCol == orderDistCol && it.q == orderQ)
        reuseOrderDist[i] = true;
      if (!it.dist && !it.isConst && it.col != sc.pk) needRow = true;
    }
    std::vector<Value> row;
    for (size_t ri = skip; ri < results.size(); ri++) {
      if (emitted++ >= lim) break;
      const Cand& c = results[ri];
      if (needRow) decodeRowAt(*t, c.rowid, row);
      std::vector<ResultCell> out;
      out.reserve(items.size());
      for (size_t i = 0; i < items.size(); i++) {
        const auto& it = items[i];
        if (it.dist) {
          double d2 = reuseOrderDist[i] ? c.d : distOf(c.rowid, it.q);
          out.emplace_back(it.squared ? d2 : std::sqrt(d2));
        } else if (it.isConst) {
          out.push_back(resultCell(it.cval));
        } else if (it.col == sc.pk) {
          out.emplace_back(c.pk);
        } else {
          out.push_back(resultCell(row[it.col]));
        }
      }
      rs.rows.push_back(std::move(out));
    }
    return;
  }

  // --- general path
  struct SortRow {
    uint64_t rowid;
    int64_t pk;
    double d;
    int64_t i;
    std::string s;
  };
  std::vector<SortRow> rows;
  bool needKey = st.select.hasOrder;
  if (orderCol >= 0) ensureCaches(*t, orderCol);
  for (uint64_t r = 0; r < t->nextRowid; r++) {
    if (!t->live[r]) continue;
    if (!passes(r)) continue;
    SortRow sr;
    sr.rowid = r;
    sr.pk = t->pkOf[r];
    sr.d = 0;
    sr.i = 0;
    if (needKey) {
      if (orderDist) {
        sr.d = distOf(r, orderQ);
      } else if (orderCol >= 0) {
        const ColCache& cc = t->cache[orderCol];
        if (sc.cols[orderCol].type == Type::BIGINT && cc.ready && cc.ints.size() > r) {
          sr.i = cc.ints[r];
        } else if (sc.cols[orderCol].type == Type::VARCHAR && cc.ready && cc.lens.size() > r) {
          sr.s.assign(cc.arena.data() + cc.offs[r], cc.lens[r]);
        } else {
          std::vector<Value> row;
          decodeRowAt(*t, r, row);
          if (sc.cols[orderCol].type == Type::BIGINT) sr.i = row[orderCol].i;
          else if (sc.cols[orderCol].type == Type::VARCHAR) sr.s = row[orderCol].s;
          else throwErr(1235, "42000", "ORDER BY on VECTOR column is not supported");
        }
      }
    }
    rows.push_back(std::move(sr));
    if (!needKey && lim > 0 && (int64_t)rows.size() >= lim + off) break;
  }
  if (needKey) {
    std::sort(rows.begin(), rows.end(), [&](const SortRow& a, const SortRow& b) {
      int c = 0;
      if (orderDist) c = (a.d < b.d) ? -1 : (a.d > b.d ? 1 : 0);
      else if (orderCol >= 0) {
        if (sc.cols[orderCol].type == Type::BIGINT) c = (a.i < b.i) ? -1 : (a.i > b.i ? 1 : 0);
        else c = a.s.compare(b.s);
      }
      if (orderDesc && c != 0) c = -c;
      if (c != 0) return c < 0;
      return a.pk < b.pk;
    });
    if (off > 0) {
      if ((int64_t)rows.size() > off) rows.erase(rows.begin(), rows.begin() + (size_t)off);
      else rows.clear();
    }
    if (lim >= 0 && (int64_t)rows.size() > lim) rows.resize((size_t)lim);
  }
  std::vector<Value> outRow;
  for (auto& sr : rows) {
    decodeRowAt(*t, sr.rowid, outRow);
    std::vector<ResultCell> out;
    out.reserve(items.size());
    for (auto& it : items) {
      if (it.dist) {
        double d2 = distOf(sr.rowid, it.q);
        out.emplace_back(it.squared ? d2 : std::sqrt(d2));
      } else if (it.isConst) {
        out.push_back(resultCell(it.cval));
      } else {
        out.push_back(resultCell(outRow[it.col]));
      }
    }
    rs.rows.push_back(std::move(out));
  }
}

// ------------------------------------------------------------ misc commands ---

void Database::execShowTables(ResultSet& rs) {
  rs.hasResult = true;
  rs.headers.push_back("Tables");
  std::vector<std::string> names;
  for (auto& t : tables_) names.push_back(t->schema.name);
  std::sort(names.begin(), names.end());
  for (auto& n : names) rs.rows.push_back({n});
}

void Database::execDescribe(const Stmt& st, ResultSet& rs) {
  Table* t = getTableOrThrow(st.tableName);
  rs.hasResult = true;
  rs.headers = {"Field", "Type", "Key"};
  for (size_t i = 0; i < t->schema.cols.size(); i++) {
    const Column& c = t->schema.cols[i];
    std::string type;
    if (c.type == Type::BIGINT) type = "bigint";
    else if (c.type == Type::VARCHAR) type = "varchar(" + std::to_string(c.len) + ")";
    else type = "vector(" + std::to_string(c.len) + ")";
    rs.rows.push_back({c.name, type, std::string((int)i == t->schema.pk ? "PRI" : "")});
  }
}

void Database::compact() {
  if (inTxn_) throwErr(1235, "42000", "COMPACT is not allowed inside a transaction");
  std::string blob = "MVDBWAL1";
  auto addRec = [&](const std::string& body) {
    putU32(blob, (uint32_t)body.size());
    putU32(blob, crc32Of(body.data(), body.size()));
    blob += body;
  };
  for (auto& tp : tables_) {
    Table& t = *tp;
    std::string cb;
    encodeCreateTable(t, cb);
    addRec(cb);
    for (uint64_t r = 0; r < t.nextRowid; r++) {
      if (!t.live[r]) continue;
      std::string body;
      body.push_back((char)3);
      putVarint(body, t.id);
      putVarint(body, r);
      for (int k = 0; k < 8; k++) body.push_back((char)((uint64_t)t.pkOf[r] >> (8 * k)));
      ensureMapped();
      body.append((const char*)map_.data() + t.off[r], t.len[r]);
      addRec(body);
    }
  }
  {
    std::string c;
    c.push_back((char)6);
    addRec(c);
  }
  std::string tmp = walPath_ + ".compact";
  {
    File f;
    if (!f.openWrite(tmp)) throwErr(1105, "HY000", "cannot write compaction file");
    if (!f.writeAt(0, blob.data(), blob.size())) throwErr(1105, "HY000", "compaction write failed");
    f.sync();
    f.close();
  }
  map_.unmap();
  mappedLen_ = 0;
  wal_.close();
#ifdef _WIN32
  _unlink(walPath_.c_str());
#endif
  if (rename(tmp.c_str(), walPath_.c_str()) != 0)
    throwErr(1105, "HY000", "cannot install compacted WAL");
  tables_.clear();
  undo_.clear();
  inTxn_ = false;
  nextTableId_ = 1;
  openWal();
  recover();
}

// ------------------------------------------------------------------ execute ---

void Database::execute(const std::string& sql, ResultSet& rs, const std::string& searchMode) {
  static const bool timing = getenv("MVDB_TIMING") != nullptr;
  int64_t t0 = timing ? nowMicros() : 0;
  Stmt st = parseStatement(sql);
  int64_t t1 = timing ? nowMicros() : 0;
  switch (st.kind) {
    case Stmt::EMPTY:
      return;
    case Stmt::BEGIN_TX:
      beginTxn();
      return;
    case Stmt::COMMIT_TX:
      if (inTxn_) commitTxn();
      return;
    case Stmt::ROLLBACK_TX:
      rollbackTxn();
      return;
    case Stmt::SET: {
      // The search mode is a hint only: the exact scan is always correct, so an
      // unrecognised SET is ignored instead of failing the statement.
      std::string v = toLowerAscii(st.setValue);
      if (v == "ann" || v == "exact") {
        vectorMode_ = v;
      } else if (st.setKey == "ann") {
        vectorMode_ = (v == "1" || v == "on" || v == "true") ? "ann" : "exact";
      }
      return;
    }
    case Stmt::SHOW_TABLES:
      execShowTables(rs);
      return;
    case Stmt::DESCRIBE:
      execDescribe(st, rs);
      return;
    case Stmt::COMPACT:
      compact();
      return;
    case Stmt::EXIT:
      return;
    default:
      break;
  }

  // DDL implicitly commits an open transaction. Capture the rollback point
  // after that commit so a failed DDL cannot truncate the earlier commit.
  if (inTxn_ && (st.kind == Stmt::CREATE_TABLE || st.kind == Stmt::DROP_TABLE ||
                 st.kind == Stmt::CREATE_INDEX)) {
    commitTxn();
  }
  bool autoCommit = !inTxn_;
  uint64_t walMark = wal_.size();
  size_t undoMark = undo_.size();
  try {
    switch (st.kind) {
      case Stmt::CREATE_TABLE:
      case Stmt::DROP_TABLE:
      case Stmt::CREATE_INDEX:
        if (st.kind == Stmt::CREATE_TABLE) execCreateTable(st, rs);
        else if (st.kind == Stmt::DROP_TABLE) execDropTable(st, rs);
        else execCreateIndex(st, rs);
        break;
      case Stmt::INSERT: execInsert(st, rs); break;
      case Stmt::SELECT: execSelect(st, rs, searchMode.empty() ? vectorMode_ : searchMode); break;
      case Stmt::UPDATE: execUpdate(st, rs); break;
      case Stmt::DELETE_: execDelete(st, rs); break;
      default: throwErr(1235, "42000", "unsupported statement");
    }
  } catch (...) {
    undoTo(undoMark);
    truncateWal(walMark);
    ensureMapped();
    throw;
  }
  if (autoCommit && wal_.size() > walMark) {
    if (st.kind == Stmt::INSERT || st.kind == Stmt::UPDATE || st.kind == Stmt::DELETE_) {
      syncWal();  // appendRecords already included the commit footer
    } else {
      appendCommitAndSync();
    }
    undo_.clear();
  }
  if (timing) {
    int64_t t2 = nowMicros();
    fprintf(stderr, "TIMING parse=%lld exec=%lld\n", (long long)(t1 - t0), (long long)(t2 - t1));
  }
}

void Database::close() {
  dumpInsertTiming();
  if (inTxn_) {
    try {
      rollbackTxn();
    } catch (...) {
    }
  }
  if (wal_.valid()) {
    wal_.sync();
  }
  map_.unmap();
  mappedLen_ = 0;
  wal_.close();
  tables_.clear();
}

}  // namespace mvdb
