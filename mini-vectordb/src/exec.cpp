#include "engine.h"
#include "simd.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>
#include <cstdio>
#include <set>

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
      out = Value::ofInt(e.i);
      break;
    }
    case Type::VARCHAR: {
      std::string s;
      if (e.kind == Expr::STR_LIT) s = e.s;
      else if (e.kind == Expr::INT_LIT) s = std::to_string(e.i);
      else
        throwErr(1366, "HY000", "Incorrect string value for column '" + c.name + "' at row " +
                                     std::to_string(rowNo));
      checkVarchar(c, s, rowNo);
      out = Value::ofStr(std::move(s));
      break;
    }
    case Type::VECTOR: {
      if (e.kind != Expr::VEC_LIT)
        throwErr(1366, "HY000", "Incorrect vector value for column '" + c.name + "' at row " +
                                     std::to_string(rowNo));
      std::vector<float> v;
      std::string err;
      if (!parseVectorLiteral(e.vecText, c.len, v, err))
        throwErr(1366, "HY000", "Incorrect vector value for column '" + c.name + "' at row " +
                                     std::to_string(rowNo) + ": " + err);
      out = Value::ofVec(std::move(v));
      break;
    }
  }
}

void Database::execInsert(const Stmt& st, ResultSet& rs) {
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
  std::vector<std::vector<Value>> rows;
  rows.reserve(st.insert.rows.size());
  std::set<int64_t> seen;
  for (size_t r = 0; r < st.insert.rows.size(); r++) {
    const auto& src = st.insert.rows[r];
    if (src.size() != colMap.size())
      throwErr(1136, "21S01", "Column count doesn't match value count at row " +
                                   std::to_string(r + 1));
    std::vector<Value> row(sc.cols.size());
    for (size_t i = 0; i < sc.cols.size(); i++) row[i].type = sc.cols[i].type;
    std::vector<char> filled(sc.cols.size(), 0);
    for (size_t i = 0; i < src.size(); i++) {
      coerceValue(sc.cols[colMap[i]], *src[i], row[colMap[i]], (int)r + 1);
      filled[colMap[i]] = 1;
    }
    for (size_t i = 0; i < sc.cols.size(); i++) {
      if (filled[i] || (int)i == sc.pk) continue;
      if (sc.cols[i].type != Type::VARCHAR)
        throwErr(1364, "HY000", "Field '" + sc.cols[i].name + "' doesn't have a default value");
      row[i] = Value::ofStr("");
    }
    int64_t pk = row[sc.pk].i;
    uint64_t existing = 0;
    if (t->pk.get(pk, existing))
      throwErr(1062, "23000", "Duplicate entry '" + std::to_string(pk) + "' for key 'PRIMARY'");
    if (seen.count(pk))
      throwErr(1062, "23000", "Duplicate entry '" + std::to_string(pk) + "' for key 'PRIMARY'");
    seen.insert(pk);
    rows.push_back(std::move(row));
  }

  std::vector<std::string> bodies;
  std::vector<std::string> payloads;
  std::vector<uint64_t> rowids;
  std::vector<int64_t> pks;
  bodies.reserve(rows.size());
  payloads.reserve(rows.size());
  rowids.reserve(rows.size());
  pks.reserve(rows.size());
  for (auto& row : rows) {
    std::string payload;
    encodeRow(sc, row, payload);
    uint64_t rowid = t->nextRowid++;
    int64_t pk = row[sc.pk].i;
    std::string body;
    body.push_back((char)3);
    putVarint(body, t->id);
    putVarint(body, rowid);
    for (int k = 0; k < 8; k++) body.push_back((char)((uint64_t)pk >> (8 * k)));
    body.append(payload);
    bodies.push_back(std::move(body));
    payloads.push_back(std::move(payload));
    rowids.push_back(rowid);
    pks.push_back(pk);
  }
  std::vector<uint64_t> offs;
  appendRecords(bodies, offs);
  rs.affected = (int64_t)rowids.size();
  for (size_t i = 0; i < rowids.size(); i++) {
    // payload sits at the tail of the record body
    uint64_t payloadOff = offs[i] + (bodies[i].size() - payloads[i].size());
    applyPendingInsert(*t, rowids[i], pks[i], payloads[i].data(), (uint32_t)payloads[i].size(),
                       payloadOff, true);
  }
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

void Database::execSelect(const Stmt& st, ResultSet& rs) {
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
        if (!parseVectorLiteral(e->vecText, dim, it.q, err))
          throwErr(1366, "HY000", "Incorrect vector value: " + err);
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
      if (!parseVectorLiteral(oe->vecText, sc.cols[orderDistCol].len, orderQ, err))
        throwErr(1366, "HY000", "Incorrect vector value: " + err);
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
  // Output distances are recomputed in float64 for the (few) returned rows so
  // they agree with a double-precision reference implementation.
  auto distOf = [&](uint64_t r, const std::vector<float>& q) -> double {
    return l2sqRef(vbase + r * (size_t)dim, q.data(), dim);
  };

  int64_t lim = st.select.hasLimit ? st.select.limit : -1;
  if (lim == 0) return;  // LIMIT 0 never returns rows

  // --- vector top-k fast path
  if (orderDist && lim > 0 && dim > 0) {
    // Never ask the scan kernel for more slots than there are rows: LIMIT may
    // legally be far larger than the table.
    int64_t k = std::min<int64_t>(lim, (int64_t)t->nextRowid);
    if (k <= 0) return;
    std::vector<Cand> results;
    bool usedIndex = false;
    if (vectorMode_ == "ann" && t->hnsw) {
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
    if (!usedIndex) {
      std::vector<CandOut> buf((size_t)k);
      int got = 0;
      if (where) {
        // Filter first: only surviving rows pay for a distance computation.
        std::vector<uint32_t> hits;
        hits.reserve(1024);
        for (uint64_t r = 0; r < t->nextRowid; r++) {
          if (!t->live[r]) continue;
          if (!evalCond(*t, *where, r)) continue;
          hits.push_back((uint32_t)r);
        }
        got = scanTopKGather(vbase, dim, hits.data(), hits.size(), t->pkOf.data(),
                             orderQ.data(), (int)k, orderDesc ? 1 : 0, buf.data());
      } else {
        got = scanTopKRange(vbase, dim, t->live.data(), t->pkOf.data(), t->nextRowid,
                            orderQ.data(), (int)k, orderDesc ? 1 : 0, buf.data());
      }
      for (int i = 0; i < got; i++)
        results.push_back(Cand{(double)buf[i].d, buf[i].pk, buf[i].rowid});
    }
    std::sort(results.begin(), results.end(), [&](const Cand& a, const Cand& b) {
      if (a.d != b.d) return orderDesc ? (a.d > b.d) : (a.d < b.d);
      return a.pk < b.pk;
    });
    if ((int64_t)results.size() > k) results.resize((size_t)k);
    for (auto& c : results) {
      std::vector<Value> row;
      decodeRowAt(*t, c.rowid, row);
      std::vector<std::string> out;
      for (auto& it : items) {
        if (it.dist) {
          double d2 = distOf(c.rowid, it.q);
          out.push_back(formatDouble(it.squared ? d2 : std::sqrt(d2)));
        } else if (it.isConst) {
          out.push_back(formatValue(it.cval));
        } else {
          out.push_back(formatValue(row[it.col]));
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
    if (!needKey && lim > 0 && (int64_t)rows.size() >= lim) break;
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
    if (lim >= 0 && (int64_t)rows.size() > lim) rows.resize((size_t)lim);
  }
  for (auto& sr : rows) {
    std::vector<Value> row;
    decodeRowAt(*t, sr.rowid, row);
    std::vector<std::string> out;
    for (auto& it : items) {
      if (it.dist) {
        double d2 = distOf(sr.rowid, it.q);
        out.push_back(formatDouble(it.squared ? d2 : std::sqrt(d2)));
      } else if (it.isConst) {
        out.push_back(formatValue(it.cval));
      } else {
        out.push_back(formatValue(row[it.col]));
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
    rs.rows.push_back({c.name, type, (int)i == t->schema.pk ? "PRI" : ""});
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

void Database::execute(const std::string& sql, ResultSet& rs) {
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

  bool autoCommit = !inTxn_;
  uint64_t walMark = wal_.size();
  size_t undoMark = undo_.size();
  try {
    switch (st.kind) {
      case Stmt::CREATE_TABLE:
      case Stmt::DROP_TABLE:
      case Stmt::CREATE_INDEX:
        // MySQL semantics: DDL commits the open transaction before it runs.
        if (inTxn_) commitTxn();
        if (st.kind == Stmt::CREATE_TABLE) execCreateTable(st, rs);
        else if (st.kind == Stmt::DROP_TABLE) execDropTable(st, rs);
        else execCreateIndex(st, rs);
        break;
      case Stmt::INSERT: execInsert(st, rs); break;
      case Stmt::SELECT: execSelect(st, rs); break;
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
    appendCommitAndSync();
    undo_.clear();
  }
  if (timing) {
    int64_t t2 = nowMicros();
    fprintf(stderr, "TIMING parse=%lld exec=%lld\n", (long long)(t1 - t0), (long long)(t2 - t1));
  }
}

void Database::close() {
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
