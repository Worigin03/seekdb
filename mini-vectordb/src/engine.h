// Storage engine: WAL, tables, vector store, HNSW index, SQL execution.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <variant>

#include "platform.h"
#include "types.h"
#include "sql.h"

namespace mvdb {

// ------------------------------------------------------------- vector store ---

class VecStore {
 public:
  VecStore() = default;
  ~VecStore();
  VecStore(const VecStore&) = delete;
  VecStore& operator=(const VecStore&) = delete;

  void init(int dim, const std::string& path, size_t memBudgetBytes);
  void reset();
  void ensure(uint64_t rowid);
  void set(uint64_t rowid, const float* v);
  float* ptr(uint64_t rowid);
  const float* ptr(uint64_t rowid) const;
  // Contiguous base pointer; valid while no insert/update grows the store.
  const float* base() const {
    return disk_ ? (const float*)map_.data() : mem_;
  }
  int dim() const { return dim_; }
  uint64_t capacityRows() const { return capRows_; }
  // Per-row 8-bit sketches are only a conservative pruning aid; exact float64
  // distances are still computed for every row that can enter the result.
  const uint8_t* sketchBase() const { return sketch_.data(); }
  const uint8_t* sketchSafe() const { return sketchSafe_.data(); }
  bool sketchReady(uint64_t rows) const {
    return sketchEnabled_ && dim_ > 0 && sketchSafe_.size() >= rows &&
           sketch_.size() >= (size_t)rows * (size_t)dim_;
  }
  bool sketchQuery(const float* q, std::vector<uint8_t>& out) const;

 private:
  void switchToDisk();
  void growDisk(uint64_t needRows);
  bool growMem(uint64_t needRows);

  int dim_ = 0;
  uint64_t slot_ = 0;      // bytes per row
  uint64_t capRows_ = 0;
  size_t budget_ = 0;
  bool disk_ = false;
  float* mem_ = nullptr;   // owned; grown with realloc (no zero-fill, may extend in place)
  bool sketchEnabled_ = false;
  std::vector<uint8_t> sketch_;
  std::vector<uint8_t> sketchSafe_;
  File file_;
  MappedFile map_;
  uint64_t fileCap_ = 0;
  std::string path_;
};

// --------------------------------------------------------------- pk index ---

class PkIndex {
 public:
  void init(size_t minCap = 1024);
  void clear();
  bool get(int64_t key, uint64_t& val) const;
  void put(int64_t key, uint64_t val);
  bool erase(int64_t key);
  size_t size() const { return count_; }

 private:
  void grow();
  std::vector<int64_t> keys_;
  std::vector<uint64_t> vals_;
  std::vector<uint8_t> used_;
  size_t mask_ = 0;
  size_t count_ = 0;
};

// -------------------------------------------------------------------- hnsw ---

class Table;

class Hnsw {
 public:
  void build(Table* t);
  void addRow(uint64_t rowid);
  void removeRow(uint64_t rowid);
  bool ready() const { return entry_ >= 0; }
  size_t nodeCount() const { return nodes_.size(); }

  // Returns up to k rowids (best first) accepted by `accept`.
  void search(const float* q, int k, int ef,
              const std::function<bool(uint64_t)>& accept,
              std::vector<std::pair<float, uint64_t>>& out);

 private:
  struct Node {
    int level = 0;
    std::vector<std::vector<uint32_t>> links;
  };

  float dist(const float* a, const float* b) const;
  float nodeDist(const float* q, uint32_t node) const;
  uint64_t rowidOfNode(uint32_t node) const { return rowidOf_[node]; }
  int greedyStep(const float* q, int ep, int level) const;
  void searchLayer(const float* q, int ep, int ef, int level,
                   std::vector<std::pair<float, uint32_t>>& out,
                   std::vector<uint32_t>& visited) const;
  void selectNeighbors(const float* q, std::vector<std::pair<float, uint32_t>>& cands,
                       int m, std::vector<uint32_t>& out) const;

  Table* table_ = nullptr;
  int dim_ = 0;
  int M_ = 16;
  int M0_ = 32;
  int efConstruction_ = 160;
  std::vector<Node> nodes_;
  std::vector<int32_t> nodeOf_;   // rowid -> node id
  std::vector<uint64_t> rowidOf_; // node id -> rowid
  std::vector<uint8_t> dead_;
  int entry_ = -1;
  int maxLevel_ = -1;
  mutable std::vector<uint32_t> visit_;
  mutable uint32_t visitStamp_ = 1;
};

// ------------------------------------------------------------- predicates ---

struct CompiledCond {
  enum Kind { AND, OR, NOT, CMP } kind = CMP;
  int col = -1;
  Cond::Op op = Cond::EQ;
  Type type = Type::BIGINT;
  int64_t ival = 0;
  uint64_t shash = 0;
  std::string sval;
  std::vector<std::unique_ptr<CompiledCond>> kids;
};

// ------------------------------------------------------------------- table ---

struct ColCache {
  bool ready = false;
  std::vector<int64_t> ints;
  std::vector<uint64_t> hashes;
  std::vector<uint32_t> offs;
  std::vector<uint32_t> lens;
  std::string arena;
};

struct TopKFilter {
  int column;
  std::string value;
  bool usable = true;
  std::vector<uint32_t> rowids;
};

struct Table {
  uint32_t id = 0;
  Schema schema;
  std::vector<uint64_t> off;   // payload offset in WAL, per rowid
  std::vector<uint32_t> len;   // payload length
  std::vector<uint8_t> live;   // 1 = visible
  std::vector<int64_t> pkOf;   // primary key per rowid
  PkIndex pk;
  uint64_t nextRowid = 0;
  std::unique_ptr<VecStore> vecs;
  std::unique_ptr<Hnsw> hnsw;

  // Lazy per-column filter caches (indexed by rowid).
  std::vector<ColCache> cache;
  std::vector<TopKFilter> topkFilters;

  int vecCol() const { return schema.vec; }
  int pkCol() const { return schema.pk; }
};

// ---------------------------------------------------------------- database ---

class Database {
 public:
  using ResultCell = std::variant<int64_t, std::string, std::vector<float>, double>;

  struct ResultSet {
    bool hasResult = false;
    // Rows touched by a data-modifying statement (0 for DDL and queries).
    int64_t affected = 0;
    std::vector<std::string> headers;
    std::vector<std::vector<ResultCell>> rows;
  };

  Database() = default;
  ~Database();

  void open(const std::string& dir);
  void close();
  void execute(const std::string& sql, ResultSet& rs, const std::string& searchMode = {});
  void dumpInsertTiming();

  bool inTransaction() const { return inTxn_; }
  const std::string& vectorMode() const { return vectorMode_; }

 private:
  // ---- WAL
  void openWal();
  void recover();
  uint64_t appendRecord(const std::string& body);
  void appendRecords(const std::vector<std::string>& bodies, std::vector<uint64_t>& offs);
  void appendCommit();
  void appendCommitAndSync();
  void syncWal();
  void truncateWal(uint64_t off);
  const char* payloadPtr(const Table& t, uint64_t rowid);
  void ensureMapped();

  // ---- record encoding / applying
  void encodeCreateTable(const Table& t, std::string& out);
  void applyRecord(const char* body, size_t len, uint64_t off);
  void applyInsert(uint32_t tableId, uint64_t rowid, int64_t pk, const char* payload,
                   uint32_t plen, uint64_t off);
  void applyUpdate(uint32_t tableId, uint64_t oldRowid, uint64_t newRowid, int64_t pk,
                   const char* payload, uint32_t plen, uint64_t off);
  void applyDelete(uint32_t tableId, uint64_t rowid, int64_t pk);
  void applyCreateTable(const char* body, size_t len);
  void applyDropTable(uint32_t tableId);

  // ---- helpers
  Table* findTable(const std::string& name);
  Table* getTableOrThrow(const std::string& name);
  void ensureCaches(Table& t, int col);
  std::unique_ptr<CompiledCond> compileCond(Table& t, const Cond& c);
  bool evalCond(Table& t, const CompiledCond& c, uint64_t rowid);
  const std::vector<uint32_t>* topkFilterRows(Table& t, const CompiledCond& condition);
  void decodeRowAt(Table& t, uint64_t rowid, std::vector<Value>& out);

  // ---- statements
  void execCreateTable(const Stmt& st, ResultSet& rs);
  void execDropTable(const Stmt& st, ResultSet& rs);
  void execCreateIndex(const Stmt& st, ResultSet& rs);
  void execInsert(const Stmt& st, ResultSet& rs);
  void execSelect(const Stmt& st, ResultSet& rs, const std::string& searchMode);
  void execUpdate(const Stmt& st, ResultSet& rs);
  void execDelete(const Stmt& st, ResultSet& rs);
  void execShowTables(ResultSet& rs);
  void execDescribe(const Stmt& st, ResultSet& rs);
  void compact();

  // transaction plumbing
  struct Undo {
    uint8_t kind;  // 0 = insert (undo = delete), 1 = delete (undo = restore)
    uint32_t tableId;
    uint64_t rowid;
    int64_t pk;
  };
  void beginTxn();
  void commitTxn();
  void rollbackTxn();
  void undoTo(size_t mark);
  void undoAll();
  void applyPendingInsert(Table& t, uint64_t rowid, int64_t pk, const char* payload,
                          uint32_t plen, uint64_t off, bool recordUndo);
  void applyPendingDelete(Table& t, uint64_t rowid, int64_t pk, bool recordUndo);

  std::string dir_;
  std::string walPath_;
  File wal_;
  MappedFile map_;
  uint64_t mappedLen_ = 0;
  uint64_t groupStart_ = 0;

  std::vector<std::unique_ptr<Table>> tables_;
  uint32_t nextTableId_ = 1;

  bool inTxn_ = false;
  std::vector<Undo> undo_;
  std::string vectorMode_ = "exact";
  size_t vecMemBudget_ = 128ull << 20;
};

}  // namespace mvdb
