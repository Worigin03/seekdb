// SQL lexer, parser and statement splitter for the MiniVectorDB dialect.
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

#include "types.h"

namespace mvdb {

struct Expr {
  enum Kind { COLUMN, INT_LIT, STR_LIT, VEC_LIT, DIST, STAR } kind = COLUMN;
  std::string name;               // column name (COLUMN) or alias
  int64_t i = 0;                  // INT_LIT
  std::string s;                  // STR_LIT text
  std::vector<float> vec;         // VEC_LIT values, parsed once at parse time
  bool vecOk = false;             // vec holds a valid literal
  std::string vecText;            // raw text of vector literal (errors / echo)
  std::string distCol;            // DIST: vector column name
  std::string alias;              // explicit alias
  bool hasAlias = false;
  std::string raw;                // raw SQL text (used for column naming)
};

using ExprPtr = std::shared_ptr<Expr>;

struct Cond {
  enum Kind { AND, OR, NOT, CMP } kind = CMP;
  enum Op { EQ, NE, LT, LE, GT, GE } op = EQ;
  std::string col;
  Value val;
  std::vector<std::shared_ptr<Cond>> kids;
};

using CondPtr = std::shared_ptr<Cond>;

struct CreateTableStmt {
  std::string name;
  bool ifNotExists = false;
  std::vector<Column> cols;
  std::string pkCol;
  std::string pkConstraint;
  bool hasPkConstraint = false;
};

struct DropTableStmt {
  std::string name;
  bool ifExists = false;
};

struct CreateIndexStmt {
  std::string indexName;
  std::string table;
  std::string column;
  std::string algorithm = "hnsw";
};

struct InsertStmt {
  std::string table;
  std::vector<std::string> cols;  // empty means all columns in schema order
  std::vector<std::vector<ExprPtr>> rows;
};

struct SelectStmt {
  std::string table;
  bool star = false;
  std::vector<ExprPtr> items;
  CondPtr where;
  bool hasOrder = false;
  ExprPtr orderExpr;
  int orderIdx = -1;      // index into items
  bool orderDesc = false;
  int64_t offset = 0;
  bool hasLimit = false;
  int64_t limit = -1;
};

struct UpdateStmt {
  std::string table;
  std::vector<std::pair<std::string, ExprPtr>> sets;
  CondPtr where;
};

struct DeleteStmt {
  std::string table;
  CondPtr where;
};

struct Stmt {
  enum Kind {
    EMPTY,
    CREATE_TABLE,
    DROP_TABLE,
    CREATE_INDEX,
    INSERT,
    SELECT,
    UPDATE,
    DELETE_,
    BEGIN_TX,
    COMMIT_TX,
    ROLLBACK_TX,
    SET,
    SHOW_TABLES,
    DESCRIBE,
    COMPACT,
    EXIT
  } kind = EMPTY;

  CreateTableStmt createTable;
  DropTableStmt dropTable;
  CreateIndexStmt createIndex;
  InsertStmt insert;
  SelectStmt select;
  UpdateStmt update;
  DeleteStmt del;
  std::string setKey;
  std::string setValue;
  std::string tableName;  // DESCRIBE
};

// Parses one statement (without the trailing semicolon). Returns false and
// throws SqlError on syntax errors.
Stmt parseStatement(const std::string& sql);

// Streaming splitter: accumulate raw stdin text, pop complete statements.
// Returns true when `stmt` was filled.
bool extractStatement(std::string& buf, std::string& stmt);

// Incremental splitter. It remembers how far the current buffer has already
// been scanned, so feeding a 1 MB INSERT in 64 KB pipe reads costs one pass
// over the text instead of one pass per read.
class Splitter {
 public:
  void append(const char* p, size_t n) { buf_.append(p, n); }
  bool empty() const { return buf_.empty(); }
  const std::string& buffer() const { return buf_; }
  // Pops one statement: up to a top-level ';', or up to a newline when the
  // line cannot be a partial statement. Returns false when more input is
  // needed.
  bool next(std::string& stmt);
  // Emits whatever is buffered as a single statement (idle peer / EOF).
  bool takeRest(std::string& stmt);
  void clear() {
    buf_.clear();
    resetScan();
  }

 private:
  void resetScan() {
    scan_ = 0;
    inSingle_ = inDouble_ = inBacktick_ = inLineComment_ = inBlockComment_ = false;
  }
  std::string buf_;
  size_t scan_ = 0;
  bool inSingle_ = false, inDouble_ = false, inBacktick_ = false;
  bool inLineComment_ = false, inBlockComment_ = false;
};

std::string trimSpace(const std::string& s);

}  // namespace mvdb
