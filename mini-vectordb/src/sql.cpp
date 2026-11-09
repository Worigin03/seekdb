#include "sql.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace mvdb {

std::string trimSpace(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && isspace((unsigned char)s[a])) a++;
  while (b > a && isspace((unsigned char)s[b - 1])) b--;
  return s.substr(a, b - a);
}

// Parses an integer token, applying an optional unary minus. INT64_MIN is
// accepted (its magnitude does not fit in a positive int64), everything outside
// the int64 range is rejected.
static int64_t parseIntToken(const std::string& text, bool neg) {
  if (text.find('.') != std::string::npos || text.find('e') != std::string::npos ||
      text.find('E') != std::string::npos)
    throwErr(1064, "42000", "expected integer literal");
  errno = 0;
  char* endp = nullptr;
  unsigned long long mag = strtoull(text.c_str(), &endp, 10);
  if (errno == ERANGE || endp == text.c_str() || *endp != '\0')
    throwErr(1264, "22003", "integer out of range");
  const unsigned long long kMax = 9223372036854775807ull;  // INT64_MAX
  if (neg) {
    if (mag > kMax + 1) throwErr(1264, "22003", "integer out of range");
    if (mag == kMax + 1) return (int64_t)(-9223372036854775807ll - 1);
    return -(int64_t)mag;
  }
  if (mag > kMax) throwErr(1264, "22003", "integer out of range");
  return (int64_t)mag;
}

// ------------------------------------------------------------------ lexer ---

namespace {

enum TokKind { TK_END = 0, TK_IDENT, TK_NUM, TK_STR, TK_PUNCT };

struct Token {
  TokKind kind = TK_END;
  std::string text;
  size_t pos = 0;
  size_t end = 0;
};

class Lexer {
 public:
  explicit Lexer(const std::string& s) : s_(s) {}

  bool run(std::vector<Token>& out, std::string& err) {
    size_t i = 0;
    const size_t n = s_.size();
    while (i < n) {
      unsigned char c = (unsigned char)s_[i];
      if (isspace(c)) {
        i++;
        continue;
      }
      if (c == '-' && i + 1 < n && s_[i + 1] == '-') {
        while (i < n && s_[i] != '\n') i++;
        continue;
      }
      if (c == '#') {
        while (i < n && s_[i] != '\n') i++;
        continue;
      }
      if (c == '/' && i + 1 < n && s_[i + 1] == '*') {
        size_t j = s_.find("*/", i + 2);
        if (j == std::string::npos) {
          err = "unterminated comment";
          return false;
        }
        i = j + 2;
        continue;
      }
      size_t start = i;
      if (c == '\'' || c == '"') {
        char q = (char)c;
        i++;
        std::string val;
        bool closed = false;
        while (i < n) {
          char d = s_[i];
          if (d == '\\' && i + 1 < n) {
            char e = s_[i + 1];
            switch (e) {
              case 'n': val.push_back('\n'); break;
              case 't': val.push_back('\t'); break;
              case 'r': val.push_back('\r'); break;
              case '0': val.push_back('\0'); break;
              default: val.push_back(e); break;
            }
            i += 2;
            continue;
          }
          if (d == q) {
            if (i + 1 < n && s_[i + 1] == q) {
              val.push_back(q);
              i += 2;
              continue;
            }
            i++;
            closed = true;
            break;
          }
          val.push_back(d);
          i++;
        }
        if (!closed) {
          err = "unterminated string literal";
          return false;
        }
        Token t;
        t.kind = TK_STR;
        t.text = val;
        t.pos = start;
        t.end = i;
        out.push_back(t);
        continue;
      }
      if (c == '`') {
        i++;
        std::string val;
        bool closed = false;
        while (i < n) {
          if (s_[i] == '`') {
            if (i + 1 < n && s_[i + 1] == '`') {
              val.push_back('`');
              i += 2;
              continue;
            }
            i++;
            closed = true;
            break;
          }
          val.push_back(s_[i++]);
        }
        if (!closed) {
          err = "unterminated quoted identifier";
          return false;
        }
        Token t;
        t.kind = TK_IDENT;
        t.text = val;
        t.pos = start;
        t.end = i;
        out.push_back(t);
        continue;
      }
      if (isalpha(c) || c == '_' || c == '$') {
        i++;
        while (i < n && (isalnum((unsigned char)s_[i]) || s_[i] == '_' || s_[i] == '$')) i++;
        Token t;
        t.kind = TK_IDENT;
        t.text = s_.substr(start, i - start);
        t.pos = start;
        t.end = i;
        out.push_back(t);
        continue;
      }
      if (isdigit(c) || (c == '.' && i + 1 < n && isdigit((unsigned char)s_[i + 1]))) {
        i++;
        bool seenDot = (c == '.');
        while (i < n) {
          char d = s_[i];
          if (isdigit((unsigned char)d)) {
            i++;
          } else if (d == '.' && !seenDot) {
            seenDot = true;
            i++;
          } else if ((d == 'e' || d == 'E') && i + 1 < n &&
                     (isdigit((unsigned char)s_[i + 1]) || s_[i + 1] == '+' || s_[i + 1] == '-')) {
            i += 2;
          } else {
            break;
          }
        }
        Token t;
        t.kind = TK_NUM;
        t.text = s_.substr(start, i - start);
        t.pos = start;
        t.end = i;
        out.push_back(t);
        continue;
      }
      // punctuation
      {
        size_t len = 1;
        if ((c == '<' || c == '>' || c == '!') && i + 1 < n && s_[i + 1] == '=') len = 2;
        else if (c == '<' && i + 1 < n && s_[i + 1] == '>') len = 2;
        Token t;
        t.kind = TK_PUNCT;
        t.text = s_.substr(i, len);
        t.pos = start;
        t.end = i + len;
        out.push_back(t);
        i += len;
        continue;
      }
    }
    Token e;
    e.kind = TK_END;
    e.pos = n;
    e.end = n;
    out.push_back(e);
    return true;
  }

 private:
  const std::string& s_;
};

bool isKeyword(const Token& t, const char* kw) {
  return t.kind == TK_IDENT && iequals(t.text, kw);
}

}  // namespace

// ----------------------------------------------------------------- parser ---

namespace {

class Parser {
 public:
  Parser(const std::string& sql, std::vector<Token> toks)
      : sql_(sql), t_(std::move(toks)) {}

  bool atEnd() const { return cur().kind == TK_END; }
  const std::string& rest() const { return cur().text; }

  Stmt parse() {
    Stmt st;
    if (cur().kind == TK_END) return st;
    if (isKeyword(cur(), "CREATE")) return parseCreate();
    if (isKeyword(cur(), "DROP")) return parseDrop();
    if (isKeyword(cur(), "INSERT")) return parseInsert();
    if (isKeyword(cur(), "SELECT")) return parseSelect();
    if (isKeyword(cur(), "UPDATE")) return parseUpdate();
    if (isKeyword(cur(), "DELETE")) return parseDelete();
    if (isKeyword(cur(), "BEGIN")) {
      next();
      acceptKeyword("WORK");
      st.kind = Stmt::BEGIN_TX;
      return st;
    }
    if (isKeyword(cur(), "START")) {
      next();
      if (!acceptKeyword("TRANSACTION")) syntax("expected TRANSACTION");
      if (acceptKeyword("READ")) {
        if (!acceptKeyword("WRITE")) acceptKeyword("ONLY");
      }
      st.kind = Stmt::BEGIN_TX;
      return st;
    }
    if (isKeyword(cur(), "COMMIT")) {
      next();
      acceptKeyword("WORK");
      st.kind = Stmt::COMMIT_TX;
      return st;
    }
    if (isKeyword(cur(), "ROLLBACK")) {
      next();
      acceptKeyword("WORK");
      st.kind = Stmt::ROLLBACK_TX;
      return st;
    }
    if (isKeyword(cur(), "SET")) return parseSet();
    if (isKeyword(cur(), "SHOW")) {
      next();
      if (!acceptKeyword("TABLES")) syntax("expected TABLES");
      st.kind = Stmt::SHOW_TABLES;
      return st;
    }
    if (isKeyword(cur(), "DESC") || isKeyword(cur(), "DESCRIBE")) {
      next();
      st.kind = Stmt::DESCRIBE;
      st.tableName = expectIdent();
      return st;
    }
    if (isKeyword(cur(), "COMPACT") || isKeyword(cur(), "CHECKPOINT")) {
      next();
      st.kind = Stmt::COMPACT;
      return st;
    }
    if (isKeyword(cur(), "EXIT") || isKeyword(cur(), "QUIT")) {
      next();
      st.kind = Stmt::EXIT;
      return st;
    }
    syntax("unsupported statement");
  }

 private:
  // --- token helpers
  const Token& cur() const { return t_[i_]; }
  const Token& peek(size_t k = 1) const {
    size_t j = std::min(i_ + k, t_.size() - 1);
    return t_[j];
  }
  void next() {
    if (i_ + 1 < t_.size()) i_++;
  }
  bool isPunct(const char* p) const { return cur().kind == TK_PUNCT && cur().text == p; }
  bool acceptPunct(const char* p) {
    if (isPunct(p)) {
      next();
      return true;
    }
    return false;
  }
  void expectPunct(const char* p) {
    if (!acceptPunct(p)) syntax(std::string("expected '") + p + "'");
  }
  bool acceptKeyword(const char* kw) {
    if (isKeyword(cur(), kw)) {
      next();
      return true;
    }
    return false;
  }
  void expectKeyword(const char* kw) {
    if (!acceptKeyword(kw)) syntax(std::string("expected ") + kw);
  }
  std::string expectIdent() {
    if (cur().kind != TK_IDENT) syntax("expected identifier");
    std::string s = cur().text;
    next();
    return s;
  }
  [[noreturn]] void syntax(const std::string& msg) {
    throwErr(1064, "42000", msg);
  }

  int64_t expectInt() {
    bool neg = false;
    if (isPunct("-")) {
      neg = true;
      next();
    }
    if (cur().kind != TK_NUM) syntax("expected integer");
    int64_t v = parseIntToken(cur().text, neg);
    next();
    return v;
  }

  // --- expression helpers
  ExprPtr makeColumn(const std::string& name, size_t pos, size_t end) {
    auto e = std::make_shared<Expr>();
    e->kind = Expr::COLUMN;
    e->name = name;
    e->raw = sql_.substr(pos, end - pos);
    return e;
  }

  ExprPtr parseVectorExpr() {
    auto e = std::make_shared<Expr>();
    e->kind = Expr::VEC_LIT;
    size_t startPos = cur().pos;
    if (cur().kind == TK_STR) {
      std::string txt = cur().text;
      size_t endPos = cur().end;
      next();
      e->raw = sql_.substr(startPos, endPos - startPos);
      size_t k = 0;
      while (k < txt.size() && isspace((unsigned char)txt[k])) k++;
      if (k < txt.size() && txt[k] == '[') {
        e->kind = Expr::VEC_LIT;
        e->vecText = txt;
      } else {
        e->kind = Expr::STR_LIT;
        e->s = txt;
      }
      return e;
    }
    if (!isPunct("[")) syntax("expected vector literal");
    std::string text = "[";
    next();
    bool first = true;
    while (!isPunct("]")) {
      if (cur().kind == TK_END) syntax("unterminated vector literal");
      if (!first) {
        expectPunct(",");
        text += ",";
      }
      first = false;
      bool neg = false;
      if (isPunct("-")) {
        neg = true;
        next();
      }
      if (cur().kind != TK_NUM) syntax("expected number in vector literal");
      text += neg ? "-" : "";
      text += cur().text;
      next();
    }
    expectPunct("]");
    text += "]";
    e->vecText = text;
    e->raw = text;
    return e;
  }

  ExprPtr parseValueExpr() {
    if (cur().kind == TK_NUM) {
      auto e = std::make_shared<Expr>();
      e->kind = Expr::INT_LIT;
      e->i = parseIntToken(cur().text, false);
      e->raw = cur().text;
      next();
      return e;
    }
    if (isPunct("-")) {
      next();
      if (cur().kind != TK_NUM) syntax("expected number");
      auto e = std::make_shared<Expr>();
      e->kind = Expr::INT_LIT;
      e->i = parseIntToken(cur().text, true);
      e->raw = "-" + cur().text;
      next();
      return e;
    }
    if (cur().kind == TK_STR || isPunct("[")) return parseVectorExpr();
    if (cur().kind == TK_IDENT) {
      std::string nm = cur().text;
      size_t p = cur().pos, en = cur().end;
      next();
      return makeColumn(nm, p, en);
    }
    syntax("expected value");
  }

  ExprPtr parseExpr() {
    if (isKeyword(cur(), "L2_DISTANCE") || isKeyword(cur(), "L2_SQUARED_DISTANCE") ||
        isKeyword(cur(), "EUCLIDEAN_DISTANCE")) {
      bool squared = isKeyword(cur(), "L2_SQUARED_DISTANCE");
      size_t pos = cur().pos;
      next();
      expectPunct("(");
      auto e = std::make_shared<Expr>();
      e->kind = Expr::DIST;
      if (cur().kind != TK_IDENT) syntax("expected vector column");
      e->distCol = cur().text;
      next();
      expectPunct(",");
      ExprPtr v = parseVectorExpr();
      e->vecText = v->vecText;
      expectPunct(")");
      e->s = squared ? "sq" : "l2";
      e->raw = sql_.substr(pos, t_[i_ - 1].end - pos);
      return e;
    }
    if (cur().kind == TK_IDENT) {
      std::string nm = cur().text;
      size_t p = cur().pos, en = cur().end;
      next();
      return makeColumn(nm, p, en);
    }
    return parseValueExpr();
  }

  // --- conditions
  CondPtr parseOr() {
    CondPtr left = parseAnd();
    while (isKeyword(cur(), "OR")) {
      next();
      CondPtr right = parseAnd();
      auto n = std::make_shared<Cond>();
      n->kind = Cond::OR;
      n->kids.push_back(left);
      n->kids.push_back(right);
      left = n;
    }
    return left;
  }

  CondPtr parseAnd() {
    CondPtr left = parseNot();
    while (isKeyword(cur(), "AND")) {
      next();
      CondPtr right = parseNot();
      auto n = std::make_shared<Cond>();
      n->kind = Cond::AND;
      n->kids.push_back(left);
      n->kids.push_back(right);
      left = n;
    }
    return left;
  }

  CondPtr parseNot() {
    if (isKeyword(cur(), "NOT")) {
      next();
      auto n = std::make_shared<Cond>();
      n->kind = Cond::NOT;
      n->kids.push_back(parseNot());
      return n;
    }
    if (isPunct("(")) {
      next();
      CondPtr c = parseOr();
      expectPunct(")");
      return c;
    }
    return parseComparison();
  }

  CondPtr parseComparison() {
    ExprPtr a = parseExpr();
    Cond::Op op;
    if (isPunct("=")) op = Cond::EQ;
    else if (isPunct("!=") || isPunct("<>")) op = Cond::NE;
    else if (isPunct("<")) op = Cond::LT;
    else if (isPunct("<=")) op = Cond::LE;
    else if (isPunct(">")) op = Cond::GT;
    else if (isPunct(">=")) op = Cond::GE;
    else syntax("expected comparison operator");
    next();
    ExprPtr b = parseExpr();

    auto c = std::make_shared<Cond>();
    c->kind = Cond::CMP;
    c->op = op;
    ExprPtr colExpr, litExpr;
    if (a->kind == Expr::COLUMN && b->kind != Expr::COLUMN) {
      colExpr = a;
      litExpr = b;
    } else if (b->kind == Expr::COLUMN && a->kind != Expr::COLUMN) {
      colExpr = b;
      litExpr = a;
      switch (op) {
        case Cond::LT: c->op = Cond::GT; break;
        case Cond::LE: c->op = Cond::GE; break;
        case Cond::GT: c->op = Cond::LT; break;
        case Cond::GE: c->op = Cond::LE; break;
        default: break;
      }
    } else {
      syntax("comparison must reference exactly one column");
    }
    c->col = colExpr->name;
    switch (litExpr->kind) {
      case Expr::INT_LIT: c->val = Value::ofInt(litExpr->i); break;
      case Expr::STR_LIT: c->val = Value::ofStr(litExpr->s); break;
      case Expr::VEC_LIT: c->val = Value::ofVec({}); c->val.type = Type::VECTOR; c->val.s = litExpr->vecText; break;
      default: syntax("comparison value must be a literal");
    }
    return c;
  }

  // --- statements
  Stmt parseCreate() {
    next();  // CREATE
    if (acceptKeyword("VECTOR")) {
      if (!acceptKeyword("INDEX")) syntax("expected INDEX");
      return parseCreateIndex();
    }
    if (acceptKeyword("INDEX")) return parseCreateIndex();
    if (acceptKeyword("TABLE")) {
      Stmt st;
      st.kind = Stmt::CREATE_TABLE;
      if (acceptKeyword("IF")) {
        expectKeyword("NOT");
        expectKeyword("EXISTS");
        st.createTable.ifNotExists = true;
      }
      st.createTable.name = expectIdent();
      expectPunct("(");
      bool first = true;
      while (true) {
        if (!first) {
          if (!acceptPunct(",")) break;
        }
        first = false;
        if (isKeyword(cur(), "PRIMARY")) {
          next();
          expectKeyword("KEY");
          expectPunct("(");
          st.createTable.pkCol = expectIdent();
          if (st.createTable.pkConstraint.empty())
            st.createTable.pkConstraint = st.createTable.pkCol;
          expectPunct(")");
          continue;
        }
        Column col;
        col.name = expectIdent();
        if (isKeyword(cur(), "BIGINT") || isKeyword(cur(), "INT") || isKeyword(cur(), "INTEGER")) {
          next();
          col.type = Type::BIGINT;
          col.len = 8;
        } else if (isKeyword(cur(), "VARCHAR") || isKeyword(cur(), "CHAR") ||
                   isKeyword(cur(), "TEXT") || isKeyword(cur(), "STRING")) {
          next();
          col.type = Type::VARCHAR;
          col.len = 65535;
          if (acceptPunct("(")) {
            int64_t n = expectInt();
            if (n <= 0 || n > 65535) throwErr(1064, "42000", "invalid VARCHAR length");
            col.len = (int)n;
            expectPunct(")");
          }
        } else if (isKeyword(cur(), "VECTOR")) {
          next();
          col.type = Type::VECTOR;
          expectPunct("(");
          int64_t n = expectInt();
          if (n <= 0 || n > 65536) throwErr(1064, "42000", "invalid VECTOR dimension");
          col.len = (int)n;
          expectPunct(")");
        } else {
          syntax("unsupported column type");
        }
        if (isKeyword(cur(), "PRIMARY")) {
          next();
          acceptKeyword("KEY");
          st.createTable.pkCol = col.name;
          if (st.createTable.pkConstraint.empty()) st.createTable.pkConstraint = col.name;
        }
        if (isKeyword(cur(), "NOT")) {
          next();
          expectKeyword("NULL");
        }
        st.createTable.cols.push_back(col);
      }
      expectPunct(")");
      // Table options are accepted and ignored (ENGINE / CHARSET / COLLATE ...).
      while (cur().kind == TK_IDENT) {
        std::string opt = toLowerAscii(cur().text);
        if (opt != "engine" && opt != "default" && opt != "charset" && opt != "collate" &&
            opt != "comment" && opt != "auto_increment" && opt != "row_format")
          break;
        next();
        acceptPunct("=");
        if (cur().kind == TK_IDENT || cur().kind == TK_STR || cur().kind == TK_NUM) next();
      }
      return st;
    }
    syntax("expected TABLE or VECTOR INDEX");
  }

  Stmt parseCreateIndex() {
    Stmt st;
    st.kind = Stmt::CREATE_INDEX;
    if (acceptKeyword("IF")) {
      expectKeyword("NOT");
      expectKeyword("EXISTS");
    }
    st.createIndex.indexName = expectIdent();
    expectKeyword("ON");
    st.createIndex.table = expectIdent();
    expectPunct("(");
    st.createIndex.column = expectIdent();
    expectPunct(")");
    if (acceptKeyword("USING")) {
      st.createIndex.algorithm = toLowerAscii(expectIdent());
    }
    if (acceptKeyword("WITH")) {
      // ignore index options: WITH (M = 16, EF_CONSTRUCTION = 200)
      if (acceptPunct("(")) {
        int depth = 1;
        while (depth > 0 && cur().kind != TK_END) {
          if (isPunct("(")) depth++;
          else if (isPunct(")")) depth--;
          next();
        }
      }
    }
    return st;
  }

  Stmt parseDrop() {
    next();  // DROP
    Stmt st;
    if (acceptKeyword("TABLE")) {
      st.kind = Stmt::DROP_TABLE;
      if (acceptKeyword("IF")) {
        expectKeyword("EXISTS");
        st.dropTable.ifExists = true;
      }
      st.dropTable.name = expectIdent();
      return st;
    }
    if (acceptKeyword("INDEX")) {
      // DROP INDEX name ON table: index is derived state, nothing to drop.
      st.kind = Stmt::EMPTY;
      expectIdent();
      expectKeyword("ON");
      expectIdent();
      return st;
    }
    syntax("expected TABLE");
  }

  Stmt parseInsert() {
    Stmt st;
    st.kind = Stmt::INSERT;
    next();  // INSERT
    expectKeyword("INTO");
    st.insert.table = expectIdent();
    if (isPunct("(")) {
      next();
      while (true) {
        st.insert.cols.push_back(expectIdent());
        if (!acceptPunct(",")) break;
      }
      expectPunct(")");
    }
    expectKeyword("VALUES");
    while (true) {
      expectPunct("(");
      std::vector<ExprPtr> row;
      while (true) {
        row.push_back(parseValueExpr());
        if (!acceptPunct(",")) break;
      }
      expectPunct(")");
      st.insert.rows.push_back(std::move(row));
      if (!acceptPunct(",")) break;
    }
    return st;
  }

  Stmt parseSelect() {
    Stmt st;
    st.kind = Stmt::SELECT;
    next();  // SELECT
    if (isPunct("*")) {
      st.select.star = true;
      next();
    } else {
      while (true) {
        ExprPtr e = parseExpr();
        if (acceptKeyword("AS")) {
          e->alias = expectIdent();
          e->hasAlias = true;
        } else if (cur().kind == TK_IDENT && !isKeyword(cur(), "FROM")) {
          e->alias = cur().text;
          e->hasAlias = true;
          next();
        }
        if (e->kind == Expr::COLUMN && !e->hasAlias) e->alias = e->name;
        st.select.items.push_back(e);
        if (!acceptPunct(",")) break;
      }
    }
    if (!acceptKeyword("FROM")) syntax("expected FROM");
    st.select.table = expectIdent();
    if (acceptKeyword("WHERE")) st.select.where = parseOr();
    if (acceptKeyword("ORDER")) {
      expectKeyword("BY");
      st.select.hasOrder = true;
      ExprPtr e = parseExpr();
      st.select.orderExpr = e;
      if (acceptKeyword("DESC")) st.select.orderDesc = true;
      else acceptKeyword("ASC");
    }
    if (acceptKeyword("LIMIT")) {
      st.select.hasLimit = true;
      int64_t n = expectInt();
      if (n < 0) throwErr(1064, "42000", "LIMIT must not be negative");
      st.select.limit = n;
    }
    // trailing OFFSET is not part of the required subset; reject explicitly.
    if (isKeyword(cur(), "OFFSET")) syntax("OFFSET is not supported");
    return st;
  }

  Stmt parseUpdate() {
    Stmt st;
    st.kind = Stmt::UPDATE;
    next();  // UPDATE
    st.update.table = expectIdent();
    expectKeyword("SET");
    while (true) {
      std::string col = expectIdent();
      expectPunct("=");
      ExprPtr v = parseValueExpr();
      st.update.sets.emplace_back(col, v);
      if (!acceptPunct(",")) break;
    }
    if (acceptKeyword("WHERE")) st.update.where = parseOr();
    return st;
  }

  Stmt parseDelete() {
    Stmt st;
    st.kind = Stmt::DELETE_;
    next();  // DELETE
    expectKeyword("FROM");
    st.del.table = expectIdent();
    if (acceptKeyword("WHERE")) st.del.where = parseOr();
    return st;
  }

  Stmt parseSet() {
    Stmt st;
    st.kind = Stmt::SET;
    next();  // SET
    while (acceptKeyword("SESSION") || acceptKeyword("GLOBAL") || acceptKeyword("LOCAL")) {
    }
    while (acceptPunct("@")) {
    }
    std::string key;
    if (cur().kind != TK_IDENT) syntax("expected variable name");
    key = cur().text;
    next();
    while (isPunct(".") && cur().kind == TK_PUNCT) {
      next();
      if (cur().kind != TK_IDENT) break;
      key = cur().text;
      next();
    }
    if (!acceptPunct(":=")) {
      // SET vector_mode 'ann'  (tolerate a missing '=')
      acceptPunct("=");
    }
    if (cur().kind == TK_STR) {
      st.setValue = cur().text;
      next();
    } else if (cur().kind == TK_IDENT) {
      st.setValue = cur().text;
      next();
    } else if (cur().kind == TK_NUM) {
      st.setValue = cur().text;
      next();
    } else {
      syntax("expected value");
    }
    st.setKey = toLowerAscii(key);
    return st;
  }

  std::string sql_;
  std::vector<Token> t_;
  size_t i_ = 0;
};

}  // namespace

Stmt parseStatement(const std::string& sql) {
  std::vector<Token> toks;
  std::string err;
  Lexer lx(sql);
  if (!lx.run(toks, err)) throwErr(1064, "42000", err);
  Parser p(sql, std::move(toks));
  Stmt st = p.parse();
  // Anything left over means the statement is not fully understood: report it
  // instead of silently executing a prefix (the statement forbids that).
  if (!p.atEnd())
    throwErr(1064, "42000", "syntax error near '" + p.rest() + "'");
  return st;
}

// --------------------------------------------------------------- splitter ---

bool extractStatement(std::string& buf, std::string& stmt) {
  size_t i = 0;
  bool inSingle = false, inDouble = false, inBacktick = false, inLineComment = false,
       inBlockComment = false;
  while (i < buf.size()) {
    char c = buf[i];
    if (inLineComment) {
      if (c == '\n') inLineComment = false;
      i++;
      continue;
    }
    if (inBlockComment) {
      if (c == '*' && i + 1 < buf.size() && buf[i + 1] == '/') {
        inBlockComment = false;
        i += 2;
        continue;
      }
      i++;
      continue;
    }
    if (inSingle) {
      if (c == '\\' && i + 1 < buf.size()) {
        i += 2;
        continue;
      }
      if (c == '\'') {
        if (i + 1 < buf.size() && buf[i + 1] == '\'') {
          i += 2;
          continue;
        }
        inSingle = false;
      }
      i++;
      continue;
    }
    if (inDouble) {
      if (c == '\\' && i + 1 < buf.size()) {
        i += 2;
        continue;
      }
      if (c == '"') {
        if (i + 1 < buf.size() && buf[i + 1] == '"') {
          i += 2;
          continue;
        }
        inDouble = false;
      }
      i++;
      continue;
    }
    if (inBacktick) {
      if (c == '`') inBacktick = false;
      i++;
      continue;
    }
    if (c == '\'') {
      inSingle = true;
      i++;
      continue;
    }
    if (c == '"') {
      inDouble = true;
      i++;
      continue;
    }
    if (c == '`') {
      inBacktick = true;
      i++;
      continue;
    }
    if (c == '-' && i + 1 < buf.size() && buf[i + 1] == '-') {
      inLineComment = true;
      i += 2;
      continue;
    }
    if (c == '#') {
      inLineComment = true;
      i++;
      continue;
    }
    if (c == '/' && i + 1 < buf.size() && buf[i + 1] == '*') {
      inBlockComment = true;
      i += 2;
      continue;
    }
    if (c == ';') {
      stmt = buf.substr(0, i);
      buf.erase(0, i + 1);
      return true;
    }
    i++;
  }
  return false;
}

}  // namespace mvdb
