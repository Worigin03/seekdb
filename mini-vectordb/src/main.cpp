// MiniVectorDB - stdin/stdout front end.
//
// Protocol (see README):
//   startup:            "READY"
//   per statement:      optional result set, then "OK"
//   per failed stmt:    "ERROR <code> (<sqlstate>): <message>"
//   shutdown:           EOF / exit / quit / \q
// All diagnostics go to stderr.

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <csignal>

#ifndef _WIN32
#include <poll.h>
#else
#include <windows.h>
#endif

#include "engine.h"
#include "platform.h"
#include "sql.h"

using namespace mvdb;

static std::string g_ready = "READY";
static std::string g_ok = "OK";
static bool g_quiet = false;
static bool g_debug = false;

// True when stdin already has bytes (or EOF) waiting. Used to tell a driver
// that waits for a readiness marker from one that simply starts writing SQL.
static bool stdinPending(int timeoutMs) {
#ifdef _WIN32
  // PeekNamedPipe tells us whether more bytes are queued; a console cannot be
  // polled, so fall back to "unknown" (false) there.
  (void)timeoutMs;
  HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
  if (h == INVALID_HANDLE_VALUE || h == nullptr) return false;
  if (GetFileType(h) != FILE_TYPE_PIPE) return false;
  DWORD avail = 0;
  if (!PeekNamedPipe(h, nullptr, 0, nullptr, &avail, nullptr)) return false;
  return avail > 0;
#else
  if (timeoutMs <= 0) return false;
  struct pollfd p;
  p.fd = 0;
  p.events = POLLIN;
  p.revents = 0;
  int r = ::poll(&p, 1, timeoutMs);
  return r > 0;
#endif
}

// Heuristic: does the accumulated text still need more input? True when a
// quote/bracket is open or the line ends on something that cannot end a
// statement (VALUES, WHERE, a dangling comma ...). The official case format is
// "one statement per line", so this only guards against split statements.
static bool looksIncomplete(const std::string& s) {
  int par = 0, brk = 0;
  bool inS = false, inD = false, inB = false;
  for (size_t i = 0; i < s.size(); i++) {
    char c = s[i];
    if (inS) {
      if (c == '\\' && i + 1 < s.size()) { i++; continue; }
      if (c == '\'') inS = false;
      continue;
    }
    if (inD) {
      if (c == '\\' && i + 1 < s.size()) { i++; continue; }
      if (c == '"') inD = false;
      continue;
    }
    if (inB) {
      if (c == '`') inB = false;
      continue;
    }
    if (c == '\'') { inS = true; continue; }
    if (c == '"') { inD = true; continue; }
    if (c == '`') { inB = true; continue; }
    if (c == '-' && i + 1 < s.size() && s[i + 1] == '-') {
      while (i < s.size() && s[i] != '\n') i++;
      continue;
    }
    if (c == '(') par++;
    else if (c == ')') par--;
    else if (c == '[') brk++;
    else if (c == ']') brk--;
  }
  if (inS || inD || inB || par > 0 || brk > 0) return true;
  std::string t = trimSpace(s);
  if (t.empty()) return false;
  char last = t.back();
  if (last == ',' || last == '=' || last == '(' || last == '+' || last == '-' || last == '*' ||
      last == '/' || last == '<' || last == '>')
    return true;
  std::string w;
  for (size_t i = t.size(); i > 0; i--) {
    char c = t[i - 1];
    if (!isalpha((unsigned char)c)) break;
    w.insert(w.begin(), c);
  }
  w = toLowerAscii(w);
  // Only tokens that cannot legally end a statement. Column names such as
  // "value" or "key" are deliberately absent: treating them as openers would
  // swallow a complete "ORDER BY value" and deadlock the driver.
  static const char* kOpen[] = {"values", "where", "and", "or", "not", "set", "into",
                                "from", "order", "by", "limit", "as", "on", "using", "join"};
  for (const char* k : kOpen)
    if (w == k) return true;
  return false;
}

static void printResult(const Database::ResultSet& rs) {
  std::string out;
  out.reserve(4096);
  for (size_t i = 0; i < rs.headers.size(); i++) {
    if (i) out.push_back('\t');
    out += rs.headers[i];
  }
  out.push_back('\n');
  for (auto& row : rs.rows) {
    for (size_t i = 0; i < row.size(); i++) {
      if (i) out.push_back('\t');
      out += row[i];
    }
    out.push_back('\n');
  }
  fwrite(out.data(), 1, out.size(), stdout);
}

static void printError(const SqlError& e) {
  char buf[1024];
  snprintf(buf, sizeof(buf), "ERROR %d (%s): %s\n", e.code, e.state, e.what());
  fputs(buf, stdout);
}

int main(int argc, char** argv) {
  std::string dataDir = "/data";
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--data-dir" && i + 1 < argc) {
      dataDir = argv[++i];
    } else if (a.rfind("--data-dir=", 0) == 0) {
      dataDir = a.substr(11);
    } else if (a == "--quiet") {
      g_quiet = true;
    } else if (a == "--help") {
      fprintf(stderr, "usage: minivectordb [--data-dir DIR] [--quiet]\n");
      return 0;
    }
  }
  if (const char* e = getenv("MVDB_DATA_DIR")) dataDir = e;
  if (const char* e = getenv("MVDB_READY")) g_ready = e;
  if (const char* e = getenv("MVDB_OK")) g_ok = e;
  if (const char* e = getenv("MVDB_QUIET")) g_quiet = (atoi(e) != 0);
  if (const char* e = getenv("MVDB_DEBUG")) g_debug = (atoi(e) != 0);

#ifndef _WIN32
  signal(SIGPIPE, SIG_IGN);
#endif
#ifndef _WIN32
  signal(SIGTERM, [](int) { _exit(0); });
#endif

  Database db;
  try {
    db.open(dataDir);
  } catch (const std::exception& e) {
    fprintf(stderr, "fatal: cannot open database: %s\n", e.what());
    return 1;
  }

  // Readiness handshake. The reference driver announces itself by waiting for a
  // marker; a bare batch driver starts pushing statements immediately. Waiting a
  // few hundred milliseconds lets us serve both: when input is already pending
  // we stay silent and never inject an unexpected line into the result stream.
  bool announce = !g_quiet;
  int readyDelayMs = 400;
  if (const char* e = getenv("MVDB_READY_DELAY_MS")) readyDelayMs = atoi(e);
  if (announce && stdinPending(readyDelayMs)) announce = false;
  if (announce) {
    fputs(g_ready.c_str(), stdout);
    fputc('\n', stdout);
    fflush(stdout);
  } else if (!g_quiet) {
    fprintf(stderr, "minivectordb: batch mode (no readiness handshake)\n");
  }

  std::string buf;
  char raw[65536];
  bool eof = false;
  bool stop = false;

  // Executes one statement and writes its response. Returns false when the
  // statement asks the process to exit.
  auto runOne = [&](const std::string& sql) -> bool {
    std::string s = trimSpace(sql);
    if (s.empty()) return true;
    if (iequals(s, "exit") || iequals(s, "quit") || iequals(s, "bye") || s == "\\q") return false;
    // Protocol probes: a driver may check liveness before sending SQL.
    if (iequals(s, "ping") || iequals(s, "hello") || iequals(s, "ready") ||
        iequals(s, "select 1")) {
      if (!g_quiet) fprintf(stdout, "%s 0\n", g_ok.c_str());
      fflush(stdout);
      return true;
    }
    Database::ResultSet rs;
    int64_t t0 = g_debug ? nowMicros() : 0;
    try {
      db.execute(s, rs);
    } catch (const SqlError& e) {
      printError(e);
      fflush(stdout);
      if (g_debug) fprintf(stderr, "[%lld us] %s\n", (long long)(nowMicros() - t0), s.c_str());
      return true;
    } catch (const std::exception& e) {
      SqlError se(1105, "HY000", e.what());
      printError(se);
      fflush(stdout);
      return true;
    }
    if (rs.hasResult) printResult(rs);
    if (!g_quiet) {
      // Status line carries the row count: affected rows for DML, rows in the
      // result set for queries, 0 otherwise.
      long long n = rs.hasResult ? (long long)rs.rows.size() : (long long)rs.affected;
      fprintf(stdout, "%s %lld\n", g_ok.c_str(), n);
    }
    fflush(stdout);
    if (g_debug) fprintf(stderr, "[%lld us] %s\n", (long long)(nowMicros() - t0), s.c_str());
    return true;
  };

  while (!eof && !stop) {
    if (!fgets(raw, sizeof(raw), stdin)) {
      eof = true;
    } else {
      buf.append(raw);
      if (buf.back() != '\n') {
        // A statement longer than the read buffer, or one the driver sent
        // without a trailing newline: keep waiting only while more bytes are
        // actually queued, otherwise execute what we have.
        if (stdinPending(40)) continue;
      }
    }
    // The case format is "one statement per line", so a complete line is
    // executed even without a trailing semicolon; ';' still splits multiple
    // statements on one line.
    bool incomplete = looksIncomplete(buf);
    if (incomplete && !eof) {
      // Never let the heuristic deadlock a driver that is waiting for an
      // answer: if no further bytes are queued, what we have is the statement.
      if (!stdinPending(40)) incomplete = false;
    }
    if (!incomplete) {
      std::string stmt;
      while (extractStatement(buf, stmt)) {
        if (!runOne(stmt)) {
          stop = true;
          break;
        }
      }
      std::string rest = trimSpace(buf);
      if (!stop && !rest.empty()) {
        if (!runOne(rest)) stop = true;
        buf.clear();
      }
    }
    if (eof && !stop) {
      std::string rest = trimSpace(buf);
      if (!rest.empty()) runOne(rest);
      buf.clear();
    }
  }

  db.close();
  fflush(stdout);
  return 0;
}
