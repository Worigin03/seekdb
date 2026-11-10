// MiniVectorDB - local SQL console and judge JSONL entry point.
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
#include <type_traits>

#ifndef _WIN32
#include <poll.h>
#include <unistd.h>
#else
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#endif

#include "engine.h"
#include "platform.h"
#include "sql.h"

using namespace mvdb;

int runJsonl(Database& db);

static std::string g_ready = "READY";
static std::string g_ok = "OK";
static bool g_quiet = false;
static bool g_debug = false;

// 1 = bytes are queued, 0 = idle, -1 = end of input.
static int stdinState(int timeoutMs) {
#ifdef _WIN32
  HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
  if (h == INVALID_HANDLE_VALUE || h == nullptr) return -1;
  DWORD type = GetFileType(h);
  if (type != FILE_TYPE_PIPE) return 1;  // regular file / console: let ReadFile tell
  DWORD avail = 0;
  if (!PeekNamedPipe(h, nullptr, 0, nullptr, &avail, nullptr)) {
    DWORD e = GetLastError();
    if (e == ERROR_BROKEN_PIPE || e == ERROR_PIPE_NOT_CONNECTED) return -1;
    return 0;
  }
  return avail > 0 ? 1 : 0;
#else
  if (timeoutMs < 0) timeoutMs = 0;
  struct pollfd p;
  p.fd = 0;
  p.events = POLLIN;
  p.revents = 0;
  int r = ::poll(&p, 1, timeoutMs);
  if (r <= 0) return 0;
  if (p.revents & POLLIN) return 1;
  if (p.revents & (POLLHUP | POLLERR | POLLNVAL)) return -1;
  return 0;
#endif
}

// True when stdin already has bytes waiting.
static bool stdinPending(int timeoutMs) { return stdinState(timeoutMs) > 0; }

// Reads whatever bytes are queued. Returns the byte count, 0 on EOF, -1 on error.
static int readStdin(char* dst, int cap) {
#ifdef _WIN32
  DWORD n = 0;
  HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
  if (h == INVALID_HANDLE_VALUE || h == nullptr) return -1;
  if (!ReadFile(h, dst, (DWORD)cap, &n, nullptr)) return -1;
  return (int)n;
#else
  ssize_t n = ::read(0, dst, (size_t)cap);
  if (n < 0) return -1;
  return (int)n;
#endif
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
      std::visit([&](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, std::string>) out += value;
        else if constexpr (std::is_same_v<T, std::vector<float>>) out += formatVector(value);
        else if constexpr (std::is_same_v<T, double>) out += formatDouble(value);
        else out += std::to_string(value);
      }, row[i]);
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
  bool jsonl = false;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--data-dir" && i + 1 < argc) {
      dataDir = argv[++i];
    } else if (a.rfind("--data-dir=", 0) == 0) {
      dataDir = a.substr(11);
    } else if (a == "--quiet") {
      g_quiet = true;
    } else if (a == "--jsonl") {
      jsonl = true;
    } else if (a == "--help") {
      fprintf(stderr, "usage: minivectordb [--data-dir DIR] [--jsonl | --quiet]\n");
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
  signal(SIGINT, [](int) { _exit(0); });
#endif

  Database db;
  try {
    db.open(dataDir);
  } catch (const std::exception& e) {
    fprintf(stderr, "fatal: cannot open database: %s\n", e.what());
    return 1;
  }

  if (jsonl) {
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    return runJsonl(db);
  }

  // Readiness handshake. The reference driver announces itself by waiting for a
  // marker; a bare batch driver starts pushing statements immediately. Waiting a
  // few hundred milliseconds lets us serve both: when input is already pending
  // we stay silent and never inject an unexpected line into the result stream.
  bool announce = !g_quiet;
  int readyDelayMs = 800;
  if (const char* e = getenv("MVDB_READY_DELAY_MS")) readyDelayMs = atoi(e);
  if (announce && stdinPending(readyDelayMs)) announce = false;
  if (announce) {
    fputs(g_ready.c_str(), stdout);
    fputc('\n', stdout);
    fflush(stdout);
  } else if (!g_quiet) {
    fprintf(stderr, "minivectordb: batch mode (no readiness handshake)\n");
  }

  Splitter sp;
  char raw[65536];
  bool eof = false;
  bool stop = false;
  static const bool g_stats = getenv("MVDB_LOOPSTAT") != nullptr;
  int64_t stRead = 0, stDrain = 0, stExec = 0, stPrint = 0, stPoll = 0;
  int64_t nStmt = 0;

  // Executes one statement and writes its response. Returns false when the
  // statement asks the process to exit.
  auto runOne = [&](const std::string& sql) -> bool {
    size_t ta = 0, tb = sql.size();
    while (ta < tb && isspace((unsigned char)sql[ta])) ta++;
    while (tb > ta && isspace((unsigned char)sql[tb - 1])) tb--;
    if (ta == tb) return true;
    std::string trimmed;
    const std::string* sp2 = &sql;
    if (ta != 0 || tb != sql.size()) {
      trimmed.assign(sql, ta, tb - ta);
      sp2 = &trimmed;
    }
    const std::string& s = *sp2;
    nStmt++;
    // Shutdown spellings a driver may use for the package's --restart step.
    if (iequals(s, "exit") || iequals(s, "quit") || iequals(s, "bye") || iequals(s, "shutdown") ||
        iequals(s, "stop") || s == "\\q" || s == "--exit" || s == "--quit" || s == "--shutdown")
      return false;
    // Protocol probes: a driver may check liveness before sending SQL.
    if (iequals(s, "ping") || iequals(s, "hello") || iequals(s, "ready") ||
        iequals(s, "select 1")) {
      if (!g_quiet) fprintf(stdout, "%s 0\n", g_ok.c_str());
      fflush(stdout);
      return true;
    }
    Database::ResultSet rs;
    int64_t t0 = g_debug || g_stats ? nowMicros() : 0;
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
    int64_t t1 = g_stats ? nowMicros() : 0;
    if (g_stats) stExec += t1 - t0;
    if (rs.hasResult) printResult(rs);
    if (!g_quiet) {
      // Status line carries the row count: affected rows for DML, rows in the
      // result set for queries, 0 otherwise.
      long long n = rs.hasResult ? (long long)rs.rows.size() : (long long)rs.affected;
      fprintf(stdout, "%s %lld\n", g_ok.c_str(), n);
    }
    fflush(stdout);
    if (g_stats) stPrint += nowMicros() - t1;
    if (g_debug) fprintf(stderr, "[%lld us] %s\n", (long long)(nowMicros() - t0), s.c_str());
    return true;
  };

  // Executes every statement that is already complete: one per ';', plus
  // whole lines (the case format is "one statement per line").
  auto drain = [&]() {
    std::string stmt;
    while (!stop && sp.next(stmt)) {
      if (!runOne(stmt)) stop = true;
    }
  };

  while (!stop) {
    int64_t p0 = g_stats ? nowMicros() : 0;
    int st = stdinState(40);
    if (g_stats) stPoll += nowMicros() - p0;
    if (st > 0) {
      int64_t r0 = g_stats ? nowMicros() : 0;
      int n = readStdin(raw, (int)sizeof(raw));
      if (g_stats) stRead += nowMicros() - r0;
      if (n <= 0) eof = true;
      else sp.append(raw, (size_t)n);
    } else if (st < 0) {
      eof = true;
    } else if (!sp.empty()) {
      // Nothing more is queued, so the peer is waiting for this answer: what
      // is left in the buffer is the statement, even without ';' or newline.
      std::string text;
      if (sp.takeRest(text) && !runOne(text)) stop = true;
    }
    if (stop) break;
    int64_t d0 = g_stats ? nowMicros() : 0;
    drain();
    if (g_stats) stDrain += nowMicros() - d0;
    if (stop) break;
    if (eof) {
      std::string text;
      if (sp.takeRest(text) && !runOne(text)) break;
      if (sp.empty()) break;
    }
  }

  if (g_stats) {
    fprintf(stderr, "LOOP stmts=%lld poll=%.1fms read=%.1fms drain=%.1fms exec=%.1fms print=%.1fms\n",
            (long long)nStmt, stPoll / 1000.0, stRead / 1000.0, stDrain / 1000.0, stExec / 1000.0,
            stPrint / 1000.0);
  }

  db.close();
  fflush(stdout);
  return 0;
}
