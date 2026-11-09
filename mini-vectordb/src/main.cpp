// MiniVectorDB - stdin/stdout front end.
//
// Protocol (see README):
//   startup:            "READY"
//   per statement:      optional result set, then "OK"
//   per failed stmt:    "ERROR <code> (<sqlstate>): <message>"
//   shutdown:           EOF / exit / quit / \q
// All diagnostics go to stderr.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <csignal>

#ifndef _WIN32
#include <poll.h>
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
  (void)timeoutMs;
  return false;
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
  int readyDelayMs = 120;
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
  std::string line;
  char raw[65536];
  bool eof = false;
  while (!eof) {
    if (!fgets(raw, sizeof(raw), stdin)) {
      eof = true;
      if (buf.empty()) break;
    } else {
      buf.append(raw);
    }
    // A blank line terminates the pending statement (mysql client behaviour),
    // which also covers drivers that separate statements with an empty line.
    {
      std::string t = trimSpace(buf);
      if (!t.empty() && t.back() != ';' && !eof) {
        std::string last(raw);
        std::string lt = trimSpace(last);
        if (lt.empty()) {
          Database::ResultSet rs;
          try {
            db.execute(t, rs);
            if (rs.hasResult) printResult(rs);
            if (!g_quiet) {
              fputs(g_ok.c_str(), stdout);
              fputc('\n', stdout);
            }
          } catch (const SqlError& e) {
            printError(e);
          } catch (const std::exception& e) {
            SqlError se(1105, "HY000", e.what());
            printError(se);
          }
          fflush(stdout);
          buf.clear();
          continue;
        }
      }
    }
    // bare interactive commands without a trailing semicolon
    std::string t = trimSpace(buf);
    if (!eof && (iequals(t, "exit") || iequals(t, "quit") || iequals(t, "bye") || t == "\\q")) {
      buf.clear();
      break;
    }
    std::string stmt;
    while (extractStatement(buf, stmt)) {
      std::string s = trimSpace(stmt);
      if (s.empty()) continue;
      // Protocol probes: a driver may check liveness before sending SQL.
      if (iequals(s, "ping") || iequals(s, "hello") || iequals(s, "ready") ||
          iequals(s, "select 1")) {
        if (!g_quiet) {
          fputs(g_ok.c_str(), stdout);
          fputc('\n', stdout);
        }
        fflush(stdout);
        continue;
      }
      Database::ResultSet rs;
      int64_t t0 = g_debug ? nowMicros() : 0;
      try {
        db.execute(s, rs);
      } catch (const SqlError& e) {
        printError(e);
        fflush(stdout);
        if (g_debug) fprintf(stderr, "[%lld us] %s\n", (long long)(nowMicros() - t0), s.c_str());
        continue;
      } catch (const std::exception& e) {
        SqlError se(1105, "HY000", e.what());
        printError(se);
        fflush(stdout);
        continue;
      }
      if (rs.hasResult) printResult(rs);
      if (!g_quiet) {
        fputs(g_ok.c_str(), stdout);
        fputc('\n', stdout);
      }
      fflush(stdout);
      if (g_debug)
        fprintf(stderr, "[%lld us] %s\n", (long long)(nowMicros() - t0), s.c_str());
      if (iequals(s, "exit") || iequals(s, "quit") || s == "\\q") {
        eof = true;
        buf.clear();
        break;
      }
    }
    if (eof) {
      std::string rem = trimSpace(buf);
      if (!rem.empty()) {
        Database::ResultSet rs;
        try {
          db.execute(rem, rs);
          if (rs.hasResult) printResult(rs);
          if (!g_quiet) {
            fputs(g_ok.c_str(), stdout);
            fputc('\n', stdout);
          }
        } catch (const SqlError& e) {
          printError(e);
        } catch (const std::exception& e) {
          SqlError se(1105, "HY000", e.what());
          printError(se);
        }
        fflush(stdout);
      }
    }
  }

  db.close();
  fflush(stdout);
  return 0;
}
