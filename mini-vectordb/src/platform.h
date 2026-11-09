// MiniVectorDB - platform layer: file IO, mmap, time, misc utils.
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <functional>
#include <thread>
#include <condition_variable>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <direct.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <dirent.h>
#endif

namespace mvdb {

// ---------------------------------------------------------------- errors ---

struct SqlError : public std::runtime_error {
  int code;
  const char* state;
  SqlError(int c, const char* s, const std::string& m)
      : std::runtime_error(m), code(c), state(s) {}
};

[[noreturn]] inline void throwErr(int code, const char* state, const std::string& msg) {
  throw SqlError(code, state, msg);
}

// ----------------------------------------------------------------- utils ---

uint32_t crc32Of(const void* data, size_t len);

inline void putU32(std::string& out, uint32_t v) {
  char b[4];
  b[0] = (char)(v & 0xff);
  b[1] = (char)((v >> 8) & 0xff);
  b[2] = (char)((v >> 16) & 0xff);
  b[3] = (char)((v >> 24) & 0xff);
  out.append(b, 4);
}

inline uint32_t readU32(const char* p) {
  return (uint32_t)(uint8_t)p[0] | ((uint32_t)(uint8_t)p[1] << 8) |
         ((uint32_t)(uint8_t)p[2] << 16) | ((uint32_t)(uint8_t)p[3] << 24);
}

inline void putU64(std::string& out, uint64_t v) {
  putU32(out, (uint32_t)(v & 0xffffffffu));
  putU32(out, (uint32_t)(v >> 32));
}

inline void putVarint(std::string& out, uint64_t v) {
  while (v >= 0x80) {
    out.push_back((char)((v & 0x7f) | 0x80));
    v >>= 7;
  }
  out.push_back((char)v);
}

// Reads a LEB128 varint. Returns false on overflow/truncation.
inline bool getVarint(const char*& p, const char* end, uint64_t& out) {
  uint64_t v = 0;
  int shift = 0;
  while (p < end) {
    uint8_t c = (uint8_t)*p++;
    if (shift > 63) return false;
    v |= (uint64_t)(c & 0x7f) << shift;
    if (!(c & 0x80)) {
      out = v;
      return true;
    }
    shift += 7;
  }
  return false;
}

std::string toLowerAscii(const std::string& s);
bool iequals(const std::string& a, const std::string& b);
bool istartsWith(const std::string& s, const std::string& prefix);

inline int64_t nowMicros() {
#ifdef _WIN32
  LARGE_INTEGER f, c;
  QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&c);
  return (int64_t)((double)c.QuadPart * 1000000.0 / (double)f.QuadPart);
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
#endif
}

void makeDirs(const std::string& path);
bool pathExists(const std::string& path);
bool readWholeFile(const std::string& path, std::string& out);
bool writeWholeFileAtomic(const std::string& path, const std::string& data);

// ------------------------------------------------------------- thread pool ---
//
// Small persistent pool used to spread one top-k scan over several cores. The
// calling thread takes part in every fan-out, so a single-core machine pays no
// extra wake-up cost when the fan-out is skipped by the caller.
class ThreadPool {
 public:
  static ThreadPool& get();

  // Number of threads that can run a chunk, including the caller.
  int size() const { return (int)workers_.size() + 1; }

  // Runs fn(i) for i in [0, jobs); returns when every index is done.
  void parallelFor(int jobs, const std::function<void(int)>& fn);

 private:
  ThreadPool();
  ~ThreadPool();
  void loop();

  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable done_;
  const std::function<void(int)>* job_ = nullptr;
  int jobs_ = 0;
  int next_ = 0;
  int active_ = 0;
  uint64_t gen_ = 0;
  std::atomic<uint64_t> genAtom_{0};
  bool stop_ = false;
  std::atomic<bool> stopAtom_{false};
};

// ------------------------------------------------------------------ file ---

class File {
 public:
  File() = default;
  ~File() { close(); }
  File(const File&) = delete;
  File& operator=(const File&) = delete;

  bool open(const std::string& path, bool write, bool create);
  bool openRead(const std::string& path) { return open(path, false, false); }
  bool openWrite(const std::string& path) { return open(path, true, true); }
  void close();
  bool valid() const { return fd_ >= 0; }

  uint64_t size() const { return size_; }
  bool readAt(uint64_t off, void* buf, size_t len) const;
  bool writeAt(uint64_t off, const void* buf, size_t len);
  bool append(const void* buf, size_t len) { return writeAt(size_, buf, len); }
  bool truncate(uint64_t len);
  void sync();
  int fd() const { return fd_; }

 private:
  int fd_ = -1;
  uint64_t size_ = 0;
#ifdef _WIN32
  mutable std::mutex mu_;
#endif
};

class MappedFile {
 public:
  MappedFile() = default;
  ~MappedFile() { unmap(); }
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  bool map(const File& f, uint64_t len, bool write);
  bool mapFd(int fd, uint64_t len, bool write);
  void unmap();
  uint8_t* data() const { return (uint8_t*)base_; }
  uint64_t size() const { return len_; }
  bool mapped() const { return base_ != nullptr; }

 private:
  void* base_ = nullptr;
  uint64_t len_ = 0;
#ifdef _WIN32
  void* mapping_ = nullptr;
  void* fileHandle_ = nullptr;
#endif
};

}  // namespace mvdb
