#include "platform.h"

#include <cerrno>
#include <algorithm>

namespace mvdb {

// ------------------------------------------------------------------ crc32 ---

namespace {
struct Crc32Table {
  uint32_t t[8][256];
  Crc32Table() {
    for (uint32_t i = 0; i < 256; i++) {
      uint32_t c = i;
      for (int k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      t[0][i] = c;
    }
    // Slicing-by-8: each further table advances the CRC by one byte.
    for (int k = 1; k < 8; k++)
      for (uint32_t i = 0; i < 256; i++)
        t[k][i] = (t[k - 1][i] >> 8) ^ t[0][t[k - 1][i] & 0xff];
  }
};
const Crc32Table& crcTable() {
  static Crc32Table tab;
  return tab;
}
}  // namespace

uint32_t crc32Of(const void* data, size_t len) {
  const Crc32Table& tab = crcTable();
  const uint8_t* p = (const uint8_t*)data;
  uint32_t c = 0xFFFFFFFFu;
  // Bulk of the WAL payload is long records; eight bytes per iteration keeps
  // the checksum off the insert hot path.
  while (len >= 8) {
    uint32_t one, two;
    memcpy(&one, p, 4);
    memcpy(&two, p + 4, 4);
    one ^= c;
    c = tab.t[7][one & 0xff] ^ tab.t[6][(one >> 8) & 0xff] ^ tab.t[5][(one >> 16) & 0xff] ^
        tab.t[4][(one >> 24) & 0xff] ^ tab.t[3][two & 0xff] ^ tab.t[2][(two >> 8) & 0xff] ^
        tab.t[1][(two >> 16) & 0xff] ^ tab.t[0][(two >> 24) & 0xff];
    p += 8;
    len -= 8;
  }
  for (size_t i = 0; i < len; i++) c = tab.t[0][(c ^ p[i]) & 0xff] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

std::string toLowerAscii(const std::string& s) {
  std::string r = s;
  for (auto& c : r)
    if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
  return r;
}

bool iequals(const std::string& a, const std::string& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); i++) {
    char x = a[i], y = b[i];
    if (x >= 'A' && x <= 'Z') x = (char)(x - 'A' + 'a');
    if (y >= 'A' && y <= 'Z') y = (char)(y - 'A' + 'a');
    if (x != y) return false;
  }
  return true;
}

bool istartsWith(const std::string& s, const std::string& prefix) {
  if (s.size() < prefix.size()) return false;
  return iequals(s.substr(0, prefix.size()), prefix);
}

void makeDirs(const std::string& path) {
  if (path.empty()) return;
  std::string cur;
  for (size_t i = 0; i < path.size(); i++) {
    char c = path[i];
    cur.push_back(c);
    bool last = (i + 1 == path.size());
    if (c == '/' || c == '\\' || last) {
      if (cur.size() <= 1) continue;
      std::string p = cur;
      while (!p.empty() && (p.back() == '/' || p.back() == '\\')) p.pop_back();
      if (p.empty()) continue;
#ifdef _WIN32
      _mkdir(p.c_str());
#else
      mkdir(p.c_str(), 0755);
#endif
    }
  }
}

bool pathExists(const std::string& path) {
#ifdef _WIN32
  return _access(path.c_str(), 0) == 0;
#else
  return access(path.c_str(), F_OK) == 0;
#endif
}

bool readWholeFile(const std::string& path, std::string& out) {
  File f;
  if (!f.openRead(path)) return false;
  uint64_t n = f.size();
  out.resize((size_t)n);
  if (n && !f.readAt(0, &out[0], (size_t)n)) return false;
  return true;
}

bool writeWholeFileAtomic(const std::string& path, const std::string& data) {
  std::string tmp = path + ".tmp";
  File f;
  if (!f.openWrite(tmp)) return false;
  if (!f.writeAt(0, data.data(), data.size())) return false;
  f.sync();
  f.close();
#ifdef _WIN32
  _unlink(path.c_str());
#endif
  if (rename(tmp.c_str(), path.c_str()) != 0) return false;
#ifndef _WIN32
  // fsync the directory so the rename itself is durable.
  int dfd = ::open(path.substr(0, path.find_last_of('/')).c_str(), O_RDONLY | O_DIRECTORY);
  if (dfd >= 0) {
    ::fsync(dfd);
    ::close(dfd);
  }
#endif
  return true;
}

// -------------------------------------------------------------------- File ---

bool File::open(const std::string& path, bool write, bool create) {
  close();
#ifdef _WIN32
  int flags = _O_BINARY | (write ? _O_RDWR : _O_RDONLY);
  if (create) flags |= _O_CREAT;
  int fd = _open(path.c_str(), flags, _S_IREAD | _S_IWRITE);
#else
  int flags = write ? O_RDWR : O_RDONLY;
  if (create) flags |= O_CREAT;
  int fd = ::open(path.c_str(), flags, 0644);
#endif
  if (fd < 0) return false;
  fd_ = fd;
#ifdef _WIN32
  struct _stat64 st;
  if (_fstat64(fd_, &st) == 0) size_ = (uint64_t)st.st_size;
#else
  struct stat st;
  if (fstat(fd_, &st) == 0) size_ = (uint64_t)st.st_size;
#endif
  return true;
}

void File::close() {
  if (fd_ >= 0) {
#ifdef _WIN32
    _close(fd_);
#else
    ::close(fd_);
#endif
    fd_ = -1;
  }
  size_ = 0;
}

bool File::readAt(uint64_t off, void* buf, size_t len) const {
  if (fd_ < 0) return false;
  if (off + len > size_) return false;
  size_t done = 0;
  char* p = (char*)buf;
#ifdef _WIN32
  std::lock_guard<std::mutex> lk(mu_);
  if (_lseeki64(fd_, (__int64)off, SEEK_SET) < 0) return false;
  while (done < len) {
    unsigned int chunk = (unsigned int)std::min<size_t>(len - done, 1u << 30);
    int n = _read(fd_, p + done, chunk);
    if (n <= 0) return false;
    done += (size_t)n;
  }
#else
  while (done < len) {
    ssize_t n = ::pread(fd_, p + done, len - done, (off_t)(off + done));
    if (n <= 0) {
      if (n < 0 && errno == EINTR) continue;
      return false;
    }
    done += (size_t)n;
  }
#endif
  return true;
}

bool File::writeAt(uint64_t off, const void* buf, size_t len) {
  if (fd_ < 0) return false;
  size_t done = 0;
  const char* p = (const char*)buf;
#ifdef _WIN32
  std::lock_guard<std::mutex> lk(mu_);
  if (_lseeki64(fd_, (__int64)off, SEEK_SET) < 0) return false;
  while (done < len) {
    unsigned int chunk = (unsigned int)std::min<size_t>(len - done, 1u << 30);
    int n = _write(fd_, p + done, chunk);
    if (n <= 0) return false;
    done += (size_t)n;
  }
#else
  while (done < len) {
    ssize_t n = ::pwrite(fd_, p + done, len - done, (off_t)(off + done));
    if (n <= 0) {
      if (n < 0 && errno == EINTR) continue;
      return false;
    }
    done += (size_t)n;
  }
#endif
  if (off + len > size_) size_ = off + len;
  return true;
}

bool File::truncate(uint64_t len) {
  if (fd_ < 0) return false;
#ifdef _WIN32
  std::lock_guard<std::mutex> lk(mu_);
  if (_chsize_s(fd_, (__int64)len) != 0) return false;
#else
  if (::ftruncate(fd_, (off_t)len) != 0) return false;
#endif
  if (len < size_) size_ = len;
  return true;
}

void File::sync() {
  if (fd_ < 0) return;
#ifdef _WIN32
  std::lock_guard<std::mutex> lk(mu_);
  _commit(fd_);
#else
  ::fdatasync(fd_);
#endif
}

// -------------------------------------------------------------- MappedFile ---

bool MappedFile::map(const File& f, uint64_t len, bool write) {
  unmap();
  if (len == 0 || !f.valid()) return false;
  return mapFd(f.fd(), len, write);
}

bool MappedFile::mapFd(int fd, uint64_t len, bool write) {
  unmap();
  if (fd < 0 || len == 0) return false;
#ifdef _WIN32
  HANDLE h = (HANDLE)_get_osfhandle(fd);
  if (h == INVALID_HANDLE_VALUE) return false;
  DWORD prot = write ? PAGE_READWRITE : PAGE_READONLY;
  HANDLE m = CreateFileMappingA(h, NULL, prot, (DWORD)(len >> 32), (DWORD)(len & 0xffffffffu), NULL);
  if (!m) return false;
  void* p = MapViewOfFile(m, write ? FILE_MAP_WRITE : FILE_MAP_READ, 0, 0, (SIZE_T)len);
  if (!p) {
    CloseHandle(m);
    return false;
  }
  base_ = p;
  mapping_ = m;
  fileHandle_ = h;
  len_ = len;
  return true;
#else
  int prot = PROT_READ | (write ? PROT_WRITE : 0);
  void* p = mmap(nullptr, (size_t)len, prot, MAP_SHARED, fd, 0);
  if (p == MAP_FAILED) return false;
  base_ = p;
  len_ = len;
  return true;
#endif
}

void MappedFile::unmap() {
#ifdef _WIN32
  if (base_) UnmapViewOfFile(base_);
  if (mapping_) CloseHandle((HANDLE)mapping_);
  mapping_ = nullptr;
  fileHandle_ = nullptr;
#else
  if (base_) munmap(base_, (size_t)len_);
#endif
  base_ = nullptr;
  len_ = 0;
}

// -------------------------------------------------------------- thread pool ---

namespace {

// CPUs this process may actually run on. hardware_concurrency() reports the
// host's CPU count, which overstates a container with a CPU quota; oversizing
// the pool there would make the workers fight the scanning thread for cores.
int usableCpus() {
  unsigned hw = std::thread::hardware_concurrency();
  if (hw == 0) hw = 1;
  int n = (int)hw;
#ifndef _WIN32
  // Affinity mask: containers pinned with cpuset report all host CPUs above.
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) {
    int c = CPU_COUNT(&set);
    if (c >= 1 && c < n) n = c;
  }
  auto clampQuota = [&](double quota, double period) {
    if (quota > 0 && period > 0) {
      int q = (int)((quota + period - 1) / period);
      if (q >= 1 && q < n) n = q;
    }
  };
  if (FILE* f = fopen("/sys/fs/cgroup/cpu.max", "r")) {  // cgroup v2
    char q[64] = {0};
    long long period = 0;
    if (fscanf(f, "%63s %lld", q, &period) == 2 && strcmp(q, "max") != 0)
      clampQuota(atof(q), (double)period);
    fclose(f);
  }
  if (FILE* f = fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r")) {  // cgroup v1
    long long quota = -1, period = 0;
    if (fscanf(f, "%lld", &quota) == 1 && quota > 0) {
      if (FILE* g = fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r")) {
        if (fscanf(g, "%lld", &period) == 1) clampQuota((double)quota, (double)period);
        fclose(g);
      }
    }
    fclose(f);
  }
#endif
  return n < 1 ? 1 : n;
}

inline void cpuRelax() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  __builtin_ia32_pause();
#else
  std::this_thread::yield();
#endif
}

}  // namespace

ThreadPool& ThreadPool::get() {
  static ThreadPool pool;
  return pool;
}

ThreadPool::ThreadPool() {
  int wanted = usableCpus();
  if (const char* e = getenv("MVDB_THREADS")) {
    int v = atoi(e);
    if (v > 0) wanted = v;
  }
  if (wanted > 8) wanted = 8;
  if (wanted < 1) wanted = 1;
  workers_.reserve((size_t)(wanted - 1));
  for (int i = 1; i < wanted; i++) workers_.emplace_back([this] { loop(); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
    stopAtom_.store(true, std::memory_order_release);
  }
  cv_.notify_all();
  for (auto& t : workers_)
    if (t.joinable()) t.join();
}

void ThreadPool::loop() {
  uint64_t seen = 0;
  static const int kSpin = getenv("MVDB_SPIN") ? atoi(getenv("MVDB_SPIN")) : 50000;
  for (;;) {
    // Spin briefly before sleeping: back-to-back queries fan out again within
    // microseconds, and a futex round trip would cost more than the scan.
    for (int i = 0; i < kSpin && genAtom_.load(std::memory_order_acquire) == seen; i++) cpuRelax();
    std::unique_lock<std::mutex> lk(mu_);
    if (gen_ == seen) {
      cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
    }
    if (stop_) return;
    seen = gen_;
    while (next_ < jobs_) {
      int i = next_++;
      const std::function<void(int)>* fn = job_;
      lk.unlock();
      (*fn)(i);
      lk.lock();
    }
    if (--active_ == 0) done_.notify_one();
  }
}

void ThreadPool::parallelFor(int jobs, const std::function<void(int)>& fn) {
  if (jobs <= 1 || workers_.empty()) {
    for (int i = 0; i < jobs; i++) fn(i);
    return;
  }
  {
    std::lock_guard<std::mutex> lk(mu_);
    job_ = &fn;
    jobs_ = jobs;
    next_ = 0;
    active_ = (int)workers_.size();
    gen_++;
    genAtom_.store(gen_, std::memory_order_release);
  }
  cv_.notify_all();
  for (;;) {
    int i;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (next_ >= jobs_) break;
      i = next_++;
    }
    fn(i);
  }
  std::unique_lock<std::mutex> lk(mu_);
  done_.wait(lk, [&] { return active_ == 0; });
  job_ = nullptr;
}

}  // namespace mvdb
