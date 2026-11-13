#include "engine.h"
#include "simd.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>
#include <random>

namespace mvdb {

// --------------------------------------------------------------- VecStore ---

VecStore::~VecStore() {
  map_.unmap();
  file_.close();
  free(mem_);
  mem_ = nullptr;
}

void VecStore::init(int dim, const std::string& path, size_t memBudgetBytes) {
  dim_ = dim;
  slot_ = (uint64_t)dim * sizeof(float);
  path_ = path;
  budget_ = memBudgetBytes;
  capRows_ = 0;
  disk_ = false;
  free(mem_);
  mem_ = nullptr;
  sketch_.clear();
  sketchSafe_.clear();
  sketchNormScaled_.clear();
  sketchSum_.clear();
  sketchAbs_.clear();
  sketchEnabled_ = dim > 0 && dim <= 1024;
}

void VecStore::reset() {
  map_.unmap();
  file_.close();
  free(mem_);
  mem_ = nullptr;
  capRows_ = 0;
  disk_ = false;
  fileCap_ = 0;
  sketch_.clear();
  sketchSafe_.clear();
  sketchNormScaled_.clear();
  sketchSum_.clear();
  sketchAbs_.clear();
}

void VecStore::switchToDisk() {
  disk_ = true;
  if (!file_.openWrite(path_)) throwErr(1105, "HY000", "cannot open vector file");
  if (capRows_ > 0) {
    file_.writeAt(0, mem_, (size_t)(capRows_ * slot_));
  }
  uint64_t bytes = (uint64_t)capRows_ * slot_;
  if (bytes == 0) bytes = slot_ * 1024;
  file_.truncate(bytes);
  fileCap_ = bytes;
  free(mem_);
  mem_ = nullptr;
  if (!map_.map(file_, fileCap_, true)) throwErr(1105, "HY000", "cannot map vector file");
}

void VecStore::growDisk(uint64_t needRows) {
  uint64_t newCap = std::max<uint64_t>(needRows, std::max<uint64_t>(capRows_ * 2, 1024));
  uint64_t bytes = newCap * slot_;
  if (bytes > fileCap_) {
    map_.unmap();
    uint64_t target = std::max<uint64_t>(bytes, fileCap_ * 2);
    if (!file_.truncate(target)) throwErr(1105, "HY000", "cannot grow vector file");
    fileCap_ = target;
    if (!map_.map(file_, fileCap_, true)) throwErr(1105, "HY000", "cannot map vector file");
  }
  capRows_ = newCap;
}

// Grows the in-memory store. realloc is deliberate: glibc serves large blocks
// with mmap and can usually extend them with mremap, so doubling a 100 MB
// store costs microseconds instead of a full copy plus a zero-fill.
bool VecStore::growMem(uint64_t needRows) {
  uint64_t newCap = std::max<uint64_t>(needRows, std::max<uint64_t>(capRows_ * 2, 1024));
  uint64_t bytes = newCap * slot_;
  if (bytes > budget_) return false;
  float* p = (float*)realloc(mem_, (size_t)bytes);
  if (!p) return false;
  mem_ = p;
  capRows_ = newCap;
  return true;
}

void VecStore::ensure(uint64_t rowid) {
  if (rowid < capRows_) return;
  uint64_t need = rowid + 1;
  if (!disk_) {
    if (growMem(need)) return;
    switchToDisk();
    growDisk(need);
    return;
  }
  growDisk(need);
}

void VecStore::set(uint64_t rowid, const float* v) {
  ensure(rowid);
  float* p = ptr(rowid);
  memcpy(p, v, slot_);
  if (!sketchEnabled_ || rowid >= 131072) {
    sketchEnabled_ = false;
    sketch_.clear();
    sketchSafe_.clear();
    sketchNormScaled_.clear();
    sketchSum_.clear();
    sketchAbs_.clear();
    return;
  }
  const size_t need = (size_t)(rowid + 1) * (size_t)dim_;
  if (sketch_.size() < need) sketch_.resize(need);
  if (sketchSafe_.size() <= rowid) sketchSafe_.resize((size_t)rowid + 1);
  if (sketchNormScaled_.size() <= rowid) sketchNormScaled_.resize((size_t)rowid + 1);
  if (sketchSum_.size() <= rowid) sketchSum_.resize((size_t)rowid + 1);
  if (sketchAbs_.size() <= rowid) sketchAbs_.resize((size_t)rowid + 1);
  int8_t* dst = sketch_.data() + (size_t)rowid * (size_t)dim_;
  bool safe = true;
  double norm = 0.0;
  int32_t ssum = 0, sabs = 0;
  constexpr double kStep = 0.008;
  for (int i = 0; i < dim_; i++) {
    // Fixed grid: step 0.008, signed byte zero is the origin. Values outside
    // [-1.024, 1.016] are stored clamped and flagged; those rows always take
    // the exact path. Non-finite input is likewise flagged.
    float scaled = v[i] * 125.0f;
    bool finite = std::isfinite(scaled);
    long q = finite ? std::lrintf(scaled) : 0;
    if (!finite || q < -128) {
      safe = false;
      q = -128;
    } else if (q > 127) {
      safe = false;
      q = 127;
    }
    dst[i] = (int8_t)q;
    norm += (double)v[i] * (double)v[i];
    ssum += (int32_t)q;
    sabs += (int32_t)(q < 0 ? -q : q);
  }
  sketchSafe_[rowid] = safe ? 1 : 0;
  sketchNormScaled_[rowid] = norm / (kStep * kStep);
  sketchSum_[rowid] = ssum;
  sketchAbs_[rowid] = sabs;
}

bool VecStore::sketchQuery(const float* q, std::vector<uint8_t>& out) const {
  out.resize((size_t)dim_);
  bool safe = true;
  for (int i = 0; i < dim_; i++) {
    float scaled = q[i] * 125.0f;
    bool finite = std::isfinite(scaled);
    long v = finite ? std::lrintf(scaled) + 128 : 0;
    if (!finite || v < 0) {
      safe = false;
      v = 0;
    } else if (v > 255) {
      safe = false;
      v = 255;
    }
    out[i] = (uint8_t)v;
  }
  return safe;
}

float* VecStore::ptr(uint64_t rowid) {
  if (disk_) return (float*)(map_.data() + rowid * slot_);
  return mem_ + rowid * (uint64_t)dim_;
}

const float* VecStore::ptr(uint64_t rowid) const {
  if (disk_) return (const float*)(map_.data() + rowid * slot_);
  return mem_ + rowid * (uint64_t)dim_;
}

// ---------------------------------------------------------------- PkIndex ---

static inline uint64_t fnv1a(const std::string& s) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

static inline uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

void PkIndex::init(size_t minCap) {
  size_t c = 1024;
  while (c < minCap) c <<= 1;
  keys_.assign(c, 0);
  vals_.assign(c, 0);
  used_.assign(c, 0);
  mask_ = c - 1;
  count_ = 0;
}

void PkIndex::clear() {
  std::fill(used_.begin(), used_.end(), 0);
  count_ = 0;
}

bool PkIndex::get(int64_t key, uint64_t& val) const {
  if (used_.empty()) return false;
  size_t i = (size_t)mix64((uint64_t)key) & mask_;
  while (true) {
    if (used_[i] == 0) return false;
    if (used_[i] == 1 && keys_[i] == key) {
      val = vals_[i];
      return true;
    }
    i = (i + 1) & mask_;
  }
}

void PkIndex::put(int64_t key, uint64_t val) {
  if (used_.empty()) init(1024);
  size_t i = (size_t)mix64((uint64_t)key) & mask_;
  size_t tomb = (size_t)-1;
  while (true) {
    if (used_[i] == 0) {
      size_t slot = (tomb != (size_t)-1) ? tomb : i;
      keys_[slot] = key;
      vals_[slot] = val;
      used_[slot] = 1;
      count_++;
      break;
    }
    if (used_[i] == 1 && keys_[i] == key) {
      vals_[i] = val;
      return;
    }
    if (used_[i] == 2 && tomb == (size_t)-1) tomb = i;
    i = (i + 1) & mask_;
  }
  if (count_ * 10 >= (mask_ + 1) * 7) grow();
}

bool PkIndex::erase(int64_t key) {
  if (used_.empty()) return false;
  size_t i = (size_t)mix64((uint64_t)key) & mask_;
  while (true) {
    if (used_[i] == 0) return false;
    if (used_[i] == 1 && keys_[i] == key) {
      used_[i] = 2;
      count_--;
      return true;
    }
    i = (i + 1) & mask_;
  }
}

void PkIndex::grow() {
  std::vector<int64_t> ok = std::move(keys_);
  std::vector<uint64_t> ov = std::move(vals_);
  std::vector<uint8_t> ou = std::move(used_);
  size_t nc = (mask_ + 1) * 2;
  keys_.assign(nc, 0);
  vals_.assign(nc, 0);
  used_.assign(nc, 0);
  mask_ = nc - 1;
  count_ = 0;
  for (size_t i = 0; i < ou.size(); i++) {
    if (ou[i] == 1) put(ok[i], ov[i]);
  }
}

// ------------------------------------------------------------------- Hnsw ---

float Hnsw::dist(const float* a, const float* b) const { return l2sq(a, b, dim_); }

float Hnsw::nodeDist(const float* q, uint32_t node) const {
  return dist(q, table_->vecs->ptr(rowidOfNode(node)));
}

// rowid lookup helper is inlined via nodeOf_/nodes_ mapping below

int Hnsw::greedyStep(const float* q, int ep, int level) const {
  float d = nodeDist(q, ep);
  bool changed = true;
  while (changed) {
    changed = false;
    if (level >= (int)nodes_[ep].links.size()) break;
    for (uint32_t nb : nodes_[ep].links[level]) {
      float dn = nodeDist(q, nb);
      if (dn < d) {
        d = dn;
        ep = (int)nb;
        changed = true;
      }
    }
  }
  return ep;
}

void Hnsw::searchLayer(const float* q, int ep, int ef, int level,
                       std::vector<std::pair<float, uint32_t>>& out,
                       std::vector<uint32_t>& visited) const {
  out.clear();
  visitStamp_++;
  auto mark = [&](uint32_t n) {
    if (visit_[n] != visitStamp_) {
      visit_[n] = visitStamp_;
      return true;
    }
    return false;
  };
  using P = std::pair<float, uint32_t>;
  std::priority_queue<P, std::vector<P>, std::greater<P>> cand;
  std::vector<P> best;  // max-heap emulation: keep sorted desc
  float d0 = nodeDist(q, ep);
  cand.push({d0, (uint32_t)ep});
  mark((uint32_t)ep);
  best.push_back({d0, (uint32_t)ep});
  while (!cand.empty()) {
    P c = cand.top();
    cand.pop();
    float worst = best.size() >= (size_t)ef ? best.back().first : std::numeric_limits<float>::max();
    if (best.size() >= (size_t)ef && c.first > worst) break;
    uint32_t node = c.second;
    if (level >= (int)nodes_[node].links.size()) continue;
    for (uint32_t nb : nodes_[node].links[level]) {
      if (!mark(nb)) continue;
      float dn = nodeDist(q, nb);
      float w = best.size() >= (size_t)ef ? best.back().first : std::numeric_limits<float>::max();
      if (best.size() < (size_t)ef || dn < w) {
        cand.push({dn, nb});
        auto it = std::lower_bound(best.begin(), best.end(), P(dn, nb),
                                   [](const P& a, const P& b) { return a.first < b.first; });
        best.insert(it, {dn, nb});
        if (best.size() > (size_t)ef) best.pop_back();
      }
    }
  }
  out = best;
}

void Hnsw::selectNeighbors(const float* q, std::vector<std::pair<float, uint32_t>>& cands,
                           int m, std::vector<uint32_t>& out) const {
  std::sort(cands.begin(), cands.end());
  out.clear();
  for (auto& c : cands) {
    if ((int)out.size() >= m) break;
    bool keep = true;
    for (uint32_t s : out) {
      float ds = dist(table_->vecs->ptr(rowidOfNode(c.second)),
                      table_->vecs->ptr(rowidOfNode(s)));
      if (ds < c.first) {
        keep = false;
        break;
      }
    }
    if (keep) out.push_back(c.second);
  }
}

void Hnsw::addRow(uint64_t rowid) {
  if (dim_ <= 0) return;
  if (rowid >= nodeOf_.size()) {
    nodeOf_.resize(rowid + 1024, -1);
  }
  if (nodeOf_[rowid] >= 0) return;  // already indexed
  uint32_t id = (uint32_t)nodes_.size();
  double ml = 1.0 / std::log((double)M_);
  static thread_local std::mt19937_64 rng(0x9E3779B97F4A7C15ull);
  std::uniform_real_distribution<double> uni(1e-12, 1.0);
  int level = (int)(-std::log(uni(rng)) * ml);
  if (level > 24) level = 24;
  nodes_.emplace_back();
  nodes_.back().level = level;
  nodes_.back().links.resize(level + 1);
  dead_.push_back(0);
  rowidOf_.push_back(rowid);
  nodeOf_[rowid] = (int32_t)id;
  if (visit_.size() < nodes_.size()) visit_.resize(nodes_.size() + 1024, 0);
  if (entry_ < 0) {
    entry_ = (int)id;
    maxLevel_ = level;
    return;
  }
  const float* q = table_->vecs->ptr(rowid);
  int ep = entry_;
  for (int l = maxLevel_; l > level; l--) ep = greedyStep(q, ep, l);
  std::vector<std::pair<float, uint32_t>> cands;
  std::vector<uint32_t> chosen;
  for (int l = std::min(level, maxLevel_); l >= 0; l--) {
    searchLayer(q, ep, efConstruction_, l, cands, visit_);
    selectNeighbors(q, cands, (l == 0 ? M0_ : M_), chosen);
    nodes_[id].links[l] = chosen;
    for (uint32_t nb : chosen) {
      auto& lnk = nodes_[nb].links[l];
      lnk.push_back(id);
      int cap = (l == 0 ? M0_ : M_);
      if ((int)lnk.size() > cap) {
        std::vector<std::pair<float, uint32_t>> cc;
        const float* base = table_->vecs->ptr(rowidOfNode(nb));
        cc.reserve(lnk.size());
        for (uint32_t x : lnk) cc.push_back({dist(base, table_->vecs->ptr(rowidOfNode(x))), x});
        std::vector<uint32_t> keep;
        selectNeighbors(base, cc, cap, keep);
        lnk = keep;
      }
    }
    if (!cands.empty()) ep = (int)cands[0].second;
  }
  if (level > maxLevel_) {
    maxLevel_ = level;
    entry_ = (int)id;
  }
}

void Hnsw::removeRow(uint64_t rowid) {
  if (rowid >= nodeOf_.size()) return;
  int32_t n = nodeOf_[rowid];
  if (n >= 0 && (size_t)n < dead_.size()) dead_[n] = 1;
}

void Hnsw::build(Table* t) {
  table_ = t;
  if (t->schema.vec < 0) return;
  dim_ = t->schema.cols[t->schema.vec].len;
  nodes_.clear();
  dead_.clear();
  nodeOf_.clear();
  rowidOf_.clear();
  visit_.clear();
  entry_ = -1;
  maxLevel_ = -1;
  visitStamp_ = 1;
  nodeOf_.resize(t->nextRowid + 1024, -1);
  for (uint64_t r = 0; r < t->nextRowid; r++) {
    if (t->live[r]) addRow(r);
  }
}

void Hnsw::search(const float* q, int k, int ef,
                  const std::function<bool(uint64_t)>& accept,
                  std::vector<std::pair<float, uint64_t>>& out) {
  out.clear();
  if (entry_ < 0 || k <= 0) return;
  int ep = entry_;
  for (int l = maxLevel_; l > 0; l--) ep = greedyStep(q, ep, l);
  std::vector<std::pair<float, uint32_t>> cands;
  searchLayer(q, ep, std::max(ef, k * 4), 0, cands, visit_);
  for (auto& c : cands) {
    uint32_t node = c.second;
    if (dead_[node]) continue;
    uint64_t rid = rowidOfNode(node);
    if (!accept(rid)) continue;
    out.push_back({c.first, rid});
  }
  std::sort(out.begin(), out.end());
  if ((int)out.size() > k) out.resize(k);
}

// ------------------------------------------------------------------ Table ---

// (Table is a plain struct; helpers live in Database.)

// --------------------------------------------------------------- Database ---

static const char* kWALMagic = "MVDBWAL1";

Database::~Database() { close(); }

void Database::open(const std::string& dir) {
  initSimd();
  dir_ = dir;
  makeDirs(dir_);
  if (const char* b = getenv("MVDB_VEC_MEM_MB")) {
    long v = atol(b);
    if (v > 0) vecMemBudget_ = (size_t)v << 20;
  }
  walPath_ = dir_ + "/wal.log";
  openWal();
  recover();
}

void Database::openWal() {
  if (!wal_.open(walPath_, true, true)) throwErr(1105, "HY000", "cannot open " + walPath_);
  if (wal_.size() < 8) {
    std::string hdr = kWALMagic;
    hdr.append(8 - hdr.size(), '\0');
    wal_.writeAt(0, hdr.data(), hdr.size());
    wal_.sync();
  }
}

void Database::ensureMapped() {
  uint64_t sz = wal_.size();
  if (sz == 0) return;
  if (map_.mapped() && sz <= mappedLen_) return;
  if (!map_.map(wal_, sz, false)) throwErr(1105, "HY000", "cannot map WAL");
  mappedLen_ = sz;
}

const char* Database::payloadPtr(const Table& t, uint64_t rowid) {
  ensureMapped();
  return (const char*)map_.data() + t.off[rowid];
}

uint64_t Database::appendRecord(const std::string& body) {
  std::string rec;
  rec.reserve(body.size() + 8);
  putU32(rec, (uint32_t)body.size());
  putU32(rec, crc32Of(body.data(), body.size()));
  rec.append(body);
  uint64_t off = wal_.size() + 8;
  if (!wal_.writeAt(wal_.size(), rec.data(), rec.size()))
    throwErr(1105, "HY000", "WAL write failed");
  return off;
}

void Database::appendCommit() {
  std::string body;
  body.push_back((char)6);
  appendRecord(body);
}

void Database::appendRecords(const std::vector<std::string>& bodies, std::vector<uint64_t>& offs) {
  std::string buf;
  size_t bytes = inTxn_ ? 0 : 9;  // autocommit footer shares the same pwrite
  for (const auto& body : bodies) bytes += 8 + body.size();
  buf.reserve(bytes);
  uint64_t base = wal_.size();
  offs.clear();
  offs.reserve(bodies.size());
  for (auto& b : bodies) {
    offs.push_back(base + buf.size() + 8);
    putU32(buf, (uint32_t)b.size());
    putU32(buf, crc32Of(b.data(), b.size()));
    buf.append(b);
  }
  if (!inTxn_ && !bodies.empty()) {
    const char commit = 6;
    putU32(buf, 1);
    putU32(buf, crc32Of(&commit, 1));
    buf.push_back(commit);
  }
  if (!buf.empty()) {
    if (!wal_.writeAt(base, buf.data(), buf.size()))
      throwErr(1105, "HY000", "WAL write failed");
  }
}

void Database::syncWal() { wal_.sync(); }

void Database::truncateWal(uint64_t off) {
  if (wal_.size() <= off) return;
  map_.unmap();
  mappedLen_ = 0;
  wal_.truncate(off);
}

void Database::recover() {
  ensureMapped();
  uint64_t size = wal_.size();
  if (size < 8) {
    // A crash between creating the file and writing its header (or a zeroed
    // file) must not brick the database: an empty log is simply an empty one.
    std::string hdr = kWALMagic;
    hdr.append(8 - hdr.size(), '\0');
    if (!wal_.writeAt(0, hdr.data(), hdr.size()))
      throwErr(1105, "HY000", "cannot initialise " + walPath_);
    wal_.sync();
    map_.unmap();
    mappedLen_ = 0;
    ensureMapped();
    size = wal_.size();
  }
  if (size < 8 || memcmp(map_.data(), kWALMagic, 7) != 0)
    throwErr(1105, "HY000", "bad WAL header");
  uint64_t pos = 8;
  uint64_t lastGood = 8;
  std::vector<std::pair<const char*, size_t>> group;
  std::vector<uint64_t> groupOffs;
  while (pos + 8 <= size) {
    uint32_t len = readU32((const char*)map_.data() + pos);
    uint32_t crc = readU32((const char*)map_.data() + pos + 4);
    if (len == 0 || len > (64u << 20)) break;
    if (pos + 8 + len > size) break;
    const char* body = (const char*)map_.data() + pos + 8;
    if (crc32Of(body, len) != crc) break;
    if ((uint8_t)body[0] == 6) {
      for (size_t i = 0; i < group.size(); i++) applyRecord(group[i].first, group[i].second, groupOffs[i]);
      group.clear();
      groupOffs.clear();
      pos += 8 + len;
      lastGood = pos;
    } else {
      group.push_back({body, len});
      groupOffs.push_back(pos + 8);
      pos += 8 + len;
    }
  }
  if (lastGood < size) {
    map_.unmap();
    mappedLen_ = 0;
    wal_.truncate(lastGood);
    ensureMapped();
  }
  groupStart_ = wal_.size();
}

void Database::encodeCreateTable(const Table& t, std::string& out) {
  out.clear();
  out.push_back((char)1);
  putVarint(out, t.id);
  putVarint(out, t.schema.name.size());
  out.append(t.schema.name);
  putVarint(out, t.schema.cols.size());
  for (auto& c : t.schema.cols) {
    putVarint(out, c.name.size());
    out.append(c.name);
    out.push_back((char)c.type);
    putVarint(out, (uint64_t)c.len);
  }
  putVarint(out, (uint64_t)t.schema.pk);
  putVarint(out, (uint64_t)t.schema.vec);
}

void Database::applyCreateTable(const char* body, size_t len) {
  const char* p = body + 1;
  const char* end = body + len;
  uint64_t tid = 0, n = 0;
  if (!getVarint(p, end, tid)) throwErr(1105, "HY000", "corrupt CREATE TABLE record");
  if (!getVarint(p, end, n)) throwErr(1105, "HY000", "corrupt CREATE TABLE record");
  std::string tname(p, (size_t)n);
  p += n;
  uint64_t ncols = 0;
  if (!getVarint(p, end, ncols)) throwErr(1105, "HY000", "corrupt CREATE TABLE record");
  auto t = std::make_unique<Table>();
  t->id = (uint32_t)tid;
  t->schema.name = tname;
  for (uint64_t i = 0; i < ncols; i++) {
    uint64_t nl = 0;
    if (!getVarint(p, end, nl) || p + nl > end) throwErr(1105, "HY000", "corrupt schema");
    Column c;
    c.name.assign(p, (size_t)nl);
    p += nl;
    if (p >= end) throwErr(1105, "HY000", "corrupt schema");
    c.type = (Type)(uint8_t)*p++;
    uint64_t cl = 0;
    if (!getVarint(p, end, cl)) throwErr(1105, "HY000", "corrupt schema");
    c.len = (int)cl;
    t->schema.cols.push_back(c);
  }
  uint64_t v = 0;
  if (!getVarint(p, end, v)) throwErr(1105, "HY000", "corrupt schema");
  t->schema.pk = (int)v;
  if (!getVarint(p, end, v)) throwErr(1105, "HY000", "corrupt schema");
  t->schema.vec = (int)v;
  t->pk.init(1024);
  t->cache.resize(t->schema.cols.size());
  if (t->schema.vec >= 0) {
    t->vecs = std::make_unique<VecStore>();
    t->vecs->init(t->schema.cols[t->schema.vec].len,
                  dir_ + "/t" + std::to_string(t->id) + ".vec", vecMemBudget_);
  }
  if (t->id >= nextTableId_) nextTableId_ = t->id + 1;
  tables_.push_back(std::move(t));
}

void Database::applyDropTable(uint32_t tableId) {
  for (size_t i = 0; i < tables_.size(); i++) {
    if (tables_[i]->id == tableId) {
      tables_.erase(tables_.begin() + i);
      return;
    }
  }
}

void Database::applyInsert(uint32_t tableId, uint64_t rowid, int64_t pk, const char* payload,
                           uint32_t plen, uint64_t off) {
  Table* t = nullptr;
  for (auto& p : tables_)
    if (p->id == tableId) t = p.get();
  if (!t) return;
  t->topkFilters.clear();
  if (rowid >= t->off.size()) {
    size_t n = (size_t)rowid + 1;
    t->off.resize(n, 0);
    t->len.resize(n, 0);
    t->live.resize(n, 0);
    t->pkOf.resize(n, 0);
  }
  t->off[rowid] = off;
  t->len[rowid] = plen;
  t->live[rowid] = 1;
  t->pkOf[rowid] = pk;
  t->pk.put(pk, rowid);
  if (rowid >= t->nextRowid) t->nextRowid = rowid + 1;
  if (t->vecs) {
    // copy the vector straight out of the payload
    int vi = t->schema.vec;
    const char* p = payload;
    const char* end = payload + plen;
    bool ok = true;
    for (size_t i = 0; i < t->schema.cols.size() && ok; i++) {
      if ((int)i == t->schema.pk) continue;
      if ((int)i == vi) {
        int dim = t->schema.cols[i].len;
        if (p + (size_t)dim * 4 > end) {
          ok = false;
          break;
        }
        t->vecs->set(rowid, (const float*)p);
        p += (size_t)dim * 4;
      } else if (t->schema.cols[i].type == Type::BIGINT) {
        p += 8;
      } else {
        uint64_t n = 0;
        if (!getVarint(p, end, n)) {
          ok = false;
          break;
        }
        p += n;
      }
    }
  }
  // keep filter caches in sync for tables that already have one built
  bool anyCache = false;
  for (auto& c : t->cache)
    if (c.ready) anyCache = true;
  if (anyCache) {
    std::vector<Value> tmp;
    decodeRowAt(*t, rowid, tmp);
    for (size_t ci = 0; ci < t->cache.size(); ci++) {
      if (!t->cache[ci].ready) continue;
      ColCache& cc = t->cache[ci];
      if (t->schema.cols[ci].type == Type::BIGINT) {
        if (cc.ints.size() <= rowid) cc.ints.resize(rowid + 1, 0);
        cc.ints[rowid] = tmp[ci].i;
      } else if (t->schema.cols[ci].type == Type::VARCHAR) {
        if (cc.lens.size() <= rowid) {
          cc.lens.resize(rowid + 1, 0);
          cc.offs.resize(rowid + 1, 0);
          cc.hashes.resize(rowid + 1, 0);
        }
        cc.offs[rowid] = (uint32_t)cc.arena.size();
        cc.lens[rowid] = (uint32_t)tmp[ci].s.size();
        cc.hashes[rowid] = fnv1a(tmp[ci].s);
        cc.arena += tmp[ci].s;
      }
    }
  }
}

void Database::applyUpdate(uint32_t tableId, uint64_t oldRowid, uint64_t newRowid, int64_t pk,
                           const char* payload, uint32_t plen, uint64_t off) {
  Table* t = nullptr;
  for (auto& p : tables_)
    if (p->id == tableId) t = p.get();
  if (!t) return;
  t->topkFilters.clear();
  if (oldRowid < t->live.size()) {
    t->live[oldRowid] = 0;
    uint64_t cur = 0;
    if (t->pk.get(pk, cur) && cur == oldRowid) t->pk.erase(pk);
    if (t->hnsw) t->hnsw->removeRow(oldRowid);
  }
  applyInsert(tableId, newRowid, pk, payload, plen, off);
}

void Database::applyDelete(uint32_t tableId, uint64_t rowid, int64_t pk) {
  Table* t = nullptr;
  for (auto& p : tables_)
    if (p->id == tableId) t = p.get();
  if (!t) return;
  t->topkFilters.clear();
  if (rowid < t->live.size()) {
    t->live[rowid] = 0;
    uint64_t cur = 0;
    if (t->pk.get(pk, cur) && cur == rowid) t->pk.erase(pk);
    if (t->hnsw) t->hnsw->removeRow(rowid);
  }
}

void Database::applyRecord(const char* body, size_t len, uint64_t off) {
  if (len == 0) return;
  uint8_t type = (uint8_t)body[0];
  const char* p = body + 1;
  const char* end = body + len;
  uint64_t a = 0, b = 0;
  switch (type) {
    case 1:
      applyCreateTable(body, len);
      break;
    case 2:
      if (getVarint(p, end, a)) applyDropTable((uint32_t)a);
      break;
    case 3: {
      if (!getVarint(p, end, a)) return;
      if (!getVarint(p, end, b)) return;
      if (p + 8 > end) return;
      uint64_t u = 0;
      for (int k = 0; k < 8; k++) u |= (uint64_t)(uint8_t)p[k] << (8 * k);
      p += 8;
      uint64_t payloadOff = off + (uint64_t)(p - body);
      applyInsert((uint32_t)a, b, (int64_t)u, p, (uint32_t)(end - p), payloadOff);
      break;
    }
    case 4: {
      if (!getVarint(p, end, a)) return;
      if (!getVarint(p, end, b)) return;
      if (p + 8 > end) return;
      uint64_t u = 0;
      for (int k = 0; k < 8; k++) u |= (uint64_t)(uint8_t)p[k] << (8 * k);
      applyDelete((uint32_t)a, b, (int64_t)u);
      break;
    }
    case 5: {
      uint64_t oldR = 0, newR = 0;
      if (!getVarint(p, end, a)) return;
      if (!getVarint(p, end, oldR)) return;
      if (!getVarint(p, end, newR)) return;
      if (p + 8 > end) return;
      uint64_t u = 0;
      for (int k = 0; k < 8; k++) u |= (uint64_t)(uint8_t)p[k] << (8 * k);
      p += 8;
      uint64_t payloadOff = off + (uint64_t)(p - body);
      applyUpdate((uint32_t)a, oldR, newR, (int64_t)u, p, (uint32_t)(end - p), payloadOff);
      break;
    }
    default:
      break;
  }
}

// ---------------------------------------------------------- transaction ---

void Database::beginTxn() {
  if (inTxn_) throwErr(1235, "42000", "nested transactions are not supported");
  inTxn_ = true;
  undo_.clear();
  groupStart_ = wal_.size();
}

void Database::appendCommitAndSync() {
  appendCommit();
  syncWal();
}

void Database::commitTxn() {
  if (inTxn_) {
    if (wal_.size() > groupStart_) appendCommitAndSync();
    inTxn_ = false;
    undo_.clear();
    groupStart_ = wal_.size();
  } else {
    // autocommit
    appendCommitAndSync();
  }
}

void Database::rollbackTxn() {
  if (!inTxn_) return;
  undoAll();
  truncateWal(groupStart_);
  ensureMapped();
  inTxn_ = false;
  undo_.clear();
}

void Database::undoTo(size_t mark) {
  std::vector<uint32_t> touched;
  while (undo_.size() > mark) {
    Undo u = undo_.back();
    undo_.pop_back();
    Table* t = nullptr;
    for (auto& p : tables_)
      if (p->id == u.tableId) t = p.get();
    if (!t) continue;
    t->topkFilters.clear();
    if (u.kind == 0) {  // undo insert
      if (u.rowid < t->live.size()) {
        t->live[u.rowid] = 0;
        uint64_t cur = 0;
        if (t->pk.get(u.pk, cur) && cur == u.rowid) t->pk.erase(u.pk);
      }
    } else {  // undo delete
      if (u.rowid < t->live.size()) {
        // A column cache first built while this row was deleted skipped it.
        for (auto& cache : t->cache)
          if (cache.ready) cache = ColCache{};
        t->live[u.rowid] = 1;
        t->pk.put(u.pk, u.rowid);
      }
    }
    if (t->hnsw) {
      bool seen = false;
      for (uint32_t x : touched)
        if (x == t->id) seen = true;
      if (!seen) touched.push_back(t->id);
    }
  }
  // The graph cannot be repaired incrementally after a rollback (row ids are
  // reused), so rebuild it from the restored rows.
  for (uint32_t id : touched) {
    for (auto& t : tables_)
      if (t->id == id && t->hnsw) {
        t->hnsw->build(t.get());
        break;
      }
  }
}

void Database::undoAll() { undoTo(0); }

void Database::applyPendingInsert(Table& t, uint64_t rowid, int64_t pk, const char* payload,
                                  uint32_t plen, uint64_t off, bool recordUndo) {
  applyInsert(t.id, rowid, pk, payload, plen, off);
  if (t.hnsw) t.hnsw->addRow(rowid);
  if (recordUndo) undo_.push_back(Undo{0, t.id, rowid, pk});
}

void Database::applyPendingDelete(Table& t, uint64_t rowid, int64_t pk, bool recordUndo) {
  applyDelete(t.id, rowid, pk);
  if (recordUndo) undo_.push_back(Undo{1, t.id, rowid, pk});
}

}  // namespace mvdb
