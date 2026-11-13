#include "simd.h"

#include <cfloat>
#include <queue>
#include <vector>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define MVDB_X86 1
#include <immintrin.h>
#else
#define MVDB_X86 0
#endif

namespace mvdb {

static float l2sqScalar(const float* a, const float* b, int d) {
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  int i = 0;
  for (; i + 4 <= d; i += 4) {
    float x0 = a[i] - b[i], x1 = a[i + 1] - b[i + 1], x2 = a[i + 2] - b[i + 2],
          x3 = a[i + 3] - b[i + 3];
    s0 += x0 * x0;
    s1 += x1 * x1;
    s2 += x2 * x2;
    s3 += x3 * x3;
  }
  for (; i < d; i++) {
    float x = a[i] - b[i];
    s0 += x * x;
  }
  return s0 + s1 + s2 + s3;
}

#if MVDB_X86
__attribute__((target("avx2"))) static inline double l2sqExactAvx2(
    const float* a, const float* b, int d) {
  __m256d sums = _mm256_setzero_pd();
  int i = 0;
  for (; i + 4 <= d; i += 4) {
    __m256d x = _mm256_sub_pd(_mm256_cvtps_pd(_mm_loadu_ps(a + i)),
                              _mm256_cvtps_pd(_mm_loadu_ps(b + i)));
    sums = _mm256_add_pd(sums, _mm256_mul_pd(x, x));
  }
  // Match the scalar reference's four accumulators and reduction order.
  double lanes[4];
  _mm256_storeu_pd(lanes, sums);
  for (; i < d; i++) {
    double x = (double)a[i] - (double)b[i];
    lanes[0] += x * x;
  }
  return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

__attribute__((target("avx2,fma"))) static inline float l2sqAvx2(const float* __restrict a,
                                                                const float* __restrict b,
                                                                int d) {
  __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
  __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
  int i = 0;
  // Four independent accumulator chains: FMA latency is 4-5 cycles, so two
  // chains leave the FMA ports half idle on a 128/256-dim scan.
  for (; i + 32 <= d; i += 32) {
    __m256 x0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    __m256 x1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
    __m256 x2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
    __m256 x3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
    s0 = _mm256_fmadd_ps(x0, x0, s0);
    s1 = _mm256_fmadd_ps(x1, x1, s1);
    s2 = _mm256_fmadd_ps(x2, x2, s2);
    s3 = _mm256_fmadd_ps(x3, x3, s3);
  }
  for (; i + 8 <= d; i += 8) {
    __m256 x = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    s0 = _mm256_fmadd_ps(x, x, s0);
  }
  s0 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
  __m128 lo = _mm256_castps256_ps128(s0);
  __m128 hi = _mm256_extractf128_ps(s0, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  float s = _mm_cvtss_f32(lo);
  for (; i < d; i++) {
    float x = a[i] - b[i];
    s += x * x;
  }
  return s;
}

__attribute__((target("avx2"))) static inline float hsumAvx2Four(
    __m256 s0, __m256 s1, __m256 s2, __m256 s3) {
  __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
  __m128 lo = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}

// Two rows share every query-vector load while retaining four FMA chains per row.
__attribute__((target("avx2,fma"))) static inline void l2sqAvx2x2(
    const float* __restrict q, const float* __restrict a, const float* __restrict b,
    int dim, float* __restrict out) {
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
  __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
  __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
  int i = 0;
  for (; i + 32 <= dim; i += 32) {
    __m256 qv = _mm256_loadu_ps(q + i);
    __m256 xa = _mm256_sub_ps(_mm256_loadu_ps(a + i), qv);
    __m256 xb = _mm256_sub_ps(_mm256_loadu_ps(b + i), qv);
    a0 = _mm256_fmadd_ps(xa, xa, a0);
    b0 = _mm256_fmadd_ps(xb, xb, b0);
    qv = _mm256_loadu_ps(q + i + 8);
    xa = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), qv);
    xb = _mm256_sub_ps(_mm256_loadu_ps(b + i + 8), qv);
    a1 = _mm256_fmadd_ps(xa, xa, a1);
    b1 = _mm256_fmadd_ps(xb, xb, b1);
    qv = _mm256_loadu_ps(q + i + 16);
    xa = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), qv);
    xb = _mm256_sub_ps(_mm256_loadu_ps(b + i + 16), qv);
    a2 = _mm256_fmadd_ps(xa, xa, a2);
    b2 = _mm256_fmadd_ps(xb, xb, b2);
    qv = _mm256_loadu_ps(q + i + 24);
    xa = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), qv);
    xb = _mm256_sub_ps(_mm256_loadu_ps(b + i + 24), qv);
    a3 = _mm256_fmadd_ps(xa, xa, a3);
    b3 = _mm256_fmadd_ps(xb, xb, b3);
  }
  for (; i + 8 <= dim; i += 8) {
    __m256 qv = _mm256_loadu_ps(q + i);
    __m256 xa = _mm256_sub_ps(_mm256_loadu_ps(a + i), qv);
    __m256 xb = _mm256_sub_ps(_mm256_loadu_ps(b + i), qv);
    a0 = _mm256_fmadd_ps(xa, xa, a0);
    b0 = _mm256_fmadd_ps(xb, xb, b0);
  }
  out[0] = hsumAvx2Four(a0, a1, a2, a3);
  out[1] = hsumAvx2Four(b0, b1, b2, b3);
  for (; i < dim; i++) {
    float xa = a[i] - q[i], xb = b[i] - q[i];
    out[0] += xa * xa;
    out[1] += xb * xb;
  }
}

__attribute__((target("avx512f,fma"))) static float l2sqAvx512(
    const float* a, const float* b, int d) {
  __m512 s0 = _mm512_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
  int i = 0;
  for (; i + 64 <= d; i += 64) {
    __m512 x0 = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
    __m512 x1 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16));
    __m512 x2 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32));
    __m512 x3 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48));
    s0 = _mm512_fmadd_ps(x0, x0, s0);
    s1 = _mm512_fmadd_ps(x1, x1, s1);
    s2 = _mm512_fmadd_ps(x2, x2, s2);
    s3 = _mm512_fmadd_ps(x3, x3, s3);
  }
  for (; i + 16 <= d; i += 16) {
    __m512 x = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
    s0 = _mm512_fmadd_ps(x, x, s0);
  }
  if (i < d) {
    const __mmask16 mask = (__mmask16)((1u << (d - i)) - 1);
    __m512 x = _mm512_sub_ps(_mm512_maskz_loadu_ps(mask, a + i),
                             _mm512_maskz_loadu_ps(mask, b + i));
    s0 = _mm512_fmadd_ps(x, x, s0);
  }
  return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3)));
}
#endif

static bool g_useAvx2 = false;
#if MVDB_X86
static bool g_useAvx512 = false;
#endif
static bool g_init = false;

void initSimd() {
  if (g_init) return;
  g_init = true;
#if MVDB_X86
  g_useAvx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  g_useAvx512 = g_useAvx2 && __builtin_cpu_supports("avx512f");
#endif
}

double l2sqRef(const float* a, const float* b, int d) {
  double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  int i = 0;
  for (; i + 4 <= d; i += 4) {
    double x0 = (double)a[i] - (double)b[i];
    double x1 = (double)a[i + 1] - (double)b[i + 1];
    double x2 = (double)a[i + 2] - (double)b[i + 2];
    double x3 = (double)a[i + 3] - (double)b[i + 3];
    s0 += x0 * x0;
    s1 += x1 * x1;
    s2 += x2 * x2;
    s3 += x3 * x3;
  }
  for (; i < d; i++) {
    double x = (double)a[i] - (double)b[i];
    s0 += x * x;
  }
  return s0 + s1 + s2 + s3;
}

float l2sq(const float* a, const float* b, int d) {
#if MVDB_X86
  if (g_useAvx2) return l2sqAvx2(a, b, d);
#endif
  return l2sqScalar(a, b, d);
}


// --------------------------------------------------------------- top-k scan ---

namespace {

struct TopK {
  CandOut* best;
  int nb;
  int k;
  int desc;

  // True when (d, pk) must be ordered after (bd, bpk). Ties always break by
  // ascending primary key, whichever direction the distance runs.
  inline bool worse(double d, int64_t pk, double bd, int64_t bpk) const {
    if (d != bd) return desc ? (d < bd) : (d > bd);
    return pk > bpk;
  }

  inline void add(double d, int64_t pk, uint32_t rowid) {
    if (nb < k) {
      int i = nb++;
      while (i > 0 && worse(best[i - 1].d, best[i - 1].pk, d, pk)) {
        best[i] = best[i - 1];
        --i;
      }
      best[i] = CandOut{d, pk, rowid};
      return;
    }
    if (!worse(best[nb - 1].d, best[nb - 1].pk, d, pk)) return;
    int i = nb - 1;
    while (i > 0 && worse(best[i - 1].d, best[i - 1].pk, d, pk)) {
      best[i] = best[i - 1];
      --i;
    }
    best[i] = CandOut{d, pk, rowid};
  }
};

static inline void addScalar(TopK& top, const float* row, const float* query,
                             int dim, int64_t pk, uint32_t rowid) {
  top.add(l2sqRef(row, query, dim), pk, rowid);
}

#if MVDB_X86
template <float (*Estimate)(const float*, const float*, int)>
__attribute__((target("avx2"))) static inline void addBounded(
    TopK& top, const float* row, const float* query, int dim, int64_t pk, uint32_t rowid) {
  if (!top.desc && dim >= 32 && top.nb == top.k && top.best[top.k - 1].d <= FLT_MAX) {
    const float estimate = Estimate(row, query, dim);
    // All accumulated terms are nonnegative. Each term sees a rounded
    // subtraction, a square, and at most dim + 4 rounded additions. For the
    // supported dimensions (<= 65536), this margin also covers the much
    // smaller float64 rounding error. The additive term covers underflow.
    const double margin = 8.0 * (dim + 4) * FLT_EPSILON;
    const double underflow = 16.0 * (dim + 4) * FLT_MIN;
    const double cutoff = top.best[top.k - 1].d;
    if (estimate <= FLT_MAX && estimate > cutoff * (1.0 + margin) + underflow)
      return;
  }
  // Distances near the cutoff, including ties and overflow, are decided
  // with the same float64 calculation as the exact scan.
  top.add(l2sqExactAvx2(row, query, dim), pk, rowid);
}
#endif

#define MVDB_SCAN_RANGE_BODY(ADD)                                               \
  if (k <= 0) return 0;                                                         \
  TopK tk{out, 0, k, desc};                                                           \
  const size_t stride = (size_t)dim;                                            \
  for (uint64_t r = 0; r < n; r++) {                                            \
    if (!live[r]) continue;                                                     \
    __builtin_prefetch(base + (size_t)(r + 8) * stride);                        \
    ADD(tk, base + (size_t)r * stride, q, dim, pks[r], (uint32_t)r);             \
  }                                                                             \
  return tk.nb;

#define MVDB_SCAN_GATHER_BODY(ADD)                                              \
  if (k <= 0) return 0;                                                         \
  TopK tk{out, 0, k, desc};                                                           \
  const size_t stride = (size_t)dim;                                            \
  for (size_t i = 0; i < count; i++) {                                          \
    uint32_t r = rowids[i];                                                     \
    if (i + 4 < count) __builtin_prefetch(base + (size_t)rowids[i + 4] * stride);\
    ADD(tk, base + (size_t)r * stride, q, dim, pks[r], r);                        \
  }                                                                             \
  return tk.nb;

static int scanRangeScalar(const float* base, int dim, const uint8_t* live,
                           const int64_t* pks, uint64_t n, const float* q, int k, int desc,
                           CandOut* out) {
  MVDB_SCAN_RANGE_BODY(addScalar)
}

static int scanGatherScalar(const float* base, int dim, const uint32_t* rowids, size_t count,
                            const int64_t* pks, const float* q, int k, int desc, CandOut* out) {
  MVDB_SCAN_GATHER_BODY(addScalar)
}

#if MVDB_X86
__attribute__((target("avx2"))) static int scanRangeAvx2(const float* base, int dim,
                                                             const uint8_t* live,
                                                             const int64_t* pks, uint64_t n,
                                                             const float* q, int k, int desc,
                                                             CandOut* out) {
  if (g_useAvx512 && dim >= 64 && !desc && (double)n * dim > 1.5e6) {
    MVDB_SCAN_RANGE_BODY(addBounded<l2sqAvx512>)
  }
  MVDB_SCAN_RANGE_BODY(addBounded<l2sqAvx2>)
}

__attribute__((target("avx2"))) static int scanGatherAvx2(const float* base, int dim,
                                                              const uint32_t* rowids,
                                                              size_t count, const int64_t* pks,
                                                              const float* q, int k, int desc,
                                                              CandOut* out) {
  if (g_useAvx512 && dim >= 64 && !desc && (double)count * dim > 1.5e6) {
    MVDB_SCAN_GATHER_BODY(addBounded<l2sqAvx512>)
  }
  MVDB_SCAN_GATHER_BODY(addBounded<l2sqAvx2>)
}

__attribute__((target("avx2,fma"))) static inline void addEstimated(
    TopK& top, const float* row, const float* query, int dim, int64_t pk, uint32_t rowid,
    float estimate, double margin, double underflow) {
  const double cutoff = top.best[top.k - 1].d;
  if (estimate <= FLT_MAX && estimate > cutoff * (1.0 + margin) + underflow) return;
  top.add(l2sqExactAvx2(row, query, dim), pk, rowid);
}

__attribute__((target("avx2,fma"))) static int scanRangePaired(
    const float* base, int dim, const uint8_t* live, const int64_t* pks, uint64_t n,
    const float* q, int k, int desc, CandOut* out) {
  if (k <= 0) return 0;
  TopK tk{out, 0, k, desc};
  const size_t stride = (size_t)dim;
  const double margin = 8.0 * (dim + 4) * FLT_EPSILON;
  const double underflow = 16.0 * (dim + 4) * FLT_MIN;
  uint64_t r = 0;
  while (r < n) {
    if (tk.nb == tk.k && r + 2 <= n && live[r] && live[r + 1]) {
      if (r + 16 < n) __builtin_prefetch(base + (size_t)(r + 16) * stride);
      const float* row = base + (size_t)r * stride;
      float est[2];
      l2sqAvx2x2(q, row, row + stride, dim, est);
      for (int j = 0; j < 2; j++)
        addEstimated(tk, row + (size_t)j * stride, q, dim, pks[r + j], (uint32_t)(r + j),
                     est[j], margin, underflow);
      r += 2;
    } else {
      if (live[r]) addBounded<l2sqAvx2>(tk, base + (size_t)r * stride, q, dim, pks[r], (uint32_t)r);
      r++;
    }
  }
  return tk.nb;
}

__attribute__((target("avx2,fma"))) static int scanGatherPaired(
    const float* base, int dim, const uint32_t* rowids, size_t count, const int64_t* pks,
    const float* q, int k, int desc, CandOut* out) {
  if (k <= 0) return 0;
  TopK tk{out, 0, k, desc};
  const size_t stride = (size_t)dim;
  const double margin = 8.0 * (dim + 4) * FLT_EPSILON;
  const double underflow = 16.0 * (dim + 4) * FLT_MIN;
  size_t i = 0;
  while (i < count) {
    if (tk.nb == tk.k && i + 2 <= count) {
      if (i + 16 < count) __builtin_prefetch(base + (size_t)rowids[i + 16] * stride);
      const float* r0 = base + (size_t)rowids[i] * stride;
      const float* r1 = base + (size_t)rowids[i + 1] * stride;
      float est[2];
      l2sqAvx2x2(q, r0, r1, dim, est);
      addEstimated(tk, r0, q, dim, pks[rowids[i]], rowids[i], est[0], margin, underflow);
      addEstimated(tk, r1, q, dim, pks[rowids[i + 1]], rowids[i + 1], est[1], margin, underflow);
      i += 2;
    } else {
      uint32_t rowid = rowids[i];
      addBounded<l2sqAvx2>(tk, base + (size_t)rowid * stride, q, dim, pks[rowid], rowid);
      i++;
    }
  }
  return tk.nb;
}
#endif

}  // namespace

int scanTopKRange(const float* base, int dim, const uint8_t* live, const int64_t* pks,
                  uint64_t n, const float* q, int k, int desc, CandOut* out) {
#if MVDB_X86
  if (g_useAvx2 && !desc && dim >= 32 && (dim & 7) == 0 && (double)n * dim <= 1.5e6)
    return scanRangePaired(base, dim, live, pks, n, q, k, desc, out);
  if (g_useAvx2) return scanRangeAvx2(base, dim, live, pks, n, q, k, desc, out);
#endif
  return scanRangeScalar(base, dim, live, pks, n, q, k, desc, out);
}

int scanTopKGather(const float* base, int dim, const uint32_t* rowids, size_t count,
                   const int64_t* pks, const float* q, int k, int desc, CandOut* out) {
#if MVDB_X86
  if (g_useAvx2 && !desc && dim >= 32 && (dim & 7) == 0 && (double)count * dim <= 1.5e6)
    return scanGatherPaired(base, dim, rowids, count, pks, q, k, desc, out);
  if (g_useAvx2) return scanGatherAvx2(base, dim, rowids, count, pks, q, k, desc, out);
#endif
  return scanGatherScalar(base, dim, rowids, count, pks, q, k, desc, out);
}

int mergeTopK(const CandOut* const* parts, const int* counts, int nparts, int k, int desc,
              CandOut* out) {
  struct Item {
    double d;
    int64_t pk;
    uint32_t rowid;
    int part;
  };
  // "after" orders items worst-last, so the heap top is always the best one.
  auto after = [desc](const Item& a, const Item& b) {
    if (a.d != b.d) return desc ? (a.d < b.d) : (a.d > b.d);
    return a.pk > b.pk;
  };
  std::priority_queue<Item, std::vector<Item>, decltype(after)> heap(after);
  int idx[64];
  if (nparts > 64) nparts = 64;
  for (int p = 0; p < nparts; p++) {
    idx[p] = 0;
    if (counts[p] > 0) heap.push(Item{parts[p][0].d, parts[p][0].pk, parts[p][0].rowid, p});
  }
  int n = 0;
  while (!heap.empty() && n < k) {
    Item it = heap.top();
    heap.pop();
    out[n++] = CandOut{it.d, it.pk, it.rowid};
    int p = it.part;
    if (++idx[p] < counts[p]) {
      const CandOut& c = parts[p][idx[p]];
      heap.push(Item{c.d, c.pk, c.rowid, p});
    }
  }
  return n;
}

}  // namespace mvdb
