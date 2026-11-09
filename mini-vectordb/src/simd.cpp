#include "simd.h"

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
__attribute__((target("avx2,fma"))) static float l2sqAvx2(const float* a, const float* b, int d) {
  __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
  int i = 0;
  for (; i + 16 <= d; i += 16) {
    __m256 x0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    __m256 x1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
    s0 = _mm256_fmadd_ps(x0, x0, s0);
    s1 = _mm256_fmadd_ps(x1, x1, s1);
  }
  for (; i + 8 <= d; i += 8) {
    __m256 x = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    s0 = _mm256_fmadd_ps(x, x, s0);
  }
  s0 = _mm256_add_ps(s0, s1);
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
#endif

static bool g_useAvx2 = false;
static bool g_init = false;

void initSimd() {
  if (g_init) return;
  g_init = true;
#if MVDB_X86
  g_useAvx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
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
  inline bool worse(float d, int64_t pk, float bd, int64_t bpk) const {
    if (d != bd) return desc ? (d < bd) : (d > bd);
    return pk > bpk;
  }

  inline void add(float d, int64_t pk, uint32_t rowid) {
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

#define MVDB_SCAN_RANGE_BODY(DIST)                                              \
  if (k <= 0) return 0;                                                         \
  TopK tk{out, 0, k, desc};                                                           \
  const size_t stride = (size_t)dim;                                            \
  for (uint64_t r = 0; r < n; r++) {                                            \
    if (!live[r]) continue;                                                     \
    __builtin_prefetch(base + (size_t)(r + 8) * stride);                        \
    float d = DIST(base + (size_t)r * stride, q, dim);                          \
    tk.add(d, pks[r], (uint32_t)r);                                             \
  }                                                                             \
  return tk.nb;

#define MVDB_SCAN_GATHER_BODY(DIST)                                             \
  if (k <= 0) return 0;                                                         \
  TopK tk{out, 0, k, desc};                                                           \
  const size_t stride = (size_t)dim;                                            \
  for (size_t i = 0; i < count; i++) {                                          \
    uint32_t r = rowids[i];                                                     \
    if (i + 4 < count) __builtin_prefetch(base + (size_t)rowids[i + 4] * stride);\
    float d = DIST(base + (size_t)r * stride, q, dim);                          \
    tk.add(d, pks[r], r);                                                       \
  }                                                                             \
  return tk.nb;

static int scanRangeScalar(const float* base, int dim, const uint8_t* live,
                           const int64_t* pks, uint64_t n, const float* q, int k, int desc,
                           CandOut* out) {
  MVDB_SCAN_RANGE_BODY(l2sqScalar)
}

static int scanGatherScalar(const float* base, int dim, const uint32_t* rowids, size_t count,
                            const int64_t* pks, const float* q, int k, int desc, CandOut* out) {
  MVDB_SCAN_GATHER_BODY(l2sqScalar)
}

#if MVDB_X86
__attribute__((target("avx2,fma"))) static int scanRangeAvx2(const float* base, int dim,
                                                             const uint8_t* live,
                                                             const int64_t* pks, uint64_t n,
                                                             const float* q, int k, int desc,
                                                             CandOut* out) {
  MVDB_SCAN_RANGE_BODY(l2sqAvx2)
}

__attribute__((target("avx2,fma"))) static int scanGatherAvx2(const float* base, int dim,
                                                              const uint32_t* rowids,
                                                              size_t count, const int64_t* pks,
                                                              const float* q, int k, int desc,
                                                              CandOut* out) {
  MVDB_SCAN_GATHER_BODY(l2sqAvx2)
}
#endif

}  // namespace

int scanTopKRange(const float* base, int dim, const uint8_t* live, const int64_t* pks,
                  uint64_t n, const float* q, int k, int desc, CandOut* out) {
#if MVDB_X86
  if (g_useAvx2) return scanRangeAvx2(base, dim, live, pks, n, q, k, desc, out);
#endif
  return scanRangeScalar(base, dim, live, pks, n, q, k, desc, out);
}

int scanTopKGather(const float* base, int dim, const uint32_t* rowids, size_t count,
                   const int64_t* pks, const float* q, int k, int desc, CandOut* out) {
#if MVDB_X86
  if (g_useAvx2) return scanGatherAvx2(base, dim, rowids, count, pks, q, k, desc, out);
#endif
  return scanGatherScalar(base, dim, rowids, count, pks, q, k, desc, out);
}

}  // namespace mvdb
