// Distance kernels with runtime CPU dispatch.
#pragma once

#include <cstdint>
#include <cmath>

namespace mvdb {

void initSimd();
// Squared L2 distance between two float32 vectors of length d.
float l2sq(const float* a, const float* b, int d);
// Same value accumulated in float64: used when materialising result rows so the
// printed distance matches a double-precision reference.
double l2sqRef(const float* a, const float* b, int d);

struct CandOut {
  float d;
  int64_t pk;
  uint32_t rowid;
};

// Exact top-k over rows [0, n) that are marked live. Results are written to
// `out` (capacity k) ordered by (distance, pk); `desc` keeps the largest
// distances instead. Returns the number of results.
int scanTopKRange(const float* base, int dim, const uint8_t* live, const int64_t* pks,
                  uint64_t n, const float* q, int k, int desc, CandOut* out);

// Same, but over an explicit row-id list (used after a filter pass).
int scanTopKGather(const float* base, int dim, const uint32_t* rowids, size_t count,
                   const int64_t* pks, const float* q, int k, int desc, CandOut* out);

// Merges per-chunk result lists (each already ordered best-first by the same
// comparator) into the global best k. Returns the number written to `out`.
int mergeTopK(const CandOut* const* parts, const int* counts, int nparts, int k, int desc,
              CandOut* out);

}  // namespace mvdb
