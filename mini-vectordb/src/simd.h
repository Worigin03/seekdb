// Distance kernels with runtime CPU dispatch.
#pragma once

#include <cstdint>
#include <cmath>

namespace mvdb {

void initSimd();
// Squared L2 distance between two float32 vectors of length d.
float l2sq(const float* a, const float* b, int d);
// Squared L2 accumulated in float64 for exact ranking and distance projection.
double l2sqRef(const float* a, const float* b, int d);

struct CandOut {
  double d;
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

// Exact scans that first use a per-row 8-bit sketch as a conservative lower
// bound. `qbase` holds dim bytes per row and `qsafe[r] != 0` means every
// component of row r is inside the sketch's representable interval. Rows
// outside that interval, and all queries outside it, fall back to the exact
// kernel. Results are bit-for-bit identical to scanTopKRange/Gather.
int scanTopKRangeQuant(const float* base, int dim, const uint8_t* qbase,
                       const uint8_t* qsafe, const uint8_t* live, const int64_t* pks,
                       uint64_t n, const float* q, const uint8_t* qquant, int k,
                       int desc, CandOut* out);
int scanTopKGatherQuant(const float* base, int dim, const uint8_t* qbase,
                        const uint8_t* qsafe, const uint32_t* rowids, size_t count,
                        const int64_t* pks, const float* q, const uint8_t* qquant, int k,
                        int desc, CandOut* out);

// Merges per-chunk result lists (each already ordered best-first by the same
// comparator) into the global best k. Returns the number written to `out`.
int mergeTopK(const CandOut* const* parts, const int* counts, int nparts, int k, int desc,
              CandOut* out);

}  // namespace mvdb
