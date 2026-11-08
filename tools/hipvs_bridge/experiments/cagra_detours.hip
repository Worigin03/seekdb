// Standalone experiment for CAGRA's rank-based two-hop edge counts.
#include <hip/hip_runtime.h>
#ifdef SEEKDB_IVF_FLAT_BUILD
#include <hip/hip_fp16.h>
#endif
#include <cstdint>

namespace {
constexpr unsigned MAX_DEGREE = 256;
constexpr unsigned HASH_CAPACITY = 2 * MAX_DEGREE;
constexpr uint32_t EMPTY = UINT32_MAX;

__device__ unsigned hash_slot(uint32_t id)
{
  return ((id ^ (id >> 16)) * 2654435761u) >> 23;
}

__global__ void count_detours(const uint32_t *graph, uint8_t *output, unsigned rows,
                              unsigned degree, bool use_hash)
{
  __shared__ uint32_t keys[HASH_CAPACITY];
  __shared__ unsigned ranks[HASH_CAPACITY];
  __shared__ unsigned counts[MAX_DEGREE];
  __shared__ unsigned needs_scan;

  const size_t row = blockIdx.x;
  const uint32_t *direct = graph + row * degree;
  for (unsigned rank = threadIdx.x; rank < degree; rank += blockDim.x) {
    counts[rank] = 0;
  }
  if (use_hash) {
    for (unsigned slot = threadIdx.x; slot < HASH_CAPACITY; slot += blockDim.x) {
      keys[slot] = EMPTY;
    }
    if (threadIdx.x == 0) needs_scan = 0;
    __syncthreads();

    for (unsigned rank = threadIdx.x; rank < degree; rank += blockDim.x) {
      const uint32_t id = direct[rank];
      if (id >= rows) {
        atomicExch(&needs_scan, 1u);
        continue;
      }
      unsigned slot = hash_slot(id);
      while (true) {
        const uint32_t previous = atomicCAS(keys + slot, EMPTY, id);
        if (previous == EMPTY) {
          ranks[slot] = rank;
          break;
        }
        if (previous == id) {
          // The reference selects the first occurrence *after* the via rank.
          atomicExch(&needs_scan, 1u);
          break;
        }
        slot = (slot + 1) & (HASH_CAPACITY - 1);
      }
    }
  }
  __syncthreads();

  const bool use_lookup = use_hash && needs_scan == 0;
  for (unsigned via_rank = 0; via_rank + 1 < degree; ++via_rank) {
    const uint32_t via_id = direct[via_rank];
    if (via_id < rows) {
      const uint32_t *next = graph + static_cast<size_t>(via_id) * degree;
      for (unsigned next_rank = threadIdx.x; next_rank < degree; next_rank += blockDim.x) {
        const uint32_t target = next[next_rank];
        if (use_lookup) {
          unsigned slot = hash_slot(target);
          while (keys[slot] != EMPTY && keys[slot] != target) {
            slot = (slot + 1) & (HASH_CAPACITY - 1);
          }
          if (keys[slot] != EMPTY && ranks[slot] > via_rank) {
            atomicAdd(counts + ranks[slot], 1u);
          }
        } else {
          for (unsigned rank = via_rank + 1; rank < degree; ++rank) {
            if (direct[rank] == target) {
              atomicAdd(counts + rank, 1u);
              break;
            }
          }
        }
      }
    }
  }
  __syncthreads();

  for (unsigned rank = threadIdx.x; rank < degree; rank += blockDim.x) {
    output[row * degree + rank] = static_cast<uint8_t>(counts[rank] > 255 ? 255 : counts[rank]);
  }
}

__global__ void reverse_rank(const uint32_t *forward, uint32_t *reverse, uint32_t *counts,
                             unsigned rows, unsigned degree, unsigned rank)
{
  const size_t source = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (source < rows) {
    const uint32_t destination = forward[source * degree + rank];
    if (destination < rows) {
      const unsigned position = atomicAdd(counts + destination, 1u);
      if (position < degree) {
        reverse[static_cast<size_t>(destination) * degree + position] =
            static_cast<uint32_t>(source);
      }
    }
  }
}
}  // namespace

hipError_t launch_detours(const uint32_t *graph, uint8_t *output, unsigned rows,
                          unsigned degree, unsigned threads, bool use_hash)
{
  if (rows == 0 || degree == 0 || degree > MAX_DEGREE || threads == 0) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(count_detours, dim3(rows), dim3(threads), 0, nullptr,
                     graph, output, rows, degree, use_hash);
  return hipGetLastError();
}

hipError_t launch_reverse(const uint32_t *forward, uint32_t *reverse, uint32_t *counts,
                          unsigned rows, unsigned degree)
{
  if (rows == 0 || degree == 0 || degree > MAX_DEGREE) return hipErrorInvalidValue;
  constexpr unsigned THREADS = 256;
  const unsigned blocks = 1 + (rows - 1) / THREADS;
  // Launch ranks in order so a lower forward rank keeps priority when the
  // destination row has more incoming edges than fit in the reverse graph.
  for (unsigned rank = 0; rank < degree; ++rank) {
    hipLaunchKernelGGL(reverse_rank, dim3(blocks), dim3(THREADS), 0, nullptr,
                       forward, reverse, counts, rows, degree, rank);
    const auto status = hipGetLastError();
    if (status != hipSuccess) return status;
  }
  return hipSuccess;
}

#ifdef SEEKDB_IVF_FLAT_BUILD
namespace {
__global__ void convert_ivf_vectors(const float *source, __half *destination, size_t count)
{
  const size_t offset = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (offset < count) {
    destination[offset] = __float2half(source[offset]);
  }
}
}  // namespace

hipError_t launch_ivf_half(const float *source, void *destination, size_t count)
{
  constexpr unsigned THREADS = 256;
  const auto blocks = static_cast<unsigned>((count + THREADS - 1) / THREADS);
  hipLaunchKernelGGL(convert_ivf_vectors, dim3(blocks), dim3(THREADS), 0, nullptr,
                     source, static_cast<__half *>(destination), count);
  return hipGetLastError();
}
#endif
