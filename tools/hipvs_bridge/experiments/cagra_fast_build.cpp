// Assemble a CAGRA graph for the AMD bridge. Search and snapshots use cuVS.
#include <cuvs/neighbors/cagra.h>
#include "cagra_finish.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

extern "C" int seekdb_probe_nn_descent(const float *, unsigned, unsigned, unsigned, unsigned,
                                      uint32_t *, double *);
extern "C" int seekdb_fast_detour_counts(const uint32_t *, unsigned, unsigned, uint8_t *);
extern "C" int seekdb_fast_reverse(const uint32_t *, unsigned, unsigned, uint32_t *, uint32_t *);
#ifdef SEEKDB_IVF_FLAT_BUILD
extern "C" int seekdb_ivf_flat_graph(const float *, unsigned, unsigned, unsigned,
                                     uint32_t *, double *);
#endif

namespace {
using namespace cagra_experiment;
using Clock = std::chrono::steady_clock;

double since(Clock::time_point start)
{
  return std::chrono::duration<double>(Clock::now() - start).count();
}

template <typename Function>
void for_rows(unsigned rows, Function function)
{
  const unsigned threads = std::min(24u, std::max(1u, std::thread::hardware_concurrency()));
  std::vector<std::thread> workers;
  workers.reserve(threads - 1);
  auto run = [&](unsigned part) {
    const size_t first = static_cast<uint64_t>(rows) * part / threads;
    const size_t last = static_cast<uint64_t>(rows) * (part + 1) / threads;
    for (size_t row = first; row < last; ++row) function(row);
  };
  try {
    for (unsigned part = 1; part < threads; ++part) workers.emplace_back(run, part);
    run(0);
  } catch (...) {
    for (auto &worker : workers) worker.join();
    throw;
  }
  for (auto &worker : workers) worker.join();
}
}  // namespace

extern "C" int seekdb_fast_cagra_build(cuvsResources_t resource, DLManagedTensor *dataset,
                                      unsigned iterations,
                                      cuvsCagraIndex_t handle)
{
  // Also let the C API destroy the empty handle if construction throws.
  handle->dtype = dataset->dl_tensor.dtype;
  try {
    const auto rows = static_cast<unsigned>(dataset->dl_tensor.shape[0]);
    const auto dimension = static_cast<unsigned>(dataset->dl_tensor.shape[1]);
    const auto *base = static_cast<const float *>(dataset->dl_tensor.data);
    double candidate_seconds = 0;
    double count_seconds = 0;
    double select_seconds = 0;
    double reverse_seconds = 0;
    bool ivf_built = false;
    std::vector<uint32_t> forward(static_cast<size_t>(rows) * OUTPUT_DEGREE);
    {
      const size_t edges = static_cast<size_t>(rows) * INPUT_DEGREE;
      std::vector<uint32_t> neighbors(edges);
#ifdef SEEKDB_IVF_FLAT_BUILD
      // The 512 IVF lists need enough training points. Keep the FP32 input
      // intact and use NN-Descent when FP16 conversion would overflow.
      if (rows >= 8192 &&
          std::all_of(base, base + static_cast<size_t>(rows) * dimension, [](float value) {
            return value >= -65504.0f && value <= 65504.0f;
          })) {
        ivf_built = seekdb_ivf_flat_graph(base, rows, dimension, INPUT_DEGREE, neighbors.data(),
                                         &candidate_seconds) == 0;
      }
#endif
      if (!ivf_built &&
          seekdb_probe_nn_descent(base, rows, dimension, INPUT_DEGREE, iterations, neighbors.data(),
                                 &candidate_seconds) != 0) {
        throw std::runtime_error("intermediate graph build failed");
      }

      const auto start_count = Clock::now();
      std::vector<uint8_t> counts(edges);
      if (seekdb_fast_detour_counts(neighbors.data(), rows, INPUT_DEGREE, counts.data())) {
        throw std::runtime_error("detour counting failed");
      }
      count_seconds = since(start_count);

      const auto start_select = Clock::now();
      std::atomic<bool> valid{true};
      for_rows(rows, [&](size_t row) {
        if (!select_edges(neighbors.data() + row * INPUT_DEGREE, counts.data() + row * INPUT_DEGREE,
                          forward.data() + row * OUTPUT_DEGREE, rows)) {
          valid.store(false, std::memory_order_relaxed);
        }
      });
      if (!valid.load(std::memory_order_relaxed)) {
        throw std::runtime_error("a row has fewer than 128 valid distinct neighbors");
      }
      select_seconds = since(start_select);
    }

    const auto start_reverse = Clock::now();
    {
      std::vector<uint32_t> reverse(forward.size());
      std::vector<uint32_t> counts(rows);
      if (seekdb_fast_reverse(forward.data(), rows, OUTPUT_DEGREE, reverse.data(), counts.data())) {
        throw std::runtime_error("reverse graph build failed");
      }
      for_rows(rows, [&](size_t row) {
        merge_reverse(forward.data() + row * OUTPUT_DEGREE,
                      reverse.data() + row * OUTPUT_DEGREE, counts[row]);
      });
    }
    reverse_seconds = since(start_reverse);

    const auto start_attach = Clock::now();
    int64_t shape[]{rows, OUTPUT_DEGREE};
    DLManagedTensor graph{};
    graph.dl_tensor.data = forward.data();
    graph.dl_tensor.device = {kDLCPU, 0};
    graph.dl_tensor.ndim = 2;
    graph.dl_tensor.dtype = {kDLUInt, 32, 1};
    graph.dl_tensor.shape = shape;
    // The pinned constructor copies both host tensors to index-owned storage.
    if (cuvsCagraIndexFromArgs(resource, L2Expanded, &graph, dataset, handle) != CUVS_SUCCESS) {
      const char *detail = cuvsGetLastErrorText();
      throw std::runtime_error(detail ? detail : "could not attach the CAGRA graph");
    }
    std::fprintf(stderr, "[fast CAGRA] builder=%s candidates_s=%.6f count_s=%.6f select_s=%.6f reverse_s=%.6f "
                         "attach_s=%.6f\n",
                 ivf_built ? "ivf_flat_fp16" : "nn_descent", candidate_seconds, count_seconds,
                 select_seconds, reverse_seconds, since(start_attach));
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "[fast CAGRA] %s\n", error.what());
    return 1;
  }
}
