// Optional builder for the Radeon experiment. Search and snapshots use cuVS.
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/cagra.hpp>
#include "cagra_finish.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

extern "C" int seekdb_probe_nn_descent(const float *, unsigned, unsigned, unsigned, unsigned,
                                       uint32_t *, double *);
extern "C" int seekdb_fast_detour_counts(const uint32_t *, unsigned, unsigned, uint8_t *);
extern "C" int seekdb_fast_reverse(const uint32_t *, unsigned, unsigned, uint32_t *, uint32_t *);

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

extern "C" int seekdb_fast_cagra_build(cuvsResources_t resource, const float *base,
                                      unsigned rows, unsigned dimension, unsigned iterations,
                                      cuvsCagraIndex_t handle)
{
  // Also let the C API destroy the empty handle if construction throws.
  handle->dtype = {kDLFloat, 32, 1};
  try {
    double nnd_seconds = 0;
    double count_seconds = 0;
    double select_seconds = 0;
    double reverse_seconds = 0;
    std::vector<uint32_t> forward(static_cast<size_t>(rows) * OUTPUT_DEGREE);
    {
      const size_t edges = static_cast<size_t>(rows) * INPUT_DEGREE;
      std::vector<uint32_t> neighbors(edges);
      if (seekdb_probe_nn_descent(base, rows, dimension, INPUT_DEGREE, iterations,
                                  neighbors.data(), &nnd_seconds)) {
        throw std::runtime_error("NN-Descent build failed");
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
        if (!select_edges(neighbors.data() + row * INPUT_DEGREE,
                          counts.data() + row * INPUT_DEGREE,
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
    // The pinned C API stores a raft::resources pointer in cuvsResources_t.
    auto &res = *reinterpret_cast<raft::resources *>(resource);
    auto vectors = raft::make_host_matrix_view<const float, int64_t>(base, rows, dimension);
    auto graph = raft::make_host_matrix_view<const uint32_t, int64_t>(
        forward.data(), rows, OUTPUT_DEGREE);
    auto index = std::make_unique<cuvs::neighbors::cagra::index<float, uint32_t>>(
        res, cuvs::distance::DistanceType::L2Expanded, vectors, graph);
    handle->addr = reinterpret_cast<uintptr_t>(index.release());
    std::fprintf(stderr, "[fast CAGRA] nnd_s=%.6f count_s=%.6f select_s=%.6f reverse_s=%.6f "
                         "attach_s=%.6f\n",
                 nnd_seconds, count_seconds, select_seconds, reverse_seconds, since(start_attach));
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "[fast CAGRA] %s\n", error.what());
    return 1;
  }
}
