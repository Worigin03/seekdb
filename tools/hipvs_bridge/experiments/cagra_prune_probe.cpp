// Diagnostic bridge. It does not participate in seekdb index construction.
#include <hip/hip_runtime.h>
#include <cuvs/neighbors/nn_descent.h>
#ifdef SEEKDB_IVF_FLAT_BUILD
#include <cuvs/neighbors/ivf_flat.h>
#include <algorithm>
#endif

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

hipError_t launch_detours(const uint32_t *, uint8_t *, unsigned, unsigned, unsigned, bool);
hipError_t launch_reverse(const uint32_t *, uint32_t *, uint32_t *, unsigned, unsigned);
#ifdef SEEKDB_IVF_FLAT_BUILD
hipError_t launch_ivf_half(const float *, void *, size_t);
#endif

namespace {
using Clock = std::chrono::steady_clock;

void require(hipError_t status, const char *operation)
{
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
  }
}

void require(cuvsError_t status, const char *operation)
{
  if (status != CUVS_SUCCESS) {
    const char *detail = cuvsGetLastErrorText();
    throw std::runtime_error(std::string(operation) + ": " + (detail ? detail : "unknown error"));
  }
}

template <typename T>
struct DeviceBuffer {
  T *data = nullptr;
  explicit DeviceBuffer(size_t elements)
  {
    require(hipMalloc(reinterpret_cast<void **>(&data), elements * sizeof(T)), "hipMalloc");
  }
  ~DeviceBuffer() { (void)hipFree(data); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

struct Event {
  hipEvent_t event = nullptr;
  Event() { require(hipEventCreate(&event), "hipEventCreate"); }
  ~Event() { (void)hipEventDestroy(event); }
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;
};

struct NNDescent {
  cuvsResources_t resources = 0;
  cuvsNNDescentIndexParams_t params = nullptr;
  cuvsNNDescentIndex_t index = nullptr;
  ~NNDescent()
  {
    if (index) cuvsNNDescentIndexDestroy(index);
    if (params) cuvsNNDescentIndexParamsDestroy(params);
    if (resources) cuvsResourcesDestroy(resources);
  }
};

DLManagedTensor host_matrix(void *data, int64_t *shape, uint8_t type)
{
  DLManagedTensor result{};
  result.dl_tensor.data = data;
  result.dl_tensor.device = {kDLCPU, 0};
  result.dl_tensor.ndim = 2;
  result.dl_tensor.dtype = {type, 32, 1};
  result.dl_tensor.shape = shape;
  return result;
}

float time_detours(const uint32_t *graph, uint8_t *output, unsigned rows, unsigned degree,
                   unsigned threads, bool use_hash, const Event &start, const Event &stop)
{
  require(hipEventRecord(start.event, nullptr), "start kernel timer");
  require(launch_detours(graph, output, rows, degree, threads, use_hash), "launch detour kernel");
  require(hipEventRecord(stop.event, nullptr), "stop kernel timer");
  require(hipEventSynchronize(stop.event), "wait for detour kernel");
  float elapsed_ms = 0;
  require(hipEventElapsedTime(&elapsed_ms, start.event, stop.event), "read kernel timer");
  return elapsed_ms;
}
}  // namespace

extern "C" int seekdb_probe_nn_descent(const float *vectors, unsigned rows, unsigned dimension,
                                      unsigned degree, unsigned iterations, uint32_t *graph,
                                      double *seconds)
{
  try {
    NNDescent build;
    require(cuvsResourcesCreate(&build.resources), "create cuVS resources");
    require(cuvsNNDescentIndexParamsCreate(&build.params), "create NN-Descent parameters");
    build.params->metric = L2Expanded;
    build.params->graph_degree = degree;
    build.params->intermediate_graph_degree = degree * 3 / 2;
    build.params->max_iterations = iterations;
    build.params->termination_threshold = 0.0001f;
    build.params->return_distances = false;
    int64_t vector_shape[]{rows, dimension};
    int64_t graph_shape[]{rows, degree};
    auto data_tensor = host_matrix(const_cast<float *>(vectors), vector_shape, kDLFloat);
    auto graph_tensor = host_matrix(graph, graph_shape, kDLUInt);
    require(cuvsNNDescentIndexCreate(&build.index), "create NN-Descent index");
    const auto start = Clock::now();
    require(cuvsNNDescentBuild(build.resources, build.params, &data_tensor, &graph_tensor,
                              build.index), "build NN-Descent graph");
    require(cuvsStreamSync(build.resources), "wait for NN-Descent graph");
    *seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "[prune probe] %s\n", error.what());
    return 1;
  }
}

#ifdef SEEKDB_IVF_FLAT_BUILD
extern "C" int seekdb_ivf_flat_graph(const float *vectors, unsigned rows, unsigned dimension,
                                     unsigned degree, uint32_t *graph, double *seconds)
{
  constexpr unsigned LISTS = 512;
  constexpr unsigned PROBES = 40;
  constexpr unsigned BATCH = 512;
  struct IvfFlat {
    cuvsResources_t resource = 0;
    cuvsIvfFlatIndexParams_t params = nullptr;
    cuvsIvfFlatSearchParams_t search = nullptr;
    cuvsIvfFlatIndex_t index = nullptr;
    ~IvfFlat()
    {
      if (index) cuvsIvfFlatIndexDestroy(index);
      if (search) cuvsIvfFlatSearchParamsDestroy(search);
      if (params) cuvsIvfFlatIndexParamsDestroy(params);
      if (resource) cuvsResourcesDestroy(resource);
    }
  };
  try {
    const auto start = Clock::now();
    int device = 0;
    require(hipGetDevice(&device), "get active GPU");
    const size_t elements = static_cast<size_t>(rows) * dimension;
    DeviceBuffer<uint16_t> dataset(elements);
    {
      constexpr size_t CONVERSION_ELEMENTS = 1 << 20;
      DeviceBuffer<float> staging(std::min(elements, CONVERSION_ELEMENTS));
      for (size_t offset = 0; offset < elements; offset += CONVERSION_ELEMENTS) {
        const size_t count = std::min(elements - offset, CONVERSION_ELEMENTS);
        require(hipMemcpy(staging.data, vectors + offset, count * sizeof(float),
                          hipMemcpyHostToDevice), "upload FP32 chunk");
        require(launch_ivf_half(staging.data, dataset.data + offset, count),
                "convert IVF-Flat chunk");
      }
      require(hipDeviceSynchronize(), "wait for IVF-Flat conversion");
    }
    auto device_matrix = [&](void *data, int64_t *shape, uint8_t type, uint8_t bits = 32) {
      auto tensor = host_matrix(data, shape, type);
      tensor.dl_tensor.device = {kDLCUDA, device};
      tensor.dl_tensor.dtype.bits = bits;
      return tensor;
    };
    int64_t shape[]{rows, dimension};
    auto data_tensor = device_matrix(dataset.data, shape, kDLFloat, 16);
    IvfFlat build;
    require(cuvsResourcesCreate(&build.resource), "create IVF-Flat resources");
    require(cuvsIvfFlatIndexParamsCreate(&build.params), "create IVF-Flat parameters");
    require(cuvsIvfFlatSearchParamsCreate(&build.search), "create IVF-Flat search parameters");
    require(cuvsIvfFlatIndexCreate(&build.index), "create IVF-Flat index");
    build.params->metric = L2Expanded;
    build.params->n_lists = LISTS;
    build.params->kmeans_n_iters = 10;
    build.params->kmeans_trainset_fraction = 0.125;
    build.params->add_data_on_build = true;
    build.params->conservative_memory_allocation = true;
    build.search->n_probes = PROBES;
    require(cuvsIvfFlatBuild(build.resource, build.params, &data_tensor, build.index),
            "build IVF-Flat");

    const unsigned k = degree + 1;  // One additional result to remove the query itself.
    DeviceBuffer<int64_t> neighbors(BATCH * k);
    DeviceBuffer<float> distances(BATCH * k);
    std::vector<int64_t> ids(BATCH * k);
    cuvsFilter filter{};
    filter.type = NO_FILTER;
    for (size_t start_row = 0; start_row < rows; start_row += BATCH) {
      const auto count = std::min<size_t>(BATCH, rows - start_row);
      int64_t query_shape[]{static_cast<int64_t>(count), dimension};
      int64_t result_shape[]{static_cast<int64_t>(count), k};
      auto queries = device_matrix(dataset.data + start_row * dimension, query_shape, kDLFloat, 16);
      auto results = device_matrix(neighbors.data, result_shape, kDLInt, 64);
      auto dists = device_matrix(distances.data, result_shape, kDLFloat);
      require(cuvsIvfFlatSearch(build.resource, build.search, build.index, &queries, &results,
                                &dists, filter), "search IVF-Flat for graph candidates");
      require(cuvsStreamSync(build.resource), "wait for IVF-Flat candidates");
      require(hipMemcpy(ids.data(), neighbors.data, count * k * sizeof(int64_t),
                        hipMemcpyDeviceToHost), "read IVF-Flat candidates");
      for (size_t row = 0; row < count; ++row) {
        auto *edges = graph + (start_row + row) * degree;
        unsigned kept = 0;
        for (unsigned rank = 0; rank < k && kept < degree; ++rank) {
          const auto id = ids[row * k + rank];
          if (id >= 0 && id < rows && static_cast<size_t>(id) != start_row + row) {
            edges[kept++] = static_cast<uint32_t>(id);
          }
        }
        if (kept != degree) throw std::runtime_error("IVF-Flat did not fill a graph row");
      }
    }
    *seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "[fast CAGRA] %s\n", error.what());
    return 1;
  }
}
#endif

extern "C" int seekdb_probe_detours(const uint32_t *graph, unsigned rows, unsigned degree,
                                   uint8_t *reference, uint8_t *optimized, float *milliseconds)
{
  try {
    const size_t elements = static_cast<size_t>(rows) * degree;
    DeviceBuffer<uint32_t> input(elements);
    DeviceBuffer<uint8_t> output(elements);
    require(hipMemcpy(input.data, graph, elements * sizeof(uint32_t), hipMemcpyHostToDevice),
            "copy input graph");

    int device = 0;
    hipDeviceProp_t properties{};
    require(hipGetDevice(&device), "get active GPU");
    require(hipGetDeviceProperties(&properties, device), "get GPU wave size");
    Event start, stop;
    // The stock prune kernel uses one wave per row. Also check a wider hash block.
    const auto wave = static_cast<unsigned>(properties.warpSize);
    milliseconds[0] = time_detours(input.data, output.data, rows, degree, wave, false, start, stop);
    require(hipMemcpy(reference, output.data, elements, hipMemcpyDeviceToHost), "read linear counts");
    unsigned measurement = 1;
    for (const unsigned threads : {wave, 128u}) {
      milliseconds[measurement++] =
          time_detours(input.data, output.data, rows, degree, threads, true, start, stop);
      require(hipMemcpy(optimized, output.data, elements, hipMemcpyDeviceToHost), "read hash counts");
      for (size_t i = 0; i < elements; ++i) {
        if (reference[i] != optimized[i]) {
          std::fprintf(stderr, "[prune probe] hash mismatch (%u threads) at row %zu rank %zu: %u != %u\n",
                       threads, i / degree, i % degree, reference[i], optimized[i]);
          return 2;
        }
      }
    }
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "[prune probe] %s\n", error.what());
    return 1;
  }
}

extern "C" int seekdb_fast_detour_counts(const uint32_t *graph, unsigned rows, unsigned degree,
                                        uint8_t *counts)
{
  try {
    const size_t elements = static_cast<size_t>(rows) * degree;
    DeviceBuffer<uint32_t> input(elements);
    DeviceBuffer<uint8_t> output(elements);
    require(hipMemcpy(input.data, graph, elements * sizeof(uint32_t), hipMemcpyHostToDevice),
            "copy input graph");
    require(launch_detours(input.data, output.data, rows, degree, 128, true), "count detours");
    require(hipMemcpy(counts, output.data, elements, hipMemcpyDeviceToHost), "read detour counts");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "[fast CAGRA] %s\n", error.what());
    return 1;
  }
}

extern "C" int seekdb_fast_reverse(const uint32_t *forward, unsigned rows, unsigned degree,
                                  uint32_t *reverse, uint32_t *counts)
{
  try {
    const size_t elements = static_cast<size_t>(rows) * degree;
    DeviceBuffer<uint32_t> input(elements);
    DeviceBuffer<uint32_t> output(elements);
    DeviceBuffer<uint32_t> sizes(rows);
    require(hipMemcpy(input.data, forward, elements * sizeof(uint32_t), hipMemcpyHostToDevice),
            "copy pruned graph");
    require(hipMemset(output.data, 0xff, elements * sizeof(uint32_t)), "initialize reverse rows");
    require(hipMemset(sizes.data, 0, rows * sizeof(uint32_t)), "zero reverse degrees");
    // Only the first min(count, degree) entries of each reverse row are read.
    require(launch_reverse(input.data, output.data, sizes.data, rows, degree), "build reverse graph");
    require(hipMemcpy(reverse, output.data, elements * sizeof(uint32_t), hipMemcpyDeviceToHost),
            "read reverse graph");
    require(hipMemcpy(counts, sizes.data, rows * sizeof(uint32_t), hipMemcpyDeviceToHost),
            "read reverse degrees");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "[fast CAGRA] %s\n", error.what());
    return 1;
  }
}
