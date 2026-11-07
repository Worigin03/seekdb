#include "seekdb_cuvs_bridge.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static bool valid_results(const std::vector<unsigned int> &ids, long nq, long topk, long count)
{
  for (long query = 0; query < nq; ++query) {
    std::vector<bool> seen(count, false);
    for (long neighbor = 0; neighbor < topk; ++neighbor) {
      const unsigned int id = ids[query * topk + neighbor];
      if (id >= static_cast<unsigned long>(count) || seen[id]) return false;
      seen[id] = true;
    }
  }
  return true;
}

int main(int argc, char **argv)
{
  if (argc != 3) return 2;
  const long count = 10000;
  const long dimension = 32;
  std::vector<float> vectors(count * dimension);
  std::mt19937 rng(2026);
  for (float &value : vectors) {
    value = static_cast<float>(rng() >> 8) * (1.0f / 16777216.0f);
  }
  void *index = nullptr;
  if (std::strcmp(argv[1], "write") == 0) {
    index = seekdb_cuvs_build(vectors.data(), count, dimension);
    if (!index || seekdb_cuvs_serialize(index, argv[2]) != 0) return 3;
  } else {
    void *invalid = seekdb_cuvs_deserialize(argv[2], count - 1, dimension);
    if (invalid) { seekdb_cuvs_free(invalid); return 6; }
    invalid = seekdb_cuvs_deserialize(argv[2], count, dimension + 1);
    if (invalid) { seekdb_cuvs_free(invalid); return 7; }
    index = seekdb_cuvs_deserialize(argv[2], count, dimension);
    if (!index) return 4;
  }
  std::vector<unsigned int> ids(10);
  std::vector<float> distances(10);
  int result = seekdb_cuvs_search(index, vectors.data(), 1, 10, ids.data(), distances.data());
  if (result != 0 || ids[0] != 0 || std::fabs(distances[0]) > 0.0001f ||
      !valid_results(ids, 1, 10, count)) {
    seekdb_cuvs_free(index);
    return 5;
  }
  std::printf("%s: n=%ld top1=%u distance=%g\n", argv[1], count, ids[0], distances[0]);

  // Exercise growth and subsequent reuse of the cached query/result buffers.
  ids.resize(4 * 10);
  distances.resize(4 * 10);
  result = seekdb_cuvs_search(index, vectors.data(), 4, 10, ids.data(), distances.data());
  if (result != 0 || !valid_results(ids, 4, 10, count)) {
    seekdb_cuvs_free(index);
    return 8;
  }
  ids.resize(2 * 100);
  distances.resize(2 * 100);
  result = seekdb_cuvs_search(index, vectors.data(), 2, 100, ids.data(), distances.data());
  if (result != 0 || !valid_results(ids, 2, 100, count)) {
    seekdb_cuvs_free(index);
    return 9;
  }
  ids.resize(10);
  distances.resize(10);
  result = seekdb_cuvs_search(index, vectors.data(), 1, 10, ids.data(), distances.data());
  seekdb_cuvs_free(index);
  if (result != 0 || ids[0] != 0 || std::fabs(distances[0]) > 0.0001f ||
      !valid_results(ids, 1, 10, count)) return 10;
  return 0;
}
