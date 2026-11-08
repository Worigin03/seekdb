#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

namespace cagra_experiment {
constexpr unsigned INPUT_DEGREE = 256;
constexpr unsigned OUTPUT_DEGREE = 128;

// Stable ordering by (detour count, original nearest-neighbor rank).
inline bool select_edges(const uint32_t *input, const uint8_t *counts, uint32_t *output,
                         unsigned rows)
{
  std::array<uint16_t, INPUT_DEGREE> starts{};
  std::array<uint16_t, INPUT_DEGREE> ranked{};
  for (unsigned rank = 0; rank < INPUT_DEGREE; ++rank) {
    ++starts[counts[rank]];
  }
  uint16_t offset = 0;
  for (auto &start : starts) {
    const auto count = start;
    start = offset;
    offset += count;
  }
  for (unsigned rank = 0; rank < INPUT_DEGREE; ++rank) {
    ranked[starts[counts[rank]]++] = rank;
  }

  std::array<uint32_t, 2 * OUTPUT_DEGREE> selected;
  selected.fill(UINT32_MAX);
  unsigned size = 0;
  for (const auto rank : ranked) {
    const uint32_t id = input[rank];
    if (id >= rows) continue;
    auto slot = ((id ^ (id >> 16)) * 2654435761u) >> 24;
    while (selected[slot] != UINT32_MAX && selected[slot] != id) {
      slot = (slot + 1) & (selected.size() - 1);
    }
    if (selected[slot] != UINT32_MAX) continue;
    selected[slot] = id;
    output[size++] = id;
    if (size == OUTPUT_DEGREE) return true;
  }
  return false;
}

// Keep the best half of the outgoing row and insert incoming edges in rank order.
inline void merge_reverse(uint32_t *forward, const uint32_t *reverse, unsigned count)
{
  uint32_t *const insert_at = forward + OUTPUT_DEGREE / 2;
  uint32_t *const end = forward + OUTPUT_DEGREE;
  for (unsigned remaining = std::min(count, OUTPUT_DEGREE); remaining > 0; --remaining) {
    const auto id = reverse[remaining - 1];
    const auto old = std::find(forward, end, id);
    if (old < insert_at) continue;
    auto *const last = old == end ? end - 1 : old;
    std::move_backward(insert_at, last, last + 1);
    *insert_at = id;
  }
}
}  // namespace cagra_experiment
