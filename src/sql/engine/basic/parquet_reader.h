/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "file_scan_utils.h"
#include <memory>

namespace oceanbase { namespace sql { namespace file_sql {
class ParquetReader {
public:
  explicit ParquetReader(FILE *file, int64_t file_size);
  ~ParquetReader();
  ParquetReader(const ParquetReader &) = delete;
  ParquetReader &operator=(const ParquetReader &) = delete;
  const std::vector<Column> &columns() const;
  void project(const std::vector<int64_t> &columns);
  void rescan();
  bool next(std::vector<Cell> &row);
  static std::string decode_snappy(const std::string &input, size_t expected_size);
private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}}}
