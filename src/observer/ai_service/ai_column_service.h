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
#include "lib/thread/thread_pool.h"
#include <mutex>
#include <atomic>
namespace oceanbase { namespace observer {
class AiColumnService final : public lib::ThreadPool {
public:
  static AiColumnService &instance();
  int start_service();
  void stop_service();
  void run1() override;
private:
  std::mutex mutex_;
  bool started_ = false;
  std::atomic<bool> recovered_{false};
};
}}
