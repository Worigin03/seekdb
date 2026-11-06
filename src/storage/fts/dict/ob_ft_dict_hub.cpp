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

#define USING_LOG_PREFIX STORAGE_FTS

#include "storage/fts/dict/ob_ft_dict_hub.h"

#include "lib/alloc/alloc_struct.h"
#include "lib/ob_errno.h"
#include "lib/oblog/ob_log_module.h"
#include "lib/utility/ob_macro_utils.h"
#include "storage/fts/dict/ob_ft_cache_container.h"
#include "storage/fts/dict/ob_ft_dict_def.h"
#include "storage/fts/dict/ob_ft_range_dict.h"
namespace oceanbase
{
namespace storage
{
struct ObFTDictHub::BuiltinDict
{
  explicit BuiltinDict(const ObFTDictDesc &desc)
      : allocator_(lib::ObMemAttr("IKBuiltinDict")),
        cache_(allocator_),
        dict_(allocator_, &cache_, desc)
  {}

  ObArenaAllocator allocator_;
  ObFTCacheRangeContainer cache_;
  ObFTRangeDict dict_;
};

ObFTDictHub::~ObFTDictHub()
{
  for (auto &slot : builtin_dicts_) {
    BuiltinDict *entry = slot.load(std::memory_order_relaxed);
    if (nullptr != entry) {
      OB_DELETE(BuiltinDict, "IKBuiltinDict", entry);
    }
  }
}

int ObFTDictHub::get_builtin_ik_dict(ObFTDictType type, const ObIFTDict *&dict)
{
  int ret = OB_SUCCESS;
  dict = nullptr;
  if (type != ObFTDictType::DICT_IK_MAIN && type != ObFTDictType::DICT_IK_QUAN) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    auto &slot = builtin_dicts_[type == ObFTDictType::DICT_IK_MAIN ? 0 : 1];
    BuiltinDict *entry = slot.load(std::memory_order_acquire);
    if (nullptr == entry) {
      std::lock_guard<std::mutex> guard(build_mutex_);
      entry = slot.load(std::memory_order_relaxed);
      if (nullptr == entry) {
        const ObFTDictDesc desc(type == ObFTDictType::DICT_IK_MAIN ? "main_dict" : "quan_dict",
                                type, CHARSET_UTF8MB4, CS_TYPE_UTF8MB4_BIN);
        if (OB_ISNULL(entry = OB_NEW(BuiltinDict, "IKBuiltinDict", desc))) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else if (OB_FAIL(ObFTRangeDict::build_cache_from_ik_dict(desc, entry->cache_))) {
        } else if (OB_FAIL(entry->dict_.init())) {
        } else {
          slot.store(entry, std::memory_order_release);
        }
        if (OB_FAIL(ret) && nullptr != entry) {
          OB_DELETE(BuiltinDict, "IKBuiltinDict", entry);
        }
      }
    }
    if (OB_SUCC(ret)) {
      dict = &entry->dict_;
    }
  }
  return ret;
}
} //  namespace storage
} //  namespace oceanbase
