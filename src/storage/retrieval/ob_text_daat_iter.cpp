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

#define USING_LOG_PREFIX STORAGE

#include "ob_text_daat_iter.h"

namespace oceanbase
{
namespace storage
{
int ObTextDaaTIter::init(const ObTextDaaTParam &param)
{
  int ret = OB_SUCCESS;
  if (IS_INIT) {
    ret = OB_INIT_TWICE;
  } else if (OB_ISNULL(param.base_param_) || OB_ISNULL(param.dim_iters_) || OB_ISNULL(param.allocator_)
      || OB_ISNULL(param.relevance_collector_) || OB_ISNULL(param.base_param_->id_proj_expr_)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    mode_flag_ = param.mode_flag_;
    function_lookup_mode_ = param.function_lookup_mode_;
    const sql::ObDatumMeta &id_meta = param.base_param_->id_proj_expr_->datum_meta_;
    use_small_union_ = sql::ObMatchAgainstMode::NATURAL_LANGUAGE_MODE == mode_flag_
        && !function_lookup_mode_ && !param.base_param_->need_project_relevance()
        && !param.base_param_->need_filter() && nullptr == param.base_param_->dim_weights_
        && param.minimum_should_match_ <= 1
        && (ObUInt64Type == id_meta.type_ || (ObVarcharType == id_meta.type_ && CS_TYPE_BINARY == id_meta.cs_type_))
        && param.dim_iters_->count() <= ObSRSimpleMerger::USE_SIMPLE_MERGER_MAX_TABLE_CNT;
    need_merge_heap_ = !use_small_union_;
    need_buffered_domain_ids_ = !(use_small_union_ && ObUInt64Type == id_meta.type_);
  }
  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(ObSRDaaTIterImpl::init(*param.base_param_, *param.dim_iters_,
                                            *param.allocator_, *param.relevance_collector_))) {
  } else if (!need_buffered_domain_ids_ && OB_FAIL(init_posting_batches())) {
  } else if (param.base_param_->need_project_relevance()
      && OB_FAIL(bm25_param_estimator_.init(param.bm25_param_est_ctx_))) {
  }
  return ret;
}

void ObTextDaaTIter::reset()
{
  bm25_param_estimator_.reset();
  small_union_ids_.reset();
  posting_ids_.reset();
  posting_batch_size_ = 0;
  active_postings_ = pending_postings_ = 0;
  // this class can know the type of dim_iters_, but ObSRDaaTIterImpl maybe not
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i))->reset();
    }
  }
  ObSRDaaTIterImpl::reset();
}

void ObTextDaaTIter::reuse(const bool switch_tablet)
{
  if ((OB_NOT_NULL(dim_iters_) && 0 == dim_iters_->count())) {
    // do nothing
  } else {
    bm25_param_estimator_.reuse(switch_tablet);
  }
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i))->reuse();
    }
  }
  ObSRDaaTIterImpl::reuse(switch_tablet);
  if (!need_buffered_domain_ids_) {
    rewind_posting_batches();
  }
}

int ObTextDaaTIter::pre_process()
{
  int ret = OB_SUCCESS;
  if (dim_iters_->count() == 0) {
    ret = OB_ITER_END;
  } else if (iter_param_->need_project_relevance()) {
    if (OB_FAIL(bm25_param_estimator_.do_estimation(*iter_param_->eval_ctx_))) {
    }
  }
  return ret;
}

int ObTextDaaTIter::init_posting_batches()
{
  int ret = OB_SUCCESS;
  const int64_t output_capacity = OB_MAX(iter_param_->eval_ctx_->max_batch_size_, iter_param_->max_batch_size_);
  posting_batch_size_ = OB_MAX(iter_param_->eval_ctx_->max_batch_size_, 1);
  const int64_t input_capacity = posting_batch_size_ * dim_iters_->count();
  small_union_ids_.set_allocator(iter_allocator_);
  posting_ids_.set_allocator(iter_allocator_);
  if (OB_FAIL(small_union_ids_.init(output_capacity))) {
  } else if (OB_FAIL(small_union_ids_.prepare_allocate(output_capacity))) {
  } else if (input_capacity > 0 && OB_FAIL(posting_ids_.init(input_capacity))) {
  } else if (input_capacity > 0 && OB_FAIL(posting_ids_.prepare_allocate(input_capacity))) {
  } else {
    rewind_posting_batches();
  }
  return ret;
}

void ObTextDaaTIter::rewind_posting_batches()
{
  active_postings_ = pending_postings_ = (1U << dim_iters_->count()) - 1;
}

int ObTextDaaTIter::refill_posting_batches(const int64_t capacity)
{
  int ret = OB_SUCCESS;
  while (OB_SUCC(ret) && pending_postings_ != 0) {
    const int64_t idx = __builtin_ctz(pending_postings_);
    const unsigned bit = 1U << idx;
    pending_postings_ &= ~bit;
    auto *iter = static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(idx));
    uint64_t *ids = &posting_ids_[idx * posting_batch_size_];
    int64_t count = 0;
    if (OB_FAIL(iter->read_uint_ids(ids, capacity, count))) {
      if (OB_ITER_END == ret) {
        active_postings_ &= ~bit;
        ret = OB_SUCCESS;
      }
    } else {
      postings_[idx].next_ = ids;
      postings_[idx].end_ = ids + count;
    }
  }
  return ret;
}

int ObTextDaaTIter::merge_next_rows(const int64_t capacity, int64_t &count)
{
  if (need_buffered_domain_ids_) {
    return ObSRDaaTIterImpl::merge_next_rows(capacity, count);
  }

  int ret = OB_SUCCESS;
  const common::ObLimitParam &limit = *iter_param_->limit_param_;
  const bool has_limit = limit.is_valid();
  const int64_t offset = has_limit ? limit.offset_ : 0;
  // The scan expression frame is shared by the term scans. Keep one owned
  // batch per term and fetch again only after consuming that term's batch.
  while (OB_SUCC(ret) && count < capacity) {
    if (pending_postings_ != 0) {
      int64_t fetch_count = posting_batch_size_;
      if (has_limit) {
        fetch_count = MIN(fetch_count, limit.limit_ - output_row_cnt_);
        const int64_t offset_remaining = MAX(0, offset - input_row_cnt_);
        fetch_count += MIN(posting_batch_size_ - fetch_count, offset_remaining);
      }
      if (OB_FAIL(refill_posting_batches(fetch_count))) {
        break;
      }
    }
    if (active_postings_ == 0) {
      ret = OB_ITER_END;
      break;
    }

    uint64_t min_id = UINT64_MAX;
    unsigned matches = 0;
    for (unsigned active = active_postings_; active != 0; active &= active - 1) {
      const int64_t idx = __builtin_ctz(active);
      const uint64_t id = *postings_[idx].next_;
      if (id < min_id) {
        min_id = id;
        matches = 0;
      }
      if (id == min_id) {
        matches |= 1U << idx;
      }
    }
    while (matches != 0) {
      const int64_t idx = __builtin_ctz(matches);
      PostingBatch &posting = postings_[idx];
      if (++posting.next_ == posting.end_) {
        pending_postings_ |= 1U << idx;
      }
      matches &= matches - 1;
    }

    if (++input_row_cnt_ > offset) {
      small_union_ids_[count++] = min_id;
      ++output_row_cnt_;
      if (has_limit && output_row_cnt_ >= limit.limit_) {
        ret = OB_ITER_END;
      }
    }
  }
  return ret;
}

int ObTextDaaTIter::project_results(const int64_t count)
{
  if (need_buffered_domain_ids_) {
    return ObSRDaaTIterImpl::project_results(count);
  }
  int ret = OB_SUCCESS;
  sql::ObExpr *id_proj_expr = iter_param_->id_proj_expr_;
  sql::ObEvalCtx *eval_ctx = iter_param_->eval_ctx_;
  sql::ObEvalCtx::BatchInfoScopeGuard guard(*eval_ctx);
  if (eval_ctx->is_vectorized()) {
    sql::ObBitVector &evaluated_flags = id_proj_expr->get_evaluated_flags(*eval_ctx);
    for (int64_t i = 0; i < count; ++i) {
      guard.set_batch_idx(i);
      id_proj_expr->locate_datum_for_write(*eval_ctx).set_uint(small_union_ids_[i]);
      evaluated_flags.set(i);
    }
    id_proj_expr->set_evaluated_projected(*eval_ctx);
  } else if (OB_UNLIKELY(1 != count)) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    guard.set_batch_idx(0);
    id_proj_expr->locate_datum_for_write(*eval_ctx).set_uint(small_union_ids_[0]);
    id_proj_expr->set_evaluated_projected(*eval_ctx);
  }
  return ret;
}

int ObTextDaaTIter::advance_small_union_heads()
{
  int ret = OB_SUCCESS;
  // Advance only streams that supplied the previous minimum. The remaining
  // heads are still valid in their iterator-owned posting batches.
  for (int64_t i = 0; OB_SUCC(ret) && i < next_round_cnt_; ++i) {
    const int64_t idx = next_round_iter_idxes_[i];
    auto *iter = static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(idx));
    if (OB_FAIL(iter->get_next_row())) {
      if (OB_ITER_END == ret) {
        iter_domain_ids_[idx] = nullptr;
        ret = OB_SUCCESS;
      }
    } else if (OB_FAIL(iter->get_curr_id(iter_domain_ids_[idx]))) {
    }
  }

  return ret;
}

int ObTextDaaTIter::do_one_merge_round(int64_t &count)
{
  if (!use_small_union_) {
    return ObSRDaaTIterImpl::do_one_merge_round(count);
  }

  int ret = OB_SUCCESS;
  if (OB_FAIL(advance_small_union_heads())) {
  } else {
    next_round_cnt_ = 0;
    const ObDatum *minimum = nullptr;
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      const ObDatum *head = iter_domain_ids_[i];
      if (nullptr != head) {
        const int cmp = nullptr == minimum ? -1 : head->get_string().compare(minimum->get_string());
        if (cmp < 0) {
          minimum = head;
          next_round_cnt_ = 0;
        }
        if (cmp <= 0) {
          next_round_iter_idxes_[next_round_cnt_++] = i;
        }
      }
    }
    ret = nullptr == minimum ? OB_ITER_END : cache_result(count, *minimum, 1.0);
  }
  return ret;
}

ObTextBMWIter::ObTextBMWIter()
  : ObSRBMWIterImpl(),
    bm25_param_estimator_() {}

void ObTextBMWIter::reuse(const bool switch_tablet)
{
  bm25_param_estimator_.reuse(switch_tablet);
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalBlockMaxIter *>(dim_iters_->at(i))->reuse();
    }
  }
  ObSRBMWIterImpl::reuse(switch_tablet);
}

void ObTextBMWIter::reset()
{
  bm25_param_estimator_.reset();
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalBlockMaxIter *>(dim_iters_->at(i))->reset();
    }
  }
  ObSRBMWIterImpl::reset();
}

int ObTextBMWIter::init(const ObTextDaaTParam &param)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(param.base_param_) || OB_ISNULL(param.dim_iters_) || OB_ISNULL(param.allocator_)
      || OB_ISNULL(param.relevance_collector_)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObSRBMWIterImpl::init(*param.base_param_, *param.dim_iters_,
                                           *param.allocator_, *param.relevance_collector_))) {
  } else if (OB_FAIL(bm25_param_estimator_.init(param.bm25_param_est_ctx_))) {
  }
  return ret;
}

int ObTextBMWIter::get_next_rows(const int64_t capacity, int64_t &count)
{
  int ret = OB_SUCCESS;
  if (IS_NOT_INIT) {
    ret = OB_NOT_INIT;
  } else if (dim_iters_->count() == 0) {
    ret = OB_ITER_END;
  } else if (OB_FAIL(bm25_param_estimator_.do_estimation(*iter_param_->eval_ctx_))) {
  }

  if (FAILEDx(ObSRBMWIterImpl::get_next_rows(capacity, count))) {
    if (OB_UNLIKELY(OB_ITER_END != ret)) {
    }
  }
  return ret;
}

int ObTextBMWIter::init_before_wand_process()
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(!bm25_param_estimator_.is_estimated())) {
    ret = OB_ERR_UNEXPECTED;
  }

  for (int64_t i = 0; OB_SUCC(ret) && i < dim_iters_->count(); ++i) {
    ObTextRetrievalBlockMaxIter *block_max_iter = static_cast<ObTextRetrievalBlockMaxIter *>(dim_iters_->at(i));
    if (OB_FAIL(block_max_iter->init_block_max_iter(
        bm25_param_estimator_.get_total_doc_cnt(), bm25_param_estimator_.get_avg_doc_token_cnt()))) {
    }
  }
  return ret;
}

} // namespace storage
} // namespace oceanbase
