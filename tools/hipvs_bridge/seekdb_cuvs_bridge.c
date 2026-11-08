/* seekdb <-> hipVS bridge: plain-C CAGRA kNN; all cuVS/ROCm deps live here. */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#ifdef SEEKDB_EXPERIMENTAL_GPU_PRUNE
extern int seekdb_fast_cagra_build(cuvsResources_t, const float *, unsigned, unsigned,
                                  unsigned, cuvsCagraIndex_t);
#endif

typedef struct {
  cuvsResources_t        res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t       idx;
  cuvsCagraSearchParams_t sp;
  float*                  qd;
  unsigned int*           nd;
  float*                  dd;
  size_t                  query_capacity;
  size_t                  result_capacity;
  long n, dim;
} sk_cuvs_index;

/* SQL executes nq=4096 as individual statements. Keep the small per-index
 * scratch buffers and search params instead of paying RMM/setup costs per row.
 * seekdb serializes handle use with g_ob_cuvs_mu; one-shot batch handles are
 * private to their worker, so a handle is never searched concurrently. */
static int sk_cuvs_reserve(sk_cuvs_index* h, long nq, long topk)
{
  size_t query_count, result_count;
  if (!h || nq <= 0 || topk <= 0 || h->dim <= 0) return 2;
  if ((size_t)nq > (size_t)-1 / (size_t)h->dim ||
      (size_t)nq > (size_t)-1 / (size_t)topk) return 2;
  query_count = (size_t)nq * (size_t)h->dim;
  result_count = (size_t)nq * (size_t)topk;
  if (query_count > (size_t)-1 / sizeof(float) ||
      result_count > (size_t)-1 / sizeof(unsigned int)) return 2;

  if (query_count > h->query_capacity) {
    float* qd = 0;
    if (cuvsRMMAlloc(h->res, (void**)&qd, sizeof(float) * query_count) != CUVS_SUCCESS) return 10;
    if (h->qd) cuvsRMMFree(h->res, h->qd, sizeof(float) * h->query_capacity);
    h->qd = qd;
    h->query_capacity = query_count;
  }
  if (result_count > h->result_capacity) {
    unsigned int* nd = 0;
    float* dd = 0;
    if (cuvsRMMAlloc(h->res, (void**)&nd, sizeof(unsigned int) * result_count) != CUVS_SUCCESS) return 11;
    if (cuvsRMMAlloc(h->res, (void**)&dd, sizeof(float) * result_count) != CUVS_SUCCESS) {
      cuvsRMMFree(h->res, nd, sizeof(unsigned int) * result_count);
      return 12;
    }
    if (h->nd) cuvsRMMFree(h->res, h->nd, sizeof(unsigned int) * h->result_capacity);
    if (h->dd) cuvsRMMFree(h->res, h->dd, sizeof(float) * h->result_capacity);
    h->nd = nd;
    h->dd = dd;
    h->result_capacity = result_count;
  }
  return 0;
}

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { free(h); return NULL; }

  DLManagedTensor dset; long ds[2] = {n, dim};
  dset.dl_tensor.data = (void*)base;
  dset.dl_tensor.device.device_type = kDLCPU; dset.dl_tensor.device.device_id = 0;
  dset.dl_tensor.ndim = 2;
  dset.dl_tensor.dtype.code = kDLFloat; dset.dl_tensor.dtype.bits = 32; dset.dl_tensor.dtype.lanes = 1;
  dset.dl_tensor.shape = ds; dset.dl_tensor.strides = 0; dset.dl_tensor.byte_offset = 0;

  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) goto fail_res;
  h->ip->build_algo = NN_DESCENT;
  h->ip->graph_degree = 128;
  h->ip->intermediate_graph_degree = 256;
  // Keep V5's graph connectivity while reducing the NN-Descent iteration cap.
  h->ip->nn_descent_niter = 48;
  if (cuvsCagraIndexCreate(&h->idx)      != CUVS_SUCCESS) goto fail_ip;
#ifdef SEEKDB_EXPERIMENTAL_GPU_PRUNE
  if (n > 384 && (unsigned long)n <= UINT32_MAX &&
      dim > 0 && (unsigned long)dim <= UINT32_MAX) {
    if (seekdb_fast_cagra_build(h->res, base, (unsigned)n, (unsigned)dim,
                              (unsigned)h->ip->nn_descent_niter, h->idx) != 0) goto fail_idx;
  } else
#endif
  if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) goto fail_idx;
  h->n = n; h->dim = dim;
  return h;

fail_idx: cuvsCagraIndexDestroy(h->idx);
fail_ip:  cuvsCagraIndexParamsDestroy(h->ip);
fail_res: cuvsResourcesDestroy(h->res); free(h);
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  return NULL;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h || !query || !out_ids) return 1;
  int rc = sk_cuvs_reserve(h, nq, topk);
  if (rc) goto done;
  if (cudaMemcpy(h->qd, query, sizeof(float)*nq*h->dim, cudaMemcpyDefault) != cudaSuccess) { rc=13; goto done; }
  {
    DLManagedTensor qt, nt, dt; long qs[2]={nq,h->dim}, nsh[2]={nq,topk}, dsh[2]={nq,topk};
    qt.dl_tensor.data=h->qd; qt.dl_tensor.device.device_type=kDLCUDA; qt.dl_tensor.device.device_id=0;
    qt.dl_tensor.ndim=2; qt.dl_tensor.dtype.code=kDLFloat; qt.dl_tensor.dtype.bits=32; qt.dl_tensor.dtype.lanes=1;
    qt.dl_tensor.shape=qs; qt.dl_tensor.strides=0; qt.dl_tensor.byte_offset=0;
    nt=qt; nt.dl_tensor.data=h->nd; nt.dl_tensor.dtype.code=kDLUInt; nt.dl_tensor.shape=nsh;
    dt=qt; dt.dl_tensor.data=h->dd; dt.dl_tensor.dtype.code=kDLFloat; dt.dl_tensor.shape=dsh;
    if (!h->sp && cuvsCagraSearchParamsCreate(&h->sp) != CUVS_SUCCESS) { rc=14; goto done; }
    {
      size_t required_itopk = (size_t)topk * 4;
      if (required_itopk < 512) required_itopk = 512;
      if (h->sp->itopk_size < required_itopk) h->sp->itopk_size = required_itopk;
      h->sp->search_width = 4;
      // Use multiple blocks for a single query with the standard candidate budget.
      // Reset both settings because the same index can also serve batch searches.
      const bool use_multi_cta = nq == 1 && h->sp->itopk_size == 512;
      h->sp->algo = use_multi_cta ? MULTI_CTA : SINGLE_CTA;
      h->sp->max_iterations = use_multi_cta ? 48 : 0;
    }
    cuvsFilter f; f.type=NO_FILTER; f.addr=(uintptr_t)0;
    if (cuvsCagraSearch(h->res, h->sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc=15; goto done; }
    if (cudaMemcpy(out_ids, h->nd, sizeof(unsigned int)*nq*topk, cudaMemcpyDefault) != cudaSuccess) { rc=16; goto done; }
    if (out_dist && cudaMemcpy(out_dist, h->dd, sizeof(float)*nq*topk, cudaMemcpyDefault) != cudaSuccess) {
      rc=17; goto done;
    }
  }
done:
  if (rc) fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  return rc;
}

void seekdb_cuvs_free(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return;
  cuvsCagraIndexDestroy(h->idx);
  cuvsCagraIndexParamsDestroy(h->ip);
  if (h->sp) cuvsCagraSearchParamsDestroy(h->sp);
  if (h->dd) cuvsRMMFree(h->res, h->dd, sizeof(float) * h->result_capacity);
  if (h->nd) cuvsRMMFree(h->res, h->nd, sizeof(unsigned int) * h->result_capacity);
  if (h->qd) cuvsRMMFree(h->res, h->qd, sizeof(float) * h->query_capacity);
  cuvsResourcesDestroy(h->res);
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_cuvs_index* index = (sk_cuvs_index*)handle;
  if (!index || !filename) return 1;
  if (cuvsCagraSerialize(index->res, filename, index->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) return NULL;
  sk_cuvs_index* index = (sk_cuvs_index*)calloc(1, sizeof(*index));
  if (!index) return NULL;
  if (cuvsResourcesCreate(&index->res) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraIndexParamsCreate(&index->ip) != CUVS_SUCCESS) goto fail_res;
  if (cuvsCagraIndexCreate(&index->idx) != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraDeserialize(index->res, filename, index->idx) != CUVS_SUCCESS) goto fail_idx;
  {
    int64_t actual_n = 0;
    int64_t actual_dim = 0;
    if (cuvsCagraIndexGetSize(index->idx, &actual_n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(index->idx, &actual_dim) != CUVS_SUCCESS ||
        actual_n != n || actual_dim != dim) {
      fprintf(stderr, "[seekdb_cuvs] snapshot shape mismatch: %ld x %ld, expected %ld x %ld\n",
              (long)actual_n, (long)actual_dim, n, dim);
      goto fail_idx;
    }
  }
  index->n = n;
  index->dim = dim;
  return index;
fail_idx:
  cuvsCagraIndexDestroy(index->idx);
fail_ip:
  cuvsCagraIndexParamsDestroy(index->ip);
fail_res:
  cuvsResourcesDestroy(index->res);
fail:
  fprintf(stderr, "[seekdb_cuvs] deserialize failed: %s\n", cuvsGetLastErrorText());
  free(index);
  return NULL;
}

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids)
{
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
