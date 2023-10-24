
#ifndef TOPK_KERNEL_CU_H_
#define TOPK_KERNEL_CU_H_

// This file contains bespoke and optimized implementation for TopK shapes. When
// adding support for new shapes/dtypes, you also need to modify the rewritter
// on topk_specializer.cc for these changes to be picked up.

#include "topk_kernel.h"
#include "common_funcs.cu.h"

#include <cstddef>
#include <cstdint>
#include <limits>

// bitonic TopK: https://github.com/anilshanbhag/gpu-topk

template <size_t WarpSize, size_t K, typename KT, typename VT>
struct TopK {
  
  struct KVT {
    KT key;
    VT idx;
    __device__ __forceinline__ bool operator >(const KVT& rhs) {
      return key == rhs.key ? idx < rhs.idx : key > rhs.key;
    }
  };

  __device__ TopK(void* buffer, int num_outputs)
      : buffer_(reinterpret_cast<KVT*>(buffer)), num_outputs_(num_outputs) {}
 
__device__ __forceinline__ uint32_t Idx(uint32_t i) {
  return blockDim.x * i + threadIdx.x;
}
  // Compute a per-warp topk of a slice of data.
  __device__ void PerWarpTopK(KT* key, int n) {

    KVT tmp[K];
    // TODO(doak): Use bitonic sort.
#pragma unroll    
    for (int i = 0; i < K; i++) {
        tmp[i] = {key[Idx(i)], Idx(i)};
    }
    for (int i = 0; i < K; i++) {
#pragma unroll          
      for (int j = i + 1; j < K; j++) {
        KVT ti = tmp[i];
        KVT tj = tmp[j];
        bool cmp = ti > tj;
        tmp[i] = cmp ? ti : tj;
        tmp[j] = cmp ? tj : ti;
      }
    }

    for (int idx = K; idx < n; idx++) {
      KVT kv{key[Idx(idx)], Idx(idx)};
      Push(tmp, kv);
    }

    Reduce(tmp, WarpSize);

    if (threadIdx.x % WarpSize != 0) return;
    int warp_id = threadIdx.x / WarpSize;
    for (int i = 0; i < K; i++) {
      buffer_[i * WarpSize + warp_id] = tmp[i];
    }
  }

  // Merge the per-warp topks into a single topk. The final data is written to
  // `keys` and `idxs`
  __device__ void MergeTopKs(KT *keys, uint32_t *idxs) {
    KVT tmp[K];
    // We only use one warp for this step.
    if (threadIdx.x >= WarpSize) return;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < K; i++) {
      tmp[i] = buffer_[i * WarpSize + threadIdx.x];
    }
    Reduce(tmp, blockDim.x / WarpSize);
    if (threadIdx.x != 0) return;
    for (int i = 0; i < num_outputs_; ++i) {
      keys[i] = tmp[i].key;
      idxs[i] = tmp[i].idx;
    }
  }

  // Merge `tmp` (a reverse-sorted array) from (0, `num_lanes`) lanes. The
  // resulting array is stored in the tmp array of lane 0. For all other lanes,
  // `tmp` is unspecified after this function is called.
  __device__ __forceinline__ void Reduce(KVT tmp[K], int num_lanes) {
    int lane_id = threadIdx.x % WarpSize;
    for (int offset = num_lanes / 2; offset > 0; offset /= 2) {
#pragma unroll
      for (int i = 0; i < K; i++) {
        KVT kv = shflType< stDown >(tmp[i], offset);
        if (lane_id >= offset) continue;
        Push(tmp, kv);
      }
    }
  }

//! Gt: return lhs.k_ == rhs.k_ ? lhs.v_ < rhs.v_ : lhs.k_ > rhs.k_;
  // Given a K-array of previously reverse-sorted KVTs, add kv to it and
  // remove the smallest element of the resulting array. Preserves the sorted
  // order of `tmp`.
  static __device__ __forceinline__ bool Push(KVT tmp[K], const KVT& kv) 
  {
    if (tmp[K - 1] > kv) return false;
    tmp[K - 1] = kv; // (K-1)th is the smallest element out of K
#pragma unroll
    for (int i = K - 2; i >= 0; --i) {
      if (tmp[i] > kv) break;
      // Swap
      KVT t = tmp[i];
      tmp[i] = tmp[i + 1];
      tmp[i + 1] = t;
    }
    return true;
  }

  KVT* buffer_;
  int num_outputs_;
};

// This shared memory buffer needs to be declared outside of the templated
// Run(), as otherwise it would generate name conflicts from the multiple
// instantiations of Run() from the multiple monomorphizations of Run().
extern __device__ __shared__ int shmem[];

template <size_t WarpSize, size_t K, typename KT, typename VT>
__device__ void topk_kernel_main(KT* data, int n, KT* result, uint32_t* result_idxs, int k) 
{
  TopK<WarpSize, K, KT, VT> obj(shmem, k);
  
  auto in = data + n * blockIdx.x;
  auto vals_out = result + k * blockIdx.x;
  auto idxs_out = result_idxs + k * blockIdx.x;
  int slice_size = n / blockDim.x;
  if (threadIdx.x < n % blockDim.x) {
    slice_size++;
  }
  
  obj.PerWarpTopK(in, slice_size);
  obj.MergeTopKs(vals_out, idxs_out);
}

template <size_t WarpSize, size_t K, typename KT, typename VT>
__launch_bounds__(512, 1) __global__
    void Run512(KT* data, int n, KT* result, uint32_t* result_idxs, int k) 
{
  topk_kernel_main< WarpSize, K, KT, VT >(data, n, result, result_idxs, k);
}

template <size_t WarpSize, size_t K, typename KT, typename VT>
__launch_bounds__(1024, 1) __global__
    void Run1024(KT* data, int n, KT* result, uint32_t* result_idxs, int k) 
{
  topk_kernel_main< WarpSize, K, KT, VT >(data, n, result, result_idxs, k);
}

template <typename T, size_t K>
void* GetTopKKernelForK(size_t n_threads) {
  
  return reinterpret_cast<void*>
#if COMPILE_FOR_ROCM  // warp size is 64 for ROCM
      (n_threads <= 512 ? &Run512<64, K, T, uint32_t> : &Run1024<64, K, T, uint32_t>);
#else
      (n_threads <= 512 ? &Run512<32, K, T, uint32_t> : &Run1024<32, K, T, uint32_t>);
#endif
}

#endif  // TOPK_KERNEL_CU_H_
