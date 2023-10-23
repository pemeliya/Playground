
#ifndef TOPK_KERNEL_CU_H_
#define TOPK_KERNEL_CU_H_

// This file contains bespoke and optimized implementation for TopK shapes. When
// adding support for new shapes/dtypes, you also need to modify the rewritter
// on topk_specializer.cc for these changes to be picked up.

#include "topk_kernel.h"

#include <cstddef>
#include <cstdint>
#include <limits>

// Default implementation for KV holder. Useful for testing while adding support
// for a new type, but generally bitpacking those values is more efficient. See
// implementations below.
template <typename T, typename V>
struct Descending {
  struct KVT {

    __device__ explicit KVT(T k = {}, V v = {}) : k_(k), v_(v) {}
    __forceinline__ __device__ void Write(T* key, uint32_t* value) const {
      *key = k_;
      *value = v_;
    }

    __device__ __forceinline__ KVT ShuffleDown(int offset) const {
      unsigned FULL_MASK = 0xffffffff;
      // The static casts here are necessary because some types will be
      // broadened (e.g. bfloat16 -> f32), so we need to narrow them back after
      // the shuffle.
      return KVT(static_cast<T>(__shfl_down_sync(FULL_MASK, k_, offset)),
                 static_cast<V>(__shfl_down_sync(FULL_MASK, v_, offset)));
    }

   private:
    T k_;
    V v_;
    friend struct Descending<T, V>;
  };

  __device__ __forceinline__ static constexpr bool Gt(const KVT& lhs,
                                                      const KVT& rhs) {
    return lhs.k_ == rhs.k_ ? lhs.v_ < rhs.v_ : lhs.k_ > rhs.k_;
  }
};

__device__ __forceinline__ int Idx(int i) {
  return blockDim.x * i + threadIdx.x;
}

template <size_t K, typename KT, typename VT,
          template <typename KT1, typename VT2> class Traits = Descending>
class TopK {
 public:
  using Trait = Traits<KT, VT>;
  using KVT = typename Trait::KVT;

  __device__ TopK(void* buffer, int num_outputs)
      : buffer_(reinterpret_cast<KVT*>(buffer)), num_outputs_(num_outputs) {}

  __device__ void Run(KT* key, int n, KT* keys, uint32_t* values) {
    PerWarpTopK(key, n);
    MergeTopKs(keys, values);
  }

  //  TopK<K, KT, VT> top_k(shmem, k);
  // int slice_size = (n + blockDim.x-1) / blockDim.x;
  // top_k.Run(&data[n * blockIdx.x], slice_size, &result[k * blockIdx.x],
  //           &result_idxs[k * blockIdx.x]);

 private:
  // Compute a per-warp topk of a slice of data.
  __device__ void PerWarpTopK(KT* key, int n) {

// __device__ __forceinline__ int Idx(int i) {
//   return blockDim.x * i + threadIdx.x;
// }

    KVT tmp[K];
    // TODO(doak): Use bitonic sort.
#pragma unroll
    for (int i = 0; i < K; i++) tmp[i] = KVT(key[Idx(i)], Idx(i));
#pragma unroll
    for (int i = 0; i < K; i++) {
#pragma unroll
      for (int j = i + 1; j < K; j++) {
        KVT ti = tmp[i];
        KVT tj = tmp[j];
        bool cmp = Trait::Gt(ti, tj);
        tmp[i] = cmp ? ti : tj;
        tmp[j] = cmp ? tj : ti;
      }
    }

    for (int idx = K; idx < n; idx++) {
      KVT kv(key[Idx(idx)], Idx(idx));
      Push(tmp, kv);
    }

    Reduce(tmp, 32);

    if (threadIdx.x % 32 != 0) return;
    int warp_id = threadIdx.x / 32;
    for (int i = 0; i < K; i++) {
      buffer_[i * 32 + warp_id] = tmp[i];
    }
  }

  // Merge the per-warp topks into a single topk. The final data is written to
  // `keys` and `values`
  __device__ void MergeTopKs(KT* keys, uint32_t* values) {
    KVT tmp[K];
    // We only use one warp for this step.
    if (threadIdx.x / 32 != 0) return;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < K; i++) tmp[i] = buffer_[i * 32 + threadIdx.x];
    Reduce(tmp, blockDim.x / 32);
    if (threadIdx.x != 0) return;
    for (int i = 0; i < num_outputs_; ++i) {
      tmp[i].Write(&keys[i], &values[i]);
    }
  }

  // Merge `tmp` (a reverse-sorted array) from (0, `num_lanes`) lanes. The
  // resulting array is stored in the tmp array of lane 0. For all other lanes,
  // `tmp` is unspecified after this function is called.
  __device__ __forceinline__ void Reduce(KVT tmp[K], int num_lanes) {
    int lane_id = threadIdx.x % 32;
    for (int offset = num_lanes / 2; offset > 0; offset /= 2) {
#pragma unroll
      for (int i = 0; i < K; i++) {
        KVT kv = tmp[i].ShuffleDown(offset);
        if (lane_id >= offset) continue;
        Push(tmp, kv);
      }
    }
  }

//! Gt: return lhs.k_ == rhs.k_ ? lhs.v_ < rhs.v_ : lhs.k_ > rhs.k_;
  // Given a K-array of previously reverse-sorted KVTs, add kv to it and
  // remove the smallest element of the resulting array. Preserves the sorted
  // order of `tmp`.
  static __device__ __forceinline__ bool Push(KVT tmp[K], const KVT& kv) {
    if (Trait::Gt(tmp[K - 1], kv)) return false;
    tmp[K - 1] = kv; // (K-1)th is the smallest element out of K
    if constexpr (K >= 2) {
#pragma unroll
      for (int i = K - 2; i >= 0; --i) {
        if (Trait::Gt(tmp[i], kv)) break;
        // Swap
        KVT t = tmp[i];
        tmp[i] = tmp[i + 1];
        tmp[i + 1] = t;
      }
    }
    return true;
  }

  int source_ = 0;
  KVT* buffer_;
  int num_outputs_;
};

// This shared memory buffer needs to be declared outside of the templated
// Run(), as otherwise it would generate name conflicts from the multiple
// instantiations of Run() from the multiple monomorphizations of Run().
extern __device__ __shared__ int shmem[];

template <size_t K, typename KT, typename VT>
__launch_bounds__(kTopKMaxThreadsPerBlock, 1) __global__
    void Run(KT* data, int n, KT* result, uint32_t* result_idxs, int k) {
  TopK<K, KT, VT> top_k(shmem, k);
  int slice_size = (n + blockDim.x-1) / blockDim.x;
  top_k.Run(&data[n * blockIdx.x], slice_size, &result[k * blockIdx.x],
            &result_idxs[k * blockIdx.x]);
}

template <typename T, size_t K>
void* GetTopKKernelForK(int n) {
  // TODO(doak): Switch to uint32_t if we don't have an efficient
  // implemementation for uint16_t.
  return //n < std::numeric_limits<uint16_t>::max()
           //  ? reinterpret_cast<void*>(&Run<K, T, uint16_t>)
             reinterpret_cast<void*>(&Run<K, T, uint32_t>);
}

#endif  // TOPK_KERNEL_CU_H_
