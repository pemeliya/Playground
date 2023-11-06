
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

#if COMPILE_FOR_ROCM  // warp size is 64 for ROCM
#ifndef __AMDGCN_WAVEFRONT_SIZE
#error Wavefront size is not defined! Please use HIPCC compiler!
#else
#define WAVEFRONT_SIZE __AMDGCN_WAVEFRONT_SIZE
#endif
#else // NVIDIA
#define WAVEFRONT_SIZE 32 
#endif

#define OUTZ(fmt, ...) printf(fmt"\n", ##__VA_ARGS__)

#define LOUTZ(fmt, ...) printf("%d :" fmt "\n", lane, ##__VA_ARGS__)


// bitonic TopK: https://github.com/anilshanbhag/gpu-topk

template <size_t K, typename KT>
struct TopK {
  
  struct KVT {
    KT key;
    uint32_t idx;
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
#pragma unroll    
    for (int i = 0; i < K; i++) {
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
    constexpr uint32_t WarpSize = WAVEFRONT_SIZE;
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
    constexpr uint32_t WarpSize = WAVEFRONT_SIZE;
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
    constexpr uint32_t WarpSize = WAVEFRONT_SIZE;
    int lane_id = threadIdx.x % WarpSize;
    for (int offset = num_lanes / 2; offset > 0; offset /= 2) {
#pragma unroll
      for (int i = 0; i < K; i++) {
        KVT kv = gpuShuffle< ShflType::Down >(tmp[i], offset);
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
    for (int i = (int)K - 2; i >= 0; --i) {
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


template <size_t K, typename KT>
__launch_bounds__(1024, 1) __global__
    void RunTopK_default(KT* data, int n, KT* result, uint32_t* result_idxs, int k) 
{
  TopK<K, KT> obj(shmem, k);
  
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

template <class NT, class LessOp >
__device__ void bitonic_warp_sort(uint32_t lane, NT& d, LessOp op)
{
  constexpr uint32_t WarpSize = WAVEFRONT_SIZE;
#pragma unroll  
  for(int32_t i = 2; i < WarpSize; i *= 2) {
    int32_t biti = (lane & i) == i; // // __builtin_amdgcn_ubfe ??
#pragma unroll    
    for(int32_t j = i / 2; j >= 1; j /= 2) {
      int32_t bitj = (lane & j) == j;
      auto xd = gpuShuffle< ShflType::Xor >(d, j); 
      if((biti ^ bitj ^ op(d, xd)) == 0) {
        d = xd;
      }
    }
    // after step i, we have bitonic sequences of size i*2
    // i.e., after the whole loop we have a bitonic sequence of WarpSize
    // if (bit1 == 0) { // sort min <-- max
    //   d = (bit0 == 0) ? min(d, xd) : max(d, xd);
    // } else { // sort max <-- min
    //   d = (bit0 == 0) ? max(d, xd) : min(d, xd);
    // }
  }
#pragma unroll  
  for(int32_t i = WarpSize/2; i >= 1; i /= 2) {
    int32_t biti = (lane & i) == i;
    auto xd = gpuShuffle< ShflType::Xor >(d, i); 
    if((biti ^ op(d, xd)) == 0) {
      d = xd;
    }
  }
}

constexpr uint32_t log2xN(uint32_t x) {
#pragma unroll
  for(uint32_t i = 0; i < 16; i++) {
    if(x == (1U << i)) {
      return i;
    }
  }
  return 0xFFFFFFFFU;
  //uint32_t r = 0;
  // if constexpr ( x & 0xffff0000UL ) { x >>= 16; r += 16; }
  // if constexpr ( x & 0x0000ff00UL ) { x >>= 8; r += 8; }
  // if constexpr ( x & 0x000000f0UL ) { x >>= 4; r += 4; }
  // if constexpr ( x & 0x0000000cUL ) { x >>= 2; r += 2; }
  // if constexpr ( x & 0x00000002UL ) { r += 1; }
  //return r;
}

// local sort produces sorted runs of size K: ascending / descending
// i.e. for K = 4: [3 5 8 9; 5 3 1 1; 2 5 6 10; 11 4 3 2]
template <uint32_t K, class NT>
__device__ FORCEINLINE void bitonic_local_sort(uint32_t lane, NT& A) 
{
#pragma unroll    
// this produces sequences of length K: alternating ascending - descending
  for(int32_t i = 2; i <= K; i *= 2) {
    int32_t biti = (lane & i) == i; // // __builtin_amdgcn_ubfe ??
#pragma unroll    
    // after step i, we have bitonic sequences of size i*2
    // i == 2: j = 1
    // i == 4: j = 2, 1
    // i == 8: j = 4, 2, 1
    for(int32_t j = i / 2; j >= 1; j /= 2) {
      int32_t bitj = (lane & j) == j;
      auto xA = gpuShuffle< ShflType::Xor >(A, j); 
      if((biti ^ bitj ^ (A < xA)) == 0) {
        A = xA;
      }
    }
  }
}

template <uint32_t K, class NT>
__device__ FORCEINLINE void bitonic_merge(uint32_t lane, NT& A) 
{
    auto xA = gpuShuffle< ShflType::Xor >(A, K); 
    A = max(A, xA);
}

template <uint32_t K, class NT>
__device__ FORCEINLINE void bitonic_rebuild(uint32_t lane, NT& A) 
{
  // rebuild bitonic sequences of length K to sorted ones
  int32_t bitK = (lane & K) == K; 
#pragma unroll    
  for(int32_t j = K / 2; j >= 1; j /= 2) {
    int32_t bitj = (lane & j) == j;
    auto xA = gpuShuffle< ShflType::Xor >(A, j); 
    if((bitK ^ bitj ^ (A < xA)) == 0) {
      A = xA;
    }
  }
}

template <uint32_t K, typename KT>
__global__ void RunTopK_new(KT* data, uint32_t n, KT* result, uint32_t* result_idxs, int k) 
{
  constexpr uint32_t WarpSize = WAVEFRONT_SIZE;
  constexpr auto minVal = std::numeric_limits< KT >::min();

  const uint32_t bidx = blockIdx.x, blockSz = blockDim.x;
  auto in = data + n * bidx;
  auto vals_out = result + k * bidx;
  auto idxs_out = result_idxs + k * bidx;

  const uint32_t thid = threadIdx.x, lane = thid % WarpSize;
  uint32_t idx = thid;

  auto A = (idx < n ? in[idx] : minVal);
  //LOUTZ("original: %d", A);

  bitonic_local_sort<K>(lane, A);

  uint32_t i = 0;
  for(idx += blockSz; ; idx += blockSz, i++) {
    auto warpId = idx & ~(WarpSize - 1);
    //OUTZ("idx: %d, warpID: %d", idx, warpId);
    if(warpId >= n) { // retire completely unused warps
       break;
    }
    auto B = (idx < n ? in[idx] : minVal);
    //LOUTZ("loaded B: %d = %d", idx, B);
    bitonic_local_sort<K>(lane, B);

    // merge K-sorted runs and keep only K greater elements
    bitonic_merge<K>(lane, A);
    bitonic_merge<K>(lane, B);
    // merge maximal elements of A and B: shift B by K elements
    // A: ABCD xxxx EFGH xxxx IJKL xxxx
    // B: xxxx ABCD xxxx EFGH xxxx IJKL 
    auto xA = A, xB = B;
    // B[lane] = B[lane - K]
    //! do we really need to shuffle ??
    B = gpuShuffle< ShflType::Up >(B, K); 
    if(lane & K) {
      A = B;
    }
    //OUTZ("%d: idx: %d: xA = %d, A = %d; xB = %d, B = %d", thid, idx, xA, A, xB, B);
    bitonic_rebuild< K >(lane, A);
    
    // if(idx < n)
    // if(lane == 0) {
    //   OUTZ("i = %d/%d; n = %d; mywarp: %d; my range: [%d; %d]", i, warpId, n, 
    //         thid / WarpSize, warpId, warpId + WarpSize);
    // }
    // __syncthreads();
  }
  auto sh = (KT *)shmem;

  //OUTZ("%d: xfinal: %d", thid, A);
  //return;
  const auto warpId = thid / WarpSize;
#if 0
  sh[thid] = A;
   __syncthreads();
  if(warpId == 0) {
    for(uint32_t i = 1; i < blockSz / WarpSize; i++) {
      auto B = sh[lane + i*WarpSize];
      bitonic_merge<K>(lane, A);
      bitonic_merge<K>(lane, B);
      // optional ??
      B = gpuShuffle< ShflType::Up >(B, K); 
      if(lane & K) {
        A = B;
      }
      bitonic_rebuild< K >(lane, A);
    }
  }
#else
  if(int32_t idx = thid - blockSz / 2; idx >= 0) { // only upper threads need to write
    sh[idx] = A;
  }
  for(uint32_t ofs = blockSz / 2, memofs = 0; ofs >= WarpSize; ofs = ofs / 2) {

    __syncthreads();
    if(thid < ofs) {
      // save upper half in shared memory  
      //int32_t idx = thid - (bidx / WarpSize + 1) / 2 * WarpSize; 
      auto B = sh[thid + memofs];

      bitonic_merge<K>(lane, A);
      bitonic_merge<K>(lane, B);
      // optional ??
      B = gpuShuffle< ShflType::Up >(B, K); 
      if(lane & K) {
        A = B;
      }
      bitonic_rebuild< K >(lane, A);
      // actually only upper blocks need to write back
      if(thid >= ofs/2) {
        sh[thid + memofs] = A; // write to the same place where we read from
      }
      // if(thid == 0)
      // LOUTZ("ofs: %d memofs: %d, A = %d, B = %d", ofs, memofs, A, B);
    }
    memofs += ofs/2;
  } // for ofs
#endif
    //LOUTZ("final: %d", A);
  if(warpId == 0) {
#pragma unroll      
    for(uint32_t i = WarpSize / K, div = WarpSize/2; i > 1; i /= 2, div /= 2) {
      bitonic_merge<K>(lane, A);
      // same as: (lane & ~(K-1))*2 + (lane & (K-1))
      auto idx = (lane / K)*2*K + lane % K;
      A = gpuShuffle< ShflType::Sync >(A, idx); 
      if(lane >= div) {
        A = minVal; // remove lower unused elements
      }
      bitonic_rebuild< K >(lane, A);
    }
    LOUTZ("final: %d", A);
  }
}

template <typename KT>
__global__ void RunTopK_newZZZ(KT* data, int n, KT* result, uint32_t* result_idxs, int k) 
{
  constexpr uint32_t WarpSize = WAVEFRONT_SIZE;
  uint32_t lane = threadIdx.x;
  if(lane >= WarpSize)
    return;

  constexpr uint32_t K = 4, logK = log2xN(K);
  uint32_t A = ((lane+1)*(lane-1)*lane) % 113, minVal = 0;
  uint32_t B = ((lane+1)*(lane-1)*(lane-1)) % 97;

#define XS(idx, val)  if(lane == idx) A = val
  XS(0, 3);
  XS(1, 7);
  XS(2, 4);
  XS(3, 8);
  XS(4, 6);
  XS(5, 2);
  XS(6, 1);
  XS(7, 5);

  XS(8, 15);
  XS(9, 3);
  XS(10, 7);
  XS(11, 7);
  XS(12, 1);
  XS(13, 11);
  XS(14, 8);
  XS(15, 2);

  LOUTZ("original: %d", A);
  bitonic_local_sort<K>(lane, A);
  
  bitonic_local_sort<K>(lane, B);
  LOUTZ("local sorted: A: %d; B: %d", A, B);

  // merge K-sorted runs and keep only K greater elements
  bitonic_merge<K>(lane, A);
  bitonic_merge<K>(lane, B);
  // merge maximal elements of A and B: shift B by K elements
  // A: ABCD xxxx EFGH xxxx IJKL xxxx
  // B: xxxx ABCD xxxx EFGH xxxx IJKL 
  B = gpuShuffle< ShflType::Down >(B, K); 
  if(lane & K) {
    A = B;
  }
  LOUTZ("merged bitonic K-sequences: A: %d", A);

  bitonic_rebuild< K >(lane, A);
  LOUTZ("rebuilt K-runs: A: %d", A);

  // we have in total WarpSize / K sorted runs in A
  // 32 / 8 = 4
  // merge = 2
  // merge = 1
#pragma unroll
  for(uint32_t i = WarpSize / K, div = WarpSize/2; i > 1; i /= 2, div /= 2) {
    bitonic_merge<K>(lane, A);
    // same as: (lane & ~(K-1))*2 + (lane & (K-1))
    auto idx = (lane / K)*2*K + lane % K;
    A = gpuShuffle< ShflType::Sync >(A, idx); 
    if(lane >= div) {
      A = minVal; // remove lower unused elements
    }
    bitonic_rebuild< K >(lane, A);
  }
  //LOUTZ("local sorted: %d; merged: %d, shifted: %d", A, mA, zA);
  OUTZ("%d: squashed: %d", lane, A);
}

template <typename T, size_t K>
void* GetTopKKernelForK(size_t n_threads) {

  return reinterpret_cast<void*>(RunTopK_new<K, T>);
  //return reinterpret_cast<void*>(RunTopK_default<K, T);
}

#endif  // TOPK_KERNEL_CU_H_
