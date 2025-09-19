
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

/*
__device__
inline
int __shfl_xor(int var, int lane_mask, int width = WARP_SIZE) {
    int self = __lane_id();
    int index = self^lane_mask;
    index = index >= ((self+width)&~(width-1))?self:index;
    return __builtin_amdgcn_ds_bpermute(index<<2, var);
}
*/

#define OUTZ(fmt, ...) printf(fmt"\n", ##__VA_ARGS__)
#define OUTREGSZ(SZ, fmt, ...) \
    for(uint32_t i = 0; i < SZ; i++) { \
      printf("%d: " fmt "\n", i, ##__VA_ARGS__); \
    }
#define LOUTZ(fmt, ...) printf("%d :" fmt "\n", lane, ##__VA_ARGS__)


// bitonic TopK: https://github.com/anilshanbhag/gpu-topk

template <size_t K, typename KT>
struct TopK {
  struct KVT {
    KT key;
    uint32_t idx;
    __device__ FORCEINLINE bool operator >(const KVT& rhs) {
      return key == rhs.key ? idx < rhs.idx : key > rhs.key;
    }
  };

  __device__ TopK(void* buffer, int num_outputs)
      : buffer_(reinterpret_cast<KVT*>(buffer)), num_outputs_(num_outputs) {}
 
__device__ FORCEINLINE uint32_t Idx(uint32_t i) {
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
    
    Reduce(tmp, WARP_SIZE);

    if (threadIdx.x % WARP_SIZE != 0) return;
    int warp_id = threadIdx.x / WARP_SIZE;
    for (int i = 0; i < K; i++) {
      buffer_[i * WARP_SIZE + warp_id] = tmp[i];
    }
  }

  // Merge the per-warp topks into a single topk. The final data is written to
  // `keys` and `idxs`
  __device__ void MergeTopKs(KT *keys, uint32_t *idxs) {
    KVT tmp[K];
    
    // We only use one warp for this step.
    if (threadIdx.x >= WARP_SIZE) return;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < K; i++) {
      tmp[i] = buffer_[i * WARP_SIZE + threadIdx.x];
    }
    Reduce(tmp, blockDim.x / WARP_SIZE);
    if (threadIdx.x != 0) return;
    for (int i = 0; i < num_outputs_; ++i) {
      keys[i] = tmp[i].key;
      idxs[i] = tmp[i].idx;
    }
  }

  // Merge `tmp` (a reverse-sorted array) from (0, `num_lanes`) lanes. The
  // resulting array is stored in the tmp array of lane 0. For all other lanes,
  // `tmp` is unspecified after this function is called.
  __device__ FORCEINLINE void Reduce(KVT tmp[K], int num_lanes) {
    
    int lane_id = threadIdx.x % WARP_SIZE;
    for (int offset = num_lanes / 2; offset > 0; offset /= 2) {
#pragma unroll
      for (int i = 0; i < K; i++) {
        KVT kv = gpuShuffle< ShflType::Down >(tmp[i], offset);
        if (lane_id >= offset) continue;
        Push(tmp, kv);
      }
    }
  }

  // Given a K-array of previously reverse-sorted KVTs, add kv to it and
  // remove the smallest element of the resulting array. Preserves the sorted
  // order of `tmp`.
  static __device__ FORCEINLINE bool Push(KVT tmp[K], const KVT& kv) 
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
extern __device__ __shared__ int32_t g_shared_mem[];


template <size_t K, typename KT>
__launch_bounds__(1024, 1) __global__
    void RunTopK_default(KT* data, int n, KT* result, uint32_t* result_idxs, int k) 
{
  TopK<K, KT> obj(g_shared_mem, k);
  
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

template <class NT>
__device__ FORCEINLINE void xswap(NT& a, NT& b) {
  auto c = a;
  a = b, b = c;
}

// K - topk size
// N - elements per thread
template < class KT, uint32_t K >
struct BitonicTopK {
  
  constexpr static uint32_t logK = log2xN(K);
  struct KVT {
    KT key;
    uint32_t idx;
    __device__ FORCEINLINE bool operator <(const KVT& rhs) {
      return key == rhs.key ? idx < rhs.idx : key < rhs.key;
    }
  };

  // local sort produces sorted runs of size K: ascending / descending
  // i.e. for K = 4: [3 5 8 9; 5 3 1 1; 2 5 6 10; 11 4 3 2]
  // index I - is just vector size we want to use
  template <uint32_t SZ, class NT> // SZ >= I
  __device__ FORCEINLINE void local_sort(NT (&A)[SZ], uint32_t I = 1)
  {
    auto lane = gpuLaneId();
    // this produces sequences of length K: alternating ascending - descending
#pragma unroll
    for(int32_t i = 1; i <= logK; i++) {
      int32_t biti = gpuGetBit(lane, i);
      // after step i, we have bitonic sequences of size i*2
      // i == 2: j = 1
      // i == 4: j = 2, 1
      // i == 8: j = 4, 2, 1
#pragma unroll
      for(int32_t j = i - 1; j >= 0; j--) {
        int32_t bit = biti ^ gpuGetBit(lane, j);
#pragma unroll
        for(uint32_t v = 0; v < I; v++) {
          auto xA = gpuShuffle< ShflType::Xor >(A[v], 1u << j); 
          if((bit ^ (A[v] < xA)) == 0) {
            A[v] = xA;
          }
        }
      } // for j
    } // for i
  }

  template <class NT>
  __device__ FORCEINLINE void merge(NT& A) 
  {
    auto xA = gpuShuffle< ShflType::Xor >(A, K); 
    A = A < xA ? xA : A;
  }

    template <class NT>
  __device__ FORCEINLINE void merge(NT& A, NT& B) 
  {
      const auto lane = gpuLaneId();
      if constexpr(K < WARP_SIZE) {
        // merge K-sorted runs and keep only K greater elements
        merge(A);
        merge(B); 
        // e.g. K = 4, WARP_SIZE = 16
        // lane =     0123 4567 89AB CDEF
        // A =        ABCD ABCD MNPQ MNPQ
        // B =        IJKL IJKL UWXY UWXY  
        // merged A = ABCD IJKL MNPQ UWXY
        // but this does not work when K == WARP_SIZE
        // merge maximal elements of A and B: shift B by K elements
        // B[lane] = B[lane - K]
        if(lane & K) {
          A = B;
        }
      } else { // A & B are just two ascending sequences => merge directly
        // sync wraps around
        B = gpuShuffle< ShflType::Sync >(B, WARP_SIZE - 1 - lane); 
        A = A < B ? B : A;
      }
  }

  template <uint32_t SZ, class NT>
  __device__ FORCEINLINE void rebuild(NT (&A)[SZ], uint32_t I = 1)
  {
    auto lane = gpuLaneId();
    // rebuild bitonic sequences of length K to sorted ones
    int32_t bitK = gpuGetBit(lane, logK); 
#pragma unroll
    for(int32_t j = logK - 1; j >= 0; j--) {
      int32_t bit = bitK ^ gpuGetBit(lane, j); // we can save on 'gpuGetBit' !
      // src = lane ^ (1u << j);
#pragma unroll      
      for(uint32_t v = 0; v < I; v++) {
        auto xA = gpuShuffle< ShflType::Xor >(A[v], 1u << j); 
        if((bit ^ (A[v] < xA)) == 0) {
          A[v] = xA;
        }
      }
    }
  }

  template <uint32_t SZ, class NT>
  __device__ FORCEINLINE void local_sort_regs(NT (&A)[SZ])
  {
    int uu = 0;
#pragma unroll
  // this produces sequences of length K: alternating ascending - descending
    for(int32_t i = 2; i < K; i *= 2) { // less than K since the last step is not needed
#pragma unroll    
      for(int32_t j = i / 2; j >= 1; j /= 2) {
    // after step i, we have bitonic sequences of size i*2
    // i.e., after the whole loop we have a bitonic sequence of WARP_SIZE
    // if (bit1 == 0) { // sort min <-- max
    //   d = (bit0 == 0) ? min(d, xd) : max(d, xd);
    // } else { // sort max <-- min
    //   d = (bit0 == 0) ? max(d, xd) : min(d, xd);
    // }
#pragma unroll
        for(uint32_t n = 0; n < SZ; n++) {
          if((n & j) == 0) {
            int32_t nj = n ^ j, biti = (n & i) == i;
            if((biti ^ (A[n] < A[nj])) == 0){
              xswap(A[n], A[nj]);
            }
          }
        } // for n
      } // for j
    } // for i
#pragma unroll
    for(int32_t i = K/2; i >= 1; i /= 2) {
#pragma unroll      
      for(uint32_t n = 0; n < SZ; n++) {
        if((n & i) == 0) {
          int32_t ni = n ^ i;
          if(!(A[n] < A[ni])) {
            xswap(A[n], A[ni]);
          }
        }
      } // for n
    } // for i
    // 120 comparisons vs 80 comparision for bitonic topK
  }

  // merge results from upper half of threads to the lower ones
  template <uint32_t SZ, class NT>
  __device__ FORCEINLINE void merge_regs(NT (&A)[SZ])
  {
    auto lane = gpuLaneId();
    if(lane >= WARP_SIZE/2) {
#pragma unroll      
      for(uint32_t n = 0; n < SZ/2; n++) {
        xswap(A[n], A[SZ - 1 - n]);
      }
    }
#pragma unroll
    for(uint32_t n = 0; n < SZ; n++) {
      // for now assume that SZ == K
      NT U = A[n];
      // read from upper half of threads
      U = gpuShuffle< ShflType::Down >(U, WARP_SIZE/2); 
      A[n] = (A[n] < U ? U : A[n]);
    }
  }

  template <uint32_t SZ, class NT>
  __device__ FORCEINLINE void rebuild_regs(NT (&A)[SZ])
  {
#pragma unroll    
    for(int32_t j = K / 2; j >= 1; j /= 2) {
#pragma unroll
      for(uint32_t n = 0; n < SZ; n++) {
        if((n & j) == 0) {
          int32_t nj = n ^ j, bitK = (n & K) == K;
          if((bitK ^ (A[n] < A[nj])) == 0){
            xswap(A[n], A[nj]);
          }
        }
      } // for n
    } // for j
  }

  // reduces values in A to return top-K elements per warp
  template <bool FinalRebuild, uint32_t SZ, class NT>
  __device__ FORCEINLINE void final_reduce(NT (&A)[SZ], const NT& minV, uint32_t I = 1) 
  {
    const auto lane = gpuLaneId();
#pragma unroll    
    for(uint32_t i = WARP_SIZE / K, div = WARP_SIZE/2; i > 1; i /= 2, div /= 2) {
      // same as: (lane & ~(K-1))*2 + (lane & (K-1))
      auto idx = (lane / K)*2*K + lane % K;
#pragma unroll      
      for(uint32_t v = 0; v < I; v++) {
        merge(A[v]);
        A[v] = gpuShuffle< ShflType::Sync >(A[v], idx); 
        if(lane >= div) A[v] = minV; // remove lower unused elements
      }
      if(i > 2 || FinalRebuild) { // final rebuild is not always necessary
        rebuild(A);
      }
    } // for
  }

  // inter-warps merge of sorted values in A: results are returned by the 1st warp
  // shared memory requirements: sizeof(NT)*BlockSize / 2
  // values of A must be alternating sorted k-sequences
  template < uint32_t SZ, class NT >
  __device__ FORCEINLINE void merge_warps(const uint32_t tid, 
        const uint32_t blockSz, NT (&A)[1], NT *sh) 
  {
    const auto warpId = tid / WARP_SIZE;
#if 0 
    sh[tid] = A[0];
    __syncthreads();
    if(warpId == 0) {
      for(uint32_t i = 1; i < blockSz / WARP_SIZE; i++) {
        auto B = sh[lane + i*WARP_SIZE];
        merge(A[0], B);
        rebuild(A);
      }
    }
#else
    if(int32_t idx = tid - blockSz / 2; idx >= 0) { // only upper threads need to write
      sh[idx] = A[0];
    }
    for(uint32_t ofs = blockSz / 2, memofs = 0; ofs >= WARP_SIZE; ofs = ofs / 2) 
    {
      __syncthreads();
      if(tid < ofs) {
        // save upper half in shared memory  
        //int32_t idx = tid - (bidx / WARP_SIZE + 1) / 2 * WARP_SIZE; 
        auto B = sh[tid + memofs];
        merge(A[0], B);
        rebuild(A);
        // actually only upper blocks need to write back
        if(tid >= ofs/2) {
          sh[tid + memofs] = A[0]; // write to the same place where we read from
        }
      }
      memofs += ofs/2;
    } // for ofs
#endif
  }

  // each thread does the following: 
  // 1. local_sort
  // 2. merge results pairwise
  // 3. again merge to local mem

}; // BitonicTopK

//__attribute__((amdgpu_flat_work_group_size(1, 256)))

template <uint32_t K, typename KT>
__launch_bounds__(1024, 1)
__global__ void RunTopK_bitonic_shuffle(const KT * __restrict__ data, uint32_t n, 
    KT * __restrict__ result, uint32_t * __restrict__ result_idxs, uint32_t k) 
{
  using TopK = BitonicTopK< KT, K >;
  using KVT = typename TopK::KVT;
  TopK topk;
  
  constexpr auto minVal = std::numeric_limits< KT >::min();
  const uint32_t bidx = blockIdx.x, blockSz = blockDim.x;
  auto *in = data + n * bidx;
  const uint32_t tid = threadIdx.x, lane = tid % WARP_SIZE;
  uint32_t idx = tid;

  constexpr uint32_t I = 1;
  KVT A[I] = {idx < n ? in[idx] : minVal, idx};
  //LOUTZ("original: %d", A);
  topk.local_sort(A, I);
    
  uint32_t i = 0;
  for(idx += blockSz; ; idx += blockSz, i++) {
    auto warpId = idx & ~(WARP_SIZE - 1);
    //OUTZ("idx: %d, warpID: %d", idx, warpId);
    if(warpId >= n) { // retire completely unused warps
       break;
    }
     //__builtin_amdgcn_update_dpp();
    //__builtin_amdgcn_ds_swizzle();

    KVT B[I] = {idx < n ? in[idx] : minVal, idx};
    // if(tid == 1023)
    // LOUTZ("loaded B: %d = %d", idx, B.key);
    topk.local_sort(B);
    topk.merge(A, B);
    //OUTZ("%d: idx: %d: xA = %d, A = %d; xB = %d, B = %d", tid, idx, xA, A, xB, B);
    topk.rebuild(A);
    
    // if(idx < n)
    // if(lane == 0) {
    //   OUTZ("i = %d/%d; n = %d; mywarp: %d; my range: [%d; %d]", i, warpId, n, 
    //         tid / WARP_SIZE, warpId, warpId + WARP_SIZE);
    // }
    // __syncthreads();
  } // for idx

  topk.merge_warps(tid, blockSz, A, (KVT *)g_shared_mem);
  if(tid < WARP_SIZE) {

    topk.final_reduce(A, KVT{minVal, 0});
 
    //LOUTZ("final: %d", A.key);
    auto vals_out = result + k * bidx + tid;
    auto idxs_out = result_idxs + k * bidx + tid;

    uint32_t diff = tid - (K - k);
    if(diff < k) { // use unsigned compare ! 
      vals_out[0] = A.key;
      idxs_out[0] = A.idx;
    }
  } // if(warpId)  
} // RunTopK_bitonic_shuffle


// B - number of top elements which are searched for each subrange [1,2,3,4]
// this also defines the minimal number of registers per thread
// K - next power-of-two value of K for top-k (real k <= K)
// for small K: K == B

// this function emulates loading ith data element from global memory
template <typename KT>
__device__ KT loadData(uint32_t tid, uint32_t N, uint32_t i) {
#if 1
  auto u = i + 13, v = tid - 11711;
  KT x = u*u*v*v % 259; 
#else
  KT x = tid*N + i + 1;
#endif
  return x;
}

template <uint32_t K__, typename KT>
__global__ void RunTopK_subranges(const KT * __restrict__ data, uint32_t n, 
    KT * __restrict__ result, uint32_t * __restrict__ result_idxs, uint32_t k) 
{
  // each thread works on 16 elements, hence one warp - 1024 elements
  using VecT = uint4;
  constexpr uint32_t K = 16, Nloads = K * sizeof(KT) / sizeof(VecT);
  uint32_t tid = threadIdx.x, blockSz = blockDim.x;

  auto shmem = (KT *)g_shared_mem;

  // block is 64 threads: 64*16 -> 1024 elements to read
  // read 4 elements by each thread: total 4 * BlockSize

  constexpr uint32_t N = 8, // subrange size processed by each thread
                     S = 2; // top-S elements are computed on-the-fly
  // for small values of K: S == K (i.e. K <= 4)

  using TopKType = BitonicTopK< KT, K >;
  TopKType topk;
  // uint4 - 4 32-bit ints or 8 16-bit values
  KT A[N], B[S]; // ascending sorted array: B[0] is the smallest
                 // B[0] <= B[1] <= .. <= B[S-1]
#pragma unroll  
  for(uint32_t i = 0; i < N; i++) {
    KT v = loadData<KT>(tid, N, i);
    if(tid < 1) {
      OUTZ("%d: v[%d] = %d", tid, i, v);
    }
#pragma unroll    
    for(uint32_t j = 0; j < S; j++) {
      if(i == 0) {
        A[j] = v;
      } else {
        if(v < A[j]) break;
        if(j != 0) A[j - 1] = A[j];
        A[j] = v;
      }
    }
  }
#pragma unroll // copy over first S sorted elements
  for(uint32_t j = 0; j < S; j++) {
    B[j] = A[j];
  }
  if(tid < 1) {
    OUTREGSZ(S, "tid: %d; sorted: %d", tid, B[i]);
  }
  // what can we do for large values of K ???
  // we can keep K-sequence in 2 registers A[0/1] ??

  // do S-way bitonic local-sort for each warp
#pragma unroll
  for(uint32_t i = S; i > 1; i /= 2) {
    topk.local_sort(B, i);
#pragma unroll  
    for(uint32_t j = 0; j < i/2; j++) { // pairwise merge: B is maximal 4
      topk.merge(B[j], B[j + (i + 1)/2]); // B = 3: A[0] + A[2]
                                          // B = 4: A[0] + A[2], A[1] + A[3] 
    } // for j
    topk.rebuild(B, i/2);
  } // for i
  if(S == 1) {
    topk.rebuild(B, 1);
  }
  // and now we need to merge all warps together..
  
  // NOTE: makes sense to put 'rebuild' at the beginning..
  topk.merge_warps(tid, blockSz, B[0], (KT *)g_shared_mem);
  auto lane = gpuLaneId();

  auto shmem_top = (KT *)g_shared_mem;
  if(tid < WARP_SIZE) {
    constexpr KT minVal = std::numeric_limits< KT >::min();
    topk.template final_reduce<false>(B[0], minVal);
    LOUTZ("final sorted: %d", B[0]);
    if(tid < K) {
      shmem_top[tid] = B[0];
    }
  }

  // threads 
  // A B C D | E F G H | I J K L
  
  // for K = 4, we have: A, C, G, L
  // hence only the first group is needed
  // primitive algorithm: 
  // 1. load K elements to shared memory: shmem_top[K]
  // 2. each thread needs to check if its S elements B[0..S) are in shmem_top
  int z = 0;
  __syncthreads();

  OUTREGSZ(S, "tid: %d; initial: %d", tid, A[i]);
  return;

  // TODO what to do with duplicate elements ???
#pragma unroll  
  for(uint32_t i = 0; i < K; i++) {
    for(uint32_t j = 0; j < S; j++) {
      // NOTE: here we need to compare against A since these are not touched
      if(shmem_top[i] == A[j]) {
        z++;
      }
    }
  }
  // max (K/S) hits => (K/S) * rangeSz elements to process
  OUTZ("%d: number of hits: %d", tid, z);
 
  return;

// #pragma unroll
//   for(uint32_t i = 0, ofs = 0; i < Nloads; i++, ofs += blockSz) {
//     ((VecT *)regs)[i] = ((const VecT * __restrict__ )data)[ofs + tid];
//   }
  // N elements per thread
  // 1. run topk on 2 max delegates for each thread: total of 2*BlockSz elements
  // 2. take only those subranges of size N where both delegates present:
  // 3. in worst case, all 2 delegates from each group present: hence in total
  //    we have (k/2)*N elements to be scanned for 2nd topk
  // for example: k = 16, N = 16 => 128 elements to be scanned (quite small)
} // RunTopK_subranges

template <typename KT>
__global__ void RunTopK_test(KT* data, uint32_t n, KT* result, uint32_t* result_idxs, uint32_t k) 
{
  
  uint32_t lane = threadIdx.x;
  if(lane >= WARP_SIZE)
    return;

  // auto z = __builtin_amdgcn_ubfe(lane, 0, 1);
  // auto z2 = __builtin_amdgcn_ubfe(lane, 1, 1);
  
  constexpr uint32_t K = 64;
  using TopK = BitonicTopK< KT, K >;
  using KVT = typename TopK::KVT;
  TopK topk;

  KVT A[1] = {((lane+1)*(lane-1)*lane) % 113, lane}, 
      minVal{0, 0};
  KVT B[1] = {((lane+1)*(lane-1)*(lane-1)) % 97, lane};

#define XS(idx, val)  if(lane == idx) A[0].key = val
  XS(0, 3);
  XS(1, 7);
  XS(2, 4);
  XS(3, 8);
  XS(4, 6);
  XS(5, 2);
  XS(6, 1);
  XS(7, 5111);

  XS(8, 15);
  XS(9, 3);
  XS(10, 7);
  XS(11, 7);
  XS(12, 1);
  XS(13, 11);
  XS(14, 8);
  XS(15, 2);

  LOUTZ("original: A: %d B: %d", A[0].key, B[0].key);
  topk.local_sort(A);
  topk.local_sort(B);
  LOUTZ("local sorted: A: %d; B: %d", A[0].key, B[0].key);

  // merge K-sorted runs and keep only K greater elements
  topk.merge(A[0], B[0]);
  LOUTZ("merged bitonic K-sequences: A: %d", A[0].key);

  topk.rebuild(A);
  LOUTZ("rebuilt K-runs: A: %d", A[0].key);

  // we have in total WARP_SIZE / K sorted runs in A
  // 32 / 8 = 4
  // merge = 2
  // merge = 1
  topk.template final_reduce< true >(A, minVal);
  //LOUTZ("local sorted: %d; merged: %d, shifted: %d", A, mA, zA);
  OUTZ("%d: squashed: %d", lane, A[0].key);
}

template <typename T, size_t K>
void* GetTopKKernelForK(size_t n_threads) {
#if USE_TOPK_DEFAULT
  return reinterpret_cast<void*>(RunTopK_default<K, T>);
#else  
  return reinterpret_cast<void*>(
        //RunTopK_subranges<K, T>);
        //RunTopK_bitonic_shuffle<K, T>);
        RunTopK_test<T>);
#endif
}

#endif  // TOPK_KERNEL_CU_H_
