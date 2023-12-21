
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
/opt/rocm/include/hip/amd_detail/amd_warp_functions.h:
static constexpr int warpSize = __AMDGCN_WAVEFRONT_SIZE;

__device__
inline
int __shfl_xor(int var, int lane_mask, int width = warpSize) {
    int self = __lane_id();
    int index = self^lane_mask;
    index = index >= ((self+width)&~(width-1))?self:index;
    return __builtin_amdgcn_ds_bpermute(index<<2, var);
}
*/

#define OUTZ(fmt, ...) printf(fmt"\n", ##__VA_ARGS__)
#define OUTREGSZ(fmt, ...) \
    for(uint32_t i = 0; i < N; i++) { \
      printf(fmt"\n", ##__VA_ARGS__); \
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
  __device__ FORCEINLINE void Reduce(KVT tmp[K], int num_lanes) {
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

__device__ FORCEINLINE uint32_t gpuLaneId() {
  uint32_t lane_id;
#if !COMPILE_FOR_ROCM
#if __clang__
  return __nvvm_read_ptx_sreg_laneid();
#else   // __clang__
  asm("mov.u32 %0, %%laneid;" : "=r"(lane_id));
#endif  // __clang__
#else
  lane_id = __lane_id();
#endif
  return lane_id;
}

// K - topk size
// N - elements per thread
template < class KT, uint32_t K, uint32_t N = 0 >
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
  template <class NT>
  __device__ FORCEINLINE void local_sort(NT& A) 
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
        int32_t bitj = gpuGetBit(lane, j);
        auto xA = gpuShuffle< ShflType::Xor >(A, 1u << j); 
        if((biti ^ bitj ^ (A < xA)) == 0) {
          A = xA;
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
  __device__ FORCEINLINE void rebuild(NT& A) 
  {
    auto lane = gpuLaneId();
    // rebuild bitonic sequences of length K to sorted ones
    int32_t bitK = gpuGetBit(lane, logK); 
#pragma unroll
    for(int32_t j = logK - 1; j >= 0; j--) {
      int32_t bitj = gpuGetBit(lane, j);
      // src = lane ^ (1u << j);
      auto xA = gpuShuffle< ShflType::Xor >(A, 1u << j); 
      if((bitK ^ bitj ^ (A < xA)) == 0) {
        A = xA;
      }
    }
  }

  template <class NT>
  __device__ FORCEINLINE void merge(NT& A, NT& B) 
  {
      // merge K-sorted runs and keep only K greater elements
      merge(A);
      merge(B);
      // merge maximal elements of A and B: shift B by K elements
      // A: ABCD xxxx EFGH xxxx IJKL xxxx
      // B: xxxx ABCD xxxx EFGH xxxx IJKL 
      // B[lane] = B[lane - K]
      auto lane = gpuLaneId();
      if(lane & K) {
        A = B;
      }
  }

  __device__ void operator()(const KT * __restrict__ data, uint32_t n, 
      KT * __restrict__ result, uint32_t * __restrict__ result_idxs, uint32_t k) 
  {
    constexpr uint32_t WarpSize = WAVEFRONT_SIZE;
    constexpr auto minVal = std::numeric_limits< KT >::min();

    const uint32_t bidx = blockIdx.x, blockSz = blockDim.x;
    auto in = data + n * bidx;
    const uint32_t thid = threadIdx.x, lane = thid % WarpSize;
    uint32_t idx = thid;

    KVT A{idx < n ? in[idx] : minVal, idx};
    //LOUTZ("original: %d", A);
    local_sort(A);
    
    uint32_t i = 0;
    for(idx += blockSz; ; idx += blockSz, i++) {
      auto warpId = idx & ~(WarpSize - 1);
      //OUTZ("idx: %d, warpID: %d", idx, warpId);
      if(warpId >= n) { // retire completely unused warps
         break;
      }
     //__builtin_amdgcn_update_dpp();
    //__builtin_amdgcn_ds_sizzle();

      KVT B{idx < n ? in[idx] : minVal, idx};
      // if(thid == 1023)
      // LOUTZ("loaded B: %d = %d", idx, B.key);
      local_sort(B);
      merge(A, B);
      //OUTZ("%d: idx: %d: xA = %d, A = %d; xB = %d, B = %d", thid, idx, xA, A, xB, B);
      rebuild(A);
    
      // if(idx < n)
      // if(lane == 0) {
    //   OUTZ("i = %d/%d; n = %d; mywarp: %d; my range: [%d; %d]", i, warpId, n, 
    //         thid / WarpSize, warpId, warpId + WarpSize);
    // }
    // __syncthreads();
    } // for idx
    auto sh = (KVT *)shmem;

    const auto warpId = thid / WarpSize;
#if 0 
    sh[thid] = A;
     __syncthreads();
    if(warpId == 0) {
      for(uint32_t i = 1; i < blockSz / WarpSize; i++) {
        auto B = sh[lane + i*WarpSize];
        merge(A, B);
        rebuild(A);
      }
    }
#else
    if(int32_t idx = thid - blockSz / 2; idx >= 0) { // only upper threads need to write
      sh[idx] = A;
    }
    for(uint32_t ofs = blockSz / 2, memofs = 0; ofs >= WarpSize; ofs = ofs / 2) 
    {
      __syncthreads();
      if(thid < ofs) {
        // save upper half in shared memory  
        //int32_t idx = thid - (bidx / WarpSize + 1) / 2 * WarpSize; 
        auto B = sh[thid + memofs];
        merge(A, B);
        rebuild(A);
        // actually only upper blocks need to write back
        if(thid >= ofs/2) {
          sh[thid + memofs] = A; // write to the same place where we read from
        }
      }
      memofs += ofs/2;
    } // for ofs
#endif
    if(warpId == 0) {
#pragma unroll      
      for(uint32_t i = WarpSize / K, div = WarpSize/2; i > 1; i /= 2, div /= 2) {
        merge(A);
        // same as: (lane & ~(K-1))*2 + (lane & (K-1))
        auto idx = (lane / K)*2*K + lane % K;
        A = gpuShuffle< ShflType::Sync >(A, idx); 
        if(lane >= div) {
          A = KVT{minVal, 0}; // remove lower unused elements
        }
        rebuild(A);
      }
  
      //LOUTZ("final: %d", A.key);
      auto vals_out = result + k * bidx + thid;
      auto idxs_out = result_idxs + k * bidx + thid;

      uint32_t diff = thid - (K - k);
      if(diff < k) { // use unsigned compare ! 
        vals_out[0] = A.key;
        idxs_out[0] = A.idx;
      }
    } // if(warpId)  
  }

  template <class NT>
  __device__ FORCEINLINE void local_sort_regs(NT (&A)[N])
  {
    int uu = 0;
#pragma unroll
  // this produces sequences of length K: alternating ascending - descending
    for(int32_t i = 2; i < K; i *= 2) { // less than K since the last step is not needed
#pragma unroll    
      for(int32_t j = i / 2; j >= 1; j /= 2) {
    // after step i, we have bitonic sequences of size i*2
    // i.e., after the whole loop we have a bitonic sequence of WarpSize
    // if (bit1 == 0) { // sort min <-- max
    //   d = (bit0 == 0) ? min(d, xd) : max(d, xd);
    // } else { // sort max <-- min
    //   d = (bit0 == 0) ? max(d, xd) : min(d, xd);
    // }
#pragma unroll
        for(uint32_t n = 0; n < N; n++) {
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
      for(uint32_t n = 0; n < N; n++) {
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
  template <class NT>
  __device__ FORCEINLINE void merge_regs(NT (&A)[N])
  {
    auto lane = gpuLaneId();
    constexpr uint32_t WarpSize = WAVEFRONT_SIZE;

    if(lane >= warpSize/2) {
#pragma unroll      
      for(uint32_t n = 0; n < N/2; n++) {
        xswap(A[n], A[N-1-n]);
      }
    }
#pragma unroll
    for(uint32_t n = 0; n < N; n++) {
      // for now assume that N == K
      NT U = A[n];
      // read from upper half of threads
      U = gpuShuffle< ShflType::Down >(U, WarpSize/2); 
      A[n] = (A[n] < U ? U : A[n]);
    }
  }

    template <class NT>
  __device__ FORCEINLINE void rebuild_regs(NT (&A)[N])
  {
#pragma unroll    
    for(int32_t j = K / 2; j >= 1; j /= 2) {
#pragma unroll
      for(uint32_t n = 0; n < N; n++) {
        if((n & j) == 0) {
          int32_t nj = n ^ j, bitK = (n & K) == K;
          if((bitK ^ (A[n] < A[nj])) == 0){
            xswap(A[n], A[nj]);
          }
        }
      } // for n
    } // for j
  }

  // each thread does the following: 
  // 1. local_sort
  // 2. merge results pairwise
  // 3. again merge to local mem

}; // BitonicTopK

//__attribute__((amdgpu_flat_work_group_size(1, 256)))

template <uint32_t K, typename KT>
__launch_bounds__(1024, 1)
__global__ void RunTopK_new(const KT * __restrict__ data, uint32_t n, 
    KT * __restrict__ result, uint32_t * __restrict__ result_idxs, uint32_t k) 
{
  BitonicTopK< KT, K >()(data, n, result, result_idxs, k);
}

template <uint32_t K__, typename KT>
__global__ void RunTopK_subranges(const KT * __restrict__ data, uint32_t n, 
    KT * __restrict__ result, uint32_t * __restrict__ result_idxs, uint32_t k) 
{
  // each thread works on 16 elements, hence one warp - 1024 elements
  using VecT = uint4;
  constexpr uint32_t K = 16, Nloads = K * sizeof(KT) / sizeof(VecT);
  uint32_t thid = threadIdx.x, blockSz = blockDim.x;

  // block is 64 threads: 64*16 -> 1024 elements to read
  // read 4 elements by each thread: total 4 * BlockSize

  constexpr uint32_t N = K;
  using TopKType = BitonicTopK< KT, K, N >;
  TopKType topk;
  // uint4 - 4 32-bit ints or 8 16-bit values
  KT A[N];

#pragma unroll  
  for(uint32_t i = 0; i < N; i++) {
    auto u = i + 1;
    A[i] = u*u*(thid + 17)*1223 % 117; 
    if(thid == 0) {
      OUTZ("A[%d] = %d", i, A[i]);
    }
    if(i >= 1) {
      if(A[0] < A[i]) {
        xswap(A[0], A[1]);
        if(i >= 2) {
          xswap(A[0], A[i]);
        }
      } else if(A[1] < A[i]) {
        xswap(A[1], A[i]);
      }
    }
  }
  if(thid == 0) {
    OUTREGSZ("%d orig: %d", i, A[i]);
  }

// #pragma unroll
//   for(uint32_t i = 0, ofs = 0; i < Nloads; i++, ofs += blockSz) {
//     ((VecT *)regs)[i] = ((const VecT * __restrict__ )data)[ofs + thid];
//   }
  // N elements per thread
  // 1. run topk on 2 max delegates for each thread: total of 2*BlockSz elements
  // 2. take only those subranges of size N where both delegates present:
  // 3. in worst case, all 2 delegates from each group present: hence in total
  //    we have (k/2)*N elements to be scanned for 2nd topk
  // for example: k = 16, N = 16 => 128 elements to be scanned (quite small)

  auto lane = gpuLaneId();
  topk.local_sort(A[0]);
  topk.local_sort(A[1]);
  topk.merge(A[0], A[1]);
  // now the question is: how to identify from which subrange a particular element came ??

  LOUTZ("final sorted: %d", A[0]);

#if 0
  topk.local_sort_regs(A);
  if(thid == 0) {
    OUTREGSZ("%d 0-before merge: %d", i, A[i]);
  }
  if(thid == 16) {
    OUTREGSZ("%d 16-before merge: %d", i, A[i]);
  }

  topk.merge_regs(A);
  if(thid == 0) {
    OUTREGSZ("%d after merge: %d", i, A[i]);
  }
  topk.rebuild_regs(A);
  if(thid == 0) {
    OUTREGSZ("%d after rebuild: %d", i, A[i]);
  }
#endif
}

template <typename KT>
__global__ void RunTopK_test(KT* data, uint32_t n, KT* result, uint32_t* result_idxs, uint32_t k) 
{
  constexpr uint32_t WarpSize = WAVEFRONT_SIZE;
  uint32_t lane = threadIdx.x;
  if(lane >= WarpSize)
    return;

  // auto z = __builtin_amdgcn_ubfe(lane, 0, 1);
  // auto z2 = __builtin_amdgcn_ubfe(lane, 1, 1);
  
  constexpr uint32_t K = 4;
  using TopK = BitonicTopK< KT, K >;
  using KVT = typename TopK::KVT;
  TopK topk;

  KVT A{((lane+1)*(lane-1)*lane) % 113,lane}, 
      minVal{0, 0};
  KVT B{((lane+1)*(lane-1)*(lane-1)) % 97, lane};

#define XS(idx, val)  if(lane == idx) A.key = val
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
  topk.local_sort(A);
  topk.local_sort(B);
  LOUTZ("local sorted: A: %d; B: %d", A, B);

  // merge K-sorted runs and keep only K greater elements
  topk.merge(A, B);
  LOUTZ("merged bitonic K-sequences: A: %d", A);

  topk.rebuild(lane, A);
  LOUTZ("rebuilt K-runs: A: %d", A);

  // we have in total WarpSize / K sorted runs in A
  // 32 / 8 = 4
  // merge = 2
  // merge = 1
#pragma unroll
  for(uint32_t i = WarpSize / K, div = WarpSize/2; i > 1; i /= 2, div /= 2) {
    topk.merge(A);
    // same as: (lane & ~(K-1))*2 + (lane & (K-1))
    auto idx = (lane / K)*2*K + lane % K;
    A = gpuShuffle< ShflType::Sync >(A, idx); 
    if(lane >= div) {
      A = minVal; // remove lower unused elements
    }
    topk.rebuild(A);
  }
  //LOUTZ("local sorted: %d; merged: %d, shifted: %d", A, mA, zA);
  OUTZ("%d: squashed: %d", lane, A);
}

template <typename T, size_t K>
void* GetTopKKernelForK(size_t n_threads) {
#if USE_TOPK_DEFAULT
  return reinterpret_cast<void*>(RunTopK_default<K, T>);
#else  
  return reinterpret_cast<void*>(RunTopK_subranges<K, T>);
  //return reinterpret_cast<void*>(RunTopK_new<K, T>);
          //RunTopK_test<T>);
#endif
}

#endif  // TOPK_KERNEL_CU_H_
