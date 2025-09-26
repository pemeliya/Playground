
#include "topk_kernel.h"
#include "common_funcs.cu.h"
#include "common/common_utils.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

// bitonic TopK: https://github.com/anilshanbhag/gpu-topk

extern __device__ __shared__ int32_t g_shared_mem[];

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

template < class T, class U, class Common = std::common_type_t<T, U> >
Common CeilOfRatio(T num, U denom) {
  return (num + denom - 1) / denom;
}

// K - topk size
// N - elements per thread
template < uint32_t K, class NT >
struct BitonicTopK {
  
  constexpr static uint32_t logK = log2xN(K);
  struct KVT {
    NT key;
    uint32_t idx;
    // NOTE NOTE: I am not sure if we need this complicated condition!!
    __device__ FORCEINLINE bool operator <(const KVT& rhs) {
      return key == rhs.key ? idx < rhs.idx : key < rhs.key;
    }
  };

  // local sort produces sorted runs of size K: ascending / descending
  // i.e. for K = 4: [3 5 8 9; 5 3 1 1; 2 5 6 10; 11 4 3 2]
  // index I - is just vector size we want to use
  template < uint32_t SZ > // SZ >= I
  __device__ FORCEINLINE void local_sort(KVT (&A)[SZ], uint32_t I = 1)
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

/*
int __shfl_xor(int var, int lane_mask, int width = WARP_SIZE) {
    int self = __lane_id();
    int index = self^lane_mask;
    index = index >= ((self+width)&~(width-1))?self:index;
    return __builtin_amdgcn_ds_bpermute(index<<2, var);
}*/
        
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

  __device__ FORCEINLINE void merge(KVT& A) {
    auto xA = gpuShuffle< ShflType::Xor >(A, K); 
    A = A < xA ? xA : A;
  }

  __device__ FORCEINLINE void merge(KVT& A, KVT& B) 
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

  template <uint32_t SZ>
  __device__ FORCEINLINE void rebuild(KVT (&A)[SZ], uint32_t I = 1)
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

  template <uint32_t SZ>
  __device__ FORCEINLINE void local_sort_regs(KVT (&A)[SZ])
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
  template <uint32_t SZ>
  __device__ FORCEINLINE void merge_regs(KVT (&A)[SZ])
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
      KVT U = A[n];
      // read from upper half of threads
      U = gpuShuffle< ShflType::Down >(U, WARP_SIZE/2); 
      A[n] = (A[n] < U ? U : A[n]);
    }
  }

  template <uint32_t SZ>
  __device__ FORCEINLINE void rebuild_regs(KVT (&A)[SZ])
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
  template <bool FinalRebuild, uint32_t SZ >
  __device__ FORCEINLINE void final_reduce(KVT (&A)[SZ], const KVT& minV, uint32_t I = 1) 
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
  // shared memory requirements: sizeof(KVT)*BlockSize / 2
  // values of A must be alternating sorted k-sequences
  __device__ FORCEINLINE void merge_warps(const uint32_t tid, 
        const uint32_t blockSz, KVT (&A)[1], KVT *sh) 
  {
    const auto warpId = tid / WARP_SIZE;
#if 0 // debug version simple
    const auto lane = gpuLaneId();
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
    // NOTE NOTE: this version assumes that blockSz is a power of two: full binary tree
    // warps:
    // 0 1 2 3 4 5 6 7
    // step 1: 0+4 1+5 2+6 3+7

    // warps:
    // 0 1 2 3 4 5 6 7
    // step 1: 0+4 1+5 2+6 3+7


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

template <typename NT>
__global__ void RunTopK_test(NT* data, uint32_t n, NT* result, uint32_t* result_idxs, uint32_t k) 
{
  
  uint32_t lane = threadIdx.x;
  if(lane >= WARP_SIZE)
    return;

  // auto z = __builtin_amdgcn_ubfe(lane, 0, 1);
  // auto z2 = __builtin_amdgcn_ubfe(lane, 1, 1);
  
  constexpr uint32_t K = 64;
  using TopK = BitonicTopK< K, NT >;
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

//__attribute__((amdgpu_flat_work_group_size(1, 256)))

template < class NT > 
struct TopKKernelParams {
  const NT SGLOBAL* __restrict__ in_values;
  uint32_t SGLOBAL* __restrict__ in_indices;
  NT SGLOBAL* __restrict__ out_values; 
  uint32_t SGLOBAL* __restrict__ out_indices;
  uint32_t n_total, n_per_block, k;
};

template < uint32_t K, typename NT>
__launch_bounds__(512, 1) // HACK HACK
__global__ void RunTopK_bitonic_shuffle(TopKKernelParams< NT > args) 
{
  using TopK = BitonicTopK< K, NT >;
  using KVT = typename TopK::KVT;
  TopK topk;
  
  constexpr auto minVal = std::numeric_limits< NT >::min();
  const uint32_t bidx = blockIdx.x, bidy = blockIdx.y, BlockSz = blockDim.x;
  auto *in = args.in_values + args.n_total * bidy; // different batches
  const uint32_t tid = threadIdx.x, lane = tid % WARP_SIZE;
  uint32_t idx = tid + BlockSz * bidx;

  constexpr uint32_t I = 1;
  KVT A[I] = {idx < args.n_total ? in[idx] : minVal, idx};
  // LOUTZ("%d first idx: %d / %d", bidx, A[0].key, A[0].idx);
  topk.local_sort(A, I);

  uint32_t i = 0;
  for(idx += BlockSz * gridDim.x; ; idx += BlockSz * gridDim.x, i++) {
    auto warpId = idx & ~(WARP_SIZE - 1);
    
    //OUTZ("idx: %d, warpID: %d", idx, warpId);
    if(warpId >= args.n_total) { // retire completely unused warps
       break;
    }
     //__builtin_amdgcn_update_dpp();
    //__builtin_amdgcn_ds_swizzle();

    KVT B[I] = {idx < args.n_total ? in[idx] : minVal, idx};
    // if(idx < args.n_total)
    // LOUTZ("%d loaded B: %d = %d", bidx, idx, B[0].key);

    topk.local_sort(B);
    topk.merge(A[0], B[0]);
    //OUTZ("%d: idx: %d: xA = %d, A = %d; xB = %d, B = %d", tid, idx, xA, A, xB, B);
    topk.rebuild(A);
    
    // if(idx < n)
    // if(lane == 0) {
    //   OUTZ("i = %d/%d; n = %d; mywarp: %d; my range: [%d; %d]", i, warpId, n, 
    //         tid / WARP_SIZE, warpId, warpId + WARP_SIZE);
    // }
    // __syncthreads();
  } // for idx

  // LOUTZ("final A: %u, %d", A[0].key, A[0].idx);

  topk.merge_warps(tid, BlockSz, A, (KVT *)g_shared_mem);

  // NOTE: I assume finalRebuild is only necessary when we are not planning
  // to merge results from several blocks..
  // otherwise - there is no point doing it => we can just write the result as it is to global mem
  // and then read full warps ?
  if(tid < WARP_SIZE) {
    topk.template final_reduce< true >(A, KVT{minVal, 0});
 
    //LOUTZ("final: %d", A[0].key);
    uint64_t ofs = static_cast< uint64_t>(args.k) * (bidx + gridDim.x * bidy);
    auto vals_out = args.out_values + ofs + tid;
    auto idxs_out = args.out_indices + ofs + tid;

    uint32_t diff = tid - (K - args.k);
    if(diff < args.k) { // use unsigned compare ! 
       vals_out[0] = A[0].key;
       idxs_out[0] = A[0].idx;
      // LOUTZ("batch: %d bidx: %d output key: %u idx: %u", bidy, bidx, A[0].key, A[0].idx);
    }
  } // if(warpId)  
} // RunTopK_bitonic_shuffle

// this kernel takes n_total elements (n_total = k * n_topk_seqs)
// forming N sorted topk sequences of size k, batched by blockIdx.y
// returns merged topk sequence of size k batched by blockIdx.y
template < uint32_t K, typename NT>
__launch_bounds__(kTopKMaxThreadsPerBlock, 1)
__global__ void RunTopK_bitonic_merge(TopKKernelParams< NT > args)
{
  using TopK = BitonicTopK< K, NT >;
  using KVT = typename TopK::KVT;
  TopK topk;
  
  constexpr auto minVal = std::numeric_limits< NT >::min();
  const uint32_t bidy = blockIdx.y, BlockSz = blockDim.x;
  const uint32_t tid = threadIdx.x, wid = tid / WARP_SIZE,
                 lane = tid % WARP_SIZE;

  auto *in_vals = args.in_values + args.n_total * bidy;
  auto *in_idxs = args.in_indices + args.n_total * bidy;

  // TODO: we need a loop here in general case: when args.n_total 
  // cannot be handled by a single block

  // each warp reads one size-k sequence.. might not be very efficient though..
  uint32_t idx = wid * args.k + lane;
  constexpr uint32_t I = 1;
  KVT A[I] = {minVal, 0};
  if (lane < args.k) {
    A[0].key = in_vals[idx];
    A[0].idx = in_idxs[idx];
//     OUTZ("batch: %d tid: %d original: %u %u", bidy, tid, A[0].key, A[0].idx);
  }

  // n_total = blocks.x * args.k
  // suppose block.x = 98
  // BlockSz = 1024 => 16 warps
  // hence per iteration we can process 16 packets
  // idx = lane + args.k * (warpID + i*numWarps)
  // this index must be < n_total but we need full warps!

  int i = 0;
  for(idx += BlockSz / WARP_SIZE * args.k; ; idx += BlockSz / WARP_SIZE * args.k, i++) {
    
    uint32_t maxi = idx - lane + args.k - 1; // maximal index per warp
    if(maxi >= args.n_total) { // retire completely unused warps
       break;
    }

    KVT B[I] = {minVal, 0};
    if (lane < args.k && idx < args.n_total) {
      B[0] = { in_vals[idx], in_idxs[idx]};
//      OUTZ("batch: %d tid: %d idx: %d, B: %d / %d", bidy, tid, idx, B[0].key, B[0].idx);
    }
    topk.merge(A[0], B[0]);
    topk.rebuild(A);
  }
  
  topk.merge_warps(tid, BlockSz, A, (KVT *)g_shared_mem);

  if (tid < WARP_SIZE) {
    //OUTZ("batch: %d tid: %d after merge A: %d / %d", bidy, tid, A[0].key, A[0].idx);
    topk.template final_reduce< true >(A, KVT{minVal, 0});
 
    uint64_t ofs = static_cast< uint64_t>(args.k) * bidy;
    auto vals_out = args.out_values + ofs + tid;
    auto idxs_out = args.out_indices + ofs + tid;

    uint32_t diff = tid - (K - args.k);
    if(diff < args.k) { // use unsigned compare ! 
       vals_out[0] = A[0].key;
       idxs_out[0] = A[0].idx;
       // LOUTZ("output key: %u idx: %u", A[0].key, A[0].idx);
    }
  } // if(warpId)  
}

// B - number of top elements which are searched for each subrange [1,2,3,4]
// this also defines the minimal number of registers per thread
// K - next power-of-two value of K for top-k (real k <= K)
// for small K: K == B

// this function emulates loading ith data element from global memory
template <typename NT>
__device__ NT loadData(uint32_t tid, uint32_t N, uint32_t i) {
#if 1
  auto u = i + 13, v = tid - 11711;
  NT x = u*u*v*v % 259; 
#else
  NT x = tid*N + i + 1;
#endif
  return x;
}

// warp1 - top2 elements
// warp2 - top2 elements
// warp3 - top2 elements

// 10 9 7 8 8 6
// 5 4 -1 2  2
// if top 
template <uint32_t K__, typename NT>
__global__ void RunTopK_subranges(const NT * __restrict__ data, uint32_t n, 
    NT * __restrict__ result, uint32_t * __restrict__ result_idxs, uint32_t k) 
{
  // each thread works on 16 elements, hence one warp - 1024 elements
  using VecT = uint4;
  constexpr uint32_t K = 16, Nloads = K * sizeof(NT) / sizeof(VecT);
  uint32_t tid = threadIdx.x, blockSz = blockDim.x;

  auto shmem = (NT *)g_shared_mem;

  // block is 64 threads: 64*16 -> 1024 elements to read
  // read 4 elements by each thread: total 4 * BlockSize

  constexpr uint32_t N = 8, // subrange size processed by each thread
                     S = 2; // top-S elements are computed on-the-fly
  // for small values of K: S == K (i.e. K <= 4)

  using TopKType = BitonicTopK< K, NT >;
  TopKType topk;
  // uint4 - 4 32-bit ints or 8 16-bit values
  NT A[N], B[S]; // ascending sorted array: B[0] is the smallest
                 // B[0] <= B[1] <= .. <= B[S-1]
#pragma unroll  
  for(uint32_t i = 0; i < N; i++) {
    NT v = loadData<NT>(tid, N, i);
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
  topk.merge_warps(tid, blockSz, B[0], (NT *)g_shared_mem);
  auto lane = gpuLaneId();

  auto shmem_top = (NT *)g_shared_mem;
  if(tid < WARP_SIZE) {
    constexpr NT minVal = std::numeric_limits< NT >::min();
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

template < class T >
void TypedTopK(TopkArgs& args) 
{
#define XKERNELS(K) if(args.k <= K) { \
      return std::tuple {             \
        reinterpret_cast< void *>(RunTopK_bitonic_shuffle< K, T>), \
        reinterpret_cast< void *>(RunTopK_bitonic_merge< K, T>)  \
      }; }
  auto get_kernels = [&args]() -> std::tuple<void *,void *> {
    //XKERNELS(1) XKERNELS(2) XKERNELS(4) XKERNELS(8) 
    XKERNELS(16)
    return std::tuple{ nullptr, nullptr };
  };
#undef XKERNELS

  // elems_per_thread - #of elements processed per thread (average)
  uint32_t block_sz = 256, elems_per_thread = 4,
                elems_per_block = block_sz * elems_per_thread;

  // #threads: 256, #per_block = 256*4 = 1024
  // if we have 1025 elems => that would make 2 blocks which does not make any sense...

  // simple case: we do not use vector loads: instead iterate over gridDim.x dimension
  // instead, now we need to calculate elems_per_block - so that one block can process no more than this amount
    // suppose we have n_per_block=1024
  // say we have 2049 elems => we will still use 2 blocks
  // first block 1025, 2nd block - 1024
  
  dim3 blocks;
  // Example: num = 2049 elems_per_block = 1024 -> 2 blocks
  // num = (2048 + 512) elems_per_block = 1024 -> 3 blocks (so the last block is at least half busy)
  blocks.x = (args.num_elems + elems_per_block/2) / elems_per_block;
  blocks.y = args.batch_size;
  
  // 16Kb per block => 4096 words of mem; 512 threads => 16 elements per thread
  uint32_t shmem_size = block_sz * sizeof(uint64_t); 

  auto block_values = blocks.x == 1 ? HVector< T >{} : 
                      HVector< T >(args.k * blocks.x * blocks.y, false);
  auto block_indices = blocks.x == 1 ? HVector< uint32_t >{} :
                      HVector< uint32_t >(args.k * blocks.x * blocks.y, false);

  VLOG(0) << "Testing N = " << args.num_elems << "; K = " << args.k <<
          "; batch_size: " << args.batch_size << 
          "; #blocks: [" << blocks.x << "," << blocks.y << "] shmem_size: " << shmem_size
        << " #threads: " << block_sz;

  auto [sort_kernel, merge_kernel] = get_kernels();

  TopKKernelParams<T> params = {
    .in_values = (const T SGLOBAL* __restrict__)args.data,
    .in_indices = nullptr, // unused here
    .out_values = (T SGLOBAL* __restrict__)
          (blocks.x > 1 ? block_values.devPtr : args.top_elements),
    .out_indices = (uint32_t SGLOBAL* __restrict__)
          (blocks.x > 1 ? block_indices.devPtr : args.top_indices),
    .n_total = static_cast< uint32_t >(args.num_elems),
    .n_per_block = elems_per_block, // can be set as a template ??
    .k = args.k,
  };
  void* kernel_args[] = {&params};

  CU_BEGIN_TIMING(5)
  (void)cudaLaunchKernel(sort_kernel, blocks, block_sz, kernel_args,
                        shmem_size, 0); 

  if (blocks.x > 1) {

    TopKKernelParams<T> merge_params = {
      .in_values = params.out_values,
      .in_indices = params.out_indices,
      .out_values = (T SGLOBAL* __restrict__)args.top_elements,
      .out_indices = (uint32_t SGLOBAL* __restrict__)args.top_indices,
      .n_total = static_cast< uint32_t >(blocks.x * args.k), // this is a step for batch size
      .n_per_block = 0, 
      .k = args.k,
    };

    uint32_t n_warps = std::min(blocks.x, kTopKMaxThreadsPerBlock / WARP_SIZE);
    // we have to ensure block size is a power-of-two: otherwise algo
    if (n_warps & (n_warps-1)) {
      for (int x_warps = 2; x_warps <= kTopKMaxThreadsPerBlock / WARP_SIZE; x_warps *= 2) {
        if (n_warps <= x_warps) {
          n_warps = x_warps;
          break;
        }
      }
      // e.g. if blocks.x=9 => n_warps=16 => waste of resources
      // we have: blocks.x < n_warps
      if (auto ratio = (float)blocks.x / n_warps; ratio < 0.9f) n_warps /= 2;
    }
    uint32_t merge_block_sz = n_warps * WARP_SIZE;
    
    dim3 merge_blocks(1, blocks.y);
    uint32_t mshmem_size = merge_block_sz * sizeof(uint64_t) * 4; 
    // VLOG(0) << "-------- launching merge kernel #blockSz " << merge_block_sz
    //          << " n_total " << merge_params.n_total;
    void* kargs[] = {&merge_params};
    (void)cudaLaunchKernel(merge_kernel, merge_blocks, merge_block_sz, kargs,
                        mshmem_size, 0);

  }

  CU_END_TIMING("TopK N = %zu; K = %u; batch_size: %u", 
      args.num_elems, args.k, args.batch_size);

  CHK(cudaPeekAtLastError());
  (void)cudaDeviceSynchronize();                       
}

void RunBitonicTopK(TopkArgs& args) {
  switch(args.type) {
    case TopKType::I16:
    case TopKType::U16:
    case TopKType::I32:
      break;
    case TopKType::U32:
      return TypedTopK< uint32_t >(args);
    case TopKType::F16:
    case TopKType::BF16:
      break;
    case TopKType::F32:
      break;
    case TopKType::F64: 
      break;
  }
  throw std::runtime_error("NYI");
}
