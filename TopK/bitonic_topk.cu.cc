
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

// K - topk size
// N - elements per thread
template < uint32_t K, class NT >
struct BitonicTopK {
  
  constexpr static uint32_t logK = log2xN(K);
  struct KVT {
    NT key;
    uint32_t idx;
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

template <uint32_t K, typename NT>
__launch_bounds__(1024, 1)
__global__ void RunTopK_bitonic_shuffle(const NT * __restrict__ data, uint32_t n, 
    NT * __restrict__ result, uint32_t * __restrict__ result_idxs, uint32_t k) 
{
  using TopK = BitonicTopK< K, NT >;
  using KVT = typename TopK::KVT;
  TopK topk;
  
  constexpr auto minVal = std::numeric_limits< NT >::min();
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

  topk.merge_warps(tid, blockSz, A, (KVT *)g_shared_mem);
  if(tid < WARP_SIZE) {
    // NOTE: do we need that final rebuild ??
    topk.template final_reduce< true >(A, KVT{minVal, 0});
 
    //LOUTZ("final: %d", A[0].key);
    auto vals_out = result + k * bidx + tid;
    auto idxs_out = result_idxs + k * bidx + tid;

    uint32_t diff = tid - (K - k);
    if(diff < k) { // use unsigned compare ! 
      vals_out[0] = A[0].key;
      idxs_out[0] = A[0].idx;
    }
    LOUTZ("key: %u idx: %u", A[0].key, A[0].idx);
  } // if(warpId)  
} // RunTopK_bitonic_shuffle


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

template < class T >
void TypedTopK(TopkArgs& args) 
{
#define XKERNEL(K) if(args.k <= K) { \
      return reinterpret_cast< void *>(RunTopK_bitonic_shuffle<K, T>); }
  auto get_kernel = [&args]() -> void * {
    //XKERNEL(1) XKERNEL(2) XKERNEL(4) XKERNEL(8) 
    XKERNEL(16)
    return nullptr;
  };

  uint32_t n_blocks = args.batch_size;
  
  //num_threads = args.num_elements / std::bit_ceil(args.k);
  uint32_t num_threads = 64;
  num_threads = (num_threads + WARP_SIZE - 1) & ~(WARP_SIZE - 1);
  // 16Kb per block => 4096 words of mem; 512 threads => 16 elements per thread
  uint32_t shmem_size = num_threads * sizeof(uint32_t) / 2;

  VLOG(0) << "Testing N = " << args.num_elems << "; K = " << args.k <<
          "; batch_size: " << args.batch_size << 
          "; n_blocks: " << n_blocks << "; shmem_size: " << shmem_size
        << " num_threads: " << num_threads;

  auto kernel = get_kernel();
  void* kernel_args[] = {&args.data, &args.num_elems, &args.top_elements,
                         &args.top_indices, &args.k};

  CU_BEGIN_TIMING(0)
  (void)cudaLaunchKernel(kernel, n_blocks, num_threads, kernel_args,
                        shmem_size, 0);
  CU_END_TIMING("TopK N = %zu; K = %zu; batch_size: %zu", 
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
