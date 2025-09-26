
#include "common_funcs.cu.h"
#include "topk_kernel.h"
#include "common/common_utils.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

template <size_t K, typename NT>
struct TopK {
  struct KVT {
    NT key;
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
  __device__ void PerWarpTopK(const NT* key, int n) {

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
  __device__ void MergeTopKs(NT *keys, uint32_t *idxs) {
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

extern __device__ __shared__ int32_t g_shared_mem[];

template < class NT, size_t K >
__launch_bounds__(1024, 1) __global__
    void PerWarpTopK_kernel(const NT* data, uint32_t n, 
                            NT* result, uint32_t* result_idxs, uint32_t k) 
{
  TopK<K, NT> obj(g_shared_mem, k);
  
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

void calcOccupancy(const void *kernel) {
  // size_t dynSHMem = 0;
  // CHK(cudaOccupancyAvailableDynamicSMemPerBlock(&dynSHMem, kernel, 4, 512)); 
  // VLOG(0) << "Shared mem available: " << (double)dynSHMem/1024.0 << "KB" ;

  int numBlocks = 0;
  CHK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, 
      kernel, 512, 24*1024));
  VLOG(0) << "Max active blocks: " << numBlocks;

  int minGridSize = 0, blockSize = 0;
  CHK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 
      32*1024, 512));
  VLOG(0) << "minGridSize: " << minGridSize << " potential BlockSize: " << blockSize;
}

template < class T >
void TypedTopK(TopkArgs& args) 
{
  // Estimate number of threads per block that can run concurrently given the
  // register footprint.
  size_t threads_per_block =
      std::min(512 * (16 / args.k), kTopKMaxThreadsPerBlock);
  // Minimum amount of data that each thread needs to receive for the algorithm.
  size_t min_slice = std::bit_floor(args.num_elems / std::bit_ceil(args.k));
  VLOG(0) << "threads_per_block, min_slice: " << threads_per_block << ',' << min_slice;
  uint32_t num_threads = std::min(threads_per_block, min_slice);

  auto get_kernel = [](int k) -> decltype(&PerWarpTopK_kernel<T, 16>) {
    // if (k <= 1) return PerWarpTopK_kernel<T, 1>;
    // if (k <= 2) return PerWarpTopK_kernel<T, 2>;
    // if (k <= 4) return PerWarpTopK_kernel<T, 4>;
    //if (k <= 8) return PerWarpTopK_kernel<T, 8>;
    if (k <= 16) return PerWarpTopK_kernel<T, 16>;
    return nullptr;
  };

  uint32_t blocks_per_grid = args.batch_size;
  constexpr size_t max_kv_size = sizeof(uint64_t);
  uint32_t shmem_size = std::bit_ceil(args.k) * max_kv_size * WARP_SIZE;

  VLOG(0) << "Testing N = " << args.num_elems << "; K = " << args.k <<
          "; batch_size: " << args.batch_size << 
          "; n_blocks: " << blocks_per_grid << "; shmem_size: " << shmem_size
        << " num_threads: " << num_threads;

  auto kernel = reinterpret_cast< void *>(get_kernel(args.k));
  void* kernel_args[] = {&args.data, &args.num_elems, &args.top_elements,
                         &args.top_indices, &args.k};
  calcOccupancy(kernel);

  CU_BEGIN_TIMING(0)
  (void)cudaLaunchKernel(kernel, blocks_per_grid, num_threads, kernel_args,
                        shmem_size, 0);
  CU_END_TIMING("TopK N = %zu; K = %u; batch_size: %u", 
      args.num_elems, args.k, args.batch_size);

  CHK(cudaPeekAtLastError());
  (void)cudaDeviceSynchronize();                       
}

void RunPerWarpTopK(TopkArgs& args) {
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
      return TypedTopK< float >(args);
    case TopKType::F64: 
      break;
  }
  throw std::runtime_error("NYI");
}
