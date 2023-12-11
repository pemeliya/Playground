
#include <algorithm>
#include <stdexcept>
#include <bit>
#include <numeric>
#include <iostream>
#include <random>
#include "topk_kernel.h"
#include "common/example_utils.hpp"

size_t NumThreadsNew(size_t n, size_t k, size_t batch_size) 
{
  size_t warp_size = COMPILE_FOR_ROCM ? 64 : 32;
  size_t n_threads = (n / k), n_threads_warp = 
        ((n_threads + warp_size - 1) / warp_size) * warp_size;

  // for large k, reduce maximal block size to reduce register pressure
  size_t max_threads = (k > 8 ? 512 : 1024);
  n_threads_warp = std::min(n_threads_warp, max_threads);
  
  VLOG("n: " << n << "; k: " << k << " n_threads: " << n_threads 
      << "; n_threads_warp: " << n_threads_warp);
  return 0;
}

size_t NumThreads(size_t n, size_t k, size_t batch_size) {
  // Estimate number of threads per block that can run concurrently given the
  // register footprint.
  size_t threads_per_block =
      std::min(512 * (16 / k), kTopKMaxThreadsPerBlock);
  // Minimum amount of data that each thread needs to receive for the algorithm.
  size_t min_slice = std::bit_floor(n / std::bit_ceil(k));
  VLOG("threads_per_block, min_slice: " << threads_per_block << ',' << min_slice);
  return std::min(threads_per_block, min_slice);
}

// Helper type for converting the untyped arguments of RunTopk to TypedTopk
template <typename T>
struct TopkArgs {
  T* data;
  size_t num_elements;
  T* top_elements;
  uint32_t* top_indices;
  size_t k;
  size_t batch_size;
};

template <typename T>
void TypedTopK(TopkArgs<T> args) 
{
  uint32_t num_threads = NumThreads(args.num_elements, args.k, args.batch_size);
  if (num_threads == 0) {
    throw std::runtime_error(
        "Invalid kernel parameters. This is likely a bug in the "
        "TopkSpecializer.");
  }
  void* kernel = GetKernel<T>(num_threads, args.k);

  uint32_t blocks_per_grid = args.batch_size;
  constexpr size_t max_kv_size = sizeof(uint64_t);
  // Allocate shmem assuming we have a full reduction.
#if USE_TOPK_DEFAULT  
  uint32_t shmem_size = std::bit_ceil(args.k) * max_kv_size * WAVEFRONT_SIZE;
#else  
  //num_threads = 1024;
  num_threads = args.num_elements / std::bit_ceil(args.k);
  num_threads = (num_threads + WAVEFRONT_SIZE - 1) & ~(WAVEFRONT_SIZE - 1);
  uint32_t shmem_size = num_threads * sizeof(max_kv_size) / 2;
#endif
  VLOG("Testing N = " << args.num_elements << "; K = " << args.k <<
          "; batch_size: " << args.batch_size << 
          "; n_blocks: " << blocks_per_grid << "; shmem_size: " << shmem_size
        << " num_threads: " << num_threads);

  void* kernel_args[] = {&args.data, &args.num_elements, &args.top_elements,
                         &args.top_indices, &args.k};
  
  CU_BEGIN_TIMING(0)
  (void)cudaLaunchKernel(kernel, blocks_per_grid, num_threads, kernel_args,
                       shmem_size, 0);
  CU_END_TIMING("TopK N = %zu; K = %zu; batch_size: %zu", 
      args.num_elements, args.k, args.batch_size);

  CHK(cudaPeekAtLastError());
  (void)cudaDeviceSynchronize();                       
}

template < class NT >
void benchmark_topk(size_t batch_size, size_t N, size_t K, bool verify = true) 
{
  const size_t in_total = batch_size * N,
         out_total = batch_size * K;
  HVector< NT > values(in_total), top_elems(out_total);
  HVector< int32_t > indices(out_total);

  std::random_device rd;
  int seed = rd();  
  mersenne::init_genrand(seed);
  for(size_t i = 0; i < in_total; i++) {
    RandomBits(values[i]);
    //values[i] = i+1;
  }
  values.copyHToD();
  TypedTopK< NT >({values.devPtr, N, top_elems.devPtr, 
         (uint32_t *)indices.devPtr, K, batch_size});

  if(!verify) {
    return;
  }

  top_elems.copyDToH();
  indices.copyDToH();

  auto vptr = values.data();  // original CPU data (unsorted)
  auto gpu_iptr = indices.data();
  auto gpu_vptr = top_elems.data();

  std::vector< NT > truth_vals(K), gpu_vals(K);
  std::vector< int32_t > idxs(N);
  for(size_t i = 0; i < batch_size; i++, gpu_iptr += K, gpu_vptr += K, vptr += N) {
    
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [vptr](const auto& a, const auto& b) {
      return vptr[a] > vptr[b];
    });
    // save truth values before sorting idxs to keep it consistent with gpu_vptr
    for(size_t j = 0; j < K; j++) {
      truth_vals[j] = vptr[idxs[j]];
    }

    bool print_if_differs = true;
    int32_t eps = 0;
    std::sort(idxs.begin(), idxs.begin() + K);
    std::sort(gpu_iptr, gpu_iptr + K);
    // descending sort !! 
    std::sort(gpu_vptr, gpu_vptr + K, std::greater<NT>());

    checkme(gpu_iptr, idxs.data(), K, K, 1, eps, print_if_differs);
    //VLOG("------------------------------------------------------")
    checkme(gpu_vptr, truth_vals.data(), K, K, 1, (NT)1e-5, print_if_differs);
  }
    
}

int main() try 
{
  DeviceInit();

  benchmark_topk< uint32_t >(1, 1024*2, 16, false);
  return 0;

  //size_t batch_size, size_t N, size_t K
  for(size_t batch_size: {10, 20, 100, 200, 1000}) 
  {
    for(size_t N: {100, 200, 300, 999, 1050, 2000, 6333, 7889, 12312})
    //for(size_t N: {1024, 2048, 4096, 8192}) 
    {
      for(size_t K: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
      {  
        //benchmark_topk< float >(batch_size, N, K);
      }
    }
  }
}
catch(std::exception& ex) {
  VLOG("Exception: " << ex.what());
}
