
#include <algorithm>
#include <stdexcept>
#include <bit>
#include <numeric>
#include "topk_kernel.h"
#include "example_utils.hpp"

size_t NumThreads(size_t n, size_t k, size_t batch_size) {
  // Estimate number of threads per block that can run concurrently given the
  // register footprint.
  size_t simultaneous_threads_per_block = 512 * (16 / k);
  size_t threads_per_block =
      std::min(simultaneous_threads_per_block, kTopKMaxThreadsPerBlock);
  // Minimum amount of data that each thread needs to receive for the algorithm.
  size_t min_slice = std::bit_floor(n / std::bit_ceil(k));
  //VLOG("threads_per_block, min_slice: " << threads_per_block << ',' << min_slice);
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
  int num_threads = NumThreads(args.num_elements, args.k, args.batch_size);
  if (num_threads == 0) {
    throw std::runtime_error(
        "Invalid kernel parameters. This is likely a bug in the "
        "TopkSpecializer.");
  }

  void* kernel = GetKernel<T>(args.num_elements, args.k);

  int blocks_per_grid = args.batch_size;
  constexpr size_t max_kv_size = sizeof(uint64_t);
  // Allocate shmem assuming we have a full reduction.
  int shmem_size = std::bit_ceil(args.k) * max_kv_size * 32;

  int slice_size = (args.num_elements + num_threads-1) / num_threads;
  // VLOG("Testing N = " << args.num_elements << "; K = " << args.k <<
  //         "; batch_size: " << args.batch_size << 
  //         "; n_blocks: " << blocks_per_grid << "; shmem_size: " << shmem_size
  //       << " num_threads: " << num_threads << " slice_size: " << slice_size);

  void* kernel_args[] = {&args.data, &args.num_elements, &args.top_elements,
                         &args.top_indices, &args.k};
  
  CU_BEGIN_TIMING(5)
  cudaLaunchKernel(kernel, blocks_per_grid, num_threads, kernel_args,
                       shmem_size, 0);
  CU_END_TIMING("TopK N = %zu; K = %zu; batch_size: %zu", 
      args.num_elements, args.k, args.batch_size);

  CHK(cudaPeekAtLastError());
  cudaDeviceSynchronize();                       
}

template < class NT >
void benchmark_topk(size_t batch_size, size_t N, size_t K, bool verify = true) 
{
  const size_t in_total = batch_size * N,
         out_total = batch_size * K;
  MappedVector< NT > values(in_total + out_total);  // K outputs
  MappedVector< int32_t > indices(out_total);
  auto top_elems = values.devPtr + in_total;

  int seed = 12345678;  
  mersenne::init_genrand(seed);
  for(size_t i = 0; i < in_total; i++) {
    RandomBits(values[i]);
  }
  TypedTopK< NT >({values.devPtr, N, top_elems, 
         (uint32_t *)indices.devPtr, K, batch_size});

  if(!verify) {
    return;
  }

  auto vptr = values.devPtr;
  auto iptr = indices.devPtr;
  for(size_t i = 0; i < batch_size; i++, iptr += K, vptr += N) {
    std::vector< int32_t > idxs(N);
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [vptr](const auto& a, const auto& b) {
      return vptr[a] > vptr[b];
    });

    bool print_if_differs = true;
    int32_t eps = 0;
    checkme(iptr, idxs.data(), K, K, 1, eps, print_if_differs);
  }
    
}

int main() try {

  //size_t batch_size, size_t N, size_t K
  for(size_t batch_size: {1, 10, 20, 100, 200, 1000}) {
    for(size_t N: {1024, 2048, 4096, 8192}) {
      //for(size_t K: {1, 2, 3, 4, 6, 8, 12, 16}) {
      for(size_t K: {16}) {  
        benchmark_topk< float >(batch_size, N, K);
        break;
      }
    }
  }

}
catch(std::exception& ex) {
  VLOG("Exception: " << ex.what());
}
