
#include <algorithm>
#include <stdexcept>
#include <bit>
#include <numeric>
#include <iostream>
#include <random>
#include "topk_kernel.h"
#include "common/common_utils.hpp"

template < class NT >
void benchmark_topk(TopKType type, 
            size_t batch_size, size_t N, size_t K, bool verify = true) 
{
  const size_t in_total = batch_size * N, out_total = batch_size * K;
  HVector< NT > values(in_total), top_elems(out_total);
  HVector< int32_t > indices(out_total);

  std::random_device rd;
  int seed = 1112;//rd();   // ensure determinism
  mersenne::init_genrand(seed);
  for(size_t i = 0; i < in_total; i++) {
    RandomBits(values[i]);     //values[i] = i+1;
  }
  values.copyHToD();
  TopkArgs args =  {
      .data = values.devPtr,
      .top_elements = top_elems.devPtr,
      .top_indices = (uint32_t *)indices.devPtr,
      .type = type,
      .num_elems = N,
      .k = K,
      .batch_size = batch_size};
  
  //RunPerWarpTopK(args);
  RunBitonicTopK(args);
  if(!verify) return;

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
  benchmark_topk< uint32_t >(TopKType::U32, /*batch_size*/1, 1024*2, 16, true);
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
  VLOG(0) << "Exception: " << ex.what();
}
