#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <random>

#include "common/gpu_prim.h"
#include "common/common_utils.hpp"

//! hipcc -std=c++17 -O3 benchmark.cc --offload-arch=gfx90a
//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

#define OUTZ(x) std::cerr << x << std::endl   

gpuprim::CachingDeviceAllocator  g_allocator;  // Caching allocator for device memory

template <typename KeyT>
auto CubSortKeys(void* d_temp_storage, size_t& temp_bytes,
                         const void* d_keys_in, void* d_keys_out,
                         size_t num_items, bool descending) {
  return descending
          ? gpuprim::DeviceRadixSort::SortKeysDescending<KeyT>(
                d_temp_storage, temp_bytes, static_cast<const KeyT*>(d_keys_in),
                static_cast<KeyT*>(d_keys_out), num_items)
          : gpuprim::DeviceRadixSort::SortKeys<KeyT>(
                d_temp_storage, temp_bytes, static_cast<const KeyT*>(d_keys_in),
                static_cast<KeyT*>(d_keys_out), num_items);
}

struct SumOp
{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

template < class KeyT, class DevKeyT = KeyT >
void benchmark_sort(const char *name, size_t num_items) {

    static_assert(sizeof(KeyT) == sizeof(DevKeyT), "Must be equal in size!");

    HVector<KeyT> keys_in(num_items), keys_out(num_items);

    for(size_t i = 0; i < keys_in.size(); i++) {
        RandomBits(keys_in[i]);
    }
    // std::sort(items.begin(), items.end());
    // for(size_t i = 0; i < keys_in.size(); i++) {
    //       OUTZ(i << " = " << keys_in[i]);
    //  }
    
    size_t temp_bytes  = 0, num_iters = 5;
    CHK(CubSortKeys<DevKeyT>(nullptr, temp_bytes, keys_in.devPtr, keys_out.devPtr, num_items, false));
        
    HVector< uint8_t > temp(temp_bytes);

    GpuTimer timer;
    for(int j = 0; j < 2; j++) {
        bool descending = j > 0;
        for(size_t i = 0; i < num_iters; i++) {
            if(i == 1) timer.Start(); // skip first iteration for warm-up
            keys_in.copyHToD();
            CHK(CubSortKeys<DevKeyT>(temp.devPtr, temp_bytes, keys_in.devPtr, keys_out.devPtr, num_items, descending));
            keys_out.copyDToH();
        }
        CHK(hipDeviceSynchronize())
        timer.Stop();
        OUTZ(name << " sorting " << num_items << " items " << 
                (descending ? "desc: " : "asc: ") << timer.ElapsedMillis()/(num_iters-1) << " ms");
    }
    // for(size_t i = 0; i < keys_out.size(); i++) {
    //     OUTZ(i << " = " << keys_out[i]);
    // }
}


template < class KeyT >
void benchmark_reduce(const char *name, const std::vector< float >& input) {

    auto num_items = input.size();
    //VLOG(0) << name << " reducting of " << num_items  << " elements";
    HVector<KeyT> keys_in(num_items), keys_out(16);

     //std::mt19937 e2(111);
    //std::knuth_b e2(rd());
    //std::default_random_engine e2(rd()) ;
    ///std::uniform_real_distribution<> dist(-5, 5);

    KeyT reduceF{};
    for(size_t i = 0; i < keys_in.size(); i++) {
        //RandomBits(keys_in[i]);
        keys_in[i] = (KeyT)input[i];//dist(e2);
        reduceF += keys_in[i];
    }

    keys_in.copyHToD();

   size_t temp_bytes = 0;
    auto reduce = [&](void* temp_storage_ptr) {
    
    CHK(gpuprim::DeviceReduce::Reduce(temp_storage_ptr, temp_bytes, keys_in.devPtr,
                                      keys_out.devPtr, num_items, SumOp{}, KeyT{}, 0));

  };

  reduce(nullptr);  // Get required amount of temp storage.
  HVector< uint8_t > temp(temp_bytes);

  reduce(temp.devPtr);

  keys_out.copyDToH();

  VLOG(0) << "gpu: " << keys_out[0] << " truth: " << reduceF;

}

int main(int argc, char** argv) try
{
    int num_items = argc > 1 ? atoi(argv[1]) : 1000000;

    DeviceInit();

    std::ifstream ifs("input.csv");
    std::vector< float > input;
    input.reserve(100000);
    while(ifs) {
        float a;
        ifs >> a;
        input.push_back(a);
    }

    for(int i = 0; i < 100; i++) {
    benchmark_reduce< float >("float", input);
    benchmark_reduce< hip_bfloat16 >("bfloat16", input);
    }

    //benchmark_sort<float>();
    // benchmark_sort<int16_t>("int16_t", num_items);
    // benchmark_sort<int32_t>("int32_t", num_items);
    // benchmark_sort<uint16_t, __half>("halffloat", num_items);
    // benchmark_sort<uint16_t, hip_bfloat16>("bfloat16", num_items);
    // benchmark_sort<float>("float", num_items);
    // benchmark_sort<double>("double", num_items);
}
catch(std::exception& ex) {
    OUTZ("Exception: " << ex.what());
}