#include <stdio.h>
#include <algorithm>
#include <iostream>

#include "gpu_prim.h"
#include "example_utils.hpp"

//! hipcc -std=c++17 -O3 benchmark.cc --offload-arch=gfx90a
//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

#define CHK(x) \
   if(auto res = (x); res != hipSuccess) { \
      throw std::runtime_error(#x " failed with code: " + std::to_string(res) + ": " + std::string(hipGetErrorString(res))); \
   }

#define OUTZ(x) std::cerr << x << std::endl   

gpuprim::CachingDeviceAllocator  g_allocator;  // Caching allocator for device memory

template < class NT >
struct HVector : std::vector< NT > {
   
   using Base = std::vector< NT >;

   HVector(std::initializer_list< NT > l) : Base(l) {
       CHK(g_allocator.DeviceAllocate((void**)&devPtr, l.size()*sizeof(NT)))
   }
   HVector(size_t N) : Base(N, NT{}) {
       CHK(g_allocator.DeviceAllocate((void**)&devPtr, N*sizeof(NT)))
   }
   void copyHToD() {
      CHK(hipMemcpy(devPtr, this->data(), this->size()*sizeof(NT), hipMemcpyHostToDevice))
   }
   void copyDToH() {
      CHK(hipMemcpy(this->data(), devPtr, this->size()*sizeof(NT), hipMemcpyDeviceToHost))
   }
   ~HVector() {
      if(devPtr) {
        (void)g_allocator.DeviceFree(devPtr);
      }
   }
   NT *devPtr = nullptr;
};

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

int main(int argc, char** argv) try
{
    int num_items = argc > 1 ? atoi(argv[1]) : 1000000;

    // Initialize device
    CHK(DeviceInit())

    int devID = 0;
    CHK(hipGetDevice(&devID));
    size_t freeX = 0, total = 0;
    CHK(hipMemGetInfo(&freeX, &total));

    //benchmark_sort<float>();
    benchmark_sort<int16_t>("int16_t", num_items);
    benchmark_sort<int32_t>("int32_t", num_items);
    benchmark_sort<uint16_t, __half>("halffloat", num_items);
    benchmark_sort<uint16_t, hip_bfloat16>("bfloat16", num_items);
    benchmark_sort<float>("float", num_items);
    benchmark_sort<double>("double", num_items);
}
catch(std::exception& ex) {
    OUTZ("Exception: " << ex.what());
}