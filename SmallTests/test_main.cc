
// hipcc -I.. -DCOMPILE_FOR_ROCM=1 -std=c++17 --offload-arch=gfx90a test_main.cc

#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
//#include "common/common_utils.hpp"
#include <hip/hip_runtime.h>

#define VLOG(x) std::cerr << x

#define CHK(x) if(auto res = (x); res != hipSuccess) { \
  throw std::runtime_error(std::to_string(__LINE__) + " failed!"); \
}

int main() try 
{
  size_t bytes = 4*sizeof(float);
  void *ptr0, *ptr1;
  std::vector< float > vec1{41.0f,42.0f,43.0f,44.0f}, 
        vec0(4, 0);

  {
    CHK(hipSetDevice(0));
    CHK(hipDeviceEnablePeerAccess(1, 0));
    CHK(hipMalloc(&ptr0, bytes));
  }

  {
    CHK(hipSetDevice(1));
    CHK(hipDeviceEnablePeerAccess(0, 0));
    CHK(hipMalloc(&ptr1, bytes));
    CHK(hipMemcpy(ptr1, vec1.data(), bytes, hipMemcpyHostToDevice));
  }
 
  // copy from dev1 to dev0
  CHK(hipMemcpyPeer(ptr0, 0, ptr1, 1, bytes));
  
  CHK(hipSetDevice(1));  
  CHK(hipMemcpy(vec0.data(), ptr0, bytes, hipMemcpyDeviceToHost));

  for(auto v : vec0) {
    VLOG(0) << "v = " << v << '\n';
  }
}
catch(std::exception& ex) {
  VLOG(0) << "Exception: " << ex.what();
}
