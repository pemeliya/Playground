
// hipcc -I.. -DCOMPILE_FOR_ROCM=1 -std=c++17 --offload-arch=gfx90a test_main.cc

#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
//#include "common/example_utils.hpp"
#include<hip/hip_runtime.h>

#define VLOG(x) std::cerr << x

__global__ void AtomicAddKernel(float *ptr, float *u)
{
  int thid = threadIdx.x;
  atomicAdd(ptr + thid, u[0]);
  __threadfence_system();
}


int main() try 
{
   //DeviceInit();
   AtomicAddKernel<<<1,256>>>((float *)&main, nullptr);
}
catch(std::exception& ex) {
  VLOG("Exception: " << ex.what());
}
