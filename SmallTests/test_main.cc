
// hipcc -I.. -DCOMPILE_FOR_ROCM=1 -std=c++17 --offload-arch=gfx90a test_main.cc

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thread>

#include "common/common_utils.hpp"

using namespace std::chrono_literals;

void threadFunc1(GPUStream *stream) {
  VLOG(0) << "Beginning stream capture: ";
  CHK(cudaStreamBeginCapture(stream->get(), cudaStreamCaptureModeThreadLocal));
  std::this_thread::sleep_for(1s);
  cudaGraph_t graph;
  VLOG(0) << "End stream capture";
  CHK(cudaStreamEndCapture(stream->get(), &graph));
}

void threadFunc2() {
  VLOG(0) << "Calling cudaMalloc!";
  void *ptr = nullptr;
  CHK(cudaMalloc(&ptr, 256));
  VLOG(0) << "Calling cudaMemset!";
  CHK(cudaMemset(ptr, 0, 256));
  VLOG(0) << "Calling cudaFree ptr: " << ptr;
  CHK(cudaFree(ptr));
}

int main() try 
{
  DeviceInit();
  GPUStream s1(0);
  {
  std::jthread t1(threadFunc1, &s1);
  std::this_thread::sleep_for(100ms);
  std::jthread t2(threadFunc2);
  }
  VLOG(0) << "Graceful exit!";
  return 0;
}
catch(std::exception& ex) {
  VLOG(0) << "Exception: " << ex.what();
  return 1;
}
