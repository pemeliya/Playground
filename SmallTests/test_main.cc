
// hipcc -I.. -DCOMPILE_FOR_ROCM=1 -std=c++17 --offload-arch=gfx90a test_main.cc

#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include "common/example_utils.hpp"


__global__ void powerf32_kernel(const float *X, const float *Y, float *out, uint32_t n) {

  for(uint32_t idx = threadIdx.x; idx < n; idx += blockDim.x) {
    auto x = X[idx], y = Y[idx];
    out[idx] = powf(x, y);
  }
}

template <typename T>
std::pair<std::vector<T>, std::vector<T>> AllSignedPairs(
    const std::vector<T>& abs_vals) {
  std::vector<T> ys, xs;
  const size_t n = 4 * abs_vals.size() * abs_vals.size();
  ys.reserve(n);
  xs.reserve(n);
  for (auto abs_y : abs_vals) {
    for (auto y : {-abs_y, abs_y}) {
      for (auto abs_x : abs_vals) {
        for (auto x : {-abs_x, abs_x}) {
          ys.push_back(y);
          xs.push_back(x);
        }
      }
    }
  }
  return {xs, ys};
}

void runPowf32() 
{
  float xmax = std::numeric_limits< float >::max(),
        xeps = 1.0f + std::numeric_limits< float >::epsilon();
  
  auto [xs,ys] = AllSignedPairs< float >({xeps, xmax});
  HVector< float > X(std::move(xs)), Y(std::move(ys)), Z(X.size());

  X.copyHToD();
  Y.copyHToD();

  uint32_t nblocks = 1, nthreads = 256;
  powerf32_kernel<<<nblocks, nthreads>>>(X.devPtr, Y.devPtr, Z.devPtr, X.size());

  CHK(cudaPeekAtLastError());
  (void)cudaDeviceSynchronize();                       
  Z.copyDToH();

  VLOG(std::setprecision(8));
  for(int i = 0; i < Z.size(); i++) {
    auto z = std::pow(X[i], Y[i]);
    VLOG("pow(" << X[i] << ", " << Y[i] << ") = " << Z[i] << " truth: " << z);
  }
}

#if 0
23-11-23 17:20:00.917758: E external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:1708] failed to query device memory info: HIP_ERROR_InvalidValue
2023-11-23 17:20:00.917833: E external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:1708] failed to query device memory info: HIP_ERROR_InvalidValue
2023-11-23 17:20:00.917973: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1929] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 173 MB memory:  -> device: 0, name: AMD Instinct MI210, pci bus id: 0000:03:00.0
2023-11-23 17:20:01.188576: I external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:949] successfully created stream 0x5633c045cf10 for device 1 on thread
2023-11-23 17:20:01.201311: I external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:949] successfully created stream 0x5633c2ca3f80 for device 1 on thread
2023-11-23 17:20:01.213315: I external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:949] successfully created stream 0x5633c27917c0 for device 1 on thread
2023-11-23 17:20:01.225426: I external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:949] successfully created stream 0x5633c2c82f90 for device 1 on thread
2023-11-23 17:20:01.226038: I external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:949] successfully created stream 0x5633c2adb090 for device 1 on thread
2023-11-23 17:20:01.226659: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1929] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 173 MB memory:  -> device: 1, name: AMD Instinct MI210, pci bus id: 0000:83:00.0
2023-11-23 17:20:01.477425: I external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:949] successfully created stream 0x5633c2c95ff0 for device 2 on thread
2023-11-23 17:20:01.485839: I external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:949] successfully created stream 0x5633c262f810 for device 2 on thread
2023-11-23 17:20:01.495361: I external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:949] successfully created stream 0x5633c2b0c6e0 for device 2 on thread
2023-11-23 17:20:01.504350: I external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:949] successfully created stream 0x5633c27a2650 for device 2 on thread
2023-11-23 17:20:01.504855: I external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:949] successfully created stream 0x5633c2ce2060 for device 2 on thread
Running test!!!!!!!!!!!!!!!!!
2023-11-23 17:20:01.566662: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:388] MLIR V1 optimization pass is not enabled
2023-11-23 17:20:01.652796: I external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:1304] ----------------------- CPU fsum: -1963.84
2023-11-23 17:20:01.654297: I external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:1319] successfully enqueued async memcpy h2d 0x7fec90200000 -> 0x7fe851000000 of 4000000 bytes on stream 0x5633c27917c0 for device: 1
2023-11-23 17:20:01.731908: I external/local_xla/xla/service/gpu/runtime/xxdebug_kernel.cu.cc:42] stream: 0x5633c27917c0; reduce result -1963.84
2023-11-23 17:20:01.731929: I external/local_xla/xla/service/gpu/runtime/xxdebug_kernel.cu.cc:42] stream: 0x5633c27917c0; reduce result -4.80781
2023-11-23 17:20:01.731936: I external/local_xla/xla/service/gpu/runtime/xxdebug_kernel.cu.cc:42] stream: 0x5633c27917c0; reduce result 5.20957
2023-11-23 17:20:01.731942: I external/local_xla/xla/service/gpu/runtime/xxdebug_kernel.cu.cc:42] stream: 0x5633c27917c0; reduce result 8.42432e+09
2023-11-23 17:20:01.732639: I ./tensorflow/core/kernels/reduction_ops_common.h:198] Running simple reduce!
2023-11-23 17:20:01.732737: I ./tensorflow/core/kernels/reduction_gpu_kernels.cu.h:805] HACK----------------------- LaunchScalarReduction: 1000000 on stream: 0x5633c045cf10
2023-11-23 17:20:01.732865: I external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:1281] successfully enqueued async memcpy d2h of 4 bytes from 0x7fe8513d0e00 to 0x7fec90200300 on stream 0x5633c2c82f90
2023-11-23 17:20:01.732933: I external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:1200] successfully synchronized stream 0x5633c045cf10 on device 1
range: [-4.807811260223389 - 5.20956563949585] ----------- outbf16: 0 -- out_f32: -0.37254899740219116 -- np: -1963.8616943359375
INFO:tensorflow:time(__main__.ReduceTest.testReduceExtendType): 1.05s
I1123 17:20:01.734216 140672160917312 test_util.py:2574] time(__main__.ReduceTest.testReduceExtendType): 1.05s
INFO:tensorflow:Running testReduceExtendType in EAGER mode.
I1123 17:20:01.734754 140672160917312 test_util.py:1540] Running testReduceExtendType in EAGER mode.
Running test!!!!!!!!!!!!!!!!!
2023-11-23 17:20:01.756793: E external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:1708] failed to query device memory info: HIP_ERROR_InvalidValue
2023-11-23 17:20:01.756830: E external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:1708] failed to query device memory info: HIP_ERROR_InvalidValue
2023-11-23 17:20:01.756929: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1929] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 173 MB memory:  -> device: 0, name: AMD Instinct MI210, pci bus id: 0000:03:00.0
2023-11-23 17:20:01.756977: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1929] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 173 MB memory:  -> device: 1, name: AMD Instinct MI210, pci bus id: 0000:83:00.0
2023-11-23 17:20:01.776576: I external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:1304] ----------------------- CPU fsum: -1963.84
2023-11-23 17:20:01.777514: I external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:1319] successfully enqueued async memcpy h2d 0x5633c36ec780 -> 0x7fe851000000 of 4000000 bytes on stream 0x5633c27917c0 for device: 1
2023-11-23 17:20:01.853994: I external/local_xla/xla/service/gpu/runtime/xxdebug_kernel.cu.cc:42] stream: 0x5633c27917c0; reduce result -inf
2023-11-23 17:20:01.854009: I external/local_xla/xla/service/gpu/runtime/xxdebug_kernel.cu.cc:42] stream: 0x5633c27917c0; reduce result -2.91911e+38
2023-11-23 17:20:01.854016: I external/local_xla/xla/service/gpu/runtime/xxdebug_kernel.cu.cc:42] stream: 0x5633c27917c0; reduce result 2.73525e+38
2023-11-23 17:20:01.854022: I external/local_xla/xla/service/gpu/runtime/xxdebug_kernel.cu.cc:42] stream: 0x5633c27917c0; reduce result -2.42957e+17
2023-11-23 17:20:01.866144: I external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:1304] ----------------------- CPU fsum: -1963.84
2023-11-23 17:20:01.867049: I external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:1319] successfully enqueued async memcpy h2d 0x5633c3abd100 -> 0x7fe851000000 of 4000000 bytes on stream 0x5633c27917c0 for device: 1
2023-11-23 17:20:01.943297: I external/local_xla/xla/service/gpu/runtime/xxdebug_kernel.cu.cc:42] stream: 0x5633c27917c0; reduce result -1848.46
2023-11-23 17:20:01.943309: I external/local_xla/xla/service/gpu/runtime/xxdebug_kernel.cu.cc:42] stream: 0x5633c27917c0; reduce result -4.80781
2023-11-23 17:20:01.943316: I external/local_xla/xla/service/gpu/runtime/xxdebug_kernel.cu.cc:42] stream: 0x5633c27917c0; reduce result 5.20957
2023-11-23 17:20:01.943324: I external/local_xla/xla/service/gpu/runtime/xxdebug_kernel.cu.cc:42] stream: 0x5633c27917c0; reduce result 0.00592327
2023-11-23 17:20:01.949021: I external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:1304] ----------------------- CPU fsum: -1963.84
2023-11-23 17:20:01.949808: I external/local_xla/xla/stream_executor/rocm/rocm_driver.cc:1319] successfully enqueued async memcpy h2d 0x5633c62b14c0 -> 0x7fe851000000 of 4000000 bytes on stream 0x5633c27917c0 for device: 1
2023-11-23 17:20:02.026040: I external/local_xla/xla/service/gpu/runtime/xxdebug_kernel.cu.cc:42] stream: 0x5633c27917c0; reduce result -1963.84
2023-11-23 17:20:02.026064: I external/local_xla/xla/service/gpu/runtime/xxdebug_kernel.cu.cc:42] stream: 0x5633c27917c0; reduce result -4.80781
2023-11-23 17:20:02.026071: I external/local_xla/xla/service/gpu/runtime/xxdebug_kernel.cu.cc:42] stream: 0x5633c27917c0; reduce result 5.20957
2023-11-23 17:20:02.026077: I external/local_xla/xla/service/gpu/runtime/xxdebug_kernel.cu.cc:42] stream: 0x5633c27917c0; reduce result 8.42432e+09
2023-11-23 17:20:02.026999: I ./tensorflow/core/kernels/reduction_ops_common.h:198] Running simple reduce!
#endif

template <typename IN_T, typename T>
__global__ void ReduceSumKernel(IN_T in, int in_size, T *out) {

  int thid = threadIdx.x;
  if(thid == 0) {
    float zmin = std::numeric_limits< T >::max(),
      zmax = std::numeric_limits< T >::min(), zz = 0;
  union {
    T t;
    uint32_t u;
  } X, Y;
    Y.u = 0;
    for(int i = 0; i < in_size; i++, in++) {
      auto Z = *in;
      zz += Z;
      zmin = min(zmin, Z);
      zmax = max(zmax, Z);
      X.t = Z;
      Y.u += X.u;
    }
    out[0] = zz;
    out[1] = zmin;
    out[2] = zmax;
    out[3] = Y.t;
  }
}

template < class T >
void runMemcpyTest(size_t N) {

  GPUStream s[5];
  T *devSrc = nullptr;
  uint32_t bytes = 181403648;
  CHK(cudaMalloc((void**)&devSrc, bytes))

  std::vector< T > hosts[5];
  for(auto& H : hosts) {
    H.resize(N);
  }

  for(int i = 0; i < 1000; i++) {
  
  auto& hostD = hosts[i % 5];
  for(uint32_t i = 0; i < N; i++) {
    hostD[i] = i+1;
  }

  VLOG("memcpy " << hostD.data() << " -> " << devSrc);
  CHK(hipMemcpyHtoDAsync(
        devSrc, hostD.data(), N * sizeof(T), s[0].get()))
  //CHK(hipStreamSynchronize(s1.get()))


  HVector< T > zz(16);
  ReduceSumKernel<<<1, 128, 0, s[0].get()>>>(devSrc, N, zz.devPtr);
  //CHK(hipStreamSynchronize(s2.get()));    

  zz.copyDToH();
  for(int i = 0; i < 1; i++) {
     VLOG(i << ": reduce result " << zz[i]);
  }
  } // for

  CHK(cudaFree(devSrc))
}

int main() try 
{
   DeviceInit();
   runMemcpyTest<float>(1000000);
}
catch(std::exception& ex) {
  VLOG("Exception: " << ex.what());
}
