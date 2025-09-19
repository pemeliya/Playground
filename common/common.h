
#ifndef PLAYGROUND_COMMON_H
#define PLAYGROUND_COMMON_H 1

#include <stdint.h>
#include <stdarg.h>
#include <limits>
#include <vector>
#include <stdexcept>
#include <random>
#include <sstream>

#if COMPILE_FOR_ROCM
#include<hip/hip_runtime.h>
#include<hip/hip_cooperative_groups.h>
#include <rccl/rccl.h>
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t

#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer 
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyPeer hipMemcpyPeer
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaHostAlloc hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaSuccess hipSuccess
#define cudaGetLastError hipGetLastError
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaSetDevice hipSetDevice
#define cudaMemGetInfo hipMemGetInfo
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamBeginCaptureToGraph hipStreamBeginCaptureToGraph
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamDefault hipStreamDefault

#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaGraph_t hipGraph_t
#define cudaGraphExec_t hipGraphExec_t
#define cudaGraphCreate hipGraphCreate
#define cudaGraphInstantiate hipGraphInstantiate
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphDestroy hipGraphDestroy

#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaLaunchKernel hipLaunchKernel
#define cudaOccupancyAvailableDynamicSMemPerBlock hipOccupancyAvailableDynamicSMemPerBlock
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize

#define FORCEINLINE inline
// #include <hipcub/util_type.hpp>
// #include <hipcub/util_allocator.hpp>
// #include <hipcub/iterator/discard_output_iterator.hpp>

#else
#include <cuda_runtime.h>
#define FORCEINLINE __forceinline__
#endif

struct XLogMessage : public std::basic_ostringstream<char> {
  XLogMessage(const char* fname, int line, int severity);
  ~XLogMessage();
private:
  const char *fname_;
  int line_;
};

#define VLOG(severity) XLogMessage(__FILE__, __LINE__, severity)

#define PRINTZ(fmt, ...) fprintf(stderr, fmt"\n", ##__VA_ARGS__)
#define BUGTRACE VLOG(0) << std::this_thread::get_id() << " OK";

#define CHK(x) if(auto res = (x); res != cudaSuccess) { \
  ThrowError<256>(#x " failed with: '%s'(%d) at %s:%d\n", cudaGetErrorString(res),  \
                res, __FILE__, __LINE__); \
}

namespace std {

//#define __cpp_lib_int_pow2 aaa

constexpr uint32_t bit_floor(uint32_t n)
{
    constexpr uint32_t k = std::numeric_limits<uint32_t>::digits;
    uint32_t shift = k - 1u - __builtin_clz(n);
    return (1u << (shift & (k - 1u))) & n;
}

constexpr uint32_t bit_ceil(uint32_t n) {
  if (n < 2)
    return 1;
  const unsigned __n = std::numeric_limits<uint32_t>::digits - __builtin_clz(n - 1u);
  return 1u << __n;
}

} // namespace std

template < uint32_t SZ = 256 >
[[noreturn]] void ThrowError(const char *fmt, ...) {
  static char buf[SZ];
  va_list args;
  va_start (args, fmt);
  vsnprintf(buf, SZ-1, fmt, args);
  va_end (args);
  throw std::runtime_error(buf);
}

template < class T >
void initVec(T *ptr, std::initializer_list< double > l) 
{
  for(const auto& elem : l) {
    *ptr++ = static_cast< T >(elem);
  }
}

template < class T >
void initRange(T *ptr, double start, double step, size_t n) 
{
  for(size_t i = 0; i < n; i++) {
    *ptr++ = static_cast< T >(start + i*step);
  }
}

template < class T >
void initRandomFloat(T *ptr, double dmin, double dmax, size_t n, size_t seed) {
   std::mt19937 gen(seed); 
   std::uniform_real_distribution<> dis(dmin, dmax);
   for(size_t i = 0; i < n; i++) {
    *ptr++ = static_cast< T >(dis(gen));
  }
}

__device__ FORCEINLINE uint32_t gpuLaneId() {
  uint32_t lane_id;
#if !COMPILE_FOR_ROCM
#if 0 // __clang__
  return __nvvm_read_ptx_sreg_laneid();
#else   // __clang__
  asm("mov.u32 %0, %%laneid;" : "=r"(lane_id));
#endif  // __clang__
#else
  lane_id = __lane_id();
#endif
  return lane_id;
}

// using caching allocator ???
// gpuprim::CachingDeviceAllocator  g_allocator;  // Caching allocator for device memory

#endif // PLAYGROUND_COMMON_H
