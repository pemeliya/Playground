
#ifndef PLAYGROUND_COMMON_H
#define PLAYGROUND_COMMON_H 1

#include <stdint.h>
#include <stdarg.h>
#include <limits>
#include <vector>
#include <stdexcept>

#if COMPILE_FOR_ROCM
#include<hip/hip_runtime.h>
#include<hip/hip_cooperative_groups.h>
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t

#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaHostAlloc hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
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
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#define cudaStreamDestroy hipStreamDestroy
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaLaunchKernel hipLaunchKernel
#define cudaStreamDefault hipStreamDefault
#define FORCEINLINE inline
// #include <hipcub/util_type.hpp>
// #include <hipcub/util_allocator.hpp>
// #include <hipcub/iterator/discard_output_iterator.hpp>

#else
#include <cuda_runtime.h>
#define FORCEINLINE __forceinline__
#endif

#define VLOG(x) std::cerr << x << std::endl;

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

template < uint32_t SZ >
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

// using caching allocator ???
// gpuprim::CachingDeviceAllocator  g_allocator;  // Caching allocator for device memory

#endif // PLAYGROUND_COMMON_H
