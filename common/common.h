
#ifndef PLAYGROUND_COMMON_H
#define PLAYGROUND_COMMON_H 1

#include <stdint.h>
#include <iostream>
#include <limits>

#if COMPILE_FOR_ROCM
#include<hip/hip_runtime.h>
#include<hip/hip_cooperative_groups.h>
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMallocHost hipMallocHost
#define cudaFreeHost hipFreeHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaSuccess hipSuccess
#define cudaGetLastError hipGetLastError
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaSetDevice hipSetDevice
#define cudaMemGetInfo hipMemGetInfo
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaEventElapsedTime hipEventElapsedTime

// #include <hipcub/util_type.hpp>
// #include <hipcub/util_allocator.hpp>
// #include <hipcub/iterator/discard_output_iterator.hpp>

#else
#include <cuda_runtime_api.h>
#endif

#define CHK(x) \
   if(auto res = (x); res != cudaSuccess) { \
      throw std::runtime_error(#x " failed: " + std::string(cudaGetErrorName(res)) + " at line: " + std::to_string(__LINE__)); \
   }

#define VLOG(x) std::cerr << x << std::endl;

namespace std {

//#define __cpp_lib_int_pow2 aaa

constexpr uint32_t bit_floor(uint32_t n)
{
    constexpr uint32_t k = std::numeric_limits<uint32_t>::digits;
    uint32_t shift = k - 1u - __builtin_clz(n);
    return (1u << (shift & (k - 1u))) & n;
}

template <class _Tp>
constexpr _Tp bit_ceil(_Tp __t) noexcept {
  if (__t < 2)
    return 1;
  const unsigned __n = std::numeric_limits<_Tp>::digits - std::__countl_zero((_Tp)(__t - 1u));

  if constexpr (sizeof(_Tp) >= sizeof(unsigned))
    return _Tp{1} << __n;
  else {
    const unsigned __extra = std::numeric_limits<unsigned>::digits - std::numeric_limits<_Tp>::digits;
    const unsigned __ret_val = 1u << (__n + __extra);
    return (_Tp)(__ret_val >> __extra);
  }
}

} // namespace std


template < class NT >
struct HVector : std::vector< NT > {
   
   using Base = std::vector< NT >;

   HVector(std::initializer_list< NT > l) : Base(l) {
       CHK(cudaMalloc((void**)&devPtr, l.size()*sizeof(NT)))
   }
   HVector(size_t N) : Base(N, NT{}) {
       CHK(cudaMalloc((void**)&devPtr, N*sizeof(NT)))
   }
   void copyHToD() {
      CHK(cudaMemcpy(devPtr, this->data(), this->size()*sizeof(NT), cudaMemcpyHostToDevice))
   }
   void copyDToH() {
      CHK(cudaMemcpy(this->data(), devPtr, this->size()*sizeof(NT), cudaMemcpyDeviceToHost))
   }
   ~HVector() {
      if(devPtr) {
        cudaFree(devPtr);
      }
   }
   NT *devPtr = nullptr;
};

template< class NT >
struct MappedVector {

   MappedVector(size_t N_) : N(N_) {
       CHK(cudaMallocHost((void**)&devPtr, N*sizeof(NT)))
   }
   void copyHToD() {
   }
   void copyDToH() {
   }
   size_t size() const { return N; }
   NT& operator[](size_t i) {
      return devPtr[i];
   }
   ~MappedVector() {
      (void)cudaFreeHost(devPtr);
   }
   size_t N;
   NT *devPtr;
};

template <class T,
          std::enable_if_t< !std::is_floating_point_v<T>, bool> = true >
bool checkNaN(T) {
    return false;
}

template <class T,
     std::enable_if_t< std::is_floating_point_v<T>, bool> = true >
bool checkNaN(T a) {
    return std::isnan(a);
}

//! compares 2D arrays of data, \c width elements per row stored with \c stride (stride == width)
//! number of rows given by \c n_batches
//! \c print_when_differs :  indicates whether print elements only if they
//! differ (default)
//! \c print_max : maximal # of entries to print
template < bool Reverse = false, class NT >
bool checkme(const NT *checkit, const NT *truth, size_t width, size_t stride,
        size_t n_batches, const NT& eps, bool print_when_differs = true,
             size_t print_max = std::numeric_limits< size_t >::max()) {

    if(checkit == NULL)
        return false;

    bool res = true;
    size_t printed = 0;
    //std::cerr << "\nlegend: batch_id (element_in_batch)\n";

    int inc = Reverse ? -1 : 1;
    size_t jbeg = 0, jend = n_batches,
           ibeg = 0, iend = width;

    if(Reverse) {
        jbeg = n_batches - 1, jend = (size_t)-1;
        ibeg = width - 1, iend = jend;
    }

    for(size_t j = jbeg; j != jend; j += inc) {

        const NT *pcheckit = checkit + j * stride,
            *ptruth = truth + j * stride;

        for(size_t i = ibeg; i != iend; i += inc) {

            bool nan1 = checkNaN(ptruth[i]), nan2 = checkNaN(pcheckit[i]);
            if(nan1 && nan2)
                continue;

            NT diff = pcheckit[i] - ptruth[i];
            bool isDiff = std::abs(diff) > eps || nan1 || nan2;
            if(isDiff)
                res = false;

            if((isDiff || !print_when_differs) && printed < print_max) {
                NT check = pcheckit[i], truth = ptruth[i];

                printed++;
                std::cerr << j << '(' << i << ") (GPU, truth): " <<
                   check << " and " << truth << " ; diff: " << diff << (isDiff ? " DIFFERS\n" : "\n");
            }
        }
    }
    return res;
}

#endif // PLAYGROUND_COMMON_H
