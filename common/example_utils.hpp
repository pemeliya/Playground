/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2021-2023, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef EXAMPLES_EXAMPLE_UTILS_HPP
#define EXAMPLES_EXAMPLE_UTILS_HPP

#include <vector>
#include <cmath>
#include <chrono>
#include <memory.h>
#include "common/common.h"
#include "common/mersenne.h"


inline void DeviceInit(int dev = 0)
{
    int deviceCount;
    CHK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        throw std::runtime_error("No devices supporting CUDA/HIP");
    }

    CHK(cudaSetDevice(dev));

    std::size_t device_free_physmem = 0, device_total_physmem = 0;
    CHK(cudaMemGetInfo(&device_free_physmem, &device_total_physmem)); // this fails on rocm-5.7.0

    cudaDeviceProp deviceProp;

    CHK(cudaGetDeviceProperties(&deviceProp, dev));
    if (deviceProp.major < 1) {
        throw std::runtime_error("Device does not support CUDA/HIP");
    }

    auto device_giga_bandwidth = float(deviceProp.memoryBusWidth) * deviceProp.memoryClockRate * 2 / 8 / 1000 / 1000;
    {
        //printf("%llu --- %llu\n", device_free_physmem, device_total_physmem);
        printf("Using device %d: %s ( SM%d, %d SMs, "
                        "%lld free / %lld total MB physmem, "
                        "%.3f GB/s @ %d kHz mem clock, ECC %s)\n",
                    dev,
                    deviceProp.name,
                    deviceProp.major * 100 + deviceProp.minor * 10,
                    deviceProp.multiProcessorCount,
                    (unsigned long long) device_free_physmem / 1024 / 1024,
                    (unsigned long long) device_total_physmem / 1024 / 1024,
                    device_giga_bandwidth,
                    deviceProp.memoryClockRate,
                    (deviceProp.ECCEnabled) ? "on" : "off");
        fflush(stdout);
    }
}

class GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;
public:
    GpuTimer()
    {
        (void)cudaEventCreate(&start);
        (void)cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        (void)cudaEventDestroy(start);
        (void)cudaEventDestroy(stop);
    }

    void Start()
    {
        (void)cudaEventRecord(start, 0);
    }

    void Stop()
    {
        (void)cudaEventRecord(stop, 0);
    }

    float ElapsedMillis()
    {
        float elapsed;
        (void)cudaEventSynchronize(stop);
        (void)cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

template < class NT >
struct HVector : std::vector< NT > {
   
   using Base = std::vector< NT >;

   HVector(Base&& b) : Base(std::move(b)) {
       CHK(cudaMalloc((void**)&devPtr, Base::size()*sizeof(NT)))
   }

   HVector(const HVector&) = delete;
   HVector& operator=(const HVector&) = delete;

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
        (void)cudaFree(devPtr);
      }
   }
   NT *devPtr = nullptr;
};

template< class NT >
struct MappedVector {

   MappedVector(size_t N_, int flags = 0) : N(N_) {
       CHK(cudaHostAlloc((void**)&devPtr, N*sizeof(NT), flags))
   }
   void copyHToD() {
   }
   void copyDToH() {
   }
   size_t size() const { return N; }
   NT *data() {
    return devPtr;
   }
   const NT *data() const {
    return devPtr;
   }
   NT& operator[](size_t i) {
      return devPtr[i];
   }
   const NT& operator[](size_t i) const {
      return devPtr[i];
   }
   ~MappedVector() {
      (void)cudaFreeHost(devPtr);
   }
   size_t N;
   NT *devPtr;
};

struct GPUStream {

  explicit GPUStream(int priority = 0) 
  {
    if (priority == 0) {
      CHK(cudaStreamCreateWithFlags(&handle_, cudaStreamDefault)) 
    } else {
      CHK(cudaStreamCreateWithPriority(&handle_, cudaStreamDefault, priority))
    }
  }
  cudaStream_t get() const {
    return handle_;
  }
  ~GPUStream() {
    (void)cudaStreamDestroy(handle_);
  }
private:
  cudaStream_t handle_;
};

#define CPU_BEGIN_TIMING(ID) \
        auto z1_##ID = std::chrono::high_resolution_clock::now()

#define CPU_END_TIMING(ID, fmt, ...)                                                    \
        auto z2_##ID = std::chrono::high_resolution_clock::now();              \
        std::chrono::duration<double, std::milli> ms_##ID = z2_##ID - z1_##ID; \
        fprintf(stderr, "%s: " fmt " elapsed: %f msec\n", #ID, ##__VA_ARGS__, ms_##ID.count())

#define CU_BEGIN_TIMING(N_ITERS) { \
    (void)cudaDeviceSynchronize();       \
    GpuTimer timer;                 \
    uint32_t nIters = N_ITERS;       \
    for(unsigned i = 0; i < nIters + 1; i++) {

#define CU_END_TIMING(fmt, ...)           \
        if(i == 0) {                    \
            (void)cudaDeviceSynchronize();    \
            timer.Start();              \
        }                               \
    }                                   \
    timer.Stop();                       \
    float ms = timer.ElapsedMillis();                  \
    if(nIters > 0) ms /= nIters;                        \
    fprintf(stderr, fmt "; time elapsed: %.3f ms\n", ##__VA_ARGS__, ms); \
    }

//! compares 2D arrays of data, \c width elements per row stored with \c stride (stride == width)
//! number of rows given by \c n_batches
//! \c print_when_differs :  indicates whether print elements only if they differ (default)
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
            
            NT diff = pcheckit[i] - ptruth[i];
            bool isDiff = false;
            if constexpr(std::is_floating_point_v< NT >) {
                bool nan1 = std::isnan(ptruth[i]), nan2 = std::isnan(pcheckit[i]);
                if(nan1 && nan2)
                  continue;
                isDiff = std::abs(diff) > eps || nan1 || nan2;
            } else {
                isDiff = std::abs(std::make_signed_t<NT>(diff)) > eps;
            }
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

template <typename K>
void RandomBits(
    K &key,
    int entropy_reduction = 0,
    int begin_bit = 0,
    int end_bit = sizeof(K) * 8)
{
    const int NUM_BYTES = sizeof(K);
    const int WORD_BYTES = sizeof(unsigned int);
    const int NUM_WORDS = (NUM_BYTES + WORD_BYTES - 1) / WORD_BYTES;

    unsigned int word_buff[NUM_WORDS];

    if (entropy_reduction == -1)
    {   
        key = {};
        return;
    }

    if (end_bit < 0)
        end_bit = sizeof(K) * 8;

    while (true)
    {
        // Generate random word_buff
        for (int j = 0; j < NUM_WORDS; j++)
        {
            int current_bit = j * WORD_BYTES * 8;

            unsigned int word = 0xffffffff;
            word &= 0xffffffff << std::max(0, begin_bit - current_bit);
            word &= 0xffffffff >> std::max(0, (current_bit + (WORD_BYTES * 8)) - end_bit);

            for (int i = 0; i <= entropy_reduction; i++)
            {
                // Grab some of the higher bits from rand (better entropy, supposedly)
                word &= mersenne::genrand_int32();
            }

            word_buff[j] = word;
        }
        memcpy(&key, word_buff, sizeof(K));

        K copy = key;
        if constexpr(std::is_floating_point<K>::value) {
#ifndef _WIN32
            if(!std::isnan(copy))
#else
            if(!std::isnan(static_cast<double>(copy)))
#endif
                break; // avoids NaNs when generating random floating point numbers
        } else 
            break;
    }
}

#endif
