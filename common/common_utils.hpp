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

#ifndef COMMON_UTILS_HPP
#define COMMON_UTILS_HPP 1

#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <memory.h>
#include "common/common.h"
#include "common/mersenne.h"


void DeviceInit(int dev = 0);

class GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;
public:
    GpuTimer();
    ~GpuTimer();
    void Start();
    void Stop();
    float ElapsedMillis();
};

template < class NT >
struct HVector : std::vector< NT > {
   
   using Base = std::vector< NT >;

//    HVector(Base&& b) noexcept : Base(std::move(b)) {
//        CHK(cudaMalloc((void**)&devPtr, Base::size()*sizeof(NT)))
//    }

   HVector() = default;

   HVector(const HVector&) = delete;
   HVector& operator=(const HVector&) = delete;

   HVector(HVector&& rhs) noexcept : Base(std::move(rhs)) {
     devPtr = rhs.devPtr;
     rhs.devPtr = nullptr;
   }

   HVector& operator=(HVector&& rhs) noexcept {
     rhs.swap(*this);
     return *this;
   }

   void swap(HVector& lhs) noexcept {
     std::swap(devPtr, lhs.devPtr);
     Base::swap(lhs);
   }

   HVector(std::initializer_list< NT > l) : Base(l) {
       CHK(cudaMalloc((void**)&devPtr, l.size()*sizeof(NT)))
   }
   HVector(size_t N, bool allocHost = true) {
      if (allocHost) {
        Base::resize(N, NT{});
      }
      CHK(cudaMalloc((void**)&devPtr, N*sizeof(NT)))
   }
   void copyHToD() {
      //if (Base::empty()) throw std::runtime_error("copyHToD empty!");
      CHK(cudaMemcpy(devPtr, this->data(), this->size()*sizeof(NT), cudaMemcpyHostToDevice))
   }
   void copyDToH() {
      //if (Base::empty()) throw std::runtime_error("copyDToH empty!");
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

  explicit GPUStream(int priority = 0);
  cudaStream_t get() const {
    return handle_;
  }
  ~GPUStream();
private:
  cudaStream_t handle_;
};

#define CPU_BEGIN_TIMING(ID) \
        auto z1_##ID = std::chrono::high_resolution_clock::now()

#define CPU_END_TIMING(ID, num_runs, fmt, ...)                                                    \
        auto z2_##ID = std::chrono::high_resolution_clock::now();              \
        std::chrono::duration<double, std::milli> ms_##ID = z2_##ID - z1_##ID; \
        fprintf(stderr, "%s: " fmt " elapsed: %f msec\n", #ID, ##__VA_ARGS__, ms_##ID.count() / num_runs)

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
    fprintf(stderr, fmt "; time elapsed: %.3f us\n", ##__VA_ARGS__, ms*1000.f); \
    }

//! compares 2D arrays of data, \c width elements per row stored with \c stride (stride == width)
//! number of rows given by \c n_batches
//! \c print_when_differs :  indicates whether print elements only if they differ (default)
//! \c print_max : maximal # of entries to print
template < bool Reverse = false, class NT >
bool checkme(const std::string& msg, const NT *checkit, const NT *truth, size_t width, size_t stride,
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

    using FT = double;
    for(size_t j = jbeg; j != jend; j += inc) {

        const NT *pcheckit = checkit + j * stride,
            *ptruth = truth + j * stride;

        for(size_t i = ibeg; i != iend; i += inc) {
            
            auto check = (FT)pcheckit[i], truth = (FT)ptruth[i],
                 diff = check - truth;
            bool isDiff = false;
            if constexpr(std::is_floating_point_v< FT >) {
                bool nan1 = std::isnan(truth), nan2 = std::isnan(check);
                if(nan1 && nan2)
                  continue;
                isDiff = std::abs(diff) > FT(eps) || nan1 || nan2;
            } else {
                isDiff = std::abs(std::make_signed_t<NT>(diff)) > eps;
            }
            if(isDiff)
                res = false;

            if((isDiff || !print_when_differs) && printed < print_max) {
                printed++;
                std::cerr << msg << ": " << j << '(' << i << ") (GPU, truth): " <<
                   check << " and " << truth << " ; diff: " << diff 
                   << (isDiff ? " DIFFERS\n" : "\n");
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

#endif // COMMON_UTILS_HPP
