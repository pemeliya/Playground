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
#include <memory.h>
#include "common.h"
#include "mersenne.h"


inline void DeviceInit(int dev = 0)
{
    int deviceCount;
    CHK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        throw std::runtime_error("No devices supporting CUDA/HIP");
    }

    CHK(cudaSetDevice(dev));

    std::size_t device_free_physmem = 0, device_total_physmem = 0;
    CHK(cudaMemGetInfo(&device_free_physmem, &device_total_physmem));

    cudaDeviceProp deviceProp;

    CHK(cudaGetDeviceProperties(&deviceProp, dev));
    if (deviceProp.major < 1) {
        throw std::runtime_error("Device does not support CUDA/HIP");
    }

    auto device_giga_bandwidth = float(deviceProp.memoryBusWidth) * deviceProp.memoryClockRate * 2 / 8 / 1000 / 1000;
    {
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

#define CU_BEGIN_TIMING(N_ITERS) { \
    cudaDeviceSynchronize();       \
    GpuTimer timer;                 \
    uint32_t nIters = N_ITERS;       \
    for(unsigned i = 0; i < nIters + 1; i++) {

#define CU_END_TIMING(fmt, ...)           \
        if(i == 0) {                    \
            cudaDeviceSynchronize();    \
            timer.Start();              \
        }                               \
    }                                   \
    timer.Stop();                       \
    float ms = timer.ElapsedMillis();                  \
    if(nIters > 0) ms /= nIters;                        \
    fprintf(stderr, fmt "; time elapsed: %.3f ms\n", ##__VA_ARGS__, ms); \
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
