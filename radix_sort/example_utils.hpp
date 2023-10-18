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
#include <sstream>
#include <iostream>

#include <hipcub/util_type.hpp>
#include <hipcub/util_allocator.hpp>
#include <hipcub/iterator/discard_output_iterator.hpp>

#include "mersenne.h"

#define AssertEquals(a, b) if ((a) != (b)) { std::cerr << "\n(" << __FILE__ << ": " << __LINE__ << ")\n"; exit(1);}

template <typename T>
T CoutCast(T val) { return val; }

int CoutCast(char val) { return val; }

int CoutCast(unsigned char val) { return val; }

int CoutCast(signed char val) { return val; }

  /**
     * Initialize device
     */
    hipError_t DeviceInit(int dev = 0)
    {
        hipError_t error = hipSuccess;

        do
        {
            int deviceCount;
            error = hipGetDeviceCount(&deviceCount);
            if (error) break;

            if (deviceCount == 0) {
                fprintf(stderr, "No devices supporting CUDA.\n");
                exit(1);
            }

            error = hipSetDevice(dev);
            if (error) break;

            std::size_t device_free_physmem, device_total_physmem;
            (void)hipMemGetInfo(&device_free_physmem, &device_total_physmem);

            hipDeviceProp_t             deviceProp;

            error = hipGetDeviceProperties(&deviceProp, dev);
            if (error) break;

            if (deviceProp.major < 1) {
                fprintf(stderr, "Device does not support Hip.\n");
                exit(1);
            }

            auto device_giga_bandwidth = float(deviceProp.memoryBusWidth) * deviceProp.memoryClockRate * 2 / 8 / 1000 / 1000;
            {
                printf(
                        "Using device %d: %s ( SM%d, %d SMs, "
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

        } while (0);

        return error;
    }

/******************************************************************************
 * Helper routines for list comparison and display
 ******************************************************************************/

struct GpuTimer
{
    hipEvent_t start;
    hipEvent_t stop;

    GpuTimer()
    {
        (void)hipEventCreate(&start);
        (void)hipEventCreate(&stop);
    }

    ~GpuTimer()
    {
        (void)hipEventDestroy(start);
        (void)hipEventDestroy(stop);
    }

    void Start()
    {
        (void)hipEventRecord(start, 0);
    }

    void Stop()
    {
        (void)hipEventRecord(stop, 0);
    }

    float ElapsedMillis()
    {
        float elapsed;
        (void)hipEventSynchronize(stop);
        (void)hipEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

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
        memset((void *) &key, 0, sizeof(key));
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
