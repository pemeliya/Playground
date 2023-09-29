
#include<stdio.h>
#include<hip/hip_runtime.h>

// hipcc -std=c++17 MeasureSleep.hip --offload-arch=gfx90a

#define CHK(x) \
   if(auto res = (x); res != hipSuccess) { \
      throw std::runtime_error("Failed with " + std::to_string(res) + " at line: " + std::to_string(__LINE__)); \
   }

#define BEGIN_TIMING(N_ITERS) { \
    CHK(hipDeviceSynchronize())       \
    uint32_t nIters = N_ITERS;  \
    for(unsigned i = 0; i < nIters + 1; i++) {

#define END_TIMING(name) \
        if(i == 0) { \
            (void)hipDeviceSynchronize(); \
            (void)hipEventRecord(e_start, 0); \
        } \
    } \
    (void)hipEventRecord(e_end, 0); \
    (void)hipEventSynchronize(e_end); \
    float ms = 0;                  \
    (void)hipEventElapsedTime(&ms, e_start, e_end); \
    if(nIters > 0) ms /= nIters; \
    printf("%s time elapsed: %.3f us\n", #name, (ms*1e3)); \
    }   
  
__global__ void sleep_kernel(uint32_t nclocks) 
{  
    if(threadIdx.x == 0) {
        uint32_t niters = (nclocks + 32) / 64; // round the number of iterations
        for(int i = 0; i < niters; i++) {
            __builtin_amdgcn_s_sleep(64);
        }
    }
    __syncthreads();
}

int main(int argc, char **argv) try
{
    int devID = 0;
    hipDeviceProp_t prop{};

    CHK(hipSetDevice(devID));
    CHK(hipGetDeviceProperties(&prop, devID));

    // s_sleep causes a wave to sleep for (64 * SIMM16[6:0] + 1..64) clocks. The exact amount of delay is approximate. Compare with S_NOP

    uint32_t sleep_ns = (argc > 1 ? atoi(argv[1]) : 2000*1000);
    uint64_t nclocks = (uint64_t)((double)prop.clockRate / 64e6 * sleep_ns);

    printf("device clock rate: %d; sleep_clocks: %lu\n", prop.clockRate, nclocks);

    hipEvent_t e_start, e_end;
    CHK(hipEventCreate(&e_start));
    CHK(hipEventCreate(&e_end));

    printf("Running with sleep duration: %d ns\n", sleep_ns);

    void* kernel = reinterpret_cast<void*>(&sleep_kernel);
    void* kernel_args[] = {&nclocks};
    dim3 gridDim = {1, 1, 1};
    dim3 blockDim = {512, 1, 1};
    BEGIN_TIMING(10)
    (void)hipLaunchKernel(kernel, gridDim, blockDim, kernel_args, 0, 0);
    END_TIMING(sleep kernel)

    return 0;
}
catch(std::exception& ex) {
   printf("Exception: %s", ex.what());
}