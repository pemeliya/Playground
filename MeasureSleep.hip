
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
            CHK(hipDeviceSynchronize()) \
            CHK(hipEventRecord( e_start, 0 )) \
        } \
    } \
    CHK(hipEventRecord( e_end, 0 )) \
    CHK(hipEventSynchronize( e_end )) \
    float ms = 0;                  \
    CHK(hipEventElapsedTime( &ms, e_start, e_end )) \
    if(nIters > 0) ms /= nIters; \
    fprintf(stderr, "%s time elapsed: %.f ns\n", #name, (ms*1e6)); \
    }   


__global__ void sleep_kernel(uint32_t num) 
{  
    for(int i = 0; i < num; i++) {
        __builtin_amdgcn_s_sleep(10);
    }
}

int main(int argc, char **argv) try
{
    hipEvent_t e_start, e_end;
    CHK(hipEventCreate(&e_start));
    CHK(hipEventCreate(&e_end));

    uint32_t sleep_duration = (argc > 1 ? atoi(argv[1]) : 100);

    fprintf(stderr, "Running with sleep duration: %d\n", sleep_duration);

    void* kernel = reinterpret_cast<void*>(&sleep_kernel);
    void* kernel_args[] = {&sleep_duration};
    dim3 gridDim = {1, 1, 1};
    dim3 blockDim = {512, 1, 1};
    BEGIN_TIMING(10)
    hipLaunchKernel(kernel, gridDim, blockDim, kernel_args, 0, 0);
    END_TIMING(sleep kernel)

    return 0;
}
catch(std::exception& ex) {
   printf("Exception: %s", ex.what());
}