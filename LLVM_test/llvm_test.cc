
#include <iostream>
#include <hip/hip_runtime.h>
#include "common/example_utils.hpp"

__global__ void add_arrays(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int size = 1024;
    const int bytes = size * sizeof(float);

    // Allocate and initialize host memory
    float* h_a = new float[size];
    float* h_b = new float[size];
    float* h_c = new float[size];

    for (int i = 0; i < size; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);

    // Copy data to device
    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    hipLaunchKernelGGL(add_arrays, dim3(blocks), dim3(threads), 0, 0, d_a, d_b, d_c, size);

    // Copy result back to host
    hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < size; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cerr << "Error at index " << i << std::endl;
            return -1;
        }
    }
    std::cout << "Success!" << std::endl;

    // Clean up
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);

    return 0;
}
