// Original program by Chen, Wen adapted

/******************** compile with : *************************************/
// hipcc -lhipblaslt -std=c++17 --offload-arch=gfx90a hipblaslt_test.cc -Wno-backslash-newline-escape
// HIPBLASLT_LOG_MASK=5 HIPBLASLT_LOG_LEVEL=5 ./a.out

#include <iostream>
#include <random>
#include <optional>
#include <iomanip> // For formatting output

#include "common/common_utils.hpp"
#include "common/hipblaslt_gemm.hpp"

#define LOG(x) std::cerr << x << std::endl

#define CHK_HIP(error) if(error != hipSuccess) { \
        fprintf(stderr, "Hip error: '%s'(%d) at %s:%d\n", hipGetErrorString(error),  \
                error, __FILE__, __LINE__); throw 0;  \
    }

// Helper function to handle hip errors
#define CHECK_HIP_ERROR(expr)                                         \
   do {                                                              \
       hipError_t err = (expr);                                      \
       if (err != hipSuccess) {                                      \
           std::cerr << "HIP error: " << hipGetErrorString(err) << '\n'; \
           exit(EXIT_FAILURE);                                       \
       }                                                             \
   } while (0)

// Helper function to handle hipBLASLt errors
#define CHECK_HIPBLASLT_ERROR(expr)                                 \
   do {                                                            \
       hipblasStatus_t status = (expr);                            \
       if (status != HIPBLAS_STATUS_SUCCESS) {                     \
           std::cerr << "hipBLASLt error: " << status << '\n';     \
           exit(EXIT_FAILURE);                                     \
       }                                                           \
   } while (0)

template <typename T>
void printMatrix(const T* matrix, int rows, int cols, int maxRows = 5, int maxCols = 5) {
    std::cout << "Matrix (" << rows << " x " << cols << "):\n";
    
    // Limit output to avoid flooding the terminal for large matrices
    int displayRows = std::min(rows, maxRows);
    int displayCols = std::min(cols, maxCols);

    for (int i = 0; i < displayRows; ++i) {
        for (int j = 0; j < displayCols; ++j) {
            // Access element in row-major order: index = row * cols + col
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) 
                      << matrix[i * cols + j];
        }
        // Add ellipsis if there are more columns
        if (displayCols < cols) {
            std::cout << " ...";
        }
        std::cout << "\n";
    }
    // Add ellipsis if there are more rows
    if (displayRows < rows) {
        std::cout << "  ...\n";
    }
    std::cout << "\n";
}


int main()  {
  hipblasLtHandle_t handle;
  CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));

  // Dimensions of the matrices
  int m = 512;  // rows of A and C
  int n = 512;  // columns of B and C
  int k = 512;  // columns of A and rows of B

  // Scalar values
  float alpha = 1.0f;
  float beta = 0.0f;

  // Allocate and initialize matrices on the host
  float *h_A = new float[m * k];
  float *h_B = new float[k * n];
  float *h_C = new float[m * n];

  // Initialize data (for demonstration purposes, use dummy values)
  for (int i = 0; i < m * k; ++i) h_A[i] = static_cast<float>(i % 10);
  for (int i = 0; i < k * n; ++i) h_B[i] = static_cast<float>(i % 10);
  for (int i = 0; i < m * n; ++i) h_C[i] = 0.0f;
  std::cout << "Initialize data is done.\n";

  // Allocate memory on the device
  float *d_A, *d_B, *d_C;
  CHECK_HIP_ERROR(hipMalloc(&d_A, m * k * sizeof(float)));
  CHECK_HIP_ERROR(hipMalloc(&d_B, k * n * sizeof(float)));
  CHECK_HIP_ERROR(hipMalloc(&d_C, m * n * sizeof(float)));
  std::cout << "Allocate memory on the device is done.\n";
  
  // Copy data from host to device
  CHECK_HIP_ERROR(hipMemcpy(d_A, h_A, m * k * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(d_B, h_B, k * n * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(d_C, h_C, m * n * sizeof(float), hipMemcpyHostToDevice));
  std::cout << "Copy data from host to device is done.\n";

  // Create matrix descriptors
  hipblasLtMatmulDesc_t matmulDesc;
  hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_32F, m, k, m));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_32F, k, n, k));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_32F, m, n, m));
  std::cout << "Create matrix descriptors is done.\n";

  // Perform matrix multiplication
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(
      handle, matmulDesc,
      &alpha, d_A, layoutA,
      d_B, layoutB,
      &beta, d_C, layoutC,
      d_C, layoutC,
      nullptr, nullptr, 0, 0));

  std::cout << "Perform matrix multiplication is done.\n";

  // Copy the result back to the host
  CHECK_HIP_ERROR(hipMemcpy(h_C, d_C, m * n * sizeof(float), hipMemcpyDeviceToHost));
  std::cout << "Copy the result back to the host is done.\n";

  // Print matrices before cleanup
  printMatrix(h_A, m, k); std::cout << "Printed h_A\n" << std::flush;
  printMatrix(h_B, k, n); std::cout << "Printed h_B\n" << std::flush;
  printMatrix(h_C, m, n); std::cout << "Printed h_C\n" << std::flush;

  CHECK_HIP_ERROR(hipDeviceSynchronize());
  std::cout << "GPU synchronization is done.\n";
  // Clean up
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  std::cout << "Clean up h_A, h_B, h_C are done.\n";
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(layoutA));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(layoutB));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(layoutC));
  std::cout << "hipblasLtMatrixLayoutDestroy layoutA, layoutB, layoutC are done.\n";
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmulDesc));
  std::cout << "hipblasLtMatmulDescDestroy is done.\n";

  std::cout << "Matrix multiplication completed successfully.\n";
  return 0;
}
