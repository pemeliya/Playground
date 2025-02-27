// Original program by Chen, Wen adapted
// HIPBLASLT_LOG_MASK=5 HIPBLASLT_LOG_LEVEL=5 ./a.out

#include <iostream>
#include <random>
#include <optional>
#include <iomanip> // For formatting output
#include <fstream>
#include "common/common_utils.hpp"
#include "common/hipblaslt_gemm.hpp"

// #include <Eigen/Dense>
// #include <unsupported/Eigen/CXX11/Tensor> // For bfloat16

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
void printMatrix(const T* matrix, int rows, int cols, int maxRows = 3, int maxCols = 3) {
    std::cout << "Printing matrix at " << matrix << "\n" << std::flush;
    std::cout << "Matrix (" << rows << " x " << cols << "):\n" << std::flush;
    int displayRows = std::min(rows, maxRows);
    int displayCols = std::min(cols, maxCols);
    for (int i = 0; i < displayRows; ++i) {
        for (int j = 0; j < displayCols; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) 
                        << static_cast<float>(matrix[i * cols + j]);
        }
        if (displayCols < cols) std::cout << " ...";
        std::cout << "\n" << std::flush;
    }
    if (displayRows < rows) std::cout << "  ...\n" << std::flush;
    std::cout << "\n" << std::flush;
}


// Templated function to read a matrix from a binary file
template <typename T>
T* readMatrixFromBinary(const char* filename, int size) {
    T* matrix = new T[size];
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open " << filename << " for reading\n";
        delete[] matrix;
        exit(EXIT_FAILURE);
    }
    in.read(reinterpret_cast<char*>(matrix), size * sizeof(T));
    if (!in) {
        std::cerr << "Failed to read " << size << " elements from " << filename << "\n";
        delete[] matrix;
        exit(EXIT_FAILURE);
    }
    in.close();
    return matrix;
}


int main()  {
  hipblasLtHandle_t handle;
  CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));

  // Dimensions of the matrices
  int m = 512;  // rows of A and C
  int n = 512;  // columns of B and C
  int k = 128;  // columns of A and rows of B

  // Scalar values
  float alpha = 1.0f;
  float beta = 0.0f;
  
  using TEST_DATATYPE = float;
  
  // Read matrices from binary files
  // using Eigen::bfloat16;
//   TEST_DATATYPE* h_A = readMatrixFromBinary<TEST_DATATYPE>("matrix_A.bin", m * k);
//   TEST_DATATYPE* h_B = readMatrixFromBinary<TEST_DATATYPE>("matrix_B.bin", k * n);
//   TEST_DATATYPE* h_C = readMatrixFromBinary<TEST_DATATYPE>("matrix_C.bin", m * n);
//   std::cout << "Read matrices from binary files is done.\n";    

//   // Allocate and initialize matrices on the host
  TEST_DATATYPE *h_A = new TEST_DATATYPE[m * k];
  TEST_DATATYPE *h_B = new TEST_DATATYPE[k * n];
  TEST_DATATYPE *h_C = new TEST_DATATYPE[m * n];
  TEST_DATATYPE *h_D = new TEST_DATATYPE[m * n];  
  // Initialize data (for demonstration purposes, use dummy values)
  for (int i = 0; i < m * k; ++i) h_A[i] = static_cast<TEST_DATATYPE>(i % 10);
  for (int i = 0; i < k * n; ++i) h_B[i] = static_cast<TEST_DATATYPE>(i % 10);
  for (int i = 0; i < m * n; ++i) {h_C[i] = 0.0f, h_D[i] = 0.0f;};
  std::cout << "Initialize data is done.\n";

  // Allocate memory on the device
  TEST_DATATYPE *d_A, *d_B, *d_C, *d_D;
  CHECK_HIP_ERROR(hipMalloc(&d_A, m * k * sizeof(TEST_DATATYPE)));
  CHECK_HIP_ERROR(hipMalloc(&d_B, k * n * sizeof(TEST_DATATYPE)));
  CHECK_HIP_ERROR(hipMalloc(&d_C, m * n * sizeof(TEST_DATATYPE)));
  CHECK_HIP_ERROR(hipMalloc(&d_D, m * n * sizeof(TEST_DATATYPE)));

  std::cout << "Allocate memory on the device is done.\n";
  
  // Copy data from host to device
  CHECK_HIP_ERROR(hipMemcpy(d_A, h_A, m * k * sizeof(TEST_DATATYPE), hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(d_B, h_B, k * n * sizeof(TEST_DATATYPE), hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(d_C, h_C, m * n * sizeof(TEST_DATATYPE), hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(d_D, h_D, m * n * sizeof(TEST_DATATYPE), hipMemcpyHostToDevice));

  std::cout << "Copy data from host to device is done.\n";

  // Create matrix descriptors
  hipblasLtMatmulDesc_t matmulDesc;
  hipblasLtMatrixLayout_t layoutA, layoutB, layoutC, layoutD;

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_32F, m, k, m));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_32F, k, n, k));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_32F, m, n, m));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&layoutD, HIP_R_32F, m, n, m));

  std::cout << "Create matrix descriptors is done.\n";

  // Perform matrix multiplication
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(
      handle, matmulDesc,
      &alpha, d_A, layoutA,
      d_B, layoutB,
      &beta, d_C, layoutC,
      d_D, layoutD,
      nullptr, nullptr, 0, 0));

  std::cout << "Perform matrix multiplication is done.\n";

  // Copy the result back to the host
  CHECK_HIP_ERROR(hipMemcpy(h_D, d_D, m * n * sizeof(TEST_DATATYPE), hipMemcpyDeviceToHost));
  std::cout << "Copy the result back to the host is done.\n";

  CHECK_HIP_ERROR(hipDeviceSynchronize());
  std::cout << "GPU synchronization is done.\n";
  CHECK_HIP_ERROR(hipFree(d_A));
  CHECK_HIP_ERROR(hipFree(d_B));
  CHECK_HIP_ERROR(hipFree(d_C));
  CHECK_HIP_ERROR(hipFree(d_D));
  std::cout << "hipFree() all done.\n";
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(layoutA));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(layoutB));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(layoutC));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(layoutD));
  std::cout << "hipblasLtMatrixLayoutDestroy() is done.\n";
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmulDesc));
  std::cout << "hipblasLtMatmulDescDestroy() is done.\n";

  CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
  std::cout << "hipblasLtDestroy() is done.\n";

  // Print matrices before cleanup
  printMatrix(h_A, m, k); std::cout << "Printed h_A\n";
  printMatrix(h_B, k, n); std::cout << "Printed h_B\n";
  printMatrix(h_C, m, n); std::cout << "Printed h_C\n";
  printMatrix(h_D, m, n); std::cout << "Printed h_D\n";

  // Clean up
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] h_D;

  std::cout << "Clean up h_A, h_B, h_C and h_D are done.\n";

  std::cout << "Matrix multiplication completed successfully.\n";
  return 0;
}
