#include <iostream>
#include <fstream>
#include <cstdlib>

// Templated function to write a matrix to a binary file
template <typename T>
void writeMatrixToBinary(const char* filename, T* matrix, int size) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open " << filename << " for writing\n";
        exit(EXIT_FAILURE);
    }
    out.write(reinterpret_cast<char*>(matrix), size * sizeof(T));
    if (!out) {
        std::cerr << "Failed to write " << size << " elements to " << filename << "\n";
        exit(EXIT_FAILURE);
    }
    out.close();
    std::cout << "Wrote " << filename << " successfully\n";
}

int main() {
    // Dimensions matching your hipBLASLt code
    int m = 512;  // rows of A and C
    int n = 512;  // columns of B and C
    int k = 512;  // columns of A and rows of B

    // Allocate matrices
    float* h_A = new float[m * k];
    float* h_B = new float[k * n];
    float* h_C = new float[m * n];

    // Fill with predictable data for testing
    for (int i = 0; i < m * k; ++i) {
        h_A[i] = static_cast<float>(i % 10) + 0.1f;  // e.g., 0.1, 1.1, ..., 9.1
    }
    for (int i = 0; i < k * n; ++i) {
        h_B[i] = static_cast<float>(i % 5) + 0.2f;  // e.g., 0.2, 1.2, ..., 4.2
    }
    for (int i = 0; i < m * n; ++i) {
        h_C[i] = 0.0f;  // Initialize C to zero (pre-GEMM state)
    }

    // Write to binary files
    writeMatrixToBinary("matrix_A.bin", h_A, m * k);
    writeMatrixToBinary("matrix_B.bin", h_B, k * n);
    writeMatrixToBinary("matrix_C.bin", h_C, m * n);

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    std::cout << "Binary matrix files generated successfully.\n";
    return 0;
}