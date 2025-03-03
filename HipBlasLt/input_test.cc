#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath> // For std::isnan
#include <cstring>


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

// Assuming bfloat16 is a custom type; you'll need to adjust conversion if different
struct bfloat16 {
    uint16_t bits; // 16-bit representation
};

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

bool hasNaN(const bfloat16* matrix, int rows, int cols){
    int size = rows * cols;
    for (int i=0; i<size; ++i){
        uint16_t value = matrix[i].bits;
        if (value & 0x7F80 == 0x7F80 && value & 0x007F != 0){
            std::cout << "NaN found at index " << i << " (" << i / cols << ", " << i % cols << ")\n";
            return true;
        }
    }
    return false;
}

int main() {

    const char* input_m = "matrix_A.bin";
    int m = 48, n = 176, k = 1024;
    bfloat16* matrix = readMatrixFromBinary<bfloat16>(input_m, m*k);

    if (hasNaN(matrix, m, k)){
        std::cout << "Matrix contains NaN values.\n";
    } else {
        std::cout << "No NaN values found in the matrix.\n";
    }
    delete [] matrix;
    // Dimensions matching your hipBLASLt code
    // int m = 512;  // rows of A and C
    // int n = 512;  // columns of B and C
    // int k = 512;  // columns of A and rows of B

    // // Allocate matrices
    // float* h_A = new float[m * k];
    // float* h_B = new float[k * n];
    // float* h_C = new float[m * n];

    // // Fill with predictable data for testing
    // for (int i = 0; i < m * k; ++i) {
    //     h_A[i] = static_cast<float>(i % 10) + 0.1f;  // e.g., 0.1, 1.1, ..., 9.1
    // }
    // for (int i = 0; i < k * n; ++i) {
    //     h_B[i] = static_cast<float>(i % 5) + 0.2f;  // e.g., 0.2, 1.2, ..., 4.2
    // }
    // for (int i = 0; i < m * n; ++i) {
    //     h_C[i] = 0.0f;  // Initialize C to zero (pre-GEMM state)
    // }

    // // Write to binary files
    // writeMatrixToBinary("matrix_A.bin", h_A, m * k);
    // writeMatrixToBinary("matrix_B.bin", h_B, k * n);
    // writeMatrixToBinary("matrix_C.bin", h_C, m * n);

    // // Clean up
    // delete[] h_A;
    // delete[] h_B;
    // delete[] h_C;

    // std::cout << "Binary matrix files generated successfully.\n";
    return 0;
}