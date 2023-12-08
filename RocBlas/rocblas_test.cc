
/******************** compile with : *************************************/
// HIPBLASLT_LOG_MASK=5 HIPBLASLT_LOG_LEVEL=5 ./a.out

// https://github.com/ROCmSoftwarePlatform/rocBLAS-Examples/blob/develop/Extensions/gemm_ex_f16_r/gemm_ex_f16_r.cpp

#define ROCBLAS_BETA_FEATURES_API
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_complex.h>
#include <memory>
#include <iostream>
#include <rocblas/rocblas.h>
#include "common/example_utils.hpp"

#define USE_BATCHED_GEMM 0

#define CHK_ROCBLAS(error) if(error != rocblas_status_success) { \
    ThrowError<256>("RocBlas error %s at %s:%d\n", rocblas_status_to_string(error), \
            __FILE__, __LINE__);  \
}   

std::ostream& operator<<(std::ostream& os, hipFloatComplex Z) {
  return os << Z.x << "+i*" << Z.y;
}

std::ostream& operator<<(std::ostream& os, hipDoubleComplex Z) {
  return os << Z.x << "+i*" << Z.y;
}

template <typename T, typename U = T, typename V = U>
void matMatMultMixPrec(T alpha, T beta, int M, int N, int K,
    U*  A, int As1, int As2,
    U*  B, int Bs1, int Bs2,
    V*  C, int Cs1, int Cs2,
    V*  D, int Ds1, int Ds2)
{
  for(int i1 = 0; i1 < M; i1++)
  {
    for(int i2 = 0; i2 < N; i2++)
    {
      T t = T(0.0);
      for(int i3 = 0; i3 < K; i3++)
      {
        t += T(A[i1 * As1 + i3 * As2]) * T(B[i3 * Bs1 + i2 * Bs2]);
      }
      D[i1 * Ds1 + i2 * Ds2] = V(beta * T(C[i1 * Cs1 + i2 * Cs2]) + alpha * t);
    }
  }
}

struct BlasGemm
{

  BlasGemm() {
    CHK_ROCBLAS(rocblas_create_handle(&handle_));
    CHK_ROCBLAS(rocblas_set_pointer_mode(handle_, rocblas_pointer_mode_host))
  }
  ~BlasGemm() {
    (void)rocblas_destroy_handle(handle_);
  }

  struct Config {
    int64_t M, N, K;
    rocblas_operation transA, transB;
    rocblas_gemm_algo algo;
    int64_t ldA, ldB, ldC, ldD;
    int64_t sizeA, sizeB, sizeC, sizeD;
    int64_t strideA1, strideA2;
    int64_t strideB1, strideB2;
    int32_t batchCount = 0;
    int32_t solutionIndex = 0;
    uint32_t flags = 0;
  };

  Config FillParams(int64_t M, int64_t N, int64_t K, 
        rocblas_operation transA, rocblas_operation transB, 
        int32_t batchCount = 0) 
  {
    Config cfg = {
      .M = M, .N = N, .K = K,
      .transA = transA, .transB = transB,
      .algo = rocblas_gemm_algo_standard,
      .batchCount = batchCount,
    };
    if(cfg.transA == rocblas_operation_none) {
      cfg.ldA = cfg.M, cfg.sizeA = cfg.K * cfg.ldA;
      cfg.strideA1 = 1, cfg.strideA2 = cfg.ldA;
    } else {
      cfg.ldA = cfg.K, cfg.sizeA = cfg.M * cfg.ldA;
      cfg.strideA1 = cfg.ldA, cfg.strideA2 = 1;
    }
    if(cfg.transB == rocblas_operation_none) {
      cfg.ldB = cfg.K, cfg.sizeB = cfg.N * cfg.ldB;
      cfg.strideB1 = 1, cfg.strideB2 = cfg.ldB;
    } else {
      cfg.ldB = cfg.N, cfg.sizeB = cfg.K * cfg.ldB;
      cfg.strideB1 = cfg.ldB, cfg.strideB2 = 1;
    }
    cfg.ldC = cfg.M, cfg.sizeC = cfg.N * cfg.ldC;
    cfg.ldD = cfg.M, cfg.sizeD = cfg.N * cfg.ldD;
    return cfg;
  }

  template < class T >
  constexpr rocblas_datatype RocBlasType(const T *) {
    if constexpr (std::is_same_v<T, __half>) 
      return rocblas_datatype_f16_r;
    if constexpr (std::is_same_v<T, hip_bfloat16>) 
      return rocblas_datatype_bf16_r;
    if constexpr (std::is_same_v<T, float>) 
      return rocblas_datatype_f32_r;
    if constexpr (std::is_same_v<T, double>) 
      return rocblas_datatype_f64_r;
    if constexpr (std::is_same_v<T, int32_t>) 
      return rocblas_datatype_i32_r;
    if constexpr (std::is_same_v<T, int8_t>) 
      return rocblas_datatype_i8_r;
    if constexpr (std::is_same_v<T, hipFloatComplex>) 
      return rocblas_datatype_f32_c;
    if constexpr (std::is_same_v<T, hipDoubleComplex>) 
      return rocblas_datatype_f64_c;
    return (rocblas_datatype)-1;
  }

  template < class TypeA, class TypeD, class Scalar >
  auto get_solutions_by_type(const TypeA *dA, TypeD *dD, Scalar alpha) {

    int num_sols = 0;
    CHK_ROCBLAS(rocblas_gemm_ex_get_solutions_by_type(
      handle_, RocBlasType(dA), RocBlasType(dD), 
          RocBlasType(&alpha), 0, nullptr, &num_sols));

    std::vector< int32_t > sols(num_sols);
    VLOG("Found solutions: " << num_sols);

    CHK_ROCBLAS(rocblas_gemm_ex_get_solutions_by_type(
      handle_, RocBlasType(dA), RocBlasType(dD), 
          RocBlasType(&alpha), 0, sols.data(), &num_sols));
    return sols;
  }

  template < class TypeA, class TypeB, class TypeC, class TypeD, class Scalar >
  void gemm_strided_batched_ex(const TypeA *dA, const TypeB *dB, const TypeC *dC, 
      TypeD *dD, Scalar alpha, Scalar beta, const Config& cfg) {

    CHK_ROCBLAS(rocblas_gemm_strided_batched_ex(handle_,
           cfg.transA, cfg.transB, cfg.M, cfg.N, cfg.K,
           &alpha, dA, RocBlasType(dA), cfg.ldA, cfg.sizeA,
           dB, RocBlasType(dB), cfg.ldB, cfg.sizeB, &beta,
           dC, RocBlasType(dC), cfg.ldC, cfg.sizeC,
           dD, RocBlasType(dD), cfg.ldD, cfg.sizeD,
           cfg.batchCount,
           RocBlasType(&alpha), cfg.algo,
           cfg.solutionIndex, cfg.flags))
  }

  template < class TypeA, class TypeB, class TypeC, class TypeD, class Scalar >
  void gemm_ex(const TypeA *dA, const TypeB *dB, const TypeC *dC, TypeD *dD, Scalar alpha, 
      Scalar beta, const Config& cfg) {

    CHK_ROCBLAS(rocblas_gemm_ex(handle_,
           cfg.transA, cfg.transB, cfg.M, cfg.N, cfg.K,
           &alpha, dA, RocBlasType(dA), cfg.ldA,
           dB, RocBlasType(dB), cfg.ldB, &beta,
           dC, RocBlasType(dC), cfg.ldC,
           dD, RocBlasType(dD), cfg.ldD,
           RocBlasType(&alpha), cfg.algo,
           cfg.solutionIndex, cfg.flags))
  }

private:
  rocblas_handle handle_; 
};

int main(int argc, char *argv[]) try
{
  using TypeA = float;
  using TypeB = float;
  using TypeC = float;
  using TypeD = float;

	int M = 600, N = 512, K = 300;
  auto transA = rocblas_operation_transpose,
       transB = rocblas_operation_none;
  TypeD alpha{1}, beta{0};

  int64_t batchCount = USE_BATCHED_GEMM ? 1000 : 1;
  BlasGemm gemm;
  auto cfg = gemm.FillParams(M, N, K, transA, transB, batchCount);

  // int32_t num_sols = 0;
  // rocblas_handle H;
  // rocblas_gemm_ex_get_solutions_by_type(H, 
  //        rocblas_datatype_f32_r, rocblas_datatype_f32_r, 
  //       rocblas_datatype_f32_r, 0, nullptr, &num_sols);
  
  size_t totalA = cfg.sizeA * batchCount,
         totalB = cfg.sizeB * batchCount,
         totalC = cfg.sizeC * batchCount,
         totalD = cfg.sizeD * batchCount;

  HVector< TypeA > a(totalA);
  HVector< TypeB > b(totalB);
  HVector< TypeC > c(totalC);
  HVector< TypeD > d(totalD), dHost(cfg.sizeD);

  initRange(a.data(), 1.0, 0.01, a.size());
  initRange(b.data(), 3.0, 0.5, b.size());
  initRange(c.data(), 1113.0, -0.5, c.size());

  a.copyHToD();
  b.copyHToD();
  c.copyHToD();

  int ofs = 5, mod = 8;
  uint32_t minT = 1000000, maxT = 0;
  //for(int i = 0; i < 10; i++) {
  // for(int m = M - ofs, z = 0; m <= M; m++) 
  // for(int n = N - ofs; n <= N; n++) 
  // for(int k = K - ofs; k <= K; k++, z++) {

    //int zM = m & 1, zN = n & 1, zK = k & 1;
    //int oddM = m % mod, oddN = n % mod, oddK = k % mod;
    cfg = gemm.FillParams(M, N, K, transA, transB, batchCount);
    // if(oddM + oddN + oddK > 1)
    //   continue;

#if !USE_BATCHED_GEMM
  //CPU_BEGIN_TIMING(gemm);    

  auto sols = gemm.get_solutions_by_type(a.devPtr, d.devPtr, alpha);
  
  //for(int sol = 0; sol < 100000; sol++) 
  for(auto sol : sols)
  try {
    cfg.solutionIndex = sol;
    cfg.algo = rocblas_gemm_algo_solution_index;
    gemm.gemm_ex(a.devPtr, b.devPtr, c.devPtr, d.devPtr, 
       alpha, beta, cfg);
   VLOG("Testing with sol: " << sol << " succeeded!");
  }
  catch(std::exception& ex) {
    //VLOG("Failed: " << ex.what());
  }

  d.copyDToH();
  //CPU_END_TIMING(gemm, "iter %d: %d x %d x %d", z, m, n, k);
#else // USE_BATCHED_GEMM
  CPU_BEGIN_TIMING(gemm_batched);    
  gemm.gemm_strided_batched_ex(a.devPtr, b.devPtr, c.devPtr, d.devPtr, 
       alpha, beta, cfg);
  d.copyDToH();
  CPU_END_TIMING(gemm_batched, "iter %d: batch: %d, %d x %d x %d", z, 
      batchCount, m, n, k);
#endif // USE_BATCHED_GEMM
//  } // for

#if 0
  for(int i = 0; i < batchCount; i++) {
    matMatMultMixPrec(alpha, beta,
      M, N, K, 
      a.data(), strideA1, strideA2,
      b.data(), strideB1, strideB2,
      c.data(), 1, ldc,
      dHost.data(), 1, ldd);

    TypeD eps = 1e-4;
    checkme(d.data(), dHost.data(), d.size(), d.size(),
        1, eps, true);
  }
#endif        
  return 0;
}
catch(std::exception& ex) {
  VLOG("Exception: " << ex.what());
}
catch(...) {
  VLOG("Unknown exception");
}