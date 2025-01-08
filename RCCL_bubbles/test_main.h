
#ifndef TEST_MAIN_H 
#define TEST_MAIN_H 1

#include <iostream>
#include <fstream>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_complex.h>
#include <rocblas/rocblas.h>

#include "common/common_utils.hpp"
#include "common/threading.hpp"

#define CHK_ROCBLAS(error) if(error != rocblas_status_success) { \
    fprintf(stderr, "RocBlas error %s at %s:%d\n", rocblas_status_to_string(error), \
            __FILE__, __LINE__);  \
    fflush(stderr); \
} 

template < class NT >
struct DeviceBuf {
   
   DeviceBuf() = default;

   DeviceBuf(const DeviceBuf&) = delete;
   DeviceBuf& operator=(const DeviceBuf&) = delete;

   void swap(DeviceBuf& lhs) {
    std::swap(devPtr, lhs.devPtr);
   }
   DeviceBuf(size_t N) {
       CHK(cudaMalloc((void**)&devPtr, N*sizeof(NT)))
       CHK(cudaMemset(devPtr, 0x11, N*sizeof(NT)))
   }
   ~DeviceBuf() {
      if(devPtr) {
        (void)cudaFree(devPtr);
      }
   }
   NT *devPtr = nullptr;
};

struct BlasGemm
{
  using TypeA = float;
  using TypeB = float;
  using TypeC = float;
  using TypeD = float;

  BlasGemm() = default;

  void init(cudaStream_t stream) {
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

  void FillParams(int64_t M, int64_t N, int64_t K, 
        rocblas_operation transA, rocblas_operation transB, 
        int32_t batchCount = 0) 
  {
    cfg_ = Config{
      .M = M, .N = N, .K = K,
      .transA = transA, .transB = transB,
      .algo = rocblas_gemm_algo_standard,
      .batchCount = batchCount,
      .solutionIndex = -1,
    };
    if(cfg_.transA == rocblas_operation_none) {
      cfg_.ldA = cfg_.M, cfg_.sizeA = cfg_.K * cfg_.ldA;
      cfg_.strideA1 = 1, cfg_.strideA2 = cfg_.ldA;
    } else {
      cfg_.ldA = cfg_.K, cfg_.sizeA = cfg_.M * cfg_.ldA;
      cfg_.strideA1 = cfg_.ldA, cfg_.strideA2 = 1;
    }
    if(cfg_.transB == rocblas_operation_none) {
      cfg_.ldB = cfg_.K, cfg_.sizeB = cfg_.N * cfg_.ldB;
      cfg_.strideB1 = 1, cfg_.strideB2 = cfg_.ldB;
    } else {
      cfg_.ldB = cfg_.N, cfg_.sizeB = cfg_.K * cfg_.ldB;
      cfg_.strideB1 = cfg_.ldB, cfg_.strideB2 = 1;
    }
    cfg_.ldC = cfg_.M, cfg_.sizeC = cfg_.N * cfg_.ldC;
    cfg_.ldD = cfg_.M, cfg_.sizeD = cfg_.N * cfg_.ldD;
  }

  void AllocBuffers() 
  {
      size_t totalA = cfg_.sizeA * cfg_.batchCount,
         totalB = cfg_.sizeB * cfg_.batchCount,
         totalC = cfg_.sizeC * cfg_.batchCount,
         totalD = cfg_.sizeD * cfg_.batchCount;

      decltype(a)(totalA).swap(a);
      decltype(b)(totalB).swap(b);
      decltype(c)(totalC).swap(c);
      decltype(d)(totalD).swap(d);
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

  template < class TA, class TB, class TC, class TD, class Scalar >
  void gemm_strided_batched_ex(const TA *dA, const TB *dB, const TC *dC, TD *dD,
      Scalar alpha, Scalar beta) {

    CHK_ROCBLAS(rocblas_gemm_strided_batched_ex(handle_,
           cfg_.transA, cfg_.transB, cfg_.M, cfg_.N, cfg_.K,
           &alpha, dA, RocBlasType(dA), cfg_.ldA, cfg_.sizeA,
           dB, RocBlasType(dB), cfg_.ldB, cfg_.sizeB, &beta,
           dC, RocBlasType(dC), cfg_.ldC, cfg_.sizeC,
           dD, RocBlasType(dD), cfg_.ldD, cfg_.sizeD,
           cfg_.batchCount,
           RocBlasType(&alpha), cfg_.algo,
           cfg_.solutionIndex, cfg_.flags))
  }

  template < class TA, class TB, class TC, class TD, class Scalar >
  void gemm_ex(const TA *dA, const TB *dB, const TC *dC, TD *dD, Scalar alpha, 
      Scalar beta) {

    CHK_ROCBLAS(rocblas_gemm_ex(handle_,
           cfg_.transA, cfg_.transB, cfg_.M, cfg_.N, cfg_.K,
           &alpha, dA, RocBlasType(dA), cfg_.ldA,
           dB, RocBlasType(dB), cfg_.ldB, &beta,
           dC, RocBlasType(dC), cfg_.ldC,
           dD, RocBlasType(dD), cfg_.ldD,
           RocBlasType(&alpha), cfg_.algo,
           cfg_.solutionIndex, cfg_.flags))
  }

  void run(cudaStream_t stream, int n_times) {

    CHK_ROCBLAS(rocblas_set_stream(handle_, stream));
    TypeD alpha{1}, beta{0};
    for(int i = 0; i < n_times; i++) {
      if(cfg_.batchCount == 1) {
        gemm_ex(a.devPtr, b.devPtr, c.devPtr, d.devPtr, 
              alpha, beta);
      } else {
        gemm_strided_batched_ex(a.devPtr, b.devPtr, c.devPtr, d.devPtr, 
            alpha, beta);
      }
    }
  }

  DeviceBuf< TypeA > a;
  DeviceBuf< TypeB > b;
  DeviceBuf< TypeC > c;
  DeviceBuf< TypeB > d;

private:
  rocblas_handle handle_; 
  Config cfg_;
};

struct TestFramework {

  using T = uint32_t;
public:  
  struct ThreadInfo {
    int gpuId;            // gpu ID assigned to this thread
    cudaStream_t stream; // associated streams
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    bool graphCreated;
    T *sendBuf, *recvBuf; // send and receive buffers
    ncclComm_t comm;      // NCCL handle
    BlasGemm gemm;        // gemm op handle
    double elapsedMs;     // time elapsed per thread
  };

  constexpr static uint32_t s_bogus = 0xFFFFFFFFu; // to catch uninitialized entries
  constexpr static uint8_t s_fillValue = 0xAA;

public:
  TestFramework(size_t nGpus, const uint32_t *gpuIDs, size_t maxElems);
  ~TestFramework();

  constexpr int32_t getNcclType() {
#define OO(type, id) \
  if constexpr(std::is_same_v<T, type>) return id
    OO(int8_t, ncclInt8);
    OO(uint8_t, ncclUint8);
    OO(int32_t, ncclInt32);
    OO(uint32_t, ncclUint32);
    OO(int64_t, ncclInt64);
    OO(uint64_t, ncclUint64);
    OO(half, ncclFloat16);
    OO(float, ncclFloat32);
    OO(double, ncclFloat64);
#undef OO
  }

  void run_rccl_op(int id, int iter);
  void init_gemm_op(int id);
  void run_gemm_op(int id, int nIters);

  void run(size_t numElems, int numIters, bool measureTime = false, bool verifyData = false);
  void run_thread(int id, int numIters, bool verifyData);
 
private:
  T getElement(int device, size_t idx);
  void fill_verify_data(int id);
  void verify(int id);

private:
  ncclUniqueId m_ncclId;
  size_t m_nGpus, m_maxElems, m_curElems; // total and current data transfer size

  bool m_measureTime = false;
  std::vector< ThreadInfo > m_infos;
  std::vector< T > m_hostBuf;
  std::mutex m_verifyMtx;
  Barrier m_barrier;
  ThreadPool m_pool;
}; // struct TestFramework

#endif // TEST_MAIN_H
