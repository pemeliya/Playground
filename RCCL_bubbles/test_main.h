
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
struct TestFramework {

  using T = uint32_t;
public:  
  struct ThreadInfo {
    int gpuId;            // gpu ID assigned to this thread
    cudaStream_t stream; // associated streams
    T *sendBuf, *recvBuf; // send and receive buffers
    ncclComm_t comm;      // NCCL handle
    double elapsedMs;     // time elapsed per thread
  };

public:
  TestFramework(size_t nGpus, const uint32_t *gpuIDs);
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

private:
  ncclUniqueId m_ncclId;
  size_t m_nGpus; // total and current data transfer size

  bool m_measureTime = false;
  std::vector< ThreadInfo > m_infos;
  std::vector< T > m_hostBuf;
  std::mutex m_verifyMtx;
  Barrier m_barrier;
  ThreadPool m_pool;
}; // struct TestFramework

#endif // TEST_MAIN_H
