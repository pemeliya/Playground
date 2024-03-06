
#ifndef TEST_MAIN_H 
#define TEST_MAIN_H 1

#include <iostream>
#include <fstream>

#include "common/common.h"
#include "common/threading.hpp"

template < class T >
struct Matrix : std::vector< T > {

  using Base = std::vector< T >;

  Matrix(uint32_t nrows, uint32_t ncols, const T& val = {}) 
            : Base(ncols*nrows, val),
        m_nrows(nrows), m_ncols(ncols) {
  }
  // access the ith row
  T *operator[](uint32_t i) { 
    return Base::data() + i*m_ncols;
  }
  const T *operator[](uint32_t i) const {
    return Base::data() + i*m_ncols;
  }
  auto numRows() const {
    return m_nrows;
  }
  auto numCols() const {
    return m_ncols;
  }

  std::string printRow(uint32_t row) const {
    auto prow = (*this)[row];
    std::ostringstream oss;
    oss << '{';
    for(uint32_t i = 0; i < m_ncols; i++) {
      oss << prow[i];
      if(i < m_ncols-1) oss << ',';
    }
    oss << '}';
    return oss.str();
  }

private:
  uint32_t m_nrows, m_ncols;
};

struct Node {
  uint32_t in, out; // this Node sends to Node[out] and receives from Node[in]
};

constexpr static uint32_t s_bogus = 0xFFFFFFFFu; // to catch uninitialized entries

void output_dot(const Matrix<Node>& stageA, const Matrix<Node>& stageB);
std::vector< uint32_t > permute_op(uint32_t nGpus);

template < class T >
class GpuComm {

  struct ThreadInfo {
    int gpuId;            // gpu ID assigned to this thread
    std::vector< cudaStream_t > streams; // associated streams
    T *sendBuf, *recvBuf; // send and receive buffers
#if !USE_CUSTOM_QCCL
    ncclComm_t comm;      // NCCL handle
#endif
    double elapsedMs;     // time elapsed per thread
  };

  ncclUniqueId m_ncclId;
  size_t m_nGpus, m_maxElems, m_curElems; // total and current data transfer size
  bool m_measureTime = false;
  std::vector< ThreadInfo > m_infos;
  std::vector< T > m_hostBuf;
  std::mutex m_verifyMtx;
  Barrier m_barrier;
  ThreadPool m_pool;
  Matrix<Node> m_stageA, m_stageB; // "topology graphs" for stage1 all-to-all and stage 2

  constexpr static uint8_t s_fillValue = 0xAA;
  constexpr static uint8_t s_oobValue = 0xDD;
  constexpr static uint32_t s_redzoneElems = 64; // number of OOB elements for redzone check
  constexpr static uint32_t s_nExtraPeers = 0; // if zero, all traffic is sent directly
  constexpr static double s_splitFactor = 1.0; // this much of traffic is sent to target GPUs directly

  std::vector< size_t > m_offsets, m_sizes;

public:
  GpuComm(size_t nGpus, size_t maxElems);

  ~GpuComm();

  constexpr auto getNcclType() {
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

  void init_extra_peers();

  T getElement(int device, size_t idx);

  void init();

  void fill_verify_data(int id);

  void verify(int id);

  void run_single_gpu(int id, int stage);

  void run(size_t numElems, int numIters, bool measureTime = false, bool verifyData = false);

  void run_thread(int id, int numIters, bool verifyData);

}; // struct GpuComm

#endif // TEST_MAIN_H
