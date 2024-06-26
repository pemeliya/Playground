
#ifndef TEST_MAIN_H 
#define TEST_MAIN_H 1

#include <iostream>
#include <fstream>

#include "common/common.h"
#include "common/threading.hpp"

// whether to test all-to-all or collective-permute
#define TEST_ALL_TO_ALL 1

#define USE_CUSTOM_QCCL 1
// the number of GPUs communicating (set to -1 to use all available GPUs)
#define NUM_ACTIVE_GPUS 8

#if !TEST_ALL_TO_ALL
// whether to use light variant with just 3 GPUs for debugging extra peers
#define USE_DEBUG_CONFIG_3_GPUS 0
// if zero, all traffic is sent directly to target GPUs 
// this has no effect if USE_CUSTOM_QCCL = 0
#define NUM_EXTRA_PEERS 1
// this portion of traffic is sent to target GPUs directly (1: whole traffic)
// this has no effect if USE_CUSTOM_QCCL = 0
#define EXTRA_PEERS_SPLIT_FACTOR 0.7
#else
#define USE_DEBUG_CONFIG_3_GPUS 0
#define NUM_EXTRA_PEERS 0
#define EXTRA_PEERS_SPLIT_FACTOR 1.0
#endif

#define VERIFY_DATA 1
// run only one verify iteration and then quit
#define STOP_AFTER_VERIFY 0

#if 0
#define NUM_ELEMS_MIN 2322432
#define NUM_ELEMS_MAX 9289728*8
#else
#define NUM_ELEMS_MIN 1024*1024
#define NUM_ELEMS_MAX 1024*1024*8
#endif

#if USE_DEBUG_CONFIG_3_GPUS && !USE_CUSTOM_QCCL
#error Debug config only works for custom QCCL!
#endif

#define CHKNCCL(cmd) \
  if(auto res = (cmd); res != ncclSuccess) {           \
    PRINTZ("Test NCCL failure %s:%d '%s'",              \
        __FILE__,__LINE__, ncclGetErrorString(res));     \
  }


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

struct TestFramework {

  using T = uint32_t;
public:  
  struct ThreadInfo {
    int gpuId;            // gpu ID assigned to this thread
    cudaStream_t stream; // associated streams
    T *sendBuf, *recvBuf; // send and receive buffers
#if !USE_CUSTOM_QCCL
    ncclComm_t comm;      // NCCL handle
#endif
    double elapsedMs;     // time elapsed per thread
  };

  struct Node {
    uint32_t in, out; // this Node sends to Node[out] and receives from Node[in]
  };
  constexpr static uint32_t s_bogus = 0xFFFFFFFFu; // to catch uninitialized entries
  constexpr static uint8_t s_fillValue = 0xAA;
  constexpr static uint8_t s_oobValue = 0xDD;
  constexpr static uint32_t s_redzoneElems = 64; // number of OOB elements for redzone check

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

  void run_single_gpu(int id);
  void run(size_t numElems, int numIters, bool measureTime = false, bool verifyData = false);
  void run_thread(int id, int numIters, bool verifyData);

private:
  void init_extra_peers();
  T getElement(int device, size_t idx);
  void fill_verify_data(int id);
  void verify(int id);

  void output_dot();
  std::vector< uint32_t > permute_op();

  friend std::ostream& operator<<(std::ostream& ofs, const Node& n) {
    return ofs << '(' << n.in << ',' << n.out << ')';
  }

private:
  ncclUniqueId m_ncclId;
  size_t m_nGpus, m_maxElems, m_curElems; // total and current data transfer size
  size_t m_nExtraPeers; // if zero, all traffic is sent directly
  double m_splitFactor; // this much of traffic is sent to target GPUs directly

  bool m_measureTime = false;
  std::vector< ThreadInfo > m_infos;
  std::vector< T > m_hostBuf;
  std::mutex m_verifyMtx;
  Barrier m_barrier;
  ThreadPool m_pool;
  Matrix<Node> m_commGraph; // "topology graph" for all-to-all communication
  std::vector< size_t > m_offsets, m_sizes;

}; // struct TestFramework

#endif // TEST_MAIN_H
