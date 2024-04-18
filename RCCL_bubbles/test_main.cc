
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include "common/threading.hpp"
#include "common/example_utils.hpp"
#include "common/roc_profiler.h"

#include "test_main.h"

// the number of GPUs communicating (set to -1 to use all available GPUs)
#define NUM_ACTIVE_GPUS 8
#define VERIFY_DATA 0

#define TEST_COLLECTIVE_PERMUTE 0
#define TEST_ALL_REDUCE 1
#define TEST_ALL_TO_ALL 0

#define NUM_ELEMS_MIN 0x10000
#define NUM_ELEMS_MAX 0x6000000

#define CHKNCCL(cmd) \
  if(auto res = (cmd); res != ncclSuccess) {           \
    ThrowError<>("Test NCCL failure %s:%d '%s'",              \
        __FILE__,__LINE__, ncclGetErrorString(res));     \
  }

TestFramework::TestFramework(size_t nGpus, const uint32_t *gpuIDs,
       size_t maxElems) : m_nGpus(nGpus), m_maxElems(maxElems),
      m_curElems(maxElems), 
      m_infos(nGpus), m_barrier(nGpus), m_pool(nGpus)
{ 
  CHKNCCL(ncclGetUniqueId(&m_ncclId));

  m_pool.runJob([this, gpuIDs](int id) {

     VLOG("initializing rank " << id);
    auto& info = m_infos[id];
    size_t nBytes = m_maxElems*sizeof(T);
    info.gpuId = gpuIDs != nullptr ? gpuIDs[id] : id;
    CHK(cudaSetDevice(info.gpuId));
    int flags = hipDeviceMallocDefault;
                //hipDeviceMallocFinegrained;
                // hipDeviceMallocUncached;
                // hipMallocSignalMemory;
    CHK(hipExtMallocWithFlags((void **)&info.sendBuf, nBytes*2, flags));
    info.recvBuf = info.sendBuf + m_maxElems;
    CHK(cudaStreamCreateWithFlags(&info.stream, cudaStreamNonBlocking));

    CHK(cudaMemsetAsync(info.sendBuf, s_fillValue ^ 0xFF, nBytes, info.stream));
    CHK(cudaMemsetAsync(info.recvBuf, s_fillValue, nBytes, info.stream));

    CHKNCCL(ncclCommInitRank(&info.comm, m_nGpus, m_ncclId, id));
    init_gemm_op(id);
  }); // runJob
  CHK(cudaDeviceSynchronize());
    VLOG("Init finished..");
}

TestFramework::~TestFramework() {
  for(auto& info : m_infos) {
    (void)cudaSetDevice(info.gpuId);
    (void)cudaStreamDestroy(info.stream);
    (void)cudaFree(info.sendBuf);
    (void)ncclCommDestroy(info.comm);
  }
}

auto TestFramework::getElement(int device, size_t idx) -> T {
  //return static_cast<T>(100.0f*std::sin((float)idx*2*M_PI/(device+2)));
  int ii = idx + 1;
  return static_cast<T>(device + 11111 ^ (ii*ii*ii));
  //return static_cast<T>(device + ii);
}

void TestFramework::fill_verify_data(int id) {

  size_t nBytes = m_curElems*sizeof(T);
  auto& info = m_infos[id];
  CHK(cudaSetDevice(info.gpuId));

  auto fillVal = 0x80 + id; // s_fillValue
  CHK(cudaMemsetAsync(info.sendBuf, fillVal ^ 0xFF, nBytes, info.stream));
  CHK(cudaMemsetAsync(info.recvBuf, fillVal, nBytes, info.stream));

  std::vector< T > refBuf(m_curElems);
#if VERIFY_DATA
  for(size_t i = 0; i < m_curElems; i++) {
    refBuf[i] = getElement(id, i);
  }
#else
  std::fill(refBuf.begin(), refBuf.end(), T(id));
#endif
  CHK(cudaMemcpyAsync(info.sendBuf, refBuf.data(), nBytes, cudaMemcpyHostToDevice,
            info.stream));
}

void TestFramework::verify(int id) {
  std::lock_guard _(m_verifyMtx);
  auto sz = m_curElems;
  if(m_hostBuf.size() < sz) {
    m_hostBuf.resize(sz);
  }
  auto dst = m_hostBuf.data();
  CHK(cudaMemcpy(dst, m_infos[id].recvBuf, sz*sizeof(T), cudaMemcpyDeviceToHost));
#if TEST_COLLECTIVE_PERMUTE
  auto t = (id - 1 + m_nGpus) % m_nGpus;
  VLOG("Device " << id << " verifying: expecting data from: " << t);
  for(uint32_t j = 0, num = 0; j < m_curElems; j++) {
    auto truth = getElement(t, j);
    if(dst[j] != truth) {
      //ThrowError<>("%d: verify failed truth: %f gpu: %f", j, truth, dst[j]);
      PRINTZ("0x%X/%d: verify failed truth: %d gpu: %d (%X)", j, j, 
              truth, dst[j], dst[j]);
      if(num++ >= 5)
        break;
    }
  }
#endif
}

void TestFramework::run_rccl_op(int id, int iter) 
{
  auto& info = m_infos[id];
  auto type = (ncclDataType_t)getNcclType();
  int rank;
  // CHKNCCL(ncclCommCount(m_comms[i], &nRanks));
  // CHKNCCL(ncclCommCuDevice(m_comms[i], &dev));
  ncclCommUserRank(info.comm, &rank);

  switch(iter % 4) {
  case 0: { // collective-permute
    CHKNCCL(ncclGroupStart());
    int sendP = (id + 1) % m_nGpus,
      recvP = (id - 1 + m_nGpus) % m_nGpus;
    CHKNCCL(ncclSend(info.sendBuf, 
        m_curElems, type, sendP, info.comm, info.stream));
    CHKNCCL(ncclRecv(info.recvBuf, 
        m_curElems, type, recvP, info.comm, info.stream));
    CHKNCCL(ncclGroupEnd());
    break;
  }
  case 1: { // all-reduce
    ncclRedOp_t op = ncclSum;
    CHKNCCL(ncclAllReduce(info.sendBuf, info.recvBuf, m_curElems, 
        type, op, info.comm, info.stream));
    break;
  }
  case 2: { // all-to-all
    // each GPU sends its part of the buf to all other GPUs..
    CHKNCCL(ncclAllToAll(info.sendBuf, info.recvBuf, m_curElems / m_nGpus, 
        type, info.comm, info.stream));
    break;
  }
  case 3: { // reduce-scatter
    ncclRedOp_t op = ncclSum;
    CHKNCCL(ncclReduceScatter(info.sendBuf, info.recvBuf, m_curElems / m_nGpus, 
        type, op, info.comm, info.stream));
    break;
  }
  } // switch
}

void TestFramework::init_gemm_op(int id) {
  	
  VLOG("Init gemm for rank: " << id);
  auto& gemm = m_infos[id].gemm;

  int M = 600, N = 512, K = 300;
  auto transA = rocblas_operation_transpose,
       transB = rocblas_operation_none;

  int64_t batchCount = 1000;
  gemm.init();
  gemm.FillParams(M, N, K, transA, transB, batchCount);
  gemm.AllocBuffers();
}

void TestFramework::run_gemm_op(int id, int nIters) {
  
  auto& info = m_infos[id];
  info.gemm.run(nIters, info.stream);
}

void TestFramework::run_thread(int id, int numIters, bool verifyData) 
{
  m_barrier.wait(); // wait all threads to arrive here before starting timing
  auto& info = m_infos[id];

  CPU_BEGIN_TIMING(T); 
  for(int i = 0; i < numIters; i++) {
    //VLOG("\n============================ " << m_curElems << " =============================\n");
    run_gemm_op(id, 2);
    run_rccl_op(id, i);
  } 

  CHK(cudaStreamSynchronize(info.stream));
  auto tnow = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> ms = tnow - z1_T;
  size_t bytes = m_curElems*sizeof(T);
  info.elapsedMs = ms.count() / numIters;
  
  m_barrier.wait(); // wait before data verification since it requires all GPUs
  if(id == 0 && m_measureTime) {
    double avgMs = 0;
    for(const auto& s : m_infos) {
      avgMs += s.elapsedMs;
    }
    avgMs /= m_nGpus;
    PRINTZ("Data size: %.2f Mb; avg time elapsed: %.3f ms", 
          (double)bytes/(1024*1024), avgMs);
  }
  if(verifyData) {
    verify(id);
  }
}

void TestFramework::run(size_t numElems, int numIters, bool measureTime, bool verifyData) {

  m_measureTime = measureTime;
  m_curElems = numElems;
  if(m_curElems > m_maxElems) {
    ThrowError< >("numElems must be <= m_maxElems");
  }
  if(verifyData) {
    m_pool.runJob([&,this](int id) {
      fill_verify_data(id);
    });
  }
  if(verifyData) {
    size_t total = m_curElems * sizeof(T);
    PRINTZ("curElems: %zu / 0x%lX (%zu / %lX bytes)", 
          m_curElems, m_curElems, total, total);
  }
  m_pool.runJob([&,this](int id) {
    run_thread(id, numIters, verifyData);
  });
}

void runRCCLTest(size_t elemsMin, size_t elemsMax)
{
  int nGpus = 0, nwarmups = 10, niters = 20;
  CHK(hipGetDeviceCount(&nGpus));
  if(NUM_ACTIVE_GPUS > 0) nGpus = NUM_ACTIVE_GPUS;
  VLOG("Num devices: " << nGpus << "; max data size: " << 
      (double)(elemsMax*sizeof(TestFramework::T))/(1024*1024) << 
        " Mb; neighbour exchange with RCCL"
  );
  std::vector< uint32_t > deviceAssignment{ 0, 1, 2, 3, 4, 5, 6, 7 };
  if(nGpus > deviceAssignment.size()) {
    throw std::runtime_error("Invalid device assignment!");
  }

  TestFramework obj(nGpus, deviceAssignment.data(), elemsMax);
#if VERIFY_DATA
  obj.run(elemsMin, 1, false, true); // first run to verify data
#endif
  //obj.run(elemsMax, (nwarmups+1)/2);
  obj.run(elemsMin, nwarmups);

#if 1
  for(size_t sz = elemsMin; sz <= elemsMax; ) {
    obj.run(sz, niters, true);
    if(sz == elemsMax)
      break;
    sz = std::min(sz * 3 / 2, elemsMax);
  }
#else
  for(size_t sz = elemsMax; sz >= elemsMin; ) {
    obj.run(sz, niters, true);
    if(sz == elemsMin)
      break;
    sz = std::max(sz * 2 / 3, elemsMin);
  }
#endif

}
// NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL

int main() try 
{
  DeviceInit(0);
  runRCCLTest(NUM_ELEMS_MIN, NUM_ELEMS_MIN);
}
catch(std::exception& ex) {
  VLOG("Exception: " << ex.what());
}
catch(...) {
  VLOG("Unknown exception");
}
