
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include "common/threading.hpp"
#include "common/example_utils.hpp"

#include "test_main.h"

// the number of GPUs communicating (set to -1 to use  available GPUs)
#define NUM_ACTIVE_GPUS 1
#define VERIFY_DATA 0
#define USE_GRAPH_API 1

#define NUM_ELEMS_MIN 0x6000000
#define NUM_ELEMS_MAX 0x6000000


TestFramework::TestFramework(size_t nGpus, const uint32_t *gpuIDs,
       size_t maxElems) : m_nGpus(nGpus), m_maxElems(maxElems),
      m_curElems(maxElems), 
      m_infos(nGpus), m_barrier(nGpus), m_pool(nGpus)
{ 
  m_pool.runJob([this, gpuIDs](int id) {

    auto& info = m_infos[id];
    size_t nBytes = m_maxElems*sizeof(T);
    info.gpuId = gpuIDs != nullptr ? gpuIDs[id] : id;
    info.graph = {};
    info.graphExec = {};
    info.graphCreated = false;
    CHK(cudaSetDevice(info.gpuId));
    int flags = hipDeviceMallocDefault;

    CHK(hipExtMallocWithFlags((void **)&info.sendBuf, nBytes*2, flags));
    info.recvBuf = info.sendBuf + m_maxElems;
    CHK(cudaStreamCreateWithFlags(&info.stream, cudaStreamNonBlocking));

    CHK(cudaMemsetAsync(info.sendBuf, s_fillValue ^ 0xFF, nBytes, info.stream));
    CHK(cudaMemsetAsync(info.recvBuf, s_fillValue, nBytes, info.stream));
    init_gemm_op(id);
  }); // runJob
  CHK(cudaDeviceSynchronize());
    VLOG("Init finished..");
}

TestFramework::~TestFramework() {
  for(auto& info : m_infos) {
    (void)cudaSetDevice(info.gpuId);
#if USE_GRAPH_API
    if(info.graphCreated) {
      (void)cudaGraphExecDestroy(info.graphExec);
      (void)cudaGraphDestroy(info.graph);
    }
#endif
    (void)cudaStreamDestroy(info.stream);
    (void)cudaFree(info.sendBuf);
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

void TestFramework::init_gemm_op(int id) {
  	
  VLOG("Init gemm for rank: " << id);
  auto& gemm = m_infos[id].gemm;

  int M = 600, N = 512, K = 300;
  auto transA = rocblas_operation_transpose,
       transB = rocblas_operation_none;

  int64_t batchCount = 1000;
  gemm.init(m_infos[id].stream);
  gemm.FillParams(M, N, K, transA, transB, batchCount);
  gemm.AllocBuffers();
}

void TestFramework::run_gemm_op(int id, int nIters) {
  
  auto& info = m_infos[id];
  auto stream_status = hipStreamCaptureStatusNone;
  CHK(hipStreamIsCapturing(info.stream, &stream_status));
  VLOG("Stream is capture " << stream_status);

  info.gemm.run(info.stream, nIters);
}

void TestFramework::run_thread(int id, int numIters, bool verifyData) 
{
  m_barrier.wait(); // wait all threads to arrive here before starting timing
  auto& info = m_infos[id];

  if(!info.graphCreated) {

    VLOG("Starting stream capture.." << numIters);

    CHK(cudaStreamBeginCapture(info.stream, cudaStreamCaptureModeThreadLocal));
    for(int i = 0; i < 1; i++) {
      run_gemm_op(id, 5);
    }
    CHK(cudaStreamEndCapture(info.stream, &info.graph));
    CHK(cudaGraphInstantiate(&info.graphExec, info.graph, NULL, NULL, 0));

    info.graphCreated = true;
  }
  CPU_BEGIN_TIMING(T); 
  CHK(cudaGraphLaunch(info.graphExec, info.stream));

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
}

void TestFramework::run(size_t numElems, int numIters, bool measureTime, bool verifyData) {

  m_measureTime = measureTime;
  m_curElems = numElems;
  if(m_curElems > m_maxElems) {
    ThrowError< >("numElems must be <= m_maxElems");
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
  
  std::vector< uint32_t > deviceAssignment{0, 1, 2, 3, 4, 5, 6, 7 };
  if(nGpus > deviceAssignment.size()) {
    throw std::runtime_error("Invalid device assignment!");
  }

  TestFramework obj(nGpus, deviceAssignment.data(), elemsMax);
  obj.run(elemsMin, nwarmups);
}

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
