
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include "common/threading.hpp"
#include "common/common_utils.hpp"
#include "common/roc_profiler.h"

#include "test_main.h"

// the number of GPUs communicating (set to -1 to use all available GPUs)
#define NUM_ACTIVE_GPUS 4
#define VERIFY_DATA 0
#define USE_GRAPH_API 1

#define NUM_ELEMS_MIN 0x6000000
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

    VLOG(0) << "initializing rank " << id;
    auto& info = m_infos[id];
    size_t nBytes = m_maxElems*sizeof(T);
    info.gpuId = gpuIDs != nullptr ? gpuIDs[id] : id;
    info.graph = {};
    info.graphExec = {};
    info.graphCreated = false;
    CHK(cudaSetDevice(info.gpuId));

    CHK(cudaStreamCreateWithFlags(&info.stream, cudaStreamNonBlocking));

    size_t totalBytes = s_rcclElems * m_nGpus * sizeof(float);
    CHK(cudaMalloc(&info.rcclSendBuf, totalBytes));
    CHK(cudaMalloc(&info.rcclRecvBuf, totalBytes));
    CHK(cudaMemset(info.rcclSendBuf, 0x42, totalBytes));
    CHK(cudaMemset(info.rcclRecvBuf, 0, totalBytes));

    CHKNCCL(ncclCommInitRank(&info.comm, m_nGpus, m_ncclId, id));

    init_gemm_op(id);
  }); // runJob
  CHK(cudaDeviceSynchronize());
  VLOG(0) << "Init finished..";
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
    (void)ncclCommDestroy(info.comm);
    (void)cudaFree(info.rcclSendBuf);
    (void)cudaFree(info.rcclRecvBuf);
    (void)cudaStreamDestroy(info.stream);
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
  // CHK(cudaMemsetAsync(info.sendBuf, fillVal ^ 0xFF, nBytes, info.stream));
  // CHK(cudaMemsetAsync(info.recvBuf, fillVal, nBytes, info.stream));
}

void TestFramework::verify(int id) {
}

void TestFramework::init_gemm_op(int id) {
  	
  VLOG(0) << "Init gemm for rank: " << id;
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
  info.gemm.run(info.stream, nIters);
}

void TestFramework::run_rccl_op(int id) {

  auto& info = m_infos[id];
  CHKNCCL(ncclGroupStart());
  for(size_t peer = 0; peer < m_nGpus; peer++) {
    CHKNCCL(ncclSend(info.rcclSendBuf + peer * s_rcclElems,
          s_rcclElems, ncclFloat, peer, info.comm, info.stream));
    CHKNCCL(ncclRecv(info.rcclRecvBuf + peer * s_rcclElems,
          s_rcclElems, ncclFloat, peer, info.comm, info.stream));
  }
  CHKNCCL(ncclGroupEnd());
}

void TestFramework::run_thread(int id, int numIters) 
{
  m_barrier.wait(); // wait all threads to arrive here before starting timing
  auto& info = m_infos[id];

#if USE_GRAPH_API
  if(!info.graphCreated) {

    VLOG(0) << "Starting stream capture.." << numIters;
    //CHK(cudaGraphCreate(&info.graph, /*flags=*/0));

    CHK(cudaStreamBeginCapture(info.stream, cudaStreamCaptureModeThreadLocal));
    for(int i = 0; i < 1; i++) {
      run_gemm_op(id, 5);
      run_rccl_op(id);
    }
    CHK(cudaStreamEndCapture(info.stream, &info.graph));

    // VLOG(0) << "Starting stream capture to existing graph..";
    // CHK(cudaStreamBeginCaptureToGraph(info.stream, info.graph,
    //         nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
    // for(int i = 0; i < numIters-1; i++) {
    //   // run_gemm_op(id, 5);
    //   run_rccl_op(id);
    // }

    CHK(hipGetLastError());

    cudaGraph_t graph;
    CHK(cudaStreamEndCapture(info.stream, &graph));
    VLOG(0) << "graph " << graph << " --- " << info.graph;
    if(graph != info.graph) {
      ThrowError<>("Graphs differ!");
    }

    CHK(cudaGraphInstantiate(&info.graphExec, info.graph, NULL, NULL, 0));

    info.graphCreated = true;
  }
  CPU_BEGIN_TIMING(T); 
  CHK(cudaGraphLaunch(info.graphExec, info.stream));
#else
  CPU_BEGIN_TIMING(T); 
  for(int i = 0; i < numIters; i++) {
    run_rccl_op(id);
    CHK(hipGetLastError());
    run_gemm_op(id, 2);
    CHK(hipGetLastError());
  } 
#endif  

  CHK(cudaStreamSynchronize(info.stream));

  auto lastErr = hipGetLastError();
  if(lastErr != hipSuccess) {
    PRINTZ("rank %d: hipGetLastError returned sticky error: %s (%d)",
          id, hipGetErrorString(lastErr), (int)lastErr);
  } else {
    VLOG(0) << "rank " << id << ": hipGetLastError = hipSuccess (no sticky error)";
  }

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

void TestFramework::run(size_t numElems, int numIters, bool measureTime) {

  m_measureTime = measureTime;
  m_curElems = numElems;
  if(m_curElems > m_maxElems) {
    ThrowError< >("numElems must be <= m_maxElems");
  }
  m_pool.runJob([&,this](int id) {
    run_thread(id, numIters);
  });
}

void runTest(size_t elemsMin, size_t elemsMax)
{
  int nGpus = 0, nwarmups = 10, niters = 20;
  CHK(hipGetDeviceCount(&nGpus));
  if(NUM_ACTIVE_GPUS > 0) nGpus = NUM_ACTIVE_GPUS;
  VLOG(0) << "Num devices: " << nGpus;
  
  std::vector< uint32_t > deviceAssignment{0, 1, 2, 3, 4, 5, 6, 7 };
  if(nGpus > deviceAssignment.size()) {
    throw std::runtime_error("Invalid device assignment!");
  }

  //RocProfilerSession s;

  TestFramework obj(nGpus, deviceAssignment.data(), elemsMax);

  //s.start();
  obj.run(elemsMin, nwarmups);
  //s.stop();

}

int main() try 
{
  DeviceInit(0);
  runTest(NUM_ELEMS_MIN, NUM_ELEMS_MIN);
}
catch(std::exception& ex) {
  VLOG(0) << "Exception: " << ex.what();
}
catch(...) {
  VLOG(0) << "Unknown exception";
}
