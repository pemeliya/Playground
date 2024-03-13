
// hipcc -I.. -DCOMPILE_FOR_ROCM=1 -std=c++17 --offload-arch=gfx90a test_main.cc

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
#include "qccl_lib.h"

#define USE_CUSTOM_QCCL 1
// whether to use light variant with just 3 GPUs for debugging extra peers
#define USE_DEBUG_CONFIG_3_GPUS 1
// if zero, all traffic is sent directly to target GPUs
#define NUM_EXTRA_PEERS 0
// this portion of traffic is sent to target GPUs directly (1: whole traffic)
#define EXTRA_PEERS_SPLIT_FACTOR 1 
#define VERIFY_DATA 1

#if USE_DEBUG_CONFIG_3_GPUS && !USE_CUSTOM_QCCL
#error Debug config only works for custom QCCL!
#endif

#if 0
Using device 0: AMD Instinct MI300X ( SM940, 304 SMs, 196148 free / 196592 total MB physmem, 2662.400 GB/s @ 1300000 kHz mem clock, ECC off)
Num devices: 2; max data size: 283.5 Mb; neighbour exchange with RCCL
Data size: 8.86 Mb; time elapsed: 0.493 ms, bandwidth: 18.847 Gb/s
Data size: 13.29 Mb; time elapsed: 0.702 ms, bandwidth: 19.840 Gb/s
Data size: 19.93 Mb; time elapsed: 1.030 ms, bandwidth: 20.302 Gb/s
Data size: 29.90 Mb; time elapsed: 1.502 ms, bandwidth: 20.876 Gb/s
Data size: 44.85 Mb; time elapsed: 1.368 ms, bandwidth: 34.386 Gb/s
Data size: 67.28 Mb; time elapsed: 2.004 ms, bandwidth: 35.207 Gb/s
Data size: 100.91 Mb; time elapsed: 2.976 ms, bandwidth: 35.556 Gb/s
Data size: 151.37 Mb; time elapsed: 4.422 ms, bandwidth: 35.892 Gb/s
Data size: 227.06 Mb; time elapsed: 6.565 ms, bandwidth: 36.265 Gb/s
Data size: 283.50 Mb; time elapsed: 8.175 ms, bandwidth: 36.365 Gb/s
Thread pool joined
#endif

#define CHKNCCL(cmd) \
  if(auto res = (cmd); res != ncclSuccess) {           \
    PRINTZ("Test NCCL failure %s:%d '%s'",              \
        __FILE__,__LINE__, ncclGetErrorString(res));     \
  }

TestFramework::TestFramework(size_t nGpus, const uint32_t *gpuIDs,
       size_t maxElems) : m_nGpus(nGpus), m_maxElems(maxElems),
      m_curElems(maxElems), 
      m_nExtraPeers(NUM_EXTRA_PEERS), 
      m_splitFactor(EXTRA_PEERS_SPLIT_FACTOR), 
      m_infos(nGpus), m_barrier(nGpus), m_pool(nGpus),
      m_commGraph(nGpus, m_nExtraPeers + 1, Node{s_bogus,s_bogus}) 
{ 
  if((m_nExtraPeers == 0 && m_splitFactor != 1.0) || m_nExtraPeers >= m_nGpus-1) {
    throw std::runtime_error("Wrong number of extra peers!");
  }

#if USE_CUSTOM_QCCL
  CHKQCCL(qcclInit(m_nGpus, gpuIDs));
#else
  CHKNCCL(ncclGetUniqueId(&m_ncclId));
#endif

  m_pool.runJob([this, gpuIDs](int id) {
    auto& info = m_infos[id];
    size_t obytes = s_redzoneElems*sizeof(T),
          nBytes = (m_maxElems + s_redzoneElems)*sizeof(T);
    {
      std::lock_guard _(m_verifyMtx);
      VLOG("Allocating buffers and data init for GPU " << id);
    }
    info.gpuId = gpuIDs != nullptr ? gpuIDs[id] : id;
    CHK(cudaSetDevice(info.gpuId));
    int flags = hipDeviceMallocDefault;
                //hipDeviceMallocFinegrained;
                // hipDeviceMallocUncached;
                // hipMallocSignalMemory;
    CHK(hipExtMallocWithFlags((void **)&info.sendBuf, nBytes*2, flags));
    info.recvBuf = info.sendBuf + m_maxElems + s_redzoneElems;
    CHK(cudaStreamCreateWithFlags(&info.stream, cudaStreamNonBlocking));

    CHK(cudaMemsetAsync(info.sendBuf, s_fillValue ^ 0xFF, nBytes, info.stream));
    CHK(cudaMemsetAsync(info.recvBuf, s_fillValue, nBytes, info.stream));

#if !USE_CUSTOM_QCCL
    CHKNCCL(ncclCommInitRank(&info.comm, m_nGpus, m_ncclId, id));
#endif
  }); // runJob
  init_extra_peers();
  CHK(cudaDeviceSynchronize());
}

TestFramework::~TestFramework() {
  for(auto& info : m_infos) {
    (void)cudaSetDevice(info.gpuId);
    (void)cudaStreamDestroy(info.stream);
    (void)cudaFree(info.sendBuf);
#if !USE_CUSTOM_QCCL
    (void)ncclCommDestroy(info.comm);
#endif
    }
}

auto TestFramework::getElement(int device, size_t idx) -> T {
  //return static_cast<T>(100.0f*std::sin((float)idx*2*M_PI/(device+2)));
  return static_cast<T>(device - 11111);
}

void TestFramework::fill_verify_data(int id) {

  size_t obytes = s_redzoneElems*sizeof(T),
         nBytes = m_curElems*sizeof(T);
  auto& info = m_infos[id];
  CHK(cudaSetDevice(info.gpuId));

  CHK(cudaMemsetAsync(info.sendBuf, s_fillValue ^ 0xFF, nBytes, info.stream));
  CHK(cudaMemsetAsync(info.sendBuf + m_curElems, s_oobValue ^ 0xFF, obytes, info.stream));
  CHK(cudaMemsetAsync(info.recvBuf, s_fillValue, nBytes, info.stream));
  CHK(cudaMemsetAsync(info.recvBuf + m_curElems, s_oobValue, obytes, info.stream));

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
  auto sz = m_curElems + s_redzoneElems;
  if(m_hostBuf.size() < sz) {
    m_hostBuf.resize(sz);
  }
  auto dst = m_hostBuf.data();
  CHK(cudaMemcpy(dst, m_infos[id].recvBuf, sz*sizeof(T), cudaMemcpyDeviceToHost));
  // Node id should receive original data from node m_commGraph[id][0].in
  auto t = m_commGraph[id][0].in;
#if USE_DEBUG_CONFIG_3_GPUS  
  if(id == 2) // HACK HACK
    return;
  t = 1 - id;
#endif

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
  auto bdst = (const uint8_t *)(dst + m_curElems);
  for(uint32_t j = 0, num = 0; j < s_redzoneElems*sizeof(T); j++) {
    if(bdst[j] != s_oobValue) {
      PRINTZ("%X: redzone value modified truth: %X gpu %X", j, s_oobValue, bdst[j]);
      if(num++ >= 5)
        break;
    }
  }
}

#if USE_CUSTOM_QCCL
void TestFramework::run_single_gpu(int id) 
{
  auto& info = m_infos[id];
  const auto& V = m_commGraph[id];
#if !USE_DEBUG_CONFIG_3_GPUS
  uint32_t numSubscribedPeers = 1 + m_nExtraPeers;
  for(int i = 0; i <= m_nExtraPeers; i++) {
      int inP = V[i].in, outP = V[i].out;
      auto size = m_sizes[i] * sizeof(T);
      if(i == 0) {
        CHKQCCL(qcclSendRecv(id, numSubscribedPeers, inP, info.recvBuf, size,
            outP, info.sendBuf, size));
      } else {
        CHKQCCL(qcclGatewaySend(id, numSubscribedPeers, inP, outP, 
              m_offsets[i], size));
      }
  } 
#else
  uint32_t numSubscribedPeers = 2;
  // make sizes divisible by 4
  size_t size = m_curElems * sizeof(T),
         sz1 = (size * 3 / 3 + 3) & ~3, 
         sz2 = (size - sz1);

  if(id == 0 || id == 1) {
    // we send and receive to/from the same node (bidirectional)
    int sendP = 1 - id, recvP = 1 - id;
    CHKQCCL(qcclSendRecv(id, numSubscribedPeers, recvP, info.recvBuf, sz1,
          sendP, info.sendBuf, sz1));
  } else if(id == 2) {
    // create gateway peer
    CHKQCCL(qcclGatewaySend(id, numSubscribedPeers, 0, 1, sz1, sz2));
    CHKQCCL(qcclGatewaySend(id, numSubscribedPeers, 1, 0, sz1, sz2));
  }
#endif
  CHKQCCL(qcclRun(id, info.stream));
}
#else // !USE_CUSTOM_QCCL

void TestFramework::run_single_gpu(int id) 
{
  auto& info = m_infos[id];
  const auto& V = m_commGraph[id];

  auto type = (ncclDataType_t)getNcclType();
  int rank;
  // CHKNCCL(ncclCommCount(m_comms[i], &nRanks));
  // CHKNCCL(ncclCommCuDevice(m_comms[i], &dev));
  ncclCommUserRank(info.comm, &rank);
  CHKNCCL(ncclGroupStart());
  int sendP = V[0].out, recvP = V[0].in;
  CHKNCCL(ncclSend(info.sendBuf, 
        m_curElems, type, sendP, info.comm, info.stream));
  CHKNCCL(ncclRecv(info.recvBuf, 
        m_curElems, type, recvP, info.comm, info.stream));
  CHKNCCL(ncclGroupEnd());
}
#endif // USE_CUSTOM_QCCL

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

  m_offsets.resize(m_nExtraPeers + 1);
  m_sizes.resize(m_nExtraPeers + 1);
  m_offsets[0] = 0;
  m_sizes[0] = ((size_t)(m_curElems * m_splitFactor) + 3) & ~3;
  // remaining is to be split evenly between m_nExtraPeers
  size_t remaining = m_curElems - m_sizes[0], ofs = m_sizes[0],
      step = m_nExtraPeers > 0 ? (remaining / m_nExtraPeers + 3) & ~3 : 0;
  for(uint32_t i = 1; i <= m_nExtraPeers; i++, ofs += step) {
    m_offsets[i] = ofs;
    m_sizes[i] = step;
  }
  m_sizes[m_nExtraPeers] = m_curElems - m_offsets[m_nExtraPeers];
#if !USE_DEBUG_CONFIG_3_GPUS // this is not relevant for debug config
  if(verifyData) {
    PRINTZ("curElems: %u / 0x%lX", m_curElems, m_curElems);
  }
  for(uint32_t i = 0; i <= m_nExtraPeers && verifyData; i++) {
    PRINTZ("%d: ofs: %lX; size: %lX; sum: %lX elems", 
          i, m_offsets[i], m_sizes[i], m_offsets[i] + m_sizes[i]);
  }
#endif
  m_pool.runJob([&,this](int id) {
    run_thread(id, numIters, verifyData);
  });
}

void TestFramework::run_thread(int id, int numIters, bool verifyData) 
{
  m_barrier.wait(); // wait all threads to arrive here before starting timing
  auto& info = m_infos[id];

  CPU_BEGIN_TIMING(T);
  for(int i = 0; i < numIters; i++) {
    //VLOG("\n============================ " << m_curElems << " =============================\n");
    run_single_gpu(id);
    CHK(cudaStreamSynchronize(info.stream));
    //std::this_thread::sleep_for(std::chrono::milliseconds(100));
  } // for

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
    double baseBw = (double)bytes / 1.0E6 / avgMs;
    PRINTZ("Data size: %.2f Mb; time elapsed: %.3f ms, bandwidth: %.3f Gb/s", 
          (double)bytes/(1024*1024), avgMs, baseBw);
  }
  if(verifyData) {
    verify(id);
  }
}

void runRCCLTest(size_t elemsMin, size_t elemsMax)
{
  int nGpus = 0, nwarmups = 5, niters = 10;
  CHK(hipGetDeviceCount(&nGpus));
  nGpus = 3;
#if USE_DEBUG_CONFIG_3_GPUS  
  if(nGpus != 3) throw std::runtime_error("Only 3 GPUs in debug variant!!");
#endif  
  VLOG("Num devices: " << nGpus << "; max data size: " << 
      (double)(elemsMax*sizeof(TestFramework::T))/(1024*1024) << 
        " Mb; neighbour exchange with "
#if USE_CUSTOM_QCCL
      "MINI QCCL"
#else
      "RCCL"
#endif
  );
  std::vector< uint32_t > deviceAssignment{ 2, 3, 4 };
  if(nGpus > deviceAssignment.size()) {
    throw std::runtime_error("Invalid device assignment!");
  }

  TestFramework obj(nGpus, deviceAssignment.data(), elemsMax);
#if VERIFY_DATA
  obj.run(elemsMin, 1, false, true); // first run to verify data
#endif
  return;

  obj.run(elemsMax, (nwarmups+1)/2);
  obj.run(elemsMin, nwarmups/2);

#if 0
  {
    void *gpuMem;
    std::vector< uint8_t > zz(16);
  RocProfilerSession sess;
  sess.start();
//    obj.run(elemsMin, 1);
  sess.stop();
  (void)hipFree(gpuMem);
  }
#endif

  for(size_t sz = elemsMin; sz <= elemsMax; ) {
    obj.run(sz, niters, true);
    if(sz == elemsMax)
      break;
    sz = std::min(sz * 3 / 2, elemsMax);
  }
}
// NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL

int main() try 
{
  DeviceInit(0);
  //size_t sMin = 2322432, sMax = 9289728*8;
  size_t sMin = 1011*111, sMax = sMin;
  runRCCLTest(sMin, sMax);
}
catch(std::exception& ex) {
  VLOG("Exception: " << ex.what());
}
catch(...) {
  VLOG("Unknown exception");
}
