
// hipcc -I.. -DCOMPILE_FOR_ROCM=1 -std=c++17 --offload-arch=gfx90a test_main.cc

#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include "common/threading.hpp"
#include "common/example_utils.hpp"

#define USE_MEMCPY_PEER 0
#define VERIFY_DATA 1

#define NCCLCHECK(cmd) \
  if(auto res = (cmd); res != ncclSuccess) {           \
    PRINTZ("Test NCCL failure %s:%d '%s'",              \
        __FILE__,__LINE__,ncclGetErrorString(res));     \
  }

template < class T >
struct GpuComm {

  ncclUniqueId m_ncclId;
  size_t m_nGpus, m_maxSize, m_curSize; // total and current data transfer size
  bool m_measureTime = false;
  std::vector< int > m_gpus;
  std::vector< double > m_timings;
  std::vector< hipStream_t > m_streams;
  std::vector< T *> m_sendBufs, m_recvBufs;
  std::vector <ncclComm_t > m_comms;
  std::vector< T > m_hostBuf;
  std::mutex m_verifyMtx;
  Barrier m_barrier;

  ThreadPool m_pool;

public:
  GpuComm(size_t nGpus, size_t maxSize) : m_nGpus(nGpus), m_maxSize(maxSize),
      m_curSize(maxSize), m_gpus(nGpus), m_timings(nGpus),
      m_streams(nGpus), m_sendBufs(nGpus), m_recvBufs(nGpus),
      m_hostBuf(maxSize), m_barrier(nGpus), m_pool(nGpus)
  { }

  ~GpuComm() {
    for (size_t i = 0; i < m_nGpus; i++) {
      (void)cudaSetDevice(m_gpus[i]);
      (void)cudaStreamDestroy(m_streams[i]);
      (void)cudaFree(m_sendBufs[i]);
      (void)cudaFree(m_recvBufs[i]);
    }
    for(auto& comm : m_comms) {
      (void)ncclCommDestroy(comm);
    }
  }

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

  T getElement(int device, size_t idx) {
    return static_cast<T>(100.0f*std::sin((float)idx*2*M_PI/(device+2)));
  }

  int sendPeer(int i) {
    return (i + 1)%m_nGpus;
  }
  int recvPeer(int i) {
    return (i - 1 + m_nGpus)%m_nGpus;
  }

  void init() 
  {
    for(uint32_t i = 0; i < m_nGpus; i++) {
      m_gpus[i] = i;
    }
#if !USE_MEMCPY_PEER
    m_comms.resize(m_nGpus);
    NCCLCHECK(ncclGetUniqueId(&m_ncclId));
    NCCLCHECK(ncclCommInitAll(m_comms.data(), m_nGpus, m_gpus.data()));
#endif

    m_pool.runJob([this](int id) {
      VLOG("Allocating buffers and data init for GPU " << id);
      CHK(cudaSetDevice(m_gpus[id]));
      int flags = hipDeviceMallocDefault;
                  //hipDeviceMallocFinegrained;
                  // hipDeviceMallocUncached;
                  // hipMallocSignalMemory;
      CHK(hipExtMallocWithFlags((void **)(m_sendBufs.data() + id), m_maxSize*sizeof(T), flags));
      CHK(hipExtMallocWithFlags((void **)(m_recvBufs.data() + id), m_maxSize*sizeof(T), flags));
      CHK(cudaStreamCreateWithFlags(m_streams.data() + id, cudaStreamNonBlocking));
	//       if (cumask[0] || cumask[1] || cumask[2] || cumask[3]) {
	//          PRINT("cumask: ");
	//          for (int i = 0; i < 4 ; i++) PRINT("%x,", cumask[i]);
	//          PRINT("\n");
	//          HIPCHECK(hipExtStreamCreateWithCUMask(m_streams+i, 4, cumask));
      CHK(cudaMemset(m_recvBufs[id], 0, m_maxSize*sizeof(T)));
      std::vector< T > refBuf(m_maxSize);
#if VERIFY_DATA
      for(size_t i = 0; i < m_maxSize; i++) {
        refBuf[i] = getElement(m_gpus[id], i);
      }
#else
      std::fill(refBuf.begin(), refBuf.end(), T(id));
#endif
      CHK(cudaMemcpy(m_sendBufs[id], refBuf.data(), m_maxSize*sizeof(T), cudaMemcpyHostToDevice));
#if USE_MEMCPY_PEER
      int peer = sendPeer(id), enable = 0;
      CHK(cudaDeviceCanAccessPeer(&enable, id, peer));
      if(enable == 0) {
        CHK(cudaDeviceEnablePeerAccess(peer, 0));
      }
#endif
    }); // runJob
    CHK(cudaDeviceSynchronize());
  }

  void verify(int id) {
    std::lock_guard _(m_verifyMtx);
    auto dst = m_hostBuf.data();
    CHK(cudaMemcpy(dst, m_recvBufs[id], m_curSize*sizeof(T), cudaMemcpyDeviceToHost));
    VLOG("Device " << id << " verifying");
    for(size_t j = 0; j < m_curSize; j++) {
      auto truth = getElement(recvPeer(id), j);
      if(dst[j] != truth) {
        ThrowError< 256 >("%d: verify failed truth: %f gpu: %f", j, truth, dst[j]);
      }
    }
  }

  void run_single_gpu(int i) {
#if USE_MEMCPY_PEER
    int peer = sendPeer(i);
    CHK(cudaMemcpyPeerAsync(m_recvBufs[peer],
        peer, m_sendBufs[i], i, m_curSize*sizeof(T), m_streams[i]));
#else
    auto type = getNcclType();
    int rank;
    // NCCLCHECK(ncclCommCount(m_comms[i], &nRanks));
    // NCCLCHECK(ncclCommCuDevice(m_comms[i], &dev));
    ncclCommUserRank(m_comms[i], &rank);
    int recvP = recvPeer(rank), sendP = sendPeer(rank);
    // VLOG(std::hex << std::this_thread::get_id() << std::dec << 
    //       ": Running thread with rank: " << rank << " peers: " 
    //       << recvP << " " << sendP);

    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclSend(m_sendBufs[i], m_curSize, type, sendP, m_comms[i], m_streams[i]));
    NCCLCHECK(ncclRecv(m_recvBufs[i], m_curSize, type, recvP, m_comms[i], m_streams[i]));
    NCCLCHECK(ncclGroupEnd());
#endif // USE_MEMCPY_PEER
  }

  void run(size_t numElems, int numIters, bool measureTime = false, bool verifyData = false) {

    m_measureTime = measureTime;
    m_curSize = numElems;
    if(m_curSize > m_maxSize) {
       ThrowError< 256 >("numElems must be <= m_maxSize");
    }
    m_pool.runJob([&,this](int id) {
      run_thread(id, numIters, verifyData);
    });
  }

  void run_thread(int id, int numIters, bool verifyData) 
  {
    m_barrier.wait(); // wait all threads to arrive here before starting timing

    CPU_BEGIN_TIMING(T);
    for(int i = 0; i < numIters; i++) {
      run_single_gpu(id);
      CHK(cudaStreamSynchronize(m_streams[id]));
    } // for

    auto tnow = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = tnow - z1_T;
    size_t bytes = m_curSize*sizeof(T);
    m_timings[id] = ms.count() / numIters;

    m_barrier.wait(); // wait before data verification since it requires all GPUs
    if(id == 0 && m_measureTime) {
      auto avgMs = std::accumulate(begin(m_timings), end(m_timings), 0.0) / m_nGpus;
      double baseBw = (double)bytes / 1.0E6 / avgMs;
      PRINTZ("Data size: %.2f Mb; time elapsed: %.3f ms, bandwidth: %.3f Gb/s", 
            (double)bytes/(1024*1024), avgMs, baseBw);
    }
    if(verifyData) {
      verify(id);
    }
  }
}; // struct GpuComm

template < class T >
void runRCCLTest(size_t elemsMin, size_t elemsMax)
{
  int m_nGpus = 0, nwarmups = 5, niters = 10;
  CHK(hipGetDeviceCount(&m_nGpus));
  VLOG("Num devices: " << m_nGpus << "; max data size: " << (double)(elemsMax*sizeof(T))/(1024*1024) << 
        " Mb; neighbour exchange with "
#if USE_MEMCPY_PEER
      "hipMemcpyPeerAsync"
#else
      "RCCL"
#endif
      );

  GpuComm<T> obj(m_nGpus, elemsMax);
  obj.init();

  obj.run(elemsMax, (nwarmups+1)/2);
  obj.run(elemsMin, nwarmups/2);

  for(size_t sz = elemsMin; sz <= elemsMax; ) {
    obj.run(sz, niters, true);
    if(sz == elemsMax)
      break;
    sz = std::min(sz * 3 / 2, elemsMax);
  }

#if VERIFY_DATA
  obj.run(elemsMin, 1, false, true); // last run to verify data
#endif
  
//      } else {
//        NCCLCHECK(ncclGroupStart());
//        for (int ii=0; ii<m_nGpus*nThreads; ii++) {
//          HIPCHECK(hipSetDevice(m_gpus[ii]));
// 	 if (!enable_multiranks) {
// 	   NCCLCHECK(ncclCommInitRank(m_comms+ii, ncclProcs*nThreads*m_nGpus, ncclId, proc*nThreads*m_nGpus+ii));
// 	 }
}
// NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL

int main() try 
{
  DeviceInit(0);
  size_t sMin = 1024*1024, sMax = 32*1024*1024;
  runRCCLTest<float>(sMin, sMax);
}
catch(std::exception& ex) {
  VLOG("Exception: " << ex.what());
}
