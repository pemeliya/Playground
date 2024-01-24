
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
class GpuComm {

  struct ThreadInfo {
    int gpuId;            // gpu ID assigned to this thread
    cudaStream_t stream;  // associated stream
    T *sendBuf, *recvBuf; // send and receive buffers
#if !USE_MEMCPY_PEER
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

  double m_splitFactor = 0.33; // this much of traffic is sent to target GPUs directly
  uint32_t m_numExtraPeers = 3; // if zero, all traffic is sent directly

  std::vector< size_t > m_offsets, m_sizes;

public:
  GpuComm(size_t nGpus, size_t maxElems) : m_nGpus(nGpus), m_maxElems(maxElems),
      m_curElems(maxElems), m_infos(nGpus), m_barrier(nGpus), m_pool(nGpus)
  { }

  ~GpuComm() {
    for(auto& info : m_infos) {
      (void)cudaSetDevice(info.gpuId);
      (void)cudaStreamDestroy(info.stream);
      (void)cudaFree(info.sendBuf);
#if !USE_MEMCPY_PEER
      (void)ncclCommDestroy(info.comm);
#endif
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
    //return static_cast<T>(100.0f*std::sin((float)idx*2*M_PI/(device+2)));
    return static_cast<T>(device);
  }

  int sendPeer(int i) {
    return (i + 1)%m_nGpus;
  }
  int recvPeer(int i) {
    return (i - 1 + m_nGpus)%m_nGpus;
  }

  void init() 
  {
#if !USE_MEMCPY_PEER
    NCCLCHECK(ncclGetUniqueId(&m_ncclId));
#endif
    m_pool.runJob([this](int id) {
      auto& info = m_infos[id];
      size_t nBytes = m_maxElems*sizeof(T);
      {
        std::lock_guard _(m_verifyMtx);
        VLOG("Allocating buffers and data init for GPU " << id);
      }
      info.gpuId = id;
      CHK(cudaSetDevice(info.gpuId));
      int flags = hipDeviceMallocDefault;
                  //hipDeviceMallocFinegrained;
                  // hipDeviceMallocUncached;
                  // hipMallocSignalMemory;
      CHK(hipExtMallocWithFlags((void **)&info.sendBuf, nBytes*2, flags));
      info.recvBuf = info.sendBuf + m_maxElems;

      CHK(cudaStreamCreateWithFlags(&info.stream, cudaStreamNonBlocking));
	//       if (cumask[0] || cumask[1] || cumask[2] || cumask[3]) {
	//          PRINT("cumask: ");
	//          for (int i = 0; i < 4 ; i++) PRINT("%x,", cumask[i]);
	//          PRINT("\n");
	//          HIPCHECK(hipExtStreamCreateWithCUMask(m_streams+i, 4, cumask));
      CHK(cudaMemset(info.recvBuf, 0, nBytes));
      std::vector< T > refBuf(m_maxElems);
#if VERIFY_DATA
      for(size_t i = 0; i < m_maxElems; i++) {
        refBuf[i] = getElement(info.gpuId, i);
      }
#else
      std::fill(refBuf.begin(), refBuf.end(), T(id));
#endif
      CHK(cudaMemcpyAsync(info.sendBuf, refBuf.data(), nBytes, cudaMemcpyHostToDevice,
            info.stream));
#if USE_MEMCPY_PEER
      int peer = sendPeer(id), enable = 0;
      CHK(cudaDeviceCanAccessPeer(&enable, id, peer));
      if(enable == 0) {
        CHK(cudaDeviceEnablePeerAccess(peer, 0));
      }
#else
      NCCLCHECK(ncclCommInitRank(&info.comm, m_nGpus, m_ncclId, id));
#endif
    }); // runJob
    CHK(cudaDeviceSynchronize());
  }

  void verify(int id) {
    std::lock_guard _(m_verifyMtx);
    if(m_hostBuf.size() < m_curElems) {
      m_hostBuf.resize(m_curElems);
    }
    auto dst = m_hostBuf.data();
    CHK(cudaMemcpy(dst, m_infos[id].recvBuf, m_curElems*sizeof(T), cudaMemcpyDeviceToHost));
    VLOG("Device " << id << " verifying");
    for(uint32_t j = 0, num = 0; j < m_curElems; j++) {
      auto truth = getElement(recvPeer(id), j);
      if(dst[j] != truth) {
        //ThrowError< 256 >("%d: verify failed truth: %f gpu: %f", j, truth, dst[j]);
        PRINTZ("%X: verify failed truth: %f gpu: %f", j, truth, dst[j]);
        if(num++ >= 5)
          break;
      }
    }
  }

  void run_single_gpu(int id, int step) 
  {
    auto& info = m_infos[id];
#if USE_MEMCPY_PEER
    int peer = sendPeer(id);
    CHK(cudaMemcpyPeerAsync(m_recvBufs[peer],
        peer, m_sendBufs[id], id, m_curElems*sizeof(T), m_streams[id]));
#else
    auto type = getNcclType();
    int rank;
    // NCCLCHECK(ncclCommCount(m_comms[i], &nRanks));
    // NCCLCHECK(ncclCommCuDevice(m_comms[i], &dev));
    ncclCommUserRank(info.comm, &rank);
    int recvP = recvPeer(rank), sendP = sendPeer(rank);
    // VLOG(std::hex << std::this_thread::get_id() << std::dec << 
    //       ": Running thread with rank: " << rank << " peers: " 
    //       << recvP << " " << sendP);

    NCCLCHECK(ncclGroupStart());
    if(step == 0) {
      
      // NCCLCHECK(ncclSend(m_sendBufs[i], m_curElems, type, sendP, m_comms[i], m_streams[i]));
      // NCCLCHECK(ncclRecv(m_recvBufs[i], m_curElems, type, recvP, m_comms[i], m_streams[i]));
      for(int i = 0; i <= m_numExtraPeers; i++) {
        sendP = (rank + i + 1) % m_nGpus;
        NCCLCHECK(ncclSend(info.sendBuf + m_offsets[i], 
              m_sizes[i], type, sendP, info.comm, info.stream));

        recvP = (rank - 1 - i + m_nGpus) % m_nGpus;
        NCCLCHECK(ncclRecv(info.recvBuf + m_offsets[i], 
              m_sizes[i], type, recvP, info.comm, info.stream));
//        VLOG(id << "  " << sendP);
      }
    } else {
      // std::lock_guard _(m_verifyMtx);
      // NOTE: you screw up send buffer here => hence verify iter must be the 1st one !!!
      for(int i = 1; i <= m_numExtraPeers; i++) {
        CHK(cudaMemcpyAsync(info.sendBuf + m_offsets[i], info.recvBuf + m_offsets[i],
            m_sizes[i]*sizeof(T), cudaMemcpyDeviceToDevice, info.stream));

        sendP = (rank - i + m_nGpus) % m_nGpus;
        recvP = (rank + i) % m_nGpus;
        // VLOG(id << " gpu sends back to " << sendP << " and recv from " << recvP);
        // but you have to swap recv and send bufs here !!!
        NCCLCHECK(ncclSend(info.sendBuf + m_offsets[i], 
               m_sizes[i], type, sendP, info.comm, info.stream));

        NCCLCHECK(ncclRecv(info.recvBuf + m_offsets[i], 
               m_sizes[i], type, recvP, info.comm, info.stream));
      }
    }
    NCCLCHECK(ncclGroupEnd());
#endif // USE_MEMCPY_PEER
  }

  void run(size_t numElems, int numIters, bool measureTime = false, bool verifyData = false) {

    m_measureTime = measureTime;
    m_curElems = numElems;
    if(m_curElems > m_maxElems) {
       ThrowError< 256 >("numElems must be <= m_maxElems");
    }

    m_offsets.resize(m_numExtraPeers + 1);
    m_sizes.resize(m_numExtraPeers + 1);
    m_offsets[0] = 0;
    m_sizes[0] = ((size_t)(m_curElems * m_splitFactor) + 3) & ~3;
    // remaining is to be split evenly between m_numExtraPeers
    size_t remaining = m_curElems - m_sizes[0], 
            step = (remaining / m_numExtraPeers + 3) & ~3, ofs = m_sizes[0];
    for(uint32_t i = 1; i <= m_numExtraPeers; i++, ofs += step) {
      m_offsets[i] = ofs;
      m_sizes[i] = step;
    }
    m_sizes[m_numExtraPeers] = m_curElems - m_offsets[m_numExtraPeers];
    for(uint32_t i = 0; i <= m_numExtraPeers; i++) {
      PRINTZ("%d: ofs: %lX; size: %lX; sum: %lX", 
      i, m_offsets[i], m_sizes[i], m_offsets[i] + m_sizes[i]);
    }
    
    m_pool.runJob([&,this](int id) {
      run_thread(id, numIters, verifyData);
    });
  }

  void run_thread(int id, int numIters, bool verifyData) 
  {
    m_barrier.wait(); // wait all threads to arrive here before starting timing
    auto& info = m_infos[id];

    CPU_BEGIN_TIMING(T);
    for(int i = 0; i < numIters; i++) {
      run_single_gpu(id, 0);
      CHK(cudaStreamSynchronize(info.stream));
    //m_barrier.wait(); // wait all threads to arrive here before starting timing
      run_single_gpu(id, 1);
      CHK(cudaStreamSynchronize(info.stream));
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
#if VERIFY_DATA
   obj.run(elemsMin, 1, false, true); // last run to verify data
#endif

  obj.run(elemsMax, (nwarmups+1)/2);
  // obj.run(elemsMin, nwarmups/2);

  for(size_t sz = elemsMin; sz <= elemsMax; ) {
    obj.run(sz, niters, true);
    if(sz == elemsMax)
      break;
    sz = std::min(sz * 3 / 2, elemsMax);
  }
  
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
  size_t sMin = 32*1024*1024, sMax = 32*1024*1024;
  runRCCLTest<float>(sMin, sMax);
}
catch(std::exception& ex) {
  VLOG("Exception: " << ex.what());
}
