
// hipcc -I.. -DCOMPILE_FOR_ROCM=1 -std=c++17 --offload-arch=gfx90a test_main.cc

#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include "common/example_utils.hpp"

#define USE_MEMCPY_PEER 1
#define VERIFY_DATA 0

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    fprintf(stderr, "Test NCCL failure %s:%d '%s'\n",    \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
     \
  }                                                 \
} while(0)

template < class T >
struct GpuComm {

  ncclUniqueId ncclId;
  size_t nGpus, maxSize, curSize; // total and current data transfer size
  std::vector< int > gpus;
  std::vector< hipStream_t > streams;
  std::vector< T *> sendBufs, recvBufs;
  std::vector <ncclComm_t > comms;
  std::vector< T > hostBuf;
  std::vector< std::thread > threads;

public:
  GpuComm(size_t nGpus_, size_t maxSize_) : nGpus(nGpus_), maxSize(maxSize_),
      curSize(maxSize_), gpus(nGpus_), streams(nGpus_), sendBufs(nGpus_), recvBufs(nGpus_),
      comms(nGpus_), hostBuf(maxSize_ * 2) // one for reference solution
  { 
    NCCLCHECK(ncclGetUniqueId(&ncclId));

    for (size_t i = 0; i < nGpus; i++) 
    {
      gpus[i] = i;
      CHK(cudaSetDevice(gpus[i]));

      int flags = hipDeviceMallocDefault;
                  //hipDeviceMallocFinegrained;
                  // hipDeviceMallocUncached;
                  // hipMallocSignalMemory;
      CHK(hipExtMallocWithFlags((void **)(sendBufs.data() + i), maxSize*sizeof(T), flags));
      CHK(hipExtMallocWithFlags((void **)(recvBufs.data() + i), maxSize*sizeof(T), flags));
      CHK(cudaStreamCreateWithFlags(streams.data() + i, cudaStreamNonBlocking));
  
  //     if (streamnull)
  //     	streams[i] = NULL;
  //     else {
	//       if (cumask[0] || cumask[1] || cumask[2] || cumask[3]) {
	//          PRINT("cumask: ");
	//          for (int i = 0; i < 4 ; i++) PRINT("%x,", cumask[i]);
	//          PRINT("\n");
	//          HIPCHECK(hipExtStreamCreateWithCUMask(streams+i, 4, cumask));
	//       } else
	
  //     }
  //   }
   } // for
   NCCLCHECK(ncclCommInitAll(comms.data(), nGpus, gpus.data()));
  }

  ~GpuComm() {
    for (size_t i = 0; i < nGpus; i++) {
      (void)cudaSetDevice(gpus[i]);
      (void)cudaStreamDestroy(streams[i]);
      (void)cudaFree(sendBufs[i]);
      (void)cudaFree(recvBufs[i]);
    }
    for(size_t i = 0; i < nGpus; i++) {
      (void)ncclCommDestroy(comms[i]);
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

  void initBuf(int device, T *gpubuf) {
    auto ref = hostBuf.data();
#if VERIFY_DATA
    for(size_t i = 0; i < maxSize; i++) {
      ref[i] = getElement(device, i);
    }
#else
    std::fill(ref, ref + maxSize, T(device));
#endif
    CHK(cudaMemcpy(gpubuf, ref, maxSize*sizeof(T), cudaMemcpyHostToDevice));
  }

  int sendPeer(int i) {
    return (i + 1)%nGpus;
  }
  int recvPeer(int i) {
    return (i - 1 + nGpus)%nGpus;
  }

  void init() {
    for (int i = 0; i < nGpus; i++) 
    {
      VLOG("Allocating buffers and data init for GPU " << i);
      gpus[i] = i;
      CHK(cudaSetDevice(gpus[i]));
      CHK(cudaMemset(recvBufs[i], 0, maxSize));
      initBuf(gpus[i], sendBufs[i]);
#if USE_MEMCPY_PEER
      int peer = sendPeer(i), enable = 0;
      CHK(cudaDeviceCanAccessPeer(&enable, i, peer));
      if(enable == 0) {
        CHK(cudaDeviceEnablePeerAccess(peer, 0));
      }
#endif
    }
    CHK(cudaDeviceSynchronize());
  }

  void verify(int i) {
    int dev;
    auto dst = hostBuf.data();
    NCCLCHECK(ncclCommCuDevice(comms[i], &dev));
    CHK(cudaSetDevice(dev));
    CHK(cudaMemcpy(dst, recvBufs[i], curSize*sizeof(T), cudaMemcpyDeviceToHost));
    VLOG("Device " << dev << " verifying");
    for(size_t j = 0; j < curSize; j++) {
      auto truth = getElement(recvPeer(dev), j);
      if(dst[j] != truth) {
        //PRINTZ("%d: %f %f", j, truth, dst[j]);
        ThrowError< 256 >("%d: verify failed truth: %f gpu: %f", j, truth, dst[j]);
      }
    }
  }

  void run_single_gpu_memcpy(int i) {

    //NCCLCHECK(ncclCommCount(comms[i], &nRanks));
    int dev = i;
    CHK(cudaSetDevice(dev));
    int peer = sendPeer(i);
    CHK(cudaMemcpyPeerAsync(recvBufs[peer],
        peer, sendBufs[i], i, curSize*sizeof(T), streams[i]));
  }

  void run_single_gpu(int i) {

    auto type = getNcclType();
    int nRanks, rank, dev;

    NCCLCHECK(ncclCommCount(comms[i], &nRanks));
    NCCLCHECK(ncclCommCuDevice(comms[i], &dev));
    CHK(cudaSetDevice(dev));

    ncclCommUserRank(comms[i], &rank);
    int recvP = recvPeer(rank), sendP = sendPeer(rank);
    // VLOG(std::hex << std::this_thread::get_id() << std::dec << 
    //       ": Running thread with rank: " << rank << " peers: " 
    //       << recvP << " " << sendP);

    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclSend(sendBufs[i], curSize, type, sendP, comms[i], streams[i]));
    NCCLCHECK(ncclRecv(recvBufs[i], curSize, type, recvP, comms[i], streams[i]));
    NCCLCHECK(ncclGroupEnd());
  }

  void run(size_t numElems, bool verifyData = false) 
  {
    curSize = numElems;
    if(curSize > maxSize) {
       ThrowError< 256 >("numElems must be <= maxSize");
    }

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGpus; i++) {
#if USE_MEMCPY_PEER
      run_single_gpu_memcpy(i);
#else
      run_single_gpu(i);
#endif
    }
    NCCLCHECK(ncclGroupEnd());

    //CHK(cudaDeviceSynchronize());
    for (int i = 0; i < nGpus; i++) {
      CHK(cudaSetDevice(i));
      CHK(cudaStreamSynchronize(streams[i]));
      if(verifyData) {
        verify(i);
      }
    }
  }
};

template < class T >
void runRCCLTest(size_t elemsMin, size_t elemsMax)
{
  int nGpus = 0, nwarmups = 5, niters = 10;
  CHK(hipGetDeviceCount(&nGpus));
  VLOG("Num devices: " << nGpus << "; max data size: " << (double)(elemsMax*sizeof(T))/(1024*1024) << 
        " Mb; neighbour exchange with "
#if USE_MEMCPY_PEER
      "hipMemcpyPeerAsync"
#else
      "RCCL"
#endif
      );

  GpuComm<T> obj(nGpus, elemsMax);
  obj.init();

  for(int i = 0; i < (nwarmups+1)/2; i++) { // warm-up
    obj.run(elemsMax);
  }
  for(int i = 0; i < nwarmups/2; i++) { // warm-up
    obj.run(elemsMin);
  }

  for(size_t sz = elemsMin; sz <= elemsMax; ) {
    CPU_BEGIN_TIMING(sendrecv);
    for(int i = 0; i < niters; i++) { 
      obj.run(sz);
    }
    auto tnow = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = tnow - z1_sendrecv;
    size_t bytes = sz*sizeof(T);
    double avgMs = ms.count() / niters,
          baseBw = (double)bytes / 1.0E6 / avgMs;
    PRINTZ("Data size: %.2f Mb; time elapsed: %.3f ms, bandwidth: %.3f Gb/s", 
            (double)bytes/(1024*1024), avgMs, baseBw);
    if(sz == elemsMax)
      break;
    sz = std::min(sz * 3 / 2, elemsMax);
  }

#if VERIFY_DATA
  obj.setCurrentSize(elemsMin);
  obj.run(true); // last run to verify data
#endif
  
//      } else {
//        NCCLCHECK(ncclGroupStart());
//        for (int ii=0; ii<nGpus*nThreads; ii++) {
//          HIPCHECK(hipSetDevice(gpus[ii]));
// 	 if (!enable_multiranks) {
// 	   NCCLCHECK(ncclCommInitRank(comms+ii, ncclProcs*nThreads*nGpus, ncclId, proc*nThreads*nGpus+ii));
// 	 }
// #ifdef RCCL_MULTIRANKPERGPU
// 	 else
// 	   for (int j=0; j<ranksPerGpu; j++) {
// 	     int i = ii*ranksPerGpu+j;
// 	     NCCLCHECK(ncclCommInitRankMulti(comms+i, ncclProcs*nThreads*nGpus*ranksPerGpu, ncclId,
// 					     proc*nThreads*nGpus*ranksPerGpu+i, proc*nThreads*nGpus*ranksPerGpu+i));
// 	   }
// #endif
//        }
//        NCCLCHECK(ncclGroupEnd());
//      }
}
// NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL

int main() try 
{
  DeviceInit(0);
  size_t sMin = 1024*1024, sMax = 128*1024*1024;
  runRCCLTest<float>(sMin, sMax);
}
catch(std::exception& ex) {
  VLOG("Exception: " << ex.what());
}
