
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
  size_t nGpus, bufSize;
  std::vector< int > gpus;
  std::vector< hipStream_t > streams;
  std::vector< T *> sendBufs, recvBufs;
  std::vector <ncclComm_t > comms;
  std::vector< T > hostBuf;
  std::vector< std::thread > threads;

public:
  GpuComm(size_t nGpus_, size_t bufSize_) : nGpus(nGpus_), bufSize(bufSize_),
      gpus(nGpus_), streams(nGpus_), sendBufs(nGpus_), recvBufs(nGpus_),
      comms(nGpus_), hostBuf(nGpus_ * bufSize_)
  { 
    NCCLCHECK(ncclGetUniqueId(&ncclId));

    for (size_t i = 0; i < nGpus; i++) 
    {
      gpus[i] = i;
      CHK(cudaSetDevice(gpus[i]));

      int flags = //hipDeviceMallocDefault;
                  //hipDeviceMallocFinegrained;
                   hipDeviceMallocUncached;
                  // hipMallocSignalMemory;
      CHK(hipExtMallocWithFlags((void **)(sendBufs.data() + i), bufSize*sizeof(T), flags));
      CHK(hipExtMallocWithFlags((void **)(recvBufs.data() + i), bufSize*sizeof(T), flags));
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

  void initBuf(T *buf, const T& val) {
    std::fill(buf, buf + bufSize, val);
  }

  void init() {
    for (int i = 0; i < nGpus; i++) 
    {
      gpus[i] = i;
      CHK(cudaSetDevice(gpus[i]));
      CHK(cudaMemset(recvBufs[i], 0, bufSize));
      initBuf(sendBufs[i], T(i));
#if USE_MEMCPY_PEER
      int peer = (i + 1) % nGpus, enable = 0;
      CHK(cudaDeviceCanAccessPeer(&enable, i, peer));
      if(enable == 0) {
        CHK(cudaDeviceEnablePeerAccess(peer, 0));
      }
      VLOG("Can access peer: " << enable);
#endif
    }
    CHK(cudaDeviceSynchronize());
  }

  void verify(int i) {
    int dev;
    auto dst = hostBuf.data() + i * bufSize;
    NCCLCHECK(ncclCommCuDevice(comms[i], &dev));
    CHK(cudaSetDevice(dev));
    CHK(cudaMemcpy(dst, recvBufs[i], bufSize*sizeof(T), cudaMemcpyDeviceToHost));
    VLOG("Device " << dev << " verify: element: " << dst[0]);
  }

  void run_single_gpu_memcpy(int i) {

    //NCCLCHECK(ncclCommCount(comms[i], &nRanks));
    int dev = i;
    CHK(cudaSetDevice(dev));
    int sendPeer = (i + 1) % nGpus;
    CHK(cudaMemcpyPeerAsync(recvBufs[sendPeer],
        sendPeer, sendBufs[i], i, bufSize*sizeof(T), streams[i]));
  }

  void run_single_gpu(int i) {

    int count = bufSize;
    auto type = getNcclType();
    int nRanks, rank, dev;

    NCCLCHECK(ncclCommCount(comms[i], &nRanks));
    NCCLCHECK(ncclCommCuDevice(comms[i], &dev));
    CHK(cudaSetDevice(dev));

    ncclCommUserRank(comms[i], &rank);
    int recvPeer = (rank-1+nRanks) % nRanks;
    int sendPeer = (rank+1) % nRanks;

    // VLOG(std::hex << std::this_thread::get_id() << std::dec << 
    //       ": Running thread with rank: " << rank << " peers: " 
    //       << recvPeer << " " << sendPeer);

    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclSend(sendBufs[i], count, type, sendPeer, comms[i], streams[i]));
    NCCLCHECK(ncclRecv(recvBufs[i], count, type, recvPeer, comms[i], streams[i]));
    NCCLCHECK(ncclGroupEnd());
  }

  void run(bool verifyData = false) 
  {
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
void runRCCLTest(size_t len)
{
  int nGpus = 0, nwarmups = 5, niters = 10;
  size_t nbytes = len * sizeof(T);
  CHK(hipGetDeviceCount(&nGpus));
  VLOG("Num devices: " << nGpus << ", data to be sent: " << (double)nbytes/(1024*1024) << " Mb");

  GpuComm<T> obj(nGpus, len);
  obj.init();

  for(int i = 0; i <nwarmups; i++) { // warm-up
    obj.run(true);
  }

  CPU_BEGIN_TIMING(sendrecv);
  for(int i = 0; i < niters; i++) { 
    obj.run(false);
  }
  auto tnow = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> ms = tnow - z1_sendrecv;
  double avgMs = ms.count() / niters;
  double baseBw = (double)nbytes / 1.0E6 / avgMs;

  PRINTZ("Time elapsed: %.3f ms, bandwidth: %.3f Gb/s", avgMs, baseBw);

  
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
 
  // GPUStream s[5];
  // T *devSrc = nullptr;
  // uint32_t bytes = 181403648;
  // CHK(cudaMalloc((void**)&devSrc, bytes))

  // std::vector< T > hosts[5];
  // for(auto& H : hosts) {
  //   H.resize(N);
  // }

  // for(int i = 0; i < 1000; i++) {
  
  // auto& hostD = hosts[i % 5];
  // for(uint32_t i = 0; i < N; i++) {
  //   hostD[i] = i+1;
  // }

  // VLOG("memcpy " << hostD.data() << " -> " << devSrc);
  // CHK(hipMemcpyHtoDAsync(
  //       devSrc, hostD.data(), N * sizeof(T), s[0].get()))
  // //CHK(hipStreamSynchronize(s1.get()))


  // HVector< T > zz(16);
  // ReduceSumKernel<<<1, 128, 0, s[0].get()>>>(devSrc, N, zz.devPtr);
  // //CHK(hipStreamSynchronize(s2.get()));    

  // zz.copyDToH();
  // for(int i = 0; i < 1; i++) {
  //    VLOG(i << ": reduce result " << zz[i]);
  // }
  // } // for

  // CHK(cudaFree(devSrc))


}

// NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL

int main() try 
{
   runRCCLTest<float>(16*1024*1024);
}
catch(std::exception& ex) {
  VLOG("Exception: " << ex.what());
}
