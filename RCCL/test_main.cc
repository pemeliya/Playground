
// hipcc -I.. -DCOMPILE_FOR_ROCM=1 -std=c++17 --offload-arch=gfx90a test_main.cc

#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include "rccl/rccl.h"
#include "common/example_utils.hpp"

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    fprintf(stderr, "Test NCCL failure %s:%d '%s'\n",    \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
     \
  }                                                 \
} while(0)

struct GpuComm {
  explicit GpuComm(size_t nGpus_, size_t bufSize_) : nGpus(nGpus_), bufSize(bufSize_),
      gpus(nGpus_), streams(nGpus_), sendBufs(nGpus_), recvBufs(nGpus_),
      comms(nGpus_), hostBuf(nGpus_ * bufSize_)
  { 
    NCCLCHECK(ncclGetUniqueId(&ncclId));


    for (size_t i = 0; i < nGpus; i++) 
    {
      gpus[i] = i;
      CHK(cudaSetDevice(gpus[i]));
      //  AllocateBuffs(sendBufs+i, sendBytes, recvBufs+i, recvBytes, expected+i, (size_t)maxBytes);
      CHK(cudaMalloc(sendBufs.data() + i, bufSize));
      CHK(cudaMalloc(recvBufs.data() + i, bufSize));
      CHK(hipStreamCreateWithFlags(streams.data() + i, hipStreamNonBlocking));
  
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

  template < class T >
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

  template < class T >
  void initBuf(T *buf, const T& val) {
    std::fill(buf, buf + bufSize / sizeof(T), val);
  }

  template < class T >
  void init() {
    for (int i = 0; i < nGpus; i++) 
    {
      gpus[i] = i;
      CHK(cudaSetDevice(gpus[i]));
      CHK(cudaMemset(recvBufs[i], 0, bufSize));
      initBuf((T *)sendBufs[i], T(i));
    }
    CHK(cudaDeviceSynchronize());
  }

  template < class T >
  void verify(int i) {
    int dev;
    auto dst = (T *)(hostBuf.data() + i * bufSize);
    NCCLCHECK(ncclCommCuDevice(comms[i], &dev));
    CHK(cudaSetDevice(dev));
    CHK(cudaMemcpy(dst, recvBufs[i], bufSize, cudaMemcpyDeviceToHost));
    VLOG(dev << ": element: " << dst[0]);
  }

  template < class T >
  void run() 
  {
    init< T >();
    int count = bufSize / sizeof(T);
    auto type = getNcclType<T>();

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGpus; i++) {
      int nRanks, rank, dev;
      NCCLCHECK(ncclCommCount(comms[i], &nRanks));
      NCCLCHECK(ncclCommCuDevice(comms[i], &dev));
      CHK(cudaSetDevice(dev));

      ncclCommUserRank(comms[i], &rank);
      int recvPeer = (rank-1+nRanks) % nRanks;
      int sendPeer = (rank+1) % nRanks;

      VLOG(std::hex << (uint64_t)pthread_self() << std::dec << ": Running thread with rank: " << rank << " peers: " 
        << recvPeer << " " << sendPeer);

      NCCLCHECK(ncclSend(sendBufs[i], count, type, sendPeer, comms[i], streams[i]));
      NCCLCHECK(ncclRecv(recvBufs[i], count, type, recvPeer, comms[i], streams[i]));
      // int dev;
      // ncclCommCuDevice(comms[i], &dev);
    }
    NCCLCHECK(ncclGroupEnd());

    CHK(cudaDeviceSynchronize());
    for (int i = 0; i < nGpus; i++) {
      verify<T>(i);
    }
  }

  ncclUniqueId ncclId;
  size_t nGpus, bufSize;
  std::vector< int > gpus;
  std::vector< hipStream_t > streams;
  std::vector< void*> sendBufs, recvBufs;
  std::vector <ncclComm_t > comms;
  std::vector< uint8_t > hostBuf;
};

template < class T >
void runRCCLTest(size_t len)
{
  int nGpus = 0;
  CHK(hipGetDeviceCount(&nGpus));
  VLOG("Num devices: " << nGpus);

  GpuComm obj(nGpus, len);
  obj.run< float >();

  
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

int main() try 
{
   runRCCLTest<float>(1000000);
}
catch(std::exception& ex) {
  VLOG("Exception: " << ex.what());
}
