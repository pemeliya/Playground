
// hipcc -I.. -DCOMPILE_FOR_ROCM=1 -std=c++17 --offload-arch=gfx90a test_main.cc

#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <thread>
#include "common/threading.hpp"
#include "common/example_utils.hpp"
#include "common/roc_profiler.h"

#define USE_MEMCPY_PEER 0
#define VERIFY_DATA 1

#define gprint(fmt, ...) printf(fmt"\n", ##__VA_ARGS__)

#define CHKNCCL(cmd) \
  if(auto res = (cmd); res != ncclSuccess) {           \
    PRINTZ("Test NCCL failure %s:%d '%s'",              \
        __FILE__,__LINE__,ncclGetErrorString(res));     \
  }

struct P2PWorkItem {
  void **exchangeBuf; // shared buffer for exchanging pointers between GPUs:
                      // this should be set accordingly for each p2p channel (pair of GPUs)
  void *dataBuf;      // send/recv buffer 
  size_t size;        // buffer size in bytes
};

struct WorkInfo
{
  P2PWorkItem recvItem, sendItem;
};

static_assert(sizeof(WorkInfo) % sizeof(uint64_t) == 0, 
    "Size must be aligned by 8 bytes");

__shared__ WorkInfo s_workInfo;

// A -> B, B -> C, C -> A

__device__ void doReceive(uint32_t thid) {

  // we provide the sender our receive buffer
  if(thid == 0) {
    auto& item = s_workInfo.recvItem;
    void *volatile *slot = item.exchangeBuf;
    // Wait for consumer to consume previous value before trampling it.
    while((void *)atomicAdd((uint64_t *)slot, 0) != nullptr);
    // Encode pointer by XOR'ing against some address they definitely wouldn't send
    // since we want to allow them sending us nullptr while not colliding with
    // the empty slot value.
    *slot = (void *)(reinterpret_cast<uintptr_t>(item.dataBuf) ^ 
                     reinterpret_cast<uintptr_t>(slot));
    gprint("Sent target buffer pointer: %p to the receiver", item.dataBuf);
  }
}

__device__ void doSend(uint32_t thid) {

  if(thid == 0) {
    auto& item = s_workInfo.sendItem;
    void *volatile *slot = item.exchangeBuf;
    void *ptr;
    while (true) {
      ptr = (void *)atomicAdd((uint64_t *)slot, 0);
      if (ptr != nullptr) break;    
    }
    auto targetBuf = (void *)(reinterpret_cast<uintptr_t>(ptr) ^ 
                              reinterpret_cast<uintptr_t>(slot));
    *slot = nullptr;
    gprint("Received target buffer pointer: %p", targetBuf);
  }
}

// there maybe many work items: one for each gpu block..
template < uint32_t BlockSz >
__global__ void rcclKernel(WorkInfo *gworkInfo) { 

  constexpr uint32_t s_num = sizeof(WorkInfo) / sizeof(uint64_t),
            warpSize = 64;

  uint32_t thid = threadIdx.x, groupID = thid / warpSize;
  if(thid < s_num) {
    auto d = ((uint64_t *)gworkInfo)[thid];
    ((uint64_t *)&s_workInfo)[thid] = d;
  }
  __syncthreads();

  // we will use directWrite: that is, each sender writes data to receiver buffer directly
  // for that, receiver should provide sender the buffer address
  if(groupID < 2) {
    doReceive(thid);
  } else {
    doSend(thid);
  }
}

template < class T >
struct SendRecvItem {

  int gpuId = -1;
  size_t maxElems = 0;
  T *sendBuf = nullptr, *recvBuf = nullptr; // send and receive buffers
  cudaStream_t stream;  // associated stream

  SendRecvItem() = default;
  SendRecvItem(SendRecvItem&) = delete;
  SendRecvItem& operator =(SendRecvItem&) = delete;

  void init(int _gpuId, size_t _maxElems)
  {
    gpuId = _gpuId, maxElems = _maxElems; 
    size_t nBytes = maxElems * sizeof(T);
    CHK(cudaSetDevice(gpuId));
    int flags = hipDeviceMallocDefault;
                //hipDeviceMallocFinegrained;
                // hipDeviceMallocUncached;
                // hipMallocSignalMemory;
    CHK(hipExtMallocWithFlags((void **)&sendBuf, nBytes*2, flags));
    recvBuf = sendBuf + maxElems;
    CHK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }

  ~SendRecvItem() {
    if(gpuId >= 0) {
      (void)cudaSetDevice(gpuId);
      (void)cudaStreamDestroy(stream);
      (void)cudaFree(sendBuf);
    }
  }
};

class GpuCommLib {

  struct ThreadInfo {
    int gpuId;               // gpu ID assigned to this thread
    WorkInfo *workBuf;
    void **exchangeBuf;    // shared buffer for exchanging pointers
  };

  size_t m_nGpus; // total and current data transfer size
  bool m_measureTime = false;
  std::vector< ThreadInfo > m_infos;
  std::mutex m_verifyMtx;
  
public:
  // there should be a common buffer between each pair of GPUs communicated
  GpuCommLib(size_t nGpus) : m_nGpus(nGpus), m_infos(nGpus) 
  {
    int i = 0;
    size_t exchangeSz = m_nGpus * sizeof(void *), 
           bytes = sizeof(WorkInfo) + exchangeSz;
    for(auto& info : m_infos) {
      info.gpuId = i++;
      CHK(cudaSetDevice(info.gpuId));
      int flags = hipDeviceMallocDefault;
                  //hipDeviceMallocFinegrained;
                  // hipDeviceMallocUncached;
                  // hipMallocSignalMemory;
      CHK(hipExtMallocWithFlags((void **)&info.workBuf, bytes, flags));
      info.exchangeBuf = (void **)(info.workBuf + 1);
      CHK(cudaMemset(info.exchangeBuf, 0, exchangeSz));
    }
  }

  // function run on a thread: this ID receives from recvPeer and 
  // sends to sendPeer
  void runSendRecv(uint32_t ID, uint32_t recvPeer, void *recvBuf,
        size_t recvSize, uint32_t sendPeer, void *sendBuf, size_t sendSize, 
        cudaStream_t stream) {
    
    // NOTE: exchange pointers are always allocated on the receiver side!!
    WorkInfo w = {
      .recvItem = { // whom we are receiving from
        // exchange buf on the receiver side
        .exchangeBuf = (void **)m_infos[ID].exchangeBuf[recvPeer],
        .dataBuf = recvBuf,
        .size = recvSize,
      },
      .sendItem = { // whom we are sending to
        // exchange buf on the receiver side
        .exchangeBuf = (void **)m_infos[sendPeer].exchangeBuf[ID], 
        .dataBuf = sendBuf,
        .size = sendSize,
      },
    };
    CHK(cudaSetDevice(m_infos[ID].gpuId));
    CHK(cudaMemcpyAsync(m_infos[ID].workBuf, &w, sizeof(WorkInfo), 
                                               cudaMemcpyHostToDevice, stream));
    constexpr uint32_t BlockSz = 256;
    rcclKernel<BlockSz><<<1, BlockSz, 0, stream>>>(m_infos[ID].workBuf);
  }

  ~GpuCommLib() {
    for(auto& info : m_infos) {
      (void)cudaSetDevice(info.gpuId);
      (void)cudaFree(info.workBuf);
    }
  }

  void run() {
    
    for(const auto& info : m_infos) {
      
      CHK(cudaSetDevice(info.gpuId));
      for(uint32_t j = 0; j < m_nGpus; j++) {
        if(info.gpuId == j)
          continue;
        int enable;
        CHK(cudaDeviceCanAccessPeer(&enable, info.gpuId, j));
        if(enable == 0) {
          CHK(cudaDeviceEnablePeerAccess(j, 0));
        }
      }
    } // for info
    // one work item for 
  }
}; // GpuCommLib

//Matrix< WorkInfo > p2pSend, p2pRecv;
// ncclSend(a,b) -> set p2pSend(a,b) = sendBuf, sendSize (a sends to b) => a needs to get recv buffer from b
// ncclRecv(b,a) -> set p2pRecv(b,a) = recvBuf, recvSize (b receives from a) => b needs give its recv buffer to a

// each thread for one GPU executes ncclRecv and ncclSend which creates
// one kernel with one channel (with correct buffer exchange pointer)
// that is ncclRecv (exchange buf to be shared between GPUs a and b)

// __global__ void atomicKernel(float *ptr) { 

//   int tid = threadIdx.x, bid = blockIdx.x;
//   auto val = atomicAdd(ptr, 1.0f);
//   if(val >= 512*512-10) {
//     printf("bid: %d, tid: %d, val: %f\n", bid, tid, val);
//   }
// }

template < class T >
void runRCCLTest()
{
  int nGpus = 0, elems = 10000;
  CHK(cudaGetDeviceCount(&nGpus));
  VLOG("Num devices: " << nGpus);
  nGpus = 2;
  GpuCommLib obj(nGpus);
  ThreadPool pool(nGpus);

  // float *devPtr;
  // (void)hipMalloc((void**)&devPtr, 16);
  // (void)hipMemset(devPtr, 0, 16);
  // atomicKernel<<<512, 512, 0>>>(devPtr);
  // (void)hipFree(devPtr);

  std::vector< SendRecvItem<T> > items(nGpus);
  pool.runJob([&](int id) {
    auto& item = items[id];
    int sendPeer = (id + 1)%nGpus, 
        recvPeer = (id - 1 + nGpus)%nGpus;
    PRINTZ("GPU %d recv from %d and sends to %d", id, recvPeer, sendPeer);
    item.init(id, elems);
  });
}

int main() try 
{
  DeviceInit(0);
  runRCCLTest<float>();
}
catch(std::exception& ex) {
  VLOG("Exception: " << ex.what());
}
catch(...) {
  VLOG("Unknown exception");
}
