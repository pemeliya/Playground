
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

// NOTE: if data size is small, maybe makes sense to use just normal load/store?
#define LOAD(addr) __builtin_nontemporal_load(addr)
#define STORE(x, addr) __builtin_nontemporal_store((x), (addr))

struct P2PWorkItem {
  uint32_t peer;      // send/recv peer
  uint32_t size;      // buffer size in bytes (limited by 4GB!)
  void **exchangeBuf; // shared buffer for exchanging pointers between GPUs:
                      // this should be set accordingly for each p2p channel (pair of GPUs)
  void *dataBuf;      // send/recv buffer 
};

struct WorkInfo
{
  P2PWorkItem recvItem, sendItem;
  void *targetBuf;    // target buffer address obtained from the receiver
};

static_assert(sizeof(WorkInfo) % sizeof(uint64_t) == 0, 
    "Size must be aligned by 8 bytes");

template < class T >
struct SendRecvItem {

  int gpuId = -1;
  size_t numElems = 0;
  T *sendBuf = nullptr, *recvBuf = nullptr; // send and receive buffers
  std::vector< T > hostBuf;
  cudaStream_t stream;  // associated stream

  SendRecvItem() = default;
  SendRecvItem(SendRecvItem&) = delete;
  SendRecvItem& operator =(SendRecvItem&) = delete;

  T getElem(int ID, int idx) {
    return (T)ID + 11;
  }

  void init(int _gpuId, size_t _numElems)
  {
    gpuId = _gpuId, numElems = _numElems; 
    size_t nBytes = numElems * sizeof(T);
    hostBuf.resize(numElems);
    CHK(cudaSetDevice(gpuId));
    int flags = hipDeviceMallocDefault;
                //hipDeviceMallocFinegrained;
                // hipDeviceMallocUncached;
                // hipMallocSignalMemory;
    CHK(hipExtMallocWithFlags((void **)&sendBuf, nBytes*2, flags));
    recvBuf = sendBuf + numElems;
    CHK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    for(uint32_t i = 0; i < numElems; i++) {
      hostBuf[i] = getElem(gpuId, i);
    }
    CHK(cudaMemcpyAsync(sendBuf, hostBuf.data(), nBytes, 
                                           cudaMemcpyHostToDevice, stream));
  }

  void verify(uint32_t recvPeer) {
    
    VLOG("Device " << gpuId << " verifying: data from node: " << recvPeer
        << " recvBuf: " << recvBuf);
    for(uint32_t j = 0, num = 0; j < numElems; j++) {
      auto truth = getElem(recvPeer, j);
      if(hostBuf[j] != truth) {
        //ThrowError<>("%d: verify failed truth: %f gpu: %f", j, truth, dst[j]);
        PRINTZ("%X: verify failed truth: %u gpu: %u", j, truth, hostBuf[j]);
        if(num++ >= 5)
          break;
      }
    }
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
      int flags = //hipDeviceMallocDefault;
                  hipDeviceMallocFinegrained;
                  // hipDeviceMallocUncached;
                  // hipMallocSignalMemory;
      CHK(hipExtMallocWithFlags((void **)&info.workBuf, bytes, flags));
      info.exchangeBuf = (void **)(info.workBuf + 1);
      CHK(cudaMemset(info.exchangeBuf, 0, exchangeSz));
    }
    for(const auto& info : m_infos) {
      CHK(cudaSetDevice(info.gpuId));
      for(uint32_t j = 0; j < m_nGpus; j++) {
        if(info.gpuId == j)
          continue;
        int enable = -1;
        CHK(cudaDeviceCanAccessPeer(&enable, info.gpuId, j));
        if(enable == 0) {
          ThrowError<>("GPU %d is unable to access peer %d",
                info.gpuId, j);
        }
        CHK(cudaDeviceEnablePeerAccess(j, 0));
      }
    } // for info
  } // ctor

  ~GpuCommLib() {
    for(auto& info : m_infos) {
      (void)cudaSetDevice(info.gpuId);
      (void)cudaFree(info.workBuf);
    }
  }

  // function run on a thread: this ID receives from recvPeer and 
  // sends to sendPeer
  void runSendRecv(uint32_t ID, uint32_t recvPeer, void *recvBuf,
        size_t recvSize, uint32_t sendPeer, void *sendBuf, size_t sendSize, 
        cudaStream_t stream);
}; // GpuCommLib

//Matrix< WorkInfo > p2pSend, p2pRecv;
// ncclSend(a,b) -> set p2pSend(a,b) = sendBuf, sendSize (a sends to b) => a needs to get recv buffer from b
// ncclRecv(b,a) -> set p2pRecv(b,a) = recvBuf, recvSize (b receives from a) => b needs give its recv buffer to a

__shared__ WorkInfo s_workInfo;

struct CollectiveOps {
  uint32_t ltid; // local thread ID within a send/recv group

};

// ltid is a local tid within group !!!
__device__ void doReceive(uint32_t ltid) {

  // we provide the sender our receive buffer
  if(ltid == 0) {
    auto& item = s_workInfo.recvItem;
    void *volatile *slot = item.exchangeBuf;

    // Wait for consumer to consume previous value before trampling it.
    while((void *)atomicAdd((uint64_t *)slot, 0) != nullptr);
    // Encode pointer by XOR'ing against some address they definitely wouldn't send
    // since we want to allow them sending us nullptr while not colliding with
    // the empty slot value.
    *slot = (void *)(reinterpret_cast<uintptr_t>(item.dataBuf) ^ 
                     reinterpret_cast<uintptr_t>(slot));
    gprint("%p Sent target buffer: %p to the sender peer %d", 
          slot, item.dataBuf, item.peer);
  }
}

__device__ void doSend(uint32_t ltid) {

  auto& item = s_workInfo.sendItem;
  if(ltid == 0) {
    void *volatile *slot = item.exchangeBuf;
    void *ptr;
    while (true) {
      ptr = (void *)atomicAdd((uint64_t *)slot, 0);
      if (ptr != nullptr) break;    
    }
    s_workInfo.targetBuf = (void *)(reinterpret_cast<uintptr_t>(ptr) ^ 
                                    reinterpret_cast<uintptr_t>(slot));
    *slot = nullptr;
    gprint("%p: Received target buf: %p from peer %d", 
            slot, s_workInfo.targetBuf, item.peer);
  }
}

// there maybe many work items: one for each gpu block..
template < uint32_t BlockSz >
__global__ void rcclKernel(WorkInfo *gworkInfo) { 

  constexpr uint32_t s_num = sizeof(WorkInfo) / sizeof(uint64_t),
            warpSize = 64;

  uint32_t tid = threadIdx.x, groupID = tid / warpSize;
  if(tid < s_num) {
    auto d = ((uint64_t *)gworkInfo)[tid];
    ((uint64_t *)&s_workInfo)[tid] = d;
  }
  __syncthreads();

  // we will use directWrite: that is, each sender writes data to receiver buffer directly
  // for that, receiver should provide sender the buffer address
  if(groupID < 2) {
    doReceive(tid);
  } else {
    doSend(tid - 2*warpSize);
  }

  __syncthreads();

  auto srcBuf = (const uint64_t *)s_workInfo.sendItem.dataBuf;
  const uint32_t bytes = s_workInfo.sendItem.size, 
             n64words = bytes / sizeof(uint64_t);

  auto targetBuf = (uint64_t *)s_workInfo.targetBuf;
  constexpr uint32_t NumRegs = 4;
  uint64_t regs[NumRegs];

  // each thread loads 8 or 16 bytes of data ??
  for(uint32_t ofs = 0; ofs < n64words; ofs += BlockSz*NumRegs) {
    if(tid == 0) {
      gprint("running ofs: %d n64words: %d", ofs, n64words);
    }
#pragma unroll
    // we can load BlockSz * NumRegs in one step here
    for(uint32_t i = 0; i < NumRegs/2; i++) {
      //if(tid*2 < n64words - 2) 
      {
        regs[2*i] = LOAD(srcBuf + tid*2);
        regs[2*i + 1] = LOAD(srcBuf + tid*2 + 1);
      }
      // loaded two 64-bit words per thread in one go
      srcBuf += BlockSz*2;
    }
#pragma unroll
    for(uint32_t i = 0; i < NumRegs/2; i++) {
      //if(tid*2 < n64words - 2) 
      {
        STORE(regs[2*i], targetBuf + tid*2);
        STORE(regs[2*i + 1], targetBuf + tid*2 + 1);
      }
      // loaded two 64-bit words per thread in one go
      targetBuf += BlockSz*2;
    }
    //n64words -= BlockSz*NumRegs;
  } // for ofs
  __threadfence();
  // NOTE: receiver block must spin here until sender is ready in order
  // to finish kernel when data transfer is finished..
}

// function run on a thread: this ID receives from recvPeer and 
// sends to sendPeer
void GpuCommLib::runSendRecv(uint32_t ID, uint32_t recvPeer, void *recvBuf,
        size_t recvSize, uint32_t sendPeer, void *sendBuf, size_t sendSize, 
        cudaStream_t stream) {
    
    // NOTE: exchange pointers are always allocated on the receiver side!!
  WorkInfo w = {
      .recvItem = { // whom we are receiving from
        .peer = recvPeer,
        .size = (uint32_t)recvSize,
        // exchange buf on the receiver side
        .exchangeBuf = (void **)m_infos[ID].exchangeBuf + recvPeer,
        .dataBuf = recvBuf,
      },
      .sendItem = { // whom we are sending to
        .peer = sendPeer,
        .size = (uint32_t)sendSize,
        // exchange buf on the receiver side
        .exchangeBuf = (void **)m_infos[sendPeer].exchangeBuf + ID, 
        .dataBuf = sendBuf,
      },
      .targetBuf = nullptr
  };
  VLOG("Work Item size: " << sizeof(WorkInfo));
  CHK(cudaSetDevice(m_infos[ID].gpuId));
  CHK(cudaMemcpyAsync(m_infos[ID].workBuf, &w, sizeof(WorkInfo), 
                                               cudaMemcpyHostToDevice, stream));
  constexpr uint32_t BlockSz = 256;
  rcclKernel<BlockSz><<<1, BlockSz, 0, stream>>>(m_infos[ID].workBuf);
  // CHK(cudaMemcpyPeerAsync(recvBuf, 
  //          sendP, info.sendBuf + m_offsets[i], id, m_sizes[i]*sizeof(T), info.streams[i]));
}

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
  int nGpus = 0, elems = 8192;
  CHK(cudaGetDeviceCount(&nGpus));
  VLOG("Num devices: " << nGpus);
  nGpus = 2;
  GpuCommLib commLib(nGpus);
  ThreadPool pool(nGpus);
  Barrier barrier(nGpus);

  // float *devPtr;
  // (void)hipMalloc((void**)&devPtr, 16);
  // (void)hipMemset(devPtr, 0, 16);
  // atomicKernel<<<512, 512, 0>>>(devPtr);
  // (void)hipFree(devPtr);

  std::vector< SendRecvItem<T> > items(nGpus);
  std::mutex mtx;

  pool.runJob([&](int id) {

    auto& item = items[id];
    int sendPeer = (id + 1)%nGpus, 
        recvPeer = (id - 1 + nGpus)%nGpus;
    PRINTZ("GPU %d recv from %d and sends to %d", id, recvPeer, sendPeer);
    item.init(id, elems);
    auto size = item.numElems * sizeof(T);
    commLib.runSendRecv(id, recvPeer, item.recvBuf, size, 
                            sendPeer, item.sendBuf, size, item.stream);

    CHK(cudaMemcpyAsync(item.hostBuf.data(), item.recvBuf, size, 
                          cudaMemcpyDeviceToHost, item.stream));
    CHK(cudaStreamSynchronize(item.stream));

    // so far receiver side is not synchronized => hence we need barrier
    barrier.wait(); 
    std::lock_guard _(mtx);
    item.verify(recvPeer);
  });
}

int main() try 
{
  DeviceInit(0);
  runRCCLTest<uint32_t>();
}
catch(std::exception& ex) {
  VLOG("Exception: " << ex.what());
}
catch(...) {
  VLOG("Unknown exception");
}
