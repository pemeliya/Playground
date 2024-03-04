
// hipcc -I.. -DCOMPILE_FOR_ROCM=1 -std=c++17 --offload-arch=gfx90a test_main.cc

#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <thread>
#include "qccl_lib.h"
#include "common/threading.hpp"
#include "common/example_utils.hpp"

#if 0
#define gprint(fmt, ...) printf(fmt"\n", ##__VA_ARGS__)
#else
#define gprint(fmt, ...)
#endif

// NOTE: if data size is small, maybe makes sense to use just normal load/store?
#if 0
#define LOAD(addr) __builtin_nontemporal_load(addr)
#define STORE(x, addr) __builtin_nontemporal_store((x), (addr))
#else
#define LOAD(addr) (*(addr))
#define STORE(x, addr) *(addr) = (x)
#endif

#define ATOMIC_LOAD(VAR)       __atomic_load_n((VAR),         __ATOMIC_ACQUIRE)
#define ATOMIC_STORE(DST, SRC) __atomic_store_n((DST), (SRC), __ATOMIC_RELEASE)

struct P2PWorkItem {
  uint32_t peer;      // send/recv peer
  uint32_t size;      // buffer size in bytes (limited by 4GB!)
  void **exchangeBuf; // shared buffer for exchanging pointers between GPUs:
                      // this should be set accordingly for each p2p channel (pair of GPUs)
                      // It has two entries: one for ptr exchange and one for end-of-transfer flag
  void *dataBuf;      // send/recv buffer 
};

struct WorkInfo
{
  P2PWorkItem recvItem, sendItem;
  void *targetBuf;    // target buffer address obtained from the receiver
};

static_assert(sizeof(WorkInfo) % sizeof(uint64_t) == 0, 
    "Size must be aligned by 8 bytes");

template < uint32_t BlockSz, uint32_t NumRegs >
__global__ void rcclKernel(WorkInfo *gworkInfo);

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

  static constexpr size_t s_defNumWorkItems = 8;
  static constexpr size_t s_numWorkThreads = 512;
  static constexpr size_t s_numRegsPerThread = 32;

  struct ThreadInfo {
    int gpuId;             // gpu ID assigned to this thread
    WorkInfo *workBuf;     // work buffer global memory
    void **exchangeBuf;    // shared buffer for exchanging pointers
    size_t numDevWorkItems;   // the number of workBuf items preallocated in device mem
    std::vector< WorkInfo > workItems;  // the list of current work items submitted
  };

  bool m_initialized = false;
  std::vector< ThreadInfo > m_infos;

public:
  static GpuCommLib& i() {
    static GpuCommLib obj;
    return obj;
  }

  QCCL_Result init(size_t nGpus) {
    if(m_initialized) return QCCL_Result::OK;

    m_infos.resize(nGpus);
    int i = 0;
    size_t exchangeSz = nGpus * sizeof(void *);
    for(auto& info : m_infos) {
      info.gpuId = i++;
      info.workItems.reserve(s_defNumWorkItems);
      CHK(cudaSetDevice(info.gpuId));
      int flags = //hipDeviceMallocDefault;
                  hipDeviceMallocFinegrained;
                  // hipDeviceMallocUncached;
                  // hipMallocSignalMemory;
      CHK(hipExtMallocWithFlags((void **)&info.exchangeBuf, exchangeSz*2, flags));
      CHK(cudaMemset(info.exchangeBuf, 0, exchangeSz*2));
      allocWorkBuf(&info, s_defNumWorkItems);
    }
    for(const auto& info : m_infos) {
      CHK(cudaSetDevice(info.gpuId));
      for(uint32_t j = 0; j < nGpus; j++) {
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
    m_initialized = true;
    return QCCL_Result::OK;
  }

  // function run on a thread: this ID receives from recvPeer and 
  // sends to sendPeer
  QCCL_Result enqueueSendRecv(uint32_t ID, uint32_t recvPeer, void *recvBuf,
        size_t recvSize, uint32_t sendPeer, void *sendBuf, size_t sendSize) {

    if(!m_initialized) return QCCL_Result::NotInitialized;
    if(ID >= m_infos.size()) return QCCL_Result::InvalidParams;
    auto& info = m_infos[ID];
    // NOTE: exchange pointers are always allocated on the receiver side!!
    auto& w = info.workItems.emplace_back();
    w.recvItem = { // whom we are receiving from
          .peer = recvPeer,
          .size = (uint32_t)recvSize,
          // exchange buf on the receiver side: two entries per link
          .exchangeBuf = (void **)m_infos[ID].exchangeBuf + recvPeer*2,
          .dataBuf = recvBuf,
    };
    w.sendItem = { // whom we are sending to
          .peer = sendPeer,
          .size = (uint32_t)sendSize,
          // exchange buf on the receiver side: two entries per link
          .exchangeBuf = (void **)m_infos[sendPeer].exchangeBuf + ID*2, 
          .dataBuf = sendBuf,
    };
    w.targetBuf = nullptr;
    return QCCL_Result::OK;
  }

  // execute previously enqueued send-recv tasks for this thread (one GPU)
  QCCL_Result run(uint32_t ID, cudaStream_t stream) {

    auto& info = m_infos[ID];
    CHK(cudaSetDevice(info.gpuId));

    if(info.numDevWorkItems < info.workItems.size()) {
      CHK(cudaFree(info.workBuf));
      allocWorkBuf(&info, info.numDevWorkItems * 3 / 2);
    }
    uint32_t nBlocks = info.workItems.size();
    // VLOG("Work Item size: " << sizeof(WorkInfo) << " executing with #blocks:" 
    //       << nBlocks);

    CHK(cudaMemcpyAsync(info.workBuf, info.workItems.data(), 
          sizeof(WorkInfo) * nBlocks, cudaMemcpyHostToDevice, stream));
    
    constexpr uint32_t BlockSz = s_numWorkThreads;
    rcclKernel<BlockSz, s_numRegsPerThread><<<nBlocks, BlockSz, 0, stream>>>
                                      (info.workBuf);
    info.workItems.clear();
    return QCCL_Result::OK;
  }

  ~GpuCommLib() {
    for(auto& info : m_infos) {
      (void)cudaSetDevice(info.gpuId);
      (void)cudaFree(info.workBuf);
      (void)cudaFree(info.exchangeBuf);
    }
  }
  
protected:
  // there should be a common buffer between each pair of GPUs communicated
  GpuCommLib() = default;

  QCCL_Result allocWorkBuf(ThreadInfo *pinfo, size_t num) {
    pinfo->numDevWorkItems = num;
    auto bytes = sizeof(WorkInfo) * num;
    CHK(hipExtMallocWithFlags((void **)&pinfo->workBuf, bytes, hipDeviceMallocDefault));
    return QCCL_Result::OK;
  }

}; // GpuCommLib

QCCL_Result qcclInit(uint32_t nGpus) {
  return GpuCommLib::i().init(nGpus);
}

QCCL_Result qcclSendRecv(uint32_t ID, uint32_t recvPeer, void *recvBuf,
        size_t recvSize, uint32_t sendPeer, void *sendBuf, size_t sendSize) {
  return GpuCommLib::i().enqueueSendRecv(ID, recvPeer, recvBuf,
        recvSize, sendPeer, sendBuf, sendSize);
}

QCCL_Result qcclRun(uint32_t ID, cudaStream_t stream) {
  return GpuCommLib::i().run(ID, stream);
}

//Matrix< WorkInfo > p2pSend, p2pRecv;
// ncclSend(a,b) -> set p2pSend(a,b) = sendBuf, sendSize (a sends to b) => a needs to get recv buffer from b
// ncclRecv(b,a) -> set p2pRecv(b,a) = recvBuf, recvSize (b receives from a) => b needs give its recv buffer to a

__shared__ WorkInfo s_workInfo;


// ltid is a local tid within group !!!
__device__ void doReceive(uint32_t ltid) {

  // we provide the sender our receive buffer
  if(ltid == 0) {
    auto& item = s_workInfo.recvItem;
    void *volatile *slot = item.exchangeBuf;
    *(uint32_t *)(slot + 1) = 0; // reset 'receive complete' flag

    // Wait for consumer to consume previous value before trampling it.
    while((void *)atomicAdd((uint64_t *)slot, 0) != nullptr);
    // Encode pointer by XOR'ing against some address they definitely wouldn't send
    // since we want to allow them sending us nullptr while not colliding with
    // the empty slot value.
    *slot = (void *)(reinterpret_cast<uintptr_t>(item.dataBuf) ^ 
                     reinterpret_cast<uintptr_t>(slot));
    gprint("%p Sent target buffer: %p to the sender peer %d", 
          slot, item.dataBuf, s_workInfo.sendItem.peer);
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
            slot, s_workInfo.targetBuf, s_workInfo.recvItem.peer);
  }
}

// there maybe many work items: one for each gpu block..
template < uint32_t BlockSz, uint32_t NumRegs >
__launch_bounds__(BlockSz, 1)
__global__ void rcclKernel(WorkInfo *gworkInfo) { 

  constexpr uint32_t s_num = sizeof(WorkInfo) / sizeof(uint64_t),
            warpSize = 64;

  uint32_t tid = threadIdx.x;
  if(tid < s_num) {
    auto d = ((uint64_t *)gworkInfo)[tid];
    ((uint64_t *)&s_workInfo)[tid] = d;
  }
  __syncthreads();

  // we will use directWrite: that is, each sender writes data to receiver buffer directly
  // for that, receiver should provide sender the buffer address
  if(tid < BlockSz/2) {
    doReceive(tid);
  } else {
    doSend(tid - BlockSz/2);
  }

  __syncthreads();

  using Word = uint64_t;
  auto srcBuf = (const Word *)s_workInfo.sendItem.dataBuf;
  const uint32_t bytes = s_workInfo.sendItem.size, 
             nwords = bytes / sizeof(Word);

  auto targetBuf = (Word *)s_workInfo.targetBuf;
  Word regs[NumRegs]; 

  // each thread loads 8 or 16 bytes of data ??
  for(uint32_t ofs = tid*2; ofs < nwords; ) {
    if(tid == 0) {
      gprint("running ofs: %d nwords: %d", ofs, nwords);
    }
    auto src_ofs = ofs;
#pragma unroll
    // we can load BlockSz * NumRegs in one step here
    for(uint32_t i = 0; i < NumRegs/2; i++) {
      //if(src_ofs < nwords)
        regs[2*i] = LOAD(srcBuf + src_ofs);
      //if(src_ofs + 1< nwords)
        regs[2*i + 1] = LOAD(srcBuf + src_ofs + 1);
      src_ofs += BlockSz*2;
    }
#pragma unroll
    for(uint32_t i = 0; i < NumRegs/2; i++) {
      //if(ofs < nwords)
        STORE(regs[2*i], targetBuf + ofs);
      //if(ofs + 1 < nwords)
        STORE(regs[2*i + 1], targetBuf + ofs + 1);
      ofs += BlockSz*2;
    }
  } // for ofs
  __threadfence_system(); // TODO check if it's correct

  // NOTE: it could be that some channel is only sender or only receiver ??

  auto recvDone = (volatile uint32_t *)(s_workInfo.recvItem.exchangeBuf + 1);
  auto sendDone = (volatile uint32_t *)(s_workInfo.sendItem.exchangeBuf + 1);
  sendDone[0] = 11111;
  // __atomic_store_n(send_done, 1, __ATOMIC_SEQ_CST);

  if(tid == 0) {
    gprint("Receiver waiting peer: %d", s_workInfo.sendItem.peer);
    while(atomicAdd((uint32_t *)recvDone, 0u) != 11111u);
    gprint("Waiting done.. %d", s_workInfo.sendItem.peer);
  }
}

#if 0
template < class T >
void runRCCLTest()
{
  int nGpus = 0, elems = 1002;
  CHK(cudaGetDeviceCount(&nGpus));
  VLOG("Num devices: " << nGpus);
  nGpus = 2;
  GpuCommLib commLib(nGpus);
  ThreadPool pool(nGpus);
  Barrier barrier(nGpus);

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
#endif 
