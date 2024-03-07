
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

#if 1
#define gprint(fmt, ...) printf(fmt"\n", ##__VA_ARGS__)
#else
#define gprint(fmt, ...)
#endif

// NOTE: if data size is small, maybe makes sense to use just normal load/store?
#if 0
#define LOAD(addr) __builtin_nontemporal_load(addr)
#else
#define LOAD(addr) (addr)[0]
#endif
// it seems that loading with cache and storing without it gives the best results
#if 1
#define STORE(x, addr) __builtin_nontemporal_store((x), (addr))
#else
#define STORE(x, addr) (addr)[0] = (x)
#endif

#define ATOMIC_LOAD(VAR)       __atomic_load_n((VAR),         __ATOMIC_ACQUIRE)
#define ATOMIC_STORE(PTR, VAL) __atomic_store_n((PTR), (VAL), __ATOMIC_RELEASE)

enum SlotInfo {
  STargetBuf = 0,
  SSourceBuf,
  SPtrExchangeCounter,
  SReadyFlag,
  SReadyFlagCounter,
  STotalSlots,
};

struct SendWorkItem {
  uint32_t peer;      // send/recv peer
  uint32_t size;      // buffer size in bytes (limited by 4GB!)
  void **exchangeBuf; // shared buffer for exchanging pointers between GPUs:
                      // this should be set accordingly for each p2p channel (pair of GPUs)
                      // It has two entries: one for ptr exchange and one for end-of-transfer flag
  void *dataBuf;      // send/recv buffer 
};

struct RecvWorkItem {
  uint32_t peer;      // send/recv peer
  uint32_t role;      // role: used to distinguish between normal and gateway nodes
  void **exchangeBuf; // shared buffer for exchanging pointers between GPUs:
                      // this should be set accordingly for each p2p channel (pair of GPUs)
                      // It has two entries: one for ptr exchange and one for end-of-transfer flag
  void *dataBuf;      // send/recv buffer 
};

struct WorkInfo
{
  uint32_t ID;        // my own ID
  RecvWorkItem recv;
  SendWorkItem send;
  void *targetBuf;    // target buffer address obtained from the receiver
};

static_assert(sizeof(WorkInfo) % sizeof(uint64_t) == 0, 
    "Size must be aligned by 8 bytes");

template < uint32_t BlockSz, uint32_t NumRegs >
__global__ void rcclKernel(WorkInfo *gworkInfo);

class GpuCommLib {

  static constexpr size_t s_defNumWorkItems = 8;
  static constexpr size_t s_numWorkThreads = 512;
  static constexpr size_t s_numRegsPerThread = 24;

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
    size_t exchangeSz = sizeof(void *) * STotalSlots;
    for(auto& info : m_infos) {
      info.gpuId = i++;
      info.workItems.reserve(s_defNumWorkItems);
      CHK(cudaSetDevice(info.gpuId));
      int flags = //hipDeviceMallocDefault;
                  hipDeviceMallocFinegrained;
                  // hipDeviceMallocUncached;
                  // hipMallocSignalMemory;
      CHK(hipExtMallocWithFlags((void **)&info.exchangeBuf, exchangeSz, flags));
      CHK(cudaMemset(info.exchangeBuf, 0, exchangeSz));
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
          ThrowError<>("GPU %d is unable to access peer %d", info.gpuId, j);
        }
        CHK(cudaDeviceEnablePeerAccess(j, 0));
      }
    } // for info
    m_initialized = true;

#if 0
    int nBlocks = 0;
    CHK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&nBlocks, 
      rcclKernel<s_numWorkThreads, s_numRegsPerThread>, s_numWorkThreads, 
      sizeof(WorkInfo)));
    VLOG("Max blocks per SM: " << nBlocks);
#endif
    return QCCL_Result::OK;
  }

  QCCL_Result sendRecv(uint32_t ID, uint32_t recvPeer, void *recvBuf,
        size_t recvSize, uint32_t sendPeer, void *sendBuf, size_t sendSize) {

    // exchangeBuf[0] - initial pointer exchange 
    // exchangeBuf[1] - subscription count (how many gpus already read it)
    // exchangeBuf[2] - writing done flag.. (we also should have several ones)

    VLOG(ID << ": sendRecv recvPeer: " << recvPeer << " -- sendPeer "  << sendPeer);

    if(!m_initialized) return QCCL_Result::NotInitialized;
    if(ID >= m_infos.size()) return QCCL_Result::InvalidParams;
    auto& info = m_infos[ID];
    // NOTE: exchange pointers are always allocated on the receiver side!!
    auto& w = info.workItems.emplace_back();
    w.ID = ID;
    w.recv = { // whom we are receiving from
          .peer = recvPeer,
          .role = 0,
          // exchange buf on the receiver side: two entries per link
          // (we are receiver here): there we always publish our pointer
          .exchangeBuf = (void **)m_infos[ID].exchangeBuf,
          .dataBuf = recvBuf,
    };
    w.send = { // whom we are sending to
          .peer = sendPeer,
          .size = (uint32_t)sendSize,
          // exchange buf on the receiver side: two entries per link
          // node 'sendPeer' is a receiver 
          .exchangeBuf = (void **)m_infos[sendPeer].exchangeBuf, 
          .dataBuf = sendBuf,
    };
    w.targetBuf = nullptr;
    return QCCL_Result::OK;
  }

  // 0 --> 1
  // 0 --> 2 --> 1
  // 2 needs read buffer from 0 and write buffer from 1
  // for node zero sendPeer is 1, hence node 0 will be waiting for node 1
  // to get write buffer from node 1, the gateway node '2' must connect 
  // to node's 1 exchange buffer with node 0

  QCCL_Result gatewaySend(uint32_t ID, uint32_t peerStart, uint32_t peerEnd, 
        size_t dataOfs, size_t dataSize) {

    if(!m_initialized) return QCCL_Result::NotInitialized;
    if(ID >= m_infos.size()) return QCCL_Result::InvalidParams;
    auto& info = m_infos[ID];

    // here we are receiving from 'peerStart' and forwarding to 'peerEnd'
    auto& w = info.workItems.emplace_back();
    w.ID = ID;
    // exchange buf is always on the "other" side for gateway nodes
    // we read pointers from peerStart and peerEnd
    w.recv = { // whom we are receiving from
          .peer = peerStart,
          .role = 1,
          .exchangeBuf = (void **)m_infos[peerStart].exchangeBuf, 
          .dataBuf = nullptr,
    };
    w.send = { // whom we are sending to
          .peer = peerEnd,
          .size = (uint32_t)dataSize,
          // we are attaching to peerStart -- peerEnd communication link
          // since there must be a direct connection from peerStart to peerEnd too
          .exchangeBuf = (void **)m_infos[peerEnd].exchangeBuf, 
          .dataBuf = nullptr,
    };
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
    VLOG(ID << ": workItemSz: " << sizeof(WorkInfo) << " running with #blocks:" 
           << nBlocks);

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

//Matrix< WorkInfo > p2pSend, p2pRecv;
// ncclSend(a,b) -> set p2pSend(a,b) = sendBuf, sendSize (a sends to b) => a needs to get recv buffer from b
// ncclRecv(b,a) -> set p2pRecv(b,a) = recvBuf, recvSize (b receives from a) => b needs give its recv buffer to a

__shared__ WorkInfo s_workInfo;

__forceinline__ __device__ void setupRecvPtrs() {

  // we provide the sender our receive buffer
  auto& item = s_workInfo.recv;
  void *volatile *slot = item.exchangeBuf;
  *(uint32_t *)(slot + SReadyFlag) = 0; // reset 'receive complete' flag

  // Wait for consumer to consume previous value before trampling it.
  while((void *)atomicAdd((uint64_t *)slot, 0) != nullptr);
  // Encode pointer by XOR'ing against some address they definitely wouldn't send
  // since we want to allow them sending us nullptr while not colliding with
  // the empty slot value.
  *slot = (void *)(reinterpret_cast<uintptr_t>(item.dataBuf) ^ 
                   reinterpret_cast<uintptr_t>(slot));
  // TODO: maybe do this in another warp ??
  // we also publish our own data buf (sendBuf)
  slot[SSourceBuf] = (void *)(s_workInfo.send.dataBuf);
  gprint("%d / %p: Sent target buffer: %p to send peer %d", 
          s_workInfo.ID, slot, item.dataBuf, s_workInfo.recv.peer);
}

__forceinline__ __device__ void setupSendPtrs() {

  auto& item = s_workInfo.send;
  void *volatile *slot = item.exchangeBuf;
  gprint("%d / %p: Starting receive target buf",
      s_workInfo.ID, slot);

  void *ptr;
  while (true) {
    ptr = (void *)atomicAdd((uint64_t *)slot, 0);
    if (ptr != nullptr) break;    
  }
  s_workInfo.targetBuf = (void *)(reinterpret_cast<uintptr_t>(ptr) ^ 
                                    reinterpret_cast<uintptr_t>(slot));
  gprint("%d / %p: Received target buf: %p from recv peer %d", 
            s_workInfo.ID, slot, s_workInfo.targetBuf, s_workInfo.send.peer);
}

__forceinline__ __device__ void setupGatewayPtrs() {

  auto& item = s_workInfo.recv;
  void *volatile *slot = item.exchangeBuf + SSourceBuf;
  gprint("%d / %p: Starting receive GW input buf",
      s_workInfo.ID, slot);

  void *ptr;
  while (true) {
    ptr = (void *)atomicAdd((uint64_t *)slot, 0);
    if (ptr != nullptr) break;    
  }
  auto srcBuf = (void *)(reinterpret_cast<uintptr_t>(ptr));
  gprint("%d / %p: Received SRC buf: %p from peer %d", 
            s_workInfo.ID, slot, srcBuf, s_workInfo.send.peer);
}

template < typename Word, uint32_t BlockSz, uint32_t NumRegs, 
        bool UseOuterLoop, bool Check >
__forceinline__ __device__ 
void copyMainLoop(uint32_t ofs, const uint32_t niters, const uint32_t nwords) {

  auto srcBuf = (const Word *)s_workInfo.send.dataBuf;
  auto targetBuf = (Word *)s_workInfo.targetBuf;

  Word regs[NumRegs];
  for(uint32_t s = 0; s < niters; s++) {
    auto src_ofs = ofs;
#pragma unroll
    for(uint32_t i = 0; i < NumRegs/2; i++, src_ofs += BlockSz*2) {
      if(!Check || src_ofs < nwords)
        regs[2*i] = LOAD(srcBuf + src_ofs);
      if(!Check || src_ofs + 1 < nwords)
        regs[2*i + 1] = LOAD(srcBuf + src_ofs + 1);
    }
#pragma unroll
    for(uint32_t i = 0; i < NumRegs/2; i++, ofs += BlockSz*2) {
      if(!Check || ofs < nwords)
        STORE(regs[2*i], targetBuf + ofs);
      if(!Check || ofs + 1 < nwords)
        STORE(regs[2*i + 1], targetBuf + ofs + 1);
    }
    if(!UseOuterLoop) break;
  } // for ofs
  if(s_workInfo.recv.peer == 0) {
    // uint32_t tid = threadIdx.x;
    // int diff = s_workInfo.send.size - ofs*sizeof(Word);
    // gprint("%d: ofs: %d byteOfs: %d diff: %d", tid, ofs, ofs*sizeof(Word), 
    //       diff);
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
    auto pblock = (uint64_t *)(gworkInfo + blockIdx.x);
    auto d = pblock[tid];
    ((uint64_t *)&s_workInfo)[tid] = d;
  }
  __syncthreads();

  bool isGateway = s_workInfo.recv.role == 1;

  // we will use directWrite: that is, each sender writes data to receiver buffer directly
  // for that, receiver should provide sender the buffer address
  if(tid == 0) {
    if(!isGateway) {
      setupRecvPtrs(); // share pointers from whom we are receiving
    } else {
      setupGatewayPtrs();
    }
  } else if(tid == warpSize) {
    setupSendPtrs(); // obtain pointers to whom we are sending
  }

  if(isGateway) {
    //s_workInfo.send.dataBuf = <source buffer>
  }

  __syncthreads();

  if(tid == 0) {
    gprint("============= %d: my target buffer: %p", s_workInfo.ID, s_workInfo.targetBuf);
  }

  if(isGateway)
    return;

  using Word = uint64_t;
  const uint32_t bytes = s_workInfo.send.size, 
                 nwords = bytes / sizeof(Word),
                 niters = nwords / (BlockSz * NumRegs);
/*
Speed with 24 regs / 512 threads
Data size: 8.86 Mb; time elapsed: 0.307 ms, bandwidth: 30.245 Gb/s
Data size: 13.29 Mb; time elapsed: 0.434 ms, bandwidth: 32.139 Gb/s
Data size: 19.93 Mb; time elapsed: 0.608 ms, bandwidth: 34.401 Gb/s
Data size: 29.90 Mb; time elapsed: 0.878 ms, bandwidth: 35.701 Gb/s
Data size: 44.85 Mb; time elapsed: 1.287 ms, bandwidth: 36.554 Gb/s
Data size: 67.28 Mb; time elapsed: 1.905 ms, bandwidth: 37.038 Gb/s
Data size: 100.91 Mb; time elapsed: 2.824 ms, bandwidth: 37.475 Gb/s
Data size: 151.37 Mb; time elapsed: 4.226 ms, bandwidth: 37.562 Gb/s
Data size: 227.06 Mb; time elapsed: 6.295 ms, bandwidth: 37.820 Gb/s
Data size: 283.50 Mb; time elapsed: 8.474 ms, bandwidth: 35.082 Gb/s

Speed with 32 regs / 512 threads
Data size: 8.86 Mb; time elapsed: 0.309 ms, bandwidth: 30.089 Gb/s
Data size: 13.29 Mb; time elapsed: 0.430 ms, bandwidth: 32.391 Gb/s
Data size: 19.93 Mb; time elapsed: 0.615 ms, bandwidth: 34.005 Gb/s
Data size: 29.90 Mb; time elapsed: 0.886 ms, bandwidth: 35.406 Gb/s
Data size: 44.85 Mb; time elapsed: 1.309 ms, bandwidth: 35.930 Gb/s
Data size: 67.28 Mb; time elapsed: 1.908 ms, bandwidth: 36.981 Gb/s
Data size: 100.91 Mb; time elapsed: 2.831 ms, bandwidth: 37.374 Gb/s
Data size: 151.37 Mb; time elapsed: 4.228 ms, bandwidth: 37.544 Gb/s
Data size: 227.06 Mb; time elapsed: 6.300 ms, bandwidth: 37.793 Gb/s
Data size: 283.50 Mb; time elapsed: 8.613 ms, bandwidth: 34.515 Gb/s
*/
  copyMainLoop< Word, BlockSz, NumRegs, true, false >
                         (tid*2, niters, 0);

  if(1)
  {
    constexpr uint32_t bytesPerIter = BlockSz*NumRegs*sizeof(Word);
    const uint32_t bytesLeft = bytes - niters*bytesPerIter,
                   wordsLeft = bytesLeft / sizeof(uint32_t),
                   nwords32 = bytes / sizeof(uint32_t);

    if(wordsLeft >= bytesPerIter/2) {
      // do one full iteration with reduced size
    }

    if(tid == 0) {
      gprint("nwords: %d; bytes: %d mod16: %d niters: %d "
             "bytesPerIter: %d bytesLeft: %d wordsLeft: %d ll: %d", 
            nwords, bytes, bytes%16, niters, bytesPerIter, 
            bytesLeft, wordsLeft, nwords32);
    }
    // we are left with at most BlockSz*NumRegs*sizeof(Word)/sizeof(uint32_t)
    // 32-bit words to process: 512*16*2 = 16384 words at most
    // nbytes divisible by 4 !!
    // or if we double the number of 32-bit regs => no need for outer loop ??
    auto ofs = tid*2 + niters*bytesPerIter/sizeof(uint32_t),
         src_ofs = ofs;
    copyMainLoop< uint32_t, BlockSz, NumRegs*2, false, true >
                          (ofs, 1, nwords32);
  }
  __threadfence_system(); // TODO check if it's correct

  // NOTE: it could be that some channel is only sender or only receiver ??

  if(tid == 0) {
    void *volatile *sendSlot = s_workInfo.send.exchangeBuf;
    auto sendDone = (volatile uint32_t *)(sendSlot + SReadyFlag);
    sendDone[0] = 11111; //  __atomic_store_n(send_done, 1, __ATOMIC_SEQ_CST);
    //__atomic_store_n(sendSlot, 0, __ATOMIC_RELAXED);
    sendSlot[STargetBuf] = nullptr;
    sendSlot[SSourceBuf] = nullptr; // cleanup for the next iteration

  } else if(tid == warpSize) {
    auto recvDone = (volatile uint32_t *)
        (s_workInfo.recv.exchangeBuf + SReadyFlag);
    gprint("Receiver waiting peer: %d", s_workInfo.send.peer);
    while(atomicAdd((uint32_t *)recvDone, 0u) != 11111u);
    gprint("Waiting done.. %d", s_workInfo.send.peer);
  }
}

QCCL_Result qcclInit(uint32_t nGpus) {
  return GpuCommLib::i().init(nGpus);
}

QCCL_Result qcclSendRecv(uint32_t ID, uint32_t recvPeer, void *recvBuf,
        size_t recvSize, uint32_t sendPeer, void *sendBuf, size_t sendSize) {
  return GpuCommLib::i().sendRecv(ID, recvPeer, recvBuf,
        recvSize, sendPeer, sendBuf, sendSize);
}

// register node ID as being a gateway for sending data from peerStart to peerEnd
QCCL_Result qcclGatewaySend(uint32_t ID, uint32_t peerStart, uint32_t peerEnd, 
        size_t dataOfs, size_t dataSize) {
  return GpuCommLib::i().gatewaySend(ID, peerStart, peerEnd, dataOfs, dataSize);
}

QCCL_Result qcclRun(uint32_t ID, cudaStream_t stream) {
  return GpuCommLib::i().run(ID, stream);
}

