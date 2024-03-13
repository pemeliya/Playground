
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

#define GLOBAL __attribute__((address_space(1)))

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
  SBufsReceivedCounter,
  SReadyFlagCounter, // steady counter used to monitor if data write is done
  STotalSlots,
};

struct OutgoingWorkItem { // outgoing/send work item (what this node sends out)
  uint32_t peer;      // send peer
  uint32_t size;      // buffer size in bytes (limited by 4GB!)
  void **exchangeBuf; // shared buffer for exchanging pointers between GPUs:
                      // this should be set accordingly for each p2p channel (pair of GPUs)
                      // It has two entries: one for ptr exchange and one for end-of-transfer flag
  uint8_t *sourceBuf; // source (send) buffer 
};

struct IncomingWorkItem { // incoming/recv work item (place where to receive the data)
  uint32_t peer;      // recv peer
  uint32_t size;      // buffer size in bytes (limited by 4GB!)
  void **exchangeBuf; // shared buffer for exchanging pointers between GPUs:
                      // this should be set accordingly for each p2p channel (pair of GPUs)
                      // It has two entries: one for ptr exchange and one for end-of-transfer flag
  uint8_t *targetBuf; // target (recv) buffer 
};

struct WorkInfo
{
  uint32_t ID;        // my own ID (debug only)
  uint32_t nPeers;    // this should be the number of peers connected to a given 
                      // exchange buffer
  uint32_t dataOfs;   // data offset usually only set for gateway nodes !!
  uint32_t readyFlagCache; // entry used to cache SReadyFlagCounter values
  IncomingWorkItem incoming;
  OutgoingWorkItem outgoing;
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

  QCCL_Result sendRecv(uint32_t ID, uint32_t numSubscribedPeers, 
        uint32_t inPeer, void *targetBuf, size_t inSize, 
        uint32_t outPeer, void *sourceBuf, size_t outSize) {

    // exchangeBuf[0] - initial pointer exchange 
    // exchangeBuf[1] - subscription count (how many gpus already read it)
    // exchangeBuf[2] - writing done flag.. (we also should have several ones)

    if(!m_initialized) return QCCL_Result::NotInitialized;
    if(ID >= m_infos.size()) return QCCL_Result::InvalidParams;
    auto& info = m_infos[ID];
    // NOTE: exchange pointers are always allocated on the receiver side!!
    auto& w = info.workItems.emplace_back();
    w.ID = ID;
    w.nPeers = numSubscribedPeers, // usually we know how many peers are there
    w.dataOfs = 0,
    w.incoming = { // whom we are receiving from
          .peer = inPeer,
          .size = (uint32_t)inSize,
          // exchange buf on the receiver side: two entries per link
          // (we are receiver here): there we always publish our pointer
          .exchangeBuf = (void **)m_infos[ID].exchangeBuf,
          .targetBuf = (uint8_t *)targetBuf,
    };
    w.outgoing = { // whom we are sending to
          .peer = outPeer,
          .size = (uint32_t)outSize,
          // exchange buf on the receiver side: two entries per link
          // node 'outPeer' is a receiver 
          .exchangeBuf = (void **)m_infos[outPeer].exchangeBuf, 
          .sourceBuf = (uint8_t *)sourceBuf,
    };
    return QCCL_Result::OK;
  }

  // 0 --> 1
  // 0 --> 2 --> 1
  // 2 needs read buffer from 0 and write buffer from 1
  // for node zero sendPeer is 1, hence node 0 will be waiting for node 1
  // to get write buffer from node 1, the gateway node '2' must connect 
  // to node's 1 exchange buffer with node 0

  QCCL_Result gatewaySend(uint32_t ID, uint32_t numSubscribedPeers,
         uint32_t peerStart, uint32_t peerEnd, 
         size_t dataOfs, size_t dataSize) {

    if(!m_initialized) return QCCL_Result::NotInitialized;
    if(ID >= m_infos.size()) return QCCL_Result::InvalidParams;
    auto& info = m_infos[ID];

    static int ii = 111; ii++;
    // here we are receiving from 'peerStart' and forwarding to 'peerEnd'
    auto& w = info.workItems.emplace_back();
    w.ID = ID + ii;
    w.nPeers = numSubscribedPeers,
    w.dataOfs = dataOfs,
    // exchange buf is always on the "other" side for gateway nodes
    // we read pointers from peerStart and peerEnd
    w.incoming = { // whom we are receiving from
          .peer = peerStart,
          .size = (uint32_t)dataSize,
          .exchangeBuf = (void **)m_infos[peerStart].exchangeBuf, 
          .targetBuf = nullptr,
    };
    w.outgoing = { // whom we are sending to
          .peer = peerEnd,
          .size = (uint32_t)dataSize,
          // we are attaching to peerStart -- peerEnd communication link
          // since there must be a direct connection from peerStart to peerEnd too
          .exchangeBuf = (void **)m_infos[peerEnd].exchangeBuf, 
          .sourceBuf = nullptr,
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
    // VLOG(ID << ": workItemSz: " << sizeof(WorkInfo) << " running with #blocks:" 
    //        << nBlocks);
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

__shared__ WorkInfo ds_work;

// using index_t = int32_t;
// typedef int32_t int32x4_t __attribute__((ext_vector_type(4)));
// typedef float float2_t __attribute__((ext_vector_type(2)));

// __device__ void __llvm_amdgcn_buffer_store_f32x2(float2_t vdata,
//                                                  int32x4_t rsrc,
//                                                  index_t vindex,
//                                                  index_t offset,
//                                                  bool glc,
//                                                  bool slc) __asm("llvm.amdgcn.buffer.store.v2f32");

extern uint __llvm_amdgcn_readfirstlane(uint) __asm("llvm.amdgcn.readfirstlane");

// __device__ int32_t
// llvm_amdgcn_raw_buffer_load_i32(int32x4_t srsrc,
//                                 index_t voffset,
//                                 index_t soffset,
//                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i32");


__forceinline__ __device__ void setupInPtrs(void *targetBuf) {

  // we provide the sender our incoming buffer
  auto slot = (void *volatile GLOBAL *)ds_work.incoming.exchangeBuf;
  auto counter = (uint32_t GLOBAL *)(slot + SReadyFlagCounter);

  //! NOTE hangs here because we reset ready flag too fast (or too late)
  ds_work.readyFlagCache = ATOMIC_LOAD(counter);
  //atomicAdd(counter, 0);

  // Wait for consumer to consume previous value before trampling it.
  while((void *)ATOMIC_LOAD((uint64_t GLOBAL *)slot) != nullptr);
  // Encode pointer by XOR'ing against some address they definitely wouldn't send
  // since we want to allow them sending us nullptr while not colliding with
  // the empty slot value.
  auto xorval = (void *)(reinterpret_cast<uintptr_t>(targetBuf) ^ 
                         reinterpret_cast<uintptr_t>(slot));
  ATOMIC_STORE(slot, xorval);
  
  // TODO: maybe do this in another warp ??
  // we also publish our own data buf (sendBuf)
  ATOMIC_STORE(slot + SSourceBuf, (void *)(ds_work.outgoing.sourceBuf));
  // gprint("%d / %p: Sent target buffer: %p to send peer %d", 
  //         ds_work.ID, slot, targetBuf, ds_work.incoming.peer);
}

__forceinline__ __device__ void setupGatewayPtrs() {

  // we read source buffer from incoming peer since we would like to 
  // forward its data to the outgoing peer
  auto& item = ds_work.incoming;
  auto slot = (void *volatile GLOBAL *)item.exchangeBuf + SSourceBuf;
  // gprint("%d / %p: Starting receive GW input buf", ds_work.ID, slot);

  void *ptr;
  while (true) {
    ptr = (void *)ATOMIC_LOAD((uint64_t GLOBAL *)slot);
    if (ptr != nullptr) break;
  }
  ds_work.outgoing.sourceBuf = (uint8_t *)(reinterpret_cast<uintptr_t>(ptr));
  gprint("%d / %p: Received SRC buf: %p from peer %d", 
            ds_work.ID, slot, ds_work.outgoing.sourceBuf, 
                        ds_work.outgoing.peer);
}

__forceinline__ __device__ void setupOutPtrs() {

  auto& item = ds_work.outgoing;
  void *volatile *slot = item.exchangeBuf;
  gprint("%d / %p: Starting receive target buf", ds_work.ID, slot);

  void *ptr;
  while (true) {
    ptr = (void *)ATOMIC_LOAD((uint64_t GLOBAL*)slot);
    if (ptr != nullptr) break;    
  }
  ds_work.incoming.targetBuf = (uint8_t *)(reinterpret_cast<uintptr_t>(ptr) ^ 
                                           reinterpret_cast<uintptr_t>(slot));
  gprint("%d / %p: Received target buf: %p from recv peer %d", 
            ds_work.ID, slot, ds_work.incoming.targetBuf, 
                                 ds_work.outgoing.peer);
}

__forceinline__ __device__ void resetBufferPtrs() 
{
  auto& item = ds_work.outgoing;
  auto slot = (void * GLOBAL *)item.exchangeBuf;
  auto counter = (uint32_t *)(slot + SBufsReceivedCounter);
  auto val = 1 + atomicAdd(counter, 1);

  if(val % ds_work.nPeers == 0) {
    gprint("%d: resetting buffers val = %d if zero", ds_work.ID, val);
    ATOMIC_STORE(slot + STargetBuf, nullptr);
  }
}

__forceinline__ __device__ void finalizeSendRecv(uint32_t tid) {

  // auto tid = gpuLaneId();
  
  if(tid == 0) {
    void *volatile *slot = ds_work.outgoing.exchangeBuf;
    //  __atomic_store_n(send_done, 1, __ATOMIC_SEQ_CST);
    auto readyCnt = (uint32_t *)(slot + SReadyFlagCounter);
    auto val = 1 + atomicAdd(readyCnt, 1u);
    gprint("%d: incremented ready counter: %p / %d", 
            ds_work.ID, readyCnt, val);

    // gateway nodes do not need to wait here
  } else if(tid == warpSize && ds_work.dataOfs == 0) {

    void *volatile *slot = ds_work.incoming.exchangeBuf;
    auto readyCnt = (uint32_t GLOBAL *)(slot + SReadyFlagCounter);

    uint32_t cacheVal = ds_work.readyFlagCache, 
            numPeers = ds_work.nPeers;
    gprint("%d: Receiver waiting peer counter: %p / %d", 
        ds_work.ID, readyCnt, readyCnt[0], cacheVal);

    while(1) {
      auto val =  ATOMIC_LOAD(readyCnt);
      if(val - cacheVal == numPeers) {
        gprint("%d: Waiting done: counter: %d, cacheVal: %d", 
            ds_work.ID, val, cacheVal);
        break;
      }
      // __builtin_amdgcn_s_sleep(1);
    }
  }
}

template < typename Word, uint32_t BlockSz, uint32_t NumRegs, 
        bool UseOuterLoop, bool Check >
__forceinline__ __device__ 
void copyMainLoop(uint32_t ofs, const uint32_t niters, const uint32_t nwords) {

  const uint32_t dataOfs = ds_work.dataOfs;
  auto srcBuf = (const Word GLOBAL *)(ds_work.outgoing.sourceBuf + dataOfs);
  auto targetBuf = (Word GLOBAL *)(ds_work.incoming.targetBuf + dataOfs);

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
    auto pblock = (uint64_t*)(gworkInfo + blockIdx.x);
    auto d = pblock[tid];
    ((uint64_t *)&ds_work)[tid] = d;
  }
  __syncthreads();
  // target buffer is going to be overwritten
  auto targetBuf = ds_work.incoming.targetBuf;

  __syncthreads(); // need another sync here to avoid possible data race with another warp

  // we will use directWrite: that is, each sender writes data to receiver buffer directly
  // for that, receiver should provide sender the buffer address
  if(tid == 0) {
    if(ds_work.dataOfs == 0) { // normal nodes always start from 0
      setupInPtrs(targetBuf); // share pointers from whom we are receiving
    } else {
      setupGatewayPtrs();
    }
  } else if(tid == warpSize) {
    setupOutPtrs(); // obtain pointers to whom we are sending
  }
  __syncthreads();

  if(tid == 0) {
    resetBufferPtrs(); // when spinning is done, we reset buffer pointers
    // gprint("============= %d: sourceBuf: %p, targetBuf: %p isGateway: %d", 
    //   ds_work.ID, ds_work.outgoing.sourceBuf,
    //   ds_work.incoming.targetBuf, isGateway);
  }
  //if(isGateway) // HACK we let gateways to do all job
  //  return;

  using Word = uint64_t;
  const uint32_t bytes = ds_work.outgoing.size, 
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
*/
  copyMainLoop< Word, BlockSz, NumRegs, true, false >
                         (tid*2, niters, 0);

  {
    constexpr uint32_t bytesPerIter = BlockSz*NumRegs*sizeof(Word);
    const uint32_t bytesLeft = bytes - niters*bytesPerIter,
                   wordsLeft = bytesLeft / sizeof(uint32_t),
                   nwords32 = bytes / sizeof(uint32_t);

    if(wordsLeft >= bytesPerIter/2) {
      // do one full iteration with reduced size
    }

    if(tid == 0) {
      // gprint("nwords: %d; bytes: %d mod16: %d niters: %d "
      //        "bytesPerIter: %d bytesLeft: %d wordsLeft: %d ll: %d", 
      //       nwords, bytes, bytes%16, niters, bytesPerIter, 
      //       bytesLeft, wordsLeft, nwords32);
    }
    // we are left with at most BlockSz*NumRegs*sizeof(Word)/sizeof(uint32_t)
    // 32-bit words to process: 512*16*2 = 16384 words at most
    // nbytes divisible by 4 !!
    auto ofs = tid*2 + niters*bytesPerIter/sizeof(uint32_t);
    copyMainLoop< uint32_t, BlockSz, NumRegs*2, false, true >
                          (ofs, 1, nwords32);
  }
  __threadfence_system(); // TODO check if it's correct

  // NOTE: it could be that some channel is only sender or only receiver ??
  
  // NOTE: we should set 'done' flag only then when all blocks have 
  // successfully written to the destination
  
  // sender (outgoing) increments 'ready' flag
  // receiver (incoming) waits for ready flag to be set properly

  // loop {
    
  // }
  // receiver should wait until all senders increment its 'ready flag'

  finalizeSendRecv(tid);
}

QCCL_Result qcclInit(uint32_t nGpus) {
  return GpuCommLib::i().init(nGpus);
}

QCCL_Result qcclSendRecv(uint32_t ID, uint32_t numSubscribedPeers, 
        uint32_t recvPeer, void *targetBuf, size_t recvSize, 
        uint32_t sendPeer, void *sourceBuf, size_t sendSize) {
  return GpuCommLib::i().sendRecv(ID, numSubscribedPeers, recvPeer, targetBuf,
        recvSize, sendPeer, sourceBuf, sendSize);
}

// register node ID as being a gateway for sending data from peerStart to peerEnd
QCCL_Result qcclGatewaySend(uint32_t ID, uint32_t numSubscribedPeers, 
        uint32_t peerStart, uint32_t peerEnd, 
        size_t dataOfs, size_t dataSize) {
  return GpuCommLib::i().gatewaySend(ID, numSubscribedPeers, 
          peerStart, peerEnd, dataOfs, dataSize);
}

QCCL_Result qcclRun(uint32_t ID, cudaStream_t stream) {
  return GpuCommLib::i().run(ID, stream);
}

