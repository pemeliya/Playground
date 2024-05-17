
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
#include "buffer_addressing.hpp"
#include "common/threading.hpp"
#include "common/example_utils.hpp"

// AMDGPU docs https://llvm.org/docs/AMDGPU/

#if 0
#define gprint(fmt, ...) printf(fmt"\n", ##__VA_ARGS__)
#else
#define gprint(fmt, ...)
#endif

#define USE_CONSTANT_MEM 0
#define USE_BUFFER_LOADS 0
#define MAX_NUM_NODES 16
// experimental feature to preload registers for the first iteration
#define USE_PRELOAD_REGS 0
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

#if 1
#define ATOMIC_LOAD(VAR)       __atomic_load_n((VAR),         __ATOMIC_ACQUIRE)
#define ATOMIC_STORE(PTR, VAL) __atomic_store_n((PTR), (VAL), __ATOMIC_RELEASE)
#else
#define ATOMIC_LOAD(VAR)       (VAR)[0]
#define ATOMIC_STORE(PTR, VAL) (PTR)[0] = (VAL)
#endif

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
  uint8_t *targetBuf;         // target buffer to be shared 
  IncomingWorkItem incoming;
  OutgoingWorkItem outgoing;
};

static_assert(sizeof(WorkInfo) % sizeof(uint64_t) == 0, 
    "Size must be aligned by 8 bytes");

template < uint32_t BlockSz, uint32_t NumRegs >
__global__ void rcclKernel(WorkInfo *gworkInfo);

class GpuCommLib {

/* best all-to-all
Data size: 4.00 Mb; time elapsed: 0.131 ms, bandwidth: 32.028 Gb/s
Data size: 6.00 Mb; time elapsed: 0.123 ms, bandwidth: 51.146 Gb/s
Data size: 9.00 Mb; time elapsed: 0.124 ms, bandwidth: 76.156 Gb/s
Data size: 13.50 Mb; time elapsed: 0.126 ms, bandwidth: 112.331 Gb/s
Data size: 20.25 Mb; time elapsed: 0.146 ms, bandwidth: 145.579 Gb/s
Data size: 30.38 Mb; time elapsed: 0.179 ms, bandwidth: 177.933 Gb/s
Data size: 32.00 Mb; time elapsed: 0.183 ms, bandwidth: 183.046 Gb/s
*/

  static constexpr size_t s_defNumWorkItems = 8;
  static constexpr size_t s_numWorkThreads = 256;
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

  QCCL_Result init(size_t nGpus, const uint32_t *gpuIds) {
    if(m_initialized) return QCCL_Result::OK;

    m_infos.resize(nGpus);
    size_t exchangeSz = sizeof(void *) * STotalSlots * 8;
    for(uint32_t i = 0; i < nGpus; i++) {
      auto& info = m_infos[i];
      info.gpuId = gpuIds != nullptr ? gpuIds[i] : i;
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
        auto gj = m_infos[j].gpuId;
        if(info.gpuId == gj)
          continue;
        int enable = -1;
        CHK(cudaDeviceCanAccessPeer(&enable, info.gpuId, gj));
        if(enable == 0) {
          ThrowError<>("GPU %d is unable to access peer %d", info.gpuId, gj);
        }
        CHK(cudaDeviceEnablePeerAccess(gj, 0));
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
    w.readyFlagCache = 0,
    w.incoming = { // whom we are receiving from
          .peer = inPeer,
          .size = (uint32_t)inSize,
          // exchange buf on the receiver side: two entries per link
          // (we are receiver here): there we always publish our pointer
          .exchangeBuf = (void **)m_infos[ID].exchangeBuf + inPeer*STotalSlots,
          .targetBuf = (uint8_t *)targetBuf,
    };
    w.outgoing = { // whom we are sending to
          .peer = outPeer,
          .size = (uint32_t)outSize,
          // exchange buf on the receiver side: two entries per link
          // node 'outPeer' is a receiver 
          .exchangeBuf = (void **)m_infos[outPeer].exchangeBuf + ID*STotalSlots,
          .sourceBuf = (uint8_t *)sourceBuf,
    };
    return QCCL_Result::OK;
  }

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

    static int ii = 1000;
    // here we are receiving from 'peerStart' and forwarding to 'peerEnd'
    auto& w = info.workItems.emplace_back();
    w.ID = 1000 + ID;
    w.nPeers = numSubscribedPeers,
    w.dataOfs = dataOfs,
    w.readyFlagCache = 0,
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
    // VLOG(ID << ": workItemSz: " << sizeof(WorkInfo) << " running with #blocks: " 
    //          << nBlocks);
    // NOTE: do we really need to copy workBuf all the time ???
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
    CHK(hipExtMallocWithFlags((void **)&pinfo->workBuf, bytes, 
          hipDeviceMallocFinegrained));
    return QCCL_Result::OK;
  }

}; // GpuCommLib

#if !USE_CONSTANT_MEM
__shared__ WorkInfo ds_work;
//#define WORK(x) 
#else
__constant__ WorkInfo ds_work[MAX_NUM_NODES];
#endif

__forceinline__ __device__ void setupInPtrs() {

  if(ds_work.ID == ds_work.incoming.peer) {
    return; // we are receiving from ourselves
  }

  // we provide the sender our incoming buffer
  // NOTE adding GLOBAL here, inserts s_waitcnt vmcnt(7) after global stores!!!.
  // but why ???
  auto slot = (void *volatile *)ds_work.incoming.exchangeBuf;
  auto counter = (uint32_t GLOBAL *)(slot + SReadyFlagCounter);
  auto targetBuf = ds_work.incoming.targetBuf;

  //! NOTE hangs here because we reset ready flag too fast (or too late)
  ds_work.readyFlagCache = ATOMIC_LOAD(counter);
  //atomicAdd(counter, 0);

  // Wait for consumer to consume previous value before trampling it.
  while((void *)ATOMIC_LOAD((uint64_t GLOBAL *)(slot + STargetBuf)) != nullptr);
  // Encode pointer by XOR'ing against some address they definitely wouldn't send
  // since we want to allow them sending us nullptr while not colliding with
  // the empty slot value.
  auto xorval = (void *)(reinterpret_cast<uintptr_t>(targetBuf) ^ 
                         reinterpret_cast<uintptr_t>(slot));
  ATOMIC_STORE(slot + STargetBuf, xorval);
  
  // we also publish our own data buf (sendBuf)
  xorval = (void *)(reinterpret_cast<uintptr_t>(ds_work.outgoing.sourceBuf) ^ 
                    reinterpret_cast<uintptr_t>(slot));
  // NOTE: we should also do a while loop for source buf since otherwise
  // it is unsynchronized..
  ATOMIC_STORE(slot + SSourceBuf, xorval);
  // gprint("%d / %p: Sent target buffer: %p to send peer %d", 
  //         ds_work.ID, slot, targetBuf, ds_work.incoming.peer);
}

__forceinline__ __device__ void setupGatewayPtrs() {

  // we read source buffer from incoming peer since we would like to 
  // forward its data to the outgoing peer
  auto& item = ds_work.incoming;
  auto slot = (void *volatile GLOBAL *)item.exchangeBuf;
  // gprint("%d / %p: Starting receive GW input buf", ds_work.ID, slot);

  void *ptr;
  while (true) {
    ptr = (void *)ATOMIC_LOAD((uint64_t GLOBAL *)(slot + SSourceBuf));
    if (ptr != nullptr) break;
  }
  ds_work.outgoing.sourceBuf = (uint8_t *)(reinterpret_cast<uintptr_t>(ptr) ^ 
                                           reinterpret_cast<uintptr_t>(slot));
  // gprint("%d / %p: Received SRC buf: %p from peer %d", 
  //           ds_work.ID, slot, ds_work.outgoing.sourceBuf, 
  //                       ds_work.outgoing.peer);
}

__forceinline__ __device__ void setupOutPtrs() {

  // we are sending to ourselves
  if(ds_work.ID == ds_work.outgoing.peer) {
    ds_work.targetBuf = ds_work.incoming.targetBuf;
    return; 
  }

  auto& item = ds_work.outgoing;
  void *volatile *slot = item.exchangeBuf;
  // gprint("%d / %p: Starting receive target buf", ds_work.ID, slot);

  void *ptr;
  while (true) {
    ptr = (void *)ATOMIC_LOAD((uint64_t GLOBAL*)(slot + STargetBuf));
    if (ptr != nullptr) break;    
  }
  ds_work.targetBuf = (uint8_t *)(reinterpret_cast<uintptr_t>(ptr) ^ 
                                  reinterpret_cast<uintptr_t>(slot));
  // gprint("%d / %p: Received target buf: %p from recv peer %d", 
  //           ds_work.ID, slot, ds_work.incoming.targetBuf, 
  //                                ds_work.outgoing.peer);
}

__forceinline__ __device__ void resetBufferPtrs() 
{
  auto& item = ds_work.outgoing;
  auto slot = (void * GLOBAL *)item.exchangeBuf;
  auto counter = (uint32_t *)(slot + SBufsReceivedCounter);
  auto val = 1 + atomicAdd(counter, 1);

  if(val % ds_work.nPeers == 0) {
    // gprint("%d: resetting buffers val = %d if zero", ds_work.ID, val);
    ATOMIC_STORE(slot + STargetBuf, nullptr);
  }
}

__forceinline__ __device__ void finalizeSendRecv(uint32_t tid) {

  // auto tid = gpuLaneId();
  if(tid == 0) {
    auto slot = (void *GLOBAL *)ds_work.outgoing.exchangeBuf;
    //  __atomic_store_n(send_done, 1, __ATOMIC_SEQ_CST);
    auto readyCnt = (uint32_t *)(slot + SReadyFlagCounter);
    auto val = 1 + atomicAdd(readyCnt, 1u);
    gprint("%d: incremented ready counter: %p / %d", 
            ds_work.ID, readyCnt, val);

    // gateway nodes do not need to wait here, also do not wait if
    // target buffer is NULL (we do not receive anything) 
  } else if(tid == warpSize && ds_work.dataOfs == 0 && 
            ds_work.incoming.targetBuf != nullptr) {

    if(ds_work.ID == ds_work.incoming.peer)
      return;

    auto slot = (void *GLOBAL *)ds_work.incoming.exchangeBuf;
    auto readyCnt = (uint32_t GLOBAL *)(slot + SReadyFlagCounter);

    uint32_t cacheVal = ds_work.readyFlagCache, 
            numPeers = ds_work.nPeers;
    // gprint("%d: Receiver waiting peer counter: %p / %d", 
    //     ds_work.ID, readyCnt, readyCnt[0], cacheVal);
    // bool sleeping = false;
    while(1) {
      //__builtin_amdgcn_s_sleep(1);
      auto val =  ATOMIC_LOAD(readyCnt);
      if(val - cacheVal == numPeers) {
        gprint("%d: Waiting done: counter: %d, cacheVal: %d", 
            ds_work.ID, val, cacheVal);
        break;
      }
    }
    // this causes the program to crash
    // __asm__ __volatile__("s_wakeup");
  }
}

template < typename Word, uint32_t BlockSz, uint32_t NumRegs >
__forceinline__ __device__ 
void loadRegs(Word (&regs)[NumRegs], uint32_t src_ofs) {

  const auto& work = ds_work;//const_work[ds_work.ID];
  auto srcBuf = (const Word GLOBAL *)(work.outgoing.sourceBuf);
  const uint32_t dataOfs = work.dataOfs;
  // preloading is only possible for non-gateway blocks
  if(!(srcBuf != nullptr && dataOfs == 0))
    return;

  #pragma unroll
  for(uint32_t i = 0; i < NumRegs/2; i++, src_ofs += BlockSz*2) {
    regs[2*i] = LOAD(srcBuf + src_ofs);
    regs[2*i + 1] = LOAD(srcBuf + src_ofs + 1);
  }
}

template < typename Word, uint32_t BlockSz, uint32_t NumRegs >
__forceinline__ __device__ 
void storeRegs(Word (&regs)[NumRegs], uint32_t ofs) {

  const auto& work = ds_work;//const_work[ds_work.ID];
  const uint32_t dataOfs = work.dataOfs;
  auto srcBuf = (const Word GLOBAL *)(work.outgoing.sourceBuf);
  auto targetBuf = (Word GLOBAL *)(work.targetBuf);

  if(!(srcBuf != nullptr && dataOfs == 0))
    return;

  #pragma unroll
  for(uint32_t i = 0; i < NumRegs/2; i++, ofs += BlockSz*2) {
    STORE(regs[2*i], targetBuf + ofs);
    STORE(regs[2*i + 1], targetBuf + ofs + 1);
  }
}

__constant__ WorkInfo const_work[MAX_NUM_NODES];

template < typename Word, uint32_t BlockSz, uint32_t NumRegs, 
        bool UseOuterLoop, bool Check >
__forceinline__ __device__ 
void copyMainLoop(Word (&regs)[NumRegs], uint32_t ofs, const uint32_t niters, const uint32_t nwords) {

  const auto& work = ds_work; //const_work[ds_work.ID];
  const uint32_t dataOfs = work.dataOfs;
  auto srcBuf = (const Word GLOBAL *)(work.outgoing.sourceBuf + dataOfs);
  auto targetBuf = (Word GLOBAL *)(work.targetBuf + dataOfs);

#if USE_BUFFER_LOADS
  // buffer load/stores perform range check automatically
  const auto src_res = make_buffer_resource(work.outgoing.sourceBuf, 
        work.outgoing.size + dataOfs);
  const auto dst_res = make_buffer_resource(work.targetBuf, 
        work.outgoing.size + dataOfs);

  int32x4_t bregs[NumRegs/2];

  union Split64 {
    void *ptr;
    struct {
      uint32_t lo;
      uint32_t hi;
    };
  };
/*
__device__ void
llvm_amdgcn_raw_buffer_store_i32x4(int32x4_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i32");

__device__ int32x4_t
llvm_amdgcn_raw_buffer_load_i32x4(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

{
    union {
        int32x4_t content;
        struct {
            void *address;
            int32_t range;
            int32_t config;
        };
    } S = {
        .address = (void *)p_wave,
        .range = space_size,
        .config = CK_BUFFER_RESOURCE_3RD_DWORD,
    };
    return S.content;
*/    

#define BUFFFER_LOADx4(result, sreg, vofs) \
    asm volatile("buffer_load_dwordx4 %0, %1, s[" #sreg ":" #sreg "+3], 0 offen" \
          : "=v"(result) : "v"(vofs), "0"(result))

    // printf("range: %d ofs: %d\n", work.outgoing.size,
    //     dataOfs);

    for(uint32_t s = 0; s < niters; s++) 
    {
      // TODO: dataOfs can go into wave offset ??
      auto src_ofs = ofs;
#pragma unroll
      for(uint32_t i = 0; i < NumRegs/2; i++, src_ofs += BlockSz*2) {
#if 1
          Split64 V = { .ptr = work.outgoing.sourceBuf + dataOfs };
          const uint32_t range = work.outgoing.size + dataOfs;
          const uint32_t config = CK_BUFFER_RESOURCE_3RD_DWORD;
          asm volatile(R"(
          v_readfirstlane_b32 s4, %0
          v_readfirstlane_b32 s5, %1
          v_readfirstlane_b32 s6, %2
          v_readfirstlane_b32 s7, %3
          s_nop 0)"
                : 
                : "v"(V.lo),      // 0
                  "v"(V.hi),
                  "v"(range), // 1
                  "v"(config)  // 2
                );
          const uint32_t bofs = src_ofs*sizeof(Word);
          BUFFFER_LOADx4(bregs[i], 4, bofs);
#else
          bregs[i] = amd_buffer_load(src_res, src_ofs*sizeof(Word), 0);
#endif
      }
      // if(UseOuterLoop && ds_work.ID == 0 && s == 1) {

      //   gprint("ofs: %d: %u %u %u %u", ofs, X.d[0], X.d[1], X.d[2], X.d[3]);
      // }
      uint32_t tid = threadIdx.x;
#pragma unroll
      for(uint32_t i = 0; i < NumRegs/2; i++, ofs += BlockSz*2) {
        union {
          int32x4_t z;
          struct { int64_t d[2]; };
        } X = { .z = bregs[i] };
       // STORE(X.d[0], targetBuf );
        if(tid >= 511) {
          //printf("%d/%d: ofs: %d size: %d\n", s, i, ofs*sizeof(Word), work.outgoing.size);
        }
        // STORE(X.d[1], targetBuf  + 1);
       amd_buffer_store< BufCoherence::Def >(dst_res, ofs*sizeof(Word), 0, bregs[i]);
      }
      //if(!UseOuterLoop) 
      break;
    } // for s
#else // USE_BUFFER_LOADS

  for(uint32_t s = 0; s < niters; s++) {
    auto src_ofs = ofs;
#pragma unroll
    for(uint32_t i = 0; i < NumRegs/2; i++, src_ofs += BlockSz*2) {
      if(!Check || src_ofs + 1 < nwords) {
        regs[2*i] = LOAD(srcBuf + src_ofs);
        regs[2*i + 1] = LOAD(srcBuf + src_ofs + 1);
      }
    }
#pragma unroll
    for(uint32_t i = 0; i < NumRegs/2; i++, ofs += BlockSz*2) {
      if(!Check || ofs + 1 < nwords) {
        STORE(regs[2*i], targetBuf + ofs);
        STORE(regs[2*i + 1], targetBuf + ofs + 1);
      }
    }
    if(!UseOuterLoop) break;
  }
#endif // USE_BUFFER_LOADS
}

// there maybe many work items: one for each gpu block..
template < uint32_t BlockSz, uint32_t NumRegs >
__launch_bounds__(BlockSz, 1)
__global__ void rcclKernel(WorkInfo *gworkInfo) { 

  using Word = uint64_t;
  
  constexpr uint32_t s_num = sizeof(WorkInfo) / sizeof(uint64_t),
            warpSize = 64;

  const uint32_t tid = threadIdx.x;

  if(tid < s_num) {
    auto pblock = (uint64_t*)(gworkInfo + blockIdx.x);
    auto d = pblock[tid];
    ((uint64_t *)&ds_work)[tid] = d;
  }
  __syncthreads();
#if USE_PRELOAD_REGS
  Word regs[NumRegs];
  loadRegs< Word, BlockSz, NumRegs >(regs, tid*2);
#endif
  // we will use directWrite: that is, each sender writes data to receiver buffer directly
  // for that, receiver should provide sender the buffer address
  if(tid == 0) {
    if(ds_work.dataOfs == 0) { // normal nodes always start from 0
      setupInPtrs(); // share pointers from whom we are receiving
    } else {
      setupGatewayPtrs();
    }
  } else if(tid == warpSize) {
    setupOutPtrs(); // obtain pointers to whom we are sending
  }
  // NOTE: this sync is needed in order to share output pointers
  __syncthreads();

  if(tid == 0) {
    resetBufferPtrs(); // when spinning is done, we reset buffer pointers
    gprint("============= %d: sourceBuf: %p, targetBuf: %p dataOfs: %d / %X size: %d / %X", 
      ds_work.ID, ds_work.outgoing.sourceBuf,
      ds_work.incoming.targetBuf, ds_work.dataOfs, ds_work.dataOfs,
      ds_work.outgoing.size, ds_work.outgoing.size);
  }
#if 0
  if(ds_work.dataOfs != 0) {   // force quit gateway nodes earlier
    finalizeSendRecv(tid);
    return;
  }
#endif

  __syncthreads();

  // check if this node sends anything..
  if(ds_work.outgoing.sourceBuf != nullptr) 
  {
    const uint32_t bytes = ds_work.outgoing.size, 
                 nwords = bytes / sizeof(Word),
                 totalIters = nwords / (BlockSz * NumRegs);
    uint32_t ofs = tid*2, niters = totalIters;
#if USE_PRELOAD_REGS
    storeRegs< Word, BlockSz, NumRegs >(regs, ofs);
    if(ds_work.dataOfs == 0) {// one less iteration for main nodes
      ofs += BlockSz*NumRegs;
      niters--;
    }
#else
    Word regs[NumRegs];
#endif
    copyMainLoop< Word, BlockSz, NumRegs, true, false >
                         (regs, ofs, niters, 0);

    constexpr uint32_t bytesPerIter = BlockSz*NumRegs*sizeof(Word);
    const uint32_t bytesLeft = bytes - totalIters*bytesPerIter,
                   wordsLeft = bytesLeft / sizeof(Word);

    if(tid == 0) {
      gprint("ID %d; nwords: %d; bytes: %d mod16: %d niters: %d "
             "bytesPerIter: %d bytesLeft: %d wordsLeft: %d", 
            ds_work.ID, nwords, bytes, bytes%16, totalIters, bytesPerIter, 
            bytesLeft, wordsLeft);
    }

    // we are left with at most BlockSz*NumRegs
    ofs += niters*BlockSz*NumRegs;
    // the loop above covers bytes divisible by 16...
    copyMainLoop< Word, BlockSz, NumRegs, false, true >
                           (regs, ofs, 1, nwords);
   
    const uint32_t nbytes16 = bytes % 16;
    if(tid < nbytes16 / 4) { // 12, 8 or 4 bytes left
      const uint32_t dataOfs = ds_work.dataOfs + (bytes & ~15) + tid*4;
      auto srcBuf = (const uint32_t GLOBAL *)(ds_work.outgoing.sourceBuf + dataOfs);
      auto targetBuf = (uint32_t GLOBAL *)(ds_work.targetBuf + dataOfs);
      auto val = LOAD(srcBuf);
      STORE(val, targetBuf);
    }
  }
  __threadfence(); // TODO check if it's correct
  finalizeSendRecv(tid);
}

QCCL_Result qcclInit(uint32_t nGpus, const uint32_t *gpuIds) {
  return GpuCommLib::i().init(nGpus, gpuIds);
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

