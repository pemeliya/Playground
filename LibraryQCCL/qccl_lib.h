#ifndef QCCL_LIB_H
#define QCCL_LIB_H 1

// QCCL = Quick CCL

#include "common/common.h"

enum QCCL_Result : uint32_t {
  OK,
  NotInitialized,
  InvalidParams,
  Failed,
};

#define CHKQCCL(cmd) \
  if(auto res = (cmd); res != QCCL_Result::OK) {   \
    ThrowError<>("%s:%d: QCCL failed with %d", __FILE__, __LINE__, (int)res); \
  }

// if gpuIds is nullptr, then GPU IDs assigned are [0..nGpus-1]
QCCL_Result qcclInit(uint32_t nGpus, const uint32_t *gpuIds);

// function run on a thread: this ID receives from recvPeer and sends to sendPeer
QCCL_Result qcclSendRecv(uint32_t ID, uint32_t numSubscribedPeers, 
        uint32_t recvPeer, void *recvBuf, size_t recvSize, 
        uint32_t sendPeer, void *sendBuf, size_t sendSize);

// register node ID as being a gateway for sending data from peerStart to peerEnd
QCCL_Result qcclGatewaySend(uint32_t ID, uint32_t numSubscribedPeers, 
        uint32_t peerStart, uint32_t peerEnd, 
        size_t dataOfs, size_t dataSize);

// run previously enqueued send-recv primitives on a stream
QCCL_Result qcclRun(uint32_t ID, cudaStream_t stream);

#endif // QCCL_LIB_H