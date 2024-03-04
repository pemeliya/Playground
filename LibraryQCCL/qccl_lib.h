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

QCCL_Result qcclInit(uint32_t nGpus);

QCCL_Result qcclSendRecv(uint32_t ID, uint32_t recvPeer, void *recvBuf,
        size_t recvSize, uint32_t sendPeer, void *sendBuf, size_t sendSize);

QCCL_Result qcclRun(uint32_t ID, cudaStream_t stream);

#endif // QCCL_LIB_H