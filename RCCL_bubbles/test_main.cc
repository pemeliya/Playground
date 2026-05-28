
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include "common/threading.hpp"
#include "common/common_utils.hpp"

#include "test_main.h"

// the number of GPUs communicating (set to -1 to use all available GPUs)
#define NUM_ACTIVE_GPUS 8

#define CHKNCCL(cmd) \
  if(auto res = (cmd); res != ncclSuccess) {           \
    ThrowError<>("Test NCCL failure %s:%d '%s'",              \
        __FILE__,__LINE__, ncclGetErrorString(res));     \
  }

TestFramework::TestFramework(size_t nGpus, const uint32_t *gpuIDs) : m_nGpus(nGpus), 
      m_infos(nGpus), m_barrier(nGpus), m_pool(nGpus)
{ 
  CHKNCCL(ncclGetUniqueId(&m_ncclId));

  m_pool.runJob([this, gpuIDs](int id) {

     VLOG(0) << "initializing rank " << id;
    auto& info = m_infos[id];

    info.gpuId = gpuIDs != nullptr ? gpuIDs[id] : id;
    CHK(cudaSetDevice(info.gpuId));
    int flags = hipDeviceMallocDefault;
    CHK(cudaStreamCreateWithFlags(&info.stream, cudaStreamNonBlocking));
    CHKNCCL(ncclCommInitRank(&info.comm, m_nGpus, m_ncclId, id));
  }); // runJob
  CHK(cudaDeviceSynchronize());
  VLOG(0) << "Init finished..";
}

TestFramework::~TestFramework() {
  for(auto& info : m_infos) {
    (void)cudaSetDevice(info.gpuId);
    (void)cudaStreamDestroy(info.stream);
    (void)ncclCommDestroy(info.comm);
  }
}

int main() try 
{
  DeviceInit(0);
  
  int nGpus = 0;
  CHK(hipGetDeviceCount(&nGpus));
  VLOG(0) << "Num devices: " << nGpus;
  
  std::vector< uint32_t > deviceAssignment{0, 1, 2, 3, 4, 5, 6, 7 };
  if(nGpus > deviceAssignment.size()) {
    throw std::runtime_error("Invalid device assignment!");
  }
  TestFramework obj(nGpus, deviceAssignment.data());
}
catch(std::exception& ex) {
  VLOG(0) << "Exception: " << ex.what();
}
catch(...) {
  VLOG(0) << "Unknown exception";
}
