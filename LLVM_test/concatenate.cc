

#include "llvm_test.h"

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

#if 1
#define gprint(fmt, ...) printf(fmt"\n", ##__VA_ARGS__)
#else
#define gprint(fmt, ...)
#endif

struct Slice {
  const uint8_t *src;
  uint32_t bytes;
};

template < uint32_t BlockSz, class NT >
__launch_bounds__(BlockSz, 1)
__global__ void concat_naive_kernel(Slice s1, Slice s2, uint8_t * __restrict__ dst,
            const uint32_t total_bytes) { 
  
  uint32_t bidx = blockIdx.x, tid = threadIdx.x,
           idx = (bidx * BlockSz + tid) * sizeof(NT);

  uint32_t diff = idx - s1.bytes;
  const NT * __restrict__ src = nullptr;
  if ((int32_t)diff < 0) {
    src = decltype(src)(s1.src + idx);
  } else if (diff < s2.bytes) {
    src = decltype(src)(s2.src + diff);
  }

  if (idx < total_bytes) {
    // gprint("%d:%d idx: %d total: %d", bidx, tid, idx, total_bytes);
    NT val = LOAD(src);
    STORE(val, (NT *__restrict__)(dst + idx));
  }
}

void TestFramework::run() {
  
  Slice s1{ .src = (const uint8_t *)src_bufs_[0].devPtr, 
            .bytes = (uint32_t)(src_bufs_[0].size() * sizeof(NT))};
  Slice s2{ .src = (const uint8_t *)src_bufs_[1].devPtr, 
            .bytes = (uint32_t)(src_bufs_[1].size() * sizeof(NT))};

  size_t total = ref_buf_.size(); // NOTE dst_buf_.size() is different !! (OOB)
  VLOG(0) << "Total concat elements: " << total;
  constexpr uint32_t BlockSz = 128;
  size_t nBlocks = (total + BlockSz - 1) / BlockSz;
  VLOG(0) << "Launching with " << nBlocks << " blocks";

  concat_naive_kernel<BlockSz, NT><<<nBlocks, BlockSz, 0, 0>>>
        (s1, s2, (uint8_t *)dst_buf_.devPtr, total * sizeof(NT));
  CHK(cudaDeviceSynchronize());
  CHK(cudaPeekAtLastError());
}
