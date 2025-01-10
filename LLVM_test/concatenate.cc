

#include "llvm_test.h"

#define GLOBAL __attribute__((address_space(1)))

// NOTE: if data size is small, maybe makes sense to use just normal load/store?
#if 1
#define LOAD(addr) __builtin_nontemporal_load(addr)
#else
#define LOAD(addr) (addr)[0]
#endif
// it seems that loading with cache and storing without it gives the best results
#if 0
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

// either load sequential but store with strides
// or load with strides and srore sequential
struct Slice {
  const uint8_t *src;
  uint32_t bytes_per_row;
  uint32_t total_bytes;
};

  // [3,1] + [3,4] + [3,3] = [3,7]
  // C   AAAA   BBB   CAAAABBB
  // C + AAAA + BBB = CAAAABBB
  // C   AAAA   BBB   CAAAABBB

template < uint32_t BlockSz, class NT >
__launch_bounds__(BlockSz, 1)
__global__ void concat_naive_kernel(Slice s1, Slice s2, 
            uint8_t * __restrict__ g_dst,
            const uint32_t bytes_per_row) { 
  
  const uint32_t bidx = blockIdx.x, tid = threadIdx.x;
  const uint32_t idx = (bidx * BlockSz + tid) * sizeof(NT),
        idx2 = idx - s1.total_bytes;

  // trying sequential load, strided store..
  const NT * __restrict__ src = nullptr;

  if ((int32_t)idx2 < 0) {
    src = decltype(src)(s1.src + idx);
    NT val = LOAD(src);
    uint32_t row = idx / s1.bytes_per_row,
             col = idx - row * s1.bytes_per_row;
    auto ptr = g_dst + row*bytes_per_row + col;
    STORE(val, (NT *__restrict__)ptr);
  }
  else if (idx2 < s2.total_bytes) {
    src = decltype(src)(s2.src + idx2);
    NT val = LOAD(src);
    uint32_t row = idx2 / s2.bytes_per_row,
             col = idx2 - row * s2.bytes_per_row;
    auto ptr = g_dst + row*bytes_per_row + col + s1.bytes_per_row;
    STORE(val, (NT *__restrict__)ptr);
  }
}

void TestFramework::run_naive_concat() {
  
  Slice s1{ .src = (const uint8_t *)src_bufs_[0].devPtr, 
            .bytes_per_row = (uint32_t)(concat_sizes_[0] * sizeof(NT)),
            .total_bytes = (uint32_t)(src_bufs_[0].size() * sizeof(NT))
            };
  Slice s2{ .src = (const uint8_t *)src_bufs_[1].devPtr, 
            .bytes_per_row = (uint32_t)(concat_sizes_[1] * sizeof(NT)),
            .total_bytes = (uint32_t)(src_bufs_[1].size() * sizeof(NT))
            };

  size_t total = ref_buf_.size(); // NOTE dst_buf_.size() is different !! (OOB)
  constexpr size_t BlockSz = 128;
  size_t nBlocks = (total + BlockSz - 1) / BlockSz;
  clean_output_buf();
 
  CU_BEGIN_TIMING(5)
    concat_naive_kernel<BlockSz, NT><<<nBlocks, BlockSz, 0, 0>>>
        (s1, s2, (uint8_t *)dst_buf_.devPtr, concat_num_cols_ * sizeof(NT));
  CU_END_TIMING("Naive concat kernel: #total elems: %zu; "
    "#blocks: %zu; #threads: %zu", total, nBlocks, BlockSz);

  CHK(cudaDeviceSynchronize());
  CHK(cudaPeekAtLastError());
}
