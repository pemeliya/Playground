

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
template < class NT >
struct Slice {
  const NT *src;
  uint32_t elems_per_row;
  uint32_t total;
};

  // [3,1] + [3,4] + [3,3] = [3,7]
  // C   AAAA   BBB   CAAAABBB
  // C + AAAA + BBB = CAAAABBB
  // C   AAAA   BBB   CAAAABBB

template < class NT, class TSlice, class ...Slices >
__device__ void naive_kernel_block(const uint32_t idx,
            const uint32_t col_ofs,
            const uint32_t elems_per_row, 
            NT * __restrict__ g_dst, TSlice s1, Slices... rest) {

  static_assert(std::is_same_v<TSlice, Slice<NT>>,
      "Slices must be of predefined type!");

  const uint32_t idx2 = idx - s1.total;
  if ((int32_t)idx2 < 0) {
    auto *src = (const NT * __restrict__)(s1.src + idx);
    NT val = LOAD(src);
    uint32_t row = idx / s1.elems_per_row,
             col = idx - row * s1.elems_per_row;
    auto ptr = g_dst + row*elems_per_row + col + col_ofs;
    STORE(val, ptr);
    return;
  }
  if constexpr(sizeof...(Slices) != 0) {
    naive_kernel_block(idx2, col_ofs + s1.elems_per_row, 
            elems_per_row, g_dst, rest...);
  }
}

template < uint32_t BlockSz, class NT, class ...Slices >
__launch_bounds__(BlockSz, 4)
__global__ void concat_naive_seq_load( 
            const uint32_t elems_per_row, 
            NT * __restrict__ g_dst, Slices... rest) { 
  
  const uint32_t bidx = blockIdx.x, tid = threadIdx.x;
  const uint32_t idx = bidx * BlockSz + tid;

  // trying sequential load, strided store..
  naive_kernel_block(idx, 0, elems_per_row, g_dst, rest...);
}


// template < class NT, class TSlice, class ...Slices >
// __kernel__ void concat_seq_store_block(const uint32_t idx,
//             const uint32_t col_ofs,
//             const uint32_t elems_per_row, 
//             NT * __restrict__ g_dst, TSlice s1, Slices... rest) {

//   static_assert(std::is_same_v<TSlice, Slice<NT>>,
//       "Slices must be of predefined type!");

//   const uint32_t idx2 = idx - s1.total;
//   if ((int32_t)idx2 < 0) {
//     auto *src = (const NT * __restrict__)(s1.src + idx);
//     NT val = LOAD(src);
//     uint32_t row = idx / s1.elems_per_row,
//              col = idx - row * s1.elems_per_row;
//     auto ptr = g_dst + row*elems_per_row + col + col_ofs;
//     STORE(val, ptr);
//     return;
//   }
//   if constexpr(sizeof...(Slices) != 0) {
//     naive_kernel_block(idx2, col_ofs + s1.elems_per_row, 
//             elems_per_row, g_dst, rest...);
//   }
// }

  // [3,1] + [3,4] + [3,3] = [3,7]
  // C   AAAA   BBB   CAAAABBB
  // C + AAAA + BBB = CAAAABBB
  // C   AAAA   BBB   CAAAABBB

template < uint32_t BlockSz, class NT, class ...Slices >
__launch_bounds__(BlockSz, 4)
__global__ void concat_naive_seq_store( 
            const uint32_t elems_per_row, 
            const uint32_t total_elems, 
            NT * __restrict__ g_dst, Slice<NT> s1, Slice<NT> s2) { 
  
  const uint32_t bidx = blockIdx.x, tid = threadIdx.x;
  const uint32_t idx = bidx * BlockSz + tid;

  if (idx >= total_elems) return;
  const uint32_t row = idx / elems_per_row,
           col = idx - row*elems_per_row;

  const NT * __restrict__ src = nullptr;
  uint32_t col2 = col - s1.elems_per_row;
  if ((int32_t)col2 < 0) {
    src = decltype(src)(s1.src + row*s1.elems_per_row + col);
  } else {
    uint32_t col3 = col2 - s2.elems_per_row;
    if ((int32_t)col3 < 0) {
      src = decltype(src)(s2.src + row*s2.elems_per_row + col2);
    }
  }
  if (src != nullptr) {
    NT val = LOAD(src);
    STORE(val, g_dst + idx);
  }
}


void TestFramework::run_naive_concat() {
  
  std::vector< Slice<NT> > slices(concat_sizes_.size());
  for (size_t i = 0; i < slices.size(); i++) {
    slices[i].src = src_bufs_[i].devPtr;
    slices[i].elems_per_row = concat_sizes_[i];
    slices[i].total = src_bufs_[i].size();
  }

  size_t total = ref_buf_.size(); // NOTE dst_buf_.size() is different !! (OOB)
  constexpr size_t BlockSz = 128;
  size_t nBlocks = (total + BlockSz - 1) / BlockSz;
  clean_output_buf();

#if 0 
  CU_BEGIN_TIMING(5)
    concat_naive_kernel<BlockSz><<<nBlocks, BlockSz, 0, 0>>>
        (concat_num_cols_, dst_buf_.devPtr, 
        slices[0], 
        slices[1],
        slices[2]
        );
  CU_END_TIMING("Naive seq-load kernel: #total elems: %zu; "
    "#blocks: %zu; #threads: %zu", total, nBlocks, BlockSz);
#else
  CU_BEGIN_TIMING(5)
    concat_naive_seq_store<BlockSz><<<nBlocks, BlockSz, 0, 0>>>
        (concat_num_cols_, total,
        dst_buf_.devPtr, 
        slices[0], 
        slices[1]);
  CU_END_TIMING("Naive seq-store kernel: #total elems: %zu; "
    "#blocks: %zu; #threads: %zu", total, nBlocks, BlockSz);
#endif

  CHK(cudaDeviceSynchronize());
  CHK(cudaPeekAtLastError());
}
