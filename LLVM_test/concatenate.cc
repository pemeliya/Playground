

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
  union {
    uint32_t total;
    uint32_t col;   // just another name for the field
  };
};

template < class NT >
struct XSlice {
  const NT *src;
  uint32_t elems_per_row;
  uint32_t idx_new;
  union {
    uint32_t total;
    uint32_t col;   // just another name for the field
  };
};

  // [3,1] + [3,4] + [3,3] = [3,7]
  // C   AAAA   BBB   CAAAABBB
  // C + AAAA + BBB = CAAAABBB
  // C   AAAA   BBB   CAAAABBB

template < class NT, class TSlice, class ...Slices >
__device__ XSlice<NT> concat_seq_load_block(const uint32_t idx,
            const uint32_t col_ofs, TSlice s, Slices... rest) {

  static_assert(std::is_same_v<TSlice, Slice<NT>>,
      "Slices must be of predefined type!");

  if constexpr(sizeof...(Slices) == 0) {
    return XSlice<NT>{ s.src, s.elems_per_row, idx, {col_ofs} };
  } else {
    const uint32_t idx2 = idx - s.total;
    if ((int32_t)idx2 < 0) {
      return XSlice<NT>{ s.src, s.elems_per_row, idx, {col_ofs} };
    }
    return concat_seq_load_block<NT>(idx2, col_ofs + s.elems_per_row, rest...);
  }
}

template < uint32_t BlockSz, class NT, class ...Slices >
__launch_bounds__(BlockSz, 4)
__global__ void concat_naive_seq_load( 
            const uint32_t elems_per_row, 
            const uint32_t total_elems, 
            NT * __restrict__ g_dst, Slices... rest) { 
  
  const uint32_t bidx = blockIdx.x, tid = threadIdx.x;
  const uint32_t idx = bidx * BlockSz + tid;
  if (idx >= total_elems) return;

  // trying sequential load, strided store..
  auto S = concat_seq_load_block<NT>(idx, 0, rest...);
  auto *src = (const NT * __restrict__)(S.src + S.idx_new);
  NT val = LOAD(src);
  uint32_t row = S.idx_new / S.elems_per_row,
           col = S.idx_new - row * S.elems_per_row;
  auto ptr = g_dst + row*elems_per_row + col + S.col;
  STORE(val, ptr);
}


template < class NT, class TSlice, class ...Slices >
__device__ Slice<NT> concat_seq_store_block(const uint32_t col,
            TSlice s, Slices... rest) {

  static_assert(std::is_same_v<TSlice, Slice<NT>>,
      "Slices must be of predefined type!");

  if constexpr(sizeof...(Slices) == 0) {
    return Slice<NT>{ s.src, s.elems_per_row, {col} };
  } else {
    uint32_t col2 = col - s.elems_per_row;
    if ((int32_t)col2 < 0) {
       return Slice<NT>{ s.src, s.elems_per_row, {col} };
    } 
    return concat_seq_store_block<NT>(col2, rest...);
  }
}

template < uint32_t BlockSz, class NT, class ...Slices >
__launch_bounds__(BlockSz, 4)
__global__ void concat_naive_seq_store( 
            const uint32_t elems_per_row, 
            const uint32_t total_elems, 
            NT * __restrict__ g_dst, Slices... rest) { 
  
  const uint32_t bidx = blockIdx.x, tid = threadIdx.x;
  const uint32_t idx = bidx * BlockSz + tid;

  if (idx >= total_elems) return;
  const uint32_t row = idx / elems_per_row,
           col = idx - row*elems_per_row;

  auto S = concat_seq_store_block<NT>(col, rest...);
  auto *src = (const NT * __restrict__)
                    (S.src + row*S.elems_per_row + S.col);
  NT val = LOAD(src);
  STORE(val, g_dst + idx);
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

#if 1
  CU_BEGIN_TIMING(5)
    concat_naive_seq_load<BlockSz><<<nBlocks, BlockSz, 0, 0>>>
        (concat_num_cols_, total, dst_buf_.devPtr, 
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
        slices[1],
        slices[2]
        );
  CU_END_TIMING("Naive seq-store kernel: #total elems: %zu; "
    "#blocks: %zu; #threads: %zu", total, nBlocks, BlockSz);
#endif

  CHK(cudaDeviceSynchronize());
  CHK(cudaPeekAtLastError());
}
