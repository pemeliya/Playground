#ifndef LLVM_TEST_H
#define LLVM_TEST_H 1

#include <cstdint>
#include <vector>
#include "common/common_utils.hpp"

struct TestFramework {

  using NT = float;
  using Vector = HVector< NT >;

  constexpr static uint32_t s_bogus = 0xFFFFFFFFu; // to catch uninitialized entries
  constexpr static uint8_t s_fillValue = 0xAA;
  constexpr static uint8_t s_oobValue = 0xDD;
  constexpr static uint32_t s_redzoneElems = 256; // number of OOB elements for redzone check

  TestFramework(size_t num_rows, const std::vector< size_t >& concat_cols);
  ~TestFramework();

  void initialize_bufs();
  void run_naive_concat();
  void verify();
  void clean_output_buf();

private:
  // concatenates shapes:
  // [num_rows_, concat_sizes_[0]]
  // [num_rows_, concat_sizes_[1]]
  // ...
  // into [num_rows_, sum(concat_sizes_)]
  // num_rows = 3, concat_sizes = {1, 4, 3}
  // C   AAAA   BBB   CAAAABBB
  // C + AAAA + BBB = CAAAABBB
  // C   AAAA   BBB   CAAAABBB
  size_t num_rows_, concat_num_cols_; 
  std::vector< size_t > concat_sizes_;    // concatenate offsets
  std::vector< Vector > src_bufs_;
  Vector dst_buf_;
  std::vector< NT > ref_buf_; // reference solution
};

#endif // LLVM_TEST_H
