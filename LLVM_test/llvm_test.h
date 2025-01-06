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
  constexpr static uint32_t s_redzoneElems = 64; // number of OOB elements for redzone check

  TestFramework(const std::vector< size_t >& ofs);
  ~TestFramework();

  void initialize_bufs();
  void run();

private:
  std::vector< size_t > concat_sizes_;    // concatenate offsets
  std::vector< Vector > src_bufs_;
  Vector dst_buf_;
  std::vector< NT > ref_buf_; // reference solution
};

#endif // LLVM_TEST_H
