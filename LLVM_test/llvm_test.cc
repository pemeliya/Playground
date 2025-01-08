
#include "llvm_test.h"

TestFramework::TestFramework(const std::vector< size_t >& offsets) : 
        concat_sizes_(offsets) {
  
  // NOTE: concat_sizes_[0] is always 0 !!!
  src_bufs_.reserve(concat_sizes_.size());
  size_t total = 0;
  for (auto sz : concat_sizes_) {
    src_bufs_.push_back(Vector(sz));
    total += sz;
  }
  VLOG(0) << "Allocating device buf of " << total * sizeof(NT) << " bytes";
  dst_buf_ = Vector(total + s_redzoneElems);
  ref_buf_.resize(total);
}

TestFramework::~TestFramework() {}

void TestFramework::initialize_bufs() {
  
  NT z = 1;
  auto ptr = ref_buf_.begin();
  for (auto& buf : src_bufs_) {
    for (size_t i = 0; i < buf.size(); i++) {
        NT x(i + 1);
        buf[i] = x*x - z;
    }
    std::copy(buf.begin(), buf.end(), ptr);
    z++, ptr += buf.size();
    buf.copyHToD();  // copy to device
  }
  // guard for OOB detection
  cudaMemset(dst_buf_.devPtr + ref_buf_.size(), 
                s_oobValue, s_redzoneElems*sizeof(NT));
}

void TestFramework::verify() {
  dst_buf_.copyDToH();

  checkme< false >(dst_buf_.data(), ref_buf_.data(), 
        ref_buf_.size(), ref_buf_.size(), 1,
        NT(1e-10), 
        /*print_when_differs*/true, 
        /*print_max*/1000);

  auto ptr = (const uint8_t *)(dst_buf_.data() + ref_buf_.size());
  for (uint32_t i = 0; i < s_redzoneElems; i++) {
    if 
  }
}

int main() try {

    DeviceInit();
    // these are the sizes of partitions!
    TestFramework test({100, 200});
    test.initialize_bufs();
    test.run();
    test.verify();

    return 0;
}
catch(std::exception& ex) {
  VLOG(0) << 0) << "Exception: " << ex.what(;
  return 1;
}
catch(...) {
  VLOG(0) << "Unhandled exception";
  return 1;
}
