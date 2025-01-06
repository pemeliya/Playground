
#include "llvm_test.h"

TestFramework::TestFramework(const std::vector< size_t >& offsets) : 
        concat_ofs_(offsets) {
  
  // NOTE: concat_ofs_[0] is always 0 !!!
  src_bufs_.reserve(concat_ofs_.size());
  size_t total = 0, prev = 0;
  for(auto ofs : concat_ofs_) {
    src_bufs_.push_back(Vector(ofs - prev));
    total += ofs, prev = ofs;
  }
  VLOG(0) << "Allocating device buf of " << total * sizeof(NT) << " bytes";
  dst_buf_ = Vector(total);
}

TestFramework::~TestFramework() {}

int main() try {

    DeviceInit();
    TestFramework test({100, 200, 300});

    return 0;
}
catch(std::exception& ex) {
  VLOG(0) << "Exception: " << ex.what();
  return 1;
}
catch(...) {
  VLOG(0) << "Unhandled exception";
  return 1;
}
