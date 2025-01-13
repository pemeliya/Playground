
#include "llvm_test.h"

TestFramework::TestFramework(size_t num_rows, const std::vector< size_t >& concat_cols) : 
        num_rows_(num_rows), concat_sizes_(concat_cols) {
  
  src_bufs_.reserve(concat_sizes_.size());
  concat_num_cols_ = 0;
  VLOG(0) << "Concatenating: ";
  for (auto sz : concat_sizes_) {
    VLOG(0) << '[' << num_rows_ << 'x' << sz << ']';
    src_bufs_.push_back(Vector(sz*num_rows_));
    concat_num_cols_ += sz;
  }
  auto total = concat_num_cols_ * num_rows_;
  double Mb = (double)(total * sizeof(NT)) / (1<<20);
  VLOG(0) << "Allocating output buf of " << Mb << " Mb";
  dst_buf_ = Vector(total + s_redzoneElems);
  ref_buf_.resize(total);
}

TestFramework::~TestFramework() {}

void TestFramework::initialize_bufs() {
  
  NT z = 1;
  for (auto& buf : src_bufs_) {
    for (size_t i = 0; i < buf.size(); i++) {
      NT x(i + 1);
      buf[i] = x*x - z*z/2;
    }
    z++, buf.copyHToD();  // copy to device
  }

  std::fill(ref_buf_.begin(), ref_buf_.end(), NT{-777777});
  auto ref_ptr = ref_buf_.begin();
  for (size_t s = 0; s < src_bufs_.size(); s++) {
    auto src = src_bufs_[s].begin();
    auto dst = ref_ptr;
    for (size_t i = 0; i < num_rows_; i++) {
      std::copy(src, src + concat_sizes_[s], dst);
      src += concat_sizes_[s];
      dst += concat_num_cols_;
    }
    ref_ptr += concat_sizes_[s];
  }
#if 0
  ref_ptr = ref_buf_.begin();
  VLOG(0) << "----------------------------- " << num_rows_ << 'x' << concat_num_cols_ << " -----------------------------";
  for (size_t i = 0; i < num_rows_; i++) {
    std::ostringstream os;
    for (size_t j = 0; j < concat_num_cols_; j++) {
      os << *ref_ptr++ << ',';
    }
    VLOG(0) << os.str();
  }
  VLOG(0) << "-------------------------------------------------------------------------";
#endif
}

void TestFramework::clean_output_buf() {

  CHK(cudaMemset(dst_buf_.devPtr, s_fillValue, ref_buf_.size()*sizeof(NT)));
  // guard for OOB detection
  CHK(cudaMemset(dst_buf_.devPtr + ref_buf_.size(), 
                s_oobValue, s_redzoneElems*sizeof(NT)));
}

void TestFramework::verify() {
  dst_buf_.copyDToH();
  checkme< false >(dst_buf_.data(), ref_buf_.data(), 
        //size_t width, size_t stride, size_t n_batches
        concat_num_cols_, concat_num_cols_, num_rows_,
        /*eps*/NT(1e-10), 
        /*print_when_differs*/true, 
        /*print_max*/1000);

  auto ptr = (const uint8_t *)(dst_buf_.data() + ref_buf_.size());
  for (uint32_t i = 0, num_errors = 0; i < s_redzoneElems; i++) {
    if (ptr[i] != s_oobValue && num_errors++ < 50) {
      VLOG(0) << i << " OOB error: 0x" << std::hex << (uint32_t)ptr[i];
    }
  }
}

int main() try {

    DeviceInit();
    TestFramework test(22220, {400, 700, 1111});
    test.initialize_bufs();
    test.run_naive_concat();
    test.verify();

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
