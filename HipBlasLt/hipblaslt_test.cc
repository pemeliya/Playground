// Original program by Chen, Wen adapted

/******************** compile with : *************************************/
// hipcc -lhipblaslt -std=c++17 --offload-arch=gfx90a hipblaslt_test.cc -Wno-backslash-newline-escape
// HIPBLASLT_LOG_MASK=5 HIPBLASLT_LOG_LEVEL=5 ./a.out

#include <iostream>
#include <random>
#include <optional>
#include <fstream>

#include "common/common_utils.hpp"
#include "common/hipblaslt_gemm.hpp"

#define USE_MXFP8 1

#define LOG(x) std::cerr << x << std::endl
int main(int argc, char *argv[]) try
{
  // DeviceInit(0, true);
	int m = 12288, k = 4608, n = 1536, 
      batch_size = 1,
      mk = std::max(m, k), mn = std::max(m, n),
      nk = std::max(n, k);
  float alpha{1.0}, beta{1}, Xascale{1}, Xbscale{1};
#if USE_MXFP8
  using TypeA = hipblaslt_f8;
  using TypeB = hipblaslt_f8;
  using TypeC = float;
  using TypeD = float;
#else
  using TypeA = hip_bfloat16;
  using TypeB = hip_bfloat16;
  using TypeC = hip_bfloat16;
  using TypeD = hip_bfloat16;
#endif

  size_t extra = 0;
  HVector< TypeA > a(m * k * batch_size + extra);
  HVector< TypeB > b(n * k * batch_size + extra);
  HVector< TypeC > c(m * n * batch_size + extra);
  HVector< TypeD > d(m * n * batch_size + extra); 
  HVector< float> ascale(k), bscale(k);

  TypeD *bias = nullptr; // not used

  initRange(a.data(), 0.01, 0.002, m*k);
  initRange(b.data(), 0.0, 0.0, n*k);
  initRange(c.data(), 0.0, 0.0, m*n);
  initRange(ascale.data(), Xascale, 0.0, k);
  initRange(bscale.data(), Xbscale, 0.0, k);

  a.copyHToD();
  b.copyHToD();
  c.copyHToD();
  ascale.copyHToD();
  bscale.copyHToD();

  BlasLtGemm gemm;
  BlasLtGemm::Config cfg;

  cfg = BlasLtGemm::Config{
      .trans_a = HIPBLAS_OP_T,
      .trans_b = HIPBLAS_OP_N,
      .compute_type = HIPBLAS_COMPUTE_32F,
      .orderA = HIPBLASLT_ORDER_COL,
      .orderB = HIPBLASLT_ORDER_COL,
      .orderCD = HIPBLASLT_ORDER_COL,
      .m = m,
      .n = n,
      .k = k,
      .batch_size = batch_size,
      .epilogue = HIPBLASLT_EPILOGUE_DEFAULT,
      .max_algorithms = 128,
      .max_workspace_size = 67108864,
#if USE_MXFP8
      .scale_mode = BlasLtGemm::kBlockScaling,
#else
      .scale_mode = BlasLtGemm::kNone,
#endif
      .stream = 0,
    };

  auto plan = gemm.createPlan(a.devPtr, b.devPtr, c.devPtr, 
      bias, d.devPtr, alpha, beta, cfg);

  auto algos = gemm.getAlgorithms(plan, cfg, bias);
  VLOG(0) << "Total algos found " << algos.size();

    CHK(cudaMemset(d.devPtr, 0xCC, d.size()*sizeof(TypeD)));

  CU_BEGIN_TIMING(100);
    gemm.run(a.devPtr, b.devPtr, c.devPtr, bias,
      d.devPtr, alpha, beta, ascale.devPtr, bscale.devPtr,
      cfg, plan, algos[0]);
  
  CU_END_TIMING("gemm.run");
  d.copyDToH();
  
    // auto ptr = d.data();
    // for (int i = 0; i < m; i++) {
    //   for (int j = 0; j < n; j++, ptr++) {
    //     float z = (float)ptr[0];
    //     // check for unexpected values
    //     if (std::abs(z) != 0 || !std::isfinite(z)) {
    //       VLOG(0) << i << ',' << j << " wrong " << z;
    //     }
    //   } // for j
    // } // for i
  return 0;
}
catch(std::exception& ex) {
  VLOG(0) << "Exception: " << ex.what();
}
catch(...) {
  VLOG(0) << "Unknown exception";
}