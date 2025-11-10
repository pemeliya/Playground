
#include <iostream>
#include <random>
#include <optional>
#include <fstream>

#include "common/common_utils.hpp"
#include "common/hipblaslt_gemm.hpp"

template <typename T>
using Xvec = HVector<T>;

int main(int argc, char *argv[]) try
{
  // DeviceInit(0, true);
	size_t m = 262144, n = 131072, k = 1024, batch_size = 1,
      mk = std::max(m, k), mn = std::max(m, n),
      nk = std::max(n, k);
  float alpha{1.0}, beta{1}, Xascale{1}, Xbscale{1};

  using TypeA = hip_bfloat16;
  using TypeB = hip_bfloat16;
  using TypeC = hip_bfloat16;
  using TypeD = hip_bfloat16;

  size_t extra = 0;
  Xvec< TypeA > a(m * k * batch_size + extra);
  Xvec< TypeB > b(n * k * batch_size + extra);
  HVector< TypeC > c(1); //m * n * batch_size + extra);
  Xvec< TypeD > d(m * n * batch_size + extra, false); 
  Xvec< TypeD > bias(m + extra);
  Xvec< float> ascale(4), bscale(4);

  TypeD *biasPtr = nullptr; //bias.devPtr;

  initRange(a.data(), 0.01, 0.002, m*k);
  initRange(b.data(), 0.0, 0.0, n*k);
  initRange(bias.data(), 0.0, -0.15, m);
  
  initRange(ascale.data(), Xascale, 0.0, 4);
  initRange(bscale.data(), Xbscale, 0.0, 4);

  a.copyHToD();
  b.copyHToD();
  c.copyHToD();
  bias.copyHToD();

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
      .max_workspace_size = 76*1024*1024,
      .use_ascale = false,
      .use_bscale = false,
      .stream = 0,
    };

  VLOG(0) << "Running algorithm default";
  auto plan = gemm.createPlan(a.devPtr, b.devPtr, c.devPtr, biasPtr,
      d.devPtr, alpha, beta, cfg);

  auto algos = gemm.getAlgorithms(plan, cfg, biasPtr);
  VLOG(0) << "total algos found " << algos.size();

  for(int ii = 0; ii < algos.size(); ii++) {
    VLOG(0) << "algo: " << ii;

  gemm.run(a.devPtr, b.devPtr, c.devPtr, biasPtr,
      d.devPtr, alpha, beta, 
      ascale.devPtr, bscale.devPtr,
      cfg, plan, algos[ii]);

  } // for ii

  return 0;
}
catch(std::exception& ex) {
  VLOG(0) << "Exception: " << ex.what();
}
catch(...) {
  VLOG(0) << "Unknown exception";
}