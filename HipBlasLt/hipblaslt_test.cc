// Original program by Chen, Wen adapted

/******************** compile with : *************************************/
// hipcc -lhipblaslt -std=c++17 --offload-arch=gfx90a hipblaslt_test.cc -Wno-backslash-newline-escape
// HIPBLASLT_LOG_MASK=5 HIPBLASLT_LOG_LEVEL=5 ./a.out

#include <hip/hip_fp16.h>
#include <hip/hip_complex.h>
#include <iostream>
#include <random>
#include <optional>

#include "common/example_utils.hpp"
#include "common/hipblaslt_gemm.hpp"

#define LOG(x) std::cerr << x << std::endl

#define CHK_HIP(error) if(error != hipSuccess) { \
        fprintf(stderr, "Hip error: '%s'(%d) at %s:%d\n", hipGetErrorString(error),  \
                error, __FILE__, __LINE__); throw 0;  \
    }

template < class Float, class ComputeT > 
void populateRandomVec(std::vector< Float >& vec, ComputeT) {

  std::uniform_real_distribution<ComputeT> generator(-0.1f, 0.2f);
  std::minstd_rand0 engine;
  for(auto& v : vec) {
    v = static_cast< Float >(generator(engine));
  }
}

int main(int argc, char *argv[]) try
{
	int m = 768, n = 4096, k = 6144;
  float alpha{1.0}, beta{0.0};

#if 1
  using TypeA = hip_bfloat16;
  using TypeB = hip_bfloat16;
  using TypeC = hip_bfloat16;
  using TypeD = hip_bfloat16;
#else
  using TypeA = float;
  using TypeB = float;
  using TypeC = float;
  using TypeD = float;
#endif

  size_t extra = 0, extra2 = 16*1024*1024;
  HVector< TypeA > a(m * k + extra);
  HVector< TypeB > b(n * k + extra);
  HVector< TypeC > c(m * n + extra);
  HVector< TypeD > d1(m * n + extra2),
                   d2(m * n + extra2); 
  HVector< TypeD > bias(m + extra);

#if 0
  initRange(a.data(), 0.0, 0.02, m*k);
  initRange(b.data(), 10.0, -0.01, n*k);
  initRange(c.data(), 0.0, -0.005, m*n);
  initRange(bias.data(), 0.0, -0.15, m);
#else
  populateRandomVec(a, float{});
  populateRandomVec(b, float{});
  populateRandomVec(c, float{});
  populateRandomVec(bias, float{});
#endif

  a.copyHToD();
  b.copyHToD();
  c.copyHToD();
  bias.copyHToD();

  BlasLtGemm gemm;
  BlasLtGemm::Config cfg{
    .trans_a = HIPBLAS_OP_T,
    .trans_b = HIPBLAS_OP_N,
    .compute_type = HIPBLAS_COMPUTE_32F,
    .m = m,
    .n = n,
    .k = k,
    .epilogue = HIPBLASLT_EPILOGUE_DEFAULT,
    .max_algorithms = 512,
    .max_workspace_size = 1ull << 32,
    .stream = 0,
  };

  VLOG("Running algorithm default");
  auto plan = gemm.createPlan(a.devPtr, b.devPtr, c.devPtr, bias.devPtr,
      d1.devPtr, alpha, beta, cfg);

  auto algos = gemm.getAlgorithms(plan, cfg);

#if 0
  gemm.run(a.devPtr, b.devPtr, c.devPtr, bias.devPtr,
      d1.devPtr, alpha, beta, cfg, plan, algos[0]);
  d1.copyDToH();
#else
  matMatMultMixPrec(alpha, beta, m, n, k,
    a.data(), k, 1, // m x k: (k, 1) or (1, m)
    b.data(), 1, k, // n x k: (n, 1) or (1, k)
    c.data(), 1, 1, // does not matter
    d1.data(), 1, m); // m x n: (n, 1)  or (1, m)
#endif

  auto check_results = [&](const auto& truth, const auto& test, auto tolerance) {
    for(size_t i = 0; i < m*n; i++) {
      auto v1 = truth[i];
      auto v2 = test[i];
      if(!(std::isfinite(v1) == std::isfinite(v2) &&
          std::abs(v1 - v2) /
              (std::max(std::abs(v1), std::abs(v2)) + 1) < tolerance)) {
        return false;
      }
    }
    return true;
  };

  float tolerance = 0.01f;
  uint32_t totalFailed = 0;
  for(uint32_t i = 0; i < algos.size(); i++) {
    gemm.run(a.devPtr, b.devPtr, c.devPtr, bias.devPtr,
      d2.devPtr, alpha, beta, cfg, plan, algos[i]);
    d2.copyDToH();

    auto OK = check_results(d1, d2, tolerance);
    if(!OK) {
      auto [index, fallback] = gemm.getAlgoIndex(algos[i]);
      VLOG(i << ": algorithm " << index << " accuracy mismatch vs CPU algorithm!");
      totalFailed++;
    }
  }
  VLOG("Accuracy check failed for " << totalFailed << " out of " 
      << algos.size() << " algorithms!");
  return 0;
}
catch(std::exception& ex) {
  LOG("Exception: " << ex.what());
}
catch(...) {
  LOG("Unknown exception");
}