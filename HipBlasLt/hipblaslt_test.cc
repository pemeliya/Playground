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

template <typename T, typename U = T, typename V = U>
void matMatMultMixPrecEnumerate(T alpha, T beta, int m, int n, int k,
    U* A, U* B, V* C, V* D, V *bias, int bitfield) {
  
  int bit = bitfield & 1;
  int As1 = bit ? k : 1, As2 = bit ? 1 : m;

  bit = bitfield & 2;
  int Bs1 = bit ? n : 1, Bs2 = bit ? 1 : k;

  bit = bitfield & 4;
  int Cs1 = bit ? n : 1, Cs2 = bit ? 1 : m;

  bit = bitfield & 8;
  int Ds1 = bit ? n : 1, Ds2 = bit ? 1 : m;

  VLOG(std::hex << "bits: 0x" << bitfield << std::dec <<
        " m = " << m << " n = " << n << " k = " << k <<
        "\nAs: " << As1 << "," << As2 << 
        "\nBs: " << Bs1 << "," << Bs2 << 
        "\nCs: " << Cs1 << "," << Cs2 << 
        "\nDs: " << Ds1 << "," << Ds2);

  matMatMultMixPrec(alpha, beta, m, n, k,
    A, As1, As2, B, Bs1, Bs2,
    C, Cs1, Cs2, D, Ds1, Ds2, bias);
}

int main(int argc, char *argv[]) try
{
	int m = 40, n = 20, k = 30;
  float alpha{1.0}, beta{0.0};

#if 1
  using TypeA = _Float16;
  using TypeB = _Float16;
  using TypeC = _Float16;
  using TypeD = _Float16;
#else
  using TypeA = float;
  using TypeB = float;
  using TypeC = float;
  using TypeD = float;
#endif

  size_t extra = 0, extra2 = 16*1024*1024;
  HVector< TypeA > a(m * k + extra);
  HVector< TypeB > b(n * k + extra);
  HVector< TypeC > c(4 + extra);
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
  //populateRandomVec(bias, float{});
  initRange(bias.data(), 1.0, 0.0, m);
#endif

  initRange(d1.data(), 0.0, 0.0, m*n);
  initRange(d2.data(), 0.0, 0.0, m*n);

  a.copyHToD();
  b.copyHToD();
  c.copyHToD();
  bias.copyHToD();

  BlasLtGemm gemm;
  BlasLtGemm::Config cfg{
    .trans_a = HIPBLAS_OP_N,
    .trans_b = HIPBLAS_OP_N,
    .compute_type = HIPBLAS_COMPUTE_32F_FAST_TF32,
    .m = m,
    .n = n,
    .k = k,
    .epilogue = HIPBLASLT_EPILOGUE_DEFAULT,
    .max_algorithms = 2,
    .max_workspace_size = 1ull << 20,
    .stream = 0,
  };

  VLOG("Running algorithm default");
  auto plan = gemm.createPlan(a.devPtr, b.devPtr, c.devPtr, bias.devPtr,
      d1.devPtr, alpha, beta, cfg);

  auto algos = gemm.getAlgorithms(plan, cfg, bias.devPtr);

#if 0
  gemm.run(a.devPtr, b.devPtr, c.devPtr, bias.data(),
      d1.devPtr, alpha, beta, cfg, plan, algos[0]);
  d1.copyDToH();
#else
  matMatMultMixPrec(alpha, beta, m, n, k,
    a.data(), 1, m, // m x k: (k, 1) or (1, m)
    b.data(), 1, k, // n x k: (n, 1) or (1, k)
    c.data(), 1, m, // does not matter
    d1.data(), 1, m, 
    cfg.epilogue == HIPBLASLT_EPILOGUE_DEFAULT ? nullptr : bias.data()); // m x n: (n, 1)  or (1, m)
#endif

  using ZT = double;
  auto check_results = [&](const auto& truth, const auto& test, auto tolerance) {
    int nfailed = 0;
    for(size_t i = 0; i < m*n; i++) {
      auto v1 = (ZT)truth[i];
      auto v2 = (ZT)test[i];
      if(!(std::isfinite(v1) == std::isfinite(v2) &&
          std::abs(v1 - v2) /
              (std::max(std::abs(v1), std::abs(v2)) + 1) < tolerance)) {
        nfailed++;
        if(nfailed < 10) {
          VLOG(i << ": truth: " << v1 << " gpu: " << v2 << " diff: " << (v1 - v2));
        } else 
          return false;
      }
    }
    return nfailed == 0;
  };

  float tolerance = 0.01f;
  uint32_t totalFailed = 0, N = algos.size();
  for(uint32_t i = 0; i < N; i++) {
#if 1
    gemm.run(a.devPtr, b.devPtr, c.devPtr, bias.devPtr,
      d2.devPtr, alpha, beta, cfg, plan, algos[i]);
    d2.copyDToH();
#else
  matMatMultMixPrecEnumerate(alpha, beta, m, n, k,
    a.data(), b.data(), c.data(), d2.data(), bias.data(), i);
#endif

    auto OK = check_results(d1, d2, tolerance);
    auto [index, fallback] = gemm.getAlgoIndex(algos[i]);
    if(!OK) {
      VLOG(i << ": algorithm " << index << " accuracy mismatch vs CPU algorithm!");
      totalFailed++;
    } else {
      VLOG(i << ": algorithm " << index << " OK");
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