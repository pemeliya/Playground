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

  VLOG(0) << std::hex << "bits: 0x" << bitfield << std::dec <<
        " m = " << m << " n = " << n << " k = " << k <<
        "\nAs: " << As1 << "," << As2 << 
        "\nBs: " << Bs1 << "," << Bs2 << 
        "\nCs: " << Cs1 << "," << Cs2 << 
        "\nDs: " << Ds1 << "," << Ds2;

  matMatMultMixPrec(alpha, beta, m, n, k,
    A, As1, As2, B, Bs1, Bs2,
    C, Cs1, Cs2, D, Ds1, Ds2, bias);
}

template < class TypeA, class TypeB, class TypeC, class TypeD, class Scalar >
void benchmark(const TypeA *dA, const TypeB *dB, 
      const TypeC *dC, const TypeD *dBias,
      TypeD *dD, Scalar alpha, Scalar beta, const BlasLtGemm::Config& cfg) {
  
  BlasLtGemm gemm;
  auto plan = gemm.createPlan(dA, dB, dC, dBias, dD, alpha, beta, cfg);
  auto algos = gemm.getAlgorithms(plan, cfg, dBias);
  if (algos.empty()) throw std::runtime_error("No algorithms found!!");

  auto t2s = [](auto t){ return t == HIPBLAS_OP_N ? "N" : "T"; };
  auto o2s = [](auto o){ return o == HIPBLASLT_ORDER_ROW ? "R" : "C"; };

  size_t n_warmups = 2, n_runs = 10;
  for (size_t i = 0; i < n_warmups; i++) {
    gemm.run(dA, dB, dC, dBias, dD, alpha, beta, cfg, plan, algos[0]);
  }
  cudaStreamSynchronize(cfg.stream);

  CPU_BEGIN_TIMING(GEMM);
  for (size_t i = 0; i < n_runs; i++) {
    gemm.run(dA, dB, dC, dBias, dD, alpha, beta, cfg, plan, algos[0]);
  }
  cudaStreamSynchronize(cfg.stream);
  CPU_END_TIMING(GEMM, n_runs, "%ld x %ld x %ld batch %ld (%s,%s -> %s) trans: %s/%s order: %s/%s/%s",
        cfg.m, cfg.n, cfg.k, cfg.batch_size,
        HipBlasltStr(dA), HipBlasltStr(dB), HipBlasltStr(dD),
        t2s(cfg.trans_a), t2s(cfg.trans_b),
        o2s(cfg.orderA), o2s(cfg.orderB), o2s(cfg.orderCD));
}

int main(int argc, char *argv[]) try
{
	int m = 8192, n = 32768, k = 1024, batch_size = 1,
      mk = std::max(m, k), mn = std::max(m, n),
      nk = std::max(n, k);
  float alpha{1.0}, beta{1.0};

#if 1
  using TypeA = hipblaslt_f8;
  using TypeB = hipblaslt_bf8;
  using TypeC = hip_bfloat16;
  using TypeD = hip_bfloat16;
#else
  using TypeA = float;
  using TypeB = float;
  using TypeC = float;
  using TypeD = float;
#endif

  size_t extra = 0, extra2 = 16;
  HVector< TypeA > a(m * k * batch_size + extra);
  HVector< TypeB > b(n * k * batch_size + extra);
  HVector< TypeC > c(m * n * batch_size + extra);
  HVector< TypeD > d(m * n * batch_size + extra); 
  HVector< TypeD > bias(m + extra);
  HVector< float> ascale(4), bscale(4);

#if 1
  initRange(a.data(), 0.01, 0.002, m*k);
  initRange(b.data(), 0.0, 0.0, n*k);
  initRange(c.data(), 0.0, 0.0, m*n);
  initRange(bias.data(), 0.0, -0.15, m);
  
  initRange(ascale.data(), 1.0, 0.0, 4);
  initRange(bscale.data(), 1.0, 0.0, 4);
#else
  populateRandomVec(a, float{});
  populateRandomVec(b, float{});
  populateRandomVec(c, float{});
  //populateRandomVec(bias, float{});
  initRange(bias.data(), 1.0, 0.0, m);
#endif

  std::ifstream ifs("/tf/xla/matrixb.bin", std::ios::in | std::ios::binary);
  if (!ifs) throw std::runtime_error("Unable to open file for reading!");

  ifs.read((char*)a.data(), m*k);

  a.copyHToD();
  b.copyHToD();
  c.copyHToD();
  ascale.copyHToD();
  bscale.copyHToD();
  bias.copyHToD();

  BlasLtGemm gemm;
  BlasLtGemm::Config cfg;

  cfg = BlasLtGemm::Config{
      .trans_a = HIPBLAS_OP_T,
      .trans_b = HIPBLAS_OP_N,
      .compute_type = HIPBLAS_COMPUTE_32F,
      .orderA = HIPBLASLT_ORDER_ROW,
      .orderB = HIPBLASLT_ORDER_COL,
      .orderCD = HIPBLASLT_ORDER_COL,
      .m = m,
      .n = n,
      .k = k,
      .batch_size = batch_size,
      .epilogue = HIPBLASLT_EPILOGUE_DEFAULT,
      .max_algorithms = 16,
      .max_workspace_size = 67108864,
      .stream = 0,
    };

#if 1
  VLOG(0) << "Running algorithm default";
  auto plan = gemm.createPlan(a.devPtr, b.devPtr, c.devPtr, bias.devPtr,
      d.devPtr, alpha, beta, cfg);

  auto algos = gemm.getAlgorithms(plan, cfg, bias.devPtr);
  VLOG(0) << "total algos found " << algos.size();

#if 1
  gemm.run(a.devPtr, b.devPtr, c.devPtr, bias.data(),
      d.devPtr, alpha, beta, 
      ascale.devPtr, bscale.devPtr,
      cfg, plan, algos[0]);
  d.copyDToH();
#else
  matMatMultMixPrec(alpha, beta, m, n, k,
    a.data(), 1, m, // m x k: (k, 1) or (1, m)
    b.data(), 1, k, // n x k: (n, 1) or (1, k)
    c.data(), 1, m, // does not matter
    d.data(), 1, m, 
    bias.data()); // m x n: (n, 1)  or (1, m)
#endif

  auto check_results = [&](const auto& truth, const auto& test, auto tolerance) {
    int nfailed = 0;
    for(size_t i = 0; i < m*n; i++) {
      auto v1 = truth[i];
      auto v2 = test[i];
      if(!(std::isfinite(v1) == std::isfinite(v2) &&
          std::abs(v1 - v2) /
              (std::max(std::abs(v1), std::abs(v2)) + 1) < tolerance)) {
        nfailed++;
        if(nfailed < 10) {
          VLOG(0) << i << ": truth: " << v1 << " gpu: " << v2 << " diff: " << (v1 - v2);
        } else 
          return false;
      }
    }
    return nfailed == 0;
  };

  for (int i = 0; i < d.size(); i++) {
    float z = (float)d[i];
    if (std::abs(z) > 1e5) {
      VLOG(0) << i << " -- " << z;
    }
  }

//   float tolerance = 0.01f;
//   uint32_t totalFailed = 0, N = algos.size();
//   for(uint32_t i = 0; i < N; i++) {
// #if 0
//     gemm.run(a.devPtr, b.devPtr, c.devPtr, bias.devPtr,
//       d2.devPtr, alpha, beta, cfg, plan, algos[i]);
//     d2.copyDToH();
// #else
//   matMatMultMixPrecEnumerate(alpha, beta, m, n, k,
//     a.data(), b.data(), c.data(), d2.data(), bias.data(), i);
// #endif

//     auto OK = check_results(d1, d2, tolerance);
//     auto [index, fallback] = gemm.getAlgoIndex(algos[i]);
//     if(!OK) {
//       VLOG(0) << i << ": algorithm " << index << " accuracy mismatch vs CPU algorithm!";
//       totalFailed++;
//     } else {
//       VLOG(0) << i << ": algorithm " << index << " OK";
//     }
//   }
//   VLOG(0) << "Accuracy check failed for " << totalFailed << " out of " 
//       << algos.size() << " algorithms!";
#endif
  return 0;
}
catch(std::exception& ex) {
  LOG("Exception: " << ex.what());
}
catch(...) {
  LOG("Unknown exception");
}