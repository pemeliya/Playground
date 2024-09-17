
#ifndef HIPBLASLT_GEMM_HPP
#define HIPBLASLT_GEMM_HPP 1

#include <hip/hip_fp16.h>
#include <hip/hip_complex.h>
#include <iostream>
#include <optional>

#include "common/common.h"
#include <hipblaslt/hipblaslt.h>

#if HIPBLASLT_VERSION_MINOR < 6
#define hipDataType hipblasDatatype_t
#define HIP_R_16F HIPBLAS_R_16F
#define HIP_R_16BF HIPBLAS_R_16B
#define HIP_R_32F HIPBLAS_R_32F
#define HIP_R_64F HIPBLAS_R_64F
#define HIP_R_8I HIPBLAS_R_8I
#define HIP_R_32I HIPBLAS_R_32I
#define HIP_C_32F HIPBLAS_C_32F
#define HIP_C_64F HIPBLAS_C_64F
#define hipblasComputeType_t hipblasLtComputeType_t
#define HIPBLAS_COMPUTE_32F HIPBLASLT_COMPUTE_F32
#define HIPBLAS_COMPUTE_64F HIPBLASLT_COMPUTE_F64
#define HIPBLAS_COMPUTE_32I HIPBLASLT_COMPUTE_I32
#endif

#define MOVABLE_HANDLE(Class)           \
    Class(Class&) = delete;             \
    Class& operator=(Class&) = delete;  \
    Class(Class&& rhs) {                \
      *this = std::move(rhs);           \
    }                                   \
    Class& operator=(Class&& rhs) {     \
      std::swap(handle, rhs.handle);  \
      return *this;                     \
    }                                   \
    Class() = default

#define CHK_HIPBLASLT(error) if(error != HIPBLAS_STATUS_SUCCESS) { \
       fprintf(stderr, "hipBLASLt error %s at %s:%d\n", hipblasStatusToString(error), \
            __FILE__, __LINE__); throw 0;  \
    }

#define SET_ATTR(setter, handle, attr, value) \
  CHK_HIPBLASLT(setter(handle, attr, &value, sizeof(decltype(value))))

template <typename T>
void SetAttr(hipblasLtMatrixLayout_t handle,
                    hipblasLtMatrixLayoutAttribute_t attr, T value) {
  SET_ATTR(hipblasLtMatrixLayoutSetAttribute, handle, attr, value);
}

template <typename T>
void SetAttr(hipblasLtMatmulDesc_t handle,
                    hipblasLtMatmulDescAttributes_t attr, T value) {
  SET_ATTR(hipblasLtMatmulDescSetAttribute, handle, attr, value);
}
template <typename T>
void SetAttr(hipblasLtMatmulPreference_t handle,
                    hipblasLtMatmulPreferenceAttributes_t attr, T value) {
  SET_ATTR(hipblasLtMatmulPreferenceSetAttribute, handle, attr,
                  value);
}

inline std::ostream& operator<<(std::ostream& os, hipFloatComplex Z) {
  return os << Z.x << "+i*" << Z.y;
}

inline std::ostream& operator<<(std::ostream& os, hipDoubleComplex Z) {
  return os << Z.x << "+i*" << Z.y;
}

template < class T >
constexpr hipDataType HipBlasltType(const T *) {
  if constexpr (std::is_same_v<T, _Float16>) 
    return HIP_R_16F;
  if constexpr (std::is_same_v<T, __half>) 
    return HIP_R_16F;
  if constexpr (std::is_same_v<T, hip_bfloat16>) 
    return HIP_R_16BF;
  if constexpr (std::is_same_v<T, float>) 
    return HIP_R_32F;
  if constexpr (std::is_same_v<T, double>) 
    return HIP_R_64F;
  if constexpr (std::is_same_v<T, int32_t>) 
    return HIP_R_32I;
  if constexpr (std::is_same_v<T, int8_t>) 
    return HIP_R_8I;
  if constexpr (std::is_same_v<T, hipFloatComplex>) 
    return HIP_C_32F;
  if constexpr (std::is_same_v<T, hipDoubleComplex>) 
    return HIP_C_64F;
  
  return (hipDataType)-1;
}

struct HipMatrixLayout {

    enum class Order { kRowMajor, kColumnMajor };

    MOVABLE_HANDLE(HipMatrixLayout);

    HipMatrixLayout(hipDataType type, size_t num_rows, size_t num_cols, 
          Order order, size_t batch_size = 1,
          std::optional<int64_t> leading_dim_stride = std::nullopt,
          std::optional<int64_t> batch_stride = std::nullopt) 
    {
      if (!leading_dim_stride) {
        leading_dim_stride = (order == Order::kRowMajor) ? num_cols : num_rows;
      }
      CHK_HIPBLASLT(hipblasLtMatrixLayoutCreate(
        &handle, type, num_rows, num_cols, *leading_dim_stride));

      // Wrap hipblas handle immediately, so it is cleaned up if an error occurs.
      SetAttr(handle, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                              static_cast<int32_t>(batch_size));

      if (!batch_stride) {
        batch_stride = (batch_size > 1) ? num_rows * num_cols : 0;
      }
      VLOG("MatrixLayout type: " << (int)type
          << " rows: " << num_rows << " cols: " << num_cols
          << " batch_size: " << batch_size
          << " leading_dim_stride: " << *leading_dim_stride
          << " batch_stride: " << *batch_stride);

      SetAttr(handle, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, 
            *batch_stride);
    }
    ~HipMatrixLayout() {
      if(handle != hipblasLtMatrixLayout_t{}) {
        (void)hipblasLtMatrixLayoutDestroy(handle);
      }
    }
    hipblasLtMatrixLayout_t handle{};
};

struct HipMatmulDesc {

    MOVABLE_HANDLE(HipMatmulDesc);

    HipMatmulDesc(hipblasComputeType_t compute_type, hipDataType scale_type,
        hipblasOperation_t trans_a, hipblasOperation_t trans_b,
        hipblasLtEpilogue_t epilogue) {

    VLOG("BlasLt::MatmulDesc compute_type: " << int(compute_type)
          << " scale_type: " << int(scale_type)
          << " epilogue: " << (int)epilogue << " trans_a: " << (int)trans_a
          << " trans_b: " << (int)trans_b);

      CHK_HIPBLASLT(hipblasLtMatmulDescCreate(
        &handle, compute_type, scale_type));
        // Wrap hipblas handle immediately, so it is cleaned up if an error occurs.

      SetAttr(handle, HIPBLASLT_MATMUL_DESC_TRANSA, trans_a);
      SetAttr(handle, HIPBLASLT_MATMUL_DESC_TRANSB, trans_b);
      SetAttr(handle, HIPBLASLT_MATMUL_DESC_EPILOGUE, epilogue);
    }
    ~HipMatmulDesc() {
      if(handle != hipblasLtMatmulDesc_t{}) {
        hipblasLtMatmulDescDestroy(handle);
      }
    }
    hipblasLtMatmulDesc_t handle{};
  };

struct MatmulPlan {

  HipMatmulDesc desc;
  HipMatrixLayout matA, matB, matC, matD;
};

struct BlasLtGemm {
  BlasLtGemm() {
    CHK_HIPBLASLT(hipblasLtCreate(&blas_lt_));
  }

  ~BlasLtGemm() {
    (void)hipblasLtDestroy(blas_lt_);
    if(workspace_ != nullptr) {
      (void)hipFree(workspace_);
    }
  }

  struct Config {
    hipblasOperation_t trans_a;
    hipblasOperation_t trans_b;
    hipblasComputeType_t compute_type;
    int64_t            m;
    int64_t            n;
    int64_t            k;
    hipblasLtEpilogue_t epilogue;
    uint64_t            max_algorithms;
    uint64_t            max_workspace_size;
    hipStream_t        stream;
  };

  auto handle() { return blas_lt_; }

  bool hasBias(hipblasLtEpilogue_t epi) {
    return (epi == HIPBLASLT_EPILOGUE_BIAS || epi == HIPBLASLT_EPILOGUE_RELU_BIAS ||
            epi == HIPBLASLT_EPILOGUE_GELU_BIAS ||
            epi == HIPBLASLT_EPILOGUE_GELU_AUX_BIAS);
  }

  template < class TypeA, class TypeB, class TypeC, class TypeD, class Scalar >
  MatmulPlan createPlan(const TypeA *dA, const TypeB *dB, const TypeC *dC, const TypeD *dBias,
      TypeD *dD, Scalar alpha, Scalar beta, const Config& cfg) {

    auto order = HipMatrixLayout::Order::kColumnMajor;

    MatmulPlan plan = {
      .desc = HipMatmulDesc(cfg.compute_type, HipBlasltType(&alpha),
            cfg.trans_a, cfg.trans_b, cfg.epilogue),
      .matA = HipMatrixLayout(HipBlasltType(dA), cfg.m, cfg.k, order),
      .matB = HipMatrixLayout(HipBlasltType(dB), cfg.k, cfg.n, order),
      .matC = HipMatrixLayout(HipBlasltType(dC), cfg.m, cfg.n, order),
      .matD = HipMatrixLayout(HipBlasltType(dD), cfg.m, cfg.n, order),
    };

    return std::move(plan);
  }

  // template <typename T>
  // void  SetAttr(hipblasLtMatmulDesc_t handle,
  //                    hipblasLtMatmulDescAttributes_t attr, T value) {
  //   CHK_HIPBLASLT(hipblasLtMatmulDescSetAttribute, handle, attr, value);
  // }

  template < class TypeD >
  std::vector< hipblasLtMatmulHeuristicResult_t > getAlgorithms(
          const MatmulPlan& plan, const Config& cfg, const TypeD *dBias) {
     // Set User Preference attributes
    hipblasLtMatmulPreference_t pref;
    CHK_HIPBLASLT(hipblasLtMatmulPreferenceCreate(&pref));
    CHK_HIPBLASLT(hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &cfg.max_workspace_size,
                                              sizeof(cfg.max_workspace_size)));

    if (hasBias(cfg.epilogue)) {
      auto dtype = HipBlasltType(dBias);
      CHK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
         plan.desc.handle, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
             &dtype, sizeof(hipDataType)));

      static int dummy = 0xF0F0F0F0; // actually it's enough to set some non-zero value
		  CHK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
			     plan.desc.handle, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, 
             &dummy, sizeof(void *)));
    }

    std::vector< hipblasLtMatmulHeuristicResult_t > 
        algo_results(cfg.max_algorithms), filtered;
    int returnedAlgoCount = 0;
    CHK_HIPBLASLT(hipblasLtMatmulAlgoGetHeuristic(blas_lt_, plan.desc.handle,
                     plan.matA.handle, plan.matB.handle, plan.matC.handle, plan.matD.handle,
                     pref, cfg.max_algorithms, algo_results.data(), 
                     &returnedAlgoCount));
    hipblasLtMatmulPreferenceDestroy(pref);

    if(returnedAlgoCount == 0) {
        ThrowError<>("No valid solutions found!");
    }
    filtered.reserve(returnedAlgoCount);

    for(uint32_t i = 0; i < returnedAlgoCount; i++) {
      if(algo_results[i].state == HIPBLAS_STATUS_SUCCESS) {
        filtered.push_back(algo_results[i]);
      }
    }
    return filtered;
  }

  static std::tuple<int, int> getAlgoIndex(hipblasLtMatmulHeuristicResult_t algo) {

    struct __attribute__((packed, aligned(8))) rocblaslt_matmul_algo
    {
      uint8_t data[8]             = {0};
      bool    fallback            = false;
      size_t  max_workspace_bytes = 0;
    };
    auto roc_algo = (const rocblaslt_matmul_algo *)&algo.algo; 
    auto pindex = (const int32_t *)roc_algo->data;
    return std::tuple{ *pindex, roc_algo->fallback };
  }

  template < class TypeA, class TypeB, class TypeC, class TypeD, class Scalar >
  void run(const TypeA *dA, const TypeB *dB, const TypeC *dC, const TypeD *dBias,
      TypeD *dD, Scalar alpha, Scalar beta, const Config& cfg, 
      const MatmulPlan& plan, hipblasLtMatmulHeuristicResult_t algo)
  {
    if (hasBias(cfg.epilogue)) {
      auto dtype = HipBlasltType(dBias);
      CHK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
        plan.desc.handle, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
            &dtype, sizeof(hipDataType)));

		  CHK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
			    plan.desc.handle, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, 
              &dBias, sizeof(void *)));
    }

    if(algo.workspaceSize > workspace_sz_) {
      if(workspace_ != nullptr) {
        CHK(hipFree(workspace_));
      }
      workspace_sz_ = algo.workspaceSize;
      CHK(hipMalloc(&workspace_, workspace_sz_));
    }
    CHK_HIPBLASLT(hipblasLtMatmul(blas_lt_, plan.desc.handle, &alpha,
                 dA, plan.matA.handle,
                 dB, plan.matB.handle, &beta,
                 dC, plan.matC.handle,
                 dD, plan.matD.handle,
                 &algo.algo,
                 workspace_, workspace_sz_, cfg.stream));
  }

 private:
  void *workspace_ = nullptr;
  size_t workspace_sz_ = 0;
  hipblasLtHandle_t blas_lt_;
};

template <typename T, typename U = T, typename V = U>
void matMatMultMixPrec(T alpha, T beta, int M, int N, int K,
    U*  A, int As1, int As2,
    U*  B, int Bs1, int Bs2,
    V*  C, int Cs1, int Cs2,
    V*  D, int Ds1, int Ds2,
    V *bias = nullptr)
{
  for(int i1 = 0; i1 < M; i1++)
  {
    for(int i2 = 0; i2 < N; i2++)
    {
      T t = T(0.0);
      for(int i3 = 0; i3 < K; i3++)
      {
        t += T(A[i1 * As1 + i3 * As2]) * T(B[i3 * Bs1 + i2 * Bs2]);
      }
      V v(beta * T(C[i1 * Cs1 + i2 * Cs2]) + alpha * t);
      if(bias != nullptr) {
        v += bias[i1];
      }
      D[i1 * Ds1 + i2 * Ds2] = v;
    }
  }
}

#endif // HIPBLASLT_GEMM_HPP
