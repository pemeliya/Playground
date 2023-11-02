// Original program by Chen, Wen adapted

/******************** compile with : *************************************/
// hipcc -lhipblaslt -std=c++17 --offload-arch=gfx90a hipblaslt_test.cc -Wno-backslash-newline-escape
// HIPBLASLT_LOG_MASK=5 HIPBLASLT_LOG_LEVEL=5 ./a.out

#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <optional>

//#if TF_ROCM_VERSION < 60000
#define hipblasltDatatype_t hipblasDatatype_t
#define HIPBLASLT_R_16F HIPBLAS_R_16F
#define HIPBLASLT_R_16B HIPBLAS_R_16B
#define HIPBLASLT_R_32F HIPBLAS_R_32F
#define HIPBLASLT_R_64F HIPBLAS_R_64F
#define HIPBLASLT_R_8I HIPBLAS_R_8I
#define HIPBLASLT_R_32I HIPBLAS_R_32I
#define HIPBLASLT_C_32F HIPBLAS_C_32F
#define HIPBLASLT_C_64F HIPBLAS_C_64F
//#endif

#define LOG(x) std::cerr << x << std::endl

#ifndef CHK_HIP
#define CHK_HIP(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr, "Hip error: '%s'(%d) at %s:%d\n",  hipGetErrorString(error),         \ 
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        throw 0;  \
    }
#endif

#ifndef CHK_HIPBLASLT
#define CHK_HIPBLASLT(error)                                                      \
    if(error != HIPBLAS_STATUS_SUCCESS)                                                   \
    {                                                                                     \
        fprintf(stderr, "hipBLASLt error(Err=%d) at %s:%d\n", error, __FILE__, __LINE__); \
        fprintf(stderr, "\n");                                                            \
        throw 0;  \
    }
#endif

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

template< class NT >
struct MappedVector {
    
   explicit MappedVector(size_t N_) : N(N_) {
       CHK_HIP(hipHostMalloc((void**)&devPtr, N*sizeof(NT)))
   }
   size_t size() const { return N; }
   NT& operator[](size_t i) {
      return devPtr[i];
   }
   ~MappedVector() {
      (void)hipHostFree(devPtr);
   }
   size_t N;
   NT *devPtr;
};

template < class T >
constexpr hipblasltDatatype_t HipBlasltType(const T *) {
  if constexpr (std::is_same_v<T, __half>) 
    return HIPBLASLT_R_16F;
  if constexpr (std::is_same_v<T, hip_bfloat16>) 
    return HIPBLASLT_R_16B;
  if constexpr (std::is_same_v<T, float>) 
    return HIPBLASLT_R_32F;
  if constexpr (std::is_same_v<T, double>) 
    return HIPBLASLT_R_64F;
  if constexpr (std::is_same_v<T, int32_t>) 
    return HIPBLASLT_R_32I;
  if constexpr (std::is_same_v<T, int8_t>) 
    return HIPBLASLT_R_8I;
  if constexpr (std::is_same_v<T, std::complex< float >>) 
    return HIPBLASLT_C_32F;
  if constexpr (std::is_same_v<T, std::complex< double >>) 
    return HIPBLASLT_C_64F;
  
  return (hipblasltDatatype_t)-1;
}

template <typename T>
  using Owned =
      std::unique_ptr<std::remove_pointer_t<T>, hipblasStatus_t (*)(T)>;

struct BlasLt {
  BlasLt() : blas_lt_(nullptr, hipblasLtDestroy) {
    hipblasLtHandle_t H;
    CHK_HIPBLASLT(hipblasLtCreate(&H));
    blas_lt_.reset(H);
  }

  hipblasLtHandle_t get() {
    return blas_lt_.get();
  }

 private:
  Owned<hipblasLtHandle_t> blas_lt_;
};

struct HipMatrixLayout {

    enum class Order { kRowMajor, kColumnMajor };

    // If `leading_dim_stride` is not specified, it defaults to:
    //  - `num_cols` if `order == kRowMajor`,
    //  - `num_rows` if `order == kColumnMajor`.
    // If `batch_stride` is not specified, it defaults to `num_rows * num_cols`
    // if `batch_size > 1`, otherwise `0`.
    static HipMatrixLayout Create(
        hipblasltDatatype_t type, size_t num_rows, size_t num_cols, Order order,
        size_t batch_size = 1,
        std::optional<int64_t> leading_dim_stride = std::nullopt,
        std::optional<int64_t> batch_stride = std::nullopt) {

      if (!leading_dim_stride) {
        leading_dim_stride = (order == Order::kRowMajor) ? num_cols : num_rows;
      }
      hipblasLtMatrixLayout_t hip_layout;
      CHK_HIPBLASLT(hipblasLtMatrixLayoutCreate(
        &hip_layout, type, num_rows, num_cols,
        *leading_dim_stride));
  
      // Wrap hipblas handle immediately, so it is cleaned up if an error occurs.
      HipMatrixLayout layout(hip_layout);
      SetAttr(hip_layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                              static_cast<int32_t>(batch_size));

      if (!batch_stride) {
        batch_stride = (batch_size > 1) ? num_rows * num_cols : 0;
      }

      LOG("MatrixLayout create: type: " << (int)type << ","
          << num_rows << "x" << num_cols << " leadingdim: "
          << *leading_dim_stride << " batchstride: " <<  *batch_stride);

      SetAttr(
           hip_layout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, *batch_stride);
      return std::move(layout);
    }

    hipblasLtMatrixLayout_t get() const { return handle_.get(); }

   private:
    explicit HipMatrixLayout(hipblasLtMatrixLayout_t handle)
        : handle_(handle, hipblasLtMatrixLayoutDestroy) {}

    Owned<hipblasLtMatrixLayout_t> handle_;
};

struct MatmulDesc {

    static MatmulDesc Create(
        hipblasLtComputeType_t compute_type, hipblasltDatatype_t scale_type,
        hipblasOperation_t trans_a,
        hipblasOperation_t trans_b,
        hipblasLtEpilogue_t epilogue) {

      hipblasLtMatmulDesc_t hip_desc;
      LOG("MatmulDesc::Create compute_type: " << compute_type
          << " scale_type " << scale_type << " epilogue " << int(epilogue)
          << " transA " << trans_a << " transB " << trans_b);

      CHK_HIPBLASLT(hipblasLtMatmulDescCreate(
        &hip_desc, compute_type, scale_type));
        // Wrap hipblas handle immediately, so it is cleaned up if an error occurs.
      MatmulDesc desc(hip_desc);

      SetAttr(hip_desc, HIPBLASLT_MATMUL_DESC_TRANSA, trans_a);
      SetAttr(hip_desc, HIPBLASLT_MATMUL_DESC_TRANSB, trans_b);
      SetAttr(hip_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, epilogue);

      return std::move(desc);
    }

    hipblasLtMatmulDesc_t get() const { return handle_.get(); }

   private:
    explicit MatmulDesc(hipblasLtMatmulDesc_t handle)
        : handle_(handle, hipblasLtMatmulDescDestroy) {}

    Owned<hipblasLtMatmulDesc_t> handle_;
  };

struct GemmConfig {
  hipblasLtHandle_t  handle;
  hipblasOperation_t trans_a;
  hipblasOperation_t trans_b;
  int64_t            m;
  int64_t            n;
  int64_t            k;
  float              alpha;
  float              beta;
  hipblasltDatatype_t type_a;
  hipblasltDatatype_t type_b;
  hipblasltDatatype_t type_c;
  hipblasltDatatype_t type_d;
  void*              d_a;
  void*              d_b;
  void*              d_c;
  void*              d_d;
  void*              d_bias;
  hipblasLtEpilogue_t epilogue;
  uint64_t            max_workspace_size;
  hipStream_t        stream;
};

void simpleGemm(const GemmConfig& cfg)
{
    //LOG("Using datatype " << (int)HIPBLAS_R_16F << "," << HIPBLAS_R_16B << " HIPBLASLT_COMPUTE_F32 = " << HIPBLASLT_COMPUTE_F32) ;
    //LOG("Epilogue: " << HIPBLASLT_EPILOGUE_BIAS << ", HIPBLAS_OP_T =" << HIPBLAS_OP_T << ",HIPBLAS_OP_N = " << HIPBLAS_OP_N);

    auto desc = MatmulDesc::Create(HIPBLASLT_COMPUTE_F32, HIPBLASLT_R_32F,
            cfg.trans_a, cfg.trans_b, cfg.epilogue);

    auto order = HipMatrixLayout::Order::kColumnMajor;
    auto matA = HipMatrixLayout::Create(cfg.type_a, cfg.m, cfg.k, order);
    auto matB = HipMatrixLayout::Create(cfg.type_b, cfg.k, cfg.n, order);
    auto matC = HipMatrixLayout::Create(cfg.type_c, cfg.m, cfg.n, order);
    auto matD = HipMatrixLayout::Create(cfg.type_d, cfg.m, cfg.n, order);

    if (cfg.epilogue == HIPBLASLT_EPILOGUE_BIAS) {
      static int dummy;
			CHK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
			 		desc.get(), HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &dummy, sizeof(void*)));
    }

    // Set User Preference attributes
    hipblasLtMatmulPreference_t pref;
    CHK_HIPBLASLT(hipblasLtMatmulPreferenceCreate(&pref));
    CHK_HIPBLASLT(hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &cfg.max_workspace_size,
                                              sizeof(cfg.max_workspace_size)));
    size_t max_algos = 128;
    std::vector< hipblasLtMatmulHeuristicResult_t > algo_results(max_algos);
    int returnedAlgoCount = 0;
    CHK_HIPBLASLT(hipblasLtMatmulAlgoGetHeuristic(cfg.handle, desc.get(),
                     matA.get(), matB.get(), matC.get(), matD.get(),
                     pref, max_algos, algo_results.data(), &returnedAlgoCount));
    hipblasLtMatmulPreferenceDestroy(pref);

    if(returnedAlgoCount == 0) {
        throw std::runtime_error("No valid solutions found!");
    }
    algo_results.resize(returnedAlgoCount);

    hipblasLtMatmulHeuristicResult_t algoOK{
      .state = HIPBLAS_STATUS_UNKNOWN,
    };

    uint64_t workspace_size = 0, i = 0;
    void *d_workspace = nullptr;
    for (const hipblasLtMatmulHeuristicResult_t& result : algo_results) {
        if (result.state == HIPBLAS_STATUS_SUCCESS) {  // Skip failed algos.
          algoOK = result;
          break;
        }
    }
     //workspace_size = max(workspace_size, heuristicResult[i].workspaceSize);
    // In this sample, the workspace is already allocated with max_workspace_size
    // If not, allocate d_workspace here
    // CHK_HIPhipMalloc(&d_workspace, workspace_size));

    if (cfg.epilogue == HIPBLASLT_EPILOGUE_BIAS) {
			CHK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
					desc.get(), HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &cfg.d_bias, sizeof(cfg.d_bias)));
    }

    CHK_HIPBLASLT(hipblasLtMatmul(cfg.handle, desc.get(), &cfg.alpha,
                 cfg.d_a, matA.get(),
                 cfg.d_b, matB.get(), &cfg.beta,
                 cfg.d_c, matC.get(),
                 cfg.d_d, matD.get(),
                 &algoOK.algo,
                 d_workspace, workspace_size, cfg.stream));
}

template < class T >
void initVec(T *ptr, std::initializer_list< double > l) 
{
  for(const auto& elem : l) {
    *ptr++ = static_cast< T >(elem);
  }
}

int main(int argc, char *argv[]) try
{
	int m = 2, n = 2, k = 2;
  float alpha = 1.0, beta = 0.0;

  BlasLt blasLtObj;

  using TypeA = hip_bfloat16;
  using TypeB = hip_bfloat16;
  using TypeC = float;
  using TypeD = float;

  MappedVector< TypeA > a(m * k);
  MappedVector< TypeB > b(n * k);
  MappedVector< TypeC > c(m * n);
  MappedVector< TypeD > d(m * n);
  MappedVector< TypeD > bias(m);

  initVec(a.devPtr, {10.0, 12.0, 11.0, 13.0});
  initVec(b.devPtr, {1.0, 3.0, 2.0, 4.0});
  initVec(bias.devPtr, {10.0, 11.0});

  size_t max_workspace_size = 1ll << 32;
  simpleGemm(GemmConfig{
    .handle = blasLtObj.get(),
    .trans_a = HIPBLAS_OP_N,
    .trans_b = HIPBLAS_OP_N,
    .m = m,
    .n = n,
    .k = k,
    .alpha = alpha,
    .beta = beta,
    .type_a = HipBlasltType(a.devPtr),
    .type_b = HipBlasltType(b.devPtr),
    .type_c = HipBlasltType(c.devPtr),
    .type_d = HipBlasltType(d.devPtr),
    .d_a = a.devPtr,
    .d_b = b.devPtr,
    .d_c = d.devPtr,
    .d_d = d.devPtr,
    .d_bias = bias.devPtr,
    .epilogue = HIPBLASLT_EPILOGUE_DEFAULT,
    .max_workspace_size = max_workspace_size,
    .stream = 0,
  });

  CHK_HIP(hipDeviceSynchronize());
  for (int i=0;i<m;i++) {
     for (int j=0;j<n;j++) {
				std::cout << static_cast<float>(d[i*n+j]) << " ";
			}
      std::cout << std::endl;
  }
  return 0;
}
catch(std::exception& ex) {
  LOG("Exception: " << ex.what());
}
catch(...) {
  LOG("Unknown exception");
}