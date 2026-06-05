#!/bin/bash

set -e
debug=${debug:-0}


# hipblaslt-bench --api_method c -m 1536 -n 12288 -k 4608 --lda 4608 --ldb 4608 --ldc 1536 --ldd 1536 --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0 --alpha 1 --beta 1 --transA T --transB T --batch_count 1 --scaleA 3 --scaleB 3 --a_type f8_r --b_type f8_r --c_type f32_r --d_type f32_r --compute_type f32_r --algo_method index --solution_index -2105014268 --rotating 0 --cold_iters 0 --iters 0

# compare vs JAX:
# hipblaslt-bench --api_method c -m 12288 -n 1536 -k 4608 --lda 4608 --ldb 4608 --ldc 12288 --ldd 12288 --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0 --alpha 1 --beta 0 --transA T --transB T --batch_count 1 --scaleA 3 --scaleB 3 --a_type f8_r --b_type f8_r --c_type f32_r --d_type f32_r --compute_type f32_r --algo_method index --solution_index -2105014268 --rotating 0 --cold_iters 0 --iters 0 

#      M     K     N      GF    bf16(ms) mxfp8(ms) speedup mxfp4(ms) speedup
#   1536  4608 12288   173.9        0.19      0.29    0.66      0.00    0.00
# I0605 13:25:54.353725   67639 rocm_profiler_sdk.cc:532] Calling toolFinalize

# BF16 perf:
# [/data/Playground/HipBlasLt/hipblaslt_test.cc:88] Total algos found 128
# gemm.run; time elapsed: 128.974 us
# mxfp8 perf:
#[/data/Playground/HipBlasLt/hipblaslt_test.cc:88] Total algos found 24
# gemm.run; time elapsed: 196.867 us


GDB=
if [[ ${debug} -eq 1 ]]; then
  GDB="rocgdb --args "
fi

export HIP_VISIBLE_DEVICES=0
# export GPU_DUMP_CODE_OBJECT=1
# export AMD_LOG_LEVEL=4
# export TENSILE_DB=255
# export HIPBLASLT_LOG_MASK=32

mkdir -p build

pushd build
cmake .. -DCOMPILE_FOR_ROCM=1 -DCMAKE_BUILD_TYPE=Release
rm -f gpucore.*
make -j 2>&1
$GDB ./bin/hipblaslt_test 2>&1 | tee ../zzzrun.log
popd

