#!/bin/bash
set -x
MYDIR=$(dirname "$0")
pushd $MYDIR

/opt/rocm/llvm/bin/amdclang++ -DCOMPILE_FOR_ROCM=1 -D_USE_MATH_DEFINES \
        -D__HIP_ROCclr__=1 -I.. -O3 -DNDEBUG -std=gnu++20 \
        --offload-arch=gfx942 -mllvm=-amdgpu-early-inline-all=true \
        -mllvm=-amdgpu-function-calls=false -D__HIP_PLATFORM_AMD__ \
        $@


# hipcc -DCOMPILE_FOR_ROCM=1 -D_USE_MATH_DEFINES \
#         -I.. -O3 -DNDEBUG -std=gnu++20 \
#         --offload-arch=gfx942 -D__HIP_PLATFORM_AMD__ \
#         -lhipblaslt -lhipblas \
#         $@

popd