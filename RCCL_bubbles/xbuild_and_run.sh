#!/bin/bash

set -e
debug=${debug:-0}

GDB=
if [[ ${debug} -eq 1 ]]; then
  GDB="rocgdb --args "
fi

#export HIP_VISIBLE_DEVICES=3
# export GPU_DUMP_CODE_OBJECT=1
# export AMD_LOG_LEVEL=4
# export TENSILE_DB=255
# export HIPBLASLT_LOG_MASK=32

mkdir -p build

pushd build
cmake .. -DCOMPILE_FOR_ROCM=1 -DCMAKE_BUILD_TYPE=Release

make -j 2>&1

export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=PROFILE 
#export NCCL_RUNTIME_CONNECT=0
export LD_PRELOAD= #/tf/clr/build/hipamd/lib/libamdhip64.so 

$GDB ./bin/RCCL_bubbles 2>&1 | tee ../zzzrun.log
popd

