#!/bin/bash

set -e
debug=${debug:-0}

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

