#!/bin/bash

set -e
debug=${debug:-0}

GDB=
if [[ ${debug} -eq 1 ]]; then
  GDB="rocgdb --args "
fi

export HIP_VISIBLE_DEVICES=0,1

mkdir -p build

pushd build
cmake .. -DCOMPILE_FOR_ROCM=1 -DCMAKE_BUILD_TYPE=Release
rm -f gpucore.*
make -j 2>&1
$GDB ./bin/graph_capture 2>&1 | tee ../zzzrun.log
popd

