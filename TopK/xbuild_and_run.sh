#!/bin/bash

set -e
debug=${debug:-0}

GDB=
if [[ ${debug} -eq 1 ]]; then
  GDB="rocgdb --args "
fi

export HIP_VISIBLE_DEVICES=13
# export AMD_LOG_LEVEL=4

pushd build
rm -f gpucore.*
make -j 2>&1
$GDB ./bin/topk 2>&1 | tee ../zzzrun.log
popd
