#!/bin/bash
#hipcc -I.. -DCOMPILE_FOR_ROCM=1 -std=c++17 -pthread --offload-arch=gfx942 $@
set +x

mkdir build
pushd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/amdclang++ ..
#cmake -DCMAKE_CXX_COMPILER=/usr/bin/clang++-18 -DCOMPILE_FOR_ROCM=0 ..
make -j 
popd