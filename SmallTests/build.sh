#!/bin/sh
#hipcc -I.. -DCOMPILE_FOR_ROCM=1 -std=c++17 -pthread --offload-arch=gfx942 $@
set +x

mkdir build
pushd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/amdclang++ ..
make -j 
popd