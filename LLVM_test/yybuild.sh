#!/bin/sh
export USE_BAZEL_VERSION=6.5.0

NUM=$#
BAZEL=bazel
ROCM_PATH=$(realpath /opt/rocm)
#export CC=$(realpath /opt/rocm/llvm/bin/amdclang)
#export CXX=$(realpath /opt/rocm/llvm/bin/amdclang++)

export CC=$ROCM_PATH/lib/llvm/bin/clang
export CXX=$ROCM_PATH/lib/llvm/bin/clang++

if [ "$1" = "clean" ]; then
  echo "------------ cleaning ----------------"
  $BAZEL clean
  shift 1
fi

if [ "$1" = "clean0" ]; then
  echo "------------ full cleaning ----------------"
  $BAZEL clean --expunge
  shift 1
fi

GDB=
if [ "$1" = "g" ]; then
  GDB="rocgdb --args "
  shift 1
fi 

        # --config=release \
        # --subcommands \
        # --crosstool_top=//crosstool:rocm-toolchain-suite \

$BAZEL --output_base=/data/bazel_llvm_test \
         build --config rocm --config release \
         //:llvm_test 2>&1 | tee build.out

$GDB ./bazel-bin/llvm_test 2>&1 | tee test.out
