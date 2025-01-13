#!/bin/sh
set -e
set -o pipefail
export USE_BAZEL_VERSION=6.5.0

NUM=$#
BAZEL=bazel
DUMP_DIR=ww_dump
EXEC=bazel-bin/llvm_test
ROCM_PATH=$(realpath /opt/rocm)
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

rm -rf $DUMP_DIR && mkdir -p $DUMP_DIR
rm -rf gpucore.*

        # --config=release \
        # --subcommands \
        # --crosstool_top=//crosstool:rocm-toolchain-suite \

$BAZEL --output_base=/data/bazel_llvm_test \
         build --config rocm --config release \
         //:llvm_test \
         2>&1 | tee build.out

roc-obj -t gfx942 -d -o $DUMP_DIR $EXEC

$GDB $EXEC 2>&1 | tee test.out
