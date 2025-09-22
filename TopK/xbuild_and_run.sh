#!/bin/bash

set -e

pushd build
make -j 2>&1
./bin/topk 2>&1 | tee ../zzzrun.log
popd
