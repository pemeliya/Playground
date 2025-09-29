/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TOPK_KERNEL_H_
#define TOPK_KERNEL_H_

#include <stddef.h>
#include <stdint.h>

enum class TopKType : int32_t {
  I16,
  U16,
  I32,
  U32,
  F16,
  BF16,
  F32,
  F64,
};

struct TopkArgs {
  const void* data;
  void* top_elements;
  uint32_t* top_indices;
  TopKType type;  
  uint32_t num_elems;
  uint32_t k;
  uint32_t batch_size;
};

void RunPerWarpTopK(TopkArgs& args);
void RunBitonicTopK(TopkArgs& args);

#endif  // TOPK_KERNEL_H_
