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

#ifndef XLA_SERVICE_GPU_RUNTIME_TOPK_KERNEL_H_
#define XLA_SERVICE_GPU_RUNTIME_TOPK_KERNEL_H_

#include <stddef.h>
#include <stdint.h>

constexpr size_t kTopKMaxThreadsPerBlock = 1024;

template <typename T, size_t K>
void* GetTopKKernelForK(size_t n_threads);

template <typename T>
void* GetKernel(size_t n_threads, size_t k) {
  // if (k <= 1) return GetTopKKernelForK<T, 1>(n_threads);
  // if (k <= 2) return GetTopKKernelForK<T, 2>(n_threads);
  // if (k <= 4) return GetTopKKernelForK<T, 4>(n_threads);
  // if (k <= 8) return GetTopKKernelForK<T, 8>(n_threads);
  if (k <= 16) return GetTopKKernelForK<T, 16>(n_threads);
  return nullptr;
}

#endif  // XLA_SERVICE_GPU_RUNTIME_TOPK_KERNEL_H_
