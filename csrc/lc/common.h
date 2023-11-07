// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef COMMON_H
#define COMMON_H

typedef enum LC_DataType_t
{
	General8bit = 0,
	FP4 = 1,
    NF4 = 2,
} LC_DataType_t;

template <typename T, int DATA_TYPE> void quantize_blockwise(const float * code, const T *A, float *absmax, unsigned char *out, int blocksize, int n);
template<typename T, int DATA_TYPE> void dequantize_blockwise(const float *code, const unsigned char *A, float *absmax, T *out, int block_size, int n);


#define CUDA_CHECK_RETURN(value) {                      \
  cudaError_t _m_cudaStat = value;                    \
  if (_m_cudaStat != cudaSuccess) {                   \
    fprintf(stderr, "Error %s at line %d in file %s\n",         \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1);                              \
  } }

#endif
