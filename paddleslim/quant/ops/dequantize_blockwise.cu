#include "paddle/extension.h"
#include<stdlib.h>
#include<string.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<unistd.h>
#include<fcntl.h>
#include<sys/mman.h>
#include<stdio.h>
#include<algorithm>
#include<cub/device/device_scan.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>
#include <math_constants.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <mma.h>
#include "common.h"

__device__ float dDequantizeFP4Tree(unsigned char val, float absmax)
{
  float sign = (val & 0b1000) == 8 ? -1.0f : 1.0f;
  if((val & 0b0100) == 4) // 0
    if((val & 0b0010) == 2) //01
      if((val & 0b0001) == 1) // 111
        return 0.25000000f*absmax*sign; // 1111
      else
        return 0.16666667f*absmax*sign; // 1110
    else
      if((val & 0b0001) == 1) // 110
        return 0.50000000f*absmax*sign; // 1101
      else
        return 0.33333333f*absmax*sign; // 1100
  else
    if((val & 0b0010) == 2) //10
      if((val & 0b0001) == 1) // 101
        return 1.00000000f*absmax*sign; // 1011
      else
        return 0.66666667f*absmax*sign; // 1010
    else
      if((val & 0b0001) == 1) // 100
        return 5.208333333e-03f*absmax*sign; // 1001
      else
        return 0.00000000f*absmax*sign; // 1000
}

__device__ float dDequantizeNF4(unsigned char val)
{

  // the values for this tree was generated by test_normal_map_tree
  // in the file tests/test_functional.py
  if((val & 0b1000) == 8)
    if((val & 0b0100) == 4) // 1
      if((val & 0b0010) == 2) // 11
        if((val & 0b0001) == 1) // 111
          return 1.0f;
        else
          return 0.7229568362236023f;
      else
        if((val & 0b0001) == 1) // 110
          return 0.5626170039176941f;
        else
          return 0.44070982933044434f;
    else
      if((val & 0b0010) == 2) //10
        if((val & 0b0001) == 1) // 101
          return 0.33791524171829224f;
        else
          return 0.24611230194568634f;
      else
        if((val & 0b0001) == 1) // 100
          return 0.16093020141124725f;
        else
          return 0.07958029955625534f;

  else
    if((val & 0b0100) == 4) // 0
      if((val & 0b0010) == 2) //01
        if((val & 0b0001) == 1) // 011
          return 0.0f;
        else
          return -0.09105003625154495f;
      else
        if((val & 0b0001) == 1) // 010
          return -0.18477343022823334f;
        else
          return -0.28444138169288635f;
    else
      if((val & 0b0010) == 2) //00
        if((val & 0b0001) == 1) // 001
          return -0.39491748809814453f;
        else
          return -0.5250730514526367f;
      else
        if((val & 0b0001) == 1) // 000
          return -0.6961928009986877f;
        else
          return -1.0f;

}


template<typename T, int TILE_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE>
__global__ void kDequantizeBlockwise(const float *code, const unsigned char * A, const float * absmax, T *out, int blocksize, int n)
{

  const int n_load = (gridDim.x * TILE_SIZE);
  int valid_items_load = 0;
  int valid_items_store = 0;
  const int base_idx = (blockIdx.x * TILE_SIZE);

  T vals[NUM_PER_TH*((DATA_TYPE > 0) ? 2 : 1)];
  unsigned char qvals[NUM_PER_TH];
  float local_abs_max = -FLT_MAX;

  typedef cub::BlockLoad<unsigned char, THREADS, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;
  typedef cub::BlockStore<T, THREADS, NUM_PER_TH*((DATA_TYPE > 0) ? 2 : 1), cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT;

  __shared__ typename LoadChar::TempStorage loadchar;
  __shared__ typename StoreT::TempStorage storet;

  for (unsigned int i = base_idx; i < n_load; i += gridDim.x*TILE_SIZE)
  {
    if(DATA_TYPE > 0)
    {
      valid_items_load = (n+1)/2 - i > TILE_SIZE ? TILE_SIZE : (n+1)/2 - i;
      valid_items_store = n - i*2 > TILE_SIZE*2 ? TILE_SIZE*2 : n - i*2;
    }
    else
    {
      valid_items_load = n - i > TILE_SIZE ? TILE_SIZE : n - i;
      valid_items_store = n - i > TILE_SIZE ? TILE_SIZE : n - i;
    }
    local_abs_max = __ldg(&absmax[(i+threadIdx.x*NUM_PER_TH)/(blocksize)]);

    __syncthreads();
    LoadChar(loadchar).Load(&(A[i]), qvals, valid_items_load, 128);

    switch(DATA_TYPE)
    {
        case General8bit:
          // load code through read-only cache via __ldg
          #pragma unroll NUM_PER_TH
          for(int j = 0; j < NUM_PER_TH; j++)
            vals[j] = __ldg(&code[qvals[j]])*local_abs_max;
          break;
        case FP4:
          #pragma unroll NUM_PER_TH
          for(int j = 0; j < NUM_PER_TH; j++)
          {
            vals[j*2] = dDequantizeFP4Tree(qvals[j] >> 4, local_abs_max);
            vals[j*2 + 1] = dDequantizeFP4Tree(qvals[j] & 0x0F, local_abs_max);
          }
          break;
        case NF4:
          #pragma unroll NUM_PER_TH
          for(int j = 0; j < NUM_PER_TH; j++)
          {
            vals[j*2] = dDequantizeNF4(qvals[j] >> 4)* local_abs_max;
            vals[j*2 + 1] = dDequantizeNF4(qvals[j] & 0x0F)* local_abs_max;
          }
          break;
    }

    __syncthreads();
    StoreT(storet).Store(&(out[(DATA_TYPE > 0) ? i*2 : i]), vals, valid_items_store);
  }
}

//template __global__ void kDequantizeBlockwise<half, 512, 64, 8, FP4>(const float *code, const unsigned char * A, const float * absmax, half *out, int blocksize, int n);
//template __global__ void kDequantizeBlockwise<half, 512, 64, 8, General8bit>(const float *code, const unsigned char * A, const float * absmax, half *out, int blocksize, int n);
//template __global__ void kDequantizeBlockwise<half, 512, 64, 8, NF4>(const float *code, const unsigned char * A, const float * absmax, half *out, int blocksize, int n);
template __global__ void kDequantizeBlockwise<float, 512, 64, 8, FP4>(const float *code, const unsigned char * A, const float * absmax, float *out, int blocksize, int n);
template __global__ void kDequantizeBlockwise<float, 512, 64, 8, General8bit>(const float *code, const unsigned char * A, const float * absmax, float *out, int blocksize, int n);
template __global__ void kDequantizeBlockwise<float, 512, 64, 8, NF4>(const float *code, const unsigned char * A, const float * absmax, float *out, int blocksize, int n);
//template __global__ void kDequantizeBlockwise<__nv_bfloat16, 512, 64, 8, FP4>(const float *code, const unsigned char * A, const float * absmax, __nv_bfloat16 *out, int blocksize, int n);
//template __global__ void kDequantizeBlockwise<__nv_bfloat16, 512, 64, 8, General8bit>(const float *code, const unsigned char * A, const float * absmax, __nv_bfloat16 *out, int blocksize, int n);
//template __global__ void kDequantizeBlockwise<__nv_bfloat16, 512, 64, 8, NF4>(const float *code, const unsigned char * A, const float * absmax, __nv_bfloat16 *out, int blocksize, int n);



template<typename T, int DATA_TYPE> void dequantize_blockwise(const float *code, const unsigned char *A, const float *absmax, T *out, int blocksize, int n)
{
  int num_blocks = n/blocksize;
  num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;
  int tile_size = (DATA_TYPE > 0) ? 1024 : 512;

  if(DATA_TYPE > 0)
    kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE><<<(n+tile_size-1)/tile_size, 64>>>(code, A, absmax, out, blocksize/2, n);
  else
    kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE><<<(n+tile_size-1)/tile_size, 64>>>(code, A, absmax, out, blocksize, n);

  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template void dequantize_blockwise<float, General8bit>(const float *code, const unsigned char *A, const float *absmax, float *out, int blocksize, int n);
template void dequantize_blockwise<float, FP4>(const float *code, const unsigned char *A, const float *absmax, float *out, int blocksize, int n);
template void dequantize_blockwise<float, NF4>(const float *code, const unsigned char *A, const float *absmax, float *out, int blocksize, int n);
//template void dequantize_blockwise<half, General8bit>(const float *code, const unsigned char *A, const float *absmax, half *out, int blocksize, int n);
//template void dequantize_blockwise<half, FP4>(const float *code, const unsigned char *A, const float *absmax, half *out, int blocksize, int n);
//template void dequantize_blockwise<half, NF4>(const float *code, const unsigned char *A, const float *absmax, half *out, int blocksize, int n);
//template void dequantize_blockwise<__nv_bfloat16, General8bit>(const float *code, const unsigned char *A, const float *absmax, __nv_bfloat16 *out, int blocksize, int n);
//template void dequantize_blockwise<__nv_bfloat16, FP4>(const float *code, const unsigned char *A, const float *absmax, __nv_bfloat16 *out, int blocksize, int n);
//template void dequantize_blockwise<__nv_bfloat16, NF4>(const float *code, const unsigned char *A, const float *absmax, __nv_bfloat16 *out, int blocksize, int n);

std::vector<paddle::Tensor> DequantizeBlockwise(const paddle::Tensor& input, const paddle::Tensor& code, const paddle::Tensor& absmax, int blocksize, int n, std::string quant_type, std::string out_type) {
    auto input_num = input.numel();
    std::vector<int64_t> out_shape = input.shape();
    if (quant_type != "8bit") { // 4bit
        out_shape = {input_num * 2, 1};
    }

    auto out = paddle::empty(out_shape, paddle::DataType::FLOAT32, input.place());

    if (quant_type == "8bit")
        dequantize_blockwise<float, General8bit>(code.data<float>(), input.data<unsigned char>(), absmax.data<float>(), out.data<float>(), blocksize, n);
    else if (quant_type == "nf4")
        dequantize_blockwise<float, NF4>(code.data<float>(), input.data<unsigned char>(), absmax.data<float>(), out.data<float>(), blocksize, n);
    else if (quant_type == "fp4")
        dequantize_blockwise<float, FP4>(code.data<float>(), input.data<unsigned char>(), absmax.data<float>(), out.data<float>(), blocksize, n);
    else
        PD_THROW("NOT supported quant type. Only 8bit, nf4, fp4 are supported. ");
    return {out};
};

std::vector<std::vector<int64_t>> GetDequantizeBlockwiseInferShape(const std::vector<int64_t>& input_shape, const std::vector<int64_t>& code_shape, const std::vector<int64_t>& abs_max_shape, int blocksize, int n, std::string quant_type, std::string out_type){
    if (quant_type != "8bit")
        return {{input_shape[0], input_shape[1] * 2}};
    else
        return {input_shape};
}
std::vector<paddle::DataType> GetDequantizeBlockwiseInferDtype(const paddle::DataType& input_dtype, const paddle::DataType& code_dtype, const paddle::DataType& abs_max_dtype){
    return {paddle::DataType::FLOAT32};
}


PD_BUILD_OP(dequant_blockwise)
    .Inputs({"input", "code", "abs_max"})
    .Outputs({"output"})
    .Attrs({"blocksize: int", "n: int", "quant_type: std::string"})
    .SetKernelFn(PD_KERNEL(DequantizeBlockwise))
    .SetInferShapeFn(PD_INFER_SHAPE(GetDequantizeBlockwiseInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetDequantizeBlockwiseInferDtype));


