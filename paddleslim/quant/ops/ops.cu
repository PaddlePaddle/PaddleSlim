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
#include "kernel.cuh"
#include "ops.cuh"


template <typename T, int DATA_TYPE> void quantize_blockwise(const float *code, const T *A, float *absmax, unsigned char *out, int blocksize, int n)
{
  int num_blocks = n/blocksize;
  num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;

  if(blocksize == 4096)
    kQuantizeBlockwise<T, 4096, 4, 0><<<num_blocks, 1024>>>(code, A, absmax, out, n);
  //else if(blocksize == 2048)
  //  kQuantizeBlockwise<T, 2048, 4, DATA_TYPE><<<num_blocks, 512>>>(code, A, absmax, out, n);
  //else if(blocksize == 1024)
  //  kQuantizeBlockwise<T, 1024, 4, DATA_TYPE><<<num_blocks, 256>>>(code, A, absmax, out, n);
  //else if(blocksize == 512)
  //  kQuantizeBlockwise<T, 512, 2, DATA_TYPE><<<num_blocks, 256>>>(code, A, absmax, out, n);
  //else if(blocksize == 256)
  //  kQuantizeBlockwise<T, 256, 2, DATA_TYPE><<<num_blocks, 128>>>(code, A, absmax, out, n);
  //else if(blocksize == 128)
  //  kQuantizeBlockwise<T, 128, 2, DATA_TYPE><<<num_blocks, 64>>>(code, A, absmax, out, n);
  //else if(blocksize == 64)
  //  kQuantizeBlockwise<T, 64, 2, DATA_TYPE><<<num_blocks, 32>>>(code, A, absmax, out, n);


  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}


template<typename T, int DATA_TYPE> void dequantize_blockwise(float *code, unsigned char *A, float *absmax, T *out, int blocksize, int n)
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

//template void quantize_blockwise<half, General8bit>(const float *code, const half *A, float *absmax, unsigned char *out, int blocksize, int n);
//template void quantize_blockwise<half, FP4>(const float *code, const half *A, float *absmax, unsigned char *out, int blocksize, int n);
//template void quantize_blockwise<half, NF4>(const float *code, const half *A, float *absmax, unsigned char *out, int blocksize, int n);
template void quantize_blockwise<float, General8bit>(const float *code, const float *A, float *absmax, unsigned char *out, int blocksize, int n);
//template void quantize_blockwise<float, FP4>(const float *code, const float *A, float *absmax, unsigned char *out, int blocksize, int n);
//template void quantize_blockwise<float, NF4>(const float *code, const float *A, float *absmax, unsigned char *out, int blocksize, int n);
//template void quantize_blockwise<__nv_bfloat16, General8bit>(const float *code, const __nv_bfloat16 *A, float *absmax, unsigned char *out, int blocksize, int n);
//template void quantize_blockwise<__nv_bfloat16, FP4>(const float *code, const __nv_bfloat16 *A, float *absmax, unsigned char *out, int blocksize, int n);
//template void quantize_blockwise<__nv_bfloat16, NF4>(const float *code, const __nv_bfloat16 *A, float *absmax, unsigned char *out, int blocksize, int n);
//
//template void dequantize_blockwise<float, General8bit>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, int n);
//template void dequantize_blockwise<float, FP4>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, int n);
//template void dequantize_blockwise<float, NF4>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, int n);
//template void dequantize_blockwise<half, General8bit>(float *code, unsigned char *A, float *absmax, half *out, int blocksize, int n);
//template void dequantize_blockwise<half, FP4>(float *code, unsigned char *A, float *absmax, half *out, int blocksize, int n);
//template void dequantize_blockwise<half, NF4>(float *code, unsigned char *A, float *absmax, half *out, int blocksize, int n);
//template void dequantize_blockwise<__nv_bfloat16, General8bit>(float *code, unsigned char *A, float *absmax, __nv_bfloat16 *out, int blocksize, int n);
//template void dequantize_blockwise<__nv_bfloat16, FP4>(float *code, unsigned char *A, float *absmax, __nv_bfloat16 *out, int blocksize, int n);
//template void dequantize_blockwise<__nv_bfloat16, NF4>(float *code, unsigned char *A, float *absmax, __nv_bfloat16 *out, int blocksize, int n);

std::vector<paddle::Tensor> QuantizeBlockwise(const paddle::Tensor& input, const paddle::Tensor& code, int blocksize, int n) {
    auto input_num = input.numel();
    std::vector<int64_t> out_shape = {input_num, 1};
    if (code.data() != 0) { // 4bit
        out_shape[0] = input_num / 2;
    }
    auto out = paddle::empty(out_shape, paddle::DataType::UINT8, input.place());
    int64_t absmax_shape = input_num / blocksize;
    auto absmax = paddle::empty({absmax_shape}, paddle::DataType::FLOAT32, input.place());
    switch(input.type()) {
        case paddle::DataType::FLOAT32:
            quantize_blockwise<float, General8bit>(code.data<float>(), input.data<float>(), absmax.data<float>(), out.data<unsigned char>(), blocksize, n);
            return {out, absmax};
        default:
            PD_THROW(
                "NOT supported data type. "
                "Only float16, bfloat16 and float32 are supported. ");
            break;
    }
};

std::vector<std::vector<int64_t>> GetQuantizeBlockwiseInferShape(const std::vector<int64_t>& input_shape, const std::vector<int64_t>& code_shape){
    return {input_shape};
}

std::vector<paddle::DataType> GetQuantizeBlockwiseInferDtype(const paddle::DataType& input_dtype, const paddle::DataType& code_dtype){
    return {paddle::DataType::UINT8};
}


PD_BUILD_OP(quant_blockwise)
    .Inputs({"input", "code"})
    .Outputs({"output", "abs_max"})
    .Attrs({"blocksize: int", "n: int"})
    .SetKernelFn(PD_KERNEL(QuantizeBlockwise))
    .SetInferShapeFn(PD_INFER_SHAPE(GetQuantizeBlockwiseInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetQuantizeBlockwiseInferDtype));

//PD_BUILD_OP(dequant_blockwise)
//    .Inputs({"input", "code", "abs_max", "blocksize", "n"})
//    .Outputs({"output"})
//    .SetKernelFn(PD_KERNEL(DequantizeBlockwise))
//    .SetInferShapeFn(PD_INFER_SHAPE(GetDequantizeBlockwiseInferShape))
//    .SetInferDtypeFn(PD_INFER_DTYPE(GetDequantizeBlockwiseInferDtype));
//

