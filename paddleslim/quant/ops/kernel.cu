#include "kernel.cuh"
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

__device__ unsigned char dQuantizeNF4(float x)
{

  // the values for this tree was generated by test_normal_map_tree
  // in the file tests/test_functional.py
  if(x > 0.03979014977812767f)
    if(x > 0.3893125355243683f) // 1
      if(x > 0.6427869200706482f) // 11
        if(x > 0.8614784181118011f) // 111
          return 0b1111;
        else
          return 0b1110;
      else
        if(x > 0.5016634166240692f) // 110
          return 0b1101;
        else
          return 0b1100;
    else
      if(x > 0.2035212516784668f) // 10
        if(x > 0.2920137718319893f) // 101
          return 0b1011;
        else
          return 0b1010;
      else
        if(x > 0.1202552504837513f) // 100
          return 0b1001;
        else
          return 0b1000;
  else
    if(x > -0.33967943489551544f) // 0
      if(x > -0.13791173323988914f) // 01
        if(x > -0.045525018125772476f) // 011
          return 0b0111;
        else
          return 0b0110;
      else
        if(x > -0.23460740596055984f) // 010
          return 0b0101;
        else
          return 0b0100;
    else
      if(x > -0.6106329262256622f) // 00
        if(x > -0.4599952697753906f) // 001
          return 0b0011;
        else
          return 0b0010;
      else
        if(x > -0.8480964004993439f) // 000
          return 0b0001;
        else
          return 0b0000;
}
__device__ unsigned char dQuantizeFP4(float x)
{
  // FP4 with bias of 3
  // first bit is a sign
  // subnormals
  // 0b000 = 0
  // 0b001 = 0.0625
  // 0b110 = 2
  // 0b111 = 3
  // 0b100 = 4
  // 0b101 = 6
  // 0b010 = 8
  // 0b011 = 12


  // we do a binary search
  // the pivots are divided by 12 (the FP4 absmax)
  // since we assum input data is in [-1.0, 1.0]

  // !be careful here, its easy to make a mistake
  // that is difficult to noice if you add an extra
  // zero somewhere!

  int sign = x < 0 ? 0b1000 : 0b0000;
  x = fabsf(x);
  if(x > 0.29166667f)
    if( x > 0.583333f)
      if( x > 0.8333333f)
        return 0b0011+sign;
      else
        return 0b0010+sign;
    else
      if(x > 0.4166667f)
        return 0b101+sign;
      else
        return 0b100+sign;
  else
    if(x > 0.0859375f)
      if(x > 0.20833333f)
        return 0b0111+sign;
      else
        return 0b0110+sign;
    else
      if(x > 0.00260417f)
        return 0b0001+sign;
      else
        return 0b0000+sign;
}

__device__ unsigned char dQuantize(float* smem_code, float x)
{
    int pivot = 127;
    int upper_pivot = 255;
    int lower_pivot = 0;

    float lower = -1.0f;
    float upper = 1.0f;

    float val = smem_code[pivot];
    // i>>=1 = {32, 16, 8, 4, 2, 1}
    for(int i = 64; i > 0; i>>=1)
    {
        if(x > val)
        {
            lower_pivot = pivot;
            lower = val;
            pivot+=i;
        }
        else
        {
            upper_pivot = pivot;
            upper = val;
            pivot-=i;
        }
        val = smem_code[pivot];
    }

    if(upper_pivot == 255)
        upper = smem_code[upper_pivot];
    if(lower_pivot == 0)
        lower = smem_code[lower_pivot];

   if(x > val)
   {
     float midpoint = (upper+val)*0.5f;
     if(x > midpoint)
     {
       return upper_pivot;
     }
     else
       return pivot;
   }
   else
   {
     float midpoint = (lower+val)*0.5f;
     if(x < midpoint)
       return lower_pivot;
     else
       return pivot;
   }
}


template<typename T, int BLOCK_SIZE, int NUM_PER_TH, int DATA_TYPE>
//__launch_bounds__(TH, 4)
__global__ void kQuantizeBlockwise(const float * code, T * __restrict__ const A, float *absmax, unsigned char *out, int n)
{
  const int n_full = gridDim.x * BLOCK_SIZE;
  int valid_items = 0;
  const int base_idx = (blockIdx.x * BLOCK_SIZE);

  T vals[NUM_PER_TH];
  unsigned char qvals[(DATA_TYPE > 0) ? NUM_PER_TH/2 : NUM_PER_TH];
  //float local_abs_max = -FLT_MAX;
  float local_abs_max = 0.0f;

  typedef cub::BlockLoad<T, BLOCK_SIZE/NUM_PER_TH, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT;
  typedef cub::BlockStore<unsigned char, BLOCK_SIZE/NUM_PER_TH, (DATA_TYPE > 0) ? NUM_PER_TH/2 : NUM_PER_TH, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreChar;
  typedef cub::BlockReduce<float, BLOCK_SIZE/NUM_PER_TH> BlockReduce;
  typedef cub::BlockLoad<float, BLOCK_SIZE/NUM_PER_TH, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadFloat;

  __shared__ typename LoadT::TempStorage loadt;
  __shared__ typename LoadFloat::TempStorage loadf;
  __shared__ typename StoreChar::TempStorage storec;
  __shared__ typename BlockReduce::TempStorage reduce;
  __shared__ float smem_code[256];
  __shared__ float smem_absmax_value[1];

  if(DATA_TYPE == General8bit)
    for(int i = threadIdx.x; i < 256; i+=blockDim.x)
      smem_code[i] = code[i];

  for (unsigned int i = base_idx; i < n_full; i += gridDim.x*BLOCK_SIZE)
  {
    valid_items = n - i > BLOCK_SIZE ? BLOCK_SIZE : n - i;
    local_abs_max = -FLT_MAX;

    __syncthreads();
    LoadT(loadt).Load(&(A[i]), vals, valid_items, (T)0.0f);

    // 1. compute local max
    // 2. broadcast local max
    // 3. normalize inputs and quantize

    #pragma unroll NUM_PER_TH
    for(int j = 0; j < NUM_PER_TH; j++)
       local_abs_max = fmaxf(local_abs_max, fabsf((float)vals[j]));

    local_abs_max = BlockReduce(reduce).Reduce(local_abs_max, cub::Max(), valid_items);

    if(threadIdx.x == 0)
      smem_absmax_value[0] = local_abs_max;

    __syncthreads();

    if(threadIdx.x == 0)
      absmax[i/BLOCK_SIZE] = local_abs_max;
    else
      local_abs_max = smem_absmax_value[0];

    __syncwarp();

    local_abs_max = 1.0f/local_abs_max;

    unsigned char packed_4bit = 0;
    switch(DATA_TYPE)
    {
        case General8bit:
            #pragma unroll NUM_PER_TH
            for(int j = 0; j < NUM_PER_TH; j++)
            {
               qvals[j] = dQuantize(smem_code, ((float)vals[j])*local_abs_max);
            }
            break;
        case FP4:
            #pragma unroll NUM_PER_TH
            for(int j = 0; j < NUM_PER_TH/2; j++)
            {
              packed_4bit |= dQuantizeFP4(((float)vals[2*j])*local_abs_max) << 4;
              packed_4bit |= dQuantizeFP4(((float)vals[2*j+1])*local_abs_max);
              qvals[j] = packed_4bit;
            }
            break;
        case NF4:
            #pragma unroll NUM_PER_TH
            for(int j = 0; j < NUM_PER_TH/2; j++)
            {
              packed_4bit |= dQuantizeNF4(((float)vals[2*j])*local_abs_max) << 4;
              packed_4bit |= dQuantizeNF4(((float)vals[2*j+1])*local_abs_max);
              qvals[j] = packed_4bit;
            }
            break;
    }

    __syncthreads();
    StoreChar(storec).Store(&(out[(DATA_TYPE > 0) ? i/2 : i]), qvals, (DATA_TYPE > 0) ? (valid_items+1)/2 : valid_items);
  }
}

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
__global__ void kDequantizeBlockwise(float *code, unsigned char * A, float * absmax, T *out, const int blocksize, const int n)
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


#define MAKE_kQuantizeBlockwise(dtype, blocksize, num_per_thread, data_type_name) \
template __global__ void kQuantizeBlockwise<dtype, blocksize, num_per_thread, data_type_name>(const float * code, dtype * __restrict__ const A, float *absmax, unsigned char *out, int n); \

MAKE_kQuantizeBlockwise(half,  4096, 4, General8bit)
MAKE_kQuantizeBlockwise(half,  2048, 4, General8bit)
MAKE_kQuantizeBlockwise(half,  1024, 4, General8bit)
MAKE_kQuantizeBlockwise(half,   512, 2, General8bit)
MAKE_kQuantizeBlockwise(half,   256, 2, General8bit)
MAKE_kQuantizeBlockwise(half,   128, 2, General8bit)
MAKE_kQuantizeBlockwise(half,    64, 2, General8bit)
MAKE_kQuantizeBlockwise(half,  4096, 4, FP4)
MAKE_kQuantizeBlockwise(half,  2048, 4, FP4)
MAKE_kQuantizeBlockwise(half,  1024, 4, FP4)
MAKE_kQuantizeBlockwise(half,   512, 2, FP4)
MAKE_kQuantizeBlockwise(half,   256, 2, FP4)
MAKE_kQuantizeBlockwise(half,   128, 2, FP4)
MAKE_kQuantizeBlockwise(half,    64, 2, FP4)
MAKE_kQuantizeBlockwise(half,  4096, 4, NF4)
MAKE_kQuantizeBlockwise(half,  2048, 4, NF4)
MAKE_kQuantizeBlockwise(half,  1024, 4, NF4)
MAKE_kQuantizeBlockwise(half,   512, 2, NF4)
MAKE_kQuantizeBlockwise(half,   256, 2, NF4)
MAKE_kQuantizeBlockwise(half,   128, 2, NF4)
MAKE_kQuantizeBlockwise(half,    64, 2, NF4)
MAKE_kQuantizeBlockwise(float, 4096, 4, General8bit)
MAKE_kQuantizeBlockwise(float, 2048, 4, General8bit)
MAKE_kQuantizeBlockwise(float, 1024, 4, General8bit)
MAKE_kQuantizeBlockwise(float,  512, 2, General8bit)
MAKE_kQuantizeBlockwise(float,  256, 2, General8bit)
MAKE_kQuantizeBlockwise(float,  128, 2, General8bit)
MAKE_kQuantizeBlockwise(float,   64, 2, General8bit)
MAKE_kQuantizeBlockwise(float, 4096, 4, FP4)
MAKE_kQuantizeBlockwise(float, 2048, 4, FP4)
MAKE_kQuantizeBlockwise(float, 1024, 4, FP4)
MAKE_kQuantizeBlockwise(float,  512, 2, FP4)
MAKE_kQuantizeBlockwise(float,  256, 2, FP4)
MAKE_kQuantizeBlockwise(float,  128, 2, FP4)
MAKE_kQuantizeBlockwise(float,   64, 2, FP4)
MAKE_kQuantizeBlockwise(float, 4096, 4, NF4)
MAKE_kQuantizeBlockwise(float, 2048, 4, NF4)
MAKE_kQuantizeBlockwise(float, 1024, 4, NF4)
MAKE_kQuantizeBlockwise(float,  512, 2, NF4)
MAKE_kQuantizeBlockwise(float,  256, 2, NF4)
MAKE_kQuantizeBlockwise(float,  128, 2, NF4)
MAKE_kQuantizeBlockwise(float,   64, 2, NF4)

MAKE_kQuantizeBlockwise(__nv_bfloat16, 4096, 4, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 2048, 4, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 1024, 4, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  512, 2, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  256, 2, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  128, 2, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16,   64, 2, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 4096, 4, FP4)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 2048, 4, FP4)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 1024, 4, FP4)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  512, 2, FP4)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  256, 2, FP4)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  128, 2, FP4)
MAKE_kQuantizeBlockwise(__nv_bfloat16,   64, 2, FP4)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 4096, 4, NF4)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 2048, 4, NF4)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 1024, 4, NF4)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  512, 2, NF4)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  256, 2, NF4)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  128, 2, NF4)
MAKE_kQuantizeBlockwise(__nv_bfloat16,   64, 2, NF4)
