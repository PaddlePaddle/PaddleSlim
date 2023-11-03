#include <float.h>
#include "ops.cuh"

#ifndef kernels
#define kernels

template<typename T, int BLOCK_SIZE, int NUM_PER_TH, int DATA_TYPE> __global__ void kQuantizeBlockwise(const float * code, T * __restrict__ const A, float *absmax, unsigned char *out, int n);
template<typename T, int BLOCK_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE> __global__ void kDequantizeBlockwise(float *code, unsigned char * A, float * absmax, T *out, const int blocksize, const int n);

#endif
