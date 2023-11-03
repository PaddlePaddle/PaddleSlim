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
