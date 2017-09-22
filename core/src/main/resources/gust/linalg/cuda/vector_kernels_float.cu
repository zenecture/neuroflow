#include <stdio.h>

#define MAKE_NAME(prefix, fun, T) prefix ## _ ## fun ## _ ## T

#define MAP_FUN_1(fun, T) \
  extern "C" \
__global__ void MAKE_NAME(map, fun, T) (int length,\
    T *out, int outStride,\
    const T *in, int inStride) {\
  for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < length; i += blockDim.x * gridDim.x) {\
    out[i * outStride] = fun(in[i * inStride]);\
  }\
}



#define MAP_FUN_2(fun, T) \
extern "C" \
__global__ void MAKE_NAME(map2, fun, T) (int length,\
    T *out, int outStride,\
    const T *a, int aStride,\
    const T *b, int bStride) {\
  for(int col = threadIdx.x + blockIdx.x * blockDim.x; col < length; col += blockDim.x * gridDim.x) {\
    out[col * outStride] = fun(a[col * aStride], b[col * bStride]);\
  }\
}\
\
extern "C" \
__global__ void MAKE_NAME(map2_v_s, fun, T) (int length,\
    T *out, int outStride,\
    const T *a, int aStride,\
    const T b) {\
  for(int col = threadIdx.x + blockIdx.x * blockDim.x; col < length; col += blockDim.x * gridDim.x) {\
    out[col * outStride] = fun(a[col * aStride], b);\
  }\
}\
\
extern "C" \
__global__ void MAKE_NAME(map2_s_v, fun, T) (int length,\
    T *out, int outStride,\
    const T a,\
    const T *b, int bStride) {\
  for(int col = threadIdx.x + blockIdx.x * blockDim.x; col < length; col += blockDim.x * gridDim.x) {\
    out[col * outStride] = fun(a, b[col * bStride]);\
  }\
}\




 static __inline__ __device__ double shfl_down(double var, int delta, int width=warpSize)
{
    int hi, lo;
    asm volatile( "mov.b64 { %0, %1 }, %2;" : "=r"(lo), "=r"(hi) : "d"(var) );
    hi = __shfl_down( hi, delta, width );
    lo = __shfl_down( lo, delta, width );
    return __hiloint2double( hi, lo );
}

static __inline__ __device__ int shfl_down(int var, int delta, int width=warpSize)
{
    return __shfl_down(var, delta, width);
}

static __inline__ __device__ unsigned int shfl_down(unsigned int var, int delta, int width=warpSize)
{
    int x = __shfl_down(*(int*)&var, delta, width);
    return *(unsigned int*)(&x);
}

static __inline__ __device__ float shfl_down(float var, int delta, int width=warpSize)
{
    return __shfl_down(var, delta, width);
}

#define laneId (threadIdx.x & 0x1f)



#define REDUCE_FUN(fun, T, identity) \
/* Each column gets 1 block of threads. TODO currently blocksize must be 1 warp*/\
extern "C" \
__global__ void MAKE_NAME(reduce, fun, T) (int length,\
    T *out,\
    const T *in, int inStride) {\
\
  T sum = identity;\
  for(int col = threadIdx.x + blockIdx.x * blockDim.x; col < length; col += blockDim.x * gridDim.x) {\
    sum = fun(sum, in[col * inStride]);\
  }\
  \
  __syncthreads();\
  for (int i = 1; i < blockDim.x; i *= 2) {\
    T x = shfl_down(sum, i);\
    sum = fun(sum, x);\
  }\
  \
  if(laneId == 0) {\
    out[blockIdx.x] = sum;\
  }\
}

#include "function_decls.cuh"
