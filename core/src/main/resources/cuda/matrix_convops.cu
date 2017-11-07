/**
  * Author: Felix Bogdanski, since 10/2017
  */

#define MAKE_NAME(prefix, T) prefix ## _ ## T

#define MAKE_IM2COL(T) \
extern "C" \
__global__ void MAKE_NAME(im2col, T)( \
    T *in, T *out, int *idc,\
    int X, int Y, int Z, \
    int fieldX, int fieldY, \
    int paddingX, int paddingY, \
    int strideX, int strideY, int withIndices) { \
  int x = threadIdx.x + blockIdx.x * blockDim.x; \
  int y = threadIdx.y + blockIdx.y * blockDim.y; \
  int z = threadIdx.z + blockIdx.z * blockDim.z; \
  int XM = ((X + 2 * paddingX - fieldX) / strideX) + 1; \
  int YM = ((Y + 2 * paddingY - fieldY) / strideY) + 1; \
  int outStride = fieldX * fieldY * Z; \
  int idcStride = Y * fieldY; \
  if (x < XM && y < YM && z < Z) { \
    int fX = 0; \
    while (fX < fieldX) { \
      int fY = 0; \
      while (fY < fieldY) { \
        int a = fY + (y * strideY); \
        int b = fX + (x * strideX); \
        if (a >= paddingY && a < (Y + paddingY) && b >= paddingX && b < (X + paddingX)) { \
          int a_nop = a - paddingY; \
          int b_nop = b - paddingX; \
          int ab_lin = b_nop * Y + a_nop; \
          int i = (x * YM) + y; \
          int c = z * fieldX * fieldY; \
          int l = c + (fX * fieldY + fY); \
          out[i * outStride + l] = in[ab_lin * Z + z]; \
          if (withIndices == 1) { \
            int id_r = a_nop * fieldY + fY; \
            int id_c = b_nop * fieldX + fX; \
            idc[id_c * idcStride + id_r] = i + 1; \
          } \
        } \
        fY++; \
      } \
      fX++; \
    } \
  } \
} \

#define MAKE_IM2COL_BACKPROP(T) \
extern "C" \
__global__ void MAKE_NAME(im2col_backprop, T)( \
  T *in, int inStride, \
  T *out, int outStride, \
  int *idc, int idcStride, \
  int X, int Y, int Z, \
  int fieldX, int fieldY) { \
  int x = threadIdx.x + blockIdx.x * blockDim.x; \
  int y = threadIdx.y + blockIdx.y * blockDim.y; \
  int z = threadIdx.z + blockIdx.z * blockDim.z; \
  if (x < X && y < Y && z < Z) { \
    int fX = 0; \
    int p = 0; \
    while (fX < fieldX) { \
      int fY = 0; \
      while (fY < fieldY) { \
        int i_r = (y * fieldY) + fY; \
        int i_c = (x * fieldX) + fX; \
        int idx = idc[i_c * idcStride + i_r]; \
        if (idx > 0) { \
          int t_r = z * fieldX * fieldY + p; \
          int t_c = x * Y + y; \
          out[t_c * outStride + t_r] = in[(idx - 1) * inStride + z]; \
        } \
        p++; \
        fY++; \
      } \
      fX++; \
    } \
  } \
} \

MAKE_IM2COL(TYPE)
MAKE_IM2COL_BACKPROP(TYPE)
