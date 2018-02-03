/**
  * Author: Felix Bogdanski, since 02.02.2018
  */

#define MAKE_NAME(prefix, T) prefix ## _ ## T


#define MAKE_CONVOLUTE(T)\
extern "C"\
__global__ void MAKE_NAME(convolute, T)(\
  T *in, T *out, int IX, int IY, int X, int Y, int Z, int BS,\
  int FX, int FY, int SX, int SY, int PX, int PY) {\
  int XB = X * BS;\
  int OS = FX * FY * Z;\
  int x = threadIdx.x + blockIdx.x * blockDim.x;\
  int y = threadIdx.y + blockIdx.y * blockDim.y;\
  int z = threadIdx.z + blockIdx.z * blockDim.z;\
  if (x < XB && y < Y && z < Z) {\
    int fX = 0;\
    while (fX < FX) {\
      int fY = 0;\
      while (fY < FY) {\
        int xs = x % X;\
        int xb = x / X;\
        int a = (xs * SX) + fX;\
        int b = (y * SY) + fY;\
        if (a >= PX && a < (PX + IX) &&\
            b >= PY && b < (PY + IY)) {\
          int aNp = a - PX;\
          int bNp = b - PY;\
          int rowOut = (z * FX * FY) + fX * FY + fY;\
          int colIn  = (xb * IX * IY) + aNp * IY + bNp;\
          int colOut = (xb * X * Y) + xs * Y + y;\
          out[rowOut + colOut * OS] = in[z + colIn * Z];\
        }\
        fY++;\
      }\
      fX++;\
    }\
  }\
}\


#define MAKE_CONVOLUTE_BP(T)\
extern "C"\
__global__ void MAKE_NAME(convolute_bp, T)(\
  T *in, T *out, int IX, int IY, int X, int Y, int Z, int BS,\
  int FX, int FY, int SX, int SY, int PX, int PY) {\
  int XB = X * BS;\
  int OS = FX * FY * Z;\
  int x = threadIdx.x + blockIdx.x * blockDim.x;\
  int y = threadIdx.y + blockIdx.y * blockDim.y;\
  int z = threadIdx.z + blockIdx.z * blockDim.z;\
  if (x < XB && y < Y && z < Z) {\
    int fX = 0;\
    while (fX < FX) {\
      int fY = 0;\
      while (fY < FY) {\
        int xs = x % X;\
        int xb = x / X;\
        int a = (xs * SX) + fX;\
        int b = (y * SY) + fY;\
        if (a >= PX && a < (PX + IX) &&\
            b >= PY && b < (PY + IY)) {\
          int aNp = a - PX;\
          int bNp = b - PY;\
          int rowOut = (z * FX * FY) + fX * FY + fY;\
          int colIn  = (xb * X * Y) + xs * Y + y;\
          int colOut = (xb * IX * IY) + aNp * IY + bNp;\
          out[rowOut + colOut * OS] = in[z + colIn * Z];\
        }\
        fY++;\
      }\
      fX++;\
    }\
  }\
}\


#define MAKE_RESHAPE_BATCH(T)\
extern "C"\
__global__ void MAKE_NAME(reshape_batch, T)(\
  T *in, T *out, int X, int Y, int Z, int BS) {\
  int x = threadIdx.x + blockIdx.x * blockDim.x;\
  int y = threadIdx.y + blockIdx.y * blockDim.y;\
  if (x < X * Y * Z && y < BS) {\
    int a = x % (X * Y);\
    int b = x / (X * Y);\
    int c = y * (X * Y);\
    out[y + x * BS] = in[b + (c + a) * Z];\
  }\
}\


#define MAKE_RESHAPE_BATCH_BP(T)\
extern "C"\
__global__ void MAKE_NAME(reshape_batch_bp, T)(\
  T *in, T *out, int X, int Y, int Z, int BS) {\
  int x = threadIdx.x + blockIdx.x * blockDim.x;\
  int y = threadIdx.y + blockIdx.y * blockDim.y;\
  if (x < X * Y * Z && y < BS) {\
    int a = x % (X * Y);\
    int b = x / (X * Y);\
    int c = y * (X * Y);\
    out[b + (c + a) * Z] = in[y + x * BS];\
  }\
}\



MAKE_CONVOLUTE(TYPE)
MAKE_CONVOLUTE_BP(TYPE)
MAKE_RESHAPE_BATCH(TYPE)
MAKE_RESHAPE_BATCH_BP(TYPE)


