/**
  * Author: Felix Bogdanski, since 07.03.2018
  */

#define MAKE_NAME(prefix, T) prefix ## _ ## T


#define MAKE_SUBROWMAX(T)\
extern "C"\
__global__ void MAKE_NAME(subrowmax, T)(T *in, T *out, int R, int C) {\
  int row = threadIdx.x + blockIdx.x * blockDim.x;\
  if (row < R) {\
    int col = 0;\
    T max = in[row];\
    while (col < C) {\
      T t = in[row + col * R];\
      if (t > max) max = t;\
      col++;\
    }\
    col = 0;\
    while (col < C) {\
      T t = in[row + col * R];\
      out[row + col * R] = t - max;\
      col++;\
    }\
  }\
}\


MAKE_SUBROWMAX(TYPE)

