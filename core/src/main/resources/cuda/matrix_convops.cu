
#define MAKE_NAME(prefix, T) prefix ## _ ## T

#define MAKE_FUN(fun, T) \
extern "C" \
__global__ void MAKE_NAME(im2col, T)( \
	  T *in, T *out, \
	  int X, int Y, int Z, \
	  int layer, int fieldX, int fieldY, \
	  int paddingX, int paddingY, \
	  int strideX, int strideY) { \
	int x = 0; \
	int y = 0; \
	int z = 0; \
	if (x < (((X + paddingX - fieldX) / strideX) + 1) && (y < (((Y + paddingY - fieldY) / strideY) + 1)) && z < Z) { \
		int fX = 0; \
		int fY = 0; \
		while (fX < fieldX) { \
			while (fY < fieldY) { \
				int a = x + (fX * strideX); \
				int b = y + (fY * strideY); \
				int c = z * fieldX * fieldY; \
				if (a >= 0 && a < X && b >= 0 && b < Y) { \
				} \
				fY += 1; \
			} \
			fX += 1; \
		} \
	} \
} \

MAKE_FUN(im2col, TYPE)
