#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

__device__ inline TYPE relu_a(TYPE a) { if (a > 0.0) return a; else return 0.0; }
__device__ inline TYPE relu_d(TYPE a) { if (a > 0.0) return 1.0; else return 0.0; }

__device__ inline TYPE linear_a(TYPE a) { return a; }
__device__ inline TYPE linear_d(TYPE a) { return 1.0; }

__device__ inline TYPE tanh_a(TYPE a) { return tanh(a); }
__device__ inline TYPE tanh_d(TYPE a) { return 1 - pow(tanh(a), 2.0); }

__device__ inline TYPE sigmoid_a(TYPE a) { return 1.0 / (1.0 + exp(-a)); }
__device__ inline TYPE sigmoid_d(TYPE a) { return exp(a) / pow(exp(a) + 1.0, 2.0); }

__device__ inline TYPE negate(TYPE a) { return -a; }

MAP_FUN_1(relu_a, TYPE)
MAP_FUN_1(relu_d, TYPE)
MAP_FUN_1(linear_a, TYPE)
MAP_FUN_1(linear_d, TYPE)
MAP_FUN_1(tanh_a, TYPE)
MAP_FUN_1(tanh_d, TYPE)
MAP_FUN_1(sigmoid_a, TYPE)
MAP_FUN_1(sigmoid_d, TYPE)
MAP_FUN_1(negate, TYPE)
MAP_FUN_1(acos, TYPE)
MAP_FUN_1(acosh, TYPE)
MAP_FUN_1(asin, TYPE)
MAP_FUN_1(asinh, TYPE)
MAP_FUN_1(atan, TYPE)
MAP_FUN_1(atanh, TYPE)
MAP_FUN_1(cbrt, TYPE)
MAP_FUN_1(ceil, TYPE)
MAP_FUN_1(cos, TYPE)
MAP_FUN_1(cosh, TYPE)
MAP_FUN_1(cospi, TYPE)
MAP_FUN_1(erfc, TYPE)
MAP_FUN_1(erfcinv, TYPE)
MAP_FUN_1(erfcx, TYPE)
MAP_FUN_1(erf, TYPE)
MAP_FUN_1(erfinv, TYPE)
MAP_FUN_1(exp10, TYPE)
MAP_FUN_1(exp2, TYPE)
MAP_FUN_1(exp, TYPE)
MAP_FUN_1(expm1, TYPE)
MAP_FUN_1(fabs, TYPE)
MAP_FUN_1(floor, TYPE)
MAP_FUN_1(j0, TYPE)
MAP_FUN_1(j1, TYPE)
MAP_FUN_1(lgamma, TYPE)
MAP_FUN_1(log10, TYPE)
MAP_FUN_1(log1p, TYPE)
MAP_FUN_1(log2, TYPE)
MAP_FUN_1(logb, TYPE)
MAP_FUN_1(log, TYPE)
MAP_FUN_1(nearbyint, TYPE)
MAP_FUN_1(normcdf, TYPE)
MAP_FUN_1(normcdfinv, TYPE)
MAP_FUN_1(rcbrt, TYPE)
MAP_FUN_1(rint, TYPE)
MAP_FUN_1(round, TYPE)
MAP_FUN_1(rsqrt, TYPE)
MAP_FUN_1(sin, TYPE)
MAP_FUN_1(sinh, TYPE)
MAP_FUN_1(sinpi, TYPE)
MAP_FUN_1(sqrt, TYPE)
MAP_FUN_1(tan, TYPE)
MAP_FUN_1(tanh, TYPE)
MAP_FUN_1(tgamma, TYPE)
MAP_FUN_1(trunc, TYPE)
MAP_FUN_1(y0, TYPE)
MAP_FUN_1(y1, TYPE)


__device__ inline TYPE add(TYPE a, TYPE b) { return a + b; }
__device__ inline TYPE sub(TYPE a, TYPE b) { return a - b; }
__device__ inline TYPE mul(TYPE a, TYPE b) { return a * b; }
__device__ inline TYPE div(TYPE a, TYPE b) { return a / b; }
__device__ inline TYPE mod(TYPE a, TYPE b) { return fmod(a, b); }
__device__ inline TYPE set(TYPE a, TYPE b) { return b; }

MAP_FUN_2(add, TYPE)
MAP_FUN_2(sub, TYPE)
MAP_FUN_2(mul, TYPE)
MAP_FUN_2(div, TYPE)
MAP_FUN_2(mod, TYPE)
MAP_FUN_2(pow, TYPE)
MAP_FUN_2(max, TYPE)
MAP_FUN_2(min, TYPE)
MAP_FUN_2(set, TYPE)

REDUCE_FUN(add, TYPE, 0)         
REDUCE_FUN(max, TYPE, -INFINITY)         
REDUCE_FUN(min, TYPE, INFINITY) 
