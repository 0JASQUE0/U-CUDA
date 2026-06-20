// ls1d.template.cu
//
// NVRTC-шаблон для 1D спектра Ляпунова (LS). Подставляется ParametricEngine'ом
// тем же способом, что bifurcation1d/lle1d. Алгоритм — NonLinAnal::LSKernelCUDA
// (cudaLibrary.cu:2732), Wolf/Benettin с Gram-Schmidt'ом для N ортогональных
// перенормализаций. Раньше был отрезан под #ifndef __CUDACC_RTC__ вместе с
// curand_kernel.h и gramSchmidtProcess; теперь гард сужен, оба видны.

#define AMOUNTOFX {{AMOUNT_OF_X}}
// par_or_var выбирает свип-режим compile-time (LSKernelCUDA читает макрос,
// не runtime-аргумент). 1 = PARAMETER sweep, 0 = INITIAL CONDITION sweep.
#define par_or_var {{PAR_OR_VAR}}

#ifdef __CUDACC_RTC__
typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef short              int16_t;
typedef unsigned short     uint16_t;
typedef int                int32_t;
typedef unsigned int       uint32_t;
typedef long long          int64_t;
typedef unsigned long long uint64_t;
#endif

#include "cudaLibrary.cuh"

__device__ __host__ __forceinline__
void calculateDiscreteModel(numb* X, const numb* a, const numb h) {
{{KRS_BODY}}
}

#include "cudaLibrary.cu"
