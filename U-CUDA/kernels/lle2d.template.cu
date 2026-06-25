// lle2d.template.cu
//
// NVRTC-шаблон для 2D-вычисления старшего показателя Ляпунова (LLE) по двум
// параметрам (или начальным условиям) на квадратной сетке nPts × nPts. Точная
// копия lle1d.template.cu по структуре — отличие только в runtime-аргументах
// kernel'а (см. parametric_engine.cpp::run_lle_2d): dimension=2, ranges[4],
// indicesOfMutVars[2], nPtsLimiter по cell'ам, nPts — сторона сетки.
//
// Сам kernel LLEKernelCUDA — универсальный (cudaLibrary.cu:2380); его ветка
// `dimension` обрабатывает 1D/2D/N-D через getValueByIdx с valueNumber.
// Плейсхолдеры:
//   AMOUNT_OF_X — размерность системы (число переменных, int)
//   KRS_BODY    — тело calculateDiscreteModel из codegen
//   PAR_OR_VAR  — 1 (оба свипа по параметру), 0 (оба по IC), 2 (X=IC, Y=param)

#define AMOUNTOFX {{AMOUNT_OF_X}}

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
