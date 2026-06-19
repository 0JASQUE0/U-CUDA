// lle1d.template.cu
//
// NVRTC-шаблон для 1D-вычисления старшего показателя Ляпунова (LLE) по
// параметру (свип). Архитектура та же, что у bifurcation1d.template.cu:
// плейсхолдеры подставляет ParametricEngine, потом nvrtcCompileProgram.
// Плейсхолдеры:
//   AMOUNT_OF_X — размерность системы (число переменных, int)
//   KRS_BODY    — тело calculateDiscreteModel из codegen
//
// Сам алгоритм Wolf/Benettin реализован в NonLinAnal — kernel `LLEKernelCUDA`
// в cudaLibrary.cu:2379. Раньше он был отрезан гардом #ifndef __CUDACC_RTC__
// из-за curand_kernel.h; теперь curand_kernel.h подключается под NVRTC и
// LLEKernelCUDA доступен. Здесь мы только готовим точку входа (KRS + amountOfX).

#define AMOUNTOFX {{AMOUNT_OF_X}}

// Параметрический свип — тот же режим, что и для bif1d.
#define par_or_var 1

// NVRTC не подтягивает <cstdint>, а cudaLibrary.cu использует int64_t.
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
