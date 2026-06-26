// bifurcation1d.template.cu
//
// NVRTC-шаблон для 1D-бифуркации. Подставляется ParametricEngine'ом, потом
// компилируется через nvrtcCompileProgram. Плейсхолдеры:
//   AMOUNT_OF_X — размерность системы (число переменных, int)
//   KRS_BODY    — тело calculateDiscreteModel из codegen
// (без двойных фигурных скобок в комментариях — replace_all дубовый и заменит
//  их прямо тут, поломав код за пределами функции).
//
// Логика:
//   1. Подставляем AMOUNTOFX -> переопределяем размер через #define
//      (configCUDA.h оборачивает свой constexpr в #ifndef AMOUNTOFX).
//   2. Подключаем cudaLibrary.cuh — оттуда придут типы (numb), макросы, объявления.
//   3. Определяем calculateDiscreteModel с user's KRS. NVRTC видит её
//      до подключения cudaLibrary.cu, где default-версия закрыта #ifndef __CUDACC_RTC__.
//   4. Подключаем cudaLibrary.cu — kernel-ы (calculateDiscreteModelCUDA,
//      peakFinderCUDA и helpers) подхватятся и слинкуются с нашей KRS.

#define AMOUNTOFX {{AMOUNT_OF_X}}

// par_or_var выбирает свип-режим compile-time (cudaLibrary.cu использует ЭТОТ
// макрос, не runtime-аргумент Par_or_Var из сигнатуры kernel'а).
//   1 = PARAMETER sweep (перебирается localValues[...]).
//   0 = INITIAL CONDITION sweep (перебирается localX[...]).
// configCUDA.h оборачивает дефолт в #ifndef par_or_var — конфликта нет.
#define par_or_var {{PAR_OR_VAR}}

// NVRTC не подтягивает <cstdint> автоматически, а cudaLibrary.cu использует
// int64_t в getValueByIdx и др. Даём typedef'ы для базовых fixed-width int'ов
// только под NVRTC; offline nvcc эти типы получает через <cstdint> в обычной
// сборке.
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

// curand_kernel.h: перехват для CUDA 13+, где реальный header тянет
// <cuda/std/type_traits> (libcudacxx), которого NVRTC из коробки не находит.
// Pre-define guard'ов реального curand_kernel.h + inline-stub использованных
// в LLE/LS символов. Любой последующий `#include <curand_kernel.h>` из
// cudaLibrary.cuh становится no-op независимо от того, что NVRTC находит
// на диске через -I CUDA_PATH/include. Под не-NVRTC сборку блок отключён.
#ifdef __CUDACC_RTC__
#define CURAND_KERNEL_H_
#define CURAND_KERNEL_H
typedef struct { unsigned long long state; } curandState_t;
typedef curandState_t curandStateXORWOW_t;
__device__ __forceinline__ void curand_init(
    unsigned long long seed, unsigned long long, unsigned long long, curandState_t* s) {
    s->state = seed * 6364136223846793005ULL + 1442695040888963407ULL;
}
__device__ __forceinline__ float curand_uniform(curandState_t* s) {
    s->state = s->state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)(((s->state >> 40) & 0xFFFFFFULL) + 1ULL) / 16777216.0f;
}
#endif

#include "cudaLibrary.cuh"

__device__ __host__ __forceinline__
void calculateDiscreteModel(numb* X, const numb* a, const numb h) {
{{KRS_BODY}}
}

#include "cudaLibrary.cu"
