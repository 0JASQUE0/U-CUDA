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
// splitmix64-подобное перемешивание: раньше sequence/offset игнорировались,
// и общий seed для всех потоков (см. cudaLibrary.cu) давал бы одинаковое
// состояние на каждый idx. Теперь sequence реально участвует в перемешивании.
__device__ __forceinline__ void curand_init(
    unsigned long long seed, unsigned long long sequence, unsigned long long offset, curandState_t* s) {
    unsigned long long z = seed + sequence * 0x9E3779B97F4A7C15ULL + offset;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    s->state = z ^ (z >> 31);
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
