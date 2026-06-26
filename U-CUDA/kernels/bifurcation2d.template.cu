// bifurcation2d.template.cu
//
// NVRTC-шаблон для 2D-бифуркации (хитмап «период(p1, p2)» через DBSCAN).
// Структура идентична bifurcation1d.template.cu; отличие — в runtime-аргументах
// kernel'а (см. parametric_engine.cpp::run_bif2d): dimension=2, ranges[4],
// indicesOfMutVars[2], общее число ячеек nPts*nPts (чанкируется как LLE-2D).
//
// Цепочка ядер на каждый чанк:
//   calculateDiscreteModelCUDA (dimension=2) -> peakFinderCUDA -> dbscanCUDA
// Результат dbscanCUDA[cell] = число кластеров пиков = период системы.
//
// Плейсхолдеры:
//   AMOUNT_OF_X — размерность системы (число переменных, int)
//   KRS_BODY    — тело calculateDiscreteModel из codegen
//   PAR_OR_VAR  — 1 (обе оси по param), 0 (обе по IC), 2 (X=IC, Y=param)

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
