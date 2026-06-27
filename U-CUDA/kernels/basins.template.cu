// basins.template.cu
//
// NVRTC-шаблон для расчёта бассейнов притяжения. Свип ВСЕГДА по двум
// начальным условиям — par_or_var=0 жёстко. Сетка nPts × nPts.
//
// Pipeline (хост-оркестрация в parametric_engine.cpp::run_basins):
//   1. calculateDiscreteModelCUDA (dimension=2, par_or_var=0) — траектория +
//      helpfulArray flag (-1=FP / 0=Unbound / 1=Osc) в `maxValueCheckerArray`.
//   2. avgPeakFinderCUDA — d_data перезаписывается пиками, d_intervals —
//      межпиковыми; d_avgPeaks / d_avgIntervals — агрегаты.
//   3. CUDA_dbscan_* (три kernel'а: search_fixed_points / search_clear_points
//      / kernel-cluster-expand) — host-цикл из hostLibrary.cu::CUDA_dbscan.
//
// Плейсхолдеры:
//   AMOUNT_OF_X — размерность системы (число переменных, int)
//   KRS_BODY    — тело calculateDiscreteModel из codegen

#define AMOUNTOFX {{AMOUNT_OF_X}}

// par_or_var жёстко 0: оба свипа — по начальным условиям (basins of attraction
// — это всегда карта в плоскости IC). Менять не нужно, поэтому не плейсхолдер.
#define par_or_var 0

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

// curand_kernel.h: перехват для CUDA 13+ (см. lle1d.template.cu для деталей).
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
