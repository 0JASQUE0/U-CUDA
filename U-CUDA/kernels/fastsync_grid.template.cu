// fastsync_grid.template.cu
//
// NVRTC-шаблон для Fast Synchro в режиме "On Grid" (2D-карта по двум IC).
// Host-оркестрация — parametric_engine::run_fastsync(req, mode=1):
//   1. готовит ranges[4] = {x_lo, x_hi, y_lo, y_hi}, indicesOfMutVars[2].
//   2. cuLaunchKernel(calculateDiscreteModelICCforFastSynchro) — на каждую
//      ячейку сетки nPts×nPts (свип по IC) запускает synchro-цикл, пишет
//      ошибку в FastSynchroError[nPts²].
//   3. cudaMemcpyD2H FastSynchroError → FastSyncResult.heatmap.
//
// Плейсхолдеры идентичны fastsync_attr.template.cu (тот же runtime context).

#define AMOUNTOFX     {{AMOUNT_OF_X}}
#define type_of_synch {{TYPE_OF_SYNCH}}
#define error_estim   {{ERROR_ESTIM}}
#define FS_error_trs  {{FS_ERROR_TRS}}

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
