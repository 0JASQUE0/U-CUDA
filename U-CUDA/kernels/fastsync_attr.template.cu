// fastsync_attr.template.cu
//
// NVRTC-шаблон для Fast Synchro в режиме "On Attractor".
// Pipeline (host-оркестрация в parametric_engine::run_fastsync, mode=0):
//   1. integrateMasterTrajectoryFS — single-thread kernel, заливает d_timeDomain
//      полным вектором состояния X[AMOUNTOFX] на каждом шаге (через генерируемую
//      из KRS_BODY функцию calculateDiscreteModel). Сначала amountOfPointsForSkip
//      transient-шагов "вхолостую", потом amountOfPoints сохраняемых.
//      ВАЖНО: пишем полный X[], а не скалярную сводку, как делает
//      calculateDiscreteModelCUDA — последняя для bif/peakFinder и для FS не подходит.
//   2. calculateDiscreteModelforFastSynchroCUDA (из cudaLibrary.cu) — параллельно
//      на nPts threads читает окно amountOfNTPoints точек из timeDomain и
//      возвращает RMS-ошибку синхронизации в output[idx].
// Затем host копирует d_timeDomain (для отрисовки) и d_output (для colormap).
//
// Плейсхолдеры:
//   AMOUNT_OF_X     — размерность системы (int)
//   TYPE_OF_SYNCH   — 0 (unidir) | 1 (bidir)
//   ERROR_ESTIM     — 0|1|2 (см. configCUDA.h)
//   FS_ERROR_TRS    — float literal (порог ошибки)
//   KRS_BODY        — тело calculateDiscreteModel из codegen

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

// ---- Master trajectory filler (FS-specific) ----
// Single-thread интегратор master-системы: использует ИСКЛЮЧИТЕЛЬНО FS device
// function `calculateDiscreteModelforFastSynchro` с K=0 / S1=Xm (zero coupling
// = чистая интеграция master). Это полностью зеркалит то, что legacy host-side
// `FastSynchro()` в hostLibrary.cu делал CPU-loop'ом, и то, что
// `loopCalculateDiscreteModelForFastSynchro_2` делает inline в grid-mode.
//
// НЕ ИСПОЛЬЗУЕТ calculateDiscreteModelCUDA (та пишет скалярную сводку
// x[0]+pi*x[1]+e*x[2], а нам нужен полный вектор X[] на каждом шаге).
//
// Layout вывода:
//   timeDomain[i * AMOUNTOFX + j]  — j-я переменная на i-й итерации.
extern "C" __global__ void fillFSMasterTrajectory(
    const numb* values,
    const numb h,
    const numb* X0,
    const int amountOfPointsForSkip,
    const int amountOfPoints,
    numb* timeDomain)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    numb X[AMOUNTOFX];
    numb zeros[AMOUNTOFX];
    for (int i = 0; i < AMOUNTOFX; ++i) { X[i] = X0[i]; zeros[i] = 0; }

    // Transient — uncoupled forward (K=0, S1=X → coupling term зануляется).
    for (int i = 0; i < amountOfPointsForSkip; ++i)
        calculateDiscreteModelforFastSynchro(X, X, zeros, values, h, 1);

    // Save full state at every integration step.
    for (int i = 0; i < amountOfPoints; ++i) {
        for (int j = 0; j < AMOUNTOFX; ++j)
            timeDomain[(size_t)i * AMOUNTOFX + j] = X[j];
        calculateDiscreteModelforFastSynchro(X, X, zeros, values, h, 1);
    }
}
