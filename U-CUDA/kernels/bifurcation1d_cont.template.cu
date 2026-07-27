// bifurcation1d_cont.template.cu
//
// NVRTC-шаблон для 1D-бифуркации в режиме continuation: одна точка параметра
// стартует с КОНЕЧНОГО x[] предыдущей. Реализовано single-thread'ом на GPU —
// параллельный path несовместим с continuation (между параметрами нужна
// последовательность). Скомпилированный KRS пользователя переиспользуется
// через `loopCalculateDiscreteModel_int` из cudaLibrary.cu.
//
// Плейсхолдеры:
//   AMOUNT_OF_X — размерность системы.
//   KRS_BODY    — тело calculateDiscreteModel из codegen.
//
// После kernel'а engine вызывает peakFinderCUDA (из этого же модуля — он
// объявлен в cudaLibrary.cu и автоматически компилируется в PTX).

#define AMOUNTOFX {{AMOUNT_OF_X}}
#define par_or_var 1   // continuation сейчас только param-sweep.

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

// Single-thread kernel. gridDim=blockDim=1.
// d_data         — буфер [nPts * sizeOfBlock] для трактории (writable_var).
// d_amountOfPeaks — [nPts]; ставим 1 / -1 для совместимости с peakFinderCUDA.
// sweepIsH != 0 -- the swept quantity is the step h itself: a[] stays put and
//   the per-point step counts are recomputed from h. sizeOfBlock is then the
//   worst case (smallest h), and the real per-point length is reported through
//   d_actualIterations so peakFinderCUDA only scans the valid prefix.
// logScale != 0 -- points are distributed logarithmically over [lo, hi].
extern "C" __global__ void bifurcation1dContinuationKernel(
    int nPts,
    numb lo,
    numb hi,
    bool reverse,
    int logScale,
    int sweepIsH,
    int mutParamIdx,                 // 1-based индекс в a[]; ignored if sweepIsH
    const numb* baseValues,
    int amountOfValues,
    const numb* baseX,
    int amountOfX,
    numb hBase,
    numb tMax,
    numb transientTime,
    int sizeOfBlock,                 // worst-case allocation per point
    int preScaller,
    int writableVar,
    numb maxValue,
    numb* d_data,
    int*  d_amountOfPeaks,
    int*  d_actualIterations)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    numb x[AMOUNTOFX];
    numb a[64];   // kMaxAmountOfValues в engine

    for (int i = 0; i < amountOfX; ++i)    x[i] = baseX[i];
    for (int i = 0; i < amountOfValues; ++i) a[i] = baseValues[i];

    numb denom = (numb)(nPts > 1 ? nPts - 1 : 1);
    for (int j = 0; j < nPts; ++j) {
        numb t = (numb)j / denom;
        numb p;
        if (logScale) {
            numb l0 = log10(lo), l1 = log10(hi);
            p = pow((numb)10.0, reverse ? (l1 - (l1 - l0) * t) : (l0 + (l1 - l0) * t));
        } else {
            // forward: p = lo + (hi-lo)*j/(n-1)
            // reverse: p = hi - (hi-lo)*j/(n-1)
            p = reverse ? (hi - (hi - lo) * t) : (lo + (hi - lo) * t);
        }

        numb hLocal = hBase;
        if (sweepIsH) hLocal = p;
        else          a[mutParamIdx] = p;

        int blockLen = (hLocal > 0) ? (int)(tMax / hLocal / (numb)preScaller) : 0;
        if (blockLen > sizeOfBlock) blockLen = sizeOfBlock;
        int transient_steps = (hLocal > 0) ? (int)(transientTime / hLocal) : 0;
        if (d_actualIterations != nullptr) d_actualIterations[j] = blockLen;

        if (hLocal <= 0 || blockLen <= 0) { d_amountOfPeaks[j] = -1; continue; }

        // Transient: x[] не сбрасываем — карри-овер с прошлой итерации.
        int flag = loopCalculateDiscreteModel_int(
            x, a, hLocal, transient_steps, amountOfX, 1, 0,
            maxValue, nullptr, (size_t)j * sizeOfBlock, 1);
        if (flag == 0) { d_amountOfPeaks[j] = -1; continue; }

        // Запись траектории: writableVar пишется в d_data на каждой
        // итерации loopCalculateDiscreteModel_int.
        flag = loopCalculateDiscreteModel_int(
            x, a, hLocal, blockLen, amountOfX, preScaller, writableVar,
            maxValue, d_data, (size_t)j * sizeOfBlock, 1);
        d_amountOfPeaks[j] = (flag == -1) ? -1 : 1;
    }
}
