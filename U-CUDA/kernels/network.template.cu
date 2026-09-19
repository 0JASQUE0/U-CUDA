// network.template.cu
// NVRTC-шаблон вкладки Network — сеть связанных осцилляторов на произвольной
// топологии.
//
// Один БЛОК считает одну сеть, один ПОТОК — один узел. Состояние всей сети
// лежит в динамической shared: связь требует читать состояние соседей на
// каждом шаге, и из global это был бы некэшируемый разброс по всей сети.
// Отсюда же потолок на число узлов (см. kMaxNetworkNodes в network_session.h):
// узлов не больше, чем потоков в блоке.
//
// Шаг — РАСЩЕПЛЁННЫЙ, ровно по конвенции calculateDiscreteModelforFastSynchro
// из cudaLibrary.cu:
//   1. по состоянию на НАЧАЛО шага считается вклад связи C_i (Якоби, не
//      Гаусс-Зейдель: порядок обхода узлов не влияет на результат);
//   2. узел делает свой шаг телом КРС;
//   3. X_i += h * C_i.
// Связь при этом интегрируется первым порядком независимо от порядка схемы —
// это цена расщепления, и она осознанная: так на вкладке работают ВСЕ схемы,
// включая неявные и пользовательские, тело которых движок не разбирает.
//
// Рёбра — CSR по узлу-ПРИЁМНИКУ: для узла i дуги лежат в
// [edgeStart[i], edgeStart[i+1]), каждая хранит источник edgeSrc[e], вес
// edgeW[e] (уже умноженный на глобальный множитель связи) и номер закона
// edgeLaw[e]. Двунаправленное ребро хост раскладывает в две дуги.
//
// Параметры у каждого узла СВОИ: values[i * AMOUNTOFVALUES + k]. Это и есть
// «менять параметры конкретных осцилляторов» — расстройка ансамбля не требует
// ни отдельного ядра, ни перекомпиляции.
//
// Плейсхолдеры:
//   AMOUNT_OF_X      — размерность одного узла
//   AMOUNT_OF_VALUES — размер a[] одного узла (a[0] = symmetry s)
//   KRS_BODY         — тело calculateDiscreteModel из codegen
//   COUPLING_BODY    — тело switch(law): case-ветки из codegen_coupling

#define AMOUNTOFX {{AMOUNT_OF_X}}
#define AMOUNTOFVALUES {{AMOUNT_OF_VALUES}}

// Шаблон не зовёт свип-ядра cudaLibrary.cu, но configCUDA.h внутри
// cudaLibrary.cuh требует par_or_var определённым.
#define par_or_var 1

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

// Статус узла в конце расчёта.
#define NET_OK       0
#define NET_DIVERGED 1

// Вклад ВСЕХ входящих дуг узла в его правую часть.
// Xi/ai — приёмник, Xj/aj — источник, w — вес дуги, C — накопитель: имена
// зашиты в умолчаниях codegen_coupling(), менять их надо в паре.
__device__ __forceinline__ void networkCoupling(
    const numb* __restrict__ sh,
    const numb* __restrict__ values,
    const int*  __restrict__ edgeStart,
    const int*  __restrict__ edgeSrc,
    const numb* __restrict__ edgeW,
    const int*  __restrict__ edgeLaw,
    const int node,
    numb* C)
{
    for (int k = 0; k < AMOUNTOFX; ++k) C[k] = (numb)0;

    const numb* Xi = sh + (size_t)node * AMOUNTOFX;
    const numb* ai = values + (size_t)node * AMOUNTOFVALUES;

    const int e0 = edgeStart[node];
    const int e1 = edgeStart[node + 1];
    for (int e = e0; e < e1; ++e) {
        const int src = edgeSrc[e];
        const numb* Xj = sh + (size_t)src * AMOUNTOFX;
        const numb* aj = values + (size_t)src * AMOUNTOFVALUES;
        const numb w = edgeW[e];
        switch (edgeLaw[e]) {
{{COUPLING_BODY}}
        default: break;
        }
    }
}

__device__ __forceinline__ bool networkNodeBad(const numb* X, const numb maxValue) {
    numb acc = (numb)0;
    for (int k = 0; k < AMOUNTOFX; ++k) acc += fabs(X[k]);
    if (isnan(acc) || isinf(acc)) return true;
    if (maxValue != (numb)0 && acc > maxValue) return true;
    return false;
}

// Один запуск = один ЧАНК шагов, состояние переносится через state[]: хост
// режет расчёт на чанки против watchdog'а (TDR), а не ради памяти.
//
// Нумерация точек вывода сквозная по всему расчёту, поэтому ядру нужен
// stepBase — номер первого шага чанка. Точка p пишется, когда сделано
// skipSteps + p * preScaller шагов, то есть первая точка — ровно конец
// транзиента.
//
// Выход: out[p * nNodes * AMOUNTOFX + node * AMOUNTOFX + k].
extern "C" __global__ void networkIntegrateKernel(
    const numb* __restrict__ values,
    numb* __restrict__ state,
    const int*  __restrict__ edgeStart,
    const int*  __restrict__ edgeSrc,
    const numb* __restrict__ edgeW,
    const int*  __restrict__ edgeLaw,
    const int       nNodes,
    const numb      h,
    const long long stepBase,
    const long long stepsThisLaunch,
    const long long skipSteps,
    const int       preScaller,
    const int       nPoints,
    const numb      maxValue,
    numb* __restrict__ out,
    int*  __restrict__ statusOut,
    const volatile int* cancelFlag,
    int* progressCounter,
    const int progressStride)
{
    extern __shared__ numb sh[];        // [nNodes][AMOUNTOFX] — состояние сети
    __shared__ int shStop;              // 0 — идём дальше, 1 — разлёт/отмена

    const int node = (int)threadIdx.x;
    const bool live = (node < nNodes);

    if (threadIdx.x == 0) shStop = 0;
    if (live)
        for (int k = 0; k < AMOUNTOFX; ++k)
            sh[(size_t)node * AMOUNTOFX + k] = state[(size_t)node * AMOUNTOFX + k];
    __syncthreads();

    // Узел, разлетевшийся в прошлом чанке, второй раз не считаем: статус уже
    // выставлен, а NaN в shared разъехался бы по соседям через связь.
    if (live && statusOut[node] != NET_OK) shStop = 1;   // гонка безобидна: все пишут 1
    __syncthreads();

    numb C[AMOUNTOFX];
    numb X[AMOUNTOFX];
    const numb* ai = values + (size_t)node * AMOUNTOFVALUES;

    for (long long s = 0; s < stepsThisLaunch && !shStop; ++s) {
        if (live)
            networkCoupling(sh, values, edgeStart, edgeSrc, edgeW, edgeLaw, node, C);
        __syncthreads();                // связь посчитана по состоянию на начало шага

        if (live) {
            for (int k = 0; k < AMOUNTOFX; ++k) X[k] = sh[(size_t)node * AMOUNTOFX + k];
            calculateDiscreteModel(X, ai, h);
            for (int k = 0; k < AMOUNTOFX; ++k) X[k] += h * C[k];
            if (networkNodeBad(X, maxValue)) {
                statusOut[node] = NET_DIVERGED;
                shStop = 1;             // гонка безобидна: все пишут одно и то же
            }
        }
        __syncthreads();                // все дочитали sh, можно перезаписывать

        if (live)
            for (int k = 0; k < AMOUNTOFX; ++k)
                sh[(size_t)node * AMOUNTOFX + k] = X[k];
        __syncthreads();

        const long long done = stepBase + s + 1;
        if (!shStop && done >= skipSteps && (done - skipSteps) % preScaller == 0) {
            const long long p = (done - skipSteps) / preScaller;
            if (p < (long long)nPoints && live) {
                numb* dst = out + ((size_t)p * nNodes + node) * AMOUNTOFX;
                for (int k = 0; k < AMOUNTOFX; ++k) dst[k] = X[k];
            }
        }

        if (threadIdx.x == 0) {
            if (progressCounter && progressStride > 0 && (done % progressStride) == 0)
                atomicAdd(progressCounter, 1);
            if (cancelFlag && *cancelFlag) shStop = 1;
        }
        __syncthreads();
    }

    if (live)
        for (int k = 0; k < AMOUNTOFX; ++k)
            state[(size_t)node * AMOUNTOFX + k] = sh[(size_t)node * AMOUNTOFX + k];
}
