// order.template.cu
// NVRTC-шаблон для вкладки Order — оценка порядка точности по Эйткену/Ричардсону.
//
// В каждой точке сетки поток интегрирует ОДНУ И ТУ ЖЕ задачу тремя шагами
// h, h/2, h/4 в локстепе и копит два бегущих максимума разностей НА ГРУБОЙ
// сетке (в моменты, кратные h — прореживание 2 и 4):
//   E1 = max |y_h   - y_h/2|
//   E2 = max |y_h/2 - y_h/4|
//   p  = log2(E1 / E2)
// При y_h = y* + C h^p это даёт ровно p: E1 = C h^p (1-2^-p),
// E2 = C h^p 2^-p (1-2^-p), отношение = 2^p. Опорное решение не нужно —
// вариант «обе разности против h/4» даёт log2(2^p + 1), то есть +0.58 на
// p=1 и +0.32 на p=2.
//
// Траектории НЕ хранятся: на поток три вектора состояния в регистрах/локали
// и два скаляра, в глобальную память уходит 3 числа + статус на ячейку.
// Поэтому карта 512x512 стоит столько же памяти, сколько 1D-свип.
//
// Плейсхолдеры:
//   AMOUNT_OF_X      — размерность системы
//   AMOUNT_OF_VALUES — размер a[] (a[0] = symmetry s, a[1..M] = параметры)
//   KRS_BODY         — тело calculateDiscreteModel из codegen

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

// ---------------------------------------------------------------------------
// Коды статуса ячейки. Значение p пишется ВСЕГДА, кроме ORDER_DIVERGED: полка
// округления и потеря устойчивости — это осмысленно посчитанные числа, и
// именно их провалы пользователь и пришёл смотреть. NaN означает «не
// посчитано», а не «посчитано плохо».
#define ORDER_OK         0
#define ORDER_DIVERGED   1   // nan/inf или |X| > maxValue хоть в одной из трёх копий
#define ORDER_FLOOR      2   // E2 утонуло в машинной точности решения
#define ORDER_NOCONTRACT 3   // E2 >= E1, то есть p <= 0

// Ось свипа: что именно перебирается по этой координате сетки.
#define ORDER_AXIS_NONE 0
#define ORDER_AXIS_H    1
#define ORDER_AXIS_VAL  2    // элемент a[], индекс в axis*Index

__device__ __forceinline__ bool orderBadVec(const numb* X, numb maxValue) {
    numb acc = (numb)0;
    for (int i = 0; i < AMOUNTOFX; ++i) acc += fabs(X[i]);
    if (isnan(acc) || isinf(acc)) return true;
    if (maxValue != (numb)0 && acc > maxValue) return true;
    return false;
}

extern "C" __global__ void orderEstimateKernel(
    const int    nPtsX,
    const int    nPtsY,                   // 1 для 1D-свипа
    const int    nCells,                  // ячеек в этом чанке
    const int    cellOffset,              // сколько ячеек посчитано до чанка
    const numb* __restrict__ axisXVals,   // [nPtsX] — узлы уже в нужном масштабе (lin/log сделал хост)
    const numb* __restrict__ axisYVals,   // [nPtsY]
    const int    axisXKind, const int axisXIndex,
    const int    axisYKind, const int axisYIndex,
    const numb* __restrict__ X0,          // [AMOUNTOFX] начальные условия
    const numb* __restrict__ values,      // [AMOUNTOFVALUES] базовые a[]
    const numb   hBase,                   // h, когда ни одна ось не свипует h
    const numb   tMax,
    const int    snapSteps,               // 1 = h подогнать к tMax/N при целом N
    const int    endpointOnly,            // 1 = сравнивать только в t = tMax
    const numb   maxValue,
    const numb   floorEps,                // относительный порог полки округления
    numb* __restrict__ outP,
    numb* __restrict__ outE1,
    numb* __restrict__ outE2,
    numb* __restrict__ outH,              // фактический h ячейки (после snap)
    int*  __restrict__ outStatus,
    const volatile int* cancelFlag,
    int* progressCounter,
    const int progressStride)
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= nCells) return;

    const int gcell = cell + cellOffset;
    const int ix = gcell % nPtsX;
    const int iy = (nPtsY > 1) ? (gcell / nPtsX) : 0;

    numb a[AMOUNTOFVALUES];
    for (int i = 0; i < AMOUNTOFVALUES; ++i) a[i] = values[i];

    numb h = hBase;

    const numb vx = axisXVals[ix];
    if      (axisXKind == ORDER_AXIS_H)   h = vx;
    else if (axisXKind == ORDER_AXIS_VAL) { if (axisXIndex >= 0 && axisXIndex < AMOUNTOFVALUES) a[axisXIndex] = vx; }

    if (nPtsY > 1) {
        const numb vy = axisYVals[iy];
        if      (axisYKind == ORDER_AXIS_H)   h = vy;
        else if (axisYKind == ORDER_AXIS_VAL) { if (axisYIndex >= 0 && axisYIndex < AMOUNTOFVALUES) a[axisYIndex] = vy; }
    }

    if (!(h > (numb)0) || isnan(h) || isinf(h)) {
        outP[gcell] = (numb)nan(""); outE1[gcell] = (numb)nan(""); outE2[gcell] = (numb)nan("");
        outH[gcell] = h; outStatus[gcell] = ORDER_DIVERGED;
        return;
    }

    // Число ШАГОВ ГРУБОЙ копии. Без snap берём floor, как остальные вкладки:
    // тогда фактическое конечное время = N*h != tMax. У всех трёх копий оно
    // при этом одинаковое (каждая идёт ровно N грубых шагов), так что само
    // сравнение остаётся корректным — «плавает» лишь момент, в который оно
    // сделано, а вместе с ним и точка кривой ошибки.
    long long N;
    numb hEff = h;
    if (snapSteps) {
        N = (long long)(tMax / h + (numb)0.5);
        if (N < 1) N = 1;
        hEff = tMax / (numb)N;
    } else {
        N = (long long)(tMax / h);
        if (N < 1) N = 1;
    }

    const numb h1 = hEff;
    const numb h2 = hEff / (numb)2;
    const numb h4 = hEff / (numb)4;

    numb Xc[AMOUNTOFX], Xm[AMOUNTOFX], Xf[AMOUNTOFX];
    for (int i = 0; i < AMOUNTOFX; ++i) { Xc[i] = X0[i]; Xm[i] = X0[i]; Xf[i] = X0[i]; }

    numb e1 = (numb)0, e2 = (numb)0, scale = (numb)0;
    int  status = ORDER_OK;

    size_t sinceReport = 0;
    for (long long n = 0; n < N; ++n) {
        calculateDiscreteModel(Xc, a, h1);
        calculateDiscreteModel(Xm, a, h2);
        calculateDiscreteModel(Xm, a, h2);
        calculateDiscreteModel(Xf, a, h4);
        calculateDiscreteModel(Xf, a, h4);
        calculateDiscreteModel(Xf, a, h4);
        calculateDiscreteModel(Xf, a, h4);

        const bool last = (n == N - 1);
        if (!endpointOnly || last) {
            for (int i = 0; i < AMOUNTOFX; ++i) {
                const numb d1 = fabs(Xc[i] - Xm[i]);
                const numb d2 = fabs(Xm[i] - Xf[i]);
                if (d1 > e1) e1 = d1;
                if (d2 > e2) e2 = d2;
                const numb sc = fabs(Xf[i]);
                if (sc > scale) scale = sc;
            }
        }

        if ((n % (long long)CHECK_INTERVAL) == 0 || last) {
            if (orderBadVec(Xc, maxValue) || orderBadVec(Xm, maxValue) || orderBadVec(Xf, maxValue)) {
                status = ORDER_DIVERGED;
                break;
            }
            if (cancelFlag != nullptr && *cancelFlag != 0) break;
        }

        if (progressCounter != nullptr && progressStride > 0) {
            if (++sinceReport >= (size_t)progressStride) {
                sinceReport = 0;
                atomicAdd(progressCounter, 1);
            }
        }
    }

    numb p;
    if (status == ORDER_DIVERGED) {
        p = (numb)nan(""); e1 = (numb)nan(""); e2 = (numb)nan("");
    } else {
        // Полка округления: разности перестали убывать не потому, что метод
        // плох, а потому что вычитание почти одинаковых чисел даёт только шум.
        // Порог растёт как sqrt(числа шагов): ошибка округления накапливается
        // случайным блужданием по шагам, а не остаётся в пределах нескольких
        // ulp от решения. Фиксированный порог в 8 ulp пропускал полку у RK4 —
        // там при t_max/h ~ 1e4 шум доходит до 1e-14 при решении порядка 1,
        // и p уже мусор, хотя разность формально «больше нескольких ulp».
        const numb nsteps = (numb)4 * (numb)N;   // мелкая копия делает 4N шагов
        const numb tiny = floorEps * (scale > (numb)0 ? scale : (numb)1) * sqrt(nsteps);
        if (e2 <= tiny || e1 <= tiny) status = ORDER_FLOOR;
        else if (e2 >= e1)            status = ORDER_NOCONTRACT;

        if (e2 > (numb)0 && e1 > (numb)0) p = log2(e1 / e2);
        else                              p = (numb)nan("");
    }

    outP[gcell]      = p;
    outE1[gcell]     = e1;
    outE2[gcell]     = e2;
    outH[gcell]      = hEff;
    outStatus[gcell] = status;
}
