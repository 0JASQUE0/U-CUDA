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
//   REF_ENABLED      — 1, если нужна четвёртая, эталонная копия траектории
//   KRS_REF_BODY     — тело шага эталонного метода (пусто при REF_ENABLED 0)

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

// Эталонный метод (по умолчанию DOPRI78) — отдельная функция, а не параметр:
// тело подставляется на компиляции, и когда эталон не нужен, всей ветки в ядре
// нет вовсе. Это не косметика — DOPRI78 держит 13 стадий в локальных массивах,
// и присутствие ветки в коде стоило бы регистров ВСЕМ расчётам вкладки, а не
// только тем, что просили эталон.
#define ORDER_REF_ENABLED {{REF_ENABLED}}

#if ORDER_REF_ENABLED
__device__ __host__ __forceinline__
void calculateReferenceModel(numb* X, const numb* a, const numb h) {
{{KRS_REF_BODY}}
}
#endif

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

// ---------------------------------------------------------------------------
// Замер времени счёта (вкладка графика Performance).
//
// Ядро НЕ считает ни ошибок, ни порядка: его время работы И ЕСТЬ измеряемая
// величина — стоимость интегрирования одной траектории шагом h на nSteps
// шагов. Ошибки для оси X приходят из отдельного, НЕ засекаемого прогона
// orderEstimateKernel по той же сетке.
//
// Здесь намеренно нет ни cancelFlag, ни progressCounter: чтение
// глобального флага раз в CHECK_INTERVAL шагов и atomicAdd прогресса — это
// работа, которой в замеряемом интервале быть не должно. Отмена ловится
// хостом МЕЖДУ запусками.
//
// nThreads одинаковых реплик считают одну и ту же задачу: при 1 меряется
// латентность одного расчёта, при большом числе — пропускная способность
// загруженного GPU. Финальное состояние обязано уходить в глобальную память,
// иначе компилятор вправе выбросить весь цикл целиком.
extern "C" __global__ void perfIntegrateKernel(
    const int    nThreads,
    const numb* __restrict__ X0,          // [AMOUNTOFX]
    const numb* __restrict__ values,      // [AMOUNTOFVALUES]
    const numb   h,
    const long long nSteps,
    numb* __restrict__ out)               // [nThreads * AMOUNTOFX]
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nThreads) return;

    numb a[AMOUNTOFVALUES];
    for (int i = 0; i < AMOUNTOFVALUES; ++i) a[i] = values[i];
    numb X[AMOUNTOFX];
    for (int i = 0; i < AMOUNTOFX; ++i) X[i] = X0[i];

    for (long long n = 0; n < nSteps; ++n)
        calculateDiscreteModel(X, a, h);

    for (int i = 0; i < AMOUNTOFX; ++i)
        out[(size_t)tid * (size_t)AMOUNTOFX + (size_t)i] = X[i];
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
    const int    refSubsteps,             // шагов эталона на один грубый шаг (0 = эталона нет)
    numb* __restrict__ outP,
    numb* __restrict__ outE1,
    numb* __restrict__ outE2,
    numb* __restrict__ outERef,           // max|y_h - y_ref|; NaN, когда эталона нет
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

#if ORDER_REF_ENABLED
    // Эталон идёт в локстепе с ГРУБОЙ копией: refM его шагов размера h1/refM на
    // каждый шаг h1. Сравнение поэтому всегда в один и тот же момент времени, и
    // хранить траекторию не нужно. refM > 1 обязателен, когда эталонный метод
    // совпадает с испытуемым: при refM == 1 это была бы та же арифметика и
    // разность тождественно нулевая.
    const int  refM = (refSubsteps > 0) ? refSubsteps : 1;
    const numb hRef = h1 / (numb)refM;
    numb Xr[AMOUNTOFX];
    for (int i = 0; i < AMOUNTOFX; ++i) Xr[i] = X0[i];
    numb eRef = (numb)0;
#endif

    size_t sinceReport = 0;
    for (long long n = 0; n < N; ++n) {
        calculateDiscreteModel(Xc, a, h1);
        calculateDiscreteModel(Xm, a, h2);
        calculateDiscreteModel(Xm, a, h2);
        calculateDiscreteModel(Xf, a, h4);
        calculateDiscreteModel(Xf, a, h4);
        calculateDiscreteModel(Xf, a, h4);
        calculateDiscreteModel(Xf, a, h4);
#if ORDER_REF_ENABLED
        for (int q = 0; q < refM; ++q) calculateReferenceModel(Xr, a, hRef);
#endif

        const bool last = (n == N - 1);
        if (!endpointOnly || last) {
            for (int i = 0; i < AMOUNTOFX; ++i) {
                const numb d1 = fabs(Xc[i] - Xm[i]);
                const numb d2 = fabs(Xm[i] - Xf[i]);
                if (d1 > e1) e1 = d1;
                if (d2 > e2) e2 = d2;
                const numb sc = fabs(Xf[i]);
                if (sc > scale) scale = sc;
#if ORDER_REF_ENABLED
                const numb dr = fabs(Xc[i] - Xr[i]);
                if (dr > eRef) eRef = dr;
#endif
            }
        }

        if ((n % (long long)CHECK_INTERVAL) == 0 || last) {
            if (orderBadVec(Xc, maxValue) || orderBadVec(Xm, maxValue) || orderBadVec(Xf, maxValue)
#if ORDER_REF_ENABLED
                || orderBadVec(Xr, maxValue)
#endif
               ) {
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
    if (outERef != nullptr) {
#if ORDER_REF_ENABLED
        outERef[gcell] = (status == ORDER_DIVERGED) ? (numb)nan("") : eRef;
#else
        outERef[gcell] = (numb)nan("");
#endif
    }
    outH[gcell]      = hEff;
    outStatus[gcell] = status;
}

// ---------------------------------------------------------------------------
// Область устойчивости на двумерной задаче Дальквиста.
//
// Тест: X' = A X, A = [[a, b], [c, d]], где элементы НЕ вводятся, а строятся в
// каждом узле сетки из (sigma, omega) и двух настроек — коэффициента
// несимметричности k и отношения недиагоналей r:
//   d = 2*sigma/(1+k),  c = -sqrt(-(1/r)*(sigma^2*(k-1)^2/(k+1)^2 + omega^2)),
//   b = r*c,            a = k*d.
// При любом k это даёт tr A = 2*sigma и det A = sigma^2 + omega^2, то есть
// собственные числа ровно sigma +- i*omega: k гоняет матрицу по семейству с
// ОДНИМ И ТЕМ ЖЕ спектром, а r только масштабирует недиагонали. Поэтому
// картинка в осях (sigma, omega) сравнима между разными k, и видно ровно то,
// ради чего k заведён, — насколько схема чувствительна к несимметричности
// задачи при неизменных собственных числах.
//
// Шаг делается ОДИН. Тест линеен и однороден (f(0) = 0), поэтому шаг любой
// схемы — линейный оператор, и прогон от двух базисных векторов (1,0) и (0,1)
// даёт столбцы матрицы усиления R целиком. Дальше rho = max|lambda(R)| через
// след и определитель. Больше одного шага смысла не имеет: R_N = R^N, спектр
// возводится в ту же степень, и корень N-й степени вернул бы то же число —
// только с потерянными разрядами.
//
// Схема сюда не «вписывается» отдельно: гоняется ТО ЖЕ тело КРС, что и на
// остальных вкладках, поэтому s = a[0], покомпонентный Гаусс-Зейдель неявных
// полушагов, композиции и экстраполяции учитываются сами собой. Отсюда же
// требование к кастомной КРС для этого режима — обращаться к a[1..4] как к
// a, b, c, d.
#define STAB_OK   0
#define STAB_BAD  1   // матрица не построилась (k = -1, r = 0, r > 0) или R улетела

extern "C" __global__ void stabilityRegionKernel(
    const int    nPtsX,
    const int    nPtsY,
    const int    nCells,                  // ячеек в этом чанке
    const int    cellOffset,              // сколько ячеек посчитано до чанка
    const numb* __restrict__ sigVals,     // [nPtsX] — узлы по sigma
    const numb* __restrict__ omVals,      // [nPtsY] — узлы по omega
    const numb* __restrict__ values,      // [AMOUNTOFVALUES] базовые a[]
    const int    idxA, const int idxB, const int idxC, const int idxD,  // куда в a[] класть элементы
    const numb   k,                       // коэффициент несимметричности
    const numb   r,                       // отношение недиагоналей, обязан быть < 0
    const numb   h,                       // шаг, чей оператор перехода и диагонализуется
    numb* __restrict__ outRho,
    int*  __restrict__ outStatus)
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= nCells) return;

    const int gcell = cell + cellOffset;
    const int ix = gcell % nPtsX;
    const int iy = gcell / nPtsX;

    const numb sig = sigVals[ix];
    const numb om  = omVals[iy];

    const numb kp1 = k + (numb)1;
    const numb km1 = k - (numb)1;

    // Подкоренное выражение отрицательно при r > 0: c тогда мнимое, и теста
    // в вещественной арифметике не существует. Это не «плохая ячейка», а
    // неверная настройка, но ловится она здесь — хост про sigma и omega узла
    // не знает.
    numb rad = (numb)0;
    bool bad = (kp1 == (numb)0) || (r == (numb)0);
    if (!bad) {
        rad = -((numb)1 / r) * (sig * sig * km1 * km1 / (kp1 * kp1) + om * om);
        if (rad < (numb)0 || isnan(rad)) bad = true;
    }
    if (bad) {
        outRho[gcell] = (numb)nan("");
        outStatus[gcell] = STAB_BAD;
        return;
    }

    const numb dEl = (numb)2 * sig / kp1;
    const numb cEl = -sqrt(rad);
    const numb bEl = r * cEl;
    const numb aEl = k * dEl;

    numb a[AMOUNTOFVALUES];
    for (int i = 0; i < AMOUNTOFVALUES; ++i) a[i] = values[i];
    a[idxA] = aEl; a[idxB] = bEl; a[idxC] = cEl; a[idxD] = dEl;

    // Столбцы матрицы усиления: шаг от (1,0) и от (0,1).
    numb R[4];   // R[0]=R00 R[1]=R01 R[2]=R10 R[3]=R11
    for (int col = 0; col < 2; ++col) {
        numb X[AMOUNTOFX];
        for (int i = 0; i < AMOUNTOFX; ++i) X[i] = (numb)0;
        X[col] = (numb)1;
        calculateDiscreteModel(X, a, h);
        R[col]     = X[0];
        R[2 + col] = X[1];
    }

    const numb tr  = R[0] + R[3];
    const numb det = R[0] * R[3] - R[1] * R[2];
    const numb disc = tr * tr * (numb)0.25 - det;

    numb rho;
    if (disc >= (numb)0) {
        const numb sq = sqrt(disc);
        const numb l1 = fabs(tr * (numb)0.5 + sq);
        const numb l2 = fabs(tr * (numb)0.5 - sq);
        rho = (l1 > l2) ? l1 : l2;
    } else {
        // Комплексно-сопряжённая пара: |lambda|^2 = tr^2/4 + (det - tr^2/4) = det,
        // и при disc < 0 определитель заведомо положителен.
        rho = sqrt(det);
    }

    if (isnan(rho) || isinf(rho)) {
        outRho[gcell] = (numb)nan("");
        outStatus[gcell] = STAB_BAD;
    } else {
        outRho[gcell] = rho;
        outStatus[gcell] = STAB_OK;
    }
}
