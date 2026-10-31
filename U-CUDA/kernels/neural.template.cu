// neural.template.cu
// NVRTC-шаблон большой сети (будущая вкладка Neural): одна связанная сеть на много блоков.
//
// Состояние в глобальной памяти SoA: s[k * N + i]. Блоки согласуются границей запуска: один запуск
// ядра = один обмен с соседями, поэтому внутри запуска только чтение старого и запись нового, порядок
// потоков на результат не влияет. Прототип 2.1: cooperative не быстрее и упирается в число потоков.
//
// Расписание шага (NEURAL_SCHED) выбирается модулем, не ветвлением:
//   0 old:   X = Phi(X) + h C(X), семантика Network, только для сверки со старым ядром
//   1 post:  X += h C(X), затем Phi: один обмен, порядок 1, порог смещён на -h K^2
//   2 heun:  то же с подшагом Хойна для связи: два обмена, порядок 1, порог почти точный
//   3 stage: RK4 со связью в каждой стадии: четыре обмена, порядок 4
//
// Плейсхолдеры: AMOUNT_OF_X, AMOUNT_OF_VALUES, NEURAL_SCHED, NEURAL_LAWS, KRS_BODY (шаг узла),
// KRS_RHS_BODY (F = f(X), нужен stage), COUPLING_BODY (case-ветки из codegen_coupling).

#define AMOUNTOFX {{AMOUNT_OF_X}}
#define AMOUNTOFVALUES {{AMOUNT_OF_VALUES}}
#define NEURAL_SCHED {{NEURAL_SCHED}}
#define NEURAL_LAWS {{NEURAL_LAWS}}

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
void calculateDiscreteModel(numb* X, UCUDA_TP(const numb ucuda_t) const numb* a, const numb h) {
{{KRS_BODY}}
}

__device__ __forceinline__ void neuralRhs(const numb* X, const numb* a, numb* F) {
{{KRS_RHS_BODY}}
}

#include "cudaLibrary.cu"

// Раскладка совпадает с NeuralStepArgs / NeuralObserveArgs / NeuralMeanArgs в parametric_engine.cpp.
struct NeuralStepArgs {
    const numb* values;
    numb* s0;
    numb* s1;
    numb* s2;
    numb* s3;
    const int*  edgeStart;
    const int*  edgeSrc;
    const numb* edgeW;
    const int*  edgeLaw;
    const int*  inChannel;    // nullptr = без внешнего входа
    const numb* inGain;
    const numb* inSample;     // текущая строка отсчётов, удержание на весь шаг
    int*  badFlag;
    int*  status;
    numb  h;
    numb  maxValue;
    int   nNodes;
    int   phase;
    int   inVar;
    int   pad;
};

// Сосед читается только по тем переменным, что есть в законе связи: Xj[k] как у codegen_coupling.
struct NeuralNbr {
    const numb* s;
    int N, src;
    __device__ __forceinline__ numb operator[](int k) const { return __ldg(s + (size_t)k * N + src); }
};

__device__ __forceinline__ void neuralCoupling(const NeuralStepArgs& A, const numb* s, const numb* Xi,
                                               const numb* ai, const int node, numb* C) {
    for (int k = 0; k < AMOUNTOFX; ++k) C[k] = (numb)0;
    const int e1 = A.edgeStart[node + 1];
    for (int e = A.edgeStart[node]; e < e1; ++e) {
        const int src = __ldg(A.edgeSrc + e);
        const NeuralNbr Xj{ s, A.nNodes, src };
        const numb* aj = A.values + (size_t)src * AMOUNTOFVALUES;
        const numb w = __ldg(A.edgeW + e);
        (void)aj;
        // При одном законе массив законов не читается вовсе.
        switch (NEURAL_LAWS > 1 ? __ldg(A.edgeLaw + e) : 0) {
{{COUPLING_BODY}}
        default: break;
        }
    }
    if (A.inChannel != nullptr) {
        const int ch = A.inChannel[node];
        if (ch >= 0) C[A.inVar] += A.inGain[node] * A.inSample[ch];
    }
}

__device__ __forceinline__ void neuralRead(const numb* s, const int N, const int node, numb* X) {
    for (int k = 0; k < AMOUNTOFX; ++k) X[k] = s[(size_t)k * N + node];
}
__device__ __forceinline__ void neuralWrite(numb* s, const int N, const int node, const numb* X) {
    for (int k = 0; k < AMOUNTOFX; ++k) s[(size_t)k * N + node] = X[k];
}
__device__ __forceinline__ void neuralCheck(const NeuralStepArgs& A, const int node, const numb* X) {
    numb acc = (numb)0;
    for (int k = 0; k < AMOUNTOFX; ++k) acc += fabs(X[k]);
    if (isnan(acc) || isinf(acc) || (A.maxValue != (numb)0 && acc > A.maxValue)) {
        A.status[node] = 1;
        *A.badFlag = 1;     // гонка безобидна: все пишут одно и то же
    }
}

extern "C" __global__ void neuralStepKernel(const NeuralStepArgs A) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= A.nNodes) return;
    const int N = A.nNodes;
    const numb h = A.h;
    const numb* ai = A.values + (size_t)node * AMOUNTOFVALUES;
    numb X[AMOUNTOFX], C[AMOUNTOFX];

#if NEURAL_SCHED == 3
    const int j = A.phase;
    const numb* in = j == 0 ? A.s0 : (j == 2 ? A.s3 : A.s2);
    numb* out = j == 1 ? A.s3 : A.s2;
    numb* b = A.s0;
    numb* acc = A.s1;
    numb F[AMOUNTOFX];
    neuralRead(in, N, node, X);
    neuralRhs(X, ai, F);
    neuralCoupling(A, in, X, ai, node, C);
    for (int k = 0; k < AMOUNTOFX; ++k) F[k] += C[k];
    if (j == 3) {
        const numb q = h / (numb)6;
        for (int k = 0; k < AMOUNTOFX; ++k) {
            const size_t at = (size_t)k * N + node;
            X[k] = b[at] + q * (acc[at] + F[k]);
            b[at] = X[k];
        }
        neuralCheck(A, node, X);
    } else {
        const numb c = j == 2 ? h : (numb)0.5 * h;
        for (int k = 0; k < AMOUNTOFX; ++k) {
            const size_t at = (size_t)k * N + node;
            if (j == 0) acc[at] = F[k]; else acc[at] += (numb)2 * F[k];
            out[at] = b[at] + c * F[k];
        }
    }
#else
    UcudaKrsPre krsPre;
    ucudaKrsPrep(krsPre, ai, h);
    neuralRead(A.s0, N, node, X);
#if NEURAL_SCHED == 0
    neuralCoupling(A, A.s0, X, ai, node, C);
    calculateDiscreteModelP(X, ai, h, krsPre);
    for (int k = 0; k < AMOUNTOFX; ++k) X[k] += h * C[k];
#elif NEURAL_SCHED == 1
    neuralCoupling(A, A.s0, X, ai, node, C);
    for (int k = 0; k < AMOUNTOFX; ++k) X[k] += h * C[k];
    calculateDiscreteModelP(X, ai, h, krsPre);
#else
    if (A.phase == 0) {
        neuralCoupling(A, A.s0, X, ai, node, C);
        for (int k = 0; k < AMOUNTOFX; ++k) X[k] += h * C[k];
        neuralWrite(A.s2, N, node, X);
        return;
    }
    numb Y[AMOUNTOFX];
    neuralRead(A.s2, N, node, Y);
    neuralCoupling(A, A.s2, Y, ai, node, C);
    for (int k = 0; k < AMOUNTOFX; ++k) X[k] = (numb)0.5 * (X[k] + Y[k]) + (numb)0.5 * h * C[k];
    calculateDiscreteModelP(X, ai, h, krsPre);
#endif
    neuralWrite(A.s1, N, node, X);
    neuralCheck(A, node, X);
#endif
}

struct NeuralObserveArgs {
    const numb* state;
    const int*  nodes;
    const int*  vars;
    numb* out;
    int nNodes, nSel, nVars, pad;
};

// Выбранные узлы: out[sel * nVars + v].
extern "C" __global__ void neuralObserveKernel(const NeuralObserveArgs A) {
    const int sel = blockIdx.x * blockDim.x + threadIdx.x;
    if (sel >= A.nSel) return;
    const int node = A.nodes[sel];
    for (int v = 0; v < A.nVars; ++v)
        A.out[(size_t)sel * A.nVars + v] = A.state[(size_t)A.vars[v] * A.nNodes + node];
}

struct NeuralMeanArgs {
    const numb* state;
    const int*  groupStart;   // [nGroups + 1], группы подряд идущих узлов одной популяции
    const int*  groupPop;
    const int*  vars;
    numb* out;
    int nNodes, nGroups, nVars, pad;
};

// Сумма по популяции: out[pop * nVars + v]. Сумма группы локальна, atomicAdd один на группу, а не на
// узел: на узел все потоки стояли бы на одном адресе.
extern "C" __global__ void neuralMeanKernel(const NeuralMeanArgs A) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= A.nGroups) return;
    const int i0 = A.groupStart[g], i1 = A.groupStart[g + 1], pop = A.groupPop[g];
    for (int v = 0; v < A.nVars; ++v) {
        const numb* col = A.state + (size_t)A.vars[v] * A.nNodes;
        numb sum = (numb)0;
        for (int i = i0; i < i1; ++i) sum += col[i];
        atomicAdd(A.out + (size_t)pop * A.nVars + v, sum);
    }
}
