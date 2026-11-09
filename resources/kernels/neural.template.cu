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
//   4 spike: импульсная сеть, свой шаг neuralSpikeKernel: связь только спайками с задержкой
//
// NEURAL_INST 1: несвязные экземпляры. status[node] = шаг разлёта, узел с ненулевым status замирает, хост
// замораживает весь экземпляр на границе чанка; при 0 текст ядра шага прежний.
//
// NEURAL_PLASTIC 1: пластичные дуги импульсной сети; при 0 текст ядра прежний.
// NEURAL_DEV > 0: устройства на связях у stage, столько состояний на устройство, NEURAL_DEV_P параметров; при 0 текст прежний.
//
// Плейсхолдеры: AMOUNT_OF_X, AMOUNT_OF_VALUES, NEURAL_SCHED, NEURAL_LAWS, NEURAL_INST, NEURAL_PLASTIC, KRS_BODY (шаг узла),
// KRS_RHS_BODY (F = f(X), нужен stage), COUPLING_BODY (case-ветки из codegen_coupling).

#define AMOUNTOFX {{AMOUNT_OF_X}}
#define AMOUNTOFVALUES {{AMOUNT_OF_VALUES}}
#define NEURAL_SCHED {{NEURAL_SCHED}}
#define NEURAL_LAWS {{NEURAL_LAWS}}
#define NEURAL_INST {{NEURAL_INST}}
#define NEURAL_REC {{NEURAL_REC}}
#define NEURAL_PLASTIC {{NEURAL_PLASTIC}}
#define NEURAL_DEV {{NEURAL_DEV}}
#define NEURAL_DEV_P {{NEURAL_DEV_P}}

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

#if NEURAL_DEV
__device__ __forceinline__ void neuralDevRhs(const numb* X, const numb* a, numb* F) {
{{DEV_RHS_BODY}}
}
__device__ __forceinline__ void neuralDevCurrent(const numb* X, const numb* a, numb* F) {
{{DEV_CURRENT_BODY}}
}
#endif

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
    const numb* inSample;     // строка отсчётов этого запуска: на шаг или на стадию, выбирает хост
    int*  badFlag;
    int*  status;
    numb  h;
    numb  maxValue;
    int   nNodes;
    int   phase;
    int   inVar;
    int   step;       // номер шага с 1
    int   nRun;       // считаются узлы ниже него: клоны Ляпунова до посева стоят
    int   pad;
    // Устройства, только stage: d0..d3 повторяют s0..s3, [состояние][устройство].
    const int*  devStart;     // [nNodes + 1]
    const int*  devList;      // устройство * 2 + 1 у узла b: он же владелец записи; * 2 у узла a
    const int*  devA;
    const int*  devB;
    const int*  devVar;
    const numb* devGain;
    const numb* devValues;
    numb* d0;
    numb* d1;
    numb* d2;
    numb* d3;
    int*  devClips;
    numb  devMin[4];
    numb  devMax[4];
    int   nDev;
    int   pad2;
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
#if NEURAL_INST
        A.status[node] = A.step;
#else
        A.status[node] = 1;
#endif
        *A.badFlag = 1;     // гонка безобидна: все пишут одно и то же
    }
}

#if NEURAL_SCHED != 4
extern "C" __global__ void neuralStepKernel(const NeuralStepArgs A) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= A.nRun) return;
    const int N = A.nNodes;
    const numb h = A.h;
    const numb* ai = A.values + (size_t)node * AMOUNTOFVALUES;
    numb X[AMOUNTOFX], C[AMOUNTOFX];

#if NEURAL_INST
    if (A.status[node] != 0) {
#if NEURAL_SCHED != 3
        // Буферы меняются местами: состояние переносится в тот, что станет текущим.
        if (A.phase == (NEURAL_SCHED == 2 ? 1 : 0)) {
            neuralRead(A.s0, N, node, X);
            neuralWrite(A.s1, N, node, X);
        }
#endif
        return;
    }
#endif

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
#if NEURAL_DEV
    {
        const numb* dIn = j == 0 ? A.d0 : (j == 2 ? A.d3 : A.d2);
        numb* dOut = j == 1 ? A.d3 : A.d2;
        const int e1 = A.devStart[node + 1], D = A.nDev;
        for (int e = A.devStart[node]; e < e1; ++e) {
            const int pack = __ldg(A.devList + e), dev = pack >> 1, own = pack & 1, va = __ldg(A.devVar + dev);
            numb s[NEURAL_DEV], q[2 + NEURAL_DEV_P], Fd[NEURAL_DEV + 1];
            for (int k = 0; k < NEURAL_DEV; ++k) s[k] = dIn[(size_t)k * D + dev];
            q[0] = (numb)0;
            q[1] = in[(size_t)va * N + __ldg(A.devA + dev)] - in[(size_t)va * N + __ldg(A.devB + dev)];
            for (int k = 0; k < NEURAL_DEV_P; ++k) q[2 + k] = __ldg(A.devValues + (size_t)dev * NEURAL_DEV_P + k);
            neuralDevCurrent(s, q, Fd);
            const numb I = __ldg(A.devGain + dev) * Fd[0];
            C[va] += own ? I : -I;
            if (!own) continue;
            neuralDevRhs(s, q, Fd);
            for (int k = 0; k < NEURAL_DEV; ++k) {
                const size_t at = (size_t)k * D + dev;
                if (j == 3) {
                    numb x = A.d0[at] + (h / (numb)6) * (A.d1[at] + Fd[k]);
                    if (x < A.devMin[k] || x > A.devMax[k]) {
                        x = fmin(fmax(x, A.devMin[k]), A.devMax[k]);
                        atomicAdd(A.devClips, 1);
                    }
                    A.d0[at] = x;
                } else {
                    if (j == 0) A.d1[at] = Fd[k]; else A.d1[at] += (numb)2 * Fd[k];
                    dOut[at] = A.d0[at] + (j == 2 ? h : (numb)0.5 * h) * Fd[k];
                }
            }
        }
    }
#endif
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

#endif

#if NEURAL_SCHED == 4
// Импульсный шаг. Порядок фаз одного шага: доставка, шаг, порог и сброс, регистрация и рассылка спайка.
// Шаг узла локален, соседи видны только через кольцо. Спайк пишет в слот шага n + задержка >= n + window, а узел
// читает слот своего шага, поэтому запуск на window шагов не требует обмена. Веса в кольце целые: сумма
// atomicAdd не зависит от порядка потоков, и результат повторяется побитово.
// Переменные X: сначала NEURAL_NV переменных узла, затем NEURAL_REC синапсов; синапс идёт по точной экспоненте.
#define NEURAL_NV (AMOUNTOFX - NEURAL_REC)
#define NEURAL_REC_MAX 8
#define NEURAL_CH (NEURAL_REC + NEURAL_PLASTIC)

struct NeuralStdpRule {
    numb aPlus, aMinus;
    numb hPlus, hMinus;       // h / tau
    numb wMin, wMax;
    numb muPlus, muMinus;
    numb gain;
    int  nearest;
    int  pad;
};

struct NeuralSpikeArgs {
    const numb* values;
    numb* state;
    const int*  outStart;     // [nNodes + 1], исходящие дуги
    const int*  outDst;
    const int*  outPack;      // задержка * NEURAL_REC + синапс
    const unsigned long long* outW;
    unsigned long long* ring; // [slots][NEURAL_CH][nNodes], последний канал: приходы по пластичным дугам
    const int*  refrSteps;
    int*  refr;               // сколько шагов ещё держать узел
    int*  spikes;             // счётчик узла: точен и при переполнении журнала
    int*  logCount;
    int*  logNode;
    int*  logStep;            // от logBase
    const numb* srcP;         // вероятность за шаг, < 0 = обычный узел; nullptr = источников нет
    const int*  srcFrom;
    const int*  srcTo;
    const unsigned long long* srcSeed;
    const int*  inChannel;    // nullptr = без внешнего входа
    const numb* inGain;
    const numb* inSamples;    // [строки][подстроки][каналы], весь ряд
    const numb* invW;         // масштаб целых весов узла: свой у экземпляра, иначе значение свипа не равно отдельному запуску
    int*  status;             // шаг разлёта узла + 1, -1 = экземпляр заморожен хостом
    int*  badFlag;
    const int*  plStart;      // [nNodes + 1], пластичные входящие дуги, их пишет только поток приёмника
    const int*  plSrc;
    const int*  plPack;       // задержка * NEURAL_REC + синапс
    const int*  plRule;
    const NeuralStdpRule* rules;
    numb* plW;
    numb* plPre;
    numb* plPost;
    int*  plLast;             // индекс времени последнего обновления дуги
    int*  stamp;              // [slots][nNodes], шаг последнего спайка в этом слоте
    numb  decay[NEURAL_REC_MAX];
    numb  decayHalf[NEURAL_REC_MAX];
    numb  h;
    numb  maxValue;
    long long step0;
    long long logBase;
    long long inHold;
    long long inLast;
    int   nNodes;
    int   slots;
    int   steps;
    int   logCap;
    int   inChannels;
    int   inSub;              // 3 = значения t, t + h/2, t + h
    int   inVar;
    int   pad;
};

__device__ __forceinline__ unsigned long long neuralMix(unsigned long long z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
// Число от узла и шага: не зависит от разбиения на запуски. Сид узла считает хост от его номера в экземпляре.
__device__ __forceinline__ numb neuralUniform(unsigned long long nodeSeed, long long step) {
    return (numb)(neuralMix(nodeSeed ^ ((unsigned long long)step + 1ULL) * 0x165667B19E3779F9ULL) >> 11) * (numb)1.1102230246251565e-16;
}

__device__ __forceinline__ void neuralSpikeRule(numb* X, const numb* a, bool& spiked) {
{{SPIKE_BODY}}
}

__device__ __forceinline__ void neuralSpikeRk4(numb* X, const numb* ai, const NeuralSpikeArgs& A, const numb u0, const numb uh, const numb u1) {
    const numb h = A.h;
    numb Y[AMOUNTOFX], F[AMOUNTOFX], acc[AMOUNTOFX];
    for (int k = 0; k < AMOUNTOFX; ++k) Y[k] = X[k];
    neuralRhs(Y, ai, F);
    F[A.inVar] += u0;
    for (int k = 0; k < NEURAL_NV; ++k) { acc[k] = F[k]; Y[k] = X[k] + (numb)0.5 * h * F[k]; }
    for (int r = 0; r < NEURAL_REC; ++r) Y[NEURAL_NV + r] = X[NEURAL_NV + r] * A.decayHalf[r];
    neuralRhs(Y, ai, F);
    F[A.inVar] += uh;
    for (int k = 0; k < NEURAL_NV; ++k) { acc[k] += (numb)2 * F[k]; Y[k] = X[k] + (numb)0.5 * h * F[k]; }
    neuralRhs(Y, ai, F);
    F[A.inVar] += uh;
    for (int k = 0; k < NEURAL_NV; ++k) { acc[k] += (numb)2 * F[k]; Y[k] = X[k] + h * F[k]; }
    for (int r = 0; r < NEURAL_REC; ++r) Y[NEURAL_NV + r] = X[NEURAL_NV + r] * A.decay[r];
    neuralRhs(Y, ai, F);
    F[A.inVar] += u1;
    const numb q = h / (numb)6;
    for (int k = 0; k < NEURAL_NV; ++k) X[k] += q * (acc[k] + F[k]);
}

#if NEURAL_PLASTIC
// Следы дуги на момент at. Индекс времени: приход в начале шага n это n, спайк приёмника конца шага n это n + 1.
__device__ __forceinline__ void neuralArcTraces(const NeuralSpikeArgs& A, const NeuralStdpRule& r, const int e, const int at, numb& pre, numb& post) {
    const numb dt = (numb)(at - A.plLast[e]);
    pre = A.plPre[e] * exp(-dt * r.hPlus);
    post = A.plPost[e] * exp(-dt * r.hMinus);
    A.plLast[e] = at;
}
__device__ __forceinline__ numb neuralArcClip(const NeuralStdpRule& r, const numb w) { return fmin(fmax(w, r.wMin), r.wMax); }

__device__ __forceinline__ numb neuralArcArrive(const NeuralSpikeArgs& A, const int e, const int at) {
    const NeuralStdpRule& r = A.rules[__ldg(A.plRule + e)];
    numb pre, post;
    neuralArcTraces(A, r, e, at, pre, post);
    numb w = A.plW[e];
    const numb f = r.muMinus == (numb)0 ? (numb)1 : pow((w - r.wMin) / (r.wMax - r.wMin), r.muMinus);
    w = neuralArcClip(r, w - r.gain * r.aMinus * post * f);
    A.plW[e] = w;
    A.plPre[e] = r.nearest ? (numb)1 : pre + (numb)1;
    A.plPost[e] = post;
    return w;
}
__device__ __forceinline__ void neuralArcSpike(const NeuralSpikeArgs& A, const int e, const int at) {
    const NeuralStdpRule& r = A.rules[__ldg(A.plRule + e)];
    numb pre, post;
    neuralArcTraces(A, r, e, at, pre, post);
    const numb w = A.plW[e];
    const numb f = r.muPlus == (numb)0 ? (numb)1 : pow((r.wMax - w) / (r.wMax - r.wMin), r.muPlus);
    A.plW[e] = neuralArcClip(r, w + r.gain * r.aPlus * pre * f);
    A.plPre[e] = pre;
    A.plPost[e] = r.nearest ? (numb)1 : post + (numb)1;
}
#endif

extern "C" __global__ void neuralSpikeKernel(const NeuralSpikeArgs A) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = A.nNodes;
    if (node >= N || A.status[node] != 0) return;
    const numb* ai = A.values + (size_t)node * AMOUNTOFVALUES;
    const bool source = A.srcP != nullptr && A.srcP[node] >= (numb)0;
    const int ch = A.inChannel != nullptr ? A.inChannel[node] : -1;
    const numb gain = ch >= 0 ? A.inGain[node] : (numb)0;
    numb X[AMOUNTOFX];
    neuralRead(A.state, N, node, X);
    int refr = A.refr[node], fired = 0;
    const int period = A.slots * NEURAL_CH;

    for (int j = 0; j < A.steps; ++j) {
        const long long step = A.step0 + j;
        const int base = (int)(step % A.slots) * NEURAL_CH;
        bool spiked = false;
        if (source) {
            spiked = step >= A.srcFrom[node] && step < A.srcTo[node] && neuralUniform(A.srcSeed[node], step) < A.srcP[node];
        } else {
            for (int r = 0; r < NEURAL_REC; ++r) {
                unsigned long long* cell = A.ring + (size_t)(base + r) * N + node;
                const unsigned long long w = *cell;
                if (w != 0ULL) { X[NEURAL_NV + r] += (numb)(long long)w * A.invW[node]; *cell = 0ULL; }
            }
#if NEURAL_PLASTIC
            // Счёт приходов ограничивает проход: узел без приходов дуг не читает.
            unsigned long long* arrived = A.ring + (size_t)(base + NEURAL_REC) * N + node;
            long long left = (long long)*arrived;
            if (left != 0) {
                *arrived = 0ULL;
                const int slot = (int)(step % A.slots), e1 = A.plStart[node + 1];
                for (int e = A.plStart[node]; left > 0 && e < e1; ++e) {
                    const int pack = __ldg(A.plPack + e), d = pack / NEURAL_REC;
                    if (step < d) continue;
                    const int q = slot >= d ? slot - d : slot - d + A.slots;
                    if (A.stamp[(size_t)q * N + __ldg(A.plSrc + e)] != (int)(step - d)) continue;
                    --left;
                    X[NEURAL_NV + pack % NEURAL_REC] += neuralArcArrive(A, e, (int)step);
                }
            }
#endif
            if (refr > 0) --refr;
            else {
                numb u0 = (numb)0, uh = (numb)0, u1 = (numb)0;
                if (ch >= 0) {
                    long long row = step / A.inHold;
                    if (row > A.inLast) row = A.inLast;
                    const numb* q = A.inSamples + (size_t)row * A.inSub * A.inChannels + ch;
                    u0 = gain * q[0];
                    uh = A.inSub == 3 ? gain * q[A.inChannels] : u0;
                    u1 = A.inSub == 3 ? gain * q[2 * A.inChannels] : u0;
                }
                neuralSpikeRk4(X, ai, A, u0, uh, u1);
                numb acc = (numb)0;
                for (int k = 0; k < NEURAL_NV; ++k) acc += fabs(X[k]);
                if (isnan(acc) || isinf(acc) || (A.maxValue != (numb)0 && acc > A.maxValue)) {
                    A.status[node] = (int)step + 1;
                    *A.badFlag = 1;
                    break;
                }
                neuralSpikeRule(X, ai, spiked);
                if (spiked) {
                    refr = A.refrSteps[node];
#if NEURAL_PLASTIC
                    const int e1 = A.plStart[node + 1];
                    for (int e = A.plStart[node]; e < e1; ++e) neuralArcSpike(A, e, (int)step + 1);
#endif
                }
            }
            for (int r = 0; r < NEURAL_REC; ++r) X[NEURAL_NV + r] *= A.decay[r];
        }
        if (!spiked) continue;
        ++fired;
#if NEURAL_PLASTIC
        A.stamp[(size_t)(step % A.slots) * N + node] = (int)step;
#endif
        const int at = atomicAdd(A.logCount, 1);
        if (at < A.logCap) { A.logNode[at] = node; A.logStep[at] = (int)(step - A.logBase); }
        const int e1 = __ldg(A.outStart + node + 1);
        for (int e = __ldg(A.outStart + node); e < e1; ++e) {
            int idx = base + __ldg(A.outPack + e);
            if (idx >= period) idx -= period;
            atomicAdd(A.ring + (size_t)idx * N + __ldg(A.outDst + e), __ldg(A.outW + e));
        }
    }
    neuralWrite(A.state, N, node, X);
    A.refr[node] = refr;
    if (fired != 0) A.spikes[node] += fired;
}
#endif

struct NeuralObserveArgs {
    const numb* state;
    const int*  nodes;
    const int*  vars;
    numb* out;
    int nNodes, nSel, nVars, pad;
};

struct NeuralWeightArgs {
    const numb* w;
    const int*  arcs;
    numb* out;
    int n, pad;
};
extern "C" __global__ void neuralWeightKernel(const NeuralWeightArgs A) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < A.n) A.out[i] = A.w[A.arcs[i]];
}

// Выбранные узлы: out[sel * nVars + v].
extern "C" __global__ void neuralObserveKernel(const NeuralObserveArgs A) {
    const int sel = blockIdx.x * blockDim.x + threadIdx.x;
    if (sel >= A.nSel) return;
    const int node = A.nodes[sel];
    for (int v = 0; v < A.nVars; ++v)
        A.out[(size_t)sel * A.nVars + v] = A.state[(size_t)A.vars[v] * A.nNodes + node];
}

// Замкнутый выход: блок k считает y_k по строке записи и кладёт его в строку входа. Сумма в double при любом numb.
#define NEURAL_FEEDBACK_BLOCK 256
struct NeuralFeedbackArgs {
    const numb* row;
    const int*  cols;
    const int*  channels;
    const double* w;          // [K][M + 1]
    numb* out;                // строка входа
    int M, K;
};
extern "C" __global__ void neuralFeedbackKernel(const NeuralFeedbackArgs A) {
    __shared__ double part[NEURAL_FEEDBACK_BLOCK];
    const int k = blockIdx.x;
    if (k >= A.K) return;
    const double* w = A.w + (size_t)k * (A.M + 1);
    double s = 0.0;
    for (int i = threadIdx.x; i < A.M; i += blockDim.x) s += w[i] * (double)A.row[A.cols[i]];
    part[threadIdx.x] = s;
    __syncthreads();
    for (int half = blockDim.x / 2; half > 0; half >>= 1) {
        if ((int)threadIdx.x < half) part[threadIdx.x] += part[threadIdx.x + half];
        __syncthreads();
    }
    if (threadIdx.x == 0) A.out[A.channels[k]] = (numb)(part[0] + w[A.M]);
}

struct NeuralMeanArgs {
    const numb* state;
    const int*  groupStart;   // [nGroups + 1], группы подряд идущих узлов одной популяции
    const int*  groupPop;
    const int*  vars;
    numb* out;
    numb* out2;               // суммы квадратов, nullptr = не нужны
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
        numb sum = (numb)0, sum2 = (numb)0;
        for (int i = i0; i < i1; ++i) { const numb x = col[i]; sum += x; sum2 += x * x; }
        atomicAdd(A.out + (size_t)pop * A.nVars + v, sum);
        if (A.out2 != nullptr) atomicAdd(A.out2 + (size_t)pop * A.nVars + v, sum2);
    }
}

struct NeuralOrderArgs {
    const numb* column;       // наблюдаемая переменная всех узлов
    numb* prev;               // значение в прошлой записи
    int*  count;              // занято в списке пересечений
    int*  outNode;
    numb* outTime;
    numb  threshold;
    numb  t;
    numb  dt;
    int   nNodes;
    int   first;              // первая запись: только запомнить значение
    int   capacity;
    int   pad;
};

struct NeuralLyapArgs {
    numb* state;
    const int*  groupStart;   // [nGroups + 1], группы подряд идущих узлов одного экземпляра
    numb* dist;               // замер: сумма квадратов по группе
    const numb* scale;        // перенормировка: множитель группы, nullptr = только замер
    int nNodes, nGroups;      // nNodes = узлы основной сети, клон узла i лежит в N + i
};

// Запись в свой элемент, без atomicAdd: порядок сложения на GPU плавает, а d входит в состояние клона.
extern "C" __global__ void neuralLyapKernel(const NeuralLyapArgs A) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= A.nGroups) return;
    const int i0 = A.groupStart[g], i1 = A.groupStart[g + 1], N = A.nNodes;
    numb sum = (numb)0;
    for (int k = 0; k < AMOUNTOFX; ++k) {
        numb* col = A.state + (size_t)k * 2 * N;
        if (A.scale == nullptr) {
            for (int i = i0; i < i1; ++i) { const numb d = col[N + i] - col[i]; sum += d * d; }
        } else {
            const numb c = A.scale[g];
            for (int i = i0; i < i1; ++i) col[N + i] = col[i] + c * (col[N + i] - col[i]);
        }
    }
    if (A.scale == nullptr) A.dist[g] = sum;
}

// Только пересечения порога снизу вверх на сетке записи: фазу между ними считает хост, ему нужно и следующее.
extern "C" __global__ void neuralOrderKernel(const NeuralOrderArgs A) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= A.nNodes) return;
    const numb x = A.column[node], p = A.prev[node];
    A.prev[node] = x;
    if (A.first || !(p < A.threshold && x >= A.threshold)) return;
    const int at = atomicAdd(A.count, 1);
    if (at >= A.capacity) return;
    A.outNode[at] = node;
    A.outTime[at] = A.t - A.dt + A.dt * (A.threshold - p) / (x - p);
}
