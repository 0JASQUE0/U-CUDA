// dft1d_cont.template.cu
//
// NVRTC template for the 1D parametric DFT in continuation mode: every point
// starts from the state left by the previous one. Single-threaded, like the
// Bif/LLE/LS continuation kernels -- the sweep is a dependency chain.
//
// Why a dedicated kernel instead of reusing bifurcation1dContinuationKernel +
// DFT_custom (the old continuation path did that): under an h-sweep every
// point has its own step, hence its own sample count AND its own window
// length. DFT_custom takes one sizeOfBlock, one h and one precomputed window
// for the whole launch, so it cannot express that. Doing trajectory + DFT in
// one place also drops the nPts x sizeOfBlock scratch buffer down to a single
// block -- a single thread never needs more.
//
// Placeholders:
//   AMOUNT_OF_X -- system dimension.
//   KRS_BODY    -- body of calculateDiscreteModel from codegen.

#define AMOUNTOFX {{AMOUNT_OF_X}}
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

// curand stub -- DFT itself needs no RNG, but cudaLibrary.cuh pulls the header.
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

// Window value at sample n of a block of `len` samples.
// 0 = rectangular, 1 = Hanning (default), 2 = Hamming. Mirrors build_window /
// cpu_build_window on the host.
__device__ __forceinline__ numb dft_window(int n, int len, int window_type) {
    if (window_type == 0) return (numb)1.0;
    const numb gamma = (numb)2.0 * (numb)pi / (numb)(len - 1);
    if (window_type == 2) return (numb)0.53836 - (numb)0.46164 * cos(gamma * (numb)n);
    return (numb)0.5 * ((numb)1.0 - cos(gamma * (numb)n));
}

// Single-thread kernel, launched with gridDim = blockDim = 1.
//   d_data -- scratch for ONE block, length maxPointsInBlock.
//   AkCOS/BkSIN -- [nPts * nFreq], param-major / freq-minor (as elsewhere).
//   flags[j] -- 1 ok, -1 unusable (diverged transient or collapse to a point).
extern "C" __global__ void dft1dContinuationKernel(
    int nPts,
    numb lo,
    numb hi,
    int reverse,
    int logScale,
    int sweepIsH,
    int mutParamIdx,
    const numb* baseValues,
    int amountOfValues,
    const numb* baseX,
    int amountOfX,
    numb hBase,
    numb tMax,
    numb transientTime,
    int preScaller,
    int writableVar,
    numb maxValue,
    int nFreq,
    numb freqLo,
    numb freqHi,
    int logFreqAxis,
    int windowType,
    int maxPointsInBlock,
    numb* d_data,
    numb* AkCOS,
    numb* BkSIN,
    int*  flags)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    numb x[AMOUNTOFX];
    numb a[64];
    for (int i = 0; i < amountOfX; ++i)      x[i] = baseX[i];
    for (int i = 0; i < amountOfValues; ++i) a[i] = baseValues[i];

    const numb denom = (numb)(nPts > 1 ? nPts - 1 : 1);
    const int  RESET_INTERVAL = 1000;

    for (int j = 0; j < nPts; ++j) {
        const numb t = (numb)j / denom;
        numb p;
        if (logScale) {
            numb l0 = log10(lo), l1 = log10(hi);
            p = pow((numb)10.0, reverse ? (l1 - (l1 - l0) * t) : (l0 + (l1 - l0) * t));
        } else {
            p = reverse ? (hi - (hi - lo) * t) : (lo + (hi - lo) * t);
        }

        numb hLocal = hBase;
        if (sweepIsH) hLocal = p;
        else          a[mutParamIdx] = p;

        int blockLen = (hLocal > 0) ? (int)(tMax / hLocal / (numb)preScaller) : 0;
        if (blockLen > maxPointsInBlock) blockLen = maxPointsInBlock;
        const int transientSteps = (hLocal > 0) ? (int)(transientTime / hLocal) : 0;

        int dead = 0;
        if (hLocal <= 0 || blockLen <= 2) {
            dead = 1;
        } else {
            // Transient: x[] carried over from the previous point.
            int flag = loopCalculateDiscreteModel_int(
                x, a, hLocal, transientSteps, amountOfX, 1, 0,
                maxValue, nullptr, 0, 1);
            if (flag == 0) {
                dead = 1;
            } else {
                flag = loopCalculateDiscreteModel_int(
                    x, a, hLocal, blockLen, amountOfX, preScaller, writableVar,
                    maxValue, d_data, 0, 1);
                if (flag == -1) dead = 1;
            }
        }

        const size_t outBase = (size_t)j * (size_t)nFreq;
        if (dead) {
            // Same fill as DFT_custom for an unusable point.
            for (int k = 0; k < nFreq; ++k) { AkCOS[outBase + k] = (numb)-1.0; BkSIN[outBase + k] = (numb)-1.0; }
            flags[j] = -1;
            // Break the chain: next point restarts from the initial conditions.
            for (int i = 0; i < amountOfX; ++i) x[i] = baseX[i];
            continue;
        }

        // --- DFT over the block just recorded (mirrors DFT_custom) ---
        const numb hSample = hLocal * (numb)preScaller;
        const numb f_step  = (nFreq > 1) ? (freqHi - freqLo) / (numb)(nFreq - 1) : (numb)0.0;
        const numb psi     = (numb)2.0 * (numb)pi * hSample;

        for (int k = 0; k < nFreq; ++k) {
            numb f_k;
            if (logFreqAxis) {
                numb l0 = log10(freqLo), l1 = log10(freqHi);
                f_k = pow((numb)10.0, (nFreq > 1) ? (l0 + (l1 - l0) * (numb)k / (numb)(nFreq - 1)) : l0);
            } else {
                f_k = freqLo + (numb)k * f_step;
            }
            const numb cos_theta = cos((numb)2.0 * (numb)pi * hSample * f_k);
            const numb sin_theta = sin((numb)2.0 * (numb)pi * hSample * f_k);
            numb cos_n = (numb)1.0, sin_n = (numb)0.0;
            numb ak = 0, bk = 0;
            int reset_counter = RESET_INTERVAL;

            for (int n = 0; n < blockLen; ++n) {
                const numb wd = dft_window(n, blockLen, windowType) * d_data[n];
                ak += wd * cos_n;
                bk += wd * sin_n;
                const numb new_cos = cos_n * cos_theta - sin_n * sin_theta;
                const numb new_sin = sin_n * cos_theta + cos_n * sin_theta;
                cos_n = new_cos;
                sin_n = new_sin;
                if (--reset_counter == 0) {
                    reset_counter = RESET_INTERVAL;
                    if (n + 1 < blockLen) {
                        const numb exact = psi * f_k * (numb)(n + 1);
                        cos_n = cosf(exact);
                        sin_n = sinf(exact);
                    }
                }
            }
            AkCOS[outBase + k] = ak / (numb)blockLen;
            BkSIN[outBase + k] = bk / (numb)blockLen;
        }
        flags[j] = 1;
    }
}
