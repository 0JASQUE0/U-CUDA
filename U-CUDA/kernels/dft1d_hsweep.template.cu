// dft1d_hsweep.template.cu
//
// NVRTC template for the classical (non-continuation) 1D parametric DFT when
// the swept quantity is the integration step h itself. One thread per sweep
// point, like calculateDiscreteModelCUDA -- points are independent here, unlike
// the continuation chain in dft1d_cont.template.cu.
//
// Why a dedicated kernel instead of calculateDiscreteModelCUDA + DFT_custom
// (what the param/IC sweep still uses): under an h-sweep every point has its
// own step, hence its own sample count, its own window length AND its own
// sampling interval feeding the frequency basis. DFT_custom takes one
// sizeOfBlock, one precomputed window and one h for the whole launch, so it
// cannot express any of the three. Trajectory + DFT live in one kernel here,
// which also removes the need for an actualIterations round-trip.
//
// Flag semantics mirror run_dft1d_cpu exactly (that is the reference the GPU is
// compared against): transient collapsing to a fixed point (flag 0) marks the
// point dead with -1, while a block that collapses mid-way keeps its own flag
// (0 -> filled with 0.0, -1 -> filled with -1.0), matching DFT_custom's fill.
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
// 0 = rectangular, 1 = Hanning (default), 2 = Hamming. Same three formulas as
// build_window / cpu_build_window on the host and dft_window in
// dft1d_cont.template.cu.
__device__ __forceinline__ numb dft_hsweep_window(int n, int len, int window_type) {
    if (window_type == 0) return (numb)1.0;
    const numb gamma = (numb)2.0 * (numb)pi / (numb)(len - 1);
    if (window_type == 2) return (numb)0.53836 - (numb)0.46164 * cos(gamma * (numb)n);
    return (numb)0.5 * ((numb)1.0 - cos(gamma * (numb)n));
}

// One thread per sweep point of the current chunk.
//   amountOfCalculatedPoints -- index of this chunk's first point in the full
//                               sweep, so h is taken off the global grid (the
//                               host chunks by nPtsLimiter, as run_bif1d does).
//   d_data  -- scratch, nPtsLimiter * maxPointsInBlock; each thread owns the
//              slice at idx * maxPointsInBlock (worst case = smallest swept h).
//   AkCOS/BkSIN -- [nPtsLimiter * nFreq], point-major / freq-minor.
//   flags[idx]  -- 1 ok, 0 fixed point, -1 unusable.
extern "C" __global__ void dft1dHSweepKernel(
    int nPts,
    int nPtsLimiter,
    int amountOfCalculatedPoints,
    numb lo,
    numb hi,
    int logScale,
    const numb* baseValues,
    int amountOfValues,
    const numb* baseX,
    int amountOfX,
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
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= nPtsLimiter) return;

    const int gidx = amountOfCalculatedPoints + idx;

    // Same grid helpers as every other sweep, so the h axis lines up with what
    // the host reports as param_lo..param_hi.
    const numb hLocal = logScale
        ? getValueByIdx_log(gidx, nPts, lo, hi, 0)
        : getValueByIdx((size_t)gidx, nPts, lo, hi, 0);

    const size_t outBase = (size_t)idx * (size_t)nFreq;

    // Truncation, not ceil: matches run_dft1d_cpu and dft1d_cont.template.cu.
    int blockLen = (hLocal > (numb)0) ? (int)(tMax / hLocal / (numb)preScaller) : 0;
    if (blockLen > maxPointsInBlock) blockLen = maxPointsInBlock;
    const int transientSteps = (hLocal > (numb)0) ? (int)(transientTime / hLocal) : 0;

    // Fill helper: -1 and 0 reproduce DFT_custom's two dead-point fills.
    if (hLocal <= (numb)0 || blockLen <= 2) {
        for (int k = 0; k < nFreq; ++k) { AkCOS[outBase + k] = (numb)-1.0; BkSIN[outBase + k] = (numb)-1.0; }
        flags[idx] = -1;
        return;
    }

    numb x[AMOUNTOFX];
    numb a[64];
    for (int i = 0; i < amountOfX; ++i)      x[i] = baseX[i];
    for (int i = 0; i < amountOfValues; ++i) a[i] = baseValues[i];

    numb* myData = d_data + (size_t)idx * (size_t)maxPointsInBlock;

    // Transient. preScaller/writableVar are irrelevant with data == nullptr
    // (loopCalculateDiscreteModel_int only applies them when writing), so pass
    // the same 1/0 the CPU path does.
    int flag = loopCalculateDiscreteModel_int(
        x, a, hLocal, transientSteps, amountOfX, 1, 0, maxValue, nullptr, 0, 1);
    if (flag == 0) {
        for (int k = 0; k < nFreq; ++k) { AkCOS[outBase + k] = (numb)-1.0; BkSIN[outBase + k] = (numb)-1.0; }
        flags[idx] = -1;
        return;
    }

    flag = loopCalculateDiscreteModel_int(
        x, a, hLocal, blockLen, amountOfX, preScaller, writableVar, maxValue, myData, 0, 1);
    if (flag == -1 || flag == 0) {
        const numb fill = (flag == -1) ? (numb)-1.0 : (numb)0.0;
        for (int k = 0; k < nFreq; ++k) { AkCOS[outBase + k] = fill; BkSIN[outBase + k] = fill; }
        flags[idx] = flag;
        return;
    }

    // --- DFT over the block just recorded (mirrors DFT_custom / cpu_dft_block) ---
    const numb hSample = hLocal * (numb)preScaller;
    const numb f_step  = (nFreq > 1) ? (freqHi - freqLo) / (numb)(nFreq - 1) : (numb)0.0;
    const numb psi     = (numb)2.0 * (numb)pi * hSample;
    const int  RESET_INTERVAL = 1000;

    for (int k = 0; k < nFreq; ++k) {
        numb f_k;
        if (logFreqAxis) {
            const numb l0 = log10(freqLo), l1 = log10(freqHi);
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
            const numb wd = dft_hsweep_window(n, blockLen, windowType) * myData[n];
            ak += wd * cos_n;
            bk += wd * sin_n;
            const numb new_cos = cos_n * cos_theta - sin_n * sin_theta;
            const numb new_sin = sin_n * cos_theta + cos_n * sin_theta;
            cos_n = new_cos;
            sin_n = new_sin;
            if (--reset_counter == 0) {
                reset_counter = RESET_INTERVAL;
                if (n + 1 < blockLen) {
                    // cosf/sinf on purpose: DFT_custom resets through float and
                    // cpu_dft_block copies that, so all three paths drift alike.
                    const numb exact = psi * f_k * (numb)(n + 1);
                    cos_n = cosf(exact);
                    sin_n = sinf(exact);
                }
            }
        }
        AkCOS[outBase + k] = ak / (numb)blockLen;
        BkSIN[outBase + k] = bk / (numb)blockLen;
    }
    flags[idx] = 1;
}
