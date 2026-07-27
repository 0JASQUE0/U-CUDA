// lle1d_cont.template.cu
//
// NVRTC template for 1D LLE in continuation mode: every parameter point starts
// from the state left by the previous one. Single-threaded on the GPU -- the
// sweep is a dependency chain by construction, so it cannot be parallelised
// without breaking the hysteresis that continuation exists to capture. Mirrors
// bifurcation1d_cont.template.cu in shape and run_lle1d_cpu in numerics.
//
// Carried between points: the trajectory x[] AND the perturbation probe y[].
// By the end of a point the probe is already aligned with the direction of
// strongest stretching, so the next point does not have to re-align it from a
// random direction.
//
// Placeholders:
//   AMOUNT_OF_X -- system dimension.
//   KRS_BODY    -- body of calculateDiscreteModel from codegen.

#define AMOUNTOFX {{AMOUNT_OF_X}}
#define par_or_var 1   // continuation sweeps a parameter (or h), never an IC

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

// curand_kernel.h interception -- see lle1d.template.cu. The stub RNG below is
// the one run_lle1d_cpu copies verbatim, so CPU and GPU pick the same initial
// probe direction.
#ifdef __CUDACC_RTC__
#define CURAND_KERNEL_H_
#define CURAND_KERNEL_H
typedef struct { unsigned long long state; } curandState_t;
typedef curandState_t curandStateXORWOW_t;
__device__ __forceinline__ void curand_init(
    unsigned long long seed, unsigned long long sequence, unsigned long long offset, curandState_t* s) {
    unsigned long long z = seed + sequence * 0x9E3779B97F4A7C15ULL + offset;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    s->state = z ^ (z >> 31);
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

// Single-thread kernel, launched with gridDim = blockDim = 1.
//   sweepIsH  != 0 -> the swept quantity is the step h itself; a[] stays put and
//                     the per-point step counts are recomputed from h.
//   logScale  != 0 -> points are distributed logarithmically over [lo, hi].
//   result[j]      -- lambda for point j, or NaN when the point did not survive.
extern "C" __global__ void lle1dContinuationKernel(
    int nPts,
    numb lo,
    numb hi,
    int reverse,
    int logScale,
    int sweepIsH,
    int mutParamIdx,                 // 1-based index into a[]; ignored if sweepIsH
    const numb* baseValues,
    int amountOfValues,
    const numb* baseX,
    int amountOfX,
    numb hBase,
    numb NT,
    numb tMax,
    numb transientTime,
    numb eps,
    numb maxValue,
    numb* result)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    numb x[AMOUNTOFX];
    numb y[AMOUNTOFX];
    numb a[64];   // kMaxAmountOfValues in the engine

    for (int i = 0; i < amountOfX; ++i)      x[i] = baseX[i];
    for (int i = 0; i < amountOfValues; ++i) a[i] = baseValues[i];

    curandState_t rng;
    unsigned long long rngSeq = 0ULL;
    curand_init(1234567891ULL, rngSeq, 0ULL, &rng);

    // Fresh random unit direction for the probe.
    numb zPower = 0;
    for (int i = 0; i < amountOfX; ++i) { y[i] = curand_uniform(&rng) - (numb)0.5; zPower += y[i] * y[i]; }
    zPower = sqrt(zPower);
    for (int i = 0; i < amountOfX; ++i) y[i] = (zPower == 0) ? (numb)0 : y[i] / zPower;

    int probeAttached = 0;
    const numb denom = (numb)(nPts > 1 ? nPts - 1 : 1);
    const int nBlocks = (int)(tMax / NT);   // h-independent

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

        if (hLocal <= 0 || nBlocks <= 0) { result[j] = nan(""); continue; }

        const int ntSteps   = (int)(NT / hLocal);
        const int skipSteps = (int)(transientTime / hLocal);
        if (ntSteps <= 0) { result[j] = nan(""); continue; }

        int alive = 1;
        // settleBlocks: warm-up for a point that CARRIED the probe over -- run
        // in NT-blocks so the probe stays attached and keeps its orientation.
        // A point that attaches a fresh probe has already warmed up on the
        // trajectory alone (skipSteps below), so it gets no extra blocks.
        int settleBlocks = (int)(transientTime / NT);
        if (!probeAttached) {
            settleBlocks = 0;
            if (loopCalculateDiscreteModel_int(x, a, hLocal, skipSteps, amountOfX,
                                               1, 0, maxValue, nullptr, 0, 1) == 0) {
                alive = 0;
            } else {
                for (int i = 0; i < amountOfX; ++i) y[i] = y[i] * eps + x[i];
                probeAttached = 1;
            }
        }

        numb sum = 0;
        const int totalBlocks = settleBlocks + nBlocks;
        for (int b = 0; alive && b < totalBlocks; ++b) {
            if (loopCalculateDiscreteModel_int(x, a, hLocal, ntSteps, amountOfX,
                                               1, 0, maxValue, nullptr, 0, 1) == 0) { alive = 0; break; }
            if (loopCalculateDiscreteModel_int(y, a, hLocal, ntSteps, amountOfX,
                                               1, 0, maxValue, nullptr, 0, 1) == 0) { alive = 0; break; }

            numb d = 0;
            for (int l = 0; l < amountOfX; ++l) {
                numb q = (x[l] - y[l]) / eps;
                d += q * q;
            }
            d = sqrt(d);
            if (d <= (numb)1e-14) d = (numb)1e-14;

            // The first settleBlocks blocks only orient the probe; their growth
            // must not enter the average.
            if (b >= settleBlocks) sum += log(d);

            const numb inv = (numb)1.0 / d;
            for (int l = 0; l < amountOfX; ++l)
                y[l] = x[l] - ((x[l] - y[l] + (numb)1e-14) * inv);
        }

        if (alive) {
            result[j] = sum / tMax;
        } else {
            result[j] = nan("");
            // Break the chain: restart from the initial conditions with a fresh
            // probe direction so the next point does not inherit garbage.
            probeAttached = 0;
            for (int i = 0; i < amountOfX; ++i) x[i] = baseX[i];
            rngSeq = (unsigned long long)j + 1ULL;
            curand_init(1234567891ULL, rngSeq, 0ULL, &rng);
            numb zp = 0;
            for (int i = 0; i < amountOfX; ++i) { y[i] = curand_uniform(&rng) - (numb)0.5; zp += y[i] * y[i]; }
            zp = sqrt(zp);
            for (int i = 0; i < amountOfX; ++i) y[i] = (zp == 0) ? (numb)0 : y[i] / zp;
        }
    }
}
