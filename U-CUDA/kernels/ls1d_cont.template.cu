// ls1d_cont.template.cu
//
// NVRTC template for the 1D Lyapunov spectrum in continuation mode. Same shape
// as lle1d_cont.template.cu, but the tangent space is N-dimensional: N probes
// plus Gram-Schmidt re-orthogonalisation every NT time units. Mirrors
// run_ls1d_cpu in numerics.
//
// Carried between parameter points: the trajectory x[] AND all N probes y[].
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
// result[j * amountOfX + k] -- k-th exponent at point j, NaN if the point died.
extern "C" __global__ void ls1dContinuationKernel(
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
    numb NT,
    numb tMax,
    numb transientTime,
    numb eps,
    numb maxValue,
    numb* result)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    numb x[AMOUNTOFX];
    numb y[AMOUNTOFX * AMOUNTOFX];   // N probes, absolute coordinates
    numb z[AMOUNTOFX * AMOUNTOFX];   // Gram-Schmidt scratch
    numb denominators[AMOUNTOFX];
    numb sum[AMOUNTOFX];
    numb a[64];

    for (int i = 0; i < amountOfX; ++i)      x[i] = baseX[i];
    for (int i = 0; i < amountOfValues; ++i) a[i] = baseValues[i];

    curandState_t rng;
    curand_init(1234567891ULL, 0ULL, 0ULL, &rng);

    // Fresh random basis of N unit vectors.
    for (int k = 0; k < amountOfX; ++k) {
        numb zp = 0;
        for (int i = 0; i < amountOfX; ++i) {
            y[k * amountOfX + i] = curand_uniform(&rng) - (numb)0.5;
            zp += y[k * amountOfX + i] * y[k * amountOfX + i];
        }
        zp = sqrt(zp);
        for (int i = 0; i < amountOfX; ++i)
            y[k * amountOfX + i] = (zp == 0) ? (numb)0 : y[k * amountOfX + i] / zp;
    }

    int probesAttached = 0;
    const numb denom = (numb)(nPts > 1 ? nPts - 1 : 1);
    const int nBlocks = (int)(tMax / NT);

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

        const int badPoint = (hLocal <= 0 || nBlocks <= 0);
        const int ntSteps   = badPoint ? 0 : (int)(NT / hLocal);
        const int skipSteps = badPoint ? 0 : (int)(transientTime / hLocal);
        if (badPoint || ntSteps <= 0) {
            for (int k = 0; k < amountOfX; ++k) result[j * amountOfX + k] = nan("");
            continue;
        }

        int alive = 1;
        int settleBlocks = (int)(transientTime / NT);
        if (!probesAttached) {
            settleBlocks = 0;
            if (loopCalculateDiscreteModel_int(x, a, hLocal, skipSteps, amountOfX,
                                               1, 0, maxValue, nullptr, 0, 1) == 0) {
                alive = 0;
            } else {
                // Orthonormalise the random basis, then place the probes at
                // distance eps around the current trajectory point.
                gramSchmidtProcess(y, z, amountOfX);
                for (int k = 0; k < amountOfX; ++k)
                    for (int i = 0; i < amountOfX; ++i)
                        y[k * amountOfX + i] = z[k * amountOfX + i] * eps + x[i];
                probesAttached = 1;
            }
        }

        for (int k = 0; k < amountOfX; ++k) sum[k] = 0;

        const int totalBlocks = settleBlocks + nBlocks;
        for (int b = 0; alive && b < totalBlocks; ++b) {
            if (loopCalculateDiscreteModel_int(x, a, hLocal, ntSteps, amountOfX,
                                               1, 0, maxValue, nullptr, 0, 1) == 0) { alive = 0; break; }
            for (int k = 0; k < amountOfX; ++k)
                if (loopCalculateDiscreteModel_int(y + k * amountOfX, a, hLocal, ntSteps, amountOfX,
                                                   1, 0, maxValue, nullptr, 0, 1) == 0) { alive = 0; break; }
            if (!alive) break;

            for (int k = 0; k < amountOfX; ++k)
                for (int l = 0; l < amountOfX; ++l)
                    y[k * amountOfX + l] -= x[l];

            gramSchmidtProcess(y, z, amountOfX, denominators);

            for (int k = 0; k < amountOfX; ++k) {
                if (b >= settleBlocks) sum[k] += log(denominators[k] / eps);
                for (int i = 0; i < amountOfX; ++i)
                    y[k * amountOfX + i] = x[i] + z[k * amountOfX + i] * eps;
            }
        }

        if (alive) {
            for (int k = 0; k < amountOfX; ++k) result[j * amountOfX + k] = sum[k] / tMax;
        } else {
            for (int k = 0; k < amountOfX; ++k) result[j * amountOfX + k] = nan("");
            probesAttached = 0;
            for (int i = 0; i < amountOfX; ++i) x[i] = baseX[i];
            curand_init(1234567891ULL, (unsigned long long)j + 1ULL, 0ULL, &rng);
            for (int k = 0; k < amountOfX; ++k) {
                numb zp = 0;
                for (int i = 0; i < amountOfX; ++i) {
                    y[k * amountOfX + i] = curand_uniform(&rng) - (numb)0.5;
                    zp += y[k * amountOfX + i] * y[k * amountOfX + i];
                }
                zp = sqrt(zp);
                for (int i = 0; i < amountOfX; ++i)
                    y[k * amountOfX + i] = (zp == 0) ? (numb)0 : y[k * amountOfX + i] / zp;
            }
        }
    }
}
