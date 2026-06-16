#pragma once
#include "configCUDA.h"

// Под NVRTC (__CUDACC_RTC__ определён автоматически) host-only заголовки и
// макросы недоступны и не нужны — kernel-ы их не используют. Под обычным
// nvcc всё работает как раньше.
#ifndef __CUDACC_RTC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define gpuGlobalErrorCheck() { gpuAssert((cudaGetLastError()), __FILE__, __LINE__); }
#define gpuDebugF(var) { printf("%f", var); printf("\n");}
#define gpuDebugArrF(var, z) { for (int i = 0; i < z; ++i) {printf("%f ", var[i]);} printf("\n");}

//__device__ __host__ inline int min(int a, int b) { return (a < b) ? a : b; }
//__device__ __host__ inline numb min(numb a, numb b) { return (a < b) ? a : b; }
//
//__device__ __host__ inline int max(int a, int b) { return (a > b) ? a : b; }
//__device__ __host__ inline numb max(numb a, numb b) { return (a > b) ? a : b; }

void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true);
#endif
