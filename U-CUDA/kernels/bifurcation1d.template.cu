// bifurcation1d.template.cu
//
// NVRTC-шаблон для 1D-бифуркации. Подставляется ParametricEngine'ом, потом
// компилируется через nvrtcCompileProgram. Плейсхолдеры:
//   AMOUNT_OF_X — размерность системы (число переменных, int)
//   KRS_BODY    — тело calculateDiscreteModel из codegen
// (без двойных фигурных скобок в комментариях — replace_all дубовый и заменит
//  их прямо тут, поломав код за пределами функции).
//
// Логика:
//   1. Подставляем AMOUNTOFX -> переопределяем размер через #define
//      (configCUDA.h оборачивает свой constexpr в #ifndef AMOUNTOFX).
//   2. Подключаем cudaLibrary.cuh — оттуда придут типы (numb), макросы, объявления.
//   3. Определяем calculateDiscreteModel с user's KRS. NVRTC видит её
//      до подключения cudaLibrary.cu, где default-версия закрыта #ifndef __CUDACC_RTC__.
//   4. Подключаем cudaLibrary.cu — kernel-ы (calculateDiscreteModelCUDA,
//      peakFinderCUDA и helpers) подхватятся и слинкуются с нашей KRS.

#define AMOUNTOFX {{AMOUNT_OF_X}}

// Для bif1d нам нужен PARAMETER analysis (par_or_var=1). По умолчанию в
// configCUDA.h стоит 0 (variable/IC analysis) — этот #define перекрывает.
// configCUDA.h обёрнут в #ifndef par_or_var, конфликта не будет.
#define par_or_var 1

// NVRTC не подтягивает <cstdint> автоматически, а cudaLibrary.cu использует
// int64_t в getValueByIdx и др. Даём typedef'ы для базовых fixed-width int'ов
// только под NVRTC; offline nvcc эти типы получает через <cstdint> в обычной
// сборке.
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

#include "cudaLibrary.cuh"

__device__ __host__ __forceinline__
void calculateDiscreteModel(numb* X, const numb* a, const numb h) {
{{KRS_BODY}}
}

#include "cudaLibrary.cu"
