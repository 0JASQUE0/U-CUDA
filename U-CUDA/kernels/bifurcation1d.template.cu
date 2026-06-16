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

#include "cudaLibrary.cuh"

__device__ __host__ __forceinline__
void calculateDiscreteModel(numb* X, const numb* a, const numb h) {
{{KRS_BODY}}
}

#include "cudaLibrary.cu"
