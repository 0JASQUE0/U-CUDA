// circuit.template.cu
// NVRTC-шаблон схемного решателя. Плейсхолдеры:
//   CIRCUIT_BODY — весь ход потока, печатает emit_circuit_step (circuit_emit.cpp)
//
// Шаблон намеренно НИЧЕГО не знает про индексы узлов и порядок исключения: всё,
// что зависит от раскладки MNA, печатается кодогенератором. Иначе шаблон разъехался
// бы с MnaLayout при первой же правке штампов.
//
// Номиналы приходят массивом comp, а не литералами в коде. Это не стилистика:
// запечённые константы означали бы отдельную компиляцию на каждую точку свипа по
// номиналам, то есть смерть фазы 6.
//
// Один поток = одна схема. Решение принято по замеру структуры (см. фазу 3 плана):
// у Лоренца fill-in 5 клеток и 9 умножений-сложений на всю факторизацию, рабочий
// набор укладывается в регистры.

#ifdef __CUDACC_RTC__
typedef long long          int64_t;
typedef unsigned long long uint64_t;
#endif

extern "C" __global__ void circuitKernel(
    const double* __restrict__ comp,   // номиналы и параметры моделей
    const double* __restrict__ x0,     // [n_threads][n_vars] начальные условия
    double* __restrict__       out,    // [n_threads][n_samples][n_vars]
    int    n_threads,
    double h,                          // шаг в СЕКУНДАХ схемного времени
    int    n_steps,
    int    sample_every,
    long long n_samples,
    int* __restrict__ fail)            // [n_threads], 1 = встретился малый ведущий
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;

    // Отказ помечается флагом и НЕ роняет ядро: в ансамбле по допускам выпадение
    // отдельного набора номиналов — это результат, а не авария.
    int bad = 0;

{{CIRCUIT_BODY}}

    fail[tid] = bad;
}
