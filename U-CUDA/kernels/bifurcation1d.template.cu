// bifurcation1d.template.cu
//
// Шаблон CUDA-ядра для 1D-бифуркации. Загружается ParametricEngine'ом,
// плейсхолдеры заменяются на конкретные значения, результат компилируется NVRTC.
//
// Плейсхолдеры (заменяются простым std::string::replace):
//   {{AMOUNT_OF_X}}  — размерность системы (число переменных, int)
//   {{KRS_BODY}}     — тело calculateDiscreteModel (строка C++ из codegen)
//
// Контракт ядра bifurcation1d_kernel:
//   - один thread считает одну точку по параметру
//   - параметр меняется линейно от param_lo до param_hi по индексу tid
//   - сначала прокручивается transient_steps шагов (без записи)
//   - затем record_steps раз пишется значение writable_var (с pre_scaller прореживанием)
//   - если в любой момент |X[k]| > max_value или nan — flag = -1, поток выходит
//
// Layout памяти на выходе:
//   out_data[tid * record_steps + t]  — значение writable_var в момент t у параметра tid
//   out_flags[tid]                    — 1 = ok, -1 = разошлось
//
// Для первого рабочего прохода LOCAL_VALUES_MAX = 64 хватит любой реальной системе.
// Если когда-то понадобится больше — перейти на shared/dynamic.
//

typedef double numb;

#define AMOUNTOFX {{AMOUNT_OF_X}}
#define LOCAL_VALUES_MAX 64

__device__ __forceinline__ void calculateDiscreteModel(numb* X, const numb* a, numb h) {
{{KRS_BODY}}
}

// Линейная интерполяция параметра по индексу idx ∈ [0, nPts).
__device__ __forceinline__ numb param_at(int idx, int nPts, numb lo, numb hi) {
    if (nPts <= 1) return lo;
    return lo + (hi - lo) * (numb)idx / (numb)(nPts - 1);
}

__device__ __forceinline__ bool diverged(const numb* X, numb maxAbs) {
    for (int k = 0; k < AMOUNTOFX; ++k) {
        numb v = X[k];
        if (!isfinite(v) || v > maxAbs || v < -maxAbs) return true;
    }
    return false;
}

extern "C" __global__ void bifurcation1d_kernel(
    const numb* __restrict__ initial_conditions,   // [AMOUNTOFX]
    const numb* __restrict__ base_values,          // [amount_of_values]
    int amount_of_values,
    int param_index,
    numb param_lo, numb param_hi,
    int n_pts,
    int writable_var,
    numb h,
    int transient_steps,
    int record_steps,
    int pre_scaller,
    numb max_value,
    numb* __restrict__ out_data,                   // [n_pts * record_steps]
    int*  __restrict__ out_flags                   // [n_pts]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pts) return;

    numb X[AMOUNTOFX];
    for (int i = 0; i < AMOUNTOFX; ++i) X[i] = initial_conditions[i];

    numb local_values[LOCAL_VALUES_MAX];
    for (int i = 0; i < amount_of_values; ++i) local_values[i] = base_values[i];
    local_values[param_index] = param_at(tid, n_pts, param_lo, param_hi);

    int flag = 1;

    // Transient — прокрутка без записи
    for (int s = 0; s < transient_steps; ++s) {
        calculateDiscreteModel(X, local_values, h);
        if (diverged(X, max_value)) { flag = -1; break; }
    }

    // Запись record_steps значений writable_var, по pre_scaller шагов между записями
    numb* row = out_data + (size_t)tid * record_steps;
    if (flag == 1) {
        for (int t = 0; t < record_steps; ++t) {
            for (int s = 0; s < pre_scaller; ++s) calculateDiscreteModel(X, local_values, h);
            row[t] = X[writable_var];
            if (diverged(X, max_value)) { flag = -1; break; }
        }
    }

    out_flags[tid] = flag;
}
