#pragma once
#include <string>
#include <vector>
#include "circuit_codegen.h"

// Запуск схемного решателя на GPU: эмиссия -> NVRTC -> ядро (фаза 3 плана).
//
// Один поток — одна схема. Решение принято по замеру структуры, а не по оценке:
// у Лоренца fill-in 5 клеток, 9 умножений-сложений на факторизацию, ядро занимает
// 145 регистров без единого байта локальной памяти.

class CircuitGpuRunner {
public:
    ~CircuitGpuRunner();

    // Планирует исключение, печатает ядро, компилирует. Держит модуль живым.
    bool build(const CircuitGraph& g, const CircuitSolverConfig& scfg,
               const EmitConfig& ec, std::string* err);

    // x0_flat — [n_threads][n_vars]. out — [n_threads][n_samples][n_vars].
    // fail[tid] = 1, если поток встретил малый ведущий элемент; ядро при этом
    // не падает: в ансамбле по допускам выпадение набора номиналов — результат.
    bool run(const std::vector<double>& x0_flat, int n_threads,
             double h_ode, double t_end_ode, double sample_every_ode,
             std::vector<double>& out, std::vector<int>& fail, std::string* err);

    // Номиналы, которые уедут в ядро. Порядок задаёт emit_circuit_step; менять
    // их между запусками можно без перекомпиляции — в этом весь смысл.
    std::vector<double>& components() { return comp_; }

    int registers()   const { return regs_; }
    int local_bytes() const { return lmem_; }
    const EmittedKernel&   emitted() const { return emit_; }
    const EliminationPlan& plan()    const { return plan_; }
    const std::string&     source()  const { return source_; }

private:
    void release();

    EliminationPlan plan_;
    EmittedKernel   emit_;
    MnaLayout       lay_;
    std::string     source_;
    std::vector<double> comp_;

    void* module_ = nullptr;   // CUmodule
    void* func_   = nullptr;   // CUfunction
    int   device_ = -1;        // устройство с удержанным первичным контекстом
    int   regs_ = 0, lmem_ = 0;
    double time_scale_ = 0.0;
    int   n_vars_ = 0;
};

// Сверка GPU против CPU на идентичных входах и шаге. Здесь сравнение жёсткое и
// поточечное: расхождение должно быть на уровне арифметики, а не инвариантов.
std::string circuit_nvrtc_selftest(bool* ok);
