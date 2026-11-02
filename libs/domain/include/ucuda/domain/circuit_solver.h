#pragma once
#include <string>
#include <vector>
#include "ucuda/domain/circuit_synth.h"

// Референсный решатель схемы на CPU (фаза 2 плана): MNA + BDF2 + Ньютон.
//
// Схема — это ДАУ, а не ОДУ: у чисто резистивных узлов производной нет, матрица
// ёмкостей вырождена. Поэтому явные методы неприменимы принципиально, и шаг
// всегда неявный.
//
// Реактивные элементы формулируются ЧЕРЕЗ ЗАРЯД (интегрируется q, ток = dq/dt).
// Отсюда же следует, куда класть начальные условия: выход интегратора держит
// нуллор ОУ, свободной переменной он не является, а состояние сидит в заряде
// конденсатора обратной связи.
//
// Здесь только типы; класс CircuitSolver, build_mna_layout и circuit_simulate — в
// ucuda/numerics/circuit_solver.h (реализация в numerics).

struct OpAmpModel {
    // Идеальный ОУ (нуллор) — оракул фаз 1-3. Он же обесценивает проект, если
    // остаться на нём: на выходе получится ровно исходная ОДУ.
    bool   ideal   = true;
    double a0      = 1.0e5;    // усиление на постоянном токе
    double gbw_hz  = 3.0e6;    // полоса единичного усиления
    double vsat    = 13.0;     // насыщение выхода, симметричное
    // Ограничение выхода мягкое (tanh): жёсткий клип даёт нулевую производную,
    // и Ньютон на нём встаёт.
    double slew    = 0.0;      // 0 = не моделируется (решение фазы 2.5)
};

struct CircuitSolverConfig {
    OpAmpModel opamp;
    int    newton_max_iters = 20;
    // > 0 — ровно столько итераций, без выхода по невязке. Нужен сверке с GPU:
    // там ветвление убрано, и алгоритмы обязаны совпадать операция в операцию.
    int    newton_fixed     = 0;
    double newton_tol       = 1.0e-10;   // по бесконечной норме поправки, В
    double newton_max_step  = 5.0;       // limiting: потолок |dV| за итерацию
    double pivot_min        = 1.0e-14;
    // Внутренняя ёмкость полюса ОУ: значение произвольно, полюс задаётся парой
    // (gm, R_p), которая из неё и gbw вычисляется.
    double opamp_cpole      = 1.0e-9;
};

struct CircuitSolverStats {
    long long steps            = 0;
    long long newton_iters     = 0;
    int       max_iters_in_step = 0;
    double    max_residual     = 0.0;   // по последней итерации каждого шага
    bool      converged_always = true;
    bool      pivot_failed     = false;
};

// Раскладка неизвестных MNA. Вынесена наружу, потому что её обязаны разделять CPU-решатель
// и GPU-кодогенератор: разъехавшись, они дадут разные индексы при одинаковой схеме, и
// сверка GPU против CPU будет сравнивать разные вещи.
//
// Порядок неизвестных: [напряжения узлов 1..n_nodes-1][токи ветвей].
struct MnaLayout {
    enum class BranchKind { VSource, OpAmpIdeal, OpAmpReal, Multiplier };

    struct Branch {
        BranchKind kind;
        int        elem = 0;      // индекс в соответствующем списке графа
        int        out  = 0;      // узел, куда втекает ток ветви
        int        pole = -1;     // внутренний узел полюса, только у OpAmpReal
    };

    struct Cap {
        int    a = 0, b = 0;
        double farad = 0.0;
    };

    CircuitGraph        graph;      // граф, расширенный внутренними узлами полюсов
    std::vector<Branch> branches;
    std::vector<Cap>    caps;

    int    n_nodes = 0;   // включая землю и внутренние узлы
    int    n_v     = 0;   // неизвестных-напряжений
    int    size    = 0;   // полный размер системы
    double gm = 0.0, rp = 0.0;   // параметры однополюсной модели

    int idx(int node) const { return node > 0 ? node - 1 : -1; }
};

// --- прогон ------------------------------------------------------------------

struct CircuitRunResult {
    bool                             ok = false;
    std::string                      error;
    std::vector<double>              t;       // время В ЕДИНИЦАХ ОДУ
    std::vector<std::vector<double>> x;       // [переменная][отсчёт], вольты
    CircuitSolverStats               stats;
};
