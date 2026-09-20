#pragma once
#include <string>
#include <vector>
#include "circuit_synth.h"

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

bool build_mna_layout(const CircuitGraph& g, const CircuitSolverConfig& cfg,
                      MnaLayout& out, std::string* err);

class CircuitSolver {
public:
    // Раскладывает граф в MNA: узлы + токи ветвей элементов, заданных напряжением
    // (источники, выходы ОУ, выходы умножителей), + внутренние узлы полюсов ОУ.
    bool build(const CircuitGraph& g, const CircuitSolverConfig& cfg, std::string* err);

    // Начальные условия задаются ЗАРЯДАМИ интеграторов: q = C*(V(s) - V(out)).
    // Без стартового толчка схема сидит в неподвижной точке.
    void set_initial_variables(const std::vector<double>& x0);

    // Один шаг величиной h СЕКУНД схемного времени. Первый шаг идёт по BDF1:
    // второй точки истории ещё нет.
    bool step(double h, std::string* err);

    double node_voltage(int node) const;
    double variable(int i) const;          // напряжение узла переменной i
    int    dimension() const { return size_; }

    const CircuitSolverStats& stats() const { return stats_; }
    void reset_stats() { stats_ = CircuitSolverStats(); }

    const MnaLayout& layout() const { return lay_; }

    // Пересобирает якобиан в текущей точке, не решая: после step() матрица
    // разрушена LU на месте. Нужен проверке структурного портрета в codegen.
    const std::vector<double>& debug_assemble_jacobian(double h);

private:
    int  idx(int node) const { return lay_.idx(node); }
    void add_j(int r, int c, double v) { if (r >= 0 && c >= 0) jac_[(size_t)r * size_ + c] += v; }
    void add_f(int r, double v)        { if (r >= 0) res_[(size_t)r] += v; }
    void assemble(double alpha, bool bdf2, double h);
    bool solve_linear(std::string* err);

    MnaLayout           lay_;
    CircuitSolverConfig cfg_;
    CircuitSolverStats  stats_;

    int  n_v_  = 0;
    int  size_ = 0;

    std::vector<double> q_prev_, q_prev2_;   // история зарядов, по одной на конденсатор
    std::vector<double> x_;                  // текущее решение
    std::vector<double> jac_, res_, delta_;
    std::vector<int>    pivot_;
    bool                started_ = false;   // была ли уже хоть одна точка истории
};

// --- прогон ------------------------------------------------------------------

struct CircuitRunResult {
    bool                             ok = false;
    std::string                      error;
    std::vector<double>              t;       // время В ЕДИНИЦАХ ОДУ
    std::vector<std::vector<double>> x;       // [переменная][отсчёт], вольты
    CircuitSolverStats               stats;
};

// h_ode и t_end_ode — в единицах времени ОДУ; перевод в секунды делает
// g.time_scale, чтобы вызывающий не думал про RC.
CircuitRunResult circuit_simulate(const CircuitGraph& g,
                                  const CircuitSolverConfig& cfg,
                                  const std::vector<double>& x0,
                                  double h_ode, double t_end_ode,
                                  double sample_every_ode);

// Прогон синтезированного Лоренца на идеальных ОУ против эталона фазы 0.
// fixture_csv может быть nullptr — тогда поточечная часть пропускается.
std::string circuit_solver_selftest(const char* fixture_csv, bool* ok);
