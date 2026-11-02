#pragma once
// Референсный решатель схемы на CPU (MNA + BDF2 + Ньютон): класс решателя и прогон.
// Типы (OpAmpModel, CircuitSolverConfig, CircuitSolverStats, MnaLayout, CircuitRunResult)
// и обоснование формулировки через заряд — в ucuda/domain/circuit_solver.h.
#include <string>
#include <vector>
#include "ucuda/domain/circuit_solver.h"

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
