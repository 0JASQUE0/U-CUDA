#pragma once
#include <string>
#include <vector>

// Описание динамической системы для кодгена.
//   vars   — переменные состояния, маппятся на X[0..N-1]
//   params — параметры системы, маппятся на a[1..M] (a[0] зарезервирован под коэф. схемы)
//   rhs    — правые части dX[i]/dt, по одной строке на переменную
//   latex  — если true, rhs трактуются как LaTeX, иначе как обычный синтаксис
struct System {
    std::vector<std::string> vars;
    std::vector<std::string> params;
    std::vector<std::string> rhs;
    bool latex = false;
};

// Конечно-разностные схемы интегрирования.
// DOPRI78 — Dormand-Prince 8(7), 13-стадийный embedded метод (см. scheme_dopri78
// в codegen.cpp). Сейчас используется только 8-й порядок (b[0]); 7-й порядок (z)
// тоже считается — резерв под будущий адаптивный шаг по |y-z|.
enum class Scheme { Euler, EulerCromer, ExplicitMidpoint, RK4, DOPRI78, CD };

// Генерирует тело шага схемы в виде C/CUDA-кода (строки вида
// "X[0] = X[0] + h * (...);"). Бросает std::runtime_error при ошибке разбора.
std::string codegen_scheme(const System& s, Scheme sch);

// Emits a human-readable C-code mirror of what the CPU integrator
// (integrator.cpp::step_*) actually computes. For Euler/RK4/etc this matches
// codegen_scheme bit-for-bit (same AST is evaluated either way). For CD it
// differs: GPU uses analytic solving for linear-in-var components, CPU uses
// 4 simple iterations for every variable — this function returns the CPU form
// so users can compare the two side by side in the debug panel.
std::string codegen_scheme_cpu_equivalent(const System& s, Scheme sch);

// Maps UI scheme name ("Euler" / "RK4" / "CD" / ...) to the Scheme enum.
// Unknown names fall back to Scheme::Euler.
Scheme scheme_from_name(const std::string& name);

// Нормализует числовое значение/выражение параметра для подстановки в C-код:
//   "8/3"   -> "8.0/3.0"   (вещественное деление, без потери точности)
//   "1e-5"  -> "1e-05"
//   "2"     -> "2.0"
//   "8.5"   -> "8.5"
// Пустая строка возвращается пустой. Бросает при синтаксической ошибке.
std::string normalize_value(const std::string& value);

#include <memory>

// Интерпретатор системы для расчёта на CPU без компиляции.
// Парсит правые части ОДИН РАЗ в байткод (постфиксная форма), затем быстро
// вычисляет производные на каждом шаге. Используется для расчёта одиночных
// траекторий (фазовый портрет) — быстро и со сменой системы на лету.
//
// Параметры передаются в формате a[] со сдвигом: a[0] зарезервирован,
// a[1..M] — значения параметров (тот же layout, что в codegen и в движке).
class SystemEvaluator {
public:
    // Парсит rhs системы в байткод. Бросает std::runtime_error при ошибке разбора.
    explicit SystemEvaluator(const System& sys);
    ~SystemEvaluator();
    SystemEvaluator(SystemEvaluator&&) noexcept;
    SystemEvaluator& operator=(SystemEvaluator&&) noexcept;

    int dim() const;  // число переменных состояния

    // Вычисляет производные: deriv[i] = f_i(X, a).
    //   X     — текущее состояние [dim]
    //   a     — параметры со сдвигом [>= params.size()+1], a[0] не используется
    //   deriv — выход [dim]
    void eval(const double* X, const double* a, double* deriv) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};