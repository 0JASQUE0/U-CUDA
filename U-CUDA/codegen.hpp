#pragma once
#include <string>
#include <vector>
#include "configCUDA.h"   // ucmplx — состояние комплексных схем (см. eval_complex)

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

    // Newton knobs for the implicit schemes. They live here (and not in every
    // per-analysis config) because AppModel::build_system() is the single System
    // factory and its result is copied into every session, so codegen sees them
    // without touching analysis_session.h / session_io.cpp.
    // newton_full: false = modified Newton (Jacobian and LU reused across
    // iterations, refreshed when the correction stops shrinking), true = full
    // Newton (rebuilt every iteration).
    bool   newton_full      = false;
    double newton_tol       = 1e-10;
    int    newton_max_iters = 8;
};

// Конечно-разностные схемы интегрирования.
// DOPRI78 — Dormand-Prince 8(7), 13-стадийный embedded метод (см. scheme_dopri78
// в codegen.cpp). Сейчас используется только 8-й порядок (b[0]); 7-й порядок (z)
// тоже считается — резерв под будущий адаптивный шаг по |y-z|.
// ComplexCD — та же композиция, что CD, но с комплексными полушагами:
// h1 = s*h + i*h*sqrt(3)/6, h2 = (1-s)*h - i*h*sqrt(3)/6, где s = a[0] — тот же
// коэффициент симметрии, что у CD (дефолт 0.5). h1 + h2 = h при любом s; при
// s = 1/2 полушаги комплексно-сопряжены. Шаг целиком считается в комплексной
// арифметике (ucmplx), наружу пишется Re.
// ComplexCD4 — композиция ДВУХ симметричных CD (s = 1/2) с комплексными
// шагами gamma*h и conj(gamma)*h, gamma = 1/2 + i*sqrt(3)/6. Условия
// alpha+beta = 1 и alpha^3+beta^3 = 0 гасят h³-член, h⁴ по сопряжённой
// симметрии мнимый и уходит с Re — глобальный порядок 4 (замерено 4.00 на
// Лоренце и Рёсслере). Требует s = 1/2: иначе внутренний CD несимметричен и
// порядок падает до первого.
// ImplicitEuler / ImplicitMidpoint — the only A-stable methods here, and the only
// ones that solve the FULL coupled system per step (CD is diagonally implicit: it
// solves each equation for its own variable and never sees the cross terms).
// Both run Newton on a symbolically differentiated Jacobian; see scheme_implicit_common
// in codegen.cpp. ImplicitMidpoint solves for the stage value Y = (X + X_next)/2,
// which makes it the same code as ImplicitEuler with h -> h/2 plus a final X = 2Y - X.
// SEMP / SIMP — методы средней точки с последовательной (Гаусс-Зейдель)
// стадией на полушаге h1 = s*h, s = a[0] (тот же слот симметрии, что у CD):
// SEMP считает стадию явно, SIMP — диагонально-неявно (каждое уравнение решено
// относительно своей переменной, как Phi* в CD). Корректор у обеих один:
// полный шаг h от исходного состояния по значениям стадии. Порядок 2 только
// при s = 1/2, см. scheme_semi_midpoint в codegen.cpp.
// D — диагонально-неявный метод первого порядка: стадия SIMP (она же Phi* из CD)
// как самостоятельный шаг, но на полный h. Прямой порядок по компонентам,
// каждое уравнение решается относительно своей переменной; a[0] не читает.
enum class Scheme { Euler, EulerCromer, ExplicitMidpoint, RK4, DOPRI78, CD, ComplexCD, ComplexCD4,
                    ImplicitEuler, ImplicitMidpoint, SEMP, SIMP, D };

// Генерирует тело шага схемы в виде C/CUDA-кода (строки вида
// "X[0] = X[0] + h * (...);"). Бросает std::runtime_error при ошибке разбора.
std::string codegen_scheme(const System& s, Scheme sch);

// Emits a human-readable C-code mirror of what the CPU integrator
// (integrator.cpp::step_*) actually computes. Now identical to codegen_scheme
// for every scheme: the CPU path evaluates the same AST, and for the
// diagonally-implicit ones (CD, Complex CD, SIMP, D) it solves each equation
// with the same analytic formula, via SystemEvaluator::solve_diag_implicit, falling
// back to iterations exactly where the GPU emits them. Kept as the single place
// the debug panel asks for the CPU form, and as the place to express a future
// divergence between the two paths.
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

    // False when the system has no symbolic derivative (floor/ceil/fmod): explicit
    // schemes still work, the implicit ones must refuse. eval_jacobian then zeroes J.
    bool has_jacobian() const;

    // Jacobian J[i*dim + j] = df_i/dx_j (X, a), row-major [dim*dim]. Same symbolic
    // derivatives the GPU scheme emits, so the CPU integrator runs the same algorithm.
    void eval_jacobian(const double* X, const double* a, double* J) const;

    // Newton settings carried over from the System this evaluator was built from —
    // exposed here so computePhasePortraitCPU needs no extra parameters.
    bool   newton_full() const;
    double newton_tol() const;
    int    newton_max_iters() const;

    // Решает i-е уравнение диагонально-неявного полушага относительно своей
    // переменной: X[i] = X[i] + hs * f_i(X), где f_i = coef*x_i + rem и ни coef,
    // ни rem от x_i не зависят. Разложение то же, что кодоген делает над AST
    // для CD / Complex CD / SIMP / D (cd_try_extract_linear), и формула
    // собрана так, чтобы повторить эмитируемый текст операция в операцию —
    // включая порядок умножений и вынос знака.
    // Возвращает false, если f_i нелинейна по своей переменной: тогда шаг
    // обязан откатиться на простые итерации, ровно как это делает GPU.
    bool solve_diag_implicit(int i, double* X, const double* a, double hs) const;
    bool solve_diag_implicit_complex(int i, ucmplx* Z, const double* a, ucmplx hs) const;

    // То же самое и по тому же байткоду, но в комплексной арифметике —
    // для схем с комплексными коэффициентами (Complex CD). Параметры a[]
    // остаются вещественными: комплексных значений в них нет.
    void eval_complex(const ucmplx* X, const double* a, ucmplx* deriv) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};