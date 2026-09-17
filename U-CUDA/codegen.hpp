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

    // Discrete map: rhs is x_{n+1} directly, not a derivative. It lives here for
    // the same reason as the Newton knobs below - build_system() is the single
    // System factory and its result is copied into every session. The config's
    // scheme is then ignored, so a stale session JSON holding some older scheme
    // cannot substitute a different step for the map.
    bool is_map = false;

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
// Complex Implicit Euler — композиция ДВУХ неявных Эйлеров с сопряжёнными
// комплексными шагами: tau1 = h*(s + i/2), tau2 = h*(1 - s - i/2), s = a[0]
// (тот же слот симметрии, что у CD). При s = 1/2 шаги комплексно-сопряжены,
// tau1 = h*(1+i)/2 и tau2 = conj(tau1).
// Почему именно эти коэффициенты: композиция метода первого порядка с самим
// собой поднимается до второго ровно при tau1 + tau2 = h и tau1^2 + tau2^2 = 0.
// Подставив tau1 = h*u, получаем u^2 + (1-u)^2 = 0, то есть u = (1 +- i)/2 —
// решение ЕДИНСТВЕННОЕ с точностью до сопряжения, свободных параметров у
// метода нет. Отклонение s от 1/2 оставляет у h^2-члена вещественную часть
// (её Re уже не убирает), и порядок падает до первого — ровно как у CD.
// Шаг целиком считается в ucmplx, наружу пишется Re; мнимая часть живёт
// внутри одного шага и в следующий не переносится.
// Map is not an integration scheme but the right-hand side of a discrete map
// x_{n+1} = f(x_n) itself. The only emitter that never uses h.
enum class Scheme { Euler, EulerCromer, ExplicitMidpoint, RK4, DOPRI78, CD, ComplexCD, ComplexCD4,
                    ImplicitEuler, ImplicitMidpoint, SEMP, SIMP, D, ComplexIEuler, Map };

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

// Паспортный порядок и симметричность встроенной схемы по её UI-имени.
// Единственный источник этих чисел в проекте: их читает и UI (группировка
// списка схем по порядку, вкладка Order), и резолвер КРС при сборке
// экстраполяционной обёртки — разъехаться копиям негде.
// Симметричной ("разложение ошибки только по чётным степеням h") помечены
// ровно две схемы: Implicit Midpoint и CD, обе как композиция Phi* ∘ Phi.
// У CD это верно ТОЛЬКО при a[0] = 1/2 — проверить в кодогене нельзя, a[0]
// приходит в рантайме, поэтому ответственность на UI-предупреждении.
// Complex CD / Complex CD4 намеренно НЕ помечены: сопряжённые полушаги — это
// не самосопряжённость, а структура разложения после взятия Re не замерена.
// Возвращает false для неизвестного имени (в т.ч. для кастомной КРС и для
// "Extr(...)"), выходные параметры тогда не трогаются.
bool builtin_scheme_traits(const std::string& name, int* order, bool* symmetric);

// --- Экстраполяция Ричардсона поверх произвольной опорной схемы -------------
//
// Схема-обёртка живёт в имени: "Extr(<база>|n1,n2,...)". Имя самоописательно,
// поэтому хранить рядом с ним нечего — резолверу КРС хватает строки, а список
// собранных пользователем обёрток это просто vector<string> в SystemRecord.
//
// Макрошаг h делится на K независимых прогонов опорной схемы: стадия k идёт
// из ОДНОГО И ТОГО ЖЕ состояния n_k подшагами h/n_k. Результаты складываются
// с весами alpha_k, гасящими K-1 первых членов разложения ошибки.
constexpr int kExtrMinStages   = 2;
constexpr int kExtrMaxStages   = 6;
constexpr int kExtrMaxSubsteps = 1024;

struct ExtrapolationSpec {
    std::string      base;   // имя опорной схемы (built-in или кастомная КРС)
    std::vector<int> n;      // подшаги на стадию; строго возрастает
};

// Собирает имя обёртки. Обратная к parse_extrapolation_name.
std::string make_extrapolation_name(const std::string& base, const std::vector<int>& n);

// Разбирает "Extr(RK4|1,2,4)". false, если имя не экстраполяционное ИЛИ
// нарушены ограничения (число стадий, монотонность n, диапазон). err, если
// передан, получает причину — она же показывается в конструкторе схемы.
// Вложенность запрещена: база не может сама быть "Extr(...)".
bool parse_extrapolation_name(const std::string& name, ExtrapolationSpec* out,
                              std::string* err = nullptr);

// Порядок результата: p + K - 1 в общем случае, p + 2*(K-1) для симметричной
// опорной схемы (у неё разложение идёт через степень, и каждая стадия
// снимает сразу два члена).
int extrapolation_order(int stages, int p, bool symmetric);

// Веса alpha_k. Выводятся из условий Sum(alpha) = 1 и Sum(alpha_k * u_k^j) = 0,
// где u_k = 1/n_k (общий случай) либо 1/n_k^2 (симметричный):
//   alpha_k ~ n_k^p * prod_{m != k} 1/(u_k - u_m),  затем нормировка суммы в 1.
// Неверные p / symmetric веса не портят — они лишь гасят не те члены, и
// порядок падает до исходного, а не даёт мусор.
std::vector<double> extrapolation_weights(const std::vector<int>& n, int p, bool symmetric);

// Оборачивает ГОТОВОЕ тело шага (base_body — то, что вернул codegen_scheme
// либо тело кастомной КРС) в K стадий. Тело вставляется дословно и ровно один
// раз — внутрь локальной лямбды, параметр которой назван h и затеняет
// макрошаг. Поэтому обёртке не нужно понимать содержимое базы: любое "0.5 * h"
// внутри само становится подшагом.
std::string wrap_extrapolation(const std::string& base_body, int N,
                               const std::vector<int>& n, int p, bool symmetric,
                               const std::string& base_name);

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

    // То же самое по тому же байткоду, но в комплексной арифметике — нужен
    // Complex Implicit Euler, у которого и состояние, и шаг комплексные.
    // Параметры a[] остаются вещественными.
    void eval_jacobian_complex(const ucmplx* Z, const double* a, ucmplx* J) const;

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