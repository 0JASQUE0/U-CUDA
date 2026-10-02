#pragma once
#include <map>
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
// RK45 — Dormand-Prince 5(4)7M, решатель ode45 / scipy RK45: 6 стадий на шаг,
// 7-я (FSAL, f в новой точке) нужна только оценке ошибки. DOP853 — метод
// Хайрера 8(5,3), 12 стадий, со встроенными оценщиками 5-го и 3-го порядков и
// плотным выходом 7-го порядка. Обе схемы с постоянным шагом используют только
// старшее решение; оценщики — задел под адаптивный шаг.
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
// ComplexCD4S3 / ComplexCD4S4 — "CCD4 (o4s3)" и "CCD4 (o4s4)", СИММЕТРИЧНЫЕ
// схемы 4-го порядка из того же симметричного CD (s = 1/2). Complex CD4
// самосопряжённой не является и быть не может: палиндром из двух стадий
// требует g1 = g2, а тогда условия sum g = 1 и sum g^3 = 0 несовместны.
// Минимум для симметричной комплексной композиции порядка 4 — три стадии:
//   o4s3: (alpha, 1 - 2*alpha, alpha), alpha = 1/(2 - 2^(1/3)*exp(2*pi*i/3));
//   o4s4: (gamma/2, conj/2, conj/2, gamma/2) — Complex CD4, симметризованная
//         композицией с собственной сопряжённой на половинном шаге.
// Подробности, замеры и цена — у CdKind в codegen.cpp.
// ComplexCD4SS01 — "CCD4 (s-split)_{0-1}", двойственная к Complex CD4: внешние
// шаги вещественные (h/2 и h/2), а комплексно само разбиение внутри CD —
// s1 = (1 - i)/3, s2 = 1 - s1 = (2 + i)/3. Полушаги (1-i)h/6, (2+i)h/6 |
// (2+i)h/6, (1-i)h/6, Re один раз в конце, глобальный порядок 4. Коэффициенты
// фиксированы, a[0] НЕ читает. Индекс {0-1}: в проходе сначала явная (0),
// затем неявная (1) половина, как во всех CD-схемах проекта.
// ComplexCD4SS10 — "CCD4 (s-split)_{1-0}": те же полушаги, но в каждом
// проходе первой идёт неявная половина. Тоже порядок 4.
// Подробности — у CdKind::Cx4ss01 / Cx4ss10 в codegen.cpp.
// CD10, ComplexCD10, ComplexCD4_10, ComplexCD4S3_10, ComplexCD4S4_10 —
// "CD_{1-0}", "Complex CD_{1-0}", "Complex CD4_{1-0}", "CCD4 (o4s3)_{1-0}",
// "CCD4 (o4s4)_{1-0}": те же схемы, но в каждом проходе CD первой идёт
// неявная половина (шаг h1 = s*h, обратный порядок), затем явная (h2, прямой).
// Это та же композиция над сопряжённым базовым методом, порядки и симметрия
// те же, что у исходных (исходные — это {0-1}, их имена не менялись, чтобы не
// ломать сохранённые библиотеки и сессии).
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
// GBS, GBS24 ... GBS24681012 — семейство Грэгга-Булирша-Штёра. Стадия k
// (k = 1..K) — ОДНА цепочка модифицированной средней точки Грэгга из n = 2k
// подшагов h/n: явный Эйлер на старте, leapfrog z_{m+1} = z_{m-1} + 2(h/n)f(z_m),
// сглаживание (z_{n-1} + z_n + (h/n)f(z_n))/2. Стадии стартуют из одного X и
// складываются с весами Ричардсона по 1/n^2. GBS — сама опорная схема (K = 1,
// n = 2, порядок 2), GBS24 ... — экстраполяция по n = 2,4,...,2K, порядок 2K.
// Отличие от Extr(GBS|1,2,...): там стадия — повтор ОДНОШАГОВОЙ базы, цепочка
// leapfrog рвётся после каждой пары подшагов, и порядок 2-4-6 падает до 5
// (замерено на Рёсслере). Здесь цепочка на стадию одна, разложение по h^2
// честное, порядок 2K (замерено 4.1 и 6.1). См. scheme_gbs в codegen.cpp.
// Map is not an integration scheme but the right-hand side of a discrete map
// x_{n+1} = f(x_n) itself. The only emitter that never uses h.
enum class Scheme { Euler, EulerCromer, ExplicitMidpoint, RK4, DOPRI78, CD, ComplexCD, ComplexCD4,
                    ComplexCD4S3, ComplexCD4S4, ComplexCD4SS01, ComplexCD4SS10,
                    CD10, ComplexCD10, ComplexCD4_10, ComplexCD4S3_10, ComplexCD4S4_10,
                    ImplicitEuler, ImplicitMidpoint, SEMP, SIMP, D, ComplexIEuler,
                    GBS, GBS24, GBS246, GBS2468, GBS246810, GBS24681012, DOPRI78Legacy,
                    RK45, DOP853, Map };

// Число стадий K схемы ГБШ (1 у опорной GBS, 2..6 у экстраполяторов), 0 — не ГБШ.
int gbs_stage_count(Scheme sch);
// Подшаги стадий ГБШ: {2, 4, ..., 2K}.
std::vector<int> gbs_substeps(int K);

// Таблица Бутчера DOPRI78 — RK8(7)13M (Prince & Dormand, 1981). Один источник
// для кодогена (scheme_dopri78) и CPU-интегратора (step_dopri78).
// Дроби из статьи — сами приближения: условия порядка они выполняют лишь до
// ~1e-18, и в dd вкладки Order ошибка упиралась в полку ~1e-17. Поэтому
// коэффициенты уточнены Левенбергом-Марквардтом на всех 200 условиях порядка
// <= 8 (старт — дроби статьи, структура нулей та же, поправки ~1e-18, то есть
// в пределах опубликованной точности) до невязки 1.8e-39 — дальше решение
// лежит на вырожденном многообразии и Ньютон стоит, но для dd (eps 5e-32) это
// с запасом: полка ~1e-38 вместо ~1e-17. Хранятся суммой до
// четырёх неперекрывающихся double: c[0] — ближайший double, c[1..3] — остатки.
// double берёт c[0], dd — c[0..1], qd — все четыре.
// kDopri78A[i][j] = a_{i+1, j+1}; kDopri78B[0] — веса 8-го порядка, [1] — 7-го.
struct MultiDoubleCoef { double c[4]; };

// Веса оценки ошибки встроенной пары w = b - b^ (b — S весов старшего решения, b^ — lc
// весов оценочного, lc >= S; недостающие b — нули), с точностью double-double (two_sum):
// оценка E = h*sum(w_j k_j) считается одной суммой, без вычитания близких решений.
// Общая для тела emb (codegen) и CPU-интегратора — веса у них совпадают побитово.
void adaptive_err_weights(const MultiDoubleCoef* b, int S, const MultiDoubleCoef* bh, int lc,
                          MultiDoubleCoef* w);
extern const MultiDoubleCoef kDopri78A[13][12];
extern const MultiDoubleCoef kDopri78B[2][13];
// "DOPRI78 (legacy)" — дроби статьи Prince & Dormand (1981) как есть, без
// уточнения (тоже суммой 4 double, чтобы dd нёс всю точность дроби). Условия
// порядка они выполняют до ~1e-18: в double метод от уточнённого не отличить,
// в dd ошибка упирается в полку ~1e-17. Оставлена для сравнения.
extern const MultiDoubleCoef kDopri78LegacyA[13][12];
extern const MultiDoubleCoef kDopri78LegacyB[2][13];

// RK45 — Dormand-Prince 5(4)7M. Коэффициенты — точные дроби (взяты из
// исходника scipy RK45 и проверены точной арифметикой: b — порядок 5, b^ —
// порядок 4), суммой до четырёх double, как у DOPRI78.
// kRK45A[i][j] = a_{i,j}, строка 6 = b (стадия FSAL, f(y_{n+1})).
// kRK45B[0] — b (5-й порядок, им идёт шаг), [1] — b^ (4-й, оценщик).
// kRK45P — плотный выход Шампайна 4-го порядка (матрица P scipy).
extern const MultiDoubleCoef kRK45A[7][6];
extern const MultiDoubleCoef kRK45B[2][7];
extern const MultiDoubleCoef kRK45P[7][4];

// DOP853 — метод Хайрера 8(5,3), 12 стадий. Старт — 30-значные литералы
// (scipy, dop853_coefficients.py), дальше та же процедура, что у DOPRI78:
// Ньютон с минимальной нормой поправки на всех 200 условиях порядка <= 8 плюс
// суммы строк a_ij = c_i с точными узлами (в c2..c5 входит sqrt(6)); структура
// нулей и дроби 1/27, 19/512, -9/512 сохранены. Невязка 3e-78 (была 6e-29),
// поправки ~5e-29 — в пределах опубликованной точности.
// kDop853A[i][j] = a_{i,j} для стадий 0..11; kDop853B — b (8-й порядок).
// kDop853E5 / kDop853E3 — b - b^ оценщиков 5-го и 3-го порядков по 13 стадиям
// (стадия 12 = f(y_{n+1})): веса b^(5) уточнены линейно до 4e-77, b^(3) —
// точные дроби 31/127, 12675/17272, 3/136.
// kDop853AExt — стадии 13..15 плотного выхода, kDop853D — его коэффициенты
// d4..d7; они влияют только на интерполированные значения и оставлены как
// опубликованы (~30 знаков).
extern const MultiDoubleCoef kDop853A[12][11];
extern const MultiDoubleCoef kDop853B[12];
extern const MultiDoubleCoef kDop853E5[13];
extern const MultiDoubleCoef kDop853E3[13];
extern const MultiDoubleCoef kDop853AExt[3][15];
extern const MultiDoubleCoef kDop853D[4][16];

// Генерирует тело шага схемы в виде C/CUDA-кода (строки вида
// "X[0] = X[0] + h * (...);"). Бросает std::runtime_error при ошибке разбора.
std::string codegen_scheme(const System& s, Scheme sch);

// ---- Адаптивный шаг ---------------------------------------------------------
// Схемы со встроенной оценкой ошибки (RK45, DOPRI78 обоих видов, DOP853) дают,
// кроме тела шага с постоянным h, четыре тела функций. Обёртку с сигнатурой
// печатает потребитель (шаблон ядра, модуль КРС, CPU), поэтому здесь — только
// тела, а сигнатуры зафиксированы так (N — число переменных):
//   rhs  (const numb* X, const numb* a, numb* F)            F = f(X)
//   emb  (const numb* X, const numb* F0, const numb* a, const numb h,
//         numb* Y, numb* E, numb* F1, numb* W)
//        Пробный шаг из X, где F0 = f(X) уже посчитана (FSAL / прошлый шаг):
//        Y — старшее решение (побитово то же, что даёт тело постоянного шага
//        при этом h), E — nlo оценок ошибки подряд (E[m*N + i] = Y_i - Yhat_m,i,
//        считаются по разности весов, а не вычитанием решений),
//        F1 = f(Y), W — стадии W[s*N + i] (нужны плотному выходу).
//   dprep(const numb* X, const numb* Y, const numb* F0, const numb* F1,
//         const numb* a, const numb h, numb* W, numb* D)
//        Плотный выход принятого шага [t, t+h] в D (dsize*N чисел). DOP853
//        досчитывает здесь три стадии, остальные вычислений f не делают.
//   deval(const numb* D, const numb th, numb* Yo)   y(t + th*h), th в [0, 1]
struct AdaptiveCode {
    std::string rhs, emb, dprep, deval;
    int p = 0;           // порядок метода
    int q = 0;           // порядок оценщика: показатель регулятора 1/(q+1)
    int nlo = 0;         // число оценок ошибки (DOP853: [0] — по 5-му, [1] — по 3-му порядку)
    int wstages = 0;     // сколько стадий пишет W (с плотным выходом)
    int dsize = 0;       // размер D в единицах N
    int emb_rhs = 0;     // новых вычислений f на одну попытку шага
    int dprep_rhs = 0;   // вычислений f на подготовку плотного выхода
    int dense_order = 0; // порядок плотного выхода
};
// Верхние границы для буферов драйвера (kernels/ucuda_adaptive.cuh).
constexpr int kAdaptMaxStages = 16;
constexpr int kAdaptMaxDense  = 8;
constexpr int kAdaptMaxLow    = 2;
bool scheme_supports_adaptive(Scheme sch);
// Бросает std::runtime_error для схемы без встроенной оценки.
AdaptiveCode codegen_adaptive(const System& s, Scheme sch);

// Адаптивный шаг экстраполяторов: вложенная пара из одних и тех же стадий. Y — по всем K
// стадиям (веса и порядок постоянного шага), младшее решение — по первым K-1 (свои веса,
// q = extrapolation_order(K-1)), E = sum (alpha_k - beta_k) T_k. F1 = f(Y) — FSAL и эрмитов
// плотный выход 3-го порядка. Нужно K >= 2. Принятый шаг с тем же h — побитово шаг
// постоянного Extr / GBS (см. codegen.cpp).
//   base_body — тело шага базы (как для wrap_extrapolation), base_rhs — вычислений f на
//   шаг базы, 0 — неизвестно (тогда счётчик f видит только F1).
AdaptiveCode codegen_adaptive_extrapolation(const System& s, const std::string& base_body,
                                            const std::vector<int>& n, int p, bool symmetric,
                                            const std::string& base_name, int base_rhs);
// GBS 2-4 ... 2-4-6-8-10-12 (K = gbs_stage_count). f(X) первой стадии берётся из F0.
AdaptiveCode codegen_adaptive_gbs(const System& s, int K);
// Вычислений f на шаг встроенной явной схемы постоянного шага; 0 — неизвестно (неявные, CD,
// комплексные: f там по компонентам или внутри Ньютона).
int builtin_scheme_rhs_per_step(const std::string& name);
// Схема, у которой есть адаптивный шаг, — по одному имени: RK-пары, GBS 2-4 ..., Extr(база|n)
// с K >= 2 (ExtrZ — пока нет). База Extr проверяется только при сборке кода.
bool adaptive_scheme_name_supported(const std::string& name);
// Порядок оценщика по имени, для значений регулятора по умолчанию (dopri5 / dop853):
// у Extr с кастомной базой неизвестен — 7.
int adaptive_scheme_q_guess(const std::string& name);

// Emits a human-readable C-code mirror of what the CPU integrator
// (integrator.cpp::step_*) actually computes. Now identical to codegen_scheme
// for every scheme: the CPU path evaluates the same AST, and for the
// diagonally-implicit ones (CD, Complex CD, SIMP, D) it solves each equation
// with the same analytic formula, via SystemEvaluator::solve_diag_implicit, falling
// back to iterations exactly where the GPU emits them. Kept as the single place
// the debug panel asks for the CPU form, and as the place to express a future
// divergence between the two paths.
std::string codegen_scheme_cpu_equivalent(const System& s, Scheme sch);

// Правые части системы в том виде, из которого codegen_scheme собирает шаг
// ЛЮБОЙ схемы: выражения над X[0..N-1] и a[1..M], по строке на переменную
// (у отображения — сразу x_{n+1}). Это тот же rhs_over, что внутри схем, а не
// отдельный путь, — превью в Library показывает ровно то, что уйдёт в ядро.
// std::runtime_error при ошибке разбора.
std::vector<std::string> codegen_rhs(const System& s);

// Вклад ОДНОГО ребра сети в правую часть узла-приёмника (вкладка Network).
// exprs — по выражению на переменную системы; пустая строка = в это уравнение
// связь не входит, строки для неё в выводе не будет.
//
// Имена в выражении:
//   <var>, <var>_i — состояние приёмника      -> self_state[k]
//   <var>_j        — состояние источника      -> nbr_state[k]
//   <par>, <par>_i — параметры приёмника      -> self_par[1+k]
//   <par>_j        — параметры источника      -> nbr_par[1+k]
//   K, w           — вес ребра                -> weight
// Отсутствие суффикса значит «приёмник»: K*(x_j - x) и K*(x_j - x_i) — одно и
// то же. Имя системы перекрывает K: если в системе есть параметр K, то K в
// выражении — это он, и вес доступен только как w.
//
// Разбор ВСЕГДА в обычном синтаксисе, даже если система задана LaTeX'ом: в
// LaTeX подчёркивание — это индекс, и x_j читалось бы не как имя.
// Результат — строки вида "C[0] += (...);". std::runtime_error при ошибке
// разбора и на неизвестном имени.
std::string codegen_coupling(const System& s,
                             const std::vector<std::string>& exprs,
                             const std::string& self_state = "Xi",
                             const std::string& nbr_state  = "Xj",
                             const std::string& self_par   = "ai",
                             const std::string& nbr_par    = "aj",
                             const std::string& weight     = "w",
                             const std::string& dst        = "C",
                             const std::string& indent     = "        ");

// Maps UI scheme name ("Euler" / "RK4" / "CD" / ...) to the Scheme enum.
// Unknown names fall back to Scheme::Euler.
Scheme scheme_from_name(const std::string& name);

// Паспортный порядок и симметричность встроенной схемы по её UI-имени.
// Единственный источник этих чисел в проекте: их читает и UI (группировка
// списка схем по порядку, вкладка Order), и резолвер КРС при сборке
// экстраполяционной обёртки — разъехаться копиям негде.
// Симметричной ("разложение ошибки только по чётным степеням h") помечены
// четыре схемы: Implicit Midpoint и CD как композиция Phi* ∘ Phi, плюс
// CCD4 (o4s3) и CCD4 (o4s4) как палиндромные композиции этого же CD.
// У всех, кроме Implicit Midpoint, это верно ТОЛЬКО при a[0] = 1/2 — проверить
// в кодогене нельзя, a[0] приходит в рантайме, поэтому ответственность на
// UI-предупреждении.
// Complex CD / Complex CD4 намеренно НЕ помечены: сопряжённые полушаги — это
// не самосопряжённость. У Complex CD4 замеренный дефект симметрии O(h^8)
// (у палиндромных — O(h^10)), и флаг оставлен снятым, чтобы обёртка
// экстраполяции не считала чётность разложения гарантированной.
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
constexpr int kExtrMaxStages   = 10;
constexpr int kExtrMaxSubsteps = 1024;

// ExtrZ — та же экстраполяция, но Re берётся ОДИН раз, на выходе взвешенной
// суммы: стадии идут в комплексной арифметике, и мнимая часть переносится
// между подшагами. Имя "ExtrZ(<база>|n1,n2,...)". База — только встроенная
// комплексная CD-схема (scheme_has_complex_core): у кастомной КРС ядро без
// перехода X -> Z и без Re не выделить.
// Паспорт в этом режиме: порядок базы p и ВСЕГДА symmetric = true. У любой
// комплексной CD-базы член n^-j разложения стадии по подшагам вещественен
// ровно при чётном j (нечётные несут нечётное число мнимых множителей), а
// мнимые снимает Re на выходе. Отсюда p + 2(K-1), замерено в 40 знаках на
// Лоренце: ExtrZ(1,2) / ExtrZ(1,2,3) дают 4 / 6 у Complex CD, 6 / 8 у
// Complex CD4 и у CCD4 (s-split)_{0-1}. Обычный Extr с Re в каждом шаге базы у
// тех же баз даёт 3 / 3, 6 / 7 и 5 / 5.
struct ExtrapolationSpec {
    std::string      base;   // имя опорной схемы (built-in или кастомная КРС)
    std::vector<int> n;      // подшаги на стадию; строго возрастает
    std::string      label;  // имя от пользователя; пусто = имени не давали
    bool re_at_output = false;   // ExtrZ: Re только на выходе, см. выше
};

// Собирает имя обёртки. Обратная к parse_extrapolation_name. Непустая label
// добавляет метку первым полем: "Extr(<метка>|<база>|n1,n2,...)";
// re_at_output меняет префикс на "ExtrZ(".
std::string make_extrapolation_name(const std::string& base, const std::vector<int>& n,
                                    const std::string& label = {},
                                    bool re_at_output = false);

// Симметричность базы, по которой считаются веса и порядок обёртки. Одна
// функция на резолвер КРС и на UI — иначе показанные веса разъехались бы с
// теми, что ушли в ядро.
inline bool extrapolation_symmetric(bool re_at_output, bool base_symmetric) {
    return re_at_output || base_symmetric;
}

// true — у встроенной схемы есть комплексное ядро для ExtrZ.
bool scheme_has_complex_core(const std::string& name);

// Ядро комплексной схемы: тот же шаг, но над ucmplx Z[N], объявленным СНАРУЖИ,
// без перехода X -> Z на входе и без Re на выходе. std::runtime_error для схемы
// без такого ядра.
std::string codegen_scheme_complex_core(const System& s, Scheme sch);

// Разбирает "Extr(RK4|1,2,4)" и "ExtrZ(...)". false, если имя не экстраполяционное ИЛИ
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

// Те же веса ТОЧНОЙ дробью: out[k] = "16/15", "-1/3", "1". Именно они уходят
// в КРС литералами, когда сходятся, поэтому UI показывает их, а не только
// double: "alpha = 1.0666666666666667" не даёт понять, что это 16/15.
// false — точное представление не сошлось (переполнение или числитель со
// знаменателем вне 2^53); тогда остаются только double выше.
bool extrapolation_weights_exact(const std::vector<int>& n, int p, bool symmetric,
                                 std::vector<std::string>* out);

// Оборачивает ГОТОВОЕ тело шага (base_body — то, что вернул codegen_scheme
// либо тело кастомной КРС) в K стадий. Тело вставляется дословно в каждую
// стадию, блоком со своим "const numb h = <подшаг>", который затеняет
// макрошаг. Поэтому обёртке не нужно понимать содержимое базы: любое "0.5 * h"
// внутри само становится подшагом. Тело с return/goto остаётся в локальной
// лямбде (см. wrapper_can_inline в codegen.cpp).
std::string wrap_extrapolation(const std::string& base_body, int N,
                               const std::vector<int>& n, int p, bool symmetric,
                               const std::string& base_name);

// То же для ExtrZ: core_body — codegen_scheme_complex_core базы. Стадии,
// начальное состояние и накопитель — ucmplx, Re берётся после суммы.
// Веса — симметричные (extrapolation_symmetric).
std::string wrap_extrapolation_complex(const std::string& core_body, int N,
                                       const std::vector<int>& n, int p,
                                       const std::string& base_name);

// --- Composition of a base scheme with itself -------------------------------
//
// The wrapper lives in the name: "Comp(<base>|g1,g2,...,gK)". Stages run
// SEQUENTIALLY, each from the previous result, with step gk*h. That is the one
// structural difference from Extr, where K independent runs start from the same
// state and are then summed with weights -- so this wrapper needs neither the
// saved state nor the accumulator.
//
// A coefficient is any expression over the system PARAMETERS and math
// constants: "1.3512", "g1", "1 - 2*g1". Names resolve to a[k] through the same
// NameMap the right-hand sides use, so a coefficient declared as a parameter is
// swept by the Order tab -- that is what makes a p(g1, g2) map possible without
// a hand-written KRS. State variables are rejected: the step size may not
// depend on X.
//
// The order is deliberately NOT computed here. Over a symmetric base of order p
// a palindromic composition reaches p+2 exactly when Sum(g) = 1 and
// Sum(g^3) = 0, but codegen can only check that for constant coefficients
// (composition_sums); a symbolic one is known at run time only, and measuring
// it is the Order tab's job.
constexpr int kCompMinStages = 2;
// 40 стадий: столько нужно опубликованным композициям высокого порядка
// (S17o8 — 17 стадий, порядок 10 и выше — 31 и 35). Потолок упирается
// только в ширину таблицы коэффициентов в UI, генератор не ограничен.
constexpr int kCompMaxStages = 40;

struct CompositionSpec {
    std::string              base;    // base scheme name (built-in or custom KRS)
    std::vector<std::string> gammas;  // step coefficient per stage
    std::string              label;   // имя от пользователя; пусто = имени не давали
};

// Builds the wrapper name. Inverse of parse_composition_name. Непустая label
// добавляет метку первым полем: "Comp(<метка>|<база>|g1,...)".
std::string make_composition_name(const std::string& base,
                                  const std::vector<std::string>& gammas,
                                  const std::string& label = {});

// Parses "Comp(CD|g1,1-2*g1,g1)". false when the name is not a composition OR
// the limits are broken (stage count, empty or unparsable coefficient); err, if
// given, receives the reason. Nesting is refused: the base cannot itself be a
// wrapper. The coefficient SUM is not checked -- sweeping g moves it off 1 on
// purpose, and rejecting that would rule out the p(g1, g2) map.
bool parse_composition_name(const std::string& name, CompositionSpec* out,
                            std::string* err = nullptr);

// Разбирает СПИСОК коэффициентов композиции из ТЕКСТА ФАЙЛА — по одному
// коэффициенту на строку, как их печатают таблицы опубликованных методов:
//
//   0.127136927734878585
//   0.561702537988802652
//   ...
//
// Допускаются CRLF, табуляции и пробелы по краям, пустые строки, комментарии
// ('#', '%', '//'), нумерующий первый столбец ("3<tab>-0.3825...") и несколько
// коэффициентов в одной строке через запятую. Коэффициент берётся ТЕКСТОМ как
// есть и НЕ округляется: 18 значащих цифр published-таблицы — это ровно то,
// чем композиция высокого порядка держит свой порядок. По той же причине
// символьный коэффициент ("g1", "1-2*g1") в файле тоже допустим.
// false — коэффициентов меньше kCompMinStages, больше kCompMaxStages либо один
// из них не разбирается; err получает причину.
bool parse_composition_coeff_file(const std::string& text,
                                  std::vector<std::string>* out,
                                  std::string* err = nullptr);

// Sum(g) and Sum(g^3) -- the two order conditions of a palindromic composition
// over a symmetric base. false when any coefficient is symbolic, in which case
// neither sum is knowable at codegen time and the outputs are left alone.
bool composition_sums(const CompositionSpec& spec, double* sum, double* cube_sum);

// Числовые значения коэффициентов при ЗАДАННЫХ значениях параметров. Нужно
// потому, что символьный коэффициент ("g1", "1-2*g1") — это половина ответа:
// сам шаг стадии известен только вместе с param_values (имя -> текст значения,
// как в SystemRecord::param_values). Значение параметра тоже разбирается как
// выражение, так что "8/3" работает; ссылка одного параметра на другой — нет.
// out получает по элементу на стадию, нераскрывшаяся стадия получает NaN, а
// err — причину ПЕРВОЙ такой. Возвращает false, если не посчиталась хоть одна.
bool composition_gamma_values(const CompositionSpec& spec,
                              const std::map<std::string, std::string>& param_values,
                              std::vector<double>* out, std::string* err = nullptr);

// Wraps a READY step body in K sequential stages. The body is inserted verbatim
// into every stage, in a block with its own "const numb h" that shadows the
// macro step -- same trick as wrap_extrapolation, and the same reason it works
// on an opaque custom KRS (a body with return/goto stays in a lambda).
// sys is needed to resolve coefficient names into a[k]; p/symmetric only feed
// the header comment. Throws std::runtime_error on an unresolvable coefficient.
std::string wrap_composition(const std::string& base_body, const System& sys,
                             const std::vector<std::string>& gammas,
                             int p, bool symmetric, const std::string& base_name);

// --- Имя обёртки, данное пользователем --------------------------------------
// Метка живёт ВНУТРИ имени ("Comp(S17o8|Short CD|0.127,...)"), а не рядом с
// ним: имя обёртки — это её единственное удостоверение, оно уезжает в сессии,
// в JSON библиотеки и в резолвер КРС, и отдельная таблица "имя -> подпись"
// потребовала бы протащить себя в каждую из них. Метка при этом чисто
// косметическая: резолвер читает только базу и коэффициенты, поэтому старое
// имя без метки и новое с меткой дают ОДИН И ТОТ ЖЕ шаг.
//
// Плата за это — переименование меняет удостоверение: сессия, сохранённая со
// старым именем, продолжит считать ровно то же самое, но в её комбо останется
// старая подпись, пока схему не выберут заново.
bool wrapper_label_ok(const std::string& label);

// Готовит пользовательский ввод к тому, чтобы стать меткой: выбрасывает
// символы, из которых собрано само имя ('|', ',', '(', ')'), и края-пробелы.
std::string wrapper_sanitize_label(const std::string& label);

// Метка имени ("" — её нет или имя не обёрточное).
std::string wrapper_label(const std::string& name);

// Имя БЕЗ метки. Им сравнивают схемы по существу: две обёртки с разными
// подписями и одной и той же базой с коэффициентами — это одна схема.
std::string wrapper_canonical_name(const std::string& name);

// Имя с ДРУГОЙ меткой (пустая label метку снимает). Не обёрточное имя
// возвращается как есть.
std::string wrapper_relabel(const std::string& name, const std::string& label);

// Short form of a wrapper name, FOR DISPLAY ONLY: a user label, when the name
// carries one, stands in for the whole description -- that is what it is for.
// Without a label, constant coefficients are rounded to `digits` significant
// figures, and symbolic ones are printed as typed.
// "Comp(CD|1.3512071919596578,-1.7024143839193155,1.3512071919596578)" becomes
// "Comp(CD|1.3512,-1.7024,1.3512)", which is the difference between a readable
// combo row and one that needs a scrollbar at five stages.
// Anything that is neither labelled nor a composition (a built-in, a custom
// KRS, an unlabelled "Extr(...)" whose substeps are short integers anyway)
// comes back unchanged.
// NEVER use the result as a key: the full name is the scheme's identity, and
// two different methods can round to the same short form.
std::string wrapper_display_name(const std::string& name, int digits = 5);

// Порядок списка обёрток: по типу (Extr, ExtrZ, Comp), затем по опорнику —
// встроенные в порядке паспортной таблицы (builtin_scheme_traits), кастомные
// КРС после них по алфавиту, — затем по числу стадий и по самим подшагам /
// коэффициентам. Метку ключ не читает: переименование строку не двигает.
// Неразбираемое имя уходит в конец. Сортировка устойчивая.
void sort_wrapper_schemes(std::vector<std::string>& names);

// Заголовок группы, в которую sort_wrapper_schemes ставит имя:
// "Extr over RK4", "ExtrZ over Complex CD4", "Comp over CD"; "" — не обёртка.
std::string wrapper_group_title(const std::string& name);

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