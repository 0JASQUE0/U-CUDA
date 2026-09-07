#include "integrator.h"
#include <cmath>

IntScheme int_scheme_from_string(const std::string& s) {
    if (s == "Euler-Cromer")      return IntScheme::EulerCromer;
    if (s == "Explicit Midpoint") return IntScheme::ExplicitMidpoint;
    if (s == "RK4")               return IntScheme::RK4;
    if (s == "DOPRI78")           return IntScheme::DOPRI78;
    if (s == "CD")                return IntScheme::CD;
    if (s == "Complex CD")        return IntScheme::ComplexCD;
    if (s == "Complex CD4")       return IntScheme::ComplexCD4;
    return IntScheme::Euler;
}

// DOPRI78 коэффициенты, общие с codegen::scheme_dopri78.
static const double DOPRI_M[13][12] = {
    {0,0,0,0,0,0,0,0,0,0,0,0},
    {0.05555555555556,0,0,0,0,0,0,0,0,0,0,0},
    {0.02083333333333,0.0625,0,0,0,0,0,0,0,0,0,0},
    {0.03125,0,0.09375,0,0,0,0,0,0,0,0,0},
    {0.3125,0,-1.171875,1.171875,0,0,0,0,0,0,0,0},
    {0.0375,0,0,0.1875,0.15,0,0,0,0,0,0,0},
    {0.04791013711111,0,0,0.1122487127778,-0.02550567377778,0.01284682388889,0,0,0,0,0,0},
    {0.01691798978729,0,0,0.387848278486,0.0359773698515,0.1969702142157,-0.1727138523405,0,0,0,0,0},
    {0.06909575335919,0,0,-0.6342479767289,-0.1611975752246,0.1386503094588,0.9409286140358,0.2116363264819,0,0,0,0},
    {0.183556996839,0,0,-2.468768084316,-0.2912868878163,-0.02647302023312,2.847838764193,0.2813873314699,0.1237448998633,0,0,0},
    {-1.215424817396,0,0,16.67260866595,0.9157418284168,-6.056605804357,-16.00357359416,14.8493030863,-13.37157573529,5.13418264818,0,0},
    {0.2588609164383,0,0,-4.774485785489,-0.435093013777,-3.049483332072,5.577920039936,6.155831589861,-5.062104586737,2.193926173181,0.1346279986593,0},
    {0.8224275996265,0,0,-11.65867325728,-0.7576221166909,0.7139735881596,12.07577498689,-2.12765911392,1.990166207049,-0.234286471544,0.1758985777079,0}
};
static const double DOPRI_B[2][13] = {
    {0.04174749114153,0,0,0,0,-0.05545232861124,0.2393128072012,0.7035106694034,-0.7597596138145,0.6605630309223,0.1581874825101,-0.2381095387529,0.25},
    {0.02955321367635,0,0,0,0,-0.8286062764878,0.3112409000511,2.4673451906,-2.546941651842,1.443548583677,0.07941559588113,0.04444444444444,0}
};

namespace {

// Один шаг каждой схемы. X на входе — текущее состояние, на выходе — следующее.
// ev.eval(X, a, deriv) даёт производные. Буферы передаются снаружи (без аллокаций).

void step_euler(const SystemEvaluator& ev, double* X, const double* a, double h,
                int n, double* k1) {
    ev.eval(X, a, k1);
    for (int i = 0; i < n; ++i) X[i] += h * k1[i];
}

void step_euler_cromer(const SystemEvaluator& ev, double* X, const double* a, double h,
                       int n, double* k1) {
    // обновляем по очереди, используя уже обновлённые компоненты
    for (int i = 0; i < n; ++i) {
        ev.eval(X, a, k1);      // переоценка с учётом обновлённых X[<i]
        X[i] += h * k1[i];
    }
}

void step_midpoint(const SystemEvaluator& ev, double* X, const double* a, double h,
                   int n, double* k1, double* tmp) {
    ev.eval(X, a, k1);
    for (int i = 0; i < n; ++i) tmp[i] = X[i] + 0.5 * h * k1[i];
    ev.eval(tmp, a, k1);
    for (int i = 0; i < n; ++i) X[i] += h * k1[i];
}

void step_rk4(const SystemEvaluator& ev, double* X, const double* a, double h,
              int n, double* k1, double* k2, double* k3, double* k4, double* tmp) {
    ev.eval(X, a, k1);
    for (int i = 0; i < n; ++i) tmp[i] = X[i] + 0.5 * h * k1[i];
    ev.eval(tmp, a, k2);
    for (int i = 0; i < n; ++i) tmp[i] = X[i] + 0.5 * h * k2[i];
    ev.eval(tmp, a, k3);
    for (int i = 0; i < n; ++i) tmp[i] = X[i] + h * k3[i];
    ev.eval(tmp, a, k4);
    for (int i = 0; i < n; ++i)
        X[i] += h * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6.0;
}

// DOPRI78 — 13-стадийный (см. scheme_dopri78). Использует 8-й порядок (B[0]);
// для CPU-портрета нам нужен только y, 7-й порядок (z) не считаем.
// Буферы: kbuf — 13*n, X1 — n, X2 — n.
void step_dopri78(const SystemEvaluator& ev, double* X, const double* a, double h,
                  int n, double* kbuf, double* X1, double* X2) {
    auto k = [&](int stage, int comp) -> double& { return kbuf[stage * n + comp]; };
    for (int i = 0; i < n; ++i) X1[i] = X[i];
    for (int stage = 0; stage < 13; ++stage) {
        // evaluate RHS at X1 → k[stage][*]
        double deriv[32];   // amountOfX cap (см. kMaxAmountOfX в engine = 32)
        ev.eval(X1, a, deriv);
        for (int i = 0; i < n; ++i) k(stage, i) = deriv[i];
        if (stage != 12) {
            for (int l = 0; l < n; ++l) X2[l] = 0;
            for (int j = 0; j < stage + 1; ++j)
                for (int l = 0; l < n; ++l)
                    X2[l] += DOPRI_M[stage + 1][j] * k(j, l);
            for (int l = 0; l < n; ++l)
                X1[l] = X[l] + h * X2[l];
        }
    }
    // 8-й порядок → X
    for (int l = 0; l < n; ++l) X2[l] = 0;
    for (int stage = 0; stage < 13; ++stage)
        for (int l = 0; l < n; ++l)
            X2[l] += DOPRI_B[0][stage] * k(stage, l);
    for (int l = 0; l < n; ++l) X[l] += h * X2[l];
}

// CD (Composition D-method): h1 = h*a[0], h2 = h*(1-a[0]).
// Полу-шаг 1 (явный, прямой порядок): для каждой i обновляем X[i] += h1*f_i(X).
// Полу-шаг 2 (неявный, обратный порядок): для каждой i (от n-1 к 0)
// 4 простые итерации: X[i] = saved + h2 * f_i(X). Это упрощение GPU-кодгена,
// где для линейной по var компоненты есть аналитическое решение; на CPU мы
// единообразно используем итерации (упрощает код, точность достаточная).
void step_cd(const SystemEvaluator& ev, double* X, const double* a, double h,
             int n, double* k1) {
    const double s = a[0];
    const double h1 = h * s;
    const double h2 = h * (1.0 - s);

    // Φ_h1: явный полушаг, прямой порядок (как Euler-Cromer).
    for (int i = 0; i < n; ++i) {
        ev.eval(X, a, k1);
        X[i] += h1 * k1[i];
    }
    // Φ*_h2: неявный полушаг, обратный порядок, 4 итерации.
    for (int i = n - 1; i >= 0; --i) {
        double saved = X[i];
        for (int it = 0; it < 4; ++it) {
            ev.eval(X, a, k1);
            X[i] = saved + h2 * k1[i];
        }
    }
}

// Complex CD: та же композиция, что step_cd, но полушаги комплексные —
// h1 = s*h + i*h*sqrt(3)/6, h2 = (1-s)*h - i*h*sqrt(3)/6, s = a[0] (тот же слот
// симметрии, что у CD; дефолт 0.5). h1 + h2 = h при любом s, при s = 1/2
// полушаги сопряжены. Шаг целиком идёт над локальной комплексной копией Z
// состояния; в X возвращается только Re. Мнимая часть на следующий шаг НЕ
// переносится: она живёт внутри одного шага. Порядок при s = 1/2 остаётся
// вторым, как у CD: h²-член композиции несёт множитель g1² - g2² = i*sqrt(3)/3
// (чисто мнимый — Re его срезает), а h³-член несёт g1*g2 = 1/3, он
// вещественный и выживает. Выигрыш — в константе ошибки: замер против эталона
// RK4 (h = 1e-6) даёт на Лоренце ~2.4x, на маятнике с sin ~7x меньшую ошибку
// при том же шаге. При s != 1/2 в h²-члене появляется вещественная часть
// (2s - 1), Re её уже не срезает, и порядок падает до первого — ровно как у
// вещественного CD вне s = 1/2.
// Как и step_cd, неявный полушаг здесь без аналитической ветки: 4 простые
// итерации для каждой переменной (GPU-кодген для линейных по var компонент
// решает уравнение точно — расхождение то же, что и у вещественного CD).
constexpr double CCD_IMAG = 0.28867513459481288225;  // sqrt(3)/6, как в codegen.cpp

// Один проход CD в комплексной арифметике: явный полушаг h1 вперёд, неявный h2
// назад. Complex CD зовёт его один раз, Complex CD4 — дважды с сопряжёнными
// коэффициентами.
static void complex_cd_pass(const SystemEvaluator& ev, const double* a, int n,
                            ucmplx* Z, ucmplx* k1, ucmplx h1, ucmplx h2) {
    // Φ_h1: явный полушаг, прямой порядок (как Euler-Cromer).
    for (int i = 0; i < n; ++i) {
        ev.eval_complex(Z, a, k1);
        Z[i] = Z[i] + h1 * k1[i];
    }
    // Φ*_h2: неявный полушаг, обратный порядок, 4 итерации.
    for (int i = n - 1; i >= 0; --i) {
        const ucmplx saved = Z[i];
        for (int it = 0; it < 4; ++it) {
            ev.eval_complex(Z, a, k1);
            Z[i] = saved + h2 * k1[i];
        }
    }
}

void step_complex_cd(const SystemEvaluator& ev, const double* a, double h, int n,
                     double* X, ucmplx* Z, ucmplx* k1) {
    const double s = a[0];
    for (int i = 0; i < n; ++i) Z[i] = ucmplx(X[i], 0.0);
    complex_cd_pass(ev, a, n, Z, k1,
                    ucmplx(s * h, h * CCD_IMAG), ucmplx((1.0 - s) * h, -h * CCD_IMAG));
    for (int i = 0; i < n; ++i) X[i] = Z[i].re;
}

// Complex CD4: два прохода того же CD, но комплексные коэффициенты вынесены на
// уровень выше — первый проход идёт с шагом gamma*h, второй с conj(gamma)*h,
// gamma = 1/2 + i*sqrt(3)/6, а s = a[0] делит уже СВОЙ комплексный шаг внутри
// прохода. При s = 1/2 внутренний CD самосопряжён (только нечётные степени в
// разложении), условия alpha+beta = 1 и alpha^3+beta^3 = 0 гасят h³, а h⁴ по
// сопряжённой симметрии мнимый и уходит с Re — глобальный порядок 4. Замерено
// 4.00 на Лоренце (T=2) и Рёсслере (T=10) против эталона DOPRI78 h=1e-3.
// Re берётся ОДИН раз в конце: мнимая часть переносится между проходами (её
// вклад O(h³), порядка это не меняет, но и обнулять её посреди шага незачем).
void step_complex_cd4(const SystemEvaluator& ev, const double* a, double h, int n,
                      double* X, ucmplx* Z, ucmplx* k1) {
    const double s = a[0];
    const ucmplx g (0.5 * h,  h * CCD_IMAG);
    const ucmplx gc(0.5 * h, -h * CCD_IMAG);
    for (int i = 0; i < n; ++i) Z[i] = ucmplx(X[i], 0.0);
    complex_cd_pass(ev, a, n, Z, k1, g  * s, g  * (1.0 - s));
    complex_cd_pass(ev, a, n, Z, k1, gc * s, gc * (1.0 - s));
    for (int i = 0; i < n; ++i) X[i] = Z[i].re;
}

// Общий прогон траектории: transient + запись total точек с проверкой на
// nan/inf. do_step — любой callable, делающий один шаг по X. Шаблон, а не
// std::function: встроенный путь не должен получить косвенный вызов на
// каждом шаге.
// State — шаблонный параметр: встроенные схемы идут через SystemEvaluator в
// double, custom КРС — в numb (тип тела схемы). Запись наружу в обоих случаях
// расширяется до double.
template <class StepOnce, class State>
bool run_trajectory(StepOnce do_step, const State* X, int n,
                    int total, int skip,
                    std::vector<std::vector<double>>& out) {
    // transient
    for (int s = 0; s < skip; ++s) do_step();

    // основной цикл — без аллокаций на каждый шаг
    out.assign(total, std::vector<double>(n));
    for (int t = 0; t < total; ++t) {
        for (int k = 0; k < n; ++k) {
            double v = X[k];
            if (std::isnan(v) || std::isinf(v)) { out.resize(t); return false; }
            out[t][k] = v;
        }
        do_step();
    }
    return true;
}

} // namespace

bool computePhasePortraitCPU(
    const SystemEvaluator& ev,
    IntScheme scheme,
    const double* ic, int dim,
    const double* a, int amountOfValues,
    double h, int total, int skip,
    std::vector<std::vector<double>>& out)
{
    (void)amountOfValues;
    int n = dim;
    std::vector<double> X(n);
    for (int i = 0; i < n; ++i) X[i] = ic[i];

    // переиспользуемые буферы (без аллокаций в цикле)
    std::vector<double> k1(n), k2(n), k3(n), k4(n), tmp(n);
    std::vector<double> kbuf(13 * n), X1(n), X2(n);  // для DOPRI78
    // Комплексные буферы нужны только Complex CD — для остальных схем это два
    // пустых вектора, без аллокаций.
    std::vector<ucmplx> Zc, Kc;
    if (scheme == IntScheme::ComplexCD || scheme == IntScheme::ComplexCD4) { Zc.resize(n); Kc.resize(n); }

    auto do_step = [&]() {
        switch (scheme) {
        case IntScheme::Euler:            step_euler(ev, X.data(), a, h, n, k1.data()); break;
        case IntScheme::EulerCromer:      step_euler_cromer(ev, X.data(), a, h, n, k1.data()); break;
        case IntScheme::ExplicitMidpoint: step_midpoint(ev, X.data(), a, h, n, k1.data(), tmp.data()); break;
        case IntScheme::RK4:              step_rk4(ev, X.data(), a, h, n, k1.data(), k2.data(), k3.data(), k4.data(), tmp.data()); break;
        case IntScheme::DOPRI78:          step_dopri78(ev, X.data(), a, h, n, kbuf.data(), X1.data(), X2.data()); break;
        case IntScheme::CD:               step_cd(ev, X.data(), a, h, n, k1.data()); break;
        case IntScheme::ComplexCD:        step_complex_cd(ev, a, h, n, X.data(), Zc.data(), Kc.data()); break;
        case IntScheme::ComplexCD4:       step_complex_cd4(ev, a, h, n, X.data(), Zc.data(), Kc.data()); break;
        }
    };

    return run_trajectory(do_step, X.data(), n, total, skip, out);
}

bool computePhasePortraitCPU_custom(
    CustomStepFn step,
    const double* ic, int dim,
    const double* a, int amountOfValues,
    double h, int total, int skip,
    std::vector<std::vector<double>>& out)
{
    if (!step) return false;
    const int n = dim;
    // Состояние и параметры — в numb: тело КРС скомпилировано под этот тип, и
    // считать вокруг него в double значило бы гонять GPU и CPU в разной
    // точности. Наружу (out) значения расширяются обратно до double.
    std::vector<numb> X((size_t)n);
    for (int i = 0; i < n; ++i) X[(size_t)i] = (numb)ic[i];
    std::vector<numb> A((size_t)(amountOfValues > 0 ? amountOfValues : 1), (numb)0);
    for (int i = 0; i < amountOfValues; ++i) A[(size_t)i] = (numb)a[i];

    // Никаких промежуточных буферов: стадии (если они есть) живут внутри тела
    // КРС — ровно как на GPU, где calculateDiscreteModel держит их в локальных
    // массивах.
    numb* Xp = X.data();
    const numb hn = (numb)h;
    auto do_step = [&]() { step(Xp, A.data(), hn); };

    return run_trajectory(do_step, Xp, n, total, skip, out);
}
