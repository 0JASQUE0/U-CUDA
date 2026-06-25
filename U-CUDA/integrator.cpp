#include "integrator.h"
#include <cmath>

IntScheme int_scheme_from_string(const std::string& s) {
    if (s == "Euler-Cromer")      return IntScheme::EulerCromer;
    if (s == "Explicit Midpoint") return IntScheme::ExplicitMidpoint;
    if (s == "RK4")               return IntScheme::RK4;
    if (s == "DOPRI78")           return IntScheme::DOPRI78;
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

    auto do_step = [&]() {
        switch (scheme) {
        case IntScheme::Euler:            step_euler(ev, X.data(), a, h, n, k1.data()); break;
        case IntScheme::EulerCromer:      step_euler_cromer(ev, X.data(), a, h, n, k1.data()); break;
        case IntScheme::ExplicitMidpoint: step_midpoint(ev, X.data(), a, h, n, k1.data(), tmp.data()); break;
        case IntScheme::RK4:              step_rk4(ev, X.data(), a, h, n, k1.data(), k2.data(), k3.data(), k4.data(), tmp.data()); break;
        case IntScheme::DOPRI78:          step_dopri78(ev, X.data(), a, h, n, kbuf.data(), X1.data(), X2.data()); break;
        }
    };

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
