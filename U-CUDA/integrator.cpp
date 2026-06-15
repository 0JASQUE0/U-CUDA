#include "integrator.h"
#include <cmath>

IntScheme int_scheme_from_string(const std::string& s) {
    if (s == "Euler-Cromer")      return IntScheme::EulerCromer;
    if (s == "Explicit Midpoint") return IntScheme::ExplicitMidpoint;
    if (s == "RK4")               return IntScheme::RK4;
    return IntScheme::Euler;
}

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

    auto do_step = [&]() {
        switch (scheme) {
        case IntScheme::Euler:            step_euler(ev, X.data(), a, h, n, k1.data()); break;
        case IntScheme::EulerCromer:      step_euler_cromer(ev, X.data(), a, h, n, k1.data()); break;
        case IntScheme::ExplicitMidpoint: step_midpoint(ev, X.data(), a, h, n, k1.data(), tmp.data()); break;
        case IntScheme::RK4:              step_rk4(ev, X.data(), a, h, n, k1.data(), k2.data(), k3.data(), k4.data(), tmp.data()); break;
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
