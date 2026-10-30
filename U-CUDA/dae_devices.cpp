#include "dae_devices.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

// Число для текста ДАУ: парсер уравнений читает "1e-12" как 1*e - 12, поэтому показатель —
// через 10^(k).
std::string num(double v) {
    char b[48];
    std::snprintf(b, sizeof(b), "%.17g", v);
    std::string s = b;
    const size_t e = s.find('e');
    if (e == std::string::npos) return s;
    return "(" + s.substr(0, e) + "*10^(" + std::to_string(std::atoi(s.c_str() + e + 1)) + "))";
}

}  // namespace

std::string dev_explim(const std::string& u) {
    return "exp(min(" + u + ", 40))*(1 + max(" + u + " - 40, 0))";
}

std::string diode_current(const std::string& vd, const std::string& is, const std::string& n) {
    return "(" + is + "*(" + dev_explim(vd + "/(" + n + "*" + num(kDeviceVt) + ")") + " - 1) + "
         + num(kDeviceGmin) + "*" + vd + ")";
}

std::string ideal_diode_current(const std::string& vd, const std::string& vf, const std::string& ron) {
    const std::string vs = num(kIdealVs);
    const std::string u = "((" + vd + " - " + vf + ")/" + vs + ")";
    return "(" + vs + "/" + ron + "*(max(" + u + ", 0) + log(1 + exp(-abs(" + u + ")))) + " + num(kDeviceGmin) + "*" + vd + ")";
}

IdealJunction ideal_junction_fit(double is, double n, double rs, double iref) {
    const double nvt = n * kDeviceVt;
    const double vref = nvt * std::log(iref / is + 1.0) + rs * iref;
    IdealJunction j;
    j.ron = nvt / (iref + is) + rs;
    j.vf = vref - iref * j.ron;
    return j;
}

BjtCurrents bjt_currents(const std::string& vbe, const std::string& vbc,
                         const std::string& is, const std::string& bf, const std::string& br,
                         const std::string& nf, const std::string& nr,
                         const std::string& vaf_inv, const std::string& var_inv,
                         const std::string& ise, const std::string& ne,
                         const std::string& isc, const std::string& nc) {
    const std::string ebe = dev_explim(vbe + "/(" + nf + "*" + num(kDeviceVt) + ")");
    const std::string ebc = dev_explim(vbc + "/(" + nr + "*" + num(kDeviceVt) + ")");
    std::string q;
    if (!vaf_inv.empty()) q += " - " + vbc + "*" + vaf_inv;
    if (!var_inv.empty()) q += " - " + vbe + "*" + var_inv;
    std::string ict = is + "*(" + ebe + " - " + ebc + ")";
    if (!q.empty()) ict += "*(1" + q + ")";
    const std::string g = num(kDeviceGmin);
    BjtCurrents c;
    c.ic = "(" + ict + " - " + is + "/" + br + "*(" + ebc + " - 1) - " + g + "*" + vbc + ")";
    c.ib = "(" + is + "/" + bf + "*(" + ebe + " - 1) + " + is + "/" + br + "*(" + ebc + " - 1) + "
         + g + "*(" + vbe + " + " + vbc + "))";
    // Рекомбинация: без ISE / ISC текст прежний.
    auto leak = [&](const std::string& i0, const std::string& n, const std::string& v) {
        return i0 + "*(" + dev_explim(v + "/(" + n + "*" + num(kDeviceVt) + ")") + " - 1)";
    };
    if (!ise.empty()) c.ib = "(" + c.ib + " + " + leak(ise, ne, vbe) + ")";
    if (!isc.empty()) {
        const std::string l = leak(isc, nc, vbc);
        c.ib = "(" + c.ib + " + " + l + ")";
        c.ic = "(" + c.ic + " - " + l + ")";
    }
    return c;
}

std::string mosfet_drain_current(const std::string& vgs, const std::string& vds, const std::string& vgd,
                                 const std::string& vt_s, const std::string& vt_d,
                                 const std::string& beta, const std::string& lambda) {
    auto half = [&](const std::string& vg, const std::string& v, const std::string& vt) {   // F(vg, v) при v >= 0
        const std::string vov = "max(" + vg + " - " + vt + ", 0)";
        const std::string vm = "min(" + v + ", " + vov + ")";
        std::string f = beta + "*" + vm + "*(2*" + vov + " - " + vm + ")";
        if (!lambda.empty()) f += "*(1 + " + lambda + "*" + v + ")";
        return f;
    };
    return "(" + half(vgs, "max(" + vds + ", 0)", vt_s) + " - " + half(vgd, "max(-" + vds + ", 0)", vt_d) + ")";
}

std::string jfet_drain_current(const std::string& vgs, const std::string& vds, const std::string& vgd,
                               const std::string& vto, const std::string& beta, const std::string& lambda) {
    return mosfet_drain_current(vgs, vds, vgd, vto, vto, beta, lambda);
}

std::string mosfet_threshold(const std::string& vt0, const std::string& gamma, const std::string& phi,
                             const std::string& vbs) {
    return "(" + vt0 + " + " + gamma + "*(sqrt(max(" + phi + " - " + vbs + ", 0.01*" + phi + ")) - sqrt(" + phi + ")))";
}

std::string junction_capacitance(const std::string& v, const std::string& cj0, const std::string& vj,
                                 const std::string& m, const std::string& fc) {
    const std::string vfc = "(" + fc + "*" + vj + ")";
    return "(" + cj0 + "*(1 - min(" + v + ", " + vfc + ")/" + vj + ")^(-" + m + ") + " + cj0 + "*" + m + "/(" + vj
         + "*(1 - " + fc + ")^(1 + " + m + "))*max(" + v + " - " + vfc + ", 0))";
}

bool dae_dc_operating_point(const System& s, const double* a, int na, double t0, double* X,
                            int* iters, std::string* why) {
    const int N = (int)s.vars.size();
    if (iters) *iters = 0;
    if (N == 0) return true;
    // [f; g] целиком — как в dae_consistent_z: вычислитель видит ДАУ как набор выражений.
    System e = s;
    e.dae_nx = -1;
    const int M = (int)s.params.size();
    std::vector<double> ae(a, a + std::min(na, M + 1));
    ae.resize((size_t)M + 1, 0.0);
    if (system_uses_time(s)) { e.params.push_back("t"); ae.push_back(t0); }
    SystemEvaluator ev(e);
    if (!ev.has_jacobian()) { if (why) *why = "the Jacobian is not available (non-differentiable function)"; return false; }

    std::vector<double> F(N), Ft(N), J((size_t)N * N), dx(N), Xt(N);
    auto fnorm = [&](const double* x, double* f) {
        ev.eval(x, ae.data(), f);
        double m = 0;
        for (int i = 0; i < N; ++i) {
            if (!(std::fabs(f[i]) <= 1e300)) return HUGE_VAL;
            m = std::fmax(m, std::fabs(f[i]));
        }
        return m;
    };
    double fn = fnorm(X, F.data());
    if (!(fn < HUGE_VAL)) { if (why) *why = "the start point gives inf/NaN"; return false; }
    for (int it = 1; it <= 200; ++it) {
        if (iters) *iters = it;
        ev.eval_jacobian(X, ae.data(), J.data());
        for (int i = 0; i < N; ++i) dx[i] = -F[i];
        // Строки — к max|.| = 1: у схемы в одной матрице 1/C ~ 1e9 и 1/R ~ 1e-6, общий порог
        // вырожденности по max|J| принял бы проводимость за ноль.
        for (int i = 0; i < N; ++i) {
            double rm = 0;
            for (int j = 0; j < N; ++j) rm = std::fmax(rm, std::fabs(J[(size_t)i * N + j]));
            if (!(rm > 0)) {
                if (why) *why = "equation " + std::to_string(i + 1) + " does not depend on any variable at the start point";
                return false;
            }
            for (int j = 0; j < N; ++j) J[(size_t)i * N + j] /= rm;
            dx[i] /= rm;
        }
        // LU с выбором главного по столбцу (строки целиком: решение здесь же, без повтора перестановок)
        const double amax = 1.0;
        for (int k = 0; k < N; ++k) {
            int p = k;
            for (int i = k + 1; i < N; ++i)
                if (std::fabs(J[(size_t)i * N + k]) > std::fabs(J[(size_t)p * N + k])) p = i;
            if (!(std::fabs(J[(size_t)p * N + k]) > 1e-14 * (amax > 0 ? amax : 1.0))) {
                if (why) *why = "singular Jacobian at " + s.vars[(size_t)k] + ": a node without a DC path to ground, "
                                "a loop of voltage sources, or a non-isolated equilibrium";
                return false;
            }
            if (p != k) {
                for (int j = 0; j < N; ++j) std::swap(J[(size_t)k * N + j], J[(size_t)p * N + j]);
                std::swap(dx[k], dx[p]);
            }
            for (int i = k + 1; i < N; ++i) {
                const double m = J[(size_t)i * N + k] / J[(size_t)k * N + k];
                if (m == 0.0) continue;
                for (int j = k; j < N; ++j) J[(size_t)i * N + j] -= m * J[(size_t)k * N + j];
                dx[i] -= m * dx[k];
            }
        }
        for (int k = N - 1; k >= 0; --k) {
            double r = dx[k];
            for (int j = k + 1; j < N; ++j) r -= J[(size_t)k * N + j] * dx[j];
            dx[k] = r / J[(size_t)k * N + k];
        }
        // Ограничение поправки: напряжения на p-n переходах дальше 1 В за шаг не прыгают.
        double dmax = 0, xmax = 0;
        for (int i = 0; i < N; ++i) {
            if (!(std::fabs(dx[i]) <= 1e300)) {   // fmax пропустил бы NaN мимо ограничения шага
                if (why) *why = "the Newton correction is not finite at " + s.vars[(size_t)i] + " (iteration "
                                + std::to_string(it) + ")";
                return false;
            }
            dmax = std::fmax(dmax, std::fabs(dx[i]));
            xmax = std::fmax(xmax, std::fabs(X[i]));
        }
        double lam = dmax > 1.0 + xmax ? (1.0 + xmax) / dmax : 1.0;
        double fnt = HUGE_VAL;
        // 13 попыток от стартового lam (оно уже может быть мало: почти вырожденный якобиан даёт
        // огромную поправку, и порог по абсолютному lam не дал бы сделать ни одной).
        for (int tr = 0; tr < 13; ++tr, lam *= 0.5) {
            for (int i = 0; i < N; ++i) Xt[i] = X[i] + lam * dx[i];
            fnt = fnorm(Xt.data(), Ft.data());
            if (fnt <= (1.0 - 1e-4 * lam) * fn || fnt <= 1e-13) break;
        }
        if (!(fnt < HUGE_VAL)) { if (why) *why = "Newton step leads to inf/NaN"; return false; }
        std::copy(Xt.begin(), Xt.end(), X);
        F = Ft;
        fn = fnt;
        double xn = 0;
        for (int i = 0; i < N; ++i) xn = std::fmax(xn, std::fabs(X[i]));
        if (lam * dmax <= 1e-12 * (1.0 + xn) && fn <= 1e-9) return true;
        if (fn <= 1e-13) return true;
    }
    if (why) *why = "Newton did not converge in 200 iterations (max |F| = " + num(fn) + ")";
    return false;
}
