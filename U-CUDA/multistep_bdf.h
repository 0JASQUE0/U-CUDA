#pragma once
// Многошаговые методы: BDF переменного порядка 1..5 и переменного шага (этап 9c плана DAE, общий
// для ОДУ и ДАУ). Отдельный драйвер, а не тело шага: у многошагового метода состояние — история,
// а все ядра U-CUDA — отображения X -> X. Первая версия — CPU, режим Analysis (вычислитель
// SystemEvaluator, символьный якобиан); внутренний цикл — простые массивы, чтобы его можно было
// перенести в GPU-ядро.
//
// Форма Нордсика (как LSODE): Z[j] = h^j y^(j) / j!, j = 0..q. Шаг:
//   прогноз  — Z <- Z P (треугольник Паскаля);
//   коррекция — w = y_n: x-строки  w_x - x_pred - l0 (h f(w) - h x'_pred) = 0,
//               z-строки  g(w) = 0 (у ОДУ z-строк нет); Ньютон с полным якобианом;
//   e = (w - y_pred)/l0, Z[j] += l[j] e;
//   l — коэффициенты Pi_{i=1..q}(1 + x/i), нормированные к l1 = 1 (l0 = 1, 2/3, 6/11, 12/25, 60/137).
// Ошибка: LTE = C/(C - C*) (y_n - y_pred), C = -l0/(q+1) (BDF), C* = 1/(q+1)! (прогноз Тейлора);
// у переменной индекса k >= 2 — множитель h^(k-1) (как в RADAU5). Норма — WRMS с rtol/atol.
// Порядок: старт с 1, повышение на 1 после q+1 принятых шагов без смены h (новый Z[q+1] = l_q e/(q+1));
// после двух отказов подряд — понижение. Шаг меняется не чаще раза в q+1 шагов (кроме отказа):
// пересчёт Z[j] *= eta^j. Плотный выход — полином Нордсика последнего шага.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "codegen.hpp"

struct BdfSettings {
    double rtol = 1e-6;
    std::vector<double> atol = { 1e-8 };   // одно число или по переменной
    double h0 = 0;                         // 0 — автоматически
    double hmax = std::numeric_limits<double>::infinity();
    int    max_order = 5;
};

struct BdfStats {
    long long nacc = 0, nrej = 0, nrhs = 0, njac = 0, nnewton_fail = 0;
    double hmin = 0, hmax = 0, hsum = 0;
    int    max_q = 0;
    bool   diverged = false, truncated = false;
};

namespace bdf_detail {

inline bool lu_solve(std::vector<double>& A, std::vector<double>& b, int n) {
    double amax = 0;
    for (double v : A) amax = std::fmax(amax, std::fabs(v));
    for (int k = 0; k < n; ++k) {
        int p = k;
        for (int i = k + 1; i < n; ++i) if (std::fabs(A[(size_t)i * n + k]) > std::fabs(A[(size_t)p * n + k])) p = i;
        if (!(std::fabs(A[(size_t)p * n + k]) > 1e-300)) return false;
        if (p != k) {
            for (int j = 0; j < n; ++j) std::swap(A[(size_t)k * n + j], A[(size_t)p * n + j]);
            std::swap(b[(size_t)k], b[(size_t)p]);
        }
        for (int i = k + 1; i < n; ++i) {
            const double m = A[(size_t)i * n + k] / A[(size_t)k * n + k];
            if (m == 0.0) continue;
            for (int j = k; j < n; ++j) A[(size_t)i * n + j] -= m * A[(size_t)k * n + j];
            b[(size_t)i] -= m * b[(size_t)k];
        }
    }
    for (int k = n - 1; k >= 0; --k) {
        double r = b[(size_t)k];
        for (int j = k + 1; j < n; ++j) r -= A[(size_t)k * n + j] * b[(size_t)j];
        b[(size_t)k] = r / A[(size_t)k * n + k];
    }
    (void)amax;
    return true;
}

// l[0..q]: Pi_{i=1..q}(1 + x/i), нормировка l[1] = 1.
inline void bdf_l(int q, double* l) {
    double c[7] = { 1, 0, 0, 0, 0, 0, 0 };
    for (int i = 1; i <= q; ++i)
        for (int j = i; j >= 1; --j) c[j] += c[j - 1] / i;
    for (int j = 0; j <= q; ++j) l[j] = c[j] / c[1];
}

}  // namespace bdf_detail

// Траектория от y0 (t = 0) до t_skip + t_rec. raw = false — равномерная сетка total точек через
// dt от t_skip; raw = true — узлы шага после t_skip (не больше max_pts). traj[k][i], times[k].
// false + err — система не поддержана (нет якобиана) или шаг сорвался ниже h_min.
inline bool bdf_integrate(const System& s, const double* a, int na, const double* y0, const BdfSettings& st,
                          double t_skip, double t_rec, double dt, int total, bool raw, int max_pts,
                          std::vector<std::vector<double>>& traj, std::vector<double>& times, BdfStats& stats,
                          std::string* err, const std::atomic<bool>* cancel = nullptr) {
    using namespace bdf_detail;
    const int N = (int)s.vars.size();
    const int nx = s.dae_nx < 0 ? N : s.dae_nx;
    // Индекс 2 (маятник с ограничением на скорость): при постоянном шаге BDF3 даёт лишь второй
    // порядок — разгон с порядка 1 и новый Z[q+1] вносят ошибку, которую индекс 2 не гасит. Нужна
    // форма DASSL (переменные коэффициенты, без пересчёта Нордсика); пока — индекс 1.
    if (nx < N && dae_structural_index(s) != 1) {
        if (err) *err = "DAE BDF 1-5: index 1 only for now (the constraints must determine z); for index 2-3 use "
                        "Radau IIA or a GGL method";
        return false;
    }
    System e = s;
    e.dae_nx = -1;
    const int M = (int)s.params.size();
    std::vector<double> ae(a, a + std::min(na, M + 1));
    ae.resize((size_t)M + 1, 0.0);
    const bool ht = system_uses_time(s);
    if (ht) { e.params.push_back("t"); ae.push_back(0.0); }
    double& tpar = ht ? ae.back() : ae[0];   // без t — не используется
    SystemEvaluator ev(e);
    if (!ev.has_jacobian()) { if (err) *err = "BDF needs the Jacobian (non-differentiable function in the system)"; return false; }
    auto set_t = [&](double t) { if (ht) tpar = t; };
    std::vector<int> kidx((size_t)N, 1);
    for (int i = 0; i < N && i < (int)s.dae_index.size(); ++i) kidx[(size_t)i] = std::max(1, s.dae_index[(size_t)i]);
    auto atol_of = [&](int i) { return st.atol.empty() ? 1e-8 : st.atol[(size_t)std::min(i, (int)st.atol.size() - 1)]; };

    const double tend = t_skip + t_rec;
    const int QM = std::max(1, std::min(5, st.max_order));
    std::vector<double> Z((size_t)(QM + 2) * N, 0.0), Zs(Z.size()), F((size_t)N), J((size_t)N * N), A((size_t)N * N),
        R((size_t)N), w((size_t)N), yp((size_t)N), hyp((size_t)N), ecor((size_t)N), wt((size_t)N), y((size_t)N);
    auto z = [&](int j) { return &Z[(size_t)j * N]; };
    auto wrms = [&](const double* v) {
        double s2 = 0;
        for (int i = 0; i < N; ++i) { const double q = v[i] * wt[(size_t)i]; s2 += q * q; }
        return std::sqrt(s2 / N);
    };
    auto weights = [&](const double* yy) {
        for (int i = 0; i < N; ++i) wt[(size_t)i] = 1.0 / (st.rtol * std::fabs(yy[i]) + atol_of(i));
    };
    auto finite = [&](const double* v) {
        for (int i = 0; i < N; ++i) if (!(std::fabs(v[i]) <= 1e300)) return false;
        return true;
    };

    // старт
    for (int i = 0; i < N; ++i) z(0)[i] = y0[i];
    set_t(0.0);
    ev.eval(z(0), ae.data(), F.data());
    ++stats.nrhs;
    weights(z(0));
    double h = st.h0;
    if (!(h > 0)) {
        double d0 = 0, d1 = 0;
        for (int i = 0; i < nx; ++i) { d0 = std::fmax(d0, std::fabs(z(0)[i]) * wt[(size_t)i]); d1 = std::fmax(d1, std::fabs(F[(size_t)i]) * wt[(size_t)i]); }
        h = (d0 < 1e-5 || d1 < 1e-5) ? 1e-6 * std::fmax(tend, 1e-6) : 0.01 * d0 / d1;
        h = std::fmin(h, 1e-2 * std::fmax(tend, 1e-12));
    }
    h = std::fmin(h, st.hmax);
    for (int i = 0; i < nx; ++i) z(1)[i] = h * F[(size_t)i];

    traj.clear();
    times.clear();
    int q = 1, since_change = 0, fails = 0, out_k = 0;
    double t = 0.0, hlast = h;
    double l[7];
    const double eps = std::numeric_limits<double>::epsilon();
    auto emit = [&](double tt, const double* yy) {
        traj.emplace_back(yy, yy + N);
        if (raw) times.push_back(tt);
    };
    // t = 0 входит в вывод, если транзиента нет
    if (t_skip <= 0) {
        if (raw) emit(0.0, z(0));
        else { emit(0.0, z(0)); out_k = 1; }
    }
    auto done_out = [&] { return raw ? (int)traj.size() >= max_pts : out_k >= total; };
    long long iter = 0;
    while (t < tend && !done_out()) {
        if (cancel && (++iter & 1023) == 0 && cancel->load(std::memory_order_relaxed)) return true;
        const double hmin = 10 * eps * std::fmax(std::fabs(t), 1.0);
        if (h < hmin) { if (err) *err = "BDF: step size below h_min at t = " + std::to_string(t); stats.diverged = true; return false; }
        bdf_l(q, l);
        Zs.assign(Z.begin(), Z.end());
        // прогноз
        for (int k = 0; k < q; ++k)
            for (int j = q - 1; j >= k; --j)
                for (int i = 0; i < N; ++i) z(j)[i] += z(j + 1)[i];
        for (int i = 0; i < N; ++i) { yp[(size_t)i] = z(0)[i]; hyp[(size_t)i] = z(1)[i]; w[(size_t)i] = yp[(size_t)i]; }
        weights(yp.data());
        // коррекция: Ньютон по w
        set_t(t + h);
        bool conv = false;
        for (int it = 0; it < 6; ++it) {
            ev.eval(w.data(), ae.data(), F.data());
            ++stats.nrhs;
            if (!finite(F.data())) break;
            for (int i = 0; i < N; ++i)
                R[(size_t)i] = i < nx ? -(w[(size_t)i] - yp[(size_t)i] - l[0] * (h * F[(size_t)i] - hyp[(size_t)i])) : -F[(size_t)i];
            ev.eval_jacobian(w.data(), ae.data(), J.data());
            ++stats.njac;
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    const double v = J[(size_t)i * N + j];
                    A[(size_t)i * N + j] = i < nx ? (i == j ? 1.0 : 0.0) - l[0] * h * v : v;
                }
            if (!lu_solve(A, R, N)) break;
            for (int i = 0; i < N; ++i) {
                w[(size_t)i] += R[(size_t)i];
                if (kidx[(size_t)i] > 1) R[(size_t)i] *= std::pow(h, kidx[(size_t)i] - 1);
            }
            if (!finite(w.data())) break;
            if (wrms(R.data()) < 1e-3) { conv = true; break; }
        }
        double errn = HUGE_VAL;
        if (conv) {
            const double C = l[0] / (q + 1);
            double fact = 1;
            for (int j = 2; j <= q + 1; ++j) fact *= j;
            const double ratio = C / (C + 1.0 / fact);
            for (int i = 0; i < N; ++i) {
                ecor[(size_t)i] = (w[(size_t)i] - yp[(size_t)i]) / l[0];
                double le = ratio * (w[(size_t)i] - yp[(size_t)i]);
                if (kidx[(size_t)i] > 1) le *= std::pow(h, kidx[(size_t)i] - 1);
                R[(size_t)i] = le;
            }
            errn = wrms(R.data());
        } else ++stats.nnewton_fail;
        if (!(errn <= 1.0)) {   // отказ: история назад, шаг меньше
            Z.assign(Zs.begin(), Zs.end());
            ++stats.nrej;
            ++fails;
            double eta = conv ? 0.9 * std::pow(errn, -1.0 / (q + 1)) : 0.25;
            eta = std::fmax(0.1, std::fmin(0.9, eta));
            if (fails >= 2 && q > 1) {   // два отказа подряд — порядок ниже
                --q;
                for (int i = 0; i < N; ++i) z(q + 1)[i] = 0;
            }
            for (int j = 1; j <= q; ++j) { const double f = std::pow(eta, j); for (int i = 0; i < N; ++i) z(j)[i] *= f; }
            h *= eta;
            since_change = 0;
            continue;
        }
        // принят
        fails = 0;
        for (int j = 0; j <= q; ++j) for (int i = 0; i < N; ++i) z(j)[i] += l[j] * ecor[(size_t)i];
        const double tn = t + h;
        ++stats.nacc;
        stats.hmin = stats.nacc == 1 ? h : std::fmin(stats.hmin, h);
        stats.hmax = std::fmax(stats.hmax, h);
        stats.hsum += h;
        stats.max_q = std::max(stats.max_q, q);
        if (!finite(z(0))) { stats.diverged = true; return true; }
        // вывод: полином Нордсика на (t, tn]
        if (raw) {
            if (tn >= t_skip) emit(tn, z(0));
        } else {
            for (;;) {
                const double ts = t_skip + out_k * dt;
                if (out_k >= total || ts > tn) break;
                const double th = (ts - tn) / h;
                for (int i = 0; i < N; ++i) {
                    double v = z(q)[i];
                    for (int j = q - 1; j >= 0; --j) v = v * th + z(j)[i];
                    y[(size_t)i] = v;
                }
                emit(ts, y.data());
                ++out_k;
            }
        }
        t = tn;
        hlast = h;
        ++since_change;
        // порядок и шаг
        if (since_change >= q + 1) {
            double eta = 0.9 * std::pow(std::fmax(errn, 1e-10), -1.0 / (q + 1));
            if (q < QM && eta > 1.0) {   // выше порядок: новый Z[q+1] = l_q e / (q+1)
                for (int i = 0; i < N; ++i) z(q + 1)[i] = l[q] * ecor[(size_t)i] / (q + 1);
                ++q;
                since_change = 0;
            } else if (eta > 1.2 || eta < 0.9) {
                eta = std::fmin(5.0, std::fmax(0.2, eta));
                if (h * eta > st.hmax) eta = st.hmax / h;
                for (int j = 1; j <= q; ++j) { const double f = std::pow(eta, j); for (int i = 0; i < N; ++i) z(j)[i] *= f; }
                h *= eta;
                since_change = 0;
            }
        }
    }
    (void)hlast;
    if (raw && (int)traj.size() >= max_pts && t < tend) stats.truncated = true;
    return true;
}
