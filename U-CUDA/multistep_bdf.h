#pragma once
// Многошаговые методы: BDF переменного порядка 1..5 и переменного шага (этап 9c плана DAE, общий
// для ОДУ и ДАУ). Отдельный драйвер, а не тело шага: у многошагового метода состояние — история,
// а все ядра U-CUDA — отображения X -> X. Первая версия — CPU, режим Analysis (вычислитель
// SystemEvaluator, символьный якобиан); внутренний цикл — простые массивы, чтобы его можно было
// перенести в GPU-ядро.
//
// Форма DASSL (DDASTP, Brenan-Campbell-Petzold): история — модифицированные разделённые разности
// phi[1..k+1] по РЕАЛЬНЫМ узлам, коэффициенты alpha, beta, gamma, sigma, psi пересчитываются после
// смены шага / порядка; сама история при смене шага не масштабируется. (Первая версия была в форме
// Нордсика, как LSODE: пересчёт Z[j] *= eta^j на ДАУ индекса 2 раскачивал "пилу" lambda — при
// tol 1e-7 / 1e-9 маятник застревал на h ~ tol с ошибкой 7e-4 / 7e-5.)
//   прогноз   y = sum phi*_j, y' = sum gamma_j phi*_j (phi* = beta phi);
//   коррекция y' = y'_pred + cj (y - y_pred), cj = -alphas/h; x-строки y'_x - f(y) = 0, z-строки
//             g(y) = 0 (у ОДУ z-строк нет); Ньютон с полным якобианом, матрица cj I - f_y | g_y;
//   ошибка    err = ck ||e||, e = y - y_pred; порядок k-1 / k / k+1 — по оценкам sigma ||.||, как в
//             DASSL; начальная фаза: порядок +1 и шаг x2 на каждом принятом шаге;
//   норма     WRMS с rtol/atol, у переменной индекса k >= 2 — множитель h^(k-1) (как в RADAU5);
//   плотный выход — интерполяция DDATRP по phi последнего шага.
// Индекс 3 — через GGL (dae_ggl_system сводит к индексу 2; множители mu в выход не идут).

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
    return true;
}

}  // namespace bdf_detail

// Траектория от y0 (t = 0) до t_skip + t_rec. raw = false — равномерная сетка total точек через
// dt от t_skip; raw = true — узлы шага после t_skip (не больше max_pts). traj[k][i], times[k].
// false + err — система не поддержана (нет якобиана, индекс выше 3) или шаг сорвался ниже h_min.
inline bool bdf_integrate(const System& s, const double* a, int na, const double* y0, const BdfSettings& st,
                          double t_skip, double t_rec, double dt, int total, bool raw, int max_pts,
                          std::vector<std::vector<double>>& traj, std::vector<double>& times, BdfStats& stats,
                          std::string* err, const std::atomic<bool>* cancel = nullptr) {
    using namespace bdf_detail;
    const int N = (int)s.vars.size();
    const int nx = s.dae_nx < 0 ? N : s.dae_nx;
    // Индекс 3: BDF (как DASSL) держит индекс до 2 — система сводится GGL (dae_ggl_system: множители
    // mu в конце состояния, x' = f - G^T mu, 0 = g, 0 = dg/dt, индекс 2, mu = 0 на решении). mu
    // стартуют с 0 и в вывод не идут.
    if (nx < N && dae_structural_index(s) > 2) {
        System g;
        std::string why;
        if (!dae_ggl_system(s, g, &why)) { if (err) *err = "DAE BDF 1-5, index 3 via GGL: " + why; return false; }
        if (dae_structural_index(g) > 2) { if (err) *err = "DAE BDF 1-5: index above 3 after GGL"; return false; }
        const int Ng = (int)g.vars.size();   // dae_index у g уже: x — 1, z и mu — 2
        std::vector<double> yg(y0, y0 + N);
        yg.resize((size_t)Ng, 0.0);
        const bool ok = bdf_integrate(g, a, na, yg.data(), st, t_skip, t_rec, dt, total, raw, max_pts, traj, times,
                                      stats, err, cancel);
        for (auto& row : traj) row.resize((size_t)N);
        return ok;
    }
    System e = s;
    e.dae_nx = -1;
    const int M = (int)s.params.size();
    std::vector<double> ae(a, a + std::min(na, M + 1));
    ae.resize((size_t)M + 1, 0.0);
    const bool ht = system_uses_time(s);
    if (ht) { e.params.push_back("t"); ae.push_back(0.0); }
    SystemEvaluator ev(e);
    if (!ev.has_jacobian()) { if (err) *err = "BDF needs the Jacobian (non-differentiable function in the system)"; return false; }
    auto set_t = [&](double tt) { if (ht) ae.back() = tt; };
    std::vector<int> kidx((size_t)N, 1);
    for (int i = 0; i < N && i < (int)s.dae_index.size(); ++i) kidx[(size_t)i] = std::max(1, s.dae_index[(size_t)i]);
    auto atol_of = [&](int i) { return st.atol.empty() ? 1e-8 : st.atol[(size_t)std::min(i, (int)st.atol.size() - 1)]; };

    const double tend = t_skip + t_rec;
    const int MAXK = std::max(1, std::min(5, st.max_order));
    // phi[j], j = 1..MAXK+2 (как в DASSL, с единицы); коэффициенты — по 1..MAXK+1
    std::vector<double> phi((size_t)(MAXK + 3) * N, 0.0), F((size_t)N), J((size_t)N * N), A((size_t)N * N),
        R((size_t)N), y((size_t)N), yp((size_t)N), ypr((size_t)N), yprp((size_t)N), ev_((size_t)N), wt((size_t)N),
        dl((size_t)N), out((size_t)N);
    auto P = [&](int j) { return &phi[(size_t)j * N]; };
    double alpha[8] = {}, beta[8] = {}, gamma[8] = {}, sigma[8] = {}, psi[8] = {};
    double h_norm_scale = 1;   // h^(k-1) по индексу — от текущего h
    auto wrms = [&](const double* v) {
        double s2 = 0;
        for (int i = 0; i < N; ++i) {
            double q = v[i] * wt[(size_t)i];
            if (kidx[(size_t)i] > 1) q *= std::pow(h_norm_scale, kidx[(size_t)i] - 1);
            s2 += q * q;
        }
        return std::sqrt(s2 / N);
    };
    auto finite = [&](const double* v) {
        for (int i = 0; i < N; ++i) if (!(std::fabs(v[i]) <= 1e300)) return false;
        return true;
    };

    // старт: y0 согласован (z из связей — вызывающий); y'0: x-строки f, z-строки 0 — первый шаг
    // порядка 1 это переносит (как DASSL с приблизительной y').
    for (int i = 0; i < N; ++i) y[(size_t)i] = y0[i];
    set_t(0.0);
    ev.eval(y.data(), ae.data(), F.data());
    ++stats.nrhs;
    for (int i = 0; i < N; ++i) {
        yp[(size_t)i] = i < nx ? F[(size_t)i] : 0.0;
        wt[(size_t)i] = 1.0 / (st.rtol * std::fabs(y[(size_t)i]) + atol_of(i));
    }
    double h = st.h0;
    if (!(h > 0)) {   // DASSL: h0 = min(1e-3 |tout - t|, 0.5 / ||y'||)
        h_norm_scale = 1;
        const double ypn = wrms(yp.data());
        h = 1e-3 * std::fmax(tend, 1e-12);
        if (ypn > 0.5 / h) h = 0.5 / ypn;
    }
    h = std::fmin(h, st.hmax);
    for (int i = 0; i < N; ++i) { P(1)[i] = y[(size_t)i]; P(2)[i] = h * yp[(size_t)i]; }
    psi[1] = h;

    traj.clear();
    times.clear();
    int k = 1, kold = 0, ns = 0, iphase = 0, nef = 0, out_k = 0;
    double t = 0.0, hold = 0.0, cj = 0.0;
    const double eps = std::numeric_limits<double>::epsilon();
    auto emit = [&](double tt, const double* yy) {
        traj.emplace_back(yy, yy + N);
        if (raw) times.push_back(tt);
    };
    if (t_skip <= 0) { emit(0.0, y.data()); if (!raw) out_k = 1; }
    auto done_out = [&] { return raw ? (int)traj.size() >= max_pts : out_k >= total; };
    long long iter = 0;
    while (t < tend && !done_out()) {
        if (cancel && (++iter & 1023) == 0 && cancel->load(std::memory_order_relaxed)) return true;
        const double hmin = 10 * eps * std::fmax(std::fabs(t), 1.0);
        if (h < hmin) { if (err) *err = "BDF: step size below h_min at t = " + std::to_string(t); stats.diverged = true; return false; }
        const int kp1 = k + 1, kp2 = k + 2, km1 = k - 1;
        // --- коэффициенты (DDASTP, блок 200) ---
        if (h != hold || k != kold) ns = 0;
        ns = std::min(ns + 1, kold + 2);
        const int nsp1 = ns + 1;
        if (kp1 >= ns) {
            beta[1] = 1; alpha[1] = 1; gamma[1] = 0; sigma[1] = 1;
            double temp1 = h;
            for (int i = 2; i <= kp1; ++i) {
                const double temp2 = psi[i - 1];
                psi[i - 1] = temp1;
                beta[i] = beta[i - 1] * psi[i - 1] / temp2;
                temp1 = temp2 + h;
                alpha[i] = h / temp1;
                sigma[i] = (i - 1) * sigma[i - 1] * alpha[i];
                gamma[i] = gamma[i - 1] + alpha[i - 1] / h;
            }
            psi[kp1] = temp1;
        }
        double alphas = 0, alpha0 = 0;
        for (int i = 1; i <= k; ++i) { alphas -= 1.0 / i; alpha0 -= alpha[i]; }
        cj = -alphas / h;
        const double ck = std::fmax(std::fabs(alpha[kp1] + alphas - alpha0), alpha[kp1]);
        if (kp1 >= nsp1)
            for (int j = nsp1; j <= kp1; ++j) for (int i = 0; i < N; ++i) P(j)[i] *= beta[j];
        // --- прогноз ---
        const double told = t;
        t = told + h;
        set_t(t);
        for (int i = 0; i < N; ++i) {
            double yy = 0, yyp = 0;
            for (int j = 1; j <= kp1; ++j) { yy += P(j)[i]; yyp += gamma[j] * P(j)[i]; }
            ypr[(size_t)i] = yy; yprp[(size_t)i] = yyp; y[(size_t)i] = yy;
        }
        h_norm_scale = h;
        // --- коррекция: Ньютон по y ---
        bool conv = false;
        for (int it = 0; it < 6; ++it) {
            ev.eval(y.data(), ae.data(), F.data());
            ++stats.nrhs;
            if (!finite(F.data())) break;
            for (int i = 0; i < N; ++i)
                R[(size_t)i] = i < nx ? -(yprp[(size_t)i] + cj * (y[(size_t)i] - ypr[(size_t)i]) - F[(size_t)i]) : -F[(size_t)i];
            ev.eval_jacobian(y.data(), ae.data(), J.data());
            ++stats.njac;
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    const double v = J[(size_t)i * N + j];
                    A[(size_t)i * N + j] = i < nx ? (i == j ? cj : 0.0) - v : v;
                }
            if (!lu_solve(A, R, N)) break;
            for (int i = 0; i < N; ++i) y[(size_t)i] += R[(size_t)i];
            if (!finite(y.data())) break;
            if (wrms(R.data()) < 1e-3) { conv = true; break; }
        }
        double enorm = HUGE_VAL, erkm1 = 0, terkm1 = 0, est = 0, terk = 0;
        int knew = k;
        if (conv) {
            for (int i = 0; i < N; ++i) ev_[(size_t)i] = y[(size_t)i] - ypr[(size_t)i];
            enorm = wrms(ev_.data());
            const double erk = sigma[k + 1] * enorm;
            terk = (k + 1) * erk;
            est = erk;
            if (k > 1) {   // оценка для порядка k-1 (DDASTP, блок 400)
                for (int i = 0; i < N; ++i) dl[(size_t)i] = P(kp1)[i] + ev_[(size_t)i];
                erkm1 = sigma[k] * wrms(dl.data());
                terkm1 = k * erkm1;
                bool lower = false;
                if (k > 2) {
                    for (int i = 0; i < N; ++i) dl[(size_t)i] += P(k)[i];
                    const double erkm2 = sigma[k - 1] * wrms(dl.data());
                    lower = std::fmax(terkm1, (k - 1) * erkm2) <= terk;
                } else lower = terkm1 <= 0.5 * terk;
                if (lower) { knew = km1; est = erkm1; }
            }
        } else ++stats.nnewton_fail;
        const double errv = conv ? ck * enorm : HUGE_VAL;
        if (!(errv <= 1.0)) {
            // --- отказ: история назад (DDASTP, блок 690) ---
            t = told;
            set_t(t);
            if (kp1 >= nsp1)
                for (int j = nsp1; j <= kp1; ++j) for (int i = 0; i < N; ++i) P(j)[i] /= beta[j];
            for (int i = 2; i <= kp1; ++i) psi[i - 1] = psi[i] - h;
            ++stats.nrej;
            if (!conv) h *= 0.25;
            else {
                ++nef;
                if (nef == 1) {
                    k = knew;
                    double r = 0.9 * std::pow(2 * est + 1e-4, -1.0 / (k + 1));
                    r = std::fmax(0.25, std::fmin(0.9, r));
                    h *= r;
                } else if (nef == 2) { k = knew; h *= 0.25; }
                else { k = 1; h *= 0.25; }
            }
            iphase = 1;
            hold = 0;   // коэффициенты пересчитать
            continue;
        }
        // --- принят (DDASTP, блок 500) ---
        nef = 0;
        const int kold_step = k;
        kold = k;
        hold = h;
        ++stats.nacc;
        stats.hmin = stats.nacc == 1 ? h : std::fmin(stats.hmin, h);
        stats.hmax = std::fmax(stats.hmax, h);
        stats.hsum += h;
        stats.max_q = std::max(stats.max_q, k);
        if (knew == km1 || k == MAXK) iphase = 1;
        double hnew = h;
        if (iphase == 0) {
            k = kp1;
            hnew = 2 * h;
        } else {
            if (knew == km1) { k = km1; est = erkm1; }
            else if (kp1 <= ns && k < MAXK) {
                for (int i = 0; i < N; ++i) dl[(size_t)i] = ev_[(size_t)i] - P(kp2)[i];
                const double terkp1 = wrms(dl.data()), erkp1 = terkp1 / (k + 2);
                if (k > 1) {
                    if (terkm1 <= std::fmin(terk, terkp1)) { k = km1; est = erkm1; }
                    else if (!(terkp1 >= terk)) { k = kp1; est = erkp1; }
                } else if (!(terkp1 >= 0.5 * terk)) { k = kp1; est = erkp1; }
            }
            const double r = std::pow(2 * est + 1e-4, -1.0 / (k + 1));
            if (r >= 2.0) hnew = 2 * h;
            else if (r <= 1.0) hnew = h * std::fmax(0.5, std::fmin(0.9, r));
        }
        hnew = std::fmin(hnew, st.hmax);
        // разности для следующего шага (kp1, kp2 — от порядка этого шага)
        if (kold_step < MAXK) for (int i = 0; i < N; ++i) P(kp2)[i] = ev_[(size_t)i];
        for (int i = 0; i < N; ++i) P(kp1)[i] += ev_[(size_t)i];
        for (int j = kp1 - 1; j >= 1; --j) for (int i = 0; i < N; ++i) P(j)[i] += P(j + 1)[i];
        if (!finite(y.data())) { stats.diverged = true; return true; }
        for (int i = 0; i < N; ++i) wt[(size_t)i] = 1.0 / (st.rtol * std::fabs(y[(size_t)i]) + atol_of(i));
        // --- вывод: интерполяция DDATRP по phi и psi этого шага ---
        auto interp = [&](double tout, double* yo) {
            const double temp1 = tout - t;
            for (int i = 0; i < N; ++i) yo[i] = P(1)[i];
            double c = 1, gam = temp1 / psi[1];
            for (int j = 2; j <= kold_step + 1; ++j) {
                c *= gam;
                gam = (temp1 + psi[j - 1]) / psi[j];
                for (int i = 0; i < N; ++i) yo[i] += c * P(j)[i];
            }
        };
        if (raw) {
            if (t >= t_skip) emit(t, y.data());
        } else {
            for (;;) {
                const double ts = t_skip + out_k * dt;
                if (out_k >= total || ts > t) break;
                interp(ts, out.data());
                emit(ts, out.data());
                ++out_k;
            }
        }
        h = hnew;
    }
    if (raw && (int)traj.size() >= max_pts && t < tend) stats.truncated = true;
    return true;
}
