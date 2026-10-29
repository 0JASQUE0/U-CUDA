#pragma once
// Многошаговые методы: BDF переменного порядка 1..5 и переменного шага (этап 9c плана DAE, общий
// для ОДУ и ДАУ). Отдельный драйвер, а не тело шага: у многошагового метода состояние — история,
// а все ядра U-CUDA — отображения X -> X.
//
// Алгоритм (форма DASSL) — kernels/ucuda_bdf.cuh, один текст для CPU и GPU. Здесь — CPU-обёртка:
// провайдер на SystemEvaluator (символьный якобиан), отмена, индекс 3 через GGL. GPU — тот же
// заголовок в модуле NVRTC (NvrtcEngine::run_phase_portraits_bdf, тела — codegen_bdf_bodies).
// (Первая версия была в форме Нордсика, как LSODE: пересчёт Z[j] *= eta^j на ДАУ индекса 2 раскачивал
// "пилу" lambda — при tol 1e-7 / 1e-9 маятник застревал на h ~ tol с ошибкой 7e-4 / 7e-5.)
// Индекс 3 — через GGL (dae_ggl_system сводит к индексу 2; множители mu в выход не идут).

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "codegen.hpp"
#include "configCUDA.h"   // numb
#include "kernels/ucuda_bdf.cuh"

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

// Система для BDF: индекс 3 сводится GGL (mu в конце состояния). false + err — не сводится.
inline bool bdf_prepare_system(const System& s, System& out, std::string* err) {
    const int N = (int)s.vars.size();
    const int nx = s.dae_nx < 0 ? N : s.dae_nx;
    if (nx < N && dae_structural_index(s) > 2) {
        std::string why;
        if (!dae_ggl_system(s, out, &why)) { if (err) *err = "DAE BDF 1-5, index 3 via GGL: " + why; return false; }
        if (dae_structural_index(out) > 2) { if (err) *err = "DAE BDF 1-5: index above 3 after GGL"; return false; }
        return true;   // dae_index у out: x — 1, z и mu — 2
    }
    out = s;
    return true;
}

// Траектория от y0 (t = 0) до t_skip + t_rec. raw = false — равномерная сетка total точек через
// dt от t_skip; raw = true — узлы шага после t_skip (не больше max_pts). traj[k][i], times[k].
// false + err — система не поддержана (нет якобиана, индекс выше 3) или шаг сорвался ниже h_min.
inline bool bdf_integrate(const System& s0, const double* a, int na, const double* y0, const BdfSettings& st,
                          double t_skip, double t_rec, double dt, int total, bool raw, int max_pts,
                          std::vector<std::vector<double>>& traj, std::vector<double>& times, BdfStats& stats,
                          std::string* err, const std::atomic<bool>* cancel = nullptr) {
    const int N0 = (int)s0.vars.size();
    System s;
    if (!bdf_prepare_system(s0, s, err)) return false;
    const int N = (int)s.vars.size();
    const int nx = s.dae_nx < 0 ? N : s.dae_nx;
    System e = s;
    e.dae_nx = -1;
    const int M = (int)s.params.size();
    std::vector<double> ae(a, a + std::min(na, M + 1));
    ae.resize((size_t)M + 1, 0.0);
    const bool ht = system_uses_time(s);
    if (ht) { e.params.push_back("t"); ae.push_back(0.0); }
    SystemEvaluator ev(e);
    if (!ev.has_jacobian()) { if (err) *err = "BDF needs the Jacobian (non-differentiable function in the system)"; return false; }

    std::vector<int> kidx((size_t)N, 1);
    for (int i = 0; i < N && i < (int)s.dae_index.size(); ++i) kidx[(size_t)i] = std::max(1, s.dae_index[(size_t)i]);
    std::vector<numb> atol((size_t)N), y((size_t)N, 0.0), ws((size_t)ucuda_bdf_ws_size(N));
    for (int i = 0; i < N; ++i) atol[(size_t)i] = st.atol.empty() ? 1e-8 : st.atol[(size_t)std::min(i, (int)st.atol.size() - 1)];
    for (int i = 0; i < N0; ++i) y[(size_t)i] = y0[i];   // mu системы GGL — с нуля

    traj.clear();
    times.clear();
    struct Provider {
        SystemEvaluator& ev; std::vector<double>& ae; bool ht; int N0; bool raw;
        std::vector<std::vector<double>>& traj; std::vector<double>& times; const std::atomic<bool>* cancel;
        void res(const numb* yy, numb t, numb* F) { if (ht) ae.back() = t; ev.eval(yy, ae.data(), F); }
        void jac(const numb* yy, numb t, numb* J) { if (ht) ae.back() = t; ev.eval_jacobian(yy, ae.data(), J); }
        void emit(int, numb t, const numb* yy) { traj.emplace_back(yy, yy + N0); if (raw) times.push_back(t); }
        bool cancelled() const { return cancel && cancel->load(std::memory_order_relaxed); }
    } fn{ ev, ae, ht, N0, raw, traj, times, cancel };
    UcudaBdfParams P;
    P.rtol = st.rtol;
    P.h0 = st.h0;
    P.hmax = std::isfinite(st.hmax) ? st.hmax : 0.0;
    P.maxk = st.max_order;
    UcudaBdfStats cs;
    ucuda_bdf_run(fn, N, nx, kidx.data(), y.data(), atol.data(), P, t_skip, t_rec, dt, total, raw ? 1 : 0, max_pts,
                  ws.data(), cs);
    stats.nacc = cs.nacc; stats.nrej = cs.nrej; stats.nrhs = cs.nrhs; stats.njac = cs.njac;
    stats.nnewton_fail = cs.nnewton_fail;
    stats.hmin = cs.hmin; stats.hmax = cs.hmax; stats.hsum = cs.hsum; stats.max_q = cs.max_q;
    stats.diverged = cs.diverged != 0; stats.truncated = cs.truncated != 0;
    if (cs.hmin_fail) {
        if (err) *err = "BDF: step size below h_min";
        return false;
    }
    return true;
}
