// Адаптивный шаг и семейства анализа (порт смысла adaptive_test из CMake-форка на API U-CUDA):
//  - CPU-драйвер (computePhasePortraitCPU_adaptive): плотный выход, отказы, осциллятор, допуск;
//  - GPU (run_phase_portraits_adaptive) совпадает с CPU; Лоренц против мелкого RK4;
//  - адаптивные LLE/LS: показатели линейной системы (классика, продолжение, CPU = GPU);
//  - постоянный шаг: БД 1D/2D, LLE/LS 2D, DFT, бассейны, FastSync считаются; комплексные и
//    композиционные схемы — точность на x' = -x.
// Без CUDA проверяется только CPU-часть, код выхода 77.
#include "test_util.h"
#include "adaptive_settings.h"
#include "codegen.hpp"
#include "integrator.h"
#include "krs_cpu.h"
#include "nvrtc_engine.h"
#include "parametric_engine.h"
#include <cuda.h>

namespace {

System sys_of(std::vector<std::string> vars, std::vector<std::string> rhs, std::vector<std::string> params = { "p" }) {
    System s;
    s.vars = std::move(vars);
    s.rhs = std::move(rhs);
    s.params = std::move(params);
    return s;
}

// Параметры ядра адаптивного шага: DOP853, Hairer, rtol/atol.
UcudaAdaptParams params_for(const AdaptiveCode& code, int n, double span, const char* rtol, const char* atol,
                            const char* hmax = "") {
    AdaptiveSettings st;
    st.enabled = true;
    st.rtol = rtol;
    st.atol = atol;
    st.hmax = hmax;
    UcudaAdaptParams P{};
    std::string err;
    ut::require(adaptive_build_params(st, code, n, span, P, err), "adaptive_build_params: " + err);
    return P;
}

AdaptiveRequest sweep_adaptive(const System& s, const char* rtol, double span) {
    const AdaptiveCode code = codegen_adaptive(s, Scheme::DOP853);
    AdaptiveRequest a;
    a.enabled = true;
    a.rhs = code.rhs; a.emb = code.emb; a.dprep = code.dprep; a.deval = code.deval;
    a.params = params_for(code, (int)s.vars.size(), span, rtol, "1e-12");
    return a;
}

// CPU: равномерная сетка total точек через dt от 0.
std::vector<std::vector<double>> cpu_grid(const System& s, const std::vector<double>& ic, const std::vector<double>& a,
                                          const UcudaAdaptParams& P, double dt, int total, AdaptiveStats& st) {
    SystemEvaluator ev(s);
    std::vector<std::vector<double>> traj;
    std::vector<double> times, log;
    double hfin = 0;
    computePhasePortraitCPU_adaptive(ev, IntScheme::DOP853, ic.data(), (int)ic.size(), a.data(), P,
                                     false, 0.0, dt * (total - 1), dt, total, 0, 0, traj, times, log, st, hfin);
    return traj;
}

void cpu_tests() {
    const System decay = sys_of({ "x" }, { "-x" });
    const AdaptiveCode code = codegen_adaptive(decay, Scheme::DOP853);
    const std::vector<double> a = { 0.5, 0.0 };
    AdaptiveStats st;
    auto tr = cpu_grid(decay, { 1.0 }, a, params_for(code, 1, 10.0, "1e-9", "1e-12", "1"), 0.01, 1001, st);
    ut::require(tr.size() == 1001, "CPU dense grid: wrong number of samples");
    double emax = 0;
    for (int i = 0; i < 1001; ++i) emax = std::fmax(emax, std::fabs(tr[(size_t)i][0] - std::exp(-0.01 * i)));
    ut::require(emax < 1e-8, ut::num("CPU dense output error", emax, 0, 1e-8));
    ut::require(st.nacc < 250, "dense output forced a step per sample");

    const System osc = sys_of({ "x", "y" }, { "y", "-x" });
    const AdaptiveCode oc = codegen_adaptive(osc, Scheme::DOP853);
    tr = cpu_grid(osc, { 1.0, 0.0 }, a, params_for(oc, 2, 20.0, "1e-10", "1e-12"), 20.0, 2, st);
    ut::approx(tr.back()[0], std::cos(20.0), 2e-8, "oscillator x(20)");
    ut::approx(tr.back()[1], -std::sin(20.0), 2e-8, "oscillator y(20)");

    double err[2];
    const char* tols[2] = { "1e-3", "1e-8" };
    for (int k = 0; k < 2; ++k) {
        tr = cpu_grid(decay, { 1.0 }, a, params_for(code, 1, 2.0, tols[k], "1e-12"), 2.0, 2, st);
        err[k] = std::fabs(tr.back()[0] - std::exp(-2.0));
    }
    ut::require(err[1] < err[0] * 0.01, "tightening rtol did not improve accuracy");

    AdaptiveSettings bad;
    bad.enabled = true;
    bad.rtol = "nan";
    UcudaAdaptParams P{};
    std::string e;
    ut::require(!adaptive_build_params(bad, code, 1, 1.0, P, e) && !e.empty(), "NaN rtol accepted");
    bad.rtol = "0"; bad.atol = "0";
    ut::require(!adaptive_build_params(bad, code, 1, 1.0, P, e), "rtol = atol = 0 accepted");
}

void gpu_tests() {
    NvrtcEngine engine;
    const System decay = sys_of({ "x" }, { "-x" });
    const AdaptiveCode code = codegen_adaptive(decay, Scheme::DOP853);
    const UcudaAdaptParams P = params_for(code, 1, 1.1, "1e-9", "1e-12", "0.5");
    PhaseAdaptiveRequest rq;
    rq.rhs = code.rhs; rq.emb = code.emb; rq.dprep = code.dprep; rq.deval = code.deval;
    rq.amountOfX = 1; rq.ic_flat = { 1.0, 2.0 }; rq.N = 2; rq.values = { 0.5, 0.0 }; rq.params = P;
    rq.t_rec = 1.0; rq.dt = 0.01; rq.total = 101; rq.t_skip = 0.1;
    PhaseAdaptiveResult R;
    ut::require(engine.run_phase_portraits_adaptive(rq, R), "GPU adaptive: " + engine.error());
    ut::require(R.stats.size() == 2 && R.stats[0].nacc > 0, "GPU statistics missing");
    AdaptiveStats st;
    SystemEvaluator ev(decay);
    std::vector<std::vector<double>> traj;
    std::vector<double> times, log;
    double hfin = 0;
    computePhasePortraitCPU_adaptive(ev, IntScheme::DOP853, rq.ic_flat.data(), 1, rq.values.data(), P, false,
                                     0.1, 1.0, 0.01, 101, 0, 0, traj, times, log, st, hfin);
    ut::require(traj.size() == 101 && R.traj[0].size() == 101, "grid sizes");
    for (int i = 0; i < 101; ++i)
        ut::approx(R.traj[0][(size_t)i][0], traj[(size_t)i][0], 1e-9, "CPU / GPU dense output");
    ut::approx(R.traj[1].back()[0], 2.0 * std::exp(-1.1), 2e-8, "GPU ensemble x(1.1)");

    // Лоренц: адаптивный DOP853 на GPU против мелкого RK4 на CPU (cl.exe)
    System lor = sys_of({ "x", "y", "z" }, { "sigma*(y - x)", "x*(rho - z) - y", "x*y - beta*z" },
                        { "sigma", "rho", "beta" });
    const AdaptiveCode lc = codegen_adaptive(lor, Scheme::DOP853);
    PhaseAdaptiveRequest lq;
    lq.rhs = lc.rhs; lq.emb = lc.emb; lq.dprep = lc.dprep; lq.deval = lc.deval;
    lq.amountOfX = 3; lq.ic_flat = { 1, 1, 1 }; lq.N = 1; lq.values = { 0.5, 10, 28, 8.0 / 3 };
    lq.params = params_for(lc, 3, 5.0, "1e-11", "1e-13"); lq.t_rec = 5.0; lq.dt = 5.0; lq.total = 2;
    ut::require(engine.run_phase_portraits_adaptive(lq, R), "GPU Lorenz: " + engine.error());
    KrsCpuStep ref;
    std::vector<KrsCpuDiag> diags;
    ut::require(ref.compile(codegen_scheme(lor, Scheme::RK4), 3, 4, diags), "RK4 reference compile");
    numb X[3] = { 1, 1, 1 }, A[4] = { 0.5, 10, 28, (numb)(8.0 / 3) };
    for (int i = 0; i < 500000; ++i) ref.fn()(X, A, (numb)1e-5);
    for (int k = 0; k < 3; ++k) ut::approx(R.traj[0].back()[(size_t)k], X[k], 1e-5, "Lorenz x(5) vs fine RK4");

    ParametricEngine sweep;
    // адаптивные LLE / LS на линейной системе
    {
        LLE1DRequest q;
        q.krs_body = codegen_scheme(decay, Scheme::DOP853); q.amountOfX = 1; q.initial_conditions = { 1 };
        q.base_values = { 0.5, 0.0 }; q.param_index = 1; q.param_lo = 0; q.param_hi = 1; q.n_pts = 2;
        q.h = 0.01; q.t_max = 2; q.NT = 0.1; q.eps = 1e-4; q.adaptive = sweep_adaptive(decay, "1e-9", 2.0);
        auto g = sweep.run_lle_1d(q);
        ut::require(g.ok && g.lyapunov.size() == 2, "adaptive LLE GPU: " + g.error);
        ut::approx(g.lyapunov[0], -1.0, 1e-4, "adaptive LLE GPU");
        q.use_cpu = true;
        auto c = sweep.run_lle_1d(q);
        ut::require(c.ok, "adaptive LLE CPU: " + c.error);
        ut::approx(c.lyapunov[0], g.lyapunov[0], 1e-5, "adaptive LLE CPU vs GPU");
        q.use_cpu = false; q.continuation = true;
        auto k = sweep.run_lle_1d(q);
        ut::require(k.ok, "adaptive LLE continuation: " + k.error);
        ut::approx(k.lyapunov[0], -1.0, 1e-4, "adaptive LLE continuation");
    }
    const System diag = sys_of({ "x", "y" }, { "-x", "-2*y" });
    {
        LS1DRequest q;
        q.krs_body = codegen_scheme(diag, Scheme::DOP853); q.amountOfX = 2; q.initial_conditions = { 1, 1 };
        q.base_values = { 0.5, 0.0 }; q.param_index = 1; q.param_lo = 0; q.param_hi = 1; q.n_pts = 2;
        q.h = 0.01; q.t_max = 10; q.NT = 0.1; q.eps = 1e-4; q.adaptive = sweep_adaptive(diag, "1e-9", 10.0);
        auto g = sweep.run_ls_1d(q);
        ut::require(g.ok && g.spectrum.size() == 2, "adaptive LS GPU: " + g.error);
        ut::approx(g.spectrum[0][0] + g.spectrum[0][1], -3.0, 1e-4, "adaptive LS trace");
        q.use_cpu = true;
        auto c = sweep.run_ls_1d(q);
        ut::require(c.ok, "adaptive LS CPU: " + c.error);
        ut::approx(c.spectrum[0][0], g.spectrum[0][0], 1e-4, "adaptive LS CPU vs GPU");
        q.use_cpu = false; q.continuation = true;
        auto k = sweep.run_ls_1d(q);
        ut::require(k.ok, "adaptive LS continuation: " + k.error);
        ut::approx(k.spectrum[0][0] + k.spectrum[0][1], -3.0, 1e-4, "adaptive LS continuation trace");
    }
    // постоянный шаг: остальные семейства считаются
    const std::string body = codegen_scheme(diag, Scheme::RK4);
    {
        Dft1DRequest q; q.krs_body = body; q.amountOfX = 2; q.initial_conditions = { 1, 1 }; q.base_values = { 0.5, 0.0 };
        q.param_index = 1; q.param_lo = 0; q.param_hi = 1; q.n_pts = 2; q.h = 0.01; q.t_max = 1;
        auto r = sweep.run_dft_1d(q); ut::require(r.ok, "DFT 1D: " + r.error);
    }
    {
        BasinsRequest q; q.krs_body = body; q.amountOfX = 2; q.initial_conditions = { 1, 1 }; q.base_values = { 0.5, 0.0 };
        q.n_pts = 4; q.h = 0.01; q.t_max = 1; q.transient_time = 0;
        auto r = sweep.run_basins(q); ut::require(r.ok, "basins: " + r.error);
    }
    {
        FastSyncRequest q; q.krs_body = body; q.amountOfX = 2; q.values = { 0.5, 0.0 };
        q.ic_master = { 1, 1 }; q.ic_slave = { 1.01, 1.01 }; q.k_forward = { 1, 1 }; q.k_backward = { 1, 1 };
        q.h = 0.01; q.t_max = 0.1; q.window = 0.05; q.iter_of_synchr = 2; q.transient_time = 0; q.n_pts = 4;
        auto r = sweep.run_fastsync(q); ut::require(r.ok, "FastSync: " + r.error);
    }
    {
        Bifurcation2DRequest q; q.krs_body = body; q.amountOfX = 2; q.initial_conditions = { 1, 1 }; q.base_values = { 0.5, 0.0 };
        q.n_pts = 2; q.sweep_over_var = q.sweep_over_var_2 = true; q.var_sweep_index = 0; q.var_sweep_index_2 = 1;
        q.param_lo = q.param_lo_2 = 0.5; q.param_hi = q.param_hi_2 = 1; q.h = 0.01; q.t_max = 1;
        auto r = sweep.run_bifurcation_2d(q); ut::require(r.ok, "BD 2D: " + r.error);
        LLE2DRequest l; l.krs_body = body; l.amountOfX = 2; l.initial_conditions = { 1, 1 }; l.base_values = { 0.5, 0.0 };
        l.n_pts = 2; l.sweep_over_var = l.sweep_over_var_2 = true; l.var_sweep_index = 0; l.var_sweep_index_2 = 1;
        l.param_lo = l.param_lo_2 = 0.5; l.param_hi = l.param_hi_2 = 1; l.h = 0.01; l.t_max = 1; l.NT = 0.1;
        auto lr = sweep.run_lle_2d(l); ut::require(lr.ok, "LLE 2D: " + lr.error);
        LS2DRequest s2; s2.krs_body = body; s2.amountOfX = 2; s2.initial_conditions = { 1, 1 }; s2.base_values = { 0.5, 0.0 };
        s2.n_pts = 2; s2.sweep_over_var = s2.sweep_over_var_2 = true; s2.var_sweep_index = 0; s2.var_sweep_index_2 = 1;
        s2.param_lo = s2.param_lo_2 = 0.5; s2.param_hi = s2.param_hi_2 = 1; s2.h = 0.01; s2.t_max = 1; s2.NT = 0.1;
        auto sr = sweep.run_ls_2d(s2); ut::require(sr.ok, "LS 2D: " + sr.error);
    }
    // комплексные и композиционные схемы: x(1) на x' = -x, y' = -2y
    for (const char* name : { "Complex CD", "Complex CD4", "Complex Implicit Euler", "SEMP", "SIMP", "CD", "Implicit Midpoint" }) {
        const std::string b = codegen_scheme(diag, scheme_from_name(name));
        std::vector<std::vector<std::vector<double>>> out;
        ut::require(engine.compile(b, 2) && engine.run_phase_portraits({ 1, 1 }, 1, { 0.5, 0.0, 4, 1e-12, 25, 1 }, 0.01, 101, 0, out),
                    std::string(name) + ": " + engine.error());
        ut::approx(out[0].back()[0], std::exp(-1.0), 2e-4, name);
    }
}

}  // namespace

int main(int argc, char** argv) {
    return ut::run_main("adaptive", [&]() -> int {
        cpu_tests();
        if (cuInit(0) != CUDA_SUCCESS) { std::printf("no CUDA device: GPU part skipped\n"); return ut::kSkip; }
        gpu_tests();
        return 0;
    });
}
