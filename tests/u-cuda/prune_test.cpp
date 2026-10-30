// Монолитная сборка модулей (rdc off — умолчание приложения): отсечение ядер cudaLibrary.cu (этап 10).
// Модуль собирается только с ядрами из name_exprs (UCUDA_PRUNE_KERNELS / UCUDA_KEEP_<имя>) плюс держатель
// ucudaPruneKeeper. Проверка: результаты отсечённой и полной сборки (UCUDA_NVRTC_PRUNE=0) совпадают бит-в-бит
// на Лоренце (RK4 и CD fast с блоком предрасчёта) и Дуффинге с явным t — БД 1D, LLE / LS 1D, Metrics, DFT,
// включая продолжения (у них все вызовы pow идут с основанием 10, без держателя NVVM специализировал бы pow);
// печатается время сборки обоих вариантов.
#include "test_util.h"
#include "codegen.hpp"
#include "parametric_engine.h"
#include <cuda.h>
#include <chrono>
#include <cstring>

namespace {

std::vector<double> tail(std::vector<double> a) { a.insert(a.end(), { 4, 1e-12, 25, 1 }); return a; }

// Все числа прогона подряд; сравнение побитовое (NaN = NaN того же представления).
struct Run { std::vector<double> v; double seconds = 0; };

void add(Run& r, const std::vector<double>& x) { r.v.insert(r.v.end(), x.begin(), x.end()); r.v.push_back(1e300); }

Run run_all(const std::string& body, int N, const std::vector<double>& ic, const std::vector<double>& a,
            double lo, double hi, double h) {
    ParametricEngine eng;   // свой кэш модулей на каждый вариант сборки
    Run r;
    const auto t0 = std::chrono::steady_clock::now();
    for (int cont = 0; cont < 2; ++cont) {
        Bifurcation1DRequest b;
        b.krs_body = body; b.amountOfX = N; b.initial_conditions = ic; b.base_values = a;
        b.param_index = 1; b.param_lo = lo; b.param_hi = hi; b.n_pts = 6; b.writable_var = 0;
        b.h = h; b.transient_time = 10; b.t_max = 20; b.continuation = cont;
        const auto q = eng.run_bifurcation_1d(b);
        ut::require(q.ok, "BD 1D: " + q.error);
        for (const auto& p : q.bifurcation_points) add(r, p);
    }
    for (int cont = 0; cont < 2; ++cont) {
        LLE1DRequest l;
        l.krs_body = body; l.amountOfX = N; l.initial_conditions = ic; l.base_values = a;
        l.param_index = 1; l.param_lo = lo; l.param_hi = hi; l.n_pts = 4;
        l.h = h; l.transient_time = 10; l.t_max = 20; l.NT = 0.5; l.eps = 1e-6; l.continuation = cont;
        const auto q = eng.run_lle_1d(l);
        ut::require(q.ok, "LLE 1D: " + q.error);
        add(r, q.lyapunov);
        LS1DRequest s;
        s.krs_body = body; s.amountOfX = N; s.initial_conditions = ic; s.base_values = a;
        s.param_index = 1; s.param_lo = lo; s.param_hi = hi; s.n_pts = 3;
        s.h = h; s.transient_time = 10; s.t_max = 20; s.NT = 0.5; s.eps = 1e-6; s.continuation = cont;
        const auto w = eng.run_ls_1d(s);
        ut::require(w.ok, "LS 1D: " + w.error);
        for (const auto& row : w.spectrum) add(r, row);
        SignalMetricsRequest m;
        m.krs_body = body; m.amountOfX = N; m.initial_conditions = ic; m.base_values = a;
        m.param_index = 1; m.param_lo = lo; m.param_hi = hi; m.n_pts = 5;
        m.h = h; m.transient_time = 10; m.t_max = 20; m.continuation = cont;
        const auto g = eng.run_signal_metrics(m);
        ut::require(g.ok, "Metrics: " + g.error);
        for (int k = 0; k < SIGM_COUNT; ++k) add(r, g.values[k]);
        Dft1DRequest d;
        d.krs_body = body; d.amountOfX = N; d.initial_conditions = ic; d.base_values = a;
        d.param_index = 1; d.param_lo = lo; d.param_hi = hi; d.n_pts = 4; d.continuation = cont;
        d.h = h; d.transient_time = 10; d.t_max = 20; d.n_freq = 16; d.freq_lo = 0; d.freq_hi = 0.25 / h;
        const auto f = eng.run_dft_1d(d);
        ut::require(f.ok, "DFT: " + f.error);
        add(r, f.ak_cos);
    }
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

void compare(const char* name, const std::string& body, int N, const std::vector<double>& ic,
             const std::vector<double>& a, double lo, double hi, double h) {
    SetEnvironmentVariableA("UCUDA_NVRTC_PRUNE", "1");
    const Run p = run_all(body, N, ic, a, lo, hi, h);
    SetEnvironmentVariableA("UCUDA_NVRTC_PRUNE", "0");
    const Run f = run_all(body, N, ic, a, lo, hi, h);
    SetEnvironmentVariableA("UCUDA_NVRTC_PRUNE", nullptr);
    ut::require(p.v.size() == f.v.size(), std::string(name) + ": result sizes differ");
    size_t diff = 0;
    for (size_t i = 0; i < p.v.size(); ++i) diff += std::memcmp(&p.v[i], &f.v[i], sizeof(double)) != 0;
    std::printf("  %-14s %4zu numbers, %zu differ; pruned %.1f s, full %.1f s (build + run, 10 modules)\n",
                name, p.v.size() - 20, diff, p.seconds, f.seconds);
    ut::require(diff == 0, std::string(name) + ": pruned build differs from the full one");
}

}  // namespace

int main() {
    return ut::run_main("prune", []() -> int {
        SetEnvironmentVariableA("CUDA_CACHE_DISABLE", "1");   // время сборки — без кэша JIT драйвера (до cuInit)
        if (cuInit(0) != CUDA_SUCCESS) { std::printf("no CUDA device: skipped\n"); return ut::kSkip; }
        set_nvrtc_rdc(false);
        System lor;
        lor.vars = { "x", "y", "z" }; lor.params = { "sigma", "r", "b" };
        lor.rhs = { "sigma*(y - x)", "x*(r - z) - y", "x*y - b*z" };
        const auto la = tail({ 0.5, 10, 28, 8.0 / 3 });
        compare("Lorenz RK4", codegen_scheme(lor, Scheme::RK4), 3, { 1, 1, 1 }, la, 26, 30, 0.01);
        compare("Lorenz CD fast", codegen_scheme(lor, Scheme::CDFast), 3, { 1, 1, 1 }, la, 26, 30, 0.01);
        System duf;
        duf.vars = { "x", "y" }; duf.params = { "w", "d" };
        duf.rhs = { "y", "-d*y - x - x^3 + 0.5*cos(w*t)" };
        compare("Duffing (t)", codegen_scheme(duf, Scheme::RK4), 2, { 0.5, 0 }, tail({ 0.5, 1.2, 0.2 }), 1.0, 1.4, 0.01);
        return 0;
    });
}
