// Вся библиотека через движок свипов (этап 11): каждая система library/ (ОДУ, отображения, ДАУ) —
// маленькие свипы БД 1D, LLE 1D, LS 1D, Metrics, бассейны, DFT на GPU и БД 1D / LLE на CPU. Проверяется,
// что модуль собирается и считается (ok), результат нужной длины, а у точек-колебаний (флаг БД > 0) значения
// конечны; падение — ошибка движка. Разлёт / равновесие / нет пиков в окне — режим системы, не ошибка: число
// таких точек печатается для сведения.
// Схема: RK4 (ОДУ), тело отображения (Map), DAE BDF1 (ДАУ). Свип — по первому параметру +-10 %.
#include "test_util.h"
#include "codegen.hpp"
#include "sysparse.hpp"
#include "system_library.h"
#include "dae_ir.h"
#include "num_parse.h"
#include "parametric_engine.h"
#include <cuda.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>

namespace fs = std::filesystem;

namespace {

std::vector<std::string> split(const std::string& src) {
    std::vector<std::string> out; std::string cur;
    for (char c : src) {
        if (c == ',' || c == ' ' || c == '\t' || c == '\n' || c == ';') { if (!cur.empty()) { out.push_back(cur); cur.clear(); } }
        else cur += c;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}
double num(const std::map<std::string, std::string>& m, const std::string& k, double def) {
    auto it = m.find(k);
    if (it == m.end()) return def;
    return parse_num(it->second, def);   // как поля приложения: дроби, pi, приставки СИ
}
bool any_finite(const std::vector<double>& v) {
    for (double x : v) if (std::isfinite(x)) return true;
    return false;
}

struct Sys {
    std::string name, body;
    System s;
    std::vector<double> ic, a;   // a = [s, p.., хвост]
    double h = 0.01, p0 = 1.0;
    bool map = false, dae = false;
    int nx = 0;
};

int g_fail = 0, g_runs = 0, g_vals = 0, g_finite = 0;
std::vector<std::string> g_fails, g_warns;
bool g_custom = false;   // идёт прогон пользовательской КРС: её поломка — данные записи, предупреждение

// ok — расчёт прошёл; shape — длина результата и конечность там, где режим это обещает (why — что не так).
void report(const Sys& y, const char* task, bool ok, bool shape, const std::string& err, const std::string& why = {}) {
    ++g_runs;
    if (ok && shape) return;
    const std::string m = y.name + " / " + task + ": " + (ok ? why : err);
    if (g_custom) { g_warns.push_back(m); return; }
    ++g_fail;
    g_fails.push_back(m);
}
void count(const std::vector<double>& v) {
    for (double x : v) { ++g_vals; if (std::isfinite(x) && std::fabs(x) < 999) ++g_finite; }
}

void run_custom(ParametricEngine& eng, const Sys& y) {
    const int N = (int)y.s.vars.size();
    const bool has_p = !y.s.params.empty();
    const double lo = y.p0 == 0 ? -0.1 : y.p0 * 0.9, hi = y.p0 == 0 ? 0.1 : y.p0 * 1.1;
    const double T = y.map ? 400 : std::min(200.0, 20000 * y.h), Ttr = T / 2;
    for (int cpu = 0; cpu < 2; ++cpu) {
        Bifurcation1DRequest b;
        b.krs_body = y.body; b.amountOfX = N; b.initial_conditions = y.ic; b.base_values = y.a;
        b.param_index = has_p ? 1 : 0; b.param_lo = has_p ? lo : 0.5; b.param_hi = has_p ? hi : 0.5; b.n_pts = 8;
        b.writable_var = 0; b.h = y.h; b.transient_time = Ttr; b.t_max = T; b.use_cpu = cpu;
        const auto r = eng.run_bifurcation_1d(b);
        bool shape = r.flags.size() == 8 && r.bifurcation_points.size() == 8;
        for (size_t i = 0; shape && i < 8; ++i) {
            if (r.flags[i] > 0 && !any_finite(r.bifurcation_points[i])) shape = false;
            count(r.bifurcation_points[i]);
        }
        report(y, cpu ? "BD 1D CPU" : "BD 1D GPU", r.ok, shape, r.error, "8 points expected, oscillating points finite");
        LLE1DRequest l;
        l.krs_body = y.body; l.amountOfX = N; l.initial_conditions = y.ic; l.base_values = y.a;
        l.param_index = b.param_index; l.param_lo = b.param_lo; l.param_hi = b.param_hi; l.n_pts = 4;
        l.h = y.h; l.transient_time = Ttr; l.t_max = T; l.NT = y.map ? 10 : 10 * y.h; l.eps = 1e-6; l.use_cpu = cpu;
        const auto q = eng.run_lle_1d(l);
        count(q.lyapunov);
        report(y, cpu ? "LLE 1D CPU" : "LLE 1D GPU", q.ok, q.lyapunov.size() == 4, q.error, "4 exponents expected");
    }
}

void run_system(ParametricEngine& eng, const Sys& y) {
    const int N = (int)y.s.vars.size();
    const double lo = y.p0 == 0 ? -0.1 : y.p0 * 0.9, hi = y.p0 == 0 ? 0.1 : y.p0 * 1.1;
    const bool has_p = !y.s.params.empty();
    const double T = y.map ? 400 : std::min(200.0, 20000 * y.h), Ttr = T / 2;
    for (int cpu = 0; cpu < 2; ++cpu) {
        Bifurcation1DRequest b;
        b.krs_body = y.body; b.amountOfX = N; b.initial_conditions = y.ic; b.base_values = y.a;
        b.param_index = has_p ? 1 : 0; b.param_lo = has_p ? lo : 0.5; b.param_hi = has_p ? hi : 0.5; b.n_pts = 8;
        b.writable_var = 0; b.h = y.h; b.transient_time = Ttr; b.t_max = T; b.use_cpu = cpu; b.emit_all_samples = y.map;
        const auto r = eng.run_bifurcation_1d(b);
        bool shape = r.flags.size() == 8 && r.bifurcation_points.size() == 8;
        for (size_t i = 0; shape && i < 8; ++i) {
            if (r.flags[i] > 0 && !any_finite(r.bifurcation_points[i])) shape = false;
            count(r.bifurcation_points[i]);
        }
        report(y, cpu ? "BD 1D CPU" : "BD 1D GPU", r.ok, shape, r.error, "8 points expected, oscillating points finite");
    }
    for (int cpu = 0; cpu < 2; ++cpu) {
        LLE1DRequest l;
        l.krs_body = y.body; l.amountOfX = N; l.initial_conditions = y.ic; l.base_values = y.a;
        l.param_index = has_p ? 1 : 0; l.param_lo = has_p ? lo : 0.5; l.param_hi = has_p ? hi : 0.5; l.n_pts = 4;
        l.h = y.h; l.transient_time = Ttr; l.t_max = T; l.NT = y.map ? 10 : 10 * y.h; l.eps = 1e-6; l.use_cpu = cpu;
        const auto r = eng.run_lle_1d(l);
        count(r.lyapunov);
        report(y, cpu ? "LLE 1D CPU" : "LLE 1D GPU", r.ok, r.lyapunov.size() == 4, r.error, "4 exponents expected");
    }
    {
        LS1DRequest q;
        q.krs_body = y.body; q.amountOfX = N; q.initial_conditions = y.ic; q.base_values = y.a;
        q.param_index = has_p ? 1 : 0; q.param_lo = has_p ? lo : 0.5; q.param_hi = has_p ? hi : 0.5; q.n_pts = 2;
        q.h = y.h; q.transient_time = Ttr; q.t_max = T; q.NT = y.map ? 10 : 10 * y.h; q.eps = 1e-6;
        const auto r = eng.run_ls_1d(q);
        bool shape = r.spectrum.size() == 2;
        for (const auto& row : r.spectrum) { shape = shape && row.size() == (size_t)y.nx; count(row); }
        report(y, "LS 1D GPU", r.ok, shape, r.error, "2 points x nx exponents expected");
    }
    {
        SignalMetricsRequest q;
        q.krs_body = y.body; q.amountOfX = N; q.initial_conditions = y.ic; q.base_values = y.a;
        q.param_index = has_p ? 1 : 0; q.param_lo = has_p ? lo : 0.5; q.param_hi = has_p ? hi : 0.5; q.n_pts = 8;
        q.h = y.h; q.transient_time = Ttr; q.t_max = T;
        const auto r = eng.run_signal_metrics(q);
        bool shape = true;
        for (int m = 0; m < SIGM_COUNT; ++m) if (!r.values[m].empty()) { shape = shape && r.values[m].size() == 8; count(r.values[m]); }
        report(y, "Metrics GPU", r.ok, shape, r.error, "8 values per metric expected");
    }
    if (y.nx >= 2) {   // оси бассейна — x (z ДАУ осью быть не может)
        BasinsRequest q;
        q.krs_body = y.body; q.amountOfX = N; q.initial_conditions = y.ic; q.base_values = y.a;
        q.axis_x_var = 0; q.axis_y_var = 1;
        auto span = [](double v) { return std::fabs(v) > 1e-3 ? std::fabs(v) * 0.2 : 0.2; };
        q.axis_x_lo = y.ic[0] - span(y.ic[0]); q.axis_x_hi = y.ic[0] + span(y.ic[0]);
        q.axis_y_lo = y.ic[1] - span(y.ic[1]); q.axis_y_hi = y.ic[1] + span(y.ic[1]);
        q.n_pts = 6; q.h = y.h; q.transient_time = Ttr; q.t_max = T;
        const auto r = eng.run_basins(q);
        report(y, "basins GPU", r.ok, r.helpful_array.size() == 36, r.error, "36 cells expected");
    }
    if (!y.map) {
        Dft1DRequest q;
        q.krs_body = y.body; q.amountOfX = N; q.initial_conditions = y.ic; q.base_values = y.a;
        q.param_index = has_p ? 1 : 0; q.param_lo = has_p ? lo : 0.5; q.param_hi = has_p ? hi : 0.5; q.n_pts = 4;
        q.h = y.h; q.transient_time = Ttr; q.t_max = T; q.n_freq = 16; q.freq_lo = 0; q.freq_hi = 0.25 / y.h;
        const auto r = eng.run_dft_1d(q);
        count(r.ak_cos);
        report(y, "DFT GPU", r.ok, r.ak_cos.size() == 4 * 16 && r.flags.size() == 4, r.error, "4 x 16 coefficients expected");
    }
}

}  // namespace

int main() {
    return ut::run_main("library", []() -> int {
        if (cuInit(0) != CUDA_SUCCESS) { std::printf("no CUDA device: skipped\n"); return ut::kSkip; }
        set_nvrtc_rdc(true);
        const fs::path lib = fs::u8path(exe_dir()) / ".." / ".." / ".." / "U-CUDA" / "library";   // exe лежит в tests/u-cuda/build
        std::vector<fs::path> dirs;
        for (auto& e : fs::directory_iterator(lib)) dirs.push_back(e.path());
        std::sort(dirs.begin(), dirs.end());
        ParametricEngine eng;
        int nsys = 0, nskip = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (auto& d : dirs) {
            const fs::path p = d / "system.json";
            if (!fs::exists(p)) continue;
            std::ifstream f(p, std::ios::binary); std::stringstream ss; ss << f.rdbuf();
            SystemRecord r;
            try { r = record_from_json(ss.str()); } catch (...) { continue; }
            // Пользовательские КРС записи: тело как есть, переменные / параметры — из vars_text / params_text
            // (у "Burkin Matreshka", "sine" уравнения — заглушка, модель целиком в КРС). БД 1D и LLE на GPU и CPU.
            for (const auto& cs : r.custom_schemes) {
                if (cs.body.empty()) continue;
                Sys c;
                c.name = r.name + " [custom " + cs.name + "]";
                c.body = cs.body;
                c.s.vars = split(r.vars_text);
                c.s.params = split(r.params_text);
                if (c.s.vars.empty()) continue;
                c.nx = (int)c.s.vars.size();
                for (const auto& v : c.s.vars) c.ic.push_back(num(r.init_conditions, v, 0.1));
                c.a.push_back(0.5);
                for (const auto& q : c.s.params) c.a.push_back(num(r.param_values, q, 0.0));
                if (!c.s.params.empty()) c.p0 = c.a[1];
                c.a.insert(c.a.end(), { 4, 1e-10, 25, 1 });
                c.map = r.is_map;
                c.h = c.map ? 1.0 : num({ { "h", r.step_h } }, "h", 0.01);
                if (!(c.h > 0)) c.h = 0.01;
                const int before = g_fail;
                const size_t wb = g_warns.size();
                g_custom = true;
                run_custom(eng, c);
                g_custom = false;
                std::printf("  %-40s custom, %s\n", c.name.c_str(),
                            g_warns.size() == wb ? "ok" : "BROKEN (see warnings)");
                (void)before;
            }
            Sys y;
            y.name = r.name;
            try {
                FuncDefs funcs;
                if (r.use_aux_funcs && !r.func_defs_text.empty()) funcs = parse_func_defs(r.func_defs_text);
                if (r.is_dae) {
                    y.s = parse_dae(r.plain_text, r.vars_text, r.alg_text, r.params_text, r.dae_index_text, funcs).as_codegen_system();
                    y.body = codegen_dae_scheme(y.s, DaeScheme::BDF1);
                    y.dae = true;
                } else {
                    const std::string src = !r.alphabet_text.empty() ? r.alphabet_text : r.vars_text + "," + r.params_text;
                    const std::string& txt = (r.mode == "Plain") ? r.plain_text : r.latex_text;
                    y.s = parse_system_from_latex(txt, split(src),
                        r.param_order == "AsInSystem" ? ParamOrder::AsInSystem : ParamOrder::AsInAlphabet, funcs, r.is_map);
                    y.map = r.is_map;
                    y.body = codegen_scheme(y.s, y.map ? Scheme::Map : Scheme::RK4);
                }
            } catch (const std::exception& e) {
                std::printf("  %-40s skipped: %s\n", r.name.c_str(), e.what());
                ++nskip;
                continue;
            }
            y.nx = y.s.dae_nx >= 0 ? y.s.dae_nx : (int)y.s.vars.size();
            for (const auto& v : y.s.vars) y.ic.push_back(num(r.init_conditions, v, 0.1));
            y.a.push_back(0.5);
            for (const auto& q : y.s.params) y.a.push_back(num(r.param_values, q, 0.0));
            if (!y.s.params.empty()) y.p0 = y.a[1];
            y.a.insert(y.a.end(), { 4, 1e-10, 25, 1 });
            y.h = y.map ? 1.0 : num({ { "h", r.step_h } }, "h", 0.01);
            if (!(y.h > 0)) y.h = 0.01;
            const int before = g_fail;
            const auto ts = std::chrono::steady_clock::now();
            run_system(eng, y);
            ++nsys;
            std::printf("  %-40s %s %s, %2d failed, %.1f s\n", y.name.c_str(), y.dae ? "DAE" : y.map ? "map" : "ODE",
                        std::to_string(y.s.vars.size()).c_str(), g_fail - before,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count());
            std::fflush(stdout);
        }
        std::printf("  values: %d of %d finite (the rest: divergence / equilibrium / no peaks - regimes, not errors)\n",
                    g_finite, g_vals);
        std::printf("  %d systems (%d skipped), %d runs, %d failed, %.0f s\n", nsys, nskip, g_runs, g_fail,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        for (const auto& s : g_fails) std::printf("  FAIL %s\n", s.c_str());
        // Пользовательские КРС — данные записей library/, не код: их поломка печатается, но не валит тест.
        for (const auto& s : g_warns) std::printf("  WARNING (custom KRS of the record) %s\n", s.c_str());
        ut::require(g_fail == 0, std::to_string(g_fail) + " library runs failed");
        return 0;
    });
}
