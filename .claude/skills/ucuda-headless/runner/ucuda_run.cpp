// ucuda_run — расчёты U-CUDA без GUI: тот же codegen и тот же ParametricEngine (NVRTC-ядра), что у
// вкладок Parametric / Basins / Metrics / DFT в Release-сборке.
//   ucuda_run <job.txt> [key=value ...]
// job — строки "key = value", '#' — комментарий; повтор ключа дописывает значение (так задаются
// rhs по одному на строку). key=value из командной строки ЗАМЕНЯЮТ значение из файла (удобно для
// циклов в PowerShell). Ключи и форматы выхода — в reference.md рядом со скиллом.
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "codegen.hpp"
#include "parametric_engine.h"

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); }

static std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) ++b;
    while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
    return s.substr(b, e - b);
}
static std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}
static std::vector<std::string> split_list(const std::string& s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) { t = trim(t); if (!t.empty()) v.push_back(t); }
    return v;
}

// ---- Числа: "8/3", "2*pi", "-1e-3", "(1+sqrt(5))/2" ------------------------------------------------
struct NumParser {
    const std::string& s; size_t i = 0;
    explicit NumParser(const std::string& str) : s(str) {}
    void ws() { while (i < s.size() && std::isspace((unsigned char)s[i])) ++i; }
    [[noreturn]] void fail() const { throw std::runtime_error("bad number: '" + s + "'"); }
    double expr() {
        double v = term();
        for (;;) { ws(); if (i < s.size() && (s[i] == '+' || s[i] == '-')) { char op = s[i++]; double r = term(); v = op == '+' ? v + r : v - r; } else return v; }
    }
    double term() {
        double v = power();
        for (;;) { ws(); if (i < s.size() && (s[i] == '*' || s[i] == '/')) { char op = s[i++]; double r = power(); v = op == '*' ? v * r : v / r; } else return v; }
    }
    double power() {
        double b = unary(); ws();
        if (i < s.size() && s[i] == '^') { ++i; return std::pow(b, power()); }
        return b;
    }
    double unary() {
        ws();
        if (i < s.size() && s[i] == '-') { ++i; return -unary(); }
        if (i < s.size() && s[i] == '+') { ++i; return unary(); }
        return atom();
    }
    double atom() {
        ws();
        if (i >= s.size()) fail();
        if (s[i] == '(') { ++i; double v = expr(); ws(); if (i >= s.size() || s[i] != ')') fail(); ++i; return v; }
        if (std::isalpha((unsigned char)s[i])) {
            size_t b = i; while (i < s.size() && std::isalnum((unsigned char)s[i])) ++i;
            std::string id = lower(s.substr(b, i - b));
            if (id == "pi") return 3.14159265358979323846;
            if (id == "e") return 2.71828182845904523536;
            if (id == "inf") return INFINITY;
            if (id == "nan") return NAN;
            ws(); if (i >= s.size() || s[i] != '(') fail(); ++i;
            double a = expr(); ws(); if (i >= s.size() || s[i] != ')') fail(); ++i;
            if (id == "sqrt") return std::sqrt(a);
            if (id == "exp") return std::exp(a);
            if (id == "log") return std::log(a);
            if (id == "sin") return std::sin(a);
            if (id == "cos") return std::cos(a);
            fail();
        }
        const char* p = s.c_str() + i; char* end = nullptr;
        double v = std::strtod(p, &end);
        if (end == p) fail();
        i += (size_t)(end - p);
        return v;
    }
};
static double parse_num(const std::string& t) {
    NumParser p(t); double v = p.expr(); p.ws();
    if (p.i != t.size()) p.fail();
    return v;
}

// ---- Job --------------------------------------------------------------------------------------------
struct Job {
    std::map<std::string, std::vector<std::string>> kv;
    std::vector<std::string> order;   // порядок ключей — для _run.txt

    void put(const std::string& k, const std::string& v, bool append) {
        auto it = kv.find(k);
        if (it == kv.end()) { order.push_back(k); kv[k] = { v }; }
        else if (append) it->second.push_back(v);
        else it->second = { v };
    }
    bool has(const std::string& k) const { return kv.count(k) != 0; }
    std::string str(const std::string& k, const std::string& def = "") const {
        auto it = kv.find(k); return it == kv.end() ? def : it->second.back();
    }
    std::string need(const std::string& k) const {
        if (!has(k)) throw std::runtime_error("job: missing key '" + k + "'");
        return str(k);
    }
    double num(const std::string& k, double def) const { return has(k) ? parse_num(str(k)) : def; }
    int    integer(const std::string& k, int def) const { return has(k) ? (int)std::llround(parse_num(str(k))) : def; }
    bool   flag(const std::string& k, bool def) const {
        if (!has(k)) return def;
        std::string v = lower(str(k));
        return v == "1" || v == "true" || v == "yes" || v == "on";
    }
    std::vector<double> nums(const std::string& k) const {
        std::vector<double> v;
        for (auto& t : split_list(str(k))) v.push_back(parse_num(t));
        return v;
    }
};

static void load_job(const std::string& path, Job& job) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open job file " + path);
    std::string line; bool first = true;
    while (std::getline(f, line)) {
        if (first && line.size() >= 3 && (unsigned char)line[0] == 0xEF && (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF)
            line.erase(0, 3);
        first = false;
        size_t h = line.find('#'); if (h != std::string::npos) line.erase(h);
        line = trim(line);
        if (line.empty()) continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos) throw std::runtime_error("job: line without '=': " + line);
        job.put(lower(trim(line.substr(0, eq))), trim(line.substr(eq + 1)), true);
    }
}

// ---- Оси свипа ----------------------------------------------------------------------------------------
struct Axis {
    std::string name;
    bool var = false, h = false, log = false;
    int  pidx = 0, vidx = 0;
    double lo = 0, hi = 1;
};
static Axis resolve_axis(const System& s, const Job& job, const std::string& suffix) {
    Axis a;
    a.name = job.need("sweep" + suffix);
    if (a.name == "h") a.h = true;
    else if (a.name == "s") a.pidx = 0;
    else {
        auto p = std::find(s.params.begin(), s.params.end(), a.name);
        auto v = std::find(s.vars.begin(), s.vars.end(), a.name);
        if (p != s.params.end()) a.pidx = (int)(p - s.params.begin()) + 1;
        else if (v != s.vars.end()) { a.var = true; a.vidx = (int)(v - s.vars.begin()); }
        else throw std::runtime_error("sweep" + suffix + " = '" + a.name + "' is not a parameter, a variable, 'h' or 's'");
    }
    auto r = job.nums("range" + suffix);
    if (r.size() != 2) throw std::runtime_error("range" + suffix + " must be 'lo, hi'");
    a.lo = r[0]; a.hi = r[1];
    a.log = job.flag("log" + suffix, false);
    return a;
}
static double node(double lo, double hi, int i, int n, bool lg) {
    if (n < 2) return lo;
    double u = (double)i / (n - 1);
    return lg ? lo * std::pow(hi / lo, u) : lo + (hi - lo) * u;
}
static int var_index(const System& s, const std::string& name, const char* key) {
    auto v = std::find(s.vars.begin(), s.vars.end(), name);
    if (v == s.vars.end()) throw std::runtime_error(std::string(key) + " = '" + name + "' is not a variable");
    return (int)(v - s.vars.begin());
}

// Поля, у которых одни и те же имена во всех запросах свипов.
template <class R> static void fill_common(R& r, const std::string& body, const System& s,
                                           const std::vector<double>& ic, const std::vector<double>& vals, const Job& job) {
    r.krs_body = body; r.amountOfX = (int)s.vars.size();
    r.initial_conditions = ic; r.base_values = vals;
    r.h = job.num("h", 0.01);
    r.transient_time = job.num("transient", 0.0);
    r.t_max = job.num("tmax", 100.0);
    r.max_value = job.num("max_value", 1e6);
}
template <class R> static void fill_axis1(R& r, const Axis& a) {
    r.param_index = a.pidx; r.sweep_over_var = a.var; r.var_sweep_index = a.vidx; r.sweep_over_h = a.h;
    r.log_scale = a.log; r.param_lo = a.lo; r.param_hi = a.hi;
}
template <class R> static void fill_axis2(R& r, const Axis& a) {
    r.param_index_2 = a.pidx; r.sweep_over_var_2 = a.var; r.var_sweep_index_2 = a.vidx; r.sweep_over_h_2 = a.h;
    r.log_scale_2 = a.log; r.param_lo_2 = a.lo; r.param_hi_2 = a.hi;
}
template <class R> static void fill_cont(R& r, const Job& job) {
    r.continuation = job.flag("continuation", false);
    r.continuation_reverse = job.flag("reverse", false);
    r.use_cpu = job.flag("cpu", false);
}

// Прогресс в stderr раз в ~5 с — фоновые прогоны видно в выводе.
struct ProgressPrinter {
    std::shared_ptr<std::atomic<float>> p = std::make_shared<std::atomic<float>>(0.0f);
    std::atomic<bool> stop{ false };
    std::thread th;
    ProgressPrinter() {
        th = std::thread([this] {
            auto t0 = clk::now(); int tick = 0; float last = 0.0f;
            while (!stop.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                if (++tick % 25 != 0) continue;
                const float v = p->load();
                if (v == last) continue;   // 0 % на компиляции модуля не печатаем
                last = v;
                std::fprintf(stderr, "progress %5.1f%%  %.0f s\n", 100.0 * v, secs(t0)); std::fflush(stderr);
            }
        });
    }
    ~ProgressPrinter() { stop = true; th.join(); }
};

// Флаги ядра 999 / -999 («сошёл с аттрактора» / «разошёлся») — в NaN, как делает run_lle_1d;
// причина остаётся в столбце flag.
static double lyap_value(double v) { return std::fabs(v) == 999.0 ? NAN : v; }

static std::ofstream open_csv(const std::string& path, int prec) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot write " + path);
    f.precision(prec);
    return f;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: ucuda_run <job.txt> [key=value ...]\n");
        return 2;
    }
    try {
        Job job;
        load_job(argv[1], job);
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i]; size_t eq = a.find('=');
            if (eq == std::string::npos) throw std::runtime_error("override without '=': " + a);
            job.put(lower(trim(a.substr(0, eq))), trim(a.substr(eq + 1)), false);
        }

        const std::string task = lower(job.need("task"));
        const std::string out = job.need("out");
        const int prec = job.integer("precision", 12);

        // Система
        System s;
        s.vars = split_list(job.need("vars"));
        s.params = split_list(job.str("params"));
        s.rhs = job.has("rhs") ? job.kv.at("rhs") : std::vector<std::string>{};
        s.is_map = job.flag("map", false);
        if (s.rhs.size() != s.vars.size() && !job.has("body_file"))
            throw std::runtime_error("need one 'rhs =' line per variable (" + std::to_string(s.vars.size()) + "), got " + std::to_string(s.rhs.size()));
        const std::string scheme_name = s.is_map ? "Map" : job.str("scheme", "RK4");

        std::string body;
        if (job.has("body_file")) {
            std::ifstream bf(job.str("body_file"), std::ios::binary);
            if (!bf) throw std::runtime_error("cannot open body_file " + job.str("body_file"));
            std::stringstream ss; ss << bf.rdbuf(); body = ss.str();
        } else {
            const Scheme sch = scheme_from_name(scheme_name);
            if (sch == Scheme::Euler && scheme_name != "Euler")
                throw std::runtime_error("unknown scheme '" + scheme_name + "' (names as in the GUI: RK4, CD, CD fast, DOPRI78, Implicit Euler, GBS 2-4, ...)");
            body = codegen_scheme(s, sch);
        }
        if (job.flag("dump_body", false)) { std::ofstream f(out + "_body.cu"); f << body; }

        // a[] = [s, p_1..p_M, n, newton_tol, newton_max_iters, newton_full]
        std::vector<double> vals = { job.num("s", 0.5) };
        {
            auto pv = job.has("values") ? job.nums("values") : std::vector<double>{};
            if (pv.size() != s.params.size())
                throw std::runtime_error("values: " + std::to_string(pv.size()) + " numbers for " + std::to_string(s.params.size()) + " params");
            vals.insert(vals.end(), pv.begin(), pv.end());
            NewtonSettings nw;
            nw.full = job.flag("newton_full", false);
            nw.tol = job.str("newton_tol", "1e-10");
            nw.max_iters = job.str("newton_max_iters", "8");
            scheme_append_tail(vals, job.str("cd_fp_iters", "4"), nw);
        }
        const std::vector<double> ic = job.nums("ic");
        if (ic.size() != s.vars.size())
            throw std::runtime_error("ic: " + std::to_string(ic.size()) + " numbers for " + std::to_string(s.vars.size()) + " vars");
        const int N = (int)s.vars.size();

        // Глобальные настройки движка (в GUI — Settings / _app_config.json)
        PeakConfig pc = get_peak_config();
        pc.do_interpolate_peaks = job.flag("peaks_interpolate", pc.do_interpolate_peaks);
        pc.eps_fixed_point = job.num("eps_fixed_point", pc.eps_fixed_point);
        pc.eps_peak_delta = job.num("eps_peak_delta", pc.eps_peak_delta);
        pc.eps_interPeak_delta = job.num("eps_interpeak_delta", pc.eps_interPeak_delta);
        pc.peak_threshold = job.num("peak_threshold", pc.peak_threshold);
        pc.max_amount_of_peaks = job.integer("max_peaks", pc.max_amount_of_peaks);
        clamp_peak_config(pc);
        set_peak_config(pc);
        set_nvrtc_fmad(job.flag("fmad", true));
        set_nvrtc_rdc(job.flag("rdc", true));
        set_gpu_block_size(job.integer("block", kGpuBlockSizeDefault));

        const int n = job.integer("n", 500);
        const std::string gui_csv = job.flag("gui_csv", false) ? out + "_gui" : std::string();
        const int wvar = job.has("out_var") ? var_index(s, job.str("out_var"), "out_var") : 0;
        std::string file = out + "_" + task + ".csv";

        ParametricEngine eng;
        ProgressPrinter prog;
        const auto t0 = clk::now();
        std::string err;
        bool ok = false, cancelled = false;
        std::string summary;

        if (task == "bif1d") {
            Bifurcation1DRequest r; fill_common(r, body, s, ic, vals, job);
            Axis a = resolve_axis(s, job, ""); fill_axis1(r, a); fill_cont(r, job);
            r.n_pts = n; r.writable_var = wvar; r.pre_scaller = job.integer("prescaler", 1);
            r.emit_all_samples = job.flag("all_samples", false);
            r.csv_output_path = gui_csv; r.progress = prog.p;
            auto res = eng.run_bifurcation_1d(r);
            ok = res.ok; err = res.error;
            if (ok) {
                auto f = open_csv(file, prec);
                f << a.name << ",flag,value,interval\n";
                size_t np = 0;
                for (int i = 0; i < res.n_pts; ++i) {
                    const double x = node(a.lo, a.hi, res.continuation_reverse ? res.n_pts - 1 - i : i, res.n_pts, a.log);
                    const auto& pts = res.bifurcation_points[i];
                    if (res.flags[i] <= 0 || pts.empty()) { f << x << ',' << res.flags[i] << ",nan,nan\n"; continue; }
                    for (size_t j = 0; j < pts.size(); ++j) {
                        f << x << ',' << res.flags[i] << ',' << pts[j] << ',';
                        if (j < res.peak_times[i].size()) f << res.peak_times[i][j]; else f << "nan";
                        f << '\n'; ++np;
                    }
                }
                summary = std::to_string(res.n_pts) + " points, " + std::to_string(np) + " peaks";
            }
        } else if (task == "lle1d" || task == "ls1d") {
            const bool ls = task == "ls1d";
            Axis a = resolve_axis(s, job, "");
            auto setup = [&](auto& r) {
                fill_common(r, body, s, ic, vals, job); fill_axis1(r, a); fill_cont(r, job);
                r.n_pts = n; r.NT = job.num("nt", 1.0); r.eps = job.num("eps", 1e-4);
                r.vector_transient = job.num("vector_transient", 0.0);
                r.csv_output_path = gui_csv; r.progress = prog.p;
            };
            auto f = std::ofstream();
            if (!ls) {
                LLE1DRequest r; setup(r);
                auto res = eng.run_lle_1d(r); ok = res.ok; err = res.error; cancelled = res.cancelled;
                if (ok) {
                    f = open_csv(file, prec); f << a.name << ",lle,flag\n";
                    for (int i = 0; i < res.n_pts; ++i)
                        f << node(a.lo, a.hi, res.continuation_reverse ? res.n_pts - 1 - i : i, res.n_pts, a.log) << ',' << res.lyapunov[i] << ',' << res.flags[i] << '\n';
                    summary = std::to_string(res.n_pts) + " points";
                }
            } else {
                LS1DRequest r; setup(r);
                auto res = eng.run_ls_1d(r); ok = res.ok; err = res.error; cancelled = res.cancelled;
                if (ok) {
                    const size_t k = res.spectrum.empty() ? 0 : res.spectrum[0].size();
                    f = open_csv(file, prec); f << a.name;
                    for (size_t j = 0; j < k; ++j) f << ",LE" << j + 1;
                    f << ",flag\n";
                    for (int i = 0; i < res.n_pts; ++i) {
                        f << node(a.lo, a.hi, res.continuation_reverse ? res.n_pts - 1 - i : i, res.n_pts, a.log);
                        for (double v : res.spectrum[i]) f << ',' << lyap_value(v);
                        f << ',' << res.flags[i] << '\n';
                    }
                    summary = std::to_string(res.n_pts) + " points, " + std::to_string(k) + " exponents";
                }
            }
        } else if (task == "bif2d" || task == "lle2d" || task == "ls2d") {
            Axis ax = resolve_axis(s, job, ""), ay = resolve_axis(s, job, "_y");
            auto head = [&](std::ofstream& f) { f << ax.name << ',' << ay.name; };
            auto xy = [&](std::ofstream& f, int cell, int np) {
                f << node(ax.lo, ax.hi, cell % np, np, ax.log) << ',' << node(ay.lo, ay.hi, cell / np, np, ay.log);
            };
            if (task == "bif2d") {
                Bifurcation2DRequest r; fill_common(r, body, s, ic, vals, job); fill_axis1(r, ax); fill_axis2(r, ay);
                r.n_pts = n; r.writable_var = wvar; r.pre_scaller = job.integer("prescaler", 1);
                r.eps_dbscan = job.num("eps_dbscan", 0.1);
                r.mult_peak = job.num("mult_peak", r.mult_peak); r.mult_interval = job.num("mult_interval", r.mult_interval);
                r.emit_all_samples = job.flag("all_samples", false);
                r.csv_output_path = gui_csv; r.progress = prog.p;
                auto res = eng.run_bifurcation_2d(r); ok = res.ok; err = res.error; cancelled = res.cancelled;
                if (ok) {
                    auto f = open_csv(file, prec); head(f); f << ",period,flag\n";
                    for (int c = 0; c < res.n_pts * res.n_pts; ++c) { xy(f, c, res.n_pts); f << ',' << res.values[c] << ',' << res.flags[c] << '\n'; }
                }
            } else {
                auto setup = [&](auto& r) {
                    fill_common(r, body, s, ic, vals, job); fill_axis1(r, ax); fill_axis2(r, ay);
                    r.n_pts = n; r.NT = job.num("nt", 1.0); r.eps = job.num("eps", 1e-4);
                    r.vector_transient = job.num("vector_transient", 0.0);
                    r.csv_output_path = gui_csv; r.progress = prog.p;
                };
                if (task == "lle2d") {
                    LLE2DRequest r; setup(r);
                    auto res = eng.run_lle_2d(r); ok = res.ok; err = res.error; cancelled = res.cancelled;
                    if (ok) {
                        auto f = open_csv(file, prec); head(f); f << ",lle,flag\n";
                        for (int c = 0; c < res.n_pts * res.n_pts; ++c) { xy(f, c, res.n_pts); f << ',' << lyap_value(res.values[c]) << ',' << res.flags[c] << '\n'; }
                    }
                } else {
                    LS2DRequest r; setup(r);
                    auto res = eng.run_ls_2d(r); ok = res.ok; err = res.error; cancelled = res.cancelled;
                    if (ok) {
                        const int nn = res.n_pts * res.n_pts;
                        auto f = open_csv(file, prec); head(f);
                        for (int k = 0; k < res.n_exponents; ++k) f << ",LE" << k + 1;
                        f << ",flag\n";
                        for (int c = 0; c < nn; ++c) {
                            xy(f, c, res.n_pts);
                            for (int k = 0; k < res.n_exponents; ++k) f << ',' << lyap_value(res.values[(size_t)k * nn + c]);
                            f << ',' << res.flags[c] << '\n';
                        }
                    }
                }
            }
            summary = std::to_string(n) + "x" + std::to_string(n) + " cells";
        } else if (task == "basins") {
            BasinsRequest r; fill_common(r, body, s, ic, vals, job);
            r.transient_time = job.num("transient", 1000.0);
            r.axis_x_var = var_index(s, job.need("sweep"), "sweep");
            r.axis_y_var = var_index(s, job.need("sweep_y"), "sweep_y");
            auto rx = job.nums("range"), ry = job.nums("range_y");
            if (rx.size() != 2 || ry.size() != 2) throw std::runtime_error("basins: range / range_y must be 'lo, hi'");
            r.axis_x_lo = rx[0]; r.axis_x_hi = rx[1]; r.axis_y_lo = ry[0]; r.axis_y_hi = ry[1];
            r.n_pts = n; r.writable_var = wvar; r.pre_scaller = job.integer("prescaler", 1);
            r.eps_dbscan = job.num("eps_dbscan", 0.5);
            r.feature1 = job.integer("feature1", r.feature1); r.feature2 = job.integer("feature2", r.feature2);
            r.mult1 = (numb)job.num("mult1", (double)r.mult1); r.mult2 = (numb)job.num("mult2", (double)r.mult2);
            r.csv_output_path = gui_csv; r.progress = prog.p;
            auto res = eng.run_basins(r); ok = res.ok; err = res.error; cancelled = res.cancelled;
            if (ok) {
                auto f = open_csv(file, prec);
                f << s.vars[r.axis_x_var] << ',' << s.vars[r.axis_y_var] << ",basin,regime,avg_peak,avg_interval\n";
                for (int c = 0; c < res.n_pts * res.n_pts; ++c)
                    f << node(rx[0], rx[1], c % res.n_pts, res.n_pts, false) << ',' << node(ry[0], ry[1], c / res.n_pts, res.n_pts, false)
                      << ',' << res.basin_idx[c] << ',' << res.helpful_array[c] << ',' << res.avg_peaks[c] << ',' << res.avg_intervals[c] << '\n';
                summary = std::to_string(n) + "x" + std::to_string(n) + " cells, clusters " + std::to_string(res.n_clusters)
                        + " oscillating / " + std::to_string(-res.min_cluster_idx) + " fixed-point";
            }
        } else if (task == "metrics") {
            SignalMetricsRequest r; fill_common(r, body, s, ic, vals, job);
            Axis a = resolve_axis(s, job, ""); fill_axis1(r, a); fill_cont(r, job);
            if (job.has("sweep_y")) { r.dimension = 2; fill_axis2(r, resolve_axis(s, job, "_y")); }
            r.n_pts = n; r.writable_var = wvar; r.pre_scaller = job.integer("prescaler", 1);
            r.metric_mask = job.integer("metric_mask", kSignalMetricAllMask);
            r.minmax_interp = job.flag("minmax_interp", false);
            r.progress = prog.p;
            auto res = eng.run_signal_metrics(r); ok = res.ok; err = res.error; cancelled = res.cancelled;
            if (ok && !signal_metrics_write_csv(res, file, err)) ok = false;
            summary = r.dimension == 2 ? std::to_string(n) + "x" + std::to_string(n) + " cells" : std::to_string(n) + " points";
        } else if (task == "dft1d") {
            Dft1DRequest r; fill_common(r, body, s, ic, vals, job);
            Axis a = resolve_axis(s, job, ""); fill_axis1(r, a); fill_cont(r, job);
            r.n_pts = n; r.writable_var = wvar; r.pre_scaller = job.integer("prescaler", 1);
            r.n_freq = job.integer("n_freq", 200);
            auto fr = job.has("freq_range") ? job.nums("freq_range") : std::vector<double>{ 0.0, 10.0 };
            if (fr.size() != 2) throw std::runtime_error("freq_range must be 'lo, hi'");
            r.freq_lo = fr[0]; r.freq_hi = fr[1];
            r.freq_log_scale = job.flag("freq_log", false);
            r.window_type = job.integer("window", 1);
            r.csv_output_path = gui_csv; r.progress = prog.p;
            auto res = eng.run_dft_1d(r); ok = res.ok; err = res.error; cancelled = res.cancelled;
            if (ok) {
                auto f = open_csv(file, prec);
                f << a.name << ",freq,amplitude,ak_cos,bk_sin,flag\n";
                for (int i = 0; i < res.n_pts; ++i) {
                    const double x = node(a.lo, a.hi, res.continuation_reverse ? res.n_pts - 1 - i : i, res.n_pts, a.log);
                    for (int k = 0; k < res.n_freq; ++k) {
                        const size_t c = (size_t)i * res.n_freq + k;
                        f << x << ',' << node(res.freq_lo, res.freq_hi, k, res.n_freq, r.freq_log_scale) << ','
                          << std::hypot(res.ak_cos[c], res.bk_sin[c]) << ',' << res.ak_cos[c] << ',' << res.bk_sin[c] << ',' << res.flags[i] << '\n';
                    }
                }
                summary = std::to_string(res.n_pts) + " points x " + std::to_string(res.n_freq) + " frequencies";
            }
        } else {
            throw std::runtime_error("unknown task '" + task + "' (bif1d, lle1d, ls1d, bif2d, lle2d, ls2d, basins, metrics, dft1d)");
        }

        const double wall = secs(t0);
        {   // _run.txt: фактический job (с подстановками из командной строки) + итог
            std::ofstream f(out + "_run.txt");
            for (auto& k : job.order) for (auto& v : job.kv[k]) f << k << " = " << v << '\n';
            f << "# scheme_resolved = " << scheme_name << "\n# ok = " << ok << "\n# wall_s = " << wall << '\n';
            if (!err.empty()) f << "# error = " << err << '\n';
        }
        if (!ok) { std::fprintf(stderr, "FAILED %s: %s\n", task.c_str(), err.c_str()); return 1; }
        std::printf("OK %s %s, %.2f s%s -> %s\n", task.c_str(), summary.c_str(), wall, cancelled ? " (partial)" : "", file.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
