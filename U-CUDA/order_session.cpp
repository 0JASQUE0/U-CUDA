#include "order_session.h"
#include "num_parse.h"
#include "krs_cpu.h"      // CPU-ветка считает шаг тем же телом КРС, что уходит в NVRTC
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

namespace {

double parse_d(const std::string& s, double def) { return parse_num(s, def); }
int    parse_i(const std::string& s, int def)    { return parse_num_int(s, def); }

void log_run_completed(const char* label, bool ok, double secs) {
    std::printf("[run] %s %s in %.2fs\n",
                ok ? "Done" : "Cancelled",
                (label && *label) ? label : "(unnamed)",
                secs);
    std::fflush(stdout);
}

// Один axis_*_target -> OrderAxis движка. amountOfValues нужен, чтобы
// свипуемый индекс a[] не уехал за пределы массива при сломанной сессии.
OrderAxis to_engine_axis(int target, const std::string& lo, const std::string& hi,
                         bool log_scale, const std::string& n_text,
                         bool enabled, int amountOfValues)
{
    OrderAxis ax;
    if (!enabled) { ax.kind = OrderAxisKind::None; return ax; }
    if (target == kOrderTargetH) {
        ax.kind  = OrderAxisKind::H;
        ax.index = 0;
    } else {
        ax.kind  = OrderAxisKind::Value;
        ax.index = (target >= 0 && target < amountOfValues) ? target : 0;
    }
    ax.lo        = parse_d(lo, 0.0);
    ax.hi        = parse_d(hi, 1.0);
    if (ax.hi < ax.lo) std::swap(ax.lo, ax.hi);
    ax.log_scale = log_scale;
    ax.n_pts     = std::max(1, parse_i(n_text, 200));
    return ax;
}

OrderRequest build_order_request(const OrderAnalysisSession& s, const OrderConfig& c) {
    OrderRequest req;
    req.krs_body  = compute_krs_for_scheme(s.custom_schemes, s.sys, c.scheme);
    req.amountOfX = (int)s.vars.size();

    req.initial_conditions.resize(req.amountOfX);
    for (int i = 0; i < req.amountOfX; ++i) {
        auto it = c.initial_conditions.find(s.vars[i]);
        req.initial_conditions[(size_t)i] = (it != c.initial_conditions.end()) ? parse_d(it->second, 0.0) : 0.0;
    }

    // a[0] — коэффициент симметрии s, a[1..M] — параметры системы. Ровно та же
    // раскладка, что во всех остальных вкладках (см. build_basins_request).
    const int nparams = (int)s.params.size();
    req.values.assign((size_t)nparams + 1, 0.0);
    req.values[0] = parse_d(c.symmetry_s, 0.5);
    for (int i = 0; i < nparams; ++i) {
        auto it = c.param_values.find(s.params[(size_t)i]);
        req.values[(size_t)i + 1] = (it != c.param_values.end()) ? parse_d(it->second, 0.0) : 0.0;
    }
    const int amountOfValues = (int)req.values.size();

    req.axis_x = to_engine_axis(c.axis_x_target, c.axis_x_lo_text, c.axis_x_hi_text,
                                c.axis_x_log, c.axis_x_n_text, true, amountOfValues);
    // Та же сетка по обеим осям — см. OrderConfig::axis_x_n_text.
    req.axis_y = to_engine_axis(c.axis_y_target, c.axis_y_lo_text, c.axis_y_hi_text,
                                c.axis_y_log, c.axis_x_n_text, c.two_d, amountOfValues);

    req.h             = parse_d(c.h_text, 0.01);
    req.t_max         = parse_d(c.t_max_text, 10.0);
    req.max_value     = parse_d(c.max_value_text, 1.0e6);
    req.snap_steps    = c.snap_steps;
    req.endpoint_only = c.endpoint_only;
    return req;
}

PerfRequest build_perf_request(const OrderAnalysisSession& s, const OrderConfig& c) {
    // Всё общее с Order берётся из того же построителя: расхождение настроек
    // между двумя расчётами одной вкладки было бы худшим из возможных багов —
    // ошибка и время перестали бы относиться к одной и той же задаче.
    const OrderRequest o = build_order_request(s, c);

    PerfRequest req;
    req.krs_body           = o.krs_body;
    req.amountOfX          = o.amountOfX;
    req.initial_conditions = o.initial_conditions;
    req.values             = o.values;
    req.axis               = o.axis_x;
    req.h                  = o.h;
    req.t_max              = o.t_max;
    req.snap_steps         = o.snap_steps;
    req.endpoint_only      = o.endpoint_only;
    req.max_value          = o.max_value;
    req.ref_substeps       = std::max(0, parse_i(c.perf_ref_substeps_text, 4));
    if (req.ref_substeps > 0 && !c.perf_ref_scheme.empty())
        req.ref_krs_body   = compute_krs_for_scheme(s.custom_schemes, s.sys, c.perf_ref_scheme);
    req.repeats            = std::max(1, parse_i(c.perf_repeats_text, 20));
    req.warmup             = std::max(0, parse_i(c.perf_warmup_text, 2));
    req.replicas           = std::max(1, parse_i(c.perf_replicas_text, 1));
    return req;
}

void apply_perf_result(OrderConfig& c, PerfResult&& r) {
    c.perf_result = std::move(r);
    c.perf_last_run_ok = c.perf_result.ok;
    if (!c.perf_result.ok) c.last_error = c.perf_result.error;
    c.perf_data_generation++;
    c.perf_fit_request = true;
}

void apply_order_result(OrderConfig& c, OrderResult&& r) {
    c.result = std::move(r);
    c.last_run_ok = c.result.ok;
    if (!c.result.ok) c.last_error = c.result.error;
    c.data_generation++;
    c.fit_request = true;
}

} // namespace

long long order_steps_for_h(double h, double t_max, bool snap) {
    if (!(h > 0.0) || !(t_max > 0.0)) return 0;
    long long N = snap ? (long long)(t_max / h + 0.5) : (long long)(t_max / h);
    return N < 1 ? 1 : N;
}

void OrderAnalysisSession::load_from_record(const SystemRecord& r,
    const std::vector<std::string>& vars_,
    const std::vector<std::string>& params_)
{
    vars   = vars_;
    params = params_;
    custom_schemes = r.custom_schemes;
    enabled_builtin_schemes = enabled_builtins_from_record(r);

    OrderConfig c;
    c.label      = "Order 1";
    c.h_text     = default_h_from_record(r);
    c.scheme     = default_scheme_from_record(r, c.scheme);
    c.symmetry_s = r.symmetry_s.empty() ? std::string("0.5") : r.symmetry_s;

    for (const auto& p : params) {
        auto it = r.param_values.find(p);
        c.param_values[p] = (it != r.param_values.end()) ? it->second : "";
    }
    for (const auto& v : vars) {
        auto it = r.init_conditions.find(v);
        c.initial_conditions[v] = (it != r.init_conditions.end()) ? it->second : "";
    }

    // Дефолт — диаграмма p(h): именно с неё имеет смысл начинать, она
    // показывает и рабочее окно шага, и обе полки.
    c.axis_x_target = kOrderTargetH;
    // По второй оси по умолчанию предлагаем s (a[0]): у CD/SEMP/SIMP
    // заявленный порядок достигается только при s = 0.5, и карта (h, s) —
    // первое, что хочется увидеть.
    c.axis_y_target = 0;

    configs.clear();
    configs.push_back(std::move(c));
    active_config_index  = 0;
    running_config_index = -1;
}

void OrderAnalysisSession::add_config() {
    OrderConfig c;
    if (!configs.empty()) {
        c = configs.back();
        c.result = OrderResult{};
        c.last_run_ok = false;
        c.last_error.clear();
        c.data_generation = 0;
        c.fit_request = false;
        c.perf_result = PerfResult{};
        c.perf_last_run_ok = false;
        c.perf_data_generation = 0;
        c.perf_fit_request = false;
    } else {
        for (const auto& p : params) c.param_values[p] = "";
        for (const auto& v : vars)   c.initial_conditions[v] = "";
    }
    c.label = "Order " + std::to_string(configs.size() + 1);
    configs.push_back(std::move(c));
    active_config_index = (int)configs.size() - 1;
}

void OrderAnalysisSession::remove_config(int i) {
    if (i < 0 || i >= (int)configs.size()) return;
    // Как в BasinsAnalysisSession: config, чей расчёт идёт, удалять нельзя —
    // poll() положил бы результат в несуществующий слот.
    if (in_flight && running_config_index == i) return;
    configs.erase(configs.begin() + i);
    if (in_flight && running_config_index > i) running_config_index--;
    if (active_config_index >= (int)configs.size())
        active_config_index = (int)configs.size() - 1;
    if (active_config_index < 0) active_config_index = 0;
}

// ---------------------------------------------------------------------------
// CPU-ветка Order / Performance.
//
// Спецификацией служит kernels/order.template.cu: формулы, пороги и коды
// статусов повторены оттуда один в один, иначе две ветки давали бы разные
// картинки на одном и том же входе, и сравнить их было бы нельзя.
//
// Шаг считается ТЕМ ЖЕ текстом КРС, который уходит в NVRTC: тело компилируется
// cl.exe в нативную функцию (krs_cpu.h). Интерпретатор выражений
// (SystemEvaluator) здесь не годится принципиально — на диаграмме
// производительности он мерил бы стоимость интерпретации, а не стоимость
// схемы, а это разные величины, отличающиеся на порядок.
//
// Реплик на CPU нет: прогоны последовательные, и измеряется время ОДНОЙ
// траектории. На GPU replicas меняют смысл замера с латентности на пропускную
// способность; здесь такого выбора нет, поле просто игнорируется.

// Тот же предикат «улетела», что orderBadVec в ядре.
static bool order_bad_vec_cpu(const numb* X, int n, numb maxValue) {
    numb acc = (numb)0;
    for (int i = 0; i < n; ++i) acc += std::fabs(X[i]);
    if (std::isnan(acc) || std::isinf(acc)) return true;
    if (maxValue != (numb)0 && acc > maxValue) return true;
    return false;
}

// Узлы оси — копия order_axis_nodes из parametric_engine.cpp.
static bool order_axis_nodes_cpu(const OrderAxis& ax, const char* what,
                                 std::vector<double>& out, std::string& err) {
    const int n = (ax.kind == OrderAxisKind::None) ? 1 : ax.n_pts;
    if (n <= 0)     { err = std::string(what) + ": the number of points must be > 0"; return false; }
    if (n > 100000) { err = std::string(what) + ": the number of points is too large"; return false; }
    out.assign((size_t)n, 0.0);
    if (ax.kind == OrderAxisKind::None) { out[0] = 0.0; return true; }
    if (n == 1) { out[0] = ax.lo; return true; }
    if (ax.log_scale) {
        if (!(ax.lo > 0.0) || !(ax.hi > 0.0)) {
            err = std::string(what) + ": log scale requires both bounds > 0";
            return false;
        }
        const double k = std::log(ax.hi / ax.lo) / (double)(n - 1);
        for (int i = 0; i < n; ++i) out[(size_t)i] = ax.lo * std::exp(k * (double)i);
    } else {
        const double d = (ax.hi - ax.lo) / (double)(n - 1);
        for (int i = 0; i < n; ++i) out[(size_t)i] = ax.lo + d * (double)i;
    }
    return true;
}

// Число грубых шагов ячейки — та же формула, что в ядре и на хосте GPU-ветки.
static long long order_steps_for_cpu(double h, double tMax, bool snap) {
    if (!(h > 0.0)) return 1;
    long long N = snap ? (long long)(tMax / h + 0.5) : (long long)(tMax / h);
    return N < 1 ? 1 : N;
}

// Компиляция тела (и эталона) в нативные шаги. Ошибки компилятора приходят с
// номерами строк В ТЕЛЕ КРС, поэтому их видно так же, как в редакторе схем.
static bool compile_cpu_step(const std::string& body, int amountOfX, int amountOfValues,
                             const char* what, KrsCpuStep& out, std::string& err) {
    std::vector<KrsCpuDiag> diags;
    if (out.compile(body, amountOfX, amountOfValues, diags)) return true;
    err = std::string("CPU ") + what + ":";
    for (const auto& d : diags) {
        err += "\n";
        if (d.line > 0) err += "line " + std::to_string(d.line) + ": ";
        err += d.message;
    }
    return false;
}

static OrderResult run_order_cpu(const OrderRequest& req) {
    OrderResult res;
    res.axis_x = req.axis_x;
    res.axis_y = req.axis_y;
    auto fail = [&](const std::string& msg) -> OrderResult { res.error = msg; return res; };

    std::string why;
    if (!krs_cpu_backend_available(&why))
        return fail("CPU backend is unavailable: " + why);

    std::string err;
    if (!order_axis_nodes_cpu(req.axis_x, "X axis", res.axis_x_vals, err)) return fail(err);
    if (!order_axis_nodes_cpu(req.axis_y, "Y axis", res.axis_y_vals, err)) return fail(err);
    res.n_pts_x = (int)res.axis_x_vals.size();
    res.n_pts_y = (int)res.axis_y_vals.size();
    const size_t total_cells = (size_t)res.n_pts_x * (size_t)res.n_pts_y;
    if (total_cells == 0) return fail("empty grid");

    const int n = req.amountOfX;
    if (n < 1) return fail("the system is empty");
    if ((int)req.initial_conditions.size() < n) return fail("not enough initial conditions");
    const int amountOfValues = (int)req.values.size();
    if (amountOfValues < 1) return fail("the values array is empty");

    const bool ref_on = !req.ref_krs_body.empty() && req.ref_substeps > 0;
    KrsCpuStep step_main, step_ref;
    if (!compile_cpu_step(req.krs_body, n, amountOfValues, "KRS", step_main, err))
        return fail(err);
    if (ref_on && !compile_cpu_step(req.ref_krs_body, n, amountOfValues, "reference KRS", step_ref, err))
        return fail(err);
    const KrsCpuStep::StepFn fmain = step_main.fn();
    const KrsCpuStep::StepFn fref  = ref_on ? step_ref.fn() : nullptr;

    res.p.assign(total_cells, 0.0);
    res.e1.assign(total_cells, 0.0);
    res.e2.assign(total_cells, 0.0);
    res.e_ref.assign(total_cells, std::numeric_limits<double>::quiet_NaN());
    res.h_eff.assign(total_cells, 0.0);
    res.status.assign(total_cells, ORDER_ST_OK);

    // Полная работа — для прогресса: при свипе по h число шагов отличается от
    // узла к узлу на порядки, и «доля посчитанных ячеек» давала бы бар, который
    // стоит, а потом прыгает в конец.
    auto cell_h = [&](int ix, int iy) -> double {
        double h = req.h;
        if (req.axis_x.kind == OrderAxisKind::H) h = res.axis_x_vals[(size_t)ix];
        if (req.axis_y.kind == OrderAxisKind::H) h = res.axis_y_vals[(size_t)iy];
        return h;
    };
    double total_steps = 0.0;
    for (int iy = 0; iy < res.n_pts_y; ++iy)
        for (int ix = 0; ix < res.n_pts_x; ++ix)
            total_steps += (double)order_steps_for_cpu(cell_h(ix, iy), req.t_max, req.snap_steps);
    if (total_steps > 1.0e13)
        return fail("t_max / h too large for the CPU: the work does not fit into a reasonable time");
    if (total_steps <= 0.0) total_steps = 1.0;
    double done_steps = 0.0;

    const numb floorEps = (numb)2 * std::numeric_limits<numb>::epsilon();
    const numb maxValue = (numb)req.max_value;

    std::vector<numb> a((size_t)amountOfValues);
    std::vector<numb> Xc((size_t)n), Xm((size_t)n), Xf((size_t)n), Xr((size_t)n);

    for (int iy = 0; iy < res.n_pts_y; ++iy) {
        for (int ix = 0; ix < res.n_pts_x; ++ix) {
            const size_t gcell = (size_t)iy * (size_t)res.n_pts_x + (size_t)ix;

            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
                res.cancelled = true;
                return res;
            }

            for (int i = 0; i < amountOfValues; ++i) a[(size_t)i] = (numb)req.values[(size_t)i];
            double h = req.h;
            const double vx = res.axis_x_vals[(size_t)ix];
            if      (req.axis_x.kind == OrderAxisKind::H)     h = vx;
            else if (req.axis_x.kind == OrderAxisKind::Value) {
                if (req.axis_x.index >= 0 && req.axis_x.index < amountOfValues)
                    a[(size_t)req.axis_x.index] = (numb)vx;
            }
            if (req.axis_y.kind != OrderAxisKind::None) {
                const double vy = res.axis_y_vals[(size_t)iy];
                if      (req.axis_y.kind == OrderAxisKind::H)     h = vy;
                else if (req.axis_y.kind == OrderAxisKind::Value) {
                    if (req.axis_y.index >= 0 && req.axis_y.index < amountOfValues)
                        a[(size_t)req.axis_y.index] = (numb)vy;
                }
            }

            if (!(h > 0.0) || std::isnan(h) || std::isinf(h)) {
                res.p[gcell] = res.e1[gcell] = res.e2[gcell]
                             = std::numeric_limits<double>::quiet_NaN();
                res.h_eff[gcell]  = h;
                res.status[gcell] = ORDER_ST_DIVERGED;
                continue;
            }

            long long N;
            numb hEff = (numb)h;
            if (req.snap_steps) {
                N = (long long)(req.t_max / h + 0.5);
                if (N < 1) N = 1;
                hEff = (numb)req.t_max / (numb)N;
            } else {
                N = (long long)(req.t_max / h);
                if (N < 1) N = 1;
            }

            const numb h1 = hEff;
            const numb h2 = hEff / (numb)2;
            const numb h4 = hEff / (numb)4;

            for (int i = 0; i < n; ++i) {
                const numb x0 = (numb)req.initial_conditions[(size_t)i];
                Xc[(size_t)i] = x0; Xm[(size_t)i] = x0; Xf[(size_t)i] = x0; Xr[(size_t)i] = x0;
            }

            numb e1 = (numb)0, e2 = (numb)0, scale = (numb)0, eRef = (numb)0;
            int  status = ORDER_ST_OK;

            const int  refM = (req.ref_substeps > 0) ? req.ref_substeps : 1;
            const numb hRef = h1 / (numb)refM;

            bool cancelled_here = false;
            for (long long k = 0; k < N; ++k) {
                fmain(Xc.data(), a.data(), h1);
                fmain(Xm.data(), a.data(), h2);
                fmain(Xm.data(), a.data(), h2);
                fmain(Xf.data(), a.data(), h4);
                fmain(Xf.data(), a.data(), h4);
                fmain(Xf.data(), a.data(), h4);
                fmain(Xf.data(), a.data(), h4);
                if (fref)
                    for (int q = 0; q < refM; ++q) fref(Xr.data(), a.data(), hRef);

                const bool last = (k == N - 1);
                if (!req.endpoint_only || last) {
                    for (int i = 0; i < n; ++i) {
                        const numb d1 = std::fabs(Xc[(size_t)i] - Xm[(size_t)i]);
                        const numb d2 = std::fabs(Xm[(size_t)i] - Xf[(size_t)i]);
                        if (d1 > e1) e1 = d1;
                        if (d2 > e2) e2 = d2;
                        const numb sc = std::fabs(Xf[(size_t)i]);
                        if (sc > scale) scale = sc;
                        if (fref) {
                            const numb dr = std::fabs(Xc[(size_t)i] - Xr[(size_t)i]);
                            if (dr > eRef) eRef = dr;
                        }
                    }
                }

                if ((k % (long long)CHECK_INTERVAL) == 0 || last) {
                    if (order_bad_vec_cpu(Xc.data(), n, maxValue) ||
                        order_bad_vec_cpu(Xm.data(), n, maxValue) ||
                        order_bad_vec_cpu(Xf.data(), n, maxValue) ||
                        (fref && order_bad_vec_cpu(Xr.data(), n, maxValue))) {
                        status = ORDER_ST_DIVERGED;
                        done_steps += (double)(N - k);
                        break;
                    }
                    if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
                        cancelled_here = true;
                        break;
                    }
                    if (req.progress) {
                        req.progress->store((float)(done_steps / total_steps),
                                            std::memory_order_relaxed);
                    }
                }
                done_steps += 1.0;
            }
            if (cancelled_here) { res.cancelled = true; return res; }

            double p;
            double o1 = (double)e1, o2 = (double)e2;
            if (status == ORDER_ST_DIVERGED) {
                p = o1 = o2 = std::numeric_limits<double>::quiet_NaN();
            } else {
                // Полка округления — тот же порог, что в ядре: он растёт как
                // sqrt(числа шагов), потому что шум вычитания копится
                // случайным блужданием, а не держится в пределах пары ulp.
                const numb nsteps = (numb)4 * (numb)N;
                const numb tiny = floorEps * (scale > (numb)0 ? scale : (numb)1) * std::sqrt(nsteps);
                if (e2 <= tiny || e1 <= tiny) status = ORDER_ST_FLOOR;
                else if (e2 >= e1)            status = ORDER_ST_NOCONTRACT;
                p = (e2 > (numb)0 && e1 > (numb)0)
                        ? std::log2((double)(e1 / e2))
                        : std::numeric_limits<double>::quiet_NaN();
            }

            res.p[gcell]      = p;
            res.e1[gcell]     = o1;
            res.e2[gcell]     = o2;
            res.e_ref[gcell]  = fref ? ((status == ORDER_ST_DIVERGED)
                                            ? std::numeric_limits<double>::quiet_NaN()
                                            : (double)eRef)
                                     : std::numeric_limits<double>::quiet_NaN();
            res.h_eff[gcell]  = (double)hEff;
            res.status[gcell] = status;
        }
    }

    bool first_p = true, first_e = true, first_er = true;
    for (size_t i = 0; i < total_cells; ++i) {
        switch (res.status[i]) {
            case ORDER_ST_DIVERGED:   ++res.n_diverged;   break;
            case ORDER_ST_FLOOR:      ++res.n_floor;      break;
            case ORDER_ST_NOCONTRACT: ++res.n_nocontract; break;
            default:                  ++res.n_ok;         break;
        }
        if (res.status[i] != ORDER_ST_OK) continue;
        if (std::isfinite(res.p[i])) {
            if (first_p) { res.p_min = res.p_max = res.p[i]; first_p = false; }
            else { if (res.p[i] < res.p_min) res.p_min = res.p[i];
                   if (res.p[i] > res.p_max) res.p_max = res.p[i]; }
        }
        if (std::isfinite(res.e1[i]) && res.e1[i] > 0.0) {
            if (first_e) { res.e1_min = res.e1_max = res.e1[i]; first_e = false; }
            else { if (res.e1[i] < res.e1_min) res.e1_min = res.e1[i];
                   if (res.e1[i] > res.e1_max) res.e1_max = res.e1[i]; }
        }
        if (std::isfinite(res.e_ref[i]) && res.e_ref[i] > 0.0) {
            if (first_er) { res.eref_min = res.eref_max = res.e_ref[i]; first_er = false; }
            else { if (res.e_ref[i] < res.eref_min) res.eref_min = res.e_ref[i];
                   if (res.e_ref[i] > res.eref_max) res.eref_max = res.e_ref[i]; }
        }
    }
    if (req.progress) req.progress->store(1.0f, std::memory_order_relaxed);
    res.ok = true;
    return res;
}

static PerfResult run_performance_cpu(const PerfRequest& req) {
    PerfResult res;
    res.axis = req.axis;
    auto fail = [&](const std::string& msg) -> PerfResult { res.error = msg; return res; };

    if (req.axis.kind == OrderAxisKind::None) return fail("the axis is not set");
    if (req.repeats < 1) return fail("the number of measurements must be >= 1");
    if (req.warmup  < 0) return fail("the number of warmup runs cannot be negative");

    // ---- Проход 1: ошибки (не засекается), та же сетка ----
    OrderRequest oreq;
    oreq.krs_body           = req.krs_body;
    oreq.amountOfX          = req.amountOfX;
    oreq.initial_conditions = req.initial_conditions;
    oreq.values             = req.values;
    oreq.axis_x             = req.axis;
    oreq.axis_y             = OrderAxis{};
    oreq.h                  = req.h;
    oreq.t_max              = req.t_max;
    oreq.snap_steps         = req.snap_steps;
    oreq.endpoint_only      = req.endpoint_only;
    oreq.max_value          = req.max_value;
    oreq.ref_krs_body       = req.ref_krs_body;
    oreq.ref_substeps       = req.ref_substeps;
    oreq.cancel             = req.cancel;
    // Прогресс первого прохода наружу не отдаём: второй проход перезапишет
    // его с нуля, и бар дёргался бы назад. Своя шкала ставится ниже.

    const OrderResult ores = run_order_cpu(oreq);
    if (ores.cancelled) { res.cancelled = true; return res; }
    if (!ores.ok) return fail(ores.error);

    res.n_pts     = ores.n_pts_x;
    res.axis_vals = ores.axis_x_vals;
    res.e1        = ores.e1;
    res.e2        = ores.e2;
    res.e_ref     = ores.e_ref;
    res.p         = ores.p;
    res.h_eff     = ores.h_eff;
    res.status    = ores.status;
    res.n_ok         = ores.n_ok;
    res.n_diverged   = ores.n_diverged;
    res.n_floor      = ores.n_floor;
    res.n_nocontract = ores.n_nocontract;
    res.e1_min = ores.e1_min;   res.e1_max = ores.e1_max;
    res.eref_min = ores.eref_min; res.eref_max = ores.eref_max;
    res.repeats  = req.repeats;
    res.warmup   = req.warmup;
    res.replicas = 1;           // последовательный прогон: реплик на CPU нет

    const int n = req.amountOfX;
    const int amountOfValues = (int)req.values.size();
    const int npts = res.n_pts;
    res.t_min.assign((size_t)npts, std::numeric_limits<double>::quiet_NaN());
    res.t_max.assign((size_t)npts, std::numeric_limits<double>::quiet_NaN());
    res.t_avg.assign((size_t)npts, std::numeric_limits<double>::quiet_NaN());
    res.n_steps.assign((size_t)npts, 0);

    std::string err;
    KrsCpuStep step;
    if (!compile_cpu_step(req.krs_body, n, amountOfValues, "KRS", step, err)) return fail(err);
    const KrsCpuStep::StepFn fstep = step.fn();

    std::vector<numb> a((size_t)amountOfValues);
    std::vector<numb> X((size_t)n);
    // Сток для конечного состояния: тело КРС живёт в DLL, выбросить цикл
    // компилятор не может, но пусть и формально результат кто-то читает.
    volatile double sink = 0.0;

    bool first_t = true;
    for (int i = 0; i < npts; ++i) {
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true;
            return res;
        }
        if (req.progress)
            req.progress->store((float)i / (float)(npts > 0 ? npts : 1),
                                std::memory_order_relaxed);

        const double h_node = res.h_eff[(size_t)i];
        if (!(h_node > 0.0) || !std::isfinite(h_node)) continue;   // узел без времени: см. status

        for (int k = 0; k < amountOfValues; ++k) a[(size_t)k] = (numb)req.values[(size_t)k];
        if (req.axis.kind == OrderAxisKind::Value) {
            const int vi = req.axis.index;
            if (vi >= 0 && vi < amountOfValues) a[(size_t)vi] = (numb)res.axis_vals[(size_t)i];
        }

        const long long N = order_steps_for_cpu(h_node, req.t_max, req.snap_steps);
        res.n_steps[(size_t)i] = N;

        auto one_run = [&]() {
            for (int k = 0; k < n; ++k) X[(size_t)k] = (numb)req.initial_conditions[(size_t)k];
            const numb hh = (numb)h_node;
            for (long long k = 0; k < N; ++k) fstep(X.data(), a.data(), hh);
            double acc = 0.0;
            for (int k = 0; k < n; ++k) acc += (double)X[(size_t)k];
            sink = acc;
        };

        for (int w = 0; w < req.warmup; ++w) one_run();

        double tmin = 0.0, tmx = 0.0, tsum = 0.0;
        int got = 0;
        for (int rep = 0; rep < req.repeats; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            one_run();
            const auto t1 = std::chrono::steady_clock::now();
            const double us =
                std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(t1 - t0).count();
            if (got == 0) { tmin = tmx = us; }
            else { if (us < tmin) tmin = us; if (us > tmx) tmx = us; }
            tsum += us;
            ++got;
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
                res.cancelled = true;
                return res;
            }
        }
        if (got > 0) {
            res.t_min[(size_t)i] = tmin;
            res.t_max[(size_t)i] = tmx;
            res.t_avg[(size_t)i] = tsum / (double)got;
            if (first_t) { res.t_lo = tmin; res.t_hi = tmx; first_t = false; }
            else { if (tmin < res.t_lo) res.t_lo = tmin; if (tmx > res.t_hi) res.t_hi = tmx; }
        }
    }

    (void)sink;
    if (req.progress) req.progress->store(1.0f, std::memory_order_relaxed);
    res.ok = true;
    return res;
}

bool OrderAnalysisSession::run_async(ParametricEngine& engine, int config_idx) {
    if (in_flight) return false;
    if (config_idx < 0 || config_idx >= (int)configs.size()) return false;

    OrderConfig& c = configs[(size_t)config_idx];
    c.last_error.clear();
    last_run_label.clear();
    cancel_token   = std::make_shared<std::atomic<bool>>(false);
    progress_token = std::make_shared<std::atomic<float>>(0.0f);

    if (c.calc_kind == kOrderCalcPerf) {
        PerfRequest preq = build_perf_request(*this, c);
        if (preq.krs_body.empty()) {
            c.last_error = "krs_code is empty (no valid system or scheme)";
            cancel_token.reset();
            progress_token.reset();
            return false;
        }
        preq.cancel   = cancel_token;
        preq.progress = progress_token;

        in_flight = true;
        running_is_perf = true;
        running_config_index = config_idx;
        compute_start_time = std::chrono::steady_clock::now();
        const bool on_gpu = c.use_gpu;
        perf_future = std::async(std::launch::async, [&engine, on_gpu, preq = std::move(preq)]() {
            return on_gpu ? engine.run_performance(preq) : run_performance_cpu(preq);
        });
        return true;
    }

    OrderRequest req = build_order_request(*this, c);
    if (req.krs_body.empty()) {
        c.last_error = "krs_code is empty (no valid system or scheme)";
        cancel_token.reset();
        progress_token.reset();
        return false;
    }
    req.cancel   = cancel_token;
    req.progress = progress_token;

    in_flight = true;
    running_is_perf = false;
    running_config_index = config_idx;
    compute_start_time = std::chrono::steady_clock::now();
    const bool on_gpu = c.use_gpu;
    run_future = std::async(std::launch::async, [&engine, on_gpu, req = std::move(req)]() {
        return on_gpu ? engine.run_order(req) : run_order_cpu(req);
    });
    return true;
}

void OrderAnalysisSession::request_cancel() {
    if (cancel_token) cancel_token->store(true, std::memory_order_relaxed);
}

bool OrderAnalysisSession::poll() {
    if (!in_flight) return false;
    std::future<OrderResult>& of = run_future;
    std::future<PerfResult>&  pf = perf_future;
    const bool is_perf = running_is_perf;
    if (is_perf ? !pf.valid() : !of.valid()) {
        in_flight = false;
        running_is_perf = false;
        running_config_index = -1;
        cancel_token.reset(); progress_token.reset();
        return false;
    }
    if ((is_perf ? pf.wait_for(std::chrono::seconds(0)) : of.wait_for(std::chrono::seconds(0)))
        != std::future_status::ready) return false;

    const int  idx = running_config_index;
    std::string label = (idx >= 0 && idx < (int)configs.size() && !configs[(size_t)idx].label.empty())
                            ? configs[(size_t)idx].label : std::string("order");
    bool cancelled = false;
    if (is_perf) {
        PerfResult r = pf.get();
        cancelled = r.cancelled;
        if (!cancelled && idx >= 0 && idx < (int)configs.size())
            apply_perf_result(configs[(size_t)idx], std::move(r));
    } else {
        OrderResult r = of.get();
        cancelled = r.cancelled;
        if (!cancelled && idx >= 0 && idx < (int)configs.size())
            apply_order_result(configs[(size_t)idx], std::move(r));
    }

    last_run_completed_at = std::chrono::steady_clock::now();
    last_run_seconds = std::chrono::duration<double>(last_run_completed_at - compute_start_time).count();
    last_run_label = label;
    last_run_succeeded = !cancelled;
    log_run_completed(last_run_label.c_str(), last_run_succeeded, last_run_seconds);

    in_flight = false;
    running_is_perf = false;
    running_config_index = -1;
    cancel_token.reset();
    progress_token.reset();
    return true;
}

std::vector<std::string> OrderAnalysisSession::axis_target_names() const {
    std::vector<std::string> out;
    out.reserve(params.size() + 2);
    out.push_back("h (step)");
    out.push_back("s (a[0])");
    for (size_t i = 0; i < params.size(); ++i)
        out.push_back(params[i] + " (a[" + std::to_string(i + 1) + "])");
    return out;
}

std::string OrderAnalysisSession::axis_target_label(int target) const {
    if (target == kOrderTargetH) return "h";
    if (target == 0)             return "s";
    const size_t i = (size_t)(target - 1);
    if (i < params.size())       return params[i];
    return "a[" + std::to_string(target) + "]";
}
