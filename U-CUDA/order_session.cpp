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

// Неявные схемы решают шаг Ньютоном и останавливаются по ||dX|| < newton_tol.
// Системная настройка (по умолчанию 1e-10) стала бы потолком РАНЬШЕ арифметики:
// в dd полка округления лежит на 1e-29, то есть допуск съел бы девятнадцать
// порядков и расширенная точность не дала бы ничего. На этой вкладке меряют
// порядок СХЕМЫ, а не точность решателя, поэтому допуск подтягивается под
// выбранную арифметику — но только в сторону ужесточения: если пользователь
// поставил ещё туже, его значение остаётся.
//
// Итераций Ньютон требует немногим больше: сходимость квадратичная, каждый шаг
// удваивает разряды, и с типовых 1e-3 до 1e-58 хватает шести.
System order_newton_system(const System& sys, int prec) {
    const double tol = order_newton_tol_for_precision(prec);
    if (!(tol > 0.0)) return sys;
    System out = sys;
    if (!(out.newton_tol > 0.0) || out.newton_tol > tol) out.newton_tol = tol;
    if (out.newton_max_iters < 12) out.newton_max_iters = 12;
    return out;
}

OrderRequest build_order_request(const OrderAnalysisSession& s, const OrderConfig& c) {
    OrderRequest req;
    // На GPU расширенной точности нет, и молча считать в double при выбранном
    // dd было бы хуже, чем не дать выбрать: цифры на графике те же, а смысл
    // другой. Поэтому UI гасит селектор при GPU, а здесь режим снимается.
    req.cpu_prec  = c.use_gpu ? kOrderPrecDouble : c.cpu_precision;
    req.krs_body  = compute_krs_for_scheme(s.custom_schemes,
                                           order_newton_system(s.sys, req.cpu_prec),
                                           c.scheme);
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
    req.cpu_prec           = o.cpu_prec;
    req.ref_substeps       = std::max(0, parse_i(c.perf_ref_substeps_text, 4));
    // Эталон собирается с тем же допуском Ньютона, что и испытуемая схема:
    // иначе неявный эталон упёрся бы в 1e-10 и E_ref мерил бы его решатель.
    if (req.ref_substeps > 0 && !c.perf_ref_scheme.empty())
        req.ref_krs_body   = compute_krs_for_scheme(s.custom_schemes,
                                                    order_newton_system(s.sys, req.cpu_prec),
                                                    c.perf_ref_scheme);
    req.repeats            = std::max(1, parse_i(c.perf_repeats_text, 20));
    req.warmup             = std::max(0, parse_i(c.perf_warmup_text, 2));
    req.replicas           = std::max(1, parse_i(c.perf_replicas_text, 1));
    return req;
}

StabilityRequest build_stability_request(const OrderAnalysisSession& s, const OrderConfig& c) {
    // Всё общее с Order снова берётся из одного построителя: схема, допуск
    // Ньютона под выбранную точность и раскладка a[] обязаны совпадать с
    // остальными расчётами вкладки.
    const OrderRequest o = build_order_request(s, c);

    StabilityRequest req;
    req.krs_body  = o.krs_body;
    req.amountOfX = o.amountOfX;
    req.values    = o.values;
    req.cpu_prec  = o.cpu_prec;

    req.idx_a = c.stab_idx_a; req.idx_b = c.stab_idx_b;
    req.idx_c = c.stab_idx_c; req.idx_d = c.stab_idx_d;

    req.k = parse_d(c.stab_k_text, 1.0);
    req.r = parse_d(c.stab_r_text, -1.0);
    req.h = parse_d(c.stab_h_text, 1.0);

    req.sigma_lo = parse_d(c.stab_sig_lo_text, -4.1);
    req.sigma_hi = parse_d(c.stab_sig_hi_text,  0.1);
    req.omega_lo = parse_d(c.stab_om_lo_text,  -2.75);
    req.omega_hi = parse_d(c.stab_om_hi_text,   2.65);
    if (req.sigma_hi < req.sigma_lo) std::swap(req.sigma_lo, req.sigma_hi);
    if (req.omega_hi < req.omega_lo) std::swap(req.omega_lo, req.omega_hi);
    req.n_pts = std::max(1, parse_i(c.stab_n_text, 256));
    return req;
}

void apply_stability_result(OrderConfig& c, StabilityResult&& r) {
    c.stab_result = std::move(r);
    c.stab_last_run_ok = c.stab_result.ok;
    if (!c.stab_result.ok) c.last_error = c.stab_result.error;
    c.stab_data_generation++;
    c.stab_fit_request = true;
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

double order_newton_tol_for_precision(int prec) {
    switch (prec) {
        case kOrderPrecQD: return 1.0e-58;
        case kOrderPrecDD: return 1.0e-28;
        default:           return 0.0;
    }
}

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
        c.stab_result = StabilityResult{};
        c.stab_last_run_ok = false;
        c.stab_data_generation = 0;
        c.stab_fit_request = false;
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

// Точность CPU-ветки — параметр шаблона. Скаляров два: numb (= double, бит в
// бит как на GPU) и ucuda::dd (~32 цифры, kernels/ucuda_hp.h). Расширенная
// точность нужна ровно из-за полки округления в оценке порядка: при eps =
// 2.2e-16 разность E1 = |y_h - y_h/2| для схемы порядка 8 уходит под шум
// раньше, чем схема выйдет на асимптотику, и p не измеряется вовсе.
//
// Различий между скалярами ровно два — код точности для компилятора КРС и
// сигнатура готового шага (в dd шаг h передаётся указателем, см. StepFnDD).
// Всё остальное пишется одним текстом.
template <class S> struct CpuScalarTraits;

template <> struct CpuScalarTraits<numb> {
    using Fn = KrsCpuStep::StepFn;
    static constexpr KrsCpuPrec prec = KrsCpuPrec::Double;
    static Fn   fn(const KrsCpuStep& s) { return s.fn(); }
    static void call(Fn f, numb* X, const numb* a, const numb& h) { f(X, a, h); }
};

template <> struct CpuScalarTraits<ucuda::dd> {
    using Fn = KrsCpuStep::StepFnDD;
    static constexpr KrsCpuPrec prec = KrsCpuPrec::DD;
    static Fn   fn(const KrsCpuStep& s) { return s.fn_dd(); }
    static void call(Fn f, ucuda::dd* X, const ucuda::dd* a, const ucuda::dd& h) { f(X, a, &h); }
};

template <> struct CpuScalarTraits<ucuda::qd> {
    using Fn = KrsCpuStep::StepFnQD;
    static constexpr KrsCpuPrec prec = KrsCpuPrec::QD;
    static Fn   fn(const KrsCpuStep& s) { return s.fn_qd(); }
    static void call(Fn f, ucuda::qd* X, const ucuda::qd* a, const ucuda::qd& h) { f(X, a, &h); }
};

// Результат наружу всегда double: p, e1, e2 — это логарифмы отношений, лишние
// разряды в них смысла не несут, а график и CSV везде работают с double.
static double as_d(numb x)             { return x; }
static double as_d(const ucuda::dd& x) { return x.hi; }
static double as_d(const ucuda::qd& x) { return x.x[0]; }

// Тот же предикат «улетела», что orderBadVec в ядре.
template <class S>
static bool order_bad_vec_cpu(const S* X, int n, const S& maxValue) {
    using std::fabs; using std::isnan; using std::isinf;
    S acc = S(0);
    for (int i = 0; i < n; ++i) acc += fabs(X[i]);
    if (isnan(acc) || isinf(acc)) return true;
    if (maxValue != S(0) && acc > maxValue) return true;
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
template <class S>
static bool compile_cpu_step(const std::string& body, int amountOfX, int amountOfValues,
                             const char* what, KrsCpuStep& out, std::string& err) {
    std::vector<KrsCpuDiag> diags;
    if (out.compile(body, amountOfX, amountOfValues, CpuScalarTraits<S>::prec, diags)) return true;
    err = std::string("CPU ") + what + ":";
    for (const auto& d : diags) {
        err += "\n";
        if (d.line > 0) err += "line " + std::to_string(d.line) + ": ";
        err += d.message;
    }
    return false;
}

template <class S>
static OrderResult run_order_cpu_t(const OrderRequest& req) {
    using Tr = CpuScalarTraits<S>;
    using std::fabs; using std::sqrt;
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
    if (!compile_cpu_step<S>(req.krs_body, n, amountOfValues, "KRS", step_main, err))
        return fail(err);
    if (ref_on && !compile_cpu_step<S>(req.ref_krs_body, n, amountOfValues, "reference KRS", step_ref, err))
        return fail(err);
    const typename Tr::Fn fmain = Tr::fn(step_main);
    const typename Tr::Fn fref  = ref_on ? Tr::fn(step_ref) : nullptr;

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

    const S floorEps = S(2) * std::numeric_limits<S>::epsilon();
    const S maxValue = S(req.max_value);

    std::vector<S> a((size_t)amountOfValues);
    std::vector<S> Xc((size_t)n), Xm((size_t)n), Xf((size_t)n), Xr((size_t)n);

    for (int iy = 0; iy < res.n_pts_y; ++iy) {
        for (int ix = 0; ix < res.n_pts_x; ++ix) {
            const size_t gcell = (size_t)iy * (size_t)res.n_pts_x + (size_t)ix;

            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
                res.cancelled = true;
                return res;
            }

            for (int i = 0; i < amountOfValues; ++i) a[(size_t)i] = S(req.values[(size_t)i]);
            double h = req.h;
            const double vx = res.axis_x_vals[(size_t)ix];
            if      (req.axis_x.kind == OrderAxisKind::H)     h = vx;
            else if (req.axis_x.kind == OrderAxisKind::Value) {
                if (req.axis_x.index >= 0 && req.axis_x.index < amountOfValues)
                    a[(size_t)req.axis_x.index] = S(vx);
            }
            if (req.axis_y.kind != OrderAxisKind::None) {
                const double vy = res.axis_y_vals[(size_t)iy];
                if      (req.axis_y.kind == OrderAxisKind::H)     h = vy;
                else if (req.axis_y.kind == OrderAxisKind::Value) {
                    if (req.axis_y.index >= 0 && req.axis_y.index < amountOfValues)
                        a[(size_t)req.axis_y.index] = S(vy);
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
            S hEff = S(h);
            if (req.snap_steps) {
                N = (long long)(req.t_max / h + 0.5);
                if (N < 1) N = 1;
                hEff = S(req.t_max) / S((double)N);
            } else {
                N = (long long)(req.t_max / h);
                if (N < 1) N = 1;
            }

            const S h1 = hEff;
            const S h2 = hEff / S(2);
            const S h4 = hEff / S(4);

            for (int i = 0; i < n; ++i) {
                const S x0 = S(req.initial_conditions[(size_t)i]);
                Xc[(size_t)i] = x0; Xm[(size_t)i] = x0; Xf[(size_t)i] = x0; Xr[(size_t)i] = x0;
            }

            S   e1 = S(0), e2 = S(0), scale = S(0), eRef = S(0);
            int status = ORDER_ST_OK;

            const int refM = (req.ref_substeps > 0) ? req.ref_substeps : 1;
            const S   hRef = h1 / S(refM);

            bool cancelled_here = false;
            for (long long k = 0; k < N; ++k) {
                Tr::call(fmain, Xc.data(), a.data(), h1);
                Tr::call(fmain, Xm.data(), a.data(), h2);
                Tr::call(fmain, Xm.data(), a.data(), h2);
                Tr::call(fmain, Xf.data(), a.data(), h4);
                Tr::call(fmain, Xf.data(), a.data(), h4);
                Tr::call(fmain, Xf.data(), a.data(), h4);
                Tr::call(fmain, Xf.data(), a.data(), h4);
                if (fref)
                    for (int q = 0; q < refM; ++q) Tr::call(fref, Xr.data(), a.data(), hRef);

                const bool last = (k == N - 1);
                if (!req.endpoint_only || last) {
                    for (int i = 0; i < n; ++i) {
                        const S d1 = fabs(Xc[(size_t)i] - Xm[(size_t)i]);
                        const S d2 = fabs(Xm[(size_t)i] - Xf[(size_t)i]);
                        if (d1 > e1) e1 = d1;
                        if (d2 > e2) e2 = d2;
                        const S sc = fabs(Xf[(size_t)i]);
                        if (sc > scale) scale = sc;
                        if (fref) {
                            const S dr = fabs(Xc[(size_t)i] - Xr[(size_t)i]);
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
            double o1 = as_d(e1), o2 = as_d(e2);
            if (status == ORDER_ST_DIVERGED) {
                p = o1 = o2 = std::numeric_limits<double>::quiet_NaN();
            } else {
                // Полка округления — тот же порог, что в ядре: он растёт как
                // sqrt(числа шагов), потому что шум вычитания копится
                // случайным блужданием, а не держится в пределах пары ulp.
                const S nsteps = S(4.0 * (double)N);
                const S tiny = floorEps * (scale > S(0) ? scale : S(1)) * sqrt(nsteps);
                if (e2 <= tiny || e1 <= tiny) status = ORDER_ST_FLOOR;
                else if (e2 >= e1)            status = ORDER_ST_NOCONTRACT;
                p = (e2 > S(0) && e1 > S(0))
                        ? std::log2(as_d(e1 / e2))
                        : std::numeric_limits<double>::quiet_NaN();
            }

            res.p[gcell]      = p;
            res.e1[gcell]     = o1;
            res.e2[gcell]     = o2;
            res.e_ref[gcell]  = fref ? ((status == ORDER_ST_DIVERGED)
                                            ? std::numeric_limits<double>::quiet_NaN()
                                            : as_d(eRef))
                                     : std::numeric_limits<double>::quiet_NaN();
            res.h_eff[gcell]  = as_d(hEff);
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

static OrderResult run_order_cpu(const OrderRequest& req) {
    switch (req.cpu_prec) {
        case kOrderPrecQD: return run_order_cpu_t<ucuda::qd>(req);
        case kOrderPrecDD: return run_order_cpu_t<ucuda::dd>(req);
        default:           return run_order_cpu_t<numb>(req);
    }
}

template <class S>
static PerfResult run_performance_cpu_t(const PerfRequest& req) {
    using Tr = CpuScalarTraits<S>;
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

    const OrderResult ores = run_order_cpu_t<S>(oreq);
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
    if (!compile_cpu_step<S>(req.krs_body, n, amountOfValues, "KRS", step, err)) return fail(err);
    const typename Tr::Fn fstep = Tr::fn(step);

    std::vector<S> a((size_t)amountOfValues);
    std::vector<S> X((size_t)n);
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

        for (int k = 0; k < amountOfValues; ++k) a[(size_t)k] = S(req.values[(size_t)k]);
        if (req.axis.kind == OrderAxisKind::Value) {
            const int vi = req.axis.index;
            if (vi >= 0 && vi < amountOfValues) a[(size_t)vi] = S(res.axis_vals[(size_t)i]);
        }

        const long long N = order_steps_for_cpu(h_node, req.t_max, req.snap_steps);
        res.n_steps[(size_t)i] = N;

        auto one_run = [&]() {
            for (int k = 0; k < n; ++k) X[(size_t)k] = S(req.initial_conditions[(size_t)k]);
            const S hh = S(h_node);
            for (long long k = 0; k < N; ++k) Tr::call(fstep, X.data(), a.data(), hh);
            double acc = 0.0;
            for (int k = 0; k < n; ++k) acc += as_d(X[(size_t)k]);
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

static PerfResult run_performance_cpu(const PerfRequest& req) {
    switch (req.cpu_prec) {
        case kOrderPrecQD: return run_performance_cpu_t<ucuda::qd>(req);
        case kOrderPrecDD: return run_performance_cpu_t<ucuda::dd>(req);
        default:           return run_performance_cpu_t<numb>(req);
    }
}

// ---------------------------------------------------------------------------
// CPU-ветка областей устойчивости.
//
// Спецификацией, как и у Order, служит ядро: stabilityRegionKernel в
// kernels/order.template.cu. Формулы построения матрицы, порядок прогона
// базисных векторов и коды статусов повторены оттуда один в один.
//
// Расширенная точность здесь нужна не ради полки округления, а ради ГРАНИЦЫ:
// у схемы высокого порядка rho подходит к единице настолько полого, что в
// double ширина переходной полосы сравнима с шумом, и линия rho = 1 на карте
// начинает рваться.
template <class S>
static StabilityResult run_stability_cpu_t(const StabilityRequest& req) {
    using Tr = CpuScalarTraits<S>;
    using std::sqrt; using std::fabs;
    using std::isnan; using std::isinf;

    StabilityResult res;
    const std::string bad = stability_validate(req);
    if (!bad.empty()) { res.error = bad; return res; }
    stability_fill_axes(req, res);

    const int amountOfValues = (int)req.values.size();
    KrsCpuStep step;
    std::string err;
    if (!compile_cpu_step<S>(req.krs_body, req.amountOfX, amountOfValues, "step", step, err)) {
        res.error = err;
        return res;
    }
    typename Tr::Fn fn = Tr::fn(step);
    if (fn == nullptr) { res.error = "CPU step: the compiled body is unavailable"; return res; }

    std::vector<S> a0((size_t)amountOfValues);
    for (int i = 0; i < amountOfValues; ++i) a0[(size_t)i] = S(req.values[(size_t)i]);

    const S k  = S(req.k);
    const S r  = S(req.r);
    const S h  = S(req.h);
    const S kp1 = k + S(1);
    const S km1 = k - S(1);

    const size_t total_cells = (size_t)res.n_pts_x * (size_t)res.n_pts_y;
    res.rho.assign(total_cells, std::numeric_limits<double>::quiet_NaN());
    res.err.assign(total_cells, std::numeric_limits<double>::quiet_NaN());
    res.status.assign(total_cells, STAB_ST_BAD);

    std::vector<S> a((size_t)amountOfValues);
    std::vector<S> X((size_t)req.amountOfX);

    for (int iy = 0; iy < res.n_pts_y; ++iy) {
        for (int ix = 0; ix < res.n_pts_x; ++ix) {
            const size_t cell = (size_t)iy * (size_t)res.n_pts_x + (size_t)ix;
            const S sig = S(res.sigma_vals[(size_t)ix]);
            const S om  = S(res.omega_vals[(size_t)iy]);

            const S rad = -(S(1) / r) * (sig * sig * km1 * km1 / (kp1 * kp1) + om * om);
            if (isnan(rad) || rad < S(0)) continue;   // статус уже BAD

            const S dEl = S(2) * sig / kp1;
            const S cEl = -sqrt(rad);
            const S bEl = r * cEl;
            const S aEl = k * dEl;

            a = a0;
            a[(size_t)req.idx_a] = aEl; a[(size_t)req.idx_b] = bEl;
            a[(size_t)req.idx_c] = cEl; a[(size_t)req.idx_d] = dEl;

            S R[4];
            for (int col = 0; col < 2; ++col) {
                for (int i = 0; i < req.amountOfX; ++i) X[(size_t)i] = S(0);
                X[(size_t)col] = S(1);
                Tr::call(fn, X.data(), a.data(), h);
                R[col]     = X[0];
                R[2 + col] = X[1];
            }

            const S tr   = R[0] + R[3];
            const S det  = R[0] * R[3] - R[1] * R[2];
            const S disc = tr * tr * S(0.25) - det;

            S rho;
            if (disc >= S(0)) {
                const S sq = sqrt(disc);
                const S l1 = fabs(tr * S(0.5) + sq);
                const S l2 = fabs(tr * S(0.5) - sq);
                rho = (l1 > l2) ? l1 : l2;
            } else {
                rho = sqrt(det);
            }

            if (isnan(rho) || isinf(rho)) continue;   // статус уже BAD
            res.rho[cell]    = as_d(rho);
            res.status[cell] = STAB_ST_OK;
            // Ошибка шага — в double при любой точности (см. stability_step_error).
            res.err[cell] = stability_step_error(res.sigma_vals[(size_t)ix], res.omega_vals[(size_t)iy],
                                                 req.k, req.r, req.h,
                                                 as_d(R[0]), as_d(R[1]), as_d(R[2]), as_d(R[3]));
        }

        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true;
            return res;
        }
        if (req.progress)
            req.progress->store((float)((double)(iy + 1) / (double)res.n_pts_y),
                                std::memory_order_relaxed);
    }

    stability_summarize(res);
    res.ok = true;
    return res;
}

static StabilityResult run_stability_cpu(const StabilityRequest& req) {
    switch (req.cpu_prec) {
        case kOrderPrecQD: return run_stability_cpu_t<ucuda::qd>(req);
        case kOrderPrecDD: return run_stability_cpu_t<ucuda::dd>(req);
        default:           return run_stability_cpu_t<numb>(req);
    }
}

bool OrderAnalysisSession::run_async(ParametricEngine& engine, int config_idx) {
    if (in_flight) return false;
    if (config_idx < 0 || config_idx >= (int)configs.size()) return false;

    OrderConfig& c = configs[(size_t)config_idx];
    c.last_error.clear();
    last_run_label.clear();
    cancel_token   = std::make_shared<std::atomic<bool>>(false);
    progress_token = std::make_shared<std::atomic<float>>(0.0f);

    if (c.calc_kind == kOrderCalcStab) {
        StabilityRequest sreq = build_stability_request(*this, c);
        if (sreq.krs_body.empty()) {
            c.last_error = "krs_code is empty (no valid system or scheme)";
            cancel_token.reset();
            progress_token.reset();
            return false;
        }
        sreq.cancel   = cancel_token;
        sreq.progress = progress_token;

        in_flight = true;
        running_kind = kOrderCalcStab;
        running_config_index = config_idx;
        compute_start_time = std::chrono::steady_clock::now();
        const bool on_gpu = c.use_gpu;
        stab_future = std::async(std::launch::async, [&engine, on_gpu, sreq = std::move(sreq)]() {
            return on_gpu ? engine.run_stability(sreq) : run_stability_cpu(sreq);
        });
        return true;
    }

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
        running_kind = kOrderCalcPerf;
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
    running_kind = kOrderCalcOrder;
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
    // Три расчёта вкладки возвращают три разных типа, поэтому и future три;
    // забирается то, что отвечает running_kind.
    const int kind = running_kind;
    const bool valid = (kind == kOrderCalcStab) ? stab_future.valid()
                     : (kind == kOrderCalcPerf) ? perf_future.valid()
                                                : run_future.valid();
    if (!valid) {
        in_flight = false;
        running_kind = kOrderCalcOrder;
        running_config_index = -1;
        cancel_token.reset(); progress_token.reset();
        return false;
    }
    const std::future_status st =
        (kind == kOrderCalcStab) ? stab_future.wait_for(std::chrono::seconds(0))
      : (kind == kOrderCalcPerf) ? perf_future.wait_for(std::chrono::seconds(0))
                                 : run_future.wait_for(std::chrono::seconds(0));
    if (st != std::future_status::ready) return false;

    const int  idx = running_config_index;
    std::string label = (idx >= 0 && idx < (int)configs.size() && !configs[(size_t)idx].label.empty())
                            ? configs[(size_t)idx].label : std::string("order");
    const bool slot_ok = (idx >= 0 && idx < (int)configs.size());
    bool cancelled = false;
    if (kind == kOrderCalcStab) {
        StabilityResult r = stab_future.get();
        cancelled = r.cancelled;
        if (!cancelled && slot_ok)
            apply_stability_result(configs[(size_t)idx], std::move(r));
    } else if (kind == kOrderCalcPerf) {
        PerfResult r = perf_future.get();
        cancelled = r.cancelled;
        if (!cancelled && slot_ok)
            apply_perf_result(configs[(size_t)idx], std::move(r));
    } else {
        OrderResult r = run_future.get();
        cancelled = r.cancelled;
        if (!cancelled && slot_ok)
            apply_order_result(configs[(size_t)idx], std::move(r));
    }

    last_run_completed_at = std::chrono::steady_clock::now();
    last_run_seconds = std::chrono::duration<double>(last_run_completed_at - compute_start_time).count();
    last_run_label = label;
    last_run_succeeded = !cancelled;
    log_run_completed(last_run_label.c_str(), last_run_succeeded, last_run_seconds);

    in_flight = false;
    running_kind = kOrderCalcOrder;
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
