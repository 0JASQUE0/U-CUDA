#include "order_session.h"
#include "num_parse.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

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
    req.axis_y = to_engine_axis(c.axis_y_target, c.axis_y_lo_text, c.axis_y_hi_text,
                                c.axis_y_log, c.axis_y_n_text, c.two_d, amountOfValues);

    req.h             = parse_d(c.h_text, 0.01);
    req.t_max         = parse_d(c.t_max_text, 10.0);
    req.max_value     = parse_d(c.max_value_text, 1.0e6);
    req.snap_steps    = c.snap_steps;
    req.endpoint_only = c.endpoint_only;
    return req;
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

bool OrderAnalysisSession::run_async(ParametricEngine& engine, int config_idx) {
    if (in_flight) return false;
    if (config_idx < 0 || config_idx >= (int)configs.size()) return false;

    OrderConfig& c = configs[(size_t)config_idx];
    c.last_error.clear();
    last_run_label.clear();
    cancel_token   = std::make_shared<std::atomic<bool>>(false);
    progress_token = std::make_shared<std::atomic<float>>(0.0f);

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
    running_config_index = config_idx;
    compute_start_time = std::chrono::steady_clock::now();
    run_future = std::async(std::launch::async, [&engine, req = std::move(req)]() {
        return engine.run_order(req);
    });
    return true;
}

void OrderAnalysisSession::request_cancel() {
    if (cancel_token) cancel_token->store(true, std::memory_order_relaxed);
}

bool OrderAnalysisSession::poll() {
    if (!in_flight) return false;
    if (!run_future.valid()) {
        in_flight = false;
        running_config_index = -1;
        cancel_token.reset(); progress_token.reset();
        return false;
    }
    if (run_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;

    OrderResult r = run_future.get();
    const bool cancelled = r.cancelled;
    const int  idx = running_config_index;
    std::string label = (idx >= 0 && idx < (int)configs.size() && !configs[(size_t)idx].label.empty())
                            ? configs[(size_t)idx].label : std::string("order");
    if (!cancelled && idx >= 0 && idx < (int)configs.size())
        apply_order_result(configs[(size_t)idx], std::move(r));

    last_run_completed_at = std::chrono::steady_clock::now();
    last_run_seconds = std::chrono::duration<double>(last_run_completed_at - compute_start_time).count();
    last_run_label = label;
    last_run_succeeded = !cancelled;
    log_run_completed(last_run_label.c_str(), last_run_succeeded, last_run_seconds);

    in_flight = false;
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
