#include "session_internal.h"
namespace ucuda::analysis_detail {
SignalMetricsRequest build_metrics_request(const SignalMetricsAnalysisSession& s,
                                                  const SignalMetricsConfig& c) {
    SignalMetricsRequest req;
    req.krs_body  = compute_krs_for_scheme(s.custom_schemes, s.sys, c.scheme);
    req.amountOfX = (int)s.vars.size();
    req.dimension = c.mode_2d ? 2 : 1;

    req.initial_conditions.resize(req.amountOfX);
    for (int i = 0; i < req.amountOfX; ++i) {
        auto it = c.initial_conditions.find(s.vars[i]);
        req.initial_conditions[i] = (it != c.initial_conditions.end()) ? parse_d(it->second, 0.0) : 0.0;
    }

    int nparams = (int)s.params.size();
    req.base_values.assign((size_t)nparams + 1, 0.0);
    req.base_values[0] = parse_d(c.symmetry_s, 0.5); // CD: коэф. симметрии (a[0])
    for (int i = 0; i < nparams; ++i) {
        auto it = c.param_values.find(s.params[i]);
        req.base_values[i + 1] = (it != c.param_values.end()) ? parse_d(it->second, 0.0) : 0.0;
    }

    req.param_index       = sweep_param_slot(c.param_index, nparams);
    req.sweep_over_var    = c.sweep_over_var;
    req.sweep_over_h      = c.sweep_over_h;
    req.log_scale         = c.log_scale;
    req.var_sweep_index   = (c.var_sweep_index >= 0 && c.var_sweep_index < req.amountOfX)
                            ? c.var_sweep_index : 0;
    req.param_lo          = parse_d(c.param_lo_text, 0.0);
    req.param_hi          = parse_d(c.param_hi_text, 1.0);
    if (req.param_hi < req.param_lo) std::swap(req.param_lo, req.param_hi);

    if (c.mode_2d) {
        req.param_index_2     = sweep_param_slot(c.param_index_2, nparams);
        req.sweep_over_var_2  = c.sweep_over_var_2;
        req.sweep_over_h_2    = c.sweep_over_h_2;
        req.log_scale_2       = c.log_scale_2;
        req.var_sweep_index_2 = (c.var_sweep_index_2 >= 0 && c.var_sweep_index_2 < req.amountOfX)
                                ? c.var_sweep_index_2 : 0;
        req.param_lo_2        = parse_d(c.param_lo_2_text, 0.0);
        req.param_hi_2        = parse_d(c.param_hi_2_text, 1.0);
        if (req.param_hi_2 < req.param_lo_2) std::swap(req.param_lo_2, req.param_hi_2);
    }

    // Continuation и CPU — только 1D (движок откажет иначе, а UI и так блокирует).
    req.continuation         = !c.mode_2d && c.continuation;
    req.continuation_reverse = req.continuation && c.continuation_reverse;
    req.use_cpu              = !c.mode_2d && !c.use_gpu;
    req.n_pts          = parse_i(c.n_pts_text, c.mode_2d ? 200 : 500);
    req.writable_var   = (c.writable_var >= -1 && c.writable_var < req.amountOfX) ? c.writable_var : 0;
    req.h              = parse_d(c.h_text, 0.01);
    req.t_max          = parse_d(c.t_max_text, 100.0);
    req.transient_time = parse_d(c.transient_text, 100.0);
    req.pre_scaller    = std::max(1, parse_i(c.pre_scaller_text, 1));
    req.max_value      = parse_d(c.max_value_text, 1.0e6);
    req.metric_mask    = c.metric_mask & kSignalMetricAllMask;
    req.csv_output_path = c.csv_save_enabled ? c.csv_output_path : std::string{};
    req.minmax_interp  = c.adaptive.minmax_interp_fixed;
    // 1D: вторая ось не участвует, и её код от режима 2D не должен уводить расчёт из модуля
    // узлов (ad_nodes_module смотрит обе оси).
    fill_adaptive_request(s.custom_schemes, s.sys, c.scheme, c.adaptive, req.amountOfX,
                          req.transient_time + req.t_max, c.ad_axis, c.mode_2d ? c.ad_axis_2 : 0, req.adaptive);
    return req;
}


void apply_metrics_result(SignalMetricsConfig& c, SignalMetricsResult&& r) {
    if (r.dimension == 2) {
        c.result_2d = std::move(r);
        c.last_run_2d_ok = c.result_2d.ok;
        if (!c.result_2d.ok) c.last_error = c.result_2d.error;
        c.data_generation_2d++;
        c.fit_request_2d = true;
    } else {
        c.result = std::move(r);
        c.last_run_ok = c.result.ok;
        if (!c.result.ok) c.last_error = c.result.error;
        c.data_generation++;
        c.fit_request = true;
    }
}



}
using namespace ucuda::analysis_detail;
void SignalMetricsAnalysisSession::load_from_record(const SystemRecord& r,
    const std::vector<std::string>& vars_,
    const std::vector<std::string>& params_) {
    vars = vars_;
    params = params_;
    custom_schemes = r.custom_schemes;
    enabled_builtin_schemes = enabled_builtins_from_record(r);

    configs.clear();
    SignalMetricsConfig c;
    c.label = "Metrics 1";
    c.label_is_manual = false;   // fresh config → auto-label
    c.h_text = default_h_from_record(r);
    c.scheme = default_scheme_from_record(r, c.scheme);
    c.symmetry_s = r.symmetry_s.empty() ? std::string("0.5") : r.symmetry_s;

    for (const auto& p : params) {
        auto it = r.param_values.find(p);
        c.param_values[p] = (it != r.param_values.end()) ? it->second : "";
    }
    for (const auto& v : vars) {
        auto it = r.init_conditions.find(v);
        c.initial_conditions[v] = (it != r.init_conditions.end()) ? it->second : "";
    }
    configs.push_back(std::move(c));

    active_config_index = 0;
    running_config_index = -1;
}


void SignalMetricsAnalysisSession::add_config() {
    SignalMetricsConfig c;
    if (!configs.empty()) {
        c = configs.back();
        c.result = SignalMetricsResult{};
        c.result_2d = SignalMetricsResult{};
        c.last_run_ok = false;
        c.last_run_2d_ok = false;
        c.last_error.clear();
        c.data_generation = 0;
        c.data_generation_2d = 0;
        c.fit_request = false;
        c.fit_request_2d = false;
    } else {
        for (const auto& v : vars) c.initial_conditions[v] = "";
        for (const auto& p : params) c.param_values[p] = "";
    }
    c.label = "Metrics " + std::to_string(configs.size() + 1);
    c.label_is_manual = false;
    configs.push_back(std::move(c));
    active_config_index = (int)configs.size() - 1;
}


void SignalMetricsAnalysisSession::remove_config(int i) {
    if (i < 0 || i >= (int)configs.size()) return;
    if (in_flight && running_config_index == i) return;
    configs.erase(configs.begin() + i);
    if (in_flight && running_config_index > i) running_config_index--;
    if (active_config_index >= (int)configs.size())
        active_config_index = (int)configs.size() - 1;
    if (active_config_index < 0) active_config_index = 0;
}


bool SignalMetricsAnalysisSession::run_async(ParametricEngine& engine, int config_idx) {
    if (in_flight) return false;
    if (config_idx < 0 || config_idx >= (int)configs.size()) return false;

    SignalMetricsConfig& c = configs[config_idx];
    c.last_error.clear();
    last_run_label.clear();

    SignalMetricsRequest req = build_metrics_request(*this, c);
    if (req.krs_body.empty()) {
        c.last_error = "krs_code is empty (no valid system or scheme)";
        return false;
    }
    cancel_token   = std::make_shared<std::atomic<bool>>(false);
    progress_token = std::make_shared<std::atomic<float>>(0.0f);
    req.cancel   = cancel_token;
    req.progress = progress_token;
    in_flight = true;
    is_2d_run = c.mode_2d;
    running_config_index = config_idx;
    compute_start_time = std::chrono::steady_clock::now();
    run_future = std::async(std::launch::async, [&engine, req = std::move(req)]() {
        return engine.run_signal_metrics(req);
    });
    return true;
}


void SignalMetricsAnalysisSession::request_cancel() {
    if (cancel_token) cancel_token->store(true, std::memory_order_relaxed);
}


bool SignalMetricsAnalysisSession::poll() {
    if (!in_flight) return false;
    const int idx = running_config_index;
    if (!run_future.valid()) {
        in_flight = false; running_config_index = -1;
        cancel_token.reset(); progress_token.reset();
        return false;
    }
    if (run_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;

    SignalMetricsResult r = run_future.get();
    const bool cancelled = r.cancelled;
    std::string label;
    if (idx >= 0 && idx < (int)configs.size()) {
        label = configs[idx].label;
        if (!cancelled) apply_metrics_result(configs[idx], std::move(r));
    }
    last_run_completed_at = std::chrono::steady_clock::now();
    last_run_seconds = std::chrono::duration<double>(last_run_completed_at - compute_start_time).count();
    last_run_label = label.empty() ? std::string("Metrics") : label;
    last_run_succeeded = !cancelled;
    log_run_completed(last_run_label.c_str(), last_run_succeeded, last_run_seconds);
    in_flight = false;
    running_config_index = -1;
    cancel_token.reset();
    progress_token.reset();
    return true;
}

std::function<void(ParametricEngine&)> SignalMetricsAnalysisSession::prewarm_task(int config_idx) const {
    if (config_idx < 0 || config_idx >= (int)configs.size()) return {};
    SignalMetricsRequest req = build_metrics_request(*this, configs[config_idx]);
    if (req.krs_body.empty()) return {};
    return [req](ParametricEngine& e) { e.prewarm(req); };
}
