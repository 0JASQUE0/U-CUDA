#include "session_internal.h"

namespace ucuda::analysis_detail {


Dft1DRequest build_dft1d_request(const Dft1DAnalysisSession& s,
                                        const Dft1DConfig& c) {
    Dft1DRequest req;
    req.sweep_over_h = c.sweep_over_h;
    req.log_scale    = c.log_scale;
    req.use_cpu      = !c.use_gpu;
    req.krs_body  = compute_krs_for_scheme(s.custom_schemes, s.sys, c.scheme);
    req.amountOfX = (int)s.vars.size();

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
    req.param_index = sweep_param_slot(c.param_index, nparams);
    req.sweep_over_var = c.sweep_over_var;
    req.var_sweep_index = (c.var_sweep_index >= 0 && c.var_sweep_index < req.amountOfX)
                          ? c.var_sweep_index : 0;
    req.continuation = c.continuation;
    req.continuation_reverse = c.continuation_reverse;

    req.param_lo = parse_d(c.param_lo_text, 0.0);
    req.param_hi = parse_d(c.param_hi_text, 1.0);
    if (req.param_hi < req.param_lo) std::swap(req.param_lo, req.param_hi);
    req.n_pts        = parse_i(c.n_pts_text, 500);
    // -1 — combination (см. build_bif1d_request / loopCalculateDiscreteModel_int).
    req.writable_var = (c.writable_var >= -1 && c.writable_var < req.amountOfX) ? c.writable_var : 0;

    req.n_freq  = parse_i(c.n_freq_text, 200);
    req.freq_lo = parse_d(c.freq_lo_text, 0.0);
    req.freq_hi = parse_d(c.freq_hi_text, 10.0);
    if (req.freq_hi < req.freq_lo) std::swap(req.freq_lo, req.freq_hi);
    req.freq_log_scale = c.freq_log_scale;
    req.window_type = c.window_type;

    req.h              = parse_d(c.h_text, 0.01);
    req.t_max          = parse_d(c.t_max_text, 100.0);
    req.transient_time = parse_d(c.transient_text, 100.0);
    req.pre_scaller    = std::max(1, parse_i(c.pre_scaller_text, 1));
    req.max_value      = parse_d(c.max_value_text, 1.0e6);
    req.csv_output_path = c.csv_save_enabled ? c.csv_output_path : std::string{};
    return req;
}


void apply_dft1d_result(Dft1DConfig& c, Dft1DResult&& r) {
    c.result = std::move(r);
    c.last_run_ok = c.result.ok;
    if (!c.result.ok) c.last_error = c.result.error;
    c.data_generation++;
    c.fit_request = true;
    // Инвалидируем display-кэш явно (staleness-check в draw_dft1d_plot и так
    // поймает смену data_generation, но display_built_from=-1 форсит rebuild
    // немедленно, а не на следующий кадр после сравнения полей).
    c.display_built_from = -1;
}
}
using namespace ucuda::analysis_detail;


// Dft1DAnalysisSession — N configs на сессию (по образцу BasinsAnalysisSession:
// один "kind", список конфигов, своя очередь). build_dft1d_request зеркалит
// build_bif1d_request (тот же sweep/integration/IC/param маппинг) плюс
// n_freq/freq_lo/freq_hi.

void Dft1DAnalysisSession::load_from_record(const SystemRecord& r,
    const std::vector<std::string>& vars_,
    const std::vector<std::string>& params_) {
    vars = vars_;
    params = params_;
    custom_schemes = r.custom_schemes;
    enabled_builtin_schemes = enabled_builtins_from_record(r);

    Dft1DConfig c;
    c.label = "DFT 1";
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
    c.writable_var = 0;

    configs.clear();
    configs.push_back(std::move(c));
    active_config_index = 0;
    running_config_index = -1;
}


void Dft1DAnalysisSession::add_config() {
    Dft1DConfig c;
    if (!configs.empty()) {
        // Глубокая копия последнего config — пользователь обычно правит 1-2 поля.
        c = configs.back();
        c.result = Dft1DResult{};
        c.last_run_ok = false;
        c.last_error.clear();
        c.data_generation = 0;
        c.fit_request = false;
        c.display_cache.clear();
        c.display_built_from = -1;
    } else {
        for (const auto& v : vars) c.initial_conditions[v] = "";
        for (const auto& p : params) c.param_values[p] = "";
    }
    c.label = "DFT " + std::to_string(configs.size() + 1);
    configs.push_back(std::move(c));
    active_config_index = (int)configs.size() - 1;
}


void Dft1DAnalysisSession::remove_config(int i) {
    if (i < 0 || i >= (int)configs.size()) return;
    // Запрещаем удалять config, чей расчёт сейчас идёт — иначе worker положит
    // результат в несуществующий слот при poll().
    if (in_flight && running_config_index == i) return;
    configs.erase(configs.begin() + i);
    if (in_flight && running_config_index > i) running_config_index--;
    if (active_config_index >= (int)configs.size())
        active_config_index = (int)configs.size() - 1;
    if (active_config_index < 0) active_config_index = 0;
}


bool Dft1DAnalysisSession::run(ParametricEngine& engine, int config_idx) {
    if (config_idx < 0 || config_idx >= (int)configs.size()) return false;
    Dft1DConfig& c = configs[config_idx];
    c.last_error.clear();

    Dft1DRequest req = build_dft1d_request(*this, c);
    if (req.krs_body.empty()) {
        c.last_error = krs_empty_error(custom_schemes, sys, c.scheme);
        return false;
    }
    Dft1DResult r = engine.run_dft_1d(req);
    bool ok = r.ok;
    apply_dft1d_result(c, std::move(r));
    return ok;
}


bool Dft1DAnalysisSession::run_async(ParametricEngine& engine, int config_idx) {
    if (in_flight) return false;
    if (config_idx < 0 || config_idx >= (int)configs.size()) return false;

    Dft1DConfig& c = configs[config_idx];
    c.last_error.clear();
    last_run_label.clear();
    cancel_token   = std::make_shared<std::atomic<bool>>(false);
    progress_token = std::make_shared<std::atomic<float>>(0.0f);

    Dft1DRequest req = build_dft1d_request(*this, c);
    if (req.krs_body.empty()) {
        c.last_error = krs_empty_error(custom_schemes, sys, c.scheme);
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
        return engine.run_dft_1d(req);
    });
    return true;
}


void Dft1DAnalysisSession::request_cancel() {
    if (cancel_token) cancel_token->store(true, std::memory_order_relaxed);
}


bool Dft1DAnalysisSession::poll() {
    if (!in_flight) return false;
    if (!run_future.valid()) {
        in_flight = false;
        running_config_index = -1;
        cancel_token.reset(); progress_token.reset();
        return false;
    }
    if (run_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;

    Dft1DResult r = run_future.get();
    bool cancelled = r.cancelled;
    int idx = running_config_index;
    std::string label = (idx >= 0 && idx < (int)configs.size() && !configs[idx].label.empty())
                            ? configs[idx].label : std::string("dft1d");
    if (!cancelled && idx >= 0 && idx < (int)configs.size()) {
        apply_dft1d_result(configs[idx], std::move(r));
    }
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
