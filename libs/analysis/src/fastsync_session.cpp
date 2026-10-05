#include "session_internal.h"

namespace ucuda::analysis_detail {


FastSyncRequest build_fastsync_request(const FastSyncAnalysisSession& s,
                                              const FastSyncConfig& c) {
    FastSyncRequest req;
    req.krs_body  = compute_krs_for_scheme(s.custom_schemes, s.sys, c.scheme);
    req.amountOfX = (int)s.vars.size();

    // values[0] = a[0] (symmetry для CD), затем по параметрам.
    int nparams = (int)s.params.size();
    req.values.assign((size_t)nparams + 1, 0.0);
    req.values[0] = parse_d(c.symmetry_s, 0.5);
    for (int i = 0; i < nparams; ++i) {
        auto it = c.param_values.find(s.params[i]);
        req.values[i + 1] = (it != c.param_values.end()) ? parse_d(it->second, 0.0) : 0.0;
    }

    // Per-var массивы.
    req.ic_master.resize(req.amountOfX);
    req.ic_slave.resize(req.amountOfX);
    req.k_forward.resize(req.amountOfX);
    req.k_backward.resize(req.amountOfX);
    for (int i = 0; i < req.amountOfX; ++i) {
        auto im = c.ic_master.find(s.vars[i]);
        auto is = c.ic_slave.find(s.vars[i]);
        auto kf = c.k_forward.find(s.vars[i]);
        auto kb = c.k_backward.find(s.vars[i]);
        req.ic_master[i]  = (im != c.ic_master.end())  ? parse_d(im->second, 0.0) : 0.0;
        req.ic_slave[i]   = (is != c.ic_slave.end())   ? parse_d(is->second, 0.0) : 0.0;
        req.k_forward[i]  = (kf != c.k_forward.end())  ? parse_d(kf->second, 0.0) : 0.0;
        req.k_backward[i] = (kb != c.k_backward.end()) ? parse_d(kb->second, 0.0) : 0.0;
    }

    req.mode           = (c.mode == 1) ? 1 : 0;
    req.h              = parse_d(c.h_text, 0.01);
    req.iter_of_synchr = std::max(1, parse_i(c.iter_of_synchr_text, 100));
    // On Grid: decimator не применяется к сетке ошибок синхронизации —
    // всегда полная точность, чтобы _config.csv не расходился с реальным
    // расчётом.
    req.pre_scaller    = (req.mode == 1) ? 1 : std::max(1, parse_i(c.pre_scaller_text, 1));
    req.max_value      = parse_d(c.max_value_text, 1e6);

    req.t_max          = parse_d(c.t_max_text, 100.0);
    req.transient_time       = parse_d(c.transient_text, 0.0);
    req.transient_time_slave = parse_d(c.transient_slave_text, 0.0);
    req.window               = (numb)parse_d(c.window_text, 50.0);

    req.axis_x_var = (c.axis_x_var >= 0 && c.axis_x_var < req.amountOfX) ? c.axis_x_var : 0;
    req.axis_y_var = (c.axis_y_var >= 0 && c.axis_y_var < req.amountOfX) ? c.axis_y_var
                                                                         : (req.amountOfX > 1 ? 1 : 0);
    req.axis_x_lo  = parse_d(c.axis_x_lo_text, -10.0);
    req.axis_x_hi  = parse_d(c.axis_x_hi_text,  10.0);
    req.axis_y_lo  = parse_d(c.axis_y_lo_text, -10.0);
    req.axis_y_hi  = parse_d(c.axis_y_hi_text,  10.0);
    if (req.axis_x_hi < req.axis_x_lo) std::swap(req.axis_x_lo, req.axis_x_hi);
    if (req.axis_y_hi < req.axis_y_lo) std::swap(req.axis_y_lo, req.axis_y_hi);
    req.n_pts      = std::max(1, parse_i(c.n_pts_text, 200));
    req.grid_swap_master_slave = c.grid_swap_master_slave;

    req.ic_random_offset = c.ic_random_offset;
    // eps по модулю: отступ и так симметричный, отрицательный ввод — опечатка,
    // а не «зеркальный» режим.
    req.ic_eps           = std::fabs(parse_d(c.ic_eps_text, 1e-3));
    req.ic_seed          = (unsigned long long)std::max(0, parse_i(c.ic_seed_text, 12345));
    req.gs_warmup        = std::max(0, parse_i(c.gs_warmup_text, 0));

    req.type_of_synch = (c.type_of_synch == 1) ? 1 : 0;
    int ee = c.error_estim;
    if (ee < 0 || ee > 7) ee = 2;
    req.error_estim   = ee;
    req.fs_error_trs  = parse_d(c.fs_error_trs_text, 1e-12);

    req.csv_output_path = c.csv_save_enabled ? c.csv_output_path : std::string{};
    req.var_names       = s.vars;

    return req;
}


void apply_fastsync_result(FastSyncConfig& c, FastSyncResult&& r) {
    c.result = std::move(r);
    c.last_run_ok = c.result.ok;
    if (!c.result.ok) c.last_error = c.result.error;
    c.data_generation++;
    c.fit_request = true;
}
}
using namespace ucuda::analysis_detail;


// FastSyncAnalysisSession — зеркалит BasinsAnalysisSession (multi-config,
// async worker, futures + cancel/progress). build_request читает text-fields
// и собирает FastSyncRequest. По умолчанию k_forward/k_backward/ic_slave
// заполняются нулями — пользователь видит сразу все vars в UI.
void FastSyncAnalysisSession::load_from_record(const SystemRecord& r,
    const std::vector<std::string>& vars_,
    const std::vector<std::string>& params_) {
    vars = vars_;
    params = params_;
    custom_schemes = r.custom_schemes;
    enabled_builtin_schemes = enabled_builtins_from_record(r);

    FastSyncConfig c;
    c.label = "FastSync 1";
    c.h_text = default_h_from_record(r);
    c.scheme = default_scheme_from_record(r, c.scheme);
    c.symmetry_s = r.symmetry_s.empty() ? std::string("0.5") : r.symmetry_s;

    // Параметры — из record, остальные секции инициализируем нулями.
    for (const auto& p : params) {
        auto it = r.param_values.find(p);
        c.param_values[p] = (it != r.param_values.end()) ? it->second : std::string("0");
    }
    for (const auto& v : vars) {
        auto it = r.init_conditions.find(v);
        // ic_master из record (если есть), остальные три — нули.
        c.ic_master[v]  = (it != r.init_conditions.end()) ? it->second : std::string("0");
        c.ic_slave[v]   = "0";
        c.k_forward[v]  = "0";
        c.k_backward[v] = "0";
    }
    c.axis_x_var = 0;
    c.axis_y_var = (vars.size() > 1) ? 1 : 0;

    configs.clear();
    configs.push_back(std::move(c));
    active_config_index = 0;
    running_config_index = -1;
}


void FastSyncAnalysisSession::add_config() {
    FastSyncConfig c;
    if (!configs.empty()) {
        c = configs.back();
        c.result = FastSyncResult{};
        c.last_run_ok = false;
        c.last_error.clear();
        c.data_generation = 0;
        c.fit_request = false;
    } else {
        for (const auto& v : vars) {
            c.ic_master[v] = "0"; c.ic_slave[v] = "0";
            c.k_forward[v] = "0"; c.k_backward[v] = "0";
        }
        for (const auto& p : params) c.param_values[p] = "0";
    }
    c.label = "FastSync " + std::to_string(configs.size() + 1);
    configs.push_back(std::move(c));
    active_config_index = (int)configs.size() - 1;
}


void FastSyncAnalysisSession::remove_config(int i) {
    if (i < 0 || i >= (int)configs.size()) return;
    if (in_flight && running_config_index == i) return;
    configs.erase(configs.begin() + i);
    if (in_flight && running_config_index > i) running_config_index--;
    if (active_config_index >= (int)configs.size())
        active_config_index = (int)configs.size() - 1;
    if (active_config_index < 0) active_config_index = 0;
}


bool FastSyncAnalysisSession::run(ParametricEngine& engine, int config_idx) {
    if (config_idx < 0 || config_idx >= (int)configs.size()) return false;
    FastSyncConfig& c = configs[config_idx];
    c.last_error.clear();
    FastSyncRequest req = build_fastsync_request(*this, c);
    if (req.krs_body.empty()) { c.last_error = "krs_code is empty"; return false; }
    FastSyncResult r = engine.run_fastsync(req);
    bool ok = r.ok;
    apply_fastsync_result(c, std::move(r));
    return ok;
}


bool FastSyncAnalysisSession::run_async(ParametricEngine& engine, int config_idx) {
    if (in_flight) return false;
    if (config_idx < 0 || config_idx >= (int)configs.size()) return false;
    FastSyncConfig& c = configs[config_idx];
    c.last_error.clear();
    last_run_label.clear();
    cancel_token   = std::make_shared<std::atomic<bool>>(false);
    progress_token = std::make_shared<std::atomic<float>>(0.0f);

    FastSyncRequest req = build_fastsync_request(*this, c);
    if (req.krs_body.empty()) {
        c.last_error = "krs_code is empty";
        cancel_token.reset(); progress_token.reset();
        return false;
    }
    req.cancel   = cancel_token;
    req.progress = progress_token;

    in_flight = true;
    running_config_index = config_idx;
    compute_start_time = std::chrono::steady_clock::now();
    run_future = std::async(std::launch::async, [&engine, req = std::move(req)]() {
        return engine.run_fastsync(req);
    });
    return true;
}


void FastSyncAnalysisSession::request_cancel() {
    if (cancel_token) cancel_token->store(true, std::memory_order_relaxed);
}


bool FastSyncAnalysisSession::poll() {
    if (!in_flight) return false;
    if (!run_future.valid()) {
        in_flight = false; running_config_index = -1;
        cancel_token.reset(); progress_token.reset();
        return false;
    }
    if (run_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;

    FastSyncResult r = run_future.get();
    bool cancelled = r.cancelled;
    int idx = running_config_index;
    std::string label = (idx >= 0 && idx < (int)configs.size() && !configs[idx].label.empty())
                            ? configs[idx].label : std::string("fastsync");
    if (!cancelled && idx >= 0 && idx < (int)configs.size()) {
        apply_fastsync_result(configs[idx], std::move(r));
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
