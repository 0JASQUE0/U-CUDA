#include "session_internal.h"

namespace ucuda::analysis_detail {


LLE1DRequest build_lle1d_request(const LLEAnalysisSession& s,
                                        const LLECurveConfig& c) {
    LLE1DRequest req;
    req.continuation         = c.continuation;
    req.continuation_reverse = c.continuation_reverse;
    req.use_cpu              = !c.use_gpu;
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
    req.sweep_over_h   = c.sweep_over_h;
    req.log_scale      = c.log_scale;
    req.var_sweep_index = (c.var_sweep_index >= 0 && c.var_sweep_index < req.amountOfX)
                          ? c.var_sweep_index : 0;

    req.param_lo       = parse_d(c.param_lo_text, 0.0);
    req.param_hi       = parse_d(c.param_hi_text, 1.0);
    if (req.param_hi < req.param_lo) std::swap(req.param_lo, req.param_hi);
    req.n_pts          = parse_i(c.n_pts_text, 500);
    req.h              = parse_d(c.h_text, 0.01);
    req.t_max          = parse_d(c.t_max_text, 100.0);
    req.transient_time = parse_d(c.transient_text, 100.0);
    req.max_value      = parse_d(c.max_value_text, 1.0e6);
    req.NT             = parse_d(c.nt_text,  1.0);
    req.eps            = parse_d(c.eps_text, 1.0e-4);
    req.vector_transient = parse_d(c.vtr_text, 0.0);
    req.csv_output_path = c.csv_save_enabled ? c.csv_output_path : std::string{};
    fill_adaptive_request(s.custom_schemes, s.sys, c.scheme, c.adaptive, req.amountOfX,
                          req.transient_time + req.t_max, c.ad_axis, 0, req.adaptive);
    return req;
}


void apply_lle1d_result(LLECurveConfig& c, LLE1DResult&& r) {
    c.result = std::move(r);
    c.last_run_ok = c.result.ok;
    if (!c.result.ok) c.last_error = c.result.error;
    c.data_generation++;
    c.fit_request = true;
}


// Снапшот для 2D-режима. Те же общие поля, что у 1D, + поля второй оси и
// mixed_mode. Несимметричный режим (одна param, другая IC без mixed_mode)
// engine отвергает в run_lle_2d — здесь только пробрасываем поля.
LLE2DRequest build_lle2d_request(const LLEAnalysisSession& s,
                                        const LLECurveConfig& c) {
    LLE2DRequest req;
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

    req.sweep_over_var     = c.sweep_over_var;
    req.sweep_over_var_2   = c.sweep_over_var_2;
    req.sweep_over_h       = c.sweep_over_h;
    req.sweep_over_h_2     = c.sweep_over_h_2;
    req.log_scale          = c.log_scale;
    req.log_scale_2        = c.log_scale_2;
    req.param_index        = sweep_param_slot(c.param_index, nparams);
    req.param_index_2      = sweep_param_slot(c.param_index_2, nparams);
    req.var_sweep_index    = (c.var_sweep_index >= 0 && c.var_sweep_index < req.amountOfX)
                             ? c.var_sweep_index : 0;
    req.var_sweep_index_2  = (c.var_sweep_index_2 >= 0 && c.var_sweep_index_2 < req.amountOfX)
                             ? c.var_sweep_index_2 : 0;

    req.param_lo           = parse_d(c.param_lo_text,   0.0);
    req.param_hi           = parse_d(c.param_hi_text,   1.0);
    req.param_lo_2         = parse_d(c.param_lo_2_text, 0.0);
    req.param_hi_2         = parse_d(c.param_hi_2_text, 1.0);
    if (req.param_hi   < req.param_lo)   std::swap(req.param_lo,   req.param_hi);
    if (req.param_hi_2 < req.param_lo_2) std::swap(req.param_lo_2, req.param_hi_2);
    req.n_pts              = parse_i(c.n_pts_text, 200);
    req.h                  = parse_d(c.h_text, 0.01);
    req.t_max              = parse_d(c.t_max_text, 100.0);
    req.transient_time     = parse_d(c.transient_text, 100.0);
    req.max_value          = parse_d(c.max_value_text, 1.0e6);
    req.NT                 = parse_d(c.nt_text,  1.0);
    req.eps                = parse_d(c.eps_text, 1.0e-4);
    req.vector_transient   = parse_d(c.vtr_text, 0.0);
    req.csv_output_path    = c.csv_save_enabled ? c.csv_output_path : std::string{};
    fill_adaptive_request(s.custom_schemes, s.sys, c.scheme, c.adaptive, req.amountOfX,
                          req.transient_time + req.t_max, c.ad_axis, c.ad_axis_2, req.adaptive);
    return req;
}


void apply_lle2d_result(LLECurveConfig& c, LLE2DResult&& r) {
    c.result_2d = std::move(r);
    c.last_run_2d_ok = c.result_2d.ok;
    if (!c.result_2d.ok) c.last_error = c.result_2d.error;
    c.data_generation_2d++;
    c.fit_request_2d = true;
}
}
using namespace ucuda::analysis_detail;


// LLEAnalysisSession — структурно зеркалит BifurcationAnalysisSession.
// Та же async-машинерия, та же per-curve конфигурация, тот же compute_krs_for_scheme.

void LLEAnalysisSession::load_from_record(const SystemRecord& r,
    const std::vector<std::string>& vars_,
    const std::vector<std::string>& params_) {
    vars = vars_;
    params = params_;
    custom_schemes = r.custom_schemes;
    enabled_builtin_schemes = enabled_builtins_from_record(r);

    curves.clear();
    LLECurveConfig c;
    c.label = "LLE 1";
    c.label_is_manual = false;   // fresh curve → auto-label
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
    if (!params.empty()) {
        c.param_index = 0;
    }
    curves.push_back(std::move(c));

    active_curve_index = 0;
    running_curve_index = -1;
}




void LLEAnalysisSession::remove_curve(int i) {
    if (i < 0 || i >= (int)curves.size()) return;
    if (in_flight && running_curve_index == i) return;
    curves.erase(curves.begin() + i);
    if (in_flight && running_curve_index > i) running_curve_index--;
    if (active_curve_index >= (int)curves.size())
        active_curve_index = (int)curves.size() - 1;
    if (active_curve_index < 0) active_curve_index = 0;
}


bool LLEAnalysisSession::run(ParametricEngine& engine, int curve_idx) {
    if (curve_idx < 0 || curve_idx >= (int)curves.size()) return false;
    LLECurveConfig& c = curves[curve_idx];
    // Keep previous last_run_ok flags so the old plot stays visible until
    // apply_*_result swaps in fresh data.
    c.last_error.clear();

    if (c.mode_2d) {
        LLE2DRequest req = build_lle2d_request(*this, c);
        if (req.krs_body.empty()) {
            c.last_error = krs_empty_error(custom_schemes, sys, c.scheme);
            return false;
        }
        LLE2DResult r = engine.run_lle_2d(req);
        bool ok = r.ok;
        apply_lle2d_result(c, std::move(r));
        return ok;
    } else {
        LLE1DRequest req = build_lle1d_request(*this, c);
        if (req.krs_body.empty()) {
            c.last_error = krs_empty_error(custom_schemes, sys, c.scheme);
            return false;
        }
        LLE1DResult r = engine.run_lle_1d(req);
        bool ok = r.ok;
        apply_lle1d_result(c, std::move(r));
        return ok;
    }
}


bool LLEAnalysisSession::run_async(ParametricEngine& engine, int curve_idx) {
    if (in_flight) return false;
    if (curve_idx < 0 || curve_idx >= (int)curves.size()) return false;

    LLECurveConfig& c = curves[curve_idx];
    // Keep previous last_run_ok flags so the old plot stays visible until
    // apply_*_result swaps in fresh data.
    c.last_error.clear();
    last_run_label.clear();
    cancel_token   = std::make_shared<std::atomic<bool>>(false);
    progress_token = std::make_shared<std::atomic<float>>(0.0f);

    if (c.mode_2d) {
        LLE2DRequest req = build_lle2d_request(*this, c);
        if (req.krs_body.empty()) {
            c.last_error = krs_empty_error(custom_schemes, sys, c.scheme);
            cancel_token.reset();
            progress_token.reset();
            return false;
        }
        req.cancel = cancel_token;
        req.progress = progress_token;
        in_flight = true;
        is_2d_run = true;
        running_curve_index = curve_idx;
        compute_start_time = std::chrono::steady_clock::now();
        run_future_2d = std::async(std::launch::async, [&engine, req = std::move(req)]() {
            return engine.run_lle_2d(req);
        });
    } else {
        LLE1DRequest req = build_lle1d_request(*this, c);
        if (req.krs_body.empty()) {
            c.last_error = krs_empty_error(custom_schemes, sys, c.scheme);
            cancel_token.reset();
            progress_token.reset();
            return false;
        }
        req.cancel = cancel_token;
        req.progress = progress_token;
        in_flight = true;
        is_2d_run = false;
        running_curve_index = curve_idx;
        compute_start_time = std::chrono::steady_clock::now();
        run_future = std::async(std::launch::async, [&engine, req = std::move(req)]() {
            return engine.run_lle_1d(req);
        });
    }
    return true;
}


void LLEAnalysisSession::request_cancel() {
    if (cancel_token) cancel_token->store(true, std::memory_order_relaxed);
}


bool LLEAnalysisSession::poll() {
    if (!in_flight) return false;
    bool cancelled = false;
    std::string label;
    int idx = running_curve_index;
    if (idx >= 0 && idx < (int)curves.size()) label = curves[idx].label;
    if (is_2d_run) {
        if (!run_future_2d.valid()) { in_flight = false; running_curve_index = -1; cancel_token.reset(); progress_token.reset(); return false; }
        if (run_future_2d.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
        LLE2DResult r = run_future_2d.get();
        cancelled = r.cancelled;
        if (!cancelled && idx >= 0 && idx < (int)curves.size()) {
            apply_lle2d_result(curves[idx], std::move(r));
        }
    } else {
        if (!run_future.valid()) { in_flight = false; running_curve_index = -1; cancel_token.reset(); progress_token.reset(); return false; }
        if (run_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
        LLE1DResult r = run_future.get();
        cancelled = r.cancelled;
        if (!cancelled && idx >= 0 && idx < (int)curves.size()) {
            apply_lle1d_result(curves[idx], std::move(r));
        }
    }
    last_run_completed_at = std::chrono::steady_clock::now();
    last_run_seconds = std::chrono::duration<double>(last_run_completed_at - compute_start_time).count();
    last_run_label = label.empty() ? std::string("LLE") : label;
    last_run_succeeded = !cancelled;
    log_run_completed(last_run_label.c_str(), last_run_succeeded, last_run_seconds);
    in_flight = false;
    running_curve_index = -1;
    cancel_token.reset();
    progress_token.reset();
    return true;
}
