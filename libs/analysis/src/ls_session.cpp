#include "session_internal.h"

using namespace ucuda::analysis_detail;


// LyapunovSpectrumAnalysisSession — копия LLE-паттерна для LS.

void LyapunovSpectrumAnalysisSession::load_from_record(const SystemRecord& r,
    const std::vector<std::string>& vars_,
    const std::vector<std::string>& params_) {
    vars = vars_;
    params = params_;
    custom_schemes = r.custom_schemes;
    enabled_builtin_schemes = enabled_builtins_from_record(r);

    curves.clear();
    LSCurveConfig c;
    c.label = "LS 1";
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




void LyapunovSpectrumAnalysisSession::remove_curve(int i) {
    if (i < 0 || i >= (int)curves.size()) return;
    if (in_flight && running_curve_index == i) return;
    curves.erase(curves.begin() + i);
    if (in_flight && running_curve_index > i) running_curve_index--;
    if (active_curve_index >= (int)curves.size())
        active_curve_index = (int)curves.size() - 1;
    if (active_curve_index < 0) active_curve_index = 0;
}


bool LyapunovSpectrumAnalysisSession::run(ParametricEngine& engine, int curve_idx) {
    if (curve_idx < 0 || curve_idx >= (int)curves.size()) return false;
    LSCurveConfig& c = curves[curve_idx];
    // Keep previous last_run_ok flags so the old plot stays visible until
    // apply_*_result swaps in fresh data.
    c.last_error.clear();

    if (c.mode_2d) {
        LS2DRequest req = build_ls2d_request(*this, c);
        if (req.krs_body.empty()) {
            c.last_error = krs_empty_error(custom_schemes, sys, c.scheme);
            return false;
        }
        LS2DResult r = engine.run_ls_2d(req);
        bool ok = r.ok;
        apply_ls2d_result(c, std::move(r));
        return ok;
    } else {
        LS1DRequest req = build_ls1d_request(*this, c);
        if (req.krs_body.empty()) {
            c.last_error = krs_empty_error(custom_schemes, sys, c.scheme);
            return false;
        }
        LS1DResult r = engine.run_ls_1d(req);
        bool ok = r.ok;
        apply_ls1d_result(c, std::move(r));
        return ok;
    }
}


bool LyapunovSpectrumAnalysisSession::run_async(ParametricEngine& engine, int curve_idx) {
    if (in_flight) return false;
    if (curve_idx < 0 || curve_idx >= (int)curves.size()) return false;

    LSCurveConfig& c = curves[curve_idx];
    // Keep previous last_run_ok flags so the old plot stays visible until
    // apply_*_result swaps in fresh data.
    c.last_error.clear();
    last_run_label.clear();
    cancel_token   = std::make_shared<std::atomic<bool>>(false);
    progress_token = std::make_shared<std::atomic<float>>(0.0f);

    if (c.mode_2d) {
        LS2DRequest req = build_ls2d_request(*this, c);
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
            return engine.run_ls_2d(req);
        });
    } else {
        LS1DRequest req = build_ls1d_request(*this, c);
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
            return engine.run_ls_1d(req);
        });
    }
    return true;
}


void LyapunovSpectrumAnalysisSession::request_cancel() {
    if (cancel_token) cancel_token->store(true, std::memory_order_relaxed);
}


bool LyapunovSpectrumAnalysisSession::poll() {
    if (!in_flight) return false;
    bool cancelled = false;
    std::string label;
    int idx = running_curve_index;
    if (idx >= 0 && idx < (int)curves.size()) label = curves[idx].label;
    if (is_2d_run) {
        if (!run_future_2d.valid()) { in_flight = false; running_curve_index = -1; cancel_token.reset(); progress_token.reset(); return false; }
        if (run_future_2d.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
        LS2DResult r = run_future_2d.get();
        cancelled = r.cancelled;
        if (!cancelled && idx >= 0 && idx < (int)curves.size()) {
            apply_ls2d_result(curves[idx], std::move(r));
        }
    } else {
        if (!run_future.valid()) { in_flight = false; running_curve_index = -1; cancel_token.reset(); progress_token.reset(); return false; }
        if (run_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
        LS1DResult r = run_future.get();
        cancelled = r.cancelled;
        if (!cancelled && idx >= 0 && idx < (int)curves.size()) {
            apply_ls1d_result(curves[idx], std::move(r));
        }
    }
    last_run_completed_at = std::chrono::steady_clock::now();
    last_run_seconds = std::chrono::duration<double>(last_run_completed_at - compute_start_time).count();
    last_run_label = label.empty() ? std::string("LS") : label;
    last_run_succeeded = !cancelled;
    log_run_completed(last_run_label.c_str(), last_run_succeeded, last_run_seconds);
    in_flight = false;
    running_curve_index = -1;
    cancel_token.reset();
    progress_token.reset();
    return true;
}
