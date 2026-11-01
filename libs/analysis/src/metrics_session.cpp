#include "session_internal.h"
using namespace ucuda::analysis_detail;


std::function<void(ParametricEngine&)> BifurcationAnalysisSession::prewarm_task(int diagram_idx) const {
    if (diagram_idx < 0 || diagram_idx >= (int)diagrams.size()) return {};
    const BifurcationDiagramConfig& bd = diagrams[diagram_idx];
    if (bd.mode_2d) {
        Bifurcation2DRequest req = build_bif2d_request(*this, bd);
        if (req.krs_body.empty()) return {};
        return [req](ParametricEngine& e) { e.prewarm(req); };
    }
    Bifurcation1DRequest req = build_bif1d_request(*this, bd);
    if (req.krs_body.empty()) return {};
    return [req](ParametricEngine& e) { e.prewarm(req); };
}


std::function<void(ParametricEngine&)> LLEAnalysisSession::prewarm_task(int curve_idx) const {
    if (curve_idx < 0 || curve_idx >= (int)curves.size()) return {};
    const LLECurveConfig& c = curves[curve_idx];
    if (c.mode_2d) {
        LLE2DRequest req = build_lle2d_request(*this, c);
        if (req.krs_body.empty()) return {};
        return [req](ParametricEngine& e) { e.prewarm(req); };
    }
    LLE1DRequest req = build_lle1d_request(*this, c);
    if (req.krs_body.empty()) return {};
    return [req](ParametricEngine& e) { e.prewarm(req); };
}


std::function<void(ParametricEngine&)> LyapunovSpectrumAnalysisSession::prewarm_task(int curve_idx) const {
    if (curve_idx < 0 || curve_idx >= (int)curves.size()) return {};
    const LSCurveConfig& c = curves[curve_idx];
    if (c.mode_2d) {
        LS2DRequest req = build_ls2d_request(*this, c);
        if (req.krs_body.empty()) return {};
        return [req](ParametricEngine& e) { e.prewarm(req); };
    }
    LS1DRequest req = build_ls1d_request(*this, c);
    if (req.krs_body.empty()) return {};
    return [req](ParametricEngine& e) { e.prewarm(req); };
}


// SignalMetricsAnalysisSession — устройство LLEAnalysisSession, один future на
// оба режима (run_signal_metrics различает их по request.dimension).

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
    c.cd_fp_iters = r.cd_fp_iters.empty() ? std::string("4") : r.cd_fp_iters;
    c.newton = newton_from_record(r);

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

namespace ucuda::analysis_detail {


SignalMetricsRequest build_metrics_request(const SignalMetricsAnalysisSession& s,
                                                  const SignalMetricsConfig& c) {
    SignalMetricsRequest req;
    req.krs_body  = sweep_krs(s.custom_schemes, s.sys, c.scheme, c.adaptive);
    req.amountOfX = (int)s.vars.size();
    req.dimension = c.mode_2d ? 2 : 1;

    req.initial_conditions.resize(req.amountOfX);
    for (int i = 0; i < req.amountOfX; ++i) {
        auto it = c.initial_conditions.find(s.vars[i]);
        req.initial_conditions[i] = (it != c.initial_conditions.end()) ? parse_d(it->second, 0.0) : 0.0;
    }

    int nparams = (int)s.params.size();
    req.base_values.assign((size_t)nparams + 1 + kSchemeTailSlots, 0.0);   // [s, p_1..p_M, n, Newton]
    req.base_values[0] = parse_d(c.symmetry_s, 0.5); // CD: коэф. симметрии (a[0])
    scheme_fill_tail(req.base_values.data() + nparams + 1, c.cd_fp_iters, c.newton); // a[M+1..M+4]: CD FP n, Newton
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
                          req.transient_time + req.t_max, c.ad_axis, c.mode_2d ? c.ad_axis_2 : 0, req.adaptive, req.h);
    sweep_extend_state(req);
    return req;
}
}

namespace ucuda::analysis_detail {


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


std::function<void(ParametricEngine&)> SignalMetricsAnalysisSession::prewarm_task(int config_idx) const {
    if (config_idx < 0 || config_idx >= (int)configs.size()) return {};
    SignalMetricsRequest req = build_metrics_request(*this, configs[config_idx]);
    if (req.krs_body.empty()) return {};
    return [req](ParametricEngine& e) { e.prewarm(req); };
}


bool SignalMetricsAnalysisSession::run_async(ParametricEngine& engine, int config_idx) {
    if (in_flight) return false;
    if (config_idx < 0 || config_idx >= (int)configs.size()) return false;

    SignalMetricsConfig& c = configs[config_idx];
    c.last_error.clear();
    last_run_label.clear();

    SignalMetricsRequest req = build_metrics_request(*this, c);
    if (req.krs_body.empty()) {
        c.last_error = krs_empty_error(custom_schemes, sys, c.scheme);
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
    const std::string stop_note = stop_note_for(r.cancelled, r.ok, r.flags);
    std::string label;
    if (idx >= 0 && idx < (int)configs.size()) {
        label = configs[idx].label;
        if (!cancelled || partial_result_worth_showing(r.ok, r.flags)) apply_metrics_result(configs[idx], std::move(r));
    }
    last_run_completed_at = std::chrono::steady_clock::now();
    last_run_seconds = std::chrono::duration<double>(last_run_completed_at - compute_start_time).count();
    last_run_label = label.empty() ? std::string("Metrics") : label;
    last_run_label += stop_note;
    last_run_succeeded = !cancelled;
    log_run_completed(last_run_label.c_str(), last_run_succeeded, last_run_seconds);
    in_flight = false;
    running_config_index = -1;
    cancel_token.reset();
    progress_token.reset();
    return true;
}

namespace ucuda::analysis_detail {


// 1/h; при свипе по h на одной из осей — среднее по её сетке (шагов у каждой точки ~ T/h).
double work_inv_h(double h, bool sweep_x, double lo, double hi, int n, bool log_x,
                  bool sweep_y , double lo2 , double hi2 , bool log_y ) {
    if (sweep_x) return traj_mean_inv_h(lo, hi, n, log_x);
    if (sweep_y) return traj_mean_inv_h(lo2, hi2, n, log_y);
    return h > 0.0 ? 1.0 / h : 0.0;
}
}

namespace ucuda::analysis_detail {


// points — точек свипа, per_point — вызовов шага на точку. continuation на GPU — одна нить,
// проходящая все точки подряд.
TrajWork make_traj_work(const std::string& body, int n, bool adaptive, bool cpu, bool continuation,
                        double points, double per_point) {
    TrajWork w;
    if (body.empty()) { w.why = "no step body for this scheme"; return w; }
    if (adaptive)     { w.why = "adaptive step: the number of steps is not known in advance"; return w; }
    if (cpu)          { w.why = "computed on the CPU"; return w; }
    w.valid     = true;
    w.krs_body  = body;
    w.amountOfX = n;
    if (continuation) { w.threads = 1.0;    w.steps = points * per_point; }
    else              { w.threads = points; w.steps = per_point; }
    return w;
}
}

namespace ucuda::analysis_detail {


// LLE / LS: транзиент идёт одной траекторией, блоки NT (с транзиентом векторов) — основной и
// клонами: у LLE траекторий 2, у LS N + 1.
double lyap_steps(double inv_h, double transient, double t_max, double vtr, double trajectories) {
    return traj_steps(transient, inv_h) + trajectories * traj_steps(t_max + std::max(0.0, vtr), inv_h);
}
}
  // namespace

TrajWork BifurcationAnalysisSession::traj_work(int idx) const {
    if (idx < 0 || idx >= (int)diagrams.size()) return {};
    const BifurcationDiagramConfig& bd = diagrams[idx];
    if (bd.mode_2d) {
        const Bifurcation2DRequest r = build_bif2d_request(*this, bd);
        const double ih = work_inv_h(r.h, r.sweep_over_h, r.param_lo, r.param_hi, r.n_pts, r.log_scale,
                                     r.sweep_over_h_2, r.param_lo_2, r.param_hi_2, r.log_scale_2);
        return make_traj_work(r.krs_body, r.amountOfX, r.adaptive.enabled, false, false,
                              (double)r.n_pts * r.n_pts, traj_steps(r.transient_time + r.t_max, ih));
    }
    const Bifurcation1DRequest r = build_bif1d_request(*this, bd);
    const double ih = work_inv_h(r.h, r.sweep_over_h, r.param_lo, r.param_hi, r.n_pts, r.log_scale);
    return make_traj_work(r.krs_body, r.amountOfX, r.adaptive.enabled, r.use_cpu, r.continuation,
                          (double)r.n_pts, traj_steps(r.transient_time + r.t_max, ih));
}


TrajWork LLEAnalysisSession::traj_work(int idx) const {
    if (idx < 0 || idx >= (int)curves.size()) return {};
    const LLECurveConfig& c = curves[idx];
    if (c.mode_2d) {
        const LLE2DRequest r = build_lle2d_request(*this, c);
        const double ih = work_inv_h(r.h, r.sweep_over_h, r.param_lo, r.param_hi, r.n_pts, r.log_scale,
                                     r.sweep_over_h_2, r.param_lo_2, r.param_hi_2, r.log_scale_2);
        return make_traj_work(r.krs_body, r.amountOfX, r.adaptive.enabled, false, false,
                              (double)r.n_pts * r.n_pts,
                              lyap_steps(ih, r.transient_time, r.t_max, r.vector_transient, 2.0));
    }
    const LLE1DRequest r = build_lle1d_request(*this, c);
    const double ih = work_inv_h(r.h, r.sweep_over_h, r.param_lo, r.param_hi, r.n_pts, r.log_scale);
    return make_traj_work(r.krs_body, r.amountOfX, r.adaptive.enabled, r.use_cpu, r.continuation,
                          (double)r.n_pts, lyap_steps(ih, r.transient_time, r.t_max, r.vector_transient, 2.0));
}


TrajWork LyapunovSpectrumAnalysisSession::traj_work(int idx) const {
    if (idx < 0 || idx >= (int)curves.size()) return {};
    const LSCurveConfig& c = curves[idx];
    if (c.mode_2d) {
        const LS2DRequest r = build_ls2d_request(*this, c);
        const double ih = work_inv_h(r.h, r.sweep_over_h, r.param_lo, r.param_hi, r.n_pts, r.log_scale,
                                     r.sweep_over_h_2, r.param_lo_2, r.param_hi_2, r.log_scale_2);
        return make_traj_work(r.krs_body, r.amountOfX, r.adaptive.enabled, false, false,
                              (double)r.n_pts * r.n_pts,
                              lyap_steps(ih, r.transient_time, r.t_max, r.vector_transient, r.amountOfX + 1.0));
    }
    const LS1DRequest r = build_ls1d_request(*this, c);
    const double ih = work_inv_h(r.h, r.sweep_over_h, r.param_lo, r.param_hi, r.n_pts, r.log_scale);
    return make_traj_work(r.krs_body, r.amountOfX, r.adaptive.enabled, r.use_cpu, r.continuation,
                          (double)r.n_pts,
                          lyap_steps(ih, r.transient_time, r.t_max, r.vector_transient, r.amountOfX + 1.0));
}


TrajWork SignalMetricsAnalysisSession::traj_work(int idx) const {
    if (idx < 0 || idx >= (int)configs.size()) return {};
    const SignalMetricsRequest r = build_metrics_request(*this, configs[idx]);
    const double ih = work_inv_h(r.h, r.sweep_over_h, r.param_lo, r.param_hi, r.n_pts, r.log_scale,
                                 r.dimension == 2 && r.sweep_over_h_2, r.param_lo_2, r.param_hi_2, r.log_scale_2);
    const double points = r.dimension == 2 ? (double)r.n_pts * r.n_pts : (double)r.n_pts;
    return make_traj_work(r.krs_body, r.amountOfX, r.adaptive.enabled, r.use_cpu, r.continuation,
                          points, traj_steps(r.transient_time + r.t_max, ih));
}


TrajWork Dft1DAnalysisSession::traj_work(int idx) const {
    if (idx < 0 || idx >= (int)configs.size()) return {};
    const Dft1DRequest r = build_dft1d_request(*this, configs[idx]);
    const double ih = work_inv_h(r.h, r.sweep_over_h, r.param_lo, r.param_hi, r.n_pts, r.log_scale);
    return make_traj_work(r.krs_body, r.amountOfX, false, r.use_cpu, r.continuation,
                          (double)r.n_pts, traj_steps(r.transient_time + r.t_max, ih));
}


TrajWork BasinsAnalysisSession::traj_work(int idx) const {
    if (idx < 0 || idx >= (int)configs.size()) return {};
    const BasinsRequest r = build_basins_request(*this, configs[idx]);
    return make_traj_work(r.krs_body, r.amountOfX, r.adaptive.enabled, false, false,
                          (double)r.n_pts * r.n_pts,
                          traj_steps(r.transient_time + r.t_max, r.h > 0.0 ? 1.0 / r.h : 0.0));
}
