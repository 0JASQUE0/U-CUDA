#include "session_internal.h"

namespace ucuda::analysis_detail {


// Снапшот текущих GUI-полей конкретной БД в Bifurcation1DRequest. Делается на
// главном потоке перед стартом async-расчёта, чтобы worker работал со
// стабильной копией и пользователь мог в это время менять поля без race.
// KRS резолвится здесь же, не персистится в session.
Bifurcation1DRequest build_bif1d_request(const BifurcationAnalysisSession& s,
                                                const BifurcationDiagramConfig& bd) {
    Bifurcation1DRequest req;
    req.krs_body  = compute_krs_for_scheme(s.custom_schemes, s.sys, bd.scheme);
    req.amountOfX = (int)s.vars.size();

    req.initial_conditions.resize(req.amountOfX);
    for (int i = 0; i < req.amountOfX; ++i) {
        auto it = bd.initial_conditions.find(s.vars[i]);
        req.initial_conditions[i] = (it != bd.initial_conditions.end()) ? parse_d(it->second, 0.0) : 0.0;
    }

    // Конвенция codegen: a[0] зарезервирован, реальные параметры — с a[1].
    int nparams = (int)s.params.size();
    req.base_values.assign((size_t)nparams + 1, 0.0);
    req.base_values[0] = parse_d(bd.symmetry_s, 0.5); // CD: коэф. симметрии (a[0])
    for (int i = 0; i < nparams; ++i) {
        auto it = bd.param_values.find(s.params[i]);
        req.base_values[i + 1] = (it != bd.param_values.end()) ? parse_d(it->second, 0.0) : 0.0;
    }
    req.param_index = sweep_param_slot(bd.param_index, nparams);
    req.sweep_over_var = bd.sweep_over_var;
    req.sweep_over_h   = bd.sweep_over_h;
    req.log_scale      = bd.log_scale;
    req.var_sweep_index = (bd.var_sweep_index >= 0 && bd.var_sweep_index < req.amountOfX)
                          ? bd.var_sweep_index : 0;
    req.continuation = bd.continuation;
    req.continuation_reverse = bd.continuation_reverse;
    req.use_cpu = !bd.use_gpu;

    req.param_lo       = parse_d(bd.param_lo_text, 0.0);
    req.param_hi       = parse_d(bd.param_hi_text, 1.0);
    if (req.param_hi < req.param_lo) std::swap(req.param_lo, req.param_hi);
    req.n_pts          = parse_i(bd.n_pts_text, 500);
    // -1 — combination (см. loopCalculateDiscreteModel_int / draw_writable_var_combo).
    req.writable_var   = (bd.writable_var >= -1 && bd.writable_var < req.amountOfX) ? bd.writable_var : 0;
    req.h              = parse_d(bd.h_text, 0.01);
    req.t_max          = parse_d(bd.t_max_text, 100.0);
    req.transient_time = parse_d(bd.transient_text, 100.0);
    req.pre_scaller    = std::max(1, parse_i(bd.pre_scaller_text, 1));
    req.max_value      = parse_d(bd.max_value_text, 1.0e6);
    req.emit_all_samples = bd.plot_all_iterates;
    req.csv_output_path = bd.csv_save_enabled ? bd.csv_output_path : std::string{};
    fill_adaptive_request(s.custom_schemes, s.sys, bd.scheme, bd.adaptive, req.amountOfX,
                          req.transient_time + req.t_max, bd.ad_axis, 0, req.adaptive);
    return req;
}


// Применить только что готовый Bifurcation1DResult к конкретной БД.
void apply_bif1d_result(BifurcationDiagramConfig& bd, Bifurcation1DResult&& r) {
    bd.result = std::move(r);
    bd.last_run_ok = bd.result.ok;
    if (!bd.result.ok) bd.last_error = bd.result.error;
    bd.data_generation++;
    bd.fit_request = true;
}


Bifurcation2DRequest build_bif2d_request(const BifurcationAnalysisSession& s,
                                                const BifurcationDiagramConfig& bd) {
    Bifurcation2DRequest req;
    req.krs_body  = compute_krs_for_scheme(s.custom_schemes, s.sys, bd.scheme);
    req.amountOfX = (int)s.vars.size();

    req.initial_conditions.resize(req.amountOfX);
    for (int i = 0; i < req.amountOfX; ++i) {
        auto it = bd.initial_conditions.find(s.vars[i]);
        req.initial_conditions[i] = (it != bd.initial_conditions.end()) ? parse_d(it->second, 0.0) : 0.0;
    }

    int nparams = (int)s.params.size();
    req.base_values.assign((size_t)nparams + 1, 0.0);
    req.base_values[0] = parse_d(bd.symmetry_s, 0.5); // CD: коэф. симметрии (a[0])
    for (int i = 0; i < nparams; ++i) {
        auto it = bd.param_values.find(s.params[i]);
        req.base_values[i + 1] = (it != bd.param_values.end()) ? parse_d(it->second, 0.0) : 0.0;
    }

    req.sweep_over_var     = bd.sweep_over_var;
    req.sweep_over_var_2   = bd.sweep_over_var_2;
    req.sweep_over_h       = bd.sweep_over_h;
    req.sweep_over_h_2     = bd.sweep_over_h_2;
    req.log_scale          = bd.log_scale;
    req.log_scale_2        = bd.log_scale_2;
    req.param_index        = sweep_param_slot(bd.param_index, nparams);
    req.param_index_2      = sweep_param_slot(bd.param_index_2, nparams);
    req.var_sweep_index    = (bd.var_sweep_index >= 0 && bd.var_sweep_index < req.amountOfX)
                             ? bd.var_sweep_index : 0;
    req.var_sweep_index_2  = (bd.var_sweep_index_2 >= 0 && bd.var_sweep_index_2 < req.amountOfX)
                             ? bd.var_sweep_index_2 : 0;

    req.param_lo           = parse_d(bd.param_lo_text,   0.0);
    req.param_hi           = parse_d(bd.param_hi_text,   1.0);
    req.param_lo_2         = parse_d(bd.param_lo_2_text, 0.0);
    req.param_hi_2         = parse_d(bd.param_hi_2_text, 1.0);
    if (req.param_hi   < req.param_lo)   std::swap(req.param_lo,   req.param_hi);
    if (req.param_hi_2 < req.param_lo_2) std::swap(req.param_lo_2, req.param_hi_2);
    req.n_pts              = parse_i(bd.n_pts_text, 200);
    // -1 — combination, ровно как в build_bif1d_request. 2D-цепочка идёт через тот же
    // calculateDiscreteModelCUDA → loopCalculateDiscreteModel_int, где сентинел обработан явно, а
    // peakFinder/DBSCAN ниже видят просто скалярный ряд. Раньше здесь стоял кламп `>= 0` — он
    // писался до появления сентинела, поэтому комбинация молча превращалась в первую переменную.
    // Учтите, что eps_dbscan под комбинацию обычно нужен свой: сумма x + pi*y + e*z живёт в другом
    // масштабе, чем одна переменная.
    req.writable_var       = (bd.writable_var >= -1 && bd.writable_var < req.amountOfX) ? bd.writable_var : 0;
    req.h                  = parse_d(bd.h_text, 0.01);
    req.t_max              = parse_d(bd.t_max_text, 100.0);
    req.transient_time     = parse_d(bd.transient_text, 100.0);
    req.pre_scaller        = std::max(1, parse_i(bd.pre_scaller_text, 1));
    req.max_value          = parse_d(bd.max_value_text, 1.0e6);
    req.emit_all_samples   = bd.plot_all_iterates;
    req.eps_dbscan         = parse_d(bd.eps_dbscan_text, 0.1);
    req.mult_peak          = parse_d(bd.mult_peak_text,     (double)::mult_peak);
    req.mult_interval      = parse_d(bd.mult_interval_text, (double)::mult_interval);
    req.csv_output_path    = bd.csv_save_enabled ? bd.csv_output_path : std::string{};
    fill_adaptive_request(s.custom_schemes, s.sys, bd.scheme, bd.adaptive, req.amountOfX,
                          req.transient_time + req.t_max, bd.ad_axis, bd.ad_axis_2, req.adaptive);
    return req;
}


void apply_bif2d_result(BifurcationDiagramConfig& bd, Bifurcation2DResult&& r) {
    bd.result_2d = std::move(r);
    bd.last_run_2d_ok = bd.result_2d.ok;
    if (!bd.result_2d.ok) bd.last_error = bd.result_2d.error;
    bd.data_generation_2d++;
    bd.fit_request_2d = true;
}
}
using namespace ucuda::analysis_detail;


void BifurcationAnalysisSession::load_from_record(const SystemRecord& r,
    const std::vector<std::string>& vars_,
    const std::vector<std::string>& params_) {
    vars = vars_;
    params = params_;
    custom_schemes = r.custom_schemes;
    enabled_builtin_schemes = enabled_builtins_from_record(r);

    // Сбрасываем список БД и создаём один дефолтный с настройками из record.
    diagrams.clear();
    BifurcationDiagramConfig bd;
    bd.label = "BD 1";
    bd.label_is_manual = false;   // fresh diagram → auto-label
    bd.h_text = default_h_from_record(r);
    bd.scheme = default_scheme_from_record(r, bd.scheme);
    bd.plot_all_iterates = default_all_iterates_from_record(r);
    bd.symmetry_s = r.symmetry_s.empty() ? std::string("0.5") : r.symmetry_s;

    for (const auto& p : params) {
        auto it = r.param_values.find(p);
        bd.param_values[p] = (it != r.param_values.end()) ? it->second : "";
    }
    for (const auto& v : vars) {
        auto it = r.init_conditions.find(v);
        bd.initial_conditions[v] = (it != r.init_conditions.end()) ? it->second : "";
    }
    if (!params.empty()) {
        bd.param_index = 0;
    }
    diagrams.push_back(std::move(bd));

    active_diagram_index = 0;
    running_diagram_index = -1;
}




void BifurcationAnalysisSession::remove_diagram(int i) {
    if (i < 0 || i >= (int)diagrams.size()) return;
    // Запрещаем удалять БД, чей расчёт сейчас идёт — иначе worker положит
    // результат в несуществующий слот при poll().
    if (in_flight && running_diagram_index == i) return;
    diagrams.erase(diagrams.begin() + i);
    if (in_flight && running_diagram_index > i) running_diagram_index--;
    if (active_diagram_index >= (int)diagrams.size())
        active_diagram_index = (int)diagrams.size() - 1;
    if (active_diagram_index < 0) active_diagram_index = 0;
}


bool BifurcationAnalysisSession::run(ParametricEngine& engine, int diagram_idx) {
    if (diagram_idx < 0 || diagram_idx >= (int)diagrams.size()) return false;
    BifurcationDiagramConfig& bd = diagrams[diagram_idx];
    // Keep last_run_ok / last_run_2d_ok set from the previous successful
    // run so the GUI keeps drawing the old plot until apply_*_result swaps
    // in the new data.
    bd.last_error.clear();

    if (bd.mode_2d) {
        Bifurcation2DRequest req = build_bif2d_request(*this, bd);
        if (req.krs_body.empty()) {
            bd.last_error = "krs_code is empty (no valid system or scheme)";
            return false;
        }
        Bifurcation2DResult r = engine.run_bifurcation_2d(req);
        bool ok = r.ok;
        apply_bif2d_result(bd, std::move(r));
        return ok;
    } else {
        Bifurcation1DRequest req = build_bif1d_request(*this, bd);
        if (req.krs_body.empty()) {
            bd.last_error = "krs_code is empty (no valid system or scheme)";
            return false;
        }
        Bifurcation1DResult r = engine.run_bifurcation_1d(req);
        bool ok = r.ok;
        apply_bif1d_result(bd, std::move(r));
        return ok;
    }
}


bool BifurcationAnalysisSession::run_async(ParametricEngine& engine, int diagram_idx) {
    if (in_flight) return false;
    if (diagram_idx < 0 || diagram_idx >= (int)diagrams.size()) return false;

    BifurcationDiagramConfig& bd = diagrams[diagram_idx];
    // Keep last_run_ok / last_run_2d_ok set from the previous successful
    // run so the GUI keeps drawing the old plot until apply_*_result swaps
    // in the new data.
    bd.last_error.clear();
    last_run_label.clear();
    cancel_token   = std::make_shared<std::atomic<bool>>(false);
    progress_token = std::make_shared<std::atomic<float>>(0.0f);

    if (bd.mode_2d) {
        Bifurcation2DRequest req = build_bif2d_request(*this, bd);
        if (req.krs_body.empty()) {
            bd.last_error = "krs_code is empty (no valid system or scheme)";
            cancel_token.reset();
            progress_token.reset();
            return false;
        }
        req.cancel = cancel_token;
        req.progress = progress_token;
        in_flight = true;
        is_2d_run = true;
        running_diagram_index = diagram_idx;
        compute_start_time = std::chrono::steady_clock::now();
        run_future_2d = std::async(std::launch::async, [&engine, req = std::move(req)]() {
            return engine.run_bifurcation_2d(req);
        });
    } else {
        Bifurcation1DRequest req = build_bif1d_request(*this, bd);
        if (req.krs_body.empty()) {
            bd.last_error = "krs_code is empty (no valid system or scheme)";
            cancel_token.reset();
            progress_token.reset();
            return false;
        }
        req.cancel = cancel_token;
        req.progress = progress_token;
        in_flight = true;
        is_2d_run = false;
        running_diagram_index = diagram_idx;
        compute_start_time = std::chrono::steady_clock::now();
        run_future = std::async(std::launch::async, [&engine, req = std::move(req)]() {
            return engine.run_bifurcation_1d(req);
        });
    }
    return true;
}


void BifurcationAnalysisSession::request_cancel() {
    if (cancel_token) cancel_token->store(true, std::memory_order_relaxed);
}


bool BifurcationAnalysisSession::poll() {
    if (!in_flight) return false;
    bool cancelled = false;
    std::string label;
    int idx = running_diagram_index;
    if (idx >= 0 && idx < (int)diagrams.size()) label = diagrams[idx].label;
    if (is_2d_run) {
        if (!run_future_2d.valid()) { in_flight = false; running_diagram_index = -1; cancel_token.reset(); progress_token.reset(); return false; }
        if (run_future_2d.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
        Bifurcation2DResult r = run_future_2d.get();
        cancelled = r.cancelled;
        if (!cancelled && idx >= 0 && idx < (int)diagrams.size()) {
            apply_bif2d_result(diagrams[idx], std::move(r));
        }
    } else {
        if (!run_future.valid()) { in_flight = false; running_diagram_index = -1; cancel_token.reset(); progress_token.reset(); return false; }
        if (run_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
        Bifurcation1DResult r = run_future.get();
        cancelled = r.cancelled;
        if (!cancelled && idx >= 0 && idx < (int)diagrams.size()) {
            apply_bif1d_result(diagrams[idx], std::move(r));
        }
    }
    last_run_completed_at = std::chrono::steady_clock::now();
    last_run_seconds = std::chrono::duration<double>(last_run_completed_at - compute_start_time).count();
    last_run_label = label.empty() ? std::string("bifurcation") : label;
    last_run_succeeded = !cancelled;
    log_run_completed(last_run_label.c_str(), last_run_succeeded, last_run_seconds);
    in_flight = false;
    running_diagram_index = -1;
    cancel_token.reset();
    progress_token.reset();
    return true;
}
