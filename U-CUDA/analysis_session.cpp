#include "analysis_session.h"
#include "phase_portrait_nvrtc.h"
#include "integrator.h"
#include <cstdlib>
#include <cmath>
#include <memory>
#include <chrono>

// Парсит значение (число или дробь a/b) в double; пусто -> fallback.
static double parse_val(const std::string& s, double fallback) {
    if (s.empty()) return fallback;
    size_t slash = s.find('/');
    if (slash != std::string::npos) {
        double num = std::atof(s.substr(0, slash).c_str());
        double den = std::atof(s.substr(slash + 1).c_str());
        if (den != 0) return num / den;
    }
    return std::atof(s.c_str());
}

void PhaseAnalysisSession::add_ic() {
    InitialConditionSet ic;
    ic.label = "IC " + std::to_string(ic_sets.size() + 1);
    for (const auto& v : vars) ic.values[v] = "";
    ic_sets.push_back(ic);
}

void PhaseAnalysisSession::remove_ic(int i) {
    if (i >= 0 && i < (int)ic_sets.size()) ic_sets.erase(ic_sets.begin() + i);
}

void PhaseAnalysisSession::add_projection() {
    Projection p;
    int n = (int)projections.size();
    p.label = "Projection " + std::to_string(n + 1);
    p.axis_x = 0;
    p.axis_y = (vars.size() > 1) ? 1 : 0;
    p.axis_z = (vars.size() > 2) ? 2 : 0;
    // для time domain: по умолчанию показываем все переменные
    p.show_var.assign(vars.size(), true);
    projections.push_back(std::move(p));
}

void PhaseAnalysisSession::remove_projection(int i) {
    if (i >= 0 && i < (int)projections.size()) {
        projections.erase(projections.begin() + i);
    }
}

void PhaseAnalysisSession::load_from_record(const SystemRecord& r,
    const std::vector<std::string>& vars_,
    const std::vector<std::string>& params_) {
    vars = vars_;
    params = params_;
    custom_schemes = r.custom_schemes;
    step_h = r.step_h.empty() ? "0.01" : r.step_h;
    sim_time = "50";
    skip_time = "10";

    param_values.clear();
    for (const auto& p : params) {
        auto it = r.param_values.find(p);
        param_values[p] = (it != r.param_values.end()) ? it->second : "";
    }

    ic_sets.clear();
    InitialConditionSet ic;
    ic.label = "IC 1";
    for (const auto& v : vars) {
        auto it = r.init_conditions.find(v);
        ic.values[v] = (it != r.init_conditions.end()) ? it->second : "";
    }
    ic_sets.push_back(ic);

    // одна проекция по умолчанию
    projections.clear();
    add_projection();

    result = AnalysisResult{};
}

// Снапшот всех входов phase-расчёта. Создаётся на главном потоке, передаётся
// worker'у при async-режиме. Все поля — by value, никакой ссылки на session.
namespace {
struct PhaseRunInputs {
    std::vector<std::string> vars;
    std::vector<std::string> params;
    std::string step_h, sim_time, skip_time;
    std::string scheme, decimation;
    bool        use_gpu = true;
    System      sys;
    std::string krs_code;
    std::map<std::string, std::string>  param_values;
    std::vector<InitialConditionSet>    ic_sets;
};
} // namespace

// Чистая функция: входы → AnalysisResult. Не трогает session, поэтому её
// безопасно звать с любого потока.
static AnalysisResult compute_phase_portrait(const PhaseRunInputs& in) {
    AnalysisResult result;
    auto _t0 = std::chrono::high_resolution_clock::now();

    int dim = (int)in.vars.size();
    if (dim < 1) { result.error = "no variables"; return result; }
    double h = parse_val(in.step_h, 0.01);
    if (h <= 0) { result.error = "step h must be > 0"; return result; }
    double tsim = parse_val(in.sim_time, 50.0);
    double tskip = parse_val(in.skip_time, 0.0);
    int total = (int)(tsim / h); if (total <= 0) total = 1;
    int skip = (int)(tskip / h); if (skip < 0) skip = 0;

    // параметры: a[0] зарезервирован, a[1..] = params
    int nparams = (int)in.params.size();
    std::vector<double> a(nparams + 1, 0.0);
    for (int j = 0; j < nparams; ++j) {
        auto it = in.param_values.find(in.params[j]);
        a[1 + j] = parse_val(it != in.param_values.end() ? it->second : "", 0.0);
    }

    int dec = std::atoi(in.decimation.c_str()); if (dec < 1) dec = 1;

    int N = (int)in.ic_sets.size();
    if (N < 1) { result.error = "no initial conditions"; return result; }

    std::vector<double> ic_flat((size_t)N * dim, 0.0);
    for (int k = 0; k < N; ++k) {
        const auto& ic = in.ic_sets[k];
        for (int i = 0; i < dim; ++i) {
            auto it = ic.values.find(in.vars[i]);
            ic_flat[(size_t)k * dim + i] = parse_val(it != ic.values.end() ? it->second : "", 0.0);
        }
    }

    std::vector<std::vector<std::vector<double>>> raw;
    bool calc_ok = true;
    std::string calc_err;

    if (in.use_gpu) {
        if (in.krs_code.empty()) {
            result.error = "no KRS generated (check system)";
            return result;
        }
        calc_ok = computePhasePortraitsNVRTC(in.krs_code, dim,
            ic_flat.data(), N, a.data(), (int)a.size(),
            h, total, skip, raw, &calc_err);
        if (!calc_ok) {
            result.error = "GPU: " + calc_err;
            return result;
        }
    }
    else {
        std::unique_ptr<SystemEvaluator> evaluator;
        try {
            evaluator.reset(new SystemEvaluator(in.sys));
        }
        catch (const std::exception& e) {
            result.error = std::string("parse error: ") + e.what();
            return result;
        }
        raw.resize(N);
        for (int k = 0; k < N; ++k) {
            bool ok = computePhasePortraitCPU(*evaluator, int_scheme_from_string(in.scheme),
                &ic_flat[(size_t)k * dim], dim, a.data(), (int)a.size(),
                h, total, skip, raw[k]);
            if (!ok) { calc_ok = false; calc_err = "trajectory '" + in.ic_sets[k].label + "' diverged (nan/inf)"; }
        }
    }

    for (int k = 0; k < N; ++k) {
        const auto& ic = in.ic_sets[k];
        std::vector<std::vector<double>>& traj = raw[k];

        std::vector<std::vector<double>> dtraj;
        if (dec <= 1) dtraj = std::move(traj);
        else {
            dtraj.reserve(traj.size() / dec + 1);
            for (size_t t = 0; t < traj.size(); t += dec) dtraj.push_back(traj[t]);
        }
        result.trajectories.push_back(std::move(dtraj));
        result.labels.push_back(ic.label);
        result.visible.push_back(ic.visible);

        std::string txt = "(";
        for (int i = 0; i < dim; ++i) {
            auto it = ic.values.find(in.vars[i]);
            std::string val = it != ic.values.end() ? it->second : "";
            if (val.empty()) val = "0";
            txt += val; if (i + 1 < dim) txt += ",";
        }
        txt += ")";
        result.ic_text.push_back(txt);
    }

    result.ok = result.trajectories.size() > 0;
    if (!calc_ok && result.ok) result.error = calc_err;

    auto _t1 = std::chrono::high_resolution_clock::now();
    double _ms = std::chrono::duration<double, std::milli>(_t1 - _t0).count();
    if (result.error.empty())
        result.error = "recompute: " + std::to_string(_ms) + " ms";

    return result;
}

// Снапшот текущей session в PhaseRunInputs. Делается на главном потоке.
static PhaseRunInputs snapshot_phase(PhaseAnalysisSession& s) {
    // krs_code должен быть свежим — если пустой, генерим перед снапшотом.
    if (s.krs_code.empty()) s.regenerate_krs();
    PhaseRunInputs in;
    in.vars         = s.vars;
    in.params       = s.params;
    in.step_h       = s.step_h;
    in.sim_time     = s.sim_time;
    in.skip_time    = s.skip_time;
    in.scheme       = s.scheme;
    in.decimation   = s.decimation;
    in.use_gpu      = s.use_gpu;
    in.sys          = s.sys;
    in.krs_code     = s.krs_code;
    in.param_values = s.param_values;
    in.ic_sets      = s.ic_sets;
    return in;
}

void PhaseAnalysisSession::recompute() {
    PhaseRunInputs in = snapshot_phase(*this);
    result = compute_phase_portrait(in);
    fit_request = true;
    data_generation++;
}

bool PhaseAnalysisSession::recompute_async() {
    if (in_flight) return false;
    PhaseRunInputs in = snapshot_phase(*this);
    in_flight = true;
    compute_start_time = std::chrono::steady_clock::now();
    recompute_future = std::async(std::launch::async, [in = std::move(in)]() {
        return compute_phase_portrait(in);
    });
    return true;
}

bool PhaseAnalysisSession::poll() {
    if (!in_flight) return false;
    if (!recompute_future.valid()) { in_flight = false; return false; }
    if (recompute_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
    result = recompute_future.get();
    fit_request = true;
    data_generation++;
    in_flight = false;
    return true;
}


// --- генерация КРС из системы по выбранному методу (для NVRTC) ---
static Scheme scheme_from_string(const std::string& s) {
    if (s == "Euler-Cromer")      return Scheme::EulerCromer;
    if (s == "Explicit Midpoint") return Scheme::ExplicitMidpoint;
    if (s == "RK4")               return Scheme::RK4;
    if (s == "DOPRI78")           return Scheme::DOPRI78;
    return Scheme::Euler;
}

void PhaseAnalysisSession::regenerate_krs() {
    krs_code.clear();
    // Сначала ищем среди custom — имя имеет приоритет над built-in (если бы
    // совпали, что мы блокируем в System tab).
    for (const auto& cs : custom_schemes) {
        if (cs.name == scheme) { krs_code = cs.body; return; }
    }
    if (sys.rhs.empty()) return; // нет уравнений — нечего генерировать
    try {
        krs_code = codegen_scheme(sys, scheme_from_string(scheme));
    }
    catch (...) {
        krs_code.clear();
    }
}

// --- BifurcationAnalysisSession ---

// Резолвит КРС по имени scheme: сперва среди custom_schemes (имя имеет
// приоритет над built-in, что блокируется в System tab), иначе генерирует
// через codegen_scheme. Чистая функция — зовётся при сборке Request в
// момент Run (не персистится). Переиспользуется bifurcation и LLE.
static std::string compute_krs_for_scheme(const std::vector<CustomScheme>& custom_schemes,
                                          const System& sys,
                                          const std::string& scheme) {
    for (const auto& cs : custom_schemes) {
        if (cs.name == scheme) return cs.body;
    }
    if (sys.rhs.empty()) return {};
    try {
        return codegen_scheme(sys, scheme_from_string(scheme));
    }
    catch (...) {
        return {};
    }
}

void BifurcationAnalysisSession::load_from_record(const SystemRecord& r,
    const std::vector<std::string>& vars_,
    const std::vector<std::string>& params_) {
    vars = vars_;
    params = params_;
    custom_schemes = r.custom_schemes;

    // Сбрасываем список БД и создаём один дефолтный с настройками из record.
    diagrams.clear();
    BifurcationDiagramConfig bd;
    bd.label = "BD 1";
    bd.h_text = r.step_h.empty() ? std::string("0.01") : r.step_h;

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
        const std::string& pname = params[0];
        auto lo = r.param_min.find(pname);
        auto hi = r.param_max.find(pname);
        if (lo != r.param_min.end() && !lo->second.empty()) bd.param_lo_text = lo->second;
        if (hi != r.param_max.end() && !hi->second.empty()) bd.param_hi_text = hi->second;
    }
    diagrams.push_back(std::move(bd));

    active_diagram_index = 0;
    running_diagram_index = -1;
}

void BifurcationAnalysisSession::add_diagram() {
    BifurcationDiagramConfig bd;
    if (!diagrams.empty()) {
        // Глубокая копия последней БД — пользователь обычно правит 1-2 поля.
        bd = diagrams.back();
        // Свежий результат: не наследуем data старой БД.
        bd.result = Bifurcation1DResult{};
        bd.last_run_ok = false;
        bd.last_error.clear();
        bd.data_generation = 0;
        bd.fit_request = false;
    } else {
        // Пустая сессия (без vars/params) — оставляем дефолты конструктора.
        for (const auto& v : vars) bd.initial_conditions[v] = "";
        for (const auto& p : params) bd.param_values[p] = "";
    }
    bd.label = "BD " + std::to_string(diagrams.size() + 1);
    diagrams.push_back(std::move(bd));
    active_diagram_index = (int)diagrams.size() - 1;
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

// Парсит double, понимая дробь "a/b" — поведение симметрично с parse_val
// (см. строки 10–18 этого же файла, которой пользуется Phase). Иначе
// Parametric получал бы "8/3" как 8, а Phase как 2.667.
static double parse_d(const std::string& s, double def) {
    if (s.empty()) return def;
    size_t slash = s.find('/');
    if (slash != std::string::npos) {
        double num = std::atof(s.substr(0, slash).c_str());
        double den = std::atof(s.substr(slash + 1).c_str());
        if (den != 0) return num / den;
    }
    return std::atof(s.c_str());   // atof не кидает, "abc"→0
}
static int parse_i(const std::string& s, int def) {
    if (s.empty()) return def;
    try { return std::stoi(s); } catch (...) { return def; }
}

// Снапшот текущих GUI-полей конкретной БД в Bifurcation1DRequest. Делается на
// главном потоке перед стартом async-расчёта, чтобы worker работал со
// стабильной копией и пользователь мог в это время менять поля без race.
// KRS резолвится здесь же, не персистится в session.
static Bifurcation1DRequest build_bif1d_request(const BifurcationAnalysisSession& s,
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
    for (int i = 0; i < nparams; ++i) {
        auto it = bd.param_values.find(s.params[i]);
        req.base_values[i + 1] = (it != bd.param_values.end()) ? parse_d(it->second, 0.0) : 0.0;
    }
    req.param_index = (bd.param_index >= 0 && bd.param_index < nparams) ? bd.param_index + 1 : 1;
    req.sweep_over_var = bd.sweep_over_var;
    req.var_sweep_index = (bd.var_sweep_index >= 0 && bd.var_sweep_index < req.amountOfX)
                          ? bd.var_sweep_index : 0;
    req.continuation = bd.continuation;
    req.continuation_reverse = bd.continuation_reverse;

    req.param_lo       = parse_d(bd.param_lo_text, 0.0);
    req.param_hi       = parse_d(bd.param_hi_text, 1.0);
    req.n_pts          = parse_i(bd.n_pts_text, 500);
    req.writable_var   = (bd.writable_var >= 0 && bd.writable_var < req.amountOfX) ? bd.writable_var : 0;
    req.h              = parse_d(bd.h_text, 0.01);
    req.t_max          = parse_d(bd.t_max_text, 100.0);
    req.transient_time = parse_d(bd.transient_text, 100.0);
    req.pre_scaller    = std::max(1, parse_i(bd.pre_scaller_text, 1));
    req.max_value      = parse_d(bd.max_value_text, 1.0e6);
    req.csv_output_path = bd.csv_save_enabled ? bd.csv_output_path : std::string{};
    return req;
}

// Применить только что готовый Bifurcation1DResult к конкретной БД.
static void apply_bif1d_result(BifurcationDiagramConfig& bd, Bifurcation1DResult&& r) {
    bd.result = std::move(r);
    bd.last_run_ok = bd.result.ok;
    if (!bd.result.ok) bd.last_error = bd.result.error;
    bd.data_generation++;
    bd.fit_request = true;
}

static Bifurcation2DRequest build_bif2d_request(const BifurcationAnalysisSession& s,
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
    for (int i = 0; i < nparams; ++i) {
        auto it = bd.param_values.find(s.params[i]);
        req.base_values[i + 1] = (it != bd.param_values.end()) ? parse_d(it->second, 0.0) : 0.0;
    }

    req.sweep_over_var     = bd.sweep_over_var;
    req.sweep_over_var_2   = bd.sweep_over_var_2;
    req.param_index        = (bd.param_index >= 0 && bd.param_index < nparams) ? bd.param_index + 1 : 1;
    req.param_index_2      = (bd.param_index_2 >= 0 && bd.param_index_2 < nparams) ? bd.param_index_2 + 1 : 1;
    req.var_sweep_index    = (bd.var_sweep_index >= 0 && bd.var_sweep_index < req.amountOfX)
                             ? bd.var_sweep_index : 0;
    req.var_sweep_index_2  = (bd.var_sweep_index_2 >= 0 && bd.var_sweep_index_2 < req.amountOfX)
                             ? bd.var_sweep_index_2 : 0;

    req.param_lo           = parse_d(bd.param_lo_text,   0.0);
    req.param_hi           = parse_d(bd.param_hi_text,   1.0);
    req.param_lo_2         = parse_d(bd.param_lo_2_text, 0.0);
    req.param_hi_2         = parse_d(bd.param_hi_2_text, 1.0);
    req.n_pts              = parse_i(bd.n_pts_text, 200);
    req.writable_var       = (bd.writable_var >= 0 && bd.writable_var < req.amountOfX) ? bd.writable_var : 0;
    req.h                  = parse_d(bd.h_text, 0.01);
    req.t_max              = parse_d(bd.t_max_text, 100.0);
    req.transient_time     = parse_d(bd.transient_text, 100.0);
    req.pre_scaller        = std::max(1, parse_i(bd.pre_scaller_text, 1));
    req.max_value          = parse_d(bd.max_value_text, 1.0e6);
    req.eps_dbscan         = parse_d(bd.eps_dbscan_text, 0.1);
    req.csv_output_path    = bd.csv_save_enabled ? bd.csv_output_path : std::string{};
    return req;
}

static void apply_bif2d_result(BifurcationDiagramConfig& bd, Bifurcation2DResult&& r) {
    bd.result_2d = std::move(r);
    bd.last_run_2d_ok = bd.result_2d.ok;
    if (!bd.result_2d.ok) bd.last_error = bd.result_2d.error;
    bd.data_generation_2d++;
    bd.fit_request_2d = true;
}

bool BifurcationAnalysisSession::run(ParametricEngine& engine, int diagram_idx) {
    if (diagram_idx < 0 || diagram_idx >= (int)diagrams.size()) return false;
    BifurcationDiagramConfig& bd = diagrams[diagram_idx];
    bd.last_run_ok = false;
    bd.last_run_2d_ok = false;
    bd.last_error.clear();

    if (bd.mode_2d) {
        Bifurcation2DRequest req = build_bif2d_request(*this, bd);
        if (req.krs_body.empty()) {
            bd.last_error = "krs_code пуст (нет валидной системы или scheme)";
            return false;
        }
        Bifurcation2DResult r = engine.run_bifurcation_2d(req);
        bool ok = r.ok;
        apply_bif2d_result(bd, std::move(r));
        return ok;
    } else {
        Bifurcation1DRequest req = build_bif1d_request(*this, bd);
        if (req.krs_body.empty()) {
            bd.last_error = "krs_code пуст (нет валидной системы или scheme)";
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
    bd.last_run_ok = false;
    bd.last_run_2d_ok = false;
    bd.last_error.clear();

    if (bd.mode_2d) {
        Bifurcation2DRequest req = build_bif2d_request(*this, bd);
        if (req.krs_body.empty()) {
            bd.last_error = "krs_code пуст (нет валидной системы или scheme)";
            return false;
        }
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
            bd.last_error = "krs_code пуст (нет валидной системы или scheme)";
            return false;
        }
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

bool BifurcationAnalysisSession::poll() {
    if (!in_flight) return false;
    if (is_2d_run) {
        if (!run_future_2d.valid()) { in_flight = false; running_diagram_index = -1; return false; }
        if (run_future_2d.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
        Bifurcation2DResult r = run_future_2d.get();
        int idx = running_diagram_index;
        if (idx >= 0 && idx < (int)diagrams.size()) {
            apply_bif2d_result(diagrams[idx], std::move(r));
        }
    } else {
        if (!run_future.valid()) { in_flight = false; running_diagram_index = -1; return false; }
        if (run_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
        Bifurcation1DResult r = run_future.get();
        int idx = running_diagram_index;
        if (idx >= 0 && idx < (int)diagrams.size()) {
            apply_bif1d_result(diagrams[idx], std::move(r));
        }
    }
    in_flight = false;
    running_diagram_index = -1;
    return true;
}

// ============================================================================
// LLEAnalysisSession — структурно зеркалит BifurcationAnalysisSession.
// Та же async-машинерия, та же per-curve конфигурация, тот же compute_krs_for_scheme.
// ============================================================================

void LLEAnalysisSession::load_from_record(const SystemRecord& r,
    const std::vector<std::string>& vars_,
    const std::vector<std::string>& params_) {
    vars = vars_;
    params = params_;
    custom_schemes = r.custom_schemes;

    curves.clear();
    LLECurveConfig c;
    c.label = "LLE 1";
    c.h_text = r.step_h.empty() ? std::string("0.01") : r.step_h;

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
        const std::string& pname = params[0];
        auto lo = r.param_min.find(pname);
        auto hi = r.param_max.find(pname);
        if (lo != r.param_min.end() && !lo->second.empty()) c.param_lo_text = lo->second;
        if (hi != r.param_max.end() && !hi->second.empty()) c.param_hi_text = hi->second;
    }
    curves.push_back(std::move(c));

    active_curve_index = 0;
    running_curve_index = -1;
}

void LLEAnalysisSession::add_curve() {
    LLECurveConfig c;
    if (!curves.empty()) {
        c = curves.back();
        c.result = LLE1DResult{};
        c.last_run_ok = false;
        c.last_error.clear();
        c.data_generation = 0;
        c.fit_request = false;
    } else {
        for (const auto& v : vars) c.initial_conditions[v] = "";
        for (const auto& p : params) c.param_values[p] = "";
    }
    c.label = "LLE " + std::to_string(curves.size() + 1);
    curves.push_back(std::move(c));
    active_curve_index = (int)curves.size() - 1;
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

static LLE1DRequest build_lle1d_request(const LLEAnalysisSession& s,
                                        const LLECurveConfig& c) {
    LLE1DRequest req;
    req.krs_body  = compute_krs_for_scheme(s.custom_schemes, s.sys, c.scheme);
    req.amountOfX = (int)s.vars.size();

    req.initial_conditions.resize(req.amountOfX);
    for (int i = 0; i < req.amountOfX; ++i) {
        auto it = c.initial_conditions.find(s.vars[i]);
        req.initial_conditions[i] = (it != c.initial_conditions.end()) ? parse_d(it->second, 0.0) : 0.0;
    }

    int nparams = (int)s.params.size();
    req.base_values.assign((size_t)nparams + 1, 0.0);
    for (int i = 0; i < nparams; ++i) {
        auto it = c.param_values.find(s.params[i]);
        req.base_values[i + 1] = (it != c.param_values.end()) ? parse_d(it->second, 0.0) : 0.0;
    }
    req.param_index = (c.param_index >= 0 && c.param_index < nparams) ? c.param_index + 1 : 1;
    req.sweep_over_var = c.sweep_over_var;
    req.var_sweep_index = (c.var_sweep_index >= 0 && c.var_sweep_index < req.amountOfX)
                          ? c.var_sweep_index : 0;

    req.param_lo       = parse_d(c.param_lo_text, 0.0);
    req.param_hi       = parse_d(c.param_hi_text, 1.0);
    req.n_pts          = parse_i(c.n_pts_text, 500);
    req.h              = parse_d(c.h_text, 0.01);
    req.t_max          = parse_d(c.t_max_text, 100.0);
    req.transient_time = parse_d(c.transient_text, 100.0);
    req.max_value      = parse_d(c.max_value_text, 1.0e6);
    req.NT             = parse_d(c.nt_text,  1.0);
    req.eps            = parse_d(c.eps_text, 1.0e-4);
    req.csv_output_path = c.csv_save_enabled ? c.csv_output_path : std::string{};
    return req;
}

static void apply_lle1d_result(LLECurveConfig& c, LLE1DResult&& r) {
    c.result = std::move(r);
    c.last_run_ok = c.result.ok;
    if (!c.result.ok) c.last_error = c.result.error;
    c.data_generation++;
    c.fit_request = true;
}

// Снапшот для 2D-режима. Те же общие поля, что у 1D, + поля второй оси и
// mixed_mode. Несимметричный режим (одна param, другая IC без mixed_mode)
// engine отвергает в run_lle_2d — здесь только пробрасываем поля.
static LLE2DRequest build_lle2d_request(const LLEAnalysisSession& s,
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
    for (int i = 0; i < nparams; ++i) {
        auto it = c.param_values.find(s.params[i]);
        req.base_values[i + 1] = (it != c.param_values.end()) ? parse_d(it->second, 0.0) : 0.0;
    }

    req.sweep_over_var     = c.sweep_over_var;
    req.sweep_over_var_2   = c.sweep_over_var_2;
    req.param_index        = (c.param_index >= 0 && c.param_index < nparams) ? c.param_index + 1 : 1;
    req.param_index_2      = (c.param_index_2 >= 0 && c.param_index_2 < nparams) ? c.param_index_2 + 1 : 1;
    req.var_sweep_index    = (c.var_sweep_index >= 0 && c.var_sweep_index < req.amountOfX)
                             ? c.var_sweep_index : 0;
    req.var_sweep_index_2  = (c.var_sweep_index_2 >= 0 && c.var_sweep_index_2 < req.amountOfX)
                             ? c.var_sweep_index_2 : 0;

    req.param_lo           = parse_d(c.param_lo_text,   0.0);
    req.param_hi           = parse_d(c.param_hi_text,   1.0);
    req.param_lo_2         = parse_d(c.param_lo_2_text, 0.0);
    req.param_hi_2         = parse_d(c.param_hi_2_text, 1.0);
    req.n_pts              = parse_i(c.n_pts_text, 200);
    req.h                  = parse_d(c.h_text, 0.01);
    req.t_max              = parse_d(c.t_max_text, 100.0);
    req.transient_time     = parse_d(c.transient_text, 100.0);
    req.max_value          = parse_d(c.max_value_text, 1.0e6);
    req.NT                 = parse_d(c.nt_text,  1.0);
    req.eps                = parse_d(c.eps_text, 1.0e-4);
    req.csv_output_path    = c.csv_save_enabled ? c.csv_output_path : std::string{};
    return req;
}

static void apply_lle2d_result(LLECurveConfig& c, LLE2DResult&& r) {
    c.result_2d = std::move(r);
    c.last_run_2d_ok = c.result_2d.ok;
    if (!c.result_2d.ok) c.last_error = c.result_2d.error;
    c.data_generation_2d++;
    c.fit_request_2d = true;
}

bool LLEAnalysisSession::run(ParametricEngine& engine, int curve_idx) {
    if (curve_idx < 0 || curve_idx >= (int)curves.size()) return false;
    LLECurveConfig& c = curves[curve_idx];
    c.last_run_ok = false;
    c.last_run_2d_ok = false;
    c.last_error.clear();

    if (c.mode_2d) {
        LLE2DRequest req = build_lle2d_request(*this, c);
        if (req.krs_body.empty()) {
            c.last_error = "krs_code пуст (нет валидной системы или scheme)";
            return false;
        }
        LLE2DResult r = engine.run_lle_2d(req);
        bool ok = r.ok;
        apply_lle2d_result(c, std::move(r));
        return ok;
    } else {
        LLE1DRequest req = build_lle1d_request(*this, c);
        if (req.krs_body.empty()) {
            c.last_error = "krs_code пуст (нет валидной системы или scheme)";
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
    c.last_run_ok = false;
    c.last_run_2d_ok = false;
    c.last_error.clear();

    if (c.mode_2d) {
        LLE2DRequest req = build_lle2d_request(*this, c);
        if (req.krs_body.empty()) {
            c.last_error = "krs_code пуст (нет валидной системы или scheme)";
            return false;
        }
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
            c.last_error = "krs_code пуст (нет валидной системы или scheme)";
            return false;
        }
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

bool LLEAnalysisSession::poll() {
    if (!in_flight) return false;
    if (is_2d_run) {
        if (!run_future_2d.valid()) { in_flight = false; running_curve_index = -1; return false; }
        if (run_future_2d.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
        LLE2DResult r = run_future_2d.get();
        int idx = running_curve_index;
        if (idx >= 0 && idx < (int)curves.size()) {
            apply_lle2d_result(curves[idx], std::move(r));
        }
    } else {
        if (!run_future.valid()) { in_flight = false; running_curve_index = -1; return false; }
        if (run_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
        LLE1DResult r = run_future.get();
        int idx = running_curve_index;
        if (idx >= 0 && idx < (int)curves.size()) {
            apply_lle1d_result(curves[idx], std::move(r));
        }
    }
    in_flight = false;
    running_curve_index = -1;
    return true;
}

// ============================================================================
// BasinsAnalysisSession — один config на сессию, без inner tab-bar.
// ============================================================================

void BasinsAnalysisSession::load_from_record(const SystemRecord& r,
    const std::vector<std::string>& vars_,
    const std::vector<std::string>& params_) {
    vars = vars_;
    params = params_;
    custom_schemes = r.custom_schemes;

    BasinsConfig c;
    c.h_text = r.step_h.empty() ? std::string("0.01") : r.step_h;

    for (const auto& p : params) {
        auto it = r.param_values.find(p);
        c.param_values[p] = (it != r.param_values.end()) ? it->second : "";
    }
    for (const auto& v : vars) {
        auto it = r.init_conditions.find(v);
        c.initial_conditions[v] = (it != r.init_conditions.end()) ? it->second : "";
    }
    c.axis_x_var = 0;
    c.axis_y_var = (vars.size() > 1) ? 1 : 0;
    c.writable_var = 0;
    config = std::move(c);
}

static BasinsRequest build_basins_request(const BasinsAnalysisSession& s,
                                          const BasinsConfig& c) {
    BasinsRequest req;
    req.krs_body  = compute_krs_for_scheme(s.custom_schemes, s.sys, c.scheme);
    req.amountOfX = (int)s.vars.size();

    req.initial_conditions.resize(req.amountOfX);
    for (int i = 0; i < req.amountOfX; ++i) {
        auto it = c.initial_conditions.find(s.vars[i]);
        req.initial_conditions[i] = (it != c.initial_conditions.end()) ? parse_d(it->second, 0.0) : 0.0;
    }

    int nparams = (int)s.params.size();
    req.base_values.assign((size_t)nparams + 1, 0.0);
    for (int i = 0; i < nparams; ++i) {
        auto it = c.param_values.find(s.params[i]);
        req.base_values[i + 1] = (it != c.param_values.end()) ? parse_d(it->second, 0.0) : 0.0;
    }

    req.axis_x_var      = (c.axis_x_var >= 0 && c.axis_x_var < req.amountOfX) ? c.axis_x_var : 0;
    req.axis_y_var      = (c.axis_y_var >= 0 && c.axis_y_var < req.amountOfX) ? c.axis_y_var : (req.amountOfX > 1 ? 1 : 0);
    req.axis_x_lo       = parse_d(c.axis_x_lo_text, -10.0);
    req.axis_x_hi       = parse_d(c.axis_x_hi_text,  10.0);
    req.axis_y_lo       = parse_d(c.axis_y_lo_text, -10.0);
    req.axis_y_hi       = parse_d(c.axis_y_hi_text,  10.0);
    req.n_pts           = parse_i(c.n_pts_text, 200);
    req.writable_var    = (c.writable_var >= 0 && c.writable_var < req.amountOfX) ? c.writable_var : 0;
    req.h               = parse_d(c.h_text, 0.01);
    req.t_max           = parse_d(c.t_max_text, 1000.0);
    req.transient_time  = parse_d(c.transient_text, 10000.0);
    req.pre_scaller     = std::max(1, parse_i(c.pre_scaller_text, 1));
    req.max_value       = parse_d(c.max_value_text, 1.0e6);
    req.eps_dbscan      = parse_d(c.eps_dbscan_text, 0.5);
    req.csv_output_path = c.csv_save_enabled ? c.csv_output_path : std::string{};
    return req;
}

static void apply_basins_result(BasinsConfig& c, BasinsResult&& r) {
    c.result = std::move(r);
    c.last_run_ok = c.result.ok;
    if (!c.result.ok) c.last_error = c.result.error;
    c.data_generation++;
    c.fit_request = true;
}

bool BasinsAnalysisSession::run(ParametricEngine& engine) {
    BasinsConfig& c = config;
    c.last_run_ok = false;
    c.last_error.clear();

    BasinsRequest req = build_basins_request(*this, c);
    if (req.krs_body.empty()) {
        c.last_error = "krs_code пуст (нет валидной системы или scheme)";
        return false;
    }
    BasinsResult r = engine.run_basins(req);
    bool ok = r.ok;
    apply_basins_result(c, std::move(r));
    return ok;
}

bool BasinsAnalysisSession::run_async(ParametricEngine& engine) {
    if (in_flight) return false;
    BasinsConfig& c = config;
    c.last_run_ok = false;
    c.last_error.clear();

    BasinsRequest req = build_basins_request(*this, c);
    if (req.krs_body.empty()) {
        c.last_error = "krs_code пуст (нет валидной системы или scheme)";
        return false;
    }

    in_flight = true;
    compute_start_time = std::chrono::steady_clock::now();
    run_future = std::async(std::launch::async, [&engine, req = std::move(req)]() {
        return engine.run_basins(req);
    });
    return true;
}

bool BasinsAnalysisSession::poll() {
    if (!in_flight) return false;
    if (!run_future.valid()) { in_flight = false; return false; }
    if (run_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;

    BasinsResult r = run_future.get();
    apply_basins_result(config, std::move(r));
    in_flight = false;
    return true;
}

// ============================================================================
// LyapunovSpectrumAnalysisSession — копия LLE-паттерна для LS.
// ============================================================================

void LyapunovSpectrumAnalysisSession::load_from_record(const SystemRecord& r,
    const std::vector<std::string>& vars_,
    const std::vector<std::string>& params_) {
    vars = vars_;
    params = params_;
    custom_schemes = r.custom_schemes;

    curves.clear();
    LSCurveConfig c;
    c.label = "LS 1";
    c.h_text = r.step_h.empty() ? std::string("0.01") : r.step_h;

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
        const std::string& pname = params[0];
        auto lo = r.param_min.find(pname);
        auto hi = r.param_max.find(pname);
        if (lo != r.param_min.end() && !lo->second.empty()) c.param_lo_text = lo->second;
        if (hi != r.param_max.end() && !hi->second.empty()) c.param_hi_text = hi->second;
    }
    curves.push_back(std::move(c));

    active_curve_index = 0;
    running_curve_index = -1;
}

void LyapunovSpectrumAnalysisSession::add_curve() {
    LSCurveConfig c;
    if (!curves.empty()) {
        c = curves.back();
        c.result = LS1DResult{};
        c.last_run_ok = false;
        c.last_error.clear();
        c.data_generation = 0;
        c.fit_request = false;
    } else {
        for (const auto& v : vars) c.initial_conditions[v] = "";
        for (const auto& p : params) c.param_values[p] = "";
    }
    c.label = "LS " + std::to_string(curves.size() + 1);
    curves.push_back(std::move(c));
    active_curve_index = (int)curves.size() - 1;
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

static LS1DRequest build_ls1d_request(const LyapunovSpectrumAnalysisSession& s,
                                      const LSCurveConfig& c) {
    LS1DRequest req;
    req.krs_body  = compute_krs_for_scheme(s.custom_schemes, s.sys, c.scheme);
    req.amountOfX = (int)s.vars.size();

    req.initial_conditions.resize(req.amountOfX);
    for (int i = 0; i < req.amountOfX; ++i) {
        auto it = c.initial_conditions.find(s.vars[i]);
        req.initial_conditions[i] = (it != c.initial_conditions.end()) ? parse_d(it->second, 0.0) : 0.0;
    }

    int nparams = (int)s.params.size();
    req.base_values.assign((size_t)nparams + 1, 0.0);
    for (int i = 0; i < nparams; ++i) {
        auto it = c.param_values.find(s.params[i]);
        req.base_values[i + 1] = (it != c.param_values.end()) ? parse_d(it->second, 0.0) : 0.0;
    }
    req.param_index = (c.param_index >= 0 && c.param_index < nparams) ? c.param_index + 1 : 1;
    req.sweep_over_var = c.sweep_over_var;
    req.var_sweep_index = (c.var_sweep_index >= 0 && c.var_sweep_index < req.amountOfX)
                          ? c.var_sweep_index : 0;

    req.param_lo       = parse_d(c.param_lo_text, 0.0);
    req.param_hi       = parse_d(c.param_hi_text, 1.0);
    req.n_pts          = parse_i(c.n_pts_text, 500);
    req.h              = parse_d(c.h_text, 0.01);
    req.t_max          = parse_d(c.t_max_text, 100.0);
    req.transient_time = parse_d(c.transient_text, 100.0);
    req.max_value      = parse_d(c.max_value_text, 1.0e6);
    req.NT             = parse_d(c.nt_text,  1.0);
    req.eps            = parse_d(c.eps_text, 1.0e-4);
    req.csv_output_path = c.csv_save_enabled ? c.csv_output_path : std::string{};
    return req;
}

static void apply_ls1d_result(LSCurveConfig& c, LS1DResult&& r) {
    c.result = std::move(r);
    c.last_run_ok = c.result.ok;
    if (!c.result.ok) c.last_error = c.result.error;
    c.data_generation++;
    c.fit_request = true;
}

// 2D-Request: те же общие поля + поля второй оси (param_index_2 / var_sweep_index_2).
static LS2DRequest build_ls2d_request(const LyapunovSpectrumAnalysisSession& s,
                                      const LSCurveConfig& c) {
    LS2DRequest req;
    req.krs_body  = compute_krs_for_scheme(s.custom_schemes, s.sys, c.scheme);
    req.amountOfX = (int)s.vars.size();

    req.initial_conditions.resize(req.amountOfX);
    for (int i = 0; i < req.amountOfX; ++i) {
        auto it = c.initial_conditions.find(s.vars[i]);
        req.initial_conditions[i] = (it != c.initial_conditions.end()) ? parse_d(it->second, 0.0) : 0.0;
    }

    int nparams = (int)s.params.size();
    req.base_values.assign((size_t)nparams + 1, 0.0);
    for (int i = 0; i < nparams; ++i) {
        auto it = c.param_values.find(s.params[i]);
        req.base_values[i + 1] = (it != c.param_values.end()) ? parse_d(it->second, 0.0) : 0.0;
    }

    req.sweep_over_var     = c.sweep_over_var;
    req.sweep_over_var_2   = c.sweep_over_var_2;
    req.param_index        = (c.param_index >= 0 && c.param_index < nparams) ? c.param_index + 1 : 1;
    req.param_index_2      = (c.param_index_2 >= 0 && c.param_index_2 < nparams) ? c.param_index_2 + 1 : 1;
    req.var_sweep_index    = (c.var_sweep_index >= 0 && c.var_sweep_index < req.amountOfX)
                             ? c.var_sweep_index : 0;
    req.var_sweep_index_2  = (c.var_sweep_index_2 >= 0 && c.var_sweep_index_2 < req.amountOfX)
                             ? c.var_sweep_index_2 : 0;

    req.param_lo           = parse_d(c.param_lo_text,   0.0);
    req.param_hi           = parse_d(c.param_hi_text,   1.0);
    req.param_lo_2         = parse_d(c.param_lo_2_text, 0.0);
    req.param_hi_2         = parse_d(c.param_hi_2_text, 1.0);
    req.n_pts              = parse_i(c.n_pts_text, 200);
    req.h                  = parse_d(c.h_text, 0.01);
    req.t_max              = parse_d(c.t_max_text, 100.0);
    req.transient_time     = parse_d(c.transient_text, 100.0);
    req.max_value          = parse_d(c.max_value_text, 1.0e6);
    req.NT                 = parse_d(c.nt_text,  1.0);
    req.eps                = parse_d(c.eps_text, 1.0e-4);
    req.csv_output_path    = c.csv_save_enabled ? c.csv_output_path : std::string{};
    return req;
}

static void apply_ls2d_result(LSCurveConfig& c, LS2DResult&& r) {
    c.result_2d = std::move(r);
    c.last_run_2d_ok = c.result_2d.ok;
    if (!c.result_2d.ok) c.last_error = c.result_2d.error;
    // Clamp выбранной экспоненты под новый n_exponents (если N изменился).
    if (c.result_2d.n_exponents > 0 &&
        c.display_exponent_idx >= c.result_2d.n_exponents) {
        c.display_exponent_idx = 0;
    }
    c.data_generation_2d++;
    c.fit_request_2d = true;
}

bool LyapunovSpectrumAnalysisSession::run(ParametricEngine& engine, int curve_idx) {
    if (curve_idx < 0 || curve_idx >= (int)curves.size()) return false;
    LSCurveConfig& c = curves[curve_idx];
    c.last_run_ok = false;
    c.last_run_2d_ok = false;
    c.last_error.clear();

    if (c.mode_2d) {
        LS2DRequest req = build_ls2d_request(*this, c);
        if (req.krs_body.empty()) {
            c.last_error = "krs_code пуст (нет валидной системы или scheme)";
            return false;
        }
        LS2DResult r = engine.run_ls_2d(req);
        bool ok = r.ok;
        apply_ls2d_result(c, std::move(r));
        return ok;
    } else {
        LS1DRequest req = build_ls1d_request(*this, c);
        if (req.krs_body.empty()) {
            c.last_error = "krs_code пуст (нет валидной системы или scheme)";
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
    c.last_run_ok = false;
    c.last_run_2d_ok = false;
    c.last_error.clear();

    if (c.mode_2d) {
        LS2DRequest req = build_ls2d_request(*this, c);
        if (req.krs_body.empty()) {
            c.last_error = "krs_code пуст (нет валидной системы или scheme)";
            return false;
        }
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
            c.last_error = "krs_code пуст (нет валидной системы или scheme)";
            return false;
        }
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

bool LyapunovSpectrumAnalysisSession::poll() {
    if (!in_flight) return false;
    if (is_2d_run) {
        if (!run_future_2d.valid()) { in_flight = false; running_curve_index = -1; return false; }
        if (run_future_2d.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
        LS2DResult r = run_future_2d.get();
        int idx = running_curve_index;
        if (idx >= 0 && idx < (int)curves.size()) {
            apply_ls2d_result(curves[idx], std::move(r));
        }
    } else {
        if (!run_future.valid()) { in_flight = false; running_curve_index = -1; return false; }
        if (run_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
        LS1DResult r = run_future.get();
        int idx = running_curve_index;
        if (idx >= 0 && idx < (int)curves.size()) {
            apply_ls1d_result(curves[idx], std::move(r));
        }
    }
    in_flight = false;
    running_curve_index = -1;
    return true;
}