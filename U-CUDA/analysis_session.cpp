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

    // DIAG: соберём строку "a[1]=eq=1.0, a[2]=m=1.5, ..." для отладки.
    // Покажется в Phase под кнопкой Recompute как обычный error_text.
    std::string diag = "a[]: ";
    for (int j = 0; j < nparams; ++j) {
        diag += "a[" + std::to_string(j + 1) + "]=" + in.params[j] + "=" + std::to_string(a[1 + j]);
        if (j + 1 < nparams) diag += ", ";
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
        result.error = "recompute: " + std::to_string(_ms) + " ms | " + diag;

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

// --- ParametricAnalysisSession ---

void ParametricAnalysisSession::regenerate_krs() {
    krs_code.clear();
    for (const auto& cs : custom_schemes) {
        if (cs.name == scheme) { krs_code = cs.body; return; }
    }
    if (sys.rhs.empty()) return;
    try {
        krs_code = codegen_scheme(sys, scheme_from_string(scheme));
    }
    catch (...) {
        krs_code.clear();
    }
}

void ParametricAnalysisSession::load_from_record(const SystemRecord& r,
    const std::vector<std::string>& vars_,
    const std::vector<std::string>& params_) {
    vars = vars_;
    params = params_;
    custom_schemes = r.custom_schemes;
    h_text = r.step_h.empty() ? std::string("0.01") : r.step_h;

    param_values.clear();
    for (const auto& p : params) {
        auto it = r.param_values.find(p);
        param_values[p] = (it != r.param_values.end()) ? it->second : "";
    }

    initial_conditions.clear();
    for (const auto& v : vars) {
        auto it = r.init_conditions.find(v);
        initial_conditions[v] = (it != r.init_conditions.end()) ? it->second : "";
    }

    if (!params.empty()) {
        if (param_index < 0 || param_index >= (int)params.size()) param_index = 0;
        const std::string& pname = params[param_index];
        auto lo = r.param_min.find(pname);
        auto hi = r.param_max.find(pname);
        if (lo != r.param_min.end() && !lo->second.empty()) param_lo_text = lo->second;
        if (hi != r.param_max.end() && !hi->second.empty()) param_hi_text = hi->second;
    }

    result = Bifurcation1DResult{};
    last_run_ok = false;
    last_error.clear();
    data_generation = 0;
}

static double parse_d(const std::string& s, double def) {
    if (s.empty()) return def;
    try { return std::stod(s); } catch (...) { return def; }
}
static int parse_i(const std::string& s, int def) {
    if (s.empty()) return def;
    try { return std::stoi(s); } catch (...) { return def; }
}

// Снапшот текущих GUI-полей сессии в Bifurcation1DRequest. Делается на главном
// потоке перед стартом async-расчёта, чтобы worker работал со стабильной копией
// и пользователь мог в это время менять поля без race condition.
static Bifurcation1DRequest build_bif1d_request(const ParametricAnalysisSession& s) {
    Bifurcation1DRequest req;
    req.krs_body  = s.krs_code;
    req.amountOfX = (int)s.vars.size();

    req.initial_conditions.resize(req.amountOfX);
    for (int i = 0; i < req.amountOfX; ++i) {
        auto it = s.initial_conditions.find(s.vars[i]);
        req.initial_conditions[i] = (it != s.initial_conditions.end()) ? parse_d(it->second, 0.0) : 0.0;
    }

    // Конвенция codegen: a[0] зарезервирован, реальные параметры — с a[1].
    int nparams = (int)s.params.size();
    req.base_values.assign((size_t)nparams + 1, 0.0);
    for (int i = 0; i < nparams; ++i) {
        auto it = s.param_values.find(s.params[i]);
        req.base_values[i + 1] = (it != s.param_values.end()) ? parse_d(it->second, 0.0) : 0.0;
    }
    req.param_index = (s.param_index >= 0 && s.param_index < nparams) ? s.param_index + 1 : 1;

    req.param_lo       = parse_d(s.param_lo_text, 0.0);
    req.param_hi       = parse_d(s.param_hi_text, 1.0);
    req.n_pts          = parse_i(s.n_pts_text, 500);
    req.writable_var   = (s.writable_var >= 0 && s.writable_var < req.amountOfX) ? s.writable_var : 0;
    req.h              = parse_d(s.h_text, 0.01);
    req.t_max          = parse_d(s.t_max_text, 100.0);
    req.transient_time = parse_d(s.transient_text, 100.0);
    req.pre_scaller    = std::max(1, parse_i(s.pre_scaller_text, 1));
    req.max_value      = parse_d(s.max_value_text, 1.0e6);
    req.csv_output_path = s.csv_save_enabled ? s.csv_output_path : std::string{};
    return req;
}

// Применить только что готовый Bifurcation1DResult к session (мутируем main-thread
// поля: result, last_*, data_generation, fit_request). Общий код для sync и async.
static void apply_bif1d_result(ParametricAnalysisSession& s, Bifurcation1DResult&& r) {
    s.result = std::move(r);
    s.last_run_ok = s.result.ok;
    if (!s.result.ok) s.last_error = s.result.error;
    s.data_generation++;
    s.fit_request = true;
}

bool ParametricAnalysisSession::run(ParametricEngine& engine) {
    last_run_ok = false;
    last_error.clear();

    if (krs_code.empty()) regenerate_krs();
    if (krs_code.empty()) {
        last_error = "krs_code пуст (нет валидной системы)";
        return false;
    }

    Bifurcation1DRequest req = build_bif1d_request(*this);
    Bifurcation1DResult r = engine.run_bifurcation_1d(req);
    bool ok = r.ok;
    apply_bif1d_result(*this, std::move(r));
    return ok;
}

bool ParametricAnalysisSession::run_async(ParametricEngine& engine) {
    if (in_flight) return false;

    last_run_ok = false;
    last_error.clear();

    if (krs_code.empty()) regenerate_krs();
    if (krs_code.empty()) {
        last_error = "krs_code пуст (нет валидной системы)";
        return false;
    }

    Bifurcation1DRequest req = build_bif1d_request(*this);

    in_flight = true;
    compute_start_time = std::chrono::steady_clock::now();

    // Worker НЕ трогает session напрямую — только engine и собственную копию req.
    // Результат вернётся через future и будет применён в poll() на главном потоке.
    run_future = std::async(std::launch::async, [&engine, req = std::move(req)]() {
        return engine.run_bifurcation_1d(req);
    });

    return true;
}

bool ParametricAnalysisSession::poll() {
    if (!in_flight) return false;
    if (!run_future.valid()) { in_flight = false; return false; }
    if (run_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;

    Bifurcation1DResult r = run_future.get();
    apply_bif1d_result(*this, std::move(r));
    in_flight = false;
    return true;
}