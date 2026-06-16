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

void PhaseAnalysisSession::recompute() {
    auto _t0 = std::chrono::high_resolution_clock::now();
    result = AnalysisResult{};
    int dim = (int)vars.size();
    if (dim < 1) { result.error = "no variables"; return; }
    double h = parse_val(step_h, 0.01);
    if (h <= 0) { result.error = "step h must be > 0"; return; }
    double tsim = parse_val(sim_time, 50.0);
    double tskip = parse_val(skip_time, 0.0);
    int total = (int)(tsim / h); if (total <= 0) total = 1;
    int skip = (int)(tskip / h); if (skip < 0) skip = 0;

    // параметры: a[0] зарезервирован, a[1..] = params
    int nparams = (int)params.size();
    std::vector<double> a(nparams + 1, 0.0);
    for (int j = 0; j < nparams; ++j)
        a[1 + j] = parse_val(param_values[params[j]], 0.0);

    // децимация: выводить каждую dec-ю точку
    int dec = std::atoi(decimation.c_str()); if (dec < 1) dec = 1;

    int N = (int)ic_sets.size();
    if (N < 1) { result.error = "no initial conditions"; return; }

    // НУ всех наборов в плоский массив ic_flat[k*dim + i] + тексты для легенды
    std::vector<double> ic_flat((size_t)N * dim, 0.0);
    for (int k = 0; k < N; ++k) {
        const auto& ic = ic_sets[k];
        for (int i = 0; i < dim; ++i)
            ic_flat[(size_t)k * dim + i] =
            parse_val(ic.values.count(vars[i]) ? ic.values.at(vars[i]) : "", 0.0);
    }

    // сюда соберём сырые траектории (до децимации): [N][total][dim]
    std::vector<std::vector<std::vector<double>>> raw;
    bool calc_ok = true;
    std::string calc_err;

    if (use_gpu) {
        // --- GPU: все N траекторий за один запуск (поток на НУ) ---
        if (krs_code.empty()) regenerate_krs();
        if (krs_code.empty()) {
            result.error = "no KRS generated (check system)";
            return;
        }
        calc_ok = computePhasePortraitsNVRTC(krs_code, dim,
            ic_flat.data(), N, a.data(), (int)a.size(),
            h, total, skip, raw, &calc_err);
        if (!calc_ok) {
            result.error = "GPU: " + calc_err;
            return;
        }
    }
    else {
        // --- CPU: интерпретатор, по траектории на НУ ---
        std::unique_ptr<SystemEvaluator> evaluator;
        try {
            evaluator.reset(new SystemEvaluator(sys));
        }
        catch (const std::exception& e) {
            result.error = std::string("parse error: ") + e.what();
            return;
        }
        raw.resize(N);
        for (int k = 0; k < N; ++k) {
            bool ok = computePhasePortraitCPU(*evaluator, int_scheme_from_string(scheme),
                &ic_flat[(size_t)k * dim], dim, a.data(), (int)a.size(),
                h, total, skip, raw[k]);
            if (!ok) { calc_ok = false; calc_err = "trajectory '" + ic_sets[k].label + "' diverged (nan/inf)"; }
        }
    }

    // --- общая часть: децимация + заполнение результата ---
    for (int k = 0; k < N; ++k) {
        const auto& ic = ic_sets[k];
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

        // текстовое НУ для легенды: "(x0,y0,z0)"
        std::string txt = "(";
        for (int i = 0; i < dim; ++i) {
            std::string val = ic.values.count(vars[i]) ? ic.values.at(vars[i]) : "";
            if (val.empty()) val = "0";
            txt += val; if (i + 1 < dim) txt += ",";
        }
        txt += ")";
        result.ic_text.push_back(txt);
    }

    result.ok = result.trajectories.size() > 0;
    if (!calc_ok && result.ok) {
        // расчёт прошёл, но какая-то траектория разошлась — пометим
        result.error = calc_err;
    }
    fit_request = true; // запросить автоскейл осей

    auto _t1 = std::chrono::high_resolution_clock::now();
    double _ms = std::chrono::duration<double, std::milli>(_t1 - _t0).count();
    // временный замер времени (можно убрать); не затираем реальную ошибку
    if (result.error.empty())
        result.error = "recompute: " + std::to_string(_ms) + " ms";

    data_generation++;
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

bool ParametricAnalysisSession::run(ParametricEngine& engine) {
    last_run_ok = false;
    last_error.clear();

    if (krs_code.empty()) regenerate_krs();
    if (krs_code.empty()) {
        last_error = "krs_code пуст (нет валидной системы)";
        return false;
    }

    Bifurcation1DRequest req;
    req.krs_body = krs_code;
    req.amountOfX = (int)vars.size();

    req.initial_conditions.resize(req.amountOfX);
    for (int i = 0; i < req.amountOfX; ++i) {
        auto it = initial_conditions.find(vars[i]);
        req.initial_conditions[i] = (it != initial_conditions.end()) ? parse_d(it->second, 0.0) : 0.0;
    }

    // Конвенция codegen: a[0] зарезервирован, реальные параметры — с a[1].
    int nparams = (int)params.size();
    req.base_values.assign((size_t)nparams + 1, 0.0);
    for (int i = 0; i < nparams; ++i) {
        auto it = param_values.find(params[i]);
        req.base_values[i + 1] = (it != param_values.end()) ? parse_d(it->second, 0.0) : 0.0;
    }
    req.param_index = (param_index >= 0 && param_index < nparams) ? param_index + 1 : 1;

    req.param_lo       = parse_d(param_lo_text, 0.0);
    req.param_hi       = parse_d(param_hi_text, 1.0);
    req.n_pts          = parse_i(n_pts_text, 500);
    req.writable_var   = (writable_var >= 0 && writable_var < req.amountOfX) ? writable_var : 0;
    req.h              = parse_d(h_text, 0.01);
    req.t_max          = parse_d(t_max_text, 100.0);
    req.transient_time = parse_d(transient_text, 100.0);
    req.pre_scaller    = std::max(1, parse_i(pre_scaller_text, 1));
    req.max_value      = parse_d(max_value_text, 1.0e6);

    result = engine.run_bifurcation_1d(req);
    last_run_ok = result.ok;
    if (!result.ok) last_error = result.error;
    data_generation++;
    fit_request = true;
    return result.ok;
}