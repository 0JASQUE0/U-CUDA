#include "ucuda/domain/analysis_session.h"
#include <algorithm>
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
    bd.label_is_manual = false;   // fresh diagram → auto-label
    diagrams.push_back(std::move(bd));
    active_diagram_index = (int)diagrams.size() - 1;
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
    c.label_is_manual = false;   // fresh curve → auto-label
    curves.push_back(std::move(c));
    active_curve_index = (int)curves.size() - 1;
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
    c.label_is_manual = false;   // fresh curve → auto-label
    curves.push_back(std::move(c));
    active_curve_index = (int)curves.size() - 1;
}

void BasinsAnalysisSession::ensure_phase_slots() {
    while (phase_slots.size() < configs.size()) {
        auto slot = std::unique_ptr<BasinsPhaseSlot>(new BasinsPhaseSlot());
        slot->phase.vars           = vars;
        slot->phase.params         = params;
        slot->phase.sys            = sys;
        slot->phase.custom_schemes = custom_schemes;
        // Одно окно Phase 2D по умолчанию — как у PhaseAnalysisSession после
        // load_from_record. Time domain пользователь добавляет кнопкой.
        slot->phase.add_projection();
        phase_slots.push_back(std::move(slot));
    }
    if (phase_slots.size() > configs.size())
        phase_slots.resize(configs.size());
}
const char* basin_feature_name(int feature_code) {
    if (feature_code < 0 || feature_code >= BF_FEATURE_COUNT) return "unknown";
    return kBasinFeatureNames[feature_code];
}

void PhaseAnalysisSession::add_projection() {
    Projection p;
    int n = (int)projections.size();
    // A map's iterates are not joined by a continuous path, so markers, not a
    // polyline. sys is set right after load_from_record by start_phase_analysis.
    p.draw_points = sys.is_map;
    p.label = "Projection " + std::to_string(n + 1);
    p.axis_x = 0;
    p.axis_y = (vars.size() > 1) ? 1 : 0;
    p.axis_z = (vars.size() > 2) ? 2 : 0;
    // для time domain: по умолчанию показываем все переменные
    p.show_var.assign(vars.size(), true);
    projections.push_back(std::move(p));
}
