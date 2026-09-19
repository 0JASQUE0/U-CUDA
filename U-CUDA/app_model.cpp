#include "app_model.h"
#include "codegen.hpp"
#include "num_parse.h"   // fmt_num_shortest — round-trip формат для пинов узлов
#include "session_io.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "sysparse.hpp"
#include <stdexcept>

// Newton knobs are UI strings; parse them here, once, on the way into System.
// Out-of-range or unparseable input falls back to the defaults rather than
// producing a scheme body that cannot converge.
static void apply_newton_settings(System& s, bool full, const std::string& tol,
                                  const std::string& iters) {
    s.newton_full = full;
    try { double v = std::stod(tol);      if (v > 0.0) s.newton_tol = v; }       catch (...) {}
    try { int    v = std::stoi(iters);    if (v > 0)   s.newton_max_iters = v; } catch (...) {}
}

// Строит System из текущего режима ввода и алфавита.
System AppModel::build_system() const {
    std::vector<std::string> alpha = parse_alphabet();

    // вспомогательные функции (опция)
    FuncDefs funcs;
    if (use_aux_funcs && !func_defs_text.empty())
        funcs = parse_func_defs(func_defs_text);

    if (mode == InputMode::Image || mode == InputMode::Latex) {
        // и после OCR, и при ручном вводе LaTeX — многострочный LaTeX-разбор
        if (latex_text.empty()) throw std::runtime_error("LaTeX is empty");
        if (alpha.empty()) throw std::runtime_error("alphabet is empty");
        System s = parse_system_from_latex(latex_text, alpha, param_order, funcs, is_map);
        apply_newton_settings(s, newton_full, newton_tol, newton_max_iters);
        return s;
    }

    // InputMode::Plain — обычный синтаксис. Пока разбираем его тем же многострочным путём, что и
    // LaTeX: parse_system_from_latex ждёт LaTeX, и для Plain нужен отдельный парсер. На практике
    // часто срабатывает (x*(r-z) и т.п.).
    if (plain_text.empty()) throw std::runtime_error("equations are empty");
    if (alpha.empty()) throw std::runtime_error("alphabet is empty");
    System s = parse_system_from_latex(plain_text, alpha, param_order, funcs, is_map);
    apply_newton_settings(s, newton_full, newton_tol, newton_max_iters);
    return s;
}

// Парсит систему, обновляет списки символов и синхронизирует словари значений.
bool AppModel::refresh_symbols() {
    error_message.clear();

    auto sync = [](std::map<std::string, std::string>& m,
        const std::vector<std::string>& keys) {
            std::map<std::string, std::string> next;
            for (const auto& k : keys) {
                auto it = m.find(k);
                next[k] = (it != m.end()) ? it->second : std::string();
            }
            m.swap(next);
        };

    auto apply_sync = [&]() {
        sync(init_conditions, known_vars);
        sync(param_values, known_params);
    };

    // Helper: распарсить comma/space-separated список.
    auto parse_csv = [](const std::string& s) {
        std::vector<std::string> out;
        std::string cur;
        for (char c : s) {
            if (c == ',' || c == ' ' || c == '\t' || c == '\n' || c == ';') {
                if (!cur.empty()) { out.push_back(cur); cur.clear(); }
            }
            else cur += c;
        }
        if (!cur.empty()) out.push_back(cur);
        return out;
    };

    // Если заданы явные vars_text И params_text — используем их напрямую.
    // Это новый удобный формат: переменные и параметры разделены явно.
    if (!vars_text.empty() && !params_text.empty()) {
        known_vars = parse_csv(vars_text);
        known_params = parse_csv(params_text);
        apply_sync();
        return true;
    }

    // Legacy fallback: только vars_text, params = alphabet \ vars.
    if (!vars_text.empty()) {
        known_vars = parse_csv(vars_text);
        auto all = parse_csv(alphabet_text);
        std::vector<std::string> params_only;
        for (const auto& nm : all) {
            bool is_var = false;
            for (const auto& v : known_vars) if (v == nm) { is_var = true; break; }
            if (!is_var) params_only.push_back(nm);
        }
        known_params = std::move(params_only);
        apply_sync();
        return true;
    }

    // Обычный путь: парсер уравнений.
    try {
        System sys = build_system();
        known_vars = sys.vars;
        known_params = sys.params;
        apply_sync();
        return true;
    }
    catch (const std::exception& e) {
        error_message = e.what();
        return false;
    }
}

SystemRecord AppModel::to_record() const {
    SystemRecord r;
    r.name = name;
    r.note = note;
    r.is_map = is_map;
    switch (mode) {
    case InputMode::Image: r.mode = "Image"; break;
    case InputMode::Latex: r.mode = "LaTeX"; break;
    case InputMode::Plain: r.mode = "Plain"; break;
    }
    r.latex_text = latex_text;
    r.plain_text = plain_text;
    r.alphabet_text = alphabet_text;
    r.vars_text = vars_text;
    r.params_text = params_text;
    r.use_aux_funcs = use_aux_funcs;
    r.func_defs_text = func_defs_text;
    r.param_order = (param_order == ParamOrder::AsInSystem) ? "AsInSystem" : "AsInAlphabet";
    r.scheme_euler = scheme_euler;
    r.scheme_cromer = scheme_cromer;
    r.scheme_midpoint = scheme_midpoint;
    r.scheme_rk4 = scheme_rk4;
    r.scheme_dopri78 = scheme_dopri78;
    r.scheme_cd = scheme_cd;
    r.scheme_ccd = scheme_ccd;
    r.scheme_ccd4 = scheme_ccd4;
    r.scheme_ieuler = scheme_ieuler;
    r.scheme_imidpoint = scheme_imidpoint;
    r.scheme_semp = scheme_semp;
    r.scheme_simp = scheme_simp;
    r.scheme_dmethod = scheme_dmethod;
    r.scheme_cieuler = scheme_cieuler;
    r.newton_full = newton_full;
    r.newton_tol = newton_tol;
    r.newton_max_iters = newton_max_iters;
    r.symmetry_s = symmetry_s;
    r.step_h = step_h;
    r.init_conditions = init_conditions;
    r.param_values = param_values;
    r.custom_schemes = custom_schemes;
    r.wrapper_schemes = wrapper_schemes;
    return r;
}

void AppModel::from_record(const SystemRecord& r) {
    name = r.name;
    note = r.note;
    is_map = r.is_map;
    if (r.mode == "Image") mode = InputMode::Image;
    else if (r.mode == "Plain") mode = InputMode::Plain;
    else mode = InputMode::Latex;
    // Same normaliser as fresh OCR output; idempotent. Skipped for maps: it
    // only rewrites derivative forms to \dot{X}, which map notation never has.
    latex_text = (r.latex_text.empty() || r.is_map) ? r.latex_text
                                                    : format_latex(r.latex_text);
    plain_text = r.plain_text;
    alphabet_text = r.alphabet_text;
    vars_text = r.vars_text;
    params_text = r.params_text;
    use_aux_funcs = r.use_aux_funcs;
    func_defs_text = r.func_defs_text;
    param_order = (r.param_order == "AsInSystem") ? ParamOrder::AsInSystem : ParamOrder::AsInAlphabet;
    scheme_euler = r.scheme_euler;
    scheme_cromer = r.scheme_cromer;
    scheme_midpoint = r.scheme_midpoint;
    scheme_rk4 = r.scheme_rk4;
    scheme_dopri78 = r.scheme_dopri78;
    scheme_cd = r.scheme_cd;
    scheme_ccd = r.scheme_ccd;
    scheme_ccd4 = r.scheme_ccd4;
    scheme_ieuler = r.scheme_ieuler;
    scheme_imidpoint = r.scheme_imidpoint;
    scheme_semp = r.scheme_semp;
    scheme_simp = r.scheme_simp;
    scheme_dmethod = r.scheme_dmethod;
    scheme_cieuler = r.scheme_cieuler;
    newton_full = r.newton_full;
    newton_tol = r.newton_tol;
    newton_max_iters = r.newton_max_iters;
    symmetry_s = r.symmetry_s;
    step_h = r.step_h;
    init_conditions = r.init_conditions;
    param_values = r.param_values;
    custom_schemes = r.custom_schemes;
    wrapper_schemes = r.wrapper_schemes;
    loaded_name = r.name;          // запоминаем имя на диске
    // обновим списки символов (без падения, если система ещё неполна)
    refresh_symbols();
    // сразу перегенерируем код загруженной системы (если выбраны методы),
    // чтобы не показывать код от предыдущей системы.
    generated_code.clear();
    // A map has no scheme_* flag set — its single body is always generated.
    if (is_map
        || scheme_euler || scheme_cromer || scheme_midpoint || scheme_rk4 || scheme_dopri78
        || scheme_cd || scheme_ccd || scheme_ccd4 || scheme_ieuler || scheme_imidpoint
        || scheme_semp || scheme_simp || scheme_dmethod || scheme_cieuler)
        generate();
}

void AppModel::clear() {
    name.clear();
    note.clear();
    loaded_name.clear();
    latex_text.clear();
    plain_text.clear();
    func_defs_text.clear();
    use_aux_funcs = false;
    // алфавит и режим — оставляем дефолтными? обнулим алфавит тоже.
    alphabet_text.clear();
    vars_text.clear();
    params_text.clear();
    param_order = ParamOrder::AsInAlphabet;
    mode = InputMode::Image;
    is_map = false;
    scheme_euler = scheme_cromer = scheme_midpoint = scheme_rk4 = scheme_dopri78
        = scheme_cd = scheme_ccd = scheme_ccd4 = scheme_ieuler = scheme_imidpoint
        = scheme_semp = scheme_simp = scheme_dmethod = scheme_cieuler = false;
    symmetry_s = "0.5";
    newton_full = false;
    newton_tol = "1e-10";
    newton_max_iters = "8";
    step_h.clear();
    init_conditions.clear();
    param_values.clear();
    custom_schemes.clear();
    wrapper_schemes.clear();
    known_vars.clear();
    known_params.clear();
    generated_code.clear();
    error_message.clear();
}

bool AppModel::start_phase_analysis() {
    // распарсить систему -> known_vars/known_params
    if (!refresh_symbols()) return false;
    SystemRecord r = to_record();
    phase_session.load_from_record(r, known_vars, known_params);
    // передать уравнения в сессию и сгенерировать КРС для NVRTC
    try {
        phase_session.sys = build_system();
        // метод по умолчанию — из текущего выбора схемы в модели, если задан
        phase_session.regenerate_krs();
    }
    catch (...) {
        // если система ещё неполна — КРС останется пустой, recompute покажет ошибку
    }
    phase_session.loaded_system_name = name;
    return true;
}

bool AppModel::start_parametric_analysis() {
    if (!refresh_symbols()) return false;
    SystemRecord r = to_record();
    bifurcation_session.load_from_record(r, known_vars, known_params);
    lle_session.load_from_record(r, known_vars, known_params);
    ls_session.load_from_record(r, known_vars, known_params);
    try {
        // sys общий — всем трём сессиям нужен для compute_krs_for_scheme.
        // (KRS резолвится per-«прогон» в момент Run.)
        System built = build_system();
        bifurcation_session.sys = built;
        lle_session.sys         = built;
        ls_session.sys          = built;
    }
    catch (...) {
        // система неполна — Run покажет ошибку
    }
    bifurcation_session.loaded_system_name = name;
    lle_session.loaded_system_name         = name;
    ls_session.loaded_system_name          = name;
    return true;
}

bool AppModel::start_dft1d_analysis() {
    if (!refresh_symbols()) return false;
    SystemRecord r = to_record();
    dft1d_session.load_from_record(r, known_vars, known_params);
    try {
        System built = build_system();
        dft1d_session.sys = built;
    }
    catch (...) {}
    dft1d_session.loaded_system_name = name;
    return true;
}

bool AppModel::start_basins_analysis() {
    if (!refresh_symbols()) return false;
    SystemRecord r = to_record();
    basins_session.load_from_record(r, known_vars, known_params);
    try {
        System built = build_system();
        basins_session.sys = built;
    }
    catch (...) {}
    basins_session.loaded_system_name = name;
    return true;
}

bool AppModel::start_order_analysis() {
    if (!refresh_symbols()) return false;
    SystemRecord r = to_record();
    order_session.load_from_record(r, known_vars, known_params);
    try {
        System built = build_system();
        order_session.sys = built;
    }
    catch (...) {}
    order_session.loaded_system_name = name;
    return true;
}

bool AppModel::start_network_analysis() {
    if (!refresh_symbols()) return false;
    SystemRecord r = to_record();
    network_session.load_from_record(r, known_vars, known_params);
    try {
        System built = build_system();
        network_session.sys = built;
    }
    catch (...) {}
    network_session.loaded_system_name = name;
    return true;
}

bool AppModel::start_fastsync_analysis() {
    if (!refresh_symbols()) return false;
    SystemRecord r = to_record();
    fastsync_session.load_from_record(r, known_vars, known_params);
    try {
        System built = build_system();
        fastsync_session.sys = built;
    }
    catch (...) {}
    fastsync_session.loaded_system_name = name;
    return true;
}

bool AppModel::start_custom_analysis() {
    if (!refresh_symbols()) return false;
    SystemRecord r = to_record();
    custom_session.load_from_record(r, known_vars, known_params);
    try {
        System built = build_system();
        custom_session.sys = built;
        // Each owned sub-session also needs the built System for KRS codegen.
        custom_session.bif_session.sys    = built;
        custom_session.lle_session.sys    = built;
        custom_session.ls_session.sys     = built;
        custom_session.phase_session.sys  = built;
        custom_session.basins_session.sys = built;
        custom_session.phase_session.regenerate_krs();
    }
    catch (...) {
        // system incomplete — next Run will surface the error.
    }
    custom_session.loaded_system_name        = name;
    custom_session.bif_session.loaded_system_name    = name;
    custom_session.lle_session.loaded_system_name    = name;
    custom_session.ls_session.loaded_system_name     = name;
    custom_session.phase_session.loaded_system_name  = name;
    custom_session.basins_session.loaded_system_name = name;
    return true;
}

namespace {

// Цель свипа одной оси. Пишется и читается ЦЕЛИКОМ: половина применённой цели
// (скажем, par_index без снятого over_h) — это не цель, а мусор. Поэтому по
// общему строковому интерфейсу рассылки она едет одной строкой, см.
// encode_sweep / decode_sweep.
struct SweepTargetSlot {
    int*  par_index = nullptr;
    bool* over_var  = nullptr;
    int*  var_index = nullptr;
    bool* over_h    = nullptr;
    // Списки имён ЭТОЙ вкладки — для проверки границ при разборе: у сессии,
    // собранной до правки алфавита, индексов может не хватать.
    const std::vector<std::string>* params = nullptr;
    const std::vector<std::string>* vars   = nullptr;
    [[nodiscard]] bool valid() const { return par_index && over_var && var_index && over_h; }
};

// Куда broadcast_field может писать у одного конфига. nullptr = поля у этого
// конфига нет.
struct BroadcastSlots {
    std::string* h           = nullptr;
    std::string* symmetry    = nullptr;
    std::string* t_max       = nullptr;
    std::string* transient   = nullptr;
    std::string* pre_scaller = nullptr;
    std::string* max_value   = nullptr;
    std::map<std::string, std::string>* params = nullptr;
    std::map<std::string, std::string>* ics    = nullptr;
    std::string* scheme     = nullptr;
    std::string* sweep_lo   = nullptr;
    std::string* sweep_hi   = nullptr;
    std::string* sweep_lo_2 = nullptr;
    std::string* sweep_hi_2 = nullptr;
    std::string* resolution = nullptr;
    SweepTargetSlot sweep;     // первая ось свипа
    SweepTargetSlot sweep_2;   // вторая (2D-режим)
    // Подпись конфига (nullptr — у вкладки её нет: Phase и Shared config
    // Custom'а одни на вкладку). Отмена сверяет её, чтобы не записать в чужой
    // конфиг, вставший на освободившийся индекс.
    const std::string* label = nullptr;
};

// Цель свипа <-> строка. "par:2" — параметр с индексом 2, "var:0" — начальное
// условие, "h" — шаг интегрирования, "s" — коэффициент симметрии a[0]
// (par_index < 0). Лог-шкала сюда НЕ входит: это отдельный чекбокс рядом с
// комбо, и ПКМ по комбо про него ничего не обещает.
std::string encode_sweep(const SweepTargetSlot& s) {
    if (!s.valid()) return {};
    return encode_sweep_target(*s.par_index, *s.over_var, *s.var_index, *s.over_h);
}

// false = строка не разобралась или индекс за пределами списков ЭТОЙ вкладки;
// тогда поля не трогаются вовсе.
bool decode_sweep(const SweepTargetSlot& s, const std::string& v) {
    if (!s.valid() || v.empty()) return false;
    auto set = [&](int par, bool ov, int vi, bool oh) {
        *s.par_index = par; *s.over_var = ov; *s.var_index = vi; *s.over_h = oh;
        return true;
    };
    if (v == "h") return set(*s.par_index, false, *s.var_index, true);
    if (v == "s") return set(-1, false, *s.var_index, false);
    const bool is_var = (v.compare(0, 4, "var:") == 0);
    const bool is_par = (v.compare(0, 4, "par:") == 0);
    if (!is_var && !is_par) return false;
    int idx = 0;
    try { idx = std::stoi(v.substr(4)); } catch (...) { return false; }
    if (idx < 0) return false;
    const std::vector<std::string>* list = is_var ? s.vars : s.params;
    if (!list || idx >= (int)list->size()) return false;   // алфавит вкладки короче
    return is_var ? set(*s.par_index, true, idx, false)
                  : set(idx, false, *s.var_index, false);
}

constexpr BroadcastTab kBroadcastTabs[] = {
    BroadcastTab::Phase,  BroadcastTab::Bifurcation, BroadcastTab::LLE,
    BroadcastTab::LS,     BroadcastTab::Dft1D,       BroadcastTab::Basins,
    BroadcastTab::FastSync, BroadcastTab::Custom,    BroadcastTab::Order,
};

// Элемент вектора конфигов или nullptr (пустая сессия, индекс из битого
// сохранения или из записи отмены, пережившей закрытие вкладки).
template <class T>
T* config_at(std::vector<T>& v, int idx) {
    return (idx >= 0 && idx < (int)v.size()) ? &v[(size_t)idx] : nullptr;
}

// Слоты конфига вкладки. idx < 0 — взять активный конфиг (обычный путь
// broadcast'а); idx >= 0 — именно этот (путь отмены). В resolved_idx уходит
// фактический индекс, чтобы запись отмены знала, куда возвращаться.
// false = конфига нет.
bool slots_for(AppModel& m, BroadcastTab tab, int idx,
               BroadcastSlots& out, int& resolved_idx) {
    out = BroadcastSlots{};
    switch (tab) {
    case BroadcastTab::Phase: {
        // Имена полей у Phase исторически свои (step_h / sim_time / skip_time),
        // а НУ лежат списком наборов: несколько траекторий в одних осях — весь
        // смысл вкладки, поэтому трогаем только ПЕРВЫЙ набор, тот, которому
        // соответствует единственное НУ остальных вкладок.
        resolved_idx = 0;
        out.h         = &m.phase_session.step_h;
        out.symmetry  = &m.phase_session.symmetry_s;
        out.t_max     = &m.phase_session.sim_time;
        out.transient = &m.phase_session.skip_time;
        out.params    = &m.phase_session.param_values;
        out.scheme    = &m.phase_session.scheme;
        if (!m.phase_session.ic_sets.empty())
            out.ics = &m.phase_session.ic_sets[0].values;
        return true;
    }
    case BroadcastTab::Bifurcation: {
        auto& v = m.bifurcation_session.diagrams;
        resolved_idx = (idx < 0) ? m.bifurcation_session.active_diagram_index : idx;
        auto* c = config_at(v, resolved_idx);
        if (!c) return false;
        out.h = &c->h_text; out.symmetry = &c->symmetry_s;
        out.t_max = &c->t_max_text; out.transient = &c->transient_text;
        out.pre_scaller = &c->pre_scaller_text; out.max_value = &c->max_value_text;
        out.params = &c->param_values; out.ics = &c->initial_conditions;
        out.scheme = &c->scheme;
        out.sweep_lo = &c->param_lo_text;     out.sweep_hi = &c->param_hi_text;
        out.sweep_lo_2 = &c->param_lo_2_text; out.sweep_hi_2 = &c->param_hi_2_text;
        out.resolution = &c->n_pts_text;
        out.sweep   = { &c->param_index,   &c->sweep_over_var,   &c->var_sweep_index,
                        &c->sweep_over_h,   &m.bifurcation_session.params, &m.bifurcation_session.vars };
        out.sweep_2 = { &c->param_index_2, &c->sweep_over_var_2, &c->var_sweep_index_2,
                        &c->sweep_over_h_2, &m.bifurcation_session.params, &m.bifurcation_session.vars };
        out.label = &c->label;
        return true;
    }
    case BroadcastTab::LLE: {
        auto& v = m.lle_session.curves;
        resolved_idx = (idx < 0) ? m.lle_session.active_curve_index : idx;
        auto* c = config_at(v, resolved_idx);
        if (!c) return false;
        out.h = &c->h_text; out.symmetry = &c->symmetry_s;
        out.t_max = &c->t_max_text; out.transient = &c->transient_text;
        out.max_value = &c->max_value_text;
        out.params = &c->param_values; out.ics = &c->initial_conditions;
        out.scheme = &c->scheme;
        out.sweep_lo = &c->param_lo_text;     out.sweep_hi = &c->param_hi_text;
        out.sweep_lo_2 = &c->param_lo_2_text; out.sweep_hi_2 = &c->param_hi_2_text;
        out.resolution = &c->n_pts_text;
        out.sweep   = { &c->param_index,   &c->sweep_over_var,   &c->var_sweep_index,
                        &c->sweep_over_h,   &m.lle_session.params, &m.lle_session.vars };
        out.sweep_2 = { &c->param_index_2, &c->sweep_over_var_2, &c->var_sweep_index_2,
                        &c->sweep_over_h_2, &m.lle_session.params, &m.lle_session.vars };
        out.label = &c->label;
        return true;
    }
    case BroadcastTab::LS: {
        auto& v = m.ls_session.curves;
        resolved_idx = (idx < 0) ? m.ls_session.active_curve_index : idx;
        auto* c = config_at(v, resolved_idx);
        if (!c) return false;
        out.h = &c->h_text; out.symmetry = &c->symmetry_s;
        out.t_max = &c->t_max_text; out.transient = &c->transient_text;
        out.max_value = &c->max_value_text;
        out.params = &c->param_values; out.ics = &c->initial_conditions;
        out.scheme = &c->scheme;
        out.sweep_lo = &c->param_lo_text;     out.sweep_hi = &c->param_hi_text;
        out.sweep_lo_2 = &c->param_lo_2_text; out.sweep_hi_2 = &c->param_hi_2_text;
        out.resolution = &c->n_pts_text;
        out.sweep   = { &c->param_index,   &c->sweep_over_var,   &c->var_sweep_index,
                        &c->sweep_over_h,   &m.ls_session.params, &m.ls_session.vars };
        out.sweep_2 = { &c->param_index_2, &c->sweep_over_var_2, &c->var_sweep_index_2,
                        &c->sweep_over_h_2, &m.ls_session.params, &m.ls_session.vars };
        out.label = &c->label;
        return true;
    }
    case BroadcastTab::Dft1D: {
        auto& v = m.dft1d_session.configs;
        resolved_idx = (idx < 0) ? m.dft1d_session.active_config_index : idx;
        auto* c = config_at(v, resolved_idx);
        if (!c) return false;
        out.h = &c->h_text; out.symmetry = &c->symmetry_s;
        out.t_max = &c->t_max_text; out.transient = &c->transient_text;
        out.pre_scaller = &c->pre_scaller_text; out.max_value = &c->max_value_text;
        out.params = &c->param_values; out.ics = &c->initial_conditions;
        out.scheme = &c->scheme;
        out.sweep_lo = &c->param_lo_text; out.sweep_hi = &c->param_hi_text;
        out.resolution = &c->n_pts_text;   // Resolution X; по Y у DFT частоты
        out.sweep = { &c->param_index, &c->sweep_over_var, &c->var_sweep_index,
                      &c->sweep_over_h, &m.dft1d_session.params, &m.dft1d_session.vars };
        out.label = &c->label;
        return true;
    }
    case BroadcastTab::Basins: {
        auto& v = m.basins_session.configs;
        resolved_idx = (idx < 0) ? m.basins_session.active_config_index : idx;
        auto* c = config_at(v, resolved_idx);
        if (!c) return false;
        out.h = &c->h_text; out.symmetry = &c->symmetry_s;
        out.t_max = &c->t_max_text; out.transient = &c->transient_text;
        out.pre_scaller = &c->pre_scaller_text; out.max_value = &c->max_value_text;
        out.params = &c->param_values; out.ics = &c->initial_conditions;
        out.scheme = &c->scheme;
        // Свипа параметра у Basins нет: обе оси — сетка начальных условий.
        // n_pts_text при этом СТОРОНА этой сетки, и рассылка Resolution с
        // 1D-вкладки поднимет её в квадрате — счётчик "(N differ)" и Ctrl+Z
        // на этот случай и нужны.
        out.resolution = &c->n_pts_text;
        out.label = &c->label;
        return true;
    }
    case BroadcastTab::FastSync: {
        // НУ у FastSync нет: обе оси сетки — это и есть начальные условия.
        auto& v = m.fastsync_session.configs;
        resolved_idx = (idx < 0) ? m.fastsync_session.active_config_index : idx;
        auto* c = config_at(v, resolved_idx);
        if (!c) return false;
        out.h = &c->h_text; out.symmetry = &c->symmetry_s;
        out.t_max = &c->t_max_text; out.transient = &c->transient_text;
        out.pre_scaller = &c->pre_scaller_text; out.max_value = &c->max_value_text;
        out.params = &c->param_values;
        out.scheme = &c->scheme;
        out.resolution = &c->n_pts_text;   // сторона сетки, см. Basins
        out.label = &c->label;
        return true;
    }
    case BroadcastTab::Custom: {
        // Писать надо в Shared config, а не в подсессии — те всё равно
        // пересобираются из него срезами apply_shared_to_* на каждом Run.
        resolved_idx = 0;
        auto& c = m.custom_session.shared;
        out.h = &c.h_text; out.symmetry = &c.symmetry_s;
        out.t_max = &c.t_max_text; out.transient = &c.transient_text;
        out.pre_scaller = &c.pre_scaller_text; out.max_value = &c.max_value_text;
        out.params = &c.param_values; out.ics = &c.initial_conditions;
        out.scheme = &c.scheme;
        // Свип вкладки — оси ОБЩЕГО config'а (уровень 2D). Подсессии
        // пересобираются из него срезами apply_shared_to_* на каждом Run.
        out.sweep_lo = &c.axis_x_lo_text;   out.sweep_hi = &c.axis_x_hi_text;
        out.sweep_lo_2 = &c.axis_y_lo_text; out.sweep_hi_2 = &c.axis_y_hi_text;
        out.resolution = &c.resolution_text;
        out.sweep   = { &c.axis_x_par_index, &c.axis_x_over_var, &c.axis_x_var_index,
                        &c.axis_x_over_h, &m.custom_session.params, &m.custom_session.vars };
        out.sweep_2 = { &c.axis_y_par_index, &c.axis_y_over_var, &c.axis_y_var_index,
                        &c.axis_y_over_h, &m.custom_session.params, &m.custom_session.vars };
        return true;
    }
    case BroadcastTab::Order: {
        auto& v = m.order_session.configs;
        resolved_idx = (idx < 0) ? m.order_session.active_config_index : idx;
        auto* c = config_at(v, resolved_idx);
        if (!c) return false;
        out.h = &c->h_text; out.symmetry = &c->symmetry_s;
        out.t_max = &c->t_max_text; out.max_value = &c->max_value_text;
        out.params = &c->param_values; out.ics = &c->initial_conditions;
        out.scheme = &c->scheme;
        // Диапазоны и цель осей у Order своей кодировки (axis_x_target,
        // kOrderTargetH) и с параметрическим свипом не совпадают — пропускаем.
        out.label = &c->label;
        return true;
    }
    }
    return false;
}

// Строковое поле слотов по BroadcastField. nullptr — либо поле у вкладки
// отсутствует, либо это одна из карт (Param / InitCondition), их разбирает
// caller.
std::string* string_slot(const BroadcastSlots& s, BroadcastField f) {
    switch (f) {
        case BroadcastField::StepH:      return s.h;
        case BroadcastField::SymmetryS:  return s.symmetry;
        case BroadcastField::TMax:       return s.t_max;
        case BroadcastField::Transient:  return s.transient;
        case BroadcastField::PreScaller: return s.pre_scaller;
        case BroadcastField::MaxValue:   return s.max_value;
        case BroadcastField::Scheme:     return s.scheme;
        case BroadcastField::SweepLo:    return s.sweep_lo;
        case BroadcastField::SweepHi:    return s.sweep_hi;
        case BroadcastField::SweepLo2:   return s.sweep_lo_2;
        case BroadcastField::SweepHi2:   return s.sweep_hi_2;
        case BroadcastField::Resolution: return s.resolution;
        default:                         return nullptr;
    }
}

// Цель свипа по BroadcastField. valid() == false — у вкладки такой оси нет.
const SweepTargetSlot* sweep_slot(const BroadcastSlots& s, BroadcastField f) {
    if (f == BroadcastField::SweepTarget)  return &s.sweep;
    if (f == BroadcastField::SweepTarget2) return &s.sweep_2;
    return nullptr;
}

// Сколько конфигов у вкладки. У Phase и Custom он один на вкладку.
int config_count(AppModel& m, BroadcastTab tab) {
    switch (tab) {
        case BroadcastTab::Phase:       return 1;
        case BroadcastTab::Bifurcation: return (int)m.bifurcation_session.diagrams.size();
        case BroadcastTab::LLE:         return (int)m.lle_session.curves.size();
        case BroadcastTab::LS:          return (int)m.ls_session.curves.size();
        case BroadcastTab::Dft1D:       return (int)m.dft1d_session.configs.size();
        case BroadcastTab::Basins:      return (int)m.basins_session.configs.size();
        case BroadcastTab::FastSync:    return (int)m.fastsync_session.configs.size();
        case BroadcastTab::Custom:      return 1;
        case BroadcastTab::Order:       return (int)m.order_session.configs.size();
    }
    return 0;
}

// Тип конфига — то, по чему работает "применить ко всем такого же типа".
// 1D и 2D разведены сознательно: это разные диаграммы с разным смыслом осей.
std::string group_of(AppModel& m, BroadcastTab tab, int idx) {
    switch (tab) {
        case BroadcastTab::Phase: return "Phase analysis";
        case BroadcastTab::Bifurcation: {
            auto* c = config_at(m.bifurcation_session.diagrams, idx);
            return (c && c->mode_2d) ? "Bifurcation 2D" : "Bifurcation 1D";
        }
        case BroadcastTab::LLE: {
            auto* c = config_at(m.lle_session.curves, idx);
            return (c && c->mode_2d) ? "LLE 2D" : "LLE 1D";
        }
        case BroadcastTab::LS: {
            auto* c = config_at(m.ls_session.curves, idx);
            return (c && c->mode_2d) ? "LS 2D" : "LS 1D";
        }
        case BroadcastTab::Dft1D:    return "1D DFT";
        case BroadcastTab::Basins:   return "Basins";
        case BroadcastTab::FastSync: return "Fast Synchro";
        case BroadcastTab::Custom:   return "Custom";
        case BroadcastTab::Order:    return "Order";
    }
    return {};
}

std::map<std::string, std::string>* map_slot(const BroadcastSlots& s, BroadcastField f) {
    if (f == BroadcastField::Param)         return s.params;
    if (f == BroadcastField::InitCondition) return s.ics;
    return nullptr;
}

} // namespace

std::string encode_sweep_target(int par_index, bool over_var, int var_index, bool over_h) {
    if (over_h)        return "h";
    if (over_var)      return "var:" + std::to_string(var_index);
    if (par_index < 0) return "s";
    return "par:" + std::to_string(par_index);
}

void AppModel::push_undo(std::string description, std::function<void()> undo,
                         std::function<void()> redo) {
    if (!undo) return;
    auto& st = undo_stacks[app_mode];
    st.push_back({ std::move(description), std::move(undo), std::move(redo) });
    // Запись держит прежние значения полей, а не всю сессию, поэтому 200 штук
    // на вкладку стоят копейки.
    while (st.size() > kUndoDepth) st.pop_front();
    undo_just_happened_tabs[app_mode] = false;
    // Новая команда обрывает ветку повтора — как в любом редакторе: вернуть
    // отменённое после того, как поверх сделали другое, уже некуда.
    redo_stacks[app_mode].clear();
}

namespace {
// Снять верхнюю запись стека этой вкладки, применить её сторону (undo/redo) и
// переложить в противоположный стек. Обе команды отличаются только этим.
bool pop_apply_move(std::map<AppModel::AppMode, std::deque<UndoRecord>>& from,
                    std::map<AppModel::AppMode, std::deque<UndoRecord>>& to,
                    AppModel::AppMode tab, bool forward,
                    const char* verb, std::string& note) {
    auto it = from.find(tab);
    if (it == from.end() || it->second.empty()) return false;
    UndoRecord r = std::move(it->second.back());
    it->second.pop_back();
    const std::function<void()>& act = forward ? r.redo : r.undo;
    if (act) act();
    note = std::string(verb) + ": " + r.description;
    auto& dst = to[tab];
    dst.push_back(std::move(r));
    while (dst.size() > AppModel::kUndoDepth) dst.pop_front();
    return true;
}
} // namespace

bool AppModel::undo_last() {
    const bool ok = pop_apply_move(undo_stacks, redo_stacks, app_mode, /*forward*/false,
                                   "Undone", undo_note);
    if (ok) undo_just_happened_tabs[app_mode] = true;
    return ok;
}

bool AppModel::redo_last() {
    const bool ok = pop_apply_move(redo_stacks, undo_stacks, app_mode, /*forward*/true,
                                   "Redone", undo_note);
    if (ok) undo_just_happened_tabs[app_mode] = true;
    return ok;
}

bool AppModel::undo_just_happened() const {
    auto it = undo_just_happened_tabs.find(app_mode);
    return it != undo_just_happened_tabs.end() && it->second;
}

const std::string* AppModel::undo_top() const {
    auto it = undo_stacks.find(app_mode);
    return (it == undo_stacks.end() || it->second.empty())
         ? nullptr : &it->second.back().description;
}

const std::string* AppModel::redo_top() const {
    auto it = redo_stacks.find(app_mode);
    // Повторять нечего, если у записи нет обратного хода.
    if (it == redo_stacks.end() || it->second.empty()) return nullptr;
    return it->second.back().redo ? &it->second.back().description : nullptr;
}

// Есть ли у этого конфига такое поле вообще. Нужно перечислению целей: вкладка
// без поля не должна появляться в списке галочек и обещать применение.
static bool has_field(const BroadcastSlots& s, BroadcastField f) {
    if (map_slot(s, f)) return true;
    if (const SweepTargetSlot* sw = sweep_slot(s, f)) return sw->valid();
    return string_slot(s, f) != nullptr;
}

std::vector<BroadcastTarget> AppModel::broadcast_targets(BroadcastField f) {
    std::vector<BroadcastTarget> out;
    for (BroadcastTab tab : kBroadcastTabs) {
        const int n = config_count(*this, tab);
        for (int i = 0; i < n; ++i) {
            BroadcastSlots slots;
            int idx = -1;
            if (!slots_for(*this, tab, i, slots, idx)) continue;
            if (!has_field(slots, f)) continue;
            BroadcastTarget t;
            t.tab   = tab;
            t.idx   = idx;
            t.group = group_of(*this, tab, idx);
            // У Phase и Custom конфиг один и подписи не имеет — тогда именем
            // служит сам тип, иначе в списке галочек была бы пустая строка.
            t.label = slots.label ? *slots.label : t.group;
            out.push_back(std::move(t));
        }
    }
    return out;
}

int AppModel::broadcast_field(BroadcastField f, const std::string& name,
                              const std::string& value, bool apply,
                              const std::string& label,
                              const std::vector<BroadcastTarget>* only,
                              const std::string& scope_note) {
    // Прежнее состояние одного поля — для стека отмены. Адрес не храним:
    // вкладку с конфигом могут закрыть, и указатель повиснет. Храним, где
    // искать (вкладка + индекс + label) и что вернуть.
    struct Prev {
        BroadcastTab tab;
        int          idx;
        std::string  label;    // пусто = у вкладки конфигов нет (Phase, Custom)
        bool         existed;  // для карт: был ли ключ. Нет — откат его удалит
        std::string  value;
    };
    std::vector<Prev> prev;
    int n = 0;

    // Одна цель. req_idx < 0 — активный конфиг вкладки (обход "по всем
    // вкладкам"); иначе именно этот конфиг (адресный список из меню).
    auto touch = [&](BroadcastTab tab, int req_idx) {
        BroadcastSlots slots;
        int idx = -1;
        if (!slots_for(*this, tab, req_idx, slots, idx)) return;
        const std::string cfg_label = slots.label ? *slots.label : std::string();

        if (auto* m = map_slot(slots, f)) {
            auto it = m->find(name);
            if (it != m->end()) {
                if (it->second == value) return;
                ++n;
                if (apply) {
                    prev.push_back({ tab, idx, cfg_label, true, it->second });
                    it->second = value;
                }
            } else {
                // Ключа нет — конфиг собран до того, как параметр появился в
                // системе. Это расхождение; заводим ключ только когда реально
                // применяем, иначе сухой прогон менял бы состояние.
                ++n;
                if (apply) {
                    prev.push_back({ tab, idx, cfg_label, false, std::string() });
                    (*m)[name] = value;
                }
            }
            return;
        }

        if (const SweepTargetSlot* sw = sweep_slot(slots, f)) {
            if (!sw->valid()) return;
            const std::string cur = encode_sweep(*sw);
            if (cur == value) return;
            // Проверяем разбор ДО подсчёта: если индекс за пределами алфавита
            // этой вкладки, применить нечего, и показывать её в "(N differ)"
            // значит обещать то, чего не будет.
            SweepTargetSlot probe = *sw;
            int  p_par = *sw->par_index, p_vi = *sw->var_index;
            bool p_ov  = *sw->over_var,  p_oh = *sw->over_h;
            probe.par_index = &p_par; probe.over_var = &p_ov;
            probe.var_index = &p_vi;  probe.over_h   = &p_oh;
            if (!decode_sweep(probe, value)) return;
            ++n;
            if (apply) {
                prev.push_back({ tab, idx, cfg_label, true, cur });
                decode_sweep(*sw, value);
            }
            return;
        }

        std::string* dst = string_slot(slots, f);
        if (!dst || *dst == value) return;
        ++n;
        if (apply) {
            prev.push_back({ tab, idx, cfg_label, true, *dst });
            *dst = value;
        }
    };

    if (only) { for (const auto& t : *only) touch(t.tab, t.idx); }
    else      { for (BroadcastTab tab : kBroadcastTabs) touch(tab, -1); }

    if (apply && !prev.empty()) {
        // Одна запись на команду, а не по одной на конфиг: пользователь нажал
        // один пункт меню и ждёт, что один Ctrl+Z его отменит.
        const std::string& shown = label.empty() ? name : label;
        const std::string what = (shown.empty() ? ("value " + value) : (shown + " = " + value))
                               + " -> " + scope_note;

        // Отмена и повтор — одна и та же процедура, отличаются только списком
        // значений: назад кладём снятое, вперёд — то, что команда написала.
        auto writer = [this, f, name](std::vector<Prev> snaps) {
            return [this, f, name, snaps = std::move(snaps)]() {
                for (const auto& p : snaps) {
                    BroadcastSlots slots;
                    int idx = -1;
                    if (!slots_for(*this, p.tab, p.idx, slots, idx)) continue;
                    // На записанном индексе мог оказаться другой конфиг
                    // (вкладку закрыли, соседние сдвинулись) — тогда
                    // пропускаем, а не переписываем чужое.
                    if (!p.label.empty() && (!slots.label || *slots.label != p.label)) continue;
                    if (auto* m = map_slot(slots, f)) {
                        if (p.existed) (*m)[name] = p.value;
                        else           m->erase(name);
                    } else if (const SweepTargetSlot* sw = sweep_slot(slots, f)) {
                        decode_sweep(*sw, p.value);
                    } else if (std::string* dst = string_slot(slots, f)) {
                        *dst = p.value;
                    }
                }
            };
        };
        std::vector<Prev> after;
        after.reserve(prev.size());
        for (const auto& p : prev) after.push_back({ p.tab, p.idx, p.label, true, value });
        push_undo(what, writer(std::move(prev)), writer(std::move(after)));
    }
    return n;
}

void AppModel::propagate_to_sessions() {
    // refresh_symbols может упасть, если в полях ещё что-то невалидное —
    // молча игнорим, оставим прежние vars/params, пользователь увидит ошибку
    // на следующем Run.
    refresh_symbols();

    // custom_schemes — глубокая копия. Тот же sync, что и в draw_gui каждый
    // кадр, но здесь он гарантирован сразу после Save (важно, если save
    // вызвали, не открывая другие вкладки — кеш-ключ NVRTC по new body
    // должен быть готов).
    phase_session.custom_schemes       = custom_schemes;
    bifurcation_session.custom_schemes = custom_schemes;
    lle_session.custom_schemes         = custom_schemes;
    ls_session.custom_schemes          = custom_schemes;
    dft1d_session.custom_schemes       = custom_schemes;
    basins_session.custom_schemes      = custom_schemes;
    fastsync_session.custom_schemes    = custom_schemes;
    custom_session.custom_schemes                = custom_schemes;
    custom_session.bif_session.custom_schemes    = custom_schemes;
    custom_session.lle_session.custom_schemes    = custom_schemes;
    custom_session.ls_session.custom_schemes     = custom_schemes;
    custom_session.phase_session.custom_schemes  = custom_schemes;
    custom_session.basins_session.custom_schemes = custom_schemes;

    // Экстраполяционные обёртки едут тем же маршрутом: в сессии нужны только
    // имена, тело каждый раз пересобирает compute_krs_for_scheme.
    phase_session.wrapper_schemes       = wrapper_schemes;
    bifurcation_session.wrapper_schemes = wrapper_schemes;
    lle_session.wrapper_schemes         = wrapper_schemes;
    ls_session.wrapper_schemes          = wrapper_schemes;
    dft1d_session.wrapper_schemes       = wrapper_schemes;
    basins_session.wrapper_schemes      = wrapper_schemes;
    fastsync_session.wrapper_schemes    = wrapper_schemes;
    custom_session.wrapper_schemes                = wrapper_schemes;
    custom_session.bif_session.wrapper_schemes    = wrapper_schemes;
    custom_session.lle_session.wrapper_schemes    = wrapper_schemes;
    custom_session.ls_session.wrapper_schemes     = wrapper_schemes;
    custom_session.phase_session.wrapper_schemes  = wrapper_schemes;
    custom_session.basins_session.wrapper_schemes = wrapper_schemes;

    // Mirror enabled built-in schemes from the current model into every session.
    const std::vector<std::string> enabled_now = enabled_builtins_from_record(to_record());
    phase_session.enabled_builtin_schemes       = enabled_now;
    bifurcation_session.enabled_builtin_schemes = enabled_now;
    lle_session.enabled_builtin_schemes         = enabled_now;
    ls_session.enabled_builtin_schemes          = enabled_now;
    dft1d_session.enabled_builtin_schemes       = enabled_now;
    basins_session.enabled_builtin_schemes      = enabled_now;
    fastsync_session.enabled_builtin_schemes    = enabled_now;
    custom_session.enabled_builtin_schemes                = enabled_now;
    custom_session.bif_session.enabled_builtin_schemes    = enabled_now;
    custom_session.lle_session.enabled_builtin_schemes    = enabled_now;
    custom_session.ls_session.enabled_builtin_schemes     = enabled_now;
    custom_session.phase_session.enabled_builtin_schemes  = enabled_now;
    custom_session.basins_session.enabled_builtin_schemes = enabled_now;

    // sys обновляем для built-in схем (Euler/RK4/...): они используют
    // sys.rhs внутри compute_krs_for_scheme → codegen_scheme. Если уравнения
    // в Library поправили, без этой пересборки старый sys остаётся в сессиях
    // до следующего start_*_analysis.
    try {
        System built = build_system();
        phase_session.sys       = built;
        bifurcation_session.sys = built;
        lle_session.sys         = built;
        ls_session.sys          = built;
        dft1d_session.sys       = built;
        basins_session.sys      = built;
        fastsync_session.sys    = built;
        custom_session.sys                = built;
        custom_session.bif_session.sys    = built;
        custom_session.lle_session.sys    = built;
        custom_session.ls_session.sys     = built;
        custom_session.phase_session.sys  = built;
        custom_session.basins_session.sys = built;
    }
    catch (...) {
        // система ещё неполна — следующий Run покажет ошибку парсинга.
    }

    // Phase кеширует krs_code. Принудительно инвалидируем — snapshot_phase
    // на ближайшем Run пересчитает его от свежего sys / custom_schemes.
    phase_session.krs_code.clear();
}

// Peak-knobs: число -> текст буфера ввода

namespace {
// Печатает значение так, чтобы оно (а) читалось человеком и (б) парсилось обратно ровно в то же
// double. %.6g даёт "1e-14" вместо "1.000e-14", но для значений, не влезающих в шесть значащих
// цифр, молча соврал бы — поэтому повышаем точность, пока round-trip не совпадёт. %.17g —
// гарантированный потолок для double.
std::string fmt_peak_num(double v) {
    char buf[64];
    for (int prec : {6, 9, 17}) {
        std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
        if (std::strtod(buf, nullptr) == v) break;
    }
    return std::string(buf);
}
} // namespace

void AppModel::sync_peak_text() {
    peak_eps_fixed_point_text     = fmt_peak_num(peak.eps_fixed_point);
    peak_eps_peak_delta_text      = fmt_peak_num(peak.eps_peak_delta);
    peak_eps_interPeak_delta_text = fmt_peak_num(peak.eps_interPeak_delta);
    peak_threshold_text           = fmt_peak_num(peak.peak_threshold);
    peak_max_amount_text          = std::to_string(peak.max_amount_of_peaks);
}

// Cross-analysis batch queue

namespace {

// Всё, от чего зависит КЛЮЧ кэша модуля, и ничего больше: схема, система, размерность и вид свипа
// по каждой оси. Диапазоны, число точек, время интегрирования в ключ не входят — поэтому набор
// чисел в полях прогрев не перезапускает, а переключение схемы перезапускает.
std::string prewarm_session_id(const System& sys, const std::vector<std::string>& vars,
                               const std::vector<CustomScheme>& custom_schemes) {
    std::string joined;
    for (const std::string& r : sys.rhs) { joined += r; joined += '\x1f'; }
    // Тело кастомной схемы правится без смены имени, поэтому в идентификатор идёт и оно.
    for (const CustomScheme& cs : custom_schemes) { joined += cs.name; joined += cs.body; joined += '\x1f'; }
    return std::to_string(std::hash<std::string>{}(joined)) + ":" + std::to_string(vars.size());
}

template <class Cfg>
std::string prewarm_sig(const std::string& session_id, const Cfg& c) {
    std::string s = session_id;
    s += '\x1f';
    s += c.scheme;
    s += c.mode_2d              ? "|2d" : "|1d";
    s += c.sweep_over_var       ? 'v' : '-';
    s += c.sweep_over_h         ? 'h' : '-';
    s += c.sweep_over_var_2     ? 'V' : '-';
    s += c.sweep_over_h_2       ? 'H' : '-';
    s += c.continuation         ? 'c' : '-';
    s += c.use_gpu              ? 'g' : '-';
    // Настройки пиков и fmad тоже входят в ключ модуля (см. hash_key в parametric_engine.cpp).
    s += ":pk" + std::to_string(peak_config_epoch());
    s += get_nvrtc_fmad() ? ":fm1" : ":fm0";
    return s;
}

}  // namespace

// Прогрев ПЕРВОГО запуска. Зовётся каждый кадр рядом с тиками очередей; почти всегда это
// несколько сравнений строк и выход.
void AppModel::poll_parametric_prewarm() {
    // Не соревнуемся с идущим расчётом: пока он считает, хвост очереди греет
    // prewarm_rest_of_parametric_queue, а лишняя компиляция отняла бы у него CPU.
    if (bifurcation_session.in_flight || lle_session.in_flight || ls_session.in_flight) return;
    if (!parametric_queue.empty()) return;
    if (parametric_prewarm_future.valid() &&
        parametric_prewarm_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return;

    const auto now = std::chrono::steady_clock::now();
    std::function<void(ParametricEngine&)> task;

    // Скан одной сессии. Решение «что и когда греть» живёт в PrewarmWatch::pick (prewarm_watch.h),
    // здесь — только сбор сигнатур и сборка задачи. Задача строится ЗДЕСЬ, на UI-потоке (внутри
    // codegen), фоновому потоку остаётся NVRTC.
    auto scan = [&](PrewarmWatch& w, const std::string& session_id,
                    size_t count, auto&& sig_of, auto&& make_task) {
        if (task) return;
        std::vector<std::string> current;
        current.reserve(count);
        for (size_t i = 0; i < count; ++i) current.push_back(sig_of(i, session_id));
        const int idx = w.pick(current, now, std::chrono::milliseconds(400));
        if (idx < 0) return;
        auto t = make_task(idx);
        if (t) task = std::move(t);
    };

    scan(prewarm_watch_bif,
         prewarm_session_id(bifurcation_session.sys, bifurcation_session.vars,
                            bifurcation_session.custom_schemes),
         bifurcation_session.diagrams.size(),
         [&](size_t i, const std::string& id) { return prewarm_sig(id, bifurcation_session.diagrams[i]); },
         [&](int i) { return bifurcation_session.prewarm_task(i); });
    scan(prewarm_watch_lle,
         prewarm_session_id(lle_session.sys, lle_session.vars, lle_session.custom_schemes),
         lle_session.curves.size(),
         [&](size_t i, const std::string& id) { return prewarm_sig(id, lle_session.curves[i]); },
         [&](int i) { return lle_session.prewarm_task(i); });
    scan(prewarm_watch_ls,
         prewarm_session_id(ls_session.sys, ls_session.vars, ls_session.custom_schemes),
         ls_session.curves.size(),
         [&](size_t i, const std::string& id) { return prewarm_sig(id, ls_session.curves[i]); },
         [&](int i) { return ls_session.prewarm_task(i); });

    if (!task) return;
    // Движок создаётся тут же: CUDA-контекст он поднимает лениво, уже в фоновом потоке.
    if (!parametric_engine) parametric_engine = std::make_unique<ParametricEngine>();
    ParametricEngine* eng = parametric_engine.get();
    parametric_prewarm_future = std::async(std::launch::async,
                                           [eng, task = std::move(task)] { task(*eng); });
}

// Прогрев хвоста очереди. Запросы собираются ЗДЕСЬ, на UI-потоке: фоновому потоку нельзя читать
// сессии, их правит пользователь.
void AppModel::prewarm_rest_of_parametric_queue() {
    if (!parametric_engine || parametric_queue.empty()) return;
    // Присваивание поверх работающего future заблокировало бы UI-поток в его деструкторе — в этом
    // случае просто пропускаем: не прогрелось, значит Run скомпилирует сам (см. analysis_session).
    if (parametric_prewarm_future.valid() &&
        parametric_prewarm_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return;

    std::vector<std::function<void(ParametricEngine&)>> tasks;
    for (const ParametricQueueItem& it : parametric_queue) {
        std::function<void(ParametricEngine&)> t;
        switch (it.kind) {
        case ParametricQueueItem::Kind::Bifurcation: t = bifurcation_session.prewarm_task(it.index); break;
        case ParametricQueueItem::Kind::LLE:         t = lle_session.prewarm_task(it.index);         break;
        case ParametricQueueItem::Kind::LS:          t = ls_session.prewarm_task(it.index);          break;
        }
        if (t) tasks.push_back(std::move(t));
        if ((int)tasks.size() >= kPrewarmAhead) break;
    }
    if (tasks.empty()) return;

    ParametricEngine* eng = parametric_engine.get();
    parametric_prewarm_future = std::async(std::launch::async, [eng, tasks = std::move(tasks)] {
        for (const auto& task : tasks) task(*eng);
    });
}

bool AppModel::start_next_in_parametric_queue() {
    if (bifurcation_session.in_flight ||
        lle_session.in_flight ||
        ls_session.in_flight) return false;
    if (parametric_queue.empty()) return false;
    if (!parametric_engine) parametric_engine = std::make_unique<ParametricEngine>();
    while (!parametric_queue.empty()) {
        ParametricQueueItem it = parametric_queue.front();
        parametric_queue.pop_front();
        bool ok = false;
        switch (it.kind) {
        case ParametricQueueItem::Kind::Bifurcation:
            if (it.index >= 0 && it.index < (int)bifurcation_session.diagrams.size())
                ok = bifurcation_session.run_async(*parametric_engine, it.index);
            break;
        case ParametricQueueItem::Kind::LLE:
            if (it.index >= 0 && it.index < (int)lle_session.curves.size())
                ok = lle_session.run_async(*parametric_engine, it.index);
            break;
        case ParametricQueueItem::Kind::LS:
            if (it.index >= 0 && it.index < (int)ls_session.curves.size())
                ok = ls_session.run_async(*parametric_engine, it.index);
            break;
        }
        if (ok) { prewarm_rest_of_parametric_queue(); return true; }
        // ok == false (например, krs пуст / индекс плохой) — last_error
        // выставлен соответствующим run_async; идём дальше.
    }
    return false;
}

// Custom-tab pipeline queue drainer

namespace {
// Format a double for insertion into a text-based numeric input field.
// Все три вызывающих (pin_fixed_param / _h / _ic) пиннят ЗНАЧЕНИЕ УЗЛА сетки, пришедшее из snap'а
// по карте или по 1D-графику, поэтому формат обязан быть round-trip: значение уходит в ядро через
// parse_num, и шесть значащих цифр (стояли здесь раньше) отрезали ~10 знаков — drill-down считался
// не в том параметре, в котором посчитана ячейка. См. fmt_num_shortest в num_parse.h.
std::string fmt_num_for_input(double v) {
    return fmt_num_shortest(v);
}

// Закреплённая ось (та, что держится фиксированной, пока свипуется другая)
// может идти по параметру, по начальному условию или по ШАГУ — и «зафиксировать»
// её значит подставить значение в РАЗНЫЕ поля конфига. Все три подстановки
// вызываются вместе и решают по одной и той же EffectiveSweep: срабатывает
// ровно одна.
//
// Флаги приходят одним объектом намеренно. Раньше каждая функция получала свои
// флаги отдельными аргументами, и pin_fixed_param проверяла только over_var —
// а при свипе по шагу over_var как раз false (см. draw_sweep_target_combo:
// выбор "dt (h)" ставит over_h и гасит over_var). В результате значение ШАГА
// уходило в параметр с индексом par_index — обычно первый в списке, — и в
// панели Parameters на глазах менялся посторонний параметр.
void pin_fixed_param(std::map<std::string,std::string>& pv,
                     const std::vector<std::string>& params,
                     const EffectiveSweep& e, double value) {
    if (e.over_var || e.over_h) return;
    if (e.par_index < 0 || e.par_index >= (int)params.size()) return;
    pv[params[e.par_index]] = fmt_num_for_input(value);
}

void pin_fixed_h(std::string& h_text, const EffectiveSweep& e, double value) {
    if (!e.over_h) return;
    if (value <= 0.0) return;             // шаг <= 0 бессмыслен, оставляем прежний
    h_text = fmt_num_for_input(value);
}

void pin_fixed_ic(std::map<std::string,std::string>& ic,
                  const std::vector<std::string>& vars,
                  const EffectiveSweep& e, double value) {
    if (!e.over_var || e.over_h) return;
    if (e.var_index < 0 || e.var_index >= (int)vars.size()) return;
    ic[vars[e.var_index]] = fmt_num_for_input(value);
}
} // namespace

bool AppModel::start_next_in_custom_queue() {
    if (custom_session.any_in_flight()) return false;
    if (custom_queue.empty()) return false;
    if (!parametric_engine) parametric_engine = std::make_unique<ParametricEngine>();

    using K = CustomQueueItem::Kind;
    auto& cs = custom_session;
    const auto& shared = cs.shared;

    while (!custom_queue.empty()) {
        CustomQueueItem it = custom_queue.front();
        custom_queue.pop_front();
        bool ok = false;

        switch (it.kind) {
        case K::Bif2D:
            if (cs.bif_session.diagrams.size() > 0) {
                apply_shared_to_bif2d(shared, cs.bif_session.diagrams[0]);
                ok = cs.bif_session.run_async(*parametric_engine, 0);
            }
            break;
        case K::LLE2D:
            if (cs.lle_session.curves.size() > 0) {
                apply_shared_to_lle2d(shared, cs.lle_session.curves[0]);
                ok = cs.lle_session.run_async(*parametric_engine, 0);
            }
            break;
        case K::LS2D:
            if (cs.ls_session.curves.size() > 0) {
                apply_shared_to_ls2d(shared, cs.ls_session.curves[0]);
                ok = cs.ls_session.run_async(*parametric_engine, 0);
            }
            break;

        case K::Bif1D_X:
            if (cs.bif_session.diagrams.size() > 1) {
                auto& c = cs.bif_session.diagrams[1];
                apply_shared_to_bif1d(shared, c, 0);
                // Pin the OTHER (Y) axis at fix_y.
                EffectiveSweep swy = effective_sweep_y(shared);
                pin_fixed_param(c.param_values, cs.params, swy, shared.fix_y_value);
                pin_fixed_ic   (c.initial_conditions, cs.vars, swy, shared.fix_y_value);
                pin_fixed_h    (c.h_text, swy, shared.fix_y_value);
                ok = cs.bif_session.run_async(*parametric_engine, 1);
            }
            break;
        case K::Bif1D_Y:
            if (cs.bif_session.diagrams.size() > 2) {
                auto& c = cs.bif_session.diagrams[2];
                apply_shared_to_bif1d(shared, c, 1);
                EffectiveSweep swx = effective_sweep_x(shared);
                pin_fixed_param(c.param_values, cs.params, swx, shared.fix_x_value);
                pin_fixed_ic   (c.initial_conditions, cs.vars, swx, shared.fix_x_value);
                pin_fixed_h    (c.h_text, swx, shared.fix_x_value);
                ok = cs.bif_session.run_async(*parametric_engine, 2);
            }
            break;
        case K::LLE1D_X:
            if (cs.lle_session.curves.size() > 1) {
                auto& c = cs.lle_session.curves[1];
                apply_shared_to_lle1d(shared, cs.lle_session.curves[0], c, 0);
                EffectiveSweep swy = effective_sweep_y(shared);
                pin_fixed_param(c.param_values, cs.params, swy, shared.fix_y_value);
                pin_fixed_ic   (c.initial_conditions, cs.vars, swy, shared.fix_y_value);
                pin_fixed_h    (c.h_text, swy, shared.fix_y_value);
                ok = cs.lle_session.run_async(*parametric_engine, 1);
            }
            break;
        case K::LLE1D_Y:
            if (cs.lle_session.curves.size() > 2) {
                auto& c = cs.lle_session.curves[2];
                apply_shared_to_lle1d(shared, cs.lle_session.curves[0], c, 1);
                EffectiveSweep swx = effective_sweep_x(shared);
                pin_fixed_param(c.param_values, cs.params, swx, shared.fix_x_value);
                pin_fixed_ic   (c.initial_conditions, cs.vars, swx, shared.fix_x_value);
                pin_fixed_h    (c.h_text, swx, shared.fix_x_value);
                ok = cs.lle_session.run_async(*parametric_engine, 2);
            }
            break;
        case K::LS1D_X:
            if (cs.ls_session.curves.size() > 1) {
                auto& c = cs.ls_session.curves[1];
                apply_shared_to_ls1d(shared, cs.ls_session.curves[0], c, 0);
                EffectiveSweep swy = effective_sweep_y(shared);
                pin_fixed_param(c.param_values, cs.params, swy, shared.fix_y_value);
                pin_fixed_ic   (c.initial_conditions, cs.vars, swy, shared.fix_y_value);
                pin_fixed_h    (c.h_text, swy, shared.fix_y_value);
                ok = cs.ls_session.run_async(*parametric_engine, 1);
            }
            break;
        case K::LS1D_Y:
            if (cs.ls_session.curves.size() > 2) {
                auto& c = cs.ls_session.curves[2];
                apply_shared_to_ls1d(shared, cs.ls_session.curves[0], c, 1);
                EffectiveSweep swx = effective_sweep_x(shared);
                pin_fixed_param(c.param_values, cs.params, swx, shared.fix_x_value);
                pin_fixed_ic   (c.initial_conditions, cs.vars, swx, shared.fix_x_value);
                pin_fixed_h    (c.h_text, swx, shared.fix_x_value);
                ok = cs.ls_session.run_async(*parametric_engine, 2);
            }
            break;

        case K::Phase: {
            apply_shared_to_phase(shared, cs.phase_session, cs.vars);
            cs.phase_session.regenerate_krs();
            ok = cs.phase_session.recompute_async();
            break;
        }
        case K::Basins:
            if (cs.basins_session.configs.size() > 0) {
                auto& c = cs.basins_session.configs[0];
                apply_shared_to_basins(shared, c);
                ok = cs.basins_session.run_async(*parametric_engine, 0);
            }
            break;
        }

        if (ok) return true;
        // ok == false — skip and try the next item.
    }
    return false;
}

namespace {
// Помощник: убрать из очереди элементы с (kind == k && index == removed),
// для оставшихся того же kind: index > removed → --index.
void cleanup_queue_after_removal(std::deque<ParametricQueueItem>& q,
                                 ParametricQueueItem::Kind k, int removed) {
    for (auto it = q.begin(); it != q.end(); ) {
        if (it->kind == k && it->index == removed) it = q.erase(it);
        else ++it;
    }
    for (auto& it : q)
        if (it.kind == k && it.index > removed) --it.index;
}

// Помощник: у окон того же kind убрать `removed` из members, у оставшихся
// индексов > removed сделать --index. Окно, у которого members опустел
// (был последний diagram/curve, его удалили) — удаляется целиком, чтобы не
// висело пустое окно без данных.
void cleanup_plot_windows_after_removal(std::vector<ParametricPlotWindow>& wins,
                                        ParametricPlotWindow::Kind k, int removed) {
    for (auto& w : wins) {
        if (w.kind != k) continue;
        auto& m = w.members;
        m.erase(std::remove(m.begin(), m.end(), removed), m.end());
        for (auto& idx : m) if (idx > removed) --idx;
    }
    wins.erase(std::remove_if(wins.begin(), wins.end(),
                              [k](const ParametricPlotWindow& w) {
                                  return w.kind == k && w.members.empty();
                              }),
               wins.end());
}
} // namespace

void AppModel::remove_bifurcation_diagram(int i) {
    bifurcation_session.remove_diagram(i);
    cleanup_queue_after_removal(parametric_queue, ParametricQueueItem::Kind::Bifurcation, i);
    cleanup_plot_windows_after_removal(parametric_plot_windows, ParametricPlotWindow::Kind::Bifurcation, i);
    parametric_plot_windows_dirty = true;
}

void AppModel::remove_lle_curve(int i) {
    lle_session.remove_curve(i);
    cleanup_queue_after_removal(parametric_queue, ParametricQueueItem::Kind::LLE, i);
    cleanup_plot_windows_after_removal(parametric_plot_windows, ParametricPlotWindow::Kind::LLE, i);
    parametric_plot_windows_dirty = true;
}

void AppModel::remove_ls_curve(int i) {
    ls_session.remove_curve(i);
    cleanup_queue_after_removal(parametric_queue, ParametricQueueItem::Kind::LS, i);
    cleanup_plot_windows_after_removal(parametric_plot_windows, ParametricPlotWindow::Kind::LS, i);
    parametric_plot_windows_dirty = true;
}

void AppModel::add_parametric_plot_window(ParametricPlotWindow::Kind kind, bool mode_2d,
                                          std::vector<int> initial_members,
                                          bool colored_1d) {
    ParametricPlotWindow w;
    w.kind = kind;
    w.mode_2d = mode_2d;
    w.colored_1d = colored_1d;
    w.members = std::move(initial_members);
    // A 2D/Colored-1D window shows exactly one heatmap — enforce single-member
    // here too, not just in the UI, since this is the one place all windows
    // go through.
    if ((w.mode_2d || w.colored_1d) && w.members.size() > 1) w.members.resize(1);
    w.id = next_parametric_plot_window_id++;
    w.label = "Plot " + std::to_string(w.id);
    w.label_is_manual = false;   // fresh window → auto-label
    parametric_plot_windows.push_back(std::move(w));
    parametric_plot_windows_dirty = true;
}

void AppModel::remove_parametric_plot_window(int pos) {
    if (pos < 0 || pos >= (int)parametric_plot_windows.size()) return;
    parametric_plot_windows.erase(parametric_plot_windows.begin() + pos);
    parametric_plot_windows_dirty = true;
}

void AppModel::load_or_init_parametric_plot_windows(const std::string& json) {
    if (!json.empty()) {
        session_from_json_parametric_windows(json, parametric_plot_windows);
        // Bump next-id past the highest loaded id so subsequent Add window
        // calls don't collide with existing ones (which would give two rows
        // the same PushID and trigger ImGui's "conflicting ID" warning).
        int max_id = 0;
        for (const auto& w : parametric_plot_windows)
            if (w.id > max_id) max_id = w.id;
        if (max_id >= next_parametric_plot_window_id)
            next_parametric_plot_window_id = max_id + 1;
        return;
    }
    parametric_plot_windows.clear();

    // 1D: one window overlaying every 1D diagram/curve of a kind (reproduces
    // the old "everything on one shared plot" behavior). 2D: one window PER
    // diagram/curve, since a 2D window can only ever show a single heatmap.
    std::vector<int> bd1;
    for (size_t i = 0; i < bifurcation_session.diagrams.size(); ++i) {
        const auto& bd = bifurcation_session.diagrams[i];
        if (bd.mode_2d)
            add_parametric_plot_window(ParametricPlotWindow::Kind::Bifurcation, true, { (int)i });
        else if (bd.colored_1d)
            add_parametric_plot_window(ParametricPlotWindow::Kind::Bifurcation, false, { (int)i }, true);
        else
            bd1.push_back((int)i);
    }
    if (!bd1.empty()) add_parametric_plot_window(ParametricPlotWindow::Kind::Bifurcation, false, bd1);

    std::vector<int> lle1;
    for (size_t i = 0; i < lle_session.curves.size(); ++i) {
        if (lle_session.curves[i].mode_2d)
            add_parametric_plot_window(ParametricPlotWindow::Kind::LLE, true, { (int)i });
        else
            lle1.push_back((int)i);
    }
    if (!lle1.empty()) add_parametric_plot_window(ParametricPlotWindow::Kind::LLE, false, lle1);

    std::vector<int> ls1;
    for (size_t i = 0; i < ls_session.curves.size(); ++i) {
        if (ls_session.curves[i].mode_2d)
            add_parametric_plot_window(ParametricPlotWindow::Kind::LS, true, { (int)i });
        else
            ls1.push_back((int)i);
    }
    if (!ls1.empty()) add_parametric_plot_window(ParametricPlotWindow::Kind::LS, false, ls1);
}

void AppModel::remove_dft1d_config(int i) {
    dft1d_session.remove_config(i);
    // Cleanup dft1d_queue: drop items pointing at the removed index, shift
    // index > i down by one (same pattern as remove_basins_config).
    for (auto it = dft1d_queue.begin(); it != dft1d_queue.end(); ) {
        if (it->index == i) it = dft1d_queue.erase(it);
        else ++it;
    }
    for (auto& it : dft1d_queue)
        if (it.index > i) --it.index;

    // Plot windows: drop `i` from members, shift indices > i down by one;
    // drop windows left with no members (same pattern as
    // cleanup_plot_windows_after_removal, but DFT1D windows have no `kind`).
    for (auto& w : dft1d_plot_windows) {
        auto& m = w.members;
        m.erase(std::remove(m.begin(), m.end(), i), m.end());
        for (auto& idx : m) if (idx > i) --idx;
    }
    dft1d_plot_windows.erase(std::remove_if(dft1d_plot_windows.begin(), dft1d_plot_windows.end(),
                                            [](const Dft1DPlotWindow& w) { return w.members.empty(); }),
                             dft1d_plot_windows.end());
    dft1d_plot_windows_dirty = true;
}

void AppModel::add_dft1d_plot_window(std::vector<int> initial_members) {
    Dft1DPlotWindow w;
    w.members = std::move(initial_members);
    // A DFT1D window always shows exactly one config's heatmap (unlike
    // Parametric's classic-1D windows, which can overlay several) — enforce
    // here too, not just in the "Members..." popup's radio buttons.
    if (w.members.size() > 1) w.members.resize(1);
    w.id = next_dft1d_plot_window_id++;
    w.label = "Plot " + std::to_string(w.id);
    w.label_is_manual = false;   // fresh window → auto-label
    dft1d_plot_windows.push_back(std::move(w));
    dft1d_plot_windows_dirty = true;
}

void AppModel::remove_dft1d_plot_window(int pos) {
    if (pos < 0 || pos >= (int)dft1d_plot_windows.size()) return;
    dft1d_plot_windows.erase(dft1d_plot_windows.begin() + pos);
    dft1d_plot_windows_dirty = true;
}

void AppModel::load_or_init_dft1d_plot_windows(const std::string& json) {
    if (!json.empty()) {
        session_from_json_dft1d_windows(json, dft1d_plot_windows);
        int max_id = 0;
        for (const auto& w : dft1d_plot_windows)
            if (w.id > max_id) max_id = w.id;
        if (max_id >= next_dft1d_plot_window_id)
            next_dft1d_plot_window_id = max_id + 1;
        return;
    }
    dft1d_plot_windows.clear();
    // One window per config, same as Bifurcation's 2D/colored-1D default —
    // a DFT heatmap isn't overlay-friendly, so there is no shared-window case.
    for (size_t i = 0; i < dft1d_session.configs.size(); ++i)
        add_dft1d_plot_window({ (int)i });
}

bool AppModel::start_next_in_dft1d_queue() {
    if (dft1d_session.in_flight) return false;
    if (dft1d_queue.empty()) return false;
    if (!parametric_engine) parametric_engine = std::make_unique<ParametricEngine>();
    while (!dft1d_queue.empty()) {
        Dft1DQueueItem it = dft1d_queue.front();
        dft1d_queue.pop_front();
        if (it.index >= 0 && it.index < (int)dft1d_session.configs.size()) {
            if (dft1d_session.run_async(*parametric_engine, it.index)) return true;
        }
        // ok == false (krs пуст / индекс плохой) — last_error выставлен;
        // идём дальше.
    }
    return false;
}

void AppModel::remove_basins_config(int i) {
    basins_session.remove_config(i);
    // Cleanup basins_queue: drop items pointing at the removed index, shift
    // index > i down by one.
    for (auto it = basins_queue.begin(); it != basins_queue.end(); ) {
        if (it->index == i) it = basins_queue.erase(it);
        else ++it;
    }
    for (auto& it : basins_queue)
        if (it.index > i) --it.index;
}

void AppModel::remove_fastsync_config(int i) {
    fastsync_session.remove_config(i);
    // Cleanup fastsync_queue: drop items pointing at the removed index, shift
    // index > i down by one.
    for (auto it = fastsync_queue.begin(); it != fastsync_queue.end(); ) {
        if (it->index == i) it = fastsync_queue.erase(it);
        else ++it;
    }
    for (auto& it : fastsync_queue)
        if (it.index > i) --it.index;
}

void AppModel::add_order_plot_window(OrderPlotWindow::Kind kind, std::vector<int> initial_members) {
    OrderPlotWindow w;
    w.kind    = kind;
    w.members = std::move(initial_members);
    // Карта — хитмапа: наложить две друг на друга нельзя, поэтому лишние
    // члены отсекаются здесь, а не только радиокнопками в попапе.
    if (kind == OrderPlotWindow::Kind::Map && w.members.size() > 1) w.members.resize(1);
    w.id = next_order_plot_window_id++;
    w.label = "Plot " + std::to_string(w.id);
    w.label_is_manual = false;
    // Порядок p линеен по построению — логарифмировать его нечего.
    if (kind == OrderPlotWindow::Kind::P) w.y_log = false;
    order_plot_windows.push_back(std::move(w));
    order_plot_windows_dirty = true;
}

void AppModel::remove_order_plot_window(int pos) {
    if (pos < 0 || pos >= (int)order_plot_windows.size()) return;
    order_plot_windows.erase(order_plot_windows.begin() + pos);
    order_plot_windows_dirty = true;
}

void AppModel::load_or_init_order_plot_windows(const std::string& json) {
    if (!json.empty()) {
        session_from_json_order_windows(json, order_plot_windows);
        int max_id = 0;
        for (const auto& w : order_plot_windows)
            if (w.id > max_id) max_id = w.id;
        if (max_id >= next_order_plot_window_id)
            next_order_plot_window_id = max_id + 1;
        if (!order_plot_windows.empty()) return;
    }
    order_plot_windows.clear();
    next_order_plot_window_id = 1;
    // Дефолт — одно окно с кривой p, в котором лежат все сеточные конфиги:
    // именно так вкладка и выглядела до появления окон.
    std::vector<int> curves, maps, perfs;
    for (size_t i = 0; i < order_session.configs.size(); ++i) {
        const OrderConfig& c = order_session.configs[i];
        if (c.calc_kind == kOrderCalcPerf) perfs.push_back((int)i);
        else if (c.two_d)                  maps.push_back((int)i);
        else                               curves.push_back((int)i);
    }
    if (!curves.empty()) add_order_plot_window(OrderPlotWindow::Kind::P, curves);
    if (!perfs.empty())  add_order_plot_window(OrderPlotWindow::Kind::Perf, perfs);
    for (int m : maps)   add_order_plot_window(OrderPlotWindow::Kind::Map, { m });
    if (order_plot_windows.empty())
        add_order_plot_window(OrderPlotWindow::Kind::P, {});
}

void AppModel::remove_order_config(int i) {
    order_session.remove_config(i);

    // Окна графиков: выкинуть i из members, сдвинуть большие индексы,
    // выбросить окна, оставшиеся вовсе без членов (как у DFT1D).
    for (auto& w : order_plot_windows) {
        auto& m = w.members;
        m.erase(std::remove(m.begin(), m.end(), i), m.end());
        for (auto& idx : m) if (idx > i) --idx;
    }
    order_plot_windows.erase(std::remove_if(order_plot_windows.begin(), order_plot_windows.end(),
                                            [](const OrderPlotWindow& w) { return w.members.empty(); }),
                             order_plot_windows.end());
    order_plot_windows_dirty = true;
    // Та же чистка очереди, что у Basins/FastSync: выкидываем элементы,
    // указывающие на удалённый конфиг, и сдвигаем те, что были правее.
    for (auto it = order_queue.begin(); it != order_queue.end(); ) {
        if (it->index == i) it = order_queue.erase(it);
        else ++it;
    }
    for (auto& it : order_queue)
        if (it.index > i) --it.index;
}

void AppModel::remove_network_config(int i) {
    network_session.remove_config(i);
    // Та же чистка очереди, что у Order: выкидываем элементы, указывающие на
    // удалённый конфиг, и сдвигаем те, что были правее.
    for (auto it = network_queue.begin(); it != network_queue.end(); ) {
        if (it->index == i) it = network_queue.erase(it);
        else ++it;
    }
    for (auto& it : network_queue)
        if (it.index > i) --it.index;
}

bool AppModel::start_next_in_basins_queue() {
    if (basins_session.in_flight) return false;
    if (basins_queue.empty()) return false;
    if (!parametric_engine) parametric_engine = std::make_unique<ParametricEngine>();
    while (!basins_queue.empty()) {
        BasinsQueueItem it = basins_queue.front();
        basins_queue.pop_front();
        if (it.index >= 0 && it.index < (int)basins_session.configs.size()) {
            if (basins_session.run_async(*parametric_engine, it.index)) return true;
        }
        // ok == false (krs пуст / индекс плохой) — last_error выставлен;
        // идём дальше.
    }
    return false;
}

bool AppModel::start_next_in_order_queue() {
    if (order_session.in_flight) return false;
    if (order_queue.empty()) return false;
    if (!parametric_engine) parametric_engine = std::make_unique<ParametricEngine>();
    while (!order_queue.empty()) {
        OrderQueueItem it = order_queue.front();
        order_queue.pop_front();
        if (it.index >= 0 && it.index < (int)order_session.configs.size()) {
            if (order_session.run_async(*parametric_engine, it.index)) return true;
        }
        // ok == false — last_error выставлен run_async; идём дальше.
    }
    return false;
}

bool AppModel::start_next_in_network_queue() {
    if (network_session.in_flight) return false;
    if (network_queue.empty()) return false;
    if (!parametric_engine) parametric_engine = std::make_unique<ParametricEngine>();
    while (!network_queue.empty()) {
        NetworkQueueItem it = network_queue.front();
        network_queue.pop_front();
        if (it.index >= 0 && it.index < (int)network_session.configs.size()) {
            if (network_session.run_async(*parametric_engine, it.index)) return true;
        }
        // ok == false — last_error выставлен run_async; идём дальше.
    }
    return false;
}

bool AppModel::start_next_in_fastsync_queue() {
    if (fastsync_session.in_flight) return false;
    if (fastsync_queue.empty()) return false;
    if (!parametric_engine) parametric_engine = std::make_unique<ParametricEngine>();
    while (!fastsync_queue.empty()) {
        FastSyncQueueItem it = fastsync_queue.front();
        fastsync_queue.pop_front();
        if (it.index >= 0 && it.index < (int)fastsync_session.configs.size()) {
            if (fastsync_session.run_async(*parametric_engine, it.index)) return true;
        }
        // ok == false — last_error выставлен соответствующим run_async; идём дальше.
    }
    return false;
}