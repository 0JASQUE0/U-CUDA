#include "app_model.h"
#include "codegen.hpp"
#include "session_io.h"
#include <algorithm>
#include <cstdlib>
#include "sysparse.hpp"
#include <stdexcept>

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
        return parse_system_from_latex(latex_text, alpha, param_order, funcs);
    }

    // InputMode::Plain — обычный синтаксис.
    // Здесь пользователь вводит уравнения в простом виде; для простоты пока
    // используем тот же многострочный разбор, но без LaTeX-флага не получится:
    // parse_system_from_latex ждёт LaTeX. Для Plain нужен отдельный путь.
    // Пока трактуем Plain так же, как LaTeX-текст (часто работает: x*(r-z) и т.п.).
    if (plain_text.empty()) throw std::runtime_error("equations are empty");
    if (alpha.empty()) throw std::runtime_error("alphabet is empty");
    return parse_system_from_latex(plain_text, alpha, param_order, funcs);
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
    r.symmetry_s = symmetry_s;
    r.step_h = step_h;
    r.init_conditions = init_conditions;
    r.param_values = param_values;
    r.custom_schemes = custom_schemes;
    return r;
}

void AppModel::from_record(const SystemRecord& r) {
    name = r.name;
    note = r.note;
    if (r.mode == "Image") mode = InputMode::Image;
    else if (r.mode == "Plain") mode = InputMode::Plain;
    else mode = InputMode::Latex;
    latex_text = r.latex_text;
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
    symmetry_s = r.symmetry_s;
    step_h = r.step_h;
    init_conditions = r.init_conditions;
    param_values = r.param_values;
    custom_schemes = r.custom_schemes;
    loaded_name = r.name;          // запоминаем имя на диске
    // обновим списки символов (без падения, если система ещё неполна)
    refresh_symbols();
    // сразу перегенерируем код загруженной системы (если выбраны методы),
    // чтобы не показывать код от предыдущей системы.
    generated_code.clear();
    if (scheme_euler || scheme_cromer || scheme_midpoint || scheme_rk4 || scheme_dopri78 || scheme_cd)
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
    scheme_euler = scheme_cromer = scheme_midpoint = scheme_rk4 = scheme_dopri78 = scheme_cd = false;
    symmetry_s = "0.5";
    step_h.clear();
    init_conditions.clear();
    param_values.clear();
    custom_schemes.clear();
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

// ============================================================================
// Cross-analysis batch queue
// ============================================================================

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
        if (ok) return true;
        // ok == false (например, krs пуст / индекс плохой) — last_error
        // выставлен соответствующим run_async; идём дальше.
    }
    return false;
}

// ============================================================================
// Custom-tab pipeline queue drainer
// ============================================================================

namespace {
// Format a double for insertion into a text-based numeric input field. Enough
// precision to survive round-trip through parse_d (see analysis_session.cpp).
std::string fmt_num_for_input(double v) {
    char buf[64];
    // %.6g strips float→double round-trip noise (0.20000000298 → "0.2")
    // for values that came from a SliderFloat or heatmap-pixel snap. If a
    // caller ever needs more precision, promote this helper's format.
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
}

// Pin the "other axis" parameter (the one held fixed while the swept axis
// varies) into the per-config param_values. Only applies when the fixed axis
// sweeps a parameter — for var-sweeps, the "fix" applies to an initial
// condition, not a param, so we handle those separately below.
void pin_fixed_param(std::map<std::string,std::string>& pv,
                     const std::vector<std::string>& params,
                     int par_index, bool over_var, double value) {
    if (over_var) return;
    if (par_index < 0 || par_index >= (int)params.size()) return;
    pv[params[par_index]] = fmt_num_for_input(value);
}

// Третий вариант той же фиксации: закреплённая ось свипует ШАГ. Тогда «фиксация»
// — это подстановка h, а не значения параметра или НУ. Зовётся вместе с
// pin_fixed_param/pin_fixed_ic; ровно одна из трёх что-то делает, остальные
// выходят по своему флагу.
void pin_fixed_h(std::string& h_text, bool over_h, double value) {
    if (!over_h) return;
    if (value <= 0.0) return;             // шаг <= 0 бессмыслен, оставляем прежний
    h_text = fmt_num_for_input(value);
}

void pin_fixed_ic(std::map<std::string,std::string>& ic,
                  const std::vector<std::string>& vars,
                  int var_index, bool over_var, double value) {
    if (!over_var) return;
    if (var_index < 0 || var_index >= (int)vars.size()) return;
    ic[vars[var_index]] = fmt_num_for_input(value);
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
                pin_fixed_param(c.param_values, cs.params, swy.par_index, swy.over_var, shared.fix_y_value);
                pin_fixed_ic   (c.initial_conditions, cs.vars,  swy.var_index, swy.over_var, shared.fix_y_value);
                pin_fixed_h    (c.h_text, swy.over_h, shared.fix_y_value);
                ok = cs.bif_session.run_async(*parametric_engine, 1);
            }
            break;
        case K::Bif1D_Y:
            if (cs.bif_session.diagrams.size() > 2) {
                auto& c = cs.bif_session.diagrams[2];
                apply_shared_to_bif1d(shared, c, 1);
                EffectiveSweep swx = effective_sweep_x(shared);
                pin_fixed_param(c.param_values, cs.params, swx.par_index, swx.over_var, shared.fix_x_value);
                pin_fixed_ic   (c.initial_conditions, cs.vars,  swx.var_index, swx.over_var, shared.fix_x_value);
                pin_fixed_h    (c.h_text, swx.over_h, shared.fix_x_value);
                ok = cs.bif_session.run_async(*parametric_engine, 2);
            }
            break;
        case K::LLE1D_X:
            if (cs.lle_session.curves.size() > 1) {
                auto& c = cs.lle_session.curves[1];
                apply_shared_to_lle1d(shared, c, 0);
                EffectiveSweep swy = effective_sweep_y(shared);
                pin_fixed_param(c.param_values, cs.params, swy.par_index, swy.over_var, shared.fix_y_value);
                pin_fixed_ic   (c.initial_conditions, cs.vars,  swy.var_index, swy.over_var, shared.fix_y_value);
                pin_fixed_h    (c.h_text, swy.over_h, shared.fix_y_value);
                ok = cs.lle_session.run_async(*parametric_engine, 1);
            }
            break;
        case K::LLE1D_Y:
            if (cs.lle_session.curves.size() > 2) {
                auto& c = cs.lle_session.curves[2];
                apply_shared_to_lle1d(shared, c, 1);
                EffectiveSweep swx = effective_sweep_x(shared);
                pin_fixed_param(c.param_values, cs.params, swx.par_index, swx.over_var, shared.fix_x_value);
                pin_fixed_ic   (c.initial_conditions, cs.vars,  swx.var_index, swx.over_var, shared.fix_x_value);
                pin_fixed_h    (c.h_text, swx.over_h, shared.fix_x_value);
                ok = cs.lle_session.run_async(*parametric_engine, 2);
            }
            break;
        case K::LS1D_X:
            if (cs.ls_session.curves.size() > 1) {
                auto& c = cs.ls_session.curves[1];
                apply_shared_to_ls1d(shared, c, 0);
                EffectiveSweep swy = effective_sweep_y(shared);
                pin_fixed_param(c.param_values, cs.params, swy.par_index, swy.over_var, shared.fix_y_value);
                pin_fixed_ic   (c.initial_conditions, cs.vars,  swy.var_index, swy.over_var, shared.fix_y_value);
                pin_fixed_h    (c.h_text, swy.over_h, shared.fix_y_value);
                ok = cs.ls_session.run_async(*parametric_engine, 1);
            }
            break;
        case K::LS1D_Y:
            if (cs.ls_session.curves.size() > 2) {
                auto& c = cs.ls_session.curves[2];
                apply_shared_to_ls1d(shared, c, 1);
                EffectiveSweep swx = effective_sweep_x(shared);
                pin_fixed_param(c.param_values, cs.params, swx.par_index, swx.over_var, shared.fix_x_value);
                pin_fixed_ic   (c.initial_conditions, cs.vars,  swx.var_index, swx.over_var, shared.fix_x_value);
                pin_fixed_h    (c.h_text, swx.over_h, shared.fix_x_value);
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