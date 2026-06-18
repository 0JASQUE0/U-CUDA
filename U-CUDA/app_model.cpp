#include "app_model.h"
#include "codegen.hpp"
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
        sync(param_min, known_params);
        sync(param_max, known_params);
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
    r.step_h = step_h;
    r.init_conditions = init_conditions;
    r.param_values = param_values;
    r.param_min = param_min;
    r.param_max = param_max;
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
    step_h = r.step_h;
    init_conditions = r.init_conditions;
    param_values = r.param_values;
    param_min = r.param_min;
    param_max = r.param_max;
    custom_schemes = r.custom_schemes;
    loaded_name = r.name;          // запоминаем имя на диске
    // обновим списки символов (без падения, если система ещё неполна)
    refresh_symbols();
    // сразу перегенерируем код загруженной системы (если выбраны методы),
    // чтобы не показывать код от предыдущей системы.
    generated_code.clear();
    if (scheme_euler || scheme_cromer || scheme_midpoint || scheme_rk4)
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
    scheme_euler = scheme_cromer = scheme_midpoint = scheme_rk4 = false;
    step_h.clear();
    init_conditions.clear();
    param_values.clear();
    param_min.clear();
    param_max.clear();
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
    parametric_session.load_from_record(r, known_vars, known_params);
    try {
        parametric_session.sys = build_system();
        parametric_session.regenerate_krs();
    }
    catch (...) {
        // система неполна — Run покажет ошибку
    }
    parametric_session.loaded_system_name = name;
    return true;
}