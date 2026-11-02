#include "ucuda/application/app_model.h"
#include "ucuda/equations/codegen.hpp"
#include "ucuda/domain/num_parse.h"   // fmt_num_shortest — round-trip формат для пинов узлов
#include "ucuda/application/session_persistence.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "ucuda/equations/sysparse.hpp"
#include "ucuda/equations/dae_ir.h"       // build_system для ДАУ
#include "ucuda/equations/dae_devices.h"  // dae_dc_operating_point
#include "ucuda/numerics/kicad_import.h"
#include "ucuda/numerics/spice_parse.h"  // import_circuit
#include <filesystem>
#include <stdexcept>

// Строит System из текущего режима ввода и алфавита.

System SystemEditor::build_system() const {
    // ДАУ: X = [x; z], rhs = [f; g], dae_nx = число x (dae_ir.h). Схемы — только ДАУ.
    // dae_as_ode: z подставлены в f — ОДУ по x для всех схем и вкладок (dae_reduce_to_ode), а сами
    // z остаются в состоянии выходами (System::out_exprs): шаг пересчитывает их по новому x, и
    // их видят выход БД, метрики и оси графиков, как обычные переменные.
    if (is_dae) {
        // Вспомогательные функции — те же, что у ОДУ: подставляются в уравнения при разборе.
        FuncDefs funcs;
        if (use_aux_funcs && !func_defs_text.empty()) funcs = parse_func_defs(func_defs_text);
        System s = parse_dae(plain_text, vars_text, alg_text, params_text, dae_index_text, funcs).as_codegen_system();
        if (!dae_as_ode) return s;
        if (!dae_index1_structural(s)) {
            // Уровень C (индекс 2-3): связи заменяются продифференцированными со стабилизацией
            // (d/dt + alpha)^m g = 0 — дальше это индекс 1 для уровней A/B.
            double alpha = 0;
            if (!parse_num_checked(dae_stab_alpha, alpha))
                throw std::runtime_error("solve as ODE (level C): stabilization alpha is not a number");
            System s1;
            std::string why;
            if (!dae_stabilized_index1(s, alpha, s1, &why)) throw std::runtime_error("solve as ODE (level C): " + why);
            s = s1;
        }
        System ode;
        std::vector<std::string> zex;
        std::string why;
        if (!dae_reduce_to_ode(s, ode, &zex, &why)) {
            // Уровень B: связь не разрешается явно — z Ньютоном в каждом вычислении f (System::zb_g).
            // Нужен индекс 1: в каждой связи есть z (иначе dg/dz вырожден).
            ode = System{};
            ode.vars = s.vars;
            ode.params = s.params;
            ode.latex = s.latex;
            ode.cd_fp_iters = s.cd_fp_iters;
            ode.rhs.assign(s.rhs.begin(), s.rhs.begin() + s.dae_nx);
            ode.zb_g.assign(s.rhs.begin() + s.dae_nx, s.rhs.end());
            if (!dae_index1_structural(s))
                throw std::runtime_error("solve as ODE: " + why);
            return ode;
        }
        ode.vars.insert(ode.vars.end(), s.vars.begin() + s.dae_nx, s.vars.end());
        ode.out_exprs = zex;
        return ode;
    }
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
        return s;
    }

    // InputMode::Plain — обычный синтаксис. Пока разбираем его тем же многострочным путём, что и
    // LaTeX: parse_system_from_latex ждёт LaTeX, и для Plain нужен отдельный парсер. На практике
    // часто срабатывает (x*(r-z) и т.п.).
    if (plain_text.empty()) throw std::runtime_error("equations are empty");
    if (alpha.empty()) throw std::runtime_error("alphabet is empty");
    System s = parse_system_from_latex(plain_text, alpha, param_order, funcs, is_map);
    return s;
}

const SystemEditor::RhsPreview& SystemEditor::rhs_preview() const {
    // Ключ — всё, что читает build_system(), кроме настроек Ньютона: на
    // правые части они не влияют. Разделитель \x1f в тексте не встречается.
    const bool latex = (mode == InputMode::Image || mode == InputMode::Latex);
    std::string key;
    key += latex ? 'L' : 'P';
    key += is_map ? 'M' : 'O';
    key += (char)('0' + (int)param_order);
    key += use_aux_funcs ? 'F' : '-';
    for (const std::string* s : { latex ? &latex_text : &plain_text, &alphabet_text,
                                  &vars_text, &params_text, &func_defs_text }) {
        key += '\x1f';
        key += *s;
    }
    if (key == rhs_preview_.key) return rhs_preview_;

    rhs_preview_ = RhsPreview{};
    rhs_preview_.key = key;
    try {
        const System sys = build_system();
        const std::vector<std::string> rhs = codegen_rhs(sys);

        // Легенда: какое имя в какую ячейку ушло. Порядок a[] зависит от
        // param_order и от разбора, поэтому показывать его стоит всегда.
        std::string out;
        auto legend = [&out](const char* arr, int base, const std::vector<std::string>& names) {
            for (size_t i = 0; i < names.size(); ++i) {
                out += (i % 6 == 0) ? (i ? "\n//   " : "// ") : "   ";
                out += arr;
                out += "[" + std::to_string(base + (int)i) + "] = " + names[i];
            }
            if (!names.empty()) out += "\n";
        };
        legend("X", 0, sys.vars);
        legend("a", 1, sys.params);

        for (size_t i = 0; i < rhs.size(); ++i) {
            const std::string idx = std::to_string(i);
            out += sys.is_map ? "X_next[" + idx + "] = " : "dX[" + idx + "] = ";
            out += rhs[i];
            out += ";\n";
        }
        rhs_preview_.text = out;
    }
    catch (const std::exception& e) {
        rhs_preview_.error = e.what();
    }
    return rhs_preview_;
}

// Парсит систему, обновляет списки символов и синхронизирует словари значений.
bool SystemEditor::refresh_symbols() {
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

    // ДАУ: переменные — x, затем z (как в X шага), параметры — как в params_text.
    if (is_dae) {
        try {
            const System sys = build_system();
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

bool SystemEditor::import_circuit(std::string* report, std::string* err) {
    if (circuit_source.empty()) { if (err) *err = "no circuit file"; return false; }
    std::string net, log;
    if (!circuit_source_load(circuit_source, net, &log, err)) return false;
    SpiceImportOptions o;
    o.base_dir = std::filesystem::u8path(circuit_source).parent_path().u8string();
    o.ideal_opamps = circuit_ideal_opamps;
    o.no_junction_caps = circuit_no_jcaps;
    o.ideal_diodes = circuit_ideal_diodes;
    o.ideal_bjts = circuit_ideal_bjts;
    SpiceImportResult r;
    if (!spice_import(net, o, r, err)) return false;
    // Схема — источник истины: уравнения, номиналы и НУ берутся из неё целиком.
    is_dae = true;
    mode = InputMode::Plain;
    plain_text = r.plain_text;
    vars_text = r.vars_text;
    alg_text = r.alg_text;
    params_text = r.params_text;
    param_order = ParamOrder::AsInAlphabet;
    dae_index_text.clear();
    use_aux_funcs = false;
    init_conditions = r.init_conditions;
    param_values = r.param_values;
    {
        // Разложенное пользователем остаётся; новые параметры — в группы импорта.
        std::vector<std::string> names;
        for (const auto& pg : r.param_groups) names.push_back(pg.first);
        pg_merge_auto(param_groups, names, r.param_groups);
        names.clear();
        for (const auto& pg : r.ic_groups) names.push_back(pg.first);
        pg_merge_auto(ic_groups, names, r.ic_groups);
        for (const auto& kv : r.field_labels) field_labels.emplace(kv.first, kv.second);
    }
    if (!r.step_h.empty()) step_h = r.step_h;
    circuit_netlist = net;
    if (note.empty() && !r.title.empty()) note = r.title;
    // Без UIC и без НУ SPICE стартует переходный процесс из рабочей точки — так же и здесь.
    std::string op;
    if (r.uic) op = "initial values: as given (UIC)";
    else if (!r.op_plain_text.empty()) {
        // .ic без UIC, как в SPICE: рабочая точка с удержанными узлами .ic, затем узлы отпускаются (z при
        // старте расчёта согласуются с x уже по исходным уравнениям).
        try {
            const DaeSystem d = parse_dae(r.op_plain_text, vars_text, alg_text, params_text, dae_index_text, FuncDefs{});
            const System s = d.as_codegen_system();
            std::vector<double> a = { 0.5 };
            for (const std::string& p : d.params) {
                auto it = param_values.find(p);
                a.push_back(it == param_values.end() ? 0.0 : parse_num(it->second, 0.0));
            }
            std::vector<double> X;
            for (const std::string& v : s.vars) {
                auto it = init_conditions.find(v);
                X.push_back(it == init_conditions.end() ? 0.0 : parse_num(it->second, 0.0));
            }
            int it = 0;
            std::string why;
            if (dae_dc_operating_point(s, a.data(), (int)a.size(), 0.0, X.data(), &it, &why)) {
                for (size_t i = 0; i < s.vars.size(); ++i) init_conditions[s.vars[i]] = fmt_num_shortest(X[i]);
                op = "initial values: DC operating point with the .ic nodes held (" + std::to_string(it)
                   + " Newton iterations), then released";
            } else op = "initial values: .ic as given; DC operating point with the .ic nodes held failed: " + why;
        } catch (const std::exception& e) {
            op = std::string("initial values: .ic as given; DC operating point: ") + e.what();
        }
    }
    else if (r.has_ic) op = "initial values: IC= / .ic (no DC operating point)";
    else {
        std::string orep, oerr;
        op = dae_operating_point(&orep, &oerr) ? "initial values: " + orep
                                               : "initial values: zero; DC operating point failed: " + oerr;
    }
    if (report) {
        std::string s = r.summary;
        if (!r.t_stop.empty()) s += "\n.tran: h = " + r.step_h + " s, T = " + r.t_stop + " s";
        s += "\n" + op;
        for (const std::string& n : r.notes) s += "\n" + n;
        s += "\nnodes:";
        for (const auto& kv : r.node_vars) s += " " + kv.first + "=" + kv.second;
        *report = s;
    }
    return true;
}

bool SystemEditor::dae_operating_point(std::string* report, std::string* err) {
    try {
        FuncDefs funcs;
        if (use_aux_funcs && !func_defs_text.empty()) funcs = parse_func_defs(func_defs_text);
        const DaeSystem d = parse_dae(plain_text, vars_text, alg_text, params_text, dae_index_text, funcs);
        const System s = d.as_codegen_system();
        std::vector<double> a = { 0.5 };
        for (const std::string& p : d.params) {
            auto it = param_values.find(p);
            a.push_back(it == param_values.end() ? 0.0 : parse_num(it->second, 0.0));
        }
        std::vector<double> X;
        for (const std::string& v : s.vars) {
            auto it = init_conditions.find(v);
            X.push_back(it == init_conditions.end() ? 0.0 : parse_num(it->second, 0.0));
        }
        int it = 0;
        std::string why;
        if (!dae_dc_operating_point(s, a.data(), (int)a.size(), 0.0, X.data(), &it, &why)) {
            if (err) *err = why;
            return false;
        }
        for (size_t i = 0; i < s.vars.size(); ++i) init_conditions[s.vars[i]] = fmt_num_shortest(X[i]);
        if (report) *report = "DC operating point (" + std::to_string(it) + " Newton iterations)";
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

SystemRecord SystemEditor::to_record() const {
    SystemRecord r;
    r.name = name;
    r.note = note;
    r.is_map = is_map;
    r.is_dae = is_dae;
    r.alg_text = alg_text;
    r.dae_index_text = dae_index_text;
    r.dae_as_ode = dae_as_ode;
    r.dae_stab_alpha = dae_stab_alpha;
    r.circuit_source = circuit_source;
    r.circuit_netlist = circuit_netlist;
    r.circuit_ideal_opamps = circuit_ideal_opamps;
    r.circuit_no_jcaps = circuit_no_jcaps;
    r.circuit_ideal_diodes = circuit_ideal_diodes;
    r.circuit_ideal_bjts = circuit_ideal_bjts;
    r.param_groups = param_groups;
    r.ic_groups = ic_groups;
    r.field_labels = field_labels;
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
    r.scheme_dopri78_legacy = scheme_dopri78_legacy;
    r.scheme_rk45 = scheme_rk45;
    r.scheme_dop853 = scheme_dop853;
    r.scheme_cd = scheme_cd;
    r.scheme_cdfast = scheme_cdfast;
    r.scheme_cdfp = scheme_cdfp;
    r.scheme_cdfastfp = scheme_cdfastfp;
    r.scheme_ccd = scheme_ccd;
    r.scheme_ccd4 = scheme_ccd4;
    r.scheme_ccd4s3 = scheme_ccd4s3;
    r.scheme_ccd4s4 = scheme_ccd4s4;
    r.scheme_ccd4ss01 = scheme_ccd4ss01;
    r.scheme_ccd4ss10 = scheme_ccd4ss10;
    r.scheme_cd10 = scheme_cd10;
    r.scheme_ccd10 = scheme_ccd10;
    r.scheme_ccd4_10 = scheme_ccd4_10;
    r.scheme_ccd4s3_10 = scheme_ccd4s3_10;
    r.scheme_ccd4s4_10 = scheme_ccd4s4_10;
    r.scheme_ieuler = scheme_ieuler;
    r.scheme_imidpoint = scheme_imidpoint;
    r.scheme_semp = scheme_semp;
    r.scheme_simp = scheme_simp;
    r.scheme_dmethod = scheme_dmethod;
    r.scheme_cieuler = scheme_cieuler;
    r.scheme_gbs = scheme_gbs;
    r.scheme_gbs24 = scheme_gbs24;
    r.scheme_gbs246 = scheme_gbs246;
    r.scheme_gbs2468 = scheme_gbs2468;
    r.scheme_gbs246810 = scheme_gbs246810;
    r.scheme_gbs24681012 = scheme_gbs24681012;
    r.newton_full = newton_full;
    r.newton_tol = newton_tol;
    r.newton_max_iters = newton_max_iters;
    r.cd_fp_iters = cd_fp_iters;
    r.symmetry_s = symmetry_s;
    r.step_h = step_h;
    r.init_conditions = init_conditions;
    r.param_values = param_values;
    r.custom_schemes = custom_schemes;
    r.wrapper_schemes = wrapper_schemes;
    r.disabled_wrappers = disabled_wrappers;
    r.dae_schemes_off = dae_schemes_off;
    return r;
}

void SystemEditor::from_record(const SystemRecord& r) {
    name = r.name;
    note = r.note;
    is_map = r.is_map;
    is_dae = r.is_dae;
    alg_text = r.alg_text;
    dae_index_text = r.dae_index_text;
    dae_as_ode = r.dae_as_ode;
    dae_stab_alpha = r.dae_stab_alpha;
    circuit_source = r.circuit_source;
    circuit_netlist = r.circuit_netlist;
    circuit_ideal_opamps = r.circuit_ideal_opamps;
    circuit_no_jcaps = r.circuit_no_jcaps;
    circuit_ideal_diodes = r.circuit_ideal_diodes;
    circuit_ideal_bjts = r.circuit_ideal_bjts;
    param_groups = r.param_groups;
    ic_groups = r.ic_groups;
    field_labels = r.field_labels;
    pg_ui = ParamGroupsUi{};
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
    scheme_dopri78_legacy = r.scheme_dopri78_legacy;
    scheme_rk45 = r.scheme_rk45;
    scheme_dop853 = r.scheme_dop853;
    scheme_cd = r.scheme_cd;
    scheme_cdfast = r.scheme_cdfast;
    scheme_cdfp = r.scheme_cdfp;
    scheme_cdfastfp = r.scheme_cdfastfp;
    scheme_ccd = r.scheme_ccd;
    scheme_ccd4 = r.scheme_ccd4;
    scheme_ccd4s3 = r.scheme_ccd4s3;
    scheme_ccd4s4 = r.scheme_ccd4s4;
    scheme_ccd4ss01 = r.scheme_ccd4ss01;
    scheme_ccd4ss10 = r.scheme_ccd4ss10;
    scheme_cd10 = r.scheme_cd10;
    scheme_ccd10 = r.scheme_ccd10;
    scheme_ccd4_10 = r.scheme_ccd4_10;
    scheme_ccd4s3_10 = r.scheme_ccd4s3_10;
    scheme_ccd4s4_10 = r.scheme_ccd4s4_10;
    scheme_ieuler = r.scheme_ieuler;
    scheme_imidpoint = r.scheme_imidpoint;
    scheme_semp = r.scheme_semp;
    scheme_simp = r.scheme_simp;
    scheme_dmethod = r.scheme_dmethod;
    scheme_cieuler = r.scheme_cieuler;
    scheme_gbs = r.scheme_gbs;
    scheme_gbs24 = r.scheme_gbs24;
    scheme_gbs246 = r.scheme_gbs246;
    scheme_gbs2468 = r.scheme_gbs2468;
    scheme_gbs246810 = r.scheme_gbs246810;
    scheme_gbs24681012 = r.scheme_gbs24681012;
    newton_full = r.newton_full;
    newton_tol = r.newton_tol;
    newton_max_iters = r.newton_max_iters;
    cd_fp_iters = r.cd_fp_iters;
    symmetry_s = r.symmetry_s;
    step_h = r.step_h;
    init_conditions = r.init_conditions;
    param_values = r.param_values;
    custom_schemes = r.custom_schemes;
    wrapper_schemes = r.wrapper_schemes;
    sort_wrapper_schemes(wrapper_schemes);   // старые библиотеки — в порядке добавления
    disabled_wrappers = r.disabled_wrappers;
    dae_schemes_off = r.dae_schemes_off;
    loaded_name = r.name;          // запоминаем имя на диске
    // обновим списки символов (без падения, если система ещё неполна)
    refresh_symbols();
    // сразу перегенерируем код загруженной системы (если выбраны методы),
    // чтобы не показывать код от предыдущей системы.
    generated_code.clear();
    // A map has no scheme_* flag set — its single body is always generated.
    if (is_map
        || scheme_euler || scheme_cromer || scheme_midpoint || scheme_rk4 || scheme_dopri78 || scheme_dopri78_legacy
        || scheme_rk45 || scheme_dop853
        || scheme_cd || scheme_cdfast
        || scheme_cdfp || scheme_cdfastfp || scheme_ccd || scheme_ccd4|| scheme_ccd4s3 || scheme_ccd4s4 || scheme_ccd4ss01
        || scheme_ccd4ss10
        || scheme_cd10 || scheme_ccd10 || scheme_ccd4_10 || scheme_ccd4s3_10 || scheme_ccd4s4_10
        || scheme_ieuler || scheme_imidpoint
        || scheme_semp || scheme_simp || scheme_dmethod || scheme_cieuler
        || scheme_gbs || scheme_gbs24 || scheme_gbs246 || scheme_gbs2468 || scheme_gbs246810 || scheme_gbs24681012
        || !custom_schemes.empty() || !wrapper_schemes.empty())   // их код тоже в превью
        generate();
    circuit_layout_defaults();
}

std::vector<std::string> SystemEditor::ic_fields(const std::vector<std::string>& vars, int* hidden) const {
    if (hidden) *hidden = 0;
    if (show_alg_ic || !is_dae || alg_text.empty()) return vars;
    std::set<std::string> alg;
    std::string cur;
    for (size_t i = 0; i <= alg_text.size(); ++i) {
        const char c = i < alg_text.size() ? alg_text[i] : ',';
        if (c == ',') { if (!cur.empty()) alg.insert(cur); cur.clear(); }
        else if (c != ' ' && c != '\t' && c != '\r' && c != '\n') cur += c;
    }
    std::vector<std::string> out;
    for (const std::string& v : vars) {
        if (alg.count(v)) { if (hidden) ++*hidden; }
        else out.push_back(v);
    }
    return out;
}

void SystemEditor::circuit_layout_defaults() {
    if (!is_dae || circuit_netlist.empty() || !ic_groups.empty()) return;
    auto split = [](const std::string& s) {
        std::vector<std::string> v;
        std::string cur;
        for (size_t i = 0; i <= s.size(); ++i) {
            const char c = i < s.size() ? s[i] : ',';
            if (c == ',') { if (!cur.empty()) v.push_back(cur); cur.clear(); }
            else if (c != ' ' && c != '\t' && c != '\r' && c != '\n') cur += c;
        }
        return v;
    };
    const std::vector<std::string> xs = split(vars_text), zs = split(alg_text);
    std::vector<std::string> names = xs;
    names.insert(names.end(), zs.begin(), zs.end());
    SpiceImportOptions o;
    if (!circuit_source.empty()) o.base_dir = std::filesystem::u8path(circuit_source).parent_path().u8string();
    o.ideal_opamps = circuit_ideal_opamps;
    o.no_junction_caps = circuit_no_jcaps;
    o.ideal_diodes = circuit_ideal_diodes;
    o.ideal_bjts = circuit_ideal_bjts;
    SpiceImportResult r;
    std::string err;
    try {
        if (spice_import(circuit_netlist, o, r, &err)) {
            pg_merge_auto(ic_groups, names, r.ic_groups);
            for (const auto& kv : r.field_labels) field_labels.emplace(kv.first, kv.second);
        }
    } catch (const std::exception&) {}
    // Имена, которых разбор не дал (запись старше правил имён), — по роли.
    std::set<std::string> placed;
    for (const ParamGroup& g : ic_groups) placed.insert(g.params.begin(), g.params.end());
    for (const std::string& x : xs) if (!placed.count(x)) pg_get(ic_groups, "States").params.push_back(x);
    for (const std::string& z : zs)
        if (!placed.count(z))
            pg_get(ic_groups, z.rfind("v_", 0) == 0 ? "Algebraic (computed)/Node potentials"
                                                    : "Algebraic (computed)/Branch currents").params.push_back(z);
}

void SystemEditor::clear() {
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
    is_dae = false;
    alg_text.clear();
    dae_index_text.clear();
    dae_as_ode = false;
    dae_stab_alpha = "10";
    circuit_source.clear();
    circuit_netlist.clear();
    circuit_ideal_opamps = false;
    circuit_no_jcaps = circuit_ideal_diodes = circuit_ideal_bjts = false;
    param_groups.clear();
    ic_groups.clear();
    field_labels.clear();
    pg_ui = ParamGroupsUi{};
    scheme_euler = scheme_cromer = scheme_midpoint = scheme_rk4 = scheme_dopri78 = scheme_dopri78_legacy
        = scheme_rk45 = scheme_dop853
        = scheme_cd = scheme_cdfast
        = scheme_cdfp = scheme_cdfastfp = scheme_ccd = scheme_ccd4= scheme_ccd4s3 = scheme_ccd4s4 = scheme_ccd4ss01
        = scheme_ccd4ss10
        = scheme_cd10 = scheme_ccd10 = scheme_ccd4_10 = scheme_ccd4s3_10 = scheme_ccd4s4_10
        = scheme_ieuler = scheme_imidpoint
        = scheme_semp = scheme_simp = scheme_dmethod = scheme_cieuler
        = scheme_gbs = scheme_gbs24 = scheme_gbs246 = scheme_gbs2468 = scheme_gbs246810 = scheme_gbs24681012 = false;
    symmetry_s = "0.5";
    newton_full = false;
    newton_tol = "1e-10";
    newton_max_iters = "8";
    cd_fp_iters = "4";
    step_h.clear();
    init_conditions.clear();
    param_values.clear();
    custom_schemes.clear();
    wrapper_schemes.clear();
    disabled_wrappers.clear();
    dae_schemes_off.clear();
    known_vars.clear();
    known_params.clear();
    generated_code.clear();
    error_message.clear();
}
