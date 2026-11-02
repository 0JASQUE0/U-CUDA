#include "gui_common.h"
#include "gui_inputs.h"
#include "gui_fields.h"
#include "tab_library.h"
#include "tab_order.h"
#include "tab_system.h"
using namespace ucuda::ui_detail;


namespace ucuda::ui_detail {
LibTab lib_tab_of(bool is_dae, bool is_map) {
    return is_dae ? LibTab::Dae : is_map ? LibTab::Map : LibTab::Ode;
}
}


namespace ucuda::ui_detail {
void draw_library_editor(AppModel& model, SystemLibrary& lib,
                                const GuiCallbacks& cb) {
    SystemEditor& buf = *model.library_edit_buffer;
    const bool add_new = (model.library_edit_mode == AppModel::LibraryEditMode::AddNew);
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
                       add_new ? "New system" : "Editing");
    if (!add_new) {
        ImGui::SameLine();
        ImGui::TextDisabled("(original: %s)", model.edit_original_name.c_str());
    }

    ImGui::Spacing();
    ImGui::Text("Name:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-1);
    InputTextStr("##editor_name", buf.name);
    ImGui::Text("Note:");
    InputTextMultilineStr("##editor_note", buf.note, ImVec2(-1, 60));

    if (!model.edit_error.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f),
                           "%s", model.edit_error.c_str());
    }

    ImGui::Separator();

    // Редактор рисует System tab по буферу-черновику, а маска схем глобальная
    // и живёт в основной модели — иначе в редакторе показались бы скрытые схемы.
    buf.hidden_schemes = model.hidden_schemes;
    buf.hidden_dae_methods = model.hidden_dae_methods;
    if (buf.is_dae) {
        ImGui::TextWrapped("x' = f(x, z, t),  0 = g(x, z, t). One equation per line: "
                           "x' = ... for each differential variable, 0 = ... for each "
                           "algebraic one. min(a, b) / max(a, b) are available.");
        // Источник — схема: SPICE-netlist или .kicad_sch (через kicad-cli). Импорт перезаписывает
        // уравнения, переменные, параметры (номиналы) и НУ ниже.
        {
            AppModel::DaeEditorState& cst = model.dae_editor;
            ImGui::Text("Circuit:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(-ImGui::GetFontSize() * 15.0f);
            InputTextStr("##dae_circuit", buf.circuit_source);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("SPICE netlist (.cir, .net, .sp) or KiCad schematic (.kicad_sch, needs kicad-cli\n"
                                  "from KiCad 7+). Empty: the equations below are typed by hand.");
            ImGui::SameLine();
            if (ImGui::Button("Open...##dae_circuit") && cb.pick_open_file_circuit) {
                const std::string p = cb.pick_open_file_circuit();
                if (!p.empty()) { buf.circuit_source = p; cst.circuit_import_now = true; }
            }
            ImGui::SameLine();
            ImGui::BeginDisabled(buf.circuit_source.empty());
            if (ImGui::Button("Re-read##dae_circuit")) cst.circuit_import_now = true;
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::Checkbox("Watch##dae_circuit", &cst.circuit_watch);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Re-import when the file is saved (checked once a second).");
            // ОУ: как есть (их .subckt из netlist'а) или идеальные; копия на идеальных — отдельной записью.
            if (ImGui::Checkbox("Ideal op-amps##dae_circuit", &buf.circuit_ideal_opamps) && !buf.circuit_source.empty())
                cst.circuit_import_now = true;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Off: op-amps (TL072, LM741, ...) are imported as is - their .subckt models\n"
                                  "from the netlist, element by element. On: each is an ideal op-amp (nullor).\n"
                                  "An op-amp without a .subckt in the netlist is ideal either way.");
            if (!buf.circuit_source.empty() && !buf.circuit_ideal_opamps) {
                ImGui::SameLine();
                if (ImGui::Button("Save ideal copy##dae_circuit")) {
                    AppModel copy([](const std::vector<unsigned char>&) { return std::string{}; });
                    copy.from_record(buf.to_record());
                    copy.circuit_ideal_opamps = true;
                    copy.name = (buf.name.empty() ? std::string("Circuit") : buf.name) + " (ideal op-amps)";
                    std::string rep, e;
                    if (copy.import_circuit(&rep, &e)) {
                        try {
                            const std::string saved = lib.save(copy.to_record());
                            invalidate_library_cache();
                            cst.circuit_report = "saved the copy on ideal op-amps as '" + saved + "'\n" + rep;
                            cst.circuit_error.clear();
                        } catch (const std::exception& ex) { cst.circuit_error = std::string("Save ideal copy: ") + ex.what(); }
                    } else cst.circuit_error = "Ideal copy: " + e;
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Save the same circuit with ideal op-amps as a new library system\n"
                                      "(name + \" (ideal op-amps)\"), next to this one with the real models.");
            }
            // Упрощения приборов (облегчённая, нежёсткая модель): каждая галочка переимпортирует схему.
            {
                bool re = false;
                re |= ImGui::Checkbox("No junction capacitances##dae_circuit", &buf.circuit_no_jcaps);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Drop the barrier capacitances of diodes, BJTs, JFETs and MOSFETs. Each pF junction\n"
                                      "capacitance is a state variable with a ns time constant: without them the model is\n"
                                      "far less stiff (larger steps, fixed step possible), the DC laws are unchanged.");
                ImGui::SameLine();
                re |= ImGui::Checkbox("Ideal diodes / LEDs##dae_circuit", &buf.circuit_ideal_diodes);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Diode = threshold Vf + resistance Ron (the tangent of the exponential at 10 mA,\n"
                                      "RS included), the corner smoothed over 10 mV. Parameters <model>_vf, <model>_ron.");
                ImGui::SameLine();
                re |= ImGui::Checkbox("Ideal BJTs##dae_circuit", &buf.circuit_ideal_bjts);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Ebers-Moll on two such junctions (B-E, B-C) with BF, BR: saturation and cut-off\n"
                                      "stay, no Early effect and no RB / RC / RE (no internal nodes).");
                if (re && !buf.circuit_source.empty()) cst.circuit_import_now = true;
                const bool any = buf.circuit_no_jcaps || buf.circuit_ideal_diodes || buf.circuit_ideal_bjts;
                if (!buf.circuit_source.empty() && !any) {
                    ImGui::SameLine();
                    if (ImGui::Button("Save simplified copy##dae_circuit")) {
                        AppModel copy([](const std::vector<unsigned char>&) { return std::string{}; });
                        copy.from_record(buf.to_record());
                        copy.circuit_no_jcaps = copy.circuit_ideal_diodes = copy.circuit_ideal_bjts = true;
                        copy.name = (buf.name.empty() ? std::string("Circuit") : buf.name) + " (simplified)";
                        std::string rep, e;
                        if (copy.import_circuit(&rep, &e)) {
                            try {
                                const std::string saved = lib.save(copy.to_record());
                                invalidate_library_cache();
                                cst.circuit_report = "saved the simplified copy as '" + saved + "'\n" + rep;
                                cst.circuit_error.clear();
                            } catch (const std::exception& ex) { cst.circuit_error = std::string("Save simplified copy: ") + ex.what(); }
                        } else cst.circuit_error = "Simplified copy: " + e;
                    }
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Save the same circuit with all three simplifications as a new library system\n"
                                          "(name + \" (simplified)\"), next to this one with the full device models.");
                }
            }
            if (!buf.circuit_source.empty() && cst.circuit_watch) {
                const double now = ImGui::GetTime();
                if (now - cst.circuit_poll_t > 1.0) {
                    cst.circuit_poll_t = now;
                    const long long mt = circuit_file_mtime(buf.circuit_source);
                    if (cst.circuit_path != buf.circuit_source) {   // новый путь: только запомнить время
                        cst.circuit_path = buf.circuit_source;
                        cst.circuit_mtime = mt;
                    } else if (mt >= 0 && mt != cst.circuit_mtime) cst.circuit_import_now = true;
                }
            }
            if (cst.circuit_import_now && !buf.circuit_source.empty()) {
                cst.circuit_import_now = false;
                std::string rep, e;
                if (buf.import_circuit(&rep, &e)) { cst.circuit_report = rep; cst.circuit_error.clear(); }
                else cst.circuit_error = "Import failed: " + e;
                cst.circuit_path = buf.circuit_source;
                cst.circuit_mtime = circuit_file_mtime(buf.circuit_source);
            }
            if (ImGui::Button("DC operating point##dae")) {
                std::string rep, e;
                if (buf.dae_operating_point(&rep, &e)) { cst.circuit_report = rep; cst.circuit_error.clear(); }
                else cst.circuit_error = "DC operating point: " + e;
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Initial values from x' = 0, g = 0 (for a circuit: capacitor currents and inductor\n"
                                  "voltages zero). Newton starts from the current initial values.");
            if (!cst.circuit_error.empty() || !cst.circuit_report.empty()) {
                if (!cst.circuit_error.empty())
                    ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", cst.circuit_error.c_str());
                if (!cst.circuit_report.empty() && ImGui::TreeNode("Report##dae_circuit")) {
                    ImGui::TextWrapped("%s", cst.circuit_report.c_str());
                    ImGui::TreePop();
                }
            }
        }
        ImGui::Text("Equations:");
        InputTextMultilineStr("##dae_eqs", buf.plain_text, ImVec2(-1, 160));
        // Вспомогательные функции — как у ОДУ: подставляются в уравнения при разборе.
        ImGui::Checkbox("Use auxiliary functions##dae", &buf.use_aux_funcs);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("One definition per line, e.g. G(x) = Goff + (Gon - Goff)*x.\n"
                              "A call is replaced by the body with the arguments substituted;\n"
                              "a function may call another one defined above or below.");
        if (buf.use_aux_funcs)
            InputTextMultilineStr("##dae_funcs", buf.func_defs_text, ImVec2(-1, 110));
        ImGui::Text("Differential variables (x):"); ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        InputTextStr("##dae_vars", buf.vars_text);
        ImGui::Text("Algebraic variables (z):"); ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        InputTextStr("##dae_alg", buf.alg_text);
        ImGui::Text("Parameters:"); ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        InputTextStr("##dae_params", buf.params_text);
        ImGui::Text("Variable indices:"); ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        InputTextStr("##dae_index", buf.dae_index_text);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Optional, e.g. \"x=1, u=2, lambda=3\" (as nind1/2/3 in RADAU5).\n"
                              "Unnamed: differential 1, algebraic 1 if structurally index 1.");

        // Разбор — только при смене ввода.
        AppModel::DaeEditorState& st = model.dae_editor;
        const std::string key = buf.plain_text + '\x1e' + buf.vars_text + '\x1e' + buf.alg_text
                              + '\x1e' + buf.params_text + '\x1e' + buf.dae_index_text
                              + '\x1e' + (buf.use_aux_funcs ? buf.func_defs_text : std::string())
                              + '\x1e' + fl_to_text(buf.field_labels);   // подписи — в превью уравнений
        if (st.key != key) {
            st.key = key;
            try {
                FuncDefs funcs;
                if (buf.use_aux_funcs && !buf.func_defs_text.empty()) funcs = parse_func_defs(buf.func_defs_text);
                const DaeSystem d = parse_dae(buf.plain_text, buf.vars_text, buf.alg_text,
                                              buf.params_text, buf.dae_index_text, funcs);
                st.ok = true;
                st.vars = d.diff;
                st.vars.insert(st.vars.end(), d.alg.begin(), d.alg.end());
                st.params = d.params;
                std::string m = std::to_string(d.diff.size()) + " differential, "
                              + std::to_string(d.alg.size()) + " algebraic";
                if (d.structural_index_hint == 1) m += "; structurally index 1";
                else if (d.structural_index_hint == 2) {
                    m += "; index >= 2 (no constraint contains";
                    for (const auto& z : d.unmatched_alg) m += " " + z;
                    m += ")";
                }
                if (d.uses_t) m += "; uses t";
                m += "\nindices:";
                for (size_t i = 0; i < d.n_vars(); ++i)
                    m += " " + st.vars[i] + "=" + (d.index[i] ? std::to_string(d.index[i]) : "?");
                st.message = m;
                // Превью: \dot{x} = f по дифференциальным, 0 = g по связям — тем же AST, что у кодогена.
                {
                    const System ps = d.as_codegen_system();
                    st.latex.clear();
                    st.latex_lines = 0;
                    std::vector<std::string> fnames;
                    for (const auto& kv : funcs) fnames.push_back(kv.first);
                    // Сами вспомогательные функции — первыми строками (в уравнениях они уже подставлены);
                    // тело может звать другие функции (iM = G(x)*v + iS(v)) — они печатаются вызовами.
                    for (const auto& kv : funcs) {
                        std::string args;
                        for (size_t k = 0; k < kv.second.args.size(); ++k)
                            args += (k ? ", " : "") + name_to_latex(kv.second.args[k]);
                        std::string body;
                        try { body = expr_to_latex(kv.second.body, false, fnames); }
                        catch (const std::exception&) { body = "?"; }
                        st.latex += "\\mathrm{" + kv.first + "}\\left(" + args + "\\right) = " + body + "\n";
                        ++st.latex_lines;
                    }
                    // Уравнения: с функциями — исходные строки (вызовы G(xm) видны как есть), без них —
                    // разобранные f и g. Строка, которая не прочлась, берётся из разбора (с подстановкой).
                    std::vector<std::string> raw_rows;
                    if (!fnames.empty()) {
                        std::istringstream in(buf.plain_text);
                        std::string row;
                        while (std::getline(in, row)) {
                            const size_t a = row.find_first_not_of(" \t\r");
                            if (a == std::string::npos || row[a] == '#' || row[a] == '%' || row.compare(a, 2, "//") == 0) continue;
                            raw_rows.push_back(row.substr(a));
                        }
                    }
                    const bool use_raw = !fnames.empty() && raw_rows.size() == ps.rhs.size();
                    for (size_t i = 0; i < ps.rhs.size(); ++i) {
                        const bool diff = (int)i < ps.dae_nx;
                        std::string ln;
                        std::string lhs = diff ? dot_latex(ps.vars[i], &buf.field_labels) + " = " : std::string("0 = ");
                        bool done = false;
                        if (use_raw) {
                            const size_t eq = raw_rows[i].find('=');
                            if (eq != std::string::npos) {
                                std::string l = raw_rows[i].substr(0, eq);
                                l.erase(std::remove_if(l.begin(), l.end(), [](char c) { return c == ' ' || c == '\t'; }), l.end());
                                try {
                                    ln = expr_to_latex(raw_rows[i].substr(eq + 1), false, fnames, &buf.field_labels);
                                    if (l != "0") {
                                        const size_t q = l.find_first_of("'(");
                                        lhs = dot_latex(l.substr(0, q), &buf.field_labels) + " = ";
                                    }
                                    done = true;
                                } catch (const std::exception&) {}
                            }
                        }
                        if (!done) {
                            try { ln = expr_to_latex(ps.rhs[i], ps.latex, {}, &buf.field_labels); }
                            catch (const std::exception&) { ln = "?"; }
                        }
                        st.latex += lhs + ln + "\n";
                        ++st.latex_lines;
                    }
                }
                // Уровень A: можно ли решать как ОДУ (z выражены из связей явно).
                System ode;
                std::vector<std::string> zex;
                std::string why;
                st.level_c = false;
                const System dsys = d.as_codegen_system();
                st.reducible = dae_reduce_to_ode(dsys, ode, &zex, &why);
                if (st.reducible) {
                    st.reduce_msg = "explicit z(x):";
                    for (size_t j = 0; j < d.alg.size(); ++j) st.reduce_msg += "\n  " + d.alg[j] + " = " + zex[j];
                } else if (dae_index1_structural(dsys)) {
                    // Уровень B: z из связей Ньютоном в каждом вычислении f.
                    st.reducible = true;
                    st.reduce_msg = "z by Newton in every f (level B): " + why
                                  + "\n  schemes: explicit RK / DOPRI / GBS, Extr/Comp over them, adaptive";
                } else {
                    // Уровень C (индекс 2-3): связи дифференцируются до появления z и стабилизируются.
                    System s1;
                    std::string w2;
                    const int idx = dae_structural_index(dsys);
                    if (idx >= 2 && dae_stabilized_index1(dsys, 1.0, s1, &w2)) {
                        System o2;
                        std::vector<std::string> z2;
                        std::string w3;
                        const bool lev_a = dae_reduce_to_ode(s1, o2, &z2, &w3);
                        st.reducible = true;
                        st.level_c = true;
                        st.reduce_msg = "level C (index " + std::to_string(idx) + "): constraints differentiated until z "
                                        "appears, (d/dt + alpha)^m g = 0;\n  then " + std::string(lev_a ? "explicit z (level A)"
                                        : "z by Newton (level B)") + ". The original constraint is only held by the "
                                        "stabilization (drift ~ e^(-alpha t)).";
                    } else st.reduce_msg = why + (w2.empty() ? std::string() : "; " + w2);
                }
            } catch (const std::exception& e) {
                st.ok = false;
                st.message = e.what();
                st.vars.clear();
                st.params.clear();
                st.reducible = false;
                st.level_c = false;
                st.reduce_msg.clear();
                st.latex.clear();
                st.latex_lines = 0;
            }
        }
        if (st.ok) ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.45f, 1.0f), "%s", st.message.c_str());
        else       ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", st.message.c_str());
        if (st.ok && st.latex_lines > 0) {
            ImGui::TextDisabled("Equations (LaTeX preview):");
            const float row = std::floor(ImGui::GetFontSize() * 1.3f + 0.5f) * 1.9f;
            draw_latex_preview("##dae_latex", st.latex,
                               ImVec2(-1, std::min(row * (float)st.latex_lines + row * 0.5f, row * 12.0f)));
            if (ImGui::BeginPopupContextItem("##dae_latex_ctx")) {
                if (ImGui::MenuItem("Copy LaTeX") && cb.set_clipboard_text) cb.set_clipboard_text(st.latex);
                ImGui::EndPopup();
            }
        }

        // Этап 3, уровень A: ДАУ -> ОДУ подстановкой z. Снять галочку можно всегда;
        // поставить — только если сведение удаётся.
        if (st.ok) {
            ImGui::BeginDisabled(!st.reducible && !buf.dae_as_ode);
            ImGui::Checkbox("Solve as ODE", &buf.dae_as_ode);
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("If z enters its constraint linearly, z(x) is substituted\n"
                                  "into f (level A): an ODE in x for every tab and every ODE scheme.\n"
                                  "Otherwise z is found by Newton from g(x, z) = 0 before every f\n"
                                  "(level B): explicit RK / DOPRI / GBS, Extr/Comp over them, adaptive.\n"
                                  "Index 2-3 (level C): constraints are differentiated until z appears\n"
                                  "and stabilized, then level A or B.\n"
                                  "Either way z stays in the state after x, so bifurcation output,\n"
                                  "metrics and plot axes can use it; LLE/LS perturb x only.");
            ImGui::TextDisabled("%s", st.reduce_msg.c_str());
            if (buf.dae_as_ode && st.level_c) {
                InputNumStr("stabilization alpha##daestab", buf.dae_stab_alpha, 140.0f);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Level C replaces each index-(m+1) constraint by (d/dt + alpha)^m g = 0:\n"
                                      "the drift of g decays like e^(-alpha t). Larger alpha holds g tighter\n"
                                      "but makes the ODE stiffer (explicit schemes need h*alpha small).");
            }
            if (buf.dae_as_ode && !st.reducible)
                ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "cannot solve as ODE: untick");
        }

        // Начальные условия всех переменных (z тоже: согласовать их — дело
        // решателя, но стартовое приближение Ньютону даёт пользователь) и
        // значения параметров.
        if (st.ok && ImGui::BeginTable("##dae_vals", 2, ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableNextColumn();
            ImGui::TextDisabled("Initial conditions");
            ImGui::SameLine();
            if (ImGui::SmallButton("Groups...##dae_icg")) { buf.pg_ui.open = true; buf.pg_ui.kind_req = 1; }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Group and label the initial-condition fields (layout only)");
            draw_param_tree(&buf.ic_groups, st.vars, [&](const std::string& v) {
                InputNumStr((std::string("##daeic") + v).c_str(), buf.init_conditions[v], 140.0f, {}, true);
                ImGui::SameLine();
                draw_field_name(v, &buf.field_labels);
            });
            ImGui::TableNextColumn();
            ImGui::TextDisabled("Parameter values");
            ImGui::SameLine();
            if (ImGui::SmallButton("Groups...##dae_pg")) { buf.pg_ui.open = true; buf.pg_ui.kind_req = 0; }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Group and label the parameter fields (layout only)");
            draw_param_tree(&buf.param_groups, st.params, [&](const std::string& p) {
                InputNumStr((std::string("##daepv") + p).c_str(), buf.param_values[p], 140.0f, {}, true);
                ImGui::SameLine();
                draw_field_name(p, &buf.field_labels);
            });
            InputNumStr("h##daeh", buf.step_h, 140.0f, {}, true);
            // Ньютон схем ДАУ — значения по умолчанию для вкладок (как у неявных схем ОДУ).
            ImGui::TextDisabled("Newton (defaults for the tabs)");
            InputNumStr("tol##daent", buf.newton_tol, 140.0f);
            InputNumStr("max iters##daeni", buf.newton_max_iters, 140.0f);
            ImGui::Checkbox("full Newton##daenf", &buf.newton_full);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Rebuild the Jacobian every iteration. A step that runs out of\n"
                                  "iterations is accepted unconverged and the constraint drifts.");
            ImGui::EndTable();
        }
        if (st.ok) draw_param_groups_window(buf, st.params, st.vars, "lib");
        // Методы: при «Solve as ODE» система — ОДУ (схемы ОДУ, флаги scheme_*), иначе — методы ДАУ.
        ImGui::Separator();
        if (buf.dae_as_ode) draw_builtin_scheme_checkboxes(buf);
        else                draw_dae_method_checkboxes(buf);
        ImGui::Spacing();
        draw_generated_code_block(buf, cb);
    }
    else if (ImGui::BeginTabBar("editor_tabs")) {
        if (ImGui::BeginTabItem("System")) {
            draw_system_tab(buf, cb);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Parameters")) {
            draw_parameters_tab(buf);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::Separator();

    const bool save_clicked = ImGui::Button("Save", ImVec2(120, 0));
    const bool save_shortcut = ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S, false);
    if (save_clicked || save_shortcut) {
        library_editor_save(model, lib);
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(120, 0))) {
        model.edit_error.clear();
        model.library_edit_mode = AppModel::LibraryEditMode::None;
    }
}
}


void apply_system_switch(AppModel& model, SystemLibrary& lib,
                         const std::string& name)
{
    try {
        model.from_record(lib.load(name));
        switch (model.app_mode) {
        case AppModel::AppMode::Analysis: {
            model.start_phase_analysis();
            std::string j = lib.load_session(model.loaded_name, "_last");
            apply_session_json(model, j, model.phase_session, session_from_json, "_last");
            break;
        }
        case AppModel::AppMode::Parametric: {
            model.start_parametric_analysis();
            std::string jb = lib.load_session(model.loaded_name, "_last_parametric");
            apply_session_json(model, jb, model.bifurcation_session, session_from_json_parametric, "_last_parametric");
            std::string jl = lib.load_session(model.loaded_name, "_last_lle");
            apply_session_json(model, jl, model.lle_session, session_from_json_lle, "_last_lle");
            std::string js = lib.load_session(model.loaded_name, "_last_ls");
            apply_session_json(model, js, model.ls_session, session_from_json_ls, "_last_ls");
            std::string jm = lib.load_session(model.loaded_name, "_last_metrics");
            apply_session_json(model, jm, model.metrics_session, session_from_json_metrics, "_last_metrics");
            std::string jw = lib.load_session(model.loaded_name, "_last_parametric_windows");
            model.load_or_init_parametric_plot_windows(jw);
            break;
        }
        case AppModel::AppMode::Dft1D: {
            model.start_dft1d_analysis();
            std::string jd = lib.load_session(model.loaded_name, "_last_dft1d");
            apply_session_json(model, jd, model.dft1d_session, session_from_json_dft1d, "_last_dft1d");
            std::string jw = lib.load_session(model.loaded_name, "_last_dft1d_windows");
            model.load_or_init_dft1d_plot_windows(jw);
            break;
        }
        case AppModel::AppMode::Basins: {
            model.start_basins_analysis();
            std::string jb = lib.load_session(model.loaded_name, "_last_basins");
            apply_session_json(model, jb, model.basins_session, session_from_json_basins, "_last_basins");
            break;
        }
        case AppModel::AppMode::FastSync: {
            model.start_fastsync_analysis();
            std::string jf = lib.load_session(model.loaded_name, "_last_fastsync");
            apply_session_json(model, jf, model.fastsync_session, session_from_json_fastsync, "_last_fastsync");
            break;
        }
        case AppModel::AppMode::Custom: {
            // Hard reset + engine wipe. Sub-session buffers, signature caches,
            // AND the CUDA-side PTX cache in parametric_engine all need to
            // clear on system switch — otherwise the second-system Run either
            // produced a stale-looking result or zeroed out the first one.
            model.custom_session = CustomSession{};
            model.parametric_engine.reset();
            model.start_custom_analysis();
            std::string jc = lib.load_session(model.loaded_name, "_last_custom");
            apply_session_json(model, jc, model.custom_session, session_from_json_custom, "_last_custom");
            break;
        }
        case AppModel::AppMode::Order: {
            model.start_order_analysis();
            std::string jo = lib.load_session(model.loaded_name, "_last_order");
            apply_session_json(model, jo, model.order_session, session_from_json_order, "_last_order");
            std::string jw = lib.load_session(model.loaded_name, "_last_order_windows");
            model.load_or_init_order_plot_windows(jw);
            order_autosave_seed(model);
            break;
        }
        case AppModel::AppMode::Network: {
            model.start_network_analysis();
            std::string jn = lib.load_session(model.loaded_name, "_last_network");
            apply_session_json(model, jn, model.network_session, session_from_json_network, "_last_network");
            break;
        }
        case AppModel::AppMode::Neural: {
            model.start_neural_analysis();
            std::string jn = lib.load_session(model.loaded_name, "_last_neural");
            apply_session_json(model, jn, model.neural_session, session_from_json_neural, "_last_neural");
            break;
        }
        case AppModel::AppMode::Library:
        case AppModel::AppMode::Settings:
        default:
            break;
        }
        // Persist the choice so the next launch restores this system. Read-
        // modify-write so we don't clobber unrelated fields (colormaps, UI
        // scale, etc.). Silent on failure — best-effort UX polish.
        AppConfig cfg;
        load_app_config(get_exe_dir_with_sep(), cfg);
        cfg.last_system_name = model.name;
        save_app_config(get_exe_dir_with_sep(), cfg);
    } catch (...) {}
}
