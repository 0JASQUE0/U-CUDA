#include "gui_internal.h"
using namespace ucuda::ui_detail;

namespace ucuda::ui_detail {
void draw_phase_controls(PhaseAnalysisSession& s,
                                std::function<void()> on_reset_defaults,
                                AppModel* bc,
                                const GuiCallbacks* cbp) {
    // Чей конфиг правит эта панель — для ПКМ-рассылки "во все такие же".
    // В Custom эту же панель рисует уровень 3, и источником там остаётся
    // Custom: его Shared config и есть то, что уйдёт в расчёт.
    if (bc && on_reset_defaults) {
        bc->broadcast_source_tab = BroadcastTab::Phase;
        bc->broadcast_source_idx = 0;
    }
    bool changed = false;

    ImGui::Text("Phase portrait analysis");
    // Analysis-tab passes a non-null on_reset_defaults; Custom-tab passes
    // nullptr. Only Analysis is a true library-detached sandbox — in Custom,
    // this panel is one stage of a pipeline driven by the shared config,
    // so the sandbox disclaimer would be misleading.
    if (on_reset_defaults)
        ImGui::TextDisabled("Changes here are NOT saved to the library (sandbox).");

    // метод моделирования + пользовательские схемы из системы
    ImGui::Text("Method:"); ImGui::SameLine();
    changed |= draw_scheme_combo("##method", s.scheme, s.custom_schemes,
                                 [&s](const std::string&) { s.regenerate_krs(); },
                                 &s.enabled_builtin_schemes, bc, s.sys.is_map, &s.wrapper_schemes);
    // Custom КРС теперь считаются и на CPU — тело компилируется в нативный шаг
    // (см. krs_cpu.h). Принудительный GPU оставляем ровно для случая, когда
    // компилятор на машине не найден.
    if (scheme_needs_compiled_body(s.scheme, s.custom_schemes) && !s.use_gpu) {
        std::string why;
        if (!cpu_compiler_available(&why)) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1), "(custom requires GPU: %s)",
                               why.c_str());
            s.use_gpu = true;
        }
    }

    // Time, step, decimation; for a map h = 1 and "time" is the iteration index.
    if (!s.sys.is_map) {
        ImGui::Text("Step h:"); ImGui::SameLine();
        changed |= InputNumStr("##sh", s.step_h, 80); ImGui::SameLine();
    }
    ImGui::Text(s.sys.is_map ? "Iterations:" : "Time(s):"); ImGui::SameLine();
    changed |= InputNumStr("##st", s.sim_time, 70); ImGui::SameLine();
    ImGui::Text(s.sys.is_map ? "Skip:" : "Skip(s):"); ImGui::SameLine();
    changed |= InputNumStr("##ssk", s.skip_time, 70);
    // Symmetry a[0] is also available to custom KRS bodies (same slot as CD);
    // only show the field if the body actually references a[0].
    if (scheme_uses_symmetry(s.scheme, s.custom_schemes)) {
        ImGui::Text("Symmetry s:"); ImGui::SameLine();
        changed |= InputNumStr("##sym", s.symmetry_s, 70);
    }
    if (scheme_uses_cd_fp_iters(s.scheme)) {
        ImGui::Text("FP iterations n:"); ImGui::SameLine();
        changed |= InputNumStr("##cdfpn", s.cd_fp_iters, 70);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", kCdFpItersTip);
    }
    if (scheme_uses_newton(s.scheme))
        changed |= draw_newton_fields(s.newton, 70, "phase_newton");
    // ДАУ: начальные z согласуются по связи перед счётом (dae_consistent_z) — что вышло.
    if (!s.result.dae_status.empty())
        ImGui::TextDisabled("%s", s.result.dae_status.c_str());
    ImGui::Text("Decimation (every Nth point):"); ImGui::SameLine();
    changed |= InputNumStr("##dec", s.decimation, 70);
    if (!s.sys.is_map && ImGui::CollapsingHeader("Step size##phase_ad", ImGuiTreeNodeFlags_DefaultOpen)) {
        // В Custom настройки шага живут в общем конфиге (Integration вкладки) и
        // раздаются всем уровням; здесь они только показываются.
        if (on_reset_defaults) {
            if (draw_adaptive_block("phase_ad", s.adaptive, s.scheme, kAdUiRaw | kAdUiInterp | kAdUiMaxPts,
                                    s.sys.is_map, bc)) {
                changed = true;
                s.prewarm_gpu();   // включили адаптивный шаг / сменили регулятор — ядро собирается заранее
            }
        }
        else
            ImGui::TextDisabled(s.adaptive.enabled ? "Adaptive step (set in Custom -> Integration)."
                                                   : "Fixed step (set in Custom -> Integration).");
        if (s.adaptive.enabled && !s.adaptive.raw_nodes)
            ImGui::TextDisabled("h above is the output step of the uniform grid.");
        // Статистика последнего прогона: по строке на НУ.
        if (s.result.adaptive) {
            for (size_t k = 0; k < s.result.step_stats.size(); ++k) {
                const AdaptiveStats& st = s.result.step_stats[k];
                const std::string lab = k < s.result.labels.size() ? s.result.labels[k]
                                                                  : "IC" + std::to_string(k + 1);
                ImGui::TextDisabled("%s: %.0f accepted, %.0f rejected, %.0f forced, %.0f f-evals, "
                                    "h in [%.3g, %.3g], mean %.3g%s", lab.c_str(), st.nacc, st.nrej,
                                    st.nforced, st.nrhs, st.hmin, st.hmax, st.hmean,
                                    st.diverged ? " (diverged)" : (st.truncated ? " (max points)" : ""));
            }
        }
    }
    // шаг/время/децимация влияют на ось времени и сами данные: при их смене
    // просим автоскейл, чтобы time domain не "скакал" со старыми пределами.
    if (changed) s.fit_request = true;

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // переключатели
    ImGui::Checkbox("Auto recompute", &s.auto_recompute); ImGui::SameLine();
    ImGui::Checkbox("Legend shows initial conditions", &s.legend_show_ic); ImGui::SameLine();
    ImGui::Checkbox("GPU", &s.use_gpu);

    // Схемная траектория (фаза 4a). Отдельной секцией, а не ещё одной галкой в ряд:
    // настройки тут про ЖЕЛЕЗО, и путать их с параметрами интегрирования нельзя.
    if (ImGui::CollapsingHeader("Analog circuit overlay")) {
        bool cch = ImGui::Checkbox("Overlay circuit trajectory", &s.circuit_show);
        if (s.circuit_show) {
            cch |= ImGui::Checkbox("Ideal op-amp", &s.circuit_ideal_opamp);
            ImGui::SameLine(); ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("An ideal op-amp reproduces the ODE exactly, so the two curves"
                                  " coincide.\nSwitch it off to see what the hardware actually does.");
            ImGui::Text("Target amplitude (V):"); ImGui::SameLine();
            cch |= InputNumStr("##ctgt", s.circuit_target_volt, 70);
            ImGui::SameLine(); ImGui::Text("Substeps:"); ImGui::SameLine();
            cch |= InputNumStr("##csub", s.circuit_substeps, 60);
            ImGui::SameLine(); ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("The circuit is integrated with BDF2 (2nd order) while the ODE"
                                  " uses your chosen scheme.\nAt the same step the circuit would"
                                  " drift from the ODE by the method alone.");
            if (!s.circuit_ideal_opamp) {
                ImGui::SameLine(); ImGui::Text("GBW (MHz):"); ImGui::SameLine();
                cch |= InputNumStr("##cgbw", s.circuit_gbw_mhz, 70);
                ImGui::SameLine(); ImGui::Text("Vsat (V):"); ImGui::SameLine();
                cch |= InputNumStr("##cvsat", s.circuit_vsat, 70);
            }
            if (!s.result.circuit_status.empty())
                ImGui::TextWrapped("%s", s.result.circuit_status.c_str());

            ImGui::Checkbox("Multisim dialect (POLY)", &s.circuit_netlist_poly);
            ImGui::SameLine(); ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Multisim drops behavioural B sources on import, which leaves"
                                  " the multiplier nodes undriven\nand their resistors hanging by one"
                                  " end. POLY sources survive the trip.");
            ImGui::BeginDisabled(!s.result.circuit_valid || !bc);
            // Только заявка. Диалог открывается после кадра: см.
            // gui_process_deferred и AppModel::pending_netlist_export.
            if (ImGui::Button("Export SPICE netlist...") && bc) {
                bc->pending_netlist_export = true;
            }
            ImGui::EndDisabled();
        }
        if (cch) changed = true;
    }

    // Continuation is Analysis-only: Custom-tab phase is a pipeline stage
    // whose queue would fight a timer-driven re-run.
    if (on_reset_defaults) {
        // Toggle only arms the mode — the loop starts on the next Recompute so
        // enabling continuation never launches a compute on its own.
        // Unticking mid-run stops the loop (mirrors the Stop button).
        // Uncheck = reset: stop the loop and drop elapsed/peaks. Stop button preserves them.
        if (ImGui::Checkbox("Continuation (live)", &s.continuation_mode)) {
            if (!s.continuation_mode) {
                s.continuation_active  = false;
                s.continuation_paused  = false;
                s.continuation_elapsed = 0.0;
                // Диаграмма возвращается к пикам одного (последнего) прогона.
                s.rebuild_peaks_from_result();
            }
        }
        ImGui::SameLine();
        ImGui::Text("Frame delay (ms):"); ImGui::SameLine();
        InputNumStr("##cdelay", s.continuation_delay_ms, 70);
        ImGui::SameLine();
        ImGui::Text("Max peaks:"); ImGui::SameLine();
        ImGui::SetNextItemWidth(90);
        // Правится по Enter/уходу фокуса: иначе, набирая 500000, пользователь
        // на промежуточном "5" обрезал бы накопленный буфер до минимума.
        if (ImGui::InputInt("##ccap", &s.continuation_peaks_cap, 0, 0,
                            ImGuiInputTextFlags_EnterReturnsTrue)
            || ImGui::IsItemDeactivatedAfterEdit()) {
            s.continuation_peaks_cap = std::clamp(s.continuation_peaks_cap,
                PhaseAnalysisSession::kContPeaksCapMin, PhaseAnalysisSession::kContPeaksCapMax);
            s.trim_continuation_peaks();
            ++s.continuation_peaks_gen;
        }
        if (ImGui::IsItemHovered()) {
            // Буферов на IC — переменные плюс комбинированный ряд x0 + pi*x1 + e*x2
            // (тот же, что у DBSCAN в 2D-бифуркации); пик — пара double (t, peak).
            const int n_ic  = std::max(1, (int)s.ic_sets.size());
            const int n_buf = (int)s.vars.size() + 1;
            const double mb = (double)s.continuation_peaks_cap * 16.0 * n_ic * n_buf / 1.0e6;
            ImGui::SetTooltip("Continuation diagram keeps the last N peaks per buffer;\n"
                              "older ones are dropped. Range %d .. %d.\n"
                              "\n"
                              "Memory = N x 16 B x ICs x (variables + 1)\n"
                              "  16 B: a peak is two doubles (time, value)\n"
                              "  +1: the combined series x0 + pi*x1 + e*x2\n"
                              "Now: %d x 16 B x %d x %d = %.1f MB\n"
                              "(up to ~2x briefly while a buffer grows).",
                              PhaseAnalysisSession::kContPeaksCapMin,
                              PhaseAnalysisSession::kContPeaksCapMax,
                              s.continuation_peaks_cap, n_ic, n_buf, mb);
        }
    }

    ImGui::Separator();

    // параметры (общие на все проекции)
    if (!s.params.empty()) {
        ImGui::SeparatorText("Parameters");
        if (ImGui::BeginTable("aparams", 2, ImGuiTableFlags_SizingFixedFit)) {
            draw_param_tree(bc ? &bc->param_groups : nullptr, s.params, [&](const std::string& p) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); draw_field_name(p, bc ? &bc->field_labels : nullptr);
                ImGui::TableSetColumnIndex(1);
                std::string id = "##ap_" + p;
                changed |= InputNumStr(id.c_str(), s.param_values[p], 110, [&] {
                    changed |= field_restore_popup("##ap_restore", bc, BroadcastField::Param, p, s.param_values, s.params);
                });
            }, true);
            ImGui::EndTable();
        }
    }

    // начальные условия (несколько, мультистабильность)
    ImGui::SeparatorText("Initial conditions");
    if (bc) {
        int hidden = 0;
        (void)bc->ic_fields(s.vars, &hidden);
        if (hidden > 0) ImGui::TextDisabled("%d algebraic variables hidden (Settings)", hidden);
    }
    int ic_to_remove = -1;
    for (int i = 0; i < (int)s.ic_sets.size(); ++i) {
        InitialConditionSet& ic = s.ic_sets[i];
        ImGui::PushID(i);
        ImGui::Checkbox("##vis", &ic.visible); ImGui::SameLine();
        ImGui::SetNextItemWidth(70);
        InputTextStr("##label", ic.label); ImGui::SameLine();
        const FieldLabels* labels = bc ? &bc->field_labels : nullptr;
        if (bc && !bc->ic_groups.empty()) {
            // С группами НУ (схемы: десятки переменных) — набор раскрывается деревом групп, а не строкой.
            if (ImGui::SmallButton("X")) ic_to_remove = i;
            if (ImGui::TreeNode("##icvals", "values")) {
                if (ImGui::BeginTable("icv", 2, ImGuiTableFlags_SizingFixedFit)) {
                    draw_param_tree(&bc->ic_groups, bc->ic_fields(s.vars), [&](const std::string& v) {
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0); draw_field_name(v, labels);
                        ImGui::TableSetColumnIndex(1);
                        std::string id = "##icv_" + v;
                        changed |= InputNumStr(id.c_str(), ic.values[v], 110, [&] {
                            changed |= field_restore_popup("##icv_restore", bc, BroadcastField::InitCondition, v,
                                                           ic.values, s.vars);
                        });
                    }, true);
                    ImGui::EndTable();
                }
                ImGui::TreePop();
            }
        } else {
            for (const auto& v : bc ? bc->ic_fields(s.vars) : s.vars) {
                draw_field_name(v, labels); ImGui::SameLine();
                std::string id = "##icv_" + v;
                changed |= InputNumStr(id.c_str(), ic.values[v], 60, [&] {
                    changed |= field_restore_popup("##icv_restore", bc, BroadcastField::InitCondition, v,
                                                   ic.values, s.vars);
                }); ImGui::SameLine();
            }
            if (ImGui::SmallButton("X")) ic_to_remove = i;
        }
        ImGui::PopID();
    }
    if (ic_to_remove >= 0) { s.remove_ic(ic_to_remove); changed = true; }
    if (ImGui::Button("Add initial condition")) { s.add_ic(); }

    // проекции
    ImGui::SeparatorText("Projections");
    int pr_to_remove = -1;
    for (int i = 0; i < (int)s.projections.size(); ++i) {
        Projection& pr = s.projections[i];
        ImGui::PushID(1000 + i);
        bool x_drawn = false;   // кнопку удаления дерево переменных рисует в своей строке
        ImGui::SetNextItemWidth(90);
        InputTextStr("##plabel", pr.label); ImGui::SameLine();
        // тип проекции
        ImGui::SetNextItemWidth(110);
        // Порядок обязан совпадать с enum ProjType (тип пишется в сессию как int).
        const char* tnames[] = { "Phase 2D", "Time domain", "Phase 3D", "Feature diagram",
                                 "Recurrence plot", "Continuation diagram", "Step size", "Rejections" };
        int t = (int)pr.type;
        if (ImGui::Combo("##ptype", &t, tnames, IM_ARRAYSIZE(tnames))) {
            pr.type = (ProjType)t; s.fit_request = true;
        }
        ImGui::SameLine();

        if (pr.type == ProjType::Phase2D) {
            ImGui::Text("X:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(95);
            phase_axis_combo("##px", s, pr.axis_x);
            ImGui::SameLine(); ImGui::Text("Y:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(95);
            phase_axis_combo("##py", s, pr.axis_y);
        }
        else if (pr.type == ProjType::FeatureDiagram) {
            // Одна переменная: по ней ищутся пики. Оси фиксированы (значение пика /
            // интервал), поэтому переиспользуем axis_x без отдельного поля в Projection.
            // Индекс vars.size() — комбинация x0 + pi*x1 + e*x2, тот же ряд, что даёт
            // writable_var == -1 в ядре. Сентинел именно size(), а НЕ -1: соседние ветки
            // берут s.vars[axis_x] по схеме `axis_x < size() ? axis_x : 0`, и -1 ушёл бы
            // в отрицательный индекс при переключении типа проекции.
            const int nv = (int)s.vars.size();
            const bool feat_is_combo = (nv >= 2 && pr.axis_x == nv);
            const std::string preview = s.vars.empty()
                ? std::string("-")
                : (feat_is_combo ? combo_var_label(s.vars)
                                 : s.vars[pr.axis_x < nv ? pr.axis_x : 0]);

            ImGui::Text("var:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(feat_is_combo ? 170.0f : 95.0f);
            const bool open_pfv = var_combo_begin("##pfv", preview, !feat_is_combo && !s.vars.empty());
            if (open_pfv) {
                for (int k = 0; k < nv; ++k)
                    if (var_selectable(s.vars[k], pr.axis_x == k)) { pr.axis_x = k; s.fit_request = true; }
                if (nv >= 2) {
                    ImGui::Separator();
                    const std::string lbl = combo_var_label(s.vars);
                    if (ImGui::Selectable(lbl.c_str(), feat_is_combo)) { pr.axis_x = nv; s.fit_request = true; }
                }
                ImGui::EndCombo();
            }
            var_combo_preview();
            ImGui::SameLine();
            ImGui::TextDisabled("(peak vs interval)");
        }
        else if (pr.type == ProjType::RecurrencePlot) {
            // Здесь только то, без чего окно вообще не знает, что считать: чья траектория и из
            // чего строить вектор состояния. Остальные 13 параметров — в самом окне проекции,
            // по кнопке "RQA settings": в строку списка они не влезают и не должны.
            ImGui::Text("IC:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(85);
            const char* icprev = (pr.rqa_ic >= 0 && pr.rqa_ic < (int)s.ic_sets.size())
                ? s.ic_sets[pr.rqa_ic].label.c_str() : "-";
            if (ImGui::BeginCombo("##rqic", icprev)) {
                for (int k = 0; k < (int)s.ic_sets.size(); ++k)
                    if (ImGui::Selectable(s.ic_sets[k].label.c_str(), pr.rqa_ic == k)) {
                        pr.rqa_ic = k; changed = true;
                    }
                ImGui::EndCombo();
            }
            ImGui::SameLine();
            // Порядок обязан совпадать с rqa::Source (пишется в сессию как int). Нулевой пункт —
            // "не выбрано": дефолта у источника нет намеренно, см. комментарий в Projection.
            const char* snames[] = { "choose source...", "Full state vector", "Delay embedding" };
            ImGui::SetNextItemWidth(140);
            int si = (int)pr.rqa_source;
            if (ImGui::Combo("##rqsrc", &si, snames, IM_ARRAYSIZE(snames))) {
                pr.rqa_source = (rqa::Source)si; changed = true;
            }
            if (pr.rqa_source == rqa::Source::Embedding) {
                ImGui::SameLine(); ImGui::Text("var:"); ImGui::SameLine();
                ImGui::SetNextItemWidth(95);
                const int nv = (int)s.vars.size();
                if (var_combo_begin("##rqvar", s.vars.empty() ? "-" : s.vars[pr.rqa_var < nv ? pr.rqa_var : 0], !s.vars.empty())) {
                    for (int k = 0; k < nv; ++k)
                        if (var_selectable(s.vars[k], pr.rqa_var == k)) { pr.rqa_var = k; changed = true; }
                    ImGui::EndCombo();
                }
                var_combo_preview();
            }
        }
        else if (pr.type == ProjType::Phase3D) {
            ImGui::Text("X:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(95);
            phase_axis_combo("##p3x", s, pr.axis_x);
            ImGui::SameLine(); ImGui::Text("Y:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(95);
            phase_axis_combo("##p3y", s, pr.axis_y);
            ImGui::SameLine(); ImGui::Text("Z:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(95);
            phase_axis_combo("##p3z", s, pr.axis_z);
        }
        else if (pr.type == ProjType::ContinuationDiagram) {
            const int nv = (int)s.vars.size();
            const bool cont_is_combo = (nv >= 2 && pr.axis_x == nv);
            const std::string preview = s.vars.empty()
                ? std::string("-")
                : (cont_is_combo ? combo_var_label(s.vars)
                                 : s.vars[pr.axis_x < nv ? pr.axis_x : 0]);

            ImGui::Text("var:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(cont_is_combo ? 170.0f : 95.0f);
            const bool open_pcv = var_combo_begin("##pcv", preview, !cont_is_combo && !s.vars.empty());
            if (open_pcv) {
                for (int k = 0; k < nv; ++k)
                    if (var_selectable(s.vars[k], pr.axis_x == k)) { pr.axis_x = k; s.fit_request = true; }
                if (nv >= 2) {
                    ImGui::Separator();
                    const std::string lbl = combo_var_label(s.vars);
                    if (ImGui::Selectable(lbl.c_str(), cont_is_combo)) { pr.axis_x = nv; s.fit_request = true; }
                }
                ImGui::EndCombo();
            }
            var_combo_preview();
            ImGui::SameLine();
            ImGui::TextDisabled("(peak vs time)");
        }
        else if (pr.type == ProjType::StepSize) {
            ImGui::TextDisabled("(accepted h vs t; rejected and forced attempts as markers)");
        }
        else if (pr.type == ProjType::Rejections) {
            ImGui::TextDisabled("(rejected attempts per accepted step vs t)");
        }
        else { // TimeDomain — галочки переменных
            // Слотов на один больше числа переменных: последний — комбинация
            // x0 + pi*x1 + e*x2. При смене размера старые галочки переносим, а комбинацию
            // гасим: иначе у сохранённых сессий сама собой появилась бы новая кривая и
            // перемасштабировала ось Y (её амплитуда заметно больше).
            const int nv_td = (int)s.vars.size();
            if ((int)pr.show_var.size() != nv_td + 1) {
                std::vector<bool> prev = pr.show_var;
                pr.show_var.assign((size_t)nv_td + 1, true);
                for (size_t j = 0; j < prev.size() && j < (size_t)nv_td; ++j)
                    pr.show_var[j] = prev[j];
                pr.show_var[(size_t)nv_td] = false;
            }
            const FieldLabels* labels = bc ? &bc->field_labels : nullptr;
            auto var_box = [&](int k) {
                bool v = pr.show_var[k];
                ImGui::PushID(k);
                if (ImGui::Checkbox("##sv", &v)) { pr.show_var[k] = v; }
                ImGui::PopID();
                ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
                draw_field_name(s.vars[k], labels);
            };
            auto combo_box = [&] {
                if (nv_td < 2) return;
                bool v = pr.show_var[(size_t)nv_td];
                if (ImGui::Checkbox(combo_var_label(s.vars).c_str(), &v))
                    pr.show_var[(size_t)nv_td] = v;
            };
            if (bc && !bc->ic_groups.empty()) {
                // С группами НУ (схемы) — переменные тем же деревом групп, что и начальные условия.
                combo_box();
                ImGui::SameLine();
                if (ImGui::SmallButton("X")) pr_to_remove = i;
                x_drawn = true;
                std::map<std::string, int> idx;
                for (int k = 0; k < nv_td; ++k) idx.emplace(s.vars[(size_t)k], k);
                // ПКМ по группе (и по самому узлу "variables" — все переменные): включить / выключить разом.
                auto group_menu = [&](const std::vector<std::string>& names) {
                    auto set_all = [&](bool on) { for (const std::string& n : names) pr.show_var[(size_t)idx.at(n)] = on; };
                    if (ImGui::MenuItem("Select all"))   set_all(true);
                    if (ImGui::MenuItem("Deselect all")) set_all(false);
                    if (ImGui::MenuItem("Only these")) {
                        for (int k = 0; k < nv_td; ++k) pr.show_var[(size_t)k] = false;
                        set_all(true);
                    }
                };
                const bool vars_open = ImGui::TreeNode("##prvars", "variables");
                if (ImGui::BeginPopupContextItem("##prvars_menu")) {
                    group_menu(s.vars);
                    ImGui::EndPopup();
                }
                if (vars_open) {
                    draw_param_tree(&bc->ic_groups, s.vars, [&](const std::string& n) { var_box(idx.at(n)); },
                                    false, group_menu);
                    ImGui::TreePop();
                }
            } else {
                ImGui::Text("vars:"); ImGui::SameLine();
                for (int k = 0; k < nv_td; ++k) { var_box(k); ImGui::SameLine(); }
                combo_box();
                if (nv_td >= 2) ImGui::SameLine();
                ImGui::NewLine();
            }
        }
        if (!x_drawn) {
            ImGui::SameLine();
            if (ImGui::SmallButton("X")) pr_to_remove = i;
        }
        ImGui::PopID();
    }
    if (pr_to_remove >= 0) s.remove_projection(pr_to_remove);
    if (ImGui::Button("Add projection")) { s.add_projection(); }
    ImGui::SameLine();
    if (ImGui::Button("Reset windows layout")) { s.layout_generation++; }

    ImGui::Separator();
    // Debug panel: show what kernel/integrator actually computes. Useful for
    // comparing CPU vs GPU runs and catching parsing surprises.
    if (ImGui::CollapsingHeader("Debug: KRS body & parsed values")) {
        ImGui::TextDisabled("KRS body — what NVRTC compiles for GPU. On CPU the same RHS\n"
                            "is parsed into an AST and interpreted. For CD the CPU and GPU\n"
                            "algorithms differ (see the second panel). Values below are\n"
                            "the exact doubles both paths parse.");

        if (ImGui::Button("Regenerate")) s.regenerate_krs();
        ImGui::SameLine();
        ImGui::TextDisabled("(rebuilds KRS from the current system/scheme)");

        // GPU KRS body (what NVRTC compiles).
        ImGui::SeparatorText("KRS (GPU, NVRTC)");
        std::string body_gpu = !s.krs_code.empty() ? s.krs_code
            : !s.krs_error.empty() ? "(empty: " + s.krs_error + ")"
            : std::string("(empty — press Regenerate or change Method)");
        std::vector<char> buf_gpu(body_gpu.begin(), body_gpu.end());
        buf_gpu.resize(body_gpu.size() + 1);
        buf_gpu[body_gpu.size()] = '\0';
        ImGui::InputTextMultiline("##krs_gpu", buf_gpu.data(), buf_gpu.size(),
            ImVec2(-1, 200), ImGuiInputTextFlags_ReadOnly);

        // CPU equivalent: identical to GPU for every scheme — the CPU integrator
        // evaluates the same AST, and the diagonally-implicit ones solve each
        // equation with the same analytic formula (SystemEvaluator::solve_diag_implicit).
        ImGui::SeparatorText("KRS (CPU equivalent)");
        // Кодогенерация — не для каждого кадра: панель открыта, а ImGui
        // перерисовывает её 60 раз в секунду. Кэшируем по паре (схема, тело
        // GPU-KRS): regenerate_krs переписывает krs_code при любой правке
        // системы или схемы, поэтому пара однозначно описывает вход генератора.
        static std::string cpu_cache_key, cpu_cache_body;
        const std::string cpu_key = s.scheme + "\n" + s.krs_code;
        if (cpu_cache_key != cpu_key) {
            try {
                // Через тот же резолвер, что и GPU-форма. Раньше здесь стоял
                // прямой scheme_from_name(s.scheme), а он на любое незнакомое
                // имя молча отдаёт Scheme::Euler — панель показывала бы код
                // Эйлера под заголовком "эквивалент" для каждой Extr(...).
                cpu_cache_body = compute_krs_cpu_for_scheme(s.custom_schemes, s.sys, s.scheme);
            } catch (...) {
                cpu_cache_body = "(generation failed)";
            }
            if (cpu_cache_body.empty())
                cpu_cache_body = "(empty)";
            cpu_cache_key = cpu_key;
        }
        std::vector<char>& buf_cpu = input_scratch(cpu_cache_body, 1);
        ImGui::InputTextMultiline("##krs_cpu", buf_cpu.data(), buf_cpu.size(),
            ImVec2(-1, 200), ImGuiInputTextFlags_ReadOnly);
        ImGui::TextDisabled("(CPU and GPU run the same algorithm — texts match; the\n"
                            "diagonally-implicit schemes solve linear-in-var equations\n"
                            "analytically on both paths and iterate on both where they can't)");

        ImGui::SeparatorText("Parsed inputs (double, %.17g)");
        // Панель показывает ИМЕННО то, что увидит движок, поэтому обязана
        // разбирать поля тем же кодом — см. num_parse.h.
        auto parse = [](const std::string& v, double def) { return parse_num(v, def); };

        ImGui::Text("h        = %.17g", parse(s.step_h, 0.01));
        ImGui::Text("a[0] (s) = %.17g", parse(s.symmetry_s, 0.5));
        for (size_t j = 0; j < s.params.size(); ++j) {
            auto it = s.param_values.find(s.params[j]);
            double v = (it != s.param_values.end()) ? parse(it->second, 0.0) : 0.0;
            ImGui::Text("a[%zu] (%s) = %.17g", j + 1, s.params[j].c_str(), v);
        }
        for (size_t k = 0; k < s.ic_sets.size(); ++k) {
            ImGui::Text("%s:", s.ic_sets[k].label.c_str());
            for (size_t i = 0; i < s.vars.size(); ++i) {
                auto it = s.ic_sets[k].values.find(s.vars[i]);
                double v = (it != s.ic_sets[k].values.end()) ? parse(it->second, 0.0) : 0.0;
                ImGui::Text("  X[%zu] (%s) = %.17g", i, s.vars[i].c_str(), v);
            }
        }
    }

    if (on_reset_defaults) {
        ImGui::Separator();
        if (ImGui::Button("Reset to defaults")) {
            try { on_reset_defaults(); } catch (...) {}
        }
    }

    ImGui::Separator();
    // Recompute по кнопке или Ctrl+R, дисейблится во время async-расчёта.
    // Авто-сохранение _last делается в draw_gui::poll(), когда результат готов.
    bool do_recompute = false;
    if (s.continuation_active) {
        // Stop supersedes Recomputing... — the loop is expected to keep an
        // async in flight most of the time, and hiding Stop behind it would
        // trap the user until a chunk finishes.
        const float avail_w = ImGui::GetContentRegionAvail().x;
        const float spacing = ImGui::GetStyle().ItemSpacing.x;
        const float half_w  = (avail_w - spacing) * 0.5f;
        if (s.continuation_paused) {
            if (ImGui::Button("Resume (continuation)", ImVec2(half_w, 0))) {
                s.continuation_paused     = false;
                s.continuation_last_frame = std::chrono::steady_clock::now();
            }
        } else {
            if (ImGui::Button("Pause (continuation)", ImVec2(half_w, 0)))
                s.continuation_paused = true;
        }
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.60f, 0.20f, 0.20f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.75f, 0.28f, 0.28f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.50f, 0.15f, 0.15f, 1.0f));
        if (ImGui::Button("Stop (continuation)", ImVec2(-1, 0))) {
            s.continuation_active = false;
            s.continuation_paused = false;
            s.request_cancel();   // адаптивный кадр может идти долго — прервать и его
        }
        ImGui::PopStyleColor(3);
    }
    else if (s.in_flight) {
        // Отмену слушает ветка адаптивного шага; у постоянного шага число шагов известно,
        // и расчёт доходит до конца.
        const bool can_cancel = s.adaptive.enabled && !s.sys.is_map && s.cancel_token != nullptr;
        const bool cancel_sent = can_cancel && s.cancel_token->load(std::memory_order_relaxed);
        const float bw = can_cancel ? (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f
                                    : -1.0f;
        ImGui::BeginDisabled();
        ImGui::Button(cancel_sent ? "Cancelling..." : "Recomputing...", ImVec2(bw, 0));
        ImGui::EndDisabled();
        if (can_cancel) {
            ImGui::SameLine();
            ImGui::BeginDisabled(cancel_sent);
            if (ImGui::Button("Cancel##phase_cancel", ImVec2(-1, 0))) s.request_cancel();
            ImGui::EndDisabled();
        }
    }
    else {
        do_recompute = ImGui::Button("Recompute (Ctrl+R)", ImVec2(-1, 0));
        if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_R, false)) do_recompute = true;
        // Правки RQA приходят из окон проекций — другая функция GUI, свой кадр (см. rqa_dirty).
        if (s.rqa_dirty) { changed = true; s.rqa_dirty = false; }
        if (s.auto_recompute && changed) do_recompute = true;
    }
    if (s.continuation_mode)
        ImGui::TextDisabled("Simulated: %.3f s", s.continuation_elapsed);
    if (do_recompute) {
        // Fresh manual Run resets continuation seed even if the mode is on,
        // so a user tweak (params/IC/method) always starts from the edited IC.
        if (s.continuation_mode) {
            s.continuation_active      = true;
            s.continuation_paused      = false;
            s.continuation_first_frame = true;
            s.continuation_state.clear();
            s.continuation_last_frame  = std::chrono::steady_clock::now();
            s.continuation_elapsed = 0.0;
            s.continuation_peaks.clear();
            ++s.continuation_peaks_gen;
        }
        s.recompute_async();
    }

    if (!s.result.error.empty())
        ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "%s", s.result.error.c_str());
}
}

namespace ucuda::ui_detail {
void draw_builtin_scheme_checkboxes(AppModel& model) {
    ImGui::Text("Schemes to generate:");
    // И кнопки, и сам список идут по kBuiltinSchemes и пропускают схемы,
    // скрытые в Settings: скрытая схема не должна ни показываться, ни
    // включаться скопом.
    auto scheme_shown = [&](const BuiltinScheme& b) { return !model.scheme_hidden(b.name); };
    if (ImGui::Button("Select all"))
        for (const auto& b : kBuiltinSchemes) if (scheme_shown(b)) model.*b.flag = true;
    ImGui::SameLine();
    if (ImGui::Button("Clear all"))
        for (const auto& b : kBuiltinSchemes) if (scheme_shown(b)) model.*b.flag = false;

    // Схемы сгруппированы по порядку точности — тот же порядок, что в комбо
    // выбора схемы и в списке генерации (AppModel::generate). Таблица
    // отсортирована по order, поэтому шапка печатается на его смене, и группа
    // целиком скрытых схем не оставляет пустого заголовка.
    // У CD, Complex CD, SEMP и SIMP порядок группы достигается только при
    // s = a[0] = 0.5; при других s все четыре падают до первого.
    {
        std::string  shown_group;
        bool first_in_line = true;
        for (const auto& b : kBuiltinSchemes) {
            if (!scheme_shown(b)) continue;
            if (scheme_group_title(b) != shown_group) {
                ImGui::SeparatorText(scheme_group_title(b).c_str());
                shown_group   = scheme_group_title(b);
                first_in_line = true;
            }
            if (!first_in_line) ImGui::SameLine();
            first_in_line = false;
            ImGui::Checkbox(b.name, &(model.*b.flag));
            if (b.tooltip && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", b.tooltip);
        }
    }

    ImGui::TextDisabled("Generate includes only methods selected here and visible in Settings.");
}
}

namespace ucuda::ui_detail {
void draw_dae_method_checkboxes(AppModel& model) {
    ImGui::Text("DAE methods to generate:");
    // Как у схем ОДУ: методы, скрытые в Settings, не показываются и не включаются скопом.
    const std::vector<std::string> all = dae_method_list();
    auto set_on = [&](const std::string& n, bool on) {
        auto& off = model.dae_schemes_off;
        off.erase(std::remove(off.begin(), off.end(), n), off.end());
        if (!on) off.push_back(n);
    };
    if (ImGui::Button("Select all##dae"))
        for (const auto& n : all) if (!model.dae_method_hidden(n)) set_on(n, true);
    ImGui::SameLine();
    if (ImGui::Button("Clear all##dae"))
        for (const auto& n : all) if (!model.dae_method_hidden(n)) set_on(n, false);
    // Группы: неявные (любой индекс), с решением связи при каждом x (только индекс 1), многошаговый.
    // Подсказки — по имени (список методов рос, по номеру они съезжали).
    static const std::map<std::string, const char*> kTips = {
        { "DAE BDF1", "Implicit Euler on [x; z], one Newton per step. Order 1, L-stable, any index." },
        { "DAE Trapezoidal", "Trapezoid in x, constraint at the new node. Order 2, not L-stable (index 3 diverges)." },
        { "DAE TR-BDF2", "TR-BDF2: trapezoid to t + gamma h, then BDF2. Order 2, L-stable." },
        { "DAE SDIRK2", "SDIRK2 (Alexander): two implicit Euler stages with gamma h. Order 2 (z: 1 on index 2), L-stable." },
        { "DAE Implicit Midpoint", "Implicit midpoint: control method, does not hold an index-2 constraint, diverges on index 3." },
        { "DAE Radau IIA (5)", "Radau IIA, 3 stages: order 5 (index 1), x 5 / z 3 (index 2), x 3 / z 2 (index 3).\n"
                               "Newton full = 0 in Integration: RADAU5 simplified Newton (one J per step)." },
        { "DAE BDF1 + GGL", "BDF1 on the GGL-stabilised system (index 3 -> 2, multipliers local to the step)." },
        { "DAE Radau IIA + GGL", "Radau IIA on the GGL-stabilised system: index 3 with x 5 / lambda 3." },
        { "DAE Rosenbrock ROS2", "Rosenbrock ROS2: linearly implicit, one LU per step, no Newton. Order 2, L-stable." },
        { "DAE Rosenbrock ROS3P", "Rosenbrock ROS3P (Lang-Verwer): order 3 in x and z, A-stable, one LU per step." },
        { "DAE Rodas4", "Rodas4 (Hairer-Wanner): order 4, stiffly accurate, L-stable, one LU per step.\n"
                        "With Integration -> Adaptive step: embedded error estimate (index 1)." },
        { "DAE CD", "CD for DAE (level B): explicit forward half with z re-solved after every x update, implicit\n"
                    "reverse half with one Newton on (x_i, z). Order 2 at s = 0.5. Index 1 only." },
        { "DAE CD + Yoshida-4", "Yoshida composition of three CD steps: order 4 at s = 0.5. Index 1 only." },
        { "DAE CD (index 2)", "CD with z constant per half-step, chosen so that g holds at its end. Index 2 only." },
        { "DAE Explicit Euler", "Explicit Euler with z solved from the constraint at every stage. Index 1, non-stiff." },
        { "DAE Explicit Midpoint", "Explicit midpoint with z solved at every stage. Order 2, index 1, non-stiff." },
        { "DAE RK4", "RK4 with z solved at every stage. Order 4, index 1, non-stiff." },
        { kBdfSchemeDae, "BDF orders 1-5 (DASSL form), its own multistep driver: Analysis on CPU / GPU and the\n"
                         "Parametric sweeps. Adaptive step: variable step and order; fixed step: step h, order\n"
                         "ramped up 1..5, no step controller. Index 1-2 directly, index 3 through GGL.\n"
                         "Generate prints its residual [f; g] and Jacobian." },
    };
    static const char* const kGroupTitle[] = { "implicit, any index", "constraint solved whenever x changes, index 1",
                                               "multistep" };
    int group = 0, shown_group = -1;
    bool first_in_line = true;
    for (const std::string& n : all) {
        if (n == "DAE CD") group = 1;
        if (n == kBdfSchemeDae) group = 2;
        if (model.dae_method_hidden(n)) continue;
        if (group != shown_group) { ImGui::SeparatorText(kGroupTitle[group]); shown_group = group; first_in_line = true; }
        if (!first_in_line) ImGui::SameLine();
        first_in_line = false;
        bool on = model.dae_scheme_on(n);
        if (ImGui::Checkbox(n.c_str(), &on)) set_on(n, on);
        auto tip = kTips.find(n);
        if (tip != kTips.end() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip->second);
    }
    ImGui::TextDisabled("Generate includes only methods selected here and visible in Settings.");
}
}

namespace ucuda::ui_detail {
void draw_param_groups_window(AppModel& m, const std::vector<std::string>& params,
                              const std::vector<std::string>& vars, const char* id) {
    AppModel::ParamGroupsUi& u = m.pg_ui;
    if (!u.open) return;
    ImGui::SetNextWindowSize(ImVec2(760.0f, 620.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin((std::string("Parameter groups##") + id).c_str(), &u.open)) { ImGui::End(); return; }
    ImGui::TextDisabled("Only how the fields are laid out and labelled in the tabs; the computation does not change.\n"
                        "Nested groups: '/' in the name, e.g. OPA2134/XU1. Drag a group row to reorder it among\n"
                        "its siblings; drag a name onto a group row to move it there.");
    if (ImGui::BeginTabBar("##pg_kind")) {
        const int was = u.kind;
        auto req = [&](int k) { return u.kind_req == k ? ImGuiTabItemFlags_SetSelected : 0; };
        if (ImGui::BeginTabItem("Parameters", nullptr, req(0))) { u.kind = 0; ImGui::EndTabItem(); }
        if (!vars.empty() && ImGui::BeginTabItem("Initial conditions", nullptr, req(1))) { u.kind = 1; ImGui::EndTabItem(); }
        ImGui::EndTabBar();
        u.kind_req = -1;
        if (u.kind != was) { u.sel.clear(); u.target.clear(); u.rename_from.clear(); }
    }
    if (vars.empty()) u.kind = 0;
    ParamGroups& G = u.kind == 1 ? m.ic_groups : m.param_groups;
    const std::vector<std::string>& names = u.kind == 1 ? vars : params;

    InputTextStrHint("##pg_new", u.new_path, "new group, e.g. OPA2134/XU1", 300.0f);
    ImGui::SameLine();
    if (ImGui::Button("Create group")) {
        const std::string p = pg_normalize_path(u.new_path);
        if (!p.empty()) { pg_get(G, p); u.target = p; }
        u.new_path.clear();
    }

    const std::vector<std::string> rows = pg_rows(G);
    const std::set<std::string> present(names.begin(), names.end());
    auto count_in = [&](const std::string& path) {
        int n = 0;
        for (const ParamGroup& g : G)
            if (pg_in_subtree(g.path, path))
                for (const std::string& p : g.params) n += present.count(p) ? 1 : 0;
        return n;
    };

    // Перетаскивание: строка группы (среди соседей) и имя из таблицы (на строку группы).
    static const char* const kRowDrag = "UCUDA_PG_ROW";
    static const char* const kNameDrag = "UCUDA_PG_NAME";
    std::string drop_from, drop_to, drop_name;
    bool drop_after = false;

    ImGui::SeparatorText("Groups");
    if (rows.empty()) ImGui::TextDisabled("No groups: the tabs show a flat list.");
    std::string del, ren_from, ren_to;
    if (!rows.empty() && ImGui::BeginChild("##pg_rows", ImVec2(0, std::min(240.0f, 26.0f * (float)rows.size() + 8.0f)),
                                           ImGuiChildFlags_Borders)) {
        for (const std::string& r : rows) {
            ImGui::PushID(r.c_str());
            const int depth = (int)std::count(r.begin(), r.end(), '/');
            const size_t sl = r.rfind('/');
            const std::string seg = sl == std::string::npos ? r : r.substr(sl + 1);
            ImGui::Indent(16.0f * (float)depth + 1.0f);
            if (u.rename_from == r) {
                InputTextStr("##rn", u.rename_to, 260.0f);
                ImGui::SameLine();
                if (ImGui::SmallButton("OK")) { ren_from = r; ren_to = u.rename_to; }
                ImGui::SameLine();
                if (ImGui::SmallButton("Cancel")) u.rename_from.clear();
            } else {
                const std::string text = seg + "  (" + std::to_string(count_in(r)) + ")";
                ImGui::Selectable(text.c_str(), u.target == r, 0, ImVec2(330.0f - 16.0f * (float)depth, 0));
                if (ImGui::IsItemClicked()) u.target = r;
                if (ImGui::BeginDragDropSource()) {
                    ImGui::SetDragDropPayload(kRowDrag, r.data(), r.size());
                    ImGui::TextUnformatted(seg.c_str());
                    ImGui::EndDragDropSource();
                }
                if (ImGui::BeginDragDropTarget()) {
                    const ImGuiPayload* pk = ImGui::GetDragDropPayload();
                    const ImRect rc(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
                    if (pk && pk->IsDataType(kRowDrag)) {
                        const std::string from((const char*)pk->Data, (size_t)pk->DataSize);
                        if (from != r && pg_parent(from) == pg_parent(r)) {
                            const bool after = ImGui::GetMousePos().y > (rc.Min.y + rc.Max.y) * 0.5f;
                            const float y = after ? rc.Max.y : rc.Min.y;
                            ImGui::GetWindowDrawList()->AddLine(ImVec2(rc.Min.x, y), ImVec2(rc.Max.x, y),
                                                                ImGui::GetColorU32(ImGuiCol_DragDropTarget), 2.0f);
                            if (ImGui::AcceptDragDropPayload(kRowDrag, ImGuiDragDropFlags_AcceptNoDrawDefaultRect)) {
                                drop_from = from; drop_to = r; drop_after = after;
                            }
                        }
                    } else if (const ImGuiPayload* pn = ImGui::AcceptDragDropPayload(kNameDrag)) {
                        drop_name.assign((const char*)pn->Data, (size_t)pn->DataSize);
                        drop_to = r;
                    }
                    ImGui::EndDragDropTarget();
                }
                ImGui::SameLine(360.0f);
                if (ImGui::SmallButton("Rename")) { u.rename_from = r; u.rename_to = r; }
                ImGui::SameLine();
                if (ImGui::SmallButton("Delete")) del = r;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Removes the group and its subgroups; their fields stay ungrouped.");
            }
            ImGui::Unindent(16.0f * (float)depth + 1.0f);
            ImGui::PopID();
        }
    }
    if (!rows.empty()) ImGui::EndChild();
    if (!drop_from.empty()) pg_reorder(G, drop_from, drop_to, drop_after);
    if (!drop_name.empty()) {
        // Брошено выделенное имя — едут все выделенные.
        if (u.sel.count(drop_name)) {
            for (const std::string& p : names) if (u.sel.count(p)) pg_move(G, p, drop_to);
            u.sel.clear();
        } else pg_move(G, drop_name, drop_to);
    }
    if (!del.empty()) { pg_remove(G, del); if (pg_in_subtree(u.target, del)) u.target.clear(); }
    if (!ren_from.empty()) {
        pg_rename(G, ren_from, ren_to);
        u.rename_from.clear();
        u.target.clear();
    }

    ImGui::SeparatorText(u.kind == 1 ? "Initial conditions" : "Parameters");
    InputTextStr("filter##pg", u.filter, 200.0f);
    std::vector<std::string> shown;
    for (const std::string& p : names)
        if (u.filter.empty() || p.find(u.filter) != std::string::npos
            || pg_group_of(G, p).find(u.filter) != std::string::npos) shown.push_back(p);
    ImGui::SameLine();
    if (ImGui::Button("Select shown")) u.sel.insert(shown.begin(), shown.end());
    ImGui::SameLine();
    if (ImGui::Button("Clear selection")) u.sel.clear();

    // Список целей: все строки групп + "без группы".
    auto group_combo = [&](const char* label, std::string& cur) {
        bool changed = false;
        if (ImGui::BeginCombo(label, cur.empty() ? "(no group)" : cur.c_str())) {
            if (ImGui::Selectable("(no group)", cur.empty())) { cur.clear(); changed = true; }
            for (const std::string& r : rows)
                if (ImGui::Selectable(r.c_str(), cur == r)) { cur = r; changed = true; }
            ImGui::EndCombo();
        }
        return changed;
    };
    ImGui::SetNextItemWidth(260.0f);
    group_combo("##pg_target", u.target);
    ImGui::SameLine();
    int nsel = 0;
    for (const std::string& p : names) nsel += u.sel.count(p) ? 1 : 0;
    ImGui::BeginDisabled(nsel == 0);
    if (ImGui::Button(("Move selected (" + std::to_string(nsel) + ")").c_str())) {
        for (const std::string& p : names)
            if (u.sel.count(p)) pg_move(G, p, u.target);
        u.sel.clear();
    }
    ImGui::EndDisabled();

    if (ImGui::BeginTable("##pg_params", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersInnerV,
                          ImVec2(0, ImGui::GetContentRegionAvail().y))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 24.0f);
        ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthFixed, 150.0f);
        ImGui::TableSetupColumn("label (LaTeX)", ImGuiTableColumnFlags_WidthFixed, 150.0f);
        ImGui::TableSetupColumn("shown as", ImGuiTableColumnFlags_WidthFixed, 110.0f);
        ImGui::TableSetupColumn("group");
        ImGui::TableHeadersRow();
        ImGuiListClipper clip;
        clip.Begin((int)shown.size());
        while (clip.Step())
            for (int i = clip.DisplayStart; i < clip.DisplayEnd; ++i) {
                const std::string& p = shown[(size_t)i];
                ImGui::PushID(p.c_str());
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                bool s = u.sel.count(p) > 0;
                if (ImGui::Checkbox("##s", &s)) { if (s) u.sel.insert(p); else u.sel.erase(p); }
                ImGui::TableSetColumnIndex(1);
                ImGui::Selectable(p.c_str(), false);
                if (ImGui::BeginDragDropSource()) {
                    ImGui::SetDragDropPayload(kNameDrag, p.data(), p.size());
                    const int k = u.sel.count(p) ? nsel : 1;
                    if (k > 1) ImGui::Text("%s (+%d)", p.c_str(), k - 1);
                    else ImGui::TextUnformatted(p.c_str());
                    ImGui::EndDragDropSource();
                }
                ImGui::TableSetColumnIndex(2);
                {
                    auto it = m.field_labels.find(p);
                    std::string lab = it == m.field_labels.end() ? std::string() : it->second;
                    ImGui::SetNextItemWidth(-1.0f);
                    const std::string hint = field_label_auto(p);
                    if (InputTextStrHint("##lab", lab, hint.c_str())) {
                        if (lab.find_first_not_of(" \t") == std::string::npos) m.field_labels.erase(p);
                        else m.field_labels[p] = lab;
                    }
                }
                ImGui::TableSetColumnIndex(3);
                draw_field_name(p, &m.field_labels);
                ImGui::TableSetColumnIndex(4);
                std::string g = pg_group_of(G, p);
                ImGui::SetNextItemWidth(-1.0f);
                if (group_combo("##g", g)) pg_move(G, p, g);
                ImGui::PopID();
            }
        ImGui::EndTable();
    }
    ImGui::End();
}
}

namespace ucuda::ui_detail {
LibTab lib_tab_of(bool is_dae, bool is_map) {
    return is_dae ? LibTab::Dae : is_map ? LibTab::Map : LibTab::Ode;
}
}

namespace ucuda::ui_detail {

}

namespace ucuda::ui_detail {
std::string dot_latex(const std::string& name, const FieldLabels* labels ) {
    std::string tex = name_to_latex(name);
    if (labels) {
        auto it = labels->find(name);
        if (it != labels->end() && !it->second.empty()) tex = it->second;
    }
    const size_t us = tex.find("_{");
    return us == std::string::npos || us == 0 ? "\\dot{" + tex + "}" : "\\dot{" + tex.substr(0, us) + "}" + tex.substr(us);
}
}

namespace ucuda::ui_detail {
void draw_library_editor(AppModel& model, SystemLibrary& lib,
                                const GuiCallbacks& cb) {
    AppModel& buf = *model.library_edit_buffer;
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

namespace ucuda::ui_detail {
std::string phase_axis_name(const PhaseAnalysisSession& s, int ax) {
    const int nv = (int)s.vars.size();
    if (ax >= 0 && ax < nv) return s.vars[(size_t)ax];
    if (ax >= nv && ax - nv < (int)s.observables.size()) return s.observables[(size_t)(ax - nv)];
    return s.vars.empty() ? std::string("-") : s.vars[0];
}
}

namespace ucuda::ui_detail {
void phase_axis_combo(const char* id, PhaseAnalysisSession& s, int& axis) {
    if (var_combo_begin(id, phase_axis_name(s, axis))) {
        for (int k = 0; k < (int)s.vars.size(); ++k)
            if (var_selectable(s.vars[(size_t)k], axis == k)) { axis = k; s.fit_request = true; }
        if (!s.observables.empty()) {
            ImGui::SeparatorText("z (observables)");
            const int nv = (int)s.vars.size();
            for (int j = 0; j < (int)s.observables.size(); ++j)
                if (var_selectable(s.observables[(size_t)j], axis == nv + j)) { axis = nv + j; s.fit_request = true; }
        }
        ImGui::EndCombo();
    }
    var_combo_preview();
}
}
