#include "gui_internal.h"

namespace ucuda::ui_detail {


// Editor state: name/note header + System/Parameters sub-tabs + Save/Cancel.
// Edits go into model.library_edit_buffer, not model itself — Cancel simply
// drops the edit_mode without needing to restore anything on the live model.
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
    if (ImGui::BeginTabBar("editor_tabs")) {
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


// РЕЖИМ АНАЛИЗА: пространство фазовых портретов

// Панель настроек сессии: параметры (общие), НУ (список), проекции (список), время/шаг,
// метод (заглушка), кнопка пересчёта. Reset lambda — что делает «Reset to defaults»:
// Analysis-tab передаёт model.from_record + start_phase_analysis, Custom-tab nullptr
// (там перезагружаются из System tab или Run pipeline).
// cbp — только ради диалога сохранения netlist'а. Необязателен: в Custom эту же
// панель рисует уровень 3, где callbacks не прокинуты, и кнопка там неактивна.
void draw_phase_controls(PhaseAnalysisSession& s,
                                std::function<void()> on_reset_defaults,
                                AppModel* bc ,
                                const GuiCallbacks* cbp ) {
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
        if (!krs_cpu_backend_available(&why)) {
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
            for (const auto& p : s.params) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("%s", p.c_str());
                ImGui::TableSetColumnIndex(1);
                std::string id = "##ap_" + p;
                changed |= InputNumStr(id.c_str(), s.param_values[p], 110);
            }
            ImGui::EndTable();
        }
    }

    // начальные условия (несколько, мультистабильность)
    ImGui::SeparatorText("Initial conditions");
    int ic_to_remove = -1;
    for (int i = 0; i < (int)s.ic_sets.size(); ++i) {
        InitialConditionSet& ic = s.ic_sets[i];
        ImGui::PushID(i);
        ImGui::Checkbox("##vis", &ic.visible); ImGui::SameLine();
        ImGui::SetNextItemWidth(70);
        InputTextStr("##label", ic.label); ImGui::SameLine();
        for (const auto& v : s.vars) {
            ImGui::Text("%s:", v.c_str()); ImGui::SameLine();
            std::string id = "##icv_" + v;
            changed |= InputNumStr(id.c_str(), ic.values[v], 60); ImGui::SameLine();
        }
        if (ImGui::SmallButton("X")) ic_to_remove = i;
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
            ImGui::SetNextItemWidth(55);
            if (ImGui::BeginCombo("##px", s.vars.empty() ? "-" : s.vars[pr.axis_x < (int)s.vars.size() ? pr.axis_x : 0].c_str())) {
                for (int k = 0; k < (int)s.vars.size(); ++k)
                    if (ImGui::Selectable(s.vars[k].c_str(), pr.axis_x == k)) { pr.axis_x = k; s.fit_request = true; }
                ImGui::EndCombo();
            }
            ImGui::SameLine(); ImGui::Text("Y:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(55);
            if (ImGui::BeginCombo("##py", s.vars.empty() ? "-" : s.vars[pr.axis_y < (int)s.vars.size() ? pr.axis_y : 0].c_str())) {
                for (int k = 0; k < (int)s.vars.size(); ++k)
                    if (ImGui::Selectable(s.vars[k].c_str(), pr.axis_y == k)) { pr.axis_y = k; s.fit_request = true; }
                ImGui::EndCombo();
            }
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
            ImGui::SetNextItemWidth(feat_is_combo ? 170.0f : 55.0f);
            if (ImGui::BeginCombo("##pfv", preview.c_str())) {
                for (int k = 0; k < nv; ++k)
                    if (ImGui::Selectable(s.vars[k].c_str(), pr.axis_x == k)) { pr.axis_x = k; s.fit_request = true; }
                if (nv >= 2) {
                    ImGui::Separator();
                    const std::string lbl = combo_var_label(s.vars);
                    if (ImGui::Selectable(lbl.c_str(), feat_is_combo)) { pr.axis_x = nv; s.fit_request = true; }
                }
                ImGui::EndCombo();
            }
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
                ImGui::SetNextItemWidth(55);
                const int nv = (int)s.vars.size();
                if (ImGui::BeginCombo("##rqvar", s.vars.empty() ? "-" : s.vars[pr.rqa_var < nv ? pr.rqa_var : 0].c_str())) {
                    for (int k = 0; k < nv; ++k)
                        if (ImGui::Selectable(s.vars[k].c_str(), pr.rqa_var == k)) { pr.rqa_var = k; changed = true; }
                    ImGui::EndCombo();
                }
            }
        }
        else if (pr.type == ProjType::Phase3D) {
            ImGui::Text("X:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(50);
            if (ImGui::BeginCombo("##p3x", s.vars.empty() ? "-" : s.vars[pr.axis_x < (int)s.vars.size() ? pr.axis_x : 0].c_str())) {
                for (int k = 0; k < (int)s.vars.size(); ++k)
                    if (ImGui::Selectable(s.vars[k].c_str(), pr.axis_x == k)) { pr.axis_x = k; s.fit_request = true; }
                ImGui::EndCombo();
            }
            ImGui::SameLine(); ImGui::Text("Y:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(50);
            if (ImGui::BeginCombo("##p3y", s.vars.empty() ? "-" : s.vars[pr.axis_y < (int)s.vars.size() ? pr.axis_y : 0].c_str())) {
                for (int k = 0; k < (int)s.vars.size(); ++k)
                    if (ImGui::Selectable(s.vars[k].c_str(), pr.axis_y == k)) { pr.axis_y = k; s.fit_request = true; }
                ImGui::EndCombo();
            }
            ImGui::SameLine(); ImGui::Text("Z:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(50);
            if (ImGui::BeginCombo("##p3z", s.vars.empty() ? "-" : s.vars[pr.axis_z < (int)s.vars.size() ? pr.axis_z : 0].c_str())) {
                for (int k = 0; k < (int)s.vars.size(); ++k)
                    if (ImGui::Selectable(s.vars[k].c_str(), pr.axis_z == k)) { pr.axis_z = k; s.fit_request = true; }
                ImGui::EndCombo();
            }
        }
        else if (pr.type == ProjType::ContinuationDiagram) {
            const int nv = (int)s.vars.size();
            const bool cont_is_combo = (nv >= 2 && pr.axis_x == nv);
            const std::string preview = s.vars.empty()
                ? std::string("-")
                : (cont_is_combo ? combo_var_label(s.vars)
                                 : s.vars[pr.axis_x < nv ? pr.axis_x : 0]);

            ImGui::Text("var:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(cont_is_combo ? 170.0f : 55.0f);
            if (ImGui::BeginCombo("##pcv", preview.c_str())) {
                for (int k = 0; k < nv; ++k)
                    if (ImGui::Selectable(s.vars[k].c_str(), pr.axis_x == k)) { pr.axis_x = k; s.fit_request = true; }
                if (nv >= 2) {
                    ImGui::Separator();
                    const std::string lbl = combo_var_label(s.vars);
                    if (ImGui::Selectable(lbl.c_str(), cont_is_combo)) { pr.axis_x = nv; s.fit_request = true; }
                }
                ImGui::EndCombo();
            }
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
            ImGui::Text("vars:"); ImGui::SameLine();
            for (int k = 0; k < nv_td; ++k) {
                bool v = pr.show_var[k];
                if (ImGui::Checkbox(s.vars[k].c_str(), &v)) { pr.show_var[k] = v; }
                ImGui::SameLine();
            }
            if (nv_td >= 2) {
                bool v = pr.show_var[(size_t)nv_td];
                if (ImGui::Checkbox(combo_var_label(s.vars).c_str(), &v))
                    pr.show_var[(size_t)nv_td] = v;
                ImGui::SameLine();
            }
            ImGui::NewLine();
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("X")) pr_to_remove = i;
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
