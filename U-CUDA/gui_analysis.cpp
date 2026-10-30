#include "gui_internal.h"

// РЕЖИМ АНАЛИЗА: пространство фазовых портретов

// Панель настроек сессии: параметры (общие), НУ (список), проекции (список), время/шаг,
// метод (заглушка), кнопка пересчёта. Reset lambda — что делает «Reset to defaults»:
// Analysis-tab передаёт model.from_record + start_phase_analysis, Custom-tab nullptr
// (там перезагружаются из System tab или Run pipeline).
// cbp — только ради диалога сохранения netlist'а. Необязателен: в Custom эту же
// панель рисует уровень 3, где callbacks не прокинуты, и кнопка там неактивна.
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
            phase_axis_combo("##px", s, pr.axis_x);
            ImGui::SameLine(); ImGui::Text("Y:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(55);
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
            phase_axis_combo("##p3x", s, pr.axis_x);
            ImGui::SameLine(); ImGui::Text("Y:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(50);
            phase_axis_combo("##p3y", s, pr.axis_y);
            ImGui::SameLine(); ImGui::Text("Z:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(50);
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

// Единый счётчик режимов для статусных строк всех панелей. Раньше каждая панель считала
// "diverged" по flags[k] < 0, и подпись значила разное: у Bif1D/Bif2D отрицательный флаг —
// это fixed point, а у LLE/LS — расходимость. Теперь коды канонические (REGIME_* в
// configCUDA.h) и категории раскладываются одинаково везде.
static void count_regimes(const std::vector<int>& flags, int& fp, int& unb, int& osc, int* stopped = nullptr) {
    fp = unb = osc = 0;
    if (stopped) *stopped = 0;
    for (int f : flags) {
        if      (regime_is_cancelled(f))   { if (stopped) ++*stopped; }
        else if (regime_is_fixed_point(f)) ++fp;
        else if (regime_is_unbound(f))     ++unb;
        else                               ++osc;
    }
}

// Режим ячейки для подсказки хитмапы (HeatmapView::cell_regime). flags — коды режима (сырые или
// REGIME_*, см. regime_code); row_stride — длина строки: n у карт n×n, 0 — один флаг на колонку
// (colored 1D, DFT: строки там — бины Y / частоты одной точки свипа). with_oscillation — подписать
// и колебательные ячейки (карта States, где значение и есть код режима); иначе у них nullptr и
// подсказка показывает само значение. flags ловится по ссылке — колбэк одноразовый и не
// переживает render().
std::function<const char*(int, int)> regime_cell_label(const std::vector<int>& flags,
                                                              int row_stride,
                                                              bool with_oscillation) {
    return [&flags, row_stride, with_oscillation](int ix, int iy) -> const char* {
        const size_t k = (size_t)iy * (size_t)row_stride + (size_t)ix;
        if (k >= flags.size()) return nullptr;
        if (!with_oscillation && regime_is_oscillation(flags[k])) return nullptr;
        return regime_label(flags[k]);
    };
}

// Ячейки, которые не досчитал Stop (HeatmapView::cell_not_computed) — та же адресация, что у
// regime_cell_label.
std::function<bool(int, int)> cancelled_cell_mask(const std::vector<int>& flags, int row_stride) {
    return [&flags, row_stride](int ix, int iy) -> bool {
        const size_t k = (size_t)iy * (size_t)row_stride + (size_t)ix;
        return k < flags.size() && regime_is_cancelled(flags[k]);
    };
}

// Подпись под "OK: ..." — печатается только если есть что показать кроме
// колебательного режима (иначе строка была бы шумом на каждом успешном run'е).
// Сводка адаптивного шага по свипу: st — по 4 числа на точку (принятые, отвергнутые,
// вынужденные шаги, средний h), пусто при постоянном шаге.
void draw_adaptive_stats(const std::vector<double>& st) {
    if (st.size() < 4) return;
    double acc = 0, rej = 0, frc = 0, hsum = 0, hmin = 0, hmax = 0;
    size_t n = 0;
    for (size_t i = 0; i + 3 < st.size(); i += 4) {
        acc += st[i]; rej += st[i + 1]; frc += st[i + 2];
        const double hm = st[i + 3];
        if (hm > 0) {
            if (n == 0 || hm < hmin) hmin = hm;
            if (n == 0 || hm > hmax) hmax = hm;
            hsum += hm; ++n;
        }
    }
    const double att = acc + rej;
    ImGui::TextDisabled("Adaptive step: mean h %.3g (%.3g..%.3g over points), %.0f accepted, "
                        "%.0f rejected (%.1f%%), %.0f forced",
                        n ? hsum / (double)n : 0.0, hmin, hmax, acc, rej,
                        att > 0 ? 100.0 * rej / att : 0.0, frc);
}

// cache — для 2D-карт: обход flags раз на данные (gen + буфер), а не на каждый кадр (см.
// RegimeCounts). Без cache — прямой подсчёт: у 1D-свипов flags короткие.
void draw_regime_summary(const std::vector<int>& flags, RegimeCounts* cache, int gen) {
    int fp = 0, unb = 0, osc = 0, stopped = 0;
    if (!cache) {
        count_regimes(flags, fp, unb, osc, &stopped);
    } else {
        if (cache->gen != gen || cache->data != (const void*)flags.data() || cache->size != flags.size()) {
            count_regimes(flags, cache->fp, cache->unb, cache->osc, &cache->stopped);
            cache->gen = gen; cache->data = flags.data(); cache->size = flags.size();
        }
        fp = cache->fp; unb = cache->unb; osc = cache->osc; stopped = cache->stopped;
    }
    if (stopped > 0)
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f),
                           "Stopped: %d of %d points computed, the rest are grey / blank",
                           (int)flags.size() - stopped, (int)flags.size());
    if (fp == 0 && unb == 0) return;
    ImGui::TextDisabled("regimes: %d %s / %d %s / %d %s",
                        fp,  regime_label(REGIME_FIXED_POINT),
                        unb, regime_label(REGIME_UNBOUND),
                        osc, regime_label(REGIME_OSCILLATION));
}

// Подпись НУ для легенды (галка "Legend shows initial conditions"). Берём ЧИСЛА, реально
// ушедшие в расчёт (result.snapshot.ic_flat), и форматируем через fmt_tick, т.е. с Tick
// precision из Settings: так сгенерированные НУ (узлы сетки бассейнов пишутся с %.17g) не
// разносят легенду на пол-экрана, а смена точности применяется сразу, без пересчёта.
// Fallback на сохранённый result.ic_text — если снапшота нет.
[[nodiscard]] static std::string ic_legend_text(const AnalysisResult& res, size_t k) {
    if (k < res.snapshot.ic_flat.size() && !res.snapshot.ic_flat[k].empty()) {
        const std::vector<double>& v = res.snapshot.ic_flat[k];
        std::string txt = "(";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) txt += ",";
            txt += fmt_tick(v[i]);
        }
        txt += ")";
        return txt;
    }
    return (k < res.ic_text.size()) ? res.ic_text[k] : std::string();
}

void draw_projection_windows(PhaseAnalysisSession& s, const GuiCallbacks& cb,
                                    const ProjHookFn& before_begin,
                                    const ProjHookFn& after_begin,
                                    const std::string& title_suffix,
                                    int owner_id_delta,
                                    const PhaseStyleFn& style_fn,
                                    const FeatureClusterParams& clust) {
    const AnalysisResult& res = s.result;
    // Lambda installed on every projection view that has data — right-click "Export data..."
    // writes the full double-precision trajectory set (all coords for all ICs) + a _config.csv
    // sidecar. Ставится и на Phase3D: Plot3DView тоже поддерживает popup_extras, поэтому
    // экспорт доступен во всех трёх типах проекций одинаково.
    const bool phase_busy = s.in_flight;
    auto phase_popup_extras = [&res, &cb, phase_busy]() {
        const bool has_data = res.ok && !res.trajectories.empty();
        draw_export_menu_item(!has_data || phase_busy, cb, [&res](const std::string& p) {
            data_export::export_phase(res, res.snapshot, p);
        });
    };
    // Свой offscreen-рендерер (FBO/текстура) на КАЖДУЮ проекцию: иначе все окна показывали бы
    // одну общую текстуру (геометрию последней отрисованной). PlotRenderer некопируемый — храним
    // через unique_ptr и подгоняем под число проекций. Кэш ключуется по owner_id_delta: у функции
    // появился второй потребитель (Basins → фазовые портреты, свой delta на каждый config), и с
    // общим вектором переключение между ними пересоздавало бы FBO каждый кадр; Analysis передаёт 0.
    // Давно не рисовавшиеся владельцы вытесняются: каждый держит по FBO на проекцию (мегабайты
    // видеопамяти), а число систем/config'ов не ограничено.
    struct RendererBucket {
        std::vector<std::unique_ptr<PlotRenderer>> v;
        int last_frame = 0;
    };
    static std::map<int, RendererBucket> renderers_by_owner;
    const int cur_frame = ImGui::GetFrameCount();
    for (auto it = renderers_by_owner.begin(); it != renderers_by_owner.end(); ) {
        if (it->first != owner_id_delta &&
            cur_frame - it->second.last_frame > kRendererEvictFrames)
            it = renderers_by_owner.erase(it);
        else
            ++it;
    }
    RendererBucket& bucket = renderers_by_owner[owner_id_delta];
    bucket.last_frame = cur_frame;
    std::vector<std::unique_ptr<PlotRenderer>>& renderers = bucket.v;
    if ((int)renderers.size() != (int)s.projections.size()) {
        renderers.clear();
        for (size_t k = 0; k < s.projections.size(); ++k)
            renderers.push_back(std::make_unique<PlotRenderer>());
    }
    // Запрос автоскейла раздаём по проекциям и сразу гасим сессионный: дальше
    // каждая проекция отвечает за свой флаг сама (см. Projection::fit_pending).
    if (s.fit_request) {
        for (auto& p : s.projections) p.fit_pending = true;
        s.fit_request = false;
    }
    int pr_to_remove = -1;
    for (int i = 0; i < (int)s.projections.size(); ++i) {
        Projection& pr = s.projections[i];
        PlotRenderer& renderer = *renderers[i]; // рендерер этой проекции
        // ε-окружность включает ТОЛЬКО ветка FeatureDiagram ниже. Гасим её здесь каждый кадр:
        // view2d переживает смену типа проекции (объект переиспользуется, см. `if (!pr.view2d)`),
        // и без сброса кружок остался бы висеть на фазовом портрете после переключения комбо.
        if (pr.view2d) pr.view2d->hover_circle_r = std::numeric_limits<double>::quiet_NaN();
        std::string title = pr.label + "##proj" + std::to_string(i) + "_g" + std::to_string(s.layout_generation) + title_suffix;
        bool open = true; // крестик закрытия
        // Начальные позиция и размер (только при первом появлении).
        // Каскад слева-сверху: каждое следующее окно чуть смещено.
        // ИЗМЕНИТЬ РАЗМЕР МОЖНО ЗДЕСЬ: ImVec2(ширина, высота) в пикселях.
        float ox = 60.0f + (i % 5) * 35.0f;
        float oy = 80.0f + (i % 5) * 35.0f;
        ImGui::SetNextWindowPos(ImVec2(ox, oy), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(700, 550), ImGuiCond_FirstUseEver);
        if (before_begin) before_begin(i, title);
        if (ImGui::Begin(title.c_str(), &open)) {
            if (after_begin) after_begin(i, title);
            ImGui::PushID(i);   // разделить ID внутренних виджетов между окнами проекций
            if (!res.ok || res.trajectories.empty()) {
                ImGui::TextDisabled("No data. Press Recompute.");
            }
            ////////////////////////////////////////
            else if (pr.type == ProjType::Phase2D) {
                int ax = pr.axis_x, ay = pr.axis_y;
                if (!res.ok || res.trajectories.empty()) {
                    ImGui::TextDisabled("No data.");
                }
                else {
                    // создать вьюер при первой отрисовке
                    if (!pr.view2d) pr.view2d = std::make_unique<Plot2DView>();

                    // обновить имена осей
                    pr.view2d->x_axis.name = s.vars.empty() ? "x" : phase_axis_name(s, ax);
                    pr.view2d->y_axis.name = s.vars.empty() ? "y" : phase_axis_name(s, ay);
                    // No zero-axis on phase 2D — x=0 / y=0 have no meaning
                    // for a state-space trajectory.
                    pr.view2d->show_zero_x = false;
                    pr.view2d->show_zero_y = false;

                    // Toolbar над плотом: opt-in custom line styling (ImDrawList-путь
                    // с настраиваемой толщиной + α). По дефолту выключено → быстрый
                    // GL shader-line путь (1px, α=1). Текущая отрисовка не ломается.
                    draw_proj_line_style(pr, "phase2d",
                        "Draw iterates as markers instead of a polyline.\n"
                        "Default for discrete maps: consecutive iterates are\n"
                        "not joined by a continuous path, so the connecting\n"
                        "segments are an artefact.");

                    // Серии: для каждой траектории берём координаты по (ax, ay). Буфер
                    // локальный — render() ниже забирает точки синхронно. static тут был
                    // бесполезен (clear()+resize() всё равно убивает capacity) и опасен:
                    // три вызова draw_projection_windows делили бы один буфер.
                    std::vector<std::vector<double>> series_data(res.trajectories.size());

                    std::vector<PlotSeriesInput> series_in;
                    series_in.reserve(res.trajectories.size());

                    // видимость: глобальная — из живых галочек НУ (меняется без recompute)
                    std::vector<bool> init_vis(res.trajectories.size(), true);
                    std::vector<bool> glob_vis(res.trajectories.size(), true);

                    for (size_t k = 0; k < res.trajectories.size(); ++k) {
                        const auto& traj = res.trajectories[k];
                        auto& buf = series_data[k];

                        // Per-IC стиль (Basins → фазовые портреты). as_point:
                        // траектория бассейна-равновесия после транзиента стоит
                        // в одной точке, поэтому кладём в буфер ТОЛЬКО последнюю
                        // — один маркер вместо десятков тысяч совпадающих точек.
                        PhaseSeriesStyle st;
                        const bool has_style = style_fn && style_fn((int)k, st);
                        const bool as_point  = has_style && st.as_point;

                        if (as_point) {
                            if (!traj.empty()) {
                                const auto& pt = traj.back();
                                buf.push_back((double)pt[ax < (int)pt.size() ? ax : 0]);
                                buf.push_back((double)pt[ay < (int)pt.size() ? ay : 0]);
                            }
                        } else {
                            buf.reserve(traj.size() * 2);
                            for (const auto& pt : traj) {
                                buf.push_back((double)pt[ax < (int)pt.size() ? ax : 0]);
                                buf.push_back((double)pt[ay < (int)pt.size() ? ay : 0]);
                            }
                        }
                        std::string lab = (k < res.labels.size()) ? res.labels[k] : ("IC " + std::to_string(k + 1));
                        if (s.legend_show_ic) lab = ic_legend_text(res, k);

                        PlotSeriesInput si;
                        si.points = buf.data();
                        si.n_points = (int)(buf.size() / 2);
                        si.color = has_style ? st.color : ic_base_color((int)k);
                        // В custom_line_style режиме применяем α к цвету IC
                        // (rendering: ImDrawList использует color.w как alpha).
                        if (pr.custom_line_style) si.color.w = pr.alpha;
                        if (as_point) {
                            si.points_override = 1;
                            si.point_marker    = st.marker;
                            si.point_size_px   = st.point_size;
                        }
                        si.label = lab;
                        series_in.push_back(si);

                        bool vis = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;
                        glob_vis[k] = vis;
                        init_vis[k] = true;
                    }

                    int data_gen = s.data_generation * 100 + ax * 10 + ay;

                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();

                    pr.view2d->popup_extras = phase_popup_extras;
                    pr.view2d->render(renderer, origin, avail, i ^ owner_id_delta, data_gen,
                        series_in, init_vis, glob_vis, pr.fit_pending);
                }
            }
            // Диаграмма признаков: точечный график (значение пика; интервал до
            // него) по переменной pr.axis_x. Пики уже посчитаны на worker'е
            // (AnalysisResult::features) тем же алгоритмом, что и GPU-peakFinder,
            // поэтому здесь только раскладка в буферы — пересчёта нет.
            else if (pr.type == ProjType::FeatureDiagram) {
                const int av = pr.axis_x;
                if (!res.ok || res.features.empty()) {
                    ImGui::TextDisabled("No data.");
                }
                else {
                    if (!pr.view2d) pr.view2d = std::make_unique<Plot2DView>();

                    // Оси кластеризации: те же множители, что уходят в dbscan.
                    // Без DBSCAN-конфига (clust.valid == false) остаются 1 и 1,
                    // т.е. сырые признаки, как было.
                    const double mp = clust.valid ? clust.mult_peak     : 1.0;
                    const double mi = clust.valid ? clust.mult_interval : 1.0;
                    auto mult_suffix = [](double m) -> std::string {
                        if (m == 1.0) return std::string();
                        char b[32]; std::snprintf(b, sizeof(b), " x %g", m);
                        return std::string(b);
                    };

                    // av == vars.size() — комбинация (см. комбо-бокс выше).
                    const int nv_ax = (int)s.vars.size();
                    std::string ax_name;
                    if (s.vars.empty())                     ax_name = "x";
                    else if (nv_ax >= 2 && av == nv_ax)     ax_name = combo_var_label(s.vars);
                    else                                    ax_name = s.vars[av < nv_ax ? av : 0];

                    pr.view2d->x_axis.name = "peak " + ax_name + mult_suffix(mp);
                    pr.view2d->y_axis.name = "IPI" + mult_suffix(mi);   // interpeak interval
                    // Интервал неотрицателен, а пики часто лежат вокруг нуля —
                    // нулевая линия по Y тут осмысленна, по X нет.
                    pr.view2d->show_zero_x = false;
                    pr.view2d->show_zero_y = true;
                    // ε-окружность под курсором: радиус eps в ТЕХ ЖЕ осях, в которых
                    // уложены точки (множители уже применены), поэтому накрытые ею пики —
                    // то, что dbscan сольёт в один кластер. Без DBSCAN-конфига радиуса нет
                    // → NaN, ничего не рисуется. Ставится каждый кадр: immediate mode.
                    pr.view2d->hover_circle_r = (clust.valid && clust.eps > 0.0)
                        ? clust.eps
                        : std::numeric_limits<double>::quiet_NaN();

                    draw_style_toolbar("Custom point style", "featdiag", pr.custom_line_style,
                        [&pr]() {
                            bool ch = false;
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Point size##featdiag", &pr.line_width, 0.5f, 8.0f, "%.1f");
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Alpha##featdiag",      &pr.alpha,      0.0f, 1.0f, "%.2f");
                            return ch;
                        });
                    // Всегда точки, линий между пиками нет: соседние пики
                    // соединять нечем — это облако признаков, а не траектория.
                    pr.view2d->imdraw_lines = false;

                    // Локальный буфер — см. series_data в ветке Phase2D выше.
                    std::vector<std::vector<double>> feat_data(res.features.size());
                    std::vector<PlotSeriesInput> series_in;
                    series_in.reserve(res.features.size());
                    std::vector<bool> init_vis(res.features.size(), true);
                    std::vector<bool> glob_vis(res.features.size(), true);

                    size_t total_pts = 0;
                    for (size_t k = 0; k < res.features.size(); ++k) {
                        auto& buf = feat_data[k];
                        const auto& per_var = res.features[k];
                        if (av >= 0 && av < (int)per_var.size()) {
                            const FeaturePoints& fp = per_var[(size_t)av];
                            const size_t n = fp.peaks.size() < fp.intervals.size()
                                           ? fp.peaks.size() : fp.intervals.size();
                            buf.reserve(n * 2);
                            for (size_t p = 0; p < n; ++p) {
                                if (!std::isfinite(fp.peaks[p]) || !std::isfinite(fp.intervals[p]))
                                    continue;
                                buf.push_back((double)(fp.peaks[p]     * mp));
                                buf.push_back((double)(fp.intervals[p] * mi));
                            }
                            total_pts += buf.size() / 2;
                        }

                        std::string lab = (k < res.labels.size()) ? res.labels[k]
                                                                  : ("IC " + std::to_string(k + 1));
                        if (s.legend_show_ic) lab = ic_legend_text(res, k);

                        PhaseSeriesStyle st;
                        const bool has_style = style_fn && style_fn((int)k, st);

                        PlotSeriesInput si;
                        si.points   = buf.empty() ? nullptr : buf.data();
                        si.n_points = (int)(buf.size() / 2);
                        si.color    = has_style ? st.color : ic_base_color((int)k);
                        if (pr.custom_line_style) si.color.w = pr.alpha;
                        si.points_override = 1;
                        si.point_marker    = (int)PointMarker::Circle;
                        si.point_size_px   = pr.custom_line_style ? pr.line_width : 3.0f;
                        si.label = lab;
                        series_in.push_back(si);

                        glob_vis[k] = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;
                        init_vis[k] = true;
                    }

                    if (total_pts == 0)
                        ImGui::TextDisabled("No peaks found (check transient / peak thresholds in Settings).");

                    // Множители и eps входят в gen-токен: иначе Plot2DView отдал бы
                    // уже залитый VBO, и правка коэффициентов не меняла бы картинку —
                    // ровно тот класс «настройка молча не применяется», что уже
                    // случался с --fmad.
                    int data_gen = s.data_generation * 100 + av;
                    if (clust.valid) {
                        auto mix = [](int acc, double v) {
                            char b[32]; std::snprintf(b, sizeof(b), "%.9g", v);
                            for (const char* p = b; *p; ++p)
                                acc = acc * 131 + (int)(unsigned char)*p;
                            return acc;
                        };
                        data_gen = mix(mix(mix(data_gen, clust.mult_peak),
                                           clust.mult_interval), clust.eps);
                    }

                    ImVec2 avail  = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();
                    pr.view2d->popup_extras = phase_popup_extras;
                    pr.view2d->render(renderer, origin, avail, i ^ owner_id_delta, data_gen,
                        series_in, init_vis, glob_vis, pr.fit_pending);
                }
            }
            else if (pr.type == ProjType::TimeDomain) {
                if (!res.ok || res.trajectories.empty()) {
                    ImGui::TextDisabled("No data.");
                }
                else {
                    if (!pr.view2d) pr.view2d = std::make_unique<Plot2DView>();
                    // No zero-axis for time-domain — the X-axis is time,
                    // y=0 rarely coincides with a meaningful reference.
                    pr.view2d->show_zero_x = false;
                    pr.view2d->show_zero_y = false;

                    // Toolbar над плотом: opt-in custom line styling (ImDrawList-путь
                    // с настраиваемой толщиной + α). Дефолт — быстрый GL shader-line
                    // путь (1px, α=1). Аналогично Phase2D.
                    draw_proj_line_style(pr, "timedomain",
                        "Draw the samples as markers instead of a polyline\n"
                        "(on step nodes: where the adaptive step actually landed).");

                    double h = parse_ratio_or(s.step_h, 0.01);  if (h <= 0) h = 0.01;
                    int dec = parse_int_or(s.decimation, 1);    if (dec < 1) dec = 1;
                    double dt = h * dec;
                    int nvars = (int)s.vars.size();

                    // Слотов nvars + 1: последний — комбинация (см. блок галочек).
                    if ((int)pr.show_var.size() != nvars + 1) {
                        std::vector<bool> prev = pr.show_var;
                        pr.show_var.assign((size_t)nvars + 1, true);
                        for (size_t j = 0; j < prev.size() && j < (size_t)nvars; ++j)
                            pr.show_var[j] = prev[j];
                        pr.show_var[(size_t)nvars] = false;
                    }

                    // step_h == 1 for a map, so dt above equals decimation: the
                    // X axis already counts iterations, only the label changes.
                    pr.view2d->x_axis.name = s.sys.is_map ? "n" : "t";
                    pr.view2d->y_axis.name = "value";

                    pr.view2d->pad_x = false;
                    pr.view2d->show_zero_x = false;

                    // серии: одна на (траектория k, видимая переменная vi).
                    // Локальный буфер — см. ветку Phase2D выше.
                    std::vector<std::vector<double>> series_data;

                    std::vector<PlotSeriesInput> series_in;
                    std::vector<bool> init_vis;
                    std::vector<bool> glob_vis;

                    for (size_t k = 0; k < res.trajectories.size(); ++k) {
                        const auto& traj = res.trajectories[k];
                        if (traj.empty()) continue;
                        int n = (int)traj.size();
                        // видимость НУ — из живой галочки (без recompute)
                        bool ic_vis = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;

                        // Per-IC цвет (Basins → фазовые портреты). as_point тут
                        // намеренно игнорируется: во временной развёртке даже
                        // равновесие — это линия (константа), маркер не нужен.
                        PhaseSeriesStyle st;
                        const bool has_style = style_fn && style_fn((int)k, st);

                        // vi == nvars — слот комбинации; для dim < 2 он не
                        // предлагается и пропускается.
                        for (int vi = 0; vi <= nvars; ++vi) {
                            const bool is_combo = (vi == nvars);
                            if (is_combo && nvars < 2) continue;
                            if (vi < (int)pr.show_var.size() && !pr.show_var[vi]) continue;

                            series_data.emplace_back();
                            auto& buf = series_data.back();
                            buf.reserve(n * 2);
                            // Узлы адаптивного шага идут неравномерно: время — из
                            // результата, а не t * dt.
                            const std::vector<double>* tk =
                                (res.raw_nodes && k < res.times.size()) ? &res.times[k] : nullptr;
                            for (int t = 0; t < n; ++t) {
                                buf.push_back(tk && t < (int)tk->size() ? (*tk)[(size_t)t] : (double)(t * dt));
                                buf.push_back((double)(is_combo
                                    ? combo_var_value(traj[t], nvars)
                                    : traj[t][vi < (int)traj[t].size() ? vi : 0]));
                            }

                            std::string base = (k < res.labels.size()) ? res.labels[k] : ("IC" + std::to_string(k + 1));
                            std::string who = s.legend_show_ic ? ic_legend_text(res, k) : base;
                            std::string lab = (is_combo ? combo_var_label(s.vars) : s.vars[vi])
                                            + " [" + who + "]";

                            // Оттенок комбинации берём из расширенной палитры
                            // (nvars + 1), чтобы цвета самих переменных остались
                            // теми же, что были до появления слота.
                            const int shade_den = is_combo ? nvars + 1 : nvars;
                            PlotSeriesInput si;
                            si.points = buf.data();
                            si.n_points = n;
                            si.color = has_style ? shade_of(st.color, vi, shade_den)
                                                 : ic_var_shade((int)k, vi, shade_den);
                            // В custom_line_style режиме применяем α к цвету
                            // (ImDrawList использует color.w как alpha).
                            if (pr.custom_line_style) si.color.w = pr.alpha;
                            si.label = lab;
                            series_in.push_back(si);
                            init_vis.push_back(true);   // локальная (легенда) стартует с видимости НУ
                            glob_vis.push_back(ic_vis);   // глобальная = живая галочка НУ
                        }
                    }

                    int data_gen = s.data_generation * 1000 + (int)series_in.size();

                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();

                    pr.view2d->popup_extras = phase_popup_extras;
                    pr.view2d->render(renderer, origin, avail, i ^ owner_id_delta, data_gen,
                        series_in, init_vis, glob_vis, pr.fit_pending);
                }
            }
            // Recurrence plot + RQA. Матрица считается на GPU в воркере вместе с траекториями
            // (см. rqa::compute в hostLibrary.cu); здесь только показ. Переключение
            // Distance <-> Recurrence GPU не трогает: бинарная матрица получается порогом по
            // уже посчитанной dist, поэтому оно мгновенное.
            else if (pr.type == ProjType::RecurrencePlot) {
                if (!pr.viewrp) {
                    pr.viewrp = std::make_unique<HeatmapView>();
                    // Тот же приём, что у get_or_create_heatmap: сохранённый в проекции выбор
                    // имеет приоритет, -1 отдаёт дефолт. Общий model.heatmap_colormap сюда не
                    // тянем — он шарится Bif/LLE/LS, и RQA к ним не привязана (как и Basins).
                    pr.viewrp->colormap =
                        (HeatmapColormap)colormap_id_or(pr.rqa_colormap, (int)kDefaultColormap);
                    pr.viewrp->reverse_colormap = pr.rqa_reverse_cmap;
                    pr.viewrp->autoscale         = pr.rqa_autoscale;
                    pr.viewrp->manual_vmin_text  = pr.rqa_cbar_vmin_text;
                    pr.viewrp->manual_vmax_text  = pr.rqa_cbar_vmax_text;
                }

                // "(?)" с тултипом — в gui.cpp нет общего HelpMarker'а, идиома тут
                // IsItemHovered + SetTooltip (см. подсказки к схемам КРС и к шкале дБ).
                auto hint = [](const char* t) {
                    ImGui::SameLine(); ImGui::TextDisabled("(?)");
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", t);
                };

                RqaJob job;
                const bool have_job = make_rqa_job(pr, job);
                const rqa::Result* rq = have_job ? find_rqa(res, job) : nullptr;

                // ---------------- toolbar ----------------
                ImGui::SetNextItemWidth(150);
                const char* vmnames[] = { "Distance matrix", "Recurrence plot" };
                ImGui::Combo("##rqvm", &pr.rqa_view_mode, vmnames, IM_ARRAYSIZE(vmnames));
                ImGui::SameLine();
                if (ImGui::Button("RQA settings...")) ImGui::OpenPopup("rqa_cfg");
                ImGui::SameLine();
                if (!have_job) {
                    ImGui::TextDisabled("%s", "choose a state-vector source in the Projections list");
                }
                else if (!rq) {
                    // Через "%s": TextDisabled принимает формат, и текст сюда приходит переменной.
                    ImGui::TextDisabled("%s", s.in_flight ? "computing..."
                                                          : "no result for these settings - press Recompute");
                }
                else if (!rq->ok) {
                    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", rq->error.c_str());
                }
                else {
                    // n и dt печатаются всегда: ряд прорежен до n точек, и длины линий ниже —
                    // В ОТСЧЁТАХ этой выборки, а не в секундах.
                    ImGui::Text("n=%d   dt=%.4g s   eps=%.5g   RR=%.3f%%",
                                rq->n, rq->dt, rq->eps_used, 100.0 * rq->metrics.RR);
                }

                // ---------------- параметры ----------------
                if (ImGui::BeginPopup("rqa_cfg")) {
                    bool ch = false;
                    ImGui::SeparatorText("Embedding");
                    const char* snames[] = { "choose source...", "Full state vector", "Delay embedding" };
                    ImGui::SetNextItemWidth(180);
                    int si = (int)pr.rqa_source;
                    if (ImGui::Combo("source", &si, snames, IM_ARRAYSIZE(snames))) {
                        pr.rqa_source = (rqa::Source)si; ch = true;
                    }
                    if (pr.rqa_source == rqa::Source::Embedding) {
                        const int nv = (int)s.vars.size();
                        ImGui::SetNextItemWidth(180);
                        if (ImGui::BeginCombo("variable", s.vars.empty() ? "-" : s.vars[pr.rqa_var < nv ? pr.rqa_var : 0].c_str())) {
                            for (int k = 0; k < nv; ++k)
                                if (ImGui::Selectable(s.vars[k].c_str(), pr.rqa_var == k)) { pr.rqa_var = k; ch = true; }
                            ImGui::EndCombo();
                        }
                        ch |= InputNumStr("m (dimension)", pr.rqa_m_text, 100.0f);
                        ch |= InputNumStr("tau (samples)", pr.rqa_tau_text, 100.0f);
                    }

                    ImGui::SeparatorText("Threshold");
                    const char* nnames[] = { "Euclidean (L2)", "Maximum (Linf)", "Manhattan (L1)" };
                    ImGui::SetNextItemWidth(180);
                    int ni = (int)pr.rqa_norm;
                    if (ImGui::Combo("norm", &ni, nnames, IM_ARRAYSIZE(nnames))) { pr.rqa_norm = (rqa::Norm)ni; ch = true; }
                    const char* enames[] = { "absolute", "fraction of max distance",
                                             "fraction of sigma", "target recurrence rate" };
                    ImGui::SetNextItemWidth(180);
                    int ei = (int)pr.rqa_eps_mode;
                    if (ImGui::Combo("eps mode", &ei, enames, IM_ARRAYSIZE(enames))) { pr.rqa_eps_mode = (rqa::EpsMode)ei; ch = true; }
                    if (pr.rqa_eps_mode == rqa::EpsMode::Absolute)
                        ch |= InputNumStr("eps", pr.rqa_eps_text, 100.0f);
                    else if (pr.rqa_eps_mode == rqa::EpsMode::TargetRR)
                        ch |= InputNumStr("target RR, %", pr.rqa_rr_text, 100.0f);
                    else
                        ch |= InputNumStr("fraction", pr.rqa_eps_frac_text, 100.0f);

                    ImGui::SeparatorText("Colors");
                    // Отдельно от `ch`: цвет — это про отрисовку, пересчитывать из-за него RQA
                    // незачем (и в continuation это был бы лишний кадр GPU на каждый клик).
                    int cmap_idx = (int)pr.viewrp->colormap;
                    if (colormap_combo("colormap", &cmap_idx, ImGui::GetFontSize() * 12.0f)) {
                        pr.viewrp->colormap = (HeatmapColormap)cmap_idx;
                        pr.rqa_colormap = cmap_idx;
                    }
                    ImGui::Checkbox("reverse colormap", &pr.viewrp->reverse_colormap);
                    hint("Inverts the scale. For the distance matrix this switches which end of "
                         "the scale reads as closeness of points; for the binary one, which level is dark.");

                    ImGui::Checkbox("autoscale color", &pr.viewrp->autoscale);
                    hint("On: the scale is taken from the data - 0..max(D) for the distance matrix and "
                         "0..1 for the binary one. Off: the bounds are set manually in the fields below.\n"
                         "A manual range is useful for comparing two matrices on one scale, but "
                         "on the binary view it breaks the match between levels 0/1 and the colorbar bands.");
                    if (!pr.viewrp->autoscale) {
                        // Ровно как в draw_heatmap_toolbar: текст парсится каждый кадр, значение
                        // кладётся в float-поля вью (parse_ratio_or понимает и дроби вида 1/3).
                        InputNumStr("vmin", pr.viewrp->manual_vmin_text, 100.0f);
                        pr.viewrp->manual_vmin =
                            (float)parse_ratio_or(pr.viewrp->manual_vmin_text, pr.viewrp->manual_vmin);
                        InputNumStr("vmax", pr.viewrp->manual_vmax_text, 100.0f);
                        pr.viewrp->manual_vmax =
                            (float)parse_ratio_or(pr.viewrp->manual_vmax_text, pr.viewrp->manual_vmax);
                    }

                    ImGui::SeparatorText("Lines");
                    ch |= InputNumStr("Theiler window", pr.rqa_theiler_text, 100.0f);
                    hint("Pairs with |i-j| <= w are excluded entirely. For flows this is "
                        "mandatory: without the window, points adjacent in time give false diagonals and inflate DET/LAM.");
                    ch |= InputNumStr("l_min", pr.rqa_lmin_text, 100.0f);
                    ch |= InputNumStr("v_min", pr.rqa_vmin_text, 100.0f);

                    ImGui::SeparatorText("Sampling");
                    ch |= InputNumStr("RP points (n)", pr.rqa_points_text, 100.0f);
                    hint("Side of the matrix. The trajectory is decimated uniformly to n points; the metrics "
                        "are computed on THE SAME matrix that is drawn. The ceiling is 4096 (4096^2 doubles = "
                        "134 MB both on the GPU and in host memory).\n"
                        "Measured on an RTX: 512 ~ 3 ms, 1024 ~ 6 ms, 2048 ~ 20 ms, 4096 ~ 70 ms.");
                    ch |= ImGui::Checkbox("network measures (clustering, transitivity)", &pr.rqa_network);
                    hint("Counting triangles of the recurrence network: O(RR*n^3). Measured: +30 ms at "
                        "n=2048, +85 ms at n=4096 on top of the usual computation. That is why it is off by default.");
                    ch |= ImGui::Checkbox("compute in continuation mode", &pr.rqa_in_continuation);
                    hint("In continuation a frame comes once every ~50 ms, and RQA at n=2048 costs ~20 ms of that "
                        "(at 4096 it is 70 ms, i.e. the frame no longer holds). If it does not fit, clear the "
                        "checkbox: trajectories keep being computed, RQA freezes until the next ordinary Recompute.");

                    if (ch) s.rqa_dirty = true;
                    ImGui::EndPopup();
                }

                // Инверсию можно переключить и из right-click меню хитмапы, поэтому её текущее
                // состояние забираем из вью, а не наоборот (см. комментарий у rqa_reverse_cmap).
                pr.rqa_reverse_cmap = pr.viewrp->reverse_colormap;
                pr.rqa_autoscale    = pr.viewrp->autoscale;
                pr.rqa_cbar_vmin_text = pr.viewrp->manual_vmin_text;
                pr.rqa_cbar_vmax_text = pr.viewrp->manual_vmax_text;

                if (!rq || !rq->ok) {
                    ImGui::TextDisabled("No RQA data.");
                }
                else {
                    // ---------------- метрики ----------------
                    // NaN печатается прочерком, а не нулём: "линий такой длины не нашлось" и
                    // "линии есть, но метрика равна нулю" — разные утверждения.
                    if (ImGui::CollapsingHeader("Metrics", ImGuiTreeNodeFlags_DefaultOpen)) {
                        const rqa::Metrics& M = rq->metrics;
                        struct Row { const char* name; double val; const char* tip; };
                        const Row rows[] = {
                            { "RR",     M.RR,     "Recurrence rate: fraction of recurrent pairs outside the Theiler window" },
                            { "DET",    M.DET,    "Determinism: fraction of points lying on diagonals of length >= l_min" },
                            { "L",      M.L,      "Mean diagonal line length, in samples" },
                            { "L_max",  M.L_max,  "Longest diagonal (excluding the LOI), in samples" },
                            { "DIV",    M.DIV,    "1 / L_max" },
                            { "ENTR",   M.ENTR,   "Shannon entropy of the diagonal length distribution, nats" },
                            { "RATIO",  M.RATIO,  "DET / RR" },
                            { "LAM",    M.LAM,    "Laminarity: fraction of points on verticals of length >= v_min" },
                            { "TT",     M.TT,     "Trapping time: mean vertical, in samples" },
                            { "V_max",  M.V_max,  "Longest vertical, in samples" },
                            { "V_ENTR", M.V_ENTR, "Entropy of the vertical length distribution, nats" },
                            { "TREND",  M.TREND,  "Slope of the linear regression of RR on diagonal index (RR per sample)" },
                            { "T1",     M.T1,     "Recurrence time of the first kind, in samples" },
                            { "T2",     M.T2,     "Recurrence time of the second kind (from the starts of vertical blocks)" },
                            { "W",      M.W,      "Mean white vertical length, in samples" },
                            { "W_max",  M.W_max,  "Longest white vertical, in samples" },
                            { "RTE",    M.RTE,    "Recurrence time entropy, normalised to [0,1]" },
                        };
                        if (ImGui::BeginTable("rqa_metrics", 6,
                                ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV)) {
                            for (int r = 0; r < IM_ARRAYSIZE(rows); ++r) {
                                if (r % 3 == 0) ImGui::TableNextRow();
                                ImGui::TableNextColumn();
                                ImGui::TextDisabled("%s", rows[r].name);
                                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", rows[r].tip);
                                ImGui::TableNextColumn();
                                if (std::isnan(rows[r].val)) ImGui::TextDisabled("--");
                                else ImGui::Text("%.5g", rows[r].val);
                            }
                            if (M.network_valid) {
                                ImGui::TableNextRow();
                                ImGui::TableNextColumn(); ImGui::TextDisabled("C");
                                ImGui::TableNextColumn();
                                if (std::isnan(M.clustering)) ImGui::TextDisabled("--");
                                else ImGui::Text("%.5g", M.clustering);
                                ImGui::TableNextColumn(); ImGui::TextDisabled("Trans");
                                ImGui::TableNextColumn();
                                if (std::isnan(M.transitivity)) ImGui::TextDisabled("--");
                                else ImGui::Text("%.5g", M.transitivity);
                            }
                            ImGui::EndTable();
                        }
                    }

                    // ---------------- сама матрица ----------------
                    const int n = rq->n;
                    const double* values = rq->dist.data();
                    double vlo = 0.0, vhi = rq->dist_max;

                    if (pr.rqa_view_mode == 1) {
                        // Бинарная матрица строится порогом по dist и кэшируется в проекции:
                        // GPU для смены режима не нужен. Ключ кэша — data_generation: любое новое
                        // значение dist приходит только вместе с новым расчётом.
                        if (pr.rqa_binary_gen != s.data_generation ||
                            pr.rqa_binary.size() != rq->dist.size()) {
                            pr.rqa_binary.resize(rq->dist.size());
                            for (size_t t = 0; t < rq->dist.size(); ++t)
                                pr.rqa_binary[t] = (rq->dist[t] <= rq->eps_used) ? 1.0 : 0.0;
                            pr.rqa_binary_gen = s.data_generation;
                        }
                        values = pr.rqa_binary.data();
                        vlo = 0.0; vhi = 1.0;
                    }

                    // discrete форсим ТОЛЬКО в кадр смены режима: иначе каждый кадр затирался бы
                    // ручной toggle "Discrete colorbar" из right-click меню HeatmapView.
                    if (pr.rqa_view_mode_applied != pr.rqa_view_mode) {
                        pr.viewrp->discrete = (pr.rqa_view_mode == 1);
                        pr.viewrp->discrete_levels = (pr.rqa_view_mode == 1) ? 2 : 0;
                        pr.viewrp->discrete_levels_text = (pr.rqa_view_mode == 1) ? "2" : "0";
                        pr.rqa_view_mode_applied = pr.rqa_view_mode;
                    }

                    pr.viewrp->x_axis.name = s.sys.is_map ? "n_j" : "t_j, s";
                    pr.viewrp->y_axis.name = s.sys.is_map ? "n_i" : "t_i, s";
                    // Свой экспорт вместо phase_popup_extras: из этого окна нужны матрица и
                    // метрики, а не траектории. Траекторный экспорт оставлен вторым пунктом —
                    // он тут тоже осмыслен (RQA считалась именно по ним).
                    pr.viewrp->popup_extras = [&res, &cb, phase_busy, rq, &job]() {
                        if (ImGui::MenuItem("Export RQA...", nullptr, false, !phase_busy)) {
                            if (cb.pick_save_file_csv) {
                                const std::string path = cb.pick_save_file_csv();
                                if (!path.empty())
                                    data_export::export_rqa(*rq, job.cfg, job.ic, res.snapshot, path);
                            }
                        }
                        const bool has_traj = res.ok && !res.trajectories.empty();
                        draw_export_menu_item(!has_traj || phase_busy, cb, [&res](const std::string& p) {
                            data_export::export_phase(res, res.snapshot, p);
                        });
                    };

                    // Режим входит в data_generation: при переключении Distance <-> Recurrence
                    // текстуру надо перезалить, хотя расчёт тот же.
                    const int data_gen = s.data_generation * 2 + pr.rqa_view_mode;

                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();
                    pr.viewrp->render(renderer, origin, avail,
                                      (i ^ owner_id_delta) ^ 0x5251A0, data_gen,
                                      n, n, values,
                                      rq->t0, rq->t1, rq->t0, rq->t1,
                                      vlo, vhi, pr.fit_pending);
                }
            }
            else if (pr.type == ProjType::Phase3D) {
                int ax = pr.axis_x, ay = pr.axis_y, az = pr.axis_z;
                if (!res.ok || res.trajectories.empty()) {
                    ImGui::TextDisabled("No data.");
                }
                else {
                    if (!pr.view3d) pr.view3d = std::make_unique<Plot3DView>();

                    // Toolbar над плотом: opt-in custom line styling (толщина + α). В 3D нет
                    // ImDrawList-fallback (потерялся бы depth-sorting), толщина идёт через
                    // glLineWidth и может клампиться драйвером, α точно уходит в шейдер.
                    // При выключенной фиче восстанавливаем прежний хардкод 1.5f.
                    draw_style_toolbar("Custom line style", "phase3d", pr.custom_line_style,
                        [&pr]() {
                            bool ch = false;
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Line width##phase3d", &pr.line_width, 0.1f, 5.0f, "%.2f");
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Alpha##phase3d",      &pr.alpha,      0.0f, 1.0f, "%.2f");
                            return ch;
                        });
                    ImGui::SameLine();
                    ImGui::Checkbox("Points##phase3d", &pr.draw_points);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Draw iterates as markers instead of a polyline.\n"
                                          "Default for discrete maps.");
                    if (pr.draw_points) {
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(110);
                        ImGui::SliderFloat("size##phase3dpt", &pr.point_size, 0.5f, 8.0f, "%.1f");
                        if (pr.custom_line_style) {   // своя альфа точек (см. draw_proj_line_style)
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(110);
                            ImGui::SliderFloat("alpha##phase3dpt", &pr.point_alpha, 0.0f, 1.0f, "%.2f");
                        }
                    }
                    pr.view3d->point_alpha = (pr.draw_points && pr.custom_line_style) ? pr.point_alpha : -1.0f;
                    pr.view3d->line_thickness_px = pr.custom_line_style ? pr.line_width : 1.5f;
                    pr.view3d->custom_line_style = pr.custom_line_style;
                    pr.view3d->points_mode       = pr.draw_points;
                    pr.view3d->points_with_lines = pr.draw_points && pr.custom_line_style;
                    pr.view3d->point_size_px     = pr.point_size;

                    pr.view3d->x_name = s.vars.empty() ? "x" : phase_axis_name(s, ax);
                    pr.view3d->y_name = s.vars.empty() ? "y" : phase_axis_name(s, ay);
                    pr.view3d->z_name = s.vars.empty() ? "z" : phase_axis_name(s, az);

                    // Локальный буфер — см. ветку Phase2D выше.
                    std::vector<std::vector<float>> series_data(res.trajectories.size());

                    std::vector<PlotSeriesInput3D> series_in;
                    series_in.reserve(res.trajectories.size());

                    std::vector<bool> init_vis(res.trajectories.size(), true);
                    std::vector<bool> glob_vis(res.trajectories.size(), true);

                    for (size_t k = 0; k < res.trajectories.size(); ++k) {
                        const auto& traj = res.trajectories[k];
                        auto& buf = series_data[k];
                        buf.reserve(traj.size() * 3);
                        for (const auto& pt : traj) {
                            buf.push_back((double)pt[ax < (int)pt.size() ? ax : 0]);
                            buf.push_back((double)pt[ay < (int)pt.size() ? ay : 0]);
                            buf.push_back((double)pt[az < (int)pt.size() ? az : 0]);
                        }
                        std::string lab = (k < res.labels.size()) ? res.labels[k] : ("IC " + std::to_string(k + 1));
                        if (s.legend_show_ic) lab = ic_legend_text(res, k);

                        PlotSeriesInput3D si;
                        si.points = buf.data();
                        si.n_points = (int)(buf.size() / 3);
                        si.color = ic_base_color((int)k);
                        // В custom_line_style режиме применяем α к цвету IC
                        // (draw_line_3d прокидывает color[3] в шейдер как альфа).
                        if (pr.custom_line_style) si.color.w = pr.alpha;
                        si.label = lab;
                        series_in.push_back(si);

                        bool vis = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;
                        glob_vis[k] = vis;
                        init_vis[k] = true;
                    }

                    int data_gen = s.data_generation * 1000 + ax * 100 + ay * 10 + az;

                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();

                    pr.view3d->popup_extras = phase_popup_extras;
                    pr.view3d->render(renderer, origin, avail, i ^ owner_id_delta, data_gen,
                        series_in, init_vis, glob_vis, pr.fit_pending);
                }
            }
            else if (pr.type == ProjType::ContinuationDiagram) {
                const int av = pr.axis_x;
                if (s.continuation_peaks.empty()) {
                    ImGui::TextDisabled("No data. Press Recompute (Continuation (live) accumulates across runs).");
                }
                else {
                    if (!pr.view2d) pr.view2d = std::make_unique<Plot2DView>();

                    const int nv_ax = (int)s.vars.size();
                    std::string ax_name;
                    if (s.vars.empty())                     ax_name = "x";
                    else if (nv_ax >= 2 && av == nv_ax)     ax_name = combo_var_label(s.vars);
                    else                                    ax_name = s.vars[av < nv_ax ? av : 0];

                    pr.view2d->x_axis.name = s.sys.is_map ? "n" : "t, s";
                    pr.view2d->y_axis.name = "peak " + ax_name;
                    pr.view2d->show_zero_x = false;
                    pr.view2d->show_zero_y = true;

                    draw_style_toolbar("Custom point style", "contdiag", pr.custom_line_style,
                        [&pr]() {
                            bool ch = false;
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Point size##contdiag", &pr.line_width, 0.5f, 8.0f, "%.1f");
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Alpha##contdiag",      &pr.alpha,      0.0f, 1.0f, "%.2f");
                            return ch;
                        });
                    pr.view2d->imdraw_lines = false;

                    std::vector<PlotSeriesInput> series_in;
                    series_in.reserve(s.continuation_peaks.size());
                    std::vector<bool> init_vis(s.continuation_peaks.size(), true);
                    std::vector<bool> glob_vis(s.continuation_peaks.size(), true);

                    size_t total_pts = 0;
                    for (size_t k = 0; k < s.continuation_peaks.size(); ++k) {
                        const auto& per_var = s.continuation_peaks[k];
                        const double* pts = nullptr;
                        int          np  = 0;
                        if (av >= 0 && av < (int)per_var.size()) {
                            const auto& buf = per_var[(size_t)av];
                            pts = buf.empty() ? nullptr : buf.data();
                            np  = (int)(buf.size() / 2);
                            total_pts += (size_t)np;
                        }

                        std::string lab = (k < res.labels.size()) ? res.labels[k]
                                                                  : ("IC " + std::to_string(k + 1));
                        if (s.legend_show_ic && k < res.ic_text.size())
                            lab = ic_legend_text(res, k);

                        PlotSeriesInput si;
                        si.points   = pts;
                        si.n_points = np;
                        si.color    = ic_base_color((int)k);
                        if (pr.custom_line_style) si.color.w = pr.alpha;
                        si.points_override = 1;
                        si.point_marker    = (int)PointMarker::Circle;
                        si.point_size_px   = pr.custom_line_style ? pr.line_width : 3.0f;
                        si.label = lab;
                        series_in.push_back(si);

                        glob_vis[k] = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;
                        init_vis[k] = true;
                    }

                    if (total_pts == 0)
                        ImGui::TextDisabled("Waiting for peaks... (adjust transient / peak thresholds if needed).");

                    int data_gen = s.continuation_peaks_gen * 100 + av;

                    ImVec2 avail  = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();
                    pr.view2d->popup_extras = phase_popup_extras;
                    pr.view2d->render(renderer, origin, avail, i ^ owner_id_delta, data_gen,
                        series_in, init_vis, glob_vis, pr.fit_pending);
                }
            }
            // h(t) адаптивного шага: принятые шаги линией, отвергнутые попытки —
            // крестиками, вынужденные (на h_min) — квадратами. Данные — лог попыток.
            else if (pr.type == ProjType::StepSize) {
                if (!res.ok || !res.adaptive || res.step_log.empty()) {
                    ImGui::TextDisabled("No data. The step-size plot needs the adaptive step "
                                        "(Step size -> Adaptive step).");
                }
                else {
                    if (!pr.view2d) pr.view2d = std::make_unique<Plot2DView>();
                    // log Y — пересчётом в log10 h (лог-оси Y у Plot2DView нет), как у окон Order.
                    const bool prev_ylog = pr.y_log;
                    ImGui::Checkbox("log Y##stepsize", &pr.y_log);
                    if (pr.y_log != prev_ylog) pr.fit_pending = true;
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Y axis in log10 h: steps spread over orders of magnitude\n"
                                          "(stiff parts, rejections near h_min) become readable.");
                    ImGui::SameLine();
                    draw_proj_line_style(pr, "stepsize",
                        "Draw the accepted steps as markers instead of a line.");
                    pr.view2d->x_axis.name = "t";
                    pr.view2d->y_axis.name = pr.y_log ? "log10 h" : "h";
                    pr.view2d->show_zero_x = false;
                    pr.view2d->show_zero_y = false;

                    std::vector<std::vector<double>> bufs;
                    bufs.reserve(res.step_log.size() * 3);   // указатели серий не должны переехать
                    std::vector<PlotSeriesInput> series_in;
                    std::vector<bool> init_vis, glob_vis;
                    for (size_t k = 0; k < res.step_log.size(); ++k) {
                        const std::vector<double>& lg = res.step_log[k];
                        std::vector<double> acc, rej, frc;
                        for (size_t r = 0; r + 4 <= lg.size(); r += 4) {
                            const double t = lg[r], code = lg[r + 3];
                            double hh = lg[r + 1];
                            if (pr.y_log) {
                                if (!(hh > 0)) continue;   // log10 не определён
                                hh = std::log10(hh);
                            }
                            if (code == 0) { rej.push_back(t); rej.push_back(hh); continue; }
                            acc.push_back(t); acc.push_back(hh);
                            if (code == 2) { frc.push_back(t); frc.push_back(hh); }
                        }
                        const std::string who = (k < res.labels.size()) ? res.labels[k]
                                                                        : ("IC" + std::to_string(k + 1));
                        const bool ic_vis = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;
                        struct Part { std::vector<double>* v; const char* name; int marker; ImVec4 col; };
                        const Part parts[3] = {
                            { &acc, "h",        -1, ic_base_color((int)k) },
                            { &rej, "rejected", (int)PointMarker::Cross,  ImVec4(0.90f, 0.30f, 0.30f, 1.0f) },
                            { &frc, "forced",   (int)PointMarker::Square, ImVec4(1.00f, 0.65f, 0.20f, 1.0f) },
                        };
                        for (const Part& part : parts) {
                            if (part.v->empty()) continue;
                            bufs.push_back(std::move(*part.v));
                            PlotSeriesInput si;
                            si.points = bufs.back().data();
                            si.n_points = (int)(bufs.back().size() / 2);
                            si.color = part.col;
                            if (part.marker >= 0) {
                                si.points_override = 1;
                                si.point_marker    = part.marker;
                                si.point_size_px   = 4.0f;
                            }
                            else if (pr.custom_line_style) si.color.w = pr.alpha;   // α — линии, не маркерам
                            si.label = std::string(part.name) + " [" + who + "]";
                            series_in.push_back(si);
                            init_vis.push_back(true);
                            glob_vis.push_back(ic_vis);
                        }
                    }
                    int data_gen = s.data_generation * 1000 + (pr.y_log ? 507 : 7);   // log Y — другие данные
                    ImVec2 avail  = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();
                    pr.view2d->popup_extras = phase_popup_extras;
                    pr.view2d->render(renderer, origin, avail, i ^ owner_id_delta, data_gen,
                        series_in, init_vis, glob_vis, pr.fit_pending);
                }
            }
            // Отказы адаптивного шага: у каждого принятого шага — сколько попыток ему
            // отвергли (маркер, шаги без отказов не рисуются; вынужденные — квадратом), и
            // скользящее среднее по kWin принятым шагам — доля лишних вычислений схемы.
            // Точка стоит в начале шага. Данные — лог попыток, как у Step size.
            else if (pr.type == ProjType::Rejections) {
                if (!res.ok || !res.adaptive || res.step_log.empty()) {
                    ImGui::TextDisabled("No data. The rejections plot needs the adaptive step "
                                        "(Step size -> Adaptive step).");
                }
                else {
                    if (!pr.view2d) pr.view2d = std::make_unique<Plot2DView>();
                    draw_proj_line_style(pr, "rejections",
                        "Draw the moving mean as markers instead of a line.");
                    pr.view2d->x_axis.name = "t";
                    pr.view2d->y_axis.name = "rejected per step";
                    pr.view2d->show_zero_x = false;
                    pr.view2d->show_zero_y = false;

                    constexpr int kWin = 100;
                    std::vector<std::vector<double>> bufs;
                    bufs.reserve(res.step_log.size() * 3);   // указатели серий не должны переехать
                    std::vector<PlotSeriesInput> series_in;
                    std::vector<bool> init_vis, glob_vis;
                    for (size_t k = 0; k < res.step_log.size(); ++k) {
                        const std::vector<double>& lg = res.step_log[k];
                        std::vector<double> rej, frc, mean;
                        int cnt = 0, wsum = 0, wn = 0, wpos = 0;
                        int win[kWin] = {};
                        for (size_t r = 0; r + 4 <= lg.size(); r += 4) {
                            const double code = lg[r + 3];
                            if (code == 0) { ++cnt; continue; }
                            const double t = lg[r];
                            if (code == 2) { frc.push_back(t); frc.push_back(cnt); }
                            else if (cnt > 0) { rej.push_back(t); rej.push_back(cnt); }
                            wsum += cnt - (wn == kWin ? win[wpos] : 0);
                            win[wpos] = cnt; wpos = (wpos + 1) % kWin;
                            if (wn < kWin) ++wn;
                            mean.push_back(t); mean.push_back((double)wsum / wn);
                            cnt = 0;
                        }
                        const std::string who = (k < res.labels.size()) ? res.labels[k]
                                                                        : ("IC" + std::to_string(k + 1));
                        const bool ic_vis = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;
                        struct Part { std::vector<double>* v; const char* name; int marker; ImVec4 col; };
                        const Part parts[3] = {
                            { &mean, "mean of 100 steps", -1, ic_base_color((int)k) },
                            { &rej,  "rejected",          (int)PointMarker::Cross,  ImVec4(0.90f, 0.30f, 0.30f, 1.0f) },
                            { &frc,  "forced",            (int)PointMarker::Square, ImVec4(1.00f, 0.65f, 0.20f, 1.0f) },
                        };
                        for (const Part& part : parts) {
                            if (part.v->empty()) continue;
                            bufs.push_back(std::move(*part.v));
                            PlotSeriesInput si;
                            si.points = bufs.back().data();
                            si.n_points = (int)(bufs.back().size() / 2);
                            si.color = part.col;
                            if (part.marker >= 0) {
                                si.points_override = 1;
                                si.point_marker    = part.marker;
                                si.point_size_px   = 4.0f;
                            }
                            else if (pr.custom_line_style) si.color.w = pr.alpha;   // α — линии, не маркерам
                            si.label = std::string(part.name) + " [" + who + "]";
                            series_in.push_back(si);
                            init_vis.push_back(true);
                            glob_vis.push_back(ic_vis);
                        }
                    }
                    int data_gen = s.data_generation * 1000 + 8;
                    ImVec2 avail  = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();
                    pr.view2d->popup_extras = phase_popup_extras;
                    pr.view2d->render(renderer, origin, avail, i ^ owner_id_delta, data_gen,
                        series_in, init_vis, glob_vis, pr.fit_pending);
                }
            }
            ImGui::PopID();
            // Окно реально отрисовалось в этом кадре — автоскейл до него дошёл.
            // Гасим здесь, а не после цикла: у невидимых окон Begin вернул
            // false, и их запрос должен дожить до кадра, когда их покажут.
            // Ветка "No data" тоже гасит: фитить нечего, а приход результата
            // взведёт fit_request заново.
            pr.fit_pending = false;
        }
        ImGui::End();
        if (!open) pr_to_remove = i; // окно закрыто крестиком
    }
    if (pr_to_remove >= 0) s.remove_projection(pr_to_remove);
}
