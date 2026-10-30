#include "gui_internal.h"

// Combo "Writable var": переменные + (через разделитель) комбинация
// x[0] + pi*x[1] + euler*x[2] (при |vars|=2 без e*x[2], при |vars|=1 — сама переменная).
// Sentinel в hostside int: -1 = комбинация. Используется в Bifurcation и Basins, где
// kernel пишет data[i] через loopCalculateDiscreteModel_int — он умеет оба режима.
void draw_writable_var_combo(const std::vector<std::string>& vars,
                                    int& writable_var,
                                    const char* combo_id) {
    if (vars.empty()) return;
    const int N = (int)vars.size();
    // Допустимые значения: -1 (combination) или [0, N-1]. Старые сейвы могли
    // иметь невалидные индексы — clamp в 0.
    if (writable_var < -1 || writable_var >= N) writable_var = 0;
    // Для системы из одной переменной combination сворачивается до x[0] —
    // показывать отдельную опцию бессмысленно, форсим single-var.
    if (N == 1 && writable_var == -1) writable_var = 0;

    // Подпись комбинации — общая combo_var_label (раньше здесь жила её копия).
    const std::string preview =
        (writable_var == -1) ? combo_var_label(vars) : vars[writable_var];

    ImGui::SetNextItemWidth(220);
    if (ImGui::BeginCombo(combo_id, preview.c_str())) {
        for (int i = 0; i < N; ++i) {
            bool sel = writable_var == i;
            if (ImGui::Selectable(vars[i].c_str(), sel)) writable_var = i;
        }
        if (N >= 2) {
            ImGui::Separator();
            const std::string lbl = combo_var_label(vars);
            bool sel_combo = writable_var == -1;
            if (ImGui::Selectable(lbl.c_str(), sel_combo)) writable_var = -1;
        }
        ImGui::EndCombo();
    }
}

// Parametric: контролы + scatter-plot 1D-бифуркации через наш GL-renderer
// Рисует контролы одной БД внутри её таба. Возвращает true, если пользователь
// нажал Run для этой БД (внешний код может также взвести Run через Ctrl+R).
static void draw_diagram_controls(AppModel& model, BifurcationAnalysisSession& s, int idx) {
    BifurcationDiagramConfig& bd = s.diagrams[idx];
    // Чей конфиг правит эта панель — для ПКМ-рассылки "во все такие же".
    model.broadcast_source_tab = BroadcastTab::Bifurcation;
    model.broadcast_source_idx = idx;

    ImGui::SetNextItemWidth(kComboW);
    if (InputTextStr("Label", bd.label))
        bd.label_is_manual = !bd.label.empty();   // empty → back to auto
    ImGui::Separator();

    // Scheme (built-in + custom)
    draw_scheme_combo("Scheme", bd.scheme, s.custom_schemes, {}, &s.enabled_builtin_schemes,
                      &model, model.is_map, &s.wrapper_schemes);
    ImGui::Separator();

    // Sweep target (parameter ИЛИ initial condition): один combo с разделителем — сверху
    // параметры, снизу переменные (IC). Выбор переменной → БД строится по начальному
    // условию (par_or_var = 0 в engine), это runtime-флаг, без пересборки PTX.
    // dt (h) доступен и в continuation: шаг пересчитывается в каждой точке цепочки
    // (run_bif1d_continuation / _cpu). Вторую ось гасим только в 2D-режиме — в 1D флаг
    // sweep_over_h_2 не используется.
    draw_sweep_target_combo("Sweep", s.params, s.vars,
                            bd.param_index, bd.sweep_over_var, bd.var_sweep_index,
                            bd.sweep_over_h,
                            bd.mode_2d ? &bd.sweep_over_h_2 : nullptr,
                            /*note_when_empty*/ true, kComboW,
                            scheme_uses_symmetry(bd.scheme, s.custom_schemes),
                            &model, BroadcastField::SweepTarget, !model.is_map,
                            &bd.ad_axis, &bd.adaptive);
    InputNumStr(bd.sweep_over_h ? "h lo" : "Param lo", bd.param_lo_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepLo, {},
                                        bd.sweep_over_h ? "h lo" : "Param lo", bd.param_lo_text); });
    InputNumStr(bd.sweep_over_h ? "h hi" : "Param hi", bd.param_hi_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepHi, {},
                                        bd.sweep_over_h ? "h hi" : "Param hi", bd.param_hi_text); });
    ImGui::Checkbox("Log scale##bd_log", &bd.log_scale);
    if (bd.log_scale) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }

    // Continuation: каждая следующая точка стартует с конечного x[] предыдущей. Единый блок
    // с LLE и LS 1D (draw_continuation_device_block); он же гасит и сбрасывает флаги в 2D,
    // где run_bif2d идёт своим 3-kernel pipeline'ом и continuation игнорирует — скрытое
    // состояние туда утекать не должно. Классического CPU-свипа у БД нет, поэтому радио
    // GPU/CPU влияет только на continuation.
    draw_continuation_device_block(bd, "bd", bd.mode_2d || bd.sweep_over_var, bd.mode_2d);

    ImGui::Separator();

    // 2D mode: хитмап «период»(p1, p2) через DBSCAN
    if (ImGui::Checkbox("2D mode (period heatmap)", &bd.mode_2d) && bd.mode_2d)
        bd.colored_1d = false;   // взаимоисключающе с Colored 1D diagram
    if (bd.mode_2d) {
        ImGui::Indent();
        // Пункт "dt (h)" прячется, если шаг уже занял X: ровно одна ось = h.
        if (!s.params.empty() || !s.vars.empty())
            draw_sweep_target_combo("Sweep Y", s.params, s.vars,
                                    bd.param_index_2, bd.sweep_over_var_2, bd.var_sweep_index_2,
                                    bd.sweep_over_h_2, &bd.sweep_over_h,
                                    /*note_when_empty*/ false, kComboW,
                                    scheme_uses_symmetry(bd.scheme, s.custom_schemes),
                                    &model, BroadcastField::SweepTarget2, !model.is_map,
                                    &bd.ad_axis_2, &bd.adaptive);
        InputNumStr(bd.sweep_over_h_2 ? "h2 lo" : "Param2 lo", bd.param_lo_2_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepLo2, {},
                                        bd.sweep_over_h_2 ? "h2 lo" : "Param2 lo", bd.param_lo_2_text); });
        InputNumStr(bd.sweep_over_h_2 ? "h2 hi" : "Param2 hi", bd.param_hi_2_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepHi2, {},
                                        bd.sweep_over_h_2 ? "h2 hi" : "Param2 hi", bd.param_hi_2_text); });
        ImGui::Checkbox("Log scale##bd_log2", &bd.log_scale_2);
        if (bd.log_scale_2) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
        InputNumStr("DBSCAN eps", bd.eps_dbscan_text, kFieldW);
        // Множители осей кластеризации. Дефолт mult interval = 0 схлопывает
        // ось интервалов, т.е. период считается только по амплитуде пиков.
        InputNumStr("mult peak##bd_mp",     bd.mult_peak_text, kFieldW);
        InputNumStr("mult interval##bd_mi", bd.mult_interval_text, kFieldW);
        ImGui::TextDisabled("Scale the (peak, interval) axes before clustering;");
        ImGui::TextDisabled("0 disables an axis. See Feature diagram in Phase analysis.");
        ImGui::TextDisabled("Grid is square (Resolution applies to both axes).");
        ImGui::Unindent();
    }

    ImGui::Separator();

    // Variable + resolution + inter-peaks
    draw_writable_var_combo(s.vars, bd.writable_var, "Writable var##bd_wv");
    InputNumStr("Resolution", bd.n_pts_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::Resolution, {},
                                        "Resolution", bd.n_pts_text); });
    // Every iterate vs peaks only. Inter-peaks is meaningless in the first mode
    // (consecutive samples are one step apart by construction), so it is hidden.
    if (ImGui::Checkbox("Plot every iterate (no peak filter)", &bd.plot_all_iterates))
        bd.fit_request = true;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "On: the diagram gets every recorded iterate - the convention for\n"
            "discrete maps, and the default for them.\n"
            "Off: only local maxima (a Poincare section), the convention for flows.\n"
            "For a map the peak filter drops the lower branch of every period-2\n"
            "orbit and leaves the period-1 windows empty.");
    // Адаптивный шаг пишет только пики (ядра адаптивного шага сырых отсчётов не выдают).
    if (bd.plot_all_iterates && bd.adaptive.enabled && !model.is_map) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.3f, 1.0f), "(not with the adaptive step: peaks only)");
    }
    // Inter-peak times are not filled in the all-iterates path (pushRaw writes
    // zeros), so a stale toggle would plot a flat line at 0.
    if (bd.plot_all_iterates) bd.plot_inter_peaks = false;
    if (!bd.mode_2d && !bd.plot_all_iterates)
        if (ImGui::Checkbox("Plot inter-peaks instead of peak values", &bd.plot_inter_peaks))
            bd.fit_request = true;

    ImGui::Separator();

    // ----- Colored 1D diagram (collapsible): density-хитмапа поверх
    // классической БД. Вкл/выкл — чекбоксом в тулбаре над самим плотом
    // (draw_bifurcation_plot), не здесь. Здесь — только настройки, видны
    // когда режим уже активен.
    if (bd.colored_1d && ImGui::CollapsingHeader("Colored 1D diagram##bd_c1d", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextDisabled("Toggle this mode from the plot toolbar.");
        InputNumStr("Y resolution", bd.colored_1d_b_text, kFieldW);
        ImGui::TextDisabled("X resolution follows Resolution above.");
        ImGui::Checkbox("Custom Y range", &bd.colored_1d_custom_y);
        if (bd.colored_1d_custom_y) {
            InputNumStr("Ymin", bd.colored_1d_ymin_text, kFieldW);
            InputNumStr("Ymax", bd.colored_1d_ymax_text, kFieldW);
            ImGui::TextDisabled("Points outside [Ymin, Ymax] are discarded.");
        }
        ImGui::Checkbox("Logarithmic density scale", &bd.colored_1d_log);
    }

    // Integration (collapsible)
    {
        IntegrationFields f;
        f.h           = &bd.h_text;
        f.symmetry_s  = &bd.symmetry_s;
        f.cd_fp_iters = &bd.cd_fp_iters;
        f.newton      = &bd.newton;
        f.t_max       = &bd.t_max_text;
        f.transient   = &bd.transient_text;
        f.pre_scaller = &bd.pre_scaller_text;
        f.max_value   = &bd.max_value_text;
        f.adaptive      = &bd.adaptive;
        f.adaptive_opts = kAdUiRaw | kAdUiInterp;
        draw_integration_block("Integration##bd_int", bd.scheme, s.custom_schemes, f, &model, model.is_map);
    }

    draw_named_num_fields("Initial conditions##bd_ic", s.vars,   bd.initial_conditions,
                          nullptr, &model, BroadcastField::InitCondition);
    draw_named_num_fields("Parameters##bd_par",        s.params, bd.param_values,
                          nullptr, &model, BroadcastField::Param);
    draw_csv_output_block("CSV output##bd_csv", "Save to file", bd.csv_save_enabled,
                          "##csv_path", bd.csv_output_path,
                          "Path is kept even when save is off. Also writes <path>_config.csv.");

    // Run-кнопка живёт на уровне draw_parametric_controls (рядом с Run all),
    // общая для Bif/LLE/LS. Здесь — только статусная строка.

    if (bd.mode_2d) {
        if (bd.last_run_2d_ok) {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: %dx%d heatmap, period(min..max) = %.0f..%.0f",
                bd.result_2d.n_pts, bd.result_2d.n_pts,
                bd.result_2d.min_val, bd.result_2d.max_val);
            draw_regime_summary(bd.result_2d.flags, &bd.regime_counts_2d, bd.data_generation_2d);
            draw_adaptive_stats(bd.result_2d.ad_stats);
        }
        else if (!bd.last_error.empty()) {
            draw_error_box("##par_err_2d", bd.last_error);
        }
    } else {
        if (bd.last_run_ok) {
            // flags[] здесь — сырой выход peakFinder: N > 0 = число пиков.
            int total_peaks = 0, max_peaks = 0;
            for (int f : bd.result.flags)
                if (regime_is_oscillation(f)) {
                    total_peaks += f;
                    if (f > max_peaks) max_peaks = f;
                }
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: n_pts=%d, peaks total=%d (max per param=%d)",
                bd.result.n_pts, total_peaks, max_peaks);
            draw_regime_summary(bd.result.flags);
            draw_adaptive_stats(bd.result.ad_stats);
        }
        else if (!bd.last_error.empty()) {
            draw_error_box("##par_err", bd.last_error);
        }
    }
}

static void draw_bifurcation_controls(AppModel& model, SystemLibrary& /*lib*/) {
    BifurcationAnalysisSession& s = model.bifurcation_session;

    // Tab bar: одна вкладка на БД + кнопка "+" справа (копия последней). Активная вкладка
    // хранится в s.active_diagram_index и используется Ctrl+R. request_select_diagram —
    // внешний запрос выбрать вкладку от тулбара Colored 1D над плотом, иначе галка
    // включалась бы у одной БД, а настройки показывались для другой.
    // Run-кнопка и Ctrl+R живут в draw_parametric_controls (общие для Bif/LLE/LS).
    const TabBarResult tabs = draw_config_tab_bar(
        "##bd_tabs", "bd_tab_", (int)s.diagrams.size(),
        s.in_flight, s.running_diagram_index,
        [&s](int i) { return s.diagrams[i].label; },
        [&model, &s](int i) { draw_diagram_controls(model, s, i); },
        [&s]() { s.add_diagram(); },
        s.request_select_diagram);
    s.request_select_diagram = -1;   // запрос потреблён этим кадром
    if (tabs.active    >= 0) s.active_diagram_index = tabs.active;
    if (tabs.to_remove >= 0) model.remove_bifurcation_diagram(tabs.to_remove);
}

// Маппинг window-kind → нейтральный ParamPlotKind. Сама конфигурация живёт в
// configure_param_plot_view (см. блок общих хелперов вверху файла), чтобы
// Parametric и Custom настраивали одинаковые 1D-графики одним и тем же кодом.
static ParamPlotKind to_param_plot_kind(ParametricPlotWindow::Kind kind) {
    switch (kind) {
    case ParametricPlotWindow::Kind::LLE: return ParamPlotKind::LLE;
    case ParametricPlotWindow::Kind::LS:  return ParamPlotKind::LS;
    case ParametricPlotWindow::Kind::Metrics: return ParamPlotKind::Metrics;
    case ParametricPlotWindow::Kind::Bifurcation:
    default:                              return ParamPlotKind::Bifurcation;
    }
}

// One-time Plot2DView setup per kind — called by draw_parametric_plot_windows
// when it lazily constructs a fresh Plot2DView for a newly opened window
// (Plot2DView is now owned per-window there, not a function-static shared by
// every window of a kind, so this replaces the old `if (!view) {...}` inits).
static void configure_plot_view(Plot2DView& view, ParametricPlotWindow::Kind kind) {
    configure_param_plot_view(view, to_param_plot_kind(kind));
}

// Plots one Parametric plot window of kind Bifurcation. `win` supplies
// mode_2d and the diagram indices to show (`members`); `renderer`/`view`/
// `heatmap_map` are this window's own render-state instances, owned and
// cached by draw_parametric_plot_windows (keyed by win.id).
static void draw_bifurcation_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                                   ParametricPlotWindow& win,
                                   PlotRenderer& renderer, Plot2DView& view,
                                   std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) {
    BifurcationAnalysisSession& s = model.bifurcation_session;
    // Per-diagram HeatmapView, keyed by diagram index within THIS window's own map (map is
    // already per-window, so no cross-window bleed). Per-diagram persisted choice takes
    // priority; falls back to the shared "last touched anywhere" default (get_or_create_heatmap).
    auto get_bd_heatmap = [&](int idx) -> HeatmapView& {
        const int cfg_cm = (idx >= 0 && idx < (int)s.diagrams.size())
                           ? s.diagrams[idx].colormap_idx : -1;
        return get_or_create_heatmap(heatmap_map, idx, cfg_cm, model.heatmap_colormap);
    };

    if (win.members.empty()) {
        ImGui::TextDisabled("No diagrams assigned to this window.");
        return;
    }

    // Тулбар-тумблер над плотом: переключает ЭТО окно между classic scatter и colored-1D
    // heatmap напрямую, минуя Plot windows → Type → Members, держа diagram-флаг синхронно с
    // window-флагом. Если членов больше одного (overlay), оставляем первого как фокусного —
    // heatmap-режимы single-member, как и при переключении через Type combo.
    // Только для Bifurcation 1D: поверх настоящего 2D Colored 1D смысла не имеет.
    if (!win.mode_2d) {
        bool c1d = win.colored_1d;
        if (ImGui::Checkbox("Colored 1D diagram##bdtoolbar", &c1d)) {
            win.colored_1d = c1d;
            int focus = win.members[0];
            if (win.members.size() > 1) win.members.resize(1);
            if (focus >= 0 && focus < (int)s.diagrams.size()) {
                s.diagrams[focus].colored_1d = c1d;
                // Переключаем панель контролов на ЭТУ же БД: настройки
                // Colored 1D показываются для диаграммы активной вкладки, и без
                // этого галка включалась бы у одной БД, а блок настроек не
                // появлялся бы (он относился к другой).
                s.request_select_diagram = focus;
                if (!model.loaded_name.empty())
                    lib.save_session(model.loaded_name, "_last_parametric",
                                     session_to_json_parametric(model.bifurcation_session));
            }
            model.parametric_plot_windows_dirty = true;
        }
        // Custom point style — справа от Colored 1D; в colored-режиме плот становится
        // хитмапой и стиль точек неприменим → disabled. Настройка живёт в самой БД, а окно
        // может показывать несколько БД наложением: читаем у фокусного члена (members[0]),
        // а пишем во ВСЕ члены окна — маркер и размер это свойства вида, они одни на окно.
        ImGui::SameLine();
        ImGui::BeginDisabled(win.colored_1d);
        {
            const int focus = win.members[0];
            if (focus >= 0 && focus < (int)s.diagrams.size()) {
                BifurcationDiagramConfig& fbd = s.diagrams[focus];
                if (draw_point_style_toolbar(fbd, "bdtoolbar")) {
                    for (int m : win.members) {
                        if (m < 0 || m >= (int)s.diagrams.size() || m == focus) continue;
                        s.diagrams[m].custom_point_style = fbd.custom_point_style;
                        s.diagrams[m].point_marker       = fbd.point_marker;
                        s.diagrams[m].point_size         = fbd.point_size;
                        s.diagrams[m].point_alpha        = fbd.point_alpha;
                    }
                    if (!model.loaded_name.empty())
                        lib.save_session(model.loaded_name, "_last_parametric",
                                         session_to_json_parametric(model.bifurcation_session));
                }
            }
        }
        ImGui::EndDisabled();
        ImGui::Separator();
    }

    if (win.mode_2d) {
        for (size_t mi = 0; mi < win.members.size(); ++mi) {
            int idx = win.members[mi];
            if (idx < 0 || idx >= (int)s.diagrams.size()) continue;
            BifurcationDiagramConfig& bdact = s.diagrams[idx];
            if (mi > 0) ImGui::Separator();
            ImGui::PushID(idx);

            const unsigned bd_oid = 0xBD2D0000u + (unsigned)idx;
            HeatmapView& hb = get_bd_heatmap(idx);

            // Тулбар цветовой шкалы — общая реализация (см. draw_heatmap_toolbar).
            {
                HeatmapToolbarOpts topts;
                topts.persist_colormap = [&](int cm) {
                    bdact.colormap_idx = cm;   // persist per-diagram only
                    if (!model.loaded_name.empty())
                        lib.save_session(model.loaded_name, "_last_parametric",
                                         session_to_json_parametric(model.bifurcation_session));
                };
                draw_heatmap_toolbar(hb, topts);
            }

            if (!bdact.last_run_2d_ok || bdact.result_2d.values.empty()) {
                ImGui::TextDisabled("No 2D data yet. Press Run.");
                ImGui::PopID();
                continue;
            }

            // Подписи осей — общий auto_axis_name (раньше здесь жила локальная
            // копия, отличавшаяся только fallback'ом "x" вместо "var").
            hb.x_axis.name = auto_axis_name(s.params, s.vars, bdact.param_index,
                                            bdact.sweep_over_var, bdact.var_sweep_index,
                                            bdact.sweep_over_h);
            hb.y_axis.name = auto_axis_name(s.params, s.vars, bdact.param_index_2,
                                            bdact.sweep_over_var_2, bdact.var_sweep_index_2,
                                            bdact.sweep_over_h_2);
            hb.x_axis.log_scale = bdact.log_scale;
            hb.y_axis.log_scale = bdact.log_scale_2;

            bool fit = bdact.fit_request_2d;
            if (fit) bdact.fit_request_2d = false;

            const bool bd_busy = s.in_flight && s.is_2d_run && idx == s.running_diagram_index;
            // Ctrl+T: обе оси карты — это и есть диапазоны свипа.
            const std::vector<ViewRangeTarget> vrt = { vr_target(s.diagrams, idx,
                &BifurcationDiagramConfig::param_lo_text,   &BifurcationDiagramConfig::param_hi_text,
                &BifurcationDiagramConfig::param_lo_2_text, &BifurcationDiagramConfig::param_hi_2_text) };
            hb.popup_extras = [&bdact, &cb, bd_busy, &vrt, &model, &hb]() {
                draw_export_menu_item(bd_busy, cb, [&bdact](const std::string& p) {
                    data_export::export_bif2d(bdact.result_2d, p);
                });
                draw_view_range_menu(vrt, hb.x_axis, hb.y_axis, hb.swap_axes, &model);
            };

            // Значение ячейки — сырой выход DBSCAN: число кластеров пиков, -1 / 0 — коды режима.
            hb.cell_regime = regime_cell_label(bdact.result_2d.flags, bdact.result_2d.n_pts);
            hb.cell_not_computed = cancelled_cell_mask(bdact.result_2d.flags, bdact.result_2d.n_pts);
            hb.value_name = "period";
            hb.value_is_regime_code = true;
            ImVec2 avail = ImGui::GetContentRegionAvail();
            ImVec2 origin = ImGui::GetCursorScreenPos();
            hb.render(renderer, origin, avail,
                      /*owner_id*/ bd_oid, bdact.data_generation_2d,
                      bdact.result_2d.n_pts, bdact.result_2d.n_pts,
                      bdact.result_2d.values.data(),
                      bdact.result_2d.param_lo,   bdact.result_2d.param_hi,
                      bdact.result_2d.param_lo_2, bdact.result_2d.param_hi_2,
                      bdact.result_2d.min_val, bdact.result_2d.max_val,
                      fit);
            handle_view_range_keys(vrt, hb.x_axis, hb.y_axis, plot_window_active(),
                               hb.swap_axes, &model);
            ImGui::PopID();
        }
        return;
    }

    if (win.colored_1d) {
        for (size_t mi = 0; mi < win.members.size(); ++mi) {
            int idx = win.members[mi];
            if (idx < 0 || idx >= (int)s.diagrams.size()) continue;
            BifurcationDiagramConfig& bdact = s.diagrams[idx];
            if (mi > 0) ImGui::Separator();
            ImGui::PushID(idx);

            const unsigned c1d_oid = 0xBD1D0000u + (unsigned)idx;
            HeatmapView& hc = get_bd_heatmap(idx);

            {
                HeatmapToolbarOpts topts;
                topts.persist_colormap = [&](int cm) {
                    bdact.colormap_idx = cm;   // per-diagram (shared with mode_2d)
                    if (!model.loaded_name.empty())
                        lib.save_session(model.loaded_name, "_last_parametric",
                                         session_to_json_parametric(model.bifurcation_session));
                };
                draw_heatmap_toolbar(hc, topts);
            }

            if (!bdact.last_run_ok || bdact.result.bifurcation_points.empty()) {
                ImGui::TextDisabled("No 1D data yet. Press Run.");
                ImGui::PopID();
                continue;
            }

            const auto& source = bdact.plot_inter_peaks ? bdact.result.peak_times
                                                         : bdact.result.bifurcation_points;
            int npts = bdact.result.n_pts;
            int B = parse_int_or(bdact.colored_1d_b_text, 2);
            if (B < 2) B = 2;

            // Y-диапазон: авто (глобальный min/max ±2.5%, см. MATLAB Sup/Slow)
            // либо ручной (точки за пределами отбрасываются при биннинге).
            double ylo, yhi;
            if (bdact.colored_1d_custom_y) {
                ylo = parse_ratio_or(bdact.colored_1d_ymin_text, 0.0);
                yhi = parse_ratio_or(bdact.colored_1d_ymax_text, 1.0);
                if (yhi < ylo) std::swap(ylo, yhi);
                if (yhi - ylo < 1e-12) yhi = ylo + 1.0;
            } else {
                double vmin =  std::numeric_limits<double>::infinity();
                double vmax = -std::numeric_limits<double>::infinity();
                for (int k = 0; k < npts && k < (int)source.size(); ++k) {
                    if (k < (int)bdact.result.flags.size() &&
                        !regime_is_oscillation(bdact.result.flags[k])) continue;
                    for (double y : source[k]) {
                        if (y < vmin) vmin = y;
                        if (y > vmax) vmax = y;
                    }
                }
                if (!std::isfinite(vmin) || !std::isfinite(vmax)) { vmin = 0.0; vmax = 1.0; }
                double dd = vmax - vmin;
                if (dd < 1e-12) dd = 1.0;
                ylo = vmin - dd * 0.025;
                yhi = vmax + dd * 0.025;
            }

            bool stale = bdact.colored_1d_built_from       != bdact.data_generation
                      || bdact.colored_1d_cache_b           != B
                      || bdact.colored_1d_cache_log         != bdact.colored_1d_log
                      || bdact.colored_1d_cache_custom_y    != bdact.colored_1d_custom_y
                      || bdact.colored_1d_cache_inter_peaks != bdact.plot_inter_peaks
                      || bdact.colored_1d_cache_ymin_used   != ylo
                      || bdact.colored_1d_cache_ymax_used   != yhi;
            if (stale) {
                size_t plane_size = (size_t)B * (size_t)npts;
                bdact.colored_1d_cache.assign(plane_size, kSentinelNoData);
                std::vector<int> counts((size_t)B);
                double vmin =  std::numeric_limits<double>::infinity();
                double vmax = -std::numeric_limits<double>::infinity();
                double yrange = yhi - ylo;
                for (int k = 0; k < npts; ++k) {
                    bool diverged = k >= (int)bdact.result.flags.size() ||
                                    !regime_is_oscillation(bdact.result.flags[k]);
                    if (diverged) continue;   // остаётся 999 (тот же серый sentinel, что и mode_2d)
                    std::fill(counts.begin(), counts.end(), 0);
                    int colmax = 0;
                    if (k < (int)source.size()) {
                        for (double y : source[k]) {
                            if (bdact.colored_1d_custom_y && (y < ylo || y > yhi)) continue;
                            int bin = (int)std::lround((y - ylo) / yrange * (B - 1));
                            if (bin < 0) bin = 0; else if (bin >= B) bin = B - 1;
                            int cnt = ++counts[(size_t)bin];
                            if (cnt > colmax) colmax = cnt;
                        }
                    }
                    if (colmax <= 0) {
                        // Не diverged, просто ни одна точка не попала в диапазон —
                        // 0-плотность по колонке (не sentinel: это не "нет данных
                        // из-за расхождения траектории", а просто пустой срез).
                        double v0 = bdact.colored_1d_log ? std::log10(1e-6) : 0.0;
                        for (int b = 0; b < B; ++b)
                            bdact.colored_1d_cache[(size_t)b * (size_t)npts + (size_t)k] = v0;
                        continue;
                    }
                    double inv = 1.0 / (double)colmax;
                    for (int b = 0; b < B; ++b) {
                        double density = counts[(size_t)b] * inv;
                        double v = bdact.colored_1d_log ? std::log10(std::max(density, 1e-6)) : density;
                        bdact.colored_1d_cache[(size_t)b * (size_t)npts + (size_t)k] = v;
                        if (v < vmin) vmin = v;
                        if (v > vmax) vmax = v;
                    }
                }
                bdact.colored_1d_cache_vmin = std::isfinite(vmin) ? vmin : 0.0;
                bdact.colored_1d_cache_vmax = std::isfinite(vmax) ? vmax : 1.0;
                bdact.colored_1d_built_from      = bdact.data_generation;
                bdact.colored_1d_cache_b         = B;
                bdact.colored_1d_cache_log       = bdact.colored_1d_log;
                bdact.colored_1d_cache_custom_y  = bdact.colored_1d_custom_y;
                bdact.colored_1d_cache_inter_peaks = bdact.plot_inter_peaks;
                bdact.colored_1d_cache_ymin_used = ylo;
                bdact.colored_1d_cache_ymax_used = yhi;
                ++bdact.colored_1d_cache_gen;
            }

            hc.x_axis.name = auto_axis_name(s.params, s.vars, bdact.param_index,
                                             bdact.sweep_over_var, bdact.var_sweep_index, bdact.sweep_over_h);
            hc.x_axis.log_scale = bdact.log_scale;
            hc.y_axis.name = (bdact.writable_var >= 0 && bdact.writable_var < (int)s.vars.size())
                            ? s.vars[bdact.writable_var] : "X";
            if (bdact.plot_inter_peaks) hc.y_axis.name += " interval";

            bool fit = bdact.fit_request;
            if (fit) bdact.fit_request = false;

            const bool c1d_busy = s.in_flight && !s.is_2d_run && idx == s.running_diagram_index;
            // Ctrl+T: X — диапазон свипа, Y — окно гистограммы плотности.
            // Запись в ymin/ymax сама включает custom_y (см. ViewRangeTarget),
            // иначе значения легли бы мимо: при авто-Y их никто не читает.
            const std::vector<ViewRangeTarget> vrt = { vr_target(s.diagrams, idx,
                &BifurcationDiagramConfig::param_lo_text,        &BifurcationDiagramConfig::param_hi_text,
                &BifurcationDiagramConfig::colored_1d_ymin_text, &BifurcationDiagramConfig::colored_1d_ymax_text,
                &BifurcationDiagramConfig::colored_1d_custom_y) };
            hc.popup_extras = [&bdact, &cb, c1d_busy, &vrt, &model, &hc]() {
                draw_export_menu_item(c1d_busy, cb, [&bdact](const std::string& p) {
                    data_export::export_bif1d(bdact.result, p);
                });
                draw_view_range_menu(vrt, hc.x_axis, hc.y_axis, hc.swap_axes, &model);
            };

            // Тот же lo/hi/reverse-разбор, что и у классического scatter'а
            // ниже (continuation хранит снапшот в result, иначе — из текстов).
            double lo = (bdact.continuation && bdact.result.param_hi != bdact.result.param_lo)
                        ? bdact.result.param_lo : parse_ratio_or(bdact.param_lo_text, 0.0);
            double hi = (bdact.continuation && bdact.result.param_hi != bdact.result.param_lo)
                        ? bdact.result.param_hi : parse_ratio_or(bdact.param_hi_text, 1.0);
            bool rev = bdact.result.continuation_reverse;
            double x0 = rev ? hi : lo;
            double x1 = rev ? lo : hi;

            hc.cell_regime = regime_cell_label(bdact.result.flags, 0);   // флаг на колонку
            hc.cell_not_computed = cancelled_cell_mask(bdact.result.flags, 0);
            ImVec2 avail = ImGui::GetContentRegionAvail();
            ImVec2 origin = ImGui::GetCursorScreenPos();
            hc.render(renderer, origin, avail,
                      /*owner_id*/ c1d_oid, bdact.colored_1d_cache_gen,
                      npts, B,
                      bdact.colored_1d_cache.data(),
                      x0, x1,
                      ylo, yhi,
                      bdact.colored_1d_cache_vmin, bdact.colored_1d_cache_vmax,
                      fit);
            handle_view_range_keys(vrt, hc.x_axis, hc.y_axis, plot_window_active(),
                               hc.swap_axes, &model);
            ImGui::PopID();
        }
        return;
    }

    // Имеется ли хотя бы одна БД этого окна с готовыми данными?
    bool any_data = false;
    for (int idx : win.members) {
        if (idx < 0 || idx >= (int)s.diagrams.size()) continue;
        const auto& bd = s.diagrams[idx];
        if (bd.last_run_ok && !bd.result.bifurcation_points.empty()) { any_data = true; break; }
    }
    if (!any_data) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }

    // Подписи осей + X-fit диапазон (см. configure_sweep_x_axis). Заодно
    // собираем writable_var по членам — он нужен для Y-подписи ниже.
    int shared_var_idx = -2;   // -2 init, -1 combination, [0,N) — single var
    bool var_idx_mixed = false;  // отдельный флаг — нельзя реюзать -1 (теперь это combination)
    for (int idx : win.members) {
        if (idx < 0 || idx >= (int)s.diagrams.size()) continue;
        const auto& bd = s.diagrams[idx];
        if (!bd.last_run_ok) continue;
        if (shared_var_idx == -2) shared_var_idx = bd.writable_var;
        else if (shared_var_idx != bd.writable_var) var_idx_mixed = true;
    }
    configure_sweep_x_axis_from(view, win.members, s.diagrams, s.params, s.vars);
    // Y-label: combination (-1) -> "x0+pi*x1+e*x2"; single var -> имя переменной;
    // mixed / нет данных -> generic "X".
    if (var_idx_mixed || shared_var_idx == -2) {
        view.y_axis.name = "X";
    } else if (shared_var_idx == -1) {
        // Комбинация — общая подпись combo_var_label. Раньше формула была
        // выписана здесь ТРЕТЬЕЙ копией и печаталась без пробелов
        // ("x+pi*y+e*z"); теперь подпись оси совпадает с пунктом комбо
        // ("x + pi*y + e*z"), из которого пользователь эту комбинацию и выбрал.
        view.y_axis.name = s.vars.empty() ? std::string("X") : combo_var_label(s.vars);
    } else if (shared_var_idx >= 0 && shared_var_idx < (int)s.vars.size()) {
        view.y_axis.name = s.vars[shared_var_idx];
    } else {
        view.y_axis.name = "X";
    }

    // Буферы точек (по одному на серию), свои у каждого окна — см.
    // window_point_bufs: между кадрами они держат capacity.
    auto& bufs = window_point_bufs(win.id, win.members.size());
    // Узлы сетки перезаполняются каждый кадр: данные могли пересчитаться.
    view.snap_x_nodes      = nullptr;
    view.snap_x_node_count = 0;

    std::vector<PlotSeriesInput> series_in;
    std::vector<bool> init_vis;
    std::vector<bool> glob_vis;
    series_in.reserve(win.members.size());
    init_vis.reserve(win.members.size());
    glob_vis.reserve(win.members.size());

    bool any_fit = false;
    int  data_gen = 0;

    // Сетка снапа фокусной (первой посчитанной) диаграммы окна — заполняется
    // в цикле ниже теми же lo/hi/npts, по которым строятся точки.
    double snap_lo_focus = 0.0, snap_hi_focus = 0.0;
    int    snap_n_focus  = 0;

    for (size_t mi = 0; mi < win.members.size(); ++mi) {
        int idx = win.members[mi];
        if (idx < 0 || idx >= (int)s.diagrams.size()) continue;
        BifurcationDiagramConfig& bd = s.diagrams[idx];
        auto& buf = bufs[mi];
        buf.clear();
        int total_pts = 0;

        if (bd.last_run_ok && !bd.result.bifurcation_points.empty()) {
            const auto& source = bd.plot_inter_peaks ? bd.result.peak_times
                                                     : bd.result.bifurcation_points;
            // Диапазон прогона (run_bifurcation_1d), не текстовые поля: их правка до Run
            // не должна переносить посчитанные точки на другую сетку.
            const double lo = bd.result.param_lo, hi = bd.result.param_hi;
            bool rev = bd.result.continuation_reverse;
            int npts = bd.result.n_pts;
            // Сетка снапа — ровно та, по которой строятся точки ниже. Раньше
            // она бралась из bd.result.param_lo/hi, а классический (не
            // continuation) прогон их не заполняет: снап уходил на дефолты
            // структуры, [0, 1], и курсор садился на узлы чужой сетки.
            if (snap_n_focus == 0) {
                snap_lo_focus = lo; snap_hi_focus = hi; snap_n_focus = npts;
                set_snap_nodes(view, win.id, npts, lo, hi, bd.log_scale, rev, bd.continuation);
            }
            for (int k = 0; k < npts; ++k) {
                if (k < (int)bd.result.flags.size() &&
                    !regime_is_oscillation(bd.result.flags[k])) continue;
                double x = sweep_value_at(k, npts, lo, hi, bd.log_scale, rev, bd.continuation);
                if (k >= (int)source.size()) continue;
                for (double y : source[k]) {
                    buf.push_back((double)x);
                    buf.push_back((double)y);
                    ++total_pts;
                }
            }
        }

        PlotSeriesInput si;
        si.points   = buf.empty() ? nullptr : buf.data();
        si.n_points = total_pts;
        si.color    = ic_base_color((int)mi);
        // α в custom-режиме — через альфу цвета серии (GL-путь её блендит),
        // как в Custom line style у Phase2D/TimeDomain.
        if (bd.custom_point_style) si.color.w = bd.point_alpha;
        si.label    = bd.label;
        series_in.push_back(si);
        init_vis.push_back(true);
        glob_vis.push_back(true);   // membership in this window IS the visibility gate

        // data_gen накапливает per-BD generation + toggle inter_peaks
        // (как было в одно-БД версии — чтобы тоггл триггерил перерисовку VBO).
        data_gen = data_gen * 31 + bd.data_generation * 2 + (bd.plot_inter_peaks ? 1 : 0);
        if (bd.fit_request) { any_fit = true; bd.fit_request = false; }
    }

    // Right-click "Export data..." for the bifurcation 1D line plot. The
    // submenu lists every diagram in the session that has a finished run
    // (not just this window's members — export isn't scoped to a window).
    // Ctrl+T: по X идёт свип, по Y — значение переменной, входа в расчёте у
    // него нет, поэтому Y-пара остаётся нулевой. Целей столько же, сколько
    // диаграмм в окне; первая (она же фокусная — по ней берутся snap-сетка и
    // стиль точек) достаётся Ctrl+T, все — Ctrl+Shift+T.
    std::vector<ViewRangeTarget> vrt;
    for (int idx : win.members)
        if (idx >= 0 && idx < (int)s.diagrams.size())
            vrt.push_back(vr_target(s.diagrams, idx,
                &BifurcationDiagramConfig::param_lo_text, &BifurcationDiagramConfig::param_hi_text));

    view.popup_extras = [&s, &cb, &vrt, &model, &view]() {
        draw_export_submenu("bd", (int)s.diagrams.size(),
            [&s](int i) { return s.diagrams[i].label; },
            [&s](int i) { return s.diagrams[i].last_run_ok; },
            [&s](int i) { return s.in_flight && i == s.running_diagram_index; },
            [&s](int i, const std::string& p) { data_export::export_bif1d(s.diagrams[i].result, p); },
            cb);
        draw_view_range_menu(vrt, view.x_axis, view.y_axis, /*swapped*/ false, &model);
    };

    // Snap X к узлам фокусной БД — по той же сетке, что и её точки (см. выше).
    apply_snap_x(view, snap_lo_focus, snap_hi_focus, snap_n_focus);

    // Маркер/размер точек — свойства вида (одни на окно): берём у фокусного
    // члена, тулбар выше держит остальных членов синхронно.
    {
        const int aidx = win.members.empty() ? -1 : win.members[0];
        if (aidx >= 0 && aidx < (int)s.diagrams.size())
            apply_point_style(view, s.diagrams[aidx]);
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();

    view.render(renderer, origin, avail, /*owner_id*/ 0xBE0F1D, data_gen,
                series_in, init_vis, glob_vis, any_fit);
    handle_view_range_keys(vrt, view.x_axis, view.y_axis, plot_window_active(), /*swapped*/ false, &model);
}

// LLE: контролы (per-curve в табе) + line-plot λ(param)

// Контролы одной LLE-кривой. Возвращает true, если пользователь нажал Run
// для этой кривой.
static void draw_lle_curve_controls(AppModel& model, LLEAnalysisSession& s, int idx) {
    LLECurveConfig& c = s.curves[idx];
    // Чей конфиг правит эта панель — для ПКМ-рассылки "во все такие же".
    model.broadcast_source_tab = BroadcastTab::LLE;
    model.broadcast_source_idx = idx;

    ImGui::SetNextItemWidth(kComboW);
    if (InputTextStr("Label", c.label))
        c.label_is_manual = !c.label.empty();   // empty → back to auto
    ImGui::Separator();

    draw_scheme_combo("Scheme", c.scheme, s.custom_schemes, {}, &s.enabled_builtin_schemes,
                      &model, model.is_map, &s.wrapper_schemes);
    ImGui::Separator();

    // Sweep target: параметры + разделитель + переменные (IC) + dt (h). См. BD.
    draw_sweep_target_combo("Sweep", s.params, s.vars,
                            c.param_index, c.sweep_over_var, c.var_sweep_index,
                            c.sweep_over_h,
                            c.mode_2d ? &c.sweep_over_h_2 : nullptr,
                            /*note_when_empty*/ true, kComboW,
                            scheme_uses_symmetry(c.scheme, s.custom_schemes),
                            &model, BroadcastField::SweepTarget, !model.is_map,
                            &c.ad_axis, &c.adaptive);
    InputNumStr(c.sweep_over_h ? "h lo" : "Param lo", c.param_lo_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepLo, {},
                                        c.sweep_over_h ? "h lo" : "Param lo", c.param_lo_text); });
    InputNumStr(c.sweep_over_h ? "h hi" : "Param hi", c.param_hi_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepHi, {},
                                        c.sweep_over_h ? "h hi" : "Param hi", c.param_hi_text); });
    ImGui::Checkbox("Log scale##lle_log", &c.log_scale);
    if (c.log_scale) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
    InputNumStr("Resolution", c.n_pts_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::Resolution, {},
                                        "Resolution", c.n_pts_text); });

    draw_continuation_device_block(c, "lle", c.mode_2d || c.sweep_over_var, c.mode_2d);

    ImGui::Separator();
    // 2D-режим. Сетка квадратная (см. analysis_session.h:LLECurveConfig коммент)
    // — Resolution выше применяется и к X, и к Y.
    ImGui::Checkbox("2D mode (heatmap)", &c.mode_2d);
    if (c.mode_2d) {
        ImGui::Indent();
        // Sweep target для второй оси — то же комбо, симметрично первой.
        // Пункт "dt (h)" прячется, если шаг уже занял X: ровно одна ось = h.
        if (!s.params.empty() || !s.vars.empty())
            draw_sweep_target_combo("Sweep Y", s.params, s.vars,
                                    c.param_index_2, c.sweep_over_var_2, c.var_sweep_index_2,
                                    c.sweep_over_h_2, &c.sweep_over_h,
                                    /*note_when_empty*/ false, kComboW,
                                    scheme_uses_symmetry(c.scheme, s.custom_schemes),
                                    &model, BroadcastField::SweepTarget2, !model.is_map,
                                    &c.ad_axis_2, &c.adaptive);
        InputNumStr(c.sweep_over_h_2 ? "h2 lo" : "Param2 lo", c.param_lo_2_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepLo2, {},
                                        c.sweep_over_h_2 ? "h2 lo" : "Param2 lo", c.param_lo_2_text); });
        InputNumStr(c.sweep_over_h_2 ? "h2 hi" : "Param2 hi", c.param_hi_2_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepHi2, {},
                                        c.sweep_over_h_2 ? "h2 hi" : "Param2 hi", c.param_hi_2_text); });
        ImGui::Checkbox("Log scale##lle_log2", &c.log_scale_2);
        if (c.log_scale_2) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
        ImGui::TextDisabled("Grid is square (Resolution applies to both axes).");
        ImGui::Unindent();
    }

    ImGui::Separator();

    // Integration (collapsible)
    {
        IntegrationFields f;
        f.h          = &c.h_text;
        f.symmetry_s = &c.symmetry_s;
        f.cd_fp_iters = &c.cd_fp_iters;
        f.newton      = &c.newton;
        f.t_max      = &c.t_max_text;
        f.transient  = &c.transient_text;
        f.max_value  = &c.max_value_text;   // decimator'а у LLE нет
        f.adaptive      = &c.adaptive;
        f.adaptive_opts = kAdUiRenorm;
        draw_integration_block("Integration##lle_int", c.scheme, s.custom_schemes, f, &model, model.is_map);
    }

    // LLE (Wolf/Benettin) (collapsible)
    if (ImGui::CollapsingHeader("LLE (Wolf/Benettin)##lle_wb", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("eps", c.eps_text, kFieldW);
        InputNumStr("NT",  c.nt_text, kFieldW);
        InputNumStr("vector transient", c.vtr_text, kFieldW);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Tangent-vector transient: after the transient of the trajectory the\n"
                "perturbations are attached and renormalised every NT for this long\n"
                "(rounded to whole NT blocks) without adding their stretching to the sum.\n"
                "While the random initial frame turns towards the Lyapunov directions it\n"
                "collects ln(1/c) (c - its projection on them), and the exponents then\n"
                "depend on that random frame as ~ln(1/c)/t_max: point-to-point noise.\n"
                "It aligns as exp(-(L1 - L2) t), so take ~10/(L1 - L2), or 10-20%% of t_max.\n"
                "Costs this much extra time for every perturbed copy. 0 - off.");
        ImGui::TextDisabled(model.is_map
            ? "eps = initial perturbation magnitude; NT = block length\n"
              "between renormalizations (in iterations)."
            : "eps = initial perturbation magnitude; NT = block length\n"
                            "between renormalizations (in time units).");
    }

    draw_named_num_fields("Initial conditions##lle_ic", s.vars,   c.initial_conditions,
                          nullptr, &model, BroadcastField::InitCondition);
    draw_named_num_fields("Parameters##lle_par",        s.params, c.param_values,
                          nullptr, &model, BroadcastField::Param);
    draw_csv_output_block("CSV output##lle_csv", "Save to file", c.csv_save_enabled,
                          "##lle_csv_path", c.csv_output_path,
                          "Path is kept even when save is off. Also writes <path>_config.csv.");

    // Run-кнопка живёт на уровне draw_parametric_controls.

    if (c.mode_2d) {
        if (c.last_run_2d_ok) {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: %dx%d heatmap, lambda(min..max) = %.4g..%.4g",
                c.result_2d.n_pts, c.result_2d.n_pts,
                c.result_2d.min_val, c.result_2d.max_val);
            draw_regime_summary(c.result_2d.flags, &c.regime_counts_2d, c.data_generation_2d);
            draw_adaptive_stats(c.result_2d.ad_stats);
        }
        else if (!c.last_error.empty()) {
            draw_error_box("##lle_err", c.last_error);
        }
    } else {
        if (c.last_run_ok) {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: n_pts=%d, lambda-curve computed", c.result.n_pts);
            draw_regime_summary(c.result.flags);
            draw_adaptive_stats(c.result.ad_stats);
        }
        else if (!c.last_error.empty()) {
            draw_error_box("##lle_err", c.last_error);
        }
    }
}

// Tab bar по LLE-кривым + кнопка «+» (копирует последнюю).
static void draw_lle_controls(AppModel& model, SystemLibrary& /*lib*/) {
    LLEAnalysisSession& s = model.lle_session;

    // Run + Ctrl+R — в draw_parametric_controls (общая кнопка слева от Run all).
    const TabBarResult tabs = draw_config_tab_bar(
        "##lle_tabs", "lle_tab_", (int)s.curves.size(),
        s.in_flight, s.running_curve_index,
        [&s](int i) { return s.curves[i].label; },
        [&model, &s](int i) { draw_lle_curve_controls(model, s, i); },
        [&s]() { s.add_curve(); });
    if (tabs.active    >= 0) s.active_curve_index = tabs.active;
    if (tabs.to_remove >= 0) model.remove_lle_curve(tabs.to_remove);
}

// Plot LLE: линии (points_mode=false), каждая кривая — λ(param). При mode_2d=true у
// активной кривой вместо линий рисуется HeatmapView (сетка λ(p1, p2) с colormap'ом).
// Per-window-state shape — как у draw_bifurcation_plot (обоснование см. там).
static void draw_lle_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                          const ParametricPlotWindow& win,
                          PlotRenderer& renderer, Plot2DView& view,
                          std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) {
    LLEAnalysisSession& s = model.lle_session;
    auto get_lle_heatmap = [&](int idx) -> HeatmapView& {
        const int cfg_cm = (idx >= 0 && idx < (int)s.curves.size())
                           ? s.curves[idx].colormap_idx : -1;
        return get_or_create_heatmap(heatmap_map, idx, cfg_cm, model.heatmap_colormap);
    };

    if (win.members.empty()) {
        ImGui::TextDisabled("No curves assigned to this window.");
        return;
    }

    if (win.mode_2d) {
        for (size_t mi = 0; mi < win.members.size(); ++mi) {
            int idx = win.members[mi];
            if (idx < 0 || idx >= (int)s.curves.size()) continue;
            LLECurveConfig& cact = s.curves[idx];
            if (mi > 0) ImGui::Separator();
            ImGui::PushID(idx);

            const unsigned lle_oid = 0xBE110000u + (unsigned)idx;
            HeatmapView& heatmap = get_lle_heatmap(idx);

            {
                HeatmapToolbarOpts topts;
                topts.persist_colormap = [&](int cm) {
                    cact.colormap_idx = cm;   // persist per-curve only
                    if (!model.loaded_name.empty())
                        lib.save_session(model.loaded_name, "_last_lle",
                                         session_to_json_lle(model.lle_session));
                };
                draw_heatmap_toolbar(heatmap, topts);
            }

            if (!cact.last_run_2d_ok || cact.result_2d.values.empty()) {
                ImGui::TextDisabled("No 2D data yet. Press Run.");
                ImGui::PopID();
                continue;
            }

            // Подписи осей по реальным selected-полям свипа — общий auto_axis_name.
            heatmap.x_axis.name = auto_axis_name(s.params, s.vars, cact.param_index,
                                                 cact.sweep_over_var, cact.var_sweep_index,
                                                 cact.sweep_over_h);
            heatmap.y_axis.name = auto_axis_name(s.params, s.vars, cact.param_index_2,
                                                 cact.sweep_over_var_2, cact.var_sweep_index_2,
                                                 cact.sweep_over_h_2);
            heatmap.x_axis.log_scale = cact.log_scale;
            heatmap.y_axis.log_scale = cact.log_scale_2;

            bool fit = cact.fit_request_2d;
            if (fit) cact.fit_request_2d = false;

            const bool busy = s.in_flight && s.is_2d_run && idx == s.running_curve_index;
            // Ctrl+T: обе оси карты — диапазоны свипа (см. Bif 2D).
            const std::vector<ViewRangeTarget> vrt = { vr_target(s.curves, idx,
                &LLECurveConfig::param_lo_text,   &LLECurveConfig::param_hi_text,
                &LLECurveConfig::param_lo_2_text, &LLECurveConfig::param_hi_2_text) };
            heatmap.popup_extras = [&cact, &cb, busy, &vrt, &model, &heatmap]() {
                draw_export_menu_item(busy, cb, [&cact](const std::string& p) {
                    data_export::export_lle2d(cact.result_2d, p);
                });
                draw_view_range_menu(vrt, heatmap.x_axis, heatmap.y_axis, heatmap.swap_axes, &model);
            };

            heatmap.cell_regime = regime_cell_label(cact.result_2d.flags, cact.result_2d.n_pts);
            heatmap.cell_not_computed = cancelled_cell_mask(cact.result_2d.flags, cact.result_2d.n_pts);
            ImVec2 avail = ImGui::GetContentRegionAvail();
            ImVec2 origin = ImGui::GetCursorScreenPos();
            heatmap.render(renderer, origin, avail,
                           /*owner_id*/ lle_oid, cact.data_generation_2d,
                           cact.result_2d.n_pts, cact.result_2d.n_pts,
                           cact.result_2d.values.data(),
                           cact.result_2d.param_lo,   cact.result_2d.param_hi,
                           cact.result_2d.param_lo_2, cact.result_2d.param_hi_2,
                           cact.result_2d.min_val, cact.result_2d.max_val,
                           fit);
            handle_view_range_keys(vrt, heatmap.x_axis, heatmap.y_axis, plot_window_active(),
                               heatmap.swap_axes, &model);
            ImGui::PopID();
        }
        return;
    }

    bool any_data = false;
    for (int idx : win.members) {
        if (idx < 0 || idx >= (int)s.curves.size()) continue;
        const auto& c = s.curves[idx];
        if (c.last_run_ok && !c.result.lyapunov.empty()) { any_data = true; break; }
    }
    if (!any_data) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }

    // Подпись X — см. configure_sweep_x_axis_from.
    configure_sweep_x_axis_from(view, win.members, s.curves, s.params, s.vars);
    view.y_axis.name = "lambda";

    auto& bufs = window_point_bufs(win.id, win.members.size());
    // Узлы сетки перезаполняются каждый кадр: данные могли пересчитаться.
    view.snap_x_nodes      = nullptr;
    view.snap_x_node_count = 0;

    std::vector<PlotSeriesInput> series_in;
    std::vector<bool> init_vis, glob_vis;
    series_in.reserve(win.members.size());
    init_vis.reserve(win.members.size());
    glob_vis.reserve(win.members.size());

    bool any_fit = false;
    int  data_gen = 0;

    for (size_t mi = 0; mi < win.members.size(); ++mi) {
        int idx = win.members[mi];
        if (idx < 0 || idx >= (int)s.curves.size()) continue;
        LLECurveConfig& c = s.curves[idx];
        auto& buf = bufs[mi];
        buf.clear();
        int total_pts = 0;

        if (c.last_run_ok && !c.result.lyapunov.empty()) {
            // X считаются по тому диапазону, с которым реально шёл Run, а не
            // по текущим полям GUI — иначе кривая «прыгает» при редактировании
            // param_lo/hi до следующего Run.
            double lo = c.result.param_lo;
            double hi = c.result.param_hi;
            int npts = c.result.n_pts;
            // При backward-continuation точка k считалась для hi-(hi-lo)*k/(n-1)
            // (см. run_lle1d_continuation_cpu) — иначе кривая была бы зеркальной.
            const bool rev = c.result.continuation_reverse;
            if (view.snap_x_node_count == 0)
                set_snap_nodes(view, win.id, npts, lo, hi, c.log_scale, rev, c.continuation);
            for (int k = 0; k < npts; ++k) {
                if (k < (int)c.result.flags.size() &&
                    !regime_is_oscillation(c.result.flags[k])) continue;
                double x = sweep_value_at(k, npts, lo, hi, c.log_scale, rev, c.continuation);
                double y = c.result.lyapunov[k];
                if (!std::isfinite(y)) continue;
                buf.push_back((double)x);
                buf.push_back((double)y);
                ++total_pts;
            }
        }

        PlotSeriesInput si;
        si.points   = buf.empty() ? nullptr : buf.data();
        si.n_points = total_pts;
        si.color    = ic_base_color((int)mi);
        si.label    = c.label;
        series_in.push_back(si);
        init_vis.push_back(true);
        glob_vis.push_back(true);   // membership in this window IS the visibility gate

        data_gen = data_gen * 31 + c.data_generation;
        if (c.fit_request) { any_fit = true; c.fit_request = false; }
    }

    // Ctrl+T: по Y идёт сам показатель, входа в расчёте у него нет (см. Bif 1D).
    std::vector<ViewRangeTarget> vrt;
    for (int idx : win.members)
        if (idx >= 0 && idx < (int)s.curves.size())
            vrt.push_back(vr_target(s.curves, idx,
                &LLECurveConfig::param_lo_text, &LLECurveConfig::param_hi_text));

    view.popup_extras = [&s, &cb, &vrt, &model, &view]() {
        draw_export_submenu("lle", (int)s.curves.size(),
            [&s](int i) { return s.curves[i].label; },
            [&s](int i) { return s.curves[i].last_run_ok; },
            [&s](int i) { return s.in_flight && !s.is_2d_run && i == s.running_curve_index; },
            [&s](int i, const std::string& p) { data_export::export_lle1d(s.curves[i].result, p); },
            cb);
        draw_view_range_menu(vrt, view.x_axis, view.y_axis, /*swapped*/ false, &model);
    };

    // Snap X к узлам первой кривой этого окна (см. apply_snap_x_from_first_member).
    apply_snap_x_from_first_member(view, win.members, s.curves);

    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    view.render(renderer, origin, avail, /*owner_id*/ 0xBE11E5, data_gen,
                series_in, init_vis, glob_vis, any_fit);
    handle_view_range_keys(vrt, view.x_axis, view.y_axis, plot_window_active(), /*swapped*/ false, &model);
}

// LS: контролы (per-curve в табе) + line-plot λ_k(param), N экспонент
// на один спектр-«прогон». UX зеркало LLE.

static void draw_ls_curve_controls(AppModel& model, LyapunovSpectrumAnalysisSession& s, int idx) {
    LSCurveConfig& c = s.curves[idx];
    // Чей конфиг правит эта панель — для ПКМ-рассылки "во все такие же".
    model.broadcast_source_tab = BroadcastTab::LS;
    model.broadcast_source_idx = idx;

    ImGui::SetNextItemWidth(kComboW);
    if (InputTextStr("Label", c.label))
        c.label_is_manual = !c.label.empty();   // empty → back to auto
    ImGui::Separator();

    draw_scheme_combo("Scheme", c.scheme, s.custom_schemes, {}, &s.enabled_builtin_schemes,
                      &model, model.is_map, &s.wrapper_schemes);
    ImGui::Separator();

    // Sweep target: параметры + разделитель + переменные (IC) + dt (h). См. BD.
    draw_sweep_target_combo("Sweep", s.params, s.vars,
                            c.param_index, c.sweep_over_var, c.var_sweep_index,
                            c.sweep_over_h,
                            c.mode_2d ? &c.sweep_over_h_2 : nullptr,
                            /*note_when_empty*/ true, kComboW,
                            scheme_uses_symmetry(c.scheme, s.custom_schemes),
                            &model, BroadcastField::SweepTarget, !model.is_map,
                            &c.ad_axis, &c.adaptive);
    InputNumStr(c.sweep_over_h ? "h lo" : "Param lo", c.param_lo_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepLo, {},
                                        c.sweep_over_h ? "h lo" : "Param lo", c.param_lo_text); });
    InputNumStr(c.sweep_over_h ? "h hi" : "Param hi", c.param_hi_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepHi, {},
                                        c.sweep_over_h ? "h hi" : "Param hi", c.param_hi_text); });
    ImGui::Checkbox("Log scale##ls_log", &c.log_scale);
    if (c.log_scale) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
    InputNumStr("Resolution", c.n_pts_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::Resolution, {},
                                        "Resolution", c.n_pts_text); });

    draw_continuation_device_block(c, "ls", c.mode_2d || c.sweep_over_var, c.mode_2d);

    ImGui::Separator();
    // 2D-режим. Сетка квадратная (см. LLE 2D — то же ограничение getValueByIdx).
    ImGui::Checkbox("2D mode (heatmap of one exponent)", &c.mode_2d);
    if (c.mode_2d) {
        ImGui::Indent();
        // Пункт "dt (h)" прячется, если шаг уже занял X: ровно одна ось = h.
        if (!s.params.empty() || !s.vars.empty())
            draw_sweep_target_combo("Sweep Y", s.params, s.vars,
                                    c.param_index_2, c.sweep_over_var_2, c.var_sweep_index_2,
                                    c.sweep_over_h_2, &c.sweep_over_h,
                                    /*note_when_empty*/ false, kComboW,
                                    scheme_uses_symmetry(c.scheme, s.custom_schemes),
                                    &model, BroadcastField::SweepTarget2, !model.is_map,
                                    &c.ad_axis_2, &c.adaptive);
        InputNumStr(c.sweep_over_h_2 ? "h2 lo" : "Param2 lo", c.param_lo_2_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepLo2, {},
                                        c.sweep_over_h_2 ? "h2 lo" : "Param2 lo", c.param_lo_2_text); });
        InputNumStr(c.sweep_over_h_2 ? "h2 hi" : "Param2 hi", c.param_hi_2_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepHi2, {},
                                        c.sweep_over_h_2 ? "h2 hi" : "Param2 hi", c.param_hi_2_text); });
        ImGui::Checkbox("Log scale##ls_log2", &c.log_scale_2);
        if (c.log_scale_2) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
        ImGui::TextDisabled("Grid is square (Resolution applies to both axes).\nAll N exponents computed; switch in plot window.");
        ImGui::Unindent();
    }

    ImGui::Separator();

    // Integration (collapsible)
    {
        IntegrationFields f;
        f.h          = &c.h_text;
        f.symmetry_s = &c.symmetry_s;
        f.cd_fp_iters = &c.cd_fp_iters;
        f.newton      = &c.newton;
        f.t_max      = &c.t_max_text;
        f.transient  = &c.transient_text;
        f.max_value  = &c.max_value_text;   // decimator'а у LS нет
        f.adaptive      = &c.adaptive;
        f.adaptive_opts = kAdUiRenorm;
        draw_integration_block("Integration##ls_int", c.scheme, s.custom_schemes, f, &model, model.is_map);
    }

    // LS (Wolf/Benettin + Gram-Schmidt) (collapsible)
    if (ImGui::CollapsingHeader("LS (Wolf/Benettin + Gram-Schmidt)##ls_wbgs", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("eps", c.eps_text, kFieldW);
        InputNumStr("NT",  c.nt_text, kFieldW);
        InputNumStr("vector transient", c.vtr_text, kFieldW);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Tangent-vector transient: after the transient of the trajectory the\n"
                "perturbations are attached and renormalised every NT for this long\n"
                "(rounded to whole NT blocks) without adding their stretching to the sum.\n"
                "While the random initial frame turns towards the Lyapunov directions it\n"
                "collects ln(1/c) (c - its projection on them), and the exponents then\n"
                "depend on that random frame as ~ln(1/c)/t_max: point-to-point noise.\n"
                "It aligns as exp(-(L1 - L2) t), so take ~10/(L1 - L2), or 10-20%% of t_max.\n"
                "Costs this much extra time for every perturbed copy. 0 - off.");
        ImGui::TextDisabled(model.is_map
            ? "eps = initial perturbation magnitude; NT = block length\n"
              "between renormalizations (in iterations)."
            : "eps = initial perturbation magnitude; NT = block length\n"
                            "between renormalizations (in time units).");
    }

    draw_named_num_fields("Initial conditions##ls_ic", s.vars,   c.initial_conditions,
                          nullptr, &model, BroadcastField::InitCondition);
    draw_named_num_fields("Parameters##ls_par",        s.params, c.param_values,
                          nullptr, &model, BroadcastField::Param);
    draw_csv_output_block("CSV output##ls_csv", "Save to file", c.csv_save_enabled,
                          "##ls_csv_path", c.csv_output_path,
                          "Path is kept even when save is off. Also writes <path>_config.csv.");

    // Run-кнопка живёт на уровне draw_parametric_controls.

    if (c.mode_2d) {
        if (c.last_run_2d_ok) {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: %dx%d heatmap, %d exponents",
                c.result_2d.n_pts, c.result_2d.n_pts, c.result_2d.n_exponents);
            draw_regime_summary(c.result_2d.flags, &c.regime_counts_2d, c.data_generation_2d);
            draw_adaptive_stats(c.result_2d.ad_stats);
        }
        else if (!c.last_error.empty()) {
            draw_error_box("##ls_err_2d", c.last_error);
        }
    } else {
        if (c.last_run_ok) {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: n_pts=%d, n_exponents=%d", c.result.n_pts, c.result.n_exponents);
            draw_regime_summary(c.result.flags);
            draw_adaptive_stats(c.result.ad_stats);
        }
        else if (!c.last_error.empty()) {
            draw_error_box("##ls_err", c.last_error);
        }
    }
}

static void draw_ls_controls(AppModel& model, SystemLibrary& /*lib*/) {
    LyapunovSpectrumAnalysisSession& s = model.ls_session;

    // Run + Ctrl+R — в draw_parametric_controls (общая кнопка слева от Run all).
    const TabBarResult tabs = draw_config_tab_bar(
        "##ls_tabs", "ls_tab_", (int)s.curves.size(),
        s.in_flight, s.running_curve_index,
        [&s](int i) { return s.curves[i].label; },
        [&model, &s](int i) { draw_ls_curve_controls(model, s, i); },
        [&s]() { s.add_curve(); });
    if (tabs.active    >= 0) s.active_curve_index = tabs.active;
    if (tabs.to_remove >= 0) model.remove_ls_curve(tabs.to_remove);
}

// Plot LS: каждая кривая раскладывается на N линий по числу экспонент, серия =
// spectrum_idx * N + exponent_idx, цвета через ic_base_color(seq). При mode_2d=true у
// активной кривой рисуется HeatmapView с одной выбранной экспонентой; combo "Exponent"
// над хитмапой переключает плоскость без повторного Run.
// Per-window-state shape — как у draw_bifurcation_plot (обоснование см. там).
static void draw_ls_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                         const ParametricPlotWindow& win,
                         PlotRenderer& renderer, Plot2DView& view,
                         std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) {
    LyapunovSpectrumAnalysisSession& s = model.ls_session;
    auto get_ls_heatmap = [&](int idx) -> HeatmapView& {
        const bool in_range = (idx >= 0 && idx < (int)s.curves.size());
        const int  cfg_cm   = in_range ? s.curves[idx].colormap_idx : -1;
        const int  cfg_exp  = in_range ? s.curves[idx].display_exponent_idx : kNoExponent;
        return get_or_create_heatmap(heatmap_map, idx, cfg_cm, model.heatmap_colormap, cfg_exp);
    };

    if (win.members.empty()) {
        ImGui::TextDisabled("No spectra assigned to this window.");
        return;
    }

    if (win.mode_2d) {
        for (size_t mi = 0; mi < win.members.size(); ++mi) {
            int idx = win.members[mi];
            if (idx < 0 || idx >= (int)s.curves.size()) continue;
            LSCurveConfig& cact = s.curves[idx];
            if (mi > 0) ImGui::Separator();
            ImGui::PushID(idx);

            const unsigned ls_oid = 0x15A20000u + (unsigned)idx;
            HeatmapView& heatmap_ls = get_ls_heatmap(idx);

            {
                auto save_ls = [&]() {
                    if (!model.loaded_name.empty())
                        lib.save_session(model.loaded_name, "_last_ls",
                                         session_to_json_ls(model.ls_session));
                };
                HeatmapToolbarOpts topts;
                topts.persist_colormap = [&](int cm) {
                    cact.colormap_idx = cm;   // persist per-curve only
                    save_ls();
                };
                topts.extras = [&]() {
                    if (!cact.last_run_2d_ok) return;
                    draw_ls_exponent_picker(heatmap_ls, cact.result_2d.n_exponents,
                                            [&](int j) {
                                                cact.display_exponent_idx = j;
                                                save_ls();
                                            });
                };
                draw_heatmap_toolbar(heatmap_ls, topts);
            }

            if (!cact.last_run_2d_ok || cact.result_2d.values.empty()) {
                ImGui::TextDisabled("No 2D data yet. Press Run.");
                ImGui::PopID();
                continue;
            }

            // Подписи осей по реальным selected-полям свипа — общий auto_axis_name.
            heatmap_ls.x_axis.name = auto_axis_name(s.params, s.vars, cact.param_index,
                                                    cact.sweep_over_var, cact.var_sweep_index,
                                                    cact.sweep_over_h);
            heatmap_ls.y_axis.name = auto_axis_name(s.params, s.vars, cact.param_index_2,
                                                    cact.sweep_over_var_2, cact.var_sweep_index_2,
                                                    cact.sweep_over_h_2);
            heatmap_ls.x_axis.log_scale = cact.log_scale;
            heatmap_ls.y_axis.log_scale = cact.log_scale_2;

            bool fit = cact.fit_request_2d;
            if (fit) cact.fit_request_2d = false;

            // Плоскость по выбранной экспоненте (см. ls_resolve_plane).
            const double* plane_ptr = nullptr;
            double vmin = 0.0, vmax = 0.0;
            int    gen  = 0;
            ls_resolve_plane(cact, heatmap_ls.display_exponent_idx,
                             plane_ptr, vmin, vmax, gen);

            const bool busy = s.in_flight && s.is_2d_run && idx == s.running_curve_index;
            // Ctrl+T: обе оси карты — диапазоны свипа (см. Bif 2D).
            const std::vector<ViewRangeTarget> vrt = { vr_target(s.curves, idx,
                &LSCurveConfig::param_lo_text,   &LSCurveConfig::param_hi_text,
                &LSCurveConfig::param_lo_2_text, &LSCurveConfig::param_hi_2_text) };
            heatmap_ls.popup_extras = [&cact, &cb, busy, &vrt, &model, &heatmap_ls]() {
                draw_export_menu_item(busy, cb, [&cact](const std::string& p) {
                    data_export::export_ls2d(cact.result_2d, p);
                });
                draw_view_range_menu(vrt, heatmap_ls.x_axis, heatmap_ls.y_axis, heatmap_ls.swap_axes, &model);
            };

            heatmap_ls.cell_regime = regime_cell_label(cact.result_2d.flags, cact.result_2d.n_pts);
            heatmap_ls.cell_not_computed = cancelled_cell_mask(cact.result_2d.flags, cact.result_2d.n_pts);
            ImVec2 avail = ImGui::GetContentRegionAvail();
            ImVec2 origin = ImGui::GetCursorScreenPos();
            heatmap_ls.render(renderer, origin, avail,
                              /*owner_id*/ ls_oid, gen,
                              cact.result_2d.n_pts, cact.result_2d.n_pts,
                              plane_ptr,
                              cact.result_2d.param_lo,   cact.result_2d.param_hi,
                              cact.result_2d.param_lo_2, cact.result_2d.param_hi_2,
                              vmin, vmax,
                              fit);
            handle_view_range_keys(vrt, heatmap_ls.x_axis, heatmap_ls.y_axis,
                                   plot_window_active(), heatmap_ls.swap_axes, &model);
            ImGui::PopID();
        }
        return;
    }

    bool any_data = false;
    for (int idx : win.members) {
        if (idx < 0 || idx >= (int)s.curves.size()) continue;
        const auto& c = s.curves[idx];
        if (c.last_run_ok && !c.result.spectrum.empty()) { any_data = true; break; }
    }
    if (!any_data) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }

    // Подпись X — см. configure_sweep_x_axis_from.
    configure_sweep_x_axis_from(view, win.members, s.curves, s.params, s.vars);

    // Подсчитываем общее число серий этого окна — sum(n_exponents per member).
    size_t total_series = 0;
    for (int idx : win.members) {
        if (idx < 0 || idx >= (int)s.curves.size()) continue;
        const auto& c = s.curves[idx];
        int N = c.result.n_exponents > 0 ? c.result.n_exponents : (int)s.vars.size();
        total_series += (size_t)N;
    }

    auto& bufs = window_point_bufs(win.id, total_series);
    // Узлы сетки перезаполняются каждый кадр: данные могли пересчитаться.
    view.snap_x_nodes      = nullptr;
    view.snap_x_node_count = 0;

    std::vector<PlotSeriesInput> series_in;
    std::vector<bool> init_vis, glob_vis;
    series_in.reserve(total_series);
    init_vis.reserve(total_series);
    glob_vis.reserve(total_series);

    bool any_fit = false;
    int  data_gen = 0;
    size_t buf_cursor = 0;
    int    series_idx = 0;

    for (int idx : win.members) {
        if (idx < 0 || idx >= (int)s.curves.size()) continue;
        LSCurveConfig& c = s.curves[idx];
        int N = c.result.n_exponents > 0 ? c.result.n_exponents : (int)s.vars.size();

        // X — по диапазону, с которым шёл Run (см. LLE-плот).
        double lo = c.result.param_lo;
        double hi = c.result.param_hi;
        int npts = c.result.n_pts;
        // При backward-continuation точка k считалась для hi-(hi-lo)*k/(n-1)
        // (см. run_ls1d_cpu) — иначе кривая была бы зеркальной.
        const bool rev = c.result.continuation_reverse;
        bool have = c.last_run_ok && !c.result.spectrum.empty();
        if (have && view.snap_x_node_count == 0)
            set_snap_nodes(view, win.id, npts, lo, hi, c.log_scale, rev, c.continuation);

        for (int j = 0; j < N; ++j) {
            auto& buf = bufs[buf_cursor++];
            buf.clear();
            int total_pts = 0;

            if (have) {
                for (int k = 0; k < npts; ++k) {
                    if (k < (int)c.result.flags.size() &&
                        !regime_is_oscillation(c.result.flags[k])) continue;
                    if (k >= (int)c.result.spectrum.size()) continue;
                    const auto& row = c.result.spectrum[k];
                    if (j >= (int)row.size()) continue;
                    double x = sweep_value_at(k, npts, lo, hi, c.log_scale, rev, c.continuation);
                    double y = row[j];
                    if (!std::isfinite(y)) continue;
                    buf.push_back((double)x);
                    buf.push_back((double)y);
                    ++total_pts;
                }
            }

            PlotSeriesInput si;
            si.points   = buf.empty() ? nullptr : buf.data();
            si.n_points = total_pts;
            si.color    = ic_base_color(series_idx++);
            // label: "<spectrum> Lj" если N>1, иначе просто <spectrum>.
            std::string lab = (N > 1) ? (c.label + " L" + std::to_string(j + 1)) : c.label;
            si.label    = lab;
            series_in.push_back(si);
            init_vis.push_back(true);
            glob_vis.push_back(true);   // membership in this window IS the visibility gate
        }

        data_gen = data_gen * 31 + c.data_generation;
        if (c.fit_request) { any_fit = true; c.fit_request = false; }
    }

    // Ctrl+T: по Y идут показатели спектра, входа в расчёте у них нет.
    std::vector<ViewRangeTarget> vrt;
    for (int idx : win.members)
        if (idx >= 0 && idx < (int)s.curves.size())
            vrt.push_back(vr_target(s.curves, idx,
                &LSCurveConfig::param_lo_text, &LSCurveConfig::param_hi_text));

    view.popup_extras = [&s, &cb, &vrt, &model, &view]() {
        draw_export_submenu("ls", (int)s.curves.size(),
            [&s](int i) { return s.curves[i].label; },
            [&s](int i) { return s.curves[i].last_run_ok; },
            [&s](int i) { return s.in_flight && !s.is_2d_run && i == s.running_curve_index; },
            [&s](int i, const std::string& p) { data_export::export_ls1d(s.curves[i].result, p); },
            cb);
        draw_view_range_menu(vrt, view.x_axis, view.y_axis, /*swapped*/ false, &model);
    };

    // Snap X к узлам первой LS-кривой этого окна (см. apply_snap_x_from_first_member).
    apply_snap_x_from_first_member(view, win.members, s.curves);

    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    view.render(renderer, origin, avail, /*owner_id*/ 0x15A1E0, data_gen,
                series_in, init_vis, glob_vis, any_fit);
    handle_view_range_keys(vrt, view.x_axis, view.y_axis, plot_window_active(), /*swapped*/ false, &model);
}

// Metrics: контролы (per-config в табе) + графики. UX зеркалит LLE: те же
// свип, 2D-режим и блок интегрирования, плюс выбор сигнала и набора метрик.
// Какую из посчитанных метрик рисовать, решает таб-бар окна графика
// (ParametricPlotWindow::metric) — как вид в окне Basins.

// Подсказка к чекбоксу метрики — определение ровно в том виде, как считает ядро.
static const char* signal_metric_tooltip(int m) {
    switch (m) {
    case SIGM_MAX:   return "Maximum of the recorded signal over the window after the transient.";
    case SIGM_MIN:   return "Minimum of the recorded signal.";
    case SIGM_RANGE: return "Max - Min: peak-to-peak amplitude.";
    case SIGM_MEAN:  return "Time average of the recorded signal.";
    case SIGM_MEAN_FREQ:
        return "Mean of 1/T_i over the inter-peak intervals T_i.\n"
               "Peaks are found exactly as for bifurcation diagrams\n"
               "(Settings -> Peaks). Fixed point: 0. No interval in the window: NaN.";
    case SIGM_MEDIAN_FREQ:
        return "1 / median(T_i) over the inter-peak intervals.\n"
               "Needs every interval stored on the GPU, so it is the only\n"
               "metric that costs memory (up to max_amount_of_peaks per point).";
    case SIGM_VARIANCE:
        return "Variance var(x) - Hjorth activity, the signal power.";
    case SIGM_HJORTH_MOBILITY:
        return "Hjorth mobility = sqrt(var(x') / var(x)), rad per time unit.\n"
               "x' is the finite difference of the recorded samples,\n"
               "so the sample step is h * pre_scaller.";
    case SIGM_HJORTH_COMPLEXITY:
        return "Hjorth complexity = mobility(x') / mobility(x).\n"
               "Equals 1 for a pure sine, grows as the spectrum widens.";
    case SIGM_VOLUME:
        return "Volume of the box the attractor fits in:\n"
               "prod over ALL state variables of (max x_i - min x_i).\n"
               "3rd order: (x_max - x_min)(y_max - y_min)(z_max - z_min).\n"
               "Does not depend on the Signal choice.";
    case SIGM_INT_MAX:   return "Longest inter-peak interval T_i (time units).\nNo interval in the window: NaN.";
    case SIGM_INT_MIN:   return "Shortest inter-peak interval T_i (time units).\nNo interval in the window: NaN.";
    case SIGM_INT_RANGE: return "max(T_i) - min(T_i): 0 on a period-1 orbit, grows with period and chaos.";
    case SIGM_INT_MEAN:  return "Mean inter-peak interval (time units).\n"
                                "Peaks are the same as for the frequencies.";
    default: return "";
    }
}

static void draw_metrics_config_controls(AppModel& model, SignalMetricsAnalysisSession& s, int idx) {
    SignalMetricsConfig& c = s.configs[idx];
    model.broadcast_source_tab = BroadcastTab::Metrics;
    model.broadcast_source_idx = idx;

    ImGui::SetNextItemWidth(kComboW);
    if (InputTextStr("Label", c.label))
        c.label_is_manual = !c.label.empty();   // empty → back to auto
    ImGui::Separator();

    draw_scheme_combo("Scheme", c.scheme, s.custom_schemes, {}, &s.enabled_builtin_schemes,
                      &model, model.is_map, &s.wrapper_schemes);
    ImGui::Separator();

    draw_sweep_target_combo("Sweep", s.params, s.vars,
                            c.param_index, c.sweep_over_var, c.var_sweep_index,
                            c.sweep_over_h,
                            c.mode_2d ? &c.sweep_over_h_2 : nullptr,
                            /*note_when_empty*/ true, kComboW,
                            scheme_uses_symmetry(c.scheme, s.custom_schemes),
                            &model, BroadcastField::SweepTarget, !model.is_map,
                            &c.ad_axis, &c.adaptive);
    InputNumStr(c.sweep_over_h ? "h lo" : "Param lo", c.param_lo_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepLo, {},
                                        c.sweep_over_h ? "h lo" : "Param lo", c.param_lo_text); });
    InputNumStr(c.sweep_over_h ? "h hi" : "Param hi", c.param_hi_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepHi, {},
                                        c.sweep_over_h ? "h hi" : "Param hi", c.param_hi_text); });
    ImGui::Checkbox("Log scale##met_log", &c.log_scale);
    if (c.log_scale) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
    InputNumStr("Resolution", c.n_pts_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::Resolution, {},
                                        "Resolution", c.n_pts_text); });

    draw_continuation_device_block(c, "met", c.mode_2d || c.sweep_over_var, c.mode_2d);

    ImGui::Separator();
    ImGui::Checkbox("2D mode (heatmap)##met_2d", &c.mode_2d);
    if (c.mode_2d) {
        ImGui::Indent();
        if (!s.params.empty() || !s.vars.empty())
            draw_sweep_target_combo("Sweep Y", s.params, s.vars,
                                    c.param_index_2, c.sweep_over_var_2, c.var_sweep_index_2,
                                    c.sweep_over_h_2, &c.sweep_over_h,
                                    /*note_when_empty*/ false, kComboW,
                                    scheme_uses_symmetry(c.scheme, s.custom_schemes),
                                    &model, BroadcastField::SweepTarget2, !model.is_map,
                                    &c.ad_axis_2, &c.adaptive);
        InputNumStr(c.sweep_over_h_2 ? "h2 lo" : "Param2 lo", c.param_lo_2_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepLo2, {},
                                        c.sweep_over_h_2 ? "h2 lo" : "Param2 lo", c.param_lo_2_text); });
        InputNumStr(c.sweep_over_h_2 ? "h2 hi" : "Param2 hi", c.param_hi_2_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepHi2, {},
                                        c.sweep_over_h_2 ? "h2 hi" : "Param2 hi", c.param_hi_2_text); });
        ImGui::Checkbox("Log scale##met_log2", &c.log_scale_2);
        if (c.log_scale_2) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
        ImGui::TextDisabled("Grid is square (Resolution applies to both axes).");
        ImGui::Unindent();
    }

    ImGui::Separator();
    draw_writable_var_combo(s.vars, c.writable_var, "Signal##met_wv");

    // Набор метрик. Суммы под все, кроме медианы, почти бесплатны — галки в
    // первую очередь решают, что попадёт в табы графика и в CSV.
    if (ImGui::CollapsingHeader("Metrics##met_sel", ImGuiTreeNodeFlags_DefaultOpen)) {
        auto box = [&c](int m) {
            ImGui::CheckboxFlags(signal_metric_name(m), &c.metric_mask, 1 << m);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", signal_metric_tooltip(m));
        };
        ImGui::TextDisabled("Amplitude");
        box(SIGM_MAX);   ImGui::SameLine(); box(SIGM_MIN);
        box(SIGM_RANGE); ImGui::SameLine(); box(SIGM_MEAN);
        ImGui::SameLine(); box(SIGM_VOLUME);
        ImGui::TextDisabled("Inter-peak intervals");
        box(SIGM_INT_MAX);   ImGui::SameLine(); box(SIGM_INT_MIN);
        box(SIGM_INT_RANGE); ImGui::SameLine(); box(SIGM_INT_MEAN);
        ImGui::TextDisabled("Frequency (by peaks)");
        box(SIGM_MEAN_FREQ); ImGui::SameLine(); box(SIGM_MEDIAN_FREQ);
        ImGui::TextDisabled("Hjorth parameters");
        box(SIGM_VARIANCE); ImGui::SameLine(); box(SIGM_HJORTH_MOBILITY);
        ImGui::SameLine(); box(SIGM_HJORTH_COMPLEXITY);
        if (ImGui::SmallButton("All##met_all"))  c.metric_mask = kSignalMetricAllMask;
        ImGui::SameLine();
        if (ImGui::SmallButton("None##met_none")) c.metric_mask = 0;
        if (c.metric_mask == 0)
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "Select at least one metric.");
    }

    {
        IntegrationFields f;
        f.h           = &c.h_text;
        f.symmetry_s  = &c.symmetry_s;
        f.cd_fp_iters = &c.cd_fp_iters;
        f.newton      = &c.newton;
        f.t_max       = &c.t_max_text;
        f.transient   = &c.transient_text;
        f.pre_scaller = &c.pre_scaller_text;
        f.max_value   = &c.max_value_text;
        // Метрики — сетка или узлы шага; Хьорт при адаптивном шаге не считается.
        f.adaptive      = &c.adaptive;
        f.adaptive_opts = kAdUiRaw | kAdUiInterp | kAdUiMinMax;
        draw_integration_block("Integration##met_int", c.scheme, s.custom_schemes, f, &model, model.is_map);
    }

    draw_named_num_fields("Initial conditions##met_ic", s.vars,   c.initial_conditions,
                          nullptr, &model, BroadcastField::InitCondition);
    draw_named_num_fields("Parameters##met_par",        s.params, c.param_values,
                          nullptr, &model, BroadcastField::Param);
    draw_csv_output_block("CSV output##met_csv", "Save to file", c.csv_save_enabled,
                          "##met_csv_path", c.csv_output_path,
                          "One table: x [, y], regime, then every selected metric.");

    const bool ok = c.mode_2d ? c.last_run_2d_ok : c.last_run_ok;
    const SignalMetricsResult& r = c.mode_2d ? c.result_2d : c.result;
    if (ok) {
        if (c.mode_2d)
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "OK: %dx%d grid", r.n_pts, r.n_pts);
        else
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "OK: n_pts=%d", r.n_pts);
        draw_regime_summary(r.flags, &c.regime_counts, c.mode_2d ? c.data_generation_2d : c.data_generation);
        draw_adaptive_stats(r.ad_stats);
    }
    else if (!c.last_error.empty()) {
        draw_error_box("##met_err", c.last_error);
    }
}

static void draw_metrics_controls(AppModel& model, SystemLibrary& /*lib*/) {
    SignalMetricsAnalysisSession& s = model.metrics_session;
    const TabBarResult tabs = draw_config_tab_bar(
        "##met_tabs", "met_tab_", (int)s.configs.size(),
        s.in_flight, s.running_config_index,
        [&s](int i) { return s.configs[i].label; },
        [&model, &s](int i) { draw_metrics_config_controls(model, s, i); },
        [&s]() { s.add_config(); });
    if (tabs.active    >= 0) s.active_config_index = tabs.active;
    if (tabs.to_remove >= 0) model.remove_metrics_config(tabs.to_remove);
}

// Таб-бар выбора метрики над графиком. Табы — метрики, посчитанные хотя бы у
// одного члена окна (до первого прогона — отмеченные в конфигах). Выбор живёт
// в win.metric и сохраняется вместе с окнами; ImGui помнит выбранный таб сам,
// поэтому при расхождении (окно только что загружено, выбранной метрики больше
// нет) таб выставляется явно через SetSelected.
static void draw_metric_tab_bar(AppModel& model, ParametricPlotWindow& win, int avail_mask) {
    static std::map<int, int> shown;   // win.id -> метрика, которую таб-бар показывал
    if (avail_mask == 0) avail_mask = kSignalMetricAllMask;
    bool force = false;
    if (win.metric < 0 || win.metric >= SIGM_COUNT || !((avail_mask >> win.metric) & 1)) {
        for (int m : kSignalMetricDisplayOrder)
            if ((avail_mask >> m) & 1) { win.metric = m; break; }
        model.parametric_plot_windows_dirty = true;
        force = true;
    }
    auto it = shown.find(win.id);
    if (it == shown.end() || it->second != win.metric) force = true;

    if (ImGui::BeginTabBar("##metric_tabs", ImGuiTabBarFlags_FittingPolicyScroll)) {
        for (int m : kSignalMetricDisplayOrder) {
            if (!((avail_mask >> m) & 1)) continue;
            const ImGuiTabItemFlags fl = (force && m == win.metric) ? ImGuiTabItemFlags_SetSelected : 0;
            if (ImGui::BeginTabItem(signal_metric_name(m), nullptr, fl)) {
                if (!force && win.metric != m) {
                    win.metric = m;
                    model.parametric_plot_windows_dirty = true;
                }
                ImGui::EndTabItem();
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", signal_metric_tooltip(m));
        }
        ImGui::EndTabBar();
    }
    shown[win.id] = win.metric;
}

static void draw_metrics_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                              ParametricPlotWindow& win,
                              PlotRenderer& renderer, Plot2DView& view,
                              std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) {
    SignalMetricsAnalysisSession& s = model.metrics_session;

    if (win.members.empty()) {
        ImGui::TextDisabled("No configs assigned to this window.");
        return;
    }

    // Какие метрики есть у членов окна: у посчитанных — маска прогона, у ещё не
    // посчитанных — маска конфига (чтобы табы были видны до первого Run).
    int avail = 0;
    for (int idx : win.members) {
        if (idx < 0 || idx >= (int)s.configs.size()) continue;
        const SignalMetricsConfig& c = s.configs[idx];
        const bool ok = win.mode_2d ? c.last_run_2d_ok : c.last_run_ok;
        avail |= ok ? (win.mode_2d ? c.result_2d.metric_mask : c.result.metric_mask) : c.metric_mask;
    }
    draw_metric_tab_bar(model, win, avail);
    const int m = win.metric;

    if (win.mode_2d) {
        for (size_t mi = 0; mi < win.members.size(); ++mi) {
            const int idx = win.members[mi];
            if (idx < 0 || idx >= (int)s.configs.size()) continue;
            SignalMetricsConfig& cact = s.configs[idx];
            if (mi > 0) ImGui::Separator();
            ImGui::PushID(idx);

            HeatmapView& heatmap = get_or_create_heatmap(heatmap_map, idx, cact.colormap_idx,
                                                          model.heatmap_colormap);
            {
                HeatmapToolbarOpts topts;
                topts.persist_colormap = [&](int cm) {
                    cact.colormap_idx = cm;
                    if (!model.loaded_name.empty())
                        lib.save_session(model.loaded_name, "_last_metrics",
                                         session_to_json_metrics(model.metrics_session));
                };
                draw_heatmap_toolbar(heatmap, topts);
            }

            const SignalMetricsResult& r = cact.result_2d;
            if (!cact.last_run_2d_ok || r.values[m].empty()) {
                ImGui::TextDisabled(cact.last_run_2d_ok ? "This metric was not computed in the last run."
                                                        : "No 2D data yet. Press Run.");
                ImGui::PopID();
                continue;
            }

            heatmap.x_axis.name = auto_axis_name(s.params, s.vars, cact.param_index,
                                                 cact.sweep_over_var, cact.var_sweep_index,
                                                 cact.sweep_over_h);
            heatmap.y_axis.name = auto_axis_name(s.params, s.vars, cact.param_index_2,
                                                 cact.sweep_over_var_2, cact.var_sweep_index_2,
                                                 cact.sweep_over_h_2);
            heatmap.x_axis.log_scale = r.log_scale;
            heatmap.y_axis.log_scale = r.log_scale_2;

            bool fit = cact.fit_request_2d;
            if (fit) cact.fit_request_2d = false;

            const bool busy = s.in_flight && s.is_2d_run && idx == s.running_config_index;
            const std::vector<ViewRangeTarget> vrt = { vr_target(s.configs, idx,
                &SignalMetricsConfig::param_lo_text,   &SignalMetricsConfig::param_hi_text,
                &SignalMetricsConfig::param_lo_2_text, &SignalMetricsConfig::param_hi_2_text) };
            heatmap.popup_extras = [&cact, &cb, busy, &vrt, &model, &heatmap]() {
                draw_export_menu_item(busy, cb, [&cact](const std::string& p) {
                    std::string e;
                    signal_metrics_write_csv(cact.result_2d, p, e);
                });
                draw_view_range_menu(vrt, heatmap.x_axis, heatmap.y_axis, heatmap.swap_axes, &model);
            };

            // Поколение данных несёт и номер метрики: смена таба — это другие
            // значения в той же ячейке, и текстура обязана перестроиться.
            const int gen = cact.data_generation_2d * SIGM_COUNT + m;
            heatmap.cell_regime = regime_cell_label(r.flags, r.n_pts);
            heatmap.cell_not_computed = cancelled_cell_mask(r.flags, r.n_pts);
            ImVec2 avail_sz = ImGui::GetContentRegionAvail();
            ImVec2 origin   = ImGui::GetCursorScreenPos();
            heatmap.render(renderer, origin, avail_sz,
                           /*owner_id*/ 0x516E0000u + (unsigned)idx, gen,
                           r.n_pts, r.n_pts, r.values[m].data(),
                           r.param_lo,   r.param_hi,
                           r.param_lo_2, r.param_hi_2,
                           r.min_val[m], r.max_val[m],
                           fit);
            handle_view_range_keys(vrt, heatmap.x_axis, heatmap.y_axis, plot_window_active(),
                                   heatmap.swap_axes, &model);
            ImGui::PopID();
        }
        return;
    }

    bool any_data = false;
    for (int idx : win.members) {
        if (idx < 0 || idx >= (int)s.configs.size()) continue;
        const auto& c = s.configs[idx];
        if (c.last_run_ok && !c.result.values[m].empty()) { any_data = true; break; }
    }
    if (!any_data) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }

    configure_sweep_x_axis_from(view, win.members, s.configs, s.params, s.vars);
    view.y_axis.name = signal_metric_axis_label(m);

    auto& bufs = window_point_bufs(win.id, win.members.size());
    view.snap_x_nodes      = nullptr;
    view.snap_x_node_count = 0;

    std::vector<PlotSeriesInput> series_in;
    std::vector<bool> init_vis, glob_vis;
    series_in.reserve(win.members.size());
    init_vis.reserve(win.members.size());
    glob_vis.reserve(win.members.size());

    bool any_fit = false;
    int  data_gen = m;

    for (size_t mi = 0; mi < win.members.size(); ++mi) {
        const int idx = win.members[mi];
        if (idx < 0 || idx >= (int)s.configs.size()) continue;
        SignalMetricsConfig& c = s.configs[idx];
        auto& buf = bufs[mi];
        buf.clear();
        int total_pts = 0;

        const std::vector<double>& ys = c.result.values[m];
        if (c.last_run_ok && !ys.empty()) {
            const double lo = c.result.param_lo;
            const double hi = c.result.param_hi;
            const int npts  = c.result.n_pts;
            // Continuation: точка k лежит в узле цепочки (при backward — от hi).
            const bool rev  = c.result.continuation_reverse;
            const bool cont = c.result.continuation;
            if (view.snap_x_node_count == 0)
                set_snap_nodes(view, win.id, npts, lo, hi, c.result.log_scale, rev, cont);
            for (int k = 0; k < npts && k < (int)ys.size(); ++k) {
                const double y = ys[k];
                if (!std::isfinite(y)) continue;
                buf.push_back(sweep_value_at(k, npts, lo, hi, c.result.log_scale, rev, cont));
                buf.push_back(y);
                ++total_pts;
            }
        }

        PlotSeriesInput si;
        si.points   = buf.empty() ? nullptr : buf.data();
        si.n_points = total_pts;
        si.color    = ic_base_color((int)mi);
        si.label    = c.label;
        series_in.push_back(si);
        init_vis.push_back(true);
        glob_vis.push_back(true);

        data_gen = data_gen * 31 + c.data_generation;
        if (c.fit_request) { any_fit = true; c.fit_request = false; }
    }

    // Смена метрики — другие единицы по Y: без автоподгонки кривая уехала бы
    // за пределы старого вида.
    {
        static std::map<int, int> last_metric;
        auto it = last_metric.find(win.id);
        if (it == last_metric.end() || it->second != m) any_fit = true;
        last_metric[win.id] = m;
    }

    std::vector<ViewRangeTarget> vrt;
    for (int idx : win.members)
        if (idx >= 0 && idx < (int)s.configs.size())
            vrt.push_back(vr_target(s.configs, idx,
                &SignalMetricsConfig::param_lo_text, &SignalMetricsConfig::param_hi_text));

    view.popup_extras = [&s, &cb, &vrt, &model, &view]() {
        draw_export_submenu("met", (int)s.configs.size(),
            [&s](int i) { return s.configs[i].label; },
            [&s](int i) { return s.configs[i].last_run_ok; },
            [&s](int i) { return s.in_flight && !s.is_2d_run && i == s.running_config_index; },
            [&s](int i, const std::string& p) {
                std::string e;
                signal_metrics_write_csv(s.configs[i].result, p, e);
            },
            cb);
        draw_view_range_menu(vrt, view.x_axis, view.y_axis, /*swapped*/ false, &model);
    };

    apply_snap_x_from_first_member(view, win.members, s.configs);

    ImVec2 avail_sz = ImGui::GetContentRegionAvail();
    ImVec2 origin   = ImGui::GetCursorScreenPos();
    view.render(renderer, origin, avail_sz, /*owner_id*/ 0x516E1D, data_gen,
                series_in, init_vis, glob_vis, any_fit);
    handle_view_range_keys(vrt, view.x_axis, view.y_axis, plot_window_active(), /*swapped*/ false, &model);
}

// Parametric plot windows — shared setup-row helper. The row (Label | Type | Members... | X)
// is rendered both in the Plot windows manager on the settings panel and at the top of each
// per-window Begin(), so users can retype the chart without going back to the panel; the
// helpers live at file scope so both callsites share one implementation.
struct ParamPlotMatchItem { int index; std::string label; };

[[nodiscard]] static std::vector<ParamPlotMatchItem>
parametric_matching_items(AppModel& model, ParametricPlotWindow::Kind kind, bool mode_2d) {
    std::vector<ParamPlotMatchItem> out;
    if (kind == ParametricPlotWindow::Kind::Bifurcation) {
        auto& ds = model.bifurcation_session.diagrams;
        for (int i = 0; i < (int)ds.size(); ++i)
            if (ds[i].mode_2d == mode_2d) out.push_back({ i, ds[i].label });
    } else if (kind == ParametricPlotWindow::Kind::LLE) {
        auto& cs = model.lle_session.curves;
        for (int i = 0; i < (int)cs.size(); ++i)
            if (cs[i].mode_2d == mode_2d) out.push_back({ i, cs[i].label });
    } else if (kind == ParametricPlotWindow::Kind::LS) {
        auto& cs = model.ls_session.curves;
        for (int i = 0; i < (int)cs.size(); ++i)
            if (cs[i].mode_2d == mode_2d) out.push_back({ i, cs[i].label });
    } else {
        auto& cs = model.metrics_session.configs;
        for (int i = 0; i < (int)cs.size(); ++i)
            if (cs[i].mode_2d == mode_2d) out.push_back({ i, cs[i].label });
    }
    return out;
}

// Colored 1D is not a distinct combo entry — it's a toggle inside the
// plot body (see draw_bifurcation_plot). The Type combo only splits 1D/2D.
static const char* kParametricTypeNames[] = {
    "Bifurcation 1D", "Bifurcation 2D", "LLE 1D", "LLE 2D", "LS 1D", "LS 2D",
    "Metrics 1D", "Metrics 2D"
};

static int parametric_type_index_of(ParametricPlotWindow::Kind kind, bool mode_2d) {
    int base = kind == ParametricPlotWindow::Kind::Bifurcation ? 0
             : kind == ParametricPlotWindow::Kind::LLE ? 2
             : kind == ParametricPlotWindow::Kind::LS  ? 4 : 6;
    return base + (mode_2d ? 1 : 0);
}

static void parametric_type_from_index(int t, ParametricPlotWindow::Kind& kind, bool& mode_2d) {
    kind    = (t < 2) ? ParametricPlotWindow::Kind::Bifurcation
            : (t < 4) ? ParametricPlotWindow::Kind::LLE
            : (t < 6) ? ParametricPlotWindow::Kind::LS : ParametricPlotWindow::Kind::Metrics;
    mode_2d = (t % 2) == 1;
}

// Draws the Label | Type | Members... | X row for a Parametric plot window.
// Returns true iff the user clicked the row's X — caller unmounts the window.
// Assumes caller has already pushed a PushID(win.id) on the ID stack.
static bool draw_parametric_window_setup_row(AppModel& model, ParametricPlotWindow& win) {
    bool close_clicked = false;

    ImGui::SetNextItemWidth(220);
    if (InputTextStr("##wlabel", win.label)) {
        win.label_is_manual = !win.label.empty();   // empty → back to auto
        model.parametric_plot_windows_dirty = true;
    }
    ImGui::SameLine();

    // Type combo: changing kind/dimension invalidates old member indices
    // (they're only meaningful within the previous session+dimension),
    // so switching type clears members.
    int t = parametric_type_index_of(win.kind, win.mode_2d);
    ImGui::SetNextItemWidth(130);
    if (ImGui::Combo("##wtype", &t, kParametricTypeNames, IM_ARRAYSIZE(kParametricTypeNames))) {
        ParametricPlotWindow::Kind new_kind; bool new_2d;
        parametric_type_from_index(t, new_kind, new_2d);
        if (new_kind != win.kind || new_2d != win.mode_2d) {
            win.kind = new_kind;
            win.mode_2d = new_2d;
            win.colored_1d = false;   // combo no longer selects Colored 1D — reset; toggled above the plot
            win.members.clear();
            model.parametric_plot_windows_dirty = true;
        }
    }
    ImGui::SameLine();

    if (ImGui::Button("Members...")) ImGui::OpenPopup("edit_plot_window_members");
    if (ImGui::BeginPopup("edit_plot_window_members")) {
        auto items = parametric_matching_items(model, win.kind, win.mode_2d);
        if (items.empty()) {
            ImGui::TextDisabled("(none available)");
        } else if (win.mode_2d || win.colored_1d) {
            // 2D / Colored 1D show heatmaps — single-select (radio).
            for (const auto& item : items) {
                bool sel = !win.members.empty() && win.members[0] == item.index;
                std::string lbl = item.label + "##mem" + std::to_string(item.index);
                if (ImGui::RadioButton(lbl.c_str(), sel)) {
                    win.members.assign(1, item.index);
                    model.parametric_plot_windows_dirty = true;
                }
            }
        } else {
            for (const auto& item : items) {
                bool has = std::find(win.members.begin(), win.members.end(), item.index) != win.members.end();
                std::string lbl = item.label + "##mem" + std::to_string(item.index);
                if (ImGui::Checkbox(lbl.c_str(), &has)) {
                    if (has) win.members.push_back(item.index);
                    else win.members.erase(std::remove(win.members.begin(), win.members.end(), item.index), win.members.end());
                    model.parametric_plot_windows_dirty = true;
                }
            }
        }
        ImGui::EndPopup();
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("X")) close_clicked = true;

    return close_clicked;
}

// Parametric mode: dynamic plot windows (mirrors draw_projection_windows in Phase). One ImGui
// window per model.parametric_plot_windows entry, each with its own
// PlotRenderer/Plot2DView/HeatmapView-map keyed by ParametricPlotWindow::id — not vector
// position, which shifts when an earlier window is removed. Closing a window (X) removes it;
// "Reset windows layout" bumps model.parametric_layout_generation, baked into every title, so
// ImGui treats them as brand-new windows and re-cascades default positions.
void draw_parametric_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) {
    static std::map<int, std::unique_ptr<PlotRenderer>> renderers;
    static std::map<int, std::unique_ptr<Plot2DView>>    views;
    static std::map<int, std::map<int, std::unique_ptr<HeatmapView>>> heatmaps;
    // Tracks the kind the Plot2DView was last configured for; when a
    // window's kind changes (Type combo), re-run configure_plot_view so
    // e.g. LLE doesn't inherit Bifurcation's points_mode=true.
    static std::map<int, ParametricPlotWindow::Kind> view_configured_for;
    static int seen_epoch = 0;
    if (session_reloaded(model, AppModel::AppMode::Parametric, seen_epoch)) {
        invalidate_views(views);
        for (auto& kv : heatmaps) invalidate_views(kv.second);
    }

    int to_remove = -1;
    for (int i = 0; i < (int)model.parametric_plot_windows.size(); ++i) {
        ParametricPlotWindow& win = model.parametric_plot_windows[i];

        auto& renderer = renderers[win.id];
        if (!renderer) renderer = std::make_unique<PlotRenderer>();
        auto& view = views[win.id];
        bool fresh_view = !view;
        if (fresh_view) view = std::make_unique<Plot2DView>();
        auto cf_it = view_configured_for.find(win.id);
        if (fresh_view || cf_it == view_configured_for.end() || cf_it->second != win.kind) {
            configure_plot_view(*view, win.kind);
            view_configured_for[win.id] = win.kind;
        }
        auto& hm_map = heatmaps[win.id];

        // "###" makes ImGui hash the ID from the suffix only (win.id + layout generation),
        // independent of the visible label: docking/position stay stable when the user renames
        // the window, and a layout reset changes the ID to re-cascade default positions.
        std::string title = win.label + "###pwin" + std::to_string(win.id)
                           + "_g" + std::to_string(model.parametric_layout_generation);
        bool open = true;
        float ox = 60.0f + (float)(i % 5) * 35.0f, oy = 80.0f + (float)(i % 5) * 35.0f;
        ImGui::SetNextWindowPos(ImVec2(ox, oy), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(700, 550), ImGuiCond_FirstUseEver);
        bool row_close = false;
        if (ImGui::Begin(title.c_str(), &open)) {
            ImGui::PushID(win.id);
            // Duplicate of the settings-panel row (draw_parametric_controls) so
            // the chart type / members / label can be changed without leaving
            // the window. Same helper, same PushID(win.id) scope.
            row_close = draw_parametric_window_setup_row(model, win);
            ImGui::Separator();
            switch (win.kind) {
            case ParametricPlotWindow::Kind::Bifurcation:
                draw_bifurcation_plot(model, lib, cb, win, *renderer, *view, hm_map);
                break;
            case ParametricPlotWindow::Kind::LLE:
                draw_lle_plot(model, lib, cb, win, *renderer, *view, hm_map);
                break;
            case ParametricPlotWindow::Kind::LS:
                draw_ls_plot(model, lib, cb, win, *renderer, *view, hm_map);
                break;
            case ParametricPlotWindow::Kind::Metrics:
                draw_metrics_plot(model, lib, cb, win, *renderer, *view, hm_map);
                break;
            }
            ImGui::PopID();
        }
        ImGui::End();
        if (!open || row_close) to_remove = i;
    }
    if (to_remove >= 0) {
        int id = model.parametric_plot_windows[to_remove].id;
        renderers.erase(id);
        views.erase(id);
        heatmaps.erase(id);
        view_configured_for.erase(id);
        model.remove_parametric_plot_window(to_remove);
    }
}

void draw_parametric_controls(AppModel& model, SystemLibrary& lib) {
    ImGui::Text("Parametric analysis");
    ImGui::TextDisabled("Per-thread parameter sweep via NVRTC + NonLinAnal kernels.");

    // Union used by the Run / Run all... buttons below to disable themselves
    // while any sub-analysis (Bif/LLE/LS) is computing. Kept after the tab's
    // System combo was moved to the top-bar; the top-bar combo has its own
    // wider union across all 7 sessions.
    bool any_in_flight = model.bifurcation_session.in_flight
                      || model.lle_session.in_flight
                      || model.ls_session.in_flight
                      || model.metrics_session.in_flight;

    // Run (active sub-tab, active config) + batch Run all. Кнопка Run единая для Bif/LLE/LS,
    // диспатчится по parametric_active_analysis (0=Bif, 1=LLE, 2=LS) к соответствующему
    // active_*_index. Run all — глобальный по всем трём спискам: попап показывает чекбоксы
    // для каждой диаграммы / кривой / спектра, Run кладёт отмеченные в
    // model.parametric_queue и стартует первый; draw_gui тикает очередь после poll'ов.
    {
        const int kind = model.parametric_active_analysis;
        int  active_idx = -1;
        bool no_active  = false;
        if (kind == 0) {
            active_idx = model.bifurcation_session.active_diagram_index;
            no_active  = model.bifurcation_session.diagrams.empty();
        } else if (kind == 1) {
            active_idx = model.lle_session.active_curve_index;
            no_active  = model.lle_session.curves.empty();
        } else if (kind == 2) {
            active_idx = model.ls_session.active_curve_index;
            no_active  = model.ls_session.curves.empty();
        } else if (kind == 3) {
            active_idx = model.metrics_session.active_config_index;
            no_active  = model.metrics_session.configs.empty();
        }

        std::vector<RunAllGroup> groups;
        groups.push_back(RunAllGroup{
            "Bifurcation", "pbd_", (int)model.bifurcation_session.diagrams.size(),
            [&model](int i) { return model.bifurcation_session.diagrams[i].label; },
            [&model](int i) {
                model.parametric_queue.push_back({ParametricQueueItem::Kind::Bifurcation, i});
            } });
        groups.push_back(RunAllGroup{
            "LLE", "plle_", (int)model.lle_session.curves.size(),
            [&model](int i) { return model.lle_session.curves[i].label; },
            [&model](int i) {
                model.parametric_queue.push_back({ParametricQueueItem::Kind::LLE, i});
            } });
        groups.push_back(RunAllGroup{
            "LS", "pls_", (int)model.ls_session.curves.size(),
            [&model](int i) { return model.ls_session.curves[i].label; },
            [&model](int i) {
                model.parametric_queue.push_back({ParametricQueueItem::Kind::LS, i});
            } });
        groups.push_back(RunAllGroup{
            "Metrics", "pmet_", (int)model.metrics_session.configs.size(),
            [&model](int i) { return model.metrics_session.configs[i].label; },
            [&model](int i) {
                model.parametric_queue.push_back({ParametricQueueItem::Kind::Metrics, i});
            } });

        // Run all блокируется ТОЛЬКО на время расчёта: списков три, и пустой
        // активный не повод запрещать запуск остальных (в отличие от вкладок
        // с одним списком, где block_run_all включает и «нет конфигов»).
        draw_run_and_run_all("##run_all_parametric", any_in_flight, no_active,
                             any_in_flight, 140.0f,
                             [&model, kind, active_idx]() {
                                 if (active_idx < 0) return;
                                 if (!model.parametric_engine)
                                     model.parametric_engine = std::make_unique<ParametricEngine>();
                                 if      (kind == 0) model.bifurcation_session.run_async(*model.parametric_engine, active_idx);
                                 else if (kind == 1) model.lle_session.run_async(*model.parametric_engine, active_idx);
                                 else if (kind == 2) model.ls_session.run_async(*model.parametric_engine, active_idx);
                                 else if (kind == 3) model.metrics_session.run_async(*model.parametric_engine, active_idx);
                             },
                             {}, groups,
                             [&model]() { model.start_next_in_parametric_queue(); },
                             model.parametric_queue.size());
    }
    ImGui::Separator();

    if (ImGui::BeginTabBar("##parm_top")) {
        if (ImGui::BeginTabItem("Bifurcation")) {
            model.parametric_active_analysis = 0;
            draw_bifurcation_controls(model, lib);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("LLE")) {
            model.parametric_active_analysis = 1;
            draw_lle_controls(model, lib);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("LS")) {
            model.parametric_active_analysis = 2;
            draw_ls_controls(model, lib);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Metrics")) {
            model.parametric_active_analysis = 3;
            draw_metrics_controls(model, lib);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    // Plot windows: dynamic list, mirrors Phase's Projections section. A sibling of the
    // Bif/LLE/LS tabs above (not nested in any of them) — always visible, since a plot window
    // can be any (kind, dimension) combo. Each row has its own Type combo so the chart type
    // can be changed after creation instead of being fixed at "Add".
    ImGui::Separator();
    ImGui::SeparatorText("Plot windows");

    int win_to_remove = -1;
    for (int i = 0; i < (int)model.parametric_plot_windows.size(); ++i) {
        ParametricPlotWindow& win = model.parametric_plot_windows[i];
        ImGui::PushID(win.id);
        if (draw_parametric_window_setup_row(model, win)) win_to_remove = i;
        ImGui::PopID();
    }
    if (win_to_remove >= 0) model.remove_parametric_plot_window(win_to_remove);

    if (ImGui::Button("Add window")) {
        model.add_parametric_plot_window(ParametricPlotWindow::Kind::Bifurcation, false, {});
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset windows layout")) { model.parametric_layout_generation++; }
}

// Загрузка Order из файлов засевает кэш автосохранения (autosave_order_session)
// только что загруженным состоянием: файл перепишется при первой настоящей
// правке, а не сразу после загрузки. Иначе битый файл, отвергнутый
// apply_session_json, молча заменился бы умолчаниями, хотя пользователь ещё
// ничего не менял.
void order_autosave_seed(AppModel& model) {
    model.order_autosave_target       = model.loaded_name;
    model.order_autosave_json         = session_to_json_order(model.order_session);
    model.order_windows_autosave_json = session_to_json_order_windows(model.order_plot_windows);
}
