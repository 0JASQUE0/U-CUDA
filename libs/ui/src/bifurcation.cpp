#include "gui_internal.h"

namespace ucuda::ui_detail {


// Parametric: контролы + scatter-plot 1D-бифуркации через наш GL-renderer
// Рисует контролы одной БД внутри её таба. Возвращает true, если пользователь
// нажал Run для этой БД (внешний код может также взвести Run через Ctrl+R).
void draw_diagram_controls(BifurcationAnalysisSession& s, int idx) {
    BifurcationDiagramConfig& bd = s.diagrams[idx];

    ImGui::SetNextItemWidth(kComboW);
    if (InputTextStr("Label", bd.label))
        bd.label_is_manual = !bd.label.empty();   // empty → back to auto
    ImGui::Separator();

    // Scheme (built-in + custom)
    draw_scheme_combo("Scheme", bd.scheme, s.custom_schemes, {}, &s.enabled_builtin_schemes);
    draw_adaptive_controls(bd.adaptive);
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
                            scheme_uses_symmetry(bd.scheme, s.custom_schemes));
    InputNumStr(bd.sweep_over_h ? "h lo" : "Param lo", bd.param_lo_text, kFieldW);
    InputNumStr(bd.sweep_over_h ? "h hi" : "Param hi", bd.param_hi_text, kFieldW);
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
                                    scheme_uses_symmetry(bd.scheme, s.custom_schemes));
        InputNumStr(bd.sweep_over_h_2 ? "h2 lo" : "Param2 lo", bd.param_lo_2_text, kFieldW);
        InputNumStr(bd.sweep_over_h_2 ? "h2 hi" : "Param2 hi", bd.param_hi_2_text, kFieldW);
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
    InputNumStr("Resolution", bd.n_pts_text, kFieldW);
    if (!bd.mode_2d)
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
        f.t_max       = &bd.t_max_text;
        f.transient   = &bd.transient_text;
        f.pre_scaller = &bd.pre_scaller_text;
        f.max_value   = &bd.max_value_text;
        draw_integration_block("Integration##bd_int", bd.scheme, s.custom_schemes, f);
    }

    draw_named_num_fields("Initial conditions##bd_ic", s.vars,   bd.initial_conditions);
    draw_named_num_fields("Parameters##bd_par",        s.params, bd.param_values);
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
            draw_regime_summary(bd.result_2d.flags);
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
        }
        else if (!bd.last_error.empty()) {
            draw_error_box("##par_err", bd.last_error);
        }
    }
}


void draw_bifurcation_controls(AppModel& model, SystemLibrary& /*lib*/) {
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
        [&s](int i) { draw_diagram_controls(s, i); },
        [&s]() { s.add_diagram(); },
        s.request_select_diagram);
    s.request_select_diagram = -1;   // запрос потреблён этим кадром
    if (tabs.active    >= 0) s.active_diagram_index = tabs.active;
    if (tabs.to_remove >= 0) model.remove_bifurcation_diagram(tabs.to_remove);
}


// Маппинг window-kind → нейтральный ParamPlotKind. Сама конфигурация живёт в
// configure_param_plot_view (см. блок общих хелперов вверху файла), чтобы
// Parametric и Custom настраивали одинаковые 1D-графики одним и тем же кодом.
ParamPlotKind to_param_plot_kind(ParametricPlotWindow::Kind kind) {
    switch (kind) {
    case ParametricPlotWindow::Kind::LLE: return ParamPlotKind::LLE;
    case ParametricPlotWindow::Kind::LS:  return ParamPlotKind::LS;
    case ParametricPlotWindow::Kind::Bifurcation:
    default:                              return ParamPlotKind::Bifurcation;
    }
}


// One-time Plot2DView setup per kind — called by draw_parametric_plot_windows
// when it lazily constructs a fresh Plot2DView for a newly opened window
// (Plot2DView is now owned per-window there, not a function-shared by
// every window of a kind, so this replaces the old `if (!view) {...}` inits).
void configure_plot_view(Plot2DView& view, ParametricPlotWindow::Kind kind) {
    configure_param_plot_view(view, to_param_plot_kind(kind));
}


// Plots one Parametric plot window of kind Bifurcation. `win` supplies
// mode_2d and the diagram indices to show (`members`); `renderer`/`view`/
// `heatmap_map` are this window's own render-state instances, owned and
// cached by draw_parametric_plot_windows (keyed by win.id).
void draw_bifurcation_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
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
            hb.popup_extras = [&bdact, &cb, bd_busy]() {
                draw_export_menu_item(bd_busy, cb, [&bdact](const std::string& p) {
                    data_export::export_bif2d(bdact.result_2d, p);
                });
            };

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
            hc.popup_extras = [&bdact, &cb, c1d_busy]() {
                draw_export_menu_item(c1d_busy, cb, [&bdact](const std::string& p) {
                    data_export::export_bif1d(bdact.result, p);
                });
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
    // continuation сохраняет реальные lo/hi в result, иначе — из текстов.
    configure_sweep_x_axis_from(view, win.members, s.diagrams, s.params, s.vars,
        [](const BifurcationDiagramConfig& bd, double& lo, double& hi) {
            const bool from_result = bd.continuation && bd.result.param_hi != bd.result.param_lo;
            lo = from_result ? bd.result.param_lo : parse_ratio_or(bd.param_lo_text, 0.0);
            hi = from_result ? bd.result.param_hi : parse_ratio_or(bd.param_hi_text, 1.0);
        });
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

    std::vector<PlotSeriesInput> series_in;
    std::vector<bool> init_vis;
    std::vector<bool> glob_vis;
    series_in.reserve(win.members.size());
    init_vis.reserve(win.members.size());
    glob_vis.reserve(win.members.size());

    bool any_fit = false;
    int  data_gen = 0;

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
            // Continuation result хранит param_lo/hi-снапшот. Classical путь
            // его не заполняет — fallback на текущие текстовые поля.
            double lo = (bd.continuation && bd.result.param_hi != bd.result.param_lo)
                        ? bd.result.param_lo : parse_ratio_or(bd.param_lo_text, 0.0);
            double hi = (bd.continuation && bd.result.param_hi != bd.result.param_lo)
                        ? bd.result.param_hi : parse_ratio_or(bd.param_hi_text, 1.0);
            bool rev = bd.result.continuation_reverse;
            int npts = bd.result.n_pts;
            for (int k = 0; k < npts; ++k) {
                if (k < (int)bd.result.flags.size() &&
                    !regime_is_oscillation(bd.result.flags[k])) continue;
                double x = sweep_value_at(k, npts, lo, hi, bd.log_scale, rev, bd.continuation);
                if (k >= (int)source.size()) continue;
                for (double y : source[k]) {
                    buf.push_back((float)x);
                    buf.push_back((float)y);
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
    view.popup_extras = [&s, &cb]() {
        draw_export_submenu("bd", (int)s.diagrams.size(),
            [&s](int i) { return s.diagrams[i].label; },
            [&s](int i) { return s.diagrams[i].last_run_ok; },
            [&s](int i) { return s.in_flight && i == s.running_diagram_index; },
            [&s](int i, const std::string& p) { data_export::export_bif1d(s.diagrams[i].result, p); },
            cb);
    };

    // Snap X к узлам первой БД этого окна (см. apply_snap_x_from_first_member).
    apply_snap_x_from_first_member(view, win.members, s.diagrams);

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
}


// LLE: контролы (per-curve в табе) + line-plot λ(param)

// Контролы одной LLE-кривой. Возвращает true, если пользователь нажал Run
// для этой кривой.
void draw_lle_curve_controls(LLEAnalysisSession& s, int idx) {
    LLECurveConfig& c = s.curves[idx];

    ImGui::SetNextItemWidth(kComboW);
    if (InputTextStr("Label", c.label))
        c.label_is_manual = !c.label.empty();   // empty → back to auto
    ImGui::Separator();

    draw_scheme_combo("Scheme", c.scheme, s.custom_schemes, {}, &s.enabled_builtin_schemes);
    draw_adaptive_controls(c.adaptive);
    ImGui::Separator();

    // Sweep target: параметры + разделитель + переменные (IC) + dt (h). См. BD.
    draw_sweep_target_combo("Sweep", s.params, s.vars,
                            c.param_index, c.sweep_over_var, c.var_sweep_index,
                            c.sweep_over_h,
                            c.mode_2d ? &c.sweep_over_h_2 : nullptr,
                            /*note_when_empty*/ true, kComboW,
                            scheme_uses_symmetry(c.scheme, s.custom_schemes));
    InputNumStr(c.sweep_over_h ? "h lo" : "Param lo", c.param_lo_text, kFieldW);
    InputNumStr(c.sweep_over_h ? "h hi" : "Param hi", c.param_hi_text, kFieldW);
    ImGui::Checkbox("Log scale##lle_log", &c.log_scale);
    if (c.log_scale) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
    InputNumStr("Resolution", c.n_pts_text, kFieldW);

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
                                    scheme_uses_symmetry(c.scheme, s.custom_schemes));
        InputNumStr(c.sweep_over_h_2 ? "h2 lo" : "Param2 lo", c.param_lo_2_text, kFieldW);
        InputNumStr(c.sweep_over_h_2 ? "h2 hi" : "Param2 hi", c.param_hi_2_text, kFieldW);
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
        f.t_max      = &c.t_max_text;
        f.transient  = &c.transient_text;
        f.max_value  = &c.max_value_text;   // decimator'а у LLE нет
        draw_integration_block("Integration##lle_int", c.scheme, s.custom_schemes, f);
    }

    // LLE (Wolf/Benettin) (collapsible)
    if (ImGui::CollapsingHeader("LLE (Wolf/Benettin)##lle_wb", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("eps", c.eps_text, kFieldW);
        InputNumStr("NT",  c.nt_text, kFieldW);
        ImGui::TextDisabled("eps = initial perturbation magnitude; NT = block length\n"
                            "between renormalizations (in time units).");
    }

    draw_named_num_fields("Initial conditions##lle_ic", s.vars,   c.initial_conditions);
    draw_named_num_fields("Parameters##lle_par",        s.params, c.param_values);
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
            draw_regime_summary(c.result_2d.flags);
        }
        else if (!c.last_error.empty()) {
            draw_error_box("##lle_err", c.last_error);
        }
    } else {
        if (c.last_run_ok) {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: n_pts=%d, lambda-curve computed", c.result.n_pts);
            draw_regime_summary(c.result.flags);
        }
        else if (!c.last_error.empty()) {
            draw_error_box("##lle_err", c.last_error);
        }
    }
}
}
