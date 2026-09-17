#include "gui_internal.h"

namespace ucuda::ui_detail {


// Tab bar по LLE-кривым + кнопка «+» (копирует последнюю).
void draw_lle_controls(AppModel& model, SystemLibrary& /*lib*/) {
    LLEAnalysisSession& s = model.lle_session;

    // Run + Ctrl+R — в draw_parametric_controls (общая кнопка слева от Run all).
    const TabBarResult tabs = draw_config_tab_bar(
        "##lle_tabs", "lle_tab_", (int)s.curves.size(),
        s.in_flight, s.running_curve_index,
        [&s](int i) { return s.curves[i].label; },
        [&s](int i) { draw_lle_curve_controls(s, i); },
        [&s]() { s.add_curve(); });
    if (tabs.active    >= 0) s.active_curve_index = tabs.active;
    if (tabs.to_remove >= 0) model.remove_lle_curve(tabs.to_remove);
}


// Plot LLE: линии (points_mode=false), каждая кривая — λ(param). При mode_2d=true у
// активной кривой вместо линий рисуется HeatmapView (сетка λ(p1, p2) с colormap'ом).
// Per-window-state shape — как у draw_bifurcation_plot (обоснование см. там).
void draw_lle_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
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
            heatmap.popup_extras = [&cact, &cb, busy]() {
                draw_export_menu_item(busy, cb, [&cact](const std::string& p) {
                    data_export::export_lle2d(cact.result_2d, p);
                });
            };

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

    // Подпись X + X-fit диапазон — см. configure_sweep_x_axis_from.
    configure_sweep_x_axis_from(view, win.members, s.curves, s.params, s.vars,
        [](const LLECurveConfig& c, double& lo, double& hi) {
            sweep_range_from_result(c.result.param_lo, c.result.param_hi,
                                    c.param_lo_text, c.param_hi_text, lo, hi);
        });
    view.y_axis.name = "lambda";

    auto& bufs = window_point_bufs(win.id, win.members.size());

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
            for (int k = 0; k < npts; ++k) {
                if (k < (int)c.result.flags.size() &&
                    !regime_is_oscillation(c.result.flags[k])) continue;
                double x = sweep_value_at(k, npts, lo, hi, c.log_scale, rev, c.continuation);
                double y = c.result.lyapunov[k];
                if (!std::isfinite(y)) continue;
                buf.push_back((float)x);
                buf.push_back((float)y);
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

    view.popup_extras = [&s, &cb]() {
        draw_export_submenu("lle", (int)s.curves.size(),
            [&s](int i) { return s.curves[i].label; },
            [&s](int i) { return s.curves[i].last_run_ok; },
            [&s](int i) { return s.in_flight && !s.is_2d_run && i == s.running_curve_index; },
            [&s](int i, const std::string& p) { data_export::export_lle1d(s.curves[i].result, p); },
            cb);
    };

    // Snap X к узлам первой кривой этого окна (см. apply_snap_x_from_first_member).
    apply_snap_x_from_first_member(view, win.members, s.curves);

    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    view.render(renderer, origin, avail, /*owner_id*/ 0xBE11E5, data_gen,
                series_in, init_vis, glob_vis, any_fit);
}


// LS: контролы (per-curve в табе) + line-plot λ_k(param), N экспонент
// на один спектр-«прогон». UX зеркало LLE.

void draw_ls_curve_controls(LyapunovSpectrumAnalysisSession& s, int idx) {
    LSCurveConfig& c = s.curves[idx];

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
    ImGui::Checkbox("Log scale##ls_log", &c.log_scale);
    if (c.log_scale) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
    InputNumStr("Resolution", c.n_pts_text, kFieldW);

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
                                    scheme_uses_symmetry(c.scheme, s.custom_schemes));
        InputNumStr(c.sweep_over_h_2 ? "h2 lo" : "Param2 lo", c.param_lo_2_text, kFieldW);
        InputNumStr(c.sweep_over_h_2 ? "h2 hi" : "Param2 hi", c.param_hi_2_text, kFieldW);
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
        f.t_max      = &c.t_max_text;
        f.transient  = &c.transient_text;
        f.max_value  = &c.max_value_text;   // decimator'а у LS нет
        draw_integration_block("Integration##ls_int", c.scheme, s.custom_schemes, f);
    }

    // LS (Wolf/Benettin + Gram-Schmidt) (collapsible)
    if (ImGui::CollapsingHeader("LS (Wolf/Benettin + Gram-Schmidt)##ls_wbgs", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("eps", c.eps_text, kFieldW);
        InputNumStr("NT",  c.nt_text, kFieldW);
        ImGui::TextDisabled("eps = initial perturbation magnitude; NT = block length\n"
                            "between renormalizations (in time units).");
    }

    draw_named_num_fields("Initial conditions##ls_ic", s.vars,   c.initial_conditions);
    draw_named_num_fields("Parameters##ls_par",        s.params, c.param_values);
    draw_csv_output_block("CSV output##ls_csv", "Save to file", c.csv_save_enabled,
                          "##ls_csv_path", c.csv_output_path,
                          "Path is kept even when save is off. Also writes <path>_config.csv.");

    // Run-кнопка живёт на уровне draw_parametric_controls.

    if (c.mode_2d) {
        if (c.last_run_2d_ok) {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: %dx%d heatmap, %d exponents",
                c.result_2d.n_pts, c.result_2d.n_pts, c.result_2d.n_exponents);
            draw_regime_summary(c.result_2d.flags);
        }
        else if (!c.last_error.empty()) {
            draw_error_box("##ls_err_2d", c.last_error);
        }
    } else {
        if (c.last_run_ok) {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: n_pts=%d, n_exponents=%d", c.result.n_pts, c.result.n_exponents);
            draw_regime_summary(c.result.flags);
        }
        else if (!c.last_error.empty()) {
            draw_error_box("##ls_err", c.last_error);
        }
    }
}


void draw_ls_controls(AppModel& model, SystemLibrary& /*lib*/) {
    LyapunovSpectrumAnalysisSession& s = model.ls_session;

    // Run + Ctrl+R — в draw_parametric_controls (общая кнопка слева от Run all).
    const TabBarResult tabs = draw_config_tab_bar(
        "##ls_tabs", "ls_tab_", (int)s.curves.size(),
        s.in_flight, s.running_curve_index,
        [&s](int i) { return s.curves[i].label; },
        [&s](int i) { draw_ls_curve_controls(s, i); },
        [&s]() { s.add_curve(); });
    if (tabs.active    >= 0) s.active_curve_index = tabs.active;
    if (tabs.to_remove >= 0) model.remove_ls_curve(tabs.to_remove);
}


// Plot LS: каждая кривая раскладывается на N линий по числу экспонент, серия =
// spectrum_idx * N + exponent_idx, цвета через ic_base_color(seq). При mode_2d=true у
// активной кривой рисуется HeatmapView с одной выбранной экспонентой; combo "Exponent"
// над хитмапой переключает плоскость без повторного Run.
// Per-window-state shape — как у draw_bifurcation_plot (обоснование см. там).
void draw_ls_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
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
            heatmap_ls.popup_extras = [&cact, &cb, busy]() {
                draw_export_menu_item(busy, cb, [&cact](const std::string& p) {
                    data_export::export_ls2d(cact.result_2d, p);
                });
            };

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

    // Подпись X + X-fit диапазон — см. configure_sweep_x_axis_from.
    configure_sweep_x_axis_from(view, win.members, s.curves, s.params, s.vars,
        [](const LSCurveConfig& c, double& lo, double& hi) {
            sweep_range_from_result(c.result.param_lo, c.result.param_hi,
                                    c.param_lo_text, c.param_hi_text, lo, hi);
        });

    // Подсчитываем общее число серий этого окна — sum(n_exponents per member).
    size_t total_series = 0;
    for (int idx : win.members) {
        if (idx < 0 || idx >= (int)s.curves.size()) continue;
        const auto& c = s.curves[idx];
        int N = c.result.n_exponents > 0 ? c.result.n_exponents : (int)s.vars.size();
        total_series += (size_t)N;
    }

    auto& bufs = window_point_bufs(win.id, total_series);

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
                    buf.push_back((float)x);
                    buf.push_back((float)y);
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

    view.popup_extras = [&s, &cb]() {
        draw_export_submenu("ls", (int)s.curves.size(),
            [&s](int i) { return s.curves[i].label; },
            [&s](int i) { return s.curves[i].last_run_ok; },
            [&s](int i) { return s.in_flight && !s.is_2d_run && i == s.running_curve_index; },
            [&s](int i, const std::string& p) { data_export::export_ls1d(s.curves[i].result, p); },
            cb);
    };

    // Snap X к узлам первой LS-кривой этого окна (см. apply_snap_x_from_first_member).
    apply_snap_x_from_first_member(view, win.members, s.curves);

    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    view.render(renderer, origin, avail, /*owner_id*/ 0x15A1E0, data_gen,
                series_in, init_vis, glob_vis, any_fit);
}
}
