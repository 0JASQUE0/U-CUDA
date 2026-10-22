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
        [&model, &s](int i) { draw_lle_curve_controls(model, s, i); },
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

    // Подпись X + X-fit диапазон — см. configure_sweep_x_axis_from.
    configure_sweep_x_axis_from(view, win.members, s.curves, s.params, s.vars,
        [](const LLECurveConfig& c, double& lo, double& hi) {
            sweep_range_from_result(c.result.param_lo, c.result.param_hi,
                                    c.param_lo_text, c.param_hi_text, lo, hi);
        });
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

void draw_ls_curve_controls(AppModel& model, LyapunovSpectrumAnalysisSession& s, int idx) {
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
            draw_regime_summary(c.result_2d.flags);
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


void draw_ls_controls(AppModel& model, SystemLibrary& /*lib*/) {
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
const char* signal_metric_tooltip(int m) {
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


void draw_metrics_config_controls(AppModel& model, SignalMetricsAnalysisSession& s, int idx) {
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
        draw_regime_summary(r.flags);
        draw_adaptive_stats(r.ad_stats);
    }
    else if (!c.last_error.empty()) {
        draw_error_box("##met_err", c.last_error);
    }
}


void draw_metrics_controls(AppModel& model, SystemLibrary& /*lib*/) {
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
void draw_metric_tab_bar(AppModel& model, ParametricPlotWindow& win, int avail_mask) {
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


void draw_metrics_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
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

    configure_sweep_x_axis_from(view, win.members, s.configs, s.params, s.vars,
        [](const SignalMetricsConfig& c, double& lo, double& hi) {
            sweep_range_from_result(c.result.param_lo, c.result.param_hi,
                                    c.param_lo_text, c.param_hi_text, lo, hi);
        });
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
}
