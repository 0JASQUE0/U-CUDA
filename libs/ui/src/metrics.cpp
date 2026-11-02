#include "gui_common.h"
#include "gui_inputs.h"
#include "gui_integration.h"
#include "gui_fields.h"
#include "gui_plot.h"
#include "tab_parametric.h"
// Parametric -> Metrics: настройки, вкладки метрик, графики.
using namespace ucuda::ui_detail;


namespace ucuda::ui_detail {
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
}


namespace ucuda::ui_detail {
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
}


namespace ucuda::ui_detail {
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
}


namespace ucuda::ui_detail {
void draw_metric_tab_bar(AppModel& model, ParametricPlotWindow& win, int avail_mask) {
    auto& shown = ui_resource<std::map<int, int>, 3014>();   // win.id -> метрика, которую таб-бар показывал
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
}


namespace ucuda::ui_detail {
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
                    write_signal_metrics_csv(cact.result_2d, p, e);
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
        auto& last_metric = ui_resource<std::map<int, int>, 3015>();
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
                write_signal_metrics_csv(s.configs[i].result, p, e);
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
