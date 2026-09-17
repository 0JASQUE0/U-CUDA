#include "gui_internal.h"

namespace ucuda::ui_detail {


[[nodiscard]] std::vector<ParamPlotMatchItem>
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
    } else {
        auto& cs = model.ls_session.curves;
        for (int i = 0; i < (int)cs.size(); ++i)
            if (cs[i].mode_2d == mode_2d) out.push_back({ i, cs[i].label });
    }
    return out;
}


int parametric_type_index_of(ParametricPlotWindow::Kind kind, bool mode_2d) {
    int base = kind == ParametricPlotWindow::Kind::Bifurcation ? 0
             : kind == ParametricPlotWindow::Kind::LLE ? 2 : 4;
    return base + (mode_2d ? 1 : 0);
}


void parametric_type_from_index(int t, ParametricPlotWindow::Kind& kind, bool& mode_2d) {
    kind    = (t < 2) ? ParametricPlotWindow::Kind::Bifurcation
            : (t < 4) ? ParametricPlotWindow::Kind::LLE : ParametricPlotWindow::Kind::LS;
    mode_2d = (t % 2) == 1;
}


// Draws the Label | Type | Members... | X row for a Parametric plot window.
// Returns true iff the user clicked the row's X — caller unmounts the window.
// Assumes caller has already pushed a PushID(win.id) on the ID stack.
bool draw_parametric_window_setup_row(AppModel& model, ParametricPlotWindow& win) {
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
    auto& renderers = ui_resource<std::map<int, std::unique_ptr<PlotRenderer>>, 2>();
    auto& views = ui_resource<std::map<int, std::unique_ptr<Plot2DView>>, 3>();
    auto& heatmaps = ui_resource<std::map<int, std::map<int, std::unique_ptr<HeatmapView>>>, 4>();
    // Tracks the kind the Plot2DView was last configured for; when a
    // window's kind changes (Type combo), re-run configure_plot_view so
    // e.g. LLE doesn't inherit Bifurcation's points_mode=true.
    static std::map<int, ParametricPlotWindow::Kind> view_configured_for;

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


// 1D DFT: controls (system picker + Run/Run all + config tab bar, mirrors
// draw_basins_controls) + dynamic Plot windows (mirrors draw_parametric_
// controls' manager + draw_bifurcation_plot's colored_1d heatmap toolbar).

void draw_dft1d_diagram_controls(Dft1DAnalysisSession& s, int idx) {
    Dft1DConfig& c = s.configs[idx];

    ImGui::SetNextItemWidth(kComboW);
    if (InputTextStr("Label", c.label))
        c.label_is_manual = !c.label.empty();   // empty → back to auto
    ImGui::Separator();

    // Scheme (built-in + custom)
    draw_scheme_combo("Scheme", c.scheme, s.custom_schemes, {}, &s.enabled_builtin_schemes);
    draw_adaptive_controls(c.adaptive);
    ImGui::Separator();

    // Sweep target (parameter ИЛИ initial condition), см. draw_diagram_controls
    // dt (h) — как у Bif/LLE/LS 1D: число сэмплов блока и окно пересчитываются
    // под шаг каждой точки. Второй оси у DFT нет, поэтому other_over_h = nullptr.
    draw_sweep_target_combo("Sweep", s.params, s.vars,
                            c.param_index, c.sweep_over_var, c.var_sweep_index,
                            c.sweep_over_h, nullptr,
                            /*note_when_empty*/ true, kComboW,
                            scheme_uses_symmetry(c.scheme, s.custom_schemes));
    InputNumStr(c.sweep_over_h ? "h lo" : "Param lo", c.param_lo_text, kFieldW);
    InputNumStr(c.sweep_over_h ? "h hi" : "Param hi", c.param_hi_text, kFieldW);
    ImGui::Checkbox("Log scale##dft_log", &c.log_scale);
    if (c.log_scale) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }

    // Тот же блок Continuation + GPU/CPU, что у Bif / LLE / LS 1D.
    // 2D-режима у DFT нет, поэтому блокирует только IC-свип.
    draw_continuation_device_block(c, "dft", c.sweep_over_var, /*device_locked*/ false);
    ImGui::Separator();

    // Variable + Resolution X
    draw_writable_var_combo(s.vars, c.writable_var, "Writable var##dft_wv");
    InputNumStr("Resolution X", c.n_pts_text, kFieldW);
    ImGui::Separator();

    // ----- Resolution Y / Frequency range (обязательные поля для rangesFreq;
    // без auto-режима — частотный диапазон всегда физически осмыслен только
    // когда задан явно, в отличие от colored_1d's Y auto-range). -----
    if (ImGui::CollapsingHeader("Resolution Y##dft_freq", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("Resolution Y", c.n_freq_text, kFieldW);
        InputNumStr("Freq lo", c.freq_lo_text, kFieldW);
        InputNumStr("Freq hi", c.freq_hi_text, kFieldW);
        ImGui::Checkbox("Log scale (Y)##dft_freq_log", &c.freq_log_scale);
        if (c.freq_log_scale) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
        // Порядок = window_type (см. cpu_build_window). Дописывать только в
        // конец: значение уходит числом в JSON сессии.
        static const char* windows[] = { "None", "Hanning", "Hamming", "Blackman", "Blackman-Harris" };
        ImGui::SetNextItemWidth(kComboW);
        ImGui::Combo("Window", &c.window_type, windows, IM_ARRAYSIZE(windows));
    }

    // Display mode + normalize
    if (ImGui::CollapsingHeader("Display##dft_disp", ImGuiTreeNodeFlags_DefaultOpen)) {
        static const char* modes[] = { "Power spectrum", "Amplitude", "Phase" };
        ImGui::SetNextItemWidth(kComboW);
        ImGui::Combo("Mode", &c.display_mode, modes, IM_ARRAYSIZE(modes));
        ImGui::Checkbox("Normalize?", &c.normalize);
        // Шкала и её порог осмысленны только для power/amplitude — на phase (радианы) гасим оба.
        ImGui::BeginDisabled(c.display_mode == 2);
        ImGui::Checkbox("dB scale", &c.db_scale);
        if (c.display_mode != 2) {
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("В дБ power и amplitude совпадают тождественно:\n"
                                  "20*log10(sqrt(P)) = 10*log10(P).\n"
                                  "Различаются они только в линейной шкале —\n"
                                  "мощность давит слабые пики квадратично.");
        }
        ImGui::BeginDisabled(!c.db_scale);
        InputNumStr("dB floor", c.db_floor_text, kFieldW);
        ImGui::EndDisabled();
        if (c.display_mode != 2 && c.db_scale) {
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Нижняя граница логарифмической шкалы.\n"
                                  "Всё, что слабее, прижимается к этому значению —\n"
                                  "иначе нулевая ячейка дала бы -inf.");
        }
        ImGui::EndDisabled();
    }

    // Integration (collapsible)
    {
        IntegrationFields f;
        f.h           = &c.h_text;
        f.symmetry_s  = &c.symmetry_s;
        f.t_max       = &c.t_max_text;
        f.transient   = &c.transient_text;
        f.pre_scaller = &c.pre_scaller_text;
        f.max_value   = &c.max_value_text;
        draw_integration_block("Integration##dft_int", c.scheme, s.custom_schemes, f);
    }

    draw_named_num_fields("Initial conditions##dft_ic", s.vars,   c.initial_conditions);
    draw_named_num_fields("Parameters##dft_par",        s.params, c.param_values);
    draw_csv_output_block("CSV output##dft_csv", "Save to file", c.csv_save_enabled,
                          "##dft_csv_path", c.csv_output_path,
                          "Writes <path>_config.csv, _AkCOS.csv, _BkSIN.csv.");

    // Run-кнопка живёт в draw_dft1d_controls, как у Bifurcation/Basins.
    if (c.last_run_ok) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
            "OK: n_pts=%d, n_freq=%d", c.result.n_pts, c.result.n_freq);
        draw_regime_summary(c.result.flags);
    } else if (!c.last_error.empty()) {
        draw_error_box("##dft_err", c.last_error);
    }
}
}
