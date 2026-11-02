#include "gui_common.h"
#include "gui_inputs.h"
#include "gui_integration.h"
#include "gui_fields.h"
#include "gui_plot.h"
#include "tab_dft.h"
using namespace ucuda::ui_detail;

namespace ucuda::ui_detail {
void draw_dft1d_controls(AppModel& model, SystemLibrary& lib) {
    Dft1DAnalysisSession& s = model.dft1d_session;

    ImGui::Text("1D DFT");
    ImGui::TextDisabled("Parametric discrete Fourier transform via NVRTC + DFT_custom.");

    // Run / Run all... (общая реализация, см. draw_run_and_run_all)
    {
        const bool no_cfg = s.configs.empty();
        RunAllGroup g;
        g.pick_id = "pdft1d_";
        g.n       = (int)s.configs.size();
        g.label   = [&s](int i) { return s.configs[i].label; };
        g.enqueue = [&model](int i) { model.enqueue_dft1d(i); };
        draw_run_and_run_all("##run_all_dft1d", s.in_flight, no_cfg,
                             s.in_flight || no_cfg, 160.0f,
                             [&model, &s]() {
                                 model.run_dft1d(s.active_config_index);
                             },
                             {}, { g },
                             [&model]() { model.start_next_in_dft1d_queue(); },
                             model.dft1d_queue().size());
    }
    ImGui::Separator();

    // Tab bar: одна вкладка на config + "+" для add. Тело вкладки пустое —
    // настройки активного конфига рисуются ниже, после таб-бара.
    const TabBarResult tabs = draw_config_tab_bar(
        "##dft1d_tabs", "dft1d_tab_", (int)s.configs.size(),
        s.in_flight, s.running_config_index,
        [&s](int i) { return s.configs[i].label; },
        {},
        [&s]() { s.add_config(); });
    if (tabs.active    >= 0) s.active_config_index = tabs.active;
    if (tabs.to_remove >= 0) model.remove_dft1d_config(tabs.to_remove);

    if (s.configs.empty()) {
        ImGui::TextDisabled("No DFT configs. Press '+' to add one.");
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
        s.active_config_index = 0;

    draw_dft1d_diagram_controls(model, s, s.active_config_index);

    // Plot windows: dynamic list, mirrors Parametric's manager section
    // No Type combo needed (DFT1D has only one display kind); "Members..."
    // is single-select (radio) — each window shows exactly one config's
    // heatmap, same as Bifurcation's colored_1d/mode_2d windows.
    ImGui::Separator();
    ImGui::SeparatorText("Plot windows");
    int win_to_remove = -1;
    for (int i = 0; i < (int)model.dft1d_plot_windows.size(); ++i) {
        Dft1DPlotWindow& win = model.dft1d_plot_windows[i];
        ImGui::PushID(win.id);
        ImGui::SetNextItemWidth(220);
        if (InputTextStr("##wlabel", win.label)) {
            win.label_is_manual = !win.label.empty();   // empty → back to auto
            model.dft1d_plot_windows_dirty = true;
        }
        ImGui::SameLine();

        if (ImGui::Button("Members...")) ImGui::OpenPopup("edit_dft1d_window_members");
        if (ImGui::BeginPopup("edit_dft1d_window_members")) {
            if (s.configs.empty()) {
                ImGui::TextDisabled("(none available)");
            } else {
                // Single-select (radio) — a DFT1D window shows exactly one
                // config's heatmap, same as Bifurcation's mode_2d/colored_1d.
                for (int ci = 0; ci < (int)s.configs.size(); ++ci) {
                    bool sel = !win.members.empty() && win.members[0] == ci;
                    std::string lbl = s.configs[ci].label + "##dftmem" + std::to_string(ci);
                    if (ImGui::RadioButton(lbl.c_str(), sel)) {
                        win.members.assign(1, ci);
                        model.dft1d_plot_windows_dirty = true;
                    }
                }
            }
            ImGui::EndPopup();
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("X")) win_to_remove = i;
        ImGui::PopID();
    }
    if (win_to_remove >= 0) model.remove_dft1d_plot_window(win_to_remove);

    if (ImGui::Button("Add window")) {
        model.add_dft1d_plot_window({});
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset windows layout")) { model.dft1d_layout_generation++; }
}
}

namespace ucuda::ui_detail {
void draw_dft1d_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                           Dft1DPlotWindow& win,
                           PlotRenderer& renderer,
                           std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) {
    Dft1DAnalysisSession& s = model.dft1d_session;
    auto get_hm = [&](int idx) -> HeatmapView& {
        const int cfg_cm = (idx >= 0 && idx < (int)s.configs.size())
                           ? s.configs[idx].colormap_idx : -1;
        return get_or_create_heatmap(heatmap_map, idx, cfg_cm, model.heatmap_colormap);
    };

    if (win.members.empty()) {
        ImGui::TextDisabled("No DFT configs assigned to this window.");
        return;
    }

    for (size_t mi = 0; mi < win.members.size(); ++mi) {
        int idx = win.members[mi];
        if (idx < 0 || idx >= (int)s.configs.size()) continue;
        Dft1DConfig& c = s.configs[idx];
        if (mi > 0) ImGui::Separator();
        ImGui::PushID(idx);

        const unsigned oid = 0xD1FD0000u + (unsigned)idx;
        HeatmapView& hc = get_hm(idx);

        {
            HeatmapToolbarOpts topts;
            topts.persist_colormap = [&](int cm) {
                c.colormap_idx = cm;   // persist per-config
                if (!model.loaded_name.empty())
                    lib.save_session(model.loaded_name, "_last_dft1d",
                                     session_to_json_dft1d(model.dft1d_session));
            };
            draw_heatmap_toolbar(hc, topts);
        }

        if (!c.last_run_ok || c.result.ak_cos.empty()) {
            ImGui::TextDisabled("No data yet. Press Run.");
            ImGui::PopID();
            continue;
        }

        int npts  = c.result.n_pts;
        int nfreq = c.result.n_freq;

        // Лениво (пере)строим display_cache, когда расходится с текущими
        // настройками — не каждый кадр. Та же staleness-схема, что и у
        // colored_1d_cache в draw_bifurcation_plot.
        const double db_floor = parse_num(c.db_floor_text, -120.0);
        bool stale = c.display_built_from    != c.data_generation
                  || c.display_cache_mode      != c.display_mode
                  || c.display_cache_normalize != c.normalize
                  || c.display_cache_db_floor  != db_floor
                  || c.display_cache_db_scale  != c.db_scale;
        if (stale) {
            size_t plane_size = (size_t)nfreq * (size_t)npts;
            // freq-major/param-minor (idx = f*n_pts+pt) — конвенция
            // HeatmapView::render(nx=n_pts, ny=n_freq, ...). 999.0 — тот же
            // sentinel, что и colored_1d/Basins используют для "нет данных".
            c.display_cache.assign(plane_size, kSentinelNoData);
            double vmin =  std::numeric_limits<double>::infinity();
            double vmax = -std::numeric_limits<double>::infinity();
            std::vector<double> col((size_t)nfreq);
            for (int pt = 0; pt < npts; ++pt) {
                bool diverged = pt >= (int)c.result.flags.size() ||
                                !regime_is_oscillation(c.result.flags[pt]);
                if (diverged) continue;   // остаётся 999 sentinel — DFT_custom не считал эту точку

                double colmax = 0.0;
                for (int f = 0; f < nfreq; ++f) {
                    double ak = c.result.ak_cos[(size_t)pt * (size_t)nfreq + (size_t)f];
                    double bk = c.result.bk_sin[(size_t)pt * (size_t)nfreq + (size_t)f];
                    double v;
                    if      (c.display_mode == 0) v = ak * ak + bk * bk;            // power spectrum
                    else if (c.display_mode == 1) v = std::sqrt(ak * ak + bk * bk); // amplitude
                    else                          v = std::atan2(bk, ak);          // phase
                    col[(size_t)f] = v;
                    double av = std::fabs(v);
                    if (av > colmax) colmax = av;
                }
                // dB-масштаб только для power/amplitude и только если включён; phase всегда
                // остаётся в радианах. Коэффициент разный: power — величина мощности (10*log10,
                // как в референсном MATLAB-скрипте), amplitude — sqrt от неё, то есть 20*log10.
                // С одним коэффициентом на оба режима amplitude показывал ровно половину
                // децибел от power для тех же данных.
                const bool   to_db = c.db_scale && c.display_mode != 2;
                const double db_k  = (c.display_mode == 1) ? 20.0 : 10.0;
                // Порог задаём в линейной шкале, чтобы log10(0) вообще не считался:
                // 10^(floor/k) — ровно то значение, которое даст floor после лога.
                const double lin_floor = std::pow(10.0, db_floor / db_k);
                for (int f = 0; f < nfreq; ++f) {
                    double v = col[(size_t)f];
                    if (c.normalize && colmax > 0.0) v /= colmax;
                    if (to_db) v = db_k * std::log10(std::max(v, lin_floor));
                    c.display_cache[(size_t)f * (size_t)npts + (size_t)pt] = v;
                    if (v < vmin) vmin = v;
                    if (v > vmax) vmax = v;
                }
            }
            c.display_cache_vmin = std::isfinite(vmin) ? vmin : 0.0;
            c.display_cache_vmax = std::isfinite(vmax) ? vmax : 1.0;
            c.display_built_from      = c.data_generation;
            c.display_cache_mode      = c.display_mode;
            c.display_cache_normalize = c.normalize;
            c.display_cache_db_floor  = db_floor;
            c.display_cache_db_scale  = c.db_scale;
            ++c.display_cache_gen;
        }

        hc.x_axis.name = auto_axis_name(s.params, s.vars, c.param_index,
                                        c.sweep_over_var, c.var_sweep_index, c.sweep_over_h);
        hc.x_axis.log_scale = c.log_scale;
        hc.y_axis.name = "Frequency";
        hc.y_axis.log_scale = c.freq_log_scale;
        // Неколебательные точки (result.flags[pt] <= 0) уходят в кэш как sentinel и рисуются
        // чёрным, а не тёмно-серым по умолчанию: на шкале в дБ они иначе читаются как «сигнал
        // на уровне пола», хотя спектра там не считали вовсе.
        hc.nodata_color[0] = hc.nodata_color[1] = hc.nodata_color[2] = 0.0f;

        bool fit = c.fit_request;
        if (fit) c.fit_request = false;

        const bool busy = s.in_flight && idx == s.running_config_index;
        // Ctrl+T: X — свип параметра, Y — окно частот, которое DFT реально
        // считает (freq_lo/hi уходят в запрос, а не режут готовую картинку).
        const std::vector<ViewRangeTarget> vrt = { vr_target(s.configs, idx,
            &Dft1DConfig::param_lo_text, &Dft1DConfig::param_hi_text,
            &Dft1DConfig::freq_lo_text,  &Dft1DConfig::freq_hi_text) };
        hc.popup_extras = [&c, &cb, busy, &vrt, &model, &hc]() {
            draw_export_menu_item(busy, cb, [&c](const std::string& p) {
                data_export::export_dft1d(c.result, p);
            });
            draw_view_range_menu(vrt, hc.x_axis, hc.y_axis, hc.swap_axes, &model);
        };

        double lo = c.result.param_lo, hi = c.result.param_hi;
        bool rev = c.result.continuation_reverse;
        double x0 = rev ? hi : lo;
        double x1 = rev ? lo : hi;

        hc.cell_regime = regime_cell_label(c.result.flags, 0);   // флаг на точку свипа (колонку)
        hc.cell_not_computed = cancelled_cell_mask(c.result.flags, 0);
        ImVec2 avail = ImGui::GetContentRegionAvail();
        ImVec2 origin = ImGui::GetCursorScreenPos();
        hc.render(renderer, origin, avail,
                  /*owner_id*/ oid, c.display_cache_gen,
                  npts, nfreq,
                  c.display_cache.data(),
                  x0, x1,
                  c.result.freq_lo, c.result.freq_hi,
                  c.display_cache_vmin, c.display_cache_vmax,
                  fit);
        handle_view_range_keys(vrt, hc.x_axis, hc.y_axis, plot_window_active(),
                               hc.swap_axes, &model);
        ImGui::PopID();
    }
}
}

namespace ucuda::ui_detail {
void draw_dft1d_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) {
    auto& renderers = ui_resource<std::map<int, std::unique_ptr<PlotRenderer>>, 3006>();
    auto& heatmaps = ui_resource<std::map<int, std::map<int, std::unique_ptr<HeatmapView>>>, 3007>();
    static int seen_epoch = 0;
    if (session_reloaded(model, AppModel::AppMode::Dft1D, seen_epoch))
        for (auto& kv : heatmaps) invalidate_views(kv.second);

    int to_remove = -1;
    for (int i = 0; i < (int)model.dft1d_plot_windows.size(); ++i) {
        Dft1DPlotWindow& win = model.dft1d_plot_windows[i];

        auto& renderer = renderers[win.id];
        if (!renderer) renderer = std::make_unique<PlotRenderer>();
        auto& hm_map = heatmaps[win.id];

        std::string title = win.label + "###dftwin" + std::to_string(win.id)
                           + "_g" + std::to_string(model.dft1d_layout_generation);
        bool open = true;
        float ox = 60.0f + (float)(i % 5) * 35.0f, oy = 80.0f + (float)(i % 5) * 35.0f;
        ImGui::SetNextWindowPos(ImVec2(ox, oy), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(700, 550), ImGuiCond_FirstUseEver);
        if (ImGui::Begin(title.c_str(), &open)) {
            ImGui::PushID(win.id);
            draw_dft1d_plot(model, lib, cb, win, *renderer, hm_map);
            ImGui::PopID();
        }
        ImGui::End();
        if (!open) to_remove = i;
    }
    if (to_remove >= 0) {
        int id = model.dft1d_plot_windows[to_remove].id;
        renderers.erase(id);
        heatmaps.erase(id);
        model.remove_dft1d_plot_window(to_remove);
    }
}
}


namespace ucuda::ui_detail {
void draw_dft1d_diagram_controls(AppModel& model, Dft1DAnalysisSession& s, int idx) {
    Dft1DConfig& c = s.configs[idx];
    // Чей конфиг правит эта панель — для ПКМ-рассылки "во все такие же".
    model.broadcast_source_tab = BroadcastTab::Dft1D;
    model.broadcast_source_idx = idx;

    ImGui::SetNextItemWidth(kComboW);
    if (InputTextStr("Label", c.label))
        c.label_is_manual = !c.label.empty();   // empty → back to auto
    ImGui::Separator();

    // Scheme (built-in + custom)
    draw_scheme_combo("Scheme", c.scheme, s.custom_schemes, {}, &s.enabled_builtin_schemes,
                      &model, model.is_map, &s.wrapper_schemes);
    ImGui::Separator();

    // Sweep target (parameter ИЛИ initial condition), см. draw_diagram_controls
    // dt (h) — как у Bif/LLE/LS 1D: число сэмплов блока и окно пересчитываются
    // под шаг каждой точки. Второй оси у DFT нет, поэтому other_over_h = nullptr.
    draw_sweep_target_combo("Sweep", s.params, s.vars,
                            c.param_index, c.sweep_over_var, c.var_sweep_index,
                            c.sweep_over_h, nullptr,
                            /*note_when_empty*/ true, kComboW,
                            scheme_uses_symmetry(c.scheme, s.custom_schemes),
                            &model, BroadcastField::SweepTarget, !model.is_map);
    InputNumStr(c.sweep_over_h ? "h lo" : "Param lo", c.param_lo_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepLo, {},
                                        c.sweep_over_h ? "h lo" : "Param lo", c.param_lo_text); });
    InputNumStr(c.sweep_over_h ? "h hi" : "Param hi", c.param_hi_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepHi, {},
                                        c.sweep_over_h ? "h hi" : "Param hi", c.param_hi_text); });
    ImGui::Checkbox("Log scale##dft_log", &c.log_scale);
    if (c.log_scale) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }

    // Тот же блок Continuation + GPU/CPU, что у Bif / LLE / LS 1D.
    // 2D-режима у DFT нет, поэтому блокирует только IC-свип.
    draw_continuation_device_block(c, "dft", c.sweep_over_var, /*device_locked*/ false);
    ImGui::Separator();

    // Variable + Resolution X
    draw_writable_var_combo(s.vars, c.writable_var, "Writable var##dft_wv");
    InputNumStr("Resolution X", c.n_pts_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::Resolution, {},
                                        "Resolution X", c.n_pts_text); });
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
                ImGui::SetTooltip("In dB power and amplitude are identically equal:\n"
                                  "20*log10(sqrt(P)) = 10*log10(P).\n"
                                  "They differ only on a linear scale -\n"
                                  "power suppresses weak peaks quadratically.");
        }
        ImGui::BeginDisabled(!c.db_scale);
        InputNumStr("dB floor", c.db_floor_text, kFieldW);
        ImGui::EndDisabled();
        if (c.display_mode != 2 && c.db_scale) {
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Lower bound of the logarithmic scale.\n"
                                  "Everything weaker is pinned to this value -\n"
                                  "otherwise a zero bin would give -inf.");
        }
        ImGui::EndDisabled();
    }

    // Integration (collapsible)
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
        draw_integration_block("Integration##dft_int", c.scheme, s.custom_schemes, f, &model, model.is_map);
    }

    draw_named_num_fields("Initial conditions##dft_ic", s.vars,   c.initial_conditions,
                          nullptr, &model, BroadcastField::InitCondition);
    draw_named_num_fields("Parameters##dft_par",        s.params, c.param_values,
                          nullptr, &model, BroadcastField::Param);
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
