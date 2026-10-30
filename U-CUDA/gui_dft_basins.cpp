#include "gui_internal.h"

// 1D DFT: controls (system picker + Run/Run all + config tab bar, mirrors
// draw_basins_controls) + dynamic Plot windows (mirrors draw_parametric_
// controls' manager + draw_bifurcation_plot's colored_1d heatmap toolbar).

static void draw_dft1d_diagram_controls(AppModel& model, Dft1DAnalysisSession& s, int idx) {
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
        g.enqueue = [&model](int i) { model.dft1d_queue.push_back({i}); };
        draw_run_and_run_all("##run_all_dft1d", s.in_flight, no_cfg,
                             s.in_flight || no_cfg, 160.0f,
                             [&model, &s]() {
                                 if (!model.parametric_engine)
                                     model.parametric_engine = std::make_unique<ParametricEngine>();
                                 s.run_async(*model.parametric_engine, s.active_config_index);
                             },
                             {}, { g },
                             [&model]() { model.start_next_in_dft1d_queue(); },
                             model.dft1d_queue.size());
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

// Per-window heatmap rendering — mirrors the colored_1d block of
// draw_bifurcation_plot (toolbar + lazy display-cache rebuild + render()) and
// draw_basins_plot's colormap/autoscale/swap-axes toolbar.
static void draw_dft1d_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
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

void draw_dft1d_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) {
    static std::map<int, std::unique_ptr<PlotRenderer>> renderers;
    static std::map<int, std::map<int, std::unique_ptr<HeatmapView>>> heatmaps;
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

// Basins of attraction: controls + 5-plot window (inner tab-bar).

// Basins: общие хелперы карты + панель фазовых портретов по бассейнам.

// Диапазон colorbar'а таба «Basins» — единственный источник истины и для хитмапы, и для цвета
// траекторий в фазовых портретах (иначе цвет бассейна на карте и его траектории разъезжаются):
//   min_cluster_id..-1 — FP-кластеры (если есть отрицательные)
//   0                  — расходящиеся ячейки (helpful_array[i] == 0)
//   1..n_clusters      — осциллирующие кластеры
// Когда FP-кластеров нет и ничего не разошлось, «кластера 0» не существует — сдвигаем vmin к 1,
// чтобы в colorbar'е не было фантомной полосы.
static void basins_colorbar_range(const BasinsConfig& c, double& vmin, double& vmax) {
    const int n_clusters     = c.renumber_spiral ? c.n_clusters_spiral      : c.result.n_clusters;
    const int min_cluster_id = c.renumber_spiral ? c.min_cluster_idx_spiral : c.result.min_cluster_idx;
    bool has_diverged = false;
    for (int f : c.result.helpful_array)
        if (regime_is_unbound(f)) { has_diverged = true; break; }
    if (min_cluster_id < 0)  vmin = (double)min_cluster_id;
    else if (has_diverged)   vmin = 0.0;
    else                     vmin = 1.0;
    vmax = (double)n_clusters;
    if (vmax < vmin) vmax = vmin;
}

// HeatmapView таба «Basins» каждого config'а. Раньше жил function-local static
// внутри draw_basins_plot; поднят на уровень файла, потому что цвет траекторий
// в окнах фазовых портретов обязан читать ЖИВЫЕ discrete/reverse/colormap
// именно этого вью — иначе цвета совпадали бы только при дефолтных настройках.
static std::map<unsigned, std::unique_ptr<HeatmapView>> g_hm_basins;

// owner_id блока плотов одного basins-config'а (5 внутренних табов на config).
[[nodiscard]] static unsigned basins_base_oid(int config_index) {
    return 0x1BA50000u + (unsigned)config_index * 5u;
}

// Цвет бассейна `display_id` — ровно тот, которым его красит хитмапа.
// Повторяет квантование из FS_HEATMAP (plot_renderer.cpp): t по [vmin,vmax],
// дискретизация по полосам с edge-aligned сэмплированием, затем reverse.
// vmin/vmax — из basins_colorbar_range: перегрузка для тех, кто красит много id за кадр
// (Scatter) и не хочет на каждый id заново сканировать helpful_array.
static ImVec4 basins_id_color(const BasinsConfig& c, const AppModel& model,
                              int config_index, int display_id,
                              double vmin, double vmax) {
    int  cm     = (c.colormap_idx[0] >= 0) ? c.colormap_idx[0] : model.basins_colormap;
    bool reverse = false;
    // discrete_default для этого вью = true, поэтому дефолт (пока панель ни
    // разу не рисовалась) — дискретный, по одной полосе на целый id.
    int  n_disc  = std::max(1, (int)std::lround(vmax - vmin) + 1);

    auto it = g_hm_basins.find(basins_base_oid(config_index) + 0u);
    if (it != g_hm_basins.end() && it->second) {
        const HeatmapView& hv = *it->second;
        cm      = (int)hv.colormap;
        reverse = hv.reverse_colormap;
        n_disc  = hv.discrete ? (hv.discrete_levels > 0 ? hv.discrete_levels : n_disc) : 0;
    }
    cm = colormap_id_or(cm, kColormapTurbo);

    // Нормировка — по РАЗДВИНУТОМУ диапазону, ровно как в HeatmapView::render (там vmax <= vmin
    // даёт vmax = vmin + 1), а число полос считается по исходному — тоже как там. Раньше здесь
    // вырожденный диапазон давал t = 0.5, а карта для того же id — t = 0, и единственный бассейн
    // получал на портрете другой цвет, чем на хитмапе.
    double vmax_n = vmax;
    if (vmax_n <= vmin) vmax_n = vmin + 1.0;
    const double range = vmax_n - vmin;
    float t = (float)(((double)display_id - vmin) / range);
    if (t < 0.0f) t = 0.0f; else if (t > 1.0f) t = 1.0f;
    if (n_disc > 0) {
        const float nb = (float)n_disc;
        float k = std::floor(t * nb);
        if (k >= nb) k = nb - 1.0f;
        t = (nb > 1.0f) ? (k / (nb - 1.0f)) : 0.5f;
    }
    if (reverse) t = 1.0f - t;
    return ImGui::ColorConvertU32ToFloat4(cmap_sample(t, (HeatmapColormap)cm));
}

static ImVec4 basins_id_color(const BasinsConfig& c, const AppModel& model,
                              int config_index, int display_id) {
    double vmin = 0.0, vmax = 0.0;
    basins_colorbar_range(c, vmin, vmax);
    return basins_id_color(c, model, config_index, display_id, vmin, vmax);
}

// Панель «Phase portraits» внутри Basins Controls. Интегратор и параметры —
// из самого basins-config'а; своё тут только время моделирования, прореживание,
// ограничитель числа аттракторов и выбор CPU/GPU.
static void draw_basins_phase_controls(AppModel& model, int cfg_idx) {
    BasinsAnalysisSession& s = model.basins_session;
    if (cfg_idx < 0 || cfg_idx >= (int)s.configs.size()) return;
    BasinsConfig&    c  = s.configs[(size_t)cfg_idx];
    BasinsPhaseSlot& sl = s.phase_slot(cfg_idx);

    ImGui::TextDisabled("One random grid cell per basin; scheme/h/params come from this config.");

    bool changed = false;

    // Run / Autorun / устройство
    const bool basins_busy = s.in_flight && s.running_config_index == cfg_idx;
    const bool phase_busy  = sl.phase.in_flight;
    const bool no_data     = !c.last_run_ok || c.result.basin_idx.empty();
    {
        const bool block = basins_busy || phase_busy || no_data;
        if (block) ImGui::BeginDisabled();
        if (ImGui::Button(phase_busy ? "Recomputing..." : "Run portraits", ImVec2(160, 0)))
            s.request_phase_run(cfg_idx);
        if (block) ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Checkbox("Autorun", &c.pp_autorun)) changed = true;
        ImGui::SameLine();
        if (ImGui::Checkbox("GPU##pp", &c.pp_use_gpu)) changed = true;
    }
    // Custom КРС на CPU требует внешнего компилятора — та же проверка, что и в
    // draw_phase_controls; без него молча возвращаем GPU.
    if (scheme_needs_compiled_body(c.scheme, s.custom_schemes) && !c.pp_use_gpu) {
        std::string why;
        if (!krs_cpu_backend_available(&why)) {
            ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1), "(custom scheme requires GPU: %s)", why.c_str());
            c.pp_use_gpu = true;
        }
    }

    // Собственные параметры интегрирования
    changed |= InputNumStr("computing time##pp", c.pp_t_max_text, kFieldW);
    changed |= InputNumStr("transient time##pp", c.pp_transient_text, kFieldW);
    changed |= InputNumStr("decimator##pp",      c.pp_prescaller_text, kFieldW);
    ImGui::TextDisabled("Decimator only thins the drawn points (applied after integration).");
    changed |= InputNumStr("max attractors##pp", c.pp_max_attractors_text, kFieldW);
    ImGui::SetNextItemWidth(200);
    if (ImGui::SliderFloat("FP marker size##pp", &c.pp_marker_size, 2.0f, 12.0f, "%.0f px"))
        changed = true;

    // Сводка: сколько бассейнов нашлось / сколько рисуем / сколько памяти
    if (no_data) {
        ImGui::TextDisabled("No basins data yet — run the map first.");
    } else {
        int n_basins = 0;
        const int  n   = c.result.n_pts;
        const int* ids = basins_display_ids(c);
        if (ids && n > 0) {
            std::unordered_set<int> seen;
            const size_t tot = (size_t)n * (size_t)n;
            for (size_t k = 0; k < tot; ++k) if (ids[k] != 0) seen.insert(ids[k]);
            n_basins = (int)seen.size();
        }
        const int drawn = (int)sl.cells.size();
        if (n_basins == 0) {
            ImGui::TextDisabled("No basins found (every cell diverged) — nothing to draw.");
        } else {
            ImGui::Text("%d basin(s) found, %d drawn", n_basins, drawn);
            int limit = parse_int_or(c.pp_max_attractors_text, 1);
            if (limit < 1) limit = 1;
            if (n_basins > limit) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f), "(limited to %d)", limit);
            }
            // Оценка объёма: движок держит траектории целиком (децимация
            // применяется уже после расчёта), поэтому память считаем по
            // полному числу шагов.
            const double h  = parse_ratio_or(c.h_text, 0.0);
            const double tt = parse_ratio_or(c.pp_t_max_text, 0.0);
            const int    kept = (drawn > 0) ? drawn : std::min(n_basins, limit);
            if (h > 0 && tt > 0 && kept > 0) {
                const double steps = tt / h;
                const double mb = steps * (double)kept * (double)s.vars.size() * 8.0 / (1024.0 * 1024.0);
                ImGui::TextDisabled("%d x %.0f points ~ %.1f MB on GPU (x4-5 on host)", kept, steps, mb);
            }
        }
    }

    // Статус актуальности
    const std::string sig = build_basins_phase_signature(c);
    if (!no_data && sig != sl.run_signature && !phase_busy) {
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f),
                           c.pp_autorun ? "Rebuilding..." : "Stale - press Run portraits");
    }
    // compute_phase_portrait кладёт в error тайминг успешного прогона
    // ("recompute: N ms") — это не ошибка, красить в оранжевый нечего.
    if (!sl.phase.result.error.empty()) {
        const bool is_timing = sl.phase.result.error.rfind("recompute:", 0) == 0;
        if (is_timing) ImGui::TextDisabled("%s", sl.phase.result.error.c_str());
        else ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "%s", sl.phase.result.error.c_str());
    }

    // Окна (проекции)
    ImGui::SeparatorText("Windows");
    int pr_to_remove = -1;
    for (int i = 0; i < (int)sl.phase.projections.size(); ++i) {
        Projection& pr = sl.phase.projections[i];
        // 3D в этом режиме не поддерживается (нет точечного рендера в
        // Plot3DView) — старые сессии с Phase3D мягко приводим к 2D.
        if (pr.type == ProjType::Phase3D) pr.type = ProjType::Phase2D;
        ImGui::PushID(2000 + i);
        ImGui::SetNextItemWidth(90);
        InputTextStr("##pplabel", pr.label); ImGui::SameLine();
        ImGui::SetNextItemWidth(110);
        const char* tnames[] = { "Phase 2D", "Time domain" };
        int t = (pr.type == ProjType::TimeDomain) ? 1 : 0;
        if (ImGui::Combo("##pptype", &t, tnames, 2)) {
            pr.type = (t == 1) ? ProjType::TimeDomain : ProjType::Phase2D;
            sl.phase.fit_request = true;
        }
        ImGui::SameLine();
        if (pr.type == ProjType::Phase2D) {
            const auto& vars = sl.phase.vars;
            ImGui::Text("X:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(55);
            if (ImGui::BeginCombo("##ppx", vars.empty() ? "-" : vars[pr.axis_x < (int)vars.size() ? pr.axis_x : 0].c_str())) {
                for (int k = 0; k < (int)vars.size(); ++k)
                    if (ImGui::Selectable(vars[k].c_str(), pr.axis_x == k)) { pr.axis_x = k; sl.phase.fit_request = true; }
                ImGui::EndCombo();
            }
            ImGui::SameLine(); ImGui::Text("Y:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(55);
            if (ImGui::BeginCombo("##ppy", vars.empty() ? "-" : vars[pr.axis_y < (int)vars.size() ? pr.axis_y : 0].c_str())) {
                for (int k = 0; k < (int)vars.size(); ++k)
                    if (ImGui::Selectable(vars[k].c_str(), pr.axis_y == k)) { pr.axis_y = k; sl.phase.fit_request = true; }
                ImGui::EndCombo();
            }
        } else {
            if ((int)pr.show_var.size() != (int)sl.phase.vars.size())
                pr.show_var.assign(sl.phase.vars.size(), true);
            ImGui::Text("vars:"); ImGui::SameLine();
            for (int k = 0; k < (int)sl.phase.vars.size(); ++k) {
                bool v = pr.show_var[k];
                if (ImGui::Checkbox(sl.phase.vars[k].c_str(), &v)) pr.show_var[k] = v;
                ImGui::SameLine();
            }
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("X")) pr_to_remove = i;
        ImGui::PopID();
    }
    if (pr_to_remove >= 0) { sl.phase.remove_projection(pr_to_remove); changed = true; }
    if (ImGui::Button("Add 2D plot")) {
        sl.phase.add_projection();
        changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Add time domain")) {
        sl.phase.add_projection();
        sl.phase.projections.back().type = ProjType::TimeDomain;
        changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset windows layout##pp")) sl.phase.layout_generation++;

    // Легенда/видимость бассейнов
    if (!sl.phase.ic_sets.empty()) {
        ImGui::SeparatorText("Basins drawn");
        ImGui::Checkbox("Legend shows initial conditions##pp", &sl.phase.legend_show_ic);
        const int* ids = basins_display_ids(c);
        for (int k = 0; k < (int)sl.phase.ic_sets.size(); ++k) {
            ImGui::PushID(3000 + k);
            ImGui::Checkbox("##ppvis", &sl.phase.ic_sets[k].visible);
            ImGui::SameLine();
            if (ids && k < (int)sl.cells.size() &&
                sl.cells[k] >= 0 && sl.cells[k] < (int)c.result.basin_idx.size()) {
                const ImVec4 col = basins_id_color(c, model, cfg_idx, ids[sl.cells[k]]);
                ImGui::ColorButton("##ppcol", col,
                                   ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoDragDrop,
                                   ImVec2(12, 12));
                ImGui::SameLine();
            }
            ImGui::TextUnformatted(sl.phase.ic_sets[k].label.c_str());
            ImGui::PopID();
        }
    }

    if (changed) s.phase_settings_dirty = true;
}

void draw_basins_controls(AppModel& model, SystemLibrary& lib) {
    BasinsAnalysisSession& s = model.basins_session;
    // Чей конфиг правит эта панель — для ПКМ-рассылки "во все такие же".
    model.broadcast_source_tab = BroadcastTab::Basins;
    model.broadcast_source_idx = s.active_config_index;

    ImGui::Text("Basins of attraction");
    ImGui::TextDisabled("DBSCAN clustering in (avgPeak, avgInterval) plane.");

    // Run / Run all... — сразу под пикером системы, над таб-баром: управляют активным конфигом
    // и должны быть доступны без прокрутки. "Run all..." кладёт выбранные индексы в
    // model.basins_queue; draw_gui тикает очередь после poll'ов.
    {
        const bool no_cfg = s.configs.empty();
        RunAllGroup g;
        g.pick_id = "pbasins_";
        g.n       = (int)s.configs.size();
        g.label   = [&s](int i) { return s.configs[i].label; };
        g.enqueue = [&model](int i) { model.basins_queue.push_back({i}); };
        draw_run_and_run_all("##run_all_basins", s.in_flight, no_cfg,
                             s.in_flight || no_cfg, 160.0f,
                             [&model, &s]() {
                                 if (!model.parametric_engine)
                                     model.parametric_engine = std::make_unique<ParametricEngine>();
                                 s.run_async(*model.parametric_engine, s.active_config_index);
                             },
                             {}, { g },
                             [&model]() { model.start_next_in_basins_queue(); },
                             model.basins_queue.size());
    }
    ImGui::Separator();

    // Tab bar: одна вкладка на Basins-config + "+" для add. Зеркалит
    // draw_bifurcation_controls. Активная вкладка хранится в
    // s.active_config_index и используется Ctrl+R + плотами.
    const TabBarResult tabs = draw_config_tab_bar(
        "##basins_tabs", "basins_tab_", (int)s.configs.size(),
        s.in_flight, s.running_config_index,
        [&s](int i) { return s.configs[i].label; },
        {},
        [&s]() { s.add_config(); });
    if (tabs.active    >= 0) s.active_config_index = tabs.active;
    if (tabs.to_remove >= 0) model.remove_basins_config(tabs.to_remove);

    if (s.configs.empty()) {
        ImGui::TextDisabled("No basins configs. Press '+' to add one.");
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
        s.active_config_index = 0;
    BasinsConfig& c = s.configs[s.active_config_index];

    // Inline rename для активной вкладки.
    draw_label_rename("Label##basins_label", c.label);
    ImGui::Separator();

    // Scheme
    draw_scheme_combo("Scheme", c.scheme, s.custom_schemes, {}, &s.enabled_builtin_schemes,
                      &model, model.is_map, &s.wrapper_schemes);
    ImGui::Separator();

    // Axes (X, Y по двум IC-переменным)
    if (!s.vars.empty()) {
        if (c.axis_x_var < 0 || c.axis_x_var >= (int)s.vars.size()) c.axis_x_var = 0;
        if (c.axis_y_var < 0 || c.axis_y_var >= (int)s.vars.size())
            c.axis_y_var = (s.vars.size() > 1) ? 1 : 0;
        const std::vector<const char*> items = c_str_list(s.vars);

        // ±log — сетка легаси basinsOfAttraction_logAxes: |x| по логарифму на [lo, hi] по обе
        // стороны от нуля, lo/hi тогда — границы модуля. Условия проверяет run_basins.
        auto symlog_toggle = [](const char* id, bool& on) {
            ImGui::SameLine();
            ImGui::Checkbox(id, &on);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("|x| log-spaced on [lo, hi] on both sides of 0\n"
                                  "(nodes -hi..-lo, +lo..+hi, monotone).\n"
                                  "lo, hi > 0; Resolution even.");
        };
        ImGui::SetNextItemWidth(kComboW);
        ImGui::Combo("Axis X (IC)", &c.axis_x_var, items.data(), (int)items.size());
        symlog_toggle("\xC2\xB1log##bas_xsym", c.axis_x_symlog);
        InputNumStr(c.axis_x_symlog ? "|X| lo" : "X lo", c.axis_x_lo_text, kFieldW);
        InputNumStr(c.axis_x_symlog ? "|X| hi" : "X hi", c.axis_x_hi_text, kFieldW);

        ImGui::SetNextItemWidth(kComboW);
        ImGui::Combo("Axis Y (IC)", &c.axis_y_var, items.data(), (int)items.size());
        symlog_toggle("\xC2\xB1log##bas_ysym", c.axis_y_symlog);
        InputNumStr(c.axis_y_symlog ? "|Y| lo" : "Y lo", c.axis_y_lo_text, kFieldW);
        InputNumStr(c.axis_y_symlog ? "|Y| hi" : "Y hi", c.axis_y_hi_text, kFieldW);
    } else {
        ImGui::TextDisabled("No variables (load a system first)");
    }
    InputNumStr("Resolution", c.n_pts_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::Resolution, {},
                                        "Resolution", c.n_pts_text); });

    // Writable var (для peak finder)
    draw_writable_var_combo(s.vars, c.writable_var, "Writable var##bas_wv");

    ImGui::Separator();

    // Integration (collapsible, swapped above Features)
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
        f.adaptive      = &c.adaptive;
        f.adaptive_opts = kAdUiRaw | kAdUiInterp;
        draw_integration_block("Integration", c.scheme, s.custom_schemes, f, &model, model.is_map);
    }

    // Features (DBSCAN axes + plot data) (collapsible)
    // 12 фич (см. BF_* в configCUDA.h / enum BasinFeature в analysis_session.h).
    // Feature 1 пишется в outAvgPeaks-буфер (X-координата DBSCAN), Feature 2 —
    // в AvgTimeOfPeaks-буфер (Y-координата). Множители применяются ПОСЛЕ
    // вычисления фичи и нужны для масштабирования кластеризации.
    if (ImGui::CollapsingHeader("Features (DBSCAN axes + plot data)",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        if (c.feature1 < 0 || c.feature1 >= BF_FEATURE_COUNT) c.feature1 = BF_FEATURE1_DEFAULT;
        if (c.feature2 < 0 || c.feature2 >= BF_FEATURE_COUNT) c.feature2 = BF_FEATURE2_DEFAULT;
        ImGui::SetNextItemWidth(220);
        ImGui::Combo("Feature 1##bas", &c.feature1, kBasinFeatureNames, BF_FEATURE_COUNT);
        ImGui::SameLine();
        InputNumStr("mult##bas_f1", c.mult_feature1_text, 80);
        ImGui::SetNextItemWidth(220);
        ImGui::Combo("Feature 2##bas", &c.feature2, kBasinFeatureNames, BF_FEATURE_COUNT);
        ImGui::SameLine();
        InputNumStr("mult##bas_f2", c.mult_feature2_text, 80);
    }

    // DBSCAN eps — кластеризационный радиус в пространстве (Feature 1, Feature 2). Оставлен
    // снаружи Features-секции: тюнится чаще, чем выбор самих фич. Рядом кнопка "Clustering":
    // перезапускает только DBSCAN-фазу по уже посчитанным фичам (avg_peaks / avg_intervals в
    // c.result) — в десятки-сотни раз быстрее полного Run. Дизейблится без валидного
    // предыдущего результата или во время расчёта.
    InputNumStr("DBSCAN eps", c.eps_dbscan_text, kFieldW);
    ImGui::SameLine();
    const bool can_recluster = !s.in_flight && c.last_run_ok &&
                               !c.result.avg_peaks.empty() &&
                               !c.result.avg_intervals.empty() &&
                               !c.result.helpful_array.empty();
    if (!can_recluster) ImGui::BeginDisabled();
    if (ImGui::Button("Clustering")) {
        if (!model.parametric_engine)
            model.parametric_engine = std::make_unique<ParametricEngine>();
        s.run_recluster_async(*model.parametric_engine, s.active_config_index);
    }
    if (!can_recluster) ImGui::EndDisabled();
    ImGui::TextDisabled("Clustering radius in (Feature 1, Feature 2) space.");

    draw_named_num_fields("Initial conditions", s.vars, c.initial_conditions,
                          "Values for non-axis variables.",
                          &model, BroadcastField::InitCondition);
    draw_named_num_fields("Parameters", s.params, c.param_values,
                          nullptr, &model, BroadcastField::Param);

    // Phase portraits from basins (collapsible)
    // Отдельные окна 2D / Time domain по одной представительной точке из
    // каждого найденного бассейна (см. draw_basins_phase_controls).
    if (ImGui::CollapsingHeader("Phase portraits")) {
        draw_basins_phase_controls(model, s.active_config_index);
    }

    draw_csv_output_block("CSV output", "Save to file", c.csv_save_enabled,
                          "##basins_csv_path", c.csv_output_path,
                          "Writes 4 files: <path>, _1.csv (Feature 1), _2.csv (Feature 2), _3.csv (states).");

    if (c.last_run_ok) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
            "OK: %dx%d, %d clusters (+ %d FP clusters)",
            c.result.n_pts, c.result.n_pts,
            c.result.n_clusters, -c.result.min_cluster_idx);
        draw_regime_summary(c.result.helpful_array, &c.regime_counts, c.data_generation);
        draw_adaptive_stats(c.result.ad_stats);
    } else if (!c.last_error.empty()) {
        draw_error_box("##basins_err", c.last_error);
    }
}

// Plot Basins: inner tab-bar по 5 представлениям. Heatmap-views — per (config × tab) в
// std::map по owner_id: HeatmapView/Plot2DView хранят data_gen_cached внутри без учёта
// owner_id, поэтому один view нельзя переиспользовать между configs (после Run All у двух
// configs одинаковый data_generation=1 → кэш не инвалидируется и на чужой вкладке виден
// предыдущий буфер). Map с lazy-init решает это и сохраняет независимый zoom/pan per (config, tab).
// spiral_coords_from_center / renumber_basins_spiral / ensure_basins_spiral_cache переехали в
// analysis_session.cpp (объявления в analysis_session.h): перенумерация нужна не только для
// отрисовки, но и для выбора представительных ячеек в фазовых портретах
// (BasinsAnalysisSession::rebuild_phase_ics), а тот слой про GUI ничего не знает.

void draw_basins_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) {
    BasinsAnalysisSession& s = model.basins_session;
    if (s.configs.empty()) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
        s.active_config_index = 0;
    BasinsConfig& c = s.configs[s.active_config_index];
    // Owner IDs зависят от индекса config — каждый basin имеет независимый
    // zoom/pan per inner tab. Схема: 0x1BA50000 + cfg*5 + tab (max 50 configs).
    const unsigned base_oid = basins_base_oid(s.active_config_index);
    static std::unique_ptr<PlotRenderer> renderer;
    // hm_basins — на уровне файла (g_hm_basins), см. комментарий у объявления.
    static std::map<unsigned, std::unique_ptr<HeatmapView>> hm_avgpk, hm_avgint, hm_states;
    static std::map<unsigned, std::unique_ptr<Plot2DView>>  scatter_views;
    // Точки Scatter, разложенные по кластерам (см. вкладку 4 ниже). Кэш
    // отрисовки, не состояние UI: восстанавливается из c.result в любой момент.
    struct ScatterClusterCache {
        bool   valid = false;
        int    gen = 0;
        bool   renumber = false;
        size_t total = 0;
        int    valid_pts = 0;
        double vmin = 0.0, vmax = 0.0;   // basins_colorbar_range
        const int*  src_idx = nullptr;
        const void* src_x   = nullptr;
        std::vector<int>                 ids;
        std::vector<std::vector<double>> bufs;
        std::vector<std::string>         labels;
    };
    static std::map<unsigned, ScatterClusterCache> scatter_cache;
    if (!renderer) renderer = std::make_unique<PlotRenderer>();
    static int seen_epoch = 0;
    if (session_reloaded(model, AppModel::AppMode::Basins, seen_epoch)) {
        scatter_cache.clear();
        invalidate_views(g_hm_basins);
        invalidate_views(hm_avgpk);
        invalidate_views(hm_avgint);
        invalidate_views(hm_states);
        invalidate_views(scatter_views);
    }
    auto get_hm = [](std::map<unsigned, std::unique_ptr<HeatmapView>>& m, unsigned oid,
                     bool discrete_default) -> HeatmapView& {
        auto& slot = m[oid];
        if (!slot) {
            slot = std::make_unique<HeatmapView>();
        }
        // Флаг discrete_default читается в render() на первом кадре с данными
        // и один раз применяется (см. heatmap_view.cpp). Ставим каждый кадр
        // сам параметр — легко и дёшево; apply идёт ровно один раз.
        slot->discrete_default = discrete_default;
        return *slot;
    };
    auto get_scatter = [](unsigned oid) -> Plot2DView& {
        auto& slot = scatter_views[oid];
        if (!slot) {
            slot = std::make_unique<Plot2DView>();
            slot->points_mode = true;
            slot->show_legend = false;
            slot->point_size_px = 3.0f;
            // Запас 2.5% размаха с каждой стороны по обеим осям: без него крайние
            // кластеры ложились на рамку и маркеры обрезались ею наполовину.
            slot->pad_x = true;
            slot->pad_y = true;
            slot->pad_frac = 0.025;
            slot->pad_degenerate_rel = true;
        }
        return *slot;
    };
    HeatmapView& hm_basins_v = get_hm(g_hm_basins, base_oid + 0u, /*discrete*/ true);
    HeatmapView& hm_avgpk_v  = get_hm(hm_avgpk,  base_oid + 1u, false);
    HeatmapView& hm_avgint_v = get_hm(hm_avgint, base_oid + 2u, false);
    HeatmapView& hm_states_v = get_hm(hm_states, base_oid + 3u, false);
    Plot2DView&  scatter_v   = get_scatter(base_oid + 4u);

    if (!c.last_run_ok || c.result.basin_idx.empty()) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }

    // Right-click "Export data..." — exports the full 4-file Basins set (basin_idx, avg_peaks,
    // avg_intervals, helpful_array) at the chosen path. All four heatmap views share the same
    // source result, so one lambda works for any tab; scatter (tab 4) uses Plot2DView and gets
    // the same hook. Если включён Renumber (spiral), basin_idx уйдёт в файл перенумерованным —
    // чтобы экспорт совпадал с тем, что пользователь видит на экране.
    const bool basins_busy = s.in_flight &&
                             s.active_config_index == s.running_config_index;
    auto basins_export_extras = [&c, &cb, basins_busy]() {
        draw_export_menu_item(basins_busy, cb, [&c](const std::string& path) {
            if (c.renumber_spiral) {
                ensure_basins_spiral_cache(c);
                BasinsResult tmp = c.result;
                tmp.basin_idx       = c.basin_idx_spiral;
                tmp.n_clusters      = c.n_clusters_spiral;
                tmp.min_cluster_idx = c.min_cluster_idx_spiral;
                data_export::export_basins(tmp, path);
            } else {
                data_export::export_basins(c.result, path);
            }
        });
    };
    // Ctrl+T: у четырёх карт обе оси — сетка начальных условий, то есть прямо
    // поля расчёта. Scatter живёт в пространстве фич (avgPeak, avgInterval) и
    // диапазонов расчёта по своим осям не имеет — ему только экспорт.
    const std::vector<ViewRangeTarget> vrt = { vr_target(s.configs, s.active_config_index,
        &BasinsConfig::axis_x_lo_text, &BasinsConfig::axis_x_hi_text,
        &BasinsConfig::axis_y_lo_text, &BasinsConfig::axis_y_hi_text) };
    auto grid_extras = [&](HeatmapView& hv) {
        return [&basins_export_extras, &vrt, &model, &hv]() {
            basins_export_extras();
            draw_view_range_menu(vrt, hv.x_axis, hv.y_axis, hv.swap_axes, &model);
        };
    };
    hm_basins_v.popup_extras = grid_extras(hm_basins_v);
    hm_avgpk_v.popup_extras  = grid_extras(hm_avgpk_v);
    hm_avgint_v.popup_extras = grid_extras(hm_avgint_v);
    hm_states_v.popup_extras = grid_extras(hm_states_v);
    scatter_v.popup_extras   = basins_export_extras;

    // Inner tab-bar — переключение по 5 видам.
    // Имена 2-го и 3-го табов нейтральные — они показывают выбранную фичу
    // (Feature 1/2 из BasinsConfig), которая может быть не "avg peaks/interval".
    const char* tab_names[5] = { "Basins", "Feature 1", "Feature 2", "States", "Scatter" };
    if (ImGui::BeginTabBar("##basins_inner")) {
        for (int t = 0; t < 5; ++t) {
            if (ImGui::BeginTabItem(tab_names[t])) {
                c.active_plot_tab = t;
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
    }

    // Combo выбора colormap — свой для каждого heatmap-таба (Basins/AvgPk/AvgInt/States).
    // Каждый выбор пишется в своё поле AppModel и персистится в _app_config.json. Эти поля
    // независимы от model.heatmap_colormap (тот шарится Bif/LLE/LS) — смена здесь их не
    // затрагивает и наоборот. Scatter-таб использует Plot2DView, без colormap.
    {
        // Per-tab colormap: приоритет — выбор, сохранённый в ЭТОМ config'е
        // (уходит в _last_basins.json); -1 = не задан → app-дефолт из Settings.
        // Так два basins-config'а могут иметь разные колормапы — как у
        // Bif/LLE/LS/DFT1D. Раньше выбор был только глобальный.
        int* cfg_field  = nullptr;
        int  app_default = kColormapTurbo;
        HeatmapView* active_hm = nullptr;
        switch (c.active_plot_tab) {
            case 0: cfg_field = &c.colormap_idx[0]; app_default = model.basins_colormap;        active_hm = &hm_basins_v; break;
            case 1: cfg_field = &c.colormap_idx[1]; app_default = model.basins_avgpk_colormap;  active_hm = &hm_avgpk_v;  break;
            case 2: cfg_field = &c.colormap_idx[2]; app_default = model.basins_avgint_colormap; active_hm = &hm_avgint_v; break;
            case 3: cfg_field = &c.colormap_idx[3]; app_default = model.basins_states_colormap; active_hm = &hm_states_v; break;
            default: break;  // Scatter (4) — Plot2DView, без colormap
        }
        // Renumber (spiral) — общий для табов 0 (Basins) и 4 (Scatter).
        // На остальных табах не показываем — там cluster id'ы не отрисовываются.
        auto draw_renumber = [&](bool same_line) {
            if (c.active_plot_tab != 0 && c.active_plot_tab != 4) return;
            if (same_line) ImGui::SameLine();
            if (ImGui::Checkbox("Renumber (spiral)", &c.renumber_spiral)) {
                // Заставляем cache пересчитаться при следующем доступе,
                // даже если data_generation тот же (после toggle off→on).
                c.basin_idx_spiral_gen = -1;
            }
        };

        if (cfg_field && active_hm) {
            // Синхронизируем выбор в сам view ДО тулбара: тулбар читает
            // hv.colormap как источник истины. Без этого на первом кадре combo
            // показал бы дефолт view'а вместо сохранённого значения.
            const int cm = colormap_id_or(*cfg_field, app_default);
            active_hm->colormap = (HeatmapColormap)cm;

            HeatmapToolbarOpts topts;
            topts.persist_colormap = [&](int picked) {
                *cfg_field = picked;   // per-config, уходит в сессию
                if (!model.loaded_name.empty())
                    lib.save_session(model.loaded_name, "_last_basins",
                                     session_to_json_basins(model.basins_session));
            };
            topts.extras = [&]() { draw_renumber(/*same_line*/ true); };
            draw_heatmap_toolbar(*active_hm, topts);
        } else {
            // Scatter-таб: Renumber и стиль точек (форма, размер, α — per-config,
            // уходят в _last_basins.json через phase_settings_dirty).
            draw_renumber(/*same_line*/ false);
            if (c.active_plot_tab == 4) {
                ImGui::SameLine();
                if (draw_style_toolbar("Custom point style", "bas_scatter", c.scatter_custom_style,
                        [&c]() {
                            if (c.scatter_marker < 0 || c.scatter_marker >= kPointMarkerCount)
                                c.scatter_marker = 0;
                            bool ch = false;
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::Combo("Marker##bas_scatter", &c.scatter_marker,
                                               kPointMarkerNames, kPointMarkerCount);
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Point size##bas_scatter", &c.scatter_point_size,
                                                     0.5f, 12.0f, "%.1f");
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Alpha##bas_scatter", &c.scatter_alpha,
                                                     0.0f, 1.0f, "%.2f");
                            return ch;
                        }))
                    s.phase_settings_dirty = true;
            }
        }
    }
    if (c.renumber_spiral) ensure_basins_spiral_cache(c);

    int n = c.result.n_pts;
    size_t total = (size_t)n * (size_t)n;
    double xlo = c.result.axis_x_lo, xhi = c.result.axis_x_hi;
    double ylo = c.result.axis_y_lo, yhi = c.result.axis_y_hi;
    // Шкала осей — свойство посчитанной сетки (результата), а не текущих галок конфига.
    for (HeatmapView* hv : { &hm_basins_v, &hm_avgpk_v, &hm_avgint_v, &hm_states_v }) {
        hv->x_axis.symlog = c.result.axis_x_symlog;
        hv->y_axis.symlog = c.result.axis_y_symlog;
    }

    auto axis_name = [&](int var_idx) -> std::string {
        if (var_idx >= 0 && var_idx < (int)s.vars.size())
            return s.vars[var_idx] + "(0)";
        return std::string("x");
    };
    std::string ax_x = axis_name(c.result.axis_x_var);
    std::string ax_y = axis_name(c.result.axis_y_var);

    bool fit = c.fit_request;
    if (fit) c.fit_request = false;

    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();

    if (c.active_plot_tab == 0) {
        // Basins idx — turbo-discrete через GL_NEAREST. Spectrum [min..max].
        // Если Renumber (spiral) включён, читаем перенумерованный basin_idx
        // и считаем colorbar-диапазон от него же — иначе brightest band не
        // совпадёт с фактическими id'ами на хитмапе.
        const int* src_idx = c.renumber_spiral
                                 ? c.basin_idx_spiral.data()
                                 : c.result.basin_idx.data();
        static std::vector<double> buf;
        buf.resize(total);
        for (size_t k = 0; k < total; ++k) buf[k] = (double)src_idx[k];
        // Диапазон colorbar'а — общий хелпер (шарится с цветом траекторий в
        // фазовых портретах по бассейнам), см. basins_colorbar_range.
        double vmin = 0.0, vmax = 0.0;
        basins_colorbar_range(c, vmin, vmax);
        // colormap уже засинхронен из model.basins_colormap перед тулбаром.
        hm_basins_v.x_axis.name = ax_x;
        hm_basins_v.y_axis.name = ax_y;
        // gen-token включает renumber_spiral, иначе HeatmapView::data_gen_cached
        // не инвалидируется при toggle галки и пиксели остаются от прошлой версии.
        // Сам swap_axes уже отлавливается через HeatmapView::swap_axes_cached_.
        int hm_basins_gen = c.data_generation * 2 + (c.renumber_spiral ? 1 : 0);
        // id бассейна: <= -1 — кластер равновесий, 0 — unbound, >= 1 — колебательный.
        hm_basins_v.cell_regime = regime_cell_label(c.result.helpful_array, n);
        hm_basins_v.value_name = "basin";
        hm_basins_v.value_is_regime_code = true;
        hm_basins_v.render(*renderer, origin, avail,
                          /*owner_id*/ base_oid + 0u, hm_basins_gen,
                          n, n, buf.data(),
                          xlo, xhi, ylo, yhi,
                          vmin, vmax, fit);
        handle_view_range_keys(vrt, hm_basins_v.x_axis, hm_basins_v.y_axis, plot_window_active(),
                               hm_basins_v.swap_axes, &model);
    }
    else if (c.active_plot_tab == 1) {
        hm_avgpk_v.x_axis.name = ax_x;
        hm_avgpk_v.y_axis.name = ax_y;
        // У равновесия здесь последняя точка траектории — настоящее значение, не код.
        hm_avgpk_v.cell_regime = regime_cell_label(c.result.helpful_array, n);
        hm_avgpk_v.value_name = "feature 1";
        hm_avgpk_v.render(*renderer, origin, avail,
                         /*owner_id*/ base_oid + 1u, c.data_generation,
                         n, n, c.result.avg_peaks.data(),
                         xlo, xhi, ylo, yhi,
                         c.result.avg_peaks_min, c.result.avg_peaks_max, fit);
        handle_view_range_keys(vrt, hm_avgpk_v.x_axis, hm_avgpk_v.y_axis, plot_window_active(),
                               hm_avgpk_v.swap_axes, &model);
    }
    else if (c.active_plot_tab == 2) {
        hm_avgint_v.x_axis.name = ax_x;
        hm_avgint_v.y_axis.name = ax_y;
        // У равновесия avgPeakFinderCUDA пишет сюда -1 — код, а не величину фичи.
        hm_avgint_v.cell_regime = regime_cell_label(c.result.helpful_array, n);
        hm_avgint_v.value_name = "feature 2";
        hm_avgint_v.value_is_regime_code = true;
        hm_avgint_v.render(*renderer, origin, avail,
                          /*owner_id*/ base_oid + 2u, c.data_generation,
                          n, n, c.result.avg_intervals.data(),
                          xlo, xhi, ylo, yhi,
                          c.result.avg_intervals_min, c.result.avg_intervals_max, fit);
        handle_view_range_keys(vrt, hm_avgint_v.x_axis, hm_avgint_v.y_axis, plot_window_active(),
                               hm_avgint_v.swap_axes, &model);
    }
    else if (c.active_plot_tab == 3) {
        // States: рисуем helpful_array КАК ЕСТЬ, в канонических REGIME_*
        // (-1 = FP, 0 = Unbound, 1 = Oscillation). Раньше здесь была
        // перенумерация в 0/1/2, из-за которой подпись под плотом, легенда и
        // коды в экспортируемом _3.csv означали разные вещи.
        static std::vector<double> buf;
        buf.resize(total);
        for (size_t k = 0; k < total; ++k)
            buf[k] = (double)regime_code(c.result.helpful_array[k]);
        hm_states_v.x_axis.name = ax_x;
        hm_states_v.y_axis.name = ax_y;
        hm_states_v.cell_regime = regime_cell_label(c.result.helpful_array, n, /*with_oscillation*/ true);
        hm_states_v.value_is_regime_code = true;
        hm_states_v.render(*renderer, origin, avail,
                          /*owner_id*/ base_oid + 3u, c.data_generation,
                          n, n, buf.data(),
                          xlo, xhi, ylo, yhi,
                          -1.0, 1.0, fit);
        handle_view_range_keys(vrt, hm_states_v.x_axis, hm_states_v.y_axis, plot_window_active(),
                               hm_states_v.swap_axes, &model);
        // Подсказка под плотом — числовые уровни, цвет зависит от выбранной colormap.
        ImGui::TextDisabled("Levels: %d = %s, %d = %s, %d = %s",
                            REGIME_FIXED_POINT, regime_label(REGIME_FIXED_POINT),
                            REGIME_UNBOUND,     regime_label(REGIME_UNBOUND),
                            REGIME_OSCILLATION, regime_label(REGIME_OSCILLATION));
    }
    else if (c.active_plot_tab == 4) {
        // Scatter (avgPeak, avgInterval), точки сгруппированы по basin_idx.
        // Каждый кластер — своя серия (PlotSeriesInput с собственным цветом).
        // С renumber_spiral берём перенумерованные id'ы, чтобы цвета и подписи
        // ("c1", "c2", ...) совпадали с тем, что показывает Basins heatmap.
        const int* src_idx = c.renumber_spiral
                                 ? c.basin_idx_spiral.data()
                                 : c.result.basin_idx.data();
        // Раскладка точек по кластерам — ОДИН раз на результат, а не каждый кадр:
        // на сетке 2000x2000 это 4 млн push_back в std::map и ~64 МБ аллокаций за
        // кадр, тогда как VBO в Plot2DView по тому же gen-токену не перезаливается
        // вовсе. Ключ кэша повторяет gen-токен scatter'а (data_generation +
        // renumber_spiral) плюс размер сетки. Прореживания нет: рисуются все точки,
        // редкие выбросы не теряются.
        ScatterClusterCache& sc = scatter_cache[base_oid + 4u];
        // Указатели на буферы результата — страховка на случай, когда тот же oid
        // достаётся другому config'у (удаление вкладки) с совпавшим data_generation.
        if (!sc.valid || sc.gen != c.data_generation || sc.renumber != c.renumber_spiral ||
            sc.total != total || sc.src_idx != src_idx || sc.src_x != c.result.avg_peaks.data()) {
            std::map<int, std::vector<double>> bufs;
            int valid_pts = 0;
            for (size_t k = 0; k < total; ++k) {
                int id = src_idx[k];
                double xp = c.result.avg_peaks[k];
                double yp = c.result.avg_intervals[k];
                if (!std::isfinite(xp) || !std::isfinite(yp)) continue;
                if (xp ==  kSentinelNoData || xp == -kSentinelNoData ||
                    yp ==  kSentinelNoData || yp == -kSentinelNoData) continue;
                bufs[id].push_back((double)xp);
                bufs[id].push_back((double)yp);
                ++valid_pts;
            }
            sc.ids.clear(); sc.bufs.clear(); sc.labels.clear();
            sc.ids.reserve(bufs.size()); sc.bufs.reserve(bufs.size()); sc.labels.reserve(bufs.size());
            for (auto& kv : bufs) {
                sc.ids.push_back(kv.first);
                sc.bufs.push_back(std::move(kv.second));
                sc.labels.push_back("c" + std::to_string(kv.first));
            }
            sc.valid_pts = valid_pts;
            // Диапазон colorbar'а карты зависит только от данных и Renumber —
            // кэшируем вместе с раскладкой (он сканирует всю helpful_array).
            basins_colorbar_range(c, sc.vmin, sc.vmax);
            sc.gen       = c.data_generation;
            sc.renumber  = c.renumber_spiral;
            sc.total     = total;
            sc.src_idx   = src_idx;
            sc.src_x     = c.result.avg_peaks.data();
            sc.valid     = true;
        }
        if (sc.valid_pts == 0) {
            ImGui::TextDisabled("No valid (avgPeak, avgInterval) points.");
            return;
        }

        std::vector<PlotSeriesInput> series_in;
        std::vector<bool> init_vis, glob_vis;
        series_in.reserve(sc.bufs.size());
        for (size_t j = 0; j < sc.bufs.size(); ++j) {
            const int id = sc.ids[j];
            // Цвет кластера — ровно тот, которым его ячейки красит карта «Basins»
            // (basins_id_color: та же колормапа, reverse, дискретизация и диапазон
            // FP..0..Osc, живые настройки вью карты). Пересчёт каждый кадр дешёвый
            // (по вызову на кластер), зато смена колормапы видна сразу.
            PlotSeriesInput si;
            si.points   = sc.bufs[j].empty() ? nullptr : sc.bufs[j].data();
            si.n_points = (int)(sc.bufs[j].size() / 2);
            si.color    = basins_id_color(c, model, s.active_config_index, id, sc.vmin, sc.vmax);
            si.label    = sc.labels[j];
            series_in.push_back(si);
            init_vis.push_back(true);
            glob_vis.push_back(true);
        }
        // Имена осей scatter'а — выбранные фичи, из общего kBasinFeatureNames
        // (analysis_session.h: тот же список, что в комбо настроек и в _config.csv).
        int f1 = (c.feature1 >= 0 && c.feature1 < BF_FEATURE_COUNT) ? c.feature1 : BF_FEATURE1_DEFAULT;
        int f2 = (c.feature2 >= 0 && c.feature2 < BF_FEATURE_COUNT) ? c.feature2 : BF_FEATURE2_DEFAULT;
        scatter_v.x_axis.name = kBasinFeatureNames[f1];
        scatter_v.y_axis.name = kBasinFeatureNames[f2];
        // ε-окружность под курсором. Здесь она точна по определению: точки scatter'а — ровно те
        // avg_peaks/avg_intervals-буферы, которые читает cell-level DBSCAN (CUDA_dbscan_kernel:
        // sqrt(dx^2+dy^2) <= eps), и множители Feature 1/2 в них уже вписаны на GPU, поэтому
        // накрытые кружком ячейки — кандидаты попасть в один бассейн.
        scatter_v.hover_circle_r = parse_ratio_or(c.eps_dbscan_text, 0.5);
        // Стиль точек — каждый кадр из config'а (immediate mode). Выключенный
        // стиль возвращает прежний вид: marker -1 = сплошной GL-пойнт 3 px.
        // α через point_alpha, а не через цвет серии: point_alpha включает
        // смешивание (у сплошного GL-пойнта альфа писалась бы прямо в текстуру).
        if (c.scatter_custom_style) {
            const int mk = (c.scatter_marker >= 0 && c.scatter_marker < kPointMarkerCount)
                         ? c.scatter_marker : 0;
            scatter_v.point_marker  = mk;
            scatter_v.point_size_px = c.scatter_point_size;
            scatter_v.point_alpha   = c.scatter_alpha;
        } else {
            scatter_v.point_marker  = -1;
            scatter_v.point_size_px = 3.0f;
            scatter_v.point_alpha   = -1.0f;
        }
        // gen-token включает renumber_spiral — иначе Plot2DView::series_cache_
        // не перезаливает GPU-буфер и подписи/цвета остаются от прошлой версии.
        int scatter_gen = c.data_generation * 2 + (c.renumber_spiral ? 1 : 0);
        scatter_v.render(*renderer, origin, avail,
                             /*owner_id*/ base_oid + 4u, scatter_gen,
                             series_in, init_vis, glob_vis, fit);
    }
}

// Окна фазовых портретов по бассейнам — только для АКТИВНОГО config'а
// (панель хитмапы показывает его же). Набор проекций и результат у каждого
// config'а свои, поэтому переключение вкладки сразу показывает свои окна с
// уже посчитанными данными.
void draw_basins_phase_windows(AppModel& model, const GuiCallbacks& cb) {
    BasinsAnalysisSession& s = model.basins_session;
    s.ensure_phase_slots();
    if (s.configs.empty()) return;
    int ci = s.active_config_index;
    if (ci < 0 || ci >= (int)s.configs.size()) ci = 0;
    BasinsConfig&    c  = s.configs[(size_t)ci];
    BasinsPhaseSlot& sl = s.phase_slot(ci);

    // 3D-проекций в этом режиме нет (Plot3DView не умеет точечные серии, а
    // бассейны-равновесия рисуются маркером) — старые сессии мягко приводим.
    for (auto& pr : sl.phase.projections)
        if (pr.type == ProjType::Phase3D) pr.type = ProjType::Phase2D;

    // display-id ячейки → цвет бассейна + признак «равновесие» (id < 0).
    const int* ids = basins_display_ids(c);
    const size_t n_cells = c.result.basin_idx.size();
    PhaseStyleFn style_fn = [&c, &sl, &model, ci, ids, n_cells]
                            (int k, PhaseSeriesStyle& out) -> bool {
        if (!ids || k < 0 || k >= (int)sl.cells.size()) return false;
        const int cell = sl.cells[(size_t)k];
        if (cell < 0 || (size_t)cell >= n_cells) return false;
        const int id = ids[(size_t)cell];
        out.color      = basins_id_color(c, model, ci, id);
        out.as_point   = (id < 0);         // FP-бассейн — одна точка
        out.point_size = c.pp_marker_size;
        out.marker     = (int)PointMarker::Circle;
        return true;
    };

    // Диаграмма признаков здесь — облако (пик; IPI) по выбранным ячейкам, а cell-level
    // DBSCAN бассейнов мерит eps в пространстве (Feature1 x mult1, Feature2 x mult2) по
    // ОДНОЙ точке на ячейку. Единицы совпадают с осями диаграммы только при выбранных
    // средних: тогда точка ячейки — центроид этого облака и eps-кружок на нём осмыслен.
    // На RMS/StDev/log-фичах множители относятся к другой величине, поэтому там оставляем
    // сырые оси и без кружка. Клампы f1/f2 повторяют engine (analysis_session.cpp): вне
    // диапазона он берёт дефолт, т.е. средние.
    FeatureClusterParams clust;
    {
        const int f1 = (c.feature1 >= 0 && c.feature1 < BF_FEATURE_COUNT) ? c.feature1 : BF_FEATURE1_DEFAULT;
        const int f2 = (c.feature2 >= 0 && c.feature2 < BF_FEATURE_COUNT) ? c.feature2 : BF_FEATURE2_DEFAULT;
        if (f1 == BF_AVG_PEAKS && f2 == BF_AVG_INTERVALS) {
            clust.valid         = true;
            clust.mult_peak     = parse_ratio_or(c.mult_feature1_text, 1.0);
            clust.mult_interval = parse_ratio_or(c.mult_feature2_text, 1.0);
            clust.eps           = parse_ratio_or(c.eps_dbscan_text,    0.5);
        }
    }

    // title_suffix / owner_id_delta — свои на config: docking-раскладка в
    // imgui.ini и кэш рендереров не должны пересекаться ни между конфигами,
    // ни с проекциями режима Phase analysis.
    const size_t n_before = sl.phase.projections.size();
    draw_projection_windows(sl.phase, cb, {}, {},
                            "##bpp" + std::to_string(ci),
                            0x1BA50000 + ci * 16,
                            style_fn, clust);
    // Окно могли закрыть крестиком — список окон изменился, надо сохранить.
    if (sl.phase.projections.size() != n_before) s.phase_settings_dirty = true;
}

// Раз в кадр: пометить устаревшие портреты и, если включён Autorun, поставить их в
// очередь. Дебаунс 300 мс — иначе каждое нажатие клавиши в текстовом поле стартовало бы
// новый GPU-прогон. Проходим по ВСЕМ config'ам: после «Run all…» автозапуск должен
// подхватить каждый пересчитанный конфиг, а не только открытый.
void basins_phase_tick(AppModel& model) {
    BasinsAnalysisSession& s = model.basins_session;
    s.ensure_phase_slots();
    const double now = ImGui::GetTime();
    for (int i = 0; i < (int)s.configs.size(); ++i) {
        BasinsConfig&    c  = s.configs[(size_t)i];
        BasinsPhaseSlot& sl = *s.phase_slots[(size_t)i];
        const std::string sig = build_basins_phase_signature(c);
        if (sig != sl.pending_signature) {
            sl.pending_signature = sig;
            sl.pending_since     = now;
        }
        if (sig == sl.run_signature) continue;         // актуально
        if (!c.pp_autorun) continue;
        if (!c.last_run_ok || c.result.basin_idx.empty()) continue;
        if (sl.phase.in_flight) continue;
        if (now - sl.pending_since < kBasinsPhaseDebounceSec) continue;   // ещё печатают
        s.request_phase_run(i);
    }
}
