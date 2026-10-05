#include "gui_internal.h"

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
    auto& renderers = ui_resource<std::map<int, std::unique_ptr<PlotRenderer>>, 5>();
    auto& heatmaps = ui_resource<std::map<int, std::map<int, std::unique_ptr<HeatmapView>>>, 6>();

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
void basins_colorbar_range(const BasinsConfig& c, double& vmin, double& vmax) {
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
}
