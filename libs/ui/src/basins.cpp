#include "gui_internal.h"

namespace ucuda::ui_detail {


// owner_id блока плотов одного basins-config'а (5 внутренних табов на config).
[[nodiscard]] unsigned basins_base_oid(int config_index) {
    return 0x1BA50000u + (unsigned)config_index * 5u;
}


// Цвет бассейна `display_id` — ровно тот, которым его красит хитмапа.
// Повторяет квантование из FS_HEATMAP (plot_renderer.cpp): t по [vmin,vmax],
// дискретизация по полосам с edge-aligned сэмплированием, затем reverse.
ImVec4 basins_id_color(const BasinsConfig& c, const AppModel& model,
                              int config_index, int display_id) {
    double vmin = 0.0, vmax = 0.0;
    basins_colorbar_range(c, vmin, vmax);

    int  cm      = (c.colormap_idx[0] >= 0) ? c.colormap_idx[0] : model.basins_colormap;
    bool reverse = false;
    // discrete_default для этого вью = true, поэтому дефолт (пока панель ни
    // разу не рисовалась) — дискретный, по одной полосе на целый id.
    int  n_disc  = std::max(1, (int)std::lround(vmax - vmin) + 1);

    auto it = basins_heatmaps().find(basins_base_oid(config_index) + 0u);
    if (it != basins_heatmaps().end() && it->second) {
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


// Панель «Phase portraits» внутри Basins Controls. Интегратор и параметры —
// из самого basins-config'а; своё тут только время моделирования, прореживание,
// ограничитель числа аттракторов и выбор CPU/GPU.
void draw_basins_phase_controls(AppModel& model, int cfg_idx) {
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
    if (is_custom_scheme(c.scheme, s.custom_schemes) && !c.pp_use_gpu) {
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
    draw_scheme_combo("Scheme", c.scheme, s.custom_schemes, {}, &s.enabled_builtin_schemes);
    draw_adaptive_controls(c.adaptive);
    ImGui::Separator();

    // Axes (X, Y по двум IC-переменным)
    if (!s.vars.empty()) {
        if (c.axis_x_var < 0 || c.axis_x_var >= (int)s.vars.size()) c.axis_x_var = 0;
        if (c.axis_y_var < 0 || c.axis_y_var >= (int)s.vars.size())
            c.axis_y_var = (s.vars.size() > 1) ? 1 : 0;
        const std::vector<const char*> items = c_str_list(s.vars);

        ImGui::SetNextItemWidth(kComboW);
        ImGui::Combo("Axis X (IC)", &c.axis_x_var, items.data(), (int)items.size());
        InputNumStr("X lo", c.axis_x_lo_text, kFieldW);
        InputNumStr("X hi", c.axis_x_hi_text, kFieldW);

        ImGui::SetNextItemWidth(kComboW);
        ImGui::Combo("Axis Y (IC)", &c.axis_y_var, items.data(), (int)items.size());
        InputNumStr("Y lo", c.axis_y_lo_text, kFieldW);
        InputNumStr("Y hi", c.axis_y_hi_text, kFieldW);
    } else {
        ImGui::TextDisabled("No variables (load a system first)");
    }
    InputNumStr("Resolution", c.n_pts_text, kFieldW);

    // Writable var (для peak finder)
    draw_writable_var_combo(s.vars, c.writable_var, "Writable var##bas_wv");

    ImGui::Separator();

    // Integration (collapsible, swapped above Features)
    {
        IntegrationFields f;
        f.h           = &c.h_text;
        f.symmetry_s  = &c.symmetry_s;
        f.t_max       = &c.t_max_text;
        f.transient   = &c.transient_text;
        f.pre_scaller = &c.pre_scaller_text;
        f.max_value   = &c.max_value_text;
        draw_integration_block("Integration", c.scheme, s.custom_schemes, f);
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
                          "Values for non-axis variables.");
    draw_named_num_fields("Parameters", s.params, c.param_values);

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
        draw_regime_summary(c.result.helpful_array);
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
    auto& renderer = ui_resource<std::unique_ptr<PlotRenderer>, 7>();
    // hm_basins — на уровне файла (basins_heatmaps()), см. комментарий у объявления.
    auto& hm_avgpk = ui_resource<std::map<unsigned, std::unique_ptr<HeatmapView>>, 1001>();
    auto& hm_avgint = ui_resource<std::map<unsigned, std::unique_ptr<HeatmapView>>, 1002>();
    auto& hm_states = ui_resource<std::map<unsigned, std::unique_ptr<HeatmapView>>, 1003>();
    auto& scatter_views = ui_resource<std::map<unsigned, std::unique_ptr<Plot2DView>>, 9>();
    if (!renderer) renderer = std::make_unique<PlotRenderer>();
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
    auto get_scatter = [&scatter_views](unsigned oid) -> Plot2DView& {
        auto& slot = scatter_views[oid];
        if (!slot) {
            slot = std::make_unique<Plot2DView>();
            slot->points_mode = true;
            slot->show_legend = false;
            slot->point_size_px = 3.0f;
            slot->pad_x = false;
        }
        return *slot;
    };
    HeatmapView& hm_basins_v = get_hm(basins_heatmaps(), base_oid + 0u, /*discrete*/ true);
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
    hm_basins_v.popup_extras = basins_export_extras;
    hm_avgpk_v.popup_extras  = basins_export_extras;
    hm_avgint_v.popup_extras = basins_export_extras;
    hm_states_v.popup_extras = basins_export_extras;
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
            draw_renumber(/*same_line*/ false);   // Scatter-таб: только Renumber
        }
    }
    if (c.renumber_spiral) ensure_basins_spiral_cache(c);

    int n = c.result.n_pts;
    size_t total = (size_t)n * (size_t)n;
    double xlo = c.result.axis_x_lo, xhi = c.result.axis_x_hi;
    double ylo = c.result.axis_y_lo, yhi = c.result.axis_y_hi;

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
        hm_basins_v.render(*renderer, origin, avail,
                          /*owner_id*/ base_oid + 0u, hm_basins_gen,
                          n, n, buf.data(),
                          xlo, xhi, ylo, yhi,
                          vmin, vmax, fit);
    }
    else if (c.active_plot_tab == 1) {
        hm_avgpk_v.x_axis.name = ax_x;
        hm_avgpk_v.y_axis.name = ax_y;
        hm_avgpk_v.render(*renderer, origin, avail,
                         /*owner_id*/ base_oid + 1u, c.data_generation,
                         n, n, c.result.avg_peaks.data(),
                         xlo, xhi, ylo, yhi,
                         c.result.avg_peaks_min, c.result.avg_peaks_max, fit);
    }
    else if (c.active_plot_tab == 2) {
        hm_avgint_v.x_axis.name = ax_x;
        hm_avgint_v.y_axis.name = ax_y;
        hm_avgint_v.render(*renderer, origin, avail,
                          /*owner_id*/ base_oid + 2u, c.data_generation,
                          n, n, c.result.avg_intervals.data(),
                          xlo, xhi, ylo, yhi,
                          c.result.avg_intervals_min, c.result.avg_intervals_max, fit);
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
        hm_states_v.render(*renderer, origin, avail,
                          /*owner_id*/ base_oid + 3u, c.data_generation,
                          n, n, buf.data(),
                          xlo, xhi, ylo, yhi,
                          -1.0, 1.0, fit);
        // Подсказка под плотом — числовые уровни, цвет зависит от выбранной colormap.
        ImGui::TextDisabled("Levels: -1 = FixedPoint, 0 = Unbound, 1 = Oscillation");
    }
    else if (c.active_plot_tab == 4) {
        // Scatter (avgPeak, avgInterval), точки сгруппированы по basin_idx.
        // Каждый кластер — своя серия (PlotSeriesInput с собственным цветом).
        // С renumber_spiral берём перенумерованные id'ы, чтобы цвета и подписи
        // ("c1", "c2", ...) совпадали с тем, что показывает Basins heatmap.
        const int* src_idx = c.renumber_spiral
                                 ? c.basin_idx_spiral.data()
                                 : c.result.basin_idx.data();
        int min_id = c.renumber_spiral ? c.min_cluster_idx_spiral : c.result.min_cluster_idx;

        // Сгруппируем точки по basin_idx.
        std::map<int, std::vector<float>> bufs;
        int valid_pts = 0;
        for (size_t k = 0; k < total; ++k) {
            int id = src_idx[k];
            double xp = c.result.avg_peaks[k];
            double yp = c.result.avg_intervals[k];
            if (!std::isfinite(xp) || !std::isfinite(yp)) continue;
            if (xp ==  kSentinelNoData || xp == -kSentinelNoData ||
                yp ==  kSentinelNoData || yp == -kSentinelNoData) continue;
            bufs[id].push_back((float)xp);
            bufs[id].push_back((float)yp);
            ++valid_pts;
        }
        if (valid_pts == 0) {
            ImGui::TextDisabled("No valid (avgPeak, avgInterval) points.");
            return;
        }

        // Буферы локальные: render() ниже забирает точки синхронно, дольше
        // вызова указатели не нужны (static здесь ещё и делил бы память между
        // config'ами бассейнов).
        std::vector<std::vector<float>> series_buffers;
        std::vector<std::string>        series_labels;
        series_buffers.reserve(bufs.size());
        series_labels.reserve(bufs.size());

        std::vector<PlotSeriesInput> series_in;
        std::vector<bool> init_vis, glob_vis;
        for (auto& kv : bufs) {
            int id = kv.first;
            series_buffers.push_back(std::move(kv.second));
            series_labels.push_back("c" + std::to_string(id));
            // Цвет per-cluster через ic_base_color (golden-ratio hash). Сдвиг
            // на (id - min_id) — чтобы FP-кластеры (отрицательные) и Osc
            // (положительные) тоже разделялись.
            PlotSeriesInput si;
            si.points   = series_buffers.back().empty() ? nullptr : series_buffers.back().data();
            si.n_points = (int)(series_buffers.back().size() / 2);
            si.color    = ic_base_color(id - min_id);
            si.label    = series_labels.back();
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


// Fast Synchro Controls + Plot — recurrent synchronization (anti-sync).
// Mode 0 = On Attractor (trajectory + per-point error); Mode 1 = On Grid.
void draw_fastsync_controls(AppModel& model, SystemLibrary& lib) {
    FastSyncAnalysisSession& s = model.fastsync_session;

    // Source of truth для custom-схем — AppModel (редактируются в Library).
    // load_from_record() сохраняет снапшот ОДНОКРАТНО при входе в режим, поэтому
    // схемы, добавленные позже, не видны без рефреша. Подтягиваем актуальный
    // список каждый кадр, чтобы Combo и compute_krs_for_scheme работали с live.
    s.custom_schemes = model.custom_schemes;

    ImGui::Text("Fast Synchro");
    ImGui::TextDisabled("Recurrent synchronization analysis (anti-sync error).");

    // Run / Cancel / Run all... — сразу под пикером системы, над таб-баром: эти кнопки
    // управляют активным конфигом и должны быть доступны без прокрутки.
    // "Run all..." кладёт выбранные индексы в model.fastsync_queue; draw_gui тикает
    // очередь после poll'ов.
    {
        const bool no_cfg = s.configs.empty();
        RunAllGroup g;
        g.pick_id = "pfs_";
        g.n       = (int)s.configs.size();
        g.label   = [&s](int i) { return s.configs[i].label; };
        g.enqueue = [&model](int i) { model.fastsync_queue.push_back({i}); };
        draw_run_and_run_all("##run_all_fastsync", s.in_flight, no_cfg,
                             s.in_flight || no_cfg, 160.0f,
                             [&model, &s]() {
                                 if (!model.parametric_engine)
                                     model.parametric_engine = std::make_unique<ParametricEngine>();
                                 s.run_async(*model.parametric_engine, s.active_config_index);
                             },
                             // Cancel — только у FastSync, между Run и Run all.
                             [&s]() {
                                 ImGui::SameLine();
                                 if (s.in_flight && ImGui::Button("Cancel")) s.request_cancel();
                             },
                             { g },
                             [&model]() { model.start_next_in_fastsync_queue(); },
                             model.fastsync_queue.size());
    }
    ImGui::Separator();

    // Tab bar для config'ов.
    const TabBarResult tabs = draw_config_tab_bar(
        "##fs_tabs", "fs_tab_", (int)s.configs.size(),
        s.in_flight, s.running_config_index,
        [&s](int i) { return s.configs[i].label; },
        {},
        [&s]() { s.add_config(); });
    if (tabs.active    >= 0) s.active_config_index = tabs.active;
    if (tabs.to_remove >= 0) model.remove_fastsync_config(tabs.to_remove);

    if (s.configs.empty()) {
        ImGui::TextDisabled("No FastSync configs. Press '+' to add one.");
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
        s.active_config_index = 0;
    FastSyncConfig& c = s.configs[s.active_config_index];

    // Label rename.
    draw_label_rename("Label##fs_label", c.label);
    ImGui::Separator();

    // Mode
    ImGui::Text("Mode:");
    ImGui::RadioButton("On Attractor", &c.mode, 0); ImGui::SameLine();
    ImGui::RadioButton("On Grid",      &c.mode, 1);
    ImGui::Separator();

    // Scheme
    draw_scheme_combo("Scheme", c.scheme, s.custom_schemes, {}, &s.enabled_builtin_schemes);
    draw_adaptive_controls(c.adaptive);
    if (scheme_uses_symmetry(c.scheme, s.custom_schemes))
        InputNumStr("symmetry s", c.symmetry_s, kFieldW);
    ImGui::Separator();

    // Mode-specific axes
    if (c.mode == 1 && !s.vars.empty()) {
        if (c.axis_x_var < 0 || c.axis_x_var >= (int)s.vars.size()) c.axis_x_var = 0;
        if (c.axis_y_var < 0 || c.axis_y_var >= (int)s.vars.size())
            c.axis_y_var = (s.vars.size() > 1) ? 1 : 0;
        const std::vector<const char*> items = c_str_list(s.vars);

        const char* axis_role_x = c.grid_swap_master_slave ? "Axis X (slave IC)" : "Axis X (master IC)";
        const char* axis_role_y = c.grid_swap_master_slave ? "Axis Y (slave IC)" : "Axis Y (master IC)";
        ImGui::SetNextItemWidth(kComboW);
        ImGui::Combo(axis_role_x, &c.axis_x_var, items.data(), (int)items.size());
        InputNumStr("X lo", c.axis_x_lo_text, kFieldW);
        InputNumStr("X hi", c.axis_x_hi_text, kFieldW);
        ImGui::SetNextItemWidth(kComboW);
        ImGui::Combo(axis_role_y, &c.axis_y_var, items.data(), (int)items.size());
        InputNumStr("Y lo", c.axis_y_lo_text, kFieldW);
        InputNumStr("Y hi", c.axis_y_hi_text, kFieldW);
        InputNumStr("Resolution", c.n_pts_text, kFieldW);
        // Какую сторону перебирать по сетке: master IC (legacy default) или slave IC.
        ImGui::Checkbox("Vary slave IC (master fixed)##fs_gridswap", &c.grid_swap_master_slave);
        ImGui::TextDisabled("Off: grid sweeps master IC, slave fixed. On: grid sweeps slave IC, master fixed.");
        ImGui::Separator();
    }
    else if (c.mode == 0 && !s.vars.empty()) {
        // На траектории — какие 2 переменные показывать как X/Y фазового портрета.
        const std::vector<const char*> items = c_str_list(s.vars);
        ImGui::SetNextItemWidth(kComboW);
        ImGui::Combo("Display X var", &c.axis_x_var, items.data(), (int)items.size());
        ImGui::SetNextItemWidth(kComboW);
        ImGui::Combo("Display Y var", &c.axis_y_var, items.data(), (int)items.size());
        ImGui::Separator();
    }

    // Integration (collapsible)
    if (ImGui::CollapsingHeader("Integration", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("h",                c.h_text, kFieldW);
        if (c.mode == 0) {
            InputNumStr("t_max",        c.t_max_text, kFieldW);
        }
        if (c.mode == 1) {
            // On Grid: у каждой системы свой транзиент. Свипуемая сеткой
            // сторона отрабатывает его в КАЖДОЙ ячейке (из её затравки),
            // фиксированная — из своей единственной точки.
            // В random-режиме ведомая сторона стартует ОТ точки свипуемой и
            // своего транзиента не отрабатывает — гасим его поле, чтобы не
            // выглядело действующим.
            const bool master_swept  = !c.grid_swap_master_slave;
            const bool master_derived = c.ic_random_offset && !master_swept;
            const bool slave_derived  = c.ic_random_offset &&  master_swept;
            ImGui::BeginDisabled(master_derived);
            InputNumStr("transient master", c.transient_text, kFieldW);
            ImGui::EndDisabled();
            ImGui::BeginDisabled(slave_derived);
            InputNumStr("transient slave",  c.transient_slave_text, kFieldW);
            ImGui::EndDisabled();
            if (c.ic_random_offset)
                ImGui::TextDisabled(master_swept
                    ? "Master settles per grid cell; slave starts from it (random IC)."
                    : "Slave settles per grid cell; master starts from it (random IC).");
            else
                ImGui::TextDisabled(c.grid_swap_master_slave
                    ? "Slave settles per grid cell, master from its fixed IC."
                    : "Master settles per grid cell, slave from its fixed IC.");
        }
        else {
            InputNumStr("transient",    c.transient_text, kFieldW);
        }
        bool window_changed = InputNumStr("window",           c.window_text, kFieldW);
        bool iter_changed   = InputNumStr("iter of synch",    c.iter_of_synchr_text, kFieldW);
        if (!window_changed && !iter_changed) {
            // window/iter — источник истины в этом кадре (или ничего не
            // менялось, напр. только что загрузили сессию): пересчитываем
            // total sync time ДО отрисовки его поля, без задержки в кадр.
            double window_v = parse_ratio_or(c.window_text, 0.0);
            double iter_v   = std::max(1.0, parse_ratio_or(c.iter_of_synchr_text, 0.0));
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.10g", (2.0 * iter_v - 1.0) * window_v);
            c.total_time_text = buf;
        }
        if (InputNumStr("total sync time", c.total_time_text, kFieldW)) {
            // Пользователь правит total → пересчитываем window (iter фиксирован).
            double total_v = parse_ratio_or(c.total_time_text, 0.0);
            double iter_v  = std::max(1.0, parse_ratio_or(c.iter_of_synchr_text, 0.0));
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.10g", total_v / (2.0 * iter_v - 1.0));
            c.window_text = buf;
        }
        if (c.mode == 0) {
            InputNumStr("decimator",    c.pre_scaller_text, kFieldW);
        }
        InputNumStr("max value",        c.max_value_text, kFieldW);
    }

    // Synchro runtime (collapsible)
    if (ImGui::CollapsingHeader("Synchro runtime", ImGuiTreeNodeFlags_DefaultOpen)) {
        static const char* tos_names[] = { "Unidirectional", "Bidirectional" };
        ImGui::SetNextItemWidth(kComboW);
        ImGui::Combo("Type of synch.", &c.type_of_synch, tos_names, IM_ARRAYSIZE(tos_names));
        // Формулировки правлены по факту кода (cudaLibrary.cu): и 0, и 2 —
        // евклидовы нормы по переменным, БЕЗ деления на их количество.
        // 0 усредняет по времени (RMS вдоль окна), 2 берёт одну точку.
        static const char* ee_names[] = {
            "0: RMS of ||e|| over last window",
            "1: # iters to reach FS_error_trs",
            "2: ||e|| at last point",
            "3: time to reach FS_error_trs",
            "4: err_stop / err_start",
            "5: log10(err_stop / err_start)",
            "6: log10(err_stop / err_start) / iter",
            "7: log10(rho) per cycle (Gram-Schmidt)"
        };
        ImGui::SetNextItemWidth(280);
        ImGui::Combo("Error estim.", &c.error_estim, ee_names, IM_ARRAYSIZE(ee_names));
        if (c.error_estim >= 4 && c.error_estim <= 6)
            ImGui::TextDisabled("err_start/err_stop: ||e|| at the window start before / after all\n"
                                "forward-backward passes. Cells where err_start == 0 come out\n"
                                "non-finite and render as diverged.\n"
                                "6 divides by iter of synch: log10 grows as iter*log10(rho),\n"
                                "so only the per-cycle rate compares with the linearised value.");
        if (c.error_estim == 7) {
            ImGui::TextDisabled("Benettin over the same forward-backward cycle: AMOUNTOFX clones\n"
                                "at master +- eps, Gram-Schmidt and renormalisation every cycle.\n"
                                "Gives log10(rho) per cycle directly - no speckle at low iter,\n"
                                "no saturation, nothing to divide afterwards.");
            InputNumStr("GS warmup cycles", c.gs_warmup_text, kFieldW);
            ImGui::TextDisabled("Cycles dropped from the average while the basis aligns.\n"
                                "Without it the start-basis bias only washes out as O(1/m),\n"
                                "which is why convergence took ~100 cycles. 0 = old behaviour.");
            if (c.type_of_synch != 0)
                ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.3f, 1.0f),
                                   "Estimator 7 needs Unidirectional synchro; every cell comes out as diverged.");
        }
        InputNumStr("FS error trs.", c.fs_error_trs_text, kFieldW);
    }

    // Parameters (collapsible, placed under Synchro runtime)
    draw_named_num_fields("Parameters", s.params, c.param_values);

    // Парные сворачиваемые секции: клик по любому заголовку сворачивает обе половины.
    // Общее состояние навязывается каждому заголовку через SetNextItemOpen каждый кадр,
    // IsItemToggledOpen ловит клик и переключает его. Переменные рисуются построчно,
    // чтобы x_master вставал напротив x_slave (то же для K forward/backward).
    auto paired_header = [](const char* label, bool& open) {
        ImGui::SetNextItemOpen(open, ImGuiCond_Always);
        ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_None);
        if (ImGui::IsItemToggledOpen()) open = !open;
    };
    auto paired_input = [&](const char* var,
                            std::map<std::string, std::string>& m,
                            const char* id_prefix) {
        std::string pid = std::string(id_prefix) + var;
        ImGui::PushID(pid.c_str());
        InputNumStr(var, m[var], 100);
        ImGui::PopID();
    };

    // Master init | Slave init.
    // В режиме случайных НУ ведущей (свипуемой) остаётся одна сторона, вторая
    // получает её точку ± eps — её поля гасим, чтобы не читались как рабочие.
    // On Attractor ведущий всегда мастер; On Grid — та сторона, которую
    // перебирает сетка (см. grid_swap_master_slave).
    const bool fs_master_leads   = (c.mode == 0) || !c.grid_swap_master_slave;
    const bool fs_master_derived = c.ic_random_offset && !fs_master_leads;
    const bool fs_slave_derived  = c.ic_random_offset &&  fs_master_leads;
    static bool ic_open = true;
    if (ImGui::BeginTable("##fs_ic_table", 2)) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        paired_header("Master init##fs_master_ic", ic_open);
        ImGui::TableSetColumnIndex(1);
        paired_header("Slave init##fs_slave_ic", ic_open);
        if (ic_open) {
            for (const auto& v : s.vars) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::BeginDisabled(fs_master_derived);
                paired_input(v.c_str(), c.ic_master, "icm_");
                ImGui::EndDisabled();
                ImGui::TableSetColumnIndex(1);
                ImGui::BeginDisabled(fs_slave_derived);
                paired_input(v.c_str(), c.ic_slave,  "ics_");
                ImGui::EndDisabled();
            }
        }
        ImGui::EndTable();
    }

    // Переключатель НУ второй системы: фиксированные (как раньше) либо
    // случайный отступ в eps-окрестности точки ведущей системы.
    ImGui::Checkbox("Random IC (eps-offset)##fs_ic_random", &c.ic_random_offset);
    if (c.ic_random_offset) {
        InputNumStr("eps",  c.ic_eps_text,  kFieldW);
        InputNumStr("seed", c.ic_seed_text, kFieldW);
        // Оценщик 7 сам расставляет клонов и НУ слейва не читает — про eps ему
        // всё сказано выше, здесь бы вышла неправда про slave.
        if (c.error_estim == 7)
            ImGui::TextDisabled("Estimator 7 ignores this: it perturbs its own clones by eps, seed unused.");
        else if (c.mode == 0)
            ImGui::TextDisabled("Slave starts at the master point of its own window, +-eps per coordinate.");
        else
            ImGui::TextDisabled(fs_master_leads
                ? "Slave starts at the master cell point (after its transient), +-eps per coordinate."
                : "Master starts at the slave cell point (after its transient), +-eps per coordinate.");
        if (c.error_estim != 7)
            ImGui::TextDisabled("Same seed reproduces the run bit-for-bit.");
    } else if (c.error_estim == 7) {
        // Оценщик 7 сам расставляет клонов вокруг мастера, seed ему не нужен
        // (базис единичный, не случайный), но величину возмущения берёт отсюда.
        InputNumStr("eps", c.ic_eps_text, kFieldW);
        ImGui::TextDisabled("Estimator 7 starts its clones at master +-eps; the slave IC above is unused.");
    } else {
        ImGui::TextDisabled("Off: the second system uses the fixed IC above (legacy behaviour).");
    }

    // K forward (h>0) | K backward (h<0).
    static bool k_open = true;
    if (ImGui::BeginTable("##fs_k_table", 2)) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        paired_header("K forward (h>0)##fs_kf",  k_open);
        ImGui::TableSetColumnIndex(1);
        paired_header("K backward (h<0)##fs_kb", k_open);
        if (k_open) {
            for (const auto& v : s.vars) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                paired_input(v.c_str(), c.k_forward,  "kf_");
                ImGui::TableSetColumnIndex(1);
                paired_input(v.c_str(), c.k_backward, "kb_");
            }
        }
        ImGui::EndTable();
    }

    // CSV output (collapsible, moved to the bottom)
    draw_csv_output_block("CSV output", "Save to file##fs_csv", c.csv_save_enabled,
                          "##fs_csv_path", c.csv_output_path,
                          c.mode == 0
                              ? "Writes one row per trajectory point: x[0],..,x[N-1],sync_error."
                              : "Writes 2 header lines (X/Y ranges) + n_pts x n_pts error matrix (row-major).");

    if (c.last_run_ok) {
        if (c.mode == 0)
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: traj %d pts, sync_err [%.4g, %.4g]",
                c.result.n_pts_traj, c.result.min_val, c.result.max_val);
        else {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: grid %dx%d, sync_err [%.4g, %.4g]",
                c.result.n_pts_grid, c.result.n_pts_grid,
                c.result.min_val, c.result.max_val);
            // Разлетевшиеся ячейки несут NaN и рисуются тёмно-серым; без счётчика
            // сплошь серая карта неотличима от "расчёт не пошёл".
            if (c.result.diverged_cells > 0)
                ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f),
                    "diverged: %d of %d cells (maxValue / nan-inf)",
                    c.result.diverged_cells, c.result.total_cells);
        }
    } else if (!c.last_error.empty()) {
        draw_error_box("##fs_err", c.last_error, /*lines*/ 10);
    }
}
}
