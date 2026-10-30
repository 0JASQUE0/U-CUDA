#include "gui_internal.h"

// Fast Synchro Controls + Plot — recurrent synchronization (anti-sync).
// Mode 0 = On Attractor (trajectory + per-point error); Mode 1 = On Grid.
void draw_fastsync_controls(AppModel& model, SystemLibrary& lib) {
    FastSyncAnalysisSession& s = model.fastsync_session;
    // Чей конфиг правит эта панель — для ПКМ-рассылки "во все такие же".
    model.broadcast_source_tab = BroadcastTab::FastSync;
    model.broadcast_source_idx = s.active_config_index;

    // Source of truth для custom-схем — AppModel (редактируются в Library).
    // load_from_record() сохраняет снапшот ОДНОКРАТНО при входе в режим, поэтому
    // схемы, добавленные позже, не видны без рефреша. Подтягиваем актуальный
    // список каждый кадр, чтобы Combo и compute_krs_for_scheme работали с live.
    s.custom_schemes = model.custom_schemes;
    s.wrapper_schemes   = model.enabled_wrapper_schemes();

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
    draw_scheme_combo("Scheme", c.scheme, s.custom_schemes, {}, &s.enabled_builtin_schemes,
                      &model, model.is_map, &s.wrapper_schemes);
    if (scheme_uses_symmetry(c.scheme, s.custom_schemes))
        InputNumStr("symmetry s", c.symmetry_s, kFieldW);
    if (scheme_uses_cd_fp_iters(c.scheme)) {
        InputNumStr("FP iterations n", c.cd_fp_iters, kFieldW);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", kCdFpItersTip);
    }
    if (scheme_uses_newton(c.scheme))
        draw_newton_fields(c.newton, kFieldW, "fs_newton");
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
        InputNumStr("Resolution", c.n_pts_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::Resolution, {},
                                        "Resolution", c.n_pts_text); });
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
        // 3D-портрет — только когда есть третья переменная.
        const bool can_3d = s.vars.size() >= 3;
        if (!can_3d) c.plot_3d = false;
        ImGui::BeginDisabled(!can_3d);
        ImGui::Checkbox("3D plot##fs_plot3d", &c.plot_3d);
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip(can_3d ? "Phase portrait in 3D, coloured by the sync error."
                                     : "Needs at least three state variables.");
        if (c.plot_3d) {
            if (c.axis_z_var < 0 || c.axis_z_var >= (int)s.vars.size()) c.axis_z_var = 2;
            ImGui::SetNextItemWidth(kComboW);
            ImGui::Combo("Display Z var", &c.axis_z_var, items.data(), (int)items.size());
        }
        ImGui::Separator();
    }

    // Integration (collapsible)
    if (ImGui::CollapsingHeader("Integration", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!model.is_map) InputNumStr("h",  c.h_text, kFieldW);
        if (c.mode == 0) {
            InputNumStr(model.is_map ? "iterations" : "t_max", c.t_max_text, kFieldW);
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
    draw_named_num_fields("Parameters", s.params, c.param_values,
                          nullptr, &model, BroadcastField::Param);

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

// Plot Fast Synchro: либо colored trajectory (mode 0), либо heatmap (mode 1).
void draw_fastsync_plot(AppModel& model, const GuiCallbacks& cb) {
    FastSyncAnalysisSession& s = model.fastsync_session;
    if (s.configs.empty()) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
        s.active_config_index = 0;
    FastSyncConfig& c = s.configs[s.active_config_index];

    const unsigned base_oid = 0x5F50000u + (unsigned)s.active_config_index;
    static std::unique_ptr<PlotRenderer> renderer;
    static std::map<unsigned, std::unique_ptr<HeatmapView>> hm_map;
    static std::map<unsigned, std::unique_ptr<Plot2DView>>  traj_map;
    if (!renderer) renderer = std::make_unique<PlotRenderer>();
    static int seen_epoch = 0;
    if (session_reloaded(model, AppModel::AppMode::FastSync, seen_epoch)) {
        invalidate_views(hm_map);
        invalidate_views(traj_map);
    }

    // Visualization toolbar — общая реализация (draw_heatmap_toolbar). Цветовая шкала
    // FastSync держится в HeatmapView даже в mode 0 (colored trajectory, где хитмапы нет),
    // чтобы тулбар совпадал с остальными вкладками до последнего виджета; персистентность
    // при этом в config'е (он уходит в сессию), поэтому синхронизируем в обе стороны.
    auto& hv_slot = hm_map[base_oid];
    if (!hv_slot) hv_slot = std::make_unique<HeatmapView>();
    HeatmapView& hv = *hv_slot;

    c.colormap_idx = colormap_id_or(c.colormap_idx, kColormapTurbo);
    hv.colormap         = (HeatmapColormap)c.colormap_idx;   // config → view
    hv.autoscale        = c.autoscale_color;
    hv.manual_vmin_text = c.c_min_text;
    hv.manual_vmax_text = c.c_max_text;
    hv.swap_axes        = c.swap_axes;
    {
        HeatmapToolbarOpts topts;
        topts.persist_colormap = [&](int cm) { c.colormap_idx = cm; };
        // Swap axes осмыслен только в grid-режиме (там оси — две IC).
        topts.show_swap = (c.mode != 0);
        if (c.mode == 0) {
            // Толщина/α для colored trajectory. Выбор Display X/Y живёт в
            // Controls-панели рядом с прочими compute-параметрами.
            topts.extras_tail = [&]() {
                ImGui::SameLine(); ImGui::SetNextItemWidth(120);
                ImGui::SliderFloat("Line width", &c.line_width, 0.1f, 5.0f, "%.2f");
                ImGui::SameLine(); ImGui::SetNextItemWidth(120);
                ImGui::SliderFloat("Alpha", &c.alpha, 0.0f, 1.0f, "%.2f");
            };
        }
        draw_heatmap_toolbar(hv, topts);
    }
    c.autoscale_color = hv.autoscale;              // view → config
    c.c_min_text      = hv.manual_vmin_text;
    c.c_max_text      = hv.manual_vmax_text;
    c.swap_axes       = hv.swap_axes;

    if (!c.last_run_ok) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }
    // Результат посчитан в другом режиме (переключили On Attractor <-> On Grid после Run):
    // рисовать нечего — траектории у сетки нет, карты у траектории нет. Раньше сетка
    // рисовалась пустой картой по старой траектории, а траектория (2D/3D) — «No trajectory data».
    if (c.result.mode != c.mode) {
        ImGui::TextDisabled("No data for this mode yet. Press Run.");
        return;
    }

    bool fit = c.fit_request;
    if (fit) c.fit_request = false;
    ImVec2 avail  = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();

    auto var_name = [&](int idx) -> std::string {
        if (idx >= 0 && idx < (int)s.vars.size()) return s.vars[idx];
        return std::string("x");
    };

    // cmin/cmax: при autoscale берём диапазон актуальных значений из result. Введённый
    // вручную vmin > vmax — это сигнал «перевернуть колормапу», а не ошибка: сортируем
    // диапазон и взводим invert_cmap, чтобы mode-0 (colored trajectory) отзеркалил
    // соответствие value→цвет. Heatmap-режим берёт user-typed как раньше — HeatmapView
    // сам защищается от обратного диапазона.
    double cmin_user, cmax_user;
    if (c.autoscale_color) {
        cmin_user = c.result.min_val;
        cmax_user = c.result.max_val;
        if (!(cmax_user > cmin_user)) cmax_user = cmin_user + 1.0;
    } else {
        cmin_user = parse_ratio_or(c.c_min_text, -12.0);
        cmax_user = parse_ratio_or(c.c_max_text,   0.0);
    }
    bool invert_cmap = (!c.autoscale_color) && (cmin_user > cmax_user);
    double cmin = std::min(cmin_user, cmax_user);
    double cmax = std::max(cmin_user, cmax_user);
    if (!(cmax > cmin)) cmax = cmin + 1.0;
    HeatmapColormap cmap = (HeatmapColormap)colormap_id_or(c.colormap_idx, kColormapTurbo);

    if (c.mode == 0) {
        // Colored trajectory + manual colorbar справа.
        const int nX = c.result.amountOfX_traj;
        if (nX <= 0 || c.result.n_pts_traj <= 0) {
            ImGui::TextDisabled("No trajectory data.");
            return;
        }
        int vx = (c.axis_x_var >= 0 && c.axis_x_var < nX) ? c.axis_x_var : 0;
        int vy = (c.axis_y_var >= 0 && c.axis_y_var < nX) ? c.axis_y_var : (nX > 1 ? 1 : 0);

        // Резервируем место справа под colorbar. Тики и ширина считаются теми
        // же общими хелперами, что у HeatmapView — раньше здесь была своя
        // копия с 5 равноотстоящими тиками и продублированными константами.
        const std::vector<ColorbarTick> cb_ticks =
            colorbar_ticks((float)cmin, (float)cmax, /*n_discrete*/ 0);
        const float cb_total = colorbar_total_width(cb_ticks);

        ImVec2 plot_avail(std::max(64.0f, avail.x - cb_total), avail.y);
        // Колорбар справа от плота — в «Copy image» он должен попасть вместе с ним.
        const float shot_extra = std::max(0.0f, avail.x - plot_avail.x);

        const bool fs_busy = s.in_flight &&
                             s.active_config_index == s.running_config_index;

        if (c.plot_3d && nX >= 3) {
            // 3D: та же окраска по ошибке синхронизации, но в GL (draw_line_3d_cmap): цвет
            // интерполируется вдоль сегмента, глубина — depth test, а не сортировка сегментов.
            static std::map<unsigned, std::unique_ptr<Plot3DView>> traj3d_map;
            static int seen_epoch_3d = 0;
            if (session_reloaded(model, AppModel::AppMode::FastSync, seen_epoch_3d))
                invalidate_views(traj3d_map);
            int vz =(c.axis_z_var >= 0 && c.axis_z_var < nX) ? c.axis_z_var : 2;
            auto& slot3 = traj3d_map[base_oid];
            if (!slot3) {
                slot3 = std::make_unique<Plot3DView>();
                slot3->show_legend = false;
            }
            Plot3DView& v3 = *slot3;
            v3.x_name = var_name(vx);
            v3.y_name = var_name(vy);
            v3.z_name = var_name(vz);
            v3.line_thickness_px = c.line_width;
            v3.screenshot_extra_right = shot_extra;

            // Смена осей — новый bbox: перезаливка (токен) и автоподгонка.
            static std::map<unsigned, int> last_axes3;
            const int axes_key = (vx * 64 + vy) * 64 + vz;
            auto it3 = last_axes3.find(base_oid);
            if (it3 == last_axes3.end() || it3->second != axes_key) {
                v3.view_valid = false;
                last_axes3[base_oid] = axes_key;
            }

            const int n_in = c.result.n_pts_traj;
            static std::vector<float> xyz_buf;
            static std::vector<float> val_buf;
            xyz_buf.resize((size_t)n_in * 3);
            val_buf.resize((size_t)n_in);
            for (int i = 0; i < n_in; ++i) {
                const double* row = &c.result.traj_full[(size_t)i * nX];
                xyz_buf[3*i + 0] = (float)row[vx];
                xyz_buf[3*i + 1] = (float)row[vy];
                xyz_buf[3*i + 2] = (float)row[vz];
                val_buf[i] = (float)c.result.sync_error[i];
            }

            std::vector<PlotSeriesInput3D> s3(1);
            s3[0].points       = xyz_buf.data();
            s3[0].n_points     = n_in;
            s3[0].color        = ImVec4(1, 1, 1, c.alpha);
            s3[0].label        = "trajectory";
            s3[0].values       = val_buf.data();
            s3[0].colormap     = (int)cmap;
            s3[0].cmin         = (float)cmin;
            s3[0].cmax         = (float)cmax;
            s3[0].cmap_reverse = invert_cmap;   // значения сырые, шкала переворачивается в шейдере
            std::vector<bool> vis(1, true);
            v3.popup_extras = [&c, &cb, fs_busy]() {
                draw_export_menu_item(fs_busy, cb, [&c](const std::string& p) {
                    data_export::export_fastsync(c.result, p);
                });
            };
            v3.render(*renderer, origin, plot_avail, (int)base_oid,
                      c.data_generation * 1000000 + axes_key, s3, vis, vis, fit);
            v3.popup_extras = nullptr;

            draw_colorbar(ImGui::GetWindowDrawList(),
                          ImVec2(v3.last_img_pos.x + v3.last_img_size.x + kColorbarGap,
                                 v3.last_img_pos.y),
                          v3.last_img_size.y, (float)cmin, (float)cmax, cmap,
                          /*reverse*/ invert_cmap, /*n_discrete*/ 0, cb_ticks);
            return;
        }

        auto& slot = traj_map[base_oid];
        if (!slot) {
            slot = std::make_unique<Plot2DView>();
            slot->points_mode  = false;
            slot->show_legend  = false;
            slot->imdraw_lines = true;
            slot->pad_x = true; slot->pad_y = true;
            // Это фазовый портрет (оси — переменные состояния), поэтому линий
            // x=0 / y=0 быть не должно — как в Phase 2D. Раньше здесь стоял
            // дефолт Plot2DView (true), и одна и та же по смыслу диаграмма
            // рисовалась в двух вкладках по-разному.
            slot->show_zero_x = false;
            slot->show_zero_y = false;
        }
        Plot2DView& v = *slot;
        v.x_axis.name = var_name(vx);
        v.y_axis.name = var_name(vy);
        v.line_thickness_px = c.line_width;
        v.screenshot_extra_right = shot_extra;

        // Если поменялись axes — форсим (a) fit, чтобы view нашёл новый bbox;
        // (b) re-upload GPU-кэша точек, иначе series_cache_.bbox() даст старый
        // диапазон. Поэтому передаём генерационный токен, зависящий от (vx,vy).
        static std::map<unsigned, std::pair<int,int>> last_axes;
        auto it_ax = last_axes.find(base_oid);
        if (it_ax == last_axes.end() || it_ax->second.first != vx || it_ax->second.second != vy) {
            v.view_valid = false;     // force autofit
            last_axes[base_oid] = { vx, vy };
        }

        // Пересобираем XY/values из полного буфера (без decimator'а — рисуем
        // все точки, ImDrawList сегмент-за-сегментом справляется).
        int n_in = c.result.n_pts_traj;
        static std::vector<double> xy_buf;
        static std::vector<float> err_buf;
        xy_buf.resize((size_t)n_in * 2);
        err_buf.resize((size_t)n_in);
        for (int i = 0; i < n_in; ++i) {
            xy_buf[2*i + 0] = (float)c.result.traj_full[(size_t)i * nX + (size_t)vx];
            xy_buf[2*i + 1] = (float)c.result.traj_full[(size_t)i * nX + (size_t)vy];
            // invert_cmap: reflect v across the sorted [cmin, cmax] midpoint.
            // Renderer computes t = (v - cmin)/(cmax - cmin); the reflection
            // maps it to 1 - t → cmap_sample sees the colors in reverse order.
            double v_err = c.result.sync_error[i];
            err_buf[i] = (float)(invert_cmap ? (cmin + cmax - v_err) : v_err);
        }

        // Painter's algorithm: сортируем сегменты по средней координате оси Z (первая var,
        // не равная vx/vy) — дальние рисуются первыми, ближние поверх. Без сортировки
        // последний по времени сегмент перекрывает более ранние независимо от их Z.
        int vz = -1;
        for (int k = 0; k < nX; ++k) if (k != vx && k != vy) { vz = k; break; }
        static std::vector<int>   seg_order;
        static std::vector<float> seg_z;
        const int n_seg = n_in - 1;
        seg_order.clear();
        if (vz >= 0 && n_seg > 0) {
            seg_order.resize(n_seg);
            seg_z.resize(n_seg);
            for (int k = 0; k < n_seg; ++k) {
                seg_order[k] = k;
                float za = (float)c.result.traj_full[(size_t)k       * nX + (size_t)vz];
                float zb = (float)c.result.traj_full[(size_t)(k + 1) * nX + (size_t)vz];
                seg_z[k] = 0.5f * (za + zb);
            }
            if (c.invert_depth) {
                std::sort(seg_order.begin(), seg_order.end(),
                          [&](int a, int b) { return seg_z[a] > seg_z[b]; });
            } else {
                std::sort(seg_order.begin(), seg_order.end(),
                          [&](int a, int b) { return seg_z[a] < seg_z[b]; });
            }
        }

        std::vector<PlotSeriesInput> series_in(1);
        series_in[0].points   = xy_buf.data();
        series_in[0].n_points = n_in;
        series_in[0].color    = ImVec4(1, 1, 1, c.alpha);  // .w → alpha-multiplier на cmap_sample()
        series_in[0].label    = "trajectory";
        series_in[0].values   = err_buf.data();
        series_in[0].colormap = cmap;
        series_in[0].cmin     = (float)cmin;
        series_in[0].cmax     = (float)cmax;
        series_in[0].segment_order = seg_order.empty() ? nullptr : seg_order.data();
        std::vector<bool> vis(1, true);
        // Synthetic generation token: меняется при смене (data_generation, vx, vy),
        // чтобы Plot2DView::series_cache_ перезалил GPU-буфер → bbox()/autofit
        // подхватили новую X/Y проекцию.
        int gen_token = c.data_generation * 1000 + vx * 10 + vy;
        // Right-click popup получает дополнительный пункт "Invert depth axis"
        // через popup_extras callback. vz/var_name захватываются по значению.
        const int   vz_capture       = vz;
        const std::string vz_name    = (vz >= 0) ? var_name(vz) : std::string{};
        v.popup_extras = [vz_capture, vz_name, &c, &cb, fs_busy]() {
            if (vz_capture >= 0) {
                std::string lbl = "Invert depth axis (" + vz_name + ")";
                ImGui::MenuItem(lbl.c_str(), nullptr, &c.invert_depth);
            } else {
                ImGui::BeginDisabled();
                ImGui::MenuItem("Invert depth axis (2D system — N/A)", nullptr, false);
                ImGui::EndDisabled();
            }
            ImGui::Separator();
            draw_export_menu_item(fs_busy, cb, [&c](const std::string& p) {
                data_export::export_fastsync(c.result, p);
            });
        };
        v.render(*renderer, origin, plot_avail,
                 /*owner_id*/ (int)base_oid, gen_token,
                 series_in, vis, vis, fit);
        v.popup_extras = nullptr; // не утечь callback в другой кадр

        // Colorbar справа — общая реализация (draw_colorbar). Марджины плота обязаны
        // совпадать с Plot2DView (он рисует внутри plot_avail): продублированные здесь
        // тремя числами, они разъезжались при любой правке лэйаута, поэтому берутся из
        // plot_2d_margins() — один источник истины.
        // Берём марджины, которыми render отработал ЭТИМ кадром: левый теперь
        // зависит от длины подписей оси Y, и пересчитать его здесь заново
        // значило бы разъехаться с плотом на кадр после каждого зума.
        const float margin_left   = v.last_margin_left;
        const float margin_top    = v.last_margin_top;
        const float margin_right  = v.last_margin_right;
        const float margin_bottom = v.last_margin_bottom;
        const float plot_w = std::max(64.0f, plot_avail.x - margin_left - margin_right);
        const float plot_h = std::max(64.0f, plot_avail.y - margin_top  - margin_bottom);
        draw_colorbar(ImGui::GetWindowDrawList(),
                      ImVec2(origin.x + margin_left + plot_w + kColorbarGap,
                             origin.y + margin_top),
                      plot_h, (float)cmin, (float)cmax, cmap,
                      /*reverse*/ invert_cmap, /*n_discrete*/ 0, cb_ticks);
    }
    else {
        // Heatmap. `hv` — тот же view, на котором выше стоял тулбар; здесь
        // только доопределяем численный диапазон (cmin/cmax уже отсортированы
        // выше, HeatmapView обратный диапазон не ждёт) и подписи осей.
        HeatmapView& h = hv;
        h.manual_vmin = (float)cmin;
        h.manual_vmax = (float)cmax;
        h.x_axis.name = var_name(c.result.axis_x_var) + "(0)";
        h.y_axis.name = var_name(c.result.axis_y_var) + "(0)";
        const bool fs_busy = s.in_flight &&
                             s.active_config_index == s.running_config_index;
        // Ctrl+T: обе оси карты — сетка начальных условий (см. Basins).
        const std::vector<ViewRangeTarget> vrt = {
            vr_target(s.configs, s.active_config_index,
                      &FastSyncConfig::axis_x_lo_text, &FastSyncConfig::axis_x_hi_text,
                      &FastSyncConfig::axis_y_lo_text, &FastSyncConfig::axis_y_hi_text) };
        h.popup_extras = [&c, &cb, fs_busy, &vrt, &model, &h]() {
            draw_export_menu_item(fs_busy, cb, [&c](const std::string& p) {
                data_export::export_fastsync(c.result, p);
            });
            draw_view_range_menu(vrt, h.x_axis, h.y_axis, h.swap_axes, &model);
        };
        h.render(*renderer, origin, avail,
                 /*owner_id*/ (int)base_oid, c.data_generation,
                 c.result.n_pts_grid, c.result.n_pts_grid,
                 c.result.heatmap.data(),
                 c.result.axis_x_lo, c.result.axis_x_hi,
                 c.result.axis_y_lo, c.result.axis_y_hi,
                 c.result.min_val, c.result.max_val,
                 fit);
        handle_view_range_keys(vrt, h.x_axis, h.y_axis, plot_window_active(),
                               h.swap_axes, &model);
    }
}

// Parametric Controls dispatcher — верхние табы Bif / LLE / LS.
// ===========================================================================
// Order — оценка порядка точности схемы.
//
// Расчёт: kernels/order.template.cu, три решения шагами h, h/2, h/4 и
// p = log2(E1/E2) по Эйткену/Ричардсону (см. order_session.h).
// Три диаграммы вкладки различаются только выбором осей, поэтому UI один.

// Паспортный порядок встроенной схемы (0 = не встроенная / неизвестно).
int builtin_scheme_order(const std::string& name) {
    for (const auto& b : kBuiltinSchemes)
        if (name == b.name) return b.order;
    return 0;
}
