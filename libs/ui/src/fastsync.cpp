#include "gui_internal.h"

namespace ucuda::ui_detail {


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
    auto& renderer = ui_resource<std::unique_ptr<PlotRenderer>, 10>();
    auto& hm_map = ui_resource<std::map<unsigned, std::unique_ptr<HeatmapView>>, 11>();
    auto& traj_map = ui_resource<std::map<unsigned, std::unique_ptr<Plot2DView>>, 12>();
    if (!renderer) renderer = std::make_unique<PlotRenderer>();

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
            auto& traj3d_map = ui_resource<std::map<unsigned, std::unique_ptr<Plot3DView>>, 504>();
            int vz = (c.axis_z_var >= 0 && c.axis_z_var < nX) ? c.axis_z_var : 2;
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


// Селектор мишени свипа: единый список "h + все элементы a[]". Параметры
// метода отдельной сущностью не являются — они объявляются обычными
// параметрами системы и читаются телом кастомной КРС как a[k], поэтому и
// попадают сюда наравне с остальными.
void draw_order_target_combo(const char* label, int& target,
                                    const OrderAnalysisSession& s) {
    const std::vector<std::string> names = s.axis_target_names();
    const int n = (int)names.size();
    int sel = target + 1;                       // -1 (h) -> 0
    if (sel < 0 || sel >= n) sel = 0;
    ImGui::SetNextItemWidth(kComboW);
    if (ImGui::BeginCombo(label, names[(size_t)sel].c_str())) {
        for (int i = 0; i < n; ++i) {
            if (ImGui::Selectable(names[(size_t)i].c_str(), i == sel)) {
                target = i - 1;
                sel = i;
            }
        }
        ImGui::EndCombo();
    }
}


// Блок настроек одной оси. Возвращает ничего: всё пишется прямо в конфиг.
// n_text == nullptr — ось не владеет числом узлов (в 2D оно общее и живёт на
// X): поле не рисуется, вместо него подпись, откуда берётся разрешение.
void draw_order_axis_block(const char* id, const char* title,
                                  const OrderAnalysisSession& s,
                                  int& target, std::string& lo, std::string& hi,
                                  bool& log_grid, std::string* n_text,
                                  bool enabled, const char* n_label ) {
    ImGui::PushID(id);
    if (!enabled) ImGui::BeginDisabled();
    ImGui::SeparatorText(title);
    draw_order_target_combo("sweep over", target, s);
    InputNumStr("from", lo, kFieldW);
    ImGui::SameLine();
    InputNumStr("to", hi, kFieldW);
    if (n_text) {
        InputNumStr(n_label, *n_text, kFieldW);
        ImGui::SameLine();
    }
    else {
        ImGui::TextDisabled("resolution: same as X");
        ImGui::SameLine();
    }
    ImGui::Checkbox("log grid", &log_grid);
    if (log_grid) {
        ImGui::SameLine();
        ImGui::TextDisabled("(from/to > 0)");
    }
    if (!enabled) ImGui::EndDisabled();
    ImGui::PopID();
}


// Индикатор числа шагов + предупреждение о нецелом t_max/h.
//
// Почему это не косметика: без snap число грубых шагов = floor(t_max/h), и
// фактическое конечное время N*h отличается от t_max на величину до h. Оно
// разное у соседних узлов сетки, так что на кривую ошибки садится вклад
// порядка O(h) — для схем порядка выше первого он полностью забивает
// измеряемую величину. На лог-сетке по h нецелым оказывается почти каждый
// узел, поэтому счётчик показывает, сколько именно узлов задеты.
void draw_order_steps_hint(const OrderConfig& c, const OrderAnalysisSession& s) {
    const double tmax = parse_ratio_or(c.t_max_text, 0.0);
    if (!(tmax > 0.0)) { ImGui::TextDisabled("computing time must be > 0"); return; }

    // Собираем узлы по h — либо одну точку (h не свипуется), либо всю ось.
    std::vector<double> hs;
    auto collect = [&](int target, const std::string& lo_s, const std::string& hi_s,
                       bool log_grid, const std::string& n_s, bool active) {
        if (!active || target != kOrderTargetH) return false;
        double lo = parse_ratio_or(lo_s, 0.0), hi = parse_ratio_or(hi_s, 0.0);
        if (hi < lo) std::swap(lo, hi);
        int n = std::max(1, (int)parse_ratio_or(n_s, 200.0));
        n = std::min(n, 100000);
        hs.reserve((size_t)n);
        if (n == 1) { hs.push_back(lo); return true; }
        if (log_grid && lo > 0.0 && hi > 0.0) {
            const double k = std::log(hi / lo) / (double)(n - 1);
            for (int i = 0; i < n; ++i) hs.push_back(lo * std::exp(k * (double)i));
        } else {
            const double d = (hi - lo) / (double)(n - 1);
            for (int i = 0; i < n; ++i) hs.push_back(lo + d * (double)i);
        }
        return true;
    };
    bool sweeping_h = collect(c.axis_x_target, c.axis_x_lo_text, c.axis_x_hi_text,
                              c.axis_x_log, c.axis_x_n_text, true);
    if (!sweeping_h)
        sweeping_h = collect(c.axis_y_target, c.axis_y_lo_text, c.axis_y_hi_text,
                             c.axis_y_log, c.axis_x_n_text, c.two_d);
    if (!sweeping_h) hs.push_back(parse_ratio_or(c.h_text, 0.0));

    long long n_min = -1, n_max = -1;
    int n_bad = 0, n_zero = 0;
    double worst_rel = 0.0;
    for (double h : hs) {
        if (!(h > 0.0)) { ++n_zero; continue; }
        const long long N = order_steps_for_h(h, tmax, c.snap_steps);
        if (n_min < 0 || N < n_min) n_min = N;
        if (n_max < 0 || N > n_max) n_max = N;
        const double r = tmax / h;
        const double frac = std::fabs(r - std::floor(r + 0.5));
        if (frac > 1e-9 * std::max(1.0, r)) {
            ++n_bad;
            // Насколько подогнанный шаг отличается от заданного.
            const double h_snap = tmax / (double)std::max(1LL, (long long)(r + 0.5));
            const double rel = std::fabs(h_snap - h) / h;
            if (rel > worst_rel) worst_rel = rel;
        }
    }

    if (n_zero > 0)
        ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f),
                           "%d nodes with h <= 0 - they will go to diverged", n_zero);
    if (n_min > 0) {
        // Полная работа: 7 вызовов КРС на грубый шаг (1 + 2 + 4).
        double total = 0.0;
        for (double h : hs) if (h > 0.0) total += (double)order_steps_for_h(h, tmax, c.snap_steps);
        double cells = (double)hs.size();
        if (c.two_d) {
            // Сетка квадратная, N x N. hs уже содержит узлы той оси, что свипует
            // h; вторая добавляет N. Если h не свипуется вовсе, hs — одна точка,
            // и обе оси идут множителями (раньше здесь терялась одна из них, и
            // счётчик ячеек занижался ровно в N раз).
            const double n_side = std::max(1.0, parse_ratio_or(c.axis_x_n_text, 200.0));
            const double other = sweeping_h ? n_side : n_side * n_side;
            total *= other;
            cells *= other;
        }
        if (n_min == n_max)
            ImGui::Text("steps per cell: %lld   cells: %.0f   KRS calls: %.3g",
                        n_min, cells, total * 7.0);
        else
            ImGui::Text("steps per cell: %lld..%lld   cells: %.0f   KRS calls: %.3g",
                        n_min, n_max, cells, total * 7.0);
    }

    if (n_bad > 0) {
        if (c.snap_steps) {
            ImGui::TextColored(ImVec4(0.65f, 0.85f, 0.65f, 1.0f),
                "t_max/h non-integer in %d of %d nodes - h snapped to t_max/N (shift up to %.2g%%)",
                n_bad, (int)hs.size(), worst_rel * 100.0);
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.3f, 1.0f),
                "t_max/h non-integer in %d of %d nodes: the final time drifts, "
                "an O(h) contribution is mixed into the error", n_bad, (int)hs.size());
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "The number of steps must be an integer. Otherwise the actual final time\n"
                "N*h differs from t_max and CHANGES from node to node, and the difference\n"
                "between solutions at different moments in time is of order O(h) - it\n"
                "adds up with the measured error and for schemes of order above\n"
                "the first it drowns it out completely.\n"
                "\"snap h -> t_max/N\" removes this by shifting the step itself slightly.");
    }
}


// ---------------------------------------------------------------------------
// Блок настроек области устойчивости.
//
// Элементы тестовой матрицы здесь НЕ вводятся — вводятся k, r и h, а a, b, c, d
// пересчитываются в каждом узле сетки так, чтобы спектр был ровно
// sigma +- i*omega (формулы в kernels/order.template.cu). Поэтому поля матрицы
// в этом блоке отсутствуют, а вместо них стоят селекторы «в какой a[] класть
// какой элемент»: пресет library/Dahlquist 2D даёт 1..4, но переименовать или
// переставить параметры системы пользователю никто не запрещает, и молча
// промахнуться мимо индекса — это неверная картинка без единого сообщения.
void draw_order_stability_block(AppModel& model, const OrderAnalysisSession& s,
                                       OrderConfig& c) {
    if (!ImGui::CollapsingHeader("Stability region", ImGuiTreeNodeFlags_DefaultOpen)) return;

    if ((int)s.vars.size() != 2) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
            "This computation needs the linear two-dimensional test system: %d state "
            "variables are loaded.", (int)s.vars.size());
        ImGui::TextDisabled("Library -> Dahlquist 2D. Run will be refused until then.");
        ImGui::Separator();
    }

    InputNumStr("asymmetry k", c.stab_k_text, kFieldW);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Drives the test matrix along the family with the SAME spectrum:\n"
            "a = k*d and d = 2*sigma/(1+k), so the trace stays 2*sigma and the\n"
            "determinant stays sigma^2 + omega^2 whatever k is. k = 1 gives the\n"
            "symmetric case a = d; the further from 1, the stronger the\n"
            "asymmetry the scheme has to cope with at unchanged eigenvalues.\n"
            "k = -1 is forbidden: it makes d infinite.");
    InputNumStr("ratio r", c.stab_r_text, kFieldW);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Scales the off-diagonal entries: b = r*c, and c itself is built so\n"
            "that the spectrum does not move. Must be NEGATIVE - with r >= 0 the\n"
            "radicand of c is not positive and no real test matrix exists.");
    InputNumStr("step h", c.stab_h_text, kFieldW);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "The step whose transition operator is diagonalised. Exactly ONE\n"
            "step is taken: the test is linear and homogeneous, so R_N = R^N and\n"
            "more steps would only raise the same spectrum to a power.\n"
            "\n"
            "At h = 1 (the default) the axes read directly as h*sigma and\n"
            "h*omega - the classic dimensionless picture.");

    const double k_val = parse_num(c.stab_k_text, 1.0);
    const double r_val = parse_num(c.stab_r_text, -1.0);
    if (k_val == -1.0)
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                           "k = -1: d = 2*sigma/(1+k) is undefined, the run will be refused.");
    if (!(r_val < 0.0))
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                           "r must be negative: with r >= 0 the matrix has no real form.");

    ImGui::Separator();
    InputNumStr("sigma from", c.stab_sig_lo_text, kFieldW);
    ImGui::SameLine();
    InputNumStr("to##sigma", c.stab_sig_hi_text, kFieldW);
    InputNumStr("omega from", c.stab_om_lo_text, kFieldW);
    ImGui::SameLine();
    InputNumStr("to##omega", c.stab_om_hi_text, kFieldW);
    InputNumStr("resolution (both axes)", c.stab_n_text, kFieldW);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "The map is always N x N: a cell costs two steps of the scheme, so\n"
            "even 512 x 512 is cheaper than a single node of the p(h) curve.");

    ImGui::Separator();
    InputNumStr("preference tolerance", c.stab_pref_tol_text, kFieldW);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Preference region (Fedoseev, Karimov, Legat, Butusov, Mathematics\n"
            "10:4327, 2022): the part of the stability region where one step is\n"
            "also ACCURATE. A cell is preferred when rho <= 1 and\n"
            "\n"
            "    err = ||e^{hA} - R||_2 / ||e^{hA}||_2 <= tolerance,\n"
            "\n"
            "e^{hA} being the exact transition operator of the test problem and\n"
            "R the one of the scheme (spectral norm). The paper uses 1, i.e. up\n"
            "to 100%% error per step - a 'not nonsense' boundary rather than an\n"
            "accuracy target; 0.1 or 0.01 show where the scheme is actually good.\n"
            "\n"
            "Applied when drawing, not during the run: change it and the overlay\n"
            "follows at once, no rerun needed.");
    const double tol_val = parse_num(c.stab_pref_tol_text, 1.0);
    if (!(tol_val > 0.0))
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.4f, 1.0f),
                           "tolerance <= 0: the preference region is empty by definition.");

    // Селекторы индексов a[]. Подписи те же, что у селектора оси свипа, минус
    // "h" и минус a[0] — коэффициент симметрии s занят схемой.
    ImGui::Separator();
    std::vector<std::string> items;
    items.reserve(s.params.size());
    for (size_t i = 0; i < s.params.size(); ++i)
        items.push_back(s.params[i] + " (a[" + std::to_string(i + 1) + "])");
    std::vector<const char*> item_ptrs;
    item_ptrs.reserve(items.size());
    for (const auto& it : items) item_ptrs.push_back(it.c_str());

    auto idx_combo = [&](const char* label, int& target) {
        int sel = target - 1;   // a[1] — первый элемент списка
        if (sel < 0 || sel >= (int)item_ptrs.size()) sel = -1;
        ImGui::SetNextItemWidth(kFieldW + 60.0f);
        if (ImGui::Combo(label, &sel, item_ptrs.empty() ? nullptr : item_ptrs.data(),
                         (int)item_ptrs.size()))
            target = sel + 1;
    };
    idx_combo("matrix a##stab", c.stab_idx_a);
    idx_combo("matrix b##stab", c.stab_idx_b);
    idx_combo("matrix c##stab", c.stab_idx_c);
    idx_combo("matrix d##stab", c.stab_idx_d);
    ImGui::TextDisabled("Which parameter of the system carries which entry of "
                        "[[a, b], [c, d]].");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "The entries are recomputed in every node from (sigma, omega, k, r)\n"
            "and written into these a[] slots, so whatever you set here is\n"
            "overwritten during the run - the values on the Parameters panel are\n"
            "not used for them.\n"
            "\n"
            "For Library -> Dahlquist 2D this is a, b, c, d = a[1..4]. A custom\n"
            "KRS for this computation must address exactly these slots.");

    (void)model;
}


// Вид окна, в котором конфигу место по его текущим настройкам.
OrderPlotWindow::Kind order_kind_for(const OrderConfig& c) {
    if (c.calc_kind == kOrderCalcPerf) return OrderPlotWindow::Kind::Perf;
    // Область устойчивости — карта по построению: сетка всегда (sigma, omega).
    if (c.calc_kind == kOrderCalcStab) return OrderPlotWindow::Kind::Map;
    return c.two_d ? OrderPlotWindow::Kind::Map : OrderPlotWindow::Kind::P;
}


// Конфиг сменил вид расчёта (или размерность сетки) — перевесить его на окно,
// которое такое умеет показывать. Без этого переключение на performance
// вынимало вкладку из всех окон, и Run выглядел как «ничего не произошло».
//
// Новое окно заводится только в крайнем случае. Раньше карта (а с ней и
// область устойчивости) ВСЕГДА получала своё окно, а окно, из которого
// конфиг ушёл, оставалось пустым, — и каждое переключение Order p /
// Performance / Stability плодило по окну. Теперь порядок такой:
//   1) конфиг уже лежит в подходящем окне — ничего не делать;
//   2) есть окно нужного вида: для одномерных — любое (оверлей), для карты —
//      только пустое (хитмапа одиночна);
//   3) иначе окно, которое опустело от этого ухода, просто меняет вид;
//   4) иначе — новое окно.
// Окно, опустевшее от ухода и не пригодившееся в 3), убирается: пустым оно
// лишь занимает место в раскладке. Окна с ручной подписью не трогаются.
void order_rehome_config(AppModel& model, int idx) {
    OrderAnalysisSession& s = model.order_session;
    if (idx < 0 || idx >= (int)s.configs.size()) return;
    const OrderConfig& c = s.configs[(size_t)idx];
    const OrderPlotWindow::Kind want = order_kind_for(c);
    auto& wins = model.order_plot_windows;

    bool placed = false;
    std::vector<int> vacated;   // id окон, опустевших от этого ухода
    for (auto& w : wins) {
        auto& mem = w.members;
        const bool has = std::find(mem.begin(), mem.end(), idx) != mem.end();
        if (!has) continue;
        if (order_member_fits(c, w.kind)) { placed = true; continue; }
        mem.erase(std::remove(mem.begin(), mem.end(), idx), mem.end());
        w.plot_sig = -1;
        if (mem.empty()) vacated.push_back(w.id);
    }
    if (!placed) {
        for (auto& w : wins) {
            if (w.kind != want) continue;
            if (want == OrderPlotWindow::Kind::Map && !w.members.empty()) continue;
            w.members.push_back(idx);
            w.plot_sig = -1;
            placed = true;
            break;
        }
    }
    if (!placed && !vacated.empty()) {
        for (auto& w : wins) {
            if (w.id != vacated.front()) continue;
            // То же, что делает смена вида в комбо "Plot windows".
            w.kind = want;
            w.members.assign(1, idx);
            if (want == OrderPlotWindow::Kind::P) w.y_log = false;
            w.plot_sig = -1;
            placed = true;
            break;
        }
        vacated.erase(vacated.begin());
    }
    if (!placed) model.add_order_plot_window(want, { idx });
    wins.erase(std::remove_if(wins.begin(), wins.end(), [&](const OrderPlotWindow& w) {
                   return w.members.empty() && !w.label_is_manual
                       && std::find(vacated.begin(), vacated.end(), w.id) != vacated.end();
               }), wins.end());
    model.order_plot_windows_dirty = true;
}


void draw_order_controls(AppModel& model, SystemLibrary& /*lib*/) {
    OrderAnalysisSession& s = model.order_session;
    // Чей конфиг правит эта панель — для ПКМ-рассылки "во все такие же".
    model.broadcast_source_tab = BroadcastTab::Order;
    model.broadcast_source_idx = s.active_config_index;
    if (s.configs.empty()) {
        // Система загружена, а вкладок нет: так остаются сессии, у которых
        // удалили последнюю вкладку до запрета на это (см. draw_config_tab_bar).
        // Таб-бара с "+" здесь нет, поэтому без этой кнопки вкладку было не
        // вернуть ничем. Новая вкладка строится из записи системы — теми же
        // умолчаниями, что и при первом входе в Order.
        if (s.loaded_system_name.empty() || s.loaded_system_name != model.name) {
            ImGui::TextDisabled("No system loaded. Pick a system in the list above.");
            return;
        }
        ImGui::TextDisabled("This system has no Order tabs.");
        if (ImGui::Button("New tab from system defaults") && model.start_order_analysis()
            && !s.configs.empty())
            order_rehome_config(model, 0);
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
        s.active_config_index = 0;

    // Run / Run all... — над таб-баром, как у Basins/FastSync: управляют активным
    // конфигом и должны быть доступны без прокрутки. Ctrl+R / Ctrl+Shift+R приходят
    // вместе с общим хелпером; "Run all..." кладёт отмеченные индексы в
    // model.order_queue, которую draw_gui тикает после poll'ов.
    {
        const bool no_cfg = s.configs.empty();
        RunAllGroup g;
        g.pick_id = "porder_";
        g.n       = (int)s.configs.size();
        g.label   = [&s](int i) { return s.configs[i].label; };
        g.enqueue = [&model](int i) { model.order_queue.push_back({ i }); };
        draw_run_and_run_all("##run_all_order", s.in_flight, no_cfg,
                             s.in_flight || no_cfg, 160.0f,
                             [&model, &s]() {
                                 if (!model.parametric_engine)
                                     model.parametric_engine = std::make_unique<ParametricEngine>();
                                 s.run_async(*model.parametric_engine, s.active_config_index);
                             },
                             [&s]() {
                                 // Локальный Stop остаётся рядом с Run: глобальный
                                 // Cancel живёт в статус-баре, но привычка жать Stop
                                 // тут же, где Run, старше вкладки.
                                 ImGui::SameLine();
                                 if (!s.in_flight) ImGui::BeginDisabled();
                                 if (ImGui::Button("Stop", ImVec2(90, 0))) s.request_cancel();
                                 if (!s.in_flight) ImGui::EndDisabled();
                             },
                             { g },
                             [&model]() { model.start_next_in_order_queue(); },
                             model.order_queue.size());
    }
    ImGui::Separator();

    const TabBarResult tabs = draw_config_tab_bar(
        "##order_tabs", "order_tab_", (int)s.configs.size(),
        s.in_flight, s.running_config_index,
        [&s](int i) { return s.configs[i].label; },
        {},
        [&s, &model]() {
            s.add_config();
            // Новая вкладка обязана где-то показаться: кладём её в первое
            // подходящее окно-оверлей, а если такого нет — заводим своё.
            const int ni = (int)s.configs.size() - 1;
            if (ni < 0) return;
            const OrderPlotWindow::Kind want = order_kind_for(s.configs[(size_t)ni]);
            if (want != OrderPlotWindow::Kind::Map) {
                for (auto& w : model.order_plot_windows)
                    if (w.kind == want) {
                        w.members.push_back(ni);
                        model.order_plot_windows_dirty = true;
                        return;
                    }
            }
            model.add_order_plot_window(want, { ni });
        });
    if (tabs.active    >= 0) s.active_config_index = tabs.active;
    if (tabs.to_remove >= 0) model.remove_order_config(tabs.to_remove);

    if (s.configs.empty()) {
        ImGui::TextDisabled("No order configs. Press '+' to add one.");
        return;
    }
    if (s.active_config_index >= (int)s.configs.size())
        s.active_config_index = (int)s.configs.size() - 1;

    const int idx = s.active_config_index;
    OrderConfig& c = s.configs[(size_t)idx];

    draw_label_rename("##order_label", c.label, kFieldW);
    ImGui::SameLine();
    ImGui::TextDisabled("|"); ImGui::SameLine();
    const int calc_kind_before = c.calc_kind;
    ImGui::RadioButton("order p", &c.calc_kind, kOrderCalcOrder);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Accuracy order and the error curve over the node grid.");
    ImGui::SameLine();
    ImGui::RadioButton("performance", &c.calc_kind, kOrderCalcPerf);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Timing over the same X axis: on every node the kernel that\n"
            "integrates a single trajectory with that node's step is launched\n"
            "several times under cudaEvents. The plot then shows time against\n"
            "the error achieved - what that accuracy costs.");
    ImGui::SameLine();
    ImGui::RadioButton("stability", &c.calc_kind, kOrderCalcStab);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Stability region on the two-dimensional Dahlquist test problem.\n"
            "The grid runs over the eigenvalues of the test matrix: sigma along\n"
            "X, omega along Y. In every node the matrix is rebuilt from\n"
            "(sigma, omega) and the settings k and r, one step of the scheme is\n"
            "taken from each basis vector, and the map shows the spectral radius\n"
            "of the resulting amplification matrix. The boundary is rho = 1.\n"
            "\n"
            "Needs the linear two-dimensional test system - Library ->\n"
            "Dahlquist 2D. Any other system is refused.");
    if (c.calc_kind != calc_kind_before) order_rehome_config(model, idx);

    // Устройство расчёта. Обе ветки считают по одним формулам и одним порогам,
    // но замер времени у них меряет РАЗНОЕ: GPU — ядро с replicas нитями,
    // CPU — одну траекторию последовательно. Числа между устройствами поэтому
    // не сопоставимы, и это не баг, а два разных вопроса к схеме.
    ImGui::SameLine();
    ImGui::TextDisabled("|"); ImGui::SameLine();
    int ord_dev = c.use_gpu ? 0 : 1;
    ImGui::RadioButton("GPU##ord_dev", &ord_dev, 0);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Kernel compiled by NVRTC (kernels/order.template.cu).");
    ImGui::SameLine();
    ImGui::RadioButton("CPU##ord_dev", &ord_dev, 1);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "The same KRS body, compiled to a native step by cl.exe instead of\n"
            "NVRTC, run sequentially on one core. Same formulas, same floor and\n"
            "divergence thresholds, so the order diagram is directly comparable\n"
            "with the GPU one.\n"
            "Performance means something else here: the time of ONE trajectory,\n"
            "not the throughput of a loaded GPU - replicas are ignored.");
    c.use_gpu = (ord_dev == 0);

    // Точность арифметики. Есть только у CPU: на GPU ядро собирается в numb,
    // и выбор там был бы обманом — цифры те же, смысл другой.
    ImGui::SameLine();
    ImGui::TextDisabled("|"); ImGui::SameLine();
    if (c.use_gpu) ImGui::BeginDisabled();
    const char* prec_items[] = { "double", "double-double", "quad-double" };
    ImGui::SetNextItemWidth(150.0f);
    ImGui::Combo("precision##ord_prec", &c.cpu_precision, prec_items, IM_ARRAYSIZE(prec_items));
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Arithmetic of the CPU branch.\n"
            "double        - 16 digits, bit-for-bit the same as the GPU.\n"
            "double-double - 32 digits (106-bit mantissa), about 35x slower.\n"
            "quad-double   - 62 digits (212-bit mantissa), about 600x slower.\n"
            "\n"
            "The order estimate subtracts nearby solutions, so it sits on a\n"
            "rounding floor of about 2*eps*|y|*sqrt(N) - near 3e-13 in double.\n"
            "A scheme of order p gives E1 ~ h^p, so from p = 8 on there is\n"
            "almost no usable window in h left: the curve hits the floor before\n"
            "the scheme reaches its asymptotic regime. double-double lowers the\n"
            "floor by 16 decades, quad-double by 47.\n"
            "\n"
            "Measured on Extr(CD|1,2,3,4): the floor sits at 5e-13 in double,\n"
            "2e-26 in double-double, below 1e-50 in quad-double.\n"
            "\n"
            "The same KRS body is rebuilt with numb = ucuda::dd / ucuda::qd;\n"
            "each build is cached separately, so switching does not recompile.");
    if (c.use_gpu) ImGui::EndDisabled();

    if (!c.use_gpu) {
        std::string why;
        if (!krs_cpu_backend_available(&why))
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.3f, 1.0f),
                               "CPU backend unavailable: %s", why.c_str());

        // Подмена допуска Ньютона обязана быть видимой: это чужая настройка,
        // и без строки пользователь не поймёт, почему неявная схема на этой
        // вкладке считается иначе, чем на остальных.
        const double ntol = order_newton_tol_for_precision(c.cpu_precision);
        if (ntol > 0.0) {
            ImGui::TextDisabled("Implicit schemes: Newton tolerance %.0e instead of the "
                                "system setting (%s)", ntol, model.newton_tol.c_str());
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Newton stops at ||dX|| < tol, so the system-wide setting would\n"
                    "cap the result long before the arithmetic does: in double-double\n"
                    "the rounding floor sits near 1e-29, and a 1e-10 tolerance would\n"
                    "throw away nineteen decades of it.\n"
                    "\n"
                    "This tab measures the order of the SCHEME, not the accuracy of\n"
                    "the solver, so the tolerance is tightened to match the chosen\n"
                    "arithmetic - never loosened: a stricter setting of your own is\n"
                    "kept. Explicit schemes are unaffected.");
        }
    }
    // Дискретное отображение шага не имеет вовсе, уточнять нечего: обе
    // диаграммы вкладки меряют, как ошибка убывает с h, и на карте x_{n+1} =
    // f(x_n) обе разности тождественно нулевые.
    if (s.sys.is_map)
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.4f, 1.0f),
                           "Discrete map: there is no step to refine, the accuracy order is undefined.");
    ImGui::Separator();

    // ---- Интегрирование ----
    if (ImGui::CollapsingHeader("Integration", ImGuiTreeNodeFlags_DefaultOpen)) {
        draw_scheme_combo("scheme", c.scheme, s.custom_schemes, {}, &s.enabled_builtin_schemes,
                          &model, model.is_map, &s.wrapper_schemes);
        if (scheme_uses_symmetry(c.scheme, s.custom_schemes))
            InputNumStr("symmetry s", c.symmetry_s, kFieldW);
        // Область устойчивости берёт из этого блока только схему и s: она
        // делает ОДИН шаг, поэтому ни времени счёта, ни порога расхождения,
        // ни подгонки числа шагов у неё нет, а собственный шаг живёт в своём
        // блоке ниже — рядом с k и r, от которых зависит смысл осей.
        if (c.calc_kind == kOrderCalcStab) {
            ImGui::TextDisabled("The step and the grid of the stability region live in "
                                "\"Stability region\" below.");
        } else {
        const bool h_swept = (c.axis_x_target == kOrderTargetH)
                          || (c.two_d && c.axis_y_target == kOrderTargetH);
        if (!model.is_map) {
            if (h_swept) ImGui::BeginDisabled();
            InputNumStr("h", c.h_text, kFieldW);
            if (h_swept) ImGui::EndDisabled();
            if (h_swept) {
                ImGui::SameLine();
                ImGui::TextDisabled("(swept along the axis)");
            }
        }
        InputNumStr(model.is_map ? "iterations" : "computing time", c.t_max_text, kFieldW);
        InputNumStr("max value", c.max_value_text, kFieldW);

        ImGui::Checkbox("snap h -> t_max/N", &c.snap_steps);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Adjust the step so that the number of steps is an integer.\n"
                "Turning this off reproduces the behaviour of the other tabs\n"
                "(floor(t_max/h)) at the cost of a parasitic O(h) contribution to the error.");
        ImGui::SameLine();
        ImGui::Checkbox("endpoint only", &c.endpoint_only);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Measure the difference only at t = t_max, not the maximum over the whole\n"
                "trajectory. On chaotic systems the maximum over the trajectory\n"
                "quickly starts reflecting divergence rather than the order of the scheme.");

        draw_order_steps_hint(c, s);
        }
    }

    // ---- Stability region ----
    if (c.calc_kind == kOrderCalcStab) draw_order_stability_block(model, s, c);

    // ---- Performance ----
    if (c.calc_kind == kOrderCalcPerf &&
        ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("measurements", c.perf_repeats_text, kFieldW);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "How many times to run the computation on every node. Minimum,\n"
                "maximum and average are taken over these measurements - all\n"
                "three go onto the plot.");
        InputNumStr("warmup runs", c.perf_warmup_text, kFieldW);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Launches that do not count, made before the measurements. The\n"
                "very first launch drags the code load into the context along with\n"
                "it and is therefore always slower than the rest.");
        ImGui::Separator();
        // Режим шага — первым: от него зависит, какой эталон у вкладки (ровно один).
        const bool perf_ad_on = !s.sys.is_map && c.adaptive.enabled;
        if (!s.sys.is_map) {
            draw_adaptive_block("order_perf_ad", c.adaptive, c.scheme, 0, s.sys.is_map, &model);
            if (c.adaptive.enabled) {
                InputNumStr("tol from", c.perf_tol_lo_text, kFieldW);
                InputNumStr("tol to", c.perf_tol_hi_text, kFieldW);
                InputNumStr("tol points", c.perf_tol_n_text, kFieldW);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "The nodes of the adaptive benchmark: a log grid of tol; on every node\n"
                        "rtol = tol and atol = tol * (atol / rtol) of the settings above. The\n"
                        "Sweep axis below is not used. Every node integrates [0, t_max] (the\n"
                        "step only, no output) and is timed like the fixed step: on the GPU by\n"
                        "a kernel, on the CPU by the same driver compiled with cl.exe (double).\n"
                        "Steps, rejections and f evaluations come from the controller.");
            }
            ImGui::Separator();
        }
        if (!perf_ad_on) {
            // Эталон постоянного шага: чем считаем «точный ответ» для Eref по оси X.
            draw_scheme_combo("reference method", c.perf_ref_scheme, s.custom_schemes, {},
                              &s.enabled_builtin_schemes, &model, s.sys.is_map, &s.wrapper_schemes,
                              /*broadcast*/ false);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "The method that stands for the exact answer in Eref =\n"
                    "max|y_h - y_ref|. DOPRI78 by default: at the steps in use its\n"
                    "own error sits below the machine precision of the solution.\n"
                    "With \"endpoint only\" and \"fit h to t_max\" Eref is the error at t_max:\n"
                    "it is then also this tab's E(T), comparable with adaptive tabs.");
            InputNumStr("reference substeps", c.perf_ref_substeps_text, kFieldW);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Reference steps per one step of the scheme under test. 0 turns\n"
                    "the reference off entirely - the branch is then not even compiled\n"
                    "into the kernel. Keep it above 1 when the reference method is the\n"
                    "same as the tested one: at an equal step that would be the very\n"
                    "same arithmetic and the difference identically zero.");
            // y*(T) для E(T) — тот же эталон, что у адаптивного шага: с ним кривые постоянного
            // и адаптивного шага ложатся на одну диаграмму (X: E(T) в окне Performance).
            const char* refs_fx[] = { "none", "DOP853 in dd (CPU)", "DOP853 in qd (CPU)" };
            int sel_fx = (c.perf_end_ref == 1 || c.perf_end_ref == 2) ? c.perf_end_ref : 0;
            ImGui::SetNextItemWidth(kComboW);
            if (ImGui::Combo("reference y*(T)##fixed", &sel_fx, refs_fx, IM_ARRAYSIZE(refs_fx)))
                c.perf_end_ref = sel_fx;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "E(T) = max|y(T) - y*(T)| at t_max against DOP853 in double-double or\n"
                    "quad-double on the CPU - the same y*(T) the adaptive step is measured\n"
                    "against, so fixed-step and adaptive curves can share one diagram\n"
                    "(X: E(T) in the Performance window). Needs \"fit h to t_max\": a node\n"
                    "that ends at N*h != t_max has no E(T). Computed after the timing and\n"
                    "not part of it. none: E(T) is Eref at t_max (\"endpoint only\").");
        } else {
            // Эталон адаптивного шага: y*(T) высокой точности — его единственная ошибка E(T).
            const char* refs[] = { "DOP853 in dd (CPU)", "DOP853 in qd (CPU)" };
            int sel = (c.perf_end_ref == 2) ? 1 : 0;
            ImGui::SetNextItemWidth(kComboW);
            if (ImGui::Combo("reference y*(T)", &sel, refs, IM_ARRAYSIZE(refs))) c.perf_end_ref = sel + 1;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "E(T) = max|y(T) - y*(T)| at the end of the interval (t_max), against\n"
                    "DOP853 with a fixed step in double-double (~32 digits) or quad-double\n"
                    "(~62 digits) arithmetic on the CPU. Its step count doubles until two\n"
                    "successive answers agree, so y* is exact to far below any error on\n"
                    "the plot. The adaptive step has no h grid, so E(T) is its only error.");
        }
        ImGui::Separator();
        // Реплики — понятие ядра: столько одинаковых нитей грузят GPU. На CPU
        // прогон последовательный, и поле нечего означать, поэтому гасим его,
        // а не молча игнорируем значение.
        if (!c.use_gpu) ImGui::BeginDisabled();
        InputNumStr("replicas", c.perf_replicas_text, kFieldW);
        if (!c.use_gpu) ImGui::EndDisabled();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                c.use_gpu
                ? "How many IDENTICAL tasks the timed launch computes.\n"
                  "1 - the time of a single computation (the GPU mostly idles);\n"
                  "a large number - the throughput of a loaded GPU, where the\n"
                  "time per task = measured / replicas."
                : "GPU only: the CPU branch runs one trajectory sequentially,\n"
                  "so the measurement is always the time of a single task.");

        const int n_nodes = c.adaptive.enabled ? std::max(1, (int)parse_num(c.perf_tol_n_text, 11.0))
                                               : std::max(1, (int)parse_num(c.axis_x_n_text, 200.0));
        const int reps    = std::max(1, (int)parse_num(c.perf_repeats_text, 20.0));
        const int wu      = std::max(0, (int)parse_num(c.perf_warmup_text, 2.0));
        const long long launches = (long long)n_nodes * (long long)(reps + wu);
        ImGui::TextDisabled("%d nodes x %d launches = %lld; plus the order pass for the errors",
                            n_nodes, reps + wu, launches);
        if (launches > 20000)
            ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.4f, 1.0f),
                               "a great many launches: the measurement will take %d times longer "
                               "than a plain run",
                               reps + wu);
        ImGui::TextDisabled("Only the kernel launch itself is timed: no H2D/D2H, no compilation.");
        // Split compilation не встраивает шаг в ядро, и та же задача считается
        // в разы дольше (на Лоренце с явным Эйлером — в 3.6 раза). Настройка
        // глобальная, поэтому кривые, снятые при разном её положении, сравнивать
        // между собой нельзя.
        if (model.nvrtc_rdc)
            ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.4f, 1.0f),
                               "Split compilation is on (Settings): the step is not inlined into "
                               "the kernel,\nso the measured time is several times larger than "
                               "with it off.");
    }

    // ---- Оси ----
    // У области устойчивости оси заданы смыслом расчёта (sigma и omega —
    // вещественная и мнимая части собственных чисел тестовой матрицы), и
    // общий селектор «h + элементы a[]» к ней неприменим: показывать его
    // рядом с настоящими границами значило бы предлагать выбор, которого нет.
    if (c.calc_kind != kOrderCalcStab &&
        ImGui::CollapsingHeader("Sweep", ImGuiTreeNodeFlags_DefaultOpen)) {
        const bool perf_mode = (c.calc_kind == kOrderCalcPerf);
        const bool perf_ad   = perf_mode && c.adaptive.enabled && !s.sys.is_map;
        if (perf_ad)
            ImGui::TextDisabled("Adaptive step: the benchmark runs over the tol nodes (Performance above);\n"
                                "the X axis below is not used.");
        if (perf_ad) ImGui::BeginDisabled();
        if (perf_mode) ImGui::BeginDisabled();
        if (ImGui::Checkbox("2D map", &c.two_d)) order_rehome_config(model, idx);
        if (perf_mode) {
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::TextDisabled("(the measurement is 1D: the X axis carries the error)");
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Off - a p(X axis) curve. On - a p(X, Y) map.");
        draw_order_axis_block("axx", "Axis X", s, c.axis_x_target,
                              c.axis_x_lo_text, c.axis_x_hi_text,
                              c.axis_x_log, &c.axis_x_n_text, true,
                              c.two_d ? "resolution (both axes)" : "points");
        draw_order_axis_block("axy", "Axis Y", s, c.axis_y_target,
                              c.axis_y_lo_text, c.axis_y_hi_text,
                              c.axis_y_log, nullptr, c.two_d);

        if (c.axis_x_target == kOrderTargetH && c.two_d && c.axis_y_target == kOrderTargetH)
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "both axes sweep h");
        else if (c.two_d && c.axis_x_target == c.axis_y_target)
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "both axes sweep the same thing");
        if (perf_ad) ImGui::EndDisabled();
    }

    // Что из этих двух блоков расчёт задаёт сам. У области устойчивости это
    // ВСЕ начальные условия (шаг делается от базисных векторов) и четыре
    // параметра, несущих элементы тестовой матрицы: они пересчитываются из
    // (sigma, omega, k, r) в каждом узле сетки. Остальные параметры — скажем,
    // коэффициенты обёртки или s_i у комплексных схем — работают как обычно.
    std::function<bool(const std::string&)> ic_overwritten, par_overwritten;
    const char* ic_note  = nullptr;
    const char* par_note =
        "Declare METHOD parameters (for composition schemes) right here as ordinary "
        "system parameters: they need not be used in the right-hand sides, "
        "and the body of a custom KRS reads them as a[k]. In the axis selector they "
        "will appear alongside the rest.";
    if (c.calc_kind == kOrderCalcStab) {
        ic_overwritten = [](const std::string&) { return true; };
        ic_note = "Not used: the step is taken from the basis vectors (1, 0) and (0, 1) - "
                  "that is what makes the result a fundamental matrix.";
        par_note = "The four entries of the test matrix are greyed out: they are rebuilt from "
                   "(sigma, omega) and k, r in every node of the grid, exactly as the "
                   "reference script does it. The rest of the parameters work as usual.";
        std::vector<std::string> taken;
        const int slots[4] = { c.stab_idx_a, c.stab_idx_b, c.stab_idx_c, c.stab_idx_d };
        for (int i = 0; i < 4; ++i) {
            const int pi = slots[i] - 1;
            if (pi >= 0 && pi < (int)s.params.size()) taken.push_back(s.params[(size_t)pi]);
        }
        par_overwritten = [taken](const std::string& n) {
            return std::find(taken.begin(), taken.end(), n) != taken.end();
        };
    }

    draw_named_num_fields("Initial conditions", s.vars, c.initial_conditions,
                          ic_note, &model, BroadcastField::InitCondition, ic_overwritten);
    draw_named_num_fields("Parameters", s.params, c.param_values,
                          par_note, &model, BroadcastField::Param, par_overwritten);

    if (!c.last_error.empty())
        draw_error_box("##order_err", c.last_error);

    // ---- Окна графиков ----
    // Ровно та же схема, что у параметрики и DFT1D: окно знает свой вид и
    // список вкладок, чьи кривые в нём лежат.
    ImGui::Separator();
    ImGui::SeparatorText("Plot windows");
    int win_to_remove = -1;
    for (int wi = 0; wi < (int)model.order_plot_windows.size(); ++wi) {
        OrderPlotWindow& win = model.order_plot_windows[wi];
        ImGui::PushID(win.id);
        ImGui::SetNextItemWidth(150);
        if (InputTextStr("##wlabel", win.label)) {
            win.label_is_manual = !win.label.empty();   // пусто -> обратно к автоподписи
            model.order_plot_windows_dirty = true;
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(140);
        int kind = (int)win.kind;
        const char* kind_names[] = { "Order p", "Error", "Order map 2D", "Performance" };
        if (ImGui::Combo("##wkind", &kind, kind_names, 4)) {
            win.kind = (OrderPlotWindow::Kind)kind;
            // Члены, которые новому виду не подходят, из окна выпадают: иначе
            // окно молча показывало бы "нет данных" от чужой вкладки.
            auto& mem = win.members;
            mem.erase(std::remove_if(mem.begin(), mem.end(), [&](int idx) {
                          return idx < 0 || idx >= (int)s.configs.size()
                              || !order_member_fits(s.configs[(size_t)idx], win.kind);
                      }), mem.end());
            if (win.kind == OrderPlotWindow::Kind::Map && mem.size() > 1) mem.resize(1);
            if (win.kind == OrderPlotWindow::Kind::P) win.y_log = false;
            win.plot_sig = -1;
            model.order_plot_windows_dirty = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Members...")) ImGui::OpenPopup("edit_order_window_members");
        if (ImGui::BeginPopup("edit_order_window_members")) {
            bool any = false;
            for (int ci = 0; ci < (int)s.configs.size(); ++ci) {
                if (!order_member_fits(s.configs[(size_t)ci], win.kind)) continue;
                any = true;
                const std::string lbl = s.configs[(size_t)ci].label + "##ordmem" + std::to_string(ci);
                const bool sel = std::find(win.members.begin(), win.members.end(), ci) != win.members.end();
                if (win.kind == OrderPlotWindow::Kind::Map) {
                    // Хитмапу поверх хитмапы не положишь — здесь выбор одиночный.
                    if (ImGui::RadioButton(lbl.c_str(), sel)) {
                        win.members.assign(1, ci);
                        model.order_plot_windows_dirty = true;
                    }
                } else {
                    bool v = sel;
                    if (ImGui::Checkbox(lbl.c_str(), &v)) {
                        if (v) win.members.push_back(ci);
                        else   win.members.erase(std::remove(win.members.begin(), win.members.end(), ci),
                                                 win.members.end());
                        win.plot_sig = -1;
                        model.order_plot_windows_dirty = true;
                    }
                }
            }
            if (!any) ImGui::TextDisabled("(no matching tabs)");
            ImGui::EndPopup();
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("X")) win_to_remove = wi;
        ImGui::PopID();
    }
    if (win_to_remove >= 0) model.remove_order_plot_window(win_to_remove);

    if (ImGui::Button("Add window"))
        model.add_order_plot_window(order_kind_for(c), { s.active_config_index });
    ImGui::SameLine();
    if (ImGui::Button("Reset windows layout")) model.order_layout_generation++;
}
}
