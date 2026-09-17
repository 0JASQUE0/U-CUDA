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
        static std::vector<float> xy_buf;
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
        const bool fs_busy = s.in_flight &&
                             s.active_config_index == s.running_config_index;
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
        float margin_left, margin_top, margin_right, margin_bottom;
        plot_2d_margins(margin_left, margin_top, margin_right, margin_bottom);
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
        h.popup_extras = [&c, &cb, fs_busy]() {
            draw_export_menu_item(fs_busy, cb, [&c](const std::string& p) {
                data_export::export_fastsync(c.result, p);
            });
        };
        h.render(*renderer, origin, avail,
                 /*owner_id*/ (int)base_oid, c.data_generation,
                 c.result.n_pts_grid, c.result.n_pts_grid,
                 c.result.heatmap.data(),
                 c.result.axis_x_lo, c.result.axis_x_hi,
                 c.result.axis_y_lo, c.result.axis_y_hi,
                 c.result.min_val, c.result.max_val,
                 fit);
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
void draw_order_axis_block(const char* id, const char* title,
                                  const OrderAnalysisSession& s,
                                  int& target, std::string& lo, std::string& hi,
                                  bool& log_grid, std::string& n_text,
                                  bool enabled) {
    ImGui::PushID(id);
    if (!enabled) ImGui::BeginDisabled();
    ImGui::SeparatorText(title);
    draw_order_target_combo("sweep over", target, s);
    InputNumStr("from", lo, kFieldW);
    ImGui::SameLine();
    InputNumStr("to", hi, kFieldW);
    InputNumStr("points", n_text, kFieldW);
    ImGui::SameLine();
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
    if (!(tmax > 0.0)) { ImGui::TextDisabled("computing time должно быть > 0"); return; }

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
                             c.axis_y_log, c.axis_y_n_text, c.two_d);
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
                           "%d узлов с h <= 0 — они уйдут в diverged", n_zero);
    if (n_min > 0) {
        // Полная работа: 7 вызовов КРС на грубый шаг (1 + 2 + 4).
        double total = 0.0;
        for (double h : hs) if (h > 0.0) total += (double)order_steps_for_h(h, tmax, c.snap_steps);
        double cells = (double)hs.size();
        if (c.two_d) {
            const double other = std::max(1.0, parse_ratio_or(
                (c.axis_x_target == kOrderTargetH) ? c.axis_y_n_text : c.axis_x_n_text, 200.0));
            total *= other;
            cells *= other;
        }
        if (n_min == n_max)
            ImGui::Text("шагов на ячейку: %lld   ячеек: %.0f   вызовов КРС: %.3g",
                        n_min, cells, total * 7.0);
        else
            ImGui::Text("шагов на ячейку: %lld..%lld   ячеек: %.0f   вызовов КРС: %.3g",
                        n_min, n_max, cells, total * 7.0);
    }

    if (n_bad > 0) {
        if (c.snap_steps) {
            ImGui::TextColored(ImVec4(0.65f, 0.85f, 0.65f, 1.0f),
                "t_max/h нецелое в %d из %d узлов — h подогнан к t_max/N (сдвиг до %.2g%%)",
                n_bad, (int)hs.size(), worst_rel * 100.0);
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.3f, 1.0f),
                "t_max/h нецелое в %d из %d узлов: конечное время плавает, "
                "в ошибку подмешан вклад O(h)", n_bad, (int)hs.size());
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Число шагов должно быть целым. Иначе фактическое конечное время\n"
                "N*h отличается от t_max и МЕНЯЕТСЯ от узла к узлу, а разница\n"
                "решений в разные моменты времени имеет порядок O(h) — она\n"
                "складывается с измеряемой ошибкой и для схем порядка выше\n"
                "первого полностью её забивает.\n"
                "\"snap h -> t_max/N\" убирает это, слегка сдвигая сам шаг.");
    }
}


void draw_order_controls(AppModel& model, SystemLibrary& /*lib*/) {
    OrderAnalysisSession& s = model.order_session;
    if (s.configs.empty()) {
        ImGui::TextDisabled("Система не загружена. Выбери систему в списке сверху.");
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
        s.active_config_index = 0;

    if (ImGui::BeginTabBar("##order_tabs", ImGuiTabBarFlags_AutoSelectNewTabs |
                                           ImGuiTabBarFlags_Reorderable)) {
        int to_remove = -1;
        for (int i = 0; i < (int)s.configs.size(); ++i) {
            bool open = true;
            ImGui::PushID(i);
            const std::string tab_id = s.configs[(size_t)i].label + "##ordertab" + std::to_string(i);
            if (ImGui::BeginTabItem(tab_id.c_str(), s.configs.size() > 1 ? &open : nullptr)) {
                s.active_config_index = i;
                ImGui::EndTabItem();
            }
            if (!open) to_remove = i;
            ImGui::PopID();
        }
        if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip))
            s.add_config();
        ImGui::EndTabBar();
        if (to_remove >= 0) s.remove_config(to_remove);
    }
    if (s.configs.empty()) return;
    if (s.active_config_index >= (int)s.configs.size())
        s.active_config_index = (int)s.configs.size() - 1;

    const int idx = s.active_config_index;
    OrderConfig& c = s.configs[(size_t)idx];

    // ---- Run / Stop ----
    {
        const bool busy = s.in_flight;
        if (busy) ImGui::BeginDisabled();
        if (ImGui::Button("Run", ImVec2(90, 0))) {
            if (!model.parametric_engine)
                model.parametric_engine = std::make_unique<ParametricEngine>();
            s.run_async(*model.parametric_engine, idx);
        }
        if (busy) ImGui::EndDisabled();
        ImGui::SameLine();
        if (!busy) ImGui::BeginDisabled();
        if (ImGui::Button("Stop", ImVec2(90, 0))) s.request_cancel();
        if (!busy) ImGui::EndDisabled();
        ImGui::SameLine();
        draw_label_rename("##order_label", c.label, kFieldW);
    }

    ImGui::Separator();

    // ---- Интегрирование ----
    if (ImGui::CollapsingHeader("Integration", ImGuiTreeNodeFlags_DefaultOpen)) {
        draw_scheme_combo("scheme", c.scheme, s.custom_schemes, {}, &s.enabled_builtin_schemes);
        if (scheme_uses_symmetry(c.scheme, s.custom_schemes))
            InputNumStr("symmetry s", c.symmetry_s, kFieldW);
        const bool h_swept = (c.axis_x_target == kOrderTargetH)
                          || (c.two_d && c.axis_y_target == kOrderTargetH);
        if (h_swept) ImGui::BeginDisabled();
        InputNumStr("h", c.h_text, kFieldW);
        if (h_swept) ImGui::EndDisabled();
        if (h_swept) {
            ImGui::SameLine();
            ImGui::TextDisabled("(свипуется по оси)");
        }
        InputNumStr("computing time", c.t_max_text, kFieldW);
        InputNumStr("max value", c.max_value_text, kFieldW);

        ImGui::Checkbox("snap h -> t_max/N", &c.snap_steps);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Подогнать шаг так, чтобы число шагов было целым.\n"
                "Выключение воспроизводит поведение остальных вкладок\n"
                "(floor(t_max/h)) ценой паразитного вклада O(h) в ошибку.");
        ImGui::SameLine();
        ImGui::Checkbox("endpoint only", &c.endpoint_only);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Мерить разность только в t = t_max, а не максимум по всей\n"
                "траектории. На хаотических системах максимум по траектории\n"
                "быстро начинает отражать разбегание, а не порядок схемы.");

        draw_order_steps_hint(c, s);
    }

    // ---- Оси ----
    if (ImGui::CollapsingHeader("Sweep", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("2D map", &c.two_d);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Выключено — кривая p(ось X). Включено — карта p(X, Y).");
        draw_order_axis_block("axx", "Axis X", s, c.axis_x_target,
                              c.axis_x_lo_text, c.axis_x_hi_text,
                              c.axis_x_log, c.axis_x_n_text, true);
        draw_order_axis_block("axy", "Axis Y", s, c.axis_y_target,
                              c.axis_y_lo_text, c.axis_y_hi_text,
                              c.axis_y_log, c.axis_y_n_text, c.two_d);

        if (c.axis_x_target == kOrderTargetH && c.two_d && c.axis_y_target == kOrderTargetH)
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "обе оси свипают h");
        else if (c.two_d && c.axis_x_target == c.axis_y_target)
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "обе оси свипают одно и то же");
    }

    draw_named_num_fields("Initial conditions", s.vars, c.initial_conditions);
    draw_named_num_fields("Parameters", s.params, c.param_values,
        "Параметры МЕТОДА (для композиционных схем) объявляй здесь же обычными "
        "параметрами системы: в правых частях их можно не использовать, "
        "а тело кастомной КРС читает их как a[k]. В селекторе оси они "
        "появятся наравне с остальными.");

    if (!c.last_error.empty())
        draw_error_box("##order_err", c.last_error);
}
}
