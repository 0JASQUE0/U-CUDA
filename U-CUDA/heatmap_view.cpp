#include "heatmap_view.h"
#include "grid_snap.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

HeatmapView::~HeatmapView() {
    if (data_tex_) glDeleteTextures(1, &data_tex_);
}

void HeatmapView::ensure_tex(int w, int h) {
    if (w == tex_w_ && h == tex_h_ && data_tex_) return;
    if (data_tex_) { glDeleteTextures(1, &data_tex_); data_tex_ = 0; }
    glGenTextures(1, &data_tex_);
    glBindTexture(GL_TEXTURE_2D, data_tex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, nullptr);
    // Без интерполяции: каждая ячейка — параметр (i,j), визуальная честность
    // важнее гладкости.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    tex_w_ = w; tex_h_ = h;
}

void HeatmapView::upload_data(int nx, int ny, const double* values) {
    if (!data_tex_) return;
    upload_buf_.resize((size_t)nx * (size_t)ny);
    // diverged/spec-значения engine помечает как ±999; заменяем на FLT_MAX,
    // шейдер увидит >=1e30 и нарисует тёмно-серым (а не верхним концом
    // colormap'а).
    const double DIV_MARK = 999.0;
    for (size_t k = 0; k < upload_buf_.size(); ++k) {
        double v = values[k];
        if (!std::isfinite(v) || v == DIV_MARK || v == -DIV_MARK) {
            upload_buf_[k] = std::numeric_limits<float>::max();
        } else {
            upload_buf_[k] = (float)v;
        }
    }
    glBindTexture(GL_TEXTURE_2D, data_tex_);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nx, ny, GL_RED, GL_FLOAT,
                    upload_buf_.data());
    glBindTexture(GL_TEXTURE_2D, 0);
}

void HeatmapView::do_autofit(double lo_x, double hi_x, double lo_y, double hi_y) {
    x_axis.view_min = lo_x;
    x_axis.view_max = hi_x;
    y_axis.view_min = lo_y;
    y_axis.view_max = hi_y;
    view_valid = true;
}

// cmap_sample / HeatmapColormap перемещены в plot_renderer.h/.cpp —
// используется ещё для colored trajectory в plot_view_2d.cpp.

// ===========================================================================
// Общий colorbar (см. heatmap_view.h). Вынесен из HeatmapView::render, чтобы
// FastSync mode-0 рисовал ровно ту же шкалу, а не свою копию.
// ===========================================================================
std::vector<ColorbarTick> colorbar_ticks(float vmin, float vmax, int n_discrete) {
    std::vector<ColorbarTick> out;
    const double range = (double)vmax - (double)vmin;
    if (n_discrete > 0) {
        const double rvmin = std::round((double)vmin);
        const double rvmax = std::round((double)vmax);
        const bool integer_like = std::abs(rvmin - (double)vmin) < 1e-6
                               && std::abs(rvmax - (double)vmax) < 1e-6
                               && (int)std::lround(rvmax - rvmin) == n_discrete - 1;
        if (integer_like) {
            for (int k = 0; k < n_discrete; ++k)
                out.push_back({ (double)vmin + (double)k,
                                ((float)k + 0.5f) / (float)n_discrete });
        } else if (range > 0.0) {
            const double bs = range / (double)n_discrete;
            for (int k = 0; k < n_discrete; ++k)
                out.push_back({ (double)vmin + ((double)k + 0.5) * bs,
                                ((float)k + 0.5f) / (float)n_discrete });
        }
    } else if (range > 0.0) {
        const double step = nice_step(range, 5);
        if (step > 0.0) {
            const double start = std::ceil((double)vmin / step) * step;
            for (double v = start; v <= (double)vmax + step * 0.5; v += step)
                out.push_back({ v, (float)((v - (double)vmin) / range) });
        }
    }
    if (out.empty()) out.push_back({ (double)vmin, 0.5f });
    return out;
}

float colorbar_total_width(const std::vector<ColorbarTick>& ticks) {
    float max_tick_w = 0.0f;
    for (const auto& t : ticks)
        max_tick_w = std::max(max_tick_w, ImGui::CalcTextSize(fmt_tick(t.label).c_str()).x);
    return kColorbarWidth + kColorbarGap + kColorbarTickLen + kColorbarTextGap
           + max_tick_w + 6.0f;
}

void draw_colorbar(ImDrawList* dl, ImVec2 top_left, float height,
                   float vmin, float vmax, HeatmapColormap cmap,
                   bool reverse, int n_discrete,
                   const std::vector<ColorbarTick>& ticks) {
    if (!dl || height <= 0.0f) return;
    const float cb_x = top_left.x, cb_y = top_left.y, cb_h = height;
    const ImU32 col_text = plot_col_text();
    const float font_h   = ImGui::GetFontSize();

    // Градиент: непрерывный режим — 256 полос (визуально не отличимо от LUT),
    // discrete — по одной полосе на диапазон, чтобы совпадало с квантованием
    // на самой картинке.
    const int n_strips = (n_discrete > 0) ? n_discrete : 256;
    for (int i = 0; i < n_strips; ++i) {
        const float t0 = (float)i       / (float)n_strips;
        const float t1 = (float)(i + 1) / (float)n_strips;
        // Верх шкалы = vmax (t=1), низ = vmin (t=0).
        const float y0 = cb_y + cb_h * (1.0f - t1);
        const float y1 = cb_y + cb_h * (1.0f - t0);
        // В discrete-режиме сэмплим так же, как шейдер (edge-aligned
        // k/(N-1)), чтобы крайние полосы точно совпали с непрерывными
        // концами. В непрерывном — центр полосы.
        float t_samp;
        if (n_discrete > 0) t_samp = (n_discrete > 1) ? (float)i / (float)(n_discrete - 1) : 0.5f;
        else                t_samp = (t0 + t1) * 0.5f;
        const ImU32 col = cmap_sample(reverse ? (1.0f - t_samp) : t_samp, cmap);
        dl->AddRectFilled(ImVec2(cb_x, y0), ImVec2(cb_x + kColorbarWidth, y1), col);
    }
    dl->AddRect(ImVec2(cb_x, cb_y), ImVec2(cb_x + kColorbarWidth, cb_y + cb_h),
                plot_col_border());

    const double range = (double)vmax - (double)vmin;
    for (const auto& t : ticks) {
        if (range <= 0.0) {
            // Вырожденный vmin == vmax: всё равно печатаем одну подпись по центру.
            const float y = cb_y + cb_h * 0.5f;
            dl->AddText(ImVec2(cb_x + kColorbarWidth + kColorbarTickLen + kColorbarTextGap,
                               y - font_h * 0.5f),
                        col_text, fmt_tick(t.label).c_str());
            continue;
        }
        float frac = t.frac;
        if (frac < -1e-4f || frac > 1.0f + 1e-4f) continue;
        frac = std::min(std::max(frac, 0.0f), 1.0f);
        const float y = cb_y + cb_h * (1.0f - frac);
        dl->AddLine(ImVec2(cb_x + kColorbarWidth, y),
                    ImVec2(cb_x + kColorbarWidth + kColorbarTickLen, y), col_text);
        dl->AddText(ImVec2(cb_x + kColorbarWidth + kColorbarTickLen + kColorbarTextGap,
                           y - font_h * 0.5f),
                    col_text, fmt_tick(t.label).c_str());
    }
}

void HeatmapView::render(PlotRenderer& renderer,
                         ImVec2 block_origin, ImVec2 avail_size,
                         int owner_id,
                         int data_generation,
                         int nx, int ny,
                         const double* values,
                         double param_lo_x, double param_hi_x,
                         double param_lo_y, double param_hi_y,
                         double engine_vmin, double engine_vmax,
                         bool fit_request)
{
    if (nx <= 0 || ny <= 0 || !values) {
        ImGui::Dummy(avail_size);
        return;
    }

    // 0. Swap-axes. Транспонируем values + меняем nx/ny и param-ranges; всё
    // дальнейшее работает с этой "пост-swap" раскладкой как с исходной. При
    // переключении флага форсируем re-upload (другая раскладка пикселей) и
    // autofit (визуальный X теперь соответствует исходному data Y).
    if (swap_axes != swap_axes_cached_) {
        data_gen_cached = -1;
        view_valid = false;
        swap_axes_cached_ = swap_axes;
    }
    std::vector<double> swap_buf;
    const double* eff_values = values;
    if (swap_axes) {
        swap_buf.resize((size_t)nx * (size_t)ny);
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                swap_buf[(size_t)i * (size_t)ny + (size_t)j]
                    = values[(size_t)j * (size_t)nx + (size_t)i];
        eff_values = swap_buf.data();
        std::swap(nx, ny);
        std::swap(param_lo_x, param_lo_y);
        std::swap(param_hi_x, param_hi_y);
    }
    // Визуальные имена осей — каллер пишет в x_axis.name то, что относится
    // к "data X". При swap_axes визуальный горизонтальный (= "post-swap X")
    // = data Y → берём y_axis.name. И симметрично для визуального Y.
    const std::string& vis_x_name = swap_axes ? y_axis.name : x_axis.name;
    const std::string& vis_y_name = swap_axes ? x_axis.name : y_axis.name;

    // 1. Текстура из снапшота (lazy upload по generation).
    if (data_generation != data_gen_cached) {
        ensure_tex(nx, ny);
        upload_data(nx, ny, eff_values);
        data_gen_cached = data_generation;
    }

    // Применяем discrete_default один раз — на первом рендере с реальными
    // данными. Идёт после upload'а, чтобы если caller взвёл discrete_default
    // ЛЕНИВО (не в конструкторе), оно всё равно подхватилось. После apply
    // пользовательский toggle через popup работает как обычно.
    if (!discrete_default_applied_) {
        if (discrete_default) discrete = true;
        discrete_default_applied_ = true;
    }

    if (!view_valid || fit_request) {
        do_autofit(param_lo_x, param_hi_x, param_lo_y, param_hi_y);
    }

    // 2. Нормализация цвета — считаем до layout'а, чтобы заранее знать ширину
    //    подписей на colorbar'е и уместить их в margin_right.
    float vmin, vmax;
    if (autoscale) {
        vmin = (float)engine_vmin;
        vmax = (float)engine_vmax;
        if (vmax <= vmin) vmax = vmin + 1.0f;
    } else {
        vmin = manual_vmin;
        vmax = (manual_vmax > manual_vmin) ? manual_vmax : (manual_vmin + 1.0f);
    }
    shown_vmin = vmin;
    shown_vmax = vmax;

    // 3. Layout. margin_right считается динамически под фактическую ширину
    //    числовых подписей colorbar'а — иначе тики типа "1.234e-05" вылезают
    //    за пределы avail_size и обрезаются.
    const float margin_left   = 78.0f;
    const float margin_top    = 20.0f;
    const float margin_bottom = 46.0f;
    // Геометрия colorbar'а — kColorbar* в heatmap_view.h (шарится с FastSync).

    // Resolve the active number of discrete bands. discrete_levels overrides
    // auto-detection; otherwise span vmin..vmax inclusive at integer steps.
    int n_disc = 0;
    if (discrete) {
        if (discrete_levels > 0) n_disc = discrete_levels;
        else {
            int span = (int)std::lround((double)((double)vmax - (double)vmin)) + 1;
            n_disc = std::max(1, span);
        }
    }

    // Тики и ширина блока colorbar'а — общие хелперы (см. heatmap_view.h).
    const std::vector<ColorbarTick> tick_vals = colorbar_ticks(vmin, vmax, n_disc);
    const float margin_right = colorbar_total_width(tick_vals);

    int plot_w = std::max(64, (int)(avail_size.x - margin_left - margin_right));
    int plot_h = std::max(64, (int)(avail_size.y - margin_top - margin_bottom));

    ImGui::Dummy(avail_size);
    ImVec2 img_pos = ImVec2(block_origin.x + margin_left, block_origin.y + margin_top);

    // 4. Маппинг view → UV данных. data range — фиксированные границы из
    //    engine'а; view может быть произвольным после zoom/pan.
    double data_rx = param_hi_x - param_lo_x;
    double data_ry = param_hi_y - param_lo_y;
    if (std::abs(data_rx) < 1e-30) data_rx = 1.0;
    if (std::abs(data_ry) < 1e-30) data_ry = 1.0;
    double view_min_x = x_axis.view_min, view_max_x = x_axis.view_max;
    double view_min_y = y_axis.view_min, view_max_y = y_axis.view_max;

    // Node step (расстояние между соседними узлами). Values хранятся в N узлах
    // на data range [param_lo, param_hi]: node k в позиции param_lo + k*step.
    // Renderer рисует ровно N пикселей текстуры GL_NEAREST, значит визуально
    // ширина пикселя должна равняться step_node, а не data_range/N — иначе
    // тики на осях не выровнены по центрам пикселей и tooltip показывает
    // не тот узел, что визуально закрашен под курсором.
    //
    // Решение: рендерим текстуру с полупиксельным паддингом с каждой стороны
    // и оси/tooltip работают в этом "visual"-диапазоне. Внутреннее состояние
    // x_axis.view_min/max остаётся в node-координатах — pan/zoom/rect-zoom
    // конвертят между vis и node на входе/выходе.
    double step_x = (nx > 1) ? data_rx / (double)(nx - 1) : 0.0;
    double step_y = (ny > 1) ? data_ry / (double)(ny - 1) : 0.0;
    double vis_view_min_x = view_min_x - step_x * 0.5;
    double vis_view_max_x = view_max_x + step_x * 0.5;
    double vis_view_min_y = view_min_y - step_y * 0.5;
    double vis_view_max_y = view_max_y + step_y * 0.5;
    double vis_data_rx    = data_rx + step_x;   // == N*step_x при nx>1
    double vis_data_ry    = data_ry + step_y;
    double vis_param_lo_x = param_lo_x - step_x * 0.5;
    double vis_param_lo_y = param_lo_y - step_y * 0.5;

    // Эффективные визуальные границы с учётом AxisInfo::invert. evis_x0 —
    // значение у ЛЕВОГО края плота, evis_x1 — у правого; evis_y0 — у НИЖНЕГО,
    // evis_y1 — у верхнего. При invert границы меняются местами, span
    // становится отрицательным, и ВСЕ маппинги screen<->world ниже (UV,
    // курсор, тики, crosshair, rect-zoom, pan) разворачиваются сами. Это тот
    // же приём, что axis_effective() в plot_axis.h у Plot2DView — до этого
    // HeatmapView поля lock/invert общей структуры AxisInfo просто игнорировал.
    const double evis_x0 = x_axis.invert ? vis_view_max_x : vis_view_min_x;
    const double evis_x1 = x_axis.invert ? vis_view_min_x : vis_view_max_x;
    const double evis_y0 = y_axis.invert ? vis_view_max_y : vis_view_min_y;
    const double evis_y1 = y_axis.invert ? vis_view_min_y : vis_view_max_y;

    float uv_off_x   = (float)((evis_x0 - vis_param_lo_x) / vis_data_rx);
    float uv_scale_x = (float)((evis_x1 - evis_x0) / vis_data_rx);
    float uv_off_y   = (float)((evis_y0 - vis_param_lo_y) / vis_data_ry);
    float uv_scale_y = (float)((evis_y1 - evis_y0) / vis_data_ry);

    // 5. FBO render. (n_disc was resolved up-front in section 3.)
    {
        float br, bg, bb, ba;
        plot_bg_color(br, bg, bb, ba);
        renderer.begin_frame(plot_w, plot_h, br, bg, bb, ba);
    }
    renderer.draw_heatmap(data_tex_, vmin, vmax, (int)colormap,
                          uv_off_x, uv_off_y, uv_scale_x, uv_scale_y,
                          n_disc, reverse_colormap);
    renderer.end_frame();

    // 6. Вставка FBO-картинки. AddImage(uv_min, uv_max) — uv_min маппится в
    //    p_min (верхний-левый угол на экране). Чтобы row 0 текстуры (iy=0 =
    //    param_lo_y) оказалась внизу плота (мат. оси Y↑), берём uv_min=(0,1) и
    //    uv_max=(1,0) — экранный верх соответствует UV.y=1 (top FBO = param_hi_y).
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddImage((ImTextureID)(intptr_t)renderer.texture_id(),
                 img_pos, ImVec2(img_pos.x + plot_w, img_pos.y + plot_h),
                 ImVec2(0, 1), ImVec2(1, 0));

    // 7. Hit-test'ы — отдельные кнопки для плота и для зон осей. Pan по X / Y
    //    через drag ЛКМ в самой оси (как у Plot2DView), общий pan/zoom — ЛКМ
    //    drag / колесо в самом плоте, rect-zoom — drag ПКМ.
    char id_buf[64];

    ImGui::SetCursorScreenPos(img_pos);
    std::snprintf(id_buf, sizeof(id_buf), "##hm_plot_%d", owner_id);
    ImGui::InvisibleButton(id_buf, ImVec2((float)plot_w, (float)plot_h),
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    bool plot_hov = ImGui::IsItemHovered();
    bool plot_dbl = plot_hov && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);

    // Right-click context menu on the heatmap area: discrete-mode toggle.
    // Open manually (instead of BeginPopupContextItem) so a RMB drag — used
    // for rect-zoom — does NOT trigger the menu. Threshold is the same as
    // ImGui's mouse-drag threshold so the gesture matches the rest of the UI.
    char ctx_id[64];
    std::snprintf(ctx_id, sizeof(ctx_id), "##hm_ctx_%d", owner_id);
    if (plot_hov && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        ImVec2 d = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right, 0.0f);
        if (std::abs(d.x) + std::abs(d.y) < ImGui::GetIO().MouseDragThreshold) {
            ImGui::OpenPopup(ctx_id);
        }
    }
    if (ImGui::BeginPopup(ctx_id)) {
        // Блок осей — тот же набор, что в Plot2DView (см. plot_view_2d.cpp
        // «12. Rect-zoom + popup»). Раньше у хитмапы этих пунктов не было
        // вовсе: одна и та же диаграмма в 1D-режиме умела Lock/Invert/Auto fit,
        // а в 2D — нет. Auto fit намеренно игнорирует lock: явная команда из
        // меню сильнее защиты от случайного зума мышью (как у Plot2DView).
        if (ImGui::MenuItem("Auto fit (both)")) view_valid = false;
        if (ImGui::MenuItem("Auto fit X")) {
            x_axis.view_min = param_lo_x; x_axis.view_max = param_hi_x;
        }
        if (ImGui::MenuItem("Auto fit Y")) {
            y_axis.view_min = param_lo_y; y_axis.view_max = param_hi_y;
        }
        ImGui::Separator();
        ImGui::MenuItem("Lock X axis", nullptr, &x_axis.lock);
        ImGui::MenuItem("Lock Y axis", nullptr, &y_axis.lock);
        ImGui::Separator();
        ImGui::MenuItem("Invert X", nullptr, &x_axis.invert);
        ImGui::MenuItem("Invert Y", nullptr, &y_axis.invert);
        ImGui::Separator();
        ImGui::Checkbox("Discrete colorbar", &discrete);
        if (discrete) {
            ImGui::SetNextItemWidth(80.0f);
            ImGui::InputInt("Levels (0=auto)", &discrete_levels, 0, 0);
            if (discrete_levels < 0) discrete_levels = 0;
        }
        ImGui::Checkbox("Reverse colormap", &reverse_colormap);
        ImGui::Separator();
        if (ImGui::MenuItem("Copy image to clipboard")) {
            request_plot_screenshot(block_origin,
                ImVec2(block_origin.x + avail_size.x, block_origin.y + avail_size.y));
        }
        // Caller-injected items (e.g. "Export data..."). Mirrors the same
        // hook on Plot2DView so both view types share the right-click pattern.
        if (popup_extras) {
            ImGui::Separator();
            popup_extras();
        }
        ImGui::EndPopup();
    }

    ImGui::SetCursorScreenPos(ImVec2(img_pos.x, img_pos.y + plot_h));
    std::snprintf(id_buf, sizeof(id_buf), "##hm_xax_%d", owner_id);
    ImGui::InvisibleButton(id_buf, ImVec2((float)plot_w, margin_bottom),
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    bool xax_hov = ImGui::IsItemHovered();
    bool xax_dbl = xax_hov && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);

    ImGui::SetCursorScreenPos(ImVec2(img_pos.x - margin_left, img_pos.y));
    std::snprintf(id_buf, sizeof(id_buf), "##hm_yax_%d", owner_id);
    ImGui::InvisibleButton(id_buf, ImVec2(margin_left, (float)plot_h),
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    bool yax_hov = ImGui::IsItemHovered();
    bool yax_dbl = yax_hov && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);

    // Per-axis context menu на зонах осей — паритет с Plot2DView (там ПКМ по
    // оси даёт Auto fit / Lock / Invert только для неё). Открываем вручную, как
    // и меню плота: ПКМ-drag по оси — это rect-zoom по одной оси (mode 2/3), и
    // он не должен всплывать меню.
    {
        auto axis_menu = [&](bool hovered, const char* tag, AxisInfo& ax,
                             double fit_lo, double fit_hi) {
            char aid[64];
            std::snprintf(aid, sizeof(aid), "##hm_%s_menu_%d", tag, owner_id);
            if (hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
                ImVec2 d = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right, 0.0f);
                if (std::abs(d.x) + std::abs(d.y) < ImGui::GetIO().MouseDragThreshold)
                    ImGui::OpenPopup(aid);
            }
            if (ImGui::BeginPopup(aid)) {
                char lbl[64];
                std::snprintf(lbl, sizeof(lbl), "Auto fit %s", tag);
                if (ImGui::MenuItem(lbl)) { ax.view_min = fit_lo; ax.view_max = fit_hi; }
                ImGui::Separator();
                std::snprintf(lbl, sizeof(lbl), "Lock %s axis", tag);
                ImGui::MenuItem(lbl, nullptr, &ax.lock);
                std::snprintf(lbl, sizeof(lbl), "Invert %s", tag);
                ImGui::MenuItem(lbl, nullptr, &ax.invert);
                ImGui::EndPopup();
            }
        };
        axis_menu(xax_hov, "X", x_axis, param_lo_x, param_hi_x);
        axis_menu(yax_hov, "Y", y_axis, param_lo_y, param_hi_y);
    }

    ImGuiIO& io = ImGui::GetIO();
    // Хелпер: мировые координаты под текущей позицией курсора (учитывает Y↑).
    // Работает в vis-домене — визуально левый край плота соответствует
    // vis_view_min_x = view_min_x - step_x/2, чтобы центр пикселя визуально
    // совпадал с позицией узла. Для нижестоящих операций (rect-zoom, wheel-
    // zoom) это надо конвертить обратно в node-домен (view_min = vis - step/2
    // и симметрично для max) — там где обновляем x_axis.view_min/view_max.
    // Единственный маппинг экран → мир. До этого та же формула была
    // продублирована в четырёх местах (mouse_world, crosshair-drag, tooltip,
    // on_left_click), что делало добавление invert трудноуловимым.
    auto screen_to_world = [&](ImVec2 pos, double& wx, double& wy) {
        double tx = (double)(pos.x - img_pos.x) / (double)plot_w;
        double ty = 1.0 - (double)(pos.y - img_pos.y) / (double)plot_h;
        wx = evis_x0 + tx * (evis_x1 - evis_x0);
        wy = evis_y0 + ty * (evis_y1 - evis_y0);
    };
    auto mouse_world = [&](double& wx, double& wy) {
        screen_to_world(io.MousePos, wx, wy);
    };
    // Хелпер: клампим view в пределы расчётных данных. Применяем после ВСЕХ
    // изменений view (pan, wheel-zoom, rect-zoom) — пользователь не может
    // случайно «уплыть» в пустую область.
    auto clamp_view = [&]() {
        // Если view стал шире данных (zoom out больше data) — обрезаем строго
        // до диапазона данных.
        double rx = x_axis.view_max - x_axis.view_min;
        if (rx >= param_hi_x - param_lo_x) {
            x_axis.view_min = param_lo_x;
            x_axis.view_max = param_hi_x;
        } else {
            if (x_axis.view_min < param_lo_x) {
                x_axis.view_min = param_lo_x;
                x_axis.view_max = x_axis.view_min + rx;
            }
            if (x_axis.view_max > param_hi_x) {
                x_axis.view_max = param_hi_x;
                x_axis.view_min = x_axis.view_max - rx;
            }
        }
        double ry = y_axis.view_max - y_axis.view_min;
        if (ry >= param_hi_y - param_lo_y) {
            y_axis.view_min = param_lo_y;
            y_axis.view_max = param_hi_y;
        } else {
            if (y_axis.view_min < param_lo_y) {
                y_axis.view_min = param_lo_y;
                y_axis.view_max = y_axis.view_min + ry;
            }
            if (y_axis.view_max > param_hi_y) {
                y_axis.view_max = param_hi_y;
                y_axis.view_min = y_axis.view_max - ry;
            }
        }
    };

    // Снапшот заблокированных осей: AxisInfo::lock означает «интеракция эту ось
    // не двигает». Проще и надёжнее один раз восстановить её после ВСЕХ мутаций
    // (wheel / pan / rect-zoom / double-click), чем расставлять проверки в
    // каждой ветке. Восстановление — перед clamp_view() ниже.
    const double lock_x_min = x_axis.view_min, lock_x_max = x_axis.view_max;
    const double lock_y_min = y_axis.view_min, lock_y_max = y_axis.view_max;

    // 8a. Wheel zoom (вокруг курсора) — оставляем только в плоте.
    if (plot_hov && io.MouseWheel != 0.0f) {
        float zoom_factor = (io.MouseWheel > 0) ? 1.0f / 1.2f : 1.2f;
        double tx = (double)(io.MousePos.x - img_pos.x) / (double)plot_w;
        double ty = 1.0 - (double)(io.MousePos.y - img_pos.y) / (double)plot_h;
        // Zoom вокруг курсора в vis-домене, чтобы пиксель под курсором
        // визуально оставался под ним. Затем возвращаемся в node-домен для
        // обновления x_axis.view_min/view_max. Работаем в effective-границах,
        // поэтому при invert зум так же остаётся «вокруг курсора».
        double world_x = evis_x0 + tx * (evis_x1 - evis_x0);
        double world_y = evis_y0 + ty * (evis_y1 - evis_y0);
        double new_rx = (evis_x1 - evis_x0) * (double)zoom_factor;
        double new_ry = (evis_y1 - evis_y0) * (double)zoom_factor;
        double a_x = world_x - tx * new_rx, b_x = a_x + new_rx;
        double a_y = world_y - ty * new_ry, b_y = a_y + new_ry;
        // node-домен всегда min<max (invert живёт только в маппинге, не в
        // хранимом view), поэтому нормализуем порядок.
        x_axis.view_min = std::min(a_x, b_x) + step_x * 0.5;
        x_axis.view_max = std::max(a_x, b_x) - step_x * 0.5;
        y_axis.view_min = std::min(a_y, b_y) + step_y * 0.5;
        y_axis.view_max = std::max(a_y, b_y) - step_y * 0.5;
    }

    // 8b. Pan ЛКМ — в плоте по обеим осям, в оси — только этой оси.
    // Delta считаем в vis-домене (тогда 1 экранный пиксель → сдвиг ровно
    // на один визуальный пиксель). Сама операция — симметричный сдвиг
    // view_min/max, поэтому вис-и-нод-домен смещаются одинаково.
    // Crosshair drag gestures (Custom-tab): MMB drag OR Shift+LMB drag →
    // move fix_x/y crosshair. Plain LMB drag (no Shift) stays a pan gesture,
    // same as Parametric, so a zoomed-in view can still be panned. The heavy
    // recompute is deferred to release (see on_left_click block below).
    {
        ImGuiIO& io = ImGui::GetIO();
        bool crosshair_gesture = on_left_drag && step_x > 0.0 && step_y > 0.0 &&
            (ImGui::IsMouseDown(ImGuiMouseButton_Middle) ||
             (ImGui::IsMouseDown(ImGuiMouseButton_Left) && io.KeyShift));
        if (plot_hov && crosshair_gesture) {
            double dx, dy; mouse_world(dx, dy);
            int ix = (int)std::floor((dx - vis_param_lo_x) / step_x);
            int iy = (int)std::floor((dy - vis_param_lo_y) / step_y);
            if (ix >= 0 && ix < nx && iy >= 0 && iy < ny) {
                double snap_x = param_lo_x + (double)ix * step_x;
                double snap_y = param_lo_y + (double)iy * step_y;
                on_left_drag(ix, iy, snap_x, snap_y);
            }
            // Consume any accumulated LMB drag delta so the pan branch below
            // (if it fires on Shift+LMB) stays a no-op.
            ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
        }
        // LMB pan — as before, unless Shift is held (that's a crosshair
        // gesture). Applies regardless of on_left_drag (Parametric parity).
        // Span берём в effective-границах: при invert он отрицательный, и
        // направление pan'а разворачивается вместе с картинкой.
        if (plot_hov && !io.KeyShift &&
            ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
            ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.0f);
            double dwx = -(double)delta.x / (double)plot_w * (evis_x1 - evis_x0);
            double dwy =  (double)delta.y / (double)plot_h * (evis_y1 - evis_y0);
            x_axis.view_min += dwx;  x_axis.view_max += dwx;
            y_axis.view_min += dwy;  y_axis.view_max += dwy;
            ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
        }
    }
    if (xax_hov && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
        ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.0f);
        double dwx = -(double)delta.x / (double)plot_w * (evis_x1 - evis_x0);
        x_axis.view_min += dwx;  x_axis.view_max += dwx;
        ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
    }
    if (yax_hov && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
        ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.0f);
        double dwy =  (double)delta.y / (double)plot_h * (evis_y1 - evis_y0);
        y_axis.view_min += dwy;  y_axis.view_max += dwy;
        ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
    }

    // 8c. Rect-zoom ПКМ. Начало drag'а — запоминаем стартовую точку в мире и
    //    режим (плот / X / Y). На release — назначаем новый view по выделенной
    //    зоне. Во время drag'а — рисуем рамку (внизу, см. ниже).
    // wx/wy приходят из mouse_world в vis-домене → на release конвертим в
    // node-домен для x_axis.view_min/max: view = vis + step/2 для min и
    // view = vis - step/2 для max.
    bool rmb_down = ImGui::IsMouseDown(ImGuiMouseButton_Right);
    if (rect_zoom_mode_ == 0 && rmb_down) {
        double wx, wy; mouse_world(wx, wy);
        if (plot_hov)      { rect_zoom_mode_ = 1; rect_zoom_x0_ = wx; rect_zoom_y0_ = wy; }
        else if (xax_hov)  { rect_zoom_mode_ = 2; rect_zoom_x0_ = wx; }
        else if (yax_hov)  { rect_zoom_mode_ = 3; rect_zoom_y0_ = wy; }
    }
    if (rect_zoom_mode_ != 0 && !rmb_down) {
        double wx, wy; mouse_world(wx, wy);
        // Применяем только если выделение не вырожденное (не точка).
        if (rect_zoom_mode_ == 1) {
            double xa = std::min(rect_zoom_x0_, wx), xb = std::max(rect_zoom_x0_, wx);
            double ya = std::min(rect_zoom_y0_, wy), yb = std::max(rect_zoom_y0_, wy);
            if (xb - xa > 1e-12 && yb - ya > 1e-12) {
                x_axis.view_min = xa + step_x * 0.5; x_axis.view_max = xb - step_x * 0.5;
                y_axis.view_min = ya + step_y * 0.5; y_axis.view_max = yb - step_y * 0.5;
            }
        } else if (rect_zoom_mode_ == 2) {
            double xa = std::min(rect_zoom_x0_, wx), xb = std::max(rect_zoom_x0_, wx);
            if (xb - xa > 1e-12) {
                x_axis.view_min = xa + step_x * 0.5; x_axis.view_max = xb - step_x * 0.5;
            }
        } else if (rect_zoom_mode_ == 3) {
            double ya = std::min(rect_zoom_y0_, wy), yb = std::max(rect_zoom_y0_, wy);
            if (yb - ya > 1e-12) {
                y_axis.view_min = ya + step_y * 0.5; y_axis.view_max = yb - step_y * 0.5;
            }
        }
        rect_zoom_mode_ = 0;
    }

    // 8d. Double-click — fit. В плоте обе оси, в оси только эта.
    if (plot_dbl)      do_autofit(param_lo_x, param_hi_x, param_lo_y, param_hi_y);
    else if (xax_dbl) { x_axis.view_min = param_lo_x; x_axis.view_max = param_hi_x; }
    else if (yax_dbl) { y_axis.view_min = param_lo_y; y_axis.view_max = param_hi_y; }

    // 8d-bis. Восстановление заблокированных осей — отменяет всё, что могли
    // изменить wheel / pan / rect-zoom / double-click выше (см. снапшот).
    if (x_axis.lock) { x_axis.view_min = lock_x_min; x_axis.view_max = lock_x_max; }
    if (y_axis.lock) { y_axis.view_min = lock_y_min; y_axis.view_max = lock_y_max; }

    // 8e. После всех изменений view — клампим в пределы данных. Render с
    //    новыми UV произойдёт в следующем кадре; для интерактивности 1-кадра
    //    задержки незаметна, а tick-labels/tooltip ниже уже читают из
    //    x_axis/y_axis напрямую и показывают актуальное состояние.
    clamp_view();

    // Гибрид тиков: без зума (view == весь диапазон данных) показываем
    // «красивые» круглые числа обычным nice_step — они почти всегда совпадают
    // с границами данных (пользователь сам их выбирает), выглядят привычно и
    // не зависят от того, насколько «неровно» N делит диапазон. При зуме
    // отдельные узлы становятся визуально различимы (один узел — несколько
    // экранных пикселей), и тики привязываем к сетке, чтобы не «плавали»
    // между цветовыми полосами. clamp_view() выше форсит view_min/max в ТОЧНОЕ
    // равенство param_lo/hi, когда пользователь зумит дальше данных — поэтому
    // сравнение на равенство (без эпсилон) надёжно отличает «без зума».
    bool x_full_view = (nx <= 1) || (x_axis.view_min == param_lo_x && x_axis.view_max == param_hi_x);
    bool y_full_view = (ny <= 1) || (y_axis.view_min == param_lo_y && y_axis.view_max == param_hi_y);

    // 9. Тики осей БЕЗ grid-сетки: короткие штрихи 5px за пределами плота +
    //    числовые подписи. Без линий через весь плот — они отвлекают от
    //    цветового поля.
    ImU32 col_text = plot_col_text();
    ImU32 col_axis = plot_col_border();
    // Tick'и: формула числа тиков и проверки overshoot/clip — те же, что в
    // plot_axis.cpp (draw_axis_x_grid/y_grid), но без сетки через плот.
    // Хелпер: собирает список tick-значений в [lo, hi] с nice-step, округлённым
    // к кратному step_node (когда step_node > 0 и n_nodes > 1), плюс форс-
    // включение крайних узлов (param_lo / param_hi), если они в кадре и не
    // слишком близко к соседнему тику. Так, во-первых, тики всегда падают на
    // узлы (нет промежуточных значений между цветовыми пикселями), и во-вторых
    // крайнее значение диапазона всегда отрисовано (напр., "30" не пропадает
    // при выборе mult, не делящего N-1).
    auto compute_axis_ticks = [](double lo, double hi, int target_count,
                            double step_node, double node_origin, int n_nodes,
                            double force_lo, double force_hi)
                        -> std::vector<double>
    {
        std::vector<double> out;
        double vr = hi - lo;
        if (std::abs(vr) < 1e-30) return out;
        double sx = nice_step(std::abs(vr), target_count);
        double xstart;
        if (step_node > 0.0 && n_nodes > 1) {
            int mult = (int)std::lround(sx / step_node);
            if (mult < 1) mult = 1;
            sx = (double)mult * step_node;
            int k_lo = (int)std::ceil((lo - node_origin) / step_node - 1e-9);
            int k_start = (int)std::ceil((double)k_lo / (double)mult - 1e-9) * mult;
            xstart = node_origin + (double)k_start * step_node;
        } else {
            xstart = std::ceil(lo / sx) * sx;
        }
        int nt = (int)std::floor((hi - xstart) / sx + 1e-9) + 1;
        if (nt < 0) nt = 0;
        for (int i = 0; i < nt; ++i) {
            double v = xstart + i * sx;
            if (v > hi + sx * 1e-6 || v < lo - sx * 1e-6) continue;
            out.push_back(v);
        }
        // Форс-включение крайних узлов сетки (snap-режим при зуме). Порог "не
        // слишком близко" = 40% от sx — подписи не будут наезжать друг на
        // друга при обычных диапазонах.
        const double gap_min = sx * 0.4;
        if (step_node > 0.0 && n_nodes > 1) {
            double first_node = node_origin;
            double last_node  = node_origin + (double)(n_nodes - 1) * step_node;
            if (last_node >= lo - step_node * 0.5 && last_node <= hi + step_node * 0.5) {
                if (out.empty() || last_node - out.back() >= gap_min) {
                    out.push_back(last_node);
                }
            }
            if (first_node >= lo - step_node * 0.5 && first_node <= hi + step_node * 0.5) {
                if (out.empty() || out.front() - first_node >= gap_min) {
                    out.insert(out.begin(), first_node);
                }
            }
        }
        // Форс-включение истинных границ диапазона (force_lo/force_hi) --
        // ВСЕГДА, а не только в snap-режиме. Без этого край мог пропасть,
        // если он не кратен "красивому" nice_step шагу (напр. 0.001 при шаге
        // 0.005) -- баг, репортнутый для h-свипа в линейном масштабе.
        if (force_hi >= lo - 1e-12 && force_hi <= hi + 1e-12) {
            if (out.empty() || force_hi - out.back() >= gap_min) out.push_back(force_hi);
        }
        if (force_lo >= lo - 1e-12 && force_lo <= hi + 1e-12) {
            if (out.empty() || out.front() - force_lo >= gap_min) out.insert(out.begin(), force_lo);
        }
        return out;
    };

    auto draw_x_ticks = [&]() {
        // vis_view_min/max_x расширяют view на полшага ЛИНЕЙНОЙ сетки, чтобы
        // цветовые ячейки центрировались на узлах. Для log-оси линейный
        // полушаг у границ (напр. 0.001) даёт заметный сдвиг подписи --
        // берём точный view_min/max без этого паддинга.
        // invert: границы уже переставлены в evis_x0/x1, для log-оси делаем то
        // же вручную (там паддинг не применяется).
        double emin = x_axis.log_scale
                      ? (x_axis.invert ? x_axis.view_max : x_axis.view_min) : evis_x0;
        double emax = x_axis.log_scale
                      ? (x_axis.invert ? x_axis.view_min : x_axis.view_max) : evis_x1;
        double vrx = emax - emin;
        if (std::abs(vrx) < 1e-30) return;
        double lo = std::min(emin, emax), hi = std::max(emin, emax);
        // Log-масштаб: см. plot_axis.cpp draw_axis_x_grid -- нет настоящей
        // лог-оси, поэтому вместо "красивых" линейных тиков (которые
        // подписали бы значения, никогда не просимулированные) рисуем
        // только границы текущего view.
        std::vector<double> ticks = x_axis.log_scale
            ? std::vector<double>{ lo, hi }
            : (x_full_view ? compute_axis_ticks(lo, hi, 8, 0.0, 0.0, 0, param_lo_x, param_hi_x)
                            : compute_axis_ticks(lo, hi, 8, step_x, param_lo_x, nx, param_lo_x, param_hi_x));
        for (double xv : ticks) {
            float px = img_pos.x + (float)((xv - emin) / vrx) * plot_w;
            dl->AddLine(ImVec2(px, img_pos.y + plot_h),
                        ImVec2(px, img_pos.y + plot_h + 5.0f), col_axis, 1.0f);
            std::string lbl = fmt_tick(xv);
            ImVec2 ts = ImGui::CalcTextSize(lbl.c_str());
            dl->AddText(ImVec2(px - ts.x * 0.5f, img_pos.y + plot_h + 7.0f),
                        col_text, lbl.c_str());
        }
    };
    auto draw_y_ticks = [&]() {
        // См. draw_x_ticks выше про vis_view-паддинг и log-масштаб.
        // См. draw_x_ticks про invert.
        double emin = y_axis.log_scale
                      ? (y_axis.invert ? y_axis.view_max : y_axis.view_min) : evis_y0;
        double emax = y_axis.log_scale
                      ? (y_axis.invert ? y_axis.view_min : y_axis.view_max) : evis_y1;
        double vry = emax - emin;
        if (std::abs(vry) < 1e-30) return;
        double lo = std::min(emin, emax), hi = std::max(emin, emax);
        // См. draw_x_ticks выше.
        std::vector<double> ticks = y_axis.log_scale
            ? std::vector<double>{ lo, hi }
            : (y_full_view ? compute_axis_ticks(lo, hi, 6, 0.0, 0.0, 0, param_lo_y, param_hi_y)
                            : compute_axis_ticks(lo, hi, 6, step_y, param_lo_y, ny, param_lo_y, param_hi_y));
        for (double yv : ticks) {
            float py = img_pos.y + (float)((emax - yv) / vry) * plot_h;
            dl->AddLine(ImVec2(img_pos.x - 5.0f, py),
                        ImVec2(img_pos.x,         py), col_axis, 1.0f);
            std::string lbl = fmt_tick(yv);
            ImVec2 ts = ImGui::CalcTextSize(lbl.c_str());
            dl->AddText(ImVec2(img_pos.x - 8.0f - ts.x, py - ts.y * 0.5f),
                        col_text, lbl.c_str());
        }
    };
    draw_x_ticks();
    draw_y_ticks();
    dl->AddRect(img_pos, ImVec2(img_pos.x + plot_w, img_pos.y + plot_h),
                plot_col_border(), 0.0f, 0, 1.0f);

    // Визуальная рамка rect-zoom во время drag'а ПКМ. Для оси (mode 2/3) —
    // полоса на всю ширину/высоту плота. Координаты в мире → экран.
    if (rect_zoom_mode_ != 0 && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
        double cur_wx, cur_wy; mouse_world(cur_wx, cur_wy);
        auto W2S_x = [&](double w) {
            return img_pos.x + (float)((w - evis_x0) / (evis_x1 - evis_x0)) * plot_w;
        };
        auto W2S_y = [&](double w) {
            return img_pos.y + (float)((evis_y1 - w) / (evis_y1 - evis_y0)) * plot_h;
        };
        ImU32 col_fill = IM_COL32(255, 220, 80, 40);
        ImU32 col_edge = IM_COL32(255, 220, 80, 200);
        if (rect_zoom_mode_ == 1) {
            float x0 = W2S_x(rect_zoom_x0_), x1 = W2S_x(cur_wx);
            float y0 = W2S_y(rect_zoom_y0_), y1 = W2S_y(cur_wy);
            ImVec2 a(std::min(x0, x1), std::min(y0, y1));
            ImVec2 b(std::max(x0, x1), std::max(y0, y1));
            dl->AddRectFilled(a, b, col_fill);
            dl->AddRect(a, b, col_edge);
        } else if (rect_zoom_mode_ == 2) {
            float x0 = W2S_x(rect_zoom_x0_), x1 = W2S_x(cur_wx);
            ImVec2 a(std::min(x0, x1), img_pos.y);
            ImVec2 b(std::max(x0, x1), img_pos.y + plot_h);
            dl->AddRectFilled(a, b, col_fill);
            dl->AddRect(a, b, col_edge);
        } else if (rect_zoom_mode_ == 3) {
            float y0 = W2S_y(rect_zoom_y0_), y1 = W2S_y(cur_wy);
            ImVec2 a(img_pos.x,         std::min(y0, y1));
            ImVec2 b(img_pos.x + plot_w, std::max(y0, y1));
            dl->AddRectFilled(a, b, col_fill);
            dl->AddRect(a, b, col_edge);
        }
    }

    // Crosshair overlay — thin lines at world coords (crosshair_x, crosshair_y).
    // NaN disables the corresponding axis; both NaN → nothing drawn (zero cost).
    // Used by the Custom-tab to visualise fix_x/fix_y slider positions.
    if (std::isfinite(crosshair_x) || std::isfinite(crosshair_y)) {
        // Double-stroke: dark halo (3px) + coloured core (1.5px). The halo
        // keeps the line visible on any colormap band; the core encodes
        // which sweep-axis the line belongs to (see crosshair_*_color).
        ImU32 col_halo = IM_COL32(0, 0, 0, 220);
        double range_x = evis_x1 - evis_x0;
        double range_y = evis_y1 - evis_y0;
        if (std::isfinite(crosshair_x) && std::abs(range_x) > 1e-30
            && crosshair_x >= std::min(evis_x0, evis_x1)
            && crosshair_x <= std::max(evis_x0, evis_x1)) {
            float px = img_pos.x + (float)((crosshair_x - evis_x0) / range_x) * plot_w;
            ImVec2 a(px, img_pos.y), b(px, img_pos.y + plot_h);
            dl->AddLine(a, b, col_halo, 3.0f);
            dl->AddLine(a, b, (ImU32)crosshair_x_color, 1.5f);
        }
        if (std::isfinite(crosshair_y) && std::abs(range_y) > 1e-30
            && crosshair_y >= std::min(evis_y0, evis_y1)
            && crosshair_y <= std::max(evis_y0, evis_y1)) {
            float py = img_pos.y + (float)((evis_y1 - crosshair_y) / range_y) * plot_h;
            ImVec2 a(img_pos.x, py), b(img_pos.x + plot_w, py);
            dl->AddLine(a, b, col_halo, 3.0f);
            dl->AddLine(a, b, (ImU32)crosshair_y_color, 1.5f);
        }
    }

    // 8. Названия осей (как в Plot2DView). Используем vis_x_name/vis_y_name —
    // они учитывают swap_axes без мутации x_axis.name / y_axis.name.
    const char* xl = vis_x_name.empty() ? "x" : vis_x_name.c_str();
    const char* yl = vis_y_name.empty() ? "y" : vis_y_name.c_str();
    float font_h = ImGui::GetFontSize();
    ImVec2 xs = ImGui::CalcTextSize(xl);
    float x_label_y = img_pos.y + plot_h + 2.0f + font_h + 6.0f;
    dl->AddText(ImVec2(img_pos.x + (plot_w - xs.x) * 0.5f, x_label_y), col_text, xl);

    // Y-метка — повёрнута на -90° (читается снизу вверх, mathematical
    // convention). Рендерим горизонтально через AddText, затем поворачиваем
    // все добавленные вершины вокруг pivot. На ТОЧНО -90° матрица имеет
    // целочисленные компоненты (cos=0, sin=-1), пиксельная сетка глифов
    // сохраняется → шрифт остаётся чётким (AA-шум бывает только на
    // произвольных углах). X-позиция считается ДИНАМИЧЕСКИ за самыми
    // широкими тиками, иначе подпись наезжает на длинные числа.
    ImVec2 ts_yl = ImGui::CalcTextSize(yl);
    if (ts_yl.x > 0.0f && ts_yl.y > 0.0f) {
        float max_tick_w = 0.0f;
        // Считаем через тот же compute_axis_ticks, что и draw_y_ticks — иначе
        // при snap-to-node фактические подписи ("-19.8", "10.8") оказываются
        // длиннее оценки по nice_step ("-20", "10") и Y-название наезжает
        // на цифры. Работает в vis-домене (как и сам рендер тиков).
        double vry_v = vis_view_max_y - vis_view_min_y;
        if (std::abs(vry_v) >= 1e-30) {
            double lo = std::min(vis_view_min_y, vis_view_max_y);
            double hi = std::max(vis_view_min_y, vis_view_max_y);
            auto ticks = y_full_view
                ? compute_axis_ticks(lo, hi, 6, 0.0, 0.0, 0, param_lo_y, param_hi_y)
                : compute_axis_ticks(lo, hi, 6, step_y, param_lo_y, ny, param_lo_y, param_hi_y);
            for (double yv : ticks) {
                std::string tl = fmt_tick(yv);
                float w = ImGui::CalcTextSize(tl.c_str()).x;
                if (w > max_tick_w) max_tick_w = w;
            }
        }
        // Pivot — позиция левого-верхнего угла ДО поворота, она же центр
        // вращения. После -90° (dx, dy) ↦ (dy, -dx):
        //   • по X текст займёт [pivot.x, pivot.x + ts_yl.y]
        //   • по Y — [pivot.y - ts_yl.x, pivot.y]
        // Снапим к целым пикселям для чёткости глифов.
        // label_gap — доп. зазор между самым широким тиком и Y-подписью:
        // без него текст вплотную касается цифр (оба span'а стыкуются в одной
        // точке img_pos.x - max_tick_w - 8).
        const float label_gap = 6.0f;
        float pivot_x = std::floor(img_pos.x - max_tick_w - 8.0f - label_gap - ts_yl.y);
        float pivot_y = std::floor(img_pos.y + (plot_h + ts_yl.x) * 0.5f);
        ImVec2 pivot(pivot_x, pivot_y);

        int idx_start = dl->VtxBuffer.Size;
        dl->AddText(pivot, col_text, yl);
        int idx_end = dl->VtxBuffer.Size;
        for (int i = idx_start; i < idx_end; ++i) {
            ImDrawVert& v = dl->VtxBuffer[i];
            float dx = v.pos.x - pivot.x;
            float dy = v.pos.y - pivot.y;
            v.pos.x = pivot.x + dy;
            v.pos.y = pivot.y - dx;
        }
    }

    // 9. Colorbar справа — общая реализация (см. draw_colorbar). tick_vals те
    //    же, по которым выше зарезервирован margin_right.
    draw_colorbar(dl, ImVec2(img_pos.x + plot_w + kColorbarGap, img_pos.y),
                  (float)plot_h, vmin, vmax, colormap,
                  reverse_colormap, n_disc, tick_vals);

    // 10. Hover-tooltip: (p1, p2, λ) по позиции курсора.
    if (plot_hov && step_x > 0.0 && step_y > 0.0) {
        // Курсор в vis-домене: тот же маппинг, что у UV/ticks выше (учитывает
        // invert, см. screen_to_world).
        double dx, dy; mouse_world(dx, dy);
        // Индекс пикселя = floor((dx - vis_param_lo) / step). Пиксель k
        // визуально центрирован на позиции узла param_lo + k*step. Показываем
        // именно эту (узловую) координату и значение узла k.
        int ix = (int)std::floor((dx - vis_param_lo_x) / step_x);
        int iy = (int)std::floor((dy - vis_param_lo_y) / step_y);
        if (ix >= 0 && ix < nx && iy >= 0 && iy < ny) {
            // log_scale: узлы движка лежат на лог-равномерной сетке
            // (getValueByIdx_log), не на линейной param_lo + ix*step — см.
            // draw_x_ticks/draw_y_ticks выше про ту же лог-специфику. Доп.
            // guard >0 — live-чекбокс может быть включён при param_lo/hi<=0
            // (ещё не запускали Run) — log10(0)=-inf даёт NaN дальше по
            // формуле, деградируем на линейную ноду вместо NaN в tooltip'е.
            double snap_x = (x_axis.log_scale && param_lo_x > 0.0 && param_hi_x > 0.0)
                ? std::pow(10.0, std::log10(param_lo_x)
                    + (double)ix * (std::log10(param_hi_x) - std::log10(param_lo_x)) / (double)(nx - 1))
                : param_lo_x + (double)ix * step_x;
            double snap_y = (y_axis.log_scale && param_lo_y > 0.0 && param_hi_y > 0.0)
                ? std::pow(10.0, std::log10(param_lo_y)
                    + (double)iy * (std::log10(param_hi_y) - std::log10(param_lo_y)) / (double)(ny - 1))
                : param_lo_y + (double)iy * step_y;
            double v = eff_values[(size_t)iy * (size_t)nx + (size_t)ix];
            const char* xn = vis_x_name.empty() ? "x" : vis_x_name.c_str();
            const char* yn = vis_y_name.empty() ? "y" : vis_y_name.c_str();
            ImGui::BeginTooltip();
            if (!std::isfinite(v) || v == 999.0 || v == -999.0) {
                ImGui::Text("%s = %.6g\n%s = %.6g\nlambda: diverged", xn, snap_x, yn, snap_y);
            } else {
                ImGui::Text("%s = %.6g\n%s = %.6g\nlambda = %.6g", xn, snap_x, yn, snap_y, v);
            }
            ImGui::EndTooltip();
        }
    }

    // 10b. Drill-down / crosshair-commit callback. Fires ONLY on release
    // of a crosshair gesture (MMB or Shift+LMB) — the drag itself was
    // live-crosshair, release means "commit + recompute". Plain LMB is
    // reserved for pan; a bare LMB click no longer moves the crosshair
    // (that was surprising when just clicking to focus the window).
    if (plot_hov && on_left_click && step_x > 0.0 && step_y > 0.0
        && !ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        ImGuiIO& io = ImGui::GetIO();
        bool trigger = false;
        // Shift+LMB release — modifier gesture.
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && io.KeyShift)
            trigger = true;
        // MMB release — end of middle-button crosshair drag.
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Middle))
            trigger = true;
        if (trigger) {
            double dx, dy; mouse_world(dx, dy);
            int ix = (int)std::floor((dx - vis_param_lo_x) / step_x);
            int iy = (int)std::floor((dy - vis_param_lo_y) / step_y);
            if (ix >= 0 && ix < nx && iy >= 0 && iy < ny) {
                double snap_x = param_lo_x + (double)ix * step_x;
                double snap_y = param_lo_y + (double)iy * step_y;
                on_left_click(ix, iy, snap_x, snap_y);
            }
        }
    }

    // 11. Двойной клик внутри плота — autofit (повторно). Сидит на отдельной
    // ветке кода чтобы не мешать tooltip.
    if (plot_hov && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        do_autofit(param_lo_x, param_hi_x, param_lo_y, param_hi_y);
    }
}
