#include "heatmap_view.h"
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

// ---- C++-side colormap для colorbar'а (рисуется через ImDrawList) ----
namespace {
struct vec3f { float r, g, b; };
inline vec3f operator+(vec3f a, vec3f b) { return {a.r+b.r, a.g+b.g, a.b+b.b}; }
inline vec3f operator*(vec3f a, float s) { return {a.r*s, a.g*s, a.b*s}; }
inline vec3f operator*(float s, vec3f a) { return a*s; }

vec3f cmap_viridis(float t) {
    const vec3f c0 = {0.2777273272f, 0.0054872578f, 0.3340998020f};
    const vec3f c1 = {0.1057220655f, 1.4046380960f, 1.3845030177f};
    const vec3f c2 = {-0.330001533f, 0.214825727f, 0.092491715f};
    const vec3f c3 = {-4.634230600f, -5.799101469f, -19.33244091f};
    const vec3f c4 = {6.228269936f, 14.17993089f, 56.69055318f};
    const vec3f c5 = {4.776384997f, -13.74514904f, -65.35303153f};
    const vec3f c6 = {-5.435455319f, 4.645852612f, 26.31243947f};
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}
vec3f cmap_inferno(float t) {
    const vec3f c0 = {0.0002189403691f, 0.001651742368f, -0.01948089833f};
    const vec3f c1 = {0.1065134194f, 0.5639564368f, 3.932712388f};
    const vec3f c2 = {11.60249308f, -3.972853966f, -15.94239411f};
    const vec3f c3 = {-41.70399613f, 17.43639888f, 44.35414519f};
    const vec3f c4 = {77.16289500f, -33.40998897f, -81.80741196f};
    const vec3f c5 = {-71.31942380f, 32.62606027f, 73.20951466f};
    const vec3f c6 = {25.13112622f, -12.24266895f, -23.07032500f};
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}
vec3f cmap_turbo(float t) {
    // Canonical Turbo (matplotlib polynomial fit). See plot_renderer.cpp for
    // the matching GLSL version. Mirrors the shader bit-for-bit so colorbar
    // strips agree with on-screen heatmap colors.
    const vec3f c0 = {0.13572138f, 0.09140261f, 0.10667330f};
    const vec3f c1 = {4.61539260f, 2.19418839f, 12.64194608f};
    const vec3f c2 = {-42.66032258f, 4.84296658f, -60.58204836f};
    const vec3f c3 = {132.13108234f, -14.18503333f, 110.36276771f};
    const vec3f c4 = {-152.94239396f, 4.27729857f, -89.90310912f};
    const vec3f c5 = {59.28637943f, 2.82956604f, 27.34824973f};
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*c5))));
}
vec3f cmap_gray(float t) { return {t, t, t}; }

ImU32 cmap_sample(float t, HeatmapColormap m) {
    t = std::min(std::max(t, 0.0f), 1.0f);
    vec3f c;
    switch (m) {
        case HeatmapColormap::Inferno: c = cmap_inferno(t); break;
        case HeatmapColormap::Turbo:   c = cmap_turbo(t);   break;
        case HeatmapColormap::Gray:    c = cmap_gray(t);    break;
        case HeatmapColormap::Viridis:
        default:                       c = cmap_viridis(t); break;
    }
    auto clamp01 = [](float v){ return std::min(std::max(v, 0.0f), 1.0f); };
    return IM_COL32((int)(clamp01(c.r) * 255.0f),
                    (int)(clamp01(c.g) * 255.0f),
                    (int)(clamp01(c.b) * 255.0f), 255);
}
} // namespace

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

    // 1. Текстура из снапшота (lazy upload по generation).
    if (data_generation != data_gen_cached) {
        ensure_tex(nx, ny);
        upload_data(nx, ny, values);
        data_gen_cached = data_generation;
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
    const float colorbar_w    = 18.0f;
    const float colorbar_gap  = 12.0f;
    const float tick_len      = 4.0f;
    const float tick_text_gap = 2.0f;

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

    // Colorbar tick: `label` is what's printed, `frac` is the 0..1 position
    // along the bar (0 = vmin / bottom, 1 = vmax / top). They're decoupled
    // because in discrete-auto mode the label is the integer level (vmin + k)
    // but the position must be the band CENTER (k + 0.5)/n_disc — otherwise
    // labels sit on segment boundaries instead of in the middle of each
    // colored rectangle (mirrors MATLAB `cb.Ticks = idx + 0.5; cb.TickLabels = idx`).
    //   - Discrete auto   (discrete_levels == 0): one tick per integer level,
    //     label = vmin + k, position = band center.
    //   - Discrete manual (discrete_levels > 0):  tick at each band center,
    //     label coincides with position (vmin + (k+0.5)*bandsize).
    //   - Continuous: ~5 nice_step values across [vmin, vmax].
    struct ColorbarTick { double label; float frac; };
    auto compute_ticks = [&]() {
        std::vector<ColorbarTick> out;
        double range = (double)vmax - (double)vmin;
        if (n_disc > 0) {
            if (discrete_levels == 0) {
                for (int k = 0; k < n_disc; ++k)
                    out.push_back({ (double)vmin + (double)k,
                                    ((float)k + 0.5f) / (float)n_disc });
            } else if (range > 0.0) {
                double bs = range / (double)n_disc;
                for (int k = 0; k < n_disc; ++k)
                    out.push_back({ (double)vmin + ((double)k + 0.5) * bs,
                                    ((float)k + 0.5f) / (float)n_disc });
            }
        } else if (range > 0.0) {
            double step = nice_step(range, 5);
            if (step > 0.0) {
                double start = std::ceil((double)vmin / step) * step;
                for (double v = start; v <= (double)vmax + step * 0.5; v += step)
                    out.push_back({ v, (float)((v - (double)vmin) / range) });
            }
        }
        if (out.empty()) out.push_back({ (double)vmin, 0.5f });
        return out;
    };
    std::vector<ColorbarTick> tick_vals = compute_ticks();
    float max_tick_w = 0.0f;
    for (const auto& t : tick_vals) {
        float w = ImGui::CalcTextSize(fmt_tick(t.label).c_str()).x;
        max_tick_w = std::max(max_tick_w, w);
    }
    const float margin_right = colorbar_w + colorbar_gap + tick_len + tick_text_gap + max_tick_w + 6.0f;

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
    float uv_off_x   = (float)((view_min_x - param_lo_x) / data_rx);
    float uv_scale_x = (float)((view_max_x - view_min_x) / data_rx);
    float uv_off_y   = (float)((view_min_y - param_lo_y) / data_ry);
    float uv_scale_y = (float)((view_max_y - view_min_y) / data_ry);

    // 5. FBO render. (n_disc was resolved up-front in section 3.)
    renderer.begin_frame(plot_w, plot_h, 0.08f, 0.08f, 0.10f, 1.0f);
    renderer.draw_heatmap(data_tex_, vmin, vmax, (int)colormap,
                          uv_off_x, uv_off_y, uv_scale_x, uv_scale_y,
                          n_disc);
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
        ImGui::Checkbox("Discrete colorbar", &discrete);
        if (discrete) {
            ImGui::SetNextItemWidth(80.0f);
            ImGui::InputInt("Levels (0=auto)", &discrete_levels, 0, 0);
            if (discrete_levels < 0) discrete_levels = 0;
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

    ImGuiIO& io = ImGui::GetIO();
    // Хелпер: мировые координаты под текущей позицией курсора (учитывает Y↑).
    auto mouse_world = [&](double& wx, double& wy) {
        double tx = (double)(io.MousePos.x - img_pos.x) / (double)plot_w;
        double ty = 1.0 - (double)(io.MousePos.y - img_pos.y) / (double)plot_h;
        wx = x_axis.view_min + tx * (x_axis.view_max - x_axis.view_min);
        wy = y_axis.view_min + ty * (y_axis.view_max - y_axis.view_min);
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

    // 8a. Wheel zoom (вокруг курсора) — оставляем только в плоте.
    if (plot_hov && io.MouseWheel != 0.0f) {
        float zoom_factor = (io.MouseWheel > 0) ? 1.0f / 1.2f : 1.2f;
        double tx = (double)(io.MousePos.x - img_pos.x) / (double)plot_w;
        double ty = 1.0 - (double)(io.MousePos.y - img_pos.y) / (double)plot_h;
        double world_x = x_axis.view_min + tx * (x_axis.view_max - x_axis.view_min);
        double world_y = y_axis.view_min + ty * (y_axis.view_max - y_axis.view_min);
        double new_rx = (x_axis.view_max - x_axis.view_min) * (double)zoom_factor;
        double new_ry = (y_axis.view_max - y_axis.view_min) * (double)zoom_factor;
        x_axis.view_min = world_x - tx * new_rx;
        x_axis.view_max = x_axis.view_min + new_rx;
        y_axis.view_min = world_y - ty * new_ry;
        y_axis.view_max = y_axis.view_min + new_ry;
    }

    // 8b. Pan ЛКМ — в плоте по обеим осям, в оси — только этой оси.
    if (plot_hov && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
        ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.0f);
        double dwx = -(double)delta.x / (double)plot_w * (x_axis.view_max - x_axis.view_min);
        double dwy =  (double)delta.y / (double)plot_h * (y_axis.view_max - y_axis.view_min);
        x_axis.view_min += dwx;  x_axis.view_max += dwx;
        y_axis.view_min += dwy;  y_axis.view_max += dwy;
        ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
    }
    if (xax_hov && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
        ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.0f);
        double dwx = -(double)delta.x / (double)plot_w * (x_axis.view_max - x_axis.view_min);
        x_axis.view_min += dwx;  x_axis.view_max += dwx;
        ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
    }
    if (yax_hov && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
        ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.0f);
        double dwy =  (double)delta.y / (double)plot_h * (y_axis.view_max - y_axis.view_min);
        y_axis.view_min += dwy;  y_axis.view_max += dwy;
        ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
    }

    // 8c. Rect-zoom ПКМ. Начало drag'а — запоминаем стартовую точку в мире и
    //    режим (плот / X / Y). На release — назначаем новый view по выделенной
    //    зоне. Во время drag'а — рисуем рамку (внизу, см. ниже).
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
                x_axis.view_min = xa; x_axis.view_max = xb;
                y_axis.view_min = ya; y_axis.view_max = yb;
            }
        } else if (rect_zoom_mode_ == 2) {
            double xa = std::min(rect_zoom_x0_, wx), xb = std::max(rect_zoom_x0_, wx);
            if (xb - xa > 1e-12) { x_axis.view_min = xa; x_axis.view_max = xb; }
        } else if (rect_zoom_mode_ == 3) {
            double ya = std::min(rect_zoom_y0_, wy), yb = std::max(rect_zoom_y0_, wy);
            if (yb - ya > 1e-12) { y_axis.view_min = ya; y_axis.view_max = yb; }
        }
        rect_zoom_mode_ = 0;
    }

    // 8d. Double-click — fit. В плоте обе оси, в оси только эта.
    if (plot_dbl)      do_autofit(param_lo_x, param_hi_x, param_lo_y, param_hi_y);
    else if (xax_dbl) { x_axis.view_min = param_lo_x; x_axis.view_max = param_hi_x; }
    else if (yax_dbl) { y_axis.view_min = param_lo_y; y_axis.view_max = param_hi_y; }

    // 8e. После всех изменений view — клампим в пределы данных. Render с
    //    новыми UV произойдёт в следующем кадре; для интерактивности 1-кадра
    //    задержки незаметна, а tick-labels/tooltip ниже уже читают из
    //    x_axis/y_axis напрямую и показывают актуальное состояние.
    clamp_view();

    // 9. Тики осей БЕЗ grid-сетки: короткие штрихи 5px за пределами плота +
    //    числовые подписи. Без линий через весь плот — они отвлекают от
    //    цветового поля.
    ImU32 col_text = IM_COL32(220, 220, 230, 255);
    ImU32 col_axis = IM_COL32(180, 180, 190, 220);
    // Tick'и: формула числа тиков и проверки overshoot/clip — те же, что в
    // plot_axis.cpp (draw_axis_x_grid/y_grid), но без сетки через плот.
    auto draw_x_ticks = [&]() {
        double emin = x_axis.view_min, emax = x_axis.view_max;
        double vrx = emax - emin;
        if (std::abs(vrx) < 1e-30) return;
        double lo = std::min(emin, emax), hi = std::max(emin, emax);
        double sx = nice_step(std::abs(vrx), 8);
        double xstart = std::ceil(lo / sx) * sx;
        int nx_ticks = (int)std::floor((hi - xstart) / sx + 1e-9) + 1;
        if (nx_ticks < 0) nx_ticks = 0;
        for (int i = 0; i < nx_ticks; ++i) {
            double xv = xstart + i * sx;
            if (xv > hi + sx * 1e-6 || xv < lo - sx * 1e-6) continue;
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
        double emin = y_axis.view_min, emax = y_axis.view_max;
        double vry = emax - emin;
        if (std::abs(vry) < 1e-30) return;
        double lo = std::min(emin, emax), hi = std::max(emin, emax);
        double sy = nice_step(std::abs(vry), 6);
        double ystart = std::ceil(lo / sy) * sy;
        int ny_ticks = (int)std::floor((hi - ystart) / sy + 1e-9) + 1;
        if (ny_ticks < 0) ny_ticks = 0;
        for (int i = 0; i < ny_ticks; ++i) {
            double yv = ystart + i * sy;
            if (yv > hi + sy * 1e-6 || yv < lo - sy * 1e-6) continue;
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
                IM_COL32(120, 120, 130, 200), 0.0f, 0, 1.0f);

    // Визуальная рамка rect-zoom во время drag'а ПКМ. Для оси (mode 2/3) —
    // полоса на всю ширину/высоту плота. Координаты в мире → экран.
    if (rect_zoom_mode_ != 0 && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
        double cur_wx, cur_wy; mouse_world(cur_wx, cur_wy);
        auto W2S_x = [&](double w) {
            return img_pos.x + (float)((w - x_axis.view_min) /
                   (x_axis.view_max - x_axis.view_min)) * plot_w;
        };
        auto W2S_y = [&](double w) {
            return img_pos.y + (float)((y_axis.view_max - w) /
                   (y_axis.view_max - y_axis.view_min)) * plot_h;
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

    // 8. Названия осей (как в Plot2DView).
    const char* xl = x_axis.name.empty() ? "x" : x_axis.name.c_str();
    const char* yl = y_axis.name.empty() ? "y" : y_axis.name.c_str();
    float font_h = ImGui::GetFontSize();
    ImVec2 xs = ImGui::CalcTextSize(xl);
    float x_label_y = img_pos.y + plot_h + 2.0f + font_h + 6.0f;
    dl->AddText(ImVec2(img_pos.x + (plot_w - xs.x) * 0.5f, x_label_y), col_text, xl);

    // Y-метка — стопкой букв. Сдвигаем X-позицию столбца ДИНАМИЧЕСКИ за самые
    // широкие Y-тики (как в Plot2DView), иначе подпись наезжает на цифры,
    // когда сетка nice-step выдала длинные числа типа "0.09177".
    size_t yl_len = std::strlen(yl);
    if (yl_len > 0) {
        float max_tick_w = 0.0f;
        double ey0 = y_axis.view_min, ey1 = y_axis.view_max;
        double vry = ey1 - ey0;
        if (std::abs(vry) >= 1e-30) {
            double lo = std::min(ey0, ey1);
            double hi = std::max(ey0, ey1);
            double sy = nice_step(std::abs(vry), 6);
            double ystart = std::ceil(lo / sy) * sy;
            int ny_ticks = (int)std::floor((hi - ystart) / sy + 1) + 1;
            for (int it = 0; it < ny_ticks; ++it) {
                std::string tl = fmt_tick(ystart + it * sy);
                float w = ImGui::CalcTextSize(tl.c_str()).x;
                if (w > max_tick_w) max_tick_w = w;
            }
        }
        float row_h   = font_h * 1.05f;
        float stack_h = row_h * (float)yl_len;
        float y_start = img_pos.y + (plot_h - stack_h) * 0.5f;
        // Колонка букв слева от тиков с зазором 8px.
        float center_x = img_pos.x - max_tick_w - 8.0f - font_h * 0.5f;
        for (size_t i = 0; i < yl_len; ++i) {
            char ch[2] = { yl[i], '\0' };
            ImVec2 cs = ImGui::CalcTextSize(ch);
            dl->AddText(ImVec2(center_x - cs.x * 0.5f, y_start + (float)i * row_h),
                        col_text, ch);
        }
    }

    // 9. Colorbar on the right: horizontal strips via ImDrawList::AddRectFilled.
    //    Continuous mode uses 64 strips (visually indistinguishable from LUT).
    //    Discrete mode uses one strip per band, sampled at the band center —
    //    matches the on-screen heatmap quantization exactly.
    {
        float cb_x = img_pos.x + plot_w + colorbar_gap;
        float cb_y = img_pos.y;
        float cb_h = (float)plot_h;
        int n_strips = (n_disc > 0) ? n_disc : 64;
        for (int i = 0; i < n_strips; ++i) {
            float t0 = (float)i       / (float)n_strips;
            float t1 = (float)(i + 1) / (float)n_strips;
            // Colorbar: top edge = vmax (t=1), bottom = vmin (t=0).
            float y0 = cb_y + cb_h * (1.0f - t1);
            float y1 = cb_y + cb_h * (1.0f - t0);
            // Sample position: in discrete mode mirror the shader (edge-
            // aligned k/(N-1)) so the first/last bands match continuous
            // endpoints exactly. Continuous mode samples the strip center.
            float t_samp;
            if (n_disc > 0) {
                t_samp = (n_disc > 1) ? (float)i / (float)(n_disc - 1) : 0.5f;
            } else {
                t_samp = (t0 + t1) * 0.5f;
            }
            ImU32 col = cmap_sample(t_samp, colormap);
            dl->AddRectFilled(ImVec2(cb_x, y0), ImVec2(cb_x + colorbar_w, y1), col);
        }
        dl->AddRect(ImVec2(cb_x, cb_y),
                    ImVec2(cb_x + colorbar_w, cb_y + cb_h),
                    IM_COL32(120, 120, 130, 200));

        // Tick marks + labels: use the pre-computed tick_vals so the labels
        // line up with the bands they describe and margin_right reserved the
        // correct width for them.
        double range = (double)vmax - (double)vmin;
        for (const auto& t : tick_vals) {
            if (range <= 0.0) {
                // Degenerate vmin == vmax: still draw the single label centered.
                float y = cb_y + cb_h * 0.5f;
                std::string s = fmt_tick(t.label);
                dl->AddText(ImVec2(cb_x + colorbar_w + tick_len + tick_text_gap,
                                   y - font_h * 0.5f),
                            col_text, s.c_str());
                continue;
            }
            float frac = t.frac;
            if (frac < -1e-4f || frac > 1.0f + 1e-4f) continue;
            frac = std::min(std::max(frac, 0.0f), 1.0f);
            float y = cb_y + cb_h * (1.0f - frac);
            dl->AddLine(ImVec2(cb_x + colorbar_w, y),
                        ImVec2(cb_x + colorbar_w + tick_len, y),
                        col_text);
            std::string s = fmt_tick(t.label);
            dl->AddText(ImVec2(cb_x + colorbar_w + tick_len + tick_text_gap,
                               y - font_h * 0.5f),
                        col_text, s.c_str());
        }
    }

    // 10. Hover-tooltip: (p1, p2, λ) по позиции курсора.
    if (plot_hov) {
        ImGuiIO& io = ImGui::GetIO();
        double ex0 = x_axis.view_min, ex1 = x_axis.view_max;
        double ey0 = y_axis.view_min, ey1 = y_axis.view_max;
        double dx = ex0 + (double)(io.MousePos.x - img_pos.x) / (double)plot_w * (ex1 - ex0);
        double dy = ey1 - (double)(io.MousePos.y - img_pos.y) / (double)plot_h * (ey1 - ey0);
        // Индекс ячейки.
        int ix = (int)std::floor((dx - param_lo_x) / (param_hi_x - param_lo_x) * (double)nx);
        int iy = (int)std::floor((dy - param_lo_y) / (param_hi_y - param_lo_y) * (double)ny);
        if (ix >= 0 && ix < nx && iy >= 0 && iy < ny) {
            double v = values[(size_t)iy * (size_t)nx + (size_t)ix];
            const char* xn = x_axis.name.empty() ? "x" : x_axis.name.c_str();
            const char* yn = y_axis.name.empty() ? "y" : y_axis.name.c_str();
            ImGui::BeginTooltip();
            if (!std::isfinite(v) || v == 999.0 || v == -999.0) {
                ImGui::Text("%s = %.6g\n%s = %.6g\nlambda: diverged", xn, dx, yn, dy);
            } else {
                ImGui::Text("%s = %.6g\n%s = %.6g\nlambda = %.6g", xn, dx, yn, dy, v);
            }
            ImGui::EndTooltip();
        }
    }

    // 11. Двойной клик внутри плота — autofit (повторно). Сидит на отдельной
    // ветке кода чтобы не мешать tooltip.
    if (plot_hov && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        do_autofit(param_lo_x, param_hi_x, param_lo_y, param_hi_y);
    }
}
