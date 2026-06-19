#include "plot_view_2d.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

void Plot2DView::do_autofit() {
    float xmin, xmax, ymin, ymax;
    if (series_cache_.bbox(xmin, xmax, ymin, ymax)) {
        double padx = pad_x ? (xmax - xmin) * 0.05 : 0.0; if (pad_x && padx < 1e-9) padx = 1.0;
        double pady = pad_y ? (ymax - ymin) * 0.05 : 0.0; if (pad_y && pady < 1e-9) pady = 1.0;
        x_axis.view_min = xmin - padx; x_axis.view_max = xmax + padx;
        y_axis.view_min = ymin - pady; y_axis.view_max = ymax + pady;
        view_valid = true;
    }
}

void Plot2DView::fit_x() {
    float xmin, xmax, ymin, ymax;
    if (series_cache_.bbox(xmin, xmax, ymin, ymax)) {
        double padx = pad_x ? (xmax - xmin) * 0.05 : 0.0; if (pad_x && padx < 1e-9) padx = 1.0;
        x_axis.view_min = xmin - padx; x_axis.view_max = xmax + padx;
    }
}

void Plot2DView::fit_y() {
    float xmin, xmax, ymin, ymax;
    if (series_cache_.bbox(xmin, xmax, ymin, ymax)) {
        double pady = pad_y ? (ymax - ymin) * 0.05 : 0.0; if (pad_y && pady < 1e-9) pady = 1.0;
        y_axis.view_min = ymin - pady; y_axis.view_max = ymax + pady;
    }
}

void Plot2DView::render(PlotRenderer& renderer,
    ImVec2 block_origin, ImVec2 avail_size,
    int owner_id,
    int data_generation,
    const std::vector<PlotSeriesInput>& series_in,
    const std::vector<bool>& init_visible,
    const std::vector<bool>& global_visible,
    bool fit_request)
{
    // 1. ����������� ���� ���� ��������� ������ ����������
    if (data_generation != series_generation) {
        bool count_changed = ((int)visible.size() != (int)series_in.size());
        series_cache_.clear();
        for (const auto& s : series_in)
            series_cache_.upload(s.points, s.n_points);
        series_generation = data_generation;
        // �� ���������� view_valid �����: ����� ������� ����� (���/���� ����������
        // ����� show_var) �� ������ ������� ���. ������� � ������ �� fit_request
        // ��� ��� ����� ������ ������ (view_valid �������� false).
        // ���������: ���� ����� ����� ���������� � ���� �� ������ (init),
        // ����� ��������� ������� (������� ��������� ����� �����������).
        if (count_changed) {
            visible.assign(series_in.size(), true);
            for (size_t k = 0; k < series_in.size() && k < init_visible.size(); ++k)
                visible[k] = init_visible[k];
        }
    }
    if (visible.size() != series_in.size()) // ��������� �������
        visible.resize(series_in.size(), true);

    // �������� ��������� �����: ���������� (������� ��) � ��������� (�������)
    auto eff_visible = [&](int k) -> bool {
        bool loc = (k < (int)visible.size()) ? visible[k] : true;
        bool glob = (k < (int)global_visible.size()) ? global_visible[k] : true;
        return loc && glob;
        };

    // 2. ������� ��� ������ ������ ��� �� ������ ������� (fit_request)
    if (!view_valid || fit_request) do_autofit();

    // 3. ������� � �������
    const float margin_left = 60.0f;
    const float margin_right = 20.0f;
    const float margin_top = 20.0f;
    const float margin_bottom = 28.0f;

    int plot_w = std::max(64, (int)(avail_size.x - margin_left - margin_right));
    int plot_h = std::max(64, (int)(avail_size.y - margin_top - margin_bottom));

    ImGui::Dummy(avail_size);
    ImVec2 img_pos = ImVec2(block_origin.x + margin_left, block_origin.y + margin_top);

    double ex0, ex1, ey0, ey1;
    axis_effective(x_axis, ex0, ex1);
    axis_effective(y_axis, ey0, ey1);

    // 4. FBO render
    renderer.begin_frame(plot_w, plot_h, 0.08f, 0.08f, 0.10f, 1.0f);
    float mvp[16];
    make_ortho_mvp(ex0, ex1, ey0, ey1, mvp);
    for (int k = (int)series_cache_.size() - 1; k >= 0; --k) {
        if (!eff_visible(k)) continue;
        const GpuLineSeries& g = series_cache_.get(k);
        if (!g.valid()) continue;
        ImVec4 c = (k < (int)series_in.size()) ? series_in[k].color : ImVec4(1, 1, 1, 1);
        float color[4] = { c.x, c.y, c.z, c.w };
        if (points_mode)
            renderer.draw_points(g.vbo, g.point_count, mvp, color, point_size_px);
        else if (!imdraw_lines)  // линии нарисуем через ImDrawList после осей
            renderer.draw_line(g.vbo, g.point_count, mvp, color, line_thickness_px);
    }
    renderer.end_frame();

    // 5. ����� FBO
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddImage((ImTextureID)(intptr_t)renderer.texture_id(),
        img_pos, ImVec2(img_pos.x + plot_w, img_pos.y + plot_h),
        ImVec2(0, 1), ImVec2(1, 0));

    // 6. ������� (������ ���� visible)
    if (show_legend) {
        std::vector<LegendEntry> legend_entries;
        legend_entries.reserve(series_in.size());
        for (const auto& s : series_in)
            legend_entries.push_back({ s.label, s.color });
        draw_legend(dl, img_pos, (float)plot_w, legend_entries, visible, global_visible, owner_id);
    }

    // 7. ���� �����������
    ImGui::SetCursorScreenPos(ImVec2(img_pos.x, img_pos.y + plot_h));
    char id_buf[48];
    std::snprintf(id_buf, sizeof(id_buf), "##xaxis_%d", owner_id);
    ImGui::InvisibleButton(id_buf, ImVec2((float)plot_w, margin_bottom),
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    bool xax_h = ImGui::IsItemHovered();
    bool xax_a = ImGui::IsItemActive();
    bool xax_dbl = xax_h && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);

    ImGui::SetCursorScreenPos(ImVec2(img_pos.x - margin_left, img_pos.y));
    std::snprintf(id_buf, sizeof(id_buf), "##yaxis_%d", owner_id);
    ImGui::InvisibleButton(id_buf, ImVec2(margin_left, (float)plot_h),
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    bool yax_h = ImGui::IsItemHovered();
    bool yax_a = ImGui::IsItemActive();
    bool yax_dbl = yax_h && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);

    ImGui::SetCursorScreenPos(img_pos);
    std::snprintf(id_buf, sizeof(id_buf), "##plot_%d", owner_id);
    ImGui::InvisibleButton(id_buf, ImVec2((float)plot_w, (float)plot_h),
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    bool plot_h_ov = ImGui::IsItemHovered();
    bool plot_a = ImGui::IsItemActive();
    bool plot_dbl = plot_h_ov && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);

    // 8. �����, ����
    ImU32 col_grid = IM_COL32(80, 80, 90, 120);
    ImU32 col_axis = IM_COL32(200, 200, 210, 200);
    ImU32 col_text = IM_COL32(220, 220, 230, 255);
    draw_axis_x_grid(dl, x_axis, img_pos, (float)plot_w, (float)plot_h, col_grid, col_text);
    draw_axis_y_grid(dl, y_axis, img_pos, (float)plot_w, (float)plot_h, col_grid, col_text);

    double vrx = ex1 - ex0;
    double vry = ey1 - ey0;
    auto X = [&](double x) { return img_pos.x + (float)((x - ex0) / vrx) * plot_w; };
    auto Y = [&](double y) { return img_pos.y + (float)((ey1 - y) / vry) * plot_h; };

    if (show_zero_x && std::min(ex0, ex1) <= 0 && std::max(ex0, ex1) >= 0)
        dl->AddLine(ImVec2(X(0), img_pos.y), ImVec2(X(0), img_pos.y + plot_h), col_axis, 1.5f);
    if (show_zero_y && std::min(ey0, ey1) <= 0 && std::max(ey0, ey1) >= 0)
        dl->AddLine(ImVec2(img_pos.x, Y(0)), ImVec2(img_pos.x + plot_w, Y(0)), col_axis, 1.5f);

    // Линии данных поверх осей через ImDrawList (только если imdraw_lines).
    // Рисуем посегментно (AddLine), а не AddPolyline — последняя на резких
    // скачках LLE/LS визуально «рвётся» из-за miter-joins при острых углах.
    // Толщина >1px надёжна (ImGui триангулирует), порядок: после осей.
    if (imdraw_lines) {
        ImVec2 cmin = img_pos;
        ImVec2 cmax(img_pos.x + plot_w, img_pos.y + plot_h);
        dl->PushClipRect(cmin, cmax, true);
        for (int k = (int)series_in.size() - 1; k >= 0; --k) {
            if (!eff_visible(k)) continue;
            const PlotSeriesInput& s = series_in[k];
            if (!s.points || s.n_points < 2) continue;
            ImU32 col = ImGui::ColorConvertFloat4ToU32(s.color);
            ImVec2 prev(X(s.points[0]), Y(s.points[1]));
            for (int i = 1; i < s.n_points; ++i) {
                ImVec2 cur(X(s.points[2 * i + 0]), Y(s.points[2 * i + 1]));
                dl->AddLine(prev, cur, col, line_thickness_px);
                prev = cur;
            }
        }
        dl->PopClipRect();
    }

    dl->AddRect(img_pos, ImVec2(img_pos.x + plot_w, img_pos.y + plot_h),
        IM_COL32(120, 120, 130, 200), 0.0f, 0, 1.0f);

    {
        const char* xl = x_axis.name.empty() ? "x" : x_axis.name.c_str();
        const char* yl = y_axis.name.empty() ? "y" : y_axis.name.c_str();
        ImVec2 xs_size = ImGui::CalcTextSize(xl);
        dl->AddText(ImVec2(img_pos.x + plot_w - xs_size.x - 6, img_pos.y + plot_h - xs_size.y - 4),
            col_text, xl);
        dl->AddText(ImVec2(img_pos.x + 6, img_pos.y + 4), col_text, yl);
    }

    // 9. ������� �����
    if (xax_dbl)   fit_x();
    if (yax_dbl)   fit_y();
    if (plot_dbl)  view_valid = false;

    // 10. ���/��� ����-�������
    if (plot_h_ov || plot_a) {
        ImGuiIO& io = ImGui::GetIO();
        float mx = (io.MousePos.x - img_pos.x) / (float)plot_w;
        float my = 1.0f - (io.MousePos.y - img_pos.y) / (float)plot_h;
        double cx = ex0 + mx * (ex1 - ex0);
        double cy = ey0 + my * (ey1 - ey0);

        if (plot_h_ov && io.MouseWheel != 0.0f) {
            double scale = std::pow(0.85, io.MouseWheel);
            if (!x_axis.lock) {
                x_axis.view_min = cx + (x_axis.view_min - cx) * scale;
                x_axis.view_max = cx + (x_axis.view_max - cx) * scale;
            }
            if (!y_axis.lock) {
                y_axis.view_min = cy + (y_axis.view_min - cy) * scale;
                y_axis.view_max = cy + (y_axis.view_max - cy) * scale;
            }
        }
        if (plot_a && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
            ImVec2 d = io.MouseDelta;
            if (!x_axis.lock) {
                double dx_data = -(double)d.x / plot_w * (ex1 - ex0);
                x_axis.view_min += dx_data; x_axis.view_max += dx_data;
            }
            if (!y_axis.lock) {
                double dy_data = (double)d.y / plot_h * (ey1 - ey0);
                y_axis.view_min += dy_data; y_axis.view_max += dy_data;
            }
        }
    }

    // 11. ���/��� �� ����� ���
    {
        ImGuiIO& io = ImGui::GetIO();
        if (xax_h && io.MouseWheel != 0.0f && !x_axis.lock) {
            float mx = (io.MousePos.x - img_pos.x) / (float)plot_w;
            double cx = ex0 + mx * (ex1 - ex0);
            double scale = std::pow(0.85, io.MouseWheel);
            x_axis.view_min = cx + (x_axis.view_min - cx) * scale;
            x_axis.view_max = cx + (x_axis.view_max - cx) * scale;
        }
        if (yax_h && io.MouseWheel != 0.0f && !y_axis.lock) {
            float my = 1.0f - (io.MousePos.y - img_pos.y) / (float)plot_h;
            double cy = ey0 + my * (ey1 - ey0);
            double scale = std::pow(0.85, io.MouseWheel);
            y_axis.view_min = cy + (y_axis.view_min - cy) * scale;
            y_axis.view_max = cy + (y_axis.view_max - cy) * scale;
        }
        if (xax_a && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f) && !x_axis.lock) {
            ImVec2 d = io.MouseDelta;
            double dx_data = -(double)d.x / plot_w * (ex1 - ex0);
            x_axis.view_min += dx_data; x_axis.view_max += dx_data;
        }
        if (yax_a && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f) && !y_axis.lock) {
            ImVec2 d = io.MouseDelta;
            double dy_data = (double)d.y / plot_h * (ey1 - ey0);
            y_axis.view_min += dy_data; y_axis.view_max += dy_data;
        }
    }

    // 12. Rect-zoom + popup
    {
        ImGuiIO& io = ImGui::GetIO();
        float mx = (io.MousePos.x - img_pos.x) / (float)plot_w;
        float my = 1.0f - (io.MousePos.y - img_pos.y) / (float)plot_h;
        double cx = ex0 + mx * (ex1 - ex0);
        double cy = ey0 + my * (ey1 - ey0);

        const float drag_threshold = 5.0f;

        if (!rect_zoom_pending_ && !rect_zoom_active_ &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Right))
        {
            if (plot_h_ov || xax_h || yax_h) {
                rect_zoom_pending_ = true;
                rect_zoom_start_x_ = io.MousePos.x;
                rect_zoom_start_y_ = io.MousePos.y;
                if (xax_h) rect_zoom_mode_ = 1;
                else if (yax_h) rect_zoom_mode_ = 2;
                else            rect_zoom_mode_ = 0;
                rect_zoom_x0_ = cx; rect_zoom_y0_ = cy;
            }
        }
        if (rect_zoom_pending_ && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            float dx = io.MousePos.x - rect_zoom_start_x_;
            float dy = io.MousePos.y - rect_zoom_start_y_;
            if (std::sqrt(dx * dx + dy * dy) > drag_threshold) {
                rect_zoom_active_ = true;
                rect_zoom_pending_ = false;
            }
        }
        if (rect_zoom_pending_ && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
            rect_zoom_pending_ = false;
            char pop_id[48];
            if (rect_zoom_mode_ == 1) std::snprintf(pop_id, sizeof(pop_id), "##xaxis_menu_%d", owner_id);
            else if (rect_zoom_mode_ == 2) std::snprintf(pop_id, sizeof(pop_id), "##yaxis_menu_%d", owner_id);
            else                            std::snprintf(pop_id, sizeof(pop_id), "##plot_menu_%d", owner_id);
            ImGui::OpenPopup(pop_id);
        }
        if (rect_zoom_active_) {
            auto d2s = [&](double x, double y) -> ImVec2 {
                float sx = img_pos.x + (float)((x - ex0) / (ex1 - ex0)) * plot_w;
                float sy = img_pos.y + (float)((ey1 - y) / (ey1 - ey0)) * plot_h;
                return ImVec2(sx, sy);
                };
            ImVec2 p0, p1;
            if (rect_zoom_mode_ == 0) {
                p0 = d2s(rect_zoom_x0_, rect_zoom_y0_); p1 = d2s(cx, cy);
            }
            else if (rect_zoom_mode_ == 1) {
                p0 = ImVec2(d2s(rect_zoom_x0_, 0).x, img_pos.y);
                p1 = ImVec2(d2s(cx, 0).x, img_pos.y + plot_h);
            }
            else {
                p0 = ImVec2(img_pos.x, d2s(0, rect_zoom_y0_).y);
                p1 = ImVec2(img_pos.x + plot_w, d2s(0, cy).y);
            }
            ImVec2 rmin = ImVec2(std::min(p0.x, p1.x), std::min(p0.y, p1.y));
            ImVec2 rmax = ImVec2(std::max(p0.x, p1.x), std::max(p0.y, p1.y));
            dl->AddRectFilled(rmin, rmax, IM_COL32(100, 180, 255, 50));
            dl->AddRect(rmin, rmax, IM_COL32(100, 180, 255, 220), 0.0f, 0, 1.5f);

            if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
                double x0 = rect_zoom_x0_, y0 = rect_zoom_y0_;
                double x1 = cx, y1 = cy;
                if (x0 > x1) std::swap(x0, x1);
                if (y0 > y1) std::swap(y0, y1);
                if (rect_zoom_mode_ == 0) {
                    if (!x_axis.lock) { x_axis.view_min = x0; x_axis.view_max = x1; }
                    if (!y_axis.lock) { y_axis.view_min = y0; y_axis.view_max = y1; }
                }
                else if (rect_zoom_mode_ == 1) {
                    if (!x_axis.lock) { x_axis.view_min = x0; x_axis.view_max = x1; }
                }
                else {
                    if (!y_axis.lock) { y_axis.view_min = y0; y_axis.view_max = y1; }
                }
                rect_zoom_active_ = false;
            }
        }
    }

    // 13. ����������� ����
    char pop_id[48];
    std::snprintf(pop_id, sizeof(pop_id), "##plot_menu_%d", owner_id);
    if (ImGui::BeginPopup(pop_id)) {
        if (ImGui::MenuItem("Auto fit (both)")) view_valid = false;
        if (ImGui::MenuItem("Auto fit X"))      fit_x();
        if (ImGui::MenuItem("Auto fit Y"))      fit_y();
        ImGui::Separator();
        ImGui::MenuItem("Show legend", nullptr, &show_legend);
        ImGui::MenuItem("Lock X axis", nullptr, &x_axis.lock);
        ImGui::MenuItem("Lock Y axis", nullptr, &y_axis.lock);
        ImGui::Separator();
        ImGui::MenuItem("Invert X", nullptr, &x_axis.invert);
        ImGui::MenuItem("Invert Y", nullptr, &y_axis.invert);
        ImGui::EndPopup();
    }
    std::snprintf(pop_id, sizeof(pop_id), "##xaxis_menu_%d", owner_id);
    if (ImGui::BeginPopup(pop_id)) {
        if (ImGui::MenuItem("Auto fit X")) fit_x();
        ImGui::Separator();
        char xmin_buf[32], xmax_buf[32];
        std::snprintf(xmin_buf, sizeof(xmin_buf), "%.6g", x_axis.view_min);
        std::snprintf(xmax_buf, sizeof(xmax_buf), "%.6g", x_axis.view_max);
        ImGui::Text("X range:");
        ImGui::SetNextItemWidth(120);
        if (ImGui::InputText("min##x", xmin_buf, sizeof(xmin_buf), ImGuiInputTextFlags_EnterReturnsTrue))
            x_axis.view_min = std::atof(xmin_buf);
        ImGui::SetNextItemWidth(120);
        if (ImGui::InputText("max##x", xmax_buf, sizeof(xmax_buf), ImGuiInputTextFlags_EnterReturnsTrue))
            x_axis.view_max = std::atof(xmax_buf);
        ImGui::Separator();
        ImGui::MenuItem("Lock X axis", nullptr, &x_axis.lock);
        ImGui::MenuItem("Invert X", nullptr, &x_axis.invert);
        ImGui::EndPopup();
    }
    std::snprintf(pop_id, sizeof(pop_id), "##yaxis_menu_%d", owner_id);
    if (ImGui::BeginPopup(pop_id)) {
        if (ImGui::MenuItem("Auto fit Y")) fit_y();
        ImGui::Separator();
        char ymin_buf[32], ymax_buf[32];
        std::snprintf(ymin_buf, sizeof(ymin_buf), "%.6g", y_axis.view_min);
        std::snprintf(ymax_buf, sizeof(ymax_buf), "%.6g", y_axis.view_max);
        ImGui::Text("Y range:");
        ImGui::SetNextItemWidth(120);
        if (ImGui::InputText("min##y", ymin_buf, sizeof(ymin_buf), ImGuiInputTextFlags_EnterReturnsTrue))
            y_axis.view_min = std::atof(ymin_buf);
        ImGui::SetNextItemWidth(120);
        if (ImGui::InputText("max##y", ymax_buf, sizeof(ymax_buf), ImGuiInputTextFlags_EnterReturnsTrue))
            y_axis.view_max = std::atof(ymax_buf);
        ImGui::Separator();
        ImGui::MenuItem("Lock Y axis", nullptr, &y_axis.lock);
        ImGui::MenuItem("Invert Y", nullptr, &y_axis.invert);
        ImGui::EndPopup();
    }
}