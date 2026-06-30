#include "plot_view_3d.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

void Plot3DView::do_autofit() {
    float xmn, xmx, ymn, ymx, zmn, zmx;
    if (series_cache_.bbox(xmn, xmx, ymn, ymx, zmn, zmx)) {
        camera.fit_to_bbox(xmn, xmx, ymn, ymx, zmn, zmx);
        view_valid = true;
    }
}

void Plot3DView::rebuild_axis_cache() {
    float xmn, xmx, ymn, ymx, zmn, zmx;
    if (!series_cache_.bbox(xmn, xmx, ymn, ymx, zmn, zmx)) return;

    // ���� bbox �� ������� - �� �������������
    if (axis_cache_.size() == 3 &&
        axis_bbox_[0] == xmn && axis_bbox_[1] == xmx &&
        axis_bbox_[2] == ymn && axis_bbox_[3] == ymx &&
        axis_bbox_[4] == zmn && axis_bbox_[5] == zmx) return;

    axis_cache_.clear();
    // ��� X: �� (xmn, ymn, zmn) �� (xmx, ymn, zmn)
    {
        float pts[6] = { xmn, ymn, zmn, xmx, ymn, zmn };
        axis_cache_.upload(pts, 2);
    }
    // ��� Y: �� (xmn, ymn, zmn) �� (xmn, ymx, zmn)
    {
        float pts[6] = { xmn, ymn, zmn, xmn, ymx, zmn };
        axis_cache_.upload(pts, 2);
    }
    // ��� Z: �� (xmn, ymn, zmn) �� (xmn, ymn, zmx)
    {
        float pts[6] = { xmn, ymn, zmn, xmn, ymn, zmx };
        axis_cache_.upload(pts, 2);
    }
    axis_bbox_[0] = xmn; axis_bbox_[1] = xmx;
    axis_bbox_[2] = ymn; axis_bbox_[3] = ymx;
    axis_bbox_[4] = zmn; axis_bbox_[5] = zmx;
}

void Plot3DView::render(PlotRenderer& renderer,
    ImVec2 block_origin, ImVec2 avail_size,
    int owner_id,
    int data_generation,
    const std::vector<PlotSeriesInput3D>& series_in,
    const std::vector<bool>& init_visible,
    const std::vector<bool>& global_visible,
    bool fit_request)
{
    // 1. ����������� ����
    if (data_generation != series_generation) {
        bool count_changed = ((int)visible.size() != (int)series_in.size());
        series_cache_.clear();
        for (const auto& s : series_in)
            series_cache_.upload(s.points, s.n_points);
        series_generation = data_generation;

        if (count_changed) {
            visible.assign(series_in.size(), true);
            for (size_t k = 0; k < series_in.size() && k < init_visible.size(); ++k)
                visible[k] = init_visible[k];
        }
    }
    if (visible.size() != series_in.size())
        visible.resize(series_in.size(), true);

    auto eff_visible = [&](int k) -> bool {
        bool loc = (k < (int)visible.size()) ? visible[k] : true;
        bool glob = (k < (int)global_visible.size()) ? global_visible[k] : true;
        return loc && glob;
        };

    // 2. �������
    if (!view_valid || fit_request) do_autofit();

    // 3. ������� � ������� (3D �� ����� margin ��� ������� ����, ��� ����� ����� -
    //    ���� ������� ��������� ������� �� �����, ����� ���� ImGui �� ��������� � ����)
    const float margin = 4.0f;
    int plot_w = (std::max)(64, (int)(avail_size.x - margin * 2));
    int plot_h = (std::max)(64, (int)(avail_size.y - margin * 2));

    ImGui::Dummy(avail_size);
    ImVec2 img_pos = ImVec2(block_origin.x + margin, block_origin.y + margin);

    // 4. FBO render � depth
    camera.aspect = (float)plot_w / (float)plot_h;
    {
        float br, bg, bb, ba;
        plot_bg_color(br, bg, bb, ba);
        // 3D-сцена чуть темнее dark / чуть тусклее light, чтобы axes-стрелки
        // оставались читабельными в обоих случаях.
        if (plot_light_theme()) { br = 0.965f; bg = 0.965f; bb = 0.965f; }
        else                    { br = 0.050f; bg = 0.050f; bb = 0.080f; }
        renderer.begin_frame(plot_w, plot_h, br, bg, bb, ba, /*with_depth=*/true);
    }
    float mvp[16];
    camera.build_mvp(mvp);
    for (int k = (int)series_cache_.size() - 1; k >= 0; --k) {
        if (!eff_visible(k)) continue;
        const GpuLineSeries3D& g = series_cache_.get(k);
        if (!g.valid()) continue;
        ImVec4 c = (k < (int)series_in.size()) ? series_in[k].color : ImVec4(1, 1, 1, 1);
        float color[4] = { c.x, c.y, c.z, c.w };
        renderer.draw_line_3d(g.vbo, g.point_count, mvp, color, 1.5f);
    }
    // ������ ��� (X=�������, Y=������, Z=�����)
    if (show_axes) {
        rebuild_axis_cache();
        if (axis_cache_.size() == 3) {
            float col_x[4] = { 1.0f, 0.4f, 0.4f, 1.0f };
            float col_y[4] = { 0.4f, 1.0f, 0.4f, 1.0f };
            float col_z[4] = { 0.4f, 0.6f, 1.0f, 1.0f };
            const GpuLineSeries3D& ax = axis_cache_.get(0);
            const GpuLineSeries3D& ay = axis_cache_.get(1);
            const GpuLineSeries3D& az = axis_cache_.get(2);
            if (ax.valid()) renderer.draw_line_3d(ax.vbo, ax.point_count, mvp, col_x, 2.0f);
            if (ay.valid()) renderer.draw_line_3d(ay.vbo, ay.point_count, mvp, col_y, 2.0f);
            if (az.valid()) renderer.draw_line_3d(az.vbo, az.point_count, mvp, col_z, 2.0f);
        }
    }
    renderer.end_frame();

    // 5. ����� FBO
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddImage((ImTextureID)(intptr_t)renderer.texture_id(),
        img_pos, ImVec2(img_pos.x + plot_w, img_pos.y + plot_h),
        ImVec2(0, 1), ImVec2(1, 0));

    // ������� ���� (����� ����� � ������� �����-����)
    {
        if (show_axes) {
            float xmn, xmx, ymn, ymx, zmn, zmx;
            if (series_cache_.bbox(xmn, xmx, ymn, ymx, zmn, zmx)) {
                ImU32 col_x = IM_COL32(255, 120, 120, 180);
                ImU32 col_y = IM_COL32(120, 255, 120, 180);
                ImU32 col_z = IM_COL32(140, 170, 255, 180);

                // ������ �� ����� ��� (� �������� ������)
                const float text_offset = 6.0f;

                auto draw_axis_label = [&](float ex, float ey, float ez,
                    const char* name, ImU32 color)
                    {
                        float zclip = 0;
                        ImVec2 p = project_to_screen(mvp, ex, ey, ez,
                            img_pos, (float)plot_w, (float)plot_h, &zclip);
                        if (zclip > 1.0f) return; // ����� �� ������� ����������
                        ImVec2 ts = ImGui::CalcTextSize(name);
                        dl->AddText(ImVec2(p.x + text_offset, p.y - ts.y * 0.5f), color, name);
                    };

                draw_axis_label(xmx, ymn, zmn, x_name.c_str(), col_x);
                draw_axis_label(xmn, ymx, zmn, y_name.c_str(), col_y);
                draw_axis_label(xmn, ymn, zmx, z_name.c_str(), col_z);
            }
        }
    }

    // 6. �������
    if (show_legend) {
        std::vector<LegendEntry> entries;
        entries.reserve(series_in.size());
        for (const auto& s : series_in) entries.push_back({ s.label, s.color });
        draw_legend(dl, img_pos, (float)plot_w, entries, visible, global_visible, owner_id);
    }

    // 7. ���� ����������� (���� �������, �� ���� ����)
    ImGui::SetCursorScreenPos(img_pos);
    char id_buf[48];
    std::snprintf(id_buf, sizeof(id_buf), "##plot3d_%d", owner_id);
    ImGui::InvisibleButton(id_buf, ImVec2((float)plot_w, (float)plot_h),
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    bool plot_h_ov = ImGui::IsItemHovered();
    bool plot_a = ImGui::IsItemActive();
    bool plot_dbl = plot_h_ov && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);

    // 8. �����
    dl->AddRect(img_pos, ImVec2(img_pos.x + plot_w, img_pos.y + plot_h),
        plot_col_border(), 0.0f, 0, 1.0f);

    // 9. ������� ���� - �������
    if (plot_dbl) view_valid = false;

    // 10. ����������: orbit ������, pan �����, zoom �������, popup-���� �� �������� ������ ����
    static bool rzoom_pending[64] = { false }; // ���������� �����-�����, ����� �� ������� ����.
    // (��� production ����� ������� pending � ���� ������, ��� � 2D)
    // ���������� ���� ����� ��������� static � �������� owner_id - ����� ������ ��� ������ ����������.
    if (owner_id < 0 || owner_id >= 64) owner_id = 0;

    static float rzoom_start_x[64] = { 0 };
    static float rzoom_start_y[64] = { 0 };

    ImGuiIO& io = ImGui::GetIO();
    const float drag_threshold = 5.0f;

    if (plot_h_ov || plot_a) {
        // Orbit ������
        if (plot_a && ImGui::IsMouseDragging(ImGuiMouseButton_Right, 0.0f)) {
            camera.orbit(io.MouseDelta.x, io.MouseDelta.y);
        }
        // Pan �����
        if (plot_a && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
            camera.pan(io.MouseDelta.x, io.MouseDelta.y, plot_w, plot_h);
        }
        // Zoom �������
        if (plot_h_ov && io.MouseWheel != 0.0f) {
            camera.zoom(std::pow(0.85f, io.MouseWheel));
        }
    }

    // �������� ������ ���� (��� drag) - popup-����
    if (!rzoom_pending[owner_id] && plot_h_ov && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        rzoom_pending[owner_id] = true;
        rzoom_start_x[owner_id] = io.MousePos.x;
        rzoom_start_y[owner_id] = io.MousePos.y;
    }
    if (rzoom_pending[owner_id] && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        float dx = io.MousePos.x - rzoom_start_x[owner_id];
        float dy = io.MousePos.y - rzoom_start_y[owner_id];
        bool was_drag = std::sqrt(dx * dx + dy * dy) > drag_threshold;
        rzoom_pending[owner_id] = false;
        if (!was_drag) {
            char pop_id[48];
            std::snprintf(pop_id, sizeof(pop_id), "##plot3d_menu_%d", owner_id);
            ImGui::OpenPopup(pop_id);
        }
    }

    // 11. ����������� ����
    char pop_id[48];
    std::snprintf(pop_id, sizeof(pop_id), "##plot3d_menu_%d", owner_id);
    if (ImGui::BeginPopup(pop_id)) {
        if (ImGui::MenuItem("Auto fit")) view_valid = false;
        ImGui::Separator();
        ImGui::MenuItem("Show legend", nullptr, &show_legend);
        ImGui::MenuItem("Show axes", nullptr, &show_axes);
        ImGui::EndPopup();
    }
}

