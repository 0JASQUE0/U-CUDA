#pragma once
#include "plot_axis.h"
#include "plot_legend.h"
#include "gpu_line_series_3d.h"
#include "plot_renderer.h"
#include "plot_camera_3d.h"
#include <vector>
#include <string>

struct PlotSeriesInput3D {
    const float* points = nullptr;   // n_points * 3 float (x,y,z ������)
    int          n_points = 0;
    ImVec4       color = ImVec4(1, 1, 1, 1);
    std::string  label;
};

class Plot3DView {
public:
    // ����� ���� (��� ����/������� ��������)
    std::string x_name = "x", y_name = "y", z_name = "z";

    // ���������
    bool show_legend = true;
    bool show_axes = true;
    bool view_valid = false;
    int  series_generation = -1;

    // Толщина линий данных (не осей). Аналогично Plot2DView::line_thickness_px.
    // По умолчанию 1.5f — совпадает со старым хардкодом, регрессии нет.
    // NB: в OpenGL core profile glLineWidth может клампиться драйвером до 1
    // (что и наблюдалось). Реальная толщина обеспечивается путём
    // custom_line_style=true → PlotRenderer::draw_line_3d(..., thick_style=true),
    // который проходит через geometry shader и рисует screen-aligned quads.
    float line_thickness_px = 1.5f;

    // true → серии данных рисуются через thick-path (GS quads + BLEND).
    // При false — старый быстрый путь (GL_LINE_STRIP + glLineWidth). Оси
    // координат ВСЕГДА идут по старому пути, независимо от этого флага.
    bool custom_line_style = false;

    // ��������� ��������� ����� (������� ����� ������, �����������).
    std::vector<bool> visible;

    // ������ � � ���������. �������� ������� (��� ���� "reset view" � �.�.).
    PlotCamera3D camera;

    void render(PlotRenderer& renderer,
        ImVec2 avail_pos, ImVec2 avail_size,
        int owner_id,
        int data_generation,
        const std::vector<PlotSeriesInput3D>& series_in,
        const std::vector<bool>& init_visible,
        const std::vector<bool>& global_visible,
        bool fit_request = false);

private:
    GpuLineSeriesSet3D series_cache_;
    GpuLineSeriesSet3D axis_cache_; // 3 �����: X, Y, Z (�� 2 ����� ������)
    float axis_bbox_[6] = { 0,0,0,0,0,0 }; // ��������� bbox, ��� �������� �������� axis_cache_

    void do_autofit();
    void rebuild_axis_cache(); // ������/������������� 3 �����-��� �� �������� bbox
};