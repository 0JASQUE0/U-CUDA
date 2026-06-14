#pragma once
#include "plot_axis.h"
#include "plot_legend.h"
#include "gpu_line_series_3d.h"
#include "plot_renderer.h"
#include "plot_camera_3d.h"
#include <vector>
#include <string>

struct PlotSeriesInput3D {
    const float* points = nullptr;   // n_points * 3 float (x,y,z подряд)
    int          n_points = 0;
    ImVec4       color = ImVec4(1, 1, 1, 1);
    std::string  label;
};

class Plot3DView {
public:
    // Имена осей (для меню/будущих подписей)
    std::string x_name = "x", y_name = "y", z_name = "z";

    // настройки
    bool show_legend = true;
    bool show_axes = true;
    bool view_valid = false;
    int  series_generation = -1;

    // Локальная видимость серий (легенда этого вьюера, независимая).
    std::vector<bool> visible;

    // Камера и её параметры. Доступна снаружи (для меню "reset view" и т.п.).
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
    GpuLineSeriesSet3D axis_cache_; // 3 серии: X, Y, Z (по 2 точки каждая)
    float axis_bbox_[6] = { 0,0,0,0,0,0 }; // последний bbox, для которого построен axis_cache_

    void do_autofit();
    void rebuild_axis_cache(); // строит/перестраивает 3 серии-оси из текущего bbox
};