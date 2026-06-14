#pragma once
#include "plot_axis.h"
#include "plot_legend.h"
#include "gpu_line_series.h"
#include "plot_renderer.h"
#include <vector>
#include <string>

struct PlotSeriesInput {
    const float* points = nullptr;
    int          n_points = 0;
    ImVec4       color = ImVec4(1, 1, 1, 1);
    std::string  label;
};

class Plot2DView {
public:
    AxisInfo x_axis;
    AxisInfo y_axis;
    bool show_legend = true;
    bool view_valid = false;
    int  series_generation = -1;

    // Ќастройки поведени€ под тип графика (адаптер выставл€ет):
    //  pad_*       Ч добавл€ть 5% отступ по оси при автофите.
    //  show_zero_* Ч рисовать линию нул€ по оси (x=0 вертикальна€, y=0 горизонтальна€).
    // 2D фазовый: всЄ true. Time domain: по X впритык и без вертикали (pad_x/show_zero_x=false).
    bool pad_x = true;
    bool pad_y = true;
    bool show_zero_x = true;
    bool show_zero_y = true;

    // ¬идимость серий Ч —¬ќя у каждого вьюера (легенды независимы между проекци€ми).
    std::vector<bool> visible;

    // render:
    //  global_visible Ч видимость от галочек Ќ” (главна€, обща€ на все проекции),
    //                   примен€етс€  ј∆ƒџ… кадр, recompute не нужен.
    //  Ћокальна€ видимость (легенда этой проекции) Ч поле visible ниже.
    //  »тог: сери€ видна = global_visible[k] && visible[k].
    //  init_visible Ч начальные значени€ Ћќ јЋ№Ќќ… видимости при смене числа серий.
    void render(PlotRenderer& renderer,
        ImVec2 avail_pos, ImVec2 avail_size,
        int owner_id,
        int data_generation,
        const std::vector<PlotSeriesInput>& series_in,
        const std::vector<bool>& init_visible,
        const std::vector<bool>& global_visible,
        bool fit_request = false);

private:
    GpuLineSeriesSet series_cache_;

    bool   rect_zoom_pending_ = false;
    bool   rect_zoom_active_ = false;
    double rect_zoom_x0_ = 0, rect_zoom_y0_ = 0;
    int    rect_zoom_mode_ = 0;
    float  rect_zoom_start_x_ = 0;
    float  rect_zoom_start_y_ = 0;

    void do_autofit();
    void fit_x();
    void fit_y();
};