#pragma once
#include "plot_axis.h"
#include "plot_renderer.h"
#include "imgui.h"
#include <glad/glad.h>
#include <string>
#include <vector>

// HeatmapView — рисует 2D-скалярное поле n×m в виде хитмапы с colormap'ом.
// Зеркало Plot2DView по структуре (свой AxisInfo, autofit, pan/zoom через ось),
// но без серий/легенды — данные одни (одна R32F-текстура).
//
// Использование: каждый кадр GUI отдаёт текущий снапшот данных в render():
//   - data_generation — если изменилось, текстура заливается заново;
//   - values — n*m doubles row-major (idx = iy*n + ix);
//   - vmin/vmax — диапазон цветовой шкалы (engine считает по валидным точкам).
//
// Спец-значения (kernel выдаёт diverged как 999/-999, либо NaN/inf):
// HeatmapView заменит их на FLT_MAX перед загрузкой, шейдер отрисует
// тёмно-серым.

enum class HeatmapColormap : int {
    Viridis = 0,
    Inferno = 1,
    Turbo   = 2,
    Gray    = 3,
};

class HeatmapView {
public:
    AxisInfo x_axis;
    AxisInfo y_axis;
    bool view_valid = false;
    int  data_gen_cached = -1;

    // Видимый диапазон цвета. autoscale → vmin/vmax из render() пересчитываются
    // каждый кадр из values; иначе используются ручные значения.
    HeatmapColormap colormap = HeatmapColormap::Viridis;
    bool   autoscale = true;
    float  manual_vmin = 0.0f;
    float  manual_vmax = 1.0f;
    // Авто-рассчитанные на последнем render (для UI-отображения).
    float  shown_vmin = 0.0f;
    float  shown_vmax = 1.0f;

    HeatmapView() = default;
    ~HeatmapView();
    HeatmapView(const HeatmapView&) = delete;
    HeatmapView& operator=(const HeatmapView&) = delete;

    // Главный рендер. block_origin/avail_size — место под весь блок (с осями
    // и colorbar'ом справа). owner_id — для уникальных ImGui-ID кнопок-осей.
    // Если data_generation совпал с кэшем — текстура не перезаливается.
    void render(PlotRenderer& renderer,
                ImVec2 block_origin, ImVec2 avail_size,
                int owner_id,
                int data_generation,
                int nx, int ny,
                const double* values,        // nx*ny, row-major
                double param_lo_x, double param_hi_x,
                double param_lo_y, double param_hi_y,
                double engine_vmin, double engine_vmax,
                bool fit_request);

private:
    GLuint data_tex_ = 0;
    int    tex_w_ = 0, tex_h_ = 0;
    std::vector<float> upload_buf_;  // переиспользуется между кадрами

    // Rect-zoom через drag ПКМ. mode: 0 = неактивен, 1 = в плоте (XY),
    // 2 = в X-оси (только X), 3 = в Y-оси (только Y). Координаты в мире
    // (значения параметров, не пиксели) — финальная зона выводится из
    // (x0, y0) до текущей позиции курсора.
    int    rect_zoom_mode_ = 0;
    double rect_zoom_x0_ = 0.0;
    double rect_zoom_y0_ = 0.0;

    void ensure_tex(int w, int h);
    void upload_data(int nx, int ny, const double* values);
    void do_autofit(double lo_x, double hi_x, double lo_y, double hi_y);
};
