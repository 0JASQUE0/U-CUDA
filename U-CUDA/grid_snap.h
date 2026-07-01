#pragma once
#include <algorithm>
#include <cmath>

// Квантование координат курсора к пикселям heatmap-текстуры и узлам
// параметрической сетки движка.
//
// Движок считает значения в N узлах с шагом step_node = (hi - lo) / (N - 1),
// inclusive на обоих концах (см. parametric_engine.cpp::getValueByIdx_local).
// Renderer заливает эти N значений в R32F-текстуру и рисует её GL_NEAREST
// на области [lo, hi]. То есть визуально это N solid-color пикселей шириной
// pixel_w = (hi - lo) / N каждый; пиксель k занимает [lo + k*pixel_w,
// lo + (k+1)*pixel_w] и показывает значение узла k.
//
// Поэтому для tooltip'а индекс считаем ПО ПИКСЕЛЬНОЙ разбивке (floor),
// чтобы курсор менял выбор ровно на границе цветовой полосы, а НЕ на середине
// между узлами. Координата, которую показываем — позиция УЗЛА (lo + idx *
// step_node), т.е. точка, где реально было посчитано значение.

struct GridInfo {
    double x_min, x_max;
    double y_min, y_max;
    int    width;   // число узлов по X
    int    height;  // число узлов по Y
};

// Квантует (cursor_x, cursor_y) к пикселю heatmap-текстуры (не к
// ближайшему узлу-центру). Возвращает false, если курсор вне [min, max] или
// сетка невалидна. При N==1 → idx=0, snapped=min.
inline bool SnapCursorToGrid(double cursor_x, double cursor_y,
                             const GridInfo& g,
                             int& out_idx_x, int& out_idx_y,
                             double& out_grid_x, double& out_grid_y)
{
    if (g.width < 1 || g.height < 1) return false;
    if (cursor_x < g.x_min || cursor_x > g.x_max) return false;
    if (cursor_y < g.y_min || cursor_y > g.y_max) return false;

    if (g.width == 1) { out_idx_x = 0; out_grid_x = g.x_min; }
    else {
        // Индекс пикселя: floor((cursor - lo) * N / (hi - lo)); clamp — для
        // случая cursor == hi ровно (тогда floor даст N вместо N-1).
        double range_x = g.x_max - g.x_min;
        int ix = (int)std::floor((cursor_x - g.x_min) * (double)g.width / range_x);
        ix = std::clamp(ix, 0, g.width - 1);
        out_idx_x  = ix;
        // Координата показываемого узла (в этом пикселе).
        double step_node = range_x / (double)(g.width - 1);
        out_grid_x = g.x_min + ix * step_node;
    }
    if (g.height == 1) { out_idx_y = 0; out_grid_y = g.y_min; }
    else {
        double range_y = g.y_max - g.y_min;
        int iy = (int)std::floor((cursor_y - g.y_min) * (double)g.height / range_y);
        iy = std::clamp(iy, 0, g.height - 1);
        out_idx_y  = iy;
        double step_node = range_y / (double)(g.height - 1);
        out_grid_y = g.y_min + iy * step_node;
    }
    return true;
}

// 1D-вариант — только X. Y-координата вызывающий трактует как непрерывную.
// Тоже floor по пиксельной разбивке, отображаем позицию узла.
inline bool SnapCursorToGrid1D(double cursor_x,
                               double x_min, double x_max,
                               int width,
                               int& out_idx_x,
                               double& out_grid_x)
{
    if (width < 1) return false;
    if (cursor_x < x_min || cursor_x > x_max) return false;
    if (width == 1) { out_idx_x = 0; out_grid_x = x_min; return true; }
    double range = x_max - x_min;
    int ix = (int)std::floor((cursor_x - x_min) * (double)width / range);
    ix = std::clamp(ix, 0, width - 1);
    out_idx_x  = ix;
    double step_node = range / (double)(width - 1);
    out_grid_x = x_min + ix * step_node;
    return true;
}
