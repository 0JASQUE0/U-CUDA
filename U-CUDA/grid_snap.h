#pragma once
#include <algorithm>
#include <cmath>
#include "configCUDA.h"   // ucuda_node_value* — общая с ядром формула узла

// Квантование координат курсора к пикселям heatmap-текстуры и узлам
// параметрической сетки движка.
//
// Движок считает значения в N узлах, inclusive на обоих концах; САМО значение
// узла считает ucuda_node_value / ucuda_node_value_log (configCUDA.h) — та же
// функция, которую зовёт ядро через getValueByIdx. Здесь она не дублируется:
// раньше формула стояла копией и расходилась с расчётом в последних битах.
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

    if (g.width == 1) { out_idx_x = 0; out_grid_x = (double)ucuda_node_value(0, 1, (numb)g.x_min, (numb)g.x_max); }
    else {
        // Индекс пикселя: floor((cursor - lo) * N / (hi - lo)); clamp — для
        // случая cursor == hi ровно (тогда floor даст N вместо N-1).
        double range_x = g.x_max - g.x_min;
        int ix = (int)std::floor((cursor_x - g.x_min) * (double)g.width / range_x);
        ix = std::clamp(ix, 0, g.width - 1);
        out_idx_x  = ix;
        // Координата показываемого узла (в этом пикселе).
        out_grid_x = (double)ucuda_node_value(ix, g.width, (numb)g.x_min, (numb)g.x_max);
    }
    if (g.height == 1) { out_idx_y = 0; out_grid_y = (double)ucuda_node_value(0, 1, (numb)g.y_min, (numb)g.y_max); }
    else {
        double range_y = g.y_max - g.y_min;
        int iy = (int)std::floor((cursor_y - g.y_min) * (double)g.height / range_y);
        iy = std::clamp(iy, 0, g.height - 1);
        out_idx_y  = iy;
        out_grid_y = (double)ucuda_node_value(iy, g.height, (numb)g.y_min, (numb)g.y_max);
    }
    return true;
}

// 1D-вариант — только X. Y-координата вызывающий трактует как непрерывную.
// Тоже floor по пиксельной разбивке, отображаем позицию узла.
// log_scale: узлы сетки движка распределены по getValueByIdx_log (лог-
// равномерно), а не линейно -- см. cudaLibrary.cu::getValueByIdx_log. Индекс
// узла (ix) считается по той же пиксельной разбивке (диапазон [x_min,x_max]
// делится на `width` равных пикселей), а вот координата узла внутри пикселя
// восстанавливается log-формулой, иначе tooltip показывает линейно
// интерполированное значение, которого движок никогда не считал.
inline bool SnapCursorToGrid1D(double cursor_x,
                               double x_min, double x_max,
                               int width,
                               int& out_idx_x,
                               double& out_grid_x,
                               bool log_scale = false)
{
    if (width < 1) return false;
    if (cursor_x < x_min || cursor_x > x_max) return false;
    if (width == 1) { out_idx_x = 0; out_grid_x = (double)ucuda_node_value(0, 1, (numb)x_min, (numb)x_max); return true; }
    double range = x_max - x_min;
    // Индекс узла обязан считаться в ТОЙ ЖЕ шкале, что и его значение ниже.
    // Раньше он всегда брался линейно, а значение — по логарифму: на диапазоне
    // 0.1..14 при 201 узле курсор на x=1.0 давал индекс 13 вместо ~93, и
    // Shift-клик возвращал 0.1377. Условие то же, что у ветки значения, плюс
    // cursor_x > 0 — иначе log10 даст -inf.
    int ix;
    if (log_scale && x_min > 0.0 && x_max > 0.0 && cursor_x > 0.0) {
        const double l0 = std::log10(x_min), l1 = std::log10(x_max);
        ix = (int)std::floor((std::log10(cursor_x) - l0) * (double)width / (l1 - l0));
    } else {
        ix = (int)std::floor((cursor_x - x_min) * (double)width / range);
    }
    ix = std::clamp(ix, 0, width - 1);
    out_idx_x  = ix;
    // log_scale валиден только для x_min>0 (log_scale-запрос это требует), но
    // сюда может долететь live-чекбокс без соответствующего результата (напр.
    // ещё не запускали Run, или снапшот от прошлого линейного прогона) --
    // тогда x_min может быть 0/дефолтным. log10(0)=-inf -> NaN дальше по
    // формуле. Деградируем на линейную ноду вместо NaN в tooltip'е.
    if (log_scale && x_min > 0.0 && x_max > 0.0) {
        out_grid_x = (double)ucuda_node_value_log(ix, width, (numb)x_min, (numb)x_max);
    } else {
        out_grid_x = (double)ucuda_node_value(ix, width, (numb)x_min, (numb)x_max);
    }
    return true;
}
