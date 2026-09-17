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

// БЛИЖАЙШИЙ узел, а не пиксельная полоса. Для 1D-графиков (Bif/LLE/LS) точки
// рисуются россыпью, а не N цветными полосами, поэтому «узел, в чей пиксель
// попал курсор» смысла не имеет — нужен ближайший посчитанный.
//
// Индекс и значение считаются в ОДНОЙ шкале и с одним знаменателем (n-1):
// SnapCursorToGrid1D ниже делит диапазон на n равных частей для индекса, а
// значение берёт по формуле с (n-1), и у краёв они расходятся на узел. На
// лог-свипе это заметно больше всего: слева узлы густые, и промах на узел —
// это заметная относительная ошибка в подписи.
inline bool NearestNode1D(double cursor_x, double x_min, double x_max, int n,
                          bool log_scale, int& out_idx, double& out_value)
{
    if (n < 1) return false;
    if (cursor_x < x_min || cursor_x > x_max) return false;
    if (n == 1) {
        out_idx = 0;
        out_value = (double)ucuda_node_value(0, 1, (numb)x_min, (numb)x_max);
        return true;
    }
    // log_scale валиден только при положительных границах; живой чекбокс без
    // прогона может долететь сюда с нулём — деградируем на линейную сетку.
    const bool use_log = log_scale && x_min > 0.0 && x_max > 0.0 && cursor_x > 0.0;
    double t;
    if (use_log) {
        const double l0 = std::log10(x_min), l1 = std::log10(x_max);
        if (!(l1 > l0)) return false;
        t = (std::log10(cursor_x) - l0) / (l1 - l0);
    } else {
        if (!(x_max > x_min)) return false;
        t = (cursor_x - x_min) / (x_max - x_min);
    }
    int idx = (int)std::lround(t * (double)(n - 1));
    idx = std::clamp(idx, 0, n - 1);
    out_idx   = idx;
    out_value = use_log ? (double)ucuda_node_value_log(idx, n, (numb)x_min, (numb)x_max)
                        : (double)ucuda_node_value(idx, n, (numb)x_min, (numb)x_max);
    return true;
}

// Половина расстояния до ближайшего соседнего узла — в той шкале, в какой
// сетка реально разложена. Нужна как допуск «эта точка данных принадлежит
// узлу idx»: линейный (hi-lo)/(n-1) на лог-свипе врёт в обе стороны — у
// густого края он в десятки раз больше настоящего зазора (в допуск попадают
// чужие узлы), у редкого меньше (не попадает ни один).
inline double NodeHalfGap(double x_min, double x_max, int n, int idx, bool log_scale) {
    if (n < 2) return 0.0;
    const bool use_log = log_scale && x_min > 0.0 && x_max > 0.0;
    auto val = [&](int k) -> double {
        k = std::clamp(k, 0, n - 1);
        return use_log ? (double)ucuda_node_value_log(k, n, (numb)x_min, (numb)x_max)
                       : (double)ucuda_node_value(k, n, (numb)x_min, (numb)x_max);
    };
    const double v = val(idx);
    double gap = 0.0;
    if (idx > 0)     gap = std::abs(v - val(idx - 1));
    if (idx < n - 1) {
        const double g2 = std::abs(val(idx + 1) - v);
        gap = (gap > 0.0) ? std::min(gap, g2) : g2;
    }
    return gap * 0.5;
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
