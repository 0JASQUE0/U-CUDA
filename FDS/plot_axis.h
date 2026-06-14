#pragma once
#include "imgui.h"
#include <string>

// AxisInfo - состояние одной оси.
// Хранится внутри Plot2DView, передаётся в render-функции по ссылке.
struct AxisInfo {
    std::string name;
    double view_min = 0;
    double view_max = 1;
    bool   lock = false;
    bool   invert = false;
};

// Эффективные границы оси (с учётом invert).
inline void axis_effective(const AxisInfo& a, double& emin, double& emax) {
    if (a.invert) { emin = a.view_max; emax = a.view_min; }
    else { emin = a.view_min; emax = a.view_max; }
}

// "Приятный" шаг сетки (1, 2 или 5 * 10^n) для ~target_count меток.
double nice_step(double range, int target_count);

// Форматирование числа метки (с эпсилон-обнулением).
std::string fmt_tick(double v);

// MVP-матрица 2D-ортографии (column-major, GL).
void make_ortho_mvp(double xmin, double xmax, double ymin, double ymax, float out[16]);

// Отрисовка сетки и подписей по X (вертикальные линии + цифры снизу).
void draw_axis_x_grid(ImDrawList* dl, const AxisInfo& x,
    ImVec2 plot_pos, float plot_w, float plot_h,
    ImU32 col_grid, ImU32 col_text);

// Отрисовка сетки и подписей по Y (горизонтальные линии + цифры слева).
void draw_axis_y_grid(ImDrawList* dl, const AxisInfo& y,
    ImVec2 plot_pos, float plot_w, float plot_h,
    ImU32 col_grid, ImU32 col_text);