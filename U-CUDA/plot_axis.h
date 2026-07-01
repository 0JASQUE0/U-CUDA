#pragma once
#include "imgui.h"
#include <string>

// AxisInfo - ��������� ����� ���.
// �������� ������ Plot2DView, ��������� � render-������� �� ������.
struct AxisInfo {
    std::string name;
    double view_min = 0;
    double view_max = 1;
    bool   lock = false;
    bool   invert = false;
};

// ����������� ������� ��� (� ������ invert).
inline void axis_effective(const AxisInfo& a, double& emin, double& emax) {
    if (a.invert) { emin = a.view_max; emax = a.view_min; }
    else { emin = a.view_min; emax = a.view_max; }
}

// "��������" ��� ����� (1, 2 ��� 5 * 10^n) ��� ~target_count �����.
double nice_step(double range, int target_count);

// �������������� ����� ����� (� �������-����������).
std::string fmt_tick(double v);

// ������ ���. ����� ������ ��� fmt_tick (���������� ����� � `%g`).
// Clamp'�� � [2, 10]. �������� ��� ������ Settings � ��� ��������.
void set_tick_precision(int n);

// MVP-������� 2D-���������� (column-major, GL).
void make_ortho_mvp(double xmin, double xmax, double ymin, double ymax, float out[16]);

// ��������� ����� � �������� �� X (������������ ����� + ����� �����).
// Опциональный snap-to-node: если snap_n > 1 и snap_hi > snap_lo, шаг тиков
// округляется к целому кратному step_node = (snap_hi - snap_lo)/(snap_n - 1),
// стартовая позиция — тоже кратна этому шагу. Значит тики попадают ровно на
// узлы параметрической сетки. Для 1D-графиков Bif/LLE/LS. Дефолты выключают
// snap (старое поведение).
void draw_axis_x_grid(ImDrawList* dl, const AxisInfo& x,
    ImVec2 plot_pos, float plot_w, float plot_h,
    ImU32 col_grid, ImU32 col_text,
    double snap_lo = 0.0, double snap_hi = 0.0, int snap_n = 0);

// ��������� ����� � �������� �� Y (�������������� ����� + ����� �����).
void draw_axis_y_grid(ImDrawList* dl, const AxisInfo& y,
    ImVec2 plot_pos, float plot_w, float plot_h,
    ImU32 col_grid, ImU32 col_text);

// =====================================================================
// Plot color palette — переключается между Dark и Light темами. AppModel
// дёргает set_plot_light_theme(bool) на смену темы в Settings, а каждый
// plot/heatmap/legend читает текущие значения через геттеры. Прежде эти
// цвета были захардкожены под Dark, и на Light подписи/сетка/рамки осей
// сливались с белым фоном плотов.
// =====================================================================
void set_plot_light_theme(bool light);
bool plot_light_theme();

ImU32 plot_col_text();        // подписи осей, тики, ярлыки серий
ImU32 plot_col_axis();        // линии x=0 / y=0 на плоте (акцент)
ImU32 plot_col_grid();        // тики основной сетки
ImU32 plot_col_border();      // рамка плота, тики на colorbar

// Цвет очистки FBO под плот. Передаётся в PlotRenderer::begin_frame.
void  plot_bg_color(float& r, float& g, float& b, float& a);