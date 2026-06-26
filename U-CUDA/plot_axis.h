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
void draw_axis_x_grid(ImDrawList* dl, const AxisInfo& x,
    ImVec2 plot_pos, float plot_w, float plot_h,
    ImU32 col_grid, ImU32 col_text);

// ��������� ����� � �������� �� Y (�������������� ����� + ����� �����).
void draw_axis_y_grid(ImDrawList* dl, const AxisInfo& y,
    ImVec2 plot_pos, float plot_w, float plot_h,
    ImU32 col_grid, ImU32 col_text);