#include "plot_axis.h"
#include <cmath>
#include <cstdio>
#include <algorithm>

double nice_step(double range, int target_count) {
    if (range <= 0) return 1.0;
    double raw = range / std::max(1, target_count);
    double mag = std::pow(10.0, std::floor(std::log10(raw)));
    double norm = raw / mag;
    double step;
    if (norm < 1.5) step = 1.0;
    else if (norm < 3.5) step = 2.0;
    else if (norm < 7.5) step = 5.0;
    else                 step = 10.0;
    return step * mag;
}

std::string fmt_tick(double v) {
    char buf[32];
    double a = std::abs(v);
    if (a < 1e-12) return "0";
    if (a != 0 && (a < 1e-3 || a >= 1e5))
        std::snprintf(buf, sizeof(buf), "%.2e", v);
    else
        std::snprintf(buf, sizeof(buf), "%.4g", v);
    return buf;
}

void make_ortho_mvp(double xmin, double xmax, double ymin, double ymax, float out[16]) {
    double dx = xmax - xmin; if (std::abs(dx) < 1e-30) dx = 1.0;
    double dy = ymax - ymin; if (std::abs(dy) < 1e-30) dy = 1.0;
    float sx = (float)(2.0 / dx);
    float sy = (float)(2.0 / dy);
    float tx = (float)(-(xmax + xmin) / dx);
    float ty = (float)(-(ymax + ymin) / dy);
    out[0] = sx; out[1] = 0;  out[2] = 0; out[3] = 0;
    out[4] = 0;  out[5] = sy; out[6] = 0; out[7] = 0;
    out[8] = 0;  out[9] = 0;  out[10] = 1; out[11] = 0;
    out[12] = tx; out[13] = ty; out[14] = 0; out[15] = 1;
}

void draw_axis_x_grid(ImDrawList* dl, const AxisInfo& x,
    ImVec2 pos, float plot_w, float plot_h,
    ImU32 col_grid, ImU32 col_text)
{
    double emin, emax;
    axis_effective(x, emin, emax);
    double vrx = emax - emin;
    if (std::abs(vrx) < 1e-30) return;

    double lo = std::min(emin, emax);
    double hi = std::max(emin, emax);
    double sx = nice_step(std::abs(vrx), 8);
    double xstart = std::ceil(lo / sx) * sx;
    int nx = (int)std::floor((hi - xstart) / sx + 1) + 1;

    for (int ix = 0; ix < nx; ++ix) {
        double xv = xstart + ix * sx;
        float px = pos.x + (float)((xv - emin) / vrx) * plot_w;
        dl->AddLine(ImVec2(px, pos.y), ImVec2(px, pos.y + plot_h), col_grid, 1.0f);
        std::string lbl = fmt_tick(xv);
        ImVec2 ts = ImGui::CalcTextSize(lbl.c_str());
        dl->AddText(ImVec2(px - ts.x * 0.5f, pos.y + plot_h + 2), col_text, lbl.c_str());
    }
}

void draw_axis_y_grid(ImDrawList* dl, const AxisInfo& y,
    ImVec2 pos, float plot_w, float plot_h,
    ImU32 col_grid, ImU32 col_text)
{
    double emin, emax;
    axis_effective(y, emin, emax);
    double vry = emax - emin;
    if (std::abs(vry) < 1e-30) return;

    double lo = std::min(emin, emax);
    double hi = std::max(emin, emax);
    double sy = nice_step(std::abs(vry), 6);
    double ystart = std::ceil(lo / sy) * sy;
    int ny = (int)std::floor((hi - ystart) / sy + 1) + 1;

    for (int iy = 0; iy < ny; ++iy) {
        double yv = ystart + iy * sy;
        float py = pos.y + (float)((emax - yv) / vry) * plot_h;
        dl->AddLine(ImVec2(pos.x, py), ImVec2(pos.x + plot_w, py), col_grid, 1.0f);
        std::string lbl = fmt_tick(yv);
        ImVec2 ts = ImGui::CalcTextSize(lbl.c_str());
        dl->AddText(ImVec2(pos.x - ts.x - 4, py - ts.y * 0.5f), col_text, lbl.c_str());
    }
}