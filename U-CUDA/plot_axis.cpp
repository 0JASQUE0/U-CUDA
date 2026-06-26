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

// Глобальное значение для fmt_tick. Менеджится set_tick_precision() (зовётся
// из app_main при загрузке config и из Settings UI при изменении слайдера).
static int g_tick_precision = 4;

void set_tick_precision(int n) {
    g_tick_precision = std::clamp(n, 2, 10);
}

std::string fmt_tick(double v) {
    char buf[32];
    double a = std::abs(v);
    if (a < 1e-12) return "0";
    // %.Ne даёт N+1 значащих цифр (1 до точки + N после); чтобы согласовать
    // с %.Ng (N значащих), для научной нотации передаём prec-1.
    if (a != 0 && (a < 1e-3 || a >= 1e5))
        std::snprintf(buf, sizeof(buf), "%.*e", std::max(1, g_tick_precision - 1), v);
    else
        std::snprintf(buf, sizeof(buf), "%.*g", g_tick_precision, v);
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
    // hi-xstart нормируется на sx → floor(...) + 1 даёт ровно столько тиков,
    // сколько помещается в [xstart, hi]. Эпсилон ловит floating-point случаи
    // когда xstart + k*sx должно совпадать с hi, но из-за accumulation lo чуть
    // меньше. Дополнительно: tick рисуем только если он в пределах view (с
    // запасом в полстепа в обе стороны) — это исключает overshoot на правом
    // краю при zoom, когда последний tick визуально выпадает за границу плота.
    int nx = (int)std::floor((hi - xstart) / sx + 1e-9) + 1;
    if (nx < 0) nx = 0;

    for (int ix = 0; ix < nx; ++ix) {
        double xv = xstart + ix * sx;
        if (xv > hi + sx * 1e-6 || xv < lo - sx * 1e-6) continue;
        float px = pos.x + (float)((xv - emin) / vrx) * plot_w;
        // Подпись центрирована по px — её половина уезжает в margin_left/right
        // (они для этого и оставлены в layout'е плота). Не клампим текст в
        // ширину плота, иначе крайние tick'и без подписей.
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
    // См. комментарий в draw_axis_x_grid про формулу и +1e-9 эпсилон.
    int ny = (int)std::floor((hi - ystart) / sy + 1e-9) + 1;
    if (ny < 0) ny = 0;

    for (int iy = 0; iy < ny; ++iy) {
        double yv = ystart + iy * sy;
        if (yv > hi + sy * 1e-6 || yv < lo - sy * 1e-6) continue;
        float py = pos.y + (float)((emax - yv) / vry) * plot_h;
        // Подпись центрирована по py — её половина уезжает в margin_top/bottom
        // (они для этого и оставлены в layout'е плота). Не клампим текст по
        // высоте плота, иначе крайние tick'и (на самой границе view) без
        // подписей.
        dl->AddLine(ImVec2(pos.x, py), ImVec2(pos.x + plot_w, py), col_grid, 1.0f);
        std::string lbl = fmt_tick(yv);
        ImVec2 ts = ImGui::CalcTextSize(lbl.c_str());
        dl->AddText(ImVec2(pos.x - ts.x - 4, py - ts.y * 0.5f), col_text, lbl.c_str());
    }
}