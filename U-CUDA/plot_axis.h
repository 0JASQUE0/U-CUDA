#pragma once
#include "imgui.h"
#include <string>
#include <functional>

// AxisInfo - состояние одной оси.
// Живёт внутри Plot2DView, в render-функции передаётся по ссылке.
struct AxisInfo {
    std::string name;
    double view_min = 0;
    double view_max = 1;
    bool   lock = false;
    bool   invert = false;
    // Sweep-сетка по этой оси log-распределена (см. BifurcationDiagramConfig::
    // log_scale и др.).
    //
    // Для оси X Plot2DView делает НАСТОЯЩЕЕ логарифмическое отображение: на
    // экран (VBO, MVP, курсор, крест) уходит log10(x), наружу всё остаётся
    // мировым. Логарифм не аффинен, поэтому в make_ortho_mvp он не помещается
    // — перенесена сама координата, см. XS/XW в Plot2DView::render.
    // Активен только при положительных границах вида: чекбокс можно включить
    // до Run, и log10(<=0) отравил бы NaN'ом весь кадр.
    //
    // draw_axis_*_grid при этом по-прежнему рисует только границы диапазона
    // (lo/hi), без промежуточных отметок — решение по оформлению.
    // У оси Y лог-отображения нет: сетка по Y логарифмической не бывает.
    bool   log_scale = false;
};

// Эффективные границы оси (с учётом invert).
inline void axis_effective(const AxisInfo& a, double& emin, double& emax) {
    if (a.invert) { emin = a.view_max; emax = a.view_min; }
    else { emin = a.view_min; emax = a.view_max; }
}

// «Красивый» шаг тиков (1, 2 или 5 * 10^n) под ~target_count делений.
double nice_step(double range, int target_count);

// Форматирует подпись тика (с текущей точностью).
std::string fmt_tick(double v);

// Точность тиков: число значащих цифр для fmt_tick (форматирует через %g).
// Clamp'ится в [2, 10]. Дёргается при смене настройки в Settings.
void set_tick_precision(int n);

// MVP-матрица 2D-ортопроекции (column-major, GL).
void make_ortho_mvp(double xmin, double xmax, double ymin, double ymax, float out[16]);

// Рисует сетку и подписи по X (вертикальные линии + текст снизу).
// Опциональный snap-to-node: если snap_n > 1 и snap_hi > snap_lo, шаг тиков
// округляется к целому кратному step_node = (snap_hi - snap_lo)/(snap_n - 1),
// стартовая позиция — тоже кратна этому шагу. Значит тики попадают ровно на
// узлы параметрической сетки. Для 1D-графиков Bif/LLE/LS. Дефолты выключают
// snap (старое поведение).
void draw_axis_x_grid(ImDrawList* dl, const AxisInfo& x,
    ImVec2 plot_pos, float plot_w, float plot_h,
    ImU32 col_grid, ImU32 col_text,
    double snap_lo = 0.0, double snap_hi = 0.0, int snap_n = 0);

// Рисует сетку и подписи по Y (горизонтальные линии + текст слева).
void draw_axis_y_grid(ImDrawList* dl, const AxisInfo& y,
    ImVec2 plot_pos, float plot_w, float plot_h,
    ImU32 col_grid, ImU32 col_text);

// Plot color palette — переключается между Dark и Light темами. AppModel
// дёргает set_plot_light_theme(bool) на смену темы в Settings, а каждый
// plot/heatmap/legend читает текущие значения через геттеры. Прежде эти
// цвета были захардкожены под Dark, и на Light подписи/сетка/рамки осей
// сливались с белым фоном плотов.
void set_plot_light_theme(bool light);
bool plot_light_theme();

ImU32 plot_col_text();        // подписи осей, тики, ярлыки серий
ImU32 plot_col_axis();        // линии x=0 / y=0 на плоте (акцент)
ImU32 plot_col_grid();        // тики основной сетки
ImU32 plot_col_border();      // рамка плота, тики на colorbar

// Цвет очистки FBO под плот. Передаётся в PlotRenderer::begin_frame.
void  plot_bg_color(float& r, float& g, float& b, float& a);

// Screenshot-to-clipboard. Право-клик "Copy image to clipboard" на любой
// диаграмме (Heatmap/Plot2D/Plot3D) заводится через request_plot_screenshot()
// — рект в экранных координатах ImGui (весь блок диаграммы: оси/colorbar/
// подписи, block_origin..+avail_size, не только FBO-картинка). app_main.cpp
// один раз подключает sink при старте (тот же принцип "set once, вызывается
// откуда угодно из plot-кода", что и set_plot_light_theme выше) — сам захват
// пикселей происходит НЕ синхронно внутри сюда, а через пару кадров (см.
// AppModel::PendingScreenshot), чтобы в кадр не попал ещё не закрывшийся popup.
void set_screenshot_request_sink(std::function<void(ImVec2 min, ImVec2 max)> sink);
void request_plot_screenshot(ImVec2 min, ImVec2 max);

// Числовой ввод в ImGui-полях: ↑/↓ шагают разряд под курсором
// (DigitInput::ComputeStep), запятая заменяется точкой. Поле обязано быть
// создано с флагами CallbackCharFilter | CallbackHistory.
//
// Живёт здесь, а не статиком в gui.cpp, потому что потребителей стало два:
// поля параметров во вкладках анализа (InputNumStr / InputNumStrCommit) и
// RGB-поля меню цвета серии в plot_view_2d.cpp. Иначе второму пришлось бы
// копировать колбэк целиком.
int digit_step_input_callback(ImGuiInputTextCallbackData* data);