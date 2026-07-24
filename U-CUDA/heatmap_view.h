#pragma once
#include "plot_axis.h"
#include "plot_renderer.h"
#include "imgui.h"
#include <glad/glad.h>
#include <string>
#include <vector>
#include <functional>
#include <limits>

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

// HeatmapColormap определён в plot_renderer.h (используется ещё и для
// colored trajectory). plot_renderer.h уже included через #include выше.

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
    // Текстовое представление manual_vmin/vmax для UI — тот же InputNumStr
    // (gui.cpp), что у Initial conditions/Parameters/Fast Synchro vmin/vmax,
    // с шаговым вводом через ↑/↓ (digit_step_callback), а не InputFloat со
    // сломанными при step=0 кнопками. Caller парсит их в manual_vmin/vmax
    // сам каждый кадр (см. draw_bifurcation_plot/draw_lle_plot/draw_ls_plot,
    // по образцу FastSync) — HeatmapView не тянет зависимость на gui.cpp.
    std::string manual_vmin_text = "0";
    std::string manual_vmax_text = "1";
    // Авто-рассчитанные на последнем render (для UI-отображения).
    float  shown_vmin = 0.0f;
    float  shown_vmax = 1.0f;

    // Discrete colormap mode: quantize values into N bands. Used for integer-
    // valued heatmaps (basin cluster IDs, BD-2D periods). When discrete_levels
    // is 0, N is auto-derived from the data range (round(vmax-vmin+1)).
    // Right-click context menu on the plot toggles `discrete`.
    bool   discrete = false;
    int    discrete_levels = 0;

    // Начальное значение `discrete`, применяемое на первом кадре с реальными
    // данными (data_generation != data_gen_cached). Нужно чтобы caller мог
    // задать «дефолт checked» для basins-heatmap'а до того, как пользователь
    // начал крутить toggle. После первого apply флаг discrete_default_applied_
    // взводится и render() больше в discrete не пишет — пользовательский
    // toggle через popup работает нормально.
    bool   discrete_default = false;

    // Swap-axes toggle: транспонирует картинку и меняет местами визуальный X<->Y
    // (диапазоны, тики, подписи и tooltip). Действует только на отрисовку —
    // исходные значения `values`, `param_lo/hi_*` и AxisInfo.name остаются
    // нетронутыми. Переключается кнопкой "Swap axes" в toolbar'е каждого
    // heatmap-плота; повторное переключение возвращает исходную ориентацию.
    bool   swap_axes = false;

    // Reverse colormap: t := 1-t перед сэмплированием (и в GPU-шейдере, и в
    // colorbar'е). Не персистится (сессионный toggle, как swap_axes) — каждый
    // HeatmapView уже создаётся отдельно на диаграмму/config, так что флаг
    // автоматически независим между ними. Переключается чекбоксом в том же
    // right-click меню, что и Discrete colorbar (см. render()).
    bool   reverse_colormap = false;

    // Индекс отображаемой экспоненты (λ1/λ2/...) — актуально только для LS2D
    // (draw_ls_plot в gui.cpp держит свою копию здесь, а не в общем
    // LSCurveConfig::display_exponent_idx, чтобы два окна с одной и той же
    // кривой не дёргали одну и ту же переменную). Остальные потребители
    // HeatmapView (Bifurcation2D/LLE2D/Basins) поле не используют.
    int    display_exponent_idx = 0;

    // Optional callback for extra items in the right-click popup menu (after
    // the standard Discrete-colorbar toggle). Caller assigns a lambda before
    // each render(); the view invokes it inside its existing BeginPopup /
    // EndPopup block. Used by gui.cpp to inject the "Export data..." action.
    // Mirrors Plot2DView::popup_extras (see plot_view_2d.h).
    std::function<void()> popup_extras;

    // Optional left-click callback: fires on mouse release inside the plot
    // when the gesture was NOT a pan-drag (release-without-drag) and NOT a
    // double-click. Arguments: pixel indices (nx_idx, ny_idx) and the
    // snapped world coordinates of that pixel's node centre (same math the
    // hover tooltip uses). Used by the Custom-tab for drill-down: click a
    // Bif2D pixel → set the Phase param_values to (snap_x, snap_y).
    std::function<void(int nx_idx, int ny_idx, double snap_x, double snap_y)> on_left_click;

    // Crosshair overlay — vertical and horizontal lines drawn on top of the
    // heatmap at the given world coordinates. NaN disables the corresponding
    // axis (both NaN by default → nothing rendered, zero cost). Used by the
    // Custom-tab to visualise fix_x/fix_y slider positions across all three
    // 2D heatmaps.
    double crosshair_x = std::numeric_limits<double>::quiet_NaN();
    double crosshair_y = std::numeric_limits<double>::quiet_NaN();

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

    // Для детекции изменения swap_axes: если флаг отличается от прошлого
    // кадра — форсируем re-upload текстуры (с транспонированной раскладкой)
    // и autofit (т.к. новые ranges).
    bool   swap_axes_cached_ = false;

    // Однократный apply discrete_default в render(). Сбрасывается только
    // при пересоздании HeatmapView (fresh app start), не при смене данных.
    bool   discrete_default_applied_ = false;

    void ensure_tex(int w, int h);
    void upload_data(int nx, int ny, const double* values);
    void do_autofit(double lo_x, double hi_x, double lo_y, double hi_y);
};
