#pragma once
#include "plot_axis.h"
#include "plot_renderer.h"
#include "imgui.h"
#include <glad/glad.h>
#include <string>
#include <vector>
#include <functional>
#include <limits>

// HeatmapView — рисует 2D-скалярное поле n×m в виде хитмапы с colormap'ом. Зеркало Plot2DView по
// структуре (свой AxisInfo, autofit, pan/zoom через ось), но без серий/легенды — данные одни (одна
// R32F-текстура).
// Использование: каждый кадр GUI отдаёт текущий снапшот данных в render():
//   - data_generation — если изменилось, текстура заливается заново;
//   - values — n*m doubles row-major (idx = iy*n + ix);
//   - vmin/vmax — диапазон цветовой шкалы (engine считает по валидным точкам).
// Спец-значения (kernel выдаёт diverged как 999/-999, либо NaN/inf) заменяются на FLT_MAX перед
// загрузкой, шейдер отрисует их тёмно-серым.

// HeatmapColormap определён в plot_renderer.h (используется ещё и для
// colored trajectory). plot_renderer.h уже included через #include выше.

// Общий вертикальный colorbar. Потребители: HeatmapView (section 9 в render) и FastSync mode-0
// (colored trajectory — хитмапы там нет, а цветовая шкала нужна). Раньше FastSync рисовал свою
// копию с марджинами плота, скопированными из Plot2DView числами (78/20/46): любая правка лэйаута
// молча разъезжала шкалу, и тики у него были 5 равноотстоящих вместо «красивых» nice_step.

// Геометрия блока — одинакова у всех потребителей.
constexpr float kColorbarWidth   = 18.0f;
constexpr float kColorbarGap     = 12.0f;
constexpr float kColorbarTickLen = 4.0f;
constexpr float kColorbarTextGap = 2.0f;

// `label` — что печатаем, `frac` — позиция 0..1 вдоль шкалы (0 = vmin/низ, 1 = vmax/верх). Они
// разделены, потому что в discrete-режиме подпись — это целый уровень (vmin + k), а позиция обязана
// быть ЦЕНТРОМ полосы (k + 0.5)/n: иначе подписи стоят на границах, а не в середине цветного
// прямоугольника (зеркалит MATLAB `cb.Ticks = idx + 0.5`).
struct ColorbarTick { double label; float frac; };

// Тики шкалы. n_discrete > 0 → по одной подписи на полосу; при целочисленном
// диапазоне подписи — сами уровни. Иначе ~5 значений через nice_step.
std::vector<ColorbarTick> colorbar_ticks(float vmin, float vmax, int n_discrete);

// Полная ширина блока (шкала + зазор + штрих + подписи) — чтобы вызывающий
// зарезервировал место справа от плота.
float colorbar_total_width(const std::vector<ColorbarTick>& ticks);

// Градиент + рамка + штрихи с подписями. top_left — верхний-левый угол самой
// цветной полосы, height — её высота (обычно = высоте плота).
void draw_colorbar(ImDrawList* dl, ImVec2 top_left, float height,
                   float vmin, float vmax, HeatmapColormap cmap,
                   bool reverse, int n_discrete,
                   const std::vector<ColorbarTick>& ticks);

class HeatmapView {
public:
    AxisInfo x_axis;
    AxisInfo y_axis;
    bool view_valid = false;
    int  data_gen_cached = -1;

    // Видимый диапазон цвета. autoscale → vmin/vmax из render() пересчитываются
    // каждый кадр из values; иначе используются ручные значения.
    HeatmapColormap colormap = kDefaultColormap;
    bool   autoscale = true;
    float  manual_vmin = 0.0f;
    float  manual_vmax = 1.0f;
    // Текстовое представление manual_vmin/vmax для UI — тот же InputNumStr (gui.cpp), что у Initial
    // conditions/Parameters/Fast Synchro vmin/vmax, с шаговым вводом через ↑/↓
    // (digit_step_callback), а не InputFloat со сломанными при step=0 кнопками. Caller парсит их в
    // manual_vmin/vmax сам каждый кадр — HeatmapView не тянет зависимость на gui.cpp.
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
    // Буфер ввода для поля "Levels". Значение остаётся в discrete_levels — текст нужен только чтобы
    // поле было обычным InputText с общим digit_step_input_callback: тогда ↑/↓ шагают разряд под
    // курсором, как во всех остальных числовых полях. Живёт в состоянии вью, а не в локальной
    // переменной кадра: ImGui immediate-mode, локальный буфер терял бы ввод.
    std::string discrete_levels_text = "0";

    // Начальное значение `discrete`, применяемое на первом кадре с реальными данными
    // (data_generation != data_gen_cached). Нужно, чтобы caller мог задать «дефолт checked» для
    // basins-хитмапы до того, как пользователь начал крутить toggle. После первого apply флаг
    // discrete_default_applied_ взводится и render() больше в discrete не пишет.
    bool   discrete_default = false;

    // Swap-axes toggle: транспонирует картинку и меняет местами визуальный X<->Y (диапазоны, тики,
    // подписи и tooltip). Действует только на отрисовку — исходные `values`, `param_lo/hi_*` и
    // AxisInfo.name остаются нетронутыми. Переключается кнопкой "Swap axes" в toolbar'е каждого
    // heatmap-плота; повторное переключение возвращает исходную ориентацию.
    bool   swap_axes = false;

    // Reverse colormap: t := 1-t перед сэмплированием (и в GPU-шейдере, и в colorbar'е). Не
    // персистится (сессионный toggle, как swap_axes) — каждый HeatmapView создаётся отдельно на
    // диаграмму/config, так что флаг автоматически независим между ними. Переключается чекбоксом в
    // том же right-click меню, что и Discrete colorbar.
    bool   reverse_colormap = false;

    // Цвет ячеек со спец-значением (diverged/нет данных: ±999, NaN, Inf). По умолчанию тот же
    // тёмно-серый, что был зашит в шейдере. 1D DFT ставит сюда чёрный: там спец-значением помечены
    // неколебательные режимы, и они не должны читаться как «очень слабый сигнал» на шкале в дБ.
    float  nodata_color[3] = { 0.12f, 0.12f, 0.14f };

    // Подмена палитры на время отрисовки: 0 — рисовать выбранной `colormap`,
    // иначе этим id (картинка и colorbar). Выбор пользователя в `colormap` при
    // этом не трогается и возвращается, как только caller снимет подмену.
    // Нужна карте устойчивости в Order: вид «как в статье» — это конкретная
    // серая шкала, а не пользовательская палитра.
    int    colormap_override = 0;
    // Значения вне [vmin, vmax]: false — прижим к краю палитры (как было), true —
    // свои цвета oor_below / oor_above. Вне диапазона — это вне ТЕКУЩЕЙ шкалы,
    // то есть с учётом ручных min/max из toolbar'а.
    bool   oor_custom = false;
    float  oor_below[3] = { 1.0f, 1.0f, 1.0f };
    float  oor_above[3] = { 1.0f, 1.0f, 1.0f };

    // Индекс отображаемой экспоненты (λ1/λ2/...) — актуально только для LS2D: draw_ls_plot держит
    // свою копию здесь, а не в общем LSCurveConfig::display_exponent_idx, чтобы два окна с одной
    // кривой не дёргали одну переменную. Остальные потребители HeatmapView
    // (Bifurcation2D/LLE2D/Basins) поле не используют.
    int    display_exponent_idx = 0;

    // Optional callback for extra items in the right-click popup menu (after the standard
    // Discrete-colorbar toggle). Caller assigns a lambda before each render(); the view invokes it
    // inside its existing BeginPopup / EndPopup block. Used by gui.cpp to inject "Export data...".
    // Mirrors Plot2DView::popup_extras (plot_view_2d.h).
    std::function<void()> popup_extras;

    // Optional left-click callback: fires on mouse release inside the plot (not on double-click).
    // Arguments: pixel indices (nx_idx, ny_idx) and the snapped world coordinates of that pixel's
    // node centre (same math the hover tooltip uses).
    // ВСЕГДА в координатах ДАННЫХ — в том же порядке осей, в котором caller передал values /
    // param_lo/hi_* в render(), независимо от swap_axes. Внутри вью после swap всё живёт в
    // визуальных координатах, и перед вызовом коллбэка пары переставляются обратно: caller про swap
    // не знает и кладёт первый аргумент в fix_x.
    // Used by the Custom tab for drill-down: release LMB after a click or drag → enqueue a Phase run
    // at (snap_x, snap_y). With `on_left_drag` also set this fires on release regardless of whether
    // the gesture was a drag; with only on_left_click set, it fires only on release-without-drag.
    std::function<void(int nx_idx, int ny_idx, double snap_x, double snap_y)> on_left_click;

    // Optional live-drag callback for LMB. When set, holding LMB inside the plot no longer pans —
    // every frame while the button is down the callback is invoked with the current cursor's pixel
    // indices and snapped world coordinates. Used by the Custom tab so the fix_x/fix_y crosshair
    // follows the cursor during a drag while the heavy recompute is deferred to on_left_click
    // (release). Leave unset in Parametric to keep the classic LMB-pan behaviour.
    std::function<void(int nx_idx, int ny_idx, double snap_x, double snap_y)> on_left_drag;

    // Crosshair overlay — vertical and horizontal lines drawn on top of the heatmap at the given
    // world coordinates. NaN disables the corresponding axis (both NaN by default → nothing
    // rendered, zero cost). Used by the Custom tab to visualise fix_x/fix_y slider positions across
    // all three 2D heatmaps. Как и у on_left_click, значения — в координатах ДАННЫХ: при swap_axes
    // crosshair_x рисуется горизонтальной линией, а не вертикальной, и цвета едут вместе со
    // значениями (они кодируют ось СВИПА, см. ниже).
    double crosshair_x = std::numeric_limits<double>::quiet_NaN();
    double crosshair_y = std::numeric_limits<double>::quiet_NaN();
    // ARGB (0xAA_RR_GG_BB) — matches IM_COL32 default layout. Two colours
    // so the vertical line (X sweep) and horizontal line (Y sweep) read
    // as different axes at a glance, and match the same-axis crosshair
    // on the corresponding 1D slice plot.
    unsigned crosshair_x_color = 0xFF50A0FFu;  // blue-ish (X sweep)
    unsigned crosshair_y_color = 0xFFFF9028u;  // orange   (Y sweep)

    // Слой-маска поверх карты: ячейки с ненулевым значением заливаются
    // overlay_fill_color, а те из них, у кого хоть один из четырёх соседей
    // снаружи, — overlay_edge_color, так что граница области читается даже
    // при бледной заливке. Используется вкладкой Order: область
    // предпочтительности поверх области устойчивости.
    // Маска — nx*ny row-major в координатах ДАННЫХ, как values; swap_axes вью
    // учитывает сама. Указатель ОДНОРАЗОВЫЙ: render() его не хранит и
    // сбрасывает в nullptr, поэтому caller назначает его перед каждым render(),
    // как popup_extras, и буфер обязан жить только до конца вызова.
    // Текстура перезаливается, когда меняется overlay_generation, один из
    // цветов или толщина контура: всё это запечено в текстуру, и caller не обязан вести для него
    // отдельное поколение. Цвета — IM_COL32 (альфа в старшем байте).
    const unsigned char* overlay_mask = nullptr;
    int      overlay_generation = 0;
    unsigned overlay_fill_color = 0x5A40E070u;   // зелёный, альфа ~0.35
    unsigned overlay_edge_color = 0xE660FF90u;   // тот же, светлее и плотнее
    // Толщина контура в ЯЧЕЙКАХ сетки: контуром считаются ячейки области,
    // до которых от ячейки снаружи не больше этого числа шагов по четырём
    // соседям. 1 — только соседи внешних ячеек, 0 — контура нет, одна заливка.
    int      overlay_edge_width = 1;

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

    // Левый отступ под подписи оси Y, измеренный ПРОШЛЫМ кадром. Тики здесь
    // снапаются к узлам параметрической сетки (compute_axis_ticks), а их
    // входные данные готовы уже после того, как отступ нужен для layout'а, —
    // поэтому берём фактическую ширину предыдущего кадра. Значение квантовано
    // до 8 px (plot_left_margin_for_width), так что при зуме оно меняется
    // редко и отставание на кадр не заметно. Старт — прежняя константа.
    float  left_margin_px_ = 78.0f;

    // Слой-маска (см. overlay_mask): RGBA8, GL_NEAREST — ячейки совпадают с
    // ячейками основной текстуры.
    GLuint overlay_tex_ = 0;
    int    overlay_tex_w_ = 0, overlay_tex_h_ = 0;
    int    overlay_gen_cached_ = -1;
    unsigned overlay_fill_cached_ = 0, overlay_edge_cached_ = 0;   // запечённые в текстуру
    int      overlay_edge_width_cached_ = -1;
    std::vector<int> overlay_dist_;   // шагов до ближайшей внешней ячейки (upload_overlay)
    std::vector<unsigned> overlay_buf_;

    void ensure_tex(int w, int h);
    void upload_data(int nx, int ny, const double* values);
    // nx, ny — уже в визуальной (после swap) раскладке.
    void upload_overlay(int nx, int ny, const unsigned char* mask);
    void do_autofit(double lo_x, double hi_x, double lo_y, double hi_y);
};
