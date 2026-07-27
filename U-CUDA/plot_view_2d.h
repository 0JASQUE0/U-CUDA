#pragma once
#include "plot_axis.h"
#include "plot_legend.h"
#include "gpu_line_series.h"
#include "plot_renderer.h"
#include <vector>
#include <string>
#include <functional>
#include <limits>

// Марджины блока Plot2DView (место под тики и подписи осей вокруг самой
// картинки). Вынесены наружу, потому что FastSync mode-0 рисует colorbar
// РЯДОМ с Plot2DView и обязан знать, где кончается плот: раньше он держал
// свою копию этих чисел, и она молча разъезжалась при правке лэйаута.
void plot_2d_margins(float& left, float& top, float& right, float& bottom);

struct PlotSeriesInput {
    const float* points = nullptr;
    int          n_points = 0;
    ImVec4       color = ImVec4(1, 1, 1, 1);
    std::string  label;

    // Per-segment colored mode (опционально). Когда values != nullptr И imdraw_lines
    // включён в Plot2DView, каждый сегмент i рисуется цветом cmap_sample(t, colormap)
    // где t = clamp((values[i] - cmin) / (cmax - cmin), 0, 1). Длина values = n_points.
    // Используется для FastSynchro colored trajectory. При nullptr — старое поведение
    // (uniform color = `color`). points_mode и shader-line режим игнорируют это поле.
    const float*    values = nullptr;
    HeatmapColormap colormap = HeatmapColormap::Viridis;
    float           cmin = 0.0f;
    float           cmax = 1.0f;

    // Optional painter's-algorithm draw order for line segments (только в
    // imdraw_lines режиме). Длина = n_points - 1; segment_order[k] = индекс
    // segment'а, который должен быть нарисован k-м (0 = первый = задний,
    // последний = передний). Используется FastSync для depth-sorting по
    // непоказываемой оси (трёхмерный аттрактор спроецированный на XY).
    // При nullptr — рисуем в порядке возрастания индекса (как раньше).
    const int*      segment_order = nullptr;
};

class Plot2DView {
public:
    AxisInfo x_axis;
    AxisInfo y_axis;
    bool show_legend = true;
    bool view_valid = false;
    int  series_generation = -1;

    // ��������� ��������� ��� ��� ������� (������� ����������):
    //  pad_*       � ��������� 5% ������ �� ��� ��� ��������.
    //  show_zero_* � �������� ����� ���� �� ��� (x=0 ������������, y=0 ��������������).
    // 2D �������: �� true. Time domain: �� X ������� � ��� ��������� (pad_x/show_zero_x=false).
    bool pad_x = true;
    bool pad_y = true;
    bool show_zero_x = true;
    bool show_zero_y = true;

    // Принудительный X-диапазон для autofit. Если x_fit_use_explicit=true,
    // do_autofit/fit_x возьмут (x_fit_min, x_fit_max) вместо bbox данных.
    // Нужно для BD/LLE/LS: X-ось должна охватывать ВЕСЬ sweep-диапазон,
    // даже если часть параметров диверговала и точки отсутствуют.
    bool   x_fit_use_explicit = false;
    double x_fit_min = 0.0;
    double x_fit_max = 1.0;

    // true: ������ ��� GL_POINTS (��� 1D-�����������).
    // false (��-���������): GL_LINE_STRIP, ��� ��� ��� ��������� ����������.
    bool points_mode = false;
    float point_size_px = 2.0f;
    // Форма маркера в points_mode (PointMarker). -1 (дефолт) — прежний
    // сплошной квадратный GL-пойнт без alpha-блендинга.
    int  point_marker = -1;
    // Толщина линий данных (когда points_mode=false). По умолчанию 1.5px —
    // совпадает со старым хардкодом, регрессии нет.
    float line_thickness_px = 1.5f;

    // true: линии рисуются через ImDrawList::AddPolyline (umеет толщину >1px
    // через триангуляцию + проходит ПОСЛЕ осей/сетки, поэтому данные оказываются
    // ПОВЕРХ). Включаем для LLE/LS, где `glLineWidth` бесполезен — драйверы
    // в core OpenGL клампят его до 1.0. false (default): GL_LINE_STRIP внутри
    // FBO, как раньше (Bif/Phase). points_mode игнорируется этим флагом — оно
    // для точечных режимов остаётся через GL.
    bool imdraw_lines = false;

    // Snap X-координаты курсора к узлам параметрической сетки (для 1D-графиков
    // Bif/LLE/LS, где значение считалось в N узлах на [snap_x_min, snap_x_max]).
    // Активируется при snap_x_to_grid && snap_x_n > 1. Y-координата остаётся
    // непрерывной. Если курсор вне [min, max] — snap отключается для этой
    // конкретной точки. Заполнять поля должен caller ПЕРЕД render'ом.
    bool   snap_x_to_grid = false;
    double snap_x_min = 0.0;
    double snap_x_max = 1.0;
    int    snap_x_n   = 0;

    // ��������� ����� � ���� � ������� ������ (������� ���������� ����� ����������).
    std::vector<bool> visible;

    // Опционально: callback для добавления custom-пунктов в right-click popup
    // (после стандартных Auto fit / Lock / Invert axis). Caller выставляет
    // лямбду перед каждым render(); plot view вызывает её внутри своего
    // BeginPopup/EndPopup. Используется, например, FastSync для "Invert depth
    // axis" пункта без необходимости open'ить отдельный popup.
    std::function<void()> popup_extras;

    // Crosshair-commit callback: fires on release of a crosshair gesture
    // (MMB or Shift+LMB) inside the plot, unless it was a double-click.
    // Argument is the world X under the cursor (snapped if snap_x_to_grid).
    // Plain LMB is reserved for pan and does NOT trigger this — the
    // Parametric habit of clicking a series to focus it must not enqueue
    // a phase run in Custom. Used by the Custom-tab 1D slice to drill
    // into Phase at the released parameter value.
    std::function<void(double world_x)> on_left_click;

    // Live-drag callback for the crosshair gesture (MMB or Shift+LMB).
    // Fires every frame while the button is held inside the plot, with
    // the current cursor's world X (snapped if snap_x_to_grid). Used by
    // the Custom-tab so the fix_x / fix_y crosshair on 1D plots follows
    // the cursor during a drag while the heavy recompute waits for
    // release (on_left_click).
    std::function<void(double world_x)> on_left_drag;

    // Vertical/horizontal crosshair overlay drawn on top of the data.
    // NaN disables the corresponding axis; both NaN (default) → nothing
    // drawn, zero cost. Used by the Custom-tab to mark the fix_x sweep
    // position on 1D slice plots (like ImPlot::PlotInfLines did before).
    // Double-stroked (black halo + coloured core). Default core colours
    // match the Custom-tab convention: vertical = X-sweep (blue),
    // horizontal = Y-sweep (orange) — so the same colour on a heatmap
    // and its slice tells you which axis the crosshair belongs to.
    // Callers can override the ImU32 colours per view before render().
    double crosshair_x = std::numeric_limits<double>::quiet_NaN();
    double crosshair_y = std::numeric_limits<double>::quiet_NaN();
    // ARGB (0xAA_RR_GG_BB) — matches IM_COL32 default layout on Windows.
    unsigned crosshair_x_color = 0xFF50A0FFu;  // blue-ish (X sweep)
    unsigned crosshair_y_color = 0xFFFF9028u;  // orange   (Y sweep)

    // render:
    //  global_visible � ��������� �� ������� �� (�������, ����� �� ��� ��������),
    //                   ����������� ������ ����, recompute �� �����.
    //  ��������� ��������� (������� ���� ��������) � ���� visible ����.
    //  ����: ����� ����� = global_visible[k] && visible[k].
    //  init_visible � ��������� �������� ��������� ��������� ��� ����� ����� �����.
    void render(PlotRenderer& renderer,
        ImVec2 avail_pos, ImVec2 avail_size,
        int owner_id,
        int data_generation,
        const std::vector<PlotSeriesInput>& series_in,
        const std::vector<bool>& init_visible,
        const std::vector<bool>& global_visible,
        bool fit_request = false);

private:
    GpuLineSeriesSet series_cache_;
    // Эффективная видимость серий на текущий кадр (visible[k] && global_visible[k]).
    // Обновляется в начале render(); do_autofit/fit_x/fit_y используют её,
    // чтобы НЕ включать скрытые серии в авто-диапазон.
    std::vector<bool> render_visible_mask_;
    // Snapshot of the previous frame's mask — compared each frame to detect
    // legend toggles. Any change (either on or off) fires an autofit so the
    // newly-visible series come into view instead of staying outside the
    // clamped-in range set by whatever was visible before.
    std::vector<bool> render_visible_mask_prev_;

    bool   rect_zoom_pending_ = false;
    bool   rect_zoom_active_ = false;
    double rect_zoom_x0_ = 0, rect_zoom_y0_ = 0;
    int    rect_zoom_mode_ = 0;
    float  rect_zoom_start_x_ = 0;
    float  rect_zoom_start_y_ = 0;

    void do_autofit();
    void fit_x();
    void fit_y();
};