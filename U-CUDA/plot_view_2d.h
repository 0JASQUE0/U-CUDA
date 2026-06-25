#pragma once
#include "plot_axis.h"
#include "plot_legend.h"
#include "gpu_line_series.h"
#include "plot_renderer.h"
#include <vector>
#include <string>

struct PlotSeriesInput {
    const float* points = nullptr;
    int          n_points = 0;
    ImVec4       color = ImVec4(1, 1, 1, 1);
    std::string  label;
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

    // ��������� ����� � ���� � ������� ������ (������� ���������� ����� ����������).
    std::vector<bool> visible;

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