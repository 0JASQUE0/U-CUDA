#pragma once
#include <glad/glad.h>
#include <cstdint>
#include "imgui.h"   // ImU32 для cmap_sample

// 2D scalar → RGB colormap. Раньше жил в heatmap_view.h; перемещён в
// plot_renderer.h, потому что:
//   1) `PlotRenderer::draw_heatmap` уже его принимает (int colormap_id);
//   2) `plot_view_2d.cpp` теперь использует то же сэмплирование для
//      per-segment colored trajectory.
enum class HeatmapColormap : int {
    Viridis = 0,
    Inferno = 1,
    Turbo   = 2,
    Gray    = 3,
    // 4-8: не полиномиальные — сэмплятся из LUT-текстуры (256x5 RGB8), не
    // из полиномов (эти colormap'ы matplotlib слишком немонотонны/многоэкс-
    // тремумны для низкостепенного fit'а). См. colormap_lut_data.h.
    GistStern    = 4,
    GnuPlot      = 5,
    GistRainbow  = 6,
    NipySpectral = 7,
    GistNcar     = 8,
};

// Общий список имён для ImGui::Combo — единственный источник истины (раньше
// было по копии "Viridis"/"Inferno"/"Turbo"/"Gray" в каждом combo по коду;
// добавление новой colormap правится теперь в одном месте).
extern const char* const kHeatmapColormapNames[9];
constexpr int kHeatmapColormapCount = 9;

// Форма маркера для точечного (points_mode) 2D-рендера. -1 в draw_points /
// Plot2DView::point_marker = «без маски», т.е. дефолтный сплошной квадратный
// GL-пойнт — старый путь остаётся пиксель-в-пиксель прежним.
enum class PointMarker : int {
    Circle       = 0,
    Square       = 1,
    Diamond      = 2,
    TriangleUp   = 3,
    TriangleDown = 4,
    Cross        = 5,
    Plus         = 6,
};

// Общий список имён для ImGui::Combo — единственный источник истины
// (см. kHeatmapColormapNames выше).
extern const char* const kPointMarkerNames[7];
constexpr int kPointMarkerCount = 7;

// CPU-side колормап. Полиномиальные (0-3) — точное зеркало GLSL из
// draw_heatmap (см. plot_renderer.cpp::compile_shaders), цвета совпадают
// bit-for-bit. LUT-based (4-8) — линейная интерполяция между соседними
// элементами таблицы (визуально идентично GPU-пути, не гарантированно
// bit-exact). t clamp'ится в [0,1]. Возвращает ImU32 (используется ImDrawList).
ImU32 cmap_sample(float t, HeatmapColormap m);

class PlotRenderer {
public:
    PlotRenderer();
    ~PlotRenderer();
    PlotRenderer(const PlotRenderer&) = delete;
    PlotRenderer& operator=(const PlotRenderer&) = delete;

    // with_depth=true ������ depth-attachment � �������� GL_DEPTH_TEST.
    // ������������ ��� 3D. ��� 2D ��������� false (�� ���������).
    void begin_frame(int w, int h, float clear_r, float clear_g, float clear_b, float clear_a,
        bool with_depth = false);

    // ������ 2D-����� (vbo � float[2] �� �������).
    void draw_line(GLuint vbo, int point_count, const float mvp[16],
        const float color[4], float line_width);

    // ������ 2D-����� (vbo � float[2] �� �������) ��� GL_POINTS.
    // ��� ������������ ��� ������� 1D-�����������.
    // marker < 0 — старый путь: сплошной квадрат, GL-состояние не трогается.
    // marker >= 0 (PointMarker) — шейдерная маска формы + alpha-блендинг
    // (color[3] перестаёт игнорироваться).
    void draw_points(GLuint vbo, int point_count, const float mvp[16],
        const float color[4], float point_size, int marker = -1);

    // Хитмапа: рендерит fullscreen-quad внутри текущего FBO (begin_frame),
    // сэмплит R32F-текстуру tex и применяет colormap.
    //   colormap_id: 0=Viridis, 1=Inferno, 2=Turbo, 3=Gray.
    //   uv_off/uv_scale: маппинг fullscreen-quad UV [0,1] в UV данных:
    //     uv_data = v_uv * uv_scale + uv_off
    //   используется для zoom/pan — view ⊂ data ставит scale<1 + offset>0;
    //   view ⊃ data → UV вылезет за [0,1], CLAMP_TO_BORDER (см. ensure_tex
    //   в HeatmapView) даст тёмный фон, чтобы пользователь видел границы.
    // Спец-значения: ячейки со значением >= 1e30, NaN или Inf шейдер
    // отображает тёмно-серым (используется engine'ом для diverged/spec).
    // n_discrete: 0 = continuous shading, N>0 = quantize into N color bands.
    // reverse: true -> t := 1-t перед сэмплированием colormap'а (после
    // discrete-квантования), т.е. разворачивает градиент целиком.
    void draw_heatmap(GLuint tex, float vmin, float vmax, int colormap_id,
                      float uv_off_x, float uv_off_y,
                      float uv_scale_x, float uv_scale_y,
                      int n_discrete = 0, bool reverse = false);

    // ������ 3D-����� (vbo � float[3] �� �������).
    // thick_style=false — старый быстрый путь: program_3d_ + glLineWidth
    // (в core-profile драйвер обычно клампит до 1px, α не блендится).
    // thick_style=true — раскрываем сегменты в screen-aligned quads через
    // geometry shader, включаем BLEND для честного alpha compositing.
    // Depth test работает в обоих случаях.
    void draw_line_3d(GLuint vbo, int point_count, const float mvp[16],
        const float color[4], float line_width, bool thick_style = false);

    void end_frame();

    GLuint texture_id() const { return color_tex_; }
    int width()  const { return fbo_w_; }
    int height() const { return fbo_h_; }

private:
    void ensure_fbo(int w, int h, bool with_depth);
    void compile_shaders();
    void destroy_fbo();
    // Одна 256x5 RGB8-текстура на все 5 LUT-colormap'ов (см. HeatmapColormap
    // 4-8) — строки в порядке GistStern/GnuPlot/GistRainbow/NipySpectral/
    // GistNcar. Создаётся один раз в конструкторе, как и шейдеры.
    void ensure_lut_texture();

    GLuint fbo_ = 0;
    GLuint color_tex_ = 0;
    GLuint depth_rbo_ = 0;
    bool   fbo_has_depth_ = false;
    int    fbo_w_ = 0;
    int    fbo_h_ = 0;

    GLuint program_2d_ = 0;
    GLuint program_points_ = 0;   // VS_2D + FS_POINT (маска формы маркера)
    GLuint program_3d_ = 0;
    GLuint program_3d_thick_ = 0;
    GLuint program_heatmap_ = 0;
    GLint  loc_mvp_2d_ = -1, loc_color_2d_ = -1, loc_point_size_2d_ = -1;
    GLint  loc_mvp_points_ = -1, loc_color_points_ = -1,
           loc_point_size_points_ = -1, loc_marker_points_ = -1;
    GLint  loc_mvp_3d_ = -1, loc_color_3d_ = -1;
    GLint  loc_mvp_3d_thick_ = -1, loc_color_3d_thick_ = -1,
           loc_viewport_3d_thick_ = -1, loc_thickness_3d_thick_ = -1;
    GLint  loc_heatmap_tex_ = -1, loc_heatmap_vmin_ = -1,
           loc_heatmap_vmax_ = -1, loc_heatmap_cmap_ = -1,
           loc_heatmap_uv_off_ = -1, loc_heatmap_uv_scale_ = -1,
           loc_heatmap_discrete_n_ = -1, loc_heatmap_lut_ = -1,
           loc_heatmap_reverse_ = -1;
    GLuint heatmap_vbo_ = 0;     // ленивая инициализация fullscreen quad
    GLuint lut_tex_ = 0;         // 256x5 RGB8, см. ensure_lut_texture()

    GLuint vao_ = 0;

    GLint  saved_viewport_[4]{};
    GLint  saved_fbo_ = 0;
    GLboolean saved_depth_test_ = GL_FALSE;
};