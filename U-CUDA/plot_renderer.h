#pragma once
#include <glad/glad.h>
#include <cstdint>

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
    void draw_points(GLuint vbo, int point_count, const float mvp[16],
        const float color[4], float point_size);

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
    void draw_heatmap(GLuint tex, float vmin, float vmax, int colormap_id,
                      float uv_off_x, float uv_off_y,
                      float uv_scale_x, float uv_scale_y);

    // ������ 3D-����� (vbo � float[3] �� �������).
    void draw_line_3d(GLuint vbo, int point_count, const float mvp[16],
        const float color[4], float line_width);

    void end_frame();

    GLuint texture_id() const { return color_tex_; }
    int width()  const { return fbo_w_; }
    int height() const { return fbo_h_; }

private:
    void ensure_fbo(int w, int h, bool with_depth);
    void compile_shaders();
    void destroy_fbo();

    GLuint fbo_ = 0;
    GLuint color_tex_ = 0;
    GLuint depth_rbo_ = 0;
    bool   fbo_has_depth_ = false;
    int    fbo_w_ = 0;
    int    fbo_h_ = 0;

    GLuint program_2d_ = 0;
    GLuint program_3d_ = 0;
    GLuint program_heatmap_ = 0;
    GLint  loc_mvp_2d_ = -1, loc_color_2d_ = -1, loc_point_size_2d_ = -1;
    GLint  loc_mvp_3d_ = -1, loc_color_3d_ = -1;
    GLint  loc_heatmap_tex_ = -1, loc_heatmap_vmin_ = -1,
           loc_heatmap_vmax_ = -1, loc_heatmap_cmap_ = -1,
           loc_heatmap_uv_off_ = -1, loc_heatmap_uv_scale_ = -1;
    GLuint heatmap_vbo_ = 0;     // ленивая инициализация fullscreen quad

    GLuint vao_ = 0;

    GLint  saved_viewport_[4]{};
    GLint  saved_fbo_ = 0;
    GLboolean saved_depth_test_ = GL_FALSE;
};