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
    GLint  loc_mvp_2d_ = -1, loc_color_2d_ = -1, loc_point_size_2d_ = -1;
    GLint  loc_mvp_3d_ = -1, loc_color_3d_ = -1;

    GLuint vao_ = 0;

    GLint  saved_viewport_[4]{};
    GLint  saved_fbo_ = 0;
    GLboolean saved_depth_test_ = GL_FALSE;
};