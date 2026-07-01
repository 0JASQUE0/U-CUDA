#include "plot_renderer.h"
#include <cstdio>
#include <algorithm>

// ---- CPU-side colormap (зеркало GLSL fragment-полиномов draw_heatmap) ----
// Используется heatmap_view'ом для рисования colorbar'а и plot_view_2d'ом
// для per-segment окраски trajectory. Раньше жил в heatmap_view.cpp::ns{}.
namespace {
struct vec3f { float r, g, b; };
inline vec3f operator+(vec3f a, vec3f b) { return {a.r+b.r, a.g+b.g, a.b+b.b}; }
inline vec3f operator*(vec3f a, float s) { return {a.r*s, a.g*s, a.b*s}; }
inline vec3f operator*(float s, vec3f a) { return a*s; }

vec3f cmap_viridis(float t) {
    const vec3f c0 = {0.2777273272f, 0.0054872578f, 0.3340998020f};
    const vec3f c1 = {0.1057220655f, 1.4046380960f, 1.3845030177f};
    const vec3f c2 = {-0.330001533f, 0.214825727f, 0.092491715f};
    const vec3f c3 = {-4.634230600f, -5.799101469f, -19.33244091f};
    const vec3f c4 = {6.228269936f, 14.17993089f, 56.69055318f};
    const vec3f c5 = {4.776384997f, -13.74514904f, -65.35303153f};
    const vec3f c6 = {-5.435455319f, 4.645852612f, 26.31243947f};
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}
vec3f cmap_inferno(float t) {
    const vec3f c0 = {0.0002189403691f, 0.001651742368f, -0.01948089833f};
    const vec3f c1 = {0.1065134194f, 0.5639564368f, 3.932712388f};
    const vec3f c2 = {11.60249308f, -3.972853966f, -15.94239411f};
    const vec3f c3 = {-41.70399613f, 17.43639888f, 44.35414519f};
    const vec3f c4 = {77.16289500f, -33.40998897f, -81.80741196f};
    const vec3f c5 = {-71.31942380f, 32.62606027f, 73.20951466f};
    const vec3f c6 = {25.13112622f, -12.24266895f, -23.07032500f};
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}
vec3f cmap_turbo(float t) {
    const vec3f c0 = {0.13572138f, 0.09140261f, 0.10667330f};
    const vec3f c1 = {4.61539260f, 2.19418839f, 12.64194608f};
    const vec3f c2 = {-42.66032258f, 4.84296658f, -60.58204836f};
    const vec3f c3 = {132.13108234f, -14.18503333f, 110.36276771f};
    const vec3f c4 = {-152.94239396f, 4.27729857f, -89.90310912f};
    const vec3f c5 = {59.28637943f, 2.82956604f, 27.34824973f};
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*c5))));
}
vec3f cmap_gray(float t) { return {t, t, t}; }
} // namespace

ImU32 cmap_sample(float t, HeatmapColormap m) {
    t = std::min(std::max(t, 0.0f), 1.0f);
    vec3f c;
    switch (m) {
        case HeatmapColormap::Inferno: c = cmap_inferno(t); break;
        case HeatmapColormap::Turbo:   c = cmap_turbo(t);   break;
        case HeatmapColormap::Gray:    c = cmap_gray(t);    break;
        case HeatmapColormap::Viridis:
        default:                       c = cmap_viridis(t); break;
    }
    auto clamp01 = [](float v){ return std::min(std::max(v, 0.0f), 1.0f); };
    return IM_COL32((int)(clamp01(c.r) * 255.0f),
                    (int)(clamp01(c.g) * 255.0f),
                    (int)(clamp01(c.b) * 255.0f), 255);
}

static const char* VS_2D = R"(
#version 330 core
layout(location = 0) in vec2 a_pos;
uniform mat4 u_mvp;
uniform float u_point_size;
void main() {
    gl_Position = u_mvp * vec4(a_pos, 0.0, 1.0);
    gl_PointSize = u_point_size;
}
)";

static const char* VS_3D = R"(
#version 330 core
layout(location = 0) in vec3 a_pos;
uniform mat4 u_mvp;
void main() {
    gl_Position = u_mvp * vec4(a_pos, 1.0);
}
)";

// Geometry shader для thick 3D-линий. Каждая пара соседних вершин
// GL_LINE_STRIP развёртывается в 4 клип-space вершины (triangle_strip)
// шириной u_thickness_px в пикселях, aspect-корректно. gl_Position.z
// сохраняем — depth test работает как для обычных линий, occlusion
// корректный. Сегменты с точками за камерой (w<=0) пропускаем — иначе
// perspective divide даёт бред и линия «взрывается» через весь экран.
static const char* GS_3D_THICK = R"(
#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;
uniform vec2  u_viewport;
uniform float u_thickness_px;
void main() {
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    if (p0.w <= 0.0 || p1.w <= 0.0) return;
    vec2 p0_ndc = p0.xy / p0.w;
    vec2 p1_ndc = p1.xy / p1.w;
    vec2 dir_px = (p1_ndc - p0_ndc) * (u_viewport * 0.5);
    float len_px = length(dir_px);
    if (len_px < 1e-6) return;
    vec2 dir_n = dir_px / len_px;
    vec2 nrm_px = vec2(-dir_n.y, dir_n.x);
    vec2 off_ndc = (nrm_px * (u_thickness_px * 0.5)) / (u_viewport * 0.5);
    gl_Position = vec4((p0_ndc + off_ndc) * p0.w, p0.z, p0.w); EmitVertex();
    gl_Position = vec4((p0_ndc - off_ndc) * p0.w, p0.z, p0.w); EmitVertex();
    gl_Position = vec4((p1_ndc + off_ndc) * p1.w, p1.z, p1.w); EmitVertex();
    gl_Position = vec4((p1_ndc - off_ndc) * p1.w, p1.z, p1.w); EmitVertex();
    EndPrimitive();
}
)";

static const char* FS = R"(
#version 330 core
uniform vec4 u_color;
out vec4 frag_color;
void main() {
    frag_color = u_color;
}
)";

// Хитмапа: fullscreen quad с texcoords, без MVP (clip-space позиции).
static const char* VS_HEATMAP = R"(
#version 330 core
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_uv = a_uv;
}
)";

// Colormap'ы — полиномиальные приближения (известные fits matplotlib-таблиц,
// 6-я степень). По времени работы — single fma-цепочка, цвета визуально
// неотличимы от LUT-варианта на 256 цветах. Спец-значения (≥1e30, NaN, Inf)
// рисуются тёмным фоном — engine помечает diverged ячейки этим маркером.
static const char* FS_HEATMAP = R"(
#version 330 core
in vec2 v_uv;
uniform sampler2D u_tex;
uniform float u_vmin;
uniform float u_vmax;
uniform int   u_colormap;
uniform int   u_discrete_n;  // 0 = continuous, N>0 = quantize t into N bands
uniform vec2  u_uv_off;
uniform vec2  u_uv_scale;
out vec4 frag_color;

vec3 viridis(float t) {
    const vec3 c0 = vec3(0.2777273272, 0.0054872578, 0.3340998020);
    const vec3 c1 = vec3(0.1057220655, 1.4046380960, 1.3845030177);
    const vec3 c2 = vec3(-0.330001533, 0.214825727, 0.092491715);
    const vec3 c3 = vec3(-4.634230600, -5.799101469, -19.33244091);
    const vec3 c4 = vec3(6.228269936, 14.17993089, 56.69055318);
    const vec3 c5 = vec3(4.776384997, -13.74514904, -65.35303153);
    const vec3 c6 = vec3(-5.435455319, 4.645852612, 26.31243947);
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}
vec3 inferno(float t) {
    const vec3 c0 = vec3(0.0002189403691, 0.001651742368, -0.01948089833);
    const vec3 c1 = vec3(0.1065134194, 0.5639564368, 3.932712388);
    const vec3 c2 = vec3(11.60249308, -3.972853966, -15.94239411);
    const vec3 c3 = vec3(-41.70399613, 17.43639888, 44.35414519);
    const vec3 c4 = vec3(77.16289500, -33.40998897, -81.80741196);
    const vec3 c5 = vec3(-71.31942380, 32.62606027, 73.20951466);
    const vec3 c6 = vec3(25.13112622, -12.24266895, -23.07032500);
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}
vec3 turbo(float t) {
    // Canonical Google Research / MATLAB Turbo, 5th-degree polynomial fit
    // (matplotlib turbo). Previous 6th-degree approximation gave P(1) ~=
    // (0.54, 0.83, -0.19) -> yellow-green after clamp; this one ends at
    // ~ (0.74, 0.21, 0.18) which is the proper Turbo bright red.
    const vec3 c0 = vec3(0.13572138, 0.09140261, 0.10667330);
    const vec3 c1 = vec3(4.61539260, 2.19418839, 12.64194608);
    const vec3 c2 = vec3(-42.66032258, 4.84296658, -60.58204836);
    const vec3 c3 = vec3(132.13108234, -14.18503333, 110.36276771);
    const vec3 c4 = vec3(-152.94239396, 4.27729857, -89.90310912);
    const vec3 c5 = vec3(59.28637943, 2.82956604, 27.34824973);
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*c5))));
}
void main() {
    vec2 uv = v_uv * u_uv_scale + u_uv_off;
    // Если view вышел за пределы данных — рисуем фон, чтобы пользователю
    // были видны границы реального датасета.
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        frag_color = vec4(0.08, 0.08, 0.10, 1.0);
        return;
    }
    float v = texture(u_tex, uv).r;
    if (v >= 1.0e30 || isnan(v) || isinf(v)) {
        frag_color = vec4(0.12, 0.12, 0.14, 1.0);
        return;
    }
    float range = u_vmax - u_vmin;
    float t = (range > 1e-30) ? clamp((v - u_vmin) / range, 0.0, 1.0) : 0.5;
    // Discrete mode: quantize t into u_discrete_n bands. Sample the colormap
    // at the band's edge-aligned position k/(N-1) so the first band picks
    // t=0 (e.g. blue end of Turbo) and the last band picks t=1 (red end).
    // Band-center sampling (k+0.5)/N would shrink the endpoints inward and
    // make discrete and continuous modes show different extreme colors.
    if (u_discrete_n > 0) {
        float n = float(u_discrete_n);
        float k = floor(t * n);
        if (k >= n) k = n - 1.0;
        t = (n > 1.0) ? (k / (n - 1.0)) : 0.5;
    }
    vec3 col;
    if      (u_colormap == 0) col = viridis(t);
    else if (u_colormap == 1) col = inferno(t);
    else if (u_colormap == 2) col = turbo(t);
    else                       col = vec3(t);
    frag_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
)";

static GLuint compile(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024]; GLsizei len = 0;
        glGetShaderInfoLog(sh, sizeof(log), &len, log);
        fprintf(stderr, "[PlotRenderer] shader compile error: %.*s\n", len, log);
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

static GLuint link_program(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024]; GLsizei len = 0;
        glGetProgramInfoLog(p, sizeof(log), &len, log);
        fprintf(stderr, "[PlotRenderer] link error: %.*s\n", len, log);
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

static GLuint link_program_3(GLuint vs, GLuint gs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, gs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024]; GLsizei len = 0;
        glGetProgramInfoLog(p, sizeof(log), &len, log);
        fprintf(stderr, "[PlotRenderer] link error (3-stage): %.*s\n", len, log);
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

PlotRenderer::PlotRenderer() {
    compile_shaders();
    glGenVertexArrays(1, &vao_);
}

PlotRenderer::~PlotRenderer() {
    destroy_fbo();
    if (program_2d_)       glDeleteProgram(program_2d_);
    if (program_3d_)       glDeleteProgram(program_3d_);
    if (program_3d_thick_) glDeleteProgram(program_3d_thick_);
    if (program_heatmap_)  glDeleteProgram(program_heatmap_);
    if (heatmap_vbo_)     glDeleteBuffers(1, &heatmap_vbo_);
    if (vao_)             glDeleteVertexArrays(1, &vao_);
}

void PlotRenderer::compile_shaders() {
    GLuint fs = compile(GL_FRAGMENT_SHADER, FS);
    GLuint vs2 = compile(GL_VERTEX_SHADER, VS_2D);
    GLuint vs3 = compile(GL_VERTEX_SHADER, VS_3D);
    GLuint gs3t = compile(GL_GEOMETRY_SHADER, GS_3D_THICK);
    if (fs && vs2) {
        program_2d_ = link_program(vs2, fs);
        if (program_2d_) {
            loc_mvp_2d_ = glGetUniformLocation(program_2d_, "u_mvp");
            loc_color_2d_ = glGetUniformLocation(program_2d_, "u_color");
            loc_point_size_2d_ = glGetUniformLocation(program_2d_, "u_point_size");
        }
    }
    if (fs && vs3) {
        program_3d_ = link_program(vs3, fs);
        if (program_3d_) {
            loc_mvp_3d_ = glGetUniformLocation(program_3d_, "u_mvp");
            loc_color_3d_ = glGetUniformLocation(program_3d_, "u_color");
        }
    }
    if (fs && vs3 && gs3t) {
        program_3d_thick_ = link_program_3(vs3, gs3t, fs);
        if (program_3d_thick_) {
            loc_mvp_3d_thick_       = glGetUniformLocation(program_3d_thick_, "u_mvp");
            loc_color_3d_thick_     = glGetUniformLocation(program_3d_thick_, "u_color");
            loc_viewport_3d_thick_  = glGetUniformLocation(program_3d_thick_, "u_viewport");
            loc_thickness_3d_thick_ = glGetUniformLocation(program_3d_thick_, "u_thickness_px");
        }
    }
    GLuint fs_h = compile(GL_FRAGMENT_SHADER, FS_HEATMAP);
    GLuint vs_h = compile(GL_VERTEX_SHADER, VS_HEATMAP);
    if (fs_h && vs_h) {
        program_heatmap_ = link_program(vs_h, fs_h);
        if (program_heatmap_) {
            loc_heatmap_tex_      = glGetUniformLocation(program_heatmap_, "u_tex");
            loc_heatmap_vmin_     = glGetUniformLocation(program_heatmap_, "u_vmin");
            loc_heatmap_vmax_     = glGetUniformLocation(program_heatmap_, "u_vmax");
            loc_heatmap_cmap_     = glGetUniformLocation(program_heatmap_, "u_colormap");
            loc_heatmap_uv_off_   = glGetUniformLocation(program_heatmap_, "u_uv_off");
            loc_heatmap_uv_scale_ = glGetUniformLocation(program_heatmap_, "u_uv_scale");
            loc_heatmap_discrete_n_ = glGetUniformLocation(program_heatmap_, "u_discrete_n");
        }
    }
    if (vs2)  glDeleteShader(vs2);
    if (vs3)  glDeleteShader(vs3);
    if (gs3t) glDeleteShader(gs3t);
    if (vs_h) glDeleteShader(vs_h);
    if (fs_h) glDeleteShader(fs_h);
    if (fs)   glDeleteShader(fs);
}

void PlotRenderer::destroy_fbo() {
    if (color_tex_) { glDeleteTextures(1, &color_tex_); color_tex_ = 0; }
    if (depth_rbo_) { glDeleteRenderbuffers(1, &depth_rbo_); depth_rbo_ = 0; }
    if (fbo_) { glDeleteFramebuffers(1, &fbo_);   fbo_ = 0; }
    fbo_w_ = fbo_h_ = 0;
    fbo_has_depth_ = false;
}

void PlotRenderer::ensure_fbo(int w, int h, bool with_depth) {
    if (w == fbo_w_ && h == fbo_h_ && fbo_ && fbo_has_depth_ == with_depth) return;
    destroy_fbo();
    fbo_w_ = w; fbo_h_ = h;
    fbo_has_depth_ = with_depth;

    glGenTextures(1, &color_tex_);
    glBindTexture(GL_TEXTURE_2D, color_tex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenFramebuffers(1, &fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D, color_tex_, 0);

    if (with_depth) {
        glGenRenderbuffers(1, &depth_rbo_);
        glBindRenderbuffer(GL_RENDERBUFFER, depth_rbo_);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
            GL_RENDERBUFFER, depth_rbo_);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
        fprintf(stderr, "[PlotRenderer] FBO incomplete: 0x%x\n", status);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void PlotRenderer::begin_frame(int w, int h, float r, float g, float b, float a, bool with_depth) {
    if (w < 1) w = 1;
    if (h < 1) h = 1;
    glGetIntegerv(GL_VIEWPORT, saved_viewport_);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &saved_fbo_);
    saved_depth_test_ = glIsEnabled(GL_DEPTH_TEST);

    ensure_fbo(w, h, with_depth);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glViewport(0, 0, w, h);
    glClearColor(r, g, b, a);
    if (with_depth) {
        glEnable(GL_DEPTH_TEST);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    else {
        glDisable(GL_DEPTH_TEST);
        glClear(GL_COLOR_BUFFER_BIT);
    }
}

void PlotRenderer::draw_line(GLuint vbo, int point_count, const float mvp[16],
    const float color[4], float line_width) {
    if (!program_2d_ || point_count < 2 || !vbo) return;
    glUseProgram(program_2d_);
    glUniformMatrix4fv(loc_mvp_2d_, 1, GL_FALSE, mvp);
    glUniform4fv(loc_color_2d_, 1, color);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glLineWidth(line_width);
    glDrawArrays(GL_LINE_STRIP, 0, point_count);
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);
}

void PlotRenderer::draw_points(GLuint vbo, int point_count, const float mvp[16],
    const float color[4], float point_size) {
    if (!program_2d_ || point_count < 1 || !vbo) return;
    glUseProgram(program_2d_);
    glUniformMatrix4fv(loc_mvp_2d_, 1, GL_FALSE, mvp);
    glUniform4fv(loc_color_2d_, 1, color);
    if (loc_point_size_2d_ >= 0) glUniform1f(loc_point_size_2d_, point_size);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    // В core profile размер берётся из gl_PointSize в шейдере только при
    // GL_PROGRAM_POINT_SIZE; иначе драйверы часто клампят до 1 px.
    glEnable(GL_PROGRAM_POINT_SIZE);
    glDrawArrays(GL_POINTS, 0, point_count);
    glDisable(GL_PROGRAM_POINT_SIZE);
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);
}

void PlotRenderer::draw_heatmap(GLuint tex, float vmin, float vmax, int colormap_id,
                                float uv_off_x, float uv_off_y,
                                float uv_scale_x, float uv_scale_y,
                                int n_discrete) {
    if (!program_heatmap_ || !tex) return;
    if (!heatmap_vbo_) {
        // Fullscreen triangle-strip: 4 точки × (pos.xy, uv.xy). Текстурные
        // координаты — нативные [0,1]: tex[0,0] = нижний-левый угол FBO.
        static const float quad[] = {
            -1.0f, -1.0f, 0.0f, 0.0f,
             1.0f, -1.0f, 1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f, 1.0f,
             1.0f,  1.0f, 1.0f, 1.0f,
        };
        glGenBuffers(1, &heatmap_vbo_);
        glBindBuffer(GL_ARRAY_BUFFER, heatmap_vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    glUseProgram(program_heatmap_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    if (loc_heatmap_tex_      >= 0) glUniform1i(loc_heatmap_tex_, 0);
    if (loc_heatmap_vmin_     >= 0) glUniform1f(loc_heatmap_vmin_, vmin);
    if (loc_heatmap_vmax_     >= 0) glUniform1f(loc_heatmap_vmax_, vmax);
    if (loc_heatmap_cmap_     >= 0) glUniform1i(loc_heatmap_cmap_, colormap_id);
    if (loc_heatmap_uv_off_   >= 0) glUniform2f(loc_heatmap_uv_off_, uv_off_x, uv_off_y);
    if (loc_heatmap_uv_scale_ >= 0) glUniform2f(loc_heatmap_uv_scale_, uv_scale_x, uv_scale_y);
    if (loc_heatmap_discrete_n_ >= 0) glUniform1i(loc_heatmap_discrete_n_, n_discrete);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, heatmap_vbo_);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (void*)(2 * sizeof(float)));
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
}

void PlotRenderer::draw_line_3d(GLuint vbo, int point_count, const float mvp[16],
    const float color[4], float line_width, bool thick_style) {
    if (point_count < 2 || !vbo) return;

    // Fallback на старый путь если thick program не собрался (нет поддержки GS).
    const bool use_thick = thick_style && program_3d_thick_ != 0;

    if (!use_thick) {
        // --- СТАРЫЙ путь: побайтово как до Custom line style patch ---
        if (!program_3d_) return;
        glUseProgram(program_3d_);
        glUniformMatrix4fv(loc_mvp_3d_, 1, GL_FALSE, mvp);
        glUniform4fv(loc_color_3d_, 1, color);
        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glLineWidth(line_width);
        glDrawArrays(GL_LINE_STRIP, 0, point_count);
        glDisableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        glUseProgram(0);
        return;
    }

    // --- ТОЛСТЫЙ путь: geometry shader раскрывает сегменты в quads ---
    // Alpha blending включаем только тут, чтобы OFF-путь не менял GL state.
    GLboolean was_blend = glIsEnabled(GL_BLEND);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(program_3d_thick_);
    glUniformMatrix4fv(loc_mvp_3d_thick_, 1, GL_FALSE, mvp);
    glUniform4fv(loc_color_3d_thick_, 1, color);
    if (loc_viewport_3d_thick_ >= 0)
        glUniform2f(loc_viewport_3d_thick_, (float)fbo_w_, (float)fbo_h_);
    if (loc_thickness_3d_thick_ >= 0)
        glUniform1f(loc_thickness_3d_thick_, line_width);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glDrawArrays(GL_LINE_STRIP, 0, point_count);
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);

    // Точный blend func не восстанавливаем — ImGui сам ставит свой перед
    // отрисовкой ImDrawList, "утечка" не важна.
    if (!was_blend) glDisable(GL_BLEND);
}

void PlotRenderer::end_frame() {
    glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)saved_fbo_);
    glViewport(saved_viewport_[0], saved_viewport_[1],
        saved_viewport_[2], saved_viewport_[3]);
    if (saved_depth_test_) glEnable(GL_DEPTH_TEST); else glDisable(GL_DEPTH_TEST);
}