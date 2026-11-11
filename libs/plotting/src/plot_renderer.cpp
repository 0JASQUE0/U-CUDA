#include "ucuda/plotting/plot_renderer.h"
// Таблица цветов slanCM (200 x 256 RGB8) + имена + категории. Файл большой
// (~700 КБ), поэтому включается только здесь — наружу всё уходит через
// объявленные в plot_renderer.h функции.
#include "ucuda/plotting/colormap_slancm_data.h"
#include <cstdio>
#include <algorithm>

// CPU-side colormap (то же сэмплирование, что в шейдере draw_heatmap)
// Используется heatmap_view'ом для рисования colorbar'а и plot_view_2d'ом
// для per-segment окраски trajectory. Раньше жил в heatmap_view.cpp::ns{}.
namespace {
struct vec3f { float r, g, b; };
inline vec3f operator+(vec3f a, vec3f b) { return {a.r+b.r, a.g+b.g, a.b+b.b}; }
inline vec3f operator*(vec3f a, float s) { return {a.r*s, a.g*s, a.b*s}; }

// LUT-based colormap (slanCM): линейная интерполяция между соседними записями
// 256-элементной таблицы. Визуально соответствует GPU-пути (bilinear sample
// текстуры на той же таблице), но не гарантированно bit-exact, в отличие от
// полиномиальных 0-3.
vec3f cmap_lut_sample(float t, const unsigned char lut[256][3]) {
    float pos = t * 255.0f;
    int i0 = (int)pos;
    if (i0 < 0) i0 = 0; else if (i0 > 255) i0 = 255;
    int i1 = (i0 < 255) ? i0 + 1 : 255;
    float frac = pos - (float)i0;
    auto to_f = [](const unsigned char* c) -> vec3f {
        return { c[0] / 255.0f, c[1] / 255.0f, c[2] / 255.0f };
    };
    return to_f(lut[i0]) * (1.0f - frac) + to_f(lut[i1]) * frac;
}
} // namespace

// Реестр slanCM: имена/категории, миграция легаси-id, набор включённых карт.

const char* slancm_name(int n) {
    return (n >= 1 && n <= kSlanCmCount) ? kSlanCmNames[n - 1] : "";
}

int slancm_category(int n) {
    return (n >= 1 && n <= kSlanCmCount) ? (int)kSlanCmCategory[n - 1] : -1;
}

const char* slancm_category_name(int cat) {
    return (cat >= 0 && cat < kSlanCmCategoryCount) ? kSlanCmCategoryNames[cat] : "";
}

const char* colormap_label(int id) {
    // Подписи строятся один раз: ImGui хочет const char*, а собирать
    // "151  gist_stern" каждый кадр на 200 строк — лишний мусор в аллокаторе.
    static const std::vector<std::string> labels = [] {
        std::vector<std::string> v((size_t)kSlanCmCount);
        for (int n = 1; n <= kSlanCmCount; ++n)
            v[n - 1] = std::to_string(n) + "  " + kSlanCmNames[n - 1];
        return v;
    }();
    // Мигрируем на всякий случай: все живые пути уже прогоняют id через
    // colormap_id_or, но подпись не должна превращаться в "?" даже если
    // где-то просочится сырое значение из старой сессии.
    const int v = colormap_id_or(id, (int)kDefaultColormap);
    return labels[slancm_number(v) - 1].c_str();
}

int colormap_id_or(int id, int fallback) {
    // Легаси 0..8 -> те же самые карты в нумерации slanCM (см. комментарий
    // к HeatmapColormap): старые сессии открываются как раньше, просто под
    // родными именами карт и без дублей в пикере.
    auto migrate = [](int v) {
        switch ((HeatmapColormap)v) {
            case HeatmapColormap::LegacyViridis:      return slancm_id(1);
            case HeatmapColormap::LegacyInferno:      return slancm_id(3);
            case HeatmapColormap::LegacyTurbo:        return slancm_id(168);
            case HeatmapColormap::LegacyGray:         return slancm_id(28);
            case HeatmapColormap::LegacyGistStern:    return slancm_id(151);
            case HeatmapColormap::LegacyGnuPlot:      return slancm_id(152);
            case HeatmapColormap::LegacyGistRainbow:  return slancm_id(162);
            case HeatmapColormap::LegacyNipySpectral: return slancm_id(170);
            case HeatmapColormap::LegacyGistNcar:     return slancm_id(171);
            default: break;
        }
        return v;
    };
    const int m = migrate(id);
    if (is_slancm_id(m)) return m;
    const int f = migrate(fallback);
    if (is_slancm_id(f)) return f;
    return (int)kDefaultColormap;
}

namespace {
// Маска включённых карт + производный от неё список для пикера. Глобальные,
// как tick_precision у осей: пикер зовётся из десятка мест в gui.cpp, и тащить
// туда AppModel ради одного набора — лишняя связность. Источник истины всё
// равно AppModel/AppConfig, здесь только зеркало (см. set_enabled_slancm).
std::string      g_slancm_mask;
std::vector<int> g_picker_ids;

void rebuild_picker_ids() {
    g_picker_ids.clear();
    g_picker_ids.reserve((size_t)kSlanCmCount);
    for (int n = 1; n <= kSlanCmCount; ++n)
        if (g_slancm_mask[n - 1] == '1') g_picker_ids.push_back(slancm_id(n));
}
} // namespace

std::string default_enabled_slancm() {
    std::string m((size_t)kSlanCmCount, '0');
    // Ровно те девять, что были доступны до Settings: четыре бывшие встроенные
    // (легаси-id 0..3) и пять бывших LUT-карт (легаси-id 4..8).
    for (int n : {1, 3, 28, 151, 152, 162, 168, 170, 171}) m[n - 1] = '1';
    return m;
}

void set_enabled_slancm(const std::string& mask) {
    // Собираем в отдельной строке, а не правим g_slancm_mask на месте: mask
    // вполне может БЫТЬ этой же строкой (вызов из picker_colormap_ids ниже
    // раньше делал ровно это), и правка под чтением затирала бы источник.
    // Короткая маска (конфиг от старой версии) дополняется нулями, длинная
    // обрезается — так добавление карт в будущем не сломает чтение конфига.
    std::string next((size_t)kSlanCmCount, '0');
    const size_t n = std::min(mask.size(), (size_t)kSlanCmCount);
    for (size_t i = 0; i < n; ++i) next[i] = (mask[i] == '1') ? '1' : '0';
    g_slancm_mask = std::move(next);
    rebuild_picker_ids();
}

const std::vector<int>& picker_colormap_ids() {
    // Проверяем МАСКУ, а не готовый список: пустой список при заданной маске —
    // законное состояние (пользователь снял все галочки), и перестраивать его
    // каждый кадр незачем. Пустая же маска значит, что bootstrap не выполнялся
    // — так бывает при первом запуске (конфига ещё нет) и в Debug-сборке, где
    // main живёт в main_NonLinAnal.cu. Без этого пикер оказался бы пустым.
    if (g_slancm_mask.size() != (size_t)kSlanCmCount)
        set_enabled_slancm(default_enabled_slancm());
    return g_picker_ids;
}

const char* const kPointMarkerNames[7] = {
    "Circle", "Square", "Diamond", "Triangle up", "Triangle down", "Cross", "Plus",
};

ImU32 cmap_sample_id(float t, int colormap_id) {
    t = std::min(std::max(t, 0.0f), 1.0f);
    const int id = colormap_id_or(colormap_id, (int)kDefaultColormap);
    const vec3f c = cmap_lut_sample(t, kSlanCmLut[slancm_number(id) - 1]);
    auto clamp01 = [](float v){ return std::min(std::max(v, 0.0f), 1.0f); };
    return IM_COL32((int)(clamp01(c.r) * 255.0f),
                    (int)(clamp01(c.g) * 255.0f),
                    (int)(clamp01(c.b) * 255.0f), 255);
}

ImU32 cmap_sample(float t, HeatmapColormap m) { return cmap_sample_id(t, (int)m); }

static const char* VS_2D = R"(
#version 330 core
layout(location = 0) in vec2 a_pos;
uniform mat4 u_mvp;
uniform float u_point_size;
uniform float u_aa_pad;   // запас спрайта под сглаженный край маркера (0, если не задан)
void main() {
    gl_Position = u_mvp * vec4(a_pos, 0.0, 1.0);
    gl_PointSize = max(u_point_size, 1.0) + u_aa_pad;
}
)";

static const char* VS_3D = R"(
#version 330 core
layout(location = 0) in vec3 a_pos;
uniform mat4 u_mvp;
uniform float u_point_size;
uniform float u_aa_pad;   // см. VS_2D
void main() {
    gl_Position = u_mvp * vec4(a_pos, 1.0);
    // Ignored for line/triangle draws; read only by draw_points_3d.
    gl_PointSize = max(u_point_size, 1.0) + u_aa_pad;
}
)";

// Geometry shader для thick 3D-линий. Каждая пара соседних вершин
// GL_LINE_STRIP развёртывается в 4 клип-space вершины (triangle_strip)
// шириной u_thickness_px в пикселях, aspect-корректно. gl_Position.z
// сохраняем — depth test работает как для обычных линий, occlusion
// корректный. Сегменты с точками за камерой (w<=0) пропускаем — иначе
// perspective divide даёт бред и линия «взрывается» через весь экран.
// Сглаживание толстых линий (FBO без MSAA): полоса раскрывается на 1 px шире с каждой стороны,
// а фрагментный шейдер гасит альфу по расстоянию от оси (aa_coverage). Линия тоньше 1 px
// рисуется шириной 1 px с альфой, пропорциональной толщине, — как у ImDrawList; иначе
// субпиксельная полоса растеризовалась бы пунктиром.
#define GS_AA_HALF_GLSL R"(
float aa_half_extent() { return 0.5 * max(u_thickness_px, 1.0) + 1.0; }
)"
#define FS_AA_COVERAGE_GLSL R"(
float aa_coverage() {
    float w = max(u_thickness_px, 1.0);
    return clamp(0.5 * w + 0.5 - abs(g_dist), 0.0, 1.0) * min(u_thickness_px, 1.0);
}
)"

static const char* GS_3D_THICK = R"(
#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;
noperspective out float g_dist;   // расстояние от оси линии, px (для сглаживания краёв)
uniform vec2  u_viewport;
uniform float u_thickness_px;
)" GS_AA_HALF_GLSL R"(
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
    float h = aa_half_extent();
    vec2 off_ndc = (nrm_px * h) / (u_viewport * 0.5);
    g_dist =  h; gl_Position = vec4((p0_ndc + off_ndc) * p0.w, p0.z, p0.w); EmitVertex();
    g_dist = -h; gl_Position = vec4((p0_ndc - off_ndc) * p0.w, p0.z, p0.w); EmitVertex();
    g_dist =  h; gl_Position = vec4((p1_ndc + off_ndc) * p1.w, p1.z, p1.w); EmitVertex();
    g_dist = -h; gl_Position = vec4((p1_ndc - off_ndc) * p1.w, p1.z, p1.w); EmitVertex();
    EndPrimitive();
}
)";

// Линия, окрашенная по значению в вершине (колормапа slanCM — та же таблица, что у хитмапы).
// Значение интерполируется вдоль сегмента, поэтому цвет течёт плавно, а не ступенькой по
// сегментам. Толщина — тот же разворот в quads, что у GS_3D_THICK.
static const char* VS_3D_VAL = R"(
#version 330 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in float a_val;
uniform mat4 u_mvp;
out float v_val;
void main() {
    gl_Position = u_mvp * vec4(a_pos, 1.0);
    v_val = a_val;
}
)";

static const char* GS_3D_THICK_VAL = R"(
#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;
in float v_val[];
out float g_val;
noperspective out float g_dist;
uniform vec2  u_viewport;
uniform float u_thickness_px;
)" GS_AA_HALF_GLSL R"(
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
    float h = aa_half_extent();
    vec2 off_ndc = (nrm_px * h) / (u_viewport * 0.5);
    g_val = v_val[0]; g_dist =  h; gl_Position = vec4((p0_ndc + off_ndc) * p0.w, p0.z, p0.w); EmitVertex();
    g_val = v_val[0]; g_dist = -h; gl_Position = vec4((p0_ndc - off_ndc) * p0.w, p0.z, p0.w); EmitVertex();
    g_val = v_val[1]; g_dist =  h; gl_Position = vec4((p1_ndc + off_ndc) * p1.w, p1.z, p1.w); EmitVertex();
    g_val = v_val[1]; g_dist = -h; gl_Position = vec4((p1_ndc - off_ndc) * p1.w, p1.z, p1.w); EmitVertex();
    EndPrimitive();
}
)";

// Общий хвост фрагментных шейдеров 3D-данных: обычный вывод или запись в накопители OIT.
// Вес по глубине: u_zrange — оконная глубина ближней и дальней точки bbox сцены, d в [0,1]
// вдоль неё; exp(-10 d) даёт ближним слоям перевес ~e^10 над дальними, так что внутренний
// аттрактор, закрытый плотной оболочкой, остаётся под ней, а не ложится поверх.
#define OIT_OUT_GLSL R"(
void oit_out(vec4 c) {
    if (u_oit == 0) { frag_color = c; return; }
    float a = clamp(c.a, 0.0, 0.999);
    float span = max(u_zrange.y - u_zrange.x, 1e-7);
    float d = clamp((gl_FragCoord.z - u_zrange.x) / span, 0.0, 1.0);
    float w = a * exp(-10.0 * d);
    frag_color  = vec4(c.rgb * w, w);
    frag_reveal = vec4(-log(1.0 - a), 0.0, 0.0, 0.0);
}
)"

static const char* FS_CMAP = R"(
#version 330 core
in float g_val;
uniform sampler2D u_cmap_lut;  // 256x200 RGB8, строка N-1 = карта slanCM #N
uniform int   u_colormap;
uniform float u_vmin;
uniform float u_vmax;
uniform int   u_reverse;
uniform float u_alpha;
uniform int   u_oit;
uniform vec2  u_zrange;
uniform float u_thickness_px;
noperspective in float g_dist;
layout(location = 0) out vec4 frag_color;
layout(location = 1) out vec4 frag_reveal;
)" OIT_OUT_GLSL FS_AA_COVERAGE_GLSL R"(
void main() {
    float range = u_vmax - u_vmin;
    float t = (range > 1e-30) ? clamp((g_val - u_vmin) / range, 0.0, 1.0) : 0.5;
    if (isnan(g_val) || isinf(g_val)) t = 1.0;   // разошедшаяся ошибка — верхний край шкалы
    if (u_reverse != 0) t = 1.0 - t;
    float row = float(u_colormap - 1001) + 0.5;
    oit_out(vec4(texture(u_cmap_lut, vec2(t, row / 200.0)).rgb, u_alpha * aa_coverage()));
}
)";

// Толстые одноцветные линии (GS_3D_THICK): FS со сглаженными краями.
static const char* FS_THICK = R"(
#version 330 core
uniform vec4  u_color;
uniform int   u_oit;
uniform vec2  u_zrange;
uniform float u_thickness_px;
noperspective in float g_dist;
layout(location = 0) out vec4 frag_color;
layout(location = 1) out vec4 frag_reveal;
)" OIT_OUT_GLSL FS_AA_COVERAGE_GLSL R"(
void main() {
    oit_out(vec4(u_color.rgb, u_color.a * aa_coverage()));
}
)";

// u_oit = 0 (значение по умолчанию у неустановленного uniform, так что 2D-программы его не
// трогают) — обычный вывод. u_oit = 1 — проход weighted blended OIT (begin_oit): во вложение 0
// копится цвет с весом по глубине, во вложение 1 — -log(1 - α) (смешивание ONE, ONE в обоих).
// Выход location 1 при FBO с одним вложением просто отбрасывается.
static const char* FS = R"(
#version 330 core
uniform vec4 u_color;
uniform int  u_oit;
uniform vec2 u_zrange;
layout(location = 0) out vec4 frag_color;
layout(location = 1) out vec4 frag_reveal;
)" OIT_OUT_GLSL R"(
void main() {
    oit_out(u_color);
}
)";

// Сведение OIT-накопителей поверх непрозрачного кадра: цвет — средневзвешенный по глубине,
// покрытие — 1 - prod(1 - α_i) = 1 - exp(-sum). Для серии одного цвета это ровно то же, что
// последовательное смешивание её слоёв, т. е. плотность витков копится как раньше.
static const char* VS_OIT_COMPOSITE = R"(
#version 330 core
void main() {
    vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
    gl_Position = vec4(p, 0.0, 1.0);
}
)";

static const char* FS_OIT_COMPOSITE = R"(
#version 330 core
uniform sampler2D u_accum;
uniform sampler2D u_reveal;
out vec4 frag_color;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 acc = texelFetch(u_accum, p, 0);
    if (acc.a <= 0.0) discard;
    float cov = 1.0 - exp(-texelFetch(u_reveal, p, 0).r);
    frag_color = vec4(acc.rgb / acc.a, cov);
}
)";

// Фрагментный шейдер точек: вырезает форму маркера из квадратного спрайта
// GL_POINTS. Дешевле любого CPU-пути — бифуркационная диаграмма легко даёт
// сотни тысяч точек, ImDrawList на таком объёме встаёт колом.
// Покрытие пикселя маркером со сглаженным краем (FBO без MSAA). Спрайт на u_aa_pad px шире
// маркера (VS: gl_PointSize = max(size, 1) + pad), край гасится по расстоянию до границы формы в
// пикселях. Точка меньше 1 px рисуется в 1 px с покрытием, пропорциональным размеру, — как
// тонкие линии (aa_coverage), а не округляется до целого пикселя.
static constexpr float kMarkerAaPad = 2.0f;   // px запаса спрайта: по 1 с каждой стороны

#define MARKER_COVERAGE_GLSL R"(
float marker_sdf(int m, vec2 p) {   // < 0 внутри, единица — полуразмер маркера
    const float t = 0.32;            // полутолщина штриха для Cross/Plus
    if      (m == 0) return length(p) - 1.0;                                    // Circle
    else if (m == 2) return (abs(p.x) + abs(p.y) - 1.0) * 0.70710678;           // Diamond
    else if (m == 3) return max((p.y + 2.0 * abs(p.x) - 1.0) * 0.4472136, -1.0 - p.y); // Triangle up
    else if (m == 4) return max((2.0 * abs(p.x) - 1.0 - p.y) * 0.4472136, p.y - 1.0);  // Triangle down
    else if (m == 5) {                                                          // Cross (x)
        vec2 q = vec2(p.x + p.y, p.x - p.y) * 0.70710678;
        return min(abs(q.x), abs(q.y)) - t;
    }
    else if (m == 6) return min(abs(p.x), abs(p.y)) - t;                         // Plus
    return max(abs(p.x), abs(p.y)) - 1.0;                                       // Square
}
float marker_coverage(int m) {
    float s = max(u_point_size, 1.0);
    // gl_PointCoord: (0,0) — левый ВЕРХНИЙ угол спрайта. Переводим в [-1,1] с осью Y вверх
    // (иначе «треугольник вверх» смотрел бы вниз) и снимаем запас u_aa_pad.
    vec2 p = vec2(gl_PointCoord.x * 2.0 - 1.0, 1.0 - gl_PointCoord.y * 2.0) * ((s + u_aa_pad) / s);
    float d_px = marker_sdf(m, p) * (0.5 * s);
    return clamp(0.5 - d_px, 0.0, 1.0) * min(u_point_size, 1.0);
}
)"

static const char* FS_POINT = R"(
#version 330 core
uniform vec4  u_color;
uniform int   u_marker;
uniform float u_point_size;
uniform float u_aa_pad;
out vec4 frag_color;
)" MARKER_COVERAGE_GLSL R"(
void main() {
    float cov = marker_coverage(u_marker);
    if (cov <= 0.0) discard;
    frag_color = vec4(u_color.rgb, u_color.a * cov);
}
)";

// 3D-точки с формой маркера (draw_points_3d, marker >= 0): та же маска, вывод — с OIT.
static const char* FS_POINT_3D = R"(
#version 330 core
uniform vec4  u_color;
uniform int   u_marker;
uniform float u_point_size;
uniform float u_aa_pad;
uniform int   u_oit;
uniform vec2  u_zrange;
layout(location = 0) out vec4 frag_color;
layout(location = 1) out vec4 frag_reveal;
)" OIT_OUT_GLSL MARKER_COVERAGE_GLSL R"(
void main() {
    float cov = marker_coverage(u_marker);
    if (cov <= 0.0) discard;
    oit_out(vec4(u_color.rgb, u_color.a * cov));
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

// Все colormap'ы сэмплятся из одной таблицы (u_cmap_lut) — полиномиальных
// приближений viridis/inferno/turbo здесь больше нет: те же карты есть в
// slanCM честными 256-цветными таблицами, а fit'ы от них заметно отставали
// (turbo — до 32/255 в тёмном конце). Спец-значения (≥1e30, NaN, Inf)
// рисуются тёмным фоном — engine помечает diverged ячейки этим маркером.
static const char* FS_HEATMAP = R"(
#version 330 core
in vec2 v_uv;
uniform sampler2D u_tex;
uniform sampler2D u_cmap_lut;  // 256x200 RGB8, строка N-1 = карта slanCM #N
uniform float u_vmin;
uniform float u_vmax;
uniform int   u_colormap;
uniform int   u_discrete_n;  // 0 = continuous, N>0 = quantize t into N bands
uniform int   u_reverse;     // 1 = t := 1-t before colormap sampling
uniform vec3  u_nodata;      // цвет спец-значений (>=1e30/NaN/Inf), см. draw_heatmap
uniform int   u_oor;         // 1 = значения вне [vmin, vmax] красятся u_below/u_above, а не краем палитры
uniform vec3  u_below;
uniform vec3  u_above;
uniform vec2  u_uv_off;
uniform vec2  u_uv_scale;
uniform int   u_smooth;      // 1 = сглаживание: среднее ЦВЕТОВ по сетке точек внутри пикселя
out vec4 frag_color;

// Цвет данных в точке uv (координаты текстуры данных).
vec3 shade(vec2 uv) {
    // Если view вышел за пределы данных — рисуем фон, чтобы пользователю
    // были видны границы реального датасета.
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
        return vec3(0.08, 0.08, 0.10);
    float v = textureLod(u_tex, uv, 0.0).r;
    // <= -1e30 — ячейку не досчитал Stop (HeatmapView::cell_not_computed): сплошной средне-серый,
    // заметно светлее тёмного «нет данных» (u_nodata) и фона, чтобы не путать с расходимостью.
    if (v <= -1.0e30 && !isinf(v)) return vec3(0.40, 0.40, 0.42);
    if (v >= 1.0e30 || isnan(v) || isinf(v)) return u_nodata;
    float range = u_vmax - u_vmin;
    if (u_oor != 0 && range > 1e-30) {
        if (v < u_vmin) return u_below;
        if (v > u_vmax) return u_above;
    }
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
    if (u_reverse != 0) t = 1.0 - t;
    // slanCM: id = 1000 + N, строка текстуры = N-1. Сэмплим ровно в центре
    // строки, чтобы vertical-bilinear не смешивал соседние colormap'ы между
    // собой. Невалидный id сюда не доходит — draw_heatmap прогоняет его через
    // colormap_id_or().
    float row = float(u_colormap - 1001) + 0.5;
    vec3 col = textureLod(u_cmap_lut, vec2(t, row / 200.0), 0.0).rgb;
    return clamp(col, 0.0, 1.0);
}

void main() {
    vec2 uv = v_uv * u_uv_scale + u_uv_off;
    // Производные — до любых ветвлений (в неоднородном потоке управления они не определены).
    vec2 du = dFdx(uv), dv = dFdy(uv);
    if (u_smooth == 0) {
        frag_color = vec4(shade(uv), 1.0);
        return;
    }
    // Сглаживание: n x n точек по площади пикселя, каждая -> цвет, усредняются ЦВЕТА. Значения не
    // усредняются: между периодом 2 и 4 (или двумя бассейнами) нет никакого «3», а смесь цветов —
    // честное «пиксель накрывает обе ячейки». Внутри крупной ячейки все точки одного цвета, и она
    // остаётся резкой; сглаживаются края. Сетка гуще, когда в пиксель попадает много ячеек.
    vec2 cells = abs(du) * vec2(textureSize(u_tex, 0)) + abs(dv) * vec2(textureSize(u_tex, 0));
    int n = int(clamp(ceil(max(cells.x, cells.y) * 2.0), 2.0, 8.0));
    vec3 acc = vec3(0.0);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            vec2 f = (vec2(float(i), float(j)) + 0.5) / float(n) - 0.5;
            acc += shade(uv + f.x * du + f.y * dv);
        }
    frag_color = vec4(acc / float(n * n), 1.0);
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
    ensure_lut_texture();
    glGenVertexArrays(1, &vao_);
}

PlotRenderer::~PlotRenderer() {
    destroy_fbo();
    if (program_2d_)       glDeleteProgram(program_2d_);
    if (program_points_)   glDeleteProgram(program_points_);
    if (program_3d_)       glDeleteProgram(program_3d_);
    if (program_3d_thick_) glDeleteProgram(program_3d_thick_);
    if (program_2d_thick_) glDeleteProgram(program_2d_thick_);
    if (program_3d_cmap_)  glDeleteProgram(program_3d_cmap_);
    if (program_heatmap_)  glDeleteProgram(program_heatmap_);
    if (program_oit_composite_) glDeleteProgram(program_oit_composite_);
    if (program_points_3d_) glDeleteProgram(program_points_3d_);
    if (heatmap_vbo_)     glDeleteBuffers(1, &heatmap_vbo_);
    if (lut_tex_)         glDeleteTextures(1, &lut_tex_);
    if (vao_)             glDeleteVertexArrays(1, &vao_);
}

void PlotRenderer::ensure_lut_texture() {
    if (lut_tex_) return;
    // 256 (t) x 200 (карты slanCM) RGB8: строка N-1 = карта #N. kSlanCmLut уже
    // лежит ровно в этом layout'е (row-major, 3 байта на пиксель, шаг строки
    // 768 байт кратен дефолтному GL_UNPACK_ALIGNMENT=4), поэтому заливаем
    // прямо из него — без промежуточного буфера на 150 КБ.
    static_assert(sizeof(kSlanCmLut) == (size_t)kSlanCmCount * 256 * 3,
                  "kSlanCmLut layout must match the 256xN RGB8 upload below");

    glGenTextures(1, &lut_tex_);
    glBindTexture(GL_TEXTURE_2D, lut_tex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 256, kSlanCmCount, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, kSlanCmLut);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
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
    // Отдельная программа для точек с формой маркера: тот же VS_2D (gl_PointSize),
    // другой FS (маска по gl_PointCoord). Без неё draw_points остаётся на
    // program_2d_ и рисует прежний сплошной квадрат.
    GLuint fs_pt = compile(GL_FRAGMENT_SHADER, FS_POINT);
    if (fs_pt && vs2) {
        program_points_ = link_program(vs2, fs_pt);
        if (program_points_) {
            loc_mvp_points_        = glGetUniformLocation(program_points_, "u_mvp");
            loc_color_points_      = glGetUniformLocation(program_points_, "u_color");
            loc_point_size_points_ = glGetUniformLocation(program_points_, "u_point_size");
            loc_marker_points_     = glGetUniformLocation(program_points_, "u_marker");
            loc_aa_pad_points_     = glGetUniformLocation(program_points_, "u_aa_pad");
        }
    }
    if (fs && vs3) {
        program_3d_ = link_program(vs3, fs);
        if (program_3d_) {
            loc_mvp_3d_ = glGetUniformLocation(program_3d_, "u_mvp");
            loc_color_3d_ = glGetUniformLocation(program_3d_, "u_color");
            loc_point_size_3d_ = glGetUniformLocation(program_3d_, "u_point_size");
            loc_oit_3d_    = glGetUniformLocation(program_3d_, "u_oit");
            loc_zrange_3d_ = glGetUniformLocation(program_3d_, "u_zrange");
        }
    }
    GLuint fs_pt3 = compile(GL_FRAGMENT_SHADER, FS_POINT_3D);
    if (fs_pt3 && vs3) {
        program_points_3d_ = link_program(vs3, fs_pt3);
        if (program_points_3d_) {
            loc_mvp_points_3d_    = glGetUniformLocation(program_points_3d_, "u_mvp");
            loc_color_points_3d_  = glGetUniformLocation(program_points_3d_, "u_color");
            loc_size_points_3d_   = glGetUniformLocation(program_points_3d_, "u_point_size");
            loc_marker_points_3d_ = glGetUniformLocation(program_points_3d_, "u_marker");
            loc_oit_points_3d_    = glGetUniformLocation(program_points_3d_, "u_oit");
            loc_zrange_points_3d_ = glGetUniformLocation(program_points_3d_, "u_zrange");
            loc_aa_pad_points_3d_ = glGetUniformLocation(program_points_3d_, "u_aa_pad");
        }
    }
    if (fs_pt3) glDeleteShader(fs_pt3);
    GLuint fs_thick = compile(GL_FRAGMENT_SHADER, FS_THICK);
    if (fs_thick && vs3 && gs3t) {
        program_3d_thick_ = link_program_3(vs3, gs3t, fs_thick);
        if (program_3d_thick_) {
            loc_mvp_3d_thick_       = glGetUniformLocation(program_3d_thick_, "u_mvp");
            loc_color_3d_thick_     = glGetUniformLocation(program_3d_thick_, "u_color");
            loc_viewport_3d_thick_  = glGetUniformLocation(program_3d_thick_, "u_viewport");
            loc_thickness_3d_thick_ = glGetUniformLocation(program_3d_thick_, "u_thickness_px");
            loc_oit_3d_thick_       = glGetUniformLocation(program_3d_thick_, "u_oit");
            loc_zrange_3d_thick_    = glGetUniformLocation(program_3d_thick_, "u_zrange");
        }
    }
    // 2D-вариант толстой линии: тот же GS (z = 0, w = 1 у 2D-вершин), вход — VS_2D.
    if (fs_thick && vs2 && gs3t) {
        program_2d_thick_ = link_program_3(vs2, gs3t, fs_thick);
        if (program_2d_thick_) {
            loc_mvp_2d_thick_       = glGetUniformLocation(program_2d_thick_, "u_mvp");
            loc_color_2d_thick_     = glGetUniformLocation(program_2d_thick_, "u_color");
            loc_viewport_2d_thick_  = glGetUniformLocation(program_2d_thick_, "u_viewport");
            loc_thickness_2d_thick_ = glGetUniformLocation(program_2d_thick_, "u_thickness_px");
        }
    }
    {
        GLuint vs_v = compile(GL_VERTEX_SHADER, VS_3D_VAL);
        GLuint gs_v = compile(GL_GEOMETRY_SHADER, GS_3D_THICK_VAL);
        GLuint fs_c = compile(GL_FRAGMENT_SHADER, FS_CMAP);
        if (vs_v && gs_v && fs_c) {
            program_3d_cmap_ = link_program_3(vs_v, gs_v, fs_c);
            if (program_3d_cmap_) {
                loc_mvp_3d_cmap_       = glGetUniformLocation(program_3d_cmap_, "u_mvp");
                loc_viewport_3d_cmap_  = glGetUniformLocation(program_3d_cmap_, "u_viewport");
                loc_thickness_3d_cmap_ = glGetUniformLocation(program_3d_cmap_, "u_thickness_px");
                loc_lut_3d_cmap_       = glGetUniformLocation(program_3d_cmap_, "u_cmap_lut");
                loc_cmap_3d_cmap_      = glGetUniformLocation(program_3d_cmap_, "u_colormap");
                loc_vmin_3d_cmap_      = glGetUniformLocation(program_3d_cmap_, "u_vmin");
                loc_vmax_3d_cmap_      = glGetUniformLocation(program_3d_cmap_, "u_vmax");
                loc_reverse_3d_cmap_   = glGetUniformLocation(program_3d_cmap_, "u_reverse");
                loc_alpha_3d_cmap_     = glGetUniformLocation(program_3d_cmap_, "u_alpha");
                loc_oit_3d_cmap_       = glGetUniformLocation(program_3d_cmap_, "u_oit");
                loc_zrange_3d_cmap_    = glGetUniformLocation(program_3d_cmap_, "u_zrange");
            }
        }
        if (vs_v) glDeleteShader(vs_v);
        if (gs_v) glDeleteShader(gs_v);
        if (fs_c) glDeleteShader(fs_c);
    }
    {
        GLuint vs_o = compile(GL_VERTEX_SHADER, VS_OIT_COMPOSITE);
        GLuint fs_o = compile(GL_FRAGMENT_SHADER, FS_OIT_COMPOSITE);
        if (vs_o && fs_o) {
            program_oit_composite_ = link_program(vs_o, fs_o);
            if (program_oit_composite_) {
                loc_oit_accum_  = glGetUniformLocation(program_oit_composite_, "u_accum");
                loc_oit_reveal_ = glGetUniformLocation(program_oit_composite_, "u_reveal");
            }
        }
        if (vs_o) glDeleteShader(vs_o);
        if (fs_o) glDeleteShader(fs_o);
    }
    GLuint fs_h = compile(GL_FRAGMENT_SHADER, FS_HEATMAP);
    GLuint vs_h = compile(GL_VERTEX_SHADER, VS_HEATMAP);
    if (fs_h && vs_h) {
        program_heatmap_ = link_program(vs_h, fs_h);
        if (program_heatmap_) {
            loc_heatmap_tex_      = glGetUniformLocation(program_heatmap_, "u_tex");
            loc_heatmap_lut_      = glGetUniformLocation(program_heatmap_, "u_cmap_lut");
            loc_heatmap_vmin_     = glGetUniformLocation(program_heatmap_, "u_vmin");
            loc_heatmap_vmax_     = glGetUniformLocation(program_heatmap_, "u_vmax");
            loc_heatmap_cmap_     = glGetUniformLocation(program_heatmap_, "u_colormap");
            loc_heatmap_uv_off_   = glGetUniformLocation(program_heatmap_, "u_uv_off");
            loc_heatmap_uv_scale_ = glGetUniformLocation(program_heatmap_, "u_uv_scale");
            loc_heatmap_discrete_n_ = glGetUniformLocation(program_heatmap_, "u_discrete_n");
            loc_heatmap_reverse_   = glGetUniformLocation(program_heatmap_, "u_reverse");
            loc_heatmap_nodata_    = glGetUniformLocation(program_heatmap_, "u_nodata");
            loc_heatmap_oor_       = glGetUniformLocation(program_heatmap_, "u_oor");
            loc_heatmap_below_     = glGetUniformLocation(program_heatmap_, "u_below");
            loc_heatmap_above_     = glGetUniformLocation(program_heatmap_, "u_above");
            loc_heatmap_smooth_    = glGetUniformLocation(program_heatmap_, "u_smooth");
        }
    }
    if (vs2)  glDeleteShader(vs2);
    if (vs3)  glDeleteShader(vs3);
    if (gs3t) glDeleteShader(gs3t);
    if (vs_h) glDeleteShader(vs_h);
    if (fs_h) glDeleteShader(fs_h);
    if (fs_pt) glDeleteShader(fs_pt);
    if (fs_thick) glDeleteShader(fs_thick);
    if (fs)   glDeleteShader(fs);
}

void PlotRenderer::destroy_oit() {
    if (oit_accum_tex_)  { glDeleteTextures(1, &oit_accum_tex_);  oit_accum_tex_ = 0; }
    if (oit_reveal_tex_) { glDeleteTextures(1, &oit_reveal_tex_); oit_reveal_tex_ = 0; }
    if (oit_fbo_)        { glDeleteFramebuffers(1, &oit_fbo_);    oit_fbo_ = 0; }
}

// Накопители OIT — того же размера, что основной FBO, и на ТОМ ЖЕ depth-буфере: непрозрачное
// (оси, серии с α = 1) уже записало туда глубину и закрывает полупрозрачное за собой.
bool PlotRenderer::ensure_oit() {
    if (oit_fbo_) return true;
    if (!fbo_ || !depth_rbo_ || !program_oit_composite_) return false;
    auto make_tex = [&](GLuint& tex, GLint ifmt, GLenum fmt) {
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, ifmt, fbo_w_, fbo_h_, 0, fmt, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    };
    // 32F, а не 16F: сотни наложенных витков с весом до e^10 переполняли бы half.
    make_tex(oit_accum_tex_, GL_RGBA32F, GL_RGBA);
    make_tex(oit_reveal_tex_, GL_R32F, GL_RED);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenFramebuffers(1, &oit_fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, oit_fbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, oit_accum_tex_, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, oit_reveal_tex_, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rbo_);
    const GLenum bufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers(2, bufs);
    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "[PlotRenderer] OIT FBO incomplete: 0x%x\n", status);
        destroy_oit();
        return false;
    }
    return true;
}

bool PlotRenderer::begin_oit(float z_near, float z_far) {
    if (oit_active_ || !fbo_has_depth_ || !ensure_oit()) return false;
    glBindFramebuffer(GL_FRAMEBUFFER, oit_fbo_);
    const float zero[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    glClearBufferfv(GL_COLOR, 0, zero);
    glClearBufferfv(GL_COLOR, 1, zero);
    oit_was_blend_ = glIsEnabled(GL_BLEND);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glDepthMask(GL_FALSE);
    oit_zrange_[0] = z_near;
    oit_zrange_[1] = z_far;
    oit_active_ = true;
    return true;
}

void PlotRenderer::end_oit() {
    if (!oit_active_) return;
    oit_active_ = false;
    glDepthMask(GL_TRUE);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    // Альфа текстуры плота остаётся 1 (как у draw_line_3d): меняется только цвет.
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
    const GLboolean was_depth = glIsEnabled(GL_DEPTH_TEST);
    glDisable(GL_DEPTH_TEST);
    glUseProgram(program_oit_composite_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, oit_accum_tex_);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, oit_reveal_tex_);
    if (loc_oit_accum_ >= 0)  glUniform1i(loc_oit_accum_, 0);
    if (loc_oit_reveal_ >= 0) glUniform1i(loc_oit_reveal_, 1);
    glBindVertexArray(vao_);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
    if (was_depth) glEnable(GL_DEPTH_TEST);
    if (!oit_was_blend_) glDisable(GL_BLEND);
}

void PlotRenderer::destroy_fbo() {
    destroy_oit();
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
    content_key_ = 0;   // текстура перерисовывается — прежний ключ содержимого недействителен

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

void PlotRenderer::draw_line_thick_2d(GLuint vbo, int point_count, const float mvp[16],
    const float color[4], float line_width) {
    if (!program_2d_thick_) { draw_line(vbo, point_count, mvp, color, line_width); return; }
    if (point_count < 2 || !vbo) return;
    const GLboolean was_blend = glIsEnabled(GL_BLEND);
    glEnable(GL_BLEND);
    // Альфа FBO держим на 1, как у draw_points (иначе полупрозрачная линия «продырявит» текстуру).
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(program_2d_thick_);
    glUniformMatrix4fv(loc_mvp_2d_thick_, 1, GL_FALSE, mvp);
    glUniform4fv(loc_color_2d_thick_, 1, color);
    if (loc_viewport_2d_thick_ >= 0)  glUniform2f(loc_viewport_2d_thick_, (float)fbo_w_, (float)fbo_h_);
    if (loc_thickness_2d_thick_ >= 0) glUniform1f(loc_thickness_2d_thick_, line_width);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glDrawArrays(GL_LINE_STRIP, 0, point_count);
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);
    if (!was_blend) glDisable(GL_BLEND);
}

void PlotRenderer::draw_points(GLuint vbo, int point_count, const float mvp[16],
    const float color[4], float point_size, int marker) {
    // Fallback на старый путь, если shaped-программа не собралась.
    const bool shaped = (marker >= 0 && program_points_ != 0);
    const GLuint prog = shaped ? program_points_ : program_2d_;
    if (!prog || point_count < 1 || !vbo) return;
    // Alpha-блендинг включаем ТОЛЬКО на shaped-пути, чтобы выключенный
    // Custom point style не менял GL-состояние (как в draw_line_3d).
    GLboolean was_blend = GL_FALSE;
    if (shaped) {
        was_blend = glIsEnabled(GL_BLEND);
        glEnable(GL_BLEND);
        // RGB — обычный src-over. Альфа-канал FBO держим на 1 (src factor ONE
        // при dst = 1 даёт 1): иначе полупрозрачная точка «продырявливала» бы
        // текстуру плота, и сквозь неё просвечивал бы фон окна ImGui поверх
        // фона самого плота — α выглядела бы вдвое сильнее заданной.
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                            GL_ONE,       GL_ONE_MINUS_SRC_ALPHA);
    }
    glUseProgram(prog);
    if (shaped) {
        glUniformMatrix4fv(loc_mvp_points_, 1, GL_FALSE, mvp);
        glUniform4fv(loc_color_points_, 1, color);
        if (loc_point_size_points_ >= 0) glUniform1f(loc_point_size_points_, point_size);
        if (loc_marker_points_ >= 0)     glUniform1i(loc_marker_points_, marker);
        if (loc_aa_pad_points_ >= 0)     glUniform1f(loc_aa_pad_points_, kMarkerAaPad);
    } else {
        glUniformMatrix4fv(loc_mvp_2d_, 1, GL_FALSE, mvp);
        glUniform4fv(loc_color_2d_, 1, color);
        if (loc_point_size_2d_ >= 0) glUniform1f(loc_point_size_2d_, point_size);
    }
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
    // Точный blend func не восстанавливаем — ImGui сам ставит свой перед
    // отрисовкой ImDrawList (см. draw_line_3d).
    if (shaped && !was_blend) glDisable(GL_BLEND);
}

void PlotRenderer::draw_heatmap(GLuint tex, float vmin, float vmax, int colormap_id,
                                float uv_off_x, float uv_off_y,
                                float uv_scale_x, float uv_scale_y,
                                int n_discrete, bool reverse,
                                const float* nodata_rgb,
                                const float* below_rgb,
                                const float* above_rgb,
                                bool smooth) {
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
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, lut_tex_);
    if (loc_heatmap_lut_      >= 0) glUniform1i(loc_heatmap_lut_, 1);
    glActiveTexture(GL_TEXTURE0);
    if (loc_heatmap_vmin_     >= 0) glUniform1f(loc_heatmap_vmin_, vmin);
    if (loc_heatmap_vmax_     >= 0) glUniform1f(loc_heatmap_vmax_, vmax);
    // Единственная точка, где id уходит в шейдер, — валидируем здесь, чтобы
    // легаси 4..8 и мусор из старых сессий были безопасны для любого вызова.
    if (loc_heatmap_cmap_     >= 0) glUniform1i(loc_heatmap_cmap_,
                                                colormap_id_or(colormap_id, 0));
    if (loc_heatmap_uv_off_   >= 0) glUniform2f(loc_heatmap_uv_off_, uv_off_x, uv_off_y);
    if (loc_heatmap_uv_scale_ >= 0) glUniform2f(loc_heatmap_uv_scale_, uv_scale_x, uv_scale_y);
    if (loc_heatmap_discrete_n_ >= 0) glUniform1i(loc_heatmap_discrete_n_, n_discrete);
    if (loc_heatmap_reverse_  >= 0) glUniform1i(loc_heatmap_reverse_, reverse ? 1 : 0);
    if (loc_heatmap_nodata_   >= 0) {
        static const float kDefaultNoData[3] = { 0.12f, 0.12f, 0.14f };
        const float* nd = nodata_rgb ? nodata_rgb : kDefaultNoData;
        glUniform3f(loc_heatmap_nodata_, nd[0], nd[1], nd[2]);
    }
    // Uniform'ы программы живут между вызовами, поэтому флаг ставится КАЖДЫЙ раз:
    // иначе режим, включённый одной картой, протёк бы на все следующие.
    const bool oor = below_rgb && above_rgb;
    if (loc_heatmap_oor_ >= 0) glUniform1i(loc_heatmap_oor_, oor ? 1 : 0);
    if (oor) {
        if (loc_heatmap_below_ >= 0) glUniform3f(loc_heatmap_below_, below_rgb[0], below_rgb[1], below_rgb[2]);
        if (loc_heatmap_above_ >= 0) glUniform3f(loc_heatmap_above_, above_rgb[0], above_rgb[1], above_rgb[2]);
    }
    if (loc_heatmap_smooth_ >= 0) glUniform1i(loc_heatmap_smooth_, smooth ? 1 : 0);

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
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
}

void PlotRenderer::draw_points_3d(GLuint vbo, int point_count, const float mvp[16],
    const float color[4], float point_size, int marker) {
    if (point_count < 1 || !vbo || !program_3d_) return;

    // Blending is on for the same reason as in draw_points: a dense map attractor
    // leans on alpha, and without it the markers would be opaque squares.
    // В OIT-проходе (begin_oit) смешивание и маску глубины уже выставил он.
    const bool oit = oit_active_;
    GLboolean was_blend = glIsEnabled(GL_BLEND);
    const bool translucent = !oit && color[3] < 1.0f;
    if (!oit) {
        glEnable(GL_BLEND);
        // Альфа FBO — на 1, полупрозрачные точки не пишут глубину: см. draw_line_3d.
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        if (translucent) glDepthMask(GL_FALSE);
    }

    // marker >= 0 — шейдерная маска формы (как draw_points в 2D); < 0 — прежний квадрат.
    if (marker >= 0 && program_points_3d_) {
        glUseProgram(program_points_3d_);
        set_oit_uniforms(loc_oit_points_3d_, loc_zrange_points_3d_);
        glUniformMatrix4fv(loc_mvp_points_3d_, 1, GL_FALSE, mvp);
        glUniform4fv(loc_color_points_3d_, 1, color);
        if (loc_size_points_3d_ >= 0)   glUniform1f(loc_size_points_3d_, point_size);
        if (loc_marker_points_3d_ >= 0) glUniform1i(loc_marker_points_3d_, marker);
        if (loc_aa_pad_points_3d_ >= 0) glUniform1f(loc_aa_pad_points_3d_, kMarkerAaPad);
    } else {
        glUseProgram(program_3d_);
        set_oit_uniforms(loc_oit_3d_, loc_zrange_3d_);
        glUniformMatrix4fv(loc_mvp_3d_, 1, GL_FALSE, mvp);
        glUniform4fv(loc_color_3d_, 1, color);
        if (loc_point_size_3d_ >= 0) glUniform1f(loc_point_size_3d_, point_size);
    }
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    // Core profile takes the size from gl_PointSize only under this flag.
    glEnable(GL_PROGRAM_POINT_SIZE);
    glDrawArrays(GL_POINTS, 0, point_count);
    glDisable(GL_PROGRAM_POINT_SIZE);
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);
    if (translucent) glDepthMask(GL_TRUE);
    if (!oit && !was_blend) glDisable(GL_BLEND);
}

void PlotRenderer::set_oit_uniforms(GLint loc_oit, GLint loc_zrange) const {
    // Uniform живёт в программе между вызовами, поэтому 0 ставим явно на каждом рисовании.
    if (loc_oit >= 0) glUniform1i(loc_oit, oit_active_ ? 1 : 0);
    if (oit_active_ && loc_zrange >= 0) glUniform2f(loc_zrange, oit_zrange_[0], oit_zrange_[1]);
}

void PlotRenderer::draw_line_3d(GLuint vbo, int point_count, const float mvp[16],
    const float color[4], float line_width, bool thick_style) {
    if (point_count < 2 || !vbo) return;

    // Fallback на старый путь если thick program не собрался (нет поддержки GS).
    const bool use_thick = thick_style && program_3d_thick_ != 0;

    if (!use_thick) {
        // СТАРЫЙ путь: побайтово как до Custom line style patch
        if (!program_3d_) return;
        glUseProgram(program_3d_);
        set_oit_uniforms(loc_oit_3d_, loc_zrange_3d_);
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

    // ТОЛСТЫЙ путь: geometry shader раскрывает сегменты в quads
    // Alpha blending включаем только тут, чтобы OFF-путь не менял GL state.
    // Альфа FBO держим на 1 (как draw_points): с glBlendFunc(SRC_ALPHA, ...) на канал альфы
    // полупрозрачная линия «продырявливала» текстуру плота, и вся траектория выходила ровным
    // бледным тоном. Полупрозрачная линия не пишет глубину: иначе тест глубины отбрасывал
    // витки за уже нарисованными, и наложения (градиент плотности) не копились. Порядок между
    // полупрозрачными сериями при этом теряется — его держит OIT-проход (begin_oit), в котором
    // смешивание и маску глубины выставил он сам.
    const bool oit = oit_active_;
    GLboolean was_blend = glIsEnabled(GL_BLEND);
    const bool translucent = !oit && color[3] < 1.0f;
    if (!oit) {
        glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        if (translucent) glDepthMask(GL_FALSE);
    }

    glUseProgram(program_3d_thick_);
    set_oit_uniforms(loc_oit_3d_thick_, loc_zrange_3d_thick_);
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
    if (translucent) glDepthMask(GL_TRUE);
    if (!oit && !was_blend) glDisable(GL_BLEND);
}

void PlotRenderer::draw_line_3d_cmap(GLuint vbo, GLuint vbo_val, int point_count,
    const float mvp[16], int colormap_id, float vmin, float vmax, bool reverse,
    float alpha, float line_width) {
    if (point_count < 2 || !vbo || !vbo_val || !program_3d_cmap_) return;

    // Смешивание и глубина — как у толстого пути draw_line_3d: альфа FBO на 1, полупрозрачная
    // линия глубину не пишет (иначе наложения витков не копятся).
    const bool oit = oit_active_;
    GLboolean was_blend = glIsEnabled(GL_BLEND);
    const bool translucent = !oit && alpha < 1.0f;
    if (!oit) {
        glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        if (translucent) glDepthMask(GL_FALSE);
    }

    glUseProgram(program_3d_cmap_);
    set_oit_uniforms(loc_oit_3d_cmap_, loc_zrange_3d_cmap_);
    glUniformMatrix4fv(loc_mvp_3d_cmap_, 1, GL_FALSE, mvp);
    if (loc_viewport_3d_cmap_ >= 0)  glUniform2f(loc_viewport_3d_cmap_, (float)fbo_w_, (float)fbo_h_);
    if (loc_thickness_3d_cmap_ >= 0) glUniform1f(loc_thickness_3d_cmap_, line_width);
    if (loc_cmap_3d_cmap_ >= 0)      glUniform1i(loc_cmap_3d_cmap_, colormap_id_or(colormap_id, 0));
    if (loc_vmin_3d_cmap_ >= 0)      glUniform1f(loc_vmin_3d_cmap_, vmin);
    if (loc_vmax_3d_cmap_ >= 0)      glUniform1f(loc_vmax_3d_cmap_, vmax);
    if (loc_reverse_3d_cmap_ >= 0)   glUniform1i(loc_reverse_3d_cmap_, reverse ? 1 : 0);
    if (loc_alpha_3d_cmap_ >= 0)     glUniform1f(loc_alpha_3d_cmap_, alpha);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, lut_tex_);
    if (loc_lut_3d_cmap_ >= 0)       glUniform1i(loc_lut_3d_cmap_, 0);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_val);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glDrawArrays(GL_LINE_STRIP, 0, point_count);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);

    if (translucent) glDepthMask(GL_TRUE);
    if (!oit && !was_blend) glDisable(GL_BLEND);
}

void PlotRenderer::end_frame() {
    end_oit();   // забытый end_oit не оставит кадр без сведения и FBO накопителей привязанным
    glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)saved_fbo_);
    glViewport(saved_viewport_[0], saved_viewport_[1],
        saved_viewport_[2], saved_viewport_[3]);
    if (saved_depth_test_) glEnable(GL_DEPTH_TEST); else glDisable(GL_DEPTH_TEST);
}
