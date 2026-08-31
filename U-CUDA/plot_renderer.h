#pragma once
#include <glad/glad.h>
#include <cstdint>
#include <string>
#include <vector>
#include "imgui.h"   // ImU32 для cmap_sample

// 2D scalar → RGB colormap. Раньше жил в heatmap_view.h; перемещён в
// plot_renderer.h, потому что:
//   1) `PlotRenderer::draw_heatmap` уже его принимает (int colormap_id);
//   2) `plot_view_2d.cpp` теперь использует то же сэмплирование для
//      per-segment colored trajectory.
//
// Id колормапа — int, и ЖИВЫХ значений ровно одно семейство: 1001..1200,
// карты slanCM (см. kSlanCmIdBase ниже). Тип остался enum'ом, потому что им
// типизированы HeatmapView::colormap и PlotSeriesInput::colormap, а сами
// перечисленные ниже константы — это таблица миграции, не выбор.
// Всё, что приходит из конфигов и сессий, прогоняется через colormap_id_or().
enum class HeatmapColormap : int {
    // 0-8 — ЛЕГАСИ, только для чтения старых конфигов и сессий; новые значения
    // сюда не пишутся. 0-3 были полиномиальными приближениями matplotlib-
    // таблиц прямо в шейдере, 4-8 — пятью LUT-картами в отдельной текстуре
    // 256x5. Все девять есть в slanCM под своими именами, поэтому дублировать
    // их в пикере незачем: colormap_id_or() переводит каждое из этих значений
    // в соответствующий slanCM-id.
    //
    // Для 4-8 замена побитовая (проверено при генерации colormap_slancm_data.h).
    // Для 0-3 — с точностью до качества старого полиномиального fit'а:
    // gray совпадает точно, viridis расходится максимум на 4/255, inferno на
    // 9/255, turbo на 32/255 в тёмном конце. Это не потеря, а исправление —
    // таблица slanCM и есть эталон, который полиномы приближали.
    LegacyViridis      = 0,
    LegacyInferno      = 1,
    LegacyTurbo        = 2,
    LegacyGray         = 3,
    LegacyGistStern    = 4,
    LegacyGnuPlot      = 5,
    LegacyGistRainbow  = 6,
    LegacyNipySpectral = 7,
    LegacyGistNcar     = 8,
};

// ---------------------------------------------------------------------------
// slanCM — 200 колормап (Zhaoxu Liu, MATLAB File Exchange #120088). Таблица
// цветов лежит в colormap_slancm_data.h и включается только в
// plot_renderer.cpp; наружу выходят имена, категории и сэмплирование.
// ---------------------------------------------------------------------------
constexpr int kSlanCmCount = 200;
// Id карты slanCM = kSlanCmIdBase + N, где N — её РОДНОЙ номер 1..200 из
// библиотеки (именно N видит пользователь в пикере и в Settings). Смещение
// нужно, чтобы не пересечься с уже сохранёнными id 0..8 в _app_config.json,
// сессиях и per-config colormap_idx.
constexpr int kSlanCmIdBase = 1000;          // валидные id: 1001..1200
constexpr int kSlanCmCategoryCount = 9;

// Дефолт для полей типа HeatmapColormap. #1 viridis — ровно та карта, что
// была дефолтом и раньше (когда встроенный Viridis считался полиномом), так
// что новые HeatmapView / PlotSeriesInput стартуют с прежним видом.
constexpr HeatmapColormap kDefaultColormap = (HeatmapColormap)(kSlanCmIdBase + 1);

constexpr bool is_slancm_id(int id) {
    return id > kSlanCmIdBase && id <= kSlanCmIdBase + kSlanCmCount;
}
// N (1..200) <-> id. slancm_number валиден только если is_slancm_id(id).
constexpr int  slancm_number(int id) { return id - kSlanCmIdBase; }
constexpr int  slancm_id(int n)      { return kSlanCmIdBase + n; }

// n = 1..200. Вне диапазона — "" / -1, чтобы вызывающему не надо было
// дублировать проверку.
const char* slancm_name(int n);
int         slancm_category(int n);            // индекс в slancm_category_name
const char* slancm_category_name(int cat);

// Подпись для UI: "151  gist_stern". Указатель живёт до конца работы
// приложения (строки строятся один раз).
const char* colormap_label(int id);

// Единственная точка валидации id, пришедшего снаружи (конфиг, сессия,
// per-diagram colormap_idx). Возвращает: сам id, если он валиден; slanCM-
// эквивалент, если это легаси 0..8; иначе — fallback (тоже провалидированный,
// а если и он мусор — #1 viridis). Отрицательный id означает «не задано» и
// уходит в fallback, что сохраняет прежнюю семантику `cfg >= 0 ? cfg : def`.
int colormap_id_or(int id, int fallback);

// ---- Набор карт, включённых пользователем (чекбоксы в Settings) ----
// Маска из kSlanCmCount символов '0'/'1', символ i — карта #(i+1). Хранится
// в _app_config.json; set_enabled_slancm зовётся на bootstrap'е и на каждое
// изменение в Settings — ровно как set_tick_precision() у осей.
void set_enabled_slancm(const std::string& mask);
// Дефолт: девять карт, которые были доступны до появления Settings, — четыре
// бывшие встроенные (#1 viridis, #3 inferno, #168 turbo, #28 gray) и пять
// бывших LUT-карт (#151 gist_stern, #152 gnuplot, #162 gist_rainbow,
// #170 nipy_spectral, #171 gist_ncar). Первый запуск выглядит как раньше.
std::string default_enabled_slancm();

// Готовый список id для пикера: включённые карты slanCM в порядке их номеров.
// Может быть пустым (пользователь снял все галочки) — пикер это переживает,
// показывая текущую карту отдельной строкой. Пересобирается только в
// set_enabled_slancm.
const std::vector<int>& picker_colormap_ids();

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

// Общий список имён для ImGui::Combo — единственный источник истины (раньше
// было по копии в каждом combo по коду).
extern const char* const kPointMarkerNames[7];
constexpr int kPointMarkerCount = 7;

// CPU-side колормап: линейная интерполяция между соседними элементами
// 256-элементной таблицы slanCM. Визуально идентично GPU-пути (там та же
// таблица сэмплится билинейно), но не гарантированно bit-exact. Легаси-id
// 0-8 сначала мигрируют через colormap_id_or. t clamp'ится в [0,1].
// Возвращает ImU32 (ImDrawList).
ImU32 cmap_sample(float t, HeatmapColormap m);
// Тот же сэмплер для int-id (в UI id таскаются как int, а не как enum).
ImU32 cmap_sample_id(float t, int colormap_id);

class PlotRenderer {
public:
    PlotRenderer();
    ~PlotRenderer();
    PlotRenderer(const PlotRenderer&) = delete;
    PlotRenderer& operator=(const PlotRenderer&) = delete;

    // with_depth=true заводит depth-attachment и включает GL_DEPTH_TEST.
    // Нужен только 3D. Для 2D оставляем false (значение по умолчанию).
    void begin_frame(int w, int h, float clear_r, float clear_g, float clear_b, float clear_a,
        bool with_depth = false);

    // Рисует 2D-линию (vbo с float[2] на вершину).
    void draw_line(GLuint vbo, int point_count, const float mvp[16],
        const float color[4], float line_width);

    // Рисует 2D-точки (vbo с float[2] на вершину) через GL_POINTS.
    // Этим же путём рисуются точки 1D-бифуркационных диаграмм.
    // marker < 0 — старый путь: сплошной квадрат, GL-состояние не трогается.
    // marker >= 0 (PointMarker) — шейдерная маска формы + alpha-блендинг
    // (color[3] перестаёт игнорироваться).
    void draw_points(GLuint vbo, int point_count, const float mvp[16],
        const float color[4], float point_size, int marker = -1);

    // Хитмапа: рендерит fullscreen-quad внутри текущего FBO (begin_frame),
    // сэмплит R32F-текстуру tex и применяет colormap.
    //   colormap_id: 1001..1200 — карта slanCM. Значение прогоняется через
    //   colormap_id_or() прямо здесь, поэтому легаси 0..8 и мусор из старых
    //   сессий безопасны для любого вызывающего.
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

    // Рисует 3D-линию (vbo с float[3] на вершину).
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
    // Одна 256x200 RGB8-текстура на все карты slanCM: строка N-1 = карта #N.
    // Заливается напрямую из kSlanCmLut (layout совпадает), создаётся один
    // раз в конструкторе, как и шейдеры. 150 КБ VRAM на всю библиотеку.
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
    GLuint lut_tex_ = 0;         // 256x200 RGB8, см. ensure_lut_texture()

    GLuint vao_ = 0;

    GLint  saved_viewport_[4]{};
    GLint  saved_fbo_ = 0;
    GLboolean saved_depth_test_ = GL_FALSE;
};