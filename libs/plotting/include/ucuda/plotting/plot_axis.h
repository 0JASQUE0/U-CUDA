#pragma once
#include "imgui.h"
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <functional>

// ---------------------------------------------------------------------------
// Пиксельная модель плота. Область плота — пиксели [x0, x0+W) x [y0, y0+H) при
// целых x0/y0/W/H (origin округляется в Plot2DView/HeatmapView, размеры и так
// целые). Рамку рисует AddRect: он сам ужимается на 0.5 (imgui_draw.cpp) и
// занимает КРАЙНИЕ пиксели области — столбцы x0 и x0+W-1, строки y0 и y0+H-1.
// Значит и линии сетки, и штрихи тиков обязаны попадать в тот же диапазон.
//
// AddLine для однопиксельных линий не годится: он прибавляет +0.5 к ОБЕИМ
// точкам и центрирует штрих по отрезку. Целая координата от этого попадает на
// стык двух пикселей, а концы отрезка вылезают на полпикселя за заданный
// диапазон — из-за этого вертикальные линии сетки торчали снизу.
// ---------------------------------------------------------------------------

// Номер пикселя (левый/верхний край) для точки v внутри области [origin,
// origin+span). Кламп обязателен: значение на верхней границе оси даёт
// origin+span — первый пиксель ЗА областью, и тик уезжал за рамку.
inline float axis_px(float v, float origin, float span) {
    float p = std::floor(v);
    const float last = origin + span - 1.0f;
    if (p < origin) p = origin;
    if (p > last)   p = last;
    return p;
}

// Однопиксельные линии — заливкой ровно одного столбца/строки пикселей.
inline void fill_col_px(ImDrawList* dl, float x_px, float y_top, float y_bottom, ImU32 col) {
    dl->AddRectFilled(ImVec2(x_px, y_top), ImVec2(x_px + 1.0f, y_bottom), col);
}
inline void fill_row_px(ImDrawList* dl, float y_px, float x_left, float x_right, ImU32 col) {
    dl->AddRectFilled(ImVec2(x_left, y_px), ImVec2(x_right, y_px + 1.0f), col);
}

// Центр пикселя — для центрирования подписи под тиком.
inline float px_center(float p) { return p + 0.5f; }

// AxisInfo - состояние одной оси.
// Живёт внутри Plot2DView, в render-функции передаётся по ссылке.
struct AxisInfo {
    std::string name;
    double view_min = 0;
    double view_max = 1;
    bool   lock = false;
    bool   invert = false;
    // Диапазон задан вручную (поля min/max в меню оси по ПКМ): вид не прижимается к границам
    // данных и не сбрасывается автоподгонкой при новых данных — можно смотреть и за пределы
    // данных. Снимается явным Auto fit (меню) или пунктом меню.
    bool   user_range = false;
    // Sweep-сетка по этой оси log-распределена (см. BifurcationDiagramConfig::
    // log_scale и др.).
    //
    // Для оси X Plot2DView делает НАСТОЯЩЕЕ логарифмическое отображение: на
    // экран (VBO, MVP, курсор, крест) уходит log10(x), наружу всё остаётся
    // мировым. Логарифм не аффинен, поэтому в make_ortho_mvp он не помещается
    // — перенесена сама координата, см. XS/XW в Plot2DView::render.
    // Активен только при положительных границах вида: чекбокс можно включить
    // до Run, и log10(<=0) отравил бы NaN'ом весь кадр.
    //
    // draw_axis_x_grid при этом рисует декадную сетку (log_axis_ticks).
    // У оси Y лог-отображения нет: сетка по Y логарифмической не бывает.
    bool   log_scale = false;
    // Сетка оси — ±log бассейнов (ucuda_node_value_symlog): |x| по логарифму на [lo, hi]
    // по обе стороны от нуля, узлы -hi..-lo, +lo..+hi. Читает только HeatmapView: ячейки
    // и так равномерны по индексу узла, меняются подписи, подсказка, снап и крест.
    bool   symlog = false;
    // Лог-ОТОБРАЖЕНИЕ оси, выбранное пользователем (lin / log в меню оси Plot2DView). В отличие
    // от log_scale это не свойство данных: вызывающий его не трогает (log_scale он пишет каждый
    // кадр из настроек свипа), и работает оно на обеих осях. Плот логарифмический по оси, если
    // стоит log_scale (только X) или display_log. Точки с координатой <= 0 на лог-оси не
    // видны и в автоподгонку не входят.
    bool   display_log = false;
};

// Эффективные границы оси (с учётом invert).
inline void axis_effective(const AxisInfo& a, double& emin, double& emax) {
    if (a.invert) { emin = a.view_max; emax = a.view_min; }
    else { emin = a.view_min; emax = a.view_max; }
}

// «Красивый» шаг тиков (1, 2 или 5 * 10^n) под ~target_count делений.
double nice_step(double range, int target_count);

// Следующий «красивый» шаг вверх: 1 -> 2 -> 5 -> 10.
double nice_step_up(double step);

// Шаг тиков, при котором подписи не наезжают друг на друга. Раньше и число
// тиков, и шаг зависели только от диапазона, поэтому в узком окне подписи
// налезали одна на другую: 8 делений по X закладывались одинаково и на
// 900 px, и на 200. Шаг только увеличивается — на широком плоте всё остаётся
// как было.
//
// По X ширина считается по ФАКТИЧЕСКИМ подписям: "1.234e-05" втрое шире
// "0.5", и одному диапазону на том же плоте хватает восьми тиков, а другому
// — трёх. По Y достаточно высоты строки.
double fit_tick_step_x(double step, double lo, double hi, float plot_w);
double fit_tick_step_y(double step, double range, float plot_h);

// Не наедет ли подпись значения `v` на подпись соседнего тика `neighbor`.
// Подписи центрированы по тику, поэтому мешать друг другу они начинают, когда
// расстояние между тиками меньше полусуммы ширин (высот) плюс зазор.
//
// Нужно для тиков, которые добавляются ВНЕ регулярного шага: границы свипа в
// draw_axis_x_grid и крайние узлы сетки в HeatmapView. Шаг под них не
// подбирался, поэтому одного fit_tick_step_* мало — при зуме подпись границы
// («4.1235») длиннее круглой соседней и наезжала на неё.
bool tick_label_fits(double v, double neighbor, double lo, double hi,
                     float span_px, bool horizontal);

// Ширина самой широкой подписи на сетке {start + k*step} в пределах [lo, hi].
float max_tick_label_width(double start, double step, double lo, double hi);

// Шаг тиков, кратный шагу узла параметрической сетки, при котором подписи
// узлов не наезжают друг на друга. Возвращает кратность (>= 1) и через
// out_start — первый тик.
//
// Отдельно от fit_tick_step_x, потому что снап к узлам меняет не только шаг,
// но и САМИ ЗНАЧЕНИЯ: вместо круглых «10.8» на оси появляются узловые
// «10.732», которые в полтора раза шире. Мерить их можно только ПОСЛЕ снапа —
// подбор шага по круглым подписям давал «влезает» там, где не влезало.
int fit_node_step(double step_node, double node_origin, int mult0,
                  double lo, double hi, float span_px, bool horizontal,
                  double& out_start);

// Имеет ли смысл притягивать шаг тиков к узлам параметрической сетки. На
// плотной сетке — нет: «честный» тик визуально не отличается от круглого, а
// подпись превращается в 5.99 вместо 6, потому что шаг тика наследует
// иррациональный шаг сетки (16/499 на свипе 4..20 из 500 точек).
bool node_snap_visible(double step_node, double range, float span_px);

// Тик логарифмической оси: major подписывается числом, остальные дают только
// линию сетки (2·10^k, 3·10^k, ...).
struct LogAxisTick {
    double value = 0.0;
    bool   major = false;
};

// Тики лог-оси на [lo, hi] (обе границы > 0) под span_px пикселей: декады,
// прорежённые до читаемой плотности, плюс промежуточные мантиссы; на слишком
// узком для декад диапазоне — обычный nice_step. Позиции считаются по log10 —
// ровно тем же преобразованием, каким ось выводится на экран, поэтому подходит
// и Plot2DView (log10 в VBO), и HeatmapView (ячейки равномерны по индексу
// узла, а узлы лог-распределены).
std::vector<LogAxisTick> log_axis_ticks(double lo, double hi, float span_px,
                                        bool horizontal);

// Приглушённый цвет minor-линий лог-сетки: те же линии, что major, сбили бы
// восприятие декад.
inline ImU32 dim_grid_col(ImU32 c) {
    const ImU32 a = (c & IM_COL32_A_MASK) >> IM_COL32_A_SHIFT;
    return (c & ~IM_COL32_A_MASK) | ((a * 45 / 100) << IM_COL32_A_SHIFT);
}

// Форматирует подпись тика (с текущей точностью).
std::string fmt_tick(double v);

// Точность тиков: число значащих цифр для fmt_tick (форматирует через %g).
// Clamp'ится в [2, 10]. Дёргается при смене настройки в Settings.
void set_tick_precision(int n);

// MVP-матрица 2D-ортопроекции (column-major, GL).
void make_ortho_mvp(double xmin, double xmax, double ymin, double ymax, float out[16]);

// Рисует сетку и подписи по X (вертикальные линии + текст снизу).
// Опциональный snap-to-node: если snap_n > 1 и snap_hi > snap_lo, шаг тиков
// округляется к целому кратному step_node = (snap_hi - snap_lo)/(snap_n - 1),
// стартовая позиция — тоже кратна этому шагу. Значит тики попадают ровно на
// узлы параметрической сетки. Для 1D-графиков Bif/LLE/LS. Дефолты выключают
// snap (старое поведение).
// tick_labels = false — только сетка, без чисел (скрыты в меню плота).
void draw_axis_x_grid(ImDrawList* dl, const AxisInfo& x,
    ImVec2 plot_pos, float plot_w, float plot_h,
    ImU32 col_grid, ImU32 col_text,
    double snap_lo = 0.0, double snap_hi = 0.0, int snap_n = 0,
    bool tick_labels = true);

// Рисует сетку и подписи по Y (горизонтальные линии + текст слева).
void draw_axis_y_grid(ImDrawList* dl, const AxisInfo& y,
    ImVec2 plot_pos, float plot_w, float plot_h,
    ImU32 col_grid, ImU32 col_text, bool tick_labels = true);

// Plot color palette — переключается между Dark и Light темами. AppModel
// дёргает set_plot_light_theme(bool) на смену темы в Settings, а каждый
// plot/heatmap/legend читает текущие значения через геттеры. Прежде эти
// цвета были захардкожены под Dark, и на Light подписи/сетка/рамки осей
// сливались с белым фоном плотов.
void set_plot_light_theme(bool light);
bool plot_light_theme();

ImU32 plot_col_text();        // подписи осей, тики, ярлыки серий
ImU32 plot_col_axis();        // линии x=0 / y=0 на плоте (акцент)
ImU32 plot_col_grid();        // тики основной сетки
ImU32 plot_col_border();      // рамка плота, тики на colorbar

// Цвет очистки FBO под плот. Передаётся в PlotRenderer::begin_frame.
void  plot_bg_color(float& r, float& g, float& b, float& a);

// LaTeX-подписи на графиках. Весь текст на плотах (тики, имена осей, легенда,
// шкала colorbar) идёт через plot_text/plot_text_size вместо dl->AddText и
// ImGui::CalcTextSize, а те набирают строку мини-верстальщиком из plot_axis.cpp:
// имена греческих букв превращаются в глифы ("sigma" -> σ, "\lambda" -> λ),
// `_`/`^` — в индексы и степени, одиночные буквы идут курсивом, слова из
// нескольких букв ("parameter", "max") — прямым, как \mathrm в TeX. Плюс
// \dot{x}, \frac{a}{b} и запись 1e-05 как 10^-5.
//
// Пару шрифтов (прямой + курсив) грузит app_main вместе с UI-шрифтом и один
// раз отдаёт сюда — тот же принцип "set once, читается откуда угодно из
// plot-кода", что и у set_plot_light_theme выше. Шрифты могут быть nullptr:
// тогда верстальщик работает текущим шрифтом ImGui (греческие буквы и индексы
// остаются, засечек нет). Выключенный режим возвращает оба вызова к обычному
// ImGui-тексту байт-в-байт, поэтому чекбокс в Settings ничего не ломает.
void set_plot_math_fonts(ImFont* roman, ImFont* italic);
void set_plot_math_enabled(bool on);
bool plot_math_enabled();

// Кегль подписей на графиках — множитель к текущему шрифту ImGui. Не зависит
// от режима выше: в обычном (не-LaTeX) виде подписи масштабируются так же.
// Клампится в [0.5, 3.0]; слайдер — в Settings, персистится в _app_config.json.
// Марджины плотов и высота строк легенды считаются от него же, поэтому крупный
// кегль не вылезает за пределы блока диаграммы.
void  set_plot_font_scale(float s);
float plot_font_scale();

// Ширина набранной строки; высота — всегда высота строки подписей (базовый
// шрифт * plot_font_scale), чтобы вертикальная вёрстка плотов не зависела от
// того, вылезла ли степень над строкой.
ImVec2 plot_text_size(const char* s);
void   plot_text(ImDrawList* dl, ImVec2 pos, ImU32 col, const char* s);

// Тот же верстальщик для формулы вне плота — превью LaTeX-поля в Library.
// Отличия от plot_text: набирает всегда (не смотрит на plot_math_enabled),
// кегль задаётся явно, и строка читается как формула TeX: пробелы исходника
// игнорируются, отступы вокруг = + − ставятся сами, "ab" — курсив, \left,
// \right, \quad, \begin{...} понимаются. Серифная пара шрифтов грузится
// только при включённом LaTeX-режиме подписей, иначе набор идёт UI-шрифтом.
//
// ascent/descent — фактический габарит над и под базовой линией, с
// дробями, точками \dot и индексами; pos у math_formula_draw — верх габарита.
struct MathExtent { float width = 0, ascent = 0, descent = 0; };
MathExtent math_formula_extent(const char* s, float size);
void       math_formula_draw(ImDrawList* dl, ImVec2 pos, ImU32 col, const char* s, float size);

// Высота строки подписей. То же, что GetTextLineHeight() до появления кегля, —
// вызывать вместо него везде, где считается вёрстка вокруг подписей плота.
float  plot_text_line_height();

// Левый марджин плота под фактические подписи оси Y: ширина самого широкого
// тика текущего вида + повёрнутое имя оси + зазоры. Раньше тут стояла
// константа 78 px, подобранная под худший случай, — на коротких подписях
// ("0", "0.5") она съедала полтора сантиметра экрана впустую, а на крупном
// кегле разрасталась пропорционально.
//
// Результат квантуется вверх до 8 px: без этого край плота дёргался бы на
// каждый пиксель, пока подписи меняют длину при зуме.
//
// Тики берутся тем же nice_step, что и draw_axis_y_grid. Кто считает их своим
// генератором (HeatmapView со снапом к узлам сетки), измеряет ширину сам и
// зовёт plot_left_margin_for_width.
float plot_y_axis_margin(const AxisInfo& y, const char* y_name);
float plot_left_margin_for_width(float max_tick_w, bool has_axis_name);

// Нижний отступ: штрих + строка подписей тиков + зазор + строка имени оси.
// Раньше — константа 46 px, которую я умножал на кегль подписей целиком, из-за
// чего на множителе 1.6 под именем оси оставалось ~20 px пустоты: масштабу
// подчиняются строки текста, а не фиксированные зазоры.
// Вид диаграммы в окне — галки меню по ПКМ: тулбар над диаграммой, подписи (имя) и числа осей,
// colorbar. Хранится на корневое окно ImGui (по имени) и пишется в imgui.ini разделом
// [UcudaPlot], так что переживает перезапуск. Тулбары (draw_heatmap_toolbar, draw_style_toolbar,
// ...) спрашивают plot_toolbar_hidden() и не рисуются — диаграмме достаётся вся высота.
// Вид берёт ссылку в начале render(): внутри своего попапа текущее окно — уже попап.
struct PlotWindowPrefs {
    bool toolbar  = true;
    bool legend   = true;
    bool x_label  = true, x_ticks = true;
    bool y_label  = true, y_ticks = true;
    bool colorbar = true;
    bool axes3d   = true;   // оси координат 3D-сцены
    bool smooth   = false;  // сглаживание карт (PlotRenderer::draw_heatmap, smooth)
    // Оси: перестановка X<->Y (карты; срезы Custom) и инверсия. Источник — эти поля: вид в начале
    // render() берёт их отсюда, пункты меню пишут сюда.
    bool swap_axes = false;
    bool invert_x  = false, invert_y = false;
    // Цвета серий из меню легенды (ПКМ по квадрату -> legend_color_popup). Ключ — подпись серии,
    // а не индекс (серии сдвигаются, цвет должен остаться на своей кривой); альфа — от вызывающего.
    std::map<std::string, ImVec4> series_colors;
    // Вкладка приложения, на которой окно рисовалось последним (plot_set_window_scope); -1 — ещё
    // не рисовалось в этом запуске. Не сохраняется: нужна только для Ctrl+Shift+клика в «Show».
    int scope = -1;
};
// Вкладка приложения на текущий кадр: окна, отрисованные после вызова, запоминают её.
void plot_set_window_scope(int scope);
// Если меню легенды поменяло цвета p.series_colors (before — копия до него), помечает imgui.ini
// к сохранению; при зажатых Ctrl+Shift те же правки (по подписям) уходят во все окна вкладки.
void plot_series_colors_changed(PlotWindowPrefs& p, const std::map<std::string, ImVec4>& before);
// Цвет серии label из окна p — во все окна той же вкладки (свой цвет или палитровый, если сброшен).
void plot_series_color_spread(PlotWindowPrefs& p, const std::string& label);
// Группы подписей одной сущности на вкладке scope (например, «IC 1» фазы и «Custom 1D X»
// бифуркации — одна траектория): разнос цвета любой из них красит всю группу.
void plot_set_series_color_links(int scope, std::vector<std::vector<std::string>> groups);
// Сразу после ImGui::CreateContext: imgui.ini читается на первом NewFrame.
void plot_install_settings_handler();
PlotWindowPrefs& plot_window_prefs_current();
bool plot_toolbar_hidden();
// Вид окна (галки Show, swap, инверсия) строкой "key=v;..." — для сохранённых раскладок окон.
// window_name — имя корневого окна ImGui; окна может ещё не быть (настройки ждут его появления).
std::string plot_window_prefs_to_string(const std::string& window_name);
void        plot_window_prefs_from_string(const std::string& window_name, const std::string& s);
// Подменю «Show» меню диаграммы: что из её элементов видно (галки не закрывают меню);
// Ctrl+Shift+клик ставит то же значение во всех окнах диаграмм текущей вкладки. Пункты —
// только те, что у вида есть: legend — есть легенда, axes2d — подписи и числа осей X/Y,
// colorbar — у карт, axes3d — оси 3D-сцены.
enum PlotShowItems : unsigned {
    PlotShowLegend   = 1u << 0,
    PlotShowAxes2D   = 1u << 1,
    PlotShowColorbar = 1u << 2,
    PlotShowAxes3D   = 1u << 3,
    PlotShowSmooth   = 1u << 4,   // сглаживание картинки карты
};
void plot_show_menu(PlotWindowPrefs& p, unsigned items);

float plot_bottom_margin();
// То же, когда подписи тиков и/или имя оси X скрыты (меню плота): отступ сжимается.
float plot_bottom_margin_for(bool tick_labels, bool axis_name);

// Screenshot-to-clipboard. Право-клик "Copy image to clipboard" на любой
// диаграмме (Heatmap/Plot2D/Plot3D) заводится через request_plot_screenshot()
// — рект в экранных координатах ImGui (весь блок диаграммы: оси/colorbar/
// подписи, block_origin..+avail_size, не только FBO-картинка). app_main.cpp
// один раз подключает sink при старте (тот же принцип "set once, вызывается
// откуда угодно из plot-кода", что и set_plot_light_theme выше) — сам захват
// пикселей происходит НЕ синхронно внутри сюда, а через пару кадров (см.
// AppModel::PendingScreenshot), чтобы в кадр не попал ещё не закрывшийся popup.
void set_screenshot_request_sink(std::function<void(ImVec2 min, ImVec2 max)> sink);
void request_plot_screenshot(ImVec2 min, ImVec2 max);

// Числовой ввод в ImGui-полях: ↑/↓ шагают разряд под курсором
// (DigitInput::ComputeStep), запятая заменяется точкой. Поле обязано быть
// создано с флагами CallbackCharFilter | CallbackHistory.
//
// Живёт здесь, а не статиком в gui.cpp, потому что потребителей стало два:
// поля параметров во вкладках анализа (InputNumStr / InputNumStrCommit) и
// RGB-поля меню цвета серии в plot_view_2d.cpp. Иначе второму пришлось бы
// копировать колбэк целиком.
int digit_step_input_callback(ImGuiInputTextCallbackData* data);
