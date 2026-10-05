#pragma once
#include <string>
#include <vector>
struct ParametricPlotWindow {
    // Номера сериализуются в _last_parametric_windows (int), новые — только в конец.
    enum class Kind { Bifurcation, LLE, LS, Metrics };
    Kind        kind    = Kind::Bifurcation;
    bool        mode_2d = false;
    // Kind::Metrics: какая метрика (SignalMetric) показана — выбирается таб-баром
    // над графиком, как вид в окне Basins. Живёт на окне, а не на конфиге: в 1D
    // окно оверлеит кривые нескольких конфигов, и показывать они обязаны одну
    // и ту же величину — у разных метрик разные единицы по Y.
    int         metric  = 0;
    // "Colored 1D diagram" (density-хитмапа) — 3-й display kind, только для
    // Bifurcation. Никогда не true одновременно с mode_2d (см.
    // draw_bifurcation_plot). Для LLE/LS остаётся false и не используется.
    bool        colored_1d = false;
    std::string label   = "Plot 1";   // window title, user-editable
    // Если false, label регенерируется из kind/mode_2d каждый кадр
    // ("Bifurcation 1D" и т.п.). Ставится в true при ручной правке,
    // сбрасывается при очистке поля. Дефолт true — старые сохранения
    // без ключа не перезатираем.
    bool        label_is_manual = true;
    // Indices into bifurcation_session.diagrams / lle_session.curves / ls_session.curves (selected
    // by `kind`), filtered to items whose .mode_2d/.colored_1d matches this window's own. Multiple
    // entries = overlay (typical for 1D); 2D/Colored-1D windows are UI-nudged, not hard-capped,
    // toward exactly one member.
    std::vector<int> members;
    // Stable identity for render-side caches (PlotRenderer/Plot2DView/
    // HeatmapView instances keyed by this, not by vector position — position
    // shifts when an earlier window is removed, this doesn't).
    int id = 0;
};

struct Dft1DPlotWindow {
    std::string label = "Plot 1";
    bool        label_is_manual = true;
    std::vector<int> members;
    int id = 0;
};

// Палитра БАЗОВЫХ цветов кривых окна Order (OrderPlotWindow::palette).
// Цвет кодирует вкладку-член окна, форма маркера — кривую внутри вкладки
// (min/avg/max, E1/E2), поэтому палитра тут нужна КАЧЕСТВЕННАЯ: соседние
// номера обязаны быть максимально далеки друг от друга. Градиентные карты
// хитмап (HeatmapColormap) для этого не годятся — они по построению непрерывны,
// и соседние кривые получают соседние оттенки.
//   kOrderPaletteApp      — ImPlot::GetColormapColor, как у фазовых портретов;
//   kOrderPaletteContrast — 12 насыщенных тонов средней светлоты, читаются и на
//                           светлой, и на тёмной теме плота;
//   kOrderPaletteOkabe    — Okabe-Ito: 8 цветов, различимых при всех трёх
//                           распространённых видах дальтонизма.
constexpr int kOrderPaletteApp      = 0;
constexpr int kOrderPaletteContrast = 1;
constexpr int kOrderPaletteOkabe    = 2;

// Одно окно графика вкладки Order — полный аналог ParametricPlotWindow.
// Что показано в окне, определяет список members (индексы
// order_session.configs), поэтому один и тот же конфиг может лежать сразу в
// нескольких окнах, а окно — собирать кривые нескольких вкладок. Ради этого
// сравнения (Эйлер против RK4 на одной диаграмме) окна и заведены.
//
// Настройки ОТОБРАЖЕНИЯ живут на окне, а не на конфиге: масштаб оси в окне
// общий для всех лежащих в нём кривых, и хранить его на конфиге значило бы
// иметь столько мнений о масштабе, сколько в окне вкладок.
struct OrderPlotWindow {
    // Одномерные виды (P / Error / Perf) оверлеят сколько угодно конфигов;
    // Map — хитмапа, и живёт ровно с одним.
    enum class Kind { P = 0, Error = 1, Map = 2, Perf = 3 };
    Kind kind = Kind::P;

    std::string label = "Plot 1";
    bool        label_is_manual = true;
    std::vector<int> members;
    int id = 0;

    bool x_log = true;
    bool y_log = true;          // лог берётся от ЗНАЧЕНИЙ: Plot2DView умеет лог только по X
    bool show_e2 = false;       // Error: вторая разность рядом с первой
    bool show_nominal = false;  // P: паспортный порядок схемы горизонталью
    bool map_error = false;     // Map: log10 E1 вместо p
    // Map, когда в окне лежит область устойчивости: что именно красить.
    //   0 — rho (спектральный радиус матрицы усиления);
    //   1 — 1 - rho: положительное = устойчиво. УМОЛЧАНИЕ, потому что именно
    //       эту величину рисует исходный матлабовский скрипт;
    //   2 — бинарная маска rho <= 1, то есть сама область без полутонов;
    //   3 — log10 err, относительная ошибка шага против e^{hA}.
    int  stab_view = 1;
    // Область предпочтительности (rho <= 1 и err <= допуска конфига)
    // полупрозрачной заливкой поверх любого из видов выше.
    bool stab_show_pref = true;
    // Цвета этого слоя, IM_COL32 (R в младшем байте, альфа в старшем):
    // заливка области и её контур. Умолчания — те же, что у HeatmapView.
    unsigned stab_pref_fill = 0x5A40E070u;   // зелёный, альфа ~0.35
    unsigned stab_pref_edge = 0xE660FF90u;   // светлее и плотнее
    int      stab_pref_edge_w = 1;           // толщина контура в ячейках сетки, 0 = без контура
    // Раскраска «как в статье» (Fedoseev et al., 2022, рис. 3-4): серая шкала
    // от чёрного к белому на [vmin, vmax], всё вне шкалы — белое. На виде
    // 1 - rho это даёт белый фон неустойчивого, чёрную границу rho = 1 и
    // светлеющую к rho = 0 внутренность. Пока включено, выбранная колормапа
    // окна не используется, но и не забывается.
    bool stab_paper_cmap = true;
    int  colormap_idx = -1;     // Map; -1 = взять app-дефолт
    int  error_source = 0;      // Perf: 0 = E1, 1 = E2 по оси X
    int  time_unit = 0;         // Perf: 0 = мкс, 1 = мс
    bool show_min = true, show_avg = true, show_max = true;   // Perf

    // ---- Стиль кривых окна ----
    // Живёт на окне по той же причине, что и масштаб осей: в окне лежат кривые
    // нескольких вкладок, и толщина линии у них обязана быть одна. Выключенный
    // custom_line_style даёт ровно прежний вид (1.5 px, непрозрачно, маркеры
    // узлов только на диаграмме производительности).
    bool  custom_line_style = false;
    float line_width   = 1.5f;
    float line_alpha   = 1.0f;
    bool  points_mode  = false;   // точки вместо линии
    int   point_marker = 0;       // PointMarker в points_mode
    float point_size   = 3.0f;
    // Маркеры в узлах ПОВЕРХ линии. На диаграмме производительности включены
    // всегда (иначе не к чему подвести курсор за подсказкой — h и время на
    // осях не лежат), здесь флаг добавляет их остальным видам.
    bool  node_markers = false;
    // Различать кривые ещё и формой маркера, а не только оттенком: min/avg/max
    // одной вкладки — один тон, и на плотной сетке оттенка не хватает.
    bool  vary_markers = true;
    // Палитра базовых цветов членов окна (см. kOrderPalette*). Умолчание —
    // контрастная, а не палитра приложения: в окне Order кривых заведомо
    // больше, чем в фазовом портрете (у каждой вкладки их до трёх), и
    // приглушённые тона ImPlot на них не расходятся.
    int   palette = kOrderPaletteContrast;

    // Сигнатура нарисованного на прошлом кадре: по её изменению запрашивается
    // autofit. Транзиентное, не сериализуется.
    int perf_cost = 0;
    int plot_sig = -1;
};
