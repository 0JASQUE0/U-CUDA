#pragma once
#include "imgui.h"
#include <map>
#include <string>
#include <vector>
// Одна строка легенды: подпись и цвет образца.
struct LegendEntry {
    std::string label;
    ImVec4      color;
};

// Режим прохода draw_legend. Позволяет вызвать функцию дважды: первый раз
// для взаимодействия (клики → toggle visible[]) в старом месте рендера,
// второй раз для визуала поверх линий данных (когда те рисуются через
// ImDrawList в custom line style). Так легенда всегда оказывается сверху,
// не ломая клики.
enum class LegendPass {
    Both,       // текущее поведение: submit InvisibleButtons + рисуем визуал
    Interact,   // только InvisibleButtons + hover/click, без dl-> draw
    Draw,       // только визуал через dl->; hover highlight через
                // IsMouseHoveringRect (InvisibleButton не создаём заново).
};

// Положение легенды — имена и смысл как у MATLAB legend('Location', ...). Внутренние стоят
// в поле графика; *outside — снаружи, вид отдаёт им место из поля (legend_reserve_margins):
// north/south outside — над и под полем по центру, остальные outside — слева или справа,
// выровнены по верху (north*), центру или низу (south*). best — внутреннее положение, где
// меньше всего точек данных (решает вид; без данных — northeast). none — ручное (pos),
// на него легенда переходит, когда её тянут ЛКМ. Номера пишутся в imgui.ini — не менять.
enum LegendLocation : int {
    kLegendNorthEast = 0, kLegendNorthWest, kLegendSouthEast, kLegendSouthWest,
    kLegendNorth, kLegendSouth, kLegendEast, kLegendWest,
    kLegendNorthOutside, kLegendSouthOutside, kLegendEastOutside, kLegendWestOutside,
    kLegendNorthEastOutside, kLegendNorthWestOutside, kLegendSouthEastOutside, kLegendSouthWestOutside,
    kLegendBest, kLegendNone,
    kLegendLocationCount
};
const char* legend_location_name(int loc);   // "northeast", ... как в MATLAB
bool        legend_location_outside(int loc);

// Где и как рисовать легенду. Хранится на окно (PlotWindowPrefs::legend_place, imgui.ini).
struct LegendPlacement {
    int   location   = kLegendNorthEast;
    // kLegendNone: левый верхний угол легенды долей свободного хода внутри поля графика
    // (0 — у левого/верхнего края, 1 — у правого/нижнего): при смене размера окна легенда
    // остаётся в том же месте поля и не выпадает за него.
    float pos[2]     = { 1.0f, 0.0f };
    bool  box_line   = true;    // Box: рамка
    bool  box_fill   = true;    // Box: подложка
    float line_alpha = 1.0f;    // непрозрачность рамки, 0..1
    float fill_alpha = 0.75f;   // непрозрачность подложки, 0..1
    // Число столбцов (MATLAB NumColumns): записи идут сверху вниз по столбцам, слева направо.
    int   columns    = 1;
};
constexpr int kLegendMaxColumns = 8;
bool legend_placement_default(const LegendPlacement& p);

// Прямоугольники, относительно которых ставится легенда: поле графика (там данные) и весь
// блок диаграммы (поле с осями и полями) — край блока служит местом для *outside.
struct LegendFrame {
    ImVec2 plot_min, plot_max;
    ImVec2 block_min, block_max;
    int    best = kLegendNorthEast;   // во что вид разрешил kLegendBest
};

// Размер легенды для этих записей в columns столбцов; (0, 0) — показывать нечего.
ImVec2 legend_size(const std::vector<LegendEntry>& entries, const std::vector<bool>& global_visible,
                   int columns = 1);
// Положение outside — прибавляет к полям вокруг графика место под легенду размера size.
void legend_reserve_margins(int loc, ImVec2 size, float& left, float& top, float& right, float& bottom);
// Левый верхний угол легенды размера size в положении loc (kLegendBest — как frame.best).
ImVec2 legend_origin(int loc, const LegendPlacement& place, ImVec2 size, const LegendFrame& frame);

// Рисует легенду в положении place.location (по умолчанию — правый верхний угол поля).
// Протяжка ЛКМ по легенде двигает её (place.location становится kLegendNone, place.pos —
// новое место); короткий щелчок по строке, как прежде, гасит/зажигает серию.
// visible        - локальное гашение: строка остаётся, но тускнеет (клик по ней).
// global_visible - внешний фильтр: записи с false в легенду не попадают вовсе
//                  и места в ней не занимают. Короче visible - хвост считаем видимым.
// owner_id       - идентификатор владельца, чтобы ID кнопок ImGui не совпали между плотами.
// out_clicks     - опционально: куда сложить результат ПКМ по легенде (см.
//                  LegendRightClick). Сам факт non-null указателя означает
//                  «у вызывающего есть меню цвета»: в Draw-проходе (где кликов
//                  не ловим) по нему рисуется рамка-подсказка на наведении
//                  квадрата. Поэтому передавать его следует в ОБА прохода, а
//                  легенда без такого меню (Plot3DView) оставляет nullptr —
//                  и подсказки не получает.

// Результат правого клика по легенде. Легенда подаёт свои InvisibleButton'ы
// РАНЬШЕ плотовского и забирает hover себе, поэтому плот про эти клики иначе
// не узнаёт вообще — а меню он показать обязан.
struct LegendRightClick {
    int  swatch_index = -1;     // ПКМ по цветному квадрату строки k -> меню цвета
    bool row_other    = false;  // ПКМ по строке мимо квадрата  -> обычное меню плота
};

void draw_legend(ImDrawList* dl,
    const LegendFrame& frame, LegendPlacement& place,
    const std::vector<LegendEntry>& entries,
    std::vector<bool>& visible,
    const std::vector<bool>& global_visible,
    int owner_id,
    LegendPass pass = LegendPass::Both,
    LegendRightClick* out_clicks = nullptr);

// Меню цвета серии (ПКМ по квадрату легенды), общее для 2D и 3D. Редактор — ColorEdit3, как у
// цвета области предпочтительности (поля RGB/HSV/Hex, пикер по клику на образец), без альфы:
// прозрачностью рулят слайдеры Alpha у графика. Вызывать каждый кадр; открывает его вызывающий
// (ImGui::OpenPopup(pop_id), target — подпись серии). current — нынешний цвет серии. Правка пишет
// overrides[target] (ключ — подпись, а не индекс: серии сдвигаются).
// Палитра серий по умолчанию (IC 1, IC 2, ... без своего цвета). Выбирается в Settings,
// хранится в _app_config.json (series_palette); те же ряды — в меню цвета легенды.
enum SeriesPaletteId { kSeriesPaletteGem12 = 0, kSeriesPaletteGlow12, kSeriesPaletteDeep, kSeriesPaletteCount };
const char* series_palette_name(int palette);
void        set_series_palette(int palette);   // вне диапазона -> gem12
int         series_palette();
int         series_palette_size(int palette);
ImVec4      series_palette_color(int palette, int index);   // index по кругу палитры
ImVec4      default_series_color(int index);                // из палитры series_palette()

// Недавние цвета меню легенды (новые первыми, до 8). Общие на всё приложение, хранятся
// в imgui.ini (секция [UcudaPlot][#RecentColours], см. plot_axis.cpp).
std::vector<ImVec4>& legend_recent_colors();
// Меню цвета серии: пипетка, пресеты (MATLAB gem / glow, основные), недавние.
// true — выбран «Apply to all plots of this tab» (разнести цвет target: plot_series_color_spread).
bool legend_color_popup(const char* pop_id, const std::string& target,
                        std::map<std::string, ImVec4>& overrides, const ImVec4& current);

// Подменю «Legend» меню диаграммы: Location (пресеты MATLAB), Box (рамка, непрозрачность
// рамки и подложки), сброс. true — что-то изменено (вызывающий помечает imgui.ini).
// spread_all — выставляется, если выбрано «Apply to all plots of this tab».
bool legend_placement_menu(LegendPlacement& place, bool has_best, bool& spread_all);
