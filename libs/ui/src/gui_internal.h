#pragma once
#include "ucuda/ui/gui.h"
#include "ui_resources.h"
#include "ucuda/platform/platform_paths.h"
#include "imgui.h"
#include "imgui_internal.h"   // DockBuilder* API — needed for Custom Workspace per-tab dockspaces.
#include "implot.h"
#include "implot3d.h"
#include "ucuda/plotting/plot_renderer.h"
#include "ucuda/storage/session_io.h"
#include "ucuda/plotting/plot_view_2d.h"
#include "ucuda/plotting/plot_view_3d.h"
#include "ucuda/plotting/heatmap_view.h"
#include "ucuda/storage/app_config.h"
#include "ucuda/storage/data_export.h"
#include "ucuda/numerics/krs_cpu.h"
#include "ucuda/domain/num_parse.h"   // parse_num — единый разбор числовых полей
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ucuda::ui_detail {


// Именованные константы вместо магических чисел
// Значение, которым все хитмапы помечают «данных нет» (расходящаяся ячейка).
// HeatmapView красит его в серый; одно и то же число в colored-1D, DFT и Basins.
inline constexpr double kSentinelNoData = 999.0;

// Colormap по умолчанию для бассейнов и FastSync, когда в конфиге ничего не
// выбрано: slanCM #168 turbo — хорошо разделяет дискретные значения. Раньше
// здесь стоял id 2 (встроенный полиномиальный Turbo), который colormap_id_or
// теперь и переводит в эту же карту.
inline constexpr int    kColormapTurbo  = slancm_id(168);

// Высота бокса ошибки в строках текста.
inline constexpr int    kErrorBoxLines  = 12;

// Через столько кадров простоя вытесняется кэш рендереров владельца.
inline constexpr int    kRendererEvictFrames = 600;

// Дебаунс автозапуска фазовых портретов по бассейнам (сек).
inline constexpr double kBasinsPhaseDebounceSec = 0.3;

// Дебаунс автопересчёта Level 1D в Custom после возни со слайдером (сек).
inline constexpr double kCustomSliderDebounceSec = 0.5;

// Как часто перечитывается список систем в Library (сек). Файлы никто не
// правит извне при работающем приложении, поэтому опрос редкий.
inline constexpr double kLibraryCacheTtlSec = 10.0;

// Стандартная ширина числового поля и комбо в панелях настроек.
inline constexpr float  kFieldW = 120.0f;

inline constexpr float  kComboW = 160.0f;


// Возвращает директорию exe со слешем в конце. Реализована в app_main.cpp
// (там же используется для resolve_python_exe / library_dir).

std::string get_exe_dir_with_sep() ;


// Базовый цвет траектории по индексу НУ (единый для 2D/3D/time domain).
ImVec4 ic_base_color(int ic_index) ;


// Оттенок базового цвета по насыщенности: для переменной vi из nv внутри
// одного НУ. vi=0 — самый насыщенный, дальше бледнее. Используется в time domain,
// чтобы переменные одного НУ были видимо родственны (один тон), но различимы.
ImVec4 shade_of(ImVec4 base, int vi, int nv) ;

ImVec4 ic_var_shade(int ic_index, int vi, int nv) ;


// helpers: std::string <-> ImGui
// Подпись комбинации переменных: x0 + pi*x1 + e*x2 — тот ряд, что строит ядро при
// writable_var == -1 (loopCalculateDiscreteModel_int), с той же деградацией при
// dim < 3. Одна на Feature diagram и Time domain; у draw_writable_var_combo своя
// копия — она старше и в этот заход не трогалась.
[[nodiscard]] std::string combo_var_label(const std::vector<std::string>& vars) ;


// Единственное место, где видны ОБА определения pi: configCUDA.h (его читает
// ядро) и num_parse_detail::kPi (его подставляет разбор полей ввода). Если они
// разъедутся, пользователь введёт "pi" и получит не то число, которое считает
// GPU, — молча. Поэтому сверка на этапе компиляции.
static_assert(static_cast<double>(::pi) == num_parse_detail::kPi,
              "pi в num_parse.h разошлась с configCUDA.h");


// Значение комбинации в точке траектории — зеркалит формулу ядра. Константы
// pi/euler берутся из configCUDA.h, а не набираются заново.
[[nodiscard]] double combo_var_value(const std::vector<double>& pt, int nvars) ;


// Скретч-буфер под ImGui::InputText*: ImGui пишет в char*, а хранить значение надо
// в std::string. Один на поток вместо vector<char> в каждом виджете на каждом кадре
// (полсотни полей = сотня heap-аллокаций в кадр). Виджеты не вложены, а ImGui
// копирует содержимое внутри вызова — дольше вызова буфер не нужен; ёмкость между
// кадрами сохраняется, поэтому assign/resize уже не аллоцируют.
std::vector<char>& input_scratch(const std::string& str, size_t extra) ;


bool InputTextMultilineStr(const char* label, std::string& str, const ImVec2& size) ;


// Drag handle below a multiline input; drags along Y and clamps `height`.
void draw_resize_handle(const char* id, float& height,
                               float min_h = 40.0f, float max_h = 1200.0f) ;

bool InputTextStr(const char* label, std::string& str, float width = 0.0f) ;


// Перехватываем символы ДО того, как ImGui положит их в буфер: замена ',' → '.'
// идёт в момент ввода и не подменяет строку между кадрами. Иначе InputText на
// каждом кадре видит внешнюю правку буфера, возвращает changed=true, и
// auto_recompute триггерится непрерывно, пока в поле есть запятая.
int filter_comma_to_dot(ImGuiInputTextCallbackData* data) ;


// Совмещённый callback (запятая→точка + digit-step на ↑/↓) переехал в
// plot_axis.cpp::digit_step_input_callback — тем же вводом пользуется меню
// цвета серии в plot_view_2d.cpp, и держать копию в gui.cpp было нельзя.

// Разбирается ли строка как число или арифметическое выражение? Пустая валидна
// (дальше подставится дефолт). Валидны: "2.5", "1e-14", "8/3", "0.5*1e-3",
// "100-1", "(1+2)*3", "-2". Невалидны: "2/x", "8/0", "8/", "5asdfaxcv", "1e400",
// "(1+2". Делегирует в parse_num_checked — ту же грамматику, которой поле потом
// реально разбирается: своя копия на std::stod успела разойтись с ней (на
// "5asdf" подсветка писала «using default», а в расчёт уходило 5).
[[nodiscard]] bool is_numeric_string(const std::string& s) ;


// Is `scheme_name` one of the session's custom KRS schemes (as opposed to a
// built-in name)?
[[nodiscard]] bool is_custom_scheme(const std::string& scheme_name,
                              const std::vector<CustomScheme>& custom_schemes) ;


// Heuristic text match: does a custom KRS body actually reference a[0]
// (the symmetry slot, same as built-in CD)? Raw C/CUDA source, not parsed
// into an AST, so this is a regex over the literal text rather than a real
// usage analysis — good enough since users write "a[0]" directly.
[[nodiscard]] bool custom_scheme_uses_symmetry(const std::string& scheme_name,
                                         const std::vector<CustomScheme>& custom_schemes) ;


// Читает ли схема коэффициент симметрии a[0] — от этого зависит, показывать ли
// поле symmetry в конфигах и пункт "s (a[0])" в выборе цели свипа. Встроенных
// таких пять: CD (h1 = s*h, h2 = (1-s)*h), оба Complex CD (то же деление шага,
// плюс мнимая часть ±h*sqrt(3)/6) и SEMP/SIMP (стадия на полушаге h1 = s*h).
// Один предикат на весь файл: раньше проверка была выписана четырьмя копиями
// `scheme == "CD" || custom_scheme_uses_symmetry(...)`, и добавление схемы
// требовало не забыть все четыре.
[[nodiscard]] bool scheme_uses_symmetry(const std::string& scheme_name,
                                         const std::vector<CustomScheme>& custom_schemes) ;


// Auto-labels for BD/LLE/LS configs and Parametric plot windows
// Каждый label регенерируется каждый кадр из текущего свипа, пока
// label_is_manual = false. Пользователь помечает как ручной, отредактировав
// поле; очистка поля возвращает в auto.

// ЕДИНСТВЕННАЯ реализация имени оси свипа (были три копии-лямбды в
// draw_bifurcation_plot / draw_lle_plot / draw_ls_plot). Fallback при индексе НУ вне
// диапазона — "var", а не "x": "x" выглядит как имя реальной, но несуществующей переменной.
[[nodiscard]] std::string auto_axis_name(const std::vector<std::string>& params,
                                   const std::vector<std::string>& vars,
                                   int param_idx, bool sweep_over_var, int var_idx,
                                   bool sweep_over_h = false) ;


// Хвост подписи 1D-свипа: "<ось> [lo..hi]". Пустое поле показывается как "?".
[[nodiscard]] std::string sweep_range_label(const std::string& axis,
                                                   const std::string& lo_text,
                                                   const std::string& hi_text) ;


// Автоподпись BD / LLE / LS: "<x> x <y>" в 2D-режиме, "<x> [lo..hi]" в 1D.
// Шаблон, а не общий базовый класс: конфиги независимы, связывать их
// наследованием ради одинакового набора полей было бы хуже. Раньше — три
// побайтово одинаковых тела (auto_label_bd / _lle / _ls).
template <class Cfg>
[[nodiscard]] inline std::string auto_label_sweep(const Cfg& c,
                                                  const std::vector<std::string>& params,
                                                  const std::vector<std::string>& vars) {
    const std::string x = auto_axis_name(params, vars, c.param_index,
                                         c.sweep_over_var, c.var_sweep_index, c.sweep_over_h);
    if (c.mode_2d) {
        const std::string y = auto_axis_name(params, vars, c.param_index_2,
                                             c.sweep_over_var_2, c.var_sweep_index_2,
                                             c.sweep_over_h_2);
        return x + " x " + y;
    }
    return sweep_range_label(x, c.param_lo_text, c.param_hi_text);
}


[[nodiscard]] std::string auto_label_window(const ParametricPlotWindow& w) ;


// У DFT1D 2D-режима нет, поэтому auto_label_sweep (он читает mode_2d и поля
// второй оси) не подходит — но хвост общий.
[[nodiscard]] std::string auto_label_dft1d(const Dft1DConfig& c,
                                    const std::vector<std::string>& params,
                                    const std::vector<std::string>& vars) ;


void refresh_auto_labels(AppModel& model) ;


// Историческое имя парсера в этом файле. Тело переехало в num_parse.h —
// теперь одна реализация на весь проект (см. комментарий там же о том, из-за
// чего пять разных версий расходились). Имя сохранено: оно стоит примерно в
// сорока местах и читается по месту лучше, чем голое parse_num.
[[nodiscard]] inline double parse_ratio_or(const std::string& v, double def) ;


// Целочисленные поля: Resolution, decimator, число бинов, N сетки.
// Тело переехало в num_parse.h к parse_num_int — там же объяснено, почему
// вторая копия этого разбора (на std::stoi, в analysis_session.cpp) молча
// врала на "512/2". Имя сохранено: оно стоит по месту в gui.cpp.
[[nodiscard]] inline int parse_int_or(const std::string& v, int def) ;


bool InputNumStr(const char* label, std::string& str, float width = 0.0f) ;


// Общие хелперы диаграмм — единый источник истины для настроек view'ов и тулбаров.
// Раньше каждая вкладка собирала всё заново, и одна и та же диаграмма отличалась
// между вкладками (линии x=0/y=0 у Bif-1D, autoscale/vmin/vmax у Basins, Swap axes
// в Custom). Всё про «как выглядит и что умеет диаграмма типа X» живёт здесь.

// Как InputNumStr, но число живёт в структуре (Settings -> PeakConfig), а текст —
// только буфер ввода: значение снимается на коммите (Enter или уход фокуса), иначе
// промежуточное "1e-" по пути к "1e-14" ушло бы в set_peak_config и дёрнуло
// перекомпиляцию ядер. Невалидный/пустой текст значение не трогает.
// Возвращает true ровно на кадре коммита.
bool InputNumStrCommit(const char* label, std::string& text, double& value,
                              float width = 0.0f) ;


// Общие блоки панелей настроек: единственные реализации элементов, ранее
// скопированных по вкладкам Bif / LLE / LS / DFT / Basins / FastSync / Custom по
// 4–10 раз. Копии расходились (в Custom НУ подписывались "IC x" вместо "x (IC)",
// decimator был не у всех), и чинить приходилось каждую отдельно.
// ImGui-идентификаторы передаются строкой снаружи и побайтово совпадают с прежними —
// раскладка imgui.ini и состояние вкладок сохраняются.

// Встроенные схемы интегрирования. Один список на весь файл: раньше этот же
// массив был выписан восемью копиями плюс девятой — для проверки конфликта
// имён пользовательских схем.
// order — порядок точности: по нему схемы сгруппированы и в комбо, и в
// чекбоксах System tab, поэтому таблица отсортирована по нему и порядок строк
// здесь = порядок в UI. У CD, Complex CD, SEMP и SIMP заявленный порядок
// достигается только при s = a[0] = 0.5 (см. codegen.cpp).
struct BuiltinScheme { const char* name; int order; };

inline const BuiltinScheme kBuiltinSchemes[] = {
    { "Euler",             1 },
    { "Euler-Cromer",      1 },
    { "D",                 1 },
    { "Implicit Euler",    1 },
    { "Explicit Midpoint", 2 },
    { "Implicit Midpoint", 2 },
    { "CD",                2 },
    { "Complex CD",        2 },
    { "Complex Implicit Euler", 2 },
    { "SEMP",              2 },
    { "SIMP",              2 },
    { "RK4",               4 },
    { "Complex CD4",       4 },
    { "DOPRI78",           8 },
};


// Имена для ImGui::Combo, которому нужен массив const char*. Указатели живут,
// пока жив исходный вектор строк, — результат используется тут же, в том же
// кадре. Раньше эти три строки были выписаны трижды (Basins и оба режима
// FastSync).
[[nodiscard]] std::vector<const char*> c_str_list(const std::vector<std::string>& v) ;


// Комбо выбора схемы: встроенные + (через разделитель) пользовательские.
// on_pick вызывается ПОСЛЕ записи имени в scheme — им пользуются Phase
// (regenerate_krs) и Custom Shared config (проброс в phase_session).
// Возвращает true, если пользователь выбрал схему в этом кадре.
bool draw_adaptive_controls(adaptive::Options& a) ;


bool draw_scheme_combo(const char* label, std::string& scheme,
                              const std::vector<CustomScheme>& custom_schemes,
                              const std::function<void(const std::string&)>& on_pick = {},
                              const std::vector<std::string>* enabled_builtins = nullptr) ;


// ЕДИНСТВЕННОЕ комбо выбора цели свипа: параметры, через разделитель начальные условия как
// "<var> (IC)", затем "dt (h)". Раньше — семь копий по ~45 строк во вкладках плюс восьмая в
// Custom со своим стилем подписей.
//
// other_over_h — флаг ВТОРОЙ оси (nullptr = 1D-контекст, ограничения нет). По h может
// свипаться ровно одна ось: движок кодирует её ОДНИМ числом hSweepAxis (-1 нет / 0 X / 1 Y) и
// запрос с обеими осями отвергает. Поэтому "dt (h)" здесь всегда доступен, а выбор ПЕРЕНОСИТ
// шаг на эту ось, гася флаг соседней. Раньше на это правило приходилось три разных поведения
// (ось X переносила, ось Y прятала пункт вовсе, Custom показывал серым), и перенести шаг
// можно было только в две операции.
//
// note_when_empty — показывать ли подсказку вместо комбо, когда в системе нет ни параметров,
// ни переменных (так делают вкладки; Custom рисует комбо всегда).
//
// allow_s — показывать ли пункт "s (a[0])", свип по коэффициенту симметрии. Он
// имеет смысл только у схем, которые a[0] читают (CD / Complex CD / Complex CD4 /
// SEMP / SIMP и custom с a[0] в теле — см. scheme_uses_symmetry), иначе свип шёл
// бы по значению, которого КРС не видит, и дал бы плоскую картинку. Выбор
// кодируется par_index = -1: в a[] это слот 0, поэтому отдельного флага (в
// отличие от over_h) не нужно — движок свипует a[0] как любой другой параметр.
void draw_sweep_target_combo(const char* label,
                                    const std::vector<std::string>& params,
                                    const std::vector<std::string>& vars,
                                    int& par_index, bool& over_var, int& var_index,
                                    bool& over_h,
                                    bool* other_over_h = nullptr,
                                    bool note_when_empty = false,
                                    float width = kComboW,
                                    bool allow_s = false) ;


// Поля секции "Integration". nullptr = поле у этого анализа отсутствует
// (например, у LLE/LS нет decimator'а). Условие показа "symmetry s" —
// схема CD либо пользовательская, чьё тело обращается к a[0].
struct IntegrationFields {
    std::string* h           = nullptr;
    std::string* symmetry_s  = nullptr;
    std::string* t_max       = nullptr;
    std::string* transient   = nullptr;
    std::string* pre_scaller = nullptr;
    std::string* max_value   = nullptr;
};


bool draw_integration_block(const char* header,
                                   const std::string& scheme,
                                   const std::vector<CustomScheme>& custom_schemes,
                                   const IntegrationFields& f) ;


// Секции "Initial conditions" и "Parameters" — это один и тот же блок:
// по числовому полю на каждое имя из списка. Раньше — около дюжины копий.
void draw_named_num_fields(const char* header,
                                  const std::vector<std::string>& names,
                                  std::map<std::string, std::string>& values,
                                  const char* note = nullptr) ;


// Секция "CSV output": галка сохранения + путь + пояснение про побочные файлы.
void draw_csv_output_block(const char* header, const char* checkbox_label,
                                  bool& enabled, const char* path_id, std::string& path,
                                  const char* hint) ;


// Бокс с текстом ошибки последнего прогона: read-only, чтобы можно было
// выделить и скопировать. const_cast здесь единственный на файл — ImGui
// требует char*, но с ImGuiInputTextFlags_ReadOnly в буфер не пишет.
void draw_error_box(const char* id, const std::string& err,
                           int lines = kErrorBoxLines) ;


// Пункт "Export data..." правого клика по диаграмме. Диалог выбора файла —
// платформенный, приходит из GuiCallbacks; write вызывается только с
// непустым путём (пустой = пользователь нажал Отмену).
void draw_export_menu_item(bool busy, const GuiCallbacks& cb,
                                  const std::function<void(const std::string&)>& write) ;


// Inline-переименование активной вкладки. Раньше — два ручных char buf[128],
// которые молча обрезали имя на 127 символах.
void draw_label_rename(const char* label_with_id, std::string& label,
                              float width = 200.0f) ;


// Результат таб-бара конфигов: индекс активной вкладки и индекс той, что
// закрыли крестиком (-1 = ничего).
struct TabBarResult {
    int active    = -1;
    int to_remove = -1;
};


// Таб-бар "одна вкладка на конфиг + кнопка +" (раньше — шесть копий).
// item_id_prefix задаёт неизменную часть ID вкладки ("bd_tab_", "lle_tab_", …), поэтому
// идентификаторы побайтово совпадают с прежними. body == nullptr — тело вкладки пустое
// (DFT/Basins/FastSync рисуют настройки активного конфига уже после EndTabBar).
// request_select — внешний запрос выбрать вкладку (тулбар Colored 1D у Bif).
TabBarResult draw_config_tab_bar(const char* bar_id, const char* item_id_prefix,
                                        int n, bool in_flight, int running_index,
                                        const std::function<std::string(int)>& label,
                                        const std::function<void(int)>& body,
                                        const std::function<void()>& on_add,
                                        int request_select = -1) ;


// Одна группа отметок в попапе "Run all...". У Parametric их три
// (Bifurcation / LLE / LS), у остальных вкладок — одна, без заголовка.
struct RunAllGroup {
    const char* title   = nullptr;   // заголовок секции; nullptr — без него
    const char* pick_id = "";        // неизменная часть ID чекбокса ("pbd_", …)
    int         n       = 0;
    std::function<std::string(int)> label;
    std::function<void(int)>        enqueue;   // положить элемент i в очередь
};


// Строка "Run (Ctrl+R) [| доп. кнопка] | Run all... (Ctrl+Shift+R) | (N queued)" вместе с
// попапом выбора; раньше — четыре копии по ~60 строк.
// Отметки хранятся ПО popup_id (единый сливал бы выбор между вкладками) и живут
// снаружи if(BeginPopup) намеренно: Ctrl+Shift+R обязан пушить те же отметки, даже если попап
// ни разу не открывали — тогда отмечено всё.
// block_run_all передаётся отдельно от busy/no_active: вкладки блокируют "Run all..." ещё и
// при пустом списке конфигов, а Parametric — только на время расчёта (у него три списка).
void draw_run_and_run_all(const char* popup_id,
                                 bool busy, bool no_active, bool block_run_all,
                                 float run_w,
                                 const std::function<void()>& on_run,
                                 const std::function<void()>& extra_after_run,
                                 const std::vector<RunAllGroup>& groups,
                                 const std::function<void()>& start_queue,
                                 size_t queued) ;


// Тип 1D-диаграммы параметрического семейства. Нейтрален к вкладке — им
// пользуются и Parametric (ParametricPlotWindow::Kind), и Custom (L1Kind),
// чтобы конфигурация Plot2DView шла из ОДНОГО места.
enum class ParamPlotKind { Bifurcation, LLE, LS };


// Единая конфигурация Plot2DView под параметрический 1D-график. Вызывается и из
// Parametric, и из Custom — иначе одинаковые диаграммы расходятся по отрисовке.
//
// Задавать здесь ТОЛЬКО свойства типа диаграммы и НИКОГДА пользовательское
// состояние (show_legend, lock/invert осей): Custom зовёт функцию каждый кадр и
// затирал бы переключённое пользователем — так ломалась галка «Show legend» на
// 1D-графиках Custom. Дефолты таких полей живут в самом Plot2DView.
//
// show_zero_y осмыслен ТОЛЬКО у LLE/LS (λ=0 — граница хаос/порядок, стандартный
// референс). По X это ось параметра, по Y у Bif — переменная состояния: ноль там
// произволен, линии нет.
void configure_param_plot_view(Plot2DView& view, ParamPlotKind kind) ;


// Custom point style (Bifurcation 1D scatter)
// ЕДИНСТВЕННАЯ реализация тулбара «Custom point style / Marker / Point size /
// Alpha»: им пользуются и Parametric (draw_bifurcation_plot), и Custom (Level-1D
// Bif-слоты), чтобы настройка вела себя одинаково везде, где есть Bifurcation 1D.
// Состояние живёт в BifurcationDiagramConfig и персистится вместе с сессией.
// Возвращает true, если пользователь что-то изменил (сохранять сессию решает вызывающий).
//
// В строке остаются только галочка и стрелка "▾", параметры — в popup'е: через
// SameLine они уезжали за правый край узкого докированного окна. Popup рисуется
// ВСЕГДА, а не только при enabled, иначе снятие галочки с открытым меню пропустило
// бы EndPopup для уже открытого ImGui-окна.
bool draw_style_toolbar(const char* label, const char* id_suffix,
                               bool& enabled,
                               const std::function<bool()>& body) ;


bool draw_point_style_toolbar(BifurcationDiagramConfig& bd, const char* id_suffix) ;


// Перенос настроек точек из конфига БД во вид. Звать ПОСЛЕ
// configure_param_plot_view (та ставит дефолтные 2px). Выключенный режим
// возвращает ровно прежний вид (marker = -1 → сплошной квадратный GL-пойнт).
void apply_point_style(Plot2DView& view, const BifurcationDiagramConfig& bd) ;


// Continuation + выбор устройства: ЕДИНЫЙ блок для Bif / LLE / LS 1D (раньше у Bif
// был свой, у LLE/LS свой, и состав элементов с ограничениями расходился):
//   continuation — точки идут цепочкой (переносятся траектория и, у LLE/LS, векторы
//     возмущения). Требует param-свипа: IC-свип с цепочкой несовместим. h-свип и
//     log-сетка поддержаны, в 2D-режиме неприменим.
//   use_gpu — где считать. Обе ветки есть у всех трёх анализов, но GPU-continuation —
//     single-thread kernel, поэтому при включении continuation уходим на CPU.
// Шаблон, а не общий базовый класс: конфиги независимы, наследование ради трёх полей
// было бы хуже. blocked считает вызывающий (у Bif/LLE/LS это mode_2d || sweep_over_var,
// у DFT1D 2D-режима нет вовсе) — это проще, чем требовать от конфигов общий набор
// полей. device_locked — выбор устройства недоступен (в 2D CPU-веток нет).
template <class Cfg>
inline void draw_continuation_device_block(Cfg& c, const char* id, bool blocked,
                                           bool device_locked) {
    const std::string sfx = id;
    if (blocked) {
        c.continuation = false;
        c.continuation_reverse = false;
    }

    ImGui::BeginDisabled(blocked);
    bool cont = c.continuation;
    if (ImGui::Checkbox(("Continuation##" + sfx).c_str(), &cont)) {
        c.continuation = cont;
        // При включении continuation по умолчанию уходим на CPU (GPU-ветка там
        // однопоточная), при выключении возвращаемся на GPU. В обе стороны это
        // только дефолт: радио ниже активно, и выбор пользователя держится.
        c.use_gpu = !cont;
    }
    if (c.continuation) {
        ImGui::SameLine();
        int dir = c.continuation_reverse ? 1 : 0;
        ImGui::RadioButton(("forward##"  + sfx).c_str(), &dir, 0); ImGui::SameLine();
        ImGui::RadioButton(("backward##" + sfx).c_str(), &dir, 1);
        c.continuation_reverse = (dir == 1);
    }
    ImGui::EndDisabled();
    if (blocked)
        ImGui::TextDisabled("(continuation: только param- или h-sweep, не IC, не 2D)");

    int dev = c.use_gpu ? 0 : 1;
    ImGui::BeginDisabled(device_locked);
    ImGui::RadioButton(("GPU##dev" + sfx).c_str(), &dev, 0); ImGui::SameLine();
    ImGui::RadioButton(("CPU##dev" + sfx).c_str(), &dev, 1);
    ImGui::EndDisabled();
    if (!device_locked) c.use_gpu = (dev == 0);
    if (c.continuation && c.use_gpu) {
        ImGui::SameLine();
        ImGui::TextDisabled("(continuation на GPU однопоточный, CPU быстрее)");
    }
    if (!c.use_gpu) {
        std::string why;
        if (!krs_cpu_backend_available(&why)) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1), "(no compiler: %s)", why.c_str());
        }
    }
}


// Snap X к узлам параметрической сетки: тики осей и hover-readout попадают
// ровно на посчитанные точки. Раньше — три идентичных блока в Bif/LLE/LS и
// полное отсутствие в Custom (из-за чего tooltip там врал между узлами).
// n < 2 отключает snap (readout остаётся непрерывным).
void apply_snap_x(Plot2DView& view, double lo, double hi, int n) ;


// Значение свипа в узле k из n — ТА ЖЕ формула, по которой движок считал эту точку
// (cont_sweep_value / getValueByIdx_log в parametric_engine.cpp). Графики когда-то
// всегда раскладывали точки линейно, и при log_scale данные уезжали относительно
// лог-оси: узлы лежат на 10^(l0 + (l1-l0)*t), а рисовались на lo + (hi-lo)*t.
// При lo/hi <= 0 на лог-ветке деградируем на линейную сетку вместо NaN: сюда может
// долететь живой чекбокс без прогона (тот же приём, что в SnapCursorToGrid1D).
//
// Классический свип идёт через общую с ядром ucuda_node_value (configCUDA.h);
// раньше обе ветки считались cont-формулой, из-за чего ось классической диаграммы
// расходилась с getValueByIdx в последних битах, а на правом конце сильнее (lerp даёт
// РОВНО hi). У continuation СВОЯ конвенция узлов (lo + (hi-lo)*t плюс reverse) — она
// живёт в kernels/*_cont.template.cu и cont_sweep_value и остаётся здесь копией: те
// шаблоны configCUDA.h не включают.
//
// `reverse` авторитетнее флага continuation: первый приходит из РЕЗУЛЬТАТА
// (result.continuation_reverse), второй из конфига, и они разъезжаются, если снять
// галочку без перезапуска. Данные тогда по-прежнему backward, а классическая ветка
// про reverse не знает и нарисовала бы кривую зеркально.
[[nodiscard]] double sweep_value_at(int k, int n, double lo, double hi,
                             bool log_scale, bool reverse, bool continuation) ;


// Описание свипа одного члена окна — всё, что нужно для общей X-оси.
struct SweepAxisMember {
    int    kind      = 0;      // 0 = параметр, 1 = переменная (IC), 2 = шаг h
    int    index     = -1;     // индекс в params/vars (для kind==2 не нужен)
    bool   log_scale = false;
    double lo = 0.0, hi = 1.0; // диапазон, с которым РЕАЛЬНО шёл Run
};


// Общая настройка X-оси 1D-графика по членам окна (раньше — три почти одинаковых
// блока по ~35 строк в Bif/LLE/LS):
//   - подпись = общий sweep-таргет; если члены свипают разное — "parameter";
//   - fit-диапазон = union(lo, hi) ВСЕХ членов: ось обязана охватывать весь свип,
//     даже если часть точек разошлась и на графике их нет;
//   - log_scale только если он одинаков у всех членов.
void configure_sweep_x_axis(Plot2DView& view,
                                   const std::vector<SweepAxisMember>& members,
                                   const std::vector<std::string>& params,
                                   const std::vector<std::string>& vars) ;


// Сбор членов окна и настройка общей X-оси одним вызовом. Раньше — три почти
// одинаковых блока по ~20 строк в Bif/LLE/LS, из которых версии LLE и LS
// совпадали буквально. Отличается только источник диапазона (у Bif снапшот в
// result пишется лишь для continuation), поэтому его отдаёт range_fn.
template <class Cfg, class RangeFn>
inline void configure_sweep_x_axis_from(Plot2DView& view,
                                        const std::vector<int>& members,
                                        const std::vector<Cfg>& cfgs,
                                        const std::vector<std::string>& params,
                                        const std::vector<std::string>& vars,
                                        RangeFn range_fn) {
    std::vector<SweepAxisMember> out;
    out.reserve(members.size());
    for (int idx : members) {
        if (idx < 0 || idx >= (int)cfgs.size()) continue;
        const Cfg& c = cfgs[(size_t)idx];
        if (!c.last_run_ok) continue;
        SweepAxisMember m;
        m.kind      = c.sweep_over_h ? 2 : (c.sweep_over_var ? 1 : 0);
        m.index     = c.sweep_over_var ? c.var_sweep_index : c.param_index;
        m.log_scale = c.log_scale;
        range_fn(c, m.lo, m.hi);
        out.push_back(m);
    }
    configure_sweep_x_axis(view, out, params, vars);
}


// Диапазон свипа для LLE/LS: снапшот прогона, а если движок его ещё не
// заполнил (lo == hi) — текстовые поля конфига.
void sweep_range_from_result(double res_lo, double res_hi,
                                    const std::string& lo_text, const std::string& hi_text,
                                    double& lo, double& hi) ;


// Snap-диапазон по ПЕРВОМУ члену окна: берём сетку успешного прогона, иначе
// парсим текстовые поля конфига (до первого Run их всё равно больше нечем
// заполнить). Раньше — три идентичных блока в Bif/LLE/LS.
void apply_snap_x_from_config(Plot2DView& view,
                                     bool have_result, double res_lo, double res_hi, int res_n,
                                     const std::string& lo_text,
                                     const std::string& hi_text,
                                     const std::string& n_text) ;


// Тот же snap, но по ПЕРВОМУ члену окна. Раньше — три идентичных хвоста
// по 12 строк в draw_bifurcation_plot / draw_lle_plot / draw_ls_plot.
template <class Cfg>
inline void apply_snap_x_from_first_member(Plot2DView& view,
                                           const std::vector<int>& members,
                                           const std::vector<Cfg>& cfgs) {
    view.snap_x_to_grid = true;
    view.snap_x_n       = 0;
    if (members.empty()) return;
    const int aidx = members[0];
    if (aidx < 0 || aidx >= (int)cfgs.size()) return;
    const Cfg& c = cfgs[(size_t)aidx];
    apply_snap_x_from_config(view, c.last_run_ok,
                             c.result.param_lo, c.result.param_hi, c.result.n_pts,
                             c.param_lo_text, c.param_hi_text, c.n_pts_text);
}


// Right-click «Export data...» подменю: перечисляет ВСЕ конфиги сессии с готовым
// прогоном, а не только members этого окна — экспорт к окну не привязан.
// ready(i) — есть законченный прогон, busy(i) — считается сейчас, do_export(i, path) —
// собственно запись. Раньше — три идентичные копии в Bif/LLE/LS.
void draw_export_submenu(const char* id_tag, int n,
                                const std::function<std::string(int)>& label,
                                const std::function<bool(int)>& ready,
                                const std::function<bool(int)>& busy,
                                const std::function<void(int, const std::string&)>& do_export,
                                const GuiCallbacks& cb) ;


// Ленивое создание per-config HeatmapView в map'е окна (map уже per-window, пересечений
// между окнами нет). При первом создании подхватывает colormap: приоритет у выбора из
// конфига (cfg_colormap, -1 = не задан), иначе общий app-дефолт. cfg_exponent_idx
// применяется только при != kNoExponent (актуально лишь для LS).
// Раньше — четыре копии лямбды get_*_heatmap (Bif / LLE / LS / DFT1D).
inline constexpr int kNoExponent = -999;

HeatmapView& get_or_create_heatmap(
        std::map<int, std::unique_ptr<HeatmapView>>& map, int idx,
        int cfg_colormap, int app_default_colormap,
        int cfg_exponent_idx = kNoExponent) ;


// Скретч-буферы точек одного plot-окна (по одному на серию). Между кадрами живут намеренно:
// buf.clear() у вызывающего сохраняет capacity, поэтому со второго кадра push_back уже не
// аллоцирует — на диаграммах в миллионы точек это заметно.
// Ключ — стабильный ParametricPlotWindow::id, как у кэшей HeatmapView выше. Раньше на каждую
// из трёх функций был один static, общий ВСЕМ окнам: при двух окнах с разным числом серий
// assign() передёргивал буферы каждый кадр, и оптимизация не работала (корректность при этом
// не страдала — render() забирает точки синхронно).
// Записи удалённых окон остаются в карте до перезапуска, как и у хитмап: окон единицы.
std::vector<std::vector<float>>& window_point_bufs(int window_id, size_t n_series) ;


// Полоска-превью колормапа как ImGui-виджет: kSeg сегментов с линейным градиентом внутри
// каждого плюс Dummy того же размера, чтобы полоска занимала место в layout'е. 24 сегмента
// хватает, чтобы читались даже рваные качественные карты (tab20, glasbey), а рисуется это
// только для видимых строк — и в попапе пикера, и под клиппером в Settings.
void colormap_strip_item(int id, float w, float h) ;


// Пикер колормапа: BeginCombo вместо ImGui::Combo, потому что список теперь разреженный
// (только отмеченные пользователем карты slanCM) и в каждой строке рисуется градиент. Текущий
// id показывается всегда, даже если карта снята галочкой в Settings — иначе открытая сессия
// молча переехала бы на другую карту. Возвращает true, если выбор изменился.
bool colormap_combo(const char* label, int* id, float width) ;


// Опции тулбара цветовой шкалы над хитмапой.
struct HeatmapToolbarOpts {
    // Куда сохранить выбор colormap'а (per-config + save_session). Пусто —
    // выбор живёт только в самом view (Custom Basins).
    std::function<void(int)> persist_colormap;
    // Доп. элементы сразу после combo Colormap: LS exponent picker,
    // Basins "Renumber (spiral)".
    std::function<void()>    extras;
    // Доп. элементы в самом конце строки (FastSync Line width / Alpha).
    std::function<void()>    extras_tail;
    // Swap axes осмыслен только там, где оси взаимозаменяемы (2D-свип).
    bool show_swap = true;
};


// ЕДИНСТВЕННАЯ реализация тулбара «Colormap / Autoscale / vmin / vmax / Swap axes»
// (раньше был скопирован в 7 мест с разным составом). Состояние живёт в самом
// HeatmapView — вызывающему достаточно передать, куда персистить colormap.
// Возвращает true, если пользователь что-то изменил.
bool draw_heatmap_toolbar(HeatmapView& hv, const HeatmapToolbarOpts& o = {}) ;


// LS exponent picker: λ1..λN + "sum L_i" (sentinel -1, а не N, чтобы выбор не съезжал
// при смене N на следующем Run). Выбор живёт в HeatmapView, а не в конфиге: два окна с
// одной кривой не должны дёргать друг у друга индекс. on_pick — персистентность на
// стороне вызывающего. Раньше — две копии (draw_ls_plot и Custom-слот 2).
void draw_ls_exponent_picker(HeatmapView& hv, int n_exponents,
                                    const std::function<void(int)>& on_pick) ;


// Выбор отображаемой плоскости LS-2D по индексу экспоненты `k`:
//   k >= 0  — готовая плоскость из result_2d.values (без копирования),
//   k == -1 — "sum L_i": поэлементная сумма всех N плоскостей, считается ОДИН раз на
//             data_generation_2d и кэшируется в конфиге (кэш общий для всех окон с этой
//             кривой). Diverged-ячейки (flags<0) получают тот же sentinel 999.0, что
//             engine пишет в одиночные плоскости, — HeatmapView красит их тем же серым.
// gen мешает поколение чанка с индексом экспоненты, иначе переключение экспоненты не
// перезалило бы текстуру. Раньше — две копии по ~40 строк.
void ls_resolve_plane(LSCurveConfig& cact, int k,
                             const double*& plane, double& vmin, double& vmax, int& gen) ;


// Вкладка System: ввод системы, методы, генерация кода
void draw_system_tab(AppModel& model, const GuiCallbacks& cb) ;


// Вкладка Parameters: НУ, значения/диапазоны параметров, шаг
void draw_parameters_tab(AppModel& model) ;


// Library tab: two states — list (table of saved systems + note preview) and editor
// (System + Parameters sub-tabs with Save/Cancel). The editor edits
// AppModel::library_edit_buffer (a scratch AppModel), never the live model, so
// browsing/editing/Cancel never affects the system active in Parametric/Phase/Basins/FastSync.

// Trim leading/trailing ASCII whitespace.
[[nodiscard]] std::string trim_copy(const std::string& s) ;


// Кэш списка систем для вкладки Library.
// draw_library_list раньше КАЖДЫЙ КАДР обходил каталог (lib.list()) и читал с диска
// system.json каждой системы — только ради бейджа "[N custom]" и ширины колонки Name:
// при двух десятках систем и 60 FPS это больше тысячи разборов JSON в секунду. Извне
// файлы никто не правит при работающем приложении, поэтому хватает редкого опроса, а
// всё, что меняет библиотеку изнутри (Save, Delete, Duplicate, Rename), сбрасывает кэш
// немедленно — иначе удалённая система висела бы в таблице до десяти секунд.
// Сюда же переехал кэш превью заметки: он жил отдельными inline'ами и не сбрасывался
// после сохранения, поэтому отредактированная заметка показывалась старой.
struct LibraryListCache {
    std::vector<std::string> names;
    std::vector<int>         custom_counts;   // параллелен names
    std::string              note_name;       // для какой системы прочитана заметка
    std::string              note_text;
    double                   filled_at = -1.0;
    bool                     valid     = false;
};

inline LibraryListCache g_library_cache;


void invalidate_library_cache() ;


const LibraryListCache& library_list_cached(SystemLibrary& lib) ;


// Save-side logic for the editor: validate name, handle rename for EditExisting, persist
// the scratch buffer via SystemLibrary. Mirrors the save into the live model only if the
// edited system is the one currently active elsewhere — saving any other system must not
// disturb what Parametric/Phase/Basins/FastSync have loaded. On success returns to list
// mode; on failure sets model.edit_error and stays in the editor.
void library_editor_save(AppModel& model, SystemLibrary& lib) ;


// List state: table of saved systems + Add-new + note preview panel.
// Interacting with this view (row select, Edit, Add new) only ever touches
// model.library_edit_buffer — never the live model — so it never changes
// the system currently active in Parametric/Phase/Basins/FastSync.
void draw_library_list(AppModel& model, SystemLibrary& lib) ;


// Editor state: name/note header + System/Parameters sub-tabs + Save/Cancel.
// Edits go into model.library_edit_buffer, not model itself — Cancel simply
// drops the edit_mode without needing to restore anything on the live model.
void draw_library_editor(AppModel& model, SystemLibrary& lib,
                                const GuiCallbacks& cb) ;


// РЕЖИМ АНАЛИЗА: пространство фазовых портретов

// Панель настроек сессии: параметры (общие), НУ (список), проекции (список), время/шаг,
// метод (заглушка), кнопка пересчёта. Reset lambda — что делает «Reset to defaults»:
// Analysis-tab передаёт model.from_record + start_phase_analysis, Custom-tab nullptr
// (там перезагружаются из System tab или Run pipeline).
void draw_phase_controls(PhaseAnalysisSession& s,
                                std::function<void()> on_reset_defaults) ;


// Единый счётчик режимов для статусных строк всех панелей. Раньше каждая панель считала
// "diverged" по flags[k] < 0, и подпись значила разное: у Bif1D/Bif2D отрицательный флаг —
// это fixed point, а у LLE/LS — расходимость. Теперь коды канонические (REGIME_* в
// configCUDA.h) и категории раскладываются одинаково везде.
void count_regimes(const std::vector<int>& flags, int& fp, int& unb, int& osc) ;


// Подпись под "OK: ..." — печатается только если есть что показать кроме
// колебательного режима (иначе строка была бы шумом на каждом успешном run'е).
void draw_regime_summary(const std::vector<int>& flags) ;


// Рисует окна проекций (каждая — отдельное docking-окно с графиком). Optional `before_begin`
// runs immediately before each projection window's ImGui::Begin (Custom mode uses it to assign
// the initial dock target); optional `after_begin` runs right after Begin returns true (Custom
// attaches the Move-to-Tab context menu there). Both empty by default = Analysis-mode behaviour.
using ProjHookFn = std::function<void(int proj_index, const std::string& title)>;


// Per-IC стиль серии (используется фазовыми портретами по бассейнам): цвет
// берётся из цвета бассейна на хитмапе, а бассейны-равновесия рисуются одним
// маркером вместо линии. Возвращает false — серия рисуется как раньше
// (ic_base_color + линия), поэтому Analysis/Custom ничего не замечают.
struct PhaseSeriesStyle {
    ImVec4 color      = ImVec4(1, 1, 1, 1);
    bool   as_point   = false;   // только Phase2D; в time domain всегда линия
    float  point_size = 6.0f;
    int    marker     = (int)PointMarker::Circle;
};

using PhaseStyleFn = std::function<bool(int ic_index, PhaseSeriesStyle& out)>;


// Подпись НУ для легенды (галка "Legend shows initial conditions"). Берём ЧИСЛА, реально
// ушедшие в расчёт (result.snapshot.ic_flat), и форматируем через fmt_tick, т.е. с Tick
// precision из Settings: так сгенерированные НУ (узлы сетки бассейнов пишутся с %.17g) не
// разносят легенду на пол-экрана, а смена точности применяется сразу, без пересчёта.
// Fallback на сохранённый result.ic_text — если снапшота нет.
[[nodiscard]] std::string ic_legend_text(const AnalysisResult& res, size_t k) ;

// `title_suffix` is appended to every projection window's title (e.g. "##sys_<name>") so
// imgui.ini stores dock state per system in Custom mode. `owner_id_delta` is XOR'd into the
// per-projection owner_id passed to Plot2D/3D so the SHARED PlotRenderer cache (inline in this
// function) is keyed per system — without it Chen and Rossler both used owner_id=0 for their
// first projection and Rossler saw Chen's cached FBO. Analysis mode passes zero.
//
// Параметры кластеризации для диаграммы признаков — те же множители осей и eps, с которыми
// работает dbscan (cudaLibrary.cu): множители применяются только внутри ядра, и без них
// диаграмма рисовала сырые (пик; IPI), по которым нельзя предсказать результат — при
// mult interval = 0 ядро видит одномерную задачу, а на картинке оставалось двумерное облако.
// valid=false — вызывающий не связан ни с каким DBSCAN-конфигом (вкладка Phase analysis):
// тогда рисуем сырые признаки и без окружности.
struct FeatureClusterParams {
    bool   valid         = false;
    double mult_peak     = 1.0;
    double mult_interval = 1.0;
    double eps           = 0.0;
};


void draw_projection_windows(PhaseAnalysisSession& s, const GuiCallbacks& cb,
                                    const ProjHookFn& before_begin = {},
                                    const ProjHookFn& after_begin  = {},
                                    const std::string& title_suffix = {},
                                    int owner_id_delta = 0,
                                    const PhaseStyleFn& style_fn = {},
                                    const FeatureClusterParams& clust = {}) ;


// Combo "Writable var": переменные + (через разделитель) комбинация
// x[0] + pi*x[1] + euler*x[2] (при |vars|=2 без e*x[2], при |vars|=1 — сама переменная).
// Sentinel в hostside int: -1 = комбинация. Используется в Bifurcation и Basins, где
// kernel пишет data[i] через loopCalculateDiscreteModel_int — он умеет оба режима.
void draw_writable_var_combo(const std::vector<std::string>& vars,
                                    int& writable_var,
                                    const char* combo_id) ;


// Parametric: контролы + scatter-plot 1D-бифуркации через наш GL-renderer
// Рисует контролы одной БД внутри её таба. Возвращает true, если пользователь
// нажал Run для этой БД (внешний код может также взвести Run через Ctrl+R).
void draw_diagram_controls(BifurcationAnalysisSession& s, int idx) ;


void draw_bifurcation_controls(AppModel& model, SystemLibrary& /*lib*/) ;


// Маппинг window-kind → нейтральный ParamPlotKind. Сама конфигурация живёт в
// configure_param_plot_view (см. блок общих хелперов вверху файла), чтобы
// Parametric и Custom настраивали одинаковые 1D-графики одним и тем же кодом.
ParamPlotKind to_param_plot_kind(ParametricPlotWindow::Kind kind) ;


// One-time Plot2DView setup per kind — called by draw_parametric_plot_windows
// when it lazily constructs a fresh Plot2DView for a newly opened window
// (Plot2DView is now owned per-window there, not a function-shared by
// every window of a kind, so this replaces the old `if (!view) {...}` inits).
void configure_plot_view(Plot2DView& view, ParametricPlotWindow::Kind kind) ;


// Plots one Parametric plot window of kind Bifurcation. `win` supplies
// mode_2d and the diagram indices to show (`members`); `renderer`/`view`/
// `heatmap_map` are this window's own render-state instances, owned and
// cached by draw_parametric_plot_windows (keyed by win.id).
void draw_bifurcation_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                                   ParametricPlotWindow& win,
                                   PlotRenderer& renderer, Plot2DView& view,
                                   std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) ;


// LLE: контролы (per-curve в табе) + line-plot λ(param)

// Контролы одной LLE-кривой. Возвращает true, если пользователь нажал Run
// для этой кривой.
void draw_lle_curve_controls(LLEAnalysisSession& s, int idx) ;


// Tab bar по LLE-кривым + кнопка «+» (копирует последнюю).
void draw_lle_controls(AppModel& model, SystemLibrary& /*lib*/) ;


// Plot LLE: линии (points_mode=false), каждая кривая — λ(param). При mode_2d=true у
// активной кривой вместо линий рисуется HeatmapView (сетка λ(p1, p2) с colormap'ом).
// Per-window-state shape — как у draw_bifurcation_plot (обоснование см. там).
void draw_lle_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                          const ParametricPlotWindow& win,
                          PlotRenderer& renderer, Plot2DView& view,
                          std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) ;


// LS: контролы (per-curve в табе) + line-plot λ_k(param), N экспонент
// на один спектр-«прогон». UX зеркало LLE.

void draw_ls_curve_controls(LyapunovSpectrumAnalysisSession& s, int idx) ;


void draw_ls_controls(AppModel& model, SystemLibrary& /*lib*/) ;


// Plot LS: каждая кривая раскладывается на N линий по числу экспонент, серия =
// spectrum_idx * N + exponent_idx, цвета через ic_base_color(seq). При mode_2d=true у
// активной кривой рисуется HeatmapView с одной выбранной экспонентой; combo "Exponent"
// над хитмапой переключает плоскость без повторного Run.
// Per-window-state shape — как у draw_bifurcation_plot (обоснование см. там).
void draw_ls_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                         const ParametricPlotWindow& win,
                         PlotRenderer& renderer, Plot2DView& view,
                         std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) ;


// Parametric plot windows — shared setup-row helper. The row (Label | Type | Members... | X)
// is rendered both in the Plot windows manager on the settings panel and at the top of each
// per-window Begin(), so users can retype the chart without going back to the panel; the
// helpers live at file scope so both callsites share one implementation.
struct ParamPlotMatchItem { int index; std::string label; };


[[nodiscard]] std::vector<ParamPlotMatchItem>
parametric_matching_items(AppModel& model, ParametricPlotWindow::Kind kind, bool mode_2d) ;


// Colored 1D is not a distinct combo entry — it's a toggle inside the
// plot body (see draw_bifurcation_plot). The Type combo only splits 1D/2D.
inline const char* kParametricTypeNames[] = {
    "Bifurcation 1D", "Bifurcation 2D", "LLE 1D", "LLE 2D", "LS 1D", "LS 2D"
};


int parametric_type_index_of(ParametricPlotWindow::Kind kind, bool mode_2d) ;


void parametric_type_from_index(int t, ParametricPlotWindow::Kind& kind, bool& mode_2d) ;


// Draws the Label | Type | Members... | X row for a Parametric plot window.
// Returns true iff the user clicked the row's X — caller unmounts the window.
// Assumes caller has already pushed a PushID(win.id) on the ID stack.
bool draw_parametric_window_setup_row(AppModel& model, ParametricPlotWindow& win) ;


// Parametric mode: dynamic plot windows (mirrors draw_projection_windows in Phase). One ImGui
// window per model.parametric_plot_windows entry, each with its own
// PlotRenderer/Plot2DView/HeatmapView-map keyed by ParametricPlotWindow::id — not vector
// position, which shifts when an earlier window is removed. Closing a window (X) removes it;
// "Reset windows layout" bumps model.parametric_layout_generation, baked into every title, so
// ImGui treats them as brand-new windows and re-cascades default positions.
void draw_parametric_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) ;


// 1D DFT: controls (system picker + Run/Run all + config tab bar, mirrors
// draw_basins_controls) + dynamic Plot windows (mirrors draw_parametric_
// controls' manager + draw_bifurcation_plot's colored_1d heatmap toolbar).

void draw_dft1d_diagram_controls(Dft1DAnalysisSession& s, int idx) ;


void draw_dft1d_controls(AppModel& model, SystemLibrary& lib) ;


// Per-window heatmap rendering — mirrors the colored_1d block of
// draw_bifurcation_plot (toolbar + lazy display-cache rebuild + render()) and
// draw_basins_plot's colormap/autoscale/swap-axes toolbar.
void draw_dft1d_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                           Dft1DPlotWindow& win,
                           PlotRenderer& renderer,
                           std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) ;


void draw_dft1d_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) ;


// Basins of attraction: controls + 5-plot window (inner tab-bar).

// Basins: общие хелперы карты + панель фазовых портретов по бассейнам.

// Диапазон colorbar'а таба «Basins» — единственный источник истины и для хитмапы, и для цвета
// траекторий в фазовых портретах (иначе цвет бассейна на карте и его траектории разъезжаются):
//   min_cluster_id..-1 — FP-кластеры (если есть отрицательные)
//   0                  — расходящиеся ячейки (helpful_array[i] == 0)
//   1..n_clusters      — осциллирующие кластеры
// Когда FP-кластеров нет и ничего не разошлось, «кластера 0» не существует — сдвигаем vmin к 1,
// чтобы в colorbar'е не было фантомной полосы.
void basins_colorbar_range(const BasinsConfig& c, double& vmin, double& vmax) ;


// HeatmapView таба «Basins» каждого config'а. Раньше жил function-local inline
// внутри draw_basins_plot; поднят на уровень файла, потому что цвет траекторий
// в окнах фазовых портретов обязан читать ЖИВЫЕ discrete/reverse/colormap
// именно этого вью — иначе цвета совпадали бы только при дефолтных настройках.
static auto& basins_heatmaps() { return ui_resource<std::map<unsigned, std::unique_ptr<HeatmapView>>, 1000>(); }


// owner_id блока плотов одного basins-config'а (5 внутренних табов на config).
[[nodiscard]] unsigned basins_base_oid(int config_index) ;


// Цвет бассейна `display_id` — ровно тот, которым его красит хитмапа.
// Повторяет квантование из FS_HEATMAP (plot_renderer.cpp): t по [vmin,vmax],
// дискретизация по полосам с edge-aligned сэмплированием, затем reverse.
ImVec4 basins_id_color(const BasinsConfig& c, const AppModel& model,
                              int config_index, int display_id) ;


// Панель «Phase portraits» внутри Basins Controls. Интегратор и параметры —
// из самого basins-config'а; своё тут только время моделирования, прореживание,
// ограничитель числа аттракторов и выбор CPU/GPU.
void draw_basins_phase_controls(AppModel& model, int cfg_idx) ;


void draw_basins_controls(AppModel& model, SystemLibrary& lib) ;


// Plot Basins: inner tab-bar по 5 представлениям. Heatmap-views — per (config × tab) в
// std::map по owner_id: HeatmapView/Plot2DView хранят data_gen_cached внутри без учёта
// owner_id, поэтому один view нельзя переиспользовать между configs (после Run All у двух
// configs одинаковый data_generation=1 → кэш не инвалидируется и на чужой вкладке виден
// предыдущий буфер). Map с lazy-init решает это и сохраняет независимый zoom/pan per (config, tab).
// spiral_coords_from_center / renumber_basins_spiral / ensure_basins_spiral_cache переехали в
// analysis_session.cpp (объявления в analysis_session.h): перенумерация нужна не только для
// отрисовки, но и для выбора представительных ячеек в фазовых портретах
// (BasinsAnalysisSession::rebuild_phase_ics), а тот слой про GUI ничего не знает.

void draw_basins_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) ;


// Окна фазовых портретов по бассейнам — только для АКТИВНОГО config'а
// (панель хитмапы показывает его же). Набор проекций и результат у каждого
// config'а свои, поэтому переключение вкладки сразу показывает свои окна с
// уже посчитанными данными.
void draw_basins_phase_windows(AppModel& model, const GuiCallbacks& cb) ;


// Раз в кадр: пометить устаревшие портреты и, если включён Autorun, поставить их в
// очередь. Дебаунс 300 мс — иначе каждое нажатие клавиши в текстовом поле стартовало бы
// новый GPU-прогон. Проходим по ВСЕМ config'ам: после «Run all…» автозапуск должен
// подхватить каждый пересчитанный конфиг, а не только открытый.


// Fast Synchro Controls + Plot — recurrent synchronization (anti-sync).
// Mode 0 = On Attractor (trajectory + per-point error); Mode 1 = On Grid.
void draw_fastsync_controls(AppModel& model, SystemLibrary& lib) ;


// Plot Fast Synchro: либо colored trajectory (mode 0), либо heatmap (mode 1).
void draw_fastsync_plot(AppModel& model, const GuiCallbacks& cb) ;


// Parametric Controls dispatcher — верхние табы Bif / LLE / LS.
// ===========================================================================
// Order — оценка порядка точности схемы.
//
// Расчёт: kernels/order.template.cu, три решения шагами h, h/2, h/4 и
// p = log2(E1/E2) по Эйткену/Ричардсону (см. order_session.h).
// Три диаграммы вкладки различаются только выбором осей, поэтому UI один.

// Паспортный порядок встроенной схемы (0 = не встроенная / неизвестно).
int builtin_scheme_order(const std::string& name) ;


// Селектор мишени свипа: единый список "h + все элементы a[]". Параметры
// метода отдельной сущностью не являются — они объявляются обычными
// параметрами системы и читаются телом кастомной КРС как a[k], поэтому и
// попадают сюда наравне с остальными.
void draw_order_target_combo(const char* label, int& target,
                                    const OrderAnalysisSession& s) ;


// Блок настроек одной оси. Возвращает ничего: всё пишется прямо в конфиг.
void draw_order_axis_block(const char* id, const char* title,
                                  const OrderAnalysisSession& s,
                                  int& target, std::string& lo, std::string& hi,
                                  bool& log_grid, std::string& n_text,
                                  bool enabled) ;


// Индикатор числа шагов + предупреждение о нецелом t_max/h.
//
// Почему это не косметика: без snap число грубых шагов = floor(t_max/h), и
// фактическое конечное время N*h отличается от t_max на величину до h. Оно
// разное у соседних узлов сетки, так что на кривую ошибки садится вклад
// порядка O(h) — для схем порядка выше первого он полностью забивает
// измеряемую величину. На лог-сетке по h нецелым оказывается почти каждый
// узел, поэтому счётчик показывает, сколько именно узлов задеты.
void draw_order_steps_hint(const OrderConfig& c, const OrderAnalysisSession& s) ;


void draw_order_controls(AppModel& model, SystemLibrary& /*lib*/) ;


void draw_order_plot(AppModel& model, const GuiCallbacks& /*cb*/) ;


void draw_parametric_controls(AppModel& model, SystemLibrary& lib) ;


// Атомарное применение сохранённой сессии.
//
// Парсеры session_from_json_* пишут поля ПО МЕРЕ чтения JSON и бросают на первой же
// структурной ошибке, поэтому на битом _last_*.json (обрезанном при падении, правленом
// руками) сессия оставалась перезаписанной наполовину: часть полей из файла, часть
// сидированная из записи системы. Возвращаемый bool при этом игнорировался во ВСЕХ точках
// вызова, и пользователь видел молча испорченные настройки.
//
// Здесь разбор идёт в два прохода: сначала в отдельный пустой объект — только чтобы
// убедиться, что файл дочитывается целиком, — и лишь потом в настоящую сессию. Второй
// проход не может упасть там, где прошёл первый: парсеры не зависят от прежнего состояния
// приёмника. Файлы сессий небольшие, двойной разбор незаметен.
// Пустой json — не ошибка: сессии просто нет (первый запуск / новая система).
template <class Session>
inline bool apply_session_json(AppModel& model,
                               const std::string& json,
                               Session& dst,
                               bool (*parse)(const std::string&, Session&),
                               const char* what)
{
    if (json.empty()) return true;
    Session probe;
    if (!parse(json, probe)) {
        model.session_load_warning =
            std::string("saved session '") + what + "' is corrupt - ignored";
        std::printf("[session] %s: parse failed, keeping defaults from record\n", what);
        std::fflush(stdout);
        return false;
    }
    if (!parse(json, dst)) {                 // не должно случаться, см. выше
        model.session_load_warning =
            std::string("saved session '") + what + "' failed to apply";
        return false;
    }
    model.session_load_warning.clear();
    return true;
}


// Global system switch — fired from the top-bar combo. Loads the record and re-inits the
// CURRENT tab (mirrors what each per-tab combo used to do); other tabs re-init on entry via
// the block in draw_gui. Non-static: also called from app_main.cpp on startup to restore the
// last-used system (see gui.h::apply_system_switch).



// Custom tab (master-detail pipeline) — see custom_session.h


// Комбо выбора цели свипа и разбор числовых полей — общие для всего файла
// (draw_sweep_target_combo / parse_ratio_or, см. верх файла). Раньше здесь
// лежали их локальные копии: комбо со своим стилем подписей ("IC x" вместо
// "x (IC)") и обёртка parse_num_default поверх того же парсера.

// Shared config panel

void draw_shared_config(CustomSession& cs,
                        const std::vector<CustomScheme>& custom_schemes) ;


// Level 2D detail

void draw_level2d_detail(CustomSession& cs) ;


// Level 1D detail

void draw_level1d_detail(CustomSession& cs) ;


// Level 3 detail

void draw_level3_detail(CustomSession& cs) ;


// Pipeline column (master)

// draw_pipeline_column / level_status_badge жили здесь и не вызывались ни
// разу: колонка со списком уровней была заменена на level_header внутри
// draw_custom_controls, а вся её логика статусов переписана там заново.
// Удалены как мёртвый код.

 // namespace (draw helpers)

void draw_custom_controls(AppModel& model, SystemLibrary& lib) ;


// Custom-tab plot windows: minimal renderer for the pipeline output. Full-featured versions
// with colormap toolbars, colorbar controls and colored-1D density are TODO; this MVP shows
// results and wires drill-down clicks / crosshairs.


// Helper: bind slider fix_x/fix_y as crosshair on the given heatmap view,
// and install a drill-down click callback that updates shared.param_values
// + fix_x/y and optionally enqueues a Phase/Basins run.
void wire_2d_heatmap_interaction(HeatmapView& hv, CustomSession& cs,
                                 std::deque<CustomQueueItem>& q) ;


 // namespace

// Custom Workspace — split-region layout for Custom AppMode.
//
// Per frame: AppMode radios on top (MainHost); below, Custom Controls panel | splitter |
// tab bar [ Tab 1 | Tab 2 | + ] over a per-tab DockSpace that hosts the plot windows.
//
// Docking model:
//   - Each tab owns one DockSpace with a STABLE id from tab.id via ws_dock_id() — deliberately
//     NOT ImGui::GetID(str), which hashes with the current window-ID stack and would differ
//     inside BeginTabItem vs outside; we submit the same dockspace both ways (KeepAliveOnly
//     before BeginTabBar, real inside BeginTabItem of the active tab).
//   - ALL per-tab dockspaces are submitted every frame with ImGuiDockNodeFlags_KeepAliveOnly so
//     ImGui keeps the nodes alive and windows docked in inactive tabs don't orphan to floating.
//   - Plot windows are docked via DockBuilderDockWindow on first appearance; imgui.ini persists
//     the internal split layout across sessions.
//   - Cross-tab drag&drop: BeginDragDropSource on each plot title bar + BeginDragDropTarget on
//     each tab item → drop calls DockBuilderDockWindow(name, target_ds) and switches tab.

// Stable dockspace ID for a workspace tab: independent of ImGui's ID-stack scope (same id
// inside or outside a window) and unlikely to collide with other IDs. Includes a hash of the
// loaded system name so imgui.ini keeps a separate dock layout per system — otherwise switching
// systems leaked one system's plot placement into another's tabs.
[[nodiscard]] ImGuiID ws_dock_id(int tab_id, const std::string& sys) ;


// Suffix appended to every Custom-mode plot window title so imgui.ini keys
// its dock state per system. Empty when no system is loaded (edge case;
// windows behave as before). "##sys_<name>" — the "##" makes ImGui strip
// the suffix from the visible title while including it in the hashed ID.
[[nodiscard]] std::string custom_win_suffix(const std::string& sys) ;


// Drag&drop payload key — passed via ImGui's built-in drag-drop channel.
// Payload data: `const char* name` of the docked window title.
inline const char* WS_DRAG_PAYLOAD = "CUSTOM_WS_WIN";


// Forward decls — the layout function calls these; they live further down.
inline void draw_custom_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb);


void draw_custom_controls_panel(AppModel& model, SystemLibrary& lib,
                                       ImVec2 pos, ImVec2 size) ;


// Runtime rename state — Custom mode is the only place we rename tabs, so
// keeping it as a file-scope inline (not persisted) is fine.
struct WsRenameState {
    int  target_id = 0;              // 0 = no rename in progress.
    char buf[128]  = {};
};

inline WsRenameState g_ws_rename;


void draw_custom_workspace_panel(CustomSession& cs,
                                        const std::string& sys,
                                        ImVec2 pos, ImVec2 size) ;


void draw_custom_mode_layout(AppModel& model, SystemLibrary& lib,
                                    const GuiCallbacks& cb,
                                    ImVec2 area_pos, ImVec2 area_size) ;


void draw_custom_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) ;


// Индикатор компьюта в верхней строке. Раньше состояние снималось лестницей из двенадцати
// почти одинаковых веток: каждая доставала из своей сессии одну и ту же четвёрку — подпись
// запущенного конфига, время старта, cancel_token, progress_token. Ветки успели разойтись:
// три подставляли запасное имя только при выходе индекса за границы, три — ещё и при пустом label.
enum class BusyKind { None, Bif, LLE, LS, Dft1D, Basins, Phase, FastSync, Custom, Order };


struct BusyInfo {
    BusyKind    kind = BusyKind::None;
    std::string what;
    std::chrono::steady_clock::time_point start{};
    bool  cancelling = false;   // пользователь нажал Stop, ждём движок
    float progress   = 0.0f;    // 0..1 из session.progress_token
};


// Подпись запущенного конфига: label под running-индексом, иначе запасное имя.
// Пустой label тоже уходит в запасное — так делали три ветки из шести, а в
// остальных пустое имя дало бы пустую строку состояния.
template <class Vec>
[[nodiscard]] inline std::string running_label(const Vec& items, int idx, const char* fallback) {
    if (idx >= 0 && idx < (int)items.size() && !items[idx].label.empty())
        return items[idx].label;
    return fallback;
}


// Снять состояние с сессии. Годится для любой, у которой есть
// compute_start_time / cancel_token / progress_token — то есть для всех, кроме
// phase: у неё нет ни отмены, ни прогресса, и её ветка заполняется вручную.
template <class Session>
[[nodiscard]] inline BusyInfo busy_from(const Session& s, BusyKind kind, std::string what) {
    BusyInfo b;
    b.kind  = kind;
    b.what  = std::move(what);
    b.start = s.compute_start_time;
    b.cancelling = s.cancel_token && s.cancel_token->load(std::memory_order_relaxed);
    if (s.progress_token)
        b.progress = s.progress_token->load(std::memory_order_relaxed);
    return b;
}
}
