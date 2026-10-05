#pragma once
#include "ucuda/ui/gui.h"
#include "ui_resources.h"
#include "ucuda/platform/platform_paths.h"
#include "ucuda/numerics/circuit_netlist.h"
#include <fstream>
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
#include "ucuda/cuda/nvrtc_engine.h"   // nvrtc_check_ctrl_body — Check в библиотеке регуляторов
#include "ucuda/domain/num_parse.h"   // parse_num — единый разбор числовых полей
#include "ucuda/plotting/digit_input.h" // DigitInput::ComputeStep — ↑/↓ по значению в списке (atol)
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
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


// Полная раскладка коэффициентов схемы-обёртки (Extr / Comp). Определение
// стоит рядом с конструкторами обёрток, а объявление здесь: тот же текст
// показывает комбо выбора схемы, а оно в файле раньше.
std::string wrapper_coefficients_text(
        const std::string& name,
        const std::vector<CustomScheme>& custom_schemes,
        const std::map<std::string, std::string>& param_values);


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
              "pi in num_parse.h diverged from configCUDA.h");


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


// Превью LaTeX-поля: те же уравнения, набранные верстальщиком подписей графиков
// (math_formula_draw в plot_axis.cpp). Уравнения делятся по `\\` и переводам
// строки; строка, от которой после разметки (\begin{cases} и т.п.) ничего не
// осталось, пропускается. Состояния нет — всё считается из текста каждый кадр.
void draw_latex_preview(const char* id, const std::string& src, ImVec2 size) ;


// Правые части в том виде, в каком их получает кодогенератор: над X[..] и
// a[..], с легендой имён. Это вход ВСЕХ схем (codegen_rhs), поэтому здесь
// видно и порядок параметров в a[], и ошибку разбора — до Generate/Run.
// Поле только для чтения, но выделяется и копируется; высота — по строкам.
void draw_rhs_preview(const AppModel& model) ;

bool InputTextStr(const char* label, std::string& str, float width = 0.0f) ;


// То же с подсказкой в пустом поле. Нужна там, где пустое поле означает не
// "ничего", а "значение по умолчанию": имя схемы-обёртки пустым берётся из
// её же описания, и подсказка показывает, какое имя получится.
bool InputTextStrHint(const char* label, std::string& str, const char* hint,
                             float width = 0.0f) ;


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


// Требует ли схема КОМПИЛЯЦИИ тела, а не интерпретации AST. Таких две группы:
// кастомные КРС (сырой C) и экстраполяционные обёртки (тело базы, размноженное
// по стадиям внутри лямбды — интерпретатору выражений это не по зубам).
// На GPU разницы нет, обе идут через NVRTC; на CPU обеим нужен внешний cl.exe.
// Тот же предикат решает krs_is_custom при сборке Request — см. analysis_session.cpp.
[[nodiscard]] bool scheme_needs_compiled_body(const std::string& scheme_name,
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


// Устойчивая часть подписи — всё до " [", то есть имя оси без диапазона.
// Нужна отмене (ViewRangeTarget::relocate), которая опознаёт конфиг по подписи.
// Автоподпись 1D-диаграммы — это "<ось> [lo..hi]" (sweep_range_label выше), а
// диапазон в ней переписывает ровно та команда, которую потом откатывают: сверять
// подпись целиком значило сверять её с тем, что сам же и изменил, и сверка
// проваливалась всегда. У 2D автоподпись "<x> x <y>" скобок не содержит, у ручных
// имён их обычно нет — там функция возвращает строку как есть, то есть поведение
// прежнее. Отсюда и симптом «Ctrl+Z не работает именно в 1D».
[[nodiscard]] std::string label_identity(const std::string& s) ;


// Автоподпись BD / LLE / LS: "<x> x <y>" в 2D-режиме, "<x> [lo..hi]" в 1D.
// Шаблон, а не общий базовый класс: конфиги независимы, связывать их
// наследованием ради одинакового набора полей было бы хуже. Раньше — три
// побайтово одинаковых тела (auto_label_bd / _lle / _ls).
template <class Cfg>
[[nodiscard]] std::string auto_label_sweep(const Cfg& c,
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
[[nodiscard]] double parse_ratio_or(const std::string& v, double def) ;


// Целочисленные поля: Resolution, decimator, число бинов, N сетки.
// Тело переехало в num_parse.h к parse_num_int — там же объяснено, почему
// вторая копия этого разбора (на std::stoi, в analysis_session.cpp) молча
// врала на "512/2". Имя сохранено: оно стоит по месту в gui.cpp.
[[nodiscard]] int parse_int_or(const std::string& v, int def) ;


// Распарсилось ли поле вообще. parse_num молча отдаёт default, и вызывающий не
// отличает «поле пустое» от «в поле ноль»; пробуем два разных дефолта —
// совпали, значит число прочиталось.
[[nodiscard]] bool parse_num_ok(const std::string& v, double& out) ;


// ctx_menu — контекстное меню поля (ПКМ). Вызывается СРАЗУ после InputText:
// ниже может встать предупреждение о невалидном числе, и BeginPopupContextItem
// внутри колбэка привязался бы к нему, а не к самому полю.
// allow_empty: пустая строка — осмысленное значение («как у всех», см. таблицу
// узлов вкладки Network), и ругаться на неё как на нечисло нельзя.
bool InputNumStr(const char* label, std::string& str, float width = 0.0f,
                        const std::function<void()>& ctx_menu = {},
                        bool allow_empty = false) ;


// ↑/↓ в поле-списке чисел через ',' или ';' (atol): шагает цифру того значения, в котором
// стоит каретка, как digit_step_input_callback у одиночного числа. Пробелы вокруг значения
// не трогаются. Запятая здесь разделитель, поэтому в точку не превращается.
int list_digit_step_callback(ImGuiInputTextCallbackData* data) ;


// Поле-список чисел (atol: одно на все переменные или по одному через ',' / ';') со
// стрелками ↑/↓ по значению под кареткой.
bool InputNumListStr(const char* label, std::string& str, float width = 0.0f) ;


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

// Откат последней массовой команды + отметка времени для подписи в шапке
// (модель про ImGui не знает, поэтому GetTime проставляется здесь).
void do_undo(AppModel& m) ;


void do_redo(AppModel& m) ;


// Пункты "Undo/Redo: ..." в контекстных меню. Дублируют хоткеи сознательно: те
// работают только вне поля ввода (внутри InputText своя отмена у ImGui), и без
// видимых пунктов это выглядело бы как "иногда работает". Стек свой у каждой
// вкладки, поэтому показывается ровно то, что откатится ЗДЕСЬ.
void draw_undo_menu_item(AppModel* m) ;


// ---------------------------------------------------------------------------
// ПКМ по числовому полю (и по комбо схемы / цели свипа) настроек.
// Три адресата на выбор, от узкого к широкому:
//   * все конфиги ТОГО ЖЕ типа, что и правящийся ("все Bifurcation 1D") —
//     обычно нужен именно он: диаграммы одного типа сравнивают между собой, и
//     разъехавшийся у одной из них шаг или время счёта делает сравнение ложью;
//   * отмеченные галочками в подменю — когда типа мало или он слишком широк;
//   * активный конфиг каждой вкладки — прежнее поведение, оставлено как есть.
// Во всех трёх случаях команда уходит в стек отмены одной записью.
void field_apply_all_menu(AppModel* model, BroadcastField f,
                                 const std::string& name, const std::string& label,
                                 const std::string& value) ;


// Встроенные схемы интегрирования. Один список на весь файл: раньше этот же
// массив был выписан восемью копиями плюс девятой — для проверки конфликта
// имён пользовательских схем.
// order — порядок точности: по нему схемы сгруппированы и в комбо, и в
// чекбоксах System tab, поэтому таблица отсортирована по нему и порядок строк
// здесь = порядок в UI. Исключение — схемы со своей группой (group, сейчас
// только ГБШ): они идут блоком в конце, со своим заголовком.
// У CD, Complex CD, SEMP и SIMP заявленный порядок достигается только при
// s = a[0] = 0.5 (см. codegen.cpp).
struct BuiltinScheme {
    const char* name;
    int         order;
    // Флаг "генерировать эту схему" в AppModel. Через него System tab рисует
    // чекбоксы циклом — раньше список схем был выписан там ещё раз руками, и
    // Complex Implicit Euler, добавленный только в одну из копий, не попадал
    // ни под "Select all", ни под "Clear all".
    bool AppModel::*flag;
    // Подсказка чекбокса; nullptr — схема говорит сама за себя.
    const char* tooltip;
    // Своя группа вместо "Order N" (nullptr — группа по порядку). Схемы одной
    // группы стоят в таблице подряд: шапка печатается на смене заголовка.
    const char* group = nullptr;
};


// Заголовок группы в комбо и чекбоксах: своя группа или "Order N".
[[nodiscard]] std::string scheme_group_title(const BuiltinScheme& b) ;

// Scheme name for a discrete map. Deliberately NOT in kBuiltinSchemes: it is
// never picked from the combo, but it is reserved as a custom-KRS name.
inline const char* const kMapSchemeName = "Map";


// "Is a map" predicate for UI code with no AppModel at hand: load_from_record
// pins scheme to "Map" for a map system, and that name is barred for custom KRS.
[[nodiscard]] bool scheme_is_map(const std::string& name) ;


inline const BuiltinScheme kBuiltinSchemes[] = {
    { "Euler",                    1, &AppModel::scheme_euler, nullptr },
    { "Euler-Cromer",             1, &AppModel::scheme_cromer, nullptr },
    { "D",                        1, &AppModel::scheme_dmethod,
      "Diagonally-implicit first-order method: the stage of SIMP\n"
      "(and the Phi* half-step of CD) taken as a step of its own,\n"
      "on the full h. X[i] = X + h*f_i(X) is solved for its own\n"
      "variable (analytically when linear in it, 4 fixed-point\n"
      "iterations otherwise), forward order, so each equation\n"
      "already sees the new X[0..i-1].\n"
      "The diagonally-implicit twin of Euler-Cromer: no Jacobian,\n"
      "no Newton, and cross terms stay outside the solve." },
    { "Implicit Euler",           1, &AppModel::scheme_ieuler,
      "X_next = X + h*f(X_next), solved by Newton on a\n"
      "symbolically differentiated Jacobian.\n"
      "Order 1, L-stable: unlike the explicit schemes it stays\n"
      "bounded on stiff systems at steps where RK4 diverges.\n"
      "Needs a differentiable RHS (no floor/ceil/fmod)." },
    { "Explicit Midpoint",        2, &AppModel::scheme_midpoint, nullptr },
    { "Implicit Midpoint",        2, &AppModel::scheme_imidpoint,
      "Solved for the stage Y = (X + X_next)/2:\n"
      "Y = X + (h/2)*f(Y), then X_next = 2*Y - X.\n"
      "Order 2 (measured 2.00 on Lorenz), A-stable, symmetric\n"
      "and symplectic. Same Newton solver as Implicit Euler." },
    { "CD",                       2, &AppModel::scheme_cd, nullptr },
    { "CD_{1-0}",                   2, &AppModel::scheme_cd10,
      "Variant {1-0} of CD: each CD pass runs the IMPLICIT\n"
      "half first (h1 = s*h, reverse order), then the explicit one\n"
      "(h2, forward order). The same composition over the adjoint\n"
      "basic method: order 2 and symmetry as in CD.\n"
      "Measured: Lorenz 1.03x, Rossler 1.6x larger error than CD.\n"
      "The plain name is the {0-1} variant (explicit half first)." },
    { "Complex CD",               2, &AppModel::scheme_ccd,
      "CD with complex half-steps:\n"
      "h1 = s*h + i*h*sqrt(3)/6, h2 = (1-s)*h - i*h*sqrt(3)/6,\n"
      "s = a[0] is the same symmetry coefficient as in CD.\n"
      "The step is evaluated in complex arithmetic; only Re is kept.\n"
      "At s = 0.5 (default) the half-steps are conjugate and order is 2." },
    { "Complex CD_{1-0}",           2, &AppModel::scheme_ccd10,
      "Variant {1-0} of Complex CD: each CD pass runs the IMPLICIT\n"
      "half first (h1 = s*h, reverse order), then the explicit one\n"
      "(h2, forward order). The same composition over the adjoint\n"
      "basic method: order 2 and symmetry as in Complex CD.\n"
      "Error about the same as Complex CD.\n"
      "The plain name is the {0-1} variant (explicit half first)." },
    { "Complex Implicit Euler",   2, &AppModel::scheme_cieuler,
      "Composition of TWO implicit Euler steps with conjugate\n"
      "complex steps: tau1 = h*(s + i/2), tau2 = h*(1 - s - i/2),\n"
      "s = a[0], the same symmetry slot as CD. Each half-step is a\n"
      "full Newton solve on the symbolic Jacobian, in complex\n"
      "arithmetic; Re is taken once at the end.\n"
      "The coefficients are forced, not tuned: composing a first-\n"
      "order method to second order needs tau1 + tau2 = h AND\n"
      "tau1^2 + tau2^2 = 0, whose only solution is (1 +- i)/2.\n"
      "Order 2 at s = 0.5 only; elsewhere the h^2 term keeps a real\n"
      "part that Re does not remove and the order drops to 1." },
    { "SEMP",                     2, &AppModel::scheme_semp,
      "Semi-explicit midpoint. Stage: X~[i] = X[i] + h1*f_i(X),\n"
      "component by component in forward order (each equation\n"
      "already sees the new X[0..i-1]), h1 = s*h. Corrector:\n"
      "X_next = X + h*f(X~). Two RHS passes, nothing to solve.\n"
      "Order 2 at s = a[0] = 0.5 only: the expansion carries\n"
      "s*h^2*F'F against h^2/2*F'F, so other s give order 1." },
    { "SIMP",                     2, &AppModel::scheme_simp,
      "Semi-implicit midpoint. Same corrector as SEMP, but the\n"
      "stage is diagonally implicit: each equation is solved for\n"
      "its own variable (analytically when linear in it, 4 fixed-\n"
      "point iterations otherwise), exactly as Phi* does in CD.\n"
      "No Jacobian, no Newton. Order 2 at s = a[0] = 0.5 only." },
    { "RK4",                      4, &AppModel::scheme_rk4, nullptr },
    { "Complex CD4",              4, &AppModel::scheme_ccd4,
      "Composition of TWO symmetric CDs with complex steps:\n"
      "pass 1 uses step gamma*h, pass 2 uses conj(gamma)*h,\n"
      "gamma = 1/2 + i*sqrt(3)/6. Re is taken once at the end.\n"
      "Order 4 (measured 4.00 on Lorenz and Rossler).\n"
      "Requires s = a[0] = 0.5: for other s the inner CD is\n"
      "asymmetric and the order drops to first." },
    { "Complex CD4_{1-0}",          4, &AppModel::scheme_ccd4_10,
      "Variant {1-0} of Complex CD4: each CD pass runs the IMPLICIT\n"
      "half first (h1 = s*h, reverse order), then the explicit one\n"
      "(h2, forward order). The same composition over the adjoint\n"
      "basic method: order 4 and symmetry as in Complex CD4.\n"
      "Measured: 4.0x smaller error on Lorenz, 1.5x on Rossler.\n"
      "The plain name is the {0-1} variant (explicit half first)." },
    { "CCD4 (o4s3)",              4, &AppModel::scheme_ccd4s3,
      "SYMMETRIC order-4 composition of THREE symmetric CDs:\n"
      "steps (alpha, 1-2*alpha, alpha)*h, palindromic, hence\n"
      "self-adjoint as a complex composition.\n"
      "alpha = 1/(2 - 2^(1/3)*exp(2*pi*i/3))\n"
      "      = 0.3243964040201712 + 0.1345862724908067i.\n"
      "Complex CD4 itself cannot be made symmetric: a palindrome\n"
      "of two stages forces g1 = g2, and then sum g = 1 and\n"
      "sum g^3 = 0 are incompatible -- three stages is the minimum.\n"
      "Same cubic as Yoshida's real triple jump, but the complex\n"
      "branch: no backward substep (min Re = 0.324) and an error\n"
      "constant 215x smaller (0.0247 against 5.29).\n"
      "Re is taken once at the end of the step; the symmetry defect\n"
      "that leaves is O(h^10), against O(h^8) for Complex CD4.\n"
      "Error expands in EVEN powers of h (measured: Richardson on\n"
      "h^2 gives order 6), so it is a legitimate extrapolation base.\n"
      "Requires s = a[0] = 0.5, like every CD-family scheme.\n"
      "Costs 3 CD passes against 2; at equal total work the error\n"
      "matches Complex CD4 to within 4%." },
    { "CCD4 (o4s3)_{1-0}",          4, &AppModel::scheme_ccd4s3_10,
      "Variant {1-0} of CCD4 (o4s3): each CD pass runs the IMPLICIT\n"
      "half first (h1 = s*h, reverse order), then the explicit one\n"
      "(h2, forward order). The same composition over the adjoint\n"
      "basic method: order 4 and symmetry as in CCD4 (o4s3).\n"
      "Measured: 3.8x smaller error on Lorenz, 2.4x on Rossler.\n"
      "The plain name is the {0-1} variant (explicit half first)." },
    { "CCD4 (o4s4)",              4, &AppModel::scheme_ccd4s4,
      "SYMMETRIC order-4 composition of FOUR symmetric CDs:\n"
      "steps (gamma/2, conj/2, conj/2, gamma/2)*h, palindromic.\n"
      "This is Complex CD4 symmetrized by composing it with its own\n"
      "adjoint on the half step; the palindromic four-stage order\n"
      "conditions have exactly this solution, up to conjugation.\n"
      "Smallest error constant of the three (0.00759), but 4 CD\n"
      "passes per step -- at equal total work it is identical to\n"
      "Complex CD4 and 4% better than CCD4 (o4s3).\n"
      "Same properties otherwise: symmetry defect O(h^10) after Re,\n"
      "even-power error expansion, requires s = a[0] = 0.5." },
    { "CCD4 (o4s4)_{1-0}",          4, &AppModel::scheme_ccd4s4_10,
      "Variant {1-0} of CCD4 (o4s4): each CD pass runs the IMPLICIT\n"
      "half first (h1 = s*h, reverse order), then the explicit one\n"
      "(h2, forward order). The same composition over the adjoint\n"
      "basic method: order 4 and symmetry as in CCD4 (o4s4).\n"
      "Measured: 4.0x smaller error on Lorenz, 1.5x on Rossler.\n"
      "The plain name is the {0-1} variant (explicit half first)." },
    { "CCD4 (s-split)_{0-1}",     4, &AppModel::scheme_ccd4ss01,
      "Dual of Complex CD4: TWO CD passes with REAL steps h/2,\n"
      "the complex part moved into the split inside each CD:\n"
      "pass 1: h1 = (1-i)h/6, h2 = (2+i)h/6  (s1 = (1-i)/3),\n"
      "pass 2: h1 = (2+i)h/6, h2 = (1-i)h/6  (s2 = 1 - s1).\n"
      "Both h^3 coefficients are purely imaginary and Re removes\n"
      "them: order 4 (measured 4.0 on Lorenz and Rossler).\n"
      "Same cost as Complex CD4; the error constant depends on\n"
      "the problem (2.9x smaller on Lorenz, 3.1x larger on\n"
      "Rossler). Slightly larger stability regions.\n"
      "NOT symmetric after Re (defect O(h^6)): Richardson on h^2\n"
      "gives order 5, not 6 (ExtrZ restores 6 and 8).\n"
      "{0-1}: explicit half first in each pass, as in every CD here.\n"
      "The split is part of the method: a[0] is NOT read." },
    { "CCD4 (s-split)_{1-0}",     4, &AppModel::scheme_ccd4ss10,
      "Same half-steps as CCD4 (s-split)_{0-1}, but each pass runs\n"
      "the IMPLICIT half first (reverse order), then the explicit one:\n"
      "E*_a, E_b | E*_b, E_a with a = (1-i)/6, b = (2+i)/6.\n"
      "This is the same composition over the adjoint basic method:\n"
      "the [N1,N2] coefficient flips sign but stays imaginary, so the\n"
      "order is still 4 (measured 4.0 on Lorenz and Rossler).\n"
      "Error constant differs from {0-1} and depends on the problem:\n"
      "10x larger on Lorenz, 2.8x smaller on Rossler.\n"
      "a[0] is NOT read." },
    { "RK45",                     5, &AppModel::scheme_rk45,
      "Dormand-Prince 5(4)7M: the ode45 / scipy RK45 pair.\n"
      "With a fixed step: 6 stages, order 5 (the 5th-order\n"
      "solution advances the step; the 4th-order one and the\n"
      "FSAL 7th stage are only for error control).\n"
      "Coefficients are the exact published fractions." },
    { "DOPRI78",                  8, &AppModel::scheme_dopri78,
      "Dormand-Prince RK8(7)13M, 13 stages, order 8 (the 7th-order\n"
      "embedded solution is computed but not used).\n"
      "Coefficients refined from the published fractions by solving\n"
      "all 200 order conditions to 1.8e-39 and stored to ~62 digits:\n"
      "double uses the nearest double, dd/qd in the Order tab get\n"
      "the full precision. Error floor ~1e-38 in dd (was ~1e-17).\n"
      "Note: the leading error term is minimized (the M in 13M), so\n"
      "the observed order at practical steps is ~9-10, reaching 8\n"
      "only at small h." },
    { "DOPRI78 (legacy)",         8, &AppModel::scheme_dopri78_legacy,
      "RK8(7)13M with the published rational coefficients of\n"
      "Prince & Dormand (1981), taken as is (no refinement).\n"
      "They satisfy the order conditions only to ~1e-18: in double\n"
      "this is indistinguishable from DOPRI78, in dd the error\n"
      "floors at ~1e-17. Kept for comparison." },
    { "DOP853",                   8, &AppModel::scheme_dop853,
      "Hairer's DOP853: explicit RK of order 8 with 12 stages,\n"
      "embedded 5th- and 3rd-order error estimators and a\n"
      "7th-order dense output (the scipy DOP853 method).\n"
      "With a fixed step only the 8th-order solution is used.\n"
      "Coefficients refined from the published 30-digit values\n"
      "by solving all 200 order conditions to 3e-78: dd/qd in\n"
      "the Order tab get the full precision." },
    { "GBS (n=2)",               2, &AppModel::scheme_gbs,
      "Gragg's modified midpoint with n = 2 substeps of h/2, the\n"
      "base of the Gragg-Bulirsch-Stoer family:\n"
      "z1 = X + (h/2) f(X),  z2 = X + h f(z1),\n"
      "X_next = (z1 + z2 + (h/2) f(z2)) / 2   (Gragg smoothing).\n"
      "Order 2, 3 RHS evaluations per step. NOT self-adjoint as a\n"
      "one-step method (defect O(h^6)), so Extr(GBS (n=2)|1,2,3)\n"
      "reaches only 5; the GBS 2-4-6 ... entries below keep ONE\n"
      "leapfrog chain per stage and reach the full 2K.", "GBS" },
    { "GBS 2-4",                 4, &AppModel::scheme_gbs24,
      "Gragg-Bulirsch-Stoer, stages n = 2, 4: stage k is ONE\n"
      "leapfrog chain of n substeps h/n started by Euler and closed\n"
      "by Gragg smoothing; stages run from the same X and are\n"
      "combined with Richardson weights in 1/n^2.\n"
      "Order 4 (measured 4.1 on Rossler), 8 RHS evaluations per step.", "GBS" },
    { "GBS 2-4-6",               6, &AppModel::scheme_gbs246,
      "Gragg-Bulirsch-Stoer, stages n = 2, 4, 6: stage k is ONE\n"
      "leapfrog chain of n substeps h/n started by Euler and closed\n"
      "by Gragg smoothing; stages run from the same X and are\n"
      "combined with Richardson weights in 1/n^2.\n"
      "Order 6 (measured 6.1 on Rossler), 15 RHS evaluations per step.", "GBS" },
    { "GBS 2-4-6-8",             8, &AppModel::scheme_gbs2468,
      "Gragg-Bulirsch-Stoer, stages n = 2, 4, 6, 8: stage k is ONE\n"
      "leapfrog chain of n substeps h/n started by Euler and closed\n"
      "by Gragg smoothing; stages run from the same X and are\n"
      "combined with Richardson weights in 1/n^2.\n"
      "Order 8, 24 RHS evaluations per step.", "GBS" },
    { "GBS 2-4-6-8-10",          10, &AppModel::scheme_gbs246810,
      "Gragg-Bulirsch-Stoer, stages n = 2, 4, 6, 8, 10: stage k is ONE\n"
      "leapfrog chain of n substeps h/n started by Euler and closed\n"
      "by Gragg smoothing; stages run from the same X and are\n"
      "combined with Richardson weights in 1/n^2.\n"
      "Order 10, 35 RHS evaluations per step.", "GBS" },
    { "GBS 2-4-6-8-10-12",       12, &AppModel::scheme_gbs24681012,
      "Gragg-Bulirsch-Stoer, stages n = 2, 4, 6, 8, 10, 12: stage k is ONE\n"
      "leapfrog chain of n substeps h/n started by Euler and closed\n"
      "by Gragg smoothing; stages run from the same X and are\n"
      "combined with Richardson weights in 1/n^2.\n"
      "Order 12, 48 RHS evaluations per step.", "GBS" },
};

inline constexpr int kBuiltinSchemeCount =
    (int)(sizeof(kBuiltinSchemes) / sizeof(kBuiltinSchemes[0]));



// Верхний ряд переключателей режимов. Один список на файл: по нему рисуется
// сам ряд и чекбоксы видимости в Settings.
// id — стабильный ключ для AppConfig::hidden_tabs: подпись менять можно, id
// нет, иначе у всех разъедется сохранённая маска.
// always = вкладку скрыть нельзя. Таких две: из Library выбирается система (без
// неё остальные вкладки пусты), а Settings — единственное место, где скрытое
// возвращается обратно.
struct ModeTab {
    const char*       id;
    const char*       label;
    AppModel::AppMode mode;
    bool              always;
};

inline const ModeTab kModeTabs[] = {
    { "Library",    "Library",        AppModel::AppMode::Library,    true  },
    { "Analysis",   "Phase analysis", AppModel::AppMode::Analysis,   false },
    { "Parametric", "Parametric",     AppModel::AppMode::Parametric, false },
    { "Dft1D",      "1D DFT",         AppModel::AppMode::Dft1D,      false },
    { "Basins",     "Basins",         AppModel::AppMode::Basins,     false },
    { "FastSync",   "Fast Synchro",   AppModel::AppMode::FastSync,   false },
    { "Custom",     "Custom",         AppModel::AppMode::Custom,     false },
    { "Order",      "Order",          AppModel::AppMode::Order,      false },
    { "Network",    "Network",        AppModel::AppMode::Network,    false },
    { "Settings",   "Settings",       AppModel::AppMode::Settings,   true  },
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
// bc — модель для ПКМ "Apply to all calculation tabs" (nullptr = не предлагать).
bool draw_scheme_combo(const char* label, std::string& scheme,
                              const std::vector<CustomScheme>& custom_schemes,
                              const std::function<void(const std::string&)>& on_pick = {},
                              const std::vector<std::string>* enabled_builtins = nullptr,
                              AppModel* bc = nullptr,
                              bool is_map = false,
                              const std::vector<std::string>* wrapper_schemes = nullptr,
                              bool broadcast = true) ;


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
// CCD4 (o4s3) / CCD4 (o4s4) / SEMP / SIMP и custom с a[0] в теле —
// см. scheme_uses_symmetry), иначе свип шёл
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
                                    bool allow_s = false,
                                    AppModel* bc = nullptr,
                                    BroadcastField which = BroadcastField::SweepTarget,
                                    bool allow_h = true,
                                    int* ad_axis = nullptr,
                                    const AdaptiveSettings* ad = nullptr) ;


// ---- Адаптивный шаг --------------------------------------------------------------
// Блок настроек AdaptiveSettings в секции Integration. opts — какие поля есть у
// вкладки (сырые узлы, интерполяция пиков, потолок узлов, перенормировка LLE/LS,
// min/max Metrics). Возвращает true, если что-то поменялось.
enum AdaptiveUiOpt {
    kAdUiRaw    = 1,    // выбор сетки вывода: равномерная / узлы шага
    kAdUiInterp = 2,    // интерполяция пиков на узлах шага
    kAdUiMaxPts = 4,    // потолок числа узлов (Analysis)
    kAdUiRenorm = 8,    // перенормировка LLE/LS
    kAdUiMinMax = 16,   // Metrics: min/max по интерполированным экстремумам и в Fixed
};


// Все настройки адаптивного шага в модели — в фиксированном порядке, чтобы откат
// мог сопоставить их с сохранёнными значениями (и отказаться, если число конфигов
// с тех пор изменилось).
std::vector<AdaptiveSettings*> collect_adaptive_slots(AppModel& m) ;


// «apply to all tabs»: одни и те же настройки шага во все вкладки, где они есть
// (Analysis, Custom, БД, LLE, LS, Basins, Metrics). Откат — Ctrl+Z.
void broadcast_adaptive(AppModel& m, const AdaptiveSettings& a) ;


// ---- Библиотека регуляторов шага ------------------------------------------------
// Окно редактора adaptive_ctrl_library(). Записи правятся на месте под замком библиотеки
// (расчётные потоки берут копию записи по имени); файл пишется, когда правка закончена —
// ни одно поле окна не активно.

std::vector<std::pair<std::string, std::string>> user_ctrl_list() ;


inline const char* const kCtrlBodyHelp =
    "void ucuda_ctrl_custom(const UcudaCtlIn& in, UcudaCtlMem& m, UcudaCtlOut& o) { <body> }\n"
    "\n"
    "in.h          size of this attempt;  in.t - t at the start of the step\n"
    "in.y0[i]      solution at t;  in.y1[i] - higher-order solution at t + h\n"
    "in.yerr[i]    error estimate y1 - yhat; in.nlo estimates of in.n values each\n"
    "              (DOP853: in.yerr[in.n + i] is the 3rd-order one)\n"
    "in.rtol, in.atol[i], in.c[k] - parameters of this controller, in.q - estimator order\n"
    "in.hmin, in.hmax (+inf: no limit; the driver clamps to it anyway), in.nrej - rejected attempts\n"
    "m.h[j], m.err[j]  accepted steps and their errors, j = 0 the latest (3 kept)\n"
    "m.nacc        accepted steps so far;  m.user[0..7] - free memory kept between steps\n"
    "o.err         error of the attempt (1 = on the tolerance): the norm or its square (Hairer\n"
    "              reports err^2); the driver only checks that it is finite; goes to the log\n"
    "o.accept      1 - take the step, 0 - retry it with o.h\n"
    "o.h           next step, or the step of the retry\n"
    "\n"
    "in.k[0..7]    constants computed by the prepare section (0 if not set)\n"
    "\n"
    "Prepare section (optional), once per trajectory:\n"
    "void ucuda_ctrl_custom_prep(const numb* c, int q, numb* k) { <prepare> }\n"
    "c[] - the parameters, q - estimator order, k[0..7] - constants for the body.\n"
    "Put divisions and powers that depend only on c[] and q here: double division\n"
    "costs as much as several RHS evaluations on GeForce cards.\n"
    "\n"
    "Helpers: ucuda_ctl_err(in) - RMS norm of the estimate, sc_i = atol_i + rtol max(|y0_i|, |y1_i|);\n"
    "ucuda_ctl_lg2err2(in, err2) - log2(err^2) in float, err^2 into err2, no sqrt (\"Hairer fast\");\n"
    "ucuda_ctl_lg2err2_numb(in, err2) - the same in numb (\"Hairer numb\");\n"
    "ucuda_ctl_pow(x, y) - fast x^y (float);  ucuda_ctl_log2(x), ucuda_ctl_exp2(f) - float;\n"
    "ucuda_clamp(x, lo, hi);  ucuda_fmin/fmax (double), ucuda_fminf/fmaxf (float).  numb is double.\n"
    "Index arrays (m.user, in.k, ...) with constants: a computed index puts the state of\n"
    "the GPU thread into local memory.\n"
    "The driver then clamps h to h_max, takes the step at h_min when it would go below\n"
    "(or after max rejects), and retries with at most 0.9 h.";


void draw_ctrl_library_window(AppModel& m) ;


bool draw_adaptive_block(const char* id, AdaptiveSettings& a, const std::string& scheme,
                                int opts, bool is_map, AppModel* bc = nullptr) ;


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
    // Адаптивный шаг: nullptr — у вкладки только постоянный шаг (DFT, FastSync,
    // Network, Stability). adaptive_opts — AdaptiveUiOpt.
    AdaptiveSettings* adaptive = nullptr;
    int adaptive_opts = 0;
};


// bc — модель для пункта "Apply to all calculation tabs" в ПКМ по полю;
// nullptr = меню не предлагать.
bool draw_integration_block(const char* header,
                                   const std::string& scheme,
                                   const std::vector<CustomScheme>& custom_schemes,
                                   const IntegrationFields& f,
                                   AppModel* bc = nullptr,
                                   bool is_map = false) ;


// Секции "Initial conditions" и "Parameters" — это один и тот же блок:
// по числовому полю на каждое имя из списка. Раньше — около дюжины копий.
// bc / field — см. draw_integration_block: ПКМ по полю предлагает разослать
// значение по вкладкам. field говорит, чем этот блок является — параметрами
// или начальными условиями; по заголовку это не определить (он ещё и с ##id).
void draw_named_num_fields(const char* header,
                                  const std::vector<std::string>& names,
                                  std::map<std::string, std::string>& values,
                                  const char* note = nullptr,
                                  AppModel* bc = nullptr,
                                  BroadcastField field = BroadcastField::Param,
                                  // Поля, которые расчёт ПЕРЕЗАПИШЕТ сам (область
                                  // устойчивости строит четыре элемента матрицы в
                                  // каждом узле сетки). Такие гасятся, а не молча
                                  // игнорируются: живое поле, ни на что не влияющее,
                                  // читается как настройка, которой оно не является.
                                  const std::function<bool(const std::string&)>& overwritten = {}) ;


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


// ---------------------------------------------------------------------------
// "Диапазон расчёта из вида" (Ctrl+T / Ctrl+Shift+T).
//
// После зума диаграмма показывает интересный кусок, а пересчитать ИМЕННО его
// можно было только переписав границы свипа руками, считывая их с подписей
// осей. Эти хелперы кладут видимый диапазон прямо в поля конфига: Ctrl+T — в
// тот конфиг, по которому построен вид (первая цель), Ctrl+Shift+T — во все
// конфиги, наложенные в этом окне в тех же осях.
//
// Ось, у которой нет входа в расчёте (значение переменной по Y у 1D-диаграммы),
// описывается нулевыми указателями и просто не трогается.
// Адреса полей одной цели. Отдельный тип, а не сам ViewRangeTarget, потому что
// отмена переразрешает их заново (см. relocate ниже), а рекурсивный
// std::function<ViewRangeTarget()> внутри ViewRangeTarget потребовал бы
// неполный тип в возвращаемом значении.
struct ViewRangeSlots {
    std::string* x_lo = nullptr;
    std::string* x_hi = nullptr;
    std::string* y_lo = nullptr;
    std::string* y_hi = nullptr;
    bool*        y_manual = nullptr;
};


struct ViewRangeTarget {
    std::string  name;                  // подпись в меню, когда целей несколько
    std::string* x_lo = nullptr;
    std::string* x_hi = nullptr;
    std::string* y_lo = nullptr;
    std::string* y_hi = nullptr;
    // Необязательный флаг "Y задан вручную" (colored-1D у Bif): без него
    // записанные ymin/ymax остались бы лежать мёртвым грузом при авто-Y.
    bool*        y_manual = nullptr;
    // Тип диаграммы — по нему работает Ctrl+Shift+T. "Parametric 2D" общий у
    // Bif/LLE/LS: у них совпадает и смысл осей, и кодировка цели свипа, поэтому
    // приблизив 2D-бифуркацию осмысленно подтянуть 2D LLE и 2D LS. У Basins и
    // FastSync оси — сетка НУ, у DFT по Y частоты; они в свои типы и попадают.
    // Пусто у целей, собранных на месте: тип заполняет только общий перечислитель.
    std::string kind;
    std::string group;   // подпись для списка галочек: "Bifurcation 2D", ...
    // Как найти ЭТУ ЖЕ цель потом. Нужно отмене: между Ctrl+T и Ctrl+Z вкладку
    // диаграммы могут закрыть, вектор конфигов переедет, и указатели выше
    // повиснут. Лямбда ищет конфиг по индексу и сверяет label; если его больше
    // нет или на его месте другой — возвращает нули, и откат этого поля просто
    // не делается. Пусто = цель незаписываемая (см. view_range_bound).
    std::function<ViewRangeSlots()> relocate;
};


// Цель Ctrl+T, живущая в векторе конфигов. Живые указатели и relocate строятся
// из ОДНИХ И ТЕХ ЖЕ указателей на члены — иначе списки полей разошлись бы.
template <class T>
[[nodiscard]] ViewRangeTarget vr_target(std::vector<T>& vec, int idx,
        std::string T::*x_lo, std::string T::*x_hi,
        std::string T::*y_lo = nullptr, std::string T::*y_hi = nullptr,
        bool T::*y_manual = nullptr) {
    ViewRangeTarget t;
    if (idx < 0 || idx >= (int)vec.size()) return t;   // цели нет — пустая
    t.relocate = [&vec, idx, label_id = label_identity(vec[(size_t)idx].label),
                  x_lo, x_hi, y_lo, y_hi, y_manual]() {
        ViewRangeSlots s;
        if (idx < 0 || idx >= (int)vec.size()) return s;
        T& c = vec[(size_t)idx];
        // Сверяем ИМЯ ОСИ, а не подпись целиком: диапазон в автоподписи 1D-диаграммы
        // меняет сама эта команда (см. label_identity).
        if (label_identity(c.label) != label_id) return s;   // на этом индексе другой конфиг
        s.x_lo = &(c.*x_lo); s.x_hi = &(c.*x_hi);
        if (y_lo)     s.y_lo = &(c.*y_lo);
        if (y_hi)     s.y_hi = &(c.*y_hi);
        if (y_manual) s.y_manual = &(c.*y_manual);
        return s;
    };
    const ViewRangeSlots live = t.relocate();
    t.name     = vec[(size_t)idx].label;
    t.x_lo     = live.x_lo;
    t.x_hi     = live.x_hi;
    t.y_lo     = live.y_lo;
    t.y_hi     = live.y_hi;
    t.y_manual = live.y_manual;
    return t;
}


// Цель, живущая в объекте с постоянным адресом (Shared config вкладки Custom —
// член CustomSession, а тот член AppModel). Вектора нет, переезжать нечему,
// поэтому relocate отдаёт те же указатели.
[[nodiscard]] ViewRangeTarget vr_target_fixed(std::string name,
        std::string* x_lo, std::string* x_hi,
        std::string* y_lo = nullptr, std::string* y_hi = nullptr) ;


// Границы вида — произвольные числа после зума. fmt_num_shortest насыпал бы в
// поле семнадцать цифр, а фиксированный %.6g схлопнул бы узкое окно
// (1.00000001 .. 1.00000002) в одно и то же число. Берём наименьшую точность,
// при которой запись расходится со значением меньше чем на миллионную долю
// ширины окна; при нулевой ширине (вырожденный вид) — round-trip как есть.
[[nodiscard]] std::string fmt_view_bound(double v, double span) ;


// Все диаграммы приложения, у которых есть диапазоны свипа. Нужен Ctrl+Shift+T
// (взять все цели того же типа) и списку галочек в меню диаграммы.
// Порядок — как во вкладках; на нём же строится маска выбора.
[[nodiscard]] std::vector<ViewRangeTarget> all_view_range_targets(AppModel& model) ;


// Цели того же типа, что и source. Источник опознаём по АДРЕСУ поля: цель,
// собранная на месте отрисовки, и цель из общего перечисления указывают на
// один и тот же конфиг, а сравнивать больше не по чему.
// Источник в списке идёт первым — Ctrl+T и Ctrl+Shift+T тогда пишут в него одно
// и то же, а не в разном порядке.
[[nodiscard]] std::vector<ViewRangeTarget>
view_range_same_kind(AppModel& model, const ViewRangeTarget& source) ;


// Применяет диапазон к первым count целям ОДНОЙ командой: в стек отмены уходит
// одна запись на нажатие, а не по одной на каждую диаграмму окна.
// Что именно раскладывается по целям. Считается ОДИН раз: у всех целей
// значения одинаковые, различается только набор полей, куда они лягут.
struct ViewRangeValues {
    bool has_x = false; std::string x_lo, x_hi;
    bool has_y = false; std::string y_lo, y_hi;
    const char* source = "view";   // для подписи в стеке отмены
};


// swapped — включённый HeatmapView::swap_axes. Вью после свапа живёт в
// ВИЗУАЛЬНЫХ координатах: x_axis.view_min/max тогда описывают ось ДАННЫХ Y и
// наоборот (имена осей при этом не свапаются — см. heatmap_view.cpp, шаг 0).
// Без этой перестановки на свапнутой карте диапазоны писались бы крест-накрест.
// src_has_y — ось Y ВИДА-ИСТОЧНИКА является входом расчёта. У 1D-диаграммы
// по Y идёт посчитанная величина, у которой входа нет, и её границы в поле
// свипа — мусор. apply_view_values страхует только ПРИЁМНИК (у 1D-цели y_lo
// нулевой, и запись просто не делается), но рассылка с 1D-вида на 2D-цель,
// у которой Y — настоящая ось свипа, клала туда диапазон значений.
[[nodiscard]] ViewRangeValues values_from_axes(const AxisInfo& x, const AxisInfo& y,
                                                      bool swapped, bool src_has_y) ;


// Значения ИЗ ПОЛЕЙ цели. Нужно после Ctrl+Z: отмена вернула поля, а картинка
// осталась зумленной, и рассылка "из вида" затёрла бы только что
// восстановленное тем, что осталось на экране.
[[nodiscard]] ViewRangeValues values_from_target(const ViewRangeTarget& t) ;


// Пишет значения в одну цель. Ось, которой у цели нет, не трогается — так 2D
// диапазон можно разложить и по 1D-диаграммам, не выдумывая им вторую ось.
void apply_view_values(const ViewRangeTarget& t, const ViewRangeValues& v) ;


void run_view_range_cmd(const std::vector<ViewRangeTarget>& targets, size_t count,
                               const ViewRangeValues& vals, AppModel* model) ;


[[nodiscard]] bool view_range_bound(const std::vector<ViewRangeTarget>& targets) ;


// Есть ли у вида-источника вход расчёта по Y. Отдельного признака заводить не
// нужно: ровно это и кодирует ненулевой y_lo его собственной цели (см.
// all_view_range_targets — 1D-целям передаются только поля X).
[[nodiscard]] bool src_y_is_input(const std::vector<ViewRangeTarget>& targets) ;


// Пункты правого клика. Зовётся из popup_extras, т.е. уже внутри BeginPopup.
// swapped и model — БЕЗ значений по умолчанию, и это принципиально. Пока они были
// умолчательными, вызов с четырьмя аргументами компилировался молча: AppModel* приводился
// к bool в параметр swapped (указатель всегда ненулевой, то есть swapped = true), а model
// оставался nullptr. У всех 1D-графиков это давало сразу три поломки — диапазон свипа брался
// с оси Y вместо X, запись не попадала в стек отмены и не отчитывалась в шапке, а
// Ctrl+Shift+T сваливался на одну цель. Теперь недостача аргумента — ошибка компиляции.
void draw_view_range_menu(const std::vector<ViewRangeTarget>& targets,
                                 const AxisInfo& x, const AxisInfo& y,
                                 bool swapped, AppModel* model) ;


// Горячие клавиши того же действия. active — окно диаграммы под курсором:
// плотов на экране бывает несколько, и глобальный хоткей (как Ctrl+R у Run)
// здесь означал бы "перепиши диапазон неизвестно чему".
void handle_view_range_keys(const std::vector<ViewRangeTarget>& targets,
                                   const AxisInfo& x, const AxisInfo& y, bool active,
                                   bool swapped, AppModel* model) ;


// Окно диаграммы под курсором — общее условие хоткея выше. Берём именно
// hover, а не focus: фокус остаётся на панели настроек, откуда зумили колесом.
[[nodiscard]] bool plot_window_active() ;


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


// Выбранные вкладки узлов дока при смене режима (Analysis, Parametric, ...). Окна чужого
// режима не выводятся, и ImGui вынимает их из узлов дока; при возвращении окно, появившееся
// последним, получает фокус, и его узел выбирает именно его, а не то, что было выбрано.
// Пока режим на экране, запоминаем выбранное окно каждого узла; первые кадры после входа
// просим узлы выбрать его снова (NextSelectedTabId: узел сам переведёт на окно и фокус).
// Кадров несколько: в первом узел ещё пересобирает таб-бар и выбирает по-своему.
// Зовётся после вывода всех окон кадра.
void track_dock_selection(AppModel& model) ;


// Таб-бар "одна вкладка на конфиг + кнопка +" (раньше — шесть копий).
// item_id_prefix задаёт неизменную часть ID вкладки ("bd_tab_", "lle_tab_", …), поэтому
// идентификаторы побайтово совпадают с прежними. ID завязан на номер конфига, поэтому
// бар НЕ Reorderable: перетаскивание меняло только порядок на экране, а не в данных, и
// после закрытия вкладки из середины ImGui считал конфиг, сдвинувшийся на её номер, новой
// вкладкой и ставил его в конец (BD1, BD4, BD3). Без Reorderable новые вкладки
// сортируются по порядку вывода, то есть по порядку конфигов. body == nullptr — тело вкладки пустое
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
enum class ParamPlotKind { Bifurcation, LLE, LS, Metrics };


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


// Стиль линейного 2D-графика проекции Analysis (Phase 2D, Time domain, Step size, Rejections):
// Custom line style — толщина и α (толстая линия), Points — точки вместо линии; обе галки —
// точки поверх линии (Plot2DView::points_with_lines). Настраивает projection_views(pr).view2d; α к цвету линейных
// серий применяет вызывающий (по pr.custom_line_style). Серии-маркеры (points_override = 1)
// стиль не трогает.
void draw_proj_line_style(Projection& pr, const char* sfx, const char* points_tip) ;


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
void draw_continuation_device_block(Cfg& c, const char* id, bool blocked,
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
        ImGui::TextDisabled("(continuation: param- or h-sweep only, not IC, not 2D)");

    int dev = c.use_gpu ? 0 : 1;
    ImGui::BeginDisabled(device_locked);
    ImGui::RadioButton(("GPU##dev" + sfx).c_str(), &dev, 0); ImGui::SameLine();
    ImGui::RadioButton(("CPU##dev" + sfx).c_str(), &dev, 1);
    ImGui::EndDisabled();
    if (!device_locked) c.use_gpu = (dev == 0);
    if (c.continuation && c.use_gpu) {
        ImGui::SameLine();
        ImGui::TextDisabled("(continuation on the GPU is single-threaded, CPU is faster)");
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
void configure_sweep_x_axis_from(Plot2DView& view,
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
void apply_snap_x_from_first_member(Plot2DView& view,
                                           const std::vector<int>& members,
                                           const std::vector<Cfg>& cfgs) {
    view.snap_x_to_grid = true;
    view.snap_x_n       = 0;
    if (members.empty()) return;
    // Первый член С РЕЗУЛЬТАТОМ, а не просто первый: сетку задаёт то, что
    // реально нарисовано, а members[0] может быть ещё не посчитан — тогда
    // снап уходил на поля конфига и дальше на их дефолты.
    const Cfg* pick = nullptr;
    for (int aidx : members) {
        if (aidx < 0 || aidx >= (int)cfgs.size()) continue;
        const Cfg& c = cfgs[(size_t)aidx];
        if (!pick) pick = &c;
        if (c.last_run_ok && c.result.n_pts > 1) { pick = &c; break; }
    }
    if (!pick) return;
    apply_snap_x_from_config(view, pick->last_run_ok,
                             pick->result.param_lo, pick->result.param_hi, pick->result.n_pts,
                             pick->param_lo_text, pick->param_hi_text, pick->n_pts_text);
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
std::vector<std::vector<double>>& window_point_bufs(int window_id, size_t n_series) ;


// Узлы сетки свипа фокусной кривой окна (Plot2DView::snap_x_nodes). Живут
// между кадрами, как и буферы точек: вид держит указатель на время render().
std::vector<double>& window_node_buf(int window_id) ;


// Заполняет узлы ТОЙ ЖЕ формулой, что строит точки, и отдаёт их виду. Узлы, на
// которых точек не оказалось (режим не колебательный, λ не финитна), здесь
// тоже есть: курсор обязан на них вставать — расчёт для них проводился.
void set_snap_nodes(Plot2DView& view, int window_id,
                           int npts, double lo, double hi,
                           bool log_scale, bool reverse, bool continuation) ;


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
// Generate button, error text and the generated-code viewer. Extracted because
// the System tab stops earlier for a map - no schemes and no custom KRS.
void draw_generated_code_block(AppModel& model, const GuiCallbacks& cb) ;


// --- Коэффициенты обёртки: то, что иначе остаётся чёрным ящиком -------------
//
// И у Extr, и у Comp имя схемы описывает ВХОД (база, подшаги, выражения для
// gamma), а в КРС работают ВЫВЕДЕННЫЕ из него числа: веса Ричардсона alpha_k
// вычисляются, символьные gamma раскрываются значениями параметров. Увидеть их
// можно было ровно в одном месте — в конструкторе, и только пока в нём набран
// ровно этот набор; в списке собранных схем и в комбо выбора не было ничего.
//
// Отчёт собирается ТЕКСТОМ, а не рисуется таблицей, по одной причине: одни и
// те же числа нужны на экране, в тултипе комбо и в буфере обмена (коэффициенты
// уезжают в статью). Три способа нарисовать одно и то же разъехались бы.
std::string wrapper_coefficients_text(
        const std::string& name,
        const std::vector<CustomScheme>& custom_schemes,
        const std::map<std::string, std::string>& param_values) ;


// Отчёт на экран. Моноширинного шрифта в проекте нет, поэтому колонки держатся
// пробелами, а блок печатается ПОСТРОЧНО через TextUnformatted: TextWrapped
// сломал бы выравнивание переносом, а один многострочный Text — отступами.
void draw_wrapper_coefficients(const std::string& name,
                                      const std::vector<CustomScheme>& custom_schemes,
                                      const std::map<std::string, std::string>& param_values,
                                      const char* id) ;


// Таблица коэффициентов — СТОЛБЦОМ, по строке на стадию: в ряд она не читается
// уже на пяти полях, а у Comp их до сорока. Высота ограничена max_rows
// строками, дальше список прокручивается, иначе сорок полей выдавливают со
// вкладки всё остальное. Открывает child — парный EndChild обязателен.
void begin_coeff_column(const char* id, int rows, int max_rows) ;


// Поле имени схемы-обёртки, общее для обоих конструкторов. Пустое поле — это
// не "без имени", а "имя по умолчанию", поэтому подсказкой в нём стоит ровно то
// описание, которое иначе и встанет в список. Возвращает готовую метку: края
// подрезаны, разделители самого имени выброшены.
std::string wrapper_name_field(const char* id, std::string& text,
                                      const std::string& default_name) ;


// Конструктор экстраполяционных обёрток. Собирает ИМЯ вида "Extr(RK4|1,2,4)" —
// тела здесь нет и не хранится нигде: его пересобирает compute_krs_for_scheme
// при каждом обращении, поэтому правка системы обёртку не протухает.
// Схемы, читающие a[0] как коэффициент симметрии, держат паспортный порядок
// только при a[0] = 0.5; проверить это в кодогене нельзя (значение приходит в
// рантайме), поэтому предупреждаем здесь.
bool extr_base_needs_half_s(const std::string& nm) ;


void draw_extrapolation_builder(AppModel& model) ;


// Читает таблицу коэффициентов композиции из файла в конструктор. Число стадий
// берётся ИЗ ФАЙЛА: у опубликованных методов высокого порядка их 17, 31, 35 —
// выставлять K руками перед загрузкой значило бы считать строки глазами.
// Разбор — parse_composition_coeff_file, здесь только файл и строка статуса.
void comp_load_coeffs_from_file(AppModel& model, const std::string& path) ;


// Конструктор композиций. Имя вида "Comp(CD|g1,1-2*g1,g1)"; тела, как и у Extr,
// нигде нет — резолвер пересобирает его из имени.
void draw_composition_builder(AppModel& model, const GuiCallbacks& cb) ;


// Общий список собранных обёрток. Один на оба конструктора: список один, и
// проверка отвалившейся базы у Extr и Comp дословно совпадает.
void draw_wrapper_list(AppModel& model) ;


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
    std::string              sel_name;        // для какой системы прочитано превью ниже
    bool                     sel_ok = false;  // запись прочиталась
    SystemRecord             sel_rec;
    std::string              sel_desc;        // describe_record(sel_rec), готов для буфера обмена
    double                   filled_at = -1.0;
    bool                     valid     = false;
};

inline LibraryListCache g_library_cache;


void invalidate_library_cache() ;


const LibraryListCache& library_list_cached(SystemLibrary& lib) ;


// Persist the Library visibility mask. Read-modify-write, like the other
// call sites: save_app_config rewrites the whole file.
void persist_hidden_systems(const AppModel& model) ;


// Plain-text dump of a library record: everything the editor shows except the
// scheme checkboxes. Lets the Library tab hand the description to the clipboard
// without opening the editor (where a stray edit could be saved over the system).
std::string describe_record(const SystemRecord& r) ;


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
// cbp — только ради диалога сохранения netlist'а. Необязателен: в Custom эту же
// панель рисует уровень 3, где callbacks не прокинуты, и кнопка там неактивна.
void draw_phase_controls(PhaseAnalysisSession& s,
                                std::function<void()> on_reset_defaults,
                                AppModel* bc = nullptr,
                                const GuiCallbacks* cbp = nullptr) ;


// Единый счётчик режимов для статусных строк всех панелей. Раньше каждая панель считала
// "diverged" по flags[k] < 0, и подпись значила разное: у Bif1D/Bif2D отрицательный флаг —
// это fixed point, а у LLE/LS — расходимость. Теперь коды канонические (REGIME_* в
// configCUDA.h) и категории раскладываются одинаково везде.
void count_regimes(const std::vector<int>& flags, int& fp, int& unb, int& osc) ;


// Подпись под "OK: ..." — печатается только если есть что показать кроме
// колебательного режима (иначе строка была бы шумом на каждом успешном run'е).
// Сводка адаптивного шага по свипу: st — по 4 числа на точку (принятые, отвергнутые,
// вынужденные шаги, средний h), пусто при постоянном шаге.
void draw_adaptive_stats(const std::vector<double>& st) ;


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
void draw_diagram_controls(AppModel& model, BifurcationAnalysisSession& s, int idx) ;


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
void draw_lle_curve_controls(AppModel& model, LLEAnalysisSession& s, int idx) ;


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

void draw_ls_curve_controls(AppModel& model, LyapunovSpectrumAnalysisSession& s, int idx) ;


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


// Metrics: контролы (per-config в табе) + графики. UX зеркалит LLE: те же
// свип, 2D-режим и блок интегрирования, плюс выбор сигнала и набора метрик.
// Какую из посчитанных метрик рисовать, решает таб-бар окна графика
// (ParametricPlotWindow::metric) — как вид в окне Basins.

// Подсказка к чекбоксу метрики — определение ровно в том виде, как считает ядро.
const char* signal_metric_tooltip(int m) ;


void draw_metrics_config_controls(AppModel& model, SignalMetricsAnalysisSession& s, int idx) ;


void draw_metrics_controls(AppModel& model, SystemLibrary& /*lib*/) ;


// Таб-бар выбора метрики над графиком. Табы — метрики, посчитанные хотя бы у
// одного члена окна (до первого прогона — отмеченные в конфигах). Выбор живёт
// в win.metric и сохраняется вместе с окнами; ImGui помнит выбранный таб сам,
// поэтому при расхождении (окно только что загружено, выбранной метрики больше
// нет) таб выставляется явно через SetSelected.
void draw_metric_tab_bar(AppModel& model, ParametricPlotWindow& win, int avail_mask) ;


void draw_metrics_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                              ParametricPlotWindow& win,
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
    "Bifurcation 1D", "Bifurcation 2D", "LLE 1D", "LLE 2D", "LS 1D", "LS 2D",
    "Metrics 1D", "Metrics 2D"
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

void draw_dft1d_diagram_controls(AppModel& model, Dft1DAnalysisSession& s, int idx) ;


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
// n_text == nullptr — ось не владеет числом узлов (в 2D оно общее и живёт на
// X): поле не рисуется, вместо него подпись, откуда берётся разрешение.
void draw_order_axis_block(const char* id, const char* title,
                                  const OrderAnalysisSession& s,
                                  int& target, std::string& lo, std::string& hi,
                                  bool& log_grid, std::string* n_text,
                                  bool enabled, const char* n_label = "points") ;


// Индикатор числа шагов + предупреждение о нецелом t_max/h.
//
// Почему это не косметика: без snap число грубых шагов = floor(t_max/h), и
// фактическое конечное время N*h отличается от t_max на величину до h. Оно
// разное у соседних узлов сетки, так что на кривую ошибки садится вклад
// порядка O(h) — для схем порядка выше первого он полностью забивает
// измеряемую величину. На лог-сетке по h нецелым оказывается почти каждый
// узел, поэтому счётчик показывает, сколько именно узлов задеты.
void draw_order_steps_hint(const OrderConfig& c, const OrderAnalysisSession& s) ;


// ---------------------------------------------------------------------------
// Блок настроек области устойчивости.
//
// Элементы тестовой матрицы здесь НЕ вводятся — вводятся k, r и h, а a, b, c, d
// пересчитываются в каждом узле сетки так, чтобы спектр был ровно
// sigma +- i*omega (формулы в kernels/order.template.cu). Поэтому поля матрицы
// в этом блоке отсутствуют, а вместо них стоят селекторы «в какой a[] класть
// какой элемент»: пресет library/Dahlquist 2D даёт 1..4, но переименовать или
// переставить параметры системы пользователю никто не запрещает, и молча
// промахнуться мимо индекса — это неверная картинка без единого сообщения.
void draw_order_stability_block(AppModel& model, const OrderAnalysisSession& s,
                                       OrderConfig& c) ;


// Определено ниже, вместе с окнами графиков: панель настроек раздаёт по ним
// вкладки и обязана знать, какой вид окна какие конфиги принимает.
bool order_member_fits(const OrderConfig& c, OrderPlotWindow::Kind k);


// Вид окна, в котором конфигу место по его текущим настройкам.
OrderPlotWindow::Kind order_kind_for(const OrderConfig& c) ;


// Конфиг сменил вид расчёта (или размерность сетки) — перевесить его на окно,
// которое такое умеет показывать. Без этого переключение на performance
// вынимало вкладку из всех окон, и Run выглядел как «ничего не произошло».
//
// Новое окно заводится только в крайнем случае. Раньше карта (а с ней и
// область устойчивости) ВСЕГДА получала своё окно, а окно, из которого
// конфиг ушёл, оставалось пустым, — и каждое переключение Order p /
// Performance / Stability плодило по окну. Теперь порядок такой:
//   1) конфиг уже лежит в подходящем окне — ничего не делать;
//   2) есть окно нужного вида: для одномерных — любое (оверлей), для карты —
//      только пустое (хитмапа одиночна);
//   3) иначе окно, которое опустело от этого ухода, просто меняет вид;
//   4) иначе — новое окно.
// Окно, опустевшее от ухода и не пригодившееся в 3), убирается: пустым оно
// лишь занимает место в раскладке. Окна с ручной подписью не трогаются.
void order_rehome_config(AppModel& model, int idx) ;


void draw_order_controls(AppModel& model, SystemLibrary& /*lib*/) ;


// ---------------------------------------------------------------------------
// Order — окна графиков (см. OrderPlotWindow в app_model.h).
//
// Окно знает, ЧТО оно рисует (kind), и какие конфиги в нём лежат (members).
// Одномерные виды оверлеят сколько угодно вкладок — ради сравнения схем между
// собой вкладка и заведена; карта p(x, y) остаётся одиночной.

const char* order_window_kind_name(OrderPlotWindow::Kind k) ;


// Может ли конфиг попасть в окно этого вида. Performance показывает только
// замеры времени, карта — только двумерные прогоны, p/ошибка — одномерные.
bool order_member_fits(const OrderConfig& c, OrderPlotWindow::Kind k) ;


// Оттенок базового цвета серии: min темнее, max светлее, avg — как есть.
// Три кривые одной вкладки обязаны читаться как одна группа.
//
// k < 1 — подмешать чёрного, k > 1 — БЕЛОГО, а не умножить на k с клампом:
// умножение на светлых цветах палитры (жёлтый, голубой) упиралось в 1.0 по
// каждому каналу, и max получался неотличим от avg — а у самых светлых карт
// сливались все три. Примесь белого светлеет всегда, независимо от того, как
// близко канал к насыщению.
ImVec4 order_shade(ImVec4 c, float k) ;


// Форма маркера по порядковому номеру кривой внутри вкладки (min/avg/max или
// E1/E2). Второй признак принадлежности помимо оттенка: на плотной сетке узлов
// и при десятке кривых в окне цвета всё равно сближаются.
int order_marker_for(int within) ;


// Имена палитр для комбо — в том же порядке, что константы kOrderPalette*.
inline const char* const kOrderPaletteNames[3] = {
    "App palette", "High contrast", "Colorblind-safe"
};


// Базовый цвет члена окна. Палитры КАЧЕСТВЕННЫЕ (см. kOrderPalette* в
// app_model.h): номер в списке — номер вкладки в окне, и соседние номера
// обязаны читаться как разные цвета, а не как соседние оттенки.
//
// High contrast — двенадцать насыщенных тонов средней светлоты: слишком
// светлые (лимон, розовый) из ходовых "distinct color" наборов выброшены,
// потому что order_shade делает из них ещё и осветлённый вариант для max, а
// тот на светлой теме плота уже не виден. Okabe-Ito взят целиком, кроме
// чёрного: на тёмной теме он сливается с фоном, поэтому заменён на серый.
ImVec4 order_base_color(int idx, int palette) ;


// Короткая сводка по статусам прогона — содержательная часть диаграммы:
// левый обвал (потеря устойчивости) и правая полка (машинная точность) видны
// именно как счётчики, а не только как форма кривой.
void draw_order_status_line(const char* label, int n_ok, int n_div, int n_floor, int n_noc) ;


// Снимок настроек вкладки для <path>_config.csv: движок его не собирает —
// OrderResult/PerfResult не носят ни схемы, ни параметров системы.
data_export::OrderSnapshot order_snapshot(const OrderAnalysisSession& s,
                                                 const OrderConfig& c,
                                                 bool two_d, bool gpu_fmad, bool gpu_rdc) ;


// ПКМ по графику: по пункту на каждую лежащую в окне вкладку — в оверлее
// «просто экспортировать» нечего, файлы у вкладок разные.
void draw_order_export_items(AppModel& model, const OrderPlotWindow& win,
                                    const GuiCallbacks& cb, bool busy) ;


// Карта области устойчивости. Живёт в окне того же вида (Map): хитмапа с одним
// членом — ровно то, что ей нужно, заводить ради неё отдельный вид окна было бы
// дублированием всей развески вкладок по окнам.
void draw_order_stability_map(AppModel& model, OrderPlotWindow& win,
                                     const GuiCallbacks& cb,
                                     PlotRenderer& renderer, HeatmapView& hv,
                                     OrderConfig& c, int mi) ;


// 2D-карта: ровно один член, хитмапа p или log10 E1.
void draw_order_map_window(AppModel& model, OrderPlotWindow& win,
                                  const GuiCallbacks& cb,
                                  PlotRenderer& renderer, HeatmapView& hv) ;


// Одномерное окно: p(x), E(x) или «время vs ошибка». Кривые всех членов
// лежат в одних осях.
void draw_order_curve_window(AppModel& model, OrderPlotWindow& win,
                                    const GuiCallbacks& cb,
                                    PlotRenderer& renderer, Plot2DView& view) ;


void draw_order_plot_windows(AppModel& model, const GuiCallbacks& cb) ;


// ---------------------------------------------------------------------------
// Network — сеть связанных осцилляторов.
//
// Три окна: Controls (топология, связь, узлы, рёбра), Network Graph (сам граф,
// раскрашенный состоянием на выбранный момент) и Node Series (кривые узлов) +
// Spatiotemporal (растр узел x время).
//
// Редактор графа держит состояние в model.network_view, а НЕ в статиках:
// перетаскивание узла и протяжка ребра живут между кадрами, и в immediate mode
// им место в модели.

inline const char* const kNetTopologyNames[] = {
    "Custom", "Chain", "Ring", "Star", "Grid", "King graph", "All-to-all", "Small world"
};


// Значение переменной var узла node в точке p.
double net_value_at(const NetworkResult& r, int p, int node, int var) ;


// Степень узла по рёбрам конфига (для таблицы узлов). Считается по рёбрам, а
// не по CSR: CSR строится под расчёт и знать про него редактору незачем.
int net_degree(const NetworkConfig& c, int node) ;


// Любая ручная правка топологии переводит конфиг в Custom: иначе следующее
// нажатие Generate молча стёрло бы правку.
void net_mark_custom(NetworkConfig& c) ;


// Проверка выражений связи без запуска расчёта: тот же путь, что у Run.
std::string net_validate_laws(const NetworkSession& s, const NetworkConfig& c) ;


void draw_network_topology_block(AppModel& model, NetworkSession& s, NetworkConfig& c) ;


void draw_network_coupling_block(AppModel& model, NetworkSession& s, NetworkConfig& c) ;


void draw_network_nodes_table(AppModel& model, NetworkSession& s, NetworkConfig& c) ;


void draw_network_edges_table(AppModel& model, NetworkSession& s, NetworkConfig& c) ;


void draw_network_controls(AppModel& model, SystemLibrary& /*lib*/) ;


// --- Окно графа ------------------------------------------------------------

void draw_network_graph_window(AppModel& model) ;


// --- Кривые узлов ----------------------------------------------------------

void draw_network_series_window(AppModel& model, PlotRenderer& renderer, Plot2DView& view) ;


// --- Растр узел x время ----------------------------------------------------

void draw_network_raster_window(AppModel& model, PlotRenderer& renderer, HeatmapView& view) ;


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
bool apply_session_json(AppModel& model,
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


// Загрузка Order из файлов засевает кэш автосохранения (autosave_order_session)
// только что загруженным состоянием: файл перепишется при первой настоящей
// правке, а не сразу после загрузки. Иначе битый файл, отвергнутый
// apply_session_json, молча заменился бы умолчаниями, хотя пользователь ещё
// ничего не менял.
void order_autosave_seed(AppModel& model) ;


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

void draw_shared_config(AppModel& model, CustomSession& cs,
                        const std::vector<CustomScheme>& custom_schemes) ;


// Level 2D detail

void draw_level2d_detail(AppModel& model, CustomSession& cs) ;


// Level 1D detail

void draw_level1d_detail(CustomSession& cs) ;


// Level 3 detail

void draw_level3_detail(AppModel& model, CustomSession& cs) ;


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
void draw_custom_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb);


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
enum class BusyKind { None, Bif, LLE, LS, Dft1D, Basins, Phase, FastSync, Custom, Order, Metrics };


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
[[nodiscard]] std::string running_label(const Vec& items, int idx, const char* fallback) {
    if (idx >= 0 && idx < (int)items.size() && !items[idx].label.empty())
        return items[idx].label;
    return fallback;
}


// Снять состояние с сессии. Годится для любой, у которой есть
// compute_start_time / cancel_token / progress_token — то есть для всех, кроме
// phase: у неё нет ни отмены, ни прогресса, и её ветка заполняется вручную.
template <class Session>
[[nodiscard]] BusyInfo busy_from(const Session& s, BusyKind kind, std::string what) {
    BusyInfo b;
    b.kind  = kind;
    b.what  = std::move(what);
    b.start = s.compute_start_time;
    b.cancelling = s.cancel_token && s.cancel_token->load(std::memory_order_relaxed);
    if (s.progress_token)
        b.progress = s.progress_token->load(std::memory_order_relaxed);
    return b;
}


// Главное окно: переключатель режимов Library / Analysis / Parametric
void gui_process_deferred(AppModel& model, const GuiCallbacks& cb) ;


// Автосохранение вкладки Order. Раньше _last_order.json писался только по
// завершении расчёта, а _last_order_windows.json — по флагу, который ставили
// лишь добавление/удаление окон и правка их членов. Всё прочее — новая или
// удалённая вкладка, переименование, схема, поля, вид карты, колормапа,
// лог-оси — до файла не доходило, пока не досчитается очередной Run, а при
// выходе без Run терялось. Хуже того, файлы расходились между собой: окна
// сохранялись со ссылкой на вкладку, которой в сохранённых конфигах ещё нет.
//
// Теперь оба файла пишутся, как только сериализация отличается от последней
// записанной. Сериализация Order — несколько килобайт текста, сравнивать её
// каждый кадр дешевле, чем расставлять флаг по сотне мест правки (включая
// ПКМ-рассылку в конфиги Order с чужих вкладок).
//
// Писать можно только сессию ТЕКУЩЕЙ системы: после смены системы на другой
// вкладке order_session держит старую, пока в Order не зайдут снова, и без
// этой проверки её содержимое легло бы в файл новой системы.
void autosave_order_session(AppModel& model, SystemLibrary& lib) ;
}
