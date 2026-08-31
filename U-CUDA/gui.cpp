#include "gui.h"
#include "imgui.h"
#include "imgui_internal.h"   // DockBuilder* API — needed for Custom Workspace per-tab dockspaces.
#include "implot.h"
#include "implot3d.h"
#include "plot_renderer.h"
#include "session_io.h"
#include "plot_view_2d.h"
#include "plot_view_3d.h"
#include "heatmap_view.h"
#include "app_config.h"
#include "data_export.h"
#include "krs_cpu.h"
#include "num_parse.h"   // parse_num — единый разбор числовых полей
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

// ---- Именованные константы вместо магических чисел ----
// Значение, которым все хитмапы помечают «данных нет» (расходящаяся ячейка).
// HeatmapView красит его в серый; одно и то же число в colored-1D, DFT и Basins.
static constexpr double kSentinelNoData = 999.0;
// Colormap по умолчанию для бассейнов и FastSync, когда в конфиге ничего не
// выбрано: slanCM #168 turbo — хорошо разделяет дискретные значения. Раньше
// здесь стоял id 2 (встроенный полиномиальный Turbo), который colormap_id_or
// теперь и переводит в эту же карту.
static constexpr int    kColormapTurbo  = slancm_id(168);
// Высота бокса ошибки в строках текста.
static constexpr int    kErrorBoxLines  = 12;
// Через столько кадров простоя вытесняется кэш рендереров владельца.
static constexpr int    kRendererEvictFrames = 600;
// Дебаунс автозапуска фазовых портретов по бассейнам (сек).
static constexpr double kBasinsPhaseDebounceSec = 0.3;
// Дебаунс автопересчёта Level 1D в Custom после возни со слайдером (сек).
static constexpr double kCustomSliderDebounceSec = 0.5;
// Как часто перечитывается список систем в Library (сек). Файлы никто не
// правит извне при работающем приложении, поэтому опрос редкий.
static constexpr double kLibraryCacheTtlSec = 10.0;
// Стандартная ширина числового поля и комбо в панелях настроек.
static constexpr float  kFieldW = 120.0f;
static constexpr float  kComboW = 160.0f;

// Возвращает директорию exe со слешем в конце. Реализована в app_main.cpp
// (там же используется для resolve_python_exe / library_dir).
extern std::string exe_dir();
static std::string get_exe_dir_with_sep() {
    std::string d = exe_dir();
    if (!d.empty() && d.back() != '\\' && d.back() != '/') d += "\\";
    return d;
}

// Базовый цвет траектории по индексу НУ (единый для 2D/3D/time domain).
static ImVec4 ic_base_color(int ic_index) {
    return ImPlot::GetColormapColor(ic_index);
}

// Оттенок базового цвета по насыщенности: для переменной vi из nv внутри
// одного НУ. vi=0 — самый насыщенный, дальше бледнее. Используется в time domain,
// чтобы переменные одного НУ были видимо родственны (один тон), но различимы.
static ImVec4 shade_of(ImVec4 base, int vi, int nv) {
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(base.x, base.y, base.z, h, s, v);
    // распределяем насыщенность от 1.0 (vi=0) до ~0.35 (последняя переменная)
    float frac = (nv <= 1) ? 0.0f : (float)vi / (float)(nv - 1);
    float new_s = s * (1.0f - 0.45f * frac); // от s до 0.55*s (было 0.65 -> 0.35, тускло)
    float r, g, b;
    ImGui::ColorConvertHSVtoRGB(h, new_s, v, r, g, b);
    return ImVec4(r, g, b, base.w);
}
static ImVec4 ic_var_shade(int ic_index, int vi, int nv) {
    return shade_of(ic_base_color(ic_index), vi, nv);
}

// ---- helpers: std::string <-> ImGui ----
// Подпись комбинации переменных: x0 + pi*x1 + e*x2 — тот ряд, что ядро строит
// при writable_var == -1 (cudaLibrary.cu, loopCalculateDiscreteModel_int), с
// той же деградацией при dim < 3. Одна на все места, где комбинация
// предлагается пользователю: Feature diagram и Time domain.
// (draw_writable_var_combo держит свою копию — она старше и в этот заход не
// трогалась.)
[[nodiscard]] static std::string combo_var_label(const std::vector<std::string>& vars) {
    if (vars.size() >= 3) return vars[0] + " + pi*" + vars[1] + " + e*" + vars[2];
    if (vars.size() == 2) return vars[0] + " + pi*" + vars[1];
    return vars.empty() ? std::string("x") : vars[0];
}

// Значение комбинации в точке траектории — зеркалит формулу ядра. Константы
// pi/euler берутся из configCUDA.h, а не набираются заново.
[[nodiscard]] static double combo_var_value(const std::vector<double>& pt, int nvars) {
    double v = !pt.empty() ? pt[0] : 0.0;
    if (nvars >= 2 && pt.size() > 1) v += ::pi    * pt[1];
    if (nvars >= 3 && pt.size() > 2) v += ::euler * pt[2];
    return v;
}

// Скретч-буфер под ImGui::InputText*: ImGui пишет в сырой char*, а хранить
// значение нам надо в std::string. Раньше КАЖДЫЙ такой виджет заводил свой
// std::vector<char> на 1–4 КБ НА КАЖДОМ КАДРЕ — на панели с полусотней полей
// это сотня heap-аллокаций в кадр впустую. Буфер один на поток и
// переиспользуется: виджеты не вложены друг в друга, а ImGui копирует
// содержимое к себе внутри вызова, поэтому дольше вызова буфер не нужен.
// Ёмкость между кадрами сохраняется, так что assign/resize уже не аллоцируют.
static std::vector<char>& input_scratch(const std::string& str, size_t extra) {
    static thread_local std::vector<char> buf;
    buf.assign(str.begin(), str.end());
    buf.resize(str.size() + extra);
    buf[str.size()] = '\0';
    return buf;
}

static bool InputTextMultilineStr(const char* label, std::string& str, const ImVec2& size) {
    std::vector<char>& buf = input_scratch(str, 4096);
    bool changed = ImGui::InputTextMultiline(label, buf.data(), buf.size(), size);
    if (changed) str = buf.data();
    return changed;
}
static bool InputTextStr(const char* label, std::string& str, float width = 0.0f) {
    std::vector<char>& buf = input_scratch(str, 1024);
    if (width > 0) ImGui::SetNextItemWidth(width);
    bool changed = ImGui::InputText(label, buf.data(), buf.size());
    if (changed) str = buf.data();
    return changed;
}

// Перехватываем символы ДО того, как ImGui положит их в буфер.
// Так замена ',' → '.' происходит в момент ввода и НЕ модифицирует строку
// между кадрами — иначе ImGui::InputText на каждом следующем кадре видит
// внешнюю подмену буфера и возвращает changed=true, отчего auto_recompute
// триггерится непрерывно при наличии запятой в поле.
static int filter_comma_to_dot(ImGuiInputTextCallbackData* data) {
    if (data->EventChar == ',') data->EventChar = '.';
    return 0;
}

// Совмещённый callback (запятая→точка + digit-step на ↑/↓) переехал в
// plot_axis.cpp::digit_step_input_callback — тем же вводом пользуется меню
// цвета серии в plot_view_2d.cpp, и держать копию в gui.cpp было нельзя.

// Проверка: парсится ли строка как число (или валидная дробь "a/b")?
// Важно: std::stod НЕ кидает на "5asdfaxcv" — он парсит ведущее "5"
// и тихо игнорирует остальное. Поэтому проверяем pos — что вся строка
// (после возможных пробелов) реально была сконвертирована.
// Пустая считается валидной (дефолт подставится дальше).
// "8/3" — валидная дробь, "2/x" / "8/0" / "8/" / "5asdfaxcv" — нет.
[[nodiscard]] static bool is_numeric_string(const std::string& s) {
    if (s.empty()) return true;

    // Полностью ли строка v сконвертирована в число (плюс trailing whitespace)?
    auto parse_complete = [](const std::string& v) -> bool {
        if (v.empty()) return false;
        try {
            size_t pos = 0;
            // Результат намеренно отбрасываем — нужен только pos (сколько
            // символов реально разобрано). Явный (void), иначе MSVC ругается
            // на проигнорированный [[nodiscard]] у std::stod.
            (void)std::stod(v, &pos);
            for (size_t i = pos; i < v.size(); ++i)
                if (!std::isspace(static_cast<unsigned char>(v[i]))) return false;
            return true;
        } catch (...) { return false; }
    };

    size_t slash = s.find('/');
    if (slash != std::string::npos) {
        std::string num = s.substr(0, slash);
        std::string den = s.substr(slash + 1);
        if (!parse_complete(num) || !parse_complete(den)) return false;
        try {
            // знаменатель не должен быть нулём
            return std::stod(den) != 0.0;
        } catch (...) { return false; }
    }
    return parse_complete(s);
}

// Is `scheme_name` one of the session's custom KRS schemes (as opposed to a
// built-in name)?
[[nodiscard]] static bool is_custom_scheme(const std::string& scheme_name,
                              const std::vector<CustomScheme>& custom_schemes) {
    for (const auto& cs : custom_schemes) if (cs.name == scheme_name) return true;
    return false;
}

// Heuristic text match: does a custom KRS body actually reference a[0]
// (the symmetry slot, same as built-in CD)? Raw C/CUDA source, not parsed
// into an AST, so this is a regex over the literal text rather than a real
// usage analysis — good enough since users write "a[0]" directly.
[[nodiscard]] static bool custom_scheme_uses_symmetry(const std::string& scheme_name,
                                         const std::vector<CustomScheme>& custom_schemes) {
    static const std::regex re(R"(a\s*\[\s*0\s*\])");
    for (const auto& cs : custom_schemes)
        if (cs.name == scheme_name) return std::regex_search(cs.body, re);
    return false;
}

// ---- Auto-labels for BD/LLE/LS configs and Parametric plot windows ----
// Каждый label регенерируется каждый кадр из текущего свипа, пока
// label_is_manual = false. Пользователь помечает как ручной, отредактировав
// поле; очистка поля возвращает в auto.

// ЕДИНСТВЕННАЯ реализация имени оси свипа. Раньше рядом жили три её копии —
// локальные лямбды в draw_bifurcation_plot / draw_lle_plot / draw_ls_plot; они
// отличались только fallback'ом при индексе НУ вне диапазона ("x" вместо
// "var"). Оставлено "var": "x" выглядит как реальное имя переменной, которого
// в системе нет.
[[nodiscard]] static std::string auto_axis_name(const std::vector<std::string>& params,
                                   const std::vector<std::string>& vars,
                                   int param_idx, bool sweep_over_var, int var_idx,
                                   bool sweep_over_h = false) {
    if (sweep_over_h) return "h";
    if (sweep_over_var) {
        if (var_idx >= 0 && var_idx < (int)vars.size()) return vars[var_idx] + " (IC)";
        return "var";
    }
    if (param_idx >= 0 && param_idx < (int)params.size()) return params[param_idx];
    return "param";
}

// Хвост подписи 1D-свипа: "<ось> [lo..hi]". Пустое поле показывается как "?".
[[nodiscard]] static std::string sweep_range_label(const std::string& axis,
                                                   const std::string& lo_text,
                                                   const std::string& hi_text) {
    const std::string lo = lo_text.empty() ? std::string("?") : lo_text;
    const std::string hi = hi_text.empty() ? std::string("?") : hi_text;
    return axis + " [" + lo + ".." + hi + "]";
}

// Автоподпись BD / LLE / LS: "<x> x <y>" в 2D-режиме, "<x> [lo..hi]" в 1D.
// Шаблон, а не общий базовый класс: конфиги независимы, связывать их
// наследованием ради одинакового набора полей было бы хуже. Раньше — три
// побайтово одинаковых тела (auto_label_bd / _lle / _ls).
template <class Cfg>
[[nodiscard]] static std::string auto_label_sweep(const Cfg& c,
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

[[nodiscard]] static std::string auto_label_window(const ParametricPlotWindow& w) {
    const char* kn = w.kind == ParametricPlotWindow::Kind::Bifurcation ? "Bifurcation"
                    : w.kind == ParametricPlotWindow::Kind::LLE ? "LLE" : "LS";
    const char* suffix = w.colored_1d ? "Colored 1D" : (w.mode_2d ? "2D" : "1D");
    return std::string(kn) + " " + suffix;
}

// У DFT1D 2D-режима нет, поэтому auto_label_sweep (он читает mode_2d и поля
// второй оси) не подходит — но хвост общий.
[[nodiscard]] static std::string auto_label_dft1d(const Dft1DConfig& c,
                                    const std::vector<std::string>& params,
                                    const std::vector<std::string>& vars) {
    return sweep_range_label(auto_axis_name(params, vars, c.param_index, c.sweep_over_var,
                                            c.var_sweep_index, c.sweep_over_h),
                             c.param_lo_text, c.param_hi_text);
}

static void refresh_auto_labels(AppModel& model) {
    for (auto& bd : model.bifurcation_session.diagrams)
        if (!bd.label_is_manual)
            bd.label = auto_label_sweep(bd, model.bifurcation_session.params, model.bifurcation_session.vars);
    for (auto& c : model.lle_session.curves)
        if (!c.label_is_manual)
            c.label = auto_label_sweep(c, model.lle_session.params, model.lle_session.vars);
    for (auto& c : model.ls_session.curves)
        if (!c.label_is_manual)
            c.label = auto_label_sweep(c, model.ls_session.params, model.ls_session.vars);
    for (auto& c : model.dft1d_session.configs)
        if (!c.label_is_manual)
            c.label = auto_label_dft1d(c, model.dft1d_session.params, model.dft1d_session.vars);
    for (auto& w : model.parametric_plot_windows)
        if (!w.label_is_manual)
            w.label = auto_label_window(w);
    for (auto& w : model.dft1d_plot_windows)
        if (!w.label_is_manual)
            w.label = "DFT 1D";
}

// Историческое имя парсера в этом файле. Тело переехало в num_parse.h —
// теперь одна реализация на весь проект (см. комментарий там же о том, из-за
// чего пять разных версий расходились). Имя сохранено: оно стоит примерно в
// сорока местах и читается по месту лучше, чем голое parse_num.
[[nodiscard]] static inline double parse_ratio_or(const std::string& v, double def) {
    // Round-trip через numb: интерфейс обязан показывать и отдавать в расчёт
    // РОВНО то число, которое увидит ядро. Сегодня numb == double и это no-op,
    // но если numb станет float — подпись оси, кламп fix-значения и привязка
    // крестика к сетке не начнут врать относительно посчитанного.
    return (double)(numb)parse_num(v, def);
}

// Целочисленные поля: Resolution, decimator, число бинов, N сетки. Раньше их
// читал std::atoi, а он молча ломается на научной записи — atoi("1e3") == 1,
// то есть введённое в Resolution "1e3" давало сетку 1x1 без единого сообщения.
// Идём через тот же parse_num, что и вещественные поля, поэтому и "64/2"
// работает. Мусор даёт 0, как и раньше у atoi (см. комментарий в num_parse.h).
[[nodiscard]] static inline int parse_int_or(const std::string& v, int def) {
    const double d = parse_num(v, (double)def);
    if (!std::isfinite(d)) return def;
    const double lo = (double)std::numeric_limits<int>::min();
    const double hi = (double)std::numeric_limits<int>::max();
    return (int)std::llround(std::max(lo, std::min(hi, d)));
}

static bool InputNumStr(const char* label, std::string& str, float width = 0.0f) {
    std::vector<char>& buf = input_scratch(str, 1024);
    if (width > 0) ImGui::SetNextItemWidth(width);
    // CallbackHistory — ↑/↓ в активном InputText, обрабатываем в digit_step_input_callback.
    // CallbackCharFilter — прежняя замена запятой на точку, тоже в нём.
    bool changed = ImGui::InputText(label, buf.data(), buf.size(),
        ImGuiInputTextFlags_CallbackCharFilter | ImGuiInputTextFlags_CallbackHistory,
        digit_step_input_callback);
    if (changed) str = buf.data();

    // Inline-предупреждение, если содержимое не парсится как число.
    // Default из engine'а (0) всё равно применится, но пользователю
    // явно сигналим, что введённое значение игнорируется.
    if (!is_numeric_string(str)) {
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.3f, 1.0f),
            "  invalid number, using default");
    }
    return changed;
}

// ============================================================================
// Общие хелперы диаграмм — единый источник истины для настроек view'ов и для
// тулбаров. До этого каждая вкладка собирала всё заново, поэтому ОДНА И ТА ЖЕ
// диаграмма отличалась между вкладками: Bif-1D в Parametric рисовал линии
// x=0/y=0, а в Custom — нет; у Basins-хитмапы не было autoscale/vmin/vmax,
// хотя у Bif-2D было; у Custom Basins не было даже Swap axes. Всё, что
// относится к «как выглядит и что умеет диаграмма типа X», живёт здесь.
// ============================================================================

// Как InputNumStr, но число живёт не в строке, а в структуре (Settings ->
// PeakConfig): текст здесь только буфер ввода, поэтому значение снимаем на
// commit'е — Enter или уход фокуса, а не на каждом нажатии клавиши. Иначе
// промежуточное "1e-" по пути к "1e-14" ушло бы в set_peak_config и дёрнуло
// перекомпиляцию ядер. Невалидный/пустой текст значение не трогает.
// Возвращает true ровно на кадре коммита.
static bool InputNumStrCommit(const char* label, std::string& text, double& value,
                              float width = 0.0f) {
    std::vector<char>& buf = input_scratch(text, 1024);
    if (width > 0) ImGui::SetNextItemWidth(width);
    if (ImGui::InputText(label, buf.data(), buf.size(),
                         ImGuiInputTextFlags_CallbackCharFilter | ImGuiInputTextFlags_CallbackHistory,
                         digit_step_input_callback))
        text = buf.data();

    // Снимаем ДО возможного TextColored: тот станет last item, и IsItem* уже
    // спрашивал бы про него.
    const bool committed = ImGui::IsItemDeactivatedAfterEdit();
    const bool valid     = !text.empty() && is_numeric_string(text);
    if (!valid)
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.3f, 1.0f),
            "  invalid number, keeping previous value");

    if (committed && valid) {
        value = parse_ratio_or(text, value);
        return true;
    }
    return false;
}

// ============================================================================
// Общие блоки панелей настроек.
//
// Ниже — единственные реализации элементов, которые до этого были скопированы
// по вкладкам Bif / LLE / LS / DFT / Basins / FastSync / Custom по 4–10 раз.
// Копии успевали разойтись (в Custom пункты НУ подписывались "IC x" вместо
// "x (IC)", у одних диаграмм был decimator, у других нет), а чинить такое
// приходилось в каждой копии отдельно.
//
// ImGui-идентификаторы всюду передаются строкой снаружи и совпадают с прежними
// побайтово — раскладка окон в imgui.ini и состояние вкладок сохраняются.
// ============================================================================

// Встроенные схемы интегрирования. Один список на весь файл: раньше этот же
// массив был выписан восемью копиями плюс девятой — для проверки конфликта
// имён пользовательских схем.
static const char* const kBuiltinSchemeNames[] = {
    "Euler", "Euler-Cromer", "Explicit Midpoint", "RK4", "DOPRI78", "CD"
};

// Имена для ImGui::Combo, которому нужен массив const char*. Указатели живут,
// пока жив исходный вектор строк, — результат используется тут же, в том же
// кадре. Раньше эти три строки были выписаны трижды (Basins и оба режима
// FastSync).
[[nodiscard]] static std::vector<const char*> c_str_list(const std::vector<std::string>& v) {
    std::vector<const char*> out;
    out.reserve(v.size());
    for (const auto& s : v) out.push_back(s.c_str());
    return out;
}

// Комбо выбора схемы: встроенные + (через разделитель) пользовательские.
// on_pick вызывается ПОСЛЕ записи имени в scheme — им пользуются Phase
// (regenerate_krs) и Custom Shared config (проброс в phase_session).
// Возвращает true, если пользователь выбрал схему в этом кадре.
static bool draw_scheme_combo(const char* label, std::string& scheme,
                              const std::vector<CustomScheme>& custom_schemes,
                              const std::function<void(const std::string&)>& on_pick = {}) {
    bool picked = false;
    auto choose = [&](const std::string& nm) {
        scheme = nm;
        if (on_pick) on_pick(nm);
        picked = true;
    };
    ImGui::SetNextItemWidth(kComboW);
    if (ImGui::BeginCombo(label, scheme.c_str())) {
        for (const char* m : kBuiltinSchemeNames)
            if (ImGui::Selectable(m, scheme == m)) choose(m);
        if (!custom_schemes.empty()) ImGui::Separator();
        for (const auto& cs : custom_schemes)
            if (ImGui::Selectable((cs.name + " (custom)").c_str(), scheme == cs.name))
                choose(cs.name);
        ImGui::EndCombo();
    }
    return picked;
}

// ЕДИНСТВЕННОЕ комбо выбора цели свипа: параметры, затем через разделитель
// начальные условия как "<var> (IC)", затем "dt (h)". Раньше — семь копий по
// ~45 строк во вкладках плюс восьмая в Custom со своим стилем подписей.
//
// other_over_h — флаг ВТОРОЙ оси (nullptr = 1D-контекст, ограничения нет).
// Ровно одна ось может свипаться по h: движок кодирует её ОДНИМ числом
// hSweepAxis (-1 нет / 0 X / 1 Y), и запрос с обеими осями по h отвергает.
// Поэтому пункт "dt (h)" здесь всегда доступен, а выбор ПЕРЕНОСИТ шаг на эту
// ось, гася флаг соседней. Раньше на одно это правило приходилось три разных
// поведения: ось X вкладок переносила, ось Y вкладок прятала пункт вообще (и
// было не понять, куда он делся), а Custom показывал его серым — то есть
// перенести шаг с оси на ось можно было только в две операции.
//
// note_when_empty — показывать ли подсказку вместо комбо, когда в системе нет
// ни параметров, ни переменных (так делают вкладки; Custom рисует комбо всегда).
static void draw_sweep_target_combo(const char* label,
                                    const std::vector<std::string>& params,
                                    const std::vector<std::string>& vars,
                                    int& par_index, bool& over_var, int& var_index,
                                    bool& over_h,
                                    bool* other_over_h = nullptr,
                                    bool note_when_empty = false,
                                    float width = kComboW) {
    if (params.empty() && vars.empty() && note_when_empty) {
        ImGui::TextDisabled("No parameters/variables (select a system first)");
        return;
    }
    // Кламп индексов: сохранённая сессия могла прийти от системы с другим
    // числом переменных/параметров.
    if (par_index < 0 || par_index >= (int)params.size()) par_index = 0;
    if (var_index < 0 || var_index >= (int)vars.size())   var_index = 0;

    const std::string preview =
          over_h                      ? std::string("dt (h)")
        : (over_var && !vars.empty()) ? (vars[var_index] + " (IC)")
        : (!params.empty())           ? params[par_index]
                                      : std::string("?");
    ImGui::SetNextItemWidth(width);
    if (!ImGui::BeginCombo(label, preview.c_str())) return;

    for (int i = 0; i < (int)params.size(); ++i) {
        const bool sel = !over_var && !over_h && par_index == i;
        if (ImGui::Selectable(params[i].c_str(), sel)) {
            par_index = i; over_var = false; over_h = false;
        }
    }
    if (!params.empty() && !vars.empty()) ImGui::Separator();
    for (int i = 0; i < (int)vars.size(); ++i) {
        const bool sel = over_var && !over_h && var_index == i;
        if (ImGui::Selectable((vars[i] + " (IC)").c_str(), sel)) {
            var_index = i; over_var = true; over_h = false;
        }
    }

    ImGui::Separator();
    if (ImGui::Selectable("dt (h)", over_h)) {
        over_h = true; over_var = false;
        if (other_over_h) *other_over_h = false;   // ровно одна ось = h
    }
    ImGui::EndCombo();
}

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

static bool draw_integration_block(const char* header,
                                   const std::string& scheme,
                                   const std::vector<CustomScheme>& custom_schemes,
                                   const IntegrationFields& f) {
    if (!ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen)) return false;
    bool changed = false;
    if (f.h) changed |= InputNumStr("h", *f.h, kFieldW);
    if (f.symmetry_s && (scheme == "CD" || custom_scheme_uses_symmetry(scheme, custom_schemes)))
        changed |= InputNumStr("symmetry s", *f.symmetry_s, kFieldW);
    if (f.t_max)       changed |= InputNumStr("computing time", *f.t_max,       kFieldW);
    if (f.transient)   changed |= InputNumStr("transient time", *f.transient,   kFieldW);
    if (f.pre_scaller) changed |= InputNumStr("decimator",      *f.pre_scaller, kFieldW);
    if (f.max_value)   changed |= InputNumStr("max value",      *f.max_value,   kFieldW);
    return changed;
}

// Секции "Initial conditions" и "Parameters" — это один и тот же блок:
// по числовому полю на каждое имя из списка. Раньше — около дюжины копий.
static void draw_named_num_fields(const char* header,
                                  const std::vector<std::string>& names,
                                  std::map<std::string, std::string>& values,
                                  const char* note = nullptr) {
    if (!ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen)) return;
    if (note) ImGui::TextDisabled("%s", note);
    for (const auto& n : names) {
        ImGui::PushID(n.c_str());
        InputNumStr(n.c_str(), values[n], kFieldW);
        ImGui::PopID();
    }
}

// Секция "CSV output": галка сохранения + путь + пояснение про побочные файлы.
static void draw_csv_output_block(const char* header, const char* checkbox_label,
                                  bool& enabled, const char* path_id, std::string& path,
                                  const char* hint) {
    if (!ImGui::CollapsingHeader(header)) return;
    ImGui::Checkbox(checkbox_label, &enabled);
    InputTextStr(path_id, path);
    ImGui::TextDisabled("%s", hint);
}

// Бокс с текстом ошибки последнего прогона: read-only, чтобы можно было
// выделить и скопировать. const_cast здесь единственный на файл — ImGui
// требует char*, но с ImGuiInputTextFlags_ReadOnly в буфер не пишет.
static void draw_error_box(const char* id, const std::string& err,
                           int lines = kErrorBoxLines) {
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error (selectable, Ctrl+C):");
    const ImVec2 sz(-1.0f, ImGui::GetTextLineHeight() * (float)lines);
    ImGui::InputTextMultiline(id, const_cast<char*>(err.c_str()), err.size() + 1,
                              sz, ImGuiInputTextFlags_ReadOnly);
}

// Пункт "Export data..." правого клика по диаграмме. Диалог выбора файла —
// платформенный, приходит из GuiCallbacks; write вызывается только с
// непустым путём (пустой = пользователь нажал Отмену).
static void draw_export_menu_item(bool busy, const GuiCallbacks& cb,
                                  const std::function<void(const std::string&)>& write) {
    if (!ImGui::MenuItem("Export data...", nullptr, false, !busy)) return;
    if (!cb.pick_save_file_csv) return;
    const std::string path = cb.pick_save_file_csv();
    if (!path.empty() && write) write(path);
}

// Inline-переименование активной вкладки. Раньше — два ручных char buf[128],
// которые молча обрезали имя на 127 символах.
static void draw_label_rename(const char* label_with_id, std::string& label,
                              float width = 200.0f) {
    ImGui::SetNextItemWidth(width);
    InputTextStr(label_with_id, label);
}

// Результат таб-бара конфигов: индекс активной вкладки и индекс той, что
// закрыли крестиком (-1 = ничего).
struct TabBarResult {
    int active    = -1;
    int to_remove = -1;
};

// Таб-бар "одна вкладка на конфиг + кнопка +". Раньше — шесть копий.
// item_id_prefix задаёт неизменную часть ID вкладки ("bd_tab_", "lle_tab_", …),
// поэтому идентификаторы совпадают с прежними побайтово.
// body == nullptr — тело вкладки пустое (DFT/Basins/FastSync рисуют настройки
// активного конфига уже после EndTabBar).
// request_select — внешний запрос выбрать вкладку (тулбар Colored 1D у Bif).
static TabBarResult draw_config_tab_bar(const char* bar_id, const char* item_id_prefix,
                                        int n, bool in_flight, int running_index,
                                        const std::function<std::string(int)>& label,
                                        const std::function<void(int)>& body,
                                        const std::function<void()>& on_add,
                                        int request_select = -1) {
    TabBarResult r;
    if (!ImGui::BeginTabBar(bar_id, ImGuiTabBarFlags_Reorderable |
                                    ImGuiTabBarFlags_AutoSelectNewTabs |
                                    ImGuiTabBarFlags_FittingPolicyScroll))
        return r;
    for (int i = 0; i < n; ++i) {
        ImGui::PushID(i);
        bool open = true;
        // ID завязан на индекс, а видимая часть — на label: пользователь видит
        // свежее имя сразу после редактирования.
        const std::string tab_id = label(i) + "###" + item_id_prefix + std::to_string(i);
        // Запрещаем закрывать вкладку, чей расчёт сейчас идёт.
        const bool can_close = !(in_flight && running_index == i);
        ImGuiTabItemFlags flags = ImGuiTabItemFlags_None;
        if (request_select == i) flags |= ImGuiTabItemFlags_SetSelected;
        if (ImGui::BeginTabItem(tab_id.c_str(), can_close ? &open : nullptr, flags)) {
            r.active = i;
            if (body) body(i);
            ImGui::EndTabItem();
        }
        if (!open) r.to_remove = i;
        ImGui::PopID();
    }
    // Кнопка "+" справа: Trailing удерживает её в конце, NoTooltip убирает
    // дефолтную подсказку.
    if (!in_flight && on_add) {
        if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing |
                                      ImGuiTabItemFlags_NoTooltip))
            on_add();
    }
    ImGui::EndTabBar();
    return r;
}

// Одна группа отметок в попапе "Run all...". У Parametric их три
// (Bifurcation / LLE / LS), у остальных вкладок — одна, без заголовка.
struct RunAllGroup {
    const char* title   = nullptr;   // заголовок секции; nullptr — без него
    const char* pick_id = "";        // неизменная часть ID чекбокса ("pbd_", …)
    int         n       = 0;
    std::function<std::string(int)> label;
    std::function<void(int)>        enqueue;   // положить элемент i в очередь
};

// Строка "Run (Ctrl+R) [| доп. кнопка] | Run all... (Ctrl+Shift+R) | (N queued)"
// вместе с попапом выбора. Раньше — четыре копии по ~60 строк.
//
// Отметки хранятся ПО popup_id: единый static сливал бы выбор между вкладками.
// Живут они снаружи if(BeginPopup) намеренно — Ctrl+Shift+R обязан пушить те же
// отметки, даже если попап ни разу не открывали (тогда отмечено всё).
//
// block_run_all передаётся отдельно от busy/no_active: вкладки блокируют
// "Run all..." ещё и при пустом списке конфигов, а Parametric — только на время
// расчёта (у него три независимых списка).
static void draw_run_and_run_all(const char* popup_id,
                                 bool busy, bool no_active, bool block_run_all,
                                 float run_w,
                                 const std::function<void()>& on_run,
                                 const std::function<void()>& extra_after_run,
                                 const std::vector<RunAllGroup>& groups,
                                 const std::function<void()>& start_queue,
                                 size_t queued) {
    const ImGuiIO& io = ImGui::GetIO();

    bool do_run = false;
    if (busy) {
        ImGui::BeginDisabled();
        ImGui::Button("Running...", ImVec2(run_w, 0));
        ImGui::EndDisabled();
    } else {
        if (no_active) ImGui::BeginDisabled();
        do_run = ImGui::Button("Run (Ctrl+R)", ImVec2(run_w, 0));
        if (no_active) ImGui::EndDisabled();
    }
    if (!busy && !no_active && io.KeyCtrl && !io.KeyShift &&
        ImGui::IsKeyPressed(ImGuiKey_R, false))
        do_run = true;
    if (do_run && on_run) on_run();

    if (extra_after_run) extra_after_run();

    static std::map<std::string, std::vector<std::vector<bool>>> picks_by_popup;
    std::vector<std::vector<bool>>& picks = picks_by_popup[popup_id];
    if (picks.size() != groups.size()) picks.assign(groups.size(), {});
    for (size_t g = 0; g < groups.size(); ++g)
        if (picks[g].size() != (size_t)groups[g].n)
            picks[g].assign((size_t)groups[g].n, true);

    auto run_all_marked = [&]() {
        for (size_t g = 0; g < groups.size(); ++g)
            for (size_t i = 0; i < picks[g].size(); ++i)
                if (picks[g][i] && groups[g].enqueue) groups[g].enqueue((int)i);
        if (start_queue) start_queue();
    };
    if (!block_run_all && io.KeyCtrl && io.KeyShift &&
        ImGui::IsKeyPressed(ImGuiKey_R, false))
        run_all_marked();

    ImGui::SameLine();
    if (block_run_all) ImGui::BeginDisabled();
    if (ImGui::Button("Run all... (Ctrl+Shift+R)")) ImGui::OpenPopup(popup_id);
    if (block_run_all) ImGui::EndDisabled();
    if (queued > 0) {
        ImGui::SameLine();
        ImGui::TextDisabled("(%zu queued)", queued);
    }
    if (ImGui::BeginPopup(popup_id)) {
        ImGui::TextDisabled("Sequential (one CUDA context).");
        for (size_t g = 0; g < groups.size(); ++g) {
            if (picks[g].empty()) continue;
            if (groups[g].title) ImGui::SeparatorText(groups[g].title);
            for (size_t i = 0; i < picks[g].size(); ++i) {
                bool v = picks[g][i];
                const std::string lbl = groups[g].label((int)i) + "###" +
                                        groups[g].pick_id + std::to_string(i);
                if (ImGui::Checkbox(lbl.c_str(), &v)) picks[g][i] = v;
            }
        }
        ImGui::Separator();
        if (ImGui::Button("All"))  { for (auto& p : picks) for (auto&& b : p) b = true;  }
        ImGui::SameLine();
        if (ImGui::Button("None")) { for (auto& p : picks) for (auto&& b : p) b = false; }
        ImGui::SameLine();
        if (ImGui::Button("Run")) {
            run_all_marked();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

// Тип 1D-диаграммы параметрического семейства. Нейтрален к вкладке — им
// пользуются и Parametric (ParametricPlotWindow::Kind), и Custom (L1Kind),
// чтобы конфигурация Plot2DView шла из ОДНОГО места.
enum class ParamPlotKind { Bifurcation, LLE, LS };

// Единая конфигурация Plot2DView под параметрический 1D-график.
// Вызывается и из Parametric, и из Custom — иначе одинаковые диаграммы
// расходятся по отрисовке.
//
// ВАЖНО: здесь задаются ТОЛЬКО свойства типа диаграммы и НИКОГДА —
// пользовательское состояние. Custom вызывает эту функцию каждый кадр, поэтому
// всё, что пользователь может переключить сам (show_legend, lock/invert осей),
// затиралось бы на следующем же кадре. Именно так ломалась галка «Show legend»
// на 1D-графиках Custom: show_legend стоял здесь. Дефолты пользовательских
// полей живут в самом Plot2DView (show_legend = true) — дублировать их тут не
// нужно и нельзя.
//
// Про show_zero_*: по X это всегда ось параметра, ноль на ней произволен →
// линии нет. По Y ноль осмыслен ТОЛЬКО у LLE/LS: λ=0 — граница хаос/порядок,
// стандартный референс в литературе, поэтому её оставляем. У Bif по Y идёт
// переменная состояния — ноль произволен, линии нет.
static void configure_param_plot_view(Plot2DView& view, ParamPlotKind kind) {
    view.pad_x       = false;   // данные вплотную к боковым рамкам
    view.show_zero_x = false;
    switch (kind) {
    case ParamPlotKind::Bifurcation:
        view.points_mode       = true;
        view.point_size_px     = 2.0f;
        view.imdraw_lines      = false;
        view.show_zero_y       = false;
        view.x_axis.name       = "parameter";
        view.y_axis.name       = "X";
        break;
    case ParamPlotKind::LLE:
    case ParamPlotKind::LS:
        view.points_mode       = false;   // непрерывная линия
        view.line_thickness_px = 1.5f;
        // glLineWidth в core-profile клампится драйвером до 1px, поэтому
        // линии идут через ImDrawList (см. Plot2DView::imdraw_lines).
        view.imdraw_lines      = true;
        view.show_zero_y       = true;    // λ=0 — граница хаос/порядок
        view.x_axis.name       = "parameter";
        view.y_axis.name       = "lambda";
        break;
    }
}

// ---- Custom point style (Bifurcation 1D scatter) ----
// ЕДИНСТВЕННАЯ реализация тулбара «Custom point style / Marker / Point size /
// Alpha» — им пользуются и Parametric (draw_bifurcation_plot), и Custom
// (Level-1D Bif-слоты), чтобы настройка вела себя одинаково во всех вкладках,
// где есть Bifurcation 1D. Состояние живёт в самой БД
// (BifurcationDiagramConfig) и персистится вместе с сессией.
// Возвращает true, если пользователь что-то изменил (вызывающий решает, надо
// ли сохранять сессию).
// Галочка стиля + параметры в выпадающем меню. Раньше параметры выкладывались
// в ту же строку через SameLine: в узком докированном окне они уезжали за
// правый край, и добраться до них можно было только расширив окно. Теперь в
// строке остаются только галочка и стрелка "▾", а параметры живут в popup'е.
//
// Popup рисуется ВСЕГДА, а не только при enabled: иначе, сняв галочку с
// открытым меню, мы бы пропустили EndPopup для уже открытого ImGui-окна.
// Возвращает true, если пользователь что-то изменил.
static bool draw_style_toolbar(const char* label, const char* id_suffix,
                               bool& enabled,
                               const std::function<bool()>& body) {
    const std::string sfx = id_suffix;
    bool changed = false;
    if (ImGui::Checkbox((std::string(label) + "##" + sfx).c_str(), &enabled))
        changed = true;

    const std::string pop_id = "##stylepop_" + sfx;
    if (enabled) {
        ImGui::SameLine(0.0f, 4.0f);
        if (ImGui::ArrowButton(("##styleopen_" + sfx).c_str(), ImGuiDir_Down))
            ImGui::OpenPopup(pop_id.c_str());
    }
    if (ImGui::BeginPopup(pop_id.c_str())) {
        if (body) changed |= body();
        ImGui::EndPopup();
    }
    return changed;
}

static bool draw_point_style_toolbar(BifurcationDiagramConfig& bd, const char* id_suffix) {
    const std::string sfx = id_suffix;
    return draw_style_toolbar("Custom point style", id_suffix, bd.custom_point_style,
        [&bd, &sfx]() {
            if (bd.point_marker < 0 || bd.point_marker >= kPointMarkerCount) bd.point_marker = 0;
            bool ch = false;
            ImGui::SetNextItemWidth(150);
            ch |= ImGui::Combo(("Marker##" + sfx).c_str(), &bd.point_marker,
                               kPointMarkerNames, kPointMarkerCount);
            ImGui::SetNextItemWidth(150);
            ch |= ImGui::SliderFloat(("Point size##" + sfx).c_str(), &bd.point_size,
                                     0.5f, 12.0f, "%.1f");
            ImGui::SetNextItemWidth(150);
            ch |= ImGui::SliderFloat(("Alpha##" + sfx).c_str(), &bd.point_alpha,
                                     0.0f, 1.0f, "%.2f");
            return ch;
        });
}

// Перенос настроек точек из конфига БД во вид. Звать ПОСЛЕ
// configure_param_plot_view (та ставит дефолтные 2px). Выключенный режим
// возвращает ровно прежний вид (marker = -1 → сплошной квадратный GL-пойнт).
static void apply_point_style(Plot2DView& view, const BifurcationDiagramConfig& bd) {
    view.point_marker  = bd.custom_point_style ? bd.point_marker : -1;
    view.point_size_px = bd.custom_point_style ? bd.point_size   : 2.0f;
}

// ---- Continuation + выбор устройства: ЕДИНЫЙ блок для Bif / LLE / LS 1D ----
// Раньше у Bif он был свой, а у LLE/LS свой — состав элементов и ограничения
// расходились. Теперь одна реализация на все три:
//   continuation — точки идут цепочкой (переносятся траектория и, у LLE/LS,
//     векторы возмущения). Требует param-свипа: IC-свип с цепочкой
//     несовместим. h-свип и log-сетка поддержаны. В 2D-режиме неприменим.
//   use_gpu — где считать. Обе ветки существуют для всех трёх анализов;
//     GPU-continuation — single-thread kernel, поэтому CPU там быстрее, и при
//     включении continuation мы переключаемся на CPU по умолчанию.
// Шаблон, а не общий базовый класс: конфиги независимы, связывать их
// наследованием ради трёх полей было бы хуже.
// blocked считает вызывающий: у Bif/LLE/LS это mode_2d || sweep_over_var, у
// DFT1D 2D-режима нет вовсе. Передавать условие снаружи проще, чем требовать
// от всех конфигов одинакового набора полей.
// device_locked — выбор устройства недоступен (2D-режим: CPU-веток для 2D нет).
template <class Cfg>
static void draw_continuation_device_block(Cfg& c, const char* id, bool blocked,
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
        // При включении continuation по умолчанию уходим на CPU: GPU-ветка
        // там однопоточная и заметно медленнее. При выключении возвращаемся на
        // GPU — в классическом свипе точки независимы и он там кратно быстрее.
        // В обе стороны это только дефолт: радио ниже остаётся активным, и
        // выбор пользователя держится, пока он снова не щёлкнет галку.
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
static void apply_snap_x(Plot2DView& view, double lo, double hi, int n) {
    view.snap_x_to_grid = true;
    if (n > 1) {
        view.snap_x_min = lo;
        view.snap_x_max = hi;
        view.snap_x_n   = n;
    } else {
        view.snap_x_n = 0;
    }
}

// Значение свипа в узле k из n — ТА ЖЕ формула, по которой движок считал эту
// точку (cont_sweep_value / getValueByIdx_log в parametric_engine.cpp).
// Раньше графики раскладывали точки линейно всегда, поэтому при log_scale
// данные уезжали относительно лог-оси: узлы лежат на 10^(l0 + (l1-l0)*t), а
// рисовались на lo + (hi-lo)*t.
// lo/hi <= 0 при log_scale — сюда может долететь живой чекбокс без
// соответствующего прогона; деградируем на линейную сетку вместо NaN
// (тот же приём, что в SnapCursorToGrid1D).
// continuation — у него СВОЯ конвенция узлов (lo + (hi-lo)*t, плюс reverse):
// она живёт в kernels/*_cont.template.cu и в cont_sweep_value
// (parametric_engine.cpp), и здесь остаётся копией — те шаблоны configCUDA.h
// не включают, свести их в одну функцию этой правкой нельзя.
// Классический свип идёт через общую с ядром ucuda_node_value (configCUDA.h).
// Раньше обе ветки считались cont-формулой, поэтому ось классической диаграммы
// и значение fix-слайдера расходились с getValueByIdx в последних битах, а на
// правом конце оси — сильнее (lerp даёт РОВНО hi, cont-форма — не обязательно).
//
// `reverse` тут авторитетнее флага: он приходит из РЕЗУЛЬТАТА
// (result.continuation_reverse), а `continuation` — из конфига, и они разъезжаются,
// если пользователь снял галочку continuation, не перезапустив расчёт. Данные в
// этом случае по-прежнему backward, и классическая ветка (она reverse не знает)
// нарисовала бы кривую зеркально. Поэтому reverse == true всегда идёт в
// cont-ветку — для не-continuation данных он и не выставляется.
[[nodiscard]] static double sweep_value_at(int k, int n, double lo, double hi,
                             bool log_scale, bool reverse, bool continuation) {
    if (continuation || reverse) {
        // Та же функция, что у cont-ядер и у cont_sweep_value — конвенция
        // continuation живёт в ucuda_node_value_cont (configCUDA.h). Гард
        // lo/hi > 0 на лог-ветке оставлен здесь: сюда может долететь живой
        // чекбокс лога без соответствующего прогона, а log10(<=0) отравил бы
        // ось NaN'ом (тот же приём, что в SnapCursorToGrid1D).
        const bool log_ok_cont = log_scale && lo > 0.0 && hi > 0.0;
        return (double)ucuda_node_value_cont(k, n, (numb)lo, (numb)hi, log_ok_cont, reverse);
    }
    if (log_scale && lo > 0.0 && hi > 0.0)
        return (double)ucuda_node_value_log(k, n, (numb)lo, (numb)hi);
    return (double)ucuda_node_value(k, n, (numb)lo, (numb)hi);
}

// Описание свипа одного члена окна — всё, что нужно для общей X-оси.
struct SweepAxisMember {
    int    kind      = 0;      // 0 = параметр, 1 = переменная (IC), 2 = шаг h
    int    index     = -1;     // индекс в params/vars (для kind==2 не нужен)
    bool   log_scale = false;
    double lo = 0.0, hi = 1.0; // диапазон, с которым РЕАЛЬНО шёл Run
};

// Общая настройка X-оси 1D-графика по членам окна:
//   - подпись = общий sweep-таргет; если члены свипают разное — "parameter";
//   - fit-диапазон = union(lo, hi) ВСЕХ членов: ось обязана охватывать весь
//     свип, даже если часть точек разошлась и на графике их нет;
//   - log_scale только если он одинаков у всех членов.
// Раньше — три почти идентичных блока по ~35 строк в Bif/LLE/LS.
static void configure_sweep_x_axis(Plot2DView& view,
                                   const std::vector<SweepAxisMember>& members,
                                   const std::vector<std::string>& params,
                                   const std::vector<std::string>& vars) {
    int  shared_kind = -2;   // -2 = ещё не видели, -1 = смешанные
    int  shared_idx  = -2;
    bool shared_log = false, log_seen = false, log_mismatch = false;
    double x_fit_lo = 0.0, x_fit_hi = 0.0;
    bool   x_fit_any = false;

    for (const auto& m : members) {
        const int kidx = (m.kind == 2) ? -1 : m.index;
        if (shared_kind == -2) { shared_kind = m.kind; shared_idx = kidx; }
        else if (shared_kind != m.kind || shared_idx != kidx) { shared_kind = -1; shared_idx = -1; }

        if (!log_seen) { shared_log = m.log_scale; log_seen = true; }
        else if (shared_log != m.log_scale) { log_mismatch = true; }

        const double a = std::min(m.lo, m.hi), b = std::max(m.lo, m.hi);
        if (!x_fit_any) { x_fit_lo = a; x_fit_hi = b; x_fit_any = true; }
        else { x_fit_lo = std::min(x_fit_lo, a); x_fit_hi = std::max(x_fit_hi, b); }
    }

    view.x_fit_use_explicit = x_fit_any;
    view.x_fit_min = x_fit_lo;
    view.x_fit_max = x_fit_hi;
    view.x_axis.log_scale = shared_log && !log_mismatch;

    if (shared_kind == 0 && shared_idx >= 0 && shared_idx < (int)params.size())
        view.x_axis.name = params[shared_idx];
    else if (shared_kind == 1 && shared_idx >= 0 && shared_idx < (int)vars.size())
        view.x_axis.name = vars[shared_idx] + " (IC)";
    else if (shared_kind == 2)
        view.x_axis.name = "h";
    else
        view.x_axis.name = "parameter";
}

// Сбор членов окна и настройка общей X-оси одним вызовом. Раньше — три почти
// одинаковых блока по ~20 строк в Bif/LLE/LS, из которых версии LLE и LS
// совпадали буквально. Отличается только источник диапазона (у Bif снапшот в
// result пишется лишь для continuation), поэтому его отдаёт range_fn.
template <class Cfg, class RangeFn>
static void configure_sweep_x_axis_from(Plot2DView& view,
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
static void sweep_range_from_result(double res_lo, double res_hi,
                                    const std::string& lo_text, const std::string& hi_text,
                                    double& lo, double& hi) {
    lo = res_lo;
    hi = res_hi;
    if (hi == lo) {
        lo = parse_ratio_or(lo_text, 0.0);
        hi = parse_ratio_or(hi_text, 1.0);
    }
}

// Snap-диапазон по ПЕРВОМУ члену окна: берём сетку успешного прогона, иначе
// парсим текстовые поля конфига (до первого Run их всё равно больше нечем
// заполнить). Раньше — три идентичных блока в Bif/LLE/LS.
static void apply_snap_x_from_config(Plot2DView& view,
                                     bool have_result, double res_lo, double res_hi, int res_n,
                                     const std::string& lo_text,
                                     const std::string& hi_text,
                                     const std::string& n_text) {
    if (have_result && res_n > 1) {
        apply_snap_x(view, res_lo, res_hi, res_n);
        return;
    }
    const int n = parse_int_or(n_text, 0);
    apply_snap_x(view, parse_ratio_or(lo_text, 0.0), parse_ratio_or(hi_text, 1.0), n);
}

// Тот же snap, но по ПЕРВОМУ члену окна. Раньше — три идентичных хвоста
// по 12 строк в draw_bifurcation_plot / draw_lle_plot / draw_ls_plot.
template <class Cfg>
static void apply_snap_x_from_first_member(Plot2DView& view,
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

// Right-click «Export data...» подменю: перечисляет ВСЕ конфиги сессии с
// готовым прогоном (не только members этого окна — экспорт не привязан к
// окну). Раньше — три идентичные копии в Bif/LLE/LS.
//   ready(i) — есть ли законченный прогон, busy(i) — считается ли сейчас,
//   do_export(i, path) — собственно запись.
static void draw_export_submenu(const char* id_tag, int n,
                                const std::function<std::string(int)>& label,
                                const std::function<bool(int)>& ready,
                                const std::function<bool(int)>& busy,
                                const std::function<void(int, const std::string&)>& do_export,
                                const GuiCallbacks& cb) {
    if (!ImGui::BeginMenu("Export data...")) return;
    bool any = false;
    for (int i = 0; i < n; ++i) {
        if (!ready(i)) continue;
        any = true;
        char item[192];
        std::snprintf(item, sizeof(item), "%s##exp_%s_%d", label(i).c_str(), id_tag, i);
        if (ImGui::MenuItem(item, nullptr, false, !busy(i))) {
            if (cb.pick_save_file_csv) {
                std::string path = cb.pick_save_file_csv();
                if (!path.empty()) do_export(i, path);
            }
        }
    }
    if (!any) ImGui::TextDisabled("(no completed runs)");
    ImGui::EndMenu();
}

// Ленивое создание per-config HeatmapView в map'е окна (map уже per-window,
// поэтому пересечений между окнами нет). При первом создании подхватывает
// colormap: приоритет у выбора, сохранённого в конфиге (cfg_colormap, -1 =
// не задан), иначе общий app-дефолт. cfg_exponent_idx применяется только если
// != kNoExponent (актуально лишь для LS).
// Раньше — четыре копии лямбды get_*_heatmap (Bif / LLE / LS / DFT1D).
static constexpr int kNoExponent = -999;
static HeatmapView& get_or_create_heatmap(
        std::map<int, std::unique_ptr<HeatmapView>>& map, int idx,
        int cfg_colormap, int app_default_colormap,
        int cfg_exponent_idx = kNoExponent) {
    auto& slot = map[idx];
    if (!slot) {
        slot = std::make_unique<HeatmapView>();
        slot->colormap = (HeatmapColormap)colormap_id_or(cfg_colormap, app_default_colormap);
        if (cfg_exponent_idx != kNoExponent) slot->display_exponent_idx = cfg_exponent_idx;
    }
    return *slot;
}

// Скретч-буферы точек одного plot-окна (по одному на серию). Между кадрами
// живут намеренно: buf.clear() у вызывающего сохраняет capacity, поэтому со
// второго кадра push_back уже не аллоцирует — на диаграммах в миллионы точек
// это заметно.
//
// Ключ — стабильный ParametricPlotWindow::id, как у кэшей HeatmapView выше.
// Раньше на каждую из трёх функций был один static, общий ВСЕМ окнам: при двух
// окнах с разным числом серий assign() передёргивал буферы каждый кадр, и
// оптимизация не работала вовсе. Корректности это не нарушало (render()
// забирает точки синхронно), но и пользы не приносило.
//
// Записи удалённых окон остаются в карте до перезапуска — как и у хитмап;
// окон единицы, и создаёт их руками пользователь.
static std::vector<std::vector<float>>& window_point_bufs(int window_id, size_t n_series) {
    static std::map<int, std::vector<std::vector<float>>> cache;
    auto& bufs = cache[window_id];
    if (bufs.size() != n_series) bufs.assign(n_series, {});
    return bufs;
}

// Полоска-превью колормапа как ImGui-виджет: kSeg сегментов с линейным
// градиентом внутри каждого, плюс Dummy того же размера, чтобы полоска
// занимала место в layout'е. 24 сегмента хватает, чтобы даже рваные
// качественные карты (tab20, glasbey) читались, а рисуется это только для
// видимых строк — и в попапе пикера, и под клиппером в Settings.
static void colormap_strip_item(int id, float w, float h) {
    constexpr int kSeg = 24;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    for (int i = 0; i < kSeg; ++i) {
        const float t0 = (float)i / kSeg, t1 = (float)(i + 1) / kSeg;
        const ImU32 c0 = cmap_sample_id(t0, id);
        const ImU32 c1 = cmap_sample_id(t1, id);
        // +0.5 к правой границе — иначе между сегментами проступают щели
        // на дробном UI-скейле.
        dl->AddRectFilledMultiColor(ImVec2(p.x + w * t0, p.y),
                                    ImVec2(p.x + w * t1 + 0.5f, p.y + h),
                                    c0, c1, c1, c0);
    }
    dl->AddRect(p, ImVec2(p.x + w, p.y + h), ImGui::GetColorU32(ImGuiCol_Border));
    ImGui::Dummy(ImVec2(w, h));
}

// Пикер колормапа: BeginCombo вместо ImGui::Combo, потому что список теперь
// разреженный (только отмеченные пользователем карты slanCM) и в каждой
// строке рисуется градиент. Текущий id показывается всегда, даже если карта
// снята галочкой в Settings — иначе открытая сессия молча переехала бы на
// другую карту. Возвращает true, если выбор изменился.
static bool colormap_combo(const char* label, int* id, float width) {
    bool changed = false;
    const float h = ImGui::GetTextLineHeight();
    const float strip_w = h * 4.0f;

    ImGui::SetNextItemWidth(width);
    // HeightLarge = 20 строк вместо дефолтных 8: набор карт пользователь
    // набирает сам, но и десяток в попапе размером с восемь строк листать
    // неудобно.
    if (!ImGui::BeginCombo(label, colormap_label(*id), ImGuiComboFlags_HeightLarge))
        return false;

    const std::vector<int>& ids = picker_colormap_ids();
    // Показываем текущую карту первой, если её нет во включённом наборе.
    const bool orphan = std::find(ids.begin(), ids.end(), *id) == ids.end();
    auto row = [&](int cid) {
        ImGui::PushID(cid);
        const bool sel = (cid == *id);
        // Selectable без явной ширины растягивается на всю строку — он служит
        // подложкой-хитбоксом, а градиент и подпись рисуются поверх. SameLine()
        // после него встал бы у ПРАВОГО края (offset_from_start_x==0 = «после
        // предыдущего элемента»), поэтому возвращаем курсор в начало строки.
        const ImVec2 p0 = ImGui::GetCursorPos();
        if (ImGui::Selectable("##pick", sel, 0, ImVec2(0, h))) {
            if (!sel) { *id = cid; changed = true; }
        }
        if (sel) ImGui::SetItemDefaultFocus();
        ImGui::SetCursorPos(p0);
        colormap_strip_item(cid, strip_w, h);
        ImGui::SameLine(0.0f, ImGui::GetStyle().ItemSpacing.x);
        ImGui::TextUnformatted(colormap_label(cid));
        ImGui::PopID();
    };
    if (orphan) { row(*id); ImGui::Separator(); }
    for (int cid : ids) row(cid);
    // Все галочки сняты — пикер пуст. Текущая карта показана выше отдельной
    // строкой (orphan), так что состояние не тупиковое, но подсказать надо.
    if (ids.empty())
        ImGui::TextDisabled("(no colormaps ticked — pick some in Settings)");

    ImGui::EndCombo();
    return changed;
}

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

// ЕДИНСТВЕННАЯ реализация тулбара «Colormap / Autoscale / vmin / vmax /
// Swap axes» — раньше он был скопирован в 7 мест с разным составом
// элементов. Всё состояние живёт в самом HeatmapView, поэтому вызывающему
// достаточно передать, куда персистить colormap.
// Возвращает true, если пользователь что-то изменил.
static bool draw_heatmap_toolbar(HeatmapView& hv, const HeatmapToolbarOpts& o = {}) {
    bool changed = false;

    int cmap_idx = (int)hv.colormap;
    if (colormap_combo("Colormap", &cmap_idx, ImGui::GetFontSize() * 12.0f)) {
        hv.colormap = (HeatmapColormap)cmap_idx;
        if (o.persist_colormap) o.persist_colormap(cmap_idx);
        changed = true;
    }

    if (o.extras) o.extras();

    ImGui::SameLine();
    if (ImGui::Checkbox("Autoscale color", &hv.autoscale)) changed = true;
    if (!hv.autoscale) {
        ImGui::SameLine();
        if (InputNumStr("vmin", hv.manual_vmin_text, 80)) changed = true;
        hv.manual_vmin = (float)parse_ratio_or(hv.manual_vmin_text, hv.manual_vmin);
        ImGui::SameLine();
        if (InputNumStr("vmax", hv.manual_vmax_text, 80)) changed = true;
        hv.manual_vmax = (float)parse_ratio_or(hv.manual_vmax_text, hv.manual_vmax);
    }

    if (o.show_swap) {
        ImGui::SameLine();
        if (ImGui::Button(hv.swap_axes ? "Swap axes (on)" : "Swap axes")) {
            hv.swap_axes = !hv.swap_axes;
            changed = true;
        }
    }

    if (o.extras_tail) o.extras_tail();
    return changed;
}

// LS exponent picker: λ1..λN + "sum L_i" (sentinel -1, а не N — чтобы выбор не
// «съезжал», если N поменяется на следующем Run). Выбор живёт в HeatmapView, а
// не в конфиге: два окна с одной и той же кривой не должны дёргать друг у
// друга индекс. on_pick — персистентность на стороне вызывающего.
// Раньше — две почти идентичные копии (draw_ls_plot и Custom-слот 2).
static void draw_ls_exponent_picker(HeatmapView& hv, int n_exponents,
                                    const std::function<void(int)>& on_pick) {
    if (n_exponents <= 0) return;
    const int N = n_exponents;
    if (hv.display_exponent_idx != -1 &&
        (hv.display_exponent_idx < 0 || hv.display_exponent_idx >= N))
        hv.display_exponent_idx = 0;

    const std::string preview = (hv.display_exponent_idx == -1)
        ? "sum L_i" : ("L" + std::to_string(hv.display_exponent_idx + 1));
    auto pick = [&](int j) {
        hv.display_exponent_idx = j;
        if (on_pick) on_pick(j);
    };
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::BeginCombo("Exponent", preview.c_str())) {
        for (int j = 0; j < N; ++j) {
            std::string lbl = "L" + std::to_string(j + 1);
            if (ImGui::Selectable(lbl.c_str(), hv.display_exponent_idx == j)) pick(j);
        }
        ImGui::Separator();
        if (ImGui::Selectable("sum L_i", hv.display_exponent_idx == -1)) pick(-1);
        ImGui::EndCombo();
    }
}

// Выбор отображаемой плоскости LS-2D по индексу экспоненты `k`.
//   k >= 0  — готовая плоскость из result_2d.values (без копирования),
//   k == -1 — "sum L_i": поэлементная сумма всех N плоскостей, считается ОДИН
//             раз на data_generation_2d и кэшируется в самом конфиге (кэш общий
//             для всех окон с этой кривой). Diverged-ячейки (flags<0) получают
//             sentinel 999.0 — тот же, что engine пишет в одиночные плоскости,
//             чтобы HeatmapView закрасил их тем же серым.
// gen мешает поколение чанка с индексом экспоненты, иначе переключение
// экспоненты не перезалило бы текстуру.
// Раньше — две копии по ~40 строк (draw_ls_plot и draw_custom_plot_windows).
static void ls_resolve_plane(LSCurveConfig& cact, int k,
                             const double*& plane, double& vmin, double& vmax, int& gen) {
    const size_t plane_size = (size_t)cact.result_2d.n_pts * (size_t)cact.result_2d.n_pts;
    const int    N          = cact.result_2d.n_exponents;

    if (k == -1) {
        if (cact.sum_cache_gen != cact.data_generation_2d) {
            cact.sum_cache.assign(plane_size, 0.0);
            double smin =  std::numeric_limits<double>::infinity();
            double smax = -std::numeric_limits<double>::infinity();
            for (size_t c2 = 0; c2 < plane_size; ++c2) {
                // Всё, что не колебательный режим (FP/unbound) — под sentinel:
                // раньше проверялось flags < 0, но у LS unbound теперь 0.
                if (c2 < cact.result_2d.flags.size() &&
                    !regime_is_oscillation(cact.result_2d.flags[c2])) {
                    cact.sum_cache[c2] = kSentinelNoData;
                    continue;
                }
                double sum = 0.0;
                for (int j = 0; j < N; ++j)
                    sum += cact.result_2d.values[(size_t)j * plane_size + c2];
                cact.sum_cache[c2] = sum;
                if (sum < smin) smin = sum;
                if (sum > smax) smax = sum;
            }
            cact.sum_cache_min = std::isfinite(smin) ? smin : 0.0;
            cact.sum_cache_max = std::isfinite(smax) ? smax : 0.0;
            cact.sum_cache_gen = cact.data_generation_2d;
        }
        plane = cact.sum_cache.data();
        vmin  = cact.sum_cache_min;
        vmax  = cact.sum_cache_max;
        gen   = cact.data_generation_2d * 64 + N;
    } else {
        plane = cact.result_2d.values.data() + (size_t)k * plane_size;
        vmin  = (k >= 0 && k < (int)cact.result_2d.min_val.size()) ? cact.result_2d.min_val[k] : 0.0;
        vmax  = (k >= 0 && k < (int)cact.result_2d.max_val.size()) ? cact.result_2d.max_val[k] : 0.0;
        gen   = cact.data_generation_2d * 64 + k;
    }
}

// ============================================================
// Вкладка System: ввод системы, методы, генерация кода
// ============================================================
static void draw_system_tab(AppModel& model, const GuiCallbacks& cb) {
    model.poll(); // забрать результат OCR, если готов

    // режим ввода
    ImGui::Text("Input mode:");
    ImGui::SameLine();
    int mode = (int)model.mode;
    ImGui::RadioButton("Image", &mode, (int)InputMode::Image); ImGui::SameLine();
    ImGui::RadioButton("LaTeX", &mode, (int)InputMode::Latex); ImGui::SameLine();
    ImGui::RadioButton("Plain", &mode, (int)InputMode::Plain);
    model.mode = (InputMode)mode;
    ImGui::Separator();

    // источник картинки
    if (model.mode == InputMode::Image) {
        if (ImGui::Button("Choose image file...")) {
            if (cb.pick_image_file) {
                std::string path = cb.pick_image_file();
                if (!path.empty()) model.start_ocr(std::make_unique<FileImageSource>(path));
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Paste from clipboard"))
            model.start_ocr(std::make_unique<ClipboardImageSource>());
        if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_V, false))
            model.start_ocr(std::make_unique<ClipboardImageSource>());
        ImGui::SameLine();
        switch (model.ocr_state()) {
        case OcrState::Running: ImGui::TextColored(ImVec4(1, 1, 0, 1), "Recognizing..."); break;
        case OcrState::Done:    ImGui::TextColored(ImVec4(0, 1, 0, 1), "Recognized"); break;
        case OcrState::Failed:  ImGui::TextColored(ImVec4(1, 0, 0, 1), "OCR failed: %s", model.ocr_error().c_str()); break;
        default: ImGui::TextDisabled("(no image)"); break;
        }
        ImGui::TextDisabled("Tip: Win+Shift+S to snip, then Ctrl+V or 'Paste from clipboard'.");
    }

    // поле ввода
    if (model.mode == InputMode::Image || model.mode == InputMode::Latex) {
        ImGui::Text("LaTeX (editable - fix OCR errors here):");
        InputTextMultilineStr("##latex", model.latex_text, ImVec2(-1, 90));
        if (ImGui::CollapsingHeader("LaTeX format examples")) {
            ImGui::TextDisabled(
                "Each equation on its own line, LHS must have a derivative:\n"
                "  \\dot{x} = \\sigma(y-x) \\\\\n  \\dot{y} = x(\\rho-z)-y\n"
                "Supported: \\frac{a}{b}, x^{2}, \\sin x, \\sin^{2} x, \\cdot, |x|,\n"
                "  subscripts x_{m}, greek \\sigma. Derivatives: \\dot{x}, x', dx/dt.");
        }
    }
    else {
        ImGui::Text("Equations (plain syntax):");
        InputTextMultilineStr("##plain", model.plain_text, ImVec2(-1, 90));
        if (ImGui::CollapsingHeader("Plain format examples")) {
            ImGui::TextDisabled(
                "  \\dot{x} = sigma*(y - x) \\\\\n  \\dot{y} = x*(rho - z) - y\n"
                "Use * for multiplication, ^ for powers. LHS needs \\dot{x}= or x'=.");
        }
    }

    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

    // вспомогательные функции
    ImGui::Checkbox("Use auxiliary functions", &model.use_aux_funcs);
    if (model.use_aux_funcs) {
        ImGui::Text("Function definitions (one per line, e.g. h(x) = m_1 x + ...):");
        InputTextMultilineStr("##funcs", model.func_defs_text, ImVec2(-1, 60));
        if (ImGui::CollapsingHeader("Auxiliary function examples")) {
            ImGui::TextDisabled(
                "h(x) = m_1 x + \\frac{1}{2}(m_0-m_1)(|x+1| - |x-1|)\n"
                "Then call h(x) in equations. Body is inlined.\n"
                "IMPORTANT: function params (m_0, m_1) must be in the alphabet too.");
        }
    }

    // ----- Variables / Parameters (новый раздельный формат) -----
    ImGui::Text("Variables:");
    ImGui::SameLine();
    if (ImGui::SmallButton("Auto-detect")) {
        // Scan latex_text (or plain_text если режим Plain) и заполнить vars/params.
        const std::string& src = (model.mode == InputMode::Plain)
                                 ? model.plain_text
                                 : model.latex_text;
        DetectedAlphabet det = detect_alphabet(src);
        auto join = [](const std::vector<std::string>& v) {
            std::string out;
            for (size_t i = 0; i < v.size(); ++i) {
                if (i) out += ", ";
                out += v[i];
            }
            return out;
        };
        model.vars_text   = join(det.vars);
        model.params_text = join(det.params);
    }
    ImGui::TextDisabled("Comma-sep, e.g. x,y,z. These are X[0..N-1] in KRS.");
    InputTextStr("##vars_text", model.vars_text);

    ImGui::Text("Parameters:");
    ImGui::TextDisabled("Comma-sep, e.g. sigma,rho,beta. These are a[1..M] in KRS.");
    InputTextStr("##params_text", model.params_text);

    // ----- Legacy: единый алфавит -----
    if (ImGui::CollapsingHeader("Legacy: single alphabet field")) {
        ImGui::TextDisabled("Used by older systems where vars/params live in one list.\n"
                            "If Variables AND Parameters above are both filled, this is ignored.");
        InputTextStr("##alphabet", model.alphabet_text);
    }

    // порядок параметров
    ImGui::Text("Parameter order in a[]:");
    ImGui::SameLine();
    int porder = (int)model.param_order;
    ImGui::RadioButton("as in alphabet", &porder, (int)ParamOrder::AsInAlphabet); ImGui::SameLine();
    ImGui::RadioButton("as in system", &porder, (int)ParamOrder::AsInSystem);
    model.param_order = (ParamOrder)porder;

    ImGui::Separator();

    // методы
    ImGui::Text("Schemes to generate:");
    if (ImGui::Button("Select all")) model.scheme_euler = model.scheme_cromer = model.scheme_midpoint = model.scheme_rk4 = model.scheme_dopri78 = model.scheme_cd = true;
    ImGui::SameLine();
    if (ImGui::Button("Clear all"))  model.scheme_euler = model.scheme_cromer = model.scheme_midpoint = model.scheme_rk4 = model.scheme_dopri78 = model.scheme_cd = false;
    ImGui::Checkbox("Euler", &model.scheme_euler); ImGui::SameLine();
    ImGui::Checkbox("Euler-Cromer", &model.scheme_cromer); ImGui::SameLine();
    ImGui::Checkbox("Explicit Midpoint", &model.scheme_midpoint); ImGui::SameLine();
    ImGui::Checkbox("RK4", &model.scheme_rk4); ImGui::SameLine();
    ImGui::Checkbox("DOPRI78", &model.scheme_dopri78); ImGui::SameLine();
    ImGui::Checkbox("CD", &model.scheme_cd);

    // ----- Custom KRS schemes (raw C/CUDA код вместо codegen) -----
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Custom KRS schemes",
        model.custom_schemes.empty() ? 0 : ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextDisabled(
            "Raw C/CUDA in calculateDiscreteModel body. Available: X[0..N-1] (vars),\n"
            "a[0] (symmetry s, same slot as CD), a[1..M] (params), h (step), AMOUNTOFX.\n"
            "if/for/while + math functions OK.\n"
            "Runs on GPU (NVRTC) and on CPU (compiled with cl.exe from Visual Studio).");

        // существующие схемы
        int to_delete = -1;
        for (int i = 0; i < (int)model.custom_schemes.size(); ++i) {
            auto& cs = model.custom_schemes[i];
            ImGui::PushID(i);
            ImGui::SetNextItemWidth(220);
            std::string name_label = "name##cs_name_" + std::to_string(i);
            InputTextStr(name_label.c_str(), cs.name);
            ImGui::SameLine();
            if (ImGui::SmallButton("Delete")) to_delete = i;
            std::string body_label = "##cs_body_" + std::to_string(i);
            InputTextMultilineStr(body_label.c_str(), cs.body, ImVec2(-1, 100));
            // Проверка обращений X[k] / a[k] с константным индексом против
            // размерности ТЕКУЩЕЙ системы. Статическая и живая — тела короткие,
            // проход по строке стоит копейки. Пока система не распознана
            // (known_vars пуст) молчим, иначе выдали бы ложные ошибки на всё.
            if (!model.known_vars.empty()) {
                std::vector<KrsCpuDiag> diags;
                krs_cpu_check_indices(cs.body,
                                      (int)model.known_vars.size(),
                                      (int)model.known_params.size() + 1,
                                      diags);
                for (const auto& d : diags)
                    ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "  line %d: %s",
                                       d.line, d.message.c_str());
            }
            ImGui::Spacing();
            ImGui::PopID();
        }
        if (to_delete >= 0) model.custom_schemes.erase(model.custom_schemes.begin() + to_delete);

        // блокируем добавление с уже существующим/built-in именем
        // (список встроенных — общий kBuiltinSchemeNames, см. верх файла)
        if (ImGui::Button("+ Add custom scheme")) {
            // подобрать уникальное имя "Custom N"
            int n = (int)model.custom_schemes.size() + 1;
            std::string candidate;
            auto name_clash = [&](const std::string& nm) {
                for (const char* b : kBuiltinSchemeNames) if (nm == b) return true;
                for (const auto& cs : model.custom_schemes) if (cs.name == nm) return true;
                return false;
            };
            do { candidate = "Custom " + std::to_string(n++); } while (name_clash(candidate));
            CustomScheme cs; cs.name = candidate;
            model.custom_schemes.push_back(std::move(cs));
        }

        // подсветка конфликтов
        for (const auto& cs : model.custom_schemes) {
            for (const char* b : kBuiltinSchemeNames) {
                if (cs.name == b) {
                    ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1),
                        "  '%s' conflicts with a built-in scheme name; rename it.",
                        cs.name.c_str());
                    break;
                }
            }
        }
        // дубликаты между custom
        for (size_t i = 0; i < model.custom_schemes.size(); ++i) {
            for (size_t j = i + 1; j < model.custom_schemes.size(); ++j) {
                if (!model.custom_schemes[i].name.empty() &&
                    model.custom_schemes[i].name == model.custom_schemes[j].name) {
                    ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1),
                        "  duplicate custom name '%s'.",
                        model.custom_schemes[i].name.c_str());
                }
            }
        }
    }

    if (ImGui::Button("Generate")) model.generate();
    if (!model.error_message.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "Error: %s", model.error_message.c_str());
    }

    if (!model.generated_code.empty()) {
        ImGui::Separator();
        ImGui::Text("Generated code:");
        if (ImGui::Button("Copy")) {
            if (cb.set_clipboard_text) cb.set_clipboard_text(model.generated_code);
        }
        ImGui::InputTextMultiline("##code",
            (char*)model.generated_code.c_str(), model.generated_code.size() + 1,
            ImVec2(-1, 220), ImGuiInputTextFlags_ReadOnly);
    }
}

// ============================================================
// Вкладка Parameters: НУ, значения/диапазоны параметров, шаг
// ============================================================
static void draw_parameters_tab(AppModel& model) {
    ImGui::TextDisabled("Parameter fields appear after parsing the system.");
    if (ImGui::Button("Refresh from system")) {
        model.refresh_symbols();
    }
    if (!model.error_message.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "%s", model.error_message.c_str());
    }
    ImGui::Separator();

    // шаг дискретизации
    ImGui::Text("Discretization step h:");
    ImGui::SameLine();
    InputNumStr("##step_h", model.step_h, kFieldW);
    ImGui::TextDisabled("(leave empty to skip)");

    ImGui::Spacing();

    // начальные условия
    if (!model.known_vars.empty()) {
        ImGui::SeparatorText("Initial conditions");
        for (const auto& v : model.known_vars) {
            ImGui::Text("%s(0) =", v.c_str());
            ImGui::SameLine();
            std::string id = "##ic_" + v;
            InputNumStr(id.c_str(), model.init_conditions[v], 140);
        }
    }

    ImGui::Spacing();

    // параметры: значение
    if (!model.known_params.empty()) {
        ImGui::SeparatorText("Parameters (value)");
        if (ImGui::BeginTable("params", 2, ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableSetupColumn("name");
            ImGui::TableSetupColumn("value");
            ImGui::TableHeadersRow();
            for (const auto& p : model.known_params) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("%s", p.c_str());
                ImGui::TableSetColumnIndex(1);
                { std::string id = "##val_" + p; InputNumStr(id.c_str(), model.param_values[p], 100); }
            }
            ImGui::EndTable();
        }
        ImGui::TextDisabled("Empty fields are left unset.");
    }

    if (model.known_vars.empty() && model.known_params.empty()) {
        ImGui::Spacing();
        ImGui::TextDisabled("No symbols yet. Enter a system and press 'Refresh from system'.");
    }
}

// ============================================================
// Library tab: two states — list (table of saved systems + note preview)
// and editor (System + Parameters sub-tabs with Save/Cancel). The editor
// edits AppModel::library_edit_buffer (a scratch AppModel), never the live
// model, so browsing/editing/Cancel never affects the system currently
// active in Parametric/Phase/Basins/FastSync.
// ============================================================

// Trim leading/trailing ASCII whitespace.
[[nodiscard]] static std::string trim_copy(const std::string& s) {
    auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// ---- Кэш списка систем для вкладки Library ----
// Раньше draw_library_list КАЖДЫЙ КАДР обходил каталог (lib.list()) и читал с
// диска system.json КАЖДОЙ системы — только чтобы показать бейдж "[N custom]"
// и подобрать ширину колонки Name. При двух десятках систем и 60 FPS это больше
// тысячи разборов JSON в секунду.
//
// Файлы библиотеки никто не правит извне при работающем приложении, поэтому
// хватает редкого опроса. Всё, что меняет библиотеку изнутри (Save, Delete,
// Duplicate, Rename), сбрасывает кэш немедленно: иначе удалённая система висела
// бы в таблице до десяти секунд, и это читалось бы как зависание.
//
// Заодно сюда переехал кэш превью заметки — он жил рядом отдельными static'ами
// и НЕ сбрасывался после сохранения, поэтому отредактированная заметка
// показывалась старой, пока не выберешь другую строку и не вернёшься.
struct LibraryListCache {
    std::vector<std::string> names;
    std::vector<int>         custom_counts;   // параллелен names
    std::string              note_name;       // для какой системы прочитана заметка
    std::string              note_text;
    double                   filled_at = -1.0;
    bool                     valid     = false;
};
static LibraryListCache g_library_cache;

static void invalidate_library_cache() {
    g_library_cache.valid = false;
    g_library_cache.note_name.clear();
    g_library_cache.note_text.clear();
}

static const LibraryListCache& library_list_cached(SystemLibrary& lib) {
    const double now = ImGui::GetTime();
    if (g_library_cache.valid && now - g_library_cache.filled_at < kLibraryCacheTtlSec)
        return g_library_cache;

    g_library_cache.names = lib.list();
    g_library_cache.custom_counts.assign(g_library_cache.names.size(), 0);
    for (size_t i = 0; i < g_library_cache.names.size(); ++i) {
        try {
            g_library_cache.custom_counts[i] =
                (int)lib.load(g_library_cache.names[i]).custom_schemes.size();
        } catch (...) {}
    }
    g_library_cache.filled_at = now;
    g_library_cache.valid     = true;
    return g_library_cache;
}

// Save-side logic for the editor: validate name, handle rename for
// EditExisting, persist the scratch buffer via SystemLibrary. Only mirrors
// the save into the live model if the edited system is the one currently
// active elsewhere — editing/saving any other (inactive) system must not
// disturb what Parametric/Phase/Basins/FastSync currently have loaded. On
// success returns to list mode; on failure sets model.edit_error and stays
// in the editor.
static void library_editor_save(AppModel& model, SystemLibrary& lib) {
    AppModel& buf = *model.library_edit_buffer;
    std::string name = trim_copy(buf.name);
    if (name.empty()) {
        model.edit_error = "Name is required.";
        return;
    }
    buf.name = name;

    try {
        if (model.library_edit_mode == AppModel::LibraryEditMode::EditExisting) {
            const bool renaming = (name != model.edit_original_name);
            if (renaming) {
                if (lib.exists(name)) {
                    model.edit_error = "Name '" + name + "' already exists.";
                    return;
                }
                if (!lib.rename(model.edit_original_name, name)) {
                    model.edit_error = "Rename failed.";
                    return;
                }
            }
        } else {
            // AddNew
            if (lib.exists(name)) {
                model.edit_error = "Name '" + name + "' already exists.";
                return;
            }
        }
        buf.name = lib.save(buf.to_record());
        invalidate_library_cache();   // список и превью заметки устарели

        const bool editing_active_system =
            model.library_edit_mode == AppModel::LibraryEditMode::EditExisting &&
            !model.loaded_name.empty() &&
            model.edit_original_name == model.loaded_name;
        if (editing_active_system) {
            model.from_record(buf.to_record());
            model.loaded_name = model.name;
            // Push live custom_schemes / sys / vars / params into every session
            // so the next Run in Parametric/Basins/FastSync/Phase picks up the
            // edited state without reopening the tab.
            model.propagate_to_sessions();
            model.generate();
        }

        model.edit_error.clear();
        model.library_edit_mode = AppModel::LibraryEditMode::None;
    }
    catch (const std::exception& e) {
        model.edit_error = e.what();
    }
}

// List state: table of saved systems + Add-new + note preview panel.
// Interacting with this view (row select, Edit, Add new) only ever touches
// model.library_edit_buffer — never the live model — so it never changes
// the system currently active in Parametric/Phase/Basins/FastSync.
static void draw_library_list(AppModel& model, SystemLibrary& lib) {
    if (ImGui::Button("Add new system")) {
        model.library_edit_buffer->clear();
        model.edit_original_name.clear();
        model.edit_error.clear();
        model.library_edit_mode = AppModel::LibraryEditMode::AddNew;
    }
    ImGui::Separator();

    // Список и счётчики custom-схем — из кэша (см. library_list_cached):
    // раньше здесь был обход каталога и чтение всех system.json на каждом кадре.
    const LibraryListCache&         cache     = library_list_cached(lib);
    const std::vector<std::string>& names     = cache.names;
    const std::vector<int>&         cs_counts = cache.custom_counts;
    if (names.empty()) {
        ImGui::TextDisabled("(library is empty)");
        return;
    }

    // Delete confirmation flow: row buttons only set pending_delete + a
    // one-shot open flag; OpenPopup/BeginPopupModal are called at a stable
    // ID-stack level below the table so the popup id stays consistent.
    static std::string pending_delete;
    static bool        want_open_confirm = false;

    // Name column auto-sized to the longest (name [+ "[N custom]" badge]),
    // instead of proportional stretch, so short names don't waste space.
    float name_col_w = ImGui::CalcTextSize("Name").x;
    for (size_t i = 0; i < names.size(); ++i) {
        float w = ImGui::CalcTextSize(names[i].c_str()).x;
        if (cs_counts[i] > 0) {
            char badge[32];
            std::snprintf(badge, sizeof(badge), " [%d custom]", cs_counts[i]);
            w += ImGui::GetStyle().ItemSpacing.x + ImGui::CalcTextSize(badge).x;
        }
        if (w > name_col_w) name_col_w = w;
    }
    name_col_w += ImGui::GetStyle().CellPadding.x * 2.0f + 8.0f;

    if (ImGui::BeginTable("libtbl", 4,
        ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_SizingFixedFit |
        ImGuiTableFlags_NoHostExtendX)) {
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, name_col_w);
        ImGui::TableSetupColumn("",     ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("",     ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("",     ImGuiTableColumnFlags_WidthFixed);

        for (size_t row = 0; row < names.size(); ++row) {
            const std::string& n = names[row];
            ImGui::TableNextRow();
            ImGui::PushID(n.c_str());

            ImGui::TableSetColumnIndex(0);
            bool selected = (model.library_selected_name == n);
            if (ImGui::Selectable(n.c_str(), selected)) {
                model.library_selected_name = n;
            }
            if (cs_counts[row] > 0) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.6f, 0.85f, 1.0f, 1.0f),
                    " [%d custom]", cs_counts[row]);
            }

            ImGui::TableSetColumnIndex(1);
            if (ImGui::SmallButton("Edit")) {
                try {
                    model.library_edit_buffer->from_record(lib.load(n));
                    model.edit_original_name = n;
                    model.edit_error.clear();
                    model.library_edit_mode = AppModel::LibraryEditMode::EditExisting;
                }
                catch (const std::exception& e) { model.error_message = e.what(); }
            }

            ImGui::TableSetColumnIndex(2);
            if (ImGui::SmallButton("Duplicate")) {
                try { lib.duplicate(n); invalidate_library_cache(); }
                catch (const std::exception& e) { model.error_message = e.what(); }
            }

            ImGui::TableSetColumnIndex(3);
            if (ImGui::SmallButton("Delete")) {
                pending_delete = n;
                want_open_confirm = true;
            }

            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    if (want_open_confirm) {
        ImGui::OpenPopup("Delete system?");
        want_open_confirm = false;
    }

    if (ImGui::BeginPopupModal("Delete system?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Delete \"%s\"?", pending_delete.c_str());
        ImGui::TextDisabled("The system folder (including sessions/) will be removed.");
        ImGui::Separator();
        if (ImGui::Button("Delete", ImVec2(120, 0))) {
            lib.remove(pending_delete);
            invalidate_library_cache();
            if (model.library_selected_name == pending_delete)
                model.library_selected_name.clear();
            pending_delete.clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            pending_delete.clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    // Note preview panel for the selected row. Cached by name to avoid a
    // disk read every frame.
    ImGui::Separator();
    if (model.library_selected_name.empty()) {
        ImGui::TextDisabled("Click a row to preview its note.");
    } else {
        // Заметка читается один раз на выбранную систему и живёт в общем кэше
        // (см. LibraryListCache) — сохранение в редакторе его сбрасывает,
        // поэтому отредактированный текст показывается сразу.
        if (g_library_cache.note_name != model.library_selected_name) {
            try { g_library_cache.note_text = lib.load(model.library_selected_name).note; }
            catch (...) { g_library_cache.note_text.clear(); }
            g_library_cache.note_name = model.library_selected_name;
        }
        ImGui::Text("Selected: %s", model.library_selected_name.c_str());
        if (g_library_cache.note_text.empty()) {
            ImGui::TextDisabled("(no note)");
        } else {
            ImGui::TextWrapped("%s", g_library_cache.note_text.c_str());
        }
    }
}

// Editor state: name/note header + System/Parameters sub-tabs + Save/Cancel.
// Edits go into model.library_edit_buffer, not model itself — Cancel simply
// drops the edit_mode without needing to restore anything on the live model.
static void draw_library_editor(AppModel& model, SystemLibrary& lib,
                                const GuiCallbacks& cb) {
    AppModel& buf = *model.library_edit_buffer;
    const bool add_new = (model.library_edit_mode == AppModel::LibraryEditMode::AddNew);
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
                       add_new ? "New system" : "Editing");
    if (!add_new) {
        ImGui::SameLine();
        ImGui::TextDisabled("(original: %s)", model.edit_original_name.c_str());
    }

    ImGui::Spacing();
    ImGui::Text("Name:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-1);
    InputTextStr("##editor_name", buf.name);
    ImGui::Text("Note:");
    InputTextMultilineStr("##editor_note", buf.note, ImVec2(-1, 60));

    if (!model.edit_error.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f),
                           "%s", model.edit_error.c_str());
    }

    ImGui::Separator();

    if (ImGui::BeginTabBar("editor_tabs")) {
        if (ImGui::BeginTabItem("System")) {
            draw_system_tab(buf, cb);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Parameters")) {
            draw_parameters_tab(buf);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::Separator();

    const bool save_clicked = ImGui::Button("Save", ImVec2(120, 0));
    const bool save_shortcut = ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S, false);
    if (save_clicked || save_shortcut) {
        library_editor_save(model, lib);
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(120, 0))) {
        model.edit_error.clear();
        model.library_edit_mode = AppModel::LibraryEditMode::None;
    }
}


// ============================================================
// РЕЖИМ АНАЛИЗА: пространство фазовых портретов
// ============================================================

// Панель настроек сессии: параметры (общие), НУ (список), проекции (список),
// время/шаг, метод (заглушка), кнопка пересчёта.
// Reset lambda: what happens when the user hits "Reset to defaults".
// Analysis-tab passes model.from_record + start_phase_analysis; Custom-tab
// passes nullptr (Custom users reload from System tab or Run pipeline).
static void draw_phase_controls(PhaseAnalysisSession& s,
                                std::function<void()> on_reset_defaults) {
    bool changed = false;

    ImGui::Text("Phase portrait analysis");
    // Analysis-tab passes a non-null on_reset_defaults; Custom-tab passes
    // nullptr. Only Analysis is a true library-detached sandbox — in Custom,
    // this panel is one stage of a pipeline driven by the shared config,
    // so the sandbox disclaimer would be misleading.
    if (on_reset_defaults)
        ImGui::TextDisabled("Changes here are NOT saved to the library (sandbox).");


    // метод моделирования + пользовательские схемы из системы
    ImGui::Text("Method:"); ImGui::SameLine();
    changed |= draw_scheme_combo("##method", s.scheme, s.custom_schemes,
                                 [&s](const std::string&) { s.regenerate_krs(); });
    // Custom КРС теперь считаются и на CPU — тело компилируется в нативный шаг
    // (см. krs_cpu.h). Принудительный GPU оставляем ровно для случая, когда
    // компилятор на машине не найден.
    if (is_custom_scheme(s.scheme, s.custom_schemes) && !s.use_gpu) {
        std::string why;
        if (!krs_cpu_backend_available(&why)) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1), "(custom requires GPU: %s)",
                               why.c_str());
            s.use_gpu = true;
        }
    }

    // время, шаг, децимация
    ImGui::Text("Step h:"); ImGui::SameLine();
    changed |= InputNumStr("##sh", s.step_h, 80); ImGui::SameLine();
    ImGui::Text("Time(s):"); ImGui::SameLine();
    changed |= InputNumStr("##st", s.sim_time, 70); ImGui::SameLine();
    ImGui::Text("Skip(s):"); ImGui::SameLine();
    changed |= InputNumStr("##ssk", s.skip_time, 70);
    // Symmetry a[0] is also available to custom KRS bodies (same slot as CD);
    // only show the field if the body actually references a[0].
    if (s.scheme == "CD" || custom_scheme_uses_symmetry(s.scheme, s.custom_schemes)) {
        ImGui::Text("Symmetry s:"); ImGui::SameLine();
        changed |= InputNumStr("##sym", s.symmetry_s, 70);
    }
    ImGui::Text("Decimation (every Nth point):"); ImGui::SameLine();
    changed |= InputNumStr("##dec", s.decimation, 70);
    // шаг/время/децимация влияют на ось времени и сами данные: при их смене
    // просим автоскейл, чтобы time domain не "скакал" со старыми пределами.
    if (changed) s.fit_request = true;

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // переключатели
    ImGui::Checkbox("Auto recompute", &s.auto_recompute); ImGui::SameLine();
    ImGui::Checkbox("Legend shows initial conditions", &s.legend_show_ic); ImGui::SameLine();
    ImGui::Checkbox("GPU", &s.use_gpu);

    ImGui::Separator();

    // параметры (общие на все проекции)
    if (!s.params.empty()) {
        ImGui::SeparatorText("Parameters");
        if (ImGui::BeginTable("aparams", 2, ImGuiTableFlags_SizingFixedFit)) {
            for (const auto& p : s.params) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("%s", p.c_str());
                ImGui::TableSetColumnIndex(1);
                std::string id = "##ap_" + p;
                changed |= InputNumStr(id.c_str(), s.param_values[p], 110);
            }
            ImGui::EndTable();
        }
    }

    // начальные условия (несколько, мультистабильность)
    ImGui::SeparatorText("Initial conditions");
    int ic_to_remove = -1;
    for (int i = 0; i < (int)s.ic_sets.size(); ++i) {
        InitialConditionSet& ic = s.ic_sets[i];
        ImGui::PushID(i);
        ImGui::Checkbox("##vis", &ic.visible); ImGui::SameLine();
        ImGui::SetNextItemWidth(70);
        InputTextStr("##label", ic.label); ImGui::SameLine();
        for (const auto& v : s.vars) {
            ImGui::Text("%s:", v.c_str()); ImGui::SameLine();
            std::string id = "##icv_" + v;
            changed |= InputNumStr(id.c_str(), ic.values[v], 60); ImGui::SameLine();
        }
        if (ImGui::SmallButton("X")) ic_to_remove = i;
        ImGui::PopID();
    }
    if (ic_to_remove >= 0) { s.remove_ic(ic_to_remove); changed = true; }
    if (ImGui::Button("Add initial condition")) { s.add_ic(); }

    // проекции
    ImGui::SeparatorText("Projections");
    int pr_to_remove = -1;
    for (int i = 0; i < (int)s.projections.size(); ++i) {
        Projection& pr = s.projections[i];
        ImGui::PushID(1000 + i);
        ImGui::SetNextItemWidth(90);
        InputTextStr("##plabel", pr.label); ImGui::SameLine();
        // тип проекции
        ImGui::SetNextItemWidth(110);
        // Порядок обязан совпадать с enum ProjType (тип пишется в сессию как int).
        const char* tnames[] = { "Phase 2D", "Time domain", "Phase 3D", "Feature diagram" };
        int t = (int)pr.type;
        if (ImGui::Combo("##ptype", &t, tnames, IM_ARRAYSIZE(tnames))) {
            pr.type = (ProjType)t; s.fit_request = true;
        }
        ImGui::SameLine();

        if (pr.type == ProjType::Phase2D) {
            ImGui::Text("X:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(55);
            if (ImGui::BeginCombo("##px", s.vars.empty() ? "-" : s.vars[pr.axis_x < (int)s.vars.size() ? pr.axis_x : 0].c_str())) {
                for (int k = 0; k < (int)s.vars.size(); ++k)
                    if (ImGui::Selectable(s.vars[k].c_str(), pr.axis_x == k)) { pr.axis_x = k; s.fit_request = true; }
                ImGui::EndCombo();
            }
            ImGui::SameLine(); ImGui::Text("Y:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(55);
            if (ImGui::BeginCombo("##py", s.vars.empty() ? "-" : s.vars[pr.axis_y < (int)s.vars.size() ? pr.axis_y : 0].c_str())) {
                for (int k = 0; k < (int)s.vars.size(); ++k)
                    if (ImGui::Selectable(s.vars[k].c_str(), pr.axis_y == k)) { pr.axis_y = k; s.fit_request = true; }
                ImGui::EndCombo();
            }
        }
        else if (pr.type == ProjType::FeatureDiagram) {
            // Одна переменная: по ней ищутся пики. Оси фиксированы (значение
            // пика / интервал), поэтому переиспользуем axis_x и не заводим
            // отдельного поля в Projection.
            // Индекс vars.size() — комбинация x0 + pi*x1 + e*x2: тот же ряд,
            // что даёт writable_var == -1 в ядре (см. draw_writable_var_combo).
            // Сентинелом взят именно size(), а НЕ -1, как у writable_var:
            // соседние ветки берут s.vars[axis_x] по схеме
            // `axis_x < size() ? axis_x : 0`, и -1 у них ушёл бы в
            // отрицательный индекс при переключении типа проекции.
            const int nv = (int)s.vars.size();
            const bool feat_is_combo = (nv >= 2 && pr.axis_x == nv);
            const std::string preview = s.vars.empty()
                ? std::string("-")
                : (feat_is_combo ? combo_var_label(s.vars)
                                 : s.vars[pr.axis_x < nv ? pr.axis_x : 0]);

            ImGui::Text("var:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(feat_is_combo ? 170.0f : 55.0f);
            if (ImGui::BeginCombo("##pfv", preview.c_str())) {
                for (int k = 0; k < nv; ++k)
                    if (ImGui::Selectable(s.vars[k].c_str(), pr.axis_x == k)) { pr.axis_x = k; s.fit_request = true; }
                if (nv >= 2) {
                    ImGui::Separator();
                    const std::string lbl = combo_var_label(s.vars);
                    if (ImGui::Selectable(lbl.c_str(), feat_is_combo)) { pr.axis_x = nv; s.fit_request = true; }
                }
                ImGui::EndCombo();
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(peak vs interval)");
        }
        else if (pr.type == ProjType::Phase3D) {
            ImGui::Text("X:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(50);
            if (ImGui::BeginCombo("##p3x", s.vars.empty() ? "-" : s.vars[pr.axis_x < (int)s.vars.size() ? pr.axis_x : 0].c_str())) {
                for (int k = 0; k < (int)s.vars.size(); ++k)
                    if (ImGui::Selectable(s.vars[k].c_str(), pr.axis_x == k)) { pr.axis_x = k; s.fit_request = true; }
                ImGui::EndCombo();
            }
            ImGui::SameLine(); ImGui::Text("Y:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(50);
            if (ImGui::BeginCombo("##p3y", s.vars.empty() ? "-" : s.vars[pr.axis_y < (int)s.vars.size() ? pr.axis_y : 0].c_str())) {
                for (int k = 0; k < (int)s.vars.size(); ++k)
                    if (ImGui::Selectable(s.vars[k].c_str(), pr.axis_y == k)) { pr.axis_y = k; s.fit_request = true; }
                ImGui::EndCombo();
            }
            ImGui::SameLine(); ImGui::Text("Z:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(50);
            if (ImGui::BeginCombo("##p3z", s.vars.empty() ? "-" : s.vars[pr.axis_z < (int)s.vars.size() ? pr.axis_z : 0].c_str())) {
                for (int k = 0; k < (int)s.vars.size(); ++k)
                    if (ImGui::Selectable(s.vars[k].c_str(), pr.axis_z == k)) { pr.axis_z = k; s.fit_request = true; }
                ImGui::EndCombo();
            }
        }
        else { // TimeDomain — галочки переменных
            // Слотов на один больше числа переменных: последний — комбинация
            // x0 + pi*x1 + e*x2. При смене размера старые галочки переносим, а
            // комбинацию гасим — иначе у сохранённых сессий сама собой
            // появилась бы новая кривая и перемасштабировала бы ось Y (её
            // амплитуда заметно больше, чем у отдельных переменных).
            const int nv_td = (int)s.vars.size();
            if ((int)pr.show_var.size() != nv_td + 1) {
                std::vector<bool> prev = pr.show_var;
                pr.show_var.assign((size_t)nv_td + 1, true);
                for (size_t j = 0; j < prev.size() && j < (size_t)nv_td; ++j)
                    pr.show_var[j] = prev[j];
                pr.show_var[(size_t)nv_td] = false;
            }
            ImGui::Text("vars:"); ImGui::SameLine();
            for (int k = 0; k < nv_td; ++k) {
                bool v = pr.show_var[k];
                if (ImGui::Checkbox(s.vars[k].c_str(), &v)) { pr.show_var[k] = v; }
                ImGui::SameLine();
            }
            if (nv_td >= 2) {
                bool v = pr.show_var[(size_t)nv_td];
                if (ImGui::Checkbox(combo_var_label(s.vars).c_str(), &v))
                    pr.show_var[(size_t)nv_td] = v;
                ImGui::SameLine();
            }
            ImGui::NewLine();
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("X")) pr_to_remove = i;
        ImGui::PopID();
    }
    if (pr_to_remove >= 0) s.remove_projection(pr_to_remove);
    if (ImGui::Button("Add projection")) { s.add_projection(); }
    ImGui::SameLine();
    if (ImGui::Button("Reset windows layout")) { s.layout_generation++; }

    ImGui::Separator();
    // Debug panel: show what kernel/integrator actually computes. Useful for
    // comparing CPU vs GPU runs and catching parsing surprises.
    if (ImGui::CollapsingHeader("Debug: KRS body & parsed values")) {
        ImGui::TextDisabled("KRS body — what NVRTC compiles for GPU. On CPU the same RHS\n"
                            "is parsed into an AST and interpreted. For CD the CPU and GPU\n"
                            "algorithms differ (see the second panel). Values below are\n"
                            "the exact doubles both paths parse.");

        if (ImGui::Button("Regenerate")) s.regenerate_krs();
        ImGui::SameLine();
        ImGui::TextDisabled("(rebuilds KRS from the current system/scheme)");

        // GPU KRS body (what NVRTC compiles).
        ImGui::SeparatorText("KRS (GPU, NVRTC)");
        std::string body_gpu = s.krs_code.empty()
            ? std::string("(empty — press Regenerate or change Method)")
            : s.krs_code;
        std::vector<char> buf_gpu(body_gpu.begin(), body_gpu.end());
        buf_gpu.resize(body_gpu.size() + 1);
        buf_gpu[body_gpu.size()] = '\0';
        ImGui::InputTextMultiline("##krs_gpu", buf_gpu.data(), buf_gpu.size(),
            ImVec2(-1, 200), ImGuiInputTextFlags_ReadOnly);

        // CPU equivalent: identical to GPU for Euler/RK4/etc; for CD it's the
        // 4-simple-iterations form that integrator.cpp::step_cd actually runs.
        ImGui::SeparatorText("KRS (CPU equivalent)");
        // Кодогенерация — не для каждого кадра: панель открыта, а ImGui
        // перерисовывает её 60 раз в секунду. Кэшируем по паре (схема, тело
        // GPU-KRS): regenerate_krs переписывает krs_code при любой правке
        // системы или схемы, поэтому пара однозначно описывает вход генератора.
        static std::string cpu_cache_key, cpu_cache_body;
        const std::string cpu_key = s.scheme + "\n" + s.krs_code;
        if (cpu_cache_key != cpu_key) {
            try {
                cpu_cache_body = codegen_scheme_cpu_equivalent(s.sys, scheme_from_name(s.scheme));
            } catch (...) {
                cpu_cache_body = "(generation failed)";
            }
            if (cpu_cache_body.empty())
                cpu_cache_body = "(empty — same as GPU for non-CD schemes)";
            cpu_cache_key = cpu_key;
        }
        std::vector<char>& buf_cpu = input_scratch(cpu_cache_body, 1);
        ImGui::InputTextMultiline("##krs_cpu", buf_cpu.data(), buf_cpu.size(),
            ImVec2(-1, 200), ImGuiInputTextFlags_ReadOnly);
        if (s.scheme != "CD")
            ImGui::TextDisabled("(for %s CPU and GPU evaluate the same AST — texts match)", s.scheme.c_str());
        else
            ImGui::TextDisabled("(for CD: GPU uses analytic solve for linear vars; CPU always uses 4 iterations)");

        ImGui::SeparatorText("Parsed inputs (double, %.17g)");
        // Панель показывает ИМЕННО то, что увидит движок, поэтому обязана
        // разбирать поля тем же кодом — см. num_parse.h.
        auto parse = [](const std::string& v, double def) { return parse_num(v, def); };

        ImGui::Text("h        = %.17g", parse(s.step_h, 0.01));
        ImGui::Text("a[0] (s) = %.17g", parse(s.symmetry_s, 0.5));
        for (size_t j = 0; j < s.params.size(); ++j) {
            auto it = s.param_values.find(s.params[j]);
            double v = (it != s.param_values.end()) ? parse(it->second, 0.0) : 0.0;
            ImGui::Text("a[%zu] (%s) = %.17g", j + 1, s.params[j].c_str(), v);
        }
        for (size_t k = 0; k < s.ic_sets.size(); ++k) {
            ImGui::Text("%s:", s.ic_sets[k].label.c_str());
            for (size_t i = 0; i < s.vars.size(); ++i) {
                auto it = s.ic_sets[k].values.find(s.vars[i]);
                double v = (it != s.ic_sets[k].values.end()) ? parse(it->second, 0.0) : 0.0;
                ImGui::Text("  X[%zu] (%s) = %.17g", i, s.vars[i].c_str(), v);
            }
        }
    }

    if (on_reset_defaults) {
        ImGui::Separator();
        if (ImGui::Button("Reset to defaults")) {
            try { on_reset_defaults(); } catch (...) {}
        }
    }

    ImGui::Separator();
    // Recompute по кнопке или Ctrl+R, дисейблится во время async-расчёта.
    // Авто-сохранение _last делается в draw_gui::poll(), когда результат готов.
    bool do_recompute = false;
    if (s.in_flight) {
        ImGui::BeginDisabled();
        ImGui::Button("Recomputing...", ImVec2(-1, 0));
        ImGui::EndDisabled();
    }
    else {
        do_recompute = ImGui::Button("Recompute (Ctrl+R)", ImVec2(-1, 0));
        if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_R, false)) do_recompute = true;
        if (s.auto_recompute && changed) do_recompute = true;
    }
    if (do_recompute) {
        s.recompute_async();
    }

    if (!s.result.error.empty())
        ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "%s", s.result.error.c_str());
}

// ============================================================================
// Единый счётчик режимов для статусных строк всех панелей.
//
// Раньше каждая панель считала "diverged" по flags[k] < 0, и одна и та же
// подпись значила разное: для Bif1D/Bif2D отрицательный флаг — это fixed
// point, а для LLE/LS — расходимость. Теперь коды канонические (REGIME_* в
// configCUDA.h), и категории раскладываются одинаково везде.
// ============================================================================
static void count_regimes(const std::vector<int>& flags, int& fp, int& unb, int& osc) {
    fp = unb = osc = 0;
    for (int f : flags) {
        if      (regime_is_fixed_point(f)) ++fp;
        else if (regime_is_unbound(f))     ++unb;
        else                               ++osc;
    }
}

// Подпись под "OK: ..." — печатается только если есть что показать кроме
// колебательного режима (иначе строка была бы шумом на каждом успешном run'е).
static void draw_regime_summary(const std::vector<int>& flags) {
    int fp = 0, unb = 0, osc = 0;
    count_regimes(flags, fp, unb, osc);
    if (fp == 0 && unb == 0) return;
    ImGui::TextDisabled("regimes: %d fixed point / %d unbound / %d oscillation",
                        fp, unb, osc);
}

// Рисует окна проекций (каждая — отдельное docking-окно с графиком).
// Optional `before_begin` runs immediately before each projection window's
// ImGui::Begin (Custom mode uses it to assign initial dock target); optional
// `after_begin` runs right after Begin returns true (Custom mode uses it to
// attach the Move-to-Tab context menu). Both empty (default) preserve the
// Analysis-mode behaviour.
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

// Подпись НУ для легенды (галка "Legend shows initial conditions").
// Берём ЧИСЛА, которые реально ушли в расчёт (result.snapshot.ic_flat), и
// форматируем через fmt_tick — то есть с Tick precision из Settings. Так
// сгенерированные НУ (узлы сетки бассейнов пишутся с %.17g) не разносят
// легенду на пол-экрана, а смена точности в Settings применяется сразу, без
// пересчёта. Fallback на сохранённый result.ic_text — если снапшота нет.
[[nodiscard]] static std::string ic_legend_text(const AnalysisResult& res, size_t k) {
    if (k < res.snapshot.ic_flat.size() && !res.snapshot.ic_flat[k].empty()) {
        const std::vector<double>& v = res.snapshot.ic_flat[k];
        std::string txt = "(";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) txt += ",";
            txt += fmt_tick(v[i]);
        }
        txt += ")";
        return txt;
    }
    return (k < res.ic_text.size()) ? res.ic_text[k] : std::string();
}
// `title_suffix` is appended to every projection window's title (e.g.
// "##sys_<name>") so imgui.ini stores dock state per system in Custom mode.
// `owner_id_delta` is XOR'd into the per-projection owner_id passed to
// Plot2D/3D so the SHARED PlotRenderer cache (static in this function) is
// keyed per system — without it, Chen and Rossler both used owner_id=0
// for their first projection and Rossler saw Chen's cached FBO texture.
// Analysis mode passes zero and gets the previous behaviour.
// Параметры кластеризации для диаграммы признаков — те же множители осей и eps,
// с которыми работает dbscan (cudaLibrary.cu). Нужны, чтобы диаграмма показывала
// ТО пространство, в котором реально считаются кластеры: множители применяются
// только внутри ядра, и без них диаграмма рисовала сырые (пик; IPI). Из-за этого
// по ней нельзя было предсказать результат — при mult interval = 0 ядро видит
// одномерную задачу, а на картинке оставалось двумерное облако.
//
// valid=false — вызывающий не связан ни с каким DBSCAN-конфигом (вкладка Phase
// analysis). Тогда рисуем сырые признаки и без окружности, как было.
struct FeatureClusterParams {
    bool   valid         = false;
    double mult_peak     = 1.0;
    double mult_interval = 1.0;
    double eps           = 0.0;
};

static void draw_projection_windows(PhaseAnalysisSession& s, const GuiCallbacks& cb,
                                    const ProjHookFn& before_begin = {},
                                    const ProjHookFn& after_begin  = {},
                                    const std::string& title_suffix = {},
                                    int owner_id_delta = 0,
                                    const PhaseStyleFn& style_fn = {},
                                    const FeatureClusterParams& clust = {}) {
    const AnalysisResult& res = s.result;
    // Lambda installed on every projection view that has data — right-click
    // "Export data..." writes the full double-precision trajectory set (all
    // coords for all ICs) + a _config.csv sidecar. Ставится и на Phase3D:
    // Plot3DView теперь тоже поддерживает popup_extras, поэтому экспорт
    // доступен во всех трёх типах проекций одинаково.
    const bool phase_busy = s.in_flight;
    auto phase_popup_extras = [&res, &cb, phase_busy]() {
        const bool has_data = res.ok && !res.trajectories.empty();
        draw_export_menu_item(!has_data || phase_busy, cb, [&res](const std::string& p) {
            data_export::export_phase(res, res.snapshot, p);
        });
    };
    // Свой offscreen-рендерер (FBO/текстура) на КАЖДУЮ проекцию: иначе все окна
    // показывали бы одну общую текстуру (геометрию последней отрисованной).
    // PlotRenderer некопируемый -> храним через unique_ptr, подгоняем под число проекций.
    // Кэш ключуется по owner_id_delta: у функции появился второй потребитель
    // (Basins → фазовые портреты, свой delta на каждый config), и с общим
    // вектором переключение между ними пересоздавало бы FBO каждый кадр.
    // Analysis передаёт 0 и получает ровно прежнее поведение.
    // Давно не рисовавшиеся владельцы вытесняются: каждый держит по FBO на
    // проекцию (мегабайты видеопамяти), а число систем/config'ов не ограничено.
    struct RendererBucket {
        std::vector<std::unique_ptr<PlotRenderer>> v;
        int last_frame = 0;
    };
    static std::map<int, RendererBucket> renderers_by_owner;
    const int cur_frame = ImGui::GetFrameCount();
    for (auto it = renderers_by_owner.begin(); it != renderers_by_owner.end(); ) {
        if (it->first != owner_id_delta &&
            cur_frame - it->second.last_frame > kRendererEvictFrames)
            it = renderers_by_owner.erase(it);
        else
            ++it;
    }
    RendererBucket& bucket = renderers_by_owner[owner_id_delta];
    bucket.last_frame = cur_frame;
    std::vector<std::unique_ptr<PlotRenderer>>& renderers = bucket.v;
    if ((int)renderers.size() != (int)s.projections.size()) {
        renderers.clear();
        for (size_t k = 0; k < s.projections.size(); ++k)
            renderers.push_back(std::make_unique<PlotRenderer>());
    }
    int pr_to_remove = -1;
    for (int i = 0; i < (int)s.projections.size(); ++i) {
        Projection& pr = s.projections[i];
        PlotRenderer& renderer = *renderers[i]; // рендерер этой проекции
        // ε-окружность включает ТОЛЬКО ветка FeatureDiagram ниже. Гасим её
        // здесь каждый кадр: view2d переживает смену типа проекции (объект
        // переиспользуется, см. `if (!pr.view2d)` в ветках), и без сброса
        // кружок остался бы висеть на фазовом портрете после переключения
        // комбо типа.
        if (pr.view2d) pr.view2d->hover_circle_r = std::numeric_limits<double>::quiet_NaN();
        std::string title = pr.label + "##proj" + std::to_string(i) + "_g" + std::to_string(s.layout_generation) + title_suffix;
        bool open = true; // крестик закрытия
        // Начальные позиция и размер (только при первом появлении).
        // Каскад слева-сверху: каждое следующее окно чуть смещено.
        // ИЗМЕНИТЬ РАЗМЕР МОЖНО ЗДЕСЬ: ImVec2(ширина, высота) в пикселях.
        float ox = 60.0f + (i % 5) * 35.0f;
        float oy = 80.0f + (i % 5) * 35.0f;
        ImGui::SetNextWindowPos(ImVec2(ox, oy), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(700, 550), ImGuiCond_FirstUseEver);
        if (before_begin) before_begin(i, title);
        if (ImGui::Begin(title.c_str(), &open)) {
            if (after_begin) after_begin(i, title);
            ImGui::PushID(i);   // разделить ID внутренних виджетов между окнами проекций
            if (!res.ok || res.trajectories.empty()) {
                ImGui::TextDisabled("No data. Press Recompute.");
            }
            ////////////////////////////////////////
            else if (pr.type == ProjType::Phase2D) {
                int ax = pr.axis_x, ay = pr.axis_y;
                if (!res.ok || res.trajectories.empty()) {
                    ImGui::TextDisabled("No data.");
                }
                else {
                    // создать вьюер при первой отрисовке
                    if (!pr.view2d) pr.view2d = std::make_unique<Plot2DView>();

                    // обновить имена осей
                    pr.view2d->x_axis.name = s.vars.empty() ? "x" : s.vars[ax < (int)s.vars.size() ? ax : 0];
                    pr.view2d->y_axis.name = s.vars.empty() ? "y" : s.vars[ay < (int)s.vars.size() ? ay : 0];
                    // No zero-axis on phase 2D — x=0 / y=0 have no meaning
                    // for a state-space trajectory.
                    pr.view2d->show_zero_x = false;
                    pr.view2d->show_zero_y = false;
                    // Alpha slider fades the trajectory, not the legend swatch.
                    pr.view2d->legend_ignore_series_alpha = true;

                    // Toolbar над плотом: opt-in custom line styling (ImDrawList-путь
                    // с настраиваемой толщиной + α). По дефолту выключено → быстрый
                    // GL shader-line путь (1px, α=1). Текущая отрисовка не ломается.
                    draw_style_toolbar("Custom line style", "phase2d", pr.custom_line_style,
                        [&pr]() {
                            bool ch = false;
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Line width##phase2d", &pr.line_width, 0.1f, 5.0f, "%.2f");
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Alpha##phase2d",      &pr.alpha,      0.0f, 1.0f, "%.2f");
                            return ch;
                        });
                    pr.view2d->imdraw_lines      = pr.custom_line_style;
                    pr.view2d->line_thickness_px = pr.line_width;

                    // подготовить серии: для каждой траектории выбираем координаты по (ax, ay).
                    // Буфер локальный: render() ниже забирает точки синхронно, дольше вызова
                    // указатели не нужны. static тут был и бесполезен (clear()+resize() всё
                    // равно уничтожает capacity), и опасен — три вызова draw_projection_windows
                    // делили бы один буфер на всех.
                    std::vector<std::vector<float>> series_data(res.trajectories.size());

                    std::vector<PlotSeriesInput> series_in;
                    series_in.reserve(res.trajectories.size());

                    // видимость: глобальная — из живых галочек НУ (меняется без recompute)
                    std::vector<bool> init_vis(res.trajectories.size(), true);
                    std::vector<bool> glob_vis(res.trajectories.size(), true);

                    for (size_t k = 0; k < res.trajectories.size(); ++k) {
                        const auto& traj = res.trajectories[k];
                        auto& buf = series_data[k];

                        // Per-IC стиль (Basins → фазовые портреты). as_point:
                        // траектория бассейна-равновесия после транзиента стоит
                        // в одной точке, поэтому кладём в буфер ТОЛЬКО последнюю
                        // — один маркер вместо десятков тысяч совпадающих точек.
                        PhaseSeriesStyle st;
                        const bool has_style = style_fn && style_fn((int)k, st);
                        const bool as_point  = has_style && st.as_point;

                        if (as_point) {
                            if (!traj.empty()) {
                                const auto& pt = traj.back();
                                buf.push_back((float)pt[ax < (int)pt.size() ? ax : 0]);
                                buf.push_back((float)pt[ay < (int)pt.size() ? ay : 0]);
                            }
                        } else {
                            buf.reserve(traj.size() * 2);
                            for (const auto& pt : traj) {
                                buf.push_back((float)pt[ax < (int)pt.size() ? ax : 0]);
                                buf.push_back((float)pt[ay < (int)pt.size() ? ay : 0]);
                            }
                        }
                        std::string lab = (k < res.labels.size()) ? res.labels[k] : ("IC " + std::to_string(k + 1));
                        if (s.legend_show_ic) lab = ic_legend_text(res, k);

                        PlotSeriesInput si;
                        si.points = buf.data();
                        si.n_points = (int)(buf.size() / 2);
                        si.color = has_style ? st.color : ic_base_color((int)k);
                        // В custom_line_style режиме применяем α к цвету IC
                        // (rendering: ImDrawList использует color.w как alpha).
                        if (pr.custom_line_style) si.color.w = pr.alpha;
                        if (as_point) {
                            si.points_override = 1;
                            si.point_marker    = st.marker;
                            si.point_size_px   = st.point_size;
                        }
                        si.label = lab;
                        series_in.push_back(si);

                        bool vis = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;
                        glob_vis[k] = vis;
                        init_vis[k] = true;
                    }

                    int data_gen = s.data_generation * 100 + ax * 10 + ay;

                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();

                    pr.view2d->popup_extras = phase_popup_extras;
                    pr.view2d->render(renderer, origin, avail, i ^ owner_id_delta, data_gen,
                        series_in, init_vis, glob_vis, s.fit_request);
                }
            }
            // Диаграмма признаков: точечный график (значение пика; интервал до
            // него) по переменной pr.axis_x. Пики уже посчитаны на worker'е
            // (AnalysisResult::features) тем же алгоритмом, что и GPU-peakFinder,
            // поэтому здесь только раскладка в буферы — пересчёта нет.
            else if (pr.type == ProjType::FeatureDiagram) {
                const int av = pr.axis_x;
                if (!res.ok || res.features.empty()) {
                    ImGui::TextDisabled("No data.");
                }
                else {
                    if (!pr.view2d) pr.view2d = std::make_unique<Plot2DView>();

                    // Оси кластеризации: те же множители, что уходят в dbscan.
                    // Без DBSCAN-конфига (clust.valid == false) остаются 1 и 1,
                    // т.е. сырые признаки, как было.
                    const double mp = clust.valid ? clust.mult_peak     : 1.0;
                    const double mi = clust.valid ? clust.mult_interval : 1.0;
                    auto mult_suffix = [](double m) -> std::string {
                        if (m == 1.0) return std::string();
                        char b[32]; std::snprintf(b, sizeof(b), " x %g", m);
                        return std::string(b);
                    };

                    // av == vars.size() — комбинация (см. комбо-бокс выше).
                    const int nv_ax = (int)s.vars.size();
                    std::string ax_name;
                    if (s.vars.empty())                     ax_name = "x";
                    else if (nv_ax >= 2 && av == nv_ax)     ax_name = combo_var_label(s.vars);
                    else                                    ax_name = s.vars[av < nv_ax ? av : 0];

                    pr.view2d->x_axis.name = "peak " + ax_name + mult_suffix(mp);
                    pr.view2d->y_axis.name = "IPI" + mult_suffix(mi);   // interpeak interval
                    // Интервал неотрицателен, а пики часто лежат вокруг нуля —
                    // нулевая линия по Y тут осмысленна, по X нет.
                    pr.view2d->show_zero_x = false;
                    pr.view2d->show_zero_y = true;
                    pr.view2d->legend_ignore_series_alpha = true;
                    // ε-окружность под курсором: радиус eps в ТЕХ ЖЕ осях, в
                    // которых уложены точки (множители уже применены ниже),
                    // поэтому накрытые ею пики — то, что dbscan сольёт в один
                    // кластер. Без DBSCAN-конфига радиуса нет → NaN, ничего не
                    // рисуется. Ставится каждый кадр: immediate mode.
                    pr.view2d->hover_circle_r = (clust.valid && clust.eps > 0.0)
                        ? clust.eps
                        : std::numeric_limits<double>::quiet_NaN();

                    draw_style_toolbar("Custom point style", "featdiag", pr.custom_line_style,
                        [&pr]() {
                            bool ch = false;
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Point size##featdiag", &pr.line_width, 0.5f, 8.0f, "%.1f");
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Alpha##featdiag",      &pr.alpha,      0.0f, 1.0f, "%.2f");
                            return ch;
                        });
                    // Всегда точки, линий между пиками нет: соседние пики
                    // соединять нечем — это облако признаков, а не траектория.
                    pr.view2d->imdraw_lines = false;

                    // Локальный буфер — см. series_data в ветке Phase2D выше.
                    std::vector<std::vector<float>> feat_data(res.features.size());
                    std::vector<PlotSeriesInput> series_in;
                    series_in.reserve(res.features.size());
                    std::vector<bool> init_vis(res.features.size(), true);
                    std::vector<bool> glob_vis(res.features.size(), true);

                    size_t total_pts = 0;
                    for (size_t k = 0; k < res.features.size(); ++k) {
                        auto& buf = feat_data[k];
                        const auto& per_var = res.features[k];
                        if (av >= 0 && av < (int)per_var.size()) {
                            const FeaturePoints& fp = per_var[(size_t)av];
                            const size_t n = fp.peaks.size() < fp.intervals.size()
                                           ? fp.peaks.size() : fp.intervals.size();
                            buf.reserve(n * 2);
                            for (size_t p = 0; p < n; ++p) {
                                if (!std::isfinite(fp.peaks[p]) || !std::isfinite(fp.intervals[p]))
                                    continue;
                                buf.push_back((float)(fp.peaks[p]     * mp));
                                buf.push_back((float)(fp.intervals[p] * mi));
                            }
                            total_pts += buf.size() / 2;
                        }

                        std::string lab = (k < res.labels.size()) ? res.labels[k]
                                                                  : ("IC " + std::to_string(k + 1));
                        if (s.legend_show_ic) lab = ic_legend_text(res, k);

                        PhaseSeriesStyle st;
                        const bool has_style = style_fn && style_fn((int)k, st);

                        PlotSeriesInput si;
                        si.points   = buf.empty() ? nullptr : buf.data();
                        si.n_points = (int)(buf.size() / 2);
                        si.color    = has_style ? st.color : ic_base_color((int)k);
                        if (pr.custom_line_style) si.color.w = pr.alpha;
                        si.points_override = 1;
                        si.point_marker    = (int)PointMarker::Circle;
                        si.point_size_px   = pr.custom_line_style ? pr.line_width : 3.0f;
                        si.label = lab;
                        series_in.push_back(si);

                        glob_vis[k] = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;
                        init_vis[k] = true;
                    }

                    if (total_pts == 0)
                        ImGui::TextDisabled("No peaks found (check transient / peak thresholds in Settings).");

                    // Множители и eps входят в gen-токен: иначе Plot2DView отдал бы
                    // уже залитый VBO, и правка коэффициентов не меняла бы картинку —
                    // ровно тот класс «настройка молча не применяется», что уже
                    // случался с --fmad.
                    int data_gen = s.data_generation * 100 + av;
                    if (clust.valid) {
                        auto mix = [](int acc, double v) {
                            char b[32]; std::snprintf(b, sizeof(b), "%.9g", v);
                            for (const char* p = b; *p; ++p)
                                acc = acc * 131 + (int)(unsigned char)*p;
                            return acc;
                        };
                        data_gen = mix(mix(mix(data_gen, clust.mult_peak),
                                           clust.mult_interval), clust.eps);
                    }

                    ImVec2 avail  = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();
                    pr.view2d->popup_extras = phase_popup_extras;
                    pr.view2d->render(renderer, origin, avail, i ^ owner_id_delta, data_gen,
                        series_in, init_vis, glob_vis, s.fit_request);
                }
            }
            else if (pr.type == ProjType::TimeDomain) {
                if (!res.ok || res.trajectories.empty()) {
                    ImGui::TextDisabled("No data.");
                }
                else {
                    if (!pr.view2d) pr.view2d = std::make_unique<Plot2DView>();
                    // No zero-axis for time-domain — the X-axis is time,
                    // y=0 rarely coincides with a meaningful reference.
                    pr.view2d->show_zero_x = false;
                    pr.view2d->show_zero_y = false;
                    // Alpha slider fades the trajectory, not the legend swatch.
                    pr.view2d->legend_ignore_series_alpha = true;

                    // Toolbar над плотом: opt-in custom line styling (ImDrawList-путь
                    // с настраиваемой толщиной + α). Дефолт — быстрый GL shader-line
                    // путь (1px, α=1). Аналогично Phase2D.
                    draw_style_toolbar("Custom line style", "timedomain", pr.custom_line_style,
                        [&pr]() {
                            bool ch = false;
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Line width##timedomain", &pr.line_width, 0.1f, 5.0f, "%.2f");
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Alpha##timedomain",      &pr.alpha,      0.0f, 1.0f, "%.2f");
                            return ch;
                        });
                    pr.view2d->imdraw_lines      = pr.custom_line_style;
                    pr.view2d->line_thickness_px = pr.line_width;

                    double h = parse_ratio_or(s.step_h, 0.01);  if (h <= 0) h = 0.01;
                    int dec = parse_int_or(s.decimation, 1);    if (dec < 1) dec = 1;
                    double dt = h * dec;
                    int nvars = (int)s.vars.size();

                    // Слотов nvars + 1: последний — комбинация (см. блок галочек).
                    if ((int)pr.show_var.size() != nvars + 1) {
                        std::vector<bool> prev = pr.show_var;
                        pr.show_var.assign((size_t)nvars + 1, true);
                        for (size_t j = 0; j < prev.size() && j < (size_t)nvars; ++j)
                            pr.show_var[j] = prev[j];
                        pr.show_var[(size_t)nvars] = false;
                    }

                    pr.view2d->x_axis.name = "t";
                    pr.view2d->y_axis.name = "value";

                    pr.view2d->pad_x = false;
                    pr.view2d->show_zero_x = false;

                    // серии: одна на (траектория k, видимая переменная vi).
                    // Локальный буфер — см. ветку Phase2D выше.
                    std::vector<std::vector<float>> series_data;

                    std::vector<PlotSeriesInput> series_in;
                    std::vector<bool> init_vis;
                    std::vector<bool> glob_vis;

                    for (size_t k = 0; k < res.trajectories.size(); ++k) {
                        const auto& traj = res.trajectories[k];
                        if (traj.empty()) continue;
                        int n = (int)traj.size();
                        // видимость НУ — из живой галочки (без recompute)
                        bool ic_vis = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;

                        // Per-IC цвет (Basins → фазовые портреты). as_point тут
                        // намеренно игнорируется: во временной развёртке даже
                        // равновесие — это линия (константа), маркер не нужен.
                        PhaseSeriesStyle st;
                        const bool has_style = style_fn && style_fn((int)k, st);

                        // vi == nvars — слот комбинации; для dim < 2 он не
                        // предлагается и пропускается.
                        for (int vi = 0; vi <= nvars; ++vi) {
                            const bool is_combo = (vi == nvars);
                            if (is_combo && nvars < 2) continue;
                            if (vi < (int)pr.show_var.size() && !pr.show_var[vi]) continue;

                            series_data.emplace_back();
                            auto& buf = series_data.back();
                            buf.reserve(n * 2);
                            for (int t = 0; t < n; ++t) {
                                buf.push_back((float)(t * dt));
                                buf.push_back((float)(is_combo
                                    ? combo_var_value(traj[t], nvars)
                                    : traj[t][vi < (int)traj[t].size() ? vi : 0]));
                            }

                            std::string base = (k < res.labels.size()) ? res.labels[k] : ("IC" + std::to_string(k + 1));
                            std::string who = s.legend_show_ic ? ic_legend_text(res, k) : base;
                            std::string lab = (is_combo ? combo_var_label(s.vars) : s.vars[vi])
                                            + " [" + who + "]";

                            // Оттенок комбинации берём из расширенной палитры
                            // (nvars + 1), чтобы цвета самих переменных остались
                            // теми же, что были до появления слота.
                            const int shade_den = is_combo ? nvars + 1 : nvars;
                            PlotSeriesInput si;
                            si.points = buf.data();
                            si.n_points = n;
                            si.color = has_style ? shade_of(st.color, vi, shade_den)
                                                 : ic_var_shade((int)k, vi, shade_den);
                            // В custom_line_style режиме применяем α к цвету
                            // (ImDrawList использует color.w как alpha).
                            if (pr.custom_line_style) si.color.w = pr.alpha;
                            si.label = lab;
                            series_in.push_back(si);
                            init_vis.push_back(true);   // локальная (легенда) стартует с видимости НУ
                            glob_vis.push_back(ic_vis);   // глобальная = живая галочка НУ
                        }
                    }

                    int data_gen = s.data_generation * 1000 + (int)series_in.size();

                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();

                    pr.view2d->popup_extras = phase_popup_extras;
                    pr.view2d->render(renderer, origin, avail, i ^ owner_id_delta, data_gen,
                        series_in, init_vis, glob_vis, s.fit_request);
                }
            }
            else if (pr.type == ProjType::Phase3D) {
                int ax = pr.axis_x, ay = pr.axis_y, az = pr.axis_z;
                if (!res.ok || res.trajectories.empty()) {
                    ImGui::TextDisabled("No data.");
                }
                else {
                    if (!pr.view3d) pr.view3d = std::make_unique<Plot3DView>();
                    // Alpha slider fades the trajectory, not the legend swatch.
                    pr.view3d->legend_ignore_series_alpha = true;

                    // Toolbar над плотом: opt-in custom line styling (толщина + α).
                    // В 3D нет ImDrawList-fallback (потерялся бы depth-sorting),
                    // толщина идёт через glLineWidth — драйвер может клампить,
                    // α точно уходит в шейдер. При выключенной фиче — восстанавливаем
                    // старый хардкод 1.5f, чтобы поведение осталось прежним.
                    draw_style_toolbar("Custom line style", "phase3d", pr.custom_line_style,
                        [&pr]() {
                            bool ch = false;
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Line width##phase3d", &pr.line_width, 0.1f, 5.0f, "%.2f");
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Alpha##phase3d",      &pr.alpha,      0.0f, 1.0f, "%.2f");
                            return ch;
                        });
                    pr.view3d->line_thickness_px = pr.custom_line_style ? pr.line_width : 1.5f;
                    pr.view3d->custom_line_style = pr.custom_line_style;

                    pr.view3d->x_name = s.vars.empty() ? "x" : s.vars[ax < (int)s.vars.size() ? ax : 0];
                    pr.view3d->y_name = s.vars.empty() ? "y" : s.vars[ay < (int)s.vars.size() ? ay : 0];
                    pr.view3d->z_name = s.vars.empty() ? "z" : s.vars[az < (int)s.vars.size() ? az : 0];

                    // Локальный буфер — см. ветку Phase2D выше.
                    std::vector<std::vector<float>> series_data(res.trajectories.size());

                    std::vector<PlotSeriesInput3D> series_in;
                    series_in.reserve(res.trajectories.size());

                    std::vector<bool> init_vis(res.trajectories.size(), true);
                    std::vector<bool> glob_vis(res.trajectories.size(), true);

                    for (size_t k = 0; k < res.trajectories.size(); ++k) {
                        const auto& traj = res.trajectories[k];
                        auto& buf = series_data[k];
                        buf.reserve(traj.size() * 3);
                        for (const auto& pt : traj) {
                            buf.push_back((float)pt[ax < (int)pt.size() ? ax : 0]);
                            buf.push_back((float)pt[ay < (int)pt.size() ? ay : 0]);
                            buf.push_back((float)pt[az < (int)pt.size() ? az : 0]);
                        }
                        std::string lab = (k < res.labels.size()) ? res.labels[k] : ("IC " + std::to_string(k + 1));
                        if (s.legend_show_ic) lab = ic_legend_text(res, k);

                        PlotSeriesInput3D si;
                        si.points = buf.data();
                        si.n_points = (int)(buf.size() / 3);
                        si.color = ic_base_color((int)k);
                        // В custom_line_style режиме применяем α к цвету IC
                        // (draw_line_3d прокидывает color[3] в шейдер как альфа).
                        if (pr.custom_line_style) si.color.w = pr.alpha;
                        si.label = lab;
                        series_in.push_back(si);

                        bool vis = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;
                        glob_vis[k] = vis;
                        init_vis[k] = true;
                    }

                    int data_gen = s.data_generation * 1000 + ax * 100 + ay * 10 + az;

                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();

                    pr.view3d->popup_extras = phase_popup_extras;
                    pr.view3d->render(renderer, origin, avail, i ^ owner_id_delta, data_gen,
                        series_in, init_vis, glob_vis, s.fit_request);
                }
            }
            ImGui::PopID();
        }
        ImGui::End();
        if (!open) pr_to_remove = i; // окно закрыто крестиком
    }
    if (pr_to_remove >= 0) s.remove_projection(pr_to_remove);
    // автоскейл применён ко всем окнам в этом кадре — сбрасываем запрос
    s.fit_request = false;
}

// ============================================================
// Combo "Writable var": список переменных + (через разделитель) комбинация
// x[0] + pi*x[1] + euler*x[2] (для |vars|=2 — без e*x[2], для |vars|=1 —
// только сама переменная). Sentinel в hostside int — -1 = combination.
// Используется в Bifurcation и Basins (где kernel пишет data[i] через
// loopCalculateDiscreteModel_int — он умеет оба режима, см. cudaLibrary.cu).
// ============================================================
static void draw_writable_var_combo(const std::vector<std::string>& vars,
                                    int& writable_var,
                                    const char* combo_id) {
    if (vars.empty()) return;
    const int N = (int)vars.size();
    // Допустимые значения: -1 (combination) или [0, N-1]. Старые сейвы могли
    // иметь невалидные индексы — clamp в 0.
    if (writable_var < -1 || writable_var >= N) writable_var = 0;
    // Для системы из одной переменной combination сворачивается до x[0] —
    // показывать отдельную опцию бессмысленно, форсим single-var.
    if (N == 1 && writable_var == -1) writable_var = 0;

    // Подпись комбинации — общая combo_var_label (раньше здесь жила её копия).
    const std::string preview =
        (writable_var == -1) ? combo_var_label(vars) : vars[writable_var];

    ImGui::SetNextItemWidth(220);
    if (ImGui::BeginCombo(combo_id, preview.c_str())) {
        for (int i = 0; i < N; ++i) {
            bool sel = writable_var == i;
            if (ImGui::Selectable(vars[i].c_str(), sel)) writable_var = i;
        }
        if (N >= 2) {
            ImGui::Separator();
            const std::string lbl = combo_var_label(vars);
            bool sel_combo = writable_var == -1;
            if (ImGui::Selectable(lbl.c_str(), sel_combo)) writable_var = -1;
        }
        ImGui::EndCombo();
    }
}

// ============================================================
// Parametric: контролы + scatter-plot 1D-бифуркации через наш GL-renderer
// ============================================================
// Рисует контролы одной БД внутри её таба. Возвращает true, если пользователь
// нажал Run для этой БД (внешний код может также взвести Run через Ctrl+R).
static void draw_diagram_controls(BifurcationAnalysisSession& s, int idx) {
    BifurcationDiagramConfig& bd = s.diagrams[idx];

    ImGui::SetNextItemWidth(kComboW);
    if (InputTextStr("Label", bd.label))
        bd.label_is_manual = !bd.label.empty();   // empty → back to auto
    ImGui::Separator();

    // ----- Scheme (built-in + custom) -----
    draw_scheme_combo("Scheme", bd.scheme, s.custom_schemes);
    ImGui::Separator();

    // ----- Sweep target (parameter ИЛИ initial condition) -----
    // Один combo с разделителем: сверху параметры, снизу переменные (IC).
    // Выбор переменной → BD строится по начальному условию (par_or_var = 0
    // в engine), runtime-флаг — никакой пересборки PTX.
    // dt (h) доступен и в continuation: шаг пересчитывается в каждой точке
    // цепочки (см. run_bif1d_continuation / _cpu). Вторую ось гасим только в
    // 2D-режиме — в 1D флаг sweep_over_h_2 не используется и трогать его незачем.
    draw_sweep_target_combo("Sweep", s.params, s.vars,
                            bd.param_index, bd.sweep_over_var, bd.var_sweep_index,
                            bd.sweep_over_h,
                            bd.mode_2d ? &bd.sweep_over_h_2 : nullptr,
                            /*note_when_empty*/ true);
    InputNumStr(bd.sweep_over_h ? "h lo" : "Param lo", bd.param_lo_text, kFieldW);
    InputNumStr(bd.sweep_over_h ? "h hi" : "Param hi", bd.param_hi_text, kFieldW);
    ImGui::Checkbox("Log scale##bd_log", &bd.log_scale);
    if (bd.log_scale) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }

    // Continuation: каждая следующая точка стартует с конечного x[] предыдущей.
    // Единый блок с LLE и LS 1D (см. draw_continuation_device_block) — он же
    // гасит и сбрасывает флаги в 2D-режиме: run_bif2d идёт своим 3-kernel
    // pipeline'ом и continuation игнорирует, скрытое состояние туда утекать не
    // должно. Классический свип у БД CPU-реализации не имеет, поэтому радио
    // GPU/CPU влияет только на continuation.
    draw_continuation_device_block(bd, "bd", bd.mode_2d || bd.sweep_over_var, bd.mode_2d);

    ImGui::Separator();

    // ----- 2D mode: хитмап «период»(p1, p2) через DBSCAN -----
    if (ImGui::Checkbox("2D mode (period heatmap)", &bd.mode_2d) && bd.mode_2d)
        bd.colored_1d = false;   // взаимоисключающе с Colored 1D diagram
    if (bd.mode_2d) {
        ImGui::Indent();
        // Пункт "dt (h)" прячется, если шаг уже занял X: ровно одна ось = h.
        if (!s.params.empty() || !s.vars.empty())
            draw_sweep_target_combo("Sweep Y", s.params, s.vars,
                                    bd.param_index_2, bd.sweep_over_var_2, bd.var_sweep_index_2,
                                    bd.sweep_over_h_2, &bd.sweep_over_h);
        InputNumStr(bd.sweep_over_h_2 ? "h2 lo" : "Param2 lo", bd.param_lo_2_text, kFieldW);
        InputNumStr(bd.sweep_over_h_2 ? "h2 hi" : "Param2 hi", bd.param_hi_2_text, kFieldW);
        ImGui::Checkbox("Log scale##bd_log2", &bd.log_scale_2);
        if (bd.log_scale_2) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
        InputNumStr("DBSCAN eps", bd.eps_dbscan_text, kFieldW);
        // Множители осей кластеризации. Дефолт mult interval = 0 схлопывает
        // ось интервалов, т.е. период считается только по амплитуде пиков.
        InputNumStr("mult peak##bd_mp",     bd.mult_peak_text, kFieldW);
        InputNumStr("mult interval##bd_mi", bd.mult_interval_text, kFieldW);
        ImGui::TextDisabled("Scale the (peak, interval) axes before clustering;");
        ImGui::TextDisabled("0 disables an axis. See Feature diagram in Phase analysis.");
        ImGui::TextDisabled("Grid is square (Resolution applies to both axes).");
        ImGui::Unindent();
    }

    ImGui::Separator();

    // ----- Variable + resolution + inter-peaks -----
    draw_writable_var_combo(s.vars, bd.writable_var, "Writable var##bd_wv");
    InputNumStr("Resolution", bd.n_pts_text, kFieldW);
    if (!bd.mode_2d)
        if (ImGui::Checkbox("Plot inter-peaks instead of peak values", &bd.plot_inter_peaks))
            bd.fit_request = true;

    ImGui::Separator();

    // ----- Colored 1D diagram (collapsible): density-хитмапа поверх
    // классической БД. Вкл/выкл — чекбоксом в тулбаре над самим плотом
    // (draw_bifurcation_plot), не здесь. Здесь — только настройки, видны
    // когда режим уже активен.
    if (bd.colored_1d && ImGui::CollapsingHeader("Colored 1D diagram##bd_c1d", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextDisabled("Toggle this mode from the plot toolbar.");
        InputNumStr("Y resolution", bd.colored_1d_b_text, kFieldW);
        ImGui::TextDisabled("X resolution follows Resolution above.");
        ImGui::Checkbox("Custom Y range", &bd.colored_1d_custom_y);
        if (bd.colored_1d_custom_y) {
            InputNumStr("Ymin", bd.colored_1d_ymin_text, kFieldW);
            InputNumStr("Ymax", bd.colored_1d_ymax_text, kFieldW);
            ImGui::TextDisabled("Points outside [Ymin, Ymax] are discarded.");
        }
        ImGui::Checkbox("Logarithmic density scale", &bd.colored_1d_log);
    }

    // ----- Integration (collapsible) -----
    {
        IntegrationFields f;
        f.h           = &bd.h_text;
        f.symmetry_s  = &bd.symmetry_s;
        f.t_max       = &bd.t_max_text;
        f.transient   = &bd.transient_text;
        f.pre_scaller = &bd.pre_scaller_text;
        f.max_value   = &bd.max_value_text;
        draw_integration_block("Integration##bd_int", bd.scheme, s.custom_schemes, f);
    }

    draw_named_num_fields("Initial conditions##bd_ic", s.vars,   bd.initial_conditions);
    draw_named_num_fields("Parameters##bd_par",        s.params, bd.param_values);
    draw_csv_output_block("CSV output##bd_csv", "Save to file", bd.csv_save_enabled,
                          "##csv_path", bd.csv_output_path,
                          "Path is kept even when save is off. Also writes <path>_config.csv.");

    // Run-кнопка живёт на уровне draw_parametric_controls (рядом с Run all),
    // общая для Bif/LLE/LS. Здесь — только статусная строка.

    if (bd.mode_2d) {
        if (bd.last_run_2d_ok) {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: %dx%d heatmap, period(min..max) = %.0f..%.0f",
                bd.result_2d.n_pts, bd.result_2d.n_pts,
                bd.result_2d.min_val, bd.result_2d.max_val);
            draw_regime_summary(bd.result_2d.flags);
        }
        else if (!bd.last_error.empty()) {
            draw_error_box("##par_err_2d", bd.last_error);
        }
    } else {
        if (bd.last_run_ok) {
            // flags[] здесь — сырой выход peakFinder: N > 0 = число пиков.
            int total_peaks = 0, max_peaks = 0;
            for (int f : bd.result.flags)
                if (regime_is_oscillation(f)) {
                    total_peaks += f;
                    if (f > max_peaks) max_peaks = f;
                }
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: n_pts=%d, peaks total=%d (max per param=%d)",
                bd.result.n_pts, total_peaks, max_peaks);
            draw_regime_summary(bd.result.flags);
        }
        else if (!bd.last_error.empty()) {
            draw_error_box("##par_err", bd.last_error);
        }
    }
}

static void draw_bifurcation_controls(AppModel& model, SystemLibrary& /*lib*/) {
    BifurcationAnalysisSession& s = model.bifurcation_session;

    // Tab bar: одна вкладка на БД + кнопка "+" справа для добавления новой БД
    // (копия последней). Активная вкладка хранится в s.active_diagram_index и
    // используется Ctrl+R. request_select_diagram — внешний запрос выбрать
    // вкладку: его шлёт тулбар Colored 1D над плотом, иначе галка включалась бы
    // у одной БД, а настройки показывались для другой.
    // Run-кнопка + Ctrl+R живут в draw_parametric_controls (общие для Bif/LLE/LS).
    const TabBarResult tabs = draw_config_tab_bar(
        "##bd_tabs", "bd_tab_", (int)s.diagrams.size(),
        s.in_flight, s.running_diagram_index,
        [&s](int i) { return s.diagrams[i].label; },
        [&s](int i) { draw_diagram_controls(s, i); },
        [&s]() { s.add_diagram(); },
        s.request_select_diagram);
    s.request_select_diagram = -1;   // запрос потреблён этим кадром
    if (tabs.active    >= 0) s.active_diagram_index = tabs.active;
    if (tabs.to_remove >= 0) model.remove_bifurcation_diagram(tabs.to_remove);
}

// Маппинг window-kind → нейтральный ParamPlotKind. Сама конфигурация живёт в
// configure_param_plot_view (см. блок общих хелперов вверху файла), чтобы
// Parametric и Custom настраивали одинаковые 1D-графики одним и тем же кодом.
static ParamPlotKind to_param_plot_kind(ParametricPlotWindow::Kind kind) {
    switch (kind) {
    case ParametricPlotWindow::Kind::LLE: return ParamPlotKind::LLE;
    case ParametricPlotWindow::Kind::LS:  return ParamPlotKind::LS;
    case ParametricPlotWindow::Kind::Bifurcation:
    default:                              return ParamPlotKind::Bifurcation;
    }
}

// One-time Plot2DView setup per kind — called by draw_parametric_plot_windows
// when it lazily constructs a fresh Plot2DView for a newly opened window
// (Plot2DView is now owned per-window there, not a function-static shared by
// every window of a kind, so this replaces the old `if (!view) {...}` inits).
static void configure_plot_view(Plot2DView& view, ParametricPlotWindow::Kind kind) {
    configure_param_plot_view(view, to_param_plot_kind(kind));
}

// Plots one Parametric plot window of kind Bifurcation. `win` supplies
// mode_2d and the diagram indices to show (`members`); `renderer`/`view`/
// `heatmap_map` are this window's own render-state instances, owned and
// cached by draw_parametric_plot_windows (keyed by win.id).
static void draw_bifurcation_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                                   ParametricPlotWindow& win,
                                   PlotRenderer& renderer, Plot2DView& view,
                                   std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) {
    BifurcationAnalysisSession& s = model.bifurcation_session;
    // Per-diagram HeatmapView, keyed by diagram index within THIS window's
    // own map (map is already per-window, so no cross-window bleed).
    // Per-diagram persisted choice takes priority; falls back to the shared
    // "last touched anywhere" default for a diagram that's never had its own
    // colormap set (см. get_or_create_heatmap).
    auto get_bd_heatmap = [&](int idx) -> HeatmapView& {
        const int cfg_cm = (idx >= 0 && idx < (int)s.diagrams.size())
                           ? s.diagrams[idx].colormap_idx : -1;
        return get_or_create_heatmap(heatmap_map, idx, cfg_cm, model.heatmap_colormap);
    };

    if (win.members.empty()) {
        ImGui::TextDisabled("No diagrams assigned to this window.");
        return;
    }

    // Тулбар-тумблер над плотом: переключает ЭТО окно между classic scatter
    // и colored-1D heatmap напрямую, без похода в Plot windows → Type →
    // Members. Держит diagram-флаг синхронно с window-флагом. Если членов
    // больше одного (overlay в classic-режиме) — оставляем первого как
    // "фокусного", остальные отбрасываются (heatmap-режимы — single-member,
    // как и раньше при переключении через Type combo).
    // Только для Bifurcation 1D — Colored 1D не имеет смысла поверх настоящего
    // 2D bifurcation (уже heatmap по двум параметрам).
    if (!win.mode_2d) {
        bool c1d = win.colored_1d;
        if (ImGui::Checkbox("Colored 1D diagram##bdtoolbar", &c1d)) {
            win.colored_1d = c1d;
            int focus = win.members[0];
            if (win.members.size() > 1) win.members.resize(1);
            if (focus >= 0 && focus < (int)s.diagrams.size()) {
                s.diagrams[focus].colored_1d = c1d;
                // Переключаем панель контролов на ЭТУ же БД: настройки
                // Colored 1D показываются для диаграммы активной вкладки, и без
                // этого галка включалась бы у одной БД, а блок настроек не
                // появлялся бы (он относился к другой).
                s.request_select_diagram = focus;
                if (!model.loaded_name.empty())
                    lib.save_session(model.loaded_name, "_last_parametric",
                                     session_to_json_parametric(model.bifurcation_session));
            }
            model.parametric_plot_windows_dirty = true;
        }
        // Custom point style — справа от Colored 1D. В colored-режиме плот
        // становится хитмапой, стиль точек к ней неприменим → disabled.
        // Настройка живёт в самой БД, а окно может показывать несколько БД
        // наложением: читаем состояние у фокусного члена (members[0]) и при
        // изменении записываем во ВСЕ члены окна — иначе overlay разъехался бы
        // по стилю (маркер/размер — свойства вида, они одни на окно).
        ImGui::SameLine();
        ImGui::BeginDisabled(win.colored_1d);
        {
            const int focus = win.members[0];
            if (focus >= 0 && focus < (int)s.diagrams.size()) {
                BifurcationDiagramConfig& fbd = s.diagrams[focus];
                if (draw_point_style_toolbar(fbd, "bdtoolbar")) {
                    for (int m : win.members) {
                        if (m < 0 || m >= (int)s.diagrams.size() || m == focus) continue;
                        s.diagrams[m].custom_point_style = fbd.custom_point_style;
                        s.diagrams[m].point_marker       = fbd.point_marker;
                        s.diagrams[m].point_size         = fbd.point_size;
                        s.diagrams[m].point_alpha        = fbd.point_alpha;
                    }
                    if (!model.loaded_name.empty())
                        lib.save_session(model.loaded_name, "_last_parametric",
                                         session_to_json_parametric(model.bifurcation_session));
                }
            }
        }
        ImGui::EndDisabled();
        ImGui::Separator();
    }

    if (win.mode_2d) {
        for (size_t mi = 0; mi < win.members.size(); ++mi) {
            int idx = win.members[mi];
            if (idx < 0 || idx >= (int)s.diagrams.size()) continue;
            BifurcationDiagramConfig& bdact = s.diagrams[idx];
            if (mi > 0) ImGui::Separator();
            ImGui::PushID(idx);

            const unsigned bd_oid = 0xBD2D0000u + (unsigned)idx;
            HeatmapView& hb = get_bd_heatmap(idx);

            // Тулбар цветовой шкалы — общая реализация (см. draw_heatmap_toolbar).
            {
                HeatmapToolbarOpts topts;
                topts.persist_colormap = [&](int cm) {
                    bdact.colormap_idx = cm;   // persist per-diagram only
                    if (!model.loaded_name.empty())
                        lib.save_session(model.loaded_name, "_last_parametric",
                                         session_to_json_parametric(model.bifurcation_session));
                };
                draw_heatmap_toolbar(hb, topts);
            }

            if (!bdact.last_run_2d_ok || bdact.result_2d.values.empty()) {
                ImGui::TextDisabled("No 2D data yet. Press Run.");
                ImGui::PopID();
                continue;
            }

            // Подписи осей — общий auto_axis_name (раньше здесь жила локальная
            // копия, отличавшаяся только fallback'ом "x" вместо "var").
            hb.x_axis.name = auto_axis_name(s.params, s.vars, bdact.param_index,
                                            bdact.sweep_over_var, bdact.var_sweep_index,
                                            bdact.sweep_over_h);
            hb.y_axis.name = auto_axis_name(s.params, s.vars, bdact.param_index_2,
                                            bdact.sweep_over_var_2, bdact.var_sweep_index_2,
                                            bdact.sweep_over_h_2);
            hb.x_axis.log_scale = bdact.log_scale;
            hb.y_axis.log_scale = bdact.log_scale_2;

            bool fit = bdact.fit_request_2d;
            if (fit) bdact.fit_request_2d = false;

            const bool bd_busy = s.in_flight && s.is_2d_run && idx == s.running_diagram_index;
            hb.popup_extras = [&bdact, &cb, bd_busy]() {
                draw_export_menu_item(bd_busy, cb, [&bdact](const std::string& p) {
                    data_export::export_bif2d(bdact.result_2d, p);
                });
            };

            ImVec2 avail = ImGui::GetContentRegionAvail();
            ImVec2 origin = ImGui::GetCursorScreenPos();
            hb.render(renderer, origin, avail,
                      /*owner_id*/ bd_oid, bdact.data_generation_2d,
                      bdact.result_2d.n_pts, bdact.result_2d.n_pts,
                      bdact.result_2d.values.data(),
                      bdact.result_2d.param_lo,   bdact.result_2d.param_hi,
                      bdact.result_2d.param_lo_2, bdact.result_2d.param_hi_2,
                      bdact.result_2d.min_val, bdact.result_2d.max_val,
                      fit);
            ImGui::PopID();
        }
        return;
    }

    if (win.colored_1d) {
        for (size_t mi = 0; mi < win.members.size(); ++mi) {
            int idx = win.members[mi];
            if (idx < 0 || idx >= (int)s.diagrams.size()) continue;
            BifurcationDiagramConfig& bdact = s.diagrams[idx];
            if (mi > 0) ImGui::Separator();
            ImGui::PushID(idx);

            const unsigned c1d_oid = 0xBD1D0000u + (unsigned)idx;
            HeatmapView& hc = get_bd_heatmap(idx);

            {
                HeatmapToolbarOpts topts;
                topts.persist_colormap = [&](int cm) {
                    bdact.colormap_idx = cm;   // per-diagram (shared with mode_2d)
                    if (!model.loaded_name.empty())
                        lib.save_session(model.loaded_name, "_last_parametric",
                                         session_to_json_parametric(model.bifurcation_session));
                };
                draw_heatmap_toolbar(hc, topts);
            }

            if (!bdact.last_run_ok || bdact.result.bifurcation_points.empty()) {
                ImGui::TextDisabled("No 1D data yet. Press Run.");
                ImGui::PopID();
                continue;
            }

            const auto& source = bdact.plot_inter_peaks ? bdact.result.peak_times
                                                         : bdact.result.bifurcation_points;
            int npts = bdact.result.n_pts;
            int B = parse_int_or(bdact.colored_1d_b_text, 2);
            if (B < 2) B = 2;

            // Y-диапазон: авто (глобальный min/max ±2.5%, см. MATLAB Sup/Slow)
            // либо ручной (точки за пределами отбрасываются при биннинге).
            double ylo, yhi;
            if (bdact.colored_1d_custom_y) {
                ylo = parse_ratio_or(bdact.colored_1d_ymin_text, 0.0);
                yhi = parse_ratio_or(bdact.colored_1d_ymax_text, 1.0);
                if (yhi < ylo) std::swap(ylo, yhi);
                if (yhi - ylo < 1e-12) yhi = ylo + 1.0;
            } else {
                double vmin =  std::numeric_limits<double>::infinity();
                double vmax = -std::numeric_limits<double>::infinity();
                for (int k = 0; k < npts && k < (int)source.size(); ++k) {
                    if (k < (int)bdact.result.flags.size() &&
                        !regime_is_oscillation(bdact.result.flags[k])) continue;
                    for (double y : source[k]) {
                        if (y < vmin) vmin = y;
                        if (y > vmax) vmax = y;
                    }
                }
                if (!std::isfinite(vmin) || !std::isfinite(vmax)) { vmin = 0.0; vmax = 1.0; }
                double dd = vmax - vmin;
                if (dd < 1e-12) dd = 1.0;
                ylo = vmin - dd * 0.025;
                yhi = vmax + dd * 0.025;
            }

            bool stale = bdact.colored_1d_built_from       != bdact.data_generation
                      || bdact.colored_1d_cache_b           != B
                      || bdact.colored_1d_cache_log         != bdact.colored_1d_log
                      || bdact.colored_1d_cache_custom_y    != bdact.colored_1d_custom_y
                      || bdact.colored_1d_cache_inter_peaks != bdact.plot_inter_peaks
                      || bdact.colored_1d_cache_ymin_used   != ylo
                      || bdact.colored_1d_cache_ymax_used   != yhi;
            if (stale) {
                size_t plane_size = (size_t)B * (size_t)npts;
                bdact.colored_1d_cache.assign(plane_size, kSentinelNoData);
                std::vector<int> counts((size_t)B);
                double vmin =  std::numeric_limits<double>::infinity();
                double vmax = -std::numeric_limits<double>::infinity();
                double yrange = yhi - ylo;
                for (int k = 0; k < npts; ++k) {
                    bool diverged = k >= (int)bdact.result.flags.size() ||
                                    !regime_is_oscillation(bdact.result.flags[k]);
                    if (diverged) continue;   // остаётся 999 (тот же серый sentinel, что и mode_2d)
                    std::fill(counts.begin(), counts.end(), 0);
                    int colmax = 0;
                    if (k < (int)source.size()) {
                        for (double y : source[k]) {
                            if (bdact.colored_1d_custom_y && (y < ylo || y > yhi)) continue;
                            int bin = (int)std::lround((y - ylo) / yrange * (B - 1));
                            if (bin < 0) bin = 0; else if (bin >= B) bin = B - 1;
                            int cnt = ++counts[(size_t)bin];
                            if (cnt > colmax) colmax = cnt;
                        }
                    }
                    if (colmax <= 0) {
                        // Не diverged, просто ни одна точка не попала в диапазон —
                        // 0-плотность по колонке (не sentinel: это не "нет данных
                        // из-за расхождения траектории", а просто пустой срез).
                        double v0 = bdact.colored_1d_log ? std::log10(1e-6) : 0.0;
                        for (int b = 0; b < B; ++b)
                            bdact.colored_1d_cache[(size_t)b * (size_t)npts + (size_t)k] = v0;
                        continue;
                    }
                    double inv = 1.0 / (double)colmax;
                    for (int b = 0; b < B; ++b) {
                        double density = counts[(size_t)b] * inv;
                        double v = bdact.colored_1d_log ? std::log10(std::max(density, 1e-6)) : density;
                        bdact.colored_1d_cache[(size_t)b * (size_t)npts + (size_t)k] = v;
                        if (v < vmin) vmin = v;
                        if (v > vmax) vmax = v;
                    }
                }
                bdact.colored_1d_cache_vmin = std::isfinite(vmin) ? vmin : 0.0;
                bdact.colored_1d_cache_vmax = std::isfinite(vmax) ? vmax : 1.0;
                bdact.colored_1d_built_from      = bdact.data_generation;
                bdact.colored_1d_cache_b         = B;
                bdact.colored_1d_cache_log       = bdact.colored_1d_log;
                bdact.colored_1d_cache_custom_y  = bdact.colored_1d_custom_y;
                bdact.colored_1d_cache_inter_peaks = bdact.plot_inter_peaks;
                bdact.colored_1d_cache_ymin_used = ylo;
                bdact.colored_1d_cache_ymax_used = yhi;
                ++bdact.colored_1d_cache_gen;
            }

            hc.x_axis.name = auto_axis_name(s.params, s.vars, bdact.param_index,
                                             bdact.sweep_over_var, bdact.var_sweep_index, bdact.sweep_over_h);
            hc.x_axis.log_scale = bdact.log_scale;
            hc.y_axis.name = (bdact.writable_var >= 0 && bdact.writable_var < (int)s.vars.size())
                            ? s.vars[bdact.writable_var] : "X";
            if (bdact.plot_inter_peaks) hc.y_axis.name += " interval";

            bool fit = bdact.fit_request;
            if (fit) bdact.fit_request = false;

            const bool c1d_busy = s.in_flight && !s.is_2d_run && idx == s.running_diagram_index;
            hc.popup_extras = [&bdact, &cb, c1d_busy]() {
                draw_export_menu_item(c1d_busy, cb, [&bdact](const std::string& p) {
                    data_export::export_bif1d(bdact.result, p);
                });
            };

            // Тот же lo/hi/reverse-разбор, что и у классического scatter'а
            // ниже (continuation хранит снапшот в result, иначе — из текстов).
            double lo = (bdact.continuation && bdact.result.param_hi != bdact.result.param_lo)
                        ? bdact.result.param_lo : parse_ratio_or(bdact.param_lo_text, 0.0);
            double hi = (bdact.continuation && bdact.result.param_hi != bdact.result.param_lo)
                        ? bdact.result.param_hi : parse_ratio_or(bdact.param_hi_text, 1.0);
            bool rev = bdact.result.continuation_reverse;
            double x0 = rev ? hi : lo;
            double x1 = rev ? lo : hi;

            ImVec2 avail = ImGui::GetContentRegionAvail();
            ImVec2 origin = ImGui::GetCursorScreenPos();
            hc.render(renderer, origin, avail,
                      /*owner_id*/ c1d_oid, bdact.colored_1d_cache_gen,
                      npts, B,
                      bdact.colored_1d_cache.data(),
                      x0, x1,
                      ylo, yhi,
                      bdact.colored_1d_cache_vmin, bdact.colored_1d_cache_vmax,
                      fit);
            ImGui::PopID();
        }
        return;
    }

    // Имеется ли хотя бы одна БД этого окна с готовыми данными?
    bool any_data = false;
    for (int idx : win.members) {
        if (idx < 0 || idx >= (int)s.diagrams.size()) continue;
        const auto& bd = s.diagrams[idx];
        if (bd.last_run_ok && !bd.result.bifurcation_points.empty()) { any_data = true; break; }
    }
    if (!any_data) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }

    // Подписи осей + X-fit диапазон (см. configure_sweep_x_axis). Заодно
    // собираем writable_var по членам — он нужен для Y-подписи ниже.
    int shared_var_idx = -2;   // -2 init, -1 combination, [0,N) — single var
    bool var_idx_mixed = false;  // отдельный флаг — нельзя реюзать -1 (теперь это combination)
    for (int idx : win.members) {
        if (idx < 0 || idx >= (int)s.diagrams.size()) continue;
        const auto& bd = s.diagrams[idx];
        if (!bd.last_run_ok) continue;
        if (shared_var_idx == -2) shared_var_idx = bd.writable_var;
        else if (shared_var_idx != bd.writable_var) var_idx_mixed = true;
    }
    // continuation сохраняет реальные lo/hi в result, иначе — из текстов.
    configure_sweep_x_axis_from(view, win.members, s.diagrams, s.params, s.vars,
        [](const BifurcationDiagramConfig& bd, double& lo, double& hi) {
            const bool from_result = bd.continuation && bd.result.param_hi != bd.result.param_lo;
            lo = from_result ? bd.result.param_lo : parse_ratio_or(bd.param_lo_text, 0.0);
            hi = from_result ? bd.result.param_hi : parse_ratio_or(bd.param_hi_text, 1.0);
        });
    // Y-label: combination (-1) -> "x0+pi*x1+e*x2"; single var -> имя переменной;
    // mixed / нет данных -> generic "X".
    if (var_idx_mixed || shared_var_idx == -2) {
        view.y_axis.name = "X";
    } else if (shared_var_idx == -1) {
        // Комбинация — общая подпись combo_var_label. Раньше формула была
        // выписана здесь ТРЕТЬЕЙ копией и печаталась без пробелов
        // ("x+pi*y+e*z"); теперь подпись оси совпадает с пунктом комбо
        // ("x + pi*y + e*z"), из которого пользователь эту комбинацию и выбрал.
        view.y_axis.name = s.vars.empty() ? std::string("X") : combo_var_label(s.vars);
    } else if (shared_var_idx >= 0 && shared_var_idx < (int)s.vars.size()) {
        view.y_axis.name = s.vars[shared_var_idx];
    } else {
        view.y_axis.name = "X";
    }

    // Буферы точек (по одному на серию), свои у каждого окна — см.
    // window_point_bufs: между кадрами они держат capacity.
    auto& bufs = window_point_bufs(win.id, win.members.size());

    std::vector<PlotSeriesInput> series_in;
    std::vector<bool> init_vis;
    std::vector<bool> glob_vis;
    series_in.reserve(win.members.size());
    init_vis.reserve(win.members.size());
    glob_vis.reserve(win.members.size());

    bool any_fit = false;
    int  data_gen = 0;

    for (size_t mi = 0; mi < win.members.size(); ++mi) {
        int idx = win.members[mi];
        if (idx < 0 || idx >= (int)s.diagrams.size()) continue;
        BifurcationDiagramConfig& bd = s.diagrams[idx];
        auto& buf = bufs[mi];
        buf.clear();
        int total_pts = 0;

        if (bd.last_run_ok && !bd.result.bifurcation_points.empty()) {
            const auto& source = bd.plot_inter_peaks ? bd.result.peak_times
                                                     : bd.result.bifurcation_points;
            // Continuation result хранит param_lo/hi-снапшот. Classical путь
            // его не заполняет — fallback на текущие текстовые поля.
            double lo = (bd.continuation && bd.result.param_hi != bd.result.param_lo)
                        ? bd.result.param_lo : parse_ratio_or(bd.param_lo_text, 0.0);
            double hi = (bd.continuation && bd.result.param_hi != bd.result.param_lo)
                        ? bd.result.param_hi : parse_ratio_or(bd.param_hi_text, 1.0);
            bool rev = bd.result.continuation_reverse;
            int npts = bd.result.n_pts;
            for (int k = 0; k < npts; ++k) {
                if (k < (int)bd.result.flags.size() &&
                    !regime_is_oscillation(bd.result.flags[k])) continue;
                double x = sweep_value_at(k, npts, lo, hi, bd.log_scale, rev, bd.continuation);
                if (k >= (int)source.size()) continue;
                for (double y : source[k]) {
                    buf.push_back((float)x);
                    buf.push_back((float)y);
                    ++total_pts;
                }
            }
        }

        PlotSeriesInput si;
        si.points   = buf.empty() ? nullptr : buf.data();
        si.n_points = total_pts;
        si.color    = ic_base_color((int)mi);
        // α в custom-режиме — через альфу цвета серии (GL-путь её блендит),
        // как в Custom line style у Phase2D/TimeDomain.
        if (bd.custom_point_style) si.color.w = bd.point_alpha;
        si.label    = bd.label;
        series_in.push_back(si);
        init_vis.push_back(true);
        glob_vis.push_back(true);   // membership in this window IS the visibility gate

        // data_gen накапливает per-BD generation + toggle inter_peaks
        // (как было в одно-БД версии — чтобы тоггл триггерил перерисовку VBO).
        data_gen = data_gen * 31 + bd.data_generation * 2 + (bd.plot_inter_peaks ? 1 : 0);
        if (bd.fit_request) { any_fit = true; bd.fit_request = false; }
    }

    // Right-click "Export data..." for the bifurcation 1D line plot. The
    // submenu lists every diagram in the session that has a finished run
    // (not just this window's members — export isn't scoped to a window).
    view.popup_extras = [&s, &cb]() {
        draw_export_submenu("bd", (int)s.diagrams.size(),
            [&s](int i) { return s.diagrams[i].label; },
            [&s](int i) { return s.diagrams[i].last_run_ok; },
            [&s](int i) { return s.in_flight && i == s.running_diagram_index; },
            [&s](int i, const std::string& p) { data_export::export_bif1d(s.diagrams[i].result, p); },
            cb);
    };

    // Snap X к узлам первой БД этого окна (см. apply_snap_x_from_first_member).
    apply_snap_x_from_first_member(view, win.members, s.diagrams);

    // Маркер/размер точек — свойства вида (одни на окно): берём у фокусного
    // члена, тулбар выше держит остальных членов синхронно.
    {
        const int aidx = win.members.empty() ? -1 : win.members[0];
        if (aidx >= 0 && aidx < (int)s.diagrams.size())
            apply_point_style(view, s.diagrams[aidx]);
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();

    view.render(renderer, origin, avail, /*owner_id*/ 0xBE0F1D, data_gen,
                series_in, init_vis, glob_vis, any_fit);
}

// ============================================================
// LLE: контролы (per-curve в табе) + line-plot λ(param)
// ============================================================

// Контролы одной LLE-кривой. Возвращает true, если пользователь нажал Run
// для этой кривой.
static void draw_lle_curve_controls(LLEAnalysisSession& s, int idx) {
    LLECurveConfig& c = s.curves[idx];

    ImGui::SetNextItemWidth(kComboW);
    if (InputTextStr("Label", c.label))
        c.label_is_manual = !c.label.empty();   // empty → back to auto
    ImGui::Separator();

    draw_scheme_combo("Scheme", c.scheme, s.custom_schemes);
    ImGui::Separator();

    // Sweep target: параметры + разделитель + переменные (IC) + dt (h). См. BD.
    draw_sweep_target_combo("Sweep", s.params, s.vars,
                            c.param_index, c.sweep_over_var, c.var_sweep_index,
                            c.sweep_over_h,
                            c.mode_2d ? &c.sweep_over_h_2 : nullptr,
                            /*note_when_empty*/ true);
    InputNumStr(c.sweep_over_h ? "h lo" : "Param lo", c.param_lo_text, kFieldW);
    InputNumStr(c.sweep_over_h ? "h hi" : "Param hi", c.param_hi_text, kFieldW);
    ImGui::Checkbox("Log scale##lle_log", &c.log_scale);
    if (c.log_scale) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
    InputNumStr("Resolution", c.n_pts_text, kFieldW);

    draw_continuation_device_block(c, "lle", c.mode_2d || c.sweep_over_var, c.mode_2d);


    ImGui::Separator();
    // 2D-режим. Сетка квадратная (см. analysis_session.h:LLECurveConfig коммент)
    // — Resolution выше применяется и к X, и к Y.
    ImGui::Checkbox("2D mode (heatmap)", &c.mode_2d);
    if (c.mode_2d) {
        ImGui::Indent();
        // Sweep target для второй оси — то же комбо, симметрично первой.
        // Пункт "dt (h)" прячется, если шаг уже занял X: ровно одна ось = h.
        if (!s.params.empty() || !s.vars.empty())
            draw_sweep_target_combo("Sweep Y", s.params, s.vars,
                                    c.param_index_2, c.sweep_over_var_2, c.var_sweep_index_2,
                                    c.sweep_over_h_2, &c.sweep_over_h);
        InputNumStr(c.sweep_over_h_2 ? "h2 lo" : "Param2 lo", c.param_lo_2_text, kFieldW);
        InputNumStr(c.sweep_over_h_2 ? "h2 hi" : "Param2 hi", c.param_hi_2_text, kFieldW);
        ImGui::Checkbox("Log scale##lle_log2", &c.log_scale_2);
        if (c.log_scale_2) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
        ImGui::TextDisabled("Grid is square (Resolution applies to both axes).");
        ImGui::Unindent();
    }

    ImGui::Separator();

    // ----- Integration (collapsible) -----
    {
        IntegrationFields f;
        f.h          = &c.h_text;
        f.symmetry_s = &c.symmetry_s;
        f.t_max      = &c.t_max_text;
        f.transient  = &c.transient_text;
        f.max_value  = &c.max_value_text;   // decimator'а у LLE нет
        draw_integration_block("Integration##lle_int", c.scheme, s.custom_schemes, f);
    }

    // ----- LLE (Wolf/Benettin) (collapsible) -----
    if (ImGui::CollapsingHeader("LLE (Wolf/Benettin)##lle_wb", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("eps", c.eps_text, kFieldW);
        InputNumStr("NT",  c.nt_text, kFieldW);
        ImGui::TextDisabled("eps = initial perturbation magnitude; NT = block length\n"
                            "between renormalizations (in time units).");
    }

    draw_named_num_fields("Initial conditions##lle_ic", s.vars,   c.initial_conditions);
    draw_named_num_fields("Parameters##lle_par",        s.params, c.param_values);
    draw_csv_output_block("CSV output##lle_csv", "Save to file", c.csv_save_enabled,
                          "##lle_csv_path", c.csv_output_path,
                          "Path is kept even when save is off. Also writes <path>_config.csv.");

    // Run-кнопка живёт на уровне draw_parametric_controls.

    if (c.mode_2d) {
        if (c.last_run_2d_ok) {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: %dx%d heatmap, lambda(min..max) = %.4g..%.4g",
                c.result_2d.n_pts, c.result_2d.n_pts,
                c.result_2d.min_val, c.result_2d.max_val);
            draw_regime_summary(c.result_2d.flags);
        }
        else if (!c.last_error.empty()) {
            draw_error_box("##lle_err", c.last_error);
        }
    } else {
        if (c.last_run_ok) {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: n_pts=%d, lambda-curve computed", c.result.n_pts);
            draw_regime_summary(c.result.flags);
        }
        else if (!c.last_error.empty()) {
            draw_error_box("##lle_err", c.last_error);
        }
    }
}

// Tab bar по LLE-кривым + кнопка «+» (копирует последнюю).
static void draw_lle_controls(AppModel& model, SystemLibrary& /*lib*/) {
    LLEAnalysisSession& s = model.lle_session;

    // Run + Ctrl+R — в draw_parametric_controls (общая кнопка слева от Run all).
    const TabBarResult tabs = draw_config_tab_bar(
        "##lle_tabs", "lle_tab_", (int)s.curves.size(),
        s.in_flight, s.running_curve_index,
        [&s](int i) { return s.curves[i].label; },
        [&s](int i) { draw_lle_curve_controls(s, i); },
        [&s]() { s.add_curve(); });
    if (tabs.active    >= 0) s.active_curve_index = tabs.active;
    if (tabs.to_remove >= 0) model.remove_lle_curve(tabs.to_remove);
}

// Plot LLE: линии (points_mode=false). Каждая кривая — λ(param).
// При mode_2d=true у активной кривой вместо линий рисуется HeatmapView
// (квадратная сетка λ(p1, p2) с colormap'ом).
// Plots one Parametric plot window of kind LLE. Same per-window-state shape
// as draw_bifurcation_plot (see there for the rationale).
static void draw_lle_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                          const ParametricPlotWindow& win,
                          PlotRenderer& renderer, Plot2DView& view,
                          std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) {
    LLEAnalysisSession& s = model.lle_session;
    auto get_lle_heatmap = [&](int idx) -> HeatmapView& {
        const int cfg_cm = (idx >= 0 && idx < (int)s.curves.size())
                           ? s.curves[idx].colormap_idx : -1;
        return get_or_create_heatmap(heatmap_map, idx, cfg_cm, model.heatmap_colormap);
    };

    if (win.members.empty()) {
        ImGui::TextDisabled("No curves assigned to this window.");
        return;
    }

    if (win.mode_2d) {
        for (size_t mi = 0; mi < win.members.size(); ++mi) {
            int idx = win.members[mi];
            if (idx < 0 || idx >= (int)s.curves.size()) continue;
            LLECurveConfig& cact = s.curves[idx];
            if (mi > 0) ImGui::Separator();
            ImGui::PushID(idx);

            const unsigned lle_oid = 0xBE110000u + (unsigned)idx;
            HeatmapView& heatmap = get_lle_heatmap(idx);

            {
                HeatmapToolbarOpts topts;
                topts.persist_colormap = [&](int cm) {
                    cact.colormap_idx = cm;   // persist per-curve only
                    if (!model.loaded_name.empty())
                        lib.save_session(model.loaded_name, "_last_lle",
                                         session_to_json_lle(model.lle_session));
                };
                draw_heatmap_toolbar(heatmap, topts);
            }

            if (!cact.last_run_2d_ok || cact.result_2d.values.empty()) {
                ImGui::TextDisabled("No 2D data yet. Press Run.");
                ImGui::PopID();
                continue;
            }

            // Подписи осей по реальным selected-полям свипа — общий auto_axis_name.
            heatmap.x_axis.name = auto_axis_name(s.params, s.vars, cact.param_index,
                                                 cact.sweep_over_var, cact.var_sweep_index,
                                                 cact.sweep_over_h);
            heatmap.y_axis.name = auto_axis_name(s.params, s.vars, cact.param_index_2,
                                                 cact.sweep_over_var_2, cact.var_sweep_index_2,
                                                 cact.sweep_over_h_2);
            heatmap.x_axis.log_scale = cact.log_scale;
            heatmap.y_axis.log_scale = cact.log_scale_2;

            bool fit = cact.fit_request_2d;
            if (fit) cact.fit_request_2d = false;

            const bool busy = s.in_flight && s.is_2d_run && idx == s.running_curve_index;
            heatmap.popup_extras = [&cact, &cb, busy]() {
                draw_export_menu_item(busy, cb, [&cact](const std::string& p) {
                    data_export::export_lle2d(cact.result_2d, p);
                });
            };

            ImVec2 avail = ImGui::GetContentRegionAvail();
            ImVec2 origin = ImGui::GetCursorScreenPos();
            heatmap.render(renderer, origin, avail,
                           /*owner_id*/ lle_oid, cact.data_generation_2d,
                           cact.result_2d.n_pts, cact.result_2d.n_pts,
                           cact.result_2d.values.data(),
                           cact.result_2d.param_lo,   cact.result_2d.param_hi,
                           cact.result_2d.param_lo_2, cact.result_2d.param_hi_2,
                           cact.result_2d.min_val, cact.result_2d.max_val,
                           fit);
            ImGui::PopID();
        }
        return;
    }

    bool any_data = false;
    for (int idx : win.members) {
        if (idx < 0 || idx >= (int)s.curves.size()) continue;
        const auto& c = s.curves[idx];
        if (c.last_run_ok && !c.result.lyapunov.empty()) { any_data = true; break; }
    }
    if (!any_data) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }


    // Подпись X + X-fit диапазон — см. configure_sweep_x_axis_from.
    configure_sweep_x_axis_from(view, win.members, s.curves, s.params, s.vars,
        [](const LLECurveConfig& c, double& lo, double& hi) {
            sweep_range_from_result(c.result.param_lo, c.result.param_hi,
                                    c.param_lo_text, c.param_hi_text, lo, hi);
        });
    view.y_axis.name = "lambda";

    auto& bufs = window_point_bufs(win.id, win.members.size());

    std::vector<PlotSeriesInput> series_in;
    std::vector<bool> init_vis, glob_vis;
    series_in.reserve(win.members.size());
    init_vis.reserve(win.members.size());
    glob_vis.reserve(win.members.size());

    bool any_fit = false;
    int  data_gen = 0;

    for (size_t mi = 0; mi < win.members.size(); ++mi) {
        int idx = win.members[mi];
        if (idx < 0 || idx >= (int)s.curves.size()) continue;
        LLECurveConfig& c = s.curves[idx];
        auto& buf = bufs[mi];
        buf.clear();
        int total_pts = 0;

        if (c.last_run_ok && !c.result.lyapunov.empty()) {
            // X считаются по тому диапазону, с которым реально шёл Run, а не
            // по текущим полям GUI — иначе кривая «прыгает» при редактировании
            // param_lo/hi до следующего Run.
            double lo = c.result.param_lo;
            double hi = c.result.param_hi;
            int npts = c.result.n_pts;
            // При backward-continuation точка k считалась для hi-(hi-lo)*k/(n-1)
            // (см. run_lle1d_continuation_cpu) — иначе кривая была бы зеркальной.
            const bool rev = c.result.continuation_reverse;
            for (int k = 0; k < npts; ++k) {
                if (k < (int)c.result.flags.size() &&
                    !regime_is_oscillation(c.result.flags[k])) continue;
                double x = sweep_value_at(k, npts, lo, hi, c.log_scale, rev, c.continuation);
                double y = c.result.lyapunov[k];
                if (!std::isfinite(y)) continue;
                buf.push_back((float)x);
                buf.push_back((float)y);
                ++total_pts;
            }
        }

        PlotSeriesInput si;
        si.points   = buf.empty() ? nullptr : buf.data();
        si.n_points = total_pts;
        si.color    = ic_base_color((int)mi);
        si.label    = c.label;
        series_in.push_back(si);
        init_vis.push_back(true);
        glob_vis.push_back(true);   // membership in this window IS the visibility gate

        data_gen = data_gen * 31 + c.data_generation;
        if (c.fit_request) { any_fit = true; c.fit_request = false; }
    }

    view.popup_extras = [&s, &cb]() {
        draw_export_submenu("lle", (int)s.curves.size(),
            [&s](int i) { return s.curves[i].label; },
            [&s](int i) { return s.curves[i].last_run_ok; },
            [&s](int i) { return s.in_flight && !s.is_2d_run && i == s.running_curve_index; },
            [&s](int i, const std::string& p) { data_export::export_lle1d(s.curves[i].result, p); },
            cb);
    };

    // Snap X к узлам первой кривой этого окна (см. apply_snap_x_from_first_member).
    apply_snap_x_from_first_member(view, win.members, s.curves);

    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    view.render(renderer, origin, avail, /*owner_id*/ 0xBE11E5, data_gen,
                series_in, init_vis, glob_vis, any_fit);
}

// ============================================================
// LS: контролы (per-curve в табе) + line-plot λ_k(param), N экспонент
// на один спектр-«прогон». UX зеркало LLE.
// ============================================================

static void draw_ls_curve_controls(LyapunovSpectrumAnalysisSession& s, int idx) {
    LSCurveConfig& c = s.curves[idx];

    ImGui::SetNextItemWidth(kComboW);
    if (InputTextStr("Label", c.label))
        c.label_is_manual = !c.label.empty();   // empty → back to auto
    ImGui::Separator();

    draw_scheme_combo("Scheme", c.scheme, s.custom_schemes);
    ImGui::Separator();

    // Sweep target: параметры + разделитель + переменные (IC) + dt (h). См. BD.
    draw_sweep_target_combo("Sweep", s.params, s.vars,
                            c.param_index, c.sweep_over_var, c.var_sweep_index,
                            c.sweep_over_h,
                            c.mode_2d ? &c.sweep_over_h_2 : nullptr,
                            /*note_when_empty*/ true);
    InputNumStr(c.sweep_over_h ? "h lo" : "Param lo", c.param_lo_text, kFieldW);
    InputNumStr(c.sweep_over_h ? "h hi" : "Param hi", c.param_hi_text, kFieldW);
    ImGui::Checkbox("Log scale##ls_log", &c.log_scale);
    if (c.log_scale) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
    InputNumStr("Resolution", c.n_pts_text, kFieldW);

    draw_continuation_device_block(c, "ls", c.mode_2d || c.sweep_over_var, c.mode_2d);

    ImGui::Separator();
    // 2D-режим. Сетка квадратная (см. LLE 2D — то же ограничение getValueByIdx).
    ImGui::Checkbox("2D mode (heatmap of one exponent)", &c.mode_2d);
    if (c.mode_2d) {
        ImGui::Indent();
        // Пункт "dt (h)" прячется, если шаг уже занял X: ровно одна ось = h.
        if (!s.params.empty() || !s.vars.empty())
            draw_sweep_target_combo("Sweep Y", s.params, s.vars,
                                    c.param_index_2, c.sweep_over_var_2, c.var_sweep_index_2,
                                    c.sweep_over_h_2, &c.sweep_over_h);
        InputNumStr(c.sweep_over_h_2 ? "h2 lo" : "Param2 lo", c.param_lo_2_text, kFieldW);
        InputNumStr(c.sweep_over_h_2 ? "h2 hi" : "Param2 hi", c.param_hi_2_text, kFieldW);
        ImGui::Checkbox("Log scale##ls_log2", &c.log_scale_2);
        if (c.log_scale_2) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
        ImGui::TextDisabled("Grid is square (Resolution applies to both axes).\nAll N exponents computed; switch in plot window.");
        ImGui::Unindent();
    }

    ImGui::Separator();

    // ----- Integration (collapsible) -----
    {
        IntegrationFields f;
        f.h          = &c.h_text;
        f.symmetry_s = &c.symmetry_s;
        f.t_max      = &c.t_max_text;
        f.transient  = &c.transient_text;
        f.max_value  = &c.max_value_text;   // decimator'а у LS нет
        draw_integration_block("Integration##ls_int", c.scheme, s.custom_schemes, f);
    }

    // ----- LS (Wolf/Benettin + Gram-Schmidt) (collapsible) -----
    if (ImGui::CollapsingHeader("LS (Wolf/Benettin + Gram-Schmidt)##ls_wbgs", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("eps", c.eps_text, kFieldW);
        InputNumStr("NT",  c.nt_text, kFieldW);
        ImGui::TextDisabled("eps = initial perturbation magnitude; NT = block length\n"
                            "between renormalizations (in time units).");
    }

    draw_named_num_fields("Initial conditions##ls_ic", s.vars,   c.initial_conditions);
    draw_named_num_fields("Parameters##ls_par",        s.params, c.param_values);
    draw_csv_output_block("CSV output##ls_csv", "Save to file", c.csv_save_enabled,
                          "##ls_csv_path", c.csv_output_path,
                          "Path is kept even when save is off. Also writes <path>_config.csv.");

    // Run-кнопка живёт на уровне draw_parametric_controls.

    if (c.mode_2d) {
        if (c.last_run_2d_ok) {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: %dx%d heatmap, %d exponents",
                c.result_2d.n_pts, c.result_2d.n_pts, c.result_2d.n_exponents);
            draw_regime_summary(c.result_2d.flags);
        }
        else if (!c.last_error.empty()) {
            draw_error_box("##ls_err_2d", c.last_error);
        }
    } else {
        if (c.last_run_ok) {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: n_pts=%d, n_exponents=%d", c.result.n_pts, c.result.n_exponents);
            draw_regime_summary(c.result.flags);
        }
        else if (!c.last_error.empty()) {
            draw_error_box("##ls_err", c.last_error);
        }
    }
}

static void draw_ls_controls(AppModel& model, SystemLibrary& /*lib*/) {
    LyapunovSpectrumAnalysisSession& s = model.ls_session;

    // Run + Ctrl+R — в draw_parametric_controls (общая кнопка слева от Run all).
    const TabBarResult tabs = draw_config_tab_bar(
        "##ls_tabs", "ls_tab_", (int)s.curves.size(),
        s.in_flight, s.running_curve_index,
        [&s](int i) { return s.curves[i].label; },
        [&s](int i) { draw_ls_curve_controls(s, i); },
        [&s]() { s.add_curve(); });
    if (tabs.active    >= 0) s.active_curve_index = tabs.active;
    if (tabs.to_remove >= 0) model.remove_ls_curve(tabs.to_remove);
}

// Plot LS: каждая кривая раскладывается на N лiний (по числу экспонент).
// Серия: spectrum_idx * N + exponent_idx. Цвета через ic_base_color(seq).
// При mode_2d=true у активной кривой вместо линий рисуется HeatmapView с
// одной выбранной экспонентой; combo "Exponent" над хитмапой переключает
// плоскость без повторного Run.
// Plots one Parametric plot window of kind LS. Same per-window-state shape
// as draw_bifurcation_plot (see there for the rationale).
static void draw_ls_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                         const ParametricPlotWindow& win,
                         PlotRenderer& renderer, Plot2DView& view,
                         std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) {
    LyapunovSpectrumAnalysisSession& s = model.ls_session;
    auto get_ls_heatmap = [&](int idx) -> HeatmapView& {
        const bool in_range = (idx >= 0 && idx < (int)s.curves.size());
        const int  cfg_cm   = in_range ? s.curves[idx].colormap_idx : -1;
        const int  cfg_exp  = in_range ? s.curves[idx].display_exponent_idx : kNoExponent;
        return get_or_create_heatmap(heatmap_map, idx, cfg_cm, model.heatmap_colormap, cfg_exp);
    };

    if (win.members.empty()) {
        ImGui::TextDisabled("No spectra assigned to this window.");
        return;
    }

    if (win.mode_2d) {
        for (size_t mi = 0; mi < win.members.size(); ++mi) {
            int idx = win.members[mi];
            if (idx < 0 || idx >= (int)s.curves.size()) continue;
            LSCurveConfig& cact = s.curves[idx];
            if (mi > 0) ImGui::Separator();
            ImGui::PushID(idx);

            const unsigned ls_oid = 0x15A20000u + (unsigned)idx;
            HeatmapView& heatmap_ls = get_ls_heatmap(idx);

            {
                auto save_ls = [&]() {
                    if (!model.loaded_name.empty())
                        lib.save_session(model.loaded_name, "_last_ls",
                                         session_to_json_ls(model.ls_session));
                };
                HeatmapToolbarOpts topts;
                topts.persist_colormap = [&](int cm) {
                    cact.colormap_idx = cm;   // persist per-curve only
                    save_ls();
                };
                topts.extras = [&]() {
                    if (!cact.last_run_2d_ok) return;
                    draw_ls_exponent_picker(heatmap_ls, cact.result_2d.n_exponents,
                                            [&](int j) {
                                                cact.display_exponent_idx = j;
                                                save_ls();
                                            });
                };
                draw_heatmap_toolbar(heatmap_ls, topts);
            }

            if (!cact.last_run_2d_ok || cact.result_2d.values.empty()) {
                ImGui::TextDisabled("No 2D data yet. Press Run.");
                ImGui::PopID();
                continue;
            }

            // Подписи осей по реальным selected-полям свипа — общий auto_axis_name.
            heatmap_ls.x_axis.name = auto_axis_name(s.params, s.vars, cact.param_index,
                                                    cact.sweep_over_var, cact.var_sweep_index,
                                                    cact.sweep_over_h);
            heatmap_ls.y_axis.name = auto_axis_name(s.params, s.vars, cact.param_index_2,
                                                    cact.sweep_over_var_2, cact.var_sweep_index_2,
                                                    cact.sweep_over_h_2);
            heatmap_ls.x_axis.log_scale = cact.log_scale;
            heatmap_ls.y_axis.log_scale = cact.log_scale_2;

            bool fit = cact.fit_request_2d;
            if (fit) cact.fit_request_2d = false;

            // Плоскость по выбранной экспоненте (см. ls_resolve_plane).
            const double* plane_ptr = nullptr;
            double vmin = 0.0, vmax = 0.0;
            int    gen  = 0;
            ls_resolve_plane(cact, heatmap_ls.display_exponent_idx,
                             plane_ptr, vmin, vmax, gen);

            const bool busy = s.in_flight && s.is_2d_run && idx == s.running_curve_index;
            heatmap_ls.popup_extras = [&cact, &cb, busy]() {
                draw_export_menu_item(busy, cb, [&cact](const std::string& p) {
                    data_export::export_ls2d(cact.result_2d, p);
                });
            };

            ImVec2 avail = ImGui::GetContentRegionAvail();
            ImVec2 origin = ImGui::GetCursorScreenPos();
            heatmap_ls.render(renderer, origin, avail,
                              /*owner_id*/ ls_oid, gen,
                              cact.result_2d.n_pts, cact.result_2d.n_pts,
                              plane_ptr,
                              cact.result_2d.param_lo,   cact.result_2d.param_hi,
                              cact.result_2d.param_lo_2, cact.result_2d.param_hi_2,
                              vmin, vmax,
                              fit);
            ImGui::PopID();
        }
        return;
    }

    bool any_data = false;
    for (int idx : win.members) {
        if (idx < 0 || idx >= (int)s.curves.size()) continue;
        const auto& c = s.curves[idx];
        if (c.last_run_ok && !c.result.spectrum.empty()) { any_data = true; break; }
    }
    if (!any_data) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }


    // Подпись X + X-fit диапазон — см. configure_sweep_x_axis_from.
    configure_sweep_x_axis_from(view, win.members, s.curves, s.params, s.vars,
        [](const LSCurveConfig& c, double& lo, double& hi) {
            sweep_range_from_result(c.result.param_lo, c.result.param_hi,
                                    c.param_lo_text, c.param_hi_text, lo, hi);
        });

    // Подсчитываем общее число серий этого окна — sum(n_exponents per member).
    size_t total_series = 0;
    for (int idx : win.members) {
        if (idx < 0 || idx >= (int)s.curves.size()) continue;
        const auto& c = s.curves[idx];
        int N = c.result.n_exponents > 0 ? c.result.n_exponents : (int)s.vars.size();
        total_series += (size_t)N;
    }

    auto& bufs = window_point_bufs(win.id, total_series);

    std::vector<PlotSeriesInput> series_in;
    std::vector<bool> init_vis, glob_vis;
    series_in.reserve(total_series);
    init_vis.reserve(total_series);
    glob_vis.reserve(total_series);

    bool any_fit = false;
    int  data_gen = 0;
    size_t buf_cursor = 0;
    int    series_idx = 0;

    for (int idx : win.members) {
        if (idx < 0 || idx >= (int)s.curves.size()) continue;
        LSCurveConfig& c = s.curves[idx];
        int N = c.result.n_exponents > 0 ? c.result.n_exponents : (int)s.vars.size();

        // X — по диапазону, с которым шёл Run (см. LLE-плот).
        double lo = c.result.param_lo;
        double hi = c.result.param_hi;
        int npts = c.result.n_pts;
        // При backward-continuation точка k считалась для hi-(hi-lo)*k/(n-1)
        // (см. run_ls1d_cpu) — иначе кривая была бы зеркальной.
        const bool rev = c.result.continuation_reverse;
        bool have = c.last_run_ok && !c.result.spectrum.empty();

        for (int j = 0; j < N; ++j) {
            auto& buf = bufs[buf_cursor++];
            buf.clear();
            int total_pts = 0;

            if (have) {
                for (int k = 0; k < npts; ++k) {
                    if (k < (int)c.result.flags.size() &&
                        !regime_is_oscillation(c.result.flags[k])) continue;
                    if (k >= (int)c.result.spectrum.size()) continue;
                    const auto& row = c.result.spectrum[k];
                    if (j >= (int)row.size()) continue;
                    double x = sweep_value_at(k, npts, lo, hi, c.log_scale, rev, c.continuation);
                    double y = row[j];
                    if (!std::isfinite(y)) continue;
                    buf.push_back((float)x);
                    buf.push_back((float)y);
                    ++total_pts;
                }
            }

            PlotSeriesInput si;
            si.points   = buf.empty() ? nullptr : buf.data();
            si.n_points = total_pts;
            si.color    = ic_base_color(series_idx++);
            // label: "<spectrum> Lj" если N>1, иначе просто <spectrum>.
            std::string lab = (N > 1) ? (c.label + " L" + std::to_string(j + 1)) : c.label;
            si.label    = lab;
            series_in.push_back(si);
            init_vis.push_back(true);
            glob_vis.push_back(true);   // membership in this window IS the visibility gate
        }

        data_gen = data_gen * 31 + c.data_generation;
        if (c.fit_request) { any_fit = true; c.fit_request = false; }
    }

    view.popup_extras = [&s, &cb]() {
        draw_export_submenu("ls", (int)s.curves.size(),
            [&s](int i) { return s.curves[i].label; },
            [&s](int i) { return s.curves[i].last_run_ok; },
            [&s](int i) { return s.in_flight && !s.is_2d_run && i == s.running_curve_index; },
            [&s](int i, const std::string& p) { data_export::export_ls1d(s.curves[i].result, p); },
            cb);
    };

    // Snap X к узлам первой LS-кривой этого окна (см. apply_snap_x_from_first_member).
    apply_snap_x_from_first_member(view, win.members, s.curves);

    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    view.render(renderer, origin, avail, /*owner_id*/ 0x15A1E0, data_gen,
                series_in, init_vis, glob_vis, any_fit);
}

// ============================================================
// Parametric plot windows — shared setup-row helper.
//
// The row (Label | Type | Members... | X) is rendered both in the Plot
// windows manager on the settings panel (draw_parametric_controls) and
// at the top of each per-window Begin() (draw_parametric_plot_windows),
// so users can retype the chart without going back to the panel. The
// helpers live at file scope so both callsites share one implementation.
// ============================================================
struct ParamPlotMatchItem { int index; std::string label; };

[[nodiscard]] static std::vector<ParamPlotMatchItem>
parametric_matching_items(AppModel& model, ParametricPlotWindow::Kind kind, bool mode_2d) {
    std::vector<ParamPlotMatchItem> out;
    if (kind == ParametricPlotWindow::Kind::Bifurcation) {
        auto& ds = model.bifurcation_session.diagrams;
        for (int i = 0; i < (int)ds.size(); ++i)
            if (ds[i].mode_2d == mode_2d) out.push_back({ i, ds[i].label });
    } else if (kind == ParametricPlotWindow::Kind::LLE) {
        auto& cs = model.lle_session.curves;
        for (int i = 0; i < (int)cs.size(); ++i)
            if (cs[i].mode_2d == mode_2d) out.push_back({ i, cs[i].label });
    } else {
        auto& cs = model.ls_session.curves;
        for (int i = 0; i < (int)cs.size(); ++i)
            if (cs[i].mode_2d == mode_2d) out.push_back({ i, cs[i].label });
    }
    return out;
}

// Colored 1D is not a distinct combo entry — it's a toggle inside the
// plot body (see draw_bifurcation_plot). The Type combo only splits 1D/2D.
static const char* kParametricTypeNames[] = {
    "Bifurcation 1D", "Bifurcation 2D", "LLE 1D", "LLE 2D", "LS 1D", "LS 2D"
};

static int parametric_type_index_of(ParametricPlotWindow::Kind kind, bool mode_2d) {
    int base = kind == ParametricPlotWindow::Kind::Bifurcation ? 0
             : kind == ParametricPlotWindow::Kind::LLE ? 2 : 4;
    return base + (mode_2d ? 1 : 0);
}

static void parametric_type_from_index(int t, ParametricPlotWindow::Kind& kind, bool& mode_2d) {
    kind    = (t < 2) ? ParametricPlotWindow::Kind::Bifurcation
            : (t < 4) ? ParametricPlotWindow::Kind::LLE : ParametricPlotWindow::Kind::LS;
    mode_2d = (t % 2) == 1;
}

// Draws the Label | Type | Members... | X row for a Parametric plot window.
// Returns true iff the user clicked the row's X — caller unmounts the window.
// Assumes caller has already pushed a PushID(win.id) on the ID stack.
static bool draw_parametric_window_setup_row(AppModel& model, ParametricPlotWindow& win) {
    bool close_clicked = false;

    ImGui::SetNextItemWidth(220);
    if (InputTextStr("##wlabel", win.label)) {
        win.label_is_manual = !win.label.empty();   // empty → back to auto
        model.parametric_plot_windows_dirty = true;
    }
    ImGui::SameLine();

    // Type combo: changing kind/dimension invalidates old member indices
    // (they're only meaningful within the previous session+dimension),
    // so switching type clears members.
    int t = parametric_type_index_of(win.kind, win.mode_2d);
    ImGui::SetNextItemWidth(130);
    if (ImGui::Combo("##wtype", &t, kParametricTypeNames, IM_ARRAYSIZE(kParametricTypeNames))) {
        ParametricPlotWindow::Kind new_kind; bool new_2d;
        parametric_type_from_index(t, new_kind, new_2d);
        if (new_kind != win.kind || new_2d != win.mode_2d) {
            win.kind = new_kind;
            win.mode_2d = new_2d;
            win.colored_1d = false;   // combo no longer selects Colored 1D — reset; toggled above the plot
            win.members.clear();
            model.parametric_plot_windows_dirty = true;
        }
    }
    ImGui::SameLine();

    if (ImGui::Button("Members...")) ImGui::OpenPopup("edit_plot_window_members");
    if (ImGui::BeginPopup("edit_plot_window_members")) {
        auto items = parametric_matching_items(model, win.kind, win.mode_2d);
        if (items.empty()) {
            ImGui::TextDisabled("(none available)");
        } else if (win.mode_2d || win.colored_1d) {
            // 2D / Colored 1D show heatmaps — single-select (radio).
            for (const auto& item : items) {
                bool sel = !win.members.empty() && win.members[0] == item.index;
                std::string lbl = item.label + "##mem" + std::to_string(item.index);
                if (ImGui::RadioButton(lbl.c_str(), sel)) {
                    win.members.assign(1, item.index);
                    model.parametric_plot_windows_dirty = true;
                }
            }
        } else {
            for (const auto& item : items) {
                bool has = std::find(win.members.begin(), win.members.end(), item.index) != win.members.end();
                std::string lbl = item.label + "##mem" + std::to_string(item.index);
                if (ImGui::Checkbox(lbl.c_str(), &has)) {
                    if (has) win.members.push_back(item.index);
                    else win.members.erase(std::remove(win.members.begin(), win.members.end(), item.index), win.members.end());
                    model.parametric_plot_windows_dirty = true;
                }
            }
        }
        ImGui::EndPopup();
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("X")) close_clicked = true;

    return close_clicked;
}

// ============================================================
// Parametric mode: dynamic plot windows (mirrors draw_projection_windows
// in Phase). One ImGui window per model.parametric_plot_windows entry,
// each with its own PlotRenderer/Plot2DView/HeatmapView-map keyed by
// ParametricPlotWindow::id (not vector position — position shifts when an
// earlier window is removed, id doesn't). Closing a window (X) removes it
// from the list; "Reset windows layout" (in draw_parametric_controls) bumps
// model.parametric_layout_generation, baked into every title, so ImGui
// treats them as brand-new windows and re-cascades default positions.
// ============================================================
static void draw_parametric_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) {
    static std::map<int, std::unique_ptr<PlotRenderer>> renderers;
    static std::map<int, std::unique_ptr<Plot2DView>>    views;
    static std::map<int, std::map<int, std::unique_ptr<HeatmapView>>> heatmaps;
    // Tracks the kind the Plot2DView was last configured for; when a
    // window's kind changes (Type combo), re-run configure_plot_view so
    // e.g. LLE doesn't inherit Bifurcation's points_mode=true.
    static std::map<int, ParametricPlotWindow::Kind> view_configured_for;

    int to_remove = -1;
    for (int i = 0; i < (int)model.parametric_plot_windows.size(); ++i) {
        ParametricPlotWindow& win = model.parametric_plot_windows[i];

        auto& renderer = renderers[win.id];
        if (!renderer) renderer = std::make_unique<PlotRenderer>();
        auto& view = views[win.id];
        bool fresh_view = !view;
        if (fresh_view) view = std::make_unique<Plot2DView>();
        auto cf_it = view_configured_for.find(win.id);
        if (fresh_view || cf_it == view_configured_for.end() || cf_it->second != win.kind) {
            configure_plot_view(*view, win.kind);
            view_configured_for[win.id] = win.kind;
        }
        auto& hm_map = heatmaps[win.id];

        // Use "###" so ImGui hashes the ID from the suffix only (win.id +
        // layout generation), independent of the visible label. This keeps
        // docking/position stable when the user renames the window; a Reset
        // windows layout bumps parametric_layout_generation, which changes
        // the ID and re-cascades default positions.
        std::string title = win.label + "###pwin" + std::to_string(win.id)
                           + "_g" + std::to_string(model.parametric_layout_generation);
        bool open = true;
        float ox = 60.0f + (float)(i % 5) * 35.0f, oy = 80.0f + (float)(i % 5) * 35.0f;
        ImGui::SetNextWindowPos(ImVec2(ox, oy), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(700, 550), ImGuiCond_FirstUseEver);
        bool row_close = false;
        if (ImGui::Begin(title.c_str(), &open)) {
            ImGui::PushID(win.id);
            // Duplicate of the settings-panel row (draw_parametric_controls) so
            // the chart type / members / label can be changed without leaving
            // the window. Same helper, same PushID(win.id) scope.
            row_close = draw_parametric_window_setup_row(model, win);
            ImGui::Separator();
            switch (win.kind) {
            case ParametricPlotWindow::Kind::Bifurcation:
                draw_bifurcation_plot(model, lib, cb, win, *renderer, *view, hm_map);
                break;
            case ParametricPlotWindow::Kind::LLE:
                draw_lle_plot(model, lib, cb, win, *renderer, *view, hm_map);
                break;
            case ParametricPlotWindow::Kind::LS:
                draw_ls_plot(model, lib, cb, win, *renderer, *view, hm_map);
                break;
            }
            ImGui::PopID();
        }
        ImGui::End();
        if (!open || row_close) to_remove = i;
    }
    if (to_remove >= 0) {
        int id = model.parametric_plot_windows[to_remove].id;
        renderers.erase(id);
        views.erase(id);
        heatmaps.erase(id);
        view_configured_for.erase(id);
        model.remove_parametric_plot_window(to_remove);
    }
}

// ============================================================
// 1D DFT: controls (system picker + Run/Run all + config tab bar, mirrors
// draw_basins_controls) + dynamic Plot windows (mirrors draw_parametric_
// controls' manager + draw_bifurcation_plot's colored_1d heatmap toolbar).
// ============================================================

static void draw_dft1d_diagram_controls(Dft1DAnalysisSession& s, int idx) {
    Dft1DConfig& c = s.configs[idx];

    ImGui::SetNextItemWidth(kComboW);
    if (InputTextStr("Label", c.label))
        c.label_is_manual = !c.label.empty();   // empty → back to auto
    ImGui::Separator();

    // ----- Scheme (built-in + custom) -----
    draw_scheme_combo("Scheme", c.scheme, s.custom_schemes);
    ImGui::Separator();

    // ----- Sweep target (parameter ИЛИ initial condition), см. draw_diagram_controls -----
    // dt (h) — как у Bif/LLE/LS 1D: число сэмплов блока и окно пересчитываются
    // под шаг каждой точки. Второй оси у DFT нет, поэтому other_over_h = nullptr.
    draw_sweep_target_combo("Sweep", s.params, s.vars,
                            c.param_index, c.sweep_over_var, c.var_sweep_index,
                            c.sweep_over_h, nullptr,
                            /*note_when_empty*/ true);
    InputNumStr(c.sweep_over_h ? "h lo" : "Param lo", c.param_lo_text, kFieldW);
    InputNumStr(c.sweep_over_h ? "h hi" : "Param hi", c.param_hi_text, kFieldW);
    ImGui::Checkbox("Log scale##dft_log", &c.log_scale);
    if (c.log_scale) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }

    // Тот же блок Continuation + GPU/CPU, что у Bif / LLE / LS 1D.
    // 2D-режима у DFT нет, поэтому блокирует только IC-свип.
    draw_continuation_device_block(c, "dft", c.sweep_over_var, /*device_locked*/ false);
    ImGui::Separator();

    // ----- Variable + Resolution X -----
    draw_writable_var_combo(s.vars, c.writable_var, "Writable var##dft_wv");
    InputNumStr("Resolution X", c.n_pts_text, kFieldW);
    ImGui::Separator();

    // ----- Resolution Y / Frequency range (обязательные поля для rangesFreq;
    // без auto-режима — частотный диапазон всегда физически осмыслен только
    // когда задан явно, в отличие от colored_1d's Y auto-range). -----
    if (ImGui::CollapsingHeader("Resolution Y##dft_freq", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("Resolution Y", c.n_freq_text, kFieldW);
        InputNumStr("Freq lo", c.freq_lo_text, kFieldW);
        InputNumStr("Freq hi", c.freq_hi_text, kFieldW);
        ImGui::Checkbox("Log scale (Y)##dft_freq_log", &c.freq_log_scale);
        if (c.freq_log_scale) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
        static const char* windows[] = { "None", "Hanning", "Hamming" };
        ImGui::SetNextItemWidth(kComboW);
        ImGui::Combo("Window", &c.window_type, windows, IM_ARRAYSIZE(windows));
    }

    // ----- Display mode + normalize -----
    if (ImGui::CollapsingHeader("Display##dft_disp", ImGuiTreeNodeFlags_DefaultOpen)) {
        static const char* modes[] = { "Power spectrum", "Amplitude", "Phase" };
        ImGui::SetNextItemWidth(kComboW);
        ImGui::Combo("Mode", &c.display_mode, modes, IM_ARRAYSIZE(modes));
        ImGui::Checkbox("Normalize?", &c.normalize);
    }

    // ----- Integration (collapsible) -----
    {
        IntegrationFields f;
        f.h           = &c.h_text;
        f.symmetry_s  = &c.symmetry_s;
        f.t_max       = &c.t_max_text;
        f.transient   = &c.transient_text;
        f.pre_scaller = &c.pre_scaller_text;
        f.max_value   = &c.max_value_text;
        draw_integration_block("Integration##dft_int", c.scheme, s.custom_schemes, f);
    }

    draw_named_num_fields("Initial conditions##dft_ic", s.vars,   c.initial_conditions);
    draw_named_num_fields("Parameters##dft_par",        s.params, c.param_values);
    draw_csv_output_block("CSV output##dft_csv", "Save to file", c.csv_save_enabled,
                          "##dft_csv_path", c.csv_output_path,
                          "Writes <path>_config.csv, _AkCOS.csv, _BkSIN.csv.");

    // Run-кнопка живёт в draw_dft1d_controls, как у Bifurcation/Basins.
    if (c.last_run_ok) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
            "OK: n_pts=%d, n_freq=%d", c.result.n_pts, c.result.n_freq);
        draw_regime_summary(c.result.flags);
    } else if (!c.last_error.empty()) {
        draw_error_box("##dft_err", c.last_error);
    }
}

static void draw_dft1d_controls(AppModel& model, SystemLibrary& lib) {
    Dft1DAnalysisSession& s = model.dft1d_session;

    ImGui::Text("1D DFT");
    ImGui::TextDisabled("Parametric discrete Fourier transform via NVRTC + DFT_custom.");


    // ----- Run / Run all... (общая реализация, см. draw_run_and_run_all) -----
    {
        const bool no_cfg = s.configs.empty();
        RunAllGroup g;
        g.pick_id = "pdft1d_";
        g.n       = (int)s.configs.size();
        g.label   = [&s](int i) { return s.configs[i].label; };
        g.enqueue = [&model](int i) { model.dft1d_queue.push_back({i}); };
        draw_run_and_run_all("##run_all_dft1d", s.in_flight, no_cfg,
                             s.in_flight || no_cfg, 160.0f,
                             [&model, &s]() {
                                 if (!model.parametric_engine)
                                     model.parametric_engine = std::make_unique<ParametricEngine>();
                                 s.run_async(*model.parametric_engine, s.active_config_index);
                             },
                             {}, { g },
                             [&model]() { model.start_next_in_dft1d_queue(); },
                             model.dft1d_queue.size());
    }
    ImGui::Separator();

    // Tab bar: одна вкладка на config + "+" для add. Тело вкладки пустое —
    // настройки активного конфига рисуются ниже, после таб-бара.
    const TabBarResult tabs = draw_config_tab_bar(
        "##dft1d_tabs", "dft1d_tab_", (int)s.configs.size(),
        s.in_flight, s.running_config_index,
        [&s](int i) { return s.configs[i].label; },
        {},
        [&s]() { s.add_config(); });
    if (tabs.active    >= 0) s.active_config_index = tabs.active;
    if (tabs.to_remove >= 0) model.remove_dft1d_config(tabs.to_remove);

    if (s.configs.empty()) {
        ImGui::TextDisabled("No DFT configs. Press '+' to add one.");
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
        s.active_config_index = 0;

    draw_dft1d_diagram_controls(s, s.active_config_index);

    // ----- Plot windows: dynamic list, mirrors Parametric's manager section -----
    // No Type combo needed (DFT1D has only one display kind); "Members..."
    // is single-select (radio) — each window shows exactly one config's
    // heatmap, same as Bifurcation's colored_1d/mode_2d windows.
    ImGui::Separator();
    ImGui::SeparatorText("Plot windows");
    int win_to_remove = -1;
    for (int i = 0; i < (int)model.dft1d_plot_windows.size(); ++i) {
        Dft1DPlotWindow& win = model.dft1d_plot_windows[i];
        ImGui::PushID(win.id);
        ImGui::SetNextItemWidth(220);
        if (InputTextStr("##wlabel", win.label)) {
            win.label_is_manual = !win.label.empty();   // empty → back to auto
            model.dft1d_plot_windows_dirty = true;
        }
        ImGui::SameLine();

        if (ImGui::Button("Members...")) ImGui::OpenPopup("edit_dft1d_window_members");
        if (ImGui::BeginPopup("edit_dft1d_window_members")) {
            if (s.configs.empty()) {
                ImGui::TextDisabled("(none available)");
            } else {
                // Single-select (radio) — a DFT1D window shows exactly one
                // config's heatmap, same as Bifurcation's mode_2d/colored_1d.
                for (int ci = 0; ci < (int)s.configs.size(); ++ci) {
                    bool sel = !win.members.empty() && win.members[0] == ci;
                    std::string lbl = s.configs[ci].label + "##dftmem" + std::to_string(ci);
                    if (ImGui::RadioButton(lbl.c_str(), sel)) {
                        win.members.assign(1, ci);
                        model.dft1d_plot_windows_dirty = true;
                    }
                }
            }
            ImGui::EndPopup();
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("X")) win_to_remove = i;
        ImGui::PopID();
    }
    if (win_to_remove >= 0) model.remove_dft1d_plot_window(win_to_remove);

    if (ImGui::Button("Add window")) {
        model.add_dft1d_plot_window({});
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset windows layout")) { model.dft1d_layout_generation++; }
}

// Per-window heatmap rendering — mirrors the colored_1d block of
// draw_bifurcation_plot (toolbar + lazy display-cache rebuild + render()) and
// draw_basins_plot's colormap/autoscale/swap-axes toolbar.
static void draw_dft1d_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                           Dft1DPlotWindow& win,
                           PlotRenderer& renderer,
                           std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) {
    Dft1DAnalysisSession& s = model.dft1d_session;
    auto get_hm = [&](int idx) -> HeatmapView& {
        const int cfg_cm = (idx >= 0 && idx < (int)s.configs.size())
                           ? s.configs[idx].colormap_idx : -1;
        return get_or_create_heatmap(heatmap_map, idx, cfg_cm, model.heatmap_colormap);
    };

    if (win.members.empty()) {
        ImGui::TextDisabled("No DFT configs assigned to this window.");
        return;
    }

    for (size_t mi = 0; mi < win.members.size(); ++mi) {
        int idx = win.members[mi];
        if (idx < 0 || idx >= (int)s.configs.size()) continue;
        Dft1DConfig& c = s.configs[idx];
        if (mi > 0) ImGui::Separator();
        ImGui::PushID(idx);

        const unsigned oid = 0xD1FD0000u + (unsigned)idx;
        HeatmapView& hc = get_hm(idx);

        {
            HeatmapToolbarOpts topts;
            topts.persist_colormap = [&](int cm) {
                c.colormap_idx = cm;   // persist per-config
                if (!model.loaded_name.empty())
                    lib.save_session(model.loaded_name, "_last_dft1d",
                                     session_to_json_dft1d(model.dft1d_session));
            };
            draw_heatmap_toolbar(hc, topts);
        }

        if (!c.last_run_ok || c.result.ak_cos.empty()) {
            ImGui::TextDisabled("No data yet. Press Run.");
            ImGui::PopID();
            continue;
        }

        int npts  = c.result.n_pts;
        int nfreq = c.result.n_freq;

        // Лениво (пере)строим display_cache, когда расходится с текущими
        // настройками — не каждый кадр. Та же staleness-схема, что и у
        // colored_1d_cache в draw_bifurcation_plot.
        bool stale = c.display_built_from    != c.data_generation
                  || c.display_cache_mode      != c.display_mode
                  || c.display_cache_normalize != c.normalize;
        if (stale) {
            size_t plane_size = (size_t)nfreq * (size_t)npts;
            // freq-major/param-minor (idx = f*n_pts+pt) — конвенция
            // HeatmapView::render(nx=n_pts, ny=n_freq, ...). 999.0 — тот же
            // sentinel, что и colored_1d/Basins используют для "нет данных".
            c.display_cache.assign(plane_size, kSentinelNoData);
            double vmin =  std::numeric_limits<double>::infinity();
            double vmax = -std::numeric_limits<double>::infinity();
            std::vector<double> col((size_t)nfreq);
            for (int pt = 0; pt < npts; ++pt) {
                bool diverged = pt >= (int)c.result.flags.size() ||
                                !regime_is_oscillation(c.result.flags[pt]);
                if (diverged) continue;   // остаётся 999 sentinel — DFT_custom не считал эту точку

                double colmax = 0.0;
                for (int f = 0; f < nfreq; ++f) {
                    double ak = c.result.ak_cos[(size_t)pt * (size_t)nfreq + (size_t)f];
                    double bk = c.result.bk_sin[(size_t)pt * (size_t)nfreq + (size_t)f];
                    double v;
                    if      (c.display_mode == 0) v = ak * ak + bk * bk;            // power spectrum
                    else if (c.display_mode == 1) v = std::sqrt(ak * ak + bk * bk); // amplitude
                    else                          v = std::atan2(bk, ak);          // phase
                    col[(size_t)f] = v;
                    double av = std::fabs(v);
                    if (av > colmax) colmax = av;
                }
                for (int f = 0; f < nfreq; ++f) {
                    double v = col[(size_t)f];
                    if (c.normalize && colmax > 0.0) v /= colmax;
                    // dB-масштаб как в референсном MATLAB-скрипте (10*log10) —
                    // только для power/amplitude; phase остаётся в радианах.
                    if (c.display_mode != 2) v = 10.0 * std::log10(std::max(v, 1e-12));
                    c.display_cache[(size_t)f * (size_t)npts + (size_t)pt] = v;
                    if (v < vmin) vmin = v;
                    if (v > vmax) vmax = v;
                }
            }
            c.display_cache_vmin = std::isfinite(vmin) ? vmin : 0.0;
            c.display_cache_vmax = std::isfinite(vmax) ? vmax : 1.0;
            c.display_built_from      = c.data_generation;
            c.display_cache_mode      = c.display_mode;
            c.display_cache_normalize = c.normalize;
            ++c.display_cache_gen;
        }

        hc.x_axis.name = auto_axis_name(s.params, s.vars, c.param_index,
                                        c.sweep_over_var, c.var_sweep_index, c.sweep_over_h);
        hc.x_axis.log_scale = c.log_scale;
        hc.y_axis.name = "Frequency";
        hc.y_axis.log_scale = c.freq_log_scale;

        bool fit = c.fit_request;
        if (fit) c.fit_request = false;

        const bool busy = s.in_flight && idx == s.running_config_index;
        hc.popup_extras = [&c, &cb, busy]() {
            draw_export_menu_item(busy, cb, [&c](const std::string& p) {
                data_export::export_dft1d(c.result, p);
            });
        };

        double lo = c.result.param_lo, hi = c.result.param_hi;
        bool rev = c.result.continuation_reverse;
        double x0 = rev ? hi : lo;
        double x1 = rev ? lo : hi;

        ImVec2 avail = ImGui::GetContentRegionAvail();
        ImVec2 origin = ImGui::GetCursorScreenPos();
        hc.render(renderer, origin, avail,
                  /*owner_id*/ oid, c.display_cache_gen,
                  npts, nfreq,
                  c.display_cache.data(),
                  x0, x1,
                  c.result.freq_lo, c.result.freq_hi,
                  c.display_cache_vmin, c.display_cache_vmax,
                  fit);
        ImGui::PopID();
    }
}

static void draw_dft1d_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) {
    static std::map<int, std::unique_ptr<PlotRenderer>> renderers;
    static std::map<int, std::map<int, std::unique_ptr<HeatmapView>>> heatmaps;

    int to_remove = -1;
    for (int i = 0; i < (int)model.dft1d_plot_windows.size(); ++i) {
        Dft1DPlotWindow& win = model.dft1d_plot_windows[i];

        auto& renderer = renderers[win.id];
        if (!renderer) renderer = std::make_unique<PlotRenderer>();
        auto& hm_map = heatmaps[win.id];

        std::string title = win.label + "###dftwin" + std::to_string(win.id)
                           + "_g" + std::to_string(model.dft1d_layout_generation);
        bool open = true;
        float ox = 60.0f + (float)(i % 5) * 35.0f, oy = 80.0f + (float)(i % 5) * 35.0f;
        ImGui::SetNextWindowPos(ImVec2(ox, oy), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(700, 550), ImGuiCond_FirstUseEver);
        if (ImGui::Begin(title.c_str(), &open)) {
            ImGui::PushID(win.id);
            draw_dft1d_plot(model, lib, cb, win, *renderer, hm_map);
            ImGui::PopID();
        }
        ImGui::End();
        if (!open) to_remove = i;
    }
    if (to_remove >= 0) {
        int id = model.dft1d_plot_windows[to_remove].id;
        renderers.erase(id);
        heatmaps.erase(id);
        model.remove_dft1d_plot_window(to_remove);
    }
}

// ============================================================
// Basins of attraction: controls + 5-plot window (inner tab-bar).
// ============================================================

// ============================================================
// Basins: общие хелперы карты + панель фазовых портретов по бассейнам.
// ============================================================

// Диапазон colorbar'а таба «Basins» — единственный источник истины и для самой
// хитмапы, и для цвета траекторий в фазовых портретах (иначе цвет бассейна на
// карте и цвет его траектории разъезжаются).
//   min_cluster_id..-1 — FP-кластеры (если есть отрицательные)
//   0                  — расходящиеся ячейки (helpful_array[i] == 0)
//   1..n_clusters      — осциллирующие кластеры
// Когда FP-кластеров нет и ничего не разошлось, «кластер 0» не существует —
// сдвигаем vmin к 1, чтобы в colorbar'е не было фантомной полосы.
// Названия 12 признаков DBSCAN (BF_* в configCUDA.h / enum BasinFeature в
// analysis_session.h). Один список на файл: раньше он был выписан дважды — в
// панели настроек и в подписях осей scatter'а — с комментарием «должны быть
// синхронизированы», то есть с ручной синхронизацией вместо общей константы.
static const char* const kFeatureNames[] = {
    "Avg peaks",             "Avg intervals",
    "RMS peaks",             "RMS intervals",
    "StDev peaks",           "StDev intervals",
    "sign\xc2\xb7log10|avg peaks|", "sign\xc2\xb7log10|avg intervals|",
    "log10 RMS peaks",       "log10 RMS intervals",
    "log10 StDev peaks",     "log10 StDev intervals",
};

static void basins_colorbar_range(const BasinsConfig& c, double& vmin, double& vmax) {
    const int n_clusters     = c.renumber_spiral ? c.n_clusters_spiral      : c.result.n_clusters;
    const int min_cluster_id = c.renumber_spiral ? c.min_cluster_idx_spiral : c.result.min_cluster_idx;
    bool has_diverged = false;
    for (int f : c.result.helpful_array)
        if (regime_is_unbound(f)) { has_diverged = true; break; }
    if (min_cluster_id < 0)  vmin = (double)min_cluster_id;
    else if (has_diverged)   vmin = 0.0;
    else                     vmin = 1.0;
    vmax = (double)n_clusters;
    if (vmax < vmin) vmax = vmin;
}

// HeatmapView таба «Basins» каждого config'а. Раньше жил function-local static
// внутри draw_basins_plot; поднят на уровень файла, потому что цвет траекторий
// в окнах фазовых портретов обязан читать ЖИВЫЕ discrete/reverse/colormap
// именно этого вью — иначе цвета совпадали бы только при дефолтных настройках.
static std::map<unsigned, std::unique_ptr<HeatmapView>> g_hm_basins;

// owner_id блока плотов одного basins-config'а (5 внутренних табов на config).
[[nodiscard]] static unsigned basins_base_oid(int config_index) {
    return 0x1BA50000u + (unsigned)config_index * 5u;
}

// Цвет бассейна `display_id` — ровно тот, которым его красит хитмапа.
// Повторяет квантование из FS_HEATMAP (plot_renderer.cpp): t по [vmin,vmax],
// дискретизация по полосам с edge-aligned сэмплированием, затем reverse.
static ImVec4 basins_id_color(const BasinsConfig& c, const AppModel& model,
                              int config_index, int display_id) {
    double vmin = 0.0, vmax = 0.0;
    basins_colorbar_range(c, vmin, vmax);

    int  cm      = (c.colormap_idx[0] >= 0) ? c.colormap_idx[0] : model.basins_colormap;
    bool reverse = false;
    // discrete_default для этого вью = true, поэтому дефолт (пока панель ни
    // разу не рисовалась) — дискретный, по одной полосе на целый id.
    int  n_disc  = std::max(1, (int)std::lround(vmax - vmin) + 1);

    auto it = g_hm_basins.find(basins_base_oid(config_index) + 0u);
    if (it != g_hm_basins.end() && it->second) {
        const HeatmapView& hv = *it->second;
        cm      = (int)hv.colormap;
        reverse = hv.reverse_colormap;
        n_disc  = hv.discrete ? (hv.discrete_levels > 0 ? hv.discrete_levels : n_disc) : 0;
    }
    cm = colormap_id_or(cm, kColormapTurbo);

    // Нормировка — по РАЗДВИНУТОМУ диапазону, ровно как в HeatmapView::render
    // (там vmax <= vmin даёт vmax = vmin + 1). Число полос при этом считается
    // по исходному — тоже как там. Раньше здесь вырожденный диапазон давал
    // t = 0.5, а карта для того же id — t = 0, и единственный бассейн
    // получал на портрете другой цвет, чем на хитмапе.
    double vmax_n = vmax;
    if (vmax_n <= vmin) vmax_n = vmin + 1.0;
    const double range = vmax_n - vmin;
    float t = (float)(((double)display_id - vmin) / range);
    if (t < 0.0f) t = 0.0f; else if (t > 1.0f) t = 1.0f;
    if (n_disc > 0) {
        const float nb = (float)n_disc;
        float k = std::floor(t * nb);
        if (k >= nb) k = nb - 1.0f;
        t = (nb > 1.0f) ? (k / (nb - 1.0f)) : 0.5f;
    }
    if (reverse) t = 1.0f - t;
    return ImGui::ColorConvertU32ToFloat4(cmap_sample(t, (HeatmapColormap)cm));
}

// Панель «Phase portraits» внутри Basins Controls. Интегратор и параметры —
// из самого basins-config'а; своё тут только время моделирования, прореживание,
// ограничитель числа аттракторов и выбор CPU/GPU.
static void draw_basins_phase_controls(AppModel& model, int cfg_idx) {
    BasinsAnalysisSession& s = model.basins_session;
    if (cfg_idx < 0 || cfg_idx >= (int)s.configs.size()) return;
    BasinsConfig&    c  = s.configs[(size_t)cfg_idx];
    BasinsPhaseSlot& sl = s.phase_slot(cfg_idx);

    ImGui::TextDisabled("One random grid cell per basin; scheme/h/params come from this config.");

    bool changed = false;

    // ---- Run / Autorun / устройство ----
    const bool basins_busy = s.in_flight && s.running_config_index == cfg_idx;
    const bool phase_busy  = sl.phase.in_flight;
    const bool no_data     = !c.last_run_ok || c.result.basin_idx.empty();
    {
        const bool block = basins_busy || phase_busy || no_data;
        if (block) ImGui::BeginDisabled();
        if (ImGui::Button(phase_busy ? "Recomputing..." : "Run portraits", ImVec2(160, 0)))
            s.request_phase_run(cfg_idx);
        if (block) ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Checkbox("Autorun", &c.pp_autorun)) changed = true;
        ImGui::SameLine();
        if (ImGui::Checkbox("GPU##pp", &c.pp_use_gpu)) changed = true;
    }
    // Custom КРС на CPU требует внешнего компилятора — та же проверка, что и в
    // draw_phase_controls; без него молча возвращаем GPU.
    if (is_custom_scheme(c.scheme, s.custom_schemes) && !c.pp_use_gpu) {
        std::string why;
        if (!krs_cpu_backend_available(&why)) {
            ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1), "(custom scheme requires GPU: %s)", why.c_str());
            c.pp_use_gpu = true;
        }
    }

    // ---- Собственные параметры интегрирования ----
    changed |= InputNumStr("computing time##pp", c.pp_t_max_text, kFieldW);
    changed |= InputNumStr("transient time##pp", c.pp_transient_text, kFieldW);
    changed |= InputNumStr("decimator##pp",      c.pp_prescaller_text, kFieldW);
    ImGui::TextDisabled("Decimator only thins the drawn points (applied after integration).");
    changed |= InputNumStr("max attractors##pp", c.pp_max_attractors_text, kFieldW);
    ImGui::SetNextItemWidth(200);
    if (ImGui::SliderFloat("FP marker size##pp", &c.pp_marker_size, 2.0f, 12.0f, "%.0f px"))
        changed = true;

    // ---- Сводка: сколько бассейнов нашлось / сколько рисуем / сколько памяти ----
    if (no_data) {
        ImGui::TextDisabled("No basins data yet — run the map first.");
    } else {
        int n_basins = 0;
        const int  n   = c.result.n_pts;
        const int* ids = basins_display_ids(c);
        if (ids && n > 0) {
            std::unordered_set<int> seen;
            const size_t tot = (size_t)n * (size_t)n;
            for (size_t k = 0; k < tot; ++k) if (ids[k] != 0) seen.insert(ids[k]);
            n_basins = (int)seen.size();
        }
        const int drawn = (int)sl.cells.size();
        if (n_basins == 0) {
            ImGui::TextDisabled("No basins found (every cell diverged) — nothing to draw.");
        } else {
            ImGui::Text("%d basin(s) found, %d drawn", n_basins, drawn);
            int limit = parse_int_or(c.pp_max_attractors_text, 1);
            if (limit < 1) limit = 1;
            if (n_basins > limit) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f), "(limited to %d)", limit);
            }
            // Оценка объёма: движок держит траектории целиком (децимация
            // применяется уже после расчёта), поэтому память считаем по
            // полному числу шагов.
            const double h  = parse_ratio_or(c.h_text, 0.0);
            const double tt = parse_ratio_or(c.pp_t_max_text, 0.0);
            const int    kept = (drawn > 0) ? drawn : std::min(n_basins, limit);
            if (h > 0 && tt > 0 && kept > 0) {
                const double steps = tt / h;
                const double mb = steps * (double)kept * (double)s.vars.size() * 8.0 / (1024.0 * 1024.0);
                ImGui::TextDisabled("%d x %.0f points ~ %.1f MB on GPU (x4-5 on host)", kept, steps, mb);
            }
        }
    }

    // ---- Статус актуальности ----
    const std::string sig = build_basins_phase_signature(c);
    if (!no_data && sig != sl.run_signature && !phase_busy) {
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f),
                           c.pp_autorun ? "Rebuilding..." : "Stale - press Run portraits");
    }
    // compute_phase_portrait кладёт в error тайминг успешного прогона
    // ("recompute: N ms") — это не ошибка, красить в оранжевый нечего.
    if (!sl.phase.result.error.empty()) {
        const bool is_timing = sl.phase.result.error.rfind("recompute:", 0) == 0;
        if (is_timing) ImGui::TextDisabled("%s", sl.phase.result.error.c_str());
        else ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "%s", sl.phase.result.error.c_str());
    }

    // ---- Окна (проекции) ----
    ImGui::SeparatorText("Windows");
    int pr_to_remove = -1;
    for (int i = 0; i < (int)sl.phase.projections.size(); ++i) {
        Projection& pr = sl.phase.projections[i];
        // 3D в этом режиме не поддерживается (нет точечного рендера в
        // Plot3DView) — старые сессии с Phase3D мягко приводим к 2D.
        if (pr.type == ProjType::Phase3D) pr.type = ProjType::Phase2D;
        ImGui::PushID(2000 + i);
        ImGui::SetNextItemWidth(90);
        InputTextStr("##pplabel", pr.label); ImGui::SameLine();
        ImGui::SetNextItemWidth(110);
        const char* tnames[] = { "Phase 2D", "Time domain" };
        int t = (pr.type == ProjType::TimeDomain) ? 1 : 0;
        if (ImGui::Combo("##pptype", &t, tnames, 2)) {
            pr.type = (t == 1) ? ProjType::TimeDomain : ProjType::Phase2D;
            sl.phase.fit_request = true;
        }
        ImGui::SameLine();
        if (pr.type == ProjType::Phase2D) {
            const auto& vars = sl.phase.vars;
            ImGui::Text("X:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(55);
            if (ImGui::BeginCombo("##ppx", vars.empty() ? "-" : vars[pr.axis_x < (int)vars.size() ? pr.axis_x : 0].c_str())) {
                for (int k = 0; k < (int)vars.size(); ++k)
                    if (ImGui::Selectable(vars[k].c_str(), pr.axis_x == k)) { pr.axis_x = k; sl.phase.fit_request = true; }
                ImGui::EndCombo();
            }
            ImGui::SameLine(); ImGui::Text("Y:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(55);
            if (ImGui::BeginCombo("##ppy", vars.empty() ? "-" : vars[pr.axis_y < (int)vars.size() ? pr.axis_y : 0].c_str())) {
                for (int k = 0; k < (int)vars.size(); ++k)
                    if (ImGui::Selectable(vars[k].c_str(), pr.axis_y == k)) { pr.axis_y = k; sl.phase.fit_request = true; }
                ImGui::EndCombo();
            }
        } else {
            if ((int)pr.show_var.size() != (int)sl.phase.vars.size())
                pr.show_var.assign(sl.phase.vars.size(), true);
            ImGui::Text("vars:"); ImGui::SameLine();
            for (int k = 0; k < (int)sl.phase.vars.size(); ++k) {
                bool v = pr.show_var[k];
                if (ImGui::Checkbox(sl.phase.vars[k].c_str(), &v)) pr.show_var[k] = v;
                ImGui::SameLine();
            }
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("X")) pr_to_remove = i;
        ImGui::PopID();
    }
    if (pr_to_remove >= 0) { sl.phase.remove_projection(pr_to_remove); changed = true; }
    if (ImGui::Button("Add 2D plot")) {
        sl.phase.add_projection();
        changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Add time domain")) {
        sl.phase.add_projection();
        sl.phase.projections.back().type = ProjType::TimeDomain;
        changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset windows layout##pp")) sl.phase.layout_generation++;

    // ---- Легенда/видимость бассейнов ----
    if (!sl.phase.ic_sets.empty()) {
        ImGui::SeparatorText("Basins drawn");
        ImGui::Checkbox("Legend shows initial conditions##pp", &sl.phase.legend_show_ic);
        const int* ids = basins_display_ids(c);
        for (int k = 0; k < (int)sl.phase.ic_sets.size(); ++k) {
            ImGui::PushID(3000 + k);
            ImGui::Checkbox("##ppvis", &sl.phase.ic_sets[k].visible);
            ImGui::SameLine();
            if (ids && k < (int)sl.cells.size() &&
                sl.cells[k] >= 0 && sl.cells[k] < (int)c.result.basin_idx.size()) {
                const ImVec4 col = basins_id_color(c, model, cfg_idx, ids[sl.cells[k]]);
                ImGui::ColorButton("##ppcol", col,
                                   ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoDragDrop,
                                   ImVec2(12, 12));
                ImGui::SameLine();
            }
            ImGui::TextUnformatted(sl.phase.ic_sets[k].label.c_str());
            ImGui::PopID();
        }
    }

    if (changed) s.phase_settings_dirty = true;
}

static void draw_basins_controls(AppModel& model, SystemLibrary& lib) {
    BasinsAnalysisSession& s = model.basins_session;

    ImGui::Text("Basins of attraction");
    ImGui::TextDisabled("DBSCAN clustering in (avgPeak, avgInterval) plane.");

    // ----- Run / Run all... (moved up to sit right under the system picker,
    // above the tab bar — these drive the currently active config so they
    // stay accessible without scrolling past every section). -----
    // Batch "Run all..." across basin configs. Pushes selected indices into
    // model.basins_queue; draw_gui ticks the queue after polls.
    {
        const bool no_cfg = s.configs.empty();
        RunAllGroup g;
        g.pick_id = "pbasins_";
        g.n       = (int)s.configs.size();
        g.label   = [&s](int i) { return s.configs[i].label; };
        g.enqueue = [&model](int i) { model.basins_queue.push_back({i}); };
        draw_run_and_run_all("##run_all_basins", s.in_flight, no_cfg,
                             s.in_flight || no_cfg, 160.0f,
                             [&model, &s]() {
                                 if (!model.parametric_engine)
                                     model.parametric_engine = std::make_unique<ParametricEngine>();
                                 s.run_async(*model.parametric_engine, s.active_config_index);
                             },
                             {}, { g },
                             [&model]() { model.start_next_in_basins_queue(); },
                             model.basins_queue.size());
    }
    ImGui::Separator();

    // Tab bar: одна вкладка на Basins-config + "+" для add. Зеркалит
    // draw_bifurcation_controls. Активная вкладка хранится в
    // s.active_config_index и используется Ctrl+R + плотами.
    const TabBarResult tabs = draw_config_tab_bar(
        "##basins_tabs", "basins_tab_", (int)s.configs.size(),
        s.in_flight, s.running_config_index,
        [&s](int i) { return s.configs[i].label; },
        {},
        [&s]() { s.add_config(); });
    if (tabs.active    >= 0) s.active_config_index = tabs.active;
    if (tabs.to_remove >= 0) model.remove_basins_config(tabs.to_remove);

    if (s.configs.empty()) {
        ImGui::TextDisabled("No basins configs. Press '+' to add one.");
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
        s.active_config_index = 0;
    BasinsConfig& c = s.configs[s.active_config_index];

    // Inline rename для активной вкладки.
    draw_label_rename("Label##basins_label", c.label);
    ImGui::Separator();

    // ----- Scheme -----
    draw_scheme_combo("Scheme", c.scheme, s.custom_schemes);
    ImGui::Separator();

    // ----- Axes (X, Y по двум IC-переменным) -----
    if (!s.vars.empty()) {
        if (c.axis_x_var < 0 || c.axis_x_var >= (int)s.vars.size()) c.axis_x_var = 0;
        if (c.axis_y_var < 0 || c.axis_y_var >= (int)s.vars.size())
            c.axis_y_var = (s.vars.size() > 1) ? 1 : 0;
        const std::vector<const char*> items = c_str_list(s.vars);

        ImGui::SetNextItemWidth(kComboW);
        ImGui::Combo("Axis X (IC)", &c.axis_x_var, items.data(), (int)items.size());
        InputNumStr("X lo", c.axis_x_lo_text, kFieldW);
        InputNumStr("X hi", c.axis_x_hi_text, kFieldW);

        ImGui::SetNextItemWidth(kComboW);
        ImGui::Combo("Axis Y (IC)", &c.axis_y_var, items.data(), (int)items.size());
        InputNumStr("Y lo", c.axis_y_lo_text, kFieldW);
        InputNumStr("Y hi", c.axis_y_hi_text, kFieldW);
    } else {
        ImGui::TextDisabled("No variables (load a system first)");
    }
    InputNumStr("Resolution", c.n_pts_text, kFieldW);

    // ----- Writable var (для peak finder) -----
    draw_writable_var_combo(s.vars, c.writable_var, "Writable var##bas_wv");

    ImGui::Separator();

    // ----- Integration (collapsible, swapped above Features) -----
    {
        IntegrationFields f;
        f.h           = &c.h_text;
        f.symmetry_s  = &c.symmetry_s;
        f.t_max       = &c.t_max_text;
        f.transient   = &c.transient_text;
        f.pre_scaller = &c.pre_scaller_text;
        f.max_value   = &c.max_value_text;
        draw_integration_block("Integration", c.scheme, s.custom_schemes, f);
    }

    // ----- Features (DBSCAN axes + plot data) (collapsible) -----
    // 12 фич (см. BF_* в configCUDA.h / enum BasinFeature в analysis_session.h).
    // Feature 1 пишется в outAvgPeaks-буфер (X-координата DBSCAN), Feature 2 —
    // в AvgTimeOfPeaks-буфер (Y-координата). Множители применяются ПОСЛЕ
    // вычисления фичи и нужны для масштабирования кластеризации.
    if (ImGui::CollapsingHeader("Features (DBSCAN axes + plot data)",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        if (c.feature1 < 0 || c.feature1 >= BF_FEATURE_COUNT) c.feature1 = BF_FEATURE1_DEFAULT;
        if (c.feature2 < 0 || c.feature2 >= BF_FEATURE_COUNT) c.feature2 = BF_FEATURE2_DEFAULT;
        ImGui::SetNextItemWidth(220);
        ImGui::Combo("Feature 1##bas", &c.feature1, kFeatureNames, IM_ARRAYSIZE(kFeatureNames));
        ImGui::SameLine();
        InputNumStr("mult##bas_f1", c.mult_feature1_text, 80);
        ImGui::SetNextItemWidth(220);
        ImGui::Combo("Feature 2##bas", &c.feature2, kFeatureNames, IM_ARRAYSIZE(kFeatureNames));
        ImGui::SameLine();
        InputNumStr("mult##bas_f2", c.mult_feature2_text, 80);
    }

    // DBSCAN eps — кластеризационный радиус в (Feature 1, Feature 2) пространстве.
    // Оставлен снаружи Features-секции: тюнится чаще, чем выбор самих фич.
    // Рядом — кнопка "Clustering": перезапускает только DBSCAN-фазу по уже
    // посчитанным фичам (avg_peaks / avg_intervals в c.result), быстрее чем
    // полный Run в десятки/сотни раз. Дизейблится, если нет валидного
    // предыдущего результата или уже идёт расчёт.
    InputNumStr("DBSCAN eps", c.eps_dbscan_text, kFieldW);
    ImGui::SameLine();
    const bool can_recluster = !s.in_flight && c.last_run_ok &&
                               !c.result.avg_peaks.empty() &&
                               !c.result.avg_intervals.empty() &&
                               !c.result.helpful_array.empty();
    if (!can_recluster) ImGui::BeginDisabled();
    if (ImGui::Button("Clustering")) {
        if (!model.parametric_engine)
            model.parametric_engine = std::make_unique<ParametricEngine>();
        s.run_recluster_async(*model.parametric_engine, s.active_config_index);
    }
    if (!can_recluster) ImGui::EndDisabled();
    ImGui::TextDisabled("Clustering radius in (Feature 1, Feature 2) space.");

    draw_named_num_fields("Initial conditions", s.vars, c.initial_conditions,
                          "Values for non-axis variables.");
    draw_named_num_fields("Parameters", s.params, c.param_values);

    // ----- Phase portraits from basins (collapsible) -----
    // Отдельные окна 2D / Time domain по одной представительной точке из
    // каждого найденного бассейна (см. draw_basins_phase_controls).
    if (ImGui::CollapsingHeader("Phase portraits")) {
        draw_basins_phase_controls(model, s.active_config_index);
    }

    draw_csv_output_block("CSV output", "Save to file", c.csv_save_enabled,
                          "##basins_csv_path", c.csv_output_path,
                          "Writes 4 files: <path>, _1.csv (Feature 1), _2.csv (Feature 2), _3.csv (states).");

    if (c.last_run_ok) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
            "OK: %dx%d, %d clusters (+ %d FP clusters)",
            c.result.n_pts, c.result.n_pts,
            c.result.n_clusters, -c.result.min_cluster_idx);
        draw_regime_summary(c.result.helpful_array);
    } else if (!c.last_error.empty()) {
        draw_error_box("##basins_err", c.last_error);
    }
}

// Plot Basins: inner tab-bar по 5 представлениям. Heatmap-views — per
// (config × tab) в std::map по owner_id. HeatmapView/Plot2DView хранят
// data_gen_cached внутри без учёта owner_id, поэтому переиспользовать один
// view между разными configs нельзя (после Run All у двух configs одинаковый
// data_generation=1 → cache не invalidate'тся и на чужой вкладке показывается
// предыдущий buffer). Map с lazy-init решает это и сохраняет независимый
// zoom/pan per (config, tab).
// spiral_coords_from_center / renumber_basins_spiral / ensure_basins_spiral_cache
// переехали в analysis_session.cpp: перенумерация нужна не только для
// отрисовки, но и для выбора представительных ячеек в фазовых портретах по
// бассейнам (BasinsAnalysisSession::rebuild_phase_ics), а тот слой про GUI
// ничего не знает. Объявления — в analysis_session.h.

static void draw_basins_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) {
    BasinsAnalysisSession& s = model.basins_session;
    if (s.configs.empty()) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
        s.active_config_index = 0;
    BasinsConfig& c = s.configs[s.active_config_index];
    // Owner IDs зависят от индекса config — каждый basin имеет независимый
    // zoom/pan per inner tab. Схема: 0x1BA50000 + cfg*5 + tab (max 50 configs).
    const unsigned base_oid = basins_base_oid(s.active_config_index);
    static std::unique_ptr<PlotRenderer> renderer;
    // hm_basins — на уровне файла (g_hm_basins), см. комментарий у объявления.
    static std::map<unsigned, std::unique_ptr<HeatmapView>> hm_avgpk, hm_avgint, hm_states;
    static std::map<unsigned, std::unique_ptr<Plot2DView>>  scatter_views;
    if (!renderer) renderer = std::make_unique<PlotRenderer>();
    auto get_hm = [](std::map<unsigned, std::unique_ptr<HeatmapView>>& m, unsigned oid,
                     bool discrete_default) -> HeatmapView& {
        auto& slot = m[oid];
        if (!slot) {
            slot = std::make_unique<HeatmapView>();
        }
        // Флаг discrete_default читается в render() на первом кадре с данными
        // и один раз применяется (см. heatmap_view.cpp). Ставим каждый кадр
        // сам параметр — легко и дёшево; apply идёт ровно один раз.
        slot->discrete_default = discrete_default;
        return *slot;
    };
    auto get_scatter = [](unsigned oid) -> Plot2DView& {
        auto& slot = scatter_views[oid];
        if (!slot) {
            slot = std::make_unique<Plot2DView>();
            slot->points_mode = true;
            slot->show_legend = false;
            slot->point_size_px = 3.0f;
            slot->pad_x = false;
        }
        return *slot;
    };
    HeatmapView& hm_basins_v = get_hm(g_hm_basins, base_oid + 0u, /*discrete*/ true);
    HeatmapView& hm_avgpk_v  = get_hm(hm_avgpk,  base_oid + 1u, false);
    HeatmapView& hm_avgint_v = get_hm(hm_avgint, base_oid + 2u, false);
    HeatmapView& hm_states_v = get_hm(hm_states, base_oid + 3u, false);
    Plot2DView&  scatter_v   = get_scatter(base_oid + 4u);

    if (!c.last_run_ok || c.result.basin_idx.empty()) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }

    // Right-click "Export data..." — exports the full 4-file Basins set
    // (basin_idx, avg_peaks, avg_intervals, helpful_array) at the chosen
    // path. All four heatmap views share the same source result, so the
    // same lambda works for any tab. Scatter (tab 4) uses Plot2DView, so
    // the same hook is set there too. Если включён Renumber (spiral),
    // basin_idx в файл уйдёт перенумерованный — чтобы экспорт совпадал с
    // тем, что пользователь видит на экране.
    const bool basins_busy = s.in_flight &&
                             s.active_config_index == s.running_config_index;
    auto basins_export_extras = [&c, &cb, basins_busy]() {
        draw_export_menu_item(basins_busy, cb, [&c](const std::string& path) {
            if (c.renumber_spiral) {
                ensure_basins_spiral_cache(c);
                BasinsResult tmp = c.result;
                tmp.basin_idx       = c.basin_idx_spiral;
                tmp.n_clusters      = c.n_clusters_spiral;
                tmp.min_cluster_idx = c.min_cluster_idx_spiral;
                data_export::export_basins(tmp, path);
            } else {
                data_export::export_basins(c.result, path);
            }
        });
    };
    hm_basins_v.popup_extras = basins_export_extras;
    hm_avgpk_v.popup_extras  = basins_export_extras;
    hm_avgint_v.popup_extras = basins_export_extras;
    hm_states_v.popup_extras = basins_export_extras;
    scatter_v.popup_extras   = basins_export_extras;

    // Inner tab-bar — переключение по 5 видам.
    // Имена 2-го и 3-го табов нейтральные — они показывают выбранную фичу
    // (Feature 1/2 из BasinsConfig), которая может быть не "avg peaks/interval".
    const char* tab_names[5] = { "Basins", "Feature 1", "Feature 2", "States", "Scatter" };
    if (ImGui::BeginTabBar("##basins_inner")) {
        for (int t = 0; t < 5; ++t) {
            if (ImGui::BeginTabItem(tab_names[t])) {
                c.active_plot_tab = t;
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
    }

    // Combo выбора colormap — свой для каждого heatmap-таба (Basins/AvgPk/
    // AvgInt/States). Каждый выбор пишется в свой field в AppModel и
    // персистится в _app_config.json. Все эти поля независимы от
    // model.heatmap_colormap (тот шарится Bif/LLE/LS) — смена здесь их не
    // затрагивает, и наоборот. Scatter-таб использует Plot2DView, без
    // colormap.
    {
        // Per-tab colormap: приоритет — выбор, сохранённый в ЭТОМ config'е
        // (уходит в _last_basins.json); -1 = не задан → app-дефолт из Settings.
        // Так два basins-config'а могут иметь разные колормапы — как у
        // Bif/LLE/LS/DFT1D. Раньше выбор был только глобальный.
        int* cfg_field  = nullptr;
        int  app_default = kColormapTurbo;
        HeatmapView* active_hm = nullptr;
        switch (c.active_plot_tab) {
            case 0: cfg_field = &c.colormap_idx[0]; app_default = model.basins_colormap;        active_hm = &hm_basins_v; break;
            case 1: cfg_field = &c.colormap_idx[1]; app_default = model.basins_avgpk_colormap;  active_hm = &hm_avgpk_v;  break;
            case 2: cfg_field = &c.colormap_idx[2]; app_default = model.basins_avgint_colormap; active_hm = &hm_avgint_v; break;
            case 3: cfg_field = &c.colormap_idx[3]; app_default = model.basins_states_colormap; active_hm = &hm_states_v; break;
            default: break;  // Scatter (4) — Plot2DView, без colormap
        }
        // Renumber (spiral) — общий для табов 0 (Basins) и 4 (Scatter).
        // На остальных табах не показываем — там cluster id'ы не отрисовываются.
        auto draw_renumber = [&](bool same_line) {
            if (c.active_plot_tab != 0 && c.active_plot_tab != 4) return;
            if (same_line) ImGui::SameLine();
            if (ImGui::Checkbox("Renumber (spiral)", &c.renumber_spiral)) {
                // Заставляем cache пересчитаться при следующем доступе,
                // даже если data_generation тот же (после toggle off→on).
                c.basin_idx_spiral_gen = -1;
            }
        };

        if (cfg_field && active_hm) {
            // Синхронизируем выбор в сам view ДО тулбара: тулбар читает
            // hv.colormap как источник истины. Без этого на первом кадре combo
            // показал бы дефолт view'а вместо сохранённого значения.
            const int cm = colormap_id_or(*cfg_field, app_default);
            active_hm->colormap = (HeatmapColormap)cm;

            HeatmapToolbarOpts topts;
            topts.persist_colormap = [&](int picked) {
                *cfg_field = picked;   // per-config, уходит в сессию
                if (!model.loaded_name.empty())
                    lib.save_session(model.loaded_name, "_last_basins",
                                     session_to_json_basins(model.basins_session));
            };
            topts.extras = [&]() { draw_renumber(/*same_line*/ true); };
            draw_heatmap_toolbar(*active_hm, topts);
        } else {
            draw_renumber(/*same_line*/ false);   // Scatter-таб: только Renumber
        }
    }
    if (c.renumber_spiral) ensure_basins_spiral_cache(c);

    int n = c.result.n_pts;
    size_t total = (size_t)n * (size_t)n;
    double xlo = c.result.axis_x_lo, xhi = c.result.axis_x_hi;
    double ylo = c.result.axis_y_lo, yhi = c.result.axis_y_hi;

    auto axis_name = [&](int var_idx) -> std::string {
        if (var_idx >= 0 && var_idx < (int)s.vars.size())
            return s.vars[var_idx] + "(0)";
        return std::string("x");
    };
    std::string ax_x = axis_name(c.result.axis_x_var);
    std::string ax_y = axis_name(c.result.axis_y_var);

    bool fit = c.fit_request;
    if (fit) c.fit_request = false;

    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();

    if (c.active_plot_tab == 0) {
        // Basins idx — turbo-discrete через GL_NEAREST. Spectrum [min..max].
        // Если Renumber (spiral) включён, читаем перенумерованный basin_idx
        // и считаем colorbar-диапазон от него же — иначе brightest band не
        // совпадёт с фактическими id'ами на хитмапе.
        const int* src_idx = c.renumber_spiral
                                 ? c.basin_idx_spiral.data()
                                 : c.result.basin_idx.data();
        static std::vector<double> buf;
        buf.resize(total);
        for (size_t k = 0; k < total; ++k) buf[k] = (double)src_idx[k];
        // Диапазон colorbar'а — общий хелпер (шарится с цветом траекторий в
        // фазовых портретах по бассейнам), см. basins_colorbar_range.
        double vmin = 0.0, vmax = 0.0;
        basins_colorbar_range(c, vmin, vmax);
        // colormap уже засинхронен из model.basins_colormap перед тулбаром.
        hm_basins_v.x_axis.name = ax_x;
        hm_basins_v.y_axis.name = ax_y;
        // gen-token включает renumber_spiral, иначе HeatmapView::data_gen_cached
        // не инвалидируется при toggle галки и пиксели остаются от прошлой версии.
        // Сам swap_axes уже отлавливается через HeatmapView::swap_axes_cached_.
        int hm_basins_gen = c.data_generation * 2 + (c.renumber_spiral ? 1 : 0);
        hm_basins_v.render(*renderer, origin, avail,
                          /*owner_id*/ base_oid + 0u, hm_basins_gen,
                          n, n, buf.data(),
                          xlo, xhi, ylo, yhi,
                          vmin, vmax, fit);
    }
    else if (c.active_plot_tab == 1) {
        hm_avgpk_v.x_axis.name = ax_x;
        hm_avgpk_v.y_axis.name = ax_y;
        hm_avgpk_v.render(*renderer, origin, avail,
                         /*owner_id*/ base_oid + 1u, c.data_generation,
                         n, n, c.result.avg_peaks.data(),
                         xlo, xhi, ylo, yhi,
                         c.result.avg_peaks_min, c.result.avg_peaks_max, fit);
    }
    else if (c.active_plot_tab == 2) {
        hm_avgint_v.x_axis.name = ax_x;
        hm_avgint_v.y_axis.name = ax_y;
        hm_avgint_v.render(*renderer, origin, avail,
                          /*owner_id*/ base_oid + 2u, c.data_generation,
                          n, n, c.result.avg_intervals.data(),
                          xlo, xhi, ylo, yhi,
                          c.result.avg_intervals_min, c.result.avg_intervals_max, fit);
    }
    else if (c.active_plot_tab == 3) {
        // States: рисуем helpful_array КАК ЕСТЬ, в канонических REGIME_*
        // (-1 = FP, 0 = Unbound, 1 = Oscillation). Раньше здесь была
        // перенумерация в 0/1/2, из-за которой подпись под плотом, легенда и
        // коды в экспортируемом _3.csv означали разные вещи.
        static std::vector<double> buf;
        buf.resize(total);
        for (size_t k = 0; k < total; ++k)
            buf[k] = (double)regime_code(c.result.helpful_array[k]);
        hm_states_v.x_axis.name = ax_x;
        hm_states_v.y_axis.name = ax_y;
        hm_states_v.render(*renderer, origin, avail,
                          /*owner_id*/ base_oid + 3u, c.data_generation,
                          n, n, buf.data(),
                          xlo, xhi, ylo, yhi,
                          -1.0, 1.0, fit);
        // Подсказка под плотом — числовые уровни, цвет зависит от выбранной colormap.
        ImGui::TextDisabled("Levels: -1 = FixedPoint, 0 = Unbound, 1 = Oscillation");
    }
    else if (c.active_plot_tab == 4) {
        // Scatter (avgPeak, avgInterval), точки сгруппированы по basin_idx.
        // Каждый кластер — своя серия (PlotSeriesInput с собственным цветом).
        // С renumber_spiral берём перенумерованные id'ы, чтобы цвета и подписи
        // ("c1", "c2", ...) совпадали с тем, что показывает Basins heatmap.
        const int* src_idx = c.renumber_spiral
                                 ? c.basin_idx_spiral.data()
                                 : c.result.basin_idx.data();
        int min_id = c.renumber_spiral ? c.min_cluster_idx_spiral : c.result.min_cluster_idx;

        // Сгруппируем точки по basin_idx.
        std::map<int, std::vector<float>> bufs;
        int valid_pts = 0;
        for (size_t k = 0; k < total; ++k) {
            int id = src_idx[k];
            double xp = c.result.avg_peaks[k];
            double yp = c.result.avg_intervals[k];
            if (!std::isfinite(xp) || !std::isfinite(yp)) continue;
            if (xp ==  kSentinelNoData || xp == -kSentinelNoData ||
                yp ==  kSentinelNoData || yp == -kSentinelNoData) continue;
            bufs[id].push_back((float)xp);
            bufs[id].push_back((float)yp);
            ++valid_pts;
        }
        if (valid_pts == 0) {
            ImGui::TextDisabled("No valid (avgPeak, avgInterval) points.");
            return;
        }

        // Буферы локальные: render() ниже забирает точки синхронно, дольше
        // вызова указатели не нужны (static здесь ещё и делил бы память между
        // config'ами бассейнов).
        std::vector<std::vector<float>> series_buffers;
        std::vector<std::string>        series_labels;
        series_buffers.reserve(bufs.size());
        series_labels.reserve(bufs.size());

        std::vector<PlotSeriesInput> series_in;
        std::vector<bool> init_vis, glob_vis;
        for (auto& kv : bufs) {
            int id = kv.first;
            series_buffers.push_back(std::move(kv.second));
            series_labels.push_back("c" + std::to_string(id));
            // Цвет per-cluster через ic_base_color (golden-ratio hash). Сдвиг
            // на (id - min_id) — чтобы FP-кластеры (отрицательные) и Osc
            // (положительные) тоже разделялись.
            PlotSeriesInput si;
            si.points   = series_buffers.back().empty() ? nullptr : series_buffers.back().data();
            si.n_points = (int)(series_buffers.back().size() / 2);
            si.color    = ic_base_color(id - min_id);
            si.label    = series_labels.back();
            series_in.push_back(si);
            init_vis.push_back(true);
            glob_vis.push_back(true);
        }
        // Имена осей scatter'а — выбранные фичи, из общего kFeatureNames
        // (тот же список, что в комбо панели настроек).
        int f1 = (c.feature1 >= 0 && c.feature1 < BF_FEATURE_COUNT) ? c.feature1 : BF_FEATURE1_DEFAULT;
        int f2 = (c.feature2 >= 0 && c.feature2 < BF_FEATURE_COUNT) ? c.feature2 : BF_FEATURE2_DEFAULT;
        scatter_v.x_axis.name = kFeatureNames[f1];
        scatter_v.y_axis.name = kFeatureNames[f2];
        // ε-окружность под курсором. Здесь она точна по определению: точки
        // scatter'а — это ровно те avg_peaks/avg_intervals-буферы, которые
        // читает cell-level DBSCAN (CUDA_dbscan_kernel: sqrt(dx^2+dy^2) <= eps),
        // и множители Feature 1/2 в них уже вписаны на GPU. Так что накрытые
        // кружком ячейки — кандидаты попасть в один бассейн.
        scatter_v.hover_circle_r = parse_ratio_or(c.eps_dbscan_text, 0.5);
        // gen-token включает renumber_spiral — иначе Plot2DView::series_cache_
        // не перезаливает GPU-буфер и подписи/цвета остаются от прошлой версии.
        int scatter_gen = c.data_generation * 2 + (c.renumber_spiral ? 1 : 0);
        scatter_v.render(*renderer, origin, avail,
                             /*owner_id*/ base_oid + 4u, scatter_gen,
                             series_in, init_vis, glob_vis, fit);
    }
}

// Окна фазовых портретов по бассейнам — только для АКТИВНОГО config'а
// (панель хитмапы показывает его же). Набор проекций и результат у каждого
// config'а свои, поэтому переключение вкладки сразу показывает свои окна с
// уже посчитанными данными.
static void draw_basins_phase_windows(AppModel& model, const GuiCallbacks& cb) {
    BasinsAnalysisSession& s = model.basins_session;
    s.ensure_phase_slots();
    if (s.configs.empty()) return;
    int ci = s.active_config_index;
    if (ci < 0 || ci >= (int)s.configs.size()) ci = 0;
    BasinsConfig&    c  = s.configs[(size_t)ci];
    BasinsPhaseSlot& sl = s.phase_slot(ci);

    // 3D-проекций в этом режиме нет (Plot3DView не умеет точечные серии, а
    // бассейны-равновесия рисуются маркером) — старые сессии мягко приводим.
    for (auto& pr : sl.phase.projections)
        if (pr.type == ProjType::Phase3D) pr.type = ProjType::Phase2D;

    // display-id ячейки → цвет бассейна + признак «равновесие» (id < 0).
    const int* ids = basins_display_ids(c);
    const size_t n_cells = c.result.basin_idx.size();
    PhaseStyleFn style_fn = [&c, &sl, &model, ci, ids, n_cells]
                            (int k, PhaseSeriesStyle& out) -> bool {
        if (!ids || k < 0 || k >= (int)sl.cells.size()) return false;
        const int cell = sl.cells[(size_t)k];
        if (cell < 0 || (size_t)cell >= n_cells) return false;
        const int id = ids[(size_t)cell];
        out.color      = basins_id_color(c, model, ci, id);
        out.as_point   = (id < 0);         // FP-бассейн — одна точка
        out.point_size = c.pp_marker_size;
        out.marker     = (int)PointMarker::Circle;
        return true;
    };

    // Диаграмма признаков в этих окнах — облако (пик; IPI) по выбранным
    // ячейкам, а cell-level DBSCAN бассейнов мерит eps в пространстве
    // (Feature1 x mult1, Feature2 x mult2), по ОДНОЙ точке на ячейку. Единицы
    // совпадают с осями диаграммы ровно тогда, когда выбраны средние: тогда
    // точка ячейки — центроид этого самого облака, и eps-кружок на нём
    // осмыслен. На RMS/StDev/log-фичах множители относятся к другой величине,
    // и растягивать на них ось «пик» значило бы врать — там оставляем сырые
    // оси и без кружка, как было. Клампы f1/f2 повторяют engine
    // (analysis_session.cpp): вне диапазона он берёт дефолт, т.е. средние.
    FeatureClusterParams clust;
    {
        const int f1 = (c.feature1 >= 0 && c.feature1 < BF_FEATURE_COUNT) ? c.feature1 : BF_FEATURE1_DEFAULT;
        const int f2 = (c.feature2 >= 0 && c.feature2 < BF_FEATURE_COUNT) ? c.feature2 : BF_FEATURE2_DEFAULT;
        if (f1 == BF_AVG_PEAKS && f2 == BF_AVG_INTERVALS) {
            clust.valid         = true;
            clust.mult_peak     = parse_ratio_or(c.mult_feature1_text, 1.0);
            clust.mult_interval = parse_ratio_or(c.mult_feature2_text, 1.0);
            clust.eps           = parse_ratio_or(c.eps_dbscan_text,    0.5);
        }
    }

    // title_suffix / owner_id_delta — свои на config: docking-раскладка в
    // imgui.ini и кэш рендереров не должны пересекаться ни между конфигами,
    // ни с проекциями режима Phase analysis.
    const size_t n_before = sl.phase.projections.size();
    draw_projection_windows(sl.phase, cb, {}, {},
                            "##bpp" + std::to_string(ci),
                            0x1BA50000 + ci * 16,
                            style_fn, clust);
    // Окно могли закрыть крестиком — список окон изменился, надо сохранить.
    if (sl.phase.projections.size() != n_before) s.phase_settings_dirty = true;
}

// Раз в кадр: пометить устаревшие портреты и, если включён Autorun, поставить
// их в очередь. Дебаунс 300 мс — иначе каждое нажатие клавиши в текстовом поле
// стартовало бы новый GPU-прогон. Проходим по ВСЕМ config'ам: после «Run all…»
// автозапуск должен подхватить каждый пересчитанный конфиг, а не только
// открытый в данный момент.
static void basins_phase_tick(AppModel& model) {
    BasinsAnalysisSession& s = model.basins_session;
    s.ensure_phase_slots();
    const double now = ImGui::GetTime();
    for (int i = 0; i < (int)s.configs.size(); ++i) {
        BasinsConfig&    c  = s.configs[(size_t)i];
        BasinsPhaseSlot& sl = *s.phase_slots[(size_t)i];
        const std::string sig = build_basins_phase_signature(c);
        if (sig != sl.pending_signature) {
            sl.pending_signature = sig;
            sl.pending_since     = now;
        }
        if (sig == sl.run_signature) continue;         // актуально
        if (!c.pp_autorun) continue;
        if (!c.last_run_ok || c.result.basin_idx.empty()) continue;
        if (sl.phase.in_flight) continue;
        if (now - sl.pending_since < kBasinsPhaseDebounceSec) continue;   // ещё печатают
        s.request_phase_run(i);
    }
}

// ============================================================
// Fast Synchro Controls + Plot — recurrent synchronization (anti-sync).
// Mode 0 = On Attractor (trajectory + per-point error); Mode 1 = On Grid.
// ============================================================
static void draw_fastsync_controls(AppModel& model, SystemLibrary& lib) {
    FastSyncAnalysisSession& s = model.fastsync_session;

    // Source of truth для custom-схем — AppModel (редактируются в Library).
    // load_from_record() сохраняет снапшот ОДНОКРАТНО при входе в режим, поэтому
    // схемы, добавленные позже, не видны без рефреша. Подтягиваем актуальный
    // список каждый кадр, чтобы Combo и compute_krs_for_scheme работали с live.
    s.custom_schemes = model.custom_schemes;

    ImGui::Text("Fast Synchro");
    ImGui::TextDisabled("Recurrent synchronization analysis (anti-sync error).");

    // ----- Run / Cancel / Run all... (moved up to sit right under the system
    // picker, above the tab bar — these buttons drive the currently active
    // config, so they stay accessible without scrolling past all sections). -----
    // Batch "Run all..." across FastSync configs. Pushes selected indices into
    // model.fastsync_queue; draw_gui ticks the queue after polls.
    {
        const bool no_cfg = s.configs.empty();
        RunAllGroup g;
        g.pick_id = "pfs_";
        g.n       = (int)s.configs.size();
        g.label   = [&s](int i) { return s.configs[i].label; };
        g.enqueue = [&model](int i) { model.fastsync_queue.push_back({i}); };
        draw_run_and_run_all("##run_all_fastsync", s.in_flight, no_cfg,
                             s.in_flight || no_cfg, 160.0f,
                             [&model, &s]() {
                                 if (!model.parametric_engine)
                                     model.parametric_engine = std::make_unique<ParametricEngine>();
                                 s.run_async(*model.parametric_engine, s.active_config_index);
                             },
                             // Cancel — только у FastSync, между Run и Run all.
                             [&s]() {
                                 ImGui::SameLine();
                                 if (s.in_flight && ImGui::Button("Cancel")) s.request_cancel();
                             },
                             { g },
                             [&model]() { model.start_next_in_fastsync_queue(); },
                             model.fastsync_queue.size());
    }
    ImGui::Separator();

    // Tab bar для config'ов.
    const TabBarResult tabs = draw_config_tab_bar(
        "##fs_tabs", "fs_tab_", (int)s.configs.size(),
        s.in_flight, s.running_config_index,
        [&s](int i) { return s.configs[i].label; },
        {},
        [&s]() { s.add_config(); });
    if (tabs.active    >= 0) s.active_config_index = tabs.active;
    if (tabs.to_remove >= 0) model.remove_fastsync_config(tabs.to_remove);

    if (s.configs.empty()) {
        ImGui::TextDisabled("No FastSync configs. Press '+' to add one.");
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
        s.active_config_index = 0;
    FastSyncConfig& c = s.configs[s.active_config_index];

    // Label rename.
    draw_label_rename("Label##fs_label", c.label);
    ImGui::Separator();

    // ----- Mode -----
    ImGui::Text("Mode:");
    ImGui::RadioButton("On Attractor", &c.mode, 0); ImGui::SameLine();
    ImGui::RadioButton("On Grid",      &c.mode, 1);
    ImGui::Separator();

    // ----- Scheme -----
    draw_scheme_combo("Scheme", c.scheme, s.custom_schemes);
    if (c.scheme == "CD" || custom_scheme_uses_symmetry(c.scheme, s.custom_schemes))
        InputNumStr("symmetry s", c.symmetry_s, kFieldW);
    ImGui::Separator();

    // ----- Mode-specific axes -----
    if (c.mode == 1 && !s.vars.empty()) {
        if (c.axis_x_var < 0 || c.axis_x_var >= (int)s.vars.size()) c.axis_x_var = 0;
        if (c.axis_y_var < 0 || c.axis_y_var >= (int)s.vars.size())
            c.axis_y_var = (s.vars.size() > 1) ? 1 : 0;
        const std::vector<const char*> items = c_str_list(s.vars);

        const char* axis_role_x = c.grid_swap_master_slave ? "Axis X (slave IC)" : "Axis X (master IC)";
        const char* axis_role_y = c.grid_swap_master_slave ? "Axis Y (slave IC)" : "Axis Y (master IC)";
        ImGui::SetNextItemWidth(kComboW);
        ImGui::Combo(axis_role_x, &c.axis_x_var, items.data(), (int)items.size());
        InputNumStr("X lo", c.axis_x_lo_text, kFieldW);
        InputNumStr("X hi", c.axis_x_hi_text, kFieldW);
        ImGui::SetNextItemWidth(kComboW);
        ImGui::Combo(axis_role_y, &c.axis_y_var, items.data(), (int)items.size());
        InputNumStr("Y lo", c.axis_y_lo_text, kFieldW);
        InputNumStr("Y hi", c.axis_y_hi_text, kFieldW);
        InputNumStr("Resolution", c.n_pts_text, kFieldW);
        // Какую сторону перебирать по сетке: master IC (legacy default) или slave IC.
        ImGui::Checkbox("Vary slave IC (master fixed)##fs_gridswap", &c.grid_swap_master_slave);
        ImGui::TextDisabled("Off: grid sweeps master IC, slave fixed. On: grid sweeps slave IC, master fixed.");
        ImGui::Separator();
    }
    else if (c.mode == 0 && !s.vars.empty()) {
        // На траектории — какие 2 переменные показывать как X/Y фазового портрета.
        const std::vector<const char*> items = c_str_list(s.vars);
        ImGui::SetNextItemWidth(kComboW);
        ImGui::Combo("Display X var", &c.axis_x_var, items.data(), (int)items.size());
        ImGui::SetNextItemWidth(kComboW);
        ImGui::Combo("Display Y var", &c.axis_y_var, items.data(), (int)items.size());
        ImGui::Separator();
    }

    // ----- Integration (collapsible) -----
    if (ImGui::CollapsingHeader("Integration", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("h",                c.h_text, kFieldW);
        if (c.mode == 0) {
            InputNumStr("t_max",        c.t_max_text, kFieldW);
        }
        InputNumStr("transient",        c.transient_text, kFieldW);
        bool window_changed = InputNumStr("window",           c.window_text, kFieldW);
        bool iter_changed   = InputNumStr("iter of synch",    c.iter_of_synchr_text, kFieldW);
        if (!window_changed && !iter_changed) {
            // window/iter — источник истины в этом кадре (или ничего не
            // менялось, напр. только что загрузили сессию): пересчитываем
            // total sync time ДО отрисовки его поля, без задержки в кадр.
            double window_v = parse_ratio_or(c.window_text, 0.0);
            double iter_v   = std::max(1.0, parse_ratio_or(c.iter_of_synchr_text, 0.0));
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.10g", (2.0 * iter_v - 1.0) * window_v);
            c.total_time_text = buf;
        }
        if (InputNumStr("total sync time", c.total_time_text, kFieldW)) {
            // Пользователь правит total → пересчитываем window (iter фиксирован).
            double total_v = parse_ratio_or(c.total_time_text, 0.0);
            double iter_v  = std::max(1.0, parse_ratio_or(c.iter_of_synchr_text, 0.0));
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.10g", total_v / (2.0 * iter_v - 1.0));
            c.window_text = buf;
        }
        if (c.mode == 0) {
            InputNumStr("decimator",    c.pre_scaller_text, kFieldW);
        }
        InputNumStr("max value",        c.max_value_text, kFieldW);
    }

    // ----- Synchro runtime (collapsible) -----
    if (ImGui::CollapsingHeader("Synchro runtime", ImGuiTreeNodeFlags_DefaultOpen)) {
        static const char* tos_names[] = { "Unidirectional", "Bidirectional" };
        ImGui::SetNextItemWidth(kComboW);
        ImGui::Combo("Type of synch.", &c.type_of_synch, tos_names, IM_ARRAYSIZE(tos_names));
        static const char* ee_names[] = {
            "0: RMS on last iter",
            "1: # iters to reach FS_error_trs",
            "2: RMS at last point"
        };
        ImGui::SetNextItemWidth(280);
        ImGui::Combo("Error estim.", &c.error_estim, ee_names, IM_ARRAYSIZE(ee_names));
        InputNumStr("FS error trs.", c.fs_error_trs_text, kFieldW);
    }

    // ----- Parameters (collapsible, placed under Synchro runtime) -----
    draw_named_num_fields("Parameters", s.params, c.param_values);

    // ----- Paired collapsible sections: one click on either header collapses
    // both halves together. Shared open state is forced into each header via
    // SetNextItemOpen each frame; IsItemToggledOpen captures the click and
    // flips the shared state. Variables render row-by-row so x_master pairs
    // horizontally with x_slave (same for K forward/backward). -----
    auto paired_header = [](const char* label, bool& open) {
        ImGui::SetNextItemOpen(open, ImGuiCond_Always);
        ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_None);
        if (ImGui::IsItemToggledOpen()) open = !open;
    };
    auto paired_input = [&](const char* var,
                            std::map<std::string, std::string>& m,
                            const char* id_prefix) {
        std::string pid = std::string(id_prefix) + var;
        ImGui::PushID(pid.c_str());
        InputNumStr(var, m[var], 100);
        ImGui::PopID();
    };

    // Master init | Slave init.
    static bool ic_open = true;
    if (ImGui::BeginTable("##fs_ic_table", 2)) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        paired_header("Master init##fs_master_ic", ic_open);
        ImGui::TableSetColumnIndex(1);
        paired_header("Slave init##fs_slave_ic", ic_open);
        if (ic_open) {
            for (const auto& v : s.vars) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                paired_input(v.c_str(), c.ic_master, "icm_");
                ImGui::TableSetColumnIndex(1);
                paired_input(v.c_str(), c.ic_slave,  "ics_");
            }
        }
        ImGui::EndTable();
    }

    // K forward (h>0) | K backward (h<0).
    static bool k_open = true;
    if (ImGui::BeginTable("##fs_k_table", 2)) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        paired_header("K forward (h>0)##fs_kf",  k_open);
        ImGui::TableSetColumnIndex(1);
        paired_header("K backward (h<0)##fs_kb", k_open);
        if (k_open) {
            for (const auto& v : s.vars) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                paired_input(v.c_str(), c.k_forward,  "kf_");
                ImGui::TableSetColumnIndex(1);
                paired_input(v.c_str(), c.k_backward, "kb_");
            }
        }
        ImGui::EndTable();
    }

    // ----- CSV output (collapsible, moved to the bottom) -----
    draw_csv_output_block("CSV output", "Save to file##fs_csv", c.csv_save_enabled,
                          "##fs_csv_path", c.csv_output_path,
                          c.mode == 0
                              ? "Writes one row per trajectory point: x[0],..,x[N-1],sync_error."
                              : "Writes 2 header lines (X/Y ranges) + n_pts x n_pts error matrix (row-major).");

    if (c.last_run_ok) {
        if (c.mode == 0)
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: traj %d pts, sync_err [%.4g, %.4g]",
                c.result.n_pts_traj, c.result.min_val, c.result.max_val);
        else
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: grid %dx%d, sync_err [%.4g, %.4g]",
                c.result.n_pts_grid, c.result.n_pts_grid,
                c.result.min_val, c.result.max_val);
    } else if (!c.last_error.empty()) {
        draw_error_box("##fs_err", c.last_error, /*lines*/ 10);
    }
}

// Plot Fast Synchro: либо colored trajectory (mode 0), либо heatmap (mode 1).
static void draw_fastsync_plot(AppModel& model, const GuiCallbacks& cb) {
    FastSyncAnalysisSession& s = model.fastsync_session;
    if (s.configs.empty()) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
        s.active_config_index = 0;
    FastSyncConfig& c = s.configs[s.active_config_index];

    const unsigned base_oid = 0x5F50000u + (unsigned)s.active_config_index;
    static std::unique_ptr<PlotRenderer> renderer;
    static std::map<unsigned, std::unique_ptr<HeatmapView>> hm_map;
    static std::map<unsigned, std::unique_ptr<Plot2DView>>  traj_map;
    if (!renderer) renderer = std::make_unique<PlotRenderer>();

    // ---- Visualization toolbar — общая реализация (draw_heatmap_toolbar) ----
    // Цветовая шкала FastSync держится в HeatmapView даже в mode 0 (colored
    // trajectory, где хитмапы нет): так тулбар совпадает с остальными
    // вкладками до последнего виджета. Персистентность при этом остаётся в
    // config'е (он уходит в сессию), поэтому синхронизируем в обе стороны.
    auto& hv_slot = hm_map[base_oid];
    if (!hv_slot) hv_slot = std::make_unique<HeatmapView>();
    HeatmapView& hv = *hv_slot;

    c.colormap_idx = colormap_id_or(c.colormap_idx, kColormapTurbo);
    hv.colormap         = (HeatmapColormap)c.colormap_idx;   // config → view
    hv.autoscale        = c.autoscale_color;
    hv.manual_vmin_text = c.c_min_text;
    hv.manual_vmax_text = c.c_max_text;
    hv.swap_axes        = c.swap_axes;
    {
        HeatmapToolbarOpts topts;
        topts.persist_colormap = [&](int cm) { c.colormap_idx = cm; };
        // Swap axes осмыслен только в grid-режиме (там оси — две IC).
        topts.show_swap = (c.mode != 0);
        if (c.mode == 0) {
            // Толщина/α для colored trajectory. Выбор Display X/Y живёт в
            // Controls-панели рядом с прочими compute-параметрами.
            topts.extras_tail = [&]() {
                ImGui::SameLine(); ImGui::SetNextItemWidth(120);
                ImGui::SliderFloat("Line width", &c.line_width, 0.1f, 5.0f, "%.2f");
                ImGui::SameLine(); ImGui::SetNextItemWidth(120);
                ImGui::SliderFloat("Alpha", &c.alpha, 0.0f, 1.0f, "%.2f");
            };
        }
        draw_heatmap_toolbar(hv, topts);
    }
    c.autoscale_color = hv.autoscale;              // view → config
    c.c_min_text      = hv.manual_vmin_text;
    c.c_max_text      = hv.manual_vmax_text;
    c.swap_axes       = hv.swap_axes;

    if (!c.last_run_ok) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }

    bool fit = c.fit_request;
    if (fit) c.fit_request = false;
    ImVec2 avail  = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();

    auto var_name = [&](int idx) -> std::string {
        if (idx >= 0 && idx < (int)s.vars.size()) return s.vars[idx];
        return std::string("x");
    };

    // cmin/cmax: при autoscale берём диапазон актуальных значений из result.
    // Если пользователь ввёл vmin > vmax вручную — это сигнал «перевернуть
    // колормапу», а не ошибка. Сортируем диапазон и взводим invert_cmap, чтобы
    // mode-0 (colored trajectory) отзеркалил соответствие value→цвет.
    // (Heatmap-режим использует user-typed как и раньше — HeatmapView сам
    // защищается от обратного диапазона.)
    double cmin_user, cmax_user;
    if (c.autoscale_color) {
        cmin_user = c.result.min_val;
        cmax_user = c.result.max_val;
        if (!(cmax_user > cmin_user)) cmax_user = cmin_user + 1.0;
    } else {
        cmin_user = parse_ratio_or(c.c_min_text, -12.0);
        cmax_user = parse_ratio_or(c.c_max_text,   0.0);
    }
    bool invert_cmap = (!c.autoscale_color) && (cmin_user > cmax_user);
    double cmin = std::min(cmin_user, cmax_user);
    double cmax = std::max(cmin_user, cmax_user);
    if (!(cmax > cmin)) cmax = cmin + 1.0;
    HeatmapColormap cmap = (HeatmapColormap)colormap_id_or(c.colormap_idx, kColormapTurbo);

    if (c.mode == 0) {
        // Colored trajectory + manual colorbar справа.
        const int nX = c.result.amountOfX_traj;
        if (nX <= 0 || c.result.n_pts_traj <= 0) {
            ImGui::TextDisabled("No trajectory data.");
            return;
        }
        int vx = (c.axis_x_var >= 0 && c.axis_x_var < nX) ? c.axis_x_var : 0;
        int vy = (c.axis_y_var >= 0 && c.axis_y_var < nX) ? c.axis_y_var : (nX > 1 ? 1 : 0);

        // Резервируем место справа под colorbar. Тики и ширина считаются теми
        // же общими хелперами, что у HeatmapView — раньше здесь была своя
        // копия с 5 равноотстоящими тиками и продублированными константами.
        const std::vector<ColorbarTick> cb_ticks =
            colorbar_ticks((float)cmin, (float)cmax, /*n_discrete*/ 0);
        const float cb_total = colorbar_total_width(cb_ticks);

        ImVec2 plot_avail(std::max(64.0f, avail.x - cb_total), avail.y);

        auto& slot = traj_map[base_oid];
        if (!slot) {
            slot = std::make_unique<Plot2DView>();
            slot->points_mode  = false;
            slot->show_legend  = false;
            slot->imdraw_lines = true;
            slot->pad_x = true; slot->pad_y = true;
            // Это фазовый портрет (оси — переменные состояния), поэтому линий
            // x=0 / y=0 быть не должно — как в Phase 2D. Раньше здесь стоял
            // дефолт Plot2DView (true), и одна и та же по смыслу диаграмма
            // рисовалась в двух вкладках по-разному.
            slot->show_zero_x = false;
            slot->show_zero_y = false;
        }
        Plot2DView& v = *slot;
        v.x_axis.name = var_name(vx);
        v.y_axis.name = var_name(vy);
        v.line_thickness_px = c.line_width;

        // Если поменялись axes — форсим (a) fit, чтобы view нашёл новый bbox;
        // (b) re-upload GPU-кэша точек, иначе series_cache_.bbox() даст старый
        // диапазон. Поэтому передаём генерационный токен, зависящий от (vx,vy).
        static std::map<unsigned, std::pair<int,int>> last_axes;
        auto it_ax = last_axes.find(base_oid);
        if (it_ax == last_axes.end() || it_ax->second.first != vx || it_ax->second.second != vy) {
            v.view_valid = false;     // force autofit
            last_axes[base_oid] = { vx, vy };
        }

        // Пересобираем XY/values из полного буфера (без decimator'а — рисуем
        // все точки, ImDrawList сегмент-за-сегментом справляется).
        int n_in = c.result.n_pts_traj;
        static std::vector<float> xy_buf;
        static std::vector<float> err_buf;
        xy_buf.resize((size_t)n_in * 2);
        err_buf.resize((size_t)n_in);
        for (int i = 0; i < n_in; ++i) {
            xy_buf[2*i + 0] = (float)c.result.traj_full[(size_t)i * nX + (size_t)vx];
            xy_buf[2*i + 1] = (float)c.result.traj_full[(size_t)i * nX + (size_t)vy];
            // invert_cmap: reflect v across the sorted [cmin, cmax] midpoint.
            // Renderer computes t = (v - cmin)/(cmax - cmin); the reflection
            // maps it to 1 - t → cmap_sample sees the colors in reverse order.
            double v_err = c.result.sync_error[i];
            err_buf[i] = (float)(invert_cmap ? (cmin + cmax - v_err) : v_err);
        }

        // Painter's algorithm: сортируем сегменты по средней координате оси Z
        // (первая var, не равная vx/vy). Дальние сегменты рисуются первыми,
        // ближние — поверх. Без сортировки артефакты "красное поверх синего"
        // справа на скриншоте — последний по времени сегмент перекрывает
        // более ранние независимо от их Z.
        int vz = -1;
        for (int k = 0; k < nX; ++k) if (k != vx && k != vy) { vz = k; break; }
        static std::vector<int>   seg_order;
        static std::vector<float> seg_z;
        const int n_seg = n_in - 1;
        seg_order.clear();
        if (vz >= 0 && n_seg > 0) {
            seg_order.resize(n_seg);
            seg_z.resize(n_seg);
            for (int k = 0; k < n_seg; ++k) {
                seg_order[k] = k;
                float za = (float)c.result.traj_full[(size_t)k       * nX + (size_t)vz];
                float zb = (float)c.result.traj_full[(size_t)(k + 1) * nX + (size_t)vz];
                seg_z[k] = 0.5f * (za + zb);
            }
            if (c.invert_depth) {
                std::sort(seg_order.begin(), seg_order.end(),
                          [&](int a, int b) { return seg_z[a] > seg_z[b]; });
            } else {
                std::sort(seg_order.begin(), seg_order.end(),
                          [&](int a, int b) { return seg_z[a] < seg_z[b]; });
            }
        }

        std::vector<PlotSeriesInput> series_in(1);
        series_in[0].points   = xy_buf.data();
        series_in[0].n_points = n_in;
        series_in[0].color    = ImVec4(1, 1, 1, c.alpha);  // .w → alpha-multiplier на cmap_sample()
        series_in[0].label    = "trajectory";
        series_in[0].values   = err_buf.data();
        series_in[0].colormap = cmap;
        series_in[0].cmin     = (float)cmin;
        series_in[0].cmax     = (float)cmax;
        series_in[0].segment_order = seg_order.empty() ? nullptr : seg_order.data();
        std::vector<bool> vis(1, true);
        // Synthetic generation token: меняется при смене (data_generation, vx, vy),
        // чтобы Plot2DView::series_cache_ перезалил GPU-буфер → bbox()/autofit
        // подхватили новую X/Y проекцию.
        int gen_token = c.data_generation * 1000 + vx * 10 + vy;
        // Right-click popup получает дополнительный пункт "Invert depth axis"
        // через popup_extras callback. vz/var_name захватываются по значению.
        const int   vz_capture       = vz;
        const std::string vz_name    = (vz >= 0) ? var_name(vz) : std::string{};
        const bool fs_busy = s.in_flight &&
                             s.active_config_index == s.running_config_index;
        v.popup_extras = [vz_capture, vz_name, &c, &cb, fs_busy]() {
            if (vz_capture >= 0) {
                std::string lbl = "Invert depth axis (" + vz_name + ")";
                ImGui::MenuItem(lbl.c_str(), nullptr, &c.invert_depth);
            } else {
                ImGui::BeginDisabled();
                ImGui::MenuItem("Invert depth axis (2D system — N/A)", nullptr, false);
                ImGui::EndDisabled();
            }
            ImGui::Separator();
            draw_export_menu_item(fs_busy, cb, [&c](const std::string& p) {
                data_export::export_fastsync(c.result, p);
            });
        };
        v.render(*renderer, origin, plot_avail,
                 /*owner_id*/ (int)base_oid, gen_token,
                 series_in, vis, vis, fit);
        v.popup_extras = nullptr; // не утечь callback в другой кадр

        // ---- Colorbar справа — общая реализация (draw_colorbar) ----
        // Марджины плота обязаны совпадать с Plot2DView (он рисует внутри
        // plot_avail): раньше они были продублированы здесь тремя числами и
        // разъезжались при любой правке лэйаута. Теперь берутся из
        // plot_2d_margins(), т.е. один источник истины.
        float margin_left, margin_top, margin_right, margin_bottom;
        plot_2d_margins(margin_left, margin_top, margin_right, margin_bottom);
        const float plot_w = std::max(64.0f, plot_avail.x - margin_left - margin_right);
        const float plot_h = std::max(64.0f, plot_avail.y - margin_top  - margin_bottom);
        draw_colorbar(ImGui::GetWindowDrawList(),
                      ImVec2(origin.x + margin_left + plot_w + kColorbarGap,
                             origin.y + margin_top),
                      plot_h, (float)cmin, (float)cmax, cmap,
                      /*reverse*/ invert_cmap, /*n_discrete*/ 0, cb_ticks);
    }
    else {
        // Heatmap. `hv` — тот же view, на котором выше стоял тулбар; здесь
        // только доопределяем численный диапазон (cmin/cmax уже отсортированы
        // выше, HeatmapView обратный диапазон не ждёт) и подписи осей.
        HeatmapView& h = hv;
        h.manual_vmin = (float)cmin;
        h.manual_vmax = (float)cmax;
        h.x_axis.name = var_name(c.result.axis_x_var) + "(0)";
        h.y_axis.name = var_name(c.result.axis_y_var) + "(0)";
        const bool fs_busy = s.in_flight &&
                             s.active_config_index == s.running_config_index;
        h.popup_extras = [&c, &cb, fs_busy]() {
            draw_export_menu_item(fs_busy, cb, [&c](const std::string& p) {
                data_export::export_fastsync(c.result, p);
            });
        };
        h.render(*renderer, origin, avail,
                 /*owner_id*/ (int)base_oid, c.data_generation,
                 c.result.n_pts_grid, c.result.n_pts_grid,
                 c.result.heatmap.data(),
                 c.result.axis_x_lo, c.result.axis_x_hi,
                 c.result.axis_y_lo, c.result.axis_y_hi,
                 c.result.min_val, c.result.max_val,
                 fit);
    }
}

// ============================================================
// Parametric Controls dispatcher — верхние табы Bif / LLE / LS.
// ============================================================
static void draw_parametric_controls(AppModel& model, SystemLibrary& lib) {
    ImGui::Text("Parametric analysis");
    ImGui::TextDisabled("Per-thread parameter sweep via NVRTC + NonLinAnal kernels.");

    // Union used by the Run / Run all... buttons below to disable themselves
    // while any sub-analysis (Bif/LLE/LS) is computing. Kept after the tab's
    // System combo was moved to the top-bar; the top-bar combo has its own
    // wider union across all 7 sessions.
    bool any_in_flight = model.bifurcation_session.in_flight
                      || model.lle_session.in_flight
                      || model.ls_session.in_flight;

    // ----- Run (active sub-tab, active config) + batch Run all -----
    // Кнопка Run единая для Bif/LLE/LS, диспатчится по parametric_active_analysis
    // (0=Bif, 1=LLE, 2=LS) к соответствующему active_*_index. Run all — глобальный
    // по всем трём спискам: попап показывает чекбоксы для каждой диаграммы /
    // кривой / спектра, Run кладёт отмеченные в model.parametric_queue и
    // стартует первый; draw_gui тикает очередь после poll'ов.
    {
        const int kind = model.parametric_active_analysis;
        int  active_idx = -1;
        bool no_active  = false;
        if (kind == 0) {
            active_idx = model.bifurcation_session.active_diagram_index;
            no_active  = model.bifurcation_session.diagrams.empty();
        } else if (kind == 1) {
            active_idx = model.lle_session.active_curve_index;
            no_active  = model.lle_session.curves.empty();
        } else if (kind == 2) {
            active_idx = model.ls_session.active_curve_index;
            no_active  = model.ls_session.curves.empty();
        }

        std::vector<RunAllGroup> groups;
        groups.push_back(RunAllGroup{
            "Bifurcation", "pbd_", (int)model.bifurcation_session.diagrams.size(),
            [&model](int i) { return model.bifurcation_session.diagrams[i].label; },
            [&model](int i) {
                model.parametric_queue.push_back({ParametricQueueItem::Kind::Bifurcation, i});
            } });
        groups.push_back(RunAllGroup{
            "LLE", "plle_", (int)model.lle_session.curves.size(),
            [&model](int i) { return model.lle_session.curves[i].label; },
            [&model](int i) {
                model.parametric_queue.push_back({ParametricQueueItem::Kind::LLE, i});
            } });
        groups.push_back(RunAllGroup{
            "LS", "pls_", (int)model.ls_session.curves.size(),
            [&model](int i) { return model.ls_session.curves[i].label; },
            [&model](int i) {
                model.parametric_queue.push_back({ParametricQueueItem::Kind::LS, i});
            } });

        // Run all блокируется ТОЛЬКО на время расчёта: списков три, и пустой
        // активный не повод запрещать запуск остальных (в отличие от вкладок
        // с одним списком, где block_run_all включает и «нет конфигов»).
        draw_run_and_run_all("##run_all_parametric", any_in_flight, no_active,
                             any_in_flight, 140.0f,
                             [&model, kind, active_idx]() {
                                 if (active_idx < 0) return;
                                 if (!model.parametric_engine)
                                     model.parametric_engine = std::make_unique<ParametricEngine>();
                                 if      (kind == 0) model.bifurcation_session.run_async(*model.parametric_engine, active_idx);
                                 else if (kind == 1) model.lle_session.run_async(*model.parametric_engine, active_idx);
                                 else if (kind == 2) model.ls_session.run_async(*model.parametric_engine, active_idx);
                             },
                             {}, groups,
                             [&model]() { model.start_next_in_parametric_queue(); },
                             model.parametric_queue.size());
    }
    ImGui::Separator();

    if (ImGui::BeginTabBar("##parm_top")) {
        if (ImGui::BeginTabItem("Bifurcation")) {
            model.parametric_active_analysis = 0;
            draw_bifurcation_controls(model, lib);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("LLE")) {
            model.parametric_active_analysis = 1;
            draw_lle_controls(model, lib);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("LS")) {
            model.parametric_active_analysis = 2;
            draw_ls_controls(model, lib);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    // ----- Plot windows: dynamic list, mirrors Phase's Projections section -----
    // A sibling of the Bif/LLE/LS tabs above (not nested in any of them) —
    // always visible regardless of which analysis type tab is active, since
    // a plot window can be any (kind, dimension) combo. Each row has its own
    // Type combo (like Phase's projection type combo) so the chart type can
    // be changed freely after creation, instead of being fixed at "Add".
    ImGui::Separator();
    ImGui::SeparatorText("Plot windows");

    int win_to_remove = -1;
    for (int i = 0; i < (int)model.parametric_plot_windows.size(); ++i) {
        ParametricPlotWindow& win = model.parametric_plot_windows[i];
        ImGui::PushID(win.id);
        if (draw_parametric_window_setup_row(model, win)) win_to_remove = i;
        ImGui::PopID();
    }
    if (win_to_remove >= 0) model.remove_parametric_plot_window(win_to_remove);

    if (ImGui::Button("Add window")) {
        model.add_parametric_plot_window(ParametricPlotWindow::Kind::Bifurcation, false, {});
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset windows layout")) { model.parametric_layout_generation++; }
}

// ============================================================================
// Атомарное применение сохранённой сессии.
//
// Парсеры session_from_json_* пишут поля ПО МЕРЕ чтения JSON и бросают на
// первой же структурной ошибке. Поэтому на битом (обрезанном при падении,
// правленом руками) _last_*.json сессия оставалась перезаписанной наполовину:
// часть полей из файла, часть — сидированные из записи системы. Возвращаемый
// bool при этом игнорировался во ВСЕХ точках вызова, так что пользователь
// видел молча испорченные настройки.
//
// Здесь разбор идёт в два прохода: сначала в отдельный пустой объект — только
// чтобы убедиться, что файл дочитывается целиком, — и лишь потом в настоящую
// сессию. Второй проход не может упасть там, где прошёл первый: парсеры не
// зависят от прежнего состояния приёмника. Файлы сессий небольшие, двойной
// разбор незаметен на фоне загрузки системы.
//
// Пустой json — не ошибка: сессии просто нет (первый запуск / новая система).
// ============================================================================
template <class Session>
static bool apply_session_json(AppModel& model,
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

// Global system switch — fired from the top-bar combo. Loads the record and
// re-inits the CURRENT tab (mirrors what each per-tab combo used to do).
// Other tabs re-init on entry via the block in draw_gui.
// Non-static: also called from app_main.cpp on startup to restore the
// last-used system (see gui.h::apply_system_switch).
void apply_system_switch(AppModel& model, SystemLibrary& lib,
                         const std::string& name)
{
    try {
        model.from_record(lib.load(name));
        switch (model.app_mode) {
        case AppModel::AppMode::Analysis: {
            model.start_phase_analysis();
            std::string j = lib.load_session(model.loaded_name, "_last");
            apply_session_json(model, j, model.phase_session, session_from_json, "_last");
            break;
        }
        case AppModel::AppMode::Parametric: {
            model.start_parametric_analysis();
            std::string jb = lib.load_session(model.loaded_name, "_last_parametric");
            apply_session_json(model, jb, model.bifurcation_session, session_from_json_parametric, "_last_parametric");
            std::string jl = lib.load_session(model.loaded_name, "_last_lle");
            apply_session_json(model, jl, model.lle_session, session_from_json_lle, "_last_lle");
            std::string js = lib.load_session(model.loaded_name, "_last_ls");
            apply_session_json(model, js, model.ls_session, session_from_json_ls, "_last_ls");
            std::string jw = lib.load_session(model.loaded_name, "_last_parametric_windows");
            model.load_or_init_parametric_plot_windows(jw);
            break;
        }
        case AppModel::AppMode::Dft1D: {
            model.start_dft1d_analysis();
            std::string jd = lib.load_session(model.loaded_name, "_last_dft1d");
            apply_session_json(model, jd, model.dft1d_session, session_from_json_dft1d, "_last_dft1d");
            std::string jw = lib.load_session(model.loaded_name, "_last_dft1d_windows");
            model.load_or_init_dft1d_plot_windows(jw);
            break;
        }
        case AppModel::AppMode::Basins: {
            model.start_basins_analysis();
            std::string jb = lib.load_session(model.loaded_name, "_last_basins");
            apply_session_json(model, jb, model.basins_session, session_from_json_basins, "_last_basins");
            break;
        }
        case AppModel::AppMode::FastSync: {
            model.start_fastsync_analysis();
            std::string jf = lib.load_session(model.loaded_name, "_last_fastsync");
            apply_session_json(model, jf, model.fastsync_session, session_from_json_fastsync, "_last_fastsync");
            break;
        }
        case AppModel::AppMode::Custom: {
            // Hard reset + engine wipe. Sub-session buffers, signature caches,
            // AND the CUDA-side PTX cache in parametric_engine all need to
            // clear on system switch — otherwise the second-system Run either
            // produced a stale-looking result or zeroed out the first one.
            model.custom_session = CustomSession{};
            model.parametric_engine.reset();
            model.start_custom_analysis();
            std::string jc = lib.load_session(model.loaded_name, "_last_custom");
            apply_session_json(model, jc, model.custom_session, session_from_json_custom, "_last_custom");
            break;
        }
        case AppModel::AppMode::Library:
        case AppModel::AppMode::Settings:
        default:
            break;
        }
        // Persist the choice so the next launch restores this system. Read-
        // modify-write so we don't clobber unrelated fields (colormaps, UI
        // scale, etc.). Silent on failure — best-effort UX polish.
        AppConfig cfg;
        load_app_config(get_exe_dir_with_sep(), cfg);
        cfg.last_system_name = model.name;
        save_app_config(get_exe_dir_with_sep(), cfg);
    } catch (...) {}
}

// ============================================================
// Custom tab (master-detail pipeline) — see custom_session.h
// ============================================================

namespace {

// Комбо выбора цели свипа и разбор числовых полей — общие для всего файла
// (draw_sweep_target_combo / parse_ratio_or, см. верх файла). Раньше здесь
// лежали их локальные копии: комбо со своим стилем подписей ("IC x" вместо
// "x (IC)") и обёртка parse_num_default поверх того же парсера.

// ---- Shared config panel ----

void draw_shared_config(CustomSession& cs,
                        const std::vector<CustomScheme>& custom_schemes) {
    auto& c      = cs.shared;
    const auto& vars   = cs.vars;
    const auto& params = cs.params;
    auto& phase  = cs.phase_session;
    ImGui::SeparatorText("Shared config");

    // Scheme combo — mirrors draw_diagram_controls / draw_lle_controls layout
    // so users get the same familiar picker with built-ins + custom schemes.
    // On change, propagate to Phase (L3 shows the same picker; keep in sync)
    // and rebuild its KRS body so the next Phase Run uses the new integrator.
    // L1D/L2D pick up the new scheme via copy_integrator_and_state on Run.
    draw_scheme_combo("Scheme", c.scheme, custom_schemes,
        [&phase, &custom_schemes](const std::string& nm) {
            phase.scheme = nm;
            phase.regenerate_krs();
            // Custom КРС в Custom-вкладке считаются только на GPU.
            if (is_custom_scheme(nm, custom_schemes)) phase.use_gpu = true;
        });

    // Integration group — mirrors "Integration##bd_int" collapsing header in
    // draw_diagram_controls (per-line InputNumStr with comma→dot + ↑/↓).
    // Each edited field is mirrored into L1D's own override (`l1d_h_text` for
    // step) and into `phase_session`'s corresponding field, so the value the
    // user typed here shows up in the L1D and L3 panels without waiting for
    // Run. L1D still keeps independent transient/computing-time overrides —
    // step is intentionally the ONLY per-L1D field kept in sync with shared.
    if (ImGui::CollapsingHeader("Integration##custom_int", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (InputNumStr("h",              c.h_text, kFieldW)) {
            c.l1d_h_text  = c.h_text;
            phase.step_h  = c.h_text;
        }
        if (c.scheme == "CD" || custom_scheme_uses_symmetry(c.scheme, custom_schemes))
            if (InputNumStr("symmetry s", c.symmetry_s, kFieldW))
                phase.symmetry_s = c.symmetry_s;
        // TT before CT: transient runs first, computing-time is what's
        // actually sampled after — order matches conceptual flow.
        if (InputNumStr("transient time", c.transient_text, kFieldW))
            phase.skip_time = c.transient_text;
        if (InputNumStr("computing time", c.t_max_text, kFieldW))
            phase.sim_time  = c.t_max_text;
        if (InputNumStr("decimator",      c.pre_scaller_text, kFieldW))
            phase.decimation = c.pre_scaller_text;
        InputNumStr("max value",      c.max_value_text, kFieldW); // Phase has no analogue
    }

    // Initial conditions — one InputNumStr per line, matching draw_diagram_controls.
    // Not propagated to Phase: phase.ic_sets is multi-IC (mulistability) and
    // is edited in the L3 Phase panel; the shared IC block drives BD/LLE/LS/Basins.
    draw_named_num_fields("Initial conditions##custom_ic", vars, c.initial_conditions);

    // Parameters — same per-line layout; skip disable for params that are
    // swept on any enabled level (they get their values from the sweep).
    // Non-swept params are also mirrored to phase.param_values so the L3
    // Phase panel Parameters section shows the same live values.
    if (ImGui::CollapsingHeader("Parameters##custom_par", ImGuiTreeNodeFlags_DefaultOpen)) {
        // All params are always editable — sweep will overwrite swept-axis
        // values at run time anyway, but the manual value is useful for
        // initial-frame edits and for non-sweeping levels (e.g. L3 Phase).
        // We still MARK swept params with a "(swept)" tag so the user can
        // tell at a glance which values will be replaced by the sweep.
        std::vector<bool> is_swept(params.size(), false);
        const bool any_2d   = c.bif2d_enabled || c.lle2d_enabled || c.ls2d_enabled;
        const bool any_1d_x = c.bif1d_x_enabled || c.lle1d_x_enabled || c.ls1d_x_enabled;
        const bool any_1d_y = c.bif1d_y_enabled || c.lle1d_y_enabled || c.ls1d_y_enabled;
        auto mark = [&](int par_i, bool over_var) {
            if (over_var) return;
            if (par_i < 0 || par_i >= (int)params.size()) return;
            is_swept[par_i] = true;
        };
        if (c.level_2d_enabled && any_2d) {
            mark(c.axis_x_par_index, c.axis_x_over_var);
            mark(c.axis_y_par_index, c.axis_y_over_var);
        }
        if (c.level_1d_enabled) {
            if (any_1d_x) {
                EffectiveSweep esx = effective_sweep_x(c);
                mark(esx.par_index, esx.over_var);
            }
            if (any_1d_y) {
                EffectiveSweep esy = effective_sweep_y(c);
                mark(esy.par_index, esy.over_var);
            }
        }
        // Effective sweep axes cached once — used by both the (swept) label
        // logic below AND the crosshair sync when the user edits the value
        // of a sweep-axis param manually.
        EffectiveSweep esx = effective_sweep_x(c);
        EffectiveSweep esy = effective_sweep_y(c);
        for (int i = 0; i < (int)params.size(); ++i) {
            const auto& p = params[i];
            ImGui::PushID(p.c_str());
            if (InputNumStr(p.c_str(), c.param_values[p], kFieldW)) {
                phase.param_values[p] = c.param_values[p];
                // Bump debounce timers so the auto-recompute path (L1D
                // partial + Phase, gated by auto_recompute_1d /
                // autorun_on_drilldown) picks up the manual edit — same
                // path a slider drag settle goes through, no Run needed.
                double t = ImGui::GetTime();
                c.last_fix_x_change_time = t;
                c.last_fix_y_change_time = t;
                // If this param IS the effective sweep axis, move the
                // crosshair to the new pinned value so 1D slice plots and
                // 2D heatmaps show it there immediately.
                double new_val = parse_ratio_or(c.param_values[p], 0.0);
                if (!esx.over_var && esx.par_index == i) c.fix_x_value = new_val;
                if (!esy.over_var && esy.par_index == i) c.fix_y_value = new_val;
            }
            if (is_swept[i]) {
                ImGui::SameLine();
                ImGui::TextDisabled("(swept)");
            }
            ImGui::PopID();
        }
    }
}

// ---- Level 2D detail ----

void draw_level2d_detail(CustomSession& cs) {
    auto& c = cs.shared;
    // Header title carries the level name — no SeparatorText here.

    ImGui::Checkbox("Bif",  &c.bif2d_enabled); ImGui::SameLine();
    ImGui::Checkbox("LLE",  &c.lle2d_enabled); ImGui::SameLine();
    ImGui::Checkbox("LS",   &c.ls2d_enabled);

    // Axis X — combo (sweep target) + lo/hi/N on separate lines so the
    // per-field digit-step (↑/↓) and comma→dot filter work reliably (they
    // don't when InputText's are packed onto one SameLine chain).
    ImGui::TextUnformatted("Axis X:"); ImGui::SameLine();
    draw_sweep_target_combo("##ax", cs.params, cs.vars,
                            c.axis_x_par_index, c.axis_x_over_var, c.axis_x_var_index,
                            c.axis_x_over_h, &c.axis_y_over_h,
                            /*note_when_empty*/ false, 120.0f);
    InputNumStr("lo##ax", c.axis_x_lo_text, kFieldW);
    InputNumStr("hi##ax", c.axis_x_hi_text, kFieldW);
    // Log-сетка по оси. Свойство оси, поэтому X-срез Level 1D наследует его
    // вместе с par/lo/hi (см. EffectiveSweep::log_scale).
    ImGui::Checkbox("Log scale##ax_log", &c.axis_x_log);
    if (c.axis_x_log) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }

    ImGui::TextUnformatted("Axis Y:"); ImGui::SameLine();
    draw_sweep_target_combo("##ay", cs.params, cs.vars,
                            c.axis_y_par_index, c.axis_y_over_var, c.axis_y_var_index,
                            c.axis_y_over_h, &c.axis_x_over_h,
                            /*note_when_empty*/ false, 120.0f);
    InputNumStr("lo##ay", c.axis_y_lo_text, kFieldW);
    InputNumStr("hi##ay", c.axis_y_hi_text, kFieldW);
    ImGui::Checkbox("Log scale##ay_log", &c.axis_y_log);
    if (c.axis_y_log) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }

    // Shared N×N grid resolution — kernel `getValueByIdx` requires a square
    // grid, so one field drives both axes (matches Analysis tab).
    InputNumStr("Resolution##a", c.resolution_text, kFieldW);

    // Переменная БД — задаётся здесь и наследуется обоими 1D-срезами (см.
    // CustomTabSharedConfig::bif_writable_var). Рисуется независимо от
    // bif2d_enabled: при выключенном Bif-2D её всё равно читает Level 1D.
    // Комбинация (-1) работает и в 2D, и в 1D — одно и то же ядро.
    ImGui::Separator();
    draw_writable_var_combo(cs.vars, c.bif_writable_var, "Bif variable##custom_wv");

    // Per-type options edited directly on the sub-session's slot [0] (2D config).
    ImGui::Separator();
    if (c.bif2d_enabled && !cs.bif_session.diagrams.empty()) {
        InputNumStr("Bif DBSCAN eps", cs.bif_session.diagrams[0].eps_dbscan_text, kFieldW);
        // Множители осей кластеризации — см. draw_diagram_controls.
        InputNumStr("Bif mult peak",     cs.bif_session.diagrams[0].mult_peak_text, kFieldW);
        InputNumStr("Bif mult interval", cs.bif_session.diagrams[0].mult_interval_text, kFieldW);
    }
    // eps/NT у LLE и LS живут в слоте [0] и наследуются 1D-срезами (см.
    // apply_shared_to_lle1d / _ls1d), поэтому поля показываем и когда сама
    // карта выключена, но включён хотя бы один её срез — иначе их негде править.
    if ((c.lle2d_enabled || c.lle1d_x_enabled || c.lle1d_y_enabled) &&
        !cs.lle_session.curves.empty()) {
        InputNumStr("LLE eps", cs.lle_session.curves[0].eps_text, kFieldW);
        InputNumStr("LLE NT",  cs.lle_session.curves[0].nt_text, kFieldW);
    }
    if ((c.ls2d_enabled || c.ls1d_x_enabled || c.ls1d_y_enabled) &&
        !cs.ls_session.curves.empty()) {
        InputNumStr("LS eps", cs.ls_session.curves[0].eps_text, kFieldW);
        InputNumStr("LS NT",  cs.ls_session.curves[0].nt_text, kFieldW);
    }
    ImGui::Separator();

    // Run buttons live on the top row of draw_custom_controls — this panel
    // is edit-only.
}

// ---- Level 1D detail ----

void draw_level1d_detail(CustomSession& cs) {
    auto& c = cs.shared;
    // Header title carries the level name — no SeparatorText here.

    if (!c.level_2d_enabled) ImGui::BeginDisabled();
    ImGui::Checkbox("Inherit sweep from Level 2D", &c.inherit_sweep_from_2d);
    if (!c.level_2d_enabled) {
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::TextDisabled("(Level 2D off - L1D uses its own sweep)");
    }

    const bool inheriting = c.inherit_sweep_from_2d && c.level_2d_enabled;
    EffectiveSweep sx = effective_sweep_x(c);
    EffectiveSweep sy = effective_sweep_y(c);

    // Inherit disables ONLY the sweep axis (par target + lo/hi + log), leaving
    // N and the L1D-specific integrator fields (h/TT/CT below) always
    // editable — L1D is cheap and interactive, so users may want a finer
    // grid or a longer transient than L2D even when sharing the same axis.
    //
    // В inherit-режиме виджеты кормим КОПИЯМИ эффективного свипа: править их
    // всё равно нельзя (BeginDisabled), зато в панели видно то, что реально
    // пойдёт в расчёт. Раньше показывались собственные sweep_*, которые в этом
    // режиме не используются, — то есть панель врала об осях среза.
    auto sweep_row = [&](const char* label, const char* tag,
                         const EffectiveSweep& eff,
                         int& own_par, bool& own_ov, int& own_vi,
                         bool& own_oh, bool& own_log,
                         std::string& own_lo, std::string& own_hi,
                         bool* other_oh, std::string& n_text) {
        int         par = inheriting ? eff.par_index : own_par;
        bool        ov  = inheriting ? eff.over_var  : own_ov;
        int         vi  = inheriting ? eff.var_index : own_vi;
        bool        oh  = inheriting ? eff.over_h    : own_oh;
        bool        lg  = inheriting ? eff.log_scale : own_log;
        std::string lo  = inheriting ? eff.lo_text   : own_lo;
        std::string hi  = inheriting ? eff.hi_text   : own_hi;
        bool        other_copy = other_oh ? *other_oh : false;

        ImGui::TextUnformatted(label); ImGui::SameLine();
        if (inheriting) ImGui::BeginDisabled();
        draw_sweep_target_combo((std::string("##") + tag + "p").c_str(),
                                cs.params, cs.vars, par, ov, vi, oh,
                                inheriting ? &other_copy : other_oh,
                                /*note_when_empty*/ false, 120.0f);
        InputNumStr((std::string("lo##") + tag).c_str(), lo, kFieldW);
        InputNumStr((std::string("hi##") + tag).c_str(), hi, kFieldW);
        ImGui::Checkbox((std::string("Log scale##") + tag + "log").c_str(), &lg);
        if (lg) {
            ImGui::SameLine();
            ImGui::TextDisabled(inheriting ? "(from Level 2D)" : "(lo/hi > 0)");
        }
        if (inheriting) ImGui::EndDisabled();
        InputNumStr((std::string("N##") + tag).c_str(), n_text, kFieldW);

        // Обратная запись только в своём режиме — в inherit-режиме источник
        // истины остаётся за осями Level 2D.
        if (!inheriting) {
            own_par = par; own_ov  = ov; own_vi = vi;
            own_oh  = oh;  own_log = lg;
            own_lo  = lo;  own_hi  = hi;
        }
    };

    sweep_row("X-sweep:", "sx", sx,
              c.sweep_x_par_index, c.sweep_x_over_var, c.sweep_x_var_index,
              c.sweep_x_over_h, c.sweep_x_log,
              c.sweep_x_lo_text, c.sweep_x_hi_text,
              &c.sweep_y_over_h, c.n_x_1d_text);

    sweep_row("Y-sweep:", "sy", sy,
              c.sweep_y_par_index, c.sweep_y_over_var, c.sweep_y_var_index,
              c.sweep_y_over_h, c.sweep_y_log,
              c.sweep_y_lo_text, c.sweep_y_hi_text,
              &c.sweep_x_over_h, c.n_y_1d_text);

    // Строки выше могли поменять собственный свип L1D — перечитываем, чтобы
    // ползунки ниже снапились по сетке ЭТОГО кадра, а не прошлого.
    sx = effective_sweep_x(c);
    sy = effective_sweep_y(c);

    // L1D-specific integrator overrides — order TT before CT (per user
    // convention: transient runs first, then computing time is what's
    // actually sampled).
    ImGui::Separator();
    ImGui::TextUnformatted("L1D integrator (independent of shared):");
    InputNumStr("h##l1d",              c.l1d_h_text, kFieldW);
    InputNumStr("transient time##l1d", c.l1d_transient_text, kFieldW);
    InputNumStr("computing time##l1d", c.l1d_t_max_text, kFieldW);

    ImGui::Separator();
    // Slice sliders — clamped to the current effective ranges. Values snap
    // to the FINER of the two grids per axis: L2D `resolution_text` (N×N
    // pixels of the heatmap) and L1D `n_{x,y}_1d_text` (samples of the
    // corresponding slice). When the user cranks 1D resolution up (typical
    // — L1D is cheap, 2D is expensive), the slider gets finer too, so the
    // crosshair on the X-slice plot lands on X-slice data points instead
    // of drifting between them; on the heatmap the crosshair may then sit
    // between pixel columns, an acceptable trade-off.
    // SliderScalar<Double> stays in double throughout (SliderFloat would
    // downcast to float and land 0.2 as 0.20000000298023224). Drag only
    // moves the crosshair; recompute fires on IsItemDeactivatedAfterEdit.
    double fx_lo = parse_ratio_or(sx.lo_text, 0.0);
    double fx_hi = parse_ratio_or(sx.hi_text, 1.0);
    double fy_lo = parse_ratio_or(sy.lo_text, 0.0);
    double fy_hi = parse_ratio_or(sy.hi_text, 1.0);
    if (fx_hi < fx_lo) std::swap(fx_lo, fx_hi);
    if (fy_hi < fy_lo) std::swap(fy_lo, fy_hi);
    // Discrete slider via index-value trick: SliderScalar<Int> steps by 1
    // (thumb snaps hard, no continuous drift), value = grid index. Format
    // string is a LITERAL world-coord string (no % specifier), so the
    // bubble shows "0.158730" instead of "5". Bubble text is precomputed
    // from the CURRENT idx before SliderScalar runs — during drag it
    // shows the previous frame's snapped world value (1-frame lag on the
    // text only, but the thumb itself always sits on a grid node).
    int n_2d   = parse_int_or(c.resolution_text, 64); if (n_2d   < 2) n_2d   = 64;
    int n_1d_x = parse_int_or(c.n_x_1d_text,     64); if (n_1d_x < 2) n_1d_x = 64;
    int n_1d_y = parse_int_or(c.n_y_1d_text,     64); if (n_1d_y < 2) n_1d_y = 64;
    int n_snap_x = (n_1d_x > n_2d) ? n_1d_x : n_2d;
    int n_snap_y = (n_1d_y > n_2d) ? n_1d_y : n_2d;
    // При log-свипе узлы сетки распределены лог-равномерно (см.
    // sweep_value_at / getValueByIdx_log), поэтому и шаг ползунка обязан идти
    // по логарифму — иначе thumb «прилипает» к линейным позициям, которых в
    // данных нет, и крестик на графике уезжает с узла.
    // log при lo<=0 невалиден (движок такой Run отклонит) — деградируем на
    // линейную сетку, чтобы до Run ползунок не выдавал NaN.
    auto log_ok = [](bool log_scale, double lo, double hi) {
        return log_scale && lo > 0.0 && hi > 0.0;
    };
    auto idx_from_world = [&](double v, double lo, double hi, int n, bool log_scale) {
        if (n < 2 || hi <= lo) return 0;
        double i_d;
        if (log_ok(log_scale, lo, hi)) {
            if (!(v > 0.0)) return 0;
            const double l0 = std::log10(lo), l1 = std::log10(hi);
            i_d = (std::log10(v) - l0) / ((l1 - l0) / (double)(n - 1));
        } else {
            i_d = (v - lo) / ((hi - lo) / (double)(n - 1));
        }
        return std::clamp((int)std::round(i_d), 0, n - 1);
    };
    auto world_from_idx = [&](int i, double lo, double hi, int n, bool log_scale) {
        if (n < 2 || hi <= lo) return lo;
        return sweep_value_at(i, n, lo, hi, log_ok(log_scale, lo, hi),
                              /*reverse*/ false, /*continuation*/ false);
    };

    // Step-arrows + slider row. Arrows walk idx by ±1 (repeat on hold),
    // slider is the discrete int-index widget from above. Same pattern
    // for X and Y — factored into a lambda.
    auto arrow_row = [&](const char* id,
                         int& idx, int idx_min, int idx_max,
                         double lo, double hi, int n_snap, bool log_scale,
                         double& fix_value,
                         double& timer_to_bump,
                         const char* slider_label) {
        ImGui::PushID(id);
        ImGui::PushButtonRepeat(true);
        bool step_changed = false;
        if (ImGui::ArrowButton("l", ImGuiDir_Left)) {
            if (idx > idx_min) { --idx; step_changed = true; }
        }
        ImGui::SameLine(0.0f, 2.0f);
        if (ImGui::ArrowButton("r", ImGuiDir_Right)) {
            if (idx < idx_max) { ++idx; step_changed = true; }
        }
        ImGui::PopButtonRepeat();
        ImGui::PopID();
        ImGui::SameLine();
        char fmt[64];
        std::snprintf(fmt, sizeof(fmt), "%.6g", world_from_idx(idx, lo, hi, n_snap, log_scale));
        ImGui::SetNextItemWidth(240.0f);
        ImGui::SliderScalar(slider_label, ImGuiDataType_S32, &idx,
                            &idx_min, &idx_max, fmt);
        bool released     = ImGui::IsItemDeactivatedAfterEdit();
        bool slider_edit  = ImGui::IsItemActive() || released;
        // Only overwrite fix_value on real user interaction. Otherwise the
        // idx_from_world → world_from_idx round-trip would re-snap a
        // fix_value that arrived from the heatmap (on the coarser 2D grid)
        // to the nearest slider-grid node (max(2D, 1D) — potentially the
        // finer 1D grid), silently drifting it off the 2D pixel the user
        // just clicked.
        if (slider_edit || step_changed)
            fix_value = world_from_idx(idx, lo, hi, n_snap, log_scale);
        // Arrow clicks fire the same debounce path as slider release —
        // step-changed → immediate commit, no need to wait for release.
        // Bump only the caller-supplied timer so the settled branch can
        // enqueue just the slices that actually depend on this axis.
        if (released || step_changed)
            timer_to_bump = ImGui::GetTime();
    };

    int idx_x = idx_from_world(c.fix_x_value, fx_lo, fx_hi, n_snap_x, sx.log_scale);
    arrow_row("##fix_x_arr", idx_x, 0, n_snap_x - 1,
              fx_lo, fx_hi, n_snap_x, sx.log_scale,
              c.fix_x_value, c.last_fix_x_change_time, "fix X");

    int idx_y = idx_from_world(c.fix_y_value, fy_lo, fy_hi, n_snap_y, sy.log_scale);
    arrow_row("##fix_y_arr", idx_y, 0, n_snap_y - 1,
              fy_lo, fy_hi, n_snap_y, sy.log_scale,
              c.fix_y_value, c.last_fix_y_change_time, "fix Y");

    ImGui::Separator();
    ImGui::TextUnformatted("Enable slices:");
    ImGui::Checkbox("Bif-X", &c.bif1d_x_enabled); ImGui::SameLine();
    ImGui::Checkbox("Bif-Y", &c.bif1d_y_enabled); ImGui::SameLine();
    ImGui::Checkbox("LLE-X", &c.lle1d_x_enabled); ImGui::SameLine();
    ImGui::Checkbox("LLE-Y", &c.lle1d_y_enabled); ImGui::SameLine();
    ImGui::Checkbox("LS-X",  &c.ls1d_x_enabled);  ImGui::SameLine();
    ImGui::Checkbox("LS-Y",  &c.ls1d_y_enabled);

    ImGui::Separator();
    // Переменная БД — read-only эхо настройки Level 2D (срез строится по той
    // же переменной, что и карта). Меняется в панели Level 2D.
    {
        const char* wv_name =
            (c.bif_writable_var == -1) ? "combination"
          : (c.bif_writable_var >= 0 && c.bif_writable_var < (int)cs.vars.size())
                ? cs.vars[c.bif_writable_var].c_str() : "?";
        ImGui::TextDisabled("Bif variable: %s (from Level 2D)", wv_name);
    }
    // Так же и eps/NT для LLE/LS-срезов: read-only эхо слота 2D, правятся в
    // панели Level 2D (см. apply_shared_to_lle1d — срез считается с ними же).
    if ((c.lle1d_x_enabled || c.lle1d_y_enabled) && !cs.lle_session.curves.empty()) {
        ImGui::TextDisabled("LLE eps / NT: %s / %s (from Level 2D)",
                            cs.lle_session.curves[0].eps_text.c_str(),
                            cs.lle_session.curves[0].nt_text.c_str());
    }
    if ((c.ls1d_x_enabled || c.ls1d_y_enabled) && !cs.ls_session.curves.empty()) {
        ImGui::TextDisabled("LS eps / NT: %s / %s (from Level 2D)",
                            cs.ls_session.curves[0].eps_text.c_str(),
                            cs.ls_session.curves[0].nt_text.c_str());
    }
    // Отображение Y на Bif-срезах. Пересчёт не нужен — peak_times и
    // bifurcation_points приходят из одного прогона, поэтому пишем флаг прямо
    // в слоты [1]/[2] и просим автофит; Run не требуется.
    if (ImGui::Checkbox("Plot inter-peaks instead of peak values",
                        &c.plot_inter_peaks_1d)) {
        for (int slot = 1; slot <= 2; ++slot) {
            if ((int)cs.bif_session.diagrams.size() <= slot) break;
            cs.bif_session.diagrams[slot].plot_inter_peaks = c.plot_inter_peaks_1d;
            cs.bif_session.diagrams[slot].fit_request      = true;
        }
        cs.workspace.dirty = true;   // → пересохранение _last_custom.json
    }

    ImGui::Separator();
    ImGui::Checkbox("Continuation (1D)", &c.continuation_1d_enabled);
    ImGui::SameLine();
    ImGui::Checkbox("Auto-recompute 1D on slider/sweep change", &c.auto_recompute_1d);
}

// ---- Level 3 detail ----

void draw_level3_detail(CustomSession& cs) {
    auto& c = cs.shared;
    // Header title carries the level name — no SeparatorText here.
    ImGui::RadioButton("Phase + Time-domain", &c.level3_kind, 0); ImGui::SameLine();
    ImGui::RadioButton("Basins",              &c.level3_kind, 1);

    ImGui::Separator();
    ImGui::Checkbox("Auto-run", &c.autorun_on_drilldown);

    ImGui::Separator();
    if (c.level3_kind == 0) {
        // Phase: full IC-sets / integrator / projections UI is embedded
        // right here, so everything lives inside the pipeline's L3 config
        // (no separate top-level window). The Analysis-tab uses the same
        // helper with model.phase_session — here we pass Custom's own
        // isolated cs.phase_session.
        draw_phase_controls(cs.phase_session, nullptr);
    } else {
        // Basins sub-panel: IC-space axes + features. All of scheme/h/t_max/
        // etc. come from shared; DBSCAN eps stays per-config.
        if (cs.basins_session.configs.empty()) {
            ImGui::TextDisabled("Basins config not initialised.");
        } else {
            auto& bc = cs.basins_session.configs[0];
            ImGui::TextUnformatted("IC-space X var:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::BeginCombo("##bxv",
                bc.axis_x_var >= 0 && bc.axis_x_var < (int)cs.vars.size()
                    ? cs.vars[bc.axis_x_var].c_str() : "var")) {
                for (int i = 0; i < (int)cs.vars.size(); ++i)
                    if (ImGui::Selectable(cs.vars[i].c_str(), bc.axis_x_var == i))
                        bc.axis_x_var = i;
                ImGui::EndCombo();
            }
            InputNumStr("lo##bx", bc.axis_x_lo_text, kFieldW);
            InputNumStr("hi##bx", bc.axis_x_hi_text, kFieldW);

            ImGui::TextUnformatted("IC-space Y var:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::BeginCombo("##byv",
                bc.axis_y_var >= 0 && bc.axis_y_var < (int)cs.vars.size()
                    ? cs.vars[bc.axis_y_var].c_str() : "var")) {
                for (int i = 0; i < (int)cs.vars.size(); ++i)
                    if (ImGui::Selectable(cs.vars[i].c_str(), bc.axis_y_var == i))
                        bc.axis_y_var = i;
                ImGui::EndCombo();
            }
            InputNumStr("lo##by", bc.axis_y_lo_text, kFieldW);
            InputNumStr("hi##by", bc.axis_y_hi_text, kFieldW);
            InputNumStr("N##bxy", bc.n_pts_text, kFieldW);

            // Переменная, чью траекторию читает feature-экстрактор. Своя, а не
            // от Level 2D: у Basins другая задача (оси — IC-пространство), и в
            // отдельной вкладке Basins этот выбор тоже свой (см. gui.cpp
            // draw_basins_controls). Без этого комбо в Custom всегда шла
            // первая переменная.
            draw_writable_var_combo(cs.vars, bc.writable_var, "Writable var##custom_bas_wv");

            ImGui::InputInt("feature1", &bc.feature1);
            ImGui::InputInt("feature2", &bc.feature2);
            InputNumStr("DBSCAN eps", bc.eps_dbscan_text, kFieldW);
        }
    }
}

// ---- Pipeline column (master) ----

// draw_pipeline_column / level_status_badge жили здесь и не вызывались ни
// разу: колонка со списком уровней была заменена на level_header внутри
// draw_custom_controls, а вся её логика статусов переписана там заново.
// Удалены как мёртвый код.

} // namespace (draw helpers)

static void draw_custom_controls(AppModel& model, SystemLibrary& lib) {
    (void)lib;
    auto& cs = model.custom_session;
    auto& c  = cs.shared;

    // Top row: Run + Stop + status. Also bound to Ctrl+R (same shortcut
    // Phase controls / Bif controls use for their local Run). Dirty-tracked:
    // build each level's signature, compare with the last committed one; if
    // nothing changed for a level AND its last run succeeded, skip enqueue.
    // Newly-enqueued signatures land in `pending_armed`; a successful poll
    // then promotes pending → committed (see poll_and_commit below).
    bool run_now = ImGui::Button("Run");
    if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_R, false))
        run_now = true;
    if (run_now) {
        auto try_enqueue = [&](CustomSession::LevelSig& sig,
                               const std::string& current,
                               std::function<void()> do_enqueue) {
            if (current == sig.committed && sig.committed_ok) return;
            do_enqueue();
            sig.pending = current;
            sig.pending_armed = true;
        };
        try_enqueue(cs.sig_l2d, build_l2d_signature(c, cs),
                    [&](){ cs.enqueue_level_2d(model.custom_queue); });
        try_enqueue(cs.sig_l1d, build_l1d_signature(c, cs),
                    [&](){ cs.enqueue_level_1d(model.custom_queue); });
        try_enqueue(cs.sig_l3,  build_l3_signature (c, cs),
                    [&](){ cs.enqueue_level_3 (model.custom_queue); });
    }
    ImGui::SameLine();
    if (ImGui::Button("Stop / clear queue")) {
        model.custom_session.request_cancel_all();
        model.custom_queue.clear();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(%zu queued)", model.custom_queue.size());

    // Auto-recompute 1D on slider settle, plus Phase (if L3=Phase and
    // autorun_on_drilldown is on) so drag-releasing the crosshair or the
    // slider gives immediate feedback. Debounce 200 ms + no in-flight.
    // Per-axis timers (last_fix_{x,y}_change_time) let us enqueue ONLY the
    // slices that actually depend on the moved axis — fix_x drag re-runs
    // Y-slices (they pin X at fix_x), fix_y drag re-runs X-slices. Without
    // the split the untouched slice would re-run with identical data and
    // trigger an autofit that resets the user's manual zoom. Heatmap drag
    // bumps BOTH timers, so it still recomputes both sides.
    if (c.level_1d_enabled && !cs.any_in_flight()) {
        double now = ImGui::GetTime();
        // 500 ms debounce: long enough that stepping a numeric field with
        // the arrow buttons (multiple ticks in quick succession) coalesces
        // into a single recompute instead of firing on every click.
        const double debounce_s = kCustomSliderDebounceSec;
        bool x_settled = c.last_fix_x_change_time > 0.0
                      && now - c.last_fix_x_change_time > debounce_s;
        bool y_settled = c.last_fix_y_change_time > 0.0
                      && now - c.last_fix_y_change_time > debounce_s;
        bool queue_free = model.custom_queue.empty();
        if ((x_settled || y_settled) && queue_free) {
            if (c.auto_recompute_1d) {
                // fix_x moved → Y-slices depend on fix_x. fix_y → X-slices.
                cs.enqueue_level_1d_partial(model.custom_queue,
                    /*x_slices*/ y_settled,
                    /*y_slices*/ x_settled);
            }
            // Pin the shared param_values from the EFFECTIVE sweep axes.
            // Previously this was inside `if (autorun_on_drilldown && ...)`
            // AND always used the L2D axes — so a 1D-standalone sweep
            // (inherit off) never propagated a slider drag back to the
            // shared Parameters section. Now we pin unconditionally on
            // settle, using effective_sweep_x/y so L2D-inherit and L1D-
            // standalone both update the correct param.
            // Ось может свипаться по параметру, по НУ или по шагу h. Здесь
            // отражаем в общие поля только param-случай (НУ и h подставляются
            // в под-конфиги перед самим Run — см. pin_fixed_* в app_model.cpp);
            // для h дополнительно обновляем общий h_text, чтобы панель
            // показывала тот шаг, с которым реально пойдёт расчёт.
            // Формат — round-trip (fmt_num_shortest), а НЕ %.6g: v это значение
            // узла сетки, посчитанное ucuda_node_value, и оно уходит в ядро
            // через parse_num. Шесть цифр отрезали ~10 знаков, и drill-down
            // считался в параметре, в котором ячейка не считалась.
            auto pin_axis = [&](const EffectiveSweep& e, double v) {
                if (e.over_h) {
                    if (v > 0.0) c.l1d_h_text = fmt_num_shortest(v);
                    return;
                }
                if (e.over_var) return;
                if (e.par_index < 0 || e.par_index >= (int)cs.params.size()) return;
                c.param_values[cs.params[e.par_index]] = fmt_num_shortest(v);
            };
            EffectiveSweep esx = effective_sweep_x(c);
            EffectiveSweep esy = effective_sweep_y(c);
            if (x_settled) pin_axis(esx, c.fix_x_value);
            if (y_settled) pin_axis(esy, c.fix_y_value);
            if (c.autorun_on_drilldown && c.level_phase_enabled) {
                model.custom_queue.push_back({ c.level3_kind == 0
                    ? CustomQueueItem::Kind::Phase
                    : CustomQueueItem::Kind::Basins });
            }
            if (x_settled) c.last_fix_x_change_time = 0.0;
            if (y_settled) c.last_fix_y_change_time = 0.0;
        }
    }

    // Shared config panel — always visible at the top.
    draw_shared_config(cs, cs.custom_schemes);

    ImGui::Separator();

    // Each level = enable-checkbox + CollapsingHeader + status label,
    // headers stay open/closed as the user last left them (SetNextItemOpen
    // uses FirstUseEver, so the first-ever appearance opens L2D and closes
    // the others; later, the ImGui-managed state wins). Multiple headers
    // can be open at once — no forced current-level.
    // Level colour ladder: base = theme's Header colour, HUE rotated by
    // +45°/level so the three markers land on visually distinct points of
    // the colour wheel (default dark theme's blue → violet → magenta).
    // Saturation was previously index-tied — the three shades read as "the
    // same colour, just fading", so telling levels apart at a glance was
    // hard. Rotating hue keeps the theme-derived feel but each level is a
    // recognisably different colour. Both saturation and value are pinned
    // at the base's values so all three stay on the theme's "brightness".
    // When a user-picked primary colour lands later, swap `base` and the
    // rotation cascades automatically.
    auto level_marker_color = [](int level_idx) {
        const ImVec4 base = ImGui::GetStyleColorVec4(ImGuiCol_Header);
        float h, s, v;
        ImGui::ColorConvertRGBtoHSV(base.x, base.y, base.z, h, s, v);
        const int   idx      = (level_idx >= 0 && level_idx < 3) ? level_idx : 0;
        const float hue_step = 45.0f / 360.0f;   // ImGui uses 0..1 for hue.
        float nh = h + hue_step * (float)idx;
        nh -= std::floor(nh);                    // wrap into [0, 1).
        // Lift saturation to full so the markers pop against ambient body
        // background (base Header colour is quite pale in default themes).
        const float ns = std::max(0.75f, s);
        const float nv = std::max(0.85f, v);
        ImVec4 col(0, 0, 0, 1.0f);
        ImGui::ColorConvertHSVtoRGB(nh, ns, nv, col.x, col.y, col.z);
        return col;
    };

    auto level_header = [&](int level_idx,
                            const char* enable_id,
                            bool& enabled,
                            const char* title,
                            bool running, bool has_result,
                            bool default_open,
                            const std::function<void()>& body) {
        ImGui::Checkbox(enable_id, &enabled);
        ImGui::SameLine();
        // Tinted status suffix on the header title itself so it's visible
        // even when the header is collapsed.
        const char* status =
            !enabled ? "off" :
            running  ? "running" :
            has_result ? "ok" : "idle";
        ImVec4 status_col(0.6f, 0.6f, 0.6f, 1.0f);
        if      (!std::strcmp(status, "running")) status_col = ImVec4(1.0f, 0.85f, 0.35f, 1.0f);
        else if (!std::strcmp(status, "ok"))      status_col = ImVec4(0.5f, 0.9f,  0.5f,  1.0f);
        else if (!std::strcmp(status, "off"))     status_col = ImVec4(0.5f, 0.5f,  0.5f,  1.0f);
        char header_label[128];
        std::snprintf(header_label, sizeof(header_label), "%s##hdr_%s", title, enable_id);
        ImGui::SetNextItemOpen(default_open, ImGuiCond_FirstUseEver);
        bool open = ImGui::CollapsingHeader(header_label);
        // Status label on the SAME line as the header (rendered AFTER the
        // header so it sits inline).
        ImGui::SameLine();
        ImGui::TextColored(status_col, "[%s]", status);
        if (open) {
            // Colored left-side bar spanning the body — always in peripheral
            // vision, so whichever setting you're looking at, the coloured
            // stripe on the left tells you which level it belongs to. Bar
            // is drawn AFTER the body so its height matches actual content.
            const float bar_w   = 2.5f;         // 1.5× thinner than before.
            const float bar_gap = 8.0f;         // px between bar and content.
            const float indent  = bar_w + bar_gap;
            const ImVec2 top_screen = ImGui::GetCursorScreenPos();
            ImGui::Indent(indent);
            body();
            ImGui::Unindent(indent);
            const ImVec2 bot_screen = ImGui::GetCursorScreenPos();
            // Pull bar bottom up by ItemSpacing so it doesn't run into the
            // next level's checkbox row (cursor after body sits at the row
            // start of what comes next).
            const float  bar_bot_y  = bot_screen.y - ImGui::GetStyle().ItemSpacing.y;
            const ImU32  col        = ImGui::ColorConvertFloat4ToU32(
                                          level_marker_color(level_idx));
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(top_screen.x,          top_screen.y),
                ImVec2(top_screen.x + bar_w,  bar_bot_y),
                col, 1.5f);
        }
    };

    // Derive status flags (same logic that used to live in draw_pipeline_column).
    bool level2_running =
        (cs.bif_session.in_flight && cs.bif_session.running_diagram_index == 0) ||
        (cs.lle_session.in_flight && cs.lle_session.running_curve_index   == 0) ||
        (cs.ls_session.in_flight  && cs.ls_session.running_curve_index    == 0);
    bool level2_hasres  =
        (!cs.bif_session.diagrams.empty() && cs.bif_session.diagrams[0].last_run_2d_ok) ||
        (!cs.lle_session.curves.empty()   && cs.lle_session.curves[0].last_run_2d_ok)   ||
        (!cs.ls_session.curves.empty()    && cs.ls_session.curves[0].last_run_2d_ok);
    bool level1_running =
        (cs.bif_session.in_flight && cs.bif_session.running_diagram_index > 0) ||
        (cs.lle_session.in_flight && cs.lle_session.running_curve_index   > 0) ||
        (cs.ls_session.in_flight  && cs.ls_session.running_curve_index    > 0);
    bool level1_hasres = false;
    if (cs.bif_session.diagrams.size() > 1 && cs.bif_session.diagrams[1].last_run_ok) level1_hasres = true;
    if (cs.bif_session.diagrams.size() > 2 && cs.bif_session.diagrams[2].last_run_ok) level1_hasres = true;
    if (cs.lle_session.curves.size()   > 1 && cs.lle_session.curves[1].last_run_ok)   level1_hasres = true;
    if (cs.lle_session.curves.size()   > 2 && cs.lle_session.curves[2].last_run_ok)   level1_hasres = true;
    if (cs.ls_session.curves.size()    > 1 && cs.ls_session.curves[1].last_run_ok)    level1_hasres = true;
    if (cs.ls_session.curves.size()    > 2 && cs.ls_session.curves[2].last_run_ok)    level1_hasres = true;
    bool level3_running = cs.phase_session.in_flight || cs.basins_session.in_flight;
    bool level3_hasres  = cs.phase_session.result.ok ||
                          (!cs.basins_session.configs.empty() && cs.basins_session.configs[0].last_run_ok);
    const char* l3_title = c.level3_kind == 0 ? "Level 3 - Phase / Time-domain" : "Level 3 - Basins";

    level_header(0, "##en_l2d", c.level_2d_enabled, "Level 2D - Bif / LLE / LS",
                 level2_running, level2_hasres, /*default_open=*/true,
                 [&](){ draw_level2d_detail(cs); });
    level_header(1, "##en_l1d", c.level_1d_enabled, "Level 1D - slices",
                 level1_running, level1_hasres, /*default_open=*/false,
                 [&](){ draw_level1d_detail(cs); });
    level_header(2, "##en_l3", c.level_phase_enabled, l3_title,
                 level3_running, level3_hasres, /*default_open=*/false,
                 [&](){ draw_level3_detail(cs); });
}

// ---- Custom-tab plot windows ----
//
// Minimal renderer for the pipeline output. Full-featured versions with
// colormap toolbars, colorbar controls, and colored-1D density are TODO;
// this MVP shows results + wires drill-down clicks / crosshairs.
namespace {

// Helper: bind slider fix_x/fix_y as crosshair on the given heatmap view,
// and install a drill-down click callback that updates shared.param_values
// + fix_x/y and optionally enqueues a Phase/Basins run.
void wire_2d_heatmap_interaction(HeatmapView& hv, CustomSession& cs,
                                 std::deque<CustomQueueItem>& q) {
    hv.crosshair_x = cs.shared.level_1d_enabled ? cs.shared.fix_x_value
                                                : std::numeric_limits<double>::quiet_NaN();
    hv.crosshair_y = cs.shared.level_1d_enabled ? cs.shared.fix_y_value
                                                : std::numeric_limits<double>::quiet_NaN();
    // Drag callback: fires every frame while LMB is held inside the plot
    // (see HeatmapView on_left_drag). Cheap — just moves the crosshair by
    // updating fix_x/fix_y so the shared crosshair on every 2D window
    // follows the cursor. Recompute (Phase/Basins enqueue + param_values
    // update) is deferred to on_left_click on release.
    hv.on_left_drag = [&cs](int, int, double snap_x, double snap_y) {
        cs.shared.fix_x_value = snap_x;
        cs.shared.fix_y_value = snap_y;
        // Heatmap drag moves BOTH axes → bump both timers so the settled
        // branch re-runs X-slices AND Y-slices.
        double t = ImGui::GetTime();
        cs.shared.last_fix_x_change_time = t;
        cs.shared.last_fix_y_change_time = t;
    };
    hv.on_left_click = [&cs, &q](int, int, double snap_x, double snap_y) {
        auto& s = cs.shared;
        // Update fix_x/fix_y (also drives crosshair on other 2D windows).
        s.fix_x_value = snap_x;
        s.fix_y_value = snap_y;
        double t = ImGui::GetTime();
        s.last_fix_x_change_time = t;
        s.last_fix_y_change_time = t;
        // Update shared.param_values so any subsequent Phase/Basins run reads
        // the drilled-down location. Only pins param-sweeps (var-sweeps stay
        // as IC edits — pipeline drainer handles that path).
        // snap_x/snap_y — значения узлов от ucuda_node_value (та же функция, что
        // у ядра), поэтому пишем их round-trip форматом: %.6g, стоявший здесь,
        // ронял точность до 6 цифр, и портрет по клику считался рядом с
        // пикселем, а не в нём.
        if (!s.axis_x_over_var && s.axis_x_par_index >= 0 &&
            s.axis_x_par_index < (int)cs.params.size()) {
            s.param_values[cs.params[s.axis_x_par_index]] = fmt_num_shortest(snap_x);
        }
        if (!s.axis_y_over_var && s.axis_y_par_index >= 0 &&
            s.axis_y_par_index < (int)cs.params.size()) {
            s.param_values[cs.params[s.axis_y_par_index]] = fmt_num_shortest(snap_y);
        }
        if (s.autorun_on_drilldown && s.level_phase_enabled) {
            q.push_back({ s.level3_kind == 0 ? CustomQueueItem::Kind::Phase
                                             : CustomQueueItem::Kind::Basins });
        }
    };
}

} // namespace

// ============================================================================
// Custom Workspace — split-region layout for Custom AppMode.
//
// Structure per frame:
//   +---------------------------------------------------------------+
//   |                       AppMode radios                          |  (MainHost)
//   +---------+---+-------------------------------------------------+
//   |         | s |  [ Tab 1 | Tab 2 | + ]                          |
//   | Custom  | p |  +--------------------------------------------+ |
//   | Controls| l |  |                                            | |
//   |  panel  | i |  |   Per-tab DockSpace (plot windows here)    | |
//   |         | t |  |                                            | |
//   +---------+---+--+--------------------------------------------+-+
//
// Docking model:
//   - Each tab owns one DockSpace with a STABLE id derived from tab.id
//     via ws_dock_id() — deliberately NOT ImGui::GetID(str), because
//     GetID hashes with the current window-ID stack (different id inside
//     BeginTabItem vs outside), and we submit the same dockspace both in
//     KeepAliveOnly form (before BeginTabBar) and in real form (inside
//     BeginTabItem of the active tab).
//   - Every frame, we submit ALL per-tab dockspaces with the flag
//     ImGuiDockNodeFlags_KeepAliveOnly. This tells ImGui "these nodes
//     still exist" so windows docked inside inactive tabs don't get
//     orphaned to a floating state.
//   - The active tab additionally re-submits its dockspace with normal
//     flags inside BeginTabItem so it renders.
//   - Plot windows are docked into their tab's dockspace via
//     DockBuilderDockWindow on first appearance; imgui.ini persists the
//     internal split layout across sessions.
//   - Cross-tab drag&drop: BeginDragDropSource on each plot's title bar
//     + BeginDragDropTarget on each tab item → drop calls
//     DockBuilderDockWindow(name, target_ds) and switches to that tab.
// ============================================================================

// Stable dockspace ID for a workspace tab. Independent of ImGui's ID-stack
// scope (so the same id is produced whether we compute it inside a window
// or outside), and unlikely to collide with other IDs in the app.
// Includes a hash of the currently-loaded system name so imgui.ini stores a
// completely separate dock layout per system — otherwise switching systems
// leaked one system's plot placement into another's tabs.
[[nodiscard]] static inline ImGuiID ws_dock_id(int tab_id, const std::string& sys) {
    ImGuiID sys_hash = sys.empty() ? 0u : ImHashStr(sys.c_str());
    return (ImGuiID)0xD5000000u ^ (ImGuiID)tab_id ^ sys_hash;
}

// Suffix appended to every Custom-mode plot window title so imgui.ini keys
// its dock state per system. Empty when no system is loaded (edge case;
// windows behave as before). "##sys_<name>" — the "##" makes ImGui strip
// the suffix from the visible title while including it in the hashed ID.
[[nodiscard]] static inline std::string custom_win_suffix(const std::string& sys) {
    return sys.empty() ? std::string{} : ("##sys_" + sys);
}

// Drag&drop payload key — passed via ImGui's built-in drag-drop channel.
// Payload data: `const char* name` of the docked window title.
static const char* WS_DRAG_PAYLOAD = "CUSTOM_WS_WIN";

// Forward decls — the layout function calls these; they live further down.
static void draw_custom_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb);

static void draw_custom_controls_panel(AppModel& model, SystemLibrary& lib,
                                       ImVec2 pos, ImVec2 size) {
    ImGui::SetNextWindowPos(pos);
    ImGui::SetNextWindowSize(size);
    // No NoBringToFrontOnFocus / NoNavFocus here — clicks on Controls/Workspace
    // widgets were being swallowed with those on; keeping the panels pinned via
    // NoMove/NoResize + SetNextWindowPos/Size is enough to make them "regions".
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar
                           | ImGuiWindowFlags_NoResize
                           | ImGuiWindowFlags_NoMove
                           | ImGuiWindowFlags_NoCollapse
                           | ImGuiWindowFlags_NoDocking;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    if (ImGui::Begin("Custom Controls##panel", nullptr, flags)) {
        draw_custom_controls(model, lib);
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

// Runtime rename state — Custom mode is the only place we rename tabs, so
// keeping it as a file-scope static (not persisted) is fine.
struct WsRenameState {
    int  target_id = 0;              // 0 = no rename in progress.
    char buf[128]  = {};
};
static WsRenameState g_ws_rename;

static void draw_custom_workspace_panel(CustomSession& cs,
                                        const std::string& sys,
                                        ImVec2 pos, ImVec2 size) {
    auto& ws = cs.workspace;
    ws.ensure_default();

    ImGui::SetNextWindowPos(pos);
    ImGui::SetNextWindowSize(size);
    // No NoBringToFrontOnFocus / NoNavFocus here — clicks on Controls/Workspace
    // widgets were being swallowed with those on; keeping the panels pinned via
    // NoMove/NoResize + SetNextWindowPos/Size is enough to make them "regions".
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar
                           | ImGuiWindowFlags_NoResize
                           | ImGuiWindowFlags_NoMove
                           | ImGuiWindowFlags_NoCollapse
                           | ImGuiWindowFlags_NoDocking;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    if (!ImGui::Begin("Custom Workspace##panel", nullptr, flags)) {
        ImGui::End();
        ImGui::PopStyleVar();
        return;
    }

    // Tab bar with +, close, rename, and cross-tab drop targets.
    // NOTE ON DOCKING: the real DockSpace for the active tab is submitted
    // AFTER EndTabBar (not inside BeginTabItem body). Reason: submitting
    // DockSpace(id) inside BeginTabItem pushes an ID-stack entry (the tab
    // item), so ImGui creates an auto-child host window "…/DockSpace_XXX"
    // whose host context differs from the KeepAliveOnly submit outside
    // BeginTabItem. That mismatch caused docked windows to orphan on tab
    // switch (plots vanishing when returning to a previously-active tab).
    // Submitting real + keep-alive from the same context (workspace panel
    // window, outside BeginTabItem) keeps the host consistent.
    int close_id = 0;
    int switch_to_id = 0;
    ImGuiTabBarFlags tb_flags = ImGuiTabBarFlags_Reorderable
                              | ImGuiTabBarFlags_TabListPopupButton
                              | ImGuiTabBarFlags_AutoSelectNewTabs
                              | ImGuiTabBarFlags_FittingPolicyScroll;
    if (ImGui::BeginTabBar("##custom_ws_tabs", tb_flags)) {
        const bool allow_close = (int)ws.tabs.size() > 1;
        for (size_t i = 0; i < ws.tabs.size(); ++i) {
            WorkspaceTab& tab = ws.tabs[i];
            // Stable tab ID via "##ws_tab_<id>" suffix so rename changes
            // only the visible label, not the internal ImGui key.
            std::string label = tab.name + "##ws_tab_" + std::to_string(tab.id);
            bool keep = true;
            bool item_open = ImGui::BeginTabItem(label.c_str(),
                                                 allow_close ? &keep : nullptr,
                                                 ImGuiTabItemFlags_None);
            // Drop-target for cross-tab window move. Accepts:
            //   - Our custom payload (WS_DRAG_PAYLOAD, `const char*` name):
            //     used if we ever add an explicit drag handle.
            //   - ImGui's native window docking payload (IMGUI_PAYLOAD_TYPE_WINDOW,
            //     `ImGuiWindow*`): fired when the user drags a docked plot's
            //     tab out of its dockspace and hovers over ours. This is what
            //     enables the requested "native drag&drop between tabs" UX.
            if (ImGui::BeginDragDropTarget()) {
                if (const ImGuiPayload* p = ImGui::AcceptDragDropPayload(WS_DRAG_PAYLOAD)) {
                    const char* win_name = (const char*)p->Data;
                    if (win_name && *win_name) {
                        ImGui::DockBuilderDockWindow(win_name, ws_dock_id(tab.id, sys));
                        switch_to_id = tab.id;
                        ws.dirty = true;
                    }
                }
                if (const ImGuiPayload* p = ImGui::AcceptDragDropPayload(IMGUI_PAYLOAD_TYPE_WINDOW)) {
                    ImGuiWindow* dragged = *(ImGuiWindow**)p->Data;
                    if (dragged && dragged->Name) {
                        ImGui::DockBuilderDockWindow(dragged->Name, ws_dock_id(tab.id, sys));
                        switch_to_id = tab.id;
                        ws.dirty = true;
                    }
                }
                ImGui::EndDragDropTarget();
            }
            // Right-click menu (rename / close).
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Rename")) {
                    g_ws_rename.target_id = tab.id;
                    std::snprintf(g_ws_rename.buf, sizeof(g_ws_rename.buf),
                                  "%s", tab.name.c_str());
                }
                if (allow_close && ImGui::MenuItem("Close tab")) {
                    close_id = tab.id;
                }
                ImGui::EndPopup();
            }
            if (item_open) {
                if (ws.active_tab_id != tab.id) {
                    ws.active_tab_id = tab.id;
                    ws.dirty = true;
                }
                // Empty body — real DockSpace is submitted AFTER EndTabBar
                // (see NOTE ON DOCKING above).
                ImGui::EndTabItem();
            }
            if (!keep) close_id = tab.id;
        }
        if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing
                                    | ImGuiTabItemFlags_NoTooltip)) {
            WorkspaceTab nt;
            // id keeps monotonically incrementing — it seeds the DockSpace
            // node id, so reusing an id could collide with imgui.ini state
            // of a previously-closed tab. Display name, though, picks the
            // lowest unused "Tab N" number so the user doesn't watch the
            // counter grow to 20+ after a few open/close cycles.
            nt.id = ws.next_tab_id++;
            int label_num = 1;
            auto name_taken = [&](int n){
                std::string s = "Tab " + std::to_string(n);
                return std::any_of(ws.tabs.begin(), ws.tabs.end(),
                                   [&](const WorkspaceTab& t){ return t.name == s; });
            };
            while (name_taken(label_num)) ++label_num;
            nt.name = "Tab " + std::to_string(label_num);
            ws.tabs.push_back(nt);
            ws.active_tab_id = nt.id;
            ws.dirty = true;
        }
        // Sync ws.tabs order with ImGui's internal reordered order — user
        // can drag tabs (Reorderable flag) but ImGui only shuffles its own
        // Tabs array; without this our vector stays in the original order,
        // so plots that a user docked into "the third tab visually" end up
        // in whichever tab id happens to be third in ws.tabs on restart.
        // Sync fixes both "tab order not remembered" AND "plots appear in
        // the wrong tab after restart" symptoms.
        if (ImGuiTabBar* tb = ImGui::GetCurrentTabBar()) {
            if (tb->Tabs.Size >= 2 && (int)ws.tabs.size() == tb->Tabs.Size) {
                std::vector<WorkspaceTab> reordered;
                reordered.reserve(ws.tabs.size());
                bool all_matched = true;
                for (int i = 0; i < tb->Tabs.Size; ++i) {
                    ImGuiID tid = tb->Tabs[i].ID;
                    auto it = std::find_if(ws.tabs.begin(), ws.tabs.end(),
                        [&](const WorkspaceTab& t){
                            std::string label = t.name + "##ws_tab_" + std::to_string(t.id);
                            return ImHashStr(label.c_str()) == tid;
                        });
                    if (it == ws.tabs.end()) { all_matched = false; break; }
                    reordered.push_back(*it);
                }
                if (all_matched) {
                    // Only mark dirty when actual order changed, to avoid a
                    // needless _last_custom.json rewrite every frame.
                    bool changed = false;
                    for (size_t i = 0; i < ws.tabs.size(); ++i) {
                        if (ws.tabs[i].id != reordered[i].id) { changed = true; break; }
                    }
                    if (changed) {
                        ws.tabs = std::move(reordered);
                        ws.dirty = true;
                    }
                }
            }
        }
        ImGui::EndTabBar();
    }

    // Submit all workspace dockspaces from the SAME host context (workspace
    // panel, outside BeginTabItem). Active tab gets a real submit that
    // renders + hosts its docked windows; inactive tabs get KeepAliveOnly
    // so their docked windows remain docked (invisible until user switches
    // to that tab) instead of orphaning to floating state.
    for (const auto& tab : ws.tabs) {
        if (tab.id != ws.active_tab_id) {
            ImGui::DockSpace(ws_dock_id(tab.id, sys), ImVec2(0, 0),
                             ImGuiDockNodeFlags_KeepAliveOnly);
        }
    }
    if (std::any_of(ws.tabs.begin(), ws.tabs.end(),
                    [&](const WorkspaceTab& t){ return t.id == ws.active_tab_id; })) {
        ImGui::DockSpace(ws_dock_id(ws.active_tab_id, sys), ImVec2(0, 0),
                         ImGuiDockNodeFlags_None);
    }

    // Deferred tab close — move every window still docked in the closing
    // tab's dockspace into the fallback tab (front). DockBuilderRemoveNode
    // moves them off first, so we manually re-dock instead.
    if (close_id != 0) {
        auto it = std::find_if(ws.tabs.begin(), ws.tabs.end(),
                               [close_id](const WorkspaceTab& t){ return t.id == close_id; });
        if (it != ws.tabs.end() && ws.tabs.size() > 1) {
            ws.tabs.erase(it);
            ws.dirty = true;
            int fallback = ws.tabs.front().id;
            // Find all windows docked in the closing node and re-dock them
            // into the fallback. Iterate a copy of the node's windows list
            // because DockBuilderDockWindow mutates it under us.
            ImGuiDockNode* node = ImGui::DockBuilderGetNode(ws_dock_id(close_id, sys));
            if (node) {
                // Walk the whole subtree (splits) and re-dock every window
                // encountered. Simple recursive lambda.
                std::vector<std::string> to_move;
                std::function<void(ImGuiDockNode*)> walk = [&](ImGuiDockNode* n){
                    if (!n) return;
                    for (int wi = 0; wi < n->Windows.Size; ++wi) {
                        if (n->Windows[wi] && n->Windows[wi]->Name)
                            to_move.emplace_back(n->Windows[wi]->Name);
                    }
                    walk(n->ChildNodes[0]);
                    walk(n->ChildNodes[1]);
                };
                walk(node);
                for (const auto& name : to_move)
                    ImGui::DockBuilderDockWindow(name.c_str(), ws_dock_id(fallback, sys));
                // Now discard the closed node's split hierarchy.
                ImGui::DockBuilderRemoveNode(ws_dock_id(close_id, sys));
            }
            if (ws.active_tab_id == close_id) ws.active_tab_id = fallback;
        }
    }
    if (switch_to_id != 0) ws.active_tab_id = switch_to_id;

    ImGui::End();
    ImGui::PopStyleVar();

    // Rename modal — centered on viewport, Esc = Cancel, auto-reset if the
    // user clicks outside. IMPORTANT: SetNextWindowPos must be INSIDE the
    // "opening" branch — an unconditional call left the "next-window-pos"
    // flag dangling and got applied to the first plot window created that
    // frame (which appeared floating at viewport center and looked like a
    // dark overlay covering the workspace).
    if (g_ws_rename.target_id != 0) {
        ImGui::OpenPopup("Rename tab##custom_ws");
        ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    }
    if (ImGui::BeginPopupModal("Rename tab##custom_ws", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::SetKeyboardFocusHere();
        bool commit = ImGui::InputText("##ws_rename",
                                       g_ws_rename.buf, sizeof(g_ws_rename.buf),
                                       ImGuiInputTextFlags_EnterReturnsTrue);
        if (ImGui::Button("OK") || commit) {
            for (auto& t : ws.tabs) {
                if (t.id == g_ws_rename.target_id) {
                    std::string s = g_ws_rename.buf;
                    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();
                    if (!s.empty() && s != t.name) {
                        t.name = s;
                        ws.dirty = true;
                    }
                    break;
                }
            }
            g_ws_rename.target_id = 0;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            g_ws_rename.target_id = 0;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    } else if (g_ws_rename.target_id != 0) {
        // Popup was closed by ImGui itself (clicked outside etc.). Clear the
        // trigger flag so we don't immediately re-open.
        g_ws_rename.target_id = 0;
    }
}

static void draw_custom_mode_layout(AppModel& model, SystemLibrary& lib,
                                    const GuiCallbacks& cb,
                                    ImVec2 area_pos, ImVec2 area_size) {
    auto& ws = model.custom_session.workspace;
    ws.ensure_default();

    if (area_size.x < 100.0f || area_size.y < 40.0f) return;

    // Clamp splitter to sane bounds for the current viewport.
    const float min_ctrl   = 200.0f;
    const float min_ws     = 300.0f;
    const float splitter_w = 6.0f;
    float max_ctrl = std::max(min_ctrl, area_size.x - min_ws - splitter_w);
    if (ws.controls_width < min_ctrl) ws.controls_width = min_ctrl;
    if (ws.controls_width > max_ctrl) ws.controls_width = max_ctrl;

    ImVec2 ctrl_pos  = area_pos;
    ImVec2 ctrl_size(ws.controls_width, area_size.y);
    ImVec2 split_pos(area_pos.x + ws.controls_width, area_pos.y);
    ImVec2 split_size(splitter_w, area_size.y);
    ImVec2 ws_pos  (split_pos.x + splitter_w, area_pos.y);
    ImVec2 ws_size (area_size.x - ws.controls_width - splitter_w, area_size.y);

    // Splitter (invisible button + drawn rect).
    ImGui::SetNextWindowPos(split_pos);
    ImGui::SetNextWindowSize(split_size);
    // NoBackground — splitter draws its own filled rect manually, so ImGui's
    // window background (which showed as the "dark strip" artefact on click)
    // is redundant. NoBringToFrontOnFocus swallows the drag input entirely,
    // so we can't use it here (unlike on Controls/Workspace panels).
    ImGuiWindowFlags split_flags = ImGuiWindowFlags_NoTitleBar
                                 | ImGuiWindowFlags_NoResize
                                 | ImGuiWindowFlags_NoMove
                                 | ImGuiWindowFlags_NoScrollbar
                                 | ImGuiWindowFlags_NoScrollWithMouse
                                 | ImGuiWindowFlags_NoCollapse
                                 | ImGuiWindowFlags_NoNavFocus
                                 | ImGuiWindowFlags_NoBackground
                                 | ImGuiWindowFlags_NoDocking;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::Begin("##CustomSplitter", nullptr, split_flags);
    ImGui::InvisibleButton("##split_btn", split_size);
    if (ImGui::IsItemActive()) {
        float dx = ImGui::GetIO().MouseDelta.x;
        if (dx != 0.0f) {
            ws.controls_width += dx;
            if (ws.controls_width < min_ctrl) ws.controls_width = min_ctrl;
            if (ws.controls_width > max_ctrl) ws.controls_width = max_ctrl;
            ws.dirty = true;
        }
    }
    ImU32 col = ImGui::IsItemActive()  ? ImGui::GetColorU32(ImGuiCol_SeparatorActive)
              : ImGui::IsItemHovered() ? ImGui::GetColorU32(ImGuiCol_SeparatorHovered)
                                       : ImGui::GetColorU32(ImGuiCol_Separator);
    ImGui::GetWindowDrawList()->AddRectFilled(split_pos,
        ImVec2(split_pos.x + split_size.x, split_pos.y + split_size.y), col);
    if (ImGui::IsItemHovered() || ImGui::IsItemActive())
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    ImGui::End();
    ImGui::PopStyleVar(3);

    draw_custom_controls_panel(model, lib, ctrl_pos, ctrl_size);
    draw_custom_workspace_panel(model.custom_session, model.loaded_name, ws_pos, ws_size);
    draw_custom_plot_windows(model, lib, cb);
}

static void draw_custom_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) {
    (void)lib; (void)cb;
    auto& cs = model.custom_session;
    auto& q  = model.custom_queue;
    auto& ws = cs.workspace;
    // Per-system suffix appended to every plot window title. imgui.ini keys
    // dock state per window name, so without a per-system suffix the same
    // plot titles (e.g. "Custom Bif 2D") would share layout across systems.
    // Empty when no system is loaded — safe (title unchanged from before).
    const std::string& sys    = model.loaded_name;
    const std::string  suffix = custom_win_suffix(sys);
    // Per-system delta XOR'd into every plot's owner_id — HeatmapView /
    // PlotRenderer cache their GPU texture per owner_id, so without this
    // both systems shared one texture and the FIRST system's rendered pixels
    // stayed on screen while the SECOND system's data sat unrendered
    // underneath. Low bits of the system-name hash — enough entropy without
    // colliding with the hand-picked owner_id constants below.
    const int sys_owner_delta = sys.empty() ? 0
                              : (int)(ImHashStr(sys.c_str()) & 0x00FFFFFF);

    // Persistent per-window renderers + views. Indexed by fixed slots so their
    // ImGui-IDs stay stable across frames. Hoisted to the top of the function
    // (together with the L1D-side statics and the init-seen sentinels) so a
    // single system-change block below can reset the whole cache tier.
    static std::array<std::unique_ptr<PlotRenderer>, 3> renderers_2d;
    static std::array<HeatmapView, 3>                   heatmaps;
    static std::array<int, 3>                           hm_init_cmap_seen = { -2, -2, -2 };
    static int                                          ls_init_exp_seen  = -999;
    static std::array<std::unique_ptr<Plot2DView>,  6>  l1_views;
    static std::array<std::unique_ptr<PlotRenderer>, 6> l1_renderers;
    static std::array<std::vector<std::vector<float>>, 6> l1_bufs;
    // Custom Basins (Level-3 kind=1) HeatmapView + renderer — hoisted so the
    // system-change block below can reset them alongside the other statics.
    static HeatmapView                                  bsn_hv;
    static PlotRenderer                                 bsn_renderer;
    for (auto& r : renderers_2d) if (!r) r = std::make_unique<PlotRenderer>();

    // System-change detection. Slot-indexed static caches (HeatmapView state
    // — view_valid, autofit ranges, seen colormap; Plot2DView state — view
    // limits, cached VBO series_generation) leaked across systems because
    // slot 0 for both Chen and Rossler used the same HeatmapView instance.
    // On rename of the loaded system, wipe the per-slot view / seen state so
    // the new system starts with fresh autofit + fresh colormap seeding.
    // renderers_* (PlotRenderer/GPU textures + FBOs) are reused: owner_id
    // XOR'd with sys_owner_delta already forces PlotRenderer to treat the
    // slot as a new cache entry, and destroying them here would leak GPU
    // memory tied to the GL context.
    static std::string last_system_for_plots;
    if (last_system_for_plots != sys) {
        last_system_for_plots = sys;
        // HeatmapView / Plot2DView are non-copyable — reset only the fields
        // that gate rendering (view_valid / data_gen_cached / series_generation),
        // which are the actual source of the leak. Plot2DView caches its VBO
        // by (owner_id, series_generation); the owner_id XOR alone wasn't
        // enough — series_generation also had to be invalidated, else the
        // second system's data_generation=1 matched the first's cached
        // series_generation=1 and Plot2DView kept rendering the FIRST
        // system's VBO with its coordinates. Colormap / autoscale / manual
        // v-limits stay so user prefs aren't wiped mid-session.
        for (auto& hv : heatmaps) { hv.view_valid = false; hv.data_gen_cached = -1; }
        for (auto& v  : l1_views) if (v) { v->view_valid = false; v->series_generation = -1; }
        for (auto& b  : l1_bufs)  b.clear();
        bsn_hv.view_valid = false; bsn_hv.data_gen_cached = -1;
        hm_init_cmap_seen = { -2, -2, -2 };
        ls_init_exp_seen  = -999;
    }

    // Clamp fix_x/fix_y to the current effective sweep range every frame.
    // custom_session.h documents this ("clamped to current effective sweep
    // ranges") but it was never actually implemented — a stale fix outside
    // the sweep range fed the kernel via apply_shared_to_bif1d/pin_param and
    // produced degenerate 1D results (all-zero or empty on Run) until the
    // user dragged the slider back inside the range.
    {
        EffectiveSweep esx = effective_sweep_x(cs.shared);
        EffectiveSweep esy = effective_sweep_y(cs.shared);
        double x_lo = parse_ratio_or(esx.lo_text, 0.0), x_hi = parse_ratio_or(esx.hi_text, 1.0);
        double y_lo = parse_ratio_or(esy.lo_text, 0.0), y_hi = parse_ratio_or(esy.hi_text, 1.0);
        if (x_hi < x_lo) std::swap(x_lo, x_hi);
        if (y_hi < y_lo) std::swap(y_lo, y_hi);
        if (cs.shared.fix_x_value < x_lo) cs.shared.fix_x_value = x_lo;
        if (cs.shared.fix_x_value > x_hi) cs.shared.fix_x_value = x_hi;
        if (cs.shared.fix_y_value < y_lo) cs.shared.fix_y_value = y_lo;
        if (cs.shared.fix_y_value > y_hi) cs.shared.fix_y_value = y_hi;
    }

    // Dock a plot window into the active tab ONLY when there is no dock
    // memory for it — neither at runtime (`window->DockId`) NOR in
    // imgui.ini (`ImGuiWindowSettings::DockId`). This lets a reloaded
    // session restore its per-tab layout from ini on the very first frame
    // (when the window hasn't been Begun yet, so FindWindowByName returns
    // null but ImGui has the settings ready to apply). Overriding a
    // persisted DockId here caused "plots all get dumped into the active
    // tab on startup".
    //
    // Skipped while the mouse is held down so we don't yank a window out
    // of an in-flight drag.
    auto ensure_docked = [&](const std::string& name) {
        if (ImGui::GetIO().MouseDown[0]) return;
        ImGuiID persisted_dock = 0;
        if (ImGuiWindow* w = ImGui::FindWindowByName(name.c_str())) {
            persisted_dock = w->DockId;
        } else if (ImGuiWindowSettings* s =
                       ImGui::FindWindowSettingsByID(ImHashStr(name.c_str()))) {
            persisted_dock = s->DockId;   // will be applied on window's next Begin
        }
        if (persisted_dock != 0) return;
        ImGui::DockBuilderDockWindow(name.c_str(), ws_dock_id(ws.active_tab_id, sys));
    };

    // Cross-tab window move is handled entirely via native ImGui drag&drop:
    // drop a plot's tab onto a target tab in the workspace tab bar (see
    // BeginDragDropTarget in draw_custom_workspace_panel, which accepts
    // IMGUI_PAYLOAD_TYPE_WINDOW). No MMB menu here — MMB is claimed by
    // heatmap/1D-plot views for crosshair-drag / slice movement.

    auto axis_name = [&](bool over_var, int par_i, int var_i, bool over_h) -> std::string {
        if (over_h)   return "h";
        if (over_var) return (var_i >= 0 && var_i < (int)cs.vars.size())
                             ? cs.vars[var_i] + " (IC)" : "x";
        return (par_i >= 0 && par_i < (int)cs.params.size()) ? cs.params[par_i] : "param";
    };
    const std::string ax_x = axis_name(cs.shared.axis_x_over_var,
                                       cs.shared.axis_x_par_index,
                                       cs.shared.axis_x_var_index,
                                       cs.shared.axis_x_over_h);
    const std::string ax_y = axis_name(cs.shared.axis_y_over_var,
                                       cs.shared.axis_y_par_index,
                                       cs.shared.axis_y_var_index,
                                       cs.shared.axis_y_over_h);

    // --- Level 2D heatmaps ---
    struct L2Slot {
        std::string title;   // includes per-system suffix so imgui.ini isolates layout per system
        bool        show;
        bool        has_data;
        int         n_pts;
        const double* values;
        double lo_x, hi_x, lo_y, hi_y;
        double vmin, vmax;
        int    data_gen;
        int    owner_id;
    };
    L2Slot slots[3] = {};
    if (cs.shared.level_2d_enabled) {
        if (cs.shared.bif2d_enabled && !cs.bif_session.diagrams.empty()) {
            auto& d = cs.bif_session.diagrams[0];
            slots[0] = { "Custom Bif 2D" + suffix, true, d.last_run_2d_ok && !d.result_2d.values.empty(),
                         d.result_2d.n_pts, d.result_2d.values.data(),
                         d.result_2d.param_lo, d.result_2d.param_hi,
                         d.result_2d.param_lo_2, d.result_2d.param_hi_2,
                         d.result_2d.min_val, d.result_2d.max_val,
                         d.data_generation_2d, 0xCB1F2D ^ sys_owner_delta };
        }
        if (cs.shared.lle2d_enabled && !cs.lle_session.curves.empty()) {
            auto& c = cs.lle_session.curves[0];
            slots[1] = { "Custom LLE 2D" + suffix, true, c.last_run_2d_ok && !c.result_2d.values.empty(),
                         c.result_2d.n_pts, c.result_2d.values.data(),
                         c.result_2d.param_lo, c.result_2d.param_hi,
                         c.result_2d.param_lo_2, c.result_2d.param_hi_2,
                         c.result_2d.min_val, c.result_2d.max_val,
                         c.data_generation_2d, 0xCE1E2D ^ sys_owner_delta };
        }
        if (cs.shared.ls2d_enabled && !cs.ls_session.curves.empty()) {
            auto& c = cs.ls_session.curves[0];
            // Fallback plane pointer = first exponent (L1). Real plane for
            // has_data / render is picked later in the toolbar block, so
            // has_data must NOT depend on which exponent (or "sum") is
            // currently selected — otherwise picking "sum L_i" flips the
            // slot to "No data yet" because the fallback here goes nullptr.
            const double* values_default = c.result_2d.values.empty()
                ? nullptr : c.result_2d.values.data();
            double vmin = c.result_2d.min_val.empty() ? 0.0 : c.result_2d.min_val[0];
            double vmax = c.result_2d.max_val.empty() ? 1.0 : c.result_2d.max_val[0];
            slots[2] = { "Custom LS 2D" + suffix, true, c.last_run_2d_ok && !c.result_2d.values.empty(),
                         c.result_2d.n_pts, values_default,
                         c.result_2d.param_lo, c.result_2d.param_hi,
                         c.result_2d.param_lo_2, c.result_2d.param_hi_2,
                         vmin, vmax, c.data_generation_2d, 0xC152D0 ^ sys_owner_delta };
        }
    }
    // One-time init of per-slot HeatmapView colormap from the sub-session
    // config (or app default). Runs whenever the sub-session-level slot's
    // colormap changes, so first appearance of a slot picks up the persisted
    // choice; subsequent user picks from the toolbar update both places.
    // (hm_init_cmap_seen is hoisted to the top of this function so the
    // system-change reset can wipe it.)
    auto init_cmap_from_config = [&](int i, int cfg_cmap) {
        if (hm_init_cmap_seen[i] == cfg_cmap) return;
        hm_init_cmap_seen[i] = cfg_cmap;
        heatmaps[i].colormap =
            (HeatmapColormap)colormap_id_or(cfg_cmap, model.heatmap_colormap);
    };
    // Sync LS exponent choice from the sub-session's persisted
    // display_exponent_idx (sentinel -1 = sum L_i) on first appearance —
    // otherwise session_from_json_custom loads the pref but the HeatmapView
    // silently starts at 0 (L1) until the user re-picks. Uses the same
    // "seen" pattern as the colormap sync so subsequent user picks aren't
    // clobbered. (ls_init_exp_seen hoisted to the top of the function.)
    if (!cs.ls_session.curves.empty()) {
        int cfg_exp = cs.ls_session.curves[0].display_exponent_idx;
        if (ls_init_exp_seen != cfg_exp) {
            ls_init_exp_seen = cfg_exp;
            heatmaps[2].display_exponent_idx = cfg_exp;
        }
    }
    if (!cs.bif_session.diagrams.empty())
        init_cmap_from_config(0, cs.bif_session.diagrams[0].colormap_idx);
    if (!cs.lle_session.curves.empty())
        init_cmap_from_config(1, cs.lle_session.curves[0].colormap_idx);
    if (!cs.ls_session.curves.empty())
        init_cmap_from_config(2, cs.ls_session.curves[0].colormap_idx);

    for (int i = 0; i < 3; ++i) {
        if (!slots[i].show) continue;
        ensure_docked(slots[i].title);
        if (ImGui::Begin(slots[i].title.c_str())) {
            HeatmapView& hv = heatmaps[i];

            // Toolbar — mirrors draw_bifurcation_plot / draw_ls_plot layout
            // (colormap combo + autoscale + vmin/vmax + swap axes + LS
            // exponent picker for slot 2).
            ImGui::PushID(i);

            // LS-плоскость по выбранной экспоненте. Заполняется внутри
            // extras-колбэка тулбара (там же, где рисуется picker), а
            // используется ниже при render — поэтому объявлено до тулбара.
            const double* ls_plane = nullptr;
            double        ls_vmin = 0.0, ls_vmax = 1.0;
            int           ls_gen  = slots[i].data_gen;

            {
                HeatmapToolbarOpts topts;
                topts.persist_colormap = [&](int cm) {
                    // Persist per-slot in the owning sub-session config, so a
                    // saved _last_custom.json restores the choice.
                    if      (i == 0 && !cs.bif_session.diagrams.empty())
                        cs.bif_session.diagrams[0].colormap_idx = cm;
                    else if (i == 1 && !cs.lle_session.curves.empty())
                        cs.lle_session.curves[0].colormap_idx = cm;
                    else if (i == 2 && !cs.ls_session.curves.empty())
                        cs.ls_session.curves[0].colormap_idx = cm;
                    hm_init_cmap_seen[i] = cm;
                };
                topts.extras = [&]() {
                    if (i != 2 || cs.ls_session.curves.empty()) return;
                    auto& cact = cs.ls_session.curves[0];
                    if (!cact.last_run_2d_ok || cact.result_2d.n_exponents <= 0) return;
                    draw_ls_exponent_picker(hv, cact.result_2d.n_exponents,
                                            [&](int j) { cact.display_exponent_idx = j; });
                    ls_resolve_plane(cact, hv.display_exponent_idx,
                                     ls_plane, ls_vmin, ls_vmax, ls_gen);
                };
                draw_heatmap_toolbar(hv, topts);
            }

            if (!slots[i].has_data) {
                ImGui::TextDisabled("No data yet. Press Run / Run Level 2D.");
                ImGui::PopID();
            } else {
                hv.x_axis.name = ax_x;
                hv.y_axis.name = ax_y;
                // Лог-оси — как в Parametric (hb.x_axis.log_scale = bd.log_scale):
                // движок раскладывает узлы лог-равномерно, ось обязана это
                // повторить, иначе тики и snap курсора врут. Флаг берём из
                // конфига слота, а не из живого shared: конфиг получает log
                // вместе с диапазоном на Run (apply_shared_to_*), поэтому ось
                // и данные всегда описывают один и тот же прогон.
                if (i == 0 && !cs.bif_session.diagrams.empty()) {
                    hv.x_axis.log_scale = cs.bif_session.diagrams[0].log_scale;
                    hv.y_axis.log_scale = cs.bif_session.diagrams[0].log_scale_2;
                } else if (i == 1 && !cs.lle_session.curves.empty()) {
                    hv.x_axis.log_scale = cs.lle_session.curves[0].log_scale;
                    hv.y_axis.log_scale = cs.lle_session.curves[0].log_scale_2;
                } else if (i == 2 && !cs.ls_session.curves.empty()) {
                    hv.x_axis.log_scale = cs.ls_session.curves[0].log_scale;
                    hv.y_axis.log_scale = cs.ls_session.curves[0].log_scale_2;
                }
                wire_2d_heatmap_interaction(hv, cs, q);

                // Export menu on right-click (parity with Parametric).
                const bool bd_busy = (i == 0) && cs.bif_session.in_flight
                                     && cs.bif_session.is_2d_run
                                     && cs.bif_session.running_diagram_index == 0;
                const bool lle_busy = (i == 1) && cs.lle_session.in_flight
                                      && cs.lle_session.is_2d_run
                                      && cs.lle_session.running_curve_index == 0;
                const bool ls_busy = (i == 2) && cs.ls_session.in_flight
                                     && cs.ls_session.is_2d_run
                                     && cs.ls_session.running_curve_index == 0;
                hv.popup_extras = [i, &cs, &cb, bd_busy, lle_busy, ls_busy]() {
                    const bool busy = bd_busy || lle_busy || ls_busy;
                    draw_export_menu_item(busy, cb, [i, &cs](const std::string& path) {
                        if      (i == 0 && !cs.bif_session.diagrams.empty())
                            data_export::export_bif2d(cs.bif_session.diagrams[0].result_2d, path);
                        else if (i == 1 && !cs.lle_session.curves.empty())
                            data_export::export_lle2d(cs.lle_session.curves[0].result_2d, path);
                        else if (i == 2 && !cs.ls_session.curves.empty())
                            data_export::export_ls2d(cs.ls_session.curves[0].result_2d, path);
                    });
                };

                // Route through LS-specific plane/vmin/vmax if the user
                // picked a different exponent; otherwise use the pre-baked
                // slot values.
                const double* values_ptr = (i == 2 && ls_plane) ? ls_plane : slots[i].values;
                double        vmin_use   = (i == 2 && ls_plane) ? ls_vmin  : slots[i].vmin;
                double        vmax_use   = (i == 2 && ls_plane) ? ls_vmax  : slots[i].vmax;
                int           gen_use    = (i == 2 && ls_plane) ? ls_gen   : slots[i].data_gen;

                ImVec2 avail  = ImGui::GetContentRegionAvail();
                ImVec2 origin = ImGui::GetCursorScreenPos();
                hv.render(*renderers_2d[i], origin, avail,
                          slots[i].owner_id, gen_use,
                          slots[i].n_pts, slots[i].n_pts,
                          values_ptr,
                          slots[i].lo_x, slots[i].hi_x,
                          slots[i].lo_y, slots[i].hi_y,
                          vmin_use, vmax_use,
                          /*fit_request*/ false);
                ImGui::PopID();
            }
        }
        ImGui::End();
    }

    // --- Level 1D slice plots (six independently toggleable) ---
    // Kind enum for the slot dispatch. Each slot picks its data from the
    // owning sub-session's slot [1] (X-slice) or [2] (Y-slice).
    enum class L1Kind { Bif, LLE, LS };
    struct L1Slot {
        std::string title;   // includes per-system suffix
        bool        show = false;
        L1Kind      kind = L1Kind::Bif;
        int         cfg_idx = -1;   // 1 = X-slice, 2 = Y-slice inside sub-session
        double      fix_pt = 0.0;   // world X of the OTHER-axis crosshair
    };
    L1Slot lslots[6] = {};
    if (cs.shared.level_1d_enabled) {
        double fx = cs.shared.fix_x_value;
        double fy = cs.shared.fix_y_value;
        if (cs.shared.bif1d_x_enabled) lslots[0] = { "Custom Bif 1D (X-slice)" + suffix, true, L1Kind::Bif, 1, fx };
        if (cs.shared.bif1d_y_enabled) lslots[1] = { "Custom Bif 1D (Y-slice)" + suffix, true, L1Kind::Bif, 2, fy };
        if (cs.shared.lle1d_x_enabled) lslots[2] = { "Custom LLE 1D (X-slice)" + suffix, true, L1Kind::LLE, 1, fx };
        if (cs.shared.lle1d_y_enabled) lslots[3] = { "Custom LLE 1D (Y-slice)" + suffix, true, L1Kind::LLE, 2, fy };
        if (cs.shared.ls1d_x_enabled)  lslots[4] = { "Custom LS 1D (X-slice)"  + suffix, true, L1Kind::LS,  1, fx };
        if (cs.shared.ls1d_y_enabled)  lslots[5] = { "Custom LS 1D (Y-slice)"  + suffix, true, L1Kind::LS,  2, fy };
    }

    // (l1_views / l1_renderers / l1_bufs statics hoisted to the top of the
    // function so the system-change reset can wipe them uniformly.)

    for (int i = 0; i < 6; ++i) {
        if (!lslots[i].show) continue;
        if (!l1_views[i])     l1_views[i]     = std::make_unique<Plot2DView>();
        if (!l1_renderers[i]) l1_renderers[i] = std::make_unique<PlotRenderer>();
        Plot2DView&  view = *l1_views[i];
        PlotRenderer& rnd = *l1_renderers[i];

        ensure_docked(lslots[i].title);
        if (!ImGui::Begin(lslots[i].title.c_str())) { ImGui::End(); continue; }
        ImGui::PushID(i);

        // Resolve owning config + result + real sweep-range from *_text
        // fields (result.param_lo/hi is often 0..1 default until engine
        // fills it — falling back to config text keeps the axis correct
        // when the user set 0.1..0.35 on the sweep).
        const auto axis_label_for_slot = [&](int cfg_idx) -> std::string {
            // cfg_idx 1 = X-slice → sweep_x_par_index or axis_x if inherit;
            // cfg_idx 2 = Y-slice → sweep_y_*.
            EffectiveSweep e = (cfg_idx == 1) ? effective_sweep_x(cs.shared)
                                              : effective_sweep_y(cs.shared);
            if (e.over_h) return "h";
            if (e.over_var)
                return (e.var_index >= 0 && e.var_index < (int)cs.vars.size())
                        ? cs.vars[e.var_index] + " (IC)" : "x (IC)";
            return (e.par_index >= 0 && e.par_index < (int)cs.params.size())
                    ? cs.params[e.par_index] : "param";
        };

        // ЕДИНАЯ конфигурация вида — та же функция, что применяет Parametric.
        // Раньше здесь стоял свой набор присваиваний, из-за чего одинаковые
        // диаграммы отличались между вкладками (см. configure_param_plot_view).
        const ParamPlotKind pkind =
            (lslots[i].kind == L1Kind::Bif) ? ParamPlotKind::Bifurcation
          : (lslots[i].kind == L1Kind::LLE) ? ParamPlotKind::LLE
                                            : ParamPlotKind::LS;
        configure_param_plot_view(view, pkind);

        bool ok = false;
        int  data_gen = 0;
        double param_lo = 0.0, param_hi = 1.0;
        int    n_pts = 0;
        // Лог-масштаб оси среза. Берём из КОНФИГА слота, а не из shared:
        // конфиг получает log вместе с param_lo/hi в apply_shared_to_bif1d на
        // Run, поэтому ось и диапазон всегда описывают один и тот же прогон.
        // Живой чекбокс в панели поменяет ось после Run (он входит в
        // l1d-сигнатуру, так что Run пересчитает уровень).
        bool   slice_log = false;
        // Y-axis label + series bookkeeping filled per-kind below.

        if (lslots[i].kind == L1Kind::Bif) {
            int idx = lslots[i].cfg_idx;
            if (idx < 0 || idx >= (int)cs.bif_session.diagrams.size()) { ImGui::PopID(); ImGui::End(); continue; }
            auto& d = cs.bif_session.diagrams[idx];
            ok = d.last_run_ok;
            data_gen = d.data_generation;
            param_lo = parse_ratio_or(d.param_lo_text, 0.0);
            param_hi = parse_ratio_or(d.param_hi_text, 1.0);
            n_pts = d.result.n_pts;
            slice_log = d.log_scale;
            view.x_axis.name = axis_label_for_slot(idx);
            view.y_axis.name = (d.writable_var >= 0 && d.writable_var < (int)cs.vars.size())
                               ? cs.vars[d.writable_var] : "X";
            // Как в Parametric: при plot_inter_peaks по Y идут интервалы
            // между пиками, а не сама переменная.
            if (d.plot_inter_peaks) view.y_axis.name += " interval";
            // Custom point style — тот же тулбар и та же семантика, что в
            // Parametric (состояние живёт в самой БД, поэтому вкладки не
            // расходятся). Здесь член ровно один (X- или Y-срез),
            // синхронизировать нечего.
            if (draw_point_style_toolbar(d, "custombd"))
                cs.workspace.dirty = true;   // → пересохранение _last_custom.json
            apply_point_style(view, d);
        } else if (lslots[i].kind == L1Kind::LLE) {
            int idx = lslots[i].cfg_idx;
            if (idx < 0 || idx >= (int)cs.lle_session.curves.size()) { ImGui::PopID(); ImGui::End(); continue; }
            auto& c = cs.lle_session.curves[idx];
            ok = c.last_run_ok;
            data_gen = c.data_generation;
            param_lo = parse_ratio_or(c.param_lo_text, 0.0);
            param_hi = parse_ratio_or(c.param_hi_text, 1.0);
            n_pts = (int)c.result.lyapunov.size();
            slice_log = c.log_scale;
            view.x_axis.name = axis_label_for_slot(idx);
        } else { // LS
            int idx = lslots[i].cfg_idx;
            if (idx < 0 || idx >= (int)cs.ls_session.curves.size()) { ImGui::PopID(); ImGui::End(); continue; }
            auto& c = cs.ls_session.curves[idx];
            ok = c.last_run_ok;
            data_gen = c.data_generation;
            param_lo = parse_ratio_or(c.param_lo_text, 0.0);
            param_hi = parse_ratio_or(c.param_hi_text, 1.0);
            n_pts = c.result.n_pts;
            slice_log = c.log_scale;
            view.x_axis.name = axis_label_for_slot(idx);
            // LS shows every exponent as its own coloured line (parity with
            // draw_ls_plot in Parametric) — no exponent picker here; series
            // are built in the render block below.
        }

        if (!ok) {
            ImGui::TextDisabled("No data yet.");
            ImGui::PopID(); ImGui::End(); continue;
        }

        // X range for autofit (independent of point density).
        view.x_fit_use_explicit = true;
        view.x_fit_min = std::min(param_lo, param_hi);
        view.x_fit_max = std::max(param_lo, param_hi);
        // Лог-ось — как в Parametric; узлы движка лежат лог-равномерно, ось
        // и snap обязаны это повторить. log при lo<=0 невозможен (движок
        // такой Run отклоняет), но проверку держим здесь тоже — конфиг мог
        // прийти из старой сессии.
        view.x_axis.log_scale = slice_log && view.x_fit_min > 0.0;

        // Sweep-position crosshair — vertical line at fix_x (X-slice)
        // or fix_y (Y-slice), synced with the slider / drag on 2D heatmaps.
        // Hidden if Level 1D isn't the driver (falls out to NaN → no draw).
        view.crosshair_x = cs.shared.level_1d_enabled
                           ? lslots[i].fix_pt
                           : std::numeric_limits<double>::quiet_NaN();
        // Colour the vertical crosshair by which sweep this slice belongs
        // to, so the same colour on the 2D heatmap and its 1D slice tells
        // the eye which axis you're looking at.
        //  cfg_idx == 1 → X-slice → matches heatmap's vertical X-sweep line
        //  cfg_idx == 2 → Y-slice → matches heatmap's horizontal Y-sweep line
        view.crosshair_x_color = (lslots[i].cfg_idx == 2)
                                  ? 0xFFFF9028u   // orange = Y sweep
                                  : 0xFF50A0FFu;  // blue   = X sweep

        // Wire crosshair drag: MMB or Shift+LMB inside the plot moves the
        // corresponding fix_* value along the sweep axis of this slice.
        // Release triggers the shared auto-recompute (Level 1D + Phase)
        // via last_fix_{x,y}_change_time — same debounce path the L2D
        // heatmap drag and the fix sliders already go through, so all
        // three sources produce identical downstream behaviour.
        //  cfg_idx == 1 → X-slice sweeps X → drag updates fix_x
        //  cfg_idx == 2 → Y-slice sweeps Y → drag updates fix_y
        const bool slice_is_x = (lslots[i].cfg_idx == 1);
        // Snap crosshair drag to this slice's OWN grid nodes — the sampled
        // points the 1D compute actually produced — so the crosshair always
        // lands on a data point rather than drifting between them.
        // Captured by value (fresh lambda per frame, so no lifetime hazard).
        int   snap_n = parse_int_or(slice_is_x ? cs.shared.n_x_1d_text
                                              : cs.shared.n_y_1d_text, 2);
        if (snap_n < 2) snap_n = 2;
        const double snap_lo   = std::min(param_lo, param_hi);
        const double snap_hi   = std::max(param_lo, param_hi);
        const double snap_step = (snap_hi - snap_lo) / (double)(snap_n - 1);
        // При log-сетке узлы не равноудалены — индекс и позицию считаем по
        // логарифму (та же формула, что у движка), иначе крестик садится
        // между реальными точками.
        const bool snap_log = view.x_axis.log_scale && snap_lo > 0.0 && snap_hi > 0.0;
        // Конвенция узлов зависит от того, шёл ли срез continuation'ом: у него
        // своя формула (см. sweep_value_at). Флаг берём общий по вкладке — им
        // же apply_shared_to_bif1d проставляет continuation в под-конфиг.
        const bool snap_cont = cs.shared.continuation_1d_enabled;
        auto snap_to_grid = [snap_lo, snap_hi, snap_step, snap_n, snap_log, snap_cont](double w) {
            if (snap_step <= 0.0) return w;
            int i;
            if (snap_log) {
                if (!(w > 0.0)) return snap_lo;
                const double l0 = std::log10(snap_lo), l1 = std::log10(snap_hi);
                i = (int)std::round((std::log10(w) - l0) / ((l1 - l0) / (double)(snap_n - 1)));
            } else {
                i = (int)std::round((w - snap_lo) / snap_step);
            }
            i = std::clamp(i, 0, snap_n - 1);
            const double s = sweep_value_at(i, snap_n, snap_lo, snap_hi, snap_log,
                                            /*reverse*/ false, snap_cont);
            return std::clamp(s, snap_lo, snap_hi);
        };
        // Тот же snap и для тиков осей / hover-readout, что и в Parametric —
        // раньше в Custom он стоял только на crosshair-драге, поэтому tooltip
        // здесь показывал промежуточные X, которых в данных нет.
        apply_snap_x(view, snap_lo, snap_hi, snap_n);
        view.on_left_drag = [&cs, slice_is_x, snap_to_grid](double world_x) {
            double w = snap_to_grid(world_x);
            if (slice_is_x) {
                cs.shared.fix_x_value = w;
                cs.shared.last_fix_x_change_time = ImGui::GetTime();
            } else {
                cs.shared.fix_y_value = w;
                cs.shared.last_fix_y_change_time = ImGui::GetTime();
            }
        };
        view.on_left_click = [&cs, slice_is_x, snap_to_grid](double world_x) {
            double w = snap_to_grid(world_x);
            if (slice_is_x) {
                cs.shared.fix_x_value = w;
                cs.shared.last_fix_x_change_time = ImGui::GetTime();
            } else {
                cs.shared.fix_y_value = w;
                cs.shared.last_fix_y_change_time = ImGui::GetTime();
            }
        };

        // Build series. Bif = one scatter series with all peak samples;
        // LLE = one line; LS = N line series, one per exponent (parity
        // with draw_ls_plot).
        std::vector<PlotSeriesInput> series_in;
        std::vector<bool> init_vis, glob_vis;
        auto& bufs = l1_bufs[i];
        bufs.clear();

        int series_gen = data_gen;
        if (lslots[i].kind == L1Kind::Bif) {
            auto& d = cs.bif_session.diagrams[lslots[i].cfg_idx];
            const auto& r = d.result;
            // Как в Parametric: источник — peak_times при plot_inter_peaks,
            // иначе сами точки бифуркации. Раньше Custom всегда брал
            // bifurcation_points, из-за чего тоггл «inter-peaks» на него не влиял.
            const auto& source = d.plot_inter_peaks ? r.peak_times
                                                    : r.bifurcation_points;
            int n = r.n_pts > 0 ? r.n_pts : 1;
            bufs.emplace_back();
            auto& buf = bufs.back();
            for (int p = 0; p < (int)source.size(); ++p) {
                // Diverged-точки пропускаем (в Parametric так и было; здесь
                // проверки не было, и разошедшиеся точки попадали на график).
                if (p < (int)r.flags.size() && !regime_is_oscillation(r.flags[p])) continue;
                double px = sweep_value_at(p, n, param_lo, param_hi, slice_log,
                                           /*reverse*/ false, d.continuation);
                for (double y : source[p]) {
                    if (!std::isfinite(y)) continue;
                    buf.push_back((float)px);
                    buf.push_back((float)y);
                }
            }
            PlotSeriesInput si;
            si.points = buf.empty() ? nullptr : buf.data();
            si.n_points = (int)(buf.size() / 2);
            si.color = ic_base_color(0);
            if (d.custom_point_style) si.color.w = d.point_alpha;
            si.label = d.label.empty() ? "bd" : d.label;
            series_in.push_back(si);
            series_gen = data_gen * 2 + (d.plot_inter_peaks ? 1 : 0);
        } else if (lslots[i].kind == L1Kind::LLE) {
            auto& c = cs.lle_session.curves[lslots[i].cfg_idx];
            const auto& r = c.result;
            int n = (int)r.lyapunov.size();
            bufs.emplace_back();
            auto& buf = bufs.back();
            buf.reserve(n * 2);
            int total_pts = 0;
            for (int p = 0; p < n; ++p) {
                // Diverged / NaN пропускаем — как в Parametric. Раньше здесь
                // проверок не было, поэтому sentinel-значения расхождения
                // рисовались как выброс.
                if (p < (int)r.flags.size() && !regime_is_oscillation(r.flags[p])) continue;
                double y = r.lyapunov[p];
                if (!std::isfinite(y)) continue;
                double px = sweep_value_at(p, n, param_lo, param_hi, slice_log,
                                           /*reverse*/ false, c.continuation);
                buf.push_back((float)px);
                buf.push_back((float)y);
                ++total_pts;
            }
            PlotSeriesInput si;
            si.points = buf.empty() ? nullptr : buf.data();
            si.n_points = total_pts;
            si.color = ic_base_color(0);
            si.label = c.label.empty() ? "LLE" : c.label;
            series_in.push_back(si);
        } else { // LS — N series, one per exponent (parity with draw_ls_plot).
            auto& c = cs.ls_session.curves[lslots[i].cfg_idx];
            const auto& r = c.result;
            int n = r.n_pts;
            int N = r.n_exponents;
            if (N > 0 && n > 0 && (int)r.spectrum.size() == n) {
                bufs.reserve(N);
                for (int j = 0; j < N; ++j) {
                    bufs.emplace_back();
                    auto& buf = bufs.back();
                    buf.reserve(n * 2);
                    int total_pts = 0;
                    for (int p = 0; p < n; ++p) {
                        if (p < (int)r.flags.size() && !regime_is_oscillation(r.flags[p])) continue;
                        if (p >= (int)r.spectrum.size()) continue;
                        const auto& row = r.spectrum[p];
                        if (j >= (int)row.size()) continue;
                        double px = sweep_value_at(p, n, param_lo, param_hi, slice_log,
                                                   /*reverse*/ false, c.continuation);
                        double y = row[j];
                        if (!std::isfinite(y)) continue;
                        buf.push_back((float)px);
                        buf.push_back((float)y);
                        ++total_pts;
                    }
                    PlotSeriesInput si;
                    si.points = buf.empty() ? nullptr : buf.data();
                    si.n_points = total_pts;
                    si.color = ic_base_color(j);
                    // Как в Parametric: "<label> Lj" при N>1, иначе <label>.
                    const std::string base = c.label.empty() ? "LS" : c.label;
                    si.label = (N > 1) ? (base + " L" + std::to_string(j + 1)) : base;
                    series_in.push_back(si);
                }
                series_gen = data_gen * 64 + N;  // rebuild VBO if N changes
            }
        }
        init_vis.assign(series_in.size(), true);
        glob_vis.assign(series_in.size(), true);

        // Right-click "Export data..." — паритет с Parametric, где он есть у
        // всех 1D-графиков. В Custom его не было ни на одном из 6 слотов.
        view.popup_extras = [i, &lslots, &cs, &cb]() {
            const bool busy = (lslots[i].kind == L1Kind::Bif) ? cs.bif_session.in_flight
                            : (lslots[i].kind == L1Kind::LLE) ? cs.lle_session.in_flight
                                                              : cs.ls_session.in_flight;
            draw_export_menu_item(busy, cb, [i, &lslots, &cs](const std::string& path) {
                const int ci = lslots[i].cfg_idx;
                if (lslots[i].kind == L1Kind::Bif) {
                    if (ci >= 0 && ci < (int)cs.bif_session.diagrams.size())
                        data_export::export_bif1d(cs.bif_session.diagrams[ci].result, path);
                } else if (lslots[i].kind == L1Kind::LLE) {
                    if (ci >= 0 && ci < (int)cs.lle_session.curves.size())
                        data_export::export_lle1d(cs.lle_session.curves[ci].result, path);
                } else {
                    if (ci >= 0 && ci < (int)cs.ls_session.curves.size())
                        data_export::export_ls1d(cs.ls_session.curves[ci].result, path);
                }
            });
        };

        // Autofit whenever the underlying result changed (bif/lle/ls each
        // set fit_request in apply_*_result on completion of run_async).
        bool fit = false;
        if (lslots[i].kind == L1Kind::Bif) {
            auto& d = cs.bif_session.diagrams[lslots[i].cfg_idx];
            fit = d.fit_request; if (fit) d.fit_request = false;
        } else if (lslots[i].kind == L1Kind::LLE) {
            auto& c = cs.lle_session.curves[lslots[i].cfg_idx];
            fit = c.fit_request; if (fit) c.fit_request = false;
        } else {
            auto& c = cs.ls_session.curves[lslots[i].cfg_idx];
            fit = c.fit_request; if (fit) c.fit_request = false;
        }

        ImVec2 avail  = ImGui::GetContentRegionAvail();
        ImVec2 origin = ImGui::GetCursorScreenPos();
        view.render(rnd, origin, avail,
                    /*owner_id*/ (0xC10000 + i) ^ sys_owner_delta,
                    series_gen,
                    series_in, init_vis, glob_vis, fit);
        ImGui::PopID();
        ImGui::End();
    }

    // --- Level 3: phase controls live inside the L3 config panel (see
    // draw_level3_detail). Here we only spawn the projection windows so
    // 2D / 3D / TimeDomain plots dock alongside the other custom plots.
    if (cs.shared.level_phase_enabled && cs.shared.level3_kind == 0) {
        // Auto-place phase projections into the active tab on first appearance.
        // Suffix + owner_id delta both keep per-system isolation: suffix for
        // imgui.ini dock state, delta for the SHARED PlotRenderer cache
        // (otherwise Rossler's projection 0 saw Chen's cached FBO texture).
        // Диаграмма признаков рисуется в осях кластеризации того же 2D-конфига,
        // чьи eps/множители стоят в Shared config — иначе по ней нельзя судить,
        // что именно сольёт dbscan (см. FeatureClusterParams).
        FeatureClusterParams clust;
        if (!cs.bif_session.diagrams.empty()) {
            const auto& bd = cs.bif_session.diagrams[0];
            clust.valid         = true;
            clust.mult_peak     = parse_ratio_or(bd.mult_peak_text,     (double)mult_peak);
            clust.mult_interval = parse_ratio_or(bd.mult_interval_text, (double)mult_interval);
            clust.eps           = parse_ratio_or(bd.eps_dbscan_text,    0.1);
        }
        draw_projection_windows(cs.phase_session, cb,
            [&](int /*idx*/, const std::string& title) { ensure_docked(title); },
            {}, suffix, sys_owner_delta, {}, clust);
    }
    // --- Level 3 Basins window (unchanged HeatmapView minimal renderer) ---
    if (cs.shared.level_phase_enabled && cs.shared.level3_kind == 1) {
        std::string basins_title = "Custom Basins" + suffix;
        ensure_docked(basins_title);
        if (ImGui::Begin(basins_title.c_str())) {
                // Basins — HeatmapView of basin_idx (cluster id) — simpler
                // than draw_basins_plot's toolbar (feature switch stays in
                // detail panel), matches user request for a visible result.
                auto& bsn = cs.basins_session;
                if (bsn.in_flight) {
                    ImGui::TextDisabled("Computing basins...");
                } else if (bsn.configs.empty() || !bsn.configs[0].last_run_ok) {
                    ImGui::TextDisabled("No basins result yet.");
                } else {
                    // bsn_hv / bsn_renderer hoisted to top of function for
                    // system-change reset (see the reset block above).
                    const auto& bc = bsn.configs[0];
                    const auto& r  = bc.result;
                    if (r.n_pts <= 0 || r.basin_idx.empty()) {
                        ImGui::TextDisabled("Basins result empty.");
                    } else {
                        // Convert int cluster ids to double for HeatmapView.
                        static std::vector<double> bsn_values;
                        bsn_values.resize(r.basin_idx.size());
                        double vmin =  std::numeric_limits<double>::infinity();
                        double vmax = -std::numeric_limits<double>::infinity();
                        for (size_t k = 0; k < r.basin_idx.size(); ++k) {
                            double v = (double)r.basin_idx[k];
                            bsn_values[k] = v;
                            if (v < vmin) vmin = v;
                            if (v > vmax) vmax = v;
                        }
                        if (!std::isfinite(vmin)) { vmin = 0.0; vmax = 1.0; }
                        // Тот же тулбар, что и у остальных хитмап (раньше здесь
                        // был только combo Colormap — без autoscale/vmin/vmax
                        // и без Swap axes, хотя это ровно такая же диаграмма).
                        // Выбор colormap'а теперь персистится в config (слот 0
                        // = таб "Basins"); раньше он жил только в static-view
                        // и терялся при перезапуске.
                        {
                            auto& bcfg = bsn.configs[0];
                            const int cm = colormap_id_or(bcfg.colormap_idx[0],
                                                          model.basins_colormap);
                            bsn_hv.colormap = (HeatmapColormap)cm;

                            HeatmapToolbarOpts topts;
                            topts.persist_colormap =
                                [&bcfg](int picked) { bcfg.colormap_idx[0] = picked; };
                            draw_heatmap_toolbar(bsn_hv, topts);
                        }
                        bsn_hv.x_axis.name = (bc.axis_x_var >= 0 && bc.axis_x_var < (int)cs.vars.size())
                                              ? cs.vars[bc.axis_x_var] : "x";
                        bsn_hv.y_axis.name = (bc.axis_y_var >= 0 && bc.axis_y_var < (int)cs.vars.size())
                                              ? cs.vars[bc.axis_y_var] : "y";
                        ImVec2 avail  = ImGui::GetContentRegionAvail();
                        ImVec2 origin = ImGui::GetCursorScreenPos();
                        bsn_hv.render(bsn_renderer, origin, avail,
                                      /*owner_id*/ 0xCBA51E5 ^ sys_owner_delta,
                                      bc.data_generation,
                                      r.n_pts, r.n_pts,
                                      bsn_values.data(),
                                      r.axis_x_lo, r.axis_x_hi,
                                      r.axis_y_lo, r.axis_y_hi,
                                      vmin, vmax,
                                      /*fit_request*/ false);
                    }
                }
            }
        ImGui::End();
    }
}

// ============================================================================
// Индикатор компьюта в верхней строке.
//
// Раньше состояние снималось лестницей из двенадцати почти одинаковых веток:
// каждая доставала из своей сессии одну и ту же четвёрку — подпись
// запущенного конфига, время старта, cancel_token, progress_token. Ветки
// успели разойтись: три подставляли запасное имя только при выходе индекса за
// границы, три — ещё и при пустом label.
// ============================================================================
enum class BusyKind { None, Bif, LLE, LS, Dft1D, Basins, Phase, FastSync, Custom };

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
[[nodiscard]] static std::string running_label(const Vec& items, int idx, const char* fallback) {
    if (idx >= 0 && idx < (int)items.size() && !items[idx].label.empty())
        return items[idx].label;
    return fallback;
}

// Снять состояние с сессии. Годится для любой, у которой есть
// compute_start_time / cancel_token / progress_token — то есть для всех, кроме
// phase: у неё нет ни отмены, ни прогресса, и её ветка заполняется вручную.
template <class Session>
[[nodiscard]] static BusyInfo busy_from(const Session& s, BusyKind kind, std::string what) {
    BusyInfo b;
    b.kind  = kind;
    b.what  = std::move(what);
    b.start = s.compute_start_time;
    b.cancelling = s.cancel_token && s.cancel_token->load(std::memory_order_relaxed);
    if (s.progress_token)
        b.progress = s.progress_token->load(std::memory_order_relaxed);
    return b;
}

// ============================================================
// Главное окно: переключатель режимов Library / Analysis / Parametric
// ============================================================
void draw_gui(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) {
    // полноэкранный dockspace-хост
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::SetNextWindowViewport(vp->ID);
    ImGuiWindowFlags host_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::Begin("MainHost", nullptr, host_flags);
    ImGui::PopStyleVar(2);

    // custom_schemes — единственное поле, которое может отредактироваться
    // в System tab БЕЗ переключения режима (т.е. без start_*_analysis).
    // Чтобы scheme combo в Phase/Parametric/Basins/FastSync увидел свежий
    // список сразу после "+ Add custom scheme" ИЛИ правки тела существующей
    // схемы, синкаем копию live → сессии каждый кадр. Раньше Basins и
    // FastSync были пропущены: Basins полностью, FastSync синкался только
    // когда его окно активно (см. draw_fastsync_controls). Из-за этого
    // отредактированное тело cs.body не доходило до compute_krs_for_scheme
    // на момент Run, и NVRTC брал устаревший body из кеша / запускал старый.
    model.phase_session.custom_schemes       = model.custom_schemes;
    model.bifurcation_session.custom_schemes = model.custom_schemes;
    model.lle_session.custom_schemes         = model.custom_schemes;
    model.ls_session.custom_schemes          = model.custom_schemes;
    model.dft1d_session.custom_schemes       = model.custom_schemes;
    model.basins_session.custom_schemes      = model.custom_schemes;
    model.fastsync_session.custom_schemes    = model.custom_schemes;
    // Custom tab owns 5 isolated sub-sessions — same per-frame sync applies.
    model.custom_session.custom_schemes                = model.custom_schemes;
    model.custom_session.bif_session.custom_schemes    = model.custom_schemes;
    model.custom_session.lle_session.custom_schemes    = model.custom_schemes;
    model.custom_session.ls_session.custom_schemes     = model.custom_schemes;
    model.custom_session.phase_session.custom_schemes  = model.custom_schemes;
    model.custom_session.basins_session.custom_schemes = model.custom_schemes;

    // Auto-labels: pre-frame refresh so all label-consumers (tab-bar names,
    // plot legend, window title, Plot windows section) see the same value.
    refresh_auto_labels(model);

    // poll'им async-расчёты независимо от текущего режима — чтобы при
    // возврате в этот режим пользователь сразу увидел готовый результат
    if (model.bifurcation_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_parametric",
                             session_to_json_parametric(model.bifurcation_session));
    }
    if (model.lle_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_lle",
                             session_to_json_lle(model.lle_session));
    }
    if (model.ls_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_ls",
                             session_to_json_ls(model.ls_session));
    }
    // Plot windows list is UI-only (not tied to any session's poll()) — save
    // whenever add/remove/membership-edit touched it this frame, rather than
    // every frame.
    if (model.parametric_plot_windows_dirty) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_parametric_windows",
                             session_to_json_parametric_windows(model.parametric_plot_windows));
        model.parametric_plot_windows_dirty = false;
    }
    // DFT1D: independent multi-config session (own queue, own poll) — same
    // save-after-poll + dirty-plot-windows pattern as above.
    if (model.dft1d_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_dft1d",
                             session_to_json_dft1d(model.dft1d_session));
    }
    if (model.dft1d_plot_windows_dirty) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_dft1d_windows",
                             session_to_json_dft1d_windows(model.dft1d_plot_windows));
        model.dft1d_plot_windows_dirty = false;
    }
    if (model.phase_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last",
                             session_to_json(model.phase_session));
    }
    // Basins: один config на сессию. Сохраняем JSON каждый кадр (после poll
    // - но также при изменении полей в controls). Здесь только after-poll save.
    if (model.basins_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_basins",
                             session_to_json_basins(model.basins_session));
    }
    // Фазовые портреты по бассейнам: свой poll + свой тик очереди. Идут
    // независимо от текущего режима — вернувшись в Basins, пользователь сразу
    // видит готовый результат (как и у всех остальных сессий).
    if (model.basins_session.poll_phase()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_basins",
                             session_to_json_basins(model.basins_session));
    }
    basins_phase_tick(model);
    model.basins_session.start_next_phase();
    if (model.basins_session.phase_settings_dirty) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_basins",
                             session_to_json_basins(model.basins_session));
        model.basins_session.phase_settings_dirty = false;
    }
    // FastSync: poll worker future; on completion persist session JSON.
    // Без этого вызова in_flight никогда не сбрасывается → "Running" висит вечно.
    if (model.fastsync_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_fastsync",
                             session_to_json_fastsync(model.fastsync_session));
    }
    // Custom tab: aggregate poll of all 5 sub-sessions; one bundle save on
    // any completion so we don't rewrite _last_custom.json five times.
    // Also save on any workspace mutation (add/close/rename tab, splitter
    // drag, cross-tab window move) — the compute-completion save alone
    // meant a closed tab could resurrect after a plain app restart.
    bool custom_dirty = model.custom_session.poll_all();
    if (model.custom_session.workspace.dirty) {
        custom_dirty = true;
        model.custom_session.workspace.dirty = false;
    }
    if (custom_dirty) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_custom",
                             session_to_json_custom(model.custom_session));
        // When the whole pipeline is drained (queue empty AND nothing left
        // running), promote pending → committed for each level's signature
        // so the next Run knows which levels are still up-to-date.
        if (model.custom_queue.empty() && !model.custom_session.any_in_flight())
            model.custom_session.commit_pending_signatures();
    }

    // Tick parametric-очереди: если ни одна из BD/LLE/LS не in_flight и в
    // очереди есть элементы — берём следующий и стартуем. start_next сам
    // проверяет условие и безопасен к вызову каждый кадр.
    model.start_next_in_parametric_queue();
    // То же для dft1d-очереди (независимая).
    model.start_next_in_dft1d_queue();
    // То же для basins-очереди (независимая).
    model.start_next_in_basins_queue();
    // То же для fastsync-очереди (независимая).
    model.start_next_in_fastsync_queue();
    // Custom tab has its own queue (2D → 1D → Phase/Basins pipeline).
    model.start_next_in_custom_queue();

    // переключатель режимов
    int mode = (int)model.app_mode;
    ImGui::RadioButton("Library", &mode, (int)AppModel::AppMode::Library); ImGui::SameLine();
    ImGui::RadioButton("Phase analysis", &mode, (int)AppModel::AppMode::Analysis); ImGui::SameLine();
    ImGui::RadioButton("Parametric", &mode, (int)AppModel::AppMode::Parametric); ImGui::SameLine();
    ImGui::RadioButton("1D DFT", &mode, (int)AppModel::AppMode::Dft1D); ImGui::SameLine();
    ImGui::RadioButton("Basins", &mode, (int)AppModel::AppMode::Basins); ImGui::SameLine();
    ImGui::RadioButton("Fast Synchro", &mode, (int)AppModel::AppMode::FastSync); ImGui::SameLine();
    ImGui::RadioButton("Custom", &mode, (int)AppModel::AppMode::Custom); ImGui::SameLine();
    ImGui::RadioButton("Settings", &mode, (int)AppModel::AppMode::Settings);

    // Битый файл сессии. Висит до следующей УСПЕШНОЙ загрузки, а не до конца
    // кадра: иначе сообщение о том, что настройки не восстановились, мелькнуло
    // бы один раз при старте и пропало. Раньше эта ошибка не показывалась
    // вообще — результат разбора игнорировался во всех точках вызова.
    if (!model.session_load_warning.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.3f, 1.0f), "%s",
                           model.session_load_warning.c_str());
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("The saved session file could not be parsed and was ignored.\n"
                              "Settings from the system record are in use instead.\n"
                              "Running an analysis and switching systems will overwrite the file.");
    }

    // Индикатор компьюта — справа по правой границе окна, виден во всех режимах.
    // Layout: [text] [progress bar] [Stop] for in-flight cancellable sessions;
    // [text] only for phase or for "Done/Cancelled" persistent state. Stop also
    // drains parametric_queue and basins_queue so remaining batch items
    // don't auto-start.
    BusyInfo busy;                    // подпись/старт/отмена/прогресс — см. busy_from
    bool   show_done    = false;      // not in flight — show persistent last-run info
    bool   last_ok      = true;       // for show_done: true = green "Done", false = red "Cancelled"
    double done_seconds = 0.0;
    int    basins_phase = 0;          // 1 = sim, 2 = cluster; 0 = not basins or not started

    {
        auto& bif = model.bifurcation_session;
        auto& lle = model.lle_session;
        auto& lsp = model.ls_session;
        auto& dft = model.dft1d_session;
        auto& bas = model.basins_session;
        auto& fsy = model.fastsync_session;
        auto& cus = model.custom_session;

    if (bif.in_flight)
        busy = busy_from(bif, BusyKind::Bif,
                         running_label(bif.diagrams, bif.running_diagram_index, "bifurcation"));
    else if (lle.in_flight)
        busy = busy_from(lle, BusyKind::LLE,
                         running_label(lle.curves, lle.running_curve_index, "LLE"));
    else if (lsp.in_flight)
        busy = busy_from(lsp, BusyKind::LS,
                         running_label(lsp.curves, lsp.running_curve_index, "LS"));
    else if (dft.in_flight)
        busy = busy_from(dft, BusyKind::Dft1D,
                         running_label(dft.configs, dft.running_config_index, "dft1d"));
    else if (model.phase_session.in_flight) {
        // У phase нет ни отмены, ни прогресса — только время старта.
        busy.kind  = BusyKind::Phase;
        busy.what  = "phase";
        busy.start = model.phase_session.compute_start_time;
    }
    else if (bas.in_flight) {
        busy = busy_from(bas, BusyKind::Basins,
                         running_label(bas.configs, bas.running_config_index, "basins"));
        // Своя, только у Basins: 1 = моделирование, 2 = кластеризация.
        if (bas.progress_phase_token)
            basins_phase = bas.progress_phase_token->load(std::memory_order_relaxed);
    }
    else if (fsy.in_flight)
        busy = busy_from(fsy, BusyKind::FastSync,
                         running_label(fsy.configs, fsy.running_config_index, "fastsync"));
    else if (cus.any_in_flight()) {
        // Custom-tab: queue is drained serially, so at most one sub-session is
        // in-flight at any time — pick whichever one it is and surface its
        // label/progress under a "Custom: ..." prefix.
        if (cus.bif_session.in_flight)
            busy = busy_from(cus.bif_session, BusyKind::Custom,
                             "Custom: " + running_label(cus.bif_session.diagrams,
                                                        cus.bif_session.running_diagram_index, "Bif"));
        else if (cus.lle_session.in_flight)
            busy = busy_from(cus.lle_session, BusyKind::Custom,
                             "Custom: " + running_label(cus.lle_session.curves,
                                                        cus.lle_session.running_curve_index, "LLE"));
        else if (cus.ls_session.in_flight)
            busy = busy_from(cus.ls_session, BusyKind::Custom,
                             "Custom: " + running_label(cus.ls_session.curves,
                                                        cus.ls_session.running_curve_index, "LS"));
        else if (cus.phase_session.in_flight) {
            busy.what  = "Custom: phase";
            busy.start = cus.phase_session.compute_start_time;
        }
        else if (cus.basins_session.in_flight)
            busy = busy_from(cus.basins_session, BusyKind::Custom,
                             "Custom: " + running_label(cus.basins_session.configs,
                                                        cus.basins_session.running_config_index, "basins"));
        busy.kind = BusyKind::Custom;   // в том числе для phase-ветки выше
    }
    else {
        // Nothing in flight — pick the session whose last run finished most
        // recently (across the 4 cancellable ones) and show persistent info.
        struct DoneCand { BusyKind kind; std::chrono::steady_clock::time_point ts; const std::string* label; bool ok; double secs; };
        DoneCand candidates[5] = {
            { BusyKind::Bif,    model.bifurcation_session.last_run_completed_at,
              &model.bifurcation_session.last_run_label,
              model.bifurcation_session.last_run_succeeded,
              model.bifurcation_session.last_run_seconds },
            { BusyKind::LLE,    model.lle_session.last_run_completed_at,
              &model.lle_session.last_run_label,
              model.lle_session.last_run_succeeded,
              model.lle_session.last_run_seconds },
            { BusyKind::LS,     model.ls_session.last_run_completed_at,
              &model.ls_session.last_run_label,
              model.ls_session.last_run_succeeded,
              model.ls_session.last_run_seconds },
            { BusyKind::Dft1D,  model.dft1d_session.last_run_completed_at,
              &model.dft1d_session.last_run_label,
              model.dft1d_session.last_run_succeeded,
              model.dft1d_session.last_run_seconds },
            { BusyKind::Basins, model.basins_session.last_run_completed_at,
              &model.basins_session.last_run_label,
              model.basins_session.last_run_succeeded,
              model.basins_session.last_run_seconds },
        };
        const DoneCand* best = nullptr;
        for (const auto& c : candidates) {
            if (c.label->empty()) continue;
            if (!best || c.ts > best->ts) best = &c;
        }
        if (best) {
            busy.what    = *best->label;
            busy.kind    = best->kind;
            show_done    = true;
            last_ok      = best->ok;
            done_seconds = best->secs;
        }
    }
    }   // область видимости ссылок на сессии

    if (busy.kind != BusyKind::None) {
        char text[200];
        // Phase suffix appears only for basins while running (not on "Cancelling"
        // or "Done" — those reflect overall state, not the current sub-phase).
        const char* phase_suffix = "";
        if (busy.kind == BusyKind::Basins && !show_done && !busy.cancelling) {
            if      (basins_phase == 1) phase_suffix = " (1/2 sim)";
            else if (basins_phase == 2) phase_suffix = " (2/2 cluster)";
        }
        if (show_done) {
            std::snprintf(text, sizeof(text), "%s %s in %.1fs",
                          last_ok ? "Done" : "Cancelled",
                          busy.what.c_str(), done_seconds);
        } else if (busy.cancelling) {
            double secs = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - busy.start).count();
            std::snprintf(text, sizeof(text), "Cancelling %s... %.1fs", busy.what.c_str(), secs);
        } else {
            double secs = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - busy.start).count();
            // Queue suffix: prefer the queue that matches the running session
            // (parametric for Bif/LLE/LS, basins for Basins, fastsync for FastSync).
            size_t queue_n = 0;
            if      (busy.kind == BusyKind::Basins)   queue_n = model.basins_queue.size();
            else if (busy.kind == BusyKind::FastSync) queue_n = model.fastsync_queue.size();
            else if (busy.kind == BusyKind::Dft1D)    queue_n = model.dft1d_queue.size();
            else if (busy.kind == BusyKind::Custom)   queue_n = model.custom_queue.size();
            else                                      queue_n = model.parametric_queue.size();
            if (queue_n > 0)
                std::snprintf(text, sizeof(text), "Computing %s%s... %.1fs (+%zu)",
                              busy.what.c_str(), phase_suffix, secs, queue_n);
            else
                std::snprintf(text, sizeof(text), "Computing %s%s... %.1fs",
                              busy.what.c_str(), phase_suffix, secs);
        }

        const bool show_stop = (busy.kind != BusyKind::Phase) &&
                               !show_done && !busy.cancelling;
        const bool show_bar  = show_stop;  // bar only when running & not cancelling

        const float pad      = ImGui::GetStyle().ItemSpacing.x;
        const float bar_w    = 120.0f;
        const float bar_h    = ImGui::GetTextLineHeight();
        const float text_w   = ImGui::CalcTextSize(text).x;
        const float stop_w   = show_stop
                               ? (ImGui::CalcTextSize("Stop").x +
                                  ImGui::GetStyle().FramePadding.x * 2.0f)
                               : 0.0f;
        float total_w = text_w + 12.0f;
        if (show_bar)  total_w += bar_w + pad;
        if (show_stop) total_w += stop_w + pad;

        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetWindowSize().x - total_w);

        ImVec4 col;
        if (show_done) {
            col = last_ok ? ImVec4(0.55f, 0.95f, 0.55f, 1.0f)   // green
                          : ImVec4(0.95f, 0.45f, 0.45f, 1.0f);  // red
        } else {
            col = ImVec4(1.0f, 0.85f, 0.25f, 1.0f);              // yellow (running/cancelling)
        }
        ImGui::TextColored(col, "%s", text);

        if (show_bar) {
            ImGui::SameLine();
            const float f = std::clamp(busy.progress, 0.0f, 1.0f);
            ImGui::ProgressBar(f, ImVec2(bar_w, bar_h), "");
        }
        if (show_stop) {
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.55f, 0.15f, 0.15f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.75f, 0.20f, 0.20f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.90f, 0.25f, 0.25f, 1.0f));
            if (ImGui::SmallButton("Stop")) {
                switch (busy.kind) {
                    case BusyKind::Bif:      model.bifurcation_session.request_cancel(); break;
                    case BusyKind::LLE:      model.lle_session.request_cancel();         break;
                    case BusyKind::LS:       model.ls_session.request_cancel();          break;
                    case BusyKind::Dft1D:    model.dft1d_session.request_cancel();       break;
                    case BusyKind::Basins:   model.basins_session.request_cancel();      break;
                    case BusyKind::FastSync: model.fastsync_session.request_cancel();    break;
                    case BusyKind::Custom:   model.custom_session.request_cancel_all();  break;
                    default: break;
                }
                // Drain all batch queues so remaining items don't auto-start.
                model.parametric_queue.clear();
                model.dft1d_queue.clear();
                model.basins_queue.clear();
                model.fastsync_queue.clear();
                model.custom_queue.clear();
            }
            ImGui::PopStyleColor(3);
        }
    }

    // Global system picker — centered on the top-bar. One combo replaces the
    // five per-tab System: combos that used to live in each draw_*_controls.
    // Disabled while ANY session is in-flight so a switch can't race a worker
    // that will apply its result to the already-swapped session.
    {
        bool any_in_flight =
              model.phase_session.in_flight
           || model.bifurcation_session.in_flight
           || model.lle_session.in_flight
           || model.ls_session.in_flight
           || model.dft1d_session.in_flight
           || model.basins_session.in_flight
           || model.fastsync_session.in_flight
           || model.custom_session.any_in_flight();
        const float combo_w = 240.0f;
        std::string preview = model.name.empty() ? std::string("(select system)") : model.name;
        float cx = (ImGui::GetWindowSize().x - combo_w) * 0.5f;
        ImGui::SameLine();
        ImGui::SetCursorPosX(cx);
        ImGui::SetNextItemWidth(combo_w);
        if (any_in_flight) ImGui::BeginDisabled();
        if (ImGui::BeginCombo("##topsyssel", preview.c_str())) {
            for (const auto& nm : lib.list()) {
                if (ImGui::Selectable(nm.c_str(), model.name == nm))
                    apply_system_switch(model, lib, nm);
            }
            ImGui::EndCombo();
        }
        if (any_in_flight) ImGui::EndDisabled();
    }
    // При входе в Analysis/Parametric решаем, нужно ли (пере)инициализировать
    // сессию. Init происходит когда:
    //   1) система сменилась относительно той, для которой session была собрана;
    //   2) session ещё ни разу не была инициализирована (vars пустой) — это
    //      случай несохранённых систем, где model.name = loaded_system_name = "";
    //   3) у model сменился алфавит / vars_text, и session.vars/params уже
    //      не совпадают с актуальным model.known_vars/known_params.
    // Случай 3 раньше требовал перезапуска приложения, чтобы подхватить новый
    // алфавит — теперь подхватывается при следующем входе в режим.
    bool entering_phase  = (AppModel::AppMode)mode == AppModel::AppMode::Analysis &&
                           model.app_mode != AppModel::AppMode::Analysis;
    bool entering_par    = (AppModel::AppMode)mode == AppModel::AppMode::Parametric &&
                           model.app_mode != AppModel::AppMode::Parametric;
    bool entering_dft1d  = (AppModel::AppMode)mode == AppModel::AppMode::Dft1D &&
                           model.app_mode != AppModel::AppMode::Dft1D;
    bool entering_basins = (AppModel::AppMode)mode == AppModel::AppMode::Basins &&
                           model.app_mode != AppModel::AppMode::Basins;
    bool entering_fastsync = (AppModel::AppMode)mode == AppModel::AppMode::FastSync &&
                             model.app_mode != AppModel::AppMode::FastSync;
    bool entering_custom = (AppModel::AppMode)mode == AppModel::AppMode::Custom &&
                           model.app_mode != AppModel::AppMode::Custom;
    if (entering_phase || entering_par || entering_dft1d || entering_basins || entering_fastsync || entering_custom) {
        // обновим known_vars/known_params из живого алфавита, чтобы сравнение
        // ниже было против актуального состояния
        model.refresh_symbols();
    }
    auto phase_need_init = model.phase_session.loaded_system_name != model.name
                        || model.phase_session.vars.empty()
                        || model.phase_session.vars   != model.known_vars
                        || model.phase_session.params != model.known_params;
    auto par_need_init   = model.bifurcation_session.loaded_system_name != model.name
                        || model.bifurcation_session.vars.empty()
                        || model.bifurcation_session.vars   != model.known_vars
                        || model.bifurcation_session.params != model.known_params;
    if (entering_phase && phase_need_init) {
        model.start_phase_analysis();
        if (!model.loaded_name.empty()) {
            std::string j = lib.load_session(model.loaded_name, "_last");
            apply_session_json(model, j, model.phase_session, session_from_json, "_last");
        }
    }
    if (entering_par && par_need_init) {
        model.start_parametric_analysis();
        if (!model.loaded_name.empty()) {
            std::string j = lib.load_session(model.loaded_name, "_last_parametric");
            apply_session_json(model, j, model.bifurcation_session, session_from_json_parametric, "_last_parametric");
            std::string jl = lib.load_session(model.loaded_name, "_last_lle");
            apply_session_json(model, jl, model.lle_session, session_from_json_lle, "_last_lle");
            std::string js = lib.load_session(model.loaded_name, "_last_ls");
            apply_session_json(model, js, model.ls_session, session_from_json_ls, "_last_ls");
            std::string jw = lib.load_session(model.loaded_name, "_last_parametric_windows");
            model.load_or_init_parametric_plot_windows(jw);
        }
    }
    auto dft1d_need_init = model.dft1d_session.loaded_system_name != model.name
                        || model.dft1d_session.vars.empty()
                        || model.dft1d_session.vars   != model.known_vars
                        || model.dft1d_session.params != model.known_params;
    if (entering_dft1d && dft1d_need_init) {
        model.start_dft1d_analysis();
        if (!model.loaded_name.empty()) {
            std::string jd = lib.load_session(model.loaded_name, "_last_dft1d");
            apply_session_json(model, jd, model.dft1d_session, session_from_json_dft1d, "_last_dft1d");
            std::string jw = lib.load_session(model.loaded_name, "_last_dft1d_windows");
            model.load_or_init_dft1d_plot_windows(jw);
        }
    }
    auto basins_need_init = model.basins_session.loaded_system_name != model.name
                         || model.basins_session.vars.empty()
                         || model.basins_session.vars   != model.known_vars
                         || model.basins_session.params != model.known_params;
    if (entering_basins && basins_need_init) {
        model.start_basins_analysis();
        if (!model.loaded_name.empty()) {
            std::string jb = lib.load_session(model.loaded_name, "_last_basins");
            apply_session_json(model, jb, model.basins_session, session_from_json_basins, "_last_basins");
        }
    }
    auto fastsync_need_init = model.fastsync_session.loaded_system_name != model.name
                           || model.fastsync_session.vars.empty()
                           || model.fastsync_session.vars   != model.known_vars
                           || model.fastsync_session.params != model.known_params;
    if (entering_fastsync && fastsync_need_init) {
        model.start_fastsync_analysis();
        if (!model.loaded_name.empty()) {
            std::string jf = lib.load_session(model.loaded_name, "_last_fastsync");
            apply_session_json(model, jf, model.fastsync_session, session_from_json_fastsync, "_last_fastsync");
        }
    }
    auto custom_need_init = model.custom_session.loaded_system_name != model.name
                         || model.custom_session.vars.empty()
                         || model.custom_session.vars   != model.known_vars
                         || model.custom_session.params != model.known_params;
    if (entering_custom && custom_need_init) {
        // Hard reset — mirrors the reset in apply_system_switch for the
        // Custom case. Prevents stale sub-session results / signature cache
        // from a previously-loaded system leaking into the fresh init.
        model.custom_session = CustomSession{};
        model.start_custom_analysis();
        if (!model.loaded_name.empty()) {
            std::string jc = lib.load_session(model.loaded_name, "_last_custom");
            apply_session_json(model, jc, model.custom_session, session_from_json_custom, "_last_custom");
        }
    }
    // Persist AppMode change so the next launch restores this tab. Compare
    // BEFORE overwriting so we only write on real transitions (not on every
    // frame while sitting in the same tab).
    if ((AppModel::AppMode)mode != model.app_mode) {
        AppConfig cfg;
        load_app_config(get_exe_dir_with_sep(), cfg);
        cfg.last_app_mode = mode;
        save_app_config(get_exe_dir_with_sep(), cfg);
    }
    model.app_mode = (AppModel::AppMode)mode;
    ImGui::Separator();

    // Custom mode has its own two-pane layout (Controls | Workspace, split by
    // a draggable splitter) instead of the shared MainDockspace. Capture the
    // region below the AppMode radios so that layout can occupy exactly it.
    ImVec2 custom_area_pos{}, custom_area_size{};
    if (model.app_mode == AppModel::AppMode::Custom) {
        custom_area_pos  = ImGui::GetCursorScreenPos();
        custom_area_size = ImGui::GetContentRegionAvail();
    } else {
        // dockspace для содержимого — используется всеми режимами, кроме Custom.
        ImGuiID dockspace_id = ImGui::GetID("MainDockspace");
        ImGui::DockSpace(dockspace_id, ImVec2(0, 0), ImGuiDockNodeFlags_None);
    }

    ImGui::End(); // MainHost

    if (model.app_mode == AppModel::AppMode::Library) {
        // Library mode: list view by default, editor view (System/Parameters
        // sub-tabs + Save/Cancel) after Edit or Add new. Top-level tab
        // switching is safe during edit — the editor works on
        // model.library_edit_buffer, a scratch AppModel, so it never touches
        // the system currently active in Parametric/Phase/Basins/FastSync.
        if (ImGui::Begin("Editor")) {
            if (model.library_edit_mode == AppModel::LibraryEditMode::None)
                draw_library_list(model, lib);
            else
                draw_library_editor(model, lib, cb);
        }
        ImGui::End();
    }
    else if (model.app_mode == AppModel::AppMode::Analysis) {
        // режим анализа: панель настроек + окна проекций (докаются пользователем)
        if (ImGui::Begin("Controls")) {
            draw_phase_controls(model.phase_session, [&model, &lib]() {
                if (!model.loaded_name.empty()) {
                    model.from_record(lib.load(model.loaded_name));   // reference from disk
                    model.start_phase_analysis();
                }
            });
        }
        ImGui::End();
        draw_projection_windows(model.phase_session, cb);
    }
    else if (model.app_mode == AppModel::AppMode::Parametric) {
        // Parametric mode: controls window (Bif/LLE/LS config tabs + "Plot
        // windows" management, unchanged in shape) + a dynamic list of plot
        // windows the user opens/closes/docks freely, like Phase's
        // projection windows.
        if (ImGui::Begin("Parametric Controls")) {
            draw_parametric_controls(model, lib);
        }
        ImGui::End();
        draw_parametric_plot_windows(model, lib, cb);
    }
    else if (model.app_mode == AppModel::AppMode::Dft1D) {
        // DFT1D mode: controls window (system picker + Run/Run all + config
        // tab bar + Plot windows manager) + a dynamic list of heatmap plot
        // windows, mirrors Parametric mode's shape.
        if (ImGui::Begin("DFT1D Controls")) {
            draw_dft1d_controls(model, lib);
        }
        ImGui::End();
        draw_dft1d_plot_windows(model, lib, cb);
    }
    else if (model.app_mode == AppModel::AppMode::Basins) {
        if (ImGui::Begin("Basins Controls")) {
            draw_basins_controls(model, lib);
        }
        ImGui::End();
        if (ImGui::Begin("Basins of Attraction")) {
            draw_basins_plot(model, lib, cb);
        }
        ImGui::End();
        // Окна фазовых портретов активного config'а (2D / time domain).
        draw_basins_phase_windows(model, cb);
    }
    else if (model.app_mode == AppModel::AppMode::FastSync) {
        if (ImGui::Begin("FastSync Controls")) {
            draw_fastsync_controls(model, lib);
        }
        ImGui::End();
        if (ImGui::Begin("Fast Synchro")) {
            draw_fastsync_plot(model, cb);
        }
        ImGui::End();
    }
    else if (model.app_mode == AppModel::AppMode::Custom) {
        // Custom mode: split-region layout (Controls | Workspace) drawn into
        // the area captured above. Plot windows live in per-tab dockspaces
        // inside the Workspace panel and are moved between tabs via native
        // ImGui drag&drop (see draw_custom_mode_layout).
        draw_custom_mode_layout(model, lib, cb, custom_area_pos, custom_area_size);
    }
    else { // AppMode::Settings
        // Settings состоит из независимых виджетов, а save_app_config пишет
        // файл целиком — поэтому read-modify-write, как в draw_top_bar. Раньше
        // каждый обработчик собирал AppConfig с нуля и терял last_app_mode /
        // last_system_name (а слайдер UI scale и чекбокс шрифта — ещё и
        // dark_theme: тема сбрасывалась в Dark при правке масштаба).
        auto persist_settings = [](const AppModel& m) {
            AppConfig cfg;
            load_app_config(get_exe_dir_with_sep(), cfg);
            cfg.ui_scale_override      = m.ui_scale_override;
            cfg.use_builtin_font       = m.use_builtin_font;
            cfg.heatmap_colormap       = m.heatmap_colormap;
            cfg.basins_colormap        = m.basins_colormap;
            cfg.basins_avgpk_colormap  = m.basins_avgpk_colormap;
            cfg.basins_avgint_colormap = m.basins_avgint_colormap;
            cfg.basins_states_colormap = m.basins_states_colormap;
            cfg.slancm_enabled         = m.slancm_enabled;
            cfg.tick_precision         = m.tick_precision;
            cfg.dark_theme             = m.dark_theme;
            cfg.peak                   = m.peak;
            cfg.nvrtc_fmad             = m.nvrtc_fmad;
            save_app_config(get_exe_dir_with_sep(), cfg);
        };

        if (ImGui::Begin("Settings")) {
            ImGui::Text("Interface scale");
            ImGui::TextDisabled("Auto-detected at startup from glfwGetMonitorContentScale.");
            ImGui::TextDisabled("Override persists in _app_config.json next to exe.");
            ImGui::Separator();

            ImGui::TextUnformatted("UI scale:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(220);
            // Применяем НЕ во время drag'а, а на отпускание (IsItemDeactivatedAfterEdit) —
            // иначе UI пересобирается на каждом кадре, виджет уходит из-под курсора.
            static float ui_slider_value = -1.0f;
            if (ui_slider_value < 0.0f) ui_slider_value = model.effective_ui_scale();
            if (!ImGui::IsAnyItemActive())
                ui_slider_value = model.effective_ui_scale();
            ImGui::SliderFloat("##ui_scale", &ui_slider_value, 0.5f, 3.0f, "%.2fx");
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                model.ui_scale_override = ui_slider_value;
                persist_settings(model);
            }
            ImGui::SameLine();
            if (ImGui::Button("Auto")) {
                model.ui_scale_override = 0.0f;
                ui_slider_value = model.ui_scale_auto;
                persist_settings(model);
            }
            ImGui::TextDisabled("Auto detected: %.2fx   |   Override: %s",
                model.ui_scale_auto,
                model.ui_scale_override > 0 ?
                    (std::to_string(model.ui_scale_override) + "x").c_str() :
                    "(off)");

            ImGui::Separator();
            ImGui::Text("Font");
            bool use_builtin = model.use_builtin_font;
            if (ImGui::Checkbox("Use built-in font (ProggyClean)", &use_builtin)) {
                model.use_builtin_font = use_builtin;
                persist_settings(model);
            }
            ImGui::TextDisabled("Off: Windows Segoe UI TTF (recommended, crisp at any scale).");
            ImGui::TextDisabled("On: built-in bitmap ProggyClean (compact, pixel-perfect at 1x/2x/3x).");

            ImGui::Separator();
            ImGui::Text("Axes");
            int tp = model.tick_precision;
            ImGui::SetNextItemWidth(220);
            if (ImGui::SliderInt("Tick precision (digits)", &tp, 2, 10)) {
                model.tick_precision = tp;
                set_tick_precision(tp);
                persist_settings(model);
            }
            ImGui::TextDisabled("Significant digits in axis tick and colorbar labels.");

            ImGui::Separator();
            ImGui::Text("Theme");
            // Радио по Dark/Light. apply_ui_scale в app_main.cpp ловит изменение
            // через applied_dark_theme и пересобирает style + scale.
            int theme_idx = model.dark_theme ? 0 : 1;
            bool theme_changed = false;
            if (ImGui::RadioButton("Dark", &theme_idx, 0)) theme_changed = true;
            ImGui::SameLine();
            if (ImGui::RadioButton("Light", &theme_idx, 1)) theme_changed = true;
            if (theme_changed) {
                model.dark_theme = (theme_idx == 0);
                persist_settings(model);
            }
            ImGui::TextDisabled("Color palette for ImGui controls. Plots use their own colormap.");

            // ----------------------------------------------------------------
            // Colormaps: какие из 200 карт slanCM показывать в пикере.
            // Держать в combo все 200 неудобно, поэтому набор набирается
            // галочками здесь и живёт в _app_config.json (одна маска на
            // приложение, как и остальные настройки этой вкладки).
            // ----------------------------------------------------------------
            ImGui::Separator();
            ImGui::Text("Colormaps");
            ImGui::TextDisabled("200 colormaps from slanCM (MATLAB File Exchange #120088),");
            ImGui::TextDisabled("numbered as in that library. Ticked ones show up in the");
            ImGui::TextDisabled("Colormap picker above every heatmap. Viridis / Inferno /");
            ImGui::TextDisabled("Turbo / Gray are built in and always available.");

            // Битая или отсутствующая маска (конфиг от старой версии, ручная
            // правка JSON) — молча чинится дефолтом, а не роняет вкладку.
            if (model.slancm_enabled.size() != (size_t)kSlanCmCount) {
                model.slancm_enabled = default_enabled_slancm();
                set_enabled_slancm(model.slancm_enabled);
            }
            auto apply_cmap_mask = [&]() {
                set_enabled_slancm(model.slancm_enabled);
                persist_settings(model);
            };

            InputTextStr("Filter##cmap_filter", model.colormap_filter, 220.0f);
            ImGui::SameLine();
            ImGui::TextDisabled("name, number or category");

            // Фильтрованный список номеров. 200 сравнений на кадр дешевле, чем
            // кэш с инвалидацией по каждому нажатию в поле фильтра.
            std::vector<int> shown;
            shown.reserve(kSlanCmCount);
            {
                auto lower = [](std::string s) {
                    // Имена и категории slanCM — чистый ASCII, поэтому обходимся
                    // без <cctype> и без вопросов к локали.
                    for (char& c : s) if (c >= 'A' && c <= 'Z') c = (char)(c + 32);
                    return s;
                };
                const std::string q = lower(model.colormap_filter);
                for (int n = 1; n <= kSlanCmCount; ++n) {
                    if (q.empty()) { shown.push_back(n); continue; }
                    const std::string hay = lower(std::string(slancm_name(n)) + ' '
                        + slancm_category_name(slancm_category(n)) + ' ' + std::to_string(n));
                    if (hay.find(q) != std::string::npos) shown.push_back(n);
                }
            }

            // Кнопки работают по ВИДИМОМУ списку: с фильтром "Diverging" это
            // даёт «включить всю категорию» одним нажатием, без фильтра —
            // обычные select all / clear all по всем 200.
            auto set_shown = [&](char v) {
                for (int n : shown) model.slancm_enabled[n - 1] = v;
                apply_cmap_mask();
            };
            if (ImGui::Button("Select all")) set_shown('1');
            ImGui::SameLine();
            if (ImGui::Button("Clear all")) set_shown('0');
            ImGui::SameLine();
            if (ImGui::Button("Reset to default")) {
                model.slancm_enabled = default_enabled_slancm();
                apply_cmap_mask();
            }
            ImGui::SameLine();
            {
                const int n_on = (int)std::count(model.slancm_enabled.begin(),
                                                 model.slancm_enabled.end(), '1');
                if (shown.size() == (size_t)kSlanCmCount)
                    ImGui::Text("%d / %d selected", n_on, kSlanCmCount);
                else
                    ImGui::Text("%d / %d selected   (buttons apply to the %d shown)",
                                n_on, kSlanCmCount, (int)shown.size());
            }

            {
                const float em      = ImGui::GetFontSize();
                const float strip_w = em * 6.0f;
                const float col_num   = em * 2.2f;
                const float col_strip = em * 4.4f;
                const float col_name  = col_strip + strip_w + em * 0.6f;
                const float col_cat   = col_name + em * 11.0f;
                const float row_h     = ImGui::GetFrameHeightWithSpacing();

                ImGui::BeginChild("##cmap_list", ImVec2(0, row_h * 12.0f), true);
                // Клиппер: 200 строк с градиентами рисовать каждый кадр незачем,
                // видно от силы дюжину. Высота строк одинаковая (её задаёт
                // чекбокс), поэтому хватает Begin(count) без items_height.
                ImGuiListClipper clip;
                clip.Begin((int)shown.size());
                while (clip.Step()) {
                    for (int i = clip.DisplayStart; i < clip.DisplayEnd; ++i) {
                        const int n = shown[i];
                        ImGui::PushID(n);
                        bool on = (model.slancm_enabled[n - 1] == '1');
                        if (ImGui::Checkbox("##on", &on)) {
                            model.slancm_enabled[n - 1] = on ? '1' : '0';
                            apply_cmap_mask();
                        }
                        ImGui::SameLine(col_num);
                        ImGui::TextDisabled("%d", n);
                        ImGui::SameLine(col_strip);
                        colormap_strip_item(slancm_id(n), strip_w, ImGui::GetFrameHeight());
                        ImGui::SameLine(col_name);
                        ImGui::TextUnformatted(slancm_name(n));
                        ImGui::SameLine(col_cat);
                        ImGui::TextDisabled("%s", slancm_category_name(slancm_category(n)));
                        ImGui::PopID();
                    }
                }
                ImGui::EndChild();
                if (shown.empty())
                    ImGui::TextDisabled("Nothing matches the filter.");
            }

            // ----------------------------------------------------------------
            // Peak detection & regime thresholds — knobs configCUDA.h.
            // Уходят в NVRTC как #define, поэтому правка = перекомпиляция ядер
            // на следующем Run (set_peak_config бампает cache-epoch).
            // ----------------------------------------------------------------
            ImGui::Separator();
            ImGui::Text("Peak detection & regime thresholds");
            ImGui::TextDisabled("Applies to every GPU calculation: bifurcations, LLE/LS,");
            ImGui::TextDisabled("basins, DFT. Changing a value recompiles the NVRTC");
            ImGui::TextDisabled("kernels on the next Run.");

            PeakConfig& pk = model.peak;
            bool pk_changed = false;

            pk_changed |= ImGui::Checkbox("Calculate peaks##do_calc_peaks", &pk.do_calculate_peaks);
            ImGui::TextDisabled("Off: the raw trajectory is passed downstream instead of its peaks.");
            pk_changed |= ImGui::Checkbox("Parabolic interpolation of peaks##do_interp_peaks", &pk.do_interpolate_peaks);
            ImGui::TextDisabled("Sub-sample peak position/amplitude from the 3-point parabola.");

            // Числовые поля — тем же виджетом, что и во вкладках анализа:
            // текстовый ввод с ↑/↓ по разряду, запятой вместо точки и дробями
            // "a/b", без навязанного "%.3e". Значение снимается на commit'е,
            // см. InputNumStrCommit.
            pk_changed |= InputNumStrCommit("eps fixed point##eps_fp",
                                            model.peak_eps_fixed_point_text,
                                            pk.eps_fixed_point, 220);
            ImGui::TextDisabled("Max sum|x(n)-x(n-1)| still reported as regime -1 (fixed point).");

            pk_changed |= InputNumStrCommit("eps peak delta##eps_pd",
                                            model.peak_eps_peak_delta_text,
                                            pk.eps_peak_delta, 220);
            ImGui::TextDisabled("Min rise/fall against a neighbour to accept a sample as a peak.");

            pk_changed |= InputNumStrCommit("eps inter-peak delta##eps_ipd",
                                            model.peak_eps_interPeak_delta_text,
                                            pk.eps_interPeak_delta, 220);
            ImGui::TextDisabled("Min interspike interval; closer peaks are merged. 0 = filter off.");

            pk_changed |= InputNumStrCommit("peak threshold##pk_thr",
                                            model.peak_threshold_text,
                                            pk.peak_threshold, 220);
            ImGui::TextDisabled("Peaks below this value are ignored.");

            // max peaks целочисленный, но поле — то же самое. Диапазон режем ДО
            // приведения к int: пользователь может ввести 1e30, а это уже UB на
            // касте. Ниже clamp_peak_config всё равно отработает — здесь только
            // защита самого каста.
            double max_peaks_val = (double)pk.max_amount_of_peaks;
            if (InputNumStrCommit("max peaks##pk_max", model.peak_max_amount_text,
                                  max_peaks_val, 220)) {
                max_peaks_val = std::max((double)kPeakCountMin,
                                std::min((double)kPeakCountMax, max_peaks_val));
                pk.max_amount_of_peaks = (int)std::llround(max_peaks_val);
                pk_changed = true;
            }
            ImGui::TextDisabled("Per-trajectory cap (%d..%d). Also sizes the device peak",
                                kPeakCountMin, kPeakCountMax);
            ImGui::TextDisabled("and interval buffers allocated for every sweep point.");
            if (pk.max_amount_of_peaks > 5000)
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                    "Large values raise GPU memory use for 2D bifurcation / basins.");

            if (ImGui::Button("Reset to configCUDA.h defaults")) {
                pk = PeakConfig{};
                model.sync_peak_text();
                pk_changed = true;
            }

            if (pk_changed) {
                const PeakConfig before = pk;
                clamp_peak_config(pk);
                // Кламп мог поправить введённое (отрицательный eps, потолок max
                // peaks) — тогда текст в поле уже врёт про конфиг, пересеиваем
                // буферы из значений.
                if (before.eps_fixed_point     != pk.eps_fixed_point ||
                    before.eps_peak_delta      != pk.eps_peak_delta ||
                    before.eps_interPeak_delta != pk.eps_interPeak_delta ||
                    before.max_amount_of_peaks != pk.max_amount_of_peaks)
                    model.sync_peak_text();
                set_peak_config(pk);       // бампает epoch -> PTX-кэши инвалидируются
                persist_settings(model);
            }

            // ----------------------------------------------------------------
            // GPU floating point — опция компиляции NVRTC, а не #define, но
            // логика та же: смена значения = другой PTX, кэши модулей обоих
            // движков инвалидируются ключом (см. parametric_engine.h).
            // Настройка ОДНА на приложение сознательно: карта и фазовый портрет
            // по её ячейке обязаны считать одинаково.
            // ----------------------------------------------------------------
            ImGui::Separator();
            ImGui::Text("GPU floating point");
            ImGui::TextDisabled("Applies to every NVRTC kernel: bifurcations, LLE/LS, basins,");
            ImGui::TextDisabled("DFT, fast synchro AND phase portraits. Changing it recompiles");
            ImGui::TextDisabled("the kernels on the next Run.");

            if (ImGui::Checkbox("FMA contraction (--fmad)##nvrtc_fmad", &model.nvrtc_fmad)) {
                set_nvrtc_fmad(model.nvrtc_fmad);
                persist_settings(model);
            }
            if (model.nvrtc_fmad) {
                ImGui::TextDisabled("On (NVRTC default): a*b+c is fused, one rounding instead");
                ImGui::TextDisabled("of two. Faster and the usual choice.");
            } else {
                ImGui::TextDisabled("Off: multiply and add round separately. Closer to the CPU");
                ImGui::TextDisabled("fallback and to external references (MATLAB), slightly slower.");
            }
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                "Results computed with different settings are not bit-comparable;");
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                "on fractal basin boundaries they may differ visibly. The value in");
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                "effect at Run time is written into every exported _config.csv.");
        }
        ImGui::End();
    }
}