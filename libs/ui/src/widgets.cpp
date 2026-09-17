#include "gui_internal.h"

namespace ucuda::ui_detail {


// Возвращает директорию exe со слешем в конце. Реализована в app_main.cpp
// (там же используется для resolve_python_exe / library_dir).

std::string get_exe_dir_with_sep() {
    std::string d = platform::executable_dir();
    if (!d.empty() && d.back() != '\\' && d.back() != '/') d += "\\";
    return d;
}


// Базовый цвет траектории по индексу НУ (единый для 2D/3D/time domain).
ImVec4 ic_base_color(int ic_index) {
    return ImPlot::GetColormapColor(ic_index);
}


// Оттенок базового цвета по насыщенности: для переменной vi из nv внутри
// одного НУ. vi=0 — самый насыщенный, дальше бледнее. Используется в time domain,
// чтобы переменные одного НУ были видимо родственны (один тон), но различимы.
ImVec4 shade_of(ImVec4 base, int vi, int nv) {
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(base.x, base.y, base.z, h, s, v);
    // распределяем насыщенность от 1.0 (vi=0) до ~0.35 (последняя переменная)
    float frac = (nv <= 1) ? 0.0f : (float)vi / (float)(nv - 1);
    float new_s = s * (1.0f - 0.45f * frac); // от s до 0.55*s (было 0.65 -> 0.35, тускло)
    float r, g, b;
    ImGui::ColorConvertHSVtoRGB(h, new_s, v, r, g, b);
    return ImVec4(r, g, b, base.w);
}

ImVec4 ic_var_shade(int ic_index, int vi, int nv) {
    return shade_of(ic_base_color(ic_index), vi, nv);
}


// helpers: std::string <-> ImGui
// Подпись комбинации переменных: x0 + pi*x1 + e*x2 — тот ряд, что строит ядро при
// writable_var == -1 (loopCalculateDiscreteModel_int), с той же деградацией при
// dim < 3. Одна на Feature diagram и Time domain; у draw_writable_var_combo своя
// копия — она старше и в этот заход не трогалась.
[[nodiscard]] std::string combo_var_label(const std::vector<std::string>& vars) {
    if (vars.size() >= 3) return vars[0] + " + pi*" + vars[1] + " + e*" + vars[2];
    if (vars.size() == 2) return vars[0] + " + pi*" + vars[1];
    return vars.empty() ? std::string("x") : vars[0];
}


// Значение комбинации в точке траектории — зеркалит формулу ядра. Константы
// pi/euler берутся из configCUDA.h, а не набираются заново.
[[nodiscard]] double combo_var_value(const std::vector<double>& pt, int nvars) {
    double v = !pt.empty() ? pt[0] : 0.0;
    if (nvars >= 2 && pt.size() > 1) v += ::pi    * pt[1];
    if (nvars >= 3 && pt.size() > 2) v += ::euler * pt[2];
    return v;
}


// Скретч-буфер под ImGui::InputText*: ImGui пишет в char*, а хранить значение надо
// в std::string. Один на поток вместо vector<char> в каждом виджете на каждом кадре
// (полсотни полей = сотня heap-аллокаций в кадр). Виджеты не вложены, а ImGui
// копирует содержимое внутри вызова — дольше вызова буфер не нужен; ёмкость между
// кадрами сохраняется, поэтому assign/resize уже не аллоцируют.
std::vector<char>& input_scratch(const std::string& str, size_t extra) {
    static thread_local std::vector<char> buf;
    buf.assign(str.begin(), str.end());
    buf.resize(str.size() + extra);
    buf[str.size()] = '\0';
    return buf;
}


bool InputTextMultilineStr(const char* label, std::string& str, const ImVec2& size) {
    std::vector<char>& buf = input_scratch(str, 4096);
    bool changed = ImGui::InputTextMultiline(label, buf.data(), buf.size(), size);
    if (changed) str = buf.data();
    return changed;
}


// Drag handle below a multiline input; drags along Y and clamps `height`.
void draw_resize_handle(const char* id, float& height,
                               float min_h , float max_h ) {
    // InvisibleButton refuses non-positive dimensions — resolve -1 to the actual width.
    const float w = ImGui::GetContentRegionAvail().x;
    ImGui::InvisibleButton(id, ImVec2(w > 1.0f ? w : 1.0f, 8.0f));
    const bool active  = ImGui::IsItemActive();
    const bool hovered = ImGui::IsItemHovered();
    if (active) {
        height += ImGui::GetIO().MouseDelta.y;
        if (height < min_h) height = min_h;
        if (height > max_h) height = max_h;
    }
    if (hovered || active) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
    const ImVec2 pmin = ImGui::GetItemRectMin();
    const ImVec2 pmax = ImGui::GetItemRectMax();
    const ImU32 col = ImGui::GetColorU32(active  ? ImGuiCol_SeparatorActive
                                       : hovered ? ImGuiCol_SeparatorHovered
                                                 : ImGuiCol_Separator);
    const float y = (pmin.y + pmax.y) * 0.5f;
    ImGui::GetWindowDrawList()->AddLine(ImVec2(pmin.x, y), ImVec2(pmax.x, y), col, 3.0f);
}

bool InputTextStr(const char* label, std::string& str, float width ) {
    std::vector<char>& buf = input_scratch(str, 1024);
    if (width > 0) ImGui::SetNextItemWidth(width);
    bool changed = ImGui::InputText(label, buf.data(), buf.size());
    if (changed) str = buf.data();
    return changed;
}


// Перехватываем символы ДО того, как ImGui положит их в буфер: замена ',' → '.'
// идёт в момент ввода и не подменяет строку между кадрами. Иначе InputText на
// каждом кадре видит внешнюю правку буфера, возвращает changed=true, и
// auto_recompute триггерится непрерывно, пока в поле есть запятая.
int filter_comma_to_dot(ImGuiInputTextCallbackData* data) {
    if (data->EventChar == ',') data->EventChar = '.';
    return 0;
}


// Совмещённый callback (запятая→точка + digit-step на ↑/↓) переехал в
// plot_axis.cpp::digit_step_input_callback — тем же вводом пользуется меню
// цвета серии в plot_view_2d.cpp, и держать копию в gui.cpp было нельзя.

// Разбирается ли строка как число или арифметическое выражение? Пустая валидна
// (дальше подставится дефолт). Валидны: "2.5", "1e-14", "8/3", "0.5*1e-3",
// "100-1", "(1+2)*3", "-2". Невалидны: "2/x", "8/0", "8/", "5asdfaxcv", "1e400",
// "(1+2". Делегирует в parse_num_checked — ту же грамматику, которой поле потом
// реально разбирается: своя копия на std::stod успела разойтись с ней (на
// "5asdf" подсветка писала «using default», а в расчёт уходило 5).
[[nodiscard]] bool is_numeric_string(const std::string& s) {
    if (s.empty()) return true;
    double v = 0.0;
    return parse_num_checked(s, v);
}


// Is `scheme_name` one of the session's custom KRS schemes (as opposed to a
// built-in name)?
[[nodiscard]] bool is_custom_scheme(const std::string& scheme_name,
                              const std::vector<CustomScheme>& custom_schemes) {
    for (const auto& cs : custom_schemes) if (cs.name == scheme_name) return true;
    return false;
}


// Heuristic text match: does a custom KRS body actually reference a[0]
// (the symmetry slot, same as built-in CD)? Raw C/CUDA source, not parsed
// into an AST, so this is a regex over the literal text rather than a real
// usage analysis — good enough since users write "a[0]" directly.
[[nodiscard]] bool custom_scheme_uses_symmetry(const std::string& scheme_name,
                                         const std::vector<CustomScheme>& custom_schemes) {
    static const std::regex re(R"(a\s*\[\s*0\s*\])");
    for (const auto& cs : custom_schemes)
        if (cs.name == scheme_name) return std::regex_search(cs.body, re);
    return false;
}


// Читает ли схема коэффициент симметрии a[0] — от этого зависит, показывать ли
// поле symmetry в конфигах и пункт "s (a[0])" в выборе цели свипа. Встроенных
// таких пять: CD (h1 = s*h, h2 = (1-s)*h), оба Complex CD (то же деление шага,
// плюс мнимая часть ±h*sqrt(3)/6) и SEMP/SIMP (стадия на полушаге h1 = s*h).
// Один предикат на весь файл: раньше проверка была выписана четырьмя копиями
// `scheme == "CD" || custom_scheme_uses_symmetry(...)`, и добавление схемы
// требовало не забыть все четыре.
[[nodiscard]] bool scheme_uses_symmetry(const std::string& scheme_name,
                                         const std::vector<CustomScheme>& custom_schemes) {
    return scheme_name == "CD" || scheme_name == "Complex CD" || scheme_name == "Complex CD4"
        || scheme_name == "Complex Implicit Euler"
        || scheme_name == "SEMP" || scheme_name == "SIMP"
        || custom_scheme_uses_symmetry(scheme_name, custom_schemes);
}


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
                                   bool sweep_over_h ) {
    if (sweep_over_h) return "h";
    if (sweep_over_var) {
        if (var_idx >= 0 && var_idx < (int)vars.size()) return vars[var_idx] + " (IC)";
        return "var";
    }
    // -1 — цель "s": коэффициент симметрии живёт в a[0], а не в списке
    // параметров системы, поэтому имени для него в params[] нет.
    if (param_idx < 0) return "s";
    if (param_idx < (int)params.size()) return params[param_idx];
    return "param";
}


// Хвост подписи 1D-свипа: "<ось> [lo..hi]". Пустое поле показывается как "?".
[[nodiscard]] std::string sweep_range_label(const std::string& axis,
                                                   const std::string& lo_text,
                                                   const std::string& hi_text) {
    const std::string lo = lo_text.empty() ? std::string("?") : lo_text;
    const std::string hi = hi_text.empty() ? std::string("?") : hi_text;
    return axis + " [" + lo + ".." + hi + "]";
}


[[nodiscard]] std::string auto_label_window(const ParametricPlotWindow& w) {
    const char* kn = w.kind == ParametricPlotWindow::Kind::Bifurcation ? "Bifurcation"
                    : w.kind == ParametricPlotWindow::Kind::LLE ? "LLE" : "LS";
    const char* suffix = w.colored_1d ? "Colored 1D" : (w.mode_2d ? "2D" : "1D");
    return std::string(kn) + " " + suffix;
}


// У DFT1D 2D-режима нет, поэтому auto_label_sweep (он читает mode_2d и поля
// второй оси) не подходит — но хвост общий.
[[nodiscard]] std::string auto_label_dft1d(const Dft1DConfig& c,
                                    const std::vector<std::string>& params,
                                    const std::vector<std::string>& vars) {
    return sweep_range_label(auto_axis_name(params, vars, c.param_index, c.sweep_over_var,
                                            c.var_sweep_index, c.sweep_over_h),
                             c.param_lo_text, c.param_hi_text);
}


void refresh_auto_labels(AppModel& model) {
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
[[nodiscard]] inline double parse_ratio_or(const std::string& v, double def) {
    // Round-trip через numb: интерфейс обязан показывать и отдавать в расчёт
    // РОВНО то число, которое увидит ядро. Сегодня numb == double и это no-op,
    // но если numb станет float — подпись оси, кламп fix-значения и привязка
    // крестика к сетке не начнут врать относительно посчитанного.
    return (double)(numb)parse_num(v, def);
}


// Целочисленные поля: Resolution, decimator, число бинов, N сетки.
// Тело переехало в num_parse.h к parse_num_int — там же объяснено, почему
// вторая копия этого разбора (на std::stoi, в analysis_session.cpp) молча
// врала на "512/2". Имя сохранено: оно стоит по месту в gui.cpp.
[[nodiscard]] inline int parse_int_or(const std::string& v, int def) {
    return parse_num_int(v, def);
}


bool InputNumStr(const char* label, std::string& str, float width ) {
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
                              float width ) {
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


// Имена для ImGui::Combo, которому нужен массив const char*. Указатели живут,
// пока жив исходный вектор строк, — результат используется тут же, в том же
// кадре. Раньше эти три строки были выписаны трижды (Basins и оба режима
// FastSync).
[[nodiscard]] std::vector<const char*> c_str_list(const std::vector<std::string>& v) {
    std::vector<const char*> out;
    out.reserve(v.size());
    for (const auto& s : v) out.push_back(s.c_str());
    return out;
}


// Комбо выбора схемы: встроенные + (через разделитель) пользовательские.
// on_pick вызывается ПОСЛЕ записи имени в scheme — им пользуются Phase
// (regenerate_krs) и Custom Shared config (проброс в phase_session).
// Возвращает true, если пользователь выбрал схему в этом кадре.
bool draw_adaptive_controls(adaptive::Options& a) {
    ImGui::PushID(&a);
    bool changed = ImGui::Combo("Integration", &a.method, "Fixed step\0Adaptive DOPRI5\0Adaptive step doubling\0");
    if (a.method != 0) {
        ImGui::TextWrapped("Step h is the output / analysis interval. Internal steps are selected automatically.");
        if (a.method == 1) ImGui::TextDisabled("DOPRI5 uses the ODE, independently of the selected base scheme.");
        ImGui::SetNextItemWidth(130);
        changed |= ImGui::InputDouble("Relative tolerance", &a.rtol, 0, 0, "%.9g");
        ImGui::SetNextItemWidth(130);
        changed |= ImGui::InputDouble("Absolute tolerance", &a.atol, 0, 0, "%.9g");
        ImGui::SetNextItemWidth(130);
        changed |= ImGui::InputDouble("Initial internal step", &a.h_initial, 0, 0, "%.9g");
        ImGui::SetNextItemWidth(130);
        changed |= ImGui::InputDouble("Minimum internal step", &a.h_min, 0, 0, "%.9g");
        ImGui::SetNextItemWidth(130);
        changed |= ImGui::InputDouble("Maximum internal step", &a.h_max, 0, 0, "%.9g");
        if (a.method == 2) {
            changed |= ImGui::InputInt("Scheme order (0 = built-in auto)", &a.order);
            ImGui::TextWrapped("Custom schemes require an explicit order. The accepted result is two half-steps, without extrapolation.");
        }
        if (ImGui::TreeNode("Adaptive limits")) {
            changed |= ImGui::InputInt("Maximum attempts", &a.max_steps);
            changed |= ImGui::InputInt("Consecutive rejection limit", &a.max_rejects);
            ImGui::TreePop();
        }
        auto check = a; if (check.method == 2 && check.order == 0) check.order = 1;
        if (!adaptive::valid(check)) ImGui::TextColored(ImVec4(1,.4f,.3f,1), "Invalid tolerance, step bounds or limits");
    }
    ImGui::PopID();
    return changed;
}


bool draw_scheme_combo(const char* label, std::string& scheme,
                              const std::vector<CustomScheme>& custom_schemes,
                              const std::function<void(const std::string&)>& on_pick ,
                              const std::vector<std::string>* enabled_builtins ) {
    bool picked = false;
    auto choose = [&](const std::string& nm) {
        scheme = nm;
        if (on_pick) on_pick(nm);
        picked = true;
    };
    // enabled_builtins empty/null = no filter. Currently-selected built-in stays visible
    // even if not enabled, so sessions saved with a now-disabled scheme aren't stranded.
    const bool filter = enabled_builtins && !enabled_builtins->empty();
    auto is_enabled = [&](const char* m) {
        if (!filter) return true;
        for (const auto& n : *enabled_builtins) if (n == m) return true;
        return false;
    };
    ImGui::SetNextItemWidth(kComboW);
    if (ImGui::BeginCombo(label, scheme.c_str())) {
        // Заголовок порядка печатаем лениво — перед ПЕРВОЙ видимой схемой
        // группы, иначе отфильтрованная группа оставила бы пустую шапку.
        int shown_order = 0;
        for (const auto& b : kBuiltinSchemes) {
            if (!is_enabled(b.name) && scheme != b.name) continue;
            if (b.order != shown_order) {
                ImGui::SeparatorText(("Order " + std::to_string(b.order)).c_str());
                shown_order = b.order;
            }
            if (ImGui::Selectable(b.name, scheme == b.name)) choose(b.name);
        }
        if (!custom_schemes.empty()) ImGui::Separator();
        for (const auto& cs : custom_schemes)
            if (ImGui::Selectable((cs.name + " (custom)").c_str(), scheme == cs.name))
                choose(cs.name);
        ImGui::EndCombo();
    }
    return picked;
}


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
                                    bool* other_over_h ,
                                    bool note_when_empty ,
                                    float width ,
                                    bool allow_s ) {
    if (params.empty() && vars.empty() && note_when_empty) {
        ImGui::TextDisabled("No parameters/variables (select a system first)");
        return;
    }
    // Кламп индексов: сохранённая сессия могла прийти от системы с другим
    // числом переменных/параметров. Отрицательный par_index — не мусор, а цель
    // "s"; сбрасываем его только там, где схема a[0] не читает (или сессия
    // пришла от схемы, которая читала).
    if (par_index < 0 && !allow_s) par_index = 0;
    if (par_index >= (int)params.size()) par_index = 0;
    if (var_index < 0 || var_index >= (int)vars.size())   var_index = 0;

    const std::string preview =
          over_h                      ? std::string("dt (h)")
        : (over_var && !vars.empty()) ? (vars[var_index] + " (IC)")
        : (par_index < 0)             ? std::string("s (a[0])")
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

    if (allow_s) {
        ImGui::Separator();
        if (ImGui::Selectable("s (a[0])", !over_var && !over_h && par_index < 0)) {
            par_index = -1; over_var = false; over_h = false;
        }
    }
    ImGui::Separator();
    if (ImGui::Selectable("dt (h)", over_h)) {
        over_h = true; over_var = false;
        if (other_over_h) *other_over_h = false;   // ровно одна ось = h
    }
    ImGui::EndCombo();
}


bool draw_integration_block(const char* header,
                                   const std::string& scheme,
                                   const std::vector<CustomScheme>& custom_schemes,
                                   const IntegrationFields& f) {
    if (!ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen)) return false;
    bool changed = false;
    if (f.h) changed |= InputNumStr("h", *f.h, kFieldW);
    if (f.symmetry_s && scheme_uses_symmetry(scheme, custom_schemes))
        changed |= InputNumStr("symmetry s", *f.symmetry_s, kFieldW);
    if (f.t_max)       changed |= InputNumStr("computing time", *f.t_max,       kFieldW);
    if (f.transient)   changed |= InputNumStr("transient time", *f.transient,   kFieldW);
    if (f.pre_scaller) changed |= InputNumStr("decimator",      *f.pre_scaller, kFieldW);
    if (f.max_value)   changed |= InputNumStr("max value",      *f.max_value,   kFieldW);
    return changed;
}


// Секции "Initial conditions" и "Parameters" — это один и тот же блок:
// по числовому полю на каждое имя из списка. Раньше — около дюжины копий.
void draw_named_num_fields(const char* header,
                                  const std::vector<std::string>& names,
                                  std::map<std::string, std::string>& values,
                                  const char* note ) {
    if (!ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen)) return;
    if (note) ImGui::TextDisabled("%s", note);
    for (const auto& n : names) {
        ImGui::PushID(n.c_str());
        InputNumStr(n.c_str(), values[n], kFieldW);
        ImGui::PopID();
    }
}


// Секция "CSV output": галка сохранения + путь + пояснение про побочные файлы.
void draw_csv_output_block(const char* header, const char* checkbox_label,
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
void draw_error_box(const char* id, const std::string& err,
                           int lines ) {
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error (selectable, Ctrl+C):");
    const ImVec2 sz(-1.0f, ImGui::GetTextLineHeight() * (float)lines);
    ImGui::InputTextMultiline(id, const_cast<char*>(err.c_str()), err.size() + 1,
                              sz, ImGuiInputTextFlags_ReadOnly);
}


// Пункт "Export data..." правого клика по диаграмме. Диалог выбора файла —
// платформенный, приходит из GuiCallbacks; write вызывается только с
// непустым путём (пустой = пользователь нажал Отмену).
void draw_export_menu_item(bool busy, const GuiCallbacks& cb,
                                  const std::function<void(const std::string&)>& write) {
    if (!ImGui::MenuItem("Export data...", nullptr, false, !busy)) return;
    if (!cb.pick_save_file_csv) return;
    const std::string path = cb.pick_save_file_csv();
    if (!path.empty() && write) write(path);
}


// Inline-переименование активной вкладки. Раньше — два ручных char buf[128],
// которые молча обрезали имя на 127 символах.
void draw_label_rename(const char* label_with_id, std::string& label,
                              float width ) {
    ImGui::SetNextItemWidth(width);
    InputTextStr(label_with_id, label);
}


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
                                        int request_select ) {
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
void configure_param_plot_view(Plot2DView& view, ParamPlotKind kind) {
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


bool draw_point_style_toolbar(BifurcationDiagramConfig& bd, const char* id_suffix) {
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
void apply_point_style(Plot2DView& view, const BifurcationDiagramConfig& bd) {
    view.point_marker  = bd.custom_point_style ? bd.point_marker : -1;
    view.point_size_px = bd.custom_point_style ? bd.point_size   : 2.0f;
}


// Snap X к узлам параметрической сетки: тики осей и hover-readout попадают
// ровно на посчитанные точки. Раньше — три идентичных блока в Bif/LLE/LS и
// полное отсутствие в Custom (из-за чего tooltip там врал между узлами).
// n < 2 отключает snap (readout остаётся непрерывным).
void apply_snap_x(Plot2DView& view, double lo, double hi, int n) {
    view.snap_x_to_grid = true;
    if (n > 1) {
        view.snap_x_min = lo;
        view.snap_x_max = hi;
        view.snap_x_n   = n;
    } else {
        view.snap_x_n = 0;
    }
}


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
                             bool log_scale, bool reverse, bool continuation) {
    if (continuation || reverse) {
        // Та же функция, что у cont-ядер и cont_sweep_value: конвенция continuation
        // живёт в ucuda_node_value_cont (configCUDA.h). Гард lo/hi > 0 на лог-ветке —
        // сюда может долететь живой чекбокс без прогона, а log10(<=0) отравил бы ось NaN'ом.
        const bool log_ok_cont = log_scale && lo > 0.0 && hi > 0.0;
        return (double)ucuda_node_value_cont(k, n, (numb)lo, (numb)hi, log_ok_cont, reverse);
    }
    if (log_scale && lo > 0.0 && hi > 0.0)
        return (double)ucuda_node_value_log(k, n, (numb)lo, (numb)hi);
    return (double)ucuda_node_value(k, n, (numb)lo, (numb)hi);
}


// Общая настройка X-оси 1D-графика по членам окна (раньше — три почти одинаковых
// блока по ~35 строк в Bif/LLE/LS):
//   - подпись = общий sweep-таргет; если члены свипают разное — "parameter";
//   - fit-диапазон = union(lo, hi) ВСЕХ членов: ось обязана охватывать весь свип,
//     даже если часть точек разошлась и на графике их нет;
//   - log_scale только если он одинаков у всех членов.
void configure_sweep_x_axis(Plot2DView& view,
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

    if (shared_kind == 0 && shared_idx < 0)
        view.x_axis.name = "s";
    else if (shared_kind == 0 && shared_idx >= 0 && shared_idx < (int)params.size())
        view.x_axis.name = params[shared_idx];
    else if (shared_kind == 1 && shared_idx >= 0 && shared_idx < (int)vars.size())
        view.x_axis.name = vars[shared_idx] + " (IC)";
    else if (shared_kind == 2)
        view.x_axis.name = "h";
    else
        view.x_axis.name = "parameter";
}


// Диапазон свипа для LLE/LS: снапшот прогона, а если движок его ещё не
// заполнил (lo == hi) — текстовые поля конфига.
void sweep_range_from_result(double res_lo, double res_hi,
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
void apply_snap_x_from_config(Plot2DView& view,
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


// Right-click «Export data...» подменю: перечисляет ВСЕ конфиги сессии с готовым
// прогоном, а не только members этого окна — экспорт к окну не привязан.
// ready(i) — есть законченный прогон, busy(i) — считается сейчас, do_export(i, path) —
// собственно запись. Раньше — три идентичные копии в Bif/LLE/LS.
void draw_export_submenu(const char* id_tag, int n,
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

HeatmapView& get_or_create_heatmap(
        std::map<int, std::unique_ptr<HeatmapView>>& map, int idx,
        int cfg_colormap, int app_default_colormap,
        int cfg_exponent_idx ) {
    auto& slot = map[idx];
    if (!slot) {
        slot = std::make_unique<HeatmapView>();
        slot->colormap = (HeatmapColormap)colormap_id_or(cfg_colormap, app_default_colormap);
        if (cfg_exponent_idx != kNoExponent) slot->display_exponent_idx = cfg_exponent_idx;
    }
    return *slot;
}


// Скретч-буферы точек одного plot-окна (по одному на серию). Между кадрами живут намеренно:
// buf.clear() у вызывающего сохраняет capacity, поэтому со второго кадра push_back уже не
// аллоцирует — на диаграммах в миллионы точек это заметно.
// Ключ — стабильный ParametricPlotWindow::id, как у кэшей HeatmapView выше. Раньше на каждую
// из трёх функций был один static, общий ВСЕМ окнам: при двух окнах с разным числом серий
// assign() передёргивал буферы каждый кадр, и оптимизация не работала (корректность при этом
// не страдала — render() забирает точки синхронно).
// Записи удалённых окон остаются в карте до перезапуска, как и у хитмап: окон единицы.
std::vector<std::vector<float>>& window_point_bufs(int window_id, size_t n_series) {
    static std::map<int, std::vector<std::vector<float>>> cache;
    auto& bufs = cache[window_id];
    if (bufs.size() != n_series) bufs.assign(n_series, {});
    return bufs;
}


// Полоска-превью колормапа как ImGui-виджет: kSeg сегментов с линейным градиентом внутри
// каждого плюс Dummy того же размера, чтобы полоска занимала место в layout'е. 24 сегмента
// хватает, чтобы читались даже рваные качественные карты (tab20, glasbey), а рисуется это
// только для видимых строк — и в попапе пикера, и под клиппером в Settings.
void colormap_strip_item(int id, float w, float h) {
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


// Пикер колормапа: BeginCombo вместо ImGui::Combo, потому что список теперь разреженный
// (только отмеченные пользователем карты slanCM) и в каждой строке рисуется градиент. Текущий
// id показывается всегда, даже если карта снята галочкой в Settings — иначе открытая сессия
// молча переехала бы на другую карту. Возвращает true, если выбор изменился.
bool colormap_combo(const char* label, int* id, float width) {
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


// ЕДИНСТВЕННАЯ реализация тулбара «Colormap / Autoscale / vmin / vmax / Swap axes»
// (раньше был скопирован в 7 мест с разным составом). Состояние живёт в самом
// HeatmapView — вызывающему достаточно передать, куда персистить colormap.
// Возвращает true, если пользователь что-то изменил.
bool draw_heatmap_toolbar(HeatmapView& hv, const HeatmapToolbarOpts& o ) {
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


// LS exponent picker: λ1..λN + "sum L_i" (sentinel -1, а не N, чтобы выбор не съезжал
// при смене N на следующем Run). Выбор живёт в HeatmapView, а не в конфиге: два окна с
// одной кривой не должны дёргать друг у друга индекс. on_pick — персистентность на
// стороне вызывающего. Раньше — две копии (draw_ls_plot и Custom-слот 2).
void draw_ls_exponent_picker(HeatmapView& hv, int n_exponents,
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


// Выбор отображаемой плоскости LS-2D по индексу экспоненты `k`:
//   k >= 0  — готовая плоскость из result_2d.values (без копирования),
//   k == -1 — "sum L_i": поэлементная сумма всех N плоскостей, считается ОДИН раз на
//             data_generation_2d и кэшируется в конфиге (кэш общий для всех окон с этой
//             кривой). Diverged-ячейки (flags<0) получают тот же sentinel 999.0, что
//             engine пишет в одиночные плоскости, — HeatmapView красит их тем же серым.
// gen мешает поколение чанка с индексом экспоненты, иначе переключение экспоненты не
// перезалило бы текстуру. Раньше — две копии по ~40 строк.
void ls_resolve_plane(LSCurveConfig& cact, int k,
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


// Вкладка System: ввод системы, методы, генерация кода
void draw_system_tab(AppModel& model, const GuiCallbacks& cb) {
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
        InputTextMultilineStr("##latex", model.latex_text, ImVec2(-1, model.latex_editor_h));
        draw_resize_handle("##latex_resize", model.latex_editor_h);
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
        InputTextMultilineStr("##plain", model.plain_text, ImVec2(-1, model.plain_editor_h));
        draw_resize_handle("##plain_resize", model.plain_editor_h);
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
        InputTextMultilineStr("##funcs", model.func_defs_text, ImVec2(-1, model.funcs_editor_h));
        draw_resize_handle("##funcs_resize", model.funcs_editor_h);
        if (ImGui::CollapsingHeader("Auxiliary function examples")) {
            ImGui::TextDisabled(
                "h(x) = m_1 x + \\frac{1}{2}(m_0-m_1)(|x+1| - |x-1|)\n"
                "Then call h(x) in equations. Body is inlined.\n"
                "IMPORTANT: function params (m_0, m_1) must be in the alphabet too.");
        }
    }

    // Variables / Parameters (новый раздельный формат)
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

    // Legacy: единый алфавит
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
    // scheme_ccd4 used to be missing from both buttons, so Complex CD4 was neither
    // selected nor cleared by them.
    if (ImGui::Button("Select all")) model.scheme_euler = model.scheme_cromer = model.scheme_midpoint = model.scheme_rk4 = model.scheme_dopri78 = model.scheme_cd = model.scheme_ccd = model.scheme_ccd4 = model.scheme_ieuler = model.scheme_imidpoint = model.scheme_semp = model.scheme_simp = model.scheme_dmethod = true;
    ImGui::SameLine();
    if (ImGui::Button("Clear all"))  model.scheme_euler = model.scheme_cromer = model.scheme_midpoint = model.scheme_rk4 = model.scheme_dopri78 = model.scheme_cd = model.scheme_ccd = model.scheme_ccd4 = model.scheme_ieuler = model.scheme_imidpoint = model.scheme_semp = model.scheme_simp = model.scheme_dmethod = false;

    // Схемы сгруппированы по порядку точности — тот же порядок, что в комбо
    // выбора схемы (kBuiltinSchemes) и в списке генерации (AppModel::generate).
    // У CD, Complex CD, SEMP и SIMP порядок группы достигается только при
    // s = a[0] = 0.5; при других s все четыре падают до первого.
    ImGui::SeparatorText("Order 1");
    ImGui::Checkbox("Euler", &model.scheme_euler); ImGui::SameLine();
    ImGui::Checkbox("Euler-Cromer", &model.scheme_cromer); ImGui::SameLine();
    ImGui::Checkbox("D", &model.scheme_dmethod);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Diagonally-implicit first-order method: the stage of SIMP\n"
                          "(and the Phi* half-step of CD) taken as a step of its own,\n"
                          "on the full h. X[i] = X + h*f_i(X) is solved for its own\n"
                          "variable (analytically when linear in it, 4 fixed-point\n"
                          "iterations otherwise), forward order, so each equation\n"
                          "already sees the new X[0..i-1].\n"
                          "The diagonally-implicit twin of Euler-Cromer: no Jacobian,\n"
                          "no Newton, and cross terms stay outside the solve.");
    ImGui::SameLine();
    ImGui::Checkbox("Implicit Euler", &model.scheme_ieuler);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("X_next = X + h*f(X_next), solved by Newton on a\n"
                          "symbolically differentiated Jacobian.\n"
                          "Order 1, L-stable: unlike the explicit schemes it stays\n"
                          "bounded on stiff systems at steps where RK4 diverges.\n"
                          "Needs a differentiable RHS (no floor/ceil/fmod).");

    ImGui::SeparatorText("Order 2");
    ImGui::Checkbox("Explicit Midpoint", &model.scheme_midpoint); ImGui::SameLine();
    ImGui::Checkbox("Implicit Midpoint", &model.scheme_imidpoint);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Solved for the stage Y = (X + X_next)/2:\n"
                          "Y = X + (h/2)*f(Y), then X_next = 2*Y - X.\n"
                          "Order 2 (measured 2.00 on Lorenz), A-stable, symmetric\n"
                          "and symplectic. Same Newton solver as Implicit Euler.");
    ImGui::SameLine();
    ImGui::Checkbox("CD", &model.scheme_cd); ImGui::SameLine();
    ImGui::Checkbox("Complex CD", &model.scheme_ccd);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("CD with complex half-steps:\n"
                          "h1 = s*h + i*h*sqrt(3)/6, h2 = (1-s)*h - i*h*sqrt(3)/6,\n"
                          "s = a[0] is the same symmetry coefficient as in CD.\n"
                          "The step is evaluated in complex arithmetic; only Re is kept.\n"
                          "At s = 0.5 (default) the half-steps are conjugate and order is 2.");
    ImGui::SameLine();
    ImGui::Checkbox("Complex Implicit Euler", &model.scheme_cieuler);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Composition of TWO implicit Euler steps with conjugate\n"
                          "complex steps: tau1 = h*(s + i/2), tau2 = h*(1 - s - i/2),\n"
                          "s = a[0], the same symmetry slot as CD. Each half-step is a\n"
                          "full Newton solve on the symbolic Jacobian, in complex\n"
                          "arithmetic; Re is taken once at the end.\n"
                          "The coefficients are forced, not tuned: composing a first-\n"
                          "order method to second order needs tau1 + tau2 = h AND\n"
                          "tau1^2 + tau2^2 = 0, whose only solution is (1 +- i)/2.\n"
                          "Order 2 at s = 0.5 only; elsewhere the h^2 term keeps a real\n"
                          "part that Re does not remove and the order drops to 1.");
    ImGui::SameLine();
    ImGui::Checkbox("SEMP", &model.scheme_semp);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Semi-explicit midpoint. Stage: X~[i] = X[i] + h1*f_i(X),\n"
                          "component by component in forward order (each equation\n"
                          "already sees the new X[0..i-1]), h1 = s*h. Corrector:\n"
                          "X_next = X + h*f(X~). Two RHS passes, nothing to solve.\n"
                          "Order 2 at s = a[0] = 0.5 only: the expansion carries\n"
                          "s*h^2*F'F against h^2/2*F'F, so other s give order 1.");
    ImGui::SameLine();
    ImGui::Checkbox("SIMP", &model.scheme_simp);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Semi-implicit midpoint. Same corrector as SEMP, but the\n"
                          "stage is diagonally implicit: each equation is solved for\n"
                          "its own variable (analytically when linear in it, 4 fixed-\n"
                          "point iterations otherwise), exactly as Phi* does in CD.\n"
                          "No Jacobian, no Newton. Order 2 at s = a[0] = 0.5 only.");

    ImGui::SeparatorText("Order 4");
    ImGui::Checkbox("RK4", &model.scheme_rk4); ImGui::SameLine();
    ImGui::Checkbox("Complex CD4", &model.scheme_ccd4);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Composition of TWO symmetric CDs with complex steps:\n"
                          "pass 1 uses step gamma*h, pass 2 uses conj(gamma)*h,\n"
                          "gamma = 1/2 + i*sqrt(3)/6. Re is taken once at the end.\n"
                          "Order 4 (measured 4.00 on Lorenz and Rossler).\n"
                          "Requires s = a[0] = 0.5: for other s the inner CD is\n"
                          "asymmetric and the order drops to first.");

    ImGui::SeparatorText("Order 8");
    ImGui::Checkbox("DOPRI78", &model.scheme_dopri78);

    // Настройки Ньютона — общие для обеих неявных схем, поэтому живут здесь,
    // на уровне системы, а не в конфиге каждого анализа.
    if (model.scheme_ieuler || model.scheme_imidpoint) {
        ImGui::Spacing();
        ImGui::Text("Newton solver:");
        int jac_mode = model.newton_full ? 1 : 0;
        ImGui::SetNextItemWidth(kFieldW * 2.0f);
        if (ImGui::Combo("jacobian", &jac_mode, "modified (once per step)\0full (every iteration)\0"))
            model.newton_full = (jac_mode == 1);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("modified: Jacobian and LU reused across iterations and\n"
                              "refreshed when the correction stops shrinking.\n"
                              "Cheaper on GPU and the right default.\n"
                              "full: rebuilt every iteration. Quadratic convergence,\n"
                              "most robust at large h, ~k times the cost.");
        InputNumStr("tolerance", model.newton_tol, kFieldW);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Newton stops once ||dX|| falls below this.");
        InputNumStr("max iterations", model.newton_max_iters, kFieldW);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Hard cap on Newton passes per step. 2-3 is typical\n"
                              "from the explicit-Euler predictor.");
    }

    // Custom KRS schemes (raw C/CUDA код вместо codegen)
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
            float& body_h = model.custom_scheme_editor_h.try_emplace(cs.name, 100.0f).first->second;
            InputTextMultilineStr(body_label.c_str(), cs.body, ImVec2(-1, body_h));
            const std::string resize_id = "##cs_body_resize_" + std::to_string(i);
            draw_resize_handle(resize_id.c_str(), body_h);
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
        // (список встроенных — общий kBuiltinSchemes, см. верх файла)
        if (ImGui::Button("+ Add custom scheme")) {
            // подобрать уникальное имя "Custom N"
            int n = (int)model.custom_schemes.size() + 1;
            std::string candidate;
            auto name_clash = [&](const std::string& nm) {
                for (const auto& b : kBuiltinSchemes) if (nm == b.name) return true;
                for (const auto& cs : model.custom_schemes) if (cs.name == nm) return true;
                return false;
            };
            do { candidate = "Custom " + std::to_string(n++); } while (name_clash(candidate));
            CustomScheme cs; cs.name = candidate;
            model.custom_schemes.push_back(std::move(cs));
        }

        // подсветка конфликтов
        for (const auto& cs : model.custom_schemes) {
            for (const auto& b : kBuiltinSchemes) {
                if (cs.name == b.name) {
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
            ImVec2(-1, model.gen_code_editor_h), ImGuiInputTextFlags_ReadOnly);
        draw_resize_handle("##code_resize", model.gen_code_editor_h);
    }
}


// Вкладка Parameters: НУ, значения/диапазоны параметров, шаг
void draw_parameters_tab(AppModel& model) {
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


// Library tab: two states — list (table of saved systems + note preview) and editor
// (System + Parameters sub-tabs with Save/Cancel). The editor edits
// AppModel::library_edit_buffer (a scratch AppModel), never the live model, so
// browsing/editing/Cancel never affects the system active in Parametric/Phase/Basins/FastSync.

// Trim leading/trailing ASCII whitespace.
[[nodiscard]] std::string trim_copy(const std::string& s) {
    auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}


void invalidate_library_cache() {
    g_library_cache.valid = false;
    g_library_cache.note_name.clear();
    g_library_cache.note_text.clear();
}


const LibraryListCache& library_list_cached(SystemLibrary& lib) {
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


// Save-side logic for the editor: validate name, handle rename for EditExisting, persist
// the scratch buffer via SystemLibrary. Mirrors the save into the live model only if the
// edited system is the one currently active elsewhere — saving any other system must not
// disturb what Parametric/Phase/Basins/FastSync have loaded. On success returns to list
// mode; on failure sets model.edit_error and stays in the editor.
void library_editor_save(AppModel& model, SystemLibrary& lib) {
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
void draw_library_list(AppModel& model, SystemLibrary& lib) {
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
}
