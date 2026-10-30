#include "gui_internal.h"

// Возвращает директорию exe со слешем в конце. Реализована в app_main.cpp
// (там же используется для resolve_python_exe / library_dir).
extern std::string exe_dir();
std::string get_exe_dir_with_sep() {
    std::string d = exe_dir();
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

// Единственное место, где видны ОБА определения pi: configCUDA.h (его читает
// ядро) и num_parse_detail::kPi (его подставляет разбор полей ввода). Если они
// разъедутся, пользователь введёт "pi" и получит не то число, которое считает
// GPU, — молча. Поэтому сверка на этапе компиляции.
// Сравнение — в numb: при numb = float ядро видит pi, округлённое до float, и поле ввода
// после приведения к numb даёт ровно его же.
static_assert(::pi == static_cast<numb>(num_parse_detail::kPi),
              "pi in num_parse.h diverged from configCUDA.h");

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

// Многострочное поле прямо над std::string с ростом буфера (CallbackResize, как imgui_stdlib). Раньше —
// скретч "длина + 4096": вставка длиннее 4096 символов в пустое поле молча резалась на 4095 (так
// обрезалось тело КРС CRK7s8_L в Lorenz-CxSearch: 5245 символов -> 4095, посреди выражения).
static int input_str_resize(ImGuiInputTextCallbackData* d) {
    if (d->EventFlag == ImGuiInputTextFlags_CallbackResize) {
        std::string* s = static_cast<std::string*>(d->UserData);
        s->resize((size_t)d->BufTextLen);
        d->Buf = s->data();
    }
    return 0;
}
bool InputTextMultilineStr(const char* label, std::string& str, const ImVec2& size) {
    return ImGui::InputTextMultiline(label, str.data(), str.capacity() + 1, size,
                                     ImGuiInputTextFlags_CallbackResize, input_str_resize, &str);
}

// Drag handle below a multiline input; drags along Y and clamps `height`.
void draw_resize_handle(const char* id, float& height,
                               float min_h, float max_h) {
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

// Превью LaTeX-поля: те же уравнения, набранные верстальщиком подписей графиков
// (math_formula_draw в plot_axis.cpp). Уравнения делятся по `\\` и переводам
// строки; строка, от которой после разметки (\begin{cases} и т.п.) ничего не
// осталось, пропускается. Состояния нет — всё считается из текста каждый кадр.
void draw_latex_preview(const char* id, const std::string& src, ImVec2 size) {
    std::vector<std::string> lines;
    std::string cur;
    auto push = [&] {
        const size_t a = cur.find_first_not_of(" \t\r");
        if (a != std::string::npos) lines.push_back(cur.substr(a));
        cur.clear();
    };
    for (size_t i = 0; i < src.size(); ++i) {
        if (src[i] == '\n') { push(); continue; }
        if (src[i] == '\\' && i + 1 < src.size() && src[i + 1] == '\\') { push(); ++i; continue; }
        cur += src[i];
    }
    push();

    ImGui::BeginChild(id, size, ImGuiChildFlags_Borders | ImGuiChildFlags_FrameStyle,
                      ImGuiWindowFlags_HorizontalScrollbar);
    const float fs  = std::floor(ImGui::GetFontSize() * 1.3f + 0.5f);
    const ImU32 col = ImGui::GetColorU32(ImGuiCol_Text);
    ImDrawList* dl  = ImGui::GetWindowDrawList();
    bool any = false;
    for (const std::string& ln : lines) {
        const MathExtent ext = math_formula_extent(ln.c_str(), fs);
        if (ext.width <= 0.0f) continue;
        const ImVec2 p = ImGui::GetCursorScreenPos();
        ImGui::Dummy(ImVec2(ext.width, ext.ascent + ext.descent + fs * 0.2f));
        if (ImGui::IsItemVisible()) math_formula_draw(dl, p, col, ln.c_str(), fs);
        any = true;
    }
    if (!any) ImGui::TextDisabled("(formula preview)");
    ImGui::EndChild();
}

// Правые части в том виде, в каком их получает кодогенератор: над X[..] и
// a[..], с легендой имён. Это вход ВСЕХ схем (codegen_rhs), поэтому здесь
// видно и порядок параметров в a[], и ошибку разбора — до Generate/Run.
// Поле только для чтения, но выделяется и копируется; высота — по строкам.
void draw_rhs_preview(const AppModel& model) {
    const std::string& src =
        (model.mode == InputMode::Plain) ? model.plain_text : model.latex_text;
    if (src.find_first_not_of(" \t\r\n") == std::string::npos) return;
    if (!ImGui::CollapsingHeader("Kernel form: X[..], a[..]", ImGuiTreeNodeFlags_DefaultOpen))
        return;
    const AppModel::RhsPreview& pv = model.rhs_preview();
    if (pv.text.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.35f, 1.0f), "Parse error: %s", pv.error.c_str());
        return;
    }
    // Длинная правая часть переносится, а не уезжает за край, поэтому высоту
    // меряем с тем же переносом (ширина поля минус отступы и полоса прокрутки).
    const ImGuiStyle& st = ImGui::GetStyle();
    const float wrap_w = ImGui::GetContentRegionAvail().x - st.FramePadding.x * 2.0f
                       - st.ScrollbarSize;
    const float text_h = ImGui::CalcTextSize(pv.text.c_str(), nullptr, false,
                                             std::max(wrap_w, 50.0f)).y;
    const float h = std::min(text_h + st.FramePadding.y * 2.0f + 2.0f, 320.0f);
    ImGui::InputTextMultiline("##rhs_preview", (char*)pv.text.c_str(), pv.text.size() + 1,
                              ImVec2(-1, h),
                              ImGuiInputTextFlags_ReadOnly | ImGuiInputTextFlags_WordWrap);
}
bool InputTextStr(const char* label, std::string& str, float width) {
    std::vector<char>& buf = input_scratch(str, 1024);
    if (width > 0) ImGui::SetNextItemWidth(width);
    bool changed = ImGui::InputText(label, buf.data(), buf.size());
    if (changed) str = buf.data();
    return changed;
}

// То же с подсказкой в пустом поле. Нужна там, где пустое поле означает не
// "ничего", а "значение по умолчанию": имя схемы-обёртки пустым берётся из
// её же описания, и подсказка показывает, какое имя получится.
bool InputTextStrHint(const char* label, std::string& str, const char* hint,
                             float width) {
    std::vector<char>& buf = input_scratch(str, 1024);
    if (width > 0) ImGui::SetNextItemWidth(width);
    bool changed = ImGui::InputTextWithHint(label, hint, buf.data(), buf.size());
    if (changed) str = buf.data();
    return changed;
}

// Перехватываем символы ДО того, как ImGui положит их в буфер: замена ',' → '.'
// идёт в момент ввода и не подменяет строку между кадрами. Иначе InputText на
// каждом кадре видит внешнюю правку буфера, возвращает changed=true, и
// auto_recompute триггерится непрерывно, пока в поле есть запятая.
static int filter_comma_to_dot(ImGuiInputTextCallbackData* data) {
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
[[nodiscard]] static bool is_numeric_string(const std::string& s) {
    if (s.empty()) return true;
    double v = 0.0;
    return parse_num_checked(s, v);
}

// Is `scheme_name` one of the session's custom KRS schemes (as opposed to a
// built-in name)?
[[nodiscard]] static bool is_custom_scheme(const std::string& scheme_name,
                              const std::vector<CustomScheme>& custom_schemes) {
    for (const auto& cs : custom_schemes) if (cs.name == scheme_name) return true;
    return false;
}

// Требует ли схема КОМПИЛЯЦИИ тела, а не интерпретации AST. Таких две группы:
// кастомные КРС (сырой C) и экстраполяционные обёртки (тело базы, размноженное
// по стадиям внутри лямбды — интерпретатору выражений это не по зубам).
// На GPU разницы нет, обе идут через NVRTC; на CPU обеим нужен внешний cl.exe.
// Тот же предикат решает krs_is_custom при сборке Request — см. analysis_session.cpp.
[[nodiscard]] bool scheme_needs_compiled_body(const std::string& scheme_name,
                              const std::vector<CustomScheme>& custom_schemes) {
    return is_custom_scheme(scheme_name, custom_schemes)
        || parse_extrapolation_name(scheme_name, nullptr)
        || parse_composition_name(scheme_name, nullptr);
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

// Читает ли схема коэффициент симметрии a[0] — от этого зависит, показывать ли
// поле symmetry в конфигах и пункт "s (a[0])" в выборе цели свипа. Встроенных
// таких пять: CD (h1 = s*h, h2 = (1-s)*h), оба Complex CD (то же деление шага,
// плюс мнимая часть ±h*sqrt(3)/6) и SEMP/SIMP (стадия на полушаге h1 = s*h).
// Один предикат на весь файл: раньше проверка была выписана четырьмя копиями
// `scheme == "CD" || custom_scheme_uses_symmetry(...)`, и добавление схемы
// требовало не забыть все четыре.
[[nodiscard]] bool scheme_uses_symmetry(const std::string& scheme_name,
                                         const std::vector<CustomScheme>& custom_schemes) {
    // Экстраполяционная обёртка читает a[0] ровно тогда, когда его читает база:
    // её тело — это тело базы, повторённое по стадиям. Без этой развёртки выбор
    // "Extr(CD|1,2)" прятал бы поле s, которым обёрнутый CD пользуется.
    ExtrapolationSpec spec;
    if (parse_extrapolation_name(scheme_name, &spec))
        return scheme_uses_symmetry(spec.base, custom_schemes);
    // Same for a composition: its body is the base repeated with scaled steps,
    // so hiding s under "Comp(CD|...)" would hide a knob the inner CD reads.
    CompositionSpec cspec;
    if (parse_composition_name(scheme_name, &cspec))
        return scheme_uses_symmetry(cspec.base, custom_schemes);
    return scheme_name == "CD" || scheme_name == "CD fast"
        || scheme_name == "CD FP" || scheme_name == "CD fast FP"
        || scheme_name == "Complex CD" || scheme_name == "Complex CD4"
        || scheme_name == "CCD4 (o4s3)" || scheme_name == "CCD4 (o4s4)"
        || scheme_name == "CD_{1-0}" || scheme_name == "Complex CD_{1-0}"
        || scheme_name == "Complex CD4_{1-0}" || scheme_name == "CCD4 (o4s3)_{1-0}"
        || scheme_name == "CCD4 (o4s4)_{1-0}"
        || scheme_name == "Complex Implicit Euler"
        || scheme_name == "SEMP" || scheme_name == "SIMP"
        || custom_scheme_uses_symmetry(scheme_name, custom_schemes);
}

// Читает ли схема число простых итераций CD FP из a[M+1] — показывать ли поле
// "FP iterations n". Обёртки Extr/Comp — по базе, как у scheme_uses_symmetry.
[[nodiscard]] bool scheme_uses_cd_fp_iters(const std::string& scheme_name) {
    ExtrapolationSpec spec;
    if (parse_extrapolation_name(scheme_name, &spec))
        return scheme_uses_cd_fp_iters(spec.base);
    CompositionSpec cspec;
    if (parse_composition_name(scheme_name, &cspec))
        return scheme_uses_cd_fp_iters(cspec.base);
    return scheme_name == "CD FP" || scheme_name == "CD fast FP";
}

// Решает ли схема шаг Ньютоном — показывать ли поля Newton (NewtonSettings, a[M+2..M+4]).
// Обёртки Extr/Comp — по базе.
[[nodiscard]] bool scheme_uses_newton(const std::string& scheme_name) {
    ExtrapolationSpec spec;
    if (parse_extrapolation_name(scheme_name, &spec))
        return scheme_uses_newton(spec.base);
    CompositionSpec cspec;
    if (parse_composition_name(scheme_name, &cspec))
        return scheme_uses_newton(cspec.base);
    return scheme_name == "Implicit Euler" || scheme_name == "Implicit Midpoint"
        || scheme_name == "Complex Implicit Euler"
        || dae_scheme_from_name(scheme_name, nullptr);   // DAE BDF1 / Trapezoidal
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
                                   bool sweep_over_h) {
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
[[nodiscard]] static std::string sweep_range_label(const std::string& axis,
                                                   const std::string& lo_text,
                                                   const std::string& hi_text) {
    const std::string lo = lo_text.empty() ? std::string("?") : lo_text;
    const std::string hi = hi_text.empty() ? std::string("?") : hi_text;
    return axis + " [" + lo + ".." + hi + "]";
}

// Устойчивая часть подписи — всё до " [", то есть имя оси без диапазона.
// Нужна отмене (ViewRangeTarget::relocate), которая опознаёт конфиг по подписи.
// Автоподпись 1D-диаграммы — это "<ось> [lo..hi]" (sweep_range_label выше), а
// диапазон в ней переписывает ровно та команда, которую потом откатывают: сверять
// подпись целиком значило сверять её с тем, что сам же и изменил, и сверка
// проваливалась всегда. У 2D автоподпись "<x> x <y>" скобок не содержит, у ручных
// имён их обычно нет — там функция возвращает строку как есть, то есть поведение
// прежнее. Отсюда и симптом «Ctrl+Z не работает именно в 1D».
[[nodiscard]] std::string label_identity(const std::string& s) {
    const size_t k = s.rfind(" [");
    return k == std::string::npos ? s : s.substr(0, k);
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
                    : w.kind == ParametricPlotWindow::Kind::LLE ? "LLE"
                    : w.kind == ParametricPlotWindow::Kind::LS  ? "LS" : "Metrics";
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
    for (auto& c : model.metrics_session.configs)
        if (!c.label_is_manual)
            c.label = auto_label_sweep(c, model.metrics_session.params, model.metrics_session.vars);
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

// Распарсилось ли поле вообще. parse_num молча отдаёт default, и вызывающий не
// отличает «поле пустое» от «в поле ноль»; пробуем два разных дефолта —
// совпали, значит число прочиталось.
[[nodiscard]] static inline bool parse_num_ok(const std::string& v, double& out) {
    const double a = parse_ratio_or(v, 0.0);
    const double b = parse_ratio_or(v, 1.0);
    if (a != b) return false;
    out = a;
    return true;
}

// ctx_menu — контекстное меню поля (ПКМ). Вызывается СРАЗУ после InputText:
// ниже может встать предупреждение о невалидном числе, и BeginPopupContextItem
// внутри колбэка привязался бы к нему, а не к самому полю.
// allow_empty: пустая строка — осмысленное значение («как у всех», см. таблицу
// узлов вкладки Network), и ругаться на неё как на нечисло нельзя.
bool InputNumStr(const char* label, std::string& str, float width,
                        const std::function<void()>& ctx_menu,
                        bool allow_empty) {
    std::vector<char>& buf = input_scratch(str, 1024);
    if (width > 0) ImGui::SetNextItemWidth(width);
    // CallbackHistory — ↑/↓ в активном InputText, обрабатываем в digit_step_input_callback.
    // CallbackCharFilter — прежняя замена запятой на точку, тоже в нём.
    bool changed = ImGui::InputText(label, buf.data(), buf.size(),
        ImGuiInputTextFlags_CallbackCharFilter | ImGuiInputTextFlags_CallbackHistory,
        digit_step_input_callback);
    if (changed) str = buf.data();
    if (ctx_menu) ctx_menu();

    // Inline-предупреждение, если содержимое не парсится как число.
    // Default из engine'а (0) всё равно применится, но пользователю
    // явно сигналим, что введённое значение игнорируется.
    if (!(allow_empty && str.empty()) && !is_numeric_string(str)) {
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.3f, 1.0f),
            "  invalid number, using default");
    }
    return changed;
}

// ↑/↓ в поле-списке чисел через ',' или ';' (atol): шагает цифру того значения, в котором
// стоит каретка, как digit_step_input_callback у одиночного числа. Пробелы вокруг значения
// не трогаются. Запятая здесь разделитель, поэтому в точку не превращается.
static int list_digit_step_callback(ImGuiInputTextCallbackData* data) {
    if (data->EventFlag != ImGuiInputTextFlags_CallbackHistory) return 0;
    int dir = 0;
    if (data->EventKey == ImGuiKey_UpArrow)   dir = +1;
    if (data->EventKey == ImGuiKey_DownArrow) dir = -1;
    if (dir == 0) return 0;
    const std::string text(data->Buf, data->Buf + data->BufTextLen);
    const int cur = std::clamp(data->CursorPos, 0, (int)text.size());
    int b = cur, e = cur;
    while (b > 0 && text[b - 1] != ',' && text[b - 1] != ';') --b;
    while (e < (int)text.size() && text[e] != ',' && text[e] != ';') ++e;
    while (b < e && text[b] == ' ') ++b;
    while (e > b && text[e - 1] == ' ') --e;
    if (b >= e) return 0;
    std::string new_tok;
    int new_cur = 0;
    if (!DigitInput::ComputeStep(text.substr(b, e - b), std::clamp(cur - b, 0, e - b), dir,
                                 new_tok, new_cur))
        return 0;
    data->DeleteChars(b, e - b);
    data->InsertChars(b, new_tok.c_str());
    const int pos = std::clamp(b + new_cur, 0, data->BufTextLen);
    data->CursorPos = data->SelectionStart = data->SelectionEnd = pos;
    return 0;
}

// Поле-список чисел (atol: одно на все переменные или по одному через ',' / ';') со
// стрелками ↑/↓ по значению под кареткой.
static bool InputNumListStr(const char* label, std::string& str, float width = 0.0f) {
    std::vector<char>& buf = input_scratch(str, 1024);
    if (width > 0) ImGui::SetNextItemWidth(width);
    const bool changed = ImGui::InputText(label, buf.data(), buf.size(),
        ImGuiInputTextFlags_CallbackHistory, list_digit_step_callback);
    if (changed) str = buf.data();
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
                              float width) {
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

// Общие блоки панелей настроек: единственные реализации элементов, ранее
// скопированных по вкладкам Bif / LLE / LS / DFT / Basins / FastSync / Custom по
// 4–10 раз. Копии расходились (в Custom НУ подписывались "IC x" вместо "x (IC)",
// decimator был не у всех), и чинить приходилось каждую отдельно.
// ImGui-идентификаторы передаются строкой снаружи и побайтово совпадают с прежними —
// раскладка imgui.ini и состояние вкладок сохраняются.

// Откат последней массовой команды + отметка времени для подписи в шапке
// (модель про ImGui не знает, поэтому GetTime проставляется здесь).
void do_undo(AppModel& m) {
    if (m.undo_last()) m.undo_note_time = ImGui::GetTime();
}

void do_redo(AppModel& m) {
    if (m.redo_last()) m.undo_note_time = ImGui::GetTime();
}

// Пункты "Undo/Redo: ..." в контекстных меню. Дублируют хоткеи сознательно: те
// работают только вне поля ввода (внутри InputText своя отмена у ImGui), и без
// видимых пунктов это выглядело бы как "иногда работает". Стек свой у каждой
// вкладки, поэтому показывается ровно то, что откатится ЗДЕСЬ.
static void draw_undo_menu_item(AppModel* m) {
    if (!m) return;
    const std::string* u = m->undo_top();
    const std::string* r = m->redo_top();
    if (!u && !r) return;
    ImGui::Separator();
    if (u) {
        const std::string lbl = "Undo: " + *u;
        if (ImGui::MenuItem(lbl.c_str(), "Ctrl+Z")) do_undo(*m);
    }
    if (r) {
        const std::string lbl = "Redo: " + *r;
        if (ImGui::MenuItem(lbl.c_str(), "Ctrl+Shift+Z")) do_redo(*m);
    }
}

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
                                 const std::string& value) {
    if (!model) return;
    // ID popup'а обязан быть свой у каждого поля: в блоке Integration шесть
    // полей лежат на ОДНОМ уровне ID-стека, и общий идентификатор склеил бы
    // их меню в одно.
    const std::string popup_id = "##apply_all_" + label + name;
    if (!ImGui::BeginPopupContextItem(popup_id.c_str())) return;
    ImGui::TextDisabled("%s = %s", label.c_str(), value.c_str());
    ImGui::Separator();

    // Сухой прогон: сколько конфигов реально РАЗОЙДЁТСЯ с этим значением.
    // Показываем числом, но НЕ гасим по нему команду: серый пункт читается как
    // подпись, и "0 differ" выглядит как сломанная рассылка, хотя на деле все
    // цели уже с этим значением.
    auto differ_count = [&](const std::vector<BroadcastTarget>* only) {
        return model->broadcast_field(f, name, value, /*apply*/false, label, only, "");
    };
    // Применяет и кладёт отчёт в шапку: адресаты бывают не открыты, и без
    // строки сверху "применилось" неотличимо от "ничего не произошло".
    auto apply_now = [&](const std::vector<BroadcastTarget>* only, const std::string& note) {
        const int n = model->broadcast_field(f, name, value, /*apply*/true, label, only, note);
        model->undo_note = n > 0
            ? ("Applied: " + label + " = " + value + " -> " + note)
            : ("Nothing to apply: " + note + " already at " + value);
        model->undo_note_time = ImGui::GetTime();
    };
    // Подпись склеиваем, а НЕ прогоняем вторым проходом через printf: в шаблон
    // подставляется имя группы, и попади в него '%', результат стал бы
    // спецификацией формата с несуществующим аргументом. Сегодня имена — жёсткие
    // литералы из group_of, но держать это на соглашении незачем.
    auto count_and_item = [&](const std::string& title,
                              const std::vector<BroadcastTarget>* only,
                              const std::string& note) {
        const std::string lbl = title + " (" + std::to_string(differ_count(only)) + " differ)";
        if (ImGui::MenuItem(lbl.c_str())) apply_now(only, note);
    };

    const std::vector<BroadcastTarget> all = model->broadcast_targets(f);

    // 1. Свой тип. Источник берём из панели, которая сейчас рисуется
    //    (broadcast_source_*): по app_mode его не вывести — вкладка Parametric
    //    покрывает сразу Bifurcation, LLE и LS.
    std::vector<BroadcastTarget> same_group;
    std::string group;
    for (const auto& t : all)
        if (t.tab == model->broadcast_source_tab && t.idx == model->broadcast_source_idx) {
            group = t.group;
            break;
        }
    if (!group.empty()) {
        for (const auto& t : all)
            if (t.group == group) same_group.push_back(t);
    }
    if (same_group.size() > 1)
        count_and_item("Apply to all " + group, &same_group, "all " + group);

    // 2. Выбор галочками. Маска живёт в модели и пересобирается, когда
    //    сменилось поле или изменился состав целей (диаграмму добавили/закрыли).
    if (ImGui::BeginMenu("Apply to...")) {
        // Подпись состава: поле + имена целей по порядку. Меняется состав —
        // маска собирается заново, а не переезжает на соседние конфиги.
        std::string sig = std::to_string((int)f);
        for (const auto& t : all) { sig += '\x1f'; sig += t.label; }
        if (model->broadcast_pick_sig != sig ||
            model->broadcast_pick.size() != all.size()) {
            model->broadcast_pick.assign(all.size(), '0');
            model->broadcast_pick_sig = sig;
            // По умолчанию отмечаем свой тип: это и подсказка, что здесь
            // выбирают, и самый частый набор.
            for (size_t i = 0; i < all.size(); ++i)
                if (all[i].group == group) model->broadcast_pick[i] = '1';
        }
        if (ImGui::SmallButton("All")) model->broadcast_pick.assign(all.size(), '1');
        ImGui::SameLine();
        if (ImGui::SmallButton("None")) model->broadcast_pick.assign(all.size(), '0');

        // Кнопка применения — СВЕРХУ, до списка: галочек бывает под два десятка,
        // и внизу она уезжает за край экрана. Кнопка, а не пункт меню: пункт в
        // конце списка читается как подпись, а не как действие.
        {
            std::vector<BroadcastTarget> sel;
            for (size_t i = 0; i < all.size(); ++i)
                if (model->broadcast_pick[i] == '1') sel.push_back(all[i]);
            char btn[96];
            std::snprintf(btn, sizeof(btn), "Apply to %d selected", (int)sel.size());
            if (sel.empty()) ImGui::BeginDisabled();
            if (ImGui::Button(btn)) {
                char note[64];
                std::snprintf(note, sizeof(note), "%d selected", (int)sel.size());
                apply_now(&sel, note);
                ImGui::CloseCurrentPopup();
            }
            if (sel.empty()) ImGui::EndDisabled();
            ImGui::SameLine();
            if (sel.empty()) {
                ImGui::TextDisabled("nothing ticked");
            } else {
                const int d = differ_count(&sel);
                if (d > 0) ImGui::TextDisabled("%d differ", d);
                else       ImGui::TextDisabled("already at this value");
            }
        }
        ImGui::Separator();

        std::string shown_group;
        for (size_t i = 0; i < all.size(); ++i) {
            if (all[i].group != shown_group) {
                shown_group = all[i].group;
                ImGui::SeparatorText(shown_group.c_str());
            }
            ImGui::PushID((int)i);
            bool on = (model->broadcast_pick[i] == '1');
            if (ImGui::Checkbox("##pick", &on))
                model->broadcast_pick[i] = on ? '1' : '0';
            ImGui::SameLine();
            ImGui::TextUnformatted(all[i].label.c_str());
            ImGui::PopID();
        }

        ImGui::EndMenu();
    }

    // 3. Прежний адресат: активный конфиг каждой вкладки (only = nullptr).
    count_and_item("Apply to all calculation tabs", nullptr, "all tabs");

    draw_undo_menu_item(model);
    ImGui::EndPopup();
}

// Заголовок группы в комбо и чекбоксах: своя группа или "Order N".
[[nodiscard]] std::string scheme_group_title(const BuiltinScheme& b) {
    return b.group ? std::string(b.group) : "Order " + std::to_string(b.order);
}

// "Is a map" predicate for UI code with no AppModel at hand: load_from_record
// pins scheme to "Map" for a map system, and that name is barred for custom KRS.
[[nodiscard]] bool scheme_is_map(const std::string& name) {
    return name == kMapSchemeName;
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
// bc — модель для ПКМ "Apply to all calculation tabs" (nullptr = не предлагать).
bool draw_scheme_combo(const char* label, std::string& scheme,
                              const std::vector<CustomScheme>& custom_schemes,
                              const std::function<void(const std::string&)>& on_pick,
                              const std::vector<std::string>* enabled_builtins,
                              AppModel* bc,
                              bool is_map,
                              const std::vector<std::string>* wrapper_schemes,
                              bool broadcast) {
    // Nothing to choose for a discrete map: the step is the right-hand side.
    if (is_map) {
        scheme = kMapSchemeName;
        ImGui::TextDisabled("Map (x_{n+1} = f(x_n)) - no scheme");
        return false;
    }
    bool picked = false;
    auto choose = [&](const std::string& nm) {
        scheme = nm;
        if (on_pick) on_pick(nm);
        picked = true;
    };
    // ДАУ: сессия получила только схемы ДАУ (enabled_builtins_from_record) — их и показываем,
    // схемы ОДУ на такой системе не строятся.
    if (enabled_builtins && !enabled_builtins->empty()
        && (dae_scheme_from_name((*enabled_builtins)[0], nullptr) || (*enabled_builtins)[0] == kBdfSchemeDae)) {
        ImGui::SetNextItemWidth(kComboW);
        if (ImGui::BeginCombo(label, scheme.c_str())) {
            ImGui::SeparatorText("DAE");
            bool ms_header = false;
            for (const auto& nm : *enabled_builtins) {
                if (scheme_is_multistep(nm) && !ms_header) {
                    ImGui::SeparatorText("multistep (adaptive step only)");
                    ms_header = true;
                }
                if (ImGui::Selectable(nm.c_str(), scheme == nm)) choose(nm);
                if (scheme_is_multistep(nm) && ImGui::IsItemHovered())
                    ImGui::SetTooltip("BDF orders 1-5 with variable step (DASSL form), its own driver.\n"
                                      "Needs Integration -> Adaptive step; index 3 goes through GGL.");
            }
            ImGui::EndCombo();
        }
        return picked;
    }
    // enabled_builtins null = no filter. Empty = no built-in is ticked: that is still
    // "no filter" only while the combo would otherwise be empty; a system that offers
    // custom KRS or wrappers (e.g. Izhikevich, whose reset lives in custom schemes)
    // gets just those. Currently-selected built-in stays visible even if not enabled,
    // so sessions saved with a now-disabled scheme aren't stranded.
    bool has_other = wrapper_schemes && !wrapper_schemes->empty();
    for (const auto& cs : custom_schemes) if (cs.enabled) { has_other = true; break; }
    const bool filter = enabled_builtins && (!enabled_builtins->empty() || has_other);
    auto is_enabled = [&](const char* m) {
        if (!filter) return true;
        for (const auto& n : *enabled_builtins) if (n == m) return true;
        return false;
    };
    ImGui::SetNextItemWidth(kComboW);
    if (ImGui::BeginCombo(label, wrapper_display_name(scheme).c_str())) {
        // Заголовок порядка печатаем лениво — перед ПЕРВОЙ видимой схемой
        // группы, иначе отфильтрованная группа оставила бы пустую шапку.
        std::string shown_group;
        for (const auto& b : kBuiltinSchemes) {
            if (!is_enabled(b.name) && scheme != b.name) continue;
            if (scheme_group_title(b) != shown_group) {
                ImGui::SeparatorText(scheme_group_title(b).c_str());
                shown_group = scheme_group_title(b);
            }
            if (ImGui::Selectable(b.name, scheme == b.name)) choose(b.name);
        }
        // Многошаговый BDF 1-5 — свой драйвер, флага scheme_* у него нет: показывается всегда
        // (enabled_builtins_from_record его тоже всегда добавляет), только с адаптивным шагом.
        if (is_enabled(kBdfSchemeOde) || scheme == kBdfSchemeOde) {
            ImGui::SeparatorText("Multistep (adaptive step only)");
            if (ImGui::Selectable(kBdfSchemeOde, scheme == kBdfSchemeOde)) choose(kBdfSchemeOde);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("BDF orders 1-5 with variable step (DASSL form), its own driver.\n"
                                  "Needs Integration -> Adaptive step; stiff problems.");
        }
        // Своя шапка, а не голая черта: у групп порядка и у обёрток заголовки
        // есть, и без неё кастомные читались как хвост последней Order-группы.
        // Выключенная галочкой КРС не предлагается, но уже выбранная остаётся
        // видна — как и у встроенных, чтобы сессия не осиротела.
        bool custom_header = false;
        for (const auto& cs : custom_schemes) {
            if (!cs.enabled && scheme != cs.name) continue;
            if (!custom_header) { ImGui::SeparatorText("Custom"); custom_header = true; }
            if (ImGui::Selectable((cs.name + "##custom").c_str(), scheme == cs.name))
                choose(cs.name);
        }
        // Обёртки — отдельной группой: паспортного порядка в таблице у них
        // нет (у Extr он считается из базы и числа стадий, у Comp вообще
        // известен только в рантайме), поэтому в группировку по Order выше
        // они не встают.
        // Сессии получают только включённые обёртки; выбранная, но выключенная
        // добавляется обратно по той же причине, что и КРС выше.
        std::vector<std::string> sorted;
        if (wrapper_schemes) sorted = *wrapper_schemes;
        {
            ExtrapolationSpec esp;
            CompositionSpec   csp;
            if ((parse_extrapolation_name(scheme, &esp) || parse_composition_name(scheme, &csp))
                && std::find(sorted.begin(), sorted.end(), scheme) == sorted.end())
                sorted.push_back(scheme);
        }
        if (!sorted.empty()) {
            ImGui::SeparatorText("Wrappers");
            // Порядок sort_wrapper_schemes: список сессии мог приехать из файла
            // в старом порядке добавления. Подзаголовок — на смене группы
            // "тип + опорник".
            sort_wrapper_schemes(sorted);
            std::string shown_wgroup;
            for (const auto& nm : sorted) {
                const std::string wg = wrapper_group_title(nm);
                if (wg != shown_wgroup) {
                    if (!wg.empty()) ImGui::TextDisabled("%s", wg.c_str());
                    shown_wgroup = wg;
                }
                const std::string shown = wrapper_display_name(nm);
                if (ImGui::Selectable((shown + "##" + nm).c_str(), scheme == nm))
                    choose(nm);
                // Коэффициенты прямо в списке выбора: именно здесь схему и
                // выбирают, а знать, какие числа за ней стоят, надо ДО того.
                // Значения параметров берутся у bc, когда он есть: символьная
                // gamma без них раскрыта не будет, и отчёт честно скажет "?".
                if (ImGui::IsItemHovered()) {
                    static const std::map<std::string, std::string> kNoParams;
                    const std::string rep = wrapper_coefficients_text(
                        nm, custom_schemes, bc ? bc->param_values : kNoParams);
                    if (!rep.empty())      ImGui::SetTooltip("%s", rep.c_str());
                    else if (shown != nm)  ImGui::SetTooltip("%s", nm.c_str());
                }
            }
        }
        ImGui::EndCombo();
    }
    // Меню вешаем ПОСЛЕ комбо: последним элементом остаётся оно само, и ПКМ
    // попадает по нему, а не по строке раскрытого списка.
    // broadcast = false — комбо не основной схемы (эталон вкладки Order):
    // рассылка BroadcastField::Scheme записала бы значение в ОСНОВНУЮ схему
    // чужих конфигов. PushID(label): ID popup'а собирается из "scheme", и две
    // комбо схемы на одном уровне ID-стека склеивали бы меню в одно окно,
    // рисуя его содержимое дважды (конфликт ID у галочек "Apply to...").
    if (broadcast) {
        ImGui::PushID(label);
        field_apply_all_menu(bc, BroadcastField::Scheme, {}, "scheme", scheme);
        ImGui::PopID();
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
                                    bool* other_over_h,
                                    bool note_when_empty,
                                    float width,
                                    bool allow_s,
                                    AppModel* bc,
                                    BroadcastField which,
                                    bool allow_h,
                                    int* ad_axis,
                                    const AdaptiveSettings* ad) {
    // Адаптивный шаг: h — шаг вывода, его не свипают; вместо него ось может свипать
    // настройку шага (*ad_axis — AdaptiveAxisKind движка, 0 — обычная цель).
    const bool ad_on = ad_axis && ad && ad->enabled;
    if (ad_axis && !ad_on) *ad_axis = 0;
    if (ad_on) allow_h = false;
    const bool ad_sel = ad_on && *ad_axis != 0;
    // A map has no step, so there is nothing to sweep over.
    if (!allow_h) over_h = false;
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

    auto ad_name = [&](int kind) -> std::string {
        if (kind == kAdAxisRtol) return "rtol";
        if (kind == kAdAxisAtol) return "atol";
        if (kind == kAdAxisTol)  return "tol (atol = rtol x ratio)";
        AdaptiveCtrlResolved cr;
        std::string cerr;
        const bool known = ad && adaptive_resolve_ctrl(ad->ctrl, 7, cr, cerr);
        const int k = kind - kAdAxisCtrl;
        return "controller: " + ((known && k >= 0 && k < (int)cr.par.size()) ? cr.par[(size_t)k] : std::string("?"));
    };
    const std::string preview =
          ad_sel                      ? ad_name(*ad_axis)
        : over_h                      ? std::string("dt (h)")
        : (over_var && !vars.empty()) ? (vars[var_index] + " (IC)")
        : (par_index < 0)             ? std::string("s (a[0])")
        : (!params.empty())           ? params[par_index]
                                      : std::string("?");
    // Значение для ПКМ-рассылки: четыре поля цели одной строкой (см.
    // encode_sweep_target). Меню вешается на комбо в ОБОИХ случаях — и когда
    // список раскрыт, и когда нет: оно принадлежит комбо, а не его строкам.
    auto apply_all_menu = [&]() {
        field_apply_all_menu(bc, which, {}, label,
            encode_sweep_target(par_index, over_var, var_index, over_h));
    };
    ImGui::SetNextItemWidth(width);
    if (!ImGui::BeginCombo(label, preview.c_str())) { apply_all_menu(); return; }

    for (int i = 0; i < (int)params.size(); ++i) {
        const bool sel = !ad_sel && !over_var && !over_h && par_index == i;
        if (ImGui::Selectable(params[i].c_str(), sel)) {
            par_index = i; over_var = false; over_h = false;
            if (ad_axis) *ad_axis = 0;
        }
    }
    if (!params.empty() && !vars.empty()) ImGui::Separator();
    for (int i = 0; i < (int)vars.size(); ++i) {
        const bool sel = !ad_sel && over_var && !over_h && var_index == i;
        if (ImGui::Selectable((vars[i] + " (IC)").c_str(), sel)) {
            var_index = i; over_var = true; over_h = false;
            if (ad_axis) *ad_axis = 0;
        }
    }

    if (allow_s) {
        ImGui::Separator();
        if (ImGui::Selectable("s (a[0])", !ad_sel && !over_var && !over_h && par_index < 0)) {
            par_index = -1; over_var = false; over_h = false;
            if (ad_axis) *ad_axis = 0;
        }
    }
    if (allow_h) {
        ImGui::Separator();
        if (ImGui::Selectable("dt (h)", over_h)) {
            over_h = true; over_var = false;
            if (other_over_h) *other_over_h = false;   // ровно одна ось = h
        }
    }
    if (ad_on) {
        ImGui::Separator();
        auto item = [&](int kind) {
            if (ImGui::Selectable(ad_name(kind).c_str(), *ad_axis == kind)) {
                *ad_axis = kind; over_var = false; over_h = false;
            }
        };
        item(kAdAxisRtol); item(kAdAxisAtol); item(kAdAxisTol);
        AdaptiveCtrlResolved cr;
        std::string cerr;
        if (adaptive_resolve_ctrl(ad->ctrl, 7, cr, cerr))
            for (int k = 0; k < (int)cr.par.size(); ++k) item(kAdAxisCtrl + k);
    }
    ImGui::EndCombo();
    apply_all_menu();
}

// Все настройки адаптивного шага в модели — в фиксированном порядке, чтобы откат
// мог сопоставить их с сохранёнными значениями (и отказаться, если число конфигов
// с тех пор изменилось).
static std::vector<AdaptiveSettings*> collect_adaptive_slots(AppModel& m) {
    std::vector<AdaptiveSettings*> v;
    v.push_back(&m.phase_session.adaptive);
    // Custom: источник — общий конфиг вкладки (apply_shared_to_* раздаёт его уровням при
    // запуске); его Phase — вместе с ним, как это делает draw_shared_config.
    v.push_back(&m.custom_session.shared.adaptive);
    v.push_back(&m.custom_session.phase_session.adaptive);
    for (auto& d : m.bifurcation_session.diagrams) v.push_back(&d.adaptive);
    for (auto& c : m.lle_session.curves)           v.push_back(&c.adaptive);
    for (auto& c : m.ls_session.curves)            v.push_back(&c.adaptive);
    for (auto& c : m.basins_session.configs)       v.push_back(&c.adaptive);
    for (auto& c : m.metrics_session.configs)      v.push_back(&c.adaptive);
    return v;
}

// «apply to all tabs»: одни и те же настройки шага во все вкладки, где они есть
// (Analysis, Custom, БД, LLE, LS, Basins, Metrics). Откат — Ctrl+Z.
static void broadcast_adaptive(AppModel& m, const AdaptiveSettings& a) {
    std::vector<AdaptiveSettings> old;
    for (AdaptiveSettings* p : collect_adaptive_slots(m)) { old.push_back(*p); *p = a; }
    AppModel* mp = &m;
    m.push_undo("adaptive step settings to all tabs",
        [mp, old]() {
            std::vector<AdaptiveSettings*> v = collect_adaptive_slots(*mp);
            if (v.size() == old.size())
                for (size_t i = 0; i < v.size(); ++i) *v[i] = old[i];
        },
        [mp, a]() { for (AdaptiveSettings* p : collect_adaptive_slots(*mp)) *p = a; });
}

// ---- Библиотека регуляторов шага ------------------------------------------------
// Окно редактора adaptive_ctrl_library(). Записи правятся на месте под замком библиотеки
// (расчётные потоки берут копию записи по имени); файл пишется, когда правка закончена —
// ни одно поле окна не активно.

static std::vector<std::pair<std::string, std::string>> user_ctrl_list() {
    AdaptiveCtrlLibrary& L = adaptive_ctrl_library();
    std::lock_guard<std::mutex> lk(L.mu);
    std::vector<std::pair<std::string, std::string>> v;
    for (const AdaptiveUserCtrl& u : L.items) {
#ifdef UCUDA_AD_HAIRER_ONLY
        if (u.kind == kUserCtrlFilter) continue;   // встроенный Filter не собран
#endif
        v.emplace_back(u.name, u.tip);
    }
    return v;
}

static const char* const kCtrlBodyHelp =
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

void draw_ctrl_library_window(AppModel& m) {
    // Результат фонового Check — и при закрытом окне, чтобы future не висел.
    if (m.ctrl_lib_check_future.valid()
        && m.ctrl_lib_check_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        const std::pair<bool, std::string> r = m.ctrl_lib_check_future.get();
        m.ctrl_lib_check_ok  = r.first;
        m.ctrl_lib_check_log = r.second;
    }
    if (!m.show_ctrl_library) return;
    ImGui::SetNextWindowSize(ImVec2(940.0f, 640.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Step controller library", &m.show_ctrl_library)) { ImGui::End(); return; }

    AdaptiveCtrlLibrary& L = adaptive_ctrl_library();
    bool edited = false;
    std::string check_body;
    bool do_check = false;
    {
        std::lock_guard<std::mutex> lk(L.mu);
        std::vector<AdaptiveUserCtrl>& items = L.items;
        if (m.ctrl_lib_sel >= (int)items.size()) m.ctrl_lib_sel = (int)items.size() - 1;

        auto taken = [&](const std::string& s, int self) {
            if (adaptive_find_builtin(s)) return true;
            for (int i = 0; i < (int)items.size(); ++i) if (i != self && items[(size_t)i].name == s) return true;
            return false;
        };
        auto add = [&](AdaptiveUserCtrl u) {
            std::string base = u.name.empty() ? std::string("controller") : u.name;
            u.name = base;
            for (int k = 2; taken(u.name, -1); ++k) u.name = base + " " + std::to_string(k);
            items.push_back(u);
            m.ctrl_lib_sel = (int)items.size() - 1;
            edited = true;
        };

        // Слева — список и кнопки.
        ImGui::BeginChild("##ctrl_list", ImVec2(230.0f, 0.0f), ImGuiChildFlags_Borders);
        if (ImGui::Button("New...")) ImGui::OpenPopup("##ctrl_new");
        if (ImGui::BeginPopup("##ctrl_new")) {
            if (ImGui::Selectable("empty C body")) {
                AdaptiveUserCtrl u;
                u.name = "my controller";
                u.body = "const numb err = ucuda_ctl_err(in);\n"
                         "o.err = err;\n"
                         "o.accept = err <= 1;\n"
                         "o.h = in.h * ucuda_clamp((numb)0.9 * ucuda_ctl_pow(err, (numb)-1 / (in.q + 1)),\n"
                         "                         (numb)0.2, o.accept ? (numb)5 : (numb)1);\n";
                add(u);
            }
#ifndef UCUDA_AD_HAIRER_ONLY   // фильтры — параметры встроенного Filter, а он не собран
            if (ImGui::Selectable("empty filter")) {
                AdaptiveUserCtrl u;
                u.name = "my filter"; u.kind = kUserCtrlFilter;
                for (double v : adaptive_ctrl_defaults(UCUDA_CTRL_FILTER, 7)) u.par_values.push_back(fmt_num_shortest(v));
                add(u);
            }
#endif
            int np = 0;
            const AdaptiveCtrlPreset* pr = adaptive_ctrl_presets(&np);
            ImGui::SeparatorText("C body presets");
            for (int i = 0; i < np; ++i) {
                if (pr[i].kind != kUserCtrlBody) continue;
                if (ImGui::Selectable(pr[i].name)) add(adaptive_ctrl_from_preset(pr[i]));
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", pr[i].tip);
            }
#ifndef UCUDA_AD_HAIRER_ONLY
            ImGui::SeparatorText("Soderlind filters");
            for (int i = 0; i < np; ++i) {
                if (pr[i].kind != kUserCtrlFilter) continue;
                if (ImGui::Selectable(pr[i].name)) add(adaptive_ctrl_from_preset(pr[i]));
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", pr[i].tip);
            }
#endif
            ImGui::EndPopup();
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(m.ctrl_lib_sel < 0);
        if (ImGui::Button("Copy") && m.ctrl_lib_sel >= 0) add(items[(size_t)m.ctrl_lib_sel]);
        ImGui::SameLine();
        if (ImGui::Button("Delete") && m.ctrl_lib_sel >= 0) {
            items.erase(items.begin() + m.ctrl_lib_sel);
            if (m.ctrl_lib_sel >= (int)items.size()) m.ctrl_lib_sel = (int)items.size() - 1;
            edited = true;
        }
        ImGui::EndDisabled();
        ImGui::Separator();
        for (int i = 0; i < (int)items.size(); ++i) {
            ImGui::PushID(i);
            const AdaptiveUserCtrl& u = items[(size_t)i];
            const std::string label = u.name + (u.kind == kUserCtrlFilter ? "  (filter)" : "");
            if (ImGui::Selectable(label.c_str(), m.ctrl_lib_sel == i)) m.ctrl_lib_sel = i;
            if (!u.tip.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", u.tip.c_str());
            ImGui::PopID();
        }
        if (items.empty()) ImGui::TextDisabled("Empty. New... adds a controller\nfrom scratch or from a preset.");
        ImGui::EndChild();

        // Справа — выбранная запись.
        ImGui::SameLine();
        ImGui::BeginChild("##ctrl_edit", ImVec2(0.0f, 0.0f), ImGuiChildFlags_Borders);
        if (m.ctrl_lib_sel >= 0) {
            AdaptiveUserCtrl& u = items[(size_t)m.ctrl_lib_sel];
            const ImVec4 warn(1.0f, 0.55f, 0.3f, 1.0f);
            edited |= InputTextStr("name", u.name, 260.0f);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Tabs refer to a controller by its name: after a rename they\n"
                                  "report the old name as unknown until you pick it again.");
            if (u.name.empty()) ImGui::TextColored(warn, "The name is empty.");
            else if (taken(u.name, m.ctrl_lib_sel))
                ImGui::TextColored(warn, "The name is taken (a built-in controller or another entry).");
            int kind = u.kind;
#ifdef UCUDA_AD_HAIRER_ONLY
            // Фильтр — параметры встроенного Filter, который в этой сборке не собран: такую
            // запись (из старой библиотеки) можно только перевести в C body или удалить.
            if (u.kind == kUserCtrlFilter) {
                ImGui::TextColored(warn, "Soderlind filters are not built in this version: the tabs do not offer it.");
                if (ImGui::SmallButton("make it a C body")) kind = kUserCtrlBody;
            }
#else
            ImGui::RadioButton("C body", &kind, kUserCtrlBody);
            ImGui::SameLine();
            ImGui::RadioButton("Soderlind filter", &kind, kUserCtrlFilter);
#endif
            if (kind != u.kind) {
                u.kind = kind;
                if (kind == kUserCtrlFilter) {
                    u.par_values.clear();
                    for (double v : adaptive_ctrl_defaults(UCUDA_CTRL_FILTER, 7)) u.par_values.push_back(fmt_num_shortest(v));
                } else if (u.par_names.size() != u.par_values.size()) {
                    u.par_values.resize(u.par_names.size());
                }
                edited = true;
            }
            edited |= InputTextStr("description", u.tip, 520.0f);

            if (u.kind == kUserCtrlFilter) {
#ifndef UCUDA_AD_HAIRER_ONLY   // иначе встроенного Filter нет (adaptive_find_builtin -> nullptr)
                const AdaptiveCtrlInfo* fi = adaptive_find_builtin("Filter");
                ImGui::TextDisabled("%s", fi->tip);
                if ((int)u.par_values.size() != fi->npar) {
                    const std::vector<double> d = adaptive_ctrl_defaults(UCUDA_CTRL_FILTER, 7);
                    u.par_values.resize((size_t)fi->npar);
                    for (int i = 0; i < fi->npar; ++i)
                        if (u.par_values[(size_t)i].empty()) u.par_values[(size_t)i] = fmt_num_shortest(d[(size_t)i]);
                    edited = true;
                }
                for (int i = 0; i < fi->npar; ++i) {
                    ImGui::PushID(i);
                    edited |= InputTextStr(fi->par[i], u.par_values[(size_t)i], kFieldW);
                    ImGui::PopID();
                }
#endif
            } else {
                bool hr = u.hairer_rules != 0;
                if (ImGui::Checkbox("Hairer's initial step and end-point rule", &hr)) { u.hairer_rules = hr; edited = true; }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("On: initial step as hinit of dopri5 / dop853 and the last step\n"
                                      "stretched when x + 1.01 h passes the end (as the built-in Hairer).\n"
                                      "Off: scipy's initial step, the step is cut exactly at the end.");
                ImGui::Text("Parameters c[k] (name, default):");
                if (u.par_values.size() != u.par_names.size()) u.par_values.resize(u.par_names.size());
                int remove = -1;
                for (int i = 0; i < (int)u.par_names.size(); ++i) {
                    ImGui::PushID(i);
                    ImGui::Text("c[%d]", i);
                    ImGui::SameLine(60.0f);
                    edited |= InputTextStr("##pn", u.par_names[(size_t)i], 140.0f);
                    ImGui::SameLine();
                    edited |= InputTextStr("##pv", u.par_values[(size_t)i], kFieldW);
                    ImGui::SameLine();
                    if (ImGui::SmallButton("x")) remove = i;
                    ImGui::PopID();
                }
                if (remove >= 0) {
                    u.par_names.erase(u.par_names.begin() + remove);
                    u.par_values.erase(u.par_values.begin() + remove);
                    edited = true;
                }
                ImGui::BeginDisabled(u.par_names.size() >= (size_t)UCUDA_CTL_NPAR);
                if (ImGui::SmallButton("+ parameter")) {
                    u.par_names.push_back("p" + std::to_string(u.par_names.size()));
                    u.par_values.push_back("0");
                    edited = true;
                }
                ImGui::EndDisabled();
                ImGui::SameLine();
                ImGui::TextDisabled("(up to %d; each is also a sweep axis)", UCUDA_CTL_NPAR);

                ImGui::Text("Prepare:");
                ImGui::SameLine();
                ImGui::TextDisabled("once per trajectory: k[0..7] from c[] and q, read by the body as in.k[]");
                edited |= InputTextMultilineStr("##prep", u.prep,
                                                ImVec2(-1.0f, ImGui::GetTextLineHeightWithSpacing() * 6.5f));
                // Проверка и её результат — по паре (подготовка, тело), как их собирает модуль.
                const std::string packed = adaptive_ctrl_pack(u.prep, u.body);
                const bool checked = !m.ctrl_lib_check_body.empty() && m.ctrl_lib_check_body == packed;
                ImGui::Text("Body:");
                ImGui::SameLine();
                ImGui::TextDisabled("(?)");
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", kCtrlBodyHelp);
                ImGui::SameLine();
                const bool checking = m.ctrl_lib_check_future.valid();
                ImGui::BeginDisabled(checking);
                if (ImGui::SmallButton("Check")) { do_check = true; check_body = packed; }
                ImGui::EndDisabled();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                    ImGui::SetTooltip("Compile the prepare section and the body with NVRTC (no run). The CPU\n"
                                      "build with cl.exe happens on the first CPU run of Analysis.");
                if (checked) {
                    ImGui::SameLine();
                    if (checking)                 ImGui::TextDisabled("checking...");
                    else if (m.ctrl_lib_check_ok) ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "compiles");
                    else                          ImGui::TextColored(warn, "does not compile:");
                }
                edited |= InputTextMultilineStr("##body", u.body, ImVec2(-1.0f, m.ctrl_lib_body_h));
                draw_resize_handle("##body_resize", m.ctrl_lib_body_h);
                if (checked && !m.ctrl_lib_check_log.empty())
                    ImGui::InputTextMultiline("##check_log", const_cast<char*>(m.ctrl_lib_check_log.c_str()),
                                              m.ctrl_lib_check_log.size() + 1, ImVec2(-1.0f, 110.0f),
                                              ImGuiInputTextFlags_ReadOnly);
                if (ImGui::TreeNode("API")) {
                    ImGui::TextUnformatted(kCtrlBodyHelp);
                    ImGui::TreePop();
                }
            }
        } else {
            ImGui::TextDisabled("Select a controller on the left, or add one with New...");
            ImGui::TextDisabled("Controllers of this library appear in the \"controller\" list of every tab.");
        }
        ImGui::EndChild();
    }
    if (do_check && !m.ctrl_lib_check_future.valid()) {
        // NVRTC в фоне: окно не замирает на время сборки. Результат забирается в начале кадра.
        m.ctrl_lib_check_body = check_body;
        m.ctrl_lib_check_ok   = false;
        m.ctrl_lib_check_log.clear();
        m.ctrl_lib_check_future = std::async(std::launch::async, [check_body]() {
            std::string log;
            const bool ok = nvrtc_check_ctrl_body(check_body, log);
            return std::make_pair(ok, log);
        });
    }
    if (edited) m.ctrl_lib_dirty = true;
    if (m.ctrl_lib_dirty && !ImGui::IsAnyItemActive()) {
        std::string err;
        m.ctrl_lib_status = save_ctrl_library(&err) ? std::string() : err;
        m.ctrl_lib_dirty = false;
    }
    if (!m.ctrl_lib_status.empty())
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", m.ctrl_lib_status.c_str());
    ImGui::End();
}

bool draw_adaptive_block(const char* id, AdaptiveSettings& a, const std::string& scheme,
                                int opts, bool is_map, AppModel* bc) {
    if (is_map) return false;   // у отображения шага нет
    ImGui::PushID(id);
    bool changed = false;
    const bool ok = adaptive_scheme_name_ok(scheme);

    int mode = a.enabled ? 1 : 0;
    changed |= ImGui::RadioButton("Fixed step", &mode, 0);
    ImGui::SameLine();
    ImGui::BeginDisabled(!ok && !a.enabled);
    changed |= ImGui::RadioButton("Adaptive step", &mode, 1);
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("Step size chosen by an error controller from the embedded\n"
                          "error estimate of the scheme. Needs RK45, DOPRI78, DOP853,\n"
                          "GBS 2-4 ... or an extrapolation Extr(...) with at least two stages:\n"
                          "its first K-1 stages give the lower-order solution, and the\n"
                          "dense output between step nodes is cubic Hermite.");
    a.enabled = (mode == 1);

    if (a.enabled) {
        if (!ok)
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.3f, 1.0f),
                               "The scheme has no error estimate: choose RK45, DOPRI78, DOP853, GBS 2-4 ...\n"
                               "or Extr(...) with at least two stages.");
        if (opts & kAdUiRaw) {
            int out = a.raw_nodes ? 1 : 0;
            changed |= ImGui::RadioButton("uniform grid", &out, 0);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Output on a uniform time grid through the dense output of\n"
                                  "the scheme; the field h is the output step.");
            ImGui::SameLine();
            changed |= ImGui::RadioButton("step nodes", &out, 1);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Output exactly at the accepted steps (including the steps\n"
                                  "shortened at the end of the transient and of the interval);\n"
                                  "the decimator keeps every N-th node, h is not used.");
            a.raw_nodes = (out == 1);
        }
        changed |= InputNumStr("rtol", a.rtol, kFieldW);
        changed |= InputNumListStr("atol", a.atol, kFieldW);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("One value for all variables, or one per variable separated by ';'.");
        changed |= InputNumStr("h0", a.h0, kFieldW, {}, true);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Initial step. Empty: chosen automatically.");
        changed |= InputNumStr("h_min", a.hmin, kFieldW, {}, true);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Smallest step. A step that would go below it is taken at h_min\n"
                              "anyway and counted as forced. Empty: the larger of 10 ulp(t) and\n"
                              "1e-12 x the length of the whole interval (transient + record).\n"
                              "A set value is raised to 10 ulp(t) where it is smaller: below that\n"
                              "t + h_min == t and the step would not advance time.");
        changed |= InputNumStr("h_max", a.hmax, kFieldW, {}, true);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Largest step. Empty: no limit.");
        changed |= InputNumStr("max rejects", a.max_rej, kFieldW, {}, true);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Rejections allowed per step. After this many in a row the next\n"
                              "attempt is made at h_min and taken unconditionally. Empty: no limit.\n"
                              "0: no retries - every attempt is taken, a rejection only shrinks the\n"
                              "next step (counted as forced); the tolerance becomes a target, not a bound.");

        int nb = 0;
        const AdaptiveCtrlInfo* bi = adaptive_builtin_ctrls(&nb);
        ImGui::SetNextItemWidth(kComboW);
        if (ImGui::BeginCombo("controller", a.ctrl.c_str())) {
            auto pick = [&](const std::string& name) {
                if (a.ctrl != name) { a.ctrl = name; a.ctrl_params.clear(); changed = true; }
            };
            for (int i = 0; i < nb; ++i) {
                if (ImGui::Selectable(bi[i].name, a.ctrl == bi[i].name)) pick(bi[i].name);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", bi[i].tip);
            }
            const std::vector<std::pair<std::string, std::string>> users = user_ctrl_list();
            if (!users.empty()) ImGui::SeparatorText("library");
            for (size_t i = 0; i < users.size(); ++i) {
                ImGui::PushID((int)i);
                if (ImGui::Selectable(users[i].first.c_str(), a.ctrl == users[i].first)) pick(users[i].first);
                if (!users[i].second.empty() && ImGui::IsItemHovered())
                    ImGui::SetTooltip("%s", users[i].second.c_str());
                ImGui::PopID();
            }
            ImGui::Separator();
            // Окно библиотеки — флаг в AppModel; без модели (bc == nullptr) пункта нет.
            if (bc != nullptr && ImGui::Selectable("edit library...")) bc->show_ctrl_library = true;
            ImGui::EndCombo();
        }
        AdaptiveCtrlResolved cur;
        std::string cur_err;
        if (!adaptive_resolve_ctrl(a.ctrl, adaptive_scheme_q(scheme), cur, cur_err)) {
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.3f, 1.0f), "%s", cur_err.c_str());
        } else {
            const int npar = (int)cur.par.size();
            const std::vector<double>& def = cur.def;
            if ((int)a.ctrl_params.size() != npar) {
                std::string line;
                for (int i = 0; i < npar; ++i) {
                    char buf[96];
                    std::snprintf(buf, sizeof(buf), "%s%s = %g", i ? ", " : "", cur.par[(size_t)i].c_str(),
                                  i < (int)def.size() ? def[(size_t)i] : 0.0);
                    line += buf;
                }
                if (npar > 0) {
                    ImGui::TextDisabled("%s", line.c_str());
                    ImGui::SameLine();
                    if (ImGui::SmallButton("edit")) {
                        a.ctrl_params.clear();
                        for (int i = 0; i < npar; ++i) a.ctrl_params.push_back(fmt_num_shortest(def[(size_t)i]));
                        changed = true;
                    }
                }
            } else {
                for (int i = 0; i < npar; ++i) {
                    ImGui::PushID(i);
                    changed |= InputNumStr(cur.par[(size_t)i].c_str(), a.ctrl_params[(size_t)i], kFieldW);
                    ImGui::PopID();
                }
                if (ImGui::SmallButton("defaults")) { a.ctrl_params.clear(); changed = true; }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(cur.user ? "Back to the defaults stored in the controller library."
                                               : "Back to the defaults for this scheme.");
            }
        }
        if ((opts & kAdUiInterp) && a.raw_nodes) {
            const char* items[] = { "node", "parabola", "Hermite by f" };
            ImGui::SetNextItemWidth(kComboW);
            changed |= ImGui::Combo("peak interpolation", &a.peak_interp, items, IM_ARRAYSIZE(items));
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("How a peak (in Metrics also a min / max) is placed between step nodes:\n"
                                  "node - the largest node itself;\n"
                                  "parabola - vertex of the parabola through three nodes\n"
                                  "  (the fixed-step formula generalised to unequal steps);\n"
                                  "Hermite by f - cubic on the step where dx/dt changes sign,\n"
                                  "  built from the nodes and the exact derivatives.");
        }
        if ((opts & kAdUiMaxPts) && a.raw_nodes)
            changed |= InputNumStr("max points", a.max_points, kFieldW);
        if (opts & kAdUiRenorm) {
            const char* items[] = { "exactly at k*NT", "first node after NT" };
            ImGui::SetNextItemWidth(kComboW);
            changed |= ImGui::Combo("renormalization", &a.lyap_renorm, items, IM_ARRAYSIZE(items));
        }
    }
    if (opts & kAdUiMinMax) {
        changed |= ImGui::Checkbox("min/max by interpolated extrema in Fixed", &a.minmax_interp_fixed);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("With the adaptive step min/max always use interpolated extrema: on the\n"
                              "uniform grid the vertex of the parabola through three samples (as peaks\n"
                              "do), on step nodes the peak interpolation. This applies the parabola to\n"
                              "the fixed step too.");
    }
    if (bc && a.enabled) {
        if (ImGui::SmallButton("apply to all tabs")) broadcast_adaptive(*bc, a);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Copy these step settings to every tab that has them\n"
                              "(Analysis, Custom, Bifurcation, LLE, LS, Basins, Metrics). Ctrl+Z undoes.");
    }
    ImGui::PopID();
    return changed;
}

// Поля Ньютона неявных схем: режим якобиана, допуск, предел итераций. Как "symmetry s":
// значения уходят в ядро рантайм-слотами a[M+2..M+4], модуль не пересобирается. id —
// суффикс ImGui-идентификаторов (несколько блоков в одном окне). Возвращает true при правке.
bool draw_newton_fields(NewtonSettings& n, float w, const char* id) {
    bool changed = false;
    ImGui::PushID(id);
    int jac_mode = n.full ? 1 : 0;
    ImGui::SetNextItemWidth(w * 2.0f);
    if (ImGui::Combo("Newton jacobian", &jac_mode, "modified (once per step)\0full (every iteration)\0")) {
        n.full = (jac_mode == 1);
        changed = true;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("modified: Jacobian and LU reused across iterations and\n"
                          "refreshed when the correction stops shrinking.\n"
                          "Cheaper on GPU and the right default.\n"
                          "full: rebuilt every iteration. Quadratic convergence,\n"
                          "most robust at large h, ~k times the cost.\n"
                          "Passed at run time as a[M+4]: no recompilation.");
    changed |= InputNumStr("Newton tolerance", n.tol, w);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Newton stops once ||dX|| falls below this (a[M+2], run time).\n"
                          "DAE methods also stop at ||dX|| < tol * |X| (relative).\n"
                          "With numb = float it also stops at the precision floor\n"
                          "||dX|| < 16 eps |X|: below that float cannot go.");
    changed |= InputNumStr("Newton max iterations", n.max_iters, w);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Hard cap on Newton passes per step (a[M+3], run time).\n"
                          "2-3 is typical from the explicit-Euler predictor.\n"
                          "A step that hits the cap is accepted unconverged: on a DAE\n"
                          "the constraint then drifts. DAE: full Newton, ~25 iterations.");
    ImGui::PopID();
    return changed;
}

// bc — модель для пункта "Apply to all calculation tabs" в ПКМ по полю;
// nullptr = меню не предлагать.
bool draw_integration_block(const char* header,
                                   const std::string& scheme,
                                   const std::vector<CustomScheme>& custom_schemes,
                                   const IntegrationFields& f,
                                   AppModel* bc,
                                   bool is_map) {
    if (!ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen)) return false;
    bool changed = false;
    auto menu = [&](BroadcastField kind, const char* label, const std::string& v) {
        return [bc, kind, label, &v] { field_apply_all_menu(bc, kind, {}, label, v); };
    };
    // h is pinned to 1 for a map (see default_h_from_record), so the field is
    // gone and the engine's "time" is exactly the iteration count.
    const char* t_max_label     = is_map ? "iterations"           : "computing time";
    const char* transient_label = is_map ? "transient iterations" : "transient time";
    if (f.h && !is_map) changed |= InputNumStr("h", *f.h, kFieldW,
                                    menu(BroadcastField::StepH, "h", *f.h));
    if (f.h && !is_map && f.adaptive && f.adaptive->enabled) {
        ImGui::SameLine();
        // LLE/LS (перенормировка в узлах шага) h не используют вовсе.
        ImGui::TextDisabled((f.adaptive_opts & kAdUiRenorm) ? "(not used)"
                            : f.adaptive->raw_nodes ? "(not used on step nodes)" : "(output step)");
    }
    if (f.symmetry_s && scheme_uses_symmetry(scheme, custom_schemes))
        changed |= InputNumStr("symmetry s", *f.symmetry_s, kFieldW,
                               menu(BroadcastField::SymmetryS, "symmetry s", *f.symmetry_s));
    if (f.cd_fp_iters && scheme_uses_cd_fp_iters(scheme)) {
        changed |= InputNumStr("FP iterations n", *f.cd_fp_iters, kFieldW,
                               menu(BroadcastField::CdFpIters, "FP iterations n", *f.cd_fp_iters));
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", kCdFpItersTip);
    }
    if (f.newton && scheme_uses_newton(scheme))
        changed |= draw_newton_fields(*f.newton, kFieldW);
    if (f.t_max)      changed |= InputNumStr(t_max_label, *f.t_max,       kFieldW,
                               menu(BroadcastField::TMax, t_max_label, *f.t_max));
    if (f.transient)   changed |= InputNumStr(transient_label, *f.transient,   kFieldW,
                               menu(BroadcastField::Transient, transient_label, *f.transient));
    if (f.pre_scaller) changed |= InputNumStr("decimator",      *f.pre_scaller, kFieldW,
                               menu(BroadcastField::PreScaller, "decimator", *f.pre_scaller));
    if (f.max_value)   changed |= InputNumStr("max value",      *f.max_value,   kFieldW,
                               menu(BroadcastField::MaxValue, "max value", *f.max_value));
    if (f.adaptive) {
        ImGui::Separator();
        changed |= draw_adaptive_block(header, *f.adaptive, scheme, f.adaptive_opts, is_map, bc);
    }
    return changed;
}

// Секции "Initial conditions" и "Parameters" — это один и тот же блок:
// по числовому полю на каждое имя из списка. Раньше — около дюжины копий.
// bc / field — см. draw_integration_block: ПКМ по полю предлагает разослать
// значение по вкладкам. field говорит, чем этот блок является — параметрами
// или начальными условиями; по заголовку это не определить (он ещё и с ##id).
void draw_named_num_fields(const char* header,
                                  const std::vector<std::string>& names,
                                  std::map<std::string, std::string>& values,
                                  const char* note,
                                  AppModel* bc,
                                  BroadcastField field,
                                  // Поля, которые расчёт ПЕРЕЗАПИШЕТ сам (область
                                  // устойчивости строит четыре элемента матрицы в
                                  // каждом узле сетки). Такие гасятся, а не молча
                                  // игнорируются: живое поле, ни на что не влияющее,
                                  // читается как настройка, которой оно не является.
                                  const std::function<bool(const std::string&)>& overwritten) {
    if (!ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen)) return;
    if (note) ImGui::TextDisabled("%s", note);
    // Группы параметров — у системы, загруженной в модель (bc); у НУ и без модели — плоский список.
    const ParamGroups* groups = field == BroadcastField::Param && bc ? &bc->param_groups : nullptr;
    draw_param_tree(groups, names, [&](const std::string& n) {
        ImGui::PushID(n.c_str());
        const bool off = overwritten && overwritten(n);
        if (off) ImGui::BeginDisabled();
        InputNumStr(n.c_str(), values[n], kFieldW,
                    [&] { field_apply_all_menu(bc, field, n, n, values[n]); });
        if (off) ImGui::EndDisabled();
        ImGui::PopID();
    });
}

// Дерево групп параметров (param_groups.h). Узлы строятся каждый кадр — групп десятки, это
// дешевле, чем следить за кэшем. Открыт ли узел, помнит ImGui (по id окна и пути группы).
namespace {
struct PgNode {
    std::string seg, path;
    const ParamGroup* g = nullptr;
    std::vector<PgNode> kids;
    int count = 0;   // параметров из names в узле и под ним
};

int pg_count(PgNode& n, const std::set<std::string>& present) {
    n.count = 0;
    if (n.g) for (const std::string& p : n.g->params) n.count += present.count(p) ? 1 : 0;
    for (PgNode& k : n.kids) n.count += pg_count(k, present);
    return n.count;
}

bool pg_header(const std::string& label, const std::string& id, bool in_table) {
    if (in_table) { ImGui::TableNextRow(); ImGui::TableSetColumnIndex(0); }
    const ImGuiTreeNodeFlags fl = in_table ? ImGuiTreeNodeFlags_SpanAllColumns | ImGuiTreeNodeFlags_LabelSpanAllColumns
                                           : ImGuiTreeNodeFlags_SpanAvailWidth;
    return ImGui::TreeNodeEx((label + "##pg_" + id).c_str(), fl);
}

void pg_draw(const PgNode& n, const std::set<std::string>& present,
             const std::function<void(const std::string&)>& item, bool in_table) {
    if (n.g)
        for (const std::string& p : n.g->params)
            if (present.count(p)) item(p);
    for (const PgNode& k : n.kids) {
        if (k.count == 0) continue;
        if (pg_header(k.seg + "  (" + std::to_string(k.count) + ")", k.path, in_table)) {
            pg_draw(k, present, item, in_table);
            ImGui::TreePop();
        }
    }
}
}  // namespace

void draw_param_tree(const ParamGroups* groups, const std::vector<std::string>& names,
                     const std::function<void(const std::string&)>& item, bool in_table) {
    if (groups == nullptr || groups->empty()) {
        for (const std::string& n : names) item(n);
        return;
    }
    const std::set<std::string> present(names.begin(), names.end());
    PgNode root;
    std::set<std::string> grouped;
    for (const ParamGroup& g : *groups) {
        PgNode* cur = &root;
        size_t pos = 0;
        while (pos <= g.path.size()) {
            size_t e = g.path.find('/', pos);
            if (e == std::string::npos) e = g.path.size();
            const std::string seg = g.path.substr(pos, e - pos);
            pos = e + 1;
            auto it = std::find_if(cur->kids.begin(), cur->kids.end(), [&](const PgNode& k) { return k.seg == seg; });
            if (it == cur->kids.end()) {
                PgNode k;
                k.seg = seg;
                k.path = g.path.substr(0, e);
                cur->kids.push_back(std::move(k));
                it = cur->kids.end() - 1;
            }
            cur = &*it;
        }
        cur->g = &g;
        grouped.insert(g.params.begin(), g.params.end());
    }
    if (pg_count(root, present) == 0) {   // ни одна группа к этим именам не относится
        for (const std::string& n : names) item(n);
        return;
    }
    pg_draw(root, present, item, in_table);
    std::vector<std::string> other;
    for (const std::string& n : names) if (!grouped.count(n)) other.push_back(n);
    if (!other.empty() && pg_header("Other  (" + std::to_string(other.size()) + ")", "\x01other", in_table)) {
        for (const std::string& n : other) item(n);
        ImGui::TreePop();
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
                           int lines) {
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

// Цель, живущая в объекте с постоянным адресом (Shared config вкладки Custom —
// член CustomSession, а тот член AppModel). Вектора нет, переезжать нечему,
// поэтому relocate отдаёт те же указатели.
[[nodiscard]] ViewRangeTarget vr_target_fixed(std::string name,
        std::string* x_lo, std::string* x_hi,
        std::string* y_lo, std::string* y_hi) {
    ViewRangeTarget t;
    t.name = std::move(name);
    t.x_lo = x_lo; t.x_hi = x_hi; t.y_lo = y_lo; t.y_hi = y_hi;
    t.relocate = [x_lo, x_hi, y_lo, y_hi]() {
        ViewRangeSlots s;
        s.x_lo = x_lo; s.x_hi = x_hi; s.y_lo = y_lo; s.y_hi = y_hi;
        return s;
    };
    return t;
}

// Границы вида — произвольные числа после зума. fmt_num_shortest насыпал бы в
// поле семнадцать цифр, а фиксированный %.6g схлопнул бы узкое окно
// (1.00000001 .. 1.00000002) в одно и то же число. Берём наименьшую точность,
// при которой запись расходится со значением меньше чем на миллионную долю
// ширины окна; при нулевой ширине (вырожденный вид) — round-trip как есть.
[[nodiscard]] static std::string fmt_view_bound(double v, double span) {
    const double tol = std::abs(span) * 1e-6;
    if (tol > 0.0) {
        char buf[64];
        for (int prec = 6; prec <= 17; ++prec) {
            std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
            if (std::abs(parse_ratio_or(buf, v) - v) <= tol) return std::string(buf);
        }
    }
    return fmt_num_shortest(v);
}

// Все диаграммы приложения, у которых есть диапазоны свипа. Нужен Ctrl+Shift+T
// (взять все цели того же типа) и списку галочек в меню диаграммы.
// Порядок — как во вкладках; на нём же строится маска выбора.
[[nodiscard]] static std::vector<ViewRangeTarget> all_view_range_targets(AppModel& model) {
    std::vector<ViewRangeTarget> out;
    auto add = [&out](ViewRangeTarget t, const char* kind, const std::string& group) {
        if (!t.relocate || (!t.x_lo && !t.y_lo)) return;   // цель без полей не нужна
        t.kind  = kind;
        t.group = group;
        out.push_back(std::move(t));
    };

    using B = BifurcationDiagramConfig;
    auto& bd = model.bifurcation_session.diagrams;
    for (int i = 0; i < (int)bd.size(); ++i) {
        if (bd[(size_t)i].mode_2d)
            add(vr_target(bd, i, &B::param_lo_text, &B::param_hi_text,
                          &B::param_lo_2_text, &B::param_hi_2_text),
                "Parametric 2D", "Bifurcation 2D");
        else if (bd[(size_t)i].colored_1d)
            // Colored 1D: по Y окно гистограммы, и запись туда включает
            // ручной режим — ровно как на самой диаграмме.
            add(vr_target(bd, i, &B::param_lo_text, &B::param_hi_text,
                          &B::colored_1d_ymin_text, &B::colored_1d_ymax_text,
                          &B::colored_1d_custom_y),
                "Parametric 1D", "Bifurcation colored 1D");
        else
            add(vr_target(bd, i, &B::param_lo_text, &B::param_hi_text),
                "Parametric 1D", "Bifurcation 1D");
    }

    using L = LLECurveConfig;
    auto& lle = model.lle_session.curves;
    for (int i = 0; i < (int)lle.size(); ++i) {
        if (lle[(size_t)i].mode_2d)
            add(vr_target(lle, i, &L::param_lo_text, &L::param_hi_text,
                          &L::param_lo_2_text, &L::param_hi_2_text),
                "Parametric 2D", "LLE 2D");
        else
            add(vr_target(lle, i, &L::param_lo_text, &L::param_hi_text),
                "Parametric 1D", "LLE 1D");
    }

    using S = LSCurveConfig;
    auto& ls = model.ls_session.curves;
    for (int i = 0; i < (int)ls.size(); ++i) {
        if (ls[(size_t)i].mode_2d)
            add(vr_target(ls, i, &S::param_lo_text, &S::param_hi_text,
                          &S::param_lo_2_text, &S::param_hi_2_text),
                "Parametric 2D", "LS 2D");
        else
            add(vr_target(ls, i, &S::param_lo_text, &S::param_hi_text),
                "Parametric 1D", "LS 1D");
    }

    using M = SignalMetricsConfig;
    auto& met = model.metrics_session.configs;
    for (int i = 0; i < (int)met.size(); ++i) {
        if (met[(size_t)i].mode_2d)
            add(vr_target(met, i, &M::param_lo_text, &M::param_hi_text,
                          &M::param_lo_2_text, &M::param_hi_2_text),
                "Parametric 2D", "Metrics 2D");
        else
            add(vr_target(met, i, &M::param_lo_text, &M::param_hi_text),
                "Parametric 1D", "Metrics 1D");
    }

    // DFT — свой тип: по X свип параметра, а по Y частоты, и подставлять туда
    // диапазон параметра со второй оси 2D-карты нельзя.
    using D = Dft1DConfig;
    auto& dft = model.dft1d_session.configs;
    for (int i = 0; i < (int)dft.size(); ++i)
        add(vr_target(dft, i, &D::param_lo_text, &D::param_hi_text,
                      &D::freq_lo_text, &D::freq_hi_text),
            "1D DFT", "1D DFT");

    // Basins и FastSync — сетка начальных условий, а не параметрический свип.
    using BA = BasinsConfig;
    auto& bas = model.basins_session.configs;
    for (int i = 0; i < (int)bas.size(); ++i)
        add(vr_target(bas, i, &BA::axis_x_lo_text, &BA::axis_x_hi_text,
                      &BA::axis_y_lo_text, &BA::axis_y_hi_text),
            "IC grid", "Basins");

    using F = FastSyncConfig;
    auto& fs = model.fastsync_session.configs;
    for (int i = 0; i < (int)fs.size(); ++i)
        add(vr_target(fs, i, &F::axis_x_lo_text, &F::axis_x_hi_text,
                      &F::axis_y_lo_text, &F::axis_y_hi_text),
            "IC grid", "Fast Synchro");

    using O = OrderConfig;
    auto& ord = model.order_session.configs;
    for (int i = 0; i < (int)ord.size(); ++i) {
        if (ord[(size_t)i].two_d)
            add(vr_target(ord, i, &O::axis_x_lo_text, &O::axis_x_hi_text,
                          &O::axis_y_lo_text, &O::axis_y_hi_text),
                "Order 2D", "Order 2D");
        else
            add(vr_target(ord, i, &O::axis_x_lo_text, &O::axis_x_hi_text),
                "Order 1D", "Order 1D");
    }

    add(vr_target_fixed("Shared config",
                        &model.custom_session.shared.axis_x_lo_text,
                        &model.custom_session.shared.axis_x_hi_text,
                        &model.custom_session.shared.axis_y_lo_text,
                        &model.custom_session.shared.axis_y_hi_text),
        "Custom", "Custom");
    return out;
}

// Цели того же типа, что и source. Источник опознаём по АДРЕСУ поля: цель,
// собранная на месте отрисовки, и цель из общего перечисления указывают на
// один и тот же конфиг, а сравнивать больше не по чему.
// Источник в списке идёт первым — Ctrl+T и Ctrl+Shift+T тогда пишут в него одно
// и то же, а не в разном порядке.
[[nodiscard]] static std::vector<ViewRangeTarget>
view_range_same_kind(AppModel& model, const ViewRangeTarget& source) {
    std::vector<ViewRangeTarget> all = all_view_range_targets(model);
    std::string kind;
    for (const auto& t : all)
        if (t.x_lo && t.x_lo == source.x_lo) { kind = t.kind; break; }
    std::vector<ViewRangeTarget> out;
    if (kind.empty()) return out;
    for (auto& t : all)
        if (t.kind == kind) {
            if (t.x_lo == source.x_lo) out.insert(out.begin(), std::move(t));
            else                       out.push_back(std::move(t));
        }
    return out;
}

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
[[nodiscard]] static ViewRangeValues values_from_axes(const AxisInfo& x, const AxisInfo& y,
                                                      bool swapped, bool src_has_y) {
    const AxisInfo& ax = swapped ? y : x;   // ось ДАННЫХ X
    const AxisInfo& ay = swapped ? x : y;   // ось ДАННЫХ Y
    // invert — только про отрисовку, view_min/view_max остаются упорядоченными;
    // min/max здесь страховка от конфига, пришедшего из чужой руки.
    ViewRangeValues v;
    // ±log-ось (бассейны): вид живёт в координатах узлов, а значения по ним идут немонотонно
    // (-hi..-lo, +lo..+hi) — диапазоном [lo, hi] его не записать, такую ось не трогаем.
    if (!ax.symlog) {
        const double span = ax.view_max - ax.view_min;
        v.has_x = true;
        v.x_lo = fmt_view_bound((std::min)(ax.view_min, ax.view_max), span);
        v.x_hi = fmt_view_bound((std::max)(ax.view_min, ax.view_max), span);
    }
    if (src_has_y && !ay.symlog) {
        const double span = ay.view_max - ay.view_min;
        v.has_y = true;
        v.y_lo = fmt_view_bound((std::min)(ay.view_min, ay.view_max), span);
        v.y_hi = fmt_view_bound((std::max)(ay.view_min, ay.view_max), span);
    }
    return v;
}

// Значения ИЗ ПОЛЕЙ цели. Нужно после Ctrl+Z: отмена вернула поля, а картинка
// осталась зумленной, и рассылка "из вида" затёрла бы только что
// восстановленное тем, что осталось на экране.
[[nodiscard]] static ViewRangeValues values_from_target(const ViewRangeTarget& t) {
    ViewRangeValues v;
    v.source = "fields";
    if (t.x_lo && t.x_hi) { v.has_x = true; v.x_lo = *t.x_lo; v.x_hi = *t.x_hi; }
    if (t.y_lo && t.y_hi) { v.has_y = true; v.y_lo = *t.y_lo; v.y_hi = *t.y_hi; }
    return v;
}

// Пишет значения в одну цель. Ось, которой у цели нет, не трогается — так 2D
// диапазон можно разложить и по 1D-диаграммам, не выдумывая им вторую ось.
static void apply_view_values(const ViewRangeTarget& t, const ViewRangeValues& v) {
    if (v.has_x && t.x_lo && t.x_hi) { *t.x_lo = v.x_lo; *t.x_hi = v.x_hi; }
    if (v.has_y && t.y_lo && t.y_hi) {
        *t.y_lo = v.y_lo; *t.y_hi = v.y_hi;
        if (t.y_manual) *t.y_manual = true;
    }
}

static void run_view_range_cmd(const std::vector<ViewRangeTarget>& targets, size_t count,
                               const ViewRangeValues& vals, AppModel* model) {
    if (targets.empty()) return;
    count = (std::min)(count, targets.size());

    // Снимок полей цели. Держим не адреса, а способ найти их заново.
    struct Snap {
        std::function<ViewRangeSlots()> relocate;
        bool has_x = false, has_y = false, has_flag = false;
        std::string x_lo, x_hi, y_lo, y_hi;
        bool flag = false;
    };
    auto capture = [](const ViewRangeTarget& t) {
        Snap p;
        p.relocate = t.relocate;
        if (t.x_lo && t.x_hi) { p.has_x = true; p.x_lo = *t.x_lo; p.x_hi = *t.x_hi; }
        if (t.y_lo && t.y_hi) { p.has_y = true; p.y_lo = *t.y_lo; p.y_hi = *t.y_hi; }
        if (t.y_manual)       { p.has_flag = true; p.flag = *t.y_manual; }
        return p;
    };
    // Отмена и повтор — одна процедура, отличаются только снимком.
    auto writer = [](std::vector<Snap> snaps) {
        return [snaps = std::move(snaps)]() {
            for (const auto& p : snaps) {
                const ViewRangeSlots s = p.relocate();
                if (p.has_x && s.x_lo && s.x_hi) { *s.x_lo = p.x_lo; *s.x_hi = p.x_hi; }
                if (p.has_y && s.y_lo && s.y_hi) { *s.y_lo = p.y_lo; *s.y_hi = p.y_hi; }
                if (p.has_flag && s.y_manual)    *s.y_manual = p.flag;
            }
        };
    };

    std::vector<Snap> before, after;
    for (size_t i = 0; i < count; ++i) {
        const ViewRangeTarget& t = targets[i];
        Snap p = capture(t);
        // Пишем ТОЛЬКО то, что умеем вернуть: цель без relocate откатить нечем,
        // и запись в неё была бы дырой в отмене — поля изменились, а Ctrl+Z про
        // них не знает. Сегодня такие цели не появляются (обе фабрики relocate
        // ставят), но инвариант лучше держать кодом, чем соглашением.
        if (!p.relocate || (!p.has_x && !p.has_y)) continue;
        before.push_back(std::move(p));
        apply_view_values(t, vals);
    }
    if (!model || before.empty()) return;
    // "После" снимаем по тем же целям и уже после записи — так повтор кладёт
    // ровно то, что команда посчитала, без второго прохода по осям.
    for (size_t i = 0; i < count; ++i) {
        Snap p = capture(targets[i]);
        if (p.relocate && (p.has_x || p.has_y)) after.push_back(std::move(p));
    }

    std::string what = std::string("calculation range from ") + vals.source;
    if (count > 1)                     what += " (" + std::to_string((int)count) + " diagrams)";
    else if (!targets[0].name.empty()) what += " (" + targets[0].name + ")";
    model->push_undo(what, writer(std::move(before)), writer(std::move(after)));
    // Отчёт в шапку — как у рассылки полей. Команда правит и диаграммы в других
    // окнах, которые сейчас не видны, и без строки это неотличимо от "ничего не
    // произошло".
    model->undo_note = "Applied: " + what;
    model->undo_note_time = ImGui::GetTime();
}

[[nodiscard]] static bool view_range_bound(const std::vector<ViewRangeTarget>& targets) {
    return !targets.empty() && (targets[0].x_lo != nullptr || targets[0].y_lo != nullptr);
}

// Есть ли у вида-источника вход расчёта по Y. Отдельного признака заводить не
// нужно: ровно это и кодирует ненулевой y_lo его собственной цели (см.
// all_view_range_targets — 1D-целям передаются только поля X).
[[nodiscard]] static bool src_y_is_input(const std::vector<ViewRangeTarget>& targets) {
    return !targets.empty() && targets[0].y_lo != nullptr;
}

// Пункты правого клика. Зовётся из popup_extras, т.е. уже внутри BeginPopup.
// swapped и model — БЕЗ значений по умолчанию, и это принципиально. Пока они были
// умолчательными, вызов с четырьмя аргументами компилировался молча: AppModel* приводился
// к bool в параметр swapped (указатель всегда ненулевой, то есть swapped = true), а model
// оставался nullptr. У всех 1D-графиков это давало сразу три поломки — диапазон свипа брался
// с оси Y вместо X, запись не попадала в стек отмены и не отчитывалась в шапке, а
// Ctrl+Shift+T сваливался на одну цель. Теперь недостача аргумента — ошибка компиляции.
void draw_view_range_menu(const std::vector<ViewRangeTarget>& targets,
                                 const AxisInfo& x, const AxisInfo& y,
                                 bool swapped, AppModel* model) {
    if (!view_range_bound(targets)) { draw_undo_menu_item(model); return; }
    ImGui::Separator();
    std::string one = "Calculation range from view";
    if (targets.size() > 1) one += " (" + targets[0].name + ")";
    if (ImGui::MenuItem(one.c_str(), "Ctrl+T"))
        run_view_range_cmd(targets, 1,
                           values_from_axes(x, y, swapped, src_y_is_input(targets)), model);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Writes the visible axis range into this diagram's sweep\n"
                          "range fields. Press Run to recompute the zoomed region\n"
                          "at full resolution.");
    // Ctrl+Shift+T — все диаграммы ТОГО ЖЕ типа, а не только наложенные в этом
    // окне: приблизив 2D-бифуркацию, обычно хотят увидеть тот же участок и на
    // 2D LLE с 2D LS. Набор считается по общему перечислению (см.
    // view_range_same_kind), поэтому в него попадают и другие окна.
    if (model) {
        std::vector<ViewRangeTarget> kin = view_range_same_kind(*model, targets[0]);
        if (kin.size() > 1) {
            // Сразу после Ctrl+Z / Ctrl+Shift+Z источником становятся ПОЛЯ, а
            // не вид: поля только что откатили, картинка осталась зумленной, и
            // разослать "из вида" значило бы отменить отмену. Пишем это прямо
            // в пункте меню — угадывать источник пользователь не должен.
            const bool from_fields = model->undo_just_happened();
            char lbl[160];
            std::snprintf(lbl, sizeof(lbl), "Calculation range from %s - all %s (%d)",
                          from_fields ? "fields" : "view",
                          kin[0].kind.c_str(), (int)kin.size());
            if (ImGui::MenuItem(lbl, "Ctrl+Shift+T")) {
                const ViewRangeValues v = from_fields
                        ? values_from_target(kin[0])
                        : values_from_axes(x, y, swapped, src_y_is_input(targets));
                run_view_range_cmd(kin, kin.size(), v, model);
            }
            if (from_fields && ImGui::IsItemHovered())
                ImGui::SetTooltip("Undo just restored the sweep fields, so this sends\n"
                                  "THOSE values on. The plot is still zoomed; sending the\n"
                                  "view instead would undo the undo.");
        }

        // Точный выбор галочками — для случаев, когда "тот же тип" не тот набор:
        // например, подтянуть к 2D-карте ещё и Basins, у которых оси другого
        // смысла и в автоматический набор они не входят.
        std::vector<ViewRangeTarget> all = all_view_range_targets(*model);
        if (!all.empty() && ImGui::BeginMenu("Calculation range to...")) {
            std::string sig;
            for (const auto& t : all) { sig += '\x1f'; sig += t.group; sig += t.name; }
            if (model->vr_pick_sig != sig || model->vr_pick.size() != all.size()) {
                model->vr_pick.assign(all.size(), '0');
                model->vr_pick_sig = sig;
                // По умолчанию отмечаем свой тип: это и подсказка, что здесь
                // выбирают, и самый частый набор.
                for (size_t i = 0; i < all.size(); ++i)
                    for (const auto& k : kin)
                        if (all[i].x_lo && all[i].x_lo == k.x_lo) model->vr_pick[i] = '1';
            }
            if (ImGui::SmallButton("All"))  model->vr_pick.assign(all.size(), '1');
            ImGui::SameLine();
            if (ImGui::SmallButton("None")) model->vr_pick.assign(all.size(), '0');

            std::vector<ViewRangeTarget> sel;
            for (size_t i = 0; i < all.size(); ++i)
                if (model->vr_pick[i] == '1') sel.push_back(all[i]);

            char btn[96];
            std::snprintf(btn, sizeof(btn), "Apply to %d selected", (int)sel.size());
            if (sel.empty()) ImGui::BeginDisabled();
            if (ImGui::Button(btn)) {
                const ViewRangeValues v = model->undo_just_happened()
                        ? values_from_target(targets[0])
                        : values_from_axes(x, y, swapped, src_y_is_input(targets));
                run_view_range_cmd(sel, sel.size(), v, model);
                ImGui::CloseCurrentPopup();
            }
            if (sel.empty()) ImGui::EndDisabled();
            ImGui::Separator();

            std::string shown_group;
            for (size_t i = 0; i < all.size(); ++i) {
                if (all[i].group != shown_group) {
                    shown_group = all[i].group;
                    ImGui::SeparatorText(shown_group.c_str());
                }
                ImGui::PushID((int)i);
                bool on = (model->vr_pick[i] == '1');
                if (ImGui::Checkbox("##pick", &on))
                    model->vr_pick[i] = on ? '1' : '0';
                ImGui::SameLine();
                ImGui::TextUnformatted(all[i].name.c_str());
                ImGui::PopID();
            }
            ImGui::EndMenu();
        }
    }
    draw_undo_menu_item(model);
}

// Горячие клавиши того же действия. active — окно диаграммы под курсором:
// плотов на экране бывает несколько, и глобальный хоткей (как Ctrl+R у Run)
// здесь означал бы "перепиши диапазон неизвестно чему".
void handle_view_range_keys(const std::vector<ViewRangeTarget>& targets,
                                   const AxisInfo& x, const AxisInfo& y, bool active,
                                   bool swapped, AppModel* model) {   // без умолчаний — см. draw_view_range_menu
    if (!active || !view_range_bound(targets)) return;
    const ImGuiIO& io = ImGui::GetIO();
    // С Alt не реагируем вовсе: Ctrl+Alt+T — не наше сочетание, и тихо
    // трактовать его как Ctrl+T значит переписать диапазон по случайному нажатию.
    if (!io.KeyCtrl || io.KeyAlt || io.WantTextInput) return;
    if (!ImGui::IsKeyPressed(ImGuiKey_T, false)) return;
    if (io.KeyShift && model) {
        // Ctrl+Shift+T — весь тип целиком (см. draw_view_range_menu). Если
        // источник в общем перечислении не нашёлся, падаем на прежнее
        // поведение — наложенные в этом окне диаграммы.
        std::vector<ViewRangeTarget> kin = view_range_same_kind(*model, targets[0]);
        const std::vector<ViewRangeTarget>& dst = kin.empty() ? targets : kin;
        // Сразу после отката источник — поля, а не вид (см. меню).
        const ViewRangeValues v = model->undo_just_happened()
                                ? values_from_target(dst[0])
                                : values_from_axes(x, y, swapped, src_y_is_input(targets));
        run_view_range_cmd(dst, dst.size(), v, model);
        return;
    }
    run_view_range_cmd(targets, 1,
                       values_from_axes(x, y, swapped, src_y_is_input(targets)), model);
}

// Окно диаграммы под курсором — общее условие хоткея выше. Берём именно
// hover, а не focus: фокус остаётся на панели настроек, откуда зумили колесом.
[[nodiscard]] bool plot_window_active() {
    return ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows |
                                  ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
}

// Inline-переименование активной вкладки. Раньше — два ручных char buf[128],
// которые молча обрезали имя на 127 символах.
void draw_label_rename(const char* label_with_id, std::string& label,
                              float width) {
    ImGui::SetNextItemWidth(width);
    InputTextStr(label_with_id, label);
}

// Выбранные вкладки узлов дока при смене режима (Analysis, Parametric, ...). Окна чужого
// режима не выводятся, и ImGui вынимает их из узлов дока; при возвращении окно, появившееся
// последним, получает фокус, и его узел выбирает именно его, а не то, что было выбрано.
// Пока режим на экране, запоминаем выбранное окно каждого узла; первые кадры после входа
// просим узлы выбрать его снова (NextSelectedTabId: узел сам переведёт на окно и фокус).
// Кадров несколько: в первом узел ещё пересобирает таб-бар и выбирает по-своему.
// Зовётся после вывода всех окон кадра.
void track_dock_selection(AppModel& model) {
    const int mode = (int)model.app_mode;
    std::vector<std::string>& mem = model.dock_selected_by_mode[mode];
    if (model.dock_prev_mode != mode) {
        model.dock_prev_mode      = mode;
        model.dock_restore_frames = 3;
    }
    if (model.dock_restore_frames > 0) {
        --model.dock_restore_frames;
        for (const std::string& name : mem) {
            ImGuiWindow* w = ImGui::FindWindowByName(name.c_str());
            if (w == nullptr || !w->Active || w->DockNode == nullptr || w->DockNode->TabBar == nullptr) continue;
            ImGuiTabBar* tb = w->DockNode->TabBar;
            if (tb->SelectedTabId != w->TabId && ImGui::TabBarFindTabByID(tb, w->TabId) != nullptr)
                tb->NextSelectedTabId = w->TabId;
        }
        return;
    }
    mem.clear();
    for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows)
        if (w->Active && w->DockIsActive && w->DockNode != nullptr && w->DockNode->TabBar != nullptr
            && w->DockNode->TabBar->SelectedTabId == w->TabId)
            mem.push_back(w->Name);
}

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
                                        int request_select) {
    TabBarResult r;
    // Выбор после закрытия вкладки (между кадрами — в хранилище ImGui, ключ от bar_id).
    // Под номер закрытой вкладки встаёт следующий конфиг, ImGui видит его как новую вкладку
    // и по AutoSelectNewTabs выбрал бы его, даже если закрывали не активную.
    ImGuiStorage* st = ImGui::GetStateStorage();
    const ImGuiID pending_key = ImGui::GetID(bar_id) ^ 0x5e1ec7u;
    const int pending = st->GetInt(pending_key, -1);
    if (pending >= 0) {
        st->SetInt(pending_key, -1);
        if (request_select < 0 && pending < n) request_select = pending;
    }
    if (!ImGui::BeginTabBar(bar_id, ImGuiTabBarFlags_AutoSelectNewTabs |
                                    ImGuiTabBarFlags_FittingPolicyScroll))
        return r;
    for (int i = 0; i < n; ++i) {
        ImGui::PushID(i);
        bool open = true;
        // ID завязан на индекс, а видимая часть — на label: пользователь видит
        // свежее имя сразу после редактирования.
        const std::string tab_id = label(i) + "###" + item_id_prefix + std::to_string(i);
        // Запрещаем закрывать вкладку, чей расчёт сейчас идёт, и ЕДИНСТВЕННУЮ
        // вкладку. Пустой список ни одна сессия не переживает: загрузчики при
        // следующем входе (перезапуск, смена системы) кладут дефолтный конфиг,
        // "+" на пустом списке — тоже дефолт, и настройки последней вкладки
        // терялись. Так же давно устроен NetworkSession::remove_config.
        const bool can_close = !(in_flight && running_index == i) && n > 1;
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
    if (r.to_remove >= 0) {
        // Активная остаётся активной (её номер сдвигается, если она правее закрытой);
        // закрыли саму активную — выбрать ту, что встала на её место (или последнюю).
        const int a = r.active, i = r.to_remove;
        const int next = (a > i) ? a - 1 : (a == i || a < 0) ? std::min(i, n - 2) : a;
        st->SetInt(pending_key, next);
    }
    return r;
}

// Строка "Run (Ctrl+R) [| доп. кнопка] | Run all... (Ctrl+Shift+R) | (N queued)" вместе с
// попапом выбора; раньше — четыре копии по ~60 строк.
// Отметки хранятся ПО popup_id (единый static сливал бы выбор между вкладками) и живут
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
    case ParamPlotKind::Metrics:
        // Линия, как у LLE, но без нулевой линии: у амплитуд и частот ноль
        // ничего не разделяет. Имя Y ставит draw_metrics_plot по выбранной метрике.
        view.points_mode       = false;
        view.line_thickness_px = 1.5f;
        view.imdraw_lines      = true;
        view.show_zero_y       = false;
        view.x_axis.name       = "parameter";
        view.y_axis.name       = "metric";
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

// Стиль линейного 2D-графика проекции Analysis (Phase 2D, Time domain, Step size, Rejections):
// Custom line style — толщина и α (толстая линия), Points — точки вместо линии; обе галки —
// точки поверх линии (Plot2DView::points_with_lines). Настраивает pr.view2d; α к цвету линейных
// серий применяет вызывающий (по pr.custom_line_style). Серии-маркеры (points_override = 1)
// стиль не трогает.
void draw_proj_line_style(Projection& pr, const char* sfx, const char* points_tip) {
    const std::string s = sfx;
    draw_style_toolbar("Custom line style", sfx, pr.custom_line_style,
        [&pr, &s]() {
            bool ch = false;
            ImGui::SetNextItemWidth(150);
            ch |= ImGui::SliderFloat(("Line width##" + s).c_str(), &pr.line_width, 0.1f, 5.0f, "%.2f");
            ImGui::SetNextItemWidth(150);
            ch |= ImGui::SliderFloat(("Alpha##" + s).c_str(), &pr.alpha, 0.0f, 1.0f, "%.2f");
            return ch;
        });
    ImGui::SameLine();
    ImGui::Checkbox(("Points##" + s).c_str(), &pr.draw_points);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s\nWith Custom line style on too: points on top of the line.", points_tip);
    if (pr.draw_points) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(110);
        ImGui::SliderFloat(("size##" + s + "pt").c_str(), &pr.point_size, 0.5f, 8.0f, "%.1f");
        if (pr.custom_line_style) {   // своя альфа точек — при Custom line style (у линии своя)
            ImGui::SameLine();
            ImGui::SetNextItemWidth(110);
            ImGui::SliderFloat(("alpha##" + s + "pt").c_str(), &pr.point_alpha, 0.0f, 1.0f, "%.2f");
        }
    }
    pr.view2d->imdraw_lines      = pr.custom_line_style;
    pr.view2d->line_thickness_px = pr.line_width;
    pr.view2d->points_mode       = pr.draw_points;
    pr.view2d->points_with_lines = pr.draw_points && pr.custom_line_style;
    pr.view2d->point_size_px     = pr.point_size;
    pr.view2d->point_alpha       = (pr.draw_points && pr.custom_line_style) ? pr.point_alpha : -1.0f;
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
    // Вырожденный диапазон — это не сетка: снап по нему сажал бы курсор на
    // узлы, которых в данных нет. Проверка здесь, в единственной точке входа.
    if (n > 1 && hi != lo) {
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
//   - fit и кламп X — по данным, не по полям диапазона: правка диапазона до Run
//     не двигает вид, новый прогон подгоняет его через fit_request;
//   - log_scale только если он одинаков у всех членов.
void configure_sweep_x_axis(Plot2DView& view,
                                   const std::vector<SweepAxisMember>& members,
                                   const std::vector<std::string>& params,
                                   const std::vector<std::string>& vars) {
    int  shared_kind = -2;   // -2 = ещё не видели, -1 = смешанные
    int  shared_idx  = -2;
    bool shared_log = false, log_seen = false, log_mismatch = false;

    for (const auto& m : members) {
        const int kidx = (m.kind == 2) ? -1 : m.index;
        if (shared_kind == -2) { shared_kind = m.kind; shared_idx = kidx; }
        else if (shared_kind != m.kind || shared_idx != kidx) { shared_kind = -1; shared_idx = -1; }

        if (!log_seen) { shared_log = m.log_scale; log_seen = true; }
        else if (shared_log != m.log_scale) { log_mismatch = true; }
    }

    view.x_fit_use_explicit = false;
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

// Snap-диапазон по ПЕРВОМУ члену окна: берём сетку успешного прогона, иначе
// парсим текстовые поля конфига (до первого Run их всё равно больше нечем
// заполнить). Раньше — три идентичных блока в Bif/LLE/LS.
void apply_snap_x_from_config(Plot2DView& view,
                                     bool have_result, double res_lo, double res_hi, int res_n,
                                     const std::string& lo_text,
                                     const std::string& hi_text,
                                     const std::string& n_text) {
    if (have_result && res_n > 1 && res_hi != res_lo) {
        apply_snap_x(view, res_lo, res_hi, res_n);
        return;
    }
    // Без прогона сетку берём из полей конфига — но ТОЛЬКО если они реально
    // прочитались. Раньше здесь стояли дефолты parse_ratio_or(lo, 0.0) и
    // (hi, 1.0): на непрочитанных полях снап включался с чужой сеткой [0, 1],
    // и курсор садился на её узлы (k/(n-1)) вместо значений свипа.
    const int n = parse_int_or(n_text, 0);
    double lo = 0.0, hi = 0.0;
    if (n > 1 && parse_num_ok(lo_text, lo) && parse_num_ok(hi_text, hi) && lo != hi)
        apply_snap_x(view, lo, hi, n);
    else
        apply_snap_x(view, 0.0, 1.0, 0);   // n = 0 -> снап выключен
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
        int cfg_exponent_idx) {
    auto& slot = map[idx];
    if (!slot) {
        slot = std::make_unique<HeatmapView>();
        slot->colormap = (HeatmapColormap)colormap_id_or(cfg_colormap, app_default_colormap);
        if (cfg_exponent_idx != kNoExponent) slot->display_exponent_idx = cfg_exponent_idx;
    }
    return *slot;
}

// Static-вью вкладок переживают пересоздание сессии (смена системы), а data_generation новых
// конфигов снова считается с 0: совпавший номер оставлял на экране картинку прошлой системы.
// Сбрасываем только то, что решает перезаливку и автофит (как в Custom); палитра и шкала остаются.
void invalidate_view(HeatmapView& hv) { hv.invalidate(); }
void invalidate_view(Plot2DView& v)   { v.view_valid = false;  v.series_generation = -1; }
void invalidate_view(Plot3DView& v)   { v.view_valid = false;  v.series_generation = -1; }
// true один раз после каждого start_*_analysis вкладки `mode`; `seen` — static вызывающего.
bool session_reloaded(const AppModel& model, AppModel::AppMode mode, int& seen) {
    const int epoch = model.session_epoch[(int)mode];
    if (seen == epoch) return false;
    seen = epoch;
    return true;
}

// Скретч-буферы точек одного plot-окна (по одному на серию). Между кадрами живут намеренно:
// buf.clear() у вызывающего сохраняет capacity, поэтому со второго кадра push_back уже не
// аллоцирует — на диаграммах в миллионы точек это заметно.
// Ключ — стабильный ParametricPlotWindow::id, как у кэшей HeatmapView выше. Раньше на каждую
// из трёх функций был один static, общий ВСЕМ окнам: при двух окнах с разным числом серий
// assign() передёргивал буферы каждый кадр, и оптимизация не работала (корректность при этом
// не страдала — render() забирает точки синхронно).
// Записи удалённых окон остаются в карте до перезапуска, как и у хитмап: окон единицы.
std::vector<std::vector<double>>& window_point_bufs(int window_id, size_t n_series) {
    static std::map<int, std::vector<std::vector<double>>> cache;
    auto& bufs = cache[window_id];
    if (bufs.size() != n_series) bufs.assign(n_series, {});
    return bufs;
}

// Узлы сетки свипа фокусной кривой окна (Plot2DView::snap_x_nodes). Живут
// между кадрами, как и буферы точек: вид держит указатель на время render().
static std::vector<double>& window_node_buf(int window_id) {
    static std::map<int, std::vector<double>> cache;
    return cache[window_id];
}

// Заполняет узлы ТОЙ ЖЕ формулой, что строит точки, и отдаёт их виду. Узлы, на
// которых точек не оказалось (режим не колебательный, λ не финитна), здесь
// тоже есть: курсор обязан на них вставать — расчёт для них проводился.
void set_snap_nodes(Plot2DView& view, int window_id,
                           int npts, double lo, double hi,
                           bool log_scale, bool reverse, bool continuation) {
    auto& nb = window_node_buf(window_id);
    nb.clear();
    if (npts > 1) {
        nb.reserve((size_t)npts);
        for (int k = 0; k < npts; ++k)
            nb.push_back(sweep_value_at(k, npts, lo, hi, log_scale, reverse, continuation));
    }
    view.snap_x_nodes      = nb.empty() ? nullptr : nb.data();
    view.snap_x_node_count = (int)nb.size();
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
bool draw_heatmap_toolbar(HeatmapView& hv, const HeatmapToolbarOpts& o) {
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
