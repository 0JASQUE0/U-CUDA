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
#include "digit_input.h"
#include "krs_cpu.h"
#include <map>
#include <memory>
#include <cstdio>
#include <regex>
#include <unordered_map>
#include <limits>
#include <array>
#include <deque>

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
static ImVec4 ic_var_shade(int ic_index, int vi, int nv) {
    ImVec4 base = ic_base_color(ic_index);
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(base.x, base.y, base.z, h, s, v);
    // распределяем насыщенность от 1.0 (vi=0) до ~0.35 (последняя переменная)
    float frac = (nv <= 1) ? 0.0f : (float)vi / (float)(nv - 1);
    float new_s = s * (1.0f - 0.45f * frac); // от s до 0.55*s (было 0.65 -> 0.35, тускло)
    float r, g, b;
    ImGui::ColorConvertHSVtoRGB(h, new_s, v, r, g, b);
    return ImVec4(r, g, b, base.w);
}
#include <vector>
#include <memory>
#include <string>
#include <unordered_set>
#include <cstring>
#include <algorithm>

// ---- helpers: std::string <-> ImGui ----
static bool InputTextMultilineStr(const char* label, std::string& str, const ImVec2& size) {
    std::vector<char> buf(str.begin(), str.end());
    buf.resize(str.size() + 4096);
    buf[str.size()] = '\0';
    bool changed = ImGui::InputTextMultiline(label, buf.data(), buf.size(), size);
    if (changed) str = buf.data();
    return changed;
}
static bool InputTextStr(const char* label, std::string& str, float width = 0.0f) {
    std::vector<char> buf(str.begin(), str.end());
    buf.resize(str.size() + 1024);
    buf[str.size()] = '\0';
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

// Совмещённый callback: запятая→точка (CallbackCharFilter) + digit-step на
// ↑/↓ (CallbackHistory). ImGui позволяет OR'ить флаги; здесь диспетчеризуем
// по EventFlag. CallbackHistory — специальный event, который ImGui шлёт
// когда в активном InputText нажали ↑/↓ (изначально сделан под REPL command
// history). Ровно то, что нам нужно: клавиша уже отфильтрована и передана
// нам через колбэк — не нужен ни IsKeyPressed, ни pending-cursor state.
static int digit_step_callback(ImGuiInputTextCallbackData* data) {
    if (data->EventFlag == ImGuiInputTextFlags_CallbackCharFilter) {
        if (data->EventChar == ',') data->EventChar = '.';
        return 0;
    }
    if (data->EventFlag == ImGuiInputTextFlags_CallbackHistory) {
        int dir = 0;
        if (data->EventKey == ImGuiKey_UpArrow)   dir = +1;
        if (data->EventKey == ImGuiKey_DownArrow) dir = -1;
        if (dir == 0) return 0;

        std::string text(data->Buf, data->Buf + data->BufTextLen);
        std::string new_text;
        int new_cursor = 0;
        if (!DigitInput::ComputeStep(text, data->CursorPos, dir,
                                     new_text, new_cursor)) {
            return 0;  // Дробь / scientific / невалидный ввод — не трогаем.
        }

        // DeleteChars + InsertChars сами выставляют BufDirty=true, ImGui
        // подхватит новую длину и вернёт changed=true из InputText.
        data->DeleteChars(0, data->BufTextLen);
        data->InsertChars(0, new_text.c_str());
        if (new_cursor < 0) new_cursor = 0;
        if (new_cursor > data->BufTextLen) new_cursor = data->BufTextLen;
        data->CursorPos      = new_cursor;
        data->SelectionStart = new_cursor;
        data->SelectionEnd   = new_cursor;
    }
    return 0;
}

// Проверка: парсится ли строка как число (или валидная дробь "a/b")?
// Важно: std::stod НЕ кидает на "5asdfaxcv" — он парсит ведущее "5"
// и тихо игнорирует остальное. Поэтому проверяем pos — что вся строка
// (после возможных пробелов) реально была сконвертирована.
// Пустая считается валидной (дефолт подставится дальше).
// "8/3" — валидная дробь, "2/x" / "8/0" / "8/" / "5asdfaxcv" — нет.
static bool is_numeric_string(const std::string& s) {
    if (s.empty()) return true;

    // Полностью ли строка v сконвертирована в число (плюс trailing whitespace)?
    auto parse_complete = [](const std::string& v) -> bool {
        if (v.empty()) return false;
        try {
            size_t pos = 0;
            std::stod(v, &pos);
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
static bool is_custom_scheme(const std::string& scheme_name,
                              const std::vector<CustomScheme>& custom_schemes) {
    for (const auto& cs : custom_schemes) if (cs.name == scheme_name) return true;
    return false;
}

// Heuristic text match: does a custom KRS body actually reference a[0]
// (the symmetry slot, same as built-in CD)? Raw C/CUDA source, not parsed
// into an AST, so this is a regex over the literal text rather than a real
// usage analysis — good enough since users write "a[0]" directly.
static bool custom_scheme_uses_symmetry(const std::string& scheme_name,
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

static std::string auto_axis_name(const std::vector<std::string>& params,
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

static std::string auto_label_bd(const BifurcationDiagramConfig& bd,
                                  const std::vector<std::string>& params,
                                  const std::vector<std::string>& vars) {
    std::string x = auto_axis_name(params, vars, bd.param_index, bd.sweep_over_var, bd.var_sweep_index, bd.sweep_over_h);
    if (bd.mode_2d) {
        std::string y = auto_axis_name(params, vars, bd.param_index_2, bd.sweep_over_var_2, bd.var_sweep_index_2, bd.sweep_over_h_2);
        return x + " x " + y;
    }
    const std::string& lo = bd.param_lo_text.empty() ? std::string("?") : bd.param_lo_text;
    const std::string& hi = bd.param_hi_text.empty() ? std::string("?") : bd.param_hi_text;
    return x + " [" + lo + ".." + hi + "]";
}

static std::string auto_label_lle(const LLECurveConfig& c,
                                   const std::vector<std::string>& params,
                                   const std::vector<std::string>& vars) {
    std::string x = auto_axis_name(params, vars, c.param_index, c.sweep_over_var, c.var_sweep_index, c.sweep_over_h);
    if (c.mode_2d) {
        std::string y = auto_axis_name(params, vars, c.param_index_2, c.sweep_over_var_2, c.var_sweep_index_2, c.sweep_over_h_2);
        return x + " x " + y;
    }
    const std::string& lo = c.param_lo_text.empty() ? std::string("?") : c.param_lo_text;
    const std::string& hi = c.param_hi_text.empty() ? std::string("?") : c.param_hi_text;
    return x + " [" + lo + ".." + hi + "]";
}

static std::string auto_label_ls(const LSCurveConfig& c,
                                  const std::vector<std::string>& params,
                                  const std::vector<std::string>& vars) {
    std::string x = auto_axis_name(params, vars, c.param_index, c.sweep_over_var, c.var_sweep_index, c.sweep_over_h);
    if (c.mode_2d) {
        std::string y = auto_axis_name(params, vars, c.param_index_2, c.sweep_over_var_2, c.var_sweep_index_2, c.sweep_over_h_2);
        return x + " x " + y;
    }
    const std::string& lo = c.param_lo_text.empty() ? std::string("?") : c.param_lo_text;
    const std::string& hi = c.param_hi_text.empty() ? std::string("?") : c.param_hi_text;
    return x + " [" + lo + ".." + hi + "]";
}

static std::string auto_label_window(const ParametricPlotWindow& w) {
    const char* kn = w.kind == ParametricPlotWindow::Kind::Bifurcation ? "Bifurcation"
                    : w.kind == ParametricPlotWindow::Kind::LLE ? "LLE" : "LS";
    const char* suffix = w.colored_1d ? "Colored 1D" : (w.mode_2d ? "2D" : "1D");
    return std::string(kn) + " " + suffix;
}

static std::string auto_label_dft1d(const Dft1DConfig& c,
                                    const std::vector<std::string>& params,
                                    const std::vector<std::string>& vars) {
    std::string x = auto_axis_name(params, vars, c.param_index, c.sweep_over_var, c.var_sweep_index);
    const std::string& lo = c.param_lo_text.empty() ? std::string("?") : c.param_lo_text;
    const std::string& hi = c.param_hi_text.empty() ? std::string("?") : c.param_hi_text;
    return x + " [" + lo + ".." + hi + "]";
}

static void refresh_auto_labels(AppModel& model) {
    for (auto& bd : model.bifurcation_session.diagrams)
        if (!bd.label_is_manual)
            bd.label = auto_label_bd(bd, model.bifurcation_session.params, model.bifurcation_session.vars);
    for (auto& c : model.lle_session.curves)
        if (!c.label_is_manual)
            c.label = auto_label_lle(c, model.lle_session.params, model.lle_session.vars);
    for (auto& c : model.ls_session.curves)
        if (!c.label_is_manual)
            c.label = auto_label_ls(c, model.ls_session.params, model.ls_session.vars);
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

// Общий парсер для полей вроде HeatmapView::manual_vmin_text — тот же
// try/stod-с-дефолтом, что и локальный parse_d_local в FastSync-плоте, но
// переиспользуемый (нужен в 4 местах: Bif2D/Colored1D/LLE2D/LS2D vmin/vmax).
static double parse_num_or(const std::string& s, double def) {
    if (s.empty()) return def;
    try { return std::stod(s); } catch (...) { return def; }
}

static bool InputNumStr(const char* label, std::string& str, float width = 0.0f) {
    std::vector<char> buf(str.begin(), str.end());
    buf.resize(str.size() + 1024);
    buf[str.size()] = '\0';
    if (width > 0) ImGui::SetNextItemWidth(width);
    // CallbackHistory — ↑/↓ в активном InputText, обрабатываем в digit_step_callback.
    // CallbackCharFilter — прежняя замена запятой на точку, тоже в digit_step_callback.
    bool changed = ImGui::InputText(label, buf.data(), buf.size(),
        ImGuiInputTextFlags_CallbackCharFilter | ImGuiInputTextFlags_CallbackHistory,
        digit_step_callback);
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

// Парсер числа с поддержкой дроби "a/b" (пользователь пишет "pi/4", "1/3" в
// полях диапазонов). Раньше — пять идентичных локальных лямбд safe_stod в
// draw_bifurcation_plot / draw_lle_plot / draw_ls_plot / draw_custom_plot_
// windows. В отличие от parse_num_or (выше) понимает слэш.
static double parse_ratio_or(const std::string& v, double def) {
    if (v.empty()) return def;
    size_t slash = v.find('/');
    if (slash != std::string::npos) {
        double num = std::atof(v.substr(0, slash).c_str());
        double den = std::atof(v.substr(slash + 1).c_str());
        if (den != 0) return num / den;
    }
    return std::atof(v.c_str());
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
static bool draw_point_style_toolbar(BifurcationDiagramConfig& bd, const char* id_suffix) {
    bool changed = false;
    const std::string sfx = id_suffix;
    if (ImGui::Checkbox(("Custom point style##" + sfx).c_str(), &bd.custom_point_style))
        changed = true;
    if (bd.custom_point_style) {
        if (bd.point_marker < 0 || bd.point_marker >= kPointMarkerCount) bd.point_marker = 0;
        ImGui::SameLine(); ImGui::SetNextItemWidth(110);
        if (ImGui::Combo(("Marker##" + sfx).c_str(), &bd.point_marker,
                         kPointMarkerNames, kPointMarkerCount)) changed = true;
        ImGui::SameLine(); ImGui::SetNextItemWidth(120);
        if (ImGui::SliderFloat(("Point size##" + sfx).c_str(), &bd.point_size,
                               0.5f, 12.0f, "%.1f")) changed = true;
        ImGui::SameLine(); ImGui::SetNextItemWidth(120);
        if (ImGui::SliderFloat(("Alpha##" + sfx).c_str(), &bd.point_alpha,
                               0.0f, 1.0f, "%.2f")) changed = true;
    }
    return changed;
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
// has_mode_2d нужен потому, что у BifurcationDiagramConfig 2D-флаг есть, а
// проверять его единообразно удобнее снаружи.
template <class Cfg>
static void draw_continuation_device_block(Cfg& c, const char* id,
                                           bool blocked_extra = false) {
    const std::string sfx = id;
    const bool blocked = c.mode_2d || c.sweep_over_var || blocked_extra;
    if (blocked) {
        c.continuation = false;
        c.continuation_reverse = false;
    }

    ImGui::BeginDisabled(blocked);
    bool cont = c.continuation;
    if (ImGui::Checkbox(("Continuation##" + sfx).c_str(), &cont)) {
        c.continuation = cont;
        // При включении continuation по умолчанию уходим на CPU: GPU-ветка
        // там однопоточная и заметно медленнее. Пользователь может вернуть GPU
        // вручную — радио ниже остаётся активным.
        if (cont) c.use_gpu = false;
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
    ImGui::BeginDisabled(c.mode_2d);
    ImGui::RadioButton(("GPU##dev" + sfx).c_str(), &dev, 0); ImGui::SameLine();
    ImGui::RadioButton(("CPU##dev" + sfx).c_str(), &dev, 1);
    ImGui::EndDisabled();
    if (!c.mode_2d) c.use_gpu = (dev == 0);
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
    const int n = std::atoi(n_text.c_str());
    apply_snap_x(view, parse_ratio_or(lo_text, 0.0), parse_ratio_or(hi_text, 1.0), n);
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
        const int cm = (cfg_colormap >= 0) ? cfg_colormap : app_default_colormap;
        if (cm >= 0 && cm < kHeatmapColormapCount) slot->colormap = (HeatmapColormap)cm;
        if (cfg_exponent_idx != kNoExponent) slot->display_exponent_idx = cfg_exponent_idx;
    }
    return *slot;
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
    ImGui::SetNextItemWidth(140);
    if (ImGui::Combo("Colormap", &cmap_idx, kHeatmapColormapNames, kHeatmapColormapCount)) {
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
        hv.manual_vmin = (float)parse_num_or(hv.manual_vmin_text, hv.manual_vmin);
        ImGui::SameLine();
        if (InputNumStr("vmax", hv.manual_vmax_text, 80)) changed = true;
        hv.manual_vmax = (float)parse_num_or(hv.manual_vmax_text, hv.manual_vmax);
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
                if (c2 < cact.result_2d.flags.size() && cact.result_2d.flags[c2] < 0) {
                    cact.sum_cache[c2] = 999.0;
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
        static const char* builtin_names[] = {
            "Euler", "Euler-Cromer", "Explicit Midpoint", "RK4", "DOPRI78", "CD"
        };
        if (ImGui::Button("+ Add custom scheme")) {
            // подобрать уникальное имя "Custom N"
            int n = (int)model.custom_schemes.size() + 1;
            std::string candidate;
            auto name_clash = [&](const std::string& nm) {
                for (const char* b : builtin_names) if (nm == b) return true;
                for (const auto& cs : model.custom_schemes) if (cs.name == nm) return true;
                return false;
            };
            do { candidate = "Custom " + std::to_string(n++); } while (name_clash(candidate));
            CustomScheme cs; cs.name = candidate;
            model.custom_schemes.push_back(std::move(cs));
        }

        // подсветка конфликтов
        for (const auto& cs : model.custom_schemes) {
            for (const char* b : builtin_names) {
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
    InputNumStr("##step_h", model.step_h, 120);
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
static std::string trim_copy(const std::string& s) {
    auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
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

    std::vector<std::string> names = lib.list();
    if (names.empty()) {
        ImGui::TextDisabled("(library is empty)");
        return;
    }

    // Delete confirmation flow: row buttons only set pending_delete + a
    // one-shot open flag; OpenPopup/BeginPopupModal are called at a stable
    // ID-stack level below the table so the popup id stays consistent.
    static std::string pending_delete;
    static bool        want_open_confirm = false;

    // Custom-scheme count per name, loaded once and reused both for sizing
    // the Name column and for the row badge (avoids loading each system twice).
    std::vector<int> cs_counts(names.size(), 0);
    for (size_t i = 0; i < names.size(); ++i) {
        try { cs_counts[i] = (int)lib.load(names[i]).custom_schemes.size(); } catch (...) {}
    }

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
                try { lib.duplicate(n); }
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
        static std::string cached_name;
        static std::string cached_note;
        if (cached_name != model.library_selected_name) {
            try { cached_note = lib.load(model.library_selected_name).note; }
            catch (...) { cached_note.clear(); }
            cached_name = model.library_selected_name;
        }
        ImGui::Text("Selected: %s", model.library_selected_name.c_str());
        if (cached_note.empty()) {
            ImGui::TextDisabled("(no note)");
        } else {
            ImGui::TextWrapped("%s", cached_note.c_str());
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
    ImGui::SetNextItemWidth(160);
    static const char* methods[] = { "Euler", "Euler-Cromer", "Explicit Midpoint", "RK4", "DOPRI78", "CD" };
    if (ImGui::BeginCombo("##method", s.scheme.c_str())) {
        for (auto m : methods)
            if (ImGui::Selectable(m, s.scheme == m)) {
                s.scheme = m; s.regenerate_krs(); changed = true;
            }
        if (!s.custom_schemes.empty()) ImGui::Separator();
        for (const auto& cs : s.custom_schemes)
            if (ImGui::Selectable((cs.name + " (custom)").c_str(), s.scheme == cs.name)) {
                s.scheme = cs.name;
                s.regenerate_krs();
                changed = true;
            }
        ImGui::EndCombo();
    }
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
        const char* tnames[] = { "Phase 2D", "Time domain", "Phase 3D" };
        int t = (int)pr.type;
        if (ImGui::Combo("##ptype", &t, tnames, 3)) { pr.type = (ProjType)t; s.fit_request = true; }
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
            // синхронизируем размер show_var
            if ((int)pr.show_var.size() != (int)s.vars.size())
                pr.show_var.assign(s.vars.size(), true);
            ImGui::Text("vars:"); ImGui::SameLine();
            for (int k = 0; k < (int)s.vars.size(); ++k) {
                bool v = pr.show_var[k];
                if (ImGui::Checkbox(s.vars[k].c_str(), &v)) { pr.show_var[k] = v; }
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
        std::string body_cpu;
        try {
            body_cpu = codegen_scheme_cpu_equivalent(s.sys, scheme_from_name(s.scheme));
        } catch (...) {
            body_cpu = "(generation failed)";
        }
        if (body_cpu.empty()) body_cpu = "(empty — same as GPU for non-CD schemes)";
        std::vector<char> buf_cpu(body_cpu.begin(), body_cpu.end());
        buf_cpu.resize(body_cpu.size() + 1);
        buf_cpu[body_cpu.size()] = '\0';
        ImGui::InputTextMultiline("##krs_cpu", buf_cpu.data(), buf_cpu.size(),
            ImVec2(-1, 200), ImGuiInputTextFlags_ReadOnly);
        if (s.scheme != "CD")
            ImGui::TextDisabled("(for %s CPU and GPU evaluate the same AST — texts match)", s.scheme.c_str());
        else
            ImGui::TextDisabled("(for CD: GPU uses analytic solve for linear vars; CPU always uses 4 iterations)");

        ImGui::SeparatorText("Parsed inputs (double, %.17g)");
        auto parse = [](const std::string& v, double def) -> double {
            if (v.empty()) return def;
            size_t sl = v.find('/');
            if (sl != std::string::npos) {
                double n = std::atof(v.substr(0, sl).c_str());
                double d = std::atof(v.substr(sl + 1).c_str());
                if (d != 0) return n / d;
            }
            return std::atof(v.c_str());
        };

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

// Рисует окна проекций (каждая — отдельное docking-окно с графиком).
// Optional `before_begin` runs immediately before each projection window's
// ImGui::Begin (Custom mode uses it to assign initial dock target); optional
// `after_begin` runs right after Begin returns true (Custom mode uses it to
// attach the Move-to-Tab context menu). Both empty (default) preserve the
// Analysis-mode behaviour.
using ProjHookFn = std::function<void(int proj_index, const std::string& title)>;
// `title_suffix` is appended to every projection window's title (e.g.
// "##sys_<name>") so imgui.ini stores dock state per system in Custom mode.
// `owner_id_delta` is XOR'd into the per-projection owner_id passed to
// Plot2D/3D so the SHARED PlotRenderer cache (static in this function) is
// keyed per system — without it, Chen and Rossler both used owner_id=0
// for their first projection and Rossler saw Chen's cached FBO texture.
// Analysis mode passes zero and gets the previous behaviour.
static void draw_projection_windows(PhaseAnalysisSession& s, const GuiCallbacks& cb,
                                    const ProjHookFn& before_begin = {},
                                    const ProjHookFn& after_begin  = {},
                                    const std::string& title_suffix = {},
                                    int owner_id_delta = 0) {
    const AnalysisResult& res = s.result;
    // Lambda installed on every projection view that has data — right-click
    // "Export data..." writes the full double-precision trajectory set (all
    // coords for all ICs) + a _config.csv sidecar. Ставится и на Phase3D:
    // Plot3DView теперь тоже поддерживает popup_extras, поэтому экспорт
    // доступен во всех трёх типах проекций одинаково.
    const bool phase_busy = s.in_flight;
    auto phase_popup_extras = [&res, &cb, phase_busy]() {
        const bool has_data = res.ok && !res.trajectories.empty();
        if (ImGui::MenuItem("Export data...", nullptr, false,
                            has_data && !phase_busy)) {
            if (cb.pick_save_file_csv) {
                std::string path = cb.pick_save_file_csv();
                if (!path.empty())
                    data_export::export_phase(res, res.snapshot, path);
            }
        }
    };
    // Свой offscreen-рендерер (FBO/текстура) на КАЖДУЮ проекцию: иначе все окна
    // показывали бы одну общую текстуру (геометрию последней отрисованной).
    // PlotRenderer некопируемый -> храним через unique_ptr, подгоняем под число проекций.
    static std::vector<std::unique_ptr<PlotRenderer>> renderers;
    if ((int)renderers.size() != (int)s.projections.size()) {
        renderers.clear();
        for (size_t k = 0; k < s.projections.size(); ++k)
            renderers.push_back(std::make_unique<PlotRenderer>());
    }
    int pr_to_remove = -1;
    for (int i = 0; i < (int)s.projections.size(); ++i) {
        Projection& pr = s.projections[i];
        PlotRenderer& renderer = *renderers[i]; // рендерер этой проекции
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

                    // Toolbar над плотом: opt-in custom line styling (ImDrawList-путь
                    // с настраиваемой толщиной + α). По дефолту выключено → быстрый
                    // GL shader-line путь (1px, α=1). Текущая отрисовка не ломается.
                    ImGui::Checkbox("Custom line style##phase2d", &pr.custom_line_style);
                    if (pr.custom_line_style) {
                        ImGui::SameLine(); ImGui::SetNextItemWidth(120);
                        ImGui::SliderFloat("Line width##phase2d", &pr.line_width, 0.1f, 5.0f, "%.2f");
                        ImGui::SameLine(); ImGui::SetNextItemWidth(120);
                        ImGui::SliderFloat("Alpha##phase2d",      &pr.alpha,      0.0f, 1.0f, "%.2f");
                    }
                    pr.view2d->imdraw_lines      = pr.custom_line_style;
                    pr.view2d->line_thickness_px = pr.line_width;

                    // подготовить серии: для каждой траектории выбираем координаты по (ax, ay)
                    // храним float-массивы локально в статике, чтобы указатели жили до конца кадра
                    static std::vector<std::vector<float>> series_data;
                    series_data.clear();
                    series_data.resize(res.trajectories.size());

                    std::vector<PlotSeriesInput> series_in;
                    series_in.reserve(res.trajectories.size());

                    // видимость: глобальная — из живых галочек НУ (меняется без recompute)
                    std::vector<bool> init_vis(res.trajectories.size(), true);
                    std::vector<bool> glob_vis(res.trajectories.size(), true);

                    for (size_t k = 0; k < res.trajectories.size(); ++k) {
                        const auto& traj = res.trajectories[k];
                        auto& buf = series_data[k];
                        buf.reserve(traj.size() * 2);
                        for (const auto& pt : traj) {
                            buf.push_back((float)pt[ax < (int)pt.size() ? ax : 0]);
                            buf.push_back((float)pt[ay < (int)pt.size() ? ay : 0]);
                        }
                        std::string lab = (k < res.labels.size()) ? res.labels[k] : ("IC " + std::to_string(k + 1));
                        if (s.legend_show_ic && k < res.ic_text.size()) lab = res.ic_text[k];

                        PlotSeriesInput si;
                        si.points = buf.data();
                        si.n_points = (int)(buf.size() / 2);
                        si.color = ic_base_color((int)k);
                        // В custom_line_style режиме применяем α к цвету IC
                        // (rendering: ImDrawList использует color.w как alpha).
                        if (pr.custom_line_style) si.color.w = pr.alpha;
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

                    // Toolbar над плотом: opt-in custom line styling (ImDrawList-путь
                    // с настраиваемой толщиной + α). Дефолт — быстрый GL shader-line
                    // путь (1px, α=1). Аналогично Phase2D.
                    ImGui::Checkbox("Custom line style##timedomain", &pr.custom_line_style);
                    if (pr.custom_line_style) {
                        ImGui::SameLine(); ImGui::SetNextItemWidth(120);
                        ImGui::SliderFloat("Line width##timedomain", &pr.line_width, 0.1f, 5.0f, "%.2f");
                        ImGui::SameLine(); ImGui::SetNextItemWidth(120);
                        ImGui::SliderFloat("Alpha##timedomain",      &pr.alpha,      0.0f, 1.0f, "%.2f");
                    }
                    pr.view2d->imdraw_lines      = pr.custom_line_style;
                    pr.view2d->line_thickness_px = pr.line_width;

                    double h = atof(s.step_h.c_str()); if (h <= 0) h = 0.01;
                    int dec = atoi(s.decimation.c_str()); if (dec < 1) dec = 1;
                    double dt = h * dec;
                    int nvars = (int)s.vars.size();

                    // синхронизируем show_var с числом переменных
                    if ((int)pr.show_var.size() != nvars)
                        pr.show_var.assign(nvars, true);

                    pr.view2d->x_axis.name = "t";
                    pr.view2d->y_axis.name = "value";

                    pr.view2d->pad_x = false;
                    pr.view2d->show_zero_x = false;

                    // серии: одна на (траектория k, видимая переменная vi)
                    // храним буферы в статике, чтобы указатели жили до render
                    static std::vector<std::vector<float>> series_data;
                    series_data.clear();

                    std::vector<PlotSeriesInput> series_in;
                    std::vector<bool> init_vis;
                    std::vector<bool> glob_vis;

                    for (size_t k = 0; k < res.trajectories.size(); ++k) {
                        const auto& traj = res.trajectories[k];
                        if (traj.empty()) continue;
                        int n = (int)traj.size();
                        // видимость НУ — из живой галочки (без recompute)
                        bool ic_vis = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;

                        for (int vi = 0; vi < nvars; ++vi) {
                            if (vi < (int)pr.show_var.size() && !pr.show_var[vi]) continue;

                            series_data.emplace_back();
                            auto& buf = series_data.back();
                            buf.reserve(n * 2);
                            for (int t = 0; t < n; ++t) {
                                buf.push_back((float)(t * dt));
                                buf.push_back((float)traj[t][vi < (int)traj[t].size() ? vi : 0]);
                            }

                            std::string base = (k < res.labels.size()) ? res.labels[k] : ("IC" + std::to_string(k + 1));
                            std::string who = (s.legend_show_ic && k < res.ic_text.size()) ? res.ic_text[k] : base;
                            std::string lab = s.vars[vi] + " [" + who + "]";

                            PlotSeriesInput si;
                            si.points = buf.data();
                            si.n_points = n;
                            si.color = ic_var_shade((int)k, vi, nvars);
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

                    // Toolbar над плотом: opt-in custom line styling (толщина + α).
                    // В 3D нет ImDrawList-fallback (потерялся бы depth-sorting),
                    // толщина идёт через glLineWidth — драйвер может клампить,
                    // α точно уходит в шейдер. При выключенной фиче — восстанавливаем
                    // старый хардкод 1.5f, чтобы поведение осталось прежним.
                    ImGui::Checkbox("Custom line style##phase3d", &pr.custom_line_style);
                    if (pr.custom_line_style) {
                        ImGui::SameLine(); ImGui::SetNextItemWidth(120);
                        ImGui::SliderFloat("Line width##phase3d", &pr.line_width, 0.1f, 5.0f, "%.2f");
                        ImGui::SameLine(); ImGui::SetNextItemWidth(120);
                        ImGui::SliderFloat("Alpha##phase3d",      &pr.alpha,      0.0f, 1.0f, "%.2f");
                    }
                    pr.view3d->line_thickness_px = pr.custom_line_style ? pr.line_width : 1.5f;
                    pr.view3d->custom_line_style = pr.custom_line_style;

                    pr.view3d->x_name = s.vars.empty() ? "x" : s.vars[ax < (int)s.vars.size() ? ax : 0];
                    pr.view3d->y_name = s.vars.empty() ? "y" : s.vars[ay < (int)s.vars.size() ? ay : 0];
                    pr.view3d->z_name = s.vars.empty() ? "z" : s.vars[az < (int)s.vars.size() ? az : 0];

                    static std::vector<std::vector<float>> series_data;
                    series_data.clear();
                    series_data.resize(res.trajectories.size());

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
                        if (s.legend_show_ic && k < res.ic_text.size()) lab = res.ic_text[k];

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

    auto combo_label = [&]() -> std::string {
        if (N >= 3)      return vars[0] + " + pi*" + vars[1] + " + e*" + vars[2];
        else /* N==2 */  return vars[0] + " + pi*" + vars[1];
    };
    const std::string preview = (writable_var == -1) ? combo_label() : vars[writable_var];

    ImGui::SetNextItemWidth(220);
    if (ImGui::BeginCombo(combo_id, preview.c_str())) {
        for (int i = 0; i < N; ++i) {
            bool sel = writable_var == i;
            if (ImGui::Selectable(vars[i].c_str(), sel)) writable_var = i;
        }
        if (N >= 2) {
            ImGui::Separator();
            const std::string lbl = combo_label();
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
static bool draw_diagram_controls(BifurcationAnalysisSession& s, int idx) {
    BifurcationDiagramConfig& bd = s.diagrams[idx];

    ImGui::SetNextItemWidth(160);
    if (InputTextStr("Label", bd.label))
        bd.label_is_manual = !bd.label.empty();   // empty → back to auto
    ImGui::Separator();

    // ----- Scheme (built-in + custom) -----
    static const char* schemes[] = { "Euler", "Euler-Cromer", "Explicit Midpoint", "RK4", "DOPRI78", "CD" };
    ImGui::SetNextItemWidth(160);
    if (ImGui::BeginCombo("Scheme", bd.scheme.c_str())) {
        for (auto m : schemes)
            if (ImGui::Selectable(m, bd.scheme == m)) bd.scheme = m;
        if (!s.custom_schemes.empty()) ImGui::Separator();
        for (const auto& cs : s.custom_schemes)
            if (ImGui::Selectable((cs.name + " (custom)").c_str(), bd.scheme == cs.name))
                bd.scheme = cs.name;
        ImGui::EndCombo();
    }
    ImGui::Separator();

    // ----- Sweep target (parameter ИЛИ initial condition) -----
    // Один combo с разделителем: сверху параметры, снизу переменные (IC).
    // Выбор переменной → BD строится по начальному условию (par_or_var = 0
    // в engine), runtime-флаг — никакой пересборки PTX.
    if (!s.params.empty() || !s.vars.empty()) {
        // Бочки валидации индексов на случай старых сохранений.
        if (bd.param_index < 0 || bd.param_index >= (int)s.params.size())
            bd.param_index = 0;
        if (bd.var_sweep_index < 0 || bd.var_sweep_index >= (int)s.vars.size())
            bd.var_sweep_index = 0;

        std::string preview;
        if (bd.sweep_over_h)
            preview = "dt (h)";
        else if (bd.sweep_over_var && !s.vars.empty())
            preview = s.vars[bd.var_sweep_index] + " (IC)";
        else if (!s.params.empty())
            preview = s.params[bd.param_index];
        else
            preview = "?";

        ImGui::SetNextItemWidth(160);
        if (ImGui::BeginCombo("Sweep", preview.c_str())) {
            for (int i = 0; i < (int)s.params.size(); ++i) {
                bool sel = !bd.sweep_over_var && !bd.sweep_over_h && bd.param_index == i;
                if (ImGui::Selectable(s.params[i].c_str(), sel)) {
                    bd.sweep_over_var = false;
                    bd.sweep_over_h = false;
                    bd.param_index = i;
                }
            }
            if (!s.params.empty() && !s.vars.empty()) ImGui::Separator();
            for (int i = 0; i < (int)s.vars.size(); ++i) {
                std::string lbl = s.vars[i] + " (IC)";
                bool sel = bd.sweep_over_var && !bd.sweep_over_h && bd.var_sweep_index == i;
                if (ImGui::Selectable(lbl.c_str(), sel)) {
                    bd.sweep_over_var = true;
                    bd.sweep_over_h = false;
                    bd.var_sweep_index = i;
                }
            }
            // dt (h) доступен и в continuation: шаг пересчитывается в каждой
            // точке цепочки (см. run_bif1d_continuation / _cpu). Раньше пункт
            // прятался, потому что h-свип с continuation отвергался движком.
            ImGui::Separator();
            if (ImGui::Selectable("dt (h)", bd.sweep_over_h)) {
                bd.sweep_over_h = true;
                bd.sweep_over_var = false;
                if (bd.mode_2d) bd.sweep_over_h_2 = false;  // ровно одна ось = h
            }
            ImGui::EndCombo();
        }
    }
    else {
        ImGui::TextDisabled("No parameters/variables (select a system first)");
    }
    InputNumStr(bd.sweep_over_h ? "h lo" : "Param lo", bd.param_lo_text, 120);
    InputNumStr(bd.sweep_over_h ? "h hi" : "Param hi", bd.param_hi_text, 120);
    ImGui::Checkbox("Log scale##bd_log", &bd.log_scale);
    if (bd.log_scale) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }

    // Continuation: каждая следующая точка стартует с конечного x[] предыдущей.
    // Единый блок с LLE и LS 1D (см. draw_continuation_device_block) — он же
    // гасит и сбрасывает флаги в 2D-режиме: run_bif2d идёт своим 3-kernel
    // pipeline'ом и continuation игнорирует, скрытое состояние туда утекать не
    // должно. Классический свип у БД CPU-реализации не имеет, поэтому радио
    // GPU/CPU влияет только на continuation.
    draw_continuation_device_block(bd, "bd");

    ImGui::Separator();

    // ----- 2D mode: хитмап «период»(p1, p2) через DBSCAN -----
    if (ImGui::Checkbox("2D mode (period heatmap)", &bd.mode_2d) && bd.mode_2d)
        bd.colored_1d = false;   // взаимоисключающе с Colored 1D diagram
    if (bd.mode_2d) {
        ImGui::Indent();
        if (!s.params.empty() || !s.vars.empty()) {
            if (bd.param_index_2 < 0 || bd.param_index_2 >= (int)s.params.size())
                bd.param_index_2 = 0;
            if (bd.var_sweep_index_2 < 0 || bd.var_sweep_index_2 >= (int)s.vars.size())
                bd.var_sweep_index_2 = 0;
            std::string preview2;
            if (bd.sweep_over_h_2)
                preview2 = "dt (h)";
            else if (bd.sweep_over_var_2 && !s.vars.empty())
                preview2 = s.vars[bd.var_sweep_index_2] + " (IC)";
            else if (!s.params.empty())
                preview2 = s.params[bd.param_index_2];
            else
                preview2 = "?";
            ImGui::SetNextItemWidth(160);
            if (ImGui::BeginCombo("Sweep Y", preview2.c_str())) {
                for (int i = 0; i < (int)s.params.size(); ++i) {
                    bool sel = !bd.sweep_over_var_2 && !bd.sweep_over_h_2 && bd.param_index_2 == i;
                    if (ImGui::Selectable(s.params[i].c_str(), sel)) {
                        bd.sweep_over_var_2 = false;
                        bd.sweep_over_h_2 = false;
                        bd.param_index_2 = i;
                    }
                }
                if (!s.params.empty() && !s.vars.empty()) ImGui::Separator();
                for (int i = 0; i < (int)s.vars.size(); ++i) {
                    std::string lbl = s.vars[i] + " (IC)";
                    bool sel = bd.sweep_over_var_2 && !bd.sweep_over_h_2 && bd.var_sweep_index_2 == i;
                    if (ImGui::Selectable(lbl.c_str(), sel)) {
                        bd.sweep_over_var_2 = true;
                        bd.sweep_over_h_2 = false;
                        bd.var_sweep_index_2 = i;
                    }
                }
                if (!bd.sweep_over_h) {  // ровно одна ось = h -- X уже занял её
                    ImGui::Separator();
                    if (ImGui::Selectable("dt (h)", bd.sweep_over_h_2)) {
                        bd.sweep_over_h_2 = true;
                        bd.sweep_over_var_2 = false;
                    }
                }
                ImGui::EndCombo();
            }
        }
        InputNumStr(bd.sweep_over_h_2 ? "h2 lo" : "Param2 lo", bd.param_lo_2_text, 120);
        InputNumStr(bd.sweep_over_h_2 ? "h2 hi" : "Param2 hi", bd.param_hi_2_text, 120);
        ImGui::Checkbox("Log scale##bd_log2", &bd.log_scale_2);
        if (bd.log_scale_2) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
        InputNumStr("DBSCAN eps", bd.eps_dbscan_text, 120);
        ImGui::TextDisabled("Grid is square (Resolution applies to both axes).");
        ImGui::Unindent();
    }

    ImGui::Separator();

    // ----- Variable + resolution + inter-peaks -----
    draw_writable_var_combo(s.vars, bd.writable_var, "Writable var##bd_wv");
    InputNumStr("Resolution", bd.n_pts_text, 120);
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
        InputNumStr("Y resolution", bd.colored_1d_b_text, 120);
        ImGui::TextDisabled("X resolution follows Resolution above.");
        ImGui::Checkbox("Custom Y range", &bd.colored_1d_custom_y);
        if (bd.colored_1d_custom_y) {
            InputNumStr("Ymin", bd.colored_1d_ymin_text, 120);
            InputNumStr("Ymax", bd.colored_1d_ymax_text, 120);
            ImGui::TextDisabled("Points outside [Ymin, Ymax] are discarded.");
        }
        ImGui::Checkbox("Logarithmic density scale", &bd.colored_1d_log);
    }

    // ----- Integration (collapsible) -----
    if (ImGui::CollapsingHeader("Integration##bd_int", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("h",              bd.h_text,           120);
        if (bd.scheme == "CD" || custom_scheme_uses_symmetry(bd.scheme, s.custom_schemes))
            InputNumStr("symmetry s", bd.symmetry_s,       120);
        InputNumStr("computing time", bd.t_max_text,       120);
        InputNumStr("transient time", bd.transient_text,   120);
        InputNumStr("decimator",      bd.pre_scaller_text, 120);
        InputNumStr("max value",      bd.max_value_text,   120);
    }

    // ----- Initial conditions (collapsible) -----
    if (ImGui::CollapsingHeader("Initial conditions##bd_ic", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (const auto& v : s.vars) {
            ImGui::PushID(v.c_str());
            InputNumStr(v.c_str(), bd.initial_conditions[v], 120);
            ImGui::PopID();
        }
    }

    // ----- Parameters (collapsible) -----
    if (ImGui::CollapsingHeader("Parameters##bd_par", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (const auto& p : s.params) {
            ImGui::PushID(p.c_str());
            InputNumStr(p.c_str(), bd.param_values[p], 120);
            ImGui::PopID();
        }
    }

    // ----- CSV output (collapsible, moved to the bottom) -----
    if (ImGui::CollapsingHeader("CSV output##bd_csv")) {
        ImGui::Checkbox("Save to file", &bd.csv_save_enabled);
        InputTextStr("##csv_path", bd.csv_output_path);
        ImGui::TextDisabled("Path is kept even when save is off. Also writes <path>_config.csv.");
    }

    // Run-кнопка теперь живёт на уровне draw_parametric_controls (рядом с
    // Run all), общая для Bif/LLE/LS. Здесь — только статусная строка.

    if (bd.mode_2d) {
        if (bd.last_run_2d_ok) {
            int total = (int)bd.result_2d.flags.size();
            int diverged = 0;
            for (int f : bd.result_2d.flags) if (f < 0) ++diverged;
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: %dx%d heatmap, period(min..max) = %.0f..%.0f",
                bd.result_2d.n_pts, bd.result_2d.n_pts,
                bd.result_2d.min_val, bd.result_2d.max_val);
            if (diverged) ImGui::TextDisabled("(%d/%d cells diverged)", diverged, total);
        }
        else if (!bd.last_error.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error (selectable, Ctrl+C):");
            ImVec2 sz(-1.0f, ImGui::GetTextLineHeight() * 12);
            ImGui::InputTextMultiline("##par_err_2d",
                const_cast<char*>(bd.last_error.c_str()),
                bd.last_error.size() + 1,
                sz,
                ImGuiInputTextFlags_ReadOnly);
        }
    } else {
        if (bd.last_run_ok) {
            int diverged = 0, total_peaks = 0, max_peaks = 0;
            for (int f : bd.result.flags) {
                if (f < 0) ++diverged;
                else { total_peaks += f; if (f > max_peaks) max_peaks = f; }
            }
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: n_pts=%d, peaks total=%d (max per param=%d)",
                bd.result.n_pts, total_peaks, max_peaks);
            if (diverged) ImGui::TextDisabled("(%d/%d trajectories diverged)", diverged, bd.result.n_pts);
        }
        else if (!bd.last_error.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error (selectable, Ctrl+C):");
            ImVec2 sz(-1.0f, ImGui::GetTextLineHeight() * 12);
            ImGui::InputTextMultiline("##par_err",
                const_cast<char*>(bd.last_error.c_str()),
                bd.last_error.size() + 1,
                sz,
                ImGuiInputTextFlags_ReadOnly);
        }
    }
    return false;  // Run-кнопка перенесена в draw_parametric_controls.
}

static void draw_bifurcation_controls(AppModel& model, SystemLibrary& /*lib*/) {
    BifurcationAnalysisSession& s = model.bifurcation_session;

    // Tab bar: одна вкладка на БД + кнопка "+" справа для добавления новой
    // БД (копия последней). Активная вкладка хранится в s.active_diagram_index
    // и используется Ctrl+R.
    int active_now = -1;
    int run_idx = -1;
    int to_remove = -1;
    if (ImGui::BeginTabBar("##bd_tabs",
                           ImGuiTabBarFlags_Reorderable |
                           ImGuiTabBarFlags_AutoSelectNewTabs |
                           ImGuiTabBarFlags_FittingPolicyScroll)) {
        for (int i = 0; i < (int)s.diagrams.size(); ++i) {
            BifurcationDiagramConfig& bd = s.diagrams[i];
            ImGui::PushID(i);
            bool open = true;
            // ID завязан на индекс, но label — на bd.label, чтобы пользователь
            // видел свежее имя сразу после редактирования.
            std::string tab_id = bd.label + "###bd_tab_" + std::to_string(i);
            // Запрещаем закрывать таб, чей расчёт сейчас идёт.
            bool can_close = !(s.in_flight && s.running_diagram_index == i);
            // Внешний запрос на выбор вкладки (тулбар Colored 1D над плотом —
            // см. request_select_diagram). Без него галка могла включиться у
            // одной БД, а настройки показывались для другой.
            ImGuiTabItemFlags tab_flags = ImGuiTabItemFlags_None;
            if (s.request_select_diagram == i) tab_flags |= ImGuiTabItemFlags_SetSelected;
            if (ImGui::BeginTabItem(tab_id.c_str(), can_close ? &open : nullptr, tab_flags)) {
                active_now = i;
                if (draw_diagram_controls(s, i)) run_idx = i;
                ImGui::EndTabItem();
            }
            if (!open) to_remove = i;
            ImGui::PopID();
        }
        // Кнопка "+ Add" справа. ImGuiTabItemFlags_Trailing удерживает её
        // в конце; NoTooltip убирает дефолтную подсказку.
        if (!s.in_flight) {
            if (ImGui::TabItemButton("+",
                                     ImGuiTabItemFlags_Trailing |
                                     ImGuiTabItemFlags_NoTooltip)) {
                s.add_diagram();
            }
        }
        ImGui::EndTabBar();
    }
    s.request_select_diagram = -1;   // запрос потреблён этим кадром
    if (active_now >= 0) s.active_diagram_index = active_now;
    if (to_remove >= 0) model.remove_bifurcation_diagram(to_remove);

    // Run-кнопка + Ctrl+R теперь живут в draw_parametric_controls (общие для
    // Bif/LLE/LS, слева от Run all). draw_diagram_controls всегда возвращает
    // false, так что run_idx тут остаётся -1 и run_async не вызывается отсюда.
    if (run_idx >= 0 && run_idx < (int)s.diagrams.size()) {
        if (!model.parametric_engine) model.parametric_engine = std::make_unique<ParametricEngine>();
        s.run_async(*model.parametric_engine, run_idx);
    }
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

            auto ax_name = [&](bool sweep_h, bool sweep_var, int p_idx, int v_idx) -> std::string {
                if (sweep_h) return "h";
                if (sweep_var)
                    return (v_idx >= 0 && v_idx < (int)s.vars.size()) ? (s.vars[v_idx] + " (IC)") : "x";
                return (p_idx >= 0 && p_idx < (int)s.params.size()) ? s.params[p_idx] : "param";
            };
            hb.x_axis.name = ax_name(bdact.sweep_over_h,   bdact.sweep_over_var,   bdact.param_index,   bdact.var_sweep_index);
            hb.y_axis.name = ax_name(bdact.sweep_over_h_2, bdact.sweep_over_var_2, bdact.param_index_2, bdact.var_sweep_index_2);
            hb.x_axis.log_scale = bdact.log_scale;
            hb.y_axis.log_scale = bdact.log_scale_2;

            bool fit = bdact.fit_request_2d;
            if (fit) bdact.fit_request_2d = false;

            const bool bd_busy = s.in_flight && s.is_2d_run && idx == s.running_diagram_index;
            hb.popup_extras = [&bdact, &cb, bd_busy]() {
                if (ImGui::MenuItem("Export data...", nullptr, false, !bd_busy)) {
                    if (cb.pick_save_file_csv) {
                        std::string path = cb.pick_save_file_csv();
                        if (!path.empty())
                            data_export::export_bif2d(bdact.result_2d, path);
                    }
                }
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
            int B = std::atoi(bdact.colored_1d_b_text.c_str());
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
                    if (k < (int)bdact.result.flags.size() && bdact.result.flags[k] < 0) continue;
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
                bdact.colored_1d_cache.assign(plane_size, 999.0);
                std::vector<int> counts((size_t)B);
                double vmin =  std::numeric_limits<double>::infinity();
                double vmax = -std::numeric_limits<double>::infinity();
                double yrange = yhi - ylo;
                for (int k = 0; k < npts; ++k) {
                    bool diverged = k >= (int)bdact.result.flags.size() || bdact.result.flags[k] < 0;
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
                if (ImGui::MenuItem("Export data...", nullptr, false, !c1d_busy)) {
                    if (cb.pick_save_file_csv) {
                        std::string path = cb.pick_save_file_csv();
                        if (!path.empty())
                            data_export::export_bif1d(bdact.result, path);
                    }
                }
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
    {
        std::vector<SweepAxisMember> sweep_members;
        for (int idx : win.members) {
            if (idx < 0 || idx >= (int)s.diagrams.size()) continue;
            const auto& bd = s.diagrams[idx];
            if (!bd.last_run_ok) continue;
            if (shared_var_idx == -2) shared_var_idx = bd.writable_var;
            else if (shared_var_idx != bd.writable_var) var_idx_mixed = true;

            SweepAxisMember m;
            m.kind      = bd.sweep_over_h ? 2 : (bd.sweep_over_var ? 1 : 0);
            m.index     = bd.sweep_over_var ? bd.var_sweep_index : bd.param_index;
            m.log_scale = bd.log_scale;
            // continuation сохраняет реальные lo/hi в result, иначе — из текстов.
            const bool from_result = bd.continuation && bd.result.param_hi != bd.result.param_lo;
            m.lo = from_result ? bd.result.param_lo : parse_ratio_or(bd.param_lo_text, 0.0);
            m.hi = from_result ? bd.result.param_hi : parse_ratio_or(bd.param_hi_text, 1.0);
            sweep_members.push_back(m);
        }
        configure_sweep_x_axis(view, sweep_members, s.params, s.vars);
    }
    // Y-label: combination (-1) -> "x0+pi*x1+e*x2"; single var -> имя переменной;
    // mixed / нет данных -> generic "X".
    if (var_idx_mixed || shared_var_idx == -2) {
        view.y_axis.name = "X";
    } else if (shared_var_idx == -1) {
        if (s.vars.size() >= 3)
            view.y_axis.name = s.vars[0] + "+pi*" + s.vars[1] + "+e*" + s.vars[2];
        else if (s.vars.size() == 2)
            view.y_axis.name = s.vars[0] + "+pi*" + s.vars[1];
        else if (!s.vars.empty())
            view.y_axis.name = s.vars[0];
        else
            view.y_axis.name = "X";
    } else if (shared_var_idx >= 0 && shared_var_idx < (int)s.vars.size()) {
        view.y_axis.name = s.vars[shared_var_idx];
    } else {
        view.y_axis.name = "X";
    }

    // Локальные буферы (по одному на серию). static, чтобы указатели жили
    // до конца кадра (PlotSeriesInput хранит сырые указатели).
    static std::vector<std::vector<float>> bufs;
    if (bufs.size() != win.members.size()) bufs.assign(win.members.size(), {});

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
                if (k < (int)bd.result.flags.size() && bd.result.flags[k] < 0) continue;
                double x;
                if (rev)
                    x = (npts > 1) ? (hi - (hi - lo) * (double)k / (double)(npts - 1)) : hi;
                else
                    x = (npts > 1) ? (lo + (hi - lo) * (double)k / (double)(npts - 1)) : lo;
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

    // Snap X к узлам первой БД этого окна (см. apply_snap_x_from_config).
    view.snap_x_to_grid = true;
    view.snap_x_n       = 0;
    if (!win.members.empty()) {
        int aidx = win.members[0];
        if (aidx >= 0 && aidx < (int)s.diagrams.size()) {
            const auto& abd = s.diagrams[aidx];
            apply_snap_x_from_config(view, abd.last_run_ok,
                                     abd.result.param_lo, abd.result.param_hi, abd.result.n_pts,
                                     abd.param_lo_text, abd.param_hi_text, abd.n_pts_text);
        }
    }

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
static bool draw_lle_curve_controls(LLEAnalysisSession& s, int idx) {
    LLECurveConfig& c = s.curves[idx];

    ImGui::SetNextItemWidth(160);
    if (InputTextStr("Label", c.label))
        c.label_is_manual = !c.label.empty();   // empty → back to auto
    ImGui::Separator();

    static const char* schemes[] = { "Euler", "Euler-Cromer", "Explicit Midpoint", "RK4", "DOPRI78", "CD" };
    ImGui::SetNextItemWidth(160);
    if (ImGui::BeginCombo("Scheme", c.scheme.c_str())) {
        for (auto m : schemes)
            if (ImGui::Selectable(m, c.scheme == m)) c.scheme = m;
        if (!s.custom_schemes.empty()) ImGui::Separator();
        for (const auto& cs : s.custom_schemes)
            if (ImGui::Selectable((cs.name + " (custom)").c_str(), c.scheme == cs.name))
                c.scheme = cs.name;
        ImGui::EndCombo();
    }
    ImGui::Separator();

    // Sweep target: параметры + разделитель + переменные (IC). См. BD.
    if (!s.params.empty() || !s.vars.empty()) {
        if (c.param_index < 0 || c.param_index >= (int)s.params.size())
            c.param_index = 0;
        if (c.var_sweep_index < 0 || c.var_sweep_index >= (int)s.vars.size())
            c.var_sweep_index = 0;
        std::string preview;
        if (c.sweep_over_h)
            preview = "dt (h)";
        else if (c.sweep_over_var && !s.vars.empty())
            preview = s.vars[c.var_sweep_index] + " (IC)";
        else if (!s.params.empty())
            preview = s.params[c.param_index];
        else
            preview = "?";
        ImGui::SetNextItemWidth(160);
        if (ImGui::BeginCombo("Sweep", preview.c_str())) {
            for (int i = 0; i < (int)s.params.size(); ++i) {
                bool sel = !c.sweep_over_var && !c.sweep_over_h && c.param_index == i;
                if (ImGui::Selectable(s.params[i].c_str(), sel)) {
                    c.sweep_over_var = false;
                    c.sweep_over_h = false;
                    c.param_index = i;
                }
            }
            if (!s.params.empty() && !s.vars.empty()) ImGui::Separator();
            for (int i = 0; i < (int)s.vars.size(); ++i) {
                std::string lbl = s.vars[i] + " (IC)";
                bool sel = c.sweep_over_var && !c.sweep_over_h && c.var_sweep_index == i;
                if (ImGui::Selectable(lbl.c_str(), sel)) {
                    c.sweep_over_var = true;
                    c.sweep_over_h = false;
                    c.var_sweep_index = i;
                }
            }
            ImGui::Separator();
            if (ImGui::Selectable("dt (h)", c.sweep_over_h)) {
                c.sweep_over_h = true;
                c.sweep_over_var = false;
                if (c.mode_2d) c.sweep_over_h_2 = false;  // ровно одна ось = h
            }
            ImGui::EndCombo();
        }
    }
    else {
        ImGui::TextDisabled("No parameters/variables (select a system first)");
    }
    InputNumStr(c.sweep_over_h ? "h lo" : "Param lo", c.param_lo_text, 120);
    InputNumStr(c.sweep_over_h ? "h hi" : "Param hi", c.param_hi_text, 120);
    ImGui::Checkbox("Log scale##lle_log", &c.log_scale);
    if (c.log_scale) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
    InputNumStr("Resolution", c.n_pts_text, 120);

    draw_continuation_device_block(c, "lle");


    ImGui::Separator();
    // 2D-режим. Сетка квадратная (см. analysis_session.h:LLECurveConfig коммент)
    // — Resolution выше применяется и к X, и к Y.
    ImGui::Checkbox("2D mode (heatmap)", &c.mode_2d);
    if (c.mode_2d) {
        ImGui::Indent();
        // Sweep target для второй оси — комбо, симметрично первой.
        if (!s.params.empty() || !s.vars.empty()) {
            if (c.param_index_2 < 0 || c.param_index_2 >= (int)s.params.size())
                c.param_index_2 = 0;
            if (c.var_sweep_index_2 < 0 || c.var_sweep_index_2 >= (int)s.vars.size())
                c.var_sweep_index_2 = 0;
            std::string preview2;
            if (c.sweep_over_h_2)
                preview2 = "dt (h)";
            else if (c.sweep_over_var_2 && !s.vars.empty())
                preview2 = s.vars[c.var_sweep_index_2] + " (IC)";
            else if (!s.params.empty())
                preview2 = s.params[c.param_index_2];
            else
                preview2 = "?";
            ImGui::SetNextItemWidth(160);
            if (ImGui::BeginCombo("Sweep Y", preview2.c_str())) {
                for (int i = 0; i < (int)s.params.size(); ++i) {
                    bool sel = !c.sweep_over_var_2 && !c.sweep_over_h_2 && c.param_index_2 == i;
                    if (ImGui::Selectable(s.params[i].c_str(), sel)) {
                        c.sweep_over_var_2 = false;
                        c.sweep_over_h_2 = false;
                        c.param_index_2 = i;
                    }
                }
                if (!s.params.empty() && !s.vars.empty()) ImGui::Separator();
                for (int i = 0; i < (int)s.vars.size(); ++i) {
                    std::string lbl = s.vars[i] + " (IC)";
                    bool sel = c.sweep_over_var_2 && !c.sweep_over_h_2 && c.var_sweep_index_2 == i;
                    if (ImGui::Selectable(lbl.c_str(), sel)) {
                        c.sweep_over_var_2 = true;
                        c.sweep_over_h_2 = false;
                        c.var_sweep_index_2 = i;
                    }
                }
                if (!c.sweep_over_h) {  // ровно одна ось = h -- X уже занял её
                    ImGui::Separator();
                    if (ImGui::Selectable("dt (h)", c.sweep_over_h_2)) {
                        c.sweep_over_h_2 = true;
                        c.sweep_over_var_2 = false;
                    }
                }
                ImGui::EndCombo();
            }
        }
        InputNumStr(c.sweep_over_h_2 ? "h2 lo" : "Param2 lo", c.param_lo_2_text, 120);
        InputNumStr(c.sweep_over_h_2 ? "h2 hi" : "Param2 hi", c.param_hi_2_text, 120);
        ImGui::Checkbox("Log scale##lle_log2", &c.log_scale_2);
        if (c.log_scale_2) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
        ImGui::TextDisabled("Grid is square (Resolution applies to both axes).");
        ImGui::Unindent();
    }

    ImGui::Separator();

    // ----- Integration (collapsible) -----
    if (ImGui::CollapsingHeader("Integration##lle_int", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("h",              c.h_text,         120);
        if (c.scheme == "CD" || custom_scheme_uses_symmetry(c.scheme, s.custom_schemes))
            InputNumStr("symmetry s", c.symmetry_s,     120);
        InputNumStr("computing time", c.t_max_text,     120);
        InputNumStr("transient time", c.transient_text, 120);
        InputNumStr("max value",      c.max_value_text, 120);
    }

    // ----- LLE (Wolf/Benettin) (collapsible) -----
    if (ImGui::CollapsingHeader("LLE (Wolf/Benettin)##lle_wb", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("eps", c.eps_text, 120);
        InputNumStr("NT",  c.nt_text,  120);
        ImGui::TextDisabled("eps = initial perturbation magnitude; NT = block length\n"
                            "between renormalizations (in time units).");
    }

    // ----- Initial conditions (collapsible) -----
    if (ImGui::CollapsingHeader("Initial conditions##lle_ic", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (const auto& v : s.vars) {
            ImGui::PushID(v.c_str());
            InputNumStr(v.c_str(), c.initial_conditions[v], 120);
            ImGui::PopID();
        }
    }

    // ----- Parameters (collapsible) -----
    if (ImGui::CollapsingHeader("Parameters##lle_par", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (const auto& p : s.params) {
            ImGui::PushID(p.c_str());
            InputNumStr(p.c_str(), c.param_values[p], 120);
            ImGui::PopID();
        }
    }

    // ----- CSV output (collapsible, moved to the bottom) -----
    if (ImGui::CollapsingHeader("CSV output##lle_csv")) {
        ImGui::Checkbox("Save to file", &c.csv_save_enabled);
        InputTextStr("##lle_csv_path", c.csv_output_path);
        ImGui::TextDisabled("Path is kept even when save is off. Also writes <path>_config.csv.");
    }

    // Run-кнопка теперь живёт на уровне draw_parametric_controls.

    if (c.mode_2d) {
        if (c.last_run_2d_ok) {
            int total = (int)c.result_2d.flags.size();
            int diverged = 0;
            for (int f : c.result_2d.flags) if (f < 0) ++diverged;
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: %dx%d heatmap, lambda(min..max) = %.4g..%.4g",
                c.result_2d.n_pts, c.result_2d.n_pts,
                c.result_2d.min_val, c.result_2d.max_val);
            if (diverged) ImGui::TextDisabled("(%d/%d cells diverged)", diverged, total);
        }
        else if (!c.last_error.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error (selectable, Ctrl+C):");
            ImVec2 sz(-1.0f, ImGui::GetTextLineHeight() * 12);
            ImGui::InputTextMultiline("##lle_err",
                const_cast<char*>(c.last_error.c_str()),
                c.last_error.size() + 1,
                sz,
                ImGuiInputTextFlags_ReadOnly);
        }
    } else {
        if (c.last_run_ok) {
            int diverged = 0;
            for (int f : c.result.flags) if (f < 0) ++diverged;
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: n_pts=%d, lambda-curve computed", c.result.n_pts);
            if (diverged) ImGui::TextDisabled("(%d/%d points diverged)", diverged, c.result.n_pts);
        }
        else if (!c.last_error.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error (selectable, Ctrl+C):");
            ImVec2 sz(-1.0f, ImGui::GetTextLineHeight() * 12);
            ImGui::InputTextMultiline("##lle_err",
                const_cast<char*>(c.last_error.c_str()),
                c.last_error.size() + 1,
                sz,
                ImGuiInputTextFlags_ReadOnly);
        }
    }
    return false;  // Run-кнопка перенесена в draw_parametric_controls.
}

// Tab bar по LLE-кривым + кнопка «+» (копирует последнюю).
static void draw_lle_controls(AppModel& model, SystemLibrary& /*lib*/) {
    LLEAnalysisSession& s = model.lle_session;

    int active_now = -1;
    int run_idx = -1;
    int to_remove = -1;
    if (ImGui::BeginTabBar("##lle_tabs",
                           ImGuiTabBarFlags_Reorderable |
                           ImGuiTabBarFlags_AutoSelectNewTabs |
                           ImGuiTabBarFlags_FittingPolicyScroll)) {
        for (int i = 0; i < (int)s.curves.size(); ++i) {
            LLECurveConfig& c = s.curves[i];
            ImGui::PushID(i);
            bool open = true;
            std::string tab_id = c.label + "###lle_tab_" + std::to_string(i);
            bool can_close = !(s.in_flight && s.running_curve_index == i);
            if (ImGui::BeginTabItem(tab_id.c_str(), can_close ? &open : nullptr)) {
                active_now = i;
                if (draw_lle_curve_controls(s, i)) run_idx = i;
                ImGui::EndTabItem();
            }
            if (!open) to_remove = i;
            ImGui::PopID();
        }
        if (!s.in_flight) {
            if (ImGui::TabItemButton("+",
                                     ImGuiTabItemFlags_Trailing |
                                     ImGuiTabItemFlags_NoTooltip)) {
                s.add_curve();
            }
        }
        ImGui::EndTabBar();
    }
    if (active_now >= 0) s.active_curve_index = active_now;
    if (to_remove >= 0) model.remove_lle_curve(to_remove);

    // Run + Ctrl+R — в draw_parametric_controls (общая кнопка слева от Run all).
    if (run_idx >= 0 && run_idx < (int)s.curves.size()) {
        if (!model.parametric_engine) model.parametric_engine = std::make_unique<ParametricEngine>();
        s.run_async(*model.parametric_engine, run_idx);
    }
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

            // Подписи осей по реальным selected-полям свипа.
            auto axis_name_for = [&](bool sweep_h, bool sweep_var, int p_idx, int v_idx) -> std::string {
                if (sweep_h) return "h";
                if (sweep_var) {
                    return (v_idx >= 0 && v_idx < (int)s.vars.size()) ? (s.vars[v_idx] + " (IC)") : "x";
                }
                return (p_idx >= 0 && p_idx < (int)s.params.size()) ? s.params[p_idx] : "param";
            };
            heatmap.x_axis.name = axis_name_for(cact.sweep_over_h,   cact.sweep_over_var,   cact.param_index,   cact.var_sweep_index);
            heatmap.y_axis.name = axis_name_for(cact.sweep_over_h_2, cact.sweep_over_var_2, cact.param_index_2, cact.var_sweep_index_2);
            heatmap.x_axis.log_scale = cact.log_scale;
            heatmap.y_axis.log_scale = cact.log_scale_2;

            bool fit = cact.fit_request_2d;
            if (fit) cact.fit_request_2d = false;

            const bool busy = s.in_flight && s.is_2d_run && idx == s.running_curve_index;
            heatmap.popup_extras = [&cact, &cb, busy]() {
                if (ImGui::MenuItem("Export data...", nullptr, false, !busy)) {
                    if (cb.pick_save_file_csv) {
                        std::string path = cb.pick_save_file_csv();
                        if (!path.empty())
                            data_export::export_lle2d(cact.result_2d, path);
                    }
                }
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


    // Подпись X + X-fit диапазон — см. configure_sweep_x_axis.
    {
        std::vector<SweepAxisMember> sweep_members;
        for (int idx : win.members) {
            if (idx < 0 || idx >= (int)s.curves.size()) continue;
            const auto& c = s.curves[idx];
            if (!c.last_run_ok) continue;
            SweepAxisMember m;
            m.kind      = c.sweep_over_h ? 2 : (c.sweep_over_var ? 1 : 0);
            m.index     = c.sweep_over_var ? c.var_sweep_index : c.param_index;
            m.log_scale = c.log_scale;
            m.lo = c.result.param_lo;
            m.hi = c.result.param_hi;
            // Движок ещё не заполнил снапшот диапазона — берём текстовые поля.
            if (m.hi == m.lo) {
                m.lo = parse_ratio_or(c.param_lo_text, 0.0);
                m.hi = parse_ratio_or(c.param_hi_text, 1.0);
            }
            sweep_members.push_back(m);
        }
        configure_sweep_x_axis(view, sweep_members, s.params, s.vars);
    }
    view.y_axis.name = "lambda";

    static std::vector<std::vector<float>> bufs;
    if (bufs.size() != win.members.size()) bufs.assign(win.members.size(), {});

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
                if (k < (int)c.result.flags.size() && c.result.flags[k] < 0) continue;
                double t = (npts > 1) ? (double)k / (double)(npts - 1) : 0.0;
                double x = rev ? (hi - (hi - lo) * t) : (lo + (hi - lo) * t);
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

    // Snap X к узлам первой кривой этого окна (см. apply_snap_x_from_config).
    view.snap_x_to_grid = true;
    view.snap_x_n       = 0;
    if (!win.members.empty()) {
        int aidx = win.members[0];
        if (aidx >= 0 && aidx < (int)s.curves.size()) {
            const auto& ac = s.curves[aidx];
            apply_snap_x_from_config(view, ac.last_run_ok,
                                     ac.result.param_lo, ac.result.param_hi, ac.result.n_pts,
                                     ac.param_lo_text, ac.param_hi_text, ac.n_pts_text);
        }
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    view.render(renderer, origin, avail, /*owner_id*/ 0xBE11E5, data_gen,
                series_in, init_vis, glob_vis, any_fit);
}

// ============================================================
// LS: контролы (per-curve в табе) + line-plot λ_k(param), N экспонент
// на один спектр-«прогон». UX зеркало LLE.
// ============================================================

static bool draw_ls_curve_controls(LyapunovSpectrumAnalysisSession& s, int idx) {
    LSCurveConfig& c = s.curves[idx];

    ImGui::SetNextItemWidth(160);
    if (InputTextStr("Label", c.label))
        c.label_is_manual = !c.label.empty();   // empty → back to auto
    ImGui::Separator();

    static const char* schemes[] = { "Euler", "Euler-Cromer", "Explicit Midpoint", "RK4", "DOPRI78", "CD" };
    ImGui::SetNextItemWidth(160);
    if (ImGui::BeginCombo("Scheme", c.scheme.c_str())) {
        for (auto m : schemes)
            if (ImGui::Selectable(m, c.scheme == m)) c.scheme = m;
        if (!s.custom_schemes.empty()) ImGui::Separator();
        for (const auto& cs : s.custom_schemes)
            if (ImGui::Selectable((cs.name + " (custom)").c_str(), c.scheme == cs.name))
                c.scheme = cs.name;
        ImGui::EndCombo();
    }
    ImGui::Separator();

    // Sweep target: параметры + разделитель + переменные (IC). См. BD.
    if (!s.params.empty() || !s.vars.empty()) {
        if (c.param_index < 0 || c.param_index >= (int)s.params.size())
            c.param_index = 0;
        if (c.var_sweep_index < 0 || c.var_sweep_index >= (int)s.vars.size())
            c.var_sweep_index = 0;
        std::string preview;
        if (c.sweep_over_h)
            preview = "dt (h)";
        else if (c.sweep_over_var && !s.vars.empty())
            preview = s.vars[c.var_sweep_index] + " (IC)";
        else if (!s.params.empty())
            preview = s.params[c.param_index];
        else
            preview = "?";
        ImGui::SetNextItemWidth(160);
        if (ImGui::BeginCombo("Sweep", preview.c_str())) {
            for (int i = 0; i < (int)s.params.size(); ++i) {
                bool sel = !c.sweep_over_var && !c.sweep_over_h && c.param_index == i;
                if (ImGui::Selectable(s.params[i].c_str(), sel)) {
                    c.sweep_over_var = false;
                    c.sweep_over_h = false;
                    c.param_index = i;
                }
            }
            if (!s.params.empty() && !s.vars.empty()) ImGui::Separator();
            for (int i = 0; i < (int)s.vars.size(); ++i) {
                std::string lbl = s.vars[i] + " (IC)";
                bool sel = c.sweep_over_var && !c.sweep_over_h && c.var_sweep_index == i;
                if (ImGui::Selectable(lbl.c_str(), sel)) {
                    c.sweep_over_var = true;
                    c.sweep_over_h = false;
                    c.var_sweep_index = i;
                }
            }
            ImGui::Separator();
            if (ImGui::Selectable("dt (h)", c.sweep_over_h)) {
                c.sweep_over_h = true;
                c.sweep_over_var = false;
                if (c.mode_2d) c.sweep_over_h_2 = false;  // ровно одна ось = h
            }
            ImGui::EndCombo();
        }
    }
    else {
        ImGui::TextDisabled("No parameters/variables (select a system first)");
    }
    InputNumStr(c.sweep_over_h ? "h lo" : "Param lo", c.param_lo_text, 120);
    InputNumStr(c.sweep_over_h ? "h hi" : "Param hi", c.param_hi_text, 120);
    ImGui::Checkbox("Log scale##ls_log", &c.log_scale);
    if (c.log_scale) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
    InputNumStr("Resolution", c.n_pts_text, 120);

    draw_continuation_device_block(c, "ls");

    ImGui::Separator();
    // 2D-режим. Сетка квадратная (см. LLE 2D — то же ограничение getValueByIdx).
    ImGui::Checkbox("2D mode (heatmap of one exponent)", &c.mode_2d);
    if (c.mode_2d) {
        ImGui::Indent();
        if (!s.params.empty() || !s.vars.empty()) {
            if (c.param_index_2 < 0 || c.param_index_2 >= (int)s.params.size())
                c.param_index_2 = 0;
            if (c.var_sweep_index_2 < 0 || c.var_sweep_index_2 >= (int)s.vars.size())
                c.var_sweep_index_2 = 0;
            std::string preview2;
            if (c.sweep_over_h_2)
                preview2 = "dt (h)";
            else if (c.sweep_over_var_2 && !s.vars.empty())
                preview2 = s.vars[c.var_sweep_index_2] + " (IC)";
            else if (!s.params.empty())
                preview2 = s.params[c.param_index_2];
            else
                preview2 = "?";
            ImGui::SetNextItemWidth(160);
            if (ImGui::BeginCombo("Sweep Y", preview2.c_str())) {
                for (int i = 0; i < (int)s.params.size(); ++i) {
                    bool sel = !c.sweep_over_var_2 && !c.sweep_over_h_2 && c.param_index_2 == i;
                    if (ImGui::Selectable(s.params[i].c_str(), sel)) {
                        c.sweep_over_var_2 = false;
                        c.sweep_over_h_2 = false;
                        c.param_index_2 = i;
                    }
                }
                if (!s.params.empty() && !s.vars.empty()) ImGui::Separator();
                for (int i = 0; i < (int)s.vars.size(); ++i) {
                    std::string lbl = s.vars[i] + " (IC)";
                    bool sel = c.sweep_over_var_2 && !c.sweep_over_h_2 && c.var_sweep_index_2 == i;
                    if (ImGui::Selectable(lbl.c_str(), sel)) {
                        c.sweep_over_var_2 = true;
                        c.sweep_over_h_2 = false;
                        c.var_sweep_index_2 = i;
                    }
                }
                if (!c.sweep_over_h) {  // ровно одна ось = h -- X уже занял её
                    ImGui::Separator();
                    if (ImGui::Selectable("dt (h)", c.sweep_over_h_2)) {
                        c.sweep_over_h_2 = true;
                        c.sweep_over_var_2 = false;
                    }
                }
                ImGui::EndCombo();
            }
        }
        InputNumStr(c.sweep_over_h_2 ? "h2 lo" : "Param2 lo", c.param_lo_2_text, 120);
        InputNumStr(c.sweep_over_h_2 ? "h2 hi" : "Param2 hi", c.param_hi_2_text, 120);
        ImGui::Checkbox("Log scale##ls_log2", &c.log_scale_2);
        if (c.log_scale_2) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
        ImGui::TextDisabled("Grid is square (Resolution applies to both axes).\nAll N exponents computed; switch in plot window.");
        ImGui::Unindent();
    }

    ImGui::Separator();

    // ----- Integration (collapsible) -----
    if (ImGui::CollapsingHeader("Integration##ls_int", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("h",              c.h_text,         120);
        if (c.scheme == "CD" || custom_scheme_uses_symmetry(c.scheme, s.custom_schemes))
            InputNumStr("symmetry s", c.symmetry_s,     120);
        InputNumStr("computing time", c.t_max_text,     120);
        InputNumStr("transient time", c.transient_text, 120);
        InputNumStr("max value",      c.max_value_text, 120);
    }

    // ----- LS (Wolf/Benettin + Gram-Schmidt) (collapsible) -----
    if (ImGui::CollapsingHeader("LS (Wolf/Benettin + Gram-Schmidt)##ls_wbgs", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("eps", c.eps_text, 120);
        InputNumStr("NT",  c.nt_text,  120);
        ImGui::TextDisabled("eps = initial perturbation magnitude; NT = block length\n"
                            "between renormalizations (in time units).");
    }

    // ----- Initial conditions (collapsible) -----
    if (ImGui::CollapsingHeader("Initial conditions##ls_ic", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (const auto& v : s.vars) {
            ImGui::PushID(v.c_str());
            InputNumStr(v.c_str(), c.initial_conditions[v], 120);
            ImGui::PopID();
        }
    }

    // ----- Parameters (collapsible) -----
    if (ImGui::CollapsingHeader("Parameters##ls_par", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (const auto& p : s.params) {
            ImGui::PushID(p.c_str());
            InputNumStr(p.c_str(), c.param_values[p], 120);
            ImGui::PopID();
        }
    }

    // ----- CSV output (collapsible, moved to the bottom) -----
    if (ImGui::CollapsingHeader("CSV output##ls_csv")) {
        ImGui::Checkbox("Save to file", &c.csv_save_enabled);
        InputTextStr("##ls_csv_path", c.csv_output_path);
        ImGui::TextDisabled("Path is kept even when save is off. Also writes <path>_config.csv.");
    }

    // Run-кнопка теперь живёт на уровне draw_parametric_controls.

    if (c.mode_2d) {
        if (c.last_run_2d_ok) {
            int total = (int)c.result_2d.flags.size();
            int diverged = 0;
            for (int f : c.result_2d.flags) if (f < 0) ++diverged;
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: %dx%d heatmap, %d exponents",
                c.result_2d.n_pts, c.result_2d.n_pts, c.result_2d.n_exponents);
            if (diverged) ImGui::TextDisabled("(%d/%d cells diverged)", diverged, total);
        }
        else if (!c.last_error.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error (selectable, Ctrl+C):");
            ImVec2 sz(-1.0f, ImGui::GetTextLineHeight() * 12);
            ImGui::InputTextMultiline("##ls_err_2d",
                const_cast<char*>(c.last_error.c_str()),
                c.last_error.size() + 1,
                sz,
                ImGuiInputTextFlags_ReadOnly);
        }
    } else {
        if (c.last_run_ok) {
            int diverged = 0;
            for (int f : c.result.flags) if (f < 0) ++diverged;
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: n_pts=%d, n_exponents=%d", c.result.n_pts, c.result.n_exponents);
            if (diverged) ImGui::TextDisabled("(%d/%d points diverged)", diverged, c.result.n_pts);
        }
        else if (!c.last_error.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error (selectable, Ctrl+C):");
            ImVec2 sz(-1.0f, ImGui::GetTextLineHeight() * 12);
            ImGui::InputTextMultiline("##ls_err",
                const_cast<char*>(c.last_error.c_str()),
                c.last_error.size() + 1,
                sz,
                ImGuiInputTextFlags_ReadOnly);
        }
    }
    return false;  // Run-кнопка перенесена в draw_parametric_controls.
}

static void draw_ls_controls(AppModel& model, SystemLibrary& /*lib*/) {
    LyapunovSpectrumAnalysisSession& s = model.ls_session;

    int active_now = -1;
    int run_idx = -1;
    int to_remove = -1;
    if (ImGui::BeginTabBar("##ls_tabs",
                           ImGuiTabBarFlags_Reorderable |
                           ImGuiTabBarFlags_AutoSelectNewTabs |
                           ImGuiTabBarFlags_FittingPolicyScroll)) {
        for (int i = 0; i < (int)s.curves.size(); ++i) {
            LSCurveConfig& c = s.curves[i];
            ImGui::PushID(i);
            bool open = true;
            std::string tab_id = c.label + "###ls_tab_" + std::to_string(i);
            bool can_close = !(s.in_flight && s.running_curve_index == i);
            if (ImGui::BeginTabItem(tab_id.c_str(), can_close ? &open : nullptr)) {
                active_now = i;
                if (draw_ls_curve_controls(s, i)) run_idx = i;
                ImGui::EndTabItem();
            }
            if (!open) to_remove = i;
            ImGui::PopID();
        }
        if (!s.in_flight) {
            if (ImGui::TabItemButton("+",
                                     ImGuiTabItemFlags_Trailing |
                                     ImGuiTabItemFlags_NoTooltip)) {
                s.add_curve();
            }
        }
        ImGui::EndTabBar();
    }
    if (active_now >= 0) s.active_curve_index = active_now;
    if (to_remove >= 0) model.remove_ls_curve(to_remove);

    // Run + Ctrl+R — в draw_parametric_controls (общая кнопка слева от Run all).
    if (run_idx >= 0 && run_idx < (int)s.curves.size()) {
        if (!model.parametric_engine) model.parametric_engine = std::make_unique<ParametricEngine>();
        s.run_async(*model.parametric_engine, run_idx);
    }
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

            // Подписи осей по реальным selected-полям свипа.
            auto ax_name = [&](bool sweep_h, bool sweep_var, int p_idx, int v_idx) -> std::string {
                if (sweep_h) return "h";
                if (sweep_var)
                    return (v_idx >= 0 && v_idx < (int)s.vars.size()) ? (s.vars[v_idx] + " (IC)") : "x";
                return (p_idx >= 0 && p_idx < (int)s.params.size()) ? s.params[p_idx] : "param";
            };
            heatmap_ls.x_axis.name = ax_name(cact.sweep_over_h,   cact.sweep_over_var,   cact.param_index,   cact.var_sweep_index);
            heatmap_ls.y_axis.name = ax_name(cact.sweep_over_h_2, cact.sweep_over_var_2, cact.param_index_2, cact.var_sweep_index_2);
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
                if (ImGui::MenuItem("Export data...", nullptr, false, !busy)) {
                    if (cb.pick_save_file_csv) {
                        std::string path = cb.pick_save_file_csv();
                        if (!path.empty())
                            data_export::export_ls2d(cact.result_2d, path);
                    }
                }
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


    // Подпись X + X-fit диапазон — см. configure_sweep_x_axis.
    {
        std::vector<SweepAxisMember> sweep_members;
        for (int idx : win.members) {
            if (idx < 0 || idx >= (int)s.curves.size()) continue;
            const auto& c = s.curves[idx];
            if (!c.last_run_ok) continue;
            SweepAxisMember m;
            m.kind      = c.sweep_over_h ? 2 : (c.sweep_over_var ? 1 : 0);
            m.index     = c.sweep_over_var ? c.var_sweep_index : c.param_index;
            m.log_scale = c.log_scale;
            m.lo = c.result.param_lo;
            m.hi = c.result.param_hi;
            if (m.hi == m.lo) {
                m.lo = parse_ratio_or(c.param_lo_text, 0.0);
                m.hi = parse_ratio_or(c.param_hi_text, 1.0);
            }
            sweep_members.push_back(m);
        }
        configure_sweep_x_axis(view, sweep_members, s.params, s.vars);
    }

    // Подсчитываем общее число серий этого окна — sum(n_exponents per member).
    size_t total_series = 0;
    for (int idx : win.members) {
        if (idx < 0 || idx >= (int)s.curves.size()) continue;
        const auto& c = s.curves[idx];
        int N = c.result.n_exponents > 0 ? c.result.n_exponents : (int)s.vars.size();
        total_series += (size_t)N;
    }

    static std::vector<std::vector<float>> bufs;
    if (bufs.size() != total_series) bufs.assign(total_series, {});

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
                    if (k < (int)c.result.flags.size() && c.result.flags[k] < 0) continue;
                    if (k >= (int)c.result.spectrum.size()) continue;
                    const auto& row = c.result.spectrum[k];
                    if (j >= (int)row.size()) continue;
                    double t = (npts > 1) ? (double)k / (double)(npts - 1) : 0.0;
                    double x = rev ? (hi - (hi - lo) * t) : (lo + (hi - lo) * t);
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

    // Snap X к узлам первой LS-кривой этого окна (см. apply_snap_x_from_config).
    view.snap_x_to_grid = true;
    view.snap_x_n       = 0;
    if (!win.members.empty()) {
        int aidx = win.members[0];
        if (aidx >= 0 && aidx < (int)s.curves.size()) {
            const auto& ac = s.curves[aidx];
            apply_snap_x_from_config(view, ac.last_run_ok,
                                     ac.result.param_lo, ac.result.param_hi, ac.result.n_pts,
                                     ac.param_lo_text, ac.param_hi_text, ac.n_pts_text);
        }
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    view.render(renderer, origin, avail, /*owner_id*/ 0x15A1E0, data_gen,
                series_in, init_vis, glob_vis, any_fit);
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
        if (ImGui::Begin(title.c_str(), &open)) {
            ImGui::PushID(win.id);
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
        if (!open) to_remove = i;
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

static bool draw_dft1d_diagram_controls(Dft1DAnalysisSession& s, int idx) {
    Dft1DConfig& c = s.configs[idx];

    ImGui::SetNextItemWidth(160);
    if (InputTextStr("Label", c.label))
        c.label_is_manual = !c.label.empty();   // empty → back to auto
    ImGui::Separator();

    // ----- Scheme (built-in + custom) -----
    static const char* schemes[] = { "Euler", "Euler-Cromer", "Explicit Midpoint", "RK4", "DOPRI78", "CD" };
    ImGui::SetNextItemWidth(160);
    if (ImGui::BeginCombo("Scheme", c.scheme.c_str())) {
        for (auto m : schemes)
            if (ImGui::Selectable(m, c.scheme == m)) c.scheme = m;
        if (!s.custom_schemes.empty()) ImGui::Separator();
        for (const auto& cs : s.custom_schemes)
            if (ImGui::Selectable((cs.name + " (custom)").c_str(), c.scheme == cs.name))
                c.scheme = cs.name;
        ImGui::EndCombo();
    }
    ImGui::Separator();

    // ----- Sweep target (parameter ИЛИ initial condition), см. draw_diagram_controls -----
    if (!s.params.empty() || !s.vars.empty()) {
        if (c.param_index < 0 || c.param_index >= (int)s.params.size())
            c.param_index = 0;
        if (c.var_sweep_index < 0 || c.var_sweep_index >= (int)s.vars.size())
            c.var_sweep_index = 0;

        std::string preview;
        if (c.sweep_over_var && !s.vars.empty())
            preview = s.vars[c.var_sweep_index] + " (IC)";
        else if (!s.params.empty())
            preview = s.params[c.param_index];
        else
            preview = "?";

        ImGui::SetNextItemWidth(160);
        if (ImGui::BeginCombo("Sweep", preview.c_str())) {
            for (int i = 0; i < (int)s.params.size(); ++i) {
                bool sel = !c.sweep_over_var && c.param_index == i;
                if (ImGui::Selectable(s.params[i].c_str(), sel)) {
                    c.sweep_over_var = false;
                    c.param_index = i;
                }
            }
            if (!s.params.empty() && !s.vars.empty()) ImGui::Separator();
            for (int i = 0; i < (int)s.vars.size(); ++i) {
                std::string lbl = s.vars[i] + " (IC)";
                bool sel = c.sweep_over_var && c.var_sweep_index == i;
                if (ImGui::Selectable(lbl.c_str(), sel)) {
                    c.sweep_over_var = true;
                    c.var_sweep_index = i;
                }
            }
            ImGui::EndCombo();
        }
    } else {
        ImGui::TextDisabled("No parameters/variables (select a system first)");
    }
    InputNumStr("Param lo", c.param_lo_text, 120);
    InputNumStr("Param hi", c.param_hi_text, 120);

    // Continuation — требует param-sweep (см. run_dft1d_continuation); UI не
    // блокирует sweep_over_var=true+continuation=true явно (как Bifurcation
    // не блокирует до Run), engine вернёт понятную ошибку в last_error.
    {
        bool cont = c.continuation;
        if (ImGui::Checkbox("Continuation", &cont)) c.continuation = cont;
        if (c.continuation) {
            ImGui::SameLine();
            int dir = c.continuation_reverse ? 1 : 0;
            ImGui::RadioButton("forward",  &dir, 0); ImGui::SameLine();
            ImGui::RadioButton("backward", &dir, 1);
            c.continuation_reverse = (dir == 1);
        }
    }
    ImGui::Separator();

    // ----- Variable + Resolution X -----
    draw_writable_var_combo(s.vars, c.writable_var, "Writable var##dft_wv");
    InputNumStr("Resolution X", c.n_pts_text, 120);
    ImGui::Separator();

    // ----- Resolution Y / Frequency range (обязательные поля для rangesFreq;
    // без auto-режима — частотный диапазон всегда физически осмыслен только
    // когда задан явно, в отличие от colored_1d's Y auto-range). -----
    if (ImGui::CollapsingHeader("Resolution Y##dft_freq", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("Resolution Y", c.n_freq_text, 120);
        InputNumStr("Freq lo", c.freq_lo_text, 120);
        InputNumStr("Freq hi", c.freq_hi_text, 120);
        ImGui::Checkbox("Log scale (Y)##dft_freq_log", &c.freq_log_scale);
        if (c.freq_log_scale) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
        static const char* windows[] = { "None", "Hanning", "Hamming" };
        ImGui::SetNextItemWidth(160);
        ImGui::Combo("Window", &c.window_type, windows, IM_ARRAYSIZE(windows));
    }

    // ----- Display mode + normalize -----
    if (ImGui::CollapsingHeader("Display##dft_disp", ImGuiTreeNodeFlags_DefaultOpen)) {
        static const char* modes[] = { "Power spectrum", "Amplitude", "Phase" };
        ImGui::SetNextItemWidth(160);
        ImGui::Combo("Mode", &c.display_mode, modes, IM_ARRAYSIZE(modes));
        ImGui::Checkbox("Normalize?", &c.normalize);
    }

    // ----- Integration (collapsible) -----
    if (ImGui::CollapsingHeader("Integration##dft_int", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("h",              c.h_text,           120);
        if (c.scheme == "CD" || custom_scheme_uses_symmetry(c.scheme, s.custom_schemes))
            InputNumStr("symmetry s", c.symmetry_s,       120);
        InputNumStr("computing time", c.t_max_text,       120);
        InputNumStr("transient time", c.transient_text,   120);
        InputNumStr("decimator",      c.pre_scaller_text, 120);
        InputNumStr("max value",      c.max_value_text,   120);
    }

    // ----- Initial conditions (collapsible) -----
    if (ImGui::CollapsingHeader("Initial conditions##dft_ic", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (const auto& v : s.vars) {
            ImGui::PushID(v.c_str());
            InputNumStr(v.c_str(), c.initial_conditions[v], 120);
            ImGui::PopID();
        }
    }

    // ----- Parameters (collapsible) -----
    if (ImGui::CollapsingHeader("Parameters##dft_par", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (const auto& p : s.params) {
            ImGui::PushID(p.c_str());
            InputNumStr(p.c_str(), c.param_values[p], 120);
            ImGui::PopID();
        }
    }

    // ----- CSV output (collapsible, moved to the bottom) -----
    if (ImGui::CollapsingHeader("CSV output##dft_csv")) {
        ImGui::Checkbox("Save to file", &c.csv_save_enabled);
        InputTextStr("##dft_csv_path", c.csv_output_path);
        ImGui::TextDisabled("Writes <path>_config.csv, _AkCOS.csv, _BkSIN.csv.");
    }

    if (c.last_run_ok) {
        int diverged = 0;
        for (int f : c.result.flags) if (f <= 0) ++diverged;
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
            "OK: n_pts=%d, n_freq=%d", c.result.n_pts, c.result.n_freq);
        if (diverged) ImGui::TextDisabled("(%d/%d points diverged)", diverged, c.result.n_pts);
    } else if (!c.last_error.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error (selectable, Ctrl+C):");
        ImVec2 sz(-1.0f, ImGui::GetTextLineHeight() * 12);
        ImGui::InputTextMultiline("##dft_err",
            const_cast<char*>(c.last_error.c_str()),
            c.last_error.size() + 1,
            sz,
            ImGuiInputTextFlags_ReadOnly);
    }
    return false;  // Run-кнопка живёт в draw_dft1d_controls, как у Bifurcation/Basins.
}

static void draw_dft1d_controls(AppModel& model, SystemLibrary& lib) {
    Dft1DAnalysisSession& s = model.dft1d_session;

    ImGui::Text("1D DFT");
    ImGui::TextDisabled("Parametric discrete Fourier transform via NVRTC + DFT_custom.");


    // ----- Run / Run all... -----
    {
        bool no_cfg = s.configs.empty();
        bool do_run = false;
        if (s.in_flight) {
            ImGui::BeginDisabled();
            ImGui::Button("Running...", ImVec2(160, 0));
            ImGui::EndDisabled();
        } else {
            if (no_cfg) ImGui::BeginDisabled();
            do_run = ImGui::Button("Run (Ctrl+R)", ImVec2(160, 0));
            if (no_cfg) ImGui::EndDisabled();
        }
        if (!s.in_flight && !no_cfg && ImGui::GetIO().KeyCtrl && !ImGui::GetIO().KeyShift &&
            ImGui::IsKeyPressed(ImGuiKey_R, false)) {
            do_run = true;
        }
        if (do_run) {
            if (!model.parametric_engine)
                model.parametric_engine = std::make_unique<ParametricEngine>();
            s.run_async(*model.parametric_engine, s.active_config_index);
        }

        static std::vector<bool> picks;
        if (picks.size() != s.configs.size()) picks.assign(s.configs.size(), true);
        auto run_all_marked = [&]() {
            for (size_t i = 0; i < picks.size(); ++i)
                if (picks[i])
                    model.dft1d_queue.push_back({(int)i});
            model.start_next_in_dft1d_queue();
        };
        if (!s.in_flight && !no_cfg && ImGui::GetIO().KeyCtrl && ImGui::GetIO().KeyShift &&
            ImGui::IsKeyPressed(ImGuiKey_R, false))
            run_all_marked();

        ImGui::SameLine();
        const bool block_run_all = s.in_flight || no_cfg;
        if (block_run_all) ImGui::BeginDisabled();
        if (ImGui::Button("Run all... (Ctrl+Shift+R)"))
            ImGui::OpenPopup("##run_all_dft1d");
        if (block_run_all) ImGui::EndDisabled();
        if (!model.dft1d_queue.empty()) {
            ImGui::SameLine();
            ImGui::TextDisabled("(%zu queued)", model.dft1d_queue.size());
        }
        if (ImGui::BeginPopup("##run_all_dft1d")) {
            ImGui::TextDisabled("Sequential (one CUDA context).");
            for (size_t i = 0; i < picks.size(); ++i) {
                bool v = picks[i];
                std::string lbl = s.configs[i].label + "###pdft1d_" + std::to_string(i);
                if (ImGui::Checkbox(lbl.c_str(), &v)) picks[i] = v;
            }
            ImGui::Separator();
            if (ImGui::Button("All"))  { for (auto&& b : picks) b = true;  }
            ImGui::SameLine();
            if (ImGui::Button("None")) { for (auto&& b : picks) b = false; }
            ImGui::SameLine();
            if (ImGui::Button("Run")) {
                run_all_marked();
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }
    ImGui::Separator();

    // Tab bar: одна вкладка на config + "+" для add. Зеркалит draw_basins_controls.
    int active_now = -1;
    int to_remove = -1;
    if (ImGui::BeginTabBar("##dft1d_tabs",
                           ImGuiTabBarFlags_Reorderable |
                           ImGuiTabBarFlags_AutoSelectNewTabs |
                           ImGuiTabBarFlags_FittingPolicyScroll)) {
        for (int i = 0; i < (int)s.configs.size(); ++i) {
            Dft1DConfig& c = s.configs[i];
            ImGui::PushID(i);
            bool open = true;
            std::string tab_id = c.label + "###dft1d_tab_" + std::to_string(i);
            bool can_close = !(s.in_flight && s.running_config_index == i);
            if (ImGui::BeginTabItem(tab_id.c_str(), can_close ? &open : nullptr)) {
                active_now = i;
                ImGui::EndTabItem();
            }
            if (!open) to_remove = i;
            ImGui::PopID();
        }
        if (!s.in_flight) {
            if (ImGui::TabItemButton("+",
                                     ImGuiTabItemFlags_Trailing |
                                     ImGuiTabItemFlags_NoTooltip)) {
                s.add_config();
            }
        }
        ImGui::EndTabBar();
    }
    if (active_now >= 0) s.active_config_index = active_now;
    if (to_remove >= 0) model.remove_dft1d_config(to_remove);

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
            c.display_cache.assign(plane_size, 999.0);
            double vmin =  std::numeric_limits<double>::infinity();
            double vmax = -std::numeric_limits<double>::infinity();
            std::vector<double> col((size_t)nfreq);
            for (int pt = 0; pt < npts; ++pt) {
                bool diverged = pt >= (int)c.result.flags.size() || c.result.flags[pt] <= 0;
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
                                        c.sweep_over_var, c.var_sweep_index);
        hc.y_axis.name = "Frequency";
        hc.y_axis.log_scale = c.freq_log_scale;

        bool fit = c.fit_request;
        if (fit) c.fit_request = false;

        const bool busy = s.in_flight && idx == s.running_config_index;
        hc.popup_extras = [&c, &cb, busy]() {
            if (ImGui::MenuItem("Export data...", nullptr, false, !busy)) {
                if (cb.pick_save_file_csv) {
                    std::string path = cb.pick_save_file_csv();
                    if (!path.empty())
                        data_export::export_dft1d(c.result, path);
                }
            }
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

static void draw_basins_controls(AppModel& model, SystemLibrary& lib) {
    BasinsAnalysisSession& s = model.basins_session;

    ImGui::Text("Basins of attraction");
    ImGui::TextDisabled("DBSCAN clustering in (avgPeak, avgInterval) plane.");

    // ----- Run / Run all... (moved up to sit right under the system picker,
    // above the tab bar — these drive the currently active config so they
    // stay accessible without scrolling past every section). -----
    {
        bool no_cfg = s.configs.empty();
        bool do_run = false;
        if (s.in_flight) {
            ImGui::BeginDisabled();
            ImGui::Button("Running...", ImVec2(160, 0));
            ImGui::EndDisabled();
        } else {
            if (no_cfg) ImGui::BeginDisabled();
            do_run = ImGui::Button("Run (Ctrl+R)", ImVec2(160, 0));
            if (no_cfg) ImGui::EndDisabled();
        }
        if (!s.in_flight && !no_cfg && ImGui::GetIO().KeyCtrl && !ImGui::GetIO().KeyShift &&
            ImGui::IsKeyPressed(ImGuiKey_R, false)) {
            do_run = true;
        }
        if (do_run) {
            if (!model.parametric_engine)
                model.parametric_engine = std::make_unique<ParametricEngine>();
            s.run_async(*model.parametric_engine, s.active_config_index);
        }

        // Batch "Run all..." across basin configs. Pushes selected indices
        // into model.basins_queue; draw_gui ticks the queue after polls.
        // picks — вне if(BeginPopup): Ctrl+Shift+R должен пушить те же
        // отметки, даже если попап ни разу не открывали (см. Parametric).
        static std::vector<bool> picks;
        if (picks.size() != s.configs.size()) picks.assign(s.configs.size(), true);
        auto run_all_marked = [&]() {
            for (size_t i = 0; i < picks.size(); ++i)
                if (picks[i])
                    model.basins_queue.push_back({(int)i});
            model.start_next_in_basins_queue();
        };
        if (!s.in_flight && !no_cfg && ImGui::GetIO().KeyCtrl && ImGui::GetIO().KeyShift &&
            ImGui::IsKeyPressed(ImGuiKey_R, false))
            run_all_marked();

        ImGui::SameLine();
        const bool block_run_all = s.in_flight || no_cfg;
        if (block_run_all) ImGui::BeginDisabled();
        if (ImGui::Button("Run all... (Ctrl+Shift+R)"))
            ImGui::OpenPopup("##run_all_basins");
        if (block_run_all) ImGui::EndDisabled();
        if (!model.basins_queue.empty()) {
            ImGui::SameLine();
            ImGui::TextDisabled("(%zu queued)", model.basins_queue.size());
        }
        if (ImGui::BeginPopup("##run_all_basins")) {
            ImGui::TextDisabled("Sequential (one CUDA context).");
            for (size_t i = 0; i < picks.size(); ++i) {
                bool v = picks[i];
                std::string lbl = s.configs[i].label + "###pbasins_" + std::to_string(i);
                if (ImGui::Checkbox(lbl.c_str(), &v)) picks[i] = v;
            }

            ImGui::Separator();
            if (ImGui::Button("All"))  { for (auto&& b : picks) b = true;  }
            ImGui::SameLine();
            if (ImGui::Button("None")) { for (auto&& b : picks) b = false; }
            ImGui::SameLine();
            if (ImGui::Button("Run")) {
                run_all_marked();
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }
    ImGui::Separator();

    // Tab bar: одна вкладка на Basins-config + "+" для add. Зеркалит
    // draw_bifurcation_controls. Активная вкладка хранится в
    // s.active_config_index и используется Ctrl+R + плотами.
    int active_now = -1;
    int to_remove = -1;
    bool run_pressed_in_tab = false;
    if (ImGui::BeginTabBar("##basins_tabs",
                           ImGuiTabBarFlags_Reorderable |
                           ImGuiTabBarFlags_AutoSelectNewTabs |
                           ImGuiTabBarFlags_FittingPolicyScroll)) {
        for (int i = 0; i < (int)s.configs.size(); ++i) {
            BasinsConfig& bc = s.configs[i];
            ImGui::PushID(i);
            bool open = true;
            std::string tab_id = bc.label + "###basins_tab_" + std::to_string(i);
            bool can_close = !(s.in_flight && s.running_config_index == i);
            if (ImGui::BeginTabItem(tab_id.c_str(), can_close ? &open : nullptr)) {
                active_now = i;
                ImGui::EndTabItem();
            }
            if (!open) to_remove = i;
            ImGui::PopID();
        }
        if (!s.in_flight) {
            if (ImGui::TabItemButton("+",
                                     ImGuiTabItemFlags_Trailing |
                                     ImGuiTabItemFlags_NoTooltip)) {
                s.add_config();
            }
        }
        ImGui::EndTabBar();
    }
    if (active_now >= 0) s.active_config_index = active_now;
    if (to_remove >= 0) model.remove_basins_config(to_remove);
    (void)run_pressed_in_tab;

    if (s.configs.empty()) {
        ImGui::TextDisabled("No basins configs. Press '+' to add one.");
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
        s.active_config_index = 0;
    BasinsConfig& c = s.configs[s.active_config_index];

    // Inline rename для активной вкладки.
    {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%s", c.label.c_str());
        ImGui::SetNextItemWidth(200);
        if (ImGui::InputText("Label##basins_label", buf, sizeof(buf)))
            c.label = buf;
    }
    ImGui::Separator();

    // ----- Scheme -----
    static const char* schemes[] = { "Euler", "Euler-Cromer", "Explicit Midpoint", "RK4", "DOPRI78", "CD" };
    ImGui::SetNextItemWidth(160);
    if (ImGui::BeginCombo("Scheme", c.scheme.c_str())) {
        for (auto m : schemes)
            if (ImGui::Selectable(m, c.scheme == m)) c.scheme = m;
        if (!s.custom_schemes.empty()) ImGui::Separator();
        for (const auto& cs : s.custom_schemes)
            if (ImGui::Selectable((cs.name + " (custom)").c_str(), c.scheme == cs.name))
                c.scheme = cs.name;
        ImGui::EndCombo();
    }
    ImGui::Separator();

    // ----- Axes (X, Y по двум IC-переменным) -----
    if (!s.vars.empty()) {
        if (c.axis_x_var < 0 || c.axis_x_var >= (int)s.vars.size()) c.axis_x_var = 0;
        if (c.axis_y_var < 0 || c.axis_y_var >= (int)s.vars.size())
            c.axis_y_var = (s.vars.size() > 1) ? 1 : 0;
        std::vector<const char*> items;
        items.reserve(s.vars.size());
        for (const auto& v : s.vars) items.push_back(v.c_str());

        ImGui::SetNextItemWidth(160);
        ImGui::Combo("Axis X (IC)", &c.axis_x_var, items.data(), (int)items.size());
        InputNumStr("X lo", c.axis_x_lo_text, 120);
        InputNumStr("X hi", c.axis_x_hi_text, 120);

        ImGui::SetNextItemWidth(160);
        ImGui::Combo("Axis Y (IC)", &c.axis_y_var, items.data(), (int)items.size());
        InputNumStr("Y lo", c.axis_y_lo_text, 120);
        InputNumStr("Y hi", c.axis_y_hi_text, 120);
    } else {
        ImGui::TextDisabled("No variables (load a system first)");
    }
    InputNumStr("Resolution", c.n_pts_text, 120);

    // ----- Writable var (для peak finder) -----
    draw_writable_var_combo(s.vars, c.writable_var, "Writable var##bas_wv");

    ImGui::Separator();

    // ----- Integration (collapsible, swapped above Features) -----
    if (ImGui::CollapsingHeader("Integration", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("h",              c.h_text,           120);
        if (c.scheme == "CD" || custom_scheme_uses_symmetry(c.scheme, s.custom_schemes))
            InputNumStr("symmetry s", c.symmetry_s,       120);
        InputNumStr("computing time", c.t_max_text,       120);
        InputNumStr("transient time", c.transient_text,   120);
        InputNumStr("decimator",      c.pre_scaller_text, 120);
        InputNumStr("max value",      c.max_value_text,   120);
    }

    // ----- Features (DBSCAN axes + plot data) (collapsible) -----
    // 12 фич (см. BF_* в configCUDA.h / enum BasinFeature в analysis_session.h).
    // Feature 1 пишется в outAvgPeaks-буфер (X-координата DBSCAN), Feature 2 —
    // в AvgTimeOfPeaks-буфер (Y-координата). Множители применяются ПОСЛЕ
    // вычисления фичи и нужны для масштабирования кластеризации.
    if (ImGui::CollapsingHeader("Features (DBSCAN axes + plot data)",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        static const char* feat_names[] = {
            "Avg peaks",             "Avg intervals",
            "RMS peaks",             "RMS intervals",
            "StDev peaks",           "StDev intervals",
            "sign\xc2\xb7log10|avg peaks|", "sign\xc2\xb7log10|avg intervals|",
            "log10 RMS peaks",       "log10 RMS intervals",
            "log10 StDev peaks",     "log10 StDev intervals",
        };
        if (c.feature1 < 0 || c.feature1 >= BF_FEATURE_COUNT) c.feature1 = BF_FEATURE1_DEFAULT;
        if (c.feature2 < 0 || c.feature2 >= BF_FEATURE_COUNT) c.feature2 = BF_FEATURE2_DEFAULT;
        ImGui::SetNextItemWidth(220);
        ImGui::Combo("Feature 1##bas", &c.feature1, feat_names, IM_ARRAYSIZE(feat_names));
        ImGui::SameLine();
        InputNumStr("mult##bas_f1", c.mult_feature1_text, 80);
        ImGui::SetNextItemWidth(220);
        ImGui::Combo("Feature 2##bas", &c.feature2, feat_names, IM_ARRAYSIZE(feat_names));
        ImGui::SameLine();
        InputNumStr("mult##bas_f2", c.mult_feature2_text, 80);
    }

    // DBSCAN eps — кластеризационный радиус в (Feature 1, Feature 2) пространстве.
    // Оставлен снаружи Features-секции: тюнится чаще, чем выбор самих фич.
    // Рядом — кнопка "Clustering": перезапускает только DBSCAN-фазу по уже
    // посчитанным фичам (avg_peaks / avg_intervals в c.result), быстрее чем
    // полный Run в десятки/сотни раз. Дизейблится, если нет валидного
    // предыдущего результата или уже идёт расчёт.
    InputNumStr("DBSCAN eps", c.eps_dbscan_text, 120);
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

    // ----- Initial conditions (collapsible) -----
    if (ImGui::CollapsingHeader("Initial conditions",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextDisabled("Values for non-axis variables.");
        for (const auto& v : s.vars) {
            ImGui::PushID(v.c_str());
            InputNumStr(v.c_str(), c.initial_conditions[v], 120);
            ImGui::PopID();
        }
    }

    // ----- Parameters (collapsible) -----
    if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (const auto& p : s.params) {
            ImGui::PushID(p.c_str());
            InputNumStr(p.c_str(), c.param_values[p], 120);
            ImGui::PopID();
        }
    }

    // ----- CSV output (collapsible, moved to the bottom) -----
    if (ImGui::CollapsingHeader("CSV output")) {
        ImGui::Checkbox("Save to file", &c.csv_save_enabled);
        InputTextStr("##basins_csv_path", c.csv_output_path);
        ImGui::TextDisabled("Writes 4 files: <path>, _1.csv (Feature 1), _2.csv (Feature 2), _3.csv (states).");
    }

    if (c.last_run_ok) {
        int total = (int)c.result.basin_idx.size();
        int diverged = 0;
        for (int f : c.result.helpful_array) if (f == 0) ++diverged;
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
            "OK: %dx%d, %d clusters (+ %d FP clusters); %d cells unbound",
            c.result.n_pts, c.result.n_pts,
            c.result.n_clusters, -c.result.min_cluster_idx, diverged);
    } else if (!c.last_error.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error (selectable, Ctrl+C):");
        ImVec2 sz(-1.0f, ImGui::GetTextLineHeight() * 12);
        ImGui::InputTextMultiline("##basins_err",
            const_cast<char*>(c.last_error.c_str()),
            c.last_error.size() + 1,
            sz,
            ImGuiInputTextFlags_ReadOnly);
    }
}

// Plot Basins: inner tab-bar по 5 представлениям. Heatmap-views — per
// (config × tab) в std::map по owner_id. HeatmapView/Plot2DView хранят
// data_gen_cached внутри без учёта owner_id, поэтому переиспользовать один
// view между разными configs нельзя (после Run All у двух configs одинаковый
// data_generation=1 → cache не invalidate'тся и на чужой вкладке показывается
// предыдущий buffer). Map с lazy-init решает это и сохраняет независимый
// zoom/pan per (config, tab).
// Координаты обхода NxN-сетки по спирали из центра (порт MATLAB
// spiral_coords_from_center). Стартовая клетка — округление к верху центра:
// для N=5 это (2,2), для N=4 это (1,1) (0-based). Дальше — right→down→left→up
// с увеличением шага на каждом цикле полу-оборотов. Точки за границей грид-а
// пропускаются, поэтому в итоге набирается ровно N*N валидных индексов.
// Возвращает row-major индексы row*N + col.
static std::vector<int> spiral_coords_from_center(int N) {
    std::vector<int> out;
    if (N <= 0) return out;
    const int total = N * N;
    out.reserve((size_t)total);
    int r = (N - 1) / 2;
    int c = (N - 1) / 2;
    out.push_back(r * N + c);
    static const int dr[4] = { 0, 1, 0, -1 };
    static const int dc[4] = { 1, 0, -1, 0 };
    int dir = 0;
    int step = 1;
    while ((int)out.size() < total) {
        for (int k = 0; k < 2; ++k) {
            for (int t = 0; t < step; ++t) {
                r += dr[dir];
                c += dc[dir];
                if (r >= 0 && r < N && c >= 0 && c < N) {
                    out.push_back(r * N + c);
                    if ((int)out.size() >= total) return out;
                }
            }
            dir = (dir + 1) & 3;
        }
        ++step;
    }
    return out;
}

// Перенумерация cluster id'ов по порядку первого появления при обходе spiral-
// from-center. Положительные оригинальные id отображаются в 1, 2, 3, ...
// отрицательные — в -1, -2, -3, ... Ноли (diverged) сохраняются как 0.
// Также возвращает via out-params количество положительных / отрицательных
// кластеров для обновления colorbar-диапазона.
static std::vector<int> renumber_basins_spiral(const std::vector<int>& src, int N,
                                               int& out_n_pos, int& out_n_neg) {
    out_n_pos = 0;
    out_n_neg = 0;
    std::vector<int> dst(src.size(), 0);
    if ((int)src.size() != N * N || N <= 0) return dst;
    const std::vector<int> order = spiral_coords_from_center(N);
    std::unordered_map<int, int> pos_map, neg_map;
    int pos_next = 1, neg_next = 1;
    for (int idx : order) {
        const int v = src[(size_t)idx];
        if (v > 0) {
            auto it = pos_map.find(v);
            if (it == pos_map.end()) { it = pos_map.emplace(v, pos_next++).first; }
            dst[(size_t)idx] = it->second;
        } else if (v < 0) {
            auto it = neg_map.find(v);
            if (it == neg_map.end()) { it = neg_map.emplace(v, -(neg_next++)).first; }
            dst[(size_t)idx] = it->second;
        }
        // v == 0 — diverged, dst остаётся 0.
    }
    out_n_pos = pos_next - 1;
    out_n_neg = neg_next - 1;
    return dst;
}

// Лениво (пере)заполнить c.basin_idx_spiral / c.n_clusters_spiral /
// c.min_cluster_idx_spiral для текущего поколения данных. Безопасно вызывать
// каждый кадр — пересчёт случается только при смене data_generation.
static void ensure_basins_spiral_cache(BasinsConfig& c) {
    if (c.basin_idx_spiral_gen == c.data_generation
        && c.basin_idx_spiral.size() == c.result.basin_idx.size()) return;
    int n_pos = 0, n_neg = 0;
    c.basin_idx_spiral = renumber_basins_spiral(c.result.basin_idx, c.result.n_pts,
                                                n_pos, n_neg);
    c.n_clusters_spiral      = n_pos;
    c.min_cluster_idx_spiral = -n_neg;
    c.basin_idx_spiral_gen   = c.data_generation;
}

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
    const unsigned base_oid = 0x1BA50000u + (unsigned)s.active_config_index * 5u;
    static std::unique_ptr<PlotRenderer> renderer;
    static std::map<unsigned, std::unique_ptr<HeatmapView>> hm_basins, hm_avgpk, hm_avgint, hm_states;
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
    HeatmapView& hm_basins_v = get_hm(hm_basins, base_oid + 0u, /*discrete*/ true);
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
        if (ImGui::MenuItem("Export data...", nullptr, false, !basins_busy)) {
            if (cb.pick_save_file_csv) {
                std::string path = cb.pick_save_file_csv();
                if (!path.empty()) {
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
                }
            }
        }
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
        int  app_default = 2;   // Turbo
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
            int cm = (*cfg_field >= 0) ? *cfg_field : app_default;
            if (cm < 0 || cm >= kHeatmapColormapCount) cm = 2;
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
        const int  src_n_clusters     = c.renumber_spiral ? c.n_clusters_spiral
                                                          : c.result.n_clusters;
        const int  src_min_cluster_id = c.renumber_spiral ? c.min_cluster_idx_spiral
                                                          : c.result.min_cluster_idx;
        static std::vector<double> buf;
        buf.resize(total);
        for (size_t k = 0; k < total; ++k) buf[k] = (double)src_idx[k];
        // Cluster IDs:
        //   min_cluster_id..-1 — FP-clusters (always present when negative)
        //   0                  — diverged/unbound cells (helpful_array[i] == 0)
        //   1..n_clusters      — oscillatory clusters
        // When no FP-clusters exist (min_cluster_id == 0) and no cell
        // diverged, "cluster 0" isn't a real ID; shift vmin to 1 so the
        // colorbar doesn't show a phantom band. If diverged cells exist,
        // keep vmin = 0 so they get their own color band.
        bool has_diverged = false;
        for (int f : c.result.helpful_array)
            if (f == 0) { has_diverged = true; break; }
        double vmin;
        if (src_min_cluster_id < 0)              vmin = (double)src_min_cluster_id;
        else if (has_diverged)                   vmin = 0.0;
        else                                     vmin = 1.0;
        double vmax = (double)src_n_clusters;
        if (vmax < vmin) vmax = vmin;
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
        // States: helpful_array → 3 категории. Маппим в дискретные значения
        // и рендерим через HeatmapView в Turbo (3 равноотстоящие точки):
        //   1 (Osc)     → 0   (синий конец turbo)
        //   -1 (FP)     → 1   (зелёный середина)
        //   0 (Unbound) → 2   (красный конец)
        // Это не точные MATLAB-цвета, но 3 различимые категории.
        static std::vector<double> buf;
        buf.resize(total);
        for (size_t k = 0; k < total; ++k) {
            int v = c.result.helpful_array[k];
            buf[k] = (v == 1) ? 0.0 : (v == -1 ? 1.0 : 2.0);
        }
        hm_states_v.x_axis.name = ax_x;
        hm_states_v.y_axis.name = ax_y;
        hm_states_v.render(*renderer, origin, avail,
                          /*owner_id*/ base_oid + 3u, c.data_generation,
                          n, n, buf.data(),
                          xlo, xhi, ylo, yhi,
                          0.0, 2.0, fit);
        // Подсказка под плотом — числовые уровни, цвет зависит от выбранной colormap.
        ImGui::TextDisabled("Levels: 0=Osc, 1=FixedPoint, 2=Unbound");
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
        int max_id = c.renumber_spiral ? c.n_clusters_spiral      : c.result.n_clusters;
        int n_total_clusters = max_id - min_id + 1;
        if (n_total_clusters < 1) n_total_clusters = 1;

        // Сгруппируем точки по basin_idx.
        std::map<int, std::vector<float>> bufs;
        int valid_pts = 0;
        for (size_t k = 0; k < total; ++k) {
            int id = src_idx[k];
            double xp = c.result.avg_peaks[k];
            double yp = c.result.avg_intervals[k];
            if (!std::isfinite(xp) || !std::isfinite(yp)) continue;
            if (xp == 999.0 || xp == -999.0 || yp == 999.0 || yp == -999.0) continue;
            bufs[id].push_back((float)xp);
            bufs[id].push_back((float)yp);
            ++valid_pts;
        }
        if (valid_pts == 0) {
            ImGui::TextDisabled("No valid (avgPeak, avgInterval) points.");
            return;
        }

        // Static-буферы должны жить весь кадр (Plot2DView хранит сырые указатели).
        static std::vector<std::vector<float>> series_buffers;
        static std::vector<std::string>        series_labels;
        series_buffers.clear();
        series_labels.clear();
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
        (void)n_total_clusters;
        // Имена осей scatter'а — выбранные фичи. Должны быть синхронизированы
        // с feat_names в draw_basins_controls (тот же порядок BF_*).
        static const char* feat_names_plot[] = {
            "Avg peaks",             "Avg intervals",
            "RMS peaks",             "RMS intervals",
            "StDev peaks",           "StDev intervals",
            "sign\xc2\xb7log10|avg peaks|", "sign\xc2\xb7log10|avg intervals|",
            "log10 RMS peaks",       "log10 RMS intervals",
            "log10 StDev peaks",     "log10 StDev intervals",
        };
        int f1 = (c.feature1 >= 0 && c.feature1 < BF_FEATURE_COUNT) ? c.feature1 : BF_FEATURE1_DEFAULT;
        int f2 = (c.feature2 >= 0 && c.feature2 < BF_FEATURE_COUNT) ? c.feature2 : BF_FEATURE2_DEFAULT;
        scatter_v.x_axis.name = feat_names_plot[f1];
        scatter_v.y_axis.name = feat_names_plot[f2];
        // gen-token включает renumber_spiral — иначе Plot2DView::series_cache_
        // не перезаливает GPU-буфер и подписи/цвета остаются от прошлой версии.
        int scatter_gen = c.data_generation * 2 + (c.renumber_spiral ? 1 : 0);
        scatter_v.render(*renderer, origin, avail,
                             /*owner_id*/ base_oid + 4u, scatter_gen,
                             series_in, init_vis, glob_vis, fit);
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
    {
        bool no_cfg = s.configs.empty();
        bool do_run = false;
        if (s.in_flight) {
            ImGui::BeginDisabled();
            ImGui::Button("Running...", ImVec2(160, 0));
            ImGui::EndDisabled();
        } else {
            if (no_cfg) ImGui::BeginDisabled();
            do_run = ImGui::Button("Run (Ctrl+R)", ImVec2(160, 0));
            if (no_cfg) ImGui::EndDisabled();
        }
        if (!s.in_flight && !no_cfg && ImGui::GetIO().KeyCtrl && !ImGui::GetIO().KeyShift &&
            ImGui::IsKeyPressed(ImGuiKey_R, false))
            do_run = true;
        if (do_run) {
            if (!model.parametric_engine)
                model.parametric_engine = std::make_unique<ParametricEngine>();
            s.run_async(*model.parametric_engine, s.active_config_index);
        }
        ImGui::SameLine();
        if (s.in_flight && ImGui::Button("Cancel")) s.request_cancel();

        // Batch "Run all..." across FastSync configs. Pushes selected indices
        // into model.fastsync_queue; draw_gui ticks the queue after polls.
        // picks — вне if(BeginPopup): Ctrl+Shift+R должен пушить те же
        // отметки, даже если попап ни разу не открывали (см. Parametric).
        static std::vector<bool> picks;
        if (picks.size() != s.configs.size()) picks.assign(s.configs.size(), true);
        auto run_all_marked = [&]() {
            for (size_t i = 0; i < picks.size(); ++i)
                if (picks[i])
                    model.fastsync_queue.push_back({(int)i});
            model.start_next_in_fastsync_queue();
        };
        if (!s.in_flight && !no_cfg && ImGui::GetIO().KeyCtrl && ImGui::GetIO().KeyShift &&
            ImGui::IsKeyPressed(ImGuiKey_R, false))
            run_all_marked();

        ImGui::SameLine();
        const bool block_run_all = s.in_flight || no_cfg;
        if (block_run_all) ImGui::BeginDisabled();
        if (ImGui::Button("Run all... (Ctrl+Shift+R)"))
            ImGui::OpenPopup("##run_all_fastsync");
        if (block_run_all) ImGui::EndDisabled();
        if (!model.fastsync_queue.empty()) {
            ImGui::SameLine();
            ImGui::TextDisabled("(%zu queued)", model.fastsync_queue.size());
        }
        if (ImGui::BeginPopup("##run_all_fastsync")) {
            ImGui::TextDisabled("Sequential (one CUDA context).");
            for (size_t i = 0; i < picks.size(); ++i) {
                bool v = picks[i];
                std::string lbl = s.configs[i].label + "###pfs_" + std::to_string(i);
                if (ImGui::Checkbox(lbl.c_str(), &v)) picks[i] = v;
            }

            ImGui::Separator();
            if (ImGui::Button("All"))  { for (auto&& b : picks) b = true;  }
            ImGui::SameLine();
            if (ImGui::Button("None")) { for (auto&& b : picks) b = false; }
            ImGui::SameLine();
            if (ImGui::Button("Run")) {
                run_all_marked();
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }
    ImGui::Separator();

    // Tab bar для config'ов.
    int active_now = -1, to_remove = -1;
    if (ImGui::BeginTabBar("##fs_tabs",
                           ImGuiTabBarFlags_Reorderable |
                           ImGuiTabBarFlags_AutoSelectNewTabs |
                           ImGuiTabBarFlags_FittingPolicyScroll)) {
        for (int i = 0; i < (int)s.configs.size(); ++i) {
            FastSyncConfig& fc = s.configs[i];
            ImGui::PushID(i);
            bool open = true;
            std::string tab_id = fc.label + "###fs_tab_" + std::to_string(i);
            bool can_close = !(s.in_flight && s.running_config_index == i);
            if (ImGui::BeginTabItem(tab_id.c_str(), can_close ? &open : nullptr)) {
                active_now = i;
                ImGui::EndTabItem();
            }
            if (!open) to_remove = i;
            ImGui::PopID();
        }
        if (!s.in_flight) {
            if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip))
                s.add_config();
        }
        ImGui::EndTabBar();
    }
    if (active_now >= 0) s.active_config_index = active_now;
    if (to_remove >= 0)  model.remove_fastsync_config(to_remove);

    if (s.configs.empty()) {
        ImGui::TextDisabled("No FastSync configs. Press '+' to add one.");
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
        s.active_config_index = 0;
    FastSyncConfig& c = s.configs[s.active_config_index];

    // Label rename.
    {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%s", c.label.c_str());
        ImGui::SetNextItemWidth(200);
        if (ImGui::InputText("Label##fs_label", buf, sizeof(buf))) c.label = buf;
    }
    ImGui::Separator();

    // ----- Mode -----
    ImGui::Text("Mode:");
    ImGui::RadioButton("On Attractor", &c.mode, 0); ImGui::SameLine();
    ImGui::RadioButton("On Grid",      &c.mode, 1);
    ImGui::Separator();

    // ----- Scheme -----
    static const char* schemes[] = { "Euler", "Euler-Cromer", "Explicit Midpoint", "RK4", "DOPRI78", "CD" };
    ImGui::SetNextItemWidth(160);
    if (ImGui::BeginCombo("Scheme", c.scheme.c_str())) {
        for (auto m : schemes)
            if (ImGui::Selectable(m, c.scheme == m)) c.scheme = m;
        if (!s.custom_schemes.empty()) ImGui::Separator();
        for (const auto& cs : s.custom_schemes)
            if (ImGui::Selectable((cs.name + " (custom)").c_str(), c.scheme == cs.name))
                c.scheme = cs.name;
        ImGui::EndCombo();
    }
    if (c.scheme == "CD" || custom_scheme_uses_symmetry(c.scheme, s.custom_schemes))
        InputNumStr("symmetry s", c.symmetry_s, 120);
    ImGui::Separator();

    // ----- Mode-specific axes -----
    if (c.mode == 1 && !s.vars.empty()) {
        if (c.axis_x_var < 0 || c.axis_x_var >= (int)s.vars.size()) c.axis_x_var = 0;
        if (c.axis_y_var < 0 || c.axis_y_var >= (int)s.vars.size())
            c.axis_y_var = (s.vars.size() > 1) ? 1 : 0;
        std::vector<const char*> items;
        items.reserve(s.vars.size());
        for (const auto& v : s.vars) items.push_back(v.c_str());

        const char* axis_role_x = c.grid_swap_master_slave ? "Axis X (slave IC)" : "Axis X (master IC)";
        const char* axis_role_y = c.grid_swap_master_slave ? "Axis Y (slave IC)" : "Axis Y (master IC)";
        ImGui::SetNextItemWidth(160);
        ImGui::Combo(axis_role_x, &c.axis_x_var, items.data(), (int)items.size());
        InputNumStr("X lo", c.axis_x_lo_text, 120);
        InputNumStr("X hi", c.axis_x_hi_text, 120);
        ImGui::SetNextItemWidth(160);
        ImGui::Combo(axis_role_y, &c.axis_y_var, items.data(), (int)items.size());
        InputNumStr("Y lo", c.axis_y_lo_text, 120);
        InputNumStr("Y hi", c.axis_y_hi_text, 120);
        InputNumStr("Resolution", c.n_pts_text, 120);
        // Какую сторону перебирать по сетке: master IC (legacy default) или slave IC.
        ImGui::Checkbox("Vary slave IC (master fixed)##fs_gridswap", &c.grid_swap_master_slave);
        ImGui::TextDisabled("Off: grid sweeps master IC, slave fixed. On: grid sweeps slave IC, master fixed.");
        ImGui::Separator();
    }
    else if (c.mode == 0 && !s.vars.empty()) {
        // На траектории — какие 2 переменные показывать как X/Y фазового портрета.
        std::vector<const char*> items;
        items.reserve(s.vars.size());
        for (const auto& v : s.vars) items.push_back(v.c_str());
        ImGui::SetNextItemWidth(160);
        ImGui::Combo("Display X var", &c.axis_x_var, items.data(), (int)items.size());
        ImGui::SetNextItemWidth(160);
        ImGui::Combo("Display Y var", &c.axis_y_var, items.data(), (int)items.size());
        ImGui::Separator();
    }

    // ----- Integration (collapsible) -----
    if (ImGui::CollapsingHeader("Integration", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("h",                c.h_text,              120);
        if (c.mode == 0) {
            InputNumStr("t_max",        c.t_max_text,          120);
        }
        InputNumStr("transient",        c.transient_text,      120);
        bool window_changed = InputNumStr("window",           c.window_text,         120);
        bool iter_changed   = InputNumStr("iter of synch",    c.iter_of_synchr_text, 120);
        if (!window_changed && !iter_changed) {
            // window/iter — источник истины в этом кадре (или ничего не
            // менялось, напр. только что загрузили сессию): пересчитываем
            // total sync time ДО отрисовки его поля, без задержки в кадр.
            double window_v = std::atof(c.window_text.c_str());
            double iter_v   = std::max(1.0, std::atof(c.iter_of_synchr_text.c_str()));
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.10g", (2.0 * iter_v - 1.0) * window_v);
            c.total_time_text = buf;
        }
        if (InputNumStr("total sync time", c.total_time_text, 120)) {
            // Пользователь правит total → пересчитываем window (iter фиксирован).
            double total_v = std::atof(c.total_time_text.c_str());
            double iter_v  = std::max(1.0, std::atof(c.iter_of_synchr_text.c_str()));
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.10g", total_v / (2.0 * iter_v - 1.0));
            c.window_text = buf;
        }
        if (c.mode == 0) {
            InputNumStr("decimator",    c.pre_scaller_text,    120);
        }
        InputNumStr("max value",        c.max_value_text,      120);
    }

    // ----- Synchro runtime (collapsible) -----
    if (ImGui::CollapsingHeader("Synchro runtime", ImGuiTreeNodeFlags_DefaultOpen)) {
        static const char* tos_names[] = { "Unidirectional", "Bidirectional" };
        ImGui::SetNextItemWidth(160);
        ImGui::Combo("Type of synch.", &c.type_of_synch, tos_names, IM_ARRAYSIZE(tos_names));
        static const char* ee_names[] = {
            "0: RMS on last iter",
            "1: # iters to reach FS_error_trs",
            "2: RMS at last point"
        };
        ImGui::SetNextItemWidth(280);
        ImGui::Combo("Error estim.", &c.error_estim, ee_names, IM_ARRAYSIZE(ee_names));
        InputNumStr("FS error trs.", c.fs_error_trs_text, 120);
    }

    // ----- Parameters (collapsible, placed under Synchro runtime) -----
    if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (const auto& p : s.params) {
            ImGui::PushID(p.c_str());
            InputNumStr(p.c_str(), c.param_values[p], 120);
            ImGui::PopID();
        }
    }

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
    if (ImGui::CollapsingHeader("CSV output")) {
        ImGui::Checkbox("Save to file##fs_csv", &c.csv_save_enabled);
        InputTextStr("##fs_csv_path", c.csv_output_path);
        if (c.mode == 0) {
            ImGui::TextDisabled("Writes one row per trajectory point: x[0],..,x[N-1],sync_error.");
        } else {
            ImGui::TextDisabled("Writes 2 header lines (X/Y ranges) + n_pts x n_pts error matrix (row-major).");
        }
    }

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
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error (selectable, Ctrl+C):");
        ImVec2 sz(-1.0f, ImGui::GetTextLineHeight() * 10);
        ImGui::InputTextMultiline("##fs_err",
            const_cast<char*>(c.last_error.c_str()),
            c.last_error.size() + 1,
            sz, ImGuiInputTextFlags_ReadOnly);
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

    if (c.colormap_idx < 0 || c.colormap_idx >= kHeatmapColormapCount) c.colormap_idx = 2;
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

    auto parse_d_local = [](const std::string& s, double def) -> double {
        if (s.empty()) return def;
        try { return std::stod(s); } catch (...) { return def; }
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
        cmin_user = parse_d_local(c.c_min_text, -12.0);
        cmax_user = parse_d_local(c.c_max_text,   0.0);
    }
    bool invert_cmap = (!c.autoscale_color) && (cmin_user > cmax_user);
    double cmin = std::min(cmin_user, cmax_user);
    double cmax = std::max(cmin_user, cmax_user);
    if (!(cmax > cmin)) cmax = cmin + 1.0;
    HeatmapColormap cmap = (HeatmapColormap)((c.colormap_idx >= 0 && c.colormap_idx < kHeatmapColormapCount) ? c.colormap_idx : 2);

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
            if (ImGui::MenuItem("Export data...", nullptr, false, !fs_busy)) {
                if (cb.pick_save_file_csv) {
                    std::string path = cb.pick_save_file_csv();
                    if (!path.empty())
                        data_export::export_fastsync(c.result, path);
                }
            }
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
            if (ImGui::MenuItem("Export data...", nullptr, false, !fs_busy)) {
                if (cb.pick_save_file_csv) {
                    std::string path = cb.pick_save_file_csv();
                    if (!path.empty())
                        data_export::export_fastsync(c.result, path);
                }
            }
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

    // ----- Run (active sub-tab, active config) -----
    // Кнопка единая для Bif/LLE/LS, диспатчится по parametric_active_analysis
    // (0=Bif, 1=LLE, 2=LS) к соответствующему active_*_index. Стоит слева от
    // Run all... в одной строке.
    {
        int kind = model.parametric_active_analysis;
        int active_idx = -1;
        bool no_active = false;
        if (kind == 0) {
            active_idx = model.bifurcation_session.active_diagram_index;
            if (model.bifurcation_session.diagrams.empty()) no_active = true;
        } else if (kind == 1) {
            active_idx = model.lle_session.active_curve_index;
            if (model.lle_session.curves.empty()) no_active = true;
        } else if (kind == 2) {
            active_idx = model.ls_session.active_curve_index;
            if (model.ls_session.curves.empty()) no_active = true;
        }

        bool do_run = false;
        if (any_in_flight) {
            ImGui::BeginDisabled();
            ImGui::Button("Running...", ImVec2(140, 0));
            ImGui::EndDisabled();
        } else {
            if (no_active) ImGui::BeginDisabled();
            do_run = ImGui::Button("Run (Ctrl+R)", ImVec2(140, 0));
            if (no_active) ImGui::EndDisabled();
        }
        if (!any_in_flight && !no_active && ImGui::GetIO().KeyCtrl && !ImGui::GetIO().KeyShift &&
            ImGui::IsKeyPressed(ImGuiKey_R, false))
            do_run = true;
        if (do_run && active_idx >= 0) {
            if (!model.parametric_engine)
                model.parametric_engine = std::make_unique<ParametricEngine>();
            if (kind == 0)
                model.bifurcation_session.run_async(*model.parametric_engine, active_idx);
            else if (kind == 1)
                model.lle_session.run_async(*model.parametric_engine, active_idx);
            else if (kind == 2)
                model.ls_session.run_async(*model.parametric_engine, active_idx);
        }
        ImGui::SameLine();
    }

    // Batch Run all — global across BD/LLE/LS. Popup shows checkboxes for
    // every configured diagram/curve/spectrum; Run pushes selected to
    // model.parametric_queue and starts the first one. draw_gui ticks the
    // queue after polls. picks_* и run_all_marked живут вне if(BeginPopup) —
    // Ctrl+Shift+R должен пушить те же отметки, даже если попап ни разу не
    // открывали в этой сессии (тогда picks_* просто "всё отмечено" по fit()).
    static std::vector<bool> picks_bd, picks_lle, picks_ls;
    auto fit = [&](std::vector<bool>& v, size_t n) {
        if (v.size() != n) v.assign(n, true);
    };
    fit(picks_bd,  model.bifurcation_session.diagrams.size());
    fit(picks_lle, model.lle_session.curves.size());
    fit(picks_ls,  model.ls_session.curves.size());

    auto run_all_marked = [&]() {
        for (size_t i = 0; i < picks_bd.size(); ++i)
            if (picks_bd[i])
                model.parametric_queue.push_back({ParametricQueueItem::Kind::Bifurcation, (int)i});
        for (size_t i = 0; i < picks_lle.size(); ++i)
            if (picks_lle[i])
                model.parametric_queue.push_back({ParametricQueueItem::Kind::LLE, (int)i});
        for (size_t i = 0; i < picks_ls.size(); ++i)
            if (picks_ls[i])
                model.parametric_queue.push_back({ParametricQueueItem::Kind::LS, (int)i});
        model.start_next_in_parametric_queue();
    };

    if (!any_in_flight && ImGui::GetIO().KeyCtrl && ImGui::GetIO().KeyShift &&
        ImGui::IsKeyPressed(ImGuiKey_R, false))
        run_all_marked();

    if (any_in_flight) ImGui::BeginDisabled();
    if (ImGui::Button("Run all... (Ctrl+Shift+R)"))
        ImGui::OpenPopup("##run_all_parametric");
    if (any_in_flight) ImGui::EndDisabled();
    if (!model.parametric_queue.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("(%zu queued)", model.parametric_queue.size());
    }
    if (ImGui::BeginPopup("##run_all_parametric")) {
        ImGui::TextDisabled("Sequential (one CUDA context).");

        if (!picks_bd.empty()) {
            ImGui::SeparatorText("Bifurcation");
            for (size_t i = 0; i < picks_bd.size(); ++i) {
                bool v = picks_bd[i];
                std::string lbl = model.bifurcation_session.diagrams[i].label
                                  + "###pbd_" + std::to_string(i);
                if (ImGui::Checkbox(lbl.c_str(), &v)) picks_bd[i] = v;
            }
        }
        if (!picks_lle.empty()) {
            ImGui::SeparatorText("LLE");
            for (size_t i = 0; i < picks_lle.size(); ++i) {
                bool v = picks_lle[i];
                std::string lbl = model.lle_session.curves[i].label
                                  + "###plle_" + std::to_string(i);
                if (ImGui::Checkbox(lbl.c_str(), &v)) picks_lle[i] = v;
            }
        }
        if (!picks_ls.empty()) {
            ImGui::SeparatorText("LS");
            for (size_t i = 0; i < picks_ls.size(); ++i) {
                bool v = picks_ls[i];
                std::string lbl = model.ls_session.curves[i].label
                                  + "###pls_" + std::to_string(i);
                if (ImGui::Checkbox(lbl.c_str(), &v)) picks_ls[i] = v;
            }
        }

        ImGui::Separator();
        if (ImGui::Button("All")) {
            for (auto&& b : picks_bd)  b = true;
            for (auto&& b : picks_lle) b = true;
            for (auto&& b : picks_ls)  b = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("None")) {
            for (auto&& b : picks_bd)  b = false;
            for (auto&& b : picks_lle) b = false;
            for (auto&& b : picks_ls)  b = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Run")) {
            run_all_marked();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
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

    struct ParamMatchItem { int index; std::string label; };
    auto matching_items = [&](ParametricPlotWindow::Kind kind, bool mode_2d) {
        std::vector<ParamMatchItem> out;
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
    };
    // Colored 1D — не отдельный "kind" в этом комбо, а тумблер поверх плота
    // (draw_bifurcation_plot). Type combo снова различает только 1D/2D, как
    // у LLE/LS.
    static const char* type_names[] = { "Bifurcation 1D", "Bifurcation 2D", "LLE 1D", "LLE 2D", "LS 1D", "LS 2D" };
    auto type_index_of = [](ParametricPlotWindow::Kind kind, bool mode_2d) -> int {
        int base = kind == ParametricPlotWindow::Kind::Bifurcation ? 0
                 : kind == ParametricPlotWindow::Kind::LLE ? 2 : 4;
        return base + (mode_2d ? 1 : 0);
    };
    auto type_from_index = [](int t, ParametricPlotWindow::Kind& kind, bool& mode_2d) {
        kind    = (t < 2) ? ParametricPlotWindow::Kind::Bifurcation
                : (t < 4) ? ParametricPlotWindow::Kind::LLE : ParametricPlotWindow::Kind::LS;
        mode_2d = (t % 2) == 1;
    };

    int win_to_remove = -1;
    for (int i = 0; i < (int)model.parametric_plot_windows.size(); ++i) {
        ParametricPlotWindow& win = model.parametric_plot_windows[i];
        ImGui::PushID(win.id);
        ImGui::SetNextItemWidth(220);
        if (InputTextStr("##wlabel", win.label)) {
            win.label_is_manual = !win.label.empty();   // empty → back to auto
            model.parametric_plot_windows_dirty = true;
        }
        ImGui::SameLine();

        // Type combo: changing kind/dimension invalidates old member indices
        // (they're only meaningful within the previous session+dimension),
        // so switching type clears members.
        int t = type_index_of(win.kind, win.mode_2d);
        ImGui::SetNextItemWidth(130);
        if (ImGui::Combo("##wtype", &t, type_names, IM_ARRAYSIZE(type_names))) {
            ParametricPlotWindow::Kind new_kind; bool new_2d;
            type_from_index(t, new_kind, new_2d);
            if (new_kind != win.kind || new_2d != win.mode_2d) {
                win.kind = new_kind;
                win.mode_2d = new_2d;
                win.colored_1d = false;   // combo больше не умеет выбирать Colored 1D — сбрасываем; включается тумблером над плотом
                win.members.clear();
                model.parametric_plot_windows_dirty = true;
            }
        }
        ImGui::SameLine();

        if (ImGui::Button("Members...")) ImGui::OpenPopup("edit_plot_window_members");
        if (ImGui::BeginPopup("edit_plot_window_members")) {
            auto items = matching_items(win.kind, win.mode_2d);
            if (items.empty()) {
                ImGui::TextDisabled("(none available)");
            } else if (win.mode_2d || win.colored_1d) {
                // 2D / Colored 1D (сейчас включён тумблером над плотом) show
                // heatmaps — single-select (radio), not independent checkboxes.
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
        if (ImGui::SmallButton("X")) win_to_remove = i;
        ImGui::PopID();
    }
    if (win_to_remove >= 0) model.remove_parametric_plot_window(win_to_remove);

    if (ImGui::Button("Add window")) {
        model.add_parametric_plot_window(ParametricPlotWindow::Kind::Bifurcation, false, {});
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset windows layout")) { model.parametric_layout_generation++; }
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
            if (!j.empty()) session_from_json(j, model.phase_session);
            break;
        }
        case AppModel::AppMode::Parametric: {
            model.start_parametric_analysis();
            std::string jb = lib.load_session(model.loaded_name, "_last_parametric");
            if (!jb.empty()) session_from_json_parametric(jb, model.bifurcation_session);
            std::string jl = lib.load_session(model.loaded_name, "_last_lle");
            if (!jl.empty()) session_from_json_lle(jl, model.lle_session);
            std::string js = lib.load_session(model.loaded_name, "_last_ls");
            if (!js.empty()) session_from_json_ls(js, model.ls_session);
            std::string jw = lib.load_session(model.loaded_name, "_last_parametric_windows");
            model.load_or_init_parametric_plot_windows(jw);
            break;
        }
        case AppModel::AppMode::Dft1D: {
            model.start_dft1d_analysis();
            std::string jd = lib.load_session(model.loaded_name, "_last_dft1d");
            if (!jd.empty()) session_from_json_dft1d(jd, model.dft1d_session);
            std::string jw = lib.load_session(model.loaded_name, "_last_dft1d_windows");
            model.load_or_init_dft1d_plot_windows(jw);
            break;
        }
        case AppModel::AppMode::Basins: {
            model.start_basins_analysis();
            std::string jb = lib.load_session(model.loaded_name, "_last_basins");
            if (!jb.empty()) session_from_json_basins(jb, model.basins_session);
            break;
        }
        case AppModel::AppMode::FastSync: {
            model.start_fastsync_analysis();
            std::string jf = lib.load_session(model.loaded_name, "_last_fastsync");
            if (!jf.empty()) session_from_json_fastsync(jf, model.fastsync_session);
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
            if (!jc.empty()) session_from_json_custom(jc, model.custom_session);
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

// Parameter/variable combo shared by shared-config sweep pickers. Renders a
// combo whose current preview is the selected name; on selection updates
// (par_index, over_var, var_index) together — the same shape existing draw_*
// blocks in this file use, just extracted here for reuse.
// other_over_h — флаг ВТОРОЙ оси. Ровно одна ось может свипаться по h
// (2D-ядра принимают один hSweepAxis), поэтому при выборе dt (h) здесь второй
// флаг гасится, а если он уже занят — пункт показывается disabled с пояснением.
// nullptr = второй оси нет (1D-контекст, ограничение неактуально).
void draw_sweep_target_combo(const char* label,
                             const std::vector<std::string>& params,
                             const std::vector<std::string>& vars,
                             int& par_index, bool& over_var, int& var_index,
                             bool& over_h, bool* other_over_h = nullptr) {
    const std::string preview =
          over_h   ? std::string("dt (h)")
        : over_var ? (var_index >= 0 && var_index < (int)vars.size() ? vars[var_index] : std::string("var"))
                   : (par_index >= 0 && par_index < (int)params.size() ? params[par_index] : std::string("par"));
    ImGui::SetNextItemWidth(120.0f);
    if (ImGui::BeginCombo(label, preview.c_str())) {
        for (int i = 0; i < (int)params.size(); ++i) {
            bool sel = !over_var && !over_h && par_index == i;
            if (ImGui::Selectable(params[i].c_str(), sel)) {
                par_index = i; over_var = false; over_h = false;
            }
        }
        if (!vars.empty()) ImGui::Separator();
        for (int i = 0; i < (int)vars.size(); ++i) {
            bool sel = over_var && !over_h && var_index == i;
            std::string lbl = std::string("IC ") + vars[i];
            if (ImGui::Selectable(lbl.c_str(), sel)) {
                var_index = i; over_var = true; over_h = false;
            }
        }
        ImGui::Separator();
        const bool h_taken = (other_over_h != nullptr) && *other_over_h && !over_h;
        ImGui::BeginDisabled(h_taken);
        if (ImGui::Selectable("dt (h)", over_h)) {
            over_h = true; over_var = false;
            if (other_over_h) *other_over_h = false;   // ровно одна ось = h
        }
        ImGui::EndDisabled();
        if (h_taken) {
            ImGui::SameLine();
            ImGui::TextDisabled("(занят другой осью)");
        }
        ImGui::EndCombo();
    }
}

// Parse a text-formatted numeric field. Empty / bad → default 0. Used only
// to clamp sliders — the actual engine calls parse_d/parse_val on its own.
double parse_num_default(const std::string& s, double def) {
    if (s.empty()) return def;
    try { return std::stod(s); } catch (...) { return def; }
}

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
    static const char* schemes[] = { "Euler", "Euler-Cromer", "Explicit Midpoint", "RK4", "DOPRI78", "CD" };
    ImGui::SetNextItemWidth(160);
    if (ImGui::BeginCombo("Scheme", c.scheme.c_str())) {
        auto pick_scheme = [&](const std::string& nm) {
            c.scheme = nm;
            phase.scheme = nm;
            phase.regenerate_krs();
        };
        for (auto m : schemes)
            if (ImGui::Selectable(m, c.scheme == m)) pick_scheme(m);
        if (!custom_schemes.empty()) ImGui::Separator();
        for (const auto& scm : custom_schemes)
            if (ImGui::Selectable((scm.name + " (custom)").c_str(), c.scheme == scm.name)) {
                pick_scheme(scm.name);
                phase.use_gpu = true;
            }
        ImGui::EndCombo();
    }

    // Integration group — mirrors "Integration##bd_int" collapsing header in
    // draw_diagram_controls (per-line InputNumStr with comma→dot + ↑/↓).
    // Each edited field is mirrored into L1D's own override (`l1d_h_text` for
    // step) and into `phase_session`'s corresponding field, so the value the
    // user typed here shows up in the L1D and L3 panels without waiting for
    // Run. L1D still keeps independent transient/computing-time overrides —
    // step is intentionally the ONLY per-L1D field kept in sync with shared.
    if (ImGui::CollapsingHeader("Integration##custom_int", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (InputNumStr("h",              c.h_text,           120)) {
            c.l1d_h_text  = c.h_text;
            phase.step_h  = c.h_text;
        }
        if (c.scheme == "CD" || custom_scheme_uses_symmetry(c.scheme, custom_schemes))
            if (InputNumStr("symmetry s", c.symmetry_s,       120))
                phase.symmetry_s = c.symmetry_s;
        // TT before CT: transient runs first, computing-time is what's
        // actually sampled after — order matches conceptual flow.
        if (InputNumStr("transient time", c.transient_text,   120))
            phase.skip_time = c.transient_text;
        if (InputNumStr("computing time", c.t_max_text,       120))
            phase.sim_time  = c.t_max_text;
        if (InputNumStr("decimator",      c.pre_scaller_text, 120))
            phase.decimation = c.pre_scaller_text;
        InputNumStr("max value",      c.max_value_text,   120); // Phase has no analogue
    }

    // Initial conditions — one InputNumStr per line, matching draw_diagram_controls.
    // Not propagated to Phase: phase.ic_sets is multi-IC (mulistability) and
    // is edited in the L3 Phase panel; the shared IC block drives BD/LLE/LS/Basins.
    if (ImGui::CollapsingHeader("Initial conditions##custom_ic", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (const auto& v : vars) {
            ImGui::PushID(v.c_str());
            InputNumStr(v.c_str(), c.initial_conditions[v], 120);
            ImGui::PopID();
        }
    }

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
            if (InputNumStr(p.c_str(), c.param_values[p], 120)) {
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
                double new_val = parse_num_default(c.param_values[p], 0.0);
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
                            c.axis_x_over_h, &c.axis_y_over_h);
    InputNumStr("lo##ax", c.axis_x_lo_text, 120);
    InputNumStr("hi##ax", c.axis_x_hi_text, 120);

    ImGui::TextUnformatted("Axis Y:"); ImGui::SameLine();
    draw_sweep_target_combo("##ay", cs.params, cs.vars,
                            c.axis_y_par_index, c.axis_y_over_var, c.axis_y_var_index,
                            c.axis_y_over_h, &c.axis_x_over_h);
    InputNumStr("lo##ay", c.axis_y_lo_text, 120);
    InputNumStr("hi##ay", c.axis_y_hi_text, 120);

    // Shared N×N grid resolution — kernel `getValueByIdx` requires a square
    // grid, so one field drives both axes (matches Analysis tab).
    InputNumStr("Resolution##a", c.resolution_text, 120);

    // Per-type options edited directly on the sub-session's slot [0] (2D config).
    ImGui::Separator();
    if (c.bif2d_enabled && !cs.bif_session.diagrams.empty())
        InputNumStr("Bif DBSCAN eps", cs.bif_session.diagrams[0].eps_dbscan_text, 120);
    if (c.lle2d_enabled && !cs.lle_session.curves.empty()) {
        InputNumStr("LLE eps", cs.lle_session.curves[0].eps_text, 120);
        InputNumStr("LLE NT",  cs.lle_session.curves[0].nt_text,  120);
    }
    if (c.ls2d_enabled && !cs.ls_session.curves.empty()) {
        InputNumStr("LS eps", cs.ls_session.curves[0].eps_text, 120);
        InputNumStr("LS NT",  cs.ls_session.curves[0].nt_text,  120);
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

    // Inherit disables ONLY the sweep axis (par target + lo/hi), leaving
    // N and the L1D-specific integrator fields (h/TT/CT below) always
    // editable — L1D is cheap and interactive, so users may want a finer
    // grid or a longer transient than L2D even when sharing the same axis.
    ImGui::TextUnformatted("X-sweep:"); ImGui::SameLine();
    if (inheriting) ImGui::BeginDisabled();
    draw_sweep_target_combo("##sxp", cs.params, cs.vars,
                            c.sweep_x_par_index, c.sweep_x_over_var, c.sweep_x_var_index,
                            c.sweep_x_over_h, &c.sweep_y_over_h);
    InputNumStr("lo##sx", c.sweep_x_lo_text, 120);
    InputNumStr("hi##sx", c.sweep_x_hi_text, 120);
    if (inheriting) ImGui::EndDisabled();
    InputNumStr("N##sx",  c.n_x_1d_text,      120);

    ImGui::TextUnformatted("Y-sweep:"); ImGui::SameLine();
    if (inheriting) ImGui::BeginDisabled();
    draw_sweep_target_combo("##syp", cs.params, cs.vars,
                            c.sweep_y_par_index, c.sweep_y_over_var, c.sweep_y_var_index,
                            c.sweep_y_over_h, &c.sweep_x_over_h);
    InputNumStr("lo##sy", c.sweep_y_lo_text, 120);
    InputNumStr("hi##sy", c.sweep_y_hi_text, 120);
    if (inheriting) ImGui::EndDisabled();
    InputNumStr("N##sy",  c.n_y_1d_text,      120);

    // L1D-specific integrator overrides — order TT before CT (per user
    // convention: transient runs first, then computing time is what's
    // actually sampled).
    ImGui::Separator();
    ImGui::TextUnformatted("L1D integrator (independent of shared):");
    InputNumStr("h##l1d",              c.l1d_h_text,          120);
    InputNumStr("transient time##l1d", c.l1d_transient_text,  120);
    InputNumStr("computing time##l1d", c.l1d_t_max_text,      120);

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
    double fx_lo = parse_num_default(sx.lo_text, 0.0);
    double fx_hi = parse_num_default(sx.hi_text, 1.0);
    double fy_lo = parse_num_default(sy.lo_text, 0.0);
    double fy_hi = parse_num_default(sy.hi_text, 1.0);
    if (fx_hi < fx_lo) std::swap(fx_lo, fx_hi);
    if (fy_hi < fy_lo) std::swap(fy_lo, fy_hi);
    // Discrete slider via index-value trick: SliderScalar<Int> steps by 1
    // (thumb snaps hard, no continuous drift), value = grid index. Format
    // string is a LITERAL world-coord string (no % specifier), so the
    // bubble shows "0.158730" instead of "5". Bubble text is precomputed
    // from the CURRENT idx before SliderScalar runs — during drag it
    // shows the previous frame's snapped world value (1-frame lag on the
    // text only, but the thumb itself always sits on a grid node).
    int n_2d = std::atoi(c.resolution_text.c_str()); if (n_2d < 2) n_2d = 64;
    int n_1d_x = std::atoi(c.n_x_1d_text.c_str());   if (n_1d_x < 2) n_1d_x = 64;
    int n_1d_y = std::atoi(c.n_y_1d_text.c_str());   if (n_1d_y < 2) n_1d_y = 64;
    int n_snap_x = (n_1d_x > n_2d) ? n_1d_x : n_2d;
    int n_snap_y = (n_1d_y > n_2d) ? n_1d_y : n_2d;
    auto idx_from_world = [](double v, double lo, double hi, int n) {
        if (n < 2 || hi <= lo) return 0;
        double step = (hi - lo) / (double)(n - 1);
        int i = (int)std::round((v - lo) / step);
        if (i < 0) i = 0; if (i > n - 1) i = n - 1;
        return i;
    };
    auto world_from_idx = [](int i, double lo, double hi, int n) {
        if (n < 2 || hi <= lo) return lo;
        return lo + (double)i * (hi - lo) / (double)(n - 1);
    };

    // Step-arrows + slider row. Arrows walk idx by ±1 (repeat on hold),
    // slider is the discrete int-index widget from above. Same pattern
    // for X and Y — factored into a lambda.
    auto arrow_row = [&](const char* id,
                         int& idx, int idx_min, int idx_max,
                         double lo, double hi, int n_snap,
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
        std::snprintf(fmt, sizeof(fmt), "%.6g", world_from_idx(idx, lo, hi, n_snap));
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
            fix_value = world_from_idx(idx, lo, hi, n_snap);
        // Arrow clicks fire the same debounce path as slider release —
        // step-changed → immediate commit, no need to wait for release.
        // Bump only the caller-supplied timer so the settled branch can
        // enqueue just the slices that actually depend on this axis.
        if (released || step_changed)
            timer_to_bump = ImGui::GetTime();
    };

    int idx_x = idx_from_world(c.fix_x_value, fx_lo, fx_hi, n_snap_x);
    arrow_row("##fix_x_arr", idx_x, 0, n_snap_x - 1,
              fx_lo, fx_hi, n_snap_x, c.fix_x_value, c.last_fix_x_change_time, "fix X");

    int idx_y = idx_from_world(c.fix_y_value, fy_lo, fy_hi, n_snap_y);
    arrow_row("##fix_y_arr", idx_y, 0, n_snap_y - 1,
              fy_lo, fy_hi, n_snap_y, c.fix_y_value, c.last_fix_y_change_time, "fix Y");

    ImGui::Separator();
    ImGui::TextUnformatted("Enable slices:");
    ImGui::Checkbox("Bif-X", &c.bif1d_x_enabled); ImGui::SameLine();
    ImGui::Checkbox("Bif-Y", &c.bif1d_y_enabled); ImGui::SameLine();
    ImGui::Checkbox("LLE-X", &c.lle1d_x_enabled); ImGui::SameLine();
    ImGui::Checkbox("LLE-Y", &c.lle1d_y_enabled); ImGui::SameLine();
    ImGui::Checkbox("LS-X",  &c.ls1d_x_enabled);  ImGui::SameLine();
    ImGui::Checkbox("LS-Y",  &c.ls1d_y_enabled);

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
            InputNumStr("lo##bx", bc.axis_x_lo_text, 120);
            InputNumStr("hi##bx", bc.axis_x_hi_text, 120);

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
            InputNumStr("lo##by", bc.axis_y_lo_text, 120);
            InputNumStr("hi##by", bc.axis_y_hi_text, 120);
            InputNumStr("N##bxy", bc.n_pts_text,     120);

            ImGui::InputInt("feature1", &bc.feature1);
            ImGui::InputInt("feature2", &bc.feature2);
            InputNumStr("DBSCAN eps", bc.eps_dbscan_text, 120);
        }
    }
}

// ---- Pipeline column (master) ----

const char* level_status_badge(bool enabled, bool in_flight, bool has_result, bool has_error) {
    if (!enabled)    return "off";
    if (in_flight)   return "running";
    if (has_error)   return "error";
    if (has_result)  return "ok";
    return "idle";
}

void draw_pipeline_column(CustomSession& cs, int& selected_level) {
    auto& c = cs.shared;
    ImGui::TextUnformatted("Pipeline");
    ImGui::Separator();

    auto row = [&](int idx, const char* name, bool& enabled, const char* status) {
        // Status shown as a tinted trailing label instead of a [tag] prefix
        // so the level name stays flush-left and readable.
        ImGui::PushID(idx);
        ImGui::Checkbox("##en", &enabled);
        ImGui::SameLine();
        if (ImGui::Selectable(name, selected_level == idx))
            selected_level = idx;
        if (status && status[0] && std::string_view(status) != "idle") {
            ImGui::SameLine();
            ImVec4 col(0.6f, 0.6f, 0.6f, 1.0f);
            std::string_view sv(status);
            if      (sv == "running") col = ImVec4(1.0f, 0.85f, 0.35f, 1.0f);
            else if (sv == "ok")      col = ImVec4(0.5f, 0.9f,  0.5f,  1.0f);
            else if (sv == "error")   col = ImVec4(1.0f, 0.45f, 0.4f,  1.0f);
            else if (sv == "off")     col = ImVec4(0.5f, 0.5f,  0.5f,  1.0f);
            ImGui::TextColored(col, "%s", status);
        }
        ImGui::PopID();
    };

    bool bif2_run = cs.bif_session.in_flight && cs.bif_session.running_diagram_index == 0;
    bool lle2_run = cs.lle_session.in_flight && cs.lle_session.running_curve_index   == 0;
    bool ls2_run  = cs.ls_session.in_flight  && cs.ls_session.running_curve_index    == 0;
    bool level2_running = bif2_run || lle2_run || ls2_run;
    bool level2_hasres  =
        (!cs.bif_session.diagrams.empty() && cs.bif_session.diagrams[0].last_run_2d_ok) ||
        (!cs.lle_session.curves.empty()   && cs.lle_session.curves[0].last_run_2d_ok)   ||
        (!cs.ls_session.curves.empty()    && cs.ls_session.curves[0].last_run_2d_ok);

    row(0, "Level 2D (Bif/LLE/LS)", c.level_2d_enabled,
        level_status_badge(c.level_2d_enabled, level2_running, level2_hasres, false));

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

    row(1, "Level 1D (slices)", c.level_1d_enabled,
        level_status_badge(c.level_1d_enabled, level1_running, level1_hasres, false));

    bool level3_running = cs.phase_session.in_flight || cs.basins_session.in_flight;
    bool level3_hasres  = cs.phase_session.result.ok ||
                          (!cs.basins_session.configs.empty() && cs.basins_session.configs[0].last_run_ok);
    const char* l3_name = c.level3_kind == 0 ? "Level 3 (Phase / Time-domain)" : "Level 3 (Basins)";
    row(2, l3_name, c.level_phase_enabled,
        level_status_badge(c.level_phase_enabled, level3_running, level3_hasres, false));
}

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
        const double debounce_s = 0.5;
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
            auto pin_axis = [&](const EffectiveSweep& e, double v) {
                if (e.over_h) {
                    if (v > 0.0) {
                        char buf[64]; std::snprintf(buf, sizeof(buf), "%.6g", v);
                        c.l1d_h_text = buf;
                    }
                    return;
                }
                if (e.over_var) return;
                if (e.par_index < 0 || e.par_index >= (int)cs.params.size()) return;
                char buf[64]; std::snprintf(buf, sizeof(buf), "%.6g", v);
                c.param_values[cs.params[e.par_index]] = buf;
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
        if (!s.axis_x_over_var && s.axis_x_par_index >= 0 &&
            s.axis_x_par_index < (int)cs.params.size()) {
            char buf[64]; std::snprintf(buf, sizeof(buf), "%.6g", snap_x);
            s.param_values[cs.params[s.axis_x_par_index]] = buf;
        }
        if (!s.axis_y_over_var && s.axis_y_par_index >= 0 &&
            s.axis_y_par_index < (int)cs.params.size()) {
            char buf[64]; std::snprintf(buf, sizeof(buf), "%.6g", snap_y);
            s.param_values[cs.params[s.axis_y_par_index]] = buf;
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
static inline ImGuiID ws_dock_id(int tab_id, const std::string& sys) {
    ImGuiID sys_hash = sys.empty() ? 0u : ImHashStr(sys.c_str());
    return (ImGuiID)0xD5000000u ^ (ImGuiID)tab_id ^ sys_hash;
}

// Suffix appended to every Custom-mode plot window title so imgui.ini keys
// its dock state per system. Empty when no system is loaded (edge case;
// windows behave as before). "##sys_<name>" — the "##" makes ImGui strip
// the suffix from the visible title while including it in the hashed ID.
static inline std::string custom_win_suffix(const std::string& sys) {
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
        auto parse_dbl = [](const std::string& s, double def) -> double {
            if (s.empty()) return def;
            try { return std::stod(s); } catch (...) { return def; }
        };
        EffectiveSweep esx = effective_sweep_x(cs.shared);
        EffectiveSweep esy = effective_sweep_y(cs.shared);
        double x_lo = parse_dbl(esx.lo_text, 0.0), x_hi = parse_dbl(esx.hi_text, 1.0);
        double y_lo = parse_dbl(esy.lo_text, 0.0), y_hi = parse_dbl(esy.hi_text, 1.0);
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
        int cm = cfg_cmap >= 0 ? cfg_cmap : model.heatmap_colormap;
        if (cm >= 0 && cm < kHeatmapColormapCount)
            heatmaps[i].colormap = (HeatmapColormap)cm;
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
                    if (ImGui::MenuItem("Export data...", nullptr, false, !busy)) {
                        if (!cb.pick_save_file_csv) return;
                        std::string path = cb.pick_save_file_csv();
                        if (path.empty()) return;
                        if      (i == 0 && !cs.bif_session.diagrams.empty())
                            data_export::export_bif2d(cs.bif_session.diagrams[0].result_2d, path);
                        else if (i == 1 && !cs.lle_session.curves.empty())
                            data_export::export_lle2d(cs.lle_session.curves[0].result_2d, path);
                        else if (i == 2 && !cs.ls_session.curves.empty())
                            data_export::export_ls2d(cs.ls_session.curves[0].result_2d, path);
                    }
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
        int   snap_n = std::atoi((slice_is_x ? cs.shared.n_x_1d_text
                                             : cs.shared.n_y_1d_text).c_str());
        if (snap_n < 2) snap_n = 2;
        const double snap_lo   = std::min(param_lo, param_hi);
        const double snap_hi   = std::max(param_lo, param_hi);
        const double snap_step = (snap_hi - snap_lo) / (double)(snap_n - 1);
        auto snap_to_grid = [snap_lo, snap_hi, snap_step, snap_n](double w) {
            if (snap_step <= 0.0) return w;
            int i = (int)std::round((w - snap_lo) / snap_step);
            if (i < 0) i = 0; if (i > snap_n - 1) i = snap_n - 1;
            double s = snap_lo + (double)i * snap_step;
            if (s < snap_lo) s = snap_lo; if (s > snap_hi) s = snap_hi;
            return s;
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
                if (p < (int)r.flags.size() && r.flags[p] < 0) continue;
                double px = param_lo + (param_hi - param_lo) * (double)p /
                            (double)(n - 1 > 0 ? n - 1 : 1);
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
                if (p < (int)r.flags.size() && r.flags[p] < 0) continue;
                double y = r.lyapunov[p];
                if (!std::isfinite(y)) continue;
                double px = (n > 1) ? param_lo + (param_hi - param_lo) * (double)p /
                                       (double)(n - 1) : param_lo;
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
                        if (p < (int)r.flags.size() && r.flags[p] < 0) continue;
                        if (p >= (int)r.spectrum.size()) continue;
                        const auto& row = r.spectrum[p];
                        if (j >= (int)row.size()) continue;
                        double px = (n > 1) ? param_lo + (param_hi - param_lo) * (double)p /
                                               (double)(n - 1) : param_lo;
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
            if (!ImGui::MenuItem("Export data...", nullptr, false, !busy)) return;
            if (!cb.pick_save_file_csv) return;
            std::string path = cb.pick_save_file_csv();
            if (path.empty()) return;
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
        draw_projection_windows(cs.phase_session, cb,
            [&](int /*idx*/, const std::string& title) { ensure_docked(title); },
            {}, suffix, sys_owner_delta);
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
                            int cm = (bcfg.colormap_idx[0] >= 0)
                                     ? bcfg.colormap_idx[0] : model.basins_colormap;
                            if (cm < 0 || cm >= kHeatmapColormapCount) cm = 2;
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

    // Индикатор компьюта — справа по правой границе окна, виден во всех режимах.
    // Layout: [text] [progress bar] [Stop] for in-flight cancellable sessions;
    // [text] only for phase or for "Done/Cancelled" persistent state. Stop also
    // drains parametric_queue and basins_queue so remaining batch items
    // don't auto-start.
    enum class BusyKind { None, Bif, LLE, LS, Dft1D, Basins, Phase, FastSync, Custom };
    BusyKind busy_kind = BusyKind::None;
    std::string busy_what;
    std::chrono::steady_clock::time_point busy_start;
    bool busy_cancelling = false;     // user already pressed Stop, waiting for engine
    bool show_done = false;           // not in flight — show persistent last-run info
    bool last_ok = true;              // for show_done: true = green "Done", false = red "Cancelled"
    double done_seconds = 0.0;
    float  progress_fraction = 0.0f;  // 0..1 from session.progress_token
    int    basins_phase = 0;          // 1 = sim, 2 = cluster; 0 = not basins or not started

    if (model.bifurcation_session.in_flight) {
        int ri = model.bifurcation_session.running_diagram_index;
        busy_what = (ri >= 0 && ri < (int)model.bifurcation_session.diagrams.size())
                        ? model.bifurcation_session.diagrams[ri].label : "bifurcation";
        busy_start = model.bifurcation_session.compute_start_time;
        busy_kind = BusyKind::Bif;
        busy_cancelling = model.bifurcation_session.cancel_token &&
                          model.bifurcation_session.cancel_token->load(std::memory_order_relaxed);
        if (model.bifurcation_session.progress_token)
            progress_fraction = model.bifurcation_session.progress_token->load(std::memory_order_relaxed);
    }
    else if (model.lle_session.in_flight) {
        int ri = model.lle_session.running_curve_index;
        busy_what = (ri >= 0 && ri < (int)model.lle_session.curves.size())
                        ? model.lle_session.curves[ri].label : "LLE";
        busy_start = model.lle_session.compute_start_time;
        busy_kind = BusyKind::LLE;
        busy_cancelling = model.lle_session.cancel_token &&
                          model.lle_session.cancel_token->load(std::memory_order_relaxed);
        if (model.lle_session.progress_token)
            progress_fraction = model.lle_session.progress_token->load(std::memory_order_relaxed);
    }
    else if (model.ls_session.in_flight) {
        int ri = model.ls_session.running_curve_index;
        busy_what = (ri >= 0 && ri < (int)model.ls_session.curves.size())
                        ? model.ls_session.curves[ri].label : "LS";
        busy_start = model.ls_session.compute_start_time;
        busy_kind = BusyKind::LS;
        busy_cancelling = model.ls_session.cancel_token &&
                          model.ls_session.cancel_token->load(std::memory_order_relaxed);
        if (model.ls_session.progress_token)
            progress_fraction = model.ls_session.progress_token->load(std::memory_order_relaxed);
    }
    else if (model.dft1d_session.in_flight) {
        int ri = model.dft1d_session.running_config_index;
        busy_what = (ri >= 0 && ri < (int)model.dft1d_session.configs.size() &&
                     !model.dft1d_session.configs[ri].label.empty())
                        ? model.dft1d_session.configs[ri].label
                        : std::string("dft1d");
        busy_start = model.dft1d_session.compute_start_time;
        busy_kind  = BusyKind::Dft1D;
        busy_cancelling = model.dft1d_session.cancel_token &&
                          model.dft1d_session.cancel_token->load(std::memory_order_relaxed);
        if (model.dft1d_session.progress_token)
            progress_fraction = model.dft1d_session.progress_token->load(std::memory_order_relaxed);
    }
    else if (model.phase_session.in_flight) {
        busy_what  = "phase";
        busy_start = model.phase_session.compute_start_time;
        busy_kind  = BusyKind::Phase;
    }
    else if (model.basins_session.in_flight) {
        int ri = model.basins_session.running_config_index;
        busy_what = (ri >= 0 && ri < (int)model.basins_session.configs.size() &&
                     !model.basins_session.configs[ri].label.empty())
                        ? model.basins_session.configs[ri].label
                        : std::string("basins");
        busy_start = model.basins_session.compute_start_time;
        busy_kind  = BusyKind::Basins;
        busy_cancelling = model.basins_session.cancel_token &&
                          model.basins_session.cancel_token->load(std::memory_order_relaxed);
        if (model.basins_session.progress_token)
            progress_fraction = model.basins_session.progress_token->load(std::memory_order_relaxed);
        if (model.basins_session.progress_phase_token)
            basins_phase = model.basins_session.progress_phase_token->load(std::memory_order_relaxed);
    }
    else if (model.fastsync_session.in_flight) {
        int ri = model.fastsync_session.running_config_index;
        busy_what = (ri >= 0 && ri < (int)model.fastsync_session.configs.size() &&
                     !model.fastsync_session.configs[ri].label.empty())
                        ? model.fastsync_session.configs[ri].label
                        : std::string("fastsync");
        busy_start = model.fastsync_session.compute_start_time;
        busy_kind  = BusyKind::FastSync;
        busy_cancelling = model.fastsync_session.cancel_token &&
                          model.fastsync_session.cancel_token->load(std::memory_order_relaxed);
        if (model.fastsync_session.progress_token)
            progress_fraction = model.fastsync_session.progress_token->load(std::memory_order_relaxed);
    }
    else if (model.custom_session.any_in_flight()) {
        // Custom-tab: queue is drained serially, so at most one sub-session is
        // in-flight at any time — pick whichever one it is and surface its
        // label/progress under a "Custom: ..." prefix.
        auto& cs = model.custom_session;
        if (cs.bif_session.in_flight) {
            int ri = cs.bif_session.running_diagram_index;
            busy_what = std::string("Custom: ") +
                        ((ri >= 0 && ri < (int)cs.bif_session.diagrams.size())
                            ? cs.bif_session.diagrams[ri].label : std::string("Bif"));
            busy_start = cs.bif_session.compute_start_time;
            if (cs.bif_session.progress_token)
                progress_fraction = cs.bif_session.progress_token->load(std::memory_order_relaxed);
            busy_cancelling = cs.bif_session.cancel_token &&
                              cs.bif_session.cancel_token->load(std::memory_order_relaxed);
        } else if (cs.lle_session.in_flight) {
            int ri = cs.lle_session.running_curve_index;
            busy_what = std::string("Custom: ") +
                        ((ri >= 0 && ri < (int)cs.lle_session.curves.size())
                            ? cs.lle_session.curves[ri].label : std::string("LLE"));
            busy_start = cs.lle_session.compute_start_time;
            if (cs.lle_session.progress_token)
                progress_fraction = cs.lle_session.progress_token->load(std::memory_order_relaxed);
            busy_cancelling = cs.lle_session.cancel_token &&
                              cs.lle_session.cancel_token->load(std::memory_order_relaxed);
        } else if (cs.ls_session.in_flight) {
            int ri = cs.ls_session.running_curve_index;
            busy_what = std::string("Custom: ") +
                        ((ri >= 0 && ri < (int)cs.ls_session.curves.size())
                            ? cs.ls_session.curves[ri].label : std::string("LS"));
            busy_start = cs.ls_session.compute_start_time;
            if (cs.ls_session.progress_token)
                progress_fraction = cs.ls_session.progress_token->load(std::memory_order_relaxed);
            busy_cancelling = cs.ls_session.cancel_token &&
                              cs.ls_session.cancel_token->load(std::memory_order_relaxed);
        } else if (cs.phase_session.in_flight) {
            busy_what  = "Custom: phase";
            busy_start = cs.phase_session.compute_start_time;
        } else if (cs.basins_session.in_flight) {
            int ri = cs.basins_session.running_config_index;
            busy_what = std::string("Custom: ") +
                        ((ri >= 0 && ri < (int)cs.basins_session.configs.size() &&
                          !cs.basins_session.configs[ri].label.empty())
                            ? cs.basins_session.configs[ri].label : std::string("basins"));
            busy_start = cs.basins_session.compute_start_time;
            if (cs.basins_session.progress_token)
                progress_fraction = cs.basins_session.progress_token->load(std::memory_order_relaxed);
            busy_cancelling = cs.basins_session.cancel_token &&
                              cs.basins_session.cancel_token->load(std::memory_order_relaxed);
        }
        busy_kind = BusyKind::Custom;
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
            busy_what    = *best->label;
            busy_kind    = best->kind;
            show_done    = true;
            last_ok      = best->ok;
            done_seconds = best->secs;
        }
    }

    if (busy_kind != BusyKind::None) {
        char text[200];
        // Phase suffix appears only for basins while running (not on "Cancelling"
        // or "Done" — those reflect overall state, not the current sub-phase).
        const char* phase_suffix = "";
        if (busy_kind == BusyKind::Basins && !show_done && !busy_cancelling) {
            if      (basins_phase == 1) phase_suffix = " (1/2 sim)";
            else if (basins_phase == 2) phase_suffix = " (2/2 cluster)";
        }
        if (show_done) {
            std::snprintf(text, sizeof(text), "%s %s in %.1fs",
                          last_ok ? "Done" : "Cancelled",
                          busy_what.c_str(), done_seconds);
        } else if (busy_cancelling) {
            double secs = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - busy_start).count();
            std::snprintf(text, sizeof(text), "Cancelling %s... %.1fs", busy_what.c_str(), secs);
        } else {
            double secs = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - busy_start).count();
            // Queue suffix: prefer the queue that matches the running session
            // (parametric for Bif/LLE/LS, basins for Basins, fastsync for FastSync).
            size_t queue_n = 0;
            if      (busy_kind == BusyKind::Basins)   queue_n = model.basins_queue.size();
            else if (busy_kind == BusyKind::FastSync) queue_n = model.fastsync_queue.size();
            else if (busy_kind == BusyKind::Dft1D)    queue_n = model.dft1d_queue.size();
            else if (busy_kind == BusyKind::Custom)   queue_n = model.custom_queue.size();
            else                                      queue_n = model.parametric_queue.size();
            if (queue_n > 0)
                std::snprintf(text, sizeof(text), "Computing %s%s... %.1fs (+%zu)",
                              busy_what.c_str(), phase_suffix, secs, queue_n);
            else
                std::snprintf(text, sizeof(text), "Computing %s%s... %.1fs",
                              busy_what.c_str(), phase_suffix, secs);
        }

        const bool show_stop = (busy_kind != BusyKind::Phase) &&
                               !show_done && !busy_cancelling;
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
            float f = progress_fraction;
            if (f < 0.0f) f = 0.0f;
            if (f > 1.0f) f = 1.0f;
            ImGui::ProgressBar(f, ImVec2(bar_w, bar_h), "");
        }
        if (show_stop) {
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.55f, 0.15f, 0.15f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.75f, 0.20f, 0.20f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.90f, 0.25f, 0.25f, 1.0f));
            if (ImGui::SmallButton("Stop")) {
                switch (busy_kind) {
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
            if (!j.empty()) session_from_json(j, model.phase_session);
        }
    }
    if (entering_par && par_need_init) {
        model.start_parametric_analysis();
        if (!model.loaded_name.empty()) {
            std::string j = lib.load_session(model.loaded_name, "_last_parametric");
            if (!j.empty()) session_from_json_parametric(j, model.bifurcation_session);
            std::string jl = lib.load_session(model.loaded_name, "_last_lle");
            if (!jl.empty()) session_from_json_lle(jl, model.lle_session);
            std::string js = lib.load_session(model.loaded_name, "_last_ls");
            if (!js.empty()) session_from_json_ls(js, model.ls_session);
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
            if (!jd.empty()) session_from_json_dft1d(jd, model.dft1d_session);
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
            if (!jb.empty()) session_from_json_basins(jb, model.basins_session);
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
            if (!jf.empty()) session_from_json_fastsync(jf, model.fastsync_session);
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
            if (!jc.empty()) session_from_json_custom(jc, model.custom_session);
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
                AppConfig cfg;
                cfg.ui_scale_override = ui_slider_value;
                cfg.use_builtin_font  = model.use_builtin_font;
                cfg.heatmap_colormap  = model.heatmap_colormap;
                cfg.basins_colormap        = model.basins_colormap;
                cfg.basins_avgpk_colormap  = model.basins_avgpk_colormap;
                cfg.basins_avgint_colormap = model.basins_avgint_colormap;
                cfg.basins_states_colormap = model.basins_states_colormap;
                cfg.tick_precision         = model.tick_precision;
                save_app_config(get_exe_dir_with_sep(), cfg);
            }
            ImGui::SameLine();
            if (ImGui::Button("Auto")) {
                model.ui_scale_override = 0.0f;
                ui_slider_value = model.ui_scale_auto;
                AppConfig cfg;
                cfg.ui_scale_override = 0.0f;
                cfg.use_builtin_font  = model.use_builtin_font;
                cfg.heatmap_colormap  = model.heatmap_colormap;
                cfg.basins_colormap        = model.basins_colormap;
                cfg.basins_avgpk_colormap  = model.basins_avgpk_colormap;
                cfg.basins_avgint_colormap = model.basins_avgint_colormap;
                cfg.basins_states_colormap = model.basins_states_colormap;
                cfg.tick_precision         = model.tick_precision;
                save_app_config(get_exe_dir_with_sep(), cfg);
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
                // Сохраняем ОБА поля чтобы не сбросить override.
                AppConfig cfg;
                cfg.ui_scale_override = model.ui_scale_override;
                cfg.use_builtin_font  = use_builtin;
                cfg.heatmap_colormap  = model.heatmap_colormap;
                cfg.basins_colormap        = model.basins_colormap;
                cfg.basins_avgpk_colormap  = model.basins_avgpk_colormap;
                cfg.basins_avgint_colormap = model.basins_avgint_colormap;
                cfg.basins_states_colormap = model.basins_states_colormap;
                cfg.tick_precision         = model.tick_precision;
                save_app_config(get_exe_dir_with_sep(), cfg);
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
                AppConfig cfg;
                cfg.ui_scale_override = model.ui_scale_override;
                cfg.use_builtin_font  = model.use_builtin_font;
                cfg.heatmap_colormap  = model.heatmap_colormap;
                cfg.basins_colormap        = model.basins_colormap;
                cfg.basins_avgpk_colormap  = model.basins_avgpk_colormap;
                cfg.basins_avgint_colormap = model.basins_avgint_colormap;
                cfg.basins_states_colormap = model.basins_states_colormap;
                cfg.tick_precision         = tp;
                cfg.dark_theme             = model.dark_theme;
                save_app_config(get_exe_dir_with_sep(), cfg);
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
                AppConfig cfg;
                cfg.ui_scale_override = model.ui_scale_override;
                cfg.use_builtin_font  = model.use_builtin_font;
                cfg.heatmap_colormap  = model.heatmap_colormap;
                cfg.basins_colormap        = model.basins_colormap;
                cfg.basins_avgpk_colormap  = model.basins_avgpk_colormap;
                cfg.basins_avgint_colormap = model.basins_avgint_colormap;
                cfg.basins_states_colormap = model.basins_states_colormap;
                cfg.tick_precision         = model.tick_precision;
                cfg.dark_theme             = model.dark_theme;
                save_app_config(get_exe_dir_with_sep(), cfg);
            }
            ImGui::TextDisabled("Color palette for ImGui controls. Plots use their own colormap.");
        }
        ImGui::End();
    }
}