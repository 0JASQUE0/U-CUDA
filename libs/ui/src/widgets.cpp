#include "gui_internal.h"
// Общие поля ввода: текст, числа, списки, LaTeX-превью, цвета НУ, меню отмены.
using namespace ucuda::ui_detail;

namespace ucuda::ui_detail {
extern std::string exe_dir();
}


namespace ucuda::ui_detail {
std::string get_exe_dir_with_sep() {
    std::string d = platform::executable_dir();
    if (!d.empty() && d.back() != '\\' && d.back() != '/') d += "\\";
    return d;
}
}


namespace ucuda::ui_detail {
ImVec4 ic_base_color(int ic_index) {
    return ImPlot::GetColormapColor(ic_index);
}
}


namespace ucuda::ui_detail {
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
}


namespace ucuda::ui_detail {
ImVec4 ic_var_shade(int ic_index, int vi, int nv) {
    return shade_of(ic_base_color(ic_index), vi, nv);
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string combo_var_label(const std::vector<std::string>& vars) {
    if (vars.size() >= 3) return vars[0] + " + pi*" + vars[1] + " + e*" + vars[2];
    if (vars.size() == 2) return vars[0] + " + pi*" + vars[1];
    return vars.empty() ? std::string("x") : vars[0];
}
}


namespace ucuda::ui_detail {
static_assert(::pi == static_cast<numb>(num_parse_detail::kPi),
              "pi in num_parse.h diverged from configCUDA.h");
}


namespace ucuda::ui_detail {
[[nodiscard]] double combo_var_value(const std::vector<double>& pt, int nvars) {
    double v = !pt.empty() ? pt[0] : 0.0;
    if (nvars >= 2 && pt.size() > 1) v += ::pi    * pt[1];
    if (nvars >= 3 && pt.size() > 2) v += ::euler * pt[2];
    return v;
}
}


namespace ucuda::ui_detail {
std::vector<char>& input_scratch(const std::string& str, size_t extra) {
    static thread_local std::vector<char> buf;
    buf.assign(str.begin(), str.end());
    buf.resize(str.size() + extra);
    buf[str.size()] = '\0';
    return buf;
}
}


namespace ucuda::ui_detail {
int input_str_resize(ImGuiInputTextCallbackData* d) {
    if (d->EventFlag == ImGuiInputTextFlags_CallbackResize) {
        std::string* s = static_cast<std::string*>(d->UserData);
        s->resize((size_t)d->BufTextLen);
        d->Buf = s->data();
    }
    return 0;
}
}


namespace ucuda::ui_detail {
bool InputTextMultilineStr(const char* label, std::string& str, const ImVec2& size) {
    return ImGui::InputTextMultiline(label, str.data(), str.capacity() + 1, size,
                                     ImGuiInputTextFlags_CallbackResize, input_str_resize, &str);
}
}


namespace ucuda::ui_detail {
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
}


namespace ucuda::ui_detail {
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
}


namespace ucuda::ui_detail {
void draw_rhs_preview(const SystemEditor& model) {
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
}


namespace ucuda::ui_detail {
bool InputTextStr(const char* label, std::string& str, float width) {
    std::vector<char>& buf = input_scratch(str, 1024);
    if (width > 0) ImGui::SetNextItemWidth(width);
    bool changed = ImGui::InputText(label, buf.data(), buf.size());
    if (changed) str = buf.data();
    return changed;
}
}


namespace ucuda::ui_detail {
bool InputTextStrHint(const char* label, std::string& str, const char* hint,
                             float width) {
    std::vector<char>& buf = input_scratch(str, 1024);
    if (width > 0) ImGui::SetNextItemWidth(width);
    bool changed = ImGui::InputTextWithHint(label, hint, buf.data(), buf.size());
    if (changed) str = buf.data();
    return changed;
}
}


namespace ucuda::ui_detail {
int filter_comma_to_dot(ImGuiInputTextCallbackData* data) {
    if (data->EventChar == ',') data->EventChar = '.';
    return 0;
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool is_numeric_string(const std::string& s) {
    if (s.empty()) return true;
    double v = 0.0;
    return parse_num_checked(s, v);
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool is_custom_scheme(const std::string& scheme_name,
                              const std::vector<CustomScheme>& custom_schemes) {
    for (const auto& cs : custom_schemes) if (cs.name == scheme_name) return true;
    return false;
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool scheme_needs_compiled_body(const std::string& scheme_name,
                              const std::vector<CustomScheme>& custom_schemes) {
    return is_custom_scheme(scheme_name, custom_schemes)
        || parse_extrapolation_name(scheme_name, nullptr)
        || parse_composition_name(scheme_name, nullptr);
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool custom_scheme_uses_symmetry(const std::string& scheme_name,
                                         const std::vector<CustomScheme>& custom_schemes) {
    static const std::regex re(R"(a\s*\[\s*0\s*\])");
    for (const auto& cs : custom_schemes)
        if (cs.name == scheme_name) return std::regex_search(cs.body, re);
    return false;
}
}


namespace ucuda::ui_detail {
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
}


namespace ucuda::ui_detail {
[[nodiscard]] bool scheme_uses_cd_fp_iters(const std::string& scheme_name) {
    ExtrapolationSpec spec;
    if (parse_extrapolation_name(scheme_name, &spec))
        return scheme_uses_cd_fp_iters(spec.base);
    CompositionSpec cspec;
    if (parse_composition_name(scheme_name, &cspec))
        return scheme_uses_cd_fp_iters(cspec.base);
    return scheme_name == "CD FP" || scheme_name == "CD fast FP";
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool scheme_uses_newton(const std::string& scheme_name) {
    ExtrapolationSpec spec;
    if (parse_extrapolation_name(scheme_name, &spec))
        return scheme_uses_newton(spec.base);
    CompositionSpec cspec;
    if (parse_composition_name(scheme_name, &cspec))
        return scheme_uses_newton(cspec.base);
    return scheme_name == "Implicit Euler" || scheme_name == "Implicit Midpoint"
        || scheme_name == "Complex Implicit Euler"
        || dae_scheme_from_name(scheme_name, nullptr)    // DAE BDF1 / Trapezoidal
        || scheme_is_multistep(scheme_name);             // BDF 1-5: корректор — тот же Ньютон
}
}


namespace ucuda::ui_detail {
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
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string sweep_range_label(const std::string& axis,
                                                   const std::string& lo_text,
                                                   const std::string& hi_text) {
    const std::string lo = lo_text.empty() ? std::string("?") : lo_text;
    const std::string hi = hi_text.empty() ? std::string("?") : hi_text;
    return axis + " [" + lo + ".." + hi + "]";
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string label_identity(const std::string& s) {
    const size_t k = s.rfind(" [");
    return k == std::string::npos ? s : s.substr(0, k);
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string auto_label_window(const ParametricPlotWindow& w) {
    const char* kn = w.kind == ParametricPlotWindow::Kind::Bifurcation ? "Bifurcation"
                    : w.kind == ParametricPlotWindow::Kind::LLE ? "LLE"
                    : w.kind == ParametricPlotWindow::Kind::LS  ? "LS" : "Metrics";
    const char* suffix = w.colored_1d ? "Colored 1D" : (w.mode_2d ? "2D" : "1D");
    return std::string(kn) + " " + suffix;
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string auto_label_dft1d(const Dft1DConfig& c,
                                    const std::vector<std::string>& params,
                                    const std::vector<std::string>& vars) {
    return sweep_range_label(auto_axis_name(params, vars, c.param_index, c.sweep_over_var,
                                            c.var_sweep_index, c.sweep_over_h),
                             c.param_lo_text, c.param_hi_text);
}
}


namespace ucuda::ui_detail {
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
}


namespace ucuda::ui_detail {
[[nodiscard]] bool parse_num_ok(const std::string& v, double& out) {
    const double a = parse_ratio_or(v, 0.0);
    const double b = parse_ratio_or(v, 1.0);
    if (a != b) return false;
    out = a;
    return true;
}
}


namespace ucuda::ui_detail {
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
}


namespace ucuda::ui_detail {
int list_digit_step_callback(ImGuiInputTextCallbackData* data) {
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
}


namespace ucuda::ui_detail {
bool InputNumListStr(const char* label, std::string& str, float width ) {
    std::vector<char>& buf = input_scratch(str, 1024);
    if (width > 0) ImGui::SetNextItemWidth(width);
    const bool changed = ImGui::InputText(label, buf.data(), buf.size(),
        ImGuiInputTextFlags_CallbackHistory, list_digit_step_callback);
    if (changed) str = buf.data();
    return changed;
}
}


namespace ucuda::ui_detail {
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
}


namespace ucuda::ui_detail {
void do_undo(AppModel& m) {
    if (m.undo_last()) m.undo_note_time = ImGui::GetTime();
}
}


namespace ucuda::ui_detail {
void do_redo(AppModel& m) {
    if (m.redo_last()) m.undo_note_time = ImGui::GetTime();
}
}


namespace ucuda::ui_detail {
void draw_undo_menu_item(AppModel* m) {
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
}
