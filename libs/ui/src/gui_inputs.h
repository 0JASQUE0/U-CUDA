#pragma once
// Поля ввода: текст, числа, списки, LaTeX-превью, цвета НУ, автоподписи, отмена.
#include "gui_common.h"

namespace ucuda::ui_detail {
std::string get_exe_dir_with_sep();


ImVec4 ic_base_color(int ic_index);


ImVec4 shade_of(ImVec4 base, int vi, int nv);


ImVec4 ic_var_shade(int ic_index, int vi, int nv);


[[nodiscard]] std::string combo_var_label(const std::vector<std::string>& vars);


[[nodiscard]] double combo_var_value(const std::vector<double>& pt, int nvars);


std::vector<char>& input_scratch(const std::string& str, size_t extra);


bool InputTextMultilineStr(const char* label, std::string& str, const ImVec2& size);


void draw_resize_handle(const char* id, float& height,
                               float min_h = 40.0f, float max_h = 1200.0f);


void draw_latex_preview(const char* id, const std::string& src, ImVec2 size);


void draw_rhs_preview(const SystemEditor& model);


bool InputTextStr(const char* label, std::string& str, float width = 0.0f);


bool InputTextStrHint(const char* label, std::string& str, const char* hint,
                             float width = 0.0f);

[[nodiscard]] std::string auto_axis_name(const std::vector<std::string>& params,
                                   const std::vector<std::string>& vars,
                                   int param_idx, bool sweep_over_var, int var_idx,
                                   bool sweep_over_h = false);


[[nodiscard]] std::string label_identity(const std::string& s);


void refresh_auto_labels(AppModel& model);


[[nodiscard]] inline double parse_ratio_or(const std::string& v, double def) {
    // Round-trip через numb: интерфейс обязан показывать и отдавать в расчёт
    // РОВНО то число, которое увидит ядро. Сегодня numb == double и это no-op,
    // но если numb станет float — подпись оси, кламп fix-значения и привязка
    // крестика к сетке не начнут врать относительно посчитанного.
    return (double)(numb)parse_num(v, def);
}


[[nodiscard]] inline int parse_int_or(const std::string& v, int def) {
    return parse_num_int(v, def);
}


bool InputNumStr(const char* label, std::string& str, float width = 0.0f,
                        const std::function<void()>& ctx_menu = {},
                        bool allow_empty = false);


bool InputNumStrCommit(const char* label, std::string& text, double& value,
                              float width = 0.0f);


void do_undo(AppModel& m);


void do_redo(AppModel& m);

std::string dot_latex(const std::string& name, const FieldLabels* labels = nullptr) ;


int input_str_resize(ImGuiInputTextCallbackData* d) ;


int filter_comma_to_dot(ImGuiInputTextCallbackData* data) ;


[[nodiscard]] bool is_numeric_string(const std::string& s) ;

[[nodiscard]] std::string sweep_range_label(const std::string& axis,
                                                   const std::string& lo_text,
                                                   const std::string& hi_text) ;


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


[[nodiscard]] std::string auto_label_window(const ParametricPlotWindow& w) ;


[[nodiscard]] std::string auto_label_dft1d(const Dft1DConfig& c,
                                    const std::vector<std::string>& params,
                                    const std::vector<std::string>& vars) ;


[[nodiscard]] bool parse_num_ok(const std::string& v, double& out) ;


int list_digit_step_callback(ImGuiInputTextCallbackData* data) ;


bool InputNumListStr(const char* label, std::string& str, float width = 0.0f) ;


void draw_undo_menu_item(AppModel* m) ;
}
