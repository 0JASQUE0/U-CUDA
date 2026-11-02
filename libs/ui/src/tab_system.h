#pragma once
// Вкладка System: уравнения, сгенерированный код, Extr/Comp-обёртки, схемы.
#include "gui_common.h"

namespace ucuda::ui_detail {
void draw_generated_code_block(SystemEditor& model, const GuiCallbacks& cb) ;


void draw_wrapper_coefficients(const std::string& name,
                                      const std::vector<CustomScheme>& custom_schemes,
                                      const std::map<std::string, std::string>& param_values,
                                      const char* id) ;


void begin_coeff_column(const char* id, int rows, int max_rows) ;


std::string wrapper_name_field(const char* id, std::string& text,
                                      const std::string& default_name) ;


bool extr_base_needs_half_s(const std::string& nm) ;


void draw_extrapolation_builder(SystemEditor& model) ;


void comp_load_coeffs_from_file(SystemEditor& model, const std::string& path) ;


void draw_composition_builder(SystemEditor& model, const GuiCallbacks& cb) ;


void draw_wrapper_list(SystemEditor& model) ;


void draw_builtin_scheme_checkboxes(SystemEditor& model) ;


void draw_dae_method_checkboxes(SystemEditor& model) ;


void draw_system_tab(SystemEditor& model, const GuiCallbacks& cb) ;


void draw_parameters_tab(SystemEditor& model) ;
}
