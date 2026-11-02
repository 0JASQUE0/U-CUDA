#pragma once
// Вкладка Order: настройки, устойчивость, карты, кривые и экспорт.
#include "gui_common.h"

namespace ucuda::ui_detail {
int builtin_scheme_order(const std::string& name);


void draw_order_controls(AppModel& model, SystemLibrary& /*lib*/);


void draw_order_plot_windows(AppModel& model, const GuiCallbacks& cb);


void order_autosave_seed(AppModel& model);


void draw_order_target_combo(const char* label, int& target,
                                    const OrderAnalysisSession& s) ;


void draw_order_axis_block(const char* id, const char* title,
                                  const OrderAnalysisSession& s,
                                  int& target, std::string& lo, std::string& hi,
                                  bool& log_grid, std::string* n_text,
                                  bool enabled, const char* n_label = "points") ;


void draw_order_steps_hint(const OrderConfig& c, const OrderAnalysisSession& s) ;


void draw_order_stability_block(AppModel& model, const OrderAnalysisSession& s,
                                       OrderConfig& c) ;


OrderPlotWindow::Kind order_kind_for(const OrderConfig& c) ;


void order_rehome_config(AppModel& model, int idx) ;


void order_float_build_warning() ;


const char* order_window_kind_name(OrderPlotWindow::Kind k) ;


bool order_member_fits(const OrderConfig& c, OrderPlotWindow::Kind k) ;


ImVec4 order_shade(ImVec4 c, float k) ;


int order_marker_for(int within) ;


ImVec4 order_base_color(int idx, int palette) ;


void draw_order_status_line(const char* label, int n_ok, int n_div, int n_floor, int n_noc) ;


data_export::OrderSnapshot order_snapshot(const OrderAnalysisSession& s,
                                                 const OrderConfig& c,
                                                 bool two_d, bool gpu_fmad, bool gpu_rdc) ;


void draw_order_export_items(AppModel& model, const OrderPlotWindow& win,
                                    const GuiCallbacks& cb, bool busy) ;


void draw_order_stability_map(AppModel& model, OrderPlotWindow& win,
                                     const GuiCallbacks& cb,
                                     PlotRenderer& renderer, HeatmapView& hv,
                                     OrderConfig& c, int mi) ;


void draw_order_map_window(AppModel& model, OrderPlotWindow& win,
                                  const GuiCallbacks& cb,
                                  PlotRenderer& renderer, HeatmapView& hv) ;


void draw_order_curve_window(AppModel& model, OrderPlotWindow& win,
                                    const GuiCallbacks& cb,
                                    PlotRenderer& renderer, Plot2DView& view) ;
}
