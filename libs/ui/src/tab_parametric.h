#pragma once
// Вкладка Parametric: бифуркация, LLE, LS, метрики и окна графиков.
#include "gui_common.h"
#include "gui_plot.h"

namespace ucuda::ui_detail {
void draw_parametric_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb);


void draw_parametric_controls(AppModel& model, SystemLibrary& lib);


void draw_diagram_controls(AppModel& model, BifurcationAnalysisSession& s, int idx) ;


void draw_bifurcation_controls(AppModel& model, SystemLibrary& /*lib*/) ;


ParamPlotKind to_param_plot_kind(ParametricPlotWindow::Kind kind) ;


void configure_plot_view(Plot2DView& view, ParametricPlotWindow::Kind kind) ;


void draw_bifurcation_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                                   ParametricPlotWindow& win,
                                   PlotRenderer& renderer, Plot2DView& view,
                                   std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) ;


void draw_lle_curve_controls(AppModel& model, LLEAnalysisSession& s, int idx) ;


void draw_lle_controls(AppModel& model, SystemLibrary& /*lib*/) ;


void draw_lle_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                          const ParametricPlotWindow& win,
                          PlotRenderer& renderer, Plot2DView& view,
                          std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) ;


void draw_ls_curve_controls(AppModel& model, LyapunovSpectrumAnalysisSession& s, int idx) ;


void draw_ls_controls(AppModel& model, SystemLibrary& /*lib*/) ;


void draw_ls_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                         const ParametricPlotWindow& win,
                         PlotRenderer& renderer, Plot2DView& view,
                         std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) ;


const char* signal_metric_tooltip(int m) ;


void draw_metrics_config_controls(AppModel& model, SignalMetricsAnalysisSession& s, int idx) ;


void draw_metrics_controls(AppModel& model, SystemLibrary& /*lib*/) ;


void draw_metric_tab_bar(AppModel& model, ParametricPlotWindow& win, int avail_mask) ;


void draw_metrics_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                              ParametricPlotWindow& win,
                              PlotRenderer& renderer, Plot2DView& view,
                              std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) ;


struct ParamPlotMatchItem { int index; std::string label; };


[[nodiscard]] std::vector<ParamPlotMatchItem>
parametric_matching_items(AppModel& model, ParametricPlotWindow::Kind kind, bool mode_2d) ;


int parametric_type_index_of(ParametricPlotWindow::Kind kind, bool mode_2d) ;


void parametric_type_from_index(int t, ParametricPlotWindow::Kind& kind, bool& mode_2d) ;


bool draw_parametric_window_setup_row(AppModel& model, ParametricPlotWindow& win) ;


inline const char* kParametricTypeNames[] = {
    "Bifurcation 1D", "Bifurcation 2D", "LLE 1D", "LLE 2D", "LS 1D", "LS 2D",
    "Metrics 1D", "Metrics 2D"
};
}
