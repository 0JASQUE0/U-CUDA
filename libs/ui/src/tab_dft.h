#pragma once
// Вкладка DFT: настройки, диаграммы и окна графиков.
#include "gui_common.h"

namespace ucuda::ui_detail {
void draw_dft1d_controls(AppModel& model, SystemLibrary& lib);


void draw_dft1d_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb);


void draw_dft1d_diagram_controls(AppModel& model, Dft1DAnalysisSession& s, int idx) ;


void draw_dft1d_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                           Dft1DPlotWindow& win,
                           PlotRenderer& renderer,
                           std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) ;
}
