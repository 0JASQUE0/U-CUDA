#pragma once
// Вкладка Basins: настройки, карты бассейнов и фазовые окна конфигов.
#include "gui_common.h"

namespace ucuda::ui_detail {
void draw_basins_controls(AppModel& model, SystemLibrary& lib);


void draw_basins_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb);


void draw_basins_phase_windows(AppModel& model, const GuiCallbacks& cb);


void basins_phase_tick(AppModel& model);


static auto& basins_heatmaps() { return ui_resource<std::map<unsigned, std::unique_ptr<HeatmapView>>, 1000>(); }


void basins_colorbar_range(const BasinsConfig& c, double& vmin, double& vmax) ;


[[nodiscard]] unsigned basins_base_oid(int config_index) ;


ImVec4 basins_id_color(const BasinsConfig& c, const AppModel& model,
                              int config_index, int display_id,
                              double vmin, double vmax) ;


void draw_basins_phase_controls(AppModel& model, int cfg_idx) ;
}
