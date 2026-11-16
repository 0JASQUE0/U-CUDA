#pragma once
// Вкладка Basins: настройки, карты бассейнов и фазовые окна конфигов.
#include "gui_common.h"
#include "tab_phase.h"   // ProjHookFn
#include <optional>

namespace ucuda::ui_detail {
void draw_basins_controls(AppModel& model, SystemLibrary& lib);


void draw_basins_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb);
void draw_basins_plot(AppModel& model, BasinsAnalysisSession& session, int config_index,
                      const GuiCallbacks& cb, unsigned owner_id,
                      const std::function<void()>& save_colormap);
void forget_basins_plot(unsigned owner_id);


void draw_basins_phase_windows(AppModel& model, const GuiCallbacks& cb);
// Окна портретов одного config'а произвольной сессии (лист Basins вкладки Tree).
void draw_basins_phase_windows(AppModel& model, BasinsAnalysisSession& s, int ci, const GuiCallbacks& cb,
                               const std::string& title_suffix, int owner_id_delta,
                               const ProjHookFn& before_begin = {}, const ProjHookFn& after_begin = {},
                               std::optional<unsigned> plot_owner_id = std::nullopt,
                               const ProjVisibilityFn& visible = {});


void basins_phase_tick(AppModel& model);


[[nodiscard]] unsigned basins_base_oid(int config_index) ;


ImVec4 basins_id_color(const BasinsConfig& c, const AppModel& model,
                              int config_index, int display_id,
                              double vmin, double vmax) ;


void draw_basins_phase_controls(AppModel& model, int cfg_idx) ;
void draw_basins_phase_controls(AppModel& model, BasinsAnalysisSession& s, int cfg_idx);
}
