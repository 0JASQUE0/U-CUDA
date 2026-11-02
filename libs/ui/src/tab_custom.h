#pragma once
// Вкладка Custom: общая конфигурация, уровни, рабочие области и окна.
#include "gui_common.h"

namespace ucuda::ui_detail {
void draw_custom_mode_layout(AppModel& model, SystemLibrary& lib,
                                    const GuiCallbacks& cb,
                                    ImVec2 area_pos, ImVec2 area_size);


void draw_shared_config(AppModel& model, CustomSession& cs,
                        const std::vector<CustomScheme>& custom_schemes) ;


void draw_level2d_detail(AppModel& model, CustomSession& cs) ;


void draw_level1d_detail(CustomSession& cs) ;


void draw_level3_detail(AppModel& model, CustomSession& cs) ;


void draw_custom_controls(AppModel& model, SystemLibrary& lib) ;


void wire_2d_heatmap_interaction(HeatmapView& hv, CustomSession& cs,
                                 std::deque<CustomQueueItem>& q) ;


[[nodiscard]] ImGuiID ws_dock_id(int tab_id, const std::string& sys) ;


[[nodiscard]] std::string custom_win_suffix(const std::string& sys) ;


void draw_custom_controls_panel(AppModel& model, SystemLibrary& lib,
                                       ImVec2 pos, ImVec2 size) ;


struct WsRenameState {
    int  target_id = 0;              // 0 = no rename in progress.
    char buf[128]  = {};
};


void draw_custom_workspace_panel(CustomSession& cs,
                                        const std::string& sys,
                                        ImVec2 pos, ImVec2 size) ;


void draw_custom_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) ;


inline WsRenameState g_ws_rename;


inline const char* WS_DRAG_PAYLOAD = "CUSTOM_WS_WIN";
}
