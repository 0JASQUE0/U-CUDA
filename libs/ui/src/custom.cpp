#include "gui_internal.h"

namespace ucuda::ui_detail {


// Custom-tab plot windows: minimal renderer for the pipeline output. Full-featured versions
// with colormap toolbars, colorbar controls and colored-1D density are TODO; this MVP shows
// results and wires drill-down clicks / crosshairs.


// Helper: bind slider fix_x/fix_y as crosshair on the given heatmap view,
// and install a drill-down click callback that updates shared.param_values
// + fix_x/y and optionally enqueues a Phase/Basins run.
void wire_2d_heatmap_interaction(HeatmapView& hv, CustomSession& cs,
                                 std::deque<CustomQueueItem>& q) {
    hv.crosshair_x = cs.shared.level_1d_enabled ? cs.shared.fix_x_value
                                                : std::numeric_limits<double>::quiet_NaN();
    hv.crosshair_y = cs.shared.level_1d_enabled ? cs.shared.fix_y_value
                                                : std::numeric_limits<double>::quiet_NaN();
    // Drag callback: fires every frame while LMB is held inside the plot (HeatmapView
    // on_left_drag). Cheap — just moves the crosshair via fix_x/fix_y so the shared crosshair on
    // every 2D window follows the cursor. Recompute (Phase/Basins enqueue + param_values update)
    // is deferred to on_left_click on release.
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
        // Update shared.param_values so any later Phase/Basins run reads the drilled-down
        // location. Only param-sweeps are pinned; var-sweeps stay as IC edits (handled by the
        // pipeline drainer). snap_x/snap_y — значения узлов от ucuda_node_value (та же функция,
        // что у ядра), поэтому пишем их round-trip форматом: стоявший здесь %.6g ронял точность
        // до 6 цифр, и портрет по клику считался рядом с пикселем, а не в нём.
        // Проверка over_h обязательна: при свипе по шагу over_var == false (выбор
        // "dt (h)" в комбо гасит его), par_index остаётся от прежнего выбора, и без
        // этого условия значение ШАГА писалось в посторонний параметр. Сам шаг
        // подставляется в конфиги из fix_x/fix_y при запуске (pin_swept_h в
        // custom_session.cpp, pin_fixed_h в app_model.cpp), отдельного поля тут не надо.
        if (!s.axis_x_over_var && !s.axis_x_over_h && s.axis_x_par_index >= 0 &&
            s.axis_x_par_index < (int)cs.params.size()) {
            s.param_values[cs.params[s.axis_x_par_index]] = fmt_num_shortest(snap_x);
        }
        if (!s.axis_y_over_var && !s.axis_y_over_h && s.axis_y_par_index >= 0 &&
            s.axis_y_par_index < (int)cs.params.size()) {
            s.param_values[cs.params[s.axis_y_par_index]] = fmt_num_shortest(snap_y);
        }
        if (s.autorun_on_drilldown && s.level_phase_enabled) {
            q.push_back({ s.level3_kind == 0 ? CustomQueueItem::Kind::Phase
                                             : CustomQueueItem::Kind::Basins });
        }
    };
}


 // namespace

// Custom Workspace — split-region layout for Custom AppMode.
//
// Per frame: AppMode radios on top (MainHost); below, Custom Controls panel | splitter |
// tab bar [ Tab 1 | Tab 2 | + ] over a per-tab DockSpace that hosts the plot windows.
//
// Docking model:
//   - Each tab owns one DockSpace with a STABLE id from tab.id via ws_dock_id() — deliberately
//     NOT ImGui::GetID(str), which hashes with the current window-ID stack and would differ
//     inside BeginTabItem vs outside; we submit the same dockspace both ways (KeepAliveOnly
//     before BeginTabBar, real inside BeginTabItem of the active tab).
//   - ALL per-tab dockspaces are submitted every frame with ImGuiDockNodeFlags_KeepAliveOnly so
//     ImGui keeps the nodes alive and windows docked in inactive tabs don't orphan to floating.
//   - Plot windows are docked via DockBuilderDockWindow on first appearance; imgui.ini persists
//     the internal split layout across sessions.
//   - Cross-tab drag&drop: BeginDragDropSource on each plot title bar + BeginDragDropTarget on
//     each tab item → drop calls DockBuilderDockWindow(name, target_ds) and switches tab.

// Stable dockspace ID for a workspace tab: independent of ImGui's ID-stack scope (same id
// inside or outside a window) and unlikely to collide with other IDs. Includes a hash of the
// loaded system name so imgui.ini keeps a separate dock layout per system — otherwise switching
// systems leaked one system's plot placement into another's tabs.
[[nodiscard]] ImGuiID ws_dock_id(int tab_id, const std::string& sys) {
    ImGuiID sys_hash = sys.empty() ? 0u : ImHashStr(sys.c_str());
    return (ImGuiID)0xD5000000u ^ (ImGuiID)tab_id ^ sys_hash;
}


// Suffix appended to every Custom-mode plot window title so imgui.ini keys
// its dock state per system. Empty when no system is loaded (edge case;
// windows behave as before). "##sys_<name>" — the "##" makes ImGui strip
// the suffix from the visible title while including it in the hashed ID.
[[nodiscard]] std::string custom_win_suffix(const std::string& sys) {
    return sys.empty() ? std::string{} : ("##sys_" + sys);
}
}
