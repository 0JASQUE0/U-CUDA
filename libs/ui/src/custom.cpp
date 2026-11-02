#include "gui_internal.h"
using namespace ucuda::ui_detail;

namespace ucuda::ui_detail {
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
}

namespace ucuda::ui_detail {
[[nodiscard]] ImGuiID ws_dock_id(int tab_id, const std::string& sys) {
    ImGuiID sys_hash = sys.empty() ? 0u : ImHashStr(sys.c_str());
    return (ImGuiID)0xD5000000u ^ (ImGuiID)tab_id ^ sys_hash;
}
}

namespace ucuda::ui_detail {
[[nodiscard]] std::string custom_win_suffix(const std::string& sys) {
    return sys.empty() ? std::string{} : ("##sys_" + sys);
}
}

namespace ucuda::ui_detail {

}

namespace ucuda::ui_detail {
static void draw_custom_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb);
}

namespace ucuda::ui_detail {

}

namespace ucuda::ui_detail {
[[nodiscard]] TrajWork running_traj_work(const AppModel& m) {
    const CustomSession& cus = m.custom_session;
    if (m.bifurcation_session.in_flight)
        return m.bifurcation_session.traj_work(m.bifurcation_session.running_diagram_index);
    if (m.lle_session.in_flight)     return m.lle_session.traj_work(m.lle_session.running_curve_index);
    if (m.ls_session.in_flight)      return m.ls_session.traj_work(m.ls_session.running_curve_index);
    if (m.metrics_session.in_flight) return m.metrics_session.traj_work(m.metrics_session.running_config_index);
    if (m.dft1d_session.in_flight)   return m.dft1d_session.traj_work(m.dft1d_session.running_config_index);
    if (m.phase_session.in_flight)   return {};
    if (m.basins_session.in_flight)  return m.basins_session.traj_work(m.basins_session.running_config_index);
    if (m.fastsync_session.in_flight || m.order_session.in_flight) return {};
    if (cus.bif_session.in_flight)    return cus.bif_session.traj_work(cus.bif_session.running_diagram_index);
    if (cus.lle_session.in_flight)    return cus.lle_session.traj_work(cus.lle_session.running_curve_index);
    if (cus.ls_session.in_flight)     return cus.ls_session.traj_work(cus.ls_session.running_curve_index);
    if (cus.basins_session.in_flight) return cus.basins_session.traj_work(cus.basins_session.running_config_index);
    return {};
}
}

namespace ucuda::ui_detail {
[[nodiscard]] TrajWork current_tab_traj_work(const AppModel& m) {
    switch (m.app_mode) {
    case AppModel::AppMode::Parametric:
        switch (m.parametric_active_analysis) {
        case 0:  return m.bifurcation_session.traj_work(m.bifurcation_session.active_diagram_index);
        case 1:  return m.lle_session.traj_work(m.lle_session.active_curve_index);
        case 2:  return m.ls_session.traj_work(m.ls_session.active_curve_index);
        case 3:  return m.metrics_session.traj_work(m.metrics_session.active_config_index);
        default: return {};
        }
    case AppModel::AppMode::Dft1D:  return m.dft1d_session.traj_work(m.dft1d_session.active_config_index);
    case AppModel::AppMode::Basins: return m.basins_session.traj_work(m.basins_session.active_config_index);
    default: return {};
    }
}
}
