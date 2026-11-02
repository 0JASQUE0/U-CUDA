#pragma once
// Вкладка Phase analysis: настройки и окна проекций (их же рисуют Basins и Custom).
#include "gui_common.h"

namespace ucuda::ui_detail {
std::string phase_axis_name(const PhaseAnalysisSession& s, int ax);


void phase_axis_combo(const char* id, PhaseAnalysisSession& s, int& axis);


void draw_phase_controls(PhaseAnalysisSession& s,
                                std::function<void()> on_reset_defaults,
                                AppModel* bc = nullptr,
                                const GuiCallbacks* cbp = nullptr);


using ProjHookFn = std::function<void(int proj_index, const std::string& title)>;


struct PhaseSeriesStyle {
    ImVec4 color      = ImVec4(1, 1, 1, 1);
    bool   as_point   = false;   // только Phase2D; в time domain всегда линия
    float  point_size = 6.0f;
    int    marker     = (int)PointMarker::Circle;
};


using PhaseStyleFn = std::function<bool(int ic_index, PhaseSeriesStyle& out)>;


struct FeatureClusterParams {
    bool   valid         = false;
    double mult_peak     = 1.0;
    double mult_interval = 1.0;
    double eps           = 0.0;
};


void draw_projection_windows(PhaseAnalysisSession& s, const GuiCallbacks& cb,
                                    const ProjHookFn& before_begin = {},
                                    const ProjHookFn& after_begin  = {},
                                    const std::string& title_suffix = {},
                                    int owner_id_delta = 0,
                                    const PhaseStyleFn& style_fn = {},
                                    const FeatureClusterParams& clust = {});
}
