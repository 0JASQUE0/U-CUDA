#pragma once
#include "ucuda/domain/analysis_session.h"
#include "ucuda/plotting/plot_view_2d.h"
#include "ucuda/plotting/plot_view_3d.h"
#include "ucuda/plotting/heatmap_view.h"
#include <functional>
#include <map>
#include <memory>
#include <vector>

std::vector<std::function<void()>>& ui_resource_deleters();
template<class T, int Tag> T& ui_resource() {
    static std::unique_ptr<T> value;
    if (!value) {
        value = std::make_unique<T>();
        ui_resource_deleters().push_back([] { value.reset(); });
    }
    return *value;
}
struct ProjectionViews {
    std::unique_ptr<Plot2DView> view2d;
    std::unique_ptr<Plot3DView> view3d;
    std::unique_ptr<HeatmapView> viewrp;
    int last_frame = 0;
};
ProjectionViews& projection_views(const Projection& projection);
void prune_projection_views();
