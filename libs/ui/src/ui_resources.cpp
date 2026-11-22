#include "ui_resources.h"
#include "ucuda/ui/gui.h"
#include "imgui.h"
std::vector<std::function<void()>>& ui_resource_deleters() {
    static std::vector<std::function<void()>> deleters;
    return deleters;
}
using ProjectionCache = std::map<std::pair<std::uint64_t, int>, ProjectionViews>;
ProjectionViews& projection_views(const Projection& projection, int workspace_owner) {
    auto& view = ui_resource<ProjectionCache, 2000>()[{projection.runtime_id, workspace_owner}];
    view.last_frame = ImGui::GetFrameCount();
    return view;
}
void prune_projection_views() {
    auto& views = ui_resource<ProjectionCache, 2000>();
    for (auto it = views.begin(); it != views.end(); ) {
        if (ImGui::GetFrameCount() - it->second.last_frame > 600) it = views.erase(it);
        else ++it;
    }
}
void release_gui_resources() {
    auto& deleters = ui_resource_deleters();
    for (auto it = deleters.rbegin(); it != deleters.rend(); ++it) (*it)();
    deleters.clear();
}
