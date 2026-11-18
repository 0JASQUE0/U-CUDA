#pragma once
// Top-row mode radio buttons of MainHost (labels of kModeTabs in libs/ui/src/gui_common.h).
#include "ucuda/application/app_model.h"

namespace ui_test {

struct ModeButton { const char* label; AppModel::AppMode mode; };

inline constexpr ModeButton kModeButtons[] = {
    { "Library",        AppModel::AppMode::Library },
    { "Phase analysis", AppModel::AppMode::Analysis },
    { "Parametric",     AppModel::AppMode::Parametric },
    { "1D DFT",         AppModel::AppMode::Dft1D },
    { "Basins",         AppModel::AppMode::Basins },
    { "Fast Synchro",   AppModel::AppMode::FastSync },
    { "Custom",         AppModel::AppMode::Custom },
    { "Tree",           AppModel::AppMode::Tree },
    { "Order",          AppModel::AppMode::Order },
    { "Network",        AppModel::AppMode::Network },
    { "Neural",         AppModel::AppMode::Neural },
    { "Settings",       AppModel::AppMode::Settings },
};
static_assert(sizeof(kModeButtons) / sizeof(kModeButtons[0]) == AppModel::kAppModeCount,
              "kModeButtons must list every AppModel::AppMode");

} // namespace ui_test
