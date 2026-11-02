#pragma once
// Включения, константы и вкладки режимов: общее для всего интерфейса.
// Порядок заголовков: gui_common.h, gui_inputs.h, gui_integration.h, gui_fields.h, gui_plot.h,
// gui_tabs.h; каждый включает предыдущий. Все вместе подключает gui_internal.h.
#include "ui_resources.h"
#include "ucuda/platform/platform_paths.h"

#include "ucuda/ui/gui.h"
#include "ucuda/equations/dae_ir.h"
#include "ucuda/application/compute_services.h" // проверки, файлы схем, нетлист: без cuda / numerics
#include <fstream>
#include "imgui.h"
#include "imgui_internal.h"   // DockBuilder* API — needed for Custom Workspace per-tab dockspaces.
#include "implot.h"
#include "implot3d.h"
#include "ucuda/plotting/plot_renderer.h"
#include "ucuda/application/session_persistence.h"
#include "ucuda/plotting/plot_view_2d.h"
#include "ucuda/plotting/plot_view_3d.h"
#include "ucuda/plotting/heatmap_view.h"
#include "ucuda/storage/app_config.h"
#include "ucuda/storage/data_export.h"
#include "ucuda/domain/num_parse.h"   // parse_num — единый разбор числовых полей
#include "ucuda/plotting/digit_input.h" // DigitInput::ComputeStep — ↑/↓ по значению в списке (atol)
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>


// Именованные константы вместо магических чисел
// Значение, которым все хитмапы помечают «данных нет» (расходящаяся ячейка).
// HeatmapView красит его в серый; одно и то же число в colored-1D, DFT и Basins.


namespace ucuda::ui_detail {
static constexpr double kSentinelNoData = 999.0;


constexpr int    kColormapTurbo  = slancm_id(168);


static constexpr int    kErrorBoxLines  = 12;


static constexpr int    kRendererEvictFrames = 600;


static constexpr double kBasinsPhaseDebounceSec = 0.3;


static constexpr double kCustomSliderDebounceSec = 0.5;


static constexpr double kLibraryCacheTtlSec = 10.0;


static constexpr float  kFieldW = 120.0f;


static constexpr float  kComboW = 160.0f;

struct ModeTab {
    const char*       id;
    const char*       label;
    AppModel::AppMode mode;
    bool              always;
};


static const ModeTab kModeTabs[] = {
    { "Library",    "Library",        AppModel::AppMode::Library,    true  },
    { "Analysis",   "Phase analysis", AppModel::AppMode::Analysis,   false },
    { "Parametric", "Parametric",     AppModel::AppMode::Parametric, false },
    { "Dft1D",      "1D DFT",         AppModel::AppMode::Dft1D,      false },
    { "Basins",     "Basins",         AppModel::AppMode::Basins,     false },
    { "FastSync",   "Fast Synchro",   AppModel::AppMode::FastSync,   false },
    { "Custom",     "Custom",         AppModel::AppMode::Custom,     false },
    { "Order",      "Order",          AppModel::AppMode::Order,      false },
    { "Network",    "Network",        AppModel::AppMode::Network,    false },
    { "Neural",     "Neural",         AppModel::AppMode::Neural,     false },
    { "Settings",   "Settings",       AppModel::AppMode::Settings,   true  },
};
}
