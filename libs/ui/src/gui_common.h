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

// Цветной текст (статусы, предупреждения, уровни Custom и Tree), читаемый в
// любой теме: если контраст с фоном окна ниже WCAG 4.5, цвет сдвигается к
// белому на тёмном фоне или к чёрному на светлом, оттенок сохраняется.
// Жёлтый и салатовый, подобранные под тёмную тему, на светлой становятся
// оливковым и зелёным, синий на чёрном — голубым.
// bg — фон под текстом (непрозрачный RGB); по умолчанию фон окна.
inline ImVec4 readable_text(ImVec4 col, ImVec4 bg) {
    auto lin = [](float v) { return v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f); };
    auto lum = [&](const ImVec4& c) { return 0.2126f * lin(c.x) + 0.7152f * lin(c.y) + 0.0722f * lin(c.z); };
    const float lb = lum(bg);
    const bool dark_bg = lb < 0.18f;
    const float t = dark_bg ? 1.0f : 0.0f;
    for (int it = 0; it < 40; ++it) {
        const float lc = lum(col);
        const float r = (std::max(lc, lb) + 0.05f) / (std::min(lc, lb) + 0.05f);
        if (r >= 4.5f) break;
        col.x += (t - col.x) * 0.06f;
        col.y += (t - col.y) * 0.06f;
        col.z += (t - col.z) * 0.06f;
    }
    return col;
}
inline ImVec4 readable_text(ImVec4 col) {
    return readable_text(col, ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
}
// Цветная подпись на кнопке: фон — цвет кнопки поверх фона окна.
inline ImVec4 readable_button_text(ImVec4 col) {
    const ImVec4 w = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);
    const ImVec4 b = ImGui::GetStyleColorVec4(ImGuiCol_Button);
    const ImVec4 bg(b.x * b.w + w.x * (1 - b.w), b.y * b.w + w.y * (1 - b.w), b.z * b.w + w.z * (1 - b.w), 1.0f);
    return readable_text(col, bg);
}

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
    { "Tree",       "Tree",           AppModel::AppMode::Tree,       false },
    { "Order",      "Order",          AppModel::AppMode::Order,      false },
    { "Network",    "Network",        AppModel::AppMode::Network,    false },
    { "Neural",     "Neural",         AppModel::AppMode::Neural,     false },
    { "Settings",   "Settings",       AppModel::AppMode::Settings,   true  },
};


// Target — то, что принимает parse: сама сессия или её сохраняемое состояние (*State).
template <class Session, class Target>
static bool apply_session_json(AppModel& model,
                               const std::string& json,
                               Session& dst,
                               bool (*parse)(const std::string&, Target&),
                               const char* what)
{
    if (json.empty()) return true;
    Session probe;
    if (!parse(json, probe)) {
        model.session_load_warning =
            std::string("saved session '") + what + "' is corrupt - ignored";
        std::printf("[session] %s: parse failed, keeping defaults from record\n", what);
        std::fflush(stdout);
        return false;
    }
    if (!parse(json, dst)) {                 // не должно случаться, см. выше
        model.session_load_warning =
            std::string("saved session '") + what + "' failed to apply";
        return false;
    }
    model.session_load_warning.clear();
    return true;
}
}
