#pragma once
#include <string>
#include <vector>
struct ParametricPlotWindow {
    enum class Kind { Bifurcation, LLE, LS };
    Kind        kind    = Kind::Bifurcation;
    bool        mode_2d = false;
    // "Colored 1D diagram" (density-хитмапа) — 3-й display kind, только для
    // Bifurcation. Никогда не true одновременно с mode_2d (см.
    // draw_bifurcation_plot). Для LLE/LS остаётся false и не используется.
    bool        colored_1d = false;
    std::string label   = "Plot 1";   // window title, user-editable
    // Если false, label регенерируется из kind/mode_2d каждый кадр
    // ("Bifurcation 1D" и т.п.). Ставится в true при ручной правке,
    // сбрасывается при очистке поля. Дефолт true — старые сохранения
    // без ключа не перезатираем.
    bool        label_is_manual = true;
    // Indices into bifurcation_session.diagrams / lle_session.curves / ls_session.curves (selected
    // by `kind`), filtered to items whose .mode_2d/.colored_1d matches this window's own. Multiple
    // entries = overlay (typical for 1D); 2D/Colored-1D windows are UI-nudged, not hard-capped,
    // toward exactly one member.
    std::vector<int> members;
    // Stable identity for render-side caches (PlotRenderer/Plot2DView/
    // HeatmapView instances keyed by this, not by vector position — position
    // shifts when an earlier window is removed, this doesn't).
    int id = 0;
};

struct Dft1DPlotWindow {
    std::string label = "Plot 1";
    bool        label_is_manual = true;
    std::vector<int> members;
    int id = 0;
};