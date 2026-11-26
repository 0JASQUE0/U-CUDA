#include "gui_common.h"
#include "gui_inputs.h"
#include "gui_integration.h"
#include "gui_fields.h"
#include "gui_plot.h"
#include "tab_order.h"
#include "ucuda/domain/stability_contour.h"
#include <tuple>
using namespace ucuda::ui_detail;


namespace ucuda::ui_detail {
bool order_member_fits(const OrderConfig& c, OrderPlotWindow::Kind k);
}

namespace ucuda::ui_detail {
void order_float_build_warning() {
    if (sizeof(numb) >= sizeof(double)) return;
    ImGui::TextColored(readable_text(ImVec4(1.0f, 0.5f, 0.3f, 1.0f)),
                       "Warning: this build uses numb = float - dd/qd results may be incorrect");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "U-CUDA was compiled with numb = float (configCUDA.h). double-double and\n"
            "quad-double rebuild the KRS body in extended precision, but the code\n"
            "generator, the host side and everything stored in numb still go through\n"
            "float, so the extra digits may be lost or the result may be wrong.\n"
            "For order estimates and reference solutions rebuild with numb = double.");
}
}

namespace ucuda::ui_detail {
const char* order_window_kind_name(OrderPlotWindow::Kind k) {
    switch (k) {
        case OrderPlotWindow::Kind::P:     return "Order p";
        case OrderPlotWindow::Kind::Error: return "Error";
        case OrderPlotWindow::Kind::Map:   return "Map 2D";
        case OrderPlotWindow::Kind::Perf:  return "Performance";
        case OrderPlotWindow::Kind::ErrT:  return "Error(t)";
        case OrderPlotWindow::Kind::Bounds: return "Stability bounds";
    }
    return "Order";
}
}

namespace ucuda::ui_detail {
bool order_member_fits(const OrderConfig& c, OrderPlotWindow::Kind k) {
    switch (k) {
        case OrderPlotWindow::Kind::Perf: return c.calc_kind == kOrderCalcPerf;
        case OrderPlotWindow::Kind::ErrT: return c.calc_kind == kOrderCalcErrT;
        case OrderPlotWindow::Kind::Bounds: return c.calc_kind == kOrderCalcStab;
        case OrderPlotWindow::Kind::Map:  return c.calc_kind == kOrderCalcStab
                                              || (c.calc_kind == kOrderCalcOrder && c.two_d);
        default:                          return c.calc_kind == kOrderCalcOrder && !c.two_d;
    }
}
}

namespace ucuda::ui_detail {
ImVec4 order_shade(ImVec4 c, float k) {
    auto mix = [](float v, float to, float t) { return v + (to - v) * t; };
    auto ch  = [&](float v) {
        const float t = std::min(1.0f, std::fabs(k - 1.0f));
        return std::min(1.0f, std::max(0.0f, (k < 1.0f) ? mix(v, 0.0f, t) : mix(v, 1.0f, t)));
    };
    return ImVec4(ch(c.x), ch(c.y), ch(c.z), c.w);
}
}

namespace ucuda::ui_detail {
int order_marker_for(int within) {
    static const int kSeq[3] = { (int)PointMarker::Circle,
                                 (int)PointMarker::TriangleDown,
                                 (int)PointMarker::TriangleUp };
    return kSeq[(within < 0 ? 0 : within) % 3];
}
}

namespace ucuda::ui_detail {
static const char* const kOrderPaletteNames[3] = {
    "App palette", "High contrast", "Colorblind-safe"
};
}

namespace ucuda::ui_detail {
ImVec4 order_base_color(int idx, int palette) {
    static const ImU32 kContrast[12] = {
        IM_COL32(0xE6, 0x19, 0x4B, 255),   // алый
        IM_COL32(0x43, 0x63, 0xD8, 255),   // синий
        IM_COL32(0x3C, 0xB4, 0x4B, 255),   // зелёный
        IM_COL32(0xF5, 0x82, 0x31, 255),   // оранжевый
        IM_COL32(0x91, 0x1E, 0xB4, 255),   // фиолетовый
        IM_COL32(0x00, 0xA0, 0xB0, 255),   // бирюзовый
        IM_COL32(0xF0, 0x32, 0xE6, 255),   // пурпурный
        IM_COL32(0x8B, 0x69, 0x14, 255),   // охра
        IM_COL32(0x7F, 0xBF, 0x00, 255),   // травяной
        IM_COL32(0x00, 0x77, 0xBB, 255),   // стальной
        IM_COL32(0xA0, 0x52, 0x2D, 255),   // сиена
        IM_COL32(0x60, 0x60, 0x60, 255),   // серый
    };
    static const ImU32 kOkabe[8] = {
        IM_COL32(0xE6, 0x9F, 0x00, 255),
        IM_COL32(0x56, 0xB4, 0xE9, 255),
        IM_COL32(0x00, 0x9E, 0x73, 255),
        IM_COL32(0xF0, 0xE4, 0x42, 255),
        IM_COL32(0x00, 0x72, 0xB2, 255),
        IM_COL32(0xD5, 0x5E, 0x00, 255),
        IM_COL32(0xCC, 0x79, 0xA7, 255),
        IM_COL32(0x55, 0x55, 0x55, 255),
    };
    if (idx < 0) idx = 0;
    if (palette == kOrderPaletteContrast)
        return ImGui::ColorConvertU32ToFloat4(kContrast[(size_t)(idx % 12)]);
    if (palette == kOrderPaletteOkabe)
        return ImGui::ColorConvertU32ToFloat4(kOkabe[(size_t)(idx % 8)]);
    return ic_base_color(idx);
}
}

namespace ucuda::ui_detail {
void draw_order_status_line(const char* label, int n_ok, int n_div, int n_floor, int n_noc) {
    ImGui::TextDisabled("%s:", label);
    ImGui::SameLine(); ImGui::Text("ok %d", n_ok);
    ImGui::SameLine(); ImGui::TextColored(readable_text(ImVec4(0.6f, 0.6f, 0.6f, 1.0f)), "| diverged %d", n_div);
    ImGui::SameLine(); ImGui::TextColored(readable_text(ImVec4(0.55f, 0.75f, 1.0f, 1.0f)), "| floor %d", n_floor);
    ImGui::SameLine(); ImGui::TextColored(readable_text(ImVec4(1.0f, 0.75f, 0.4f, 1.0f)), "| p<=0 %d", n_noc);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "floor    - E2 sank into the machine precision of the solution: the right\n"
            "           plateau, the order is not measurable any further.\n"
            "p<=0     - the differences stopped contracting: loss of\n"
            "           stability, the left edge of the working window in h.\n"
            "diverged - nan/inf or |X| > max value; such cells are NaN.");
}
}

namespace ucuda::ui_detail {
data_export::OrderSnapshot order_snapshot(const OrderAnalysisSession& s,
                                                 const OrderConfig& c,
                                                 bool two_d, bool gpu_fmad, bool gpu_rdc) {
    data_export::OrderSnapshot sn;
    sn.scheme      = c.scheme;
    sn.axis_x_name = s.axis_target_label(c.axis_x_target);
    sn.axis_y_name = two_d ? s.axis_target_label(c.axis_y_target) : std::string();
    sn.var_names   = s.vars;
    sn.param_names = s.params;
    for (const auto& v : s.vars) {
        auto it = c.initial_conditions.find(v);
        sn.initial_conditions.push_back(it != c.initial_conditions.end() ? parse_num(it->second, 0.0) : 0.0);
    }
    sn.values.push_back(parse_num(c.symmetry_s, 0.5));
    for (const auto& p : s.params) {
        auto it = c.param_values.find(p);
        sn.values.push_back(it != c.param_values.end() ? parse_num(it->second, 0.0) : 0.0);
    }
    scheme_append_tail(sn.values, c.cd_fp_iters, c.newton);   // a[M+1..M+4], как в build_order_request
    sn.h             = parse_num(c.h_text, 0.01);
    sn.t_max         = parse_num(c.t_max_text, 10.0);
    sn.max_value     = parse_num(c.max_value_text, 1.0e6);
    sn.snap_steps    = c.snap_steps;
    sn.endpoint_only = c.endpoint_only;
    sn.use_gpu       = c.use_gpu;
    sn.gpu_fmad      = gpu_fmad;
    sn.gpu_rdc       = gpu_rdc;
    // Режим замера пишем ФАКТИЧЕСКИЙ, из результата: поля вкладки могли
    // измениться уже после прогона.
    sn.repeats  = c.perf_result.repeats;
    sn.warmup   = c.perf_result.warmup;
    sn.replicas = c.perf_result.replicas;
    // Эталон пишем только когда он в результате ЕСТЬ: поля вкладки могли
    // измениться после прогона, а файл описывает прогон, а не поля.
    const bool had_ref = std::any_of(c.perf_result.e_ref.begin(), c.perf_result.e_ref.end(),
                                     [](double v) { return std::isfinite(v) && v > 0.0; });
    if (had_ref) {
        sn.ref_scheme   = c.perf_ref_scheme;
        sn.ref_substeps = std::max(0, (int)parse_num(c.perf_ref_substeps_text, 4.0));
    }
    // Настройки шага — тоже из прогона (у адаптивного Eref нет вовсе, только E(T)).
    if (c.perf_result.adaptive) sn.adaptive_desc = c.perf_result.ad_desc;
    // Настройки теста устойчивости — тоже фактические, из результата; в снимке
    // остаётся лишь раскладка слотов, которой в результате нет.
    sn.stab_k = c.stab_result.k;
    sn.stab_r = c.stab_result.r;
    sn.stab_h = c.stab_result.h;
    sn.stab_pref_tol = parse_num(c.stab_pref_tol_text, 1.0);
    {
        const int slots[4] = { c.stab_idx_a, c.stab_idx_b, c.stab_idx_c, c.stab_idx_d };
        const char* names[4] = { "a", "b", "c", "d" };
        std::string acc;
        for (int i = 0; i < 4; ++i) {
            if (i) acc += ", ";
            acc += names[i];
            acc += " = ";
            const int pi = slots[i] - 1;
            if (pi >= 0 && pi < (int)s.params.size()) acc += s.params[(size_t)pi] + " ";
            acc += "a[" + std::to_string(slots[i]) + "]";
        }
        sn.stab_slots = acc;
    }
    return sn;
}
}

namespace ucuda::ui_detail {
void draw_order_export_items(AppModel& model, const OrderPlotWindow& win,
                                    const GuiCallbacks& cb, bool busy) {
    const OrderAnalysisSession& s = model.order_session;
    const bool perf = (win.kind == OrderPlotWindow::Kind::Perf);
    if (win.kind == OrderPlotWindow::Kind::Bounds) {
        // Сами линии окна: те границы, что сейчас выбраны в "show", всех членов.
        if (ImGui::MenuItem("Export boundaries...", nullptr, false, !busy) && cb.pick_save_file_csv) {
            const std::string path = cb.pick_save_file_csv();
            if (!path.empty()) {
                std::vector<StabilityContour> keep;
                keep.reserve(win.members.size() * 2);
                std::vector<data_export::StabBoundExport> items;
                for (int mi : win.members) {
                    if (mi < 0 || mi >= (int)s.configs.size()) continue;
                    const OrderConfig& c = s.configs[(size_t)mi];
                    if (c.calc_kind != kOrderCalcStab || !c.stab_last_run_ok) continue;
                    const double tol = parse_num(c.stab_pref_tol_text, 1.0);
                    for (int which = 0; which < 2; ++which) {
                        if (win.bound_which != 2 && win.bound_which != which) continue;
                        keep.push_back(stability_contour(c.stab_result, which, tol));
                        data_export::StabBoundExport e;
                        e.tab = c.label;
                        e.scheme = c.scheme;
                        e.boundary = which == kStabBoundPreference ? "preference" : "stability";
                        e.tolerance = tol;
                        items.push_back(e);
                    }
                }
                for (size_t k = 0; k < items.size(); ++k) items[k].xy = &keep[k].xy;
                data_export::export_stability_bounds(items, path);
            }
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("The lines of this window as CSV: tab, scheme, boundary, tolerance,\n"
                              "line, sigma, omega - ready to redraw in MATLAB or pgfplots.");
        ImGui::Separator();
    }
    for (int mi : win.members) {
        if (mi < 0 || mi >= (int)s.configs.size()) continue;
        const OrderConfig& c = s.configs[(size_t)mi];
        const bool stab = (c.calc_kind == kOrderCalcStab);
        const bool errt = (win.kind == OrderPlotWindow::Kind::ErrT);
        const bool has = errt ? c.errt_last_run_ok
                       : stab ? c.stab_last_run_ok : (perf ? c.perf_last_run_ok : c.last_run_ok);
        const std::string item = "Export " + c.label + "...";
        if (!ImGui::MenuItem(item.c_str(), nullptr, false, has && !busy)) continue;
        if (!cb.pick_save_file_csv) continue;
        const std::string path = cb.pick_save_file_csv();
        if (path.empty()) continue;
        const data_export::OrderSnapshot sn =
            order_snapshot(s, c, win.kind == OrderPlotWindow::Kind::Map,
                           model.nvrtc_fmad, model.nvrtc_rdc);
        if (errt) {
            data_export::OrderSnapshot se = sn;
            se.ref_scheme = c.errt_result.ref_is_half ? std::string() : c.errt_ref_scheme;
            se.endpoint_only = false;
            data_export::export_errt(c.errt_result, se, path);
        }
        else if (stab) data_export::export_stability(c.stab_result, sn, path);
        else if (perf) data_export::export_perf(c.perf_result, sn, path);
        else           data_export::export_order(c.result, sn, path);
    }
}
}

namespace ucuda::ui_detail {
void draw_order_stability_map(AppModel& model, OrderPlotWindow& win,
                                     const GuiCallbacks& cb,
                                     PlotRenderer& renderer, HeatmapView& hv,
                                     OrderConfig& c, int mi) {
    OrderAnalysisSession& s = model.order_session;
    const StabilityResult& r = c.stab_result;
    // Строки над картой — тулбар: скрытый в её меню (Show → Toolbar) прячет их все, карта
    // занимает освободившуюся высоту. Настройки вью ниже применяются независимо от этого.
    toolbar_row([&]() {
    ImGui::TextDisabled("show:"); ImGui::SameLine();
    ImGui::RadioButton("1 - rho##stabview", &win.stab_view, 1); ImGui::SameLine();
    ImGui::RadioButton("rho##stabview", &win.stab_view, 0); ImGui::SameLine();
    ImGui::RadioButton("region##stabview", &win.stab_view, 2); ImGui::SameLine();
    ImGui::RadioButton("log10 err##stabview", &win.stab_view, 3);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "1 - rho   - positive means stable. The default, because this is the\n"
            "            quantity the reference MATLAB script plots, and the\n"
            "            colour scale is capped at [0, 1] the same way its\n"
            "            contour levels are: everything unstable collapses into\n"
            "            the bottom band, which makes the rho = 1 boundary a\n"
            "            sharp edge instead of a gradient.\n"
            "rho       - the spectral radius itself, scaled over the computed\n"
            "            range: use it to see HOW unstable the outside is.\n"
            "region    - the plain mask rho <= 1: the stability region itself.\n"
            "log10 err - the relative error of one step against the exact\n"
            "            operator, ||e^{hA} - R|| / ||e^{hA}||: the field the\n"
            "            preference region is cut from.\n"
            "\n"
            "The colour range of any of them can be pinned by hand in the\n"
            "toolbar below.");

    // Цвета слоя берутся у самой вью: образец рядом с чекбоксом обязан
    // совпадать с тем, что нарисовано на карте.
    ImGui::Checkbox("preference region##stabpref", &win.stab_show_pref);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Overlay the preference region on the map: cells with rho <= 1 AND\n"
            "a one-step relative error ||e^{hA} - R|| / ||e^{hA}|| not above the\n"
            "tolerance set in \"Stability region\" (Fedoseev et al., 2022).\n"
            "The stability region says the solution does not blow up; the\n"
            "preference region says it also still resembles the exact one.\n"
            "For A-stable schemes the difference is dramatic: the whole left\n"
            "half-plane is stable, only a small part of it is preferred.");
    });   // bar
    // Цвета слоя живут на окне (сохраняются вместе с ним) и отдаются вью каждый
    // кадр; перезаливку текстуры при смене цвета вью делает сама.
    hv.overlay_fill_color = win.stab_pref_fill;
    hv.overlay_edge_color = win.stab_pref_edge;
    hv.overlay_edge_width = win.stab_pref_edge_w;
    toolbar_row([&]() {
    ImGui::SameLine();
    // Образец — контур поверх заливки, как область и выглядит на карте.
    if (ImGui::ColorButton("##stabpref_swatch", ImGui::ColorConvertU32ToFloat4(win.stab_pref_edge),
                           ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_AlphaPreviewHalf,
                           ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
        ImGui::OpenPopup("stabpref_colors");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Colour of the preference region - click to change.");
    if (ImGui::BeginPopup("stabpref_colors")) {
        // Редактор работает с float[4]; источник правды — IM_COL32 на окне,
        // обратно он пишется только при правке.
        auto edit = [](const char* label, unsigned& col) {
            ImVec4 v = ImGui::ColorConvertU32ToFloat4(col);
            float f[4] = { v.x, v.y, v.z, v.w };
            if (ImGui::ColorEdit4(label, f, ImGuiColorEditFlags_AlphaBar
                                          | ImGuiColorEditFlags_AlphaPreviewHalf))
                col = ImGui::ColorConvertFloat4ToU32(ImVec4(f[0], f[1], f[2], f[3]));
        };
        edit("fill##stabpref", win.stab_pref_fill);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Inside of the region. Keep the alpha low: the map\n"
                              "underneath has to stay readable.");
        edit("edge##stabpref", win.stab_pref_edge);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Cells on the boundary of the region (a neighbour is\n"
                              "outside it) - the outline.");
        item_w(ImGui::CalcItemWidth());
        ImGui::SliderInt("edge width##stabpref", &win.stab_pref_edge_w, 0, 10, "%d cells",
                         ImGuiSliderFlags_AlwaysClamp);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Thickness of the outline in cells of the grid: a cell\n"
                              "of the region is painted with the edge colour when it is\n"
                              "at most this many steps from a cell outside. 0 = fill only.\n"
                              "Counted in cells, not pixels, so the outline stays glued\n"
                              "to the grid at any zoom; at a coarse grid 1 is already thick.");
        if (ImGui::Button("Reset##stabpref")) {
            const OrderPlotWindow defaults;
            win.stab_pref_fill   = defaults.stab_pref_fill;
            win.stab_pref_edge   = defaults.stab_pref_edge;
            win.stab_pref_edge_w = defaults.stab_pref_edge_w;
        }
        ImGui::EndPopup();
    }

    ImGui::SameLine();
    ImGui::TextDisabled("|"); ImGui::SameLine();
    ImGui::Checkbox("paper colours##stabpaper", &win.stab_paper_cmap);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Colour the map the way the figures of Fedoseev et al. (2022) do:\n"
            "a grey scale from black to white over the colour range, and WHITE\n"
            "for everything outside it - below as well as above.\n"
            "\n"
            "On the \"1 - rho\" view (range [0, 1]) this gives a white unstable\n"
            "outside, a black rho = 1 boundary and an inside that lightens\n"
            "towards rho = 0: the darker, the closer to losing stability.\n"
            "\n"
            "While this is on, the colormap chosen in the toolbar is not used\n"
            "(it is kept and comes back when you turn this off).");
    });   // bar
    // Раскраска отдаётся вью каждый кадр: HeatmapView общий для всех видов
    // окна, и снятая галочка обязана вернуть пользовательскую палитру сразу.
    // Серая шкала — slanCM #28 "gray", чистое чёрное -> белое.
    hv.colormap_override = win.stab_paper_cmap ? slancm_id(28) : 0;
    hv.oor_custom        = win.stab_paper_cmap;
    for (int q = 0; q < 3; ++q) hv.oor_below[q] = hv.oor_above[q] = 1.0f;

    if (!c.stab_last_run_ok || r.rho.empty()) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }

    // Маска предпочтительности нужна и для счётчика, поэтому строится всегда, а
    // не только при включённом слое. Допуск читается с вкладки каждый кадр:
    // в расчёт он не входит, и его правка обязана доходить до карты сразу.
    const double pref_tol = parse_num(c.stab_pref_tol_text, 1.0);
    std::vector<unsigned char> pref_mask(r.rho.size(), 0);
    int n_pref = 0;
    for (size_t i = 0; i < r.rho.size(); ++i)
        if (stability_cell_preferred(r, i, pref_tol)) { pref_mask[i] = 1; ++n_pref; }

    toolbar_row([&]() {
    ImGui::SameLine();
    ImGui::TextDisabled("|"); ImGui::SameLine();
    const int n_cells = r.n_ok + r.n_bad;
    ImGui::Text("%s: %d stable, %d preferred (err <= %g) of %d",
                c.label.c_str(), r.n_stable, n_pref, pref_tol, n_cells);
    if (r.n_bad > 0) {
        ImGui::SameLine();
        ImGui::TextColored(readable_text(ImVec4(1.0f, 0.55f, 0.3f, 1.0f)), "(%d not computed)", r.n_bad);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Cells where the test matrix could not be built or the step\n"
                              "returned nan/inf.");
    }
    ImGui::TextDisabled("k = %g, r = %g, h = %g; the boundary is rho = 1", r.k, r.r, r.h);
    });   // bar

    {
        HeatmapToolbarOpts topts;
        topts.persist_colormap = [&win](int cm) { win.colormap_idx = cm; };
        draw_heatmap_toolbar(hv, topts);
    }

    // Потолок шкалы. Неустойчивые углы карты уходят в rho на много порядков, и
    // без ограничения вся рабочая окрестность границы становится одноцветной.
    // Ячейки выше потолка не пропадают — упираются в край палитры, ровно как
    // на карте порядка.
    const double rho_cap = (r.rho_max > 2.0) ? 2.0 : r.rho_max;

    std::vector<double> buf(r.rho.size());
    double vmin = 0.0, vmax = 1.0;
    if (win.stab_view == 2) {
        for (size_t i = 0; i < r.rho.size(); ++i)
            buf[i] = std::isfinite(r.rho[i]) ? (r.rho[i] <= 1.0 ? 1.0 : 0.0)
                                             : std::numeric_limits<double>::quiet_NaN();
        vmin = 0.0; vmax = 1.0;
    } else if (win.stab_view == 1) {
        for (size_t i = 0; i < r.rho.size(); ++i) buf[i] = 1.0 - r.rho[i];
        // Диапазон ровно тот, что задают уровни contourf(X, Y, 1-Rh, [1 0:0.01:1])
        // в исходном скрипте: шкала кончается на нуле, и всё неустойчивое
        // (1 - rho < 0) ложится в нижнюю полосу одним цветом. Это не потеря
        // данных, а то, ради чего картинка и строится: граница rho = 1
        // становится краем палитры, а не серединой градиента. Кому нужна
        // глубина неустойчивости — рядом есть вид rho.
        vmin = 0.0;
        vmax = 1.0;
    } else if (win.stab_view == 3) {
        // Точный ноль ошибки бывает (в начале координат шаг любой схемы — это
        // I), и log10 превратил бы его в -inf, то есть в «нет данных». Пол
        // 1e-17 — ниже eps double, отличить его от настоящего нуля нечем.
        // +inf (точная экспонента выродилась) остаётся нечисловым и красится
        // как непосчитанная ячейка. Потолок 1e2 — по той же причине, что rho_cap:
        // иначе дальний край карты съедает всю шкалу, а интересна окрестность
        // допуска.
        const double floor_err = 1e-17;
        for (size_t i = 0; i < r.rho.size(); ++i) {
            const double e = (i < r.err.size()) ? r.err[i] : std::numeric_limits<double>::quiet_NaN();
            buf[i] = std::isfinite(e) ? std::log10(std::max(e, floor_err))
                                      : std::numeric_limits<double>::quiet_NaN();
        }
        vmin = std::log10(std::max(r.err_min, floor_err));
        vmax = std::min(std::log10(std::max(r.err_max, floor_err)), 2.0);
    } else {
        buf = r.rho;
        vmin = r.rho_min;
        vmax = rho_cap;
    }
    if (!(vmax > vmin)) { vmin -= 0.5; vmax += 0.5; }

    // Слой предпочтительности. Поколение текстуры слоя меняется и с данными,
    // и с допуском: допуск правится на лету, и без него в ключе слой остался
    // бы от старого порога.
    if (win.stab_show_pref) {
        std::uint64_t tol_bits = 0;
        std::memcpy(&tol_bits, &pref_tol, sizeof(tol_bits));
        hv.overlay_mask = pref_mask.data();
        hv.overlay_generation = (int)((unsigned)c.stab_data_generation * 1000003u
                                      ^ (unsigned)(tol_bits ^ (tol_bits >> 32)));
    }

    hv.x_axis.name = "sigma";
    hv.y_axis.name = "omega";
    hv.x_axis.log_scale = false;
    hv.y_axis.log_scale = false;

    const int sig = c.stab_data_generation * 8 + win.stab_view;
    bool fit = c.stab_fit_request;
    if (win.plot_sig != sig) { fit = true; win.plot_sig = sig; }
    if (c.stab_fit_request) c.stab_fit_request = false;

    // Ctrl+T меняет сами границы сетки: у этой карты оси и есть диапазоны расчёта.
    const std::vector<ViewRangeTarget> vrt = {
        vr_target(s.configs, mi,
                  &OrderConfig::stab_sig_lo_text, &OrderConfig::stab_sig_hi_text,
                  &OrderConfig::stab_om_lo_text,  &OrderConfig::stab_om_hi_text) };
    hv.popup_extras = [&vrt, &model, &hv, &win, &cb, &s]() {
        draw_view_range_menu(vrt, hv.x_axis, hv.y_axis, hv.swap_axes, &model);
        ImGui::Separator();
        draw_order_export_items(model, win, cb, s.in_flight);
    };

    ImVec2 avail  = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    hv.render(renderer, origin, avail,
              /*owner_id*/ 0x0BDE0000 + win.id,
              sig,
              r.n_pts_x, r.n_pts_y, buf.data(),
              r.sigma_lo, r.sigma_hi, r.omega_lo, r.omega_hi,
              vmin, vmax, fit);
    handle_view_range_keys(vrt, hv.x_axis, hv.y_axis, plot_window_active(), hv.swap_axes, &model);
}
}

namespace ucuda::ui_detail {
namespace {
// Ломаные границ одного члена окна. Ключ — (окно, вкладка, какая граница);
// пересчёт — при новом прогоне вкладки или (у предпочтительности) смене допуска.
struct OrderBoundsEntry {
    int gen = -1;
    const void* data = nullptr;   // rho.data(): вкладку на этом месте могли заменить
    double tol = std::numeric_limits<double>::quiet_NaN();
    StabilityContour contour;
};
using OrderBoundsCache = std::map<std::tuple<int, int, int>, OrderBoundsEntry>;
OrderBoundsCache& order_bounds_caches() { return ui_resource<OrderBoundsCache, 3029>(); }

const char* const kBoundDashNames[4] = { "solid", "dashed", "dotted", "long dash" };

// Штрихи в px, пропорционально толщине: на толстой линии мелкий пунктир
// сливается в сплошную.
void bound_dash_pattern(int style, float width, float& period, float& on) {
    const float u = std::max(width, 1.0f);
    switch (style) {
        case 1:  period = 6.0f * u;  on = 3.5f * u; break;
        case 2:  period = 3.0f * u;  on = 1.2f * u; break;
        case 3:  period = 12.0f * u; on = 8.5f * u; break;
        default: period = 0.0f;      on = 0.0f;     break;
    }
}
}  // namespace

void draw_order_bounds_window(AppModel& model, OrderPlotWindow& win,
                              const GuiCallbacks& cb,
                              PlotRenderer& renderer, Plot2DView& view) {
    OrderAnalysisSession& s = model.order_session;
    auto& cache = order_bounds_caches();
    if (win.bound_dash.size() < win.members.size()) win.bound_dash.resize(win.members.size(), -1);
    if (win.bound_dash_pref.size() < win.members.size()) win.bound_dash_pref.resize(win.members.size(), -1);
    // Умолчание: устойчивость сплошной, предпочтительность штрихом.
    auto style_of = [&win](size_t j, int which) {
        const std::vector<int>& v = (which == kStabBoundPreference) ? win.bound_dash_pref : win.bound_dash;
        const int st = (j < v.size()) ? v[j] : -1;
        return (st >= 0 && st < 4) ? st : (which == kStabBoundPreference ? 1 : 0);
    };

    toolbar_row([&]() {
        ImGui::TextDisabled("show:"); ImGui::SameLine();
        ImGui::RadioButton("stability##bw", &win.bound_which, 0); ImGui::SameLine();
        ImGui::RadioButton("preference##bw", &win.bound_which, 1); ImGui::SameLine();
        ImGui::RadioButton("both##bw", &win.bound_which, 2);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "stability  - the line rho = 1 of every tab of the window.\n"
                "preference - the boundary of rho <= 1 AND a one-step error not above\n"
                "             the tolerance of that tab (Fedoseev et al., 2022).\n"
                "both       - both lines: by default stability solid, preference\n"
                "             dashed and lighter (\"Line styles...\" changes it).\n"
                "\n"
                "The lines are iso-lines of the computed grid, interpolated along the\n"
                "edges of the cells. A boundary that leaves the grid ends at its edge:\n"
                "widen sigma / omega of the tab to close it.");
        ImGui::SameLine();
        ImGui::TextDisabled("|"); ImGui::SameLine();
        item_w(110);
        ImGui::SliderFloat("width##bw", &win.bound_width, 0.5f, 6.0f, "%.1f px", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SameLine();
        if (ImGui::Button("Line styles...##bw")) ImGui::OpenPopup("bound_styles");
        if (ImGui::BeginPopup("bound_styles")) {
            item_w(160);
            ImGui::Combo("palette##bw", &win.palette, kOrderPaletteNames, 3);
            for (size_t j = 0; j < win.members.size(); ++j) {
                const int mi = win.members[j];
                if (mi < 0 || mi >= (int)s.configs.size()) continue;
                ImGui::PushID((int)j);
                ImGui::ColorButton("##sw", order_base_color((int)j, win.palette),
                                   ImGuiColorEditFlags_NoTooltip,
                                   ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight()));
                ImGui::SameLine();
                int st = style_of(j, kStabBoundStability);
                item_w(110);
                if (ImGui::Combo("##stab", &st, kBoundDashNames, 4)) win.bound_dash[j] = st;
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Stability boundary (rho = 1).");
                ImGui::SameLine();
                int sp = style_of(j, kStabBoundPreference);
                item_w(110);
                if (ImGui::Combo("##pref", &sp, kBoundDashNames, 4)) win.bound_dash_pref[j] = sp;
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Preference boundary.");
                ImGui::SameLine();
                ImGui::TextUnformatted(s.configs[(size_t)mi].label.c_str());
                ImGui::PopID();
            }
            ImGui::TextDisabled("Columns: stability, preference.\n"
                                "Colour of a line: right click its entry in the legend.");
            ImGui::EndPopup();
        }
    });

    if (win.members.empty()) {
        ImGui::TextDisabled("The window is empty - add stability tabs via \"Members...\".");
        return;
    }

    const bool both = (win.bound_which == 2);
    std::vector<PlotSeriesInput> series;
    std::vector<bool> init_vis, glob_vis;
    std::vector<std::string> missing, empty_lines;
    double xlo = 0.0, xhi = 0.0, ylo = 0.0, yhi = 0.0;
    bool got_range = false;
    unsigned long long h = 1469598103934665603ULL;
    auto mixh = [&h](unsigned long long v) { h = (h ^ v) * 1099511628211ULL; };

    for (size_t j = 0; j < win.members.size(); ++j) {
        const int mi = win.members[j];
        if (mi < 0 || mi >= (int)s.configs.size()) continue;
        const OrderConfig& c = s.configs[(size_t)mi];
        const StabilityResult& r = c.stab_result;
        if (c.calc_kind != kOrderCalcStab || !c.stab_last_run_ok || r.rho.empty()) {
            missing.push_back(c.label);
            continue;
        }
        if (!got_range) { xlo = r.sigma_lo; xhi = r.sigma_hi; ylo = r.omega_lo; yhi = r.omega_hi; got_range = true; }
        else {
            xlo = std::min(xlo, r.sigma_lo); xhi = std::max(xhi, r.sigma_hi);
            ylo = std::min(ylo, r.omega_lo); yhi = std::max(yhi, r.omega_hi);
        }
        const double tol = parse_num(c.stab_pref_tol_text, 1.0);
        const ImVec4 base = order_base_color((int)j, win.palette);
        for (int which = 0; which < 2; ++which) {
            if (!both && win.bound_which != which) continue;
            OrderBoundsEntry& e = cache[std::make_tuple(win.id, mi, which)];
            const bool tol_changed = (which == kStabBoundPreference) && !(e.tol == tol);
            if (e.gen != c.stab_data_generation || e.data != (const void*)r.rho.data() || tol_changed) {
                e.contour = stability_contour(r, which, tol);
                e.gen  = c.stab_data_generation;
                e.data = r.rho.data();
                e.tol  = tol;
            }
            const char* what = (which == kStabBoundPreference) ? "preference" : "stability";
            if (e.contour.n_lines == 0) empty_lines.push_back(c.label + " (" + what + ")");
            PlotSeriesInput si;
            si.points   = e.contour.xy.data();
            si.n_points = (int)(e.contour.xy.size() / 2);
            si.label    = both ? c.label + ": " + what : c.label;
            const bool minor = both && which == kStabBoundPreference;
            si.color    = minor ? order_shade(base, 1.35f) : base;
            float period = 0.0f, on = 0.0f;
            bound_dash_pattern(style_of(j, which), win.bound_width, period, on);
            si.dash_period_px = period;
            si.dash_on_px     = on;
            series.push_back(si);
            init_vis.push_back(true);
            glob_vis.push_back(true);
            mixh((unsigned long long)mi); mixh((unsigned long long)which);
            mixh((unsigned long long)e.gen); mixh((unsigned long long)(uintptr_t)e.data);
            std::uint64_t tb = 0; std::memcpy(&tb, &e.tol, sizeof(tb)); mixh(tb);
        }
    }

    toolbar_row([&]() {
        if (!missing.empty()) {
            std::string m;
            for (const auto& l : missing) { if (!m.empty()) m += ", "; m += l; }
            ImGui::TextDisabled("no data yet: %s", m.c_str());
        }
        if (!empty_lines.empty()) {
            std::string m;
            for (const auto& l : empty_lines) { if (!m.empty()) m += ", "; m += l; }
            ImGui::TextColored(readable_text(ImVec4(1.0f, 0.75f, 0.4f, 1.0f)),
                               "no boundary inside the grid: %s", m.c_str());
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("The whole grid is on one side of the boundary (all inside or\n"
                                  "all outside the region). Widen sigma / omega of that tab.");
        }
    });

    if (series.empty()) {
        ImGui::TextDisabled("No data yet. Press Run on the stability tabs of this window.");
        return;
    }

    view.imdraw_lines      = true;
    view.gpu_thick_lines   = true;    // пунктир есть только у GL-ломаной
    view.line_thickness_px = win.bound_width;
    view.points_mode       = false;
    view.point_markers     = false;
    view.show_zero_x       = false;
    view.show_zero_y       = false;
    view.x_axis.name = "sigma";
    view.y_axis.name = "omega";
    view.x_axis.log_scale   = false;
    view.y_axis.display_log = false;
    // Автоподбор — по сеткам расчёта, а не по самим линиям: оборванная на краю
    // граница должна читаться как упёршаяся в край сетки.
    view.x_fit_use_explicit = got_range && xhi > xlo;
    view.x_fit_min = xlo; view.x_fit_max = xhi;
    view.y_fit_use_explicit = got_range && yhi > ylo;
    view.y_fit_min = ylo; view.y_fit_max = yhi;
    view.pad_x = view.pad_y = false;

    mixh((unsigned long long)series.size());
    const int sig = (int)((h ^ (h >> 32)) & 0x3FFFFFFFULL);
    bool fit = false;
    if (win.plot_sig != sig) { fit = true; win.plot_sig = sig; }

    view.popup_extras = [&model, &win, &cb, &s]() {
        draw_order_export_items(model, win, cb, s.in_flight);
    };
    ImVec2 avail  = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    view.render(renderer, origin, avail, /*owner_id*/ 0x0BDE3000 + win.id, sig,
                series, init_vis, glob_vis, fit);
}

void draw_order_map_window(AppModel& model, OrderPlotWindow& win,
                                  const GuiCallbacks& cb,
                                  PlotRenderer& renderer, HeatmapView& hv) {
    OrderAnalysisSession& s = model.order_session;
    if (win.members.empty()) { ImGui::TextDisabled("The window is empty - add a tab via \"Members...\"."); return; }
    const int mi = win.members[0];
    if (mi < 0 || mi >= (int)s.configs.size()) { ImGui::TextDisabled("The tab has been removed."); return; }
    OrderConfig& c = s.configs[(size_t)mi];
    // Раскраску «как в статье» ставит только карта устойчивости, а вью у окна
    // одна на все его члены: без сброса подмена палитры пережила бы смену
    // члена окна на обычную карту порядка.
    hv.colormap_override = 0;
    hv.oor_custom        = false;
    if (c.calc_kind == kOrderCalcStab) {
        draw_order_stability_map(model, win, cb, renderer, hv, c, mi);
        return;
    }
    const OrderResult& r = c.result;

    toolbar_row([&]() {   // тулбар карты (Show → Toolbar)
        ImGui::Checkbox("show error", &win.map_error);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Off - the map of the order p. On - log10 E1.");
        ImGui::SameLine();
        draw_order_status_line(c.label.c_str(), r.n_ok, r.n_diverged, r.n_floor, r.n_nocontract);
    });

    if (!c.last_run_ok || r.p.empty() || r.n_pts_y <= 1) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }

    {
        HeatmapToolbarOpts topts;
        topts.persist_colormap = [&win](int cm) { win.colormap_idx = cm; };
        draw_heatmap_toolbar(hv, topts);
    }

    // Ошибка гуляет на много порядков — в карту идёт log10, иначе одна ячейка
    // растягивает шкалу и остальное становится однотонным.
    std::vector<double> buf;
    const std::vector<double>* vals = &r.p;
    if (win.map_error) {
        buf.resize(r.e1.size());
        for (size_t i = 0; i < r.e1.size(); ++i)
            buf[i] = (std::isfinite(r.e1[i]) && r.e1[i] > 0.0)
                         ? std::log10(r.e1[i]) : std::numeric_limits<double>::quiet_NaN();
        vals = &buf;
    }
    // Диапазон цвета — по ячейкам со статусом OK (движок отдаёт их в p_min/p_max
    // и e1_min/e1_max). Полка округления и разошедшиеся ячейки исключены
    // намеренно: одна такая растягивает шкалу так, что рабочая область
    // становится одноцветной. Они не пропадают — упираются в край палитры.
    double vmin = 0.0, vmax = 0.0;
    if (win.map_error) {
        vmin = (r.e1_min > 0.0) ? std::log10(r.e1_min) : 0.0;
        vmax = (r.e1_max > 0.0) ? std::log10(r.e1_max) : 0.0;
    } else {
        vmin = r.p_min;
        vmax = r.p_max;
    }
    if (!(vmax > vmin)) {
        bool first = true;
        for (double v : *vals) {
            if (!std::isfinite(v)) continue;
            if (first) { vmin = vmax = v; first = false; }
            else { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
        }
        if (!(vmax > vmin)) { vmin -= 0.5; vmax += 0.5; }
    }

    hv.x_axis.name = s.axis_target_label(c.axis_x_target);
    hv.y_axis.name = s.axis_target_label(c.axis_y_target);
    hv.x_axis.log_scale = r.axis_x.log_scale;
    hv.y_axis.log_scale = r.axis_y.log_scale;

    const int sig = c.data_generation * 2 + (win.map_error ? 1 : 0);
    bool fit = c.fit_request;
    if (win.plot_sig != sig) { fit = true; win.plot_sig = sig; }
    if (c.fit_request) c.fit_request = false;

    // Ctrl+T: обе оси карты — диапазоны свипа (по умолчанию X — сам шаг h).
    const std::vector<ViewRangeTarget> vrt = {
        vr_target(s.configs, mi,
                  &OrderConfig::axis_x_lo_text, &OrderConfig::axis_x_hi_text,
                  &OrderConfig::axis_y_lo_text, &OrderConfig::axis_y_hi_text) };
    hv.popup_extras = [&vrt, &model, &hv, &win, &cb, &s]() {
        draw_view_range_menu(vrt, hv.x_axis, hv.y_axis, hv.swap_axes, &model);
        ImGui::Separator();
        draw_order_export_items(model, win, cb, s.in_flight);
    };

    ImVec2 avail  = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    hv.render(renderer, origin, avail,
              /*owner_id*/ 0x0BDE0000 + win.id,
              sig,
              r.n_pts_x, r.n_pts_y, vals->data(),
              r.axis_x.lo, r.axis_x.hi, r.axis_y.lo, r.axis_y.hi,
              vmin, vmax, fit);
    handle_view_range_keys(vrt, hv.x_axis, hv.y_axis, plot_window_active(), hv.swap_axes, &model);
}
}

namespace ucuda::ui_detail {
using OrderCurveMember = std::tuple<int, int, const double*, bool, std::string, std::string>;
using OrderCurveKey = std::tuple<int, int, bool, bool, bool, bool, bool, bool, bool, bool,
                                 int, int, int, int, int, bool, std::vector<OrderCurveMember>>;
struct OrderCurveData {
    std::vector<std::vector<double>> bufs, tags;
    std::vector<std::string> labels;
    std::vector<ImVec4> colors;
    std::vector<int> markers;
    double xlo = 0.0, xhi = 0.0;
    bool got_x = false;
};

// Поля стиля окна в разметке меню Phase Analysis (draw_line_style_controls).
LineStyleFields order_style_fields(OrderPlotWindow& win) {
    return { &win.custom_line_style, &win.line_width, &win.line_alpha, &win.points_mode,
             &win.point_size, &win.point_alpha, &win.point_marker };
}

auto& order_curve_caches() {
    return ui_resource<std::map<int, PlotDataCache<OrderCurveKey, OrderCurveData>>, 3015>();
}

void draw_order_curve_window(AppModel& model, OrderPlotWindow& win,
                                    const GuiCallbacks& cb,
                                    PlotRenderer& renderer, Plot2DView& view) {
    OrderAnalysisSession& s = model.order_session;
    const bool is_perf  = (win.kind == OrderPlotWindow::Kind::Perf);
    const bool is_error = (win.kind == OrderPlotWindow::Kind::Error);
    const bool is_p     = (win.kind == OrderPlotWindow::Kind::P);
    // Есть ли в окне адаптивная вкладка: без неё окно Performance — ровно прежнее
    // (тулбар, оси, подсказки точек), без осей «точность — затраты».
    bool any_ad = false;
    for (int mi : win.members)
        if (mi >= 0 && mi < (int)s.configs.size())
            any_ad |= s.configs[(size_t)mi].adaptive.enabled || s.configs[(size_t)mi].perf_result.adaptive;

    // ---- Тулбар ---- (скрывается в меню графика: Show → Toolbar)
    const bool bar = !plot_toolbar_hidden();
    toolbar_row([&]() {
    ImGui::Checkbox("log X", &win.x_log);
    if (!is_p) {
        ImGui::SameLine();
        ImGui::Checkbox("log Y", &win.y_log);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "The logarithm is taken of THE VALUES THEMSELVES while the axis stays linear:\n"
                "Plot2DView can only do log on X. The slope of the line in log-log\n"
                "does not change because of that, but the axis labels read as log10.");
    }
    if (is_error) { ImGui::SameLine(); ImGui::Checkbox("show E2", &win.show_e2); }
    if (is_p)     { ImGui::SameLine(); ImGui::Checkbox("nominal order line", &win.show_nominal); }
    if (is_perf) {
        ImGui::SameLine(); ImGui::TextDisabled("| X:"); ImGui::SameLine();
        ImGui::RadioButton("E1##perrsrc", &win.error_source, 0); ImGui::SameLine();
        ImGui::RadioButton("E2##perrsrc", &win.error_source, 1); ImGui::SameLine();
        ImGui::RadioButton("Eref##perrsrc", &win.error_source, 2);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "The difference against the reference solution, max|y_h - y_ref|:\n"
                "the actual error of the scheme at step h, rather than the\n"
                "Richardson estimate that E1 and E2 are. The reference method is\n"
                "chosen in the tab's settings (DOPRI78 by default).");
        ImGui::SameLine();
        ImGui::TextDisabled("| t:"); ImGui::SameLine();
        ImGui::RadioButton("us##ptu", &win.time_unit, 0); ImGui::SameLine();
        ImGui::RadioButton("ms##ptu", &win.time_unit, 1); ImGui::SameLine();
        ImGui::TextDisabled("|"); ImGui::SameLine();
        ImGui::Checkbox("min", &win.show_min); ImGui::SameLine();
        ImGui::Checkbox("avg", &win.show_avg); ImGui::SameLine();
        ImGui::Checkbox("max", &win.show_max);

        // Вторая строка — «точность — затраты»: ошибка в T против эталона и другие оси
        // затрат. Есть всегда: E(T) постоянного шага считается против того же y*(T), что у
        // адаптивного (reference y*(T) в настройках вкладки), так что и окно из одних Fixed-
        // вкладок можно строить в тех же осях, что смешанное.
        {
        ImGui::TextDisabled("X:"); ImGui::SameLine();
        ImGui::RadioButton("E(T)##perrsrc", &win.error_source, 3);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Error at t_max, max|y(T) - y*(T)|, against DOP853 in dd/qd on the CPU.\n"
                "Adaptive tabs always have it (their only error). Fixed tabs: with\n"
                "\"reference y*(T)\" set in the tab and \"fit h to t_max\" on - the same\n"
                "y*(T), so fixed-step and adaptive curves share one diagram; without it,\n"
                "Eref at t_max (\"endpoint only\" and \"fit h to t_max\").");
        ImGui::SameLine();
        ImGui::TextDisabled("| Y:"); ImGui::SameLine();
        ImGui::RadioButton("time##pcost", &win.perf_cost, 0); ImGui::SameLine();
        ImGui::RadioButton("f evals##pcost", &win.perf_cost, 1);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Evaluations of the right-hand side per run (one = all N components):\n"
                              "the controller's count for the adaptive step (rejected attempts\n"
                              "included), f per step x steps for the fixed one. CD family: passes x\n"
                              "(explicit half 1 + implicit half: 1 per component linear in its own\n"
                              "variable, 4 iterations otherwise, / N); Extr: substeps x the base;\n"
                              "Comp: stages x the base; GBS: sum n_k + 1. For custom KRS,\n"
                              "set f evals per step in System. Curves with unknown cost\n"
                              "are left out.\n"
                              "min / avg / max apply to time only.");
        ImGui::SameLine();
        ImGui::RadioButton("steps##pcost", &win.perf_cost, 2);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Accepted steps per run (the rejected ones are in the node tooltip).");
        }
    }
    });   // bar

    // Стиль кривых — тот же тулбар-с-попапом, что у точек бифуркационных
    // диаграмм и у Node Series: выключенный режим даёт прежний вид.
    if (bar) {
        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
    }
    {
        const std::string sfx = "ordplot" + std::to_string(win.id);
        // Палитра стоит В СТРОКЕ, а не в попапе: попап открывается только при
        // включённом custom_line_style, а цвет кривых нужен и без него.
        if (win.palette < 0 || win.palette > kOrderPaletteOkabe)
            win.palette = kOrderPaletteContrast;
        toolbar_row([&win, sfx]() {
            item_w(ImGui::GetFontSize() * 9.0f, combo_items_need(kOrderPaletteNames, 3));
            ImGui::Combo(("##pal" + sfx).c_str(), &win.palette,
                         kOrderPaletteNames, 3);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("The colour codes the TAB, the marker shape codes the curve inside it\n"
                                  "(min / avg / max). Both extra palettes are qualitative: adjacent\n"
                                  "numbers are deliberately far apart in hue, unlike the heatmap maps.");
            ImGui::SameLine();
        });
        // sfx — копией: элемент отдаётся и в меню диаграммы, а оно рисуется после этого блока.
        // Стиль — то же меню, что у проекций Phase Analysis.
        draw_line_style_controls(order_style_fields(win), sfx.c_str(), "Points at the nodes.");
    }

    const bool   ylog       = win.y_log && !is_p;
    const double time_scale = (is_perf && win.time_unit == 1) ? 1.0e-3 : 1.0;

    std::vector<ViewRangeTarget> vrt;
    std::vector<OrderCurveMember> members;
    bool any_data = false, fit = false;
    int sig = (int)win.kind * 1000003;
    for (int mi : win.members) {
        if (mi < 0 || mi >= (int)s.configs.size()) continue;
        auto& c = s.configs[(size_t)mi];
        const int generation = is_perf ? c.perf_data_generation : c.data_generation;
        const bool have = is_perf ? c.perf_last_run_ok && c.perf_result.n_pts > 0
                                  : c.last_run_ok && !c.result.p.empty() && c.result.n_pts_y <= 1;
        const double* axis = is_perf ? c.perf_result.axis_vals.data() : c.result.axis_x_vals.data();
        members.emplace_back(mi, generation, axis, have, c.label, c.scheme);
        sig = sig * 131 + mi * 17 + generation;
        if (!have) continue;
        any_data = true;
        bool& fit_request = is_perf ? c.perf_fit_request : c.fit_request;
        fit |= fit_request;
        fit_request = false;
        if (!is_perf)
            vrt.push_back(vr_target(s.configs, mi,
                                    &OrderConfig::axis_x_lo_text, &OrderConfig::axis_x_hi_text));
    }

    const OrderCurveKey key{model.session_epoch[(int)AppModel::AppMode::Order], (int)win.kind,
                            win.x_log, ylog, win.show_e2, win.show_nominal, win.show_min, win.show_avg,
                            win.show_max, any_ad, win.error_source, win.time_unit, win.perf_cost,
                            win.palette, series_palette(), win.vary_markers, std::move(members)};
    auto& cache = order_curve_caches()[win.id];
    if (cache.update(key, [&](OrderCurveData& data) {
        data = {};
        auto& bufs = data.bufs;
        auto& tags = data.tags;
        auto& labels = data.labels;
        auto& colors = data.colors;
        auto& markers = data.markers;

        auto note_x = [&data](double x) {
            if (!data.got_x) {
                data.xlo = data.xhi = x;
                data.got_x = true;
            } else {
                data.xlo = std::min(data.xlo, x);
                data.xhi = std::max(data.xhi, x);
            }
        };
        for (size_t j = 0; j < win.members.size(); ++j) {
            const int mi = win.members[j];
            if (mi < 0 || mi >= (int)s.configs.size()) continue;
            OrderConfig& c = s.configs[(size_t)mi];
            const ImVec4 base = order_base_color((int)j, win.palette);
            if (is_perf) {
                const PerfResult& r = c.perf_result;
                if (!c.perf_last_run_ok || r.n_pts <= 0) continue;
                // Точка существует, только когда есть ОБЕ координаты: ошибка узла
                // и его затраты. Сортировка по X обязательна — ось X здесь не сетка
                // узлов, а посчитанная величина, и монотонность её не гарантирована.
                const double qn = std::numeric_limits<double>::quiet_NaN();
                auto at = [qn](const std::vector<double>& v, int i) { return (size_t)i < v.size() ? v[(size_t)i] : qn; };
                // tsrc — время, которое попадает в подсказку: у кривых времени своё (min / avg /
                // max), как было до осей f и шагов; у них — среднее.
                auto add_cost_series = [&](const std::function<double(int)>& cost, const std::vector<double>& tsrc,
                                           const std::string& suffix, float shade_k, int marker_slot) {
                    // Теги — h (или tol у адаптивного), ошибка, время, шаги, f, отказы. Они
                    // обязаны ехать вместе с точкой через сортировку по X: после неё позиция
                    // в массиве уже ничего не говорит о том, какому узлу точка принадлежала.
                    struct Pt { double x, y, tg[6]; };
                    std::vector<Pt> pts;
                    pts.reserve((size_t)r.n_pts);
                    for (int i = 0; i < r.n_pts; ++i) {
                        // Eref / E(T) есть не в каждом результате: эталон можно было и не
                        // просить, а старые прогоны про него вовсе не знают.
                        const double e = (win.error_source == 3) ? at(r.e_end, i)
                                       : (win.error_source == 2) ? at(r.e_ref, i)
                                       : (win.error_source == 1) ? at(r.e2, i) : at(r.e1, i);
                        const double cv = cost(i);
                        if (!std::isfinite(e) || !std::isfinite(cv)) continue;
                        if (win.x_log && !(e > 0.0)) continue;
                        if (ylog && !(cv > 0.0)) continue;
                        const double ns = ((size_t)i < r.n_steps.size()) ? (double)r.n_steps[(size_t)i] : qn;
                        // e и cv кладём в теги ОТДЕЛЬНО от координат: в лог-режиме в
                        // координатах лежат log10. В подсказке хочется исходные числа.
                        Pt pt{ win.x_log ? std::log10(e) : e, ylog ? std::log10(cv) : cv,
                               { r.adaptive ? at(r.axis_vals, i) : at(r.h_eff, i), e,
                                 at(tsrc, i) * time_scale, ns, at(r.n_rhs, i), at(r.n_rej, i) } };
                        pts.push_back(pt);
                    }
                    if (pts.empty()) return;
                    std::sort(pts.begin(), pts.end(),
                              [](const Pt& a, const Pt& b) { return a.x < b.x; });
                    std::vector<double> xy, tg;
                    xy.reserve(pts.size() * 2);
                    tg.reserve(pts.size() * (any_ad ? 6 : 4));
                    for (const auto& pr : pts) {
                        xy.push_back(pr.x); xy.push_back(pr.y);
                        for (int q = 0; q < (any_ad ? 6 : 4); ++q) tg.push_back(pr.tg[q]);   // Fixed: h, E, t, steps
                        note_x(pr.x);
                    }
                    bufs.push_back(std::move(xy));
                    tags.resize(bufs.size());
                    tags.back() = std::move(tg);
                    labels.push_back(c.label + (suffix.empty() ? std::string() : " " + suffix));
                    colors.push_back(order_shade(base, shade_k));
                    markers.push_back(win.vary_markers ? order_marker_for(marker_slot) : -1);
                };
                const std::string kind_tag = r.adaptive ? std::string("(adaptive)") : std::string();
                if (win.perf_cost == 1) {
                    add_cost_series([&](int i) { return at(r.n_rhs, i); }, r.t_avg, kind_tag, 1.0f, 1);
                } else if (win.perf_cost == 2) {
                    add_cost_series([&](int i) {
                        return (size_t)i < r.n_steps.size() && r.n_steps[(size_t)i] > 0 ? (double)r.n_steps[(size_t)i] : qn;
                    }, r.t_avg, kind_tag, 1.0f, 1);
                } else {
                    const std::string sp = kind_tag.empty() ? std::string() : kind_tag + " ";
                    if (win.show_min) add_cost_series([&](int i) { return at(r.t_min, i) * time_scale; }, r.t_min, sp + "min", 0.55f, 0);
                    if (win.show_avg) add_cost_series([&](int i) { return at(r.t_avg, i) * time_scale; }, r.t_avg, sp + "avg", 1.0f,  1);
                    if (win.show_max) add_cost_series([&](int i) { return at(r.t_max, i) * time_scale; }, r.t_max, sp + "max", 1.5f,  2);
                }
                continue;
            }
            const OrderResult& r = c.result;
            if (!c.last_run_ok || r.p.empty() || r.n_pts_y > 1) continue;
            const int n = std::min((int)r.axis_x_vals.size(), (int)r.p.size());
            auto add_node_series = [&](int which, const std::string& suffix,
                                       float shade_k, int marker_slot) {
                std::vector<double> xy;
                xy.reserve((size_t)n * 2);
                for (int i = 0; i < n; ++i) {
                    const double x = r.axis_x_vals[(size_t)i];
                    if (win.x_log && !(x > 0.0)) continue;
                    double y;
                    if (which == 0) {
                        // Ячейки floor / p<=0 НЕ выбрасываем: их провал и есть ответ
                        // на вопрос, где метод ещё держит свой порядок. Прячем
                        // только diverged — там ничего не посчитано.
                        y = r.p[(size_t)i];
                        if (!std::isfinite(y)) continue;
                    } else {
                        const double e = (which == 1) ? r.e1[(size_t)i] : r.e2[(size_t)i];
                        if (!std::isfinite(e) || (ylog && !(e > 0.0))) continue;
                        y = ylog ? std::log10(e) : e;
                    }
                    xy.push_back(x); xy.push_back(y); note_x(x);
                }
                if (xy.empty()) return;
                bufs.push_back(std::move(xy));
                labels.push_back(suffix.empty() ? c.label : (c.label + " " + suffix));
                colors.push_back(order_shade(base, shade_k));
                markers.push_back(win.vary_markers ? order_marker_for(marker_slot) : -1);
            };
            if (is_error) {
                add_node_series(1, "E1", 1.0f,  0);   // circle
                if (win.show_e2) add_node_series(2, "E2", 0.55f, 1);   // triangle down
            } else {
                add_node_series(0, "", 1.0f, 0);
                if (win.show_nominal) {
                    const int nom = builtin_scheme_order(c.scheme);
                    if (nom > 0 && !r.axis_x_vals.empty()) {
                        double a = r.axis_x_vals.front(), b = r.axis_x_vals.back();
                        if (b < a) std::swap(a, b);
                        if (!win.x_log || (a > 0.0 && b > 0.0)) {
                            std::vector<double> xy = { a, (double)nom, b, (double)nom };
                            bufs.push_back(std::move(xy));
                            labels.push_back(c.label + " nominal " + std::to_string(nom));
                            colors.push_back(ImVec4(0.6f, 0.6f, 0.6f, 0.9f));
                            markers.push_back(-1);   // горизонталь, узлов у неё нет
                        }
                    }
                }
            }
        }
    })) view.series_generation = -1;
    auto& [bufs, tags, labels, colors, markers, xlo, xhi, got_x] = cache.data;
    // Сводка по статусам — по строке на член окна. Часть тулбара: скрытый прячет и её.
    for (size_t j = 0; bar && j < win.members.size(); ++j) {
        const int mi = win.members[j];
        if (mi < 0 || mi >= (int)s.configs.size()) continue;
        const OrderConfig& c = s.configs[(size_t)mi];
        if (is_perf) {
            if (!c.perf_last_run_ok) continue;
            const PerfResult& r = c.perf_result;
            draw_order_status_line(c.label.c_str(), r.n_ok, r.n_diverged, r.n_floor, r.n_nocontract);
            ImGui::SameLine();
            ImGui::TextDisabled("| %d measurements x %d replicas, %d warmup",
                                r.repeats, r.replicas, r.warmup);
            if (win.error_source == 2) {
                ImGui::SameLine();
                const bool has_ref = std::any_of(r.e_ref.begin(), r.e_ref.end(),
                                                 [](double v) { return std::isfinite(v) && v > 0.0; });
                if (has_ref)
                    ImGui::TextDisabled("| reference: %s x%s",
                                        c.perf_ref_scheme.c_str(), c.perf_ref_substeps_text.c_str());
                else
                    ImGui::TextColored(readable_text(ImVec4(1.0f, 0.75f, 0.4f, 1.0f)),
                                       r.adaptive ? "| an adaptive tab has only E(T)"
                                                  : "| no reference in this result: set reference substeps > 0 and run again");
            } else if (win.error_source == 3) {
                ImGui::SameLine();
                const bool any_e = std::any_of(r.e_end.begin(), r.e_end.end(),
                                               [](double v) { return std::isfinite(v); });
                if (any_e && r.ref_prec > 0)
                    ImGui::TextDisabled("| y*(T): DOP853 in %s, %lld steps, its own error ~%.1e",
                                        r.ref_prec == 2 ? "qd" : "dd", r.ref_steps, r.ref_err);
                else if (any_e)
                    ImGui::TextDisabled("| E(T) = Eref at t_max: %s x%s",
                                        c.perf_ref_scheme.c_str(), c.perf_ref_substeps_text.c_str());
                else
                    ImGui::TextColored(readable_text(ImVec4(1.0f, 0.75f, 0.4f, 1.0f)),
                                       r.adaptive ? "| no y*(T) in this result: run again"
                                                  : "| no E(T): set \"reference y*(T)\" (or \"endpoint only\") with \"fit h to t_max\" and run again");
            } else if (r.adaptive) {
                ImGui::SameLine();
                ImGui::TextColored(readable_text(ImVec4(1.0f, 0.75f, 0.4f, 1.0f)), "| an adaptive tab has only E(T)");
            }
            if (r.adaptive) {
                double rej = 0, acc = 0;
                for (int i = 0; i < r.n_pts; ++i) {
                    if ((size_t)i < r.n_rej.size() && std::isfinite(r.n_rej[(size_t)i])) rej += r.n_rej[(size_t)i];
                    if ((size_t)i < r.n_steps.size()) acc += (double)r.n_steps[(size_t)i];
                }
                ImGui::SameLine();
                ImGui::TextDisabled("| tol %.0e..%.0e, rejected %.1f%% of attempts",
                                    r.axis_vals.empty() ? 0.0 : r.axis_vals.front(),
                                    r.axis_vals.empty() ? 0.0 : r.axis_vals.back(),
                                    acc + rej > 0 ? 100.0 * rej / (acc + rej) : 0.0);
            }
        } else {
            if (!c.last_run_ok) continue;
            const OrderResult& r = c.result;
            draw_order_status_line(c.label.c_str(), r.n_ok, r.n_diverged, r.n_floor, r.n_nocontract);
        }
    }

    if (win.members.empty()) {
        ImGui::TextDisabled("The window is empty - add a tab via \"Members...\".");
        return;
    }
    if (!any_data || bufs.empty()) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }

    // Стиль — как у 2D-проекций Phase Analysis (apply_line_style).
    const LineStyleFields style = order_style_fields(win);
    apply_line_style(view, style);
    const bool pts = win.points_mode;
    view.show_zero_x  = false;
    // Линия y = 0 осмысленна только на графике порядка: ниже неё разности
    // перестали сокращаться. В логарифме ошибки нуль — это E = 1, рисовать
    // его незачем.
    view.show_zero_y  = is_p;

    // Подписи осей. У членов окна ось X по построению одна и та же величина,
    // поэтому имя берём у первого valid-члена.
    std::string xname = "x";
    for (int mi : win.members) {
        if (mi < 0 || mi >= (int)s.configs.size()) continue;
        xname = is_perf ? ((win.error_source == 3) ? "E(T)" : (win.error_source == 2) ? "Eref"
                            : (win.error_source == 1) ? "E2" : "E1")
                        : s.axis_target_label(s.configs[(size_t)mi].axis_x_target);
        break;
    }
    // На диаграмме производительности лог по X берётся ЛОГАРИФМОМ ДАННЫХ, а не
    // логарифмическим масштабом оси — как это давно сделано по Y. Причина в
    // подписях: на лог-оси тики печатаются как сами значения, и на
    // double-double ошибка уходит к 1e-30, где формат схлопывает их в "0" и
    // ось становится нечитаемой. log10(E) — обычные числа вроде -16, они
    // помещаются всегда. Исходное E остаётся в подсказке узла (тег "E1").
    // На диаграмме порядка по X лежит сетка свипа, там масштаб оси и остаётся.
    if (is_perf && win.x_log) xname = "log10 " + xname;
    view.x_axis.name = xname;
    if (is_perf) {
        const char* unit = (win.time_unit == 1) ? "ms" : "us";
        const std::string yn = win.perf_cost == 1 ? std::string("f evals")
                             : win.perf_cost == 2 ? std::string("steps")
                                                  : std::string("t, ") + unit;
        view.y_axis.name = ylog ? "log10 " + yn : yn;
    } else {
        view.y_axis.name = is_error ? (ylog ? "log10 E" : "E") : "p";
    }
    view.x_axis.log_scale = !is_perf && win.x_log;

    // Явный X-диапазон по всем кривым окна: без него Plot2DView берёт границы
    // клампа из bbox VBO, а там на лог-оси лежит log10(x). На диаграмме
    // производительности ось линейная, и note_x собирал уже те же значения,
    // что легли в VBO, — единицы сходятся и тут.
    view.x_fit_use_explicit = got_x && (xhi > xlo);
    view.x_fit_min = xlo;
    view.x_fit_max = xhi;

    // Маркеры узлов нужны именно на диаграмме производительности: ни шага, ни
    // самого времени в лог-режиме на осях нет. На диаграмме порядка h лежит
    // прямо на оси X, спрашивать нечего.
    view.point_markers = !pts && is_perf;
    view.point_marker_px = 3.5f;
    if (is_perf) {
        const char* ename = (win.error_source == 3) ? "E(T)" : (win.error_source == 2) ? "Eref"
                          : (win.error_source == 1) ? "E2" : "E1";
        const char* unit  = (win.time_unit == 1) ? "ms" : "us";
        if (any_ad) view.point_tag_names = { "h / tol", ename, std::string("t, ") + unit, "steps", "f evals", "rejected" };
        else        view.point_tag_names = { "h", ename, std::string("t, ") + unit, "steps" };
    }

    std::vector<PlotSeriesInput> series;
    std::vector<bool> init_vis, glob_vis;
    series.reserve(bufs.size());
    tags.resize(bufs.size());
    for (size_t i = 0; i < bufs.size(); ++i) {
        PlotSeriesInput si;
        si.points   = bufs[i].empty() ? nullptr : bufs[i].data();
        si.n_points = (int)(bufs[i].size() / 2);
        si.color    = colors[i];
        if (win.custom_line_style) si.color.w = win.line_alpha;
        si.label    = labels[i];
        if (i < markers.size()) si.node_marker = markers[i];
        // Длины обязаны сойтись: если кривая отфильтровалась иначе, чем
        // собирались теги, лучше остаться без подсказки, чем показать чужое h.
        const int tag_cols = (int)view.point_tag_names.size();
        if (tag_cols > 0 && !tags[i].empty() &&
            tags[i].size() == (size_t)si.n_points * (size_t)tag_cols) {
            si.point_tags      = tags[i].data();
            si.point_tag_count = tag_cols;
        }
        series.push_back(si);
        init_vis.push_back(true);
        glob_vis.push_back(true);
    }

    // Сигнатура рисуемого: новые данные любого члена, смена масштаба любой
    // оси, состав кривых. Служит и поколением для перезаливки VBO, и триггером
    // autofit — log10 E и E отличаются на четырнадцать порядков, а p от них обоих.
    sig = sig * 131 + (win.x_log ? 1 : 0) + (ylog ? 2 : 0) + (win.show_e2 ? 4 : 0)
        + (win.show_nominal ? 8 : 0) + (win.show_min ? 16 : 0) + (win.show_avg ? 32 : 0)
        + (win.show_max ? 64 : 0) + win.error_source * 128 + win.time_unit * 512
        + win.perf_cost * 1024 + (int)bufs.size() * 4096;
    // Стиль в сигнатуру НЕ входит намеренно: содержимое VBO от него не
    // зависит (толщина, маркер, альфа — решения момента рисования), а смена
    // сигнатуры тянет за собой autofit и сбросила бы текущий масштаб вида.
    if (win.plot_sig != sig) { fit = true; win.plot_sig = sig; }

    // Ctrl+T есть только там, где по X лежит вход расчёта: на диаграмме
    // производительности ось X — посчитанная ошибка, задать её нельзя.
    view.popup_extras = [&vrt, &model, &view, &win, &cb, &s]() {
        if (!vrt.empty()) {
            draw_view_range_menu(vrt, view.x_axis, view.y_axis, /*swapped*/ false, &model);
            ImGui::Separator();
        }
        draw_order_export_items(model, win, cb, s.in_flight);
    };

    ImVec2 avail  = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    view.render(renderer, origin, avail,
                /*owner_id*/ 0x0BDE1000 + win.id, sig,
                series, init_vis, glob_vis, fit);
    if (!vrt.empty())
        handle_view_range_keys(vrt, view.x_axis, view.y_axis, plot_window_active(), /*swapped*/ false, &model);
}
}

namespace ucuda::ui_detail {
// Окно Error(t): ось X — время, Y — ошибка против референса (по умолчанию
// log10: ошибка у устойчивой схемы растёт на порядки, у хаотической системы —
// экспоненциально, и в линейном масштабе начало траектории не читается).
// Кривые: норма разности, оценка Ричардсона (референс h/2) и, по желанию,
// |d_i(t)| по переменным. Под тулбаром — итог: max, t максимума и RMS.
using OrderErrtKey = std::vector<long long>;
auto& order_errt_caches() {
    return ui_resource<std::map<int, PlotDataCache<OrderErrtKey, OrderCurveData>>, 3027>();
}

void draw_order_errt_window(AppModel& model, OrderPlotWindow& win,
                            const GuiCallbacks& cb,
                            PlotRenderer& renderer, Plot2DView& view) {
    OrderAnalysisSession& s = model.order_session;
    const bool bar = !plot_toolbar_hidden();
    bool any_rich = false;
    for (int mi : win.members)
        if (mi >= 0 && mi < (int)s.configs.size() && !s.configs[(size_t)mi].errt_result.e_rich.empty())
            any_rich = true;

    // Масштаб Y (lin/log) — в меню графика по ПКМ (AxisInfo::display_log): в VBO
    // идут сами ошибки, своей галки и своего log10 у окна нет.
    toolbar_row([&]() {
        ImGui::Checkbox("per variable", &win.errt_show_vars);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Add |x_h,i(t) - x_ref,i(t)| for every variable next to the norm.");
        if (any_rich) {
            ImGui::SameLine();
            ImGui::Checkbox("Richardson", &win.errt_show_rich);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Reference h/2 only: the estimate of the error of x_h itself,\n"
                    "|x_h - x_h/2| * 2^p / (2^p - 1), p - the order of the scheme.\n"
                    "The bare difference |x_h - x_h/2| is the error of x_h minus the\n"
                    "error of x_h/2, i.e. (1 - 2^-p) of the error of x_h.");
        }
        ImGui::SameLine();
        if (win.palette < 0 || win.palette > kOrderPaletteOkabe) win.palette = kOrderPaletteContrast;
        item_w(ImGui::GetFontSize() * 9.0f, combo_items_need(kOrderPaletteNames, 3));
        ImGui::Combo("##palerrt", &win.palette, kOrderPaletteNames, 3);
    });
    if (bar) {
        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
    }
    {
        // Стиль — то же меню, что у проекций Phase Analysis.
        const std::string sfx = "orderrt" + std::to_string(win.id);
        draw_line_style_controls(order_style_fields(win), sfx.c_str(), "Points at the nodes t_k.");
    }

    OrderErrtKey key = { model.session_epoch[(int)AppModel::AppMode::Order],
                         win.errt_show_vars ? 1 : 0, win.errt_show_rich ? 1 : 0, win.palette,
                         (long long)series_palette() };
    bool any_data = false, fit = false;
    for (int mi : win.members) {
        if (mi < 0 || mi >= (int)s.configs.size()) continue;
        OrderConfig& c = s.configs[(size_t)mi];
        key.push_back(mi);
        key.push_back(c.errt_data_generation);
        if (!c.errt_last_run_ok || c.errt_result.t.empty()) continue;
        any_data = true;
        fit |= c.errt_fit_request;
        c.errt_fit_request = false;
    }

    auto& cache = order_errt_caches()[win.id];
    if (cache.update(key, [&](OrderCurveData& data) {
        data = {};
        auto add = [&](const std::vector<double>& t, const std::vector<double>& e,
                       const std::string& label, ImVec4 color) {
            std::vector<double> xy;
            xy.reserve(t.size() * 2);
            const size_t n = std::min(t.size(), e.size());
            for (size_t k = 0; k < n; ++k) {
                const double v = e[k];
                if (!std::isfinite(v)) continue;
                xy.push_back(t[k]); xy.push_back(v);
                if (!data.got_x) { data.xlo = data.xhi = t[k]; data.got_x = true; }
                else { data.xlo = std::min(data.xlo, t[k]); data.xhi = std::max(data.xhi, t[k]); }
            }
            if (xy.empty()) return;
            data.bufs.push_back(std::move(xy));
            data.labels.push_back(label);
            data.colors.push_back(color);
            data.markers.push_back(-1);
        };
        for (size_t j = 0; j < win.members.size(); ++j) {
            const int mi = win.members[j];
            if (mi < 0 || mi >= (int)s.configs.size()) continue;
            const OrderConfig& c = s.configs[(size_t)mi];
            const ErrTimeResult& r = c.errt_result;
            if (!c.errt_last_run_ok || r.t.empty()) continue;
            const ImVec4 base = order_base_color((int)j, win.palette);
            add(r.t, r.e_norm, c.label + (r.norm == kErrTimeNormL2 ? " ||d||2" : " ||d||inf"), base);
            if (win.errt_show_rich && !r.e_rich.empty())
                add(r.t, r.e_rich, c.label + " Richardson p=" + std::to_string(r.order_p), order_shade(base, 0.55f));
            if (win.errt_show_vars) {
                const int nv = (int)r.e_var.size();
                for (int i = 0; i < nv; ++i) {
                    const float k = nv > 1 ? 0.5f + 1.0f * (float)i / (float)(nv - 1) : 1.3f;
                    const std::string vn = (size_t)i < s.vars.size() ? s.vars[(size_t)i] : std::to_string(i);
                    add(r.t, r.e_var[(size_t)i], c.label + " |d " + vn + "|", order_shade(base, k));
                }
            }
        }
    })) view.series_generation = -1;
    auto& data = cache.data;

    // Итог по каждому члену: часть тулбара, скрытый прячет и его.
    for (size_t j = 0; bar && j < win.members.size(); ++j) {
        const int mi = win.members[j];
        if (mi < 0 || mi >= (int)s.configs.size()) continue;
        const OrderConfig& c = s.configs[(size_t)mi];
        if (!c.errt_last_run_ok) continue;
        const ErrTimeResult& r = c.errt_result;
        ImGui::TextDisabled("%s:", c.label.c_str());
        ImGui::SameLine();
        ImGui::Text("max %.3e at t = %.6g, RMS %.3e", r.max_norm, r.t_at_max, r.rms_norm);
        if (ImGui::IsItemHovered()) {
            std::string tip = "Over all " + std::to_string(r.n_steps) + " nodes t_k = k*h:\n";
            char b[200];
            for (size_t i = 0; i < r.max_var.size(); ++i) {
                std::snprintf(b, sizeof(b), "  %s: max %.3e, RMS %.3e\n",
                              i < s.vars.size() ? s.vars[i].c_str() : "?", r.max_var[i],
                              i < r.rms_var.size() ? r.rms_var[i] : 0.0);
                tip += b;
            }
            if (r.stride > 1) {
                std::snprintf(b, sizeof(b), "Plot points hold the maximum over %d nodes each.", r.stride);
                tip += b;
            }
            ImGui::SetTooltip("%s", tip.c_str());
        }
        if (!r.e_rich.empty()) {
            ImGui::SameLine();
            ImGui::TextDisabled("| Richardson (p=%d): max %.3e, RMS %.3e", r.order_p, r.max_rich, r.rms_rich);
        }
        ImGui::SameLine();
        if (r.ref_is_half)
            ImGui::TextDisabled("| ref: h/2, h = %.6g", r.h_eff);
        else
            ImGui::TextDisabled("| ref: %s x%d, h = %.6g", c.errt_ref_scheme.c_str(), r.ref_substeps, r.h_eff);
        if (r.diverged) {
            ImGui::SameLine();
            ImGui::TextColored(readable_text(ImVec4(1.0f, 0.5f, 0.3f, 1.0f)), "| diverged at t = %.6g", r.t_end);
        }
    }

    if (win.members.empty()) {
        ImGui::TextDisabled("The window is empty - add a tab via \"Members...\".");
        return;
    }
    if (!any_data || data.bufs.empty()) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }

    // Стиль — как у 2D-проекций Phase Analysis (apply_line_style).
    apply_line_style(view, order_style_fields(win));
    view.show_zero_x   = false;
    view.show_zero_y   = false;
    view.point_markers = false;
    view.x_axis.name = "t";
    view.x_axis.log_scale = false;
    view.y_axis.name = "E";
    view.point_tag_names.clear();
    view.x_fit_use_explicit = data.got_x && (data.xhi > data.xlo);
    view.x_fit_min = data.xlo;
    view.x_fit_max = data.xhi;

    std::vector<PlotSeriesInput> series;
    std::vector<bool> init_vis, glob_vis;
    for (size_t i = 0; i < data.bufs.size(); ++i) {
        PlotSeriesInput si;
        si.points   = data.bufs[i].data();
        si.n_points = (int)(data.bufs[i].size() / 2);
        si.color    = data.colors[i];
        if (win.custom_line_style) si.color.w = win.line_alpha;
        si.label    = data.labels[i];
        series.push_back(si);
        init_vis.push_back(true);
        glob_vis.push_back(true);
    }
    unsigned long long h = 1469598103934665603ULL;
    for (long long v : key) h = h * 1099511628211ULL + (unsigned long long)v;
    const int sig = (int)((h ^ (h >> 32)) & 0x3FFFFFFFULL) + (int)data.bufs.size();
    if (win.plot_sig != sig) { fit = true; win.plot_sig = sig; }

    view.popup_extras = [&model, &win, &cb, &s]() {
        draw_order_export_items(model, win, cb, s.in_flight);
    };
    ImVec2 avail  = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    view.render(renderer, origin, avail, /*owner_id*/ 0x0BDE2000 + win.id, sig,
                series, init_vis, glob_vis, fit);
}
}

namespace ucuda::ui_detail {
void draw_order_plot_windows(AppModel& model, const GuiCallbacks& cb) {
    auto& renderers = ui_resource<std::map<int, std::unique_ptr<PlotRenderer>>, 3016>();
    auto& heatmaps = ui_resource<std::map<int, std::unique_ptr<HeatmapView>>, 3017>();
    auto& curves = ui_resource<std::map<int, std::unique_ptr<Plot2DView>>, 3018>();
    static int seen_epoch = 0;
    if (session_reloaded(model, AppModel::AppMode::Order, seen_epoch)) {
        invalidate_views(heatmaps);
        invalidate_views(curves);
        order_curve_caches().clear();
        order_errt_caches().clear();
    }

    int to_remove = -1;
    for (int i = 0; i < (int)model.order_plot_windows.size(); ++i) {
        OrderPlotWindow& win = model.order_plot_windows[i];

        auto& renderer = renderers[win.id];
        if (!renderer) renderer = std::make_unique<PlotRenderer>();

        // Автоподпись держит вид окна: пользователь переименовал — не трогаем.
        if (!win.label_is_manual)
            win.label = std::string(order_window_kind_name(win.kind)) + " " + std::to_string(win.id);

        std::string title = win.label + "###orderwin" + std::to_string(win.id)
                          + "_g" + std::to_string(model.order_layout_generation);
        bool open = true;
        float ox = 60.0f + (float)(i % 5) * 35.0f, oy = 80.0f + (float)(i % 5) * 35.0f;
        ImGui::SetNextWindowPos(ImVec2(ox, oy), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(700, 550), ImGuiCond_FirstUseEver);
        if (ImGui::Begin(title.c_str(), &open)) {
            ImGui::PushID(win.id);
            if (win.kind == OrderPlotWindow::Kind::Map) {
                order_curve_caches().erase(win.id);
                auto& hv = heatmaps[win.id];
                if (!hv) {
                    hv = std::make_unique<HeatmapView>();
                    hv->colormap = (HeatmapColormap)colormap_id_or(win.colormap_idx, model.heatmap_colormap);
                }
                draw_order_map_window(model, win, cb, *renderer, *hv);
            } else if (win.kind == OrderPlotWindow::Kind::ErrT) {
                order_curve_caches().erase(win.id);
                auto& cv = curves[win.id];
                if (!cv) {
                    cv = std::make_unique<Plot2DView>();
                    // Ошибка растёт на порядки — по умолчанию лог; меню ПКМ переключает.
                    cv->y_axis.display_log = true;
                }
                draw_order_errt_window(model, win, cb, *renderer, *cv);
            } else if (win.kind == OrderPlotWindow::Kind::Bounds) {
                order_curve_caches().erase(win.id);
                auto& cv = curves[win.id];
                if (!cv) cv = std::make_unique<Plot2DView>();
                draw_order_bounds_window(model, win, cb, *renderer, *cv);
            } else {
                auto& cv = curves[win.id];
                if (!cv) cv = std::make_unique<Plot2DView>();
                draw_order_curve_window(model, win, cb, *renderer, *cv);
            }
            ImGui::PopID();
        }
        ImGui::End();
        if (!open) to_remove = i;
    }
    if (to_remove >= 0) {
        const int id = model.order_plot_windows[to_remove].id;
        renderers.erase(id);
        heatmaps.erase(id);
        curves.erase(id);
        order_curve_caches().erase(id);
        order_errt_caches().erase(id);
        {
            auto& bc = order_bounds_caches();
            for (auto it = bc.begin(); it != bc.end(); )
                it = (std::get<0>(it->first) == id) ? bc.erase(it) : std::next(it);
        }
        model.remove_order_plot_window(to_remove);
    }
}
}


namespace ucuda::ui_detail {
void order_autosave_seed(AppModel& model) {
    model.order_autosave_target       = model.loaded_name;
    model.order_autosave_json         = session_to_json_order(model.order_session);
    model.order_windows_autosave_json = session_to_json_order_windows(model.order_plot_windows);
}
}


namespace ucuda::ui_detail {
int builtin_scheme_order(const std::string& name) {
    for (const auto& b : kBuiltinSchemes)
        if (name == b.name) return b.order;
    return 0;
}
}


namespace ucuda::ui_detail {
void draw_order_target_combo(const char* label, int& target,
                                    const OrderAnalysisSession& s) {
    const std::vector<std::string> names = s.axis_target_names();
    const int n = (int)names.size();
    int sel = target + 1;                       // -1 (h) -> 0
    if (sel < 0 || sel >= n) sel = 0;
    item_w(kComboW);
    if (ImGui::BeginCombo(label, names[(size_t)sel].c_str())) {
        for (int i = 0; i < n; ++i) {
            if (ImGui::Selectable(names[(size_t)i].c_str(), i == sel)) {
                target = i - 1;
                sel = i;
            }
        }
        ImGui::EndCombo();
    }
}
}


namespace ucuda::ui_detail {
void draw_order_axis_block(const char* id, const char* title,
                                  const OrderAnalysisSession& s,
                                  int& target, std::string& lo, std::string& hi,
                                  bool& log_grid, std::string* n_text,
                                  bool enabled, const char* n_label ) {
    ImGui::PushID(id);
    if (!enabled) ImGui::BeginDisabled();
    ucuda::ui_detail::block_separator(title);
    draw_order_target_combo("sweep over", target, s);
    InputNumStr("from", lo, kFieldW);
    ImGui::SameLine();
    InputNumStr("to", hi, kFieldW);
    if (n_text) {
        InputNumStr(n_label, *n_text, kFieldW);
        ImGui::SameLine();
    }
    else {
        ImGui::TextDisabled("resolution: same as X");
        ImGui::SameLine();
    }
    ImGui::Checkbox("log grid", &log_grid);
    if (log_grid) {
        ImGui::SameLine();
        ImGui::TextDisabled("(from/to > 0)");
    }
    if (!enabled) ImGui::EndDisabled();
    ImGui::PopID();
}
}


namespace ucuda::ui_detail {
void draw_order_steps_hint(const OrderConfig& c, const OrderAnalysisSession& s) {
    const double tmax = parse_ratio_or(c.t_max_text, 0.0);
    if (!(tmax > 0.0)) { ImGui::TextDisabled("computing time must be > 0"); return; }

    // Собираем узлы по h — либо одну точку (h не свипуется), либо всю ось.
    std::vector<double> hs;
    auto collect = [&](int target, const std::string& lo_s, const std::string& hi_s,
                       bool log_grid, const std::string& n_s, bool active) {
        if (!active || target != kOrderTargetH) return false;
        double lo = parse_ratio_or(lo_s, 0.0), hi = parse_ratio_or(hi_s, 0.0);
        if (hi < lo) std::swap(lo, hi);
        int n = std::max(1, (int)parse_ratio_or(n_s, 200.0));
        n = std::min(n, 100000);
        hs.reserve((size_t)n);
        if (n == 1) { hs.push_back(lo); return true; }
        if (log_grid && lo > 0.0 && hi > 0.0) {
            const double k = std::log(hi / lo) / (double)(n - 1);
            for (int i = 0; i < n; ++i) hs.push_back(lo * std::exp(k * (double)i));
        } else {
            const double d = (hi - lo) / (double)(n - 1);
            for (int i = 0; i < n; ++i) hs.push_back(lo + d * (double)i);
        }
        return true;
    };
    bool sweeping_h = collect(c.axis_x_target, c.axis_x_lo_text, c.axis_x_hi_text,
                              c.axis_x_log, c.axis_x_n_text, true);
    if (!sweeping_h)
        sweeping_h = collect(c.axis_y_target, c.axis_y_lo_text, c.axis_y_hi_text,
                             c.axis_y_log, c.axis_x_n_text, c.two_d);
    if (!sweeping_h) hs.push_back(parse_ratio_or(c.h_text, 0.0));

    long long n_min = -1, n_max = -1;
    int n_bad = 0, n_zero = 0;
    double worst_rel = 0.0;
    for (double h : hs) {
        if (!(h > 0.0)) { ++n_zero; continue; }
        const long long N = order_steps_for_h(h, tmax, c.snap_steps);
        if (n_min < 0 || N < n_min) n_min = N;
        if (n_max < 0 || N > n_max) n_max = N;
        const double r = tmax / h;
        const double frac = std::fabs(r - std::floor(r + 0.5));
        if (frac > 1e-9 * std::max(1.0, r)) {
            ++n_bad;
            // Насколько подогнанный шаг отличается от заданного.
            const double h_snap = tmax / (double)std::max(1LL, (long long)(r + 0.5));
            const double rel = std::fabs(h_snap - h) / h;
            if (rel > worst_rel) worst_rel = rel;
        }
    }

    if (n_zero > 0)
        ImGui::TextColored(readable_text(ImVec4(1.0f, 0.45f, 0.45f, 1.0f)),
                           "%d nodes with h <= 0 - they will go to diverged", n_zero);
    if (n_min > 0) {
        // Полная работа: 7 вызовов КРС на грубый шаг (1 + 2 + 4).
        double total = 0.0;
        for (double h : hs) if (h > 0.0) total += (double)order_steps_for_h(h, tmax, c.snap_steps);
        double cells = (double)hs.size();
        if (c.two_d) {
            // Сетка квадратная, N x N. hs уже содержит узлы той оси, что свипует
            // h; вторая добавляет N. Если h не свипуется вовсе, hs — одна точка,
            // и обе оси идут множителями (раньше здесь терялась одна из них, и
            // счётчик ячеек занижался ровно в N раз).
            const double n_side = std::max(1.0, parse_ratio_or(c.axis_x_n_text, 200.0));
            const double other = sweeping_h ? n_side : n_side * n_side;
            total *= other;
            cells *= other;
        }
        if (n_min == n_max)
            ImGui::Text("steps per cell: %lld   cells: %.0f   KRS calls: %.3g",
                        n_min, cells, total * 7.0);
        else
            ImGui::Text("steps per cell: %lld..%lld   cells: %.0f   KRS calls: %.3g",
                        n_min, n_max, cells, total * 7.0);
    }

    if (n_bad > 0) {
        if (c.snap_steps) {
            ImGui::TextColored(readable_text(ImVec4(0.65f, 0.85f, 0.65f, 1.0f)),
                "t_max/h non-integer in %d of %d nodes - h snapped to t_max/N (shift up to %.2g%%)",
                n_bad, (int)hs.size(), worst_rel * 100.0);
        } else {
            ImGui::TextColored(readable_text(ImVec4(1.0f, 0.55f, 0.3f, 1.0f)),
                "t_max/h non-integer in %d of %d nodes: the final time drifts, "
                "an O(h) contribution is mixed into the error", n_bad, (int)hs.size());
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "The number of steps must be an integer. Otherwise the actual final time\n"
                "N*h differs from t_max and CHANGES from node to node, and the difference\n"
                "between solutions at different moments in time is of order O(h) - it\n"
                "adds up with the measured error and for schemes of order above\n"
                "the first it drowns it out completely.\n"
                "\"snap h -> t_max/N\" removes this by shifting the step itself slightly.");
    }
}
}


namespace ucuda::ui_detail {
void draw_order_stability_block(AppModel& model, const OrderAnalysisSession& s,
                                       OrderConfig& c) {
    if (!ucuda::ui_detail::block_header("Stability region", ImGuiTreeNodeFlags_DefaultOpen)) return;

    if ((int)s.vars.size() != 2) {
        ImGui::TextColored(readable_text(ImVec4(1.0f, 0.4f, 0.4f, 1.0f)),
            "This computation needs the linear two-dimensional test system: %d state "
            "variables are loaded.", (int)s.vars.size());
        ImGui::TextDisabled("Library -> Dahlquist 2D. Run will be refused until then.");
        ImGui::Separator();
    }

    InputNumStr("asymmetry k", c.stab_k_text, kFieldW);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Drives the test matrix along the family with the SAME spectrum:\n"
            "a = k*d and d = 2*sigma/(1+k), so the trace stays 2*sigma and the\n"
            "determinant stays sigma^2 + omega^2 whatever k is. k = 1 gives the\n"
            "symmetric case a = d; the further from 1, the stronger the\n"
            "asymmetry the scheme has to cope with at unchanged eigenvalues.\n"
            "k = -1 is forbidden: it makes d infinite.");
    InputNumStr("ratio r", c.stab_r_text, kFieldW);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Scales the off-diagonal entries: b = r*c, and c itself is built so\n"
            "that the spectrum does not move. Must be NEGATIVE - with r >= 0 the\n"
            "radicand of c is not positive and no real test matrix exists.");
    InputNumStr("step h", c.stab_h_text, kFieldW);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "The step whose transition operator is diagonalised. Exactly ONE\n"
            "step is taken: the test is linear and homogeneous, so R_N = R^N and\n"
            "more steps would only raise the same spectrum to a power.\n"
            "\n"
            "At h = 1 (the default) the axes read directly as h*sigma and\n"
            "h*omega - the classic dimensionless picture.");

    const double k_val = parse_num(c.stab_k_text, 1.0);
    const double r_val = parse_num(c.stab_r_text, -1.0);
    if (k_val == -1.0)
        ImGui::TextColored(readable_text(ImVec4(1.0f, 0.4f, 0.4f, 1.0f)),
                           "k = -1: d = 2*sigma/(1+k) is undefined, the run will be refused.");
    if (!(r_val < 0.0))
        ImGui::TextColored(readable_text(ImVec4(1.0f, 0.4f, 0.4f, 1.0f)),
                           "r must be negative: with r >= 0 the matrix has no real form.");

    ImGui::Separator();
    InputNumStr("sigma from", c.stab_sig_lo_text, kFieldW);
    ImGui::SameLine();
    InputNumStr("to##sigma", c.stab_sig_hi_text, kFieldW);
    InputNumStr("omega from", c.stab_om_lo_text, kFieldW);
    ImGui::SameLine();
    InputNumStr("to##omega", c.stab_om_hi_text, kFieldW);
    InputNumStr("resolution (both axes)", c.stab_n_text, kFieldW);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "The map is always N x N: a cell costs two steps of the scheme, so\n"
            "even 512 x 512 is cheaper than a single node of the p(h) curve.");

    ImGui::Separator();
    InputNumStr("preference tolerance", c.stab_pref_tol_text, kFieldW);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Preference region (Fedoseev, Karimov, Legat, Butusov, Mathematics\n"
            "10:4327, 2022): the part of the stability region where one step is\n"
            "also ACCURATE. A cell is preferred when rho <= 1 and\n"
            "\n"
            "    err = ||e^{hA} - R||_2 / ||e^{hA}||_2 <= tolerance,\n"
            "\n"
            "e^{hA} being the exact transition operator of the test problem and\n"
            "R the one of the scheme (spectral norm). The paper uses 1, i.e. up\n"
            "to 100%% error per step - a 'not nonsense' boundary rather than an\n"
            "accuracy target; 0.1 or 0.01 show where the scheme is actually good.\n"
            "\n"
            "Applied when drawing, not during the run: change it and the overlay\n"
            "follows at once, no rerun needed.");
    const double tol_val = parse_num(c.stab_pref_tol_text, 1.0);
    if (!(tol_val > 0.0))
        ImGui::TextColored(readable_text(ImVec4(1.0f, 0.75f, 0.4f, 1.0f)),
                           "tolerance <= 0: the preference region is empty by definition.");

    // Селекторы индексов a[]. Подписи те же, что у селектора оси свипа, минус
    // "h" и минус a[0] — коэффициент симметрии s занят схемой.
    ImGui::Separator();
    std::vector<std::string> items;
    items.reserve(s.params.size());
    for (size_t i = 0; i < s.params.size(); ++i)
        items.push_back(s.params[i] + " (a[" + std::to_string(i + 1) + "])");
    std::vector<const char*> item_ptrs;
    item_ptrs.reserve(items.size());
    for (const auto& it : items) item_ptrs.push_back(it.c_str());

    auto idx_combo = [&](const char* label, int& target) {
        int sel = target - 1;   // a[1] — первый элемент списка
        if (sel < 0 || sel >= (int)item_ptrs.size()) sel = -1;
        item_w(kFieldW + 60.0f, combo_items_need(item_ptrs.empty() ? nullptr : item_ptrs.data(), (int)item_ptrs.size()));
        if (ImGui::Combo(label, &sel, item_ptrs.empty() ? nullptr : item_ptrs.data(),
                         (int)item_ptrs.size()))
            target = sel + 1;
    };
    idx_combo("matrix a##stab", c.stab_idx_a);
    idx_combo("matrix b##stab", c.stab_idx_b);
    idx_combo("matrix c##stab", c.stab_idx_c);
    idx_combo("matrix d##stab", c.stab_idx_d);
    ImGui::TextDisabled("Which parameter of the system carries which entry of "
                        "[[a, b], [c, d]].");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "The entries are recomputed in every node from (sigma, omega, k, r)\n"
            "and written into these a[] slots, so whatever you set here is\n"
            "overwritten during the run - the values on the Parameters panel are\n"
            "not used for them.\n"
            "\n"
            "For Library -> Dahlquist 2D this is a, b, c, d = a[1..4]. A custom\n"
            "KRS for this computation must address exactly these slots.");

    (void)model;
}
}


namespace ucuda::ui_detail {
OrderPlotWindow::Kind order_kind_for(const OrderConfig& c) {
    if (c.calc_kind == kOrderCalcPerf) return OrderPlotWindow::Kind::Perf;
    if (c.calc_kind == kOrderCalcErrT) return OrderPlotWindow::Kind::ErrT;
    // Область устойчивости — карта по построению: сетка всегда (sigma, omega).
    if (c.calc_kind == kOrderCalcStab) return OrderPlotWindow::Kind::Map;
    return c.two_d ? OrderPlotWindow::Kind::Map : OrderPlotWindow::Kind::P;
}
}


namespace ucuda::ui_detail {
void order_rehome_config(AppModel& model, int idx) {
    OrderAnalysisSession& s = model.order_session;
    if (idx < 0 || idx >= (int)s.configs.size()) return;
    const OrderConfig& c = s.configs[(size_t)idx];
    const OrderPlotWindow::Kind want = order_kind_for(c);
    auto& wins = model.order_plot_windows;

    bool placed = false;
    std::vector<int> vacated;   // id окон, опустевших от этого ухода
    for (auto& w : wins) {
        auto& mem = w.members;
        const bool has = std::find(mem.begin(), mem.end(), idx) != mem.end();
        if (!has) continue;
        if (order_member_fits(c, w.kind)) { placed = true; continue; }
        mem.erase(std::remove(mem.begin(), mem.end(), idx), mem.end());
        w.plot_sig = -1;
        if (mem.empty()) vacated.push_back(w.id);
    }
    if (!placed) {
        for (auto& w : wins) {
            if (w.kind != want) continue;
            if (want == OrderPlotWindow::Kind::Map && !w.members.empty()) continue;
            w.members.push_back(idx);
            w.plot_sig = -1;
            placed = true;
            break;
        }
    }
    if (!placed && !vacated.empty()) {
        for (auto& w : wins) {
            if (w.id != vacated.front()) continue;
            // То же, что делает смена вида в комбо "Plot windows".
            w.kind = want;
            w.members.assign(1, idx);
            if (want == OrderPlotWindow::Kind::P) w.y_log = false;
            w.plot_sig = -1;
            placed = true;
            break;
        }
        vacated.erase(vacated.begin());
    }
    if (!placed) model.add_order_plot_window(want, { idx });
    wins.erase(std::remove_if(wins.begin(), wins.end(), [&](const OrderPlotWindow& w) {
                   return w.members.empty() && !w.label_is_manual
                       && std::find(vacated.begin(), vacated.end(), w.id) != vacated.end();
               }), wins.end());
    model.order_plot_windows_dirty = true;
}
}


namespace ucuda::ui_detail {
void draw_order_controls(AppModel& model, SystemLibrary& /*lib*/) {
    OrderAnalysisSession& s = model.order_session;
    // Чей конфиг правит эта панель — для ПКМ-рассылки "во все такие же".
    model.broadcast_source_tab = BroadcastTab::Order;
    model.broadcast_source_idx = s.active_config_index;
    if (s.configs.empty()) {
        // Система загружена, а вкладок нет: так остаются сессии, у которых
        // удалили последнюю вкладку до запрета на это (см. draw_config_tab_bar).
        // Таб-бара с "+" здесь нет, поэтому без этой кнопки вкладку было не
        // вернуть ничем. Новая вкладка строится из записи системы — теми же
        // умолчаниями, что и при первом входе в Order.
        if (s.loaded_system_name.empty() || s.loaded_system_name != model.name) {
            ImGui::TextDisabled("No system loaded. Pick a system in the list above.");
            return;
        }
        ImGui::TextDisabled("This system has no Order tabs.");
        if (ImGui::Button("New tab from system defaults") && model.start_order_analysis()
            && !s.configs.empty())
            order_rehome_config(model, 0);
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
        s.active_config_index = 0;

    // Run / Run all... — над таб-баром, как у Basins/FastSync: управляют активным
    // конфигом и должны быть доступны без прокрутки. Ctrl+R / Ctrl+Shift+R приходят
    // вместе с общим хелпером; "Run all..." кладёт отмеченные индексы в
    // model.order_queue, которую draw_gui тикает после poll'ов.
    {
        const bool no_cfg = s.configs.empty();
        RunAllGroup g;
        g.pick_id = "porder_";
        g.n       = (int)s.configs.size();
        g.label   = [&s](int i) { return s.configs[i].label; };
        g.enqueue = [&model](int i) { model.enqueue_order(i); };
        draw_run_and_run_all("##run_all_order", s.in_flight, no_cfg,
                             s.in_flight || no_cfg, 160.0f,
                             [&model, &s]() {
                                 model.run_order(s.active_config_index);
                             },
                             [&s]() {
                                 // Локальный Stop остаётся рядом с Run: глобальный
                                 // Cancel живёт в статус-баре, но привычка жать Stop
                                 // тут же, где Run, старше вкладки.
                                 ImGui::SameLine();
                                 if (!s.in_flight) ImGui::BeginDisabled();
                                 if (ImGui::Button("Stop", ImVec2(90, 0))) s.request_cancel();
                                 if (!s.in_flight) ImGui::EndDisabled();
                             },
                             { g },
                             [&model]() { model.start_next_in_order_queue(); },
                             model.order_queue().size());
        ImGui::SameLine();
        ImGui::Checkbox("Auto-compute", &s.auto_compute);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Recompute the active tab as soon as its settings change.\n"
                              "A change made during a run starts a new run after it.");
    }
    ImGui::Separator();

    const TabBarResult tabs = draw_config_tab_bar(
        "##order_tabs", "order_tab_", (int)s.configs.size(),
        s.in_flight, s.running_config_index,
        [&s](int i) { return s.configs[i].label; },
        {},
        [&s, &model]() {
            s.add_config();
            // Новая вкладка обязана где-то показаться: кладём её в первое
            // подходящее окно-оверлей, а если такого нет — заводим своё.
            const int ni = (int)s.configs.size() - 1;
            if (ni < 0) return;
            const OrderPlotWindow::Kind want = order_kind_for(s.configs[(size_t)ni]);
            if (want != OrderPlotWindow::Kind::Map) {
                for (auto& w : model.order_plot_windows)
                    if (w.kind == want) {
                        w.members.push_back(ni);
                        model.order_plot_windows_dirty = true;
                        return;
                    }
            }
            model.add_order_plot_window(want, { ni });
        });
    if (tabs.active    >= 0) s.active_config_index = tabs.active;
    if (tabs.to_remove >= 0) model.remove_order_config(tabs.to_remove);

    if (s.configs.empty()) {
        ImGui::TextDisabled("No order configs. Press '+' to add one.");
        return;
    }
    if (s.active_config_index >= (int)s.configs.size())
        s.active_config_index = (int)s.configs.size() - 1;

    const int idx = s.active_config_index;
    OrderConfig& c = s.configs[(size_t)idx];
    // Поля активного конфига: ПКМ — во все конфиги вкладки и сброс к умолчанию.
    const ConfigFieldScope<OrderConfig> cfg_field_scope(model, "Order", s.configs, idx);

    // Шапка в три строки: метка; вид расчёта; устройство и точность.
    draw_label_rename("##order_label", c.label, kFieldW);
    const int calc_kind_before = c.calc_kind;
    ImGui::RadioButton("order p", &c.calc_kind, kOrderCalcOrder);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Accuracy order and the error curve over the node grid.");
    ImGui::SameLine();
    ImGui::RadioButton("performance", &c.calc_kind, kOrderCalcPerf);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Timing over the same X axis: on every node the kernel that\n"
            "integrates a single trajectory with that node's step is launched\n"
            "several times under cudaEvents. The plot then shows time against\n"
            "the error achieved - what that accuracy costs.");
    ImGui::SameLine();
    ImGui::RadioButton("stability", &c.calc_kind, kOrderCalcStab);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Stability region on the two-dimensional Dahlquist test problem.\n"
            "The grid runs over the eigenvalues of the test matrix: sigma along\n"
            "X, omega along Y. In every node the matrix is rebuilt from\n"
            "(sigma, omega) and the settings k and r, one step of the scheme is\n"
            "taken from each basis vector, and the map shows the spectral radius\n"
            "of the resulting amplification matrix. The boundary is rho = 1.\n"
            "\n"
            "Needs the linear two-dimensional test system - Library ->\n"
            "Dahlquist 2D. Any other system is refused.");
    ImGui::SameLine();
    ImGui::RadioButton("error(t)", &c.calc_kind, kOrderCalcErrT);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Error of the scheme in time on this system: |x_h(t) - x_ref(t)| at the\n"
            "nodes t_k = k*h against a reference - the same scheme with h/2 (plus the\n"
            "Richardson estimate) or another scheme with h/m. The reference runs in\n"
            "lockstep, so the grids coincide exactly and nothing is interpolated.\n"
            "One trajectory, computed on the CPU.");
    if (c.calc_kind != calc_kind_before) order_rehome_config(model, idx);
    const bool errt_mode = (c.calc_kind == kOrderCalcErrT);

    // Устройство расчёта. Обе ветки считают по одним формулам и одним порогам,
    // но замер времени у них меряет РАЗНОЕ: GPU — ядро с replicas нитями,
    // CPU — одну траекторию последовательно. Числа между устройствами поэтому
    // не сопоставимы, и это не баг, а два разных вопроса к схеме.
    // Error(t) — одна траектория, только CPU: переключатель устройства гасится,
    // настройка вкладки при этом не трогается (её читают остальные расчёты).
    const bool gpu_eff = c.use_gpu && !errt_mode;
    if (errt_mode) ImGui::BeginDisabled();
    int ord_dev = gpu_eff ? 0 : 1;
    ImGui::RadioButton("GPU##ord_dev", &ord_dev, 0);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Kernel compiled by NVRTC (kernels/order.template.cu).");
    ImGui::SameLine();
    ImGui::RadioButton("CPU##ord_dev", &ord_dev, 1);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "The same KRS body, compiled to a native step by cl.exe instead of\n"
            "NVRTC, run sequentially on one core. Same formulas, same floor and\n"
            "divergence thresholds, so the order diagram is directly comparable\n"
            "with the GPU one.\n"
            "Performance means something else here: the time of ONE trajectory,\n"
            "not the throughput of a loaded GPU - replicas are ignored.");
    if (errt_mode) ImGui::EndDisabled();
    else           c.use_gpu = (ord_dev == 0);

    // Точность арифметики. Есть только у CPU: на GPU ядро собирается в numb,
    // и выбор там был бы обманом — цифры те же, смысл другой.
    ImGui::SameLine();
    ImGui::TextDisabled("|"); ImGui::SameLine();
    if (gpu_eff) ImGui::BeginDisabled();
    const char* prec_items[] = { "double", "double-double", "quad-double" };
    item_w(150.0f, combo_items_need(prec_items, IM_ARRAYSIZE(prec_items)));
    ImGui::Combo("precision##ord_prec", &c.cpu_precision, prec_items, IM_ARRAYSIZE(prec_items));
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Arithmetic of the CPU branch.\n"
            "double        - 16 digits, bit-for-bit the same as the GPU.\n"
            "double-double - 32 digits (106-bit mantissa), about 35x slower.\n"
            "quad-double   - 62 digits (212-bit mantissa), about 600x slower.\n"
            "\n"
            "The order estimate subtracts nearby solutions, so it sits on a\n"
            "rounding floor of about 2*eps*|y|*sqrt(N) - near 3e-13 in double.\n"
            "A scheme of order p gives E1 ~ h^p, so from p = 8 on there is\n"
            "almost no usable window in h left: the curve hits the floor before\n"
            "the scheme reaches its asymptotic regime. double-double lowers the\n"
            "floor by 16 decades, quad-double by 47.\n"
            "\n"
            "Measured on Extr(CD|1,2,3,4): the floor sits at 5e-13 in double,\n"
            "2e-26 in double-double, below 1e-50 in quad-double.\n"
            "\n"
            "The same KRS body is rebuilt with numb = ucuda::dd / ucuda::qd;\n"
            "each build is cached separately, so switching does not recompile.");
    if (gpu_eff) ImGui::EndDisabled();
    if (!gpu_eff && c.cpu_precision > 0) order_float_build_warning();

    if (!gpu_eff) {
        std::string why;
        if (!cpu_compiler_available(&why))
            ImGui::TextColored(readable_text(ImVec4(1.0f, 0.5f, 0.3f, 1.0f)),
                               "CPU backend unavailable: %s", why.c_str());

        // Подмена допуска Ньютона обязана быть видимой: это чужая настройка,
        // и без строки пользователь не поймёт, почему неявная схема на этой
        // вкладке считается иначе, чем на остальных.
        const double ntol = order_newton_tol_for_precision(c.cpu_precision);
        if (ntol > 0.0) {
            ImGui::TextDisabled("Implicit schemes: Newton tolerance at most %.0e, at least 12 "
                                "iterations (Integration: %s, %s)", ntol, c.newton.tol.c_str(),
                                c.newton.max_iters.c_str());
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Newton stops at ||dX|| < tol, so the system-wide setting would\n"
                    "cap the result long before the arithmetic does: in double-double\n"
                    "the rounding floor sits near 1e-29, and a 1e-10 tolerance would\n"
                    "throw away nineteen decades of it.\n"
                    "\n"
                    "This tab measures the order of the SCHEME, not the accuracy of\n"
                    "the solver, so the tolerance is tightened to match the chosen\n"
                    "arithmetic - never loosened: a stricter setting of your own is\n"
                    "kept. Explicit schemes are unaffected.");
        }
    }
    // Дискретное отображение шага не имеет вовсе, уточнять нечего: обе
    // диаграммы вкладки меряют, как ошибка убывает с h, и на карте x_{n+1} =
    // f(x_n) обе разности тождественно нулевые.
    if (s.sys.is_map)
        ImGui::TextColored(readable_text(ImVec4(1.0f, 0.75f, 0.4f, 1.0f)),
                           "Discrete map: there is no step to refine, the accuracy order is undefined.");
    ImGui::Separator();

    // ---- Интегрирование ----
    if (ucuda::ui_detail::block_header("Integration", ImGuiTreeNodeFlags_DefaultOpen)) {
        draw_scheme_combo("scheme", c.scheme, s.custom_schemes, {}, &s.enabled_builtin_schemes,
                          &model, model.is_map, &s.wrapper_schemes);
        // ПКМ по полю — та же рассылка по вкладкам, что у блока Integration остальных вкладок.
        auto bmenu = [&model](BroadcastField f, const char* lbl, const std::string& v) {
            return [&model, f, lbl, &v] { field_apply_all_menu(&model, f, {}, lbl, v); };
        };
        if (scheme_uses_symmetry(c.scheme, s.custom_schemes))
            InputNumStr("symmetry s", c.symmetry_s, kFieldW, bmenu(BroadcastField::SymmetryS, "symmetry s", c.symmetry_s));
        if (scheme_uses_cd_fp_iters(c.scheme)) {
            InputNumStr("FP iterations n", c.cd_fp_iters, kFieldW,
                        bmenu(BroadcastField::CdFpIters, "FP iterations n", c.cd_fp_iters));
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", kCdFpItersTip);
        }
        if (scheme_uses_newton(c.scheme))
            draw_newton_fields(c.newton, kFieldW, "order_newton");
        // Область устойчивости берёт из этого блока только схему и s: она
        // делает ОДИН шаг, поэтому ни времени счёта, ни порога расхождения,
        // ни подгонки числа шагов у неё нет, а собственный шаг живёт в своём
        // блоке ниже — рядом с k и r, от которых зависит смысл осей.
        if (c.calc_kind == kOrderCalcStab) {
            ImGui::TextDisabled("The step and the grid of the stability region live in "
                                "\"Stability region\" below.");
        } else {
        const bool h_swept = !errt_mode && ((c.axis_x_target == kOrderTargetH)
                          || (c.two_d && c.axis_y_target == kOrderTargetH));
        if (!model.is_map) {
            if (h_swept) ImGui::BeginDisabled();
            InputNumStr("h", c.h_text, kFieldW, bmenu(BroadcastField::StepH, "h", c.h_text));
            if (h_swept) ImGui::EndDisabled();
            if (h_swept) {
                ImGui::SameLine();
                ImGui::TextDisabled("(swept along the axis)");
            }
        }
        InputNumStr(model.is_map ? "iterations" : "computing time", c.t_max_text, kFieldW,
                    bmenu(BroadcastField::TMax, model.is_map ? "iterations" : "computing time", c.t_max_text));
        InputNumStr("max value", c.max_value_text, kFieldW,
                    bmenu(BroadcastField::MaxValue, "max value", c.max_value_text));

        ImGui::Checkbox("snap h -> t_max/N", &c.snap_steps);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Adjust the step so that the number of steps is an integer.\n"
                "Turning this off reproduces the behaviour of the other tabs\n"
                "(floor(t_max/h)) at the cost of a parasitic O(h) contribution to the error.");
        if (!errt_mode) {
        ImGui::SameLine();
        ImGui::Checkbox("endpoint only", &c.endpoint_only);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Measure the difference only at t = t_max, not the maximum over the whole\n"
                "trajectory. On chaotic systems the maximum over the trajectory\n"
                "quickly starts reflecting divergence rather than the order of the scheme.");

        draw_order_steps_hint(c, s);
        }
        }
    }

    // ---- Error vs time ----
    if (errt_mode && ucuda::ui_detail::block_header("Error vs time", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextDisabled("reference:"); ImGui::SameLine();
        ImGui::RadioButton("same scheme, h/2##errtref", &c.errt_ref, kErrTimeRefHalf);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "The same scheme with half the step, two substeps per step of x_h.\n"
                "|x_h - x_h/2| is the error of x_h up to a factor (1 - 2^-p); the\n"
                "Richardson curve restores it: |x_h - x_h/2| * 2^p / (2^p - 1).");
        ImGui::SameLine();
        ImGui::RadioButton("other scheme, h/m##errtref", &c.errt_ref, kErrTimeRefScheme);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Another scheme (DOP853 by default) with m substeps per step of x_h.\n"
                "With its own error far below that of x_h, the difference IS the\n"
                "error of the scheme under test.");
        if (c.errt_ref == kErrTimeRefScheme) {
            draw_scheme_combo("reference scheme", c.errt_ref_scheme, s.custom_schemes, {},
                              &s.enabled_builtin_schemes, &model, s.sys.is_map, &s.wrapper_schemes,
                              /*broadcast*/ false);
            InputNumStr("substeps m", c.errt_ref_substeps_text, kFieldW);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Reference steps per step of the tested scheme: the reference step is h/m.");
        } else {
            InputNumStr("order p (Richardson)", c.errt_order_text, kFieldW);
            const int pp = order_scheme_passport_order(s, c.scheme);
            const int pe = order_errt_effective_order(s, c);
            ImGui::SameLine();
            if (pe > 0)
                ImGui::TextDisabled(c.errt_order_text.empty() ? "(nominal p = %d)" : "(p = %d)", pe);
            else
                ImGui::TextDisabled("(order unknown: enter p for the Richardson estimate)");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Empty - the nominal order of the scheme%s.",
                                  pp > 0 ? "" : " (not known for this scheme)");
        }
        const char* norms[] = { "L-inf (max over variables)", "L2 (Euclidean)" };
        item_w(kComboW, combo_items_need(norms, IM_ARRAYSIZE(norms)));
        ImGui::Combo("norm##errt", &c.errt_norm, norms, IM_ARRAYSIZE(norms));
        const double hh = parse_num(c.h_text, 0.01), tt = parse_num(c.t_max_text, 10.0);
        const long long N = order_steps_for_h(hh, tt, c.snap_steps);
        const int m = (c.errt_ref == kErrTimeRefHalf) ? 2
                    : std::max(1, (int)parse_num(c.errt_ref_substeps_text, 4.0));
        ImGui::TextDisabled("%lld nodes t_k = k*h, the reference takes %d substep%s each",
                            N, m, m == 1 ? "" : "s");
    }

    // ---- Stability region ----
    if (c.calc_kind == kOrderCalcStab) draw_order_stability_block(model, s, c);

    // ---- Performance ----
    if (c.calc_kind == kOrderCalcPerf &&
        ucuda::ui_detail::block_header("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("measurements", c.perf_repeats_text, kFieldW);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "How many times to run the computation on every node. Minimum,\n"
                "maximum and average are taken over these measurements - all\n"
                "three go onto the plot.");
        InputNumStr("warmup runs", c.perf_warmup_text, kFieldW);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Launches that do not count, made before the measurements. The\n"
                "very first launch drags the code load into the context along with\n"
                "it and is therefore always slower than the rest.");
        ImGui::Separator();
        // Режим шага — первым: от него зависит, какой эталон у вкладки (ровно один).
        const bool perf_ad_on = !s.sys.is_map && c.adaptive.enabled;
        if (!s.sys.is_map) {
            draw_adaptive_block("order_perf_ad", c.adaptive, c.scheme, 0, s.sys.is_map, &model);
            if (c.adaptive.enabled) {
                InputNumStr("tol from", c.perf_tol_lo_text, kFieldW);
                InputNumStr("tol to", c.perf_tol_hi_text, kFieldW);
                InputNumStr("tol points", c.perf_tol_n_text, kFieldW);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "The nodes of the adaptive benchmark: a log grid of tol; on every node\n"
                        "rtol = tol and atol = tol * (atol / rtol) of the settings above. The\n"
                        "Sweep axis below is not used. Every node integrates [0, t_max] (the\n"
                        "step only, no output) and is timed like the fixed step: on the GPU by\n"
                        "a kernel, on the CPU by the same driver compiled with cl.exe (double).\n"
                        "Steps, rejections and f evaluations come from the controller.");
            }
            ImGui::Separator();
        }
        if (!perf_ad_on) {
            // Эталон постоянного шага: чем считаем «точный ответ» для Eref по оси X.
            draw_scheme_combo("reference method", c.perf_ref_scheme, s.custom_schemes, {},
                              &s.enabled_builtin_schemes, &model, s.sys.is_map, &s.wrapper_schemes,
                              /*broadcast*/ false);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "The method that stands for the exact answer in Eref =\n"
                    "max|y_h - y_ref|. DOPRI78 by default: at the steps in use its\n"
                    "own error sits below the machine precision of the solution.\n"
                    "With \"endpoint only\" and \"fit h to t_max\" Eref is the error at t_max:\n"
                    "it is then also this tab's E(T), comparable with adaptive tabs.");
            InputNumStr("reference substeps", c.perf_ref_substeps_text, kFieldW);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Reference steps per one step of the scheme under test. 0 turns\n"
                    "the reference off entirely - the branch is then not even compiled\n"
                    "into the kernel. Keep it above 1 when the reference method is the\n"
                    "same as the tested one: at an equal step that would be the very\n"
                    "same arithmetic and the difference identically zero.");
            // y*(T) для E(T) — тот же эталон, что у адаптивного шага: с ним кривые постоянного
            // и адаптивного шага ложатся на одну диаграмму (X: E(T) в окне Performance).
            const char* refs_fx[] = { "none", "DOP853 in dd (CPU)", "DOP853 in qd (CPU)" };
            int sel_fx = (c.perf_end_ref == 1 || c.perf_end_ref == 2) ? c.perf_end_ref : 0;
            item_w(kComboW, combo_items_need(refs_fx, IM_ARRAYSIZE(refs_fx)));
            if (ImGui::Combo("reference y*(T)##fixed", &sel_fx, refs_fx, IM_ARRAYSIZE(refs_fx)))
                c.perf_end_ref = sel_fx;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "E(T) = max|y(T) - y*(T)| at t_max against DOP853 in double-double or\n"
                    "quad-double on the CPU - the same y*(T) the adaptive step is measured\n"
                    "against, so fixed-step and adaptive curves can share one diagram\n"
                    "(X: E(T) in the Performance window). Needs \"fit h to t_max\": a node\n"
                    "that ends at N*h != t_max has no E(T). Computed after the timing and\n"
                    "not part of it. none: E(T) is Eref at t_max (\"endpoint only\").");
            if (c.perf_end_ref == 1 || c.perf_end_ref == 2) order_float_build_warning();
        } else {
            // Эталон адаптивного шага: y*(T) высокой точности — его единственная ошибка E(T).
            const char* refs[] = { "DOP853 in dd (CPU)", "DOP853 in qd (CPU)" };
            int sel = (c.perf_end_ref == 2) ? 1 : 0;
            item_w(kComboW, combo_items_need(refs, IM_ARRAYSIZE(refs)));
            if (ImGui::Combo("reference y*(T)", &sel, refs, IM_ARRAYSIZE(refs))) c.perf_end_ref = sel + 1;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "E(T) = max|y(T) - y*(T)| at the end of the interval (t_max), against\n"
                    "DOP853 with a fixed step in double-double (~32 digits) or quad-double\n"
                    "(~62 digits) arithmetic on the CPU. Its step count doubles until two\n"
                    "successive answers agree, so y* is exact to far below any error on\n"
                    "the plot. The adaptive step has no h grid, so E(T) is its only error.");
            order_float_build_warning();
        }
        ImGui::Separator();
        // Реплики — понятие ядра: столько одинаковых нитей грузят GPU. На CPU
        // прогон последовательный, и поле нечего означать, поэтому гасим его,
        // а не молча игнорируем значение.
        if (!c.use_gpu) ImGui::BeginDisabled();
        InputNumStr("replicas", c.perf_replicas_text, kFieldW);
        if (!c.use_gpu) ImGui::EndDisabled();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                c.use_gpu
                ? "How many IDENTICAL tasks the timed launch computes.\n"
                  "1 - the time of a single computation (the GPU mostly idles);\n"
                  "a large number - the throughput of a loaded GPU, where the\n"
                  "time per task = measured / replicas."
                : "GPU only: the CPU branch runs one trajectory sequentially,\n"
                  "so the measurement is always the time of a single task.");

        const int n_nodes = c.adaptive.enabled ? std::max(1, (int)parse_num(c.perf_tol_n_text, 11.0))
                                               : std::max(1, (int)parse_num(c.axis_x_n_text, 200.0));
        const int reps    = std::max(1, (int)parse_num(c.perf_repeats_text, 20.0));
        const int wu      = std::max(0, (int)parse_num(c.perf_warmup_text, 2.0));
        const long long launches = (long long)n_nodes * (long long)(reps + wu);
        ImGui::TextDisabled("%d nodes x %d launches = %lld; plus the order pass for the errors",
                            n_nodes, reps + wu, launches);
        if (launches > 20000)
            ImGui::TextColored(readable_text(ImVec4(1.0f, 0.75f, 0.4f, 1.0f)),
                               "a great many launches: the measurement will take %d times longer "
                               "than a plain run",
                               reps + wu);
        ImGui::TextDisabled("Only the kernel launch itself is timed: no H2D/D2H, no compilation.");
        // Split compilation не встраивает шаг в ядро, и та же задача считается
        // в разы дольше (на Лоренце с явным Эйлером — в 3.6 раза). Настройка
        // глобальная, поэтому кривые, снятые при разном её положении, сравнивать
        // между собой нельзя.
        if (model.nvrtc_rdc)
            ImGui::TextColored(readable_text(ImVec4(1.0f, 0.75f, 0.4f, 1.0f)),
                               "Split compilation is on (Settings): the step is not inlined into "
                               "the kernel,\nso the measured time is several times larger than "
                               "with it off.");
    }

    // ---- Оси ----
    // У области устойчивости оси заданы смыслом расчёта (sigma и omega —
    // вещественная и мнимая части собственных чисел тестовой матрицы), и
    // общий селектор «h + элементы a[]» к ней неприменим: показывать его
    // рядом с настоящими границами значило бы предлагать выбор, которого нет.
    if (c.calc_kind != kOrderCalcStab && c.calc_kind != kOrderCalcErrT &&
        ucuda::ui_detail::block_header("Sweep", ImGuiTreeNodeFlags_DefaultOpen)) {
        const bool perf_mode = (c.calc_kind == kOrderCalcPerf);
        const bool perf_ad   = perf_mode && c.adaptive.enabled && !s.sys.is_map;
        if (perf_ad)
            ImGui::TextDisabled("Adaptive step: the benchmark runs over the tol nodes (Performance above);\n"
                                "the X axis below is not used.");
        if (perf_ad) ImGui::BeginDisabled();
        if (perf_mode) ImGui::BeginDisabled();
        if (ImGui::Checkbox("2D map", &c.two_d)) order_rehome_config(model, idx);
        if (perf_mode) {
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::TextDisabled("(the measurement is 1D: the X axis carries the error)");
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Off - a p(X axis) curve. On - a p(X, Y) map.");
        draw_order_axis_block("axx", "Axis X", s, c.axis_x_target,
                              c.axis_x_lo_text, c.axis_x_hi_text,
                              c.axis_x_log, &c.axis_x_n_text, true,
                              c.two_d ? "resolution (both axes)" : "points");
        draw_order_axis_block("axy", "Axis Y", s, c.axis_y_target,
                              c.axis_y_lo_text, c.axis_y_hi_text,
                              c.axis_y_log, nullptr, c.two_d);

        if (c.axis_x_target == kOrderTargetH && c.two_d && c.axis_y_target == kOrderTargetH)
            ImGui::TextColored(readable_text(ImVec4(1.0f, 0.45f, 0.45f, 1.0f)), "both axes sweep h");
        else if (c.two_d && c.axis_x_target == c.axis_y_target)
            ImGui::TextColored(readable_text(ImVec4(1.0f, 0.45f, 0.45f, 1.0f)), "both axes sweep the same thing");
        if (perf_ad) ImGui::EndDisabled();
    }

    // Что из этих двух блоков расчёт задаёт сам. У области устойчивости это
    // ВСЕ начальные условия (шаг делается от базисных векторов) и четыре
    // параметра, несущих элементы тестовой матрицы: они пересчитываются из
    // (sigma, omega, k, r) в каждом узле сетки. Остальные параметры — скажем,
    // коэффициенты обёртки или s_i у комплексных схем — работают как обычно.
    std::function<bool(const std::string&)> ic_overwritten, par_overwritten;
    const char* ic_note  = nullptr;
    const char* par_note =
        "Declare METHOD parameters (for composition schemes) right here as ordinary "
        "system parameters: they need not be used in the right-hand sides, "
        "and the body of a custom KRS reads them as a[k]. In the axis selector they "
        "will appear alongside the rest.";
    if (c.calc_kind == kOrderCalcStab) {
        ic_overwritten = [](const std::string&) { return true; };
        ic_note = "Not used: the step is taken from the basis vectors (1, 0) and (0, 1) - "
                  "that is what makes the result a fundamental matrix.";
        par_note = "The four entries of the test matrix are greyed out: they are rebuilt from "
                   "(sigma, omega) and k, r in every node of the grid, exactly as the "
                   "reference script does it. The rest of the parameters work as usual.";
        std::vector<std::string> taken;
        const int slots[4] = { c.stab_idx_a, c.stab_idx_b, c.stab_idx_c, c.stab_idx_d };
        for (int i = 0; i < 4; ++i) {
            const int pi = slots[i] - 1;
            if (pi >= 0 && pi < (int)s.params.size()) taken.push_back(s.params[(size_t)pi]);
        }
        par_overwritten = [taken](const std::string& n) {
            return std::find(taken.begin(), taken.end(), n) != taken.end();
        };
    }

    draw_named_num_fields("Initial conditions", s.vars, c.initial_conditions,
                          ic_note, &model, BroadcastField::InitCondition, ic_overwritten);
    draw_named_num_fields("Parameters", s.params, c.param_values,
                          par_note, &model, BroadcastField::Param, par_overwritten);

    if (!c.last_error.empty())
        draw_error_box("##order_err", c.last_error);

    // ---- Окна графиков ----
    // Ровно та же схема, что у параметрики и DFT1D: окно знает свой вид и
    // список вкладок, чьи кривые в нём лежат.
    ImGui::Separator();
    ucuda::ui_detail::block_separator("Plot windows");
    int win_to_remove = -1;
    for (int wi = 0; wi < (int)model.order_plot_windows.size(); ++wi) {
        OrderPlotWindow& win = model.order_plot_windows[wi];
        ImGui::PushID(win.id);
        item_w(150);
        if (InputTextStr("##wlabel", win.label)) {
            win.label_is_manual = !win.label.empty();   // пусто -> обратно к автоподписи
            model.order_plot_windows_dirty = true;
        }
        ImGui::SameLine();
        item_w(140);
        int kind = (int)win.kind;
        const char* kind_names[] = { "Order p", "Error", "Order map 2D", "Performance", "Error(t)",
                                     "Stability bounds" };
        if (ImGui::Combo("##wkind", &kind, kind_names, 6)) {
            win.kind = (OrderPlotWindow::Kind)kind;
            // Члены, которые новому виду не подходят, из окна выпадают: иначе
            // окно молча показывало бы "нет данных" от чужой вкладки.
            auto& mem = win.members;
            mem.erase(std::remove_if(mem.begin(), mem.end(), [&](int idx) {
                          return idx < 0 || idx >= (int)s.configs.size()
                              || !order_member_fits(s.configs[(size_t)idx], win.kind);
                      }), mem.end());
            if (win.kind == OrderPlotWindow::Kind::Map && mem.size() > 1) mem.resize(1);
            if (win.kind == OrderPlotWindow::Kind::P) win.y_log = false;
            win.plot_sig = -1;
            model.order_plot_windows_dirty = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Members...")) ImGui::OpenPopup("edit_order_window_members");
        if (ImGui::BeginPopup("edit_order_window_members")) {
            bool any = false;
            for (int ci = 0; ci < (int)s.configs.size(); ++ci) {
                if (!order_member_fits(s.configs[(size_t)ci], win.kind)) continue;
                any = true;
                const std::string lbl = s.configs[(size_t)ci].label + "##ordmem" + std::to_string(ci);
                const bool sel = std::find(win.members.begin(), win.members.end(), ci) != win.members.end();
                if (win.kind == OrderPlotWindow::Kind::Map) {
                    // Хитмапу поверх хитмапы не положишь — здесь выбор одиночный.
                    if (ImGui::RadioButton(lbl.c_str(), sel)) {
                        win.members.assign(1, ci);
                        model.order_plot_windows_dirty = true;
                    }
                } else {
                    bool v = sel;
                    if (ImGui::Checkbox(lbl.c_str(), &v)) {
                        if (v) win.members.push_back(ci);
                        else   win.members.erase(std::remove(win.members.begin(), win.members.end(), ci),
                                                 win.members.end());
                        win.plot_sig = -1;
                        model.order_plot_windows_dirty = true;
                    }
                }
            }
            if (!any) ImGui::TextDisabled("(no matching tabs)");
            ImGui::EndPopup();
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("X")) win_to_remove = wi;
        ImGui::PopID();
    }
    if (win_to_remove >= 0) model.remove_order_plot_window(win_to_remove);

    if (ImGui::Button("Add window"))
        model.add_order_plot_window(order_kind_for(c), { s.active_config_index });
    ImGui::SameLine();
    if (c.calc_kind == kOrderCalcStab) {
        if (ImGui::Button("Compare boundaries")) {
            std::vector<int> stabs;
            for (int ci = 0; ci < (int)s.configs.size(); ++ci)
                if (s.configs[(size_t)ci].calc_kind == kOrderCalcStab) stabs.push_back(ci);
            model.add_order_plot_window(OrderPlotWindow::Kind::Bounds, stabs);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("A window with the stability and preference boundaries of ALL\n"
                              "stability tabs as lines on one diagram - one colour and line\n"
                              "style per tab. Pick the tabs later via \"Members...\".");
        ImGui::SameLine();
    }
    if (ImGui::Button("Reset windows layout")) model.order_layout_generation++;

    // ---- Auto-compute ----
    // Сигнатура настроек — JSON сессии (результатов в нём нет). Переключение вкладки,
    // загрузка сессии и само включение флажка лишь запоминают её; пересчёт — когда она
    // поменялась при той же активной вкладке, и не во время расчёта (тогда — после него).
    {
        struct AutoState { int epoch = -1, active = -1; std::string sig; bool pending = false; };
        auto& st = ui_resource<AutoState, 3028>();
        const int epoch = model.session_epoch[(int)AppModel::AppMode::Order];
        const std::string sig = session_to_json_order(s);
        if (st.epoch != epoch || st.active != s.active_config_index || !s.auto_compute) {
            st.epoch = epoch; st.active = s.active_config_index; st.sig = sig; st.pending = false;
        } else if (sig != st.sig) {
            st.sig = sig;
            st.pending = true;
        }
        if (st.pending && !s.in_flight && s.auto_compute) {
            st.pending = false;
            model.run_order(s.active_config_index);
        }
    }
}
}
