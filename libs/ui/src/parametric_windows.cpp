#include "gui_common.h"
#include "gui_inputs.h"
#include "gui_plot.h"
#include "tab_parametric.h"
using namespace ucuda::ui_detail;


namespace ucuda::ui_detail {
[[nodiscard]] std::vector<ParamPlotMatchItem>
parametric_matching_items(AppModel& model, ParametricPlotWindow::Kind kind, bool mode_2d) {
    std::vector<ParamPlotMatchItem> out;
    if (kind == ParametricPlotWindow::Kind::Bifurcation) {
        auto& ds = model.bifurcation_session.diagrams;
        for (int i = 0; i < (int)ds.size(); ++i)
            if (ds[i].mode_2d == mode_2d) out.push_back({ i, ds[i].label });
    } else if (kind == ParametricPlotWindow::Kind::LLE) {
        auto& cs = model.lle_session.curves;
        for (int i = 0; i < (int)cs.size(); ++i)
            if (cs[i].mode_2d == mode_2d) out.push_back({ i, cs[i].label });
    } else if (kind == ParametricPlotWindow::Kind::LS) {
        auto& cs = model.ls_session.curves;
        for (int i = 0; i < (int)cs.size(); ++i)
            if (cs[i].mode_2d == mode_2d) out.push_back({ i, cs[i].label });
    } else {
        auto& cs = model.metrics_session.configs;
        for (int i = 0; i < (int)cs.size(); ++i)
            if (cs[i].mode_2d == mode_2d) out.push_back({ i, cs[i].label });
    }
    return out;
}
}

namespace ucuda::ui_detail {
int parametric_type_index_of(ParametricPlotWindow::Kind kind, bool mode_2d) {
    int base = kind == ParametricPlotWindow::Kind::Bifurcation ? 0
             : kind == ParametricPlotWindow::Kind::LLE ? 2
             : kind == ParametricPlotWindow::Kind::LS  ? 4 : 6;
    return base + (mode_2d ? 1 : 0);
}
}

namespace ucuda::ui_detail {
void parametric_type_from_index(int t, ParametricPlotWindow::Kind& kind, bool& mode_2d) {
    kind    = (t < 2) ? ParametricPlotWindow::Kind::Bifurcation
            : (t < 4) ? ParametricPlotWindow::Kind::LLE
            : (t < 6) ? ParametricPlotWindow::Kind::LS : ParametricPlotWindow::Kind::Metrics;
    mode_2d = (t % 2) == 1;
}
}

namespace ucuda::ui_detail {
bool draw_parametric_window_setup_row(AppModel& model, ParametricPlotWindow& win) {
    bool close_clicked = false;

    ImGui::SetNextItemWidth(220);
    if (InputTextStr("##wlabel", win.label)) {
        win.label_is_manual = !win.label.empty();   // empty → back to auto
        model.parametric_plot_windows_dirty = true;
    }
    ImGui::SameLine();

    // Type combo: changing kind/dimension invalidates old member indices
    // (they're only meaningful within the previous session+dimension),
    // so switching type clears members.
    int t = parametric_type_index_of(win.kind, win.mode_2d);
    ImGui::SetNextItemWidth(130);
    if (ImGui::Combo("##wtype", &t, kParametricTypeNames, IM_ARRAYSIZE(kParametricTypeNames))) {
        ParametricPlotWindow::Kind new_kind; bool new_2d;
        parametric_type_from_index(t, new_kind, new_2d);
        if (new_kind != win.kind || new_2d != win.mode_2d) {
            win.kind = new_kind;
            win.mode_2d = new_2d;
            win.colored_1d = false;   // combo no longer selects Colored 1D — reset; toggled above the plot
            win.members.clear();
            model.parametric_plot_windows_dirty = true;
        }
    }
    ImGui::SameLine();

    if (ImGui::Button("Members...")) ImGui::OpenPopup("edit_plot_window_members");
    if (ImGui::BeginPopup("edit_plot_window_members")) {
        draw_parametric_members_list(model, win);
        ImGui::EndPopup();
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("X")) close_clicked = true;

    return close_clicked;
}
}

namespace ucuda::ui_detail {
void draw_parametric_members_list(AppModel& model, ParametricPlotWindow& win) {
    auto items = parametric_matching_items(model, win.kind, win.mode_2d);
    if (items.empty()) {
        ImGui::TextDisabled("(none available)");
    } else if (win.mode_2d || win.colored_1d) {
        // 2D / Colored 1D show heatmaps — single-select (radio).
        for (const auto& item : items) {
            bool sel = !win.members.empty() && win.members[0] == item.index;
            std::string lbl = item.label + "##mem" + std::to_string(item.index);
            if (ImGui::RadioButton(lbl.c_str(), sel)) {
                win.members.assign(1, item.index);
                model.parametric_plot_windows_dirty = true;
            }
        }
    } else {
        for (const auto& item : items) {
            bool has = std::find(win.members.begin(), win.members.end(), item.index) != win.members.end();
            std::string lbl = item.label + "##mem" + std::to_string(item.index);
            if (ImGui::Checkbox(lbl.c_str(), &has)) {
                if (has) win.members.push_back(item.index);
                else win.members.erase(std::remove(win.members.begin(), win.members.end(), item.index), win.members.end());
                model.parametric_plot_windows_dirty = true;
            }
        }
    }
}
}

namespace ucuda::ui_detail {
// Те же подпись / тип / члены, что в строке draw_parametric_window_setup_row, но столбиком —
// для меню окна графика (вверху меню тулбара карты или под своей стрелкой). Закрывается окно
// крестиком в заголовке, поэтому X здесь нет.
static void draw_parametric_window_menu(AppModel& model, ParametricPlotWindow& win) {
    ImGui::SeparatorText("Window");
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 12.0f);
    if (InputTextStr("Label##wlabel_m", win.label)) {
        win.label_is_manual = !win.label.empty();   // empty → back to auto
        model.parametric_plot_windows_dirty = true;
    }
    int t = parametric_type_index_of(win.kind, win.mode_2d);
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 12.0f);
    if (ImGui::Combo("Type##wtype_m", &t, kParametricTypeNames, IM_ARRAYSIZE(kParametricTypeNames))) {
        ParametricPlotWindow::Kind new_kind; bool new_2d;
        parametric_type_from_index(t, new_kind, new_2d);
        if (new_kind != win.kind || new_2d != win.mode_2d) {   // как в строке настроек
            win.kind = new_kind;
            win.mode_2d = new_2d;
            win.colored_1d = false;
            win.members.clear();
            model.parametric_plot_windows_dirty = true;
        }
    }
    ImGui::SeparatorText("Members");
    draw_parametric_members_list(model, win);
}
}

namespace ucuda::ui_detail {
void draw_parametric_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) {
    auto& renderers = ui_resource<std::map<int, std::unique_ptr<PlotRenderer>>, 3019>();
    auto& views = ui_resource<std::map<int, std::unique_ptr<Plot2DView>>, 3020>();
    auto& heatmaps = ui_resource<std::map<int, std::map<int, std::unique_ptr<HeatmapView>>>, 3021>();
    // Tracks the kind the Plot2DView was last configured for; when a
    // window's kind changes (Type combo), re-run configure_plot_view so
    // e.g. LLE doesn't inherit Bifurcation's points_mode=true.
    auto& view_configured_for = ui_resource<std::map<int, ParametricPlotWindow::Kind>, 3022>();
    static int seen_epoch = 0;
    if (session_reloaded(model, AppModel::AppMode::Parametric, seen_epoch)) {
        invalidate_views(views);
        for (auto& kv : heatmaps) invalidate_views(kv.second);
    }

    int to_remove = -1;
    for (int i = 0; i < (int)model.parametric_plot_windows.size(); ++i) {
        ParametricPlotWindow& win = model.parametric_plot_windows[i];

        auto& renderer = renderers[win.id];
        if (!renderer) renderer = std::make_unique<PlotRenderer>();
        auto& view = views[win.id];
        bool fresh_view = !view;
        if (fresh_view) view = std::make_unique<Plot2DView>();
        auto cf_it = view_configured_for.find(win.id);
        if (fresh_view || cf_it == view_configured_for.end() || cf_it->second != win.kind) {
            configure_plot_view(*view, win.kind);
            view_configured_for[win.id] = win.kind;
        }
        auto& hm_map = heatmaps[win.id];

        // "###" makes ImGui hash the ID from the suffix only (win.id + layout generation),
        // independent of the visible label: docking/position stay stable when the user renames
        // the window, and a layout reset changes the ID to re-cascade default positions.
        std::string title = win.label + "###pwin" + std::to_string(win.id)
                           + "_g" + std::to_string(model.parametric_layout_generation);
        bool open = true;
        float ox = 60.0f + (float)(i % 5) * 35.0f, oy = 80.0f + (float)(i % 5) * 35.0f;
        ImGui::SetNextWindowPos(ImVec2(ox, oy), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(700, 550), ImGuiCond_FirstUseEver);
        if (ImGui::Begin(title.c_str(), &open)) {
            ImGui::PushID(win.id);
            // Подпись / тип / члены окна — меню, а не строка над графиком: у карты (2D,
            // Colored 1D) это верх меню её тулбара (draw_heatmap_toolbar забирает шапку), у
            // графиков без тулбара — своя стрелка. Есть ли тулбар, узнаём по прошлому кадру:
            // шапка отдаётся всегда, и если тулбар её не взял, со следующего кадра рисуется
            // своя стрелка.
            auto& head_in_toolbar = ui_resource<std::map<int, bool>, 3023>();
            auto hit = head_in_toolbar.find(win.id);
            const bool expect_toolbar = hit != head_in_toolbar.end() ? hit->second
                                                                     : (win.mode_2d || win.colored_1d);
            auto head = [&model, &win]() { draw_parametric_window_menu(model, win); };
            if (!expect_toolbar && !plot_toolbar_hidden()) {
                if (ImGui::ArrowButton("##pwin_menu_open", ImGuiDir_Down))
                    ImGui::OpenPopup("##pwin_menu");
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Label, type and members of this window");
                constrain_popup_height_below_item();
                if (ImGui::BeginPopup("##pwin_menu")) {
                    head();
                    ImGui::EndPopup();
                }
            }
            set_heatmap_toolbar_head(head);
            switch (win.kind) {
            case ParametricPlotWindow::Kind::Bifurcation:
                draw_bifurcation_plot(model, lib, cb, win, *renderer, *view, hm_map);
                break;
            case ParametricPlotWindow::Kind::LLE:
                draw_lle_plot(model, lib, cb, win, *renderer, *view, hm_map);
                break;
            case ParametricPlotWindow::Kind::LS:
                draw_ls_plot(model, lib, cb, win, *renderer, *view, hm_map);
                break;
            case ParametricPlotWindow::Kind::Metrics:
                draw_metrics_plot(model, lib, cb, win, *renderer, *view, hm_map);
                break;
            }
            // Скрытый тулбар шапку не берёт — это не повод переводить окно на свою стрелку.
            const bool head_used = take_heatmap_toolbar_head_used();
            if (!plot_toolbar_hidden()) head_in_toolbar[win.id] = head_used;
            ImGui::PopID();
        }
        ImGui::End();
        if (!open) to_remove = i;
    }
    if (to_remove >= 0) {
        int id = model.parametric_plot_windows[to_remove].id;
        renderers.erase(id);
        views.erase(id);
        heatmaps.erase(id);
        view_configured_for.erase(id);
        model.remove_parametric_plot_window(to_remove);
    }
}
}


namespace ucuda::ui_detail {
void draw_parametric_controls(AppModel& model, SystemLibrary& lib) {
    ImGui::Text("Parametric analysis");
    ImGui::TextDisabled("Per-thread parameter sweep via NVRTC + NonLinAnal kernels.");

    // Union used by the Run / Run all... buttons below to disable themselves
    // while any sub-analysis (Bif/LLE/LS) is computing. Kept after the tab's
    // System combo was moved to the top-bar; the top-bar combo has its own
    // wider union across all 7 sessions.
    bool any_in_flight = model.bifurcation_session.in_flight
                      || model.lle_session.in_flight
                      || model.ls_session.in_flight
                      || model.metrics_session.in_flight;

    // Run (active sub-tab, active config) + batch Run all. Кнопка Run единая для Bif/LLE/LS,
    // диспатчится по parametric_active_analysis (0=Bif, 1=LLE, 2=LS) к соответствующему
    // active_*_index. Run all — глобальный по всем трём спискам: попап показывает чекбоксы
    // для каждой диаграммы / кривой / спектра, Run кладёт отмеченные в
    // model.parametric_queue и стартует первый; draw_gui тикает очередь после poll'ов.
    {
        const int kind = model.parametric_active_analysis;
        int  active_idx = -1;
        bool no_active  = false;
        if (kind == 0) {
            active_idx = model.bifurcation_session.active_diagram_index;
            no_active  = model.bifurcation_session.diagrams.empty();
        } else if (kind == 1) {
            active_idx = model.lle_session.active_curve_index;
            no_active  = model.lle_session.curves.empty();
        } else if (kind == 2) {
            active_idx = model.ls_session.active_curve_index;
            no_active  = model.ls_session.curves.empty();
        } else if (kind == 3) {
            active_idx = model.metrics_session.active_config_index;
            no_active  = model.metrics_session.configs.empty();
        }

        std::vector<RunAllGroup> groups;
        groups.push_back(RunAllGroup{
            "Bifurcation", "pbd_", (int)model.bifurcation_session.diagrams.size(),
            [&model](int i) { return model.bifurcation_session.diagrams[i].label; },
            [&model](int i) {
                model.enqueue_parametric({ParametricQueueItem::Kind::Bifurcation, i});
            } });
        groups.push_back(RunAllGroup{
            "LLE", "plle_", (int)model.lle_session.curves.size(),
            [&model](int i) { return model.lle_session.curves[i].label; },
            [&model](int i) {
                model.enqueue_parametric({ParametricQueueItem::Kind::LLE, i});
            } });
        groups.push_back(RunAllGroup{
            "LS", "pls_", (int)model.ls_session.curves.size(),
            [&model](int i) { return model.ls_session.curves[i].label; },
            [&model](int i) {
                model.enqueue_parametric({ParametricQueueItem::Kind::LS, i});
            } });
        groups.push_back(RunAllGroup{
            "Metrics", "pmet_", (int)model.metrics_session.configs.size(),
            [&model](int i) { return model.metrics_session.configs[i].label; },
            [&model](int i) {
                model.enqueue_parametric({ParametricQueueItem::Kind::Metrics, i});
            } });

        // Run all блокируется ТОЛЬКО на время расчёта: списков три, и пустой
        // активный не повод запрещать запуск остальных (в отличие от вкладок
        // с одним списком, где block_run_all включает и «нет конфигов»).
        draw_run_and_run_all("##run_all_parametric", any_in_flight, no_active,
                             any_in_flight, 140.0f,
                             [&model, kind, active_idx]() {
                                 if (active_idx < 0) return;
                                 if      (kind == 0) model.run_bifurcation(active_idx);
                                 else if (kind == 1) model.run_lle(active_idx);
                                 else if (kind == 2) model.run_ls(active_idx);
                                 else if (kind == 3) model.run_metrics(active_idx);
                             },
                             {}, groups,
                             [&model]() { model.start_next_in_parametric_queue(); },
                             model.parametric_queue().size());
    }
    ImGui::Separator();

    if (ImGui::BeginTabBar("##parm_top")) {
        if (ImGui::BeginTabItem("Bifurcation")) {
            model.parametric_active_analysis = 0;
            draw_bifurcation_controls(model, lib);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("LLE")) {
            model.parametric_active_analysis = 1;
            draw_lle_controls(model, lib);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("LS")) {
            model.parametric_active_analysis = 2;
            draw_ls_controls(model, lib);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Metrics")) {
            model.parametric_active_analysis = 3;
            draw_metrics_controls(model, lib);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    // Plot windows: dynamic list, mirrors Phase's Projections section. A sibling of the
    // Bif/LLE/LS tabs above (not nested in any of them) — always visible, since a plot window
    // can be any (kind, dimension) combo. Each row has its own Type combo so the chart type
    // can be changed after creation instead of being fixed at "Add".
    ImGui::Separator();
    ImGui::SeparatorText("Plot windows");

    int win_to_remove = -1;
    for (int i = 0; i < (int)model.parametric_plot_windows.size(); ++i) {
        ParametricPlotWindow& win = model.parametric_plot_windows[i];
        ImGui::PushID(win.id);
        if (draw_parametric_window_setup_row(model, win)) win_to_remove = i;
        ImGui::PopID();
    }
    if (win_to_remove >= 0) model.remove_parametric_plot_window(win_to_remove);

    if (ImGui::Button("Add window")) {
        model.add_parametric_plot_window(ParametricPlotWindow::Kind::Bifurcation, false, {});
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset windows layout")) { model.parametric_layout_generation++; }
}
}
