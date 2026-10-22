#include "gui_internal.h"

namespace ucuda::ui_detail {


void draw_custom_controls_panel(AppModel& model, SystemLibrary& lib,
                                       ImVec2 pos, ImVec2 size) {
    ImGui::SetNextWindowPos(pos);
    ImGui::SetNextWindowSize(size);
    // No NoBringToFrontOnFocus / NoNavFocus here — clicks on Controls/Workspace
    // widgets were being swallowed with those on; keeping the panels pinned via
    // NoMove/NoResize + SetNextWindowPos/Size is enough to make them "regions".
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar
                           | ImGuiWindowFlags_NoResize
                           | ImGuiWindowFlags_NoMove
                           | ImGuiWindowFlags_NoCollapse
                           | ImGuiWindowFlags_NoDocking;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    if (ImGui::Begin("Custom Controls##panel", nullptr, flags)) {
        draw_custom_controls(model, lib);
    }
    ImGui::End();
    ImGui::PopStyleVar();
}


void draw_custom_workspace_panel(CustomSession& cs,
                                        const std::string& sys,
                                        ImVec2 pos, ImVec2 size) {
    auto& ws = cs.workspace;
    ws.ensure_default();

    ImGui::SetNextWindowPos(pos);
    ImGui::SetNextWindowSize(size);
    // No NoBringToFrontOnFocus / NoNavFocus here — clicks on Controls/Workspace
    // widgets were being swallowed with those on; keeping the panels pinned via
    // NoMove/NoResize + SetNextWindowPos/Size is enough to make them "regions".
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar
                           | ImGuiWindowFlags_NoResize
                           | ImGuiWindowFlags_NoMove
                           | ImGuiWindowFlags_NoCollapse
                           | ImGuiWindowFlags_NoDocking;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    if (!ImGui::Begin("Custom Workspace##panel", nullptr, flags)) {
        ImGui::End();
        ImGui::PopStyleVar();
        return;
    }

    // Tab bar with +, close, rename, and cross-tab drop targets.
    // NOTE ON DOCKING: the real DockSpace for the active tab is submitted AFTER EndTabBar, not
    // inside the BeginTabItem body. Submitting DockSpace(id) inside BeginTabItem pushes an
    // ID-stack entry, so ImGui creates an auto-child host "…/DockSpace_XXX" whose host context
    // differs from the KeepAliveOnly submit outside — that mismatch orphaned docked windows on
    // tab switch (plots vanishing when returning to a tab). Real + keep-alive from the same
    // context (workspace panel window) keeps the host consistent.
    int close_id = 0;
    int switch_to_id = 0;
    ImGuiTabBarFlags tb_flags = ImGuiTabBarFlags_Reorderable
                              | ImGuiTabBarFlags_TabListPopupButton
                              | ImGuiTabBarFlags_AutoSelectNewTabs
                              | ImGuiTabBarFlags_FittingPolicyScroll;
    if (ImGui::BeginTabBar("##custom_ws_tabs", tb_flags)) {
        const bool allow_close = (int)ws.tabs.size() > 1;
        for (size_t i = 0; i < ws.tabs.size(); ++i) {
            WorkspaceTab& tab = ws.tabs[i];
            // Stable tab ID via "##ws_tab_<id>" suffix so rename changes
            // only the visible label, not the internal ImGui key.
            std::string label = tab.name + "##ws_tab_" + std::to_string(tab.id);
            bool keep = true;
            bool item_open = ImGui::BeginTabItem(label.c_str(),
                                                 allow_close ? &keep : nullptr,
                                                 ImGuiTabItemFlags_None);
            // Drop-target for cross-tab window move. Accepts our own payload (WS_DRAG_PAYLOAD,
            // `const char*` name) for a future explicit drag handle, and ImGui's native window
            // docking payload (IMGUI_PAYLOAD_TYPE_WINDOW, `ImGuiWindow*`), which fires when the
            // user drags a docked plot's tab out of its dockspace onto ours — that is what makes
            // native drag&drop between tabs work.
            if (ImGui::BeginDragDropTarget()) {
                if (const ImGuiPayload* p = ImGui::AcceptDragDropPayload(WS_DRAG_PAYLOAD)) {
                    const char* win_name = (const char*)p->Data;
                    if (win_name && *win_name) {
                        ImGui::DockBuilderDockWindow(win_name, ws_dock_id(tab.id, sys));
                        switch_to_id = tab.id;
                        ws.dirty = true;
                    }
                }
                if (const ImGuiPayload* p = ImGui::AcceptDragDropPayload(IMGUI_PAYLOAD_TYPE_WINDOW)) {
                    ImGuiWindow* dragged = *(ImGuiWindow**)p->Data;
                    if (dragged && dragged->Name) {
                        ImGui::DockBuilderDockWindow(dragged->Name, ws_dock_id(tab.id, sys));
                        switch_to_id = tab.id;
                        ws.dirty = true;
                    }
                }
                ImGui::EndDragDropTarget();
            }
            // Right-click menu (rename / close).
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Rename")) {
                    g_ws_rename.target_id = tab.id;
                    std::snprintf(g_ws_rename.buf, sizeof(g_ws_rename.buf),
                                  "%s", tab.name.c_str());
                }
                if (allow_close && ImGui::MenuItem("Close tab")) {
                    close_id = tab.id;
                }
                ImGui::EndPopup();
            }
            if (item_open) {
                if (ws.active_tab_id != tab.id) {
                    ws.active_tab_id = tab.id;
                    ws.dirty = true;
                }
                // Empty body — real DockSpace is submitted AFTER EndTabBar
                // (see NOTE ON DOCKING above).
                ImGui::EndTabItem();
            }
            if (!keep) close_id = tab.id;
        }
        if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing
                                    | ImGuiTabItemFlags_NoTooltip)) {
            WorkspaceTab nt;
            // id keeps monotonically incrementing — it seeds the DockSpace node id, and reusing
            // one could collide with imgui.ini state of a previously-closed tab. The display name
            // instead picks the lowest unused "Tab N" so the counter doesn't grow to 20+.
            nt.id = ws.next_tab_id++;
            int label_num = 1;
            auto name_taken = [&](int n){
                std::string s = "Tab " + std::to_string(n);
                return std::any_of(ws.tabs.begin(), ws.tabs.end(),
                                   [&](const WorkspaceTab& t){ return t.name == s; });
            };
            while (name_taken(label_num)) ++label_num;
            nt.name = "Tab " + std::to_string(label_num);
            ws.tabs.push_back(nt);
            ws.active_tab_id = nt.id;
            ws.dirty = true;
        }
        // Sync ws.tabs order with ImGui's internal reordered order: the user can drag tabs
        // (Reorderable), but ImGui only shuffles its own Tabs array. Without this our vector keeps
        // the original order, so plots docked into "the third tab visually" land in whichever tab
        // id is third in ws.tabs on restart — the sync fixes both the forgotten tab order and the
        // wrong-tab plots.
        if (ImGuiTabBar* tb = ImGui::GetCurrentTabBar()) {
            if (tb->Tabs.Size >= 2 && (int)ws.tabs.size() == tb->Tabs.Size) {
                std::vector<WorkspaceTab> reordered;
                reordered.reserve(ws.tabs.size());
                bool all_matched = true;
                for (int i = 0; i < tb->Tabs.Size; ++i) {
                    ImGuiID tid = tb->Tabs[i].ID;
                    auto it = std::find_if(ws.tabs.begin(), ws.tabs.end(),
                        [&](const WorkspaceTab& t){
                            std::string label = t.name + "##ws_tab_" + std::to_string(t.id);
                            return ImHashStr(label.c_str()) == tid;
                        });
                    if (it == ws.tabs.end()) { all_matched = false; break; }
                    reordered.push_back(*it);
                }
                if (all_matched) {
                    // Only mark dirty when actual order changed, to avoid a
                    // needless _last_custom.json rewrite every frame.
                    bool changed = false;
                    for (size_t i = 0; i < ws.tabs.size(); ++i) {
                        if (ws.tabs[i].id != reordered[i].id) { changed = true; break; }
                    }
                    if (changed) {
                        ws.tabs = std::move(reordered);
                        ws.dirty = true;
                    }
                }
            }
        }
        ImGui::EndTabBar();
    }

    // Submit all workspace dockspaces from the SAME host context (workspace panel, outside
    // BeginTabItem). The active tab gets a real submit that renders and hosts its docked windows;
    // inactive tabs get KeepAliveOnly so their windows stay docked (invisible until switched to)
    // instead of orphaning to floating state.
    for (const auto& tab : ws.tabs) {
        if (tab.id != ws.active_tab_id) {
            ImGui::DockSpace(ws_dock_id(tab.id, sys), ImVec2(0, 0),
                             ImGuiDockNodeFlags_KeepAliveOnly);
        }
    }
    if (std::any_of(ws.tabs.begin(), ws.tabs.end(),
                    [&](const WorkspaceTab& t){ return t.id == ws.active_tab_id; })) {
        ImGui::DockSpace(ws_dock_id(ws.active_tab_id, sys), ImVec2(0, 0),
                         ImGuiDockNodeFlags_None);
    }

    // Deferred tab close — move every window still docked in the closing
    // tab's dockspace into the fallback tab (front). DockBuilderRemoveNode
    // moves them off first, so we manually re-dock instead.
    if (close_id != 0) {
        auto it = std::find_if(ws.tabs.begin(), ws.tabs.end(),
                               [close_id](const WorkspaceTab& t){ return t.id == close_id; });
        if (it != ws.tabs.end() && ws.tabs.size() > 1) {
            ws.tabs.erase(it);
            ws.dirty = true;
            int fallback = ws.tabs.front().id;
            // Find all windows docked in the closing node and re-dock them
            // into the fallback. Iterate a copy of the node's windows list
            // because DockBuilderDockWindow mutates it under us.
            ImGuiDockNode* node = ImGui::DockBuilderGetNode(ws_dock_id(close_id, sys));
            if (node) {
                // Walk the whole subtree (splits) and re-dock every window
                // encountered. Simple recursive lambda.
                std::vector<std::string> to_move;
                std::function<void(ImGuiDockNode*)> walk = [&](ImGuiDockNode* n){
                    if (!n) return;
                    for (int wi = 0; wi < n->Windows.Size; ++wi) {
                        if (n->Windows[wi] && n->Windows[wi]->Name)
                            to_move.emplace_back(n->Windows[wi]->Name);
                    }
                    walk(n->ChildNodes[0]);
                    walk(n->ChildNodes[1]);
                };
                walk(node);
                for (const auto& name : to_move)
                    ImGui::DockBuilderDockWindow(name.c_str(), ws_dock_id(fallback, sys));
                // Now discard the closed node's split hierarchy.
                ImGui::DockBuilderRemoveNode(ws_dock_id(close_id, sys));
            }
            if (ws.active_tab_id == close_id) ws.active_tab_id = fallback;
        }
    }
    if (switch_to_id != 0) ws.active_tab_id = switch_to_id;

    ImGui::End();
    ImGui::PopStyleVar();

    // Rename modal — centered on viewport, Esc = Cancel, auto-reset if the user clicks outside.
    // IMPORTANT: SetNextWindowPos must be INSIDE the "opening" branch — unconditional, it left
    // the next-window-pos flag dangling and got applied to the first plot window created that
    // frame, which appeared floating at viewport center like a dark overlay.
    if (g_ws_rename.target_id != 0) {
        ImGui::OpenPopup("Rename tab##custom_ws");
        ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    }
    if (ImGui::BeginPopupModal("Rename tab##custom_ws", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::SetKeyboardFocusHere();
        bool commit = ImGui::InputText("##ws_rename",
                                       g_ws_rename.buf, sizeof(g_ws_rename.buf),
                                       ImGuiInputTextFlags_EnterReturnsTrue);
        if (ImGui::Button("OK") || commit) {
            for (auto& t : ws.tabs) {
                if (t.id == g_ws_rename.target_id) {
                    std::string s = g_ws_rename.buf;
                    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();
                    if (!s.empty() && s != t.name) {
                        t.name = s;
                        ws.dirty = true;
                    }
                    break;
                }
            }
            g_ws_rename.target_id = 0;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            g_ws_rename.target_id = 0;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    } else if (g_ws_rename.target_id != 0) {
        // Popup was closed by ImGui itself (clicked outside etc.). Clear the
        // trigger flag so we don't immediately re-open.
        g_ws_rename.target_id = 0;
    }
}


void draw_custom_mode_layout(AppModel& model, SystemLibrary& lib,
                                    const GuiCallbacks& cb,
                                    ImVec2 area_pos, ImVec2 area_size) {
    auto& ws = model.custom_session.workspace;
    ws.ensure_default();

    if (area_size.x < 100.0f || area_size.y < 40.0f) return;

    // Clamp splitter to sane bounds for the current viewport.
    const float min_ctrl   = 200.0f;
    const float min_ws     = 300.0f;
    const float splitter_w = 6.0f;
    float max_ctrl = std::max(min_ctrl, area_size.x - min_ws - splitter_w);
    if (ws.controls_width < min_ctrl) ws.controls_width = min_ctrl;
    if (ws.controls_width > max_ctrl) ws.controls_width = max_ctrl;

    ImVec2 ctrl_pos  = area_pos;
    ImVec2 ctrl_size(ws.controls_width, area_size.y);
    ImVec2 split_pos(area_pos.x + ws.controls_width, area_pos.y);
    ImVec2 split_size(splitter_w, area_size.y);
    ImVec2 ws_pos  (split_pos.x + splitter_w, area_pos.y);
    ImVec2 ws_size (area_size.x - ws.controls_width - splitter_w, area_size.y);

    // Splitter (invisible button + drawn rect).
    ImGui::SetNextWindowPos(split_pos);
    ImGui::SetNextWindowSize(split_size);
    // NoBackground — splitter draws its own filled rect manually, so ImGui's
    // window background (which showed as the "dark strip" artefact on click)
    // is redundant. NoBringToFrontOnFocus swallows the drag input entirely,
    // so we can't use it here (unlike on Controls/Workspace panels).
    ImGuiWindowFlags split_flags = ImGuiWindowFlags_NoTitleBar
                                 | ImGuiWindowFlags_NoResize
                                 | ImGuiWindowFlags_NoMove
                                 | ImGuiWindowFlags_NoScrollbar
                                 | ImGuiWindowFlags_NoScrollWithMouse
                                 | ImGuiWindowFlags_NoCollapse
                                 | ImGuiWindowFlags_NoNavFocus
                                 | ImGuiWindowFlags_NoBackground
                                 | ImGuiWindowFlags_NoDocking;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::Begin("##CustomSplitter", nullptr, split_flags);
    ImGui::InvisibleButton("##split_btn", split_size);
    if (ImGui::IsItemActive()) {
        float dx = ImGui::GetIO().MouseDelta.x;
        if (dx != 0.0f) {
            ws.controls_width += dx;
            if (ws.controls_width < min_ctrl) ws.controls_width = min_ctrl;
            if (ws.controls_width > max_ctrl) ws.controls_width = max_ctrl;
            ws.dirty = true;
        }
    }
    ImU32 col = ImGui::IsItemActive()  ? ImGui::GetColorU32(ImGuiCol_SeparatorActive)
              : ImGui::IsItemHovered() ? ImGui::GetColorU32(ImGuiCol_SeparatorHovered)
                                       : ImGui::GetColorU32(ImGuiCol_Separator);
    ImGui::GetWindowDrawList()->AddRectFilled(split_pos,
        ImVec2(split_pos.x + split_size.x, split_pos.y + split_size.y), col);
    if (ImGui::IsItemHovered() || ImGui::IsItemActive())
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    ImGui::End();
    ImGui::PopStyleVar(3);

    draw_custom_controls_panel(model, lib, ctrl_pos, ctrl_size);
    draw_custom_workspace_panel(model.custom_session, model.loaded_name, ws_pos, ws_size);
    draw_custom_plot_windows(model, lib, cb);
}


void draw_custom_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) {
    (void)lib; (void)cb;
    auto& cs = model.custom_session;
    auto& q  = model.custom_queue;
    auto& ws = cs.workspace;
    // Per-system suffix appended to every plot window title. imgui.ini keys
    // dock state per window name, so without a per-system suffix the same
    // plot titles (e.g. "Custom Bif 2D") would share layout across systems.
    // Empty when no system is loaded — safe (title unchanged from before).
    const std::string& sys    = model.loaded_name;
    const std::string  suffix = custom_win_suffix(sys);
    // Per-system delta XOR'd into every plot's owner_id: HeatmapView / PlotRenderer cache their
    // GPU texture per owner_id, so without it both systems shared one texture and the FIRST
    // system's pixels stayed on screen while the SECOND system's data sat unrendered underneath.
    // Low bits of the system-name hash — enough entropy without colliding with the hand-picked
    // owner_id constants below.
    const int sys_owner_delta = sys.empty() ? 0
                              : (int)(ImHashStr(sys.c_str()) & 0x00FFFFFF);

    // Persistent per-window renderers + views. Indexed by fixed slots so their
    // ImGui-IDs stay stable across frames. Hoisted to the top of the function
    // (together with the L1D-side statics and the init-seen sentinels) so a
    // single system-change block below can reset the whole cache tier.
    auto& renderers_2d = ui_resource<std::array<std::unique_ptr<PlotRenderer>, 3>, 16>();
    auto& heatmaps = ui_resource<std::array<HeatmapView, 3>, 17>();
    static std::array<int, 3>                           hm_init_cmap_seen = { -2, -2, -2 };
    static int                                          ls_init_exp_seen  = -999;
    auto& l1_views = ui_resource<std::array<std::unique_ptr<Plot2DView>,  6>, 18>();
    auto& l1_renderers = ui_resource<std::array<std::unique_ptr<PlotRenderer>, 6>, 19>();
    static std::array<std::vector<std::vector<double>>, 6> l1_bufs;
    // Custom Basins (Level-3 kind=1) HeatmapView + renderer — hoisted so the
    // system-change block below can reset them alongside the other statics.
    auto& bsn_hv = ui_resource<HeatmapView, 20>();
    auto& bsn_renderer = ui_resource<PlotRenderer, 21>();
    for (auto& r : renderers_2d) if (!r) r = std::make_unique<PlotRenderer>();

    // System-change detection. Slot-indexed static caches (HeatmapView view_valid, autofit ranges,
    // seen colormap; Plot2DView view limits and cached VBO series_generation) leaked across systems
    // because slot 0 of Chen and Rossler shared one HeatmapView instance. On rename of the loaded
    // system, wipe the per-slot view / seen state so the new one starts with fresh autofit and
    // colormap seeding. renderers_* (PlotRenderer/GPU textures + FBOs) are reused: the owner_id XOR
    // with sys_owner_delta already makes PlotRenderer treat the slot as a new cache entry, and
    // destroying them here would leak GPU memory tied to the GL context.
    static std::string last_system_for_plots;
    if (last_system_for_plots != sys) {
        last_system_for_plots = sys;
        // HeatmapView / Plot2DView are non-copyable — reset only the fields that gate rendering
        // (view_valid / data_gen_cached / series_generation), which are the actual leak. Plot2DView
        // caches its VBO by (owner_id, series_generation), and the owner_id XOR alone wasn't enough:
        // the second system's data_generation=1 matched the first's cached series_generation=1, so
        // Plot2DView kept rendering the FIRST system's VBO. Colormap / autoscale / manual v-limits
        // stay, so user prefs aren't wiped mid-session.
        for (auto& hv : heatmaps) { hv.view_valid = false; hv.data_gen_cached = -1; }
        for (auto& v  : l1_views) if (v) { v->view_valid = false; v->series_generation = -1; }
        for (auto& b  : l1_bufs)  b.clear();
        bsn_hv.view_valid = false; bsn_hv.data_gen_cached = -1;
        hm_init_cmap_seen = { -2, -2, -2 };
        ls_init_exp_seen  = -999;
    }

    // Clamp fix_x/fix_y to the current effective sweep range every frame. custom_session.h
    // documents this ("clamped to current effective sweep ranges") but it was never implemented —
    // a stale fix outside the range fed the kernel via apply_shared_to_bif1d/pin_param and produced
    // degenerate 1D results (all-zero or empty on Run) until the user dragged the slider back in.
    {
        EffectiveSweep esx = effective_sweep_x(cs.shared);
        EffectiveSweep esy = effective_sweep_y(cs.shared);
        double x_lo = parse_ratio_or(esx.lo_text, 0.0), x_hi = parse_ratio_or(esx.hi_text, 1.0);
        double y_lo = parse_ratio_or(esy.lo_text, 0.0), y_hi = parse_ratio_or(esy.hi_text, 1.0);
        if (x_hi < x_lo) std::swap(x_lo, x_hi);
        if (y_hi < y_lo) std::swap(y_lo, y_hi);
        if (cs.shared.fix_x_value < x_lo) cs.shared.fix_x_value = x_lo;
        if (cs.shared.fix_x_value > x_hi) cs.shared.fix_x_value = x_hi;
        if (cs.shared.fix_y_value < y_lo) cs.shared.fix_y_value = y_lo;
        if (cs.shared.fix_y_value > y_hi) cs.shared.fix_y_value = y_hi;
    }

    // Dock a plot window into the active tab ONLY when there is no dock memory for it — neither at
    // runtime (`window->DockId`) NOR in imgui.ini (`ImGuiWindowSettings::DockId`). This lets a
    // reloaded session restore its per-tab layout from ini on the very first frame, when the window
    // hasn't been Begun yet (FindWindowByName returns null) but ImGui has the settings ready.
    // Overriding a persisted DockId here dumped all plots into the active tab on startup.
    // Skipped while the mouse is held down, so an in-flight drag isn't yanked.
    auto ensure_docked = [&](const std::string& name) {
        if (ImGui::GetIO().MouseDown[0]) return;
        ImGuiID persisted_dock = 0;
        if (ImGuiWindow* w = ImGui::FindWindowByName(name.c_str())) {
            persisted_dock = w->DockId;
        } else if (ImGuiWindowSettings* s =
                       ImGui::FindWindowSettingsByID(ImHashStr(name.c_str()))) {
            persisted_dock = s->DockId;   // will be applied on window's next Begin
        }
        if (persisted_dock != 0) return;
        ImGui::DockBuilderDockWindow(name.c_str(), ws_dock_id(ws.active_tab_id, sys));
    };

    // Cross-tab window move is handled entirely via native ImGui drag&drop: drop a plot's tab onto
    // a target tab in the workspace tab bar (BeginDragDropTarget in draw_custom_workspace_panel
    // accepts IMGUI_PAYLOAD_TYPE_WINDOW). No MMB menu here — MMB is claimed by heatmap/1D-plot
    // views for crosshair-drag / slice movement.

    auto axis_name = [&](bool over_var, int par_i, int var_i, bool over_h) -> std::string {
        if (over_h)   return "h";
        if (over_var) return (var_i >= 0 && var_i < (int)cs.vars.size())
                             ? cs.vars[var_i] + " (IC)" : "x";
        return (par_i >= 0 && par_i < (int)cs.params.size()) ? cs.params[par_i] : "param";
    };
    const std::string ax_x = axis_name(cs.shared.axis_x_over_var,
                                       cs.shared.axis_x_par_index,
                                       cs.shared.axis_x_var_index,
                                       cs.shared.axis_x_over_h);
    const std::string ax_y = axis_name(cs.shared.axis_y_over_var,
                                       cs.shared.axis_y_par_index,
                                       cs.shared.axis_y_var_index,
                                       cs.shared.axis_y_over_h);

    // Level 2D heatmaps
    struct L2Slot {
        std::string title;   // includes per-system suffix so imgui.ini isolates layout per system
        bool        show;
        bool        has_data;
        int         n_pts;
        const double* values;
        double lo_x, hi_x, lo_y, hi_y;
        double vmin, vmax;
        int    data_gen;
        int    owner_id;
        const std::vector<int>* flags;   // режим ячейки для подсказки (regime_cell_label)
    };
    L2Slot slots[3] = {};
    if (cs.shared.level_2d_enabled) {
        if (cs.shared.bif2d_enabled && !cs.bif_session.diagrams.empty()) {
            auto& d = cs.bif_session.diagrams[0];
            slots[0] = { "Custom Bif 2D" + suffix, true, d.last_run_2d_ok && !d.result_2d.values.empty(),
                         d.result_2d.n_pts, d.result_2d.values.data(),
                         d.result_2d.param_lo, d.result_2d.param_hi,
                         d.result_2d.param_lo_2, d.result_2d.param_hi_2,
                         d.result_2d.min_val, d.result_2d.max_val,
                         d.data_generation_2d, 0xCB1F2D ^ sys_owner_delta, &d.result_2d.flags };
        }
        if (cs.shared.lle2d_enabled && !cs.lle_session.curves.empty()) {
            auto& c = cs.lle_session.curves[0];
            slots[1] = { "Custom LLE 2D" + suffix, true, c.last_run_2d_ok && !c.result_2d.values.empty(),
                         c.result_2d.n_pts, c.result_2d.values.data(),
                         c.result_2d.param_lo, c.result_2d.param_hi,
                         c.result_2d.param_lo_2, c.result_2d.param_hi_2,
                         c.result_2d.min_val, c.result_2d.max_val,
                         c.data_generation_2d, 0xCE1E2D ^ sys_owner_delta, &c.result_2d.flags };
        }
        if (cs.shared.ls2d_enabled && !cs.ls_session.curves.empty()) {
            auto& c = cs.ls_session.curves[0];
            // Fallback plane pointer = first exponent (L1). The real plane for has_data / render is
            // picked later in the toolbar block, so has_data must NOT depend on the currently
            // selected exponent — otherwise picking "sum L_i" flips the slot to "No data yet"
            // because the fallback here goes nullptr.
            const double* values_default = c.result_2d.values.empty()
                ? nullptr : c.result_2d.values.data();
            double vmin = c.result_2d.min_val.empty() ? 0.0 : c.result_2d.min_val[0];
            double vmax = c.result_2d.max_val.empty() ? 1.0 : c.result_2d.max_val[0];
            slots[2] = { "Custom LS 2D" + suffix, true, c.last_run_2d_ok && !c.result_2d.values.empty(),
                         c.result_2d.n_pts, values_default,
                         c.result_2d.param_lo, c.result_2d.param_hi,
                         c.result_2d.param_lo_2, c.result_2d.param_hi_2,
                         vmin, vmax, c.data_generation_2d, 0xC152D0 ^ sys_owner_delta,
                         &c.result_2d.flags };
        }
    }
    // One-time init of per-slot HeatmapView colormap from the sub-session config (or app default).
    // Runs whenever the sub-session-level slot's colormap changes, so a slot's first appearance
    // picks up the persisted choice while later toolbar picks update both places.
    // (hm_init_cmap_seen is hoisted to the top of this function so the system-change reset can wipe it.)
    auto init_cmap_from_config = [&](int i, int cfg_cmap) {
        if (hm_init_cmap_seen[i] == cfg_cmap) return;
        hm_init_cmap_seen[i] = cfg_cmap;
        heatmaps[i].colormap =
            (HeatmapColormap)colormap_id_or(cfg_cmap, model.heatmap_colormap);
    };
    // Sync LS exponent choice from the sub-session's persisted display_exponent_idx (sentinel -1 =
    // sum L_i) on first appearance — otherwise session_from_json_custom loads the pref but the
    // HeatmapView silently starts at 0 (L1) until the user re-picks. Same "seen" pattern as the
    // colormap sync, so later user picks aren't clobbered. (ls_init_exp_seen hoisted to the top.)
    if (!cs.ls_session.curves.empty()) {
        int cfg_exp = cs.ls_session.curves[0].display_exponent_idx;
        if (ls_init_exp_seen != cfg_exp) {
            ls_init_exp_seen = cfg_exp;
            heatmaps[2].display_exponent_idx = cfg_exp;
        }
    }
    if (!cs.bif_session.diagrams.empty())
        init_cmap_from_config(0, cs.bif_session.diagrams[0].colormap_idx);
    if (!cs.lle_session.curves.empty())
        init_cmap_from_config(1, cs.lle_session.curves[0].colormap_idx);
    if (!cs.ls_session.curves.empty())
        init_cmap_from_config(2, cs.ls_session.curves[0].colormap_idx);

    // Автоскейл 2D-карт. Тот же узор, что в Parametric (draw_bifurcation_plot:
    // "bool fit = bdact.fit_request_2d; if (fit) ... = false"): запрос живёт в
    // конфиге подсессии и взводится приходом результата. Здесь в render уходило
    // жёстко false, поэтому после первого прогона HeatmapView фиксировал вид
    // (view_valid) и больше его не трогал: пересчёт в ДРУГОМ диапазоне рисовался
    // в старых границах — при расширении диапазона это выглядело как "новый
    // расчёт не сработал", потому что на экране оставался прежний узкий кусок.
    //
    // Забираем флаг у самой отрисовки, а не при сборке slots: карты сложены в
    // один док, у невидимой вкладки Begin вернёт false, и съеденный за неё
    // запрос пропал бы (ровно та же ловушка, что у Projection::fit_pending).
    auto take_fit_2d = [&cs](int slot) {
        bool* f = nullptr;
        if      (slot == 0 && !cs.bif_session.diagrams.empty())
            f = &cs.bif_session.diagrams[0].fit_request_2d;
        else if (slot == 1 && !cs.lle_session.curves.empty())
            f = &cs.lle_session.curves[0].fit_request_2d;
        else if (slot == 2 && !cs.ls_session.curves.empty())
            f = &cs.ls_session.curves[0].fit_request_2d;
        if (!f || !*f) return false;
        *f = false;
        return true;
    };
    for (int i = 0; i < 3; ++i) {
        if (!slots[i].show) continue;
        ensure_docked(slots[i].title);
        if (ImGui::Begin(slots[i].title.c_str())) {
            HeatmapView& hv = heatmaps[i];

            // Toolbar — mirrors draw_bifurcation_plot / draw_ls_plot layout
            // (colormap combo + autoscale + vmin/vmax + swap axes + LS
            // exponent picker for slot 2).
            ImGui::PushID(i);

            // LS-плоскость по выбранной экспоненте. Заполняется внутри
            // extras-колбэка тулбара (там же, где рисуется picker), а
            // используется ниже при render — поэтому объявлено до тулбара.
            const double* ls_plane = nullptr;
            double        ls_vmin = 0.0, ls_vmax = 1.0;
            int           ls_gen  = slots[i].data_gen;

            {
                HeatmapToolbarOpts topts;
                topts.persist_colormap = [&](int cm) {
                    // Persist per-slot in the owning sub-session config, so a
                    // saved _last_custom.json restores the choice.
                    if      (i == 0 && !cs.bif_session.diagrams.empty())
                        cs.bif_session.diagrams[0].colormap_idx = cm;
                    else if (i == 1 && !cs.lle_session.curves.empty())
                        cs.lle_session.curves[0].colormap_idx = cm;
                    else if (i == 2 && !cs.ls_session.curves.empty())
                        cs.ls_session.curves[0].colormap_idx = cm;
                    hm_init_cmap_seen[i] = cm;
                };
                topts.extras = [&]() {
                    if (i != 2 || cs.ls_session.curves.empty()) return;
                    auto& cact = cs.ls_session.curves[0];
                    if (!cact.last_run_2d_ok || cact.result_2d.n_exponents <= 0) return;
                    draw_ls_exponent_picker(hv, cact.result_2d.n_exponents,
                                            [&](int j) { cact.display_exponent_idx = j; });
                    ls_resolve_plane(cact, hv.display_exponent_idx,
                                     ls_plane, ls_vmin, ls_vmax, ls_gen);
                };
                draw_heatmap_toolbar(hv, topts);
            }

            if (!slots[i].has_data) {
                ImGui::TextDisabled("No data yet. Press Run / Run Level 2D.");
                ImGui::PopID();
            } else {
                hv.x_axis.name = ax_x;
                hv.y_axis.name = ax_y;
                // Лог-оси — как в Parametric (hb.x_axis.log_scale = bd.log_scale): движок
                // раскладывает узлы лог-равномерно, ось обязана это повторить, иначе тики и
                // snap курсора врут. Флаг берём из конфига слота, а не из живого shared:
                // конфиг получает log вместе с диапазоном на Run (apply_shared_to_*), поэтому
                // ось и данные всегда описывают один прогон.
                if (i == 0 && !cs.bif_session.diagrams.empty()) {
                    hv.x_axis.log_scale = cs.bif_session.diagrams[0].log_scale;
                    hv.y_axis.log_scale = cs.bif_session.diagrams[0].log_scale_2;
                } else if (i == 1 && !cs.lle_session.curves.empty()) {
                    hv.x_axis.log_scale = cs.lle_session.curves[0].log_scale;
                    hv.y_axis.log_scale = cs.lle_session.curves[0].log_scale_2;
                } else if (i == 2 && !cs.ls_session.curves.empty()) {
                    hv.x_axis.log_scale = cs.ls_session.curves[0].log_scale;
                    hv.y_axis.log_scale = cs.ls_session.curves[0].log_scale_2;
                }
                wire_2d_heatmap_interaction(hv, cs, q);

                // Export menu on right-click (parity with Parametric).
                const bool bd_busy = (i == 0) && cs.bif_session.in_flight
                                     && cs.bif_session.is_2d_run
                                     && cs.bif_session.running_diagram_index == 0;
                const bool lle_busy = (i == 1) && cs.lle_session.in_flight
                                      && cs.lle_session.is_2d_run
                                      && cs.lle_session.running_curve_index == 0;
                const bool ls_busy = (i == 2) && cs.ls_session.in_flight
                                     && cs.ls_session.is_2d_run
                                     && cs.ls_session.running_curve_index == 0;
                // Ctrl+T: цель — ОБЩИЙ config вкладки, а не конфиг подсессии.
                // Подсессии всё равно пересобираются из него срезами
                // apply_shared_to_* на каждом Run, и запись в них пропала бы.
                const std::vector<ViewRangeTarget> vrt = {
                    vr_target_fixed("Shared config",
                        &cs.shared.axis_x_lo_text, &cs.shared.axis_x_hi_text,
                        &cs.shared.axis_y_lo_text, &cs.shared.axis_y_hi_text) };
                hv.popup_extras = [i, &cs, &cb, bd_busy, lle_busy, ls_busy, &vrt, &model, &hv]() {
                    const bool busy = bd_busy || lle_busy || ls_busy;
                    draw_export_menu_item(busy, cb, [i, &cs](const std::string& path) {
                        if      (i == 0 && !cs.bif_session.diagrams.empty())
                            data_export::export_bif2d(cs.bif_session.diagrams[0].result_2d, path);
                        else if (i == 1 && !cs.lle_session.curves.empty())
                            data_export::export_lle2d(cs.lle_session.curves[0].result_2d, path);
                        else if (i == 2 && !cs.ls_session.curves.empty())
                            data_export::export_ls2d(cs.ls_session.curves[0].result_2d, path);
                    });
                    draw_view_range_menu(vrt, hv.x_axis, hv.y_axis, hv.swap_axes, &model);
                };

                // Route through LS-specific plane/vmin/vmax if the user
                // picked a different exponent; otherwise use the pre-baked
                // slot values.
                const double* values_ptr = (i == 2 && ls_plane) ? ls_plane : slots[i].values;
                double        vmin_use   = (i == 2 && ls_plane) ? ls_vmin  : slots[i].vmin;
                double        vmax_use   = (i == 2 && ls_plane) ? ls_vmax  : slots[i].vmax;
                int           gen_use    = (i == 2 && ls_plane) ? ls_gen   : slots[i].data_gen;
                if (slots[i].flags)
                    hv.cell_regime = regime_cell_label(*slots[i].flags, slots[i].n_pts);
                if (i == 0) {   // Bif 2D: значение — период, -1 / 0 — коды режима (как в Parametric)
                    hv.value_name = "period";
                    hv.value_is_regime_code = true;
                }

                ImVec2 avail  = ImGui::GetContentRegionAvail();
                ImVec2 origin = ImGui::GetCursorScreenPos();
                hv.render(*renderers_2d[i], origin, avail,
                          slots[i].owner_id, gen_use,
                          slots[i].n_pts, slots[i].n_pts,
                          values_ptr,
                          slots[i].lo_x, slots[i].hi_x,
                          slots[i].lo_y, slots[i].hi_y,
                          vmin_use, vmax_use,
                          take_fit_2d(i));
                handle_view_range_keys(vrt, hv.x_axis, hv.y_axis, plot_window_active(),
                               hv.swap_axes, &model);
                ImGui::PopID();
            }
        }
        ImGui::End();
    }

    // Level 1D slice plots (six independently toggleable)
    // Kind enum for the slot dispatch. Each slot picks its data from the
    // owning sub-session's slot [1] (X-slice) or [2] (Y-slice).
    enum class L1Kind { Bif, LLE, LS };
    struct L1Slot {
        std::string title;   // includes per-system suffix
        bool        show = false;
        L1Kind      kind = L1Kind::Bif;
        int         cfg_idx = -1;   // 1 = X-slice, 2 = Y-slice inside sub-session
        double      fix_pt = 0.0;   // world X of the OTHER-axis crosshair
    };
    L1Slot lslots[6] = {};
    if (cs.shared.level_1d_enabled) {
        double fx = cs.shared.fix_x_value;
        double fy = cs.shared.fix_y_value;
        if (cs.shared.bif1d_x_enabled) lslots[0] = { "Custom Bif 1D (X-slice)" + suffix, true, L1Kind::Bif, 1, fx };
        if (cs.shared.bif1d_y_enabled) lslots[1] = { "Custom Bif 1D (Y-slice)" + suffix, true, L1Kind::Bif, 2, fy };
        if (cs.shared.lle1d_x_enabled) lslots[2] = { "Custom LLE 1D (X-slice)" + suffix, true, L1Kind::LLE, 1, fx };
        if (cs.shared.lle1d_y_enabled) lslots[3] = { "Custom LLE 1D (Y-slice)" + suffix, true, L1Kind::LLE, 2, fy };
        if (cs.shared.ls1d_x_enabled)  lslots[4] = { "Custom LS 1D (X-slice)"  + suffix, true, L1Kind::LS,  1, fx };
        if (cs.shared.ls1d_y_enabled)  lslots[5] = { "Custom LS 1D (Y-slice)"  + suffix, true, L1Kind::LS,  2, fy };
    }

    // (l1_views / l1_renderers / l1_bufs statics hoisted to the top of the
    // function so the system-change reset can wipe them uniformly.)

    for (int i = 0; i < 6; ++i) {
        if (!lslots[i].show) continue;
        if (!l1_views[i])     l1_views[i]     = std::make_unique<Plot2DView>();
        if (!l1_renderers[i]) l1_renderers[i] = std::make_unique<PlotRenderer>();
        Plot2DView&  view = *l1_views[i];
        PlotRenderer& rnd = *l1_renderers[i];

        ensure_docked(lslots[i].title);
        if (!ImGui::Begin(lslots[i].title.c_str())) { ImGui::End(); continue; }
        ImGui::PushID(i);

        // Resolve owning config + result + real sweep-range from *_text
        // fields (result.param_lo/hi is often 0..1 default until engine
        // fills it — falling back to config text keeps the axis correct
        // when the user set 0.1..0.35 on the sweep).
        const auto axis_label_for_slot = [&](int cfg_idx) -> std::string {
            // cfg_idx 1 = X-slice → sweep_x_par_index or axis_x if inherit;
            // cfg_idx 2 = Y-slice → sweep_y_*.
            EffectiveSweep e = (cfg_idx == 1) ? effective_sweep_x(cs.shared)
                                              : effective_sweep_y(cs.shared);
            if (e.over_h) return "h";
            if (e.over_var)
                return (e.var_index >= 0 && e.var_index < (int)cs.vars.size())
                        ? cs.vars[e.var_index] + " (IC)" : "x (IC)";
            return (e.par_index >= 0 && e.par_index < (int)cs.params.size())
                    ? cs.params[e.par_index] : "param";
        };

        // ЕДИНАЯ конфигурация вида — та же функция, что применяет Parametric.
        // Раньше здесь стоял свой набор присваиваний, из-за чего одинаковые
        // диаграммы отличались между вкладками (см. configure_param_plot_view).
        const ParamPlotKind pkind =
            (lslots[i].kind == L1Kind::Bif) ? ParamPlotKind::Bifurcation
          : (lslots[i].kind == L1Kind::LLE) ? ParamPlotKind::LLE
                                            : ParamPlotKind::LS;
        configure_param_plot_view(view, pkind);

        bool ok = false;
        int  data_gen = 0;
        double param_lo = 0.0, param_hi = 1.0;
        int    n_pts = 0;
        // Лог-масштаб оси среза берём из КОНФИГА слота, а не из shared: конфиг получает log
        // вместе с param_lo/hi в apply_shared_to_bif1d на Run, поэтому ось и диапазон описывают
        // один прогон. Живой чекбокс в панели поменяет ось после Run (он входит в l1d-сигнатуру).
        bool   slice_log = false;
        // Y-axis label + series bookkeeping filled per-kind below.

        if (lslots[i].kind == L1Kind::Bif) {
            int idx = lslots[i].cfg_idx;
            if (idx < 0 || idx >= (int)cs.bif_session.diagrams.size()) { ImGui::PopID(); ImGui::End(); continue; }
            auto& d = cs.bif_session.diagrams[idx];
            ok = d.last_run_ok;
            data_gen = d.data_generation;
            param_lo = parse_ratio_or(d.param_lo_text, 0.0);
            param_hi = parse_ratio_or(d.param_hi_text, 1.0);
            n_pts = d.result.n_pts;
            slice_log = d.log_scale;
            view.x_axis.name = axis_label_for_slot(idx);
            view.y_axis.name = (d.writable_var >= 0 && d.writable_var < (int)cs.vars.size())
                               ? cs.vars[d.writable_var] : "X";
            // Как в Parametric: при plot_inter_peaks по Y идут интервалы
            // между пиками, а не сама переменная.
            if (d.plot_inter_peaks) view.y_axis.name += " interval";
            // Custom point style — тот же тулбар и та же семантика, что в
            // Parametric (состояние живёт в самой БД, поэтому вкладки не
            // расходятся). Здесь член ровно один (X- или Y-срез),
            // синхронизировать нечего.
            if (draw_point_style_toolbar(d, "custombd"))
                cs.workspace.dirty = true;   // → пересохранение _last_custom.json
            apply_point_style(view, d);
        } else if (lslots[i].kind == L1Kind::LLE) {
            int idx = lslots[i].cfg_idx;
            if (idx < 0 || idx >= (int)cs.lle_session.curves.size()) { ImGui::PopID(); ImGui::End(); continue; }
            auto& c = cs.lle_session.curves[idx];
            ok = c.last_run_ok;
            data_gen = c.data_generation;
            param_lo = parse_ratio_or(c.param_lo_text, 0.0);
            param_hi = parse_ratio_or(c.param_hi_text, 1.0);
            n_pts = (int)c.result.lyapunov.size();
            slice_log = c.log_scale;
            view.x_axis.name = axis_label_for_slot(idx);
        } else { // LS
            int idx = lslots[i].cfg_idx;
            if (idx < 0 || idx >= (int)cs.ls_session.curves.size()) { ImGui::PopID(); ImGui::End(); continue; }
            auto& c = cs.ls_session.curves[idx];
            ok = c.last_run_ok;
            data_gen = c.data_generation;
            param_lo = parse_ratio_or(c.param_lo_text, 0.0);
            param_hi = parse_ratio_or(c.param_hi_text, 1.0);
            n_pts = c.result.n_pts;
            slice_log = c.log_scale;
            view.x_axis.name = axis_label_for_slot(idx);
            // LS shows every exponent as its own coloured line (parity with
            // draw_ls_plot in Parametric) — no exponent picker here; series
            // are built in the render block below.
        }

        if (!ok) {
            ImGui::TextDisabled("No data yet.");
            ImGui::PopID(); ImGui::End(); continue;
        }

        // X range for autofit (independent of point density).
        view.x_fit_use_explicit = true;
        view.x_fit_min = std::min(param_lo, param_hi);
        view.x_fit_max = std::max(param_lo, param_hi);
        // Лог-ось — как в Parametric; узлы движка лежат лог-равномерно, ось
        // и snap обязаны это повторить. log при lo<=0 невозможен (движок
        // такой Run отклоняет), но проверку держим здесь тоже — конфиг мог
        // прийти из старой сессии.
        view.x_axis.log_scale = slice_log && view.x_fit_min > 0.0;

        // Sweep-position crosshair — vertical line at fix_x (X-slice)
        // or fix_y (Y-slice), synced with the slider / drag on 2D heatmaps.
        // Hidden if Level 1D isn't the driver (falls out to NaN → no draw).
        view.crosshair_x = cs.shared.level_1d_enabled
                           ? lslots[i].fix_pt
                           : std::numeric_limits<double>::quiet_NaN();
        // Красим вертикальный крест по тому свипу, которому принадлежит срез: один цвет на 2D
        // хитмапе и её 1D срезе показывает, на какую ось смотришь.
        //  cfg_idx == 1 → X-срез → совпадает с вертикальной X-линией хитмапы
        //  cfg_idx == 2 → Y-срез → совпадает с горизонтальной Y-линией
        view.crosshair_x_color = (lslots[i].cfg_idx == 2)
                                  ? 0xFFFF9028u   // orange = Y sweep
                                  : 0xFF50A0FFu;  // blue   = X sweep

        // Перетаскивание креста: MMB или Shift+LMB внутри плота двигает соответствующий fix_*
        // вдоль оси свипа этого среза (cfg_idx 1 → fix_x, 2 → fix_y). Отпускание запускает общий
        // авто-пересчёт (Level 1D + Phase) через last_fix_{x,y}_change_time — тот же дебаунс,
        // что у drag'а L2D-хитмапы и fix-слайдеров, чтобы все три источника вели себя одинаково.
        const bool slice_is_x = (lslots[i].cfg_idx == 1);
        // Snap crosshair drag to this slice's OWN grid nodes — the sampled
        // points the 1D compute actually produced — so the crosshair always
        // lands on a data point rather than drifting between them.
        // Captured by value (fresh lambda per frame, so no lifetime hazard).
        int   snap_n = parse_int_or(slice_is_x ? cs.shared.n_x_1d_text
                                              : cs.shared.n_y_1d_text, 2);
        if (snap_n < 2) snap_n = 2;
        const double snap_lo   = std::min(param_lo, param_hi);
        const double snap_hi   = std::max(param_lo, param_hi);
        const double snap_step = (snap_hi - snap_lo) / (double)(snap_n - 1);
        // При log-сетке узлы не равноудалены — индекс и позицию считаем по
        // логарифму (та же формула, что у движка), иначе крестик садится
        // между реальными точками.
        const bool snap_log = view.x_axis.log_scale && snap_lo > 0.0 && snap_hi > 0.0;
        // Конвенция узлов зависит от того, шёл ли срез continuation'ом: у него
        // своя формула (см. sweep_value_at). Флаг берём общий по вкладке — им
        // же apply_shared_to_bif1d проставляет continuation в под-конфиг.
        const bool snap_cont = cs.shared.continuation_1d_enabled;
        auto snap_to_grid = [snap_lo, snap_hi, snap_step, snap_n, snap_log, snap_cont](double w) {
            if (snap_step <= 0.0) return w;
            int i;
            if (snap_log) {
                if (!(w > 0.0)) return snap_lo;
                const double l0 = std::log10(snap_lo), l1 = std::log10(snap_hi);
                i = (int)std::round((std::log10(w) - l0) / ((l1 - l0) / (double)(snap_n - 1)));
            } else {
                i = (int)std::round((w - snap_lo) / snap_step);
            }
            i = std::clamp(i, 0, snap_n - 1);
            const double s = sweep_value_at(i, snap_n, snap_lo, snap_hi, snap_log,
                                            /*reverse*/ false, snap_cont);
            return std::clamp(s, snap_lo, snap_hi);
        };
        // Тот же snap и для тиков осей / hover-readout, что и в Parametric —
        // раньше в Custom он стоял только на crosshair-драге, поэтому tooltip
        // здесь показывал промежуточные X, которых в данных нет.
        apply_snap_x(view, snap_lo, snap_hi, snap_n);
        view.on_left_drag = [&cs, slice_is_x, snap_to_grid](double world_x) {
            double w = snap_to_grid(world_x);
            if (slice_is_x) {
                cs.shared.fix_x_value = w;
                cs.shared.last_fix_x_change_time = ImGui::GetTime();
            } else {
                cs.shared.fix_y_value = w;
                cs.shared.last_fix_y_change_time = ImGui::GetTime();
            }
        };
        view.on_left_click = [&cs, slice_is_x, snap_to_grid](double world_x) {
            double w = snap_to_grid(world_x);
            if (slice_is_x) {
                cs.shared.fix_x_value = w;
                cs.shared.last_fix_x_change_time = ImGui::GetTime();
            } else {
                cs.shared.fix_y_value = w;
                cs.shared.last_fix_y_change_time = ImGui::GetTime();
            }
        };

        // Build series. Bif = one scatter series with all peak samples;
        // LLE = one line; LS = N line series, one per exponent (parity
        // with draw_ls_plot).
        std::vector<PlotSeriesInput> series_in;
        std::vector<bool> init_vis, glob_vis;
        auto& bufs = l1_bufs[i];
        bufs.clear();

        int series_gen = data_gen;
        if (lslots[i].kind == L1Kind::Bif) {
            auto& d = cs.bif_session.diagrams[lslots[i].cfg_idx];
            const auto& r = d.result;
            // Как в Parametric: источник — peak_times при plot_inter_peaks,
            // иначе сами точки бифуркации. Раньше Custom всегда брал
            // bifurcation_points, из-за чего тоггл «inter-peaks» на него не влиял.
            const auto& source = d.plot_inter_peaks ? r.peak_times
                                                    : r.bifurcation_points;
            int n = r.n_pts > 0 ? r.n_pts : 1;
            bufs.emplace_back();
            auto& buf = bufs.back();
            for (int p = 0; p < (int)source.size(); ++p) {
                // Diverged-точки пропускаем (в Parametric так и было; здесь
                // проверки не было, и разошедшиеся точки попадали на график).
                if (p < (int)r.flags.size() && !regime_is_oscillation(r.flags[p])) continue;
                double px = sweep_value_at(p, n, param_lo, param_hi, slice_log,
                                           /*reverse*/ false, d.continuation);
                for (double y : source[p]) {
                    if (!std::isfinite(y)) continue;
                    buf.push_back((double)px);
                    buf.push_back((double)y);
                }
            }
            PlotSeriesInput si;
            si.points = buf.empty() ? nullptr : buf.data();
            si.n_points = (int)(buf.size() / 2);
            si.color = ic_base_color(0);
            if (d.custom_point_style) si.color.w = d.point_alpha;
            si.label = d.label.empty() ? "bd" : d.label;
            series_in.push_back(si);
            series_gen = data_gen * 2 + (d.plot_inter_peaks ? 1 : 0);
        } else if (lslots[i].kind == L1Kind::LLE) {
            auto& c = cs.lle_session.curves[lslots[i].cfg_idx];
            const auto& r = c.result;
            int n = (int)r.lyapunov.size();
            bufs.emplace_back();
            auto& buf = bufs.back();
            buf.reserve(n * 2);
            int total_pts = 0;
            for (int p = 0; p < n; ++p) {
                // Diverged / NaN пропускаем — как в Parametric. Раньше здесь
                // проверок не было, поэтому sentinel-значения расхождения
                // рисовались как выброс.
                if (p < (int)r.flags.size() && !regime_is_oscillation(r.flags[p])) continue;
                double y = r.lyapunov[p];
                if (!std::isfinite(y)) continue;
                double px = sweep_value_at(p, n, param_lo, param_hi, slice_log,
                                           /*reverse*/ false, c.continuation);
                buf.push_back((double)px);
                buf.push_back((double)y);
                ++total_pts;
            }
            PlotSeriesInput si;
            si.points = buf.empty() ? nullptr : buf.data();
            si.n_points = total_pts;
            si.color = ic_base_color(0);
            si.label = c.label.empty() ? "LLE" : c.label;
            series_in.push_back(si);
        } else { // LS — N series, one per exponent (parity with draw_ls_plot).
            auto& c = cs.ls_session.curves[lslots[i].cfg_idx];
            const auto& r = c.result;
            int n = r.n_pts;
            int N = r.n_exponents;
            if (N > 0 && n > 0 && (int)r.spectrum.size() == n) {
                bufs.reserve(N);
                for (int j = 0; j < N; ++j) {
                    bufs.emplace_back();
                    auto& buf = bufs.back();
                    buf.reserve(n * 2);
                    int total_pts = 0;
                    for (int p = 0; p < n; ++p) {
                        if (p < (int)r.flags.size() && !regime_is_oscillation(r.flags[p])) continue;
                        if (p >= (int)r.spectrum.size()) continue;
                        const auto& row = r.spectrum[p];
                        if (j >= (int)row.size()) continue;
                        double px = sweep_value_at(p, n, param_lo, param_hi, slice_log,
                                                   /*reverse*/ false, c.continuation);
                        double y = row[j];
                        if (!std::isfinite(y)) continue;
                        buf.push_back((double)px);
                        buf.push_back((double)y);
                        ++total_pts;
                    }
                    PlotSeriesInput si;
                    si.points = buf.empty() ? nullptr : buf.data();
                    si.n_points = total_pts;
                    si.color = ic_base_color(j);
                    // Как в Parametric: "<label> Lj" при N>1, иначе <label>.
                    const std::string base = c.label.empty() ? "LS" : c.label;
                    si.label = (N > 1) ? (base + " L" + std::to_string(j + 1)) : base;
                    series_in.push_back(si);
                }
                series_gen = data_gen * 64 + N;  // rebuild VBO if N changes
            }
        }
        init_vis.assign(series_in.size(), true);
        glob_vis.assign(series_in.size(), true);

        // Right-click "Export data..." — паритет с Parametric, где он есть у
        // всех 1D-графиков. В Custom его не было ни на одном из 6 слотов.
        // Ctrl+T: срез идёт по своей оси общего config'а (слот 1 — X, 2 — Y),
        // по Y — сама считаемая величина, входа в расчёте у неё нет.
        //
        // Писать нужно в ДЕЙСТВУЮЩИЕ поля, а не всегда в sweep_*. При
        // inherit_sweep_from_2d (значение по умолчанию) effective_sweep_x/y
        // читают axis_*, а sweep_* лежат мёртвым грузом: запись туда не меняла
        // ни сам срез, ни 2D-карту, и Ctrl+T выглядел как ничего не делающий.
        // Через axis_* диапазон заодно приезжает на соответствующую ось L2D —
        // это и есть "та же зона" на двумерной диаграмме.
        const bool  vr_slice_is_x = (lslots[i].cfg_idx == 1);
        const bool  vr_inherited  = cs.shared.inherit_sweep_from_2d
                                 && cs.shared.level_2d_enabled;
        const std::vector<ViewRangeTarget> vrt = { vr_target_fixed(
            vr_inherited ? (vr_slice_is_x ? "Shared config (2D X axis)"
                                          : "Shared config (2D Y axis)")
                         : (vr_slice_is_x ? "Shared config (X slice)"
                                          : "Shared config (Y slice)"),
            vr_inherited ? (vr_slice_is_x ? &cs.shared.axis_x_lo_text
                                          : &cs.shared.axis_y_lo_text)
                         : (vr_slice_is_x ? &cs.shared.sweep_x_lo_text
                                          : &cs.shared.sweep_y_lo_text),
            vr_inherited ? (vr_slice_is_x ? &cs.shared.axis_x_hi_text
                                          : &cs.shared.axis_y_hi_text)
                         : (vr_slice_is_x ? &cs.shared.sweep_x_hi_text
                                          : &cs.shared.sweep_y_hi_text)) };
        view.popup_extras = [i, &lslots, &cs, &cb, &vrt, &model, &view]() {
            const bool busy = (lslots[i].kind == L1Kind::Bif) ? cs.bif_session.in_flight
                            : (lslots[i].kind == L1Kind::LLE) ? cs.lle_session.in_flight
                                                              : cs.ls_session.in_flight;
            draw_export_menu_item(busy, cb, [i, &lslots, &cs](const std::string& path) {
                const int ci = lslots[i].cfg_idx;
                if (lslots[i].kind == L1Kind::Bif) {
                    if (ci >= 0 && ci < (int)cs.bif_session.diagrams.size())
                        data_export::export_bif1d(cs.bif_session.diagrams[ci].result, path);
                } else if (lslots[i].kind == L1Kind::LLE) {
                    if (ci >= 0 && ci < (int)cs.lle_session.curves.size())
                        data_export::export_lle1d(cs.lle_session.curves[ci].result, path);
                } else {
                    if (ci >= 0 && ci < (int)cs.ls_session.curves.size())
                        data_export::export_ls1d(cs.ls_session.curves[ci].result, path);
                }
            });
            draw_view_range_menu(vrt, view.x_axis, view.y_axis, /*swapped*/ false, &model);
        };

        // Autofit whenever the underlying result changed (bif/lle/ls each
        // set fit_request in apply_*_result on completion of run_async).
        bool fit = false;
        if (lslots[i].kind == L1Kind::Bif) {
            auto& d = cs.bif_session.diagrams[lslots[i].cfg_idx];
            fit = d.fit_request; if (fit) d.fit_request = false;
        } else if (lslots[i].kind == L1Kind::LLE) {
            auto& c = cs.lle_session.curves[lslots[i].cfg_idx];
            fit = c.fit_request; if (fit) c.fit_request = false;
        } else {
            auto& c = cs.ls_session.curves[lslots[i].cfg_idx];
            fit = c.fit_request; if (fit) c.fit_request = false;
        }

        ImVec2 avail  = ImGui::GetContentRegionAvail();
        ImVec2 origin = ImGui::GetCursorScreenPos();
        view.render(rnd, origin, avail,
                    /*owner_id*/ (0xC10000 + i) ^ sys_owner_delta,
                    series_gen,
                    series_in, init_vis, glob_vis, fit);
        handle_view_range_keys(vrt, view.x_axis, view.y_axis, plot_window_active(), /*swapped*/ false, &model);
        ImGui::PopID();
        ImGui::End();
    }

    // --- Level 3: phase controls live inside the L3 config panel (see
    // draw_level3_detail). Here we only spawn the projection windows so
    // 2D / 3D / TimeDomain plots dock alongside the other custom plots.
    if (cs.shared.level_phase_enabled && cs.shared.level3_kind == 0) {
        // Auto-place phase projections into the active tab on first appearance. Suffix + owner_id
        // delta both keep per-system isolation: suffix for imgui.ini dock state, delta for the
        // SHARED PlotRenderer cache (otherwise Rossler's projection 0 saw Chen's cached FBO).
        // Диаграмма признаков рисуется в осях кластеризации того же 2D-конфига, чьи eps/множители
        // стоят в Shared config — иначе по ней нельзя судить, что сольёт dbscan (FeatureClusterParams).
        FeatureClusterParams clust;
        if (!cs.bif_session.diagrams.empty()) {
            const auto& bd = cs.bif_session.diagrams[0];
            clust.valid         = true;
            clust.mult_peak     = parse_ratio_or(bd.mult_peak_text,     (double)mult_peak);
            clust.mult_interval = parse_ratio_or(bd.mult_interval_text, (double)mult_interval);
            clust.eps           = parse_ratio_or(bd.eps_dbscan_text,    0.1);
        }
        draw_projection_windows(cs.phase_session, cb,
            [&](int /*idx*/, const std::string& title) { ensure_docked(title); },
            {}, suffix, sys_owner_delta, {}, clust);
    }
    // Level 3 Basins window (unchanged HeatmapView minimal renderer)
    if (cs.shared.level_phase_enabled && cs.shared.level3_kind == 1) {
        // Автоскейл — как у 2D-карт выше (см. take_fit_2d): сюда тоже уходило
        // жёсткое false, и карта, посчитанная в новой сетке НУ, рисовалась в
        // границах предыдущего прогона.
        auto take_fit_basins = [&cs]() {
            if (cs.basins_session.configs.empty()) return false;
            bool& f = cs.basins_session.configs[0].fit_request;
            if (!f) return false;
            f = false;
            return true;
        };
        std::string basins_title = "Custom Basins" + suffix;
        ensure_docked(basins_title);
        if (ImGui::Begin(basins_title.c_str())) {
                // Basins — HeatmapView of basin_idx (cluster id) — simpler
                // than draw_basins_plot's toolbar (feature switch stays in
                // detail panel), matches user request for a visible result.
                auto& bsn = cs.basins_session;
                if (bsn.in_flight) {
                    ImGui::TextDisabled("Computing basins...");
                } else if (bsn.configs.empty() || !bsn.configs[0].last_run_ok) {
                    ImGui::TextDisabled("No basins result yet.");
                } else {
                    // bsn_hv / bsn_renderer hoisted to top of function for
                    // system-change reset (see the reset block above).
                    const auto& bc = bsn.configs[0];
                    const auto& r  = bc.result;
                    if (r.n_pts <= 0 || r.basin_idx.empty()) {
                        ImGui::TextDisabled("Basins result empty.");
                    } else {
                        // Convert int cluster ids to double for HeatmapView.
                        static std::vector<double> bsn_values;
                        bsn_values.resize(r.basin_idx.size());
                        double vmin =  std::numeric_limits<double>::infinity();
                        double vmax = -std::numeric_limits<double>::infinity();
                        for (size_t k = 0; k < r.basin_idx.size(); ++k) {
                            double v = (double)r.basin_idx[k];
                            bsn_values[k] = v;
                            if (v < vmin) vmin = v;
                            if (v > vmax) vmax = v;
                        }
                        if (!std::isfinite(vmin)) { vmin = 0.0; vmax = 1.0; }
                        // Тот же тулбар, что и у остальных хитмап: раньше здесь был только combo
                        // Colormap, без autoscale/vmin/vmax и Swap axes, хотя диаграмма ровно такая
                        // же. Выбор colormap'а теперь персистится в config (слот 0 = таб "Basins") —
                        // раньше он жил только в static-view и терялся при перезапуске.
                        {
                            auto& bcfg = bsn.configs[0];
                            const int cm = colormap_id_or(bcfg.colormap_idx[0],
                                                          model.basins_colormap);
                            bsn_hv.colormap = (HeatmapColormap)cm;

                            HeatmapToolbarOpts topts;
                            topts.persist_colormap =
                                [&bcfg](int picked) { bcfg.colormap_idx[0] = picked; };
                            draw_heatmap_toolbar(bsn_hv, topts);
                        }
                        bsn_hv.x_axis.name = (bc.axis_x_var >= 0 && bc.axis_x_var < (int)cs.vars.size())
                                              ? cs.vars[bc.axis_x_var] : "x";
                        bsn_hv.y_axis.name = (bc.axis_y_var >= 0 && bc.axis_y_var < (int)cs.vars.size())
                                              ? cs.vars[bc.axis_y_var] : "y";
                        bsn_hv.cell_regime = regime_cell_label(r.helpful_array, r.n_pts);
                        bsn_hv.value_name = "basin";
                        bsn_hv.value_is_regime_code = true;
                        ImVec2 avail  = ImGui::GetContentRegionAvail();
                        ImVec2 origin = ImGui::GetCursorScreenPos();
                        bsn_hv.render(bsn_renderer, origin, avail,
                                      /*owner_id*/ 0xCBA51E5 ^ sys_owner_delta,
                                      bc.data_generation,
                                      r.n_pts, r.n_pts,
                                      bsn_values.data(),
                                      r.axis_x_lo, r.axis_x_hi,
                                      r.axis_y_lo, r.axis_y_hi,
                                      vmin, vmax,
                                      take_fit_basins());
                    }
                }
            }
        ImGui::End();
    }
}
}

using namespace ucuda::ui_detail;
void gui_process_deferred(AppModel& model, const GuiCallbacks& cb) {
    if (!model.pending_netlist_export) return;
    model.pending_netlist_export = false;
    if (!cb.pick_save_file_netlist) return;

    PhaseAnalysisSession& s = model.phase_session;
    if (!s.result.circuit_valid) return;

    const std::string path = cb.pick_save_file_netlist();
    if (path.empty()) return;

    NetlistOptions no;
    no.title      = "U-CUDA synthesized circuit";
    no.var_names  = s.vars;
    no.rhs_text   = s.sys.rhs;
    no.scale      = s.result.circuit_scale;
    no.x0_volts   = s.result.circuit_x0;
    no.opamp      = s.result.circuit_opamp;
    no.h_ode      = s.result.circuit_h_ode;
    no.t_end_ode  = s.result.circuit_t_end_ode;
    no.dialect    = s.circuit_netlist_poly ? NetlistDialect::Poly : NetlistDialect::Behavioral;

    const std::string text = emit_spice_netlist(s.result.circuit_graph, no);
    std::ofstream f(path, std::ios::binary);
    if (f) { f << text; s.result.circuit_status = "netlist written to " + path; }
    else     s.result.circuit_status = "cannot write " + path;
}
