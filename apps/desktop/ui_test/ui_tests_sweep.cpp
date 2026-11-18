// "sweep/all_tabs": walks every mode tab, every reachable sub-tab and every scroll page.
// For each input field: hover it (ImGui counts items sharing the hovered ID -> ID conflict) and
// right-click it (its context menu is summarised per tab). Output: "[sweep] ..." lines on stdout,
// a screenshot per visited state in <out>/tabs/. Fails on ID conflicts or on the removed
// "apply to active config of each tab" menu item.
#include "ui_test_host.h"
#include "ui_test_modes.h"

#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_te_engine.h"
#include "imgui_te_context.h"
#include "imgui_capture_tool.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace ui_test {
namespace {

constexpr int kMaxStatesPerMode   = 60;    // sub-tab selections per mode tab
constexpr int kMaxFieldsPerState  = 150;   // new fields checked per state
constexpr int kMaxPagesPerWindow  = 12;    // scroll pages per window

struct Sweep {
    ImGuiTestContext* ctx;
    AppModel* m_model = nullptr;
    std::set<ImGuiID> checked;                      // fields already checked (whole sweep)
    std::map<std::string, int> menus;               // menu signature -> count (current mode)
    int conflicts = 0, removed_item = 0, fields = 0, menus_n = 0, states = 0, shots = 0;
    std::string mode;
    bool verbose = std::getenv("UCUDA_SWEEP_VERBOSE") != nullptr;           // per-window counts
    bool edit    = std::getenv("UCUDA_SWEEP_NOEDIT") == nullptr;            // field edit checks
    int edited = 0, edited_numeric = 0, edit_changed = 0, edit_failed = 0, edit_multiline = 0, edit_readonly = 0;
    const char* only = std::getenv("UCUDA_SWEEP_ONLY");                      // mode label substring

    static std::string file_safe(const std::string& s) {
        std::string r;
        for (char c : s) r += (std::isalnum((unsigned char)c) ? c : '_');
        return r;
    }
    // Digits -> '#': "Reset to default (9950)" and "(0.5)" give the same signature.
    static std::string signature(const std::string& s) {
        std::string r;
        for (size_t i = 0; i < s.size(); ++i) {
            const bool num = std::isdigit((unsigned char)s[i]) ||
                ((s[i] == '.' || s[i] == '-' || s[i] == 'e') && i + 1 < s.size() && std::isdigit((unsigned char)s[i + 1]) &&
                 i > 0 && (std::isdigit((unsigned char)s[i - 1]) || s[i - 1] == '('));
            if (num) { if (r.empty() || r.back() != '#') r += '#'; }
            else r += s[i];
        }
        return r;
    }

    void screenshot(const std::string& tag) {
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        std::string file = (std::filesystem::path(options().out_dir) / "tabs" /
                            (std::to_string(++shots) + "_" + file_safe(tag) + ".png")).string();
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(file).parent_path(), ec);
        ctx->CaptureReset();
        ImStrncpy(ctx->CaptureArgs->InOutputFile, file.c_str(), IM_ARRAYSIZE(ctx->CaptureArgs->InOutputFile));
        ctx->CaptureArgs->InCaptureRect = ImRect(vp->Pos, ImVec2(vp->Pos.x + vp->Size.x, vp->Pos.y + vp->Size.y));
        ctx->CaptureArgs->InPadding = 0.0f;
        ctx->CaptureScreenshot(ImGuiCaptureFlags_Instant | ImGuiCaptureFlags_HideMouseCursor);
    }

    // One field: hover -> ID conflict check, right-click -> menu summary.
    void check_field(ImGuiID id, const char* label, const char* win) {
        ImGuiContext& g = *GImGui;
        ctx->MouseMove(id, ImGuiTestOpFlags_NoError | ImGuiTestOpFlags_NoFocusWindow);
        ctx->Yield(2);
        if (g.HoveredIdPreviousFrame != id) return;   // covered by another window
        ++fields;
        if (g.HoveredIdPreviousFrameItemCount > 1) {
            ++conflicts;
            std::printf("[sweep] %s: ID CONFLICT x%d  0x%08X \"%s\" in \"%s\"\n", mode.c_str(),
                        g.HoveredIdPreviousFrameItemCount, id, label, win);
        }
        ctx->ItemClick(id, ImGuiMouseButton_Right, ImGuiTestOpFlags_NoError | ImGuiTestOpFlags_NoFocusWindow);
        ctx->Yield(2);
        if (g.OpenPopupStack.Size > 0 && g.OpenPopupStack.back().Window) {
            ++menus_n;
            ImGuiTestItemList pitems;
            ctx->GatherItems(&pitems, ImGuiTestRef(g.OpenPopupStack.back().Window->ID), 1);
            std::string labels;
            for (const ImGuiTestItemInfo& pi : pitems) {
                if (std::strncmp(pi.DebugLabel, "Apply to active", 15) == 0 ||
                    std::strncmp(pi.DebugLabel, "Apply to all calc", 17) == 0) {
                    ++removed_item;
                    std::printf("[sweep] %s: REMOVED ITEM back in menu of \"%s\"\n", mode.c_str(), label);
                }
                if (pi.DebugLabel[0]) { if (!labels.empty()) labels += " | "; labels += pi.DebugLabel; }
            }
            ++menus[signature(labels)];
            ctx->PopupCloseAll();
            ctx->Yield(2);
        }
        if (edit) check_edit(id, label, win);
    }

    // "0.01" -> "0.010": same number, different text. Only plain decimals (no exponent, not an
    // integer: an integer field may reject "200.0").
    static std::string equivalent(const char* v) {
        std::string s = v;
        if (s.empty() || s.find('.') == std::string::npos || s.find_first_of("eE") != std::string::npos) return {};
        char* end = nullptr;
        std::strtod(s.c_str(), &end);
        if (!end || *end) return {};
        return s + "0";
    }

    // Field edit: read, type an equivalent value, read, put the original back, read.
    // A field that cannot be restored exactly fails the test; a field that shows something else
    // than what was typed is reported (it may normalise the text on purpose).
    void check_edit(ImGuiID id, const char* label, const char* win) {
        ImGuiTestItemInfo info = ctx->ItemInfo(id, ImGuiTestOpFlags_NoError);
        if (!info.ID) return;
        if (info.RectFull.GetHeight() > ImGui::GetFrameHeight() * 1.5f) { ++edit_multiline; return; }
        if (info.ItemFlags & ImGuiItemFlags_ReadOnly) { ++edit_readonly; return; }
        char v0[1024] = {}, v1[1024] = {}, v2[1024] = {};
        ctx->ItemReadAsString(id, v0, sizeof v0);
        if (ctx->IsError()) return;
        ++edited;
        const std::string alt = equivalent(v0);
        if (!alt.empty()) {
            ++edited_numeric;
            ctx->ItemInputValue(id, alt.c_str());
            ctx->Yield(2);
            ctx->ItemReadAsString(id, v1, sizeof v1);
            if (alt != v1) {
                ++edit_changed;
                std::printf("[sweep] %s: EDIT typed \"%s\", field shows \"%s\"  (\"%s\" in \"%s\")\n",
                            mode.c_str(), alt.c_str(), v1, label, win);
            }
        }
        ctx->ItemInputValue(id, v0);
        ctx->Yield(2);
        ctx->ItemReadAsString(id, v2, sizeof v2);
        if (std::strcmp(v0, v2) != 0) {
            ++edit_failed;
            std::printf("[sweep] %s: EDIT NOT RESTORED \"%s\" -> \"%s\"  (\"%s\" in \"%s\")\n",
                        mode.c_str(), v0, v2, label, win);
        }
        ctx->Yield(1);
    }

    // All windows of the current frame, every scroll page, every new input field.
    void scan_state(const std::string& tag) {
        ImGuiContext& g = *GImGui;
        ++states;
        screenshot(mode + " - " + tag);
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        const ImRect view(vp->Pos, ImVec2(vp->Pos.x + vp->Size.x, vp->Pos.y + vp->Size.y));
        std::vector<ImGuiWindow*> wins;
        for (ImGuiWindow* w : g.Windows)
            if (w->WasActive && !w->Hidden && !(w->Flags & (ImGuiWindowFlags_Popup | ImGuiWindowFlags_Tooltip)))
                wins.push_back(w);
        int fresh = 0, pages_total = 0;
        for (ImGuiWindow* w : wins) {
            const float scroll_max = w->ScrollMax.y;
            const float step = std::max(60.0f, (w->InnerRect.GetHeight()) * 0.8f);
            int pages = 1;
            if (scroll_max > 0.0f) pages = std::min(kMaxPagesPerWindow, 1 + (int)std::ceil(scroll_max / step));
            for (int p = 0; p < pages && fresh < kMaxFieldsPerState; ++p) {
                if (scroll_max > 0.0f) {
                    ImGui::SetScrollY(w, std::min(scroll_max, p * step));
                    ctx->Yield(3);
                }
                ++pages_total;
                ImGuiTestItemList items;
                ctx->GatherItems(&items, ImGuiTestRef(w->ID), -1);
                struct F { ImGuiID id; std::string label; };
                std::vector<F> todo;
                int n_in = 0, n_vis = 0, n_view = 0, n_dup = 0;
                for (const ImGuiTestItemInfo& it : items) {
                    if (!(it.StatusFlags & ImGuiItemStatusFlags_Inputable)) continue;
                    ++n_in;
                    if (!(it.StatusFlags & ImGuiItemStatusFlags_Visible) || (it.ItemFlags & ImGuiItemFlags_Disabled)) continue;
                    ++n_vis;
                    if (!view.Contains(it.RectClipped.GetCenter()) || it.RectClipped.GetWidth() < 4.0f ||
                        it.RectClipped.GetHeight() < 4.0f) continue;
                    ++n_view;
                    if (checked.count(it.ID)) { ++n_dup; continue; }
                    todo.push_back({ it.ID, it.DebugLabel });
                }
                if (verbose)
                    std::printf("[sweep]   win \"%s\" page %d/%d scroll %.0f/%.0f: items %d inputable %d visible %d in-view %d seen %d\n",
                                w->Name, p + 1, pages, w->Scroll.y, scroll_max, items.GetSize(), n_in, n_vis, n_view, n_dup);
                for (const F& f : todo) {
                    if (fresh >= kMaxFieldsPerState) break;
                    checked.insert(f.id);
                    ++fresh;
                    check_field(f.id, f.label.c_str(), w->Name);
                }
            }
            if (scroll_max > 0.0f) { ImGui::SetScrollY(w, 0.0f); ctx->Yield(1); }
        }
        std::printf("[sweep] %s | %s: %d window(s), %d page(s), %d new field(s)\n",
                    mode.c_str(), tag.c_str(), (int)wins.size(), pages_total, fresh);
        std::fflush(stdout);
    }

    // Tab bars drawn in the last frames, first one with a tab not yet visited.
    // Widget tab bars live in g.TabBars; dock node tab bars (docked windows, e.g. "Bifurcation 2D |
    // LLE 2D", Library "ODE | DAE | MAP") belong to their dock nodes.
    bool next_tab(std::set<std::pair<ImGuiID, ImGuiID>>& visited, std::string& name) {
        ImGuiContext& g = *GImGui;
        std::vector<ImGuiTabBar*> bars;
        for (int i = 0; i < g.TabBars.GetMapSize(); ++i)
            if (ImGuiTabBar* tb = g.TabBars.TryGetMapData(i)) bars.push_back(tb);
        for (const ImGuiStoragePair& p : g.DockContext.Nodes.Data)
            if (ImGuiDockNode* node = (ImGuiDockNode*)p.val_p)
                if (node->TabBar && node->IsVisible) bars.push_back(node->TabBar);
        for (ImGuiTabBar* tb : bars) {
            if (tb->CurrFrameVisible < g.FrameCount - 2) continue;
            for (ImGuiTabItem& tab : tb->Tabs) if (tab.ID == tb->SelectedTabId) visited.insert({ tb->ID, tab.ID });
            for (ImGuiTabItem& tab : tb->Tabs) {
                if (tab.Flags & (ImGuiTabItemFlags_Button | ImGuiTabItemFlags_Leading | ImGuiTabItemFlags_Trailing)) continue;
                if (tab.LastFrameVisible < g.FrameCount - 2) continue;
                if (!visited.insert({ tb->ID, tab.ID }).second) continue;
                name = ImGui::TabBarGetTabName(tb, &tab);
                ImGui::TabBarQueueFocus(tb, &tab);
                return true;
            }
        }
        return false;
    }

    void run_mode(const ModeButton& b) {
        mode = b.label;
        menus.clear();
        const int f0 = fields, m0 = menus_n, s0 = states;
        ctx->SetRef("//MainHost");
        ctx->ItemClick(b.label);
        ctx->Yield(10);
        scan_state("(initial)");
        std::set<std::pair<ImGuiID, ImGuiID>> visited;
        std::string name;
        auto walk_tabs = [&] {
            for (int k = 0; k < kMaxStatesPerMode; ++k) {
                if (!next_tab(visited, name)) break;
                ctx->Yield(6);
                scan_state(name);
            }
        };
        walk_tabs();
        // Library: each of ODE / DAE / MAP opens the system editor (tabs System / Parameters) with
        // "Edit" on its first row; Cancel closes it without saving. The library is the isolated copy.
        if (b.mode == AppModel::AppMode::Library) {
            static const char* const kLibWindows[][2] = {
                { "//###Editor", "ODE" }, { "//###LibraryDAE", "DAE" }, { "//###LibraryMAP", "MAP" } };
            for (const auto& lw : kLibWindows) {
                ctx->WindowFocus(lw[0], ImGuiTestOpFlags_NoError);
                ctx->Yield(4);
                auto find_button = [&](const char* text) -> ImGuiID {
                    ImGuiTestItemList items;
                    ctx->GatherItems(&items, lw[0], -1);
                    for (const ImGuiTestItemInfo& it : items)
                        if (std::strcmp(it.DebugLabel, text) == 0 && !(it.ItemFlags & ImGuiItemFlags_Disabled)) return it.ID;
                    return 0;
                };
                const ImGuiID edit_btn = find_button("Edit");
                if (!edit_btn) { std::printf("[sweep] Library: no %s system to edit\n", lw[1]); continue; }
                ctx->ItemClick(edit_btn, 0, ImGuiTestOpFlags_NoError);
                ctx->Yield(6);
                scan_state(std::string(lw[1]) + " editor");
                walk_tabs();
                // By path: the engine scrolls to it (below the fold its label is unknown to a gather).
                const std::string cancel_path = std::string(lw[0]) + "/Cancel";
                if (ctx->ItemExists(cancel_path.c_str())) ctx->ItemClick(cancel_path.c_str(), 0, ImGuiTestOpFlags_NoError);
                else std::printf("[sweep] Library: %s editor has no Cancel button\n", lw[1]);
                ctx->Yield(4);
                if (m_model && m_model->library_edit_mode != AppModel::LibraryEditMode::None)
                    std::printf("[sweep] Library: %s editor still open after Cancel\n", lw[1]);
            }
        }
        std::printf("[sweep] %s: TOTAL states %d, fields %d, context menus %d\n",
                    mode.c_str(), states - s0, fields - f0, menus_n - m0);
        for (const auto& [sig, n] : menus)
            std::printf("[sweep] %s:   menu x%d: %s\n", mode.c_str(), n, sig.c_str());
        std::fflush(stdout);
    }
};

} // namespace

void register_sweep_tests(ImGuiTestEngine* engine) {
    ImGuiTest* t = IM_REGISTER_TEST(engine, "sweep", "all_tabs");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppModel* m = app_model();
        IM_CHECK_NO_RET(m != nullptr);
        if (!m) return;
        m->hidden_tabs.clear();
        ctx->Yield(3);
        Sweep s{ ctx };
        s.m_model = m;
        for (const ModeButton& b : kModeButtons) {
            ctx->SetRef("//MainHost");
            if (!ctx->ItemExists(b.label)) { IM_ERRORF_NOHDR("mode button '%s' missing", b.label); return; }
            if (s.only && !std::strstr(b.label, s.only)) continue;
            s.run_mode(b);
            IM_CHECK_EQ((int)m->app_mode, (int)b.mode);
        }
        std::printf("[sweep] ALL: states %d, fields %d, context menus %d, ID conflicts %d, removed item %d\n",
                    s.states, s.fields, s.menus_n, s.conflicts, s.removed_item);
        std::printf("[sweep] EDIT: fields edited %d (numeric re-typed %d), shows other text %d, NOT restored %d, "
                    "skipped multiline %d, read-only %d\n", s.edited, s.edited_numeric, s.edit_changed,
                    s.edit_failed, s.edit_multiline, s.edit_readonly);
        std::fflush(stdout);
        IM_CHECK_EQ(s.conflicts, 0);
        IM_CHECK_EQ(s.removed_item, 0);
        IM_CHECK_EQ(s.edit_failed, 0);
    };
}

} // namespace ui_test
