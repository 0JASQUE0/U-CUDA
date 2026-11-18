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
    bool widths   = std::getenv("UCUDA_SWEEP_NOWIDTH") == nullptr;          // width handle checks
    bool sections = std::getenv("UCUDA_SWEEP_NOOPEN") == nullptr;           // open closed sections
    int actions = 0, action_failed = 0, width_checked = 0, width_failed = 0, width_fit_changed = 0;
    int no_menu = 0, sections_opened = 0, width_offscreen = 0;
    std::set<ImGuiID> opened_sections;
    std::map<std::string, std::vector<std::string>> no_menu_labels;         // mode -> fields without RMB menu
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

    struct MenuActions { std::string reset, restore; };   // full (maybe truncated) labels

    static std::string paren_value(const std::string& label) {
        const size_t a = label.find('('), b = label.rfind(')');
        if (a == std::string::npos || b == std::string::npos || b <= a) return {};   // truncated label
        return label.substr(a + 1, b - a - 1);
    }

    // One field: hover -> ID conflict, right-click -> menu summary, then edits, menu actions, width.
    void check_field(ImGuiID id, const char* label, const char* win, ImGuiID win_id, ImGuiID handle, bool inputable) {
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
        MenuActions acts;
        if (inputable) {
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
                    if (std::strncmp(pi.DebugLabel, "Reset to default", 16) == 0)     acts.reset = pi.DebugLabel;
                    if (std::strncmp(pi.DebugLabel, "Restore from library", 20) == 0) acts.restore = pi.DebugLabel;
                    if (pi.DebugLabel[0]) { if (!labels.empty()) labels += " | "; labels += pi.DebugLabel; }
                }
                ++menus[signature(labels)];
                ctx->PopupCloseAll();
                ctx->Yield(2);
            } else {
                ++no_menu;
                no_menu_labels[mode].push_back(label);
            }
            if (edit) check_edit(id, label, win, acts);
        }
        if (handle && widths) check_width(id, win_id, label, win);
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

    bool read_field(ImGuiID id, char* buf, size_t n) {
        buf[0] = 0;
        ctx->ItemReadAsString(id, buf, n);
        return !ctx->IsError();
    }

    // Put v0 back and check it is exact.
    void restore_value(ImGuiID id, const char* v0, const char* what, const char* label, const char* win) {
        char v[1024];
        ctx->ItemInputValue(id, v0);
        ctx->Yield(2);
        if (!read_field(id, v, sizeof v)) return;
        if (std::strcmp(v0, v) != 0) {
            ++edit_failed;
            std::printf("[sweep] %s: NOT RESTORED after %s \"%s\" -> \"%s\"  (\"%s\" in \"%s\")\n",
                        mode.c_str(), what, v0, v, label, win);
        }
    }

    // Run one context-menu item and compare the field with the value the item names.
    void check_action(ImGuiID id, const std::string& item, const char* v0, const char* label, const char* win) {
        ImGuiContext& g = *GImGui;
        const std::string want = paren_value(item);
        const std::string prefix = item.substr(0, std::min(item.size(), item.find('(')));
        ctx->ItemClick(id, ImGuiMouseButton_Right, ImGuiTestOpFlags_NoError | ImGuiTestOpFlags_NoFocusWindow);
        ctx->Yield(2);
        if (g.OpenPopupStack.Size == 0 || !g.OpenPopupStack.back().Window) return;
        ImGuiTestItemList pitems;
        ctx->GatherItems(&pitems, ImGuiTestRef(g.OpenPopupStack.back().Window->ID), 1);
        ImGuiID hit = 0;
        for (const ImGuiTestItemInfo& pi : pitems)
            if (std::strncmp(pi.DebugLabel, prefix.c_str(), prefix.size()) == 0) { hit = pi.ID; break; }
        if (!hit) { ctx->PopupCloseAll(); ctx->Yield(2); return; }
        ctx->ItemClick(hit, 0, ImGuiTestOpFlags_NoError);
        ctx->Yield(3);
        if (g.OpenPopupStack.Size > 0) { ctx->PopupCloseAll(); ctx->Yield(2); }
        char v[1024];
        if (!read_field(id, v, sizeof v)) return;
        ++actions;
        if (!want.empty()) {
            char* e1 = nullptr; char* e2 = nullptr;
            const double a = std::strtod(want.c_str(), &e1), b = std::strtod(v, &e2);
            const bool same = want == v || (e1 && !*e1 && e2 && !*e2 && a == b);
            if (!same) {
                ++action_failed;
                std::printf("[sweep] %s: ACTION \"%s\" -> field \"%s\", expected \"%s\"  (\"%s\" in \"%s\")\n",
                            mode.c_str(), prefix.c_str(), v, want.c_str(), label, win);
            }
        }
        restore_value(id, v0, prefix.c_str(), label, win);
    }

    // Field edit: read, type an equivalent value, read, put the original back, read; then the
    // context-menu actions Reset to default / Restore from library, each followed by the restore.
    void check_edit(ImGuiID id, const char* label, const char* win, const MenuActions& acts) {
        ImGuiTestItemInfo info = ctx->ItemInfo(id, ImGuiTestOpFlags_NoError);
        if (!info.ID) return;
        if (info.RectFull.GetHeight() > ImGui::GetFrameHeight() * 1.5f) { ++edit_multiline; return; }
        if (info.ItemFlags & ImGuiItemFlags_ReadOnly) { ++edit_readonly; return; }
        char v0[1024], v1[1024];
        if (!read_field(id, v0, sizeof v0)) return;
        ++edited;
        const std::string alt = equivalent(v0);
        if (!alt.empty()) {
            ++edited_numeric;
            ctx->ItemInputValue(id, alt.c_str());
            ctx->Yield(2);
            if (read_field(id, v1, sizeof v1) && alt != v1) {
                ++edit_changed;
                std::printf("[sweep] %s: EDIT typed \"%s\", field shows \"%s\"  (\"%s\" in \"%s\")\n",
                            mode.c_str(), alt.c_str(), v1, label, win);
            }
        }
        restore_value(id, v0, "edit", label, win);
        if (!acts.reset.empty())   check_action(id, acts.reset, v0, label, win);
        if (!acts.restore.empty()) check_action(id, acts.restore, v0, label, win);
        ctx->Yield(1);
    }

    // Width handle: drag +30 px, double-click (default), right-click (fit), double-click again.
    // The field must follow the drag and come back to its width; the stored widths must come back
    // to what they were (no stray AppConfig::field_widths entry).
    // The handle follows the field's right edge, so it is looked up again before every action.
    struct Handle { ImGuiID id = 0; ImRect rect, clipped; };
    Handle find_handle(ImGuiID field, ImGuiID win_id) {
        ctx->Yield(2);
        ImGuiTestItemList items;
        ctx->GatherItems(&items, ImGuiTestRef(win_id), -1);
        const ImGuiTestItemInfo* f = items.GetByID(field);
        Handle h;
        if (!f) return h;
        h.id = handle_of(*f, items);
        if (const ImGuiTestItemInfo* hi = h.id ? items.GetByID(h.id) : nullptr) { h.rect = hi->RectFull; h.clipped = hi->RectClipped; }
        else h.id = 0;
        return h;
    }

    // Mouse on the handle centre; true when ImGui really hovers it (an engine ItemClick would
    // abort the whole test otherwise).
    bool hover_handle(const Handle& h) {
        ctx->MouseMoveToPos(h.rect.GetCenter());
        ctx->Yield(2);
        return GImGui->HoveredIdPreviousFrame == h.id;
    }

    // Handle the user can reach: not clipped by its own window / table cell (RectClipped is the
    // rect after the clip at submission time) and inside the viewport.
    static bool reachable(const Handle& h, ImGuiID /*win_id*/) {
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        const ImRect view(vp->Pos, ImVec2(vp->Pos.x + vp->Size.x, vp->Pos.y + vp->Size.y));
        return h.clipped.GetWidth() >= h.rect.GetWidth() - 0.5f && view.Contains(h.rect.GetCenter());
    }

    // Width checks for one field. Whatever happens, the stored widths are put back afterwards,
    // so a failure never leaks a width into the fields checked next.
    void check_width(ImGuiID id, ImGuiID win_id, const char* label, const char* win) {
        if (!m_model) return;
        const std::map<std::string, float> saved = m_model->field_widths;
        check_width_steps(id, win_id, label, win);
        if (m_model->field_widths != saved) {
            m_model->field_widths = saved;
            m_model->field_widths_dirty = true;
            ctx->Yield(3);
        }
    }

    void check_width_steps(ImGuiID id, ImGuiID win_id, const char* label, const char* win) {
        auto width = [&] { return ctx->ItemInfo(id, ImGuiTestOpFlags_NoError).RectFull.GetWidth(); };
        auto fail = [&](const char* what, float now, float was) {
            ++width_failed;
            std::printf("[sweep] %s: WIDTH %s: %.0f (was %.0f), stored widths %d  (\"%s\" in \"%s\")\n",
                        mode.c_str(), what, now, was, (int)m_model->field_widths.size(), label, win);
        };
        // Next handle position; false = out of view (the column grew past the window edge):
        // that is layout, not a defect, so the field is skipped.
        auto next = [&](Handle& h, const char* when) -> bool {
            h = find_handle(id, win_id);
            if (!h.id) { fail((std::string("handle lost ") + when).c_str(), width(), 0.0f); return false; }
            if (!reachable(h, win_id)) { ++width_offscreen; return false; }
            if (!hover_handle(h)) { fail((std::string("handle not hoverable ") + when).c_str(), width(), 0.0f); return false; }
            return true;
        };
        Handle h = find_handle(id, win_id);
        if (!h.id) return;
        if (!reachable(h, win_id)) { ++width_offscreen; return; }
        const float w0 = width();
        const size_t n0 = m_model->field_widths.size();
        ++width_checked;
        // 1. Drag +30.
        if (!hover_handle(h)) { fail("handle not hoverable", w0, w0); return; }
        ctx->MouseDown(0);
        ctx->MouseMoveToPos(ImVec2(h.rect.GetCenter().x + 30.0f, h.rect.GetCenter().y));
        ctx->Yield(2);
        ctx->MouseUp(0);
        ctx->Yield(3);
        const float w1 = width();
        if (std::fabs(w1 - (w0 + 30.0f)) > 2.0f) fail("drag +30", w1, w0);
        // 2. Double-click: default width, no stored entry.
        if (!next(h, "after drag")) return;
        ctx->MouseDoubleClick(0);
        ctx->Yield(3);
        const float w2 = width();
        if (std::fabs(w2 - w0) > 1.0f || m_model->field_widths.size() != n0) fail("double-click", w2, w0);
        // 3. Right-click: fit to contents; no context menu may open.
        if (!next(h, "after reset")) return;
        ctx->MouseClick(1);
        ctx->Yield(3);
        if (GImGui->OpenPopupStack.Size > 0) {
            const ImGuiPopupData& pd = GImGui->OpenPopupStack.back();
            std::printf("[sweep]   popup \"%s\" opened (field 0x%08X, handle 0x%08X)\n",
                        pd.Window ? pd.Window->Name : "?", id, h.id);
            fail("right-click opened a popup", width(), w0);
            ctx->PopupCloseAll();
            ctx->Yield(2);
        }
        const float w3 = width();
        if (w3 != w0) ++width_fit_changed;
        // 4. Double-click again.
        if (!next(h, "after fit")) return;
        ctx->MouseDoubleClick(0);
        ctx->Yield(3);
        const float w4 = width();
        if (std::fabs(w4 - w0) > 1.0f || m_model->field_widths.size() != n0) fail("fit + double-click", w4, w0);
    }

    // Field rect -> its width handle: the narrow item at the field's right edge (widgets.cpp:
    // width_handle_rect, bb.Min.x = field.Max.x - 2).
    static ImGuiID handle_of(const ImGuiTestItemInfo& f, const ImGuiTestItemList& items) {
        // Exact handle width (widgets.cpp): from Max.x - 2 to Max.x + max(3, ItemSpacing.x - 1).
        const float hw = 2.0f + std::max(3.0f, ImGui::GetStyle().ItemSpacing.x - 1.0f);
        for (const ImGuiTestItemInfo& h : items) {
            if (h.ID == f.ID || std::fabs(h.RectFull.GetWidth() - hw) > 0.6f) continue;
            if (std::fabs(h.RectFull.Min.x - (f.RectFull.Max.x - 2.0f)) < 1.5f &&
                std::fabs(h.RectFull.Min.y - f.RectFull.Min.y) < 1.0f &&
                std::fabs(h.RectFull.GetHeight() - f.RectFull.GetHeight()) < 1.0f)
                return h.ID;
        }
        return 0;
    }

    // Open closed tree nodes / collapsing headers in a window (a few passes: opening one can
    // reveal more). Remembered, so each is opened once.
    int open_sections(ImGuiWindow* w, const ImRect& view) {
        int opened = 0;
        for (int pass = 0; pass < 3; ++pass) {
            ImGuiTestItemList items;
            ctx->GatherItems(&items, ImGuiTestRef(w->ID), -1);
            std::vector<ImGuiID> todo;
            for (const ImGuiTestItemInfo& it : items)
                if ((it.StatusFlags & ImGuiItemStatusFlags_Openable) && !(it.StatusFlags & ImGuiItemStatusFlags_Opened) &&
                    (it.StatusFlags & ImGuiItemStatusFlags_Visible) && !(it.ItemFlags & ImGuiItemFlags_Disabled) &&
                    view.Contains(it.RectClipped.GetCenter()) && !opened_sections.count(it.ID))
                    todo.push_back(it.ID);
            if (todo.empty()) break;
            // Open state straight into the window storage (what TreeNode / CollapsingHeader read):
            // the engine's ItemOpen aborts the test on nodes that refuse to open (Tree's node list).
            for (ImGuiID id : todo) {
                opened_sections.insert(id);
                w->StateStorage.SetInt(id, 1);
                ++opened;
            }
            ctx->Yield(3);
        }
        sections_opened += opened;
        return opened;
    }

    // All windows of the current frame, every scroll page, every new field.
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
        int fresh = 0, pages_total = 0, opened = 0;
        for (ImGuiWindow* w : wins) {
            float scroll_max = w->ScrollMax.y;
            const float step = std::max(60.0f, (w->InnerRect.GetHeight()) * 0.8f);
            int pages = 1;
            for (int p = 0; p < pages && fresh < kMaxFieldsPerState; ++p) {
                if (p > 0 || scroll_max > 0.0f) {
                    ImGui::SetScrollY(w, std::min(scroll_max, p * step));
                    ctx->Yield(3);
                }
                if (sections) opened += open_sections(w, view);
                scroll_max = w->ScrollMax.y;   // opened sections make the window longer
                if (scroll_max > 0.0f) pages = std::min(kMaxPagesPerWindow, 1 + (int)std::ceil(scroll_max / step));
                ++pages_total;
                ImGuiTestItemList items;
                ctx->GatherItems(&items, ImGuiTestRef(w->ID), -1);
                struct F { ImGuiID id; std::string label; ImGuiID handle; bool inputable; };
                std::vector<F> todo;
                for (const ImGuiTestItemInfo& it : items) {
                    if (!(it.StatusFlags & ImGuiItemStatusFlags_Visible) || (it.ItemFlags & ImGuiItemFlags_Disabled)) continue;
                    if (!view.Contains(it.RectClipped.GetCenter()) || it.RectClipped.GetWidth() < 4.0f ||
                        it.RectClipped.GetHeight() < 4.0f) continue;
                    if (checked.count(it.ID)) continue;
                    const bool inputable = (it.StatusFlags & ImGuiItemStatusFlags_Inputable) != 0;
                    const ImGuiID handle = handle_of(it, items);
                    if (!inputable && (!handle || it.RectFull.GetWidth() < 40.0f)) continue;   // combos: width only
                    todo.push_back({ it.ID, it.DebugLabel, handle, inputable });
                }
                for (const F& f : todo) {
                    if (fresh >= kMaxFieldsPerState) break;
                    checked.insert(f.id);
                    ++fresh;
                    check_field(f.id, f.label.c_str(), w->Name, w->ID, f.handle, f.inputable);
                }
            }
            if (w->ScrollMax.y > 0.0f) { ImGui::SetScrollY(w, 0.0f); ctx->Yield(1); }
        }
        std::printf("[sweep] %s | %s: %d window(s), %d page(s), %d new field(s), %d section(s) opened\n",
                    mode.c_str(), tag.c_str(), (int)wins.size(), pages_total, fresh, opened);
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
        std::printf("[sweep] MENU ACTIONS: run %d, wrong value %d; input fields without a context menu %d\n",
                    s.actions, s.action_failed, s.no_menu);
        for (const auto& [mode, labels] : s.no_menu_labels) {
            std::string list;
            for (const std::string& l : labels) { if (!list.empty()) list += ", "; list += l.empty() ? "(no label)" : l; }
            std::printf("[sweep]   no menu in %s (%d): %s\n", mode.c_str(), (int)labels.size(), list.c_str());
        }
        std::printf("[sweep] WIDTH: fields with a handle %d, failed %d, fit changed the width %d, "
                    "handle out of view (skipped) %d; sections opened %d\n",
                    s.width_checked, s.width_failed, s.width_fit_changed, s.width_offscreen, s.sections_opened);
        std::fflush(stdout);
        IM_CHECK_EQ(s.conflicts, 0);
        IM_CHECK_EQ(s.removed_item, 0);
        IM_CHECK_EQ(s.edit_failed, 0);
        IM_CHECK_EQ(s.action_failed, 0);
        IM_CHECK_EQ(s.width_failed, 0);
    };
}

} // namespace ui_test
