// Command-script test: one text file of UI actions, executed by the Test Engine as test "script/<file>".
// Every command and its arguments are documented in apps/desktop/ui_test/README.md.
#include "ui_test_host.h"

#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_te_engine.h"
#include "imgui_te_context.h"
#include "imgui_capture_tool.h"

#include "ucuda/application/app_model.h"
#include "ucuda/plotting/plot_axis.h"   // plot_probe_to_screen

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace ui_test {
namespace {

struct Line { int no; std::vector<std::string> args; };

std::vector<Line> g_lines;
std::string       g_script_name;

// Splits on blanks; "double quotes" group words. Inside quotes \n is a newline, \t a tab, and
// \x (any other x, e.g. \" or \\) is x.
std::vector<std::string> tokenize(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && std::isspace((unsigned char)s[i])) ++i;
        if (i >= s.size()) break;
        std::string tok;
        if (s[i] == '"') {
            ++i;
            while (i < s.size() && s[i] != '"') {
                if (s[i] == '\\' && i + 1 < s.size()) {
                    ++i;
                    if (s[i] == 'n') { tok += '\n'; ++i; continue; }
                    if (s[i] == 't') { tok += '\t'; ++i; continue; }
                }
                tok += s[i++];
            }
            ++i;
        } else {
            while (i < s.size() && !std::isspace((unsigned char)s[i])) tok += s[i++];
        }
        out.push_back(tok);
    }
    return out;
}

bool load_script(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string text;
    int no = 0;
    while (std::getline(f, text)) {
        ++no;
        if (!text.empty() && text.back() == '\r') text.pop_back();
        if (no == 1 && text.size() >= 3 && (unsigned char)text[0] == 0xEF) text.erase(0, 3);  // UTF-8 BOM
        size_t p = text.find_first_not_of(" \t");
        if (p == std::string::npos || text[p] == '#') continue;
        Line l{no, tokenize(text)};
        if (!l.args.empty()) g_lines.push_back(std::move(l));
    }
    return true;
}

// "0x1234ABCD" or "#0x1234ABCD" -> item ID, anything else -> path.
ImGuiTestRef make_ref(const std::string& s) {
    const char* p = s.c_str();
    if (*p == '#') ++p;
    if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X'))
        return ImGuiTestRef((ImGuiID)std::strtoul(p + 2, nullptr, 16));
    return ImGuiTestRef(s.c_str());
}

bool ieq(const char* a, const char* b) {
    for (; *a && *b; ++a, ++b)
        if (std::tolower((unsigned char)*a) != std::tolower((unsigned char)*b)) return false;
    return *a == *b;
}

// "Ctrl+Shift+Z", "Enter", "F5", "Escape" -> key chord; 0 if unknown.
ImGuiKeyChord parse_chord(const std::string& s) {
    ImGuiKeyChord chord = 0;
    size_t start = 0;
    while (start <= s.size()) {
        size_t end = s.find('+', start);
        if (end == start) end = s.find('+', start + 1);   // "Ctrl++" style: key named "+"
        std::string part = s.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if      (ieq(part.c_str(), "ctrl"))  chord |= ImGuiMod_Ctrl;
        else if (ieq(part.c_str(), "shift")) chord |= ImGuiMod_Shift;
        else if (ieq(part.c_str(), "alt"))   chord |= ImGuiMod_Alt;
        else if (ieq(part.c_str(), "super")) chord |= ImGuiMod_Super;
        else {
            ImGuiKey found = ImGuiKey_None;
            for (int k = ImGuiKey_NamedKey_BEGIN; k < ImGuiKey_NamedKey_END; ++k)
                if (ieq(ImGui::GetKeyName((ImGuiKey)k), part.c_str())) { found = (ImGuiKey)k; break; }
            if (found == ImGuiKey_None) return 0;
            chord |= found;
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return chord;
}

const char* kModeNames[] = { "Library", "Analysis", "Parametric", "Dft1D", "Basins", "FastSync",
                             "Custom", "Settings", "Order", "Network", "Neural", "Tree" };
static_assert(sizeof(kModeNames) / sizeof(kModeNames[0]) == AppModel::kAppModeCount,
              "kModeNames must follow AppModel::AppMode");

std::string out_path(const std::string& file) {
    std::filesystem::path p(file);
    if (p.is_relative()) p = std::filesystem::path(options().out_dir) / p;
    std::error_code ec;
    std::filesystem::create_directories(p.parent_path(), ec);
    return p.string();
}

void print_item(const ImGuiTestItemInfo& it) {
    std::string flags;
    const ImGuiItemStatusFlags f = it.StatusFlags;
    if (f & ImGuiItemStatusFlags_Checkable) flags += (f & ImGuiItemStatusFlags_Checked) ? " [x]" : " [ ]";
    if (f & ImGuiItemStatusFlags_Openable)  flags += (f & ImGuiItemStatusFlags_Opened) ? " open" : " closed";
    if (f & ImGuiItemStatusFlags_Inputable) flags += " input";
    if (it.ItemFlags & ImGuiItemFlags_Disabled) flags += " disabled";
    if (!(f & ImGuiItemStatusFlags_Visible)) flags += " hidden";
    std::printf("[ui]   0x%08X d%d \"%s\" @(%.0f,%.0f %.0fx%.0f)%s  win=\"%s\"\n",
                it.ID, (int)it.Depth, it.DebugLabel, it.RectFull.Min.x, it.RectFull.Min.y,
                it.RectFull.GetWidth(), it.RectFull.GetHeight(), flags.c_str(),
                it.Window ? it.Window->Name : "");
}

void print_windows(bool all) {
    ImGuiContext& g = *GImGui;
    for (ImGuiWindow* w : g.Windows) {
        if (!w->WasActive || w->Hidden) continue;
        const bool child = (w->Flags & ImGuiWindowFlags_ChildWindow) != 0 && !w->DockIsActive;
        if (child && !all) continue;
        std::printf("[ui]   0x%08X \"%s\" @(%.0f,%.0f %.0fx%.0f)%s%s%s\n", w->ID, w->Name, w->Pos.x, w->Pos.y,
                    w->Size.x, w->Size.y, child ? " child" : "", w->DockIsActive ? " docked" : "",
                    (w->Flags & ImGuiWindowFlags_Popup) ? " popup" : "");
    }
}

// ---- Label search: references "SCOPE>>LABEL" -------------------------------------------------
// ID paths miss items under PushID(int), tab items (their IDs hang off the tab bar) and labels
// that carry state ("Bif N=100###cfg"). A label reference searches the items under SCOPE at any
// depth instead: LABEL matches the full label ("Run##bif") or its visible part ("Run"), and a
// trailing '*' makes it a prefix. Items clipped out of view have no label yet, so when nothing
// matches, SCOPE's window and its scrollable child windows are scrolled page by page.

bool label_matches(const char* item_label, const std::string& want) {
    const bool prefix = !want.empty() && want.back() == '*';
    const std::string w = prefix ? want.substr(0, want.size() - 1) : want;
    const std::string full = item_label ? item_label : "";
    const std::string vis = full.substr(0, full.find("##"));
    if (prefix) return full.compare(0, w.size(), w) == 0 || vis.compare(0, w.size(), w) == 0;
    return full == w || (!vis.empty() && vis == w);
}

// Child windows (BeginChild) start an ID stack of their own, so a gather under a window misses
// their items: when the scope is a window, its active descendant windows are searched as well.
std::vector<ImGuiID> scope_ids(ImGuiTestContext* ctx, ImGuiTestRef scope) {
    std::vector<ImGuiID> scopes{ ctx->GetID(scope) };
    if (ImGuiWindow* root = ctx->GetWindowByRef(scope))
        for (ImGuiWindow* w : GImGui->Windows) {
            if (w == root || !w->WasActive || w->Hidden || w->DockIsActive) continue;
            for (ImGuiWindow* p = w->ParentWindow; p; p = p->ParentWindow)
                if (p == root) { scopes.push_back(w->ID); break; }
        }
    return scopes;
}

// "LABEL@N" picks the N-th match (1-based, in drawing order) among same-labelled items, e.g. the
// type combo of the second projection: "//Controls>>##ptype@2". Without @N: the first visible match.
ImGuiID gather_match(ImGuiTestContext* ctx, ImGuiTestRef scope, const std::string& want_in) {
    std::string want = want_in;
    int nth = 0;
    const size_t at = want.rfind('@');
    if (at != std::string::npos && at + 1 < want.size() &&
        want.find_first_not_of("0123456789", at + 1) == std::string::npos) {
        nth = std::atoi(want.c_str() + at + 1);
        want.resize(at);
    }
    ImGuiID first = 0;
    int count = 0;
    for (ImGuiID s : scope_ids(ctx, scope)) {
        ImGuiTestItemList items;
        ctx->GatherItems(&items, ImGuiTestRef(s), -1);
        for (int i = 0; i < items.GetSize(); ++i) {
            const ImGuiTestItemInfo* it = items.GetByIndex(i);
            if (!label_matches(it->DebugLabel, want)) continue;
            if (nth > 0) { if (++count == nth) return it->ID; continue; }
            if (it->StatusFlags & ImGuiItemStatusFlags_Visible) return it->ID;
            if (!first) first = it->ID;
        }
    }
    return first;
}

std::vector<ImGuiID> root_windows() {
    std::vector<ImGuiID> out;
    for (ImGuiWindow* w : GImGui->Windows)
        // Every active window, child windows included: docked windows and dock-node hosts (their
        // tab bars) carry ImGuiWindowFlags_ChildWindow but have ID stacks of their own.
        if (w->WasActive && !w->Hidden) out.push_back(w->ID);
    return out;
}

// Tab items (BeginTabItem) report no label to the engine: match the names ImGui keeps for the
// tab bars drawn in the last frames. The scope is not checked, so tab names should be distinct.
ImGuiID find_tab_by_label(const std::string& want) {
    ImGuiContext& g = *GImGui;
    for (int i = 0; i < g.TabBars.GetMapSize(); ++i) {
        ImGuiTabBar* tb = g.TabBars.TryGetMapData(i);
        if (!tb || tb->PrevFrameVisible < g.FrameCount - 2) continue;
        for (ImGuiTabItem& t : tb->Tabs)
            if (label_matches(ImGui::TabBarGetTabName(tb, &t), want)) return t.ID;
    }
    return 0;
}

ImGuiID find_by_label(ImGuiTestContext* ctx, const std::string& scope_s, const std::string& want,
                      bool scroll = true) {
    if (scope_s.rfind("*>>", 0) == 0) {   // scope = window found by name: "*>>Tree Controls*>>Run"
        const std::string name = scope_s.substr(3);
        for (ImGuiWindow* w : GImGui->Windows)
            if (w->WasActive && !w->Hidden && label_matches(w->Name, name)) {
                char hex[16];
                std::snprintf(hex, sizeof(hex), "0x%08X", w->ID);
                return find_by_label(ctx, hex, want, scroll);
            }
        return 0;
    }
    if (scope_s == "*") {   // every visible window: first without scrolling, then with
        for (int pass = 0; pass < (scroll ? 2 : 1); ++pass)
            for (ImGuiID w : root_windows()) {
                char hex[16];
                std::snprintf(hex, sizeof(hex), "0x%08X", w);
                if (ImGuiID id = find_by_label(ctx, hex, want, pass == 1)) return id;
            }
        // Dock tabs are not reported with labels: match the docked window's name, and clicking
        // its tab ID selects it as the user would ("*>>Bifurcation 2D").
        for (ImGuiWindow* w : GImGui->Windows)
            if (w->DockNode && w->TabId && label_matches(w->Name, want)) return w->TabId;
        return 0;
    }
    const ImGuiTestRef scope = scope_s.empty() ? ImGuiTestRef(ctx->RefID) : make_ref(scope_s);
    if (ImGuiID id = gather_match(ctx, scope, want)) return id;
    if (ImGuiID id = find_tab_by_label(want)) return id;
    if (!scroll) return 0;
    ImGuiWindow* root = ctx->GetWindowByRef(scope);
    if (!root) return 0;
    std::vector<ImGuiWindow*> scrollable;
    for (ImGuiWindow* w : GImGui->Windows) {
        if (!w->WasActive || w->Hidden || w->ScrollMax.y <= 0.0f) continue;
        for (ImGuiWindow* p = w; p; p = p->ParentWindow)
            if (p == root) { scrollable.push_back(w); break; }
    }
    for (ImGuiWindow* w : scrollable) {
        const float back = w->Scroll.y, page = ImMax(40.0f, w->InnerRect.GetHeight() * 0.8f);
        // At most 40 pages: a clipped list can be millions of pixels tall (float scroll stops being exact).
        for (float y = 0.0f; y < page * 40.0f; y += page) {
            ctx->ScrollToY(ImGuiTestRef(w->ID), ImMin(y, w->ScrollMax.y));
            ctx->Yield(2);
            if (ImGuiID id = gather_match(ctx, scope, want)) return id;
            if (y >= w->ScrollMax.y) break;
        }
        ctx->ScrollToY(ImGuiTestRef(w->ID), back);
    }
    return 0;
}

// Modifier-only chord ("Ctrl", "Shift+Alt") held around a mouse action; 0 = none.
ImGuiKeyChord parse_chord(const std::string& s);
ImGuiKeyChord parse_mods(const std::string& s) {
    const ImGuiKeyChord c = parse_chord(s);
    return (c & ~ImGuiMod_Mask_) ? 0 : c;
}

ImGuiMouseButton parse_button(const std::string& s) {
    if (ieq(s.c_str(), "right")  || s == "1") return ImGuiMouseButton_Right;
    if (ieq(s.c_str(), "middle") || s == "2") return ImGuiMouseButton_Middle;
    return ImGuiMouseButton_Left;
}

#define SCRIPT_FAIL(...) do { IM_ERRORF_NOHDR(__VA_ARGS__); return; } while (0)

void run_line(ImGuiTestContext* ctx, const Line& l) {
    const auto& a = l.args;
    const std::string& cmd = a[0];
    const int n = (int)a.size() - 1;
    auto need = [&](int k) {
        if (n < k) { IM_ERRORF_NOHDR("line %d: '%s' needs %d argument(s)", l.no, cmd.c_str(), k); return false; }
        return true;
    };
    std::printf("[ui] %d: %s", l.no, cmd.c_str());
    for (int i = 1; i <= n; ++i) std::printf(" \"%s\"", a[i].c_str());
    std::printf("\n");
    std::fflush(stdout);

    // a[k] as a reference: an ID, an ID path, or "SCOPE>>LABEL" (label search, resolved to an ID).
    auto R = [&](int k) -> ImGuiTestRef {
        const std::string& s = a[k];
        const size_t p = s.rfind(">>");
        if (p == std::string::npos) return make_ref(s);
        const ImGuiID id = find_by_label(ctx, s.substr(0, p), s.substr(p + 2));
        if (!id) IM_ERRORF_NOHDR("line %d: no item labelled \"%s\" under '%s'", l.no,
                                 s.substr(p + 2).c_str(), s.substr(0, p).c_str());
        return ImGuiTestRef(id);
    };
    auto ok = [&] { return !ctx->IsError(); };
    // a[k] as a window: a path ("//Controls") or "*>>NAME" matching the window's name like a label
    // ("*>>Projection 3": the ##id suffix of app windows changes when siblings close).
    auto W = [&](int k) -> ImGuiTestRef {
        const std::string& s = a[k];
        if (s.rfind("*>>", 0) != 0) return make_ref(s);
        for (ImGuiWindow* w : GImGui->Windows)
            if (w->WasActive && !w->Hidden && label_matches(w->Name, s.substr(3))) return ImGuiTestRef(w->ID);
        // A docked window behind another tab of its node is not active: still a valid target.
        for (ImGuiWindow* w : GImGui->Windows)
            if (w->DockNode && label_matches(w->Name, s.substr(3))) return ImGuiTestRef(w->ID);
        IM_ERRORF_NOHDR("line %d: no window named \"%s\"", l.no, s.substr(3).c_str());
        return ImGuiTestRef();
    };
    // Optional modifier argument a[k] ("Ctrl", "Shift+Alt") held down around fn.
    auto with_mods = [&](int k, const std::function<void()>& fn) {
        ImGuiKeyChord m = 0;
        if (n >= k) {
            m = parse_mods(a[k]);
            if (!m) { IM_ERRORF_NOHDR("line %d: '%s' is not a modifier (Ctrl, Shift, Alt, Super)", l.no, a[k].c_str()); return; }
        }
        if (m) ctx->KeyDown(m);
        fn();
        if (m) ctx->KeyUp(m);
    };
    // Screen point inside item r from a[k], a[k+1]: fractions of its rectangle (0..1, from the
    // top-left corner) or, for plot_*, data coordinates of a 2D plot or heatmap (plot_axis.h probe).
    auto point = [&](ImGuiTestRef r, int k, bool data, ImVec2* out) -> bool {
        ImGuiTestItemInfo it = ctx->ItemInfo(r);
        if (!it.ID) return false;
        if (!(it.StatusFlags & ImGuiItemStatusFlags_Visible) || it.RectClipped.GetWidth() <= 0.0f) {
            ctx->ScrollToItemY(r);
            ctx->Yield(2);
            it = ctx->ItemInfo(r);
        }
        const double u = std::atof(a[k].c_str()), v = std::atof(a[k + 1].c_str());
        if (data) {
            ctx->Yield(1);   // the probe holds the view of the last drawn frame
            if (!plot_probe_to_screen(it.ID, u, v, out)) {
                IM_ERRORF_NOHDR("line %d: '%s' is not a 2D plot or heatmap area (see `plots`)", l.no, a[1].c_str());
                return false;
            }
        } else {
            *out = ImVec2(it.RectFull.Min.x + (float)u * it.RectFull.GetWidth(),
                          it.RectFull.Min.y + (float)v * it.RectFull.GetHeight());
        }
        if (!it.RectClipped.Contains(*out)) {
            IM_ERRORF_NOHDR("line %d: point (%s, %s) is outside the visible part of '%s' (screen %.0f,%.0f)",
                            l.no, a[k].c_str(), a[k + 1].c_str(), a[1].c_str(), out->x, out->y);
            return false;
        }
        return true;
    };

    if (cmd == "ref")            { if (need(1)) { auto r = R(1); if (ok()) ctx->SetRef(r); } }
    else if (cmd == "click" || cmd == "rclick" || cmd == "mclick" || cmd == "dclick") {
        if (!need(1)) return;
        auto r = R(1);
        if (!ok()) return;
        with_mods(2, [&] {
            if (cmd == "dclick") ctx->ItemDoubleClick(r);
            else ctx->ItemClick(r, cmd == "rclick" ? ImGuiMouseButton_Right
                                 : cmd == "mclick" ? ImGuiMouseButton_Middle : ImGuiMouseButton_Left);
        });
    }
    else if (cmd == "hover")     { if (need(1)) { auto r = R(1); if (ok()) ctx->MouseMove(r); } }
    else if (cmd == "check")     { if (need(1)) { auto r = R(1); if (ok()) ctx->ItemCheck(r); } }
    else if (cmd == "uncheck")   { if (need(1)) { auto r = R(1); if (ok()) ctx->ItemUncheck(r); } }
    else if (cmd == "open")      { if (need(1)) { auto r = R(1); if (ok()) ctx->ItemOpen(r); } }
    else if (cmd == "close")     { if (need(1)) { auto r = R(1); if (ok()) ctx->ItemClose(r); } }
    else if (cmd == "input")     { if (need(2)) { auto r = R(1); if (ok()) ctx->ItemInputValue(r, a[2].c_str()); } }
    else if (cmd == "set_text") {
        // Replace the whole text of a field without pressing Enter (multiline editors: Enter is a newline).
        if (!need(2)) return;
        auto r = R(1);
        if (!ok()) return;
        ctx->ItemClick(r);
        ctx->KeyPress(ImGuiMod_Ctrl | ImGuiKey_A);
        ctx->KeyPress(ImGuiKey_Delete);
        ctx->KeyChars(a[2].c_str());
        // Leave the field as a click elsewhere would: the text stays (Escape would revert it).
        ImGui::ClearActiveID();
        ctx->Yield(2);
    }
    else if (cmd == "type")      { if (need(1)) ctx->KeyChars(a[1].c_str()); }
    else if (cmd == "menu")      { if (need(1)) ctx->MenuClick(make_ref(a[1])); }
    else if (cmd == "combo")     { if (need(1)) ctx->ComboClick(make_ref(a[1])); }
    else if (cmd == "select") {
        // Open combo R and pick the entry labelled ITEM (label search inside the popup, so it works
        // for entries under PushID and for long lists that need scrolling).
        if (!need(2)) return;
        auto r = R(1);
        if (!ok()) return;
        ctx->ItemClick(r);
        ctx->Yield(2);
        ImGuiID id = find_by_label(ctx, "//$FOCUSED", a[2], false);
        // Variable and parameter names drawn as formulas are Selectables labelled "##vs_<name>".
        if (!id) id = find_by_label(ctx, "//$FOCUSED", "##vs_" + a[2], false);
        if (!id) id = find_by_label(ctx, "//$FOCUSED", a[2]);
        if (!id) SCRIPT_FAIL("line %d: the popup of '%s' has no entry \"%s\"", l.no, a[1].c_str(), a[2].c_str());
        ctx->ItemClick(id);
    }
    else if (cmd == "tab_close") { if (need(1)) { auto r = R(1); if (ok()) ctx->TabClose(r); } }
    else if (cmd == "focus")     { if (need(1)) { auto w = W(1); if (ok()) ctx->WindowFocus(w); } }
    else if (cmd == "window_close")    { if (need(1)) { auto w = W(1); if (ok()) ctx->WindowClose(w); } }
    else if (cmd == "window_collapse") { if (need(1)) { auto w = W(1); if (ok()) ctx->WindowCollapse(w, true); } }
    else if (cmd == "window_expand")   { if (need(1)) { auto w = W(1); if (ok()) ctx->WindowCollapse(w, false); } }
    else if (cmd == "dock") {
        if (!need(2)) return;
        ImGuiDir dir = ImGuiDir_None;
        if (n >= 3) {
            if      (ieq(a[3].c_str(), "left"))  dir = ImGuiDir_Left;
            else if (ieq(a[3].c_str(), "right")) dir = ImGuiDir_Right;
            else if (ieq(a[3].c_str(), "up"))    dir = ImGuiDir_Up;
            else if (ieq(a[3].c_str(), "down"))  dir = ImGuiDir_Down;
            else SCRIPT_FAIL("line %d: dock direction is left, right, up or down", l.no);
        }
        auto src = W(1);
        auto dst = ok() ? W(2) : ImGuiTestRef();
        if (ok()) ctx->DockInto(src, dst, dir);
    }
    else if (cmd == "click_at" || cmd == "dclick_at" || cmd == "hover_at" ||
             cmd == "plot_click" || cmd == "plot_dclick" || cmd == "plot_hover") {
        // click_at R FX FY [button] [mods]   plot_click R X Y [button] [mods]
        if (!need(3)) return;
        auto r = R(1);
        ImVec2 p;
        if (!ok() || !point(r, 2, cmd.rfind("plot_", 0) == 0, &p)) { if (ok()) SCRIPT_FAIL("line %d: '%s' not found", l.no, a[1].c_str()); return; }
        const bool hover = cmd == "hover_at" || cmd == "plot_hover";
        const bool dbl = cmd == "dclick_at" || cmd == "plot_dclick";
        const ImGuiMouseButton b = n >= 4 ? parse_button(a[4]) : ImGuiMouseButton_Left;
        with_mods(5, [&] {
            ctx->MouseMoveToPos(p);
            if (hover) return;
            if (dbl) ctx->MouseDoubleClick(b); else ctx->MouseClick(b);
        });
    }
    else if (cmd == "drag_at" || cmd == "plot_drag") {
        // drag_at R FX0 FY0 FX1 FY1 [button] [mods]   plot_drag R X0 Y0 X1 Y1 [button] [mods]
        if (!need(5)) return;
        auto r = R(1);
        const bool data = cmd == "plot_drag";
        ImVec2 p0, p1;
        if (!ok() || !point(r, 2, data, &p0) || !point(r, 4, data, &p1)) { if (ok()) SCRIPT_FAIL("line %d: '%s' not found", l.no, a[1].c_str()); return; }
        const ImGuiMouseButton b = n >= 6 ? parse_button(a[6]) : ImGuiMouseButton_Left;
        with_mods(7, [&] {
            ctx->MouseMoveToPos(p0);
            ctx->MouseDown(b);
            for (int s = 1; s <= 4; ++s)   // several frames of motion: drag thresholds and live updates
                ctx->MouseMoveToPos(ImVec2(p0.x + (p1.x - p0.x) * s / 4.0f, p0.y + (p1.y - p0.y) * s / 4.0f));
            ctx->MouseUp(b);
        });
    }
    else if (cmd == "wheel_at" || cmd == "plot_wheel") {
        // wheel_at R FX FY DY [mods]   plot_wheel R X Y DY [mods]   (DY > 0: wheel up)
        if (!need(4)) return;
        auto r = R(1);
        ImVec2 p;
        if (!ok() || !point(r, 2, cmd == "plot_wheel", &p)) { if (ok()) SCRIPT_FAIL("line %d: '%s' not found", l.no, a[1].c_str()); return; }
        with_mods(5, [&] {
            ctx->MouseMoveToPos(p);
            ctx->MouseWheelY((float)std::atof(a[4].c_str()));
        });
    }
    else if (cmd == "plots") {
        // Plot areas under W (default: every visible window): 2D plots, heatmaps, 3D plots.
        std::vector<ImGuiID> roots;
        if (n >= 1) { auto w = W(1); if (!ok()) return; roots = scope_ids(ctx, w); }
        else roots = root_windows();
        int count = 0;
        std::vector<ImGuiID> seen;
        for (ImGuiID root : roots) {
            ImGuiTestItemList items;
            ctx->GatherItems(&items, ImGuiTestRef(root), -1);
            for (int i = 0; i < items.GetSize(); ++i) {
                const ImGuiTestItemInfo* it = items.GetByIndex(i);
                const char* lb = it->DebugLabel;
                if (std::strncmp(lb, "##plot_", 7) && std::strncmp(lb, "##hm_plot_", 10) && std::strncmp(lb, "##plot3d_", 9))
                    continue;
                if (std::find(seen.begin(), seen.end(), it->ID) != seen.end()) continue;
                seen.push_back(it->ID);
                ++count;
                print_item(*it);
            }
        }
        std::printf("[ui]   %d plot area(s)\n", count);
    }
    else if (cmd == "pace") {
        // pace F: mouse movement and typing F times faster than the engine defaults (600 px/s,
        // 20 chars/s); affects -Speed normal/cinematic only, fast mode teleports anyway.
        if (!need(1)) return;
        const float f = (float)std::atof(a[1].c_str());
        if (!(f > 0.0f)) SCRIPT_FAIL("line %d: pace needs a positive factor", l.no);
        ctx->EngineIO->MouseSpeed  = 600.0f * f;
        ctx->EngineIO->TypingSpeed = 20.0f * f;
    }
    else if (cmd == "key_down" || cmd == "key_up") {
        if (!need(1)) return;
        const ImGuiKeyChord c = parse_chord(a[1]);
        if (!c) SCRIPT_FAIL("line %d: unknown key '%s'", l.no, a[1].c_str());
        if (cmd == "key_down") ctx->KeyDown(c); else ctx->KeyUp(c);
    }
    else if (cmd == "file_dialog") {
        // Answer for the next native file dialog: a path (relative = under --ui-out) or "cancel".
        if (!need(1)) return;
        queue_file_dialog(ieq(a[1].c_str(), "cancel") ? std::string() : out_path(a[1]));
    }
    else if (cmd == "expect_no_overlap") {
        // Floating windows (not docked, not MainHost, not popups/tooltips) must not cover each other.
        std::vector<ImGuiWindow*> ws;
        for (ImGuiWindow* w : GImGui->Windows) {
            if (!w->WasActive || w->Hidden || w->DockIsActive || std::strcmp(w->Name, "MainHost") == 0) continue;
            if (w->Flags & (ImGuiWindowFlags_ChildWindow | ImGuiWindowFlags_Popup | ImGuiWindowFlags_Tooltip)) continue;
            ws.push_back(w);
        }
        int bad = 0;
        for (size_t i = 0; i < ws.size(); ++i)
            for (size_t j = i + 1; j < ws.size(); ++j) {
                ImRect r = ws[i]->Rect();
                r.ClipWithFull(ws[j]->Rect());
                if (r.GetWidth() > 1.0f && r.GetHeight() > 1.0f) {
                    std::printf("[ui]   overlap: \"%s\" and \"%s\" (%.0fx%.0f)\n", ws[i]->Name, ws[j]->Name,
                                r.GetWidth(), r.GetHeight());
                    ++bad;
                }
            }
        if (bad) SCRIPT_FAIL("line %d: %d pair(s) of floating windows overlap (use Layout)", l.no, bad);
        std::printf("[ui]   %d floating window(s), no overlap\n", (int)ws.size());
    }
    else if (cmd == "expect_file") {
        // A file written by the app (export, save): exists and is not empty. Relative = under --ui-out.
        if (!need(1)) return;
        std::error_code ec;
        const std::string p = out_path(a[1]);
        const auto size = std::filesystem::file_size(p, ec);
        if (ec || size == 0) SCRIPT_FAIL("line %d: file %s is missing or empty", l.no, p.c_str());
        std::printf("[ui]   file %s: %llu bytes\n", p.c_str(), (unsigned long long)size);
    }
    else if (cmd == "clipboard" || cmd == "expect_clipboard") {
        const char* c = ImGui::GetClipboardText();
        std::printf("[ui]   clipboard = \"%s\"\n", c ? c : "");
        if (cmd == "expect_clipboard" && need(1) && a[1] != (c ? c : ""))
            SCRIPT_FAIL("line %d: clipboard is \"%s\", expected \"%s\"", l.no, c ? c : "", a[1].c_str());
    }
    else if (cmd == "expect_label") {
        if (!need(2)) return;
        auto r = R(1);
        if (!ok()) return;
        ImGuiTestItemInfo it = ctx->ItemInfo(r);
        if (!it.ID) return;
        if (!label_matches(it.DebugLabel, a[2]))
            SCRIPT_FAIL("line %d: '%s' is labelled \"%s\", expected \"%s\"", l.no, a[1].c_str(), it.DebugLabel, a[2].c_str());
    }
    else if (cmd == "expect_enabled" || cmd == "expect_disabled") {
        if (!need(1)) return;
        auto r = R(1);
        if (!ok()) return;
        ImGuiTestItemInfo it = ctx->ItemInfo(r);
        if (!it.ID) return;
        const bool disabled = (it.ItemFlags & ImGuiItemFlags_Disabled) != 0;
        if (disabled != (cmd == "expect_disabled"))
            SCRIPT_FAIL("line %d: '%s' is %s", l.no, a[1].c_str(), disabled ? "disabled" : "enabled");
    }
    else if (cmd == "popup_close")     { ctx->PopupCloseOne(); }
    else if (cmd == "popup_close_all") { ctx->PopupCloseAll(); }
    else if (cmd == "mouse_void") {
        // MouseMoveToVoid() needs a spot with no window, and MainHost covers the whole viewport:
        // move to its bottom-right corner instead (a real mouse move re-enables hover after Escape).
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ctx->MouseMoveToPos(ImVec2(vp->Pos.x + vp->Size.x - 3.0f, vp->Pos.y + vp->Size.y - 3.0f));
    }
    else if (cmd == "key") {
        if (!need(1)) return;
        const ImGuiKeyChord c = parse_chord(a[1]);
        if (!c) SCRIPT_FAIL("line %d: unknown key '%s'", l.no, a[1].c_str());
        ctx->KeyPress(c, n >= 2 ? std::atoi(a[2].c_str()) : 1);
    }
    else if (cmd == "drag") {
        if (!need(3)) return;
        auto r = R(1);
        if (ok()) ctx->ItemDragWithDelta(r, ImVec2((float)std::atof(a[2].c_str()), (float)std::atof(a[3].c_str())));
    }
    else if (cmd == "scroll_y")  { if (need(2)) { auto w = W(1); if (ok()) ctx->ScrollToY(w, (float)std::atof(a[2].c_str())); } }
    else if (cmd == "scroll_to") { if (need(1)) { auto r = R(1); if (ok()) ctx->ScrollToItemY(r); } }
    else if (cmd == "resize") {
        if (!need(3)) return;
        auto w = W(1);
        if (ok()) ctx->WindowResize(w, ImVec2((float)std::atof(a[2].c_str()), (float)std::atof(a[3].c_str())));
    }
    else if (cmd == "yield")     { ctx->Yield(n >= 1 ? std::atoi(a[1].c_str()) : 1); }
    else if (cmd == "sleep")     { if (need(1)) ctx->SleepNoSkip((float)std::atof(a[1].c_str()), 1.0f / 60.0f); }
    else if (cmd == "wait_exists" || cmd == "wait_missing") {
        if (!need(1)) return;
        const bool want = cmd == "wait_exists";
        const double timeout = n >= 2 ? std::atof(a[2].c_str()) : 10.0;
        const double t0 = ImGui::GetTime();
        const size_t lp = a[1].rfind(">>");
        auto exists = [&] {
            return lp == std::string::npos ? ctx->ItemExists(make_ref(a[1]))
                                           : find_by_label(ctx, a[1].substr(0, lp), a[1].substr(lp + 2), false) != 0;
        };
        while (exists() != want) {
            if (ImGui::GetTime() - t0 > timeout)
                SCRIPT_FAIL("line %d: '%s' still %s after %.1f s", l.no, a[1].c_str(),
                            want ? "missing" : "present", timeout);
            ctx->SleepNoSkip(0.1f, 1.0f / 60.0f);
        }
    }
    else if (cmd == "wait_idle") {
        // Until no GPU/CPU session of the app is in flight (Run finished).
        AppModel* m = app_model();
        const double timeout = n >= 1 ? std::atof(a[1].c_str()) : 120.0;
        const double t0 = ImGui::GetTime();
        ctx->Yield(2);
        // The app's own list (all tabs, Tree and basins portraits included); a copy here missed several.
        // Idle must hold for 0.7 s: Tree recomputes the cursor's nodes 0.5 s after the cursor stops
        // (debounce), and runs its queued nodes one after another.
        double quiet_since = -1.0;
        while (m) {
            const bool busy = m->any_gpu_job_in_flight() || !m->tree_queue().empty();
            const double now = ImGui::GetTime();
            if (busy) quiet_since = -1.0;
            else if (quiet_since < 0.0) quiet_since = now;
            else if (now - quiet_since >= 0.7) break;
            if (now - t0 > timeout) SCRIPT_FAIL("line %d: still computing after %.0f s", l.no, timeout);
            ctx->SleepNoSkip(0.1f, 1.0f / 60.0f);
        }
    }
    else if (cmd == "expect_exists" || cmd == "expect_missing") {
        if (!need(1)) return;
        const bool want = cmd == "expect_exists";
        const size_t lp = a[1].rfind(">>");
        const bool has = lp == std::string::npos
            ? ctx->ItemExists(make_ref(a[1]))
            : find_by_label(ctx, a[1].substr(0, lp), a[1].substr(lp + 2)) != 0;
        if (has != want)
            SCRIPT_FAIL("line %d: expected '%s' to be %s", l.no, a[1].c_str(), want ? "present" : "absent");
    }
    else if (cmd == "expect_checked" || cmd == "expect_unchecked") {
        if (!need(1)) return;
        const bool want = cmd == "expect_checked";
        auto r = R(1);
        if (ok() && ctx->ItemIsChecked(r) != want)
            SCRIPT_FAIL("line %d: expected '%s' %s", l.no, a[1].c_str(), want ? "checked" : "unchecked");
    }
    else if (cmd == "expect_open" || cmd == "expect_closed") {
        if (!need(1)) return;
        const bool want = cmd == "expect_open";
        auto r = R(1);
        if (ok() && ctx->ItemIsOpened(r) != want)
            SCRIPT_FAIL("line %d: expected '%s' %s", l.no, a[1].c_str(), want ? "open" : "closed");
    }
    else if (cmd == "read" || cmd == "expect_value") {
        if (!need(cmd == "read" ? 1 : 2)) return;
        char buf[4096] = {};
        auto r = R(1);
        if (!ok()) return;
        const ImGuiTestItemInfo info = ctx->ItemInfo(r);
        if (!info.ID) return;
        if (info.RectFull.GetHeight() > ImGui::GetFrameHeight() * 1.5f) {
            // Multiline editor: the engine's read ends with Escape, which reverts such a field.
            // Select all, copy, and leave the field as a click elsewhere would.
            ctx->ItemClick(r);
            ctx->KeyPress(ImGuiMod_Ctrl | ImGuiKey_A);
            ctx->KeyPress(ImGuiMod_Ctrl | ImGuiKey_C);
            ImGui::ClearActiveID();
            ctx->Yield(2);
            const char* c = ImGui::GetClipboardText();
            ImStrncpy(buf, c ? c : "", sizeof(buf));
        } else {
            ctx->ItemReadAsString(r, buf, sizeof(buf));
        }
        if (ctx->IsError()) return;
        std::printf("[ui]   value \"%s\" = \"%s\"\n", a[1].c_str(), buf);
        if (cmd == "expect_value" && a[2] != buf)
            SCRIPT_FAIL("line %d: '%s' is \"%s\", expected \"%s\"", l.no, a[1].c_str(), buf, a[2].c_str());
    }
    else if (cmd == "expect_mode") {
        if (!need(1)) return;
        AppModel* m = app_model();
        const int mode = m ? (int)m->app_mode : -1;
        const char* cur = (mode >= 0 && mode < AppModel::kAppModeCount) ? kModeNames[mode] : "?";
        if (!ieq(cur, a[1].c_str())) SCRIPT_FAIL("line %d: app mode is %s, expected %s", l.no, cur, a[1].c_str());
    }
    else if (cmd == "rect") {
        if (!need(1)) return;
        auto r = R(1);
        if (!ok()) return;
        ImGuiTestItemInfo it = ctx->ItemInfo(r);
        if (it.ID) print_item(it);
    }
    else if (cmd == "dump") {
        ImGuiTestItemList items;
        const int depth = n >= 2 ? std::atoi(a[2].c_str()) : 1;
        if (n >= 1) { auto w = W(1); if (!ok()) return; ctx->GatherItems(&items, w, depth); }
        else if (ctx->RefID) ctx->GatherItems(&items, ctx->GetRef(), depth);
        else { print_windows(false); return; }
        std::printf("[ui]   %d item(s)\n", items.GetSize());
        for (int i = 0; i < items.GetSize(); ++i) print_item(*items.GetByIndex(i));
    }
    else if (cmd == "find" || cmd == "click_prefix" || cmd == "rclick_prefix") {
        // Items under W ("*" = every window) whose label starts with PREFIX: `find` prints all of
        // them, click_prefix / rclick_prefix click the first (labels with counters: "Reset to default (0)").
        if (!need(2)) return;
        std::vector<ImGuiID> roots;
        if (a[1] == "*") roots = root_windows(); else { auto w = W(1); if (!ok()) return; roots = scope_ids(ctx, w); }
        ImGuiID id = 0;
        int hits = 0;
        std::vector<ImGuiID> seen;
        for (ImGuiID root : roots) {
            ImGuiTestItemList items;
            ctx->GatherItems(&items, ImGuiTestRef(root), -1);
            for (int i = 0; i < items.GetSize(); ++i) {
                const ImGuiTestItemInfo* it = items.GetByIndex(i);
                if (std::strncmp(it->DebugLabel, a[2].c_str(), a[2].size()) != 0) continue;
                if (std::find(seen.begin(), seen.end(), it->ID) != seen.end()) continue;
                seen.push_back(it->ID);
                if (!id) id = it->ID;
                if (cmd == "find" || hits == 0) print_item(*it);
                ++hits;
            }
        }
        if (!id) SCRIPT_FAIL("line %d: no item starting with \"%s\" under '%s'", l.no, a[2].c_str(), a[1].c_str());
        if (cmd == "click_prefix")  ctx->ItemClick(id);
        if (cmd == "rclick_prefix") ctx->ItemClick(id, ImGuiMouseButton_Right);
    }
    else if (cmd == "windows")   { print_windows(n >= 1 && a[1] == "all"); }
    else if (cmd == "screenshot") {
        if (!need(1)) return;
        ImRect rect(ImGui::GetMainViewport()->Pos,
                    ImVec2(ImGui::GetMainViewport()->Pos.x + ImGui::GetMainViewport()->Size.x,
                           ImGui::GetMainViewport()->Pos.y + ImGui::GetMainViewport()->Size.y));
        if (n >= 2) {
            auto wr = W(2); if (!ok()) return; ImGuiTestItemInfo wi = ctx->WindowInfo(wr);
            if (!wi.Window) SCRIPT_FAIL("line %d: window '%s' not found", l.no, a[2].c_str());
            rect = wi.Window->Rect();
        }
        ctx->CaptureReset();
        const std::string file = out_path(a[1]);
        ImStrncpy(ctx->CaptureArgs->InOutputFile, file.c_str(), IM_ARRAYSIZE(ctx->CaptureArgs->InOutputFile));
        ctx->CaptureArgs->InCaptureRect = rect;
        ctx->CaptureArgs->InPadding = 0.0f;
        ctx->CaptureScreenshot(ImGuiCaptureFlags_Instant | ImGuiCaptureFlags_HideMouseCursor);
        std::printf("[ui]   saved %s\n", file.c_str());
    }
    else if (cmd == "log")       { std::printf("[ui]   %s\n", n >= 1 ? a[1].c_str() : ""); }
    else if (cmd == "fail")      { SCRIPT_FAIL("line %d: %s", l.no, n >= 1 ? a[1].c_str() : "fail"); }
    else SCRIPT_FAIL("line %d: unknown command '%s'", l.no, cmd.c_str());
    std::fflush(stdout);
}

void script_test(ImGuiTestContext* ctx) {
    ctx->SetRef("//MainHost");
    for (const Line& l : g_lines) {
        run_line(ctx, l);
        if (ctx->IsError()) {
            std::printf("[ui] stopped at line %d\n", l.no);
            std::fflush(stdout);
            return;
        }
    }
}

} // namespace

ImGuiTest* register_script_test(ImGuiTestEngine* engine, const std::string& path) {
    if (!load_script(path)) {
        std::fprintf(stderr, "cannot read script %s\n", path.c_str());
        std::exit(2);
    }
    g_script_name = std::filesystem::path(path).filename().string();
    ImGuiTest* t = IM_REGISTER_TEST(engine, "script", g_script_name.c_str());
    t->TestFunc = script_test;
    return t;
}

} // namespace ui_test
