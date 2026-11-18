// Command-script test: one text file of UI actions, executed by the Test Engine as test "script/<file>".
// Every command and its arguments are documented in apps/desktop/ui_test/README.md.
#include "ui_test_host.h"

#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_te_engine.h"
#include "imgui_te_context.h"
#include "imgui_capture_tool.h"

#include "ucuda/application/app_model.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace ui_test {
namespace {

struct Line { int no; std::vector<std::string> args; };

std::vector<Line> g_lines;
std::string       g_script_name;

// Splits on blanks; "double quotes" group words, \" and \\ escape inside quotes.
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
                if (s[i] == '\\' && i + 1 < s.size()) ++i;
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
        const bool child = (w->Flags & ImGuiWindowFlags_ChildWindow) != 0;
        if (child && !all) continue;
        std::printf("[ui]   0x%08X \"%s\" @(%.0f,%.0f %.0fx%.0f)%s%s%s\n", w->ID, w->Name, w->Pos.x, w->Pos.y,
                    w->Size.x, w->Size.y, child ? " child" : "", w->DockIsActive ? " docked" : "",
                    (w->Flags & ImGuiWindowFlags_Popup) ? " popup" : "");
    }
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

    if (cmd == "ref")            { if (need(1)) ctx->SetRef(make_ref(a[1])); }
    else if (cmd == "click")     { if (need(1)) ctx->ItemClick(make_ref(a[1])); }
    else if (cmd == "rclick")    { if (need(1)) ctx->ItemClick(make_ref(a[1]), ImGuiMouseButton_Right); }
    else if (cmd == "dclick")    { if (need(1)) ctx->ItemDoubleClick(make_ref(a[1])); }
    else if (cmd == "hover")     { if (need(1)) ctx->MouseMove(make_ref(a[1])); }
    else if (cmd == "check")     { if (need(1)) ctx->ItemCheck(make_ref(a[1])); }
    else if (cmd == "uncheck")   { if (need(1)) ctx->ItemUncheck(make_ref(a[1])); }
    else if (cmd == "open")      { if (need(1)) ctx->ItemOpen(make_ref(a[1])); }
    else if (cmd == "close")     { if (need(1)) ctx->ItemClose(make_ref(a[1])); }
    else if (cmd == "input")     { if (need(2)) ctx->ItemInputValue(make_ref(a[1]), a[2].c_str()); }
    else if (cmd == "type")      { if (need(1)) ctx->KeyChars(a[1].c_str()); }
    else if (cmd == "menu")      { if (need(1)) ctx->MenuClick(make_ref(a[1])); }
    else if (cmd == "combo")     { if (need(1)) ctx->ComboClick(make_ref(a[1])); }
    else if (cmd == "tab_close") { if (need(1)) ctx->TabClose(make_ref(a[1])); }
    else if (cmd == "focus")     { if (need(1)) ctx->WindowFocus(make_ref(a[1])); }
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
        if (need(3)) ctx->ItemDragWithDelta(make_ref(a[1]), ImVec2((float)std::atof(a[2].c_str()),
                                                                     (float)std::atof(a[3].c_str())));
    }
    else if (cmd == "scroll_y")  { if (need(2)) ctx->ScrollToY(make_ref(a[1]), (float)std::atof(a[2].c_str())); }
    else if (cmd == "scroll_to") { if (need(1)) ctx->ScrollToItemY(make_ref(a[1])); }
    else if (cmd == "resize") {
        if (need(3)) ctx->WindowResize(make_ref(a[1]), ImVec2((float)std::atof(a[2].c_str()),
                                                               (float)std::atof(a[3].c_str())));
    }
    else if (cmd == "yield")     { ctx->Yield(n >= 1 ? std::atoi(a[1].c_str()) : 1); }
    else if (cmd == "sleep")     { if (need(1)) ctx->SleepNoSkip((float)std::atof(a[1].c_str()), 1.0f / 60.0f); }
    else if (cmd == "wait_exists" || cmd == "wait_missing") {
        if (!need(1)) return;
        const bool want = cmd == "wait_exists";
        const double timeout = n >= 2 ? std::atof(a[2].c_str()) : 10.0;
        const double t0 = ImGui::GetTime();
        while (ctx->ItemExists(make_ref(a[1])) != want) {
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
        while (m && (m->bifurcation_session.in_flight || m->lle_session.in_flight || m->ls_session.in_flight ||
                     m->metrics_session.in_flight || m->basins_session.in_flight || m->dft1d_session.in_flight ||
                     m->fastsync_session.in_flight || m->custom_session.any_in_flight())) {
            if (ImGui::GetTime() - t0 > timeout) SCRIPT_FAIL("line %d: still computing after %.0f s", l.no, timeout);
            ctx->SleepNoSkip(0.1f, 1.0f / 60.0f);
        }
    }
    else if (cmd == "expect_exists" || cmd == "expect_missing") {
        if (!need(1)) return;
        const bool want = cmd == "expect_exists";
        if (ctx->ItemExists(make_ref(a[1])) != want)
            SCRIPT_FAIL("line %d: expected '%s' to be %s", l.no, a[1].c_str(), want ? "present" : "absent");
    }
    else if (cmd == "expect_checked" || cmd == "expect_unchecked") {
        if (!need(1)) return;
        const bool want = cmd == "expect_checked";
        if (ctx->ItemIsChecked(make_ref(a[1])) != want)
            SCRIPT_FAIL("line %d: expected '%s' %s", l.no, a[1].c_str(), want ? "checked" : "unchecked");
    }
    else if (cmd == "expect_open" || cmd == "expect_closed") {
        if (!need(1)) return;
        const bool want = cmd == "expect_open";
        if (ctx->ItemIsOpened(make_ref(a[1])) != want)
            SCRIPT_FAIL("line %d: expected '%s' %s", l.no, a[1].c_str(), want ? "open" : "closed");
    }
    else if (cmd == "read" || cmd == "expect_value") {
        if (!need(cmd == "read" ? 1 : 2)) return;
        char buf[1024] = {};
        ctx->ItemReadAsString(make_ref(a[1]), buf, sizeof(buf));
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
        ImGuiTestItemInfo it = ctx->ItemInfo(make_ref(a[1]));
        if (it.ID) print_item(it);
    }
    else if (cmd == "dump") {
        ImGuiTestItemList items;
        const int depth = n >= 2 ? std::atoi(a[2].c_str()) : 1;
        if (n >= 1) ctx->GatherItems(&items, make_ref(a[1]), depth);
        else if (ctx->RefID) ctx->GatherItems(&items, ctx->GetRef(), depth);
        else { print_windows(false); return; }
        std::printf("[ui]   %d item(s)\n", items.GetSize());
        for (int i = 0; i < items.GetSize(); ++i) print_item(*items.GetByIndex(i));
    }
    else if (cmd == "find" || cmd == "click_prefix" || cmd == "rclick_prefix") {
        // First item under W whose label starts with PREFIX (labels with counters: "Reset to default (0)").
        if (!need(2)) return;
        ImGuiTestItemList items;
        ctx->GatherItems(&items, make_ref(a[1]), -1);
        const ImGuiTestItemInfo* hit = nullptr;
        for (int i = 0; i < items.GetSize() && !hit; ++i)
            if (std::strncmp(items.GetByIndex(i)->DebugLabel, a[2].c_str(), a[2].size()) == 0)
                hit = items.GetByIndex(i);
        if (!hit) SCRIPT_FAIL("line %d: no item starting with \"%s\" under '%s'", l.no, a[2].c_str(), a[1].c_str());
        print_item(*hit);
        const ImGuiID id = hit->ID;
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
            ImGuiTestItemInfo wi = ctx->WindowInfo(make_ref(a[2]));
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
