#pragma once
// Свёрнутые блоки между запусками. ImGui держит их в StateStorage окна и в imgui.ini не пишет, поэтому
// свой раздел [UcudaTree][имя окна] с парами id=0/1. Id блока это хеш подписи, он от запуска к запуску тот же.
#include "imgui.h"
#include "imgui_internal.h"
#include <cstdio>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace tree_state {

using Pairs = std::vector<std::pair<ImGuiID, int>>;

// Считано из файла и ещё не отдано окну: окно появится, когда откроют его вкладку.
inline std::map<std::string, Pairs>& waiting() {
    static std::map<std::string, Pairs> m;
    return m;
}
inline void* read_open(ImGuiContext*, ImGuiSettingsHandler*, const char* name) { return &waiting()[name]; }
inline void read_line(ImGuiContext*, ImGuiSettingsHandler*, void* entry, const char* line) {
    unsigned id = 0;
    int v = 0;
    if (std::sscanf(line, "%x=%d", &id, &v) == 2) ((Pairs*)entry)->push_back({ (ImGuiID)id, v });
}
inline void write_pairs(ImGuiTextBuffer* out, const char* name, const Pairs& pairs) {
    if (pairs.empty()) return;
    out->appendf("[UcudaTree][%s]\n", name);
    for (const auto& p : pairs) out->appendf("%08x=%d\n", p.first, p.second);
    out->append("\n");
}
inline void write_all(ImGuiContext* ctx, ImGuiSettingsHandler*, ImGuiTextBuffer* out) {
    for (ImGuiWindow* w : ctx->Windows) {
        if (w->Flags & (ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_Popup)) continue;
        Pairs pairs;
        for (const ImGuiStoragePair& p : w->StateStorage.Data)
            if (p.val_i == 0 || p.val_i == 1) pairs.push_back({ p.key, p.val_i });
        write_pairs(out, w->Name, pairs);
    }
    for (const auto& kv : waiting()) write_pairs(out, kv.first.c_str(), kv.second);
}

// Сразу после CreateContext: файл читается на первом NewFrame.
inline void install() {
    ImGuiSettingsHandler h;
    h.TypeName = "UcudaTree";
    h.TypeHash = ImHashStr("UcudaTree");
    h.ReadOpenFn = read_open;
    h.ReadLineFn = read_line;
    h.WriteAllFn = write_all;
    ImGui::AddSettingsHandler(&h);
}
// Каждый кадр после NewFrame: окно уже создано его первым Begin.
inline void apply() {
    auto& m = waiting();
    for (auto it = m.begin(); it != m.end(); ) {
        ImGuiWindow* w = ImGui::FindWindowByName(it->first.c_str());
        if (!w) { ++it; continue; }
        for (const auto& p : it->second) w->StateStorage.SetInt(p.first, p.second);
        it = m.erase(it);
    }
}

}  // namespace tree_state
