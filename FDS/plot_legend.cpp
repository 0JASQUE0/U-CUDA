#include "plot_legend.h"
#include <cstdio>
#include <algorithm>

void draw_legend(ImDrawList* dl,
    ImVec2 img_pos, float plot_w,
    const std::vector<LegendEntry>& entries,
    std::vector<bool>& visible,
    const std::vector<bool>& global_visible,
    int owner_id)
{
    auto is_global_vis = [&](int k) -> bool {
        return (k < (int)global_visible.size()) ? global_visible[k] : true;
        };

    float line_h = ImGui::GetTextLineHeight();
    float row_h = line_h + 2;
    float marker_w = 14;
    float gap = 4;

    // ширина Ч по самым длинным ¬»ƒ»ћџћ глобально подпис€м
    float max_text_w = 0;
    int visible_count = 0;
    for (int k = 0; k < (int)entries.size(); ++k) {
        if (!is_global_vis(k)) continue;
        max_text_w = std::max(max_text_w, ImGui::CalcTextSize(entries[k].label.c_str()).x);
        ++visible_count;
    }
    if (visible_count == 0) return; // нечего показывать

    float pad = 6;
    float entry_w = marker_w + gap + max_text_w;
    float legend_w = entry_w + pad * 2;
    int nrows = std::min(visible_count, 20);
    float legend_h = nrows * row_h + pad * 2;

    ImVec2 leg_min = ImVec2(img_pos.x + plot_w - legend_w - 6, img_pos.y + 6);
    ImVec2 leg_max = ImVec2(leg_min.x + legend_w, leg_min.y + legend_h);
    dl->AddRectFilled(leg_min, leg_max, IM_COL32(20, 20, 25, 180), 4.0f);
    dl->AddRect(leg_min, leg_max, IM_COL32(80, 80, 90, 200), 4.0f);

    int row = 0;
    for (int k = 0; k < (int)entries.size(); ++k) {
        if (!is_global_vis(k)) continue;   // глобально выключенные не показываем
        if (row >= 20) break;
        bool v = (k < (int)visible.size()) ? visible[k] : true;
        ImVec4 c = entries[k].color;
        if (!v) c.w *= 0.35f;
        ImU32 cu = ImGui::ColorConvertFloat4ToU32(c);
        ImU32 text_cu = v ? IM_COL32(220, 220, 230, 255)
            : IM_COL32(180, 180, 190, 110);
        float row_y = leg_min.y + pad + row * row_h;
        ImVec2 mmin = ImVec2(leg_min.x + pad, row_y + 2);
        ImVec2 mmax = ImVec2(mmin.x + marker_w, row_y + row_h - 2);
        dl->AddRectFilled(mmin, mmax, cu);
        dl->AddText(ImVec2(mmax.x + gap, row_y), text_cu, entries[k].label.c_str());
        ImVec2 emin = ImVec2(leg_min.x + pad, row_y);
        ImVec2 emax = ImVec2(leg_max.x - pad, row_y + row_h);
        ImGui::SetCursorScreenPos(emin);
        char btn_id[48];
        std::snprintf(btn_id, sizeof(btn_id), "##legend_%d_%d", owner_id, k);
        ImGui::InvisibleButton(btn_id, ImVec2(emax.x - emin.x, emax.y - emin.y));
        if (ImGui::IsItemHovered())
            dl->AddRectFilled(emin, emax, IM_COL32(255, 255, 255, 18));
        if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
            if (k < (int)visible.size()) visible[k] = !visible[k];
        }
        ++row;
    }
}