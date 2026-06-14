#pragma once
#include "imgui.h"
#include <string>
#include <vector>
// Описание одного элемента легенды.
struct LegendEntry {
    std::string label;
    ImVec4      color;
};
// Отрисовка легенды в правом верхнем углу плот-области.
// visible        - локальная видимость (клик по строке инвертирует флаг).
// global_visible - глобальная видимость (галочки НУ): строки с false НЕ показываются
//                  в легенде вообще. Если пусто/короче — считаются видимыми.
// owner_id       - индекс владельца (проекции), для уникальности ID кнопок ImGui.
void draw_legend(ImDrawList* dl,
    ImVec2 plot_pos, float plot_w,
    const std::vector<LegendEntry>& entries,
    std::vector<bool>& visible,
    const std::vector<bool>& global_visible,
    int owner_id);