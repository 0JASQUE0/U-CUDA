#pragma once
#include "imgui.h"
#include <string>
#include <vector>
// �������� ������ �������� �������.
struct LegendEntry {
    std::string label;
    ImVec4      color;
};

// Режим прохода draw_legend. Позволяет вызвать функцию дважды: первый раз
// для взаимодействия (клики → toggle visible[]) в старом месте рендера,
// второй раз для визуала поверх линий данных (когда те рисуются через
// ImDrawList в custom line style). Так легенда всегда оказывается сверху,
// не ломая клики.
enum class LegendPass {
    Both,       // текущее поведение: submit InvisibleButtons + рисуем визуал
    Interact,   // только InvisibleButtons + hover/click, без dl-> draw
    Draw,       // только визуал через dl->; hover highlight через
                // IsMouseHoveringRect (InvisibleButton не создаём заново).
};

// ��������� ������� � ������ ������� ���� ����-�������.
// visible        - ��������� ��������� (���� �� ������ ����������� ����).
// global_visible - ���������� ��������� (������� ��): ������ � false �� ������������
//                  � ������� ������. ���� �����/������ � ��������� ��������.
// owner_id       - ������ ��������� (��������), ��� ������������ ID ������ ImGui.
void draw_legend(ImDrawList* dl,
    ImVec2 plot_pos, float plot_w,
    const std::vector<LegendEntry>& entries,
    std::vector<bool>& visible,
    const std::vector<bool>& global_visible,
    int owner_id,
    LegendPass pass = LegendPass::Both);