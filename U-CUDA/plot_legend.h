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
// out_clicks     - опционально: куда сложить результат ПКМ по легенде (см.
//                  LegendRightClick). Сам факт non-null указателя означает
//                  «у вызывающего есть меню цвета»: в Draw-проходе (где кликов
//                  не ловим) по нему рисуется рамка-подсказка на наведении
//                  квадрата. Поэтому передавать его следует в ОБА прохода, а
//                  легенда без такого меню (Plot3DView) оставляет nullptr —
//                  и подсказки не получает.

// Результат правого клика по легенде. Легенда подаёт свои InvisibleButton'ы
// РАНЬШЕ плотовского и забирает hover себе, поэтому плот про эти клики иначе
// не узнаёт вообще — а меню он показать обязан.
struct LegendRightClick {
    int  swatch_index = -1;     // ПКМ по цветному квадрату строки k -> меню цвета
    bool row_other    = false;  // ПКМ по строке мимо квадрата  -> обычное меню плота
};

void draw_legend(ImDrawList* dl,
    ImVec2 plot_pos, float plot_w,
    const std::vector<LegendEntry>& entries,
    std::vector<bool>& visible,
    const std::vector<bool>& global_visible,
    int owner_id,
    LegendPass pass = LegendPass::Both,
    LegendRightClick* out_clicks = nullptr);