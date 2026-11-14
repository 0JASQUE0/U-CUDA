#pragma once
// Вкладка Tree: дерево анализа, панель параметров узла, окна листьев.
#include "gui_common.h"

namespace ucuda::ui_detail {
// Дерево по умолчанию из текущей системы, затем _last_tree.json.
void load_tree_session(AppModel& model, SystemLibrary& lib);
void draw_tree_controls(AppModel& model, SystemLibrary& lib);
void draw_tree_plot_windows(AppModel& model, const GuiCallbacks& cb);
// Курсор владельца (регион или срез под System) сдвинут: автопересчёт после паузы.
void tree_cursor_moved(int owner_id);
// Цвет слота размерности: 0 System, 1 — 2D, 2 — 1D, 3 — 0D; выключенный — приглушённый.
[[nodiscard]] ImVec4 tree_dim_color(int slot, bool active);
// Лист, чьё окно вывести вперёд на этом кадре (0 — нет).
inline int g_tree_focus_leaf = 0;
// Dockspace вкладки (ставит gui.cpp): сюда докуются новые окна листьев.
inline ImGuiID g_tree_dockspace_id = 0;
}
