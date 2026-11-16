#pragma once
// Вкладка Tree: дерево анализа, панель параметров узла, окна листьев.
#include "gui_common.h"

namespace ucuda::ui_detail {
// Дерево по умолчанию из текущей системы, затем _last_tree.json.
void load_tree_session(AppModel& model, SystemLibrary& lib);
void draw_tree_controls(AppModel& model, SystemLibrary& lib);
void draw_tree_plot_windows(AppModel& model, const GuiCallbacks& cb);
std::string tree_controls_title(const std::string& system);
std::vector<std::string> tree_layout_windows(const AppModel& model);
void apply_tree_layout_windows(AppModel& model, const std::vector<std::string>& unmatched);
void tree_hide_layout_windows(const std::string& system, const std::vector<std::string>& windows);
bool tree_has_hidden_windows(const std::string& system);
// Курсор владельца (регион или срез под System) сдвинут: автопересчёт после паузы.
void tree_cursor_moved(int owner_id);
// Цвет слота размерности: 0 System, 1 — 2D, 2 — 1D, 3 — 0D; выключенный — приглушённый.
[[nodiscard]] ImVec4 tree_dim_color(int slot, bool active);
// Лист, чьё окно вывести вперёд на этом кадре (0 — нет), и номер окна (0 — основное).
inline int g_tree_focus_leaf = 0;
inline int g_tree_focus_view = 0;
// Окно листа закрыто из дерева: виды окон с номера from_view начинают заново.
void tree_forget_views(int leaf_id, int from_view);
// Dockspace вкладки (ставит gui.cpp): сюда докуются новые окна листьев.
inline ImGuiID g_tree_dockspace_id = 0;
}
