#pragma once
// Вкладка Tree: дерево анализа, панель параметров узла, окна листьев.
#include "gui_common.h"

namespace ucuda::ui_detail {
// Дерево по умолчанию из текущей системы, затем _last_tree.json.
void load_tree_session(AppModel& model, SystemLibrary& lib);
void draw_tree_controls(AppModel& model, SystemLibrary& lib);
void draw_tree_plot_windows(AppModel& model, const GuiCallbacks& cb);
void draw_tree_workspace_tabs(TreeSession& session, const std::string& system,
                              const std::function<void(int)>& on_close);
std::string tree_tab_suffix(const std::string& system, int tab);
std::string tree_controls_title(const std::string& system, int tab);
std::vector<std::string> tree_layout_windows(const AppModel& model);
std::string describe_tree_layout_window(const AppModel& model, const std::string& title);
std::string resolve_tree_layout_window(AppModel& model, const std::string& descriptor);
void apply_tree_layout_windows(AppModel& model, const std::vector<std::string>& matched);
void tree_forget_extra_views(int leaf, int first_view);
// Курсор владельца (регион или срез под System) сдвинут: автопересчёт после паузы.
void tree_cursor_moved(int owner_id);
// Цвет слота размерности: 0 System, 1 — 2D, 2 — 1D, 3 — 0D; выключенный — приглушённый.
[[nodiscard]] ImVec4 tree_dim_color(int slot, bool active);
// Лист, чьё окно вывести вперёд на этом кадре (0 — нет), и номер окна (0 — основное).
inline int g_tree_focus_leaf = 0;
inline int g_tree_focus_view = 0;
inline char g_tree_focus_type = 'w';
// Dockspace вкладки (ставит gui.cpp): сюда докуются новые окна листьев.
inline ImGuiID g_tree_dockspace_id = 0;
}
