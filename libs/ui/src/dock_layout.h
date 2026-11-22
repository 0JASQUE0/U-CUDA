#pragma once
// Раскладки окон (кнопка Layout в шапке): встроенные пресеты Tabs / Columns / Rows / Grid и
// сохранённые пользователем раскладки. У каждой вкладки приложения свой dockspace, раскладка
// перестраивает именно его.
//
// Сохранённая раскладка — файл layouts/<вкладка>/<имя>.layout рядом с exe (текст, переносится
// копированием, как система). Дерево узлов хранится долями, а не пикселями: на мониторе
// другого размера окна встают в тех же пропорциях.
#include "imgui.h"
#include <functional>
#include <string>
#include <vector>

struct ImGuiWindow;

namespace ucuda::ui_detail {

enum DockLayoutPreset { kDockTabs = 0, kDockColumns, kDockRows, kDockGrid, kDockPresetCount };

// Тело всплывающего меню кнопки Layout для вкладки mode (AppModel::AppMode).
void dock_layout_menu(int mode, ImGuiID target = 0);

// Запросить встроенный пресет из кода (например, Tabs для вкладки без раскладки).
void dock_layout_request(int mode, int preset, ImGuiID target = 0);
// Запросить сохранённую (или встроенную, например Custom "Default") раскладку по имени.
// delay_frames — подождать, пока вкладка создаст свои окна (первый показ вкладки).
void dock_layout_request_saved(int mode, const std::string& name, int delay_frames = 0);
bool dock_layout_has_request(int mode, ImGuiID target = 0);
void dock_layout_forget_target(ImGuiID target);

// Окна-объекты вкладки (проекции, окна Parametric / DFT1D / Order): раскладка хранит их
// настройки и при применении создаёт недостающие.
struct DockWindowFactory {
    // Настройки объекта за окном name (JSON одной строкой); пусто — окно не создаваемое.
    std::function<std::string(const std::string& name)> describe;
    // Создать объект по настройкам; вернуть имя его будущего окна (пусто — не вышло).
    std::function<std::string(const std::string& json)> create;
    // Состояние вкладки, от которого зависит набор окон (в Custom — галки уровней и
    // срезов): строка без переводов строк. load_state возвращает true, если что-то
    // поменялось, — тогда раскладка ставится через пару кадров, когда окна появятся.
    std::function<std::string()> save_state;
    std::function<bool(const std::string&)> load_state;
    // Tree: all known windows in this workspace, including hidden copies.
    std::function<std::vector<std::string>()> available_windows;
    std::function<std::string(const std::string& descriptor)> resolve;
    std::function<void(const std::vector<std::string>&)> apply_matched;
};

// В точке, где вкладка подаёт свой dockspace (до ImGui::DockSpace): выполнить запрос вкладки
// mode над dock_id размера size. wins — окна вкладки (порядок — для встроенных пресетов),
// side_window — панель настроек, которую пресеты держат справа (nullptr — нет),
// factory — создание недостающих окон сохранённой раскладки (nullptr — не создаются).
void dock_layout_process(int mode, ImGuiID dock_id, ImVec2 size, const char* side_window,
                         const std::vector<ImGuiWindow*>& wins,
                         const DockWindowFactory* factory = nullptr);

// Сразу после ImGui::DockSpace(dock_id): разрезы дока держат свои доли, когда их длина меняется
// не их собственной границей (окно приложения, граница панели настроек или соседнего узла).
// ImGui в этом случае отдаёт всё изменение центральному узлу, а соседям оставляет прежние
// пиксели, и границы, стоявшие на одной линии в разных строках, расходились. Доля разреза
// запоминается, пока его длина не меняется (в том числе пока тянут его собственную границу).
// fixed_window — панель настроек: разрез, отделяющий её лист, остаётся по правилам ImGui
// (панель держит ширину в пикселях).
void dock_keep_split_ratios(ImGuiID dock_id, const char* fixed_window = nullptr);

// Прямо перед ImGui::DockSpace(dock_id): перетаскиваемая граница дока притягивается к разделителю
// той же оси в соседней строке (столбце) — встаёт с ним перекрестием — и к середине своего узла,
// в радиусе ~0.6 высоты шрифта. Видно во время перетаскивания: ImGui сам ставит границу и окна.
void dock_snap_dragged_split(ImGuiID dock_id);

// Ставит своё меню треугольника у вкладки окна (ImGuiContext::DockNodeWindowMenuHandler):
// «Hide tab bar» с зажатыми Ctrl+Shift скрывает (или показывает) полосы вкладок у всех
// окон этой док-области, остальное как в стандартном меню. Звать каждый кадр дёшево.
void dock_install_window_menu();

// Щелчок с Ctrl+Shift на треугольник в углу окна со скрытой полосой вкладок
// показывает полосы у всех окон док-области. Щелчок ImGui ловит в Begin окна и
// применяет в DockNodeUpdate следующего кадра, поэтому звать перед DockSpace.
void dock_unhide_all_on_modifier(ImGuiID dock_id);

}  // namespace ucuda::ui_detail
