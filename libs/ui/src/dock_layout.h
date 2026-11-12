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
void dock_layout_menu(int mode);

// Запросить встроенный пресет из кода (например, Tabs для вкладки без раскладки).
void dock_layout_request(int mode, int preset);
// Запросить сохранённую (или встроенную, например Custom "Default") раскладку по имени.
// delay_frames — подождать, пока вкладка создаст свои окна (первый показ вкладки).
void dock_layout_request_saved(int mode, const std::string& name, int delay_frames = 0);
bool dock_layout_has_request(int mode);

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
};

// В точке, где вкладка подаёт свой dockspace (до ImGui::DockSpace): выполнить запрос вкладки
// mode над dock_id размера size. wins — окна вкладки (порядок — для встроенных пресетов),
// side_window — панель настроек, которую пресеты держат справа (nullptr — нет),
// factory — создание недостающих окон сохранённой раскладки (nullptr — не создаются).
void dock_layout_process(int mode, ImGuiID dock_id, ImVec2 size, const char* side_window,
                         const std::vector<ImGuiWindow*>& wins,
                         const DockWindowFactory* factory = nullptr);

}  // namespace ucuda::ui_detail
