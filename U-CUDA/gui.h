#pragma once
#include "app_model.h"
#include "system_library.h"

// Платформенные операции, передаются из app_main (чтобы gui не зависел от Win32).
struct GuiCallbacks {
    std::function<std::string()> pick_image_file;          // диалог выбора файла
    std::function<void(const std::string&)> set_clipboard_text; // копировать в буфер
    // Native save-file dialog for the right-click "Export data..." action on
    // plots. Returns the chosen absolute path, or empty string on cancel.
    // Default filter = CSV (.csv); the file may not exist yet.
    std::function<std::string()> pick_save_file_csv;
    std::function<std::string()> pick_save_file_netlist;   // диалог сохранения .cir
    // Диалог ОТКРЫТИЯ текстового файла. Пока один потребитель — таблица
    // коэффициентов композиции в конструкторе схем (System tab). Возвращает
    // абсолютный путь либо пустую строку при отмене.
    std::function<std::string()> pick_open_file_text;
    // Диалог открытия схемы для вкладки DAE: SPICE-netlist или .kicad_sch.
    std::function<std::string()> pick_open_file_circuit;
};

// Рисует один кадр интерфейса. lib — библиотека систем (вкладка Library).
// Отложенные действия кадра: то, что НЕЛЬЗЯ делать внутри кадра ImGui.
// Зовётся из главного цикла ПОСЛЕ ImGui::Render().
void gui_process_deferred(AppModel& model, const GuiCallbacks& cb);

void draw_gui(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb);

// Маски Settings на свежей установке (ключа в _app_config.json нет): показаны только Euler,
// Explicit Midpoint, CD, RK4, DOPRI78, GBS (n=2) и из методов ДАУ — Trapezoidal и BDF 1-5.
// Возвращают СКРЫТЫЕ имена, как AppModel::hidden_schemes / hidden_dae_methods.
std::vector<std::string> default_hidden_schemes();
std::vector<std::string> default_hidden_dae_methods();
// Все методы ДАУ в порядке списка редактора: dae_scheme_names() + BDF 1-5.
std::vector<std::string> dae_method_list();

// Global system switch: loads the record from `lib` into `model` and re-inits
// the CURRENT tab's session (mirroring the top-bar combo). Exposed so
// app_main can restore the last-used system on startup.
void apply_system_switch(AppModel& model, SystemLibrary& lib, const std::string& name);