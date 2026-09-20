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
};

// Рисует один кадр интерфейса. lib — библиотека систем (вкладка Library).
// Отложенные действия кадра: то, что НЕЛЬЗЯ делать внутри кадра ImGui.
// Зовётся из главного цикла ПОСЛЕ ImGui::Render().
void gui_process_deferred(AppModel& model, const GuiCallbacks& cb);

void draw_gui(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb);

// Global system switch: loads the record from `lib` into `model` and re-inits
// the CURRENT tab's session (mirroring the top-bar combo). Exposed so
// app_main can restore the last-used system on startup.
void apply_system_switch(AppModel& model, SystemLibrary& lib, const std::string& name);