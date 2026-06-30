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
};

// Рисует один кадр интерфейса. lib — библиотека систем (вкладка Library).
void draw_gui(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb);