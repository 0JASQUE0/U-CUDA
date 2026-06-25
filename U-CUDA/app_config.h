#pragma once
#include <string>

// Глобальные настройки приложения, не привязанные к конкретной системе или
// сессии. Хранятся в файле `_app_config.json` рядом с exe. Сейчас единственное
// поле — `ui_scale_override`; добавлять новые поля по мере необходимости.
struct AppConfig {
    // 0.0 == «использовать auto-detected scale» (из glfwGetMonitorContentScale).
    // Положительное значение — пользовательский override (слайдер в GUI).
    float ui_scale_override = 0.0f;

    // true → bitmap ProggyClean (как было до Segoe UI). false (дефолт) → TTF
    // Segoe UI с `C:\Windows\Fonts\segoeui.ttf`. Чекбокс в Settings.
    bool use_builtin_font = false;

    // Последний выбранный colormap для HeatmapView (LLE-2D и пр.).
    // 0=Viridis, 1=Inferno, 2=Turbo, 3=Gray. Дефолт — Viridis.
    int heatmap_colormap = 0;
};

// Загружает `_app_config.json` из `dir` (директория exe). Если файл
// отсутствует или не парсится — возвращает false, `out` остаётся как есть
// (с дефолтами). dir должен оканчиваться слешем — пути склеиваются конкатенацией.
bool load_app_config(const std::string& dir, AppConfig& out);

// Сохраняет `_app_config.json` в `dir`. Атомарно: пишет в `<dir>_app_config.json.tmp`,
// потом ReplaceFile (или rename) — чтобы при сбое не получился полупустой файл.
bool save_app_config(const std::string& dir, const AppConfig& cfg);
