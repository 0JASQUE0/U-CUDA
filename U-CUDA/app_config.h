#pragma once
#include <string>
#include <vector>
#include "parametric_engine.h"   // PeakConfig

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
    // 1001..1200 — карта slanCM; 0..8 — легаси, мигрируют при чтении (см.
    // colormap_id_or в plot_renderer.h). Дефолт 0 читается как #1 viridis.
    int heatmap_colormap = 0;

    // Colormap для табов панели бассейнов (независимо для каждого таба).
    // Дефолт 2 читается как #168 turbo: хорошо разделяет дискретные /
    // категориальные значения.
    int basins_colormap        = 2;
    int basins_avgpk_colormap  = 2;
    int basins_avgint_colormap = 2;
    int basins_states_colormap = 2;

    // Какие из 200 карт slanCM показывать в пикере Colormap: строка из
    // kSlanCmCount символов '0'/'1', символ i — карта #(i+1). Пустая строка =
    // ключа в JSON не было (конфиг от старой версии) → берётся
    // default_enabled_slancm(), т.е. те же пять карт, что были доступны
    // раньше. Правится галочками в Settings.
    std::string slancm_enabled;

    // Кол-во значащих цифр в подписях тиков осей и colorbar'а. Минимум 2
    // (исключает пустые подписи), максимум 10 (предел double-precision в `%g`).
    // По умолчанию 4 (сохраняет старое поведение).
    int tick_precision = 4;

    // Цветовая тема ImGui. true (дефолт) = StyleColorsDark, false = StyleColorsLight.
    // Переключается в Settings. apply_ui_scale в app_main.cpp перечитывает
    // это поле через model.dark_theme и пересоздаёт style на каждом изменении.
    bool dark_theme = true;

    // Последний выбранный тип анализа (значение AppModel::AppMode как int).
    // На старте app_main выставляет model.app_mode из этого поля (после
    // клампа на валидный диапазон enum'а). Дефолт 1 = Analysis — сохраняет
    // старое поведение "первый запуск открывается на Analysis".
    int last_app_mode = 1;

    // Имя последней загруженной системы. Пусто = ничего не грузим на старте
    // (первый запуск / система удалена из library). Если непустое и имя
    // существует в library — app_main вызовет apply_system_switch на bootstrap.
    std::string last_system_name;

    // Knobs configCUDA.h, настраиваемые во вкладке Settings (peak-finder и
    // пороги режимов). Дефолты структуры = значения из configCUDA.h, поэтому
    // отсутствие полей в JSON воспроизводит поведение до появления настройки.
    PeakConfig peak;

    // Вкладки верхнего ряда, СКРЫТЫЕ пользователем (Settings -> Tabs).
    // Элементы — стабильные идентификаторы режимов ("Analysis", "Parametric",
    // "Dft1D", "Basins", "FastSync", "Custom", "Order"); Library и Settings
    // скрыть нельзя и в списке не появляются. Храним именно скрытые, а не
    // видимые: отсутствие ключа (конфиг от старой версии) = ничего не скрыто,
    // т.е. прежнее поведение, и новая вкладка появляется видимой сама.
    std::vector<std::string> hidden_tabs;

    // Встроенные схемы, СКРЫТЫЕ из списка "Schemes to generate" (Settings ->
    // Integration schemes). Элементы — имена из kBuiltinSchemes в gui.cpp
    // ("RK4", "Complex CD4", ...). Логика та же, что у hidden_tabs: пусто =
    // показываем все.
    std::vector<std::string> hidden_schemes;

    // Системы библиотеки, СКРЫТЫЕ из глобального выпадающего списка (галочки
    // во вкладке Library). Логика та же, что у hidden_tabs: пусто = видны все,
    // и новая система появляется видимой. Имена систем произвольные, поэтому
    // список пишется через '\n', а не через запятую как hidden_tabs.
    std::vector<std::string> hidden_systems;

    // FMA-контракция во ВСЕХ NVRTC-сборках (см. set_nvrtc_fmad). true = дефолт
    // NVRTC; старые конфиги без этого ключа читаются как true, т.е. поведение
    // не меняется. Одно значение на приложение — раздельные флаги по типам
    // расчёта вернули бы рассогласование карты и портрета.
    bool nvrtc_fmad = true;

    // Раздельная компиляция NVRTC (см. set_nvrtc_rdc). false = дефолт: смена схемы компилируется
    // в разы быстрее, но шаг перестаёт инлайниться, и на схемах с длинным телом FMA-контракция
    // через границу вызова пропадает — последние разряды расходятся с прежними результатами.
    bool nvrtc_rdc = false;

    // Ширина блока запуска CUDA-ядер (см. set_gpu_block_size). На результат не влияет
    // вообще — только на скорость; конфиг без ключа читается как kGpuBlockSizeDefault.
    int gpu_block_size = kGpuBlockSizeDefault;
};

// Загружает `_app_config.json` из `dir` (директория exe). Если файл
// отсутствует или не парсится — возвращает false, `out` остаётся как есть
// (с дефолтами). dir должен оканчиваться слешем — пути склеиваются конкатенацией.
bool load_app_config(const std::string& dir, AppConfig& out);

// Сохраняет `_app_config.json` в `dir`. Атомарно: пишет в `<dir>_app_config.json.tmp`,
// потом ReplaceFile (или rename) — чтобы при сбое не получился полупустой файл.
bool save_app_config(const std::string& dir, const AppConfig& cfg);
