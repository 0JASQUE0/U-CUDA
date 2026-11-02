#pragma once
// Зеркало настроек приложения (AppConfig) в модели: видимость вкладок и схем, масштаб, шрифты,
// палитры, тема, параметры поиска пиков, флаги NVRTC. Часть AppModel (AppModel : SystemEditor : AppSettings).
#include "ucuda/domain/analysis_types.h"   // PeakConfig, kGpuBlockSizeDefault
#include <algorithm>
#include <string>
#include <vector>

struct AppSettings {
    // Настройки видимости из Settings. Зеркалят AppConfig::hidden_tabs /
    // hidden_schemes: СКРЫТЫЕ идентификаторы вкладок (kModeTabs в gui.cpp) и
    // имена встроенных схем (kBuiltinSchemes там же). Храним скрытые, а не
    // видимые, поэтому пустой список = показываем всё, и новая вкладка/схема
    // появляется видимой даже у тех, у кого конфиг уже записан.
    //
    // Единственное исключение — Network: на свежей установке (конфига рядом с
    // exe ещё нет) вкладка скрыта, включается галочкой в Settings -> Tabs. У
    // того, у кого конфиг уже есть, список берётся из него и этот дефолт не
    // применяется вовсе. Тот же дефолт продублирован в AppConfig::hidden_tabs —
    // на случай конфига без ключа.
    std::vector<std::string> hidden_tabs { "Dft1D", "FastSync", "Order", "Network", "Neural" };
    std::vector<std::string> hidden_schemes;
    // Методы ДАУ, скрытые из списка редактора ДАУ (Settings -> DAE methods). Зеркалит
    // AppConfig::hidden_dae_methods; дефолт свежей установки ставит app_main.
    std::vector<std::string> hidden_dae_methods;

    // То же для систем библиотеки: имена, скрытые из глобального комбо
    // (галочки во вкладке Library). Зеркалит AppConfig::hidden_systems.
    std::vector<std::string> hidden_systems;

    static bool name_hidden(const std::vector<std::string>& v, const std::string& n) {
        return std::find(v.begin(), v.end(), n) != v.end();
    }
    static void set_name_hidden(std::vector<std::string>& v, const std::string& n, bool hidden) {
        auto it = std::find(v.begin(), v.end(), n);
        if (hidden && it == v.end()) v.push_back(n);
        else if (!hidden && it != v.end()) v.erase(it);
    }
    bool tab_hidden(const std::string& id) const { return name_hidden(hidden_tabs, id); }
    bool scheme_hidden(const std::string& name) const { return name_hidden(hidden_schemes, name); }
    bool dae_method_hidden(const std::string& name) const { return name_hidden(hidden_dae_methods, name); }
    bool system_hidden(const std::string& name) const { return name_hidden(hidden_systems, name); }

    // UI scale (DPI-aware). `ui_scale_auto` выставляется однократно при старте из
    // glfwGetMonitorContentScale, `ui_scale_override` — слайдер в GUI, где 0.0 означает
    // «использовать auto». Эффективный scale идёт в apply_ui_scale в app_main каждый кадр: если
    // изменился, шрифт и ImGui-style пересоздаются.
    float ui_scale_auto     = 1.0f;
    float ui_scale_override = 0.0f;
    float effective_ui_scale() const {
        return ui_scale_override > 0.0f ? ui_scale_override : ui_scale_auto;
    }

    // Выбор шрифта. false (дефолт) → TTF Segoe UI (антиалиаc, читаемо на 1080p
    // и крупных DPI). true → bitmap ProggyClean (компактный, классический ImGui).
    // Чекбокс — в Settings; персистится в _app_config.json.
    bool use_builtin_font = false;

    // Кегль подписей на графиках — множитель к UI-шрифту. Зеркалит
    // AppConfig::plot_font_scale; при изменении в Settings зовётся
    // set_plot_font_scale() из plot_axis.h, чтобы подписи и марджины плотов
    // подхватили новое значение тем же кадром.
    float plot_font_scale = 1.0f;

    // LaTeX-подписи на графиках (см. plot_text в plot_axis.h): серифная пара
    // шрифтов + греческие буквы, индексы и курсив. Применяется тем же
    // apply_ui_scale в app_main.cpp, что и use_builtin_font — шрифты живут в
    // общем атласе и пересоздаются вместе с UI-шрифтом.
    bool plot_math_font = true;
    // Имена параметров и переменных в полях — формулами (AppConfig::field_latex).
    bool field_latex = true;
    // Поля НУ алгебраических переменных ДАУ (AppConfig::show_alg_ic); выключено — ic_fields их убирает.
    bool show_alg_ic = false;

    // Последний выбранный colormap для HeatmapView (LLE-2D и пр.). 1001..1200 — карта slanCM
    // (id = 1000 + её родной номер), легаси 0..8 мигрируют при чтении конфига; дефолт 2 = #168
    // turbo. Персистится в _app_config.json. Combo «Colormap» над хитмапой пишет сюда, а static
    // HeatmapView в draw_lle_plot читает при первом создании.
    int heatmap_colormap = 2;

    // Colormap для табов панели бассейнов. Каждый таб (Basins/AvgPk/AvgInt/States) хранит свой
    // независимый выбор, чтобы переключение в одном не влияло на остальные; отдельно от
    // heatmap_colormap, который шарится Bif/LLE/LS. Дефолт 2 = #168 turbo (хорошо разделяет
    // дискретные / категориальные значения).
    int basins_colormap        = 2;
    int basins_avgpk_colormap  = 2;
    int basins_avgint_colormap = 2;
    int basins_states_colormap = 2;

    // Маска включённых карт slanCM (kSlanCmCount символов '0'/'1') — зеркало
    // AppConfig::slancm_enabled. При изменении в Settings вызывается
    // set_enabled_slancm() из plot_renderer.h, чтобы пикер сразу подхватил,
    // — ровно как tick_precision + set_tick_precision() у осей.
    std::string slancm_enabled;

    // Строка фильтра в списке колормапов (Settings). Живёт в модели, а не в
    // static внутри draw_settings: immediate-mode UI не хранит состояние
    // между кадрами сам (см. CLAUDE.md).
    std::string colormap_filter;

    // Кол-во значащих цифр в подписях тиков осей и colorbar'а (2-10).
    // Зеркалит AppConfig::tick_precision; при изменении в Settings вызывается
    // set_tick_precision() из plot_axis.h, чтобы fmt_tick() сразу подхватил.
    int tick_precision = 4;

    // ImGui color theme. true = Dark (default), false = Light. Применяется
    // через apply_ui_scale в app_main.cpp: на каждый scale-apply сначала
    // выбирается ImGui::StyleColorsDark/Light, потом скейлится. На изменение
    // в Settings forced re-apply через сравнение applied_dark_theme.
    bool dark_theme = true;

    // Knobs configCUDA.h (peak-finder + пороги режимов). Зеркалит
    // AppConfig::peak; ImGui-состояние обязано жить здесь, а не в локальных
    // переменных кадра. Settings после правки пушит его в set_peak_config(),
    // который бампает epoch и инвалидирует PTX-кэши движка.
    PeakConfig peak;

    // Текстовые буферы полей peak-конфига в Settings. Ввод чисел там такой же, как во вкладках
    // анализа (InputNumStr поверх InputText: ↑/↓ шаг по разряду, запятая→точка, дроби "a/b",
    // inline-предупреждение), а не InputDouble с фиксированным "%.3e". Строка — только буфер ввода;
    // источник истины сам `peak`, поэтому буферы пересеиваются из него через sync_peak_text() при
    // старте, после загрузки конфига и после клампа/сброса.
    std::string peak_eps_fixed_point_text;
    std::string peak_eps_peak_delta_text;
    std::string peak_eps_interPeak_delta_text;
    std::string peak_threshold_text;
    std::string peak_max_amount_text;

    // Перезаписывает буферы выше текущими значениями `peak`.
    void sync_peak_text();

    // FMA-контракция во всех NVRTC-сборках. Зеркалит AppConfig::nvrtc_fmad;
    // Settings после правки пушит значение в set_nvrtc_fmad(), которое меняет
    // ключи кэшей модулей в обоих движках (см. parametric_engine.h).
    bool nvrtc_fmad = true;

    // Раздельная компиляция NVRTC. Зеркалит AppConfig::nvrtc_rdc тем же способом, что nvrtc_fmad:
    // Settings пушит значение в set_nvrtc_rdc(). Входит в ключ кэша модулей, т.е. переключение
    // заставляет пересобрать их на следующем Run.
    bool nvrtc_rdc = false;

    // Ширина блока запуска CUDA-ядер. Зеркалит AppConfig::gpu_block_size; Settings после
    // правки пушит значение в set_gpu_block_size(). На РЕЗУЛЬТАТ не влияет — в отличие от
    // nvrtc_fmad, кэши модулей не трогает и перекомпиляции не требует.
    int gpu_block_size = kGpuBlockSizeDefault;

};
