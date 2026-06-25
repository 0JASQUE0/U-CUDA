#pragma once
#include <string>
#include <vector>
#include <map>

// Именованная пользовательская КРС. Это "ещё один scheme" — встаёт в scheme
// combo рядом с Euler/Cromer/Midpoint/RK4. Имя не должно совпадать с built-in.
// body — сырой C/CUDA-текст, подставляется напрямую в NVRTC-шаблон вместо
// codegen-вывода. Доступны X[0..N-1], a[1..M], h, AMOUNTOFX, обычная математика.
struct CustomScheme {
    std::string name;
    std::string body;
};

// Полная запись системы в библиотеке: весь ввод (для редактирования) +
// значения по умолчанию (все опциональны: пустая строка = не задано).
struct SystemRecord {
    // --- метаданные ---
    std::string name;            // имя (для списка); если пусто — автоимя при сохранении
    std::string note;            // заметка/ссылка на статью

    // --- ввод (для редактирования) ---
    std::string mode;            // "Image" | "LaTeX" | "Plain"
    std::string latex_text;
    std::string plain_text;
    std::string alphabet_text;       // legacy: один список (vars + params вперемешку)
    // Явные списки переменных и параметров. Приоритет над alphabet_text:
    // если оба непустые — используются они напрямую; alphabet_text остаётся
    // у старых записей как legacy fallback (либо когда equations должны
    // сами вывести vars/params).
    std::string vars_text;
    std::string params_text;
    bool        use_aux_funcs = false;
    std::string func_defs_text;
    std::string param_order;     // "AsInAlphabet" | "AsInSystem"

    // выбранные методы
    bool scheme_euler    = false;
    bool scheme_cromer   = false;
    bool scheme_midpoint = false;
    bool scheme_rk4      = false;
    bool scheme_dopri78  = false;

    // Пользовательские именованные КРС (см. CustomScheme выше).
    std::vector<CustomScheme> custom_schemes;

    // --- значения по умолчанию (всё опционально, пустое = не задано) ---
    std::string step_h;          // шаг дискретизации (строка, пустая = не задано)

    // начальные условия по имени переменной: "x" -> "1.0" (пустое = не задано)
    std::map<std::string, std::string> init_conditions;

    // значения параметров по имени: "sigma" -> "10" (пустое = не задано)
    std::map<std::string, std::string> param_values;

    // диапазоны параметров для бифуркаций: имя -> (min, max), строки
    std::map<std::string, std::string> param_min;
    std::map<std::string, std::string> param_max;
};
