#pragma once
// Импорт SPICE-netlist'а в ДАУ вкладки DAE (этап 8 плана).
//
// Схема переводится в полуявную ДАУ модифицированным узловым методом (MNA) и отдаётся
// ТЕКСТОМ — теми же полями, что пользователь вводит руками (уравнения, x, z, параметры,
// НУ). Дальше её разбирает parse_dae, как любую ДАУ, и она идёт во все режимы анализа.
//
//   x — напряжения конденсаторов vc_<C> и токи индуктивностей il_<L>;
//   z — потенциалы узлов v_<узел>, токи конденсаторов ic_<C> и токи ветвей,
//       заданных напряжением (V, E, H, B с V=, выход нуллора): i_<имя>;
//   уравнения: vc' = ic/C, il' = (Va - Vb)/L; по узлу — сумма вытекающих токов = 0;
//       по конденсатору Va - Vb - vc = 0; по ветви — её уравнение напряжения.
//
// Номиналы (R, C, L, усиления, DC и параметры SIN источников) становятся параметрами a[]
// с именами элементов — по ним идёт свип без перекомпиляции.
//
// Поддержано: R, C, L, V, I (DC, SIN, PULSE, PWL, EXP), E, G (усиление, VALUE=, POLY(n)),
// F, H, B (V= / I=), X (.subckt с параметрами, вложенные), .param, .ic, .tran, .include.
// ОУ из таблицы замен (TL072, LM741, ...) — идеальный нуллор. E с |усилением| >= 1e4
// по умолчанию тоже считается нуллором (так эмиттер U-CUDA пишет идеальный ОУ).
// Диоды (D) и биполярные транзисторы (Q, NPN/PNP) — по .model, см. dae_devices.h; RS/RB/RC/RE
// дают внутренние узлы. Полевые транзисторы (M, J) — отказ с сообщением.
// Проверки топологии: контур из одних источников напряжения и узел только с источниками
// тока — ошибка; узел без пути к земле по постоянному току — замечание в отчёте.

#include <map>
#include <string>
#include <vector>

#include "circuit_synth.h"

struct SpiceImportOptions {
    // E с |gain| >= nullor_gain_min — нуллор (V+ = V-, ток выхода свободен). Иначе
    // уравнение V(out) = A (V+ - V-) с A ~ 1e6 даёт ту же схему с погрешностью ~1/A.
    bool   vcvs_as_nullor  = true;
    double nullor_gain_min = 1.0e4;
    // Каталог файла — от него считаются относительные пути .include.
    std::string base_dir;
};

struct SpiceImportResult {
    std::string title;
    // Поля записи ДАУ (режим Plain).
    std::string plain_text;
    std::string vars_text;     // x
    std::string alg_text;      // z
    std::string params_text;
    std::map<std::string, std::string> init_conditions;
    std::map<std::string, std::string> param_values;
    std::string step_h;        // из .tran (пусто — директивы нет)
    std::string t_stop;
    bool uic = false;          // .tran ... UIC
    bool has_ic = false;       // в netlist'е есть IC= или .ic
    // Без UIC и без НУ SPICE начинает переходный процесс с рабочей точки — так и импорт
    // (AppModel::import_circuit через dae_dc_operating_point).
    // Диагностика: замены моделей, пропущенные директивы, контуры C-V.
    std::vector<std::string> notes;
    std::string summary;       // "узлов N, R .., C .., ..."
    // Узел netlist'а -> имя переменной ДАУ (для подсказки в UI).
    std::vector<std::pair<std::string, std::string>> node_vars;
};

// false + err — синтаксис, неподдержанный элемент, неизвестная подсхема, ссылка
// на несуществующий источник. Текст ДАУ проверяет вызывающий (parse_dae).
bool spice_import(const std::string& netlist, const SpiceImportOptions& opt,
                  SpiceImportResult& out, std::string* err);

// Обратный путь для кругового теста: netlist, написанный emit_spice_netlist
// (R, C, V DC, X uc_opamp, умножители B / E POLY), -> CircuitGraph с теми же
// именами узлов. Земля — узел 0 "gnd", как у синтезатора.
bool spice_to_circuit_graph(const std::string& netlist, CircuitGraph& g, std::string* err);
