#pragma once
#include <string>
#include <vector>
#include "circuit_solver.h"

// Экспорт синтезированной схемы в SPICE-netlist.
//
// Диалект — Berkeley SPICE 3f5 с B-источниками, то есть NGSpice и LTspice.
// Multisim netlist импортирует (File -> Open), но автораскладка даёт нечитаемое
// «спагетти»: координат в netlist'е нет по определению.
//
// Модели компонентов выводятся ЯВНО, подсхемами, а не ссылками на TL072/AD633 из
// чужой библиотеки. Причина прагматичная: так внешний симулятор считает ровно ту
// же модель, что и наш решатель, и расхождение означает ошибку решателя, а не
// разницу моделей. Для сверки с реальным железом подсхему меняют на вендорскую —
// в шапке netlist'а об этом сказано.

struct NetlistOptions {
    std::string title = "U-CUDA synthesized circuit";
    // Строки системы — уходят в шапку как происхождение схемы.
    std::vector<std::string> var_names;
    std::vector<std::string> rhs_text;
    // Масштабы амплитуды: x_i = scale[i] * V(узел_i). Пусто — не печатать.
    std::vector<double> scale;
    // Начальные условия в ВОЛЬТАХ, по одному на переменную. Без них схема сидит
    // в неподвижной точке — хаос требует стартового толчка.
    std::vector<double> x0_volts;

    OpAmpModel opamp;
    // Горизонт и шаг в единицах времени ОДУ; в секунды переводит graph.time_scale.
    double t_end_ode = 50.0;
    double h_ode     = 0.001;
    // Внутренняя ёмкость полюса — та же, что у решателя, иначе модели разъедутся.
    double opamp_cpole = 1.0e-9;
};

std::string emit_spice_netlist(const CircuitGraph& g, const NetlistOptions& o);
