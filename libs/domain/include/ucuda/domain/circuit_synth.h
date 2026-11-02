#pragma once
#include <string>
#include <vector>
#include "ucuda/domain/system.h"   // System (SystemEvaluator нужен только реализации)

// Синтез аналоговой схемы на ОУ по системе ОДУ (фаза 1 плана).
//
// Работает в два шага: правые части раскладываются на мономы степени <= 2, по
// мономам строится граф схемы. Номиналы получаются из коэффициентов по правилу
// R = R_unit / |k|, масштабирование переменных — задача более поздней фазы.
//
// Здесь только типы; функции разложения, масштабирования и синтеза — в
// ucuda/numerics/circuit_synth.h (реализация в numerics).

// --- Разложение правой части на мономы ---------------------------------------
//
// AST кодогена лежит в анонимном namespace codegen.cpp и наружу не виден, а
// сторонняя CAS в проект не тянется. Поэтому коэффициенты снимаются численным
// зондированием публичного SystemEvaluator: для полинома степени <= 2 их дают
// точные конечные формулы, а не подгонка.

// Моном: coeff * x[v1] * x[v2]. v1 = v2 = -1 — свободный член, v2 = -1 — линейный.
struct PolyTerm {
    int    v1 = -1;
    int    v2 = -1;
    double coeff = 0.0;
};

struct PolyRhs {
    std::vector<PolyTerm> terms;   // только ненулевые, порядок канонический
};

struct PolyExtractConfig {
    // Порог отсева коэффициента, относительный к масштабу правой части.
    double zero_tol = 1e-9;
    // Порог невязки при проверке "это действительно полином степени <= 2".
    double residual_tol = 1e-7;
    // Сколько случайных точек берётся на проверку невязки.
    int    residual_samples = 16;
    // Радиус зондирования и проверки в пространстве состояний.
    double probe_radius = 1.0;
};

// --- Масштабирование по амплитуде ---------------------------------------------
//
// Схема живёт в вольтах, система — в своих единицах. Для Лоренца из библиотеки
// это ±45 В, то есть за пределами любого питания. Подстановка x_i = S_i*u_i для
// полинома степени <= 2 — прямой пересчёт коэффициентов, а сами S_i не ищутся
// оптимизацией: они ИЗМЕРЯЮТСЯ по прогону ОДУ, который всё равно уже сделан.
//
// Это не вся фаза 5: ряд E24, увязка постоянной времени с полосой ОУ и проверка
// ограничений на номиналы остаются за ней.

struct ScalePlan {
    std::vector<double> s;             // x_i = s[i] * u_i
    double target_volt = 3.0;          // желаемая амплитуда переменной в вольтах
};

// --- Граф схемы ---------------------------------------------------------------

// Узел 0 — земля, всегда присутствует.
struct CircuitNode {
    std::string name;
};

// Проводимость, а не сопротивление: MNA штампует G, а моном с нулевым на
// текущем наборе параметров коэффициентом иначе дал бы деление на ноль.
struct CircuitResistor {
    int         a = 0, b = 0;
    double      siemens = 0.0;
    std::string origin;   // моном, из которого он взялся — для диагностики и netlist'а

    double ohm() const { return siemens != 0.0 ? 1.0 / siemens : 0.0; }  // 0 = разрыв
};

struct CircuitCapacitor {
    int    a = 0, b = 0;
    double farad = 0.0;
};

// Идеальный ОУ на этапе синтеза; модель неидеальностей подставляет решатель.
struct CircuitOpAmp {
    int in_plus = 0, in_minus = 0, out = 0;
};

// AD633: out = (x1 - x2) * (y1 - y2) / 10 + z.
struct CircuitMultiplier {
    int x1 = 0, x2 = 0, y1 = 0, y2 = 0, z = 0, out = 0;
};

// Источник постоянного напряжения — нужен только свободным членам правой части.
struct CircuitVSource {
    int    node = 0;
    double volt = 0.0;
};

struct CircuitGraph {
    std::vector<CircuitNode>       nodes;
    std::vector<CircuitResistor>   resistors;
    std::vector<CircuitCapacitor>  capacitors;
    std::vector<CircuitOpAmp>      opamps;
    std::vector<CircuitMultiplier> multipliers;
    std::vector<CircuitVSource>    vsources;

    // Узел выхода интегратора для каждой переменной состояния (несёт +x_i).
    std::vector<int> var_node;
    // Узел с -x_i, либо -1, если инвертор не понадобился.
    std::vector<int> var_node_inv;

    // 1/(R_unit*C): одна единица времени ОДУ = 1/time_scale секунд схемного времени.
    double time_scale = 0.0;

    int add_node(const std::string& name) {
        nodes.push_back(CircuitNode{ name });
        return (int)nodes.size() - 1;
    }
};

struct SynthesisConfig {
    double r_unit = 1.0e5;    // резистор единичного коэффициента
    double c_int  = 1.0e-8;   // ёмкость интегратора
    // Формировать инверторы для ВСЕХ переменных, а не только для нужных: избыточно
    // по элементам, зато полярность любой ветви доступна без переразводки.
    bool   always_invert = false;
};
