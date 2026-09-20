#pragma once
#include <string>
#include <vector>
#include "codegen.hpp"   // System, SystemEvaluator

// Синтез аналоговой схемы на ОУ по системе ОДУ (фаза 1 плана).
//
// Работает в два шага: правые части раскладываются на мономы степени <= 2, по
// мономам строится граф схемы. Номиналы получаются из коэффициентов по правилу
// R = R_unit / |k|, масштабирование переменных — задача более поздней фазы.

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

// Снимает мономы правых частей при ОДНОМ наборе параметров.
//   values — массив в раскладке a[]: values[0] зарезервирован, values[1..M] — параметры.
// false + err, если правая часть не полином степени <= 2 (невязка выше порога)
// либо если система содержит недифференцируемые функции.
bool extract_quadratic(const System& sys,
                       const std::vector<double>& values,
                       const PolyExtractConfig& cfg,
                       std::vector<PolyRhs>& out,
                       std::string* err);

// То же, но по НЕСКОЛЬКИМ наборам параметров: структура берётся объединением
// ненулевых мономов, числа — из values_sets[0].
//
// Зачем: коэффициент, обнулившийся именно на текущем наборе (sigma = 0 и т.п.),
// иначе выкинул бы элемент из топологии, и правка параметра в UI его бы не
// вернула — пересинтез topology_generation не бампается.
bool extract_quadratic_union(const System& sys,
                             const std::vector<std::vector<double>>& values_sets,
                             const PolyExtractConfig& cfg,
                             std::vector<PolyRhs>& out,
                             std::string* err);

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

// Масштабы из размахов траектории: traj[шаг][координата], переходный участок
// вызывающий отбрасывает сам. Вырожденную переменную (размах ~0) оставляет
// единичной, иначе делили бы на ноль.
bool plan_amplitude_scaling(const std::vector<std::vector<double>>& traj,
                            double target_volt, ScalePlan& out, std::string* err);

// Пересчёт коэффициентов под масштаб: du_i/dt = f_i(S*u) / S_i.
void apply_amplitude_scaling(const std::vector<PolyRhs>& in, const ScalePlan& sp,
                             std::vector<PolyRhs>& out);

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

    int add_node(const std::string& name);
};

struct SynthesisConfig {
    double r_unit = 1.0e5;    // резистор единичного коэффициента
    double c_int  = 1.0e-8;   // ёмкость интегратора
    // Формировать инверторы для ВСЕХ переменных, а не только для нужных: избыточно
    // по элементам, зато полярность любой ветви доступна без переразводки.
    bool   always_invert = false;
};

// Строит граф по разложенным правым частям. false + err на нереализуемом
// мономе (степень 3 появиться не может — её отсеивает extract_*).
bool synthesize_circuit(const System& sys,
                        const std::vector<PolyRhs>& rhs,
                        const SynthesisConfig& cfg,
                        CircuitGraph& out,
                        std::string* err);

// --- Диагностика и сравнение --------------------------------------------------

// Канонический текстовый слепок топологии: имена узлов нормализованы, элементы
// отсортированы. Две схемы одинаковы тогда и только тогда, когда слепки равны.
std::string circuit_topology_signature(const CircuitGraph& g);

// Человекочитаемый дамп со всеми номиналами — для отладочной панели и логов.
std::string circuit_dump(const CircuitGraph& g);

// Проверки, которые дешевле сделать здесь, чем ловить вырожденной матрицей в
// решателе: висячий узел, узел без пути на землю, выход ОУ, никуда не идущий.
bool circuit_check(const CircuitGraph& g, std::vector<std::string>& problems);

// Прогон синтезатора на масштабированном Лоренце с проверкой против
// tests/fixtures/lorenz_reference.cir. Возвращает отчёт; ok — сошлось ли.
std::string circuit_synth_selftest(bool* ok);
