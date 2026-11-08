#pragma once
#include "ucuda/domain/analysis_session.h"
#include "ucuda/domain/network_session.h"
#include "ucuda/domain/neural_types.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

// Neural — большая сеть: популяции и проекции вместо узлов и рёбер. Сессия хранит рецепт (правила, seed),
// граф строится заново при запуске; хеш ловит тихую смену генератора.
// Network остаётся отдельной вкладкой со своим ядром; отсюда берутся законы связи и codegen.

// Версия генератора входит в рецепт: новая версия меняет графы, и старые сессии должны это увидеть.
constexpr int kNeuralGraphVersion = 1;

enum class NeuralRule { Probability = 0, FixedDegree, Chain, Ring, Grid, King, SmallWorld };
// Деление веса дуги: без него, на степень этой проекции, на всю входящую степень узла (как в Network).
enum class NeuralNorm { None = 0, Projection, Total };

struct NeuralPopulation {
    std::string name = "pop";
    std::string size_text = "1000";
    // Значение и половина ширины равномерного разброса вокруг него; пустое = 0.
    std::map<std::string, std::string> param_values, param_spread;
    std::map<std::string, std::string> initial_conditions, ic_spread;
    // Импульсный режим: источник спайков без динамики, спайк с вероятностью rate * h за шаг на [from, to), to = 0 = до конца.
    bool spike_source = false;
    std::string rate_text = "0.1", source_from_text = "0", source_to_text = "0";
};

// Синапс импульсной сети: своя постоянная спада и свой вклад в правую часть, gain * s или gain * s * (E - x).
struct NeuralReceptor {
    std::string name = "syn";
    std::string tau_text = "5";
    int target = 0;                    // переменная узла
    bool conductance = false;
    std::string reversal_text = "0";
    std::string gain_text = "1";       // выражение над параметрами узла
};

// Правило STDP, на него ссылается проекция. a_plus и a_minus в единицах веса дуги.
struct NeuralPlasticRule {
    std::string name = "stdp";
    std::string a_plus_text = "0.01", a_minus_text = "0.012";
    std::string tau_plus_text = "20", tau_minus_text = "20";
    std::string w_min_text = "0", w_max_text = "1";
    std::string mu_plus_text = "0", mu_minus_text = "0";
    bool nearest = false;
};

// Устройство на связи, одна модель на эксперимент. В текстах доступны v, состояние и параметры, params_text = "a = 1, b = 2".
struct NeuralDeviceModel {
    std::string state_name = "m";
    std::string rhs_text, current_text, params_text;
    std::string initial_text = "0.5", min_text = "0", max_text = "1";
};
enum class NeuralDevicePreset { Vteam = 0, Team };
const char* neural_device_preset_name(NeuralDevicePreset p);
NeuralDeviceModel neural_device_preset(NeuralDevicePreset p);

struct NeuralProjection {
    int from = 0, to = 0;
    NeuralRule rule = NeuralRule::FixedDegree;
    std::string p_text = "0.01";       // Probability
    std::string degree_text = "100";   // FixedDegree: входящая степень
    std::string k_text = "1";          // Ring, SmallWorld: соседей в каждую сторону
    std::string width_text = "10";     // Grid, King: высота = размер / ширина
    std::string rewire_text = "0.1";   // SmallWorld
    bool periodic = false;             // Grid, King
    bool directed = false;             // регулярные правила: дуги вместо рёбер
    std::string weight_text = "1";
    std::string weight_spread_text = "0";  // половина ширины относительного разброса
    NeuralNorm norm = NeuralNorm::Total;
    bool norm_expected = false;        // делить на ожидаемую степень, а не на фактическую
    int law = 0;
    int receptor = 0;                  // импульсный режим вместо закона связи
    std::string delay_text = "0";      // кратна h, 0 = один шаг
    int plastic = -1;                  // импульсный режим: правило STDP, -1 = вес постоянный
    // stage: вместо закона связи устройство, gain = вес дуги. one_element: дуга и встречная дуга это один элемент с током в обе стороны;
    // иначе каждая дуга свой элемент с током только в приёмник.
    bool device = false;
    bool one_element = true;
    int device_var = 0;
};

// Свип: значения идут отдельными экземплярами одного запуска, графом из несвязных компонент.
enum class NeuralSweepKind { Coupling = 0, Parameter, ParameterSpread, Replica, InputGain };

struct NeuralSweep {
    bool enabled = false;
    NeuralSweepKind kind = NeuralSweepKind::Coupling;
    int population = -1;               // Parameter, ParameterSpread: -1 = во всех популяциях
    std::string param;
    std::string lo_text = "0", hi_text = "1", points_text = "8";
    bool log = false;
};

// Канал входа. Система из библиотеки хранится снимком, эксперимент от правок библиотеки не зависит.
// Bits: +-amplitude с равной вероятностью на каждый отсчёт входа, для XOR задержанных битов.
enum class NeuralSourceKind { Constant = 0, Sine, Square, Noise, System, Bits };

struct NeuralSource {
    std::string name = "input";
    NeuralSourceKind kind = NeuralSourceKind::Sine;
    std::string amplitude_text = "1", frequency_text = "0.1", phase_text = "0", offset_text = "0";
    std::string system_name;
    std::vector<std::string> vars, params, rhs;
    bool latex = false;                    // rhs записаны LaTeX-именами: \\sigma вместо sigma
    std::map<std::string, std::string> initial_conditions, param_values;
    int variable = 0;
    std::string step_text = "0.01";        // шаг RK4 источника, в его времени
    std::string time_scale_text = "1";     // времени источника на единицу времени сети
    std::string skip_text = "100";         // переходный процесс источника
    bool normalize = true;                 // среднее 0 и СКО 1 по поданным отсчётам
    long long normalize_first = 0;         // считать их только по первым отсчётам, 0 = по всем: задача не должна видеть тест
    // Только в запуске: сохранённый выход видел вход с нормировкой обучения, её и берём. В JSON не пишется.
    bool norm_fixed = false;
    double norm_mean = 0.0, norm_sd = 1.0;
};

// Доля узлов популяции на канал: у одного узла один канал, доли одной популяции не пересекаются.
struct NeuralLink {
    int source = 0, population = 0;
    std::string gain_text = "1", gain_spread_text = "0", fraction_text = "1";
};

enum class NeuralProtocol { Continuous = 0, Trials };

// Задача: по отклику сети на вход восстановить или предсказать переменные той же системы-источника. Отсчёт = отсчёт входа.
// Разбиение по времени: прогрев, обучение, пропуск, валидация, пропуск, тест; нормировка входа только по обучению.
// Классификация: попытка протокола Trials = образец, класс = источник входа. Во время стимула все источники-классы несут
// сигнал класса попытки, у системы это очередной кусок её траектории (между кусками она идёт и в паузе, невидимо).
// Метки сбалансированы в каждой части разбиения и перемешаны data_seed; части идут по времени: обучение, валидация, тест.
// Memory: цель = сам вход source, задержанный на d = 1..max_delay отсчётов, r^2 на тесте, их сумма = memory capacity.
// Xor: цель = XOR знаков входа на отсчётах p - d и p - d - 1, accuracy по порогу 0.5, угадывание 0.5. Обе по времени, как Regression.
enum class NeuralTaskKind { Regression = 0, Classification, Memory, Xor };

struct NeuralTask {
    bool enabled = false;
    NeuralTaskKind kind = NeuralTaskKind::Regression;
    std::vector<int> classes;              // классификация: источники-классы
    std::string bins_text = "1";           // классификация: точек признаков на стимул, последняя в его конце
    std::string data_seed_text = "1";      // классификация: метки и их порядок, не зависит от seed сети
    std::string max_delay_text = "40";     // Memory и Xor: задержки 1..max_delay
    bool control_k0 = false;               // второй прогон той же сети при K = 0: вклад рекуррентных связей
    int source = 0;
    std::vector<int> targets;              // переменные источника
    std::string horizon_text = "0";        // на сколько отсчётов вперёд, 0 = наблюдатель
    std::string washout_text = "1000", gap_text = "250";   // в отсчётах
    std::string train_text = "0.6", val_text = "0.2";      // доли остатка
    std::string filter_text = "20";        // импульсный режим: постоянная фильтра спайков
    std::string features_text = "1000";    // узлов-признаков
    bool use_saved = false;                // применить сохранённый выход ко всей записи после прогрева, без обучения
};

// Обученный выход для применения без обучения. Применение требует того же графа, признаков, целей и шага отсчётов;
// параметры узлов и источник (его начальные условия, параметры) менять можно, на этом и проверяется обобщение.
// y_k = w[k][M] + sum_i w[k][i] (f_i - mean_i) / scale_i, scale = inf у выпавшего признака.
struct NeuralReadoutSaved {
    std::string graph_hash;
    bool spikes = false;                   // признаки: фильтр спайков, иначе состояние узла
    int feature_var = 0;                   // непрерывная сеть: переменная узла
    int features_requested = 0;            // поле feature nodes на обучении
    double filter = 0.0;                   // импульсный режим: постоянная фильтра, время сети
    double sample_dt = 0.0;                // отсчёт выхода, время сети
    int source = 0, horizon = 0;
    std::vector<int> targets;
    std::vector<std::string> names;
    double input_mean = 0.0, input_sd = 1.0;   // нормировка входа на обучении
    std::vector<int> nodes;                // [M] узлы-признаки
    int M = 0, K = 0;
    std::vector<double> mean, scale;       // [M]
    std::vector<double> w;                 // [K][M + 1]
    std::vector<double> lambda, val, test; // [K], на обучении
    bool empty() const { return w.empty(); }
};

// Свип: у каждого экземпляра свой выход на одном разбиении, вход и цели у экземпляров общие.
struct NeuralTaskResult {
    bool applied = false;                  // сохранённый выход без обучения: val, delay_test пусты, test по записи после прогрева
    NeuralReadoutSaved model;              // обученный в этом расчёте, пусто при applied и в свипе
    int instances = 1;
    std::vector<std::string> names;        // [targets]
    std::vector<double> val, test, lambda; // [instance][target], NRMSE; NaN у остановленного или не обученного экземпляра
    std::vector<double> delay_test;        // [targets], ridge по задержкам входа, длина выбрана на валидации
    int delay_length = 0, features = 0;
    double t0 = 0.0, dt = 0.0;             // тест в единицах времени сети
    int points = 0;
    std::vector<double> truth;             // [point][target], тест
    std::vector<double> predicted;         // [instance][point][target]
    std::string note;                      // первый отказ экземпляра свипа
    // Классификация: names = {"accuracy"}, val/test это доли верных ответов [instance], points = попыток теста,
    // truth = их классы, predicted = ответ [instance][попытка], delay_test = ridge по сигналу стимула, delay_length его точек.
    bool classify = false;
    bool by_delay = false;                 // Memory и Xor: цели это задержки, окно Neural Readout рисует кривую по d
    bool higher_better = false;            // accuracy, r^2 и MC: лучшая сеть свипа с наибольшим значением
    std::string metric = "NRMSE";          // приставка к имени цели в окнах, пусто = имя цели уже полное
    std::vector<double> control_test, control_val;   // [target], та же сеть при K = 0; пусто = контроль выключен
    std::vector<std::string> class_names;  // [C]
    std::vector<double> confusion;         // [instance][истинный][ответ], тест, число попыток
    double chance = 0.0;                   // доля самого частого класса в тесте
};

struct NeuralConfig {
    std::string label = "Experiment";
    NeuralSchedule schedule = NeuralSchedule::Stage;
    std::string scheme = "RK4";        // шаг узла post/heun; stage всегда RK4
    std::string symmetry_s = "0.5";
    std::string cd_fp_iters = "4";
    NewtonSettings newton;

    std::string h_text = "0.01";
    std::string t_max_text = "100";
    std::string transient_text = "0";
    std::string pre_scaller_text = "10";
    std::string max_value_text = "1e6";

    std::string coupling_text = "0.1";
    std::string seed_text = "1";
    std::string observed_per_pop_text = "8";
    int order_var = 0;                 // -1 = R не считается
    std::string order_threshold_text = "0";
    std::vector<int> obs_vars_off;     // не записывать эти переменные
    std::string output_limit_text = "1024";   // МБ на хосте
    NeuralSweep sweep;
    NeuralSweep sweep2;                // вторая ось, значит только при включённой первой

    // Клон всей сети: память и время расчёта вдвое. Сид клона = seed эксперимента.
    bool lyap_on = false;
    std::string lyap_eps_text = "1e-8";       // на координату
    std::string lyap_interval_text = "1";
    std::string lyap_settle_text = "100";

    // Сводка свипа пишется в файл после каждого расчёта.
    bool csv_save_enabled = false;
    std::string csv_output_path;

    std::vector<NeuralSource> sources;
    std::vector<NeuralLink> links;
    int input_var = 0;
    std::string input_hold_text = "0";  // время удержания отсчёта, кратно h; 0 = каждый шаг

    // Попытки идут после transient: пауза, затем стимул. В паузе вход 0, время стимула отсчитывается заново.
    NeuralProtocol protocol = NeuralProtocol::Continuous;
    std::string trials_text = "4", rest_text = "50", stimulus_text = "50";
    bool reset_state = true;           // каждая попытка с initial conditions

    std::vector<NetCouplingLaw> laws;
    std::vector<NeuralPopulation> populations;
    std::vector<NeuralProjection> projections;

    // Импульсный режим: X[threshold_var] >= threshold, затем сброс; reset_text[k] пусто = переменная не меняется.
    int threshold_var = 0;
    std::string threshold_text = "1";
    std::vector<std::string> reset_text;
    std::string refractory_text = "0";
    std::vector<NeuralReceptor> receptors;
    std::vector<NeuralPlasticRule> plastic_rules;
    NeuralDeviceModel device;
    NeuralTask task;
    NeuralReadoutSaved readout;            // сохранённый кнопкой Keep, пишется в сессию
    std::string readout_path;              // файл экспорта

    // Снимок конца расчёта живёт только в памяти, пока его не сохранили в файл.
    bool keep_checkpoint = false;
    std::string checkpoint_path;
    std::shared_ptr<NeuralCheckpoint> checkpoint;

    // Рецепт, по которому получен graph_hash, и система на момент расчёта.
    std::string graph_recipe, graph_hash, model_text;

    NeuralResult result;
    std::vector<int>    instance_start;   // последнего расчёта
    std::vector<double> sweep_values;     // значения первой оси, пусто без свипа
    std::vector<double> sweep2_values;    // значения второй оси, пусто = одна ось; экземпляр = i1 * n2 + i2
    std::vector<double> sweep_order, sweep_spread;  // [instance][pop], среднее по времени, NaN у разлетевшихся
    std::vector<double> sweep_lyap;       // [instance]
    std::vector<double> input_view;       // [point][source], прореженный вход первого экземпляра
    double input_view_t0 = 0.0, input_view_dt = 0.0;
    int pops_per_instance = 0;
    std::vector<int> pop_start;           // [pops + 1] последнего расчёта
    std::vector<double> pop_rate;         // импульсный режим: спайков на узел в единицу времени
    int watched_per_instance = 0;         // столбцов weight_trace или device_trace на экземпляр
    std::vector<std::string> watch_labels;   // подписи этих столбцов
    std::vector<double> weight_summary;   // [instance][3]: min, среднее, max пластичных дуг в конце
    NeuralTaskResult task_result;
    long long arcs = 0;
    int degree_min = 0, degree_max = 0;
    double build_seconds = 0.0;
    bool last_run_ok = false;
    std::string last_error;
    bool hash_mismatch = false;
    int data_generation = 0;
};

// CSR по приёмнику в канонической нумерации: популяции подряд, в популяции по номеру.
struct NeuralGraph {
    int N = 0;
    std::vector<int> pop_start;
    std::vector<int> start, src, law;
    std::vector<int> delay;            // только импульсный режим, в шагах
    std::vector<int> rule;             // только импульсный режим, -1 = постоянный вес
    std::vector<double> w;
    std::vector<int> dev_a, dev_b, dev_var, dev_two_way;   // устройства: их дуг в CSR нет
    std::vector<double> dev_gain;
    std::string hash;
    int degree_min = 0, degree_max = 0;
};

// Готовый запуск: граф и запрос всех экземпляров. Без свипа это один граф и его же хеш.
struct NeuralPlan {
    NeuralRequest request;
    std::string hash;
    std::vector<std::string> instance_hashes;   // граф каждой сети свипа: сохранённый выход сверяется с каждой
    long long arcs = 0;
    int degree_min = 0, degree_max = 0;
    int pops_per_instance = 0;
    int watched_per_instance = 0;
    std::vector<std::string> watch_labels;
    std::vector<double> sweep_values, sweep2_values;
    std::vector<double> input_view;
    double input_view_t0 = 0.0, input_view_dt = 0.0;
};

struct NeuralOutcome {
    NeuralResult result;
    std::vector<int>    instance_start;
    std::vector<double> sweep_values, sweep2_values, sweep_order, sweep_spread, sweep_lyap, input_view;
    double input_view_t0 = 0.0, input_view_dt = 0.0;
    int pops_per_instance = 0;
    std::vector<int> pop_start;
    std::vector<double> pop_rate;
    int watched_per_instance = 0;
    std::vector<std::string> watch_labels;
    std::vector<double> weight_summary;
    NeuralTaskResult task_result;
    bool resumed = false;
    std::string hash, recipe;
    long long arcs = 0;
    int degree_min = 0, degree_max = 0;
    double build_seconds = 0.0;
    bool hash_mismatch = false;
};

struct NeuralState {
    std::vector<std::string> vars, params;
    System sys;
    std::vector<CustomScheme> custom_schemes;
    std::vector<std::string>  wrapper_schemes;
    std::vector<std::string>  enabled_builtin_schemes;
    std::string loaded_system_name;

    std::vector<NeuralConfig> configs;
    int active_config_index = 0;
    int running_config_index = -1;


    std::string last_run_label;
    double last_run_seconds = 0.0;
    bool   last_run_succeeded = false;


};
