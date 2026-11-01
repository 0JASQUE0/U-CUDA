#pragma once
// Запрос и результат движка больших сетей (kernels/neural.template.cu, ParametricEngine::run_neural).
// Как у Network, топология приходит развёрнутой в CSR по приёмнику, движок графов не знает.

#include <atomic>
#include <memory>
#include <string>
#include <vector>

// Номера совпадают с NEURAL_SCHED в шаблоне.
enum class NeuralSchedule { Old = 0, Post = 1, Heun = 2, Stage = 3 };

// Внешнее воздействие: отсчёты держатся hold_steps шагов, после конца держится последний.
// Вход складывается со связью, поэтому попадает в каждую стадию stage. Границы удержания кратны шагу.
// stage_rows = 3 (только stage и hold_steps = 1): на шаг три строки t, t + h/2, t + h, стадии берут 0, 1, 1, 2.
struct NeuralInput {
    int channels = 0;                 // 0 = входа нет
    int var = 0;                      // переменная узла, в правую часть которой идёт вход
    long long hold_steps = 1;
    int stage_rows = 1;
    std::vector<double> samples;      // [rows * stage_rows * channels]
    std::vector<int>    node_channel; // [n_nodes], -1 = узел не подключён
    std::vector<double> node_gain;    // [n_nodes]
};

// Вместо полной истории: выбранные узлы и средние по популяциям. Популяции идут подряд по номерам узлов.
struct NeuralObservation {
    std::vector<int> nodes;           // выбранные узлы
    std::vector<int> vars;            // их переменные и переменные средних; пусто = ничего не пишем
    std::vector<int> pop_start;       // [pops + 1]; пусто = без средних
    bool spread = false;              // СКО по узлам популяции для тех же переменных

    // R(t) по популяциям. Фаза узла растёт на 2 pi линейно между соседними пересечениями порога снизу вверх,
    // поэтому до первого и после последнего пересечения её нет, а R считается в конце запуска. Пересечения ищутся
    // и во время transient, на той же сетке, поэтому после него R есть с первой точки записи.
    // Узел, замолчавший до конца, понижает R; точка, где фазы нет хотя бы у одного из остальных, остаётся NaN:
    // неполный набор фаз смещает R.
    int    order_var = -1;            // -1 = без R
    double order_threshold = 0.0;
};

// Старший показатель Ляпунова (Бенеттин): клон узла i лежит в узле N + i, с теми же параметрами, входом и связями.
// Каждый экземпляр нормируется сам: начальная и восстанавливаемая норма eps*sqrt(узлы * переменные), то есть eps на
// координату. Клон на основную сеть не влияет. Клона, разошедшегося до разлёта, основная сеть разделяет: экземпляр
// останавливается целиком.
struct NeuralLyapunov {
    bool   enabled = false;
    double eps = 1.0e-8;
    double interval = 1.0;     // время между перенормировками, кратно h
    double settle = 100.0;     // столько времени до записи направление выстраивается, сумма не копится
    unsigned long long seed = 1;
    std::vector<unsigned long long> instance_seed;   // [instances], пусто = seed у всех
};

struct NeuralRequest {
    NeuralSchedule schedule = NeuralSchedule::Stage;
    std::string krs_body;             // шаг узла; stage его не зовёт, но шаблону он нужен
    std::string rhs_body;             // codegen_rhs_body, обязателен для stage (RK4)
    std::string coupling_body;        // net_coupling_body: case-ветки по номеру закона
    int n_laws = 1;
    int amountOfX = 0;
    int amountOfValues = 0;
    int n_nodes = 0;

    std::vector<double> values;              // [n_nodes * amountOfValues]
    std::vector<double> initial_conditions;  // [n_nodes * amountOfX], узел за узлом
    std::vector<int>    edge_start;          // [n_nodes + 1]
    std::vector<int>    edge_src;
    std::vector<double> edge_w;
    std::vector<int>    edge_law;            // можно пустым при n_laws == 1

    // Несвязные сети одного запуска: узлы подряд, [instances + 1], пусто = одна сеть. Дуга между экземплярами
    // и популяция на их границе отказывают. Разлетевшийся экземпляр замирает, остальные считаются дальше.
    std::vector<int>    instance_start;

    // Шаги, на которых состояние возвращается к initial_conditions, по возрастанию. Шаг чанка под них подгоняется,
    // поэтому от длины чанка результат не зависит. С R несовместимо: фаза через сброс смысла не имеет.
    std::vector<long long> reset_at;

    double h = 0.01;
    double t_max = 100.0;
    double transient = 0.0;
    int    pre_scaller = 1;
    double max_value = 1.0e6;

    NeuralInput       input;
    NeuralObservation observe;
    NeuralLyapunov    lyapunov;
    size_t max_output_bytes = size_t(1) << 30;   // лимит записанного на хосте

    int block_size = 0;                // 0 = kNeuralDefaultBlock
    long long steps_per_sync = 0;      // 0 = авто; только сокращает, для проверок чанков
    bool keep_final_state = false;

    std::shared_ptr<std::atomic<bool>>  cancel;
    std::shared_ptr<std::atomic<float>> progress;
};

struct NeuralResult {
    bool ok = false;
    bool cancelled = false;
    bool diverged = false;
    std::string error;

    int n_nodes = 0, n_vars = 0, n_points = 0;
    double h = 0.0, t0 = 0.0, dt = 0.0;

    // Точки после остановки NaN: у экземпляра с шага разлёта, он от длины чанка не зависит.
    // У post и heun наблюдается состояние до связи.
    std::vector<int>    observed;      // номера выбранных узлов
    std::vector<int>    observed_vars; // записанные переменные: var ниже это номер в этом списке
    std::vector<double> traces;        // [point][selected][var]
    std::vector<double> means;         // [point][pop][var]
    std::vector<double> spread;        // [point][pop][var]
    std::vector<double> order;         // [point][pop], NaN = ни у одного узла нет фазы
    std::vector<double> final_state;   // [n_nodes * amountOfX], узел за узлом
    std::vector<int>    status;        // [n_nodes], 1 = разлетелся
    int n_diverged = 0;
    std::vector<long long> instance_stop;  // [instances], шаг остановки, -1 = дошёл до конца

    std::vector<double> lyapunov;      // [instances], NaN у остановленных; -inf = клон слился с сетью до последнего бита
    std::vector<double> lyap_log;      // [block][instance], log(d / d0) за блок записи
    double lyap_dt = 0.0;              // длина блока
    int    lyap_blocks = 0;

    long long launches = 0, syncs = 0, steps_done = 0;
    int block_size = 0;
    long long steps_per_sync = 0;
    double wall_ms = 0.0;
    size_t device_bytes = 0;
};
