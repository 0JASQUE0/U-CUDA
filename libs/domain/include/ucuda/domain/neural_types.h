#pragma once
// Запрос и результат движка больших сетей (kernels/neural.template.cu, ParametricEngine::run_neural).
// Как у Network, топология приходит развёрнутой в CSR по приёмнику, движок графов не знает.

#include <atomic>
#include <memory>
#include <string>
#include <vector>

// Номера совпадают с NEURAL_SCHED в шаблоне.
enum class NeuralSchedule { Old = 0, Post = 1, Heun = 2, Stage = 3, Spike = 4 };

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
    double settle = 100.0;     // время выстраивания направления: до записи, при обратной связи — после её включения
    unsigned long long seed = 1;
    std::vector<unsigned long long> instance_seed;   // [instances], пусто = seed у всех
};

// STDP на дуге: следы pre и post спадают точной экспонентой, вес ограничен [w_min, w_max] после каждого обновления.
// mu = 0 аддитивное правило, mu = 1 мультипликативное. nearest: след равен 1 после спайка, а не копится.
// gain: множитель обновления, место для третьего фактора.
struct NeuralStdpRule {
    double a_plus = 0.0, a_minus = 0.0;      // оба >= 0: потенциация и депрессия
    double tau_plus = 20.0, tau_minus = 20.0;
    double w_min = 0.0, w_max = 1.0;
    double mu_plus = 0.0, mu_minus = 0.0;
    double gain = 1.0;
    bool   nearest = false;
};

// Импульсная сеть, schedule = Spike. Шаг только RK4, связь только спайками. Последние receptors переменных X это
// синапсы: дуга прибавляет к ним вес, а между спайками они спадают по точной экспоненте. Правая часть читает их как
// обычные переменные, поэтому токовый и проводимостный синапс различаются только текстом правой части.
// Спайк шага n (индекс с 0) это состояние времени (n + 1) h, а дуга с задержкой d даёт вклад в начале шага n + d.
// Дуга: edge_law = номер синапса, edge_w = вес, delay в шагах >= 1.
// Пластичную дугу пишет только поток приёмника. Спайк приёмника конца шага m и приход начала шага m + 1 это один момент:
// сначала спайк приёмника, затем приход, поэтому при равных временах дуга получает депрессию.
struct NeuralSpiking {
    int receptors = 0;                // не больше 8
    std::vector<double> tau;          // [receptors]
    std::string spike_body;           // codegen_spike_body
    std::vector<int> refractory_steps; // [n_nodes], столько шагов узел держится после сброса, синапсы спадают; пусто = 0
    std::vector<int> edge_delay;      // [arcs]

    // Источник спайков: узел без динамики, спайк с вероятностью p за шаг на шагах [from, to). p < 0 = обычный узел.
    std::vector<double> source_p;     // [n_nodes], пусто = источников нет
    std::vector<int> source_from, source_to;
    unsigned long long seed = 1;
    std::vector<unsigned long long> source_seed;   // [n_nodes], neural_source_seed; пусто = от seed и номера узла

    std::vector<NeuralStdpRule> rules;
    std::vector<int> edge_rule;       // [arcs], -1 = статическая дуга; пусто = пластичности нет
    std::vector<int> watch_arcs;      // номера дуг запроса, их веса пишутся на сетке записи

    int window = 0;                   // шагов на запуск, 0 = авто; не больше наименьшей задержки
    int log_capacity = 0;             // записей журнала на чанк, 0 = авто; лишние отбрасываются, счётчики узлов точны
};

// Сид источника от номера узла внутри экземпляра: значение свипа = отдельный запуск.
inline unsigned long long neural_source_seed(unsigned long long seed, long long node) {
    const auto mix = [](unsigned long long z) {
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    return mix(mix(seed + 0x9E3779B97F4A7C15ULL) ^ ((unsigned long long)node + 1ULL) * 0xC2B2AE3D27D4EB4FULL);
}

// Устройство на связи, только расписание stage: элемент между узлами a и b на переменной var, v = x_a - x_b, ток I уходит
// из a и приходит в b, в правую часть он входит с gain. Состояния идут в тех же стадиях RK4, что и узлы. two_way = 0: ток только в b,
// a не нагружен. Пишет только поток узла b. Тела печатает codegen_rhs_body по системе с параметрами [v, p...]:
// rhs_body даёт производные состояний, current_body даёт F[0] = I. Переживает сброс попытки, как вес.
struct NeuralDevices {
    int states = 0;                   // состояний на устройство, 0 = устройств нет, не больше 4
    int params = 0;
    std::string rhs_body, current_body;
    std::vector<int> a, b, var, two_way;   // [devices]
    std::vector<double> gain;         // [devices]
    std::vector<double> values;       // [devices * params]
    std::vector<double> initial;      // [devices * states]
    std::vector<double> s_min, s_max; // [states], жёсткие границы в конце шага
    std::vector<int> watch;           // устройства, их состояния пишутся на сетке записи
};

// Снимок расчёта на границе чанка. Случайные числа и вход зависят только от шага и seed, поэтому своего состояния у них нет.
// shape описывает форму запроса: другая форма отказывает. Параметры можно менять, например заморозить обучение.
struct NeuralCheckpoint {
    long long step = 0;
    std::string shape;
    std::vector<char> data;
};

// Замкнутый выход: с строки from_row вход каналов channels считает сама сеть, y_k = w[k][M] + sum_i w[k][i] * row[cols_i],
// где row это строка записи точки p (после шага p * hold_steps), а результат идёт в строку входа p, до её шагов.
// Только непрерывные расписания, stage_rows = 1, pre_scaller = hold_steps, без transient.
// Для показателя Ляпунова каждый клон вычисляет свою строку входа.
struct NeuralFeedback {
    long long from_row = -1;          // -1 = вход задан целиком
    std::vector<int> cols;            // [M], номера в строке записи
    std::vector<int> channels;        // [K]
    std::vector<double> w;            // [K * (M + 1)], последний столбец свободный член
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
    NeuralSpiking     spike;
    NeuralDevices     devices;
    NeuralFeedback    feedback;
    std::shared_ptr<const NeuralCheckpoint> resume;   // продолжить с этой точки; R и показатель Ляпунова несовместимы
    // Входит в форму точки: хеш графа, сеть того же размера с другим графом отказывает. Часть после '|' описывает запись:
    // сессия дополняет ею продолжение, и при другом прореживании склейка вышла бы неверной.
    std::string checkpoint_tag;
    bool keep_checkpoint = false;     // отдать снимок конца расчёта
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

    // Журнал спайков записи, по шагу и узлу. Переполнение чанка или лимит вывода режут журнал, но не динамику.
    std::vector<long long> spike_step;
    std::vector<int>       spike_node;
    std::vector<int>       spike_count;    // [n_nodes], за запись, точно
    long long spikes_dropped = 0;
    std::vector<double>    weight_trace;   // [point][watch], после остановки экземпляра NaN
    std::vector<double>    final_weights;  // [arcs] в порядке запроса; пусто без пластичности

    std::vector<double>    device_trace;   // [point][watch][state]
    std::vector<double>    final_devices;  // [devices * states]
    long long              device_clips = 0;   // сколько раз сработала граница состояния
    std::shared_ptr<NeuralCheckpoint> checkpoint;
    std::vector<double>    feedback_rows;  // [строка - from_row][K], поданный вход; после остановки строки не считались

    long long launches = 0, syncs = 0, steps_done = 0;
    int block_size = 0;
    long long steps_per_sync = 0;
    double wall_ms = 0.0;
    size_t device_bytes = 0;
};
