#pragma once
#include "analysis_session.h"
#include "system_record.h"
#include "parametric_engine.h"
#include <atomic>
#include <chrono>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Network — вкладка сетей связанных осцилляторов.
//
// Сеть — это N копий ОДНОЙ системы (той, что выбрана в библиотеке) со своими
// параметрами и начальными условиями у каждого узла, плюс рёбра, по которым
// узлы тянут друг друга. Шаг расщеплён: узел делает свой шаг телом КРС, потом
// к нему добавляется h * (вклад связи) — см. kernels/network.template.cu.
// Поэтому на вкладке работают ВСЕ схемы, включая неявные и пользовательские,
// но сама связь проинтегрирована первым порядком.
//
// Что где живёт:
//   NetworkConfig  — всё, что задаёт расчёт: топология, законы связи, узлы,
//                    рёбра, интегрирование. Сохраняется в сессию целиком.
//   NetworkSession — список конфигов + асинхронный прогон, как у Order.
// Настройки ОТОБРАЖЕНИЯ (какой узел на кривой, какая переменная в раскраске)
// живут на окнах графиков в AppModel, а не здесь.

// Встроенные топологии. Custom — «не трогать рёбра генератором»: пользователь
// собрал их руками. Любая другая после генерации тоже правится руками, и
// первая же ручная правка переводит конфиг в Custom — иначе следующий вызов
// генератора молча стёр бы правки.
enum class NetTopology {
    Custom = 0,
    Chain,      // цепь: i - (i+1)
    Ring,       // кольцо с k соседями в каждую сторону
    Star,       // звезда: узел 0 связан со всеми
    Grid,       // прямоугольная решётка, 4 соседа
    King,       // та же решётка + диагонали (королевский граф), 8 соседей
    AllToAll,   // полный граф
    SmallWorld, // Уоттс-Строгац: кольцо с k соседями + перепривязка с вероятностью p
};

// Закон связи: по выражению на каждую переменную системы. Пустая строка =
// в это уравнение связь не входит. Имена в выражении разбирает
// codegen_coupling (см. codegen.hpp): x_j — состояние соседа, x или x_i — своё,
// K или w — вес ребра.
struct NetCouplingLaw {
    std::string name = "diffusive";
    std::vector<std::string> expr;   // размер = числу переменных системы
};

struct NetNode {
    std::string label;
    // Переопределения. Пустая строка (или отсутствие ключа) = берётся общее
    // значение конфига: расстройка задаётся точечно, а не копированием всего
    // набора параметров в каждый узел.
    std::map<std::string, std::string> param_values;
    std::map<std::string, std::string> initial_conditions;
    // Положение в редакторе графа, в долях холста [0..1]. Генераторы
    // расставляют его сами, руками таскается мышью.
    float ui_x = 0.5f, ui_y = 0.5f;
};

struct NetEdge {
    int  from = 0;                  // источник (его состояние читают)
    int  to   = 0;                  // приёмник (в его правую часть идёт вклад)
    bool bidirectional = true;      // ребро = две дуги
    std::string weight_text = "1";  // множится на общий coupling_text
    int  law = 0;                   // индекс в NetworkConfig::laws
};

struct NetworkConfig {
    std::string label  = "Network";
    std::string scheme = "Euler";
    std::string symmetry_s = "0.5";

    // ---- Интегрирование ----
    std::string h_text           = "0.01";
    std::string t_max_text       = "100";
    std::string transient_text   = "0";
    std::string pre_scaller_text = "1";
    std::string max_value_text   = "1e6";
    // Потолок записанных точек по времени: движок поднимет прореживание, если
    // при заданном не влезает, и вернёт фактическое значение.
    std::string max_points_text  = "20000";

    // ---- Связь ----
    // Общий множитель: домножает вес КАЖДОГО ребра. Отдельное поле, потому что
    // «покрутить силу связи целиком» — самая частая операция на вкладке, и
    // ради неё не должно приходиться править сотню весов.
    std::string coupling_text = "0.1";
    std::vector<NetCouplingLaw> laws;

    // ---- Топология ----
    NetTopology topology = NetTopology::Ring;
    std::string gen_n_text = "16";    // узлов (chain/ring/star/all-to-all/small world)
    std::string gen_k_text = "1";     // соседей в каждую сторону (ring/small world)
    std::string gen_w_text = "4";     // ширина решётки (grid/king)
    std::string gen_h_text = "4";     // высота решётки (grid/king)
    std::string gen_p_text = "0.1";   // вероятность перепривязки (small world)
    std::string gen_seed_text = "1";
    bool        gen_periodic = false; // замыкать решётку в тор
    bool        gen_directed = false; // генерировать дуги вместо рёбер

    std::vector<NetNode> nodes;
    std::vector<NetEdge> edges;

    // ---- Общие значения параметров и НУ ----
    // База для всех узлов; узел переопределяет только то, что ему нужно.
    std::map<std::string, std::string> param_values;
    std::map<std::string, std::string> initial_conditions;
    // Разброс НУ по узлам: к общему значению каждой переменной добавляется
    // равномерный шум в [-spread, +spread]. Нужен, потому что из ИДЕНТИЧНЫХ
    // начальных условий диффузионно связанная сеть стартует уже
    // синхронизованной и такой и останется — вклад связи тождественно ноль.
    std::string ic_spread_text = "0.1";
    std::string ic_seed_text   = "1";

    // ---- Результат ----
    NetworkResult result;
    bool        last_run_ok = false;
    std::string last_error;
    int         data_generation = 0;
};

struct NetworkSession {
    std::vector<std::string> vars;
    std::vector<std::string> params;
    System sys;
    std::vector<CustomScheme> custom_schemes;
    std::vector<std::string>  wrapper_schemes;
    std::vector<std::string>  enabled_builtin_schemes;
    std::string loaded_system_name;

    std::vector<NetworkConfig> configs;

    int active_config_index  = 0;
    int running_config_index = -1;

    std::future<NetworkResult> run_future;
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;

    std::shared_ptr<std::atomic<bool>>  cancel_token;
    std::shared_ptr<std::atomic<float>> progress_token;

    std::string last_run_label;
    double last_run_seconds = 0.0;
    bool   last_run_succeeded = false;
    std::chrono::steady_clock::time_point last_run_completed_at;

    NetworkSession() = default;
    NetworkSession(NetworkSession&&) = default;
    NetworkSession& operator=(NetworkSession&&) = default;
    NetworkSession(const NetworkSession&) = delete;
    NetworkSession& operator=(const NetworkSession&) = delete;

    void load_from_record(const SystemRecord& r,
                          const std::vector<std::string>& vars_,
                          const std::vector<std::string>& params_);

    void add_config();
    void remove_config(int i);

    bool run_async(ParametricEngine& engine, int config_idx);
    void request_cancel();
    bool poll();
};

// Закон связи по умолчанию для системы размерности n_vars: диффузионная связь
// по ПЕРВОЙ переменной, K*(x_j - x_i). Ровно то, с чего начинают в 99%
// постановок, и единственный вариант, который можно предложить, не зная
// смысла переменных.
NetCouplingLaw net_default_law(const std::vector<std::string>& vars);

// Перестраивает nodes/edges под текущие настройки генератора. Существующие
// переопределения параметров сохраняются для узлов, которые остались (по
// индексу): смена топологии не должна стирать расстройку ансамбля.
// Возвращает текст ошибки (пусто = успех).
std::string net_generate_topology(NetworkConfig& c, const std::vector<std::string>& vars);

// Раскладывает рёбра в дуги и считает CSR по узлу-приёмнику. Веса уже
// домножены на общий множитель связи. Используется и движком, и редактором
// (чтобы показать степень узла).
void net_build_csr(const NetworkConfig& c, double coupling,
                   std::vector<int>& edge_start, std::vector<int>& edge_src,
                   std::vector<double>& edge_w, std::vector<int>& edge_law);

// Готовое тело switch(law) для ядра. Бросает std::runtime_error с текстом от
// кодогена, если выражение закона не разобралось.
std::string net_coupling_body(const System& sys, const std::vector<NetCouplingLaw>& laws);

// Человекочитаемое имя топологии (для комбо и подписей).
const char* net_topology_name(NetTopology t);
