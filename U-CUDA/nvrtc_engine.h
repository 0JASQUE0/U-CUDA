#pragma once
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "adaptive_settings.h"

// Фазовые портреты с адаптивным шагом (драйвер kernels/ucuda_adaptive.cuh).
// Поток на траекторию, как у phase_kernel. Транзиент кончается ровно в t_skip
// (последний его шаг обрезается), дальше — либо равномерная сетка через плотный
// выход (total точек с шагом dt, первая — в t_skip), либо сами узлы шага.
struct PhaseAdaptiveRequest {
    std::string rhs, emb, dprep, deval;   // тела AdaptiveCode
    std::string ctrl_body;                // C body пользовательского регулятора, иначе пусто
    int    amountOfX = 0;
    std::vector<double> ic_flat;          // [N*amountOfX]
    int    N = 0;
    std::vector<double> values;           // a[], общие для всех траекторий
    UcudaAdaptParams params{};
    bool   raw = false;                   // узлы шага вместо равномерной сетки
    double t_skip = 0;                    // транзиент
    double t_rec = 0;                     // время записи после транзиента
    double dt = 0;                        // равномерная сетка: шаг вывода
    int    total = 0;                     // равномерная сетка: число точек
    int    max_pts = 0;                   // узлы шага: потолок числа узлов
    int    log_cap = 0;                   // лог попыток на траекторию (0 — не вести)
    // Отмена: ядро смотрит флаг раз в 1024 принятых шага (флаг в mapped-памяти, его ставит
    // поток ожидания, пока ядро идёт). nullptr — без отмены. Отменённый расчёт возвращает
    // false с error() == kNvrtcCancelled.
    std::shared_ptr<std::atomic<bool>> cancel;
};

// Текст error() отменённого расчёта (PhaseAdaptiveRequest::cancel, AdaptiveEndpointRequest::cancel).
inline const char* const kNvrtcCancelled = "Cancelled by user";

// Замер адаптивного шага (Order -> Performance): replicas одинаковых нитей интегрируют
// [0, T] от ic и останавливаются ровно в T; время ОДНОГО запуска меряет само ядро (%globaltimer
// вокруг init и цикла шагов), как у perfIntegrateKernel постоянного шага. Плотного выхода нет (UCUDA_AD_NO_DENSE),
// состояние в регистрах (UCUDA_AD_STATIC_N) — ядро меряет сам шаг, а не запись.
struct AdaptiveEndpointRequest {
    std::string rhs, emb;                 // тела AdaptiveCode (dprep/deval не нужны)
    std::string ctrl_body;                // C body пользовательского регулятора, иначе пусто
    int    amountOfX = 0;
    std::vector<double> ic;               // [amountOfX]
    std::vector<double> values;           // a[]
    UcudaAdaptParams params{};
    double T = 0;
    int    replicas = 1, repeats = 1, warmup = 0;
    std::shared_ptr<std::atomic<bool>> cancel;   // см. PhaseAdaptiveRequest::cancel
};

struct AdaptiveEndpointResult {
    std::vector<double> y_end;            // y(T) нити 0
    AdaptiveStats stats;                  // нити 0
    double t_min = 0, t_avg = 0, t_max = 0;   // мкс на запуск
};

// Проверка тела пользовательского регулятора шага (C body из библиотеки): компиляция NVRTC
// без запуска. log — сообщения компилятора (строки "controller(N): ..." — в координатах тела).
bool nvrtc_check_ctrl_body(const std::string& body, std::string& log);

struct PhaseAdaptiveResult {
    std::vector<std::vector<std::vector<double>>> traj;   // [N][pts][dim]; короче, если разошлось
    std::vector<std::vector<double>> times;               // узлы шага: [N][pts]
    std::vector<std::vector<double>> log;                 // [N][4*k]: t, h, err, код
    std::vector<AdaptiveStats> stats;                     // [N]
    std::vector<double> final_h;                          // [N] — следующий предложенный шаг
};

// Переиспользуемый движок рантайм-компиляции CUDA через NVRTC.
// Компилирует КРС (тело calculateDiscreteModel из codegen) вместе с ядром
// траектории в рантайме под текущую GPU и запускает на ней расчёт.
// Один экземпляр держит CUDA-контекст живым; compile() можно звать многократно
// (при смене системы/метода). Хранит до kCacheCapacity последних
// скомпилированных вариантов (по КРС+размерности) — переключение между
// недавно использованными системами/методами не требует перекомпиляции.
// Потокобезопасен (внутренний мьютекс): compile() зовётся и из фонового
// прогрева при смене системы/метода (см. prewarmPhasePortraitsNVRTC), и из
// потока реального расчёта — конкурентные вызовы просто сериализуются.
class NvrtcEngine {
public:
    NvrtcEngine();
    ~NvrtcEngine();

    // Инициализация CUDA Driver API (один раз, потокобезопасно). false + error() при сбое.
    bool init();

    // Компилирует КРС-тело (то, что выдаёт codegen_scheme — тело функции,
    // использующее X[], a[], h) в ядро, либо переиспользует закэшированный
    // модуль, если такие (krs_body, amountOfX) уже компилировались.
    // amountOfX — размерность системы.
    // Возвращает false при ошибке компиляции (error() содержит лог NVRTC).
    bool compile(const std::string& krs_body, int amountOfX);

    // Считает N траекторий на GPU параллельно (поток на траекторию).
    //   ic_flat — начальные условия всех НУ, плоско: ic_flat[tid*amountOfX + k]
    //   N       — число траекторий
    //   values  — параметры a[] [amountOfValues], ОБЩИЕ для всех траекторий
    //   h, total, skip — шаг, число точек, transient
    //   out     — [N][total][amountOfX]
    // Требует успешного compile(). false + error() при сбое.
    bool run_phase_portraits(const std::vector<double>& ic_flat, int N,
        const std::vector<double>& values,
        double h, int total, int skip,
        std::vector<std::vector<std::vector<double>>>& out);

    // Адаптивный шаг: компиляция (с кэшем по телам + размерности) и счёт.
    // Не трогает состояние compile()/run_phase_portraits. false + error() при сбое.
    bool run_phase_portraits_adaptive(const PhaseAdaptiveRequest& rq, PhaseAdaptiveResult& out);

    // Только сборка адаптивного ядра Analysis в кэш (фоновый прогрев): rq.rhs/emb/dprep/deval,
    // ctrl_body и amountOfX — те же, что потом уйдут в run_phase_portraits_adaptive.
    bool prewarm_adaptive(const PhaseAdaptiveRequest& rq);

    // Замер адаптивного шага до T (см. AdaptiveEndpointRequest). Модуль — свой, в том же кэше.
    bool run_adaptive_endpoint(const AdaptiveEndpointRequest& rq, AdaptiveEndpointResult& out);

    const std::string& error() const { return error_; }
    bool ready() const { return compiled_; }

private:
    // Сколько последних уникальных (КРС+amountOfX) держим скомпилированными
    // одновременно. В пределах этого окна переключение — мгновенное
    // (cache hit), за пределами — вытесняется (LRU).
    static constexpr size_t kCacheCapacity = 8;

    struct CacheEntry {
        std::string key;
        void* module = nullptr;  // CUmodule
        void* kernel = nullptr;  // CUfunction
    };

    std::recursive_mutex mutex_;  // защищает все поля ниже

    std::string error_;
    bool inited_ = false;
    bool compiled_ = false;
    int  amountOfX_ = 0;

    // непрозрачные хэндлы CUDA (void* чтобы не тащить cuda.h в заголовок)
    void* context_ = nullptr;  // CUcontext
    void* kernel_ = nullptr;   // CUfunction активной (последней использованной) записи кэша
    int   cc_major_ = 0, cc_minor_ = 0;

    std::vector<CacheEntry> cache_;  // MRU в конце

    // Флаг отмены адаптивных ядер: int в mapped-памяти хоста (ядро читает его через PCIe).
    int*               cancel_host_ = nullptr;
    unsigned long long cancel_dev_  = 0;   // CUdeviceptr

    void unload_all();
    // Флаг отмены (выделяется при первом вызове), сброшенный в 0 перед запуском.
    bool cancel_flag_reset();
    // Ждёт конца ядер на потоке по умолчанию; пока ждёт, переносит запрос отмены в флаг ядра.
    // false — ошибка CUDA (текст в error_); cancelled — отмена была запрошена.
    bool wait_default_stream(const std::shared_ptr<std::atomic<bool>>& cancel, bool& cancelled);
    // Компилирует или берёт из кэша адаптивное ядро; *fn — CUfunction.
    // variant 0 — phase_kernel_ad (Analysis), 1 — endpoint_kernel_ad (замер до T).
    bool compile_adaptive(const PhaseAdaptiveRequest& rq, void** fn, int variant = 0);
};