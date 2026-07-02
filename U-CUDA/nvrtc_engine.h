#pragma once
#include <mutex>
#include <string>
#include <vector>

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

    void unload_all();
};