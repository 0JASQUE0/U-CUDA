#pragma once
#include <string>
#include <vector>

// Переиспользуемый движок рантайм-компиляции CUDA через NVRTC.
// Компилирует КРС (тело calculateDiscreteModel из codegen) вместе с ядром
// траектории в рантайме под текущую GPU и запускает на ней расчёт.
// Один экземпляр держит CUDA-контекст живым; compile() можно звать многократно
// (при смене системы/метода).
class NvrtcEngine {
public:
    NvrtcEngine();
    ~NvrtcEngine();

    // Инициализация CUDA Driver API (один раз). false + error() при сбое.
    bool init();

    // Компилирует КРС-тело (то, что выдаёт codegen_scheme — тело функции,
    // использующее X[], a[], h) в ядро. amountOfX — размерность системы.
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
    std::string error_;
    bool inited_ = false;
    bool compiled_ = false;
    int  amountOfX_ = 0;

    // непрозрачные хэндлы CUDA (void* чтобы не тащить cuda.h в заголовок)
    void* context_ = nullptr;  // CUcontext
    void* module_ = nullptr;  // CUmodule
    void* kernel_ = nullptr;  // CUfunction
    int   cc_major_ = 0, cc_minor_ = 0;

    void release_module();
};