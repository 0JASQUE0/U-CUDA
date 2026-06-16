#include "parametric_engine.h"
//
// Phase 2 (NVRTC + NonLinAnal include) пока не реализован — этот файл вернётся
// к жизни на следующей итерации. Сейчас API сохранён (заголовок), но Run всегда
// возвращает "not implemented yet". GUI это видит и показывает текст ошибки.
//
// План Phase 2:
//   1. parametric_engine.cpp загружает с диска cudaLibrary.cu/.cuh, cudaMacros.cuh,
//      configCUDA.h и передаёт NVRTC как виртуальные заголовки.
//   2. Шаблон ядра определяет {{AMOUNT_OF_X}} + KRS, потом #include "cudaLibrary.cuh"
//      и #include "cudaLibrary.cu" — kernel-ы NonLinAnal подхватятся без дублирования.
//   3. Host-оркестрация bifurcation1D из hostLibrary.cu переносится сюда (driver API,
//      cuLaunchKernel вместо <<<>>>).
//

struct ParametricEngine::Impl {
    // здесь будет CUcontext, кэш PTX и т.п. — пока пусто
};

ParametricEngine::ParametricEngine()  : impl_(std::make_unique<Impl>()) {}
ParametricEngine::~ParametricEngine() = default;

Bifurcation1DResult ParametricEngine::run_bifurcation_1d(const Bifurcation1DRequest& req) {
    Bifurcation1DResult r;
    (void)req;
    r.ok = false;
    r.error = "ParametricEngine: реализация переехала на NonLinAnal-через-NVRTC, Phase 2 в работе";
    return r;
}
