#include "parametric_engine.h"
//
// Заглушка — на этом шаге собирается, но возвращает "not implemented".
// Реальная NVRTC-обвязка появится на следующем шаге, когда будем порт-ить
// host-оркестрацию bifurcation1D из hostLibrary.cu в этот файл.
//

struct ParametricEngine::Impl {
    // Здесь будут жить:
    //   - CUcontext / NvrtcEngine
    //   - кэш скомпилированных модулей по hash(krs_body + amountOfX)
    //   - загруженный текст шаблона bifurcation1d.template.cu
};

ParametricEngine::ParametricEngine() : impl_(std::make_unique<Impl>()) {}
ParametricEngine::~ParametricEngine() = default;

Bifurcation1DResult ParametricEngine::run_bifurcation_1d(const Bifurcation1DRequest& req) {
    Bifurcation1DResult r;
    (void)req;
    r.ok = false;
    r.error = "ParametricEngine::run_bifurcation_1d: not implemented yet (skeleton stage)";
    return r;
}
