#include "phase_portrait_nvrtc.h"
#include "nvrtc_engine.h"

// Глобальный движок (один CUDA-контекст на приложение). Кэш скомпилированных
// вариантов живёт внутри NvrtcEngine (см. nvrtc_engine.h/.cpp) — он же
// потокобезопасен, так что этот движок может одновременно обслуживать и
// фоновый прогрев (prewarmPhasePortraitsNVRTC), и настоящий расчёт.
static NvrtcEngine g_engine;

bool computePhasePortraitsNVRTC(
    const std::string& krs_body,
    int amountOfX,
    const double* ic_flat, int N,
    const double* values, int amountOfValues,
    double h, int totalPoints, int skipPoints,
    std::vector<std::vector<std::vector<double>>>& outTrajectories,
    std::string* err)
{
    outTrajectories.clear();
    if (!g_engine.init()) { if (err)*err = g_engine.error(); return false; }
    // compile() сам проверяет кэш (по krs_body+amountOfX) и перекомпилирует
    // только при реальном промахе.
    if (!g_engine.compile(krs_body, amountOfX)) { if (err)*err = g_engine.error(); return false; }
    std::vector<double> ic(ic_flat, ic_flat + (size_t)N * amountOfX);
    std::vector<double> a(values, values + amountOfValues);
    if (!g_engine.run_phase_portraits(ic, N, a, h, totalPoints, skipPoints, outTrajectories)) {
        if (err)*err = g_engine.error(); return false;
    }
    return true;
}

void prewarmPhasePortraitsNVRTC(const std::string& krs_body, int amountOfX) {
    if (krs_body.empty() || amountOfX <= 0) return;
    if (!g_engine.init()) return;
    g_engine.compile(krs_body, amountOfX);
}