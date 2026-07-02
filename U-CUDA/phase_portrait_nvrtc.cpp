#include "phase_portrait_nvrtc.h"
#include "nvrtc_engine.h"

// Глобальный движок (один CUDA-контекст на приложение) + память последней
// скомпилированной КРС, чтобы не перекомпилировать без изменений.
static NvrtcEngine g_engine;
static std::string g_last_krs;
static int         g_last_nx = -1;
static bool        g_inited = false;

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
    if (!g_inited) {
        if (!g_engine.init()) { if (err)*err = g_engine.error(); return false; }
        g_inited = true;
    }
    // перекомпиляция только если КРС или размерность изменились
    if (krs_body != g_last_krs || amountOfX != g_last_nx) {
        if (!g_engine.compile(krs_body, amountOfX)) { if (err)*err = g_engine.error(); return false; }
        g_last_krs = krs_body;
        g_last_nx = amountOfX;
    }
    std::vector<double> ic(ic_flat, ic_flat + (size_t)N * amountOfX);
    std::vector<double> a(values, values + amountOfValues);
    if (!g_engine.run_phase_portraits(ic, N, a, h, totalPoints, skipPoints, outTrajectories)) {
        if (err)*err = g_engine.error(); return false;
    }
    return true;
}