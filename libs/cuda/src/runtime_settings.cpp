#include "ucuda/cuda/runtime_settings.h"
#include <mutex>
namespace {
std::mutex g_peak_mu;
PeakConfig g_peak_cfg;
uint64_t   g_peak_epoch = 1;
}  // namespace

void set_peak_config(const PeakConfig& c) {
    std::lock_guard<std::mutex> lk(g_peak_mu);
    g_peak_cfg = c;
    ++g_peak_epoch;
}
PeakConfig get_peak_config() {
    std::lock_guard<std::mutex> lk(g_peak_mu);
    return g_peak_cfg;
}
uint64_t peak_config_epoch() {
    std::lock_guard<std::mutex> lk(g_peak_mu);
    return g_peak_epoch;
}

// FMA-контракция для NVRTC (развёрнуто в parametric_engine.h). Флаг всегда передаётся ЯВНО,
// даже совпадая с дефолтом NVRTC: молчаливое «по умолчанию» уже один раз разъехалось между
// картами и портретами. Хранится атомарно, а не под мьютексом: одно bool-поле, читателям нужен
// только свежий снимок.
namespace {
std::atomic<bool> g_nvrtc_fmad{ true };
}  // namespace

void set_nvrtc_fmad(bool enabled) { g_nvrtc_fmad.store(enabled, std::memory_order_relaxed); }
bool get_nvrtc_fmad()             { return g_nvrtc_fmad.load(std::memory_order_relaxed); }
