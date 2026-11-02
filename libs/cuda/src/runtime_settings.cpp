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
std::atomic<bool> g_nvrtc_rdc{ false };
std::atomic<int> g_gpu_block_size{ kGpuBlockSizeDefault };
}  // namespace

void set_nvrtc_fmad(bool enabled) { g_nvrtc_fmad.store(enabled, std::memory_order_relaxed); }
bool get_nvrtc_fmad()             { return g_nvrtc_fmad.load(std::memory_order_relaxed); }

// Раздельная компиляция (см. build_module). Отключается на случай драйвера, на котором линковка
// поведёт себя не так, как на проверенных: тогда движок собирает всё одной единицей трансляции,
// как раньше. Хранится атомарно по той же причине, что и fmad.
void set_nvrtc_rdc(bool enabled) { g_nvrtc_rdc.store(enabled, std::memory_order_relaxed); }

bool get_nvrtc_rdc()             { return g_nvrtc_rdc.load(std::memory_order_relaxed); }

void set_gpu_block_size(int threads) {
    g_gpu_block_size.store(clamp_gpu_block_size(threads), std::memory_order_relaxed);
}

int get_gpu_block_size() { return g_gpu_block_size.load(std::memory_order_relaxed); }
