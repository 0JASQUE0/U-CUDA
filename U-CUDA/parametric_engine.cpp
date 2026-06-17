#include "parametric_engine.h"
//
// Phase 2: реальная реализация. NVRTC компилит наш шаблон, который через #include
// подтягивает cudaLibrary.cuh / cudaLibrary.cu из NonLinAnal. User's KRS определена
// в шаблоне, default-версия в NonLinAnal закрыта #ifndef __CUDACC_RTC__.
// Host-оркестрация мирорит bifurcation1D из hostLibrary.cu (без chunking для MVP).
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <windows.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <vector>

namespace {

constexpr int kBlockSize         = 32;     // как в NonLinAnal::bifurcation1D
constexpr int kMaxAmountOfX      = 32;
constexpr int kMaxAmountOfValues = 64;

std::string exe_dir() {
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n == 0) return ".";
    std::string p(buf, n);
    auto pos = p.find_last_of("\\/");
    return (pos == std::string::npos) ? std::string(".") : p.substr(0, pos);
}

std::string read_text_file(const std::string& path, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "не удалось открыть " + path; return {}; }
    std::ostringstream ss; ss << f.rdbuf();
    std::string s = ss.str();
    // UTF-8 BOM: 0xEF 0xBB 0xBF в начале — NVRTC от него спотыкается.
    if (s.size() >= 3 &&
        (unsigned char)s[0] == 0xEF &&
        (unsigned char)s[1] == 0xBB &&
        (unsigned char)s[2] == 0xBF) {
        s.erase(0, 3);
    }
    // NVRTC не любит не-ASCII символы даже внутри комментариев (компилит как
    // device code, не предполагает многобайтовых юникод-точек). Заменяем все
    // байты со старшим битом на пробел — это безвредно для комментариев и
    // не задевает строковые литералы из ASCII.
    for (char& c : s) {
        if ((unsigned char)c >= 0x80) c = ' ';
    }
    return s;
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

std::string hash_key(const std::string& krs_body, int amountOfX) {
    return std::to_string(std::hash<std::string>{}(krs_body)) + ":" + std::to_string(amountOfX);
}

std::string cu_err(CUresult r) {
    const char* name = nullptr;
    cuGetErrorString(r, &name);
    return name ? std::string(name) : ("CUresult " + std::to_string((int)r));
}

// Локальная host-копия __device__ __host__ getValueByIdx из cudaLibrary.cu:1266.
// Для 1D-бифуркации valueNumber всегда 0, поэтому просто линейная интерполяция.
// Нужна для расчёта значения параметра в CSV (host-side).
inline double getValueByIdx_local(size_t idx, int nPts, double lo, double hi) {
    if (nPts <= 1) return lo;
    return lo + (hi - lo) * (double)idx / (double)(nPts - 1);
}

}  // namespace

// ---------------------------------------------------------------------------

struct ParametricEngine::Impl {
    bool       inited   = false;
    CUcontext  context  = nullptr;
    CUdevice   device   = 0;
    int        cc_major = 0;
    int        cc_minor = 0;

    // Закэшированные тексты NonLinAnal headers (читаются один раз)
    std::string src_cudaLibrary_cu;
    std::string src_cudaLibrary_cuh;
    std::string src_cudaMacros_cuh;
    std::string src_configCUDA_h;
    std::string src_template;
    bool srcs_loaded = false;

    struct CachedModule {
        std::string key;
        CUmodule    module      = nullptr;
        CUfunction  kernel_traj = nullptr;  // calculateDiscreteModelCUDA
        CUfunction  kernel_peak = nullptr;  // peakFinderCUDA
    };
    CachedModule cached;

    ~Impl() {
        if (inited) {
            cuCtxSetCurrent(context);
            release_module();
            cuCtxDestroy(context);
        }
    }

    void release_module() {
        if (cached.module) {
            cuModuleUnload(cached.module);
            cached.module = nullptr;
            cached.kernel_traj = nullptr;
            cached.kernel_peak = nullptr;
            cached.key.clear();
        }
    }

    bool ensure_init(std::string& err) {
        if (inited) return true;
        CUresult r = cuInit(0);
        if (r != CUDA_SUCCESS) { err = "cuInit: " + cu_err(r); return false; }
        r = cuDeviceGet(&device, 0);
        if (r != CUDA_SUCCESS) { err = "cuDeviceGet: " + cu_err(r); return false; }
#if CUDA_VERSION >= 13000
        r = cuCtxCreate(&context, nullptr, 0, device);
#else
        r = cuCtxCreate(&context, 0, device);
#endif
        if (r != CUDA_SUCCESS) { err = "cuCtxCreate: " + cu_err(r); return false; }
        cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
        cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
        inited = true;
        return true;
    }

    bool load_sources(std::string& err) {
        if (srcs_loaded) return true;
        std::string root = exe_dir() + "\\kernels\\";
        std::string e;
        src_template          = read_text_file(root + "bifurcation1d.template.cu", e); if (!e.empty()) { err = e; return false; }
        src_cudaLibrary_cu    = read_text_file(root + "cudaLibrary.cu",            e); if (!e.empty()) { err = e; return false; }
        src_cudaLibrary_cuh   = read_text_file(root + "cudaLibrary.cuh",           e); if (!e.empty()) { err = e; return false; }
        src_cudaMacros_cuh    = read_text_file(root + "cudaMacros.cuh",            e); if (!e.empty()) { err = e; return false; }
        src_configCUDA_h      = read_text_file(root + "configCUDA.h",              e); if (!e.empty()) { err = e; return false; }
        srcs_loaded = true;
        return true;
    }

    bool compile_if_needed(const std::string& krs_body, int amountOfX, std::string& err) {
        std::string key = hash_key(krs_body, amountOfX);
        if (cached.module && cached.key == key) return true;  // cache hit
        release_module();

        if (!load_sources(err)) return false;

        // Подстановка плейсхолдеров в шаблон
        std::string src = src_template;
        src = replace_all(src, "{{AMOUNT_OF_X}}", std::to_string(amountOfX));
        src = replace_all(src, "{{KRS_BODY}}",    krs_body);

        // Виртуальные заголовки для NVRTC
        const char* header_sources[] = {
            src_cudaLibrary_cu.c_str(),
            src_cudaLibrary_cuh.c_str(),
            src_cudaMacros_cuh.c_str(),
            src_configCUDA_h.c_str(),
        };
        const char* header_names[] = {
            "cudaLibrary.cu",
            "cudaLibrary.cuh",
            "cudaMacros.cuh",
            "configCUDA.h",
        };
        constexpr int n_headers = 4;

        nvrtcProgram prog = nullptr;
        nvrtcResult nr = nvrtcCreateProgram(&prog, src.c_str(), "bifurcation1d.cu",
                                            n_headers, header_sources, header_names);
        if (nr != NVRTC_SUCCESS) { err = std::string("nvrtcCreateProgram: ") + nvrtcGetErrorString(nr); return false; }

        // Регистрируем kernel-имена ДО компиляции, чтобы потом через
        // nvrtcGetLoweredName достать их mangled-варианты для cuModuleGetFunction.
        nvrtcAddNameExpression(prog, "calculateDiscreteModelCUDA");
        nvrtcAddNameExpression(prog, "peakFinderCUDA");

        char arch[64];
        snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", cc_major, cc_minor);

        // NVRTC по умолчанию не знает CUDA-include путей (math_constants.h и т.п.).
        // Берём CUDA_PATH из окружения (его проставляет CUDA Toolkit installer).
        std::string cuda_include_opt;
        {
            char buf[MAX_PATH];
            DWORD nlen = GetEnvironmentVariableA("CUDA_PATH", buf, MAX_PATH);
            if (nlen > 0 && nlen < MAX_PATH) {
                cuda_include_opt = std::string("-I") + std::string(buf, nlen) + "\\include";
            }
        }
        if (cuda_include_opt.empty()) {
            err = "переменная окружения CUDA_PATH не задана — NVRTC не найдёт math_constants.h "
                  "(установи CUDA Toolkit или задай CUDA_PATH=...)";
            nvrtcDestroyProgram(&prog);
            return false;
        }

        std::string std_opt = "--std=c++17";
        const char* opts[] = {
            arch,
            std_opt.c_str(),
            "-default-device",
            cuda_include_opt.c_str()
        };

        nr = nvrtcCompileProgram(prog, 4, opts);
        if (nr != NVRTC_SUCCESS) {
            size_t logsz = 0; nvrtcGetProgramLogSize(prog, &logsz);
            std::string log;
            if (logsz > 1) { log.resize(logsz); nvrtcGetProgramLog(prog, &log[0]); }
            err = "NVRTC compile failed:\n" + log;
            nvrtcDestroyProgram(&prog);
            return false;
        }

        // Mangled-имена для обоих kernel-ов. Нужно скопировать в свои строки
        // ДО nvrtcDestroyProgram — после destroy указатели становятся невалидны.
        const char* mangled_traj_ptr = nullptr;
        const char* mangled_peak_ptr = nullptr;
        nvrtcGetLoweredName(prog, "calculateDiscreteModelCUDA", &mangled_traj_ptr);
        nvrtcGetLoweredName(prog, "peakFinderCUDA",             &mangled_peak_ptr);
        std::string mangled_traj = mangled_traj_ptr ? mangled_traj_ptr : "calculateDiscreteModelCUDA";
        std::string mangled_peak = mangled_peak_ptr ? mangled_peak_ptr : "peakFinderCUDA";

        size_t ptxsz = 0; nvrtcGetPTXSize(prog, &ptxsz);
        std::string ptx(ptxsz, '\0');
        nvrtcGetPTX(prog, &ptx[0]);
        nvrtcDestroyProgram(&prog);

        CUresult r = cuModuleLoadDataEx(&cached.module, ptx.c_str(), 0, nullptr, nullptr);
        if (r != CUDA_SUCCESS) { err = "cuModuleLoadDataEx: " + cu_err(r); return false; }

        r = cuModuleGetFunction(&cached.kernel_traj, cached.module, mangled_traj.c_str());
        if (r != CUDA_SUCCESS) {
            err = "cuModuleGetFunction(" + mangled_traj + "): " + cu_err(r);
            release_module(); return false;
        }
        r = cuModuleGetFunction(&cached.kernel_peak, cached.module, mangled_peak.c_str());
        if (r != CUDA_SUCCESS) {
            err = "cuModuleGetFunction(" + mangled_peak + "): " + cu_err(r);
            release_module(); return false;
        }

        cached.key = key;
        return true;
    }

    // =========================================================================
    // run_bif1d — порт NonLinAnal::bifurcation1D из hostLibrary.cu:165-655.
    // Идея: брать оригинальный код почти как есть, чтобы при обновлениях
    // NonLinAnal перенос был механическим diff → patch. Изменения, которые
    // ОБЯЗАТЕЛЬНЫ (помечены комментарием [ADAPT]):
    //   - <<<grid, block, shared>>>(...) → cuLaunchKernel(CUfunction, ...) —
    //     потому что наш kernel-модуль скомпилирован NVRTC'ом во время работы
    //     и недоступен по compile-time символу
    //   - gpuErrorCheck(...) — у NonLinAnal он зовёт exit() при ошибке; здесь
    //     заменено локальным макросом BIF_CHECK, который пишет в res.error и
    //     возвращает результат с cleanup'ом
    //   - OUT_FILE_PATH приходит из req.csv_output_path (пусто = CSV не пишем)
    //   - continuation_bif1D == 1 ветка отключена — она использует host-side
    //     calculateDiscreteModel (default Lorenz из cudaLibrary.cu:120), а не
    //     user's KRS; для нашего адаптера это не сработает
    //   - calculate_mean_med_freq / calculate_mean_and_variance отключены —
    //     соответствующие kernel-ы (MeanAndMedianFreqCUDA и т.д.) не входят
    //     в NVRTC-bundle. Включим по мере необходимости
    //   - Результат пишется в Bifurcation1DResult (память + CSV), не в файл
    // =========================================================================
    Bifurcation1DResult run_bif1d(const Bifurcation1DRequest& req) {
        Bifurcation1DResult res;
        auto fail = [&](const std::string& msg) -> Bifurcation1DResult& { res.error = msg; return res; };

        // ---- валидация ----
        if (req.krs_body.empty())                                   return fail("krs_body пуст");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX вне [1," + std::to_string(kMaxAmountOfX) + "]");
        if ((int)req.initial_conditions.size() != req.amountOfX)    return fail("initial_conditions.size() != amountOfX");
        // base_values уже идёт со сдвигом +1 (a[0] зарезервирован):
        if ((int)req.base_values.size() > kMaxAmountOfValues)       return fail("base_values слишком много");
        if (req.param_index <= 0 || req.param_index >= (int)req.base_values.size())
                                                                    return fail("param_index вне диапазона");
        if (req.writable_var < 0 || req.writable_var >= req.amountOfX)
                                                                    return fail("writable_var вне диапазона");
        if (req.n_pts <= 0)         return fail("n_pts должно быть > 0");
        if (req.h <= 0.0)           return fail("h должно быть > 0");
        if (req.t_max <= 0.0)       return fail("t_max должно быть > 0");
        if (req.transient_time < 0) return fail("transient_time должно быть >= 0");
        if (req.pre_scaller <= 0)   return fail("pre_scaller должно быть > 0");

        // ---- init + контекст ----
        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);

        // ---- компиляция или cache hit ----
        if (!compile_if_needed(req.krs_body, req.amountOfX, err)) return fail(err);

        // ====================================================================
        // ПОРТ NonLinAnal::bifurcation1D (hostLibrary.cu:165-655).
        // Локальные имена мапятся на аргументы функции NonLinAnal для удобства
        // дифа — слева name из req, справа name как в hostLibrary.
        // ====================================================================
        const double tMax                       = req.t_max;
        const int    nPts                       = req.n_pts;
        const double h                          = req.h;
        const int    amountOfInitialConditions  = req.amountOfX;
        const double* initialConditions         = req.initial_conditions.data();
        double ranges[2]                        = { req.param_lo, req.param_hi };
        int    indicesOfMutVars[1]              = { req.param_index };          // уже 1-индексированный
        const int    writableVar                = req.writable_var;
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const double* values                    = req.base_values.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const int    preScaller                 = req.pre_scaller;
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        // Константы из configCUDA.h — у NonLinAnal они constexpr,
        // здесь литералы (значения те же).
        constexpr int  blockSize_setup          = 32;
        constexpr int  set_precision            = 15;
        // [ADAPT] continuation_bif1D, calculate_mean_med_freq отключены —
        // см. комментарий в шапке функции.

        // --- amountOfPointsInBlock / amountOfPointsForSkip — порт строк 196-200 NL ---
        int amountOfPointsInBlock = (int)(tMax / h / preScaller);
        int amountOfPointsForSkip = (int)(transientTime / h);

        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max/h/pre_scaller слишком малы)");

        // --- Memory budget (порт строк 202-244 NL) ---
        size_t freeMemory = 0, totalMemory = 0;
        if (cudaMemGetInfo(&freeMemory, &totalMemory) != cudaSuccess)
            return fail("cudaMemGetInfo failed");
        freeMemory = (size_t)((double)freeMemory * 0.92);

        size_t memPerSystem =
            3 * (size_t)amountOfPointsInBlock * sizeof(double) +  // d_data, d_outPeaks, d_timeOfPeaks
            2 * sizeof(double) +                                  // d_meanFreq, d_medianFreq (зарезервировано)
            sizeof(int);                                          // d_amountOfPeaks
        size_t memConstants =
            2 * sizeof(double) +
            sizeof(int) +
            (size_t)amountOfInitialConditions * sizeof(double) +
            (size_t)amountOfValues * sizeof(double);
        constexpr double SAFETY_FACTOR = 0.9;
        size_t safeFree = (size_t)((double)freeMemory * SAFETY_FACTOR);
        if (memConstants >= safeFree) return fail("not enough GPU memory for constants");
        size_t availableMemory = safeFree - memConstants;

        size_t nPtsLimiter = availableMemory / memPerSystem;
        if (nPtsLimiter < (size_t)blockSize_setup) nPtsLimiter = (size_t)blockSize_setup;
        if (nPtsLimiter > (size_t)nPts)            nPtsLimiter = (size_t)nPts;
        nPtsLimiter = (nPtsLimiter / blockSize_setup) * blockSize_setup;
        if (nPtsLimiter == 0)
            return fail("not enough GPU memory: per-system buffer too large");
        size_t originalNPtsLimiter = nPtsLimiter;

        // --- Host buffers (порт строк 257-264 NL) ---
        // h_data/h_meanFreq/h_medianFreq/h_localX/h_localValues нужны только для
        // continuation_bif1D и mean/median — мы их не используем.
        std::vector<double> h_outPeaks   (nPtsLimiter * (size_t)amountOfPointsInBlock);
        std::vector<double> h_timeOfPeaks(nPtsLimiter * (size_t)amountOfPointsInBlock);
        std::vector<int>    h_amountOfPeaks(nPtsLimiter);

        // --- Device buffers (порт строк 297-306 NL, без d_meanFreq/d_medianFreq) ---
        double* d_data              = nullptr;
        double* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        double* d_initialConditions = nullptr;
        double* d_values            = nullptr;
        int*    d_amountOfPeaks     = nullptr;
        double* d_outPeaks          = nullptr;
        double* d_timeOfPeaks       = nullptr;

        auto cleanup = [&]() {
            if (d_data)              cudaFree(d_data);
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_amountOfPeaks)     cudaFree(d_amountOfPeaks);
            if (d_outPeaks)          cudaFree(d_outPeaks);
            if (d_timeOfPeaks)       cudaFree(d_timeOfPeaks);
        };

        // [ADAPT] gpuErrorCheck → BIF_CHECK: пишем в res.error и выходим с cleanup
        #define BIF_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define BIF_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)

        BIF_CHECK(cudaMalloc((void**)&d_data,              nPtsLimiter * (size_t)amountOfPointsInBlock * sizeof(double)), "cudaMalloc d_data");
        BIF_CHECK(cudaMalloc((void**)&d_ranges,            2 * sizeof(double)),                                          "cudaMalloc d_ranges");
        BIF_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  1 * sizeof(int)),                                             "cudaMalloc d_indicesOfMutVars");
        BIF_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)amountOfInitialConditions * sizeof(double)),          "cudaMalloc d_initialConditions");
        BIF_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(double)),                     "cudaMalloc d_values");
        BIF_CHECK(cudaMalloc((void**)&d_outPeaks,          nPtsLimiter * (size_t)amountOfPointsInBlock * sizeof(double)), "cudaMalloc d_outPeaks");
        BIF_CHECK(cudaMalloc((void**)&d_timeOfPeaks,       nPtsLimiter * (size_t)amountOfPointsInBlock * sizeof(double)), "cudaMalloc d_timeOfPeaks");
        BIF_CHECK(cudaMalloc((void**)&d_amountOfPeaks,     nPtsLimiter * sizeof(int)),                                   "cudaMalloc d_amountOfPeaks");

        // --- H2D констант (порт строк 314-319 NL) ---
        BIF_CHECK(cudaMemcpy(d_ranges,            ranges,             2 * sizeof(double),                                cudaMemcpyHostToDevice), "memcpy d_ranges");
        BIF_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,   1 * sizeof(int),                                   cudaMemcpyHostToDevice), "memcpy d_indices");
        BIF_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)amountOfInitialConditions * sizeof(double), cudaMemcpyHostToDevice), "memcpy d_ic");
        BIF_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(double),            cudaMemcpyHostToDevice), "memcpy d_values");
        BIF_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)nPts / (double)nPtsLimiter);

        // --- Config CSV (порт строк 331-376 NL — упрощённо, только если путь задан) ---
        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            if (cfg.is_open()) {
                cfg << std::setprecision(set_precision);
                cfg << "1D classical bifurcation\n";
                cfg << "Parameter estimation\n";
                cfg << "a[" << amountOfValues << "] = { ";
                for (int kk = 0; kk < amountOfValues; ++kk) {
                    cfg << values[kk];
                    if (kk != amountOfValues - 1) cfg << ", "; else cfg << " }\n";
                }
                cfg << "X0[" << amountOfInitialConditions << "] = { ";
                for (int kk = 0; kk < amountOfInitialConditions; ++kk) {
                    cfg << initialConditions[kk];
                    if (kk != amountOfInitialConditions - 1) cfg << ", "; else cfg << " }\n";
                }
                cfg << "CT = "       << tMax << "\n";
                cfg << "TT = "       << transientTime << "\n";
                cfg << "h = "        << h << "\n";
                cfg << "decimator = " << preScaller << "\n";
                cfg << "indexVar for peakfinder = " << writableVar << "\n";
                cfg << "indexPar for estimation = " << indicesOfMutVars[0] << "\n";
                cfg << "start value = " << ranges[0] << ", stop value = " << ranges[1] << "\n";
            }
            // обнуляем основной файл данных
            std::ofstream trunc(OUT_FILE_PATH);
            trunc.close();
        }

        // --- результат-аккумулятор (для GUI) ---
        res.n_pts        = nPts;
        res.record_steps = amountOfPointsInBlock;
        res.flags.assign(nPts, 0);
        res.bifurcation_points.assign(nPts, {});
        res.peak_times.assign(nPts, {});

        // --- Главный цикл (порт строк 396-630 NL) ---
        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            // последний чанк может быть меньше
            if (iter == amountOfIteration - 1)
                nPtsLimiter = nPts - (originalNPtsLimiter * iter);

            int blockSize = 32;
            int gridSize  = (int)((nPtsLimiter + blockSize - 1) / blockSize);

            // [ADAPT] <<<>>> → cuLaunchKernel.
            // continuation_bif1D == 0 ветка (classical). Continuation отключена.
            int    nPts_int                  = nPts;
            int    nPtsLimiter_int           = (int)nPtsLimiter;
            size_t sizeOfBlock_s             = (size_t)amountOfPointsInBlock;
            size_t amountOfCalculatedPoints  = iter * originalNPtsLimiter;
            size_t amountOfPointsForSkip_s   = (size_t)amountOfPointsForSkip;
            int    dimension                 = 1;
            double h_arg                     = h;
            int    amountOfInitialConditions_int = amountOfInitialConditions;
            int    amountOfValues_int        = amountOfValues;
            size_t amountOfIterations_arg    = (size_t)amountOfPointsInBlock;
            int    preScaller_int            = preScaller;
            int    writableVar_int           = writableVar;
            double maxValue_arg              = maxValue;
            bool   par_or_var_arg            = true;   // parametric analysis

            void* args_traj[] = {
                &nPts_int,
                &nPtsLimiter_int,
                &sizeOfBlock_s,
                &amountOfCalculatedPoints,
                &amountOfPointsForSkip_s,
                &dimension,
                &d_ranges,
                &h_arg,
                &d_indicesOfMutVars,
                &d_initialConditions,
                &amountOfInitialConditions_int,
                &d_values,
                &amountOfValues_int,
                &amountOfIterations_arg,
                &preScaller_int,
                &writableVar_int,
                &maxValue_arg,
                &d_data,
                &d_amountOfPeaks,
                &par_or_var_arg
            };

            unsigned int shared = (unsigned int)((amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize);

            BIF_CHECK_CU(cuLaunchKernel(cached.kernel_traj,
                                        gridSize, 1, 1, blockSize, 1, 1,
                                        shared, nullptr, args_traj, nullptr),
                         "cuLaunchKernel(traj)");
            BIF_CHECK(cudaDeviceSynchronize(), "sync after traj");

            // peakFinderCUDA
            double timeStep_arg = h * (double)preScaller;
            void* args_peak[] = {
                &d_data,
                &sizeOfBlock_s,
                &nPtsLimiter_int,
                &d_amountOfPeaks,
                &d_outPeaks,
                &d_timeOfPeaks,
                &timeStep_arg
            };
            BIF_CHECK_CU(cuLaunchKernel(cached.kernel_peak,
                                        gridSize, 1, 1, blockSize, 1, 1,
                                        0, nullptr, args_peak, nullptr),
                         "cuLaunchKernel(peak)");
            BIF_CHECK(cudaDeviceSynchronize(), "sync after peak");

            // D2H (порт строк 538-540 NL)
            BIF_CHECK(cudaMemcpy(h_outPeaks.data(),       d_outPeaks,       nPtsLimiter * (size_t)amountOfPointsInBlock * sizeof(double), cudaMemcpyDeviceToHost), "memcpy h_outPeaks");
            BIF_CHECK(cudaMemcpy(h_amountOfPeaks.data(),  d_amountOfPeaks,  nPtsLimiter * sizeof(int),                                    cudaMemcpyDeviceToHost), "memcpy h_amountOfPeaks");
            BIF_CHECK(cudaMemcpy(h_timeOfPeaks.data(),    d_timeOfPeaks,    nPtsLimiter * (size_t)amountOfPointsInBlock * sizeof(double), cudaMemcpyDeviceToHost), "memcpy h_timeOfPeaks");
            BIF_CHECK(cudaDeviceSynchronize(), "sync after D2H");

            // --- CSV + аккумуляция результата (порт строк 574-608 NL) ---
            std::ofstream out;
            if (!OUT_FILE_PATH.empty()) {
                out.open(OUT_FILE_PATH, std::ios::app);
                if (out.is_open()) out << std::setprecision(set_precision);
            }

            for (size_t k = 0; k < nPtsLimiter; ++k) {
                size_t global_idx = originalNPtsLimiter * iter + k;
                double param_val  = getValueByIdx_local(global_idx, nPts, ranges[0], ranges[1]);
                int    npeaks     = h_amountOfPeaks[k];

                // CSV — формат NonLinAnal: param, peak, time
                if (out.is_open()) {
                    if (npeaks == 0) {
                        out << param_val << ", " << 0 << ", " << 0 << '\n';
                    } else if (npeaks == -1) {
                        out << param_val << ", " << 0 << ", " << -1 << '\n';
                    } else {
                        for (size_t j = 0; j < (size_t)npeaks; ++j) {
                            out << param_val << ", "
                                << h_outPeaks   [k * (size_t)amountOfPointsInBlock + j] << ", "
                                << h_timeOfPeaks[k * (size_t)amountOfPointsInBlock + j] << '\n';
                        }
                    }
                }

                // В память для GUI: значения пиков и межпиковые интервалы
                res.flags[global_idx] = npeaks;
                auto& dst_peaks = res.bifurcation_points[global_idx];
                auto& dst_times = res.peak_times[global_idx];
                if (npeaks > 0) {
                    int n = npeaks;
                    if (n > amountOfPointsInBlock) n = amountOfPointsInBlock;
                    const double* peakRow = h_outPeaks.data()    + k * (size_t)amountOfPointsInBlock;
                    const double* timeRow = h_timeOfPeaks.data() + k * (size_t)amountOfPointsInBlock;
                    dst_peaks.assign(peakRow, peakRow + n);
                    dst_times.assign(timeRow, timeRow + n);
                } else {
                    dst_peaks.clear();
                    dst_times.clear();
                }
            }

            if (out.is_open()) out.close();
        }

        cleanup();
        #undef BIF_CHECK
        #undef BIF_CHECK_CU
        res.ok = true;
        return res;
    }
};

// ---------------------------------------------------------------------------

ParametricEngine::ParametricEngine()  : impl_(std::make_unique<Impl>()) {}
ParametricEngine::~ParametricEngine() = default;

Bifurcation1DResult ParametricEngine::run_bifurcation_1d(const Bifurcation1DRequest& req) {
    return impl_->run_bif1d(req);
}
