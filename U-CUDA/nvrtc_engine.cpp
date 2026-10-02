#include "nvrtc_engine.h"
#include "parametric_engine.h"   // get_nvrtc_fmad(): режим FMA общий с картами
#include <cuda.h>
#include <nvrtc.h>
#include <chrono>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <thread>

// exe_dir(): единственное внешнее определение живёт в app_main.cpp (Release) /
// main_NonLinAnal.cu (Debug) — копия в parametric_engine.cpp лежит в
// анонимном namespace и снаружи не видна.
extern std::string exe_dir();

// помощники проверки ошибок, пишут в error_
#define NV_FAIL(msg) do { error_ = (msg); return false; } while(0)

static bool nvrtc_ok(nvrtcResult r, std::string& err, const char* where) {
    if (r != NVRTC_SUCCESS) { err = std::string("NVRTC: ") + nvrtcGetErrorString(r) + " @ " + where; return false; }
    return true;
}
static bool cu_ok(CUresult r, std::string& err, const char* where) {
    if (r != CUDA_SUCCESS) {
        const char* m = nullptr; cuGetErrorString(r, &m);
        err = std::string("CUDA: ") + (m ? m : "?") + " @ " + where; return false;
    }
    return true;
}
#define NVOK(x,w) do { if(!nvrtc_ok((x),error_,w)) return false; } while(0)
#define CUOK(x,w) do { if(!cu_ok((x),error_,w)) return false; } while(0)

NvrtcEngine::NvrtcEngine() {}
NvrtcEngine::~NvrtcEngine() {
    unload_all();
    if (cancel_host_) { cuMemFreeHost(cancel_host_); cancel_host_ = nullptr; cancel_dev_ = 0; }
    if (context_) { cuCtxDestroy((CUcontext)context_); context_ = nullptr; }
}

bool NvrtcEngine::cancel_flag_reset() {
    if (!cancel_host_) {
        void* p = nullptr;
        CUOK(cuMemHostAlloc(&p, sizeof(int), CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE), "cancelFlagAlloc");
        CUdeviceptr d = 0;
        if (!cu_ok(cuMemHostGetDevicePointer(&d, p, 0), error_, "cancelFlagDevPtr")) { cuMemFreeHost(p); return false; }
        cancel_host_ = (int*)p;
        cancel_dev_  = (unsigned long long)d;
    }
    *(volatile int*)cancel_host_ = 0;
    return true;
}

bool NvrtcEngine::wait_default_stream(const std::shared_ptr<std::atomic<bool>>& cancel, bool& cancelled) {
    // Первые миллисекунды — опрос с yield (короткие ядра Analysis не ждут кванта планировщика),
    // дальше — сон по 1 мс.
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        const CUresult r = cuStreamQuery(nullptr);
        if (r == CUDA_SUCCESS) return true;
        if (r != CUDA_ERROR_NOT_READY) return cu_ok(r, error_, "wait(ad)");
        if (!cancelled && cancel && cancel->load(std::memory_order_relaxed)) {
            *(volatile int*)cancel_host_ = 1;
            cancelled = true;
        }
        if (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(5)) std::this_thread::yield();
        else std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void NvrtcEngine::unload_all() {
    for (auto& e : cache_) cuModuleUnload((CUmodule)e.module);
    cache_.clear();
    kernel_ = nullptr;
    compiled_ = false;
}

bool NvrtcEngine::init() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (inited_) return true;
    CUOK(cuInit(0), "cuInit");
    CUdevice dev;
    CUOK(cuDeviceGet(&dev, 0), "cuDeviceGet");
    CUcontext ctx;
    // В CUDA 13 у cuCtxCreate появился параметр CUctxCreateParams* перед flags.
    // nullptr эквивалентен поведению старого 3-арг вызова.
#if CUDA_VERSION >= 13000
    CUOK(cuCtxCreate(&ctx, nullptr, 0, dev), "cuCtxCreate");
#else
    CUOK(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");
#endif
    context_ = ctx;
    CUOK(cuDeviceGetAttribute(&cc_major_, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "ccMajor");
    CUOK(cuDeviceGetAttribute(&cc_minor_, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "ccMinor");
    inited_ = true;
    return true;
}

bool NvrtcEngine::compile(const std::string& krs_body, int amountOfX) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!inited_ && !init()) return false;
    // Если этот worker-thread унаследовал чужой текущий контекст (например от
    // ParametricEngine, который работал в той же thread-pool ячейке), модуль
    // и kernel из нашего контекста дадут "invalid resource handle @ launch".
    // Жёстко выставляем НАШ контекст текущим перед любыми CU/NVRTC-вызовами.
    cuCtxSetCurrent((CUcontext)context_);

    // Ключ кэша: КРС-текст + размерность + режим FMA. 0x1F — разделитель (не
    // встречается в сгенерированном коде), иначе "foo"+"12" неотличимо от
    // "foo1"+"2".
    // fmad меняет не текст, а опцию компиляции, т.е. при том же КРС даёт другой
    // PTX. Без него в ключе переключатель в Settings молча не применялся бы к
    // портретам до перезапуска — модуль отдавался бы из кэша.
    const bool fmad = get_nvrtc_fmad();
    std::string key = krs_body;
    key += '\x1f';
    key += std::to_string(amountOfX);
    key += '\x1f';
    key += (fmad ? '1' : '0');

    // Cache hit: систему/метод уже компилировали в этой сессии — переключаемся
    // на готовый модуль без обращения к NVRTC.
    for (size_t i = 0; i < cache_.size(); ++i) {
        if (cache_[i].key == key) {
            kernel_ = cache_[i].kernel;
            amountOfX_ = amountOfX;
            compiled_ = true;
            if (i + 1 != cache_.size()) {           // MRU: hit -> в конец
                CacheEntry hit = cache_[i];
                cache_.erase(cache_.begin() + i);
                cache_.push_back(hit);
            }
            return true;
        }
    }

    amountOfX_ = amountOfX;

    // Схемы с комплексными коэффициентами (Complex CD) держат внутри шага
    // ucmplx — тип объявлен в configCUDA.h. В отличие от параметрического
    // движка, этот исходник самодостаточен и собирается вообще без -I, поэтому
    // заголовок подаём текстом, и только когда он реально нужен: для остальных
    // схем исходник и опции компиляции остаются прежними.
    // Признак берём из самого тела — ucmplx в нём есть тогда и только тогда,
    // когда схема комплексная, а тело и так является ключом кеша модулей, так
    // что рассинхронизироваться тут нечему.
    const bool needs_complex = krs_body.find("ucmplx") != std::string::npos;
    std::string cfg_header;
    if (needs_complex) {
        const std::string path = exe_dir() + "\\kernels\\configCUDA.h";
        std::ifstream f(path, std::ios::binary);
        if (!f) NV_FAIL("missing " + path + " (needed for schemes with complex "
                        "coefficients; check that kernels\\ was copied next to the .exe)");
        std::ostringstream ss; ss << f.rdbuf();
        cfg_header = ss.str();
        // Санитайзинг — тот же, что в parametric_engine.cpp::read_text_file, и
        // по той же причине. Все исходники проекта в UTF-8 С BOM
        // (.editorconfig): файл, найденный по -I, NVRTC разбирает сам, но
        // заголовок, поданный ТЕКСТОМ, попадает в препроцессор как есть, и BOM
        // становится "unrecognized token" ещё до #pragma once (проверено: и в
        // первой строке, и в любой другой). Не-ASCII байты глушим до пробела:
        // весь не-ASCII в configCUDA.h живёт в комментариях, а исторически
        // NVRTC спотыкался и о них.
        if (cfg_header.size() >= 3 && (unsigned char)cfg_header[0] == 0xEF
            && (unsigned char)cfg_header[1] == 0xBB && (unsigned char)cfg_header[2] == 0xBF)
            cfg_header.erase(0, 3);
        for (char& c : cfg_header) if ((unsigned char)c >= 0x80) c = ' ';
    }

    // Собираем полный CUDA-исходник: тип, КРС как __device__, ядро траектории.
    // krs_body использует X[], a[], h (как выдаёт codegen). Тип numb=double тут.
    std::ostringstream src;
    src << "typedef double numb;\n"
        << "#define AMOUNTOFX " << amountOfX << "\n";
    // configCUDA.h сам объявляет numb и AMOUNTOFX (второе — под #ifndef, первое
    // повторным typedef того же типа, что легально), поэтому порядок безопасен.
    if (needs_complex) src << "#include \"configCUDA.h\"\n";
    src << "__device__ __forceinline__ void calculateDiscreteModel(numb* X, const numb* a, numb h) {\n"
        << krs_body << "\n"
        << "}\n"
        // Ядро на N траекторий: поток tid считает траекторию для НУ номер tid.
        // Все траектории параллельны. Параметры values общие для всех потоков.
        // Layout НУ:    ic[tid*AMOUNTOFX + k]
        // Layout выхода: data[(tid*total + step)*AMOUNTOFX + k]
        << "extern \"C\" __global__ void phase_kernel("
        "const numb* ic, const numb* values, numb h, int total, int skip, int N, numb* data) {\n"
        << "    int tid = blockIdx.x*blockDim.x + threadIdx.x;\n"
        << "    if (tid >= N) return;\n"
        << "    numb X[AMOUNTOFX];\n"
        << "    for (int i=0;i<AMOUNTOFX;++i) X[i]=ic[tid*AMOUNTOFX + i];\n"
        << "    for (int s=0;s<skip;++s) calculateDiscreteModel(X, values, h);\n"
        << "    numb* out = data + (size_t)tid*total*AMOUNTOFX;\n"
        << "    for (int t=0;t<total;++t) {\n"
        << "        for (int k=0;k<AMOUNTOFX;++k) out[t*AMOUNTOFX+k]=X[k];\n"
        << "        calculateDiscreteModel(X, values, h);\n"
        << "    }\n"
        << "}\n";
    std::string code = src.str();

    nvrtcProgram prog;
    const char* hdr_src[]  = { cfg_header.c_str() };
    const char* hdr_name[] = { "configCUDA.h" };
    NVOK(nvrtcCreateProgram(&prog, code.c_str(), "model.cu",
                            needs_complex ? 1 : 0,
                            needs_complex ? hdr_src  : nullptr,
                            needs_complex ? hdr_name : nullptr), "createProgram");
    char arch[32];
    snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", cc_major_, cc_minor_);
    // FMA-контракция обязана совпадать с parametric_engine.cpp: фазовый портрет
    // рисуется по ячейке уже посчитанной карты, и если карта считалась с
    // контракцией, а портрет без — на фрактальной границе бассейнов траектория
    // уходит в другой аттрактор. Раньше здесь стоял жёсткий --fmad=false с
    // комментарием «см. parametric_engine.cpp», где этого флага не было вовсе.
    // Теперь обе стороны читают одну настройку (Settings -> get_nvrtc_fmad).
    const char* opts[] = { arch, fmad ? "--fmad=true" : "--fmad=false" };
    nvrtcResult comp = nvrtcCompileProgram(prog, 2, opts);
    // лог (при ошибке — в error_)
    size_t logsz = 0; nvrtcGetProgramLogSize(prog, &logsz);
    std::string log;
    if (logsz > 1) { log.resize(logsz); nvrtcGetProgramLog(prog, &log[0]); }
    if (comp != NVRTC_SUCCESS) {
        error_ = "NVRTC compile failed:\n" + log;
        nvrtcDestroyProgram(&prog);
        return false;
    }
    size_t ptxsz = 0; NVOK(nvrtcGetPTXSize(prog, &ptxsz), "ptxSize");
    std::string ptx(ptxsz, '\0'); NVOK(nvrtcGetPTX(prog, &ptx[0]), "getPtx");
    nvrtcDestroyProgram(&prog);

    CUmodule mod;
    CUOK(cuModuleLoadDataEx(&mod, ptx.c_str(), 0, nullptr, nullptr), "moduleLoad");
    CUfunction fn;
    CUOK(cuModuleGetFunction(&fn, mod, "phase_kernel"), "getFunction");

    // При переполнении кэша вытесняем самый старый (LRU, начало вектора).
    if (cache_.size() >= kCacheCapacity) {
        cuModuleUnload((CUmodule)cache_.front().module);
        cache_.erase(cache_.begin());
    }
    cache_.push_back({ key, mod, fn });
    kernel_ = fn;
    compiled_ = true;
    return true;
}

bool NvrtcEngine::run_phase_portraits(const std::vector<double>& ic_flat, int N,
    const std::vector<double>& values,
    double h, int total, int skip,
    std::vector<std::vector<std::vector<double>>>& out) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    out.clear();
    // см. комментарий в compile() — нужно выставить НАШ контекст текущим.
    cuCtxSetCurrent((CUcontext)context_);
    if (!compiled_) NV_FAIL("not compiled");
    if (N <= 0) NV_FAIL("N<=0");
    if ((int)ic_flat.size() != N * amountOfX_) NV_FAIL("ic_flat size != N*amountOfX");
    if (total <= 0) NV_FAIL("total<=0");

    int nx = amountOfX_, nv = (int)values.size();
    size_t data_count = (size_t)N * total * nx;
    CUdeviceptr d_ic = 0, d_val = 0, d_data = 0;
    CUOK(cuMemAlloc(&d_ic, (size_t)N * nx * sizeof(double)), "allocIc");
    CUOK(cuMemAlloc(&d_val, (nv > 0 ? nv : 1) * sizeof(double)), "allocVal");
    CUOK(cuMemAlloc(&d_data, data_count * sizeof(double)), "allocData");
    CUOK(cuMemcpyHtoD(d_ic, ic_flat.data(), (size_t)N * nx * sizeof(double)), "cpyIc");
    if (nv > 0) CUOK(cuMemcpyHtoD(d_val, values.data(), nv * sizeof(double)), "cpyVal");

    double hh = h;
    int tot = total, sk = skip, n = N;
    void* args[] = { &d_ic, &d_val, &hh, &tot, &sk, &n, &d_data };
    // N потоков: поток tid -> траектория tid. Раскладка под N.
    int threads = 256, blocks = (N + threads - 1) / threads;
    CUOK(cuLaunchKernel((CUfunction)kernel_, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr), "launch");
    CUOK(cuCtxSynchronize(), "sync");

    std::vector<double> flat(data_count);
    CUOK(cuMemcpyDtoH(flat.data(), d_data, data_count * sizeof(double)), "cpyOut");
    cuMemFree(d_ic); cuMemFree(d_val); cuMemFree(d_data);

    // раскладка: out[tid][step][coord]
    out.resize(N);
    for (int tid = 0; tid < N; ++tid) {
        auto& traj = out[tid];
        traj.resize(total);
        const double* base = flat.data() + (size_t)tid * total * nx;
        for (int t = 0; t < total; ++t) { traj[t].resize(nx); for (int k = 0; k < nx; ++k) traj[t][k] = base[(size_t)t * nx + k]; }
    }
    return true;
}

// ---- Адаптивный шаг ------------------------------------------------------------

// Текст kernels/ucuda_adaptive.cuh для NVRTC: заголовок подаётся ТЕКСТОМ, поэтому
// BOM и не-ASCII снимаются так же, как у configCUDA.h выше.
static bool read_kernel_text(const char* name, std::string& out, std::string& err) {
    const std::string path = exe_dir() + "\\kernels\\" + name;
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "missing " + path + " (check that kernels\\ was copied next to the .exe)"; return false; }
    std::ostringstream ss; ss << f.rdbuf();
    out = ss.str();
    if (out.size() >= 3 && (unsigned char)out[0] == 0xEF
        && (unsigned char)out[1] == 0xBB && (unsigned char)out[2] == 0xBF)
        out.erase(0, 3);
    for (char& c : out) if ((unsigned char)c >= 0x80) c = ' ';
    return true;
}
static bool read_adaptive_header(std::string& out, std::string& err) {
    return read_kernel_text("ucuda_adaptive.cuh", out, err);
}

// Проверка тела пользовательского регулятора (редактор библиотеки): только компиляция
// NVRTC, без контекста CUDA и без запуска. Функция — та же, что попадает в модули
// (adaptive_ctrl_source), плюс ядро-обёртка, чтобы её тело действительно собиралось.
bool nvrtc_check_ctrl_body(const std::string& body, std::string& log) {
    log.clear();
    std::string hdr, err;
    if (!read_adaptive_header(hdr, err)) { log = err; return false; }
    // #line 1 "prepare" / "controller" — в самом конце, поэтому номера ошибок — строки раздела
    // подготовки и тела (body — оба, adaptive_ctrl_pack).
    std::ostringstream src;
    src << "typedef double numb;\n"
        << "#define AMOUNTOFX 3\n"
        << "#define UCUDA_ADAPT_LAYOUT_ONLY\n#include \"ucuda_adaptive.cuh\"\n#undef UCUDA_ADAPT_LAYOUT_ONLY\n"
        << "extern \"C\" __global__ void ucuda_ctrl_check(const UcudaCtlIn* in, UcudaCtlMem* m, UcudaCtlOut* o);\n"
        << "extern \"C\" __global__ void ucuda_ctrl_check_prep(const numb* c, int q, numb* k);\n"
        << adaptive_ctrl_source(body, true)
        << "extern \"C\" __global__ void ucuda_ctrl_check(const UcudaCtlIn* in, UcudaCtlMem* m, UcudaCtlOut* o) {\n"
           "    ucuda_ctrl_custom(*in, *m, *o);\n"
           "}\n"
           "extern \"C\" __global__ void ucuda_ctrl_check_prep(const numb* c, int q, numb* k) {\n"
           "    ucuda_ctrl_custom_prep(c, q, k);\n"
           "}\n";
    const std::string code = src.str();
    nvrtcProgram prog = nullptr;
    const char* hdr_src[]  = { hdr.c_str() };
    const char* hdr_name[] = { "ucuda_adaptive.cuh" };
    if (nvrtcCreateProgram(&prog, code.c_str(), "controller_check.cu", 1, hdr_src, hdr_name) != NVRTC_SUCCESS) {
        log = "nvrtcCreateProgram failed";
        return false;
    }
    const char* opts[] = { "--std=c++17" };
    const nvrtcResult comp = nvrtcCompileProgram(prog, 1, opts);
    size_t logsz = 0; nvrtcGetProgramLogSize(prog, &logsz);
    if (logsz > 1) { log.resize(logsz); nvrtcGetProgramLog(prog, &log[0]); }
    while (!log.empty() && (log.back() == '\0' || log.back() == '\n' || log.back() == '\r')) log.pop_back();
    nvrtcDestroyProgram(&prog);
    return comp == NVRTC_SUCCESS;
}

// Ядро замера (variant 1): нить интегрирует [0, T] и пишет y(T); статистика — нить 0.
// Все нити пишут свой y, иначе компилятор вправе выбросить их работу целиком.
// Отмена — флаг cancel (mapped-память хоста) раз в 1024 принятых шага, как у ядра Analysis.
static const char* const kEndpointKernelAd =
    "extern \"C\" __global__ void endpoint_kernel_ad(const numb* ic, const numb* values,\n"
    "    UCUDA_GRID_CONST const UcudaAdaptParams P, numb T, int N, numb* yout, numb* stats,\n"
    "    const volatile int* cancel) {\n"
    "    int tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (tid >= N) return;\n"
    "    const UcudaKrsFns K{};\n"
    "    UcudaAdaptState S;\n"
    "    numb X0[AMOUNTOFX];\n"
    "    for (int i = 0; i < AMOUNTOFX; ++i) X0[i] = ic[i];\n"
    "    ucuda_ad_init(S, K, AMOUNTOFX, X0, (numb)0, values, P, nullptr, 0);\n"
    "    while (S.t < T && !S.diverged) {\n"
    "        ucuda_ad_step(S, K, values, P, T);\n"
    "        if ((S.st.nacc & 1023ULL) == 0 && *cancel != 0) break;\n"
    "    }\n"
    "    for (int k = 0; k < AMOUNTOFX; ++k) yout[(size_t)tid * AMOUNTOFX + k] = S.X[k];\n"
    "    if (tid == 0) {\n"
    "        stats[0] = (numb)S.st.nacc; stats[1] = (numb)S.st.nrej; stats[2] = (numb)S.st.nforced;\n"
    "        stats[3] = (numb)S.st.nrhs; stats[4] = S.st.hmin; stats[5] = S.st.hmax;\n"
    "        stats[6] = S.st.nacc > 0 ? S.st.hsum / (numb)S.st.nacc : (numb)0;\n"
    "        stats[7] = (numb)S.diverged;\n"
    "    }\n"
    "}\n";

bool NvrtcEngine::compile_adaptive(const PhaseAdaptiveRequest& rq, void** fn, int variant) {
    const bool fmad = get_nvrtc_fmad();
    std::string key = variant == 1 ? "adaptive_end" : "adaptive";
    for (const std::string* b : { &rq.rhs, &rq.emb, &rq.dprep, &rq.deval, &rq.ctrl_body }) { key += '\x1f'; key += *b; }
    key += '\x1f'; key += std::to_string(rq.amountOfX);
    key += '\x1f'; key += (fmad ? '1' : '0');
    for (size_t i = 0; i < cache_.size(); ++i) {
        if (cache_[i].key == key) {
            *fn = cache_[i].kernel;
            if (i + 1 != cache_.size()) {
                CacheEntry hit = cache_[i];
                cache_.erase(cache_.begin() + i);
                cache_.push_back(hit);
            }
            return true;
        }
    }

    std::string hdr;
    if (!read_adaptive_header(hdr, error_)) return false;
    // Адаптивный экстраполятор над базой с комплексными коэффициентами держит в emb ucmplx —
    // тип из configCUDA.h; признак и подача текстом — как у ядра постоянного шага (compile).
    const bool needs_complex = rq.emb.find("ucmplx") != std::string::npos;
    std::string cfg;
    if (needs_complex && !read_kernel_text("configCUDA.h", cfg, error_)) return false;

    // Функции схемы — __host__ __device__, как calculateDiscreteModel в шаблонах:
    // их зовут __host__ __device__ методы провайдера UcudaKrsFns.
    std::ostringstream src;
    src << "typedef double numb;\n"
        << "#define AMOUNTOFX " << rq.amountOfX << "\n"
        << (needs_complex ? "#include \"configCUDA.h\"\n" : "")
        << "#define UCUDA_AD_KRS_FUNCS\n"
        << "#define UCUDA_AD_STATIC_N\n"   // поток на траекторию, потоков мало: состояние — в регистры
        << (variant == 1 ? "#define UCUDA_AD_NO_DENSE 1\n" : "")
        << "__device__ __host__ __forceinline__ void ucuda_krs_rhs(const numb* X, const numb* a, numb* F) {\n"
        << rq.rhs << "\n}\n"
        << "__device__ __host__ __forceinline__ void ucuda_krs_emb(const numb* X, const numb* F0, const numb* a,\n"
           "    const numb h, numb* Y, numb* E, numb* F1, numb* W) {\n"
        << rq.emb << "\n}\n"
        << "__device__ __host__ __forceinline__ void ucuda_krs_dprep(const numb* X, const numb* Y, const numb* F0,\n"
           "    const numb* F1, const numb* a, const numb h, numb* W, numb* D) {\n"
        << rq.dprep << "\n}\n"
        << "__device__ __host__ __forceinline__ void ucuda_krs_deval(const numb* D, const numb th, numb* Yo) {\n"
        << rq.deval << "\n}\n"
        // Пользовательский регулятор — между раскладкой драйвера и им самим.
        << "#define UCUDA_ADAPT_LAYOUT_ONLY\n#include \"ucuda_adaptive.cuh\"\n#undef UCUDA_ADAPT_LAYOUT_ONLY\n"
        << adaptive_ctrl_source(rq.ctrl_body)
        << "#include \"ucuda_adaptive.cuh\"\n";
    if (variant == 1) src << kEndpointKernelAd;
    else src
        // Поток на траекторию. Параметры шага P — по значению, __grid_constant__ (с compute_70,
        // UCUDA_GRID_CONST): поля читаются из банка параметров ядра (константный кэш), без копии
        // в локальную память и без повторных загрузок из глобальной на каждой попытке шага.
        // Отмена — флаг cancel (mapped-память хоста) раз в 1024 принятых шага: шагов у
        // траектории сколько угодно (жёсткий допуск), и без флага ядро не прервать.
        // Сетка — шаги до отсчёта и плотный выход (ucuda_ad_advance_to, развёрнутый ради
        // проверки отмены между шагами; те же вызовы, тот же результат).
        // Раскладка выхода:
        //   data[(tid*cap + i)*AMOUNTOFX + k], cap = total (сетка) или max_pts (узлы),
        //   times[tid*max_pts + i] — только для узлов, stats[tid*9 + ...].
        << "#define UCUDA_PH_CANCELLED() ((S.st.nacc & 1023ULL) == 0 && *cancel != 0)\n"
           "extern \"C\" __global__ void phase_kernel_ad(const numb* ic, const numb* values,\n"
           "    UCUDA_GRID_CONST const UcudaAdaptParams P, numb t_skip, numb t_rec, numb dt, int total, int raw,\n"
           "    int max_pts, int log_cap, int N, numb* data, numb* times, int* counts,\n"
           "    numb* logs, int* log_counts, numb* stats, numb* final_h, const volatile int* cancel) {\n"
           "    int tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "    if (tid >= N) return;\n"
           "    const UcudaKrsFns K{};\n"
           "    UcudaAdaptState S;\n"
           "    numb X0[AMOUNTOFX];\n"
           "    for (int i = 0; i < AMOUNTOFX; ++i) X0[i] = ic[tid * AMOUNTOFX + i];\n"
           "    numb* lg = log_cap > 0 ? logs + (size_t)tid * log_cap * 4 : nullptr;\n"
           "    ucuda_ad_init(S, K, AMOUNTOFX, X0, (numb)0, values, P, lg, log_cap);\n"
           "    const numb tEnd = t_skip + t_rec;\n"
           "    bool cut = false;\n"
           "    while (S.t < t_skip && !S.diverged && !cut) {\n"
           "        ucuda_ad_step(S, K, values, P, t_skip);\n"
           "        cut = UCUDA_PH_CANCELLED();\n"
           "    }\n"
           "    int c = 0;\n"
           "    if (!raw) {\n"
           "        numb* out = data + (size_t)tid * total * AMOUNTOFX;\n"
           "        numb y[AMOUNTOFX];\n"
           "        for (; c < total && !S.diverged && !cut; ++c) {\n"
           "            numb tt = t_skip + (numb)c * dt;\n"
           "            if (tt > tEnd) tt = tEnd;\n"
           "            while (S.t < tt && !S.diverged && !cut) {\n"
           "                ucuda_ad_step(S, K, values, P, tEnd);\n"
           "                cut = UCUDA_PH_CANCELLED();\n"
           "            }\n"
           "            if (S.diverged || cut) break;\n"
           "            ucuda_ad_eval(S, K, values, P, tt, y);\n"
           "            for (int k = 0; k < AMOUNTOFX; ++k) out[(size_t)c * AMOUNTOFX + k] = y[k];\n"
           "        }\n"
           "    } else {\n"
           "        numb* out = data + (size_t)tid * max_pts * AMOUNTOFX;\n"
           "        numb* tm = times + (size_t)tid * max_pts;\n"
           "        if (!S.diverged && !cut && max_pts > 0) {\n"
           "            for (int k = 0; k < AMOUNTOFX; ++k) out[k] = S.X[k];\n"
           "            tm[0] = S.t; c = 1;\n"
           "        }\n"
           "        while (S.t < tEnd && c < max_pts && !S.diverged && !cut) {\n"
           "            ucuda_ad_step(S, K, values, P, tEnd);\n"
           "            if (S.diverged) break;\n"
           "            for (int k = 0; k < AMOUNTOFX; ++k) out[(size_t)c * AMOUNTOFX + k] = S.X[k];\n"
           "            tm[c] = S.t; ++c;\n"
           "            cut = UCUDA_PH_CANCELLED();\n"
           "        }\n"
           "    }\n"
           "    counts[tid] = c;\n"
           "    log_counts[tid] = S.log_n;\n"
           "    numb* st = stats + (size_t)tid * 9;\n"
           "    st[0] = (numb)S.st.nacc; st[1] = (numb)S.st.nrej; st[2] = (numb)S.st.nforced;\n"
           "    st[3] = (numb)S.st.nrhs; st[4] = S.st.hmin; st[5] = S.st.hmax;\n"
           "    st[6] = S.st.nacc > 0 ? S.st.hsum / (numb)S.st.nacc : (numb)0;\n"
           "    st[7] = (numb)S.diverged;\n"
           "    st[8] = (numb)(raw && !S.diverged && S.t < tEnd);\n"
           "    final_h[tid] = S.h;\n"
           "}\n";
    const std::string code = src.str();

    nvrtcProgram prog;
    const char* hdr_src[]  = { hdr.c_str(), cfg.c_str() };
    const char* hdr_name[] = { "ucuda_adaptive.cuh", "configCUDA.h" };
    NVOK(nvrtcCreateProgram(&prog, code.c_str(), "model_ad.cu", needs_complex ? 2 : 1, hdr_src, hdr_name),
         "createProgram(ad)");
    char arch[32];
    snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", cc_major_, cc_minor_);
    const char* opts[] = { arch, fmad ? "--fmad=true" : "--fmad=false", "--std=c++17" };
    nvrtcResult comp = nvrtcCompileProgram(prog, 3, opts);
    size_t logsz = 0; nvrtcGetProgramLogSize(prog, &logsz);
    std::string log;
    if (logsz > 1) { log.resize(logsz); nvrtcGetProgramLog(prog, &log[0]); }
    if (comp != NVRTC_SUCCESS) {
        error_ = "NVRTC compile failed (adaptive):\n" + log;
        nvrtcDestroyProgram(&prog);
        return false;
    }
    size_t ptxsz = 0; NVOK(nvrtcGetPTXSize(prog, &ptxsz), "ptxSize(ad)");
    std::string ptx(ptxsz, '\0'); NVOK(nvrtcGetPTX(prog, &ptx[0]), "getPtx(ad)");
    nvrtcDestroyProgram(&prog);

    CUmodule mod;
    CUOK(cuModuleLoadDataEx(&mod, ptx.c_str(), 0, nullptr, nullptr), "moduleLoad(ad)");
    CUfunction f;
    CUOK(cuModuleGetFunction(&f, mod, variant == 1 ? "endpoint_kernel_ad" : "phase_kernel_ad"), "getFunction(ad)");
    if (cache_.size() >= kCacheCapacity) {
        // Вытесняемый модуль может держать ядро постоянного шага: тогда compile()
        // обязан пересобрать его заново, а не звать выгруженное.
        if (cache_.front().kernel == kernel_) { kernel_ = nullptr; compiled_ = false; }
        cuModuleUnload((CUmodule)cache_.front().module);
        cache_.erase(cache_.begin());
    }
    cache_.push_back({ key, mod, f });
    *fn = f;
    return true;
}

bool NvrtcEngine::prewarm_adaptive(const PhaseAdaptiveRequest& rq) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!inited_ && !init()) return false;
    cuCtxSetCurrent((CUcontext)context_);
    void* fn = nullptr;
    return compile_adaptive(rq, &fn, 0);
}

bool NvrtcEngine::run_adaptive_endpoint(const AdaptiveEndpointRequest& rq, AdaptiveEndpointResult& out) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    out = AdaptiveEndpointResult();
    if (!inited_ && !init()) return false;
    cuCtxSetCurrent((CUcontext)context_);
    const int nx = rq.amountOfX, N = rq.replicas;
    if (nx < 1 || nx > UCUDA_AD_MAXN) NV_FAIL("adaptive: dimension out of range");
    if ((int)rq.ic.size() != nx) NV_FAIL("adaptive: ic size != amountOfX");
    if (N < 1 || rq.repeats < 1 || rq.warmup < 0) NV_FAIL("adaptive: bad replicas / repeats / warmup");
    if (!(rq.T > 0)) NV_FAIL("adaptive: T must be > 0");

    PhaseAdaptiveRequest cr;
    cr.rhs = rq.rhs; cr.emb = rq.emb; cr.ctrl_body = rq.ctrl_body; cr.amountOfX = nx;
    void* fn = nullptr;
    if (!compile_adaptive(cr, &fn, 1)) return false;

    const int nv = (int)rq.values.size();
    CUdeviceptr d_ic = 0, d_val = 0, d_y = 0, d_st = 0;
    CUevent ev_a = nullptr, ev_b = nullptr;
    auto free_all = [&]() {
        for (CUdeviceptr p : { d_ic, d_val, d_y, d_st }) if (p) cuMemFree(p);
        if (ev_a) cuEventDestroy(ev_a);
        if (ev_b) cuEventDestroy(ev_b);
    };
    bool ok = cu_ok(cuMemAlloc(&d_ic, (size_t)nx * sizeof(double)), error_, "allocIc(end)")
           && cu_ok(cuMemAlloc(&d_val, (size_t)(nv > 0 ? nv : 1) * sizeof(double)), error_, "allocVal(end)")
           && cu_ok(cuMemAlloc(&d_y, (size_t)N * nx * sizeof(double)), error_, "allocY(end)")
           && cu_ok(cuMemAlloc(&d_st, 8 * sizeof(double)), error_, "allocStats(end)")
           && cu_ok(cuMemcpyHtoD(d_ic, rq.ic.data(), (size_t)nx * sizeof(double)), error_, "cpyIc(end)")
           && (nv == 0 || cu_ok(cuMemcpyHtoD(d_val, rq.values.data(), (size_t)nv * sizeof(double)), error_, "cpyVal(end)"))
           && cu_ok(cuEventCreate(&ev_a, CU_EVENT_DEFAULT), error_, "event(end)")
           && cu_ok(cuEventCreate(&ev_b, CU_EVENT_DEFAULT), error_, "event(end)");
    double T = rq.T;
    int n = N;
    UcudaAdaptParams par = rq.params;
    ok = ok && cancel_flag_reset();
    CUdeviceptr d_cancel = (CUdeviceptr)cancel_dev_;
    void* args[] = { &d_ic, &d_val, &par, &T, &n, &d_y, &d_st, &d_cancel };
    const int threads = 32, blocks = (N + threads - 1) / threads;
    auto launch = [&]() {
        return cu_ok(cuLaunchKernel((CUfunction)fn, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr),
                     error_, "launch(end)");
    };
    // Ожидание — опросом потока: пока ядро идёт, запрос отмены уходит в его флаг. Время
    // запуска по-прежнему меряют события вокруг ядра, опрос в него не входит.
    bool cancelled = false;
    for (int w = 0; ok && !cancelled && w < rq.warmup; ++w)
        ok = launch() && wait_default_stream(rq.cancel, cancelled);
    double tsum = 0;
    for (int r = 0; ok && !cancelled && r < rq.repeats; ++r) {
        float ms = 0;
        ok = cu_ok(cuEventRecord(ev_a, nullptr), error_, "record(end)") && launch()
          && cu_ok(cuEventRecord(ev_b, nullptr), error_, "record(end)")
          && wait_default_stream(rq.cancel, cancelled)
          && cu_ok(cuEventElapsedTime(&ms, ev_a, ev_b), error_, "elapsed(end)");
        if (!ok || cancelled) break;
        const double us = (double)ms * 1000.0;
        if (r == 0) { out.t_min = out.t_max = us; }
        else { if (us < out.t_min) out.t_min = us; if (us > out.t_max) out.t_max = us; }
        tsum += us;
    }
    if (ok && cancelled) { free_all(); error_ = kNvrtcCancelled; return false; }
    std::vector<double> y((size_t)nx), st(8);
    if (ok)
        ok = cu_ok(cuMemcpyDtoH(y.data(), d_y, (size_t)nx * sizeof(double)), error_, "cpyY(end)")
          && cu_ok(cuMemcpyDtoH(st.data(), d_st, 8 * sizeof(double)), error_, "cpyStats(end)");
    free_all();
    if (!ok) return false;
    out.t_avg = tsum / (double)rq.repeats;
    out.y_end = y;
    out.stats.nacc = st[0]; out.stats.nrej = st[1]; out.stats.nforced = st[2]; out.stats.nrhs = st[3];
    out.stats.hmin = st[4]; out.stats.hmax = st[5]; out.stats.hmean = st[6];
    out.stats.diverged = st[7] != 0;
    return true;
}

bool NvrtcEngine::run_phase_portraits_adaptive(const PhaseAdaptiveRequest& rq, PhaseAdaptiveResult& out) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    out = PhaseAdaptiveResult();
    if (!inited_ && !init()) return false;
    cuCtxSetCurrent((CUcontext)context_);
    const int nx = rq.amountOfX, N = rq.N;
    if (N <= 0) NV_FAIL("N<=0");
    if (nx < 1 || nx > UCUDA_AD_MAXN) NV_FAIL("adaptive: dimension out of range");
    if ((int)rq.ic_flat.size() != N * nx) NV_FAIL("ic_flat size != N*amountOfX");
    if (rq.raw ? rq.max_pts <= 0 : rq.total <= 0) NV_FAIL("adaptive: no output points");

    void* fn = nullptr;
    if (!compile_adaptive(rq, &fn)) return false;

    const int cap = rq.raw ? rq.max_pts : rq.total;
    const int nv = (int)rq.values.size();
    const size_t data_count = (size_t)N * cap * nx;
    const size_t log_count  = (size_t)N * (rq.log_cap > 0 ? rq.log_cap : 0) * 4;
    CUdeviceptr d_ic = 0, d_val = 0, d_data = 0, d_times = 0, d_cnt = 0,
                d_log = 0, d_logc = 0, d_st = 0, d_fh = 0;
    auto free_all = [&]() {
        for (CUdeviceptr p : { d_ic, d_val, d_data, d_times, d_cnt, d_log, d_logc, d_st, d_fh })
            if (p) cuMemFree(p);
    };
    auto alloc = [&](CUdeviceptr& p, size_t bytes, const char* what) {
        CUresult r = cuMemAlloc(&p, bytes > 0 ? bytes : 8);
        if (r != CUDA_SUCCESS) { cu_ok(r, error_, what); return false; }
        return true;
    };
    if (!alloc(d_ic, (size_t)N * nx * sizeof(double), "allocIc(ad)")
        || !alloc(d_val, (size_t)(nv > 0 ? nv : 1) * sizeof(double), "allocVal(ad)")
        || !alloc(d_data, data_count * sizeof(double), "allocData(ad)")
        || !alloc(d_times, rq.raw ? (size_t)N * cap * sizeof(double) : 0, "allocTimes(ad)")
        || !alloc(d_cnt, (size_t)N * sizeof(int), "allocCnt(ad)")
        || !alloc(d_log, log_count * sizeof(double), "allocLog(ad)")
        || !alloc(d_logc, (size_t)N * sizeof(int), "allocLogc(ad)")
        || !alloc(d_st, (size_t)N * 9 * sizeof(double), "allocStats(ad)")
        || !alloc(d_fh, (size_t)N * sizeof(double), "allocFh(ad)")) {
        free_all();
        return false;
    }
    bool ok = cu_ok(cuMemcpyHtoD(d_ic, rq.ic_flat.data(), (size_t)N * nx * sizeof(double)), error_, "cpyIc(ad)")
           && (nv == 0 || cu_ok(cuMemcpyHtoD(d_val, rq.values.data(), (size_t)nv * sizeof(double)), error_, "cpyVal(ad)"));
    bool cancelled = false;
    if (ok) ok = cancel_flag_reset();
    if (ok) {
        double t_skip = rq.t_skip, t_rec = rq.t_rec, dt = rq.dt;
        int total = rq.total, raw = rq.raw ? 1 : 0, max_pts = rq.max_pts, log_cap = rq.log_cap, n = N;
        UcudaAdaptParams par = rq.params;
        CUdeviceptr d_cancel = (CUdeviceptr)cancel_dev_;
        void* args[] = { &d_ic, &d_val, &par, &t_skip, &t_rec, &dt, &total, &raw, &max_pts, &log_cap, &n,
                         &d_data, &d_times, &d_cnt, &d_log, &d_logc, &d_st, &d_fh, &d_cancel };
        const int threads = 32, blocks = (N + threads - 1) / threads;
        ok = cu_ok(cuLaunchKernel((CUfunction)fn, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr), error_, "launch(ad)")
          && wait_default_stream(rq.cancel, cancelled);
    }
    if (ok && cancelled) { free_all(); error_ = kNvrtcCancelled; return false; }
    std::vector<double> data(data_count), times(rq.raw ? (size_t)N * cap : 0), logs(log_count), st((size_t)N * 9), fh(N);
    std::vector<int> cnt(N), logc(N);
    if (ok) {
        ok = cu_ok(cuMemcpyDtoH(data.data(), d_data, data_count * sizeof(double)), error_, "cpyData(ad)")
          && (!rq.raw || cu_ok(cuMemcpyDtoH(times.data(), d_times, times.size() * sizeof(double)), error_, "cpyTimes(ad)"))
          && cu_ok(cuMemcpyDtoH(cnt.data(), d_cnt, (size_t)N * sizeof(int)), error_, "cpyCnt(ad)")
          && (log_count == 0 || cu_ok(cuMemcpyDtoH(logs.data(), d_log, log_count * sizeof(double)), error_, "cpyLog(ad)"))
          && cu_ok(cuMemcpyDtoH(logc.data(), d_logc, (size_t)N * sizeof(int)), error_, "cpyLogc(ad)")
          && cu_ok(cuMemcpyDtoH(st.data(), d_st, st.size() * sizeof(double)), error_, "cpyStats(ad)")
          && cu_ok(cuMemcpyDtoH(fh.data(), d_fh, (size_t)N * sizeof(double)), error_, "cpyFh(ad)");
    }
    free_all();
    if (!ok) return false;

    out.traj.resize(N);
    if (rq.raw) out.times.resize(N);
    out.log.resize(N);
    out.stats.resize(N);
    out.final_h = fh;
    for (int tid = 0; tid < N; ++tid) {
        const int c = cnt[tid] < 0 ? 0 : (cnt[tid] > cap ? cap : cnt[tid]);
        const double* base = data.data() + (size_t)tid * cap * nx;
        auto& tr = out.traj[tid];
        tr.resize(c);
        for (int i = 0; i < c; ++i) tr[i].assign(base + (size_t)i * nx, base + (size_t)(i + 1) * nx);
        if (rq.raw) out.times[tid].assign(times.begin() + (size_t)tid * cap, times.begin() + (size_t)tid * cap + c);
        const int lc = rq.log_cap > 0 ? (logc[tid] < rq.log_cap ? logc[tid] : rq.log_cap) : 0;
        out.log[tid].assign(logs.begin() + (size_t)tid * rq.log_cap * 4,
                            logs.begin() + (size_t)tid * rq.log_cap * 4 + (size_t)lc * 4);
        const double* s = st.data() + (size_t)tid * 9;
        AdaptiveStats& a = out.stats[tid];
        a.nacc = s[0]; a.nrej = s[1]; a.nforced = s[2]; a.nrhs = s[3];
        a.hmin = s[4]; a.hmax = s[5]; a.hmean = s[6];
        a.diverged = s[7] != 0; a.truncated = s[8] != 0;
    }
    return true;
}
