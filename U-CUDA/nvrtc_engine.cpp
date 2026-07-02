#include "nvrtc_engine.h"
#include <cuda.h>
#include <nvrtc.h>
#include <cstdio>
#include <sstream>

// --- помощники проверки ошибок, пишут в error_ ---
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
    release_module();
    if (context_) { cuCtxDestroy((CUcontext)context_); context_ = nullptr; }
}

void NvrtcEngine::release_module() {
    if (module_) { cuModuleUnload((CUmodule)module_); module_ = nullptr; }
    kernel_ = nullptr;
    compiled_ = false;
}

bool NvrtcEngine::init() {
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
    if (!inited_ && !init()) return false;
    // Если этот worker-thread унаследовал чужой текущий контекст (например от
    // ParametricEngine, который работал в той же thread-pool ячейке), модуль
    // и kernel из нашего контекста дадут "invalid resource handle @ launch".
    // Жёстко выставляем НАШ контекст текущим перед любыми CU/NVRTC-вызовами.
    cuCtxSetCurrent((CUcontext)context_);
    release_module();
    amountOfX_ = amountOfX;

    // Собираем полный CUDA-исходник: тип, КРС как __device__, ядро траектории.
    // krs_body использует X[], a[], h (как выдаёт codegen). Тип numb=double тут.
    std::ostringstream src;
    src << "typedef double numb;\n"
        << "#define AMOUNTOFX " << amountOfX << "\n"
        << "__device__ __forceinline__ void calculateDiscreteModel(numb* X, const numb* a, numb h) {\n"
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
    NVOK(nvrtcCreateProgram(&prog, code.c_str(), "model.cu", 0, nullptr, nullptr), "createProgram");
    char arch[32];
    snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", cc_major_, cc_minor_);
    // --fmad=false: см. parametric_engine.cpp — выравниваем округление под CPU.
    const char* opts[] = { arch, "--fmad=false" };
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
    module_ = mod;
    CUfunction fn;
    CUOK(cuModuleGetFunction(&fn, mod, "phase_kernel"), "getFunction");
    kernel_ = fn;
    compiled_ = true;
    return true;
}

bool NvrtcEngine::run_phase_portraits(const std::vector<double>& ic_flat, int N,
    const std::vector<double>& values,
    double h, int total, int skip,
    std::vector<std::vector<std::vector<double>>>& out) {
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