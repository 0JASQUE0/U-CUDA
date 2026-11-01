#include "engine_impl.h"
using namespace ucuda::cuda_detail;

namespace ucuda::cuda_detail {
int launch_block_size(size_t sharedPerThread) {
    int b = get_gpu_block_size();
    while (b > kGpuBlockSizeMin && sharedPerThread * (size_t)b > 48u * 1024u)
        b /= 2;
    return b;
}
}

namespace ucuda::cuda_detail {
void add_profiling_nvrtc_opts(std::vector<const char*>& opts, std::string& storage) {
    char v[16];
    unsigned long n = env_value("UCUDA_NVRTC_LINEINFO", v, sizeof v);
    if (n > 0 && n < sizeof v && v[0] == '1') opts.push_back("--generate-line-info");
    n = env_value("UCUDA_NVRTC_MAXRREG", v, sizeof v);
    if (n > 0 && n < sizeof v) { storage = std::string("--maxrregcount=") + v; opts.push_back(storage.c_str()); }
}
}

namespace ucuda::cuda_detail {
bool nvrtc_prune_kernels() {
    char v[8];
    const unsigned long n = env_value("UCUDA_NVRTC_PRUNE", v, sizeof v);
    return !(n > 0 && n < sizeof v && v[0] == '0');
}
}

namespace ucuda::cuda_detail {
std::string prune_kernel_defines(const std::vector<const char*>& name_exprs, const std::string& src) {
    std::string d = "#define UCUDA_PRUNE_KERNELS\n";
    for (const char* e : name_exprs) { d += "#define UCUDA_KEEP_"; d += e; d += '\n'; }
    if (src.find("loopCalculateDiscreteModel_int(") != std::string::npos) d += "#define UCUDA_KEEP_LOOP\n";
    if (src.find("loopCalculateDiscreteModelPeaks_int(") != std::string::npos) d += "#define UCUDA_KEEP_PEAKS_LOOP\n";
    return d;
}
}

namespace ucuda::cuda_detail {
void dump_module_ptx(const char* src_name, const std::string& src, const std::string& ptx) {
    char dir[4096];
    const unsigned long n = env_value("UCUDA_NVRTC_PTX_DIR", dir, 4096);
    if (n == 0 || n >= 4096) return;
    static std::atomic<int> seq{0};
    const std::string base = std::string(dir, n) + "\\" + src_name + "_" + std::to_string(seq++);
    for (const auto& [ext, text] : { std::pair<const char*, const std::string*>{ ".cu", &src }, { ".ptx", &ptx } }) {
        FILE* f = nullptr;
        if (fopen_s(&f, (base + ext).c_str(), "wb") == 0 && f) { std::fwrite(text->data(), 1, text->size(), f); std::fclose(f); }
    }
}
}

namespace ucuda::cuda_detail {
const char* nvrtc_fmad_opt() {
    return get_nvrtc_fmad() ? "--fmad=true" : "--fmad=false";
}
}

namespace ucuda::cuda_detail {
bool gpu_free_budget(double reserve, size_t& out_bytes) {
    size_t freeMemory = 0, totalMemory = 0;
    if (cudaMemGetInfo(&freeMemory, &totalMemory) != cudaSuccess) return false;
    out_bytes = (size_t)((double)freeMemory * reserve);
    return true;
}
}

namespace ucuda::cuda_detail {
int par_or_var_2d(bool sweep_h_x, bool sweep_h_y, bool sweep_var_x, bool sweep_var_y) {
    if (sweep_h_x || sweep_h_y) return (sweep_h_x ? sweep_var_y : sweep_var_x) ? 0 : 1;
    if (sweep_var_x == sweep_var_y) return sweep_var_x ? 0 : 1;
    return 2;
}
}

namespace ucuda::cuda_detail {
int lyap_warm_blocks(double vt, double NT)
{
    return vt > 0 ? (int)ucuda_steps_per_block((numb)vt, (numb)NT) : 0;
}
}

namespace ucuda::cuda_detail {
bool cancel_requested(const std::shared_ptr<std::atomic<bool>>& c)
{
    return c && c->load(std::memory_order_relaxed);
}
}

namespace ucuda::cuda_detail {
void mark_cancelled_tail(std::vector<int>& flags, size_t j)
{
    for (size_t k = j; k < flags.size(); ++k) flags[k] = REGIME_CANCELLED;
}
}

namespace ucuda::cuda_detail {
void fill_nan_tail(std::vector<double>& v, size_t from)
{
    for (size_t k = from; k < v.size(); ++k) v[k] = std::numeric_limits<double>::quiet_NaN();
}
}

namespace ucuda::cuda_detail {
double lyap_value_from_kernel(double v, int& flag)
{
    if (v == (double)UCUDA_CANCELLED_VALUE) { flag = REGIME_CANCELLED; return std::numeric_limits<double>::quiet_NaN(); }
    if (v == 999.0 || v == -999.0)         { flag = REGIME_UNBOUND;   return std::numeric_limits<double>::quiet_NaN(); }
    flag = REGIME_OSCILLATION;
    return v;
}
}

namespace ucuda::cuda_detail {
size_t steps_from_time_size_t(double t, double h)
{
    if (!(h > 0.0) || !(t > 0.0)) return 0;
    const double s = t / h;
    if (!std::isfinite(s)) return 0;
    return (size_t)s;
}
}

namespace ucuda::cuda_detail {
int steps_from_time_int(double t, double h)
{
    if (!(h > 0.0) || !(t > 0.0)) return 0;
    const double s = t / h;
    if (!std::isfinite(s)) return 0;
    return (s >= 2147483647.0) ? 2147483647 : (int)s;
}
}

namespace ucuda::cuda_detail {
bool dae_z_axis(const std::string& body, int var) {
    const int nx = dae_nx_from_body(body);
    return nx > 0 && var >= nx;
}
}

namespace ucuda::cuda_detail {
void dae_point_start(const KrsCpuStep& step, const std::string& body, numb* x, const numb* a) {
    if (dae_nx_from_body(body) > 0) step.fn()(x, a, (numb)0);
}
}

namespace ucuda::cuda_detail {
double cont_sweep_value(int j, int nPts, double lo, double hi,
                               bool reverse, bool log_scale) {
    return (double)ucuda_node_value_cont(j, nPts, (numb)lo, (numb)hi, log_scale, reverse);
}
}

namespace ucuda::cuda_detail {
std::vector<numb> to_numb(const std::vector<double>& v) {
    return std::vector<numb>(v.begin(), v.end());
}
}

namespace ucuda::cuda_detail {
void fill_bif1d_snapshot(data_export::Bif1DSnapshot& sn, const Bifurcation1DRequest& req,
                                bool continuation, bool on_cpu) {
    sn.values.assign(req.base_values.begin(), req.base_values.end());
    sn.initial_conditions.assign(req.initial_conditions.begin(), req.initial_conditions.end());
    sn.tMax          = req.t_max;
    sn.transientTime = req.transient_time;
    sn.h             = req.h;
    sn.gpu_fmad      = get_nvrtc_fmad();
    sn.preScaller    = req.pre_scaller;
    sn.writableVar   = req.writable_var;
    sn.indexOfMutVar = req.sweep_over_var ? req.var_sweep_index : req.param_index;
    sn.range_lo      = req.param_lo;
    sn.range_hi      = req.param_hi;
    sn.log_scale     = req.log_scale;
    sn.continuation  = continuation;
    sn.continuation_reverse = continuation && req.continuation_reverse;
    sn.on_cpu        = on_cpu;
    sn.step_control  = req.adaptive.enabled ? req.adaptive.desc : std::string();
}
}

namespace ucuda::cuda_detail {
std::string exe_dir() { return platform::executable_dir(); }
}

namespace ucuda::cuda_detail {
std::string read_text_file(const std::string& path, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "failed to open " + path; return {}; }
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
}

namespace ucuda::cuda_detail {
std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}
}

std::string hash_key(const std::string& krs_body, int amountOfX) {
    // peak_config_epoch(): knobs configCUDA.h уходят в NVRTC как #define, т.е. при их смене тот
    // же КРС даёт другой PTX. Без epoch в ключе кэши модулей отдали бы старый модуль, и настройка
    // не применилась бы до перезапуска приложения.
    // fmad — то же самое, но через опцию компиляции. Здесь в ключ идёт САМО значение, а не счётчик
    // поколений: флаг бинарный, поэтому возврат к прежнему положению переиспользует уже собранный
    // модуль вместо лишней перекомпиляции.
    return std::to_string(std::hash<std::string>{}(krs_body)) + ":" +
           std::to_string(amountOfX) + ":pk" + std::to_string(peak_config_epoch()) +
           ":fm" + (get_nvrtc_fmad() ? "1" : "0") +
           ":rdc" + (get_nvrtc_rdc() ? "1" : "0");
}

namespace ucuda::cuda_detail {
std::string lyap_nx_prefix(const std::string& krs_body) {
    const int nx = dae_nx_from_body(krs_body);
    return nx > 0 ? "#define UCUDA_LYAP_NX " + std::to_string(nx) + "\n" : std::string();
}
}

namespace ucuda::cuda_detail {
bool ad_is_bdf(const AdaptiveRequest& ad) {
    return ad.prefix.find("UCUDA_BDF_DRIVER") != std::string::npos;
}
}

namespace ucuda::cuda_detail {
std::string time_prefix(const std::string& krs_body) {
    return step_body_uses_time(krs_body) ? kNonautonomousDefine : std::string();
}
}

namespace ucuda::cuda_detail {
int lyap_dim(const std::string& krs_body, int N) {
    const int nx = dae_nx_from_body(krs_body);
    return (nx > 0 && nx < N) ? nx : N;
}
}

namespace ucuda::cuda_detail {
std::string peak_config_defines() {
    const PeakConfig c = get_peak_config();
    std::ostringstream o;
    o << std::setprecision(17);
    o << "// --- injected by parametric_engine (GUI Settings) ---\n"
      << "#define doCalculatePeaks "    << (c.do_calculate_peaks   ? 1 : 0) << "\n"
      << "#define doInterpolatePeaks "  << (c.do_interpolate_peaks ? 1 : 0) << "\n"
      << "#define eps_fixed_point ((numb)"     << c.eps_fixed_point     << ")\n"
      << "#define eps_peak_delta ((numb)"      << c.eps_peak_delta      << ")\n"
      << "#define eps_interPeak_delta ((numb)" << c.eps_interPeak_delta << ")\n"
      << "#define peak_threshold ((numb)"      << c.peak_threshold      << ")\n"
      << "#define max_amount_of_peaks "        << c.max_amount_of_peaks << "\n";
    return o.str();
}
}

namespace ucuda::cuda_detail {
std::string cu_err(CUresult r) {
    const char* name = nullptr;
    cuGetErrorString(r, &name);
    return name ? std::string(name) : ("CUresult " + std::to_string((int)r));
}
}

namespace ucuda::cuda_detail {
double getValueByIdx_local(size_t idx, int nPts, double lo, double hi) {
    // Одна арифметика с ядром — ucuda_node_value (configCUDA.h). Раньше здесь стояло
    // lo + (hi-lo)*idx/(n-1): при idx = nPts-1 это не давало РОВНО hi, и CSV сообщал параметр, в
    // котором точка не считалась. Вырожденный nPts == 1 теперь тоже как в ядре (hi, а не lo).
    // Касты явные: при numb = float сетка обязана считаться во float — ровно так, как её увидит
    // ядро (ranges уходят туда через to_numb).
    return (double)ucuda_node_value((int)idx, nPts, (numb)lo, (numb)hi);
}
}

namespace ucuda::cuda_detail {
double getValueByIdx_log_local(size_t idx, int nPts, double lo, double hi) {
    // См. getValueByIdx_local: общая формула вместо копии. Здесь копия ещё и
    // умножала до деления, а ядро (getValueByIdxLog) — наоборот.
    return (double)ucuda_node_value_log((int)idx, nPts, (numb)lo, (numb)hi);
}
}

namespace ucuda::cuda_detail {
void write_curve1d_csv(const LLE1DResult& r, const std::string& path) {
    if (!path.empty()) data_export::export_lle1d(r, path);
}
}

namespace ucuda::cuda_detail {
void write_curve1d_csv(const LS1DResult& r, const std::string& path, int K) {
    if (path.empty()) return;
    if (K >= r.n_exponents) { data_export::export_ls1d(r, path); return; }
    LS1DResult t = r;
    t.n_exponents = K;
    for (auto& v : t.spectrum) if ((int)v.size() > K) v.resize((size_t)K);
    data_export::export_ls1d(t, path);
}
}

namespace ucuda::cuda_detail {
void dae_project(const KrsCpuStep& step, const numb* x, numb* v, const numb* a, int k, int n) {
    if (k >= n) return;
    for (int i = k; i < n; ++i) v[i] = x[i];
    step.fn()(v, a, (numb)0);
}
}

namespace ucuda::cuda_detail {
void metrics_finish(SignalMetricsResult& res, const SignalMetricsRequest& req) {
    for (int m = 0; m < SIGM_COUNT; ++m) {
        double lo =  std::numeric_limits<double>::infinity();
        double hi = -std::numeric_limits<double>::infinity();
        for (double v : res.values[m]) {
            if (!std::isfinite(v)) continue;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        res.min_val[m] = std::isfinite(lo) ? lo : 0.0;
        res.max_val[m] = std::isfinite(hi) ? hi : 0.0;
    }
    if (!req.csv_output_path.empty()) {
        std::string e;
        if (!signal_metrics_write_csv(res, req.csv_output_path, e))
            std::fprintf(stderr, "[metrics] CSV: %s\n", e.c_str());
    }
    res.ok = true;
    res.cancelled = cancel_requested(req.cancel);   // ok + cancelled = частичный результат (Stop)
}
}

int ParametricEngine::Impl::progress_stride_for(size_t stepsPerPoint) {
        int s = (int)(stepsPerPoint / 64);
        s -= s % kLoopCheckBlock;
        if (s < kLoopCheckBlock) s = kLoopCheckBlock;
        return s;
    }

ParametricEngine::Impl::~Impl() {
        if (inited) {
            cuCtxSetCurrent(context);
            drain_pool(pool_bif1d);
            drain_pool(pool_lle);
            drain_pool(pool_lle_2d);
            drain_pool(pool_ls);
            drain_pool(pool_ls_2d);
            drain_pool(pool_cont);
            drain_pool(pool_lle_cont);
            drain_pool(pool_ls_cont);
            drain_pool(pool_dft_cont);
            drain_pool(pool_dft_hsweep);
            drain_pool(pool_bif2d);
            drain_pool(pool_basins);
            drain_pool(pool_ad);
            drain_pool(pool_ad_cont);
            drain_pool(pool_lyap_ad);
            drain_pool(pool_lyap_ad_cont);
            drain_pool(pool_metrics_ad);
            drain_pool(pool_fs_attr);
            drain_pool(pool_fs_grid);
            drain_pool(pool_order);
            drain_pool(pool_network);
            drain_pool(pool_metrics);
            cuCtxDestroy(context);
        }
    }

bool ParametricEngine::Impl::ensure_init(std::string& err) {
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

bool ParametricEngine::Impl::load_sources(std::string& err) {
        // Перечитывать файлы с диска надо один раз, а вот текст configCUDA.h
        // пересобирать — на каждую смену peak-настроек (см. peak_config_defines).
        const uint64_t ep = peak_config_epoch();
        if (srcs_loaded && srcs_peak_epoch == ep) return true;
        if (srcs_loaded) {
            src_configCUDA_h = peak_config_defines() + src_configCUDA_h_raw;
            srcs_peak_epoch  = ep;
            return true;
        }
        std::string root = exe_dir() + "\\kernels\\";
        std::string e;
        src_template          = read_text_file(root + "bifurcation1d.template.cu",      e); if (!e.empty()) { err = e; return false; }
        src_template_cont     = read_text_file(root + "bifurcation1d_cont.template.cu", e); if (!e.empty()) { err = e; return false; }
        src_template_lle_cont = read_text_file(root + "lle1d_cont.template.cu",         e); if (!e.empty()) { err = e; return false; }
        src_template_ls_cont  = read_text_file(root + "ls1d_cont.template.cu",          e); if (!e.empty()) { err = e; return false; }
        src_template_dft_cont = read_text_file(root + "dft1d_cont.template.cu",         e); if (!e.empty()) { err = e; return false; }
        src_template_dft_hsweep = read_text_file(root + "dft1d_hsweep.template.cu",     e); if (!e.empty()) { err = e; return false; }
        src_template_lle      = read_text_file(root + "lle1d.template.cu",              e); if (!e.empty()) { err = e; return false; }
        src_template_lle_2d   = read_text_file(root + "lle2d.template.cu",              e); if (!e.empty()) { err = e; return false; }
        src_template_ls       = read_text_file(root + "ls1d.template.cu",               e); if (!e.empty()) { err = e; return false; }
        src_template_ls_2d    = read_text_file(root + "ls2d.template.cu",               e); if (!e.empty()) { err = e; return false; }
        src_template_bif2d    = read_text_file(root + "bifurcation2d.template.cu",     e); if (!e.empty()) { err = e; return false; }
        src_template_basins   = read_text_file(root + "basins.template.cu",            e); if (!e.empty()) { err = e; return false; }
        src_template_fs_attr  = read_text_file(root + "fastsync_attr.template.cu",     e); if (!e.empty()) { err = e; return false; }
        src_template_fs_grid  = read_text_file(root + "fastsync_grid.template.cu",     e); if (!e.empty()) { err = e; return false; }
        src_template_order    = read_text_file(root + "order.template.cu",            e); if (!e.empty()) { err = e; return false; }
        src_template_network  = read_text_file(root + "network.template.cu",          e); if (!e.empty()) { err = e; return false; }
        src_template_metrics  = read_text_file(root + "signal_metrics.template.cu",   e); if (!e.empty()) { err = e; return false; }
        src_cudaLibrary_cu    = read_text_file(root + "cudaLibrary.cu",            e); if (!e.empty()) { err = e; return false; }
        src_cudaLibrary_cuh   = read_text_file(root + "cudaLibrary.cuh",           e); if (!e.empty()) { err = e; return false; }
        src_cudaMacros_cuh    = read_text_file(root + "cudaMacros.cuh",            e); if (!e.empty()) { err = e; return false; }
        src_ucuda_adaptive_cuh = read_text_file(root + "ucuda_adaptive.cuh",       e); if (!e.empty()) { err = e; return false; }
        src_ucuda_bdf_cuh     = read_text_file(root + "ucuda_bdf.cuh",            e); if (!e.empty()) { err = e; return false; }
        src_adaptive_part     = read_text_file(root + "adaptive_part.cu",          e); if (!e.empty()) { err = e; return false; }
        src_metrics_adaptive_part = read_text_file(root + "metrics_adaptive_part.cu", e); if (!e.empty()) { err = e; return false; }
        src_lyap_adaptive_part = read_text_file(root + "lyapunov_adaptive_part.cu", e); if (!e.empty()) { err = e; return false; }
        src_configCUDA_h_raw  = read_text_file(root + "configCUDA.h",              e); if (!e.empty()) { err = e; return false; }
        src_configCUDA_h = peak_config_defines() + src_configCUDA_h_raw;
        srcs_peak_epoch  = ep;
        srcs_loaded = true;
        return true;
    }

ParametricEngine::Impl::SrcSnapshot ParametricEngine::Impl::snapshot_sources(const std::string& tmpl) const {
        return { tmpl, src_cudaLibrary_cu, src_cudaLibrary_cuh, src_cudaMacros_cuh, src_configCUDA_h,
                 src_ucuda_adaptive_cuh, src_ucuda_bdf_cuh };
    }

const ParametricEngine::Impl::KrsFn* ParametricEngine::Impl::krs_fn(const std::string& ph) {
        for (const KrsFn& f : kKrsFns) if (ph == f.ph) return &f;
        return nullptr;
    }

const ParametricEngine::Impl::KrsFn* ParametricEngine::Impl::krs_fn(const std::string& ph, bool nonauto) {
        if (nonauto)
            for (const KrsFn& f : kKrsFnsT) if (ph == f.ph) return &f;
        return krs_fn(ph);
    }

bool ParametricEngine::Impl::compile_ptx(const std::string& src, const char* src_name,
                     const std::vector<const char*>& headers_src,
                     const std::vector<const char*>& headers_name,
                     const std::vector<const char*>& name_exprs,
                     bool rdc, std::string& out_ptx, std::vector<std::string>& lowered,
                     std::string& err) {
        return nvrtc_compile_ptx(cc_major, cc_minor, src, src_name, headers_src, headers_name,
                                 name_exprs, rdc, out_ptx, lowered, err);
    }

bool ParametricEngine::Impl::lib_ptx_for(const SrcSnapshot& snap, const char* src_name, const std::string& lib_key,
                     const std::vector<std::pair<std::string, std::string>>& subs,
                     const std::vector<const char*>& name_exprs,
                     CachedLibPtx& out, std::string& err) {
        std::unique_lock<std::mutex> lk(ptx_mu);
        for (;;) {
            if (pool_lib_ptx.take(lib_key, out)) return true;
            if (lib_in_flight.insert(lib_key).second) break;
            lib_ptx_done.wait(lk);
        }
        lk.unlock();

        std::string src = snap.tmpl;
        for (const auto& sub : subs) {
            if (krs_fn(sub.first)) continue;
            src = replace_all(src, sub.first, sub.second);
        }
        std::string decls;
        const bool nonauto = snap.tmpl.find(kNonautonomousDefine) != std::string::npos;
        for (const KrsFn& f0 : kKrsFns) {
            const KrsFn& f = *krs_fn(f0.ph, nonauto);
            if (src.find(f.ph) == std::string::npos) continue;
            src = replace_all(src, f.ph, f.call);
            decls += std::string("\nextern \"C\" __device__ void ") + f.name + "(" + f.params + ");";
        }
        src = replace_all(src, "#include \"cudaLibrary.cuh\"", "#include \"cudaLibrary.cuh\"" + decls);

        const std::vector<const char*> hs = { snap.lib_cu.c_str(), snap.lib_cuh.c_str(),
                                              snap.macros_cuh.c_str(), snap.config_h.c_str(),
                                              snap.adaptive_cuh.c_str(), snap.bdf_cuh.c_str() };
        const std::vector<const char*> hn = { "cudaLibrary.cu", "cudaLibrary.cuh",
                                              "cudaMacros.cuh", "configCUDA.h", "ucuda_adaptive.cuh", "ucuda_bdf.cuh" };
        CachedLibPtx fresh;
        fresh.key = lib_key;
        const bool ok = compile_ptx(src, src_name, hs, hn, name_exprs, true,
                                    fresh.ptx, fresh.lowered, err);

        lk.lock();
        lib_in_flight.erase(lib_key);
        lib_ptx_done.notify_all();
        if (!ok) return false;
        std::vector<CachedLibPtx> evicted;
        pool_lib_ptx.insert(fresh, std::string(), evicted);
        out = fresh;
        return true;
    }

bool ParametricEngine::Impl::krs_ptx_for(const SrcSnapshot& snap,
                     const std::vector<std::pair<const KrsFn*, std::string>>& fns,
                     const std::string& amount_of_x, const std::string& krs_key,
                     std::string& out_ptx, std::string& err) {
        {
            std::lock_guard<std::mutex> lk(ptx_mu);
            CachedKrsPtx hit;
            if (pool_krs_ptx.take(krs_key, hit)) { out_ptx = hit.ptx; return true; }
        }

        std::string src = "#define AMOUNTOFX " + amount_of_x + "\n"
                          "#include \"configCUDA.h\"\n";
        for (const auto& fb : fns)
            src += std::string("extern \"C\" __device__ void ") + fb.first->name + "(" + fb.first->params
                 + ") {\n" + fb.second + "\n}\n";
        const std::vector<const char*> hs = { snap.config_h.c_str() };
        const std::vector<const char*> hn = { "configCUDA.h" };
        std::vector<std::string> ignored;
        if (!compile_ptx(src, "krs.cu", hs, hn, {}, true, out_ptx, ignored, err)) return false;

        std::lock_guard<std::mutex> lk(ptx_mu);
        CachedKrsPtx fresh;
        fresh.key = krs_key;
        fresh.ptx = out_ptx;
        std::vector<CachedKrsPtx> evicted;
        pool_krs_ptx.insert(fresh, std::string(), evicted);
        return true;
    }

bool ParametricEngine::Impl::link_and_load(const std::string& lib_ptx, const std::string& krs_ptx,
                       CUmodule& out_module, std::string& err) {
        CUlinkState st = nullptr;
        // UCUDA_NVRTC_LINEINFO=1 (см. build_module): строки исходника переживают и линковку.
        // UCUDA_NVRTC_MAXRREG — тот же потолок регистров: распределяет их именно линковщик.
        CUjit_option jopt[2];
        void*        jval[2];
        unsigned     nj = 0;
        char li[16];
        unsigned long lin = env_value("UCUDA_NVRTC_LINEINFO", li, sizeof li);
        if (lin > 0 && lin < sizeof li && li[0] == '1') { jopt[nj] = CU_JIT_GENERATE_LINE_INFO; jval[nj++] = (void*)(size_t)1; }
        lin = env_value("UCUDA_NVRTC_MAXRREG", li, sizeof li);
        if (lin > 0 && lin < sizeof li) { jopt[nj] = CU_JIT_MAX_REGISTERS; jval[nj++] = (void*)(size_t)std::atoi(li); }
        CUresult r = cuLinkCreate(nj, nj ? jopt : nullptr, nj ? jval : nullptr, &st);
        if (r != CUDA_SUCCESS) { err = "cuLinkCreate: " + cu_err(r); return false; }
        auto fail = [&](const std::string& what, CUresult rr) {
            err = what + ": " + cu_err(rr);
            cuLinkDestroy(st);
            return false;
        };
        r = cuLinkAddData(st, CU_JIT_INPUT_PTX, (void*)lib_ptx.data(), lib_ptx.size() + 1,
                          "lib", 0, nullptr, nullptr);
        if (r != CUDA_SUCCESS) return fail("cuLinkAddData(lib)", r);
        r = cuLinkAddData(st, CU_JIT_INPUT_PTX, (void*)krs_ptx.data(), krs_ptx.size() + 1,
                          "krs", 0, nullptr, nullptr);
        if (r != CUDA_SUCCESS) return fail("cuLinkAddData(krs)", r);

        void* cubin = nullptr; size_t cubin_size = 0;
        r = cuLinkComplete(st, &cubin, &cubin_size);
        if (r != CUDA_SUCCESS) return fail("cuLinkComplete", r);

        CUmodule mod = nullptr;
        r = cuModuleLoadData(&mod, cubin);
        cuLinkDestroy(st);
        if (r != CUDA_SUCCESS) { err = "cuModuleLoadData(linked): " + cu_err(r); return false; }
        out_module = mod;
        return true;
    }

bool ParametricEngine::Impl::build_module(const SrcSnapshot& snap_in,
                      const char* src_name,
                      const std::vector<std::pair<std::string, std::string>>& subs,
                      const std::vector<const char*>& name_exprs,
                      CUmodule& out_module,
                      std::vector<std::string>& lowered,
                      std::string& err) {
        // Явное время t (этап 2): номер шага проведён через ядра БД, LLE/LS, Metrics, бассейнов и
        // DFT на фиксированном шаге — их модули приходят с UCUDA_NONAUTONOMOUS (time_prefix).
        // Остальным (адаптивные свипы, FastSync, Order, Network) — понятный отказ, а не
        // "identifier ucuda_t is undefined" от NVRTC.
        const bool nonauto = snap_in.tmpl.find(kNonautonomousDefine) != std::string::npos;
        for (const auto& sub : subs)
            if ((sub.first == "{{KRS_BODY}}" || sub.first == "{{KRS_RHS_BODY}}")
                && step_body_uses_time(sub.second) && !nonauto) {
                err = "explicit time t: FastSync and Network do not support it yet";
                return false;
            }
        // Стадия предрасчёта (krs_prep.h): у шага с блоком констант ("CD fast") горячие
        // циклы cudaLibrary.cu зовут ucudaKrsPrep / calculateDiscreteModelP. Их определения
        // встают в шаблон: одной единицей трансляции — перед #include "cudaLibrary.cu"
        // (сам calculateDiscreteModel остаётся самодостаточным для прочих вызовов), при
        // раздельной сборке — обёртками над krs_prep / krs_step_p модуля КРС сразу после
        // #include "ucuda/cuda/cudaLibrary.cuh". Шаблон меняется, значит меняется и ключ библиотеки.
        KrsPrepSplit prep_sp;
        bool has_prep = false;
        for (const auto& sub : subs)
            if (sub.first == "{{KRS_BODY}}") has_prep = krs_split_prep(sub.second, prep_sp);
        SrcSnapshot snap = snap_in;
        // Раздельная сборка: библиотека без КРС компилируется один раз на (шаблон, размерность,
        // настройки) и переиспользуется, шаг живёт в своём модуле, склейка — cuLink. Замерено на
        // bifurcation2d + implicit midpoint: смена КРС 0.25 с против ~15 с одной единицей
        // трансляции. Цена — вызов вместо инлайна: на RTX 2060 SUPER от -1% до +7% времени счёта,
        // результаты побитово те же (FP64 идёт 1/32 скорости, накладные прячутся в её тени).
        if (get_nvrtc_rdc()) {
            std::vector<std::pair<const KrsFn*, std::string>> krs_fns;
            std::string krs_all, amount_of_x = "3";
            std::string lib_key = src_name;
            static const KrsFn kPrepFn  = { "", "krs_prep",   krs_prep_fn_params(),   "" };
            static const KrsFn kStepPFn = { "", "krs_step_p", krs_step_p_fn_params(), "" };
            if (has_prep) {
                krs_fns.emplace_back(&kPrepFn,  krs_prep_fn_body(prep_sp));
                krs_fns.emplace_back(&kStepPFn, krs_step_p_fn_body(prep_sp));
                // Тела обеих выводятся из KRS_BODY, который ниже и так входит в ключ.
                if (!krs_insert_after_line(snap.tmpl, "#include \"cudaLibrary.cuh\"",
                                           krs_prep_lib_defs(prep_sp))) {
                    // Шаблон без этой строки: шаг остаётся самодостаточным, циклы
                    // берут пустой предрасчёт из cudaLibrary.cu.
                    krs_fns.clear();
                    snap.tmpl = snap_in.tmpl;
                }
            }
            for (const auto& sub : subs) {
                if (const KrsFn* f = krs_fn(sub.first, nonauto)) {
                    krs_fns.emplace_back(f, sub.second);
                    krs_all += f->name; krs_all += '\x1f'; krs_all += sub.second; krs_all += '\x1f';
                    continue;
                }
                if (sub.first == "{{AMOUNT_OF_X}}") amount_of_x = sub.second;
                lib_key += ''; lib_key += sub.first; lib_key += '='; lib_key += sub.second;
            }
            lib_key += get_nvrtc_fmad() ? "|fm1" : "|fm0";
            lib_key += "|pk" + std::to_string(peak_config_epoch());
            // Шаблон и запрошенные ядра — тоже часть ключа. Имя исходника их не определяет:
            // FastSync собирает "fastsync.cu" и из fs_attr, и из fs_grid с одинаковыми подстановками,
            // и On Grid получал библиотеку On Attractor — чужие ядра (картинка-мусор, illegal
            // address) или lowered короче name_exprs (чтение за концом в cuModuleGetFunction).
            lib_key += "|t" + std::to_string(std::hash<std::string>{}(snap.tmpl));
            for (const char* e : name_exprs) { lib_key += '|'; lib_key += e; }

            std::string krs_err;
            std::string krs_ptx;
            const std::string krs_key = std::to_string(std::hash<std::string>{}(krs_all))
                                      + ":" + amount_of_x
                                      + (get_nvrtc_fmad() ? ":fm1" : ":fm0")
                                      + ":pk" + std::to_string(peak_config_epoch());
            if (!krs_ptx_for(snap, krs_fns, amount_of_x, krs_key, krs_ptx, krs_err)) {
                // Не компилируется САМ шаг — это ошибка пользователя, и монолитный путь выдал бы
                // ту же самую. Отдаём как есть, без второго захода на те же грабли.
                err = krs_err;
                return false;
            }

            CachedLibPtx lib;
            std::string lib_err;
            if (lib_ptx_for(snap, src_name, lib_key, subs, name_exprs, lib, lib_err)
                && lib.lowered.size() == name_exprs.size()
                && link_and_load(lib.ptx, krs_ptx, out_module, lib_err)) {
                lowered = lib.lowered;
                return true;
            }
            // Библиотека или линковка — не пользовательская ошибка: собираем по-старому, одной
            // единицей трансляции. Медленно, но работает везде, где работало раньше.
            std::fprintf(stderr, "[nvrtc] rdc split failed (%s), falling back to one TU\n",
                         lib_err.c_str());
        }

        // snap_in, а не snap: после неудачной раздельной сборки в snap.tmpl остались бы
        // её обёртки над krs_prep / krs_step_p.
        std::string src = snap_in.tmpl;
        for (const auto& sub : subs) src = replace_all(src, sub.first, sub.second);
        if (has_prep)
            krs_insert_before_line(src, "#include \"cudaLibrary.cu\"", krs_prep_inline_defs(prep_sp));
        if (nvrtc_prune_kernels())
            krs_insert_before_line(src, "#include \"cudaLibrary.cu\"", prune_kernel_defines(name_exprs, src));

        // Виртуальные заголовки для NVRTC. curand_kernel.h-stub НЕ нужен:
        // inline-stub в шаблонах + `#define CURAND_KERNEL_H_` блокируют как
        // реальный header (по -I path), так и любой повторный inject.
        const char* header_sources[] = {
            snap.lib_cu.c_str(),
            snap.lib_cuh.c_str(),
            snap.macros_cuh.c_str(),
            snap.config_h.c_str(),
            snap.adaptive_cuh.c_str(),
            snap.bdf_cuh.c_str(),
        };
        const char* header_names[] = {
            "cudaLibrary.cu",
            "cudaLibrary.cuh",
            "cudaMacros.cuh",
            "configCUDA.h",
            "ucuda_adaptive.cuh",
            "ucuda_bdf.cuh",
        };
        constexpr int n_headers = 6;

        nvrtcProgram prog = nullptr;
        nvrtcResult nr = nvrtcCreateProgram(&prog, src.c_str(), src_name,
                                            n_headers, header_sources, header_names);
        if (nr != NVRTC_SUCCESS) {
            err = std::string("nvrtcCreateProgram(") + src_name + "): " + nvrtcGetErrorString(nr);
            return false;
        }

        // Регистрируем kernel-имена ДО компиляции, чтобы потом через
        // nvrtcGetLoweredName достать их mangled-варианты для cuModuleGetFunction.
        for (const char* sym : name_exprs) nvrtcAddNameExpression(prog, sym);

        char arch[64];
        std::snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", cc_major, cc_minor);

        // NVRTC по умолчанию не знает CUDA-include путей (math_constants.h,
        // curand_kernel.h и т.п.). Берём CUDA_PATH из окружения — его проставляет
        // CUDA Toolkit installer.
        std::string cuda_include_opt;
        {
            char buf[4096];
            unsigned long nlen = env_value("CUDA_PATH", buf, 4096);
            if (nlen > 0 && nlen < 4096)
                cuda_include_opt = std::string("-I") + std::string(buf, nlen) + "\\include";
        }
        if (cuda_include_opt.empty()) {
            err = "the CUDA_PATH environment variable is not set - NVRTC will not find math_constants.h "
                  "(install the CUDA Toolkit or set CUDA_PATH=...)";
            nvrtcDestroyProgram(&prog);
            return false;
        }

        std::string std_opt = "--std=c++17";
        std::vector<const char*> opts = { arch, std_opt.c_str(), "-default-device",
                                          cuda_include_opt.c_str(), nvrtc_fmad_opt() };
        std::string dev_opt_storage;
        add_profiling_nvrtc_opts(opts, dev_opt_storage);

        nr = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
        if (nr != NVRTC_SUCCESS) {
            size_t logsz = 0; nvrtcGetProgramLogSize(prog, &logsz);
            std::string log;
            if (logsz > 1) { log.resize(logsz); nvrtcGetProgramLog(prog, &log[0]); }
            err = std::string("NVRTC compile failed (") + src_name + "):\n" + log;
            nvrtcDestroyProgram(&prog);
            return false;
        }

        size_t ptxsz = 0; nvrtcGetPTXSize(prog, &ptxsz);
        std::string ptx(ptxsz, '\0');
        nvrtcGetPTX(prog, &ptx[0]);
        dump_module_ptx(src_name, src, ptx);

        // Mangled-имена копируем в свои строки ДО nvrtcDestroyProgram — после
        // destroy указатели становятся невалидны.
        lowered.clear();
        lowered.reserve(name_exprs.size());
        for (const char* sym : name_exprs) {
            const char* lo = nullptr;
            nvrtcGetLoweredName(prog, sym, &lo);
            lowered.push_back(lo ? lo : sym);
        }
        nvrtcDestroyProgram(&prog);

        // Грузим во временную переменную: при неудаче у вызывающего слот кэша
        // остаётся нетронутым, и release_*_module() не получит мусорный handle.
        CUmodule mod = nullptr;
        // UCUDA_NVRTC_MAXRREG (add_profiling_nvrtc_opts): регистры распределяет JIT драйвера при
        // загрузке PTX, поэтому потолок — и сюда.
        CUjit_option mopt[1] = { CU_JIT_MAX_REGISTERS };
        void*        mval[1] = { nullptr };
        unsigned     nm = 0;
        {
            char v[16];
            const unsigned long n = env_value("UCUDA_NVRTC_MAXRREG", v, sizeof v);
            if (n > 0 && n < sizeof v) { mval[0] = (void*)(size_t)std::atoi(v); nm = 1; }
        }
        CUresult r = cuModuleLoadDataEx(&mod, ptx.c_str(), nm, nm ? mopt : nullptr, nm ? mval : nullptr);
        if (r != CUDA_SUCCESS) {
            err = std::string("cuModuleLoadDataEx(") + src_name + "): " + cu_err(r);
            return false;
        }
        out_module = mod;
        return true;
    }

bool ParametricEngine::Impl::module_fn(CUmodule mod, const std::string& name, CUfunction& out, std::string& err) {
        CUresult r = cuModuleGetFunction(&out, mod, name.c_str());
        if (r != CUDA_SUCCESS) {
            err = "cuModuleGetFunction(" + name + "): " + cu_err(r);
            return false;
        }
        return true;
    }

bool ParametricEngine::Impl::compile_if_needed(const std::string& krs_body, int amountOfX,
                           int par_or_var, std::string& err, bool activate ) {
        // Если worker-thread унаследовал чужой контекст (NvrtcEngine, например),
        // компиляция и загрузка модуля прицепят символы не в тот контекст. Жёстко
        // выставляем наш перед NVRTC/CU-вызовами.
        cuCtxSetCurrent(context);
        // par_or_var в kernel'е cudaLibrary.cu — compile-time макрос. Поэтому
        // включаем его в hash-key: param-sweep и IC-sweep кешируются отдельно.
        std::string key = hash_key(krs_body, amountOfX) + ":pov" + std::to_string(par_or_var);
        return compile_into(pool_bif1d, key, cached, activate, [&](CachedModule& fresh) {
            // DFT_custom уже присутствует в этом же модуле (шаблон #include'ит
            // cudaLibrary.cu целиком) — регистрируем его тоже, чтобы run_dft_1d
            // мог переиспользовать этот кэш без отдельной компиляции.
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(time_prefix(krs_body) + lyap_nx_prefix(krs_body) + src_template), "bifurcation1d.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body },
                                { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                              { "calculateDiscreteModelCUDA", "peakFinderCUDA", "DFT_custom_pf",
                                "calculateDiscreteModelPeaksCUDA" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_traj,  err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[1], fresh.kernel_peak,  err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[2], fresh.kernel_dft,   err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[3], fresh.kernel_fused, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_ad_if_needed(const AdaptiveRequest& ad, const std::string& krs_body, int amountOfX,
                              int par_or_var, std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string bodies = krs_body;
        for (const std::string* b : { &ad.prefix, &ad.rhs, &ad.emb, &ad.dprep, &ad.deval, &ad.ctrl_body }) { bodies += '\x1f'; bodies += *b; }
        // Узлы шага — свой модуль без плотного выхода (UCUDA_AD_NO_DENSE): состояние нити
        // легче на D, Xp, Fp, а стадии живут только внутри попытки. Своё имя исходника — от
        // него зависит ключ PTX библиотеки при раздельной сборке.
        const bool nodes = ad_nodes_module(ad);
        const std::string key = hash_key(bodies, amountOfX) + ":ad:pov" + std::to_string(par_or_var)
                              + (nodes ? ":nodes" : "");
        return compile_into(pool_ad, key, cached_ad, activate, [&](CachedAdModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            const std::string tmpl = (nodes ? std::string(kAdNodesModulePrefix) : std::string())
                                   + src_template_bif2d + "\n" + src_adaptive_part;
            if (!build_module(snapshot_sources(ad.prefix + time_prefix(ad.rhs) + tmpl), nodes ? "adaptive_sweep_nodes.cu" : "adaptive_sweep.cu",
                              { { "{{AMOUNT_OF_X}}",     std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",        krs_body },
                                { "{{PAR_OR_VAR}}",      std::to_string(par_or_var) },
                                { "{{KRS_RHS_BODY}}",    ad.rhs },
                                { "{{KRS_EMB_BODY}}",    ad.emb },
                                { "{{KRS_DPREP_BODY}}",  ad.dprep },
                                { "{{KRS_DEVAL_BODY}}",  ad.deval },
                                { "{{CTRL_CUSTOM}}",     adaptive_ctrl_source(ad.ctrl_body) } },
                              { "calculateDiscreteModelPeaksAdCUDA", "calculateDiscreteModelAvgPeaksAdCUDA",
                                "dbscanCUDA",
                                "dbscanWarpCUDA", "ucudaMaxPeaksCUDA", "dbscanWarp2DCUDA" },
                              mod, mg, err))
                return false;
            fresh.key    = key;
            fresh.module = mod;
            CUfunction* slots[] = { &fresh.kernel_peaks, &fresh.kernel_avg, &fresh.kernel_dbscan,
                                    &fresh.kernel_dbscan_warp, &fresh.kernel_max_peaks, &fresh.kernel_dbscan_warp2d };
            for (int i = 0; i < 6; ++i)
                if (!module_fn(mod, mg[(size_t)i], *slots[i], err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_ad_cont_if_needed(const AdaptiveRequest& ad, const std::string& krs_body, int amountOfX,
                                   std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string bodies = krs_body;
        for (const std::string* b : { &ad.prefix, &ad.rhs, &ad.emb, &ad.dprep, &ad.deval, &ad.ctrl_body }) { bodies += '\x1f'; bodies += *b; }
        const std::string key = hash_key(bodies, amountOfX) + ":ad_cont";
        return compile_into(pool_ad_cont, key, cached_ad_cont, activate, [&](CachedSimpleContModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            const std::string tmpl = "#define UCUDA_AD_NO_SWEEP_KERNELS 1\n#define UCUDA_AD_CONT_KERNEL 1\n"
                                   + src_template_bif2d + "\n" + src_adaptive_part;
            if (!build_module(snapshot_sources(ad.prefix + time_prefix(ad.rhs) + tmpl), "adaptive_cont.cu",
                              { { "{{AMOUNT_OF_X}}",     std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",        krs_body },
                                { "{{PAR_OR_VAR}}",      "1" },
                                { "{{KRS_RHS_BODY}}",    ad.rhs },
                                { "{{KRS_EMB_BODY}}",    ad.emb },
                                { "{{KRS_DPREP_BODY}}",  ad.dprep },
                                { "{{KRS_DEVAL_BODY}}",  ad.deval },
                                { "{{CTRL_CUSTOM}}",     adaptive_ctrl_source(ad.ctrl_body) } },
                              { "calculateDiscreteModelPeaksAdContCUDA" },
                              mod, mg, err))
                return false;
            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_lyap_ad_if_needed(const AdaptiveRequest& ad, const std::string& krs_body, int amountOfX,
                                   int par_or_var, std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string bodies = krs_body;
        for (const std::string* b : { &ad.prefix, &ad.rhs, &ad.emb, &ad.dprep, &ad.ctrl_body }) { bodies += '\x1f'; bodies += *b; }
        const std::string key = hash_key(bodies, amountOfX) + ":lyap_ad:pov" + std::to_string(par_or_var);
        return compile_into(pool_lyap_ad, key, cached_lyap_ad, activate, [&](CachedLyapAdModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            const std::string tmpl = "#define UCUDA_AD_NO_SWEEP_KERNELS 1\n#define UCUDA_AD_NO_DENSE 1\n"
                                     "#define UCUDA_AD_LYAPUNOV 1\n"
                                   + src_template_bif2d + "\n" + src_adaptive_part + "\n" + src_lyap_adaptive_part;
            if (!build_module(snapshot_sources(ad.prefix + time_prefix(ad.rhs) + tmpl), "lyapunov_adaptive.cu",
                              { { "{{AMOUNT_OF_X}}",     std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",        krs_body },
                                { "{{PAR_OR_VAR}}",      std::to_string(par_or_var) },
                                { "{{KRS_RHS_BODY}}",    ad.rhs },
                                { "{{KRS_EMB_BODY}}",    ad.emb },
                                { "{{KRS_DPREP_BODY}}",  ad_is_bdf(ad) ? ad.dprep : std::string() },
                                { "{{KRS_DEVAL_BODY}}",  "" },
                                { "{{CTRL_CUSTOM}}",     adaptive_ctrl_source(ad.ctrl_body) } },
                              { "lyapunovLLEAdCUDA", "lyapunovLSAdCUDA" },
                              mod, mg, err))
                return false;
            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_lle, err) || !module_fn(mod, mg[1], fresh.kernel_ls, err)) {
                cuModuleUnload(mod); return false;
            }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_lyap_ad_cont_if_needed(const AdaptiveRequest& ad, const std::string& krs_body, int amountOfX,
                                        std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string bodies = krs_body;
        for (const std::string* b : { &ad.prefix, &ad.rhs, &ad.emb, &ad.dprep, &ad.ctrl_body }) { bodies += '\x1f'; bodies += *b; }
        const std::string key = hash_key(bodies, amountOfX) + ":lyap_ad_cont";
        return compile_into(pool_lyap_ad_cont, key, cached_lyap_ad_cont, activate, [&](CachedLyapAdModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            const std::string tmpl = "#define UCUDA_AD_NO_SWEEP_KERNELS 1\n#define UCUDA_AD_NO_DENSE 1\n"
                                     "#define UCUDA_AD_LYAPUNOV 1\n#define UCUDA_LYAP_AD_CONT_ONLY 1\n"
                                   + src_template_bif2d + "\n" + src_adaptive_part + "\n" + src_lyap_adaptive_part;
            if (!build_module(snapshot_sources(ad.prefix + time_prefix(ad.rhs) + tmpl), "lyapunov_adaptive_cont.cu",
                              { { "{{AMOUNT_OF_X}}",     std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",        krs_body },
                                { "{{PAR_OR_VAR}}",      "1" },
                                { "{{KRS_RHS_BODY}}",    ad.rhs },
                                { "{{KRS_EMB_BODY}}",    ad.emb },
                                { "{{KRS_DPREP_BODY}}",  ad_is_bdf(ad) ? ad.dprep : std::string() },
                                { "{{KRS_DEVAL_BODY}}",  "" },
                                { "{{CTRL_CUSTOM}}",     adaptive_ctrl_source(ad.ctrl_body) } },
                              { "lyapunovLLEAdContCUDA", "lyapunovLSAdContCUDA" },
                              mod, mg, err))
                return false;
            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_lle, err) || !module_fn(mod, mg[1], fresh.kernel_ls, err)) {
                cuModuleUnload(mod); return false;
            }
            return true;
        }, err);
    }

std::string ParametricEngine::Impl::ad_axis_range_error(int kind, double lo, double hi) {
        if ((kind == kAdAxisRtol || kind == kAdAxisAtol || kind == kAdAxisTol) && !(lo > 0.0 && hi > 0.0))
            return "lo/hi must be > 0 on the rtol / atol axis";
        return {};
    }

std::string ParametricEngine::Impl::ad_axes_overlap_error(int kx, int ky) {
        auto fields = [](int k) -> unsigned {
            if (k == kAdAxisRtol) return 1u;
            if (k == kAdAxisAtol) return 2u;
            if (k == kAdAxisTol)  return 3u;
            if (k >= kAdAxisCtrl && k < kAdAxisCtrl + UCUDA_CTL_NPAR) return 4u << (k - kAdAxisCtrl);
            return 0u;
        };
        if (fields(kx) & fields(ky))
            return "both axes sweep the same step setting (rtol / atol / tol / controller parameter)";
        return {};
    }

bool ParametricEngine::Impl::ad_nodes_module(const AdaptiveRequest& ad) {
        return ad.raw_nodes && ad.axis_kind[0] == kAdAxisSystem && ad.axis_kind[1] == kAdAxisSystem;
    }

void* ParametricEngine::Impl::ad_param_arg(const AdaptiveRequest& ad, UcudaAdaptParams** d_ptr) {
        return ad_nodes_module(ad) ? (void*)const_cast<UcudaAdaptParams*>(&ad.params) : (void*)d_ptr;
    }

CUresult ParametricEngine::Impl::launch_lyap_ad(bool ls, AdDeviceArgs& a, const AdaptiveRequest& ad, int nPts, int cur,
                            size_t calculated, int dimension, numb* d_ranges, int* d_idx, numb* d_ic,
                            int nIC, numb* d_vals, int nVals, double maxValue, numb* d_res, int logAxisMask,
                            double transientTime, double tMax, double NT, double eps, double vectorTransient,
                            RunSignals& sig, int progressStride) {
        const size_t sharedPerThread = (size_t)ucuda_shared_stride(nIC, nVals) * sizeof(numb);
        int blockSize = launch_block_size(sharedPerThread);
        if (blockSize < 1) blockSize = 1;
        const int gridSize  = (cur + blockSize - 1) / blockSize;
        numb   maxValue_a = (numb)maxValue, tolRatio = (numb)ad.tol_ratio, tTr = (numb)transientTime;
        numb   tMax_a = (numb)tMax, NT_a = (numb)NT, eps_a = (numb)eps;
        int    nBlocks = (int)(tMax / NT), renorm = ad.lyap_renorm;
        int    nWarm = lyap_warm_blocks(vectorTransient, NT);
        size_t units = kAdProgressUnits;
        UcudaAdaptParams* dP = a.params;
        int*   dAx = a.axis;
        numb*  dSt = a.stats;
        int*   d_cancel = sig.cancelArg();
        int*   d_prog   = sig.progressArg();
        void* args[] = { &nPts, &cur, &calculated, &dimension, &d_ranges, &d_idx, &d_ic, &nIC, &d_vals, &nVals,
                         &maxValue_a, &d_res, &logAxisMask, &dP, &dAx, &tolRatio, &tTr, &tMax_a, &NT_a,
                         &nBlocks, &nWarm, &eps_a, &renorm, &d_cancel, &d_prog, &progressStride, &units, &dSt };
        return cuLaunchKernel(ls ? cached_lyap_ad.kernel_ls : cached_lyap_ad.kernel_lle,
                              gridSize, 1, 1, blockSize, 1, 1, (unsigned int)(sharedPerThread * blockSize),
                              nullptr, args, nullptr);
    }

bool ParametricEngine::Impl::compile_lle_if_needed(const std::string& krs_body, int amountOfX,
                               int par_or_var, std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        // par_or_var в kernel — compile-time макрос (см. bif1d). Два PTX-модуля
        // кешируются отдельно: param-sweep и IC-sweep.
        std::string key = hash_key(krs_body, amountOfX) + ":lle:pov" + std::to_string(par_or_var);
        return compile_into(pool_lle, key, cached_lle, activate, [&](CachedLleModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(time_prefix(krs_body) + lyap_nx_prefix(krs_body) + src_template_lle), "lle1d.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body },
                                { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                              { "LLEKernelCUDA" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_lle, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_lle_2d_if_needed(const std::string& krs_body, int amountOfX,
                                  int par_or_var, std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":lle2d:pov" + std::to_string(par_or_var);
        return compile_into(pool_lle_2d, key, cached_lle_2d, activate, [&](CachedLleModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(time_prefix(krs_body) + lyap_nx_prefix(krs_body) + src_template_lle_2d), "lle2d.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body },
                                { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                              { "LLEKernelCUDA" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_lle, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_ls_if_needed(const std::string& krs_body, int amountOfX,
                              int par_or_var, std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":ls:pov" + std::to_string(par_or_var);
        return compile_into(pool_ls, key, cached_ls, activate, [&](CachedLsModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(time_prefix(krs_body) + lyap_nx_prefix(krs_body) + src_template_ls), "ls1d.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body },
                                { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                              { "LSKernelCUDA" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_ls, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_ls_2d_if_needed(const std::string& krs_body, int amountOfX,
                                 int par_or_var, std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":ls2d:pov" + std::to_string(par_or_var);
        return compile_into(pool_ls_2d, key, cached_ls_2d, activate, [&](CachedLsModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(time_prefix(krs_body) + lyap_nx_prefix(krs_body) + src_template_ls_2d), "ls2d.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body },
                                { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                              { "LSKernelCUDA" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_ls, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_bif1d_cont_if_needed(const std::string& krs_body, int amountOfX,
                                      std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":cont";
        return compile_into(pool_cont, key, cached_cont, activate, [&](CachedContModule& fresh) {
            // bifurcation1dContinuationKernel — extern "C" (имя не мангается), поэтому в name_exprs
            // не идёт и берётся из модуля напрямую. peakFinderCUDA и DFT_custom — обычные C++
            // символы, нужны mangled-варианты; DFT_custom уже в этом модуле (шаблон #include'ит
            // cudaLibrary.cu целиком) и нужен continuation-ветке run_dft_1d.
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(time_prefix(krs_body) + lyap_nx_prefix(krs_body) + src_template_cont), "bifurcation1d_cont.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body } },
                              { "peakFinderCUDA", "DFT_custom_pf" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, "bifurcation1dContinuationKernel", fresh.kernel_cont, err))
                { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[0], fresh.kernel_peak, err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[1], fresh.kernel_dft,  err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_simple_cont_if_needed(ModuleLru<CachedSimpleContModule>& pool,
                                       CachedSimpleContModule& slot,
                                       const std::string& tmpl,
                                       const char* kernel_name,
                                       const char* src_name,
                                       const char* key_suffix,
                                       const std::string& krs_body, int amountOfX,
                                       std::string& err) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + key_suffix;
        return compile_into(pool, key, slot, true, [&](CachedSimpleContModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;   // пуст: единственный символ — extern "C"
            if (!build_module(snapshot_sources(time_prefix(krs_body) + lyap_nx_prefix(krs_body) + tmpl), src_name,
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body } },
                              {},
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, kernel_name, fresh.kernel, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_bif2d_if_needed(const std::string& krs_body, int amountOfX,
                                 int par_or_var, std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":bif2d:" + std::to_string(par_or_var);
        return compile_into(pool_bif2d, key, cached_bif2d, activate, [&](CachedBif2dModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(time_prefix(krs_body) + lyap_nx_prefix(krs_body) + src_template_bif2d), "bifurcation2d.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body },
                                { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                              { "calculateDiscreteModelPeaksCUDA", "dbscanCUDA",
                                "dbscanWarpCUDA", "ucudaMaxPeaksCUDA", "dbscanWarp2DCUDA" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_fused,  err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[1], fresh.kernel_dbscan, err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[2], fresh.kernel_dbscan_warp, err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[3], fresh.kernel_max_peaks,   err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[4], fresh.kernel_dbscan_warp2d, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_metrics_if_needed(const std::string& krs_body, int amountOfX,
                                   int par_or_var, std::string& err, bool activate ,
                                   bool minmax ) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":metrics:" + std::to_string(par_or_var)
                        + (minmax ? ":mm" : "");
        return compile_into(pool_metrics, key, cached_metrics, activate, [&](CachedMetricsModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            const std::string tmpl = minmax ? "#define SIGM_MINMAX_INTERP 1\n" + src_template_metrics
                                            : src_template_metrics;
            if (!build_module(snapshot_sources(time_prefix(krs_body) + lyap_nx_prefix(krs_body) + tmpl), minmax ? "signal_metrics_mm.cu" : "signal_metrics.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body },
                                { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                              { "calculateDiscreteModelMetricsCUDA", "signalMetricsContinuationKernel" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel,      err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[1], fresh.kernel_cont, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_metrics_ad_if_needed(const AdaptiveRequest& ad, const std::string& krs_body, int amountOfX,
                                      int par_or_var, std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string bodies = krs_body;
        for (const std::string* b : { &ad.prefix, &ad.rhs, &ad.emb, &ad.dprep, &ad.deval, &ad.ctrl_body }) { bodies += '\x1f'; bodies += *b; }
        // Узлы шага — модуль без плотного выхода, как у БД (см. compile_ad_if_needed).
        const bool nodes = ad_nodes_module(ad);
        const std::string key = hash_key(bodies, amountOfX) + ":metrics_ad:pov" + std::to_string(par_or_var)
                              + (nodes ? ":nodes" : "");
        return compile_into(pool_metrics_ad, key, cached_metrics_ad, activate, [&](CachedMetricsModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            // Ядра свипов БД и бассейнов из adaptive_part.cu метрикам не нужны — только время ptxas.
            const std::string tmpl = std::string(nodes ? kAdNodesModulePrefix : "")
                                   + "#define SIGM_MINMAX_INTERP 1\n#define UCUDA_AD_NO_SWEEP_KERNELS 1\n"
                                   + src_template_metrics + "\n"
                                   + src_adaptive_part + "\n" + src_metrics_adaptive_part;
            if (!build_module(snapshot_sources(ad.prefix + time_prefix(ad.rhs) + tmpl), nodes ? "signal_metrics_ad_nodes.cu" : "signal_metrics_ad.cu",
                              { { "{{AMOUNT_OF_X}}",     std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",        krs_body },
                                { "{{PAR_OR_VAR}}",      std::to_string(par_or_var) },
                                { "{{KRS_RHS_BODY}}",    ad.rhs },
                                { "{{KRS_EMB_BODY}}",    ad.emb },
                                { "{{KRS_DPREP_BODY}}",  ad.dprep },
                                { "{{KRS_DEVAL_BODY}}",  ad.deval },
                                { "{{CTRL_CUSTOM}}",     adaptive_ctrl_source(ad.ctrl_body) } },
                              { "calculateDiscreteModelMetricsAdCUDA", "signalMetricsContinuationAdKernel" },
                              mod, mg, err))
                return false;
            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel,      err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[1], fresh.kernel_cont, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

std::string ParametricEngine::Impl::metrics_axes(const SignalMetricsRequest& req, MetricsAxes& a) {
        auto check_param = [&](int p) { return p >= 0 && p < (int)req.base_values.size(); };
        auto check_var   = [&](int v) { return v >= 0 && v < req.amountOfX; };

        if (req.sweep_over_var && dae_z_axis(req.krs_body, req.var_sweep_index)) return kDaeZAxisMsg;
        if (req.dimension == 2 && req.sweep_over_var_2 && dae_z_axis(req.krs_body, req.var_sweep_index_2)) return kDaeZAxisMsg;
        if (req.log_scale && !(req.param_lo > 0.0 && req.param_hi > 0.0))
            return "log scale requires param lo/hi > 0 (X axis)";

        // Адаптивный шаг: h не свипается; ось настройки шага разбирается как ось h,
        // но h-осью для ядра не становится (hSweepAxis = -1).
        const bool ad   = req.adaptive.enabled;
        const bool ad_x = ad && req.adaptive.axis_kind[0] != kAdAxisSystem;
        const bool ad_y = ad && req.adaptive.axis_kind[1] != kAdAxisSystem;
        if (ad && (req.sweep_over_h || req.sweep_over_h_2))
            return "the step h is not swept with the adaptive step: sweep rtol / atol instead";
        const bool over_h  = req.sweep_over_h   || ad_x;
        const bool over_h2 = req.sweep_over_h_2 || ad_y;
        auto any_sign = [&](int axis) { return ad && req.adaptive.axis_kind[axis] >= kAdAxisCtrl; };

        if (req.dimension == 1) {
            if (over_h) {
                if (!any_sign(0) && (req.param_lo <= 0.0 || req.param_hi <= 0.0))
                    return "lo/hi must be > 0 on the axis of h / rtol / atol";
                a.hSweepAxis = ad ? -1 : 0;
                a.par_or_var = 1;
                a.idx_x = 0;
            } else if (req.sweep_over_var) {
                if (!check_var(req.var_sweep_index)) return "var_sweep_index out of range";
                a.par_or_var = 0;
                a.idx_x = req.var_sweep_index;
            } else {
                if (!check_param(req.param_index)) return "param_index out of range";
                a.par_or_var = 1;
                a.idx_x = req.param_index;
            }
            a.lo_x = req.param_lo; a.hi_x = req.param_hi;
            a.lo_y = 0.0;          a.hi_y = 1.0;
            a.logAxisMask = req.log_scale ? 1 : 0;
            return {};
        }

        if (ad) {
            const std::string e3 = ad_axes_overlap_error(req.adaptive.axis_kind[0], req.adaptive.axis_kind[1]);
            if (!e3.empty()) return e3;
        }
        if (req.log_scale_2 && !(req.param_lo_2 > 0.0 && req.param_hi_2 > 0.0))
            return "log scale requires param lo/hi > 0 (Y axis)";
        if (req.sweep_over_h && req.sweep_over_h_2)
            return "only one axis can sweep over dt (h)";

        bool log_x = req.log_scale, log_y = req.log_scale_2;
        if (over_h && over_h2) {
            // Обе оси — настройки шага (скажем, rtol x atol): система не свипается.
            a.par_or_var = 1;
            a.idx_x = 0; a.idx_y = 0;
            a.lo_x = req.param_lo;   a.hi_x = req.param_hi;
            a.lo_y = req.param_lo_2; a.hi_y = req.param_hi_2;
            a.logAxisMask = (log_x ? 1 : 0) | (log_y ? 2 : 0);
            return {};
        }
        a.par_or_var = par_or_var_2d(over_h, over_h2,
                                     req.sweep_over_var, req.sweep_over_var_2);
        if (over_h || over_h2) {
            a.hSweepAxis = ad ? -1 : (over_h ? 0 : 1);
            if (over_h) {
                if (a.par_or_var == 1) { if (!check_param(req.param_index_2))   return "param_index_2 (Y axis) out of range"; a.idx_y = req.param_index_2; }
                else                   { if (!check_var(req.var_sweep_index_2)) return "var_sweep_index_2 (Y axis) out of range"; a.idx_y = req.var_sweep_index_2; }
                a.idx_x = 0;
            } else {
                if (a.par_or_var == 1) { if (!check_param(req.param_index))     return "param_index (X axis) out of range"; a.idx_x = req.param_index; }
                else                   { if (!check_var(req.var_sweep_index))   return "var_sweep_index (X axis) out of range"; a.idx_x = req.var_sweep_index; }
                a.idx_y = 0;
            }
            const double h_lo = over_h ? req.param_lo : req.param_lo_2;
            const double h_hi = over_h ? req.param_hi : req.param_hi_2;
            if (!any_sign(over_h ? 0 : 1) && (h_lo <= 0.0 || h_hi <= 0.0))
                return std::string("lo/hi must be > 0 on the axis of h / rtol / atol (axis ")
                       + (over_h ? "X" : "Y") + ")";
            a.lo_x = req.param_lo;   a.hi_x = req.param_hi;
            a.lo_y = req.param_lo_2; a.hi_y = req.param_hi_2;
        } else if (req.sweep_over_var == req.sweep_over_var_2) {
            if (a.par_or_var == 1) {
                if (!check_param(req.param_index))   return "param_index (X axis) out of range";
                if (!check_param(req.param_index_2)) return "param_index_2 (Y axis) out of range";
                a.idx_x = req.param_index; a.idx_y = req.param_index_2;
            } else {
                if (!check_var(req.var_sweep_index))   return "var_sweep_index (X axis) out of range";
                if (!check_var(req.var_sweep_index_2)) return "var_sweep_index_2 (Y axis) out of range";
                a.idx_x = req.var_sweep_index; a.idx_y = req.var_sweep_index_2;
            }
            a.lo_x = req.param_lo;   a.hi_x = req.param_hi;
            a.lo_y = req.param_lo_2; a.hi_y = req.param_hi_2;
        } else if (req.sweep_over_var && !req.sweep_over_var_2) {
            if (!check_var(req.var_sweep_index))  return "var_sweep_index (X axis) out of range";
            if (!check_param(req.param_index_2))  return "param_index_2 (Y axis) out of range";
            a.idx_x = req.var_sweep_index; a.idx_y = req.param_index_2;
            a.lo_x = req.param_lo;   a.hi_x = req.param_hi;
            a.lo_y = req.param_lo_2; a.hi_y = req.param_hi_2;
        } else {
            if (!check_var(req.var_sweep_index_2)) return "var_sweep_index_2 (Y axis) out of range";
            if (!check_param(req.param_index))     return "param_index (X axis) out of range";
            a.idx_x = req.var_sweep_index_2; a.idx_y = req.param_index;
            a.lo_x = req.param_lo_2; a.hi_x = req.param_hi_2;
            a.lo_y = req.param_lo;   a.hi_y = req.param_hi;
            a.swap_xy = true;
            log_x = req.log_scale_2; log_y = req.log_scale;
        }
        a.logAxisMask = (log_x ? 1 : 0) | (log_y ? 2 : 0);
        return {};
    }

SignalMetricsResult ParametricEngine::Impl::run_signal_metrics(const SignalMetricsRequest& req) {
        SignalMetricsResult res;
        auto fail = [&](const std::string& msg) -> SignalMetricsResult& { res.error = msg; return res; };

        if (req.adaptive.enabled && !req.adaptive.setup_error.empty()) return fail(req.adaptive.setup_error);
        if (req.krs_body.empty())                                    return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX out of [1," + std::to_string(kMaxAmountOfX) + "]");
        if ((int)req.initial_conditions.size() != req.amountOfX)     return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)        return fail("too many base_values");
        if (req.dimension != 1 && req.dimension != 2)                return fail("dimension must be 1 or 2");
        if (req.n_pts <= 0)          return fail("n_pts must be > 0");
        if (req.h <= 0.0)            return fail("h must be > 0");
        if (req.t_max <= 0.0)        return fail("t_max must be > 0");
        if (req.transient_time < 0)  return fail("transient_time must be >= 0");
        if (req.pre_scaller <= 0)    return fail("pre_scaller must be > 0");
        if (req.writable_var < -1 || req.writable_var >= req.amountOfX) return fail("writable_var out of range");
        const int mask = req.metric_mask & kSignalMetricAllMask;
        if (mask == 0)               return fail("no metrics selected");

        MetricsAxes ax;
        {
            const std::string e = metrics_axes(req, ax);
            if (!e.empty()) return fail(e);
        }

        // Continuation и CPU — только 1D; обе ветки уходят до ensure_init
        // (CPU-ветке CUDA не нужна вовсе).
        const bool ad = req.adaptive.enabled;
        if (req.continuation) {
            if (req.dimension != 1)  return fail("continuation works in 1D only");
            if (req.sweep_over_var)  return fail("continuation requires a param or h sweep, not an IC sweep");
            // Адаптивный: CPU — нативная цепочка (run_metrics_cpu_ad), GPU — ядро в одну нить.
            if (ad)                  return req.use_cpu ? run_metrics_cpu_ad(req, ax) : run_metrics_continuation_ad(req);
            if (req.use_cpu)         return run_metrics_cpu(req, true);
            return run_metrics_continuation_gpu(req);
        }
        if (req.use_cpu) {
            if (req.dimension != 1)  return fail("CPU computation works in 1D only");
            return ad ? run_metrics_cpu_ad(req, ax) : run_metrics_cpu(req, false);
        }

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (ad) {
            if (!compile_metrics_ad_if_needed(req.adaptive, req.krs_body, req.amountOfX, ax.par_or_var, err)) return fail(err);
        }
        else if (!compile_metrics_if_needed(req.krs_body, req.amountOfX, ax.par_or_var, err, true,
                                            req.minmax_interp)) return fail(err);

        const int    nPts          = req.n_pts;
        const int    dimension     = req.dimension;
        const int    amountOfIC    = req.amountOfX;
        const int    amountOfValues= (int)req.base_values.size();
        const int    preScaller    = req.pre_scaller;
        const std::vector<numb> ic_staged     = to_numb(req.initial_conditions);
        const std::vector<numb> values_staged = to_numb(req.base_values);
        numb ranges[4]           = { (numb)ax.lo_x, (numb)ax.hi_x, (numb)ax.lo_y, (numb)ax.hi_y };
        int  indicesOfMutVars[2] = { ax.idx_x, ax.idx_y };

        // Буфер — под худший случай шага (минимальный h на h-оси), см. run_bif2d.
        const double worstCaseH = (ax.hSweepAxis == 0) ? ranges[0] : (ax.hSweepAxis == 1) ? ranges[2] : req.h;
        int amountOfPointsInBlock = (int)std::ceil(req.t_max / worstCaseH / preScaller);
        // Узлы шага: отсчётов сетки нет, строка интервалов — под потолок пиков (как у БД).
        const bool adRaw = ad && req.adaptive.raw_nodes;
        if (adRaw) amountOfPointsInBlock = (int)max_amount_of_peaks + 1;
        const size_t amountOfPointsForSkip = steps_from_time_size_t(req.transient_time, req.h);
        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max/h/pre_scaller too small)");

        // Интервалы нужны одной медиане; "+1" — стадия интервалов съедает один
        // сырой пик (см. run_bif2d).
        const bool   needIntervals = ((mask >> SIGM_MEDIAN_FREQ) & 1) != 0;
        const size_t peakStride    = (size_t)amountOfPointsInBlock < (size_t)max_amount_of_peaks + 1
                                   ? (size_t)amountOfPointsInBlock
                                   : (size_t)max_amount_of_peaks + 1;
        const int    peakCapacity  = (int)peakStride;

        const size_t total_cells = dimension == 2 ? (size_t)nPts * (size_t)nPts : (size_t)nPts;

        size_t freeMemory = 0;
        if (!gpu_free_budget(0.92, freeMemory)) return fail("cudaMemGetInfo failed");
        const size_t memPerCell  = (size_t)SIGM_COUNT * sizeof(numb) + sizeof(int)
                                 + (needIntervals ? peakStride * sizeof(numb) : 0);
        const size_t memConstants = (4 + (size_t)amountOfIC + (size_t)amountOfValues) * sizeof(numb) + 2 * sizeof(int);
        if (memConstants >= freeMemory) return fail("not enough GPU memory for constants");

        size_t nPtsLimiter = (freeMemory - memConstants) / memPerCell;
        if (nPtsLimiter < 32)          nPtsLimiter = 32;
        if (nPtsLimiter > total_cells) nPtsLimiter = total_cells;
        const size_t chunk = nPtsLimiter;
        const size_t amountOfIteration = (total_cells + chunk - 1) / chunk;

        const size_t stepsPerCell   = ad ? kAdProgressUnits
                                         : amountOfPointsForSkip + (size_t)amountOfPointsInBlock;
        const int    progressStride = progress_stride_for(stepsPerCell);
        const double ticksPerCell   = (double)(stepsPerCell / (size_t)progressStride);
        const double ticksTotal     = (double)total_cells * ticksPerCell;

        numb* d_ranges    = nullptr;
        int*  d_indices   = nullptr;
        numb* d_ic        = nullptr;
        numb* d_values    = nullptr;
        numb* d_intervals = nullptr;
        numb* d_out       = nullptr;
        int*  d_flags     = nullptr;
        RunSignals sig;
        CUstream stream = nullptr;
        AdDeviceArgs adArgs;                  // адаптивный шаг: параметры, оси, статистика

        auto cleanup = [&]() {
            adArgs.release();
            if (d_ranges)    cudaFree(d_ranges);
            if (d_indices)   cudaFree(d_indices);
            if (d_ic)        cudaFree(d_ic);
            if (d_values)    cudaFree(d_values);
            if (d_intervals) cudaFree(d_intervals);
            if (d_out)       cudaFree(d_out);
            if (d_flags)     cudaFree(d_flags);
            sig.release();
            if (stream)      cuStreamDestroy(stream);
        };

        #define SIGM_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define SIGM_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)
        #define SIGM_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        SIGM_CHECK(cudaMalloc((void**)&d_ranges,  4 * sizeof(numb)),                           "cudaMalloc d_ranges");
        SIGM_CHECK(cudaMalloc((void**)&d_indices, 2 * sizeof(int)),                            "cudaMalloc d_indices");
        SIGM_CHECK(cudaMalloc((void**)&d_ic,      (size_t)amountOfIC * sizeof(numb)),          "cudaMalloc d_ic");
        SIGM_CHECK(cudaMalloc((void**)&d_values,  (size_t)amountOfValues * sizeof(numb)),      "cudaMalloc d_values");
        SIGM_CHECK(cudaMalloc((void**)&d_out,     chunk * (size_t)SIGM_COUNT * sizeof(numb)),  "cudaMalloc d_out");
        SIGM_CHECK(cudaMalloc((void**)&d_flags,   chunk * sizeof(int)),                        "cudaMalloc d_flags");
        if (needIntervals)
            SIGM_CHECK(cudaMalloc((void**)&d_intervals, chunk * peakStride * sizeof(numb)),    "cudaMalloc d_intervals");
        if (!sig.alloc(res.error)) { cleanup(); return res; }
        if (ad) {
            // swap_xy — только при двух системных осях, коды осей идут как есть.
            const int kinds[2] = { req.adaptive.axis_kind[0], req.adaptive.axis_kind[1] };
            if (!adArgs.alloc(req.adaptive, kinds, chunk, res.error)) { cleanup(); return res; }
            res.ad_stats.assign(total_cells * 4, 0.0);
        }

        SIGM_CHECK(cudaMemcpy(d_ranges,  ranges,               4 * sizeof(numb),                      cudaMemcpyHostToDevice), "memcpy d_ranges");
        SIGM_CHECK(cudaMemcpy(d_indices, indicesOfMutVars,     2 * sizeof(int),                       cudaMemcpyHostToDevice), "memcpy d_indices");
        SIGM_CHECK(cudaMemcpy(d_ic,      ic_staged.data(),     (size_t)amountOfIC * sizeof(numb),     cudaMemcpyHostToDevice), "memcpy d_ic");
        SIGM_CHECK(cudaMemcpy(d_values,  values_staged.data(), (size_t)amountOfValues * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_values");
        SIGM_CHECK_CU(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");

        res.dimension   = dimension;
        res.n_pts       = nPts;
        res.param_lo    = req.param_lo;   res.param_hi   = req.param_hi;
        res.param_lo_2  = req.param_lo_2; res.param_hi_2 = req.param_hi_2;
        res.log_scale   = req.log_scale;  res.log_scale_2 = req.log_scale_2;
        res.metric_mask = mask;
        // Точки чанков, до которых не дошёл Stop, остаются пустыми.
        res.flags.assign(total_cells, REGIME_CANCELLED);
        for (int m = 0; m < SIGM_COUNT; ++m)
            if ((mask >> m) & 1) res.values[m].assign(total_cells, std::numeric_limits<double>::quiet_NaN());

        std::vector<numb> h_row(chunk);
        std::vector<int>  h_flags(chunk);

        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            if (iter > 0 && cancel_requested(req.cancel)) break;   // Stop: отдаём посчитанное
            SIGM_CANCEL_CHECK();
            if (req.progress) req.progress->store((float)((double)(chunk * iter) * ticksPerCell / ticksTotal), std::memory_order_relaxed);
            const size_t cur = (iter == amountOfIteration - 1) ? total_cells - chunk * iter : chunk;

            const size_t sharedPerThread = (size_t)ucuda_shared_stride(amountOfIC, amountOfValues) * sizeof(numb);
            const int blockSize = launch_block_size(sharedPerThread);
            const int gridSize  = (int)((cur + blockSize - 1) / blockSize);

            int    nPts_arg        = nPts;
            int    limiter_arg     = (int)cur;
            size_t calculated_arg  = iter * chunk;
            size_t skip_arg        = amountOfPointsForSkip;
            int    dimension_arg   = dimension;
            numb   h_arg           = (numb)req.h;
            int    amountOfIC_arg  = amountOfIC;
            int    amountOfVal_arg = amountOfValues;
            size_t iterations_arg  = (size_t)amountOfPointsInBlock;
            int    preScaller_arg  = preScaller;
            int    writableVar_arg = req.writable_var;
            numb   maxValue_arg    = (numb)req.max_value;
            size_t peakStride_arg  = peakStride;
            int    peakCap_arg     = peakCapacity;
            size_t metricStride_arg= cur;
            int    mask_arg        = mask;
            int    hSweep_arg      = ax.hSweepAxis;
            numb   transient_arg   = (numb)req.transient_time;
            numb   tMax_arg        = (numb)req.t_max;
            int    logMask_arg     = ax.logAxisMask;
            int*   d_cancel_arg    = sig.cancelArg();
            int*   d_progress_arg  = sig.progressArg();
            int    progStride_arg  = progressStride;

            void* args[] = {
                &nPts_arg, &limiter_arg, &calculated_arg, &skip_arg, &dimension_arg,
                &d_ranges, &h_arg, &d_indices, &d_ic, &amountOfIC_arg,
                &d_values, &amountOfVal_arg, &iterations_arg, &preScaller_arg,
                &writableVar_arg, &maxValue_arg,
                &d_intervals, &peakStride_arg, &peakCap_arg,
                &d_out, &metricStride_arg, &mask_arg, &d_flags,
                &hSweep_arg, &transient_arg, &tMax_arg, &logMask_arg,
                &d_cancel_arg, &d_progress_arg, &progStride_arg
            };
            sig.resetTicks();
            const unsigned int shared = (unsigned int)(sharedPerThread * blockSize);
            if (ad) {
                UcudaAdaptParams* d_adp_arg = adArgs.params;
                int*   d_axis_arg        = adArgs.axis;
                numb   tolRatio_arg      = (numb)req.adaptive.tol_ratio;
                numb   dtOut_arg         = (numb)req.h;
                size_t progressUnits_arg = kAdProgressUnits;
                numb*  d_adStats_arg     = adArgs.stats;
                int    raw_arg           = adRaw ? 1 : 0;
                int    interp_arg        = req.adaptive.peak_interp;
                void* args_ad[] = {
                    &nPts_arg, &limiter_arg, &calculated_arg, &dimension_arg,
                    &d_ranges, &d_indices, &d_ic, &amountOfIC_arg,
                    &d_values, &amountOfVal_arg, &writableVar_arg, &maxValue_arg,
                    &d_intervals, &peakStride_arg, &peakCap_arg,
                    &d_out, &metricStride_arg, &mask_arg, &d_flags, &logMask_arg,
                    ad_param_arg(req.adaptive, &d_adp_arg), &d_axis_arg, &tolRatio_arg, &transient_arg, &tMax_arg, &dtOut_arg,
                    &preScaller_arg, &iterations_arg, &raw_arg, &interp_arg,
                    &d_cancel_arg, &d_progress_arg, &progStride_arg,
                    &progressUnits_arg, &d_adStats_arg
                };
                SIGM_CHECK_CU(cuLaunchKernel(cached_metrics_ad.kernel,
                                             gridSize, 1, 1, blockSize, 1, 1,
                                             shared, stream, args_ad, nullptr),
                              "cuLaunchKernel(signal metrics adaptive)");
            }
            else
            SIGM_CHECK_CU(cuLaunchKernel(cached_metrics.kernel,
                                         gridSize, 1, 1, blockSize, 1, 1,
                                         shared, stream, args, nullptr),
                          "cuLaunchKernel(signal metrics)");

            if (!wait_with_signals(stream, sig, req.cancel, req.progress,
                                   (double)(chunk * iter) * ticksPerCell,
                                   ticksTotal, res.error)) { cleanup(); return res; }
            SIGM_CHECK(cudaStreamSynchronize(stream), "sync stream");
            // Stop посреди чанка: прерванные точки помечены REGIME_CANCELLED, метрики у них NaN.

            // Индекс ядра -> индекс результата: в 2D при swap_xy ядро шло по осям
            // в обратном порядке, результат всегда в раскладке GUI [iy*n + ix].
            auto out_index = [&](size_t k) -> size_t {
                const size_t kernel_idx = chunk * iter + k;
                if (!ax.swap_xy) return kernel_idx;
                const size_t kix = kernel_idx % (size_t)nPts;
                const size_t kiy = kernel_idx / (size_t)nPts;
                return kix * (size_t)nPts + kiy;
            };

            SIGM_CHECK(cudaMemcpy(h_flags.data(), d_flags, cur * sizeof(int), cudaMemcpyDeviceToHost), "memcpy flags");
            for (size_t k = 0; k < cur; ++k) res.flags[out_index(k)] = h_flags[k];
            if (ad) {
                std::vector<numb> h_st(cur * 4);
                SIGM_CHECK(cudaMemcpy(h_st.data(), adArgs.stats, cur * 4 * sizeof(numb), cudaMemcpyDeviceToHost),
                           "memcpy adaptive stats");
                for (size_t k = 0; k < cur; ++k)
                    for (int q = 0; q < 4; ++q) res.ad_stats[out_index(k) * 4 + q] = (double)h_st[k * 4 + q];
            }

            for (int m = 0; m < SIGM_COUNT; ++m) {
                if (!((mask >> m) & 1)) continue;
                SIGM_CHECK(cudaMemcpy(h_row.data(), d_out + (size_t)m * cur, cur * sizeof(numb),
                                      cudaMemcpyDeviceToHost), "memcpy metrics");
                std::vector<double>& dst = res.values[m];
                for (size_t k = 0; k < cur; ++k) dst[out_index(k)] = (double)h_row[k];
            }
        }

        cleanup();
        #undef SIGM_CHECK
        #undef SIGM_CHECK_CU
        #undef SIGM_CANCEL_CHECK

        metrics_finish(res, req);
        return res;
    }

SignalMetricsResult ParametricEngine::Impl::run_metrics_continuation_gpu(const SignalMetricsRequest& req) {
        SignalMetricsResult res;
        auto fail = [&](const std::string& msg) -> SignalMetricsResult& { res.error = msg; return res; };

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_metrics_if_needed(req.krs_body, req.amountOfX, 1, err, true, req.minmax_interp)) return fail(err);

        const double worstCaseH = req.sweep_over_h
                                ? ((req.param_lo < req.param_hi) ? req.param_lo : req.param_hi)
                                : req.h;
        if (worstCaseH <= 0.0) return fail("h must be > 0 (for an h-sweep, over the whole range)");
        const int sizeOfBlock = (int)std::ceil(req.t_max / worstCaseH / req.pre_scaller);
        if (sizeOfBlock <= 0) return fail("amountOfPointsInBlock <= 0");

        const int    nPts  = req.n_pts;
        const int    mask  = req.metric_mask & kSignalMetricAllMask;
        const bool   needIntervals = ((mask >> SIGM_MEDIAN_FREQ) & 1) != 0;
        const size_t peakStride    = (size_t)sizeOfBlock < (size_t)max_amount_of_peaks + 1
                                   ? (size_t)sizeOfBlock : (size_t)max_amount_of_peaks + 1;

        numb* d_baseValues = nullptr;
        numb* d_baseX      = nullptr;
        numb* d_intervals  = nullptr;
        numb* d_out        = nullptr;
        int*  d_flags      = nullptr;
        RunSignals sig;   // однопоточное ядро: тик на точку
        auto cleanup = [&]() {
            if (d_baseValues) cudaFree(d_baseValues);
            if (d_baseX)      cudaFree(d_baseX);
            if (d_intervals)  cudaFree(d_intervals);
            if (d_out)        cudaFree(d_out);
            if (d_flags)      cudaFree(d_flags);
            sig.release();
        };
        #define SMC_CHECK(call, where) do { cudaError_t _e = (call); \
            if (_e != cudaSuccess) { res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); cleanup(); return res; } } while(0)
        #define SMC_CHECK_CU(call, where) do { CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { res.error = std::string(where) + ": " + cu_err(_r); cleanup(); return res; } } while(0)

        SMC_CHECK(cudaMalloc((void**)&d_baseValues, req.base_values.size() * sizeof(numb)),        "cudaMalloc d_baseValues");
        SMC_CHECK(cudaMalloc((void**)&d_baseX,      (size_t)req.amountOfX * sizeof(numb)),         "cudaMalloc d_baseX");
        SMC_CHECK(cudaMalloc((void**)&d_out,        (size_t)SIGM_COUNT * (size_t)nPts * sizeof(numb)), "cudaMalloc d_out");
        SMC_CHECK(cudaMalloc((void**)&d_flags,      (size_t)nPts * sizeof(int)),                   "cudaMalloc d_flags");
        if (needIntervals)
            SMC_CHECK(cudaMalloc((void**)&d_intervals, peakStride * sizeof(numb)),                 "cudaMalloc d_intervals");
        if (!sig.alloc(res.error)) { cleanup(); return res; }

        const std::vector<numb> baseValues_staged = to_numb(req.base_values);
        const std::vector<numb> baseX_staged      = to_numb(req.initial_conditions);
        SMC_CHECK(cudaMemcpy(d_baseValues, baseValues_staged.data(), req.base_values.size() * sizeof(numb),
                             cudaMemcpyHostToDevice), "memcpy d_baseValues");
        SMC_CHECK(cudaMemcpy(d_baseX, baseX_staged.data(), (size_t)req.amountOfX * sizeof(numb),
                             cudaMemcpyHostToDevice), "memcpy d_baseX");
        SMC_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        int  nPts_arg        = nPts;
        numb lo_arg          = (numb)req.param_lo;
        numb hi_arg          = (numb)req.param_hi;
        int  reverse_arg     = req.continuation_reverse ? 1 : 0;
        int  logScale_arg    = req.log_scale ? 1 : 0;
        int  sweepIsH_arg    = req.sweep_over_h ? 1 : 0;
        int  mutParamIdx_arg = req.param_index;
        int  amountOfVal_arg = (int)req.base_values.size();
        int  amountOfX_arg   = req.amountOfX;
        numb h_arg           = (numb)req.h;
        numb tMax_arg        = (numb)req.t_max;
        numb transient_arg   = (numb)req.transient_time;
        int  sizeOfBlock_arg = sizeOfBlock;
        int  preScaller_arg  = req.pre_scaller;
        int  writableVar_arg = req.writable_var;
        numb maxValue_arg    = (numb)req.max_value;
        int  peakCap_arg     = (int)peakStride;
        int  mask_arg        = mask;
        int* d_cancel_arg    = sig.cancelArg();
        int* d_progress_arg  = sig.progressArg();
        void* args[] = {
            &nPts_arg, &lo_arg, &hi_arg, &reverse_arg, &logScale_arg, &sweepIsH_arg, &mutParamIdx_arg,
            &d_baseValues, &amountOfVal_arg, &d_baseX, &amountOfX_arg,
            &h_arg, &tMax_arg, &transient_arg, &sizeOfBlock_arg, &preScaller_arg,
            &writableVar_arg, &maxValue_arg,
            &d_intervals, &peakCap_arg,
            &d_out, &mask_arg, &d_flags,
            &d_cancel_arg, &d_progress_arg
        };
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true; res.error = "Cancelled by user"; cleanup(); return res;
        }
        SMC_CHECK_CU(cuLaunchKernel(cached_metrics.kernel_cont, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
                     "cuLaunchKernel(metrics cont)");
        if (!wait_with_signals(0, sig, req.cancel, req.progress, 0.0, (double)nPts, res.error))
            { cleanup(); return res; }
        SMC_CHECK(cudaDeviceSynchronize(), "sync after metrics cont");
        // Stop посреди цепочки: ядро пометило хвост REGIME_CANCELLED / NaN, начало цело.

        res.dimension   = 1;
        res.n_pts       = nPts;
        res.param_lo    = req.param_lo;
        res.param_hi    = req.param_hi;
        res.log_scale   = req.log_scale;
        res.metric_mask = mask;
        res.continuation         = true;
        res.continuation_reverse = req.continuation_reverse;

        std::vector<int>  h_flags((size_t)nPts);
        std::vector<numb> h_row((size_t)nPts);
        SMC_CHECK(cudaMemcpy(h_flags.data(), d_flags, (size_t)nPts * sizeof(int), cudaMemcpyDeviceToHost), "memcpy flags");
        res.flags.assign(h_flags.begin(), h_flags.end());
        for (int m = 0; m < SIGM_COUNT; ++m) {
            if (!((mask >> m) & 1)) continue;
            SMC_CHECK(cudaMemcpy(h_row.data(), d_out + (size_t)m * (size_t)nPts, (size_t)nPts * sizeof(numb),
                                 cudaMemcpyDeviceToHost), "memcpy metrics");
            res.values[m].assign(h_row.begin(), h_row.end());
        }

        cleanup();
        #undef SMC_CHECK
        #undef SMC_CHECK_CU
        metrics_finish(res, req);
        return res;
    }

SignalMetricsResult ParametricEngine::Impl::run_metrics_cpu_ad(const SignalMetricsRequest& req, const MetricsAxes& ax) {
        SignalMetricsResult res;
        auto fail = [&](const std::string& msg) -> SignalMetricsResult& { res.error = msg; return res; };
        const AdaptiveRequest& ad = req.adaptive;
        const bool cont = req.continuation;
        if (req.dimension != 1) return fail("CPU computation works in 1D only");
        if (cont && !(ad.axis_kind[0] != kAdAxisSystem) && (req.param_index < 0 || req.param_index >= (int)req.base_values.size()))
            return fail("param_index out of range");

        AdaptiveCpuModule mod;
        {
            std::vector<KrsCpuDiag> diags;
            if (!mod.compile(ad.rhs, ad.emb, ad.dprep, ad.deval, ad.ctrl_body, req.amountOfX,
                             ad.prefix + peak_config_defines(), diags) || !mod.metrics()) {
                std::string e = "CPU adaptive module:";
                for (const KrsCpuDiag& d : diags)
                    e += "\n" + (d.line > 0 ? "line " + std::to_string(d.line) + ": " : std::string()) + d.message;
                if (diags.empty()) e += " no dense output for this scheme";
                return fail(e);
            }
        }

        const int    nPts  = req.n_pts;
        const int    mask  = req.metric_mask & kSignalMetricAllMask;
        const bool   raw   = ad.raw_nodes;   // узлы шага: h не участвует, интервалы — под потолок пиков
        const size_t iters = raw ? 0 : (size_t)std::ceil(req.t_max / req.h / (double)req.pre_scaller);
        if (!raw && (iters == 0 || iters > (size_t)(std::numeric_limits<int>::max)()))
            return fail("computed amountOfPointsInBlock <= 0 (t_max/h/pre_scaller too small)");
        const bool   needIntervals = ((mask >> SIGM_MEDIAN_FREQ) & 1) != 0;
        const size_t peakStride    = (raw || iters >= (size_t)max_amount_of_peaks + 1)
                                   ? (size_t)max_amount_of_peaks + 1 : iters;
        const int    peakCapacity  = (int)peakStride;
        const std::vector<double> values = req.base_values, x0 = req.initial_conditions;
        const double lo = cont ? req.param_lo : ax.lo_x, hi = cont ? req.param_hi : ax.hi_x;
        const int    mutIdx = cont ? (ad.axis_kind[0] != kAdAxisSystem ? 0 : req.param_index) : ax.idx_x;

        res.dimension   = 1;
        res.n_pts       = nPts;
        res.param_lo    = req.param_lo;
        res.param_hi    = req.param_hi;
        res.log_scale   = req.log_scale;
        res.metric_mask = mask;
        res.continuation         = cont;
        res.continuation_reverse = cont && req.continuation_reverse;
        // Точки, до которых не дошёл Stop, остаются пустыми.
        res.flags.assign((size_t)nPts, REGIME_CANCELLED);
        for (int m = 0; m < SIGM_COUNT; ++m)
            if ((mask >> m) & 1) res.values[m].assign((size_t)nPts, std::numeric_limits<double>::quiet_NaN());
        res.ad_stats.assign((size_t)nPts * 4, 0.0);

        // Stop до запуска — прежний отказ, а не пустой частичный результат.
        if (cancel_requested(req.cancel)) { res.cancelled = true; return fail("Cancelled by user"); }
        volatile int cancelFlag = 0;
        std::atomic<int> donePts{ 0 };
        std::atomic<bool> finished{ false };
        int chainProg = 0;
        std::thread watcher([&]() {
            while (!finished.load(std::memory_order_relaxed)) {
                if (req.cancel && req.cancel->load(std::memory_order_relaxed)) cancelFlag = 1;
                const int got = cont ? *(volatile int*)&chainProg : donePts.load(std::memory_order_relaxed);
                if (req.progress) req.progress->store((float)got / (float)nPts, std::memory_order_relaxed);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        });
        auto call = [&](int i0, int i1, double* iv, double* om, int* fl, double* st, int* prog) {
            return mod.metrics()(cont ? 1 : 0, i0, i1, nPts, lo, hi, req.continuation_reverse ? 1 : 0,
                                 req.log_scale ? 1 : 0, ax.par_or_var, mutIdx, values.data(), (int)values.size(),
                                 x0.data(), &ad.params, ad.axis_kind[0], ad.tol_ratio, req.writable_var,
                                 req.max_value, iv, (unsigned long long)peakStride, peakCapacity, om, mask, fl,
                                 req.transient_time, req.t_max, req.h, req.pre_scaller, (unsigned long long)iters,
                                 raw ? 1 : 0, ad.peak_interp, st, &cancelFlag, prog);
        };
        bool aborted = false;
        if (cont) {
            std::vector<double> iv(needIntervals ? peakStride : 0), om((size_t)SIGM_COUNT * nPts), st((size_t)nPts * 4);
            // Stop рвёт цепочку: хвост, до которого она не дошла, остаётся пустым.
            std::vector<int> fl((size_t)nPts, REGIME_CANCELLED);
            std::fill(om.begin(), om.end(), std::numeric_limits<double>::quiet_NaN());
            aborted = !call(0, nPts, needIntervals ? iv.data() : nullptr, om.data(), fl.data(), st.data(), &chainProg);
            for (int j = 0; j < nPts; ++j) res.flags[(size_t)j] = fl[(size_t)j];
            for (int m = 0; m < SIGM_COUNT; ++m)
                if ((mask >> m) & 1)
                    for (int j = 0; j < nPts; ++j) res.values[m][(size_t)j] = om[(size_t)m * nPts + j];
            res.ad_stats = st;
        } else {
            // Куски точек — потокам по мере освобождения; строки кусков не пересекаются.
            constexpr int kChunk = 4;
            unsigned hw = std::thread::hardware_concurrency();
            if (hw == 0) hw = 1;
            const int nChunks  = (nPts + kChunk - 1) / kChunk;
            const int nWorkers = (int)std::min<unsigned>(hw, (unsigned)nChunks);
            std::atomic<int> nextChunk{ 0 };
            std::atomic<bool> stop{ false };
            auto worker = [&]() {
                std::vector<double> iv(needIntervals ? (size_t)kChunk * peakStride : 0);
                std::vector<double> om((size_t)SIGM_COUNT * kChunk), st((size_t)kChunk * 4);
                std::vector<int> fl((size_t)kChunk);
                for (;;) {
                    const int c = nextChunk.fetch_add(1, std::memory_order_relaxed);
                    if (c >= nChunks || stop.load(std::memory_order_relaxed)) return;
                    const int i0 = c * kChunk, i1 = (std::min)(nPts, i0 + kChunk), rows = i1 - i0;
                    int prog = 0;
                    // Stop рвёт кусок: строки, до которых он не дошёл, остаются пустыми.
                    std::fill(fl.begin(), fl.end(), REGIME_CANCELLED);
                    std::fill(om.begin(), om.end(), std::numeric_limits<double>::quiet_NaN());
                    const bool whole = call(i0, i1, needIntervals ? iv.data() : nullptr, om.data(), fl.data(), st.data(), &prog) != 0;
                    if (!whole) stop.store(true);
                    for (int r = 0; r < rows; ++r) {
                        res.flags[(size_t)(i0 + r)] = fl[(size_t)r];
                        for (int q = 0; q < 4; ++q) res.ad_stats[(size_t)(i0 + r) * 4 + q] = st[(size_t)r * 4 + q];
                    }
                    for (int m = 0; m < SIGM_COUNT; ++m)
                        if ((mask >> m) & 1)
                            for (int r = 0; r < rows; ++r) res.values[m][(size_t)(i0 + r)] = om[(size_t)m * rows + r];
                    if (!whole) return;
                    donePts.fetch_add(rows, std::memory_order_relaxed);
                }
            };
            std::vector<std::thread> pool;
            for (int w = 0; w < nWorkers; ++w) pool.emplace_back(worker);
            for (std::thread& t : pool) t.join();
            aborted = stop.load();
        }
        finished.store(true, std::memory_order_relaxed);
        watcher.join();
        (void)aborted;   // Stop: частичный результат, cancelled ставит metrics_finish
        if (req.progress) req.progress->store(1.0f, std::memory_order_relaxed);
        metrics_finish(res, req);
        return res;
    }

SignalMetricsResult ParametricEngine::Impl::run_metrics_continuation_ad(const SignalMetricsRequest& req) {
        SignalMetricsResult res;
        auto fail = [&](const std::string& msg) -> SignalMetricsResult& { res.error = msg; return res; };

        const AdaptiveRequest& ad = req.adaptive;
        const bool ad_axis = ad.axis_kind[0] != kAdAxisSystem;
        if (!ad.setup_error.empty()) return fail(ad.setup_error);
        if (req.sweep_over_h)
            return fail("the step h is not swept with the adaptive step: sweep rtol / atol instead");
        if (!ad_axis && (req.param_index < 0 || req.param_index >= (int)req.base_values.size()))
            return fail("param_index out of range");
        const bool raw = ad.raw_nodes;   // узлы шага: h не участвует, интервалы — под потолок пиков
        if (!raw && req.h <= 0.0) return fail("h must be > 0");
        const size_t iters = raw ? 0 : (size_t)std::ceil(req.t_max / req.h / (double)req.pre_scaller);
        if (!raw && iters == 0) return fail("amountOfPointsInBlock <= 0");

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_metrics_ad_if_needed(ad, req.krs_body, req.amountOfX, 1, err)) return fail(err);

        const int    nPts  = req.n_pts;
        const int    mask  = req.metric_mask & kSignalMetricAllMask;
        const bool   needIntervals = ((mask >> SIGM_MEDIAN_FREQ) & 1) != 0;
        const size_t peakStride    = (raw || iters >= (size_t)max_amount_of_peaks + 1)
                                   ? (size_t)max_amount_of_peaks + 1 : iters;

        numb* d_baseValues = nullptr;
        numb* d_baseX      = nullptr;
        numb* d_intervals  = nullptr;
        numb* d_out        = nullptr;
        int*  d_flags      = nullptr;
        AdDeviceArgs adArgs;
        RunSignals sig;   // однопоточное ядро: тик на точку
        auto cleanup = [&]() {
            adArgs.release();
            if (d_baseValues) cudaFree(d_baseValues);
            if (d_baseX)      cudaFree(d_baseX);
            if (d_intervals)  cudaFree(d_intervals);
            if (d_out)        cudaFree(d_out);
            if (d_flags)      cudaFree(d_flags);
            sig.release();
        };
        #define SMA_CHECK(call, where) do { cudaError_t _e = (call); \
            if (_e != cudaSuccess) { res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); cleanup(); return res; } } while(0)
        #define SMA_CHECK_CU(call, where) do { CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { res.error = std::string(where) + ": " + cu_err(_r); cleanup(); return res; } } while(0)

        SMA_CHECK(cudaMalloc((void**)&d_baseValues, (req.base_values.empty() ? 1 : req.base_values.size()) * sizeof(numb)), "cudaMalloc d_baseValues");
        SMA_CHECK(cudaMalloc((void**)&d_baseX,      (size_t)req.amountOfX * sizeof(numb)),             "cudaMalloc d_baseX");
        SMA_CHECK(cudaMalloc((void**)&d_out,        (size_t)SIGM_COUNT * (size_t)nPts * sizeof(numb)), "cudaMalloc d_out");
        SMA_CHECK(cudaMalloc((void**)&d_flags,      (size_t)nPts * sizeof(int)),                       "cudaMalloc d_flags");
        if (needIntervals)
            SMA_CHECK(cudaMalloc((void**)&d_intervals, peakStride * sizeof(numb)),                     "cudaMalloc d_intervals");
        {
            const int kinds[2] = { ad.axis_kind[0], kAdAxisSystem };
            if (!adArgs.alloc(ad, kinds, (size_t)nPts, res.error)) { cleanup(); return res; }
        }
        if (!sig.alloc(res.error)) { cleanup(); return res; }

        const std::vector<numb> baseValues_staged = to_numb(req.base_values);
        const std::vector<numb> baseX_staged      = to_numb(req.initial_conditions);
        if (!baseValues_staged.empty())
            SMA_CHECK(cudaMemcpy(d_baseValues, baseValues_staged.data(), baseValues_staged.size() * sizeof(numb),
                                 cudaMemcpyHostToDevice), "memcpy d_baseValues");
        SMA_CHECK(cudaMemcpy(d_baseX, baseX_staged.data(), (size_t)req.amountOfX * sizeof(numb),
                             cudaMemcpyHostToDevice), "memcpy d_baseX");
        SMA_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        int    nPts_arg        = nPts;
        numb   lo_arg          = (numb)req.param_lo;
        numb   hi_arg          = (numb)req.param_hi;
        int    reverse_arg     = req.continuation_reverse ? 1 : 0;
        int    logScale_arg    = req.log_scale ? 1 : 0;
        int    mutParamIdx_arg = ad_axis ? 0 : req.param_index;
        int    amountOfVal_arg = (int)req.base_values.size();
        int    writableVar_arg = req.writable_var;
        numb   maxValue_arg    = (numb)req.max_value;
        int    peakCap_arg     = (int)peakStride;
        int    mask_arg        = mask;
        UcudaAdaptParams* d_adp_arg = adArgs.params;
        int*   d_axis_arg      = adArgs.axis;
        numb   tolRatio_arg    = (numb)ad.tol_ratio;
        numb   transient_arg   = (numb)req.transient_time;
        numb   tRec_arg        = (numb)req.t_max;
        numb   dtOut_arg       = (numb)req.h;
        int    preScaller_arg  = req.pre_scaller;
        size_t iters_arg       = iters;
        int    raw_arg         = raw ? 1 : 0;
        int    interp_arg      = ad.peak_interp;
        int*   d_cancel_arg    = sig.cancelArg();
        int*   d_progress_arg  = sig.progressArg();
        numb*  d_adStats_arg   = adArgs.stats;
        void* args[] = {
            &nPts_arg, &lo_arg, &hi_arg, &reverse_arg, &logScale_arg, &mutParamIdx_arg,
            &d_baseValues, &amountOfVal_arg, &d_baseX, &writableVar_arg, &maxValue_arg,
            &d_intervals, &peakCap_arg, &d_out, &mask_arg, &d_flags,
            &d_adp_arg, &d_axis_arg, &tolRatio_arg, &transient_arg, &tRec_arg, &dtOut_arg,
            &preScaller_arg, &iters_arg, &raw_arg, &interp_arg, &d_cancel_arg, &d_progress_arg, &d_adStats_arg
        };
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true; res.error = "Cancelled by user"; cleanup(); return res;
        }
        SMA_CHECK_CU(cuLaunchKernel(cached_metrics_ad.kernel_cont, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
                     "cuLaunchKernel(metrics adaptive continuation)");
        if (!wait_with_signals(0, sig, req.cancel, req.progress, 0.0, (double)nPts, res.error))
            { cleanup(); return res; }
        SMA_CHECK(cudaDeviceSynchronize(), "sync after metrics adaptive continuation");
        // Stop посреди цепочки: ядро пометило хвост REGIME_CANCELLED / NaN, начало цело.

        res.dimension   = 1;
        res.n_pts       = nPts;
        res.param_lo    = req.param_lo;
        res.param_hi    = req.param_hi;
        res.log_scale   = req.log_scale;
        res.metric_mask = mask;
        res.continuation         = true;
        res.continuation_reverse = req.continuation_reverse;

        std::vector<int>  h_flags((size_t)nPts);
        std::vector<numb> h_row((size_t)nPts);
        std::vector<numb> h_st((size_t)nPts * 4);
        SMA_CHECK(cudaMemcpy(h_flags.data(), d_flags, (size_t)nPts * sizeof(int), cudaMemcpyDeviceToHost), "memcpy flags");
        res.flags.assign(h_flags.begin(), h_flags.end());
        for (int m = 0; m < SIGM_COUNT; ++m) {
            if (!((mask >> m) & 1)) continue;
            SMA_CHECK(cudaMemcpy(h_row.data(), d_out + (size_t)m * (size_t)nPts, (size_t)nPts * sizeof(numb),
                                 cudaMemcpyDeviceToHost), "memcpy metrics");
            res.values[m].assign(h_row.begin(), h_row.end());
        }
        SMA_CHECK(cudaMemcpy(h_st.data(), adArgs.stats, (size_t)nPts * 4 * sizeof(numb), cudaMemcpyDeviceToHost),
                  "memcpy adaptive stats");
        res.ad_stats.assign(h_st.begin(), h_st.end());

        cleanup();
        #undef SMA_CHECK
        #undef SMA_CHECK_CU
        metrics_finish(res, req);
        return res;
    }

bool ParametricEngine::Impl::compile_basins_if_needed(const std::string& krs_body, int amountOfX,
                                  std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":basins";
        return compile_into(pool_basins, key, cached_basins, activate, [&](CachedBasinsModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(time_prefix(krs_body) + lyap_nx_prefix(krs_body) + src_template_basins), "basins.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body } },
                              { "calculateDiscreteModelAvgPeaksCUDA" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_fused, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::basins_dbscan_host(const numb* X, const numb* Y, const int* H, size_t n, double eps,
                                   const std::shared_ptr<std::atomic<bool>>& cancel,
                                   const std::shared_ptr<std::atomic<float>>& progress,
                                   int* labels, int& n_pos, int& n_neg) {
        n_pos = 0; n_neg = 0;
        numb eps2 = (numb)eps;
        eps2 = eps2 * eps2;   // в numb, как eps * eps в CUDA_dbscan_kernel
        const bool fmad = get_nvrtc_fmad();
        auto near_pt = [&](size_t a, size_t b) {
            const numb dx = X[b] - X[a];
            const numb dy = Y[b] - Y[a];
            const numb d2 = fmad ? (numb)std::fma(dx, dx, dy * dy) : dx * dx + dy * dy;
            return d2 <= eps2;
        };
        auto cancelled = [&]() { return cancel && cancel->load(std::memory_order_relaxed); };

        // Ячейка: сторона s чуть меньше eps/√2 (запас 1e-4 перекрывает округления и numb = float).
        // Ключ считается в double; при |x/s| > 2^32 погрешность ключа съела бы запас, такие точки
        // (и вообще нечисловые признаки) идут мимо сетки.
        const double s   = eps / std::sqrt(2.0) * (1.0 - 1e-4);
        const double lim = 4294967296.0;
        struct Key { long long cx, cy; uint32_t i; };
        std::vector<Key> keys;     keys.reserve(n);
        std::vector<uint32_t> offgrid;   // конечные, но вне сетки: перебор против всех
        std::vector<int> node(n, -1);   // узел union-find точки; -1 — Unbound (не участвует)
        for (size_t i = 0; i < n; ++i) {
            if (H[i] == REGIME_UNBOUND) continue;
            if (!std::isfinite((double)X[i]) || !std::isfinite((double)Y[i])) continue;   // узел получит ниже; соседей у NaN/inf нет
            const double qx = (double)X[i] / s, qy = (double)Y[i] / s;
            if (!(std::fabs(qx) <= lim && std::fabs(qy) <= lim)) { offgrid.push_back((uint32_t)i); continue; }
            keys.push_back({ (long long)std::floor(qx), (long long)std::floor(qy), (uint32_t)i });
        }
        std::sort(keys.begin(), keys.end(), [](const Key& a, const Key& b) {
            return a.cx != b.cx ? a.cx < b.cx : (a.cy != b.cy ? a.cy < b.cy : a.i < b.i);
        });
        struct Cell { long long cx, cy; size_t b, e; double x0, x1, y0, y1; };
        std::vector<Cell> cells;
        for (size_t k = 0; k < keys.size();) {
            size_t e = k;
            Cell c{ keys[k].cx, keys[k].cy, k, k, INFINITY, -INFINITY, INFINITY, -INFINITY };
            for (; e < keys.size() && keys[e].cx == c.cx && keys[e].cy == c.cy; ++e) {
                const size_t i = keys[e].i;
                c.x0 = (std::min)(c.x0, (double)X[i]); c.x1 = (std::max)(c.x1, (double)X[i]);
                c.y0 = (std::min)(c.y0, (double)Y[i]); c.y1 = (std::max)(c.y1, (double)Y[i]);
                node[i] = (int)cells.size();
            }
            c.e = e;
            cells.push_back(c);
            k = e;
        }
        int nodes = (int)cells.size();
        for (uint32_t i : offgrid) node[i] = nodes++;
        for (size_t i = 0; i < n; ++i)
            if (H[i] != REGIME_UNBOUND && node[i] < 0) node[i] = nodes++;   // NaN/inf: свой узел навсегда

        std::vector<int> parent((size_t)nodes);
        for (int v = 0; v < nodes; ++v) parent[(size_t)v] = v;
        auto find = [&](int v) {
            while (parent[(size_t)v] != v) { parent[(size_t)v] = parent[(size_t)parent[(size_t)v]]; v = parent[(size_t)v]; }
            return v;
        };
        auto unite = [&](int a, int b) {
            a = find(a); b = find(b);
            if (a != b) { if (a < b) std::swap(a, b); parent[(size_t)a] = b; }
        };

        // Пары соседних ячеек. Окно 5x5 без центра, каждая неупорядоченная пара — один раз (сосед
        // с большим ключом). Пара, уже связанная через другие ячейки, не проверяется: плотные
        // кластеры сливаются за первые же проверки. Перед перебором точек — отсев по
        // прямоугольникам ячеек с тем же запасом.
        const double gapMax2 = eps * eps * (1.0 + 1e-4);
        auto box_gap2 = [](double ax0, double ax1, double ay0, double ay1,
                           double bx0, double bx1, double by0, double by1) {
            const double gx = (std::max)(0.0, (std::max)(ax0 - bx1, bx0 - ax1));
            const double gy = (std::max)(0.0, (std::max)(ay0 - by1, by0 - ay1));
            return gx * gx + gy * gy;
        };
        auto find_cell = [&](long long cx, long long cy) -> int {
            auto it = std::lower_bound(cells.begin(), cells.end(), std::make_pair(cx, cy),
                [](const Cell& c, const std::pair<long long, long long>& k) {
                    return c.cx != k.first ? c.cx < k.first : c.cy < k.second;
                });
            return (it != cells.end() && it->cx == cx && it->cy == cy) ? (int)(it - cells.begin()) : -1;
        };
        for (size_t a = 0; a < cells.size(); ++a) {
            if ((a & 1023) == 0) {
                if (cancelled()) return false;
                if (progress) progress->store(0.9f * float(a) / float(cells.size()), std::memory_order_relaxed);
            }
            const Cell& A = cells[a];
            for (int ox = 0; ox <= 2; ++ox)
                for (int oy = -2; oy <= 2; ++oy) {
                    if (ox == 0 && oy <= 0) continue;
                    const int b = find_cell(A.cx + ox, A.cy + oy);
                    if (b < 0 || find((int)a) == find(b)) continue;
                    const Cell& B = cells[(size_t)b];
                    if (box_gap2(A.x0, A.x1, A.y0, A.y1, B.x0, B.x1, B.y0, B.y1) > gapMax2) continue;
                    bool hit = false;
                    for (size_t p = A.b; p < A.e && !hit; ++p) {
                        const size_t i = keys[p].i;
                        const double xi = (double)X[i], yi = (double)Y[i];
                        if (box_gap2(xi, xi, yi, yi, B.x0, B.x1, B.y0, B.y1) > gapMax2) continue;
                        for (size_t q = B.b; q < B.e; ++q)
                            if (near_pt(i, keys[q].i)) { hit = true; break; }
                    }
                    if (hit) unite((int)a, b);
                }
        }
        // Точки вне сетки: перебор против всех конечных точек (на практике их нет).
        for (uint32_t u : offgrid) {
            if (cancelled()) return false;
            for (size_t j = 0; j < n; ++j)
                if (j != u && H[j] != REGIME_UNBOUND && near_pt(u, j)) unite(node[u], node[j]);
        }

        // Метки: компоненты с FP — по возрастанию наименьшего индекса FP, затем остальные — по
        // наименьшему индексу Osc. Первая встреча корня при обходе по возрастанию индекса и есть
        // этот наименьший индекс.
        std::vector<int> labelOf((size_t)nodes, 0);
        for (size_t i = 0; i < n; ++i)
            if (H[i] == REGIME_FIXED_POINT) { int& l = labelOf[(size_t)find(node[i])]; if (l == 0) l = --n_neg; }
        for (size_t i = 0; i < n; ++i)
            if (H[i] == REGIME_OSCILLATION) { int& l = labelOf[(size_t)find(node[i])]; if (l == 0) l = ++n_pos; }
        for (size_t i = 0; i < n; ++i)
            labels[i] = node[i] >= 0 ? labelOf[(size_t)find(node[i])] : 0;
        if (progress) progress->store(1.0f, std::memory_order_relaxed);
        return true;
    }

bool ParametricEngine::Impl::basins_dbscan_grid(const numb* d_X, const numb* d_Y, const int* d_helpful, int* d_labels,
                            size_t total_cells, double eps,
                            const std::shared_ptr<std::atomic<bool>>& cancel,
                            const std::shared_ptr<std::atomic<float>>& progress,
                            int& n_pos, int& n_neg, std::string& err) {
        err.clear();
        const size_t n = total_cells;
        std::vector<numb> X(n), Y(n);
        std::vector<int>  H(n), labels(n);
        cudaError_t ce = cudaMemcpy(X.data(), d_X, n * sizeof(numb), cudaMemcpyDeviceToHost);
        if (ce == cudaSuccess) ce = cudaMemcpy(Y.data(), d_Y, n * sizeof(numb), cudaMemcpyDeviceToHost);
        if (ce == cudaSuccess) ce = cudaMemcpy(H.data(), d_helpful, n * sizeof(int), cudaMemcpyDeviceToHost);
        if (ce != cudaSuccess) { err = std::string("CUDA memcpy dbscan D2H: ") + cudaGetErrorString(ce); return false; }
        if (!basins_dbscan_host(X.data(), Y.data(), H.data(), n, eps, cancel, progress, labels.data(), n_pos, n_neg))
            return false;
        ce = cudaMemcpy(d_labels, labels.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        if (ce != cudaSuccess) { err = std::string("CUDA memcpy dbscan labels H2D: ") + cudaGetErrorString(ce); return false; }
        return true;
    }

bool ParametricEngine::Impl::compile_fs_module(const std::string& src_template,
                           const std::string& tag, // ":fs_attr" или ":fs_grid"
                           int amountOfX,
                           const std::string& krs_body,
                           int type_of_synch_v, int error_estim_v, double fs_error_trs_v,
                           const std::vector<const char*>& expr_kernels,
                           ModuleLru<CachedFastSyncModule>& pool,
                           CachedFastSyncModule& slot,
                           std::string& err,
                           bool activate )
    {
        cuCtxSetCurrent(context);
        char trs_buf[64];
        std::snprintf(trs_buf, sizeof(trs_buf), "%.17g", fs_error_trs_v);
        std::string key = hash_key(krs_body, amountOfX) + tag
                        + ":t" + std::to_string(type_of_synch_v)
                        + ":e" + std::to_string(error_estim_v)
                        + ":r" + trs_buf;
        return compile_into(pool, key, slot, activate, [&](CachedFastSyncModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> lowered;
            if (!build_module(snapshot_sources(src_template), "fastsync.cu",
                              { { "{{AMOUNT_OF_X}}",   std::to_string(amountOfX) },
                                { "{{TYPE_OF_SYNCH}}", std::to_string(type_of_synch_v) },
                                { "{{ERROR_ESTIM}}",   std::to_string(error_estim_v) },
                                { "{{FS_ERROR_TRS}}",  std::string(trs_buf) },
                                { "{{KRS_BODY}}",      krs_body } },
                              expr_kernels,
                              mod, lowered, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;

            // Связываем по expr_kernels индексу.
            for (size_t i = 0; i < expr_kernels.size(); ++i) {
                CUfunction f = nullptr;
                if (!module_fn(mod, lowered[i], f, err)) { cuModuleUnload(mod); return false; }
                const std::string sym_name = expr_kernels[i];
                if      (sym_name == "fillFSMasterTrajectory")                   fresh.kernel_fs_fill = f;
                else if (sym_name == "calculateDiscreteModelforFastSynchroCUDA") fresh.kernel_fs_traj = f;
                else if (sym_name == "calculateDiscreteModelICCforFastSynchro")  fresh.kernel_fs_grid = f;
            }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_order_module(bool activate, int amountOfX, int amountOfValues,
                              const std::string& krs_body, const std::string& ref_body,
                              std::string& err)
    {
        cuCtxSetCurrent(context);
        // Тело эталона входит в ключ: с ним и без него это РАЗНЫЙ модуль (ветка
        // эталона вырезается препроцессором), да и сам эталонный метод меняется.
        const std::string key = hash_key(krs_body, amountOfX) + ":order:v"
                              + std::to_string(amountOfValues)
                              + ":ref" + std::to_string(std::hash<std::string>{}(ref_body));
        return compile_into(pool_order, key, cached_order, activate, [&](CachedOrderModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> lowered;
            // orderEstimateKernel объявлено extern "C" — в name_exprs не нужно.
            if (!build_module(snapshot_sources(time_prefix(krs_body + ref_body) + src_template_order), "order.cu",
                              { { "{{AMOUNT_OF_X}}",      std::to_string(amountOfX) },
                                { "{{AMOUNT_OF_VALUES}}", std::to_string(amountOfValues) },
                                { "{{KRS_BODY}}",         krs_body },
                                { "{{REF_ENABLED}}",      ref_body.empty() ? "0" : "1" },
                                { "{{KRS_REF_BODY}}",     ref_body } },
                              {}, mod, lowered, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, "orderEstimateKernel", fresh.kernel, err)) { cuModuleUnload(mod); return false; }
            // Ядро замера лежит в том же модуле: Performance нужны ОБА прохода,
            // и второй компиляции ради этого быть не должно.
            if (!module_fn(mod, "perfIntegrateKernel", fresh.kernel_perf, err)) { cuModuleUnload(mod); return false; }
            // И ядро областей устойчивости — оттуда же. Отдельного модуля ему не
            // нужно: тело КРС в этом уже лежит, а регистров чужим расчётам оно
            // не стоит, регистровый бюджет считается на ядро.
            if (!module_fn(mod, "stabilityRegionKernel", fresh.kernel_stab, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

void ParametricEngine::Impl::prewarm_ad_cpu(const AdaptiveRequest& ad, int amountOfX) {
        if (!ad.setup_error.empty()) return;
        AdaptiveCpuModule mod;
        std::vector<KrsCpuDiag> diags;
        (void)mod.compile(ad.rhs, ad.emb, ad.dprep, ad.deval, ad.ctrl_body, amountOfX, ad.prefix + peak_config_defines(), diags);
    }

void ParametricEngine::Impl::prewarm_metrics(const SignalMetricsRequest& req) {
        if (req.use_cpu) {   // Fixed: CPU-шаг собирает cl.exe при Run; адаптивный — общая CPU-DLL
            if (req.adaptive.enabled) prewarm_ad_cpu(req.adaptive, req.amountOfX);
            return;
        }
        MetricsAxes ax;
        if (!metrics_axes(req, ax).empty()) return;
        std::string err;
        if (!ensure_init(err)) return;
        cuCtxSetCurrent(context);
        if (req.adaptive.enabled)
            compile_metrics_ad_if_needed(req.adaptive, req.krs_body, req.amountOfX, ax.par_or_var, err, false);
        else
            compile_metrics_if_needed(req.krs_body, req.amountOfX, ax.par_or_var, err, false, req.minmax_interp);
    }

StabilityResult ParametricEngine::Impl::run_stability(const StabilityRequest& req) {
        StabilityResult res;
        auto fail = [&](const std::string& msg) -> StabilityResult& { res.error = msg; return res; };

        const std::string bad = stability_validate(req);
        if (!bad.empty()) return fail(bad);
        stability_fill_axes(req, res);

        const int    amountOfValues = (int)req.values.size();
        const size_t total_cells    = (size_t)res.n_pts_x * (size_t)res.n_pts_y;

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        cudaGetLastError();   // сброс sticky-ошибки прошлого прогона, см. run_order

        // Модуль общий с Order: тот же ключ, та же компиляция, эталон выключен.
        if (!compile_order_module(true, req.amountOfX, amountOfValues, req.krs_body,
                                  std::string(), err)) return fail(err);
        if (cached_order.kernel_stab == nullptr)
            return fail("stabilityRegionKernel not found in the module");

        OrderDevBuf d_sig, d_om, d_values, d_rho, d_status, d_R;
        if (!d_sig   .alloc(res.sigma_vals.size() * sizeof(numb), "sigmaVals", err)) return fail(err);
        if (!d_om    .alloc(res.omega_vals.size() * sizeof(numb), "omegaVals", err)) return fail(err);
        if (!d_values.alloc((size_t)amountOfValues * sizeof(numb), "values",   err)) return fail(err);
        if (!d_rho   .alloc(total_cells * sizeof(numb), "outRho",    err)) return fail(err);
        if (!d_status.alloc(total_cells * sizeof(int),  "outStatus", err)) return fail(err);
        if (!d_R     .alloc(4 * total_cells * sizeof(numb), "outR",  err)) return fail(err);

        {
            const std::vector<numb> hs = to_numb(res.sigma_vals);
            const std::vector<numb> ho = to_numb(res.omega_vals);
            const std::vector<numb> va = to_numb(req.values);
            auto up = [&](void* dst, const void* src, size_t bytes, const char* what) -> bool {
                cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
                if (e != cudaSuccess) { err = std::string("memcpy ") + what + ": " + cudaGetErrorString(e); return false; }
                return true;
            };
            if (!up(d_sig.p,    hs.data(), hs.size() * sizeof(numb), "sigmaVals")) return fail(err);
            if (!up(d_om.p,     ho.data(), ho.size() * sizeof(numb), "omegaVals")) return fail(err);
            if (!up(d_values.p, va.data(), va.size() * sizeof(numb), "values"))    return fail(err);
        }

        const size_t cellsPerLaunch = 1u << 16;
        const size_t nLaunches = (total_cells + cellsPerLaunch - 1) / cellsPerLaunch;
        for (size_t L = 0; L < nLaunches; ++L) {
            const size_t offset = L * cellsPerLaunch;
            const size_t count  = (offset + cellsPerLaunch > total_cells) ? (total_cells - offset) : cellsPerLaunch;

            int   nPtsX_arg  = res.n_pts_x;
            int   nPtsY_arg  = res.n_pts_y;
            int   nCells_arg = (int)count;
            int   offset_arg = (int)offset;
            numb* sig_arg    = d_sig.as<numb>();
            numb* om_arg     = d_om.as<numb>();
            numb* values_arg = d_values.as<numb>();
            int   ia_arg = req.idx_a, ib_arg = req.idx_b, ic_arg = req.idx_c, id_arg = req.idx_d;
            numb  k_arg = (numb)req.k, r_arg = (numb)req.r, h_arg = (numb)req.h;
            numb* rho_arg = d_rho.as<numb>();
            int*  st_arg  = d_status.as<int>();
            numb* R_arg   = d_R.as<numb>();

            void* args[] = {
                &nPtsX_arg, &nPtsY_arg, &nCells_arg, &offset_arg,
                &sig_arg, &om_arg, &values_arg,
                &ia_arg, &ib_arg, &ic_arg, &id_arg,
                &k_arg, &r_arg, &h_arg,
                &rho_arg, &st_arg, &R_arg
            };

            const int blockSize = 64;
            const int gridSize  = (int)((count + blockSize - 1) / blockSize);
            CUresult r = cuLaunchKernel(cached_order.kernel_stab, gridSize, 1, 1, blockSize, 1, 1,
                                        0, nullptr, args, nullptr);
            if (r != CUDA_SUCCESS) return fail("cuLaunchKernel(stability): " + cu_err(r));
            cudaDeviceSynchronize();
            cudaError_t ce = cudaGetLastError();
            if (ce != cudaSuccess) return fail(std::string("stability kernel: ") + cudaGetErrorString(ce));

            if (req.progress)
                req.progress->store((float)((double)(offset + count) / (double)total_cells),
                                    std::memory_order_relaxed);
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
                res.cancelled = true;
                return res;
            }
        }

        {
            std::vector<numb> hrho(total_cells);
            res.status.assign(total_cells, 0);
            auto dn = [&](void* src, void* dst, size_t bytes, const char* what) -> bool {
                cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
                if (e != cudaSuccess) { err = std::string("memcpy D2H ") + what + ": " + cudaGetErrorString(e); return false; }
                return true;
            };
            if (!dn(d_rho.p,    hrho.data(),       total_cells * sizeof(numb), "rho"))    return fail(err);
            if (!dn(d_status.p, res.status.data(), total_cells * sizeof(int),  "status")) return fail(err);
            std::vector<numb> hR(4 * total_cells);
            if (!dn(d_R.p, hR.data(), hR.size() * sizeof(numb), "R")) return fail(err);

            res.rho.resize(total_cells);
            for (size_t i = 0; i < total_cells; ++i) res.rho[i] = (double)hrho[i];

            // Ошибка шага — на хосте, той же функцией, что в CPU-ветке.
            res.err.assign(total_cells, std::numeric_limits<double>::quiet_NaN());
            for (size_t i = 0; i < total_cells; ++i) {
                if (res.status[i] != STAB_ST_OK) continue;
                const size_t ix = i % (size_t)res.n_pts_x, iy = i / (size_t)res.n_pts_x;
                res.err[i] = stability_step_error(res.sigma_vals[ix], res.omega_vals[iy],
                                                  req.k, req.r, req.h,
                                                  (double)hR[i],                   (double)hR[total_cells + i],
                                                  (double)hR[2 * total_cells + i], (double)hR[3 * total_cells + i]);
            }
            stability_summarize(res);
        }

        res.ok = true;
        return res;
    }

PerfResult ParametricEngine::Impl::run_performance(const PerfRequest& req) {
        PerfResult res;
        res.axis = req.axis;
        auto fail = [&](const std::string& msg) -> PerfResult& { res.error = msg; return res; };

        if (req.axis.kind == OrderAxisKind::None) return fail("the axis is not set");
        if (req.repeats  < 1) return fail("the number of measurements must be >= 1");
        if (req.replicas < 1) return fail("the number of replicas must be >= 1");
        if (req.warmup   < 0) return fail("the number of warmup launches cannot be negative");

        // ---- Проход 1: ошибки (не засекается) ----
        OrderRequest oreq;
        oreq.krs_body           = req.krs_body;
        oreq.amountOfX          = req.amountOfX;
        oreq.initial_conditions = req.initial_conditions;
        oreq.values             = req.values;
        oreq.axis_x             = req.axis;
        oreq.axis_y             = OrderAxis{};          // kind == None -> одномерная сетка
        oreq.h                  = req.h;
        oreq.t_max              = req.t_max;
        oreq.snap_steps         = req.snap_steps;
        oreq.endpoint_only      = req.endpoint_only;
        oreq.max_value          = req.max_value;
        oreq.ref_krs_body       = req.ref_krs_body;
        oreq.ref_substeps       = req.ref_substeps;
        oreq.cancel             = req.cancel;

        // Прогресс order-прохода отдаётся в [0, 0.5]: без пересчёта бар
        // пробежал бы от нуля до конца дважды. run_order синхронный, поэтому
        // масштабированием занимается отдельный поток, живущий ровно на время
        // вызова.
        std::shared_ptr<std::atomic<float>> sub_prog;
        std::atomic<bool> watch_stop{ false };
        std::thread watcher;
        if (req.progress) {
            sub_prog = std::make_shared<std::atomic<float>>(0.0f);
            oreq.progress = sub_prog;
            watcher = std::thread([&req, &sub_prog, &watch_stop]() {
                while (!watch_stop.load(std::memory_order_relaxed)) {
                    req.progress->store(0.5f * sub_prog->load(std::memory_order_relaxed),
                                        std::memory_order_relaxed);
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
            });
        }
        OrderResult ores = run_order(oreq);
        if (watcher.joinable()) { watch_stop.store(true, std::memory_order_relaxed); watcher.join(); }

        if (ores.cancelled) { res.cancelled = true; return res; }
        if (!ores.ok) return fail(ores.error.empty() ? std::string("the order pass failed") : ores.error);

        res.n_pts     = ores.n_pts_x;
        res.axis_vals = ores.axis_x_vals;
        res.e1        = ores.e1;
        res.e2        = ores.e2;
        res.e_ref     = ores.e_ref;
        res.p         = ores.p;
        res.h_eff     = ores.h_eff;
        res.status    = ores.status;
        res.n_ok         = ores.n_ok;
        res.n_diverged   = ores.n_diverged;
        res.n_floor      = ores.n_floor;
        res.n_nocontract = ores.n_nocontract;
        res.repeats   = req.repeats;
        res.warmup    = req.warmup;
        res.replicas  = req.replicas;

        const int n = res.n_pts;
        if (n <= 0) return fail("empty grid");
        const double qnan = std::numeric_limits<double>::quiet_NaN();
        res.t_min.assign((size_t)n, qnan);
        res.t_max.assign((size_t)n, qnan);
        res.t_avg.assign((size_t)n, qnan);
        res.n_steps.assign((size_t)n, 0);

        // ---- Проход 2: замер ----
        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        cudaGetLastError();   // сброс sticky-ошибки, см. run_order

        const int amountOfValues = (int)req.values.size();
        const bool ref_on = !req.ref_krs_body.empty() && req.ref_substeps > 0;
        if (!compile_order_module(true, req.amountOfX, amountOfValues, req.krs_body,
                                  ref_on ? req.ref_krs_body : std::string(), err)) return fail(err);
        if (cached_order.kernel_perf == nullptr) return fail("perfIntegrateKernel not found in the module");

        OrderDevBuf d_X0, d_values, d_out;
        if (!d_X0    .alloc((size_t)req.amountOfX  * sizeof(numb), "X0",      err)) return fail(err);
        if (!d_values.alloc((size_t)amountOfValues * sizeof(numb), "values",  err)) return fail(err);
        if (!d_out   .alloc((size_t)req.replicas * (size_t)req.amountOfX * sizeof(numb), "perfOut", err)) return fail(err);

        std::vector<numb> h_vals = to_numb(req.values);
        {
            const std::vector<numb> x0 = to_numb(req.initial_conditions);
            cudaError_t e = cudaMemcpy(d_X0.p, x0.data(), x0.size() * sizeof(numb), cudaMemcpyHostToDevice);
            if (e != cudaSuccess) return fail(std::string("memcpy X0: ") + cudaGetErrorString(e));
            e = cudaMemcpy(d_values.p, h_vals.data(), h_vals.size() * sizeof(numb), cudaMemcpyHostToDevice);
            if (e != cudaSuccess) return fail(std::string("memcpy values: ") + cudaGetErrorString(e));
        }

        // Время меряет само ядро (%globaltimer вокруг цикла шагов, см. perfIntegrateKernel):
        // события вокруг запуска захватывали ещё и задержку запуска (~6-8 мкс на WDDM).
        OrderDevBuf d_tspan;
        if (!d_tspan.alloc(2 * sizeof(unsigned long long), "perfTimer", err)) return fail(err);

        // y(T) узла — для E(T) против эталона y*(T) (run_performance_any), как у адаптивного.
        res.y_end.assign((size_t)n, std::vector<double>());

        // Ширина блока — 32: измерено, что дальше упирается в регистры, а не в
        // лимит блоков на SM. При replicas == 1 это один активный warp.
        const int blockSize = 32;
        const int gridSize  = (req.replicas + blockSize - 1) / blockSize;
        bool first_node = true;

        for (int i = 0; i < n; ++i) {
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
                res.cancelled = true;
                return res;
            }

            const double h_node = res.h_eff[(size_t)i];
            if (!(h_node > 0.0) || !std::isfinite(h_node)) continue;   // узел без времени: см. status

            // Свип по параметру меняет a[] от узла к узлу. Загрузка делается
            // здесь, ДО прогрева, и в замер не попадает.
            if (req.axis.kind == OrderAxisKind::Value) {
                const int vi = req.axis.index;
                if (vi >= 0 && vi < amountOfValues) {
                    h_vals[(size_t)vi] = (numb)res.axis_vals[(size_t)i];
                    cudaError_t e = cudaMemcpy(d_values.p, h_vals.data(),
                                               h_vals.size() * sizeof(numb), cudaMemcpyHostToDevice);
                    if (e != cudaSuccess) return fail(std::string("memcpy values: ") + cudaGetErrorString(e));
                }
            }

            const long long N = order_steps_for(h_node, req.t_max, req.snap_steps);
            res.n_steps[(size_t)i] = N;

            int       nt_arg  = req.replicas;
            numb*     X0_arg  = d_X0.as<numb>();
            numb*     val_arg = d_values.as<numb>();
            numb      h_arg   = (numb)h_node;
            long long n_arg   = N;
            numb*     out_arg = d_out.as<numb>();
            unsigned long long* ts_none = nullptr;
            unsigned long long* ts_arg  = d_tspan.as<unsigned long long>();
            void* args_warm[] = { &nt_arg, &X0_arg, &val_arg, &h_arg, &n_arg, &out_arg, &ts_none };
            void* args[]      = { &nt_arg, &X0_arg, &val_arg, &h_arg, &n_arg, &out_arg, &ts_arg };

            auto launch = [&](const char* what, void** a) -> bool {
                CUresult r = cuLaunchKernel(cached_order.kernel_perf, gridSize, 1, 1, blockSize, 1, 1,
                                            0, nullptr, a, nullptr);
                if (r != CUDA_SUCCESS) { err = std::string("cuLaunchKernel(") + what + "): " + cu_err(r); return false; }
                return true;
            };
            auto sync = [&]() -> bool {
                cudaError_t ce = cudaDeviceSynchronize();
                if (ce != cudaSuccess) { err = std::string("perf kernel: ") + cudaGetErrorString(ce); return false; }
                return true;
            };

            // Прогрев по времени: GPU поднимает частоту только под сплошной нагрузкой, а
            // короткие запуски с синхронизацией между ними её не дают — без этого одно и то же
            // ядро мерилось то на холостой частоте, то на рабочей (до 3-5 раз разницы).
            // Пачка запусков подряд, без синхронизаций: ~300 мс перед первым узлом, ~20 мс
            // перед каждым следующим. Потом — warmup запусков из настроек, как раньше.
            if (!perf_burst_warmup(first_node ? 300.0 : 20.0,
                                   [&]() { return launch("perf warmup", args_warm); }, sync))
                return fail(err);
            first_node = false;
            for (int w = 0; w < req.warmup; ++w)
                if (!launch("perf warmup", args_warm) || !sync()) return fail(err);

            double tmin = 0.0, tmax = 0.0, tsum = 0.0;
            int got = 0;
            for (int rep = 0; rep < req.repeats; ++rep) {
                const unsigned long long ts_init[2] = { ~0ULL, 0ULL };
                cudaError_t ce = cudaMemcpy(d_tspan.p, ts_init, sizeof(ts_init), cudaMemcpyHostToDevice);
                if (ce != cudaSuccess) return fail(std::string("memcpy timer: ") + cudaGetErrorString(ce));
                if (!launch("perf", args) || !sync()) return fail(err);
                unsigned long long ts[2] = { 0, 0 };
                ce = cudaMemcpy(ts, d_tspan.p, sizeof(ts), cudaMemcpyDeviceToHost);
                if (ce != cudaSuccess) return fail(std::string("memcpy timer: ") + cudaGetErrorString(ce));
                const double us = ts[1] >= ts[0] ? (double)(ts[1] - ts[0]) * 1.0e-3 : 0.0;
                if (got == 0) { tmin = tmax = us; }
                else { if (us < tmin) tmin = us; if (us > tmax) tmax = us; }
                tsum += us;
                ++got;

                if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
                    res.cancelled = true;
                    return res;
                }
            }
            if (got > 0) {
                res.t_min[(size_t)i] = tmin;
                res.t_max[(size_t)i] = tmax;
                res.t_avg[(size_t)i] = tsum / (double)got;
            }

            // y(T) первой реплики — только если узел кончился ровно в t_max (fit h to t_max):
            // иначе сравнивать с y*(T) нечего, E(T) узла остаётся пустым.
            if ((double)N * h_node > 0.0
                && std::fabs((double)N * h_node - req.t_max) <= 1e-9 * (std::max)(1.0, std::fabs(req.t_max))) {
                std::vector<numb> y((size_t)req.amountOfX);
                cudaError_t ce = cudaMemcpy(y.data(), d_out.p, y.size() * sizeof(numb), cudaMemcpyDeviceToHost);
                if (ce != cudaSuccess) return fail(std::string("memcpy y(T): ") + cudaGetErrorString(ce));
                res.y_end[(size_t)i].assign(y.begin(), y.end());
            }

            if (req.progress)
                req.progress->store(0.5f + 0.5f * (float)(i + 1) / (float)n, std::memory_order_relaxed);
        }

        // Диапазоны для автоскейла: только узлы, у которых есть И время, И
        // конечная положительная ошибка — точка графика существует лишь тогда,
        // когда есть обе координаты.
        bool first_t = true, first_e1 = true, first_e2 = true, first_er = true;
        for (int i = 0; i < n; ++i) {
            const double ta = res.t_avg[(size_t)i];
            if (!std::isfinite(ta)) continue;
            const double lo = res.t_min[(size_t)i], hi = res.t_max[(size_t)i];
            if (first_t) { res.t_lo = lo; res.t_hi = hi; first_t = false; }
            else { if (lo < res.t_lo) res.t_lo = lo; if (hi > res.t_hi) res.t_hi = hi; }
            const double e1 = res.e1[(size_t)i], e2 = res.e2[(size_t)i];
            if (std::isfinite(e1) && e1 > 0.0) {
                if (first_e1) { res.e1_min = res.e1_max = e1; first_e1 = false; }
                else { if (e1 < res.e1_min) res.e1_min = e1; if (e1 > res.e1_max) res.e1_max = e1; }
            }
            if (std::isfinite(e2) && e2 > 0.0) {
                if (first_e2) { res.e2_min = res.e2_max = e2; first_e2 = false; }
                else { if (e2 < res.e2_min) res.e2_min = e2; if (e2 > res.e2_max) res.e2_max = e2; }
            }
            const double er = ((size_t)i < res.e_ref.size())
                                  ? res.e_ref[(size_t)i]
                                  : std::numeric_limits<double>::quiet_NaN();
            if (std::isfinite(er) && er > 0.0) {
                if (first_er) { res.eref_min = res.eref_max = er; first_er = false; }
                else { if (er < res.eref_min) res.eref_min = er; if (er > res.eref_max) res.eref_max = er; }
            }
        }

        res.ok = true;
        return res;
    }

bool ParametricEngine::Impl::compile_network_module(bool activate, int amountOfX, int amountOfValues,
                                const std::string& krs_body, const std::string& coupling_body,
                                std::string& err)
    {
        cuCtxSetCurrent(context);
        // Тело связи входит в ключ: оно подставляется в switch внутри ядра, и
        // смена закона связи — такая же перекомпиляция, как смена КРС.
        const std::string key = hash_key(krs_body, amountOfX) + ":net:v"
                              + std::to_string(amountOfValues)
                              + ":c" + std::to_string(std::hash<std::string>{}(coupling_body));
        return compile_into(pool_network, key, cached_network, activate, [&](CachedNetworkModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> lowered;
            if (!build_module(snapshot_sources(src_template_network), "network.cu",
                              { { "{{AMOUNT_OF_X}}",      std::to_string(amountOfX) },
                                { "{{AMOUNT_OF_VALUES}}", std::to_string(amountOfValues) },
                                { "{{KRS_BODY}}",         krs_body },
                                { "{{COUPLING_BODY}}",    coupling_body } },
                              {}, mod, lowered, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, "networkIntegrateKernel", fresh.kernel, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

NetworkResult ParametricEngine::Impl::run_network(const NetworkRequest& req) {
        NetworkResult res;
        auto fail = [&](const std::string& msg) -> NetworkResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX) return fail("amountOfX out of range");
        if (req.n_nodes <= 0)                                    return fail("the network has no nodes");
        if (req.n_nodes > kMaxNetworkNodes)
            return fail("too many nodes: the whole network runs in one block, the cap is "
                        + std::to_string(kMaxNetworkNodes));
        if (req.amountOfValues <= 0 || req.amountOfValues > kMaxAmountOfValues)
            return fail("amountOfValues out of range");
        if ((int)req.values.size() != req.n_nodes * req.amountOfValues)
            return fail("values.size() != n_nodes * amountOfValues");
        if ((int)req.initial_conditions.size() != req.n_nodes * req.amountOfX)
            return fail("initial_conditions.size() != n_nodes * amountOfX");
        if ((int)req.edge_start.size() != req.n_nodes + 1)
            return fail("edge_start.size() != n_nodes + 1");
        const int n_arcs = req.edge_start.back();
        if (n_arcs < 0 || (int)req.edge_src.size() != n_arcs
            || (int)req.edge_w.size() != n_arcs || (int)req.edge_law.size() != n_arcs)
            return fail("the CSR arrays disagree on the number of arcs");
        for (int i = 0; i < req.n_nodes; ++i)
            if (req.edge_start[(size_t)i] > req.edge_start[(size_t)i + 1])
                return fail("edge_start is not monotonic");
        for (int e = 0; e < n_arcs; ++e)
            if (req.edge_src[(size_t)e] < 0 || req.edge_src[(size_t)e] >= req.n_nodes)
                return fail("an arc points outside the node list");
        if (!(req.h > 0.0))     return fail("h must be > 0");
        if (!(req.t_max > 0.0)) return fail("t_max must be > 0");
        if (req.transient < 0.0) return fail("transient must be >= 0");
        if (req.pre_scaller < 1) return fail("preScaller must be >= 1");
        if (req.max_points < 2)  return fail("max_points must be >= 2");
        if (req.steps_per_launch < 0) return fail("steps_per_launch must be >= 0");

        // Состояние всей сети живёт в динамической shared — отсюда потолок.
        const size_t shared_bytes = (size_t)req.n_nodes * (size_t)req.amountOfX * sizeof(numb);
        if (shared_bytes > kNetworkSharedCap)
            return fail("the network state does not fit into shared memory ("
                        + std::to_string(shared_bytes) + " B > " + std::to_string(kNetworkSharedCap)
                        + " B): reduce the node count or the system dimension");

        const long long skipSteps  = (long long)steps_from_time_size_t(req.transient, req.h);
        const long long workSteps  = (long long)steps_from_time_size_t(req.t_max, req.h);
        const long long totalSteps = skipSteps + workSteps;
        if (totalSteps <= 0) return fail("t_max / h gives no steps");
        if ((double)totalSteps > 1.0e12)
            return fail("t_max / h too large: the work does not fit into a reasonable time");

        // Прореживание: пользовательское, но поднятое до того, при котором
        // записанное влезает в max_points. Поднимать молча нельзя — фактическое
        // значение уезжает в результат и показывается в UI.
        int preScaller = req.pre_scaller;
        {
            const long long need = workSteps / preScaller + 1;
            if (need > (long long)req.max_points) {
                const long long k = (workSteps + (long long)req.max_points - 2)
                                  / ((long long)req.max_points - 1);
                if (k > preScaller) preScaller = (int)k;
            }
        }
        const long long nPointsLL = workSteps / preScaller + 1;
        if (nPointsLL <= 0 || nPointsLL > 2147483647LL) return fail("the recorded point count is out of range");
        const int nPoints = (int)nPointsLL;

        const size_t out_elems = (size_t)nPoints * (size_t)req.n_nodes * (size_t)req.amountOfX;
        const size_t out_bytes = out_elems * sizeof(numb);
        if (out_bytes > kNetworkOutCap)
            return fail("the recorded trajectory would take " + std::to_string(out_bytes >> 20)
                        + " MB: raise the decimation or lower t_max");

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        cudaGetLastError();   // сброс sticky-ошибки прошлого прогона, см. run_fastsync

        if (!compile_network_module(true, req.amountOfX, req.amountOfValues,
                                    req.krs_body, req.coupling_body, err)) return fail(err);

        OrderDevBuf d_values, d_state, d_eStart, d_eSrc, d_eW, d_eLaw, d_out, d_status;
        if (!d_values.alloc(req.values.size() * sizeof(numb), "values", err)) return fail(err);
        if (!d_state .alloc(req.initial_conditions.size() * sizeof(numb), "state", err)) return fail(err);
        if (!d_eStart.alloc(req.edge_start.size() * sizeof(int), "edgeStart", err)) return fail(err);
        if (!d_eSrc  .alloc((size_t)(n_arcs > 0 ? n_arcs : 1) * sizeof(int),  "edgeSrc", err)) return fail(err);
        if (!d_eW    .alloc((size_t)(n_arcs > 0 ? n_arcs : 1) * sizeof(numb), "edgeW",   err)) return fail(err);
        if (!d_eLaw  .alloc((size_t)(n_arcs > 0 ? n_arcs : 1) * sizeof(int),  "edgeLaw", err)) return fail(err);
        if (!d_out   .alloc(out_bytes, "out", err)) return fail(err);
        if (!d_status.alloc((size_t)req.n_nodes * sizeof(int), "status", err)) return fail(err);

        {
            const std::vector<numb> hv  = to_numb(req.values);
            const std::vector<numb> hic = to_numb(req.initial_conditions);
            const std::vector<numb> hw  = to_numb(req.edge_w);
            auto up = [&](void* dst, const void* src, size_t bytes, const char* what) -> bool {
                if (bytes == 0) return true;
                cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
                if (e != cudaSuccess) { err = std::string("memcpy ") + what + ": " + cudaGetErrorString(e); return false; }
                return true;
            };
            if (!up(d_values.p, hv.data(),  hv.size()  * sizeof(numb), "values"))    return fail(err);
            if (!up(d_state.p,  hic.data(), hic.size() * sizeof(numb), "state"))     return fail(err);
            if (!up(d_eStart.p, req.edge_start.data(), req.edge_start.size() * sizeof(int), "edgeStart")) return fail(err);
            if (!up(d_eSrc.p,   req.edge_src.data(),   (size_t)n_arcs * sizeof(int),  "edgeSrc")) return fail(err);
            if (!up(d_eW.p,     hw.data(),             (size_t)n_arcs * sizeof(numb), "edgeW"))   return fail(err);
            if (!up(d_eLaw.p,   req.edge_law.data(),   (size_t)n_arcs * sizeof(int),  "edgeLaw")) return fail(err);
        }
        // 0xFF по всем байтам — это NaN и для float, и для double: точки после
        // разлёта обязаны читаться как «не посчитано», а не как ноль.
        if (cudaMemset(d_out.p, 0xFF, out_bytes) != cudaSuccess) return fail("memset out");
        if (cudaMemset(d_status.p, 0, (size_t)req.n_nodes * sizeof(int)) != cudaSuccess)
            return fail("memset status");

        RunSignals sig;
        if (!sig.alloc(err)) return fail(err);
        struct SigGuard { RunSignals& s; ~SigGuard() { s.release(); } } sig_guard{ sig };

        // Один блок = одна сеть, поток = узел. Идущие сверх узлов потоки стоят
        // на тех же __syncthreads, что и рабочие, и ничего не считают.
        int blockSize = ((req.n_nodes + 31) / 32) * 32;
        if (blockSize > 1024) blockSize = 1024;

        // Чанкование против watchdog'а (TDR): расчёт последователен по времени,
        // поэтому режем его по шагам, а состояние переносится через d_state.
        const double workPerStep = (double)req.n_nodes * (1.0 + (double)n_arcs / (double)req.n_nodes);
        long long stepsPerLaunch = (long long)(kNetworkWorkBudget / (workPerStep > 1.0 ? workPerStep : 1.0));
        if (stepsPerLaunch < 1)          stepsPerLaunch = 1;
        if (stepsPerLaunch > totalSteps) stepsPerLaunch = totalSteps;

        if (req.steps_per_launch > 0 && req.steps_per_launch < stepsPerLaunch)
            stepsPerLaunch = req.steps_per_launch;

        // Event pairs are allocated only for an explicit benchmark request.
        // Record the end before host polling/status copies, so these are excluded.
        struct NetworkEvents {
            CUevent begin = nullptr, end = nullptr;
            ~NetworkEvents() {
                if (end && cuEventDestroy(end) != CUDA_SUCCESS)
                    std::fprintf(stderr, "network: cuEventDestroy(end) failed\n");
                if (begin && cuEventDestroy(begin) != CUDA_SUCCESS)
                    std::fprintf(stderr, "network: cuEventDestroy(begin) failed\n");
            }
        } events;
        if (req.measure_gpu_time) {
            CUresult r = cuEventCreate(&events.begin, CU_EVENT_DEFAULT);
            if (r != CUDA_SUCCESS) return fail("network event create: " + cu_err(r));
            r = cuEventCreate(&events.end, CU_EVENT_DEFAULT);
            if (r != CUDA_SUCCESS) return fail("network event create: " + cu_err(r));
        }

        const int    progressStride = progress_stride_for((size_t)totalSteps);
        const double ticksTotal     = (double)totalSteps / (double)progressStride;
        double ticksDone = 0.0;

        for (long long base = 0; base < totalSteps; base += stepsPerLaunch) {
            const long long left  = totalSteps - base;
            const long long count = (stepsPerLaunch < left) ? stepsPerLaunch : left;

            numb*     values_arg = d_values.as<numb>();
            numb*     state_arg  = d_state.as<numb>();
            int*      eStart_arg = d_eStart.as<int>();
            int*      eSrc_arg   = d_eSrc.as<int>();
            numb*     eW_arg     = d_eW.as<numb>();
            int*      eLaw_arg   = d_eLaw.as<int>();
            int       nNodes_arg = req.n_nodes;
            numb      h_arg      = (numb)req.h;
            long long base_arg   = base;
            long long count_arg  = count;
            long long skip_arg   = skipSteps;
            int       pre_arg    = preScaller;
            int       nPts_arg   = nPoints;
            numb      maxV_arg   = (numb)req.max_value;
            numb*     out_arg    = d_out.as<numb>();
            int*      status_arg = d_status.as<int>();
            int*      cancel_arg = sig.cancelArg();
            int*      prog_arg   = sig.progressArg();
            int       stride_arg = progressStride;

            void* args[] = {
                &values_arg, &state_arg, &eStart_arg, &eSrc_arg, &eW_arg, &eLaw_arg,
                &nNodes_arg, &h_arg, &base_arg, &count_arg, &skip_arg, &pre_arg, &nPts_arg,
                &maxV_arg, &out_arg, &status_arg, &cancel_arg, &prog_arg, &stride_arg
            };

            sig.resetTicks();
            if (req.measure_gpu_time) {
                CUresult r = cuEventRecord(events.begin, nullptr);
                if (r != CUDA_SUCCESS) return fail("network event begin: " + cu_err(r));
            }
            CUresult r = cuLaunchKernel(cached_network.kernel, 1, 1, 1, blockSize, 1, 1,
                                        (unsigned)shared_bytes, nullptr, args, nullptr);
            if (r != CUDA_SUCCESS) return fail("cuLaunchKernel(network): " + cu_err(r));
            ++res.launch_count;
            if (req.measure_gpu_time) {
                r = cuEventRecord(events.end, nullptr);
                if (r != CUDA_SUCCESS) return fail("network event end: " + cu_err(r));
            }
            if (!wait_with_signals(0, sig, req.cancel, req.progress, ticksDone, ticksTotal, err))
                return fail(err);
            cudaDeviceSynchronize();
            cudaError_t ce = cudaGetLastError();
            if (ce != cudaSuccess) return fail(std::string("network kernel: ") + cudaGetErrorString(ce));

            if (req.measure_gpu_time) {
                float elapsed = 0.0f; // CUDA event API requires float milliseconds.
                r = cuEventElapsedTime(&elapsed, events.begin, events.end);
                if (r != CUDA_SUCCESS) return fail("network event elapsed: " + cu_err(r));
                res.gpu_ms += elapsed;
            }

            ticksDone += (double)count / (double)progressStride;

            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
                res.cancelled = true;
                return res;
            }

            // Разлетелась сеть — остальные чанки досчитывать нечего: ядро на
            // них всё равно сразу выйдет, а пользователь ждал бы впустую.
            std::vector<int> st((size_t)req.n_nodes, 0);
            if (cudaMemcpy(st.data(), d_status.p, st.size() * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
                return fail("memcpy D2H status");
            bool diverged = false;
            for (int v : st) if (v != NET_ST_OK) diverged = true;
            if (diverged) break;
        }

        res.status.assign((size_t)req.n_nodes, 0);
        if (cudaMemcpy(res.status.data(), d_status.p, res.status.size() * sizeof(int),
                       cudaMemcpyDeviceToHost) != cudaSuccess) return fail("memcpy D2H status");
        for (int v : res.status) if (v != NET_ST_OK) ++res.n_diverged;

        {
            std::vector<numb> host(out_elems);
            if (cudaMemcpy(host.data(), d_out.p, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
                return fail("memcpy D2H out");
            res.data.resize(out_elems);
            res.vmin.assign((size_t)req.amountOfX, 0.0);
            res.vmax.assign((size_t)req.amountOfX, 0.0);
            std::vector<bool> seen((size_t)req.amountOfX, false);
            for (size_t i = 0; i < out_elems; ++i) {
                const double v = (double)host[i];
                res.data[i] = v;
                if (!std::isfinite(v)) continue;
                const size_t k = i % (size_t)req.amountOfX;
                if (!seen[k]) { res.vmin[k] = res.vmax[k] = v; seen[k] = true; }
                else { if (v < res.vmin[k]) res.vmin[k] = v; if (v > res.vmax[k]) res.vmax[k] = v; }
            }
        }

        res.n_nodes     = req.n_nodes;
        res.n_vars      = req.amountOfX;
        res.n_points    = nPoints;
        res.h           = req.h;
        res.t0          = (double)skipSteps * req.h;
        res.dt          = (double)preScaller * req.h;
        res.pre_scaller = preScaller;
        res.ok          = true;
        return res;
    }
