#pragma once
#include "engine_helpers.h"
using namespace ucuda::cuda_detail;
struct ParametricEngine::Impl {

    bool       inited   = false;

    CUcontext  context  = nullptr;

    CUdevice   device   = 0;

    int        cc_major = 0;

    int        cc_minor = 0;


    // Сигнальная страница расчёта: [0] — запрос отмены, [1] — тики прогресса.
    //
    // Mapped host memory, а НЕ память устройства. На WDDM любой cudaMemcpy,
    // выданный пока работает ядро, встаёт за ним в очередь и блокирует хост:
    // цикл опроса делал ровно одну итерацию, и ни прогресс, ни Cancel не
    // работали. Через mapped-страницу хост читает и пишет обычными load/store,
    // без единого вызова CUDA.
    struct RunSignals {
        int* host = nullptr;   // страница на хосте
        int* dev  = nullptr;   // её же device-проекция для ядра

        bool alloc(std::string& err) {
            CUresult r = cuMemHostAlloc((void**)&host, 2 * sizeof(int), CU_MEMHOSTALLOC_DEVICEMAP);
            if (r != CUDA_SUCCESS) { err = "cuMemHostAlloc(signals): " + cu_err(r); return false; }
            r = cuMemHostGetDevicePointer((CUdeviceptr*)&dev, host, 0);
            if (r != CUDA_SUCCESS) { err = "cuMemHostGetDevicePointer: " + cu_err(r); release(); return false; }
            host[0] = 0; host[1] = 0;
            return true;
        }
        void release()          { if (host) { cuMemFreeHost(host); host = nullptr; dev = nullptr; } }
        int* cancelArg()  const { return dev; }
        int* progressArg()const { return dev ? dev + 1 : nullptr; }
        void resetTicks()       { if (host) ((volatile int*)host)[1] = 0; }
        int  ticks()      const { return host ? ((volatile int*)host)[1] : 0; }
        void raiseCancel()      { if (host) ((volatile int*)host)[0] = 1; }
    };


    // Шаг тика прогресса. Кратен CHECK_INTERVAL, потому что тики ставятся в уже
    // существующей проверке расходимости, и подобран так, чтобы точка отчиталась
    // около 64 раз за свою работу: этого хватает на гладкий бар и не создаёт
    // давки на одном адресе.
    static int progress_stride_for(size_t stepsPerPoint) ;


    // Ждём, пока отданная в stream работа закончится, обновляя прогресс из тиков
    // и передавая Cancel в ядро. В теле цикла не должно быть вызовов CUDA, кроме
    // cudaStreamQuery — см. комментарий к RunSignals.
    bool wait_with_signals(cudaStream_t stream, RunSignals& sig,
                           const std::shared_ptr<std::atomic<bool>>& cancel,
                           const std::shared_ptr<std::atomic<float>>& progress,
                           double ticksBefore, double ticksTotal, std::string& err) const
    ;


    // Закэшированные тексты NonLinAnal headers (читаются один раз)
    std::string src_cudaLibrary_cu;

    std::string src_cudaLibrary_cuh;

    std::string src_cudaMacros_cuh;

    std::string src_ucuda_adaptive_cuh;
   // драйвер адаптивного шага (kernels/ucuda_adaptive.cuh)
    std::string src_adaptive_part;
        // адаптивные ядра свипов (kernels/adaptive_part.cu)
    std::string src_metrics_adaptive_part;
 // адаптивное ядро метрик (kernels/metrics_adaptive_part.cu)
    std::string src_lyap_adaptive_part;
    // адаптивные LLE/LS (kernels/lyapunov_adaptive_part.cu)
    // curand_kernel.h перехвачен inline-stub'ом в каждом template'е (kernels/*.cu), а
    // `#define CURAND_KERNEL_H_` блокирует реальный header; virtual header'а здесь больше нет —
    // иначе он повторно объявлял бы curandState_t.
    // src_configCUDA_h = peak_config_defines() + src_configCUDA_h_raw: сырой текст держим отдельно,
    // чтобы пересборка при смене Settings не лезла на диск.
    std::string src_configCUDA_h_raw;

    std::string src_configCUDA_h;

    std::string src_template;
        // bifurcation1d.template.cu
    std::string src_template_cont;
       // bifurcation1d_cont.template.cu
    std::string src_template_lle_cont;
   // lle1d_cont.template.cu
    std::string src_template_ls_cont;
    // ls1d_cont.template.cu
    std::string src_template_dft_cont;
   // dft1d_cont.template.cu
    std::string src_template_dft_hsweep;
 // dft1d_hsweep.template.cu
    std::string src_template_lle;
    // lle1d.template.cu
    std::string src_template_lle_2d;
 // lle2d.template.cu
    std::string src_template_ls;
     // ls1d.template.cu
    std::string src_template_ls_2d;
  // ls2d.template.cu
    std::string src_template_bif2d;
  // bifurcation2d.template.cu
    std::string src_template_basins;
 // basins.template.cu
    std::string src_template_fs_attr;
 // fastsync_attr.template.cu (mode 0)
    std::string src_template_fs_grid;
 // fastsync_grid.template.cu (mode 1)
    std::string src_template_order;
   // order.template.cu
    std::string src_template_network;
 // network.template.cu
    std::string src_template_metrics;
 // signal_metrics.template.cu
    bool     srcs_loaded     = false;

    uint64_t srcs_peak_epoch = 0;
   // != peak_config_epoch() -> пересобрать configCUDA.h

    struct CachedModule {
        std::string key;
        CUmodule    module      = nullptr;
        CUfunction  kernel_traj = nullptr;  // calculateDiscreteModelCUDA
        CUfunction  kernel_peak = nullptr;  // peakFinderCUDA
        CUfunction  kernel_fused = nullptr; // calculateDiscreteModelPeaksCUDA
        CUfunction  kernel_dft  = nullptr;  // DFT_custom (used by run_dft_1d classical branch)
    };

    CachedModule cached;
          // bif1d kernels (traj + peak + dft)

    // Отдельный модуль для LLE — другой шаблон, другой kernel.
    struct CachedLleModule {
        std::string key;
        CUmodule    module      = nullptr;
        CUfunction  kernel_lle  = nullptr;  // LLEKernelCUDA
    };

    CachedLleModule cached_lle;


    // LLE-2D — отдельный шаблон/par_or_var-ветка, тот же kernel.
    CachedLleModule cached_lle_2d;


    // LS — отдельный модуль/кэш (третий слот PTX).
    struct CachedLsModule {
        std::string key;
        CUmodule    module     = nullptr;
        CUfunction  kernel_ls  = nullptr;   // LSKernelCUDA
    };

    CachedLsModule cached_ls;


    // LS-2D — отдельный шаблон/par_or_var-ветка, тот же kernel.
    CachedLsModule cached_ls_2d;


    // Continuation bif1d — четвёртый слот. Кернел single-thread, плюс
    // peakFinderCUDA из того же модуля (он определён в cudaLibrary.cu).
    struct CachedContModule {
        std::string key;
        CUmodule    module     = nullptr;
        CUfunction  kernel_cont = nullptr;  // bifurcation1dContinuationKernel
        CUfunction  kernel_peak = nullptr;  // peakFinderCUDA
        CUfunction  kernel_dft  = nullptr;  // DFT_custom (used by run_dft_1d continuation branch)
    };

    CachedContModule cached_cont;


    // Continuation LLE-1D / LS-1D — те же single-thread ядра по идее, но каждое
    // в своём модуле (у них свой шаблон и свой KRS-инстанс). Peak-finder им не
    // нужен: результат — сразу число (lambda) или N чисел (спектр).
    struct CachedSimpleContModule {
        std::string key;
        CUmodule    module = nullptr;
        CUfunction  kernel = nullptr;
    };

    CachedSimpleContModule cached_lle_cont;
   // lle1dContinuationKernel
    CachedSimpleContModule cached_ls_cont;
    // ls1dContinuationKernel
    CachedSimpleContModule cached_dft_cont;
   // dft1dContinuationKernel
    // Классический (без continuation) DFT-свип по h — тоже одноядерный модуль,
    // но ядро параллельное: поток на точку (см. dft1d_hsweep.template.cu).
    CachedSimpleContModule cached_dft_hsweep;
 // dft1dHSweepKernel

    // Bif-2D — отдельный шаблон (bifurcation2d.template.cu), три kernel'а:
    // calculateDiscreteModelPeaksCUDA + dbscanCUDA. Траектория больше не хранится,
    // поэтому traj и peak — одно ядро (см. PeakStream в cudaLibrary.cu).
    struct CachedBif2dModule {
        std::string  key;
        CUmodule     module        = nullptr;
        CUfunction   kernel_fused  = nullptr;   // calculateDiscreteModelPeaksCUDA
        CUfunction   kernel_dbscan = nullptr;   // dbscanCUDA
    };

    CachedBif2dModule cached_bif2d;


    // Basins — отдельный шаблон, пять kernel'ов (трое для DBSCAN host-цикла).
    struct CachedBasinsModule {
        std::string  key;
        CUmodule     module                  = nullptr;
        CUfunction   kernel_fused            = nullptr;  // calculateDiscreteModelAvgPeaksCUDA
        CUfunction   kernel_dbscan           = nullptr;  // CUDA_dbscan_kernel
        CUfunction   kernel_search_fixed     = nullptr;  // CUDA_dbscan_search_fixed_points_kernel
        CUfunction   kernel_search_clear     = nullptr;  // CUDA_dbscan_search_clear_points_kernel
    };

    CachedBasinsModule cached_basins;


    // Адаптивный шаг в свипах: bifurcation2d.template.cu + kernels/adaptive_part.cu. Один
    // модуль на (КРС, тела адаптивного шага, par_or_var): пики БД 1D/2D, признаки бассейнов
    // и DBSCAN обеих — всё из одного шаблона.
    struct CachedAdModule {
        std::string  key;
        CUmodule     module              = nullptr;
        CUfunction   kernel_peaks        = nullptr;  // calculateDiscreteModelPeaksAdCUDA
        CUfunction   kernel_avg          = nullptr;  // calculateDiscreteModelAvgPeaksAdCUDA
        CUfunction   kernel_dbscan       = nullptr;  // dbscanCUDA (БД 2D)
        CUfunction   kernel_cdbscan      = nullptr;  // CUDA_dbscan_kernel (бассейны)
        CUfunction   kernel_search_fixed = nullptr;  // CUDA_dbscan_search_fixed_points_kernel
        CUfunction   kernel_search_clear = nullptr;  // CUDA_dbscan_search_clear_points_kernel
    };

    CachedAdModule cached_ad;

    // Continuation БД 1D с адаптивным шагом — свой модуль (UCUDA_AD_CONT_KERNEL): ядро с
    // тремя копиями шага нужно редко, а в общем модуле удлиняло бы каждую первую сборку.
    CachedSimpleContModule cached_ad_cont;
   // calculateDiscreteModelPeaksAdContCUDA
    // LLE/LS с адаптивным шагом: bifurcation2d.template.cu + общая часть adaptive_part.cu
    // (без ядер пиков и без плотного выхода) + kernels/lyapunov_adaptive_part.cu.
    struct CachedLyapAdModule {
        std::string  key;
        CUmodule     module     = nullptr;
        CUfunction   kernel_lle = nullptr;   // lyapunovLLEAdCUDA
        CUfunction   kernel_ls  = nullptr;   // lyapunovLSAdCUDA
    };

    CachedLyapAdModule cached_lyap_ad;

    // Continuation LLE/LS с адаптивным шагом: тот же текст с UCUDA_LYAP_AD_CONT_ONLY —
    // однопоточные ядра цепочки lyapunovLLEAdContCUDA / lyapunovLSAdContCUDA.
    CachedLyapAdModule cached_lyap_ad_cont;


    // Fast Synchro — два модуля (mode 0 = on attractor, mode 1 = on grid).
    // Каждый кэширует свой PTX, ключ = hash(krs_body)+amountOfX+":fs_attr"/":fs_grid"
    // + (type_of_synch, error_estim, fs_error_trs) — все три substituted в #define
    // перед include configCUDA.h, поэтому смена любого требует recompile.
    struct CachedFastSyncModule {
        std::string  key;
        CUmodule     module          = nullptr;
        // FS-specific. НЕ используем calculateDiscreteModelCUDA (та пишет
        // скалярную сводку, не полный X[]).
        CUfunction   kernel_fs_fill  = nullptr;  // fillFSMasterTrajectory (template, mode 0)
        CUfunction   kernel_fs_traj  = nullptr;  // calculateDiscreteModelforFastSynchroCUDA (mode 0)
        CUfunction   kernel_fs_grid  = nullptr;  // calculateDiscreteModelICCforFastSynchro (mode 1)
    };

    CachedFastSyncModule cached_fs_attr;

    CachedFastSyncModule cached_fs_grid;


    // Order — одно ядро, но ключ кэша включает ещё и РАЗМЕР a[]: шаблон
    // объявляет локальный numb a[AMOUNTOFVALUES], а добавление системе
    // неиспользуемого в правых частях параметра (именно так задаются
    // параметры метода для кастомных КРС) krs_body не меняет.
    struct CachedOrderModule {
        std::string key;
        CUmodule    module = nullptr;
        CUfunction  kernel = nullptr;        // orderEstimateKernel
        CUfunction  kernel_perf = nullptr;   // perfIntegrateKernel (вкладка Performance)
        CUfunction  kernel_stab = nullptr;   // stabilityRegionKernel (области устойчивости)
    };

    CachedOrderModule cached_order;


    // Network — ключ включает размер a[] (шаблон объявляет его константой) и
    // тело связи: оно подставляется в switch внутри ядра, и смена закона
    // связи требует перекомпиляции ровно так же, как смена КРС.
    struct CachedNetworkModule {
        std::string key;
        CUmodule    module = nullptr;
        CUfunction  kernel = nullptr;        // networkIntegrateKernel
    };

    CachedNetworkModule cached_network;


    // Signal metrics — свой шаблон и одно ядро; ключ как у bif2d (+ par_or_var).
    struct CachedMetricsModule {
        std::string key;
        CUmodule    module = nullptr;
        CUfunction  kernel = nullptr;        // calculateDiscreteModelMetricsCUDA
        CUfunction  kernel_cont = nullptr;   // signalMetricsContinuationKernel
    };

    CachedMetricsModule cached_metrics;

    // Метрики с адаптивным шагом: шаблон метрик + adaptive_part.cu + metrics_adaptive_part.cu.
    // kernel — calculateDiscreteModelMetricsAdCUDA, kernel_cont — signalMetricsContinuationAdKernel.
    CachedMetricsModule cached_metrics_ad;


    // Every cached_* above is just a view on the active entry of its pool; the pool owns the
    // modules. One slot per analysis type meant recompiling on every switch back to a scheme that
    // had already been built minutes ago -- 6 seconds of NVRTC for nothing on an implicit scheme.
    static constexpr std::size_t kModuleCacheCapacity = 4;


    // Guards the pools and the active slots: background prewarming will make compile_* concurrent.
    std::recursive_mutex cache_mu;


    ModuleLru<CachedModule>            pool_bif1d     { kModuleCacheCapacity };

    ModuleLru<CachedLleModule>         pool_lle       { kModuleCacheCapacity };

    ModuleLru<CachedLleModule>         pool_lle_2d    { kModuleCacheCapacity };

    ModuleLru<CachedLsModule>          pool_ls        { kModuleCacheCapacity };

    ModuleLru<CachedLsModule>          pool_ls_2d     { kModuleCacheCapacity };

    ModuleLru<CachedContModule>        pool_cont      { kModuleCacheCapacity };

    ModuleLru<CachedSimpleContModule>  pool_lle_cont  { kModuleCacheCapacity };

    ModuleLru<CachedSimpleContModule>  pool_ls_cont   { kModuleCacheCapacity };

    ModuleLru<CachedSimpleContModule>  pool_dft_cont  { kModuleCacheCapacity };

    ModuleLru<CachedSimpleContModule>  pool_dft_hsweep{ kModuleCacheCapacity };

    ModuleLru<CachedBif2dModule>       pool_bif2d     { kModuleCacheCapacity };

    ModuleLru<CachedBasinsModule>      pool_basins    { kModuleCacheCapacity };

    ModuleLru<CachedAdModule>          pool_ad        { kModuleCacheCapacity };

    ModuleLru<CachedSimpleContModule>  pool_ad_cont   { kModuleCacheCapacity };

    ModuleLru<CachedLyapAdModule>      pool_lyap_ad   { kModuleCacheCapacity };

    ModuleLru<CachedLyapAdModule>      pool_lyap_ad_cont { kModuleCacheCapacity };

    ModuleLru<CachedMetricsModule>     pool_metrics_ad{ kModuleCacheCapacity };

    ModuleLru<CachedFastSyncModule>    pool_fs_attr   { kModuleCacheCapacity };

    ModuleLru<CachedFastSyncModule>    pool_fs_grid   { kModuleCacheCapacity };

    ModuleLru<CachedOrderModule>       pool_order     { kModuleCacheCapacity };

    ModuleLru<CachedNetworkModule>     pool_network   { kModuleCacheCapacity };

    ModuleLru<CachedMetricsModule>     pool_metrics   { kModuleCacheCapacity };


    // Activates the module already built for this key, if the pool still holds it.
    template <class T>
    bool cache_activate(ModuleLru<T>& pool, const std::string& key, T& active) ;


    // Prewarm path: park the module in the pool WITHOUT touching the active slot -- a running
    // task reads that slot from its own thread and must not see it change under it.
    template <class T>
    void cache_store(ModuleLru<T>& pool, const T& fresh, const std::string& pinned) ;


    // Keys being compiled right now, so two threads asking for the same module (a prewarm and the
    // Run that overtakes it) do not both pay NVRTC: the second waits and takes the pooled result.
    std::set<std::string>            in_flight_keys;

    std::condition_variable_any      compile_done;


    // The shared body of every compile_*_if_needed. `build` does the slow NVRTC work and runs
    // WITHOUT the lock held; `activate` distinguishes a real Run (needs the module in the active
    // slot) from a prewarm (pool only).
    template <class T, class Build>
    bool compile_into(ModuleLru<T>& pool, const std::string& key, T& active,
                      bool activate, Build&& build, std::string& err) ;


    // Pins the PREVIOUS active entry while inserting: a running task may still hold its kernels,
    // and evicting it would unload code from under a live launch.
    template <class T>
    void cache_publish(ModuleLru<T>& pool, const T& fresh, T& active) ;


    template <class T>
    void drain_pool(ModuleLru<T>& pool) ;


    ~Impl() ;













    bool ensure_init(std::string& err) ;


    bool load_sources(std::string& err) ;


    // Copy of everything build_module reads, taken under cache_mu. NVRTC then runs unlocked:
    // load_sources() rewrites configCUDA.h whenever the peak settings change, and reading a
    // std::string while another thread assigns it is a race no matter how rare.
    struct SrcSnapshot {
        std::string tmpl, lib_cu, lib_cuh, macros_cuh, config_h, adaptive_cuh;
    };


    SrcSnapshot snapshot_sources(const std::string& tmpl) const ;


    // Функции модуля КРС при раздельной сборке. Шаблон держит тело каждой под своим
    // плейсхолдером; в библиотечной половине плейсхолдер заменяется вызовом внешней функции
    // модуля КРС. Кроме шага с постоянным h, это четыре функции адаптивного шага (см.
    // AdaptiveCode в codegen.hpp) — их плейсхолдеры есть только в адаптивных шаблонах.
    struct KrsFn { const char* ph; const char* name; const char* params; const char* call; };

    static constexpr KrsFn kKrsFns[] = {
        { "{{KRS_BODY}}", "krs_step", "numb* X, const numb* a, const numb h",
          "    krs_step(X, a, h);" },
        { "{{KRS_RHS_BODY}}", "krs_rhs", "const numb* X, const numb* a, numb* F",
          "    krs_rhs(X, a, F);" },
        { "{{KRS_EMB_BODY}}", "krs_emb",
          "const numb* X, const numb* F0, const numb* a, const numb h, numb* Y, numb* E, numb* F1, numb* W",
          "    krs_emb(X, F0, a, h, Y, E, F1, W);" },
        { "{{KRS_DPREP_BODY}}", "krs_dprep",
          "const numb* X, const numb* Y, const numb* F0, const numb* F1, const numb* a, const numb h, numb* W, numb* D",
          "    krs_dprep(X, Y, F0, F1, a, h, W, D);" },
        { "{{KRS_DEVAL_BODY}}", "krs_deval", "const numb* D, const numb th, numb* Yo",
          "    krs_deval(D, th, Yo);" },
    };

    static const KrsFn* krs_fn(const std::string& ph) ;


    // Кэш PTX библиотечной половины: ключ — всё, кроме КРС. Именно это и есть смысл затеи:
    // при смене схемы 1.2 МБ PTX библиотеки переиспользуются, а заново собирается только шаг.
    struct CachedLibPtx {
        std::string              key;
        std::string              ptx;
        std::vector<std::string> lowered;   // mangled-имена ядер: они из библиотечной половины
    };

    struct CachedKrsPtx {
        std::string key;
        std::string ptx;
    };


    // Свой мьютекс: build_module работает с ОТПУЩЕННЫМ cache_mu (иначе NVRTC сериализовался бы),
    // поэтому эти кэши защищаются отдельно. Вложенности нет — cache_mu здесь не берётся.
    std::mutex                  ptx_mu;

    std::set<std::string>       lib_in_flight;

    std::condition_variable     lib_ptx_done;

    ModuleLru<CachedLibPtx>     pool_lib_ptx{ 4 };
   // ~1.2 МБ на запись
    ModuleLru<CachedKrsPtx>     pool_krs_ptx{ 8 };
   // ~50 КБ на запись

    // Компиляция одной единицы трансляции в PTX. rdc=true — для половинок раздельной сборки.
    bool compile_ptx(const std::string& src, const char* src_name,
                     const std::vector<const char*>& headers_src,
                     const std::vector<const char*>& headers_name,
                     const std::vector<const char*>& name_exprs,
                     bool rdc, std::string& out_ptx, std::vector<std::string>& lowered,
                     std::string& err) ;


    // Библиотечная половина: тот же шаблон, но вместо тела шага — вызов внешнего krs_step.
    // Объявление вставляется сразу после #include "cudaLibrary.cuh" (он тянет numb из
    // configCUDA.h, а сам КРС определяется четырьмя строками ниже — структура одинакова у всех
    // 15 шаблонов).
    bool lib_ptx_for(const SrcSnapshot& snap, const char* src_name, const std::string& lib_key,
                     const std::vector<std::pair<std::string, std::string>>& subs,
                     const std::vector<const char*>& name_exprs,
                     CachedLibPtx& out, std::string& err) ;


    // Половина с шагом: отдельная единица трансляции на одном configCUDA.h (нужен и numb, и
    // ucmplx для комплексных схем).
    bool krs_ptx_for(const SrcSnapshot& snap,
                     const std::vector<std::pair<const KrsFn*, std::string>>& fns,
                     const std::string& amount_of_x, const std::string& krs_key,
                     std::string& out_ptx, std::string& err) ;


    // Линкует половинки в cubin и грузит модуль. Указатель из cuLinkComplete живёт до
    // cuLinkDestroy — модуль обязан загрузиться ДО разрушения линкера.
    bool link_and_load(const std::string& lib_ptx, const std::string& krs_ptx,
                       CUmodule& out_module, std::string& err) ;


    // build_module — общая часть всех compile_*_if_needed: подстановка плейсхолдеров в шаблон,
    // NVRTC-компиляция с едиными опциями, добыча mangled-имён и загрузка PTX. У вызывающих
    // различаются только шаблон, имя исходника и набор символов.
    // Ради этого всё и сведено: раньше блок был скопирован под каждый анализ, и когда --fmad стал
    // настройкой, флаг проставили в nvrtc_engine.cpp, а копии здесь остались на дефолте NVRTC —
    // карта и фазовый портрет считались разной арифметикой, и на фрактальной границе траектория
    // уходила в другой аттрактор (см. nvrtc_fmad_opt).
    // Требует выставленного контекста и уже загруженных источников (load_sources). При успехе
    // out_module загружен, а lowered содержит по одному имени на каждый вход name_exprs в том же
    // порядке. Символы, объявленные extern "C", в name_exprs передавать не нужно.
    bool build_module(const SrcSnapshot& snap,
                      const char* src_name,
                      const std::vector<std::pair<std::string, std::string>>& subs,
                      const std::vector<const char*>& name_exprs,
                      CUmodule& out_module,
                      std::vector<std::string>& lowered,
                      std::string& err) ;


    // Достаёт функцию из загруженного модуля. Отдельная обёртка — чтобы у
    // вызывающих не размножалось одинаковое сообщение об ошибке.
    bool module_fn(CUmodule mod, const std::string& name, CUfunction& out, std::string& err) ;


    bool compile_if_needed(const std::string& krs_body, int amountOfX,
                           int par_or_var, std::string& err, bool activate = true) ;


    // Адаптивный модуль свипов (см. CachedAdModule). Тела адаптивного шага входят в ключ:
    // от них зависит модуль КРС, как от krs_body.
    bool compile_ad_if_needed(const AdaptiveRequest& ad, const std::string& krs_body, int amountOfX,
                              int par_or_var, std::string& err, bool activate = true) ;


    // Модуль continuation БД 1D с адаптивным шагом (см. cached_ad_cont): тот же шаблон, но
    // из adaptive_part.cu в него попадает только ядро цепочки.
    bool compile_ad_cont_if_needed(const AdaptiveRequest& ad, const std::string& krs_body, int amountOfX,
                                   std::string& err, bool activate = true) ;


    // Модуль LLE/LS с адаптивным шагом (см. CachedLyapAdModule). Оба ядра в одном модуле.
    bool compile_lyap_ad_if_needed(const AdaptiveRequest& ad, const std::string& krs_body, int amountOfX,
                                   int par_or_var, std::string& err, bool activate = true) ;


    // Модуль continuation LLE/LS с адаптивным шагом (см. cached_lyap_ad_cont).
    // activate = false — прогрев: модуль только в пул. Активный слот прогрев трогать не вправе:
    // идущий расчёт берёт ядро из него уже после своей компиляции.
    bool compile_lyap_ad_cont_if_needed(const AdaptiveRequest& ad, const std::string& krs_body, int amountOfX,
                                        std::string& err, bool activate = true) ;


    // LLE (ls = false) / LS (ls = true) 1D с адаптивным шагом одной цепочкой точек: continuation
    // на GPU (одна нить, ucuda_lyap_chain) или любой режим на CPU (AdaptiveCpuModule — те же
    // функции раздела UCUDA_AD_LYAPUNOV, собранные cl.exe). out[nPts * NC] (NaN — разлёт),
    // stats[nPts * 4]. Возвращает false с текстом в err; отмена — cancelled = true.
    template <class Req>
    bool run_lyap_ad_seq(bool ls, const Req& req, std::vector<double>& out, std::vector<double>& stats,
                         std::string& err, bool& cancelled) ;


    // БД 1D с адаптивным шагом на CPU: AdaptiveCpuModule (cl.exe) собирает адаптивные тела,
    // kernels/adaptive_part.cu и PeakStream в DLL, её вход ucuda_cpu_ad_bif — построчная копия
    // calculateDiscreteModelPeaksAdCUDA (классика) и ...AdContCUDA (continuation). Раскладка
    // пиков, флагов и статистики — как у GPU-путей, результат собирается так же. Классика
    // раздаётся потокам по кускам точек (точки независимы, порядок результата не зависит
    // от раздачи); continuation — одна цепочка в этом потоке. Валидация общая с GPU-путями:
    // классика приходит из run_bif1d после проверок, continuation — из диспетчера.
    Bifurcation1DResult run_bif1d_cpu_ad(const Bifurcation1DRequest& req) ;


    LLE1DResult run_lle1d_ad_seq(const LLE1DRequest& req) ;


    LS1DResult run_ls1d_ad_seq(const LS1DRequest& req) ;


    // Ось настройки шага: rtol / atol / tol — только положительные значения (параметр
    // регулятора может быть любым). Пусто — всё в порядке.
    static std::string ad_axis_range_error(int kind, double lo, double hi) ;


    // Обе оси 2D — настройки шага, задевающие одно поле (rtol и tol, atol и tol, один параметр
    // регулятора): ядро применяет оси по порядку, и Y затирала бы X. Пусто — всё в порядке.
    static std::string ad_axes_overlap_error(int kx, int ky) ;


    // Модуль узлов шага (raw_nodes) у БД, бассейнов и метрик: без плотного выхода и с
    // размерностью-константой — циклы драйвера разворачиваются, и состояние нити может
    // жить в регистрах, а не в локальной памяти (UCUDA_AD_N в ucuda_adaptive.cuh).
    static constexpr const char* kAdNodesModulePrefix =
        "#define UCUDA_AD_NO_DENSE 1\n#define UCUDA_AD_NODES_KERNEL 1\n#define UCUDA_AD_STATIC_N 1\n";


    // Модуль узлов — для узлов шага при свипе по параметрам системы / НУ. Его ядрам настройки
    // шага приходят по значению (константный банк аргументов) — передавать ad_param_arg.
    // Свип по настройке шага (rtol, atol, параметр регулятора) идёт общим модулем.
    static bool ad_nodes_module(const AdaptiveRequest& ad) ;

    // Аргумент «настройки шага» для cuLaunchKernel: модуль узлов — сама структура, общий —
    // указатель на её копию на устройстве (d_ptr — адрес переменной с этим указателем).
    static void* ad_param_arg(const AdaptiveRequest& ad, UcudaAdaptParams** d_ptr) ;


    // Прогресс адаптивных ядер — по модельному времени: точка отчитывает столько
    // «единиц», сколько здесь, как бы ни менялся шаг.
    static constexpr size_t kAdProgressUnits = 4096;


    // Параметры адаптивного шага на устройстве: UcudaAdaptParams, коды осей и буфер
    // статистики (4 числа на точку чанка).
    struct AdDeviceArgs {
        UcudaAdaptParams* params = nullptr;
        int*              axis   = nullptr;
        numb*             stats  = nullptr;
        void release() {
            if (params) cudaFree(params);
            if (axis)   cudaFree(axis);
            if (stats)  cudaFree(stats);
            params = nullptr; axis = nullptr; stats = nullptr;
        }
        bool alloc(const AdaptiveRequest& ad, const int* kinds, size_t cells, std::string& err) {
            cudaError_t e = cudaMalloc((void**)&params, sizeof(UcudaAdaptParams));
            if (e == cudaSuccess) e = cudaMalloc((void**)&axis, 2 * sizeof(int));
            if (e == cudaSuccess) e = cudaMalloc((void**)&stats, (cells > 0 ? cells : 1) * 4 * sizeof(numb));
            if (e == cudaSuccess) e = cudaMemcpy(params, &ad.params, sizeof(UcudaAdaptParams), cudaMemcpyHostToDevice);
            if (e == cudaSuccess) e = cudaMemcpy(axis, kinds, 2 * sizeof(int), cudaMemcpyHostToDevice);
            if (e != cudaSuccess) { err = std::string("CUDA adaptive params: ") + cudaGetErrorString(e); release(); return false; }
            return true;
        }
    };


    // Запуск ядра LLE (ls = false) или LS (ls = true) с адаптивным шагом на один чанк cur
    // точек. Раскладка результата — как у LLEKernelCUDA / LSKernelCUDA, статистика — a.stats.
    CUresult launch_lyap_ad(bool ls, AdDeviceArgs& a, const AdaptiveRequest& ad, int nPts, int cur,
                            size_t calculated, int dimension, numb* d_ranges, int* d_idx, numb* d_ic,
                            int nIC, numb* d_vals, int nVals, double maxValue, numb* d_res, int logAxisMask,
                            double transientTime, double tMax, double NT, double eps, double vectorTransient,
                            RunSignals& sig, int progressStride) ;


    // run_bif1d — порт NonLinAnal::bifurcation1D из hostLibrary.cu. Идея: брать оригинальный код
    // почти как есть, чтобы при обновлениях NonLinAnal перенос был механическим diff → patch.
    // ОБЯЗАТЕЛЬНЫЕ изменения (помечены комментарием [ADAPT]):
    //   - <<<grid, block, shared>>>(...) → cuLaunchKernel(CUfunction, ...): наш kernel-модуль
    //     скомпилирован NVRTC'ом во время работы и недоступен по compile-time символу
    //   - gpuErrorCheck(...) у NonLinAnal зовёт exit(); здесь локальный BIF_CHECK, который пишет
    //     в res.error и возвращает результат с cleanup'ом
    //   - OUT_FILE_PATH приходит из req.csv_output_path (пусто = CSV не пишем)
    //   - ветка continuation_bif1D == 1 отключена: она использует host-side
    //     calculateDiscreteModel (default Lorenz), а не user's KRS
    //   - calculate_mean_med_freq / calculate_mean_and_variance отключены — их kernel-ы не входят
    //     в NVRTC-bundle
    //   - результат пишется в Bifurcation1DResult (память + CSV), не в файл
    Bifurcation1DResult run_bif1d(const Bifurcation1DRequest& req) ;


    // compile_lle_if_needed — отдельная компиляция для LLE-шаблона.
    // Ключ кэша = hash(krs_body + amountOfX + "lle"), чтобы PTX от bif1d
    // не путался с LLE даже при одной и той же KRS.
    bool compile_lle_if_needed(const std::string& krs_body, int amountOfX,
                               int par_or_var, std::string& err, bool activate = true) ;


    // run_lle_1d — порт NonLinAnal::LLE1D из hostLibrary.cu, той же diff-friendly стратегией, что
    // run_bif1d (оригинальные имена слева, req-имена справа). [ADAPT] — отличия от NonLinAnal:
    // <<<>>> → cuLaunchKernel (NVRTC-модуль); gpuErrorCheck → локальный LLE_CHECK с cleanup'ом,
    // без exit(); OUT_FILE_PATH — req.csv_output_path (пусто = без файла); результат пишется в
    // LLE1DResult (память + опц. CSV).
    LLE1DResult run_lle_1d(const LLE1DRequest& req) ;


    // compile_lle_2d_if_needed — отдельный PTX-кэш для LLE-2D. Структура та
    // же, что у compile_lle_if_needed; отличия только в исходнике шаблона
    // (lle2d.template.cu) и в маркере ключа кэша (":lle2d"). Сам kernel
    // (LLEKernelCUDA) — тот же, ловится по тому же имени.
    bool compile_lle_2d_if_needed(const std::string& krs_body, int amountOfX,
                                  int par_or_var, std::string& err, bool activate = true) ;


    // run_lle_2d — λ(p1, p2) на квадратной сетке, порт NonLinAnal::LLE2D на NVRTC-engine.
    // Отличия от LLE1D: ranges[4], indicesOfMutVars[2], dimension=2, chunking по ячейкам
    // (n_pts × n_pts может не влезть в память за раз), результат — плоский row-major массив.
    // par_or_var (compile-time): mixed_mode=true → 2; иначе оба свипа одного типа — 1 если обе оси
    // param, 0 если обе IC. Смешанные комбинации без mixed_mode не поддержаны (validator
    // отказывает — kernel-ветки под это нет).
    LLE2DResult run_lle_2d(const LLE2DRequest& req) ;


    // compile_ls_if_needed — третий шаблон. Ключ кэша помечен ":ls".
    bool compile_ls_if_needed(const std::string& krs_body, int amountOfX,
                              int par_or_var, std::string& err, bool activate = true) ;


    // run_ls_1d — порт NonLinAnal::LS1D той же стратегией, что run_bif1d / run_lle_1d.
    // Per-system результат — вектор длины amountOfX (один Ляпунов на переменную).
    // Memory budget на поток у LS значительно больше: shared = (3*N + 2*N^2 + nValues) *
    // sizeof(numb) * blockSize, и blockSize подбирается из 32K shared-лимита (как в NonLinAnal).
    LS1DResult run_ls_1d(const LS1DRequest& req) ;


    // compile_ls_2d_if_needed — отдельный шаблон ls2d.template.cu, тот же kernel
    // LSKernelCUDA. Ключ кэша помечен ":ls2d:" — изолирован от ":ls:" slot'а.
    bool compile_ls_2d_if_needed(const std::string& krs_body, int amountOfX,
                                 int par_or_var, std::string& err, bool activate = true) ;


    // run_ls_2d — спектр Ляпунова на сетке n_pts × n_pts: гибрид run_ls_1d (per-system буфер на N
    // экспонент) и run_lle_2d (swap_xy, chunking по ячейкам). Kernel возвращает N значений на
    // ячейку; D2H копирует cur_limiter * N doubles, host распаковывает по плоскостям —
    // values[k*n*n + iy*n + ix] = k-я экспонента в ячейке (ix, iy).
    LS2DResult run_ls_2d(const LS2DRequest& req) ;


    // compile_bif1d_cont_if_needed — отдельный модуль (single-thread sequential
    // continuation kernel + peakFinderCUDA). Cache key с суффиксом :cont.
    bool compile_bif1d_cont_if_needed(const std::string& krs_body, int amountOfX,
                                      std::string& err, bool activate = true) ;


    // compile_simple_cont_if_needed — общий компилятор для одноядерных
    // continuation-модулей (LLE и LS). От compile_bif1d_cont_if_needed
    // отличается только тем, что регистрировать mangled-имена не нужно:
    // единственный нужный символ объявлен extern "C".
    bool compile_simple_cont_if_needed(ModuleLru<CachedSimpleContModule>& pool,
                                       CachedSimpleContModule& slot,
                                       const std::string& tmpl,
                                       const char* kernel_name,
                                       const char* src_name,
                                       const char* key_suffix,
                                       const std::string& krs_body, int amountOfX,
                                       std::string& err) ;


    // run_lle1d_continuation_gpu / run_ls1d_continuation_gpu — GPU-двойники
    // CPU-веток (см. run_lle1d_cpu / run_ls1d_cpu). Тот же «костыль», что у
    // Bifurcation1D: single-thread kernel, потому что continuation — цепочка.
    // Медленнее CPU, нужны для сверки и как привычный способ считать на GPU.
    LLE1DResult run_lle1d_continuation_gpu(const LLE1DRequest& req) ;


    LS1DResult run_ls1d_continuation_gpu(const LS1DRequest& req) ;


    // run_dft1d_continuation_gpu — выделенное ядро (dft1d_cont.template.cu),
    // а не связка bifurcation1dContinuationKernel + DFT_custom: под h-свипом у
    // каждой точки своя длина блока, свой шаг и своё окно, чего DFT_custom
    // выразить не может (у него один sizeOfBlock/h/window на запуск).
    Dft1DResult run_dft1d_continuation_gpu(const Dft1DRequest& req) ;


    // run_dft1d_hsweep_gpu — классический (без continuation) свип по шагу h. Отдельно от
    // run_dft1d_classical, потому что связка calculateDiscreteModelCUDA + DFT_custom тут
    // неприменима: DFT_custom берёт одну длину блока, одно готовое окно и один шаг дискретизации
    // на весь запуск, а под h-свипом все три — per-point. Ядро dft1dHSweepKernel делает траекторию
    // и DFT в одном месте, поток на точку (в отличие от однопоточного dft1dContinuationKernel:
    // здесь точки независимы).
    // Буферы считаются по worst case — наименьшему h диапазона, дающему больше всего сэмплов; так
    // же поступают run_dft1d_cpu и run_dft1d_continuation_gpu. Chunking по nPtsLimiter — как в
    // run_dft1d_classical / run_bif1d.
    Dft1DResult run_dft1d_hsweep_gpu(const Dft1DRequest& req) ;


    // run_bif1d_continuation — single-thread sequential. Каждый параметр
    // стартует с конечного x[] предыдущего. Direction (forward/reverse)
    // обрабатывается в kernel'е. PeakFinderCUDA вызывается из того же модуля.
    Bifurcation1DResult run_bif1d_continuation(const Bifurcation1DRequest& req) ;


    // run_bif1d_continuation_ad — continuation с адаптивным шагом. Одна нить идёт по
    // точкам цепочки (calculateDiscreteModelPeaksAdContCUDA) и переносит из точки в точку
    // X, предложенный шаг и память регулятора; пики пишутся сразу в короткие строки, как
    // у адаптивного ядра обычного свипа, поэтому отдельного прохода peakFinderCUDA нет.
    Bifurcation1DResult run_bif1d_continuation_ad(const Bifurcation1DRequest& req) ;


    // run_dft_1d — 1D DFT (порт bifurcation_DFT_1D из hostLibrary.cu:4900-5315).
    // Диспетчер: continuation требует param-sweep (та же причина, что и у
    // run_bif1d) — делегирует в run_dft1d_continuation, иначе classical.
    Dft1DResult run_dft_1d(const Dft1DRequest& req) ;


    // Общая для classical/continuation: строит оконную функцию длиной sizeOfBlock. DFT_custom
    // принимает готовое окно аргументом и НЕ считает его сам. Формулы и нумерация типов — в
    // cpu_build_window выше (была отдельная копия тех же формул; после добавления Blackman и
    // Blackman-Harris держать две копии в одном файле смысла нет).
    static void build_window(std::vector<numb>& out, int sizeOfBlock, int window_type) ;


    // run_dft1d_classical — порт classical-ветки bifurcation_DFT_1D. Реюзает
    // cached.kernel_traj (calculateDiscreteModelCUDA, тот же PTX-кэш что и
    // run_bif1d) для генерации сырых траекторий, затем cached.kernel_dft
    // (DFT_custom) вместо peakFinderCUDA. Chunked по nPtsLimiter как run_bif1d.
    Dft1DResult run_dft1d_classical(const Dft1DRequest& req) ;


    // run_dft1d_continuation — реюзает bifurcation1dContinuationKernel как есть: он уже пишет
    // полную decimated-траекторию в d_data и флаги в d_amountOfPeaks для ВСЕГО nPts за один
    // монолитный запуск (kernels/bifurcation1d_cont.template.cu). Второй проход — DFT_custom вместо
    // peakFinderCUDA над теми же буферами. Монолитно, без chunking, как run_bif1d_continuation.
    Dft1DResult run_dft1d_continuation(const Dft1DRequest& req) ;


    // compile_bif2d_if_needed — шаблон bifurcation2d.template.cu, два kernel'а:
    // calculateDiscreteModelPeaksCUDA + dbscanCUDA.
    // Cache key: ":bif2d:" + par_or_var.
    bool compile_bif2d_if_needed(const std::string& krs_body, int amountOfX,
                                 int par_or_var, std::string& err, bool activate = true) ;


    // run_bif2d — порт NonLinAnal::bifurcation2D (hostLibrary.cu:898-1724).
    // Три kernel'а: calculateDiscreteModelCUDA → peakFinderCUDA → dbscanCUDA.
    // Результат: число DBSCAN-кластеров на ячейку = период системы.
    // [ADAPT]: те же адаптации, что у run_bif1d + run_lle_2d (см. их комментарии).
    Bifurcation2DResult run_bif2d(const Bifurcation2DRequest& req) ;


    // compile_metrics_if_needed — шаблон signal_metrics.template.cu, одно ядро
    // calculateDiscreteModelMetricsCUDA. Cache key: ":metrics:" + par_or_var.
    // minmax — min/max по интерполированным экстремумам (SIGM_MINMAX_INTERP, см. шаблон):
    // другой модуль, поэтому и в ключе, и в имени исходника (ключ библиотечной половины).
    bool compile_metrics_if_needed(const std::string& krs_body, int amountOfX,
                                   int par_or_var, std::string& err, bool activate = true,
                                   bool minmax = false) ;


    // Метрики с адаптивным шагом (см. cached_metrics_ad).
    bool compile_metrics_ad_if_needed(const AdaptiveRequest& ad, const std::string& krs_body, int amountOfX,
                                      int par_or_var, std::string& err, bool activate = true) ;


    // Раскладка осей свипа для метрик: par_or_var, индексы, диапазоны, h-ось и
    // транспонирование. 1D — как у run_bif1d, 2D — дословно логика run_bif2d
    // (смешанный свип X = param, Y = IC ядро умеет только наоборот, отсюда swap_xy).
    struct MetricsAxes {
        int    par_or_var = 1;
        int    idx_x = 0, idx_y = 0;
        double lo_x = 0, hi_x = 1, lo_y = 0, hi_y = 1;
        bool   swap_xy = false;
        int    hSweepAxis = -1;
        int    logAxisMask = 0;
    };


    static std::string metrics_axes(const SignalMetricsRequest& req, MetricsAxes& a) ;


    // run_signal_metrics — одно fused-ядро на чанк (интегрирование + суммы +
    // PeakStream), результат — SIGM_COUNT строк SoA на ячейку. Бюджет памяти как
    // у run_bif2d: траектории нет, на ячейку — выходные строки, флаг и (только
    // под медианную частоту) строка межпиковых интервалов.
    SignalMetricsResult run_signal_metrics(const SignalMetricsRequest& req) ;


    // GPU-continuation Metrics: одно однопоточное ядро на весь свип (точки
    // зависят друг от друга). Буфер интервалов — одна строка, ядро
    // переиспользует её от точки к точке.
    SignalMetricsResult run_metrics_continuation_gpu(const SignalMetricsRequest& req) ;


    // Continuation метрик с адаптивным шагом: signalMetricsContinuationAdKernel, одна нить
    // на всю цепочку. Устроено как run_metrics_continuation_gpu; h — шаг вывода, не свипается.
    // Метрики 1D с адаптивным шагом на CPU: AdaptiveCpuModule, вход ucuda_cpu_ad_metrics — построчные
    // копии calculateDiscreteModelMetricsAdCUDA (классика, кусками по потокам) и
    // signalMetricsContinuationAdKernel (continuation, цепочка в этом потоке). Раскладка и сборка
    // результата — как у GPU-путей (run_signal_metrics / run_metrics_continuation_ad). Валидация и
    // оси — вызывающего (run_signal_metrics, metrics_axes).
    SignalMetricsResult run_metrics_cpu_ad(const SignalMetricsRequest& req, const MetricsAxes& ax) ;


    SignalMetricsResult run_metrics_continuation_ad(const SignalMetricsRequest& req) ;


    // compile_basins_if_needed — отдельный шаблон basins.template.cu, регистрирует 5 kernel'ов:
    // calculateDiscreteModelCUDA, avgPeakFinderCUDA и три DBSCAN-kernel'а (cluster,
    // search_fixed_points, search_clear_points). Cache-ключ помечен `:basins`; par_or_var=0
    // захардкожен в шаблоне, поэтому ключа не требует — только хэш krs_body/amountOfX.
    bool compile_basins_if_needed(const std::string& krs_body, int amountOfX,
                                  std::string& err, bool activate = true) ;


    // run_basins — порт NonLinAnal::basinsOfAttraction + CUDA_dbscan host-loop (hostLibrary.cu).
    // Структура: 1) traj+avg_peak chunked loop по ячейкам; 2) host-DBSCAN — цикл по точкам, на
    // каждой итерации search_fixed/search_clear + cluster_kernel с расширением через
    // neighbors-список. Возвращает basin_idx (cluster IDs), avg_peaks, avg_intervals, helpful_array.
    BasinsResult run_basins(const BasinsRequest& req) ;


    // run_basins_recluster — DBSCAN-only прогон поверх кэшированных фич. Зеркалит фазу 2
    // (host-DBSCAN) из run_basins, но пропускает sim-фазу: d_avgPeaks/d_avgIntervals/d_helpfulArray
    // заливаются из request'а вместо повторного прогона avgPeakFinderCUDA. Мульты уже применены к
    // загруженным фичам (run_basins, фаза 1), поэтому здесь не нужны.
    BasinsReclusterResult run_basins_recluster(const BasinsReclusterRequest& req) ;


    // FastSynchro — два режима.

    // Один helper для обоих режимов — параметризован шаблоном и списком
    // kernel-symbols. Ключ кэша включает все substituted-параметры
    // (type_of_synch, error_estim, fs_error_trs), чтобы при их смене recompile
    // действительно случился.
    bool compile_fs_module(const std::string& src_template,
                           const std::string& tag, // ":fs_attr" или ":fs_grid"
                           int amountOfX,
                           const std::string& krs_body,
                           int type_of_synch_v, int error_estim_v, double fs_error_trs_v,
                           const std::vector<const char*>& expr_kernels,
                           ModuleLru<CachedFastSyncModule>& pool,
                           CachedFastSyncModule& slot,
                           std::string& err,
                           bool activate = true)
    ;


    bool compile_order_module(bool activate, int amountOfX, int amountOfValues,
                              const std::string& krs_body, const std::string& ref_body,
                              std::string& err)
    ;


    // prewarm_* — компиляция без расчёта. Ключи считаются ровно так же, как в соответствующем
    // run_*, иначе прогретый модуль не был бы найден. Ошибки не всплывают: не прогрелось — Run
    // скомпилирует сам и покажет ошибку уже там.
    // CPU-ветки адаптивного шага (БД 1D, метрики, LLE/LS) считают одной cl.exe-DLL с теми же
    // аргументами сборки: прогрев собирает её в дисковый кэш (build_cached_dll), Run её только
    // загружает. GPU-модули им не нужны.
    void prewarm_ad_cpu(const AdaptiveRequest& ad, int amountOfX) ;


    void prewarm_bif1d(const Bifurcation1DRequest& req) ;

    void prewarm_metrics(const SignalMetricsRequest& req) ;

    void prewarm_bif2d(const Bifurcation2DRequest& req) ;

    void prewarm_lle1d(const LLE1DRequest& req) ;

    void prewarm_lle2d(const LLE2DRequest& req) ;

    void prewarm_ls1d(const LS1DRequest& req) ;

    void prewarm_ls2d(const LS2DRequest& req) ;

    void prewarm_basins(const BasinsRequest& req) ;


    // Узлы одной оси. Лог-сетка требует строго положительных границ: по шагу
    // это всегда так, по параметру — нет, поэтому проверка, а не clamp.
    static bool order_axis_nodes(const OrderAxis& ax, const char* what,
                                 std::vector<double>& out, std::string& err)
    ;


    // Число грубых шагов ячейки — ТА ЖЕ формула, что в ядре. Хосту она нужна,
    // чтобы посчитать полную работу для прогресс-бара: при свипе по h она
    // отличается от узла к узлу на порядки, и «среднее по сетке» давало бы
    // бар, который стоит на месте, а потом прыгает в конец.
    static long long order_steps_for(double h, double tMax, bool snap) ;


    struct OrderDevBuf {
        void* p = nullptr;
        ~OrderDevBuf() { if (p) cudaFree(p); }
        bool alloc(size_t bytes, const char* what, std::string& err) {
            cudaError_t e = cudaMalloc(&p, bytes ? bytes : 1);
            if (e != cudaSuccess) { p = nullptr; err = std::string("cudaMalloc ") + what + ": " + cudaGetErrorString(e); return false; }
            return true;
        }
        template <class T> T* as() const { return (T*)p; }
    };


    OrderResult run_order(const OrderRequest& req) ;


    // Stability — область устойчивости схемы на двумерной задаче Дальквиста.
    //
    // Считать тут почти нечего: ячейка — это ДВА шага схемы (по одному от
    // каждого базисного вектора), поэтому вся карта 512x512 стоит столько же,
    // сколько один узел p(h) при t_max/h = 1000. Чанкование оставлено только
    // ради отмены и прогресса: экстраполяционная обёртка с десятком стадий на
    // мелкой сетке всё-таки набирает заметное время.
    StabilityResult run_stability(const StabilityRequest& req) ;


    // Performance — «время счёта vs достигнутая ошибка» по одномерной сетке.
    //
    // Два прохода. Первый — обычный run_order по той же оси: он даёт E1/E2,
    // статусы и фактический шаг узла и НЕ засекается. Второй — собственно
    // замер: на каждом узле warmup холостых запусков, затем repeats запусков
    // под cudaEvent'ами, стоящими вплотную вокруг cuLaunchKernel. Всё, что
    // запуском ядра не является — компиляция модуля, H2D значений, выделение
    // выходного буфера, D2H, — из измеряемого интервала вынесено; кэш модуля
    // к этому моменту уже прогрет order-проходом.
    PerfResult run_performance(const PerfRequest& req) ;


    // run_fastsync: dispatch по req.mode
    FastSyncResult run_fastsync(const FastSyncRequest& req) ;


    // -----------------------------------------------------------------------
    // Network — сеть связанных осцилляторов (kernels/network.template.cu).

    bool compile_network_module(bool activate, int amountOfX, int amountOfValues,
                                const std::string& krs_body, const std::string& coupling_body,
                                std::string& err)
    ;


    NetworkResult run_network(const NetworkRequest& req) ;
};
