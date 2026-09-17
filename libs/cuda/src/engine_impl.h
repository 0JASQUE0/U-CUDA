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

    // curand_kernel.h перехвачен inline-stub'ом в каждом template'е (kernels/*.cu), а
    // `#define CURAND_KERNEL_H_` блокирует реальный header; virtual header'а здесь больше нет —
    // иначе он повторно объявлял бы curandState_t.
    // src_configCUDA_h = peak_config_defines() + src_configCUDA_h_raw: сырой текст держим отдельно,
    // чтобы пересборка при смене Settings не лезла на диск.
    std::string src_adaptive_h;

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
        CUfunction  kernel = nullptr;   // orderEstimateKernel
    };

    CachedOrderModule cached_order;


    ~Impl() ;


    void release_fs_attr_module() ;

    void release_fs_grid_module() ;


    void release_order_module() ;


    void release_module() ;


    void release_lle_module() ;


    void release_lle_2d_module() ;


    void release_ls_module() ;


    void release_ls_2d_module() ;


    void release_simple_cont_module(CachedSimpleContModule& m) ;


    void release_cont_module() ;


    void release_bif2d_module() ;


    void release_basins_module() ;


    bool ensure_init(std::string& err) ;


    bool load_sources(std::string& err) ;


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
    bool build_module(const std::string& tmpl,
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
                           int par_or_var, std::string& err) ;


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
                               int par_or_var, std::string& err) ;


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
                                  int par_or_var, std::string& err) ;


    // run_lle_2d — λ(p1, p2) на квадратной сетке, порт NonLinAnal::LLE2D на NVRTC-engine.
    // Отличия от LLE1D: ranges[4], indicesOfMutVars[2], dimension=2, chunking по ячейкам
    // (n_pts × n_pts может не влезть в память за раз), результат — плоский row-major массив.
    // par_or_var (compile-time): mixed_mode=true → 2; иначе оба свипа одного типа — 1 если обе оси
    // param, 0 если обе IC. Смешанные комбинации без mixed_mode не поддержаны (validator
    // отказывает — kernel-ветки под это нет).
    LLE2DResult run_lle_2d(const LLE2DRequest& req) ;


    // compile_ls_if_needed — третий шаблон. Ключ кэша помечен ":ls".
    bool compile_ls_if_needed(const std::string& krs_body, int amountOfX,
                              int par_or_var, std::string& err) ;


    // run_ls_1d — порт NonLinAnal::LS1D той же стратегией, что run_bif1d / run_lle_1d.
    // Per-system результат — вектор длины amountOfX (один Ляпунов на переменную).
    // Memory budget на поток у LS значительно больше: shared = (3*N + 2*N^2 + nValues) *
    // sizeof(numb) * blockSize, и blockSize подбирается из 32K shared-лимита (как в NonLinAnal).
    LS1DResult run_ls_1d(const LS1DRequest& req) ;


    // compile_ls_2d_if_needed — отдельный шаблон ls2d.template.cu, тот же kernel
    // LSKernelCUDA. Ключ кэша помечен ":ls2d:" — изолирован от ":ls:" slot'а.
    bool compile_ls_2d_if_needed(const std::string& krs_body, int amountOfX,
                                 int par_or_var, std::string& err) ;


    // run_ls_2d — спектр Ляпунова на сетке n_pts × n_pts: гибрид run_ls_1d (per-system буфер на N
    // экспонент) и run_lle_2d (swap_xy, chunking по ячейкам). Kernel возвращает N значений на
    // ячейку; D2H копирует cur_limiter * N doubles, host распаковывает по плоскостям —
    // values[k*n*n + iy*n + ix] = k-я экспонента в ячейке (ix, iy).
    LS2DResult run_ls_2d(const LS2DRequest& req) ;


    // compile_bif1d_cont_if_needed — отдельный модуль (single-thread sequential
    // continuation kernel + peakFinderCUDA). Cache key с суффиксом :cont.
    bool compile_bif1d_cont_if_needed(const std::string& krs_body, int amountOfX,
                                      std::string& err) ;


    // compile_simple_cont_if_needed — общий компилятор для одноядерных
    // continuation-модулей (LLE и LS). От compile_bif1d_cont_if_needed
    // отличается только тем, что регистрировать mangled-имена не нужно:
    // единственный нужный символ объявлен extern "C".
    bool compile_simple_cont_if_needed(CachedSimpleContModule& slot,
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
                                 int par_or_var, std::string& err) ;


    // run_bif2d — порт NonLinAnal::bifurcation2D (hostLibrary.cu:898-1724).
    // Три kernel'а: calculateDiscreteModelCUDA → peakFinderCUDA → dbscanCUDA.
    // Результат: число DBSCAN-кластеров на ячейку = период системы.
    // [ADAPT]: те же адаптации, что у run_bif1d + run_lle_2d (см. их комментарии).
    Bifurcation2DResult run_bif2d(const Bifurcation2DRequest& req) ;


    // compile_basins_if_needed — отдельный шаблон basins.template.cu, регистрирует 5 kernel'ов:
    // calculateDiscreteModelCUDA, avgPeakFinderCUDA и три DBSCAN-kernel'а (cluster,
    // search_fixed_points, search_clear_points). Cache-ключ помечен `:basins`; par_or_var=0
    // захардкожен в шаблоне, поэтому ключа не требует — только хэш krs_body/amountOfX.
    bool compile_basins_if_needed(const std::string& krs_body, int amountOfX,
                                  std::string& err) ;


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
                           CachedFastSyncModule& slot,
                           std::string& err)
    ;


    bool compile_order_module(int amountOfX, int amountOfValues,
                              const std::string& krs_body, std::string& err)
    ;


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


    // run_fastsync: dispatch по req.mode
    FastSyncResult run_fastsync(const FastSyncRequest& req) ;
};
