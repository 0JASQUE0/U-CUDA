#include "engine_impl.h"



    // run_basins — порт NonLinAnal::basinsOfAttraction + CUDA_dbscan host-loop (hostLibrary.cu).
    // Структура: 1) traj+avg_peak chunked loop по ячейкам; 2) host-DBSCAN — цикл по точкам, на
    // каждой итерации search_fixed/search_clear + cluster_kernel с расширением через
    // neighbors-список. Возвращает basin_idx (cluster IDs), avg_peaks, avg_intervals, helpful_array.
    BasinsResult ParametricEngine::Impl::run_basins(const BasinsRequest& req) {
        BasinsResult res;
        auto fail = [&](const std::string& msg) -> BasinsResult& { res.error = msg; return res; };
        if (req.adaptive.enabled && !req.adaptive.setup_error.empty()) return fail(req.adaptive.setup_error);

        if (req.krs_body.empty())                                  return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)   return fail("amountOfX out of the allowed range");
        if ((int)req.initial_conditions.size() != req.amountOfX)   return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)      return fail("too many base_values");
        if (req.axis_x_var < 0 || req.axis_x_var >= req.amountOfX) return fail("axis_x_var out of range");
        if (req.axis_y_var < 0 || req.axis_y_var >= req.amountOfX) return fail("axis_y_var out of range");
        if (req.axis_x_var == req.axis_y_var)                      return fail("axis_x_var == axis_y_var (pick different variables)");
        if (req.writable_var < -1 || req.writable_var >= req.amountOfX) return fail("writable_var out of range");
        if (req.n_pts <= 0)         return fail("n_pts must be > 0");
        if (req.h <= 0.0)           return fail("h must be > 0");
        if (req.t_max <= 0.0)       return fail("t_max must be > 0");
        if (req.transient_time < 0) return fail("transient_time must be >= 0");
        if (req.pre_scaller <= 0)   return fail("pre_scaller must be > 0");
        if (req.eps_dbscan <= 0.0)  return fail("eps_dbscan must be > 0");

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        const bool ad = req.adaptive.enabled;   // адаптивный шаг: ядра из адаптивного модуля
        if (ad) {
            if (!compile_ad_if_needed(req.adaptive, req.krs_body, req.amountOfX, 0, err)) return fail(err);
        }
        else if (!compile_basins_if_needed(req.krs_body, req.amountOfX, err)) return fail(err);
        const CUfunction k_cdbscan      = ad ? cached_ad.kernel_cdbscan      : cached_basins.kernel_dbscan;
        const CUfunction k_search_fixed = ad ? cached_ad.kernel_search_fixed : cached_basins.kernel_search_fixed;
        const CUfunction k_search_clear = ad ? cached_ad.kernel_search_clear : cached_basins.kernel_search_clear;

        const int    nPts                       = req.n_pts;
        const double h                          = req.h;
        const int    amountOfInitialConditions  = req.amountOfX;
        const std::vector<numb> ic_staged_       = to_numb(req.initial_conditions);
        const numb*   initialConditions         = ic_staged_.data();
        numb   ranges[4]                        = { (numb)req.axis_x_lo, (numb)req.axis_x_hi,
                                                    (numb)req.axis_y_lo, (numb)req.axis_y_hi };
        int    indicesOfMutVars[2]              = { req.axis_x_var, req.axis_y_var };
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const double tMax                       = req.t_max;
        const std::vector<numb> values_staged_   = to_numb(req.base_values);
        const numb*   values                    = values_staged_.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const int    preScaller                 = req.pre_scaller;
        const double eps_dbscan                 = req.eps_dbscan;
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        // blockSize_setup здесь больше нет: ширина запуска берётся из настройки
        // (launch_block_size), а полом бюджета памяти у бассейнов он не служил.
        constexpr int set_precision   = 15;

        int amountOfPointsInBlock = (int)(tMax / h / preScaller);
        size_t amountOfPointsForSkip = steps_from_time_size_t(transientTime, h);

        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0");

        size_t total_cells = (size_t)nPts * (size_t)nPts;

        // По памяти чанк ничем не ограничен: буферы, росшие вместе с ним,
        // ушли вместе с траекторией, а выходные массивы выделяются на всю сетку.
        size_t nPtsLimiter = total_cells;

        size_t originalNPtsLimiter = nPtsLimiter;

        numb* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        numb* d_initialConditions = nullptr;
        numb* d_values            = nullptr;
        int*    d_amountOfPeaks     = nullptr;
        int*    d_helpfulArray      = nullptr;
        int*    d_dbscanResult      = nullptr;
        RunSignals sig;   // прогресс и отмена в mapped-памяти
        numb* d_avgPeaks          = nullptr;
        numb* d_avgIntervals      = nullptr;
        int*    d_amountOfNeighbors = nullptr;
        int*    d_neighbors         = nullptr;
        int*    d_clearIdx          = nullptr;
        AdDeviceArgs adArgs;                  // адаптивный шаг: параметры, оси, статистика

        auto cleanup = [&]() {
            adArgs.release();
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_amountOfPeaks)     cudaFree(d_amountOfPeaks);
            if (d_helpfulArray)      cudaFree(d_helpfulArray);
            if (d_dbscanResult)      cudaFree(d_dbscanResult);
            sig.release();
            if (d_avgPeaks)          cudaFree(d_avgPeaks);
            if (d_avgIntervals)      cudaFree(d_avgIntervals);
            if (d_amountOfNeighbors) cudaFree(d_amountOfNeighbors);
            if (d_neighbors)         cudaFree(d_neighbors);
            if (d_clearIdx)          cudaFree(d_clearIdx);
        };

        #define BAS_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define BAS_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)
        #define BAS_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        BAS_CHECK(cudaMalloc((void**)&d_ranges,            4 * sizeof(numb)),                                           "cudaMalloc d_ranges");
        BAS_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  2 * sizeof(int)),                                              "cudaMalloc d_indicesOfMutVars");
        BAS_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)amountOfInitialConditions * sizeof(numb)),           "cudaMalloc d_initialConditions");
        BAS_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(numb)),                      "cudaMalloc d_values");
        BAS_CHECK(cudaMalloc((void**)&d_amountOfPeaks,     nPtsLimiter * sizeof(int)),                                    "cudaMalloc d_amountOfPeaks");
        BAS_CHECK(cudaMalloc((void**)&d_helpfulArray,      total_cells * sizeof(int)),                                    "cudaMalloc d_helpfulArray");
        BAS_CHECK(cudaMalloc((void**)&d_dbscanResult,      total_cells * sizeof(int)),                                    "cudaMalloc d_dbscanResult");
        if (!sig.alloc(res.error)) { cleanup(); return res; }
        if (ad) {
            const int kinds[2] = { kAdAxisSystem, kAdAxisSystem };
            if (!adArgs.alloc(req.adaptive, kinds, total_cells, res.error)) { cleanup(); return res; }
        }
        const size_t stepsPerCell   = ad ? kAdProgressUnits
                                         : amountOfPointsForSkip + (size_t)amountOfPointsInBlock;
        const int    progressStride = progress_stride_for(stepsPerCell);
        const double ticksPerCell   = (double)(stepsPerCell / (size_t)progressStride);
        const double ticksTotal     = (double)total_cells * ticksPerCell;
        BAS_CHECK(cudaMalloc((void**)&d_avgPeaks,          total_cells * sizeof(numb)),                                 "cudaMalloc d_avgPeaks");
        BAS_CHECK(cudaMalloc((void**)&d_avgIntervals,      total_cells * sizeof(numb)),                                 "cudaMalloc d_avgIntervals");
        BAS_CHECK(cudaMalloc((void**)&d_amountOfNeighbors, sizeof(int)),                                                  "cudaMalloc d_amountOfNeighbors");
        BAS_CHECK(cudaMalloc((void**)&d_neighbors,         total_cells * sizeof(int)),                                    "cudaMalloc d_neighbors");
        BAS_CHECK(cudaMalloc((void**)&d_clearIdx,          sizeof(int)),                                                  "cudaMalloc d_clearIdx");

        BAS_CHECK(cudaMemcpy(d_ranges,            ranges,            4 * sizeof(numb),                                  cudaMemcpyHostToDevice), "memcpy d_ranges");
        BAS_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,  2 * sizeof(int),                                     cudaMemcpyHostToDevice), "memcpy d_indices");
        BAS_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)amountOfInitialConditions * sizeof(numb),  cudaMemcpyHostToDevice), "memcpy d_ic");
        BAS_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(numb),             cudaMemcpyHostToDevice), "memcpy d_values");
        BAS_CHECK(cudaMemset(d_dbscanResult, 0, total_cells * sizeof(int)),  "memset d_dbscanResult");
        BAS_CHECK(cudaMemset(d_helpfulArray, 0, total_cells * sizeof(int)),  "memset d_helpfulArray");
        BAS_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)total_cells / (double)nPtsLimiter);

        // Snapshot CSV-relevant fields for GUI right-click export.
        res.snapshot.values.assign(values, values + amountOfValues);
        res.snapshot.initial_conditions.assign(initialConditions,
            initialConditions + amountOfInitialConditions);
        res.snapshot.tMax          = tMax;
        res.snapshot.transientTime = transientTime;
        res.snapshot.h             = h;
        res.snapshot.gpu_fmad      = get_nvrtc_fmad();
        res.snapshot.preScaller    = preScaller;
        res.snapshot.eps_dbscan    = eps_dbscan;
        res.snapshot.writableVar   = req.writable_var;
        res.snapshot.axis_x_var    = req.axis_x_var;
        res.snapshot.axis_y_var    = req.axis_y_var;
        res.snapshot.axis_x_lo     = req.axis_x_lo;
        res.snapshot.axis_x_hi     = req.axis_x_hi;
        res.snapshot.axis_y_lo     = req.axis_y_lo;
        res.snapshot.axis_y_hi     = req.axis_y_hi;
        res.snapshot.n_pts         = nPts;
        res.snapshot.feature1      = req.feature1;
        res.snapshot.feature2      = req.feature2;
        res.snapshot.mult1         = (double)req.mult1;
        res.snapshot.mult2         = (double)req.mult2;

        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            data_export::write_basins_config(cfg, res.snapshot);
        }

        // Pre-write the 2-line ranges header into each of the 4 data files.
        // Body is appended after compute (see the post-loop block below).
        if (!OUT_FILE_PATH.empty()) {
            auto write_ranges = [&](const std::string& path) {
                std::ofstream o(path);
                data_export::write_basins_ranges(o, res.snapshot);
            };
            write_ranges(OUT_FILE_PATH);
            write_ranges(OUT_FILE_PATH + "_1.csv");
            write_ranges(OUT_FILE_PATH + "_2.csv");
            write_ranges(OUT_FILE_PATH + "_3.csv");
        }

        // 1. Цикл по chunk'ам: traj-kernel + avg-peak-kernel
        // Two-phase progress: phase 1 here (sim), phase 2 (DBSCAN) below. Each
        // phase reports its own 0..1 fraction; GUI shows the phase label.
        if (req.progress_phase) req.progress_phase->store(1, std::memory_order_relaxed);
        if (req.progress)       req.progress->store(0.0f, std::memory_order_relaxed);
        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            BAS_CANCEL_CHECK();
            // Тик-шкала, а не доля чанков: иначе на стыке бар дёргается назад (см. wait_with_signals).
            if (req.progress) req.progress->store((float)((double)(originalNPtsLimiter * iter) * ticksPerCell / ticksTotal), std::memory_order_relaxed);
            size_t cur_limiter = originalNPtsLimiter;
            if (iter == amountOfIteration - 1)
                cur_limiter = total_cells - (originalNPtsLimiter * iter);

            // blockSize: ceil(48K / ((N + nValues) * sizeof(numb))), cap — настройка
            // (Settings -> GPU launch); формула остаётся вторым потолком, см. run_lle_1d.
            const size_t sharedPerThread = (size_t)(amountOfInitialConditions + amountOfValues) * sizeof(numb);
            int blockSize = (int)std::ceil((1024.0 * 48.0) / (double)sharedPerThread);
            if (blockSize < 1)                blockSize = 1;
            const int blockSizeCap = launch_block_size(sharedPerThread);
            if (blockSize > blockSizeCap)     blockSize = blockSizeCap;
            int gridSize = (int)((cur_limiter + blockSize - 1) / blockSize);

            // calculateDiscreteModelAvgPeaksCUDA: интегрирование и фичи в одном ядре.
            // h-свипа и лог-осей у бассейнов нет в запросе вовсе, ядро зашивает их
            // константами -1 / 0 — ровно тем, что сюда передавалась раздельная пара.
            int    nPts_arg                  = nPts;
            int    nPtsLimiter_int           = (int)cur_limiter;
            size_t amountOfCalculatedPoints  = iter * originalNPtsLimiter;
            size_t amountOfPointsForSkip_s   = (size_t)amountOfPointsForSkip;
            int    dimension_arg             = 2;
            numb h_arg                     = h;
            int    amountOfIC_int            = amountOfInitialConditions;
            int    amountOfValues_int        = amountOfValues;
            size_t amountOfIterations_arg    = (size_t)amountOfPointsInBlock;
            int    preScaller_int            = preScaller;
            int    writableVar_int           = req.writable_var;
            numb maxValue_arg              = maxValue;
            bool   par_or_var_arg            = false;   // compile-time par_or_var=0 в шаблоне
            numb transientTime_arg         = transientTime;
            numb tMax_arg                  = tMax;
            int  feature1_int = req.feature1;
            int  feature2_int = req.feature2;
            numb mult1_v      = req.mult1;
            numb mult2_v      = req.mult2;
            // Сигналы пока выключены: прогресс и Cancel у бассейнов ещё по чанкам.
            int*   d_cancel_arg              = sig.cancelArg();
            int*   d_progress_arg            = sig.progressArg();
            int    progressStride_arg        = progressStride;
            sig.resetTicks();

            // Offset-указатели: чанк пишет в свою часть общей сетки.
            int*  d_helpful_chunk    = d_helpfulArray  + iter * originalNPtsLimiter;
            numb* d_avg_peak_chunk   = d_avgPeaks      + iter * originalNPtsLimiter;
            numb* d_avg_interv_chunk = d_avgIntervals  + iter * originalNPtsLimiter;

            void* args_basins[] = {
                &nPts_arg, &nPtsLimiter_int, &amountOfCalculatedPoints,
                &amountOfPointsForSkip_s, &dimension_arg, &d_ranges, &h_arg,
                &d_indicesOfMutVars, &d_initialConditions, &amountOfIC_int,
                &d_values, &amountOfValues_int, &amountOfIterations_arg,
                &preScaller_int, &writableVar_int, &maxValue_arg,
                &d_avg_peak_chunk, &d_avg_interv_chunk, &d_helpful_chunk,
                &par_or_var_arg, &transientTime_arg, &tMax_arg,
                &feature1_int, &feature2_int, &mult1_v, &mult2_v,
                &d_cancel_arg, &d_progress_arg, &progressStride_arg
            };
            unsigned int shared_traj = (unsigned int)(ucuda_shared_stride(amountOfInitialConditions, amountOfValues) * sizeof(numb) * blockSize);
            if (ad) {
                UcudaAdaptParams* d_adp_arg = adArgs.params;
                int*   d_axis_arg        = adArgs.axis;
                numb   tolRatio_arg      = (numb)req.adaptive.tol_ratio;
                numb   dtOut_arg         = h;
                int    raw_arg           = req.adaptive.raw_nodes ? 1 : 0;
                int    interp_arg        = req.adaptive.peak_interp;
                size_t progressUnits_arg = kAdProgressUnits;
                numb*  d_adStats_arg     = adArgs.stats + iter * originalNPtsLimiter * 4;
                void* args_ad[] = {
                    &nPts_arg, &nPtsLimiter_int, &amountOfCalculatedPoints, &dimension_arg,
                    &d_ranges, &d_indicesOfMutVars, &d_initialConditions, &amountOfIC_int,
                    &d_values, &amountOfValues_int, &writableVar_int, &maxValue_arg,
                    &d_avg_peak_chunk, &d_avg_interv_chunk, &d_helpful_chunk,
                    &feature1_int, &feature2_int, &mult1_v, &mult2_v,
                    ad_param_arg(req.adaptive, &d_adp_arg), &d_axis_arg, &tolRatio_arg, &transientTime_arg, &tMax_arg, &dtOut_arg,
                    &preScaller_int, &amountOfIterations_arg, &raw_arg, &interp_arg,
                    &d_cancel_arg, &d_progress_arg, &progressStride_arg,
                    &progressUnits_arg, &d_adStats_arg
                };
                BAS_CHECK_CU(cuLaunchKernel(cached_ad.kernel_avg,
                                            gridSize, 1, 1, blockSize, 1, 1,
                                            shared_traj, nullptr, args_ad, nullptr),
                             "cuLaunchKernel(basins adaptive)");
            }
            else
            BAS_CHECK_CU(cuLaunchKernel(cached_basins.kernel_fused,
                                        gridSize, 1, 1, blockSize, 1, 1,
                                        shared_traj, nullptr, args_basins, nullptr),
                         "cuLaunchKernel(basins traj+features)");
            if (!wait_with_signals(0, sig, req.cancel, req.progress,
                                   (double)(originalNPtsLimiter * iter) * ticksPerCell,
                                   ticksTotal, res.error)) { cleanup(); return res; }
            BAS_CHECK(cudaDeviceSynchronize(), "sync after traj+features");
            BAS_CANCEL_CHECK();
        }

        if (ad) {
            std::vector<numb> h_adStats(total_cells * 4);
            BAS_CHECK(cudaMemcpy(h_adStats.data(), adArgs.stats, total_cells * 4 * sizeof(numb),
                                 cudaMemcpyDeviceToHost), "memcpy adaptive stats");
            res.ad_stats.assign(h_adStats.begin(), h_adStats.end());
        }

        // 2. Host-DBSCAN: порт hostLibrary.cu:3066 (CUDA_dbscan).
        // Динамической shared у dbscan-ядер нет — настройка идёт как есть.
        int blockSize_db = launch_block_size(0);
        int gridSize_db  = (int)((total_cells + blockSize_db - 1) / blockSize_db);
        int amountOfData_int = (int)total_cells;

        int amountOfClusters         = 0;
        int amountOfNegativeClusters = 0;
        std::vector<int> h_neighbors(total_cells, 0);
        int h_amountOfNeighbors      = 0;

        // Прогресс DBSCAN-фазы: внешний цикл идёт ОДИН РАЗ НА КЛАСТЕР (с ранним выходом, когда все
        // ячейки классифицированы), поэтому main_iter — плохая метрика: он доходит до ~10 против
        // миллионов ячеек. Вместо него после расширения каждого кластера считаем классифицированные
        // ячейки в d_dbscanResult и репортим долю. Троттлинг >200 мс между сканами держит накладные
        // расходы ограниченными на больших сетках (скан ~4 МБ D2H + линейный проход на 1М ячеек,
        // ~3-5 мс; с троттлингом это максимум ~5 сканов в секунду).
        std::vector<int> h_dbscan_check(total_cells, 0);
        auto last_progress_scan = std::chrono::steady_clock::now() - std::chrono::seconds(1);

        // Phase 2: DBSCAN expansion. Reset bar to 0..1, bump phase to 2.
        if (req.progress_phase) req.progress_phase->store(2, std::memory_order_relaxed);
        if (req.progress)       req.progress->store(0.0f, std::memory_order_relaxed);
        for (size_t main_iter = 0; main_iter < total_cells; ++main_iter) {
            BAS_CANCEL_CHECK();
            int clearIdx_init = -1;
            BAS_CHECK(cudaMemcpy(d_clearIdx, &clearIdx_init, sizeof(int), cudaMemcpyHostToDevice), "memcpy d_clearIdx init");

            // search_fixed_points: проверяет, остались ли -1-точки без cluster'а.
            void* args_search[] = { &d_avgPeaks, &d_avgIntervals, &d_helpfulArray, &d_dbscanResult,
                                    &amountOfData_int, &d_clearIdx };
            BAS_CHECK_CU(cuLaunchKernel(k_search_fixed,
                                        gridSize_db, 1, 1, blockSize_db, 1, 1,
                                        0, nullptr, args_search, nullptr),
                         "cuLaunchKernel(search_fixed)");
            BAS_CHECK(cudaDeviceSynchronize(), "sync search_fixed");

            int clearIdx = -1;
            BAS_CHECK(cudaMemcpy(&clearIdx, d_clearIdx, sizeof(int), cudaMemcpyDeviceToHost), "memcpy clearIdx D2H");

            int resultClusters = 0;
            if (clearIdx == -1) {
                // FP-точек не осталось — search_clear_points (Osc).
                BAS_CHECK_CU(cuLaunchKernel(k_search_clear,
                                            gridSize_db, 1, 1, blockSize_db, 1, 1,
                                            0, nullptr, args_search, nullptr),
                             "cuLaunchKernel(search_clear)");
                BAS_CHECK(cudaDeviceSynchronize(), "sync search_clear");
                BAS_CHECK(cudaMemcpy(&clearIdx, d_clearIdx, sizeof(int), cudaMemcpyDeviceToHost), "memcpy clearIdx D2H 2");
                // Все ячейки классифицированы — нового seed'а нет, останавливаемся.
                // NB: инкремент идёт ПОСЛЕ проверки clearIdx. Референс NonLinAnal инкрементирует до
                // неё и сообщает на один кластер больше, но он возвращает void и счётчик никто не
                // читает; мы отдаём n_clusters в UI, где off-by-one был виден как «4 кластера» при
                // трёх реальных.
                if (clearIdx == -1) break;
                ++amountOfClusters;
                resultClusters = amountOfClusters;
            } else {
                --amountOfNegativeClusters;
                resultClusters = amountOfNegativeClusters;
            }

            // Reset amountOfNeighbors на каждой итерации.
            h_amountOfNeighbors = 0;
            BAS_CHECK(cudaMemcpy(d_amountOfNeighbors, &h_amountOfNeighbors, sizeof(int), cudaMemcpyHostToDevice), "memcpy d_amountOfNeighbors=0");

            // CUDA_dbscan_kernel — расширение cluster'а от clearIdx.
            numb eps_arg = eps_dbscan;
            void* args_db[] = {
                &d_avgPeaks, &d_avgIntervals, &d_dbscanResult,
                &amountOfData_int, &eps_arg, &resultClusters,
                &d_amountOfNeighbors, &d_neighbors, &clearIdx, &d_helpfulArray
            };
            BAS_CHECK_CU(cuLaunchKernel(k_cdbscan,
                                        gridSize_db, 1, 1, blockSize_db, 1, 1,
                                        0, nullptr, args_db, nullptr),
                         "cuLaunchKernel(dbscan expand init)");
            BAS_CHECK(cudaDeviceSynchronize(), "sync dbscan init");

            BAS_CHECK(cudaMemcpy(&h_amountOfNeighbors, d_amountOfNeighbors, sizeof(int), cudaMemcpyDeviceToHost), "memcpy amountOfNeighbors D2H");
            if (h_amountOfNeighbors > 0)
                BAS_CHECK(cudaMemcpy(h_neighbors.data(), d_neighbors, (size_t)h_amountOfNeighbors * sizeof(int), cudaMemcpyDeviceToHost),
                          "memcpy neighbors D2H");

            // Цикл расширения cluster'а по всем найденным соседям.
            for (size_t ni = 0; ni < (size_t)h_amountOfNeighbors; ++ni) {
                int neighbor_idx = h_neighbors[ni];
                void* args_db2[] = {
                    &d_avgPeaks, &d_avgIntervals, &d_dbscanResult,
                    &amountOfData_int, &eps_arg, &resultClusters,
                    &d_amountOfNeighbors, &d_neighbors, &neighbor_idx, &d_helpfulArray
                };
                BAS_CHECK_CU(cuLaunchKernel(k_cdbscan,
                                            gridSize_db, 1, 1, blockSize_db, 1, 1,
                                            0, nullptr, args_db2, nullptr),
                             "cuLaunchKernel(dbscan expand neighbor)");
                BAS_CHECK(cudaDeviceSynchronize(), "sync dbscan neighbor");

                BAS_CHECK(cudaMemcpy(&h_amountOfNeighbors, d_amountOfNeighbors, sizeof(int), cudaMemcpyDeviceToHost), "memcpy amountOfNeighbors loop");
                if (h_amountOfNeighbors > 0)
                    BAS_CHECK(cudaMemcpy(h_neighbors.data(), d_neighbors, (size_t)h_amountOfNeighbors * sizeof(int), cudaMemcpyDeviceToHost),
                              "memcpy neighbors loop");
            }

            // Report progress by counting classified cells. Throttled so the
            // D2H+scan happens at most ~5x/sec regardless of cluster count.
            if (req.progress) {
                auto now = std::chrono::steady_clock::now();
                if (now - last_progress_scan >= std::chrono::milliseconds(200)) {
                    last_progress_scan = now;
                    BAS_CHECK(cudaMemcpy(h_dbscan_check.data(), d_dbscanResult,
                                         total_cells * sizeof(int), cudaMemcpyDeviceToHost),
                              "memcpy dbscanResult D2H (progress)");
                    size_t classified = 0;
                    for (size_t k = 0; k < total_cells; ++k)
                        if (h_dbscan_check[k] != 0) ++classified;
                    req.progress->store(float(classified) / float(total_cells),
                                        std::memory_order_relaxed);
                }
            }
        }

        // 3. D2H результаты
        res.n_pts       = nPts;
        res.axis_x_lo   = req.axis_x_lo;
        res.axis_x_hi   = req.axis_x_hi;
        res.axis_y_lo   = req.axis_y_lo;
        res.axis_y_hi   = req.axis_y_hi;
        res.axis_x_var  = req.axis_x_var;
        res.axis_y_var  = req.axis_y_var;
        res.basin_idx.assign(total_cells, 0);
        res.avg_peaks.assign(total_cells, 0.0);
        res.avg_intervals.assign(total_cells, 0.0);
        res.helpful_array.assign(total_cells, 0);

        BAS_CHECK(cudaMemcpy(res.basin_idx.data(),     d_dbscanResult, total_cells * sizeof(int),    cudaMemcpyDeviceToHost), "memcpy basin_idx");
        {   // numb на устройстве -> double в Result.
            std::vector<numb> ap_(total_cells), ai_(total_cells);
            BAS_CHECK(cudaMemcpy(ap_.data(), d_avgPeaks,     total_cells * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy avg_peaks");
            BAS_CHECK(cudaMemcpy(ai_.data(), d_avgIntervals, total_cells * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy avg_intervals");
            res.avg_peaks.assign(ap_.begin(), ap_.end());
            res.avg_intervals.assign(ai_.begin(), ai_.end());
        }
        BAS_CHECK(cudaMemcpy(res.helpful_array.data(), d_helpfulArray, total_cells * sizeof(int),    cudaMemcpyDeviceToHost), "memcpy helpful_array");

        // 4. Сводки + CSV
        res.n_clusters       = amountOfClusters;
        res.min_cluster_idx  = amountOfNegativeClusters;

        // Per-field min/max (исключая 999/-999/NaN).
        auto compute_minmax = [&](const std::vector<double>& v, double& mn, double& mx) {
            mn = std::numeric_limits<double>::infinity();
            mx = -std::numeric_limits<double>::infinity();
            for (double x : v) {
                if (!std::isfinite(x)) continue;
                if (x == 999.0 || x == -999.0) continue;
                if (x < mn) mn = x;
                if (x > mx) mx = x;
            }
            if (!std::isfinite(mn) || !std::isfinite(mx)) { mn = 0.0; mx = 0.0; }
        };
        compute_minmax(res.avg_peaks,     res.avg_peaks_min,     res.avg_peaks_max);
        compute_minmax(res.avg_intervals, res.avg_intervals_min, res.avg_intervals_max);

        // Append the n_pts × n_pts grid for each of the 4 fields, after the
        // 2-line ranges header already written above. Layout matches the
        // shared writers data_export uses for the GUI right-click export.
        if (!OUT_FILE_PATH.empty()) {
            auto append_int = [&](const std::string& path, const int* data) {
                std::ofstream o(path, std::ios::app);
                if (!o.is_open()) return;
                o << std::setprecision(set_precision);
                data_export::write_basins_grid_int(o, nPts, data);
            };
            auto append_double = [&](const std::string& path, const double* data) {
                std::ofstream o(path, std::ios::app);
                if (!o.is_open()) return;
                o << std::setprecision(set_precision);
                data_export::write_basins_grid_double(o, nPts, data);
            };
            append_int   (OUT_FILE_PATH,             res.basin_idx.data());
            append_double(OUT_FILE_PATH + "_1.csv",  res.avg_peaks.data());
            append_double(OUT_FILE_PATH + "_2.csv",  res.avg_intervals.data());
            append_int   (OUT_FILE_PATH + "_3.csv",  res.helpful_array.data());
        }

        cleanup();
        #undef BAS_CHECK
        #undef BAS_CHECK_CU
        res.ok = true;
        return res;
    }


    // run_basins_recluster — DBSCAN-only прогон поверх кэшированных фич. Зеркалит фазу 2
    // (host-DBSCAN) из run_basins, но пропускает sim-фазу: d_avgPeaks/d_avgIntervals/d_helpfulArray
    // заливаются из request'а вместо повторного прогона avgPeakFinderCUDA. Мульты уже применены к
    // загруженным фичам (run_basins, фаза 1), поэтому здесь не нужны.
    BasinsReclusterResult ParametricEngine::Impl::run_basins_recluster(const BasinsReclusterRequest& req) {
        BasinsReclusterResult res;
        auto fail = [&](const std::string& msg) -> BasinsReclusterResult& {
            res.error = msg; return res;
        };

        if (req.krs_body.empty())                                return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX) return fail("amountOfX out of the allowed range");
        if (req.n_pts <= 0)                                      return fail("n_pts must be > 0");
        if (req.eps_dbscan <= 0.0)                               return fail("eps_dbscan must be > 0");
        const size_t total_cells = (size_t)req.n_pts * (size_t)req.n_pts;
        if (req.avg_peaks.size()     != total_cells)             return fail("avg_peaks: size does not match n_pts²");
        if (req.avg_intervals.size() != total_cells)             return fail("avg_intervals: size does not match n_pts²");
        if (req.helpful_array.size() != total_cells)             return fail("helpful_array: size does not match n_pts²");

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_basins_if_needed(req.krs_body, req.amountOfX, err)) return fail(err);

        // Ширина запуска dbscan-ядер — из настройки (launch_block_size ниже).
        const double eps_dbscan = req.eps_dbscan;

        int*    d_helpfulArray      = nullptr;
        int*    d_dbscanResult      = nullptr;
        numb* d_avgPeaks          = nullptr;
        numb* d_avgIntervals      = nullptr;
        int*    d_amountOfNeighbors = nullptr;
        int*    d_neighbors         = nullptr;
        int*    d_clearIdx          = nullptr;

        auto cleanup = [&]() {
            if (d_helpfulArray)      cudaFree(d_helpfulArray);
            if (d_dbscanResult)      cudaFree(d_dbscanResult);
            if (d_avgPeaks)          cudaFree(d_avgPeaks);
            if (d_avgIntervals)      cudaFree(d_avgIntervals);
            if (d_amountOfNeighbors) cudaFree(d_amountOfNeighbors);
            if (d_neighbors)         cudaFree(d_neighbors);
            if (d_clearIdx)          cudaFree(d_clearIdx);
        };

        #define BRC_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define BRC_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)
        #define BRC_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        BRC_CHECK(cudaMalloc((void**)&d_helpfulArray,      total_cells * sizeof(int)),    "cudaMalloc d_helpfulArray");
        BRC_CHECK(cudaMalloc((void**)&d_dbscanResult,      total_cells * sizeof(int)),    "cudaMalloc d_dbscanResult");
        BRC_CHECK(cudaMalloc((void**)&d_avgPeaks,          total_cells * sizeof(numb)), "cudaMalloc d_avgPeaks");
        BRC_CHECK(cudaMalloc((void**)&d_avgIntervals,      total_cells * sizeof(numb)), "cudaMalloc d_avgIntervals");
        BRC_CHECK(cudaMalloc((void**)&d_amountOfNeighbors, sizeof(int)),                  "cudaMalloc d_amountOfNeighbors");
        BRC_CHECK(cudaMalloc((void**)&d_neighbors,         total_cells * sizeof(int)),    "cudaMalloc d_neighbors");
        BRC_CHECK(cudaMalloc((void**)&d_clearIdx,          sizeof(int)),                  "cudaMalloc d_clearIdx");

        BRC_CHECK(cudaMemcpy(d_avgPeaks,     to_numb(req.avg_peaks).data(),     total_cells * sizeof(numb), cudaMemcpyHostToDevice), "memcpy avg_peaks H2D");
        BRC_CHECK(cudaMemcpy(d_avgIntervals, to_numb(req.avg_intervals).data(), total_cells * sizeof(numb), cudaMemcpyHostToDevice), "memcpy avg_intervals H2D");
        BRC_CHECK(cudaMemcpy(d_helpfulArray, req.helpful_array.data(), total_cells * sizeof(int),    cudaMemcpyHostToDevice), "memcpy helpful_array H2D");
        BRC_CHECK(cudaMemset(d_dbscanResult, 0, total_cells * sizeof(int)), "memset d_dbscanResult");
        BRC_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        const int blockSize_db = launch_block_size(0);   // dbscan-ядра без динамической shared
        const int gridSize_db  = (int)((total_cells + blockSize_db - 1) / blockSize_db);
        const int amountOfData_int = (int)total_cells;

        int amountOfClusters         = 0;
        int amountOfNegativeClusters = 0;
        std::vector<int> h_neighbors(total_cells, 0);
        int h_amountOfNeighbors      = 0;
        std::vector<int> h_dbscan_check(total_cells, 0);
        auto last_progress_scan = std::chrono::steady_clock::now() - std::chrono::seconds(1);

        if (req.progress) req.progress->store(0.0f, std::memory_order_relaxed);

        for (size_t main_iter = 0; main_iter < total_cells; ++main_iter) {
            BRC_CANCEL_CHECK();
            int clearIdx_init = -1;
            BRC_CHECK(cudaMemcpy(d_clearIdx, &clearIdx_init, sizeof(int), cudaMemcpyHostToDevice), "memcpy d_clearIdx init");

            void* args_search[] = { &d_avgPeaks, &d_avgIntervals, &d_helpfulArray, &d_dbscanResult,
                                    (void*)&amountOfData_int, &d_clearIdx };
            BRC_CHECK_CU(cuLaunchKernel(cached_basins.kernel_search_fixed,
                                        gridSize_db, 1, 1, blockSize_db, 1, 1,
                                        0, nullptr, args_search, nullptr),
                         "cuLaunchKernel(search_fixed)");
            BRC_CHECK(cudaDeviceSynchronize(), "sync search_fixed");

            int clearIdx = -1;
            BRC_CHECK(cudaMemcpy(&clearIdx, d_clearIdx, sizeof(int), cudaMemcpyDeviceToHost), "memcpy clearIdx D2H");

            int resultClusters = 0;
            if (clearIdx == -1) {
                BRC_CHECK_CU(cuLaunchKernel(cached_basins.kernel_search_clear,
                                            gridSize_db, 1, 1, blockSize_db, 1, 1,
                                            0, nullptr, args_search, nullptr),
                             "cuLaunchKernel(search_clear)");
                BRC_CHECK(cudaDeviceSynchronize(), "sync search_clear");
                BRC_CHECK(cudaMemcpy(&clearIdx, d_clearIdx, sizeof(int), cudaMemcpyDeviceToHost), "memcpy clearIdx D2H 2");
                if (clearIdx == -1) break;
                ++amountOfClusters;
                resultClusters = amountOfClusters;
            } else {
                --amountOfNegativeClusters;
                resultClusters = amountOfNegativeClusters;
            }

            h_amountOfNeighbors = 0;
            BRC_CHECK(cudaMemcpy(d_amountOfNeighbors, &h_amountOfNeighbors, sizeof(int), cudaMemcpyHostToDevice), "memcpy d_amountOfNeighbors=0");

            numb eps_arg = eps_dbscan;
            void* args_db[] = {
                &d_avgPeaks, &d_avgIntervals, &d_dbscanResult,
                (void*)&amountOfData_int, &eps_arg, &resultClusters,
                &d_amountOfNeighbors, &d_neighbors, &clearIdx, &d_helpfulArray
            };
            BRC_CHECK_CU(cuLaunchKernel(cached_basins.kernel_dbscan,
                                        gridSize_db, 1, 1, blockSize_db, 1, 1,
                                        0, nullptr, args_db, nullptr),
                         "cuLaunchKernel(dbscan expand init)");
            BRC_CHECK(cudaDeviceSynchronize(), "sync dbscan init");

            BRC_CHECK(cudaMemcpy(&h_amountOfNeighbors, d_amountOfNeighbors, sizeof(int), cudaMemcpyDeviceToHost), "memcpy amountOfNeighbors D2H");
            if (h_amountOfNeighbors > 0)
                BRC_CHECK(cudaMemcpy(h_neighbors.data(), d_neighbors, (size_t)h_amountOfNeighbors * sizeof(int), cudaMemcpyDeviceToHost),
                          "memcpy neighbors D2H");

            for (size_t ni = 0; ni < (size_t)h_amountOfNeighbors; ++ni) {
                int neighbor_idx = h_neighbors[ni];
                void* args_db2[] = {
                    &d_avgPeaks, &d_avgIntervals, &d_dbscanResult,
                    (void*)&amountOfData_int, &eps_arg, &resultClusters,
                    &d_amountOfNeighbors, &d_neighbors, &neighbor_idx, &d_helpfulArray
                };
                BRC_CHECK_CU(cuLaunchKernel(cached_basins.kernel_dbscan,
                                            gridSize_db, 1, 1, blockSize_db, 1, 1,
                                            0, nullptr, args_db2, nullptr),
                             "cuLaunchKernel(dbscan expand neighbor)");
                BRC_CHECK(cudaDeviceSynchronize(), "sync dbscan neighbor");

                BRC_CHECK(cudaMemcpy(&h_amountOfNeighbors, d_amountOfNeighbors, sizeof(int), cudaMemcpyDeviceToHost), "memcpy amountOfNeighbors loop");
                if (h_amountOfNeighbors > 0)
                    BRC_CHECK(cudaMemcpy(h_neighbors.data(), d_neighbors, (size_t)h_amountOfNeighbors * sizeof(int), cudaMemcpyDeviceToHost),
                              "memcpy neighbors loop");
            }

            if (req.progress) {
                auto now = std::chrono::steady_clock::now();
                if (now - last_progress_scan >= std::chrono::milliseconds(200)) {
                    last_progress_scan = now;
                    BRC_CHECK(cudaMemcpy(h_dbscan_check.data(), d_dbscanResult,
                                         total_cells * sizeof(int), cudaMemcpyDeviceToHost),
                              "memcpy dbscanResult D2H (progress)");
                    size_t classified = 0;
                    for (size_t k = 0; k < total_cells; ++k)
                        if (h_dbscan_check[k] != 0) ++classified;
                    req.progress->store(float(classified) / float(total_cells),
                                        std::memory_order_relaxed);
                }
            }
        }

        res.n_pts = req.n_pts;
        res.basin_idx.assign(total_cells, 0);
        BRC_CHECK(cudaMemcpy(res.basin_idx.data(), d_dbscanResult, total_cells * sizeof(int), cudaMemcpyDeviceToHost), "memcpy basin_idx D2H");
        res.n_clusters      = amountOfClusters;
        res.min_cluster_idx = amountOfNegativeClusters;

        cleanup();
        #undef BRC_CHECK
        #undef BRC_CHECK_CU
        #undef BRC_CANCEL_CHECK
        res.ok = true;
        return res;
    }

    void ParametricEngine::Impl::prewarm_basins(const BasinsRequest& req) {
        std::string err;
        if (!ensure_init(err)) return;
        cuCtxSetCurrent(context);
        if (req.adaptive.enabled) compile_ad_if_needed(req.adaptive, req.krs_body, req.amountOfX, 0, err, false);
        else compile_basins_if_needed(req.krs_body, req.amountOfX, err, false);
    }
