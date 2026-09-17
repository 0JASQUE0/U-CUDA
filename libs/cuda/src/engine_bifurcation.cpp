#include "engine_impl.h"



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
    Bifurcation1DResult ParametricEngine::Impl::run_bif1d(const Bifurcation1DRequest& req) {
        // Continuation требует sequential x-carry — это совсем другой путь
        // (single-thread kernel). Отказываем при IC-sweep (не имеет смысла:
        // continuation подразумевает param как непрерывный параметр).
        if (req.continuation) {
            if (req.sweep_over_var) {
                Bifurcation1DResult r;
                r.error = "continuation требует param-sweep, не IC-sweep";
                return r;
            }
            // Continuation-ветка возвращается ДО общего блока валидации ниже,
            // поэтому log-проверку дублируем здесь: без неё log10(lo<=0) дал бы
            // NaN-сетку молча. У LLE/LS эта проверка отрабатывает раньше
            // диспетчера, там дублировать не нужно.
            if (req.log_scale && !(req.param_lo > 0.0 && req.param_hi > 0.0)) {
                Bifurcation1DResult r;
                r.error = "log scale требует param lo/hi > 0";
                return r;
            }
            // h-свип теперь поддержан обеими ветками: шаг пересчитывается в
            // каждой точке, буфер блока выделен под худший случай.
            // CPU-ветка — до ensure_init: она не трогает CUDA вообще, так что
            // считает и на машине без работающего GPU-контекста.
            if (req.use_cpu) return run_bif1d_continuation_cpu(req);
            return run_bif1d_continuation(req);
        }

        Bifurcation1DResult res;
        auto fail = [&](const std::string& msg) -> Bifurcation1DResult& { res.error = msg; return res; };

        // валидация
        if (req.krs_body.empty())                                   return fail("krs_body пуст");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX вне [1," + std::to_string(kMaxAmountOfX) + "]");
        if ((int)req.initial_conditions.size() != req.amountOfX)    return fail("initial_conditions.size() != amountOfX");
        // base_values уже идёт со сдвигом +1 (a[0] зарезервирован):
        if ((int)req.base_values.size() > kMaxAmountOfValues)       return fail("base_values слишком много");
        if (req.sweep_over_h) {
            if (req.param_lo <= 0.0 || req.param_hi <= 0.0)
                return fail("h lo/hi должны быть > 0 при sweep_over_h");
        } else if (req.sweep_over_var) {
            if (req.var_sweep_index < 0 || req.var_sweep_index >= req.amountOfX)
                return fail("var_sweep_index вне диапазона");
        } else {
            if (req.param_index < 0 || req.param_index >= (int)req.base_values.size())
                return fail("param_index вне диапазона");
        }
        // writable_var == -1 — sentinel "combination of first vars" (порт MATLAB
        // выбора x[0]+pi*x[1]+euler*x[2] в loopCalculateDiscreteModel_int).
        if (req.writable_var < -1 || req.writable_var >= req.amountOfX)
                                                                    return fail("writable_var вне диапазона");
        if (req.n_pts <= 0)         return fail("n_pts должно быть > 0");
        if (req.h <= 0.0)           return fail("h должно быть > 0");
        if (req.t_max <= 0.0)       return fail("t_max должно быть > 0");
        if (req.transient_time < 0) return fail("transient_time должно быть >= 0");
        if (req.pre_scaller <= 0)   return fail("pre_scaller должно быть > 0");
        if (req.log_scale && !(req.param_lo > 0.0 && req.param_hi > 0.0))
            return fail("log scale требует param lo/hi > 0");

        // Классический свип на CPU — для счёта без GPU и для сверки ядра с GPU
        // (то же место в конвейере, что у run_lle_1d / run_ls_1d). Стоит после
        // общей валидации и до ensure_init: CUDA этой ветке не нужна вовсе.
        if (req.use_cpu) return run_bif1d_cpu(req);

        // dt-sweep: t_max/transient_time фиксированы, число шагов на GPU
        // пересчитывается из h per-thread (см. hSweepAxis в
        // calculateDiscreteModelCUDA). hSweepAxis=0 -- в 1D всегда единственная
        // ось X.
        const int hSweepAxis = req.sweep_over_h ? 0 : -1;
        // Log-масштаб сетки (любой sweep target) -- бит 0, т.к. в 1D одна ось.
        const int logAxisMask = req.log_scale ? 1 : 0;

        // init + контекст
        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);

        // компиляция или cache hit
        if (!compile_if_needed(req.krs_body, req.amountOfX,
                               req.sweep_over_var ? 0 : 1, err)) return fail(err);

        // ПОРТ NonLinAnal::bifurcation1D (hostLibrary.cu:165-655).
        // Локальные имена мапятся на аргументы функции NonLinAnal для удобства
        // дифа — слева name из req, справа name как в hostLibrary.
        const double tMax                       = req.t_max;
        const int    nPts                       = req.n_pts;
        const double h                          = req.h;
        const int    amountOfInitialConditions  = req.amountOfX;
        const std::vector<numb> ic_staged_       = to_numb(req.initial_conditions);
        const numb*   initialConditions         = ic_staged_.data();
        numb   ranges[2]                        = { (numb)req.param_lo, (numb)req.param_hi };
        // Sweep target:
        //   param-sweep → indicesOfMutVars[0] = 1-based индекс параметра (a[]),
        //                 par_or_var_arg = true  → kernel пишет в localValues
        //   var-sweep   → indicesOfMutVars[0] = 0-based индекс переменной (X[]),
        //                 par_or_var_arg = false → kernel пишет в localX
        int    indicesOfMutVars[1]              = { req.sweep_over_var
                                                    ? req.var_sweep_index
                                                    : req.param_index };
        const int    writableVar                = req.writable_var;
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const std::vector<numb> values_staged_   = to_numb(req.base_values);
        const numb*   values                    = values_staged_.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const int    preScaller                 = req.pre_scaller;
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        // Константы из configCUDA.h — у NonLinAnal они constexpr,
        // здесь литералы (значения те же).
        constexpr int  blockSize_setup          = 32;
        constexpr int  set_precision            = 15;
        // [ADAPT] continuation_bif1D, calculate_mean_med_freq отключены —
        // см. комментарий в шапке функции.

        // amountOfPointsInBlock / amountOfPointsForSkip — порт строк 196-200 NL
        // dt-sweep: буфер/sizeOfBlock должен вмещать худший случай (минимальный
        // h в диапазоне = param_lo после lo<=hi нормализации = больше всего
        // шагов); per-thread реальное число шагов пересчитывается в кернеле и
        // не превышает эту аллокацию (см. actualIterations).
        double worstCaseH = (hSweepAxis != -1) ? ranges[0] : h;
        int amountOfPointsInBlock = (int)std::ceil(tMax / worstCaseH / preScaller);
        size_t amountOfPointsForSkip = steps_from_time_size_t(transientTime, h);

        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max/h/pre_scaller слишком малы)");

        // Пики живут в собственных коротких строках, а не в буферах длиной
        // в траекторию — см. run_bif2d. Для 1D это важнее, чем для 2D: здесь оба
        // буфера пиков едут на хост целиком каждый чанк (2D возвращал по int на ячейку).
        const size_t peakStride   = (size_t)amountOfPointsInBlock < (size_t)max_amount_of_peaks + 1
                                  ? (size_t)amountOfPointsInBlock
                                  : (size_t)max_amount_of_peaks + 1;
        const int    peakCapacity = (int)peakStride;

        // Memory budget (порт строк 202-244 NL)
        size_t freeMemory = 0;
        if (!gpu_free_budget(0.92, freeMemory)) return fail("cudaMemGetInfo failed");

        size_t memPerSystem =
            2 * peakStride * sizeof(numb) +                     // d_outPeaks, d_timeOfPeaks
            sizeof(int);                                          // d_amountOfPeaks
        size_t memConstants =
            2 * sizeof(numb) +
            sizeof(int) +
            (size_t)amountOfInitialConditions * sizeof(numb) +
            (size_t)amountOfValues * sizeof(numb);
        constexpr double SAFETY_FACTOR = 0.9;
        size_t safeFree = (size_t)((double)freeMemory * SAFETY_FACTOR);
        if (memConstants >= safeFree) return fail("not enough GPU memory for constants");
        size_t availableMemory = safeFree - memConstants;

        size_t nPtsLimiter = availableMemory / memPerSystem;
        if (nPtsLimiter < (size_t)blockSize_setup) nPtsLimiter = (size_t)blockSize_setup;
        if (nPtsLimiter > (size_t)nPts)            nPtsLimiter = (size_t)nPts;
        // Округления вниз до кратного blockSize_setup (nPtsLimiter / 32 * 32) здесь больше нет:
        // ядра сами отсекают лишние потоки через `if (idx >= nPtsLimiter) return`, а последний чанк
        // кратным 32 не бывает и всегда считался нормально. Зато при n_pts < 32 округление давало
        // 0, и Run падал с сообщением про нехватку памяти, которая была ни при чём.
        if (nPtsLimiter == 0) return fail("n_pts должно быть > 0");
        size_t originalNPtsLimiter = nPtsLimiter;

        // Host buffers (порт строк 257-264 NL)
        // h_data/h_meanFreq/h_medianFreq/h_localX/h_localValues нужны только для
        // continuation_bif1D и mean/median — мы их не используем.
        std::vector<numb> h_outPeaks   (nPtsLimiter * peakStride);
        std::vector<numb> h_timeOfPeaks(nPtsLimiter * peakStride);
        std::vector<int>    h_amountOfPeaks(nPtsLimiter);

        // Device buffers (порт строк 297-306 NL, без d_meanFreq/d_medianFreq)
        numb* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        numb* d_initialConditions = nullptr;
        numb* d_values            = nullptr;
        int*    d_amountOfPeaks     = nullptr;
        RunSignals sig;                       // прогресс и отмена в mapped-памяти
        numb* d_outPeaks          = nullptr;
        numb* d_timeOfPeaks       = nullptr;

        auto cleanup = [&]() {
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_amountOfPeaks)     cudaFree(d_amountOfPeaks);
            sig.release();
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
        // Cooperative cancellation between kernel launches.
        #define BIF_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        BIF_CHECK(cudaMalloc((void**)&d_ranges,            2 * sizeof(numb)),                                          "cudaMalloc d_ranges");
        BIF_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  1 * sizeof(int)),                                             "cudaMalloc d_indicesOfMutVars");
        BIF_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)amountOfInitialConditions * sizeof(numb)),          "cudaMalloc d_initialConditions");
        BIF_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(numb)),                     "cudaMalloc d_values");
        BIF_CHECK(cudaMalloc((void**)&d_outPeaks,          nPtsLimiter * peakStride * sizeof(numb)),                    "cudaMalloc d_outPeaks");
        BIF_CHECK(cudaMalloc((void**)&d_timeOfPeaks,       nPtsLimiter * peakStride * sizeof(numb)),                    "cudaMalloc d_timeOfPeaks");
        BIF_CHECK(cudaMalloc((void**)&d_amountOfPeaks,     nPtsLimiter * sizeof(int)),                                   "cudaMalloc d_amountOfPeaks");
        if (!sig.alloc(res.error)) { cleanup(); return res; }
        const size_t stepsPerPoint  = amountOfPointsForSkip + (size_t)amountOfPointsInBlock;
        const int    progressStride = progress_stride_for(stepsPerPoint);
        const double ticksPerPoint  = (double)(stepsPerPoint / (size_t)progressStride);
        const double ticksTotal     = (double)nPts * ticksPerPoint;

        // H2D констант (порт строк 314-319 NL)
        BIF_CHECK(cudaMemcpy(d_ranges,            ranges,             2 * sizeof(numb),                                cudaMemcpyHostToDevice), "memcpy d_ranges");
        BIF_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,   1 * sizeof(int),                                   cudaMemcpyHostToDevice), "memcpy d_indices");
        BIF_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)amountOfInitialConditions * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_ic");
        BIF_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(numb),            cudaMemcpyHostToDevice), "memcpy d_values");
        BIF_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)nPts / (double)nPtsLimiter);

        // Snapshot of CSV-relevant request fields, captured BEFORE any disk
        // write so it can be reused by the GUI right-click export (which runs
        // after `req` is gone). Engine and GUI share the same writers in
        // data_export, so the on-disk format is identical byte-for-byte.
        res.snapshot.values.assign(values, values + amountOfValues);
        res.snapshot.initial_conditions.assign(initialConditions,
            initialConditions + amountOfInitialConditions);
        res.snapshot.tMax          = tMax;
        res.snapshot.transientTime = transientTime;
        res.snapshot.h             = h;
        res.snapshot.gpu_fmad      = get_nvrtc_fmad();
        res.snapshot.preScaller    = preScaller;
        res.snapshot.writableVar   = writableVar;
        res.snapshot.indexOfMutVar = indicesOfMutVars[0];
        res.snapshot.range_lo      = ranges[0];
        res.snapshot.range_hi      = ranges[1];

        // Config CSV (порт строк 331-376 NL — упрощённо, только если путь задан)
        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            data_export::write_bif1d_config(cfg, res.snapshot);
            // обнуляем основной файл данных
            std::ofstream trunc(OUT_FILE_PATH);
            trunc.close();
        }

        // результат-аккумулятор (для GUI)
        res.n_pts        = nPts;
        res.record_steps = (int)peakStride;   // ёмкость строки пиков, а не длина траектории
        res.flags.assign(nPts, 0);
        res.bifurcation_points.assign(nPts, {});
        res.peak_times.assign(nPts, {});

        // Главный цикл (порт строк 396-630 NL)
        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            BIF_CANCEL_CHECK();
            if (req.progress) req.progress->store(float(iter) / float(amountOfIteration), std::memory_order_relaxed);
            // последний чанк может быть меньше
            if (iter == amountOfIteration - 1)
                nPtsLimiter = nPts - (originalNPtsLimiter * iter);

            int blockSize = 32;
            int gridSize  = (int)((nPtsLimiter + blockSize - 1) / blockSize);

            // [ADAPT] <<<>>> → cuLaunchKernel. Траектория и поиск пиков — одно
            // ядро, то же, что у run_bif2d. Continuation отключена.
            int    nPts_int                  = nPts;
            int    nPtsLimiter_int           = (int)nPtsLimiter;
            size_t amountOfCalculatedPoints  = iter * originalNPtsLimiter;
            size_t amountOfPointsForSkip_s   = (size_t)amountOfPointsForSkip;
            int    dimension                 = 1;
            numb h_arg                     = h;
            int    amountOfInitialConditions_int = amountOfInitialConditions;
            int    amountOfValues_int        = amountOfValues;
            size_t amountOfIterations_arg    = (size_t)amountOfPointsInBlock;
            int    preScaller_int            = preScaller;
            int    writableVar_int           = writableVar;
            numb maxValue_arg              = maxValue;
            bool   par_or_var_arg            = !req.sweep_over_var; // true=param, false=IC
            int    hSweepAxis_arg            = hSweepAxis;
            numb transientTime_arg         = transientTime;
            numb tMax_arg                  = tMax;
            int    logAxisMask_arg           = logAxisMask;
            size_t peakStride_arg            = peakStride;
            int    peakCapacity_arg          = peakCapacity;
            int*   d_cancel_arg              = sig.cancelArg();
            int*   d_progress_arg            = sig.progressArg();
            int    progressStride_arg        = progressStride;
            sig.resetTicks();

            void* args_fused[] = {
                &nPts_int, &nPtsLimiter_int, &amountOfCalculatedPoints,
                &amountOfPointsForSkip_s, &dimension, &d_ranges, &h_arg,
                &d_indicesOfMutVars, &d_initialConditions, &amountOfInitialConditions_int,
                &d_values, &amountOfValues_int, &amountOfIterations_arg,
                &preScaller_int, &writableVar_int, &maxValue_arg,
                &d_outPeaks, &d_timeOfPeaks, &d_amountOfPeaks,
                &par_or_var_arg, &hSweepAxis_arg, &transientTime_arg, &tMax_arg,
                &logAxisMask_arg, &peakStride_arg, &peakCapacity_arg,
                &d_cancel_arg, &d_progress_arg, &progressStride_arg
            };

            unsigned int shared = (unsigned int)(ucuda_shared_stride(amountOfInitialConditions, amountOfValues) * sizeof(numb) * blockSize);

            BIF_CHECK_CU(cuLaunchKernel(cached.kernel_fused,
                                        gridSize, 1, 1, blockSize, 1, 1,
                                        shared, nullptr, args_fused, nullptr),
                         "cuLaunchKernel(bif1d traj+peaks)");
            if (!wait_with_signals(0, sig, req.cancel, req.progress,
                                   (double)(originalNPtsLimiter * iter) * ticksPerPoint,
                                   ticksTotal, res.error)) { cleanup(); return res; }
            BIF_CHECK(cudaDeviceSynchronize(), "sync after traj+peaks");
            BIF_CANCEL_CHECK();

            // D2H (порт строк 538-540 NL)
            BIF_CHECK(cudaMemcpy(h_outPeaks.data(),       d_outPeaks,       nPtsLimiter * peakStride * sizeof(numb),                    cudaMemcpyDeviceToHost), "memcpy h_outPeaks");
            BIF_CHECK(cudaMemcpy(h_amountOfPeaks.data(),  d_amountOfPeaks,  nPtsLimiter * sizeof(int),                                    cudaMemcpyDeviceToHost), "memcpy h_amountOfPeaks");
            BIF_CHECK(cudaMemcpy(h_timeOfPeaks.data(),    d_timeOfPeaks,    nPtsLimiter * peakStride * sizeof(numb),                    cudaMemcpyDeviceToHost), "memcpy h_timeOfPeaks");
            BIF_CHECK(cudaDeviceSynchronize(), "sync after D2H");

            // CSV + аккумуляция результата (порт строк 574-608 NL)
            std::ofstream out;
            if (!OUT_FILE_PATH.empty()) {
                out.open(OUT_FILE_PATH, std::ios::app);
                if (out.is_open()) out << std::setprecision(set_precision);
            }

            for (size_t k = 0; k < nPtsLimiter; ++k) {
                size_t global_idx = originalNPtsLimiter * iter + k;
                double param_val  = req.log_scale ? getValueByIdx_log_local(global_idx, nPts, ranges[0], ranges[1])
                                                   : getValueByIdx_local(global_idx, nPts, ranges[0], ranges[1]);
                int    npeaks     = h_amountOfPeaks[k];

                // Поправки на h-свип здесь больше нет. Раньше peakFinderCUDA получал общий
                // на запуск h, и хост домножал времена на param_val/h; слитое ядро
                // сразу масштабирует интервалы шагом своей точки, и вторая поправка
                // применилась бы дважды (см. commit-message к слиянию ядер bif2d).

                int n = npeaks;
                if (n > (int)peakStride) n = (int)peakStride;
                const numb* peakRow = h_outPeaks.data()    + k * peakStride;
                const numb* timeRow = h_timeOfPeaks.data() + k * peakStride;

                // Экспорт и результат работают в double — расширяем numb-ряды
                // здесь, один раз, и переиспользуем для CSV и для памяти.
                std::vector<double> widePeaks, scaledTimes;
                if (n > 0) {
                    widePeaks.assign(peakRow, peakRow + n);
                    scaledTimes.assign(timeRow, timeRow + n);
                }

                if (out.is_open())
                    data_export::write_bif1d_rows(out, param_val, npeaks,
                                                  widePeaks.data(), scaledTimes.data());

                // В память для GUI: значения пиков и межпиковые интервалы
                res.flags[global_idx] = npeaks;
                auto& dst_peaks = res.bifurcation_points[global_idx];
                auto& dst_times = res.peak_times[global_idx];
                if (n > 0) {
                    dst_peaks = std::move(widePeaks);
                    dst_times = std::move(scaledTimes);
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


    // run_bif1d_continuation — single-thread sequential. Каждый параметр
    // стартует с конечного x[] предыдущего. Direction (forward/reverse)
    // обрабатывается в kernel'е. PeakFinderCUDA вызывается из того же модуля.
    Bifurcation1DResult ParametricEngine::Impl::run_bif1d_continuation(const Bifurcation1DRequest& req) {
        Bifurcation1DResult res;
        auto fail = [&](const std::string& msg) -> Bifurcation1DResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                    return fail("krs_body пуст");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)      return fail("amountOfX вне диапазона");
        if ((int)req.initial_conditions.size() != req.amountOfX)     return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)        return fail("base_values слишком много");
        if (req.param_index < 0 || req.param_index >= (int)req.base_values.size())
                                                                     return fail("param_index вне диапазона");
        // writable_var == -1 — sentinel "combination" (см. loopCalculateDiscreteModel_int).
        if (req.writable_var < -1 || req.writable_var >= req.amountOfX)
                                                                     return fail("writable_var вне диапазона");
        if (req.n_pts <= 0)         return fail("n_pts должно быть > 0");
        if (req.h <= 0.0)           return fail("h должно быть > 0");
        if (req.t_max <= 0.0)       return fail("t_max должно быть > 0");
        if (req.transient_time < 0) return fail("transient_time должно быть >= 0");
        if (req.pre_scaller <= 0)   return fail("pre_scaller должно быть > 0");

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_bif1d_cont_if_needed(req.krs_body, req.amountOfX, err))
            return fail(err);

        // При h-свипе шаг меняется от точки к точке: буфер блока — под худший
        // случай (минимальный h), реальную длину каждой точки kernel сообщает
        // через d_actualIterations, и peakFinderCUDA сканирует только её.
        const double worstCaseH = req.sweep_over_h
                                ? ((req.param_lo < req.param_hi) ? req.param_lo : req.param_hi)
                                : req.h;
        if (worstCaseH <= 0.0) return fail("h должно быть > 0 (при h-свипе — весь диапазон)");
        const int amountOfPointsInBlock = (int)std::ceil(req.t_max / worstCaseH / req.pre_scaller);
        if (amountOfPointsInBlock <= 0) return fail("amountOfPointsInBlock <= 0");

        const int nPts = req.n_pts;
        numb* d_data       = nullptr;
        numb* d_baseValues = nullptr;
        numb* d_baseX      = nullptr;
        int*    d_amountOfPeaks = nullptr;
        numb* d_outPeaks   = nullptr;
        numb* d_timeOfPeaks= nullptr;
        int*    d_actualIterations = nullptr;

        RunSignals sig;   // однопоточное ядро: тик на точку
        auto cleanup = [&]() {
            if (d_data)       cudaFree(d_data);
            if (d_baseValues) cudaFree(d_baseValues);
            if (d_baseX)      cudaFree(d_baseX);
            if (d_amountOfPeaks) cudaFree(d_amountOfPeaks);
            if (d_outPeaks)   cudaFree(d_outPeaks);
            if (d_timeOfPeaks)cudaFree(d_timeOfPeaks);
            if (d_actualIterations) cudaFree(d_actualIterations);
            sig.release();
        };
        #define C_CHECK(call, where) do { cudaError_t _e = (call); \
            if (_e != cudaSuccess) { res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); cleanup(); return res; } } while(0)
        #define C_CHECK_CU(call, where) do { CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { res.error = std::string(where) + ": " + cu_err(_r); cleanup(); return res; } } while(0)
        // Continuation kernel is monolithic (single launch sweeping all nPts),
        // so this can only catch cancellation BEFORE launch. After launch the
        // kernel runs to completion regardless.
        #define C_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        const size_t dataBytes  = (size_t)nPts * (size_t)amountOfPointsInBlock * sizeof(numb);
        C_CHECK(cudaMalloc((void**)&d_data,         dataBytes),                                       "cudaMalloc d_data");
        C_CHECK(cudaMalloc((void**)&d_baseValues,   (size_t)req.base_values.size() * sizeof(numb)), "cudaMalloc d_baseValues");
        C_CHECK(cudaMalloc((void**)&d_baseX,        (size_t)req.amountOfX * sizeof(numb)),          "cudaMalloc d_baseX");
        C_CHECK(cudaMalloc((void**)&d_amountOfPeaks,(size_t)nPts * sizeof(int)),                      "cudaMalloc d_amountOfPeaks");
        C_CHECK(cudaMalloc((void**)&d_outPeaks,     dataBytes),                                       "cudaMalloc d_outPeaks");
        C_CHECK(cudaMalloc((void**)&d_timeOfPeaks,  dataBytes),                                       "cudaMalloc d_timeOfPeaks");
        C_CHECK(cudaMalloc((void**)&d_actualIterations, (size_t)nPts * sizeof(int)),                  "cudaMalloc d_actualIterations");

        // Вход приходит из Request в double — сужаем до numb на границе.
        const std::vector<numb> baseValues_staged_ = to_numb(req.base_values);
        const std::vector<numb> baseX_staged_      = to_numb(req.initial_conditions);
        C_CHECK(cudaMemcpy(d_baseValues, baseValues_staged_.data(),
                           req.base_values.size() * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_baseValues");
        C_CHECK(cudaMemcpy(d_baseX, baseX_staged_.data(),
                           (size_t)req.amountOfX * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_baseX");
        C_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        // Launch continuation kernel
        int    nPts_arg              = nPts;
        numb lo_arg                = req.param_lo;
        numb hi_arg                = req.param_hi;
        bool   reverse_arg           = req.continuation_reverse;
        int    logScale_arg          = req.log_scale ? 1 : 0;
        int    sweepIsH_arg          = req.sweep_over_h ? 1 : 0;
        int    mutParamIdx_arg       = req.param_index;
        int    amountOfValues_arg    = (int)req.base_values.size();
        int    amountOfX_arg         = req.amountOfX;
        numb h_arg                 = req.h;
        numb tMax_arg              = req.t_max;
        numb transientTime_arg     = req.transient_time;
        int    sizeOfBlock_arg       = amountOfPointsInBlock;
        int    preScaller_arg        = req.pre_scaller;
        int    writableVar_arg       = req.writable_var;
        numb maxValue_arg          = req.max_value;

        if (!sig.alloc(res.error)) { cleanup(); return res; }
        int* d_cancel_arg   = sig.cancelArg();
        int* d_progress_arg = sig.progressArg();
        void* cont_args[] = {
            &nPts_arg, &lo_arg, &hi_arg, &reverse_arg,
            &logScale_arg, &sweepIsH_arg, &mutParamIdx_arg,
            &d_baseValues, &amountOfValues_arg,
            &d_baseX,      &amountOfX_arg,
            &h_arg, &tMax_arg, &transientTime_arg,
            &sizeOfBlock_arg, &preScaller_arg,
            &writableVar_arg, &maxValue_arg,
            &d_data, &d_amountOfPeaks, &d_actualIterations,
            &d_cancel_arg, &d_progress_arg
        };
        C_CANCEL_CHECK();
        // Continuation is monolithic: progress jumps 0 -> 0.5 around the sweep
        // kernel, then -> 1.0 after the peak-finder below. Mid-kernel reporting
        // isn't possible without restructuring the kernel itself.
        C_CHECK_CU(cuLaunchKernel(cached_cont.kernel_cont,
                                  1, 1, 1, 1, 1, 1, 0, nullptr, cont_args, nullptr),
                   "cuLaunchKernel(cont)");
        if (!wait_with_signals(0, sig, req.cancel, req.progress, 0.0, (double)nPts, res.error))
            { cleanup(); return res; }
        C_CHECK(cudaDeviceSynchronize(), "sync after cont kernel");
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true; res.error = "Cancelled by user"; cleanup(); return res;
        }
        if (req.progress) req.progress->store(0.5f, std::memory_order_relaxed);

        // Launch peakFinderCUDA на полученные данные
        // Сигнатура: (numb* data, size_t sizeOfBlock, int amountOfBlocks,
        //             int* amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb h)
        size_t sizeOfBlock_s = (size_t)amountOfPointsInBlock;
        int    nBlocks       = nPts;
        // numb, а не double: cuLaunchKernel копирует аргумент побайтово, а
        // peakFinderCUDA ждёт numb h (см. сигнатуру выше). При numb=float из
        // double уехали бы первые 4 байта, и межпиковые интервалы разъехались
        // бы на порядки. В классическом пути тот же аргумент объявлен numb.
        numb timeStep        = (numb)(req.h * (double)req.pre_scaller);
        // actualIterations — 8-й параметр peakFinderCUDA: реальная длина блока каждой точки (при
        // h-свипе она своя, буфер выделен под худший случай). Раньше аргумент не передавался вовсе:
        // cuLaunchKernel читал 8-й слот за концом peak_args[], kernel получал мусорный указатель и
        // разыменовывал его — "illegal memory access" сразу после peak-kernel'а.
        // 9-й и 10-й параметры peakFinderCUDA (peakStride/peakCapacity): 0/0 = legacy-раскладка.
        // Передаём явно по той же причине, что и actualIterations выше.
        size_t peakStride_arg   = 0;
        int    peakCapacity_arg = 0;
        void* peak_args[] = {
            &d_data, &sizeOfBlock_s, &nBlocks,
            &d_amountOfPeaks, &d_outPeaks, &d_timeOfPeaks, &timeStep,
            &d_actualIterations,
            &peakStride_arg, &peakCapacity_arg
        };
        int    peakBlock = 32;
        int    peakGrid  = (nPts + peakBlock - 1) / peakBlock;
        C_CHECK_CU(cuLaunchKernel(cached_cont.kernel_peak,
                                  peakGrid, 1, 1, peakBlock, 1, 1,
                                  0, nullptr, peak_args, nullptr),
                   "cuLaunchKernel(peak)");
        C_CHECK(cudaDeviceSynchronize(), "sync after peak");

        // D2H
        std::vector<numb> h_outPeaks   ((size_t)nPts * (size_t)amountOfPointsInBlock);
        std::vector<numb> h_timeOfPeaks((size_t)nPts * (size_t)amountOfPointsInBlock);
        std::vector<int>    h_amountOfPeaks((size_t)nPts);
        C_CHECK(cudaMemcpy(h_outPeaks.data(),    d_outPeaks,    dataBytes,             cudaMemcpyDeviceToHost), "memcpy out h_outPeaks");
        C_CHECK(cudaMemcpy(h_timeOfPeaks.data(), d_timeOfPeaks, dataBytes,             cudaMemcpyDeviceToHost), "memcpy out h_timeOfPeaks");
        C_CHECK(cudaMemcpy(h_amountOfPeaks.data(), d_amountOfPeaks, nPts * sizeof(int), cudaMemcpyDeviceToHost), "memcpy out h_amountOfPeaks");
        C_CHECK(cudaDeviceSynchronize(), "sync after D2H");

        // Заполнение Result
        res.n_pts        = nPts;
        res.record_steps = amountOfPointsInBlock;
        res.param_lo     = req.param_lo;
        res.param_hi     = req.param_hi;
        res.continuation_reverse = req.continuation_reverse;
        res.flags.assign(nPts, 0);
        res.bifurcation_points.assign(nPts, {});
        res.peak_times.assign(nPts, {});
        for (int j = 0; j < nPts; ++j) {
            int n = h_amountOfPeaks[j];
            res.flags[j] = n;
            if (n > 0) {
                if (n > amountOfPointsInBlock) n = amountOfPointsInBlock;
                const numb* pr = h_outPeaks.data()    + (size_t)j * amountOfPointsInBlock;
                const numb* tr = h_timeOfPeaks.data() + (size_t)j * amountOfPointsInBlock;
                res.bifurcation_points[j].assign(pr, pr + n);
                res.peak_times[j].assign(tr, tr + n);
                // h-свип: peakFinderCUDA умножал разности индексов на общий h,
                // а шаг этой точки — cont_sweep_value. Интервал линеен по h,
                // поэтому поправка точная (см. тот же приём в run_bif1d).
                if (req.sweep_over_h) {
                    const double h_point = cont_sweep_value(j, nPts, req.param_lo, req.param_hi,
                                                            req.continuation_reverse, req.log_scale);
                    const double scale = h_point / req.h;
                    if (scale != 1.0)
                        for (double& v : res.peak_times[j]) v *= scale;
                }
            }
        }

        cleanup();
        #undef C_CHECK
        #undef C_CHECK_CU
        res.ok = true;
        return res;
    }


    // run_bif2d — порт NonLinAnal::bifurcation2D (hostLibrary.cu:898-1724).
    // Три kernel'а: calculateDiscreteModelCUDA → peakFinderCUDA → dbscanCUDA.
    // Результат: число DBSCAN-кластеров на ячейку = период системы.
    // [ADAPT]: те же адаптации, что у run_bif1d + run_lle_2d (см. их комментарии).
    Bifurcation2DResult ParametricEngine::Impl::run_bif2d(const Bifurcation2DRequest& req) {
        Bifurcation2DResult res;
        auto fail = [&](const std::string& msg) -> Bifurcation2DResult& { res.error = msg; return res; };

        // валидация
        if (req.krs_body.empty())                                    return fail("krs_body пуст");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX вне [1," + std::to_string(kMaxAmountOfX) + "]");
        if ((int)req.initial_conditions.size() != req.amountOfX)     return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)        return fail("base_values слишком много");
        if (req.n_pts <= 0)          return fail("n_pts должно быть > 0");
        if (req.h <= 0.0)            return fail("h должно быть > 0");
        if (req.t_max <= 0.0)        return fail("t_max должно быть > 0");
        if (req.transient_time < 0)  return fail("transient_time должно быть >= 0");
        if (req.pre_scaller <= 0)    return fail("pre_scaller должно быть > 0");
        if (req.eps_dbscan <= 0.0)   return fail("eps_dbscan должно быть > 0");
        if (req.writable_var < -1 || req.writable_var >= req.amountOfX) return fail("writable_var вне диапазона");

        // par_or_var + swap_xy (та же логика что у run_lle_2d)
        auto check_param = [&](int p1based) -> bool {
            return p1based >= 0 && p1based < (int)req.base_values.size();
        };
        auto check_var = [&](int v0based) -> bool {
            return v0based >= 0 && v0based < req.amountOfX;
        };

        int par_or_var;
        int    idx_axis_x, idx_axis_y;
        double ranges_lo_x, ranges_hi_x, ranges_lo_y, ranges_hi_y;
        bool   swap_xy = false;
        int    hSweepAxis = -1;
        bool log_axis_x = req.log_scale, log_axis_y = req.log_scale_2;  // см. run_lle_2d

        if (req.log_scale && !(req.param_lo > 0.0 && req.param_hi > 0.0))
            return fail("log scale требует param lo/hi > 0 (ось X)");
        if (req.log_scale_2 && !(req.param_lo_2 > 0.0 && req.param_hi_2 > 0.0))
            return fail("log scale требует param lo/hi > 0 (ось Y)");

        if (req.sweep_over_h && req.sweep_over_h_2)
            return fail("sweep_over_h и sweep_over_h_2 не могут быть true одновременно");

        if (req.sweep_over_h || req.sweep_over_h_2) {
            // См. run_lle_2d -- симметрично по слотам, swap_xy не нужен.
            hSweepAxis = req.sweep_over_h ? 0 : 1;
            bool other_is_var = req.sweep_over_h ? req.sweep_over_var_2 : req.sweep_over_var;
            par_or_var = other_is_var ? 0 : 1;

            if (req.sweep_over_h) {
                if (par_or_var == 1) {
                    if (!check_param(req.param_index_2))   return fail("param_index_2 (ось Y) вне диапазона");
                    idx_axis_y = req.param_index_2;
                } else {
                    if (!check_var(req.var_sweep_index_2)) return fail("var_sweep_index_2 (ось Y) вне диапазона");
                    idx_axis_y = req.var_sweep_index_2;
                }
                idx_axis_x = 0;
            } else {
                if (par_or_var == 1) {
                    if (!check_param(req.param_index))     return fail("param_index (ось X) вне диапазона");
                    idx_axis_x = req.param_index;
                } else {
                    if (!check_var(req.var_sweep_index))   return fail("var_sweep_index (ось X) вне диапазона");
                    idx_axis_x = req.var_sweep_index;
                }
                idx_axis_y = 0;
            }
            // Требование "> 0" относится ТОЛЬКО к оси, которая свипает h: это шаг интегрирования,
            // на него ниже делит worstCaseH, и нулевой либо отрицательный шаг не считается. Вторая
            // ось — параметр или НУ, где отрицательные значения совершенно законны (свип sigma от
            // -5 до 5 к шагу отношения не имеет). Раньше условие требовало > 0 от ОБЕИХ осей и
            // заворачивало такой прогон сообщением про h. У run_lle_2d / run_ls_2d проверки нет вовсе.
            const double h_lo = req.sweep_over_h ? req.param_lo : req.param_lo_2;
            const double h_hi = req.sweep_over_h ? req.param_hi : req.param_hi_2;
            if (h_lo <= 0.0 || h_hi <= 0.0)
                return fail(std::string("h lo/hi должны быть > 0 при свипе по dt (h) (ось ")
                            + (req.sweep_over_h ? "X" : "Y") + ")");
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else if (req.sweep_over_var == req.sweep_over_var_2) {
            par_or_var = req.sweep_over_var ? 0 : 1;
            if (par_or_var == 1) {
                if (!check_param(req.param_index))   return fail("param_index (ось X) вне диапазона");
                if (!check_param(req.param_index_2)) return fail("param_index_2 (ось Y) вне диапазона");
                idx_axis_x = req.param_index;
                idx_axis_y = req.param_index_2;
            } else {
                if (!check_var(req.var_sweep_index))   return fail("var_sweep_index (ось X) вне диапазона");
                if (!check_var(req.var_sweep_index_2)) return fail("var_sweep_index_2 (ось Y) вне диапазона");
                idx_axis_x = req.var_sweep_index;
                idx_axis_y = req.var_sweep_index_2;
            }
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else if (req.sweep_over_var && !req.sweep_over_var_2) {
            par_or_var = 2;
            if (!check_var(req.var_sweep_index))  return fail("var_sweep_index (ось X) вне диапазона");
            if (!check_param(req.param_index_2))  return fail("param_index_2 (ось Y) вне диапазона");
            idx_axis_x = req.var_sweep_index;
            idx_axis_y = req.param_index_2;
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else {
            par_or_var = 2;
            if (!check_var(req.var_sweep_index_2)) return fail("var_sweep_index_2 (ось Y) вне диапазона");
            if (!check_param(req.param_index))     return fail("param_index (ось X) вне диапазона");
            idx_axis_x = req.var_sweep_index_2;
            idx_axis_y = req.param_index;
            ranges_lo_x = req.param_lo_2; ranges_hi_x = req.param_hi_2;
            ranges_lo_y = req.param_lo;   ranges_hi_y = req.param_hi;
            swap_xy = true;
            log_axis_x = req.log_scale_2; log_axis_y = req.log_scale;
        }

        int logAxisMask = (log_axis_x ? 1 : 0) | (log_axis_y ? 2 : 0);

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_bif2d_if_needed(req.krs_body, req.amountOfX, par_or_var, err)) return fail(err);

        // локальные переменные (порт hostLibrary.cu:bifurcation2D)
        const int    nPts                       = req.n_pts;
        const double tMax                       = req.t_max;
        const double h                          = req.h;
        const int    amountOfInitialConditions  = req.amountOfX;
        const std::vector<numb> ic_staged_       = to_numb(req.initial_conditions);
        const numb*   initialConditions         = ic_staged_.data();
        numb   ranges[4]                        = { (numb)ranges_lo_x, (numb)ranges_hi_x, (numb)ranges_lo_y, (numb)ranges_hi_y };
        int    indicesOfMutVars[2]              = { idx_axis_x, idx_axis_y };
        const int    writableVar                = req.writable_var;
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const std::vector<numb> values_staged_   = to_numb(req.base_values);
        const numb*   values                    = values_staged_.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const int    preScaller                 = req.pre_scaller;
        const double eps_dbscan                 = req.eps_dbscan;
        // Множители осей DBSCAN (пик / межпиковый интервал) — per-diagram.
        const double mult_peak_arg              = req.mult_peak;
        const double mult_interval_arg          = req.mult_interval;
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        constexpr int  blockSize_setup          = 32;
        constexpr int  set_precision            = 15;
        // Launch width. 32 = один варп на блок, а потолок по числу блоков на SM
        // режет занятость примерно вдвое. blockSize_setup остаётся полом бюджета.
        constexpr int  blockSize_launch         = 128;

        // dt-sweep: буфер должен вмещать худший случай (минимальный h в
        // диапазоне h-оси) -- см. run_bif1d.
        double worstCaseH = (hSweepAxis == 0) ? ranges[0] : (hSweepAxis == 1) ? ranges[2] : h;
        int amountOfPointsInBlock = (int)std::ceil(tMax / worstCaseH / preScaller);
        size_t amountOfPointsForSkip = steps_from_time_size_t(transientTime, h);

        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max/h/pre_scaller слишком малы)");

        // Буферы пиков больше НЕ длиной в траекторию: peakFinder всё равно отдаёт
        // не больше max_amount_of_peaks. "+1" — стадия межпиковых интервалов съедает
        // один пик, поэтому для полной строки нужно max_amount_of_peaks + 1 сырых.
        // При eps_interPeak_delta == 0 (дефолт) результат бит-в-бит такой же, как
        // у прежнего неограниченного скана (сжатие сдвигает ряд ровно на 1).
        const size_t peakStride    = (size_t)amountOfPointsInBlock < (size_t)max_amount_of_peaks + 1
                                   ? (size_t)amountOfPointsInBlock
                                   : (size_t)max_amount_of_peaks + 1;
        // dbscan держит в своей строке и метки кластеров, и стек обхода (см. dbscan()).
        const size_t helpfulStride = 2 * peakStride;
        const int    peakCapacity  = (int)peakStride;

        size_t total_cells = (size_t)nPts * (size_t)nPts;

        // Memory budget — порт hostLibrary.cu:930-950.
        size_t freeMemory = 0;
        if (!gpu_free_budget(0.92, freeMemory)) return fail("cudaMemGetInfo failed");

        // Бюджет больше не зависит от длины траектории вообще: сэмплы
        // потребляются на лету (PeakStream), в памяти остаются только пики.
        // tMax/h теперь влияет на время счёта, но не на число ячеек в чанке.
        size_t baseMemPerSystem = peakStride * 2 * sizeof(numb)                  // d_outPeaks + d_intervals
                                + helpfulStride * sizeof(numb)                   // d_helpfulArray
                                + 2 * sizeof(int);                               // d_amountOfPeaks + d_dbscanResult
        size_t memConstants     = (4 + (size_t)amountOfInitialConditions + (size_t)amountOfValues) * sizeof(numb) + 2 * sizeof(int);
        if (memConstants >= freeMemory) return fail("not enough GPU memory for constants");

        size_t nPtsLimiter = (freeMemory - memConstants) / baseMemPerSystem;
        if (nPtsLimiter < (size_t)blockSize_setup) nPtsLimiter = (size_t)blockSize_setup;
        if (nPtsLimiter > total_cells)             nPtsLimiter = total_cells;
        // Округления вниз до кратного blockSize_setup здесь больше нет — по той же причине, что в
        // run_bif1d: ядра сами отсекают лишние потоки, а при числе ячеек сетки < 32 округление
        // давало 0, и Run падал с сообщением про нехватку памяти, которая была ни при чём.
        if (nPtsLimiter == 0) return fail("сетка пуста (n_pts должно быть > 0)");

        size_t originalNPtsLimiter = nPtsLimiter;

        // Прогресс считается по СЧЁТУ, а не по готовым ячейкам: все ячейки
        // делают одну и ту же работу и финишируют пачкой, так что счётчик
        // готовых ячеек при сетке в одну волну прыгал бы сразу с 0 на 100%.
        // Шаг кратен CHECK_INTERVAL (тики ставятся в уже существующей проверке)
        // и выбран так, чтобы ячейка отчиталась около 64 раз за всю работу:
        // этого хватает для гладкого бара и не создаёт давки на одном адресе.
        const size_t stepsPerCell = amountOfPointsForSkip + (size_t)amountOfPointsInBlock;
        const int    progressStride = progress_stride_for(stepsPerCell);
        const double ticksPerCell   = (double)(stepsPerCell / (size_t)progressStride);
        const double ticksTotal     = (double)total_cells * ticksPerCell;

        // host buffers
        std::vector<int> h_dbscanResult(nPtsLimiter);

        // device buffers
        numb* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        numb* d_initialConditions = nullptr;
        numb* d_values            = nullptr;
        int*    d_amountOfPeaks     = nullptr;
        numb* d_outPeaks          = nullptr;
        numb* d_intervals         = nullptr;
        numb* d_helpfulArray      = nullptr;
        int*    d_dbscanResult      = nullptr;
        RunSignals sig;   // прогресс и отмена в mapped-памяти

        // Dedicated stream for traj→peak→dbscan within each chunk. Avoids
        // per-kernel cudaDeviceSynchronize, which on Windows/WDDM lets the GPU
        // downclock between launches; on the same stream kernels are ordered
        // implicitly and the GPU stays under continuous load.
        CUstream stream = nullptr;

        auto cleanup = [&]() {
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_amountOfPeaks)     cudaFree(d_amountOfPeaks);
            if (d_outPeaks)          cudaFree(d_outPeaks);
            if (d_intervals)         cudaFree(d_intervals);
            if (d_helpfulArray)      cudaFree(d_helpfulArray);
            if (d_dbscanResult)      cudaFree(d_dbscanResult);
            sig.release();
            if (stream)              cuStreamDestroy(stream);
        };

        #define BIF2D_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define BIF2D_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)
        #define BIF2D_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        BIF2D_CHECK(cudaMalloc((void**)&d_ranges,            4 * sizeof(numb)),                                           "cudaMalloc d_ranges");
        BIF2D_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  2 * sizeof(int)),                                              "cudaMalloc d_indicesOfMutVars");
        BIF2D_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)amountOfInitialConditions * sizeof(numb)),           "cudaMalloc d_initialConditions");
        BIF2D_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(numb)),                      "cudaMalloc d_values");
        BIF2D_CHECK(cudaMalloc((void**)&d_amountOfPeaks,     nPtsLimiter * sizeof(int)),                                    "cudaMalloc d_amountOfPeaks");
        BIF2D_CHECK(cudaMalloc((void**)&d_outPeaks,          nPtsLimiter * peakStride * sizeof(numb)),                    "cudaMalloc d_outPeaks");
        BIF2D_CHECK(cudaMalloc((void**)&d_intervals,         nPtsLimiter * peakStride * sizeof(numb)),                    "cudaMalloc d_intervals");
        BIF2D_CHECK(cudaMalloc((void**)&d_helpfulArray,      nPtsLimiter * helpfulStride * sizeof(numb)),                 "cudaMalloc d_helpfulArray");
        BIF2D_CHECK(cudaMalloc((void**)&d_dbscanResult,      nPtsLimiter * sizeof(int)),                                    "cudaMalloc d_dbscanResult");
        if (!sig.alloc(res.error)) { cleanup(); return res; }

        BIF2D_CHECK(cudaMemcpy(d_ranges,            ranges,            4 * sizeof(numb),                                  cudaMemcpyHostToDevice), "memcpy d_ranges");
        BIF2D_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,  2 * sizeof(int),                                     cudaMemcpyHostToDevice), "memcpy d_indices");
        BIF2D_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)amountOfInitialConditions * sizeof(numb),  cudaMemcpyHostToDevice), "memcpy d_ic");
        BIF2D_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(numb),             cudaMemcpyHostToDevice), "memcpy d_values");
        BIF2D_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        BIF2D_CHECK_CU(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");

        size_t amountOfIteration = (size_t)std::ceil((double)total_cells / (double)nPtsLimiter);

        res.n_pts      = nPts;
        res.param_lo   = req.param_lo;
        res.param_hi   = req.param_hi;
        res.param_lo_2 = req.param_lo_2;
        res.param_hi_2 = req.param_hi_2;
        res.values.assign(total_cells, 0.0);
        res.flags.assign(total_cells,  0);

        res.snapshot.values.assign(values, values + amountOfValues);
        res.snapshot.initial_conditions.assign(initialConditions,
            initialConditions + amountOfInitialConditions);
        res.snapshot.par_or_var    = par_or_var;
        res.snapshot.tMax          = tMax;
        res.snapshot.transientTime = transientTime;
        res.snapshot.h             = h;
        res.snapshot.gpu_fmad      = get_nvrtc_fmad();
        res.snapshot.preScaller    = preScaller;
        res.snapshot.eps_dbscan    = eps_dbscan;
        res.snapshot.mult_peak     = mult_peak_arg;
        res.snapshot.mult_interval = mult_interval_arg;
        res.snapshot.writableVar   = writableVar;
        res.snapshot.indexOfMutVar  = req.sweep_over_var   ? req.var_sweep_index
                                                           : req.param_index;
        res.snapshot.indexOfMutVar2 = req.sweep_over_var_2 ? req.var_sweep_index_2
                                                           : req.param_index_2;
        res.snapshot.range1_lo = req.param_lo;   res.snapshot.range1_hi = req.param_hi;
        res.snapshot.range2_lo = req.param_lo_2; res.snapshot.range2_hi = req.param_hi_2;
        res.snapshot.n_pts     = nPts;

        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            data_export::write_bif2d_config(cfg, res.snapshot);
            std::ofstream trunc(OUT_FILE_PATH); trunc.close();
        }

        // Главный цикл по чанкам (порт hostLibrary.cu:1212-1692)
        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            BIF2D_CANCEL_CHECK();
            if (req.progress) req.progress->store(float(iter) / float(amountOfIteration), std::memory_order_relaxed);
            size_t cur_limiter = originalNPtsLimiter;
            if (iter == amountOfIteration - 1)
                cur_limiter = total_cells - (originalNPtsLimiter * iter);

            // 48 KB shared — потолок блока без opt-in: широкая система (много X и a[])
            // при 128 потоках в него не влезет, и запуск упадёт на CUDA_ERROR_INVALID_VALUE.
            const size_t sharedPerThread = (size_t)ucuda_shared_stride(amountOfInitialConditions, amountOfValues) * sizeof(numb);
            int blockSize = blockSize_launch;
            while (blockSize > blockSize_setup && sharedPerThread * (size_t)blockSize > 48u * 1024u)
                blockSize /= 2;
            int gridSize  = (int)((cur_limiter + blockSize - 1) / blockSize);

            // 1. calculateDiscreteModelPeaksCUDA (dimension=2): интегрирование и поиск
            // пиков в одном ядре — траектория никуда не пишется.
            int    nPts_int                  = nPts;
            int    nPtsLimiter_int           = (int)cur_limiter;
            size_t amountOfCalculatedPoints  = iter * originalNPtsLimiter;
            size_t amountOfPointsForSkip_s   = (size_t)amountOfPointsForSkip;
            int    dimension_arg             = 2;
            numb h_arg                     = h;
            int    amountOfIC_int            = amountOfInitialConditions;
            int    amountOfValues_int        = amountOfValues;
            size_t amountOfIterations_arg    = (size_t)amountOfPointsInBlock;
            int    preScaller_int            = preScaller;
            int    writableVar_int           = writableVar;
            numb maxValue_arg              = maxValue;
            bool   par_or_var_arg            = (par_or_var != 0);  // true=param, false=IC (runtime hint, kernel использует compile-time макрос)
            int    hSweepAxis_arg            = hSweepAxis;
            numb transientTime_arg         = transientTime;
            numb tMax_arg                  = tMax;
            int    logAxisMask_arg           = logAxisMask;
            size_t peakStride_arg            = peakStride;
            int*   d_cancel_arg              = sig.cancelArg();
            int*   d_progress_arg            = sig.progressArg();
            int    progressStride_arg        = progressStride;
            int    peakCapacity_arg          = peakCapacity;

            void* args_fused[] = {
                &nPts_int,
                &nPtsLimiter_int,
                &amountOfCalculatedPoints,
                &amountOfPointsForSkip_s,
                &dimension_arg,
                &d_ranges,
                &h_arg,
                &d_indicesOfMutVars,
                &d_initialConditions,
                &amountOfIC_int,
                &d_values,
                &amountOfValues_int,
                &amountOfIterations_arg,
                &preScaller_int,
                &writableVar_int,
                &maxValue_arg,
                &d_outPeaks,
                &d_intervals,
                &d_amountOfPeaks,
                &par_or_var_arg,
                &hSweepAxis_arg,
                &transientTime_arg,
                &tMax_arg,
                &logAxisMask_arg,
                &peakStride_arg,
                &peakCapacity_arg,
                &d_cancel_arg,
                &d_progress_arg,
                &progressStride_arg
            };
            sig.resetTicks();
            unsigned int shared_traj = (unsigned int)(ucuda_shared_stride(amountOfInitialConditions, amountOfValues) * sizeof(numb) * blockSize);
            BIF2D_CHECK_CU(cuLaunchKernel(cached_bif2d.kernel_fused,
                                          gridSize, 1, 1, blockSize, 1, 1,
                                          shared_traj, stream, args_fused, nullptr),
                           "cuLaunchKernel(bif2d traj+peaks)");

            // 3. dbscanCUDA — same stream, ordered after peak.
            // Множители осей признаков передаём явно: при запуске через driver
            // API дефолты из объявления не подставляются (см. cudaLibrary.cuh).
            numb eps_arg      = eps_dbscan;
            numb mult_pk_arg  = mult_peak_arg;
            numb mult_int_arg = mult_interval_arg;
            size_t helpfulStride_arg = helpfulStride;
            // sizeOfBlock у dbscanCUDA остаётся ради legacy-вызовов; здесь раскладку
            // задают peakStride/helpfulStride, и этот аргумент не читается.
            size_t sizeOfBlock_s = peakStride;
            void* args_dbscan[] = {
                &d_outPeaks,          // кластеризуем пики, а не сырую траекторию
                &sizeOfBlock_s,
                &nPtsLimiter_int,
                &d_amountOfPeaks,
                &d_intervals,
                &d_helpfulArray,
                &eps_arg,
                &d_dbscanResult,
                &mult_pk_arg,
                &mult_int_arg,
                &peakStride_arg,
                &helpfulStride_arg
            };
            BIF2D_CHECK_CU(cuLaunchKernel(cached_bif2d.kernel_dbscan,
                                          gridSize, 1, 1, blockSize, 1, 1,
                                          0, stream, args_dbscan, nullptr),
                           "cuLaunchKernel(bif2d dbscan)");

            // D2H: только d_dbscanResult (число кластеров = период).
            // Async on the same stream + single sync — keeps the GPU continuously
            // loaded across the whole chunk instead of inserting 3 sync gaps.
            if (!wait_with_signals(stream, sig, req.cancel, req.progress,
                                   (double)(originalNPtsLimiter * iter) * ticksPerCell,
                                   ticksTotal, res.error)) { cleanup(); return res; }
            BIF2D_CHECK(cudaStreamSynchronize(stream), "sync stream before host loop");
            BIF2D_CANCEL_CHECK();

            // Копируем ПОСЛЕ опроса, а не до него: device -> pageable копирование
            // асинхронным не бывает (h_dbscanResult — std::vector), драйвер возвращает
            // управление лишь после завершения копии, т.е. всей очереди потока. Стоя
            // перед циклом, оно блокировало хост до конца ядра, и цикл не выполнялся
            // ни разу — ни прогресса, ни отмены.
            BIF2D_CHECK(cudaMemcpy(h_dbscanResult.data(), d_dbscanResult,
                                   cur_limiter * sizeof(int), cudaMemcpyDeviceToHost),
                        "memcpy h_dbscanResult");

            for (size_t k = 0; k < cur_limiter; ++k) {
                size_t kernel_idx = originalNPtsLimiter * iter + k;
                int    period     = h_dbscanResult[k];
                size_t out_idx;
                if (swap_xy) {
                    size_t kix = kernel_idx % (size_t)nPts;
                    size_t kiy = kernel_idx / (size_t)nPts;
                    out_idx = kix * (size_t)nPts + kiy;
                } else {
                    out_idx = kernel_idx;
                }
                // period из dbscanCUDA — сырой REGIME_*/период: -1 = fixed
                // point, 0 = unbound, N > 0 = число кластеров пиков. Раньше
                // unbound (0) получал flag 1, т.е. считался валидной ячейкой и
                // попадал в автошкалу colormap'а как «период 0».
                res.values[out_idx] = (double)period;
                res.flags[out_idx]  = period;
            }
        }

        // Write the data file once, AFTER the chunked loop — see data_export
        // header for why the grid layout is centralized there.
        if (!OUT_FILE_PATH.empty()) {
            std::ofstream out(OUT_FILE_PATH);
            if (out.is_open()) {
                out << std::setprecision(set_precision);
                data_export::write_bif2d_grid(out, nPts, res.values.data());
            }
        }

        // Авто-нормализация colormap. Верх шкалы — по осцилляционным ячейкам (там значение =
        // период), а низ опускаем до кода режима, если такие ячейки на карте есть: -1 (fixed point)
        // и 0 (unbound) — это не «период -1/0», а отдельные состояния, и в res.values они лежат как
        // есть. Раньше в шкалу шли только осцилляции, поэтому vmin был >= 1 и оба режима прижимались
        // к самому дну — неотличимо от периода 1.
        double vmin =  std::numeric_limits<double>::infinity();
        double vmax = -std::numeric_limits<double>::infinity();
        bool has_fp = false, has_unbound = false;
        for (size_t k = 0; k < total_cells; ++k) {
            const int f = res.flags[k];
            if (f == REGIME_FIXED_POINT) { has_fp      = true; continue; }
            if (f == REGIME_UNBOUND)     { has_unbound = true; continue; }
            if (!regime_is_oscillation(f)) continue;
            double v = res.values[k];
            if (!std::isfinite(v)) continue;
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
        res.min_val = std::isfinite(vmin) ? vmin : 0.0;
        res.max_val = std::isfinite(vmax) ? vmax : 0.0;
        // Приоритет у fixed point: он ниже unbound, и одной нижней границы
        // хватает на оба режима сразу.
        if      (has_fp)      res.min_val = -1.0;
        else if (has_unbound) res.min_val =  0.0;

        cleanup();
        #undef BIF2D_CHECK
        #undef BIF2D_CHECK_CU
        res.ok = true;
        return res;
    }