#include "engine_impl.h"



    // run_lle_1d — порт NonLinAnal::LLE1D из hostLibrary.cu, той же diff-friendly стратегией, что
    // run_bif1d (оригинальные имена слева, req-имена справа). [ADAPT] — отличия от NonLinAnal:
    // <<<>>> → cuLaunchKernel (NVRTC-модуль); gpuErrorCheck → локальный LLE_CHECK с cleanup'ом,
    // без exit(); OUT_FILE_PATH — req.csv_output_path (пусто = без файла); результат пишется в
    // LLE1DResult (память + опц. CSV).
    LLE1DResult ParametricEngine::Impl::run_lle_1d(const LLE1DRequest& req) {
        LLE1DResult res;
        auto fail = [&](const std::string& msg) -> LLE1DResult& { res.error = msg; return res; };

        // валидация
        if (req.krs_body.empty())                                   return fail("krs_body пуст");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX вне [1," + std::to_string(kMaxAmountOfX) + "]");
        if ((int)req.initial_conditions.size() != req.amountOfX)    return fail("initial_conditions.size() != amountOfX");
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
        if (req.n_pts <= 0)         return fail("n_pts должно быть > 0");
        if (req.h <= 0.0)           return fail("h должно быть > 0");
        if (req.t_max <= 0.0)       return fail("t_max должно быть > 0");
        if (req.transient_time < 0) return fail("transient_time должно быть >= 0");
        if (req.NT <= 0.0)          return fail("NT должно быть > 0");
        if (req.eps <= 0.0)         return fail("eps должно быть > 0");
        if (req.log_scale && !(req.param_lo > 0.0 && req.param_hi > 0.0))
            return fail("log scale требует param lo/hi > 0");

        // Continuation: точки выстроены в цепочку, поэтому IC-свип несовместим
        // (как в run_bif1d). h-свип и log-сетка поддержаны на обоих устройствах.
        if (req.continuation) {
            if (req.sweep_over_var) return fail("continuation требует param-sweep, не IC-sweep");
            return req.use_cpu ? run_lle1d_cpu(req, /*continuation*/ true)
                               : run_lle1d_continuation_gpu(req);
        }
        // Классический свип на CPU — для счёта без GPU и для сверки ядра с GPU.
        if (req.use_cpu) {
            if (req.sweep_over_var) return fail("CPU-ветка поддерживает только param-sweep");
            return run_lle1d_cpu(req, /*continuation*/ false);
        }

        // dt-sweep: t_max/transient_time/NT остаются фиксированными (см.
        // sweep_over_h в analysis_session.h), число шагов пересчитывается на GPU
        // из h per-thread -- kernel получает hSweepAxis вместо compile-time флага.
        const int hSweepAxis = req.sweep_over_h ? 0 : -1;
        const int logAxisMask = req.log_scale ? 1 : 0;

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_lle_if_needed(req.krs_body, req.amountOfX,
                                   req.sweep_over_var ? 0 : 1, err)) return fail(err);

        // ПОРТ NonLinAnal::LLE1D (hostLibrary.cu:2261-2511)
        const double tMax                       = req.t_max;
        const double NT                         = req.NT;
        const int    nPts                       = req.n_pts;
        const double h                          = req.h;
        const double eps                        = req.eps;
        const int    amountOfInitialConditions  = req.amountOfX;
        const std::vector<numb> ic_staged_       = to_numb(req.initial_conditions);
        const numb*   initialConditions         = ic_staged_.data();
        numb   ranges[2]                        = { (numb)req.param_lo, (numb)req.param_hi };
        // Sweep target: см. bif1d. При IC-свипе индекс — 0-based в localX.
        int    indicesOfMutVars[1]              = { req.sweep_over_var
                                                    ? req.var_sweep_index
                                                    : req.param_index };
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const std::vector<numb> values_staged_   = to_numb(req.base_values);
        const numb*   values                    = values_staged_.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        constexpr int blockSize_setup = 32;
        constexpr int set_precision   = 15;

        // amountOfPointsInBlock = tMax / NT — число NT-блоков интегрирования.
        int amountOfPointsInBlock = (int)(tMax / NT);
        size_t amountOfPointsForSkip = steps_from_time_size_t(transientTime, h);

        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max / NT слишком малы)");

        // Memory budget — мирор NonLinAnal LLE1D:2291-2299 (консервативно).
        size_t freeMemory = 0;
        if (!gpu_free_budget(0.9, freeMemory)) return fail("cudaMemGetInfo failed");

        // На точку реально выделяется ТОЛЬКО d_lleResult (одно numb): траектория
        // не хранится, ядро зовёт цикл интегрирования с data = nullptr. Прежняя
        // формула делила память на длину блока, которой здесь нет.
        size_t nPtsLimiter = freeMemory / sizeof(numb);
        if (nPtsLimiter == 0)            nPtsLimiter = (size_t)blockSize_setup;
        if (nPtsLimiter > (size_t)nPts)  nPtsLimiter = (size_t)nPts;
        size_t originalNPtsLimiter = nPtsLimiter;

        std::vector<numb> h_lleResult(nPtsLimiter);

        numb* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        numb* d_initialConditions = nullptr;
        numb* d_values            = nullptr;
        numb* d_lleResult         = nullptr;

        RunSignals sig;   // прогресс и отмена в mapped-памяти
        auto cleanup = [&]() {
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_lleResult)         cudaFree(d_lleResult);
        };
            sig.release();

        #define LLE_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define LLE_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)
        #define LLE_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        LLE_CHECK(cudaMalloc((void**)&d_ranges,            2 * sizeof(numb)),                                "cudaMalloc d_ranges");
        LLE_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  1 * sizeof(int)),                                   "cudaMalloc d_indicesOfMutVars");
        LLE_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)amountOfInitialConditions * sizeof(numb)),"cudaMalloc d_initialConditions");
        LLE_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(numb)),           "cudaMalloc d_values");
        LLE_CHECK(cudaMalloc((void**)&d_lleResult,         nPtsLimiter * sizeof(numb)),                      "cudaMalloc d_lleResult");
        if (!sig.alloc(res.error)) { cleanup(); return res; }
        // Шагов на точку: транзиент плюс NT-блоки по NT/h шагов.
        // Тики ставит только ведущая траектория (см. ядро), поэтому копии не считаем.
        const size_t stepsPerPoint  = amountOfPointsForSkip + (size_t)amountOfPointsInBlock * steps_from_time_size_t(NT, h);
        const int    progressStride = progress_stride_for(stepsPerPoint);
        const double ticksPerPoint  = (double)(stepsPerPoint / (size_t)progressStride);
        const double ticksTotal     = (double)nPts * ticksPerPoint;

        LLE_CHECK(cudaMemcpy(d_ranges,            ranges,            2 * sizeof(numb),                                 cudaMemcpyHostToDevice), "memcpy d_ranges");
        LLE_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,  1 * sizeof(int),                                    cudaMemcpyHostToDevice), "memcpy d_indices");
        LLE_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)amountOfInitialConditions * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_ic");
        LLE_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(numb),            cudaMemcpyHostToDevice), "memcpy d_values");
        LLE_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)nPts / (double)nPtsLimiter);

        // Snapshot CSV-relevant request fields for GUI right-click export.
        res.snapshot.values.assign(values, values + amountOfValues);
        res.snapshot.initial_conditions.assign(initialConditions,
            initialConditions + amountOfInitialConditions);
        res.snapshot.tMax          = tMax;
        res.snapshot.NT            = NT;
        res.snapshot.transientTime = transientTime;
        res.snapshot.h             = h;
        res.snapshot.gpu_fmad      = get_nvrtc_fmad();
        res.snapshot.eps           = eps;
        res.snapshot.indexOfMutVar = indicesOfMutVars[0];
        res.snapshot.range_lo      = ranges[0];
        res.snapshot.range_hi      = ranges[1];

        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            data_export::write_lle1d_config(cfg, res.snapshot);
            std::ofstream trunc(OUT_FILE_PATH); trunc.close();
        }

        // Результат-аккумулятор
        res.n_pts    = nPts;
        res.param_lo = ranges[0];
        res.param_hi = ranges[1];
        res.lyapunov.assign(nPts, 0.0);
        res.flags.assign(nPts, 0);

        // Главный цикл (порт NonLinAnal LLE1D:2403-2496)
        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            LLE_CANCEL_CHECK();
            if (req.progress) req.progress->store(float(iter) / float(amountOfIteration), std::memory_order_relaxed);
            if (iter == amountOfIteration - 1)
                nPtsLimiter = nPts - (originalNPtsLimiter * iter);

            // blockSize по формуле NonLinAnal (hostLibrary.cu:2419), cap=32
            int blockSize = (int)std::ceil((1024.0 * 32.0) / ((double)(3 * amountOfInitialConditions + amountOfValues) * (double)sizeof(numb)));
            if (blockSize < 1) blockSize = 1;
            if (blockSize > blockSize_setup) blockSize = blockSize_setup;
            int gridSize = (int)((nPtsLimiter + blockSize - 1) / blockSize);

            // Аргументы LLEKernelCUDA (cudaLibrary.cu:2379)
            int    nPts_arg                  = nPts;
            int    nPtsLimiter_arg           = (int)nPtsLimiter;
            numb NT_arg                    = NT;
            numb tMax_arg                  = tMax;
            int    sizeOfBlock_arg           = amountOfPointsInBlock;
            int    amountOfCalculatedPoints  = (int)(iter * originalNPtsLimiter);
            size_t amountOfPointsForSkip_arg = amountOfPointsForSkip;
            int    dimension_arg             = 1;
            numb h_arg                     = h;
            numb eps_arg                   = eps;
            int    amountOfIC_arg            = amountOfInitialConditions;
            int    amountOfValues_arg        = amountOfValues;
            int    amountOfIterations_arg    = (int)(tMax / NT);
            int    preScaller_arg            = 1;
            int    writableVar_arg           = 0;
            numb maxValue_arg              = maxValue;
            int    hSweepAxis_arg            = hSweepAxis;
            numb transientTime_arg         = transientTime;
            int    logAxisMask_arg           = logAxisMask;

            int*   d_cancel_arg       = sig.cancelArg();
            int*   d_progress_arg     = sig.progressArg();
            int    progressStride_arg = progressStride;
            sig.resetTicks();

            void* args[] = {
                &nPts_arg,
                &nPtsLimiter_arg,
                &NT_arg,
                &tMax_arg,
                &sizeOfBlock_arg,
                &amountOfCalculatedPoints,
                &amountOfPointsForSkip_arg,
                &dimension_arg,
                &d_ranges,
                &h_arg,
                &eps_arg,
                &d_indicesOfMutVars,
                &d_initialConditions,
                &amountOfIC_arg,
                &d_values,
                &amountOfValues_arg,
                &amountOfIterations_arg,
                &preScaller_arg,
                &writableVar_arg,
                &maxValue_arg,
                &d_lleResult,
                &hSweepAxis_arg,
                &transientTime_arg,
                &logAxisMask_arg,
                &d_cancel_arg, &d_progress_arg, &progressStride_arg
            };

            // Shared = (3 * amountOfIC + amountOfValues) * sizeof(numb) * blockSize
            unsigned int shared = (unsigned int)((3 * amountOfInitialConditions + amountOfValues)
                                                 * sizeof(numb) * blockSize);

            LLE_CHECK_CU(cuLaunchKernel(cached_lle.kernel_lle,
                                        gridSize, 1, 1, blockSize, 1, 1,
                                        shared, nullptr, args, nullptr),
                         "cuLaunchKernel(lle)");
            if (!wait_with_signals(0, sig, req.cancel, req.progress,
                                   (double)(originalNPtsLimiter * iter) * ticksPerPoint,
                                   ticksTotal, res.error)) { cleanup(); return res; }
            LLE_CHECK(cudaDeviceSynchronize(), "sync after lle");

            LLE_CHECK(cudaMemcpy(h_lleResult.data(), d_lleResult, nPtsLimiter * sizeof(numb), cudaMemcpyDeviceToHost),
                      "memcpy h_lleResult");
            LLE_CHECK(cudaDeviceSynchronize(), "sync after D2H");

            // Аккумулируем + опциональный CSV (порт NonLinAnal LLE1D:2478-2492)
            std::ofstream out;
            if (!OUT_FILE_PATH.empty()) {
                out.open(OUT_FILE_PATH, std::ios::app);
                if (out.is_open()) out << std::setprecision(set_precision);
            }
            for (size_t k = 0; k < nPtsLimiter; ++k) {
                size_t global_idx = originalNPtsLimiter * iter + k;
                double param_val  = req.log_scale ? getValueByIdx_log_local(global_idx, nPts, ranges[0], ranges[1])
                                                   : getValueByIdx_local(global_idx, nPts, ranges[0], ranges[1]);
                double v          = h_lleResult[k];

                // 999 / -999 — спец-флаги из kernel'а (нет аттрактора / разошлось). Наружу отдаём
                // NaN, а не сырой sentinel: 999 — легитимное по типу значение λ, поэтому любой
                // потребитель, забывший сверить flags[], молча рисовал выброс на 999. NaN
                // отсекается через !isfinite (и в GUI, и в min/max), т.е. безопасен по умолчанию;
                // flags[] остаётся источником истины о ПРИЧИНЕ отсутствия точки.
                const bool diverged = (v == 999.0 || v == -999.0);
                if (diverged) v = std::numeric_limits<double>::quiet_NaN();

                res.lyapunov[global_idx] = v;
                // LLE не детектирует fixed point (ветка в LLEKernelCUDA выключена),
                // поэтому единственный «плохой» код — REGIME_UNBOUND.
                res.flags[global_idx] = diverged ? REGIME_UNBOUND : REGIME_OSCILLATION;

                if (out.is_open()) data_export::write_lle1d_row(out, param_val, v);
            }
            if (out.is_open()) out.close();
        }

        cleanup();
        #undef LLE_CHECK
        #undef LLE_CHECK_CU
        res.ok = true;
        return res;
    }


    // run_lle_2d — λ(p1, p2) на квадратной сетке, порт NonLinAnal::LLE2D на NVRTC-engine.
    // Отличия от LLE1D: ranges[4], indicesOfMutVars[2], dimension=2, chunking по ячейкам
    // (n_pts × n_pts может не влезть в память за раз), результат — плоский row-major массив.
    // par_or_var (compile-time): mixed_mode=true → 2; иначе оба свипа одного типа — 1 если обе оси
    // param, 0 если обе IC. Смешанные комбинации без mixed_mode не поддержаны (validator
    // отказывает — kernel-ветки под это нет).
    LLE2DResult ParametricEngine::Impl::run_lle_2d(const LLE2DRequest& req) {
        LLE2DResult res;
        auto fail = [&](const std::string& msg) -> LLE2DResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                   return fail("krs_body пуст");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX вне допустимого диапазона");
        if ((int)req.initial_conditions.size() != req.amountOfX)    return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)       return fail("base_values слишком много");
        if (req.n_pts <= 0)         return fail("n_pts должно быть > 0");
        if (req.h <= 0.0)           return fail("h должно быть > 0");
        if (req.t_max <= 0.0)       return fail("t_max должно быть > 0");
        if (req.transient_time < 0) return fail("transient_time должно быть >= 0");
        if (req.NT <= 0.0)          return fail("NT должно быть > 0");
        if (req.eps <= 0.0)         return fail("eps должно быть > 0");

        // par_or_var — compile-time. Логика маппинга см. parametric_engine.h.
        // Дополнительно нужен swap_xy: если X=param, Y=IC, передаём в kernel с
        // X↔Y и потом транспонируем результат на хосте.
        auto check_param = [&](int p1based) -> bool {
            return p1based >= 0 && p1based < (int)req.base_values.size();
        };
        auto check_var = [&](int v0based) -> bool {
            return v0based >= 0 && v0based < req.amountOfX;
        };

        int par_or_var;
        int  idx_axis_x, idx_axis_y;   // что передаём в indicesOfMutVars[0/1]
        double ranges_lo_x, ranges_hi_x, ranges_lo_y, ranges_hi_y;
        bool swap_xy = false;
        int  hSweepAxis = -1;          // -1=off, 0=X свипует h, 1=Y свипует h
        // log_axis_x/y следуют тому же swap, что и ranges_lo_x/y ниже -- в
        // кернел они идут как биты того же физического слота (0=X,1=Y), а не
        // пользовательской оси. Валидация (lo/hi>0) проверяется по
        // ПОЛЬЗОВАТЕЛЬСКИМ req.log_scale/_2 отдельно, ниже, до свопа.
        bool log_axis_x = req.log_scale, log_axis_y = req.log_scale_2;

        if (req.log_scale && !(req.param_lo > 0.0 && req.param_hi > 0.0))
            return fail("log scale требует param lo/hi > 0 (ось X)");
        if (req.log_scale_2 && !(req.param_lo_2 > 0.0 && req.param_hi_2 > 0.0))
            return fail("log scale требует param lo/hi > 0 (ось Y)");

        if (req.sweep_over_h && req.sweep_over_h_2)
            return fail("sweep_over_h и sweep_over_h_2 не могут быть true одновременно");

        if (req.sweep_over_h || req.sweep_over_h_2) {
            // Ровно одна ось — h, другая param либо IC. Кернел-слоты X/Y совпадают с
            // пользовательскими напрямую, swap_xy тут не нужен: в смешанном param/IC случае ниже
            // swap существует только потому, что ветка par_or_var==2 захардкожена под одну
            // конкретную пару слотов, а здесь par_or_var симметричен по слотам.
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
                idx_axis_x = 0; // dummy -- слот X пропускается в кернеле (i == hSweepAxis)
            } else {
                if (par_or_var == 1) {
                    if (!check_param(req.param_index))     return fail("param_index (ось X) вне диапазона");
                    idx_axis_x = req.param_index;
                } else {
                    if (!check_var(req.var_sweep_index))   return fail("var_sweep_index (ось X) вне диапазона");
                    idx_axis_x = req.var_sweep_index;
                }
                idx_axis_y = 0; // dummy
            }
            // ranges для h-оси переиспользуют тот же param_lo/hi(_2), что и для
            // param/IC -- семантика диапазона просто меняется на "значения h".
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else if (req.sweep_over_var == req.sweep_over_var_2) {
            par_or_var = req.sweep_over_var ? 0 : 1;
            if (par_or_var == 1) {
                if (!check_param(req.param_index))    return fail("param_index (ось X) вне диапазона");
                if (!check_param(req.param_index_2))  return fail("param_index_2 (ось Y) вне диапазона");
                idx_axis_x = req.param_index;
                idx_axis_y = req.param_index_2;
            } else {
                if (!check_var(req.var_sweep_index))    return fail("var_sweep_index (ось X) вне диапазона");
                if (!check_var(req.var_sweep_index_2))  return fail("var_sweep_index_2 (ось Y) вне диапазона");
                idx_axis_x = req.var_sweep_index;
                idx_axis_y = req.var_sweep_index_2;
            }
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else if (req.sweep_over_var && !req.sweep_over_var_2) {
            // X=IC, Y=param — нативно соответствует ветке par_or_var=2 kernel'а.
            par_or_var = 2;
            if (!check_var(req.var_sweep_index))   return fail("var_sweep_index (ось X) вне диапазона");
            if (!check_param(req.param_index_2))   return fail("param_index_2 (ось Y) вне диапазона");
            idx_axis_x = req.var_sweep_index;
            idx_axis_y = req.param_index_2;
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else {
            // X=param, Y=IC — kernel напрямую не поддерживает, свопаем оси
            // под капотом и транспонируем результат при выгрузке.
            par_or_var = 2;
            if (!check_var(req.var_sweep_index_2)) return fail("var_sweep_index_2 (ось Y) вне диапазона");
            if (!check_param(req.param_index))     return fail("param_index (ось X) вне диапазона");
            // В kernel: ось 1 (IC) = пользовательский Y, ось 2 (param) = X.
            idx_axis_x = req.var_sweep_index_2;
            idx_axis_y = req.param_index;
            ranges_lo_x = req.param_lo_2; ranges_hi_x = req.param_hi_2;
            ranges_lo_y = req.param_lo;   ranges_hi_y = req.param_hi;
            swap_xy = true;
            log_axis_x = req.log_scale_2; log_axis_y = req.log_scale;  // тот же своп, что и ranges выше
        }

        int logAxisMask = (log_axis_x ? 1 : 0) | (log_axis_y ? 2 : 0);

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_lle_2d_if_needed(req.krs_body, req.amountOfX, par_or_var, err)) return fail(err);

        const double tMax                       = req.t_max;
        const double NT                         = req.NT;
        const int    nPts                       = req.n_pts;       // сторона сетки
        const double h                          = req.h;
        const double eps                        = req.eps;
        const int    amountOfInitialConditions  = req.amountOfX;
        const std::vector<numb> ic_staged_       = to_numb(req.initial_conditions);
        const numb*   initialConditions         = ic_staged_.data();
        numb   ranges[4]                        = { (numb)ranges_lo_x, (numb)ranges_hi_x,
                                                    (numb)ranges_lo_y, (numb)ranges_hi_y };
        int    indicesOfMutVars[2]              = { idx_axis_x, idx_axis_y };
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const std::vector<numb> values_staged_   = to_numb(req.base_values);
        const numb*   values                    = values_staged_.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        constexpr int blockSize_setup = 32;
        constexpr int set_precision   = 15;

        int amountOfPointsInBlock = (int)(tMax / NT);
        size_t amountOfPointsForSkip = steps_from_time_size_t(transientTime, h);

        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max / NT слишком малы)");

        size_t total_cells = (size_t)nPts * (size_t)nPts;

        // Memory budget — мирор NonLinAnal LLE2D (hostLibrary.cu:2535-2547).
        size_t freeMemory = 0;
        if (!gpu_free_budget(0.9, freeMemory)) return fail("cudaMemGetInfo failed");

        // На точку реально выделяется ТОЛЬКО d_lleResult (одно numb): траектория
        // не хранится, ядро зовёт цикл интегрирования с data = nullptr. Прежняя
        // формула делила память на длину блока, которой здесь нет.
        size_t nPtsLimiter = freeMemory / sizeof(numb);
        if (nPtsLimiter == 0)                  nPtsLimiter = (size_t)blockSize_setup;
        if (nPtsLimiter > total_cells)         nPtsLimiter = total_cells;
        size_t originalNPtsLimiter = nPtsLimiter;

        std::vector<numb> h_lleResult(nPtsLimiter);

        numb* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        numb* d_initialConditions = nullptr;
        numb* d_values            = nullptr;
        numb* d_lleResult         = nullptr;

        RunSignals sig;   // прогресс и отмена в mapped-памяти
        auto cleanup = [&]() {
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_lleResult)         cudaFree(d_lleResult);
        };
            sig.release();

        #define LLE2_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define LLE2_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)
        #define LLE2_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        LLE2_CHECK(cudaMalloc((void**)&d_ranges,            4 * sizeof(numb)),                                "cudaMalloc d_ranges");
        LLE2_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  2 * sizeof(int)),                                   "cudaMalloc d_indicesOfMutVars");
        LLE2_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)amountOfInitialConditions * sizeof(numb)),"cudaMalloc d_initialConditions");
        LLE2_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(numb)),           "cudaMalloc d_values");
        LLE2_CHECK(cudaMalloc((void**)&d_lleResult,         nPtsLimiter * sizeof(numb)),                      "cudaMalloc d_lleResult");
        if (!sig.alloc(res.error)) { cleanup(); return res; }
        // Шагов на точку: транзиент плюс NT-блоки по NT/h шагов.
        // Тики ставит только ведущая траектория (см. ядро), поэтому копии не считаем.
        const size_t stepsPerPoint  = amountOfPointsForSkip + (size_t)amountOfPointsInBlock * steps_from_time_size_t(NT, h);
        const int    progressStride = progress_stride_for(stepsPerPoint);
        const double ticksPerPoint  = (double)(stepsPerPoint / (size_t)progressStride);
        const double ticksTotal     = (double)total_cells * ticksPerPoint;

        LLE2_CHECK(cudaMemcpy(d_ranges,            ranges,            4 * sizeof(numb),                                 cudaMemcpyHostToDevice), "memcpy d_ranges");
        LLE2_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,  2 * sizeof(int),                                    cudaMemcpyHostToDevice), "memcpy d_indices");
        LLE2_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)amountOfInitialConditions * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_ic");
        LLE2_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(numb),            cudaMemcpyHostToDevice), "memcpy d_values");
        LLE2_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)total_cells / (double)nPtsLimiter);

        // В result диапазоны храним всегда «пользовательские» (X = первая ось
        // запроса), не свопнутые ranges[] — чтобы GUI рисовал оси правильно
        // независимо от того, делал ли engine внутренний свап.
        res.n_pts      = nPts;
        res.param_lo   = req.param_lo;
        res.param_hi   = req.param_hi;
        res.param_lo_2 = req.param_lo_2;
        res.param_hi_2 = req.param_hi_2;
        res.values.assign(total_cells, 0.0);
        res.flags.assign(total_cells, 0);

        // Snapshot of CSV-relevant fields in USER ordering — engine + GUI
        // share the writer, so both files agree on axis ordering even when
        // the engine has internally swapped X/Y for kernel dispatch.
        res.snapshot.values.assign(values, values + amountOfValues);
        res.snapshot.initial_conditions.assign(initialConditions,
            initialConditions + amountOfInitialConditions);
        res.snapshot.par_or_var    = par_or_var;
        res.snapshot.tMax          = tMax;
        res.snapshot.NT            = NT;
        res.snapshot.transientTime = transientTime;
        res.snapshot.h             = h;
        res.snapshot.gpu_fmad      = get_nvrtc_fmad();
        res.snapshot.eps           = eps;
        res.snapshot.indexOfMutVar  = req.sweep_over_var   ? req.var_sweep_index
                                                           : req.param_index;
        res.snapshot.indexOfMutVar2 = req.sweep_over_var_2 ? req.var_sweep_index_2
                                                           : req.param_index_2;
        res.snapshot.range1_lo = req.param_lo;   res.snapshot.range1_hi = req.param_hi;
        res.snapshot.range2_lo = req.param_lo_2; res.snapshot.range2_hi = req.param_hi_2;
        res.snapshot.n_pts     = nPts;

        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            data_export::write_lle2d_config(cfg, res.snapshot);
            std::ofstream trunc(OUT_FILE_PATH); trunc.close();
        }

        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            LLE2_CANCEL_CHECK();
            if (req.progress) req.progress->store(float(iter) / float(amountOfIteration), std::memory_order_relaxed);
            size_t cur_limiter = originalNPtsLimiter;
            if (iter == amountOfIteration - 1)
                cur_limiter = total_cells - (originalNPtsLimiter * iter);

            int blockSize = (int)std::ceil((1024.0 * 32.0) / ((double)(3 * amountOfInitialConditions + amountOfValues) * (double)sizeof(numb)));
            if (blockSize < 1) blockSize = 1;
            if (blockSize > blockSize_setup) blockSize = blockSize_setup;
            int gridSize = (int)((cur_limiter + blockSize - 1) / blockSize);

            // Аргументы LLEKernelCUDA — те же 21 параметр, что и в LLE1D,
            // но dimension=2 и амбулатура индекса nPts остаётся "сторона сетки".
            int    nPts_arg                  = nPts;
            int    nPtsLimiter_arg           = (int)cur_limiter;
            numb NT_arg                    = NT;
            numb tMax_arg                  = tMax;
            int    sizeOfBlock_arg           = amountOfPointsInBlock;
            int    amountOfCalculatedPoints  = (int)(iter * originalNPtsLimiter);
            size_t amountOfPointsForSkip_arg = amountOfPointsForSkip;
            int    dimension_arg             = 2;
            numb h_arg                     = h;
            numb eps_arg                   = eps;
            int    amountOfIC_arg            = amountOfInitialConditions;
            int    amountOfValues_arg        = amountOfValues;
            int    amountOfIterations_arg    = (int)(tMax / NT);
            int    preScaller_arg            = 1;
            int    writableVar_arg           = 0;
            numb maxValue_arg              = maxValue;
            int    hSweepAxis_arg            = hSweepAxis;
            numb transientTime_arg         = transientTime;
            int    logAxisMask_arg           = logAxisMask;

            int*   d_cancel_arg       = sig.cancelArg();
            int*   d_progress_arg     = sig.progressArg();
            int    progressStride_arg = progressStride;
            sig.resetTicks();

            void* args[] = {
                &nPts_arg,
                &nPtsLimiter_arg,
                &NT_arg,
                &tMax_arg,
                &sizeOfBlock_arg,
                &amountOfCalculatedPoints,
                &amountOfPointsForSkip_arg,
                &dimension_arg,
                &d_ranges,
                &h_arg,
                &eps_arg,
                &d_indicesOfMutVars,
                &d_initialConditions,
                &amountOfIC_arg,
                &d_values,
                &amountOfValues_arg,
                &amountOfIterations_arg,
                &preScaller_arg,
                &writableVar_arg,
                &maxValue_arg,
                &d_lleResult,
                &hSweepAxis_arg,
                &transientTime_arg,
                &logAxisMask_arg,
                &d_cancel_arg, &d_progress_arg, &progressStride_arg
            };

            unsigned int shared = (unsigned int)((3 * amountOfInitialConditions + amountOfValues)
                                                 * sizeof(numb) * blockSize);

            LLE2_CHECK_CU(cuLaunchKernel(cached_lle_2d.kernel_lle,
                                         gridSize, 1, 1, blockSize, 1, 1,
                                         shared, nullptr, args, nullptr),
                          "cuLaunchKernel(lle2d)");
            if (!wait_with_signals(0, sig, req.cancel, req.progress,
                                   (double)(originalNPtsLimiter * iter) * ticksPerPoint,
                                   ticksTotal, res.error)) { cleanup(); return res; }
            LLE2_CHECK(cudaDeviceSynchronize(), "sync after lle2d");

            LLE2_CHECK(cudaMemcpy(h_lleResult.data(), d_lleResult, cur_limiter * sizeof(numb), cudaMemcpyDeviceToHost),
                       "memcpy h_lleResult");
            LLE2_CHECK(cudaDeviceSynchronize(), "sync after D2H");

            for (size_t k = 0; k < cur_limiter; ++k) {
                size_t kernel_idx = originalNPtsLimiter * iter + k;
                double v          = h_lleResult[k];
                // Транспонирование при свопе: kernel_idx = kix + n*kiy, в
                // системе пользователя ix=kiy, iy=kix → user_idx = kix*n + kiy.
                size_t out_idx;
                if (swap_xy) {
                    size_t kix = kernel_idx % (size_t)nPts;
                    size_t kiy = kernel_idx / (size_t)nPts;
                    out_idx = kix * (size_t)nPts + kiy;
                } else {
                    out_idx = kernel_idx;
                }
                res.values[out_idx] = v;
                // 999/-999 — kernel-sentinel расходимости; FP LLE не различает.
                res.flags[out_idx]  = (v == 999.0 || v == -999.0) ? REGIME_UNBOUND
                                                                  : REGIME_OSCILLATION;
            }
        }

        // Write the data file once, AFTER the chunked loop — see data_export
        // header for why the grid layout is centralized there.
        if (!OUT_FILE_PATH.empty()) {
            std::ofstream out(OUT_FILE_PATH);
            if (out.is_open()) {
                out << std::setprecision(set_precision);
                data_export::write_lle2d_grid(out, nPts, res.values.data());
            }
        }

        // Авто-нормализация для colormap'а: min/max по валидным значениям.
        double vmin =  std::numeric_limits<double>::infinity();
        double vmax = -std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < total_cells; ++k) {
            int f = res.flags[k];
            double v = res.values[k];
            if (!regime_is_oscillation(f)) continue;   // FP и unbound вне шкалы
            if (!std::isfinite(v)) continue;
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
        if (std::isfinite(vmin) && std::isfinite(vmax)) {
            res.min_val = vmin;
            res.max_val = vmax;
        } else {
            res.min_val = 0.0;
            res.max_val = 0.0;
        }

        cleanup();
        #undef LLE2_CHECK
        #undef LLE2_CHECK_CU
        res.ok = true;
        return res;
    }


    // run_lle1d_continuation_gpu / run_ls1d_continuation_gpu — GPU-двойники
    // CPU-веток (см. run_lle1d_cpu / run_ls1d_cpu). Тот же «костыль», что у
    // Bifurcation1D: single-thread kernel, потому что continuation — цепочка.
    // Медленнее CPU, нужны для сверки и как привычный способ считать на GPU.
    LLE1DResult ParametricEngine::Impl::run_lle1d_continuation_gpu(const LLE1DRequest& req) {
        LLE1DResult res;
        auto fail = [&](const std::string& msg) -> LLE1DResult& { res.error = msg; return res; };

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_simple_cont_if_needed(cached_lle_cont, src_template_lle_cont,
                                           "lle1dContinuationKernel", "lle1d_cont.cu", ":llecont",
                                           req.krs_body, req.amountOfX, err))
            return fail(err);

        const int nPts = req.n_pts;
        numb* d_baseValues = nullptr;
        numb* d_baseX      = nullptr;
        numb* d_result     = nullptr;
        RunSignals sig;   // однопоточное ядро: тик на точку
        auto cleanup = [&]() {
            if (d_baseValues) cudaFree(d_baseValues);
            if (d_baseX)      cudaFree(d_baseX);
            if (d_result)     cudaFree(d_result);
            sig.release();
        };
        #define LC_CHECK(call, where) do { cudaError_t _e = (call); \
            if (_e != cudaSuccess) { res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); cleanup(); return res; } } while(0)
        #define LC_CHECK_CU(call, where) do { CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { res.error = std::string(where) + ": " + cu_err(_r); cleanup(); return res; } } while(0)

        LC_CHECK(cudaMalloc((void**)&d_baseValues, req.base_values.size() * sizeof(numb)), "cudaMalloc baseValues");
        LC_CHECK(cudaMalloc((void**)&d_baseX,      (size_t)req.amountOfX * sizeof(numb)),  "cudaMalloc baseX");
        LC_CHECK(cudaMalloc((void**)&d_result,     (size_t)nPts * sizeof(numb)),           "cudaMalloc result");
        // Вход приходит из Request в double — сужаем до numb на границе.
        const std::vector<numb> baseValues_staged_ = to_numb(req.base_values);
        const std::vector<numb> baseX_staged_      = to_numb(req.initial_conditions);
        LC_CHECK(cudaMemcpy(d_baseValues, baseValues_staged_.data(),
                            req.base_values.size() * sizeof(numb), cudaMemcpyHostToDevice), "memcpy baseValues");
        LC_CHECK(cudaMemcpy(d_baseX, baseX_staged_.data(),
                            (size_t)req.amountOfX * sizeof(numb), cudaMemcpyHostToDevice), "memcpy baseX");

        int    nPts_arg = nPts, reverse_arg = req.continuation_reverse ? 1 : 0;
        int    logScale_arg = req.log_scale ? 1 : 0, sweepIsH_arg = req.sweep_over_h ? 1 : 0;
        int    mutParamIdx_arg = req.param_index;
        int    amountOfValues_arg = (int)req.base_values.size(), amountOfX_arg = req.amountOfX;
        numb lo_arg = req.param_lo, hi_arg = req.param_hi;
        numb h_arg = req.h, NT_arg = req.NT, tMax_arg = req.t_max;
        numb transientTime_arg = req.transient_time, eps_arg = req.eps, maxValue_arg = req.max_value;

        if (!sig.alloc(res.error)) { cleanup(); return res; }
        int* d_cancel_arg   = sig.cancelArg();
        int* d_progress_arg = sig.progressArg();
        void* args[] = {
            &nPts_arg, &lo_arg, &hi_arg, &reverse_arg, &logScale_arg, &sweepIsH_arg,
            &mutParamIdx_arg, &d_baseValues, &amountOfValues_arg,
            &d_baseX, &amountOfX_arg,
            &h_arg, &NT_arg, &tMax_arg, &transientTime_arg,
            &eps_arg, &maxValue_arg, &d_result
            ,&d_cancel_arg, &d_progress_arg
        };
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true; res.error = "Cancelled by user"; cleanup(); return res;
        }
        LC_CHECK_CU(cuLaunchKernel(cached_lle_cont.kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
                    "cuLaunchKernel(lle cont)");
        if (!wait_with_signals(0, sig, req.cancel, req.progress, 0.0, (double)nPts, res.error))
            { cleanup(); return res; }
        LC_CHECK(cudaDeviceSynchronize(), "sync after lle cont");
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true; res.error = "Cancelled by user"; cleanup(); return res;
        }

        std::vector<numb> host((size_t)nPts);
        LC_CHECK(cudaMemcpy(host.data(), d_result, (size_t)nPts * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy result");
        cleanup();
        #undef LC_CHECK
        #undef LC_CHECK_CU

        res.n_pts    = nPts;
        res.param_lo = req.param_lo;
        res.param_hi = req.param_hi;
        res.continuation_reverse = req.continuation_reverse;
        res.lyapunov.assign(host.begin(), host.end());   // numb -> double
        res.flags.assign(nPts, 0);
        // NaN у continuation-ядра означает расходимость (FP оно не различает).
        for (int j = 0; j < nPts; ++j)
            res.flags[j] = std::isfinite(res.lyapunov[j]) ? REGIME_OSCILLATION : REGIME_UNBOUND;
        if (req.progress) req.progress->store(1.0f, std::memory_order_relaxed);
        res.ok = true;
        return res;
    }