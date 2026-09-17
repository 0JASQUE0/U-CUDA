#include "engine_impl.h"



    // run_fastsync: dispatch по req.mode
    FastSyncResult ParametricEngine::Impl::run_fastsync(const FastSyncRequest& req) {
        FastSyncResult res;
        res.mode = req.mode;
        auto fail = [&](const std::string& msg) -> FastSyncResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                  return fail("krs_body пуст");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)   return fail("amountOfX вне диапазона");
        if ((int)req.ic_master.size()  != req.amountOfX)           return fail("ic_master.size() != amountOfX");
        if ((int)req.ic_slave.size()   != req.amountOfX)           return fail("ic_slave.size() != amountOfX");
        if ((int)req.k_forward.size()  != req.amountOfX)           return fail("k_forward.size() != amountOfX");
        if ((int)req.k_backward.size() != req.amountOfX)           return fail("k_backward.size() != amountOfX");
        if ((int)req.values.size() > kMaxAmountOfValues)           return fail("values слишком много");
        if (req.h <= 0.0)             return fail("h должно быть > 0");
        if (req.iter_of_synchr <= 0)  return fail("iter_of_synchr должно быть > 0");
        if (req.pre_scaller <= 0)     return fail("pre_scaller должно быть > 0");

        // Snapshot CSV-relevant request fields for GUI right-click export.
        // FastSync has no engine-side CSV writer, so this is consumed only by
        // data_export::export_fastsync; engine writes nothing to disk.
        res.snapshot.mode           = req.mode;
        res.snapshot.values         = req.values;
        res.snapshot.ic_master      = req.ic_master;
        res.snapshot.ic_slave       = req.ic_slave;
        res.snapshot.k_forward      = req.k_forward;
        res.snapshot.k_backward     = req.k_backward;
        res.snapshot.h              = req.h;
        res.snapshot.gpu_fmad       = get_nvrtc_fmad();
        res.snapshot.iter_of_synchr = req.iter_of_synchr;
        res.snapshot.preScaller     = req.pre_scaller;
        res.snapshot.window         = (double)req.window;
        res.snapshot.type_of_synch  = req.type_of_synch;
        res.snapshot.error_estim    = req.error_estim;
        res.snapshot.fs_error_trs   = req.fs_error_trs;
        res.snapshot.tMax           = req.t_max;
        res.snapshot.transientTime  = req.transient_time;
        res.snapshot.transientTimeSlave = req.transient_time_slave;
        res.snapshot.axis_x_var     = req.axis_x_var;
        res.snapshot.axis_y_var     = req.axis_y_var;
        res.snapshot.axis_x_lo      = req.axis_x_lo;
        res.snapshot.axis_x_hi      = req.axis_x_hi;
        res.snapshot.axis_y_lo      = req.axis_y_lo;
        res.snapshot.axis_y_hi      = req.axis_y_hi;
        res.snapshot.n_pts          = req.n_pts;
        res.snapshot.grid_swap_master_slave = req.grid_swap_master_slave;
        res.snapshot.ic_random_offset = req.ic_random_offset;
        res.snapshot.ic_eps           = req.ic_eps;
        res.snapshot.ic_seed          = req.ic_seed;
        res.snapshot.gs_warmup        = req.gs_warmup;
        res.snapshot.var_names      = req.var_names;

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);

        // Сбрасываем sticky-ошибку от прошлого неудачного запуска (например OOB в shared memory):
        // без этого cudaMalloc на старте мгновенно падает с "illegal memory access", что является
        // эхом прошлой ошибки, а не текущей. Реально повреждённый контекст восстановится только
        // рестартом приложения — здесь лишь сброс флага для случаев, когда GPU ещё работоспособен.
        cudaGetLastError();

        if (req.mode == 0) {
            // On Attractor
            if (req.t_max <= 0.0)         return fail("t_max должно быть > 0");
            if (req.transient_time < 0)   return fail("transient_time должно быть >= 0");
            if (req.window <= (numb)0.0)  return fail("window должно быть > 0");

            // Two FS-specific kernels — НЕ calculateDiscreteModelCUDA:
            //   fillFSMasterTrajectory (template, single-thread) — фильтрует
            //   timeDomain полным X[] на каждом шаге через FS device function.
            //   calculateDiscreteModelforFastSynchroCUDA — параллельный синхро-проход.
            std::vector<const char*> exprs = {
                "fillFSMasterTrajectory",
                "calculateDiscreteModelforFastSynchroCUDA"
            };
            if (!compile_fs_module(src_template_fs_attr, ":fs_attr", req.amountOfX, req.krs_body,
                                   req.type_of_synch, req.error_estim, req.fs_error_trs,
                                   exprs, cached_fs_attr, err)) return fail(err);

            const int amountOfNTPoints      = (int)(req.window / req.h);
            const int amountOfCTPoints      = (int)(req.t_max / req.h);
            // size_t + потолок — как в On Grid: (int)(TT/h) при мелком h это UB.
            size_t amountOfPointsForSkip = 0;
            {
                const double skip_d = req.transient_time / req.h;   // h > 0 проверен выше
                if (!std::isfinite(skip_d) || skip_d > 1.0e15)
                    return fail("transient_time / h слишком велико");
                amountOfPointsForSkip = (size_t)skip_d;
            }
            const int nPts                  = amountOfCTPoints / (req.pre_scaller > 0 ? req.pre_scaller : 1);
            if (nPts <= 0) return fail("computed nPts <= 0 (t_max/h/preScaller)");
            // FS-kernel читает timeDomain[idx*preScaller*amountOfX + i*amountOfX + j]
            // где idx ∈ [0..nPts), i ∈ [0..amountOfNTPoints). Поэтому буфер должен
            // вмещать amountOfCTPoints + amountOfNTPoints точек (с padding'ом).
            const int traj_len_pts          = amountOfCTPoints + amountOfNTPoints;

            int amountOfIC_int     = req.amountOfX;
            int amountOfValues_int = (int)req.values.size();

            numb* d_timeDomain = nullptr; numb* d_output = nullptr;
            numb* d_Xs = nullptr;   numb* d_X0 = nullptr;
            numb* d_values = nullptr;
            numb* d_kF = nullptr;   numb* d_kB = nullptr;
            RunSignals sig;   // прогресс и отмена в mapped-памяти
            #define FS_CHECK(x, m) do { cudaError_t _e = (x); if (_e != cudaSuccess) { err = std::string(m) + ": " + cudaGetErrorString(_e); goto FS0_FAIL; } } while(0)

            size_t traj_bytes = (size_t)traj_len_pts * (size_t)amountOfIC_int * sizeof(numb);
            size_t out_bytes  = (size_t)nPts * sizeof(numb);
            const double fsW1 = (double)amountOfPointsForSkip + (double)traj_len_pts;
            const double fsW2 = 2.0 * (double)req.iter_of_synchr * (double)traj_len_pts;
            const double fsW  = (fsW1 + fsW2) > 0 ? (fsW1 + fsW2) : 1.0;
            const double fsT1 = (double)traj_len_pts;                                 // тиков в фазе 1
            const double fsT2 = (double)nPts * (double)req.iter_of_synchr;       // тиков в фазе 2
            FS_CHECK(cudaMalloc((void**)&d_timeDomain, traj_bytes), "cudaMalloc d_timeDomain");
            FS_CHECK(cudaMalloc((void**)&d_output,     out_bytes),  "cudaMalloc d_output");
            FS_CHECK(cudaMalloc((void**)&d_Xs,         amountOfIC_int * sizeof(numb)), "cudaMalloc d_Xs");
            FS_CHECK(cudaMalloc((void**)&d_X0,         amountOfIC_int * sizeof(numb)), "cudaMalloc d_X0");
            FS_CHECK(cudaMalloc((void**)&d_values,     amountOfValues_int * sizeof(numb)), "cudaMalloc d_values");
            FS_CHECK(cudaMalloc((void**)&d_kF,         amountOfIC_int * sizeof(numb)), "cudaMalloc d_kF");
            FS_CHECK(cudaMalloc((void**)&d_kB,         amountOfIC_int * sizeof(numb)), "cudaMalloc d_kB");
            if (!sig.alloc(err)) goto FS0_FAIL;
            // Бар делится между двумя фазами по их доле работы. Заполнение
            // окна идёт в один поток и стоит skip + pts шагов; проходы вперёд-назад
            // идут параллельно по точкам, так что по времени это 2 * iterOfSynchr * pts.
            // Первая версия отдавала весь бар первой фазе — он доходил до 90%
            // и там вставал.


            // См. пояснение в grid-ветке: Request хранит их как vector<double>,
            // а буферы — numb, поэтому конверсия обязательна для всех, не
            // только для values.
            FS_CHECK(cudaMemcpy(d_X0,     to_numb(req.ic_master).data(),  amountOfIC_int * sizeof(numb),     cudaMemcpyHostToDevice), "memcpy X0");
            FS_CHECK(cudaMemcpy(d_Xs,     to_numb(req.ic_slave).data(),   amountOfIC_int * sizeof(numb),     cudaMemcpyHostToDevice), "memcpy Xs");
            FS_CHECK(cudaMemcpy(d_values, to_numb(req.values).data(),     amountOfValues_int * sizeof(numb), cudaMemcpyHostToDevice), "memcpy values");
            FS_CHECK(cudaMemcpy(d_kF,     to_numb(req.k_forward).data(),  amountOfIC_int * sizeof(numb),     cudaMemcpyHostToDevice), "memcpy kF");
            FS_CHECK(cudaMemcpy(d_kB,     to_numb(req.k_backward).data(), amountOfIC_int * sizeof(numb),     cudaMemcpyHostToDevice), "memcpy kB");

            // Шаг 1: fillFSMasterTrajectory — single-thread, заливает d_timeDomain
            // полным X[] на каждом шаге (через FS device function с K=0).
            if (req.progress) req.progress->store(0.05f);
            {
                numb  h_arg     = req.h;
                size_t  skip_arg  = amountOfPointsForSkip;
                int     pts_arg   = traj_len_pts;
                int* d_cancel_arg   = sig.cancelArg();
                int* d_progress_arg = sig.progressArg();
                sig.resetTicks();
                void* args_fill[] = {
                    &d_values, &h_arg, &d_X0, &skip_arg, &pts_arg, &d_timeDomain,
                    &d_cancel_arg, &d_progress_arg
                };
                CUresult r = cuLaunchKernel(cached_fs_attr.kernel_fs_fill,
                                            1, 1, 1, 1, 1, 1,
                                            0, nullptr, args_fill, nullptr);
                if (r != CUDA_SUCCESS) { err = "cuLaunchKernel(fs_attr fill): " + cu_err(r); goto FS0_FAIL; }
                // Заполнение окна мастера идёт в один поток и занимает
                // основную часть прогона, поэтому отдаём ему весь бар.
                if (!wait_with_signals(0, sig, req.cancel, req.progress,
                                       0.0, fsT1 * fsW / (fsW1 > 0 ? fsW1 : 1.0), err)) goto FS0_FAIL;
                cudaDeviceSynchronize();
            }

            // Шаг 2: calculateDiscreteModelforFastSynchroCUDA — nPts threads, читает
            // timeDomain, пишет d_output[idx] = ошибка синхронизации на точке idx.
            {
                int    nPts_int           = nPts;
                int    nPtsLimiter_int    = nPts;
                int    amountOfNTPoints_i = amountOfNTPoints;
                numb h_arg              = req.h;
                int    iterOfSynchr_i     = req.iter_of_synchr;
                int    preScaller_i       = req.pre_scaller;
                numb maxValue_arg       = req.max_value;
                int    amountOfValues_i   = amountOfValues_int;
                // d_Xs остаётся заполненным и в random-режиме: ядро просто не
                // читает его, когда ic_random_offset взведён.
                int    icRandom_i         = req.ic_random_offset ? 1 : 0;
                numb   icEps_arg          = (numb)req.ic_eps;
                unsigned long long icSeed_arg = req.ic_seed;
                int    gsWarmup_i         = req.gs_warmup;

                int* d_cancel_arg   = sig.cancelArg();
                int* d_progress_arg = sig.progressArg();
                void* args_fs[] = {
                    &nPts_int, &nPtsLimiter_int, &amountOfNTPoints_i, &h_arg,
                    &d_Xs, &amountOfIC_int,
                    &d_values, &d_kF, &d_kB,
                    &iterOfSynchr_i, &amountOfValues_i, &amountOfNTPoints_i,
                    &maxValue_arg,
                    &d_timeDomain, &d_output, &preScaller_i,
                    &icRandom_i, &icEps_arg, &icSeed_arg, &gsWarmup_i,
                    &d_cancel_arg, &d_progress_arg
                };
                int blockSize = 32;
                int gridSize  = (nPts + blockSize - 1) / blockSize;
                CUresult r = cuLaunchKernel(cached_fs_attr.kernel_fs_traj,
                                            gridSize, 1, 1, blockSize, 1, 1,
                                            0, nullptr, args_fs, nullptr);
                if (r != CUDA_SUCCESS) { err = "cuLaunchKernel(fs_attr): " + cu_err(r); goto FS0_FAIL; }
                sig.resetTicks();
                if (!wait_with_signals(0, sig, req.cancel, req.progress,
                                       fsT2 * fsW1 / (fsW2 > 0 ? fsW2 : 1.0),
                                       fsT2 * fsW  / (fsW2 > 0 ? fsW2 : 1.0), err)) goto FS0_FAIL;
                cudaDeviceSynchronize();
            }
            if (req.progress) req.progress->store(1.0f);

            // Шаг 3: D2H — trajectory + errors. timeDomain хранит RAW точки
            // (без decimator'а); для визуализации выбираем точки с шагом preScaller.
            {
                std::vector<numb> h_traj((size_t)traj_len_pts * (size_t)amountOfIC_int);
                FS_CHECK(cudaMemcpy(h_traj.data(), d_timeDomain, traj_bytes, cudaMemcpyDeviceToHost), "memcpy traj D2H");

                std::vector<numb> h_out(nPts);
                FS_CHECK(cudaMemcpy(h_out.data(), d_output, out_bytes, cudaMemcpyDeviceToHost), "memcpy out D2H");

                // Сохраняем полную (decimated) траекторию — все каналы. GUI
                // выберет axis_x_var / axis_y_var при отрисовке (re-render
                // при смене combo'ов без повторного Run).
                res.traj_full.resize((size_t)nPts * (size_t)amountOfIC_int);
                for (int i = 0; i < nPts; ++i) {
                    size_t row = (size_t)i * (size_t)req.pre_scaller;
                    if (row >= (size_t)traj_len_pts) row = (size_t)traj_len_pts - 1;
                    for (int j = 0; j < amountOfIC_int; ++j)
                        res.traj_full[(size_t)i * amountOfIC_int + j]
                            = h_traj[row * (size_t)amountOfIC_int + j];
                }
                res.sync_error.assign(h_out.begin(), h_out.end());   // numb -> double
                res.n_pts_traj    = nPts;
                res.amountOfX_traj = amountOfIC_int;
            }

            // min/max sync_error для autoscale.
            {
                double vmin = std::numeric_limits<double>::infinity();
                double vmax = -std::numeric_limits<double>::infinity();
                for (double v : res.sync_error) {
                    if (!std::isfinite(v)) continue;
                    if (v < vmin) vmin = v;
                    if (v > vmax) vmax = v;
                }
                res.min_val = std::isfinite(vmin) ? vmin : 0.0;
                res.max_val = std::isfinite(vmax) ? vmax : 0.0;
            }

            cudaFree(d_timeDomain); cudaFree(d_output);
            cudaFree(d_Xs); cudaFree(d_X0);
            cudaFree(d_values); cudaFree(d_kF); cudaFree(d_kB);
            #undef FS_CHECK

            // CSV (On Attractor): delegate to data_export so engine and the
            // GUI right-click export produce byte-identical files.
            if (!req.csv_output_path.empty()) {
                std::ofstream cfg(req.csv_output_path + "_config.csv");
                if (cfg.is_open()) {
                    cfg << std::setprecision(set_precision);
                    data_export::write_fastsync_config(cfg, res.snapshot);
                }
                std::ofstream csv(req.csv_output_path);
                if (csv.is_open()) {
                    csv << std::setprecision(set_precision);
                    data_export::write_fastsync_attractor(csv, res, req.var_names);
                }
            }

            res.ok = true;
            return res;
            FS0_FAIL:
            sig.release();
                if (d_timeDomain) cudaFree(d_timeDomain);
                if (d_output)     cudaFree(d_output);
                if (d_Xs)         cudaFree(d_Xs);
                if (d_X0)         cudaFree(d_X0);
                if (d_values)     cudaFree(d_values);
                if (d_kF)         cudaFree(d_kF);
                if (d_kB)         cudaFree(d_kB);
            return fail(err);
        }
        else {
            // On Grid
            if (req.n_pts <= 0)         return fail("n_pts должно быть > 0");
            if (req.axis_x_var < 0 || req.axis_x_var >= req.amountOfX) return fail("axis_x_var вне диапазона");
            if (req.axis_y_var < 0 || req.axis_y_var >= req.amountOfX) return fail("axis_y_var вне диапазона");
            if (req.axis_x_var == req.axis_y_var) return fail("axis_x_var == axis_y_var");
            if (req.transient_time < 0)        return fail("transient_time должно быть >= 0");
            if (req.transient_time_slave < 0)  return fail("transient_time_slave должно быть >= 0");

            std::vector<const char*> exprs = { "calculateDiscreteModelICCforFastSynchro" };
            if (!compile_fs_module(src_template_fs_grid, ":fs_grid", req.amountOfX, req.krs_body,
                                   req.type_of_synch, req.error_estim, req.fs_error_trs,
                                   exprs, cached_fs_grid, err)) return fail(err);

            // Non-const: их адреса попадают в void*[] для cuLaunchKernel.
            int amountOfPointsInBlock = (int)(req.window / req.h / req.pre_scaller);
            if (amountOfPointsInBlock <= 0) return fail("computed amountOfPointsInBlock <= 0");

            // Transient (TT) — число шагов интегрирования на досадку, отдельно для master и для
            // slave. Уходят в grid-ядро, которое досаживает каждую систему per-cell после
            // grid-override (свипуемая сторона — из затравки своей ячейки, фиксированная — из одной
            // точки). size_t, а не int: TT/h легко перерастает 2^31 (TT=1e5 при h=1e-5 даёт 1e10) —
            // на int это UB, на практике мусор или отрицательное, и транзиент молча пропадал.
            // h > 0 уже проверен выше.
            auto skip_steps = [&](double tt, const char* what, size_t& out) -> bool {
                out = 0;
                if (tt <= 0.0) return true;
                const double skip_d = tt / req.h;
                if (!std::isfinite(skip_d) || skip_d > 1.0e15) {
                    err = std::string(what) + " / h слишком велико";
                    return false;
                }
                out = (size_t)skip_d;
                return true;
            };
            size_t amountOfPointsForSkipMaster = 0;
            size_t amountOfPointsForSkipSlave  = 0;
            if (!skip_steps(req.transient_time,       "transient_time (master)", amountOfPointsForSkipMaster)) return fail(err);
            if (!skip_steps(req.transient_time_slave, "transient_time (slave)",  amountOfPointsForSkipSlave))  return fail(err);
            const size_t total_cells = (size_t)req.n_pts * (size_t)req.n_pts;
            int amountOfIC_int     = req.amountOfX;
            int amountOfValues_int = (int)req.values.size();

            // Memory budget — per-cell trajectory buffer = sizeOfBlock * amountOfX * sizeof(numb).
            size_t freeMemory = 0;
            if (!gpu_free_budget(0.5, freeMemory)) return fail("cudaMemGetInfo failed");
            size_t perCellBytes = (size_t)amountOfPointsInBlock * (size_t)amountOfIC_int * sizeof(numb);
            if (perCellBytes == 0) perCellBytes = sizeof(numb);
            size_t nPtsLimiter = freeMemory / perCellBytes;
            if (nPtsLimiter == 0) nPtsLimiter = 32;
            if (nPtsLimiter > total_cells) nPtsLimiter = total_cells;
            RunSignals sig;   // прогресс и отмена в mapped-памяти
            const size_t originalNPtsLimiter = nPtsLimiter;

            numb* d_data    = nullptr; numb* d_ranges = nullptr;
            int*    d_idx_mv  = nullptr; numb* d_ic_m   = nullptr; numb* d_ic_s = nullptr;
            numb* d_values  = nullptr; numb* d_kF     = nullptr; numb* d_kB   = nullptr;
            int*    d_helpful = nullptr; numb* d_fs_err = nullptr;

            #define FS_GCHECK(x, m) do { cudaError_t _e = (x); if (_e != cudaSuccess) { err = std::string(m) + ": " + cudaGetErrorString(_e); goto FS1_FAIL; } } while(0)

            const int    progressStride = progress_stride_for((size_t)amountOfPointsInBlock);
            const double ticksPerCell   = (double)((size_t)amountOfPointsInBlock / (size_t)progressStride);
            const double ticksTotal     = (double)total_cells * ticksPerCell;
            size_t amountOfIteration = (total_cells + nPtsLimiter - 1) / nPtsLimiter;
            FS_GCHECK(cudaMalloc((void**)&d_data,    (size_t)nPtsLimiter * perCellBytes), "cudaMalloc d_data");
            FS_GCHECK(cudaMalloc((void**)&d_ranges,  4 * sizeof(numb)),                  "cudaMalloc d_ranges");
            FS_GCHECK(cudaMalloc((void**)&d_idx_mv,  2 * sizeof(int)),                     "cudaMalloc d_idx_mv");
            FS_GCHECK(cudaMalloc((void**)&d_ic_m,    amountOfIC_int * sizeof(numb)),     "cudaMalloc d_ic_m");
            FS_GCHECK(cudaMalloc((void**)&d_ic_s,    amountOfIC_int * sizeof(numb)),     "cudaMalloc d_ic_s");
            FS_GCHECK(cudaMalloc((void**)&d_values,  amountOfValues_int * sizeof(numb)), "cudaMalloc d_values");
            FS_GCHECK(cudaMalloc((void**)&d_kF,      amountOfIC_int * sizeof(numb)),     "cudaMalloc d_kF");
            FS_GCHECK(cudaMalloc((void**)&d_kB,      amountOfIC_int * sizeof(numb)),     "cudaMalloc d_kB");
            FS_GCHECK(cudaMalloc((void**)&d_helpful, nPtsLimiter * sizeof(int)),           "cudaMalloc d_helpful");
            FS_GCHECK(cudaMalloc((void**)&d_fs_err,  total_cells * sizeof(numb)),         "cudaMalloc d_fs_err");
            // Тики ставит цикл заполнения окна мастера — это amountOfPointsInBlock
            // шагов на ячейку; проходы вперёд-назад после него не тикают, поэтому
            // бар доходит до ста на хвосте чанка и ждёт там.
            if (!sig.alloc(err)) goto FS1_FAIL;


            {
                numb   ranges_arr[4] = { (numb)req.axis_x_lo, (numb)req.axis_x_hi, (numb)req.axis_y_lo, (numb)req.axis_y_hi };
                int    idx_mv_arr[2] = { req.axis_x_var, req.axis_y_var };
                FS_GCHECK(cudaMemcpy(d_ranges, ranges_arr, 4 * sizeof(numb),  cudaMemcpyHostToDevice), "memcpy ranges");
                FS_GCHECK(cudaMemcpy(d_idx_mv, idx_mv_arr, 2 * sizeof(int),     cudaMemcpyHostToDevice), "memcpy idx_mv");
                // Все четыре — vector<double> в Request, а буферы типа numb:
                // без to_numb при numb=float копировались бы половинки double.
                // values рядом уже конвертировался, остальные — нет.
                FS_GCHECK(cudaMemcpy(d_ic_m,   to_numb(req.ic_master).data(),  amountOfIC_int * sizeof(numb),     cudaMemcpyHostToDevice), "memcpy ic_m");
                FS_GCHECK(cudaMemcpy(d_ic_s,   to_numb(req.ic_slave).data(),   amountOfIC_int * sizeof(numb),     cudaMemcpyHostToDevice), "memcpy ic_s");
                FS_GCHECK(cudaMemcpy(d_values, to_numb(req.values).data(),     amountOfValues_int * sizeof(numb), cudaMemcpyHostToDevice), "memcpy values");
                FS_GCHECK(cudaMemcpy(d_kF,     to_numb(req.k_forward).data(),  amountOfIC_int * sizeof(numb),     cudaMemcpyHostToDevice), "memcpy kF");
                FS_GCHECK(cudaMemcpy(d_kB,     to_numb(req.k_backward).data(), amountOfIC_int * sizeof(numb),     cudaMemcpyHostToDevice), "memcpy kB");
            }

            // Transient выполняется внутри grid-ядра, per-cell и своим временем для каждой системы
            // (см. amountOfPointsForSkip* выше и комментарий в
            // calculateDiscreteModelICCforFastSynchro). Раньше здесь стоял однопоточный пре-пасс
            // fillFSTransientIC с ОДНИМ TT на обоих и только для нефиксируемой IC: свипуемая сторона
            // стартовала сырой, а прогон в один поток на длинных TT ловил TDR.


            for (size_t i = 0; i < amountOfIteration; ++i) {
                size_t cur_limiter = nPtsLimiter;
                if (i == amountOfIteration - 1) cur_limiter = total_cells - originalNPtsLimiter * i;

                int    nPts_arg                  = req.n_pts;
                int    nPtsLimiter_int           = (int)cur_limiter;
                // sizeOfBlock — РАЗМЕР per-thread слота в `data` (в numb-элементах).
                // loopCalculateDiscreteModelForFastSynchro_2 индексирует
                // data[idx*sizeOfBlock + i*amountOfX + j] → нужно вместить
                // amountOfIterations кадров по amountOfX компонент каждый.
                int    sizeOfBlock_int           = amountOfPointsInBlock * amountOfIC_int;
                int    amountOfCalculatedPoints  = (int)(i * originalNPtsLimiter);
                int    dimension_arg             = 2;
                numb h_arg                     = req.h;
                int    amountOfIterations_int    = amountOfPointsInBlock;
                int    preScaller_int            = req.pre_scaller;
                numb maxValue_arg              = req.max_value;
                int    iterOfSynchr_int          = req.iter_of_synchr;
                numb* d_fs_err_chunk           = d_fs_err + i * originalNPtsLimiter;
                int    swap_role_int             = req.grid_swap_master_slave ? 1 : 0;
                size_t skip_master_arg           = amountOfPointsForSkipMaster;
                size_t skip_slave_arg            = amountOfPointsForSkipSlave;
                int    icRandom_int              = req.ic_random_offset ? 1 : 0;
                numb   icEps_arg                 = (numb)req.ic_eps;
                unsigned long long icSeed_arg    = req.ic_seed;
                int    gsWarmup_int              = req.gs_warmup;

                int*   d_cancel_arg       = sig.cancelArg();
                int*   d_progress_arg     = sig.progressArg();
                int    progressStride_arg = progressStride;
                sig.resetTicks();
                void* args_grid[] = {
                    &nPts_arg, &nPtsLimiter_int, &sizeOfBlock_int, &amountOfCalculatedPoints,
                    &dimension_arg, &d_ranges, &h_arg, &d_idx_mv,
                    &d_ic_m, &d_ic_s, &amountOfIC_int,
                    &d_values, &amountOfValues_int,
                    &amountOfIterations_int, &preScaller_int, &maxValue_arg, &iterOfSynchr_int,
                    &d_kF, &d_kB,
                    &d_data, &d_helpful, &d_fs_err_chunk,
                    &swap_role_int, &skip_master_arg, &skip_slave_arg,
                    &icRandom_int, &icEps_arg, &icSeed_arg, &gsWarmup_int
                    ,&d_cancel_arg, &d_progress_arg, &progressStride_arg
                };
                int blockSize = 32;
                int gridSize  = (int)((cur_limiter + blockSize - 1) / blockSize);
                // Shared memory: kernel объявляет `extern __shared__ numb s[]` и кладёт туда
                // localX[amountOfIC] + localValues[amountOfValues] на поток, т.е. blockSize × (IC +
                // values) × 8 байт. Без правильного размера — OOB в shared, illegal memory access и
                // повреждённый (sticky) CUDA-контекст.
                unsigned int shared_grid = (unsigned int)((amountOfIC_int + amountOfValues_int)
                                                          * sizeof(numb) * blockSize);
                CUresult r = cuLaunchKernel(cached_fs_grid.kernel_fs_grid,
                                            gridSize, 1, 1, blockSize, 1, 1,
                                            shared_grid, nullptr, args_grid, nullptr);
                if (r != CUDA_SUCCESS) { err = "cuLaunchKernel(fs_grid): " + cu_err(r); goto FS1_FAIL; }
                if (!wait_with_signals(0, sig, req.cancel, req.progress,
                                       (double)(originalNPtsLimiter * i) * ticksPerCell,
                                       ticksTotal, err)) goto FS1_FAIL;
                cudaDeviceSynchronize();

                // Диверг-флаги чанка: ядро пишет -1 в разлетевшиеся ячейки и 0
                // в остальные, так что буфер определён целиком и читать его
                // безопасно без предварительного обнуления.
                {
                    std::vector<int> h_flags(cur_limiter);
                    if (cudaMemcpy(h_flags.data(), d_helpful, cur_limiter * sizeof(int),
                                   cudaMemcpyDeviceToHost) == cudaSuccess) {
                        for (int f : h_flags) if (f == -1) ++res.diverged_cells;
                    }
                }

                if (req.cancel && req.cancel->load()) { res.cancelled = true; goto FS1_CLEANUP; }
            }

            {
                std::vector<numb> h_out(total_cells);
                FS_GCHECK(cudaMemcpy(h_out.data(), d_fs_err, total_cells * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy fs_err D2H");
                res.heatmap.assign(h_out.begin(), h_out.end());      // numb -> double
                res.n_pts_grid = req.n_pts;
                res.total_cells = (int)total_cells;
                res.axis_x_lo = req.axis_x_lo; res.axis_x_hi = req.axis_x_hi;
                res.axis_y_lo = req.axis_y_lo; res.axis_y_hi = req.axis_y_hi;
                res.axis_x_var = req.axis_x_var; res.axis_y_var = req.axis_y_var;

                double vmin = std::numeric_limits<double>::infinity();
                double vmax = -std::numeric_limits<double>::infinity();
                for (double v : res.heatmap) {
                    if (!std::isfinite(v)) continue;
                    if (v < vmin) vmin = v;
                    if (v > vmax) vmax = v;
                }
                res.min_val = std::isfinite(vmin) ? vmin : 0.0;
                res.max_val = std::isfinite(vmax) ? vmax : 0.0;

                // CSV (On Grid): delegate to data_export so engine and the
                // GUI right-click export share the same writer (and stay
                // byte-identical). Layout: 2-line ranges header + n×n grid.
                if (!req.csv_output_path.empty()) {
                    std::ofstream cfg(req.csv_output_path + "_config.csv");
                    if (cfg.is_open()) {
                        cfg << std::setprecision(set_precision);
                        data_export::write_fastsync_config(cfg, res.snapshot);
                    }
                    std::ofstream csv(req.csv_output_path);
                    if (csv.is_open()) {
                        csv << std::setprecision(set_precision);
                        data_export::write_fastsync_grid(csv, res,
                            req.axis_x_lo, req.axis_x_hi,
                            req.axis_y_lo, req.axis_y_hi);
                    }
                }
            }

            FS1_CLEANUP:
            sig.release();
            cudaFree(d_data); cudaFree(d_ranges); cudaFree(d_idx_mv);
            cudaFree(d_ic_m); cudaFree(d_ic_s); cudaFree(d_values);
            cudaFree(d_kF); cudaFree(d_kB); cudaFree(d_helpful); cudaFree(d_fs_err);
            #undef FS_GCHECK
            if (!res.cancelled) res.ok = true;
            return res;

            FS1_FAIL:
            sig.release();
            if (d_data)    cudaFree(d_data);
            if (d_ranges)  cudaFree(d_ranges);
            if (d_idx_mv)  cudaFree(d_idx_mv);
            if (d_ic_m)    cudaFree(d_ic_m);
            if (d_ic_s)    cudaFree(d_ic_s);
            if (d_values)  cudaFree(d_values);
            if (d_kF)      cudaFree(d_kF);
            if (d_kB)      cudaFree(d_kB);
            if (d_helpful) cudaFree(d_helpful);
            if (d_fs_err)  cudaFree(d_fs_err);
            return fail(err);
        }
    }