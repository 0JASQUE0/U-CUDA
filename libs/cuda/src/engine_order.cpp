#include "engine_impl.h"



    // Узлы одной оси. Лог-сетка требует строго положительных границ: по шагу
    // это всегда так, по параметру — нет, поэтому проверка, а не clamp.
    bool ParametricEngine::Impl::order_axis_nodes(const OrderAxis& ax, const char* what,
                                 std::vector<double>& out, std::string& err)
    {
        const int n = (ax.kind == OrderAxisKind::None) ? 1 : ax.n_pts;
        if (n <= 0)    { err = std::string(what) + ": the number of points must be > 0"; return false; }
        if (n > 100000){ err = std::string(what) + ": the number of points is too large"; return false; }
        out.assign((size_t)n, 0.0);
        if (ax.kind == OrderAxisKind::None) { out[0] = 0.0; return true; }
        if (n == 1) { out[0] = ax.lo; return true; }
        if (ax.log_scale) {
            if (!(ax.lo > 0.0) || !(ax.hi > 0.0)) {
                err = std::string(what) + ": log scale requires both bounds > 0";
                return false;
            }
            const double k = std::log(ax.hi / ax.lo) / (double)(n - 1);
            for (int i = 0; i < n; ++i) out[(size_t)i] = ax.lo * std::exp(k * (double)i);
        } else {
            const double d = (ax.hi - ax.lo) / (double)(n - 1);
            for (int i = 0; i < n; ++i) out[(size_t)i] = ax.lo + d * (double)i;
        }
        return true;
    }


    // Число грубых шагов ячейки — ТА ЖЕ формула, что в ядре. Хосту она нужна,
    // чтобы посчитать полную работу для прогресс-бара: при свипе по h она
    // отличается от узла к узлу на порядки, и «среднее по сетке» давало бы
    // бар, который стоит на месте, а потом прыгает в конец.
    long long ParametricEngine::Impl::order_steps_for(double h, double tMax, bool snap) {
        if (!(h > 0.0)) return 1;
        long long N = snap ? (long long)(tMax / h + 0.5) : (long long)(tMax / h);
        return N < 1 ? 1 : N;
    }


    OrderResult ParametricEngine::Impl::run_order(const OrderRequest& req) {
        OrderResult res;
        res.axis_x = req.axis_x;
        res.axis_y = req.axis_y;
        auto fail = [&](const std::string& msg) -> OrderResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX) return fail("amountOfX out of range");
        if ((int)req.initial_conditions.size() != req.amountOfX) return fail("initial_conditions.size() != amountOfX");
        if (req.values.empty())                                  return fail("values is empty (at least a[0] is required)");
        if ((int)req.values.size() > kMaxAmountOfValues)         return fail("too many values");
        if (!(req.h > 0.0))                                      return fail("h must be > 0");
        if (!(req.t_max > 0.0))                                  return fail("t_max must be > 0");
        if (req.axis_x.kind == OrderAxisKind::None)              return fail("the X axis is not set");

        const int amountOfValues = (int)req.values.size();
        auto check_axis_index = [&](const OrderAxis& ax, const char* what) -> bool {
            if (ax.kind != OrderAxisKind::Value) return true;
            if (ax.index < 0 || ax.index >= amountOfValues) {
                res.error = std::string(what) + ": parameter index outside a[]";
                return false;
            }
            return true;
        };
        if (!check_axis_index(req.axis_x, "X axis")) return res;
        if (!check_axis_index(req.axis_y, "Y axis")) return res;
        if (req.axis_x.kind == OrderAxisKind::H && req.axis_y.kind == OrderAxisKind::H)
            return fail("both axes cannot sweep h");
        if (req.axis_x.kind == OrderAxisKind::Value && req.axis_y.kind == OrderAxisKind::Value
            && req.axis_x.index == req.axis_y.index)
            return fail("both axes sweep the same parameter");

        std::string err;
        if (!order_axis_nodes(req.axis_x, "X axis", res.axis_x_vals, err)) return fail(err);
        if (!order_axis_nodes(req.axis_y, "Y axis", res.axis_y_vals, err)) return fail(err);

        res.n_pts_x = (int)res.axis_x_vals.size();
        res.n_pts_y = (req.axis_y.kind == OrderAxisKind::None) ? 1 : (int)res.axis_y_vals.size();
        if (req.axis_y.kind == OrderAxisKind::None) res.axis_y_vals.assign(1, 0.0);

        const size_t total_cells = (size_t)res.n_pts_x * (size_t)res.n_pts_y;
        if (total_cells == 0) return fail("empty grid");

        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        cudaGetLastError();   // сброс sticky-ошибки прошлого прогона, см. run_fastsync

        // Эталон включается только когда есть И тело, И положительное число
        // подшагов: одно без другого — это просьба посчитать пустоту.
        const bool ref_on = !req.ref_krs_body.empty() && req.ref_substeps > 0;
        if (!compile_order_module(true, req.amountOfX, amountOfValues, req.krs_body,
                                  ref_on ? req.ref_krs_body : std::string(), err)) return fail(err);

        // Работа по ячейкам: h ячейки зависит только от той оси, что свипует h.
        auto cell_h = [&](int ix, int iy) -> double {
            double h = req.h;
            if (req.axis_x.kind == OrderAxisKind::H) h = res.axis_x_vals[(size_t)ix];
            if (req.axis_y.kind == OrderAxisKind::H) h = res.axis_y_vals[(size_t)iy];
            return h;
        };
        long long maxStepsPerCell = 1;
        double    totalSteps      = 0.0;
        for (int iy = 0; iy < res.n_pts_y; ++iy)
            for (int ix = 0; ix < res.n_pts_x; ++ix) {
                const long long N = order_steps_for(cell_h(ix, iy), req.t_max, req.snap_steps);
                if (N > maxStepsPerCell) maxStepsPerCell = N;
                totalSteps += (double)N;
            }
        // 1e15 шагов — это уже «никогда не досчитается», и (long long) ниже
        // всё равно переполнится на произведении с числом ячеек.
        if (totalSteps > 1.0e15) return fail("t_max / h too large: the work does not fit into a reasonable time");

        const int    progressStride = progress_stride_for((size_t)maxStepsPerCell);
        const double ticksTotal     = totalSteps / (double)progressStride;

        // Чанкование по ячейкам — не ради памяти (выход 4 числа на ячейку), а
        // ради TDR: один запуск на всю сетку при большом t_max/h легко
        // перевалит за watchdog. Бюджет — примерно столько вычислений правой
        // части, сколько GPU успевает за доли секунды; 7 вызовов КРС на
        // грубый шаг (1 + 2 + 4).
        const double kWorkBudget = 2.0e8;
        size_t cellsPerLaunch = (size_t)(kWorkBudget / (7.0 * (double)maxStepsPerCell));
        if (cellsPerLaunch < 256)          cellsPerLaunch = 256;
        if (cellsPerLaunch > total_cells)  cellsPerLaunch = total_cells;

        OrderDevBuf d_axisX, d_axisY, d_X0, d_values;
        OrderDevBuf d_p, d_e1, d_e2, d_eref, d_h, d_status;
        if (!d_axisX .alloc(res.axis_x_vals.size() * sizeof(numb), "axisXVals", err)) return fail(err);
        if (!d_axisY .alloc(res.axis_y_vals.size() * sizeof(numb), "axisYVals", err)) return fail(err);
        if (!d_X0    .alloc((size_t)req.amountOfX  * sizeof(numb), "X0",        err)) return fail(err);
        if (!d_values.alloc((size_t)amountOfValues * sizeof(numb), "values",    err)) return fail(err);
        if (!d_p     .alloc(total_cells * sizeof(numb), "outP",      err)) return fail(err);
        if (!d_e1    .alloc(total_cells * sizeof(numb), "outE1",     err)) return fail(err);
        if (!d_e2    .alloc(total_cells * sizeof(numb), "outE2",     err)) return fail(err);
        if (!d_eref  .alloc(total_cells * sizeof(numb), "outERef",   err)) return fail(err);
        if (!d_h     .alloc(total_cells * sizeof(numb), "outH",      err)) return fail(err);
        if (!d_status.alloc(total_cells * sizeof(int),  "outStatus", err)) return fail(err);

        {
            const std::vector<numb> ax = to_numb(res.axis_x_vals);
            const std::vector<numb> ay = to_numb(res.axis_y_vals);
            const std::vector<numb> x0 = to_numb(req.initial_conditions);
            const std::vector<numb> va = to_numb(req.values);
            auto up = [&](void* dst, const void* src, size_t bytes, const char* what) -> bool {
                cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
                if (e != cudaSuccess) { err = std::string("memcpy ") + what + ": " + cudaGetErrorString(e); return false; }
                return true;
            };
            if (!up(d_axisX.p,  ax.data(), ax.size() * sizeof(numb), "axisXVals")) return fail(err);
            if (!up(d_axisY.p,  ay.data(), ay.size() * sizeof(numb), "axisYVals")) return fail(err);
            if (!up(d_X0.p,     x0.data(), x0.size() * sizeof(numb), "X0"))        return fail(err);
            if (!up(d_values.p, va.data(), va.size() * sizeof(numb), "values"))    return fail(err);
        }

        RunSignals sig;
        if (!sig.alloc(err)) return fail(err);
        struct SigGuard { RunSignals& s; ~SigGuard() { s.release(); } } sig_guard{ sig };

        // Множитель порога полки округления. Само сравнение в ядре —
        // floorEps * |решение| * sqrt(4N): шум вычитания копится по шагам, так
        // что абсолютный порог обязан расти вместе с их числом (см. коммент в
        // order.template.cu). Здесь только константа при нём.
        const numb floorEps = (numb)2 * std::numeric_limits<numb>::epsilon();

        double ticksDone = 0.0;
        const size_t nLaunches = (total_cells + cellsPerLaunch - 1) / cellsPerLaunch;
        for (size_t L = 0; L < nLaunches; ++L) {
            const size_t offset = L * cellsPerLaunch;
            const size_t count  = (offset + cellsPerLaunch > total_cells) ? (total_cells - offset) : cellsPerLaunch;

            int    nPtsX_arg  = res.n_pts_x;
            int    nPtsY_arg  = res.n_pts_y;
            int    nCells_arg = (int)count;
            int    offset_arg = (int)offset;
            numb*  axX_arg    = d_axisX.as<numb>();
            numb*  axY_arg    = d_axisY.as<numb>();
            int    axXKind    = (int)req.axis_x.kind;
            int    axXIndex   = req.axis_x.index;
            int    axYKind    = (int)req.axis_y.kind;
            int    axYIndex   = req.axis_y.index;
            numb*  X0_arg     = d_X0.as<numb>();
            numb*  values_arg = d_values.as<numb>();
            numb   hBase_arg  = (numb)req.h;
            numb   tMax_arg   = (numb)req.t_max;
            int    snap_arg   = req.snap_steps    ? 1 : 0;
            int    endp_arg   = req.endpoint_only ? 1 : 0;
            numb   maxV_arg   = (numb)req.max_value;
            numb   feps_arg   = floorEps;
            int    refSub_arg = ref_on ? req.ref_substeps : 0;
            numb*  outP_arg   = d_p.as<numb>();
            numb*  outE1_arg  = d_e1.as<numb>();
            numb*  outE2_arg  = d_e2.as<numb>();
            numb*  outERef_arg = d_eref.as<numb>();
            numb*  outH_arg   = d_h.as<numb>();
            int*   outSt_arg  = d_status.as<int>();
            int*   cancel_arg = sig.cancelArg();
            int*   prog_arg   = sig.progressArg();
            int    stride_arg = progressStride;

            void* args[] = {
                &nPtsX_arg, &nPtsY_arg, &nCells_arg, &offset_arg,
                &axX_arg, &axY_arg,
                &axXKind, &axXIndex, &axYKind, &axYIndex,
                &X0_arg, &values_arg,
                &hBase_arg, &tMax_arg, &snap_arg, &endp_arg, &maxV_arg, &feps_arg,
                &refSub_arg,
                &outP_arg, &outE1_arg, &outE2_arg, &outERef_arg, &outH_arg, &outSt_arg,
                &cancel_arg, &prog_arg, &stride_arg
            };

            const int blockSize = 64;
            const int gridSize  = (int)((count + blockSize - 1) / blockSize);
            sig.resetTicks();
            CUresult r = cuLaunchKernel(cached_order.kernel, gridSize, 1, 1, blockSize, 1, 1,
                                        0, nullptr, args, nullptr);
            if (r != CUDA_SUCCESS) return fail("cuLaunchKernel(order): " + cu_err(r));
            if (!wait_with_signals(0, sig, req.cancel, req.progress, ticksDone, ticksTotal, err))
                return fail(err);
            cudaDeviceSynchronize();
            cudaError_t ce = cudaGetLastError();
            if (ce != cudaSuccess) return fail(std::string("order kernel: ") + cudaGetErrorString(ce));

            // Работа чанка в тиках — сумма по его ячейкам, а не count*средняя:
            // при свипе по h соседние ячейки различаются на порядки.
            for (size_t c = offset; c < offset + count; ++c) {
                const int ix = (int)(c % (size_t)res.n_pts_x);
                const int iy = (res.n_pts_y > 1) ? (int)(c / (size_t)res.n_pts_x) : 0;
                ticksDone += (double)order_steps_for(cell_h(ix, iy), req.t_max, req.snap_steps)
                           / (double)progressStride;
            }

            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
                res.cancelled = true;
                return res;
            }
        }

        {
            std::vector<numb> hp(total_cells), he1(total_cells), he2(total_cells),
                              herf(total_cells), hh(total_cells);
            res.status.assign(total_cells, 0);
            auto dn = [&](void* src, void* dst, size_t bytes, const char* what) -> bool {
                cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
                if (e != cudaSuccess) { err = std::string("memcpy D2H ") + what + ": " + cudaGetErrorString(e); return false; }
                return true;
            };
            if (!dn(d_p.p,      hp.data(),         total_cells * sizeof(numb), "p"))      return fail(err);
            if (!dn(d_e1.p,     he1.data(),        total_cells * sizeof(numb), "e1"))     return fail(err);
            if (!dn(d_e2.p,     he2.data(),        total_cells * sizeof(numb), "e2"))     return fail(err);
            if (!dn(d_eref.p,   herf.data(),       total_cells * sizeof(numb), "eRef"))   return fail(err);
            if (!dn(d_h.p,      hh.data(),         total_cells * sizeof(numb), "h"))      return fail(err);
            if (!dn(d_status.p, res.status.data(), total_cells * sizeof(int),  "status")) return fail(err);

            res.p.resize(total_cells); res.e1.resize(total_cells);
            res.e2.resize(total_cells); res.h_eff.resize(total_cells);
            res.e_ref.resize(total_cells);
            bool first_p = true, first_e = true, first_er = true;
            for (size_t i = 0; i < total_cells; ++i) {
                res.p[i]     = (double)hp[i];
                res.e1[i]    = (double)he1[i];
                res.e2[i]    = (double)he2[i];
                res.e_ref[i] = (double)herf[i];
                res.h_eff[i] = (double)hh[i];
                switch (res.status[i]) {
                    case ORDER_ST_DIVERGED:   ++res.n_diverged;   break;
                    case ORDER_ST_FLOOR:      ++res.n_floor;      break;
                    case ORDER_ST_NOCONTRACT: ++res.n_nocontract; break;
                    default:                  ++res.n_ok;         break;
                }
                // Диапазоны — только по чистым ячейкам (см. OrderResult::p_min).
                if (res.status[i] != ORDER_ST_OK) continue;
                if (std::isfinite(res.p[i])) {
                    if (first_p) { res.p_min = res.p_max = res.p[i]; first_p = false; }
                    else { if (res.p[i] < res.p_min) res.p_min = res.p[i];
                           if (res.p[i] > res.p_max) res.p_max = res.p[i]; }
                }
                if (std::isfinite(res.e1[i]) && res.e1[i] > 0.0) {
                    if (first_e) { res.e1_min = res.e1_max = res.e1[i]; first_e = false; }
                    else { if (res.e1[i] < res.e1_min) res.e1_min = res.e1[i];
                           if (res.e1[i] > res.e1_max) res.e1_max = res.e1[i]; }
                }
                if (std::isfinite(res.e_ref[i]) && res.e_ref[i] > 0.0) {
                    if (first_er) { res.eref_min = res.eref_max = res.e_ref[i]; first_er = false; }
                    else { if (res.e_ref[i] < res.eref_min) res.eref_min = res.e_ref[i];
                           if (res.e_ref[i] > res.eref_max) res.eref_max = res.e_ref[i]; }
                }
            }
        }

        res.ok = true;
        return res;
    }
