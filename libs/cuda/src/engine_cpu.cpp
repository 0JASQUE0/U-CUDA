#include "engine_helpers.h"

namespace ucuda::cuda_detail {


// Порт loopCalculateDiscreteModel_int (cudaLibrary.cu) на CPU. Отличие одно: там размерность —
// compile-time AMOUNTOFX, здесь она приходит параметром (в NVRTC-сборке AMOUNTOFX как раз
// раскрывается в amountOfX, т.е. численно это одно и то же).
// Возврат — REGIME_* из configCUDA.h (1 = oscillation, -1 = fixed point, 0 = unbound). Проверка
// расходимости — раз в kCheckInterval итераций, как на GPU.
int cpu_loop_model(KrsCpuStep::StepFn step,
                   numb* x, const numb* a, numb h,
                   size_t iterations, int amountOfX, int preScaller,
                   int writableVar, numb maxValue,
                   numb* data)
{
    for (size_t i = 0; i < iterations; ++i) {
        if (data != nullptr) {
            // writableVar < 0 -> комбинация первых (до трёх) переменных.
            if (writableVar < 0) {
                if      (amountOfX >= 3) data[i] = x[0] + (numb)kPi * x[1] + (numb)kEuler * x[2];
                else if (amountOfX == 2) data[i] = x[0] + (numb)kPi * x[1];
                else                     data[i] = x[0];
            } else {
                data[i] = x[writableVar];
            }
            for (int j = 0; j < preScaller; ++j) step(x, a, h);
        }
        else {
            step(x, a, h);
        }

        if (i % (size_t)kCheckInterval == 0) {
            numb checker = (numb)0;
            for (int j = 0; j < amountOfX; ++j) checker += std::fabs(x[j]);
            if (std::isnan(checker) || std::isinf(checker)) return REGIME_UNBOUND;
            if (maxValue != (numb)0 && std::fabs(checker) > maxValue) return REGIME_UNBOUND;
        }
    }

    // Лишний preScaller-шаг после основного цикла — он нужен GPU-версии для
    // проверки на неподвижную точку И заодно сдвигает x[], который в
    // continuation переносится в следующую точку параметра. Без него CPU-ветка
    // разошлась бы с GPU уже на второй точке свипа.
    static thread_local std::vector<numb> xPrev;
    xPrev.assign(x, x + amountOfX);
    for (int j = 0; j < preScaller; ++j) step(x, a, h);

    numb tempResult = (numb)0;
    for (int j = 0; j < amountOfX; ++j) tempResult += std::fabs(x[j] - xPrev[j]);
    if (std::fabs(tempResult) < (numb)get_peak_config().eps_fixed_point)
        return REGIME_FIXED_POINT;
    return REGIME_OSCILLATION;
}


// Порт peakFinder (cudaLibrary.cu) на один блок данных. Индексация здесь блок-локальная
// (startDataIndex == 0): CPU-ветка обрабатывает точки свипа потоково и держит один блок, а не
// матрицу nPts x record_steps. Логика — построчная копия, включая параболическую интерполяцию,
// сдвиг пиков влево на один и пересчёт timeOfPeaks в межпиковые интервалы.
int cpu_peak_finder(const numb* data, size_t amountOfPoints,
                    numb* outPeaks, numb* timeOfPeaks, numb h)
{
    // Снимок knobs на весь вызов: GPU-двойник видит их как compile-time
    // константы, поэтому в пределах одной точки свипа они меняться не должны.
    const PeakConfig pc = get_peak_config();

    if (!pc.do_calculate_peaks) {
        int n = (int)amountOfPoints;
        if (n >= pc.max_amount_of_peaks) n = pc.max_amount_of_peaks;
        for (int i = 0; i < n; ++i) { outPeaks[i] = data[i]; timeOfPeaks[i] = (numb)0; }
        return n - 1;
    }
    // Границы циклов на GPU считаются в size_t как amountOfPoints - 2; при
    // крошечном блоке это ушло бы в переполнение. Отсекаем такие блоки здесь —
    // пиков в них всё равно быть не может.
    if (amountOfPoints < 5) return 0;

    int amountOfPeaks = 0;
    for (size_t i = 2; i + 2 < amountOfPoints; ++i) {
        if (data[i] - data[i - 1] > (numb)pc.eps_peak_delta &&
            data[i] > (numb)pc.peak_threshold &&
            data[i] >= data[i + 1])
        {
            for (size_t j = i; j + 2 < amountOfPoints; ++j) {
                // Наткнулись на точку строго больше — это был не пик.
                if (data[j] < data[j + 1]) { i = j + 1; break; }
                if (data[j] - data[j + 1] > (numb)pc.eps_peak_delta) {
                    if (pc.do_interpolate_peaks) {
                        const numb denom = data[j - 1] - (numb)2.0 * data[j] + data[j + 1];
                        numb delta = (numb)0;
                        if (std::fabs(denom) > (numb)1e-12)
                            delta = (numb)0.5 * (data[j - 1] - data[j + 1]) / denom;
                        outPeaks[amountOfPeaks]    = data[j] - (numb)0.25 * (data[j - 1] - data[j + 1]) * delta;
                        // Здесь индекс, а не время: на время умножаем ниже.
                        timeOfPeaks[amountOfPeaks] = (numb)(j - 1) + delta;
                    } else {
                        outPeaks[amountOfPeaks]    = data[j];
                        timeOfPeaks[amountOfPeaks] = (numb)(j - 1);
                    }
                    ++amountOfPeaks;
                    i = j + 1;   // два пика подряд невозможны
                    break;
                }
            }
        }
    }

    // Anchor-фильтр по eps_interPeak_delta — построчная копия GPU-peakFinder.
    // При eps_interPeak_delta == 0 (дефолт) вырождается в прежний сдвиг влево.
    if (amountOfPeaks > 1) {
        int  writeIdx   = 0;
        numb anchorTime = timeOfPeaks[0];
        for (int i = 1; i < amountOfPeaks; ++i) {
            const numb currentTime = timeOfPeaks[i];
            const numb delta = (currentTime - anchorTime) * h;
            if (delta >= (numb)pc.eps_interPeak_delta) {
                outPeaks[writeIdx]    = outPeaks[i];
                timeOfPeaks[writeIdx] = delta;
                ++writeIdx;
                anchorTime = currentTime;
            }
        }
        amountOfPeaks = writeIdx;
    } else {
        amountOfPeaks = 0;
    }
    if (amountOfPeaks >= pc.max_amount_of_peaks) amountOfPeaks = pc.max_amount_of_peaks;
    return amountOfPeaks;
}


void cpu_curand_init(unsigned long long seed, unsigned long long sequence,
                     unsigned long long offset, CpuRandState* s) {
    unsigned long long z = seed + sequence * 0x9E3779B97F4A7C15ULL + offset;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    s->state = z ^ (z >> 31);
}


float cpu_curand_uniform(CpuRandState* s) {
    s->state = s->state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)(((s->state >> 40) & 0xFFFFFFULL) + 1ULL) / 16777216.0f;
}


// run_bif1d_continuation_cpu — CPU-двойник run_bif1d_continuation.
//
// Зачем: GPU-версия гоняет ВЕСЬ свип в ОДНОМ потоке (gridDim = blockDim = 1, см.
// bifurcation1d_cont.template.cu) — continuation последователен по построению, каждая точка
// стартует с конечного x[] предыдущей, и распараллелить это нельзя, не сломав гистерезис.
// Одиночный GPU-поток на зависимой FP64-цепочке — заведомо медленное железо.
// КРС компилируется в нативную функцию шага через krs_cpu.h: это работает и для custom, и для
// встроенных схем (req.krs_body в обоих случаях обычный C — у встроенных вывод codegen_scheme).
//
// Два отличия от GPU-двойника, оба в лучшую сторону: память — один блок траектории вместо
// матрицы nPts x record_steps (GPU держит её трижды: d_data + d_outPeaks + d_timeOfPeaks); отмена
// и прогресс работают на каждой точке свипа, а не только до запуска монолитного kernel'а.
Bifurcation1DResult run_bif1d_continuation_cpu(const Bifurcation1DRequest& req) {
    Bifurcation1DResult res;
    auto fail = [&](const std::string& msg) -> Bifurcation1DResult& {
        res.error = msg; return res;
    };

    // Та же валидация, что и у GPU-двойника.
    if (req.krs_body.empty())                                 return fail("krs_body пуст");
    if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)   return fail("amountOfX вне диапазона");
    if ((int)req.initial_conditions.size() != req.amountOfX)  return fail("initial_conditions.size() != amountOfX");
    if ((int)req.base_values.size() > kMaxAmountOfValues)     return fail("base_values слишком много");
    if (req.param_index < 0 || req.param_index >= (int)req.base_values.size())
                                                              return fail("param_index вне диапазона");
    if (req.writable_var < -1 || req.writable_var >= req.amountOfX)
                                                              return fail("writable_var вне диапазона");
    if (req.n_pts <= 0)         return fail("n_pts должно быть > 0");
    if (req.h <= 0.0)           return fail("h должно быть > 0");
    if (req.t_max <= 0.0)       return fail("t_max должно быть > 0");
    if (req.transient_time < 0) return fail("transient_time должно быть >= 0");
    if (req.pre_scaller <= 0)   return fail("pre_scaller должно быть > 0");

    KrsCpuStep step;
    std::vector<KrsCpuDiag> diags;
    if (!step.compile(req.krs_body, req.amountOfX, (int)req.base_values.size(), diags)) {
        std::string msg = "CPU KRS:";
        for (const auto& d : diags) {
            msg += "\n";
            if (d.line > 0) msg += "line " + std::to_string(d.line) + ": ";
            msg += d.message;
        }
        return fail(msg);
    }

    // При h-свипе шаг свой в каждой точке, поэтому буфер блока выделяем под
    // ХУДШИЙ случай (минимальный h => больше всего записанных точек), а
    // реальную длину считаем внутри цикла. Так же поступает классический путь.
    // (std::min в скобках: windows.h тянет макрос min, а NOMINMAX тут не задан)
    const double worstCaseH = req.sweep_over_h
                            ? ((req.param_lo < req.param_hi) ? req.param_lo : req.param_hi)
                            : req.h;
    if (worstCaseH <= 0.0) return fail("h должно быть > 0 (при h-свипе — весь диапазон)");
    const int maxPointsInBlock = (int)std::ceil(req.t_max / worstCaseH / req.pre_scaller);
    if (maxPointsInBlock <= 0) return fail("amountOfPointsInBlock <= 0");

    const int nPts = req.n_pts;
    res.n_pts        = nPts;
    res.record_steps = maxPointsInBlock;
    res.param_lo     = req.param_lo;
    res.param_hi     = req.param_hi;
    res.continuation_reverse = req.continuation_reverse;
    res.flags.assign(nPts, 0);
    res.bifurcation_points.assign(nPts, {});
    res.peak_times.assign(nPts, {});

    std::vector<numb> x(req.initial_conditions.begin(), req.initial_conditions.end());
    std::vector<numb> a(req.base_values.begin(), req.base_values.end());
    std::vector<numb> block((size_t)maxPointsInBlock);
    std::vector<numb> peaks((size_t)maxPointsInBlock);
    std::vector<numb> times((size_t)maxPointsInBlock);

    numb h_local = (numb)req.h;

    for (int j = 0; j < nPts; ++j) {
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true;
            res.error = "Cancelled by user";
            return res;
        }
        const double p = cont_sweep_value(j, nPts, req.param_lo, req.param_hi,
                                          req.continuation_reverse, req.log_scale);
        if (req.sweep_over_h) h_local = (numb)p;
        else                  a[(size_t)req.param_index] = p;

        // При h-свипе число записываемых точек своё в каждой точке; буфер
        // выделен под худший случай, поэтому clamp'им к нему.
        int pointsInBlock = (h_local > 0.0)
            ? (int)(req.t_max / h_local / req.pre_scaller) : 0;
        if (pointsInBlock > maxPointsInBlock) pointsInBlock = maxPointsInBlock;
        const size_t pointsForSkip = steps_from_time_size_t(req.transient_time, h_local);
        const numb   timeStep   = h_local * (numb)req.pre_scaller;

        // Вырожденный шаг — траектории нет: это unbound, а не fixed point.
        if (h_local <= 0.0 || pointsInBlock <= 0) {
            res.flags[j] = REGIME_UNBOUND;
            if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
            continue;
        }

        // Transient: x[] НЕ сбрасываем — это и есть перенос с прошлой точки.
        int flag = cpu_loop_model(step.fn(), x.data(), a.data(), h_local,
                                  pointsForSkip, req.amountOfX,
                                  /*preScaller*/ 1, /*writableVar*/ 0,
                                  req.max_value, nullptr);
        if (flag == REGIME_UNBOUND) {
            res.flags[j] = REGIME_UNBOUND;
            if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
            continue;
        }

        flag = cpu_loop_model(step.fn(), x.data(), a.data(), h_local,
                              pointsInBlock, req.amountOfX,
                              req.pre_scaller, req.writable_var,
                              req.max_value, block.data());

        // Зеркалит continuation-ядро (bifurcation1d_cont.template.cu): в
        // d_amountOfPeaks уходит СЫРОЙ REGIME_*, и peakFinderCUDA пропускает и
        // FP, и unbound. Раньше unbound на основном участке проходил дальше, и
        // пики искались в уже разошедшемся блоке.
        if (flag != REGIME_OSCILLATION) {
            res.flags[j] = flag;
            if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
            continue;
        }

        const int n = cpu_peak_finder(block.data(), (size_t)pointsInBlock,
                                      peaks.data(), times.data(), timeStep);
        res.flags[j] = n;
        if (n > 0) {
            res.bifurcation_points[j].assign(peaks.begin(),  peaks.begin() + n);
            res.peak_times[j].assign        (times.begin(),  times.begin() + n);
        }
        if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
    }

    res.ok = true;
    return res;
}


// run_bif1d_cpu — CPU-порт КЛАССИЧЕСКОГО (без continuation) 1D-свипа, двойник run_bif1d. Точки
// независимы: каждая стартует с одних и тех же НУ и базовых параметров, цепочки здесь нет.
// Флаги повторяют связку calculateDiscreteModelCUDA + peakFinderCUDA: kernel пишет
// checker = REGIME_* транзиента, основной участок считается только при OSCILLATION или
// FIXED_POINT, а peakFinderCUDA пропускает FP и UNBOUND, оставляя их коды в массиве.
// Continuation-ветка теперь ведёт себя так же (раньше метила unbound как -1).
// Свипуемая величина — параметр, НУ или сам шаг h, как на GPU; сетка берётся из cont_sweep_value
// (reverse только у continuation), как у CPU-веток LLE/LS/DFT.
Bifurcation1DResult run_bif1d_cpu(const Bifurcation1DRequest& req) {
    Bifurcation1DResult res;
    auto fail = [&](const std::string& msg) -> Bifurcation1DResult& {
        res.error = msg; return res;
    };

    // Диспетчер уже провалидировал req целиком; здесь — только то, от чего
    // зависит своя разметка буферов, плюс дешёвая страховка на прямой вызов.
    if (req.krs_body.empty())                                 return fail("krs_body пуст");
    if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)   return fail("amountOfX вне диапазона");
    if ((int)req.initial_conditions.size() != req.amountOfX)  return fail("initial_conditions.size() != amountOfX");
    if ((int)req.base_values.size() > kMaxAmountOfValues)     return fail("base_values слишком много");
    if (req.writable_var < -1 || req.writable_var >= req.amountOfX)
                                                              return fail("writable_var вне диапазона");
    if (req.n_pts <= 0)         return fail("n_pts должно быть > 0");
    if (req.h <= 0.0)           return fail("h должно быть > 0");
    if (req.t_max <= 0.0)       return fail("t_max должно быть > 0");
    if (req.transient_time < 0) return fail("transient_time должно быть >= 0");
    if (req.pre_scaller <= 0)   return fail("pre_scaller должно быть > 0");

    KrsCpuStep step;
    std::vector<KrsCpuDiag> diags;
    if (!step.compile(req.krs_body, req.amountOfX, (int)req.base_values.size(), diags)) {
        std::string msg = "CPU KRS:";
        for (const auto& d : diags) {
            msg += "\n";
            if (d.line > 0) msg += "line " + std::to_string(d.line) + ": ";
            msg += d.message;
        }
        return fail(msg);
    }

    // При h-свипе шаг свой в каждой точке — буфер под худший случай
    // (минимальный h => больше всего записей), как в continuation-ветке.
    const double worstCaseH = req.sweep_over_h
                            ? ((req.param_lo < req.param_hi) ? req.param_lo : req.param_hi)
                            : req.h;
    if (worstCaseH <= 0.0) return fail("h должно быть > 0 (при h-свипе — весь диапазон)");
    const int maxPointsInBlock = (int)std::ceil(req.t_max / worstCaseH / req.pre_scaller);
    if (maxPointsInBlock <= 0) return fail("amountOfPointsInBlock <= 0");

    const int nPts = req.n_pts;
    res.n_pts        = nPts;
    res.record_steps = maxPointsInBlock;
    res.param_lo     = req.param_lo;
    res.param_hi     = req.param_hi;
    res.flags.assign(nPts, 0);
    res.bifurcation_points.assign(nPts, {});
    res.peak_times.assign(nPts, {});

    // Снимок для right-click экспорта — те же поля, что заполняет run_bif1d.
    res.snapshot.values.assign(req.base_values.begin(), req.base_values.end());
    res.snapshot.initial_conditions.assign(req.initial_conditions.begin(), req.initial_conditions.end());
    res.snapshot.tMax          = req.t_max;
    res.snapshot.transientTime = req.transient_time;
    res.snapshot.h             = req.h;
    res.snapshot.gpu_fmad      = get_nvrtc_fmad();
    res.snapshot.preScaller    = req.pre_scaller;
    res.snapshot.writableVar   = req.writable_var;
    res.snapshot.indexOfMutVar = req.sweep_over_var ? req.var_sweep_index : req.param_index;
    res.snapshot.range_lo      = req.param_lo;
    res.snapshot.range_hi      = req.param_hi;

    const std::string& OUT_FILE_PATH = req.csv_output_path;
    constexpr int set_precision = 15;
    if (!OUT_FILE_PATH.empty()) {
        std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
        data_export::write_bif1d_config(cfg, res.snapshot);
        std::ofstream trunc(OUT_FILE_PATH);   // обнуляем основной файл данных
        trunc.close();
    }
    std::ofstream out;
    if (!OUT_FILE_PATH.empty()) {
        out.open(OUT_FILE_PATH, std::ios::app);
        if (out.is_open()) out << std::setprecision(set_precision);
    }

    std::vector<numb> x((size_t)req.amountOfX);
    std::vector<numb> a(req.base_values.size());
    std::vector<numb> block((size_t)maxPointsInBlock);
    std::vector<numb> peaks((size_t)maxPointsInBlock);
    std::vector<numb> times((size_t)maxPointsInBlock);

    for (int j = 0; j < nPts; ++j) {
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true;
            res.error = "Cancelled by user";
            return res;
        }
        const double p = cont_sweep_value(j, nPts, req.param_lo, req.param_hi,
                                          /*reverse*/ false, req.log_scale);

        // Каждая точка независима — состояние и параметры восстанавливаем.
        x.assign(req.initial_conditions.begin(), req.initial_conditions.end());
        a.assign(req.base_values.begin(), req.base_values.end());

        numb h_local = (numb)req.h;
        if (req.sweep_over_h)        h_local = (numb)p;
        else if (req.sweep_over_var) x[(size_t)req.var_sweep_index] = (numb)p;
        else                         a[(size_t)req.param_index]     = (numb)p;

        int pointsInBlock = (h_local > (numb)0)
            ? (int)(req.t_max / h_local / req.pre_scaller) : 0;
        if (pointsInBlock > maxPointsInBlock) pointsInBlock = maxPointsInBlock;
        const size_t pointsForSkip = steps_from_time_size_t(req.transient_time, (double)h_local);
        // Шаг между записанными сэмплами: h*preScaller. Передаём его в
        // peak-finder напрямую, поэтому поправка time_scale из run_bif1d
        // (там peakFinderCUDA получает один h на весь запуск) здесь не нужна.
        const numb timeStep = h_local * (numb)req.pre_scaller;

        auto finish_point = [&](int flag) {
            res.flags[j] = flag;
            if (out.is_open())
                data_export::write_bif1d_rows(out, p, flag, nullptr, nullptr);
            if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
        };

        // Вырожденный шаг — траектории нет: unbound, а не fixed point.
        if (h_local <= (numb)0 || pointsInBlock <= 0) { finish_point(REGIME_UNBOUND); continue; }

        int flag = cpu_loop_model(step.fn(), x.data(), a.data(), h_local,
                                  pointsForSkip, req.amountOfX,
                                  /*preScaller*/ 1, /*writableVar*/ 0,
                                  req.max_value, nullptr);
        // Ровно как в kernel'е: основной участок считается только при
        // OSCILLATION и FIXED_POINT.
        if (flag == REGIME_OSCILLATION || flag == REGIME_FIXED_POINT)
            flag = cpu_loop_model(step.fn(), x.data(), a.data(), h_local,
                                  pointsInBlock, req.amountOfX,
                                  req.pre_scaller, req.writable_var,
                                  req.max_value, block.data());
        if (flag != REGIME_OSCILLATION) { finish_point(flag); continue; }

        const int n = cpu_peak_finder(block.data(), (size_t)pointsInBlock,
                                      peaks.data(), times.data(), timeStep);
        res.flags[j] = n;
        if (n > 0) {
            res.bifurcation_points[j].assign(peaks.begin(), peaks.begin() + n);
            res.peak_times[j].assign        (times.begin(), times.begin() + n);
        }
        if (out.is_open())
            data_export::write_bif1d_rows(out, p, n,
                                          res.bifurcation_points[j].data(),
                                          res.peak_times[j].data());
        if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
    }

    if (out.is_open()) out.close();
    res.ok = true;
    return res;
}


// run_lle1d_cpu — LLE(param) на CPU. Один код обслуживает два режима.
//
// continuation = false (классический свип): построчный порт LLEKernelCUDA — то, что GPU делает в
//   каждом потоке. Точки независимы: x сбрасывается на initial_conditions, щуп берётся из RNG с
//   subsequence = индекс точки (ровно как idx в kernel'е), прогревается transient_time, дальше
//   цикл Бенеттина. Нужен для сверки CPU-ядра с GPU и для счёта без GPU.
//
// continuation = true: точки выстроены в цепочку, переносится не только траектория x[], но и САМ
//   щуп y[] — к концу точки он уже развёрнут вдоль направления максимального растяжения, и
//   следующей точке не надо разворачивать его заново из случайного направления, отчего кривая
//   lambda(param) выходит глаже. GPU-двойника у режима нет и не будет: цепочка последовательна по
//   построению и выродилась бы в один поток. Прогрев отрабатывается на КАЖДОЙ точке, но начиная
//   со второй идёт NT-блоками с перенормировкой — щуп остаётся прицепленным и не теряет
//   ориентацию, поэтому transient_time сохраняет прежний смысл, а результаты сравнимы с классикой.
//
// Внутри точки алгоритм в обоих режимах одинаков: Бенеттин с одним вектором возмущения,
// перенормировка каждые NT единиц времени, lambda = sum(log(|dX|/eps)) / tMax.
LLE1DResult run_lle1d_cpu(const LLE1DRequest& req, bool continuation) {
    LLE1DResult res;
    auto fail = [&](const std::string& msg) -> LLE1DResult& { res.error = msg; return res; };

    if (req.krs_body.empty())                                 return fail("krs_body пуст");
    if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)   return fail("amountOfX вне диапазона");
    if ((int)req.initial_conditions.size() != req.amountOfX)  return fail("initial_conditions.size() != amountOfX");
    if ((int)req.base_values.size() > kMaxAmountOfValues)     return fail("base_values слишком много");
    if (req.param_index < 0 || req.param_index >= (int)req.base_values.size())
                                                              return fail("param_index вне диапазона");
    if (req.n_pts <= 0)         return fail("n_pts должно быть > 0");
    if (req.h <= 0.0)           return fail("h должно быть > 0");
    if (req.t_max <= 0.0)       return fail("t_max должно быть > 0");
    if (req.transient_time < 0) return fail("transient_time должно быть >= 0");
    if (req.NT <= 0.0)          return fail("NT должно быть > 0");
    if (req.eps <= 0.0)         return fail("eps должно быть > 0");

    KrsCpuStep step;
    std::vector<KrsCpuDiag> diags;
    if (!step.compile(req.krs_body, req.amountOfX, (int)req.base_values.size(), diags)) {
        std::string msg = "CPU KRS:";
        for (const auto& d : diags) {
            msg += "\n";
            if (d.line > 0) msg += "line " + std::to_string(d.line) + ": ";
            msg += d.message;
        }
        return fail(msg);
    }

    const int N            = req.amountOfX;
    const int nBlocks      = (int)(req.t_max / req.NT);   // NT-блоков; от h не зависит
    const int settleBlocks = steps_from_time_int(req.transient_time, req.NT);
    if (nBlocks <= 0) return fail("computed t_max / NT <= 0");
    // Число шагов в NT-блоке и в прогреве зависит от h, а при h-свипе h своё в
    // каждой точке — считаем их внутри цикла.
    int ntSteps   = (int)(req.NT / req.h);
    size_t skipSteps = steps_from_time_size_t(req.transient_time, req.h);
    if (!req.sweep_over_h && ntSteps <= 0) return fail("computed NT / h <= 0");

    const int nPts = req.n_pts;
    res.n_pts    = nPts;
    res.param_lo = req.param_lo;
    res.param_hi = req.param_hi;
    res.continuation_reverse = continuation && req.continuation_reverse;
    res.lyapunov.assign(nPts, std::numeric_limits<double>::quiet_NaN());
    res.flags.assign(nPts, 0);

    std::vector<numb> x(req.initial_conditions.begin(), req.initial_conditions.end());
    std::vector<numb> a(req.base_values.begin(), req.base_values.end());
    std::vector<numb> y((size_t)N, (numb)0);

    // Случайное единичное направление щупа. subsequence = индекс точки: в
    // kernel'е это idx потока, поэтому классический CPU-свип получает те же
    // направления, что и GPU. В continuation цепочка одна на весь свип, и
    // subsequence используется только при разрыве.
    CpuRandState rng;
    auto seed_probe_direction = [&](unsigned long long subsequence) {
        cpu_curand_init(1234567891ULL, subsequence, 0ULL, &rng);
        double zPower = 0.0;
        for (int i = 0; i < N; ++i) { y[i] = cpu_curand_uniform(&rng) - 0.5; zPower += y[i] * y[i]; }
        zPower = std::sqrt(zPower);
        for (int i = 0; i < N; ++i) y[i] = (zPower == 0.0) ? 0.0 : y[i] / zPower;
    };
    seed_probe_direction(0ULL);

    bool probe_attached = false;   // true => y[] уже в абсолютных координатах
    numb h_local = (numb)req.h;    // при h-свипе меняется от точки к точке

    // Один NT-блок: продвинуть x и щуп, вернуть log(|dX|/eps) и вернуть щуп на
    // расстояние eps. Точная копия тела цикла LLEKernelCUDA.
    auto advance_block = [&](numb& out_log) -> bool {
        if (step.ensemble_fn()) {
            if(!step.ensemble_fn()(x.data(),y.data(),1,a.data(),h_local,ntSteps,req.max_value)) return false;
        } else {
        if (cpu_loop_model(step.fn(), x.data(), a.data(), h_local, ntSteps, N,
                           1, 0, req.max_value, nullptr) == 0) return false;
        if (cpu_loop_model(step.fn(), y.data(), a.data(), h_local, ntSteps, N,
                           1, 0, req.max_value, nullptr) == 0) return false;
        }

        double d = 0.0;
        for (int l = 0; l < N; ++l) {
            const double t = (x[l] - y[l]) / req.eps;
            d += t * t;
        }
        d = std::sqrt(d);
        if (d <= 1e-14) d = 1e-14;
        out_log = std::log(d);

        const double inv = 1.0 / d;
        for (int l = 0; l < N; ++l)
            y[l] = x[l] - ((x[l] - y[l] + 1e-14) * inv);
        return true;
    };

    for (int j = 0; j < nPts; ++j) {
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true;
            res.error = "Cancelled by user";
            return res;
        }
        // Свипуемая величина. Классика: направление и log берутся из запроса,
        // reverse не применяется (точки независимы).
        const double p = cont_sweep_value(j, nPts, req.param_lo, req.param_hi,
                                          continuation && req.continuation_reverse,
                                          req.log_scale);
        if (req.sweep_over_h) {
            // Свипуем сам шаг: a[] не трогаем, пересчитываем число шагов.
            h_local   = p;
            ntSteps   = (h_local > 0.0) ? (int)(req.NT / h_local) : 0;
            skipSteps = steps_from_time_size_t(req.transient_time, h_local);
        } else {
            a[(size_t)req.param_index] = p;
        }

        // В классическом режиме состояние не переносится: сбрасываем всё, как
        // делает kernel в начале каждого потока.
        if (!continuation) {
            x.assign(req.initial_conditions.begin(), req.initial_conditions.end());
            seed_probe_direction((unsigned long long)j);
            probe_attached = false;
        }

        if (h_local <= 0.0 || ntSteps <= 0) {
            res.flags[j] = REGIME_UNBOUND;     // вырожденный шаг — точки нет
            if (continuation) probe_attached = false;
            if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
            continue;
        }

        bool alive = true;
        if (!probe_attached) {
            // Первая точка (и любая после разрыва цепочки): выходим на аттрактор
            // одной траекторией, как классический свип, затем цепляем щуп.
            if (cpu_loop_model(step.fn(), x.data(), a.data(), h_local, skipSteps, N,
                               1, 0, req.max_value, nullptr) == 0) {
                alive = false;
            } else {
                for (int i = 0; i < N; ++i) y[i] = y[i] * req.eps + x[i];
                probe_attached = true;
            }
        } else {
            numb dummy;
            for (int b = 0; b < settleBlocks && alive; ++b)
                if (!advance_block(dummy)) alive = false;
        }

        numb sum = (numb)0;
        for (int b = 0; alive && b < nBlocks; ++b) {
            numb lg;
            if (!advance_block(lg)) alive = false;
            else                    sum += lg;
        }

        if (alive) {
            res.lyapunov[j] = (double)(sum / (numb)req.t_max);
            res.flags[j]    = REGIME_OSCILLATION;
        } else {
            // LLE не различает fixed point: единственная причина «не посчиталось»
            // — расходимость. lyapunov[j] остаётся NaN.
            res.flags[j] = REGIME_UNBOUND;
            // Continuation: цепочку рвём — сбрасываем траекторию на IC и берём
            // новое направление щупа, чтобы следующая точка стартовала с
            // чистого листа, а не с разошедшегося состояния.
            if (continuation) {
                probe_attached = false;
                x.assign(req.initial_conditions.begin(), req.initial_conditions.end());
                seed_probe_direction((unsigned long long)j + 1ULL);
            }
        }
        if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
    }

    res.ok = true;
    return res;
}


// Оконная функция для DFT. Все варианты — косинусные суммы
//   w(n) = a0 - a1*cos(g) + a2*cos(2g) - a3*cos(3g),   g = 2*pi*n/(N-1),
// так что таблицы коэффициентов хватает на все типы:
//   0 = None (rectangular)
//   1 = Hanning (default)        -31 дБ, спад -18 дБ/окт
//   2 = Hamming                  -43 дБ, спад  -6 дБ/окт
//   3 = Blackman                 -58 дБ, спад -18 дБ/окт
//   4 = Blackman-Harris (4 чл.)  -92 дБ, спад  -6 дБ/окт
// Уровень боковых лепестков — это и есть динамический диапазон: на Hanning'е
// субгармоники слабее -31 дБ тонут в утечке от основной частоты, что в каскаде
// удвоений периода отрезает всё после третьего-четвёртого удвоения.
//
// Окно нормируется на единичное среднее. Все четыре пути DFT (DFT_custom,
// cpu_dft_block и оба template-ядра) делят сумму на длину блока, а не на
// sum(w), поэтому без нормировки абсолютная амплитуда спектра зависела бы от
// выбора окна — когерентное усиление тут 1.0 / 0.5 / 0.54 / 0.42 / 0.36.
// Форма спектра от нормировки не зависит.
//
// Дублируется в dft_window (dft1d_cont.template.cu) и dft_hsweep_window
// (dft1d_hsweep.template.cu): при h-свипе и в continuation-режиме длина блока
// своя в каждой точке, и окно приходится строить прямо на GPU.
void cpu_build_window(std::vector<numb>& out, int sizeOfBlock, int window_type) {
    out.resize((size_t)sizeOfBlock);
    if (window_type <= 0 || window_type > 4 || sizeOfBlock < 2) {
        std::fill(out.begin(), out.end(), (numb)1);
        return;
    }
    static const double kCoef[5][4] = {
        { 1.0,     0.0,     0.0,     0.0     },   // 0 = None (сюда не доходим)
        { 0.5,     0.5,     0.0,     0.0     },   // 1 = Hanning
        { 0.53836, 0.46164, 0.0,     0.0     },   // 2 = Hamming
        { 0.42,    0.5,     0.08,    0.0     },   // 3 = Blackman
        { 0.35875, 0.48829, 0.14128, 0.01168 },   // 4 = Blackman-Harris
    };
    const double* c = kCoef[window_type];
    const numb gamma = (numb)2.0 * (numb)kPi / (numb)(sizeOfBlock - 1);
    numb sum = (numb)0;
    for (int n = 0; n < sizeOfBlock; ++n) {
        // cos(2g) и cos(3g) через кратные углы, а не тремя вызовами cos:
        // так же считает GPU-версия, иначе пути разошлись бы в последнем бите.
        const numb c1 = std::cos(gamma * (numb)n);
        const numb c2 = (numb)2.0 * c1 * c1 - (numb)1.0;
        const numb c3 = ((numb)4.0 * c1 * c1 - (numb)3.0) * c1;
        const numb w  = (numb)c[0] - (numb)c[1] * c1 + (numb)c[2] * c2 - (numb)c[3] * c3;
        out[(size_t)n] = w;
        sum += w;
    }
    const numb mean = sum / (numb)sizeOfBlock;
    if (mean > (numb)0)
        for (int n = 0; n < sizeOfBlock; ++n) out[(size_t)n] /= mean;
}


// Порт DFT_custom (cudaLibrary.cu) на один блок. Рекуррентный поворот вектора (cos_n, sin_n)
// вместо cos/sin на каждой итерации, со сбросом на точное значение раз в RESET_INTERVAL — как на
// GPU, включая float-точность в самом сбросе (там cosf/sinf).
void cpu_dft_block(const numb* data, int sizeOfBlock, const numb* window,
                   int nFreq, numb freq_lo, numb freq_hi, bool logFreqAxis,
                   numb h, numb* akcos, numb* bksin)
{
    constexpr int RESET_INTERVAL = 1000;
    const numb f_step = (nFreq > 1) ? (freq_hi - freq_lo) / (numb)(nFreq - 1) : (numb)0;
    const numb psi    = (numb)2.0 * (numb)kPi * h;

    for (int k = 0; k < nFreq; ++k) {
        numb ak = (numb)0, bk = (numb)0;
        const numb f_k = logFreqAxis
            ? std::pow((numb)10.0, std::log10(freq_lo) + (nFreq > 1
                  ? (std::log10(freq_hi) - std::log10(freq_lo)) * (numb)k / (numb)(nFreq - 1) : (numb)0))
            : (freq_lo + (numb)k * f_step);
        const numb cos_theta = std::cos((numb)2.0 * (numb)kPi * h * f_k);
        const numb sin_theta = std::sin((numb)2.0 * (numb)kPi * h * f_k);
        numb cos_n = (numb)1, sin_n = (numb)0;
        int reset_counter = RESET_INTERVAL;

        for (int n = 0; n < sizeOfBlock; ++n) {
            const numb wd = window[n] * data[n];
            ak += wd * cos_n;
            bk += wd * sin_n;
            const numb new_cos = cos_n * cos_theta - sin_n * sin_theta;
            const numb new_sin = sin_n * cos_theta + cos_n * sin_theta;
            cos_n = new_cos;
            sin_n = new_sin;
            if (--reset_counter == 0) {
                reset_counter = RESET_INTERVAL;
                if (n + 1 < sizeOfBlock) {
                    const numb exact = psi * f_k * (numb)(n + 1);
                    // cosf/sinf, как в kernel'е: сброс идёт через float.
                    cos_n = (numb)std::cos((float)exact);
                    sin_n = (numb)std::sin((float)exact);
                }
            }
        }
        akcos[k] = ak / (numb)sizeOfBlock;
        bksin[k] = bk / (numb)sizeOfBlock;
    }
}


// Порты projectionOperator и gramSchmidtProcess (cudaLibrary.cu:2832 и :2848).
// Оригиналы объявлены __device__ __host__ и уже работают с runtime-размерностью,
// но живут в .cu под nvcc — из обычного .cpp их не подключить, поэтому копия.
void cpu_projection_operator(const numb* a, const numb* b, numb* minuend, int n) {
    numb numerator = (numb)0, denominator = (numb)0;
    for (int i = 0; i < n; ++i) {
        numerator   += a[i] * b[i];
        denominator += b[i] * b[i];
    }
    const numb fraction = (denominator == (numb)0) ? (numb)0 : numerator / denominator;
    for (int i = 0; i < n; ++i) minuend[i] -= fraction * b[i];
}


// a — входные векторы (n штук по n компонент, row-major), b — ортонормированный
// результат. denominators (опц.) получает длины ДО нормировки — из них LS и
// собирает показатели.
void cpu_gram_schmidt(const numb* a, numb* b, int n, numb* denominators ) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) b[j + i * n] = a[j + i * n];
        for (int j = 0; j < i; ++j)
            cpu_projection_operator(a + i * n, b + j * n, b + i * n, n);
    }
    for (int i = 0; i < n; ++i) {
        numb denominator = (numb)0;
        for (int j = 0; j < n; ++j) denominator += b[i * n + j] * b[i * n + j];
        denominator = std::sqrt(denominator);
        for (int j = 0; j < n; ++j)
            b[i * n + j] = (denominator == (numb)0) ? (numb)0 : b[i * n + j] / denominator;
        if (denominators != nullptr) denominators[i] = denominator;
    }
}


// run_ls1d_cpu — спектр Ляпунова LS(param) на CPU, оба режима (см. run_lle1d_cpu).
// Внутри точки — построчный порт LSKernelCUDA: Бенеттин с N векторами возмущения и
// ортогонализацией Грама-Шмидта каждые NT единиц времени,
// lambda_k = sum(log(denominators[k]/eps)) / tMax.
// В continuation переносятся траектория x[] и ВСЕ N щупов y[] — к концу точки они уже выстроены
// вдоль собственных направлений растяжения. Отличие от LLE только в размере состояния (там один
// щуп, здесь N плюс ортогонализация), поэтому точка дороже примерно в N раз.
LS1DResult run_ls1d_cpu(const LS1DRequest& req, bool continuation) {
    LS1DResult res;
    auto fail = [&](const std::string& msg) -> LS1DResult& { res.error = msg; return res; };

    if (req.krs_body.empty())                                 return fail("krs_body пуст");
    if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)   return fail("amountOfX вне диапазона");
    if ((int)req.initial_conditions.size() != req.amountOfX)  return fail("initial_conditions.size() != amountOfX");
    if ((int)req.base_values.size() > kMaxAmountOfValues)     return fail("base_values слишком много");
    if (req.param_index < 0 || req.param_index >= (int)req.base_values.size())
                                                              return fail("param_index вне диапазона");
    if (req.n_pts <= 0)         return fail("n_pts должно быть > 0");
    if (req.h <= 0.0)           return fail("h должно быть > 0");
    if (req.t_max <= 0.0)       return fail("t_max должно быть > 0");
    if (req.transient_time < 0) return fail("transient_time должно быть >= 0");
    if (req.NT <= 0.0)          return fail("NT должно быть > 0");
    if (req.eps <= 0.0)         return fail("eps должно быть > 0");

    KrsCpuStep step;
    std::vector<KrsCpuDiag> diags;
    if (!step.compile(req.krs_body, req.amountOfX, (int)req.base_values.size(), diags)) {
        std::string msg = "CPU KRS:";
        for (const auto& d : diags) {
            msg += "\n";
            if (d.line > 0) msg += "line " + std::to_string(d.line) + ": ";
            msg += d.message;
        }
        return fail(msg);
    }

    const int N            = req.amountOfX;
    const int nBlocks      = (int)(req.t_max / req.NT);
    const int settleBlocks = steps_from_time_int(req.transient_time, req.NT);
    if (nBlocks <= 0) return fail("computed t_max / NT <= 0");
    // См. run_lle1d_cpu: при h-свипе шаг свой в каждой точке.
    int ntSteps   = (int)(req.NT / req.h);
    size_t skipSteps = steps_from_time_size_t(req.transient_time, req.h);
    if (!req.sweep_over_h && ntSteps <= 0) return fail("computed NT / h <= 0");

    const int nPts = req.n_pts;
    res.n_pts       = nPts;
    res.n_exponents = N;
    res.param_lo    = req.param_lo;
    res.param_hi    = req.param_hi;
    res.continuation_reverse = continuation && req.continuation_reverse;
    res.spectrum.assign(nPts, std::vector<double>((size_t)N,
                        std::numeric_limits<double>::quiet_NaN()));
    res.flags.assign(nPts, 0);

    std::vector<numb> x(req.initial_conditions.begin(), req.initial_conditions.end());
    std::vector<numb> a(req.base_values.begin(), req.base_values.end());
    std::vector<numb> y((size_t)N * N, (numb)0);   // N щупов, абсолютные координаты
    std::vector<numb> z((size_t)N * N, (numb)0);   // рабочий буфер Грама-Шмидта
    std::vector<numb> denominators((size_t)N, (numb)0);
    std::vector<numb> sum((size_t)N, (numb)0);

    CpuRandState rng;
    // Случайный базис из N векторов — та же процедура и тот же RNG, что в
    // kernel'е (subsequence = idx потока).
    auto seed_probe_basis = [&](unsigned long long subsequence) {
        cpu_curand_init(1234567891ULL, subsequence, 0ULL, &rng);
        for (int j = 0; j < N; ++j) {
            double zPower = 0.0;
            for (int i = 0; i < N; ++i) {
                y[(size_t)j * N + i] = cpu_curand_uniform(&rng) - 0.5;
                zPower += y[(size_t)j * N + i] * y[(size_t)j * N + i];
            }
            zPower = std::sqrt(zPower);
            for (int i = 0; i < N; ++i)
                y[(size_t)j * N + i] = (zPower == 0.0) ? 0.0 : y[(size_t)j * N + i] / zPower;
        }
    };
    seed_probe_basis(0ULL);

    bool probes_attached = false;
    numb h_local = (numb)req.h;

    // Один NT-блок: продвинуть траекторию и все щупы, ортогонализовать, вернуть
    // log(denominators[k]/eps) по каждому направлению и вернуть щупы на eps.
    // Точная копия тела цикла LSKernelCUDA.
    auto advance_block = [&](numb* out_logs) -> bool {
        if (step.ensemble_fn()) {
            if(!step.ensemble_fn()(x.data(),y.data(),N,a.data(),h_local,ntSteps,req.max_value)) return false;
        } else {
        if (cpu_loop_model(step.fn(), x.data(), a.data(), h_local, ntSteps, N,
                           1, 0, req.max_value, nullptr) == 0) return false;
        for (int j = 0; j < N; ++j)
            if (cpu_loop_model(step.fn(), y.data() + (size_t)j * N, a.data(), h_local,
                               ntSteps, N, 1, 0, req.max_value, nullptr) == 0) return false;
        }

        for (int k = 0; k < N; ++k)
            for (int l = 0; l < N; ++l)
                y[(size_t)k * N + l] -= x[l];

        cpu_gram_schmidt(y.data(), z.data(), N, denominators.data());

        for (int k = 0; k < N; ++k) {
            if (out_logs) out_logs[k] = std::log(denominators[k] / req.eps);
            for (int j = 0; j < N; ++j)
                y[(size_t)k * N + j] = x[j] + z[(size_t)k * N + j] * req.eps;
        }
        return true;
    };

    std::vector<numb> logs((size_t)N, (numb)0);

    for (int j = 0; j < nPts; ++j) {
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true;
            res.error = "Cancelled by user";
            return res;
        }
        const double p = cont_sweep_value(j, nPts, req.param_lo, req.param_hi,
                                          continuation && req.continuation_reverse,
                                          req.log_scale);
        if (req.sweep_over_h) {
            h_local   = p;
            ntSteps   = (h_local > 0.0) ? (int)(req.NT / h_local) : 0;
            skipSteps = steps_from_time_size_t(req.transient_time, h_local);
        } else {
            a[(size_t)req.param_index] = p;
        }

        if (!continuation) {
            x.assign(req.initial_conditions.begin(), req.initial_conditions.end());
            seed_probe_basis((unsigned long long)j);
            probes_attached = false;
        }

        if (h_local <= 0.0 || ntSteps <= 0) {
            res.flags[j] = REGIME_UNBOUND;     // вырожденный шаг — точки нет
            if (continuation) probes_attached = false;
            if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
            continue;
        }

        bool alive = true;
        if (!probes_attached) {
            if (cpu_loop_model(step.fn(), x.data(), a.data(), h_local, skipSteps, N,
                               1, 0, req.max_value, nullptr) == 0) {
                alive = false;
            } else {
                // z = ортонормированный базис из y, затем щупы ставятся на eps
                // вокруг текущей точки траектории (порядок как в kernel'е).
                cpu_gram_schmidt(y.data(), z.data(), N);
                for (int k = 0; k < N; ++k)
                    for (int i = 0; i < N; ++i)
                        y[(size_t)k * N + i] = z[(size_t)k * N + i] * req.eps + x[i];
                probes_attached = true;
            }
        } else {
            for (int b = 0; b < settleBlocks && alive; ++b)
                if (!advance_block(nullptr)) alive = false;
        }

        std::fill(sum.begin(), sum.end(), 0.0);
        for (int b = 0; alive && b < nBlocks; ++b) {
            if (!advance_block(logs.data())) { alive = false; break; }
            for (int k = 0; k < N; ++k) sum[k] += logs[k];
        }

        if (alive) {
            for (int k = 0; k < N; ++k) res.spectrum[j][k] = (double)(sum[k] / (numb)req.t_max);
            res.flags[j] = REGIME_OSCILLATION;
        } else {
            // Как и LLE, LS не различает fixed point — только расходимость.
            res.flags[j] = REGIME_UNBOUND;      // spectrum[j] остаётся NaN
            if (continuation) {
                probes_attached = false;
                x.assign(req.initial_conditions.begin(), req.initial_conditions.end());
                seed_probe_basis((unsigned long long)j + 1ULL);
            }
        }
        if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
    }

    res.ok = true;
    return res;
}


// run_dft1d_cpu — 1D DFT на CPU, оба режима (см. run_lle1d_cpu).
// continuation = false: точки независимы, x сбрасывается на initial_conditions (при IC-свипе
//   свипуемая координата подменяется). continuation = true: x[] переносится с предыдущей точки,
//   как в Bifurcation1D — тангенциальных векторов здесь нет.
// Внутри точки: transient, запись блока writable_var, окно, DFT (cpu_dft_block — порт
// DFT_custom). При h-свипе число сэмплов и окно пересчитываются под шаг точки; буфер выделен под
// худший случай.
Dft1DResult run_dft1d_cpu(const Dft1DRequest& req, bool continuation) {
    Dft1DResult res;
    auto fail = [&](const std::string& msg) -> Dft1DResult& { res.error = msg; return res; };

    if (req.krs_body.empty())                                 return fail("krs_body пуст");
    if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)   return fail("amountOfX вне диапазона");
    if ((int)req.initial_conditions.size() != req.amountOfX)  return fail("initial_conditions.size() != amountOfX");
    if ((int)req.base_values.size() > kMaxAmountOfValues)     return fail("base_values слишком много");
    if (!req.sweep_over_h && !req.sweep_over_var &&
        (req.param_index < 0 || req.param_index >= (int)req.base_values.size()))
                                                              return fail("param_index вне диапазона");
    if (req.sweep_over_var &&
        (req.var_sweep_index < 0 || req.var_sweep_index >= req.amountOfX))
                                                              return fail("var_sweep_index вне диапазона");
    if (req.writable_var < -1 || req.writable_var >= req.amountOfX)
                                                              return fail("writable_var вне диапазона");
    if (req.n_pts <= 0)         return fail("n_pts должно быть > 0");
    if (req.n_freq <= 0)        return fail("n_freq должно быть > 0");
    if (req.h <= 0.0)           return fail("h должно быть > 0");
    if (req.t_max <= 0.0)       return fail("t_max должно быть > 0");
    if (req.transient_time < 0) return fail("transient_time должно быть >= 0");
    if (req.pre_scaller <= 0)   return fail("pre_scaller должно быть > 0");
    if (req.freq_log_scale && !(req.freq_lo > 0.0 && req.freq_hi > 0.0))
                                return fail("log-сетка по частоте требует freq lo/hi > 0");

    KrsCpuStep step;
    std::vector<KrsCpuDiag> diags;
    if (!step.compile(req.krs_body, req.amountOfX, (int)req.base_values.size(), diags)) {
        std::string msg = "CPU KRS:";
        for (const auto& d : diags) {
            msg += "\n";
            if (d.line > 0) msg += "line " + std::to_string(d.line) + ": ";
            msg += d.message;
        }
        return fail(msg);
    }

    const double worstCaseH = req.sweep_over_h
                            ? ((req.param_lo < req.param_hi) ? req.param_lo : req.param_hi)
                            : req.h;
    if (worstCaseH <= 0.0) return fail("h должно быть > 0 (при h-свипе — весь диапазон)");
    const int maxPointsInBlock = (int)std::ceil(req.t_max / worstCaseH / req.pre_scaller);
    if (maxPointsInBlock <= 0) return fail("amountOfPointsInBlock <= 0");

    const int nPts  = req.n_pts;
    const int nFreq = req.n_freq;
    res.n_pts   = nPts;
    res.n_freq  = nFreq;
    res.param_lo = req.param_lo;
    res.param_hi = req.param_hi;
    res.freq_lo  = req.freq_lo;
    res.freq_hi  = req.freq_hi;
    res.continuation_reverse = continuation && req.continuation_reverse;
    res.ak_cos.assign((size_t)nPts * nFreq, 0.0);
    res.bk_sin.assign((size_t)nPts * nFreq, 0.0);
    res.flags.assign(nPts, 0);

    std::vector<numb> x(req.initial_conditions.begin(), req.initial_conditions.end());
    std::vector<numb> a(req.base_values.begin(), req.base_values.end());
    std::vector<numb> block((size_t)maxPointsInBlock);
    std::vector<numb> window;
    std::vector<numb> ak_tmp((size_t)nFreq), bk_tmp((size_t)nFreq);
    int window_len = -1;                 // окно перестраиваем только при смене длины

    numb h_local = (numb)req.h;

    for (int j = 0; j < nPts; ++j) {
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true;
            res.error = "Cancelled by user";
            return res;
        }
        const double p = cont_sweep_value(j, nPts, req.param_lo, req.param_hi,
                                          continuation && req.continuation_reverse,
                                          req.log_scale);

        if (!continuation)
            x.assign(req.initial_conditions.begin(), req.initial_conditions.end());

        if (req.sweep_over_h)        h_local = p;
        else if (req.sweep_over_var) x[(size_t)req.var_sweep_index] = p;
        else                         a[(size_t)req.param_index] = p;

        int pointsInBlock = (h_local > 0.0) ? (int)(req.t_max / h_local / req.pre_scaller) : 0;
        if (pointsInBlock > maxPointsInBlock) pointsInBlock = maxPointsInBlock;
        const size_t pointsForSkip = steps_from_time_size_t(req.transient_time, h_local);

        auto mark_dead = [&](int flag) {
            res.flags[j] = flag;
            // Как DFT_custom: FIXED_POINT -> -1.0, UNBOUND -> 0.0.
            const double fill = (flag == REGIME_FIXED_POINT) ? -1.0 : 0.0;
            for (int k = 0; k < nFreq; ++k) {
                res.ak_cos[(size_t)j * nFreq + k] = fill;
                res.bk_sin[(size_t)j * nFreq + k] = fill;
            }
            if (continuation) x.assign(req.initial_conditions.begin(), req.initial_conditions.end());
            if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
        };

        // Вырожденный шаг — траектории нет: unbound, а не fixed point.
        if (h_local <= 0.0 || pointsInBlock <= 2) { mark_dead(REGIME_UNBOUND); continue; }

        int flag = cpu_loop_model(step.fn(), x.data(), a.data(), h_local,
                                  pointsForSkip, req.amountOfX,
                                  /*preScaller*/ 1, /*writableVar*/ 0,
                                  req.max_value, nullptr);
        if (flag == REGIME_UNBOUND) { mark_dead(REGIME_UNBOUND); continue; }

        flag = cpu_loop_model(step.fn(), x.data(), a.data(), h_local,
                              pointsInBlock, req.amountOfX,
                              req.pre_scaller, req.writable_var,
                              req.max_value, block.data());
        // Сырой REGIME_* и для continuation, и для классики: раньше в
        // continuation unbound подменялся на 1 и спектр считался по
        // разошедшемуся блоку (см. dft1d_cont.template.cu — там та же правка).
        if (flag != REGIME_OSCILLATION) { mark_dead(flag); continue; }

        if (window_len != pointsInBlock) {
            cpu_build_window(window, pointsInBlock, req.window_type);
            window_len = pointsInBlock;
        }
        // Считаем в numb (как GPU), затем расширяем в double-хранилище результата.
        cpu_dft_block(block.data(), pointsInBlock, window.data(),
                      nFreq, (numb)req.freq_lo, (numb)req.freq_hi, req.freq_log_scale,
                      h_local * (numb)req.pre_scaller,
                      ak_tmp.data(), bk_tmp.data());
        for (int k = 0; k < nFreq; ++k) {
            res.ak_cos[(size_t)j * nFreq + k] = (double)ak_tmp[(size_t)k];
            res.bk_sin[(size_t)j * nFreq + k] = (double)bk_tmp[(size_t)k];
        }
        res.flags[j] = REGIME_OSCILLATION;
        if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
    }

    res.ok = true;
    return res;
}
}
