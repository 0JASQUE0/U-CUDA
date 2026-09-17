#include "session_internal.h"

namespace ucuda::analysis_detail {


// Чистая функция: входы → AnalysisResult. Не трогает session, поэтому её
// безопасно звать с любого потока.
AnalysisResult compute_phase_portrait(PhaseRunInputs in) {
    AnalysisResult result;
    auto _t0 = std::chrono::high_resolution_clock::now();

    if (in.adaptive.method != 0) {
        try { in.krs_code = codegen_adaptive(in.sys, in.scheme, in.krs_code, in.krs_is_custom, in.adaptive); }
        catch (const std::exception& e) { result.error = e.what(); return result; }
    }
    int dim = (int)in.vars.size();
    if (dim < 1) { result.error = "no variables"; return result; }
    double h = parse_val(in.step_h, 0.01);
    if (!std::isfinite(h) || h <= 0) { result.error = "step h must be finite and > 0"; return result; }
    double tsim = parse_val(in.sim_time, 50.0);
    double tskip = parse_val(in.skip_time, 0.0);
    if (!std::isfinite(tsim) || !std::isfinite(tskip) || tsim <= 0 || tskip < 0 ||
        tsim/h > 2147483646.0 || tskip/h > 2147483646.0) {
        result.error="Invalid simulation/skip time or too many output samples"; return result;
    }
    int total = (int)(tsim / h); if (total <= 0) total = 1;
    int skip = in.skip_transient ? (int)(tskip / h) : 0;
    if (skip < 0) skip = 0;

    // параметры: a[0] зарезервирован под CD-коэф. симметрии, a[1..] = params
    int nparams = (int)in.params.size();
    std::vector<double> a(nparams + 1, 0.0);
    a[0] = parse_val(in.symmetry_s, 0.5);
    for (int j = 0; j < nparams; ++j) {
        auto it = in.param_values.find(in.params[j]);
        a[1 + j] = parse_val(it != in.param_values.end() ? it->second : "", 0.0);
    }

    int dec = parse_num_int(in.decimation, 1); if (dec < 1) dec = 1;

    int N = (int)in.ic_sets.size();
    if (N < 1) { result.error = "no initial conditions"; return result; }

    std::vector<double> ic_flat((size_t)N * dim, 0.0);
    // Continuation seeds ic_flat from the previous chunk's final state; skip
    // parsing so a Lorenz-scale double reaches the kernel bit-exact.
    const bool use_override = (int)in.ic_override.size() == N &&
                              (in.ic_override.empty() || (int)in.ic_override[0].size() >= dim);
    for (int k = 0; k < N; ++k) {
        if (use_override) {
            for (int i = 0; i < dim; ++i)
                ic_flat[(size_t)k * dim + i] = in.ic_override[k][i];
        }
        else {
            const auto& ic = in.ic_sets[k];
            for (int i = 0; i < dim; ++i) {
                auto it = ic.values.find(in.vars[i]);
                ic_flat[(size_t)k * dim + i] = parse_val(it != ic.values.end() ? it->second : "", 0.0);
            }
        }
    }

    std::vector<std::vector<std::vector<double>>> raw;
    bool calc_ok = true;
    std::string calc_err;

    if (in.use_gpu) {
        if (in.krs_code.empty()) {
            result.error = "no KRS generated (check system)";
            return result;
        }
        calc_ok = computePhasePortraitsNVRTC(in.krs_code, dim,
            ic_flat.data(), N, a.data(), (int)a.size(),
            h, total, skip, raw, &calc_err, in.adaptive.method ? &result.adaptive_stats : nullptr);
        if (!calc_ok) {
            result.error = "GPU: " + calc_err;
            return result;
        }
    }
    else if (in.krs_is_custom || in.adaptive.method != 0) {
        // Custom КРС: тело — сырой C с объявлениями, циклами и локальными
        // массивами. SystemEvaluator интерпретирует только ВЫРАЖЕНИЯ правых
        // частей, поэтому тело компилируется в нативную функцию шага
        // (см. krs_cpu.h) и гоняется тем же циклом, что и встроенные схемы.
        KrsCpuStep step;
        std::vector<KrsCpuDiag> diags;
        if (!step.compile(in.krs_code, dim, (int)a.size(), diags)) {
            std::string msg = "CPU custom KRS:";
            for (const auto& d : diags) {
                msg += "\n";
                if (d.line > 0) msg += "line " + std::to_string(d.line) + ": ";
                msg += d.message;
            }
            result.error = msg;
            return result;
        }
        raw.resize(N);
        for (int k = 0; k < N; ++k) {
            bool ok = true;
            if (in.adaptive.method != 0) {
                if (!step.trajectory_fn()) { result.error="CPU adaptive trajectory entry point missing"; return result; }
                std::vector<double> flat((size_t)total*dim);
                adaptive::Stats stats;
                step.trajectory_fn()(&ic_flat[(size_t)k*dim],a.data(),h,total,skip,flat.data(),&stats);
                result.adaptive_stats.push_back(stats);
                if (stats.status != adaptive::Success) {
                    result.error=std::string("CPU: ")+adaptive::status_text(stats.status); return result;
                }
                raw[k].resize(total);
                for(int t=0;t<total;++t) raw[k][t].assign(flat.begin()+(size_t)t*dim,flat.begin()+(size_t)(t+1)*dim);
            } else ok = computePhasePortraitCPU_custom(step.fn(),
                &ic_flat[(size_t)k * dim], dim, a.data(), (int)a.size(), h, total, skip, raw[k]);
            if (!ok) { calc_ok = false; calc_err = "trajectory '" + in.ic_sets[k].label + "' diverged (nan/inf)"; }
        }
    }
    else {
        std::unique_ptr<SystemEvaluator> evaluator;
        try {
            evaluator.reset(new SystemEvaluator(in.sys));
        }
        catch (const std::exception& e) {
            result.error = std::string("parse error: ") + e.what();
            return result;
        }
        raw.resize(N);
        for (int k = 0; k < N; ++k) {
            bool ok = computePhasePortraitCPU(*evaluator, int_scheme_from_string(in.scheme),
                &ic_flat[(size_t)k * dim], dim, a.data(), (int)a.size(),
                h, total, skip, raw[k]);
            if (!ok) { calc_ok = false; calc_err = "trajectory '" + in.ic_sets[k].label + "' diverged (nan/inf)"; }
        }
    }

    result.features.resize(N);
    // Preallocated so a divergent trajectory (empty traj[]) still leaves a
    // zero-length slot at the same index as ic_sets[k].
    result.final_states.assign(N, std::vector<double>{});
    for (int k = 0; k < N; ++k) {
        const auto& ic = in.ic_sets[k];
        std::vector<std::vector<double>>& traj = raw[k];
        if (!traj.empty()) result.final_states[k] = traj.back();

        // Признаки (пики + интервалы) считаем по ПРОРЕЖЕННОМУ ряду и с шагом h*decimator — ровно
        // тот сигнал и тот шаг, что получает peakFinderCUDA в 2D-бифуркации (timeStep = h *
        // preScaller в parametric_engine).
        // Раньше здесь брался полноразрешающий raw с шагом h, «чтобы decimator не прорядил сами
        // пики». Формально так и было, но диаграмма из-за этого переставала предсказывать карту:
        // критерии пика (data[i]-data[i-1] > eps_peak_delta, data[i] >= data[i+1]) и интерполяция
        // вершины по трём соседям чувствительны к шагу дискретизации, и на вчетверо более редкой
        // сетке ядро находило другой набор пиков. Совпадение с DBSCAN важнее разрешения: диаграмма
        // нужна, чтобы подбирать eps и множители. При decimator == 1 обе ветки эквивалентны.
        //
        // Слотов dim + 1: последний (индекс dim) — комбинация переменных, тот же ряд, что ядро
        // строит при writable_var == -1 (x0 + pi*x1 + euler*x2, с деградацией при dim < 3).
        // Константы берём из configCUDA.h, чтобы диаграмма и бифуркация шли по одному ряду.
        result.features[k].resize((size_t)dim + 1);
        {
            const size_t stride = (dec > 1) ? (size_t)dec : (size_t)1;
            std::vector<double> series;
            series.reserve(traj.size() / stride + 1);
            for (int v = 0; v < dim; ++v) {
                series.clear();
                for (size_t t = 0; t < traj.size(); t += stride) {
                    const auto& pt = traj[t];
                    series.push_back(v < (int)pt.size() ? pt[v] : 0.0);
                }
                result.features[k][v] = find_peaks_host(series, h * (double)stride);
            }

            series.clear();
            for (size_t t = 0; t < traj.size(); t += stride) {
                const auto& pt = traj[t];
                double c = pt.size() > 0 ? pt[0] : 0.0;
                if (dim >= 2 && pt.size() > 1) c += ::pi    * pt[1];
                if (dim >= 3 && pt.size() > 2) c += ::euler * pt[2];
                series.push_back(c);
            }
            result.features[k][(size_t)dim] = find_peaks_host(series, h * (double)stride);
        }

        // RQA по этой траектории — ДО прореживания и до move(traj) ниже. Считаем по полному
        // ряду с шагом h: decimator задуман как настройка отрисовки, и наследовать его здесь
        // значило бы молча менять частоту дискретизации, от которой RQA зависит напрямую.
        // Прореживание до Config::points делает сам rqa::compute и возвращает честный dt.
        for (const RqaJob& job : in.rqa_jobs) {
            if (job.ic != k) continue;
            RqaOutcome o;
            o.job = job;
            if (traj.empty()) o.res.error = "RQA: trajectory diverged";
            else rqa::compute(traj, h, job.cfg, o.res);
            result.rqa_out.push_back(std::move(o));
        }

        std::vector<std::vector<double>> dtraj;
        if (dec <= 1) dtraj = std::move(traj);
        else {
            dtraj.reserve(traj.size() / dec + 1);
            for (size_t t = 0; t < traj.size(); t += dec) dtraj.push_back(traj[t]);
        }
        result.trajectories.push_back(std::move(dtraj));
        result.labels.push_back(ic.label);
        result.visible.push_back(ic.visible);

        std::string txt = "(";
        for (int i = 0; i < dim; ++i) {
            auto it = ic.values.find(in.vars[i]);
            std::string val = it != ic.values.end() ? it->second : "";
            if (val.empty()) val = "0";
            txt += val; if (i + 1 < dim) txt += ",";
        }
        txt += ")";
        result.ic_text.push_back(txt);
    }

    result.ok = result.trajectories.size() > 0;
    if (!calc_ok && result.ok) result.error = calc_err;

    // Snapshot of CSV-relevant inputs for the GUI right-click export.
    result.snapshot.vars      = in.vars;
    result.snapshot.params    = in.params;
    result.snapshot.a         = a;
    result.snapshot.scheme    = in.scheme;
    result.snapshot.adaptive = in.adaptive;
    result.snapshot.h         = h;
    // Настройка FMA на момент прогона — в CSV, см. Bif1DSnapshot::gpu_fmad.
    // Пишется одинаково для GPU- и CPU-ветки: это состояние настройки, а не
    // утверждение об устройстве (на CPU NVRTC не участвует вовсе).
    result.snapshot.gpu_fmad  = get_nvrtc_fmad();
    result.snapshot.t_max     = tsim;
    result.snapshot.t_skip    = tskip;
    result.snapshot.decimator = dec;
    // Snapshot IC from the user's panel text — not ic_flat, which continuation overwrites
    // with the previous chunk's final X[] so the legend would flicker every frame.
    result.snapshot.ic_flat.assign(N, std::vector<double>(dim, 0.0));
    result.snapshot.ic_labels.reserve(N);
    for (int k = 0; k < N; ++k) {
        const auto& ic = in.ic_sets[k];
        for (int i = 0; i < dim; ++i) {
            auto it = ic.values.find(in.vars[i]);
            result.snapshot.ic_flat[k][i] = parse_val(it != ic.values.end() ? it->second : "", 0.0);
        }
        result.snapshot.ic_labels.push_back(in.ic_sets[k].label);
    }

    auto _t1 = std::chrono::high_resolution_clock::now();
    double _ms = std::chrono::duration<double, std::milli>(_t1 - _t0).count();
    if (result.error.empty())
        result.error = "recompute: " + std::to_string(_ms) + " ms";

    return result;
}


// Снапшот текущей session в PhaseRunInputs. Делается на главном потоке.
PhaseRunInputs snapshot_phase(PhaseAnalysisSession& s) {
    // krs_code должен быть свежим. Раньше регенерировали только при пустом кеше — это ломало
    // правки тела custom-scheme: первый Run пёк krs_code из старого cs.body, дальнейшие правки в
    // Library на него не влияли. Регенерация дешёвая (codegen или просто копия cs.body), поэтому
    // делаем её каждый Run.
    s.regenerate_krs();
    PhaseRunInputs in;
    in.vars         = s.vars;
    in.params       = s.params;
    in.step_h       = s.step_h;
    in.sim_time     = s.sim_time;
    in.skip_time    = s.skip_time;
    in.scheme       = s.scheme;
    in.adaptive = s.adaptive;
    in.symmetry_s   = s.symmetry_s;
    in.decimation   = s.decimation;
    in.use_gpu      = s.use_gpu;
    in.sys          = s.sys;
    in.krs_code     = s.krs_code;
    in.krs_is_custom = false;
    for (const auto& cs : s.custom_schemes)
        if (cs.name == s.scheme) { in.krs_is_custom = true; break; }
    in.param_values = s.param_values;
    in.ic_sets      = s.ic_sets;
    // Continuation: on frames >= 1 resume from the previous chunk's final X[]
    // and skip the transient (already burned in). Size mismatch (user added or
    // removed an IC mid-run) falls back to first-frame semantics.
    if (s.continuation_active && !s.continuation_first_frame &&
        s.continuation_state.size() == s.ic_sets.size()) {
        in.ic_override    = s.continuation_state;
        in.skip_transient = false;
    }

    // RQA-задачи из открытых окон типа RecurrencePlot. Дедуп по (ic, cfg): два окна с одинаковыми
    // настройками (например «то же, но бинарный вид») обязаны стоить один расчёт, а не два.
    // Проверки на "окно закрыто" тут нет намеренно: крестик окна проекции УДАЛЯЕТ проекцию
    // (pr_to_remove в draw_projection_windows), состояния "есть, но закрыта" не существует —
    // Projection::open остался от старой схемы и никем не читается.
    for (const Projection& pr : s.projections) {
        if (s.continuation_active && !pr.rqa_in_continuation) continue;
        RqaJob job;
        if (!make_rqa_job(pr, job)) continue;
        if (job.ic < 0 || job.ic >= (int)s.ic_sets.size()) continue;
        bool dup = false;
        for (const RqaJob& have : in.rqa_jobs) if (have == job) { dup = true; break; }
        if (!dup) in.rqa_jobs.push_back(job);
    }
    return in;
}
}
using namespace ucuda::analysis_detail;


void PhaseAnalysisSession::add_ic() {
    InitialConditionSet ic;
    ic.label = "IC " + std::to_string(ic_sets.size() + 1);
    for (const auto& v : vars) ic.values[v] = "";
    ic_sets.push_back(ic);
}


void PhaseAnalysisSession::remove_ic(int i) {
    if (i >= 0 && i < (int)ic_sets.size()) ic_sets.erase(ic_sets.begin() + i);
}




void PhaseAnalysisSession::remove_projection(int i) {
    if (i >= 0 && i < (int)projections.size()) {
        projections.erase(projections.begin() + i);
    }
}


void PhaseAnalysisSession::load_from_record(const SystemRecord& r,
    const std::vector<std::string>& vars_,
    const std::vector<std::string>& params_) {
    vars = vars_;
    params = params_;
    custom_schemes = r.custom_schemes;
    enabled_builtin_schemes = enabled_builtins_from_record(r);
    adaptive = {};
    step_h = r.step_h.empty() ? "0.01" : r.step_h;
    symmetry_s = r.symmetry_s.empty() ? "0.5" : r.symmetry_s;
    sim_time = "50";
    skip_time = "10";

    param_values.clear();
    for (const auto& p : params) {
        auto it = r.param_values.find(p);
        param_values[p] = (it != r.param_values.end()) ? it->second : "";
    }

    ic_sets.clear();
    InitialConditionSet ic;
    ic.label = "IC 1";
    for (const auto& v : vars) {
        auto it = r.init_conditions.find(v);
        ic.values[v] = (it != r.init_conditions.end()) ? it->second : "";
    }
    ic_sets.push_back(ic);

    // одна проекция по умолчанию
    projections.clear();
    add_projection();

    result = AnalysisResult{};
}


void PhaseAnalysisSession::recompute() {
    PhaseRunInputs in = snapshot_phase(*this);
    result = compute_phase_portrait(in);
    fit_request = true;
    data_generation++;
}


bool PhaseAnalysisSession::recompute_async() {
    if (in_flight) return false;
    PhaseRunInputs in = snapshot_phase(*this);
    in_flight = true;
    compute_start_time = std::chrono::steady_clock::now();
    recompute_future = std::async(std::launch::async, [in = std::move(in)]() {
        return compute_phase_portrait(in);
    });
    return true;
}


bool PhaseAnalysisSession::poll() {
    if (!in_flight) return false;
    if (!recompute_future.valid()) { in_flight = false; return false; }
    if (recompute_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
    result = recompute_future.get();
    fit_request = true;
    data_generation++;
    in_flight = false;
    // Carry the final X[] forward for the next continuation chunk. Only
    // adopt on success so a divergent frame doesn't corrupt the seed.
    if (continuation_active && result.ok &&
        result.final_states.size() == ic_sets.size()) {
        const double chunk_t = result.snapshot.t_max;
        const double base    = continuation_elapsed;
        continuation_elapsed += chunk_t;

        const int N = (int)ic_sets.size();
        if ((int)continuation_peaks.size() != N) continuation_peaks.assign((size_t)N, {});
        for (int k = 0; k < N && k < (int)result.features.size(); ++k) {
            auto& per_var_out = continuation_peaks[(size_t)k];
            const auto& per_var_in = result.features[(size_t)k];
            if (per_var_out.size() != per_var_in.size())
                per_var_out.assign(per_var_in.size(), {});
            for (size_t v = 0; v < per_var_in.size(); ++v) {
                const FeaturePoints& fp = per_var_in[v];
                std::vector<float>& buf = per_var_out[v];
                double cumt = base;
                const size_t n = fp.peaks.size() < fp.intervals.size()
                               ? fp.peaks.size() : fp.intervals.size();
                buf.reserve(buf.size() + n * 2);
                for (size_t p = 0; p < n; ++p) {
                    if (!std::isfinite(fp.peaks[p]) || !std::isfinite(fp.intervals[p])) continue;
                    cumt += fp.intervals[p];
                    buf.push_back((float)cumt);
                    buf.push_back((float)fp.peaks[p]);
                }
                const size_t cap_pairs = (size_t)std::max(1, continuation_peaks_cap);
                const size_t cur_pairs = buf.size() / 2;
                if (cur_pairs > cap_pairs) {
                    const size_t drop_pairs = cur_pairs - cap_pairs;
                    buf.erase(buf.begin(), buf.begin() + (std::ptrdiff_t)(drop_pairs * 2));
                }
            }
        }
        ++continuation_peaks_gen;

        continuation_state       = std::move(result.final_states);
        continuation_first_frame = false;
    }
    return true;
}


void PhaseAnalysisSession::regenerate_krs() {
    krs_code.clear();
    // Сначала ищем среди custom — имя имеет приоритет над built-in (если бы
    // совпали, что мы блокируем в System tab).
    bool found_custom = false;
    for (const auto& cs : custom_schemes) {
        if (cs.name == scheme) { krs_code = cs.body; found_custom = true; break; }
    }
    if (!found_custom && !sys.rhs.empty()) {
        try {
            krs_code = codegen_scheme(sys, scheme_from_string(scheme));
        }
        catch (...) {
            krs_code.clear();
        }
    }

    // Фоновый прогрев NVRTC-кэша под новую систему/метод, не дожидаясь Run. Новый прогрев не
    // запускаем, пока предыдущий не закончился: присвоение prewarm_future иначе заблокировало бы
    // ЭТОТ (UI) поток на деструкторе предыдущей async-future. Best-effort — если не успеваем
    // прогреть, настоящий Run просто скомпилирует синхронно.
    bool prewarm_busy = prewarm_future.valid() &&
        prewarm_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
    if (use_gpu && !krs_code.empty() && !prewarm_busy) {
        std::string krs = krs_code;
        int dim = (int)vars.size();
        prewarm_future = std::async(std::launch::async, [krs, dim] {
            prewarmPhasePortraitsNVRTC(krs, dim);
        });
    }
}
