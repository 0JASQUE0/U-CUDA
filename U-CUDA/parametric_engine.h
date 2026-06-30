#pragma once
//
// parametric_engine — фасад для параметрического анализа через NVRTC.
//
// Цель: изолировать остальной проект (GUI, AppModel, session_io) от внутренностей
// hostLibrary.cu / cudaLibrary.cu / main_NonLinAnal.cu. Эти файлы дорабатываются
// внешне и иногда меняются — здесь стабильный, проектный API, который при
// обновлении NonLinAnal правится в одном-двух местах, а не по всему коду.
//
// Архитектура:
//   1. Adapter (этот файл + .cpp): хранит стабильный API
//      run_bifurcation_1d/2d/lle/etc. и реализацию через NVRTC.
//   2. Шаблоны ядер: U-CUDA/kernels/*.template.cu — текстовые файлы,
//      копируются Post-Build Event в OutDir рядом с exe. NVRTC грузит и
//      подставляет КРС-тело + размерность.
//   3. Кэш PTX по хэшу (КРС + размерность) — чтобы не перекомпилировать
//      при смене только параметров анализа.
//

#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include "configCUDA.h"   // typedef numb, BF_* feature codes, mult_avg_* defaults
#include "data_export.h"  // snapshot structs embedded into *Result for right-click export

struct Bifurcation1DRequest {
    // КРС
    std::string krs_body;                    // тело calculateDiscreteModel из codegen
    int amountOfX = 0;                       // размерность системы (число переменных)

    // Начальные условия и базовые значения параметров системы
    std::vector<double> initial_conditions;  // длина == amountOfX
    std::vector<double> base_values;         // все параметры системы

    // Изменяемый параметр (sweep target).
    // По умолчанию свип идёт по параметру (par_or_var = 1):
    //   param_index — 1-based индекс в base_values (a[0] зарезервирован).
    // При sweep_over_var = true (par_or_var = 0) свип идёт по начальному
    // условию: var_sweep_index — 0-based индекс в initial_conditions.
    int  param_index    = 0;
    bool sweep_over_var = false;
    int  var_sweep_index = 0;
    // Continuation: каждая новая точка параметра стартует с конечного x[]
    // предыдущей (вместо сброса на initial_conditions). Игнорируется при
    // sweep_over_var = true (валидатор отказывает).
    //   continuation_reverse = false → forward (lo→hi)
    //   continuation_reverse = true  → backward (hi→lo) — для гистерезиса.
    bool continuation = false;
    bool continuation_reverse = false;
    double param_lo = 0.0;
    double param_hi = 1.0;
    int n_pts = 1000;                        // разрешение по параметру

    // Запись результата
    int writable_var = 0;                    // индекс X-переменной, которую пишем

    // Интегрирование
    double h = 0.01;                         // шаг
    double transient_time = 0.0;             // burn-in (отбрасывается)
    double t_max = 0.0;                      // время записи после transient
    int pre_scaller = 1;                     // писать каждую pre_scaller-тую точку

    // Защита от расхождения
    double max_value = 1.0e6;

    // Если не пусто — engine запишет CSV с результатами по тому же формату,
    // что и NonLinAnal::bifurcation1D (для publication-quality пост-процессинга).
    // Пустая строка = ничего не пишем (только в памяти).
    std::string csv_output_path;

    // Cooperative cancellation token. Shared between caller (session) and
    // worker thread; engine checks it between kernel launches and aborts
    // with cancelled=true if set. nullptr = no cancellation.
    std::shared_ptr<std::atomic<bool>> cancel;

    // Progress reporting (0.0 .. 1.0). Engine writes after each chunk;
    // GUI reads each frame to drive a progress bar. nullptr = no reporting.
    std::shared_ptr<std::atomic<float>> progress;
};

struct Bifurcation1DResult {
    bool ok = false;
    bool cancelled = false;
    std::string error;

    // Снапшот режима continuation на момент Run — GUI использует флаг для
    // корректного построения X точек (reverse → x = hi - (hi-lo)*i/(n-1)).
    bool continuation_reverse = false;
    // Снапшот диапазона — чтобы GUI считал X для continuation одинаково,
    // не завися от правок текстовых полей после Run.
    double param_lo = 0.0;
    double param_hi = 1.0;
    int n_pts = 0;
    int record_steps = 0;

    // bifurcation_points[i] — значения writable_var для i-го значения параметра.
    // Длина каждого внутреннего вектора == record_steps.
    std::vector<std::vector<double>> bifurcation_points;

    // peak_times[i] — межпиковые интервалы (timeOfPeaks в NonLinAnal).
    // peak_times[i][j] = время от j-го пика до (j+1)-го для параметра i.
    // Та же длина, что и bifurcation_points[i].
    std::vector<std::vector<double>> peak_times;

    // flags[i]: 1 — расчёт прошёл, -1 — траектория разошлась за max_value.
    std::vector<int> flags;

    // Snapshot of the request fields needed to reproduce the _config.csv
    // header on right-click export from the GUI. Filled by engine in
    // run_bif1d() before any CSV write; consumed by data_export::export_bif1d.
    data_export::Bif1DSnapshot snapshot;
};

// ============================================================================
// LLE (Largest Lyapunov Exponent) 1D — свип параметра, λ(param).
// Алгоритм: Wolf/Benettin с малым возмущением. Реализован в NonLinAnal
// (cudaLibrary.cu:LLEKernelCUDA), engine подключает его через NVRTC и
// шаблон kernels/lle1d.template.cu.
// ============================================================================

struct LLE1DRequest {
    std::string krs_body;
    int amountOfX = 0;

    std::vector<double> initial_conditions;  // длина == amountOfX
    std::vector<double> base_values;         // все параметры системы (с a[0]=0)

    // Sweep target. По умолчанию param-свип (par_or_var=1):
    //   param_index — 1-based индекс в base_values.
    // При sweep_over_var=true (par_or_var=0) свип идёт по начальному условию:
    //   var_sweep_index — 0-based индекс в initial_conditions.
    bool sweep_over_var = false;
    int  var_sweep_index = 0;
    int param_index = 0;                     // индекс в base_values (1-based)
    double param_lo = 0.0;
    double param_hi = 1.0;
    int n_pts = 500;

    // Интегрирование (как в bif1d).
    double h = 0.01;
    double transient_time = 0.0;
    double t_max = 100.0;

    // LLE-специфика: длина одного блока интегрирования между ренормализациями
    // в единицах времени (NT), и размер возмущения (eps).
    double NT  = 1.0;
    double eps = 1.0e-4;

    // Защита от расхождения.
    double max_value = 1.0e6;

    // Опциональный CSV.
    std::string csv_output_path;

    // See Bifurcation1DRequest::cancel.
    std::shared_ptr<std::atomic<bool>> cancel;

    // See Bifurcation1DRequest::progress.
    std::shared_ptr<std::atomic<float>> progress;
};

struct LLE1DResult {
    bool ok = false;
    bool cancelled = false;
    std::string error;

    int n_pts = 0;
    // Снапшот свип-диапазона на момент Run — чтобы GUI рисовал точки именно по
    // тем X, для которых считалось. Иначе при правке param_lo/hi в панели
    // график «прыгает» до следующего Run.
    double param_lo = 0.0;
    double param_hi = 1.0;
    // lyapunov[i] — оценка λ для i-го значения параметра. Спец-значения:
    //   +999  — траектория x(t) сошла с аттрактора на transient'е (kernel-флаг),
    //   -999  — траектория разошлась за maxValue (kernel-флаг),
    //   nan/inf — численная проблема (потом фильтруется на стороне GUI).
    std::vector<double> lyapunov;
    // flags[i]: 1 (ok) / -1 (diverged) — для совместимости с UI-агрегацией.
    std::vector<int>    flags;

    // Snapshot for right-click GUI export — see Bifurcation1DResult::snapshot.
    data_export::LLE1DSnapshot snapshot;
};

// ============================================================================
// LS (Lyapunov Spectrum) 1D — на каждую точку параметра N экспонент (по числу
// переменных). Алгоритм: Wolf/Benettin + Gram-Schmidt. Реализован в NonLinAnal
// (cudaLibrary.cu:LSKernelCUDA). Поля Request — копия LLE1DRequest (тот же
// набор управляющих параметров, eps и NT). Result — матрица nPts × amountOfX.
// ============================================================================

struct LS1DRequest {
    std::string krs_body;
    int amountOfX = 0;
    std::vector<double> initial_conditions;
    std::vector<double> base_values;
    // См. LLE1DRequest — те же поля sweep target'а.
    bool sweep_over_var = false;
    int  var_sweep_index = 0;
    int param_index = 0;
    double param_lo = 0.0;
    double param_hi = 1.0;
    int    n_pts = 500;
    double h = 0.01;
    double transient_time = 0.0;
    double t_max = 100.0;
    double NT  = 1.0;
    double eps = 1.0e-4;
    double max_value = 1.0e6;
    std::string csv_output_path;

    // See Bifurcation1DRequest::cancel.
    std::shared_ptr<std::atomic<bool>> cancel;

    // See Bifurcation1DRequest::progress.
    std::shared_ptr<std::atomic<float>> progress;
};

struct LS1DResult {
    bool ok = false;
    bool cancelled = false;
    std::string error;

    int n_pts = 0;
    int n_exponents = 0;   // = amountOfX
    // Снапшот диапазона свипа на момент Run (см. LLE1DResult).
    double param_lo = 0.0;
    double param_hi = 1.0;

    // spectrum[i][k] — k-я экспонента для i-й точки параметра. Длина внутреннего
    // вектора == n_exponents. Спец-значения 999 / -999 — kernel-флаги ошибки.
    std::vector<std::vector<double>> spectrum;
    // flags[i]: 1 (ok) / -1 (diverged) — все экспоненты разом.
    std::vector<int> flags;

    // Snapshot for right-click GUI export — see Bifurcation1DResult::snapshot.
    data_export::LS1DSnapshot snapshot;
};

// ============================================================================
// LLE 2D — λ(p1, p2) на квадратной сетке n_pts × n_pts. Алгоритм тот же
// (LLEKernelCUDA, см. cudaLibrary.cu:2380) — kernel принимает runtime-аргумент
// dimension=2, ranges[4] и indicesOfMutVars[2]. par_or_var (compile-time)
// принимает три значения:
//   1 — оба свипа по параметрам;
//   0 — оба свипа по начальным условиям;
//   2 — смешанный: ось 1 (X в kernel'е) по IC, ось 2 (Y в kernel'е) по param.
// Сетка квадратная — таково ограничение getValueByIdx (cu:1276): idx∈[0,n²),
// pointIdx_x = idx%n, pointIdx_y = idx/n. Разная разрешалка по осям без
// правок NonLinAnal невозможна — поэтому здесь один n_pts на обе оси.
//
// Engine принимает любую комбинацию sweep_over_var/_2 без отдельного флага
// "mixed". Маппинг (user X/Y → kernel-ось 1/2):
//   - sweep_over_var == sweep_over_var_2 → par_or_var=0|1, оси один в один;
//   - sweep_over_var=true, sweep_over_var_2=false → par_or_var=2, оси один в один;
//   - sweep_over_var=false, sweep_over_var_2=true → par_or_var=2 + внутренний
//     своп: indicesOfMutVars/ranges подаются в kernel "перевёрнуто" (X↔Y), а
//     полученный flat-массив транспонируется на хосте, так что наружу result
//     по-прежнему row-major idx=iy*n+ix в системе пользователя.
// ============================================================================

struct LLE2DRequest {
    std::string krs_body;
    int amountOfX = 0;
    std::vector<double> initial_conditions;
    std::vector<double> base_values;

    // Sweep target по обеим осям. По умолчанию оба по параметру (par_or_var=1).
    bool sweep_over_var  = false;  // ось X
    bool sweep_over_var_2 = false; // ось Y
    // Индексы целей свипа. При sweep_over_var_*=true берётся var_sweep_index_*
    // (0-based в initial_conditions); иначе param_index_* (1-based в base_values).
    int  param_index      = 0;
    int  var_sweep_index  = 0;
    int  param_index_2    = 0;
    int  var_sweep_index_2 = 0;

    double param_lo   = 0.0;
    double param_hi   = 1.0;
    double param_lo_2 = 0.0;
    double param_hi_2 = 1.0;
    int    n_pts      = 200;     // одно значение — сетка квадратная

    // Интегрирование (как в LLE1D).
    double h              = 0.01;
    double transient_time = 0.0;
    double t_max          = 100.0;
    double NT             = 1.0;
    double eps            = 1.0e-4;
    double max_value      = 1.0e6;

    std::string csv_output_path;

    // See Bifurcation1DRequest::cancel.
    std::shared_ptr<std::atomic<bool>> cancel;

    // See Bifurcation1DRequest::progress.
    std::shared_ptr<std::atomic<float>> progress;
};

struct LLE2DResult {
    bool ok = false;
    bool cancelled = false;
    std::string error;

    int n_pts = 0;                 // сторона сетки (всего n_pts² ячеек)
    // Снапшот диапазонов на момент Run (как у LLE1DResult).
    double param_lo   = 0.0;
    double param_hi   = 1.0;
    double param_lo_2 = 0.0;
    double param_hi_2 = 1.0;

    // values[iy*n_pts + ix] = λ для (p1(ix), p2(iy)). Размер n_pts² целиком,
    // включая ячейки, где kernel вернул спец-значение (999 — нет аттрактора,
    // -999 — разошлось, NaN/inf — численная проблема). Render отфильтрует.
    std::vector<double> values;
    std::vector<int>    flags;     // 1=ok, -1=diverged

    // Авто-нормализация для colormap'а: min/max по валидным значениям
    // (без 999/-999/nan/inf). Если валидных нет — обе 0.
    double min_val = 0.0;
    double max_val = 0.0;

    // Snapshot for right-click GUI export — see Bifurcation1DResult::snapshot.
    data_export::LLE2DSnapshot snapshot;
};

// ============================================================================
// Bifurcation 2D — «период»(p1, p2) на квадратной сетке n_pts × n_pts.
// Алгоритм — порт bifurcation2D из hostLibrary.cu: три ядра на каждый чанк:
//   calculateDiscreteModelCUDA (dimension=2, par_or_var compile-time)
//   → peakFinderCUDA   → пики амплитуд + межпиковые интервалы
//   → dbscanCUDA       → число кластеров = период системы в ячейке
//
// par_or_var (compile-time) — три значения (cudaLibrary.cu:973-990):
//   1 — обе оси по параметрам;
//   0 — обе оси по начальным условиям;
//   2 — смешанный: ось 1 (X kernel) по IC, ось 2 (Y kernel) по param.
// Маппинг sweep_over_var/_2 → par_or_var и логика своп/транспонирования — та
// же что у LLE-2D (см. комментарий к LLE2DRequest выше).
// ============================================================================

struct Bifurcation2DRequest {
    std::string krs_body;
    int amountOfX = 0;
    std::vector<double> initial_conditions;
    std::vector<double> base_values;

    // Ось X (первая).
    bool sweep_over_var  = false;
    int  var_sweep_index = 0;
    int  param_index     = 0;       // 1-based в base_values

    // Ось Y (вторая).
    bool sweep_over_var_2  = false;
    int  var_sweep_index_2 = 0;
    int  param_index_2     = 0;

    double param_lo   = 0.0;
    double param_hi   = 1.0;
    double param_lo_2 = 0.0;
    double param_hi_2 = 1.0;
    int    n_pts      = 200;        // квадратная сетка n_pts²

    int    writable_var   = 0;      // индекс переменной для peak-finding
    double h              = 0.01;
    double transient_time = 0.0;
    double t_max          = 100.0;
    int    pre_scaller    = 1;
    double max_value      = 1.0e6;

    // DBSCAN-порог: радиус эпсилон для кластеризации пиков.
    // Тот же смысл, что eps в bifurcation2D NonLinAnal (hostLibrary.cu:912).
    double eps_dbscan = 0.1;

    std::string csv_output_path;

    // See Bifurcation1DRequest::cancel.
    std::shared_ptr<std::atomic<bool>> cancel;

    // See Bifurcation1DRequest::progress.
    std::shared_ptr<std::atomic<float>> progress;
};

struct Bifurcation2DResult {
    bool ok = false;
    bool cancelled = false;
    std::string error;

    int n_pts = 0;                   // сторона сетки (всего n_pts² ячеек)
    double param_lo   = 0.0;
    double param_hi   = 1.0;
    double param_lo_2 = 0.0;
    double param_hi_2 = 1.0;

    // values[iy*n_pts + ix] = (double)dbscan_result — период (число кластеров пиков).
    // Спец-значения: -1.0 = расхождение (flag=-1), 0.0 = нет пиков (фикс. точка).
    std::vector<double> values;
    std::vector<int>    flags;       // 1=ok, -1=diverged

    // Авто-нормализация для colormap (без -1/nan).
    double min_val = 0.0;
    double max_val = 0.0;

    // Snapshot for right-click GUI export — see Bifurcation1DResult::snapshot.
    data_export::Bif2DSnapshot snapshot;
};

// ============================================================================
// LS 2D — полный спектр N экспонент в каждой ячейке сетки n_pts × n_pts.
// Алгоритм тот же (LSKernelCUDA, cudaLibrary.cu:2732), kernel принимает
// runtime-аргумент dimension=2, ranges[4] и indicesOfMutVars[2]. par_or_var
// (compile-time) три значения — как у LLE-2D (см. комментарий к LLE2DRequest).
// Отличие от LLE-2D: на ячейку kernel возвращает N экспонент, не одну. D2H
// копирует cur_limiter * N значений и распаковывает по плоскостям-экспонентам.
// ============================================================================

struct LS2DRequest {
    std::string krs_body;
    int amountOfX = 0;
    std::vector<double> initial_conditions;
    std::vector<double> base_values;

    bool sweep_over_var  = false;  // ось X
    bool sweep_over_var_2 = false; // ось Y
    int  param_index      = 0;
    int  var_sweep_index  = 0;
    int  param_index_2    = 0;
    int  var_sweep_index_2 = 0;

    double param_lo   = 0.0;
    double param_hi   = 1.0;
    double param_lo_2 = 0.0;
    double param_hi_2 = 1.0;
    int    n_pts      = 200;

    double h              = 0.01;
    double transient_time = 0.0;
    double t_max          = 100.0;
    double NT             = 1.0;
    double eps            = 1.0e-4;
    double max_value      = 1.0e6;

    std::string csv_output_path;

    // See Bifurcation1DRequest::cancel.
    std::shared_ptr<std::atomic<bool>> cancel;

    // See Bifurcation1DRequest::progress.
    std::shared_ptr<std::atomic<float>> progress;
};

struct LS2DResult {
    bool ok = false;
    bool cancelled = false;
    std::string error;

    int n_pts       = 0;            // сторона сетки (всего n_pts² ячеек)
    int n_exponents = 0;            // == amountOfX
    double param_lo   = 0.0;
    double param_hi   = 1.0;
    double param_lo_2 = 0.0;
    double param_hi_2 = 1.0;

    // Layout: values[k * n_pts * n_pts + iy * n_pts + ix] = k-я экспонента в
    // ячейке (ix, iy). Группировка по экспонентам (не по ячейкам) — даёт
    // contiguous-плоскость на отрисовку HeatmapView: &values[k*n*n] передаётся
    // без копирования. Спец-значения 999/-999/NaN — как в LS1DResult.
    std::vector<double> values;
    // flags[iy*n + ix] — общий per-cell (1=ok, -1=diverged для всех экспонент).
    std::vector<int>    flags;

    // Авто-нормализация per-plane (по валидным значениям, без 999/-999/nan).
    // Размер == n_exponents; если валидных нет — обе 0.
    std::vector<double> min_val;
    std::vector<double> max_val;

    // Snapshot for right-click GUI export — see Bifurcation1DResult::snapshot.
    data_export::LS2DSnapshot snapshot;
};

// ============================================================================
// Basins of Attraction — карта классификации траекторий на сетке n_pts × n_pts
// начальных условий (axis_x_var, axis_y_var). На каждую ячейку считается:
//   - flag классификации (-1=FP / 0=Unbound / 1=Osc) через calculateDiscreteModelCUDA
//   - avgPeak и avgInterval через avgPeakFinderCUDA (с множителями mult_avg_*)
//   - cluster id через CUDA_dbscan (host-цикл из 3 kernel'ов)
// Свип ВСЕГДА по двум IC — par_or_var=0 в шаблоне жёстко.
// ============================================================================

struct BasinsRequest {
    std::string krs_body;
    int amountOfX = 0;
    std::vector<double> initial_conditions;   // default IC для не-axis переменных
    std::vector<double> base_values;

    int axis_x_var = 0;       // 0-based индекс переменной по X
    int axis_y_var = 1;       // 0-based индекс по Y
    double axis_x_lo = -10.0, axis_x_hi = 10.0;
    double axis_y_lo = -10.0, axis_y_hi = 10.0;
    int    n_pts = 200;

    int    writable_var   = 0;
    double h              = 0.01;
    double t_max          = 1000.0;
    double transient_time = 10000.0;
    int    pre_scaller    = 1;
    double max_value      = 1.0e6;
    double eps_dbscan     = 0.5;

    // avgPeakFinderCUDA feature dispatch: какие фичи писать в outAvgPeaks/
    // AvgTimeOfPeaks (см. BF_* в configCUDA.h) + множители ПОСЛЕ фич для
    // подстройки масштаба DBSCAN. Дефолты дают pre-feature-selection поведение.
    int  feature1 = BF_FEATURE1_DEFAULT;
    int  feature2 = BF_FEATURE2_DEFAULT;
    numb mult1    = mult_avg_peak;
    numb mult2    = mult_avg_interval;

    std::string csv_output_path;

    // See Bifurcation1DRequest::cancel.
    std::shared_ptr<std::atomic<bool>> cancel;

    // See Bifurcation1DRequest::progress. For basins the engine RESETS
    // `progress` to 0 between phases and bumps `progress_phase` (1 = sim,
    // 2 = cluster), so the GUI shows a per-phase 0..1 bar with a phase label.
    std::shared_ptr<std::atomic<float>> progress;
    std::shared_ptr<std::atomic<int>>   progress_phase;
};

struct BasinsResult {
    bool ok = false;
    bool cancelled = false;
    std::string error;

    int n_pts = 0;
    double axis_x_lo = 0.0, axis_x_hi = 0.0;
    double axis_y_lo = 0.0, axis_y_hi = 0.0;
    int axis_x_var = 0, axis_y_var = 1;

    // Все four-поля размер n_pts² row-major (iy*n + ix):
    std::vector<int>    basin_idx;       // ≥1 = Osc cluster; ≤-1 = FP cluster; 0 = Unbound
    std::vector<double> avg_peaks;       // 999 = no peaks / unbound, NaN→999
    std::vector<double> avg_intervals;
    std::vector<int>    helpful_array;   // -1=FP, 0=Unbound, 1=Osc

    // Сводки для отрисовки.
    int    n_clusters = 0;               // max(basin_idx); min подсчитывается отдельно
    int    min_cluster_idx = 0;          // самый отрицательный (для FP cluster id)
    double avg_peaks_min     = 0.0, avg_peaks_max     = 0.0;
    double avg_intervals_min = 0.0, avg_intervals_max = 0.0;

    // Snapshot for right-click GUI export — see Bifurcation1DResult::snapshot.
    data_export::BasinsSnapshot snapshot;
};

// ============================================================================
// Basins recluster — DBSCAN-only прогон поверх кэшированных фич (avg_peaks /
// avg_intervals / helpful_array из предыдущего run_basins). Используется
// кнопкой "Clustering" в GUI: меняешь eps_dbscan, не пересчитывая всю
// траекторно-feature часть (≈99% времени основного run_basins).
// krs_body + amountOfX нужны, чтобы compile_basins_if_needed подхватил
// кэш-ключ модуля; сами DBSCAN-kernel'ы не зависят от системы уравнений.
// ============================================================================
struct BasinsReclusterRequest {
    std::string krs_body;
    int amountOfX = 0;
    int n_pts = 0;                        // total_cells = n_pts * n_pts
    double eps_dbscan = 0.5;
    std::vector<double> avg_peaks;        // size = n_pts²; mults уже применены
    std::vector<double> avg_intervals;    // size = n_pts²
    std::vector<int>    helpful_array;    // size = n_pts²; -1=FP, 0=Unbound, 1=Osc
    std::shared_ptr<std::atomic<bool>>  cancel;
    std::shared_ptr<std::atomic<float>> progress;
};

struct BasinsReclusterResult {
    bool ok = false;
    bool cancelled = false;
    std::string error;
    int n_pts = 0;
    std::vector<int> basin_idx;           // size = n_pts²
    int n_clusters = 0;
    int min_cluster_idx = 0;
};

// ============================================================================
// Fast Synchro — анализ возвратной синхронизации.
// Два режима:
//   mode 0 ("On Attractor"): интегрируем master trajectory, в каждой её точке
//     запускаем cycle synchro → ошибка per-point. Результат — массив точек
//     (traj_x, traj_y) + sync_error длиной nPts (после decimator'а).
//   mode 1 ("On Grid"): свип по двум IC (n_pts × n_pts), per-cell sync error.
//     Результат — 2D heatmap.
// type_of_synch / error_estim / fs_error_trs — knobs из configCUDA.h,
// override'имые через NVRTC #define (поэтому PTX-кэш ключ включает их).
// ============================================================================

struct FastSyncRequest {
    std::string krs_body;
    int amountOfX = 0;
    std::vector<double> values;       // параметры системы (amountOfValues = values.size())

    int mode = 0;                     // 0 = On Attractor, 1 = On Grid

    // Общие параметры алгоритма.
    double h               = 0.01;
    int    iter_of_synchr  = 100;
    int    pre_scaller     = 1;
    double max_value       = 1e6;
    std::vector<double> k_forward;    // length = amountOfX
    std::vector<double> k_backward;   // length = amountOfX
    std::vector<double> ic_master;    // length = amountOfX
    std::vector<double> ic_slave;     // length = amountOfX

    // mode == 0 (On Attractor):
    double t_max          = 100.0;
    double transient_time = 0.0;
    numb   window         = (numb)50.0;  // окно синхронизации (numb для ABI-consistency с kernel-аргументами)

    // mode == 1 (On Grid):
    int    axis_x_var = 0;
    int    axis_y_var = 1;
    double axis_x_lo = -10.0, axis_x_hi = 10.0;
    double axis_y_lo = -10.0, axis_y_hi = 10.0;
    int    n_pts = 200;
    // false (default) — grid перебирает НУ мастера, НУ слейва фикс.
    // true            — grid перебирает НУ слейва, НУ мастера фикс.
    bool   grid_swap_master_slave = false;

    // Runtime knobs (substituted via NVRTC #define перед include configCUDA.h).
    int    type_of_synch = 0;
    int    error_estim   = 2;
    double fs_error_trs  = 1e-12;

    // CSV вывод — пустая строка отключает запись.
    std::string csv_output_path;
    // var-names для шапки CSV (mode 0 attractor). Совпадает по длине с amountOfX.
    std::vector<std::string> var_names;

    // Cancel + progress — мирорят Bifurcation*Request.
    std::shared_ptr<std::atomic<bool>>  cancel;
    std::shared_ptr<std::atomic<float>> progress;
};

struct FastSyncResult {
    bool ok = false;
    bool cancelled = false;
    std::string error;
    int  mode = 0;

    // mode == 0: trajectory + per-point sync error.
    // traj_full — row-major n_pts_traj × amountOfX_traj (полный вектор X в
    // каждой decimated-точке). Позволяет GUI пересобирать (X, Y) проекции при
    // смене axis_*_var БЕЗ повторного запуска расчёта.
    int n_pts_traj = 0;
    int amountOfX_traj = 0;
    std::vector<double> traj_full;
    std::vector<double> sync_error;        // length = n_pts_traj

    // mode == 1: heatmap n_pts × n_pts row-major (iy*n + ix).
    int n_pts_grid = 0;
    double axis_x_lo = 0.0, axis_x_hi = 0.0;
    double axis_y_lo = 0.0, axis_y_hi = 0.0;
    int axis_x_var = 0, axis_y_var = 1;
    std::vector<double> heatmap;

    // Min/max валидных значений sync_error/heatmap для autoscale colorbar.
    double min_val = 0.0;
    double max_val = 0.0;

    // Snapshot for right-click GUI export — see Bifurcation1DResult::snapshot.
    data_export::FastSyncSnapshot snapshot;
};

class ParametricEngine {
public:
    ParametricEngine();
    ~ParametricEngine();
    ParametricEngine(const ParametricEngine&) = delete;
    ParametricEngine& operator=(const ParametricEngine&) = delete;

    // 1D-бифуркация. Если КРС не изменился с прошлого запуска — переиспользуется
    // скомпилированный PTX (кэш по хэшу krs_body + amountOfX).
    Bifurcation1DResult run_bifurcation_1d(const Bifurcation1DRequest& req);

    // 2D-бифуркация — период(p1, p2) через DBSCAN на квадратной сетке.
    Bifurcation2DResult run_bifurcation_2d(const Bifurcation2DRequest& req);

    // 1D-LLE. Отдельный PTX-кэш (другое шаблоное тело и другой kernel).
    LLE1DResult run_lle_1d(const LLE1DRequest& req);

    // 1D-LS — спектр Ляпунова (N экспонент на точку параметра).
    LS1DResult run_ls_1d(const LS1DRequest& req);

    // 2D-LLE — λ(p1, p2) на квадратной сетке. Отдельный PTX-кэш (другая
    // ветка par_or_var, тот же kernel LLEKernelCUDA).
    LLE2DResult run_lle_2d(const LLE2DRequest& req);

    // 2D-LS — полный спектр N экспонент в каждой ячейке n_pts × n_pts.
    LS2DResult run_ls_2d(const LS2DRequest& req);

    // Basins of attraction — карта на сетке IC. 4 output-поля + cluster ids.
    BasinsResult run_basins(const BasinsRequest& req);

    // DBSCAN-only прогон по уже посчитанным фичам (для перекластеризации при
    // смене eps_dbscan без пересчёта траекторий).
    BasinsReclusterResult run_basins_recluster(const BasinsReclusterRequest& req);

    // Fast Synchro — режим выбирается через req.mode (0 = traj, 1 = grid).
    FastSyncResult run_fastsync(const FastSyncRequest& req);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
