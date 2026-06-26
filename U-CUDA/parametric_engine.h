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

#include <memory>
#include <string>
#include <vector>

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
};

struct Bifurcation1DResult {
    bool ok = false;
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
};

struct LLE1DResult {
    bool ok = false;
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
};

struct LS1DResult {
    bool ok = false;
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
};

struct LLE2DResult {
    bool ok = false;
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
};

struct Bifurcation2DResult {
    bool ok = false;
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

    // TODO (следующие шаги): run_ls_2d, ...

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
