#pragma once
#include <string>
#include <vector>
namespace data_export {
// Snapshot structs — exactly the data engine currently prints into _config.csv.
// Names follow the engine's local variable names (tMax / TT / etc.) so the
// 1-to-1 mapping in writers is obvious.

// Провенанс расчёта: режим FMA-контракции NVRTC, действовавший на момент Run (Settings -> GPU
// floating point, см. set_nvrtc_fmad). Пишется в _config.csv каждым writer'ом строкой
// "NVRTC --fmad = on/off": результаты, посчитанные с разным значением, не сравнимы побитово, а на
// фрактальных границах бассейнов расходятся и визуально — без пометки в файле потом не установить,
// чем считалось. Снимается на старте прогона, а не при экспорте: настройку могли переключить между
// расчётом и сохранением.
// Это настройка приложения, а не свойство устройства: две ветки считают на CPU и через NVRTC не
// проходят вовсе (run_bif1d_cpu — continuation, и фазовый портрет при снятом GPU). Там строка
// остаётся справочной — она говорит, как был настроен компилятор ядер, а не как считался файл.
struct Bif1DSnapshot {
    bool   gpu_fmad = true;
    std::vector<double> values;              // a[1..N] (engine prints a[N])
    std::vector<double> initial_conditions;  // X0[]
    double tMax = 0.0;
    double transientTime = 0.0;
    double h = 0.0;
    int    preScaller = 0;
    int    writableVar = 0;
    int    indexOfMutVar = 0;                // 1-based for param, 0-based for IC
    double range_lo = 0.0;
    double range_hi = 0.0;
    // Раскладка точек по оси — как у LLE1DSnapshot: export_bif1d считает x той же функцией,
    // что график (лог-шкала, порядок цепочки continuation), а не линейно lo..hi.
    bool   log_scale = false;
    bool   continuation = false;
    bool   continuation_reverse = false;
    bool   on_cpu = false;          // строка "device = CPU" вместо NVRTC --fmad
    std::string step_control;       // адаптивный шаг: регулятор и допуски, пусто — Fixed
};

// 1D DFT — same sweep/integration fields as Bif1DSnapshot, plus the frequency
// axis (n_freq/freq_lo/freq_hi). AkCOS/BkSIN files share this one snapshot.
struct Dft1DSnapshot {
    bool   gpu_fmad = true;   // см. Bif1DSnapshot::gpu_fmad
    std::vector<double> values;
    std::vector<double> initial_conditions;
    double tMax = 0.0;
    double transientTime = 0.0;
    double h = 0.0;
    int    preScaller = 0;
    int    writableVar = 0;
    int    indexOfMutVar = 0;
    double range_lo = 0.0;
    double range_hi = 0.0;
    int    n_freq = 0;
    double freq_lo = 0.0;
    double freq_hi = 0.0;
    int    window_type = 1;   // 0=None, 1=Hanning, 2=Hamming, 3=Blackman,
                              // 4=Blackman-Harris — см. Dft1DRequest::window_type
};

struct LLE1DSnapshot {
    bool   gpu_fmad = true;   // см. Bif1DSnapshot::gpu_fmad
    std::vector<double> values;
    std::vector<double> initial_conditions;
    double tMax = 0.0;
    double NT   = 0.0;
    double vectorTransient = 0.0;   // транзиент касательных векторов, 0 — нет
    double transientTime = 0.0;
    double h    = 0.0;
    double eps  = 0.0;
    int    indexOfMutVar = 0;
    double range_lo = 0.0;
    double range_hi = 0.0;
    // Как стоят точки по оси: export_* считает x той же функцией, что график
    // (лог-шкала, порядок цепочки continuation), а не линейно lo..hi.
    bool   log_scale = false;
    bool   continuation = false;
    bool   continuation_reverse = false;
    bool   on_cpu = false;          // строка "device = CPU" вместо NVRTC --fmad
    std::string step_control;       // адаптивный шаг: регулятор и допуски, пусто — Fixed
};

// LS1D shares the same header layout as LLE1D (engine writes "1D LS" instead
// of "1D LLE") — same fields. Kept as a distinct type for clarity at call site.
struct LS1DSnapshot {
    bool   gpu_fmad = true;   // см. Bif1DSnapshot::gpu_fmad
    std::vector<double> values;
    std::vector<double> initial_conditions;
    double tMax = 0.0;
    double NT   = 0.0;
    double vectorTransient = 0.0;   // транзиент касательных векторов, 0 — нет
    double transientTime = 0.0;
    double h    = 0.0;
    double eps  = 0.0;
    int    indexOfMutVar = 0;
    double range_lo = 0.0;
    double range_hi = 0.0;
    // Как стоят точки по оси: export_* считает x той же функцией, что график
    // (лог-шкала, порядок цепочки continuation), а не линейно lo..hi.
    bool   log_scale = false;
    bool   continuation = false;
    bool   continuation_reverse = false;
    bool   on_cpu = false;          // строка "device = CPU" вместо NVRTC --fmad
    std::string step_control;       // адаптивный шаг: регулятор и допуски, пусто — Fixed
};

// 2D: covers Bif2D (uses eps_dbscan + writableVar + preScaller fields) and
// LLE2D/LS2D (uses NT + eps fields). Each type has its own writer that pulls
// only the fields relevant to its config block; absent fields stay at 0.
struct Bif2DSnapshot {
    bool   gpu_fmad = true;   // см. Bif1DSnapshot::gpu_fmad
    std::vector<double> values;
    std::vector<double> initial_conditions;
    int    par_or_var = 1;                   // 1=param/param, 0=IC/IC, 2=mixed
    double tMax = 0.0;
    double transientTime = 0.0;
    double h    = 0.0;
    int    preScaller = 0;
    double eps_dbscan = 0.0;
    // Множители осей DBSCAN (пик / интервал) — см. Bifurcation2DRequest.
    // Легаси-путь hostLibrary.cu печатает их в CSV, здесь то же самое.
    double mult_peak     = 1.0;
    double mult_interval = 0.0;
    int    writableVar = 0;
    int    indexOfMutVar  = 0;
    int    indexOfMutVar2 = 0;
    double range1_lo = 0.0, range1_hi = 0.0;
    double range2_lo = 0.0, range2_hi = 0.0;
    int    n_pts = 0;
};

struct LLE2DSnapshot {
    bool   gpu_fmad = true;   // см. Bif1DSnapshot::gpu_fmad
    std::vector<double> values;
    std::vector<double> initial_conditions;
    int    par_or_var = 1;
    double tMax = 0.0;
    double NT   = 0.0;
    double vectorTransient = 0.0;   // транзиент касательных векторов, 0 — нет
    double transientTime = 0.0;
    double h    = 0.0;
    double eps  = 0.0;
    int    indexOfMutVar  = 0;
    int    indexOfMutVar2 = 0;
    double range1_lo = 0.0, range1_hi = 0.0;
    double range2_lo = 0.0, range2_hi = 0.0;
    int    n_pts = 0;
};

struct LS2DSnapshot {
    bool   gpu_fmad = true;   // см. Bif1DSnapshot::gpu_fmad
    std::vector<double> values;
    std::vector<double> initial_conditions;
    int    par_or_var = 1;
    double tMax = 0.0;
    double NT   = 0.0;
    double vectorTransient = 0.0;   // транзиент касательных векторов, 0 — нет
    double transientTime = 0.0;
    double h    = 0.0;
    double eps  = 0.0;
    int    indexOfMutVar  = 0;
    int    indexOfMutVar2 = 0;
    double range1_lo = 0.0, range1_hi = 0.0;
    double range2_lo = 0.0, range2_hi = 0.0;
    int    n_pts = 0;
    int    n_exponents = 0;
};

struct BasinsSnapshot {
    bool   gpu_fmad = true;   // см. Bif1DSnapshot::gpu_fmad
    std::vector<double> values;
    std::vector<double> initial_conditions;
    double tMax = 0.0;
    double transientTime = 0.0;
    double h    = 0.0;
    int    preScaller = 0;
    double eps_dbscan = 0.0;
    int    writableVar = 0;
    int    axis_x_var = 0, axis_y_var = 0;
    double axis_x_lo = 0.0, axis_x_hi = 0.0;
    double axis_y_lo = 0.0, axis_y_hi = 0.0;
    int    n_pts = 0;
    int    feature1 = 0;
    int    feature2 = 0;
    double mult1 = 0.0;
    double mult2 = 0.0;
};

struct PhaseSnapshot {
    bool   gpu_fmad = true;   // см. Bif1DSnapshot::gpu_fmad
    std::vector<std::string> vars;       // variable names (column headers)
    std::vector<std::string> params;     // param names (informational)
    std::vector<double> a;               // a[0..nparams] (a[0] = symmetry)
    std::vector<std::vector<double>> ic_flat;  // [ic][var] starting state
    std::vector<std::string> ic_labels;
    std::string scheme;
    // Discrete map: h = 1, so the CSV index column is the iteration number n
    // and t_max / t_skip are counted in iterations.
    bool   is_map   = false;
    double h        = 0.0;
    double t_max    = 0.0;
    double t_skip   = 0.0;
    int    decimator = 1;
    // Адаптивный шаг: одна строка настроек для _config.csv; пусто — шаг постоянный.
    std::string adaptive;
};

struct FastSyncSnapshot {
    bool   gpu_fmad = true;   // см. Bif1DSnapshot::gpu_fmad
    int    mode = 0;                     // 0 = On Attractor, 1 = On Grid
    std::vector<double> values;          // system parameters
    std::vector<double> ic_master;
    std::vector<double> ic_slave;
    std::vector<double> k_forward;
    std::vector<double> k_backward;
    std::vector<std::string> var_names;  // column headers for mode 0 CSV
    double h = 0.0;
    int    iter_of_synchr = 0;
    int    preScaller = 0;
    double window = 0.0;
    int    type_of_synch = 0;
    int    error_estim = 0;
    double fs_error_trs = 0.0;
    // Mode 0:
    double tMax = 0.0;
    double transientTime = 0.0;
    // Mode 1:
    // transientTime здесь — TT мастера, transientTimeSlave — TT слейва
    // (в режиме On Grid у каждой системы транзиент свой).
    double transientTimeSlave = 0.0;
    int    axis_x_var = 0, axis_y_var = 0;
    double axis_x_lo = 0.0, axis_x_hi = 0.0;
    double axis_y_lo = 0.0, axis_y_hi = 0.0;
    int    n_pts = 0;
    bool   grid_swap_master_slave = false;
    // Оба режима: НУ второй системы — фиксированные (0) либо случайный отступ
    // в eps-окрестности точки первой (1). См. FastSyncRequest::ic_random_offset.
    bool               ic_random_offset = false;
    double             ic_eps           = 0.0;
    unsigned long long ic_seed          = 0;
    // error_estim == 7: отброшенные циклы Беннеттина.
    int                gs_warmup        = 0;
};


}
