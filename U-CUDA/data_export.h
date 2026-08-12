#pragma once
//
// data_export — shared CSV writer for plot results.
//
// Both `parametric_engine` (during compute, when csv_save_enabled is on) and
// `gui` (on right-click "Export data...") call into this module so the on-disk
// format stays identical byte-for-byte. The engine streams chunked rows; the
// GUI dumps the full in-memory result. They share the same per-row formatter
// and the same `_config.csv` writer.
//
// Snapshot structs are tiny PODs captured by the engine at run start and
// carried inside the corresponding *Result so the GUI can reproduce the
// `_config.csv` header after compute (when the original Request is gone).
//

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

// Forward declarations of result types defined in parametric_engine.h. The
// .cpp pulls in the full header; consumers of this header that only need the
// snapshot structs do not need parametric_engine.h.
struct Bifurcation1DResult;
struct Bifurcation2DResult;
struct Dft1DResult;
struct LLE1DResult;
struct LLE2DResult;
struct LS1DResult;
struct LS2DResult;
struct BasinsResult;
struct FastSyncResult;
// AnalysisResult is defined in analysis_session.h.
struct AnalysisResult;

namespace data_export {

// =============================================================================
// Snapshot structs — exactly the data engine currently prints into _config.csv.
// Names follow the engine's local variable names (tMax / TT / etc.) so the
// 1-to-1 mapping in writers is obvious.
// =============================================================================

// Провенанс расчёта: режим FMA-контракции NVRTC, действовавший на момент Run
// (Settings -> GPU floating point, см. set_nvrtc_fmad). Пишется в _config.csv
// каждым writer'ом строкой "NVRTC --fmad = on/off": результаты, посчитанные с
// разным значением, не сравнимы побитово, а на фрактальных границах бассейнов
// расходятся и визуально — без пометки в файле потом не установить, чем считалось.
//
// Снимается на старте прогона, а не при экспорте: настройку могли переключить
// между расчётом и сохранением.
//
// Это настройка приложения, а не свойство устройства: две ветки считают на CPU
// и через NVRTC не проходят вовсе (run_bif1d_cpu — continuation, и фазовый
// портрет при снятом GPU). Там строка остаётся справочной — она говорит, как
// был настроен компилятор ядер, а не как считался этот конкретный файл.
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
    int    window_type = 1;   // 0=None, 1=Hanning, 2=Hamming — см. Dft1DRequest::window_type
};

struct LLE1DSnapshot {
    bool   gpu_fmad = true;   // см. Bif1DSnapshot::gpu_fmad
    std::vector<double> values;
    std::vector<double> initial_conditions;
    double tMax = 0.0;
    double NT   = 0.0;
    double transientTime = 0.0;
    double h    = 0.0;
    double eps  = 0.0;
    int    indexOfMutVar = 0;
    double range_lo = 0.0;
    double range_hi = 0.0;
};

// LS1D shares the same header layout as LLE1D (engine writes "1D LS" instead
// of "1D LLE") — same fields. Kept as a distinct type for clarity at call site.
struct LS1DSnapshot {
    bool   gpu_fmad = true;   // см. Bif1DSnapshot::gpu_fmad
    std::vector<double> values;
    std::vector<double> initial_conditions;
    double tMax = 0.0;
    double NT   = 0.0;
    double transientTime = 0.0;
    double h    = 0.0;
    double eps  = 0.0;
    int    indexOfMutVar = 0;
    double range_lo = 0.0;
    double range_hi = 0.0;
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
    double h        = 0.0;
    double t_max    = 0.0;
    double t_skip   = 0.0;
    int    decimator = 1;
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
    int    axis_x_var = 0, axis_y_var = 0;
    double axis_x_lo = 0.0, axis_x_hi = 0.0;
    double axis_y_lo = 0.0, axis_y_hi = 0.0;
    int    n_pts = 0;
    bool   grid_swap_master_slave = false;
};

// =============================================================================
// Engine-side writers. Engine calls these in chunked-streaming loops; the GUI
// calls them as part of export_*. Keeping them in this module means the row
// format only exists in one place.
// =============================================================================

// Writes the _config.csv header for a 1D classical bifurcation run. The
// caller passes an already-opened ofstream; this function sets precision
// itself, matching engine behaviour.
void write_bif1d_config(std::ofstream& out, const Bif1DSnapshot& s);

// Writes one parameter-bin's worth of rows for a 1D bifurcation. Mirrors the
// engine's inner loop at parametric_engine.cpp:806-818:
//   npeaks ==  0 → "<param>, 0, 0\n"
//   npeaks == -1 → "<param>, 0, -1\n"
//   npeaks >  0 → npeaks rows of "<param>, <peak[j]>, <time[j]>\n"
// `peaks` and `times` must point to npeaks valid doubles when npeaks > 0.
void write_bif1d_rows(std::ofstream& out,
                      double param, int npeaks,
                      const double* peaks, const double* times);

// Linear interpolation matching engine-side getValueByIdx_local. Exposed so
// GUI export can compute the same param value for index i without depending
// on the engine.
inline double param_value_at(std::size_t idx, int n_pts, double lo, double hi) {
    if (n_pts <= 1) return lo;
    return lo + (hi - lo) * static_cast<double>(idx) /
                            static_cast<double>(n_pts - 1);
}

// 1D DFT. Engine writes 3 files sharing one path prefix: <path>_config.csv
// (human-readable metadata, write_dft1d_config), <path>_AkCOS.csv and
// <path>_BkSIN.csv (write_dft1d_header once + write_dft1d_matrix per chunk/
// in full). Header rows use COMMA separators (unlike write_basins_ranges'
// space separator) to stay byte-compatible with the pre-existing MATLAB
// script that reads these files (and with hostLibrary.cu's own bifurcation_
// DFT_1D, which writes the same 2-row comma-separated header).
void write_dft1d_config(std::ofstream& out, const Dft1DSnapshot& s);

// Writes the 2 header rows: "range_lo, range_hi\n" then "freq_lo, freq_hi\n".
// Call once per file (AkCOS/BkSIN) before any data rows.
void write_dft1d_header(std::ofstream& out, const Dft1DSnapshot& s);

// Appends `n_rows` rows of `n_freq` comma-separated values, starting at
// matrix[row_offset*n_freq]. Called once per chunk by the engine (streaming
// append) and once with the full matrix (row_offset=0) by export_dft1d.
void write_dft1d_matrix(std::ofstream& out, const double* matrix,
                        std::size_t row_offset, int n_rows, int n_freq);

// LLE1D / LS1D — same row shape as engine (chunked append).
void write_lle1d_config(std::ofstream& out, const LLE1DSnapshot& s);
void write_lle1d_row(std::ofstream& out, double param, double lyapunov);

void write_ls1d_config(std::ofstream& out, const LS1DSnapshot& s);
void write_ls1d_row(std::ofstream& out, double param,
                    const double* exponents, int n_exponents);

// 2D writers operate on the whole grid (engine calls AFTER chunked compute
// loop finishes, on res.values which is already in user ordering). This drops
// the chunked / kernel-order layout the engine used to emit, in favour of a
// clean row-per-Y grid that GUI export can reproduce trivially.
void write_bif2d_config(std::ofstream& out, const Bif2DSnapshot& s);
void write_bif2d_grid(std::ofstream& out, int n_pts, const double* values);

void write_lle2d_config(std::ofstream& out, const LLE2DSnapshot& s);
void write_lle2d_grid(std::ofstream& out, int n_pts, const double* values);

// LS2D: one row per cell (in user row-major order), n_exponents values per
// cell. Cell layout in res.values is per-plane (k*n*n + iy*n + ix), so the
// writer transposes from "plane-major" to "cell-major" on the fly.
void write_ls2d_config(std::ofstream& out, const LS2DSnapshot& s);
void write_ls2d_cells(std::ofstream& out, int n_pts, int n_exponents,
                      const double* values);

// Basins. Engine writes a 4-file set sharing one path prefix: <path> (basin
// indices, int), <path>_1.csv (avg_peaks, double), <path>_2.csv (avg_intervals,
// double), <path>_3.csv (helpful_array, int). Each data file is preceded by
// a 2-line ranges header (axis_x_lo axis_x_hi / axis_y_lo axis_y_hi).
void write_basins_config(std::ofstream& out, const BasinsSnapshot& s);
void write_basins_ranges(std::ofstream& out, const BasinsSnapshot& s);
void write_basins_grid_int(std::ofstream& out, int n_pts, const int* values);
void write_basins_grid_double(std::ofstream& out, int n_pts, const double* values);

// FastSync — engine + GUI share these writers so on-disk format is
// byte-identical (the engine-side path was added by PR #49).
//   mode 0: <path> header "<var0>,...,<varN-1>,sync_error\n" + row per traj
//           point. Engine optionally calls write_fastsync_config separately;
//           the GUI export always writes both.
//   mode 1: <path> two-line ranges header + n×n grid (row per Y) of sync
//           errors with comma separator.
void write_fastsync_config(std::ofstream& out, const FastSyncSnapshot& s);
void write_fastsync_attractor(std::ofstream& out, const FastSyncResult& res,
                              const std::vector<std::string>& var_names);
void write_fastsync_grid(std::ofstream& out, const FastSyncResult& res,
                         double axis_x_lo, double axis_x_hi,
                         double axis_y_lo, double axis_y_hi);

// =============================================================================
// GUI entry points. Writes both <path> (data) and <path>_config.csv (header).
// Returns false if any file could not be opened; partial writes are possible
// only if disk fills mid-write. The caller is responsible for choosing `path`
// via a save-file dialog (or any other source).
// =============================================================================

bool export_bif1d(const Bifurcation1DResult& res, const std::string& path);
bool export_dft1d (const Dft1DResult&         res, const std::string& path);
bool export_lle1d(const LLE1DResult&         res, const std::string& path);
bool export_ls1d (const LS1DResult&          res, const std::string& path);
bool export_bif2d(const Bifurcation2DResult& res, const std::string& path);
bool export_lle2d(const LLE2DResult&         res, const std::string& path);
bool export_ls2d (const LS2DResult&          res, const std::string& path);
bool export_basins(const BasinsResult&       res, const std::string& path);
bool export_fastsync(const FastSyncResult&   res, const std::string& path);

// Phase / TimeSeries — there is no engine-side CSV; the format is defined
// fresh here. <path>_config.csv carries scheme + params + ICs + integration
// settings; <path> (single IC) or <path>_ic0.csv/_ic1.csv/... (multi-IC)
// carries one row per step with columns "t, x0, x1, ..., xN-1".
bool export_phase(const AnalysisResult& res, const PhaseSnapshot& snapshot,
                  const std::string& path);

// =============================================================================
// legacy — writer'ы `_config.csv` для hostLibrary.cu (Debug entry point).
//
// Формат ЭТИХ файлов не совпадает с writer'ами выше и намеренно оставлен как
// есть: его читают внешние скрипты обработки. Отсюда сохранённые странности —
// опечатки ("esimation", "vlaue"), пробел перед \n в заголовках, "CT =  " с
// двумя пробелами в 2D против "CT = " в 1D, "TT =" вовсе без пробела, "eps="
// без пробела перед знаком. Всё это воспроизведено дословно; «починка» любой
// строки ломает чужой парсер. Функции перенесены сюда только затем, чтобы
// CSV-форматирование проекта жило в одном модуле, а не двумя копиями.
//
// Сигнатуры берут double НАМЕРЕННО: в этих файлах числа обязаны оставаться
// double независимо от typedef numb, иначе внешние парсеры получили бы 15
// заявленных значащих цифр от float'а.
//
// Раньше это работало как растяжка: при numb = float вызовы из hostLibrary.cu
// просто перестали бы компилироваться (громкий отказ вместо тихой потери
// точности). Растяжка сработала ровно так, как задумано, и теперь заменена на
// явное расширение в единственном месте, которому оно нужно, — to_dbl() в
// hostLibrary.cu, см. комментарий там. То есть numb = float собирается, а
// конверсия видна в коде вызова, а не спрятана в сигнатуре.
// Менять эти double на numb нельзя: это и вернёт ту самую тихую потерю.
// =============================================================================
namespace legacy {

// "<name>[N] = { v, v, ... }\n". При N == 0 легаси печатает только
// "<name>[0] = { " — без закрывающей скобки и без перевода строки. Так и
// оставлено: менять — значит менять формат.
void write_array(std::ofstream& out, const char* name, const double* v, int n);

// "Parameter esimation \n" / "Initial conditions esimation \n". При прочих
// значениях par_or_var не печатает ничего — как и легаси.
void write_estimation(std::ofstream& out, int par_or_var);

// bifurcation1D.
void write_bif1d_config(std::ofstream& out, int set_precision,
                        int continuation_bif1D, int par_or_var,
                        const double* values, int amountOfValues,
                        const double* initialConditions, int amountOfInitialConditions,
                        double tMax, double transientTime, double h, int preScaller,
                        int writableVar, int indexOfMutVar,
                        double range_lo, double range_hi);

// bifurcation2D.
void write_bif2d_config(std::ofstream& out, int set_precision, int par_or_var,
                        const double* values, int amountOfValues,
                        const double* initialConditions, int amountOfInitialConditions,
                        double tMax, double transientTime, double h, int preScaller,
                        double eps, double mult_peak, double mult_interval,
                        int writableVar, int idx0, int idx1,
                        const double* ranges);

// LLE1D / LLE2D / LS1D / LS2D — один writer на четыре блока: тела совпадали
// дословно, расходились только заголовок, стиль строки index и число
// диапазонов. LLE1D — единственный, кто пишет короткое "indexPar =" вместо
// "indexPar for estimation =".
enum class LyapKind { LLE1D, LLE2D, LS1D, LS2D };
void write_lyap_config(std::ofstream& out, int set_precision, LyapKind kind,
                       int par_or_var,
                       const double* values, int amountOfValues,
                       const double* initialConditions, int amountOfInitialConditions,
                       double tMax, double NT, double transientTime, double h,
                       double eps, const int* indicesOfMutVars, const double* ranges);

// basinsOfAttraction и basinsOfAttraction_logAxes: тела совпадали, расходился
// только заголовок (и у log-варианта в нём нет хвостового пробела).
void write_basins_config(std::ofstream& out, int set_precision, bool log_axes,
                         const double* values, int amountOfValues,
                         const double* initialConditions, int amountOfInitialConditions,
                         double tMax, double transientTime, double h, int preScaller,
                         double eps, double mult_peak, double mult_interval,
                         int writableVar, int idx0, int idx1,
                         const double* ranges);

// FastSynchro (attractor).
void write_fastsync_config(std::ofstream& out, int set_precision,
                           int type_of_synch, int error_estim,
                           const double* values, int amountOfValues,
                           const double* icMaster, const double* icSlave,
                           const double* kForward, const double* kBackward,
                           int amountOfInitialConditions,
                           int iterOfSynchr, double tMax, double NTime,
                           double transientTime, double h, int preScaller);

// bifurcation_DFT_1D. От write_bif1d_config отличается заголовком ("... DFT"),
// строкой "CT =  " с двумя пробелами, "TT =" без пробела и тем, что диапазон
// печатается как "vlaue_1" при единственной оси.
void write_dft1d_config(std::ofstream& out, int set_precision,
                        int continuation_bif1D, int par_or_var,
                        const double* values, int amountOfValues,
                        const double* initialConditions, int amountOfInitialConditions,
                        double tMax, double transientTime, double h, int preScaller,
                        int writableVar, int indexOfMutVar,
                        double range_lo, double range_hi);

} // namespace legacy

} // namespace data_export
