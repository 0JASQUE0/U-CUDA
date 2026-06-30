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

#include <fstream>
#include <string>
#include <vector>

// Forward declarations of result types defined in parametric_engine.h. The
// .cpp pulls in the full header; consumers of this header that only need the
// snapshot structs do not need parametric_engine.h.
struct Bifurcation1DResult;
struct Bifurcation2DResult;
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

struct Bif1DSnapshot {
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

struct LLE1DSnapshot {
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
    std::vector<double> values;
    std::vector<double> initial_conditions;
    int    par_or_var = 1;                   // 1=param/param, 0=IC/IC, 2=mixed
    double tMax = 0.0;
    double transientTime = 0.0;
    double h    = 0.0;
    int    preScaller = 0;
    double eps_dbscan = 0.0;
    int    writableVar = 0;
    int    indexOfMutVar  = 0;
    int    indexOfMutVar2 = 0;
    double range1_lo = 0.0, range1_hi = 0.0;
    double range2_lo = 0.0, range2_hi = 0.0;
    int    n_pts = 0;
};

struct LLE2DSnapshot {
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

} // namespace data_export
