#include "data_export.h"
#include "parametric_engine.h"
#include "analysis_session.h"   // AnalysisResult for Phase export
#include "configCUDA.h"         // set_precision

#include <iomanip>
#include <cstddef>
#include <cmath>

namespace data_export {

// =============================================================================
// Bif1D
// =============================================================================

void write_bif1d_config(std::ofstream& out, const Bif1DSnapshot& s)
{
    if (!out.is_open()) return;
    out << std::setprecision(set_precision);

    out << "1D classical bifurcation\n";
    out << "Parameter estimation\n";

    const int nv = static_cast<int>(s.values.size());
    out << "a[" << nv << "] = { ";
    for (int kk = 0; kk < nv; ++kk) {
        out << s.values[kk];
        if (kk != nv - 1) out << ", "; else out << " }\n";
    }
    const int nic = static_cast<int>(s.initial_conditions.size());
    out << "X0[" << nic << "] = { ";
    for (int kk = 0; kk < nic; ++kk) {
        out << s.initial_conditions[kk];
        if (kk != nic - 1) out << ", "; else out << " }\n";
    }

    out << "CT = "       << s.tMax << "\n";
    out << "TT = "       << s.transientTime << "\n";
    out << "h = "        << s.h << "\n";
    out << "decimator = " << s.preScaller << "\n";
    out << "indexVar for peakfinder = " << s.writableVar << "\n";
    out << "indexPar for estimation = " << s.indexOfMutVar << "\n";
    out << "start value = " << s.range_lo << ", stop value = " << s.range_hi << "\n";
}

void write_bif1d_rows(std::ofstream& out,
                      double param, int npeaks,
                      const double* peaks, const double* times)
{
    if (!out.is_open()) return;
    if (npeaks == 0) {
        out << param << ", " << 0 << ", " << 0 << '\n';
    } else if (npeaks == -1) {
        out << param << ", " << 0 << ", " << -1 << '\n';
    } else {
        for (int j = 0; j < npeaks; ++j) {
            out << param << ", " << peaks[j] << ", " << times[j] << '\n';
        }
    }
}

bool export_bif1d(const Bifurcation1DResult& res, const std::string& path)
{
    std::ofstream cfg(path + "_config.csv");
    if (!cfg.is_open()) return false;
    write_bif1d_config(cfg, res.snapshot);
    cfg.close();

    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << std::setprecision(set_precision);

    const int n_pts = res.n_pts;
    for (int i = 0; i < n_pts; ++i) {
        const double param = param_value_at(static_cast<std::size_t>(i),
                                            n_pts,
                                            res.snapshot.range_lo,
                                            res.snapshot.range_hi);
        const int npeaks = (i < (int)res.flags.size()) ? res.flags[i] : 0;
        if (npeaks > 0) {
            const auto& pk = res.bifurcation_points[i];
            const auto& tm = res.peak_times[i];
            const int n = (int)pk.size();
            write_bif1d_rows(out, param, n, pk.data(), tm.data());
        } else {
            write_bif1d_rows(out, param, npeaks, nullptr, nullptr);
        }
    }
    return true;
}

// =============================================================================
// LLE1D
// =============================================================================

static void write_curve1d_config_common(std::ofstream& out,
                                        const char* title_line,
                                        const std::vector<double>& values,
                                        const std::vector<double>& initial_conditions,
                                        double tMax, double NT, double transientTime,
                                        double h, double eps,
                                        int indexOfMutVar,
                                        double range_lo, double range_hi)
{
    out << std::setprecision(set_precision);
    out << title_line << "\nParameter estimation\n";

    const int nv = static_cast<int>(values.size());
    out << "a[" << nv << "] = { ";
    for (int kk = 0; kk < nv; ++kk) {
        out << values[kk];
        if (kk != nv - 1) out << ", "; else out << " }\n";
    }
    const int nic = static_cast<int>(initial_conditions.size());
    out << "X0[" << nic << "] = { ";
    for (int kk = 0; kk < nic; ++kk) {
        out << initial_conditions[kk];
        if (kk != nic - 1) out << ", "; else out << " }\n";
    }
    out << "CT = " << tMax << "\nNT = " << NT << "\nTT = " << transientTime << "\n";
    out << "h = "  << h    << "\neps = " << eps << "\n";
    out << "indexPar = " << indexOfMutVar << "\n";
    out << "start value = " << range_lo << ", stop value = " << range_hi << "\n";
}

void write_lle1d_config(std::ofstream& out, const LLE1DSnapshot& s)
{
    if (!out.is_open()) return;
    write_curve1d_config_common(out, "1D LLE",
                                s.values, s.initial_conditions,
                                s.tMax, s.NT, s.transientTime, s.h, s.eps,
                                s.indexOfMutVar, s.range_lo, s.range_hi);
}

void write_lle1d_row(std::ofstream& out, double param, double lyapunov)
{
    if (!out.is_open()) return;
    out << param << ", " << lyapunov << '\n';
}

bool export_lle1d(const LLE1DResult& res, const std::string& path)
{
    std::ofstream cfg(path + "_config.csv");
    if (!cfg.is_open()) return false;
    write_lle1d_config(cfg, res.snapshot);
    cfg.close();

    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << std::setprecision(set_precision);

    const int n_pts = res.n_pts;
    for (int i = 0; i < n_pts; ++i) {
        const double param = param_value_at(static_cast<std::size_t>(i),
                                            n_pts,
                                            res.snapshot.range_lo,
                                            res.snapshot.range_hi);
        const double v = (i < (int)res.lyapunov.size()) ? res.lyapunov[i] : 0.0;
        write_lle1d_row(out, param, v);
    }
    return true;
}

// =============================================================================
// LS1D
// =============================================================================

void write_ls1d_config(std::ofstream& out, const LS1DSnapshot& s)
{
    if (!out.is_open()) return;
    write_curve1d_config_common(out, "1D LS",
                                s.values, s.initial_conditions,
                                s.tMax, s.NT, s.transientTime, s.h, s.eps,
                                s.indexOfMutVar, s.range_lo, s.range_hi);
}

void write_ls1d_row(std::ofstream& out, double param,
                    const double* exponents, int n_exponents)
{
    if (!out.is_open()) return;
    out << param;
    for (int j = 0; j < n_exponents; ++j) out << ", " << exponents[j];
    out << '\n';
}

bool export_ls1d(const LS1DResult& res, const std::string& path)
{
    std::ofstream cfg(path + "_config.csv");
    if (!cfg.is_open()) return false;
    write_ls1d_config(cfg, res.snapshot);
    cfg.close();

    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << std::setprecision(set_precision);

    const int n_pts = res.n_pts;
    const int n_exp = res.n_exponents;
    for (int i = 0; i < n_pts; ++i) {
        const double param = param_value_at(static_cast<std::size_t>(i),
                                            n_pts,
                                            res.snapshot.range_lo,
                                            res.snapshot.range_hi);
        if (i < (int)res.spectrum.size() && !res.spectrum[i].empty())
            write_ls1d_row(out, param, res.spectrum[i].data(), n_exp);
        else {
            std::vector<double> zeros(n_exp, 0.0);
            write_ls1d_row(out, param, zeros.data(), n_exp);
        }
    }
    return true;
}

// =============================================================================
// 2D shared config helper
// =============================================================================

static void write_par_or_var_line(std::ofstream& out, int par_or_var)
{
    if (par_or_var == 1) out << "Parameter estimation\n";
    else if (par_or_var == 0) out << "Initial conditions estimation\n";
    else if (par_or_var == 2) out << "Mixed: x=IC, y=parameter\n";
}

static void write_values_and_ic(std::ofstream& out,
                                const std::vector<double>& values,
                                const std::vector<double>& ic)
{
    const int nv = static_cast<int>(values.size());
    out << "a[" << nv << "] = { ";
    for (int kk = 0; kk < nv; ++kk) {
        out << values[kk];
        if (kk != nv - 1) out << ", "; else out << " }\n";
    }
    const int nic = static_cast<int>(ic.size());
    out << "X0[" << nic << "] = { ";
    for (int kk = 0; kk < nic; ++kk) {
        out << ic[kk];
        if (kk != nic - 1) out << ", "; else out << " }\n";
    }
}

// Writes one row per Y (iy from 0 to n_pts-1), n_pts comma-separated values
// per row. Used by Bif2D and LLE2D. Reads values in user row-major order
// (values[iy*n_pts + ix]).
static void write_grid(std::ofstream& out, int n_pts, const double* values)
{
    if (!out.is_open()) return;
    for (int iy = 0; iy < n_pts; ++iy) {
        for (int ix = 0; ix < n_pts; ++ix) {
            out << values[(std::size_t)iy * n_pts + ix];
            if (ix + 1 < n_pts) out << ", ";
        }
        out << '\n';
    }
}

// =============================================================================
// Bif2D
// =============================================================================

void write_bif2d_config(std::ofstream& out, const Bif2DSnapshot& s)
{
    if (!out.is_open()) return;
    out << std::setprecision(set_precision);
    out << "2D bifurcation (DBSCAN)\n";
    write_par_or_var_line(out, s.par_or_var);
    write_values_and_ic(out, s.values, s.initial_conditions);
    out << "CT = " << s.tMax << "\nTT = " << s.transientTime
        << "\nh = " << s.h << "\ndecimator = " << s.preScaller << "\n";
    out << "eps_DBSCAN = " << s.eps_dbscan << "\n";
    out << "indexVar for peakfinder = " << s.writableVar << "\n";
    out << "indices = " << s.indexOfMutVar << ", " << s.indexOfMutVar2 << "\n";
    out << "axis1: " << s.range1_lo << " .. " << s.range1_hi << "\n";
    out << "axis2: " << s.range2_lo << " .. " << s.range2_hi << "\n";
    out << "n_pts = " << s.n_pts << "x" << s.n_pts << "\n";
}

void write_bif2d_grid(std::ofstream& out, int n_pts, const double* values)
{
    write_grid(out, n_pts, values);
}

bool export_bif2d(const Bifurcation2DResult& res, const std::string& path)
{
    std::ofstream cfg(path + "_config.csv");
    if (!cfg.is_open()) return false;
    write_bif2d_config(cfg, res.snapshot);
    cfg.close();

    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << std::setprecision(set_precision);
    write_bif2d_grid(out, res.n_pts, res.values.data());
    return true;
}

// =============================================================================
// LLE2D
// =============================================================================

void write_lle2d_config(std::ofstream& out, const LLE2DSnapshot& s)
{
    if (!out.is_open()) return;
    out << std::setprecision(set_precision);
    out << "2D LLE\n";
    write_par_or_var_line(out, s.par_or_var);
    write_values_and_ic(out, s.values, s.initial_conditions);
    out << "CT = " << s.tMax << "\nNT = " << s.NT << "\nTT = " << s.transientTime << "\n";
    out << "h = " << s.h << "\neps = " << s.eps << "\n";
    out << "indices = " << s.indexOfMutVar << ", " << s.indexOfMutVar2 << "\n";
    out << "axis1: " << s.range1_lo << " .. " << s.range1_hi << "\n";
    out << "axis2: " << s.range2_lo << " .. " << s.range2_hi << "\n";
    out << "n_pts = " << s.n_pts << "x" << s.n_pts << "\n";
}

void write_lle2d_grid(std::ofstream& out, int n_pts, const double* values)
{
    write_grid(out, n_pts, values);
}

bool export_lle2d(const LLE2DResult& res, const std::string& path)
{
    std::ofstream cfg(path + "_config.csv");
    if (!cfg.is_open()) return false;
    write_lle2d_config(cfg, res.snapshot);
    cfg.close();

    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << std::setprecision(set_precision);
    write_lle2d_grid(out, res.n_pts, res.values.data());
    return true;
}

// =============================================================================
// LS2D
// =============================================================================

void write_ls2d_config(std::ofstream& out, const LS2DSnapshot& s)
{
    if (!out.is_open()) return;
    out << std::setprecision(set_precision);
    out << "2D LS\n";
    write_par_or_var_line(out, s.par_or_var);
    write_values_and_ic(out, s.values, s.initial_conditions);
    out << "CT = " << s.tMax << "\nNT = " << s.NT << "\nTT = " << s.transientTime << "\n";
    out << "h = " << s.h << "\neps = " << s.eps << "\n";
    out << "indices = " << s.indexOfMutVar << ", " << s.indexOfMutVar2 << "\n";
    out << "axis1: " << s.range1_lo << " .. " << s.range1_hi << "\n";
    out << "axis2: " << s.range2_lo << " .. " << s.range2_hi << "\n";
    out << "n_pts = " << s.n_pts << "x" << s.n_pts << "\n";
    out << "exponents = " << s.n_exponents << "\n";
}

void write_ls2d_cells(std::ofstream& out, int n_pts, int n_exponents,
                      const double* values)
{
    if (!out.is_open()) return;
    // values is plane-major: values[k * n_pts^2 + iy * n_pts + ix].
    // Output row order: cell-major in user ordering (iy*n + ix), each row =
    // n_exponents comma-separated values for that cell.
    const std::size_t total = (std::size_t)n_pts * n_pts;
    for (std::size_t cell = 0; cell < total; ++cell) {
        for (int k = 0; k < n_exponents; ++k) {
            out << values[(std::size_t)k * total + cell];
            if (k + 1 < n_exponents) out << ", ";
        }
        out << '\n';
    }
}

bool export_ls2d(const LS2DResult& res, const std::string& path)
{
    std::ofstream cfg(path + "_config.csv");
    if (!cfg.is_open()) return false;
    write_ls2d_config(cfg, res.snapshot);
    cfg.close();

    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << std::setprecision(set_precision);
    write_ls2d_cells(out, res.n_pts, res.n_exponents, res.values.data());
    return true;
}

// =============================================================================
// Basins
// =============================================================================

void write_basins_config(std::ofstream& out, const BasinsSnapshot& s)
{
    if (!out.is_open()) return;
    out << std::setprecision(set_precision);
    out << "Basins of attraction\n";
    write_values_and_ic(out, s.values, s.initial_conditions);
    out << "CT = " << s.tMax << "\nTT = " << s.transientTime
        << "\nh = " << s.h << "\n";
    out << "decimator = " << s.preScaller << "\neps_DBSCAN = " << s.eps_dbscan << "\n";
    out << "indexVar for peakfinder = " << s.writableVar << "\n";
    out << "indexVar for estimation = " << s.axis_x_var << ", " << s.axis_y_var << "\n";
    out << "start value_1 = " << s.axis_x_lo << ", stop value_1 = " << s.axis_x_hi << "\n";
    out << "start value_2 = " << s.axis_y_lo << ", stop value_2 = " << s.axis_y_hi << "\n";
}

void write_basins_ranges(std::ofstream& out, const BasinsSnapshot& s)
{
    if (!out.is_open()) return;
    out << std::setprecision(set_precision);
    out << s.axis_x_lo << " " << s.axis_x_hi << "\n";
    out << s.axis_y_lo << " " << s.axis_y_hi << "\n";
}

void write_basins_grid_int(std::ofstream& out, int n_pts, const int* values)
{
    if (!out.is_open()) return;
    for (int iy = 0; iy < n_pts; ++iy) {
        for (int ix = 0; ix < n_pts; ++ix) {
            out << values[(std::size_t)iy * n_pts + ix];
            if (ix + 1 < n_pts) out << ", ";
        }
        out << '\n';
    }
}

void write_basins_grid_double(std::ofstream& out, int n_pts, const double* values)
{
    if (!out.is_open()) return;
    // Engine had previously substituted NaN/inf with 999.0 when serialising;
    // preserve that to keep downstream parsers happy.
    for (int iy = 0; iy < n_pts; ++iy) {
        for (int ix = 0; ix < n_pts; ++ix) {
            double v = values[(std::size_t)iy * n_pts + ix];
            if (!std::isfinite(v)) v = 999.0;
            out << v;
            if (ix + 1 < n_pts) out << ", ";
        }
        out << '\n';
    }
}

bool export_basins(const BasinsResult& res, const std::string& path)
{
    std::ofstream cfg(path + "_config.csv");
    if (!cfg.is_open()) return false;
    write_basins_config(cfg, res.snapshot);
    cfg.close();

    auto write_one = [&](const std::string& p, auto write_grid_fn,
                         const auto* data) -> bool {
        std::ofstream o(p);
        if (!o.is_open()) return false;
        write_basins_ranges(o, res.snapshot);
        write_grid_fn(o, res.n_pts, data);
        return true;
    };

    if (!write_one(path,             write_basins_grid_int,    res.basin_idx.data()))     return false;
    if (!write_one(path + "_1.csv",  write_basins_grid_double, res.avg_peaks.data()))     return false;
    if (!write_one(path + "_2.csv",  write_basins_grid_double, res.avg_intervals.data())) return false;
    if (!write_one(path + "_3.csv",  write_basins_grid_int,    res.helpful_array.data())) return false;
    return true;
}

// =============================================================================
// FastSync
// =============================================================================

static void write_fastsync_common(std::ofstream& out, const FastSyncSnapshot& s)
{
    const int nv  = static_cast<int>(s.values.size());
    out << "a[" << nv << "] = { ";
    for (int kk = 0; kk < nv; ++kk) {
        out << s.values[kk];
        if (kk != nv - 1) out << ", "; else out << " }\n";
    }
    auto dump_vec = [&](const char* label, const std::vector<double>& v) {
        out << label << "[" << v.size() << "] = { ";
        for (std::size_t kk = 0; kk < v.size(); ++kk) {
            out << v[kk];
            if (kk + 1 != v.size()) out << ", "; else out << " }\n";
        }
    };
    dump_vec("IC_master", s.ic_master);
    dump_vec("IC_slave",  s.ic_slave);
    dump_vec("k_forward", s.k_forward);
    dump_vec("k_backward", s.k_backward);
    out << "h = " << s.h << "\n";
    out << "iter_of_synchr = " << s.iter_of_synchr << "\n";
    out << "decimator = " << s.preScaller << "\n";
    out << "type_of_synch = " << s.type_of_synch
        << ", error_estim = " << s.error_estim
        << ", fs_error_trs = " << s.fs_error_trs << "\n";
}

void write_fastsync_config(std::ofstream& out, const FastSyncSnapshot& s)
{
    if (!out.is_open()) return;
    out << std::setprecision(set_precision);
    if (s.mode == 0) out << "Fast Synchro (On Attractor)\n";
    else             out << "Fast Synchro (On Grid)\n";

    write_fastsync_common(out, s);

    if (s.mode == 0) {
        out << "CT = " << s.tMax << "\nTT = " << s.transientTime << "\n";
        out << "window = " << s.window << "\n";
    } else {
        out << "axis_x_var = " << s.axis_x_var << ", axis_y_var = " << s.axis_y_var << "\n";
        out << "axis_x: " << s.axis_x_lo << " .. " << s.axis_x_hi << "\n";
        out << "axis_y: " << s.axis_y_lo << " .. " << s.axis_y_hi << "\n";
        out << "n_pts = " << s.n_pts << "x" << s.n_pts << "\n";
        out << "grid_swap_master_slave = "
            << (s.grid_swap_master_slave ? 1 : 0) << "\n";
    }
}

// Mode 0 data: header "<var0>,<var1>,...,<varN-1>,sync_error\n" then one
// row per trajectory point "v0,v1,...,v{N-1},err\n". No leading time column
// (sample index is implicit). Matches the FastSync write added by PR #49 —
// engine and GUI share this writer to keep formats byte-identical.
void write_fastsync_attractor(std::ofstream& out, const FastSyncResult& res,
                              const std::vector<std::string>& var_names)
{
    if (!out.is_open()) return;
    const int nX = res.amountOfX_traj;
    for (int j = 0; j < nX; ++j) {
        if (j < (int)var_names.size()) out << var_names[j];
        else                            out << "x" << j;
        out << ",";
    }
    out << "sync_error\n";
    const int n = res.n_pts_traj;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < nX; ++j)
            out << res.traj_full[(std::size_t)i * nX + j] << ",";
        out << res.sync_error[i] << "\n";
    }
}

// Mode 1 data: 2-line ranges header + n×n grid row-major (iy*n + ix) with
// comma separator. Same shape as the PR #49 engine grid write.
void write_fastsync_grid(std::ofstream& out, const FastSyncResult& res,
                         double axis_x_lo, double axis_x_hi,
                         double axis_y_lo, double axis_y_hi)
{
    if (!out.is_open()) return;
    out << axis_x_lo << " " << axis_x_hi << "\n";
    out << axis_y_lo << " " << axis_y_hi << "\n";
    const int n = res.n_pts_grid;
    for (int iy = 0; iy < n; ++iy) {
        for (int ix = 0; ix < n; ++ix) {
            out << res.heatmap[(std::size_t)iy * n + ix];
            if (ix + 1 < n) out << ",";
        }
        out << "\n";
    }
}

// =============================================================================
// Phase / TimeSeries
// =============================================================================

static void write_phase_config(std::ofstream& out, const PhaseSnapshot& s)
{
    if (!out.is_open()) return;
    out << std::setprecision(set_precision);
    out << "Phase portrait / Time domain\n";
    out << "scheme = " << s.scheme << "\n";

    const int nv = static_cast<int>(s.a.size());
    out << "a[" << nv << "] = { ";
    for (int kk = 0; kk < nv; ++kk) {
        out << s.a[kk];
        if (kk != nv - 1) out << ", "; else out << " }\n";
    }
    if (!s.params.empty()) {
        out << "params =";
        for (std::size_t kk = 0; kk < s.params.size(); ++kk)
            out << " " << s.params[kk];
        out << "\n";
    }
    if (!s.vars.empty()) {
        out << "vars =";
        for (std::size_t kk = 0; kk < s.vars.size(); ++kk)
            out << " " << s.vars[kk];
        out << "\n";
    }
    for (std::size_t k = 0; k < s.ic_flat.size(); ++k) {
        out << "X0[" << k << "] (\"" << s.ic_labels[k] << "\") = { ";
        for (std::size_t i = 0; i < s.ic_flat[k].size(); ++i) {
            out << s.ic_flat[k][i];
            if (i + 1 != s.ic_flat[k].size()) out << ", "; else out << " }\n";
        }
    }
    out << "CT = "       << s.t_max     << "\n";
    out << "TT = "       << s.t_skip    << "\n";
    out << "h = "        << s.h         << "\n";
    out << "decimator = " << s.decimator << "\n";
}

static void write_phase_trajectory(std::ofstream& out,
                                   const std::vector<std::string>& vars,
                                   const std::vector<std::vector<double>>& traj,
                                   double dt)
{
    if (!out.is_open()) return;
    out << std::setprecision(set_precision);
    // Header row.
    out << "t";
    for (const auto& v : vars) out << ", " << v;
    out << '\n';
    for (std::size_t i = 0; i < traj.size(); ++i) {
        const double t = static_cast<double>(i) * dt;
        out << t;
        for (double x : traj[i]) out << ", " << x;
        out << '\n';
    }
}

bool export_phase(const AnalysisResult& res, const PhaseSnapshot& snapshot,
                  const std::string& path)
{
    std::ofstream cfg(path + "_config.csv");
    if (!cfg.is_open()) return false;
    write_phase_config(cfg, snapshot);
    cfg.close();

    // dt between recorded points = h * decimator (decimator >= 1).
    const double dt = snapshot.h * static_cast<double>(
        snapshot.decimator > 0 ? snapshot.decimator : 1);

    const std::size_t n_ic = res.trajectories.size();
    if (n_ic == 0) {
        // Empty result: keep <path> as a stub header so users see what would
        // have been written; matches the convention used elsewhere.
        std::ofstream out(path);
        if (!out.is_open()) return false;
        out << std::setprecision(set_precision);
        out << "t";
        for (const auto& v : snapshot.vars) out << ", " << v;
        out << '\n';
        return true;
    }

    if (n_ic == 1) {
        std::ofstream out(path);
        if (!out.is_open()) return false;
        write_phase_trajectory(out, snapshot.vars, res.trajectories[0], dt);
        return true;
    }

    // Multiple ICs → suffix files _ic0.csv, _ic1.csv, ... mirroring the
    // sibling pattern used by Basins (_1.csv, _2.csv, ...).
    for (std::size_t k = 0; k < n_ic; ++k) {
        std::string p = path + "_ic" + std::to_string(k) + ".csv";
        std::ofstream out(p);
        if (!out.is_open()) return false;
        write_phase_trajectory(out, snapshot.vars, res.trajectories[k], dt);
    }
    return true;
}

bool export_fastsync(const FastSyncResult& res, const std::string& path)
{
    std::ofstream cfg(path + "_config.csv");
    if (!cfg.is_open()) return false;
    write_fastsync_config(cfg, res.snapshot);
    cfg.close();

    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << std::setprecision(set_precision);

    if (res.mode == 0) {
        write_fastsync_attractor(out, res, res.snapshot.var_names);
    } else {
        write_fastsync_grid(out, res,
                            res.snapshot.axis_x_lo, res.snapshot.axis_x_hi,
                            res.snapshot.axis_y_lo, res.snapshot.axis_y_hi);
    }
    return true;
}

} // namespace data_export
