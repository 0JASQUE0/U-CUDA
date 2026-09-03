#include "data_export.h"
#include "parametric_engine.h"
#include "analysis_session.h"   // AnalysisResult for Phase export
#include "configCUDA.h"         // set_precision

#include <iomanip>
#include <cstddef>
#include <cmath>

namespace data_export {

// Провенанс арифметики (см. Bif1DSnapshot::gpu_fmad). Одна строка в каждом
// _config.csv: без неё нельзя отличить файл, посчитанный с FMA-контракцией, от
// файла без неё, а на фрактальных границах бассейнов это разные картинки.
// Объявлена до первого использования — writer'ы идут ниже по файлу.
static void write_fmad_line(std::ofstream& out, bool gpu_fmad)
{
    out << "NVRTC --fmad = " << (gpu_fmad ? "on" : "off") << "\n";
}

// Bif1D

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
    write_fmad_line(out, s.gpu_fmad);
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

// 1D DFT

void write_dft1d_config(std::ofstream& out, const Dft1DSnapshot& s)
{
    if (!out.is_open()) return;
    out << std::setprecision(set_precision);

    out << "1D bifurcation DFT\n";
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
    out << "indexVar for DFT = " << s.writableVar << "\n";
    out << "indexPar for estimation = " << s.indexOfMutVar << "\n";
    out << "start value = " << s.range_lo << ", stop value = " << s.range_hi << "\n";
    out << "n_freq = " << s.n_freq << "\n";
    out << "freq_lo = " << s.freq_lo << ", freq_hi = " << s.freq_hi << "\n";
    const char* win_name = s.window_type == 0 ? "None" : s.window_type == 2 ? "Hamming" : "Hanning";
    out << "window = " << win_name << "\n";
    write_fmad_line(out, s.gpu_fmad);
}

void write_dft1d_header(std::ofstream& out, const Dft1DSnapshot& s)
{
    if (!out.is_open()) return;
    out << std::setprecision(set_precision);
    // Comma separator — matches hostLibrary.cu's bifurcation_DFT_1D exactly
    // (and the pre-existing MATLAB script that reads these files), unlike
    // write_basins_ranges' space-separated header.
    out << s.range_lo << ", " << s.range_hi << "\n";
    out << s.freq_lo  << ", " << s.freq_hi  << "\n";
}

void write_dft1d_matrix(std::ofstream& out, const double* matrix,
                        std::size_t row_offset, int n_rows, int n_freq)
{
    if (!out.is_open()) return;
    for (int i = 0; i < n_rows; ++i) {
        const double* row = matrix + (row_offset + (std::size_t)i) * (std::size_t)n_freq;
        for (int j = 0; j < n_freq; ++j) {
            out << row[j];
            if (j != n_freq - 1) out << ", "; else out << "\n";
        }
    }
}

bool export_dft1d(const Dft1DResult& res, const std::string& path)
{
    std::ofstream cfg(path + "_config.csv");
    if (!cfg.is_open()) return false;
    write_dft1d_config(cfg, res.snapshot);
    cfg.close();

    std::ofstream akFile(path + "_AkCOS.csv");
    std::ofstream bkFile(path + "_BkSIN.csv");
    if (!akFile.is_open() || !bkFile.is_open()) return false;
    akFile << std::setprecision(set_precision);
    bkFile << std::setprecision(set_precision);

    write_dft1d_header(akFile, res.snapshot);
    write_dft1d_header(bkFile, res.snapshot);
    write_dft1d_matrix(akFile, res.ak_cos.data(), 0, res.n_pts, res.n_freq);
    write_dft1d_matrix(bkFile, res.bk_sin.data(), 0, res.n_pts, res.n_freq);
    return true;
}

// LLE1D

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
    write_fmad_line(out, s.gpu_fmad);
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

// LS1D

void write_ls1d_config(std::ofstream& out, const LS1DSnapshot& s)
{
    if (!out.is_open()) return;
    write_curve1d_config_common(out, "1D LS",
                                s.values, s.initial_conditions,
                                s.tMax, s.NT, s.transientTime, s.h, s.eps,
                                s.indexOfMutVar, s.range_lo, s.range_hi);
    write_fmad_line(out, s.gpu_fmad);
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

// 2D shared config helper

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

// Bif2D

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
    out << "mult_peak_DBSCAN  = " << s.mult_peak << "\n";
    out << "mult_interval_DBSCAN = " << s.mult_interval << "\n";
    out << "indexVar for peakfinder = " << s.writableVar << "\n";
    out << "indices = " << s.indexOfMutVar << ", " << s.indexOfMutVar2 << "\n";
    out << "axis1: " << s.range1_lo << " .. " << s.range1_hi << "\n";
    out << "axis2: " << s.range2_lo << " .. " << s.range2_hi << "\n";
    out << "n_pts = " << s.n_pts << "x" << s.n_pts << "\n";
    write_fmad_line(out, s.gpu_fmad);
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

// LLE2D

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
    write_fmad_line(out, s.gpu_fmad);
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

// LS2D

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
    write_fmad_line(out, s.gpu_fmad);
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

// Basins

void write_basins_config(std::ofstream& out, const BasinsSnapshot& s)
{
    if (!out.is_open()) return;
    out << std::setprecision(set_precision);
    out << "Basins of attraction\n";
    write_values_and_ic(out, s.values, s.initial_conditions);
    out << "CT = " << s.tMax << "\nTT = " << s.transientTime
        << "\nh = " << s.h << "\n";
    out << "decimator = " << s.preScaller << "\neps_DBSCAN = " << s.eps_dbscan << "\n";
    // Признаки, по которым кластеризовались ячейки: без них eps_DBSCAN выше не интерпретируется —
    // радиус меряется в пространстве (feature_1 * mult_1, feature_2 * mult_2), и одно и то же
    // значение eps означает разное на «Avg peaks» и на «log10 StDev intervals». Код пишем рядом с
    // названием: имя читает человек, код (BF_* в configCUDA.h) — внешние скрипты, и он переживает
    // переименование подписи.
    out << "feature_1 = " << basin_feature_name(s.feature1) << " (code " << s.feature1
        << "), mult_1 = " << s.mult1 << "\n";
    out << "feature_2 = " << basin_feature_name(s.feature2) << " (code " << s.feature2
        << "), mult_2 = " << s.mult2 << "\n";
    out << "indexVar for peakfinder = " << s.writableVar << "\n";
    out << "indexVar for estimation = " << s.axis_x_var << ", " << s.axis_y_var << "\n";
    out << "start value_1 = " << s.axis_x_lo << ", stop value_1 = " << s.axis_x_hi << "\n";
    out << "start value_2 = " << s.axis_y_lo << ", stop value_2 = " << s.axis_y_hi << "\n";
    write_fmad_line(out, s.gpu_fmad);
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

// FastSync

// Names must match the combo box entries in gui.cpp (FastSync "Synchro
// runtime" section) so the config file reads the same as the UI.
static const char* type_of_synch_name(int v) {
    return v == 1 ? "Bidirectional" : "Unidirectional";
}

static const char* error_estim_name(int v) {
    switch (v) {
        case 1:  return "# iters to reach FS_error_trs";
        case 2:  return "||e|| at last point (sqrt of sum of squares)";
        case 3:  return "time to reach FS_error_trs";
        case 4:  return "err_stop / err_start at window start";
        case 5:  return "log10(err_stop / err_start) at window start";
        default: return "RMS of ||e|| over last window";
    }
}

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
    // В random-режиме IC_slave не участвует в расчёте — пишем его как есть, но
    // рядом отмечаем, что реальные НУ были сгенерированы от первой системы.
    dump_vec("IC_slave",  s.ic_slave);
    out << "ic_random_offset = " << (s.ic_random_offset ? 1 : 0);
    if (s.ic_random_offset)
        out << ", ic_eps = " << s.ic_eps << ", ic_seed = " << s.ic_seed;
    out << "\n";
    dump_vec("k_forward", s.k_forward);
    dump_vec("k_backward", s.k_backward);
    out << "h = " << s.h << "\n";
    out << "iter_of_synchr = " << s.iter_of_synchr << "\n";
    out << "decimator = " << s.preScaller << "\n";
    out << "window = " << s.window << "\n";
    out << "type_of_synch = " << type_of_synch_name(s.type_of_synch)
        << ", error_estim = " << error_estim_name(s.error_estim)
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
    } else {
        out << "TT master = " << s.transientTime << "\n";
        out << "TT slave = "  << s.transientTimeSlave << "\n";
        out << "axis_x_var = " << s.axis_x_var << ", axis_y_var = " << s.axis_y_var << "\n";
        out << "axis_x: " << s.axis_x_lo << " .. " << s.axis_x_hi << "\n";
        out << "axis_y: " << s.axis_y_lo << " .. " << s.axis_y_hi << "\n";
        out << "n_pts = " << s.n_pts << "x" << s.n_pts << "\n";
        out << "grid_swap_master_slave = "
            << (s.grid_swap_master_slave ? 1 : 0) << "\n";
    }
    write_fmad_line(out, s.gpu_fmad);
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

// Phase / TimeSeries

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
    write_fmad_line(out, s.gpu_fmad);
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

// legacy — построчные копии блоков _config.csv из hostLibrary.cu. Каждая строка перенесена
// дословно, включая опечатки и расстановку пробелов (обоснование — в data_export.h). Сверять правки
// надо с форматом, а не с «как правильно»: эти файлы читают внешние скрипты.
namespace legacy {

void write_array(std::ofstream& out, const char* name, const double* v, int n)
{
    out << name << "[" << n << "] = { ";
    for (int kk = 0; kk < n; kk++) {
        if (kk != n - 1) out << v[kk] << ", ";
        else             out << v[kk] << " }\n";
    }
}

void write_estimation(std::ofstream& out, int par_or_var)
{
    if (par_or_var == 1) out << "Parameter esimation \n";
    if (par_or_var == 0) out << "Initial conditions esimation \n";
}

void write_bif1d_config(std::ofstream& out, int set_precision,
                        int continuation_bif1D, int par_or_var,
                        const double* values, int amountOfValues,
                        const double* initialConditions, int amountOfInitialConditions,
                        double tMax, double transientTime, double h, int preScaller,
                        int writableVar, int indexOfMutVar,
                        double range_lo, double range_hi)
{
    if (!out.is_open()) return;
    out << std::setprecision(set_precision);
    if (continuation_bif1D == 1) out << "1D continuation bifurcation \n";
    if (continuation_bif1D == 0) out << "1D classical bifurcation \n";
    write_estimation(out, par_or_var);
    write_array(out, "a",  values,            amountOfValues);
    write_array(out, "X0", initialConditions, amountOfInitialConditions);
    out << "CT = " << tMax << "\n";
    out << "TT = " << transientTime << "\n";
    out << "h = " << h << "\n";
    out << "decimator = " << preScaller << "\n";
    out << "indexVar for peakfinder = " << writableVar << "\n";
    if (par_or_var == 1) out << "indexPar for estimation = " << indexOfMutVar << "\n";
    if (par_or_var == 0) out << "indexVar for estimation = " << indexOfMutVar << "\n";
    out << "start vlaue = " << range_lo << ", stop vlaue = " << range_hi << "\n";
}

void write_bif2d_config(std::ofstream& out, int set_precision, int par_or_var,
                        const double* values, int amountOfValues,
                        const double* initialConditions, int amountOfInitialConditions,
                        double tMax, double transientTime, double h, int preScaller,
                        double eps, double mult_peak, double mult_interval,
                        int writableVar, int idx0, int idx1,
                        const double* ranges)
{
    if (!out.is_open()) return;
    out << std::setprecision(set_precision);
    out << "2D bifurcation \n";
    write_estimation(out, par_or_var);
    write_array(out, "a",  values,            amountOfValues);
    write_array(out, "X0", initialConditions, amountOfInitialConditions);
    out << "CT =  " << tMax << "\n";           // два пробела — так в легаси
    out << "TT =" << transientTime << "\n";    // и здесь ни одного
    out << "h = " << h << "\n";
    out << "decimator = " << preScaller << "\n";
    out << "eps_DBSCAN = " << eps << "\n";
    out << "mult_peak_DBSCAN  = " << mult_peak << "\n";
    out << "mult_interval_DBSCAN = " << mult_interval << "\n";
    out << "indexVar for peakfinder = " << writableVar << "\n";
    if (par_or_var == 1) out << "indexPar for estimation = " << idx0 << ", " << idx1 << "\n";
    if (par_or_var == 0) out << "indexVar for estimation = " << idx0 << ", " << idx1 << "\n";
    out << "start vlaue_1 = " << ranges[0] << ", stop vlaue_1 = " << ranges[1] << "\n";
    out << "start vlaue_2 = " << ranges[2] << ", stop vlaue_2 = " << ranges[3] << "\n";
}

void write_lyap_config(std::ofstream& out, int set_precision, LyapKind kind,
                       int par_or_var,
                       const double* values, int amountOfValues,
                       const double* initialConditions, int amountOfInitialConditions,
                       double tMax, double NT, double transientTime, double h,
                       double eps, const int* indicesOfMutVars, const double* ranges)
{
    if (!out.is_open()) return;
    out << std::setprecision(set_precision);
    switch (kind) {
        case LyapKind::LLE1D: out << "1D LLE \n"; break;
        case LyapKind::LLE2D: out << "2D LLE \n"; break;
        case LyapKind::LS1D:  out << "1D LS \n";  break;
        case LyapKind::LS2D:  out << "2D LS \n";  break;
    }
    write_estimation(out, par_or_var);
    write_array(out, "a",  values,            amountOfValues);
    write_array(out, "X0", initialConditions, amountOfInitialConditions);
    out << "CT =" << " " << tMax << "\n";
    out << "NT =" << " " << NT << "\n";
    out << "TT =" << " " << transientTime << "\n";
    out << "h =" << " " << h << "\n";
    out << "eps=" << " " << eps << "\n";

    const bool two_axes = (kind == LyapKind::LLE2D || kind == LyapKind::LS2D);
    if (kind == LyapKind::LLE1D) {
        // Единственный из четырёх, кто пишет короткое "indexPar =".
        if (par_or_var == 1) out << "indexPar =" << " " << indicesOfMutVars[0] << "\n";
        if (par_or_var == 0) out << "indexVar =" << " " << indicesOfMutVars[0] << "\n";
    } else if (two_axes) {
        if (par_or_var == 1) out << "indexPar for estimation = " << indicesOfMutVars[0] << ", " << indicesOfMutVars[1] << "\n";
        if (par_or_var == 0) out << "indexVar for estimation = " << indicesOfMutVars[0] << ", " << indicesOfMutVars[1] << "\n";
    } else {  // LS1D
        if (par_or_var == 1) out << "indexPar for estimation = " << indicesOfMutVars[0] << "\n";
        if (par_or_var == 0) out << "indexVar for estimation = " << indicesOfMutVars[0] << "\n";
    }

    if (two_axes) {
        out << "start vlaue_1 = " << ranges[0] << ", stop vlaue_1 = " << ranges[1] << "\n";
        out << "start vlaue_2 = " << ranges[2] << ", stop vlaue_2 = " << ranges[3] << "\n";
    } else {
        out << "start vlaue = " << ranges[0] << ", stop vlaue = " << ranges[1] << "\n";
    }
}

void write_basins_config(std::ofstream& out, int set_precision, bool log_axes,
                         const double* values, int amountOfValues,
                         const double* initialConditions, int amountOfInitialConditions,
                         double tMax, double transientTime, double h, int preScaller,
                         double eps, double mult_peak, double mult_interval,
                         int writableVar, int idx0, int idx1,
                         const double* ranges)
{
    if (!out.is_open()) return;
    out << std::setprecision(set_precision);
    // У log-варианта хвостового пробела в заголовке нет — так в легаси.
    out << (log_axes ? "basins of attraction log axes\n" : "basins of attraction \n");
    write_array(out, "a",  values,            amountOfValues);
    write_array(out, "X0", initialConditions, amountOfInitialConditions);
    out << "CT =  " << tMax << "\n";
    out << "TT =" << transientTime << "\n";
    out << "h = " << h << "\n";
    out << "decimator = " << preScaller << "\n";
    out << "eps_DBSCAN = " << eps << "\n";
    out << "mult_MeanPeak_DBSCAN  = " << mult_peak << "\n";
    out << "mult_MeanInterval_DBSCAN = " << mult_interval << "\n";
    out << "indexVar for peakfinder = " << writableVar << "\n";
    out << "indexVar for estimation = " << idx0 << ", " << idx1 << "\n";
    out << "start vlaue_1 = " << ranges[0] << ", stop vlaue_1 = " << ranges[1] << "\n";
    out << "start vlaue_2 = " << ranges[2] << ", stop vlaue_2 = " << ranges[3] << "\n";
}

void write_fastsync_config(std::ofstream& out, int set_precision,
                           int type_of_synch, int error_estim,
                           const double* values, int amountOfValues,
                           const double* icMaster, const double* icSlave,
                           const double* kForward, const double* kBackward,
                           int amountOfInitialConditions,
                           int iterOfSynchr, double tMax, double NTime,
                           double transientTime, double h, int preScaller)
{
    if (!out.is_open()) return;
    out << std::setprecision(set_precision);
    out << "Symmetric synch, attractor \n";
    if (type_of_synch == 0) out << "Unidirectional synch \n";
    if (type_of_synch == 1) out << "Bidirectional synch \n";
    if (error_estim == 0)   out << "RMS(error) on the last iteration \n";
    if (error_estim == 1)   out << "number of iteration to achieve RMS(error) <= FS_error_trs \n";
    if (error_estim == 2)   out << "RMS(error) at the last point \n";
    if (error_estim == 3)   out << "time to achieve RMS(error) <= FS_error_trs \n";
    if (error_estim == 4)   out << "err_stop / err_start at the window start \n";
    if (error_estim == 5)   out << "log10(err_stop / err_start) at the window start \n";
    write_array(out, "a",          values,    amountOfValues);
    write_array(out, "X0_master",  icMaster,  amountOfInitialConditions);
    write_array(out, "X0_slave",   icSlave,   amountOfInitialConditions);
    write_array(out, "K_forward",  kForward,  amountOfInitialConditions);
    write_array(out, "K_backward", kBackward, amountOfInitialConditions);
    out << "iter of synch =  " << iterOfSynchr << "\n";
    out << "CT = " << tMax << "\n";
    out << "WT = " << NTime << "\n";
    out << "TT = " << transientTime << "\n";
    out << "h = " << h << "\n";
    out << "decimator = " << preScaller << "\n";
}

void write_dft1d_config(std::ofstream& out, int set_precision,
                        int continuation_bif1D, int par_or_var,
                        const double* values, int amountOfValues,
                        const double* initialConditions, int amountOfInitialConditions,
                        double tMax, double transientTime, double h, int preScaller,
                        int writableVar, int indexOfMutVar,
                        double range_lo, double range_hi)
{
    if (!out.is_open()) return;
    out << std::setprecision(set_precision);
    if (continuation_bif1D == 1) out << "1D continuation bifurcation DFT \n";
    if (continuation_bif1D == 0) out << "1D classical bifurcation DFT \n";
    write_estimation(out, par_or_var);
    write_array(out, "a",  values,            amountOfValues);
    write_array(out, "X0", initialConditions, amountOfInitialConditions);
    out << "CT =  " << tMax << "\n";
    out << "TT =" << transientTime << "\n";
    out << "h = " << h << "\n";
    out << "decimator = " << preScaller << "\n";
    out << "indexVar for peakfinder = " << writableVar << "\n";
    if (par_or_var == 1) out << "indexPar for estimation = " << indexOfMutVar << "\n";
    if (par_or_var == 0) out << "indexVar for estimation = " << indexOfMutVar << "\n";
    out << "start vlaue_1 = " << range_lo << ", stop vlaue_1 = " << range_hi << "\n";
}

} // namespace legacy

} // namespace data_export
