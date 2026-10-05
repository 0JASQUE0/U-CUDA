#include "engine_helpers.h"
#include "engine_impl.h"
using namespace ucuda::cuda_detail;
std::string stability_validate(const StabilityRequest& req) {
    if (req.krs_body.empty()) return "krs_body is empty";
    if (req.amountOfX != 2)
        return "the stability region needs the two-dimensional Dahlquist test system "
               "(Library -> Dahlquist 2D): this one has " + std::to_string(req.amountOfX)
               + " state variables";
    if (req.values.empty())                          return "values is empty (at least a[0] is required)";
    if ((int)req.values.size() > kMaxAmountOfValues) return "too many values";

    const int   amountOfValues = (int)req.values.size();
    const int   idx[4]      = { req.idx_a, req.idx_b, req.idx_c, req.idx_d };
    const char* idx_name[4] = { "a", "b", "c", "d" };
    for (int i = 0; i < 4; ++i) {
        // a[0] занят коэффициентом симметрии s во всех расчётах проекта, и
        // положить туда элемент матрицы значило бы молча сломать схему.
        if (idx[i] < 1 || idx[i] >= amountOfValues)
            return std::string("matrix element ") + idx_name[i]
                 + ": a[" + std::to_string(idx[i]) + "] is outside the parameters "
                   "(a[0] is the symmetry s and cannot hold a matrix element)";
        for (int j = 0; j < i; ++j)
            if (idx[i] == idx[j])
                return std::string("matrix elements ") + idx_name[j] + " and " + idx_name[i]
                     + " point at the same a[" + std::to_string(idx[i]) + "]";
    }
    if (req.k == -1.0) return "k = -1 makes the matrix undefined: d = 2*sigma/(1+k)";
    if (!(req.r < 0.0))
        return "r must be negative: with r >= 0 the radicand of c is not positive and "
               "the test matrix has no real form";
    if (!(req.h > 0.0))   return "h must be > 0";
    if (req.n_pts <= 0)   return "the number of points must be > 0";
    if (req.n_pts > 8192) return "the number of points is too large";
    return {};
}

void stability_fill_axes(const StabilityRequest& req, StabilityResult& res) {
    res.n_pts_x = res.n_pts_y = req.n_pts;
    res.sigma_lo = req.sigma_lo; res.sigma_hi = req.sigma_hi;
    res.omega_lo = req.omega_lo; res.omega_hi = req.omega_hi;
    res.k = req.k; res.r = req.r; res.h = req.h;

    auto nodes = [](double lo, double hi, int n, std::vector<double>& out) {
        out.assign((size_t)n, lo);
        if (n < 2) return;
        const double d = (hi - lo) / (double)(n - 1);
        for (int i = 0; i < n; ++i) out[(size_t)i] = lo + d * (double)i;
    };
    nodes(req.sigma_lo, req.sigma_hi, res.n_pts_x, res.sigma_vals);
    nodes(req.omega_lo, req.omega_hi, res.n_pts_y, res.omega_vals);
}

void stability_summarize(StabilityResult& res) {
    res.n_ok = res.n_bad = res.n_stable = 0;
    res.rho_min = res.rho_max = 0.0;
    res.err_min = res.err_max = 0.0;
    bool first = true, first_err = true;
    for (size_t i = 0; i < res.rho.size(); ++i) {
        if (i < res.status.size() && res.status[i] != STAB_ST_OK) { ++res.n_bad; continue; }
        if (!std::isfinite(res.rho[i])) { ++res.n_bad; continue; }
        ++res.n_ok;
        if (res.rho[i] <= 1.0) ++res.n_stable;
        if (first) { res.rho_min = res.rho_max = res.rho[i]; first = false; }
        else {
            if (res.rho[i] < res.rho_min) res.rho_min = res.rho[i];
            if (res.rho[i] > res.rho_max) res.rho_max = res.rho[i];
        }
        if (i < res.err.size() && std::isfinite(res.err[i])) {
            const double e = res.err[i];
            if (first_err) { res.err_min = res.err_max = e; first_err = false; }
            else {
                if (e < res.err_min) res.err_min = e;
                if (e > res.err_max) res.err_max = e;
            }
        }
    }
}

// Спектральная норма 2x2 в замкнутом виде: sigma_max = hypot(E, H) + hypot(F, G)
// с E, F — полусуммой и полуразностью диагонали, G, H — то же для
// недиагонали. В отличие от sqrt((|M|_F^2 + sqrt(|M|_F^4 - 4 det^2)) / 2) здесь
// нет вычитания близких чисел.
static double stab_norm2(double m00, double m01, double m10, double m11) {
    const double E = 0.5 * (m00 + m11), F = 0.5 * (m00 - m11);
    const double G = 0.5 * (m10 + m01), H = 0.5 * (m10 - m01);
    return std::hypot(E, H) + std::hypot(F, G);
}

double stability_step_error(double sigma, double omega, double k, double r, double h,
                            double R00, double R01, double R10, double R11) {
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    if (!(std::isfinite(R00) && std::isfinite(R01) && std::isfinite(R10) && std::isfinite(R11)))
        return qnan;

    // Матрица — ровно та, что в stabilityRegionKernel.
    const double kp1 = k + 1.0, km1 = k - 1.0;
    if (kp1 == 0.0 || r == 0.0) return qnan;
    const double rad = -(1.0 / r) * (sigma * sigma * km1 * km1 / (kp1 * kp1) + omega * omega);
    if (!(rad >= 0.0)) return qnan;
    const double dEl = 2.0 * sigma / kp1;
    const double cEl = -std::sqrt(rad);
    const double bEl = r * cEl;
    const double aEl = k * dEl;

    // (A - sigma I)^2 = -omega^2 I (Кэли-Гамильтон при tr A = 2 sigma,
    // det A = sigma^2 + omega^2), отсюда замкнутая форма экспоненты. При
    // omega -> 0 множитель sin(h omega)/omega -> h; ряд до x^4 держит там
    // полную точность double.
    const double x = h * omega;
    const double sinc = (std::fabs(x) < 1e-4) ? 1.0 - x * x / 6.0 * (1.0 - x * x / 20.0)
                                              : std::sin(x) / x;
    const double co = std::cos(x);
    const double sc = h * sinc;
    const double g  = std::exp(h * sigma);
    const double E00 = g * (co + sc * (aEl - sigma));
    const double E01 = g * (sc * bEl);
    const double E10 = g * (sc * cEl);
    const double E11 = g * (co + sc * (dEl - sigma));

    const double ne = stab_norm2(E00, E01, E10, E11);
    if (!(ne > 0.0) || !std::isfinite(ne)) return std::numeric_limits<double>::infinity();
    const double nd = stab_norm2(E00 - R00, E01 - R01, E10 - R10, E11 - R11);
    if (!std::isfinite(nd)) return std::numeric_limits<double>::infinity();
    return nd / ne;
}

int stability_count_preferred(const StabilityResult& res, double tol) {
    int n = 0;
    for (size_t i = 0; i < res.rho.size(); ++i)
        if (stability_cell_preferred(res, i, tol)) ++n;
    return n;
}

SignalMetricsResult ParametricEngine::run_signal_metrics(const SignalMetricsRequest& req) {
    return impl_->run_signal_metrics(req);
}

NetworkResult ParametricEngine::run_network(const NetworkRequest& req) {
    return impl_->run_network(req);
}
