#pragma once
// Shared CPU/NVRTC arithmetic. Autonomous systems, as in the existing parser.
#ifndef __CUDACC_RTC__
#include <cmath>
#endif
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define UC_AD_HD __host__ __device__
#else
#define UC_AD_HD
#endif
#ifndef NAN
#define NAN (nan(""))
#endif

namespace adaptive {
enum Status { Success = 0, InvalidOptions, NonFinite, StepUnderflow, StepLimit, RejectLimit };
struct Options {
    int method = 0; // 0: fixed, 1: DOPRI5, 2: step doubling
    int order = 0;  // 0: infer built-in order; custom schemes require explicit order
    double rtol = 1e-7, atol = 1e-10;
    double h_initial = 0.01, h_min = 1e-12, h_max = 0.1;
    int max_steps = 1000000, max_rejects = 32;
};
struct Stats {
    int accepted = 0, rejected = 0, status = Success;
    double min_h = 0, max_h = 0;
};
UC_AD_HD inline bool finite(double x) { return x == x && x <= 1.7976931348623157e308 && x >= -1.7976931348623157e308; }
UC_AD_HD inline double smaller(double a, double b) { return a < b ? a : b; }
UC_AD_HD inline double larger(double a, double b) { return a > b ? a : b; }
UC_AD_HD inline double clamp(double x, double a, double b) { return smaller(b, larger(a, x)); }
UC_AD_HD inline bool valid(const Options& o) {
    return o.method >= 0 && o.method <= 2 && finite(o.rtol) && o.rtol >= 0 &&
        finite(o.atol) && o.atol > 0 && finite(o.h_min) && o.h_min > 0 &&
        finite(o.h_max) && o.h_max >= o.h_min && finite(o.h_initial) &&
        o.h_initial >= o.h_min && o.h_initial <= o.h_max &&
        o.max_steps > 0 && o.max_rejects > 0 && o.order >= 0 && o.order <= 8 &&
        (o.method != 2 || o.order > 0);
}
UC_AD_HD inline const char* status_text(int code) {
    switch (code) {
    case Success: return "success";
    case InvalidOptions: return "invalid adaptive settings or sampling interval";
    case NonFinite: return "non-finite initial state or result";
    case StepUnderflow: return "tolerance cannot be met at h_min (or time cannot advance)";
    case StepLimit: return "adaptive step budget exhausted";
    case RejectLimit: return "too many consecutive rejected steps";
    default: return "unknown adaptive failure";
    }
}

// Model provides rhs(y,a,f) and step(y,a,h). N can include a Lyapunov ensemble.
template<int N, class Model> struct Solver {
    Options options;
    Stats stats;
    Model model;
    const double* params;
    double y[N], old[N], k[7][N];
    double t = 0, old_t = 0, last_h = 0, next_h, previous_error = 1;
    int consecutive_rejects = 0;
    bool have_previous = false;

    UC_AD_HD Solver(const double* initial, const double* a, Options o, Model m = Model{})
        : options(o), model(m), params(a), next_h(o.h_initial) {
        if (!valid(o)) stats.status = InvalidOptions;
        for (int i = 0; i < N; ++i) {
            y[i] = old[i] = initial[i];
            if (!finite(y[i])) stats.status = NonFinite;
        }
    }

    // Dormand-Prince 5(4) tableau; error is the embedded fourth-minus-fifth pair.
    UC_AD_HD void trial(double dt, double* candidate, double* error) {
        if (options.method == 2) {
            double coarse[N];
            for (int i = 0; i < N; ++i) coarse[i] = candidate[i] = y[i];
            model.step(coarse, params, dt);
            model.step(candidate, params, dt * 0.5);
            model.step(candidate, params, dt * 0.5);
            const double divisor = (1 << options.order) - 1;
            for (int i = 0; i < N; ++i) error[i] = (candidate[i] - coarse[i]) / divisor;
            return;
        }
        const double a[7][7] = {
            {0}, {1./5}, {3./40, 9./40}, {44./45, -56./15, 32./9},
            {19372./6561, -25360./2187, 64448./6561, -212./729},
            {9017./3168, -355./33, 46732./5247, 49./176, -5103./18656},
            {35./384, 0, 500./1113, 125./192, -2187./6784, 11./84}
        };
        const double e[7] = {-71./57600, 0, 71./16695, -71./1920, 17253./339200, -22./525, 1./40};
        model.rhs(y, params, k[0]);
        for (int stage = 1; stage < 7; ++stage) {
            for (int i = 0; i < N; ++i) {
                double sum = 0;
                for (int j = 0; j < stage; ++j) sum += a[stage][j] * k[j][i];
                candidate[i] = y[i] + dt * sum;
            }
            model.rhs(candidate, params, k[stage]);
        }
        for (int i = 0; i < N; ++i) {
            double sum = 0;
            for (int j = 0; j < 7; ++j) sum += e[j] * k[j][i];
            error[i] = dt * sum;
        }
    }

    UC_AD_HD bool advance(double bound) {
        if (stats.status != Success) return false;
        const double direction = bound >= t ? 1. : -1.;
        while (true) {
            if (stats.accepted + stats.rejected >= options.max_steps) {
                stats.status = StepLimit; return false;
            }
            const double remaining = fabs(bound - t);
            double dt_abs = smaller(next_h, remaining);
            double dt = direction * dt_abs;
            if (!finite(bound) || t + dt == t) { stats.status = StepUnderflow; return false; }
            double candidate[N], error[N];
            trial(dt, candidate, error);
            double norm = 0;
            bool all_finite = true;
            for (int i = 0; i < N; ++i) {
                const double scale = options.atol + options.rtol * larger(fabs(y[i]), fabs(candidate[i]));
                double component = fabs(error[i]) / scale;
                if (!finite(candidate[i]) || !finite(component) || !finite(scale)) all_finite = false;
                norm = larger(norm, component);
            }
            const double exponent = 1. / (options.method == 1 ? 5 : options.order + 1);
            if (all_finite && norm <= 1) {
                double factor = norm == 0 ? 5. : 0.9 * pow(norm, -exponent);
                if (have_previous && norm > 0)
                    factor = 0.9 * pow(norm, -0.7 * exponent) * pow(previous_error, 0.4 * exponent);
                factor = clamp(factor, 0.2, consecutive_rejects ? 1. : 5.);
                old_t = t; last_h = dt;
                t = dt_abs == remaining ? bound : t + dt;
                for (int i = 0; i < N; ++i) { old[i] = y[i]; y[i] = candidate[i]; }
                ++stats.accepted;
                if (stats.min_h == 0 || dt_abs < stats.min_h) stats.min_h = dt_abs;
                stats.max_h = larger(stats.max_h, dt_abs);
                previous_error = larger(norm, 1e-16); have_previous = true;
                consecutive_rejects = 0;
                next_h = clamp(dt_abs * factor, options.h_min, options.h_max);
                return true;
            }
            ++stats.rejected;
            if (++consecutive_rejects >= options.max_rejects) { stats.status = RejectLimit; return false; }
            if (dt_abs <= options.h_min) { stats.status = StepUnderflow; return false; }
            double factor = all_finite ? clamp(0.9 * pow(norm, -exponent), 0.1, 0.5) : 0.1;
            next_h = larger(options.h_min, dt_abs * factor);
        }
    }

    // Shampine quartic dense extension (1986). Rational coefficients are
    // cross-checked against scipy.integrate.RK45's published tableau.
    UC_AD_HD void interpolate(double target, double* out) const {
        if (target == t || last_h == 0) {
            for (int i = 0; i < N; ++i) out[i] = y[i];
            return;
        }
        const double p[7][4] = {
            {1, -8048581381./2820520608, 8663915743./2820520608, -12715105075./11282082432},
            {0,0,0,0},
            {0,131558114200./32700410799,-68118460800./10900136933,87487479700./32700410799},
            {0,-1754552775./470086768,14199869525./1410260304,-10690763975./1880347072},
            {0,127303824393./49829197408,-318862633887./49829197408,701980252875./199316789632},
            {0,-282668133./205662961,2019193451./616988883,-1453857185./822651844},
            {0,40617522./29380423,-110615467./29380423,69997945./29380423}
        };
        const double theta = (target - old_t) / last_h;
        for (int i = 0; i < N; ++i) {
            double sum = 0;
            for (int stage = 0; stage < 7; ++stage)
                sum += k[stage][i] * theta * (p[stage][0] + theta *
                    (p[stage][1] + theta * (p[stage][2] + theta * p[stage][3])));
            out[i] = old[i] + last_h * sum;
        }
    }

    // Monotonic forward sample requests; dense steps never pass the final bound.
    UC_AD_HD bool sample(double target, double bound, double* out) {
        if (!finite(target) || target < old_t || target > bound) { stats.status = InvalidOptions; return false; }
        while (t < target) if (!advance(options.method == 1 ? bound : target)) return false;
        interpolate(target, out);
        for (int i = 0; i < N; ++i) if (!finite(out[i])) { stats.status = NonFinite; return false; }
        return stats.status == Success;
    }

    UC_AD_HD bool integrate(double duration, double* out) {
        if (!finite(duration)) { stats.status = InvalidOptions; return false; }
        while (t != duration) if (!advance(duration)) return false;
        for (int i = 0; i < N; ++i) out[i] = y[i];
        return stats.status == Success;
    }
};
template<int D, int Count, class Base> struct Ensemble {
    Base base;
    UC_AD_HD void rhs(const double* x, const double* a, double* f) const {
        for(int j=0;j<Count;++j) base.rhs(x+j*D,a,f+j*D);
    }
    UC_AD_HD void step(double* x, const double* a, double h) const {
        for(int j=0;j<Count;++j) base.step(x+j*D,a,h);
    }
};
template<int D, int Count, class Base>
UC_AD_HD bool integrate_ensemble(double* x, double* probes, const double* a,
    double duration, double max_value, Options options) {
    double all[D*Count];
    for(int i=0;i<D;++i) all[i]=x[i];
    for(int i=0;i<D*(Count-1);++i) all[D+i]=probes[i];
    Solver<D*Count,Ensemble<D,Count,Base>> solver(all,a,options);
    if(!solver.integrate(duration,all)) return false;
    for(int i=0;i<D*Count;++i) if(!finite(all[i]) || (max_value > 0 && fabs(all[i])>max_value)) return false;
    for(int i=0;i<D;++i) x[i]=all[i];
    for(int i=0;i<D*(Count-1);++i) probes[i]=all[D+i];
    return true;
}
// The master is sampled at the existing synchronization interval. Coupling is
// included in the local error estimate; backward time reverses the ODE only.
template<int D, class Base> struct SyncModel {
    Base base;
    const double* master;
    const double* gain;
    double direction;
    UC_AD_HD void rhs(const double* x,const double* a,double* f) const {
        base.rhs(x,a,f);
        for(int i=0;i<D;++i) f[i]=direction*f[i]+gain[i]*(master[i]-x[i]);
    }
    UC_AD_HD void step(double* x,const double* a,double h) const {
        double coupling[D];
        for(int i=0;i<D;++i) coupling[i]=gain[i]*(master[i]-x[i]);
        base.step(x,a,direction*h);
        for(int i=0;i<D;++i) x[i]+=h*coupling[i];
    }
};
} // namespace adaptive

