#pragma once
#include "ucuda/cuda/nvrtc_engine.h"
#include "ucuda/equations/adaptive_params.h"
#include "ucuda/equations/dae_ir.h"
#include "ucuda/numerics/integrator.h"
#include "ucuda/numerics/krs_cpu.h"
#include "ucuda/numerics/multistep_bdf.h"
#include <cmath>
#include <cctype>
#include <stdexcept>

namespace regression {
// The same workloads drive accuracy tests and benchmarks. Compilation is outside
// the measured warm call; every benchmark checks its trajectory before and after.
class SolverWorkload {
    std::string name_;
    bool gpu_;
    System sys_;
    std::vector<double> ic_, a_;
    KrsCpuStep fixed_;
    AdaptiveCpuModule adaptive_;
    NvrtcEngine engine_;
    PhaseAdaptiveRequest ad_;
    BdfGpuRequest bdf_;
    BdfSettings settings_;
    bool pendulum_ = false;
public:
    SolverWorkload(std::string name, bool gpu) : name_(std::move(name)), gpu_(gpu) {
        sys_.vars = {"x", "y"}; sys_.rhs = {"-x", "-2*y"};
        ic_ = {1, 1}; a_ = {.5, 4, 1e-12, 25, 1};
        std::vector<KrsCpuDiag> diags;
        if (name_.rfind("adaptive_", 0) == 0) {
            const auto method = scheme_from_name(name_.substr(9));
            const auto code = gbs_stage_count(method) ? codegen_adaptive_gbs(sys_, gbs_stage_count(method)) : codegen_adaptive(sys_, method);
            AdaptiveSettings st; st.enabled = true; st.rtol = "1e-8"; st.atol = "1e-11"; st.hmax = "0.05";
            std::string error;
            if (!adaptive_build_params(st, code, 2, .1, ad_.params, error)) throw std::runtime_error(error);
            ad_.rhs = code.rhs; ad_.emb = code.emb; ad_.dprep = code.dprep; ad_.deval = code.deval;
            ad_.amountOfX = 2; ad_.ic_flat = ic_; ad_.N = 1; ad_.values = a_;
            ad_.t_rec = .1; ad_.dt = .001; ad_.total = 101; ad_.max_pts = 101;
            if (!gpu_ && !adaptive_.compile(code.rhs, code.emb, code.dprep, code.deval, "", 2, "", diags))
                throw std::runtime_error("Adaptive CPU compilation failed");
        } else if (name_.rfind("bdf_", 0) == 0) {
            settings_.max_order = name_.back() - '0'; settings_.rtol = 1e-6; settings_.atol = {1e-9};
            settings_.hmax = .01;
            if (name_.find("fixed") != std::string::npos) settings_.hfix = .001;
            codegen_bdf_bodies(sys_, bdf_.res, bdf_.jac);
            bdf_.amountOfX = bdf_.nx = 2; bdf_.kidx = {1, 1}; bdf_.atol = {1e-9, 1e-9};
            bdf_.rtol = settings_.rtol; bdf_.hmax = settings_.hmax; bdf_.hfix = settings_.hfix;
            bdf_.maxk = settings_.max_order; bdf_.ic_flat = ic_; bdf_.values = a_; bdf_.N = 1;
            bdf_.t_rec = .1; bdf_.dt = .001; bdf_.total = 101; bdf_.max_pts = 101;
        } else {
            DaeScheme method;
            if (!dae_scheme_from_name(name_, &method)) throw std::runtime_error("Unknown DAE method");
            const bool ggl = method == DaeScheme::BDF1GGL || method == DaeScheme::Radau5GGL;
            pendulum_ = ggl || method == DaeScheme::CDIndex2;
            if (pendulum_) {
                const auto equation = std::string("x'=u\ny'=v\nu'=-lam*x\nv'=-lam*y-g\n0=")
                    + (ggl ? "x^2+y^2-L^2" : "x*u+y*v");
                sys_ = parse_dae(equation, "x,y,u,v", "lam", "L,g", ggl ? "" : "lam=2").as_codegen_system();
                ic_ = {std::sin(1.), -std::cos(1.), 0, 0, 9.81 * std::cos(1.)};
                a_ = {.5, 1, 9.81, 4, 1e-12, 25, 1};
            } else {
                sys_ = parse_dae("u'=i/C\n0=E-R*i-u", "u", "i", "R,C,E").as_codegen_system();
                ic_ = {0, 1.5}; a_ = {.5, 2, .5, 3, 4, 1e-12, 25, 1};
            }
            const auto body = codegen_dae_scheme(sys_, method);
            if (gpu_ ? !engine_.compile(body, (int)ic_.size()) : !fixed_.compile(body, (int)ic_.size(), (int)a_.size(), diags))
                throw std::runtime_error("DAE compilation failed: " + engine_.error());
        }
    }
    std::vector<std::vector<double>> run() {
        std::vector<std::vector<double>> out;
        std::string error;
        if (name_.rfind("adaptive_", 0) == 0) {
            if (gpu_) {
                PhaseAdaptiveResult result;
                if (!engine_.run_phase_portraits_adaptive(ad_, result)) throw std::runtime_error(engine_.error());
                out = std::move(result.traj.at(0));
            } else {
                std::vector<double> data(202), times(101); double stats[9]{}, h = 0; int logs = 0;
                const int count = adaptive_.phase()(ic_.data(), a_.data(), &ad_.params, 0, .1, .001, 101, 0, 101, 0,
                    data.data(), times.data(), nullptr, &logs, stats, &h, nullptr);
                if (count != 101 || stats[7] || stats[8]) throw std::runtime_error("Incomplete adaptive trajectory");
                out.resize(101, std::vector<double>(2));
                for (int i = 0; i < 101; ++i) for (int j = 0; j < 2; ++j) out[i][j] = data[i * 2 + j];
            }
        } else if (name_.rfind("bdf_", 0) == 0) {
            if (gpu_) {
                PhaseAdaptiveResult result;
                if (!engine_.run_phase_portraits_bdf(bdf_, result)) throw std::runtime_error(engine_.error());
                out = std::move(result.traj.at(0));
            } else {
                std::vector<double> times; BdfStats stats;
                if (!bdf_integrate(sys_, a_.data(), (int)a_.size(), ic_.data(), settings_, 0, .1, .001, 101, false, 101,
                    out, times, stats, &error)) throw std::runtime_error(error);
                if (stats.diverged || stats.truncated) throw std::runtime_error("Incomplete BDF trajectory");
            }
        } else if (gpu_) {
            std::vector<std::vector<std::vector<double>>> result;
            if (!engine_.run_phase_portraits(ic_, 1, a_, .001, 101, 0, result)) throw std::runtime_error(engine_.error());
            out = std::move(result.at(0));
        } else {
            fixed_.step_index(0);
            if (!computePhasePortraitCPU_custom(fixed_.fn(), ic_.data(), (int)ic_.size(), a_.data(), (int)a_.size(), .001, 101, 0, out))
                throw std::runtime_error("DAE CPU trajectory failed");
        }
        return out;
    }
    void verify(const std::vector<std::vector<double>>& out) const {
        if (out.size() != 101) throw std::runtime_error("Solver trajectory shape changed");
        for (size_t i = 0; i < out.size(); ++i) {
            const auto& y = out[i];
            if (y.size() != ic_.size()) throw std::runtime_error("Solver state shape changed");
            for (double v : y) if (!std::isfinite(v)) throw std::runtime_error("Non-finite solver result");
            const double t = .001 * i;
            if (pendulum_) {
                if (std::fabs(y[0]*y[0]+y[1]*y[1]-1) > 2e-4 || std::fabs(y[0]*y[2]+y[1]*y[3]) > 2e-4)
                    throw std::runtime_error("Pendulum constraint residual");
            } else if (name_.rfind("DAE ", 0) == 0) {
                if (std::fabs(y[0] - 3*(1-std::exp(-t))) > 2e-4 || std::fabs(3-2*y[1]-y[0]) > 1e-9)
                    throw std::runtime_error("RC analytic solution/constraint failed");
            } else for (int j = 0; j < 2; ++j)
                if (std::fabs(y[j]-std::exp(-(j+1)*t)) > 2e-4) throw std::runtime_error("Decay analytic solution failed");
        }
    }
};
inline std::vector<std::string> solver_cases() {
    auto cases = dae_scheme_names();
    for (const char* name : {"RK45", "DOPRI78", "DOPRI78 (legacy)", "DOP853", "GBS 2-4", "GBS 2-4-6", "GBS 2-4-6-8", "GBS 2-4-6-8-10", "GBS 2-4-6-8-10-12"})
        cases.push_back(std::string("adaptive_") + name);
    for (int order = 1; order <= 5; ++order) {
        cases.push_back("bdf_adaptive_" + std::to_string(order));
        cases.push_back("bdf_fixed_" + std::to_string(order));
    }
    return cases;
}
inline std::string solver_case_id(std::string name) {
    for (char& c : name) if (!std::isalnum((unsigned char)c)) c = '_';
    return name;
}
}
