#include "expression_internal.h"
using namespace ucuda::equations_detail;
bool is_adaptive_body(const std::string& body) {
    return body.find("/*UC_ADAPTIVE_V1*/") == 0;
}
std::string adaptive_model_prefix(const std::string& body) {
    auto split = body.find("/*UC_ADAPTIVE_RUN*/");
    if (!is_adaptive_body(body) || split == std::string::npos)
        throw std::runtime_error("invalid adaptive generated body");
    return body.substr(0, split);
}
std::string codegen_adaptive(const System& sys, const std::string& scheme,
    const std::string& base_body, bool custom, const adaptive::Options& settings) {
    if (settings.method == 0) return base_body;
    adaptive::Options o = settings;
    if (o.method == 2 && o.order == 0) {
        if (custom) throw std::runtime_error("Step doubling: specify the custom scheme order (1..8)");
        o.order = 1;
    }
    if (!adaptive::valid(o)) throw std::runtime_error("Invalid adaptive settings: check tolerances, step bounds and limits");
    if (custom && o.method == 1)
        throw std::runtime_error("DOPRI5 uses the ODE right-hand side; select a built-in scheme or use step doubling for custom KRS");
    std::ostringstream out;
    out << std::setprecision(17) << "/*UC_ADAPTIVE_V1*/\nstruct UcAdaptiveModel {\n"
        << "UC_AD_HD void rhs(const double* X, const double* a, double* f) const {\n";
    if (o.method == 1) {
        auto expressions = rhs_over(sys, "X");
        for (size_t i=0; i<expressions.size(); ++i) out << "f[" << i << "]=" << expressions[i] << ";\n";
    } else out << "for(int i=0;i<AMOUNTOFX;++i) f[i]=0;\n";
    out << "}\nUC_AD_HD void step(double* X, const double* a, double h) const {\n"
        << (o.method == 2 ? base_body : "") << "\n}\n};\n"
        << "adaptive::Options uc_options;\n"
        << "uc_options.method=" << o.method << ";\nuc_options.order=" << o.order << ";\n"
        << "uc_options.rtol=" << o.rtol << ";\nuc_options.atol=" << o.atol << ";\n"
        << "uc_options.h_initial=" << o.h_initial << ";\nuc_options.h_min=" << o.h_min << ";\n"
        << "uc_options.h_max=" << o.h_max << ";\nuc_options.max_steps=" << o.max_steps << ";\n"
        << "uc_options.max_rejects=" << o.max_rejects << ";\n";
    if (o.method == 2 && settings.order == 0) {
        int order = 1;
        if (scheme == "RK4") order = 4;
        else if (scheme == "DOPRI78") order = 8;
        else if (scheme == "Explicit Midpoint" || scheme == "Implicit Midpoint") order = 2;
        out << "uc_options.order=" << order << ";\n";
        if (scheme == "CD" || scheme == "Complex CD" || scheme == "SEMP" || scheme == "SIMP" || scheme == "Complex Implicit Euler")
            out << "uc_options.order=(fabs(a[0]-0.5)<1e-14 ? 2 : 1);\n";
        if (scheme == "Complex CD4") out << "uc_options.order=(fabs(a[0]-0.5)<1e-14 ? 4 : 1);\n";
    }
    out << "/*UC_ADAPTIVE_RUN*/\n"
        << "adaptive::Solver<AMOUNTOFX,UcAdaptiveModel> uc_solver(X,a,uc_options);\n"
        << "if(!uc_solver.integrate(h,X)) for(int i=0;i<AMOUNTOFX;++i) X[i]=NAN;\n";
    return out.str();
}
std::string adaptive_phase_loop(const std::string& body) {
    return "const numb* a = values;\n" + adaptive_model_prefix(body) + R"UC(
adaptive::Solver<AMOUNTOFX,UcAdaptiveModel> uc_solver(X,values,uc_options);
const double bound = ((double)skip + total - 1) * h;
if (total <= 0 || skip < 0 || !adaptive::finite(h) || h <= 0 || !adaptive::finite(bound))
    uc_solver.stats.status = adaptive::InvalidOptions;
for(int sample=0;sample<total;++sample) {
    if(uc_solver.stats.status != adaptive::Success ||
       !uc_solver.sample(((double)skip+sample)*h,bound,out+(size_t)sample*AMOUNTOFX)) {
        for(int rest=sample;rest<total;++rest)
            for(int i=0;i<AMOUNTOFX;++i) out[(size_t)rest*AMOUNTOFX+i]=NAN;
        break;
    }
}
*stats = uc_solver.stats;
)UC";
}


std::string adaptive_ensemble_function(const std::string& body, bool cpu) {
    return std::string(cpu ? "extern \"C\" UC_EXPORT " : "UC_AD_HD ") +
        "bool uc_adaptive_ensemble(numb* X,numb* probes,int count,const numb* a,numb h,int steps,numb max_value) {\n" +
        adaptive_model_prefix(body) +
        "if(count==1) return adaptive::integrate_ensemble<AMOUNTOFX,2,UcAdaptiveModel>(X,probes,a,h*steps,max_value,uc_options);\n"
        "if(count==AMOUNTOFX) return adaptive::integrate_ensemble<AMOUNTOFX,AMOUNTOFX+1,UcAdaptiveModel>(X,probes,a,h*steps,max_value,uc_options);\n"
        "return false;\n}\n";
}

std::string adaptive_sync_function(const std::string& body) {
    return std::string("UC_AD_HD void uc_adaptive_sync(numb* X,const numb* master,const numb* gain,const numb* a,numb h,bool forward) {\n") +
        adaptive_model_prefix(body) +
        "if(uc_options.method==2) uc_options.order=1;\n"
        "adaptive::SyncModel<AMOUNTOFX,UcAdaptiveModel> model{UcAdaptiveModel{},master,gain,forward?1.:-1.};\n"
        "adaptive::Solver<AMOUNTOFX,decltype(model)> solver(X,a,uc_options,model);\n"
        "if(!solver.integrate(h,X)) for(int i=0;i<AMOUNTOFX;++i) X[i]=NAN;\n}\n";
}
