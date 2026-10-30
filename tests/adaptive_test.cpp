#include "ucuda/cuda/parametric_engine.h"
#include "ucuda/domain/adaptive_solver.h"
#include "ucuda/equations/adaptive_codegen.h"
#include "ucuda/numerics/krs_cpu.h"
#include "ucuda/cuda/nvrtc_engine.h"
#include "ucuda/platform/platform_paths.h"
#include "ucuda/storage/session_io.h"
#include <cuda.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

std::string exe_dir() { return platform::executable_dir(); }
static void require(bool pass, const char* message) { if (!pass) throw std::runtime_error(message); }
struct Decay {
    void rhs(const double* x, const double*, double* f) const { f[0] = -x[0]; }
    void step(double* x, const double*, double h) const { x[0] *= 1-h; }
};
struct Oscillator {
    void rhs(const double* x, const double*, double* f) const { f[0]=x[1]; f[1]=-x[0]; }
    void step(double*, const double*, double) const {}
};

static void numerical_tests() {
    adaptive::Options o; o.method=1; o.h_initial=1; o.h_max=1; o.rtol=1e-9; o.atol=1e-12;
    double initial[1]={1}, a[1]={0.5}, out[1];
    adaptive::Solver<1,Decay> solver(initial,a,o);
    double max_error=0;
    for(int i=0;i<=1000;++i) {
        require(solver.sample(i*0.01,10,out), "DOPRI dense sample failed");
        max_error=std::max(max_error,std::abs(out[0]-std::exp(-i*0.01)));
    }
    require(max_error<1e-8,"DOPRI dense output inaccurate");
    require(solver.stats.accepted<250,"dense output forced a step per sample");
    require(solver.stats.rejected>0,"test did not exercise rejection");
    adaptive::Solver<1,Decay> backward(initial,a,o);
    require(backward.integrate(-1,out) && std::abs(out[0]-std::exp(1.))<2e-8,"backward integration failed");
    double osc[2]={1,0}, result[2];
    adaptive::Solver<2,Oscillator> harmonic(osc,a,o);
    require(harmonic.integrate(20,result),"oscillator failed");
    require(std::abs(result[0]-std::cos(20.))<2e-8 && std::abs(result[1]+std::sin(20.))<2e-8,"oscillator phase error");
    double errors[2];
    for(int i=0;i<2;++i) {
        o.rtol=i ? 1e-8 : 1e-3;
        adaptive::Solver<1,Decay> accuracy(initial,a,o);
        require(accuracy.integrate(2,out),"accuracy run failed");
        errors[i]=std::abs(out[0]-std::exp(-2.));
    }
    require(errors[1]<errors[0]*0.01,"tightening tolerance did not improve accuracy");
    o.method=2; o.order=1; o.rtol=1e-5;
    adaptive::Solver<1,Decay> doubling(initial,a,o);
    require(doubling.integrate(1,out) && std::abs(out[0]-std::exp(-1.))<0.002,"step doubling failed");
    o.method=1; o.rtol=1e-14; o.atol=1e-15; o.h_min=1;
    adaptive::Solver<1,Decay> underflow(initial,a,o);
    require(!underflow.integrate(1,out) && underflow.stats.status==adaptive::StepUnderflow,"h_min silently accepted bad step");
    o.h_min=1e-12; o.max_steps=1;
    adaptive::Solver<1,Decay> budget(initial,a,o);
    require(!budget.integrate(10,out) && budget.stats.status==adaptive::StepLimit,"step limit ignored");
    o.max_steps=100; o.max_rejects=1;
    adaptive::Solver<1,Decay> rejects(initial,a,o);
    require(!rejects.integrate(1,out) && rejects.stats.status==adaptive::RejectLimit,"reject limit ignored");
    o.rtol=std::numeric_limits<double>::quiet_NaN();
    adaptive::Solver<1,Decay> invalid(initial,a,o);
    require(!invalid.integrate(1,out) && invalid.stats.status==adaptive::InvalidOptions,"NaN settings accepted");
}

static void session_tests() {
    PhaseAnalysisSession a,b; a.adaptive.method=2; a.adaptive.order=4;
    a.adaptive.rtol=2e-9; a.adaptive.h_max=0.42;
    require(session_from_json(session_to_json(a),b),"phase JSON failed");
    require(b.adaptive.method==2 && b.adaptive.order==4 && b.adaptive.rtol==2e-9 && b.adaptive.h_max==.42,"adaptive JSON lost precision");
    require(session_from_json("{\"step_h\":\"0.1\"}",b) && b.adaptive.method==0,"old session retained adaptive settings");
    BifurcationAnalysisSession bif,bif2; bif.diagrams.emplace_back(); bif.diagrams[0].adaptive=a.adaptive;
    require(session_from_json_parametric(session_to_json_parametric(bif),bif2),"bifurcation JSON failed");
    require(bif2.diagrams[0].adaptive.method==2,"bifurcation adaptive mode lost");
}

static void generated_tests(bool gpu) {
    System sys; sys.vars={"x"}; sys.rhs={"-x"};
    adaptive::Options o; o.method=1; o.rtol=1e-9; o.h_max=.5;
    auto body=codegen_adaptive(sys,"RK4",codegen_scheme(sys,Scheme::RK4),false,o);
    KrsCpuStep cpu; std::vector<KrsCpuDiag> diags;
    if(!cpu.compile(body,1,1,diags)) {
        for(const auto& d:diags) std::cerr<<d.message<<'\n';
        throw std::runtime_error("adaptive native compilation failed");
    }
    require(cpu.trajectory_fn()!=nullptr,"missing native adaptive trajectory");
    double initial[1]={1}, params[1]={.5};
    std::vector<double> out(101); adaptive::Stats stats;
    cpu.trajectory_fn()(initial,params,.01,101,10,out.data(),&stats);
    require(stats.status==0 && std::abs(out.back()-std::exp(-1.1))<1e-8,"native dense trajectory incorrect");
    if(gpu) {
        NvrtcEngine engine;
        if(!engine.compile(body,1)) throw std::runtime_error(engine.error());
        std::vector<std::vector<std::vector<double>>> result;
        std::vector<adaptive::Stats> gs;
        if(!engine.run_phase_portraits({1,2},2,{.5},.01,101,10,result,&gs)) throw std::runtime_error(engine.error());
        require(gs.size()==2 && gs[0].accepted>0,"GPU statistics missing");
        for(int i=0;i<101;++i) require(std::abs(result[0][i][0]-out[i])<1e-10,"CPU/GPU dense output differs");
        require(std::abs(result[1].back()[0]-2*std::exp(-1.1))<2e-8,"GPU ensemble trajectory incorrect");
        o.method=2; o.order=0;
        auto doubled=codegen_adaptive(sys,"RK4",codegen_scheme(sys,Scheme::RK4),false,o);
        if(!engine.compile(doubled,1) || !engine.run_phase_portraits({1},1,{.5},.01,101,0,result,&gs)) throw std::runtime_error(engine.error());
        require(std::abs(result[0].back()[0]-std::exp(-1.))<1e-8,"GPU doubling incorrect");
        Bifurcation1DRequest req; req.krs_body=body; req.amountOfX=1; req.initial_conditions={1};
        req.base_values={.5}; req.n_pts=2; req.h=.01; req.t_max=1;
        ParametricEngine sweep;
        auto br=sweep.run_bifurcation_1d(req);
        if(!br.ok) throw std::runtime_error(br.error);

        LLE1DRequest lle; lle.krs_body=body; lle.amountOfX=1; lle.initial_conditions={1};
        lle.base_values={.5}; lle.n_pts=2; lle.h=.01; lle.t_max=2; lle.NT=.1; lle.eps=1e-4;
        auto lr=sweep.run_lle_1d(lle);
        if(!lr.ok) throw std::runtime_error(lr.error);
        require(lr.lyapunov.size()==2 && std::abs(lr.lyapunov[0]+1)<1e-4,"adaptive GPU LLE != -1");
        lle.use_cpu=true;
        auto lc=sweep.run_lle_1d(lle);
        if(!lc.ok) throw std::runtime_error(lc.error);
        require(std::abs(lc.lyapunov[0]-lr.lyapunov[0])<1e-5,"adaptive CPU/GPU LLE differs");
        lle.use_cpu=false; lle.continuation=true;
        auto lcont=sweep.run_lle_1d(lle);
        if(!lcont.ok) throw std::runtime_error(lcont.error);
        require(std::abs(lcont.lyapunov[0]+1)<1e-4,"adaptive LLE continuation incorrect");

        System diagonal; diagonal.vars={"x","y"}; diagonal.rhs={"-x","-2*y"};
        o.method=1;
        auto db=codegen_adaptive(diagonal,"RK4",codegen_scheme(diagonal,Scheme::RK4),false,o);
        LS1DRequest ls; ls.krs_body=db; ls.amountOfX=2; ls.initial_conditions={1,1};
        ls.base_values={.5}; ls.n_pts=2; ls.h=.01; ls.t_max=10; ls.NT=.1; ls.eps=1e-4;
        auto sr=sweep.run_ls_1d(ls);
        if(!sr.ok) throw std::runtime_error(sr.error);
        require(sr.spectrum.size()==2 && std::abs(sr.spectrum[0][0]+sr.spectrum[0][1]+3)<1e-4,"adaptive LS trace != -3");
        ls.use_cpu=true;
        auto sc=sweep.run_ls_1d(ls);
        if(!sc.ok) throw std::runtime_error(sc.error);
        require(std::abs(sc.spectrum[0][0]-sr.spectrum[0][0])<1e-4,"adaptive CPU/GPU LS differs");
        ls.use_cpu=false; ls.continuation=true;
        auto scont=sweep.run_ls_1d(ls);
        if(!scont.ok) throw std::runtime_error(scont.error);
        require(std::abs(scont.spectrum[0][0]+scont.spectrum[0][1]+3)<1e-4,"adaptive LS continuation incorrect");

        Dft1DRequest dft; dft.krs_body=body; dft.amountOfX=1; dft.initial_conditions={1};
        dft.base_values={.5}; dft.n_pts=2; dft.h=.01; dft.t_max=1; dft.n_freq=8;
        auto dr=sweep.run_dft_1d(dft); if(!dr.ok) throw std::runtime_error(dr.error);
        BasinsRequest basins; basins.krs_body=db; basins.amountOfX=2;
        basins.initial_conditions={1,1}; basins.base_values={.5}; basins.n_pts=2;
        basins.h=.01; basins.t_max=1; basins.transient_time=0;
        auto basin=sweep.run_basins(basins); if(!basin.ok) throw std::runtime_error(basin.error);
        FastSyncRequest sync; sync.krs_body=db; sync.amountOfX=2; sync.values={.5};
        sync.ic_master={1,1}; sync.ic_slave={1.01,1.01}; sync.k_forward={1,1}; sync.k_backward={1,1};
        sync.h=.01; sync.t_max=.1; sync.window=.05; sync.iter_of_synchr=2; sync.transient_time=0;
        auto synced=sweep.run_fastsync(sync); if(!synced.ok) throw std::runtime_error(synced.error);

        Bifurcation2DRequest b2; b2.krs_body=db; b2.amountOfX=2; b2.initial_conditions={1,1};
        b2.base_values={.5}; b2.n_pts=2; b2.sweep_over_var=b2.sweep_over_var_2=true;
        b2.var_sweep_index=0; b2.var_sweep_index_2=1; b2.h=.01; b2.t_max=1;
        auto b2r=sweep.run_bifurcation_2d(b2); if(!b2r.ok) throw std::runtime_error(b2r.error);
        LLE2DRequest l2; l2.krs_body=db; l2.amountOfX=2; l2.initial_conditions={1,1};
        l2.base_values={.5}; l2.n_pts=2; l2.sweep_over_var=l2.sweep_over_var_2=true;
        l2.var_sweep_index=0; l2.var_sweep_index_2=1; l2.h=.01; l2.t_max=1; l2.NT=.1;
        auto l2r=sweep.run_lle_2d(l2); if(!l2r.ok) throw std::runtime_error(l2r.error);
        LS2DRequest s2; s2.krs_body=db; s2.amountOfX=2; s2.initial_conditions={1,1};
        s2.base_values={.5}; s2.n_pts=2; s2.sweep_over_var=s2.sweep_over_var_2=true;
        s2.var_sweep_index=0; s2.var_sweep_index_2=1; s2.h=.01; s2.t_max=1; s2.NT=.1;
        auto s2r=sweep.run_ls_2d(s2); if(!s2r.ok) throw std::runtime_error(s2r.error);

        // Existing complex/composition schemes still compile inside the adapter.
        for (const auto& scheme : {std::string("Complex CD"),std::string("Complex CD4"),std::string("Complex Implicit Euler"),std::string("SEMP"),std::string("SIMP")}) {
            o.method=2; o.order=0;
            auto cb=codegen_adaptive(diagonal,scheme,codegen_scheme(diagonal,scheme_from_name(scheme)),false,o);
            if(!engine.compile(cb,2) || !engine.run_phase_portraits({1,1},1,{.5},.01,101,0,result)) throw std::runtime_error(engine.error());
            require(std::abs(result[0].back()[0]-std::exp(-1.))<2e-6,"adaptive complex/composite scheme inaccurate");
        }
        o.method=1; o.h_initial=o.h_min=o.h_max=1; o.rtol=1e-14; o.atol=1e-15;
        auto failing=codegen_adaptive(sys,"RK4",codegen_scheme(sys,Scheme::RK4),false,o);
        require(engine.compile(failing,1),"failure-test compilation failed");
        require(!engine.run_phase_portraits({1},1,{.5},1,2,0,result,&gs),"GPU accepted tolerance violation");
        require(gs[0].status==adaptive::StepUnderflow,"GPU lost failure reason");

        System lorenz; lorenz.vars={"x","y","z"}; lorenz.params={"sigma","rho","beta"};
        lorenz.rhs={"sigma*(y-x)","x*(rho-z)-y","x*y-beta*z"};
        o=adaptive::Options{}; o.method=1; o.rtol=1e-10; o.atol=1e-12;
        auto lb=codegen_adaptive(lorenz,"RK4",codegen_scheme(lorenz,Scheme::RK4),false,o);
        if(!engine.compile(lb,3) || !engine.run_phase_portraits({1,1,1},1,{.5,10,28,8./3},.01,501,0,result)) throw std::runtime_error(engine.error());
        KrsCpuStep reference;
        require(reference.compile(codegen_scheme(lorenz,Scheme::RK4),3,4,diags),"Lorenz reference compilation failed");
        double ref[3]={1,1,1}, lp[4]={.5,10,28,8./3};
        for(int i=0;i<50000;++i) reference.fn()(ref,lp,1e-4);
        for(int i=0;i<3;++i) require(std::abs(result[0].back()[i]-ref[i])<1e-5,"adaptive Lorenz differs from fine RK4 reference");
        rqa::Config rq; rq.source=rqa::Source::StateVector; rq.points=128;
        rqa::Result rr;
        require(rqa::compute(result[0],.01,rq,rr),"RQA failed on adaptive uniform output");
    }
    bool custom_rejected=false;
    try { o.method=2; o.order=0; codegen_adaptive(sys,"custom","X[0]-=h*X[0];",true,o); }
    catch(...) { custom_rejected=true; }
    require(custom_rejected,"custom scheme order guessed silently");
}
int main(int argc,char** argv) {
    try {
        const bool gpu=argc>1 && std::string(argv[1])=="--gpu";
        if(gpu && cuInit(0)!=CUDA_SUCCESS) return 77;
        numerical_tests(); session_tests(); generated_tests(gpu);
        std::cout<<"Adaptive numerical, serialization and "<<(gpu?"CPU/GPU":"CPU")<<" checks passed\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
