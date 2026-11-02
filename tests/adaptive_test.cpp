#include "ucuda/cuda/parametric_engine.h"
#include "ucuda/cuda/nvrtc_engine.h"
#include "ucuda/cuda/runtime_settings.h"
#include "ucuda/cuda/rqa.h"
#include "ucuda/equations/codegen.hpp"
#include "ucuda/equations/adaptive_params.h"
#include "ucuda/numerics/integrator.h"
#include "ucuda/numerics/krs_cpu.h"
#include "ucuda/platform/platform_paths.h"
#include "ucuda/application/session_persistence.h"
#include "ucuda/storage/data_export.h"
#include <cuda.h>
#ifdef UCUDA_TEST_ARTIFACTS
#include "support/artifacts.h"
#endif
#include <cmath>
#include <cstdio>
#include <string>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <thread>
#include <chrono>
#if defined(_WIN32) && !defined(NDEBUG)
#include <crtdbg.h>
#endif

std::string exe_dir() { return platform::executable_dir(); }
static void require(bool pass, const char* message) {
    if (!pass) throw std::runtime_error(message);
}
static void compile(KrsCpuStep& step, const std::string& body, int nx, int nv) {
    std::vector<KrsCpuDiag> diags;
    if (!step.compile(body,nx,nv+kSchemeTailSlots,diags)) {
        for (const auto& d:diags) std::cerr << d.message << '\n';
        throw std::runtime_error("fixed-step native compilation failed");
    }
}

template<class Session, class Save, class Load>
static void legacy_round_trip(const Session& initial, Save save, Load load) {
    // Readers may add an empty default projection to freshly constructed configurations.
    // Compare obsolete-field loading with the same fixture loaded without those fields.
    Session baseline;
    require(load(save(initial),baseline),"fixed-step fixture failed to load");
    const std::string canonical = save(baseline);
    require(canonical.find("\"adaptive\"")==std::string::npos,"new JSON includes adaptive options");
    for (int method : {1,2}) {
        // Add obsolete options to every configuration, including shared/nested Custom sessions.
        std::string legacy = canonical;
        const std::string key = "\"scheme\":";
        size_t pos=0, count=0;
        const std::string obsolete = "\"adaptive\":{\"method\":"+std::to_string(method)+
            ",\"rtol\":1e-99,\"h_min\":42,\"h_max\":0.001,\"unknown\":[1,{\"x\":2}]},";
        while ((pos=legacy.find(key,pos))!=std::string::npos) {
            legacy.insert(pos,obsolete); pos+=obsolete.size()+key.size(); ++count;
        }
        require(count>0,"legacy fixture has no analysis configuration");
        Session loaded;
        require(load(legacy,loaded),"old adaptive session failed to load");
        if (save(loaded)!=canonical) {
            const auto actual=save(loaded);
            size_t i=0; while(i<actual.size() && i<canonical.size() && actual[i]==canonical[i]) ++i;
            std::cerr << "JSON difference at " << i << ": expected " << canonical.substr(i,160)
                      << "\nactual " << actual.substr(i,160) << '\n';
            throw std::runtime_error("ignored adaptive fields changed saved settings");
        }
        Session again;
        require(load(save(loaded),again) && save(again)==canonical,"fixed-step round trip failed");
    }
    Session loaded;
    require(load(canonical,loaded) && save(loaded)==canonical,"old fixed-step session failed to load");
}

static void session_tests() {
    PhaseAnalysisSession phase; phase.step_h="0.125"; phase.scheme="Euler"; phase.decimation="3";
    legacy_round_trip(phase,session_to_json,session_from_json);
    BifurcationAnalysisSession bif; bif.diagrams.emplace_back();
    legacy_round_trip(bif,session_to_json_parametric,session_from_json_parametric);
    LLEAnalysisSession lle; lle.curves.emplace_back();
    legacy_round_trip(lle,session_to_json_lle,session_from_json_lle);
    LyapunovSpectrumAnalysisSession ls; ls.curves.emplace_back();
    legacy_round_trip(ls,session_to_json_ls,session_from_json_ls);
    Dft1DAnalysisSession dft; dft.configs.emplace_back();
    legacy_round_trip(dft,session_to_json_dft1d,session_from_json_dft1d);
    BasinsAnalysisSession basins; basins.configs.emplace_back();
    legacy_round_trip(basins,session_to_json_basins,session_from_json_basins);
    FastSyncAnalysisSession sync; sync.configs.emplace_back();
    legacy_round_trip(sync,session_to_json_fastsync,session_from_json_fastsync);
    CustomSession custom; custom.shared.h_text="0.125"; custom.shared.scheme="Euler";
    custom.bif_session.diagrams.emplace_back(); custom.lle_session.curves.emplace_back();
    custom.ls_session.curves.emplace_back(); custom.basins_session.configs.emplace_back();
    legacy_round_trip(custom,session_to_json_custom,session_from_json_custom);
}

static void numerical_tests(bool gpu) {
    System sys; sys.vars={"x","y"}; sys.rhs={"-x","-2*y"};
    const double initial[]={1,1}, params[]={0.5,4,1e-12,25,0};
    SystemEvaluator evaluator(sys);
    for (const auto& name : {"Euler","RK4","DOPRI78","Complex CD","Complex CD4","Complex Implicit Euler","SEMP","SIMP"}) {
        const auto body=compute_krs_for_scheme({},sys,name);
        if (body.empty()) throw std::runtime_error(std::string("fixed-step generation failed for ")+name);
        KrsCpuStep step; compile(step,body,2,1);
        std::vector<std::vector<double>> native, interpreted;
        require(computePhasePortraitCPU_custom(step.fn(),initial,2,params,5,.001,101,10,native),"native trajectory failed");
        require(computePhasePortraitCPU(evaluator,int_scheme_from_string(name),initial,2,params,5,.001,101,10,interpreted),"interpreted trajectory failed");
        require(native.size()==101,"sample count changed");
        for(size_t i=0;i<native.size();++i)
            for (int k=0;k<2;++k)
                require(std::abs(native[i][k]-interpreted[i][k])<1e-11,"native and interpreted fixed steps differ");
        require(std::abs(native.back()[0]-std::exp(-.110))<1e-4,"fixed-step decay inaccurate");
        if (gpu) {
            NvrtcEngine engine;
            require(engine.compile(body,2),engine.error().c_str());
            std::vector<std::vector<std::vector<double>>> result;
            require(engine.run_phase_portraits({1,1,2,2},2,{.5,4,1e-12,25,0},.001,101,10,result),engine.error().c_str());
            require(result.size()==2 && result[0].size()==101,"GPU trajectory dimensions changed");
            for(size_t i=0;i<native.size();++i)
                require(std::abs(result[0][i][0]-native[i][0])<1e-11,"CPU/GPU fixed steps differ");
            require(std::abs(result[1].back()[0]-2*native.back()[0])<1e-11,"second GPU trajectory differs");
        }
    }
    // An exact linear custom step detects subdivision, hidden scheme substitution and off-by-one samples.
    KrsCpuStep step; compile(step,"X[0] += h;",1,1);
    std::vector<std::vector<double>> out;
    require(computePhasePortraitCPU_custom(step.fn(),initial,1,params,5,.125,9,3,out),"custom trajectory failed");
    for(int i=0;i<9;++i) require(out[i][0]==1+(3+i)*.125,"custom step/sample ordering changed");
    if(gpu) {
        NvrtcEngine engine; require(engine.compile("X[0] += h;",1),engine.error().c_str());
        std::vector<std::vector<std::vector<double>>> result;
        require(engine.run_phase_portraits({1},1,{.5,4,1e-12,25,0},.125,9,3,result),engine.error().c_str());
        require(result[0]==out,"custom CPU/GPU sample ordering differs");
    }
    // Move ownership and cached custom fixed steps remain valid.
    KrsCpuStep moved(std::move(step));
    require(moved.fn() && !step.fn(),"native step move constructor lost ownership");
    step=std::move(moved);
    require(step.fn() && !moved.fn(),"native step move assignment lost ownership");
}

static void phase_session_tests(bool gpu) {
    PhaseAnalysisSession s;
    s.sys.vars={"x"}; s.sys.rhs={"-x"}; s.vars=s.sys.vars;
    CustomScheme custom; custom.name="linear"; custom.body="X[0] += h;";
    s.custom_schemes.push_back(custom); s.scheme="linear";
    s.step_h="0.125"; s.sim_time="2"; s.skip_time="0.375";
    s.decimation="3"; s.use_gpu=gpu;
    s.ic_sets.clear(); s.ic_sets.emplace_back(); s.ic_sets[0].values["x"]="1";
    s.recompute();
    require(s.result.ok,s.result.error.c_str());
    const auto& result=s.result;
    require(result.trajectories.size()==1 && result.trajectories[0].size()==6,"phase decimation count changed");
    for(size_t i=0;i<6;++i) require(result.trajectories[0][i][0]==1+(3+3*i)*.125,"phase decimation/sample time changed");
    require(result.snapshot.h==.125 && result.snapshot.decimator==3,"phase snapshot step changed");
    const auto temp=std::filesystem::temp_directory_path()/("ucuda-fixed-step-export-"+std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count())+".csv");
    struct Cleanup { std::filesystem::path path; ~Cleanup() {
        std::error_code ec; std::filesystem::remove(path,ec);
        auto config=std::filesystem::path(path.string()+"_config.csv"); std::filesystem::remove(config,ec);
    } } cleanup{temp};
    require(data_export::export_phase(result,result.snapshot,temp.string()),"phase export failed");
    std::ifstream f(temp);
    const std::string csv{std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
    require(csv.find("adaptive")==std::string::npos,"phase CSV includes adaptive metadata");
    require(csv.find("t, x\n0, 1.375\n0.375, 1.75\n")==0,"phase CSV sample times or columns changed");
    std::ifstream cfg(temp.string()+"_config.csv");
    const std::string metadata{std::istreambuf_iterator<char>(cfg),std::istreambuf_iterator<char>()};
    require(metadata.find("adaptive")==std::string::npos && metadata.find("rtol")==std::string::npos,
            "phase config CSV includes adaptive metadata");
}

static void cpu_analysis_tests() {
    System sys; sys.vars={"x"}; sys.rhs={"-x"};
    const auto body=codegen_scheme(sys,Scheme::RK4);
    ParametricEngine engine;
    Bifurcation1DRequest bif; bif.krs_body=body; bif.amountOfX=1; bif.initial_conditions={1};
    bif.base_values={.5,4,1e-12,25,0}; bif.n_pts=2; bif.h=.01; bif.t_max=1; bif.use_cpu=true;
    auto br=engine.run_bifurcation_1d(bif); require(br.ok,br.error.c_str());
    Dft1DRequest dft; dft.krs_body=body; dft.amountOfX=1; dft.initial_conditions={1};
    dft.base_values={.5,4,1e-12,25,0}; dft.n_pts=2; dft.h=.01; dft.t_max=1; dft.n_freq=8; dft.use_cpu=true; dft.pre_scaller=3;
    auto dr=engine.run_dft_1d(dft); require(dr.ok,dr.error.c_str());
    // Upstream subtracts the fixed-point check from each normalization block: exactly NT/h steps.
    LLE1DRequest lle; lle.krs_body=body; lle.amountOfX=1; lle.initial_conditions={1};
    lle.base_values={.5,4,1e-12,25,0}; lle.n_pts=2; lle.h=.01; lle.t_max=2; lle.NT=.1; lle.eps=1e-4; lle.use_cpu=true;
    for (bool continuation : {false,true}) {
        lle.continuation=continuation;
        auto r=engine.run_lle_1d(lle); require(r.ok,r.error.c_str());
        require(r.lyapunov.size()==2 && std::abs(r.lyapunov[0]+1.0)<1e-4,"fixed-step CPU LLE normalization interval changed");
    }
    System diagonal; diagonal.vars={"x","y"}; diagonal.rhs={"-x","-2*y"};
    LS1DRequest ls; ls.krs_body=codegen_scheme(diagonal,Scheme::RK4); ls.amountOfX=2;
    ls.initial_conditions={1,1}; ls.base_values={.5,4,1e-12,25,0}; ls.n_pts=2; ls.h=.01;
    ls.t_max=10; ls.NT=.1; ls.eps=1e-4; ls.use_cpu=true;
    for (bool continuation : {false,true}) {
        ls.continuation=continuation;
        auto r=engine.run_ls_1d(ls); require(r.ok,r.error.c_str());
        require(r.spectrum.size()==2 && std::abs(r.spectrum[0][0]+r.spectrum[0][1]+3.0)<1e-4,"fixed-step CPU LS normalization interval changed");
    }
}


static void sampling_sweep_tests(bool gpu) {
    Dft1DRequest req;
    req.krs_body="X[0] += h;"; req.amountOfX=1; req.initial_conditions={1}; req.base_values={.5,4,1e-12,25,0};
    req.n_pts=2; req.h=.125; req.transient_time=.375; req.t_max=3; req.pre_scaller=3;
    req.n_freq=3; req.freq_lo=0; req.freq_hi=1; req.window_type=0; req.use_cpu=!gpu;
    ParametricEngine engine;
    for (bool sweep : {false,true}) {
        req.sweep_over_h=sweep; req.param_lo=.0625; req.param_hi=.125;
        const auto result=engine.run_dft_1d(req);
        require(result.ok,result.error.c_str());
        for(int point=0;point<2;++point) {
            const double h=sweep ? (point==0?.0625:.125) : req.h;
            const int samples=(int)(req.t_max/h/req.pre_scaller);
            // Preserve the legacy extra fixed-point check after the transient loop.
            // Classical GPU DFT passes preScaller to the transient check; CPU/h-sweep pass 1.
            const int skip=(int)(req.transient_time/h)+(gpu && !sweep ? req.pre_scaller : 1);
            require(result.flags[point]==REGIME_OSCILLATION,"linear DFT trajectory was rejected");
            for(int k=0;k<req.n_freq;++k) {
                const double freq=.5*k;
                double cosine=0,sine=0;
                for(int i=0;i<samples;++i) {
                    const double x=1+(skip+i*req.pre_scaller)*h;
                    const double angle=2*3.14159265358979323846*freq*i*h*req.pre_scaller;
                    cosine+=x*std::cos(angle); sine+=x*std::sin(angle);
                }
                const size_t j=(size_t)point*req.n_freq+k;
                if (std::abs(result.ak_cos[j]-cosine/samples)>=1e-11) {
                    std::cerr << "DFT gpu=" << gpu << " sweep=" << sweep << " point=" << point
                              << " freq=" << freq << " actual=" << result.ak_cos[j] << " expected=" << cosine/samples << '\n';
                    throw std::runtime_error("DFT fixed step/preScaller sampling changed");
                }
                require(std::abs(result.bk_sin[j]-sine/samples)<1e-11,"DFT frequency sample spacing changed");
            }
        }
    }
}

static void gpu_analysis_tests() {
    System sys; sys.vars={"x"}; sys.rhs={"-x"};
    const auto body=codegen_scheme(sys,Scheme::RK4);
        Bifurcation1DRequest req; req.krs_body=body; req.amountOfX=1; req.initial_conditions={1};
        req.base_values={.5,4,1e-12,25,0}; req.n_pts=2; req.h=.01; req.t_max=1;
        ParametricEngine sweep;
        auto br=sweep.run_bifurcation_1d(req);
        if(!br.ok) throw std::runtime_error(br.error);

        LLE1DRequest lle; lle.krs_body=body; lle.amountOfX=1; lle.initial_conditions={1};
        lle.base_values={.5,4,1e-12,25,0}; lle.n_pts=2; lle.h=.01; lle.t_max=2; lle.NT=.1; lle.eps=1e-4;
        auto lr=sweep.run_lle_1d(lle);
        if(!lr.ok) throw std::runtime_error(lr.error);
        require(lr.lyapunov.size()==2 && std::abs(lr.lyapunov[0]+1.0)<1e-4,"fixed-step GPU LLE normalization interval changed");
        lle.use_cpu=true;
        auto lc=sweep.run_lle_1d(lle);
        if(!lc.ok) throw std::runtime_error(lc.error);
        require(std::abs(lc.lyapunov[0]-lr.lyapunov[0])<1e-5,"fixed-step CPU/GPU LLE differs");
        lle.use_cpu=false; lle.continuation=true;
        auto lcont=sweep.run_lle_1d(lle);
        if(!lcont.ok) throw std::runtime_error(lcont.error);
        require(std::abs(lcont.lyapunov[0]+1.0)<1e-4,"fixed-step LLE continuation incorrect");

        System diagonal; diagonal.vars={"x","y"}; diagonal.rhs={"-x","-2*y"};
        auto db=codegen_scheme(diagonal,Scheme::RK4);
        LS1DRequest ls; ls.krs_body=db; ls.amountOfX=2; ls.initial_conditions={1,1};
        ls.base_values={.5,4,1e-12,25,0}; ls.n_pts=2; ls.h=.01; ls.t_max=10; ls.NT=.1; ls.eps=1e-4;
        auto sr=sweep.run_ls_1d(ls);
        if(!sr.ok) throw std::runtime_error(sr.error);
        require(sr.spectrum.size()==2 && std::abs(sr.spectrum[0][0]+sr.spectrum[0][1]+3.0)<1e-4,"fixed-step LS normalization interval changed");
        ls.use_cpu=true;
        auto sc=sweep.run_ls_1d(ls);
        if(!sc.ok) throw std::runtime_error(sc.error);
        require(std::abs(sc.spectrum[0][0]-sr.spectrum[0][0])<1e-4,"fixed-step CPU/GPU LS differs");
        ls.use_cpu=false; ls.continuation=true;
        auto scont=sweep.run_ls_1d(ls);
        if(!scont.ok) throw std::runtime_error(scont.error);
        require(std::abs(scont.spectrum[0][0]+scont.spectrum[0][1]+3.0)<1e-4,"fixed-step LS continuation incorrect");

        Dft1DRequest dft; dft.krs_body=body; dft.amountOfX=1; dft.initial_conditions={1};
        dft.base_values={.5,4,1e-12,25,0}; dft.n_pts=2; dft.h=.01; dft.t_max=1; dft.n_freq=8;
        auto dr=sweep.run_dft_1d(dft); if(!dr.ok) throw std::runtime_error(dr.error);
        BasinsRequest basins; basins.krs_body=db; basins.amountOfX=2;
        basins.initial_conditions={1,1}; basins.base_values={.5,4,1e-12,25,0}; basins.n_pts=2;
        basins.h=.01; basins.t_max=1; basins.transient_time=0;
        auto basin=sweep.run_basins(basins); if(!basin.ok) throw std::runtime_error(basin.error);
        FastSyncRequest sync; sync.krs_body=db; sync.amountOfX=2; sync.values={.5,4,1e-12,25,0};
        sync.ic_master={1,1}; sync.ic_slave={1.01,1.01}; sync.k_forward={1,1}; sync.k_backward={1,1};
        sync.h=.01; sync.t_max=.1; sync.window=.05; sync.iter_of_synchr=2; sync.transient_time=0;
        auto synced=sweep.run_fastsync(sync); if(!synced.ok) throw std::runtime_error(synced.error);

        Bifurcation2DRequest b2; b2.krs_body=db; b2.amountOfX=2; b2.initial_conditions={1,1};
        b2.base_values={.5,4,1e-12,25,0}; b2.n_pts=2; b2.sweep_over_var=b2.sweep_over_var_2=true;
        b2.var_sweep_index=0; b2.var_sweep_index_2=1; b2.h=.01; b2.t_max=1;
        auto b2r=sweep.run_bifurcation_2d(b2); if(!b2r.ok) throw std::runtime_error(b2r.error);
        LLE2DRequest l2; l2.krs_body=db; l2.amountOfX=2; l2.initial_conditions={1,1};
        l2.base_values={.5,4,1e-12,25,0}; l2.n_pts=2; l2.sweep_over_var=l2.sweep_over_var_2=true;
        l2.var_sweep_index=0; l2.var_sweep_index_2=1; l2.h=.01; l2.t_max=1; l2.NT=.1;
        auto l2r=sweep.run_lle_2d(l2); if(!l2r.ok) throw std::runtime_error(l2r.error);
        LS2DRequest s2; s2.krs_body=db; s2.amountOfX=2; s2.initial_conditions={1,1};
        s2.base_values={.5,4,1e-12,25,0}; s2.n_pts=2; s2.sweep_over_var=s2.sweep_over_var_2=true;
        s2.var_sweep_index=0; s2.var_sweep_index_2=1; s2.h=.01; s2.t_max=1; s2.NT=.1;
        auto s2r=sweep.run_ls_2d(s2); if(!s2r.ok) throw std::runtime_error(s2r.error);


    // Uniform phase output remains valid input to RQA.
    NvrtcEngine engine; require(engine.compile(body,1),engine.error().c_str());
    std::vector<std::vector<std::vector<double>>> result;
    require(engine.run_phase_portraits({1},1,{.5,4,1e-12,25,0},.01,129,0,result),engine.error().c_str());
    rqa::Config rq; rq.source=rqa::Source::StateVector; rq.points=128;
    rqa::Result rr; require(rqa::compute(result[0],.01,rq,rr),"RQA failed on fixed-step uniform samples");
}


static void upstream_session_tests() {
    std::vector<double> atol;
    std::string tolerance_error;
    require(adaptive_parse_atol("1e-12",3,atol,tolerance_error) && atol==std::vector<double>(3,1e-12),
            "scalar atol expansion failed");
    require(adaptive_parse_atol("1e-10,2e-10",2,atol,tolerance_error) && atol.size()==2 && atol[1]==2e-10,
            "per-variable atol parsing failed");
    AdaptiveSettings settings; settings.enabled=true; settings.raw_nodes=true;
    settings.rtol="2e-9"; settings.atol="1e-11"; settings.ctrl="Hairer numb";
    settings.h0="0.04"; settings.hmax="0.2"; settings.max_rej="16";
    PhaseAnalysisSession phase; phase.scheme="RK45"; phase.adaptive=settings;
    PhaseAnalysisSession loaded;
    require(session_from_json(session_to_json(phase),loaded) && loaded.adaptive==settings,
            "upstream phase adaptive options did not round trip");
    BifurcationAnalysisSession bif; bif.diagrams.emplace_back(); bif.diagrams[0].adaptive=settings;
    BifurcationAnalysisSession bload;
    require(session_from_json_parametric(session_to_json_parametric(bif),bload) && bload.diagrams[0].adaptive==settings,
            "upstream parametric adaptive options did not round trip");
    CustomSession custom; custom.shared.adaptive=settings; custom.phase_session.adaptive=settings;
    custom.bif_session.diagrams.emplace_back(); custom.bif_session.diagrams[0].adaptive=settings;
    CustomSession cload;
    require(session_from_json_custom(session_to_json_custom(custom),cload) && cload.shared.adaptive==settings
            && cload.phase_session.adaptive==settings && cload.bif_session.diagrams[0].adaptive==settings,
            "upstream Custom adaptive options did not round trip");
}

static void upstream_adaptive_tests(bool gpu) {
    System sys; sys.vars={"x","y"}; sys.rhs={"-x","-2*y"};
    const double ic[]={1,1}, values[]={.5,4,1e-12,25,0};
    for (const std::string scheme : {"RK45","DOPRI78","DOP853","GBS 2-4","Extr(RK4|1,2)","Extr(Complex CD|2,4)"}) {
        std::cerr << "Adaptive scheme: " << scheme << std::endl;
        const AdaptiveCode code=adaptive_code_for_scheme({},sys,scheme);
        require(!code.emb.empty() && !code.deval.empty(),"upstream embedded/dense code missing");
        AdaptiveSettings settings; settings.enabled=true; settings.h0="0.15"; if(scheme=="DOP853") settings.ctrl="Hairer numb"; settings.hmax=(scheme=="RK45" || scheme=="DOP853")?"0.25":"0.05";
        UcudaAdaptParams params{}; std::string err,ctrl;
        require(adaptive_build_params(settings,code,2,1.25,params,err,&ctrl),err.c_str());
        AdaptiveCpuModule module; std::vector<KrsCpuDiag> diags;
        if (!module.compile(code.rhs,code.emb,code.dprep,code.deval,ctrl,2,"",diags)) {
            for(const auto& d:diags) std::cerr<<d.message<<'\n';
            throw std::runtime_error("upstream adaptive CPU build failed for "+scheme);
        }
        for(bool raw : {false,true}) {
            constexpr int cap=512,total=101;
            std::vector<double> data(cap*2),times(cap),log(256*4);double stats[9]{},final_h=0;int logs=0;
            const int count=module.phase()(ic,values,&params,.25,1,.01,total,raw,cap,256,
                        data.data(),times.data(),log.data(),&logs,stats,&final_h,nullptr);
            require(count>1 && (raw || count==total),"upstream adaptive sample count incorrect");
            require(stats[0]>0 && stats[3]>0 && stats[4]>0 && stats[5]>=stats[4] && !stats[7],
                    "upstream adaptive step diagnostics incorrect");
            for(int i=0;i<count;++i) {
                const double t=raw?times[i]:.25+.01*i;
                if(std::abs(data[i*2]-std::exp(-t))>=2e-6 || std::abs(data[i*2+1]-std::exp(-2*t))>=2e-6)
                    std::cerr<<scheme<<" raw="<<raw<<" t="<<t<<" x="<<data[i*2]<<" y="<<data[i*2+1]<<" expected y="<<std::exp(-2*t)<<std::endl;
                require(std::abs(data[i*2]-std::exp(-t))<2e-6 && std::abs(data[i*2+1]-std::exp(-2*t))<2e-6,
                        "upstream adaptive dense/raw trajectory is inaccurate");
                if(raw && i) require(times[i]>times[i-1],"adaptive nodes not ordered");
            }
            if(!gpu) continue;
            PhaseAdaptiveRequest req; req.rhs=code.rhs;req.emb=code.emb;req.dprep=code.dprep;req.deval=code.deval;
            req.ctrl_body=ctrl;req.amountOfX=2;req.ic_flat={1,1};req.N=1;req.values={.5,4,1e-12,25,0};req.params=params;
            req.raw=raw;req.t_skip=.25;req.t_rec=1;req.dt=.01;req.total=total;req.max_pts=cap;req.log_cap=256;
            NvrtcEngine engine; PhaseAdaptiveResult result;
            require(engine.run_phase_portraits_adaptive(req,result),engine.error().c_str());
            require(result.traj.size()==1 && result.traj[0].size()>1,"adaptive GPU output empty");
            require(result.stats[0].nacc>0 && result.stats[0].nrhs>0 && !result.stats[0].diverged,
                    "adaptive GPU diagnostics invalid");
            for(size_t i=0;i<result.traj[0].size();++i) {
                const double t=raw?result.times[0][i]:.25+.01*i;
                require(std::abs(result.traj[0][i][0]-std::exp(-t))<2e-6,
                        "upstream adaptive GPU trajectory is inaccurate");
                if(!raw) require(std::abs(result.traj[0][i][0]-data[i*2])<2e-8,"adaptive CPU/GPU dense mismatch");
            }
        }
    }
    // Exercise the complete session path and its decimation on both backends.
    PhaseAnalysisSession phase;phase.vars=sys.vars;phase.sys=sys;phase.scheme="RK45";
    phase.step_h="0.01";phase.sim_time="1";phase.skip_time="0.25";
    phase.decimation="3";phase.use_gpu=gpu;phase.adaptive.enabled=true;
    phase.ic_sets.clear();phase.ic_sets.emplace_back();phase.ic_sets[0].values={{"x","1"},{"y","1"}};
    phase.regenerate_krs();phase.recompute();
    require(phase.result.ok,phase.result.error.c_str());
    require(phase.result.adaptive && phase.result.step_stats.size()==1,"adaptive session lost diagnostics");
    require(phase.result.trajectories[0].size()==34,"adaptive session decimation count changed");
    for(size_t i=0;i<phase.result.trajectories[0].size();++i)
        require(std::abs(phase.result.trajectories[0][i][0]-std::exp(-(.25+.03*i)))<2e-6,
                "adaptive session transient/decimation spacing changed");
}


static AdaptiveRequest make_upstream_request(const System& sys) {
    const auto code=adaptive_code_for_scheme({},sys,"RK45");
    AdaptiveSettings settings;settings.enabled=true;
    AdaptiveRequest ad;ad.enabled=true;ad.rhs=code.rhs;ad.emb=code.emb;ad.dprep=code.dprep;ad.deval=code.deval;
    std::string err;
    if(!adaptive_build_params(settings,code,(int)sys.vars.size(),10,ad.params,err,&ad.ctrl_body))
        throw std::runtime_error(err);
    return ad;
}
static void upstream_sweep_tests(bool gpu) {
    System sys;sys.vars={"x","y"};sys.rhs={"-x","-2*y"};
    const auto ad=make_upstream_request(sys);
    const auto fixed=codegen_scheme(sys,Scheme::RK45);
    System scalar; scalar.vars={"x"}; scalar.rhs={"-x"};
    const auto scalar_ad=make_upstream_request(scalar);
    ParametricEngine engine;
    for(bool cpu : {true,false}) {
        if(!cpu && !gpu) continue;
        for(bool continuation : {false,true}) {
            LLE1DRequest lle;lle.krs_body=codegen_scheme(scalar,Scheme::RK45);lle.amountOfX=1;lle.initial_conditions={1};lle.base_values={.5,4,1e-12,25,0};
            lle.n_pts=2;lle.h=.01;lle.t_max=2;lle.NT=.1;lle.eps=1e-4;lle.transient_time=.25;
            lle.use_cpu=cpu;lle.continuation=continuation;lle.adaptive=scalar_ad;
            auto lr=engine.run_lle_1d(lle);if(!lr.ok) throw std::runtime_error(lr.error);
            require(lr.ad_stats.size()==8 && lr.ad_stats[0]>0,"adaptive LLE lost step statistics");
            require(std::abs(lr.lyapunov[0]+1)<.01,"adaptive LLE normalization inaccurate");
            LS1DRequest ls;ls.krs_body=fixed;ls.amountOfX=2;ls.initial_conditions={1,1};ls.base_values={.5,4,1e-12,25,0};
            ls.n_pts=2;ls.h=.01;ls.t_max=2;ls.NT=.1;ls.eps=1e-4;ls.transient_time=.25;
            ls.use_cpu=cpu;ls.continuation=continuation;ls.adaptive=ad;
            auto sr=engine.run_ls_1d(ls);if(!sr.ok) throw std::runtime_error(sr.error);
            require(sr.ad_stats.size()==8 && std::abs(sr.spectrum[0][0]+sr.spectrum[0][1]+3)<.01,
                    "adaptive LS normalization inaccurate");
            Bifurcation1DRequest bif;bif.krs_body=fixed;bif.amountOfX=2;bif.initial_conditions={1,1};bif.base_values={.5,4,1e-12,25,0};
            bif.n_pts=2;bif.h=.01;bif.t_max=1;bif.transient_time=.25;bif.use_cpu=cpu;
            bif.continuation=continuation;bif.adaptive=ad;
            auto br=engine.run_bifurcation_1d(bif);if(!br.ok) throw std::runtime_error(br.error);
            require(br.ad_stats.size()==8 && br.ad_stats[0]>0,"adaptive bifurcation statistics missing");
            SignalMetricsRequest metrics;metrics.krs_body=fixed;metrics.amountOfX=2;metrics.initial_conditions={1,1};
            metrics.base_values={.5,4,1e-12,25,0};metrics.n_pts=2;metrics.h=.01;metrics.t_max=1;metrics.transient_time=.25;
            metrics.use_cpu=cpu;metrics.continuation=continuation;metrics.adaptive=ad;
            metrics.metric_mask=(1<<SIGM_MEAN)|(1<<SIGM_MIN)|(1<<SIGM_MAX);
            for(bool raw : {false,true}) {
                metrics.adaptive.raw_nodes=raw;
                auto mr=engine.run_signal_metrics(metrics);if(!mr.ok) throw std::runtime_error(mr.error);
                require(mr.ad_stats.size()==8 && mr.ad_stats[0]>0,"adaptive metrics statistics missing");
                require(std::isfinite(mr.values[SIGM_MEAN][0]),"adaptive metrics output invalid");
            }

        }
    }
}

static void upstream_grid_tests() {
    System sys;sys.vars={"x","y"};sys.rhs={"-x","-2*y"};
    const auto ad=make_upstream_request(sys);const auto body=codegen_scheme(sys,Scheme::RK45);
    ParametricEngine engine;
    BasinsRequest basins;basins.krs_body=body;basins.amountOfX=2;basins.initial_conditions={1,1};
    basins.base_values={.5,4,1e-12,25,0};basins.n_pts=2;basins.h=.01;basins.t_max=1;basins.transient_time=.25;basins.adaptive=ad;
    auto br=engine.run_basins(basins);if(!br.ok) throw std::runtime_error(br.error);
    require(br.ad_stats.size()==16 && br.ad_stats[0]>0,"adaptive basins statistics missing");
    Bifurcation2DRequest bif;bif.krs_body=body;bif.amountOfX=2;bif.initial_conditions={1,1};bif.base_values={.5,4,1e-12,25,0};
    bif.n_pts=2;bif.sweep_over_var=bif.sweep_over_var_2=true;bif.var_sweep_index=0;bif.var_sweep_index_2=1;
    bif.h=.01;bif.t_max=1;bif.transient_time=.25;bif.adaptive=ad;
    auto b2=engine.run_bifurcation_2d(bif);if(!b2.ok) throw std::runtime_error(b2.error);
    require(b2.ad_stats.size()==16,"adaptive bifurcation grid statistics missing");
    LLE2DRequest lle;lle.krs_body=body;lle.amountOfX=2;lle.initial_conditions={1,1};lle.base_values={.5,4,1e-12,25,0};
    lle.n_pts=2;lle.sweep_over_var=lle.sweep_over_var_2=true;lle.var_sweep_index=0;lle.var_sweep_index_2=1;
    lle.h=.01;lle.t_max=2;lle.NT=.1;lle.eps=1e-4;lle.adaptive=ad;
    auto lr=engine.run_lle_2d(lle);if(!lr.ok) throw std::runtime_error(lr.error);
    require(lr.ad_stats.size()==16 && lr.values[0]>=-2.01 && lr.values[0]<=-.99,"adaptive LLE grid invalid");
    LS2DRequest ls;ls.krs_body=body;ls.amountOfX=2;ls.initial_conditions={1,1};ls.base_values={.5,4,1e-12,25,0};
    ls.n_pts=2;ls.sweep_over_var=ls.sweep_over_var_2=true;ls.var_sweep_index=0;ls.var_sweep_index_2=1;
    ls.h=.01;ls.t_max=2;ls.NT=.1;ls.eps=1e-4;ls.adaptive=ad;
    auto sr=engine.run_ls_2d(ls);if(!sr.ok) throw std::runtime_error(sr.error);
    require(sr.ad_stats.size()==16 && std::abs(sr.values[0]+sr.values[4]+3)<.01,
            "adaptive LS grid invalid");
}

// ---- Плотный выход DOP853, CPU = GPU, Лоренц, семейства движка ----
//  - CPU-драйвер (computePhasePortraitCPU_adaptive): плотный выход, отказы, осциллятор, допуск;
//  - GPU (run_phase_portraits_adaptive) совпадает с CPU; Лоренц против мелкого RK4;
//  - адаптивные LLE/LS: показатели линейной системы (классика, продолжение, CPU = GPU);
//  - постоянный шаг: БД 2D, LLE/LS 2D, DFT, бассейны, FastSync считаются; комплексные и
//    композиционные схемы — точность на x' = -x.
namespace {

void require(bool pass, const std::string& message) {
    if (!pass) throw std::runtime_error(message);
}

// "name: value vs ref (tol)" — для сообщений о расхождении чисел.
std::string num(const char* name, double v, double ref, double tol) {
    char b[256];
    std::snprintf(b, sizeof b, "%s: %.17g vs %.17g (tol %.3g)", name, v, ref, tol);
    return b;
}

// Каждое сравнение — ещё и кейс эталона регрессии. Id "upstream.adaptive.<n>" остались от
// отдельного теста upstream_adaptive, который слит сюда: порядок вызовов прежний, поэтому значения
// сверяются с уже принятыми в tests/baselines.
void approx(double v, double ref, double tol, const char* name) {
    require(std::isfinite(v) && std::fabs(v - ref) <= tol, num(name, v, ref, tol));
#ifdef UCUDA_TEST_ARTIFACTS
    static size_t index = 0;
    regression::snapshot("upstream.adaptive." + std::to_string(index++),
        {{"assertion", name}, {"reference", ref}, {"absolute_tolerance", tol}}, {v});
#endif
}

System sys_of(std::vector<std::string> vars, std::vector<std::string> rhs, std::vector<std::string> params = { "p" }) {
    System s;
    s.vars = std::move(vars);
    s.rhs = std::move(rhs);
    s.params = std::move(params);
    return s;
}

// Параметры ядра адаптивного шага: DOP853, Hairer, rtol/atol.
UcudaAdaptParams params_for(const AdaptiveCode& code, int n, double span, const char* rtol, const char* atol,
                            const char* hmax = "") {
    AdaptiveSettings st;
    st.enabled = true;
    st.rtol = rtol;
    st.atol = atol;
    st.hmax = hmax;
    UcudaAdaptParams P{};
    std::string err;
    require(adaptive_build_params(st, code, n, span, P, err), "adaptive_build_params: " + err);
    return P;
}

AdaptiveRequest sweep_adaptive(const System& s, const char* rtol, double span) {
    const AdaptiveCode code = codegen_adaptive(s, Scheme::DOP853);
    AdaptiveRequest a;
    a.enabled = true;
    a.rhs = code.rhs; a.emb = code.emb; a.dprep = code.dprep; a.deval = code.deval;
    a.params = params_for(code, (int)s.vars.size(), span, rtol, "1e-12");
    return a;
}

// CPU: равномерная сетка total точек через dt от 0.
std::vector<std::vector<double>> cpu_grid(const System& s, const std::vector<double>& ic, const std::vector<double>& a,
                                          const UcudaAdaptParams& P, double dt, int total, AdaptiveStats& st) {
    SystemEvaluator ev(s);
    std::vector<std::vector<double>> traj;
    std::vector<double> times, log;
    double hfin = 0;
    computePhasePortraitCPU_adaptive(ev, IntScheme::DOP853, ic.data(), (int)ic.size(), a.data(), P,
                                     false, 0.0, dt * (total - 1), dt, total, 0, 0, traj, times, log, st, hfin);
    return traj;
}

void dense_cpu_tests() {
    const System decay = sys_of({ "x" }, { "-x" });
    const AdaptiveCode code = codegen_adaptive(decay, Scheme::DOP853);
    const std::vector<double> a = { 0.5, 0.0 };
    AdaptiveStats st;
    auto tr = cpu_grid(decay, { 1.0 }, a, params_for(code, 1, 10.0, "1e-9", "1e-12", "1"), 0.01, 1001, st);
    require(tr.size() == 1001, "CPU dense grid: wrong number of samples");
    double emax = 0;
    for (int i = 0; i < 1001; ++i) emax = std::fmax(emax, std::fabs(tr[(size_t)i][0] - std::exp(-0.01 * i)));
    require(emax < 1e-8, num("CPU dense output error", emax, 0, 1e-8));
    require(st.nacc < 250, "dense output forced a step per sample");

    const System osc = sys_of({ "x", "y" }, { "y", "-x" });
    const AdaptiveCode oc = codegen_adaptive(osc, Scheme::DOP853);
    tr = cpu_grid(osc, { 1.0, 0.0 }, a, params_for(oc, 2, 20.0, "1e-10", "1e-12"), 20.0, 2, st);
    approx(tr.back()[0], std::cos(20.0), 2e-8, "oscillator x(20)");
    approx(tr.back()[1], -std::sin(20.0), 2e-8, "oscillator y(20)");

    double err[2];
    const char* tols[2] = { "1e-3", "1e-8" };
    for (int k = 0; k < 2; ++k) {
        tr = cpu_grid(decay, { 1.0 }, a, params_for(code, 1, 2.0, tols[k], "1e-12"), 2.0, 2, st);
        err[k] = std::fabs(tr.back()[0] - std::exp(-2.0));
    }
    require(err[1] < err[0] * 0.01, "tightening rtol did not improve accuracy");

    AdaptiveSettings bad;
    bad.enabled = true;
    bad.rtol = "nan";
    UcudaAdaptParams P{};
    std::string e;
    require(!adaptive_build_params(bad, code, 1, 1.0, P, e) && !e.empty(), "NaN rtol accepted");
    bad.rtol = "0"; bad.atol = "0";
    require(!adaptive_build_params(bad, code, 1, 1.0, P, e), "rtol = atol = 0 accepted");
}

void engine_gpu_tests() {
    NvrtcEngine engine;
    const System decay = sys_of({ "x" }, { "-x" });
    const AdaptiveCode code = codegen_adaptive(decay, Scheme::DOP853);
    const UcudaAdaptParams P = params_for(code, 1, 1.1, "1e-9", "1e-12", "0.5");
    PhaseAdaptiveRequest rq;
    rq.rhs = code.rhs; rq.emb = code.emb; rq.dprep = code.dprep; rq.deval = code.deval;
    rq.amountOfX = 1; rq.ic_flat = { 1.0, 2.0 }; rq.N = 2; rq.values = { 0.5, 0.0 }; rq.params = P;
    rq.t_rec = 1.0; rq.dt = 0.01; rq.total = 101; rq.t_skip = 0.1;
    PhaseAdaptiveResult R;
    require(engine.run_phase_portraits_adaptive(rq, R), "GPU adaptive: " + engine.error());
    require(R.stats.size() == 2 && R.stats[0].nacc > 0, "GPU statistics missing");
    AdaptiveStats st;
    SystemEvaluator ev(decay);
    std::vector<std::vector<double>> traj;
    std::vector<double> times, log;
    double hfin = 0;
    computePhasePortraitCPU_adaptive(ev, IntScheme::DOP853, rq.ic_flat.data(), 1, rq.values.data(), P, false,
                                     0.1, 1.0, 0.01, 101, 0, 0, traj, times, log, st, hfin);
    require(traj.size() == 101 && R.traj[0].size() == 101, "grid sizes");
    for (int i = 0; i < 101; ++i)
        approx(R.traj[0][(size_t)i][0], traj[(size_t)i][0], 1e-9, "CPU / GPU dense output");
    approx(R.traj[1].back()[0], 2.0 * std::exp(-1.1), 2e-8, "GPU ensemble x(1.1)");

    // Лоренц: адаптивный DOP853 на GPU против мелкого RK4 на CPU (cl.exe)
    System lor = sys_of({ "x", "y", "z" }, { "sigma*(y - x)", "x*(rho - z) - y", "x*y - beta*z" },
                        { "sigma", "rho", "beta" });
    const AdaptiveCode lc = codegen_adaptive(lor, Scheme::DOP853);
    PhaseAdaptiveRequest lq;
    lq.rhs = lc.rhs; lq.emb = lc.emb; lq.dprep = lc.dprep; lq.deval = lc.deval;
    lq.amountOfX = 3; lq.ic_flat = { 1, 1, 1 }; lq.N = 1; lq.values = { 0.5, 10, 28, 8.0 / 3 };
    lq.params = params_for(lc, 3, 5.0, "1e-11", "1e-13"); lq.t_rec = 5.0; lq.dt = 5.0; lq.total = 2;
    require(engine.run_phase_portraits_adaptive(lq, R), "GPU Lorenz: " + engine.error());
    KrsCpuStep ref;
    std::vector<KrsCpuDiag> diags;
    require(ref.compile(codegen_scheme(lor, Scheme::RK4), 3, 4, diags), "RK4 reference compile");
    numb X[3] = { 1, 1, 1 }, A[4] = { 0.5, 10, 28, (numb)(8.0 / 3) };
    for (int i = 0; i < 500000; ++i) ref.fn()(X, A, (numb)1e-5);
    for (int k = 0; k < 3; ++k) approx(R.traj[0].back()[(size_t)k], X[k], 1e-5, "Lorenz x(5) vs fine RK4");

    ParametricEngine sweep;
    // адаптивные LLE / LS на линейной системе
    {
        LLE1DRequest q;
        q.krs_body = codegen_scheme(decay, Scheme::DOP853); q.amountOfX = 1; q.initial_conditions = { 1 };
        q.base_values = { 0.5, 0.0 }; q.param_index = 1; q.param_lo = 0; q.param_hi = 1; q.n_pts = 2;
        q.h = 0.01; q.t_max = 2; q.NT = 0.1; q.eps = 1e-4; q.adaptive = sweep_adaptive(decay, "1e-9", 2.0);
        auto g = sweep.run_lle_1d(q);
        require(g.ok && g.lyapunov.size() == 2, "adaptive LLE GPU: " + g.error);
        approx(g.lyapunov[0], -1.0, 1e-4, "adaptive LLE GPU");
        q.use_cpu = true;
        auto c = sweep.run_lle_1d(q);
        require(c.ok, "adaptive LLE CPU: " + c.error);
        approx(c.lyapunov[0], g.lyapunov[0], 1e-5, "adaptive LLE CPU vs GPU");
        q.use_cpu = false; q.continuation = true;
        auto k = sweep.run_lle_1d(q);
        require(k.ok, "adaptive LLE continuation: " + k.error);
        approx(k.lyapunov[0], -1.0, 1e-4, "adaptive LLE continuation");
    }
    const System diag = sys_of({ "x", "y" }, { "-x", "-2*y" });
    {
        LS1DRequest q;
        q.krs_body = codegen_scheme(diag, Scheme::DOP853); q.amountOfX = 2; q.initial_conditions = { 1, 1 };
        q.base_values = { 0.5, 0.0 }; q.param_index = 1; q.param_lo = 0; q.param_hi = 1; q.n_pts = 2;
        q.h = 0.01; q.t_max = 10; q.NT = 0.1; q.eps = 1e-4; q.adaptive = sweep_adaptive(diag, "1e-9", 10.0);
        auto g = sweep.run_ls_1d(q);
        require(g.ok && g.spectrum.size() == 2, "adaptive LS GPU: " + g.error);
        approx(g.spectrum[0][0] + g.spectrum[0][1], -3.0, 1e-4, "adaptive LS trace");
        q.use_cpu = true;
        auto c = sweep.run_ls_1d(q);
        require(c.ok, "adaptive LS CPU: " + c.error);
        approx(c.spectrum[0][0], g.spectrum[0][0], 1e-4, "adaptive LS CPU vs GPU");
        q.use_cpu = false; q.continuation = true;
        auto k = sweep.run_ls_1d(q);
        require(k.ok, "adaptive LS continuation: " + k.error);
        approx(k.spectrum[0][0] + k.spectrum[0][1], -3.0, 1e-4, "adaptive LS continuation trace");
    }
    // постоянный шаг: остальные семейства считаются
    const std::string body = codegen_scheme(diag, Scheme::RK4);
    {
        Dft1DRequest q; q.krs_body = body; q.amountOfX = 2; q.initial_conditions = { 1, 1 }; q.base_values = { 0.5, 0.0 };
        q.param_index = 1; q.param_lo = 0; q.param_hi = 1; q.n_pts = 2; q.h = 0.01; q.t_max = 1;
        auto r = sweep.run_dft_1d(q); require(r.ok, "DFT 1D: " + r.error);
    }
    {
        BasinsRequest q; q.krs_body = body; q.amountOfX = 2; q.initial_conditions = { 1, 1 }; q.base_values = { 0.5, 0.0 };
        q.n_pts = 4; q.h = 0.01; q.t_max = 1; q.transient_time = 0;
        auto r = sweep.run_basins(q); require(r.ok, "basins: " + r.error);
    }
    {
        FastSyncRequest q; q.krs_body = body; q.amountOfX = 2; q.values = { 0.5, 0.0 };
        q.ic_master = { 1, 1 }; q.ic_slave = { 1.01, 1.01 }; q.k_forward = { 1, 1 }; q.k_backward = { 1, 1 };
        q.h = 0.01; q.t_max = 0.1; q.window = 0.05; q.iter_of_synchr = 2; q.transient_time = 0; q.n_pts = 4;
        auto r = sweep.run_fastsync(q); require(r.ok, "FastSync: " + r.error);
    }
    {
        Bifurcation2DRequest q; q.krs_body = body; q.amountOfX = 2; q.initial_conditions = { 1, 1 }; q.base_values = { 0.5, 0.0 };
        q.n_pts = 2; q.sweep_over_var = q.sweep_over_var_2 = true; q.var_sweep_index = 0; q.var_sweep_index_2 = 1;
        q.param_lo = q.param_lo_2 = 0.5; q.param_hi = q.param_hi_2 = 1; q.h = 0.01; q.t_max = 1;
        auto r = sweep.run_bifurcation_2d(q); require(r.ok, "BD 2D: " + r.error);
        LLE2DRequest l; l.krs_body = body; l.amountOfX = 2; l.initial_conditions = { 1, 1 }; l.base_values = { 0.5, 0.0 };
        l.n_pts = 2; l.sweep_over_var = l.sweep_over_var_2 = true; l.var_sweep_index = 0; l.var_sweep_index_2 = 1;
        l.param_lo = l.param_lo_2 = 0.5; l.param_hi = l.param_hi_2 = 1; l.h = 0.01; l.t_max = 1; l.NT = 0.1;
        auto lr = sweep.run_lle_2d(l); require(lr.ok, "LLE 2D: " + lr.error);
        LS2DRequest s2; s2.krs_body = body; s2.amountOfX = 2; s2.initial_conditions = { 1, 1 }; s2.base_values = { 0.5, 0.0 };
        s2.n_pts = 2; s2.sweep_over_var = s2.sweep_over_var_2 = true; s2.var_sweep_index = 0; s2.var_sweep_index_2 = 1;
        s2.param_lo = s2.param_lo_2 = 0.5; s2.param_hi = s2.param_hi_2 = 1; s2.h = 0.01; s2.t_max = 1; s2.NT = 0.1;
        auto sr = sweep.run_ls_2d(s2); require(sr.ok, "LS 2D: " + sr.error);
    }
    // комплексные и композиционные схемы: x(1) на x' = -x, y' = -2y
    for (const char* name : { "Complex CD", "Complex CD4", "Complex Implicit Euler", "SEMP", "SIMP", "CD", "Implicit Midpoint" }) {
        const std::string b = codegen_scheme(diag, scheme_from_name(name));
        std::vector<std::vector<std::vector<double>>> out;
        require(engine.compile(b, 2) && engine.run_phase_portraits({ 1, 1 }, 1, { 0.5, 0.0, 4, 1e-12, 25, 1 }, 0.01, 101, 0, out),
                    std::string(name) + ": " + engine.error());
        approx(out[0].back()[0], std::exp(-1.0), 2e-4, name);
    }
}

}  // namespace

int main(int argc,char** argv) {
    try {
#if defined(_WIN32) && !defined(NDEBUG)
        _CrtSetReportMode(_CRT_ASSERT,_CRTDBG_MODE_FILE);
        _CrtSetReportFile(_CRT_ASSERT,_CRTDBG_FILE_STDERR);
#endif
        const bool gpu=argc>1 && std::string(argv[1])=="--gpu";
        if(gpu && cuInit(0)!=CUDA_SUCCESS) return 77;
#ifdef UCUDA_TEST_ARTIFACTS
        regression::test_context()=gpu ? "adaptive_gpu" : "adaptive_cpu";   // имя теста CTest
#endif
        std::cerr<<"sessions"<<std::endl;session_tests(); upstream_session_tests();
        std::cerr<<"adaptive"<<std::endl;upstream_adaptive_tests(gpu); upstream_sweep_tests(gpu);
        std::cerr<<"fixed"<<std::endl;numerical_tests(gpu); phase_session_tests(gpu);
        cpu_analysis_tests(); sampling_sweep_tests(false);
        std::cerr<<"dense"<<std::endl;dense_cpu_tests();
        if(gpu) sampling_sweep_tests(true);
        if(gpu) { gpu_analysis_tests(); upstream_grid_tests(); engine_gpu_tests(); }
        std::cout<<"Fixed and upstream adaptive numerical, serialization and "<<(gpu?"CPU/GPU":"CPU")<<" checks passed\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
