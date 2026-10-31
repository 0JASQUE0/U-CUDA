#include "test_util.h"
#include "network_fixtures.h"
#include "network_session.h"
#include "num_parse.h"
#include "sysparse.hpp"
#include "system_library.h"
#include "nvrtc_engine.h"
#include <cuda.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <thread>

namespace fs = std::filesystem;
namespace {
NetworkSession session(const NetworkFixture& f) {
    NetworkSession s; s.sys = f.sys; s.vars = f.sys.vars; s.params = f.sys.params;
    s.loaded_system_name = f.name;
    return s;
}
NetworkConfig config(const NetworkFixture& f, int n) {
    NetworkConfig c; c.nodes.resize(n); c.ic_spread_text = "0";
    for (size_t p = 0; p < f.parameters.size(); ++p) c.param_values[f.sys.params[p]] = fmt_num_shortest(f.parameters[p]);
    for (size_t k = 0; k < f.initial.size(); ++k) c.initial_conditions[f.sys.vars[k]] = fmt_num_shortest(f.initial[k]);
    c.laws = {net_default_law(f.sys.vars)};
    return c;
}
std::vector<double> values(const NetworkMonolithic& snap) {
    std::vector<double> a{0.5};
    for (const auto& p : snap.sys.params) a.push_back(parse_num(snap.record.param_values.at(p), NAN));
    const size_t base = a.size(); a.resize(base + kSchemeTailSlots);
    scheme_fill_tail(a.data() + base, "4", NewtonSettings{});
    return a;
}
std::vector<double> initial(const NetworkMonolithic& snap) {
    std::vector<double> x;
    for (const auto& v : snap.sys.vars) x.push_back(parse_num(snap.record.init_conditions.at(v), NAN));
    return x;
}
void rejected(const std::function<void()>& f, const char* text) {
    try { f(); } catch (const std::exception& e) {
        ut::require(std::string(e.what()).find(text) != std::string::npos, e.what()); return;
    }
    ut::require(false, std::string("expected rejection: ") + text);
}
void cpu_tests() {
    const auto f = network_fixtures()[0];
    auto s = session(f); auto c = config(f, 3);
    c.coupling_text = "0.3";
    c.nodes[0].param_values["I"] = "0.6";
    c.nodes[2].param_values["tau"] = "11";
    c.nodes[2].param_values["I"] = "0.9";
    c.edges = {{0, 1, false, "2", 0}, {2, 1, false, "0.5", 1}, {1, 2, false, "-0.25", 0}};
    c.laws.push_back({"parameter-dependent", {"K*(I_j-I_i)+K*(v_j-v_i)/(1+v_j*v_j)", ""}});
    for (bool normalized : {false, true}) {
        c.normalize_coupling = normalized;
        const auto snap = net_monolithic_snapshot(s, c);
        ut::require(snap.record.param_values.at("I1") == "0.6" && snap.record.param_values.at("tau3") == "11", "node parameter names");
        const std::vector<double> x{-1.2, 0.4, -0.5, 1.1, 0.7, -0.3};
        std::vector<double> actual(6), expected(6);
        auto a = values(snap); SystemEvaluator ev(snap.sys); ev.eval(x.data(), a.data(), actual.data());
        for (int i = 0; i < 3; ++i) {
            double v = x[2*i], w = x[2*i+1], I = i == 0 ? 0.6 : (i == 2 ? 0.9 : 0.5), tau = i == 2 ? 11 : 12.5;
            expected[2*i] = v-v*v*v/3-w+I; expected[2*i+1] = (v+0.7-0.8*w)/tau;
        }
        // Written for this directed graph, independently of CSR/generator traversal.
        expected[2] += (0.6*(x[0]-x[2]) + 0.15*(0.9-0.5) + 0.15*(x[4]-x[2])/(1+x[4]*x[4]))/(normalized ? 2 : 1);
        expected[4] += -0.075*(x[2]-x[4]);
        for (int k = 0; k < 6; ++k) ut::approx(actual[k], expected[k], 1e-13, "directed weighted RHS");
        const auto rec = record_from_json(record_to_json(snap.record));
        auto alphabet = snap.sys.vars; alphabet.insert(alphabet.end(), snap.sys.params.begin(), snap.sys.params.end());
        const auto parsed = parse_system_from_latex(rec.plain_text, alphabet);
        if (parsed.vars != snap.sys.vars || parsed.params != snap.sys.params) {
            for (const auto& v : parsed.vars) std::printf("parsed var: %s\n",v.c_str());
            for (const auto& p : parsed.params) std::printf("parsed param: %s\n",p.c_str());
            for (const auto& p : snap.sys.params) std::printf("expected param: %s\n",p.c_str());
        }
        ut::require(parsed.vars == snap.sys.vars && parsed.params == snap.sys.params, "record parser symbol order");
        SystemEvaluator restored(parsed); std::vector<double> y(6);
        restored.eval(x.data(), a.data(), y.data());
        for (int k = 0; k < 6; ++k) ut::approx(y[k], expected[k], 1e-13, "record RHS roundtrip");
    }
    // AST replacement must distinguish names, functions and LaTeX structure.
    const auto expr = codegen_substitute_expression("\\frac{x}{a}+x1+sin(x)", true,
        {{"x", "v1"}, {"x1", "other"}, {"a", "tau"}});
    System e; e.vars = {"v1", "other"}; e.params = {"tau"}; e.rhs = {expr, "0"};
    double deriv[2], x[] = {2, 3}, a[] = {0.5, 4}; SystemEvaluator(e).eval(x, a, deriv);
    ut::approx(deriv[0], 3.5 + std::sin(2.0), 1e-14, "AST substitution");
    NetworkFixture constant{"constant", {{"pi"}, {}, {"pi"}}, {}, {0}};
    auto constantSession=session(constant); auto constantConfig=config(constant,1);
    const auto constantSnapshot=net_monolithic_snapshot(constantSession,constantConfig);
    auto constantValues=values(constantSnapshot);
    SystemEvaluator(constantSnapshot.sys).eval(x,constantValues.data(),deriv);
    ut::approx(deriv[0],3.14159265358979323846,1e-15,"reserved constant semantics");

    auto large = config(f, 16); ut::require(net_monolithic_snapshot(s, large).sys.vars.size() == 32, "32 variables accepted");
    large.nodes.resize(17); rejected([&] { net_monolithic_snapshot(s, large); }, "32");
    auto bad = c; bad.edges[0].to = 3; rejected([&] { net_monolithic_snapshot(s, bad); }, "edge");
    bad = c; bad.edges[0].law = 99; rejected([&] { net_monolithic_snapshot(s, bad); }, "law");
    bad = c; bad.param_values["a"] = "garbage"; rejected([&] { net_monolithic_snapshot(s, bad); }, "numeric");
    auto map = session(f); map.sys.is_map = true; rejected([&] { net_monolithic_snapshot(map, c); }, "ODE");
    // K and generated node names may already exist in the source model.
    NetworkFixture collision{"collision", {{"x"}, {"K", "x1", "I1"}, {"-x+K+x1+I1"}}, {1,2,3}, {0}};
    auto cs = session(collision); auto cc = config(collision, 2); cc.edges = {{0,1,true,"1",0}};
    auto col = net_monolithic_snapshot(cs, cc);
    ut::require(col.record.param_values.count("K_model") == 1 && col.sys.vars[0] != "x1", "collision-safe names");
    auto alpha = col.sys.vars; alpha.insert(alpha.end(), col.sys.params.begin(), col.sys.params.end());
    auto colParsed = parse_system_from_latex(col.record.plain_text, alpha);
    ut::require(colParsed.vars == col.sys.vars && colParsed.params == col.sys.params, "collision record order");
    // Engine parameter capacity includes symmetry and four scheme-tail slots.
    NetworkFixture capacity{"capacity", {{"x"}, {}, {"-x"}}, {}, {0}};
    for (int p = 0; p < 58; ++p) { auto name = "p"+std::to_string(p); capacity.sys.params.push_back(name); capacity.parameters.push_back(1); capacity.sys.rhs[0] += "+"+name; }
    auto cap = session(capacity); auto cfg = config(capacity, 1);
    ut::require(net_monolithic_snapshot(cap, cfg).sys.params.size() == 59, "59 parameters accepted");
    cap.sys.params.push_back("extra"); cap.params = cap.sys.params; cap.sys.rhs[0] += "+extra"; cfg.param_values["extra"] = "1";
    rejected([&] { net_monolithic_snapshot(cap, cfg); }, "59");
    // Saved records are independent values; source edits cannot change them.
    auto saved = net_monolithic_snapshot(s, c);
    const fs::path dir = fs::u8path(exe_dir()) / ("monolithic-tmp-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(GetTickCount64()));
    SystemLibrary lib(dir.u8string()); const auto name = lib.save(saved.record);
    c.nodes[0].param_values["I"] = "9";
    ut::require(lib.load(name).param_values.at("I1") == "0.6", "independent saved snapshot");
    std::printf("CPU: directed graph, weights/laws, symbol collisions, limits, JSON/Library OK\n");
}

System manual_pair(const NetworkFixture& f) {
    System m; m.params = f.sys.params; m.params.push_back("K");
    if (std::string(f.name) == "fhn") {
        m.vars = {"v1","w1","v2","w2"};
        m.rhs = {"v1-v1*v1*v1/3-w1+I+K*(v2-v1)", "(v1+a-b*w1)/tau",
                 "v2-v2*v2*v2/3-w2+I+K*(v1-v2)", "(v2+a-b*w2)/tau"};
    } else {
        m.vars = {"x1","y1","z1","x2","y2","z2"};
        m.rhs = {"y1-a*x1*x1*x1+b*x1*x1-z1+I+K*(x2-x1)", "c-d*x1*x1-y1", "r*(s*(x1-xr)-z1)",
                 "y2-a*x2*x2*x2+b*x2*x2-z2+I+K*(x1-x2)", "c-d*x2*x2-y2", "r*(s*(x2-xr)-z2)"};
    }
    return m;
}
std::vector<double> endpoint(NvrtcEngine& engine, const System& sys, const std::vector<double>& ic,
    const std::vector<double>& a, double h, int steps) {
    std::vector<std::vector<std::vector<double>>> out;
    ut::require(engine.compile(codegen_scheme(sys, Scheme::RK4), int(sys.vars.size())), "phase compile: " + engine.error());
    ut::require(engine.run_phase_portraits(ic, 1, a, h, steps+1, 0, out), "phase run: " + engine.error());
    return out[0].back();
}
void check_spread(ParametricEngine& network, NetworkSession& s, const NetworkConfig& c) {
    auto noisy = c; noisy.ic_spread_text = "0.125"; noisy.ic_seed_text = "17";
    noisy.t_max_text = "0.02";
    const auto snapshotIC = initial(net_monolithic_snapshot(s,noisy));
    s.configs = {noisy};
    ut::require(s.run_async(network,0),"NetworkSession initial spread run");
    while (s.in_flight) { s.poll(); std::this_thread::sleep_for(std::chrono::milliseconds(2)); }
    ut::require(s.configs[0].last_run_ok,"NetworkSession initial spread result");
    for (size_t k = 0; k < snapshotIC.size(); ++k)
        ut::require(snapshotIC[k] == s.configs[0].result.data[k],"snapshot realizes the same IC spread as Network");
    std::printf("NetworkSession/snapshot initial spread: OK\n");
}
void gpu_tests() {
    ParametricEngine network; NvrtcEngine phase;
    for (const auto& f : network_fixtures()) {
        auto s = session(f); auto c = config(f, 2); c.edges = {{0,1,true,"1",0}};
        c.coupling_text = "0.2"; c.nodes[1].initial_conditions[f.sys.vars[0]] = fmt_num_shortest(f.initial[0]+0.2);
        const auto snap = net_monolithic_snapshot(s,c);
        if (std::string(f.name) == "fhn") {
            check_spread(network,s,c);
        }
        const auto ic = initial(snap), a = values(snap);
        const auto generated = endpoint(phase, snap.sys, ic, a, 1.0/4096, 4096);
        const auto manual = endpoint(phase, manual_pair(f), ic, a, 1.0/4096, 4096);
        for (size_t k = 0; k < generated.size(); ++k) ut::approx(generated[k], manual[k], 1e-10, "generated vs manual pair");
        NetworkRequest q; q.krs_body = codegen_scheme(f.sys, Scheme::RK4); q.coupling_body = net_coupling_body(f.sys,c.laws);
        q.n_nodes = 2; q.amountOfX = int(f.sys.vars.size()); q.amountOfValues = int(f.parameters.size())+1+kSchemeTailSlots;
        std::vector<double> node{0.5}; node.insert(node.end(),f.parameters.begin(),f.parameters.end());
        const size_t base = node.size(); node.resize(base+kSchemeTailSlots); scheme_fill_tail(node.data()+base,"4",NewtonSettings{});
        q.values = node; q.values.insert(q.values.end(),node.begin(),node.end()); q.initial_conditions = ic;
        net_build_csr(c,0.2,q.edge_start,q.edge_src,q.edge_w,q.edge_law); q.t_max = 1;
        double previous = 0;
        for (int steps : {25,50,100}) {
            q.h = 1.0/steps; q.pre_scaller = steps;
            auto r = network.run_network(q); ut::require(r.ok && r.n_diverged == 0, "Network convergence: " + r.error);
            double error = 0; for (size_t k = 0; k < generated.size(); ++k) error = std::max(error,std::fabs(r.data[r.data.size()-generated.size()+k]-generated[k]));
            std::printf("%s h=%.4g split endpoint error=%.9g\n",f.name,q.h,error);
            if (previous > 0) ut::require(previous/error > 1.7 && previous/error < 2.4,"first-order split convergence");
            previous = error;
        }
        // Ordinary analysis entry points accept the generated six-/four-dimensional ODE.
        const auto body = codegen_scheme(snap.sys,Scheme::RK4);
        const int K = int(std::find(snap.sys.params.begin(),snap.sys.params.end(),"K")-snap.sys.params.begin())+1;
        Bifurcation1DRequest b; b.krs_body=body; b.amountOfX=int(ic.size()); b.initial_conditions=ic; b.base_values=a;
        b.param_index=K; b.param_lo=0.1; b.param_hi=0.3; b.n_pts=2; b.h=0.01; b.t_max=1;
        auto br=network.run_bifurcation_1d(b); ut::require(br.ok,"monolithic BD: "+br.error);
        LLE1DRequest l; l.krs_body=body; l.amountOfX=int(ic.size()); l.initial_conditions=ic; l.base_values=a;
        l.param_index=K; l.param_lo=0.1; l.param_hi=0.3; l.n_pts=2; l.h=0.01; l.t_max=1; l.NT=0.1;
        auto lr=network.run_lle_1d(l); ut::require(lr.ok,"monolithic LLE: "+lr.error);
        LS1DRequest ls; ls.krs_body=body; ls.amountOfX=int(ic.size()); ls.initial_conditions=ic; ls.base_values=a;
        ls.param_index=K; ls.param_lo=0.1; ls.param_hi=0.3; ls.n_pts=2; ls.h=0.01; ls.t_max=1; ls.NT=0.1;
        auto sr=network.run_ls_1d(ls); ut::require(sr.ok && sr.n_exponents==int(ic.size()),"monolithic LS: "+sr.error);
        for (const auto& row : sr.spectrum) for (double v : row) ut::require(std::isfinite(v),"non-finite LS");
        BasinsRequest bas; bas.krs_body=body; bas.amountOfX=int(ic.size()); bas.initial_conditions=ic; bas.base_values=a;
        bas.n_pts=2; bas.axis_x_lo=ic[0]-0.1; bas.axis_x_hi=ic[0]+0.1;
        bas.axis_y_lo=ic[1]-0.1; bas.axis_y_hi=ic[1]+0.1; bas.h=0.01; bas.t_max=1; bas.transient_time=0;
        auto ba=network.run_basins(bas); ut::require(ba.ok,"monolithic basins: "+ba.error);
        std::printf("%s: GPU handwritten pair, h convergence, BD/LLE/LS/basins over K OK\n",f.name);
    }
}
void export_pairs(const fs::path& dir) {
    fs::create_directories(dir);
    for (const auto& f : network_fixtures()) {
        auto s = session(f); auto c = config(f,2); c.label = "Pair";
        c.edges = {{0,1,true,"1",0}}; c.coupling_text = "0.2";
        c.nodes[1].initial_conditions[f.sys.vars[0]] = fmt_num_shortest(f.initial[0]+0.2);
        const auto snap = net_monolithic_snapshot(s,c);
        std::ofstream file(dir / (std::string(f.name)+"-pair.json"),std::ios::binary);
        file << record_to_json(snap.record);
        ut::require(bool(file),"write pair fixture");
        std::printf("exported %s pair\n",f.name);
    }
}
void pilot_maps(const fs::path& dir) {
    fs::create_directories(dir);
    export_pairs(dir);
    ParametricEngine engine;
    for (const auto& f : network_fixtures()) {
        auto s = session(f); auto c = config(f,2); c.edges = {{0,1,true,"1",0}};
        c.nodes[1].initial_conditions[f.sys.vars[0]] = fmt_num_shortest(f.initial[0]+0.2);
        const auto snap = net_monolithic_snapshot(s,c);
        Bifurcation2DRequest q; q.krs_body=codegen_scheme(snap.sys,Scheme::RK4);
        q.amountOfX=int(snap.sys.vars.size()); q.initial_conditions=initial(snap); q.base_values=values(snap);
        q.param_index=int(std::find(snap.sys.params.begin(),snap.sys.params.end(),"K")-snap.sys.params.begin())+1;
        q.param_index_2=int(std::find(snap.sys.params.begin(),snap.sys.params.end(),"I")-snap.sys.params.begin())+1;
        q.param_lo=0; q.param_hi=1; q.param_lo_2=std::string(f.name)=="fhn" ? 0.2 : 2.0;
        q.param_hi_2=std::string(f.name)=="fhn" ? 1.0 : 4.0;
        q.n_pts=16; q.transient_time=std::string(f.name)=="fhn" ? 200 : 1000;
        q.t_max=std::string(f.name)=="fhn" ? 400 : 1000;
        q.eps_dbscan=0.02; q.mult_peak=1; q.mult_interval=0;
        q.h=0.01; const auto coarse=engine.run_bifurcation_2d(q);
        ut::require(coarse.ok,"pilot map h: "+coarse.error);
        q.h=0.005; const auto fine=engine.run_bifurcation_2d(q);
        ut::require(fine.ok && fine.values.size()==256,"pilot map h/2: "+fine.error);
        std::ofstream csv(dir/(std::string(f.name)+"-map.csv"));
        csv << "K,I,clusters_h_0.01,clusters_h_0.005\n";
        size_t changed=0;
        for (int iy=0;iy<16;++iy) for (int ix=0;ix<16;++ix) {
            size_t k=size_t(iy)*16+ix;
            changed += coarse.values[k]!=fine.values[k];
            csv << fmt_num_shortest((double)ucuda_node_value(ix,16,(numb)q.param_lo,(numb)q.param_hi)) << ','
                << fmt_num_shortest((double)ucuda_node_value(iy,16,(numb)q.param_lo_2,(numb)q.param_hi_2))
                << ',' << coarse.values[k] << ',' << fine.values[k] << '\n';
        }
        ut::require(bool(csv),"pilot CSV write");
        std::printf("%s pilot map: changed on h/2: %zu/256\n",f.name,changed);
    }
}
}
int main(int argc, char** argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    return ut::run_main("monolithic",[&]() -> int {
        if (argc == 3 && std::string(argv[1]) == "--export") { export_pairs(fs::u8path(argv[2])); return 0; }
        if (argc == 2 && std::string(argv[1]) == "--cpu") { cpu_tests(); return 0; }
        if (argc == 2 && std::string(argv[1]) == "--spread") {
            ut::require(cuInit(0)==CUDA_SUCCESS,"spread test requires CUDA"); set_nvrtc_rdc(true);
            const auto f=network_fixtures()[0]; auto s=session(f); auto c=config(f,2);
            c.edges={{0,1,true,"1",0}}; c.nodes[1].initial_conditions["v"]="-0.8";
            ParametricEngine engine; check_spread(engine,s,c); return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--maps") {
            ut::require(cuInit(0)==CUDA_SUCCESS,"pilot maps require CUDA");
            set_nvrtc_rdc(true); pilot_maps(fs::u8path(argv[2])); return 0;
        }
        ut::require(argc == 1,"usage: monolithic.exe [--export DIR | --maps DIR | --spread | --cpu]");
        cpu_tests();
        if (cuInit(0)!=CUDA_SUCCESS) return ut::kSkip;
        set_nvrtc_rdc(true); gpu_tests(); return 0;
    });
}
