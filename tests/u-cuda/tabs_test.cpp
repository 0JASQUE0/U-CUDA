// Вкладки Order и Network через их сессии (этап 11): load_from_record -> run_async -> poll, как GUI.
//   - Order: оценка порядка на Лоренце — Euler ~1, RK4 ~4 (GPU и CPU в double); Performance и область
//     устойчивости считаются без ошибки;
//   - Network: кольцо из 16 узлов Лоренца с диффузионной связью — расчёт проходит, траектории конечны.
// FastSync покрыт тестом adaptive, остальные вкладки — library / dae / adaptive.
#include "test_util.h"
#include "order_session.h"
#include "network_session.h"
#include "parametric_engine.h"
#include "system_library.h"
#include "sysparse.hpp"
#include <cuda.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include <algorithm>

namespace fs = std::filesystem;

namespace {

std::vector<std::string> split(const std::string& src) {
    std::vector<std::string> out; std::string cur;
    for (char c : src) {
        if (c == ',' || c == ' ' || c == '\t' || c == '\n' || c == ';') { if (!cur.empty()) { out.push_back(cur); cur.clear(); } }
        else cur += c;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

template <class S> void wait(S& s) {
    while (s.in_flight) { s.poll(); std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
}

double median_ok(const OrderResult& r) {
    std::vector<double> v;
    for (size_t i = 0; i < r.p.size(); ++i)
        if (i < r.status.size() && r.status[i] == 0 && std::isfinite(r.p[i])) v.push_back(r.p[i]);
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main() {
    return ut::run_main("tabs", []() -> int {
        if (cuInit(0) != CUDA_SUCCESS) { std::printf("no CUDA device: skipped\n"); return ut::kSkip; }
        set_nvrtc_rdc(true);
        // Запись библиотеки и её система — как AppModel::build_system для ОДУ (GUI кладёт её в sys сессии)
        auto load = [](const char* dir, SystemRecord& rec, System& out) {
            const fs::path p = fs::u8path(exe_dir()) / ".." / ".." / ".." / "U-CUDA" / "library" / dir / "system.json";
            std::ifstream f(p, std::ios::binary);
            std::stringstream ss;
            ss << f.rdbuf();
            rec = record_from_json(ss.str());
            out = parse_system_from_latex(rec.mode == "Plain" ? rec.plain_text : rec.latex_text,
                split(rec.vars_text + "," + rec.params_text),
                rec.param_order == "AsInSystem" ? ParamOrder::AsInSystem : ParamOrder::AsInAlphabet, {}, false);
        };
        SystemRecord r, rd;
        System sys, sysd;
        load("Lorenz", r, sys);
        load("Dahlquist 2D", rd, sysd);   // область устойчивости — только на двумерной задаче Дальквиста
        const auto vars = split(r.vars_text), params = split(r.params_text);
        ParametricEngine eng;

        // Order: порядок
        for (int gpu = 1; gpu >= 0; --gpu)
            for (const char* sch : { "Euler", "RK4" }) {
                OrderAnalysisSession o;
                o.load_from_record(r, vars, params);
                o.sys = sys;
                if (o.configs.empty()) o.add_config();
                OrderConfig& c = o.configs[0];
                c.scheme = sch;
                c.calc_kind = kOrderCalcOrder;
                c.use_gpu = gpu;
                c.axis_x_lo_text = "1e-3"; c.axis_x_hi_text = "1e-2"; c.axis_x_n_text = "6";
                c.t_max_text = "1";
                { const bool ok_ = o.run_async(eng, 0); ut::require(ok_, std::string("Order run_async ") + sch + ": " + c.last_error); }
                wait(o);
                ut::require(c.last_run_ok && c.result.ok, std::string("Order ") + sch + ": " + c.last_error);
                const double pm = median_ok(c.result);
                const double want = sch[0] == 'R' ? 4.0 : 1.0;
                std::printf("  Order %s (%s): median order %.3f over %d ok cells of %zu\n", sch, gpu ? "GPU" : "CPU",
                            pm, c.result.n_ok, c.result.p.size());
                ut::approx(pm, want, 0.3, (std::string("Order: order of ") + sch).c_str());
            }
        // Order: Performance и устойчивость
        for (int kind : { kOrderCalcPerf, kOrderCalcStab }) {
            const bool st = kind == kOrderCalcStab;
            OrderAnalysisSession o;
            o.load_from_record(st ? rd : r, st ? split(rd.vars_text) : vars, st ? split(rd.params_text) : params);
            o.sys = st ? sysd : sys;
            if (o.configs.empty()) o.add_config();
            OrderConfig& c = o.configs[0];
            c.scheme = "RK4";
            c.calc_kind = kind;
            c.axis_x_lo_text = "1e-3"; c.axis_x_hi_text = "1e-2"; c.axis_x_n_text = "4";
            c.t_max_text = "1";
            c.perf_repeats_text = "3"; c.perf_warmup_text = "1";
            { const bool ok_ = o.run_async(eng, 0); ut::require(ok_, "Order run_async perf/stab: " + c.last_error); }
            wait(o);
            const char* nm = kind == kOrderCalcPerf ? "Performance" : "stability region";
            const bool okk = kind == kOrderCalcPerf ? c.perf_last_run_ok : c.stab_last_run_ok;   // свои флаги у расчётов
            std::printf("  Order %s: %s%s\n", nm, okk ? "ok" : "FAILED ", c.last_error.c_str());
            ut::require(okk, std::string("Order ") + nm + ": " + c.last_error);
        }

        // Network: кольцо 16 узлов
        {
            NetworkSession n;
            n.load_from_record(r, vars, params);
            n.sys = sys;
            if (n.configs.empty()) n.add_config();
            NetworkConfig& c = n.configs[0];
            c.scheme = "RK4";
            c.t_max_text = "5";
            const std::string e = net_generate_topology(c, vars);
            ut::require(e.empty(), "Network topology: " + e);
            { const bool ok_ = n.run_async(eng, 0); ut::require(ok_, "Network run_async: " + c.last_error); }
            wait(n);
            ut::require(c.last_run_ok && c.result.ok, "Network: " + c.last_error);
            size_t fin = 0;
            for (double v : c.result.data) fin += std::isfinite(v) ? 1 : 0;
            std::printf("  Network ring %d nodes x %d vars, %d points: %zu of %zu values finite, %d diverged\n",
                        c.result.n_nodes, c.result.n_vars, c.result.n_points, fin, c.result.data.size(), c.result.n_diverged);
            ut::require(c.result.n_nodes == 16 && fin == c.result.data.size() && fin > 0, "Network: finite trajectories");
        }
        return 0;
    });
}
