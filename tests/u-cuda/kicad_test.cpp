// Импорт схем KiCad и сверка с ngspice (этап 8-9 на живом KiCad). Нет KiCad — пропуск (77).
//   - каждая демо-схема share\kicad\demos\simulation: .kicad_sch -> kicad-cli (spice netlist) -> spice_import;
//     печатается, что импортировано и что отвергнуто (с причиной); простые схемы обязаны импортироваться;
//   - сверка переходного процесса с ngspice: ngspice.dll из KiCad (shared ngspice, грузится в рантайме).
#include "test_util.h"
#include "kicad_import.h"
#include "spice_parse.h"
#include "dae_devices.h"
#include "dae_ir.h"
#include "krs_cpu.h"
#include <windows.h>
#include <filesystem>
#include <map>
#include <memory>
#include <sstream>
#include <fstream>

namespace fs = std::filesystem;

namespace {

fs::path g_kicad_root;   // ...\KiCad\<ver>

// --- shared ngspice (ngspice.dll из KiCad): объявления sharedspice.h ---------------------------------
struct NgVec { char* v_name; int v_type; short v_flags; double* v_realdata; void* v_compdata; int v_length; };
typedef int(NgSendChar)(char*, int, void*);
typedef int(NgSendStat)(char*, int, void*);
typedef int(NgExit)(int, bool, bool, int, void*);
typedef int(NgData)(void*, int, int, void*);
typedef int(NgInit)(void*, int, void*);
typedef int(NgBg)(bool, int, void*);
struct Ngspice {
    HMODULE h = nullptr;
    int (*init)(NgSendChar*, NgSendStat*, NgExit*, NgData*, NgInit*, NgBg*, void*) = nullptr;
    int (*circ)(char**) = nullptr;
    int (*command)(char*) = nullptr;
    NgVec* (*vec)(char*) = nullptr;
    std::string log;
    static int on_char(char* s, int, void* u) { static_cast<Ngspice*>(u)->log += std::string(s) + "\n"; return 0; }
    static int on_stat(char*, int, void*) { return 0; }
    static int on_exit(int, bool, bool, int, void*) { return 0; }
    static int on_bg(bool, int, void*) { return 0; }
    bool load(const fs::path& dll) {
        h = LoadLibraryExW(dll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!h) return false;
        init = (decltype(init))GetProcAddress(h, "ngSpice_Init");
        circ = (decltype(circ))GetProcAddress(h, "ngSpice_Circ");
        command = (decltype(command))GetProcAddress(h, "ngSpice_Command");
        vec = (decltype(vec))GetProcAddress(h, "ngGet_Vec_Info");
        if (!init || !circ || !command || !vec) return false;
        return init(on_char, on_stat, on_exit, nullptr, nullptr, on_bg, this) == 0;
    }
    void cmd(const std::string& c) { std::string s = c; command(&s[0]); }
    // Netlist целиком (без анализов) + .options + .tran; time и узлы — по именам ngspice.
    bool tran(const std::string& net, double tstep, double tstop, double maxstep, std::vector<double>& time,
              std::map<std::string, std::vector<double>>& v, const std::vector<std::string>& nodes) {
        log.clear();
        cmd("remcirc");
        std::vector<std::string> lines;
        std::istringstream in(net);
        std::string l;
        bool ctl = false, uic = false;
        while (std::getline(in, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            std::string k = l.substr(0, l.find_first_of(" \t"));
            for (char& ch : k) ch = (char)std::tolower((unsigned char)ch);
            if (k == ".control") { ctl = true; continue; }
            if (ctl) { if (k == ".endc") ctl = false; continue; }
            if (k == ".tran") {
                std::string lo = l;
                for (char& ch : lo) ch = (char)std::tolower((unsigned char)ch);
                if (lo.find("uic") != std::string::npos) uic = true;
            }
            if (k == ".tran" || k == ".ac" || k == ".op" || k == ".dc" || k == ".end" || k == ".options") continue;
            lines.push_back(l);
        }
        char b[256];
        lines.push_back(".options reltol=1e-7 abstol=1e-15 vntol=1e-10 chgtol=1e-18 method=gear");
        std::snprintf(b, sizeof b, ".tran %g %g 0 %g%s", tstep, tstop, maxstep, uic ? " uic" : "");
        lines.push_back(b);
        lines.push_back(".end");
        std::vector<char*> arr;
        for (std::string& s : lines) arr.push_back(&s[0]);
        arr.push_back(nullptr);
        if (circ(arr.data()) != 0) return false;
        cmd("run");
        NgVec* t = vec((char*)"time");
        if (!t || t->v_length < 2) return false;
        time.assign(t->v_realdata, t->v_realdata + t->v_length);
        for (const std::string& n : nodes) {
            std::string q = n, q2 = "v(" + n + ")";
            NgVec* x = vec(&q[0]);
            if (!x || !x->v_realdata) x = vec(&q2[0]);
            if (!x || !x->v_realdata) { log += "no vector " + n + "\n"; cmd("display"); return false; }
            v[n].assign(x->v_realdata, x->v_realdata + x->v_length);
        }
        return true;
    }
};
Ngspice g_ng;
bool g_ng_ok = false;

double interp(const std::vector<double>& t, const std::vector<double>& y, double tq) {
    auto it = std::lower_bound(t.begin(), t.end(), tq);
    if (it == t.begin()) return y.front();
    if (it == t.end()) return y.back();
    const size_t k = (size_t)(it - t.begin());
    const double w = (tq - t[k - 1]) / (t[k] - t[k - 1]);
    return y[k - 1] + w * (y[k] - y[k - 1]);
}

// Наш импорт той же схемы: НУ — .ic/UIC или рабочая точка (как AppModel::import_circuit), Radau IIA шагом h.
struct Ours {
    SpiceImportResult r;
    System sys;
    std::vector<double> a, X;
    int idx(const std::string& v) const {
        for (size_t i = 0; i < sys.vars.size(); ++i) if (sys.vars[i] == v) return (int)i;
        throw std::runtime_error("no variable " + v);
    }
};
std::string sanitize_node(const std::string& n) {   // как sanitize() в spice_parse.cpp
    std::string o;
    for (char c : n) o += std::isalnum((unsigned char)c) ? (char)std::tolower((unsigned char)c) : '_';
    while (!o.empty() && o.back() == '_') o.pop_back();
    size_t a = 0;
    while (a < o.size() && o[a] == '_') ++a;
    o = o.substr(a);
    if (!o.empty() && std::isdigit((unsigned char)o[0])) o = "n" + o;
    return o;
}

// Сравнить узлы: max |ours - ngspice| на ns точках, относительно размаха сигнала.
double compare(const char* title, const std::string& net, const std::vector<std::string>& nodes, double h, double T,
               double tstep, int ns = 200) {
    Ours m;
    std::string err;
    SpiceImportOptions o;
    { const bool ok_ = spice_import(net, o, m.r, &err); ut::require(ok_, std::string(title) + ": import: " + err); }
    const DaeSystem d = parse_dae(m.r.plain_text, m.r.vars_text, m.r.alg_text, m.r.params_text);
    m.sys = d.as_codegen_system();
    std::vector<double> a = { 0.5 };
    for (const auto& p : d.params) a.push_back(std::stod(m.r.param_values.at(p)));
    a.insert(a.end(), { 4, 1e-12, 25, 1 });
    m.a = a;
    m.X.assign(m.sys.vars.size(), 0.0);
    for (size_t i = 0; i < m.sys.vars.size(); ++i) {
        auto it = m.r.init_conditions.find(m.sys.vars[i]);
        if (it != m.r.init_conditions.end()) m.X[i] = std::stod(it->second);
    }
    if (!m.r.uic && !m.r.has_ic) {
        int it = 0;
        const bool ok = dae_dc_operating_point(m.sys, m.a.data(), (int)m.a.size(), 0.0, m.X.data(), &it, &err);
        ut::require(ok, std::string(title) + ": DC OP: " + err);
    } else {
        const bool ok = dae_consistent_z(m.sys, m.a.data(), (int)m.a.size(), 0.0, m.X.data(), &err);
        ut::require(ok, std::string(title) + ": z0: " + err);
    }
    const std::string body = codegen_dae_scheme(m.sys, DaeScheme::Radau5);
    KrsCpuStep step;
    std::vector<KrsCpuDiag> diag;
    ut::require(step.compile(body, (int)m.X.size(), (int)m.a.size(), diag), std::string(title) + ": CPU compile");
    std::vector<numb> x(m.X.begin(), m.X.end()), av(m.a.begin(), m.a.end());
    step.step_index(0);
    const long n = std::lround(T / h), every = n / ns;
    std::vector<std::vector<double>> ours;
    ours.push_back(std::vector<double>(x.begin(), x.end()));
    for (long i = 1; i <= n; ++i) {
        step.fn()(x.data(), av.data(), (numb)h);
        if (i % every == 0) ours.push_back(std::vector<double>(x.begin(), x.end()));
    }
    std::vector<double> time;
    std::map<std::string, std::vector<double>> ng;
    ut::require(g_ng.tran(net, tstep, T, h, time, ng, nodes), std::string(title) + ": ngspice run\n" + g_ng.log);
    double worst = 0;
    for (const std::string& nd : nodes) {
        const int k = m.idx("v_" + sanitize_node(nd));
        double dmax = 0, range = 0;
        for (size_t s = 0; s < ours.size(); ++s) {
            const double tq = (double)s * every * h, yn = interp(time, ng[nd], tq), yo = ours[s][(size_t)k];
            dmax = std::fmax(dmax, std::fabs(yo - yn));
            range = std::fmax(range, std::fabs(yn));
        }
        if (std::getenv("UCUDA_TEST_DUMP") && nd == std::getenv("UCUDA_TEST_DUMP"))
            for (int s = 0; s <= 20; ++s) {
                const double tq = T * s / 20;
                std::printf("    t = %.6e  ngspice %.9f\n", tq, interp(time, ng[nd], tq));
            }
        const double rel = dmax / (range + 1e-3);
        worst = std::fmax(worst, rel);
        std::printf("  ngspice vs U-CUDA, %s, V(%s): max |diff| = %.2e V (signal up to %.3f V, rel %.1e)\n", title,
                    nd.c_str(), dmax, range, rel);
    }
    return worst;
}

// Схемы, где всё моделируется: допуск 1e-4 от размаха (оба интегратора точнее, расхождение — модели).
void ngspice_compare() {
    // 1. выпрямитель из демо KiCad (диод по умолчанию): .kicad_sch -> kicad-cli -> оба симулятора
    {
        std::string net, log, err;
        const fs::path p = g_kicad_root / "share" / "kicad" / "demos" / "simulation" / "rectifier" / "rectifier.kicad_sch";
        ut::require(circuit_source_load(p.u8string(), net, &log, &err), "rectifier export: " + err);
        const double w = compare("KiCad rectifier demo", net, { "/rect_out", "net-_d1-a_" }, 1e-6, 4e-3, 1e-6);
        ut::require(w < 1e-4, ut::num("rectifier = ngspice", w, 0, 1e-4));
    }
    // 1b. демо KiCad generic_opamp_bip: встроенный ОУ KiCad (G, R, C, E, ограничительные диоды N = 0.01) и каскад на BJT
    {
        std::string net, log, err;
        const fs::path p = g_kicad_root / "share" / "kicad" / "demos" / "simulation" / "generic_models" / "generic_opamp_bip.kicad_sch";
        ut::require(circuit_source_load(p.u8string(), net, &log, &err), "generic_opamp_bip export: " + err);
        const double w = compare("KiCad generic_opamp_bip demo", net, { "/out", "/outb" }, 2e-7, 3e-3, 1e-6);
        ut::require(w < 1e-3, ut::num("generic_opamp_bip = ngspice", w, 0, 1e-3));
    }
    // 2. каскад ОЭ на BJT с барьерными ёмкостями C(v) и Эрли, синус на входе
    {
        const std::string net = "* ce\nVcc vcc 0 DC 12\nVin in 0 DC 0 SIN(0 20m 10k)\nCin in b 1u\nR1 vcc b 47k\nR2 b 0 10k\n"
            "Rc vcc c 2.2k\nRe e 0 470\nQ1 c b e qn\n.model qn NPN(IS=1e-14 BF=150 BR=2 VAF=80 CJE=8p VJE=0.7 MJE=0.35 "
            "CJC=3p VJC=0.6 MJC=0.4 FC=0.5)\n";
        const double w = compare("BJT CE stage, C(v)", net, { "c", "b", "e" }, 2e-8, 3e-4, 1e-7);
        ut::require(w < 1e-4, ut::num("BJT stage = ngspice", w, 0, 1e-4));
    }
    // 3. MOSFET level 1 с эффектом подложки и перекрытиями, синус на затворе
    {
        const std::string net = "* cs\nVdd vdd 0 DC 5\nVss vss 0 DC -1\nVg g 0 DC 2 SIN(2 0.5 50k)\nRd vdd d 2k\nCl d 0 10p\n"
            "M1 d g 0 vss nm W=20u L=2u\n.model nm NMOS(LEVEL=1 VTO=0.8 KP=100u LAMBDA=0.02 GAMMA=0.4 PHI=0.65 CGSO=1n CGDO=0.5n)\n";
        const double w = compare("MOSFET CS stage", net, { "d" }, 5e-9, 4e-5, 1e-8);
        ut::require(w < 1e-4, ut::num("MOSFET stage = ngspice", w, 0, 1e-4));
    }
    // 4. JFET-повторитель с CGS / CGD
    {
        const std::string net = "* jf\nVdd vdd 0 DC 10\nVg g 0 DC 0 SIN(0 0.5 100k)\nRs s 0 1k\nJ1 vdd g s jn\n"
            ".model jn NJF(VTO=-2 BETA=1m LAMBDA=0.01 CGS=5p CGD=2p PB=0.8 IS=1e-14)\n";
        const double w = compare("JFET follower", net, { "s" }, 2e-9, 2e-5, 1e-8);
        ut::require(w < 1e-4, ut::num("JFET follower = ngspice", w, 0, 1e-4));
    }
    // 5. трансформатор K, компаратор B с ?:, источник SFFM (определение ngspice)
    {
        const std::string net = "* misc\nV1 p 0 DC 0 SIN(0 1 1k)\nR1 p a 10\nL1 a 0 1m\nL2 b 0 4m\nR2 b 0 50\nK1 L1 L2 0.9\n"
            "B1 cmp 0 V = V(p) > 0.3 ? 1 : -1\nR3 cmp f 1k\nC3 f 0 100n\n"
            "V3 fm 0 SFFM(0.2 1 20k 3 2k)\nR5 fm 0 1k\n.tran 1u 2m uic\n";
        const double w = compare("K, B ?:, SFFM", net, { "a", "b", "f", "fm" }, 5e-8, 1e-3, 1e-7);
        ut::require(w < 1e-3, ut::num("K / B / SFFM = ngspice", w, 0, 1e-3));
    }
    // AM: импорт — по руководству ngspice, VA (VO + sin(2 pi MF t)) sin(2 pi FC t). ngspice из KiCad 10.0.6 на
    // AM(..) выдаёт постоянную VA при t > 0 — сверять не с чем; печатается, чтобы заметить, если это изменится.
    {
        std::vector<double> time;
        std::map<std::string, std::vector<double>> v;
        if (g_ng.tran("* am\nV2 x 0 AM(1 0 1k 10k 0)\nR4 x 0 1k\n", 1e-6, 1e-4, 1e-7, time, v, { "x" })) {
            double lo = 1e300, hi = -1e300;
            for (size_t k = 1; k < time.size(); ++k) { lo = std::fmin(lo, v["x"][k]); hi = std::fmax(hi, v["x"][k]); }
            std::printf("  ngspice AM(1 0 1k 10k 0) for t > 0: %.6f .. %.6f (manual formula: -0.31 .. 0.31) - %s\n", lo, hi,
                        hi - lo < 1e-9 ? "constant VA in this ngspice build, AM not compared" : "now modulated: compare AM");
        }
    }
}

void survey() {
    const fs::path demos = g_kicad_root / "share" / "kicad" / "demos" / "simulation";
    std::vector<fs::path> sch;
    for (const auto& e : fs::recursive_directory_iterator(demos))
        if (e.path().extension() == ".kicad_sch") sch.push_back(e.path());
    std::sort(sch.begin(), sch.end());
    int ok = 0;
    for (const fs::path& p : sch) {
        std::string net, log, err;
        const std::string name = fs::relative(p, demos).u8string();
        if (!circuit_source_load(p.u8string(), net, &log, &err)) {
            std::printf("  %-48s export failed: %s\n", name.c_str(), err.c_str());
            continue;
        }
        SpiceImportOptions o;
        o.base_dir = p.parent_path().u8string();
        SpiceImportResult r;
        if (!spice_import(net, o, r, &err)) {
            std::printf("  %-48s refused: %s\n", name.c_str(), err.c_str());
            continue;
        }
        ++ok;
        std::printf("  %-48s ok: %s\n", name.c_str(), r.summary.c_str());
        for (const auto& n : r.notes) std::printf("      note: %s\n", n.c_str());
    }
    std::printf("  %d of %zu demo schematics imported\n", ok, sch.size());
}

}  // namespace

int main() {
    return ut::run_main("kicad", []() -> int {
        const std::string cli = kicad_find_cli();
        if (cli.empty()) { std::printf("no KiCad (kicad-cli.exe): skipped\n"); return ut::kSkip; }
        g_kicad_root = fs::u8path(cli).parent_path().parent_path();
        std::printf("  kicad-cli: %s\n", cli.c_str());
        survey();
        g_ng_ok = g_ng.load(g_kicad_root / "bin" / "ngspice.dll");
        if (!g_ng_ok) { std::printf("  ngspice.dll not loaded: comparison skipped\n"); return 0; }
        // Отладка: UCUDA_NG_PROBE=<netlist> UCUDA_NG_NODE=<узел> UCUDA_NG_T=<tstop> — узел по ngspice на 40 точках.
        if (const char* pf = std::getenv("UCUDA_NG_PROBE")) {
            std::ifstream f(pf);
            std::stringstream ss;
            ss << f.rdbuf();
            const double T = std::atof(std::getenv("UCUDA_NG_T"));
            std::vector<double> time;
            std::map<std::string, std::vector<double>> v;
            const std::string node = std::getenv("UCUDA_NG_NODE");
            ut::require(g_ng.tran(ss.str(), T / 1000, T, T / 4000, time, v, { node }), "probe:\n" + g_ng.log);
            const int np = std::getenv("UCUDA_NG_N") ? std::atoi(std::getenv("UCUDA_NG_N")) : 40;
            for (int s = 0; s <= np; ++s) std::printf("%.9e %.12f\n", T * s / np, interp(time, v[node], T * s / np));
            return 0;
        }
        ngspice_compare();
        return 0;
    });
}
