#include "ucuda/application/compute_services.h"
#include "ucuda/analysis/analysis_session.h"
#include "ucuda/cuda/nvrtc_engine.h"
#include "ucuda/cuda/runtime_settings.h"
#include "ucuda/numerics/circuit_netlist.h"
#include "ucuda/numerics/kicad_import.h"
#include "ucuda/numerics/krs_cpu.h"
#include <fstream>

bool cpu_compiler_available(std::string* why_not) {
    return krs_cpu_backend_available(why_not);
}

std::vector<BodyIndexDiag> check_body_indices(const std::string& body, int amount_of_x, int amount_of_values) {
    std::vector<KrsCpuDiag> diags;
    krs_cpu_check_indices(body, amount_of_x, amount_of_values, diags);
    std::vector<BodyIndexDiag> out;
    out.reserve(diags.size());
    for (const auto& d : diags) out.push_back({ d.line, d.message });
    return out;
}

bool check_controller_body(const std::string& body, std::string& log) {
    return nvrtc_check_ctrl_body(body, log);
}

void apply_peak_config(const PeakConfig& c) { set_peak_config(c); }
void apply_nvrtc_fmad(bool enabled)          { set_nvrtc_fmad(enabled); }
void apply_nvrtc_rdc(bool enabled)           { set_nvrtc_rdc(enabled); }
void apply_gpu_block_size(int threads)       { set_gpu_block_size(threads); }

long long circuit_file_mtime(const std::string& path) {
    return circuit_source_mtime(path);
}

void write_synthesized_netlist(PhaseAnalysisSession& s, const std::string& path) {
    NetlistOptions no;
    no.title      = "U-CUDA synthesized circuit";
    no.var_names  = s.vars;
    no.rhs_text   = s.sys.rhs;
    no.scale      = s.result.circuit_scale;
    no.x0_volts   = s.result.circuit_x0;
    no.opamp      = s.result.circuit_opamp;
    no.h_ode      = s.result.circuit_h_ode;
    no.t_end_ode  = s.result.circuit_t_end_ode;
    no.dialect    = s.circuit_netlist_poly ? NetlistDialect::Poly : NetlistDialect::Behavioral;

    const std::string text = emit_spice_netlist(s.result.circuit_graph, no);
    std::ofstream f(path, std::ios::binary);
    if (f) { f << text; s.result.circuit_status = "netlist written to " + path; }
    else     s.result.circuit_status = "cannot write " + path;
}
