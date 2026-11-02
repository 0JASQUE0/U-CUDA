#include "ucuda/application/app_model.h"
#include "ucuda/equations/codegen.hpp"
#include "ucuda/domain/num_parse.h"   // fmt_num_shortest — round-trip формат для пинов узлов
#include "ucuda/application/session_persistence.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "ucuda/equations/sysparse.hpp"
#include "ucuda/equations/dae_ir.h"       // build_system для ДАУ
#include "ucuda/equations/dae_devices.h"  // dae_dc_operating_point
#include "ucuda/numerics/kicad_import.h"
#include "ucuda/numerics/spice_parse.h"  // import_circuit
#include <filesystem>
#include <stdexcept>


// Peak-knobs: число -> текст буфера ввода

namespace {
// Печатает значение так, чтобы оно (а) читалось человеком и (б) парсилось обратно ровно в то же
// double. %.6g даёт "1e-14" вместо "1.000e-14", но для значений, не влезающих в шесть значащих
// цифр, молча соврал бы — поэтому повышаем точность, пока round-trip не совпадёт. %.17g —
// гарантированный потолок для double.
std::string fmt_peak_num(double v) {
    char buf[64];
    for (int prec : {6, 9, 17}) {
        std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
        if (std::strtod(buf, nullptr) == v) break;
    }
    return std::string(buf);
}
} // namespace

void AppSettings::sync_peak_text() {
    peak_eps_fixed_point_text     = fmt_peak_num(peak.eps_fixed_point);
    peak_eps_peak_delta_text      = fmt_peak_num(peak.eps_peak_delta);
    peak_eps_interPeak_delta_text = fmt_peak_num(peak.eps_interPeak_delta);
    peak_threshold_text           = fmt_peak_num(peak.peak_threshold);
    peak_max_amount_text          = std::to_string(peak.max_amount_of_peaks);
}
