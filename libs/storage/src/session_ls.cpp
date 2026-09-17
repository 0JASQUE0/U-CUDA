#include "session_json_internal.h"

using namespace ucuda::storage_detail;


 // namespace

std::string session_to_json_ls(const LyapunovSpectrumAnalysisSession& s) {
    std::ostringstream o;
    o << "{\n";
    o << "  \"active_curve_index\":" << s.active_curve_index << ",\n";
    o << "  \"curves\":[";
    for (size_t i = 0; i < s.curves.size(); ++i) {
        if (i) o << ",";
        o << "\n    ";
        write_ls_curve(o, s.curves[i]);
    }
    if (!s.curves.empty()) o << "\n  ";
    o << "]\n";
    o << "}\n";
    return o.str();
}


bool session_from_json_ls(const std::string& json, LyapunovSpectrumAnalysisSession& s) {
    try {
        JP p(json);
        p.expect('{');
        if (p.opt('}')) return true;
        while (true) {
            std::string key = p.str();
            p.expect(':');
            if (key == "curves") {
                s.curves.clear();
                p.expect('[');
                if (!p.opt(']')) {
                    while (true) {
                        p.expect('{');
                        LSCurveConfig c;
                        if (!p.opt('}')) {
                            while (true) {
                                std::string k2 = p.str(); p.expect(':');
                                if (!read_ls_curve_field(p, c, k2)) p.skip_value();
                                if (p.opt(',')) continue;
                                p.expect('}'); break;
                            }
                        }
                        s.curves.push_back(std::move(c));
                        if (p.opt(',')) continue;
                        p.expect(']'); break;
                    }
                }
            }
            else if (key == "active_curve_index") {
                s.active_curve_index = std::stoi(p.str_or_num());
            }
            else {
                p.skip_value();
            }
            if (p.opt(',')) continue;
            p.expect('}'); break;
        }
        if (s.curves.empty()) s.add_curve();
        if (s.active_curve_index < 0 || s.active_curve_index >= (int)s.curves.size())
            s.active_curve_index = 0;
        s.running_curve_index = -1;
        return true;
    }
    catch (...) {
        return false;
    }
}
