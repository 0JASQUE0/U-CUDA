#include "session_json_internal.h"

using namespace ucuda::storage_detail;


std::string session_to_json_dft1d(const Dft1DAnalysisSession& s) {
    std::ostringstream o;
    o << "{\n";
    o << "  \"active_config_index\":" << s.active_config_index << ",\n";
    o << "  \"configs\":[";
    for (size_t i = 0; i < s.configs.size(); ++i) {
        if (i) o << ",";
        o << "\n    ";
        write_dft1d_config(o, s.configs[i]);
    }
    if (!s.configs.empty()) o << "\n  ";
    o << "]\n";
    o << "}\n";
    return o.str();
}


bool session_from_json_dft1d(const std::string& json, Dft1DAnalysisSession& s) {
    try {
        JP p(json);
        p.expect('{');
        if (p.opt('}')) return true;
        while (true) {
            std::string key = p.str();
            p.expect(':');
            if (key == "configs") {
                s.configs.clear();
                p.expect('[');
                if (!p.opt(']')) {
                    while (true) {
                        p.expect('{');
                        Dft1DConfig c;
                        if (!p.opt('}')) {
                            while (true) {
                                std::string k2 = p.str(); p.expect(':');
                                if (!read_dft1d_field(p, c, k2)) p.skip_value();
                                if (p.opt(',')) continue;
                                p.expect('}'); break;
                            }
                        }
                        s.configs.push_back(std::move(c));
                        if (p.opt(',')) continue;
                        p.expect(']'); break;
                    }
                }
            }
            else if (key == "active_config_index") s.active_config_index = std::stoi(p.str_or_num());
            else                                    p.skip_value();
            if (p.opt(',')) continue;
            p.expect('}'); break;
        }
        if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
            s.active_config_index = 0;
        s.running_config_index = -1;
        return true;
    } catch (...) {
        return false;
    }
}
