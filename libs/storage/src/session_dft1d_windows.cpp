#include "session_json_internal.h"

using namespace ucuda::storage_detail;


std::string session_to_json_dft1d_windows(const std::vector<Dft1DPlotWindow>& wins) {
    std::ostringstream o;
    o << "{\n  \"windows\":[";
    for (size_t i = 0; i < wins.size(); ++i) {
        if (i) o << ",";
        const auto& w = wins[i];
        o << "{\"id\":" << w.id;
        o << ",\"label\":"; jstr(o, w.label);
        o << ",\"label_is_manual\":" << (w.label_is_manual ? "true" : "false");
        o << ",\"members\":[";
        for (size_t k = 0; k < w.members.size(); ++k) { if (k) o << ","; o << w.members[k]; }
        o << "]}";
    }
    o << "]\n}\n";
    return o.str();
}


bool session_from_json_dft1d_windows(const std::string& json, std::vector<Dft1DPlotWindow>& wins) {
    try {
        JP p(json);
        p.expect('{');
        if (p.opt('}')) { wins.clear(); return true; }
        while (true) {
            std::string key = p.str();
            p.expect(':');
            if (key == "windows") {
                wins.clear();
                p.expect('[');
                if (!p.opt(']')) {
                    while (true) {
                        p.expect('{');
                        Dft1DPlotWindow w;
                        if (!p.opt('}')) {
                            while (true) {
                                std::string k = p.str(); p.expect(':');
                                if      (k == "id")               w.id               = std::stoi(p.str_or_num());
                                else if (k == "label")            w.label            = p.str();
                                else if (k == "label_is_manual")  w.label_is_manual  = p.boolean();
                                else if (k == "members") {
                                    w.members.clear();
                                    p.expect('[');
                                    if (!p.opt(']')) {
                                        while (true) {
                                            w.members.push_back(std::stoi(p.str_or_num()));
                                            if (p.opt(',')) continue;
                                            p.expect(']'); break;
                                        }
                                    }
                                }
                                else p.skip_value();
                                if (p.opt(',')) continue;
                                p.expect('}'); break;
                            }
                        }
                        wins.push_back(std::move(w));
                        if (p.opt(',')) continue;
                        p.expect(']'); break;
                    }
                }
            }
            else p.skip_value();
            if (p.opt(',')) continue;
            p.expect('}'); break;
        }
        return true;
    }
    catch (...) {
        return false;
    }
}
