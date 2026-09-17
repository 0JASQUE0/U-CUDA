#include "session_json_internal.h"

using namespace ucuda::storage_detail;


// Parametric plot windows: a flat array of {id, kind, mode_2d, label,
// members[]}, modeled directly on the "projections" block above.
std::string session_to_json_parametric_windows(const std::vector<ParametricPlotWindow>& wins) {
    std::ostringstream o;
    o << "{\n  \"windows\":[";
    for (size_t i = 0; i < wins.size(); ++i) {
        if (i) o << ",";
        const auto& w = wins[i];
        o << "{\"id\":" << w.id;
        o << ",\"kind\":" << (int)w.kind;
        o << ",\"mode_2d\":" << (w.mode_2d ? "true" : "false");
        o << ",\"colored_1d\":" << (w.colored_1d ? "true" : "false");
        o << ",\"label\":"; jstr(o, w.label);
        o << ",\"label_is_manual\":" << (w.label_is_manual ? "true" : "false");
        o << ",\"members\":[";
        for (size_t k = 0; k < w.members.size(); ++k) { if (k) o << ","; o << w.members[k]; }
        o << "]}";
    }
    o << "]\n}\n";
    return o.str();
}


bool session_from_json_parametric_windows(const std::string& json, std::vector<ParametricPlotWindow>& wins) {
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
                        ParametricPlotWindow w;
                        if (!p.opt('}')) {
                            while (true) {
                                std::string k = p.str(); p.expect(':');
                                if      (k == "id")      w.id      = std::stoi(p.str_or_num());
                                else if (k == "kind")    w.kind    = (ParametricPlotWindow::Kind)std::stoi(p.str_or_num());
                                else if (k == "mode_2d") w.mode_2d = p.boolean();
                                else if (k == "colored_1d") w.colored_1d = p.boolean();
                                else if (k == "label")   w.label   = p.str();
                                else if (k == "label_is_manual") w.label_is_manual = p.boolean();
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
