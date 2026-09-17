#include "session_json_internal.h"

using namespace ucuda::storage_detail;


std::string session_to_json_parametric(const BifurcationAnalysisSession& s) {
    std::ostringstream o;
    o << "{\n";
    o << "  \"active_diagram_index\":" << s.active_diagram_index << ",\n";
    o << "  \"diagrams\":[";
    for (size_t i = 0; i < s.diagrams.size(); ++i) {
        if (i) o << ",";
        o << "\n    ";
        write_diagram(o, s.diagrams[i]);
    }
    if (!s.diagrams.empty()) o << "\n  ";
    o << "]\n";
    o << "}\n";
    return o.str();
}


bool session_from_json_parametric(const std::string& json, BifurcationAnalysisSession& s) {
    try {
        JP p(json);
        p.expect('{');
        if (p.opt('}')) return true;

        // Legacy (старые _last_parametric.json без ключа "diagrams") — собираем
        // плоские поля в одну БД. Если встречается ключ "diagrams" — переключаемся
        // на новый формат и старый buffer отбрасываем.
        BifurcationDiagramConfig legacy;
        bool legacy_has_fields = false;
        bool new_format = false;

        while (true) {
            std::string key = p.str();
            p.expect(':');
            if (key == "diagrams") {
                new_format = true;
                s.diagrams.clear();
                p.expect('[');
                if (!p.opt(']')) {
                    while (true) {
                        p.expect('{');
                        BifurcationDiagramConfig bd;
                        if (!p.opt('}')) {
                            while (true) {
                                std::string k2 = p.str(); p.expect(':');
                                if (!read_diagram_field(p, bd, k2)) p.skip_value();
                                if (p.opt(',')) continue;
                                p.expect('}'); break;
                            }
                        }
                        s.diagrams.push_back(std::move(bd));
                        if (p.opt(',')) continue;
                        p.expect(']'); break;
                    }
                }
            }
            else if (key == "active_diagram_index") {
                s.active_diagram_index = std::stoi(p.str_or_num());
            }
            else if (read_diagram_field(p, legacy, key)) {
                legacy_has_fields = true;
            }
            else {
                p.skip_value();
            }
            if (p.opt(',')) continue;
            p.expect('}'); break;
        }

        if (!new_format && legacy_has_fields) {
            // Старое сохранение: оборачиваем в одну БД и заменяем существующий
            // дефолт (load_from_record уже положил BD 1 — заменим его).
            if (legacy.label.empty()) legacy.label = "BD 1";
            s.diagrams.clear();
            s.diagrams.push_back(std::move(legacy));
        }

        if (s.diagrams.empty()) {
            // Сейв был пустой — добавим хотя бы дефолтную БД.
            s.add_diagram();
        }
        if (s.active_diagram_index < 0 || s.active_diagram_index >= (int)s.diagrams.size())
            s.active_diagram_index = 0;
        s.running_diagram_index = -1;
        return true;
    }
    catch (...) {
        return false;
    }
}
