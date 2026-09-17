#include "session_json_internal.h"

using namespace ucuda::storage_detail;


std::string session_to_json_fastsync(const FastSyncAnalysisSession& s) {
    std::ostringstream o;
    o << "{\n";
    o << "  \"active_config_index\":" << s.active_config_index << ",\n";
    o << "  \"configs\":[";
    for (size_t i = 0; i < s.configs.size(); ++i) {
        if (i) o << ",";
        o << "\n    ";
        write_fastsync_config(o, s.configs[i]);
    }
    if (!s.configs.empty()) o << "\n  ";
    o << "]\n";
    o << "}\n";
    return o.str();
}


bool session_from_json_fastsync(const std::string& json, FastSyncAnalysisSession& s) {
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
                        FastSyncConfig fc;
                        // Сентинел, чтобы отличить "ключа в файле не было" от
                        // "в файле записан 0" — см. миграцию ниже.
                        fc.transient_slave_text.clear();
                        if (!p.opt('}')) {
                            while (true) {
                                std::string k2 = p.str(); p.expect(':');
                                if (!read_fastsync_field(p, fc, k2)) p.skip_value();
                                if (p.opt(',')) continue;
                                p.expect('}'); break;
                            }
                        }
                        if (fc.transient_slave_text.empty()) {
                            // Сессия старше раздельных TT: единственный transient досаживал
                            // ФИКСИРОВАННУЮ сторону, а не мастера — при снятой галочке сетка свипует
                            // мастера, значит TT относился к слейву. Переносим, иначе старая карта
                            // пересчиталась бы с другим смыслом. Режим On Attractor не трогаем: там
                            // transient_text досаживает master-траекторию и всегда значил это.
                            if (fc.mode == 1 && !fc.grid_swap_master_slave) {
                                fc.transient_slave_text = fc.transient_text;
                                fc.transient_text       = "0";
                            } else {
                                fc.transient_slave_text = "0";
                            }
                        }
                        s.configs.push_back(std::move(fc));
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
