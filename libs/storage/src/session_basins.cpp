#include "session_json_internal.h"

using namespace ucuda::storage_detail;


std::string session_to_json_basins(const BasinsAnalysisSession& s) {
    std::ostringstream o;
    o << "{\n";
    o << "  \"active_config_index\":" << s.active_config_index << ",\n";
    o << "  \"configs\":[";
    for (size_t i = 0; i < s.configs.size(); ++i) {
        if (i) o << ",";
        o << "\n    ";
        const BasinsPhaseSlot* slot =
            (i < s.phase_slots.size() && s.phase_slots[i]) ? s.phase_slots[i].get() : nullptr;
        write_basins_config(o, s.configs[i], slot);
    }
    if (!s.configs.empty()) o << "\n  ";
    o << "]\n";
    o << "}\n";
    return o.str();
}


bool session_from_json_basins(const std::string& json, BasinsAnalysisSession& s) {
    try {
        JP p(json);
        p.expect('{');
        if (p.opt('}')) return true;

        // Legacy (старые _last_basins.json без ключа "configs") — собираем
        // плоские поля в один config. Если встречается ключ "configs" —
        // переключаемся на новый формат и legacy-buffer отбрасываем.
        BasinsConfig legacy;
        bool legacy_has_fields = false;
        bool new_format = false;
        // Проекции фазовых портретов читаются в отдельный буфер: они живут не
        // в config'е, а в parallel-слоте, который существует только после
        // ensure_phase_slots() (т.е. после того, как все configs разобраны).
        std::vector<std::vector<Projection>> phase_projs;

        while (true) {
            std::string key = p.str();
            p.expect(':');
            if (key == "configs") {
                new_format = true;
                s.configs.clear();
                p.expect('[');
                if (!p.opt(']')) {
                    while (true) {
                        p.expect('{');
                        BasinsConfig bc;
                        std::vector<Projection> bc_projs;
                        if (!p.opt('}')) {
                            while (true) {
                                std::string k2 = p.str(); p.expect(':');
                                if (k2 == "phase_projections")
                                    read_basins_phase_projections(p, bc_projs);
                                else if (!read_basins_field(p, bc, k2))
                                    p.skip_value();
                                if (p.opt(',')) continue;
                                p.expect('}'); break;
                            }
                        }
                        s.configs.push_back(std::move(bc));
                        phase_projs.push_back(std::move(bc_projs));
                        if (p.opt(',')) continue;
                        p.expect(']'); break;
                    }
                }
            }
            else if (key == "active_config_index") {
                s.active_config_index = std::stoi(p.str_or_num());
            }
            else if (read_basins_field(p, legacy, key)) {
                legacy_has_fields = true;
            }
            else {
                p.skip_value();
            }
            if (p.opt(',')) continue;
            p.expect('}'); break;
        }

        if (!new_format && legacy_has_fields) {
            // Старое сохранение: оборачиваем в один config (заменяет
            // дефолт, который положил load_from_record).
            if (legacy.label.empty()) legacy.label = "Basins 1";
            s.configs.clear();
            s.configs.push_back(std::move(legacy));
        }

        if (s.configs.empty()) {
            // JSON был пустой или только non-config поля — оставим хотя бы
            // один config (load_from_record уже положил дефолт).
            s.ensure_phase_slots();
            return true;
        }
        if (s.active_config_index < 0 ||
            s.active_config_index >= (int)s.configs.size()) {
            s.active_config_index = 0;
        }
        s.running_config_index = -1;

        // Слоты портретов подгоняем под новый список configs и раскладываем
        // сохранённые окна. Пустой список (старая сессия без ключа) оставляет
        // дефолт из ensure_phase_slots — одно окно Phase 2D.
        s.phase_slots.clear();
        s.phase_queue.clear();
        s.ensure_phase_slots();
        for (size_t i = 0; i < phase_projs.size() && i < s.phase_slots.size(); ++i) {
            if (phase_projs[i].empty()) continue;
            s.phase_slots[i]->phase.projections = std::move(phase_projs[i]);
        }
        return true;
    }
    catch (...) {
        return false;
    }
}
