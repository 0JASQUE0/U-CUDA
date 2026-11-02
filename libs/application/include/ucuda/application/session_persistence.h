#pragma once
// Сохранение сессий, которые владеют другими сессиями (слоты портретов Basins, под-сессии
// Custom). storage видит только состояния (*State) и получает проекции слотов и под-состояния
// явно; здесь — те же функции над сессией целиком, с прежними именами.
#include "ucuda/analysis/analysis_session.h"
#include "ucuda/analysis/custom_session.h"
#include "ucuda/storage/session_io.h"

inline std::string session_to_json_basins(const BasinsAnalysisSession& s) {
    return basins_state_to_json(s, s.phase_projections());
}

inline void apply_basins_slots(BasinsAnalysisSession& s, BasinsSlotsAfterLoad slots,
                               std::vector<std::vector<Projection>>&& projs) {
    if (slots == BasinsSlotsAfterLoad::Ensure) s.ensure_phase_slots();
    else if (slots == BasinsSlotsAfterLoad::Reset) s.restore_phase_slots(std::move(projs));
}

inline bool session_from_json_basins(const std::string& json, BasinsAnalysisSession& s) {
    std::vector<std::vector<Projection>> projs;
    BasinsSlotsAfterLoad slots = BasinsSlotsAfterLoad::Keep;
    if (!basins_state_from_json(json, s, projs, slots)) return false;
    apply_basins_slots(s, slots, std::move(projs));
    return true;
}

inline std::string session_to_json_custom(const CustomSession& s) {
    const CustomSubStates sub{ s.bif_session, s.lle_session, s.ls_session, s.phase_session,
                               s.basins_session, s.basins_session.phase_projections() };
    return custom_state_to_json(s, sub);
}

inline bool session_from_json_custom(const std::string& json, CustomSession& s) {
    CustomSubStatesMut sub{ s.bif_session, s.lle_session, s.ls_session, s.phase_session,
                            s.basins_session, {}, BasinsSlotsAfterLoad::Keep };
    const bool ok = custom_state_from_json(json, s, sub);
    // Как раньше: слоты применяются, даже если остальной разбор потом сорвался.
    apply_basins_slots(s.basins_session, sub.basins_slots, std::move(sub.basins_phase_projections));
    return ok;
}
