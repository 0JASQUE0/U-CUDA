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


bool AppModel::start_phase_analysis() {
    // распарсить систему -> known_vars/known_params
    if (!refresh_symbols()) return false;
    SystemRecord r = to_record();
    phase_session.load_from_record(r, known_vars, known_params);
    // передать уравнения в сессию и сгенерировать КРС для NVRTC
    try {
        phase_session.sys = build_system();
        // ДАУ, сведённая к ОДУ: z теперь в состоянии (System::out_exprs), наблюдаемые вне
        // состояния ей не нужны.
        phase_session.observables.clear();
        phase_session.obs_exprs.clear();
        // метод по умолчанию — из текущего выбора схемы в модели, если задан
        phase_session.regenerate_krs();
    }
    catch (...) {
        // если система ещё неполна — КРС останется пустой, recompute покажет ошибку
    }
    phase_session.loaded_system_name = name;
    return true;
}

bool AppModel::start_parametric_analysis() {
    if (!refresh_symbols()) return false;
    SystemRecord r = to_record();
    bifurcation_session.load_from_record(r, known_vars, known_params);
    lle_session.load_from_record(r, known_vars, known_params);
    ls_session.load_from_record(r, known_vars, known_params);
    metrics_session.load_from_record(r, known_vars, known_params);
    try {
        // sys общий — всем трём сессиям нужен для compute_krs_for_scheme.
        // (KRS резолвится per-«прогон» в момент Run.)
        System built = build_system();
        bifurcation_session.sys = built;
        lle_session.sys         = built;
        ls_session.sys          = built;
        metrics_session.sys     = built;
    }
    catch (...) {
        // система неполна — Run покажет ошибку
    }
    bifurcation_session.loaded_system_name = name;
    lle_session.loaded_system_name         = name;
    ls_session.loaded_system_name          = name;
    metrics_session.loaded_system_name     = name;
    ++session_epoch[(int)AppMode::Parametric];
    return true;
}

bool AppModel::start_dft1d_analysis() {
    if (!refresh_symbols()) return false;
    SystemRecord r = to_record();
    dft1d_session.load_from_record(r, known_vars, known_params);
    try {
        System built = build_system();
        dft1d_session.sys = built;
    }
    catch (...) {}
    dft1d_session.loaded_system_name = name;
    ++session_epoch[(int)AppMode::Dft1D];
    return true;
}

bool AppModel::start_basins_analysis() {
    if (!refresh_symbols()) return false;
    SystemRecord r = to_record();
    basins_session.load_from_record(r, known_vars, known_params);
    try {
        System built = build_system();
        basins_session.sys = built;
    }
    catch (...) {}
    basins_session.loaded_system_name = name;
    ++session_epoch[(int)AppMode::Basins];
    return true;
}

bool AppModel::start_order_analysis() {
    if (!refresh_symbols()) return false;
    SystemRecord r = to_record();
    order_session.load_from_record(r, known_vars, known_params);
    try {
        System built = build_system();
        order_session.sys = built;
    }
    catch (...) {}
    order_session.loaded_system_name = name;
    ++session_epoch[(int)AppMode::Order];
    return true;
}

bool AppModel::start_network_analysis() {
    if (!refresh_symbols()) return false;
    SystemRecord r = to_record();
    network_session.load_from_record(r, known_vars, known_params);
    try {
        System built = build_system();
        network_session.sys = built;
    }
    catch (...) {}
    network_session.loaded_system_name = name;
    ++session_epoch[(int)AppMode::Network];
    return true;
}

bool AppModel::start_neural_analysis() {
    if (!refresh_symbols()) return false;
    SystemRecord r = to_record();
    neural_session.load_from_record(r, known_vars, known_params);
    try {
        System built = build_system();
        neural_session.sys = built;
    }
    catch (...) {}
    neural_session.loaded_system_name = name;
    ++session_epoch[(int)AppMode::Neural];
    return true;
}

bool AppModel::start_fastsync_analysis() {
    if (!refresh_symbols()) return false;
    SystemRecord r = to_record();
    fastsync_session.load_from_record(r, known_vars, known_params);
    try {
        System built = build_system();
        fastsync_session.sys = built;
    }
    catch (...) {}
    fastsync_session.loaded_system_name = name;
    ++session_epoch[(int)AppMode::FastSync];
    return true;
}

bool AppModel::start_custom_analysis() {
    if (!refresh_symbols()) return false;
    SystemRecord r = to_record();
    custom_session.load_from_record(r, known_vars, known_params);
    try {
        System built = build_system();
        custom_session.sys = built;
        // Each owned sub-session also needs the built System for KRS codegen.
        custom_session.bif_session.sys    = built;
        custom_session.lle_session.sys    = built;
        custom_session.ls_session.sys     = built;
        custom_session.phase_session.sys  = built;
        custom_session.basins_session.sys = built;
        custom_session.phase_session.regenerate_krs();
    }
    catch (...) {
        // system incomplete — next Run will surface the error.
    }
    custom_session.loaded_system_name        = name;
    custom_session.bif_session.loaded_system_name    = name;
    custom_session.lle_session.loaded_system_name    = name;
    custom_session.ls_session.loaded_system_name     = name;
    custom_session.phase_session.loaded_system_name  = name;
    custom_session.basins_session.loaded_system_name = name;
    return true;
}


void AppModel::propagate_to_sessions() {
    // refresh_symbols может упасть, если в полях ещё что-то невалидное —
    // молча игнорим, оставим прежние vars/params, пользователь увидит ошибку
    // на следующем Run.
    refresh_symbols();

    // custom_schemes — глубокая копия. Тот же sync, что и в draw_gui каждый
    // кадр, но здесь он гарантирован сразу после Save (важно, если save
    // вызвали, не открывая другие вкладки — кеш-ключ NVRTC по new body
    // должен быть готов).
    phase_session.custom_schemes       = custom_schemes;
    bifurcation_session.custom_schemes = custom_schemes;
    lle_session.custom_schemes         = custom_schemes;
    ls_session.custom_schemes          = custom_schemes;
    metrics_session.custom_schemes     = custom_schemes;
    dft1d_session.custom_schemes       = custom_schemes;
    basins_session.custom_schemes      = custom_schemes;
    fastsync_session.custom_schemes    = custom_schemes;
    custom_session.custom_schemes                = custom_schemes;
    custom_session.bif_session.custom_schemes    = custom_schemes;
    custom_session.lle_session.custom_schemes    = custom_schemes;
    custom_session.ls_session.custom_schemes     = custom_schemes;
    custom_session.phase_session.custom_schemes  = custom_schemes;
    custom_session.basins_session.custom_schemes = custom_schemes;

    // Экстраполяционные обёртки едут тем же маршрутом: в сессии нужны только
    // имена, тело каждый раз пересобирает compute_krs_for_scheme.
    // Во вкладки уходят только включённые (галочка в списке обёрток).
    const std::vector<std::string> wrappers_on = enabled_wrapper_schemes();
    phase_session.wrapper_schemes       = wrappers_on;
    bifurcation_session.wrapper_schemes = wrappers_on;
    lle_session.wrapper_schemes         = wrappers_on;
    ls_session.wrapper_schemes          = wrappers_on;
    metrics_session.wrapper_schemes     = wrappers_on;
    dft1d_session.wrapper_schemes       = wrappers_on;
    basins_session.wrapper_schemes      = wrappers_on;
    fastsync_session.wrapper_schemes    = wrappers_on;
    custom_session.wrapper_schemes                = wrappers_on;
    custom_session.bif_session.wrapper_schemes    = wrappers_on;
    custom_session.lle_session.wrapper_schemes    = wrappers_on;
    custom_session.ls_session.wrapper_schemes     = wrappers_on;
    custom_session.phase_session.wrapper_schemes  = wrappers_on;
    custom_session.basins_session.wrapper_schemes = wrappers_on;

    // Mirror enabled built-in schemes from the current model into every session.
    const std::vector<std::string> enabled_now = enabled_builtins_from_record(to_record());
    phase_session.enabled_builtin_schemes       = enabled_now;
    bifurcation_session.enabled_builtin_schemes = enabled_now;
    lle_session.enabled_builtin_schemes         = enabled_now;
    ls_session.enabled_builtin_schemes          = enabled_now;
    metrics_session.enabled_builtin_schemes     = enabled_now;
    dft1d_session.enabled_builtin_schemes       = enabled_now;
    basins_session.enabled_builtin_schemes      = enabled_now;
    fastsync_session.enabled_builtin_schemes    = enabled_now;
    custom_session.enabled_builtin_schemes                = enabled_now;
    custom_session.bif_session.enabled_builtin_schemes    = enabled_now;
    custom_session.lle_session.enabled_builtin_schemes    = enabled_now;
    custom_session.ls_session.enabled_builtin_schemes     = enabled_now;
    custom_session.phase_session.enabled_builtin_schemes  = enabled_now;
    custom_session.basins_session.enabled_builtin_schemes = enabled_now;
    order_session.enabled_builtin_schemes                 = enabled_now;
    network_session.enabled_builtin_schemes               = enabled_now;
    neural_session.enabled_builtin_schemes                = enabled_now;

    // sys обновляем для built-in схем (Euler/RK4/...): они используют
    // sys.rhs внутри compute_krs_for_scheme → codegen_scheme. Если уравнения
    // в Library поправили, без этой пересборки старый sys остаётся в сессиях
    // до следующего start_*_analysis.
    try {
        System built = build_system();
        phase_session.sys       = built;
        bifurcation_session.sys = built;
        lle_session.sys         = built;
        ls_session.sys          = built;
        metrics_session.sys     = built;
        dft1d_session.sys       = built;
        basins_session.sys      = built;
        fastsync_session.sys    = built;
        neural_session.sys      = built;
        custom_session.sys                = built;
        custom_session.bif_session.sys    = built;
        custom_session.lle_session.sys    = built;
        custom_session.ls_session.sys     = built;
        custom_session.phase_session.sys  = built;
        custom_session.basins_session.sys = built;
    }
    catch (...) {
        // система ещё неполна — следующий Run покажет ошибку парсинга.
    }

    // Phase кеширует krs_code. Принудительно инвалидируем — snapshot_phase
    // на ближайшем Run пересчитает его от свежего sys / custom_schemes.
    phase_session.krs_code.clear();
    phase_session.krs_error.clear();
}
