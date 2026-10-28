// Архитектура без GUI (порт смысла architecture_test из CMake-форка на API U-CUDA):
//  - старый JSON сессии грузится, недостающие поля — по умолчанию; круг JSON -> сессия -> JSON сохраняет
//    стиль проекций и настройки адаптивного шага (Phase и Parametric);
//  - удаление диаграммы чистит и сдвигает очередь Parametric; неверные элементы очереди DFT выбрасываются;
//  - отмена ставит токен;
//  - запись системы: старый формат (alphabet_text, без kind) и ДАУ (kind, alg_text, индексы, выключенные
//    методы, alpha уровня C) переживают круг через JSON.
// Линкуется со всем приложением (сессии тянут типы окон графиков), поэтому собирается с /DUT_FULL_APP.
#include "test_util.h"
#include "app_model.h"
#include "session_io.h"
#include "system_library.h"
#include <atomic>
#include <memory>

namespace {

void session_compatibility() {
    PhaseAnalysisSession loaded;
    const std::string old_json =
        "{\"step_h\":\"0.02\",\"sim_time\":\"12\","
        "\"projections\":[{\"label\":\"legacy\",\"type\":0,"
        "\"ax\":1,\"ay\":2,\"az\":0,\"show_var\":[true,false]}]}";
    ut::require(session_from_json(old_json, loaded), "old phase session did not load");
    ut::require(loaded.step_h == "0.02" && loaded.projections.size() == 1, "old phase values changed while loading");
    ut::require(!loaded.adaptive.enabled, "a session without adaptive settings must stay on the fixed step");

    loaded.projections[0].custom_line_style = true;
    loaded.projections[0].line_width = 3.25f;
    loaded.projections[0].alpha = 0.4f;
    loaded.adaptive.enabled = true;
    loaded.adaptive.rtol = "2e-9";
    loaded.adaptive.hmax = "0.42";
    PhaseAnalysisSession rt;
    ut::require(session_from_json(session_to_json(loaded), rt), "phase session round trip failed");
    ut::require(rt.projections.size() == 1 && rt.projections[0].custom_line_style &&
                rt.projections[0].line_width == 3.25f && rt.projections[0].alpha == 0.4f,
                "display settings changed during the round trip");
    ut::require(rt.adaptive == loaded.adaptive, "adaptive settings changed during the round trip");

    BifurcationAnalysisSession bif, bif2;
    bif.diagrams.emplace_back();
    bif.diagrams[0].adaptive = loaded.adaptive;
    ut::require(session_from_json_parametric(session_to_json_parametric(bif), bif2), "parametric round trip failed");
    ut::require(bif2.diagrams.size() == 1 && bif2.diagrams[0].adaptive == loaded.adaptive,
                "bifurcation adaptive settings lost");
}

void queue_cleanup() {
    AppModel model([](const std::vector<unsigned char>&) { return std::string{}; });
    model.bifurcation_session.add_diagram();
    model.bifurcation_session.add_diagram();
    model.bifurcation_session.add_diagram();
    const size_t n0 = model.bifurcation_session.diagrams.size();
    model.parametric_queue.push_back({ ParametricQueueItem::Kind::Bifurcation, 0 });
    model.parametric_queue.push_back({ ParametricQueueItem::Kind::Bifurcation, 1 });
    model.parametric_queue.push_back({ ParametricQueueItem::Kind::Bifurcation, 2 });
    model.parametric_queue.push_back({ ParametricQueueItem::Kind::LLE, 4 });
    model.remove_bifurcation_diagram(1);
    ut::require(model.bifurcation_session.diagrams.size() == n0 - 1, "diagram removal did not update the session");
    ut::require(model.parametric_queue.size() == 3, "queue item of the removed diagram was not erased");
    ut::require(model.parametric_queue[0].index == 0 && model.parametric_queue[1].index == 1 &&
                model.parametric_queue[2].kind == ParametricQueueItem::Kind::LLE && model.parametric_queue[2].index == 4,
                "queue indices were not shifted by analysis kind");

    model.dft1d_queue.push_back({ -1 });
    model.dft1d_queue.push_back({ 99 });
    ut::require(!model.start_next_in_dft1d_queue() && model.dft1d_queue.empty(), "invalid DFT queue entries were not drained");
}

void cancellation() {
    BifurcationAnalysisSession s;
    s.in_flight = true;
    s.cancel_token = std::make_shared<std::atomic<bool>>(false);
    s.request_cancel();
    ut::require(s.cancel_token->load(), "analysis cancellation token was not set");
    s.in_flight = false;
}

void records() {
    // старая запись ОДУ: один смешанный alphabet_text, без "kind"
    const SystemRecord old = record_from_json(
        "{\"name\":\"Old\",\"mode\":\"LaTeX\",\"latex_text\":\"\\\\dot{x} = -a x\","
        "\"alphabet_text\":\"x, a\",\"init_conditions\":{\"x\":\"1\"},\"param_values\":{\"a\":\"2\"}}");
    ut::require(old.name == "Old" && !old.is_dae && !old.is_map && old.alphabet_text == "x, a", "old ODE record");

    SystemRecord d;
    d.name = "DAE round trip";
    d.is_dae = true;
    d.plain_text = "x' = -x + z\n0 = z - p*x";
    d.vars_text = "x"; d.alg_text = "z"; d.params_text = "p";
    d.dae_index_text = "z=1";
    d.dae_as_ode = true;
    d.dae_stab_alpha = "25";
    d.dae_schemes_off = { "DAE Trapezoidal", "DAE CD" };
    d.init_conditions = { { "x", "1" }, { "z", "0.5" } };
    d.param_values = { { "p", "0.5" } };
    const SystemRecord r = record_from_json(record_to_json(d));
    ut::require(r.is_dae && r.plain_text == d.plain_text && r.alg_text == "z" && r.dae_index_text == "z=1" &&
                r.dae_as_ode && r.dae_stab_alpha == "25" && r.dae_schemes_off == d.dae_schemes_off &&
                r.init_conditions.at("z") == "0.5", "DAE record lost fields in the round trip");

    // ДАУ-запись до этапов 5-7: без dae_schemes_off / dae_stab_alpha — все методы включены, alpha по умолчанию
    const SystemRecord older = record_from_json(
        "{\"name\":\"Older DAE\",\"kind\":\"dae\",\"alg_text\":\"z\",\"plain_text\":\"x' = z\\n0 = z + x\","
        "\"vars_text\":\"x\",\"params_text\":\"\"}");
    ut::require(older.is_dae && older.dae_schemes_off.empty() && older.dae_stab_alpha == "10" && !older.dae_as_ode,
                "older DAE record defaults");
}

}  // namespace

int main() {
    return ut::run_main("architecture", []() -> int {
        session_compatibility();
        queue_cleanup();
        cancellation();
        records();
        return 0;
    });
}
