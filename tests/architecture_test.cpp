// Архитектура без GUI:
//  - движок и очереди AppModel закрыты (static_assert);
//  - старый JSON сессии грузится, недостающие поля — по умолчанию; круг JSON -> сессия -> JSON сохраняет
//    стиль проекций и настройки адаптивного шага (Phase и Parametric), а runtime-id в JSON не попадает;
//  - удаление диаграммы чистит и сдвигает очередь Parametric; очереди вкладок снимают неверные элементы,
//    Stop осушает все очереди; отмена ставит токен;
//  - tick без GUI раздаёт сессиям выбранные методы (ОДУ и ДАУ);
//  - скрытые и невыбранные методы не попадают в КРС;
//  - запись системы: старый формат (alphabet_text, без kind) и ДАУ (kind, alg_text, индексы, выключенные
//    методы, alpha уровня C) переживают круг через JSON; импорт схемы заполняет поля ДАУ.
#include "ucuda/application/app_model.h"
#include "ucuda/domain/num_parse.h"
#include "ucuda/equations/dae_ir.h"
#include "ucuda/platform/platform_paths.h"
#include "ucuda/storage/session_io.h"
#include "ucuda/storage/system_library.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

std::string exe_dir() { return platform::executable_dir(); }

namespace {

// Движок и очереди AppModel закрыты: UI работает с ними только командами (run_*, enqueue_*,
// clear_*). Проверка доступа здесь — часть подстановки, поэтому закрытый член даёт false.
template <class T, class = void> struct engine_is_public : std::false_type {};
template <class T>
struct engine_is_public<T, std::void_t<decltype(std::declval<T&>().parametric_engine)>> : std::true_type {};
template <class T, class = void> struct queue_is_public : std::false_type {};
template <class T>
struct queue_is_public<T, std::void_t<decltype(std::declval<T&>().parametric_queue_)>> : std::true_type {};
static_assert(!engine_is_public<AppModel>::value, "AppModel::parametric_engine must stay private");
static_assert(!queue_is_public<AppModel>::value, "AppModel queues must stay private");

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// Личный временный каталог; удаляется, только если имя то самое.
struct TempDir {
    std::filesystem::path root, parent;
    explicit TempDir(const char* prefix) : parent(std::filesystem::temp_directory_path()) {
        root = parent / (prefix + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        require(std::filesystem::create_directory(root), "cannot create a private test directory");
    }
    ~TempDir() {
        std::error_code ec;
        if (root.parent_path() == parent && root.filename().string().rfind("ucuda-", 0) == 0)
            std::filesystem::remove_all(root, ec);
    }
};

AppModel make_model() {
    return AppModel([](const std::vector<unsigned char>&) { return std::string{}; });
}

void session_compatibility() {
    PhaseAnalysisSession loaded;
    const std::string old_json =
        "{\"step_h\":\"0.02\",\"sim_time\":\"12\","
        "\"projections\":[{\"label\":\"legacy\",\"type\":0,"
        "\"ax\":1,\"ay\":2,\"az\":0,\"show_var\":[true,false]}]}";
    require(session_from_json(old_json, loaded), "old phase session did not load");
    require(loaded.step_h == "0.02" && loaded.projections.size() == 1,
            "old phase values changed while loading");
    require(!loaded.adaptive.enabled, "a session without adaptive settings must stay on the fixed step");

    loaded.projections[0].custom_line_style = true;
    loaded.projections[0].line_width = 3.25f;
    loaded.projections[0].alpha = 0.4f;
    loaded.adaptive.enabled = true;
    loaded.adaptive.rtol = "2e-9";
    loaded.adaptive.hmax = "0.42";
    const auto original_runtime_id = loaded.projections[0].runtime_id;

    PhaseAnalysisSession round_trip;
    require(session_from_json(session_to_json(loaded), round_trip),
            "phase session round-trip failed");
    require(round_trip.projections.size() == 1 &&
            round_trip.projections[0].custom_line_style &&
            round_trip.projections[0].line_width == 3.25f &&
            round_trip.projections[0].alpha == 0.4f,
            "persisted display settings changed during round-trip");
    require(round_trip.adaptive == loaded.adaptive, "adaptive settings changed during round-trip");
    require(round_trip.projections[0].runtime_id != original_runtime_id,
            "runtime-only projection id leaked into persisted JSON");

    BifurcationAnalysisSession bif, bif2;
    bif.diagrams.emplace_back();
    bif.diagrams[0].adaptive = loaded.adaptive;
    require(session_from_json_parametric(session_to_json_parametric(bif), bif2), "parametric round-trip failed");
    require(bif2.diagrams.size() == 1 && bif2.diagrams[0].adaptive == loaded.adaptive,
            "bifurcation adaptive settings lost");
}

void queue_cleanup() {
    AppModel model = make_model();
    model.bifurcation_session.add_diagram();
    model.bifurcation_session.add_diagram();
    model.bifurcation_session.add_diagram();
    const size_t n0 = model.bifurcation_session.diagrams.size();
    model.enqueue_parametric({ParametricQueueItem::Kind::Bifurcation, 0});
    model.enqueue_parametric({ParametricQueueItem::Kind::Bifurcation, 1});
    model.enqueue_parametric({ParametricQueueItem::Kind::Bifurcation, 2});
    model.enqueue_parametric({ParametricQueueItem::Kind::LLE, 4});
    model.remove_bifurcation_diagram(1);

    require(model.bifurcation_session.diagrams.size() == n0 - 1,
            "diagram removal did not update the session");
    require(model.parametric_queue().size() == 3,
            "queue item for removed diagram was not erased");
    require(model.parametric_queue()[0].index == 0 &&
            model.parametric_queue()[1].index == 1 &&
            model.parametric_queue()[2].kind == ParametricQueueItem::Kind::LLE &&
            model.parametric_queue()[2].index == 4,
            "queue indices were not shifted by analysis kind");

    model.enqueue_dft1d(-1);
    model.enqueue_dft1d(99);
    require(!model.start_next_in_dft1d_queue() && model.dft1d_queue().empty(),
            "invalid headless queue entries were not drained");

    // Все очереди вкладок с одной сессией идут одним ходом (run_next_queued): устаревшие индексы
    // снимаются без запуска, а пока сессия считает, очередь не трогается.
    auto drains = [&](auto& session, const auto& queue, auto enqueue, auto start) {
        model.clear_all_queues();
        enqueue(-1);
        enqueue(99);
        session.in_flight = true;
        const bool busy_ok = !start() && queue.size() == 2;
        session.in_flight = false;
        return busy_ok && !start() && queue.empty();
    };
    require(drains(model.dft1d_session, model.dft1d_queue(), [&](int i) { model.enqueue_dft1d(i); }, [&] { return model.start_next_in_dft1d_queue(); }),
            "DFT queue");
    require(drains(model.basins_session, model.basins_queue(), [&](int i) { model.enqueue_basins(i); }, [&] { return model.start_next_in_basins_queue(); }),
            "Basins queue");
    require(drains(model.order_session, model.order_queue(), [&](int i) { model.enqueue_order(i); }, [&] { return model.start_next_in_order_queue(); }),
            "Order queue");
    require(drains(model.network_session, model.network_queue(), [&](int i) { model.enqueue_network(i); }, [&] { return model.start_next_in_network_queue(); }),
            "Network queue");
    require(drains(model.neural_session, model.neural_queue(), [&](int i) { model.enqueue_neural(i); }, [&] { return model.start_next_in_neural_queue(); }),
            "Neural queue");
    require(drains(model.fastsync_session, model.fastsync_queue(), [&](int i) { model.enqueue_fastsync(i); }, [&] { return model.start_next_in_fastsync_queue(); }),
            "FastSync queue");

    // Stop в шапке осушает все очереди, в том числе Network и Custom.
    model.enqueue_parametric({ParametricQueueItem::Kind::LS, 0});
    model.enqueue_dft1d(0);
    model.enqueue_basins(0);
    model.enqueue_fastsync(0);
    model.enqueue_order(0);
    model.enqueue_network(0);
    model.enqueue_neural(0);
    model.enqueue_custom({CustomQueueItem::Kind::Phase});
    model.clear_all_queues();
    require(model.parametric_queue().empty() && model.dft1d_queue().empty() &&
            model.basins_queue().empty() && model.fastsync_queue().empty() &&
            model.order_queue().empty() && model.network_queue().empty() &&
            model.neural_queue().empty() && model.custom_queue().empty(),
            "clear_all_queues left items behind");
    model.reset_engine();   // без движка и без прогрева — ничего не делает
}

void cancellation() {
    BifurcationAnalysisSession session;
    session.in_flight = true;
    session.cancel_token = std::make_shared<std::atomic<bool>>(false);
    session.request_cancel();
    require(session.cancel_token->load(), "analysis cancellation token was not set");
    session.in_flight = false;
}

void headless_method_availability() {
    TempDir dir("ucuda-tick-test-");
    SystemLibrary library(dir.root.string());
    AppModel model = make_model();
    model.scheme_rk4 = true;
    model.tick(library);
    const auto has = [](const auto& names, const std::string& name) { return std::find(names.begin(), names.end(), name) != names.end(); };
    require(has(model.phase_session.enabled_builtin_schemes, "RK4"), "headless tick lost ODE method selection");
    model.is_dae = true; model.dae_as_ode = false;
    model.dae_schemes_off = {"DAE RK4"};
    model.tick(library);
    require(!has(model.phase_session.enabled_builtin_schemes, "DAE RK4"), "disabled DAE method escaped to analysis");
    require(has(model.phase_session.enabled_builtin_schemes, kBdfSchemeDae), "headless DAE analysis lost BDF");
    for (const auto* names : {&model.dft1d_session.enabled_builtin_schemes, &model.fastsync_session.enabled_builtin_schemes,
        &model.order_session.enabled_builtin_schemes, &model.network_session.enabled_builtin_schemes})
        require(!has(*names, kBdfSchemeDae), "fixed-step-only session received BDF");
}

void generated_method_selection() {
    AppModel m = make_model();
    m.clear();   // все scheme_* = false (таблица kBuiltinSchemes — приватная у UI)
    m.mode = InputMode::Plain;
    m.plain_text = "x' = -x";
    m.vars_text = "x";
    m.params_text.clear();
    m.scheme_euler = m.scheme_rk4 = true;
    m.hidden_schemes = { "Euler" };
    require(m.generate(), "selected visible method must generate: " + m.error_message);
    require(m.generated_code.find("// ===== RK4 =====") != std::string::npos &&
            m.generated_code.find("// ===== Euler =====") == std::string::npos,
            "hidden method leaked into generated KRS");
    m.hidden_schemes.clear();
    require(m.generate() && m.generated_code.find("// ===== Euler =====") != std::string::npos,
            "showing a method must restore its selected KRS");
    m.scheme_euler = false;
    require(m.generate() && m.generated_code.find("// ===== Euler =====") == std::string::npos,
            "unselected visible method leaked into generated KRS");
    m.hidden_schemes = { "RK4" };
    require(!m.generate() && m.generated_code.empty(), "all selected methods hidden must leave no stale KRS");
    require(m.scheme_rk4, "visibility must preserve the system's method selection");
}

void records() {
    // старая запись ОДУ: один смешанный alphabet_text, без "kind"
    const SystemRecord old = record_from_json(
        "{\"name\":\"Old\",\"mode\":\"LaTeX\",\"latex_text\":\"\\\\dot{x} = -a x\","
        "\"alphabet_text\":\"x, a\",\"init_conditions\":{\"x\":\"1\"},\"param_values\":{\"a\":\"2\"}}");
    require(old.name == "Old" && !old.is_dae && !old.is_map && old.alphabet_text == "x, a", "old ODE record");

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
    d.circuit_source = "D:\\схемы\\rc \"1\".cir";
    d.circuit_netlist = "* rc\nV1 in 0 DC 5\nR1 in out 1k\nC1 out 0 1m\n";
    const SystemRecord r = record_from_json(record_to_json(d));
    require(r.is_dae && r.plain_text == d.plain_text && r.alg_text == "z" && r.dae_index_text == "z=1" &&
            r.dae_as_ode && r.dae_stab_alpha == "25" && r.dae_schemes_off == d.dae_schemes_off &&
            r.init_conditions.at("z") == "0.5", "DAE record lost fields in the round trip");
    require(r.circuit_source == d.circuit_source && r.circuit_netlist == d.circuit_netlist,
            "circuit source / netlist copy lost in the round trip");

    // ДАУ-запись до этапов 5-7: без dae_schemes_off / dae_stab_alpha — все методы включены, alpha по умолчанию
    const SystemRecord older = record_from_json(
        "{\"name\":\"Older DAE\",\"kind\":\"dae\",\"alg_text\":\"z\",\"plain_text\":\"x' = z\\n0 = z + x\","
        "\"vars_text\":\"x\",\"params_text\":\"\"}");
    require(older.is_dae && older.dae_schemes_off.empty() && older.dae_stab_alpha == "10" && !older.dae_as_ode,
            "older DAE record defaults");

    // Импорт схемы в модель (вкладка DAE): поля ДАУ, номиналы, НУ, шаг и копия netlist'а.
    TempDir dir("ucuda-arch-test-");
    const std::filesystem::path cir = dir.root / "ucuda_arch_rc.cir";
    {
        std::ofstream f(cir, std::ios::binary);
        f << "* rc from a file\nV1 in 0 DC 5\nR1 in out 1k\nC1 out 0 1m IC=1\n.tran 1m 5\n";
    }
    AppModel m = make_model();
    m.circuit_source = cir.u8string();
    std::string rep, err;
    require(m.import_circuit(&rep, &err), "import_circuit: " + err);
    // значения R / C / L импорт пишет с приставками СИ (fmt_num_si): 1k
    require(m.is_dae && m.vars_text == "vc_c1" && parse_num(m.param_values.at("r1"), 0) == 1000.0 &&
            m.init_conditions.at("vc_c1") == "1" && m.step_h == "0.001" &&
            m.circuit_netlist.find("R1 in out 1k") != std::string::npos && m.note == "rc from a file",
            "import_circuit fills the DAE fields: " + rep);
    const DaeSystem s = parse_dae(m.plain_text, m.vars_text, m.alg_text, m.params_text);
    require(s.diff.size() == 1 && s.alg.size() == 4 && s.structural_index_hint == 1,
            "imported circuit parses as an index-1 DAE");
}

} // namespace

int main() {
    try {
        session_compatibility();
        queue_cleanup();
        cancellation();
        headless_method_availability();
        generated_method_selection();
        records();
        std::cout << "Architecture, session and queue checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
