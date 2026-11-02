#include "ucuda/application/app_model.h"
#include "ucuda/platform/platform_paths.h"
#include "ucuda/storage/session_io.h"
#include "ucuda/storage/system_library.h"

#include <atomic>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <filesystem>
#include <chrono>

std::string exe_dir() { return platform::executable_dir(); }

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
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

    loaded.projections[0].custom_line_style = true;
    loaded.projections[0].line_width = 3.25f;
    loaded.projections[0].alpha = 0.4f;
    const auto original_runtime_id = loaded.projections[0].runtime_id;

    PhaseAnalysisSession round_trip;
    require(session_from_json(session_to_json(loaded), round_trip),
            "phase session round-trip failed");
    require(round_trip.projections.size() == 1 &&
            round_trip.projections[0].custom_line_style &&
            round_trip.projections[0].line_width == 3.25f &&
            round_trip.projections[0].alpha == 0.4f,
            "persisted display settings changed during round-trip");
    require(round_trip.projections[0].runtime_id != original_runtime_id,
            "runtime-only projection id leaked into persisted JSON");
}

void queue_cleanup() {
    AppModel model([](const std::vector<unsigned char>&) { return std::string{}; });
    model.bifurcation_session.add_diagram();
    model.bifurcation_session.add_diagram();
    model.bifurcation_session.add_diagram();
    model.parametric_queue.push_back({ParametricQueueItem::Kind::Bifurcation, 0});
    model.parametric_queue.push_back({ParametricQueueItem::Kind::Bifurcation, 1});
    model.parametric_queue.push_back({ParametricQueueItem::Kind::Bifurcation, 2});
    model.parametric_queue.push_back({ParametricQueueItem::Kind::LLE, 4});
    model.remove_bifurcation_diagram(1);

    require(model.bifurcation_session.diagrams.size() == 2,
            "diagram removal did not update the session");
    require(model.parametric_queue.size() == 3,
            "queue item for removed diagram was not erased");
    require(model.parametric_queue[0].index == 0 &&
            model.parametric_queue[1].index == 1 &&
            model.parametric_queue[2].kind == ParametricQueueItem::Kind::LLE &&
            model.parametric_queue[2].index == 4,
            "queue indices were not shifted by analysis kind");

    model.dft1d_queue.push_back({-1});
    model.dft1d_queue.push_back({99});
    require(!model.start_next_in_dft1d_queue() && model.dft1d_queue.empty(),
            "invalid headless queue entries were not drained");
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
    const auto parent = std::filesystem::temp_directory_path();
    const auto root = parent / ("ucuda-tick-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    require(std::filesystem::create_directory(root), "cannot create private tick fixture");
    struct Cleanup {
        std::filesystem::path root, parent;
        ~Cleanup() { std::error_code ec; if (root.parent_path() == parent && root.filename().string().rfind("ucuda-tick-test-", 0) == 0) std::filesystem::remove_all(root, ec); }
    } cleanup{root, parent};
    SystemLibrary library(root.string());
    AppModel model([](const std::vector<unsigned char>&) { return std::string{}; });
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

} // namespace

int main() {
    try {
        session_compatibility();
        queue_cleanup();
        cancellation();
        headless_method_availability();
        std::cout << "Architecture, session and queue checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
