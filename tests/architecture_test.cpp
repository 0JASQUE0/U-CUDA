#include "ucuda/application/app_model.h"
#include "ucuda/platform/platform_paths.h"
#include "ucuda/storage/session_io.h"

#include <atomic>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

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
    require(loaded.adaptive.method == 0, "missing adaptive settings did not use legacy defaults");

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

} // namespace

int main() {
    try {
        session_compatibility();
        queue_cleanup();
        cancellation();
        std::cout << "Architecture, session and queue checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
