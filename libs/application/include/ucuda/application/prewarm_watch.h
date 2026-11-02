#pragma once
#include <chrono>
#include <string>
#include <vector>

// Watches the settings that decide WHICH module a diagram needs, so the prewarm fires once the
// user stops fiddling rather than on every frame. One instance per session; `sigs` holds one
// signature per diagram/curve, and the caller decides what goes into a signature.
struct PrewarmWatch {
    std::vector<std::string>              sigs;        // as of the last pick()
    std::vector<std::string>              fired;       // recently prewarmed, oldest first
    std::chrono::steady_clock::time_point changed_at{};
    int                                   pending = -1;

    // Mirrors the module pool in parametric_engine.cpp: remembering as many keys as the pool
    // keeps means switching back to a recent scheme asks for nothing that is not already there.
    static constexpr size_t kRemembered = 4;

    // Returns the index worth prewarming, or -1. A signature that changed restarts the settle
    // timer; the same signature twice fires only once, so holding still costs nothing.
    int pick(const std::vector<std::string>& current,
             std::chrono::steady_clock::time_point now,
             std::chrono::milliseconds settle) {
        if (sigs.empty() && !current.empty()) {
            // First sighting: nothing is compiled yet, so the freshly opened diagram is exactly
            // what the user is about to Run -- that first Run is the one worth saving.
            sigs = current;
            pending = 0;
            changed_at = now;
            return -1;
        }
        if (sigs.size() != current.size()) {
            // Diagram added or removed: re-baseline instead of prewarming whatever shifted index.
            sigs = current;
            pending = -1;
            changed_at = now;
            return -1;
        }
        for (size_t i = 0; i < current.size(); ++i) {
            if (current[i] == sigs[i]) continue;
            sigs[i]    = current[i];
            pending    = (int)i;
            changed_at = now;
        }
        if (pending < 0 || pending >= (int)sigs.size()) return -1;
        if (now - changed_at < settle) return -1;
        const int idx = pending;
        pending = -1;
        for (const std::string& f : fired)
            if (f == sigs[idx]) return -1;   // already compiled for exactly this
        fired.push_back(sigs[idx]);
        if (fired.size() > kRemembered) fired.erase(fired.begin());
        return idx;
    }
};
