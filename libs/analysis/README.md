# Analysis module

## Purpose

Implements analysis sessions for phase portraits, bifurcation, LLE, Lyapunov spectrum, DFT, basins, FastSync, Order, and Custom workflows. Sessions validate snapshots, launch background work, poll futures, and apply results.

## Public API and internals

Public headers `analysis_session.h`, `custom_session.h`, `order_session.h` and `network_session.h` define the sessions: each `<Name> : <Name>State` adds to the domain state its futures, cancellation/progress tokens, `in_flight`, and the load/run/poll methods. `analysis_session.h` also declares the scheme-name resolver (`compute_krs_for_scheme`, `adaptive_code_for_scheme`, ...) and `BasinsPhaseSlot` (a phase session per basins config). Implementations are split by analysis family.

## Dependencies and consumers

Target `ucuda::analysis` depends on CUDA and equations, with private numerics/thread support. Application consumes it.

It must not depend on storage, application, plotting, ImGui, or OpenGL. It may update domain session state but may not own views or perform persistence itself.

## Runtime data flow and ownership

Application creates a session and submits a snapshot. Analysis validates it, launches CPU/GPU work, owns the future and cancellation/progress transitions, and applies a completed result during polling. Polling is driven by `AppModel::tick`, never by whether an analysis window is visible.

## Change this when

Edit the matching `<analysis>_session.cpp` for session lifecycle. Put shared private helpers in `session_helpers.cpp`/`session_internal.h`; put algorithms in CUDA or numerics.

## Build and checks

Build with `cmake --build --preset windows-debug --target ucuda_analysis`. Required checks are listed in [AGENTS.md](AGENTS.md).

## Compatibility and platforms

Preserve session state transitions, cancellation semantics, request snapshots, and ordering of queued work. The module must run in headless builds on Windows and Linux; GPU-required paths still require a real CUDA device for validation.

See [the library map](../README.md) and [the architecture document](../../ARCHITECTURE.md).
