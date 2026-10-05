# Analysis module

## Purpose

Implements analysis sessions for phase portraits, bifurcation, LLE, Lyapunov spectrum, DFT, basins, FastSync, Order, and Custom workflows. Sessions validate snapshots, launch background work, poll futures, and apply results.

## Public API and internals

Public headers re-export session interfaces from domain: `analysis_session.h`, `custom_session.h`, and `order_session.h`. Implementations are split by analysis family.

## Dependencies and consumers

Target `ucuda::analysis` depends on CUDA and equations, with private numerics/thread support. Application consumes it.

It must not depend on storage, application, plotting, ImGui, or OpenGL. It may update domain session state but may not own views or perform persistence itself.

## Runtime data flow and ownership

Application creates a session and submits a snapshot. Analysis validates it, launches CPU/GPU work, owns the future and cancellation/progress transitions, and applies a completed result during polling. Polling is driven by `AppModel::tick`, never by whether an analysis window is visible.

## Change this when

Edit the matching `<analysis>_session.cpp` for session lifecycle. Put shared private helpers in `session_helpers.cpp`/`session_internal.h`; put algorithms in CUDA or numerics.

## Build and checks

Build with `cmake --build --preset windows-debug --target ucuda_analysis`. Run `architecture`, relevant CPU/GPU tests, queue/cancellation tests, and session serialization checks.

## Compatibility and platforms

Preserve session state transitions, cancellation semantics, request snapshots, and ordering of queued work. The module must run in headless builds on Windows and Linux; GPU-required paths still require a real CUDA device for validation.

See [the library map](../README.md) and [the architecture document](../../ARCHITECTURE.md).
