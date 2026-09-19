# Domain module

## Purpose

Defines shared plain C++ types: systems and records, analysis requests/results, session data, export snapshots, plot-window descriptions, numerical configuration, RQA types, parsing helpers, and the adaptive solver. It exists so storage and computation exchange data without depending on GUI or application state.

## Public API and internals

Key headers include `system.h`, `system_record.h`, `analysis_types.h`, `analysis_session.h`, `export_snapshot.h`, `plot_window.h`, `configCUDA.h`, and `adaptive_solver.h`. `session_defaults.cpp` provides non-inline defaults.

## Dependencies and consumers

Target `ucuda::domain` has no higher-layer dependency. It is consumed by equations, CUDA, storage, analysis, plotting, and application-facing APIs.

It may use the C++ standard library, but it must not include CUDA runtime, JSON, ImGui, OpenGL, application, or storage headers. Consumers are responsible for translating domain values into persistence, execution, or presentation details.

## Runtime data flow and ownership

Parsers and UI create domain requests; analysis and compute libraries fill domain results; storage serializes the persistent subset. Session structures also carry runtime futures and cancellation/progress state, but analysis owns their lifecycle. Plot descriptions and runtime IDs are values only: UI owns views and plotting owns GL handles.

## Change this when

Change this module when a value must cross module boundaries or be serialized. Put runtime handles, futures, engines, widgets, and GL objects elsewhere.

## Build and checks

Build with `cmake --build --preset windows-debug --target ucuda_domain` or the corresponding Linux preset. Run `architecture`, serialization round trips, CPU tests, and GPU tests when request/result ABI or numerical types change.

## Compatibility and platforms

Keep defaults, field meaning, numerical precision, operation order, and serialized names compatible. Domain code must compile with both MSVC on Windows and GCC on Linux and must remain independent of GUI availability.

See [the library map](../README.md), [the architecture document](../../ARCHITECTURE.md), and [adaptive integration notes](../../ADAPTIVE_INTEGRATION.md).
