# Application module

## Purpose

Owns `AppModel`, the active system, application-wide state, cross-analysis queues, persistence decisions, and the main non-visual tick. It coordinates services without containing rendering code.

## Public API and internals

The public surface is `app_model.h`. `app_model.cpp` handles model actions and queue-safe edits; `tick.cpp` polls work, advances queues, and persists completed state.

## Dependencies and consumers

Target `ucuda::application` exposes analysis and platform requirements and privately uses storage and OCR. UI consumes it.

It must not depend on plotting, ImGui, OpenGL, GLFW, or desktop startup code. Public dependencies are limited to types visible from `app_model.h`.

## Runtime data flow and ownership

UI or an application entry point sends commands to `AppModel`. `AppModel::tick` polls all sessions, advances queues, applies cancellation, cleans indices, and requests persistence regardless of window visibility. The model owns active systems and session collections; it refers to plots through stable data identifiers and never owns GL resources.

## Change this when

Use `app_model.cpp` for commands/state transitions and `tick.cpp` for periodic background coordination. UI presentation belongs in UI.

## Build and checks

Build with `cmake --build --preset windows-debug --target ucuda_application`. Run `architecture`; test queues, index cleanup, cancellation, save-on-completion, and operation with `U_CUDA_BUILD_GUI=OFF`.

## Compatibility and platforms

Preserve queue ordering, stable session identity, save timing, and old-session behavior. Application must work with GUI disabled on Windows and Linux; optional OCR failure must not prevent startup or computation.

See [the library map](../README.md) and [the architecture document](../../ARCHITECTURE.md).
