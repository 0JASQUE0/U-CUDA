# Application module

## Purpose

Owns `AppModel`, the active system, application-wide state, cross-analysis queues, persistence decisions, and the main non-visual tick. It coordinates services without containing rendering code.

## Public API and internals

The public surface is `app_model.h`. `AppModel` is one object built from layers, so every field keeps its `model.<name>` path:

- `AppSettings` (`app_settings.h`, `app_settings.cpp`): the mirror of `AppConfig` — hidden tabs, schemes and systems, UI scale, fonts, colormaps, theme, peak finder, NVRTC flags.
- `SystemEditor : AppSettings` (`system_editor.h`, `system_editor.cpp`): the edited system — equation text, schemes, Extr/Comp builders, the DAE editor and circuit import, OCR, `generate`, `build_system`, `to_record` / `from_record`.
- `AppModel : SystemEditor` (`app_model.h`): sessions, the engine, queues, plot windows, broadcast and undo, the Library tab and its scratch `library_edit_buffer`.

Sources of `AppModel` by topic: `app_model.cpp` (`start_*_analysis`, `propagate_to_sessions`), `app_queues.cpp` (queues, prewarm, `remove_*` with queue and window cleanup), `app_plot_windows.cpp`, `app_broadcast.cpp` (broadcast and undo), `tick.cpp` (polling, queue advance, persistence). A new field goes into the lowest layer that owns it; the editor and settings layers never see sessions or the engine.

`session_persistence.h` saves and loads whole sessions that own other sessions (Basins phase slots, Custom sub-sessions) on top of storage's state serializers; that is why storage is a public dependency.

`time_estimate.h` owns asynchronous GPU cost/calibration estimation. Its plain
trajectory work descriptions (`TrajWork`) remain in domain `traj_work.h`; domain
never launches calibration.

## Dependencies and consumers

Target `ucuda::application` exposes analysis, platform and storage requirements and privately uses OCR. UI consumes it.

It must not depend on plotting, ImGui, OpenGL, GLFW, or desktop startup code. Public dependencies are limited to types visible from `app_model.h`.

## Runtime data flow and ownership

UI or an application entry point sends commands to `AppModel`. `AppModel::tick` polls all sessions, advances queues, applies cancellation, cleans indices, and requests persistence regardless of window visibility. The model owns active systems and session collections; it refers to plots through stable data identifiers and never owns GL resources.

## Change this when

Use `system_editor.cpp` for the edited system, `app_queues.cpp` for queue changes, `app_model.cpp` for starting analyses and `tick.cpp` for periodic background coordination. UI presentation belongs in UI.

## Build and checks

Build with `U-CUDA.sln` (Windows) or `cmake --build --preset windows-debug --target ucuda_application`. Run `architecture`; test queues, index cleanup, cancellation, save-on-completion, and operation with `U_CUDA_BUILD_GUI=OFF`.

## Compatibility and platforms

Preserve queue ordering, stable session identity, save timing, and old-session behavior. Application must work with GUI disabled on Windows and Linux; optional OCR failure must not prevent startup or computation.

See [the library map](../README.md) and [the architecture document](../../ARCHITECTURE.md).
