# Ui module

## Purpose

Implements ImGui windows, controls, workspaces, and per-analysis presentation. It owns plot-view instances and maps them to data descriptions using runtime IDs.

## Public API and internals

`gui.h` exposes the desktop draw entry points. Implementation is split by analysis family; `ui_resources.*` owns plot views and cleanup.

## Dependencies and consumers

Target `ucuda::ui` depends on application and storage and privately on plotting/numerics. Only the desktop application should consume it.

UI may issue model commands and render model state. It must not implement numerical algorithms, launch independent worker queues, own persistence policy, or place view resources inside domain sessions.

## Runtime data flow and ownership

The desktop loop calls application ticking and then UI drawing. UI maps stable session/projection IDs to UI-owned 2D, 3D, and heatmap views, forwards commands to `AppModel`, and releases views before graphics shutdown. Visibility affects drawing only, never background task progress.

## Change this when

Edit family files for controls/windows, `widgets.cpp` for shared widgets, `workspace.cpp` for layout, `library.cpp` for system management, and `ui_resources.cpp` for view lifetime.

## Build and checks

Build with `cmake --build --preset windows-debug --target ucuda_ui U-CUDA`. Smoke-test session switching, window creation/closing, queues, save/load/export, OCR, screenshots, and shutdown.

## Compatibility and platforms

UI is excluded when `U_CUDA_BUILD_GUI=OFF`. Preserve saved display settings and stable runtime associations without serializing pointers or graphics handles; verify behavior on the available Windows and Linux GUI environments.

See [the library map](../README.md) and [the architecture document](../../ARCHITECTURE.md).
