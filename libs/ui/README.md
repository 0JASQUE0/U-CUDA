# Ui module

## Purpose

Implements ImGui windows, controls, workspaces, and per-analysis presentation. It owns plot-view instances and maps them to data descriptions using runtime IDs.

## Public API and internals

`gui.h` exposes the desktop draw entry points. Implementation is split by analysis family; `ui_resources.*` owns plot views and cleanup.

## Dependencies and consumers

Target `ucuda::ui` depends on application and storage and privately on plotting. Only the desktop application should consume it. It never includes CUDA or numerics headers (`layer_rules` checks it); direct checks such as the CPU compiler or KRS indices go through application `compute_services.h`.

UI may issue model commands and render model state. It must not implement numerical algorithms, launch independent worker queues, own persistence policy, or place view resources inside domain sessions.

## Runtime data flow and ownership

The desktop loop calls application ticking and then UI drawing. UI maps stable session/projection IDs to UI-owned 2D, 3D, and heatmap views, forwards commands to `AppModel`, and releases views before graphics shutdown. Visibility affects drawing only, never background task progress.

## Change this when

Edit family files for controls/windows, `widgets.cpp` for shared input fields, `integration_widgets.cpp` for the Integration block and controller library, `field_names.cpp` for field names and parameter-group trees, `plot_widgets.cpp` for plot toolbars, view ranges and colormaps, `system_tab.cpp` and `library_list.cpp` for the System tab and the library list, `workspace.cpp` for layout, `library.cpp` for system management, and `ui_resources.cpp` for view lifetime.

Shared declarations live in `src/gui_internal.h`, which every UI source includes. It only pulls in the topic headers, each including the previous one: `gui_common.h` (includes, constants, mode tabs), `gui_inputs.h` (input fields, auto labels, undo), `gui_integration.h` (built-in scheme table, Integration block, controllers), `gui_fields.h` (field names, parameter-group trees), `gui_plot.h` (export, view ranges, toolbars, colormaps) and `gui_tabs.h` (per-tab functions and shared window state). Put a new declaration into the topic header of the file that defines it.

## Build and checks

Build with `cmake --build --preset windows-debug --target ucuda_ui U-CUDA`. Required checks are listed in [AGENTS.md](AGENTS.md).

## Compatibility and platforms

UI is excluded when `U_CUDA_BUILD_GUI=OFF`. Preserve saved display settings and stable runtime associations without serializing pointers or graphics handles; verify behavior on the available Windows and Linux GUI environments.

See [the library map](../README.md) and [the architecture document](../../ARCHITECTURE.md).
