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

One tab, one set of files: `phase.cpp`; Parametric in `bifurcation.cpp`, `lyapunov.cpp` (LLE, LS), `metrics.cpp` and `parametric_windows.cpp`; `dft.cpp`; `basins.cpp`; `fastsync.cpp`; `order.cpp`; `network.cpp`; `neural.cpp`; Custom in `custom.cpp` and `workspace.cpp`; `system_tab.cpp`; Library in `library.cpp` (editor, system switch) and `library_list.cpp`. Shared widgets: `widgets.cpp` (input fields, auto labels, undo), `integration_widgets.cpp` (scheme table, Integration block, controller library), `field_names.cpp` (field names, parameter-group trees), `plot_widgets.cpp` (export, view ranges, toolbars, colormaps, regime summaries). `gui.cpp` is the top-level dispatcher, `ui_resources.cpp` owns view lifetime.

Private headers, each compiling on its own:

- `gui_common.h`: includes, constants, mode tabs; every UI source includes it.
- Topic headers for the shared widgets: `gui_inputs.h`, `gui_integration.h`, `gui_fields.h`, `gui_plot.h`.
- One header per tab, `tab_<tab>.h` (`tab_phase.h`, `tab_parametric.h`, `tab_dft.h`, `tab_basins.h`, `tab_fastsync.h`, `tab_order.h`, `tab_network.h`, `tab_neural.h`, `tab_custom.h`, `tab_system.h`, `tab_library.h`), with the functions and window state of that tab.

A source includes `gui_common.h` and only the topic and tab headers whose names it uses; only `gui.cpp` sees every tab. Put a new declaration into the header of the file that defines it, and a function into the file of its tab.

## Build and checks

Build with `cmake --build --preset windows-debug --target ucuda_ui U-CUDA`. Required checks are listed in [AGENTS.md](AGENTS.md).

## Compatibility and platforms

UI is excluded when `U_CUDA_BUILD_GUI=OFF`. Preserve saved display settings and stable runtime associations without serializing pointers or graphics handles; verify behavior on the available Windows and Linux GUI environments.

See [the library map](../README.md) and [the architecture document](../../ARCHITECTURE.md).
