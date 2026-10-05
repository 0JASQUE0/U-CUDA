# Plotting module

## Purpose

Owns reusable OpenGL plotting resources: 2D/3D views, GPU line series, axes, legends, cameras, heatmaps, digit input helpers, and framebuffer capture.

## Public API and internals

Public headers expose renderers and view classes plus `screenshot.h`. Plotting consumes domain descriptions but does not know `AppModel`.

## Dependencies and consumers

Target `ucuda::plotting` depends on domain, platform, GLAD, and GUI graphics dependencies. UI and desktop consume it.

It must not include or retain `AppModel`, storage services, analysis workers, or session ownership. Platform services are private implementation details where possible.

## Runtime data flow and ownership

UI supplies domain plot descriptions and data buffers to views. Plotting owns cameras, textures, buffers, interop registrations, and framebuffer capture until explicit cleanup; it returns captured pixels for platform transfer. Stable session/projection IDs are lookup keys and do not transfer resource ownership.

## Change this when

Edit the matching renderer/view source for visual behavior. Put clipboard transfer in platform and window/session coordination in UI/application.

## Build and checks

Build with `cmake --build --preset windows-debug --target ucuda_plotting U-CUDA`. Exercise 2D, 3D, heatmap, zoom/camera, screenshot, and context shutdown paths.

## Compatibility and platforms

Plotting exists only when `U_CUDA_BUILD_GUI=ON`. Preserve view behavior and release every GL/CUDA interop resource before the OpenGL context is destroyed on both Windows and Linux.

See [the library map](../README.md) and [the architecture document](../../ARCHITECTURE.md).
