# Desktop application

## Purpose

`app_main.cpp` is the composition root for the interactive `U-CUDA` executable. It initializes GLFW, GLAD, ImGui, ImPlot, and ImPlot3D; constructs `AppModel`, `SystemLibrary`, and OCR services; calls the application tick and UI draw functions; handles screenshots; and destroys graphical resources in the correct order.

## Dependencies and flow

The executable links UI, plotting, and OCR. It may know concrete implementations because it is the outermost layer. Source builds use compile-time repository paths; portable builds use resources beside the executable. `U_CUDA_LIBRARY` has highest priority for writable systems.

## Change this when

- Change startup, main-loop, or shutdown order in `app_main.cpp`.
- Change resource discovery here only when runtime layout changes with [cmake/README.md](../../cmake/README.md).
- Change widgets in [libs/ui](../../libs/ui/README.md), rendering in [libs/plotting](../../libs/plotting/README.md), and task processing in [libs/application](../../libs/application/README.md).

## Build and checks

Target: `U-CUDA`. Build Windows Debug/Release, start it from a working directory outside the source tree, and verify GL resources are released before context destruction.
