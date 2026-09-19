# Platform module

## Purpose

Contains operating-system services: executable/resource paths, process and IPC helpers, file dialogs, clipboard transfer, and image byte handling. It isolates Win32/POSIX details from computation and UI.

## Public API and internals

Public entry points are `platform_paths.h`, `platform_posix.h`, and `image_source.h`. `image_source.cpp` implements clipboard/image operations and uses stb privately.

## Dependencies and consumers

Target `ucuda::platform` is a low-level dependency of numerics, CUDA, storage, OCR, application, and plotting. It must not depend on those consumers.

Platform-specific system libraries and stb are private implementation details. Public headers must not expose GUI, CUDA, storage, or application types.

## Runtime data flow and ownership

Callers provide paths, process requests, or image bytes and receive platform-neutral results. Platform owns OS handles only for the duration defined by its API; plotting owns framebuffer capture, while platform owns transfer of completed pixels to the clipboard or filesystem.

## Change this when

Change this module for OS integration or image transfer. OpenGL framebuffer capture belongs to plotting; only transferring the resulting pixels belongs here.

## Build and checks

Build with `cmake --build --preset windows-debug --target ucuda_platform` or `cmake --build --preset linux-debug --target ucuda_platform`. On Linux run `platform`; on Windows smoke-test paths, dialogs, child processes, and clipboard behavior.

## Compatibility and platforms

Windows implementations use Win32 services; Linux implementations use POSIX paths, processes, and dynamic loading. Keep public behavior consistent and resolve packaged paths relative to the executable rather than the current working directory.

See [the library map](../README.md) and [the architecture document](../../ARCHITECTURE.md).
