# Ocr module

## Purpose

Provides the C++ client for the optional Python OCR server. It launches and communicates with the process through platform abstractions and returns recognized LaTeX to the application.

## Public API and internals

`ocr_client_win.h` exposes the client and encoding helpers. The server implementation and Python dependencies live in `tools/ocr`.

## Dependencies and consumers

Target `ucuda::ocr` depends privately on platform and threads. Application and desktop consume it.

It must not depend on GUI, plotting, storage, CUDA, or the application model. Python packages and recognition models remain outside the C++ target.

## Runtime data flow and ownership

The client launches the packaged server through platform process APIs, sends framed image data, and returns LaTeX or a structured error. The client owns the child process and communication buffers; application owns when OCR is requested and how the result is applied.

## Change this when

Change process protocol/client behavior here and Python recognition behavior in `tools/ocr`.

## Build and checks

Build with `cmake --build --preset windows-debug --target ucuda_ocr` or the Linux equivalent. Run Linux `platform` protocol tests where available and perform an optional end-to-end OCR smoke test with the configured Python environment.

## Compatibility and platforms

Keep the framing protocol synchronized with `tools/ocr/ocr_server.py`. Missing Python, missing optional packages, or a stopped server must produce a bounded failure rather than block application startup; preserve UTF-8/LaTeX text on Windows and Linux.

See [the library map](../README.md) and [the architecture document](../../ARCHITECTURE.md).
