# Numerics module

## Purpose

Provides CPU integration and native compilation of generated KRS code. It is the CPU execution counterpart to the CUDA module and supports CPU fallbacks and comparisons.

## Public API and internals

`integrator.h` exposes CPU integration; `krs_cpu.h` exposes compilation/loading and diagnostics for user-generated schemes.

## Dependencies and consumers

Target `ucuda::numerics` depends on equations and privately on platform, dynamic loading, and threads. CUDA and analysis consume it.

It must not depend on CUDA, storage, application, plotting, or UI. Platform remains private because public numerical APIs do not expose OS handles or paths.

## Runtime data flow and ownership

Numerics receives domain systems or generated host source, performs CPU integration or compiles a temporary native library, and returns ordinary results and diagnostics. Temporary files, dynamic-library handles, and compiler processes are owned and cleaned up inside the module.

## Change this when

Edit `integrator.cpp` for CPU stepping and `krs_cpu.cpp` for compiler invocation, temporary libraries, symbol loading, or diagnostics.

## Build and checks

Build with `cmake --build --preset windows-debug --target ucuda_numerics` or the Linux equivalent. Run `adaptive_cpu`, `architecture`, and Linux `platform`; compare CPU/GPU deterministic calculations after numerical changes.

## Compatibility and platforms

MSVC is the native compiler path on Windows and GCC is used on Linux. Preserve generated ABI, solver defaults, floating-point precision, and step ordering so CPU/GPU comparisons remain meaningful.

See [the library map](../README.md), [the architecture document](../../ARCHITECTURE.md), and [integration notes](../../INTEGRATION.md).
