# Cuda module

## Purpose

Owns CUDA/NVRTC module compilation, kernel launch infrastructure, module caches, GPU algorithms, CPU fallbacks used by parameter sweeps, and legacy host CUDA entry points.

## Public API and internals

Main facades are `NvrtcEngine`, `ParametricEngine`, phase portrait APIs, runtime settings, and legacy declarations in `hostLibrary.cuh`. Requests/results come from domain.

## Dependencies and consumers

Target `ucuda::cuda` depends on domain, numerics, platform, CUDA Toolkit libraries, threads, and the internal export-format target. It is consumed by analysis and legacy.

It must not depend on application, storage orchestration, plotting, ImGui, or OpenGL. The internal export-format dependency contains formatting only and does not pull the storage library into compute code.

## Runtime data flow and ownership

Analysis submits typed requests and generated source. CUDA resolves runtime templates, builds or reuses a cache entry, owns CUDA modules and device allocations, launches kernels, and returns domain results/progress. Cache identity must cover source, compile flags, architecture, precision, FMA, and every algorithm option that changes PTX or behavior.

## Change this when

Use `engine_*` files for modern analysis families, `host_*` files for legacy entry points, runtime/engine files for compilation and caches, and `device/*.cuh` plus kernel templates for device algorithms.

## Build and checks

Build with `cmake --build --preset windows-debug --target ucuda_cuda` and repeat for Release and `U-CUDA-legacy`. Run `adaptive_gpu` and `gpu_smoke` on a real GPU with a clean runtime kernel package.

## Compatibility and platforms

NVCC and NVRTC must consume the same device algorithm sources. Preserve kernel names, parameters, launch assumptions, operation order, precision, and FMA behavior. Windows uses NVCC with MSVC; Linux uses NVCC with the configured GCC host compiler.

See [the library map](../README.md), [the architecture document](../../ARCHITECTURE.md), and [adaptive integration notes](../../ADAPTIVE_INTEGRATION.md).
