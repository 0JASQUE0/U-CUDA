# Cuda module

## Purpose

Owns CUDA/NVRTC module compilation, kernel launch infrastructure, module caches, GPU algorithms, CPU fallbacks used by parameter sweeps, and the NVCC-built RQA kernels.

## Public API and internals

Main facades are `NvrtcEngine`, `ParametricEngine`, phase portrait APIs, and runtime settings. Requests/results come from domain.

## Dependencies and consumers

Target `ucuda::cuda` depends on domain, numerics, platform, CUDA Toolkit libraries, threads, and the internal export-format target. It is consumed by analysis.

Every analysis runs through NVRTC modules (`cuLaunchKernel`); `src/cudaLibrary.cu` is not compiled by NVCC, it is the shared kernel source that `PackRuntime.cmake` copies into `kernels/` for the templates. The only NVCC kernels are RQA (`host_rqa.cu`).

It must not depend on application, storage orchestration, plotting, ImGui, or OpenGL. The internal export-format dependency contains formatting only and does not pull the storage library into compute code.

## Runtime data flow and ownership

Analysis submits typed requests and generated source. CUDA resolves runtime templates, builds or reuses a cache entry, owns CUDA modules and device allocations, launches kernels, and returns domain results/progress. Cache identity must cover source, compile flags, architecture, precision, FMA, and every algorithm option that changes PTX or behavior.

## Change this when

Use `engine_*` files for modern analysis families (`engine_runtime.cpp` holds shared helpers, source loading and module builds; `engine_metrics.cpp`, `engine_network.cpp`, `engine_order.cpp` with Stability and Performance, and so on per family), with `ParametricEngine::Impl` (`src/engine_impl.h`) layered as `EngineContext` (CUDA context, run signals) <- `EngineSources` (template texts) <- `EngineModuleCache` (module pools, locks, cancel mirrors, `wait_with_signals`) <- `EngineModuleBuilder` (NVRTC, PTX caches, linking) <- `Impl` (`compile_*_if_needed` and `run_*` per family), `host_rqa.cu` for RQA, runtime/engine files for compilation and caches, and `device/*.cuh` plus kernel templates for device algorithms.

## Build and checks

Build with `cmake --build --preset windows-release --target ucuda_cuda` and the Release|x64 solution. Required checks are listed in [AGENTS.md](AGENTS.md).

## Compatibility and platforms

NVCC and NVRTC must consume the same device algorithm sources. Preserve kernel names, parameters, launch assumptions, operation order, precision, and FMA behavior. Windows uses NVCC with MSVC; Linux uses NVCC with the configured GCC host compiler.

See [the library map](../README.md), [the architecture document](../../ARCHITECTURE.md), and [integration notes](../../INTEGRATION.md).
