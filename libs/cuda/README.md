# Cuda module

## Purpose

Owns CUDA/NVRTC module compilation, kernel launch infrastructure, module caches, GPU algorithms, CPU fallbacks used by parameter sweeps, and legacy host CUDA entry points.

## Public API and internals

Main facades are `NvrtcEngine`, `ParametricEngine`, phase portrait APIs, runtime settings, and legacy declarations in `hostLibrary.cuh`. Requests/results come from domain.

## Dependencies and consumers

Target `ucuda::cuda` depends on domain, numerics, platform, CUDA Toolkit libraries, threads, and the internal export-format target. It is consumed by analysis and legacy.

The NVCC-built legacy host library (`hostLibrary.cuh`: `cudaLibrary.cu` and `host_*.cu` except `host_rqa.cu`) is its own target `ucuda::cuda_legacy`, linked only by `U-CUDA-legacy`. The desktop app and the tests run every analysis through NVRTC modules (`cuLaunchKernel`), so they do not link it; the only NVCC kernels they need are RQA (`host_rqa.cu`, in `ucuda::cuda`). `ucuda::cuda_legacy` is built in every configuration so it keeps compiling.

It must not depend on application, storage orchestration, plotting, ImGui, or OpenGL. The internal export-format dependency contains formatting only and does not pull the storage library into compute code.

## Runtime data flow and ownership

Analysis submits typed requests and generated source. CUDA resolves runtime templates, builds or reuses a cache entry, owns CUDA modules and device allocations, launches kernels, and returns domain results/progress. Cache identity must cover source, compile flags, architecture, precision, FMA, and every algorithm option that changes PTX or behavior.

## Change this when

Use `engine_*` files for modern analysis families (`engine_runtime.cpp` holds shared helpers, source loading and module builds; `engine_metrics.cpp`, `engine_network.cpp`, `engine_order.cpp` with Stability and Performance, and so on per family), with `ParametricEngine::Impl` (`src/engine_impl.h`) layered as `EngineContext` (CUDA context, run signals) <- `EngineSources` (template texts) <- `EngineModuleCache` (module pools, locks, cancel mirrors, `wait_with_signals`) <- `EngineModuleBuilder` (NVRTC, PTX caches, linking) <- `Impl` (`compile_*_if_needed` and `run_*` per family), `host_*` files for legacy entry points, runtime/engine files for compilation and caches, and `device/*.cuh` plus kernel templates for device algorithms.

## Build and checks

Build with `cmake --build --preset windows-debug --target ucuda_cuda` (and `ucuda_cuda_legacy` for `host_*` changes) and repeat for Release and `U-CUDA-legacy`. Required checks are listed in [AGENTS.md](AGENTS.md).

## Compatibility and platforms

NVCC and NVRTC must consume the same device algorithm sources. Preserve kernel names, parameters, launch assumptions, operation order, precision, and FMA behavior. Windows uses NVCC with MSVC; Linux uses NVCC with the configured GCC host compiler.

See [the library map](../README.md), [the architecture document](../../ARCHITECTURE.md), and [integration notes](../../INTEGRATION.md).
