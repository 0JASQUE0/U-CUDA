# Legacy batch application

## Purpose

`main_NonLinAnal.cu` preserves the original non-interactive calculation workflow. The optional `U-CUDA-legacy` target links the CUDA and platform libraries and is useful for compatibility with older scripts and host-library entry points.

## Boundaries

Keep orchestration and legacy argument/file behavior here. Shared kernels, request types, algorithms, and CSV formatting belong in libraries. The executable receives the same runtime kernel package as the desktop application.

## Change this when

Edit this entry point only for legacy workflow behavior. Change shared signatures in [libs/cuda](../../libs/cuda/README.md) and formats in [libs/storage](../../libs/storage/README.md).

## Build and checks

Configure with `U_CUDA_BUILD_LEGACY=ON`, build target `U-CUDA-legacy`, and verify its expected input/output contract whenever host CUDA signatures change.
