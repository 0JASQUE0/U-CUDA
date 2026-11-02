# cuda agent rules

Use [README.md](README.md) for module entry points; consult [ARCHITECTURE.md](../../ARCHITECTURE.md) only for boundary or ownership changes.

- Keep NVCC and NVRTC algorithm sources synchronized through `PackRuntime.cmake`.
- Preserve kernel argument order, precision, operation order, and FMA settings.
- Include every code-affecting option in module cache keys.
- Check every CUDA/driver/NVRTC result and keep cancellation/progress cooperative.
- Do not include UI, plotting, or application headers.

Validation: Build Debug/Release and legacy; run `adaptive_gpu` and `gpu_smoke` on a real GPU with a clean runtime kernel package.
