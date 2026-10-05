# Kernel template rules

- Preserve template markers, kernel names, parameter order, numeric types, and launch assumptions.
- Keep NVRTC code compatible with the configured CUDA standard and packaged flat include names.
- Put shared device algorithms in libs/cuda/src/device, then update runtime packaging if needed.
- Preserve FMA configuration and include it in cache identity.
- Test on a real GPU from a clean runtime package; a skipped test is insufficient.

See [the kernel README](README.md) and [the root architecture](../../ARCHITECTURE.md).
