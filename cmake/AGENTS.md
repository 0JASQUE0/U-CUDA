# CMake agent rules

- CMake is the only build system; do not add handwritten project files.
- Keep target includes scoped and dependencies explicit.
- Use PUBLIC only for public-header requirements.
- Keep U_CUDA_BUILD_GUI=OFF free of GUI dependency discovery.
- Update PackRuntime.cmake whenever NVRTC-required files change.
- Validate configure/build in a fresh directory and inspect the generated dependency graph after target changes.

See [the CMake README](README.md) and [the root architecture](../ARCHITECTURE.md).
