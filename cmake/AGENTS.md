# CMake agent rules

- CMake builds tests, Linux and the headless configuration; the classic Visual Studio solution (`U-CUDA.sln`, `U-CUDA/U-CUDA.vcxproj`) builds the Windows app from the same sources. A source added to a library `CMakeLists.txt` goes into the `.vcxproj` too (`vcxproj_sync` checks it).
- Keep target includes scoped and dependencies explicit.
- Use PUBLIC only for public-header requirements.
- Keep U_CUDA_BUILD_GUI=OFF free of GUI dependency discovery.
- Update PackRuntime.cmake whenever NVRTC-required files change.
- Validate configure/build in a fresh directory and inspect the generated dependency graph after target changes.

See [the CMake README](README.md) and [the root architecture](../ARCHITECTURE.md).
