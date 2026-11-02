# Resource rules

- Keep runtime names compatible with code and PackRuntime.cmake.
- Do not place generated build output here.
- Never replace U-CUDA/library with library_template.
- Validate portable layout after adding, removing, or renaming resources.
- Kernel-template changes require real NVRTC/GPU tests.
- Windows resource changes require rebuilding U-CUDA.

See [the resource README](README.md) and [the root architecture](../ARCHITECTURE.md).
