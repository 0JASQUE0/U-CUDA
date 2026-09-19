# VS Code agent rules

- Keep launch paths based on CMake commands or workspace-relative paths.
- Support Windows MSVC and Linux GCC workflows.
- Do not commit machine-specific compiler, CUDA, home, or extension paths.
- Keep default build tasks aligned with target U-CUDA.
- Validate JSON syntax and document any new task in README.md.

See [the local workflow](README.md) and [the root architecture](../ARCHITECTURE.md).
