# Legacy application rules

- Preserve existing batch inputs, file names, and output formats.
- Do not fork shared numerical implementations into this directory.
- Keep the target buildable with both MSVC+nvcc and GCC+nvcc.
- Build `U-CUDA-legacy` after any host-library, domain type, CUDA, or export-format change.
- Treat existing legacy CSV spelling and spacing as compatibility-sensitive.

See [the legacy README](README.md) and [the root architecture](../../ARCHITECTURE.md).
