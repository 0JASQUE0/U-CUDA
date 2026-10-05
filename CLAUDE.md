# U-CUDA agent entry point

Canonical project documentation:

- [README.md](README.md): setup, build matrix, and “where to edit” navigation.
- [ARCHITECTURE.md](ARCHITECTURE.md): dependency graph, runtime flows, and ownership.
- [AGENTS.md](AGENTS.md): repository-wide implementation rules and required checks.
- [INTEGRATION.md](INTEGRATION.md): fixed-step integration and sampling.

Use the nearest `README.md` when locating module ownership or changing its contract; scoped `AGENTS.md` files supply operational rules. CMake is the only build system. Preserve numerical and serialization compatibility, keep GUI resources outside sessions, and never modify `U-CUDA/library/`.
