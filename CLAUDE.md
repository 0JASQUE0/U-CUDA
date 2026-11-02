# U-CUDA agent entry point

Canonical project documentation:

- [README.md](README.md): setup, build matrix, and “where to edit” navigation.
- [ARCHITECTURE.md](ARCHITECTURE.md): dependency graph, runtime flows, and ownership.
- [AGENTS.md](AGENTS.md): repository-wide implementation rules and required checks.
- [INTEGRATION.md](INTEGRATION.md): fixed-step integration and sampling.
- [NEURAL.md](NEURAL.md): the Neural tab (large networks): kernel schedules, observations, spiking network, STDP, checkpoints, readout.

Use the nearest `README.md` when locating module ownership or changing its contract; scoped `AGENTS.md` files supply operational rules. Windows builds from the classic solution `U-CUDA.sln` (Release = app, Debug = legacy, output in `x64\<Configuration>\`); CMake builds the tests (`ctest`), Linux and the headless configuration. Keep `U-CUDA/U-CUDA.vcxproj` and the library `CMakeLists.txt` on the same sources (`vcxproj_sync`). Preserve numerical and serialization compatibility, keep GUI resources outside sessions, and never modify `U-CUDA/library/`.
