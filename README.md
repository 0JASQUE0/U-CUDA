# U-CUDA

U-CUDA analyzes nonlinear dynamical systems. It parses ODEs, generates numerical schemes, compiles user-specific code for CPU or CUDA/NVRTC, runs analyses, persists sessions, and renders results through ImGui and OpenGL.

## Start here

Инструкция на русском: [установка проекта, правила кода, MCP и скиллы](SETUP.md).

| Goal | Read or edit |
|---|---|
| Understand module boundaries and data flow | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Change an executable entry point | [apps/README.md](apps/README.md) |
| Find the correct library | [libs/README.md](libs/README.md) |
| Change equations or generated schemes | [libs/equations/README.md](libs/equations/README.md) |
| Change CPU integration or native KRS compilation | [libs/numerics/README.md](libs/numerics/README.md) |
| Change CUDA, NVRTC, kernels, or caches | [libs/cuda/README.md](libs/cuda/README.md) |
| Change sessions, queues, or background work | [libs/analysis/README.md](libs/analysis/README.md) and [libs/application/README.md](libs/application/README.md) |
| Change JSON, system library, or CSV export | [libs/storage/README.md](libs/storage/README.md) |
| Change plots or GUI | [libs/plotting/README.md](libs/plotting/README.md) and [libs/ui/README.md](libs/ui/README.md) |
| Change packaged files | [resources/README.md](resources/README.md) and [cmake/README.md](cmake/README.md) |
| Add or run checks | [tests/README.md](tests/README.md) |
| Configure VS Code | [.vscode/README.md](.vscode/README.md) |

Every documented directory also has an `AGENTS.md` with short, scoped rules for coding agents. The nearest file applies in addition to the root [AGENTS.md](AGENTS.md).

## Build matrix

| Host | C++ compiler | CUDA compiler | Configure | Build |
|---|---|---|---|---|
| Windows | MSVC 2022 | nvcc with MSVC host | `cmake --preset windows` | `cmake --build --preset windows-debug` |
| Windows Release | MSVC 2022 | nvcc with MSVC host | same tree | `cmake --build --preset windows-release` |
| Linux Debug | GCC 15 | nvcc with GCC host | `cmake --preset linux-debug` | `cmake --build --preset linux-debug` |
| Linux Release | GCC 15 | nvcc with GCC host | `cmake --preset linux-release` | `cmake --build --preset linux-release` |

CUDA Toolkit is required in every configuration. CMake is the only source-controlled build system. Visual Studio can open the repository as a CMake project or open the generated `build/windows/U_CUDA.sln`.

The repository does not define a MinGW/GCC-on-Windows preset. The supported Windows CUDA host compiler is MSVC; GCC is used by the Linux presets.

Useful options:

- `-DU_CUDA_BUILD_GUI=OFF`: build computation, storage, the OCR client, and tests without fetching GUI dependencies; the legacy executable is also available when enabled separately.
- `-DU_CUDA_BUILD_LEGACY=ON`: add the `U-CUDA-legacy` batch executable.
- `-DBUILD_TESTING=OFF`: omit CTest executables.

## Applications and runtime files

`U-CUDA` is the desktop application. `U-CUDA-legacy` preserves the old batch workflow. Post-build packaging places `kernels/`, `library_template/`, `ocr_server.py`, and required Windows DLLs beside each executable.

The development user library remains in `U-CUDA/library/`. It is ignored, must not be overwritten, and can be relocated with `U_CUDA_LIBRARY`. OCR is optional; run `setup.ps1` to install [tools/ocr/requirements.txt](tools/ocr/requirements.txt).

## Validation

```powershell
ctest --preset windows-debug
ctest --preset windows-release
```

`adaptive_gpu` and `gpu_smoke` require a working NVIDIA GPU. A skipped GPU test does not validate GPU calculations. Linux additionally builds the `platform` test.

Numerical adaptive integration details are documented in [ADAPTIVE_INTEGRATION.md](ADAPTIVE_INTEGRATION.md).
