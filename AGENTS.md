# Repository Guidelines

## Project Structure & Module Organization

`U-CUDA/` contains the C++17/CUDA desktop application. `app_main.cpp` starts the GUI; `app_model.*` owns application state. Keep interface code (`gui.*`, `plot_*`) separate from computation (`nvrtc_engine.*`, `parametric_engine.*`, `hostLibrary.*`). Runtime-generated kernels use `U-CUDA/kernels/*.template.cu`. Shared numerical types live in `configCUDA.h`.

`tests/` holds standalone test executables. `U-CUDA/library_template/` contains distributable system presets; `U-CUDA/library/` is ignored user data. CMake outputs go under `build/`. `.vscode/` provides editor tasks and launch configurations.

## Build, Test, and Development Commands

Use CMake 3.25+, Git, CUDA Toolkit, and a supported host compiler. Windows requires Visual Studio 2022 C++ tools and CUDA integration.

```powershell
cmake --preset windows                      # Configure and fetch dependencies
cmake --build --preset windows-debug        # Build GUI and tests
ctest --preset windows-debug                # Run tests
.\build\windows\bin\Debug\U-CUDA.exe         # Launch application
```

Use `windows-release` for optimized builds/tests. On Arch Linux, configure, build, and test with `linux-debug` or `linux-release`; presets assume `/opt/cuda` and `/usr/bin/g++-15`.

In VS Code, open the repository root, select configure/build presets and the `U-CUDA` target; use Ctrl+Shift+B and the platform-specific F5 configuration.

CMake builds the GUI in both configurations. Enable `-DU_CUDA_BUILD_LEGACY=ON` to build `U-CUDA-legacy`. The original `.sln` retains different Debug/Release entry points. Older build statements in `CLAUDE.md` are outdated; consult current CMake files and README.

## Coding Style & Naming Conventions

Follow surrounding code: four-space indentation, snake_case filenames/functions, and PascalCase types. Preserve `.editorconfig` encoding: C++/CUDA sources use UTF-8 BOM; Markdown/JSON use plain UTF-8. Retain MSVC `/utf-8` for Russian text. No formatter or linter configuration is supplied.

Use `numb` in existing numerical kernels and check CUDA errors. Store persistent UI state in model/session structures.

## Testing Guidelines

CTest runs standalone C++ tests: `gpu_smoke` validates NVRTC and GPU calculations; `platform` tests Linux IPC and CPU integration. GPU unavailability produces a skipped test, not validation. No coverage threshold is configured. Add focused tests under `tests/`, register them with CTest, and build the legacy target when changing shared calculation signatures.

## Commit & Pull Request Guidelines

Prefer the history's descriptive prefixes, such as `order: fix session switching` or `docs: clarify build steps`. Keep commits focused. PRs should explain behavior changes, link relevant issues, report build/test configurations and skipped checks, and include screenshots for UI changes. Exclude build artifacts, local presets, and user libraries.
