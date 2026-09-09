# U-CUDA Project Context for AI Agents

## Overview
High-performance desktop GPU application for nonlinear dynamical systems analysis.
Users input ODE systems (LaTeX/plain text/photo via OCR), the app generates CUDA kernels at runtime via NVRTC, integrates on GPU, and renders results (2D/3D phase portraits, bifurcations, LLE, LS, basins of attraction, fast synchronization) in ImGui/ImPlot.

**Key feature:** Data buffers remain GPU-resident; OpenGL uses zero-copy interop—no host↔device copying for rendering.

---

## Build System
- **OS:** Windows 10/11 x64 (only supported platform)
- **Compiler:** Visual Studio 2022, toolset v143, C++17, x64
- **Build:** MSBuild via `.vcxproj` — **NOT CMake**. There is no `CMakeLists.txt`.
- **C++ packages:** vcpkg manifest mode (`vcpkg.json`): glfw3, imgui (docking-experimental + glfw + opengl3), implot, implot3d
- **CUDA:** 12.x or 13.x (project uses "CUDA 13.0" toolset). Code has `#if CUDA_VERSION >= 13000` for `cuCtxCreate` signature change.
- **Links:** cudart_static.lib, nvrtc.lib, cuda.lib, opengl32.lib
- **Solution:** `U-CUDA.sln` (root)
- **Project:** `U-CUDA/U-CUDA.vcxproj`
- **Source encoding:** UTF-8 **with BOM** for all C/C++/CUDA sources (enforced by `.editorconfig`), and `ClCompile` passes `/utf-8` in both configurations. Both halves are required: without the BOM, Visual Studio guesses the system ANSI codepage (CP1251) for a file with Cyrillic comments and bakes in mojibake on save; without `/utf-8`, MSVC re-encodes Russian UI string literals to CP1251 and ImGui — which renders UTF-8 — draws them as boxes.

---

## ⚠️ CRITICAL RULES (Mandatory)
1. **NEVER modify `.vcxproj` files** without explicit user permission.
2. **DO NOT touch `CMakeLists.txt`** (it doesn't exist).
3. **DO NOT delete or rename files** without explicit permission.
4. **DO NOT modify Python files** unless explicitly requested.
5. **After changes:** Build the project (Debug and/or Release as appropriate) and verify compilation.
6. **Two entry points:** Any change to `hostLibrary` signatures must be tested in **BOTH** configurations (Debug legacy scripts must still compile).
7. **CUDA specifics:**
   - Use `__host__ __device__` qualifiers explicitly
   - Prefer SoA (Structure of Arrays) over AoS
   - Wrap all CUDA calls in error-checking macros (`gpuErrchk` or similar)
8. **ImGui:** Immediate mode—do NOT store UI state in local variables between frames; it must live in `AppModel`/state structures.
9. **Separate UI logic from compute logic.**
10. **Encoding:** Do NOT strip the UTF-8 BOM from sources and do NOT remove `/utf-8` from `ClCompile` (see Build System). Losing either one breaks Russian text silently — the build still succeeds. When touching encoding, verify the result in the built `.exe`, not just by compiling: search the binary for a known Russian literal and confirm it is stored as UTF-8.

---

## 🚨 Two Entry Points (Important!)
The project builds **ONE .exe** with different `main()` depending on configuration:

- **Debug|x64:** `main_NonLinAnal.cu` included, `app_main.cpp` excluded
  - Legacy scripting entry point (~4000 lines)
  - Contains commented-out templates for calling calculation functions from `hostLibrary.cu`
  - Used for offline research, batch runs, numerical scheme comparison
  - Usually one call is uncommented: run, save CSV, done

- **Release|x64:** `main_NonLinAnal.cu` excluded, `app_main.cpp` included
  - Modern UI version (ImGui/ImPlot, OCR, ODE editor, NVRTC engines)
  - Main path for end users

**Any signature change in calculation functions must be verified in BOTH configurations.**

---

## CUDA-Specific Guidelines

### Type System
- **Use `numb`** (typedef in `configCUDA.h`, currently `= double`)
- **NEVER hardcode `double` or `float`** in CUDA code—use `numb`
- **`AMOUNTOFX`** = phase space dimensionality (default 3), redefined via `#define` before `#include configCUDA.h`
  - Used in NVRTC pipeline for arbitrary-dimension systems

### Memory & Performance
- Prefer **SoA (Structure of Arrays)** over AoS for GPU data
- Use `__ldg()` for read-only kernel parameters
- Be aware of: warp divergence, shared memory bank conflicts, memory coalescing
- Wrap every CUDA API call with error-checking macro (`gpuErrchk`)

### Debugging
- Suggest `compute-sanitizer` or `nsight-compute` for CUDA issues
- For NVRTC problems: check that headers were copied to `kernels/` (Post-Build Event)

---

## 📁 Directory Structure & Key Files

```
D:\U-CUDA\
├── CLAUDE.md              ← You are reading this (rules for AI)
├── README.md              ← Human installation guide
├── U-CUDA.sln             ← VS solution
├── vcpkg.json             ← Dependencies manifest
├── requirements.txt       ← Python deps for OCR server
├── setup.ps1              ← Helper install script
├── ucuda-dashboard/       ← Separate web panel (NOT part of C++ build)
├── x64/{Debug,Release}/   ← Build output — this is `OutDir` (in .gitignore).
│                            `U-CUDA.exe` and the runtime `kernels/` live HERE, at solution level
└── U-CUDA/                ← C++/CUDA sources
    ├── app_main.cpp       ← Release entry point (modern UI)
    ├── main_NonLinAnal.cu ← Debug entry point (legacy scripts)
    ├── kernels/           ← .template.cu for NVRTC (filled with user ODEs at runtime)
    ├── library/           ← System presets (Lorenz, Rossler, Chen, ...), each = folder with system.json + sessions/*.json
    ├── include/           ← Third-party headers (glad, KHR)
    ├── ocr_server.py      ← Python OCR server (pix2text), IPC with C++
    ├── stb_image_write.h  ← Only third-party header in root
    └── x64/               ← ⚠️ STALE leftovers from an older layout, NOT the build output.
                             Nothing writes here anymore — checking these copies of `kernels/`
                             to see whether a template was deployed will mislead you.
```

### Computational Core (host-side orchestration)
- **hostLibrary.cu / .cuh** — Implementation of all "top-level" functions:
  - `bifurcation1D` / `bifurcation2D`
  - `LLE1D` / `LLE2D`, `LS1D` / `LS2D`
  - `basinsOfAttraction` (+ `_logAxes`)
  - `FastSynchro`, `FastSynchro_2`
  - `TimeDomainCalculation`
  - `distributedSystemSimulation`
  - `bifurcation1DForH`, `bifurcation_DFT_1D`
  - `neuronClasterization2D`

- **cudaLibrary.cu / .cuh** — Low-level CUDA primitives:
  - Integrators (Euler / Cromer / Midpoint / RK4)
  - DBSCAN on GPU
  - Reductions, etc.

- **cudaMacros.cu / .cuh** — Macros: `gpuErrchk` and similar
- **configCUDA.h** — `typedef numb = double`, `AMOUNTOFX`, flags

### Runtime Code Generation
- **codegen.cpp / .hpp** — Converts user system to code, glues with template from `kernels/`
- **nvrtc_engine.cpp / .h** — NVRTC wrapper: compiles resulting `.cu`, returns `CUmodule`/`CUfunction`
- **parametric_engine.cpp / .h** — Bifurcation/LLE/LS sweeps via NVRTC
- **phase_portrait_nvrtc.cpp / .h** — Phase portrait integration via NVRTC
- **integrator.cpp / .h** — CPU-fallback integrators (not GPU)
- **sysparse.cpp / .hpp** — Parsing LaTeX/plain-string ODEs

### Kernel Templates (NVRTC substitutes generated code here)
- `kernels/bifurcation1d.template.cu` and `*_cont`
- `kernels/bifurcation2d.template.cu`
- `kernels/lle1d.template.cu` / `lle2d.template.cu`
- `kernels/ls1d.template.cu` / `ls2d.template.cu`
- `kernels/basins.template.cu`
- `kernels/fastsync_attr.template.cu` / `fastsync_grid.template.cu`

### App Model / State
- **app_model.h / .cpp** — `AppModel`: modes (Library / Analysis / Parametric / Basins / FastSync / Settings), task queues (`ParametricQueueItem`, `BasinsQueueItem`, `FastSyncQueueItem`), OCR state (`OcrState`), selected integration schemes
- **app_config.cpp / .h** — App settings serialization
- **analysis_session.cpp / .h** — Snapshot of parameters for one "analysis session"
- **session_io.cpp / .h** — Save/load sessions to JSON
- **system_library.cpp / .h** — Working with `library/*/system.json`
- **system_record.h** — Struct for one ODE system (name, latex, param_order, initial conditions, values, selected numerical schemes)

### UI (ImGui/ImPlot)
- **gui.cpp / .h** — Top-level UI, ImGui widgets, menus
- **plot_renderer.cpp / .h** — General plot renderer
- **plot_view_2d.cpp / .h** — 2D phase portraits, time series
- **plot_view_3d.cpp / .h** — 3D phase portraits
- **plot_axis.cpp / .h** — Common axes
- **plot_camera_3d.cpp / .h** — Camera for 3D plot
- **plot_legend.cpp / .h** — Legends
- **heatmap_view.cpp / .h** — Heatmap rendering for bifurcations/LLE/basins
- **gpu_line_series.cpp / .h** — GPU-resident 2D line (zero-copy GL)
- **gpu_line_series_3d.cpp / .h** — Same for 3D
- **image_source.cpp / .h** — Image loading/decoding (for OCR input)
- **glad.c** — OpenGL loader (glad)

### Data & Export
- **data_export.cpp / .h** — Export results (CSV, PNG, etc.)
- **ocr_client_win.cpp / .h** — IPC client to `ocr_server.py` (Windows-only)

---

## 🔄 NVRTC ODE Code Generation Pipeline
1. User inputs system (LaTeX/plain/OCR) → `sysparse` builds AST
2. `codegen` emits `.cu` source with device function that computes RHS + required Jacobians (for LLE/LS)
3. This code is substituted into template from `kernels/<task>.template.cu`
4. `nvrtc_engine` compiles, returns `CUfunction`
5. `parametric_engine` / `phase_portrait_nvrtc` launches kernel, retrieves result into GPU buffers, passes to renderer
6. Renderer uses the same memory via OpenGL interop—no copies
7. **Post-Build Event** copies runtime data to `<solution>/x64/{Debug,Release}/` (= `OutDir`): `ocr_server.py`, the whole `kernels/` folder, plus `cudaLibrary.cu`, `cudaLibrary.cuh`, `cudaMacros.cuh` and `configCUDA.h` into `kernels/`—NVRTC picks them up at runtime. Note: `library/` is **not** copied (it is resolved from source at runtime), and neither is `hostLibrary.cuh`.

---

## ImGui/ImPlot Guidelines
- **Immediate mode:** Do NOT store UI state in local variables between frames
- State must live in `AppModel` or state structures
- **Separate UI logic from compute logic**
- UI thread should not block on heavy calculations
- Use high-contrast, non-standard color palettes for basins of attraction and phase portraits
- Ensure readable font sizes
- All phase portrait figures must have captions

---

## Working with System Presets
To add a new system to the library:
- Create folder: `U-CUDA/library/<SystemName>/`
- Add `system.json` with fields:
  - `name`
  - `latex_text` / `plain_text`
  - `alphabet_text`
  - `init_conditions`
  - `param_values`
  - `scheme_*` flags for available numerical schemes
- See example: `library/Lorenz/system.json`

---

## Common Workflows

### Build Commands
Build the **solution**, from the repo root. `OutDir` is derived from `$(SolutionDir)`, so building `U-CUDA/U-CUDA.vcxproj` directly puts the binary and its `kernels/` somewhere else and the post-build copy no longer matches what the .exe loads.

```powershell
# Build Release UI
MSBuild U-CUDA.sln /p:Configuration=Release /p:Platform=x64

# Build Debug legacy
MSBuild U-CUDA.sln /p:Configuration=Debug /p:Platform=x64
```

### Run Application
```powershell
x64/{Debug|Release}/U-CUDA.exe
```

### Debugging
- **CUDA:** `compute-sanitizer` / Nsight Compute
- **NVRTC issues:** Check that headers were copied to `kernels/` (Post-Build Event)

---

## How to Write Prompts for This Project (Hints for AI)
1. Always mention whether you're working with UI version (Release / `app_main`) or legacy scripts (Debug / `main_NonLinAnal.cu`)
2. For changes to calculation functions, mention both `.cuh` and `.cu`—signatures live in both places
3. If something involves NVRTC—check that the header you're referencing is actually copied to `kernels/` post-build (otherwise runtime kernel compilation won't find it)
4. Specify which configuration needs to be rebuilt after changes
5. Don't ask to modify `.vcxproj` without good reason (see rules)
6. For new system presets—see format in `library/Lorenz/system.json`
7. Use `numb` type in CUDA code, not `double`/`float`
8. Remember `AMOUNTOFX` for phase space dimensionality

---

**End of CLAUDE.md. If project structure changes significantly, update this file.**