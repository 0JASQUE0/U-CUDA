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

**Code map — `ARCHITECTURE.md` (root):** layers, one frame of the UI loop, a table tab → GUI file → session /
request builder → engine entry → template / kernel → CPU twin, how a scheme name becomes a step body,
persistence, a "where to look" index and recipes (new scheme, setting, kernel). Read it first when looking
for code; update it when a tab, engine entry, template or persistence path is added or moved.

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
- **krs_prep.h** — header-only: стадия предрасчёта КРС. Тело шага начинается блоком между
  маркерами `// @ucuda-krs-prep-begin/end` и работает как есть. Две формы блока: строки
  `const numb <имя> = <выражение над a[], h>;` (CD fast) или `numb P[K];` + произвольный C-код,
  заполняющий `P[]` (пользовательские КРС, Matreshka 4.0 fast v3); шаг читает `P[]`.
  `build_module` / Analysis-ядро разбирают блок и дают горячим циклам `ucudaKrsPrep` (раз до
  траектории) + `calculateDiscreteModelP`. У прочих схем это пустая структура и прямой вызов
  шага (умолчания в `cudaLibrary.cu`), код не меняется
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
- `kernels/signal_metrics.template.cu` — Parametric → Metrics: max/min/mean,
  частоты по пикам (тот же `PeakStream`, что у БД) и параметры Хьорта,
  потоково, без хранения траектории. Номера строк выхода (`SIGM_*`) обязаны
  совпадать с `SignalMetric` в `parametric_engine.h`
- `kernels/network.template.cu` — сеть связанных осцилляторов: блок на сеть,
  поток на узел, состояние в shared. Два плейсхолдера вместо одного:
  `{{KRS_BODY}}` (шаг узла) и `{{COUPLING_BODY}}` (case-ветки switch по
  номеру закона связи, их печатает `codegen_coupling`).
- `kernels/order.template.cu` — вкладка Order: оценка порядка, замер
  Performance (`perfIntegrateKernel`), область устойчивости.
- Адаптивный шаг (см. раздел «Adaptive step» ниже): `kernels/ucuda_adaptive.cuh`
  (регуляторы и драйвер), `kernels/adaptive_part.cu` (свипы БД/бассейнов),
  `kernels/metrics_adaptive_part.cu`, `kernels/lyapunov_adaptive_part.cu` —
  не самостоятельные шаблоны, а хвосты, которые движок приклеивает к
  `bifurcation2d.template.cu` / `signal_metrics.template.cu`.

### App Model / State
- **app_model.h / .cpp** — `AppModel`: modes (Library / Analysis / Parametric / Dft1D / Basins / FastSync / Custom / Order / Network / Settings), task queues (`ParametricQueueItem`, `BasinsQueueItem`, `FastSyncQueueItem`), OCR state (`OcrState`), selected integration schemes
- **app_config.cpp / .h** — App settings serialization
- **analysis_session.cpp / .h** — Snapshot of parameters for one "analysis session"
- **network_session.cpp / .h** — вкладка Network: топология (генераторы кольца,
  цепи, звезды, решётки, королевского графа, полного графа и Уоттса-Строгаца),
  узлы с переопределением параметров, рёбра с весом и законом связи. Связь
  считается РАСЩЕПЛЕНИЕМ (шаг узла, затем `X += h*coupling`), как в
  `calculateDiscreteModelforFastSynchro`, — поэтому на вкладке работают все
  схемы, но сама связь интегрируется первым порядком
- **session_io.cpp / .h** — Save/load sessions to JSON; also the step controller
  library file (`load_ctrl_library` / `save_ctrl_library`)
- **order_session.cpp / .h** — вкладка Order: порядок, Performance (в т.ч.
  «точность — затраты» для адаптивного шага), устойчивость; CPU-ветки в
  double / dd / qd через `krs_cpu`
- **adaptive_settings.h** — header-only: `AdaptiveSettings` (блок Integration
  каждой вкладки), встроенные регуляторы, библиотека пользовательских
  регуляторов, `adaptive_build_params` (настройки → `UcudaAdaptParams`)
- **krs_cpu.cpp / .h** — cl.exe-сборка тела КРС в DLL для CPU-путей
  (`KrsCpuStep`, double/dd/qd) и тела пользовательского регулятора (`CtrlCpuFn`). В double/float-теле
  вызовы `fmin(` / `fmax(` (без `::` перед именем) заменяются встроенными `ucuda_fmin_i` / `ucuda_fmax_i`
  (`inline_fmin_fmax`; CRT-вызов в DLL дороже): мультивибратор BDF1 92 -> 88 мкс на шаг, PUT burst neuron
  Radau IIA 914 -> 843 мкс, X побитово тот же
- **system_library.cpp / .h** — Working with `library/*/system.json`
- **system_record.h** — Struct for one ODE system (name, latex, param_order, initial conditions, values, selected numerical schemes)

### UI (ImGui/ImPlot)
- **gui.cpp / .h** — Top-level UI: `draw_gui` (frame, menus, tab dispatch) and `gui_process_deferred`. Split
  2026-10-30 (mechanically, no code changes) into one file per tab: `gui_widgets.cpp` (shared widgets: input
  fields, scheme combo, Integration/Newton blocks, toolbars, export menus), `gui_system.cpp` (System tab,
  Extr/Comp builders, Generate, parameters, library list/editor incl. DAE), `gui_analysis.cpp`,
  `gui_parametric.cpp` (BD / LLE / LS / Metrics + Parametric windows), `gui_dft_basins.cpp`, `gui_fastsync.cpp`,
  `gui_order.cpp`, `gui_network.cpp`, `gui_custom.cpp`. **gui_internal.h** — what several of them share:
  includes, types, constants, templates, prototypes (default arguments live only there). A helper used by one
  tab stays `static` in that tab's file; when another tab needs it, move its prototype into gui_internal.h and
  drop `static` (and the default arguments from the definition).
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
1. User inputs system (LaTeX/plain/OCR) → `sysparse` normalizes the text and splits it into `System{vars, params, rhs}`. This is string-level work (regex/char scanning): `sysparse` builds **no** AST, and `rhs` are plain strings.
2. `codegen` parses those strings into its own AST (`Node`/`Parser`, private to the anonymous namespace in `codegen.cpp`) and emits the **body of the integration step** — that is, the body of `calculateDiscreteModel`, and nothing else.
3. This code is substituted into template from `kernels/<task>.template.cu` at the `{{KRS_BODY}}` placeholder
4. `nvrtc_engine` compiles, returns `CUfunction`
5. `parametric_engine` / `phase_portrait_nvrtc` launches kernel, retrieves result into GPU buffers, passes to renderer
6. Renderer uses the same memory via OpenGL interop—no copies
7. **Post-Build Event** copies runtime data to `<solution>/x64/{Debug,Release}/` (= `OutDir`): `ocr_server.py`, the whole `kernels/` folder, plus `cudaLibrary.cu`, `cudaLibrary.cuh`, `cudaMacros.cuh` and `configCUDA.h` into `kernels/`—NVRTC picks them up at runtime. Note: `library/` is **not** copied (it is resolved from source at runtime), and neither is `hostLibrary.cuh`.

### How a sweep module is compiled (monolith vs rdc)
`build_module` (`parametric_engine.cpp`) has two paths; `set_nvrtc_rdc` picks one. The code default is
**off** (`g_nvrtc_rdc{false}`, `AppConfig::nvrtc_rdc = false`); only a saved `_app_config.json` turns it on.
- **Monolith (rdc off):** the template defines `calculateDiscreteModel` (`__forceinline__`) and then
  `#include "cudaLibrary.cu"` — one translation unit. **Kernel pruning (stage 10):** every `__global__` of
  `cudaLibrary.cu` (33) sits in `#if !defined(UCUDA_PRUNE_KERNELS) || defined(UCUDA_KEEP_<name>)`, and
  `build_module` puts `#define UCUDA_PRUNE_KERNELS` + `#define UCUDA_KEEP_<e>` for each `name_exprs` entry before
  that include (`prune_kernel_defines`; the template's own `extern "C"` kernels are always there). Without the
  macro the text is the old one (rdc library, CPU DLL, Debug). Bit equality with the full build: NVVM tunes a
  non-inlined shared function to ALL its call sites (an argument constant in every call is folded, a pointer's
  address space is inferred), so a pruned module must keep the full build's variety of calls. (1) `pow`: the
  continuation templates call it only as pow(10, x) -> keeper kernel `ucudaPruneKeeper` (end of `cudaLibrary.cu`,
  never launched) calls `pow(a, b)` with runtime arguments. (2) Loop wrappers (`loopCalculateDiscreteModel_int` /
  `...Peaks_int`, not inlined on a big step body): a block before the first kernel keeps `calculateDiscreteModelCUDA`
  + `LLEKernelCUDA` (+ `calculateDiscreteModelPeaksCUDA` for Peaks_int) whenever anything left calls a wrapper (a
  listed library kernel, or the template itself: `UCUDA_KEEP_LOOP` / `UCUDA_KEEP_PEAKS_LOOP`, set by the engine from
  the template text). Together they give every argument kind of the full build (data / flags from parameters or
  nullptr, local `&ticks` or nullptr, preScaller 1 or runtime). A synthetic keeper with such calls failed: NVVM
  inlined them into the keeper itself, and the remaining call was the constant one (Radau LLE: the wrapper lost 8 of
  13 parameters). Checked by PTX per function (labels, `callseq`, `__local_depot` numbering normalised; a full PTX =
  the dumped pruned source minus the prune lines, one NVRTC run): identical on Radau IIA DAE (11 modules: BD 1D/2D,
  continuation, LLE/LS 1D + cont, Metrics, basins, DFT cont / h-sweep), Lorenz RK4, adaptive DOP853 / BDF 1-5 sweeps;
  Rodas4 adaptive: 1 function of 35 differs by one 16-byte local load split in two (same addresses, no arithmetic),
  results bit-identical. Results bit-identical also on Lorenz CD fast (prep block), Duffing with t, FastSync
  attr/grid, Order, Network. Test `tests/u-cuda/prune_test.cpp` (results, pruned vs `UCUDA_NVRTC_PRUNE=0`). Knobs:
  `UCUDA_NVRTC_PRUNE=0` = full build; `UCUDA_NVRTC_PTX_DIR=<dir>` dumps the source (`.cu`) and PTX of every
  monolithic build. Cost (RTX 2060, `CUDA_CACHE_DISABLE=1`, build + run of all modules of the probe): Lorenz RK4
  14-26 s vs 95 s, Duffing 24 s vs 172 s, Rodas4 adaptive sweeps 61 s vs 325 s; Radau IIA PTX per module
  15-17 MB -> 3.3-9.9 MB (NVRTC alone ~210 s -> 42-145 s). Before pruning (Mikhailov DAU_n, DAE CD, 45 KB body): LLE
  2D 266 s (NVRTC 62 s + ptxas 204 s, 30 entries, 15 MB PTX). A new kernel in `cudaLibrary.cu` needs the same
  `#if` wrapper (and a place in the wrapper list if it calls a loop wrapper); a new non-inlined shared function
  called with constants may need a keeper call (check with the PTX dump).
- **rdc:** the library half (template with `extern "C" krs_step(...)` calls instead of the body) is compiled
  once per (template, N, par_or_var, fmad, peak settings, kernels) and cached as PTX (`lib_ptx_for`); the step
  is its own small TU (`krs_ptx_for`); `cuLinkAddData` x2 + `cuLinkComplete` join them (PTX link, no LTO, so
  the step is a real call). The lost FMA contraction across that call changes last bits on some schemes.
- **Why it matters.** An earlier scratch try (no keepers) gave LS 1D on Mikhailov ~1e-12 differences in 8 of 16
  rows: with every remaining caller passing `data = nullptr`, NVVM specialised `loopCalculateDiscreteModel_int`, the
  CFG changed and FMA contraction in the step copies landed differently (with `--fmad=false` both agree exactly).

### Where linearization actually comes from
**LLE/LS do not use a Jacobian.** They linearize with perturbed clone trajectories plus Gram-Schmidt (`LLEKernelCUDA` / `LSKernelCUDA` in `cudaLibrary.cu`) — the Benettin/Wolf method. No variational equations, no monodromy matrix anywhere.

Symbolic differentiation does exist, but it serves **only the implicit schemes** (`Implicit Euler`, `Implicit Midpoint`): `pn_diff` / `jac_over` in `codegen.cpp` emit an N×N Jacobian into the step body for the Newton solve. For anyone extending it:
- build results through the `pn_*` peephole constructors, but do **not** make `pn_add`/`pn_sub` fold — the CD emitter depends on their exact shape; zero-folding for the chain rule lives in the `jd_*` wrappers;
- `d(fabs)` emits `copysign(1,u)`, not the device-side `sign()` — that one is invisible to the phase-portrait NVRTC path, the CPU bytecode and the bare-`cl.exe` KRS path;
- `floor`/`ceil`/`fmod` are rejected at codegen time (`jac_check_differentiable`).

---

## Explicit time t (non-autonomous systems)
A right-hand side may use `t` (not a variable or parameter). Autonomous systems must stay
**byte-identical** in every generated body — check with a dump of `codegen_scheme` /
`codegen_adaptive` / CPU / complex-core output for all `library/` systems before and after.
- `system_uses_time(System)` (AST scan); the body then reads `kStepTimeArg` = `ucuda_t`, and its
  signature becomes `calculateDiscreteModel(X, ucuda_t, a, h)`. Engines detect it from the body
  text (`step_body_uses_time`), like `needs_complex` — the body is already the cache key.
- Stage times are printed by the scheme (`rhs_over(s, st, tname)`, `time_name`): RK4
  `ucuda_ts = ucuda_t + {0, h/2, h/2, h}`, explicit RK tables `ucuda_t + Cs[i]*h` (`Cs` = row sums
  of A, `stage_time_table`). One t for all stages silently drops the order to 1.
- Schemes (`scheme_supports_time`): Euler, Euler-Cromer, Explicit Midpoint, RK4, DOPRI78(+legacy),
  RK45, DOP853, Implicit Euler / Midpoint, SEMP / SIMP / D (stage at t or t + h1, corrector at t + h1;
  D at t + h), CD / CD {1-0} / CD FP (explicit half at the start of its sub-interval, implicit half at
  its end), CD fast / CD fast FP — generated as CD / CD FP when t is used (the prep block would freeze
  t), GBS (leapfrog z_m at t + m*hs), Extr/Comp over any of them (each substep shadows `ucuda_t`;
  lambda form gets it as a second argument), all DAE methods. Adaptive: RK45 / DOP853 / DOPRI78, GBS
  and Extr (same stage times as the fixed ones, F1 = f(t + h, Y)). Refused: complex CD family and
  Complex Implicit Euler (complex stage times), ExtrZ, FastSync, Network.
- GPU fixed-step loops: `cudaLibrary.cuh` macros, all empty without `UCUDA_NONAUTONOMOUS`. A kernel
  keeps the trajectory's step index (`UCUDA_NVAR(ucudaStep, 0)`) and passes it to every loop call
  (`UCUDA_NA(&ucudaStep)`, or `UCUDA_NA_AFTER_WS` for calls cut off at `writeStep`); the loop wrapper
  copies it into a register (`UCUDA_NBEGIN` / `UCUDA_NEND`), the inlined core steps with
  `UCUDA_STEP` = `calculateDiscreteModelP(X, n++ * h, a, h, P)`, so transient and record calls share
  one time axis. LLE/LS clones start each block from the main trajectory's index (`ucudaClone` /
  `ucudaBlock0`); continuation kernels keep it along the chain and reset it with x
  (`UCUDA_NSET`). Converted: BD (`calculateDiscreteModelCUDA`, `...PeaksCUDA`, bif1d_cont), basins
  (`...AvgPeaksCUDA`), LLE/LS (classic, 2D, cont), Metrics (classic, cont), DFT (cont, h-sweep),
  Order (`perfIntegrateKernel` n*h; `orderEstimateKernel` substeps at t_n + k*h/2, t_n + k*h/4).
  Legacy kernels with defaulted loop arguments (distributed, IC, CUDA_H) start t at 0 per call.
- Engine: `time_prefix(body)` puts `#define UCUDA_NONAUTONOMOUS` before the template of the modules
  above (and the adaptive ones); `build_module` refuses a time body in a module without it (FastSync,
  Network). rdc: `kKrsFnsT` — `krs_step_t(X, ucuda_t, a, h)` and the adaptive `krs_*_t`.
- Adaptive sweeps: the driver's `S.t`, start 0. Adaptive continuation restarts each point at t = 0
  (the point's transient / record targets are absolute times), unlike the fixed-step chain.
- CPU: the `krs_cpu` step DLL of a time body keeps the step index itself (`thread_local`, t = n*h of
  the last nonzero h, so the DAE projection step with h = 0 sees its trajectory's t and does not
  advance it); `KrsCpuStep::step_index(v)` sets / reads it — 0 at each trajectory start, the main
  index for LLE/LS clones, explicit k / 2k / 4k for Order's three copies. CPU loops are unchanged.
  The adaptive CPU DLL (`AdaptiveCpuModule`) gets `UCUDA_NONAUTONOMOUS` and its own `UCUDA_TA/TP`
  (it has no cudaLibrary.cuh); CPU Analysis sends time systems to it and to `krs_cpu`.
- Verified on x'' + 0.2x' + x = cos(wt) (amplitude R(w) exact, LLE = LS = -0.1): BD 1D/2D, Metrics,
  LLE/LS (classic and cont), basins on GPU fixed step (rdc and one TU), adaptive DOP853 / GBS / Extr
  (grid and nodes, GPU and CPU), CPU fixed-step sweeps (= GPU to all digits); Order orders: RK4 4,
  Explicit Midpoint 2, SEMP / SIMP 2, D 1, CD 2, GBS 2-4 4. Lorenz sweep dumps (24 runs) bit-identical.
- Adaptive: with `#define UCUDA_NONAUTONOMOUS` the module's `ucuda_krs_rhs/emb/dprep` take
  `ucuda_t` before `a`, and the driver passes `S.t` / `t0 + h0` / `S.tp` through `UCUDA_TA(t)` /
  `UCUDA_TP(decl)` (empty otherwise, so autonomous modules preprocess to the old text). The emb body
  prints stage times `ucuda_t + Cs[i]*h`, f(Y) at `ucuda_t + h`; DOP853 dense stages use `CE[i]`.
  DOPRI78's cubic Hermite dense output is coarse on long steps (h ~ 0.8 → 5e-3 between nodes) —
  that is the interpolant, not time handling.
- t is never part of X: not in overflow/divergence thresholds, fixed-point check, error norms,
  DBSCAN, clones.
- `sysparse` strips `(t)` after a variable (`x(t)` → `x`) but keeps it after a math function
  (`sin(t)`).

## DAE (Library → DAE tab) — in progress
- `dae_ir.h` (header-only): `parse_dae` → `DaeSystem`; `as_codegen_system()` gives a `System` with
  `vars = [x; z]`, `rhs = [f; g]`, `dae_nx` = number of x (`-1` = ODE). `AppModel::build_system` returns
  it for `is_dae` records.
- A DAE method is an ordinary step body over X = [x; z] (`codegen_dae_scheme`), so it runs in the
  existing kernels with the usual a[] layout and Newton tail. Methods (`dae_scheme_names`):
  BDF1, Trapezoidal, SDIRK2, Implicit Midpoint, Radau IIA (5) (see "Stage 5" below), TR-BDF2 (two passes of `scheme_implicit_common` in their own blocks, h shadowed),
  Rosenbrock ROS2 (`dae_ros2`; with t it adds +-gamma h^2 F_t, F_t from `pn_diff` by t), Rosenbrock
  ROS3P (Lang-Verwer, order 3) and Rodas4 (rodas.f METH 1, order 4, stiffly accurate) through one
  emitter `dae_rosenbrock(s, RosTab)` in the Hairer-Wanner form (M/(gamma h) - J) u_i = F(t + alpha_i h,
  y + sum a_ij u_j) + M sum (c_ij/h) u_j + gt_i h F_t, y1 = y + sum m_i u_i (one LU per step; orders 3 / 4
  in x and z checked with and without t — with a CONSISTENT start: an inconsistent z0 drops every
  method to order 1, test harnesses project z first with the h = 0 step), CD and
  CD + Yoshida-4 (`dae_cd`, level B: D explicit forward with z re-solved after every x update, D*
  implicit reverse with one Newton on (x_i, z)), and explicit Euler / Midpoint / RK4 with the
  constraint solved per stage (`dae_explicit_rk`). CD and the explicit ones are index 1 only —
  `dae_require_index1` refuses a constraint with no z in it. Verified against the CD-DAE report:
  tables 4, 5, 7 reproduce (IE, trapezoid, CD, RK4+constraint, explicit RK2); CD + Yoshida-4 is order 4.
- Level A ("Solve as ODE", record field `dae_as_ode`): `dae_reduce_to_ode` solves each constraint that
  is linear in exactly one remaining z (z = -rem/coef, AST substitution, `pn_print` back to text) and
  returns an ODE in x; then `AppModel::build_system` hands that ODE to every tab and ODE scheme (CD,
  Comp/Yoshida, implicit, sweeps), with z kept in the state as outputs: `System::out_exprs` = z(x),
  vars = [x; z], rhs = f only. `codegen_scheme` steps x (`sys_core`) and then recomputes
  X[nx + j] = phi_j(X) at t + h (`outputs_tail`), plus `dae_nx_tag(nx)` — LLE/LS perturb x only and the
  clone projection (h = 0) just recomputes z. Adaptive (`codegen_adaptive`, GBS, Extr): the z rows get
  dz/dt = grad phi . f + dphi/dt (`sys_augmented`, `pn_diff` + `pn_print`; `copysign` from d|u| is now
  in `known_funcs`) for F and dense output, then Y_z = phi(Y), E_z = 0 (`adaptive_outputs_tail`). So
  every tab sees z as a variable: BD / Metrics output, axes, basins. Refused: ExtrZ (complex core),
  `SystemEvaluator` (CPU Analysis goes through `krs_cpu` / the adaptive DLL instead). Checked on Chua
  with the diode current: BD on x and LS bit-identical to the plain ODE, Metrics on z equal to the
  native DAE RK4 to 1e-10 (periodic points). The old Analysis observables
  (`PhaseAnalysisSession::observables/obs_exprs`, appended after the run) are no longer filled.
- Level B ("Solve as ODE" when a constraint is not explicitly solvable but every g has a z —
  `dae_index1_structural`): `System::zb_g` = g, rhs = f(x, z) (x rows), vars = [x; z]. The body starts
  with `zb_prelude`: a lambda `ucuda_zb(xs, t)` — full Newton on g(xs, z, t) = 0 (symbolic dg/dz, LU,
  Newton slots of a[]), warm start from X[nx + k], cached by (x, t). `build_namemap` maps z names
  (`zb_names`, set by `sys_core`) to `ucuda_zb_v[k]`, and `rhs_over` prints every f as
  `(ucuda_zb(st, t), (f))`, so each stage solves z for its own x; `zb_tail` writes z of the new state
  into X[nx + k]; tag nx as for outputs. Only schemes whose every f goes through `rhs_over`
  (`zb_scheme_ok`: explicit RK, DOPRI78, RK45, DOP853, GBS; Extr/Comp over them); CD, SEMP/SIMP/D,
  implicit and complex print f their own way and are refused (`kZbSchemeMsg`). Adaptive
  (`sys_augmented_b`): z rows dz_k/dt = -(sum dg_k/dx_i f_i + dg_k/dt) / (dg_k/dz_k) when each g holds
  only its own z (else refused), prelude added to rhs/emb/dprep, Y_z by Newton, E_z = 0. Checked: RK4 of
  level B = DAE RK4 (same method) to all digits on nonlinear coupling, Kepler and a two-z system with t;
  GBS 2-4 order 4; Mikhailov LS on GPU identical to DAE RK4, adaptive DOP853 Metrics = fixed RK4 to 1e-6.
- DAE editor (Library → DAE): auxiliary functions (`use_aux_funcs` / `func_defs_text`, inlined by
  `parse_dae(.., funcs)` exactly as for ODEs), a LaTeX preview (`expr_to_latex` / `name_to_latex` in codegen:
  \frac, ^{}, \mathrm{f}, |..|, x1 → x_{1}, greek names; with aux functions the raw rows are shown with the
  calls kept — `Parser::extra_funcs`, preview only), method checkboxes (`dae_schemes_off` in the record,
  a list of the OFF ones so old records keep all on; with "Solve as ODE" the ODE `scheme_*` flags via the
  shared `draw_builtin_scheme_checkboxes`) and Generate (`AppModel::generate` prints every enabled DAE method;
  a method that does not apply prints its reason instead of failing the rest).
- `min(a, b)` / `max(a, b)` are parser functions (normalised to `fmin` / `fmax`: NVRTC, CRT and dd/qd all
  have them), with a derivative for the implicit schemes (branch by `copysign(1, a - b)`) and evaluator
  opcodes; refused in complex schemes. Before them min was written as (a + b - |a - b|)/2.
- Mikhailov presets: "Mikhailov DAU_n" (original), "Mikhailov DAU_n (compact)" (same DAE via aux
  functions and min — equal to 2e-15) and "Mikhailov ODE 6D (Baumgarte)" (ODE: both constraints
  differentiated, VM and vd are variables, `-kb*g` keeps them on the constraint; RK4 agrees with the DAE to
  1e-10, |g| <= 4e-9 over t = 1500; every ODE scheme incl. CD). Neither constraint is explicitly solvable:
  the Schottky branch reaches 7-127% of the linear current and the diode drop 70-75% of lam*u1 on the
  attractor, so dropping them is not an option.
- Consistent initial z: `dae_consistent_z` (CPU, `SystemEvaluator` on a copy with dae_nx = -1) solves
  g(x0, z) = 0 before Analysis runs; index >= 2 is reported in `AnalysisResult::dae_status`.
- Guards: `codegen_scheme` / `codegen_adaptive` refuse `dae_nx >= 0`; `SystemEvaluator` refuses it (it
  would integrate g as a derivative). Analysis on the CPU builds the DAE body with `krs_cpu` instead.
  Other tabs with a native DAE get a valid step body but are untested; with "Solve as ODE" they are ODE.
- LLE/LS on a DAE method (body tag `// @ucuda-dae-nx`, `UCUDA_LYAP_NX`): clones are perturbed and
  orthonormalised in x only; `ucudaGramSchmidtDae` zeroes the z part, and every placed clone goes through
  `ucudaDaeProject` (z from the trajectory, then the step body with h = 0, which `codegen_dae_scheme` prints as
  BDF1 = Newton on g(x, z) = 0 with x fixed; empty for index >= 2). CPU twins: `cpu_gram_schmidt_dae`,
  `dae_project`. Carrying the z part as a linear combination broke on Mikhailov DAU_n: with lambda4 ~ -4..-7 the x
  part of a vector shrinks e^4 per block, the z noise does not, and the clone's diode voltage went to -4.6
  (sinh ~ 1e42, Newton failed, LS 999 for A11 >= 0.5). Without the projection, methods that use f(x_n, z_n)
  (trapezoid, TR-BDF2, ROS2, CD) start each block with an O(eps) inconsistent z: bias O(h/NT).
  Kernels still return N exponents (tail NaN); `trim_ls_spectrum` cuts the result to nx, and every LS CSV
  writer (streamed 1D rows, 2D cells, `write_curve1d_csv(.., K)`) writes nx columns too.
  Headless probes with `set_nvrtc_rdc` off compile a DAE LLE/LS module for Mikhailov in ~200-270 s (one
  monolithic module); with rdc on (set in `_app_config.json`; the code default is off) it is ~11 s. Why: see
  "How a sweep module is compiled".
- Stage 5, part 1: DAE SDIRK2 (Alexander, `dae_sdirk2`: two implicit Euler passes with gamma*h, like
  TR-BDF2), DAE Implicit Midpoint (`scheme_implicit_common` Midpoint with dae_nx — a control method: it
  does not hold an index-2 constraint and diverges on index 3, as the plan says), DAE Radau IIA (5)
  (`dae_radau5`: 3 stages, full Newton on 3N unknowns, x rows Y_i - x_n - h sum a_ij f(Y_j), constraint
  rows g(Y_i) = 0, loop-form LU 3N x 3N). With newton_full = 0 the body first runs RADAU5's simplified Newton
  (Hairer-Wanner IV.8): J at X once per step, T^-1 A^-1 T = diag(gamma, [[alpha, -beta], [beta, alpha]]) (T and
  the eigenvalues computed in codegen from the table and checked), a real N x N system (gamma/h M - J) and the
  complex one ((alpha + i beta)/h M - J) as a real 2N x 2N; if it diverges / is singular / does not converge in
  nmax, the same step goes through the old Newton from the predictor (`rd_done`). newton_full = 1 (DAE default)
  runs the old text. Orders unchanged (index 1: 5/5, 2: 5/3, 3 + GGL: 5/3), = full Newton to 6e-13 on the
  index-2 pendulum, converges without the fallback (dae_test). Speed (stage 10, Metrics sweep, GPU 2048 points /
  CPU 16): the simplified Newton does NOT pay off here - double pendulum GPU 0.48 s (full, sparse LU) vs 1.84 s
  (simplified), Andrews 6.0 vs 9.1 s, Mikhailov DAU_n 2.94 vs 2.84 s, pendulum index 2 0.11 vs 0.14 s; CPU the
  same picture (double pendulum 3.9 vs 5.6 ms per point, Mikhailov 29 vs 23). The full Newton gets the hybrid
  sparse LU of the 3N system, the simplified one factors dense N x N and 2N x 2N; keep newton_full = 1 (DAE
  default). GPU vs CPU per sweep point (CPU DLL, all cores): BDF1 9-24x, Radau IIA 14-20x.
  `System::dae_index` (from `dae_index_text`, `DaeSystem::as_codegen_system`) weights the Newton
  correction of an index-k variable by h^(k-1) (`dae_norm_w`), in `scheme_implicit_common` and Radau;
  index <= 1 systems print the old text. Orders (headless probe, self-convergence): Radau IIA index 1:
  5 / 5, index 2 (pendulum): x 5, z 3, index 3 (pendulum, double pendulum): x 3 (velocities), lambda 2 —
  the Hairer-Lubich-Roche orders 2s-1 / s / s-1. SDIRK2: x 2, z 1 on index 2.
- **LU bug (fixed in stage 5):** every printed LU (`scheme_implicit_common` loop and unrolled, Rosenbrock,
  Radau) and `integrator.cpp` `lu_factor` swapped WHOLE rows (with the L multipliers of earlier
  columns), while the solve replays the swaps one by one LINPACK-style — that needs the multipliers left
  in place. Whenever pivoting fired after the first column, the Newton correction (and the Rosenbrock
  stage) was wrong. ODE implicit schemes with I - hJ almost never pivot (no change there); DAE
  constraint rows always do. Symptoms it caused: ~9 Newton iterations per DAE step, the double pendulum
  "diverging at any h" on BDF1, Trapezoidal diverging on the index-3 pendulum. Now swaps cover columns
  >= k only. After the fix the double pendulum runs on BDF1, TR-BDF2, SDIRK2 and Radau IIA (|g| ~ 1e-16
  over t = 10, h = 1e-3); Trapezoidal and Implicit Midpoint still diverge there (not L-stable), Rosenbrock
  methods lose order on index 2-3 (they are index-1 methods).
- Stage 5, part 2. Core: `dae_deriv_array` (anonymous namespace) differentiates each constraint along
  x' = f (`pn_total_dt`: sum dc/dx_i f_i + dc/dt) until a z appears; the length of that chain is the
  constraint's index, `dae_structural_index(s)` the maximum (0: no z after 3 derivatives). On it:
  - Consistent initial values for index >= 2 (`dae_consistent_hidden`, called by `dae_consistent_z` when
    any chain is longer than 1): x0 is projected onto the z-free levels by minimum-norm Gauss-Newton
    (x -= C^T (C C^T)^-1 c), then z from the last levels by Newton; expressions are evaluated by
    `SystemEvaluator`s over chunks of N rows. Analysis shows "initial values: hidden constraints (index k):
    x0 projected (max shift ..)". Pendulum index 3 from x0 + 0.01: back on the circle, velocity tangent.
  - GGL: `dae_ggl_system` adds multipliers `ucuda_ggl_mu<k>` for the position-level (index-3)
    constraints: x' = f - (dg/dx)^T mu, 0 = g, 0 = d/dt g (index 2, mu = 0 on the solution). The
    methods "DAE BDF1 + GGL" / "DAE Radau IIA + GGL" keep mu local to the step (X shadowed by a copy
    [x; z; mu] with mu = 0), so the state and every tab stay as they are. Radau + GGL on the pendulum and
    double pendulum: x 5, lambda 3 (plain Radau: 3 / 2).
  - Level C ("Solve as ODE" when not structurally index 1): `dae_stabilized_index1` replaces an
    index-(m+1) constraint by (d/dt + alpha)^m g = 0 (binomial weights, a double root: Baumgarte), then
    levels A/B take over (`AppModel::build_system`). alpha: record field `dae_stab_alpha` (default 10,
    editor field shown for level C). RK4 order 4, |g| 1e-7..1e-9 over t = 20 (pendulums); a stiff system
    still needs a stiff ODE scheme (Circle stiff, lambda = 1e4, diverges on explicit RK4).
  - "DAE CD (index 2)" (`dae_cd2`): z constant per half-step, chosen (Newton with a finite-difference
    Jacobian, the report's secant) so that g holds at the end of the half-step; D explicit, D* implicit
    per variable (scalar Newton); X gets (z1 + z2)/2 — the mid-step z, so compared at t_n it converges with
    order 1 (the t_n - h/2 comparison is not checked yet); x order 2. Refused on index 1 (use DAE CD) and index 3.
  - Index 3 is refused by Trapezoidal, Implicit Midpoint, ROS2 / ROS3P / Rodas4 and CD (index 2).
- Stage 6, sparse LU: `plan_elimination` / `elimination_report` and `MatEntry` / `EliminationPlan` moved
  from `circuit_codegen` to header-only `sparse_lu.h` (the circuit code includes it, unchanged); new
  optional `diag_first` = rows eliminated first, each on its own diagonal. Codegen (`SparseLu`,
  `sparse_lu_build/decl/factor/solve`) uses a HYBRID for DAE Newton matrices: the x rows of I - hJ
  (diagonal ~1) are eliminated by the plan as straight-line code into `Asp[]` (multipliers in place), the
  rest — the z x z Schur block (`Asp[0..nb^2)`, rows `ucuda_rb`) — by a dense LU with partial pivoting
  at run time. Pure structural (Markowitz) pivots failed on the constraint block: index-3 rows are
  scaled ~h^2 after the x elimination, and Andrews squeezer did not pass the threshold on a single
  factorization (q2 = 0 makes G^T entries vanish). Phase 1 checks a threshold (|pivot| >= 1e-3 max|column|);
  a failure sends that factorization to the full dense LU (`ucuda_dn`; never observed on the stands).
  Used by `scheme_implicit_common` for DAE with N > 8 (N <= 8 keeps the unrolled dense text) and by
  Radau IIA (3N x 3N) always — unless the cost model says no: phase-1 ops + nb^3/3 must be under half of
  n^3/3 and phase 1 at most 4000 multiply-adds (Andrews Radau + GGL: 7884 ops looked 4.6x cheaper on paper
  and ran 1.4x slower — the straight-line code size, not arithmetic). `UCUDA_DAE_DENSE_LU=1` forces the
  dense LU (comparison knob). CPU probe, same results to the last digit: Andrews BDF1 114 -> 69 ms,
  Radau IIA 915 -> 250 ms (both match the Test Set q(0.03): Radau 1.4e-9), double pendulum BDF1
  27 -> 16 ms, Radau 590 -> 465 ms. GPU (stage 10, Metrics sweep 2048 points, RTX 2060, rdc on; same results,
  sums equal to 1e-12, Andrews Radau 1e-8): double pendulum BDF1 0.25 -> 0.095 s, Radau IIA 11.3 -> 0.48 s;
  Andrews (t = 0.003, h = 1e-5) BDF1 2.67 -> 0.21 s, Radau IIA 47.9 -> 6.0 s - the sparse LU gains more on
  the GPU (8-23x) than on the CPU: the dense 3N x 3N loop LU lives in local memory. Module build with rdc:
  Andrews Radau 53 s sparse vs 32 s dense (straight-line code), double pendulum 10 s vs 2 s.
- Stage 7, sweeps on a native DAE: `lyap_nx_prefix` (`#define UCUDA_LYAP_NX nx` from the body tag) now goes
  to the BD 1D / continuation / BD 2D / Metrics / basins modules too, not only LLE/LS. With it every sweep
  point makes z0 consistent with ITS parameters (the constraint depends on them): the DAE step with h = 0
  (Newton on g with x fixed; empty for index >= 2) at the end of `ucudaSetupSweepPoint` (BD, peaks,
  basins, classic Metrics), at the start of `LLEKernelCUDA` / `LSKernelCUDA`, and after each new
  parameter in the continuation kernels (bif1d_cont, lle1d_cont, ls1d_cont, Metrics cont; the start is
  the previous point's z — same branch). CPU twin `dae_point_start` in every CPU sweep (BD classic/cont,
  LLE/LS, DFT, Metrics). The user's z0 is only Newton's start, i.e. the branch rule. Axes / IC sweeps on
  an algebraic z are refused (`dae_z_axis`, `kDaeZAxisMsg`): 1D var sweeps, LLE/LS/BD 2D, Metrics, basins.
  GGL bodies write the extended state into `ucuda_xg[]` (not a shadowed `X`): `krs_cpu_check_indices`
  rejects X[i] beyond N. Headless GPU probe: a sweep from z0 = 0 equals per-point runs from a consistent
  z0 to 4e-15 (GPU, CPU, classic, continuation); double pendulum Metrics GPU vs CPU: BDF1 (sparse LU)
  7e-12, Radau IIA 2e-10, Radau + GGL 8e-14 — the stage 5-6 bodies compile and run under NVRTC. No root:
  the projection (h = 0 branch of `codegen_dae_scheme`) writes NaN into z when its Newton ends singular, non-finite
  or with a correction > 1e-3 |X| after all iterations; the sweep loops see that as divergence (dae_test
  `dae_no_root_point`: 0 = z^2 + p, p = 1 -> NaN / flag 0, p <= 0 measured; GPU and CPU). Steps with h > 0 are
  unchanged (an unconverged step is still accepted). DFT: classic GPU goes through
  `calculateDiscreteModelCUDA` (projected), the h-sweep and continuation templates (`dft1d_hsweep`, `dft1d_cont`)
  project too (the step with h = 0 under `UCUDA_LYAP_NX`; dae_test `dft_dae`: z0 = 0 equals a consistent z0 to
  the last bit on all six paths, GPU vs CPU 2e-8 - FMA in the trapezoid Newton).
- Stage 8, circuits: `spice_parse.cpp/.h` turns a SPICE netlist into the TEXT of a semi-explicit DAE
  (the same record fields a user types: equations, x, z, params, IC, h), so a circuit is just a DAE
  record and goes everywhere a DAE goes. MNA form: x = capacitor voltages `vc_<C>` and inductor
  currents `il_<L>`; z = node potentials `v_<node>`, capacitor currents `ic_<C>`, currents of
  voltage-defined branches `i_<name>` (V, E, H, B with V=, nullor outputs); rows: vc' = ic/C,
  il' = (Va - Vb)/L, KCL per node (sum of leaving currents), Va - Vb - vc = 0 per capacitor, the
  branch voltage equation per branch. Element values (R, C, L, gains, DC, SIN vo/va/f) are
  parameters named after the element (`r1`, `x2_c1` inside subcircuits) - sweepable. Supported:
  R C L, V/I with DC / SIN / PULSE (periodic: unrolled up to TSTOP, max 200) / PWL / EXP (all as
  min/max/exp of t, no fmod - implicit methods need derivatives), E/G (gain, VALUE=, POLY(n) in the
  SPICE 2 term order), F/H, B (V=/I=, V(a,b), I(Vx), time), X with .subckt params, .param, .ic,
  .tran, .include/.lib. E with |gain| >= 1e4 becomes a nullor (V+ = V-, free output current) - that
  is how `emit_spice_netlist` writes an ideal op-amp; op-amp subcircuits by name (TL072, LM741, ...,
  3 or 5 pins) are replaced by a nullor too, reported in the notes. D/Q/M/J are refused (stage 9).
  Loops of capacitors / voltage sources are reported (index 2). `spice_to_circuit_graph` reads an
  emitted netlist back into a `CircuitGraph` (round-trip test). `kicad_import.cpp/.h`: `.kicad_sch`
  goes through `kicad-cli sch export netlist --format spice` (found via UCUDA_KICAD_CLI, PATH,
  Program Files\KiCad\<ver>\bin; it blocks the UI thread for the export, ~1-2 s).
  Record fields `circuit_source` (path) and `circuit_netlist` (copy at import time);
  `AppModel::import_circuit` overwrites the DAE fields; the DAE editor has Circuit / Open / Re-read /
  Watch (mtime polled once a second) and an import report. Tests: `tests/u-cuda/spice_test.cpp` -
  RC with SIN / PULSE / PWL against RK4 of the scalar ODE (1e-8), integrator on TL072 inside a
  .subckt, F/H, refusals, and the round trip Lorenz -> synthesis -> netlist (B and POLY) -> graph
  (same topology signature) -> DAE: Radau IIA = `circuit_simulate` to 1e-5 V at t = 1.
  Not done: the KiCad exporter command (`U-CUDA.exe --import-netlist` into a running instance),
  L-I cutset detection, mutual inductance K, conditionals in B expressions.
- Stage 9a, devices and DC OP (`dae_devices.cpp/.h`): diodes (D, SPICE level 1 DC: IS, N, RS -> internal
  node `<name>#a`, CJO - see "Stage 9 tails") and BJTs (Q, NPN/PNP: Ebers-Moll transport form with Early
  VAF/VAR, IS/BF/BR/NF/NR, RB/RC/RE -> internal nodes, CJE/CJC constant) from `.model`; area factor. Model
  parameters become a[] parameters `<model>_<param>` once per model (zero-default ones like RS/VAF only if
  given); parameters that are set but not modelled (BV, TT, IKF, ISE, ...) are listed in the notes.
  Exponentials go through `dev_explim` (exp up to 40, tangent line above) - without it Newton from a far
  start overflows. MOSFET / JFET: stage 9d and "Stage 9 tails". Numbers in generated DAE text never use e-notation
  (`fmt` / `num` print `(m*10^(k))`): the equation parser reads `1e-12` as `1*e - 12`.
  Topology at import: a loop of voltage sources or a node with only current sources -> error; a node with no
  DC path to ground -> note. `dae_dc_operating_point` (any DAE: f = 0 and g = 0 in all variables; for a
  circuit capacitor currents and inductor voltages zero): Newton with row scaling (a circuit mixes 1/C ~ 1e9
  and 1/R ~ 1e-6 in one Jacobian), step limit and Armijo backtracking. `AppModel::import_circuit` uses it when
  the netlist has neither UIC nor IC= / .ic (as SPICE does); the DAE editor has a "DC operating point" button.
  `dae_consistent_z`: the line search now also accepts a full step when the correction is already below
  1e-12 (at the root max|g| sits at round-off and does not strictly decrease; the transistor stage from
  vc = 0 never left the loop). Tests (spice_test): rectifier vs the scalar ODE, CE stage NPN/PNP OP vs an
  independent 2D Newton (and PNP = -NPN), diode RS, topology errors, Chua with a negative G and antiparallel
  diodes = the same system written as an ODE (1e-6 at 1 ms). ngspice comparison: see "KiCad and ngspice".
- Stage 9b, adaptive step on a DAE: `codegen_adaptive_dae(s, DaeScheme::Rodas4)` (`adaptive_code_for_scheme`
  routes every native DAE there; `adaptive_scheme_name_supported("DAE Rodas4")`). The emb body is
  `dae_rosenbrock(s, T, emb = true)` - the same stages, `Y` instead of `X`, E = sum (m - mhat) u = u6
  (`RosTab::mhat`, rodas.f ERR = AK6, estimator order 3), all stages into W, F1 = [f; g](t + h, Y); the
  fixed-step text is unchanged (emb defaults to false). Dense output is rodas.f CONTRO:
  D = [y0, y1, d2.u, d3.u], y(th) = (1-th) y0 + th (y1 + (1-th)(D2 + th D3)) (`RosTab::d2/d3`), order 3
  in x AND z (Hermite from F would use g as a derivative). Index 1 only (the driver's error norm has
  no h^(k-1) scaling by variable index). Analysis only: CPU through the adaptive DLL (`dae_nx >= 0` goes to
  `AdaptiveCpuModule`), GPU through the usual phase_kernel_ad; z0 is made consistent on the CPU before the
  run, as for fixed steps. Sweeps: `fill_adaptive_request` adds `AdaptiveRequest::prefix` =
  `#define UCUDA_AD_DAE_NX nx`; then the driver (`ucuda_adaptive.cuh`) projects z onto g = 0 in `ucuda_ad_init` /
  `ucuda_ad_restart` (`ucuda_ad_dae_project`: Newton in z, dg/dz by forward differences of the rhs body [f; g],
  the module has no Jacobian) and every LLE/LS clone after placement / renormalisation; clone directions and
  their Gram-Schmidt are in x only (z part copied from the base, then projected). Without the macro the text is
  the old one. The same macro is set for DAE BDF sweeps (x-only clones). Tests (dae_test `dae_sweeps` run for
  BDF and Rodas4): Metrics from z0 = 0 = per-point consistent (1e-8), BD 2D, basins, LLE / LS -0.500000 /
  -2.000000 GPU = CPU. Tests (dae_test): RC exact u(5) within 1.4e-6 /
  2e-10 at tol 1e-5 / 1e-9 (19 / 163 steps), dense output on a 0.05 grid 6.5e-9 (CPU), the same on the
  GPU, Kepler coupling = fixed Rodas4 h = 1e-3, refusals (index 2, BDF1).
  Every other one-step DAE method takes the adaptive step too (Integration -> Adaptive step), by step doubling:
  `codegen_adaptive_dae_doubling(s, sch, name)` (codegen.cpp) prints the fixed-step body of `codegen_dae_scheme`
  three times into emb (one step of h -> YA, two steps of h/2 -> Y, the midpoint into W), E = (Y - YA)/(2^p - 1)
  with p = `dae_scheme_order` (x order: BDF1 / Explicit Euler 1, Trapezoidal / TR-BDF2 / SDIRK2 / Implicit Midpoint /
  ROS2 / CD / CD (index 2) / Explicit Midpoint 2, ROS3P 3, CD + Yoshida-4 / RK4 4, Radau IIA (+ GGL) 5), E of a
  variable of index k times h^(k-1) (as the DASSL / RADAU5 norms - so index 2-3 work with the generic driver), dense
  output = the parabola through X, W, Y (no derivative needed for z), F1 = [f; g](t + h, Y). Same Newton, projection
  and index rules as the fixed step (each method refuses what it refuses there). `adaptive_code_for_scheme` routes
  every DAE method except Rodas4 there; `adaptive_scheme_name_supported` / `adaptive_scheme_q_guess` know them, so
  Analysis (GPU phase_kernel_ad, CPU DLL) and the sweeps (UCUDA_AD_DAE_NX projection) take them unchanged. Cost:
  3 steps per attempt. dae_test `adaptive_doubling_cpu`: all 13 one-step methods on RC follow the tolerance (order 1:
  global ~ sqrt(tol)), index-2 pendulum Radau IIA 1.2e-9 at tol 1e-8; GPU = CPU (Trapezoidal, same step count,
  1e-13); `dae_sweeps` on Trapezoidal (LLE -0.500002, LS -0.5 / -2.00001).
- Stage 9c, multistep BDF 1-5 (`multistep_bdf.h`, header-only, its own driver - the user wants a
  separate multistep kernel for ODEs too): variable order 1..5 and variable step in the DASSL form
  (DDASTP): history = modified divided differences phi on the real nodes, coefficients alpha / beta /
  gamma / sigma / psi recomputed after a step or order change, no history rescale; corrector
  y' = y'_pred + cj (y - y_pred), x rows y'_x - f(y) = 0, z rows g(y) = 0 (an ODE has none), Newton with
  the full symbolic Jacobian (`SystemEvaluator`, CPU), matrix cj I - f_y | g_y; err = ck ||y - y_pred||;
  order k-1 / k / k+1 from the sigma estimates, initial phase (order +1, h x2 per step); WRMS norm with
  h^(k-1) by variable index; dense output = DDATRP interpolation. Index 1-2 directly; index 3 through
  `dae_ggl_system` (multipliers mu appended to the state from 0, dropped from the output) - pendulum
  index 3 at tol 1e-7 / 1e-9: 117 / 240 steps, x 4e-7 / 1.2e-8 against Radau IIA + GGL, |g| 2e-10 / 3e-12.
  The first version was a Nordsieck form (LSODE): its rescale Z[j] *= eta^j left an undamped sawtooth of
  lambda at index 2 - at tol 1e-7 / 1e-9 the pendulum locked into h ~ tol (3156 / 31015 steps, error
  7e-4 / 7e-5) while 1e-6 / 1e-8 / 1e-10 were fine; Robertson took 2779 steps.
  Names `kBdfSchemeOde` "BDF 1-5" (always in the ODE scheme list) and `kBdfSchemeDae` "DAE BDF 1-5"
  (after the DAE methods); `scheme_is_multistep`; Analysis (`run_adaptive_phase` branch, both Fixed and Adaptive
  step - fixed step below; GPU when "use GPU" is on) and the sweeps (`fill_adaptive_request`, below); a fixed-step
  body (`krs_for_scheme_impl`) and `adaptive_code_for_scheme` refuse them, so do the other tabs.
  Tests: x' = -2x + cos t (order reaches 5, error ~1-2 tol, 89 / 339 steps at 1e-5 / 1e-9), Robertson DAE
  vs Test Set at t = 40 (rtol 1e-8: 326 steps, y1 to 9e-9), index-2 pendulum vs Radau IIA at tol 1e-6 ..
  1e-10 (87..322 steps, x error ~20 tol, lambda ~100 tol), index-3 pendulum via GGL.
  GPU: the algorithm lives in `kernels/ucuda_bdf.cuh` - one text for CPU and GPU, a template over a provider
  (res / jac / emit / cancelled): CPU = `multistep_bdf.h` (SystemEvaluator), GPU = `NvrtcEngine::run_phase_portraits_bdf`
  (`BdfGpuRequest`, bodies from `codegen_bdf_bodies`: F = [f; g] and the full symbolic J, kernel `bdf_phase_kernel`,
  thread per trajectory, DASSL workspace in thread-local memory; `computePhasePortraitsBdfNVRTC`). Index 3 is
  turned into GGL on the host (`bdf_prepare_system`), mu start at 0 and are cut from the output; h_min failure
  comes back in stats slot 2. GPU = CPU on the stands: the same step counts (164 / 196 / 326), max diff ~2e-12.
  Step log for Analysis ("Step size" projection, `AnalysisResult::step_log`): `UcudaBdfCore::log / log_cap /
  log_n`, one record per attempt [t at the step start, h, err, code 1 accepted / 0 rejected], written in
  `ucuda_bdf_step`; only `ucuda_bdf_run` turns it on (optional arguments), sweeps and clones keep nullptr. CPU:
  `bdf_integrate(.., log, log_cap)`, GPU: `BdfGpuRequest::log_cap` -> `PhaseAdaptiveResult::log`. dae_test: log
  length = 4 (accepted + rejected), codes = stats, GPU log = CPU log; trajectories unchanged. The log starts with the
  step that crosses t_skip (`ucuda_bdf_run` switches `C.log` on there; the adaptive drivers log from t = 0 and
  `trim_step_log_to_record` cuts their transient), so h(t) shares the time axis of the other projections.
  User h_min (Analysis "h_min" -> `BdfSettings::hmin` / `BdfGpuRequest::hmin` -> `UcudaBdfParams::hmin`): the step
  never goes below it; an attempt at h_min that fails the error test is accepted as forced (code 2, `nforced`), like
  the adaptive driver; Newton not converging at h_min ends the run ("lower h_min"). Sweeps keep hmin = 0 (only the
  10 ulp(t) guard). dae_test `bdf_hmin_and_skip`.
  Newton of the corrector = the tab's Newton fields (the a[] tail, `newton_slot`): `UcudaBdfParams::nmax / nfull /
  ntol` - at most nmax iterations, converged when ||dy|| <= ntol ||y|| (as the DAE bodies), nfull = 0 = Jacobian and LU
  once per attempt (`ucuda_bdf_lu_factor` / `_apply`, LINPACK-style pivots); Analysis reads them from `a`
  (`BdfSettings::newton_*`, `BdfGpuRequest::newton_*`), sweeps from a[] via `#define UCUDA_BDF_NSLOT`. nmax <= 0 keeps
  the old DASSL form (6 iterations, WRMS < 1e-3, Jacobian every iteration) - the tests' defaults, bit-identical.
  Fixed step (Integration -> Fixed step, `UcudaBdfParams::hfix` = h; Analysis `BdfSettings::hfix`, sweeps
  `fill_adaptive_request(.., h_fixed)` -> settings pinned to h + `#define UCUDA_BDF_FIXED`, `sweep_krs` gives the
  stub body for BDF in both modes): start = the same BDF with error control (tolerance min(rtol, atol, 1e-10), h <=
  hfix) up to t0 + 5 hfix, y(t0 + j hfix) taken by DDATRP (`ucuda_bdf_fixed_collect`), the order-5 history built
  from them as backward differences (constant-step DASSL differences), psi_i = i h; then no controller at all - no
  error norm, no order choice, no rejections, step hfix, order 5; Newton failure ends the run ("lower h"). Tried and
  dropped: orders 1..5 on the full h (global order 2: the order-1 first step leaves O(h^2)) and a ramp doubling h
  from h/1024 (BDF 4-5 are not zero-stable for step ratio 2 - diverged). dae_test `bdf_fixed_cpu`: pendulum index 2
  at h = 0.02 / 0.01 / 0.005: 1.5e-5 / 4.0e-7 / 1.2e-8 (order 5), start ~75-100 steps; GPU = CPU 3e-13;
  `dae_sweeps` with fixed BDF (BD 1D peak 2.008620, LLE -0.50006, LS -0.50006 / -1.99997). The start of the fixed
  step forces a step at 100 x the 10-ulp guard (no failure on a steep front), and `ucuda_bdf_start` raises an
  automatic h0 below 100 ulp(t) (a circuit from the zero state: y' ~ 1e15 gave h0 under the guard, 0 for y' = inf).
  Sweeps (Parametric: BD 1D/2D classic and continuation, Metrics, basins, LLE/LS): the adaptive sweep
  modules run on BDF without new kernels - `ucuda_adaptive.cuh` switches to the BDF section of
  `ucuda_bdf.cuh` under `UCUDA_BDF_DRIVER` (the same names `UcudaAdaptState`, `ucuda_ad_init / restart /
  step / eval / advance_to`; S.F0 = y' of the DDATRP interpolant at the node; steps stop exactly at tEnd -
  DASSL TSTOP), and replaces `UcudaLyapClones` / `ucuda_lyap_point` by a BDF version: per block of NT the
  base trajectory steps to the block end, each clone restarts BDF (order 1, z projected) from its
  renormalised state and steps to the same end. Without the macro the header text the compiler sees is the
  old one. `fill_adaptive_request` builds the BDF request: `codegen_bdf_bodies(.., "D")` - rhs = [f; g],
  dprep slot = the Jacobian written into D, a stub deval (the CPU module reads an empty one as "no dense
  output"), no controller, `AdaptiveRequest::prefix` = `#define UCUDA_BDF_DRIVER 1`, `UCUDA_BDF_NX`,
  `UCUDA_BDF_KIDX {..}`; the engine puts the prefix before the module text, into the cache keys and into the
  CPU DLL prelude, and registers `ucuda_bdf.cuh` as a virtual NVRTC header. The template still needs a
  fixed-step body: `sweep_krs` gives BDF1 (DAE) / Euler (ODE) for BDF in both modes (its kernels are never
  called; fixed step goes through the BDF driver with `UCUDA_BDF_FIXED`). Every sweep point starts with
  z projected onto g = 0 (Newton in z, x fixed), so a DAE sweep from one z0 = a run from a consistent z0 per
  point. Index 3: `fill_adaptive_request` builds the bodies over `bdf_prepare_system` (GGL) and sets
  `AdaptiveRequest::extra_x` = number of multipliers; every sweep builder calls `sweep_extend_state` (amountOfX
  + extra_x, IC tail = 0), results are read by the old indices (dae_test `bdf_index3_sweep`: pendulum Metrics
  max x = sin 1 = Analysis, GPU and CPU). LLE/LS note: clones are independent BDF
  trajectories, so their accuracy relative to the perturbation is ~ tol / eps - take tol << eps (the
  adaptive step controls the clone error jointly; BDF cannot).
  Tests (dae_test): Metrics on x' = -x + z, 0 = z - p (cos x + 1) from z0 = 0 = per-point consistent runs
  (1e-8, GPU and CPU); BD 1D van der Pol mu = 1: peak 2.008620 (classic GPU, continuation GPU and CPU); LLE
  and LS on x1' = -x1 + z, x2' = -2 x2, 0 = z - 0.5 x1: -0.4985 / -2.0007 (exact -0.5 / -2), CPU vs GPU 1e-5.
  BD 2D (DAE van der Pol mu x k: equilibrium / period 1) and basins (r' = -r (r^2 - 0.36)(r^2 - 1.44): equilibrium
  inside r = 0.6, cycle outside) on BDF and on adaptive Rodas4.
- Stage 9d, op-amps "as is" plus an ideal copy (user's request): `SpiceImportOptions::ideal_opamps` (record
  field `circuit_ideal_opamps`, JSON "1"). Off (default): an op-amp from the table or `uc_opamp` whose
  .subckt is in the netlist is expanded element by element (the vendor model); on: it is a nullor. An op-amp
  without a .subckt is a nullor either way (reported). The DAE editor: "Ideal op-amps" checkbox (re-imports)
  and "Save ideal copy" (a new library record "<name> (ideal op-amps)"). For vendor macro models: JFET (J,
  NJF/PJF, Shichman-Hodges written with min/max - no branches; vds < 0 swaps drain and source; gate diodes;
  RD/RS internal nodes; CGS/CGD constant; VTO < 0 for both types, the type gives the sign), F/H POLY(n) on
  source currents, .model inside .subckt (registered globally). DC OP: the line search now always tries
  13 halvings from the initial (step-limited) lambda - with a nearly singular Jacobian that lambda was
  already below the old absolute floor 1/4096 and no step was tried; a non-finite correction is refused.
  Test (spice_test): JFET saturation / linear / P = -N, F POLY(2), an inverting amplifier x(-10) with a
  JFET-input macro model of my own named TL072: as is -0.999988 V, ideal -1.000000 V.
- Stage 9 tails (devices): MOSFET level 1 (M d g s b, NMOS/PMOS, W/L/M, `mosfet_drain_current` = the JFET
  half-formula with BETA = KP W/(2L) and a separate threshold per half - vbs forward, vbd reverse - for the body
  effect GAMMA/PHI, `mosfet_threshold`; RD/RS, constant overlap CGSO/CGDO; LEVEL != 1 is refused, not
  replaced). Junction capacitances of D (CJO/VJ/M/FC), Q (CJE/VJE/MJE, CJC/VJC/MJC) and J (CGS/CGD/PB/M) are
  voltage-dependent (`junction_capacitance`: the SPICE depletion formula, tangent above FC*VJ, min/max form;
  `capacitor()` takes an expression with "@V" = its own voltage; PNP / P-JFET pass "(-@V)"). Mutual inductance
  K (any groups): coupled inductors get z `dl_<L>` = il' and rows V = L dl + sum k sqrt(Li Lj) dl_j (index 1,
  |k| < 1). Tests (spice_test `devices_tails`): MOSFET saturation / linear / PMOS / body effect exact,
  diode C(v) charging to -5 V and +1 V = scalar ODE to 1e-9, transformer K = 2D ODE to 1e-9.
  Conditionals in B / VALUE= (`CondRewriter` in spice_parse.cpp): comparisons, && || !, c ? a : b, if(c, a, b),
  u(), uramp(), limit(), sgn() become branch-free expressions with st(u) = (1 + copysign(1, u))/2 (a >= b:
  st(a - b); a > b: 1 - st(b - a); == exact via st(a-b) st(b-a)); expressions without them print as before.
  `pn_diff` now differentiates copysign(a, b) (sgn a sgn b da). L-I cutsets (`loops()`): contract every branch
  except inductors and current sources; an inductor between different super-nodes is reported (index 2).
  Tests (spice_test `b_conditionals`, `li_cutsets`): comparator -> RC = scalar ODE (4e-6, the jump costs O(h)),
  an if/&&/!/limit/uramp node exact, single '=' refused; series L-L through a bare node reported, RL not.
- KiCad and ngspice (live, KiCad 10.0.6 in %LOCALAPPDATA%\Programs\KiCad, found by `kicad_find_cli`): test
  `tests/u-cuda/kicad_test.cpp` (skip 77 without KiCad). (1) Every demo of share\kicad\demos\simulation goes
  .kicad_sch -> kicad-cli -> `spice_import`: 11 of 20 import; refused with a reason: XSPICE A elements (5),
  VDMOS (2), TRNOISE, PULSE with 29750 periods; Q17 fails in kicad-cli itself (KiCad's own library). (2) ngspice:
  no separate install - KiCad's `bin\ngspice.dll` (shared ngspice) loaded at run time, the API declared in the
  test; the same netlist runs in ngspice (gear, reltol 1e-7, maxstep = our h) and in our import (Radau IIA, DC OP
  or UIC as AppModel). Max diff / signal range: KiCad rectifier 5e-5, KiCad generic_opamp_bip 9e-5, BJT CE with
  C(v) 1.3e-5, MOSFET with body effect 1e-7, JFET 4e-8, K 1.4e-5, B ?: 8e-5, SFFM 1.4e-4. AM: this ngspice build
  returns a constant VA for AM(..) at t > 0 (not the manual formula, which the import follows) - the test prints
  it and does not compare. Probe mode: UCUDA_NG_PROBE=<netlist> UCUDA_NG_NODE UCUDA_NG_T [UCUDA_NG_N].
  Import changes for the demos: .model parameters are kept as text and evaluated per use in the instance scope
  (Is={Ices/2} with .subckt params; a model from a .subckt is per instance, its a[] names prefixed; unused XSPICE /
  text-field models no longer break loading), .param of a .subckt evaluated before its elements (PSpice LM317),
  R with an expression value (r='TIME > 350m ? 8 : 89' -> I = V/R(t, V)), "DC ac=1" key=value on sources, SINE,
  AM, SFFM, PSpice "VALUE {..}" without '=', K with |k| = 1 (ideal transformer, index-2 note), clear refusals
  for A elements, VDMOS and noise sources.
  `.ic` without UIC follows SPICE: the DC operating point is solved with the .ic nodes HELD (the node's KCL row
  replaced by v - V = 0, `SpiceImportResult::op_plain_text`), then released - `AppModel::import_circuit` runs
  `dae_dc_operating_point` on that text and takes all initial values from it; before, the import kept the .ic
  values and zeros elsewhere (no OP). This is how KiCad's astable multivibrator (`.ic v(/3)=0`) gets its kick: with
  the fix BDF 1-5 adaptive oscillates like ngspice (v(/1) at t = 0.2 / 0.5 / 0.7 / 1.1 / 1.4 s: 3.814 / 3.428 / 3.694
  / 3.820 / 3.520 V, ngspice 3.818 / 3.448 / 3.700 / 3.822 / 3.514). Its switching front needs steps down to
  ~1e-14 s (pF junction capacitances): fixed-step BDF stops there at the same t for any h (2e-6 .. 5e-5; damping
  the Newton did not help) - use the adaptive step on such circuits. spice_test `ic_without_uic`.
  Field layout (UI only, `param_groups.h`): initial conditions are grouped like parameters (`SystemRecord::ic_groups`,
  JSON "ic_groups"): the import puts capacitor / inductor states in "Capacitors" / "Inductors" (or the subcircuit
  group), junction capacitances of a device in "Devices/<name>", every z in "Algebraic (computed)/Node potentials" /
  "/Branch currents" (their IC is only Newton's start - recomputed by the DC OP / projection). Import groups are
  ordered by kind (`Builder::group_rank`: R, C, L, K, sources, controlled, devices, subcircuits, models, algebraic);
  after that the vector order is the screen order and group headers are dragged among siblings in every field tree and
  in the Groups window (`pg_reorder`; `pg_get` appends a new group after its siblings, `pg_rename` works in place).
  Field names are drawn as formulas (`draw_field_name`, `math_formula_draw`): a user label (`field_labels`, JSON
  "field_labels", edited in the Groups window) or `name_to_latex` - the part after the first '_' is the subscript
  (vc_c1 -> vc_{c1}, vc_q1_cje -> vc_{q1,cje}; also in the DAE equation preview); inside a subscript the renderer no
  longer drops trailing digits a level lower. KiCad unnamed nets lose the "net-" prefix (`net_short`:
  net-_led1-k_ -> v_led1_k); re-importing an old circuit record renames those variables.
  The import also gives labels with the netlist's case (`SpiceImportResult::field_labels`): R / C / L values
  R_{1}, C_{1}, L_{1} (`elem_tex`), element variables vc_{C1}, ic_{C1}, il_{L1}, i_{V1}, vc_{Q1,cje} (`var_tex`);
  `import_circuit` adds them to `field_labels` where the user has none. A circuit record saved before IC groups gets
  them on load (`AppModel::circuit_layout_defaults`: re-parse of `circuit_netlist`, names it does not give go to
  "States" / "Algebraic (computed)"); nothing else in the record changes. Projection variable checkboxes (Analysis)
  use the same labels. Settings -> "Use LaTeX for parameters and variables" (`AppConfig::field_latex`,
  `set_field_latex`): off = plain names. Variable combos of the projections (Phase 2D / 3D axes, feature /
  continuation / recurrence variable) go through `var_combo_begin` / `var_selectable` / `var_combo_preview`
  (formula preview via `ImGuiComboFlags_CustomPreview`); plot axis and series names through `plot_var_name` (TeX label
  only when the plot LaTeX labels are on too). `set_field_labels(&model.field_labels)` each frame makes
  `field_label(nullptr, ..)` use the loaded system's labels. The Time domain variable checkboxes follow `ic_groups`
  (a "variables" tree). Settings -> "Initial conditions of algebraic variables (DAE)" (`AppConfig::show_alg_ic`,
  default off): `AppModel::ic_fields` drops the z (alg_text) from the IC fields of the tabs (Analysis, System,
  `draw_named_num_fields`) with a "N algebraic variables hidden" note; their values stay in the session and are
  recomputed anyway; the library editor shows all. Right click on a group header of the Time domain variable tree
  (or on "variables"): Select all / Deselect all / Only these (`draw_param_tree(.., group_menu)`). Combo previews
  draw the formula at the size of the field labels on the UI text baseline (`field_tex_draw(.., fit)`), clipped to
  the frame.
  SI prefixes in every numeric field (`num_parse.h`: `si_prefix`, `scale_pow10`): f p n u/µ m k M G T and meg/Meg
  (SPICE mega), case-sensitive as in SI (m milli, M mega), only right after a number ("2pi", "5asdf", "1kk" stay
  invalid). Exact: a negative prefix divides by an exact 10^k, so "10n" is the very double 1e-8. The ↑/↓ digit step
  keeps the prefix (`DigitInput::ComputeStep`). The import writes R / C / L values with prefixes (`fmt_num_si`:
  shortest text that parses back to the same double: 15k, 10n, 4.7m). Every reader of field strings must go through
  `parse_num` (`std::stod("10k")` = 10): AppModel DC OP / consistent start and the tests were switched. The SPICE
  value parser (`spice_parse.cpp`) scales the same way (10u = exactly 1e-5; it multiplied by an inexact 1e-6).
  DAE equation preview: symbols use the record's `field_labels` (`expr_to_latex(.., labels)`, part of the preview
  cache key), the derivative dot sits on the base only (`dot_latex`: \dot{vc}_{C1}), (m*10^(k)) prints as 10^{k} /
  m \cdot 10^{k} (10*10^(-13) folds to 10^{-12}), and `\frac` places numerator / denominator by their real extent
  (`layout_extent`), so a fraction inside a denominator no longer runs into the numerator.
  Device simplifications (`SpiceImportOptions::no_junction_caps / ideal_diodes / ideal_bjts`, record fields
  `circuit_no_jcaps / circuit_ideal_diodes / circuit_ideal_bjts`, JSON "1", DAE editor checkboxes that re-import,
  "Save simplified copy" = all three as "<name> (simplified)"): no barrier capacitances of D / Q / J / M; ideal diode
  = threshold Vf + Ron (`ideal_diode_current`: softplus over kIdealVs = 10 mV; `ideal_junction_fit`: the tangent of
  the Shockley exponential with RS at kIdealIref = 10 mA), parameters `<model>_vf / _ron`, no internal RS node; ideal
  BJT = Ebers-Moll on two such junctions (Ic = Dbe - Dbc (1 + 1/BR), Ib = Dbe/BF + Dbc/BR), `<model>_vfbe / _ronbe /
  _vfbc / _ronbc / _bf / _br`, no Early effect, no RB / RC / RE. spice_test `device_simplifications`: rectifier loses
  the C(v) state, the ideal diode is exact at 10 mA, ideal-BJT CE stage Vc 4.06 V vs 3.81 V full (Early dropped).
  On the astable multivibrator they do NOT help: without junction capacitances the regenerative switching has no fast
  state left - the algebraic equations lose their root at the switching instant (an impasse point; the DAE must
  jump), so even the adaptive step stops there; ideal BJTs without RB / RE put the junction capacitances into loops with
  C1 / C2 (index 2) and shift the period by a few %; scaling the junction capacitances from pF to 0.1 - 10 nF changes
  the step count only from 11.2k to 10.0k (2 s, BDF rtol 1e-6, ~0.8 s CPU) - the cost is the exponential switching
  itself, as in SPICE. Useful for circuits without regeneration (amplifiers, rectifiers, filters with diodes).
  Still open: GUI check.
- Stage 12 (planned, not started), NI Multisim import - the same role as KiCad. Found on this machine: Multisim 14.2
  (`%ProgramFiles(x86)%\National Instruments\Circuit Design Suite 14.2`, 133 samples `.ms14` in
  `%PUBLIC%\Documents\National Instruments\Circuit Design Suite 14.2\samples`). Differences from KiCad: `.ms14` is a
  closed binary format and there is no CLI exporter; there is a COM automation API (`MultisimInterface.MultisimApp` /
  `MultisimCircuit`, 32-bit in-proc `MSInterface.dll` - not loadable into the x64 exe, so a 32-bit helper process,
  like `kicad-cli`). Methods (from the type library, not tried yet): `Connect`, `OpenFile`, `ReportNetlist(probes,
  Text|CSV, file)` (looks like a component/pin/net table, not SPICE), `DoCommandLine(cmdfile, logfile)` (XSPICE
  commands - maybe the way to the simulated SPICE netlist), `EnumComponents`, `RLCValue`, `SaveAs`. Plan:
  (1) probe: open 2-3 samples through COM from a 32-bit script (Multisim starts on screen; may stop on a license
  dialog), see what `ReportNetlist` / `DoCommandLine` give; (2) the manual path regardless: Multisim's own SPICE
  netlist export (`.cir`) -> `spice_import`, adapt to its dialect on the samples; (3) if COM gives a netlist:
  `circuit_source_load` accepts `.ms14` (Open / Re-read / Watch) through the helper; (4) test like kicad_test
  (skip without Multisim), compare with ngspice where the netlist allows. Expected limits: XSPICE code models
  (A elements: switches, logic, part of the virtual parts) are refused, as for the KiCad demos.
- Stage 11, global testing (2026-10-30): `tests/u-cuda` has 11 suites (`run_tests.bat` with no arguments runs all,
  ~20 min): gpu_smoke, krs_cpu, adaptive, architecture, sessions, tabs, dae, spice, kicad, library. New ones:
  `sessions` (every system.json and all 242 session JSON files round-trip unchanged), `tabs` (Order: Euler ~1 /
  RK4 ~4 GPU and CPU, Performance, stability region on Dahlquist 2D; Network ring of 16 Lorenz nodes),
  `library` (every library system - ODE, map, DAE - through BD 1D / LLE on GPU and CPU, LS, Metrics, basins, DFT
  on GPU, plus the records' custom KRS; 884 runs). Byte-level regression probes (scratchpad, not in the repo):
  codegen dump of all 56 ODE library systems (1906 bodies incl. CPU, adaptive, complex core, GBS) identical to
  b95ed43; DAE dump of 20 systems identical after removing exactly the intended blocks (root check of the
  projection, RADAU5); Lorenz sweep dump (24 runs) and Duffing with t identical to the stage-2 baseline.
  Found by them: `InputTextMultilineStr` (gui.cpp) gave ImGui a "length + 4096" scratch buffer, so pasting more
  than 4096 characters into an empty multiline field silently cut the text at 4095 - the custom KRS CRK7s8_L of
  Lorenz-CxSearch in library/ (gitignored user data) is cut mid-expression and does not compile. Fixed with
  `ImGuiInputTextFlags_CallbackResize` (the string grows, imgui_stdlib style); the record itself is left as is.
  Closing (2026-10-30): suite 12 `prune` (monolith with kernel pruning = full build, bit for bit; the only suite
  that runs the app default rdc off - the others set rdc on; a test must not be named after a library it links:
  `nvrtc.exe` left `nvrtc.lib` in build\ and the linker took it for CUDA's). numb = float build: compiles again
  (`multistep_bdf.h`: the BDF CPU provider calls `SystemEvaluator` through double buffers, `bdf_eval`); checked
  by building into separate dirs (`-p:OutDir=x64\ReleaseFloat\ -p:IntDir=U-CUDA\x64\ReleaseFloat\`, configCUDA.h
  switched only for the build) and running the suite against those objects with `/DUCUDA_NUMB_TYPE=float`:
  gpu_smoke, krs_cpu pass; adaptive (dense output to 1e-8) and dae (orders by self-convergence) fail on their
  double-precision tolerances - float limits, not breakage. GUI by hand: `tests/u-cuda/GUI_CHECKLIST.md`.
  Debug is not built (by agreement).
- Newton for DAE: the predictor of `scheme_implicit_common` (BDF1, Trapezoidal, TR-BDF2, SDIRK2,
  Implicit Midpoint) is X_n for x and z alike, like Radau. The explicit-Euler x predictor the ODE
  schemes keep threw Newton far from the root on stiff circuits: "DAE-1 PUT burst neuron" at h = 5e-4
  got the internal PUT node to ~1.4 V (currents ~e^40) at a spike front, and 25 full-Newton iterations
  did not converge; the step was accepted unconverged and the run turned to noise. DAE presets and new DAE systems use full Newton,
  25 iterations. The DAE body also stops on a relative test (||dX||^2 < tol^2 |X|^2): on index 2-3 the
  absolute tol is below the round-off of the correction.

## Adaptive step (RK45 / DOPRI78 / DOP853)
Fixed step stays the default and must stay **bit-for-bit** unchanged: the adaptive path is
a separate module/branch everywhere, never an `if` inside the fixed-step kernels.

- **Codegen:** `codegen_adaptive(sys, scheme)` → `AdaptiveCode` with four bodies — `rhs`
  (f), `emb` (one attempt: higher-order `Y`, error estimates `E = h*sum((b - b^)k)`, FSAL
  `F1`), `dprep`/`deval` (dense output) — plus `q`, `nlo`, f-counts. Only schemes with an
  embedded estimate (`scheme_supports_adaptive`). The same bodies feed the CPU driver
  (`integrator.cpp`, `CpuAdaptiveKrs`).
- **Extrapolators:** `GBS 2-4 ...` (`codegen_adaptive_gbs`) and `Extr(base|n1..nK)`, K >= 2
  (`codegen_adaptive_extrapolation`, base = built-in or custom KRS) are embedded pairs built from the
  same stages: Y over all K stages (weights of the fixed step), the lower solution over the first K-1,
  `E = sum (alpha_k - beta_k) T_k`, q = `extrapolation_order(K-1)`; F1 = f(Y), cubic Hermite dense
  output. The base body is printed into `emb` with a local `numb* const X` shadowing the const input.
  Y is printed exactly like `wrap_extrapolation` / `scheme_gbs`, so a step pinned to h (h0 = h_max = h,
  `UCUDA_AD_NO_RETRY`, clip 0) is bit-identical to the fixed Extr / GBS on CPU (GBS takes f(X) from F0).
  Entry by name: `adaptive_code_for_scheme(custom_schemes, sys, scheme)` (analysis_session) — all
  callers go through it; `adaptive_scheme_name_ok` / `adaptive_scheme_hint` for UI and setup errors.
  Analysis on CPU: the exe driver is Butcher-table only, so extrapolators go through the cl.exe DLL —
  entry `ucuda_cpu_ad_phase` (`AdaptiveCpuModule::phase`), a line-for-line copy of `phase_kernel_ad`;
  RK pairs keep the exe driver. CPU == GPU in accepted/rejected counts, grid samples to ~1e-8 on Rossler
  at t = 150 (node times differ ~1e-6: the float controller). ExtrZ (`codegen_adaptive_extrapolation_complex`):
  the same over the complex core (stages and both sums `ucmplx`, Y and E are their Re, symmetric weights),
  built-in complex CD bases only; ExtrZ(Complex CD4|1,2,3) (q = 6) needs 136 steps at tol 1e-10 on Rossler.
  Not yet: Comp.
  `compile_adaptive` feeds configCUDA.h when `emb` holds `ucmplx` (Extr over a complex CD base).
- **Driver:** `kernels/ucuda_adaptive.cuh`, one text for GPU (NVRTC) and CPU (exe, cl.exe
  DLL). Two parts: the *layout* (`UcudaAdaptParams`, controller in/out/memory, error norms,
  built-in controllers — Hairer / SciPy / I / PI / Filter) and the *driver*
  (`ucuda_ad_init`, `ucuda_ad_step[_x]`, dense output). A module includes it twice:
  `#define UCUDA_ADAPT_LAYOUT_ONLY` + include, then the user controller
  (`{{CTRL_CUSTOM}}` → `adaptive_ctrl_source`), then the full include.
- **Macros:** `UCUDA_AD_STATIC_N` (state in registers; only the Analysis kernel — in
  sweeps it kills occupancy), `UCUDA_AD_NO_DENSE` (LLE/LS, Performance), `UCUDA_AD_EXACT_CTL`
  (controller math in double — for step-by-step comparison with Hairer's dop853.c / scipy;
  by default norms and `pow` run in float with a double fallback).
- **Modules:** sweep kernels are `bifurcation2d.template.cu` + `adaptive_part.cu` (+ the
  metrics / LLE-LS tail); placeholders `{{KRS_RHS_BODY}} {{KRS_EMB_BODY}} {{KRS_DPREP_BODY}}
  {{KRS_DEVAL_BODY}} {{CTRL_CUSTOM}}`. Every body (incl. the controller body) must be in the
  module cache key. Analysis uses `NvrtcEngine::compile_adaptive` (phase kernel / endpoint
  kernel for Order → Performance).
- **LLE/LS:** clones step through `ucuda_ad_step_x` with `UcudaLyapClones` — same attempt,
  same embedded method, and the controller also sees the clones' error (`err_extra`).
  Without that the step grows to the stability limit near a stable equilibrium and the
  exponents come out ~0.
- **Controller library:** global, `library\step_controllers.json`; entries are a C body of
  `ucuda_ctrl_custom(in, m, o)` or a named Soderlind filter. Sessions embed the definition
  (`ctrl_def`) and import it where it is missing; if the library has the name with a different
  definition, the session gets its own entry (one with the same definition, or "<name> (session)"),
  so a session runs the controller it was saved with (`import_session_ctrl`). The file is replaced
  in one step (`filesystem::rename`) under a writer lock. `nvrtc_check_ctrl_body` = the editor's
  Check. PI/Filter: `safety` sets the target error `safety^(q+1)`, it does not multiply rho.
  A C body may have a **prepare section** `ucuda_ctrl_custom_prep(c, q, k)`: run once per trajectory
  (`ucuda_ad_prepare_ctl` in init/restart), `k[0..UCUDA_CTL_NK-1]` lands in `UcudaCtlConst::k` and the
  body reads it as `in.k[]` — the same role as `UcudaCtlConst` for the built-ins (no double divisions
  per attempt). Prepare + body travel as ONE string (`adaptive_ctrl_pack`: prep, `\x1e`, body) through
  every `ctrl_body` field, cache key and DLL hash; `adaptive_ctrl_source` / `make_ctrl_source` unpack
  it. With `UCUDA_AD_HAIRER_ONLY` (default) the built-ins are Hairer only, but C bodies work:
  `ucuda_step_ctrl` compiles exactly one law per module (the custom one when the module has its body,
  else Hairer), no switch; Soderlind filters (they need the built-in Filter) are hidden. The presets
  "Hairer fast (C)" and "Hairer numb (C)" are the built-ins written out one-to-one ("numb (C)" uses the
  very same body/prep strings): all sweep modes and both CPU paths are bit-identical to the built-in,
  same registers and speed (Rossler DOP853 stands).
- **Built-in Hairer, two variants:** "Hairer fast" (`UCUDA_CTRL_HAIRER`, the default) and "Hairer numb"
  (a C body + prepare shipped as a built-in `AdaptiveCtrlInfo` with `body/prep`, id `UCUDA_CTRL_CUSTOM`;
  works wherever library bodies work). Same law in the log domain:
  `h_new = h * 2^clamp(-expo1/2 * log2(err²) + C, log2 fac1, log2 fac2)`, where C = `beta*log2(facold) +
  log2(safe)` is stored in `m.user[0]` on acceptance; after a rejection the upper bound is
  `min(log2 fac2, 0)`. All logs of the parameters are taken once in `ucuda_ad_prepare_ctl`
  (`UcudaCtlConst::he/lsafe/lfac1/lfac2/lc0`). "fast" does the norm and logs in float
  (`ucuda_ctl_lg2err2`, fallback to double on overflow/NaN), "numb" in numb (`ucuda_ctl_lg2err2_numb`).
  Differences are ulp-level: on Lorenz (T = 2, tol 1e-3..1e-11, RK45/DOP853, CPU/GPU) steps, rejections
  and f counts equal the old controller; y(T) differs by ≤ 1.3e-7 at tol 1e-3, ~1e-14 at tight tol.
  `UCUDA_AD_EXACT_CTL` keeps the old double version (dop853.c comparison).
  The law does not clip to h_max — the driver does (`if (h > P.hmax) h = P.hmax`); "no limit" is
  `hmax = +inf` (`adaptive_build_params`), so the controllers need no `hmax > 0` test.
  `o.err` of Hairer is **err²** (the driver only checks it is finite; the log of attempts shows t, h and
  the code, not the error). Sessions: `"ctrl_v":2`; an old "Hairer" loads as "Hairer fast" (no ctrl_v)
  or "Hairer numb" (ctrl_v 2).
- **New built-in controllers: simplify to the bone.** The controller sits on the critical path of every
  attempt (norm → law → h, which the next attempt waits for), so:
  - everything that depends only on parameters, q or constants goes to prepare (`ucuda_ad_prepare_ctl` /
    `UcudaCtlConst`, or the prep section of a C body) — logs of safety/fac bounds, exponents, products
    of parameters, the state before the first accepted step;
  - no divisions, square roots or `pow` per attempt where avoidable: work in the log2 domain
    (`h * 2^clamp(...)`, clamps of logs instead of factors), with the norm squared (`err²`, compare with
    1, `log2 err = 1/2 log2 err²`), multiply by precomputed reciprocals;
  - store in controller memory what the next attempt needs in its final form (e.g. `beta*log2 facold +
    log2 safe`), not raw values to be transformed again;
  - no checks the driver already does (h_max clamp, finiteness of the error);
  - ship a float version ("fast") and a numb version, plus C-body presets that reproduce each one
    bit for bit (check on the Order → Performance stand: steps, rejections, f and y(T)).
- **CPU:** `AdaptiveCpuModule` (`krs_cpu`) builds `kernels/adaptive_part.cu` itself into a cl.exe
  DLL — placeholders substituted, `PeakStream` cut out of `kernels/cudaLibrary.cu`, the engine's
  `peak_config_defines()` as prelude, `par_or_var` a thread-local. Entries: endpoint (Order →
  Performance), `ucuda_lyap_chain` (LLE/LS), `ucuda_cpu_ad_bif` (1D bifurcation: classic over
  threads, continuation as one chain — line-for-line copies of the GPU kernels). Adaptive
  continuation belongs on the CPU: the GPU version is one thread, ~25-50x slower.
- **Regression:** after touching anything shared, compare the fixed-step dumps against the
  baseline — any mismatch outside the intended change is a bug.

### Pitfalls found the hard way
- **Layout of a[]** in every tab: `[s, p_1..p_M, n, newton_tol, newton_max_iters, newton_full]` —
  a[0] = symmetry s, a[M+1] = CD FP iterations, a[M+2..M+4] = Newton settings of the implicit schemes
  (`NewtonSettings` in `codegen.hpp`, filled by `scheme_fill_tail` / `scheme_append_tail`; size
  `M + 1 + kSchemeTailSlots`). Every slot exists for every scheme, so a[] size never depends on the
  scheme. Newton settings are per tab (fields in the Integration block); the system editor only holds
  the defaults for new tabs. The generated step reads them at run time (no recompilation), and so does
  the CPU `SystemEvaluator`. A builder that sizes a[] by hand reads past its end on an implicit scheme.
- Implicit schemes: LU/solve are printed unrolled for N <= 8 (Am in registers; loop form above). With
  numb = float Newton also stops at ||dX||^2 < (16 eps)^2 |Xn|^2 — float cannot reach the usual 1e-10,
  and without the floor every step ran all max_iters passes with a Jacobian rebuild on most of them.
- `loopCalculateDiscreteModel_int` (and CPU `cpu_loop_model`) take **one more step after
  the loop** (fixed-point check). A block of k steps is k-1 iterations; a transient of N
  steps is N-1 (and no call for N = 0). Step counts from time go through
  `ucuda_steps_per_block` (round to nearest), not `(size_t)(T/h)`.
- NVRTC `curand` stubs differ per template: LLE/LS templates mix the subsequence (point
  index) into the seed, the bifurcation/basins ones **ignore it**. Anything random per point
  built on a bifurcation template needs its own generator (see `ucudaLyapRng`).
- Finite-T LLE/LS depend on the random initial frame as ~ln(1/c)/t_max — point-to-point
  noise, not a bug. `vector transient` (renormalised but unsummed blocks) removes it.
- Output mode of adaptive sweeps (`AdaptiveRequest::raw_nodes`, "uniform grid / step nodes" in
  the Integration block): BD 1D/2D, basins and Metrics. Step nodes have no dense output per sample
  and no warp divergence on the grid, so they are 4-5x faster (Metrics 2D 128x128: 3.4 → 0.8 s).
  Metrics on nodes (`MetricsAccumNU`, `ucudaAdMetricsRun` in `metrics_adaptive_part.cu`): mean and
  variance are TIME averages (Hermite quadrature with the exact derivative f, 4th order). A plain
  mean over nodes is biased towards small steps. Extrema and peaks are interpolated between nodes
  (`ucudaAdNodeVertex`, mode = peak interpolation), and the decimator keeps every N-th node. Nodes
  are as accurate as the grid or better (Rössler, periodic: mean 2.5e-5 vs 2.7e-4 against a fine
  fixed step). CPU and GPU agree only to ~1e-5 in peak times, because nodes land differently.
- Warp divergence, not memory, is what slows adaptive sweeps: neighbouring lanes need steps,
  rejections and samples at different moments. One attempt per loop iteration
  (`ucuda_ad_try_x`) made LS 2D 1.5x faster, but the same restructure made the bifurcation kernels
  1.3-1.6x *slower* — measure every loop-shape change (results stay bit-identical either way).
- The clone error control asks for `rtol*|delta|`; below double resolution of the perturbation
  (`rtol*eps << 1e-16*|x|`, e.g. rtol 1e-9, eps 1e-8) the step used to collapse chasing roundoff
  (~200x more steps, same exponents). The scale now has a per-component floor
  `max(rtol*RMS(delta), K*eps_machine*max|x_i, y_i|)`, K = `UCUDA_LYAP_NOISE_K` = 100. It is active
  at ordinary settings too (rtol 1e-8, eps 1e-6): steps ~10x longer, exponents within the
  finite-T scatter. The norm uses weights 1/sc_i^2: one division per clone, plus one per
  component where the floor is active (LS 2D GPU 5-8% faster; results bit-identical on the stands).
- **Step-node kernel** (`raw_nodes` with a system/IC sweep: `ad_nodes_module`, prefix
  `kAdNodesModulePrefix` = `UCUDA_AD_NO_DENSE` + `UCUDA_AD_NODES_KERNEL` + `UCUDA_AD_STATIC_N`):
  own module for BD 1D/2D, basins and Metrics. The step settings come by value as a kernel argument
  (`__grid_constant__`, `UCUDA_AD_P_ARG`, launch arg `ad_param_arg`) — no per-thread copy; a sweep
  over a step setting (rtol/atol/controller parameter) goes to the general module. Hairer or a C body
  from the library, no dense output, no attempt log, no h/err history (kept for a C body), no tp/hp. The observable is picked by weights
  (`ucudaAdObsW`): `v[writableVar]` with a runtime index (and a select loop, which the compiler folds
  back into it) kept the whole state in local memory. Result: 124 registers, 0 B local memory
  (was 214 / 1584 B), bit-identical results. The per-step progress atomic into mapped host memory
  (PCIe) and its BSYNC were the main stall. Rossler 256x256 DOP853: BD 2D kernel 1.65 -> 0.73 s,
  Metrics 1.64 -> 0.98 s; at real clocks the kernel is now FP64-bound (85%).
- **Progress and cancel in all adaptive GPU kernels:** progress is reported per point
  (`ucudaProgressTopUp` at the end of the point; no ticks inside it); the threshold check (`ucudaAdOut` /
  `ucuda_lyap_out`) and the cancel flag run every `CHECK_INTERVAL` accepted steps (counted in steps,
  not output samples) plus once after the last step. `S.diverged` (non-finite error at h_min) is
  still tested every step: it is set by the step itself and ends the loop. Cost of the old scheme
  (measured with the kernels given `nullptr` signals): adaptive LLE 2D 128x128 0.49 -> 0.93 s,
  BD / Metrics 2D on the output grid 7.2 -> 8.7 s, LS 2D +5-15%; fixed-step kernels and the
  step-node kernel: no measurable cost.
- **ncu locks clocks to base by default**: a kernel limited by memory/latency (or by atomics to host
  memory) looks FP64-bound there. Use `--clock-control none` and check with nsys kernel times.
- The step body (`adaptive_emb_body`) is unrolled and skips zero Butcher coefficients (DOP853: ~30%
  of stage-combination FMAs). Bit-identical for finite stages; a trial step whose stage overflows to
  inf used to give NaN (0*inf) and now a finite error — adaptive LLE 2D on Rossler differed in 51 of
  16384 points (chaotic ones).
- Adaptive sweep kernels are **FP64-bound** (ncu, Rossler 256x256 DOP853 on RTX 2060: FP64 pipe
  84% busy, 214 registers, 8 warps/SM). Register caps (168/128), block width 64/128 and
  `UCUDA_AD_STATIC_N` do not help or hurt. The uniform output grid is ~4x slower than step nodes
  because of warp divergence: the step (~410 FP64 instructions) and dprep run for 2-4 of 32 lanes
  (81% of warp instructions). Tried and reverted (results bit-identical, all slower at 256x256):
  step-then-samples loop (7.6 -> 7.9 s), the same with `__all_sync` before every attempt
  (-> 13.1 s: the warp waits for the lane with the longest sample loop and the kernel becomes
  local-memory latency bound), dense coefficients in registers (255 registers, no gain). Two-phase grid (all lanes step in lockstep — `__activemask` / `__any_sync` rounds of W samples — buffering dense coefficients, then samples in a separate pass; bit-identical): under ncu 20% faster (warp instructions /2.3, 14 vs 6 lanes per instruction), but 5-30% slower in real time — it turns local-memory latency bound, and boost clocks speed up only the FP64-bound one-phase kernel. ncu locks base clocks: judge loop changes by nsys/wall time. Unexplained: the first launch in a process runs the two-phase kernel ~35% faster than later ones (not `CU_CTX_LMEM_RESIZE_TO_MAX`, not the L1 carveout, not the chunking).
- Order → Performance, "adaptive loses to fixed": check three things before suspecting a bug.
  (1) Chaos: E(T) grows ~e^(λT). Rössler (λ ≈ 0.07) at T = 400 gives E(T) ~ 1..10 (attractor size) for
  every h and tol, so the curves are noise; use T ≈ 20. (2) The problem: on Rössler at equal E(T) the
  adaptive step needs about as many f evaluations as the fixed one (RK45 ~10% more, DOP853 10-30%
  fewer). The spike is where local error control spends steps, but errors there decay. Van der Pol
  μ = 10 is the counter-check: adaptive needs 3-5x fewer f evaluations and is faster in time.
  (3) Cost per attempt on a cheap RHS (CPU, 3D): the RK45 scheme ~50 ns, the controller ~50 ns,
  bookkeeping ~20 ns, against ~41 ns for a fixed RK45 step, i.e. ~20 vs ~7 ns per f. The controller
  cost is the latency of the chain norm → log2 → exp2 → h, which the next attempt waits for. The CPU
  versions of log2/exp2/fmin/fmax/nextafter in `ucuda_adaptive.cuh` are inline (Estrin, ≤ 0.5 ulp
  float) because the CRT calls cost ~100 ns more per attempt in the /MD exe.
- **Order → Performance timing (GPU):** the kernels time themselves — `%globaltimer` around the step
  loop (`perfIntegrateKernel`) / init + loop (`endpoint_kernel_ad`), atomicMin start / atomicMax end over
  replicas; launch latency (~6-8 us on WDDM) and loads/stores are outside. Before measuring, a burst of
  back-to-back launches (`perf_burst_warmup`, ~300 ms before the first node, ~20 ms before the rest):
  short launches with a sync after each never raise the GPU clock, and the same kernel measured 3-5x
  apart (Lorenz RK45 adaptive: 16 vs 3 us per step). E(T) of a fixed-step tab is against the same
  dd/qd DOP853 y*(T) as the adaptive one when "reference y*(T)" is set (needs "fit h to t_max").
- Default `h_min` is `max(10 ulp(t), 1e-12*span)`: 10 ulp alone let a controller pinned at an
  unreachable tolerance take ~1e15 steps. A user `h_min` is raised to 10 ulp(t): below that
  `t + h_min == t` and a forced step never advanced time.
- **Cancel in Analysis / Order → Performance (adaptive):** `phase_kernel_ad`, `endpoint_kernel_ad`,
  the DLL entries `ucuda_cpu_ad_phase` / `ucuda_cpu_ad_endpoint` and `computePhasePortraitCPU_adaptive`
  check a cancel flag every 1024 accepted steps (GPU: an int in mapped host memory, `NvrtcEngine`
  polls the stream and sets it; DLL: a volatile int set by a watcher thread). A cancelled run returns
  `kNvrtcCancelled`; `PhaseAnalysisSession::poll` keeps the previous plots. The grid loop there is
  `ucuda_ad_advance_to` unrolled (same calls, same results). TDR is not addressed: a single launch
  longer than the Windows watchdog is still reset by the driver.
- `__grid_constant__` only from compute_70 (`UCUDA_GRID_CONST`); on older GPUs the parameter is a plain
  by-value one (same results, maybe a local copy).
- Prewarm must never activate a module (`compile_*_if_needed(..., activate = false)`): a running task
  reads its kernel from the active slot after its own compile.
- New `.h`-only files need no `.vcxproj` change; new `.cpp` files do — ask first.

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
  - `vars_text` / `params_text` — two comma-separated lists: state variables
    (they become `X[0..N-1]` in the KRS) and parameters (`a[1..M]`, in this
    exact order when `param_order` is `AsInAlphabet`). They must match what
    the equation parser derives from the right-hand sides: `refresh_symbols`
    takes the UI/session lists from here, while `build_system` re-derives the
    kernel order from the equations, and a disagreement silently shifts `a[]`.
  - `alphabet_text` — **legacy**, one mixed list; still read for old records,
    and it still wins over `vars_text`/`params_text` in `parse_alphabet`.
    Leave it empty in new presets.
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
- **Nsight Compute (`ncu`, in PATH):** GPU counters are enabled on this machine. Under the Russian
  Windows locale `ncu` crashes with "bad conversion" when it PRINTS values (console or
  `--import --csv`); profiling itself works. Write a report (`ncu -o rep ...`) and read it with the
  Python module `ncu_report` (`<Nsight Compute>\extras\python`). Kernel regex: `--kernel-name regex:Ad`,
  plus `--launch-count 1 --kill on`.
- **Profiling knobs (env, off by default, kernel code unchanged when unset):**
  `UCUDA_NVRTC_LINEINFO=1` — line info in the modules (SourceCounters per source line);
  `UCUDA_NVRTC_MAXRREG=<n>` — register cap (NVRTC, cuModuleLoadDataEx and cuLink).
  `CUDA_CACHE_DISABLE=1` to measure a cold module build (the driver JIT cache hides it).

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