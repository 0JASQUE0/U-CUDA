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
  (`KrsCpuStep`, double/dd/qd) и тела пользовательского регулятора (`CtrlCpuFn`)
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
1. User inputs system (LaTeX/plain/OCR) → `sysparse` normalizes the text and splits it into `System{vars, params, rhs}`. This is string-level work (regex/char scanning): `sysparse` builds **no** AST, and `rhs` are plain strings.
2. `codegen` parses those strings into its own AST (`Node`/`Parser`, private to the anonymous namespace in `codegen.cpp`) and emits the **body of the integration step** — that is, the body of `calculateDiscreteModel`, and nothing else.
3. This code is substituted into template from `kernels/<task>.template.cu` at the `{{KRS_BODY}}` placeholder
4. `nvrtc_engine` compiles, returns `CUfunction`
5. `parametric_engine` / `phase_portrait_nvrtc` launches kernel, retrieves result into GPU buffers, passes to renderer
6. Renderer uses the same memory via OpenGL interop—no copies
7. **Post-Build Event** copies runtime data to `<solution>/x64/{Debug,Release}/` (= `OutDir`): `ocr_server.py`, the whole `kernels/` folder, plus `cudaLibrary.cu`, `cudaLibrary.cuh`, `cudaMacros.cuh` and `configCUDA.h` into `kernels/`—NVRTC picks them up at runtime. Note: `library/` is **not** copied (it is resolved from source at runtime), and neither is `hostLibrary.cuh`.

### Where linearization actually comes from
**LLE/LS do not use a Jacobian.** They linearize with perturbed clone trajectories plus Gram-Schmidt (`LLEKernelCUDA` / `LSKernelCUDA` in `cudaLibrary.cu`) — the Benettin/Wolf method. No variational equations, no monodromy matrix anywhere.

Symbolic differentiation does exist, but it serves **only the implicit schemes** (`Implicit Euler`, `Implicit Midpoint`): `pn_diff` / `jac_over` in `codegen.cpp` emit an N×N Jacobian into the step body for the Newton solve. For anyone extending it:
- build results through the `pn_*` peephole constructors, but do **not** make `pn_add`/`pn_sub` fold — the CD emitter depends on their exact shape; zero-folding for the chain rule lives in the `jd_*` wrappers;
- `d(fabs)` emits `copysign(1,u)`, not the device-side `sign()` — that one is invisible to the phase-portrait NVRTC path, the CPU bytecode and the bare-`cl.exe` KRS path;
- `floor`/`ceil`/`fmod` are rejected at codegen time (`jac_check_differentiable`).

---

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
  at t = 150 (node times differ ~1e-6: the float controller). Not yet: ExtrZ, Comp.
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
  (`ctrl_def`) and import it where it is missing. `nvrtc_check_ctrl_body` = the editor's
  Check. PI/Filter: `safety` sets the target error `safety^(q+1)`, it does not multiply rho.
  A C body may have a **prepare section** `ucuda_ctrl_custom_prep(c, q, k)`: run once per trajectory
  (`ucuda_ad_prepare_ctl` in init/restart), `k[0..UCUDA_CTL_NK-1]` lands in `UcudaCtlConst::k` and the
  body reads it as `in.k[]` — the same role as `UcudaCtlConst` for the built-ins (no double divisions
  per attempt). Prepare + body travel as ONE string (`adaptive_ctrl_pack`: prep, `\x1e`, body) through
  every `ctrl_body` field, cache key and DLL hash; `adaptive_ctrl_source` / `make_ctrl_source` unpack
  it. With `UCUDA_AD_HAIRER_ONLY` (default) the built-ins are Hairer only, but C bodies work:
  `ucuda_step_ctrl` compiles exactly one law per module (the custom one when the module has its body,
  else Hairer), no switch; Soderlind filters (they need the built-in Filter) are hidden. The preset
  "Hairer (C)" is the built-in float arithmetic one-to-one: all sweep modes and both CPU paths are
  bit-identical to the built-in Hairer, same registers and speed (Rossler DOP853 stands).
- **CPU:** `AdaptiveCpuModule` (`krs_cpu`) builds `kernels/adaptive_part.cu` itself into a cl.exe
  DLL — placeholders substituted, `PeakStream` cut out of `kernels/cudaLibrary.cu`, the engine's
  `peak_config_defines()` as prelude, `par_or_var` a thread-local. Entries: endpoint (Order →
  Performance), `ucuda_lyap_chain` (LLE/LS), `ucuda_cpu_ad_bif` (1D bifurcation: classic over
  threads, continuation as one chain — line-for-line copies of the GPU kernels). Adaptive
  continuation belongs on the CPU: the GPU version is one thread, ~25-50x slower.
- **Regression:** after touching anything shared, compare the fixed-step dumps against the
  baseline — any mismatch outside the intended change is a bug.

### Pitfalls found the hard way
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
- Default `h_min` is `max(10 ulp(t), 1e-12*span)`: 10 ulp alone let a controller pinned at an
  unreachable tolerance take ~1e15 steps.
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