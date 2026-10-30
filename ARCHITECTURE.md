# U-CUDA: code map

How the pieces connect and where to look. `CLAUDE.md` holds the rules and the history of each feature;
this file is the map. All names are functions/types — search them with grep, line numbers drift.
Paths are relative to `U-CUDA/` unless stated.

## 1. Layers

```
app_main.cpp            window, GL, main loop, AppConfig load/save
   │
gui*.cpp (ImGui)        draws AppModel, never computes          ── gui_internal.h (shared prototypes)
   │  reads/writes
AppModel (app_model.*)  editor state, one *Session per tab, run queues, build_system / generate / import_circuit
   │  owns
*Session                analysis_session.* (Analysis, BD, LLE, LS, Metrics, DFT, Basins, FastSync),
                        order_session.*, network_session.*, custom_session.*
   │  build a Request, run it on a worker thread (run_async), poll() each frame
Engines                 ParametricEngine (parametric_engine.*)   — every sweep + Order + Network + FastSync
                        NvrtcEngine (nvrtc_engine.*)             — Analysis phase portraits (fixed, adaptive, BDF)
                        integrator.* / krs_cpu.* / multistep_bdf.h — CPU paths
   │  step bodies from
codegen.cpp/.hpp        System → step body text (fixed / adaptive / DAE / BDF / complex / CPU)
   │  System from
sysparse.* (ODE text), dae_ir.h (DAE text), spice_parse.* + kicad_import.* (circuits → DAE text)
   │  pasted into
kernels/*.template.cu + cudaLibrary.cu (+ ucuda_adaptive.cuh, ucuda_bdf.cuh) → NVRTC → CUfunction
```

Rule of thumb: a numeric bug lives in codegen / kernels / engine; a "wrong setting reached the kernel"
bug lives in the session's `build_*_request` / `fill_adaptive_request`; a "value not saved" bug lives
in `session_io.cpp` or `system_library.cpp`.

## 2. One frame

`app_main.cpp` loop → `draw_gui(model, lib, cb)` (`gui.cpp`):
1. polls: `model.phase_session.poll()`, `poll_all()` of the parametric sessions, `poll_phase()` of basins,
   `poll_parametric_prewarm()`;
2. queues: `start_next_in_{parametric,basins,fastsync,order,network,custom}_queue()` (`app_model.cpp`) start
   the next queued item when the previous one finished;
3. tab dispatch on `AppModel::app_mode` → `draw_<tab>_controls` + `draw_<tab>_plot*` (table below);
4. `gui_process_deferred` (after the frame): actions that must not run mid-frame (load/save, re-import).

Run button → `AppModel::start_<tab>_analysis()` → session `run_async(engine, idx)` → worker thread →
`ParametricEngine::run_*` → result stored in the session → next frame `poll()` moves it into the config
the plot draws from (`apply_*_result`).

## 3. Tabs

| Tab (AppMode) | GUI | Session / request builder | Engine entry | GPU (template → kernel) | CPU twin |
|---|---|---|---|---|---|
| Library | `gui_system.cpp` (`draw_library_list/editor`, DAE editor, Generate) | `AppModel` editor fields, `SystemRecord` | — | — | — |
| Analysis | `gui_analysis.cpp` (`draw_phase_controls`, `draw_projection_windows`) | `PhaseAnalysisSession`: `snapshot_phase` → `compute_phase_portrait` / `run_adaptive_phase` | `phase_portrait_nvrtc.*` → `NvrtcEngine` | `phase_kernel`, `phase_kernel_ad`, `bdf_phase_kernel` (sources inline in `nvrtc_engine.cpp`) | `computePhasePortraitCPU` (SystemEvaluator), `_custom` (krs_cpu DLL: custom KRS, DAE, t), `_adaptive`; `AdaptiveCpuModule::phase`; `bdf_integrate` |
| Parametric: BD 1D/2D | `gui_parametric.cpp` | `BifurcationAnalysisSession`: `build_bif1d_request` / `build_bif2d_request` | `run_bifurcation_1d/2d` → `Impl::run_bif1d` / `run_bif2d` | `bifurcation1d` / `bifurcation2d.template.cu` → `calculateDiscreteModel[Peaks]CUDA`, `dbscanWarp[2D]CUDA`; cont: `bifurcation1dContinuationKernel`; adaptive: `adaptive_part.cu` `...PeaksAdCUDA` / `...PeaksAdContCUDA` | `run_bif1d_cpu`, `run_bif1d_continuation_cpu`, `run_bif1d_cpu_ad` |
| Parametric: LLE 1D/2D | same | `LLEAnalysisSession`: `build_lle1d/2d_request` | `run_lle_1d/2d` | `lle1d`/`lle2d` → `LLEKernelCUDA`; cont `lle1dContinuationKernel`; adaptive `lyapunov_adaptive_part.cu` `lyapunovLLEAd[Cont]CUDA` | `run_lle1d_cpu`, `run_lle1d_ad_seq` |
| Parametric: LS 1D/2D | same | `LyapunovSpectrumAnalysisSession`: `build_ls1d/2d_request` | `run_ls_1d/2d` (+ `trim_ls_spectrum` for DAE) | `ls1d`/`ls2d` → `LSKernelCUDA`; cont `ls1dContinuationKernel`; adaptive `lyapunovLSAd[Cont]CUDA` | `run_ls1d_cpu`, `run_ls1d_ad_seq` |
| Parametric: Metrics | same | `SignalMetricsAnalysisSession` | `run_signal_metrics` | `signal_metrics.template.cu` → `calculateDiscreteModelMetricsCUDA`, `signalMetricsContinuationKernel`; adaptive `metrics_adaptive_part.cu` | `run_metrics_cpu`, `run_metrics_cpu_ad`, `run_metrics_continuation_ad` |
| DFT 1D | `gui_dft_basins.cpp` | `Dft1DAnalysisSession`: `build_dft1d_request` | `run_dft_1d` → classical / continuation / h-sweep | classical via `calculateDiscreteModelCUDA`; `dft1dContinuationKernel`, `dft1dHSweepKernel` | `run_dft1d_cpu` |
| Basins | `gui_dft_basins.cpp` | `BasinsAnalysisSession`: `build_basins_request`, recluster, phase slots | `run_basins`, `run_basins_recluster` | `basins.template.cu` → `calculateDiscreteModelAvgPeaks[Ad]CUDA`, `dbscanCUDA` | — (GPU only) |
| FastSync | `gui_fastsync.cpp` | `FastSyncAnalysisSession` | `run_fastsync` | `fastsync_attr/grid.template.cu` → `fillFSMasterTrajectory`, `calculateDiscreteModelforFastSynchroCUDA` | — |
| Order | `gui_order.cpp` | `OrderAnalysisSession` (`order_session.cpp`) | `run_order`, `run_performance`, `run_stability` | `order.template.cu` → `orderEstimateKernel`, `perfIntegrateKernel`, `stabilityRegionKernel`; adaptive Performance: `endpoint_kernel_ad` (NvrtcEngine) | dd/qd branches via `krs_cpu` |
| Network | `gui_network.cpp` | `NetworkSession` (`network_session.cpp`) | `run_network` | `network.template.cu` → `networkIntegrateKernel` (`{{COUPLING_BODY}}` from `codegen_coupling`) | — |
| Custom | `gui_custom.cpp` | `CustomSession`: owns its own Bif/LLE/LS/phase/basins sub-sessions (2D level, 1D slices, level 3) | same entries as above | same | same |
| Settings | `gui_widgets.cpp` / `gui.cpp` | `AppConfig` (`app_config.*`) | `set_nvrtc_rdc/fmad`, `set_peak_config`, `set_gpu_block_size` | — | — |

Engine dispatch pattern (each `Impl::run_*`): validation → `continuation` → `adaptive.enabled` →
`use_cpu` → GPU fixed step. The CPU branches return before `ensure_init`, so they work without CUDA.

## 4. From text to a step body

```
SystemRecord (system_record.h)
  └─ AppModel::build_system()  (app_model.cpp)
       ODE:  parse_system_from_latex (sysparse) → System
       DAE:  parse_dae (dae_ir.h) → DaeSystem::as_codegen_system()  (vars = [x; z], dae_nx)
             "Solve as ODE": dae_reduce_to_ode (A) / zb_g (B) / dae_stabilized_index1 (C)
       circuit: AppModel::import_circuit → spice_import (spice_parse.cpp) writes DAE TEXT into the record
  └─ scheme NAME → body: analysis_session.cpp
       krs_for_scheme_impl / compute_krs_for_scheme      fixed step, every tab
           built-in  → scheme_from_name → codegen_scheme(System, Scheme)
           DAE       → dae_scheme_from_name → codegen_dae_scheme
           custom    → CustomScheme body from the record
           Extr(...)/Comp(...) → parse_extrapolation_name / parse_composition_name + wrapper
       adaptive_code_for_scheme                          adaptive step → AdaptiveCode {rhs, emb, dprep, deval}
           codegen_adaptive / _gbs / _extrapolation[_complex] / codegen_adaptive_dae
       sweep_krs + fill_adaptive_request                 sweeps; BDF: codegen_bdf_bodies + prefix macros
       compute_krs_cpu_for_scheme                        CPU (cl.exe DLL) variant
```

Inside `codegen.cpp` (anonymous namespace): `Parser`/`Node` AST, `pn_*` printers, `pn_diff` (Jacobians for
implicit / DAE / Rosenbrock), `sys_core`, `rhs_over`, `scheme_implicit_common` (Newton + LU), `dae_*` emitters,
`dae_deriv_array` (index), `sparse_lu_*` (with `sparse_lu.h`). `SystemEvaluator` (end of codegen.hpp) is the CPU
bytecode interpreter of the same AST.

## 5. Engines in detail

**ParametricEngine** (`parametric_engine.cpp`, pimpl `Impl`):
- `ensure_init` reads all templates once (`src_template_*`, `src_*_adaptive_part`).
- `build_module(snap, ...)` — one place that turns (template, body, defines) into a CUmodule:
  monolith or rdc (`lib_ptx_for` + `krs_ptx_for` + cuLink). Prefix macros added here or by callers:
  `time_prefix` (`UCUDA_NONAUTONOMOUS`), `lyap_nx_prefix` (`UCUDA_LYAP_NX`), `AdaptiveRequest::prefix`
  (`UCUDA_AD_DAE_NX`, `UCUDA_BDF_DRIVER`), `peak_config_defines`.
- `compile_*_if_needed` per task + `ModuleLru` (`module_lru.h`) caches; `prewarm_*` compile without activating.
- CPU twins live at the top of the file (`run_*_cpu`, `cpu_loop_model`, `cpu_gram_schmidt[_dae]`,
  `dae_point_start`) and use `KrsCpuStep` (krs_cpu) for the step.

**NvrtcEngine** (`nvrtc_engine.cpp`): Analysis only. Kernel sources are string literals in this file
(`phase_kernel`, `phase_kernel_ad`, `endpoint_kernel_ad`, `bdf_phase_kernel`, `ucuda_ctrl_check`).
`phase_portrait_nvrtc.*` = thin wrappers with a global engine instance.

**CPU**: `integrator.cpp` (built-in schemes over `SystemEvaluator`, Butcher-table adaptive driver),
`krs_cpu.cpp` (`KrsCpuStep` — step body built by cl.exe into a DLL, double/dd/qd; `AdaptiveCpuModule` —
`adaptive_part.cu` built as a DLL with entries `ucuda_cpu_ad_phase/endpoint/bif`, `ucuda_lyap_chain`),
`multistep_bdf.h` (BDF over SystemEvaluator).

**Shared GPU/CPU headers in `kernels/`**: `ucuda_adaptive.cuh` (controllers + adaptive driver),
`ucuda_bdf.cuh` (DASSL BDF core), `ucuda_hp.h`. They are compiled both by NVRTC and by MSVC — keep them
plain C++ with `__host__ __device__` macros.

**`cudaLibrary.cu`** (33 `__global__`): the classic sweep kernels and loops (`loopCalculateDiscreteModel_int`,
`LLEKernelCUDA`, `LSKernelCUDA`, DBSCAN, peaks). Templates `#include` it after defining
`calculateDiscreteModel`. `cudaLibrary.cuh` holds the time macros (`UCUDA_NA`, `UCUDA_STEP`, ...).
`hostLibrary.cu` and `main_NonLinAnal.cu` are the Debug legacy path, not used by the UI.

## 6. Persistence

| What | Type | Read/write |
|---|---|---|
| System (equations, params, IC, scheme flags, custom KRS, DAE/circuit fields) | `SystemRecord` | `record_to_json` / `record_from_json` (`system_library.cpp`), folder `library/<Name>/system.json` |
| Tab sessions | `*Session` | `session_to_json_*` / `session_from_json_*` (`session_io.cpp`), `library/<Name>/sessions/*.json` |
| Plot windows | `ParametricPlotWindow`, `OrderPlotWindow` | `session_*_windows` (`session_io.cpp`) |
| Step controllers | `AdaptiveCtrlInfo` | `load_ctrl_library` / `save_ctrl_library`, `library/step_controllers.json` |
| App settings | `AppConfig` | `load_app_config` / `save_app_config`, `x64/Release/_app_config.json` |

A new field must be added in three places: the struct, `*_to_json`, `*_from_json` (with a default for old
files). `tests/u-cuda/sessions_test.cpp` checks that every library file round-trips.

## 7. Where to look

| Question | Place |
|---|---|
| a[] layout, Newton tail | `scheme_fill_tail` / `scheme_append_tail` (codegen.hpp), `sweep_param_slot` (analysis_session.cpp) |
| Scheme name → enum, lists in combos | `scheme_from_name`, `dae_scheme_names`, `enabled_builtins_from_record`, `draw_scheme_combo` (`gui_widgets.cpp`) |
| Which schemes support t / adaptive / multistep | `scheme_supports_time`, `scheme_supports_adaptive`, `adaptive_scheme_name_supported`, `scheme_is_multistep` |
| Integration block settings → driver params | `adaptive_settings.h` (`adaptive_build_params`), `fill_adaptive_request` |
| Why a sweep refused a DAE axis | `dae_z_axis`, `kDaeZAxisMsg` (parametric_engine.cpp) |
| Peak detection settings | `PeakConfig`, `set_peak_config`, `peak_config_defines` |
| Module cache key / rebuild | `hash_key`, `compile_*_if_needed`, `build_module` |
| Consistent z0, DAE index | `dae_consistent_z`, `dae_consistent_hidden`, `dae_structural_index` (codegen.cpp) |
| Circuit import, devices | `spice_import` (spice_parse.cpp), `dae_devices.cpp`, `kicad_import.cpp`; DC OP `dae_dc_operating_point` |
| CSV / PNG export | `data_export.cpp`, export menus in `gui_widgets.cpp` |
| Plots | `plot_renderer.*`, `plot_view_2d/3d.*`, `heatmap_view.*`, GPU lines `gpu_line_series*.*` |
| Time estimate before a run | `time_estimate.h`, `krs_step_cost`, `gpu_calibrate` |
| OCR | `ocr_client_win.*` ↔ `ocr_server.py`; `AppModel::start_ocr` |
| Undo/redo, broadcast of settings between tabs | `push_undo` / `undo_last` / `slots_for` (app_model.cpp) |
| GUI helper used by several tabs | `gui_internal.h` (prototypes with default args) |

## 8. Recipes

- **New fixed-step scheme:** add to `Scheme` + `scheme_from_name` + emitter in `codegen_scheme`; flags
  (`scheme_supports_time`, adaptive, `scheme_rhs_per_step`); CPU: `int_scheme_from_string` (integrator) or
  rely on krs_cpu; record flag `scheme_*` + `enabled_builtins_from_record`; checkbox in `gui_system.cpp`.
  Verify autonomous library bodies stay byte-identical (codegen dump).
- **New DAE method:** `DaeScheme` + `dae_scheme_names` + emitter called from `codegen_dae_scheme`; refusals by
  index; checkbox list is generated from `dae_scheme_names`.
- **New setting of a sweep:** field in the `*Config` (analysis_session.h) → `build_*_request` → `*Request`
  (parametric_engine.h) → kernel argument / define; GUI in `gui_parametric.cpp`; JSON in `session_io.cpp`.
- **New kernel/template:** file in `kernels/` (post-build copies the folder), `read_text_file` in
  `ensure_init`, a `compile_*_if_needed` with its own cache, CPU twin, test in `tests/u-cuda`.
- **New .cpp:** needs `.vcxproj` + `.vcxproj.filters` (ask first) and, if tests link it, `run_tests.bat`.

## 9. Tests

`tests/u-cuda/` (`run_tests.bat [suite]`, all suites ~20 min): `gpu_smoke`, `krs_cpu_test`, `adaptive_test`,
`architecture_test`, `sessions_test`, `tabs_test`, `dae_test`, `spice_test`, `kicad_test`, `library_test`;
helpers in `test_util.h`. They link the Release `.obj` files, so build Release first. Headless sweeps for
ODEs: skill `.claude/skills/ucuda-headless` (`ucuda_run.exe`).
