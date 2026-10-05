# NVRTC kernel templates

These **.template.cu** files are runtime compilation units for bifurcation, LLE/LS, DFT, basins, FastSync, Order, signal metrics, networks and circuit simulation. Engines substitute generated scheme code and runtime settings, then compile with NVRTC.

- **signal_metrics.template.cu** — Parametric → Metrics (max/min/mean, peak frequencies, Hjorth parameters), streamed without storing the trajectory. Output row numbers (`SIGM_*`) must match `SignalMetric` in `ucuda/domain/analysis_types.h`.
- **network.template.cu** — coupled oscillator networks: one block per network, one thread per node, state in shared memory. Two placeholders: `{{KRS_BODY}}` (node step) and `{{COUPLING_BODY}}` (switch cases per coupling law, printed by `codegen_coupling`).
- **circuit.template.cu** — GPU transient of a synthesized analog circuit (see `ucuda/cuda/circuit_nvrtc.h`).
- **ucuda_adaptive.cuh** — upstream adaptive driver and parameter/controller layouts, shared with native CPU compilation. `adaptive_part.cu`, `lyapunov_adaptive_part.cu` and `metrics_adaptive_part.cu` provide sampling, Lyapunov and metric algorithms. Domain uses the layout-only section; equations supplies generated embedded/dense bodies.
- **ucuda_hp.h** — double-double / quad-double arithmetic shared by NVRTC kernels and the CPU KRS build (extended-precision Order). The numerics target adds this directory to its include path.

Shared device algorithms come from CUDA sources packaged into the runtime kernels directory by PackRuntime.cmake; do not duplicate helpers into individual templates. Template file names are runtime contracts referenced by engine code.

After changes, clear the copied kernel directory or use a fresh build, run **gpu_smoke**, run the relevant GPU analysis test, and verify cache invalidation inputs.
