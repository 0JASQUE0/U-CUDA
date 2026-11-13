# Validation on RTX 4070 SUPER

Local validation used Ryzen 7 7700, RTX 4070 SUPER, NVIDIA 621.01,
CUDA/NVRTC 13.0, MSVC 19.44 and CMake 4.2.0-rc1 on Windows.
The merged implementation is commit a9adf0d9a58ed421ffe9227c39eae95c504a0b7d;
subsequent commits refine the test runner and references.

A later direct comparison builds the unmodified main source independently.
The [paired comparison report](parity/RESULTS.md) records exact CPU/GPU results,
additional interpreted CPU coverage and the deliberate compatibility differences.

## Correctness and builds

- Original modular branch: 4/4 original CTest checks passed before integration.
- Merged Release full run 1793543155109894600: 286 passed, no failures,
  one optional KiCad/ngspice check skipped because KiCad is absent.
  CUDA tests ran on the real GPU, including NVRTC and extended-precision paths.
  CTest elapsed time was 861.19 seconds.
- That run contains 1,502 numerical snapshots and 168 benchmark scenarios.
  The matrix includes every built-in method, supported analysis/backend pairs,
  DAE, adaptive and fixed BDF, library fixtures, import and session compatibility.
- Debug GUI/legacy/test targets built; 21 selected CPU/GPU checks passed.
  After runner/reference refinements, another 13 selected Debug checks passed.
- A separate GUI-off configuration built regression, DAE, benchmark,
  architecture and legacy targets; all six selected architecture/policy checks passed.
  The final runner/regression/benchmark rebuild and the same six checks also passed.
- Final Release follow-up: all 20 selected policy, CD convergence/matrix and
  Network checks passed. CD fast/FP fixtures now include independent second-order
  step-halving checks on both CPU and GPU (scenario version 2).
- The actual `cmake --build --preset windows-release --target check-smoke`
  command passed against the accepted references.
- The `check-all` target passed with explicit selection of the eight smoke/policy
  checks and CPU RK4 benchmark, using the final one-second timing defaults.
  This verifies the combined build/run/compare path; it is not a second full
  matrix run. The CMake selection filters were then restored to empty.
- GUI startup with an isolated library and normal window closure exited with
  code zero. Interactive GUI workflows were not exhaustively tested.
- The available WSL distribution failed to start with Wsl/Service/E_UNEXPECTED;
  Linux compilation and execution were not performed.

## Migration and failure detection

The [migration comparison](baselines/migration-68d05ba/comparison.csv) contains
eight original-versus-merged CPU/GPU comparisons. All pass the unchanged
10% slowdown policy; maximum endpoint difference is 1.11e-16.
[INTEGRATION.md](../INTEGRATION.md) lists adopted upstream behavior changes.

Policy tests exercise numerical corruption, artificial sustained slowdown,
unstable measurements, incompatible inputs/environments, immutable history,
selective acceptance and scenario versioning. Native runner negative checks
also reject hidden GPU hardware, missing references and empty selections.
CPU-only smoke succeeds with the GPU hidden.
The final accepted reference also passes CPU-only smoke with the GPU hidden;
requesting GPU coverage in that environment fails before execution.
Additional comparator fixtures derived from accepted RK4 results pass unchanged,
fail after numerical corruption or a 25% timing increase, and leave the accepted
manifest's SHA-256 unchanged. These are deliberately modified comparator fixtures,
not additional solver measurements.

## Performance reference policy

The initial full run passed independent correctness checks but returned
inconclusive because 38 of 168 timing scenarios exceeded 10% relative spread
between series medians. This is deliberately a non-success result.
Its 130 stable timing scenarios can be accepted separately from noisy ones.
The raw initial run and all repetitions remain in build/windows/test-results/.

Run 1793545746103918500 used one-second warmup and 0.3-second minimum measurement
and supplied another 27 stable timing references. Run 1793546738462598300 used
one-second warmup and minimum measurement and supplied the remaining 11;
their relative spreads were 0.08%-2.36%. Thus all 168 timing references are
individually stable, with all 27 raw samples retained per scenario. They are
assembled from reviewed runs rather than described as one passing initial run.
The one-second durations are now the defaults. Neither 10% threshold was relaxed.

The accepted catalog contains 1,502 numerical and 168 timing scenarios. The
six CD fast/FP numerical scenarios and kernel-only Network scenario use version
2; historical version 1 references and all original raw captures remain available.

Kernel-only Network workload version 2 uses 256 nodes and 10,000 steps in one
launch, with analytic correctness checks outside the timed region. The original
short workload was too sensitive to timing variation. This input change has
an explicit scenario version increment; other workload versions are unchanged.

Accepted manifests and case artifacts are stored in [baselines](baselines/README.md).
The ordinary check targets never replace them. Performance comparisons require
a compatible hardware/software environment and stable measurements.

## Small CUBA regression, 9 November 2026

The `neuromorphic` working branch was fast-forwarded to GitHub `main`
`4163962`. Local `.codex/config.toml` changes were preserved. The CUBA change
adds tests and documentation only; no production integrator or kernel changed.

Validation used Windows, RTX 2060 SUPER, driver 591.86, CUDA/NVRTC 12.8.93,
MSVC 19.43.34810 and the Release headless CMake configuration. The
`neural_cuba_test`, `adaptive_test`, `architecture_test` and `gpu_smoke`
targets built successfully. All eight selected CTest checks passed, without
skips: `neural_cuba_oracle`, `neural_cuba_gpu`, `adaptive_cpu`, `adaptive_gpu`,
`architecture`, `layer_rules`, `vcxproj_sync`, and `gpu_smoke` (125.36 seconds
total). The full regression suite, performance suite and GUI build were not run.

The three fixed CUBA fixtures produced 4, 2 and 37 spikes respectively.
Every `(step, neuron)` event and per-neuron count agreed with the independent
closed-form CPU reference; every recorded state and final state met the
documented truncation bounds. Block/chunk changes preserved trajectories
bit-for-bit. Disabling observations also tested actual multi-step spike
launches and preserved final states and event logs. A zero-delay edge was
rejected. This validates a small fixed configuration, not a full 4000-node
CUBA benchmark or equivalence with another simulator.

Two deliberate faults were tested in the **packaged build copy** of
`kernels/neural.template.cu`, in separate fresh processes:

- Adding `+ NEURAL_CH` to `int idx = base + __ldg(A.outPack + e);` shifted
  delivery by one step. The pulse fixture failed on the first affected trace
  (scalar 42: -60 instead of approximately -59.877321).
- Replacing `if (refr > 0) --refr;` with `if (refr > 1) --refr;` released
  the membrane early. The pinned boundary fixture failed on its spike events.

Both mutations returned exit code 1. The packaged template was restored
byte-for-byte and checked against the repository source; normal checks then
passed. These mutations never changed a tracked kernel or the user library.
Local logs are under `build/windows/` (`cuba-build.log`,
`cuba-final-build.log`, `cuba-mutation-check.log`, and `cuba-validation.xml`).
`git diff --check` passed. Accepted baseline artifacts were not changed.

## Saved classifier and LIF-LSM validation, 9 November 2026

Release headless targets and the desktop `U-CUDA.sln` (Release|x64) built on
the RTX 2060 SUPER setup above. Eight selected CTest checks passed without
skips: `adaptive_cpu`, `layer_rules`, `vcxproj_sync`, `architecture`, both CUBA
checks and both classifier checks. The CTest report is
`build/windows/classifier-validation.xml`. The full regression suite and the
separate `adaptive_gpu` and `gpu_smoke` suites were not run for this change.

Both state and LIF classifiers matched the original held-out predictions and
confusion matrix after session load and readout export/import. A deliberately
changed saved model gave different predictions without training. Old Neural
sessions and regression files still load. The scaler check uses a ramp with a
known training mean for both a class source and an auxiliary source.

The LIF fixture uses four neurons, 12 separate trials (6 train, 3 validation,
3 test), two feature windows per stimulus and a 0.125 step. It produced 30
spikes and test accuracy 1.000 with confusion matrix `[[2, 0], [0, 1]]`.
Trials 2, 6 and 12 matched isolated runs event for event after aligning the
step numbers. The stimuli are deliberately simple to check execution and
reset; this result does not establish an advantage over an input baseline.

Plan 6.6.2 passed the automated checks and the user confirmed the manual LIF
run with the isolated `build/windows/lif-lsm-ui-library`. See [the procedure
and expected values](README.md#saved-classifier-and-lif-trial-protocol).
No validation fixture was added to the shipped library.

## Neural Results and spike statistics, 9 November 2026

Release headless targets and the desktop solution built successfully on the
RTX 2060 SUPER setup above. All 12 selected CTest checks passed without
skips, including `adaptive_gpu`, `gpu_smoke`, and the CPU and GPU Neural
Results, classifier and CUBA checks. The report is
`build/windows/neural-results-validation.xml`; the full suite was not run.

The statistics checks cover known ISIs and CVs, histogram totals, silence,
insufficient spikes, trial resets and truncated logs. The rate check covers
an event on a bin boundary and normalisation of the short final bin. A live
GPU run with two regular sources also verified the published statistics and
continuation: continuing with its earlier log gives 22 intervals, while a
checkpoint without the log is marked partial and gives no ISI/CV. Failed
runs and edits in flight preserve the old result; queued and reindexed
experiments retain their result identity. The standalone benchmark took a
median of about 30 ms for 2 million events from 100,000 neurons.

Desktop visual inspection of the new layout and the linked time axis is
pending. An isolated Release copy and LIF library can be opened via
`build/windows/Run-Neural-Results.cmd`; see the [desktop smoke procedure](README.md#neural-results-and-spike-statistics).

The Activity follow-up adds trial bands in the native `Plot2DView`, using
the timing and class order captured in the completed run. Release built,
and all eight selected checks passed (`activity-trials-validation.xml`).
The LIF fixture produced 30 spikes in High current, zero in Low current,
and zero during rest. Classifier replay retained the same annotations;
continuous spiking published no trial bands. Visual inspection remains manual.

## Autonomous Lorenz forecast from fixed starts, 12 November 2026

The Release headless run of `neural_forecast_gpu` passed on an RTX 4090
with CUDA 12.6. The fixed fitting prefix and five held-out starts are
described in the [forecast protocol](README.md#autonomous-lorenz-forecast-from-fixed-starts).
All three models crossed the threshold within the forecast window at
every start. Horizons are in Lyapunov times.

| Start sample | Source time after warmup | Tanh reservoir | Autoregression | NVAR |
|---:|---:|---:|---:|---:|
| 2200 | 44 | 0.21744 | 0.18120 | 8.53452 |
| 2900 | 58 | 2.40996 | 0.36240 | 9.27744 |
| 3600 | 72 | 0.27180 | 0.12684 | 9.45864 |
| 4300 | 86 | 0.74292 | 0.16308 | 7.55604 |
| 5000 | 100 | 1.55832 | 0.28992 | 8.48016 |
| Mean | | 1.04009 | 0.22469 | 8.66136 |
| Range | | 0.21744–2.40996 | 0.12684–0.36240 | 7.55604–9.45864 |

The spread is across initialisation times on one trajectory and one graph
seed. The comparison measures short-term forecast skill for this setup.

`adaptive_cpu`, `adaptive_gpu`, `architecture`, `layer_rules`,
`vcxproj_sync` and `gpu_smoke` also passed without skips. The GUI build
and full regression suite were not run.

## Closed-loop attractor and Lyapunov exponent, 12 November 2026

The [closed-loop protocol](README.md#closed-loop-lyapunov-exponent-and-long-run-statistics)
uses the same 96-node reservoir and fixed readout at both steps. The first
500 autonomous samples were discarded. The 2,000-sample window is a prefix
of the 4,000-sample run. Reported channel statistics are
triples in x, y, z order and use the normalisation fitted before the test.
The reservoir exponent is expressed per source time unit. The Lorenz
exponent in the table is the 0.906 reference used in the protocol.

| System | Integration step | Autonomous samples | Mean | Standard deviation | Lag-one correlation | Largest exponent |
|---|---:|---:|---|---|---|---:|
| Lorenz | 0.01 | 2000 | 0.037, 0.027, 0.044 | 0.994, 0.979, 0.952 | 0.993, 0.989, 0.982 | 0.906 |
| Lorenz | 0.01 | 4000 | 0.048, 0.036, 0.023 | 0.991, 0.984, 0.972 | 0.993, 0.989, 0.983 | 0.906 |
| Reservoir | 0.1 | 2000 | 0.742, 0.648, 0.057 | 0.774, 0.739, 0.720 | 0.991, 0.986, 0.987 | 0.1773 |
| Reservoir | 0.1 | 4000 | 0.949, 0.823, 0.085 | 0.538, 0.507, 0.472 | 0.991, 0.987, 0.987 | -0.0957 |
| Reservoir | 0.05 | 2000 | 0.095, 0.084, -0.051 | 0.969, 0.972, 1.034 | 0.992, 0.987, 0.983 | 1.0138 |
| Reservoir | 0.05 | 4000 | 0.395, 0.342, 0.018 | 0.929, 0.906, 0.861 | 0.994, 0.989, 0.983 | 0.6704 |

The reservoir stays bounded across the measured windows. The exponent
changes materially with integration step and window length. The positive
x and y means at 4,000 samples show a bias relative to the source, even
with the smaller step. These runs leave convergence and reproduction of
the Lorenz attractor unconfirmed for this reservoir.

The analytical GPU check passed on RTX 4090 at both integration steps
for Stage, Heun, Post and Old. In each case, the measured exponent agreed
with the exact discrete map within `1e-8` per network time unit. The primary
trajectory and feedback were identical with and without the clone.

Release builds passed with CMake headless and Visual Studio Release|x64.
The forecast and closed-loop Lyapunov tests, `adaptive_cpu`, `adaptive_gpu`,
`gpu_smoke`, `architecture`, `layer_rules` and `vcxproj_sync` passed without
skips. The full regression suite and manual GUI checks were not run.

## Initial-state forgetting and conditional Lyapunov exponent, 12 November 2026

The [conditional protocol](README.md#initial-state-forgetting-and-conditional-lyapunov-exponent)
passed on RTX 4090 in Release. Each of the 24 renormalisation blocks was
checked against the independent CPU pair using the same time grid.
The refined CPU reference gave −0.7862247533 per network time unit.

| GPU integration step | Conditional exponent | Absolute error against refined reference |
|---:|---:|---:|
| 0.125 | −0.7862245207 | 2.33e-7 |
| 0.0625 | −0.7862245577 | 1.96e-7 |

Under the same prescribed Lorenz input, the maximum pairwise RMS distance
of the three Tanh reservoir replicas fell from 2 at time 0 to
0.06801 at time 8, 0.002144 at time 16, and 1.34e-8 at time 32.
It stayed below 2e-5 from time 23 through 64. The driven reservoir's
conditional exponent over time 16–64 was −0.6586620415 per network time
unit. This describes the fixed input, graph and initial states used here.

The Release test target built and all eight selected checks passed without
skips. The selection included the three Neural forecast and Lyapunov
checks, `adaptive_cpu`, `gpu_smoke`, `architecture`, `layer_rules` and
`vcxproj_sync`. The full regression suite was not run.
