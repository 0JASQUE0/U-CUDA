# CPU/GPU checks and performance history

Tests use C++17, GoogleTest, Google Benchmark and CTest. CMake pins dependency
commits in [TestDependencies.cmake](TestDependencies.cmake). The runner needs no
Python and builds with `U_CUDA_BUILD_GUI=OFF`. CUDA Toolkit remains required.
Optional OCR Python services are unrelated to these checks.

For a separate build of the original main sources and direct CPU/GPU output
comparison, see [upstream parity](parity/README.md) and its [results](parity/RESULTS.md).

```powershell
cmake --preset windows
cmake --build --preset windows-release --target check-smoke
cmake --build --preset windows-release --target check-regression
cmake --build --preset windows-release --target check-performance
cmake --build --preset windows-release --target check-all
```

Each target builds required executables. Smoke covers CPU/GPU RK4, artifact
policy, NVRTC compilation/loading. Regression adds builtins, solver and analysis
families, parsing/codegen, imports, caches, old JSON, CSV, queues/cancellation.
Performance runs smoke plus benchmarks; all runs regression plus benchmarks.
Original CTest names adaptive_cpu, adaptive_gpu, architecture, gpu_smoke and
Linux platform remain available. GoogleTest names start with numerical.
`layer_rules` ([layer_rules.cmake](layer_rules.cmake)) checks module boundaries the
compiler cannot see: domain includes no higher module, `<future>`/`<thread>` or engine
names; storage includes no compute/application/GUI header; computation includes no GUI.
`vcxproj_sync` ([vcxproj_sync.cmake](vcxproj_sync.cmake)) checks that the classic Visual Studio project
`U-CUDA/U-CUDA.vcxproj` compiles exactly the library sources listed in `libs/*/CMakeLists.txt`.

For frequent runs `UCUDA_TEST_SKIP_GGL=1` drops the "+ GGL" DAE methods from the DAE
suite (their bodies build slowest); add `-E GGL` to skip their solver-matrix cases too.

```powershell
ctest --preset windows-release -R 'adaptive_cpu|architecture'
cmake --preset windows -DU_CUDA_TEST_BACKEND=cpu
cmake --build --preset windows-release --target check-regression
```

U_CUDA_TEST_BACKEND accepts all/cpu/gpu. U_CUDA_TEST_CASES selects CTest names
by regex; U_CUDA_TEST_BENCHMARKS selects benchmarks. U_CUDA_TEST_BASELINE selects
an accepted JSON file. Clear filters and restore all when finished. Linux uses
the corresponding linux-release preset. Missing GPU, required results or
baseline fails the overall check. Missing KiCad/ngspice is an explicitly
reported optional external skip, separate from computation validation.

## Small CUBA regression

`neural_cuba_oracle` and `neural_cuba_gpu` are included in the regression/all
modes. They use fixed initial states and directed edges, without a random graph
or the user library. Build and run on Windows:

```powershell
cmake --build --preset windows-release --target neural_cuba_test
ctest --preset windows-release -R '^neural_cuba_(oracle|gpu)$'
```

| Path | Capability | Independent check |
|---|---|---|
| CPU oracle | Closed-form LIF with exponential current synapses; not a CPU Neural backend | Pinned threshold/reset/refractory events, signed and simultaneous arrivals, delay and refractory sensitivity |
| GPU Neural | CUBA RK4 membrane with two exact exponential synapses | Every `(step, neuron)` event, per-neuron counts, all recorded states and final states against the analytic oracle |
| GPU execution variants | Different block sizes, host chunks and spike windows | Identical spike logs and bitwise equal trajectories; delay zero is rejected |

Spike indices start at zero: index `n` means the threshold was checked at
`(n + 1) * h`. An edge of delay `d >= 1` from that spike is applied at the
beginning of step `n + d`. The oracle schedules deliveries in an absolute-time
table and integrates LIF in closed form; it reuses neither the RK stages nor
the device ring or fixed-point accumulation. Binary-exact weights avoid
quantization ambiguity. The recurrent fixture includes both signed synapses,
nonzero initial currents, and equal membrane/synaptic time constants.

The fixed step is 0.125 ms; the longest fixture is 64 ms. Membrane comparisons
allow 2e-6 mV for RK4 truncation; synaptic comparisons allow 2e-11. Every spike
must match exactly and the log must be complete. The pinned boundary fixture
tests equality at threshold, reset, two full refractory steps, and a node that
starts at threshold but falls below it before the first threshold check.
This is a small regression, not the 4-second, 4000-node benchmark or a
cross-simulator reproduction. A skipped GPU test does not validate the engine.

## Autonomous Lorenz forecast from fixed starts

`neural_forecast_gpu` runs the stage-scheduled Tanh reservoir on CUDA and the
autoregression and NVAR baselines on CPU. The synthetic source is Lorenz
(`sigma = 10`, `rho = 28`, `beta = 8/3`), sampled every 0.02 source time
units. The 96-node reservoir uses graph seed 1. Input normalisation, readout
fitting and baseline selection use the split of samples `[0, 2000)`. The five
starts at samples 2200, 2900, 3600, 4300 and 5000 use only earlier samples to
initialise the reservoir and lag features. Each start uses the same graph,
split and normalisation.

```powershell
cmake --build --preset windows-release --target neural_forecast_test
ctest --preset windows-release -V -R '^neural_forecast_gpu$'
```

The test checks the common input history and graph, alignment of the forecast
target, unchanged fit across starts, and independence from changes to the
future target. It reports the horizon to the first normalised RMS error
above 0.5 for each model and start, in Lyapunov times (`lambda = 0.906`), with
the mean and range across starts. A forecast still within threshold at the
end of the 600-step window is marked censored and contributes a lower bound.
The horizons describe this fixed configuration and are not pass thresholds.

### Closed-loop Lyapunov exponent and long-run statistics

`neural_closed_loop_lyapunov_gpu` checks an independent exact sample map
for a one-node system with held feedback `u = a*x`. Over a sample of
duration `dt`, the map multiplies `x` by `1 + a*dt`, so the exponent is
`log(1 + a*dt) / dt`. The check covers all four continuous schedules,
two integration steps at fixed `dt`, a delayed feedback start, an
off-grid interval, and exact agreement of the primary trajectory with
the run without a clone.

```powershell
cmake --build --preset windows-release --target neural_forecast_test
ctest --preset windows-release --output-on-failure -R '^neural_closed_loop_lyapunov_gpu$'
& .\build\windows\bin\Release\neural_forecast_test.exe --attractor
```

The long run uses the first held-out start and the fixed readout fitted
at network step 0.1. Both runs use the same graph, observations, input
normalisation and readout. Network steps 0.1 and 0.05 give the same
input sample interval, 0.02 in source time. The first 500 autonomous
samples are excluded. The report compares mean, population standard
deviation and lag-one autocorrelation of the normalised x, y and z
over 2,000 and 4,000 autonomous samples. It also reports the exponent
in inverse source time, using a renormalisation every five samples.
The source comparison uses the Lorenz trajectory over the same windows
and the reference exponent 0.906 from the forecast protocol. Statistics
and the exponent describe this configuration. The exact sample map
provides the regression thresholds.

## Saved classifier and LIF trial protocol

`neural_classifier_cpu` checks old Neural JSON, the v1 regression file format
and training-only input normalisation. `neural_classifier_gpu` trains and
replays both state and LIF classifiers through `NeuralSession`, including a
session round trip and export/import. It checks predictions and confusion
matrices, that changing the saved weights changes the replay, and rejection
of mismatched graphs, class order, features, inputs and trial settings. The
LIF case compares individual trials against the combined run by spike step
and neuron, to check reset at the boundaries.

```powershell
cmake --build --preset windows-release --target neural_classifier_test
ctest --preset windows-release -R '^neural_classifier_(cpu|gpu)$'
```

To check the same LIF fixture in the desktop app, create an isolated library
in a directory that does not yet exist. From the repo root in PowerShell,
after building the desktop solution in Release|x64:

```powershell
$lifLibrary = Join-Path $PWD 'build\windows\lif-lsm-ui-library'
& .\build\windows\bin\Release\neural_classifier_test.exe --write-lif-library $lifLibrary
if ($LASTEXITCODE -ne 0) { throw 'Could not create the LIF test library' }
$previousLibrary = $env:U_CUDA_LIBRARY
try {
    $env:U_CUDA_LIBRARY = $lifLibrary
    & .\x64\Release\U-CUDA.exe
} finally { $env:U_CUDA_LIBRARY = $previousLibrary }
```

Load **LIF LSM validation** from Library. In Neural, run the saved experiment,
then compare Neural Readout to the headless fixture: 3 test trials, accuracy
1.000, confusion matrix `[[2, 0], [0, 1]]` (Low current, High current), and
30 spikes across all 12 trials. Press **Keep**, tick **use the saved classifier,
no training**, and run again. Check the same confusion matrix and the saved
classifier label. Reopen the library session and repeat; the saved output and
its input scaler should persist. This library is for validation only and is
separate from the user's library and from the shipped presets.

## Neural Results and spike statistics

`neural_results_cpu` checks result-view eligibility, setting changes, analytic
ISI/CV examples, reset boundaries, missing spikes, and rate normalisation for
a short final bin. `neural_results_gpu` checks that completed and queued runs
retain their own labels and settings, that failed runs preserve the previous
result, and that spike statistics handle complete and incomplete continuation.
The optional `--benchmark` measures the CPU aggregation on 2 million events
without changing any accepted benchmark references.

```powershell
cmake --build --preset windows-release --target neural_results_test
ctest --preset windows-release -R '^neural_results_(cpu|gpu)$'
& .\build\windows\bin\Release\neural_results_test.exe --benchmark
```

Desktop smoke check with the LIF fixture above:

1. Open Neural and run **LIF LSM validation**. Results appear in **Neural
   Results**; the old individual plot windows do not open.
2. In **Activity**, zoom or pan either plot and confirm the other follows the
   time range while Y stays fixed. Double-click inside either plot and check
   that both reset to the full time range. Check that each population uses
   the same color in both plots. The gray bands indicate rest and the colored
   bands show the stimulus class and trial number. Hover the strip to see
   details. All 30 spikes in this fixture fall in **High current** bands;
   there are none in **Low current** or the pauses. A continuous run has no
   trial bands.
3. Open **Spike statistics** and check the filled native histogram bars.
   Hover a bin to see its interval and count. ISI counts same-neuron intervals within one
   reset segment; CV is the population standard deviation of a neuron's
   intervals divided by its mean, with at least two intervals required. The
   CV histogram gives one observation per eligible neuron. A truncated log
   has no ISI/CV result; binned rates and raster are explicitly partial.
4. Check the **Classification** matrix, change the schedule or population
   name without running, and check that the previous-result notice appears
   while the original graphs, labels and matrix remain. Restore the original
   settings and check the notice disappears. Switch away and back on the same
   system; the result remains attached to that experiment.
5. With other configurations, check the continuous means and R(t), forecast
   above the error plot and its optional phase portrait, the memory/XOR
   delay curves, and the sweep curve (one axis) or heatmap (two axes).

## First baseline and subsequent comparisons

```powershell
$tool = '.\build\windows\bin\Release\ucuda_test_tool.exe'
& $tool run --mode all --backend all --record
# Use the result directory printed by the preceding command:
& $tool baseline accept --run '<result-directory>' --reason 'Reviewed initial merged reference'
& $tool run --mode all --backend all
& $tool compare --run '<result-directory>' --baseline '.\tests\baselines\active.json'
& $tool baseline accept --run '<result-directory>' --cases 'fixed.cpu.3,fixed.gpu.3' --reason 'Reviewed intentional change'
```

Record creates a candidate when no baseline exists; its status is recorded,
not a regression pass. It never accepts results. Existing baselines are still
compared. Use --baseline with a separate file for another hardware/software
environment. A full accepted run supplies coverage for its standard submodes
and backends. Filtered runs reuse their accepted full profile and require all
matching reference outputs. A selection without an accepted parent needs explicit
acceptance. For run, --cases is a CTest regex; for baseline accept it is a
comma-separated snapshot ID list. --benchmarks is a benchmark regex.

Acceptance requires completed independent checks and complete results. It prints
differences, records a reason, writes a new immutable history version and
atomically replaces the active file. Previous versions stay in
<baseline-name>-history/. Version manifests share immutable, content-addressed
JSON case files in <baseline-name>-data/, avoiding duplicated numerical arrays.
Each accepted case records its originating run and Git revision.
Commit the manifests and their case directory together. Selected acceptance preserves other scenarios. Runs
and comparisons never update accepted references. Changing scenario inputs or
meaning requires a new version and review. Analytic failures cannot be hidden
by accepting a baseline. Commit reviewed baseline files explicitly.

Unstable timing cannot be accepted. Rerun it and accept only stable cases;
previous accepted cases are retained. For long selections use
`--cases-file <file.json>` with a JSON array of scenario IDs, avoiding Windows
command-line length limits. Changed inputs require an incremented scenario
version (`snapshot` accepts an optional final version argument; benchmarks use
the `input_workload_version` counter).

Partial acceptance exposes a standard submode only after every required case
in that submode is accepted. A complete catalog can consolidate independently
accepted reruns, preserving explicitly accepted newer scenario versions.

## Artifacts and timing policy

Every run gets a unique build/<preset>/test-results/ directory (override with
--output). It contains run.json, numeric inputs/outputs, CTest plan, JUnit/logs,
raw benchmark JSON/logs per series, Git SHA/status/diff, runtime fingerprints
and hardware/compiler/driver/build metadata. Accepted references live separately
in tests/baselines/. No history is automatically deleted.

[test_config.json](test_config.json) controls tolerances, thresholds and timing
(--config selects another file). Defaults: 10% slowdown, one-second warmup and
minimum measurement time, three series of nine repetitions. The longer timing
interval avoids the observed noise of short Windows/GPU measurements; both
durations remain configurable. Per-ID cases overrides allow absolute, relative,
slowdown_limit. Release measurements run serially and compare series medians
only on compatible hardware/software. Excess variation yields inconclusive,
which is not success; sustained threshold excess is a regression. Correctness
checks run outside timing. Google Benchmark batches short workloads. Public GPU
calls synchronize; Network also records kernel-only CUDA event timing. Cold
compilation has separate cases.

Independent oracles include analytic decay/RC, convergence, DAE constraints,
short Lorenz references and invariants. CPU/GPU agreement complements these.
[support/builtin_cases.h](support/builtin_cases.h),
[support/solver_workloads.h](support/solver_workloads.h) and
[support/analysis_workloads.h](support/analysis_workloads.h) define the matrix.
Coverage tests detect unregistered builtins/analysis APIs; rejection tests check
unsupported paths. Imported checks are documented in [u-cuda](u-cuda/README.md).
The FullMatrix cases exercise each continuous builtin across supported analysis
backends; maps have separate discrete fixtures. First-order spectrum/network
oracles use truncation-error bounds appropriate to their fixed step. Existing
RK4 bounds stay unchanged. High-order Order calculations preserve error outputs
even when the observed order reaches the floating-point noise floor.
Versioned fixtures live under fixtures/. Library tests copy them into private
temporary directories and never write U-CUDA/library/.
