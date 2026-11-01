# CPU/GPU checks and performance history

Tests use C++17, GoogleTest, Google Benchmark and CTest. CMake pins dependency
commits in [TestDependencies.cmake](TestDependencies.cmake). The runner needs no
Python and builds with `U_CUDA_BUILD_GUI=OFF`. CUDA Toolkit remains required.
Optional OCR Python services are unrelated to these checks.

```powershell
cmake --preset windows -DU_CUDA_BUILD_LEGACY=ON
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
(--config selects another file). Defaults: 10% slowdown, 300 ms warmup, three
series of nine repetitions. Per-ID cases overrides allow absolute, relative,
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
