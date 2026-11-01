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
