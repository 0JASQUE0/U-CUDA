# Direct comparison with upstream main

This is a Windows reference build of the unmodified monolithic main sources.
It compiles the original CPU/CUDA/NVRTC, parser/codegen, session, storage and
plotting sources directly from a detached worktree. Plotting is linked because
upstream sessions own plotting objects; the comparison does not open a GUI.
Forwarding headers map module include names to original upstream headers.
The only platform adapter supplies the test executable directory. No upstream
algorithm source is replaced by the modular implementation.

Both builds use the same versioned test sources, fixtures, C++17, MSVC, CUDA,
pinned GoogleTest/Benchmark dependencies and UTF-8 test process manifest.
Native compilation caches and fixture libraries are isolated for each build.
No test reads or writes the user library.

## Reproduce

Use PowerShell 7, the Windows CUDA toolchain and an NVIDIA GPU. From the repo root:

```powershell
git worktree add --detach build/main-parity-source 81989f49193f990d0311b8dab73f1282ff52e486
cmake -S tests/parity -B build/main-parity -G "Visual Studio 17 2022" -A x64 `
  -DUCUDA_MAIN_SOURCE="$PWD/build/main-parity-source"
cmake --build build/main-parity --config Release --parallel 4
cmake --build --preset windows-release
./tests/parity/run.ps1
```

An existing build/windows dependency source cache is reused when available;
otherwise CMake fetches the same pinned dependencies. The source worktree must
remain unchanged. `run.ps1` records its SHA and working tree, both build caches,
GPU/CPU/OS information, executable/runtime/test-source hashes, logs, JUnit and
the exact comparison report in a new build/upstream-parity/<id>/ directory.
An exit code 77 is a failure here: missing hardware must not count as parity.

The run includes the shared numerical matrix, DAE/BDF tests, native compiler,
adaptive methods, architecture, sessions, tabs, 24 Network fixtures, library,
SPICE and NVRTC pruning. Observations additionally compare generated source
for all 36 builtins on three models, 11 session kinds, three window formats,
application settings, CSV text, 36 interpreted CPU methods, 16 interpreted
adaptive grid/node paths and 24 RQA configurations. RQA has independent
CPU distance and recurrence-rate oracles, symmetry and refusal checks.

`ucuda_parity_compare` compares every JSON output recursively. It checks file
sets, types, sizes, integer/string values and finite binary64 bits, including
signed zero. Non-finite library/metric values use explicit NaN/Inf strings.
Empty or missing results cannot pass. Optional `--tolerances <abs> <rel>` creates
a separate numerical-equivalence assessment; the default run uses exact bits.
The ordinary runner's accepted references are never updated by this workflow.

The interpreted DOPRI78 variants use cubic Hermite dense interpolation.
Its independent error oracle uses `lambda^4*hmax^4/384 + 2*rtol` for exponential
decay, and verifies fourth-order dense convergence when hmax is halved.
Endpoint tolerances alone do not bound the interpolation error between nodes.
This accounts for behavior present in main; numerical code and ordinary
regression tolerances remain unchanged.

## Documented compatibility differences

These probes are stored separately in each stage's compatibility.json, outside
the common computation output set. They are deliberate behavior differences,
so literal identity for every possible session input is not claimed:

- Main regenerates the phase KRS cache while loading JSON. The modular storage
  layer clears it; the analysis layer regenerates it before calculation.
  Explicit regeneration produces the same source in both builds.
- An obsolete local `adaptive.method` block belongs to the retired modular
  solver. The modular loader ignores it and keeps upstream adaptation disabled.
  Main ignores the unknown method key and can enable its solver from enabled.
  This preserves the compatibility rule described in [INTEGRATION.md](../../INTEGRATION.md).

## Representative performance comparison

After an exact run succeeds, use its printed directory:

```powershell
./tests/parity/measure.ps1 -ParityRun 'build/upstream-parity/<id>'
```

This compares 16 workloads: CPU/GPU Euler, RK4, DOPRI78, Complex CD, fixed BDF5,
DAE Radau IIA, cold compilation, and GPU Network full-call/kernel timing. It interleaves original
and modular series, uses the frozen test_config.json protocol, retains every
raw sample and applies the existing native comparator's unchanged 10% slowdown
and variation limits. It is representative timing coverage, not a timing claim
for every analysis/input. Correctness is checked outside benchmark timing.

GUI interactions and external KiCad/ngspice validation are outside this run.
The original main source requires Windows; this reference project does not
claim Linux parity. See [the validation report](../VALIDATION.md) for actual runs.
