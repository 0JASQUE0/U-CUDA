# Direct main comparison, 2026-11-01

Reference: unmodified main 81989f49193f990d0311b8dab73f1282ff52e486.
Modular numerical implementation: 737ae312a967815a91a08320dabc47a97e05abaf.
Only verification tools/tests were added during this comparison; production
calculation sources and accepted numerical/performance baselines were unchanged.

Environment: Windows, Ryzen 7 7700, RTX 4070 SUPER, NVIDIA 621.01,
CUDA/NVRTC 13.0, MSVC 19.44 for the executables, the same installed native
compiler for runtime DLLs. Both builds use C++17 and identical numerical flags.

## Numerical and functional results

Primary run: build/upstream-parity/1793550152141/.
Each implementation completed all 11 suite executables successfully:
251 shared matrix tests, 17 DAE/BDF tests, and the native compiler, adaptive,
architecture, sessions, tabs, Network, library, SPICE and pruning suites.

The exact comparison covered 1,756 JSON files and 5,127,351 numeric entries:

- 1,502 numerical scenario snapshots, including all builtins and supported
  CPU/GPU analysis combinations, adaptive methods, DAE and fixed/adaptive BDF.
- 252 full library input/result records, including trajectories and non-finite
  markers rather than just aggregate success counts.
- Two supplemental JSON observations of codegen, storage and RQA.

All files, types, shapes, strings, integers and finite binary64 values matched.
Bit differences: **0**. Maximum absolute numerical difference: **0**.
Signed zeros are checked; undefined library/RQA numbers use explicit NaN/Inf
markers. Every suite exit code was zero, so missing/skipped GPU work did not
produce a success result.

Extended run: build/upstream-parity-additional/1793552996509/.
Its two files and 222,351 numeric entries also matched exactly. This adds all
36 interpreted fixed CPU methods and 16 interpreted adaptive paths, including
uniform/node output and two maximum-step sizes. It also compares 108 generated
sources, 11 session kinds, three window formats, application settings, CSV text
and 24 RQA configurations. These entries overlap the supplemental observations
above and must not be added to the primary count as entirely new data.

Interpreted DOPRI78 dense output is cubic Hermite in both versions. A coarse
step can exceed the endpoint tolerance between nodes: both versions produced
the same 2.0684e-7 interpolation error in the diagnostic fixture. The independent
oracle therefore uses the cubic interpolation bound `lambda^4*h^4/384`, a
separate endpoint budget, and verifies fourth-order convergence on step halving.
No solver setting/default or ordinary regression tolerance was changed.

Additional coverage checks read the original main enum/API headers directly:
both passed. New comparator, coverage and observation checks passed in Release
(5/5), Debug (5/5) and the GUI-off observation configuration (1/1).

## Deliberate compatibility differences

These are separately recorded by the compatibility probe and are not part of
the common numerical-output set:

| Input/operation | Main | Modular version |
|---|---|---|
| Load ordinary phase JSON | Generates KRS cache immediately | Clears cache; generates it in analysis before execution |
| Explicitly regenerate that KRS | Same generated source | Same generated source |
| Retired local adaptive.method block with enabled=true | Can enable the upstream adaptive solver | Ignores the retired block; upstream solver stays disabled |

Thus literal identity of every internal field and obsolete-format behavior is
not claimed. These differences implement the architecture and compatibility
rules already documented in [INTEGRATION.md](../../INTEGRATION.md).

## Performance and limits

Representative paired timing uses [measure.ps1](measure.ps1), 16 workloads,
three interleaved series of nine repetitions, one-second warmup/minimum time
and the unchanged 10% slowdown/variation limits. Its separate raw report is
retained under the primary run's performance/ directory.

The user stopped testing during the second main series. Only the first main
and modular series completed; the planned three-series comparison did not.
No performance pass or absence of slowdown is claimed. Raw partial results
remain in performance/1793553240034/, marked stopped_by_user.

GUI interaction/rendering, external KiCad/ngspice and Linux execution are not
validated by this Windows numerical comparison. Source-body token checks also
confirm unchanged CUDA, integrator, codegen and CSV algorithms, but are not
presented as a proof of preprocessor context, signatures or every possible input.

Reproduction and tool behavior: [README.md](README.md).
