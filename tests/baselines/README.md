# Accepted baselines and migration evidence

Accepted JSON manifests, immutable case artifacts and history versions live here.
`active.json` is the current manifest; `active-data/` shares unchanged case files
between the manifests in `active-history/`. All three belong in version control.
Raw run histories remain under build/<preset>/test-results/ and are not pruned.
See [the runner documentation](../README.md) for explicit acceptance commands.
Actual local checks and timing limitations are recorded in [validation](../VALIDATION.md).

The pre-merge source was modular-adaptive
68d05ba4899d1e74377148f42bc7225c53b7151a; incoming main was
81989f49193f990d0311b8dab73f1282ff52e486. Original CTest passed 4/4 on
RTX 4070 SUPER, NVIDIA 621.01, Ryzen 7 7700, MSVC 19.44, CUDA 13.0.
The migration CSV records Euler, RK4, DOPRI78 and Complex CD on CPU/GPU,
x'=-x and y'=-2y, h=.001, 1001 output points, 300 ms warmup and 27 timing
samples. Each sample batches 50 full calls. These values are evidence from
the previous branch, not an automatically accepted reference for new features.

Independent correctness checks must pass before a merged reference is accepted.
The reviewed behavior changes are listed in [INTEGRATION.md](../../INTEGRATION.md).

The [comparison](migration-68d05ba/comparison.csv) uses consecutive original and
merged executions with a fresh private native-compiler cache. All eight
CPU/GPU cases pass: maximum endpoint difference is 1.11e-16; no sustained
slowdown exceeds 10%. RK4 is about 16% faster on CPU and 29% faster on GPU in
this workload. These are small trajectory workloads, not general speedup claims.
The [original capture](migration-68d05ba/numerical-performance.csv),
[original CTest log](migration-68d05ba/ctest.txt), repeat captures and the first
comparison are retained. The first repeat's Euler CPU timing was inconclusive;
the clean-cache repeat passed the same variation limit without changing it.
