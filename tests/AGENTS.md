# Test rules

- Add focused behavior tests; do not mirror implementation line by line.
- Every objectively testable feature or behavior change requires automated coverage in the same change, including meaningful boundary cases and failures. Bug fixes require a reproducer.
- New numerical methods require independent accuracy/convergence checks on every supported CPU/GPU path; extend benchmarks for computation and other performance-sensitive paths.
- Register new cases in the appropriate smoke/regression/performance modes and the capability matrix. Never loosen tolerances or refresh baselines merely to pass.
- When automation is objectively impossible, explain why and provide a reproducible manual check in the change description.
- Serialization changes require an old-format fixture and a round trip.
- Queue/cancellation behavior must be testable without GUI.
- CUDA/NVRTC changes require a real, non-skipped GPU run.
- Platform tests must use temporary paths and processes.
- Register new executables with CTest and ucuda_runtime() when they load runtime kernels.

See [the test README](README.md) and [the root architecture](../ARCHITECTURE.md).
