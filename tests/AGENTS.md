# Test rules

- Add focused behavior tests; do not mirror implementation line by line.
- Serialization changes require an old-format fixture and a round trip.
- Queue/cancellation behavior must be testable without GUI.
- CUDA/NVRTC changes require a real, non-skipped GPU run.
- Platform tests must use temporary paths and processes.
- Register new executables with CTest and ucuda_runtime() when they load runtime kernels.

See [the test README](README.md) and [the root architecture](../ARCHITECTURE.md).
