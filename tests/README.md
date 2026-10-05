# Tests

CTest registers standalone executables:

| Test | Purpose | Hardware |
|---|---|---|
| **adaptive_cpu** | Fixed and upstream adaptive CPU integration, code generation, JSON compatibility | CPU |
| **adaptive_gpu** | CPU/GPU agreement and GPU analysis families | NVIDIA GPU |
| **architecture** | old JSON, round trips, queue cleanup, cancellation without GUI | CPU |
| **gpu_smoke** | clean NVRTC compile, module loading, minimal GPU calculation | NVIDIA GPU |
| **platform** | Linux process/OCR IPC and native compilation | Linux |

The historical names `adaptive_cpu` and `adaptive_gpu` are retained for command compatibility; both validate fixed-step regressions and the upstream adaptive driver, including compatibility with the retired local JSON schema.

GPU tests return skip code 77 when CUDA is unavailable. That is environment information, not computation validation. Tests receive the same packaged runtime resources as applications.
