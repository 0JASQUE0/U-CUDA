# Tests

CTest registers standalone executables:

| Test | Purpose | Hardware |
|---|---|---|
| **adaptive_cpu** | CPU adaptive solver, code generation, JSON compatibility | CPU |
| **adaptive_gpu** | CPU/GPU agreement and GPU analysis families | NVIDIA GPU |
| **architecture** | old JSON, round trips, queue cleanup, cancellation without GUI | CPU |
| **gpu_smoke** | clean NVRTC compile, module loading, minimal GPU calculation | NVIDIA GPU |
| **platform** | Linux process/OCR IPC and native compilation | Linux |

GPU tests return skip code 77 when CUDA is unavailable. That is environment information, not computation validation. Tests receive the same packaged runtime resources as applications.
