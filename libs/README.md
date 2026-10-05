# Libraries

Each module is a static CMake target with public headers in `include/ucuda/<module>/` and private implementation in `src/`.

| Module | Responsibility | Target |
|---|---|---|
| [domain](domain/README.md) | Shared plain types and persisted state | `ucuda::domain` |
| [platform](platform/README.md) | OS paths, processes, dialogs, clipboard | `ucuda::platform` |
| [equations](equations/README.md) | Parsing, evaluation, scheme code generation | `ucuda::equations` |
| [numerics](numerics/README.md) | CPU integration and native KRS compilation | `ucuda::numerics` |
| [cuda](cuda/README.md) | NVCC/NVRTC infrastructure and GPU algorithms | `ucuda::cuda` |
| [analysis](analysis/README.md) | Sessions and asynchronous analysis workflows | `ucuda::analysis` |
| [storage](storage/README.md) | JSON, system library, settings, CSV | `ucuda::storage` |
| [ocr](ocr/README.md) | OCR process client | `ucuda::ocr` |
| [application](application/README.md) | App model, queues, global tick | `ucuda::application` |
| [plotting](plotting/README.md) | OpenGL plots and framebuffer capture | `ucuda::plotting` |
| [ui](ui/README.md) | ImGui panels and UI-owned views | `ucuda::ui` |

See the root [ARCHITECTURE.md](../ARCHITECTURE.md) for the dependency graph.
