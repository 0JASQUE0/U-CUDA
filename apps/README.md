# Applications

This directory contains executable composition roots. Applications may connect libraries, select resource paths, and own process lifetime; reusable behavior belongs in `libs/`.

- [desktop](desktop/README.md): interactive ImGui/OpenGL application, target `U-CUDA`.
- [legacy](legacy/README.md): old batch calculation entry point, target `U-CUDA-legacy`.

Both targets use `ucuda_runtime()` to receive kernels, templates, OCR server, and Windows runtime DLLs. Add a new executable here only when it represents a distinct process or workflow.
