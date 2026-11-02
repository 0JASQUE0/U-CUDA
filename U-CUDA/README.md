# Local compatibility data

This directory is intentionally retained after the source-code migration.

- **library/** is the ignored writable user system library used by source builds.
- **ocr_error.log**, when present, is local runtime output and is not architecture.
- C++/CUDA sources, kernels, OCR tools, and Windows resources live elsewhere.

Do not move, normalize, seed over, or commit files from **library/**. Distributed baseline systems live in [resources/library_template](../resources/README.md). Set **U_CUDA_LIBRARY** to use another writable location.
