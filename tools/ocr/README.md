# OCR server

**ocr_server.py** is the optional Python process used to convert images into LaTeX. **requirements.txt** pins its environment; **setup.ps1** creates the repository virtual environment and sets **U_CUDA_PYTHON**.

The C++ peer is [libs/ocr](../../libs/ocr/README.md). Keep request framing, base64/image handling, output framing, errors, and shutdown behavior synchronized. The application must remain usable when Python or OCR models are unavailable.

Use a test environment for protocol work. The dependency set is large and is not installed by the CMake build.
