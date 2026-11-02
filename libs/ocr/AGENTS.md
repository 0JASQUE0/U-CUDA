# ocr agent rules

Use [README.md](README.md) for module entry points; consult [ARCHITECTURE.md](../../ARCHITECTURE.md) only for boundary or ownership changes.

- OCR failure must not prevent the rest of the application from starting.
- Keep process/path handling in platform abstractions.
- Keep the wire protocol synchronized with `tools/ocr/ocr_server.py`.
- Do not add GUI dependencies.

Validation: Run Linux `platform` protocol tests where available and perform an optional end-to-end OCR smoke test with the configured Python environment.
