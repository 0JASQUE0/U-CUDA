# OCR tool rules

- Keep ocr_server.py protocol-compatible with libs/ocr.
- Return structured failures instead of hanging or corrupting stdout framing.
- Keep logs and model caches out of Git.
- Update requirements.txt only for intentional, tested dependency changes.
- Verify missing-Python behavior and one end-to-end recognition after protocol changes.

See [the OCR tool README](README.md) and [the root architecture](../../ARCHITECTURE.md).
