# ui agent rules

Use [README.md](README.md) for module entry points; consult [ARCHITECTURE.md](../../ARCHITECTURE.md) only for boundary or ownership changes.

- UI sends commands and renders state; it does not poll workers or advance queues.
- Never store plot views or GL objects in domain/analysis sessions.
- Use stable runtime IDs when associating views with projections.
- Release all UI resources before the OpenGL context is destroyed.
- Keep numerical algorithms out of draw code.

Validation: Build and smoke-test GUI session switching, window creation/closing, queues, save/load/export, OCR, screenshots, and shutdown.
