# application agent rules

Use [README.md](README.md) for module entry points; consult [ARCHITECTURE.md](../../ARCHITECTURE.md) only for boundary or ownership changes.

- Keep `AppModel` independent of ImGui and OpenGL.
- Call all poll/queue/persistence logic from `tick`, not draw visibility paths.
- Maintain valid queue indices when configs are removed.
- Inject OCR behavior through `OcrFn` rather than coupling the model to a concrete client.

Validation: Run `architecture`; test queues, index cleanup, cancellation, save-on-completion, and operation with GUI disabled.
