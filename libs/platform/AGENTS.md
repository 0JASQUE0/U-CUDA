# platform agent rules

Use [README.md](README.md) for module entry points; consult [ARCHITECTURE.md](../../ARCHITECTURE.md) only for boundary or ownership changes.

- Keep OpenGL, ImGui, analysis, and application types out of public headers.
- Use executable-relative paths for portable assets.
- Keep POSIX declarations buildable on Windows through guarded stubs/includes.
- Do not mix framebuffer capture with clipboard encoding.

Validation: Build on the affected OS. On Linux run `platform`; on Windows smoke-test dialogs and clipboard behavior.
