# numerics agent rules

Use [README.md](README.md) for module entry points; consult [ARCHITECTURE.md](../../ARCHITECTURE.md) only for boundary or ownership changes.

- Use the same generated scheme semantics as CUDA.
- Keep compiler paths and temporary outputs platform-aware.
- Do not introduce GUI or storage dependencies.
- Preserve diagnostics and cleanup on compilation failure.

Validation: Run `adaptive_cpu`, `architecture`, and Linux `platform`; compare CPU/GPU deterministic calculations after numerical changes.
