# equations agent rules

Use [README.md](README.md) for module entry points; consult [ARCHITECTURE.md](../../ARCHITECTURE.md) only for boundary or ownership changes.

- Preserve expression precedence and existing accepted syntax.
- Keep generated source valid for the native compiler and NVRTC.
- Do not perform file I/O, UI work, or GPU launch logic here.
- Preserve scheme operation order unless a numerical change is explicitly intended.

Validation: Run `adaptive_cpu`; run `adaptive_gpu` and `gpu_smoke` when generated code changes.
