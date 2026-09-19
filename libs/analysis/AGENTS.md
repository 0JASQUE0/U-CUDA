# analysis agent rules

Use [README.md](README.md) for module entry points; consult [ARCHITECTURE.md](../../ARCHITECTURE.md) only for boundary or ownership changes.

- Workers operate on request copies and result objects, not mutable UI/session containers.
- Keep cancellation tokens, progress, future polling, and running indices consistent.
- Sessions contain no plot views, ImGui state, or OpenGL resources.
- Do not advance global queues or persist files from draw functions.

Validation: Run `architecture`, relevant CPU/GPU tests, queue/cancellation tests, and session serialization checks.
