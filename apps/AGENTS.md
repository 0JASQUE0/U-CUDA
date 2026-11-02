# Application rules

Use [README.md](README.md) for application ownership; consult the root [ARCHITECTURE.md](../ARCHITECTURE.md) when changing boundaries or runtime flow.

- Keep applications thin: create components, resolve resources, run the main loop, and tear down in dependency order.
- Put reusable computation, persistence, platform, plotting, and UI behavior in the matching library.
- Apply `ucuda_runtime()` to executable targets that need packaged resources.
- Preserve executable names and portable layout.
- Build both applications after shared CUDA or domain API changes.
