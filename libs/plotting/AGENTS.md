# plotting agent rules

Use [README.md](README.md) for module entry points; consult [ARCHITECTURE.md](../../ARCHITECTURE.md) only for boundary or ownership changes.

- Plotting must not depend on application or storage.
- Own and destroy every GL buffer/texture in this layer.
- Capture pixels here; transfer/encode them through platform.
- Keep rendering resources out of serialized sessions.

Validation: Build `U-CUDA`; exercise 2D, 3D, heatmap, zoom/camera, screenshot, and context shutdown paths.
