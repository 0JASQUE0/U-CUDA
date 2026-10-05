# Desktop agent rules

- Keep this directory a composition root; do not implement analysis algorithms here.
- Call `AppModel::tick` every main-loop iteration, regardless of visible windows.
- Release UI-owned views and plotting resources before destroying ImGui/GLFW/OpenGL.
- Resolve packaged resources relative to the executable.
- Never overwrite `U-CUDA/library/`.
- After edits, build `U-CUDA` and perform a short startup/shutdown smoke test from another working directory.

See [the desktop README](README.md) and [the root architecture](../../ARCHITECTURE.md).
