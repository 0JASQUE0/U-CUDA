# Third-party rules

- Do not edit vendored/generated sources for project behavior.
- Keep fetched revisions pinned and mutually compatible.
- Keep demo, test, and example subprojects disabled.
- Do not load GUI dependencies when U_CUDA_BUILD_GUI=OFF.
- Record why a dependency revision changes and rebuild GUI targets on both supported platforms.

See [the dependency README](README.md) and [the root architecture](../ARCHITECTURE.md).
