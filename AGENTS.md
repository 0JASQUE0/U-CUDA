# Repository-wide agent instructions

Use [README.md](README.md) to locate the owning module. Consult [ARCHITECTURE.md](ARCHITECTURE.md) when changing dependencies, ownership, runtime flow, or moving code across modules. Nested `AGENTS.md` files add scoped rules and are loaded automatically.

## Workflow

- Inspect the working tree first and preserve unrelated changes. Keep edits focused; do not bundle cleanup, dependency upgrades, or public API redesigns.
- Read a module README when first entering that module or when changing its contract, build, ownership, or runtime behavior. Use [SETUP.md](SETUP.md) only for environment setup and command examples.
- Ask only when requirements leave materially different behavior choices unresolved, especially numerical defaults, persisted formats, or destructive data operations.
- Use the installed `codebase-memory` skill for structural exploration. Check index freshness and cited-path coverage; use source reads and `rg` for stale, partial, or missing coverage.

## Project invariants

- Use C++17 and CUDA C++17 as configured by CMake. Match local naming, formatting, includes, and error-handling conventions. Follow `.editorconfig`; keep `setup.ps1` ASCII-only.
- CMake is the only build definition. Public headers belong in `include/ucuda/<module>/`; implementation details belong in `src/`. Use `PUBLIC` dependencies only when exposed by a public header.
- Keep dependencies one-way. Domain is independent of CUDA, storage, application, plotting, and UI. Storage and computation build without GUI.
- Sessions contain serializable computation state, never ImGui/OpenGL resources. UI owns views; plotting owns GL objects and releases them before the GL context.
- Background work uses request snapshots, result objects, and existing cancellation/progress mechanisms. Polling, queue advancement, persistence, and cancellation run through `AppModel::tick`; worker threads do not access UI/GL state.
- Release CUDA, NVRTC, OS, process, and GL resources on failure and cancellation. Check fallible operations and validate external sizes, ranges, and buffer compatibility.

## Compatibility

- Preserve numerical operation order, precision, FMA settings, kernel arguments, and defaults unless a behavior change is requested.
- NVCC and NVRTC consume the same device algorithm sources. Update runtime packaging when a required source or header changes.
- Preserve old JSON sessions, system records, settings, and CSV formats. Serialization changes require old-format load and round-trip coverage.
- Keep C++/CUDA source encoding and MSVC `/utf-8`. Never modify or seed over `U-CUDA/library/`.

## Verification

- Add focused regression tests for behavior changes and bug fixes where practical.
- Build affected targets; build `U-CUDA-legacy` when shared compute signatures change.
- Run `adaptive_cpu`, `architecture`, and relevant platform tests. CUDA/NVRTC changes also require non-skipped `adaptive_gpu` and `gpu_smoke` runs on a real GPU.
- Configure once with `U_CUDA_BUILD_GUI=OFF` after dependency-boundary changes.
- Run `git diff --check`. Report passed, failed, skipped, and not-run checks accurately.
- For documentation-only changes, verify local links and commands plus `git diff --check`; compilation is not required.
