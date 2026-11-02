# Storage module

## Purpose

Owns application settings, system-library layout, JSON session persistence, and CSV export formatting. Per-analysis serializers are separated so compatibility changes remain local.

## Public API and internals

Public APIs are `app_config.h`, `system_library.h`, `session_io.h`, and `data_export.h`. The internal `ucuda_export_format` target shares byte-identical writers with compute-time streaming.

Session serializers take the domain `<Name>State`. Sessions that own other sessions get the owned parts explicitly: `basins_state_to_json` / `basins_state_from_json` take the phase-portrait projections of the slots as a list and report what the slot owner must do (`BasinsSlotsAfterLoad`); `custom_state_to_json` / `custom_state_from_json` take the five sub-states. The whole-session wrappers with the old names (`session_to_json_basins`, `session_from_json_custom`, ...) are in application `session_persistence.h`.

## Dependencies and consumers

`ucuda::storage` depends on domain and privately on platform/export formatting. Application and UI consume it. Storage must remain GUI-free.

It must not include CUDA, analysis engines, application state, plotting, ImGui, or OpenGL. The internal export-format library may be consumed by CUDA streaming code but contains no persistence orchestration.

## Runtime data flow and ownership

Storage translates domain values to and from JSON, resolves the writable system library, and writes export snapshots to CSV. It owns files and temporary serialization buffers only during an operation; application owns save timing and the live model.

## Change this when

Edit the matching `session_<family>.cpp` for session JSON, `system_library.cpp` for library layout, `app_config.cpp` for settings, and `data_export.cpp` for CSV.

## Build and checks

Build with `cmake --build --preset windows-debug --target ucuda_storage`. Required checks are listed in [AGENTS.md](AGENTS.md).

## Compatibility and platforms

Old JSON sessions, systems, settings, missing optional fields, and CSV column order/format are compatibility contracts. Keep path behavior portable and never overwrite the user-owned `U-CUDA/library/` while testing.

See [the library map](../README.md) and [the architecture document](../../ARCHITECTURE.md).
