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

`load_neural_dataset_csv` reads dense labelled vectors from a local UTF-8 CSV.
The header starts with `id,split,label,group`, followed by unique channel names.
Each row contains a unique sample ID, an explicit split (`train`, `validation`
or `test`), a label, an optional group ID and one finite decimal value per
channel. Group IDs must stay in one split. The loader preserves row and channel
order. Fields are comma-delimited and unquoted. The group field can be empty,
and both LF and CRLF are accepted. All three splits must contain rows.

The loader caps the input file at 512 MiB, each row at 1 MiB, and the decoded
data at 1,000,000 rows and 64 Mi values. Values keep their original scale.

For a Neural classification run, enter the path in **Dataset CSV** and select
**Load dataset**. In **Input**, link the CSV channels to populations. In
**Protocol**, select separate trials and restart from the initial conditions.
The CSV determines the trial count. Trials are grouped by split and preserve
their relative file order within each split. The stimulus holds each vector at
its original scale; the rest supplies zero input. The first path requires
fixed links and runs with continuous or spiking nodes.

The session stores the path and classifier, while the CSV stays external.
After reopening the session, load the CSV before running. The saved classifier
checks the dataset and trial protocol before repeating the held-out test.

Old JSON sessions, systems, settings, missing optional fields, and CSV column order/format are compatibility contracts. Keep path behavior portable and never overwrite the user-owned `U-CUDA/library/` while testing.

See [the library map](../README.md) and [the architecture document](../../ARCHITECTURE.md).
