# storage agent rules

Use [README.md](README.md) for module entry points; consult [ARCHITECTURE.md](../../ARCHITECTURE.md) only for boundary or ownership changes.

- Never require `AppModel` in serialization.
- Treat field names, missing-field defaults, enum values, CSV spelling, separators, and ordering as compatibility contracts.
- Ignore unknown JSON fields where existing readers do so; never resurrect runtime work on load.
- Do not read or overwrite user library data during tests without a temporary root.

Validation: Run `architecture`, old-save load tests, round trips, and byte/shape checks for affected CSV outputs.
