# domain agent rules

Use [README.md](README.md) for module entry points; consult [ARCHITECTURE.md](../../ARCHITECTURE.md) only for boundary or ownership changes.

- Keep types GUI-free, storage-free, and independent of CUDA runtime APIs.
- Persist only stable data; runtime projection IDs, futures, tokens, and device/GL handles are not serialized.
- Preserve enum numeric order and defaults used by old JSON or kernels.
- Keep `configCUDA.h` compatible with both native compilation and NVRTC packaging.

Validation: Run `architecture`, serialization round trips, CPU tests, and GPU tests when request/result ABI or numerical types change.
