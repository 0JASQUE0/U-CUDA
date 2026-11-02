# Equations module

## Purpose

Turns user equations into executable numerical source. It parses systems and expressions, evaluates expressions, maps scheme names, generates fixed-step and embedded adaptive schemes.

## Public API and internals

Public APIs are `sysparse.hpp` and `codegen.hpp`. `adaptive_params.h` turns domain
`AdaptiveSettings` and generated `AdaptiveCode` into the driver's `UcudaAdaptParams`;
the settings themselves stay in domain `adaptive_settings.h`. Private parsing/evaluation helpers are split across `expression_parser.cpp`, `system_evaluator.cpp`, and `expression_internal.h`.

`dae_ir.h` represents differential/algebraic equations and index metadata;
`dae_devices.h` expands circuit devices. DAE, BDF residual/Jacobian, adaptive
and fixed generators share the private equations helpers. `krs_prep.h` prepares
generated steps without depending on the native compiler or CUDA.

## Dependencies and consumers

Target `ucuda::equations` depends publicly on domain. Numerics and analysis consume it; it must not depend on CUDA, storage, application, or GUI.

Domain is public because generated-code APIs expose shared system and numerical types. Parser and generator helpers remain private.

## Runtime data flow and ownership

Equation text enters the parser, produces a domain system, and then feeds expression evaluation or scheme generation. Generated source is an owned string passed to numerics or CUDA; this module does not compile, cache, launch, or persist it.

## Change this when

Change parser syntax in parsing files, expression semantics in evaluator files, and numerical source layout in scheme generators.

## Build and checks

Build with `cmake --build --preset windows-debug --target ucuda_equations`. Required checks are listed in [AGENTS.md](AGENTS.md).

## Compatibility and platforms

Preserve accepted syntax, diagnostic meaning, generated function signatures, operation order, precision, and FMA-sensitive expressions. Generated source must remain valid for both native host compilation and NVRTC.

See [the library map](../README.md), [the architecture document](../../ARCHITECTURE.md), and [integration notes](../../INTEGRATION.md).
