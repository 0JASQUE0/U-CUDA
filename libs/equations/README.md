# Equations module

## Purpose

Turns user equations into executable numerical source. It parses systems and expressions, evaluates expressions, maps scheme names, generates fixed-step schemes, and wraps schemes with adaptive integration code.

## Public API and internals

Public APIs are `sysparse.hpp`, `codegen.hpp`, and `adaptive_codegen.h`. Private parsing/evaluation helpers are split across `expression_parser.cpp`, `system_evaluator.cpp`, and `expression_internal.h`.

## Dependencies and consumers

Target `ucuda::equations` depends publicly on domain. Numerics and analysis consume it; it must not depend on CUDA, storage, application, or GUI.

Domain is public because generated-code APIs expose shared system and numerical types. Parser and generator helpers remain private.

## Runtime data flow and ownership

Equation text enters the parser, produces a domain system, and then feeds expression evaluation or scheme generation. Generated source is an owned string passed to numerics or CUDA; this module does not compile, cache, launch, or persist it.

## Change this when

Change parser syntax in parsing files, expression semantics in evaluator files, and numerical source layout in scheme/adaptive generators.

## Build and checks

Build with `cmake --build --preset windows-debug --target ucuda_equations`. Run `adaptive_cpu`; run `adaptive_gpu` and `gpu_smoke` when generated code changes.

## Compatibility and platforms

Preserve accepted syntax, diagnostic meaning, generated function signatures, operation order, precision, and FMA-sensitive expressions. Generated source must remain valid for both native host compilation and NVRTC.

See [the library map](../README.md), [the architecture document](../../ARCHITECTURE.md), and [adaptive integration notes](../../ADAPTIVE_INTEGRATION.md).
