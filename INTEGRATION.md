# Integration: fixed and upstream adaptive steps

The adaptive implementation comes from [0JASQUE0/U-CUDA, commit 77568e0](https://github.com/0JASQUE0/U-CUDA/commit/77568e0349f8190187db28748f8783bbe161f355). It replaces the retired local DOPRI5 wrapper. The upstream sources are adapted to this repository's modules and portable runtime compilation; the user library is not copied or overwritten.

## Fixed step

Fixed is the default. The selected built-in or custom scheme advances with constant `h`; phase `decimation` and parameter `preScaller` retain their existing sampling roles. Complex schemes, discrete maps, and sweeps over `h` remain available. FastSync keeps the upstream fixed-step path.

Upstream corrects LLE/LS normalization blocks: the extra fixed-point check is included in `NT/h`, rather than adding another integration step. For `NT=0.1`, `h=0.01`, a block advances 10 steps. This intentionally adopts upstream behavior.

## Adaptive step

Choose an embedded scheme: RK45, DOPRI78 (including legacy coefficients), DOP853, GBS 2-4 and higher, or `Extr(base|n1,...,nK)` with at least two stages. Extrapolation bases may be custom or complex schemes. An unsupported scheme reports a validation error.

The upstream driver lives in `resources/kernels/ucuda_adaptive.cuh`. Native CPU modules and NVRTC kernels share it. The interpreted CPU phase path supports the embedded RK schemes; generated native modules handle the remaining schemes. Settings include `rtol`, scalar or per-variable `atol`, initial/minimum/maximum steps, rejection limits and controller parameters. Hairer fast is the upstream default; Hairer numb computes its controller in double. User controller definitions can be edited and travel with saved sessions.

Output can use a uniform grid through dense interpolation or accepted-step nodes. `h` specifies uniform output spacing; decimation reduces the output. Accepted nodes have explicit times, and step diagnostics report accepted/rejected/forced steps, RHS evaluations and observed step sizes. Dense interpolation order depends on the scheme; tolerances control accepted endpoints and do not guarantee the same error between endpoints.

Adaptive paths cover phase portraits, bifurcation, LLE/LS, basins, metrics, supported continuation modes, Order/performance and Custom pipelines. The standalone DFT sweep and FastSync retain upstream fixed-step execution. Unsupported combinations use upstream validation. LLE/LS can normalize exactly at multiples of `NT` or at the first accepted node after each interval.

## Sessions and exports

Sessions without an adaptive block load as Fixed. Default settings omit the block when saving, preserving fixed CSV formats. Non-default upstream settings round trip in ordinary, parameter and Custom sessions. Adaptive exports include their step-control metadata and explicit node times where applicable.

The retired local JSON schema with a numeric `adaptive.method` is ignored, without enabling the upstream solver. Scheme, `h` and other settings remain intact. Upstream controller names and embedded definitions use upstream compatibility rules.

## Validation

```powershell
cmake --preset windows
cmake --build --preset windows-debug
ctest --preset windows-debug
```

`adaptive_cpu` and `adaptive_gpu` cover both fixed regression cases and the upstream solver, including scalar tolerance expansion, transient endpoints, dense/node output, custom/complex extrapolation, serialization and CPU/GPU agreement. `architecture` covers GUI-free lifecycle and serialization; `gpu_smoke` checks packaged NVRTC resources. GPU tests must pass on hardware without skipping.
