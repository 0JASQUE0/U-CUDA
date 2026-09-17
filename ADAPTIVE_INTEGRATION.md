# Adaptive integration

## Using the solver

In an analysis configuration, open **Integration** and choose:

- **Fixed step**: existing behavior, also the default for older sessions.
- **Adaptive DOPRI5**: Dormand–Prince 5(4), evaluating the system's ODE right-hand side. This replaces the selected base scheme. It requires a built-in ODE system, not a custom step-only KRS.
- **Adaptive step doubling**: compare one full step with two half-steps of the selected scheme. The accepted state is the two-half-step solution; no extrapolation correction is applied. Custom KRS requires an explicit order from 1 to 8. Built-in order is inferred, including the symmetry coefficient where relevant.

`Step h` now means the **output/analysis interval** in adaptive mode. It is independent of the initial/minimum/maximum internal step. `rtol` controls relative local error, `atol` absolute local error near zero. Defaults are `1e-7` and `1e-10`. Tolerances apply to the maximum scaled component error. They do not bound long-term trajectory error in a chaotic system.

Each configuration persists its settings in session JSON. The Custom pipeline propagates its shared settings to its analyses. Phase CSV metadata includes adaptive settings. For other analyses, retain the session JSON alongside exported data. Phase results show accepted/rejected step totals and the observed internal-step range. A failure reports its reason rather than silently accepting an inaccurate step; parameter-map failures use the existing invalid-cell flags.

## Algorithms and analysis integration

- DOPRI5 uses its embedded fourth-order estimate and a PI step controller; rejected attempts roll back state and time. Step doubling uses Richardson's error estimate divided by `2^p - 1`.
- Phase trajectories keep independent controllers per initial condition. DOPRI5 uses Shampine's quartic dense extension to emit uniform samples without forcing a step for each sample. Step doubling lands on sample boundaries.
- Bifurcation, DFT and basin kernels retain their existing uniform sampling contracts. Their adaptive substeps end at each analysis interval; controller state restarts at that boundary. This is less efficient than persistent phase integration but preserves existing kernel interfaces.
- LLE/LS integrate the main trajectory and probes as one ensemble with a common error norm and accepted steps. Renormalization/Gram–Schmidt stays at the existing physical-time block boundaries. Both classical and continuation CPU/GPU paths are supported.
- Fast Synchro includes coupling in the local estimate. The master is held at its sampled value during each synchronization interval; backward integration reverses the ODE, not the attractive coupling. Step doubling uses first-order error scaling for this split map. Reducing the synchronization interval remains necessary to check convergence.
- RQA receives uniform phase samples. **Order** intentionally continues to assess fixed-step base schemes.

Existing peak detection and interpolation on uniformly sampled data are preserved. Dense-output derivative root refinement of maxima is not implemented. No adaptive implicit stiff solver, per-component tolerances, or GPU regrouping by step size is provided.

## Validation

```powershell
cmake --preset windows
cmake --build --preset windows-debug
ctest --preset windows-debug
```

`adaptive_cpu` tests analytic decay, a harmonic oscillator, backward integration, dense output, tolerance refinement, limits, generated native code, and session compatibility. `adaptive_gpu` additionally compares CPU/GPU trajectories and Lyapunov results, exercises parameter analyses and complex schemes, and compares Lorenz against a fine-step RK4 reference. A missing CUDA device skips only the GPU test. Native generated-code tests require a C++ compiler.

The DOPRI5 and dense-output coefficient tables were cross-checked against the [SciPy RK45 implementation](https://github.com/scipy/scipy/blob/main/scipy/integrate/_ivp/rk.py), which cites Dormand–Prince (1980) and Shampine (1986). The solver/controller implementation here is shared between native C++ and NVRTC in `U-CUDA/adaptive_solver.h`.
