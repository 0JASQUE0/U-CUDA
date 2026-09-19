# NVRTC kernel templates

These **.template.cu** files are runtime compilation units for bifurcation, LLE/LS, DFT, basins, FastSync, and Order analyses. Engines substitute generated scheme code and runtime settings, then compile with NVRTC.

Shared device algorithms come from CUDA sources packaged into the runtime kernels directory by PackRuntime.cmake; do not duplicate helpers into individual templates. Template file names are runtime contracts referenced by engine code.

After changes, clear the copied kernel directory or use a fresh build, run **gpu_smoke**, run the relevant GPU analysis test, and verify cache invalidation inputs.
