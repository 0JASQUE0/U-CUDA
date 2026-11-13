# Resources

This directory contains source-controlled data copied into runtime packages.

- [kernels](kernels/README.md): NVRTC templates loaded at runtime.
- **library_template/**: baseline systems copied beside executables and used only to seed a missing writable library.
- **windows/**: executable icon and resource script.
- **themes/**: window-background textures of the camouflage UI themes (binary P6 PPM, 512×512, seamless, muted), loaded by `apps/desktop/app_main.cpp` (`kCamoPalettes`). `multicam.ppm` is downscaled from a photo of the fabric; `dpm.ppm`, `ddpm.ppm` (two-colour) and `flora.ppm` (VSR-98) are procedural (blurred wrap-around noise thresholded into layers). Without a file the theme falls back to a plain background.

[cmake/PackRuntime.cmake](../cmake/PackRuntime.cmake) creates the executable-side layout. Resources must be found relative to the executable so launching from another current working directory works. **library_template** is distributable input; **U-CUDA/library** is private writable user data.
