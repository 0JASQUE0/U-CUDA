# Resources

This directory contains source-controlled data copied into runtime packages.

- [kernels](kernels/README.md): NVRTC templates loaded at runtime.
- **library_template/**: baseline systems copied beside executables and used only to seed a missing writable library.
- **windows/**: executable icon and resource script.

[cmake/PackRuntime.cmake](../cmake/PackRuntime.cmake) creates the executable-side layout. Resources must be found relative to the executable so launching from another current working directory works. **library_template** is distributable input; **U-CUDA/library** is private writable user data.
