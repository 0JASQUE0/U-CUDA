# VS Code configuration

The workspace uses CMake Tools as the source of build targets and executable paths.

- **extensions.json**: recommended Windows/Linux C++ and CMake extensions.
- **settings.json**: CMake and language-service integration.
- **tasks.json**: configure/build/run tasks.
- **launch.json**: Windows MSVC debugger and Linux GDB launch configurations.

Open the repository root, select a configure preset, select target **U-CUDA**, then use Ctrl+Shift+B or F5. Do not hard-code a developer-specific CUDA, compiler, or repository path; use CMakeUserPresets.json for local overrides.
