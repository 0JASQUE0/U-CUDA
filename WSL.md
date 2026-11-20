# Building and running under WSL2

U-CUDA builds and runs in an Ubuntu WSL2 distribution, GUI included: CUDA works through the Windows NVIDIA driver, and the window is shown by WSLg. The `linux-wsl` preset in [CMakePresets.json](CMakePresets.json) covers this setup; the other Linux presets assume GCC 15 and CUDA in `/opt/cuda`.

Checked on Ubuntu 24.04, GCC 13.3, CUDA 13.2, CMake 3.28 and an RTX 5070 Ti.

## Prerequisites

- **Windows side:** a current NVIDIA driver and Windows 11 or a Windows 10 build with WSLg. Do not install a Linux NVIDIA driver inside WSL. `nvidia-smi` in the distribution must list the GPU.
- **CUDA Toolkit** in `/usr/local/cuda`, installed from NVIDIA's WSL-Ubuntu package repository (see the [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/)). Install the toolkit only, without a driver package.
- **Build tools and GUI libraries:**

```bash
sudo apt install cmake ninja-build libgl1-mesa-dev libx11-dev libxrandr-dev \
  libxinerama-dev libxcursor-dev libxi-dev libxkbcommon-dev wl-clipboard zenity
```

CMake 3.25 or newer is required. Both GCC 13 and CUDA 13.2 come as Ubuntu 24.04 packages. `zenity` provides the file-open and save dialogs. `wl-clipboard` (or `xclip`) is used for copying images to the clipboard and for OCR from the clipboard.

## Configure, build, test

```bash
cmake --preset linux-wsl
cmake --build --preset linux-wsl
ctest --preset linux-wsl
```

The output goes to `build/linux-wsl/bin/`. With `CMAKE_CUDA_ARCHITECTURES=native` the static CUDA code is built for the GPU found when you configure. NVRTC kernels are always compiled at runtime for the device in use.

`ctest --preset linux-wsl` runs the whole suite, which takes about 25 minutes on an RTX 5070 Ti. For a quick check after building:

```bash
build/linux-wsl/bin/gpu_smoke
ctest --preset linux-wsl -R 'Smoke|architecture|layer_rules'
```

## Run the desktop app

```bash
build/linux-wsl/bin/U-CUDA
```

Packaged resources (`kernels/`, `library_template/`, `ocr_server.py`) are copied next to the binary at build time. As on Windows, a source build keeps user systems in `U-CUDA/library/`. To work on a separate library, set `U_CUDA_LIBRARY=/some/dir`.

## Notes

- **GLFW runs on X11.** The preset sets `GLFW_BUILD_WAYLAND=OFF`, and the window goes through XWayland in WSLg. The Wayland backend needs `pkg-config`, `wayland-protocols` and `libwayland-dev`. To try it anyway, install those and configure with `-DGLFW_BUILD_WAYLAND=ON`.
- **OpenGL under WSLg** runs on Mesa's D3D12 driver, which is slower than native OpenGL. Computation is unaffected, because plots receive their data through host memory rather than CUDA–OpenGL interop.
- **Exit code 139 on close.** Closing the window with the close button that Windows draws on WSLg windows ends the process with a segmentation fault (exit code 139) during shutdown. Sessions are saved while the app runs, so no work is lost, and nothing else is affected.
- **`CUDA_PATH` is optional on Linux.** NVRTC falls back to the toolkit include directory found by CMake.
- **The CPU backend** compiles user schemes with `$CXX` or, if that is unset, `c++`. It needs no extra setup on Linux.
- **OCR is optional.** See [tools/ocr/README.md](tools/ocr/README.md). `setup.ps1` is Windows-only; on Linux, create a virtual environment from `tools/ocr/requirements.txt` and set `U_CUDA_PYTHON` to its interpreter.
- **A headless build** (`-DU_CUDA_BUILD_GUI=OFF`) needs none of the GUI packages.
