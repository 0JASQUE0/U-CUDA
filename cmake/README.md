# CMake infrastructure

- **Targets.cmake** defines common static-library conventions and aliases.
- **Runtime.cmake** defines **ucuda_runtime()**, output locations, post-build packaging, and Windows DLL discovery.
- **PackRuntime.cmake** copies kernel templates, assembles flat NVRTC headers from shared sources, and copies OCR/library templates.
- **CopyRuntimeDlls.cmake** copies resolved runtime DLLs while filtering invalid entries.

The root CMakeLists.txt controls module order and optional GUI/legacy/test subtrees. Keep build-tree paths out of sources and preserve executable-relative runtime behavior.
