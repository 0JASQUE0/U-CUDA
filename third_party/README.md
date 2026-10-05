# Third-party dependencies

This directory contains checked-in GLAD and stb sources plus the CMake description for GUI dependencies. CMake fetches pinned ImGui, ImPlot, ImPlot3D, and GLFW revisions; it may use a system GLFW when available.

Only adapter/build concerns belong here. Application behavior stays in project libraries. Updating a pinned revision requires Windows and Linux GUI builds plus visual smoke tests for docking, 2D/3D plots, input, and shutdown.
