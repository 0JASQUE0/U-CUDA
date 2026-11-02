# Imported upstream checks

Adapted from main 81989f49193f990d0311b8dab73f1282ff52e486. Sources use public
module headers and CMake targets upstream_<name>. There is no separate build
script. See [the test runner](../README.md) for modes and artifact history.

```powershell
cmake --build --preset windows-release --target upstream_network upstream_sessions upstream_spice
ctest --preset windows-release -R '^upstream_'
```

DAE checks are split into GoogleTest cases in ../dae_gtest.cpp. Library/session
tests use temporary fixture copies. KiCad/ngspice checks require external tools
and explicitly report skip code 77 when unavailable.

The historical RTX 2060 SUPER archive and [Network notes](NETWORK_BASELINE.md)
are preserved as upstream evidence; performance comparisons on other hardware
use a separate reviewed baseline from the native runner.
