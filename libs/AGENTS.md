# Library rules

- Use the module README when locating ownership or changing a module contract.
- Public declarations go under `include/ucuda/<module>/`; private helpers go under `src/`.
- Do not use a repository-wide include path.
- Declare only required CMake dependencies and use `PUBLIC` only for public-header requirements.
- Keep the dependency graph acyclic and preserve headless builds.
- Moving a type requires updating all public includes, CMake links, runtime packaging where relevant, and compatibility tests.

See [the library map](README.md) and [the root architecture](../ARCHITECTURE.md).
