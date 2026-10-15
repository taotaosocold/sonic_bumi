# Parallel-ankle vendor packages

Each series keeps its own algorithm and parameters. Do not substitute one
series for the other without validating the mechanism and motor conventions.

- `n3/` contains the N3HW-derived source algorithm. `.hpp` and `.cpp` live
  together. SDK2 builds these five `.cpp` files into `noetix_n3_hardware`.
- `bumi/` contains the original Bumi vendor API headers and prebuilt
  `libparallel_ankle_cpp.a` archives, selected by target architecture.

The public `noetix_sdk2` API and runtime hardware configuration paths are
unchanged.
