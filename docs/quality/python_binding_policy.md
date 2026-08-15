# Python binding policy

Wolfgang's native extension has two registration layers:

- `stable_bindings.cpp` and `pauli_sum_py.cpp` register the supported Python API.
- `internal_bindings.cpp` owns operational introspection plus unsupported research,
  benchmark, probe, and test hooks. Underscored hooks are not compatibility promises.

`WOLFGANG_ENABLE_INTERNAL_BINDINGS=OFF` is the default for every source and wheel
build. Repository developers explicitly opt in with
`-Ccmake.define.WOLFGANG_ENABLE_INTERNAL_BINDINGS=ON` when running validation
campaigns. Keeping the default off removes research, benchmark, probe, and test
registrations. The underscored build
and accelerator status functions remain because the supported `capabilities()` API
uses them internally; callers should use `wolfgang_quantum.capabilities()` instead.

Official release wheels set `WOLFGANG_ENABLE_INTERNAL_BINDINGS=OFF` explicitly in
the cibuildwheel configuration. Release wheels therefore contain the stable API and
the minimum internal operational introspection needed to implement it, but not the
unsupported campaign machinery. Source distributions retain the code so developers
can opt in when validating an accelerator source build.

The option does not enable an accelerator and makes no accelerator support claim.
CUDA, HIP, and Metal availability remains governed by their dedicated build options
and the documented support matrix.