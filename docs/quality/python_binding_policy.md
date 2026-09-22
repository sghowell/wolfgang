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
function remains because the supported `capabilities()` API uses it internally.
Public device discovery supplies accelerator records when private status helpers
are absent; callers should use `wolfgang_quantum.capabilities()` instead.

Official release wheels set `WOLFGANG_ENABLE_INTERNAL_BINDINGS=OFF` explicitly in
the cibuildwheel configuration. Release wheels therefore contain the stable API and
the minimum internal operational introspection needed to implement it, but not the
unsupported campaign machinery. Git checkouts retain the research registration and native tests for opt-in
validation; release source distributions omit those development-only files.

The option does not enable an accelerator and makes no accelerator support claim.
CUDA, HIP, and Metal availability remains governed by their dedicated build options
and the documented support matrix.

## CPU execution boundary

`wolfgang_core` is an independent CMake static library. `_wolfgang_core` links
that library and contains Python registration, parsing and result conversion.
The internal static target does not introduce a binary ABI or installation
promise. Standalone native contracts build with `WOLFGANG_BUILD_PYTHON=OFF` and
`WOLFGANG_BUILD_NATIVE_TESTS=ON`; no Python interpreter or nanobind is required.

CPU arithmetic, simplify, multiplication, commutation, grouping and expectation
release the GIL around native computation. Python parsing, buffer acquisition,
error translation and result conversion hold the GIL. Statevector buffers stay
pinned for the call; callers must not mutate an input array or change dispatch
environment variables concurrently with an operation. Device methods retain the
GIL pending a separate accelerator cache and concurrency review.

Coefficient export allocates one owning NumPy complex128 array and copies the
contiguous native buffer, avoiding per-coefficient Python objects. Returned arrays
remain independent of the native operator.
