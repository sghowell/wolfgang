# Wolfgang Code Standards

Wolfgang code should read like it was written by an expert numerical systems team: explicit invariants, predictable ownership, measured performance, and no ambiguity around correctness.

This document sets the coding standard for C++, CUDA, Python, tests, and comments.

## Global Code Principles

```text
correctness before optimization
measurement before performance claims
simple ownership before clever abstractions
small focused files before large mixed-responsibility files
structured parsers and APIs before ad hoc string handling
deterministic behavior before incidental container iteration
explicit guardrails before large allocations
```

Every nontrivial code path should make its preconditions, invariants, and failure behavior obvious either in the API contract, the type shape, or a concise local comment.

## C++ Standard

Wolfgang uses C++20 as the baseline.

Preferred C++ practices:

```text
use RAII for ownership
use std::span for non-owning contiguous views
use std::size_t for sizes and indices unless an external API requires another type
use fixed-width integers for packed data, especially std::uint64_t
use std::complex<double> for initial host coefficients
use std::popcount for scalar popcount paths
use constexpr helpers for small phase tables and pure bit operations
keep portable scalar code free of target-specific instruction requirements
```

Avoid:

```text
raw owning pointers in CPU code
global mutable state
exceptions for internal control flow
implicit narrowing conversions
undefined behavior for speed
silent overflow in size calculations
```

All allocation-size calculations must check overflow before multiplying dimensions such as:

```text
num_terms * words
lhs_terms * rhs_terms
statevector_length * sizeof(complex)
```

## C++ API Design

Public native APIs should be small and stable. Prefer free functions over broad mutable classes for core algorithms:

```text
from_labels
to_labels
simplify
multiply
expectation_statevector
```

API boundaries should:

```text
accept spans or lightweight views where practical
return owning values for new operators
avoid hidden global configuration
surface guardrails as explicit parameters
translate invalid user input to clear Python exceptions at the binding boundary
```

Header placement should preserve the public/private boundary:

```text
put documented user-facing C++ APIs in include/wolfgang
put native implementation helpers in src/detail
do not expose packed bit helpers, parse/export helpers, phase helpers, or allocation guard helpers as public headers unless users need them directly
promoting a helper header to include/wolfgang requires API-stability review and documentation
```

## Memory Layout And Performance

Use structure-of-arrays layout for Pauli data:

```text
x: contiguous uint64 words
z: contiguous uint64 words
coeffs: contiguous complex values
```

Performance-sensitive code should:

```text
specialize hot paths for words == 1, words == 2, and small fixed word counts when benchmarks justify it
keep scalar baseline implementations available
avoid per-term heap allocations in inner loops
avoid Python callbacks in C++ hot paths
make data movement explicit
preserve deterministic output order
keep release-wheel defaults compatible with docs/architecture/hardware_targets_and_testing.md
```

Do not introduce oneTBB, SIMD, or CUDA variants without:

```text
scalar semantic tests
scalar benchmark baseline
optimized-path equivalence tests
benchmark evidence for the optimized path
```

Do not compile release wheels with `-march=native` or equivalent native CPU tuning. Native tuning is allowed only for local source builds with explicit opt-in.

## CPU Dispatch Standards

CPU feature dispatch code should:

```text
make scalar the always-available fallback
detect required feature groups before executing specialized code
allow scalar and optimized paths to be forced in tests
report the active path in benchmark metadata
fail clearly when a forced path is unavailable
avoid hidden global state that makes tests order-dependent
```

SIMD implementations must document the exact feature group they require. AVX-512 paths must not assume every AVX-512 CPU supports the same instruction subsets. ARM NEON or SVE paths must be guarded and forced independently from x86 SIMD paths, and Apple Silicon CPU results must be reported separately from x86_64 results.

## CUDA Code Standards

CUDA is required, but CUDA kernels are introduced only after CPU semantics and benchmark gates are stable.

CUDA code should:

```text
keep host/device ownership explicit
avoid unified memory in the first CUDA milestone
check every CUDA API call through a common error helper
validate device ordinal and array-device compatibility
synchronize before returning to Python in the initial API
guard every device allocation with overflow checks
use CUB or Thrust for sort/reduce/scan/compact before custom kernels
```

CUDA kernels should:

```text
minimize divergent control flow in hot loops
use coalesced memory access where the representation permits it
make reduction strategy explicit
avoid hidden host-device transfers inside benchmarked device-resident paths
document assumptions about block size, occupancy, and shared memory use when they affect correctness or performance
document the architecture features required by kernels when they are not available across the full CUDA target set
```

Do not write custom GPU sort/reduce in the initial CUDA milestone.

ROCm/HIP and Apple Metal are post-CUDA accelerator targets. Do not mix their
build flags, runtime checks, memory ownership rules, or benchmark claims into
CUDA code paths. MPS and MPSGraph are optional Apple implementation adjuncts or
external baselines, not Wolfgang backend identities.

## Python Code Standards

Python code is used for package glue, adapters, tests, validation, and benchmarks.

Python code should:

```text
use type annotations for public functions
keep optional imports inside adapter methods or adapter modules
raise ImportError with installation hints for missing optional dependencies
avoid import-time work that can fail due to optional dependencies
prefer pathlib for filesystem paths
keep scripts deterministic and explicit about commands they run
```

Validation scripts should print the check name before running it and return a nonzero exit code on failure.

## Comments And Docstrings

Comments must explain why, invariants, edge cases, and performance assumptions. They must not narrate obvious syntax.

Good comments:

```text
explain bit-order or external-library conventions
document non-obvious phase formulas
identify overflow or allocation guardrails
state why a CUDA synchronization is required
record benchmark-backed threshold choices
explain deterministic tie-breaking
```

Avoid comments like:

```text
// increment i
// loop over terms
// set value
```

Public Python methods need docstrings once they are exposed beyond the minimal scaffold. Docstrings should include:

```text
one-sentence purpose
parameter meanings and accepted types
return value
exceptions
endianness or ordering behavior when relevant
small example for nontrivial methods
```

C++ public headers should document:

```text
ownership expectations
input shape requirements
output ordering
exceptions or error behavior
complexity when important
```

CUDA kernels and device helpers should document:

```text
input layout
thread-to-data mapping
reduction scheme
synchronization assumptions
known limits
```

## Error Handling

Invalid user input must produce actionable errors:

```text
what was invalid
which argument was invalid
what range or shape was expected
```

Bindings must translate C++ exceptions into Python exception types documented in `docs/architecture/semantic_contracts.md`.

Internal errors should fail closed. Do not continue after failed allocation, failed CUDA API calls, impossible shape checks, or violated representation invariants.

## Reliability And Determinism

Wolfgang must be deterministic by default:

```text
canonical ordering is stable
grouping tie-breakers are stable
property tests use reproducible seeds
benchmarks record seeds and dataset parameters
parallel paths produce the same result ordering as scalar paths
```

Do not use unordered container iteration for externally visible output ordering unless the output is explicitly re-sorted before return.

## Review Checklist

Before code is merged:

```text
does the code preserve semantic contracts?
are ownership and lifetimes obvious?
are allocation-size calculations overflow-safe?
are public APIs documented?
are comments useful and not noisy?
is the scalar path tested before optimized paths?
does the optimized path have equivalence tests?
does any performance claim cite benchmark evidence?
does the code keep CPU-only builds independent of CUDA?
has the required review stage in docs/quality/code_review.md completed?
```

Phase 1 should add mechanical checks for the parts of this standard that can be checked immediately. Later phases should add stricter format, lint, static-analysis, and compile-warning gates as tools are introduced.
