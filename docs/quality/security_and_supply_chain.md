# Wolfgang Security And Supply Chain Standards

Wolfgang is a C++/CUDA-backed Python package. Security and reliability risks include memory safety, unsafe native builds, dependency drift, malformed user inputs, and binary distribution integrity.

## Security Principles

```text
fail closed on invalid input
guard every large allocation
avoid undefined behavior
make dependency and build assumptions explicit
keep optional dependencies optional
prefer reproducible release evidence
```

Security work must be practical and phase-appropriate. Early phases should establish checks and policy; later phases should add sanitizers, fuzzers, and release provenance as implementation surfaces exist.

## Native Code Safety

C++ and CUDA code must protect against:

```text
integer overflow in size calculations
out-of-bounds packed-word access
invalid final-word high bits
invalid qubit indices
shape and dtype mismatches
unsafe host-device copies
CUDA device mismatch
use-after-free through device or Python bindings
```

Required practices:

```text
centralized size-multiplication helpers before large allocations
explicit validation at public boundaries
RAII ownership for CPU resources
explicit owning device wrappers for CUDA resources
common CUDA error-checking helper once CUDA lands
tests for guardrails before allocation
```

## Sanitizers And Dynamic Analysis

Sanitizer support should be added as soon as there is meaningful native code.

Target checks:

```text
AddressSanitizer for C++ CPU tests
UndefinedBehaviorSanitizer for C++ CPU tests
ThreadSanitizer only if threading bugs become plausible and tooling supports the stack
cuda-memcheck or compute-sanitizer for CUDA kernels when CUDA lands
```

Sanitizer jobs may be separate from fast CI if runtime is high, but release candidates should have sanitizer evidence for relevant native code.

CPU sanitizer and CUDA sanitizer expectations must line up with the ladders in `docs/architecture/hardware_targets_and_testing.md`.

## Fuzz And Property Testing

Wolfgang should use property tests early and fuzzing where it adds value.

Property-test targets:

```text
dense label parsing and export
sparse list parsing and export
simplify idempotence
multiplication associativity for small operators
commutation symmetry
grouping validity
statevector expectation against dense oracle
```

Fuzz candidates after C++ parsing exists:

```text
dense label parser
sparse-list input conversion
packed representation invariant checks
```

Fuzz findings must be reduced to deterministic regression tests.

## Dependency Policy

Dependencies must be justified by phase need.

Runtime dependencies:

```text
NumPy is required
Qiskit is optional
OpenFermion is optional
CUDA runtime is optional and only required for CUDA builds or CUDA execution
```

Build and test dependencies must be declared in project metadata once the scaffold exists.

Hardware-specific build dependencies such as oneTBB, CUDA, and architecture-specific compiler support must stay optional unless the release artifact explicitly targets them.

Rules:

```text
do not import optional dependencies at package import time
do not add broad dependencies for small helper tasks
pin lower bounds only after validation
document optional dependency extras
run adapter tests in an environment where optional dependencies are installed
```

## Build And Release Integrity

Release artifacts should be built from clean source states.

Release candidates require:

```text
clean git status except ignored local files
validated source distribution
validated CPU wheel build
validated CUDA source build on at least one CUDA environment once CUDA support lands
recorded git revision
recorded validation commands
recorded benchmark evidence for performance claims
```

Release automation uses:

```text
artifact checksums
immutable full-SHA GitHub Action pins
read-only default workflow permissions
job-scoped OIDC only for artifact attestation and trusted publishing
exact release-tag-to-project-version binding
dependency auditing and weekly Dependabot updates
CycloneDX software bills of materials
GitHub artifact attestations
```

Future hardening may add signed tags. Do not claim signing or other provenance
features before they are implemented.

## Secrets And Credentials

Wolfgang should not require credentials for normal build, test, or benchmark workflows.

Rules:

```text
do not commit secrets
do not require private tokens for local validation
do not print environment variables wholesale in CI logs
keep release credentials in GitHub secrets or equivalent managed secret storage
```

## Vulnerability Handling

Until a formal security policy is added, handle security issues by:

```text
reproducing privately when possible
adding a regression test
fixing with minimal scope
documenting affected versions once releases exist
cutting a patch release for released vulnerable code
```

Phase 1 should add a basic `SECURITY.md` only if the repo starts accepting external reports before the first release. Before that, this document is the internal security standard.

## Security Review Checklist

Before merging native or packaging changes:

```text
are all public inputs validated?
can size calculations overflow?
can malformed labels or indices reach unchecked memory access?
does CPU-only build avoid CUDA requirements?
are optional dependencies still optional?
does CI avoid leaking secrets?
does any release or benchmark claim require stronger evidence?
has a security-focused review been completed for high-risk native, CUDA, packaging, or release changes?
```
