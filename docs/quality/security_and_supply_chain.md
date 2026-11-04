# Wolfgang Security And Supply Chain Standards

Wolfgang is a Python package backed by C++ and CUDA. Its security and reliability risks include memory safety, unsafe native builds, dependency drift, malformed user input, and the integrity of distributed binaries.

## Security Principles

```text
fail closed on invalid input
guard every large allocation
avoid undefined behavior
make dependency and build assumptions explicit
keep optional dependencies optional
prefer reproducible release evidence
```

Security work must be practical and fit the phase. Early phases should establish checks and policy; later phases should add sanitizers, fuzzers, and release provenance as the code they protect lands.

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
CUDA-Q is optional
SciPy is optional and only imported by PauliSum.to_matrix(sparse=True)
CUDA runtime is optional and only required for CUDA builds or CUDA execution
```

Build and test dependencies must be declared in project metadata once the scaffold exists.

Hardware-specific build dependencies such as oneTBB, CUDA, and architecture-specific compiler support must stay optional unless the release artifact explicitly targets them.

Release CPU wheels explicitly target oneTBB. The wheel workflow builds oneTBB
`v2022.3.0` from its pinned commit (`f1862f38f83568d96e814e469ab61f88336cc595`,
checked after cloning; see `scripts/build_onetbb.sh`), bundles only the core
`tbb` runtime through auditwheel or delocate, and ships oneTBB's Apache-2.0
`LICENSE.txt` and `third-party-programs.txt` as license files in the wheel
metadata. The build fails if the cloned notices differ from the committed
copies in `third_party/onetbb/`. Every wheel also compiles in nanobind's
support library and the tsl-robin-map hash table nanobind vendors;
`pyproject.toml` pins nanobind exactly (`nanobind==3.1.0` in
`[build-system] requires`, and the same pin in CI), and the wheels ship
nanobind's BSD-3-Clause and robin-map's MIT license from
`third_party/nanobind/`. The project license expression is
`MIT AND Apache-2.0 AND BSD-3-Clause` (Wolfgang and robin-map; DLPack and
oneTBB; nanobind and oneTBB's ITT Notify). Updating oneTBB or nanobind means
changing the pins and the committed notices together and re-running the wheel
smoke tests.
Each kind of release artifact has its own CycloneDX 1.7 SBOM, written by
`scripts/sbom_bundled_components.py` from the Python dependencies
`cyclonedx-py` lists: `wolfgang-quantum-sdist.cdx.json`,
`wolfgang-quantum-manylinux-x86_64.cdx.json` and
`wolfgang-quantum-macos-arm64.cdx.json`. Each names wolfgang-quantum (version,
license expression and repository from `pyproject.toml`) as
`metadata.component`; lists the Python dependencies as external components
with the version range `pyproject.toml` accepts, in vers syntax, since no
wheel bundles them (external components with ranges are why the documents use
CycloneDX 1.7); and lists the native code that artifact ships. The sdist
carries the vendored DLPack header source (version from `dlpack.h`, with its
SHA-256; see `third_party/dlpack/README.md`). Wheels compile that header in,
compile in nanobind and robin-map (versions from `third_party/nanobind/README.md`,
checked against the pin), and bundle the oneTBB runtime (tag and commit from
`scripts/build_onetbb.sh`), licensed `Apache-2.0 AND BSD-3-Clause` on x86_64,
where the BSD-3-Clause ITT Notify code is compiled in, and `Apache-2.0` on
arm64. The release workflow
validates every document against the CycloneDX schema and attests each with
`actions/attest` against the files it describes, beside the build provenance of
the whole payload. The artifact kinds follow the wheel build matrix; a new
wheel platform needs its own kind, SBOM and attestation step. Source builds
keep oneTBB optional.

To check that a downloaded artifact was signed by this repository's release
workflow, and to read the SBOM attested for it:

```bash
gh attestation verify wolfgang_quantum-0.2.3-cp312-cp312-macosx_11_0_arm64.whl \
  --repo sghowell/wolfgang \
  --signer-workflow sghowell/wolfgang/.github/workflows/release-wheelhouse.yml \
  --predicate-type https://cyclonedx.org/bom \
  --format json --jq '.[0].verificationResult.statement.predicate'
```

`--repo` alone accepts any workflow in the repository. Adding
`--source-ref refs/tags/v0.2.3` also requires the published tag; dry runs from
branches do not meet it.

Without `--predicate-type`, `gh attestation verify` checks the build
provenance (`https://slsa.dev/provenance/v1`).

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

`SECURITY.md` at the repository root tells reporters how to report a
vulnerability privately. Handle a report by:

```text
reproducing privately when possible
adding a regression test
fixing with minimal scope
documenting affected versions once releases exist
cutting a patch release for released vulnerable code
```

This document is the internal security standard; `SECURITY.md` is the public reporting policy.

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
