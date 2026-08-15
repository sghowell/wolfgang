# Wolfgang Documentation Standards

Wolfgang documentation should meet the bar of a top-tier open source numerical library: accurate, concise, example-driven, and synchronized with implementation.

## Documentation Audiences

Wolfgang docs serve four audiences:

```text
users who want to install and use the Python package
scientific Python developers comparing semantics with Qiskit or OpenFermion
systems engineers working on C++/CUDA internals
agents implementing and validating the roadmap
```

Each document should make its audience clear by content and placement.

## Source-Of-Truth Layers

Use these layers:

```text
README.md: public entrypoint, install, quickstart, project scope
AGENTS.md: short agent operating map
docs/roadmap.md: phase order and release readiness
docs/plans/: implementation plans
docs/architecture/: semantic and backend contracts
docs/quality/: quality gates, harness, code standards, documentation standards
docs/benchmarks/: benchmark protocol and future benchmark reports
future docs/api/: generated or curated API reference
future docs/user/: user guides and examples
```

Do not duplicate detailed contracts across many files. Link to the source document instead.

## README Quality Bar

Before the first release, `README.md` should include:

```text
one-paragraph project purpose
non-goals and scope boundaries
installation instructions
minimal quickstart
Qiskit conversion example
OpenFermion conversion example
simplify and arithmetic example
expectation-value example
CUDA status and build instructions when CUDA support exists
testing and benchmark commands
links to API docs, roadmap, and contributing guidance
license
```

README claims must be evidence-based. Do not add badges for workflows, package indexes, CUDA support, or benchmarks until the corresponding artifact exists.

## User Guides

User-facing guides should be added as functionality lands.

Expected guides:

```text
docs/user/installation.md
docs/user/quickstart.md
docs/user/pauli_conventions.md
docs/user/qiskit_integration.md
docs/user/openfermion_integration.md
docs/user/simplify_and_arithmetic.md
docs/user/expectation_values.md
docs/user/cuda.md
docs/user/performance.md
```

Guides should:

```text
start with a working example
state version or backend requirements
explain endianness when labels, bitstrings, or sparse indices appear
show expected output for small examples
link to API reference for details
avoid performance claims without benchmark evidence
```

## API Documentation

Every public Python method should have API documentation covering:

```text
signature
purpose
parameters
returns
exceptions
ordering and endianness behavior
dtype behavior
small example when useful
```

Every public C++ header should document:

```text
ownership
input requirements
output ordering
complexity where relevant
error behavior
thread-safety or backend assumptions when relevant
```

CUDA-facing APIs should document:

```text
device ownership
stream and synchronization behavior
host-device transfer behavior
accepted array layouts
error and availability behavior
```

## Examples

Examples must be executable or trivially copyable into a Python session once the relevant phase has landed.

Examples should prefer:

```text
small operators
explicit expected labels or coefficients
deterministic seeds
short output snippets
clear dependency markers for Qiskit, OpenFermion, or CUDA
clear hardware target markers for CPU dispatch, CUDA architectures, and wheel platform claims
```

Do not show large benchmark outputs in user guides. Link to benchmark reports or summarize with caveats.

## Performance Documentation

Performance docs must follow `docs/benchmarks/protocol.md`.

Any performance section must include:

```text
benchmark command
git revision or release version
hardware and software environment
CPU backend, CPU feature set, compiler flags, and thread settings when CPU performance is discussed
CUDA toolkit, driver, device model, compute capability, and compiled architecture set when CUDA performance is discussed
dataset parameters
baseline
result
limitations
```

CUDA docs must clearly distinguish:

```text
transfer-inclusive timings
device-resident timings
CPU-faster regimes
CUDA-faster regimes
transfer-bound regimes
```

CPU optimization docs must clearly distinguish:

```text
portable scalar baseline
runtime-dispatched optimized paths
native-tuned local source builds
release-wheel behavior
```

## Documentation Style

Use direct, precise prose. Prefer concrete examples over broad claims.

Formatting rules:

```text
short sections with descriptive headings
code blocks for commands and examples
tables only when comparison is clearer than prose
consistent terminology: PauliSum, dense label, sparse list, canonical order, QWC, full commuting
ASCII unless a file already requires non-ASCII
```

Avoid:

```text
marketing language
unsupported superiority claims
ambiguous terms like fast or optimized without evidence
duplicated contract text that can drift
long unexplained formulas in user guides
```

## Documentation Tests And Checks

Phase 1 should introduce basic documentation checks:

```text
source-of-truth files exist
AGENTS.md links resolve
README.md links resolve
stale marker scan runs
```

Later phases should add:

```text
example execution tests
API docstring coverage checks
optional docs build with warnings as errors
link checking for local docs
benchmark report schema checks
```

## Drift Policy

Docs must change in the same slice as behavior when:

```text
public API changes
exceptions or guardrails change
ordering or endianness behavior changes
optional dependency behavior changes
benchmark protocol changes
CUDA availability or transfer behavior changes
```

If implementation contradicts docs, fix the implementation or update the source-of-truth doc before closing the phase. Do not leave drift as a chat-only note.
