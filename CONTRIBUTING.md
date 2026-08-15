# Contributing to Wolfgang

Wolfgang welcomes correctness fixes, documentation, portability work, benchmark improvements, and carefully evidenced kernel optimization. Human- and agent-authored contributions follow the same standards: reproducible evidence, explicit uncertainty, reviewable scope, and no fabricated runtime claims.

By participating, you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

Wolfgang requires Python 3.10+, a C++20 compiler, CMake 3.24+, and Git. The default developer path is CPU-only and does not require an accelerator toolkit.

```bash
git clone https://github.com/sghowell/Wolfgang.git
cd Wolfgang
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]" \
  --config-settings=cmake.define.WOLFGANG_ENABLE_INTERNAL_BINDINGS=ON
python scripts/validate.py
```

For faster dependency setup, `uv sync --all-extras` is also supported when `uv` is available. Source-build accelerator commands and exact evidence requirements are documented under `docs/accelerators/` and `docs/architecture/`.

## Development workflow

1. Open or reference an issue for substantial changes.
2. Create a focused feature branch from current `main`.
3. Read the relevant public contract and architecture document.
4. Write a failing test before behavioral production code.
5. Make the smallest implementation pass.
6. Run focused tests, then `python scripts/validate.py`.
7. Update user docs, changelog, support matrix, and benchmark evidence when affected.
8. Open a pull request using the repository template.

`AGENTS.md` contains additional navigation for automated coding agents; it is not required reading for ordinary users.

The full review checklist is in [`docs/quality/code_review.md`](docs/quality/code_review.md).

## Correctness and native safety

Public boundaries must validate dtype, shape, indices, packed-word invariants, allocation arithmetic, device ownership, and lifetime. Native or interop changes should include malformed-input and moved-from/lifetime cases. CPU semantics are the correctness oracle for optimized paths unless a more authoritative independent oracle is documented.

## Performance evidence

A speed claim must identify:

- operation and semantic mapping;
- deterministic dataset and seed;
- transfer-inclusive, device-resident, allocation, reuse, and synchronization boundary;
- warmup and repeat policy;
- CPU/GPU, compiler, runtime, driver, and relevant build options;
- correctness comparison;
- raw measurements or a sanitized machine-readable summary;
- uncertainty and regimes where the optimization does not win.

Do not commit raw profiler databases, host inventories, SSH targets, arbitrary environment dumps, private paths, or cloud identifiers. Follow `docs/quality/public_artifact_policy.md` when it exists and run the public-artifact audit before submission.

## Style

- C++: C++20, RAII, checked arithmetic, explicit ownership, no hidden synchronization changes.
- Python: clear public types, Ruff formatting/lint policy, portable paths, no shell-string command construction.
- Docs: user-first, evidence-backed, concise at the landing page, detailed in the appropriate guide or report.
- Commits: focused Conventional Commit messages such as `fix:`, `feat:`, `perf:`, `docs:`, `test:`, or `ci:`.

## Pull request checklist

- [ ] The change is focused and its public contract is clear.
- [ ] Behavioral code was preceded by a failing test.
- [ ] Focused and full relevant tests pass.
- [ ] CPU-only import/build remains valid.
- [ ] Accelerator claims match available runtime evidence.
- [ ] No private infrastructure or credential material is included.
- [ ] Documentation and changelog are synchronized.
- [ ] Release/package surfaces remain coherent.

## Reporting security issues

Do not open a public issue for vulnerabilities. Follow [SECURITY.md](SECURITY.md).
