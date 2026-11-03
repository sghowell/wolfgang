# Installation

## CPU package

The default Wolfgang installation is a CPU package with a portable scalar baseline. When a package-index release exists for your platform, install it with pip:

```bash
python -m pip install wolfgang-quantum
```

Before package-index publication, install from a tagged source archive or a local checkout:

```bash
git clone https://github.com/sghowell/Wolfgang.git
cd Wolfgang
python -m pip install .
```

Verify the installation:

```bash
python - <<'PY'
import wolfgang_quantum
from wolfgang_quantum import PauliSum

print("Wolfgang", wolfgang_quantum.__version__)
print(PauliSum.from_labels(["X"]).to_labels())
PY
```

## Requirements

- CPython in the range declared by `pyproject.toml`;
- a supported wheel platform, or CMake 3.24+ and a C++20 compiler for source builds;
- NumPy at runtime.

The Qiskit, OpenFermion and CUDA-Q adapters are optional extras:

```bash
python -m pip install ".[qiskit]"
python -m pip install ".[openfermion]"
python -m pip install ".[cudaq]"        # Linux x86_64/aarch64 only; CUDA-Q 0.16 needs Python 3.11+
```

## Developer install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]" \
  --config-settings=cmake.define.WOLFGANG_ENABLE_INTERNAL_BINDINGS=ON
python scripts/validate.py --profile core
```

The `all` profile adds lint, type, spelling and strict docs checks. Install the
quality tools pinned in `.github/workflows/quality.yml` and the `docs`, `qiskit`,
and `openfermion` extras, then run `python scripts/validate.py --profile all`.

## Accelerator source builds

Accelerator builds are opt-in CMake configurations, not automatic runtime downloads. Read the [accelerator overview](../accelerators/overview.md) and the [support matrix](../release/support_matrix.md) before building. Support is decided by evidence for the exact toolkit, architecture, compiler, runtime, and hardware.

Never enable `WOLFGANG_ENABLE_NATIVE=ON` for a portable release wheel. Never infer broad architecture support from one successful GPU.

## Troubleshooting

When a build or import fails, capture the complete configure output and the native module's `_build_info()` report. Include the Wolfgang revision, Python, compiler, OS, architecture, and accelerator runtime in your report, and remove usernames, addresses, tokens, hostnames, and private paths before sending it. See the [support policy](https://github.com/sghowell/Wolfgang/blob/main/SUPPORT.md).
