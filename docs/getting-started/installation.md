# Installation

## CPU package

The ordinary Wolfgang installation is a CPU package with a portable scalar baseline. When a package-index release is available for your platform:

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
import wolfgang-quantum
from wolfgang-quantum import PauliSum

print("Wolfgang", wolfgang-quantum.__version__)
print(PauliSum.from_labels(["X"]).to_labels())
PY
```

## Requirements

- CPython in the range declared by `pyproject.toml`;
- a supported wheel platform, or CMake 3.24+ and a C++20 compiler for source builds;
- NumPy at runtime.

Qiskit and OpenFermion adapters are optional:

```bash
python -m pip install ".[qiskit]"
python -m pip install ".[openfermion]"
```

## Developer install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]" \
  --config-settings=cmake.define.FASTPAULI_ENABLE_INTERNAL_BINDINGS=ON
python scripts/validate.py
```

## Accelerator source builds

Accelerator builds are opt-in CMake configurations, not automatic runtime downloads. Read the [accelerator overview](../accelerators/overview.md) and the [support matrix](../release/support_matrix.md) before building. Exact toolkit, architecture, compiler, runtime, and hardware evidence determines support.

Never enable `FASTPAULI_ENABLE_NATIVE=ON` for a portable release wheel. Never infer broad architecture support from one successful GPU.

## Troubleshooting

Capture the complete configure output and the native module's `_build_info()` report. Report the Wolfgang revision, Python, compiler, OS, architecture, and accelerator runtime after removing usernames, addresses, tokens, hostnames, and private paths. See the [support policy](https://github.com/sghowell/Wolfgang/blob/main/SUPPORT.md).
