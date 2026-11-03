"""Metal hybrid simplify (GPU key sort, CPU float64 reduction) equals CPU simplify.

From the hybrid threshold, one-word Metal simplify sorts on the GPU and sums
duplicate runs in input order on the CPU, so results are bit-identical to CPU
simplify. The selector WOLFGANG_METAL_BENCH_SIMPLIFY=host_bridge keeps the
previous host bridge for comparison, and =hybrid runs the hybrid at any size so
small and odd sizes exercise the sort network's tile and sentinel edges.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import wolfgang_quantum
import wolfgang_quantum._wolfgang_core as core

HYBRID_MIN_TERMS = 12_288
SELECTOR = "WOLFGANG_METAL_BENCH_SIMPLIFY"


def _require_metal() -> None:
    status = dict(core._metal_status())
    if not status.get("built") or not status.get("runtime_available"):
        pytest.skip(str(status.get("skip_reason", "Metal is unavailable")))


def _operator(num_qubits: int, num_terms: int, duplicates: float, seed: int) -> wolfgang_quantum.PauliSum:
    rng = np.random.default_rng(seed)
    unique = max(1, int(num_terms * (1.0 - duplicates)))
    letters = np.asarray(list("IXYZ"), dtype="<U1")
    chars = letters[rng.integers(0, 4, size=(unique, num_qubits))]
    base = np.ascontiguousarray(chars).view(f"<U{num_qubits}").ravel()
    labels = np.concatenate([base, base[rng.integers(0, unique, num_terms - unique)]]).tolist()
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return wolfgang_quantum.PauliSum.from_labels(labels, coeffs)


def _with_negated_zeros(op: wolfgang_quantum.PauliSum) -> wolfgang_quantum.PauliSum:
    # Zero the imaginary part of every third coefficient, then negate: those
    # components become -0.0, which plain -op of random coefficients never has.
    x, z, coeffs = (array.copy() for array in op.to_arrays())
    coeffs[::3] = coeffs[::3].real
    negated = -wolfgang_quantum.PauliSum.from_arrays(x, z, coeffs, op.num_qubits)
    assert np.signbit(negated.to_arrays()[2].imag[::3]).all()
    return negated


def _hybrid_runs() -> int:
    hook = getattr(core, "_metal_hybrid_simplify_runs_for_testing", None)
    if hook is None:
        pytest.skip("internal bindings are not built")
    return int(hook())


def _assert_bit_identical(actual, expected) -> None:
    # == treats 0.0 and -0.0 as equal; compare raw coefficient bits too.
    assert actual == expected
    assert np.array_equal(actual.to_arrays()[2].view(np.uint64), expected.to_arrays()[2].view(np.uint64))


@pytest.mark.parametrize("num_qubits", [5, 33, 64])
@pytest.mark.parametrize("duplicates", [0.0, 0.9])
def test_hybrid_simplify_is_bit_identical_to_cpu(num_qubits: int, duplicates: float) -> None:
    _require_metal()
    op = _operator(num_qubits, HYBRID_MIN_TERMS + 17, duplicates, seed=num_qubits)
    device = op.to_device(backend="metal")
    for kwargs in ({}, {"atol": 0.5}, {"atol": 0.0, "rtol": 0.2}):
        _assert_bit_identical(device.simplify(**kwargs).to_host(), op.simplify(**kwargs))
    negated = _with_negated_zeros(op)
    _assert_bit_identical(negated.to_device(backend="metal").simplify().to_host(), negated.simplify())


def test_hybrid_simplify_matches_host_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_metal()
    op = _operator(40, HYBRID_MIN_TERMS, 0.5, seed=7)
    device = op.to_device(backend="metal")
    hybrid = device.simplify().to_host()
    monkeypatch.setenv("WOLFGANG_METAL_BENCH_SIMPLIFY", "host_bridge")
    assert device.simplify().to_host() == hybrid


def test_hybrid_simplify_errors_match_cpu() -> None:
    _require_metal()
    base = _operator(10, HYBRID_MIN_TERMS, 0.0, seed=3)
    x, z, coeffs = (array.copy() for array in base.to_arrays())
    coeffs[5] = complex(math.inf, 0.0)
    non_finite = wolfgang_quantum.PauliSum.from_arrays(x, z, coeffs, 10)
    with pytest.raises(ValueError, match="finite"):
        non_finite.simplify()
    with pytest.raises(ValueError, match="finite"):
        non_finite.to_device(backend="metal").simplify()

    coeffs[:] = 1.7e308
    z[:] = 0
    x[:] = 1  # every term is the same Pauli, so the sum overflows
    overflow = wolfgang_quantum.PauliSum.from_arrays(x, z, coeffs, 10)
    with pytest.raises(OverflowError):
        overflow.simplify()
    with pytest.raises(OverflowError):
        overflow.to_device(backend="metal").simplify()


def test_hybrid_simplify_full_cancellation() -> None:
    _require_metal()
    op = _operator(20, HYBRID_MIN_TERMS // 2, 0.0, seed=11)
    cancelled = (op + (-op)).to_device(backend="metal").simplify()
    assert cancelled.num_terms == 0
    assert cancelled.num_qubits == 20


@pytest.mark.parametrize("num_terms", [1, 2, 3, 1023, 1024, 1025, 2047, 2049, 5000, 70_001])
def test_forced_hybrid_matches_cpu_at_tile_edges(monkeypatch: pytest.MonkeyPatch, num_terms: int) -> None:
    _require_metal()
    monkeypatch.setenv(SELECTOR, "hybrid")
    for duplicates in (0.0, 0.7):
        op = _operator(64, num_terms, duplicates, seed=num_terms)
        _assert_bit_identical(op.to_device(backend="metal").simplify().to_host(), op.simplify())


def test_forced_hybrid_orders_all_y_key_before_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    # An all-Y 64-qubit term packs to x = z = 2**64 - 1, the padding sentinel's
    # key; the input-index tie-break must still keep it inside the output.
    _require_metal()
    monkeypatch.setenv(SELECTOR, "hybrid")
    labels = ["Y" * 64, "X" * 64, "Y" * 64, "Z" * 64] * 300 + ["Y" * 64]
    coeffs = np.arange(1, len(labels) + 1, dtype=np.float64) * (1.0 + 0.5j)
    op = wolfgang_quantum.PauliSum.from_labels(labels, coeffs)
    _assert_bit_identical(op.to_device(backend="metal").simplify().to_host(), op.simplify())


@pytest.mark.parametrize("num_terms", [262_145, 300_007])
def test_hybrid_simplify_just_past_a_power_of_two(num_terms: int) -> None:
    _require_metal()
    op = _operator(48, num_terms, 0.5, seed=num_terms)
    _assert_bit_identical(op.to_device(backend="metal").simplify().to_host(), op.simplify())


def test_public_simplify_takes_the_hybrid_from_its_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both paths return identical results, so count hybrid runs to prove which
    # path each call took.
    _require_metal()
    monkeypatch.delenv(SELECTOR, raising=False)
    at_threshold = _operator(40, HYBRID_MIN_TERMS, 0.3, seed=1).to_device(backend="metal")
    below = _operator(40, HYBRID_MIN_TERMS - 1, 0.3, seed=2).to_device(backend="metal")
    wide = _operator(65, HYBRID_MIN_TERMS, 0.3, seed=3).to_device(backend="metal")
    start = _hybrid_runs()
    at_threshold.simplify()
    assert _hybrid_runs() == start + 1
    below.simplify()
    wide.simplify()  # two packed words keep the host bridge
    assert _hybrid_runs() == start + 1
    monkeypatch.setenv(SELECTOR, "host_bridge")
    at_threshold.simplify()
    assert _hybrid_runs() == start + 1
    monkeypatch.setenv(SELECTOR, "hybrid")
    below.simplify()
    assert _hybrid_runs() == start + 2


@pytest.mark.parametrize("num_terms", [7, HYBRID_MIN_TERMS + 5])
def test_hybrid_preserves_signed_zeros(monkeypatch: pytest.MonkeyPatch, num_terms: int) -> None:
    # Single-term runs keep a -0.0 component; -0.0 + -0.0 duplicate runs stay
    # -0.0 because each run is seeded with its first coefficient.
    _require_metal()
    monkeypatch.setenv(SELECTOR, "hybrid")
    rng = np.random.default_rng(num_terms)
    unique = _operator(30, num_terms, 0.0, seed=num_terms)
    x, z, _ = unique.to_arrays()
    x = np.concatenate([x, x[: num_terms // 2]])
    z = np.concatenate([z, z[: num_terms // 2]])
    imag = rng.normal(size=len(x))
    coeffs = np.where(np.arange(len(x)) % 2 == 0, -0.0, 1.5) + 1j * imag
    op = wolfgang_quantum.PauliSum.from_arrays(x, z, coeffs, 30)
    expected = op.simplify(atol=0.0)
    assert np.signbit(expected.to_arrays()[2].real).any()
    _assert_bit_identical(op.to_device(backend="metal").simplify(atol=0.0).to_host(), expected)


def test_simplify_kernels_are_embedded_at_build_time() -> None:
    # Installed builds must not read the source tree: the kernel source is
    # compiled into the extension from a CMake-generated include.
    root = Path(__file__).resolve().parents[1]
    host = (root / "src/metal/simplify_metal.mm").read_text(encoding="utf-8")
    assert '#include "metal/simplify_kernel_source.inc"' in host
    assert "ifstream" not in host and "__FILE__" not in host
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "src/metal/kernels/simplify.metal" in cmake
    assert "simplify_kernel_source.inc" in cmake
