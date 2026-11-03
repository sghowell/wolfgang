"""The multi-launch CUDA and HIP simplify keeps the CPU simplify contract.

Operators the fused kernel does not take (more than 1,024 terms, or more than
four packed words) run the multi-launch pipeline in
`src/detail/device_simplify_pipeline.cuh`. Every test here also sets
WOLFGANG_<BACKEND>_BENCH_SMALL_SIMPLIFY=off, so small inputs, down to two
terms, reach the same pipeline. The pipeline keeps its counts and status flags
on the device until one synchronization, so these tests pin what that must not
change: canonical output, the drop rule (separately rounded, in half units when
the threshold overflows), ValueError for non-finite input taking precedence over
OverflowError, and no state carried from one call (or thread) into the next.
Duplicate runs are summed in a tree order, so coefficients match to rounding.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import wolfgang_quantum
import wolfgang_quantum._wolfgang_core as core

BACKENDS = ("cuda", "hip")
# One and two words take packed sort keys (<= 32 qubits in one 64-bit key);
# wider operators sort term indices.
WIDTHS = [1, 5, 31, 32, 33, 64, 65, 128, 129, 300]


def _require(backend: str) -> None:
    status = getattr(core, f"_{backend}_status", None)
    if status is None:
        pytest.skip(f"{backend} status is not exposed")
    report = dict(status())
    if not report.get("built") or not report.get("runtime_available"):
        pytest.skip(str(report.get("skip_reason", f"{backend} is unavailable")))


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> str:
    backend = request.param
    _require(backend)
    monkeypatch.setenv(f"WOLFGANG_{backend.upper()}_BENCH_SMALL_SIMPLIFY", "off")
    return backend


def _labels(num_qubits: int, count: int, rng: np.random.Generator) -> list[str]:
    letters = np.asarray(list("IXYZ"), dtype="<U1")
    chars = letters[rng.integers(0, 4, size=(count, num_qubits))]
    return np.ascontiguousarray(chars).view(f"<U{num_qubits}").ravel().tolist()


def _operator(num_qubits: int, num_terms: int, duplicates: float, seed: int) -> wolfgang_quantum.PauliSum:
    rng = np.random.default_rng(seed)
    unique = max(1, int(num_terms * (1.0 - duplicates)))
    base = _labels(num_qubits, unique, rng)
    labels = base + [base[i] for i in rng.integers(0, unique, num_terms - unique)]
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return wolfgang_quantum.PauliSum.from_labels(labels, coeffs)


def _filler(num_qubits: int, count: int, seed: int) -> list[str]:
    # Distinct labels with a fixed first letter, so they never equal a test term
    # that starts with another letter.
    rng = np.random.default_rng(seed)
    labels = sorted({"I" + label[1:] for label in _labels(num_qubits, count * 2, rng)})
    return labels[:count]


def _assert_matches_cpu(backend: str, op: wolfgang_quantum.PauliSum, **kwargs: float) -> None:
    expected = op.simplify(**kwargs)
    actual = op.to_device(backend=backend).simplify(**kwargs).to_host()
    assert actual.num_qubits == expected.num_qubits
    assert list(actual.to_labels()[0]) == list(expected.to_labels()[0])
    np.testing.assert_allclose(actual.to_arrays()[2], expected.to_arrays()[2], rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("pipeline", BACKENDS, indirect=True)
@pytest.mark.parametrize("num_qubits", WIDTHS)
@pytest.mark.parametrize(("num_terms", "duplicates"), [(1, 0.0), (2, 0.0), (2, 0.5), (3, 0.6), (2000, 0.0), (2000, 0.9)])
def test_pipeline_matches_cpu(pipeline: str, num_qubits: int, num_terms: int, duplicates: float) -> None:
    op = _operator(num_qubits, num_terms, duplicates, seed=num_qubits * 31 + num_terms)
    for kwargs in ({}, {"atol": 0.5}, {"atol": 0.0, "rtol": 0.2}):
        _assert_matches_cpu(pipeline, op, **kwargs)


@pytest.mark.parametrize("pipeline", BACKENDS, indirect=True)
@pytest.mark.parametrize("num_qubits", [1, 24, 40, 100, 130])
def test_pipeline_full_cancellation_and_single_survivor(pipeline: str, num_qubits: int) -> None:
    rng = np.random.default_rng(num_qubits)
    labels = _labels(num_qubits, 1500, rng)
    coeffs = rng.normal(size=1500) + 1j * rng.normal(size=1500)
    cancelling = wolfgang_quantum.PauliSum.from_labels(labels + labels, np.concatenate([coeffs, -coeffs]))
    assert cancelling.to_device(backend=pipeline).simplify().to_host().num_terms == 0
    survivor = wolfgang_quantum.PauliSum.from_labels(
        labels + labels + [labels[0]], np.concatenate([coeffs, -coeffs, [2.5 - 1.0j]])
    )
    _assert_matches_cpu(pipeline, survivor)
    assert survivor.to_device(backend=pipeline).simplify().to_host().num_terms == 1


@pytest.mark.parametrize("pipeline", BACKENDS, indirect=True)
@pytest.mark.parametrize("num_qubits", [24, 40, 100, 130])
def test_pipeline_raises_on_duplicate_sum_overflow(pipeline: str, num_qubits: int) -> None:
    label = "X" * num_qubits
    filler = _filler(num_qubits, 1200, seed=num_qubits)
    op = wolfgang_quantum.PauliSum.from_labels(
        filler + [label, label], np.concatenate([np.ones(len(filler)), [1.7e308, 1.7e308]])
    )
    with pytest.raises(OverflowError):
        op.simplify()
    with pytest.raises(OverflowError):
        op.to_device(backend=pipeline).simplify()


@pytest.mark.parametrize("pipeline", BACKENDS, indirect=True)
@pytest.mark.parametrize("num_qubits", [24, 60, 100, 130])
def test_pipeline_non_finite_input_takes_precedence_over_overflow(pipeline: str, num_qubits: int) -> None:
    # CPU simplify validates every input before summing, so an operator that has
    # both a non-finite coefficient and an overflowing duplicate run raises
    # ValueError. The pipeline learns both on the device and must report the same.
    label = "X" * num_qubits
    filler = _filler(num_qubits, 1200, seed=num_qubits + 1)
    for position in (0, len(filler) // 2, len(filler) - 1):
        coeffs = np.concatenate([np.ones(len(filler), dtype=np.complex128), [1.7e308, 1.7e308]])
        coeffs[position] = complex(math.nan, 0.0)
        op = wolfgang_quantum.PauliSum.from_labels(filler + [label, label], coeffs)
        with pytest.raises(ValueError, match="finite"):
            op.simplify()
        with pytest.raises(ValueError, match="finite"):
            op.to_device(backend=pipeline).simplify()


@pytest.mark.parametrize("pipeline", BACKENDS, indirect=True)
@pytest.mark.parametrize("num_qubits", [24, 60, 100, 130])
def test_pipeline_overflowing_magnitude_uses_half_units(pipeline: str, num_qubits: int) -> None:
    # |c| exceeds DBL_MAX although both components are finite. At rtol=0 the CPU
    # threshold is atol itself; with rtol > 0 it is compared in half units.
    huge = complex(1.5e308, 1.5e308)
    a = "X" * num_qubits
    b = "Z" * num_qubits
    filler = _filler(num_qubits, 1200, seed=num_qubits + 2)
    op = wolfgang_quantum.PauliSum.from_labels(
        filler + [a, b, b], np.concatenate([np.full(len(filler), 1e-3), [huge, 1.0, 2.0]])
    )
    _assert_matches_cpu(pipeline, op)
    _assert_matches_cpu(pipeline, op, atol=1e-12, rtol=1e-300)
    _assert_matches_cpu(pipeline, op, atol=0.0, rtol=1e-308)
    # rtol * |huge| / 2 is about 1.06e8 in half units, which drops every
    # filler term and the combined Z term but keeps the huge one.
    assert op.to_device(backend=pipeline).simplify(rtol=1e-300).to_host().num_terms == 1


@pytest.mark.parametrize("pipeline", BACKENDS, indirect=True)
@pytest.mark.parametrize("num_qubits", [24, 40, 100, 130])
def test_pipeline_drop_threshold_is_separately_rounded(pipeline: str, num_qubits: int) -> None:
    # atol + rtol * max as one fused multiply-add lands an ulp above the Z
    # coefficient and drops it; the separately rounded threshold keeps it.
    x = "X" * num_qubits
    z = "Z" * num_qubits
    filler = _filler(num_qubits, 1200, seed=num_qubits + 3)
    op = wolfgang_quantum.PauliSum.from_labels(
        filler + [x, z], np.concatenate([np.full(len(filler), 0.5), [1.3927415900879758, 0.2785483180185952]])
    )
    _assert_matches_cpu(pipeline, op, rtol=0.2)
    labels = op.to_device(backend=pipeline).simplify(rtol=0.2).to_host().to_labels()[0]
    assert z in list(labels)


@pytest.mark.parametrize("pipeline", BACKENDS, indirect=True)
def test_pipeline_calls_carry_no_state(pipeline: str) -> None:
    # Status flags and counts are reset on the device by each call; an error in
    # one call must not leak into the next, on any key width.
    for num_qubits in (24, 40, 100, 130):
        label = "X" * num_qubits
        filler = _filler(num_qubits, 1200, seed=num_qubits + 4)
        overflow = wolfgang_quantum.PauliSum.from_labels(
            filler + [label, label], np.concatenate([np.ones(len(filler)), [1.7e308, 1.7e308]])
        ).to_device(backend=pipeline)
        coeffs = np.ones(len(filler) + 2, dtype=np.complex128)
        coeffs[3] = complex(math.inf, 0.0)
        non_finite = wolfgang_quantum.PauliSum.from_labels(filler + [label, label], coeffs).to_device(backend=pipeline)
        valid = _operator(num_qubits, 3000, 0.5, seed=num_qubits + 5)
        valid_device = valid.to_device(backend=pipeline)
        cancelling = wolfgang_quantum.PauliSum.from_labels(filler + filler, np.concatenate(
            [np.ones(len(filler)), -np.ones(len(filler))]
        )).to_device(backend=pipeline)
        expected = valid.simplify()
        for _ in range(2):
            with pytest.raises(OverflowError):
                overflow.simplify()
            actual = valid_device.simplify().to_host()
            assert list(actual.to_labels()[0]) == list(expected.to_labels()[0])
            with pytest.raises(ValueError, match="finite"):
                non_finite.simplify()
            assert cancelling.simplify().to_host().num_terms == 0
            actual = valid_device.simplify().to_host()
            np.testing.assert_allclose(actual.to_arrays()[2], expected.to_arrays()[2], rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("pipeline", BACKENDS, indirect=True)
def test_pipeline_is_safe_across_threads(pipeline: str) -> None:
    ops = [_operator(width, 4000, 0.5, seed=width) for width in (24, 40, 100, 130)]
    devices = [op.to_device(backend=pipeline) for op in ops]
    expected = [op.simplify() for op in ops]

    def run(index: int) -> wolfgang_quantum.PauliSum:
        return devices[index % len(devices)].simplify().to_host()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(32)))
    for index, actual in enumerate(results):
        reference = expected[index % len(expected)]
        assert list(actual.to_labels()[0]) == list(reference.to_labels()[0])
        np.testing.assert_allclose(actual.to_arrays()[2], reference.to_arrays()[2], rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("pipeline", BACKENDS, indirect=True)
@pytest.mark.parametrize("num_qubits", [24, 40, 100, 130])
def test_pipeline_matmul_matches_cpu(pipeline: str, num_qubits: int) -> None:
    lhs = _operator(num_qubits, 70, 0.2, seed=num_qubits + 6)
    rhs = _operator(num_qubits, 60, 0.2, seed=num_qubits + 7)
    expected = lhs.matmul(rhs)
    actual = lhs.to_device(backend=pipeline).matmul(rhs.to_device(backend=pipeline)).to_host()
    assert list(actual.to_labels()[0]) == list(expected.to_labels()[0])
    np.testing.assert_allclose(actual.to_arrays()[2], expected.to_arrays()[2], rtol=1e-10, atol=1e-10)
