"""Time simplify at one operator size, for allocation profiling.

Run under a tracer to count and time allocation calls, for example:

    rocprofv3 --hip-runtime-trace --stats -d <dir> -o run --output-format csv -- \\
        python benchmarks/probe_simplify_allocations.py hip 24 50000 40

Arguments: backend, qubits, terms, timed calls. The operator has 50% duplicates.
It prints median device-resident and transfer-inclusive times (5 warmup calls
each), so a traced run also covers to_device and to_host allocations.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import wolfgang_quantum


def main() -> int:
    backend, num_qubits, num_terms, calls = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    rng = np.random.default_rng(7)
    unique = num_terms // 2
    letters = np.asarray(list("IXYZ"))
    chars = letters[rng.integers(0, 4, size=(unique, num_qubits))]
    base = np.ascontiguousarray(chars).view(f"<U{num_qubits}").ravel()
    labels = np.concatenate([base, base[rng.integers(0, unique, num_terms - unique)]]).tolist()
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    op = wolfgang_quantum.PauliSum.from_labels(labels, coeffs)
    device = op.to_device(backend=backend)
    variants = (
        ("device_resident", device.simplify),
        ("transfer_inclusive", lambda: op.to_device(backend=backend).simplify().to_host()),
    )
    for name, fn in variants:
        for _ in range(5):
            fn()
        samples = []
        for _ in range(calls):
            start = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - start)
        print(f"{name:20s} median {np.median(samples) * 1e6:9.1f} us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
