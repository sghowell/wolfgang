"""Time the first and second to_host on fresh threads.

The first call on a thread pays for its pinned staging buffer
(src/detail/staged_transfer.hpp). For 1, 128 and 1,000,000 terms at 24 qubits,
each of 15 fresh threads times one cold and one warm call; the medians are
printed. Usage: python benchmarks/probe_cold_to_host.py <cuda|hip>
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np
import wolfgang_quantum


def main() -> int:
    backend = sys.argv[1]
    for num_terms in (1, 128, 1_000_000):
        rng = np.random.default_rng(1)
        x = rng.integers(0, 1 << 24, size=(num_terms, 1), dtype=np.uint64)
        op = wolfgang_quantum.PauliSum.from_arrays(x, x.copy(), rng.normal(size=num_terms) + 0j, 24)
        device = op.to_device(backend=backend)
        device.to_host()
        cold: list[float] = []
        warm: list[float] = []

        def run() -> None:
            start = time.perf_counter()
            device.to_host()
            cold.append(time.perf_counter() - start)
            start = time.perf_counter()
            device.to_host()
            warm.append(time.perf_counter() - start)

        for _ in range(15):
            thread = threading.Thread(target=run)
            thread.start()
            thread.join()
        print(
            f"{num_terms:>9,d} terms: first call on a new thread {np.median(cold) * 1e6:9.1f} us, "
            f"second {np.median(warm) * 1e6:9.1f} us"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
