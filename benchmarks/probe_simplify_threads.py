"""Time repeated CPU simplify calls under an optional thread cap.

Meant to run inside a CPU quota smaller than the visible cores, for example:

    sudo systemd-run --scope -p CPUQuota=400% --uid=$USER \\
        python benchmarks/probe_simplify_threads.py 0 60
    sudo systemd-run --scope -p CPUQuota=400% --uid=$USER \\
        python benchmarks/probe_simplify_threads.py 4 60

Arguments: thread cap (0 for none) and timed calls. The operator is the
500,000-term operator from bench_cpu_threads.py. Prints one JSON line with the
median, mean, 90th-percentile and maximum call times in milliseconds.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wolfgang_quantum  # noqa: E402
from bench_cpu_threads import make_operator  # noqa: E402


def main() -> int:
    cap, calls = int(sys.argv[1]), int(sys.argv[2])
    if cap:
        wolfgang_quantum.set_num_threads(cap)
    op = make_operator(500_000)
    op.simplify()
    times = []
    for _ in range(calls):
        start = time.perf_counter()
        op.simplify()
        times.append((time.perf_counter() - start) * 1e3)
    ordered = sorted(times)
    print(json.dumps({
        "cap": cap,
        "threads": wolfgang_quantum.get_num_threads(),
        "calls": calls,
        "median_ms": round(float(np.median(times)), 2),
        "mean_ms": round(float(np.mean(times)), 2),
        "p90_ms": round(ordered[int(0.9 * (calls - 1))], 2),
        "max_ms": round(ordered[-1], 2),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
