"""Zero-copy device-array export and device-to-device import versus host round trips.

For operators already on a CUDA device, compares handing the x, z and
coefficient buffers to CuPy:

- export_dlpack: DevicePauliSum.device_arrays() then cupy.from_dlpack (no copy);
- export_via_host: to_host().to_arrays() then cupy.asarray (device to host to device);

and building a DevicePauliSum from CuPy arrays:

- import_device: DevicePauliSum.from_device_arrays (device-to-device copy);
- import_via_host: cupy.asnumpy, PauliSum.from_arrays, to_device.

Each variant runs --repeat times after warmups in a randomized order; results
are checked for equality before timing.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from _benchmark_metadata import benchmark_environment, command_string, git_provenance
except ModuleNotFoundError:
    from benchmarks._benchmark_metadata import (
        benchmark_environment,
        command_string,
        git_provenance,
    )

import wolfgang_quantum
import wolfgang_quantum._wolfgang_core as core


def packed_operator(rng: np.random.Generator, num_qubits: int, num_terms: int) -> wolfgang_quantum.PauliSum:
    words = (num_qubits + 63) // 64
    x = rng.integers(0, 2**63, size=(num_terms, words), dtype=np.uint64)
    z = rng.integers(0, 2**63, size=(num_terms, words), dtype=np.uint64)
    tail = num_qubits % 64
    if tail:
        mask = np.uint64((1 << tail) - 1)
        x[:, -1] &= mask
        z[:, -1] &= mask
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return wolfgang_quantum.PauliSum.from_arrays(x, z, coeffs, num_qubits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 100_000, 1_000_000])
    parser.add_argument("--qubits", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not core._cuda_status().get("runtime_available"):
        sys.stdout.write(json.dumps({"status": "skipped"}) + "\n")
        return 0
    import cupy

    rng = np.random.default_rng(args.seed)
    order_rng = random.Random(args.seed)
    rows = []
    for num_terms in args.sizes:
        op = packed_operator(rng, args.qubits, num_terms)
        device = op.to_device(backend="cuda")
        host_x, host_z, host_coeffs = op.to_arrays()
        dx, dz, dc = (cupy.asarray(array) for array in (host_x, host_z, host_coeffs))

        def export_dlpack() -> Any:
            arrays = [cupy.from_dlpack(view) for view in device.device_arrays()]
            cupy.cuda.runtime.deviceSynchronize()
            return arrays

        def export_via_host() -> Any:
            arrays = [cupy.asarray(array) for array in device.to_host().to_arrays()]
            cupy.cuda.runtime.deviceSynchronize()
            return arrays

        def import_device() -> Any:
            return wolfgang_quantum.DevicePauliSum.from_device_arrays(dx, dz, dc, args.qubits)

        def import_via_host() -> Any:
            host = wolfgang_quantum.PauliSum.from_arrays(
                cupy.asnumpy(dx), cupy.asnumpy(dz), cupy.asnumpy(dc), args.qubits
            )
            return host.to_device(backend="cuda")

        variants = {
            "export_dlpack": export_dlpack,
            "export_via_host": export_via_host,
            "import_device": import_device,
            "import_via_host": import_via_host,
        }
        for name in ("export_dlpack", "export_via_host"):
            got = [cupy.asnumpy(array) for array in variants[name]()]
            for actual, expected in zip(got, (host_x, host_z, host_coeffs), strict=True):
                if not np.array_equal(actual, expected):
                    raise RuntimeError(f"{name} returned different arrays")
        for name in ("import_device", "import_via_host"):
            if variants[name]().to_host() != op:
                raise RuntimeError(f"{name} built a different operator")
        samples: dict[str, list[float]] = {name: [] for name in variants}
        for iteration in range(3 + args.repeat):
            names = list(variants)
            order_rng.shuffle(names)
            for name in names:
                start = time.perf_counter()
                variants[name]()
                elapsed = time.perf_counter() - start
                if iteration >= 3:
                    samples[name].append(elapsed)
        rows.append(
            {
                "num_terms": num_terms,
                "num_qubits": args.qubits,
                "variants": {
                    name: {"median_seconds": float(np.median(values)), "raw_seconds": values}
                    for name, values in samples.items()
                },
            }
        )
    report = {
        "benchmark": "device_arrays",
        "command": command_string(),
        "git": git_provenance(),
        "seed": args.seed,
        "repeat": args.repeat,
        "cupy_version": cupy.__version__,
        "environment": benchmark_environment(dict(core._build_info()), numpy_version=np.__version__),
        "cuda_status": dict(core._cuda_status()),
        "rows": rows,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
