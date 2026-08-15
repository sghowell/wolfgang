#!/usr/bin/env python3
"""Measure ROCm free-memory deltas across process lifetime without normalizing growth away."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]


def load_hip_library() -> ctypes.CDLL:
    for candidate in ("libamdhip64.so", "/opt/rocm/lib/libamdhip64.so"):
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    raise RuntimeError("unable to load libamdhip64.so")


def hip_mem_info() -> dict[str, int]:
    lib = load_hip_library()
    free_mem = ctypes.c_size_t()
    total_mem = ctypes.c_size_t()
    rc = lib.hipMemGetInfo(ctypes.byref(free_mem), ctypes.byref(total_mem))
    if rc != 0:
        raise RuntimeError(f"hipMemGetInfo failed with rc={rc}")
    return {"free": int(free_mem.value), "total": int(total_mem.value)}


def settled_sample(tag: str, *, sample_count: int = 5, sleep_seconds: float = 0.05) -> dict[str, Any]:
    samples: list[dict[str, int]] = []
    for index in range(1, sample_count + 1):
        reading = hip_mem_info()
        samples.append({"index": index, **reading})
        if sleep_seconds and index != sample_count:
            time.sleep(sleep_seconds)
    free_values = sorted(sample["free"] for sample in samples)
    median_free = free_values[len(free_values) // 2]
    return {
        "tag": tag,
        "samples": samples,
        "min_free": min(free_values),
        "max_free": max(free_values),
        "median_free": median_free,
        "spread_bytes": max(free_values) - min(free_values),
    }


def summarize_probe(
    *,
    before_process: dict[str, Any],
    after_first_kernel: dict[str, Any],
    after_cycles: dict[str, Any],
    after_exit: dict[str, Any],
    construct_destroy_samples: list[dict[str, int]] | None = None,
) -> dict[str, Any]:
    first_kernel_reservation = after_first_kernel["median_free"] - before_process["median_free"]
    growth_after_cycles = after_cycles["median_free"] - after_first_kernel["median_free"]
    residual_after_exit = after_exit["median_free"] - before_process["median_free"]
    post_exit_recovered = after_exit["median_free"] - after_cycles["median_free"]
    cycle_samples = construct_destroy_samples or []
    if cycle_samples:
        first_cycle_free = cycle_samples[0]["free"]
        cycle_plateau_free = max(sample["free"] for sample in cycle_samples)
        cycle_plateau_growth = after_cycles["median_free"] - first_cycle_free
        warmup_reservation = first_cycle_free - after_first_kernel["median_free"]
    else:
        cycle_plateau_free = after_cycles["median_free"]
        cycle_plateau_growth = growth_after_cycles
        warmup_reservation = 0
    if residual_after_exit == 0 and cycle_plateau_growth == 0:
        adjudicated_retained_growth = 0
    else:
        adjudicated_retained_growth = growth_after_cycles
    return {
        "before_process": before_process,
        "after_first_kernel": after_first_kernel,
        "after_cycles": after_cycles,
        "after_exit": after_exit,
        "first_kernel_reservation_bytes": first_kernel_reservation,
        "growth_after_cycles_bytes": growth_after_cycles,
        "residual_after_exit_bytes": residual_after_exit,
        "one_time_reservation_bytes": first_kernel_reservation,
        "retained_growth_bytes": growth_after_cycles,
        "cycle_plateau_free_bytes": cycle_plateau_free,
        "cycle_plateau_growth_bytes": cycle_plateau_growth,
        "warmup_reservation_bytes": warmup_reservation,
        "adjudicated_retained_growth_bytes": adjudicated_retained_growth,
        "post_exit_recovered_bytes": post_exit_recovered,
    }


def child_probe_payload() -> dict[str, Any]:
    import fastpauli
    import fastpauli._fastpauli_core as core
    import numpy as np

    status = core._hip_status()
    if not status.get("runtime_available", False):
        raise RuntimeError(status.get("skip_reason") or "HIP runtime unavailable")

    rng = np.random.default_rng(12345)

    def labels_from_rng(num_qubits: int, terms: int, weight: int) -> list[str]:
        labels: list[str] = []
        alphabet = np.array(list("IXYZ"))
        for _ in range(terms):
            chars = np.full(num_qubits, "I", dtype="<U1")
            indices = rng.choice(num_qubits, size=weight, replace=False)
            chars[indices] = rng.choice(alphabet[1:], size=weight, replace=True)
            labels.append("".join(chars.tolist()))
        return labels

    before_process = settled_sample("before_process")

    base = fastpauli.PauliSum.from_labels(
        labels_from_rng(193, 4096, 10),
        (rng.normal(size=4096) + 1j * rng.normal(size=4096)).tolist(),
    )
    device_base = base.to_device()
    simplified = device_base.simplify()
    _ = simplified.to_host().num_terms
    after_first_kernel = settled_sample("after_first_kernel")

    construct_destroy_samples: list[dict[str, int]] = []
    for cycle in range(20):
        op = fastpauli.PauliSum.from_labels(
            labels_from_rng(193, 8192, 10),
            (rng.normal(size=8192) + 1j * rng.normal(size=8192)).tolist(),
        )
        device = op.to_device()
        simplified_cycle = device.simplify()
        _ = simplified_cycle.to_host().num_terms
        lhs = fastpauli.PauliSum.from_labels(
            labels_from_rng(193, 128, 6),
            (rng.normal(size=128) + 1j * rng.normal(size=128)).tolist(),
        )
        rhs = fastpauli.PauliSum.from_labels(
            labels_from_rng(193, 128, 6),
            (rng.normal(size=128) + 1j * rng.normal(size=128)).tolist(),
        )
        _ = lhs.to_device().matmul(rhs.to_device(), simplify=True).to_host().num_terms
        del device, simplified_cycle, op, lhs, rhs
        gc.collect()
        if cycle in {0, 4, 9, 14, 19}:
            construct_destroy_samples.append({"cycle": cycle + 1, **hip_mem_info()})
    after_cycles = settled_sample("after_cycles")

    del device_base, simplified, base
    gc.collect()

    return {
        "hip_status": status,
        "build_info": core._build_info(),
        "before_process": before_process,
        "after_first_kernel": after_first_kernel,
        "after_cycles": after_cycles,
        "construct_destroy_samples": construct_destroy_samples,
    }


def _child_main(output_path: Path) -> None:
    payload = child_probe_payload()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_child_probe(output_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child", "--output", str(output_path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout or f"child probe failed: {completed.returncode}")
    return json.loads(output_path.read_text(encoding="utf-8"))


def run_parent_probe(
    *,
    workspace: Path,
    settled_sample: Callable[..., dict[str, Any]] = settled_sample,
    run_child_probe: Callable[[Path], dict[str, Any]] = run_child_probe,
) -> dict[str, Any]:
    output_path = workspace / "rocm_memory_probe.child.json"
    before_process = settled_sample("before_process")
    child_payload = run_child_probe(output_path)
    after_exit = settled_sample("after_exit")
    summary = summarize_probe(
        before_process=before_process,
        after_first_kernel=child_payload["after_first_kernel"],
        after_cycles=child_payload["after_cycles"],
        after_exit=after_exit,
        construct_destroy_samples=child_payload.get("construct_destroy_samples", []),
    )
    summary.update(
        {
            "hip_status": child_payload.get("hip_status"),
            "build_info": child_payload.get("build_info"),
            "construct_destroy_samples": child_payload.get("construct_destroy_samples", []),
            "child_output": str(output_path),
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Where to write the JSON summary.")
    parser.add_argument("--child", action="store_true", help="Run the in-process child probe payload.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.child:
        _child_main(args.output)
        return
    summary = run_parent_probe(workspace=args.output.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
