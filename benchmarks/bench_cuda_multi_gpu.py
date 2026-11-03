"""Multi-GPU CUDA behavior: thread scaling across two devices and the cost of peer access.

``--suite threads`` times each operation from one thread on device 0, then from
two threads at once (one on device 0, one on device 1, each making the same
number of calls), and reports the two-thread throughput as a multiple of the
one-thread throughput. A device call that holds the GIL cannot overlap with
another, so the ratio stays near 1; with the GIL released it approaches 2.

``--suite peer`` measures, in fresh processes, what mapping a device's memory
pool into its peer costs: the first and a later ``device_arrays()`` export, the
memory the process holds on the peer GPU afterwards (from ``nvidia-smi``),
growing the pool by 1 GiB, and a small simplify per call, each with and
without a prior export. A build without the mapping (main before slice 6)
gives the baseline for the export rows.

Needs two CUDA devices; prints ``{"status": "skipped"}`` otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
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


def operator(num_terms: int, seed: int, duplicates: float = 0.5) -> wolfgang_quantum.PauliSum:
    """A 24-qubit operator in which `duplicates` of the terms repeat earlier ones."""
    rng = np.random.default_rng(seed)
    distinct = max(1, int(num_terms * (1 - duplicates)))
    x = rng.integers(0, 1 << 24, size=(distinct, 1), dtype=np.uint64)
    z = rng.integers(0, 1 << 24, size=(distinct, 1), dtype=np.uint64)
    pick = rng.integers(0, distinct, size=num_terms)
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return wolfgang_quantum.PauliSum.from_arrays(x[pick], z[pick], coeffs, 24)


def operations(device: int) -> dict[str, tuple[Callable[[], Any], int]]:
    """Name -> (call, calls per thread per batch), with operands on `device`."""
    large = operator(1_000_000, 1)
    medium = operator(4_096, 2)
    small = operator(128, 3)
    square = operator(2_048, 4, duplicates=0.0)
    on = {"large": large.to_device(device=device), "medium": medium.to_device(device=device),
          "small": small.to_device(device=device), "square": square.to_device(device=device)}
    return {
        "simplify_1m_terms_24q": (on["large"].simplify, 100),
        "simplify_4096_terms_24q": (on["medium"].simplify, 1_000),
        "matmul_128x128_24q": (lambda: on["small"].matmul(on["small"]), 1_000),
        "commutes_with_device_2048x2048_count": (
            lambda: on["square"].commutes_with_device(on["square"]).count_commuting(), 300),
        "to_host_1m_terms_24q": (on["large"].to_host, 30),
        "to_device_1m_terms_24q": (lambda: large.to_device(device=device), 30),
    }


def run_threads(plans: list[tuple[Callable[[], Any], int]]) -> float:
    """Start one thread per (call, count) plan together; return the wall time."""
    barrier = threading.Barrier(len(plans) + 1)

    def work(call: Callable[[], Any], count: int) -> None:
        barrier.wait()
        for _ in range(count):
            call()

    threads = [threading.Thread(target=work, args=plan) for plan in plans]
    for thread in threads:
        thread.start()
    barrier.wait()
    start = time.perf_counter()
    for thread in threads:
        thread.join()
    return time.perf_counter() - start


def threads_suite(repeat: int) -> dict[str, Any]:
    ops = {device: operations(device) for device in (0, 1)}
    rows: dict[str, Any] = {}
    for name, (call0, count) in ops[0].items():
        call1 = ops[1][name][0]
        for _ in range(max(3, count // 10)):
            call0()
            call1()
        one, two, same = [], [], []
        for _ in range(repeat):
            one.append(run_threads([(call0, count)]))
            two.append(run_threads([(call0, count), (call1, count)]))
            same.append(run_threads([(call0, count), (call0, count)]))
        one_s, two_s, same_s = (float(np.median(v)) for v in (one, two, same))
        rows[name] = {
            "calls_per_thread": count,
            "one_thread_seconds_per_call": one_s / count,
            "two_devices_wall_seconds": two,
            "one_thread_wall_seconds": one,
            "same_device_two_threads_wall_seconds": same,
            "two_device_throughput_vs_one_thread": 2 * one_s / two_s,
            "same_device_throughput_vs_one_thread": 2 * one_s / same_s,
        }
    return rows


def peer_gpu_memory_mib(peer: int) -> float:
    """Memory this process holds on GPU `peer`, from nvidia-smi (0 when it has no context there)."""
    uuid = subprocess.run(
        ["nvidia-smi", "-i", str(peer), "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    for line in apps.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3 and fields[0] == uuid and fields[1] == str(os.getpid()):
            return float(fields[2])
    return 0.0


def peer_child(export: bool) -> dict[str, Any]:
    """One fresh-process measurement on device 1 with device 0 as the peer."""
    op = operator(1_000_000, 5)
    device_op = op.to_device(device=1)
    device_op.simplify()
    result: dict[str, Any] = {"export": export, "peer_memory_mib_before": peer_gpu_memory_mib(0)}
    if export:
        start = time.perf_counter()
        views = device_op.device_arrays()
        result["first_export_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        device_op.device_arrays()
        result["second_export_seconds"] = time.perf_counter() - start
        del views
    result["peer_memory_mib_after"] = peer_gpu_memory_mib(0)
    # Grow the pool by about 1 GiB: eight 4M-term operators held at once.
    block = operator(4_000_000, 6)
    start = time.perf_counter()
    held = [block.to_device(device=1) for _ in range(8)]
    result["grow_pool_1gib_seconds"] = time.perf_counter() - start
    del held
    small = operator(4_096, 7).to_device(device=1)
    for _ in range(100):
        small.simplify()
    start = time.perf_counter()
    for _ in range(1_000):
        small.simplify()
    result["simplify_4096_terms_seconds_per_call"] = (time.perf_counter() - start) / 1_000
    return result


def peer_suite(repeat: int) -> dict[str, Any]:
    samples: dict[str, list[dict[str, Any]]] = {"export": [], "no_export": []}
    for _ in range(repeat):
        for export in (True, False):
            out = subprocess.run(
                [sys.executable, __file__, "--child-peer", "1" if export else "0"],
                capture_output=True, text=True, check=True,
            ).stdout
            samples["export" if export else "no_export"].append(json.loads(out))
    summary: dict[str, Any] = {}
    for key, runs in samples.items():
        for field in runs[0]:
            if field == "export":
                continue
            values = [run[field] for run in runs]
            summary.setdefault(key, {})[field] = {"median": float(np.median(values)), "values": values}
    return {"samples": samples, "summary": summary}


def summarize_reports(directory: Path) -> dict[str, Any]:
    """Medians across saved reports, per label, with each run's provenance."""
    runs: list[dict[str, Any]] = []
    threads: dict[str, dict[str, dict[str, list[float]]]] = {}
    peer: dict[str, dict[str, dict[str, list[float]]]] = {}
    for path in sorted(directory.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("benchmark") not in ("cuda_multi_gpu_threads", "cuda_multi_gpu_peer"):
            continue
        label = report.get("label", "")
        runs.append({
            "file": path.name,
            "benchmark": report["benchmark"],
            "label": label,
            "command": report.get("command"),
            "git_commit": report["git"].get("commit"),
            "git_dirty": report["git"].get("dirty"),
            "loaded_native_build": report["environment"].get("loaded_native_build", {}),
        })
        if report["benchmark"] == "cuda_multi_gpu_threads":
            for name, row in report["rows"].items():
                fields = threads.setdefault(name, {}).setdefault(label, {})
                for field in ("one_thread_seconds_per_call", "two_device_throughput_vs_one_thread",
                              "same_device_throughput_vs_one_thread"):
                    fields.setdefault(field, []).append(row[field])
        else:
            for mode, fields_summary in report["rows"]["summary"].items():
                fields = peer.setdefault(mode, {}).setdefault(label, {})
                for field, values in fields_summary.items():
                    fields.setdefault(field, []).extend(values["values"])

    def medians(table: dict[str, dict[str, dict[str, list[float]]]]) -> dict[str, Any]:
        return {
            name: {
                label: {field: {"values": values, "median": float(np.median(values))} for field, values in fields.items()}
                for label, fields in by_label.items()
            }
            for name, by_label in table.items()
        }

    return {"runs": runs, "threads": medians(threads), "peer": medians(peer)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("threads", "peer"), default="threads")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--label", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summarize", type=Path, help="aggregate saved reports in this directory")
    parser.add_argument("--child-peer", choices=("0", "1"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.summarize:
        text = json.dumps(summarize_reports(args.summarize), indent=2, sort_keys=True)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        else:
            sys.stdout.write(text + "\n")
        return 0
    if args.child_peer is not None:
        sys.stdout.write(json.dumps(peer_child(args.child_peer == "1")) + "\n")
        return 0
    status = dict(core._cuda_status())
    if not status.get("runtime_available") or len(wolfgang_quantum.cuda_devices()) < 2:
        sys.stdout.write(json.dumps({"status": "skipped"}) + "\n")
        return 0
    rows = threads_suite(args.repeat) if args.suite == "threads" else peer_suite(args.repeat)
    report = {
        "benchmark": f"cuda_multi_gpu_{args.suite}",
        "label": args.label,
        "command": command_string(),
        "git": git_provenance(),
        "repeat": args.repeat,
        "environment": benchmark_environment(dict(core._build_info()), numpy_version=np.__version__),
        "cuda_status": status,
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
