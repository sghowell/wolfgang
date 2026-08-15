#!/usr/bin/env python3
"""Generate a reproducible public/private cloud qualification harness bundle."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from archive_portability import resolve_source_snapshot

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LaneSpec:
    lane: str
    provider: str
    backend: str
    architecture: str
    accelerator_label: str
    toolkit_label: str
    compiler_probe: str
    inventory_collector: str
    build_flags: list[str]
    diagnostics: list[str]
    interop_checks: list[str]
    test_targets: list[str]
    benchmark_command: str
    benchmark_baseline_label: str
    cleanup_note: str


LANES: dict[str, LaneSpec] = {
    "hopper": LaneSpec(
        lane="hopper",
        provider="lambda_h100",
        backend="cuda",
        architecture="sm_90",
        accelerator_label="NVIDIA Hopper (H100/H200/GH200 only when justified)",
        toolkit_label="cuda",
        compiler_probe="nvcc --version",
        inventory_collector="tools/remote/collect_cuda_inventory.sh",
        build_flags=[
            "FASTPAULI_ENABLE_CUDA=ON",
            "FASTPAULI_ENABLE_HIP=OFF",
            "FASTPAULI_ENABLE_METAL=OFF",
            "FASTPAULI_CUDA_ARCHITECTURES=90",
            "FASTPAULI_ENABLE_NATIVE=OFF",
        ],
        diagnostics=["compute_sanitizer_memcheck", "compute_sanitizer_racecheck"],
        interop_checks=["dlpack_roundtrip", "cuda_array_interface", "cross_stream_sync"],
        test_targets=[
            "tests/test_phase11_cuda_kernels.py",
            "tests/test_dlpack_and_cuda_interop_contract.py",
        ],
        benchmark_command=(
            ".venv/bin/python benchmarks/bench_cuda_kernels.py --smoke "
            "--warmup 10 --repeat 30 --json --output private/raw/cuda_smoke.json"
        ),
        benchmark_baseline_label="hopper_frozen_release_baseline",
        cleanup_note="Terminate the paid NVIDIA instance immediately if any required gate fails.",
    ),
    "blackwell": LaneSpec(
        lane="blackwell",
        provider="lambda_b200",
        backend="cuda",
        architecture="sm_100+",
        accelerator_label="NVIDIA Blackwell (prefer B200)",
        toolkit_label="cuda",
        compiler_probe="nvcc --version",
        inventory_collector="tools/remote/collect_cuda_inventory.sh",
        build_flags=[
            "FASTPAULI_ENABLE_CUDA=ON",
            "FASTPAULI_ENABLE_HIP=OFF",
            "FASTPAULI_ENABLE_METAL=OFF",
            "FASTPAULI_CUDA_ARCHITECTURES=100-real;120",
            "FASTPAULI_ENABLE_NATIVE=OFF",
        ],
        diagnostics=["compute_sanitizer_memcheck", "compute_sanitizer_racecheck"],
        interop_checks=["dlpack_roundtrip", "cuda_array_interface", "cross_stream_sync"],
        test_targets=[
            "tests/test_phase11_cuda_kernels.py",
            "tests/test_dlpack_and_cuda_interop_contract.py",
        ],
        benchmark_command=(
            ".venv/bin/python benchmarks/bench_cuda_kernels.py --smoke "
            "--warmup 10 --repeat 30 --json --output private/raw/cuda_blackwell_smoke.json"
        ),
        benchmark_baseline_label="blackwell_frozen_release_baseline",
        cleanup_note="Terminate the paid Blackwell instance immediately if any required gate fails.",
    ),
    "mi300x": LaneSpec(
        lane="mi300x",
        provider="azure_nd_mi300x_v5",
        backend="rocm",
        architecture="gfx942",
        accelerator_label="AMD Instinct MI300X",
        toolkit_label="rocm",
        compiler_probe="hipcc --version",
        inventory_collector="tools/remote/collect_rocm_inventory.sh",
        build_flags=[
            "FASTPAULI_ENABLE_HIP=ON",
            "FASTPAULI_HIP_ARCHITECTURES=gfx942",
            "FASTPAULI_ENABLE_CUDA=OFF",
            "FASTPAULI_ENABLE_METAL=OFF",
            "FASTPAULI_ENABLE_NATIVE=OFF",
        ],
        diagnostics=["rocprof_trace_stats", "rocm_debug_or_sanitizer_reduced_pass"],
        interop_checks=["host_device_roundtrip", "non_contiguous_input", "cross_stream_sync"],
        test_targets=[
            "tests/test_phase12_rocm_foundation.py",
            "tests/test_backend_neutral_prep.py",
        ],
        benchmark_command=(
            ".venv/bin/python benchmarks/bench_rocm_kernels.py --profile campaign7-release-smoke "
            "--warmup 10 --repeat 30 --json --output private/raw/rocm_smoke.json"
        ),
        benchmark_baseline_label="mi300x_frozen_release_baseline",
        cleanup_note="Terminate the paid MI300X instance immediately if any required gate fails.",
    ),
}


def public_manifest(
    spec: LaneSpec,
    *,
    source_commit: str | None = None,
    source_tree_state: str | None = None,
) -> dict[str, object]:
    source = resolve_source_snapshot(
        ROOT,
        source_commit=source_commit,
        source_tree_state=source_tree_state,
    )
    return {
        "schema_version": 1,
        "lane": spec.lane,
        "provider": spec.provider,
        "backend": spec.backend,
        "architecture": spec.architecture,
        "accelerator_label": spec.accelerator_label,
        "source": source,
        "build": {
            "compiler": {"command": spec.compiler_probe, "value": "capture_on_remote"},
            "build_flags": spec.build_flags,
            "artifact_hashes": {"sdist_sha256": "capture_on_remote", "wheel_sha256": "capture_on_remote"},
        },
        "runtime": {
            "driver": "capture_on_remote",
            "toolkit": {spec.toolkit_label: "capture_on_remote"},
            "device": "capture_on_remote",
            "os": "capture_on_remote",
            "python": "capture_on_remote",
        },
        "test_counts": {"total": "capture_on_remote", "passed": "capture_on_remote", "failed": 0, "skipped": "capture_on_remote"},
        "diagnostics": {
            "required": spec.diagnostics,
            "status": "capture_on_remote",
            "summary": "sanitized_derived_evidence_only",
        },
        "numerical_parity": {
            "cpu_reference_commit": source["short_commit"],
            "fp64": {"max_relative_error": "capture_on_remote", "max_absolute_error": "capture_on_remote"},
            "fp32": {"max_relative_error": "capture_on_remote", "max_absolute_error": "capture_on_remote"},
            "nan_inf_mismatch": False,
        },
        "interop_checks": {
            key: "capture_on_remote" for key in spec.interop_checks
        },
        "benchmarks": {
            "policy": "public/benchmark_policy.json",
            "baseline_label": spec.benchmark_baseline_label,
            "median_variance_limit_percent": 5,
            "result_summary": "capture_on_remote",
        },
        "reproducibility": {
            "same_image_reruns": 2,
            "fresh_provision_reruns": 1,
            "test_count_match_required": True,
        },
        "public_artifact_policy": {
            "sanitized_derived_evidence_only": True,
            "raw_environment_dumps_in_public_tree": "forbidden",
            "raw_profiler_data_in_public_tree": "forbidden",
            "hostnames_ips_cloud_ids": "forbidden",
        },
        "cleanup": {
            "fail_closed": True,
            "note": spec.cleanup_note,
        },
    }


def benchmark_policy(spec: LaneSpec) -> dict[str, object]:
    return {
        "lane": spec.lane,
        "warmup_iterations": 10,
        "timed_iterations": 30,
        "cross_architecture_comparisons": "forbidden",
        "baseline_policy": "compare_only_against_same_architecture_frozen_baseline",
        "informational_rows_require_label": True,
        "benchmark_command": spec.benchmark_command,
    }


def runbook_text(spec: LaneSpec) -> str:
    return f"""# Wolfgang cloud hardware qualification harness ({spec.lane})

This bundle is a dry-runable, non-secret harness for the {spec.accelerator_label} lane.

## Public/private split

- `public/` contains sanitized derived evidence only.
- `private/` stores raw logs, profiler captures, cloud account details, and any other non-public material outside the repository release surface.
- Never copy hostnames, IPs, SSH targets, subscription IDs, or raw profiler databases into `public/`.

## Required captures

The final `public/qualification_manifest.json` must capture commit/tree state, compiler,
driver, {spec.toolkit_label.upper()} version, device, architecture, build flags, test counts,
diagnostics summary, numerical parity, interop checks, benchmark timing policy, and
cleanup/termination outcomes.

## Fail-closed policy

If any functional, parity, interop, diagnostics, or benchmark gate fails, stop the run,
record the failure in sanitized derived evidence, and terminate the cloud instance rather
than continuing with partial or ambiguous evidence. This harness is fail-closed by policy.

## Cleanup

{spec.cleanup_note}
Keep raw environment dumps and raw profiler traces under `private/` only, then run the
public artifact audit before moving any derived file into version control.
"""


def runner_script(spec: LaneSpec) -> str:
    flags = " \\\n  ".join(spec.build_flags)
    tests = " \\\n  ".join(spec.test_targets)
    inventory = spec.inventory_collector
    return f"""#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
BUNDLE_DIR=$(cd "$(dirname "$0")/.." && pwd)
PUBLIC_DIR="$BUNDLE_DIR/public"
PRIVATE_DIR="$BUNDLE_DIR/private"
RAW_DIR="$PRIVATE_DIR/raw"
LOG_DIR="$PRIVATE_DIR/logs"
PROFILER_DIR="$PRIVATE_DIR/profiler"
TERMINATE_CLOUD_INSTANCE_COMMAND="${{TERMINATE_CLOUD_INSTANCE_COMMAND:-echo 'set TERMINATE_CLOUD_INSTANCE_COMMAND before live use'}}"

cleanup() {{
  status=$?
  if [[ $status -ne 0 ]]; then
    echo "qualification failed; invoking fail-closed termination guidance" >&2
    eval "$TERMINATE_CLOUD_INSTANCE_COMMAND" || true
  fi
  exit $status
}}
trap cleanup EXIT

mkdir -p "$PUBLIC_DIR" "$RAW_DIR" "$LOG_DIR" "$PROFILER_DIR"
cp "$PUBLIC_DIR/qualification_manifest.json" "$PUBLIC_DIR/qualification_manifest.preflight.json"

bash "$ROOT_DIR/{inventory}" > "$LOG_DIR/inventory.log"
{spec.compiler_probe} > "$LOG_DIR/compiler.log" 2>&1 || true

cat > "$LOG_DIR/build-flags.txt" <<'EOF'
{flags}
EOF

printf '%s\n' \
  "tests to execute:" \
  {tests!r} > "$LOG_DIR/test-targets.txt"

printf '%s\n' {spec.benchmark_command!r} > "$LOG_DIR/benchmark-command.txt"
python scripts/audit_public_artifacts.py --path "$PUBLIC_DIR"

echo "Update $PUBLIC_DIR/qualification_manifest.json with sanitized derived evidence after the live run."
"""


def private_readme(spec: LaneSpec) -> str:
    return f"""# Private qualification material for {spec.lane}

Keep raw host inventories, profiler traces, private SSH/provider details, and verbose command logs here.

Rules:
- `private/` is intentionally separate from `public/`.
- Do not commit anything from `private/`.
- Derive the sanitized summary fields needed by `public/qualification_manifest.json` and `public/benchmark_policy.json`.
- If you must share a failure externally, summarize it in sanitized derived evidence rather than copying raw files.
"""


def bundle_lane(
    spec: LaneSpec,
    output_dir: Path,
    *,
    source_commit: str | None = None,
    source_tree_state: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "public").mkdir(exist_ok=True)
    (output_dir / "private").mkdir(exist_ok=True)
    (output_dir / "scripts").mkdir(exist_ok=True)

    (output_dir / "public" / "qualification_manifest.json").write_text(
        json.dumps(
            public_manifest(
                spec,
                source_commit=source_commit,
                source_tree_state=source_tree_state,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "public" / "benchmark_policy.json").write_text(
        json.dumps(benchmark_policy(spec), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "RUNBOOK.md").write_text(runbook_text(spec), encoding="utf-8")
    runner = output_dir / "scripts" / "run_lane.sh"
    runner.write_text(runner_script(spec), encoding="utf-8")
    runner.chmod(0o755)
    (output_dir / "private" / "README.md").write_text(private_readme(spec), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bundle = subparsers.add_parser("bundle", help="write a dry-run qualification bundle")
    bundle.add_argument("--lane", choices=sorted(LANES), required=True)
    bundle.add_argument("--output-dir", type=Path, required=True)
    bundle.add_argument("--source-commit")
    bundle.add_argument("--source-tree-state")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "bundle":
        spec = LANES[args.lane]
        bundle_lane(
            spec,
            args.output_dir,
            source_commit=args.source_commit,
            source_tree_state=args.source_tree_state,
        )
        print(f"wrote {spec.lane} qualification harness bundle to {args.output_dir}")
        return
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
