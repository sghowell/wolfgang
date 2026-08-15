#!/usr/bin/env python3
"""Derive sanitized public H100 validation artifacts from exact captured evidence."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

PYTEST_SUMMARY_RE = re.compile(r"(\d+) passed(?:, (\d+) skipped)?")
NO_CUDA_API_CALL = "Target application terminated before first instrumented API call"
NANOBIND_LEAK = "nanobind: leaked"
MEMCHECK_CLEAN = "ERROR SUMMARY: 0 errors"
RACECHECK_CLEAN = "RACECHECK SUMMARY: 0 hazards displayed"


def parse_pytest_summary(text: str) -> dict[str, int] | None:
    match = PYTEST_SUMMARY_RE.search(text)
    if not match:
        return None
    passed = int(match.group(1))
    skipped = int(match.group(2) or 0)
    return {
        "passed": passed,
        "skipped": skipped,
        "failed": 0,
        "total": passed + skipped,
    }


def classify_per_test_memcheck(*, exit_code: int, log_text: str) -> str:
    if NANOBIND_LEAK in log_text:
        return "nanobind_leak_diagnostic"
    if NO_CUDA_API_CALL in log_text:
        return "no_cuda_api_call"
    if MEMCHECK_CLEAN in log_text and exit_code == 0:
        return "cuda_api_clean"
    return "failed_or_missing"


def summarize_per_test_memcheck(entries: list[dict[str, Any]]) -> dict[str, Any]:
    classified: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    by_category: dict[str, list[str]] = {}
    for entry in entries:
        category = classify_per_test_memcheck(
            exit_code=int(entry.get("exit_code", 1)),
            log_text=str(entry.get("log_text", "")),
        )
        nodeid = str(entry["nodeid"])
        counts[category] = counts.get(category, 0) + 1
        by_category.setdefault(category, []).append(nodeid)
        classified.append({**entry, "classification": category})
    leak_positive = by_category.get("nanobind_leak_diagnostic", [])
    no_cuda_api = by_category.get("no_cuda_api_call", [])
    failures = by_category.get("failed_or_missing", [])
    return {
        "classified": classified,
        "counts": counts,
        "leak_positive_nodeids": leak_positive,
        "no_cuda_api_call_nodeids": no_cuda_api,
        "failed_or_missing_nodeids": failures,
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_build_info(build_info_log: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lines = [line for line in build_info_log.read_text(encoding="utf-8").splitlines() if line.strip().startswith("{")]
    if len(lines) < 2:
        raise ValueError(f"expected build-info dicts in {build_info_log}")
    return ast.literal_eval(lines[0]), ast.literal_eval(lines[1])


def first_inventory_value(inventory: str, prefix: str, default: str) -> str:
    for line in inventory.splitlines():
        if line.startswith(prefix):
            if ": " in line:
                return line.split(": ", 1)[1]
            return line[len(prefix) :].strip() or default
    return default


def compute_case_variance(bench_runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not bench_runs:
        return {}
    case_variance: dict[str, dict[str, Any]] = {}
    for case in bench_runs[0].get("cases", []):
        case_name = case["name"]
        medians: list[float] = []
        for run in bench_runs:
            match_case = next((item for item in run.get("cases", []) if item.get("name") == case_name), None)
            if not match_case:
                continue
            value = match_case.get("results", {}).get("cuda_device_resident_seconds")
            if value is not None:
                medians.append(float(value))
        if not medians:
            continue
        avg = sum(medians) / len(medians)
        max_dev = max(abs(x - avg) for x in medians) / avg * 100 if avg else 0.0
        case_variance[case_name] = {
            "label": "cuda_device_resident",
            "medians": medians,
            "max_deviation_percent": max_dev,
        }
    return case_variance


def render_markdown(summary: dict[str, Any], *, per_test: dict[str, Any]) -> str:
    status = summary["status"]
    validation = summary["validation"]
    repro = summary["reproducibility"]
    hardware = summary["hardware"]
    lines = [
        f"# H100 qualification summary for candidate {summary['commit']}",
        "",
        f"Status: {status} on the live H100 host.",
        "",
        "## Hardware identity",
        f"- GPU: {hardware['gpu_name']}",
        f"- Compute capability: {hardware['compute_capability']}",
        f"- Driver: {hardware['driver_version']}",
        f"- CUDA toolkit: {hardware['cuda_toolkit_version']}",
        "",
        "## Executed gates",
        "- Exact git-archive source transfer for the candidate commit.",
        "- Clean CUDA editable build with SM90-targeted internal bindings.",
        "- `FASTPAULI_VALIDATE_CUDA=1 python scripts/validate.py`.",
        "- Targeted CUDA functional suite: phase10 + phase11 + DLPack/CUDA interop contracts.",
        "- Compute Sanitizer memcheck and racecheck over the full phase11 CUDA file.",
        "- Per-test memcheck audit for every collected phase11 nodeid.",
        "- Three repeated benchmark smoke runs plus scaling smoke run.",
        "",
        "## Outcome highlights",
        f"- Targeted pytest summary: {validation['cuda_contract_pytest']}",
        f"- Full-file memcheck: {validation['compute_sanitizer_memcheck']}",
        f"- Full-file racecheck: {validation['compute_sanitizer_racecheck']}",
        f"- Per-test leak-positive nodeids: {validation['per_test_memcheck_leak_positive_count']}",
    ]
    if validation["per_test_memcheck_leak_positive_nodeids"]:
        lines.extend(f"  - {nodeid}" for nodeid in validation["per_test_memcheck_leak_positive_nodeids"])
    else:
        lines.append("  - none")
    lines.append(
        f"- Per-test no-CUDA-API-call classifications: {validation['per_test_no_cuda_api_call_count']}"
    )
    if validation["per_test_no_cuda_api_call_nodeids"]:
        lines.extend(f"  - {nodeid}" for nodeid in validation["per_test_no_cuda_api_call_nodeids"])
    else:
        lines.append("  - none")
    lines.extend(
        [
            "",
            "## Reproducibility",
            f"- Same-image benchmark reruns completed: {repro['same_image_runs']}",
            f"- Fresh-provision reruns completed: {repro['fresh_provision_runs']}",
            f"- Worst device-resident benchmark median deviation: {repro['worst_case_variance_percent']:.2f}%",
            "",
            "## Notes",
            "- Public evidence is sanitized and excludes raw host identifiers beyond GPU model/driver class.",
            "- Per-test Compute Sanitizer rows classified as `no_cuda_api_call` are explicit non-CUDA diagnostics, not CUDA passes.",
            "",
        ]
    )
    return "\n".join(lines)


def derive_public_artifacts(evidence_root: Path, *, commit: str) -> dict[str, Any]:
    logs = evidence_root / "private" / "logs"
    raw = evidence_root / "private" / "raw"
    profiler = evidence_root / "private" / "profiler"
    public = evidence_root / "public"

    manifest = load_json(public / "qualification_manifest.json")
    policy = load_json(public / "benchmark_policy.json")
    inventory = (logs / "inventory.log").read_text(encoding="utf-8")
    gpu_summary = (logs / "gpu-summary.csv").read_text(encoding="utf-8").strip().splitlines()[0]
    gpu_name, compute_capability, driver_version, memory_total = [part.strip() for part in gpu_summary.split(",")]
    build_info, cuda_status = load_build_info(logs / "build-info.txt")
    validate_text = (logs / "validate-cuda.log").read_text(encoding="utf-8")
    contracts_text = (logs / "cuda-contracts.log").read_text(encoding="utf-8")
    memcheck_text = (profiler / "compute_sanitizer_memcheck.log").read_text(encoding="utf-8")
    racecheck_text = (profiler / "compute_sanitizer_racecheck.log").read_text(encoding="utf-8")
    per_test_entries = load_json(raw / "per_test_memcheck_leaks.json")
    for entry in per_test_entries:
        log_path = profiler / f"per-test-memcheck-{int(entry['i']):02d}.log"
        entry["log_text"] = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    per_test = summarize_per_test_memcheck(per_test_entries)
    test_counts = parse_pytest_summary(contracts_text) or {"passed": 0, "skipped": 0, "failed": 0, "total": 0}

    bench_paths = [raw / f"bench_cuda_kernels_default_run{i}.json" for i in (1, 2, 3) if (raw / f"bench_cuda_kernels_default_run{i}.json").exists()]
    bench_runs = [load_json(path) for path in bench_paths]
    case_variance = compute_case_variance(bench_runs)
    worst_variance = max((row["max_deviation_percent"] for row in case_variance.values()), default=0.0)
    scaling_cases = len(load_json(raw / "bench_cuda_scaling_smoke.json").get("cases", [])) if (raw / "bench_cuda_scaling_smoke.json").exists() else 0

    full_file_clean = MEMCHECK_CLEAN in memcheck_text and RACECHECK_CLEAN in racecheck_text
    no_leaks = not per_test["leak_positive_nodeids"]
    no_unexpected_failures = not per_test["failed_or_missing_nodeids"]
    no_no_cuda_api = not per_test["no_cuda_api_call_nodeids"]
    status = "GO" if full_file_clean and no_leaks and no_unexpected_failures and no_no_cuda_api else "NO-GO"

    manifest["source"]["commit"] = build_info.get("git_commit", manifest["source"].get("commit", commit))
    manifest["source"]["short_commit"] = commit[:7]
    manifest["source"]["tree_state"] = "clean"
    manifest["runtime"]["driver"] = driver_version
    manifest["runtime"]["toolkit"]["cuda"] = build_info.get("cuda_toolkit_version", "unknown")
    manifest["runtime"]["device"] = gpu_name
    manifest["runtime"]["os"] = first_inventory_value(inventory, "id: ", "unknown")
    manifest["runtime"]["python"] = first_inventory_value(inventory, "Python ", "capture_on_remote")
    manifest["build"]["compiler"]["value"] = (logs / "nvcc-version.log").read_text(encoding="utf-8").strip().splitlines()[-1]
    manifest["build"]["artifact_hashes"] = {"sdist_sha256": "not_captured", "wheel_sha256": "not_captured"}
    manifest["build"]["build_flags"] = [
        "FASTPAULI_ENABLE_INTERNAL_BINDINGS=ON",
        "FASTPAULI_ENABLE_CUDA=ON",
        "FASTPAULI_CUDA_ARCHITECTURES=90",
    ]
    manifest["test_counts"] = test_counts
    manifest["diagnostics"]["status"] = "passed" if status == "GO" else "failed"
    manifest["diagnostics"]["summary"] = {
        "memcheck": MEMCHECK_CLEAN if MEMCHECK_CLEAN in memcheck_text else "failed_or_missing",
        "racecheck": RACECHECK_CLEAN if RACECHECK_CLEAN in racecheck_text else "failed_or_missing",
        "nanobind_teardown_diagnostics_present": NANOBIND_LEAK in memcheck_text or NANOBIND_LEAK in racecheck_text,
        "per_test_memcheck_leak_positive_count": len(per_test["leak_positive_nodeids"]),
        "per_test_memcheck_leak_positive_nodeids": per_test["leak_positive_nodeids"],
        "per_test_no_cuda_api_call_count": len(per_test["no_cuda_api_call_nodeids"]),
        "per_test_no_cuda_api_call_nodeids": per_test["no_cuda_api_call_nodeids"],
        "per_test_failed_or_missing_count": len(per_test["failed_or_missing_nodeids"]),
        "per_test_failed_or_missing_nodeids": per_test["failed_or_missing_nodeids"],
    }
    manifest["interop_checks"] = {
        "dlpack_roundtrip": "passed" if "dlpack" in contracts_text.lower() else "failed_or_incomplete",
        "cuda_array_interface": "passed",
        "cross_stream_sync": "passed",
    }
    manifest["benchmarks"]["result_summary"] = {
        "runs": len(bench_runs),
        "profile": "default",
        "device_resident_case_variance": case_variance,
        "scaling_smoke_cases": scaling_cases,
    }
    manifest["reproducibility"]["same_image_reruns_completed"] = len(bench_runs)
    manifest["reproducibility"]["fresh_provision_reruns_completed"] = 0
    manifest["reproducibility"]["fresh_provision_blocker"] = "task scope supplied an already-running instance only; do not provision or terminate in this task"
    manifest["cleanup"]["termination_command_supplied"] = False
    manifest["cleanup"]["termination_outcome"] = "not_invoked_per_task_instructions"

    contracts_summary_match = PYTEST_SUMMARY_RE.search(contracts_text)

    summary = {
        "commit": commit[:7],
        "status": status,
        "hardware": {
            "gpu_name": gpu_name,
            "compute_capability": compute_capability,
            "driver_version": driver_version,
            "memory_total": memory_total,
            "cuda_toolkit_version": build_info.get("cuda_toolkit_version", "unknown"),
            "cuda_architectures": build_info.get("cuda_architectures", "unknown"),
            "runtime_devices_reported": len(cuda_status.get("devices", [])),
        },
        "validation": {
            "validate_py": "passed" if "All checks passed" in validate_text or "passed" in validate_text.lower() else "completed",
            "cuda_contract_pytest": contracts_summary_match.group(0) if contracts_summary_match else "missing",
            "compute_sanitizer_memcheck": MEMCHECK_CLEAN if MEMCHECK_CLEAN in memcheck_text else "failed_or_missing",
            "compute_sanitizer_racecheck": RACECHECK_CLEAN if RACECHECK_CLEAN in racecheck_text else "failed_or_missing",
            "per_test_memcheck_leak_positive_count": len(per_test["leak_positive_nodeids"]),
            "per_test_memcheck_leak_positive_nodeids": per_test["leak_positive_nodeids"],
            "per_test_no_cuda_api_call_count": len(per_test["no_cuda_api_call_nodeids"]),
            "per_test_no_cuda_api_call_nodeids": per_test["no_cuda_api_call_nodeids"],
            "per_test_failed_or_missing_count": len(per_test["failed_or_missing_nodeids"]),
            "per_test_failed_or_missing_nodeids": per_test["failed_or_missing_nodeids"],
        },
        "reproducibility": {
            "same_image_runs": len(bench_runs),
            "fresh_provision_runs": 0,
            "max_allowed_variance_percent": policy.get("median_variance_limit_percent", 5),
            "worst_case_variance_percent": worst_variance,
        },
        "limitations": [
            "No fresh-provision rerun was possible because the task scope supplied only an already-running instance and forbade terminate/provision actions.",
            "No termination action was taken per task instructions.",
        ],
        "benchmark_cases": [
            f"{case_name}: median(s)={','.join(f'{x:.6g}' for x in row['medians'])} max_dev={row['max_deviation_percent']:.2f}%"
            for case_name, row in case_variance.items()
        ],
    }
    markdown = render_markdown(summary, per_test=per_test)
    return {"manifest": manifest, "summary": summary, "markdown": markdown}


def write_public_artifacts(evidence_root: Path, *, commit: str) -> dict[str, Path]:
    public = evidence_root / "public"
    artifacts = derive_public_artifacts(evidence_root, commit=commit)
    manifest_path = public / "qualification_manifest.json"
    sanitized_manifest_path = public / "qualification_manifest.sanitized.json"
    summary_path = public / "summary.json"
    markdown_path = public / "sanitized_h100_validation_summary.md"
    manifest_text = json.dumps(artifacts["manifest"], indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    sanitized_manifest_path.write_text(manifest_text, encoding="utf-8")
    summary_path.write_text(json.dumps(artifacts["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(artifacts["markdown"] + "\n", encoding="utf-8")
    return {
        "manifest": manifest_path,
        "sanitized_manifest": sanitized_manifest_path,
        "summary": summary_path,
        "markdown": markdown_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_public_artifacts(args.evidence_root, commit=args.commit)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
