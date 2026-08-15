#!/usr/bin/env python3
"""Render Apple Metal optimization evidence and the broad README landscape."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "docs/benchmarks/data/apple_metal_optimization_campaign8_2026-05-07"
DEFAULT_PLOT_DIR = ROOT / "docs/benchmarks/plots"
PRIOR_LANDSCAPE_SUMMARY = (
    ROOT / "docs/benchmarks/data/rocm_mi300x_campaign7_2026-04-30/summary.json"
)
PRIOR_APPLE_CAMPAIGN4_SUMMARY = (
    ROOT / "docs/benchmarks/data/apple_metal_optimization_campaign4_2026-05-06/summary.json"
)
PRIOR_APPLE_CAMPAIGN3_SUMMARY = (
    ROOT / "docs/benchmarks/data/apple_metal_optimization_campaign3_2026-05-06/summary.json"
)
PRIOR_APPLE_CAMPAIGN2_SUMMARY = (
    ROOT / "docs/benchmarks/data/apple_metal_optimization_campaign2_2026-05-05/summary.json"
)
LANDSCAPE_PLOT = "accelerator_landscape_with_rocm.svg"
LATEST_APPLE_METAL_CAMPAIGN = "apple_metal_optimization_campaign8"

CAMPAIGN_CONFIGS = {
    "apple_metal_optimization_campaign1": {
        "date": "2026-05-05",
        "label": "Apple Metal Campaign 1",
        "limitations": [
            "Campaign 1 refreshes README landscape evidence but does not change public Metal API support.",
            "Compact Metal consumers are still CPU scans over shared Metal storage in this campaign.",
        ],
        "profiler": {
            "status": "retained_from_bringup",
            "source": "docs/benchmarks/data/apple_metal_bringup_2026-05-01/profiler/metal_system_trace_summary.json",
            "remaining": "shader-counter and shader-timeline profiling require a deeper Instruments capture",
        },
    },
    "apple_metal_optimization_campaign2": {
        "date": "2026-05-05",
        "label": "Apple Metal Campaign 2",
        "limitations": [
            "Campaign 2 retains the source-build-only Metal API boundary.",
            "Campaign 2 changes the internal commutation kernel and dispatch shape but does not add raw Metal buffer exports, public queue APIs, Metal wheels, MPSGraph kernels, or PyTorch MPS paths.",
            "Compact Metal consumers are still CPU scans over shared Metal storage in this campaign.",
        ],
        "profiler": {
            "status": "not_recaptured",
            "source": "docs/benchmarks/data/apple_metal_bringup_2026-05-01/profiler/metal_system_trace_summary.json",
            "remaining": "shader-counter and shader-timeline profiling remain the next Instruments evidence target",
        },
    },
    "apple_metal_optimization_campaign3": {
        "date": "2026-05-06",
        "label": "Apple Metal Campaign 3",
        "limitations": [
            "Campaign 3 keeps the source-build-only Metal API boundary.",
            "Campaign 3 private storage, offline metallib loading, and GPU compact reductions are benchmark-only experimental selectors.",
            "MPSGraph and PyTorch MPS are recorded as external baseline statuses, not FastPauli backend identities.",
        ],
        "profiler": {
            "status": "campaign3_inventory_recorded",
            "source": "docs/benchmarks/data/apple_metal_optimization_campaign3_2026-05-06/profiler/metal_campaign3_profiler_evidence.json",
            "remaining": "shader-counter export quality depends on local Instruments template availability",
        },
    },
    "apple_metal_optimization_campaign4": {
        "date": "2026-05-06",
        "label": "Apple Metal Campaign 4",
        "limitations": [
            "Campaign 4 keeps the source-build-only Metal API boundary.",
            "Campaign 4 parallel compact reductions remain benchmark-only experimental selectors.",
            "PyPI publication, Windows support, and older macOS compatibility are out of scope for this Apple Metal optimization slice.",
            "MPSGraph and PyTorch MPS remain external baseline statuses unless an exact sparse Pauli mapping exists.",
        ],
        "profiler": {
            "status": "derived_counter_export_blocked",
            "source": "docs/benchmarks/data/apple_metal_optimization_campaign4_2026-05-06/profiler/metal_campaign4_profiler_evidence.json",
            "remaining": (
                "sanitized derived shader-counter exports remain blocked when local Instruments exports "
                "cannot emit narrow value CSVs; raw trace bundles are not retained"
            ),
        },
    },
    "apple_metal_optimization_campaign5": {
        "date": "2026-05-06",
        "label": "Apple Metal Campaign 5",
        "limitations": [
            "Campaign 5 keeps the source-build-only Metal API boundary.",
            "Campaign 5 retained simplify is a transfer-reference correctness bridge unless a device-resident candidate is proven correct and faster.",
            "Metal statevector expectation, Metal matmul, Metal wheels, PyPI publication, Windows support, and older macOS compatibility are out of scope.",
        ],
        "profiler": {
            "status": "not_recaptured",
            "source": "docs/benchmarks/data/apple_metal_optimization_campaign4_2026-05-06/profiler/metal_campaign4_profiler_evidence.json",
            "remaining": "Campaign 5 simplify profiling should focus on timing-boundary evidence before shader-counter capture.",
        },
    },
    "apple_metal_optimization_campaign6": {
        "date": "2026-05-07",
        "label": "Apple Metal Campaign 6",
        "limitations": [
            "Campaign 6 keeps the source-build-only Metal API boundary.",
            "Campaign 6 retains a private MetalWorkspace allocation model and benchmark status row for device-resident simplify groundwork.",
            "The Campaign 6 device-resident simplify candidate remains blocked until checked Metal sort/prefix/reduce primitives exist.",
            "The public Metal simplify behavior remains the Campaign 5 transfer-reference correctness bridge.",
        ],
        "profiler": {
            "status": "not_recaptured",
            "source": "docs/benchmarks/data/apple_metal_optimization_campaign4_2026-05-06/profiler/metal_campaign4_profiler_evidence.json",
            "remaining": "Simplify shader-counter capture remains blocked until a retained device-resident simplify kernel exists.",
        },
    },
    "apple_metal_optimization_campaign7": {
        "date": "2026-05-07",
        "label": "Apple Metal Campaign 7",
        "limitations": [
            "Campaign 7 keeps the source-build-only Metal API boundary.",
            "Campaign 7 adds a benchmark-only checked one-word Metal simplify primitive stack.",
            "The Campaign 7 device candidate is limited to coefficients exactly representable as signed fixed32 dyadic values whose accumulated sums and tolerance threshold fit exact uint64 squared-magnitude comparison because Apple Metal does not support double arithmetic in kernels.",
            "The public Metal simplify behavior remains the Campaign 5 transfer-reference correctness bridge unless a future slice promotes a broader retained implementation.",
        ],
        "profiler": {
            "status": "not_recaptured",
            "source": "docs/benchmarks/data/apple_metal_optimization_campaign4_2026-05-06/profiler/metal_campaign4_profiler_evidence.json",
            "remaining": "Shader-counter capture should follow only after a retained general Metal simplify kernel exists.",
        },
    },
    "apple_metal_optimization_campaign8": {
        "date": "2026-05-07",
        "label": "Apple Metal Campaign 8",
        "limitations": [
            "Campaign 8 keeps the source-build-only Metal API boundary.",
            "Campaign 8 adds timing decomposition and cache-boundary evidence for the benchmark-only checked one-word Metal simplify primitive stack.",
            "The public Metal simplify behavior remains the Campaign 5 transfer-reference correctness bridge unless a later design proves a broader retained implementation.",
            "Campaign 8 does not claim general FP64, multi-word, Metal wheel, MPSGraph, or PyTorch MPS sparse-Pauli simplify support.",
        ],
        "profiler": {
            "status": "timing_decomposition_recorded",
            "source": "private_hook_internal_steady_clock_fields_in_campaign8_rows",
            "remaining": "Shader-counter capture should follow only after a lower-pass candidate or public lifetime boundary exists.",
        },
    },
}

VARIANT_SERIES = {
    "cpu_default": "CPU default",
    "cpu_scalar": "CPU scalar",
    "cpu_neon": "CPU NEON",
    "metal_transfer_inclusive": "Apple Metal transfer-inclusive",
    "metal_device_resident": "Apple Metal device-resident host output",
    "metal_device_matrix": "Apple Metal device matrix allocate",
    "metal_device_matrix_reuse": "Apple Metal device matrix reuse",
    "metal_device_matrix_reuse_auto_ab": "Apple Metal auto A/B reuse",
    "metal_device_matrix_reuse_generic2d_baseline": "Apple Metal generic 2D baseline",
    "metal_device_matrix_reuse_flat_generic_baseline": "Apple Metal flat generic baseline",
    "metal_device_matrix_reuse_words1_candidate": "Apple Metal words=1 specialized candidate",
    "metal_device_matrix_reuse_words2_candidate": "Apple Metal words=2 specialized candidate",
    "metal_device_matrix_reuse_metallib_auto": "Apple Metal `.metallib` reuse",
    "metal_private_blit_host_output": "Apple Metal private blit host output",
    "metal_device_matrix_to_host": "Apple Metal device matrix to_host",
    "metal_compact_consumer": "Apple Metal compact count",
    "metal_compact_count_axis0": "Apple Metal compact column counts",
    "metal_compact_count_axis1": "Apple Metal compact row counts",
    "metal_compact_consumer_gpu_total": "Apple Metal GPU compact count",
    "metal_compact_consumer_gpu_parallel_total": "Apple Metal GPU parallel compact count",
    "metal_compact_count_axis0_gpu": "Apple Metal GPU compact column counts",
    "metal_compact_count_axis1_gpu": "Apple Metal GPU compact row counts",
    "metal_simplify_transfer_reference": "Apple Metal simplify transfer reference",
    "metal_simplify_device_candidate": "Apple Metal simplify device candidate",
    "metal_simplify_workspace_probe": "Apple Metal simplify workspace probe",
}
REQUIRED_LANDSCAPE_SERIES = {
    "CPU scalar",
    "CPU default",
    "CPU NEON",
    "Apple Metal transfer-inclusive",
    "Apple Metal device-resident host output",
    "Apple Metal device matrix allocate",
    "Apple Metal device matrix reuse",
    "Apple Metal device matrix to_host",
    "Apple Metal compact count",
}
CAMPAIGN2_EXTRA_LANDSCAPE_SERIES = {
    "Apple Metal auto A/B reuse",
    "Apple Metal generic 2D baseline",
    "Apple Metal flat generic baseline",
    "Apple Metal words=1 specialized candidate",
    "Apple Metal words=2 specialized candidate",
}
CAMPAIGN3_EXTRA_LANDSCAPE_SERIES = {
    *CAMPAIGN2_EXTRA_LANDSCAPE_SERIES,
    "Apple Metal `.metallib` reuse",
    "Apple Metal private blit host output",
    "Apple Metal GPU compact count",
    "Apple Metal GPU compact column counts",
    "Apple Metal GPU compact row counts",
}
CAMPAIGN4_EXTRA_LANDSCAPE_SERIES = {
    *CAMPAIGN3_EXTRA_LANDSCAPE_SERIES,
    "Apple Metal GPU parallel compact count",
}
CAMPAIGN5_EXTRA_LANDSCAPE_SERIES = {
    *CAMPAIGN4_EXTRA_LANDSCAPE_SERIES,
    "Apple Metal simplify transfer reference",
}
CAMPAIGN6_EXTRA_LANDSCAPE_SERIES = {
    *CAMPAIGN5_EXTRA_LANDSCAPE_SERIES,
}
CAMPAIGN7_EXTRA_LANDSCAPE_SERIES = {
    *CAMPAIGN6_EXTRA_LANDSCAPE_SERIES,
    "Apple Metal simplify device candidate",
}
CAMPAIGN8_EXTRA_LANDSCAPE_SERIES = {
    *CAMPAIGN7_EXTRA_LANDSCAPE_SERIES,
}
REQUIRED_BROAD_LANDSCAPE_TOKENS = ("CPU", "CUDA", "ROCm", "CuPy", "Apple Metal")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def load_latest_benchmark(data_dir: Path) -> tuple[Path, dict[str, Any]]:
    raw_dir = data_dir / "raw"
    candidates = sorted(raw_dir.glob("metal_benchmark_*.json"))
    if not candidates:
        raise FileNotFoundError(f"no Apple Metal benchmark JSON found under {raw_dir}")
    selected = candidates[-1]
    payload = read_json(selected)
    if payload.get("benchmark") != "apple_metal_kernels":
        raise ValueError(f"unexpected benchmark payload in {selected}")
    return selected, payload


def timed_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload.get("cases", []):
        if not isinstance(row, dict):
            continue
        timing = row.get("timing")
        if row.get("status") == "ok" and isinstance(timing, dict):
            median = timing.get("median")
            if isinstance(median, (int, float)) and median > 0:
                rows.append(row)
    return rows


def benchmark_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload.get("cases", []):
        timing = row.get("timing")
        item = {
            "case": row.get("case", {}).get("name", "unknown"),
            "median_seconds": timing.get("median") if isinstance(timing, dict) else None,
            "object_backend": row.get("object_backend"),
            "operation": row.get("operation"),
            "status": row.get("status"),
            "transfer_boundary": row.get("transfer_boundary"),
            "variant": row.get("variant"),
            "metal_execution": row.get("metal_execution"),
        }
        if row.get("operation") == "simplify":
            for key in (
                "atol",
                "correct",
                "duplicate_rate",
                "metal_simplify_strategy",
                "metal_simplify_strategy_status",
                "metal_simplify_coefficient_domain",
                "metal_simplify_primitive_stack",
                "metal_simplify_workspace_model",
                "bitonic_passes",
                "num_terms",
                "output_terms",
                "padded_terms",
                "prefix_sum_passes",
                "rtol",
                "workspace_reserved_bytes",
                "campaign8_timing_schema",
                "timing_decomposition_seconds",
                "dispatch_counts",
                "pipeline_cache",
                "performance_decision",
            ):
                if row.get(key) is not None:
                    item[key] = row[key]
        rows.append(item)
    return rows


def load_prior_landscape(campaign: str) -> list[dict[str, Any]]:
    if campaign == "apple_metal_optimization_campaign8":
        campaign7_summary = (
            ROOT / "docs/benchmarks/data/apple_metal_optimization_campaign7_2026-05-07/summary.json"
        )
        if campaign7_summary.exists():
            prior = read_json(campaign7_summary).get("readme_performance_landscape", [])
            return [row for row in prior if isinstance(row, dict)]
    if campaign == "apple_metal_optimization_campaign7":
        campaign6_summary = (
            ROOT / "docs/benchmarks/data/apple_metal_optimization_campaign6_2026-05-07/summary.json"
        )
        if campaign6_summary.exists():
            prior = read_json(campaign6_summary).get("readme_performance_landscape", [])
            return [row for row in prior if isinstance(row, dict)]
    if campaign == "apple_metal_optimization_campaign6":
        campaign5_summary = (
            ROOT / "docs/benchmarks/data/apple_metal_optimization_campaign5_2026-05-06/summary.json"
        )
        if campaign5_summary.exists():
            prior = read_json(campaign5_summary).get("readme_performance_landscape", [])
            return [row for row in prior if isinstance(row, dict)]
    if campaign == "apple_metal_optimization_campaign5" and PRIOR_APPLE_CAMPAIGN4_SUMMARY.exists():
        prior = read_json(PRIOR_APPLE_CAMPAIGN4_SUMMARY).get("readme_performance_landscape", [])
        return [row for row in prior if isinstance(row, dict)]
    if campaign == "apple_metal_optimization_campaign4" and PRIOR_APPLE_CAMPAIGN3_SUMMARY.exists():
        prior = read_json(PRIOR_APPLE_CAMPAIGN3_SUMMARY).get("readme_performance_landscape", [])
        return [row for row in prior if isinstance(row, dict)]
    if campaign == "apple_metal_optimization_campaign3" and PRIOR_APPLE_CAMPAIGN2_SUMMARY.exists():
        prior = read_json(PRIOR_APPLE_CAMPAIGN2_SUMMARY).get("readme_performance_landscape", [])
        return [row for row in prior if isinstance(row, dict)]
    if not PRIOR_LANDSCAPE_SUMMARY.exists():
        return []
    prior = read_json(PRIOR_LANDSCAPE_SUMMARY).get("readme_performance_landscape", [])
    return [row for row in prior if isinstance(row, dict)]


def infer_campaign(data_dir: Path) -> str:
    name = data_dir.name
    if "apple_metal_optimization_campaign8" in name:
        return "apple_metal_optimization_campaign8"
    if "apple_metal_optimization_campaign7" in name:
        return "apple_metal_optimization_campaign7"
    if "apple_metal_optimization_campaign6" in name:
        return "apple_metal_optimization_campaign6"
    if "apple_metal_optimization_campaign5" in name:
        return "apple_metal_optimization_campaign5"
    if "apple_metal_optimization_campaign4" in name:
        return "apple_metal_optimization_campaign4"
    if "apple_metal_optimization_campaign3" in name:
        return "apple_metal_optimization_campaign3"
    if "apple_metal_optimization_campaign2" in name:
        return "apple_metal_optimization_campaign2"
    if "apple_metal_optimization_campaign1" in name:
        return "apple_metal_optimization_campaign1"
    raise ValueError(f"cannot infer Apple Metal campaign from data directory: {data_dir}")


def campaign_config(campaign: str) -> dict[str, Any]:
    try:
        return CAMPAIGN_CONFIGS[campaign]
    except KeyError as exc:
        raise ValueError(f"unsupported Apple Metal campaign: {campaign}") from exc


def apple_case_scale(case: dict[str, Any], row: dict[str, Any]) -> str:
    if row.get("operation") == "simplify":
        return (
            f'{case.get("num_terms")} terms before simplify, '
            f'{case.get("output_terms", "unknown")} survivor terms, '
            f'duplicate_rate={case.get("duplicate_rate", "unknown")}, '
            f'{case.get("num_qubits")} qubits, {case.get("packed_words")} words'
        )
    return (
        f'{case.get("lhs_terms")}x{case.get("rhs_terms")} terms, '
        f'{case.get("num_qubits")} qubits, {case.get("packed_words")} words'
    )


def apple_landscape_rows(payload: dict[str, Any], *, campaign_label: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in timed_rows(payload):
        case = row.get("case", {})
        if not isinstance(case, dict):
            continue
        name = str(case.get("name", "unknown"))
        item = grouped.setdefault(
            name,
            {
                "category": campaign_label,
                "scale": apple_case_scale(case, row),
                "mode": name,
                "final_status": row.get("status", "ok"),
                "gpu_name": payload.get("metal_status", {}).get("metal_device_name", "Apple Metal"),
                "points": [],
            },
        )
        variant = str(row.get("variant", ""))
        series = VARIANT_SERIES.get(variant)
        timing = row.get("timing", {})
        seconds = timing.get("median") if isinstance(timing, dict) else None
        if series is not None and isinstance(seconds, (int, float)) and seconds > 0:
            item["points"].append({"series": series, "seconds": float(seconds)})
    return list(grouped.values())


def required_landscape_series(campaign: str) -> set[str]:
    required = set(REQUIRED_LANDSCAPE_SERIES)
    if campaign == "apple_metal_optimization_campaign2":
        required.update(CAMPAIGN2_EXTRA_LANDSCAPE_SERIES)
    if campaign == "apple_metal_optimization_campaign3":
        required.update(CAMPAIGN3_EXTRA_LANDSCAPE_SERIES)
    if campaign == "apple_metal_optimization_campaign4":
        required.update(CAMPAIGN4_EXTRA_LANDSCAPE_SERIES)
    if campaign == "apple_metal_optimization_campaign5":
        required.update(CAMPAIGN5_EXTRA_LANDSCAPE_SERIES)
    if campaign == "apple_metal_optimization_campaign6":
        required.update(CAMPAIGN6_EXTRA_LANDSCAPE_SERIES)
    if campaign == "apple_metal_optimization_campaign7":
        required.update(CAMPAIGN7_EXTRA_LANDSCAPE_SERIES)
    if campaign == "apple_metal_optimization_campaign8":
        required.update(CAMPAIGN8_EXTRA_LANDSCAPE_SERIES)
    return required


def validate_summary(summary: dict[str, Any], *, expected_campaign: str) -> None:
    if summary.get("campaign") != expected_campaign:
        raise ValueError(f"summary campaign must be {expected_campaign}")
    if summary.get("status") != "ok":
        raise ValueError("Apple Metal summary must have status ok")
    source = ROOT / str(summary.get("source_benchmark", ""))
    if not source.exists():
        raise FileNotFoundError(f"summary source benchmark is missing: {source}")
    if not summary.get("benchmark_rows"):
        raise ValueError("summary must contain benchmark_rows")
    flat_rows = flat_landscape_rows(summary.get("readme_performance_landscape", []))
    if not flat_rows:
        raise ValueError("Apple Metal landscape has no plottable rows")
    landscape_text = " ".join(f'{row.get("label", "")} {row.get("series", "")}' for row in flat_rows)
    missing_tokens = [token for token in REQUIRED_BROAD_LANDSCAPE_TOKENS if token not in landscape_text]
    if missing_tokens:
        raise ValueError(f"broad accelerator landscape is missing rows for: {missing_tokens}")
    series = {row.get("series") for row in flat_rows}
    missing = sorted(required_landscape_series(expected_campaign) - series)
    if missing:
        raise ValueError(f"Apple Metal landscape is missing series: {missing}")


def build_summary(data_dir: Path) -> dict[str, Any]:
    campaign = infer_campaign(data_dir)
    config = campaign_config(campaign)
    benchmark_path, payload = load_latest_benchmark(data_dir)
    apple_rows = apple_landscape_rows(payload, campaign_label=str(config["label"]))
    summary = {
        "campaign": campaign,
        "date": config["date"],
        "status": payload.get("status", "unknown"),
        "profile": payload.get("profile", "unknown"),
        "git_commit": payload.get("git_commit", "unknown"),
        "source_benchmark": rel(benchmark_path),
        "benchmark_rows": benchmark_rows(payload),
        "metal_status": payload.get("metal_status", {}),
        "environment": payload.get("environment", {}),
        "limitations": [
            *payload.get("limitations", []),
            *config["limitations"],
        ],
        "profiler": config["profiler"],
        "plots": {
            "landscape": f"docs/benchmarks/plots/{LANDSCAPE_PLOT}",
        },
        "readme_performance_landscape": [*load_prior_landscape(campaign), *apple_rows],
    }
    if "external_baselines" in payload:
        summary["external_baselines"] = payload["external_baselines"]
    if "git_provenance" in payload:
        summary["git_provenance"] = payload["git_provenance"]
    if "offline_metallib" in payload:
        summary["offline_metallib"] = payload["offline_metallib"]
    validate_summary(summary, expected_campaign=campaign)
    return summary


def flat_landscape_rows(landscape: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in landscape:
        direct_seconds = item.get("seconds")
        if isinstance(direct_seconds, (int, float)) and direct_seconds > 0:
            label = str(item.get("label", item.get("series", "unknown")))
            rows.append(
                {
                    "color": str(item.get("color", row_color(label))),
                    "label": label,
                    "seconds": float(direct_seconds),
                    "series": str(item.get("series", label)),
                }
            )
            continue
        for point in item.get("points", []):
            seconds = point.get("seconds")
            if not isinstance(seconds, (int, float)) or seconds <= 0:
                continue
            series = str(point.get("series", ""))
            rows.append(
                {
                    "color": row_color(f'{item.get("gpu_name", "")} {series}'),
                    "label": (
                        f'{item.get("gpu_name", "")} / {item.get("mode", "")}: '
                        f"{series}"
                    ),
                    "seconds": float(seconds),
                    "series": series,
                }
            )
    return sorted(rows, key=lambda row: row["seconds"])


def row_color(label: str) -> str:
    if "Apple Metal" in label or "Apple M" in label:
        return "#0f766e"
    if "ROCm" in label or "HIP" in label or "MI300X" in label:
        return "#dc2626"
    if "CUDA" in label or "H100" in label or "A100" in label or "RTX" in label:
        return "#2563eb"
    if "CPU" in label:
        return "#059669"
    if "DLPack" in label or "CuPy" in label or "PyTorch" in label:
        return "#7c3aed"
    return "#64748b"


def render_landscape_svg(landscape: list[dict[str, Any]]) -> str:
    rows = flat_landscape_rows(landscape)
    width = 1320
    row_h = 24
    height = 118 + max(1, len(rows)) * row_h
    min_value = min((row["seconds"] for row in rows), default=1.0)
    max_value = max((row["seconds"] for row in rows), default=1.0)
    lo = math.log10(min_value)
    hi = math.log10(max_value)
    span = max(hi - lo, 1e-9)
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text id="title" x="36" y="44" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="24" font-weight="700" fill="#0f172a">Wolfgang accelerator performance landscape</text>',
        '<text id="desc" x="36" y="70" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="13" fill="#475569">Broad checked evidence across CPU, CUDA, ROCm/HIP, Apple Metal, and available external baseline rows. Bars use log-scaled seconds; labels keep timing boundaries explicit.</text>',
        '<text x="36" y="92" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="11" fill="#64748b">Source: checked benchmark JSON summaries under docs/benchmarks/data; latest Apple Metal rows from local Apple M4 Pro source-build evidence.</text>',
    ]
    if not rows:
        body.append(
            '<text x="36" y="126" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="13" fill="#64748b">No rows available.</text>'
        )
    for index, row in enumerate(rows):
        y = 124 + index * row_h
        value = row["seconds"]
        scaled = (math.log10(value) - lo) / span
        bar_w = 24 + 356 * scaled
        label = row["label"]
        if len(label) > 118:
            label = label[:115] + "..."
        body.extend(
            [
                f'<text x="36" y="{y}" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="10.5" fill="#0f172a">{escape(label)}</text>',
                f'<rect x="850" y="{y - 12}" width="384" height="10" rx="3" fill="#e2e8f0"/>',
                f'<rect x="850" y="{y - 12}" width="{bar_w:.2f}" height="10" rx="3" fill="{escape(row.get("color", row_color(row["label"])))}"/>',
                f'<text x="1244" y="{y - 3}" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="10" fill="#334155">{value:.4g}s</text>',
            ]
        )
    body.append("</svg>")
    return "\n".join(body)


def generated_summary_text(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True) + "\n"


def load_checked_summary(data_dir: Path) -> dict[str, Any]:
    summary_path = data_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing retained public summary: {summary_path}")
    summary = read_json(summary_path)
    landscape = summary.get("readme_performance_landscape")
    if not isinstance(landscape, list) or not landscape:
        raise ValueError(f"retained public summary missing landscape rows: {summary_path}")
    return summary


def validate_plot_text(text: str) -> None:
    missing = [token for token in REQUIRED_BROAD_LANDSCAPE_TOKENS if token not in text]
    if missing:
        raise ValueError(f"broad accelerator landscape plot is missing tokens: {missing}")


def render_assets(data_dir: Path, plot_dir: Path) -> dict[str, Any]:
    try:
        summary = build_summary(data_dir)
        write_text(data_dir / "summary.json", generated_summary_text(summary))
    except FileNotFoundError:
        summary = load_checked_summary(data_dir)
    plot_text = render_landscape_svg(summary["readme_performance_landscape"])
    write_text(plot_dir / LANDSCAPE_PLOT, plot_text)
    validate_plot_text(plot_text)
    return summary


def validate_checked_assets(data_dir: Path, plot_dir: Path) -> None:
    summary_path = data_dir / "summary.json"
    plot_path = plot_dir / LANDSCAPE_PLOT
    try:
        expected_summary = build_summary(data_dir)
    except FileNotFoundError:
        expected_summary = load_checked_summary(data_dir)
    checked_summary_text = summary_path.read_text(encoding="utf-8")
    expected_summary_text = generated_summary_text(expected_summary)
    if checked_summary_text != expected_summary_text:
        raise ValueError(f"{rel(summary_path)} is stale; rerun scripts/render_apple_metal_assets.py")
    expected_plot_text = render_landscape_svg(expected_summary["readme_performance_landscape"]) + "\n"
    checked_plot_text = plot_path.read_text(encoding="utf-8")
    if checked_plot_text != expected_plot_text:
        if expected_summary["campaign"] == LATEST_APPLE_METAL_CAMPAIGN:
            raise ValueError(f"{rel(plot_path)} is stale; rerun scripts/render_apple_metal_assets.py")
        validate_plot_text(checked_plot_text)
    validate_plot_text(checked_plot_text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        summary = build_summary(args.data_dir)
        validate_checked_assets(args.data_dir, args.plot_dir)
        label = campaign_config(str(summary["campaign"]))["label"]
        print(f"{label} assets validated")
    else:
        summary = render_assets(args.data_dir, args.plot_dir)
        print(
            json.dumps(
                {
                    "campaign": summary["campaign"],
                    "status": summary["status"],
                    "benchmark_rows": len(summary["benchmark_rows"]),
                    "plot": summary["plots"]["landscape"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
