#!/usr/bin/env python3
"""Render Campaign 9 deferred-headroom summaries and plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29"
DEFAULT_PLOT_DIR = ROOT / "docs/benchmarks/plots"
PORTABILITY_SUMMARY = (
    ROOT
    / "docs/benchmarks/data/cuda_portability_campaign9_non_h100_nvidia_2026-04-29/summary.json"
)
CAMPAIGN8_SUMMARY = (
    ROOT / "docs/benchmarks/data/cuda_deep_optimization_h100_campaign8_2026-04-29/summary.json"
)

PLOT_FILES = {
    "status": "cuda_campaign9_deferred_headroom_status.svg",
    "ncu": "cuda_campaign9_privileged_ncu.svg",
    "portability": "cuda_campaign9_portability.svg",
    "landscape": "cuda_campaign9_performance_landscape.svg",
}

HEADROOM_LABELS = {
    1: "Non-H100 NVIDIA portability",
    2: "Privileged Nsight Compute counters",
    3: "Public fused grouping API",
    4: "DLPack interop",
    5: "Stream / CUDA Graph replay",
    6: "CSR scatter reopen",
}

STATUS_COLORS = {
    "implemented": "#15803d",
    "passed": "#15803d",
    "rejected_with_evidence": "#b45309",
    "blocked_external": "#64748b",
    "accepted": "#0369a1",
    "failed": "#b91c1c",
    "not_applicable": "#94a3b8",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def load_raw_cases(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((data_dir / "raw").glob("*.json")):
        if path.name.endswith("_smoke.json"):
            continue
        payload = read_json(path)
        for case in payload.get("cases", []):
            if case.get("campaign") == "cuda_deferred_headroom_campaign9":
                row = dict(case)
                try:
                    row["source_file"] = str(path.resolve().relative_to(ROOT))
                except ValueError:
                    row["source_file"] = str(
                        DEFAULT_DATA_DIR.relative_to(ROOT) / "raw" / path.name
                    )
                rows.append(row)
    return rows


def assert_non_deferred(rows: list[dict[str, Any]], portability: dict[str, Any]) -> None:
    statuses = [row.get("final_status") for row in rows]
    statuses.append(portability.get("final_status"))
    if "deferred" in statuses:
        raise SystemExit("Campaign 9 summary may not contain final_status='deferred'")
    items = {int(row["campaign8_headroom_item"]) for row in rows}
    items.add(int(portability["campaign8_headroom_item"]))
    if items != set(HEADROOM_LABELS):
        raise SystemExit(f"Campaign 9 summary does not cover all headroom items: {sorted(items)}")
    for row in rows:
        if row.get("deferred_status_allowed") is not False:
            raise SystemExit(f"Campaign 9 row allows deferred status: {row.get('source_file')}")


def decision_rows(rows: list[dict[str, Any]], portability: dict[str, Any]) -> list[dict[str, Any]]:
    by_item: dict[int, dict[str, Any]] = {
        int(portability["campaign8_headroom_item"]): {
            "campaign8_headroom_item": int(portability["campaign8_headroom_item"]),
            "label": HEADROOM_LABELS[int(portability["campaign8_headroom_item"])],
            "mode": "non_h100_portability",
            "final_status": portability["final_status"],
            "decision_doc": portability["report"],
            "evidence": "concrete access check",
        }
    }
    for row in rows:
        item = int(row["campaign8_headroom_item"])
        if item not in by_item:
            by_item[item] = {
                "campaign8_headroom_item": item,
                "label": HEADROOM_LABELS[item],
                "mode": row["mode"],
                "final_status": row["final_status"],
                "decision_doc": row["decision_doc"],
                "evidence": row["source_file"],
            }
    return [by_item[index] for index in sorted(HEADROOM_LABELS)]


def parse_ncu_summary(data_dir: Path) -> list[dict[str, Any]]:
    csv_path = data_dir / "profiler/ncu_campaign9_compact_consumers_details.csv"
    if not csv_path.exists():
        return []
    wanted = {
        "fastpauli::<unnamed>::commutation_kernel": "commutation fill",
        "fastpauli::<unnamed>::count_row_conflicts_kernel": "row conflicts",
        "fastpauli::<unnamed>::count_col_conflicts_kernel": "column conflicts",
        "fastpauli::<unnamed>::scatter_csr_conflicts_sorted_by_row_kernel": "CSR scatter baseline",
    }
    metrics: dict[str, dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            kernel = row["Kernel Name"]
            label = next((value for prefix, value in wanted.items() if kernel.startswith(prefix)), None)
            if label is None:
                continue
            bucket = metrics.setdefault(label, {})
            name = row["Metric Name"]
            if name in {
                "Duration",
                "Compute (SM) Throughput",
                "Memory Throughput",
                "L2 Cache Throughput",
            } and name not in bucket:
                bucket[name] = row["Metric Value"]
    return [
        {
            "kernel": label,
            "duration_us": float(values.get("Duration", "0").replace(",", "")),
            "compute_sm_pct": float(values.get("Compute (SM) Throughput", "0").replace(",", "")),
            "memory_pct": float(values.get("Memory Throughput", "0").replace(",", "")),
            "l2_pct": float(values.get("L2 Cache Throughput", "0").replace(",", "")),
        }
        for label, values in metrics.items()
    ]


def performance_landscape(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prior = read_json(CAMPAIGN8_SUMMARY).get("readme_performance_landscape", [])
    landscape: list[dict[str, Any]] = [
        {
            "category": f"Campaign 8 reference: {item.get('category', '')}".strip(),
            "scale": item.get("scale", ""),
            "mode": item.get("operation", item.get("label", "campaign8_reference")),
            "final_status": "reference",
            "points": item.get("points", []),
        }
        for item in prior
    ]
    for row in rows:
        if row.get("boundary") == "profiler_only" or row.get("mode") == "privileged_ncu":
            continue
        results = row["results"]
        points: list[dict[str, Any]] = []
        candidates = {
            "CPU scalar": "cpu_scalar_seconds",
            "CPU default": "cpu_default_seconds",
            "CPU optimized": "cpu_optimized_seconds",
            "CUDA transfer-inclusive": "cuda_transfer_inclusive_seconds",
            "CUDA device-resident": "cuda_device_resident_seconds",
            "CUDA Campaign 8 graph compact": "campaign8_device_resident_graph_compact_seconds",
            "CUDA Campaign 8 grouping compact": "campaign8_device_grouping_consumer_seconds",
            "CUDA CSR export baseline": "campaign7_csr_graph_export_seconds",
            "CUDA conflict degrees total": "conflict_degrees_axis_none_seconds",
            "CUDA conflict degrees rows": "conflict_degrees_axis_1_seconds",
            "CuPy CUDA Array Interface": "cupy_asarray_export_seconds",
            "CuPy DLPack": "cupy_dlpack_from_dlpack_seconds",
        }
        for series, key in candidates.items():
            value = results.get(key)
            if isinstance(value, (int, float)):
                points.append({"series": series, "seconds": float(value)})
        cpu_timings = results.get("cpu_optimized_timings", {})
        if isinstance(cpu_timings, dict):
            selector_labels = {
                "tbb": "CPU TBB",
                "avx2": "CPU AVX2",
                "avx512": "CPU AVX-512",
                "neon": "CPU NEON",
                "sve": "CPU SVE",
            }
            for selector, payload in sorted(cpu_timings.items()):
                if isinstance(payload, dict) and isinstance(payload.get("seconds"), (int, float)):
                    points.append(
                        {
                            "series": selector_labels.get(selector, f"CPU {selector}"),
                            "seconds": float(payload["seconds"]),
                        }
                    )
        if points:
            landscape.append(
                {
                    "category": "Campaign 9 deferred-headroom closure",
                    "scale": row["scale"],
                    "mode": row["mode"],
                    "final_status": row["final_status"],
                    "points": points,
                }
            )
    return landscape


def render_status_svg(
    decisions: list[dict[str, Any]],
    *,
    title: str = "Campaign 9 closes every Campaign 8 deferred item",
    subtitle: str = "Final statuses are evidence-bearing and none may be deferred.",
) -> str:
    width = 980
    row_h = 58
    height = 96 + row_h * len(decisions)
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="36" y="44" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="24" font-weight="700" fill="#0f172a">{escape(title)}</text>',
        f'<text x="36" y="70" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="13" fill="#475569">{escape(subtitle)}</text>',
    ]
    for index, row in enumerate(decisions):
        y = 100 + index * row_h
        color = STATUS_COLORS.get(row["final_status"], "#64748b")
        body.extend(
            [
                f'<rect x="36" y="{y - 26}" width="908" height="42" rx="6" fill="#ffffff" stroke="#dbe3ef"/>',
                f'<circle cx="58" cy="{y - 5}" r="7" fill="{color}"/>',
                f'<text x="78" y="{y - 10}" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="14" font-weight="700" fill="#0f172a">{escape(row["label"])}</text>',
                f'<text x="78" y="{y + 8}" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12" fill="#64748b">{escape(row["mode"])} · {escape(row["final_status"])}</text>',
            ]
        )
    body.append("</svg>")
    return "\n".join(body)


def render_bar_svg(title: str, subtitle: str, rows: list[dict[str, Any]], *, value_key: str, unit: str) -> str:
    width = 980
    row_h = 58
    height = 104 + max(1, len(rows)) * row_h
    max_value = max((float(row.get(value_key, 0.0)) for row in rows), default=1.0) or 1.0
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="36" y="44" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="24" font-weight="700" fill="#0f172a">{escape(title)}</text>',
        f'<text x="36" y="70" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="13" fill="#475569">{escape(subtitle)}</text>',
    ]
    if not rows:
        body.append('<text x="36" y="118" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="14" fill="#64748b">No rows available.</text>')
    for index, row in enumerate(rows):
        y = 108 + index * row_h
        value = float(row.get(value_key, 0.0))
        bar_w = 560 * value / max_value
        body.extend(
            [
                f'<text x="36" y="{y}" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="13" font-weight="700" fill="#0f172a">{escape(row.get("label", row.get("kernel", row.get("series", ""))))}</text>',
                f'<rect x="300" y="{y - 14}" width="560" height="14" rx="3" fill="#e2e8f0"/>',
                f'<rect x="300" y="{y - 14}" width="{bar_w:.2f}" height="14" rx="3" fill="#2563eb"/>',
                f'<text x="874" y="{y - 3}" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12" fill="#334155">{value:.4g} {escape(unit)}</text>',
            ]
        )
    body.append("</svg>")
    return "\n".join(body)


def render_portability_svg(portability: dict[str, Any]) -> str:
    return render_status_svg(
        [
            {
                "label": "Named non-H100 NVIDIA source build",
                "mode": portability.get("host_label", "not_available"),
                "final_status": portability["final_status"],
            }
        ],
        title="Campaign 9 non-H100 NVIDIA portability",
        subtitle="Concrete access evidence is recorded separately from H100 source-build evidence.",
    )


def render_landscape_svg(landscape: list[dict[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
    for item in landscape:
        for point in item["points"]:
            seconds = point.get("seconds")
            if not isinstance(seconds, (int, float)):
                continue
            rows.append(
                {
                    "label": f'{item.get("category", "evidence")} / {item["mode"]}: {point["series"]}',
                    "seconds": float(seconds),
                }
            )
    rows = sorted(rows, key=lambda row: row["seconds"])
    if not rows:
        return render_bar_svg(
            "Campaign 9 broad performance landscape",
            "No checked rows were available.",
            [],
            value_key="seconds",
            unit="s",
        )

    width = 1280
    row_h = 28
    height = 110 + row_h * len(rows)
    min_value = min(row["seconds"] for row in rows if row["seconds"] > 0)
    max_value = max(row["seconds"] for row in rows)
    lo = math.log10(min_value)
    hi = math.log10(max_value)
    span = max(hi - lo, 1e-9)
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="36" y="44" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="24" font-weight="700" fill="#0f172a">Campaign 9 broad performance landscape</text>',
        '<text x="36" y="70" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="13" fill="#475569">Checked CPU selectors, CUDA transfer/device-resident paths, compact consumers, framework consumers, CSR baselines, and external baselines. Bars use log-scaled seconds.</text>',
    ]
    for index, row in enumerate(rows):
        y = 104 + index * row_h
        value = row["seconds"]
        scaled = (math.log10(max(value, min_value)) - lo) / span
        bar_w = 120 + 500 * scaled
        label = row["label"]
        if len(label) > 108:
            label = label[:105] + "..."
        fill = "#2563eb"
        if "External baseline" in row["label"]:
            fill = "#64748b"
        elif "CPU" in row["label"]:
            fill = "#059669"
        elif "DLPack" in row["label"] or "CuPy" in row["label"]:
            fill = "#7c3aed"
        body.extend(
            [
                f'<text x="36" y="{y}" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="11" fill="#0f172a">{escape(label)}</text>',
                f'<rect x="820" y="{y - 12}" width="380" height="11" rx="3" fill="#e2e8f0"/>',
                f'<rect x="820" y="{y - 12}" width="{bar_w:.2f}" height="11" rx="3" fill="{fill}"/>',
                f'<text x="1210" y="{y - 3}" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="10" fill="#334155">{value:.4g}s</text>',
            ]
        )
    body.append("</svg>")
    return "\n".join(body)


def build_summary(data_dir: Path) -> dict[str, Any]:
    rows = load_raw_cases(data_dir)
    portability = read_json(PORTABILITY_SUMMARY)
    assert_non_deferred(rows, portability)
    ncu = parse_ncu_summary(data_dir)
    summary = {
        "campaign": "cuda_deferred_headroom_campaign9",
        "date": "2026-04-29",
        "deferred_status_allowed": False,
        "decisions": decision_rows(rows, portability),
        "raw_rows": rows,
        "ncu_summary": ncu,
        "readme_performance_landscape": performance_landscape(rows),
        "artifacts": {
            "ncu_report_binary": {
                "checked_in": False,
                "reason": "Raw profiler evidence is retained only in the private research archive.",
            }
        },
    }
    return summary


def render_assets(data_dir: Path, plot_dir: Path) -> dict[str, Any]:
    summary = build_summary(data_dir)
    write_text_if_changed(data_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True))
    decisions = summary["decisions"]
    portability = read_json(PORTABILITY_SUMMARY)
    plots = {
        "status": render_status_svg(decisions),
        "ncu": render_bar_svg(
            "Campaign 9 privileged Nsight Compute",
            "Selected first-observed full-set metrics from retained compact-consumer kernels.",
            summary["ncu_summary"],
            value_key="duration_us",
            unit="us",
        ),
        "portability": render_portability_svg(portability),
        "landscape": render_landscape_svg(summary["readme_performance_landscape"]),
    }
    for key, svg in plots.items():
        write_text_if_changed(plot_dir / PLOT_FILES[key], svg)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = render_assets(args.data_dir, args.plot_dir)
    print(json.dumps({"campaign": summary["campaign"], "decisions": summary["decisions"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
