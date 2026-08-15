#!/usr/bin/env python3
"""Render Campaign 10 cross-architecture CUDA summaries and plots."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29"
DEFAULT_PLOT_DIR = ROOT / "docs/benchmarks/plots"
CAMPAIGN9_SUMMARY = (
    ROOT / "docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29/summary.json"
)

PLOT_FILES = {
    "cross_architecture": "cuda_campaign10_cross_architecture.svg",
    "dlpack_consumers": "cuda_campaign10_dlpack_consumers.svg",
    "headroom_status": "cuda_campaign10_headroom_status.svg",
    "landscape": "cuda_campaign10_performance_landscape.svg",
}

HEADROOM_LABELS = {
    1: "Non-H100 NVIDIA portability",
    2: "PyTorch CUDA DLPack",
    3: "Public grouping API",
    4: "Stream / CUDA Graph reprobe",
    5: "CSR scatter reprobe",
}

CAMPAIGN10_MODES = {
    "cross_arch_portability",
    "dlpack_pytorch",
    "public_grouping_api",
    "stream_graph_reprobe",
    "csr_scatter_reprobe",
    "readme_landscape",
}

FINAL_STATUSES = {
    "implemented",
    "passed",
    "rejected_with_evidence",
    "blocked_external",
    "blocked_toolchain",
    "blocked_dependency",
}

REQUIRED_ROW_FIELDS = {
    "campaign",
    "mode",
    "campaign9_headroom_item",
    "final_status",
    "deferred_status_allowed",
    "decision_doc",
    "provider_instance_type",
    "gpu_name",
    "gpu_compute_capability",
    "cuda_driver",
    "cuda_runtime",
    "cuda_toolkit",
    "compiled_architectures",
    "architecture_compile_status",
    "git_revision",
    "command",
    "correctness_digest",
    "unavailable_reason",
}

STATUS_COLORS = {
    "implemented": "#15803d",
    "passed": "#15803d",
    "rejected_with_evidence": "#b45309",
    "blocked_external": "#64748b",
    "blocked_toolchain": "#b91c1c",
    "blocked_dependency": "#7c3aed",
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


def _rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    cases = payload.get("cases")
    if isinstance(cases, list):
        return [case for case in cases if isinstance(case, dict)]
    if payload.get("campaign") == "cuda_cross_architecture_campaign10":
        return [payload]
    return []


def load_raw_cases(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((data_dir / "raw").glob("*.json")):
        payload = read_json(path)
        for case in _rows_from_payload(payload):
            if case.get("campaign") != "cuda_cross_architecture_campaign10":
                continue
            row = dict(case)
            if data_dir.name == DEFAULT_DATA_DIR.name:
                row["source_file"] = str(
                    Path("docs/benchmarks/data") / DEFAULT_DATA_DIR.name / "raw" / path.name
                )
            else:
                try:
                    row["source_file"] = str(path.resolve().relative_to(ROOT))
                except ValueError:
                    row["source_file"] = str(path)
            rows.append(row)
    return rows


def _validate_blackwell_row(row: dict[str, Any]) -> None:
    name = str(row.get("gpu_name", "")).lower()
    capability = str(row.get("gpu_compute_capability", ""))
    architectures = str(row.get("compiled_architectures", ""))
    compile_status = str(row.get("architecture_compile_status", ""))
    is_blackwell = "rtx pro 6000" in name or capability == "12.0" or "120" in architectures
    if not is_blackwell:
        return
    if compile_status in {"", "not_checked"}:
        raise ValueError(
            f"Blackwell row lacks an explicit architecture compile outcome: {row.get('source_file')}"
        )
    if "120" not in architectures and row.get("final_status") != "blocked_toolchain":
        raise ValueError(
            "Blackwell rows must either compile sm_120 or record blocked_toolchain: "
            f"{row.get('source_file')}"
        )


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Campaign 10 summary has no raw rows")
    items: set[int] = set()
    for row in rows:
        missing = sorted(REQUIRED_ROW_FIELDS - set(row))
        if missing:
            raise ValueError(f"Campaign 10 row omitted required field(s) {missing}: {row.get('source_file')}")
        if row["final_status"] == "deferred":
            raise ValueError("Campaign 10 summary may not contain final_status='deferred'")
        if row["final_status"] not in FINAL_STATUSES:
            raise ValueError(f"invalid Campaign 10 final_status: {row['final_status']}")
        if row["mode"] not in CAMPAIGN10_MODES:
            raise ValueError(f"invalid Campaign 10 mode: {row['mode']}")
        if row.get("deferred_status_allowed") is not False:
            raise ValueError(f"Campaign 10 row allows deferred status: {row.get('source_file')}")
        item = int(row["campaign9_headroom_item"])
        if item not in HEADROOM_LABELS:
            raise ValueError(f"invalid Campaign 10 headroom item: {item}")
        items.add(item)
        if row.get("gpu_name") or row.get("gpu_compute_capability"):
            for field in ("cuda_driver", "cuda_runtime", "cuda_toolkit", "compiled_architectures"):
                if str(row.get(field, "")) == "":
                    raise ValueError(f"hardware row lacks {field}: {row.get('source_file')}")
        _validate_blackwell_row(row)
    if items != set(HEADROOM_LABELS):
        raise ValueError(f"Campaign 10 summary does not cover all headroom items: {sorted(items)}")


def decision_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {
        "implemented": 0,
        "passed": 1,
        "rejected_with_evidence": 2,
        "blocked_dependency": 3,
        "blocked_toolchain": 4,
        "blocked_external": 5,
    }
    decisions: list[dict[str, Any]] = []
    for item, label in HEADROOM_LABELS.items():
        item_rows = [row for row in rows if int(row["campaign9_headroom_item"]) == item]
        selected = sorted(item_rows, key=lambda row: priority[str(row["final_status"])])[0]
        decisions.append(
            {
                "campaign9_headroom_item": item,
                "label": label,
                "mode": selected["mode"],
                "final_status": selected["final_status"],
                "decision_doc": selected["decision_doc"],
                "evidence": selected.get("source_file", ""),
            }
        )
    return decisions


def hardware_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["gpu_name"])
        if key not in by_key:
            by_key[key] = {
                "provider_instance_type": row["provider_instance_type"],
                "gpu_name": row["gpu_name"],
                "gpu_compute_capability": row["gpu_compute_capability"],
                "cuda_driver": row["cuda_driver"],
                "cuda_runtime": row["cuda_runtime"],
                "cuda_toolkit": row["cuda_toolkit"],
                "compiled_architectures": row["compiled_architectures"],
                "architecture_compile_status": row["architecture_compile_status"],
                "statuses": sorted({r["final_status"] for r in rows if str(r["gpu_name"]) == key}),
            }
    return sorted(by_key.values(), key=lambda row: str(row["gpu_name"]))


def _point(results: dict[str, Any], key: str, series: str) -> dict[str, Any] | None:
    value = results.get(key)
    if isinstance(value, (int, float)) and value > 0:
        return {"series": series, "seconds": float(value)}
    return None


def performance_landscape(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    landscape: list[dict[str, Any]] = []
    if CAMPAIGN9_SUMMARY.exists():
        prior = read_json(CAMPAIGN9_SUMMARY).get("readme_performance_landscape", [])
        for item in prior:
            landscape.append(
                {
                    "category": f"Campaign 9 reference: {item.get('category', '')}".strip(),
                    "scale": item.get("scale", ""),
                    "mode": item.get("mode", item.get("operation", "campaign9_reference")),
                    "final_status": "reference",
                    "gpu_name": "H100 reference or CPU reference",
                    "points": item.get("points", []),
                }
            )
    for row in rows:
        if row.get("boundary") == "profiler_only":
            continue
        points: list[dict[str, Any]] = []
        results = row.get("results", {})
        for key, series in {
            "cpu_scalar_seconds": "CPU scalar",
            "cpu_default_seconds": "CPU default",
            "cpu_optimized_seconds": "CPU optimized",
            "cuda_transfer_inclusive_seconds": "CUDA transfer-inclusive",
            "cuda_device_resident_seconds": "CUDA device-resident",
            "campaign8_device_resident_graph_compact_seconds": "CUDA compact graph consumer",
            "campaign8_device_grouping_consumer_seconds": "CUDA compact grouping consumer",
            "conflict_degrees_axis_none_seconds": "CUDA conflict degrees total",
            "cupy_asarray_export_seconds": "CuPy CUDA Array Interface",
            "cupy_dlpack_from_dlpack_seconds": "CuPy DLPack",
            "torch_dlpack_from_dlpack_seconds": "PyTorch DLPack",
            "campaign7_csr_graph_export_seconds": "CUDA CSR export baseline",
        }.items():
            point = _point(results, key, series)
            if point is not None:
                points.append(point)
        for selector, payload in sorted(results.get("cpu_optimized_timings", {}).items()):
            if isinstance(payload, dict):
                point = _point(payload, "seconds", f"CPU {selector.upper()}")
                if point is not None:
                    points.append(point)
        if points:
            landscape.append(
                {
                    "category": "Campaign 10 cross-architecture",
                    "scale": row.get("scale", ""),
                    "mode": row["mode"],
                    "final_status": row["final_status"],
                    "gpu_name": row["gpu_name"],
                    "points": points,
                }
            )
    return landscape


def render_status_svg(decisions: list[dict[str, Any]]) -> str:
    width = 980
    row_h = 58
    height = 96 + row_h * len(decisions)
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="36" y="44" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="24" font-weight="700" fill="#0f172a">Campaign 10 headroom closure</text>',
        '<text x="36" y="70" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="13" fill="#475569">Every Campaign 9 remaining-headroom item has a terminal non-deferred outcome.</text>',
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


def render_hardware_svg(hardware: list[dict[str, Any]]) -> str:
    width = 1120
    row_h = 64
    height = 104 + max(1, len(hardware)) * row_h
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="36" y="44" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="24" font-weight="700" fill="#0f172a">Campaign 10 cross-architecture CUDA coverage</text>',
        '<text x="36" y="70" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="13" fill="#475569">Source-build and runtime evidence by measured NVIDIA host.</text>',
    ]
    if not hardware:
        body.append('<text x="36" y="118" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="14" fill="#64748b">No hardware rows available.</text>')
    for index, row in enumerate(hardware):
        y = 108 + index * row_h
        status_text = ", ".join(row["statuses"])
        body.extend(
            [
                f'<rect x="36" y="{y - 30}" width="1048" height="50" rx="6" fill="#ffffff" stroke="#dbe3ef"/>',
                f'<text x="58" y="{y - 10}" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="14" font-weight="700" fill="#0f172a">{escape(row["gpu_name"] or "GPU unavailable")}</text>',
                f'<text x="58" y="{y + 8}" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12" fill="#475569">cc {escape(row["gpu_compute_capability"])} · arch {escape(row["compiled_architectures"])} · driver {escape(row["cuda_driver"])} · {escape(status_text)}</text>',
            ]
        )
    body.append("</svg>")
    return "\n".join(body)


def render_bar_svg(title: str, subtitle: str, rows: list[dict[str, Any]], *, value_key: str, unit: str) -> str:
    width = 1080
    row_h = 48
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
                f'<text x="36" y="{y}" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12" font-weight="700" fill="#0f172a">{escape(row.get("label", ""))}</text>',
                f'<rect x="390" y="{y - 14}" width="560" height="14" rx="3" fill="#e2e8f0"/>',
                f'<rect x="390" y="{y - 14}" width="{bar_w:.2f}" height="14" rx="3" fill="#2563eb"/>',
                f'<text x="964" y="{y - 3}" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="11" fill="#334155">{value:.4g} {escape(unit)}</text>',
            ]
        )
    body.append("</svg>")
    return "\n".join(body)


def dlpack_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if row.get("mode") != "dlpack_pytorch":
            continue
        results = row.get("results", {})
        for key, label in {
            "cupy_dlpack_from_dlpack_seconds": "CuPy DLPack",
            "torch_dlpack_from_dlpack_seconds": "PyTorch DLPack",
            "cupy_asarray_export_seconds": "CuPy CUDA Array Interface",
        }.items():
            value = results.get(key)
            if isinstance(value, (int, float)) and value > 0:
                output.append(
                    {
                        "label": f'{row.get("gpu_name", "GPU")} / {label}',
                        "seconds": float(value),
                    }
                )
    return sorted(output, key=lambda row: row["seconds"])


def render_landscape_svg(landscape: list[dict[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
    for item in landscape:
        for point in item["points"]:
            seconds = point.get("seconds")
            if not isinstance(seconds, (int, float)) or seconds <= 0:
                continue
            rows.append(
                {
                    "label": f'{item.get("gpu_name", "")} / {item["mode"]}: {point["series"]}',
                    "seconds": float(seconds),
                }
            )
    rows = sorted(rows, key=lambda row: row["seconds"])
    if not rows:
        return render_bar_svg(
            "Campaign 10 performance landscape",
            "No checked rows were available.",
            [],
            value_key="seconds",
            unit="s",
        )

    width = 1280
    row_h = 26
    height = 110 + row_h * len(rows)
    min_value = min(row["seconds"] for row in rows)
    max_value = max(row["seconds"] for row in rows)
    lo = math.log10(min_value)
    hi = math.log10(max_value)
    span = max(hi - lo, 1e-9)
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="36" y="44" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="24" font-weight="700" fill="#0f172a">Campaign 10 performance landscape</text>',
        '<text x="36" y="70" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="13" fill="#475569">CPU, H100 reference, A100, RTX-class, transfer-inclusive, device-resident, compact consumer, and DLPack paths. Bars use log-scaled seconds.</text>',
    ]
    for index, row in enumerate(rows):
        y = 104 + index * row_h
        value = row["seconds"]
        scaled = (math.log10(value) - lo) / span
        bar_w = 24 + 356 * scaled
        label = row["label"]
        if len(label) > 110:
            label = label[:107] + "..."
        fill = "#2563eb"
        if "CPU" in row["label"]:
            fill = "#059669"
        elif "DLPack" in row["label"] or "CuPy" in row["label"]:
            fill = "#7c3aed"
        elif "H100 reference" in row["label"]:
            fill = "#64748b"
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
    validate_rows(rows)
    summary = {
        "campaign": "cuda_cross_architecture_campaign10",
        "date": "2026-04-29",
        "deferred_status_allowed": False,
        "allowed_final_statuses": sorted(FINAL_STATUSES),
        "required_row_fields": sorted(REQUIRED_ROW_FIELDS),
        "decisions": decision_rows(rows),
        "hardware": hardware_rows(rows),
        "raw_rows": rows,
        "readme_performance_landscape": performance_landscape(rows),
    }
    return summary


def render_assets(data_dir: Path, plot_dir: Path) -> dict[str, Any]:
    summary = build_summary(data_dir)
    write_text_if_changed(data_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True))
    plots = {
        "cross_architecture": render_hardware_svg(summary["hardware"]),
        "dlpack_consumers": render_bar_svg(
            "Campaign 10 DLPack consumers",
            "Measured framework consumer costs for DeviceCommutationMatrix exports.",
            dlpack_rows(summary["raw_rows"]),
            value_key="seconds",
            unit="s",
        ),
        "headroom_status": render_status_svg(summary["decisions"]),
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
