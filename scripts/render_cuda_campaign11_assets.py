#!/usr/bin/env python3
"""Render Campaign 11 residual-risk summary evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "docs/benchmarks/data/cuda_residual_risk_campaign11_2026-04-29"

CAMPAIGN = "cuda_residual_risk_campaign11"
RESIDUAL_ITEMS = {
    "non_h100_ncu_counters": "Non-H100 Nsight Compute counters",
    "nanobind_refleak_investigation": "Nanobind reference-leak diagnostics",
}
HOSTS = {
    "a100": {
        "compiled_architectures": "80",
    },
    "rtxpro6000blackwell": {
        "compiled_architectures": "120",
    },
}
FINAL_STATUSES = {
    "passed",
    "fixed",
    "rejected_with_evidence",
    "blocked_toolchain",
    "blocked_permissions",
    "blocked_dependency",
    "blocked_external",
}
SUCCESS_STATUSES = {"passed", "fixed", "rejected_with_evidence"}
REQUIRED_ROW_FIELDS = {
    "campaign",
    "residual_item",
    "final_status",
    "deferred_status_allowed",
    "host_id",
    "gpu_name",
    "gpu_compute_capability",
    "cuda_driver",
    "cuda_runtime",
    "cuda_toolkit",
    "compiled_architectures",
    "git_revision",
    "command",
    "artifact_paths",
    "limitation",
    "decision",
}
NCU_FIELDS = {
    "ncu_install_status",
    "ncu_version",
    "profiler_permission_status",
}
REFLEAK_FIELDS = {
    "compute_sanitizer_status",
    "nanobind_diagnostic_classification",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    cases = payload.get("cases")
    if isinstance(cases, list):
        return [case for case in cases if isinstance(case, dict)]
    if payload.get("campaign") == CAMPAIGN:
        return [payload]
    return []


def load_raw_rows(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((data_dir / "raw").glob("*.json")):
        payload = read_json(path)
        for case in _rows_from_payload(payload):
            if case.get("campaign") != CAMPAIGN:
                continue
            row = dict(case)
            try:
                row["source_file"] = str(path.resolve().relative_to(ROOT))
            except ValueError:
                row["source_file"] = str(path)
            rows.append(row)
    return rows


def _require_non_empty(row: dict[str, Any], field: str) -> None:
    value = row.get(field)
    if value is None or value == "" or value == []:
        raise ValueError(f"Campaign 11 row has empty {field}: {row.get('source_file')}")


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Campaign 11 summary has no raw rows")

    covered: set[tuple[str, str]] = set()
    for row in rows:
        missing = sorted(REQUIRED_ROW_FIELDS - set(row))
        if missing:
            raise ValueError(
                f"Campaign 11 row omitted required field(s) {missing}: {row.get('source_file')}"
            )
        if row["final_status"] == "deferred":
            raise ValueError("Campaign 11 summary may not contain final_status='deferred'")
        if row["final_status"] not in FINAL_STATUSES:
            raise ValueError(f"invalid Campaign 11 final_status: {row['final_status']}")
        if row.get("deferred_status_allowed") is not False:
            raise ValueError(f"Campaign 11 row allows deferred status: {row.get('source_file')}")

        item = str(row["residual_item"])
        host = str(row["host_id"])
        if item not in RESIDUAL_ITEMS:
            raise ValueError(f"invalid Campaign 11 residual item: {item}")
        if host not in HOSTS:
            raise ValueError(f"invalid Campaign 11 host_id: {host}")
        if str(row["compiled_architectures"]) != HOSTS[host]["compiled_architectures"]:
            raise ValueError(
                f"Campaign 11 row has unexpected compiled_architectures: {row.get('source_file')}"
            )
        for field in (
            "gpu_name",
            "gpu_compute_capability",
            "cuda_driver",
            "cuda_runtime",
            "cuda_toolkit",
            "git_revision",
            "command",
            "decision",
        ):
            _require_non_empty(row, field)
        if not isinstance(row["artifact_paths"], list) or not row["artifact_paths"]:
            raise ValueError(f"Campaign 11 row requires artifact_paths: {row.get('source_file')}")

        if item == "non_h100_ncu_counters":
            missing_ncu = sorted(NCU_FIELDS - set(row))
            if missing_ncu:
                raise ValueError(
                    f"Campaign 11 ncu row omitted field(s) {missing_ncu}: {row.get('source_file')}"
                )
            if row["final_status"] == "passed":
                for field in ("ncu_version", "profiler_permission_status"):
                    _require_non_empty(row, field)
                if row["profiler_permission_status"] != "counters_captured":
                    raise ValueError(
                        "Campaign 11 passed ncu rows must capture counters: "
                        f"{row.get('source_file')}"
                    )
            if row["final_status"] == "blocked_permissions" and not str(row["limitation"]):
                raise ValueError(f"blocked ncu row lacks limitation: {row.get('source_file')}")

        if item == "nanobind_refleak_investigation":
            missing_ref = sorted(REFLEAK_FIELDS - set(row))
            if missing_ref:
                raise ValueError(
                    f"Campaign 11 nanobind row omitted field(s) {missing_ref}: {row.get('source_file')}"
                )
            if row["final_status"] in {"fixed", "rejected_with_evidence"}:
                _require_non_empty(row, "nanobind_diagnostic_classification")
                _require_non_empty(row, "compute_sanitizer_status")

        covered.add((item, host))

    expected = {
        (item, host)
        for item in RESIDUAL_ITEMS
        for host in HOSTS
    }
    if covered != expected:
        raise ValueError(f"Campaign 11 summary does not cover all item/host pairs: {sorted(covered)}")


def _aggregate_status(rows: list[dict[str, Any]]) -> str:
    statuses = {str(row["final_status"]) for row in rows}
    if statuses.issubset(SUCCESS_STATUSES):
        if "fixed" in statuses:
            return "fixed"
        if "rejected_with_evidence" in statuses:
            return "rejected_with_evidence"
        return "passed"
    for status in ("blocked_external", "blocked_toolchain", "blocked_dependency", "blocked_permissions"):
        if status in statuses:
            return status
    return sorted(statuses)[0]


def decision_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for item, label in RESIDUAL_ITEMS.items():
        item_rows = [row for row in rows if row["residual_item"] == item]
        decisions.append(
            {
                "residual_item": item,
                "label": label,
                "final_status": _aggregate_status(item_rows),
                "hosts": sorted({row["host_id"] for row in item_rows}),
                "evidence": sorted({row["source_file"] for row in item_rows}),
            }
        )
    return decisions


def host_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hosts: list[dict[str, Any]] = []
    for host_id in HOSTS:
        host_rows_for_id = [row for row in rows if row["host_id"] == host_id]
        representative = host_rows_for_id[0]
        hosts.append(
            {
                "host_id": host_id,
                "gpu_name": representative["gpu_name"],
                "gpu_compute_capability": representative["gpu_compute_capability"],
                "cuda_driver": representative["cuda_driver"],
                "cuda_runtime": representative["cuda_runtime"],
                "cuda_toolkit": representative["cuda_toolkit"],
                "compiled_architectures": representative["compiled_architectures"],
                "statuses": sorted({row["final_status"] for row in host_rows_for_id}),
            }
        )
    return hosts


def build_summary(data_dir: Path) -> dict[str, Any]:
    rows = load_raw_rows(data_dir)
    validate_rows(rows)
    return {
        "campaign": CAMPAIGN,
        "date": "2026-04-29",
        "deferred_status_allowed": False,
        "allowed_final_statuses": sorted(FINAL_STATUSES),
        "required_row_fields": sorted(REQUIRED_ROW_FIELDS),
        "required_hosts": sorted(HOSTS),
        "required_residual_items": sorted(RESIDUAL_ITEMS),
        "decisions": decision_rows(rows),
        "hardware": host_rows(rows),
        "raw_rows": rows,
    }


def render_assets(data_dir: Path) -> dict[str, Any]:
    summary = build_summary(data_dir)
    write_json(data_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = render_assets(args.data_dir)
    print(json.dumps({"campaign": summary["campaign"], "decisions": summary["decisions"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
