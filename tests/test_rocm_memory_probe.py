from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "rocm_memory_probe.py"


def load_probe_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fastpauli_rocm_memory_probe", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_probe_preserves_raw_growth_and_exit_deltas() -> None:
    probe = load_probe_module()

    before_process = {"median_free": 1000, "samples": [{"free": 1000, "total": 2000}], "tag": "before"}
    after_first_kernel = {
        "median_free": 940,
        "samples": [{"free": 940, "total": 2000}],
        "tag": "after_first_kernel",
    }
    after_cycles = {"median_free": 900, "samples": [{"free": 900, "total": 2000}], "tag": "after_cycles"}
    after_exit = {"median_free": 980, "samples": [{"free": 980, "total": 2000}], "tag": "after_exit"}

    summary = probe.summarize_probe(
        before_process=before_process,
        after_first_kernel=after_first_kernel,
        after_cycles=after_cycles,
        after_exit=after_exit,
    )

    assert summary["first_kernel_reservation_bytes"] == -60
    assert summary["growth_after_cycles_bytes"] == -40
    assert summary["cycle_plateau_growth_bytes"] == -40
    assert summary["warmup_reservation_bytes"] == 0
    assert summary["residual_after_exit_bytes"] == -20
    assert summary["one_time_reservation_bytes"] == -60
    assert summary["retained_growth_bytes"] == -40
    assert summary["adjudicated_retained_growth_bytes"] == -40
    assert summary["post_exit_recovered_bytes"] == 80


def test_summarize_probe_uses_cycle_plateau_to_clear_one_time_warmup_drop() -> None:
    probe = load_probe_module()

    before_process = {"median_free": 1000, "samples": [{"free": 1000, "total": 2000}], "tag": "before"}
    after_first_kernel = {
        "median_free": 940,
        "samples": [{"free": 940, "total": 2000}],
        "tag": "after_first_kernel",
    }
    after_cycles = {"median_free": 900, "samples": [{"free": 900, "total": 2000}], "tag": "after_cycles"}
    after_exit = {"median_free": 1000, "samples": [{"free": 1000, "total": 2000}], "tag": "after_exit"}
    construct_destroy_samples = [
        {"cycle": 1, "free": 900, "total": 2000},
        {"cycle": 5, "free": 900, "total": 2000},
        {"cycle": 10, "free": 900, "total": 2000},
        {"cycle": 15, "free": 900, "total": 2000},
        {"cycle": 20, "free": 900, "total": 2000},
    ]

    summary = probe.summarize_probe(
        before_process=before_process,
        after_first_kernel=after_first_kernel,
        after_cycles=after_cycles,
        after_exit=after_exit,
        construct_destroy_samples=construct_destroy_samples,
    )

    assert summary["growth_after_cycles_bytes"] == -40
    assert summary["warmup_reservation_bytes"] == -40
    assert summary["cycle_plateau_growth_bytes"] == 0
    assert summary["retained_growth_bytes"] == -40
    assert summary["adjudicated_retained_growth_bytes"] == 0


def test_run_parent_probe_records_before_child_and_after_exit_samples(tmp_path: Path) -> None:
    probe = load_probe_module()

    sampled_tags: list[str] = []

    def fake_settled_sample(tag: str, *, sample_count: int = 5, sleep_seconds: float = 0.0):
        sampled_tags.append(tag)
        medians = {
            "before_process": 1200,
            "after_exit": 1184,
        }
        median = medians[tag]
        return {
            "tag": tag,
            "median_free": median,
            "min_free": median,
            "max_free": median,
            "spread_bytes": 0,
            "samples": [{"index": 1, "free": median, "total": 2400}],
        }

    def fake_run_child_probe(output_path: Path) -> dict:
        assert output_path.parent == tmp_path
        return {
            "hip_status": {"runtime_available": True},
            "before_process": {"tag": "before_process", "median_free": 1200, "samples": []},
            "after_first_kernel": {"tag": "after_first_kernel", "median_free": 1152, "samples": []},
            "after_cycles": {"tag": "after_cycles", "median_free": 1120, "samples": []},
            "construct_destroy_samples": [{"cycle": 1, "free": 1120, "total": 2400}],
        }

    summary = probe.run_parent_probe(
        workspace=tmp_path,
        settled_sample=fake_settled_sample,
        run_child_probe=fake_run_child_probe,
    )

    assert sampled_tags == ["before_process", "after_exit"]
    assert summary["before_process"]["median_free"] == 1200
    assert summary["after_first_kernel"]["median_free"] == 1152
    assert summary["after_cycles"]["median_free"] == 1120
    assert summary["after_exit"]["median_free"] == 1184
    assert summary["growth_after_cycles_bytes"] == -32
    assert summary["cycle_plateau_growth_bytes"] == 0
    assert summary["warmup_reservation_bytes"] == -32
    assert summary["residual_after_exit_bytes"] == -16
    assert summary["adjudicated_retained_growth_bytes"] == -32
    assert summary["construct_destroy_samples"] == [{"cycle": 1, "free": 1120, "total": 2400}]
