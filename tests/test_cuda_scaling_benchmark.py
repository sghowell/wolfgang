"""CUDA scaling benchmark orchestration tests."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cuda_scaling_benchmark_defines_extreme_profile_without_running_it() -> None:
    script = ROOT / "benchmarks" / "bench_cuda_scaling.py"
    spec = importlib.util.spec_from_file_location("bench_cuda_scaling", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    profile = module.SCALE_PROFILES["extreme"]
    assert set(profile) == {
        "simplify_duplicate_pressure",
        "statevector_expectation",
        "pairwise_commutation",
        "matmul_product_generation_simplify",
    }
    assert profile["pairwise_commutation"][-1]["scale"] == "terms_16384x16384"
    assert profile["matmul_product_generation_simplify"][-1]["scale"] == "terms_4096x4096"


def test_cuda_scaling_benchmark_defines_materialization_profile_without_running_it() -> None:
    script = ROOT / "benchmarks" / "bench_cuda_scaling.py"
    spec = importlib.util.spec_from_file_location("bench_cuda_scaling", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    profile = module.SCALE_PROFILES["materialization"]
    assert set(profile) == {
        "simplify_duplicate_pressure",
        "statevector_expectation",
        "pairwise_commutation",
        "matmul_product_generation_simplify",
    }
    assert profile["simplify_duplicate_pressure"][-1]["scale"] == (
        "terms_200000_pathological_duplicate"
    )
    assert profile["pairwise_commutation"][0]["scale"].startswith("host_output_")


def test_cuda_scaling_benchmark_defines_campaign4_workspace_profile_without_running_it() -> None:
    script = ROOT / "benchmarks" / "bench_cuda_scaling.py"
    spec = importlib.util.spec_from_file_location("bench_cuda_scaling", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    profile = module.SCALE_PROFILES["campaign4_workspace"]
    assert set(profile) == {
        "simplify_duplicate_pressure",
        "statevector_expectation",
        "pairwise_commutation",
        "matmul_product_generation_simplify",
    }
    assert profile["simplify_duplicate_pressure"][0]["scale"].startswith("oneword_low")
    assert profile["simplify_duplicate_pressure"][-1]["num_qubits"] > 64
    assert {row["output_target"] for row in profile["pairwise_commutation"]} == {
        "host_vector",
        "caller_owned_host_bytes",
        "caller_owned_device_bytes",
        "bitpacked_device_words",
    }
    assert {row["statevector_dtype"] for row in profile["statevector_expectation"]} == {
        "complex64",
        "complex128",
    }


def test_cuda_scaling_benchmark_defines_campaign5_device_output_profile_without_running_it() -> None:
    script = ROOT / "benchmarks" / "bench_cuda_scaling.py"
    spec = importlib.util.spec_from_file_location("bench_cuda_scaling", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    profile = module.SCALE_PROFILES["campaign5_device_output"]
    assert set(profile) == {"pairwise_commutation"}
    assert [row["scale"] for row in profile["pairwise_commutation"]][:3] == [
        "device_output_terms_1024x1024",
        "device_output_terms_2048x2048",
        "device_output_terms_4096x4096",
    ]
    assert profile["pairwise_commutation"][-2]["scale"] == "device_output_terms_8192x8192"
    assert profile["pairwise_commutation"][-1]["scale"] == "device_output_terms_16384x16384"
    assert {row["output_target"] for row in profile["pairwise_commutation"]} == {
        "device_uint8_matrix"
    }


def test_cuda_scaling_benchmark_defines_campaign6_consumer_profile_without_running_it() -> None:
    script = ROOT / "benchmarks" / "bench_cuda_scaling.py"
    spec = importlib.util.spec_from_file_location("bench_cuda_scaling", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    profile = module.SCALE_PROFILES["campaign6_consumers"]
    assert set(profile) == {"pairwise_commutation"}
    assert [row["scale"] for row in profile["pairwise_commutation"]] == [
        "dense_consumer_terms_2048x2048",
        "dense_consumer_terms_8192x8192",
        "dense_consumer_terms_16384x16384",
    ]
    assert {row["output_target"] for row in profile["pairwise_commutation"]} == {
        "device_uint8_matrix_consumer"
    }


def test_cuda_scaling_benchmark_defines_campaign7_fused_consumer_profile_without_running_it() -> None:
    script = ROOT / "benchmarks" / "bench_cuda_scaling.py"
    spec = importlib.util.spec_from_file_location("bench_cuda_scaling", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    profile = module.SCALE_PROFILES["campaign7_fused_consumers"]
    assert set(profile) == {"pairwise_commutation"}
    assert [row["scale"] for row in profile["pairwise_commutation"]] == [
        "fused_consumer_terms_2048x2048",
        "fused_consumer_terms_8192x8192",
        "fused_consumer_terms_16384x16384",
    ]
    assert {row["output_target"] for row in profile["pairwise_commutation"]} == {
        "device_uint8_matrix_fused_consumer"
    }


def test_cuda_scaling_benchmark_defines_campaign8_profiles_and_schema_without_running_them() -> None:
    script = ROOT / "benchmarks" / "bench_cuda_scaling.py"
    spec = importlib.util.spec_from_file_location("bench_cuda_scaling", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required_status_fields = {
        "device_resident_graph_status",
        "public_grouping_api_status",
        "dlpack_interop_status",
        "non_h100_portability_status",
        "stream_graph_status",
        "scatter_tuning_status",
    }
    assert set(module.CAMPAIGN8_REQUIRED_STATUS_FIELDS) == required_status_fields
    assert {
        "campaign8-device-graph",
        "campaign8-grouping-consumer",
        "campaign8-interop",
        "campaign8-stream-graph",
        "campaign8-scatter-ab",
        "campaign8-portability",
        "fused-graph-stress",
    }.issubset(module.SCALE_PROFILES)
    assert module.SCALE_PROFILES["fused-graph-stress"] == module.SCALE_PROFILES[
        "campaign7_fused_consumers"
    ]
    assert {row["output_target"] for row in module.SCALE_PROFILES["campaign8-device-graph"]["pairwise_commutation"]} == {
        "campaign8_device_resident_graph"
    }
    assert {row["output_target"] for row in module.SCALE_PROFILES["campaign8-grouping-consumer"]["pairwise_commutation"]} == {
        "campaign8_device_grouping_consumer"
    }


def test_cuda_scaling_campaign8_cpu_unavailable_rows_include_required_schema() -> None:
    script = ROOT / "benchmarks" / "bench_cuda_scaling.py"
    spec = importlib.util.spec_from_file_location("bench_cuda_scaling", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    build_info = {
        "cuda_toolkit_version": "not_available",
        "cuda_architectures": "70,75,80,86,89,90",
    }
    cuda_status = {
        "built": False,
        "runtime_available": False,
        "skip_reason": "CUDA unavailable for test",
        "driver_version": "",
        "runtime_version": "",
        "devices": [],
    }

    rows = module.campaign8_cpu_unavailable_cases(
        profile="campaign8-device-graph",
        operations=["pairwise_commutation"],
        build_info=build_info,
        cuda_status=cuda_status,
        git_revision="f" * 40,
    )

    assert rows
    required = {
        "campaign",
        "mode",
        "boundary",
        "timing_boundary",
        "correctness_digest",
        "unavailable_reason",
        "git_revision",
        "cuda_driver",
        "cuda_runtime",
        "cuda_toolkit",
        "compiled_architectures",
        "gpu_name",
        "gpu_compute_capability",
        "device_resident_graph_status",
        "public_grouping_api_status",
        "dlpack_interop_status",
        "non_h100_portability_status",
        "stream_graph_status",
        "scatter_tuning_status",
    }
    for row in rows:
        assert required.issubset(row)
        assert row["campaign"] == "h100_campaign8"
        assert row["mode"] == "device_resident_graph"
        assert row["status"] == "unavailable"
        assert row["device_resident_graph_status"] == "unavailable"
        assert row["unavailable_reason"] == "CUDA unavailable for test"


def test_cuda_scaling_benchmark_defines_campaign9_profiles_and_schema_without_running_them() -> None:
    script = ROOT / "benchmarks" / "bench_cuda_scaling.py"
    spec = importlib.util.spec_from_file_location("bench_cuda_scaling", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "deferred" not in module.CAMPAIGN9_FINAL_STATUSES
    assert set(module.CAMPAIGN9_REQUIRED_ROW_FIELDS) == {
        "campaign",
        "mode",
        "boundary",
        "campaign8_headroom_item",
        "final_status",
        "deferred_status_allowed",
        "decision_doc",
        "correctness_digest",
        "unavailable_reason",
        "git_revision",
        "cuda_driver",
        "cuda_runtime",
        "cuda_toolkit",
        "compiled_architectures",
        "gpu_name",
        "gpu_compute_capability",
    }
    expected_profiles = {
        "campaign9-privileged-ncu": ("privileged_ncu", 2),
        "campaign9-non-h100-portability": ("non_h100_portability", 1),
        "campaign9-public-grouping-api": ("public_grouping_api", 3),
        "campaign9-dlpack-interop": ("dlpack_interop", 4),
        "campaign9-stream-graph": ("stream_graph", 5),
        "campaign9-csr-scatter-ab": ("csr_scatter_reopen", 6),
    }
    assert expected_profiles.keys() <= module.SCALE_PROFILES.keys()
    assert set(module.CAMPAIGN9_MODES_BY_TARGET.values()) == {
        mode for mode, _ in expected_profiles.values()
    }
    for profile, (mode, item) in expected_profiles.items():
        row = module.SCALE_PROFILES[profile]["pairwise_commutation"][0]
        assert module.CAMPAIGN9_MODES_BY_TARGET[row["output_target"]] == mode
        assert module.CAMPAIGN9_MODE_METADATA[mode]["campaign8_headroom_item"] == item
        assert module.CAMPAIGN9_MODE_METADATA[mode]["default_final_status"] != "deferred"


def test_cuda_scaling_benchmark_defines_campaign10_profiles_and_schema_without_running_them() -> None:
    script = ROOT / "benchmarks" / "bench_cuda_scaling.py"
    spec = importlib.util.spec_from_file_location("bench_cuda_scaling", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "deferred" not in module.CAMPAIGN10_FINAL_STATUSES
    assert set(module.CAMPAIGN10_REQUIRED_ROW_FIELDS) == {
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
    expected_profiles = {
        "campaign10-portability": ("cross_arch_portability", 1),
        "campaign10-dlpack-pytorch": ("dlpack_pytorch", 2),
        "campaign10-public-grouping-api": ("public_grouping_api", 3),
        "campaign10-stream-graph-reprobe": ("stream_graph_reprobe", 4),
        "campaign10-csr-scatter-reprobe": ("csr_scatter_reprobe", 5),
    }
    assert expected_profiles.keys() <= module.SCALE_PROFILES.keys()
    assert set(module.CAMPAIGN10_MODES_BY_TARGET.values()) == {
        mode for mode, _ in expected_profiles.values()
    }
    for profile, (mode, item) in expected_profiles.items():
        row = module.SCALE_PROFILES[profile]["pairwise_commutation"][0]
        assert module.CAMPAIGN10_MODES_BY_TARGET[row["output_target"]] == mode
        assert module.CAMPAIGN10_MODE_METADATA[mode]["campaign9_headroom_item"] == item
        assert module.CAMPAIGN10_MODE_METADATA[mode]["default_final_status"] != "deferred"


def test_cuda_scaling_campaign9_cpu_unavailable_rows_include_required_schema() -> None:
    script = ROOT / "benchmarks" / "bench_cuda_scaling.py"
    spec = importlib.util.spec_from_file_location("bench_cuda_scaling", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    build_info = {
        "cuda_toolkit_version": "not_available",
        "cuda_architectures": "70,75,80,86,89,90",
    }
    cuda_status = {
        "built": False,
        "runtime_available": False,
        "skip_reason": "CUDA unavailable for Campaign 9 test",
        "driver_version": "",
        "runtime_version": "",
        "devices": [],
    }

    rows = module.campaign9_cpu_unavailable_cases(
        profile="campaign9-public-grouping-api",
        operations=["pairwise_commutation"],
        build_info=build_info,
        cuda_status=cuda_status,
        git_revision="f" * 40,
    )

    assert rows
    for row in rows:
        assert set(module.CAMPAIGN9_REQUIRED_ROW_FIELDS).issubset(row)
        assert row["campaign"] == "cuda_deferred_headroom_campaign9"
        assert row["mode"] == "public_grouping_api"
        assert row["campaign8_headroom_item"] == 3
        assert row["final_status"] == "blocked_external"
        assert row["final_status"] != "deferred"
        assert row["deferred_status_allowed"] is False
        assert row["correctness_digest"] == ""
        assert row["unavailable_reason"] == "CUDA unavailable for Campaign 9 test"


def test_cuda_scaling_campaign10_cpu_unavailable_rows_include_required_schema(monkeypatch) -> None:
    monkeypatch.delenv("WOLFGANG_CAMPAIGN10_SSH_TARGET", raising=False)
    monkeypatch.delenv("WOLFGANG_CAMPAIGN10_PROVIDER_INSTANCE_TYPE", raising=False)
    monkeypatch.delenv("WOLFGANG_CAMPAIGN10_ARCHITECTURE_COMPILE_STATUS", raising=False)

    script = ROOT / "benchmarks" / "bench_cuda_scaling.py"
    spec = importlib.util.spec_from_file_location("bench_cuda_scaling", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    build_info = {
        "cuda_toolkit_version": "not_available",
        "cuda_architectures": "70,75,80,86,89,90",
    }
    cuda_status = {
        "built": False,
        "runtime_available": False,
        "skip_reason": "CUDA unavailable for Campaign 10 test",
        "driver_version": "",
        "runtime_version": "",
        "devices": [],
    }

    rows = module.campaign10_cpu_unavailable_cases(
        profile="campaign10-dlpack-pytorch",
        operations=["pairwise_commutation"],
        build_info=build_info,
        cuda_status=cuda_status,
        git_revision="f" * 40,
    )

    assert rows
    for row in rows:
        assert set(module.CAMPAIGN10_REQUIRED_ROW_FIELDS).issubset(row)
        assert row["campaign"] == "cuda_cross_architecture_campaign10"
        assert row["mode"] == "dlpack_pytorch"
        assert row["campaign9_headroom_item"] == 2
        assert row["final_status"] == "blocked_external"
        assert row["final_status"] != "deferred"
        assert row["deferred_status_allowed"] is False
        assert row["provider_instance_type"] == "not_available_to_agent"
        assert row["architecture_compile_status"] == "not_checked"
        assert row["correctness_digest"] == ""
        assert row["unavailable_reason"] == "CUDA unavailable for Campaign 10 test"


def test_cuda_campaign9_public_grouping_final_status_is_not_deferred() -> None:
    script = ROOT / "benchmarks" / "bench_cuda_scaling.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--profile",
            "campaign9-public-grouping-api",
            "--repeat",
            "1",
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["campaign9"]["deferred_status_allowed"] is False
    assert "deferred" not in report["campaign9"]["allowed_final_statuses"]
    assert report["cases"]
    assert {row["final_status"] for row in report["cases"]} <= set(
        report["campaign9"]["allowed_final_statuses"]
    )
    assert all(row["final_status"] != "deferred" for row in report["cases"])
    assert all(row["deferred_status_allowed"] is False for row in report["cases"])


def test_cuda_campaign10_profiles_emit_non_deferred_cpu_unavailable_rows() -> None:
    script = ROOT / "benchmarks" / "bench_cuda_scaling.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--profile",
            "campaign10-dlpack-pytorch",
            "--repeat",
            "1",
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["campaign10"]["deferred_status_allowed"] is False
    assert "deferred" not in report["campaign10"]["allowed_final_statuses"]
    assert report["cases"]
    assert {row["final_status"] for row in report["cases"]} <= set(
        report["campaign10"]["allowed_final_statuses"]
    )
    assert all(row["final_status"] != "deferred" for row in report["cases"])
    assert all(row["deferred_status_allowed"] is False for row in report["cases"])


def test_cuda_scaling_campaign10_dependency_failure_becomes_blocked_dependency() -> None:
    script = ROOT / "benchmarks" / "bench_cuda_scaling.py"
    spec = importlib.util.spec_from_file_location("bench_cuda_scaling", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    row = {"correctness_digest": ""}
    module.add_campaign10_row_schema_fields(
        row,
        mode="dlpack_pytorch",
        build_info={"cuda_toolkit_version": "13.3", "cuda_architectures": "100-real,120"},
        cuda_status={
            "driver_version": "13.0",
            "runtime_version": "12.8",
            "devices": [{"name": "NVIDIA B300 SXM6 AC", "compute_capability": (10, 3)}],
        },
        git_revision="f" * 40,
        unavailable_reason="CuPy runtime cannot target compute capability 10.3",
        dependency_unavailable=True,
    )

    assert row["final_status"] == "blocked_dependency"
    assert row["unavailable_reason"] == "CuPy runtime cannot target compute capability 10.3"


def test_cuda_scaling_benchmark_smoke_reports_planned_scales() -> None:
    script = ROOT / "benchmarks" / "bench_cuda_scaling.py"
    assert script.exists()

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--profile",
            "smoke",
            "--repeat",
            "1",
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["benchmark"] == "cuda_scaling"
    assert report["scale_profile"] == "smoke"
    assert report["correctness_checks"]["enabled"] is True
    assert set(report["campaign7"]) == {
        "fused_graph_csr",
        "fused_conflict_degrees",
        "fused_grouping_summary",
        "count_specialization_status",
        "bitpacked_decision_status",
        "portability_gpu",
    }
    assert report["campaign7"]["fused_graph_csr"]["mode"] == "csr_anticommutation_graph"
    assert report["campaign7"]["fused_conflict_degrees"]["mode"] == "conflict_degrees"
    assert report["campaign7"]["fused_grouping_summary"]["mode"] == "grouping_summary"
    assert report["campaign7"]["count_specialization_status"] == "rejected_not_dominant"
    assert (
        report["campaign7"]["bitpacked_decision_status"]
        == "deferred_no_dense_capacity_or_bandwidth_trigger"
    )
    assert {case["name"] for case in report["planned_cases"]} == {
        "simplify_duplicate_pressure",
        "statevector_expectation",
        "pairwise_commutation",
        "matmul_product_generation_simplify",
    }
    assert set(report["campaign8"]["required_status_fields"]) == {
        "device_resident_graph_status",
        "public_grouping_api_status",
        "dlpack_interop_status",
        "non_h100_portability_status",
        "stream_graph_status",
        "scatter_tuning_status",
    }
    assert report["campaign8"]["public_grouping_api_status"] == "deferred"
    assert report["campaign8"]["dlpack_interop_status"] == "deferred"
    assert report["campaign8"]["stream_graph_status"] == "deferred"
    assert report["campaign9"]["campaign"] == "cuda_deferred_headroom_campaign9"
    assert report["campaign9"]["deferred_status_allowed"] is False
    assert "deferred" not in report["campaign9"]["allowed_final_statuses"]
    assert report["campaign10"]["campaign"] == "cuda_cross_architecture_campaign10"
    assert report["campaign10"]["deferred_status_allowed"] is False
    assert "deferred" not in report["campaign10"]["allowed_final_statuses"]
    assert all(case["scales"] for case in report["planned_cases"])

    if report["cuda_status"]["built"] and report["cuda_status"]["runtime_available"]:
        assert report["cases"]
        first = report["cases"][0]
        assert "scale" in first
        assert "dataset" in first
        assert "results" in first
        assert "cuda_device_resident_seconds" in first["results"]
        assert "workspace_mode" in first["instrumentation"]
        assert "result_materialization_target" in first["instrumentation"]


def test_cuda_scaling_benchmark_output_option_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "cuda_scaling_smoke.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks" / "bench_cuda_scaling.py"),
            "--profile",
            "smoke",
            "--repeat",
            "1",
            "--json",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(completed.stdout)
