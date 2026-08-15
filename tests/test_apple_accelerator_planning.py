from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE_PATH = "docs/architecture/apple_accelerator.md"
PLAN_PATH = "docs/plans/apple_metal_mps_bringup_plan.md"
REPORT_PATH = "docs/benchmarks/reports/apple_metal_bringup_2026-05-01.md"


def load_validate_module() -> ModuleType:
    validate_path = ROOT / "scripts" / "validate.py"
    spec = importlib.util.spec_from_file_location("fastpauli_validate", validate_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def normalized(path: str) -> str:
    return " ".join(read(path).split())


def test_apple_docs_are_registered_as_source_of_truth() -> None:
    validate = load_validate_module()

    for path in (ARCHITECTURE_PATH, PLAN_PATH):
        assert (ROOT / path).exists()
        assert path in validate.SOURCE_OF_TRUTH_PATHS
        assert path in read("docs/research/provenance.md")
        assert path in read("AGENTS.md")
        assert path in read("docs/roadmap.md")

    assert (ROOT / REPORT_PATH).exists()
    assert REPORT_PATH in read("docs/research/provenance.md")
    assert REPORT_PATH in read("AGENTS.md")
    assert REPORT_PATH in read("docs/roadmap.md")

    for path in (
        "docs/architecture/backend_neutral_accelerators.md",
        "docs/architecture/hardware_targets_and_testing.md",
        "docs/architecture/semantic_contracts.md",
        "docs/architecture/api_stability.md",
        "docs/quality/release_and_packaging.md",
    ):
        text = read(path)
        assert ARCHITECTURE_PATH in text

    assert PLAN_PATH in read("docs/architecture/backend_neutral_accelerators.md")
    assert PLAN_PATH in read("docs/architecture/hardware_targets_and_testing.md")
    assert PLAN_PATH in read("docs/quality/release_and_packaging.md")


def test_apple_architecture_defines_target_specific_metal_boundary() -> None:
    text = normalized(ARCHITECTURE_PATH)

    for required in (
        "backend identity for this target is:",
        "FASTPAULI_ENABLE_METAL=ON",
        "Metal-target source build",
        "mutually exclusive with CUDA and HIP",
        "Wolfgang should not expose",
        "Wolfgang-owned `MTLBuffer` storage",
        "MTLResourceStorageModeShared",
        "Public async, user-provided command queue, command-buffer, event, heap, or workspace APIs require a separate API plan",
        "The initial public API must not export raw Metal buffers, DLPack capsules",
        "accelerator_build_mode == \"metal_only\"",
        "Metal support is source-build-only",
        "Metal wheel support requires",
    ):
        assert required in text


def test_apple_bringup_plan_is_executable_and_scope_limited() -> None:
    text = normalized(PLAN_PATH)

    for required in (
        "REQUIRED SUB-SKILL",
        "Task 1: Build Flag, Source Layout, And CPU Safety",
        "Task 2: Status And Build Metadata",
        "Task 3: Transfers And Object Identity",
        "Task 4: Pairwise Commutation And Compact Consumers",
        "Task 5: Report, Review, And Closeout",
        "MPSGraph-first sparse kernels",
        "mixed CUDA/HIP/Metal source builds",
        "FASTPAULI_ENABLE_METAL=ON source build passes on Apple Silicon",
    ):
        assert required in text

    assert "This plan is executable only after" in text
    assert "must not add CUDA, ROCm/HIP, combined accelerator, or Metal wheel support" in text


def test_apple_planning_has_entered_source_build_implementation() -> None:
    cmake_text = read("CMakeLists.txt")

    assert "FASTPAULI_ENABLE_METAL" in cmake_text
    assert (ROOT / "src" / "metal").exists()


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_benchmark_and_release_docs_capture_apple_evidence_requirements() -> None:
    benchmark_protocol = read("docs/benchmarks/protocol.md")
    release_policy = read("docs/quality/release_and_packaging.md")
    bringup_report_normalized = normalized(REPORT_PATH)

    for required in (
        "metal_only",
        "object_backend: cpu, cuda, hip, or metal",
        "Xcode or Command Line Tools version",
        "command-buffer synchronization boundary",
        "`xctrace`",
        "MPS, MPSGraph, and PyTorch `mps` are external baselines only",
    ):
        assert required in benchmark_protocol

    for required in (
        "FASTPAULI_ENABLE_METAL=ON source build command",
        "named Apple Silicon SoC and Metal device",
        "Metal wheels remain unavailable",
        "framework linkage policy for Metal, Foundation, MetalPerformanceShaders, and MetalPerformanceShadersGraph",
    ):
        assert required in release_policy

    for required in (
        "FASTPAULI_ENABLE_METAL=ON",
        "accelerator_build_mode: metal_only",
        "Metal source build: passed",
        "MTLCreateSystemDefaultDevice(): nil",
        "Elevated MTLCreateSystemDefaultDevice(): Optional(<AGXG16SDevice",
        "system_profiler SPDisplaysDataType: Apple M4 Pro GPU, Metal: Supported",
        "Metal validation in this environment an elevated-command requirement",
        "Metal System Trace: captured all-process trace with FastPauli python3.12 process present",
        "Sanitized Metal System Trace summary: 50 Metal/GPU/graphics/MPS schemas",
        "TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.883 xcrun metal -v: Apple metal version 32023.883",
        "The default template has GPU counter profile and shader timeline disabled",
    ):
        assert required in bringup_report_normalized

    summary = json.loads(
        read("docs/benchmarks/data/apple_metal_bringup_2026-05-01/summary.json")
    )
    raw_benchmark = json.loads(
        read(
            "docs/benchmarks/data/apple_metal_bringup_2026-05-01/raw/"
            "metal_benchmark_smoke.json"
        )
    )
    assert summary["benchmark"] == "apple_metal_kernels"
    assert summary["status"] == "ok"
    assert summary["metal_status"]["runtime_available"] is True
    assert raw_benchmark["environment"]["cmake"] == "cmake version 4.3.2"
    assert summary["benchmark_rows"] == [
        {
            "median_seconds": case["timing"]["median"],
            "object_backend": case["object_backend"],
            "status": case["status"],
            "transfer_boundary": case["transfer_boundary"],
            "variant": case["variant"],
        }
        for case in raw_benchmark["cases"]
    ]
    assert summary["trace"]["template_name"] == "Metal System Trace"
    assert summary["trace"]["fastpauli_python_processes"]
    assert summary["trace"]["checked_summary"].endswith("metal_system_trace_summary.json")
    assert summary["trace"]["privacy"]["full_process_inventory"] == "omitted"
    assert summary["trace"]["fastpauli_python_processes"][0]["path"] == ".venv/bin/python"
    assert summary["trace"]["metal_schema_count"] >= 1
    assert "metal-command-buffer-completed" in summary["trace"]["metal_schemas"]

    trace_summary = json.loads(
        read(
            "docs/benchmarks/data/apple_metal_bringup_2026-05-01/profiler/"
            "metal_system_trace_summary.json"
        )
    )
    assert trace_summary["privacy"]["device_uuid"] == "omitted"
    assert trace_summary["privacy"]["full_process_inventory"] == "omitted"
    assert trace_summary["fastpauli_processes"][0]["pid"] == "redacted"
    assert "metal-command-buffer-completed" in trace_summary["metal_schemas"]


def test_official_reference_links_are_present() -> None:
    architecture = read(ARCHITECTURE_PATH)
    hardware_targets = read("docs/architecture/hardware_targets_and_testing.md")

    for url in (
        "https://developer.apple.com/documentation/metal/mtldevice",
        "https://developer.apple.com/documentation/metal/mtlbuffer",
        "https://developer.apple.com/documentation/metal/mtlcommandbuffer",
        "https://developer.apple.com/metal/cpp/",
        "https://developer.apple.com/documentation/metalperformanceshaders",
        "https://developer.apple.com/documentation/metalperformanceshadersgraph",
        "https://docs.pytorch.org/docs/stable/notes/mps",
    ):
        assert url in architecture
        assert url in hardware_targets
