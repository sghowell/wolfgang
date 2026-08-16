"""Shared metadata helpers for Wolfgang benchmark reports."""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DARWIN_ARM_FEATURE_KEYS = {
    "hw.optional.neon": "neon",
    "hw.optional.armv8_1_atomics": "armv8_1_atomics",
    "hw.optional.arm.FEAT_AES": "aes",
    "hw.optional.arm.FEAT_BF16": "bf16",
    "hw.optional.arm.FEAT_CRC32": "crc32",
    "hw.optional.arm.FEAT_DotProd": "dotprod",
    "hw.optional.arm.FEAT_FCMA": "fcma",
    "hw.optional.arm.FEAT_FP16": "fp16",
    "hw.optional.arm.FEAT_I8MM": "i8mm",
    "hw.optional.arm.FEAT_JSCVT": "jscvt",
    "hw.optional.arm.FEAT_LSE": "lse",
    "hw.optional.arm.FEAT_PMULL": "pmull",
    "hw.optional.arm.FEAT_SHA1": "sha1",
    "hw.optional.arm.FEAT_SHA256": "sha256",
}

CPU_FLAG_PREFIXES = (
    "-march",
    "-mcpu",
    "-mtune",
    "-mavx",
    "-mno-",
    "-mfpu",
    "-msse",
    "-mneon",
    "-mfloat-abi",
    "/arch:",
)
CPU_FLAG_TOKENS = {"-arch"}

ACCELERATOR_TRANSFER_BOUNDARIES = {
    "transfer_inclusive",
    "device_resident",
    "device_output_allocating",
    "device_output_reused",
    "device_output_to_host",
    "device_resident_private_output_blit_to_shared_staging",
    "device_to_host_cpu_simplify_host_to_device",
    "host_materialized",
    "compact_consumer",
    "compact_consumer_gpu_reduction",
    "compact_consumer_gpu_parallel_block_reduction",
    "status_only",
}
ACCELERATOR_OBJECT_BACKENDS = {"cpu", "cuda", "hip", "metal"}
PUBLIC_COMPILER_BUILD_KEYS = {
    "CMAKE_BUILD_TYPE",
    "CMAKE_CUDA_COMPILER_ID",
    "CMAKE_CUDA_COMPILER_VERSION",
    "CMAKE_CXX_COMPILER_ID",
    "CMAKE_CXX_COMPILER_VERSION",
    "CMAKE_HIP_COMPILER_ID",
    "CMAKE_HIP_COMPILER_VERSION",
    "CMAKE_OBJCXX_COMPILER_ID",
    "CMAKE_OBJCXX_COMPILER_VERSION",
}


def git_commit() -> str:
    override = os.environ.get("WOLFGANG_BENCHMARK_GIT_COMMIT")
    if override:
        return override
    provenance = git_provenance()
    return str(provenance["commit_label"])


def git_status_short() -> list[str]:
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return []
    return [line for line in completed.stdout.splitlines() if line.strip()]


def git_provenance() -> dict[str, Any]:
    override = os.environ.get("WOLFGANG_BENCHMARK_GIT_COMMIT")
    if override:
        return {
            "commit": override,
            "commit_label": override,
            "dirty": False,
            "source": "WOLFGANG_BENCHMARK_GIT_COMMIT",
            "working_tree_status": [],
        }
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {
            "commit": "unknown",
            "commit_label": "unknown",
            "dirty": False,
            "source": "git_unavailable",
            "working_tree_status": [],
        }
    commit = completed.stdout.strip()
    status = git_status_short()
    dirty = bool(status)
    return {
        "commit": commit,
        "commit_label": f"{commit}+dirty" if dirty else commit,
        "dirty": dirty,
        "source": "git",
        "working_tree_status": status,
    }


def command_first_line(command: list[str]) -> str:
    if shutil.which(command[0]) is None:
        return "not_available"
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "not_available"
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "not_available"


def command_output(command: list[str]) -> str:
    if shutil.which(command[0]) is None:
        return "not_available"
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "not_available"
    value = completed.stdout.strip()
    return value if value else "not_available"


def command_result(command: list[str]) -> tuple[int, str, str]:
    if shutil.which(command[0]) is None:
        return 127, "", f"{command[0]} not found"
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def optional_int(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return "not_available"


def physical_core_count() -> int | str:
    apple_hardware = apple_hardware_overview()
    if "total_core_count" in apple_hardware:
        return apple_hardware["total_core_count"]
    if platform.system() == "Darwin":
        return optional_int(command_output(["sysctl", "-n", "hw.physicalcpu"]))
    return "not_available"


def apple_hardware_overview() -> dict[str, Any]:
    if platform.system() != "Darwin":
        return {}

    output = command_output(["system_profiler", "SPHardwareDataType"])
    if output == "not_available":
        return {}

    overview: dict[str, Any] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Chip:"):
            overview["chip"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Model Name:"):
            overview["model_name"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Model Identifier:"):
            overview["model_identifier"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Total Number of Cores:"):
            value = stripped.split(":", 1)[1].strip()
            overview["core_summary"] = value
            try:
                overview["total_core_count"] = int(value.split(" ", 1)[0])
            except (ValueError, IndexError):
                pass
    return overview


def preferred_cpuinfo_value(cpuinfo_text: str) -> str | None:
    values: dict[str, str] = {}
    for line in cpuinfo_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key not in values:
            values[key] = value.strip()

    for key in ("model name", "Hardware", "vendor_id"):
        value = values.get(key)
        if value:
            return value
    return None


def cpu_vendor_or_soc() -> str:
    if platform.system() == "Darwin":
        apple_hardware = apple_hardware_overview()
        if "chip" in apple_hardware:
            return str(apple_hardware["chip"])
        brand = command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand != "not_available":
            return brand
        model = command_output(["sysctl", "-n", "hw.model"])
        if model != "not_available":
            return model

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        value = preferred_cpuinfo_value(cpuinfo.read_text(encoding="utf-8", errors="ignore"))
        if value is not None:
            return value
    return platform.processor() or "not_available"


def instruction_set_report() -> dict[str, Any]:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith(("flags", "Features")):
                return {
                    "features": sorted(line.split(":", 1)[1].strip().split()),
                    "status": "available",
                    "source": "/proc/cpuinfo",
                }

    if platform.system() == "Darwin":
        sources: list[str] = []
        errors: list[str] = []
        feature_values: set[str] = set()

        features = command_output(["sysctl", "-n", "machdep.cpu.features"])
        if features != "not_available":
            feature_values.update(feature.lower() for feature in features.split())
            sources.append("machdep.cpu.features")

        for sysctl_key, feature_name in DARWIN_ARM_FEATURE_KEYS.items():
            returncode, stdout, stderr = command_result(["sysctl", "-n", sysctl_key])
            if returncode == 0 and stdout.strip() in {"1", "true", "True", "yes"}:
                feature_values.add(feature_name)
                sources.append(sysctl_key)
            elif returncode != 0 and stderr:
                errors.append(f"{sysctl_key}: {stderr}")

        if feature_values:
            return {
                "features": sorted(feature_values),
                "status": "available",
                "source": sorted(set(sources)),
            }
        return {
            "features": [],
            "status": "not_available",
            "source": "darwin_sysctl",
            "reason": errors[0] if errors else "no Darwin CPU feature sysctl values were readable",
        }

    return {
        "features": [],
        "status": "not_available",
        "source": "not_available",
        "reason": f"no CPU feature probe is defined for {platform.system()}",
    }


def compiler_cpu_flags(build_info: dict[str, Any]) -> dict[str, Any]:
    sources = {
        "CFLAGS": os.environ.get("CFLAGS", "unset"),
        "CXXFLAGS": os.environ.get("CXXFLAGS", "unset"),
        "CPPFLAGS": os.environ.get("CPPFLAGS", "unset"),
        "ARCHFLAGS": os.environ.get("ARCHFLAGS", "unset"),
        "CMAKE_ARGS": os.environ.get("CMAKE_ARGS", "unset"),
        "SKBUILD_CMAKE_ARGS": os.environ.get("SKBUILD_CMAKE_ARGS", "unset"),
    }
    compiler_config = build_info.get("compiler_build_config", {})
    for key in ("CMAKE_CXX_FLAGS", "CMAKE_BUILD_TYPE"):
        sources[key] = compiler_config.get(key, "not_recorded")

    detected: list[str] = []
    for value in sources.values():
        if value in {"unset", "not_recorded", ""}:
            continue
        try:
            tokens = shlex.split(str(value))
        except ValueError:
            tokens = str(value).split()
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token in CPU_FLAG_TOKENS and index + 1 < len(tokens):
                detected.append(f"{token} {tokens[index + 1]}")
                index += 2
                continue
            if token.startswith(CPU_FLAG_PREFIXES):
                detected.append(token)
            index += 1

    return {
        "detected": sorted(set(detected)),
        # Record which allowlisted inputs were inspected, never their unrestricted
        # values: compiler variables can contain private include and build paths.
        "sources_checked": sorted(sources),
    }


def accelerator_build_mode(build_info: dict[str, Any]) -> str:
    """Return the Campaign 9 accelerator source-build boundary for a report."""

    cuda_enabled = bool(build_info.get("cuda_enabled", False))
    hip_enabled = bool(build_info.get("hip_enabled", False))
    metal_enabled = bool(build_info.get("metal_enabled", False))
    enabled_count = sum((cuda_enabled, hip_enabled, metal_enabled))
    if enabled_count > 1:
        raise ValueError(
            "unsupported mixed accelerator build metadata: Wolfgang uses "
            "target-specific accelerator builds"
        )
    if cuda_enabled:
        return "cuda_only"
    if hip_enabled:
        return "hip_only"
    if metal_enabled:
        return "metal_only"
    return "cpu_only"


def accelerator_backend_sets(build_info: dict[str, Any]) -> dict[str, list[str]]:
    compiled_accelerators = list(build_info.get("compiled_accelerator_backends", []))
    runtime_visible_accelerators = list(
        build_info.get("runtime_visible_accelerator_backends", [])
    )

    if not compiled_accelerators:
        if build_info.get("cuda_enabled", False):
            compiled_accelerators.append("cuda")
        if build_info.get("hip_enabled", False):
            compiled_accelerators.append("hip")
        if build_info.get("metal_enabled", False):
            compiled_accelerators.append("metal")

    if not runtime_visible_accelerators:
        if build_info.get("cuda_runtime_available", False):
            runtime_visible_accelerators.append("cuda")
        if build_info.get("hip_runtime_available", False):
            runtime_visible_accelerators.append("hip")
        if build_info.get("metal_runtime_available", False):
            runtime_visible_accelerators.append("metal")

    return {
        "compiled_accelerator_backends": compiled_accelerators,
        "runtime_visible_accelerator_backends": runtime_visible_accelerators,
        "compiled_backends": ["cpu", *compiled_accelerators],
        "runtime_visible_backends": ["cpu", *runtime_visible_accelerators],
    }


def benchmark_row_boundary(
    *,
    build_info: dict[str, Any],
    object_backend: str,
    transfer_boundary: str,
) -> dict[str, Any]:
    if object_backend not in ACCELERATOR_OBJECT_BACKENDS:
        raise ValueError("object_backend must be one of: cpu, cuda, hip, metal")
    if transfer_boundary not in ACCELERATOR_TRANSFER_BOUNDARIES:
        raise ValueError(
            "transfer_boundary must be one of: "
            + ", ".join(sorted(ACCELERATOR_TRANSFER_BOUNDARIES))
        )

    backend_sets = accelerator_backend_sets(build_info)
    return {
        "build_mode": accelerator_build_mode(build_info),
        "object_backend": object_backend,
        "compiled_backends": backend_sets["compiled_backends"],
        "runtime_visible_backends": backend_sets["runtime_visible_backends"],
        "transfer_boundary": transfer_boundary,
    }


def public_compiler_build_config(build_info: dict[str, Any]) -> dict[str, Any]:
    """Return compiler identity/version metadata without paths or raw flags."""

    config = build_info.get("compiler_build_config", {})
    if not isinstance(config, dict):
        return {}
    return {key: config[key] for key in sorted(PUBLIC_COMPILER_BUILD_KEYS) if key in config}


def benchmark_environment(build_info: dict[str, Any], *, numpy_version: str) -> dict[str, Any]:
    instruction_report = instruction_set_report()
    cpu_backend_build_flags = build_info.get("cpu_backend_build_flags", {"scalar": True})
    accelerator_sets = accelerator_backend_sets(build_info)
    environment = {
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": numpy_version,
        "compiler": command_first_line(["c++", "--version"]),
        "cmake": command_first_line(["cmake", "--version"]),
        "cpu_model": platform.processor() or "unknown",
        "cpu_architecture": platform.machine(),
        "cpu_vendor_or_soc": cpu_vendor_or_soc(),
        "physical_core_count": physical_core_count(),
        "logical_cpu_count": os.cpu_count(),
        "instruction_sets": instruction_report["features"],
        "instruction_set_probe": instruction_report,
        "active_fastpauli_cpu_backend": build_info["cpu_backend"],
        "requested_fastpauli_cpu_backend": build_info.get("requested_cpu_backend", "unknown"),
        "compiled_fastpauli_cpu_backends": build_info.get("compiled_cpu_backends", ["scalar"]),
        "available_fastpauli_cpu_backends": build_info.get("available_cpu_backends", ["scalar"]),
        "unavailable_fastpauli_cpu_backends": build_info.get("unavailable_cpu_backends", {}),
        "accelerator_build_mode": build_info.get(
            "accelerator_build_mode",
            accelerator_build_mode(build_info),
        ),
        "compiled_accelerator_backends": accelerator_sets["compiled_accelerator_backends"],
        "runtime_visible_accelerator_backends": accelerator_sets[
            "runtime_visible_accelerator_backends"
        ],
        "compiled_backends": accelerator_sets["compiled_backends"],
        "runtime_visible_backends": accelerator_sets["runtime_visible_backends"],
        "cpu_cmake_options": build_info.get("cpu_cmake_options", {}),
        "cpu_backend_build_flags": cpu_backend_build_flags,
        "compiler_build_config": public_compiler_build_config(build_info),
        "compiler_cpu_flags": compiler_cpu_flags(build_info),
        "oneTBB": {
            "enabled": bool(cpu_backend_build_flags.get("tbb", False)),
            "version": build_info.get("oneTBB_version", "not_available"),
        },
        "CUDA": {
            "enabled": build_info["cuda_enabled"],
        },
        "HIP": {
            "enabled": bool(build_info.get("hip_enabled", False)),
            "architectures": build_info.get("hip_architectures", "not_available"),
            "runtime_available": bool(build_info.get("hip_runtime_available", False)),
            "runtime_version": build_info.get("hip_runtime_version", "not_available"),
            "driver_version": build_info.get("hip_driver_version", "not_available"),
            "rocm_toolkit_version": build_info.get("rocm_toolkit_version", "not_available"),
        },
        "Metal": {
            "enabled": bool(build_info.get("metal_enabled", False)),
            "runtime_available": bool(build_info.get("metal_runtime_available", False)),
            "device_name": build_info.get("metal_device_name", "not_available"),
            "macos_version": build_info.get("metal_macos_version", "not_available"),
            "xcode_or_clt_version": build_info.get(
                "metal_xcode_or_clt_version",
                "not_available",
            ),
            "storage_mode": build_info.get("metal_storage_mode", "not_available"),
            "capability_summary": build_info.get(
                "metal_capability_summary",
                "not_available",
            ),
        },
        "thread_settings": {
            "controlled_thread_count": "not_controlled",
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "unset"),
        },
    }
    apple_hardware = apple_hardware_overview()
    if apple_hardware:
        environment["apple_hardware"] = apple_hardware
    return environment


def command_string() -> str:
    return " ".join([sys.executable, *sys.argv])
