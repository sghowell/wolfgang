#!/usr/bin/env python3
"""Capture ROCm Campaign 5 temporary HIP DLPack candidate evidence.

This script is intentionally for candidate builds only. The retained public
ROCm build keeps HIP DLPack unavailable; Campaign 5 uses this probe to record
why the temporary kDLROCM export was rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import wolfgang_quantum._wolfgang_core as core
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.bench_rocm_kernels import CAMPAIGN5_INTEROP_CASES, make_operator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the candidate probe JSON artifact.",
    )
    parser.add_argument(
        "--public-base-git-commit",
        required=True,
        help="Public retained-build commit that the temporary candidate patch was based on.",
    )
    return parser.parse_args()


def git_short_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def build_artifact(public_base_git_commit: str) -> dict[str, Any]:
    case = CAMPAIGN5_INTEROP_CASES[0]
    lhs = make_operator(case, side="lhs")
    rhs = make_operator(case, side="rhs")
    expected = np.asarray(
        lhs.commutes_with(rhs),
        dtype=np.bool_,
    ).reshape(lhs.num_terms, rhs.num_terms)
    expected_u8 = expected.astype(np.uint8, copy=False)
    host_sum_before = int(expected_u8.sum(dtype=np.uint64))
    digest = hashlib.sha256(np.asarray(expected_u8, order="C").tobytes()).hexdigest()

    lhs_device = lhs.to_device(device=0)
    rhs_device = rhs.to_device(device=0)
    matrix = lhs_device.commutes_with_device(rhs_device)

    tensor = torch.utils.dlpack.from_dlpack(matrix)
    torch.cuda.synchronize()
    consumer_sum_before = int(tensor.sum().item())
    correctness_passed = (
        consumer_sum_before == host_sum_before and tuple(tensor.shape) == expected.shape
    )

    flat = tensor.reshape(-1)
    first_value_before = int(flat[0].item())
    first_value_after_target = 0 if first_value_before else 1
    mutation_error = ""
    try:
        flat[0] = first_value_after_target
        torch.cuda.synchronize()
        mutation_result = "accepted_mutation"
    except Exception as exc:
        mutation_result = "raised_exception"
        mutation_error = f"{type(exc).__name__}: {exc}"

    after_host = np.asarray(
        matrix.to_host(),
        dtype=np.bool_,
    ).reshape(expected.shape).astype(np.uint8, copy=False)
    host_sum_after = int(after_host.sum(dtype=np.uint64))
    first_value_after = int(after_host.reshape(-1)[0])
    mutation_changed_device_buffer = (
        mutation_result == "accepted_mutation"
        and first_value_after == first_value_after_target
        and first_value_after != first_value_before
    )
    read_only_enforced = mutation_result != "accepted_mutation" or not mutation_changed_device_buffer

    status = core._hip_status()
    devices = status.get("devices", [])
    first_device = devices[0] if devices else {}

    return {
        "artifact_type": "candidate_dlpack_probe",
        "campaign": "rocm_mi300x_campaign5",
        "evidence_kind": "temporary_candidate_build",
        "final_status": "rejected_with_evidence",
        "git_commit": git_short_head(),
        "public_base_git_commit": public_base_git_commit,
        "temporary_candidate_patch": (
            "src/hip/device_commutation_matrix.hip.cpp returned impl_->data from "
            "data_pointer_for_dlpack(), returned DLPack device type kDLROCM (10), "
            "and the temporary binding accepted PyTorch's stream=0 token; this "
            "patch is not retained in public FastPauli because the consumer did "
            "not enforce read-only views."
        ),
        "command": (
            "PATH=/opt/rocm/bin:$PATH .venv/bin/python "
            "scripts/rocm_campaign5_candidate_probe.py --output <artifact> "
            "--public-base-git-commit <commit>"
        ),
        "consumer_library": "torch",
        "consumer_version": getattr(torch, "__version__", "unknown"),
        "consumer_backend": "rocm" if getattr(torch.version, "hip", None) else "not_rocm",
        "consumer_hip_version": getattr(torch.version, "hip", None),
        "consumer_available": bool(getattr(torch.version, "hip", None) and torch.cuda.is_available()),
        "consumer_import_error": "",
        "device_name": first_device.get("name", "unknown"),
        "gfx_target": first_device.get("gfx_target", "unknown"),
        "dataset": {
            "case": case["name"],
            "num_qubits": int(case["num_qubits"]),
            "lhs_terms": int(lhs.num_terms),
            "rhs_terms": int(rhs.num_terms),
            "entries": int(lhs.num_terms * rhs.num_terms),
            "term_weight": int(case["term_weight"]),
            "random_seed": int(case["random_seed"]),
        },
        "dlpack_device_type": 10,
        "dlpack_device_type_name": "kDLROCM",
        "tensor_shape": [int(dim) for dim in tensor.shape],
        "tensor_dtype": str(tensor.dtype),
        "candidate_probe_consumer_correctness_passed": bool(correctness_passed),
        "candidate_probe_consumer_read_only_enforced": bool(read_only_enforced),
        "candidate_probe_mutation_result": mutation_result,
        "consumer_mutation_error": mutation_error,
        "mutation_changed_device_buffer": bool(mutation_changed_device_buffer),
        "first_value_before": first_value_before,
        "first_value_after_target": first_value_after_target,
        "first_value_after": first_value_after,
        "host_sum_before": host_sum_before,
        "consumer_sum_before": consumer_sum_before,
        "host_sum_after": host_sum_after,
        "canonical_matrix_hash_before": digest,
    }


def main() -> None:
    args = parse_args()
    artifact = build_artifact(args.public_base_git_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
