#!/usr/bin/env python3
"""Render a resumable Blackwell/B300 remote qualification runner."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path


_TEMPLATE = r'''#!/usr/bin/env bash
set -euo pipefail

commit=__COMMIT__
source_dir=__SOURCE_DIR__
evidence=__EVIDENCE_ROOT__
export commit source_dir evidence
STATE_DIR="$evidence/private/state"
mkdir -p "$evidence/private/logs" "$evidence/private/raw/dist" "$evidence/private/profiler" "$STATE_DIR"

run_step() {
  local name="$1"
  shift
  local stamp="$STATE_DIR/${name}.done"
  if [[ -f "$stamp" ]]; then
    echo "[resume] skipping $name"
    return 0
  fi
  "$@"
  touch "$stamp"
}

cd "$source_dir"
export FASTPAULI_VALIDATE_CUDA=1
export WOLFGANG_CUDA_ARCHITECTURES='100-real;120'
export EVIDENCE_ROOT="$evidence"

run_step install_cuda_python_deps sh -c 'python -m pip install --upgrade pip setuptools wheel build scikit-build-core nanobind > "$evidence/private/logs/pip-bootstrap.log" 2>&1 && python -m pip install "cupy-cuda12x>=13,<14" "torch>=2.4,<2.7" > "$evidence/private/logs/cuda-python-deps.log" 2>&1'
run_step install_editable sh -c 'python -m pip install -e .[test] --config-settings=cmake.define.WOLFGANG_ENABLE_INTERNAL_BINDINGS=ON --config-settings=cmake.define.WOLFGANG_ENABLE_CUDA=ON --config-settings=cmake.define.WOLFGANG_ENABLE_HIP=OFF --config-settings=cmake.define.WOLFGANG_ENABLE_METAL=OFF --config-settings=cmake.define.WOLFGANG_CUDA_ARCHITECTURES=100-real\;120 --config-settings=cmake.define.WOLFGANG_ENABLE_NATIVE=OFF > "$evidence/private/logs/cuda-install.log" 2>&1'
run_step record_build_info sh -c 'python - <<"PY" > "$evidence/private/logs/build-info.txt"
import fastpauli._fastpauli_core as core
print(core._build_info())
print(core._cuda_status())
print(core._accelerator_status())
PY
'
run_step validate sh -c 'python scripts/validate.py > "$evidence/private/logs/validate-cuda.log" 2>&1'
run_step affected_cuda_suite sh -c 'python -m pytest tests/test_competitive_baselines_benchmark.py tests/test_cuda_scaling_benchmark.py::test_cuda_campaign10_profiles_emit_non_deferred_cpu_unavailable_rows tests/test_phase11_cuda_kernels.py tests/test_dlpack_and_cuda_interop_contract.py -q > "$evidence/private/logs/affected-cuda-suite.log" 2>&1'
run_step build_dist sh -c 'python -m build --sdist --wheel --outdir "$evidence/private/raw/dist" --no-isolation > "$evidence/private/logs/build-dist.log" 2>&1'
run_step hash_artifacts python - <<'PY'
import hashlib, json, os
from pathlib import Path

root = Path(os.environ["EVIDENCE_ROOT"])
dist = root / "private/raw/dist"
wheel = next(dist.glob("*.whl"))
sdist = next(dist.glob("*.tar.gz"))
(root / "private/raw/build_artifact_hashes.json").write_text(
    json.dumps(
        {
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "sdist_sha256": hashlib.sha256(sdist.read_bytes()).hexdigest(),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
run_step full_memcheck sh -c 'compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest tests/test_phase11_cuda_kernels.py -q > "$evidence/private/profiler/compute_sanitizer_memcheck.log" 2>&1'
run_step full_racecheck sh -c 'compute-sanitizer --tool racecheck --error-exitcode 99 python -m pytest tests/test_phase11_cuda_kernels.py -q > "$evidence/private/profiler/compute_sanitizer_racecheck.log" 2>&1'
run_step collect_phase11_nodeids sh -c 'python -m pytest tests/test_phase11_cuda_kernels.py --collect-only -q > "$evidence/private/logs/phase11-collect.log" 2>&1 && python - <<"PY"
import json, os
from pathlib import Path
root = Path(os.environ["EVIDENCE_ROOT"])
collect_log = root / "private/logs/phase11-collect.log"
nodeids = [
    line.strip()
    for line in collect_log.read_text().splitlines()
    if line.strip().startswith("tests/test_phase11_cuda_kernels.py::")
]
(root / "private/raw/phase11_nodeids.json").write_text(json.dumps(nodeids, indent=2) + "\n", encoding="utf-8")
PY
'
run_step per_test_memcheck sh -c 'python - <<"PY"
import json, os, subprocess
from pathlib import Path
root = Path(os.environ["EVIDENCE_ROOT"])
nodeids = json.loads((root / "private/raw/phase11_nodeids.json").read_text())
rows = []
for idx, nodeid in enumerate(nodeids, start=1):
    log_path = root / "private/profiler" / f"per-test-memcheck-{idx:02d}.log"
    completed = subprocess.run(
        ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99", "python", "-m", "pytest", "-q", nodeid],
        text=True,
        capture_output=True,
    )
    text = (completed.stdout or "") + (completed.stderr or "")
    log_path.write_text(text, encoding="utf-8")
    summary = ""
    for line in reversed(text.splitlines()):
        if " passed" in line or " skipped" in line or "failed" in line:
            summary = line.strip()
            break
    rows.append({"i": idx, "nodeid": nodeid, "exit_code": completed.returncode, "leak": "nanobind: leaked" in text, "error_summary_zero": "ERROR SUMMARY: 0 errors" in text, "summary": summary})
(root / "private/raw/per_test_memcheck_leaks.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
PY
'
run_step benchmark_smoke_1 sh -c 'python benchmarks/bench_cuda_kernels.py --smoke --warmup 10 --repeat 30 --json --output "$evidence/private/raw/bench_cuda_kernels_smoke_run1.json" > "$evidence/private/logs/bench_cuda_kernels_smoke_run1.log" 2>&1'
run_step benchmark_smoke_2 sh -c 'python benchmarks/bench_cuda_kernels.py --smoke --warmup 10 --repeat 30 --json --output "$evidence/private/raw/bench_cuda_kernels_smoke_run2.json" > "$evidence/private/logs/bench_cuda_kernels_smoke_run2.log" 2>&1'
run_step benchmark_smoke_3 sh -c 'python benchmarks/bench_cuda_kernels.py --smoke --warmup 10 --repeat 30 --json --output "$evidence/private/raw/bench_cuda_kernels_smoke_run3.json" > "$evidence/private/logs/bench_cuda_kernels_smoke_run3.log" 2>&1'
run_step derive_public sh -c 'python scripts/b300_blackwell_validation_artifacts.py --evidence-root "$evidence" --commit "$commit" > "$evidence/private/logs/derive-public.log" 2>&1 && python scripts/audit_public_artifacts.py --path "$evidence/public" > "$evidence/private/logs/public-audit.log" 2>&1'
run_step final_nvidia_smi sh -c 'nvidia-smi > "$evidence/private/logs/nvidia-smi.after.log" 2>&1'
'''


def render_remote_script(*, commit: str, source_dir: str, evidence_root: str) -> str:
    return (
        _TEMPLATE.replace("__COMMIT__", shlex.quote(commit))
        .replace("__SOURCE_DIR__", shlex.quote(source_dir))
        .replace("__EVIDENCE_ROOT__", shlex.quote(evidence_root))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    script = render_remote_script(
        commit=args.commit,
        source_dir=args.source_dir,
        evidence_root=args.evidence_root,
    )
    if args.output is None:
        print(script)
        return
    args.output.write_text(script, encoding="utf-8")
    args.output.chmod(0o755)


if __name__ == "__main__":
    main()
