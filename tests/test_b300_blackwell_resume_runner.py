from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "b300_blackwell_resume_runner.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("b300_blackwell_resume_runner", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_runner_uses_reproducible_build_path_after_semantic_gates() -> None:
    runner = load_runner_module()
    script = runner.render_remote_script(
        commit="0123456789abcdef0123456789abcdef01234567",
        source_dir="/tmp/src",
        evidence_root="/tmp/evidence",
    )

    assert "python -m build --sdist --wheel --outdir \"$evidence/private/raw/dist\" --no-isolation" in script
    assert "pip download --no-deps --no-binary :all:" not in script
    assert script.index('python scripts/validate.py > "$evidence/private/logs/validate-cuda.log" 2>&1') < script.index(
        'python -m build --sdist --wheel --outdir "$evidence/private/raw/dist" --no-isolation > "$evidence/private/logs/build-dist.log" 2>&1'
    )
    assert script.index('python -m pytest tests/test_competitive_baselines_benchmark.py tests/test_cuda_scaling_benchmark.py::test_cuda_campaign10_profiles_emit_non_deferred_cpu_unavailable_rows tests/test_phase11_cuda_kernels.py tests/test_dlpack_and_cuda_interop_contract.py -q > "$evidence/private/logs/affected-cuda-suite.log" 2>&1') < script.index(
        'python -m build --sdist --wheel --outdir "$evidence/private/raw/dist" --no-isolation > "$evidence/private/logs/build-dist.log" 2>&1'
    )
    assert 'python scripts/b300_blackwell_validation_artifacts.py --evidence-root "$evidence" --commit "$commit" > "$evidence/private/logs/derive-public.log" 2>&1' in script
    assert 'python scripts/audit_public_artifacts.py --path "$evidence/public" > "$evidence/private/logs/public-audit.log" 2>&1' in script
    assert 'compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest tests/test_phase11_cuda_kernels.py -q > "$evidence/private/profiler/compute_sanitizer_memcheck.log" 2>&1' in script


def test_resume_runner_is_resumable_by_phase_stamp() -> None:
    runner = load_runner_module()
    script = runner.render_remote_script(
        commit="0123456789abcdef0123456789abcdef01234567",
        source_dir="/tmp/src",
        evidence_root="/tmp/evidence",
    )

    assert 'STATE_DIR="$evidence/private/state"' in script
    assert "export commit source_dir evidence" in script
    assert 'run_step() {' in script
    assert 'stamp="$STATE_DIR/${name}.done"' in script
    assert 'if [[ -f "$stamp" ]]; then' in script
    assert 'run_step benchmark_smoke_3' in script
