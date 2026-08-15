"""Policy tests for the native Python binding layers."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_stable_and_internal_bindings_have_separate_registration_units() -> None:
    expected = {
        "bindings/python/bindings.hpp",
        "bindings/python/stable_bindings.cpp",
        "bindings/python/internal_bindings.cpp",
    }
    for relative_path in expected:
        assert (ROOT / relative_path).is_file(), f"missing binding layer: {relative_path}"

    module = (ROOT / "bindings/python/module.cpp").read_text(encoding="utf-8")
    assert "register_stable_bindings(module)" in module
    assert "register_internal_bindings(module)" in module
    assert module.index("register_stable_bindings(module)") < module.index(
        "register_internal_bindings(module)"
    )


def test_stable_registration_does_not_define_private_diagnostic_hooks() -> None:
    stable_sources = (
        ROOT / "bindings/python/stable_bindings.cpp",
        ROOT / "bindings/python/pauli_sum_py.cpp",
    )
    private_registration = re.compile(
        r'module\.def\(\s*"_|"_[^"]*(?:for_testing|packed_words)|campaign\d'
    )
    offenders: list[str] = []
    for path in stable_sources:
        if path.exists() and private_registration.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, "private hooks belong in internal_bindings.cpp: " + ", ".join(offenders)


def test_release_builds_explicitly_disable_internal_bindings() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    policy = ROOT / "docs/quality/python_binding_policy.md"

    assert "WOLFGANG_ENABLE_INTERNAL_BINDINGS" in cmake
    assert "WOLFGANG_ENABLE_INTERNAL_BINDINGS" in cmake
    assert re.search(
        r"option\(\s*WOLFGANG_ENABLE_INTERNAL_BINDINGS[\s\S]*?\sOFF\)", cmake
    )
    assert "bindings/python/internal_bindings.cpp" in cmake
    assert '"cmake.define.WOLFGANG_ENABLE_INTERNAL_BINDINGS" = "OFF"' in pyproject
    assert '"cmake.define.WOLFGANG_ENABLE_INTERNAL_BINDINGS" = "OFF"' in pyproject
    assert policy.is_file()
    policy_text = policy.read_text(encoding="utf-8")
    assert "unsupported" in policy_text.lower()
    assert "release wheels" in policy_text.lower()
