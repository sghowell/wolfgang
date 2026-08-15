"""Compatibility package for the legacy fastpauli import path."""
from __future__ import annotations

import warnings

from wolfgang_quantum import *  # noqa: F403
from wolfgang_quantum import WolfgangCapabilities as WolfgangCapabilities
from wolfgang_quantum import __all__ as wolfgang_all

from . import _fastpauli_core as _fastpauli_core

warnings.warn(
    "fastpauli is deprecated; import wolfgang_quantum instead.",
    DeprecationWarning,
    stacklevel=2,
)

FastPauliCapabilities = WolfgangCapabilities

__all__ = [*wolfgang_all, "FastPauliCapabilities", "_fastpauli_core"]
