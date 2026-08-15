"""Compatibility layer for legacy fastpauli capability imports."""
from wolfgang_quantum._capabilities import *  # noqa: F403
from wolfgang_quantum._capabilities import WolfgangCapabilities as WolfgangCapabilities

FastPauliCapabilities = WolfgangCapabilities
