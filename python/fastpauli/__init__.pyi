from wolfgang_quantum import *  # noqa: F403
from wolfgang_quantum import WolfgangCapabilities as WolfgangCapabilities
from wolfgang_quantum import __all__ as wolfgang_all

FastPauliCapabilities = WolfgangCapabilities

__all__ = [*wolfgang_all, "FastPauliCapabilities"]
