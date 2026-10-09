"""Agent modes — the loop / prompting method (axis 1; one per node).

The mode contract lives in `common`, helper functions in `utils`; each concrete
mode lives in its own module and self-registers on import. Importing this package
populates `MODES` (before compile). Add a mode: drop a module here that
`@register_mode("name")`s a function (importing the contract from `.common`),
then import it below.

Tools are the node's explicitly authored grants. ``plain`` ignores them;
``tool_calling`` binds and executes them in returned order. The run's
``ask_user`` follows the same path and blocks inside the live node call.
"""

from yian.nodes.agent.modes.common import (
    DEFAULT_SYSTEM,
    AgentLoopError,
    AgentMode,
    AgentRunContext,
    MODES,
    register_mode,
)
from yian.nodes.agent.modes.utils import text_of
from yian.nodes.agent.modes import plain, tool_calling  # noqa: F401  register on import

__all__ = [
    "AgentRunContext",
    "AgentMode",
    "AgentLoopError",
    "MODES",
    "register_mode",
    "DEFAULT_SYSTEM",
    "text_of",
    "plain",
    "tool_calling",
]
