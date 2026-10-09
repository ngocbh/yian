"""TOOL — a direct, deterministic invocation of one tool from the run's `ToolSet`.

No LLM in the loop (distinct from an AGENT node's tool loop). The node's `input:` entries are
(untyped) bindings like any other kind's, resolved by the engine's `eval_node` bind
seam — a whole-string `${ref}` resolves against the pool, a literal passes through,
and every value is deep-copied (the tool sees only its own record `inputs`, never
the live pool). The tool is looked up in the run's `ToolSet` (the global registry when
the run passed no `tools=`, otherwise only the run's mapping) and invoked through its
public schema.
"""

from yian.nodes.base import Node, NodeKind, Output
from yian.tools import ToolSet


class ToolNode(Node):
    """
    A direct, deterministic invocation of one tool from the run's `ToolSet` (no LLM in the loop).

    The node's bound input record is passed to the tool's `.invoke(...)`; the tool sees
    only its own record, never the live pool.

    Args:
        node_id (`str`):
            The node's unique id.
        tool_id (`str`):
            The tool id, looked up in the run's `ToolSet`; an unknown id raises at run time.
        title (`str`, *optional*, defaults to `None`):
            Display title.
    """

    kind = NodeKind.TOOL
    needs_tools = True

    def __init__(self, node_id: str, *, tool_id: str, title=None) -> None:
        super().__init__(node_id, title=title)
        self.tool_id = tool_id

    def forward(self, inputs: dict, *, tools: ToolSet | None = None) -> Output:
        tools = tools if tools is not None else ToolSet()
        return Output(value=tools.invoke(self.tool_id, inputs))
