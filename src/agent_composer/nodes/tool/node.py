"""TOOL — a direct, deterministic invocation of one `TOOL_REGISTRY` entry.

No LLM in the loop (distinct from an AGENT node's tool loop). The node's `input:` entries are
(untyped) bindings like any other kind's, resolved by the engine's `eval_node` bind
seam — a whole-string `${ref}` resolves against the pool, a literal passes through,
and every value is deep-copied (the tool sees only its own record `inputs`, never
the live pool). The registry tool is a LangChain `BaseTool`, invoked via
`.invoke(inputs)`.
"""

from agent_composer.nodes.base import Node, NodeKind, Output
from agent_composer.tools import invoke_tool


class ToolNode(Node):
    """
    A direct, deterministic invocation of one `TOOL_REGISTRY` entry (no LLM in the loop).

    The node's bound input record is passed to the registry tool's `.invoke(...)`; the tool sees
    only its own record, never the live pool.

    Args:
        node_id (`str`):
            The node's unique id.
        tool_id (`str`):
            The registry key of the tool to invoke; an unknown id raises at run time.
        title (`str`, *optional*, defaults to `None`):
            Display title.
    """

    kind = NodeKind.TOOL

    def __init__(self, node_id: str, *, tool_id: str, title=None) -> None:
        super().__init__(node_id, title=title)
        self.tool_id = tool_id

    def forward(self, inputs: dict) -> Output:
        return Output(value=invoke_tool(self.tool_id, inputs))
