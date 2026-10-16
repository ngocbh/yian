"""TOOL — a direct, deterministic invocation of one tool from the run's `ToolSet`.

No LLM in the loop (distinct from an AGENT node's tool loop). The node's `input:` entries are
bindings like any other kind's, resolved by the engine's `eval_node` bind
seam — a whole-string `${ref}` resolves against the pool, a literal passes through,
and every value is deep-copied (the tool sees only its own record `inputs`, never
the live pool). The tool is looked up in the run's `ToolSet` (the global registry when
the run passed no `tools=`, otherwise only the run's mapping) and invoked through its
public schema. An optional `output:` declaration sets the node's output contract, which the
engine checks against the tool's return value on every call.
"""

from types import MappingProxyType

from yian.nodes.base import Node, NodeKind, Output
from yian.nodes.contracts import HostToolCall, output_contract_type
from yian.tools import ToolSet
from yian.typesys import Type


class ToolNode(Node):
    """
    A direct, deterministic invocation of one tool from the run's `ToolSet` (no LLM in the loop).

    The node's bound input record is passed to `ToolSet.invoke`; the tool sees only its
    own record, never the live pool. A tool that declares a `ToolContext` first
    parameter also receives one naming this node, with the call id
    `"<run_id>/<node id>"`, which a resumed re-run of the node sees again. When
    `output_contract` is set (from an optional authored `output:`), the engine
    validates every returned value against it.

    `input_types` maps each input name to the type the loader proved its binding has
    (a literal's exact type, a typed binding's declared type, or a reference's inferred
    type); an input whose type is not proven is absent.

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
        self.input_types: dict[str, Type] = {}

    def host_tool_call(self) -> HostToolCall:
        """Return this node's tool id, input names, proven input types, and output type."""
        return HostToolCall(
            tool_id=self.tool_id,
            inputs=tuple(parameter.name for parameter in self.params or ()),
            input_types=MappingProxyType(self.input_types),
            output_type=output_contract_type(self.output_contract),
        )

    def forward(self, inputs: dict, *, tools: ToolSet | None = None) -> Output:
        tools = tools if tools is not None else ToolSet()
        return Output(value=tools.invoke(self.tool_id, inputs, node=self.id))
