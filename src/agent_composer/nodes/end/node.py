"""END_ID — the flow's return boundary, internal-only.

The loader synthesizes ONE EndNode per flow as its single terminal; the engine's run result is
END_ID's committed value.

Two modes:
- RECORD (a flow's `output:`): params one per declared output; `run` reproduces
  terminal_output()'s arity — 0 -> None, 1 -> bare inputs[name], >=2 -> {name: inputs[name]}.
- LIST (a MAP fan-in): params e0..e{n-1}, each <- a MAP element's child
  END_ID; `run` -> Output([inputs[ei]...]) in over order (index); n=0 -> [].

END_ID carries the flow's POST asserts (${X.output}); the ordinary node wrapper fires them
after END builds its output. A required terminal output co-skipped is a normal
required-data-group co-skip derived by the runtime reducer's edge dispositions — END_ID has
no special co-skip logic. Hidden from authors: a reserved __-prefixed id, never parseable.
"""

from typing import Optional

from agent_composer.nodes.base import Node, NodeKind, Output
from agent_composer.nodes.binding import ParamDecl


class EndNode(Node):
    """
    The flow's return boundary — internal-only, loader-synthesized.

    One per flow (the single terminal); the run result is this node's committed value. RECORD mode
    (a flow's `output:`) reproduces arity — 0 inputs -> None, 1 -> the bare value, >=2 -> a keyed
    dict; LIST mode (a MAP fan-in) joins the element outputs into a list in `over` order. Use the
    [`record`][agent_composer.nodes.end.node.EndNode.record] /
    [`list_`][agent_composer.nodes.end.node.EndNode.list_] factories rather than the ctor.

    Args:
        node_id (`str`):
            The node's id (`ID` at top level; deep-namespaced in a child).
        output_names (`list[str]`, *optional*, defaults to `None`):
            RECORD mode: one param per declared output. Mutually exclusive with `n`.
        n (`int`, *optional*, defaults to `None`):
            LIST mode: the element count. Mutually exclusive with `output_names`.
        title (`str`, *optional*, defaults to `None`):
            Display title.

    Raises:
        ValueError: If not exactly one of `output_names` / `n` is given.
    """

    kind = NodeKind.END

    #: Reserved id of the synthesized return boundary (the old `__end__` sentinel).
    #: The canonical source of truth; `compile.model` re-exports it as `END_ID`/`END_ID`.
    #: One per flow (deep-namespaced in a child); the run result is `store[<this id>]`.
    #: Authors can never name a node this (parser reserved-id ban).
    ID = "__end__"

    def __init__(self, node_id: str, *, output_names: Optional[list[str]] = None,
                 n: Optional[int] = None, title: Optional[str] = None) -> None:
        # Exactly one of `output_names` (record mode) / `n` (list mode). Use the `record`/`list_`
        # factories rather than this ctor directly.
        super().__init__(node_id, title=title)
        if (output_names is None) == (n is None):
            raise ValueError("EndNode takes exactly one of output_names= (record) or n= (list)")
        if output_names is not None:
            self._mode = "record"
            self._names = list(output_names)
            self.params = [ParamDecl(name=name) for name in self._names]
        else:
            self._mode = "list"
            self._n = n
            self.params = [ParamDecl(name=f"e{i}") for i in range(n)]

    @classmethod
    def record(cls, node_id: str, *, output_names: list[str],
               title: Optional[str] = None) -> "EndNode":
        """Build a RECORD-mode terminal — one param per declared flow `output:` name."""
        return cls(node_id, output_names=output_names, title=title)

    @classmethod
    def list_(cls, node_id: str, *, n: int, title: Optional[str] = None) -> "EndNode":
        """Build a LIST-mode terminal — `n` params (`e0..e{n-1}`) joined as a MAP fan-in."""
        return cls(node_id, n=n, title=title)

    def _run_impl(self, inputs: dict) -> Output:
        if self._mode == "record":
            # terminal arity rule: 0 -> None; 1 -> bare; >=2 -> keyed object.
            if not self._names:
                return Output(value=None)
            if len(self._names) == 1:
                return Output(value=inputs[self._names[0]])
            return Output(value={name: inputs[name] for name in self._names})
        # list mode (the MAP fan-in): join in over order (index); n=0 -> [].
        return Output(value=[inputs[f"e{i}"] for i in range(self._n)])
