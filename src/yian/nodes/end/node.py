"""END_ID — the flow's return boundary, internal-only.

The loader synthesizes ONE EndNode per flow as its single terminal; the engine's run result is
END_ID's committed value.

Three modes:
- VALUE (a flow's `output: ${x}`, or no `output:`): at most one param; `run` returns
  inputs[name], or None without a name.
- RECORD (a flow's `output: {...}`): params one per declared output; `run` returns
  {name: inputs[name]} for every name, so one field gives a one-key dict and none gives {}.
- LIST (a MAP fan-in): params e0..e{n-1}, each <- a MAP element's child
  END_ID; `run` -> Output([inputs[ei]...]) in over order (index); n=0 -> [].

END_ID carries the flow's POST asserts (${X.output}); the ordinary node wrapper fires them
after END builds its output. A required terminal output co-skipped is a normal
required-data-group co-skip derived by the runtime reducer's edge dispositions — END_ID has
no special co-skip logic. Hidden from authors: a reserved __-prefixed id, never parseable.
"""

from typing import Optional

from yian.expr.lineage import FeedbackValue
from yian.nodes.base import (
    BackwardContext, BackwardResult, Node, NodeKind, Output, _project_structural_feedback,
)
from yian.nodes.contracts import Parameter


class EndNode(Node):
    """
    The flow's return boundary — internal-only, loader-synthesized.

    One per flow (the single terminal); the run result is this node's committed value. VALUE mode
    (a flow's `output: ${x}`, or no `output:`) returns its one input, or `None` without one;
    RECORD mode (a flow's `output: {...}`) returns a dict keyed by every declared name, even for
    one name or none; LIST mode (a MAP fan-in) joins the element outputs into a list in
    `over` order. Use the [`value`][yian.nodes.end.node.EndNode.value] /
    [`record`][yian.nodes.end.node.EndNode.record] /
    [`list_`][yian.nodes.end.node.EndNode.list_] factories rather than the ctor.

    Args:
        node_id (`str`):
            The node's id (`ID` at top level; deep-namespaced in a child).
        output_names (`list[str]`, *optional*, defaults to `None`):
            RECORD or VALUE mode: one param per declared output. Mutually exclusive with `n`.
        n (`int`, *optional*, defaults to `None`):
            LIST mode: the element count. Mutually exclusive with `output_names`.
        title (`str`, *optional*, defaults to `None`):
            Display title.
        value (`bool`, *optional*, defaults to `False`):
            With `output_names`, selects VALUE mode instead of RECORD mode.

    Raises:
        ValueError: If not exactly one of `output_names` / `n` is given, or if `value` is set
            with more than one output name or without `output_names`.
    """

    kind = NodeKind.END

    #: Reserved id of the synthesized return boundary.
    #: The canonical source of truth; `compile.model` re-exports it as `END_ID`/`END_ID`.
    #: One per flow (deep-namespaced in a child); the run result is `store[<this id>]`.
    #: Authors can never name a node this (parser reserved-id ban).
    ID = "__end__"

    def __init__(self, node_id: str, *, output_names: Optional[list[str]] = None,
                 n: Optional[int] = None, title: Optional[str] = None,
                 value: bool = False) -> None:
        # Exactly one of `output_names` (value/record mode) / `n` (list mode). Use the
        # `value`/`record`/`list_` factories rather than this ctor directly.
        super().__init__(node_id, title=title)
        if (output_names is None) == (n is None):
            raise ValueError(
                "EndNode takes exactly one of output_names= (value/record) or n= (list)"
            )
        if value and (output_names is None or len(output_names) > 1):
            raise ValueError("EndNode value mode takes at most one output name")
        if output_names is not None:
            self._mode = "value" if value else "record"
            self._names = list(output_names)
            self.params = [Parameter(name=name) for name in self._names]
        else:
            self._mode = "list"
            self._n = n
            self.params = [Parameter(name=f"e{i}") for i in range(n)]

    @classmethod
    def value(cls, node_id: str, *, output_name: Optional[str],
              title: Optional[str] = None) -> "EndNode":
        """Build a VALUE-mode terminal that returns its one input as the flow's value.

        Args:
            node_id (`str`):
                The node's id.
            output_name (`str`, *optional*):
                The one param's name, or `None` for a terminal with no params that returns `None`.
            title (`str`, *optional*, defaults to `None`):
                Display title.

        Returns:
            `EndNode`: The VALUE-mode terminal.
        """
        names = [] if output_name is None else [output_name]
        return cls(node_id, output_names=names, title=title, value=True)

    @classmethod
    def record(cls, node_id: str, *, output_names: list[str],
               title: Optional[str] = None) -> "EndNode":
        """Build a RECORD-mode terminal — one param per declared flow `output:` name."""
        return cls(node_id, output_names=output_names, title=title)

    @classmethod
    def list_(cls, node_id: str, *, n: int, title: Optional[str] = None) -> "EndNode":
        """Build a LIST-mode terminal — `n` params (`e0..e{n-1}`) joined as a MAP fan-in."""
        return cls(node_id, n=n, title=title)

    @property
    def returns_record(self) -> bool:
        """Whether this terminal is in RECORD mode (a flow's `output: {...}`)."""
        return self._mode == "record"

    def forward(self, inputs: dict) -> Output:
        if self._mode == "value":
            return Output(value=inputs[self._names[0]] if self._names else None)
        if self._mode == "record":
            return Output(value={name: inputs[name] for name in self._names})
        # list mode (the MAP fan-in): join in over order (index); n=0 -> [].
        return Output(value=[inputs[f"e{i}"] for i in range(self._n)])

    def backward_input_names(self) -> tuple[str, ...]:
        """Return actual output contributors, excluding augmented assertion params.

        Returns:
            `tuple[str, ...]`: Stored value or record output names, or ordered `e0`
                through `eN` list slots. Empty output layouts have an empty inventory.
        """
        if self._mode != "list":
            return tuple(self._names)
        return tuple(f"e{index}" for index in range(self._n))

    def capture_backward_context(self) -> dict:
        """Capture the output layout (value, record fields, or list positions) before the body executes.

        Returns:
            `dict`: The `input_output_paths` mapping: no paths for zero outputs,
                `[]` for a VALUE terminal's one input, a named path for each record
                field, or indexed paths for a list. Assertion inputs are excluded.
        """
        names = self.backward_input_names()
        if self._mode == "list":
            paths = {name: [index] for index, name in enumerate(names)}
        elif self._mode == "record":
            paths = {name: [name] for name in names}
        else:
            paths = {name: [] for name in names}
        return {"input_output_paths": paths}

    def backward(
        self, context: BackwardContext, feedback: tuple[FeedbackValue, ...],
    ) -> BackwardResult:
        """Project through the recorded layout without consulting mutable node fields.

        Args:
            context (`BackwardContext`): Recorded inputs, output or failure,
                actual output-name inventory, and captured output paths.
            feedback (`tuple[FeedbackValue, ...]`): Ordered text or structural
                selections over the accepted output. Text stays unchanged when
                sent to multiple contributors; a failed END accepts text or `None`.

        Returns:
            `BackwardResult`: Independent input message batches. Zero-output and
                empty-list layouts have no contributors; missing bindings stop.

        Raises:
            `TypeError`: If a message batch or captured path has an invalid type.
            `ValueError`: If feedback or a captured path does not match the
                recorded output, or the required structural layout is absent.
        """
        return _project_structural_feedback(context, feedback)
