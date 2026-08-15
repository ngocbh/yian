"""START_ID — the flow's parameter-binding boundary, internal-only. PURE.

The loader synthesizes ONE StartNode per flow from its `input:` decls; it is the flow's single
root. `self.params` are the input NAMES (ParamDecl per input) so the engine's eval_node ->
bind_params reads START_ID's sources from flow.wiring[START_ID] (top-level: the run-arg seed;
child: the call-arg edges). `run(record)` = coerce_inputs + declared-type check + apply_defaults,
returning the bound input record as ONE object keyed by input name -> store[<start id>], the
`inputs`-namespace replacement (binding at the boundary, not the driver).

The flow's PRE `asserts:` are lowered onto START's ordinary `post_asserts` (each `${input.X}`
rewritten to `${output.X}`) and fire through the generic wrapper/`eval_node` seam against START's
COERCED output record — START is an ordinary scheduled node, not a bypass. This unifies top-level
and child (subflow) boundary asserts: both flow through the same post-assert seam. NOT START_ID's
job: system/clock/run_id. Hidden from authors: a reserved __-prefixed id, never parseable.
"""

from typing import Optional

from agent_composer.events import SourceSpan
from agent_composer.nodes.base import FieldRanges, Node, NodeKind, Output
from agent_composer.nodes.binding import ParamDecl
from agent_composer.typesys import TypeCheckError, build_value_as
from agent_composer.typesys.seeding import apply_defaults, coerce_inputs


class StartNode(Node):
    """
    The flow's parameter-binding boundary — internal-only, loader-synthesized, pure.

    One per flow (the single root). Its params are the input names; the engine reads each source
    from `flow.wiring[START_ID]` (the run-arg seed at top level, the call-arg edges in a child).
    `run` coerces the wired args to declared types, type-checks them, and fills declared defaults
    for omitted inputs, returning the bound record as one object keyed by input name.

    Args:
        node_id (`str`):
            The node's id (`ID` at top level; deep-namespaced in a child).
        input_decls (`list`, *optional*, defaults to `None`):
            The flow's `InputDecl`s; the params (and their defaults/required) derive from these.
        field_ranges (`dict`, *optional*, defaults to `None`):
            Render-only named boundary field ranges used to locate input declaration failures.
            Flow assertions use the separate positional range bridge. `None` for a spanless flow.
        title (`str`, *optional*, defaults to `None`):
            Display title.
    """

    kind = NodeKind.START

    #: Reserved id of the synthesized input boundary (the old `__start__` sentinel).
    #: The canonical source of truth; `compile.model` re-exports it as `START_ID`/`START_ID`.
    #: At top level the loader mints one StartNode with this id; a child flow's is
    #: deep-namespaced (`<callsite>/__start__`). Authors can never name a node this
    #: (parser reserved-id ban).
    ID = "__start__"

    def __init__(self, node_id: str, *, input_decls: Optional[list] = None,
                 field_ranges: Optional[dict] = None,
                 title: Optional[str] = None) -> None:
        super().__init__(node_id, title=title)
        self.input_decls = list(input_decls or [])
        # Render-only per-boundary-field source ranges: the flow's `input:` decls, keyed
        # `("input_decl", name)` like every other named site. The loader stamps these so `run` can
        # attach a precise `range` to a boundary type failure — the flow-boundary twin of a body
        # node's per-field ranges. Empty for a spanless / in-memory flow. (Flow-level `asserts:`
        # are NOT here: their text is not a unique identity, so they are located by position via
        # `pre_assert_ranges`/`post_assert_ranges`.) Rewrapped as `FieldRanges` so the bare-key
        # guard survives a caller passing a plain dict.
        if field_ranges:
            self.field_ranges = FieldRanges(field_ranges)
        # params = the input names PLUS each input's default/required: eval_node->bind_params reads
        # each source from flow.wiring[START_ID][name] (the seed at top level; the call-arg edges in
        # a child), and for an OMITTED input fills the declared default itself (or fails a required
        # one). This is what lets the child START_ID own omitted-input defaulting — without
        # the REF/MAP driver pre-defaulting (a naive driver-drop otherwise loses the default, since a
        # bare param binds an absent input as present-None which run()'s apply_defaults can't fill).
        # `type`/`type_str` are intentionally NOT carried: run() is the sole type authority, so
        # bind_params must not type-check first with its own BindingError message.
        self.params = [
            ParamDecl(name=d.name, required=d.required, default=d.default)
            for d in self.input_decls
        ]

    def _run_impl(self, inputs: dict) -> Output:
        # The seeding.py pipeline lifted onto the node: coerce the wired args to the
        # declared types, type-check each, then fill each declared default for an OMITTED
        # input. Returns the bound record as ONE object keyed by input name. The flow's
        # boundary `asserts:` are NOT fired here — they ride START's ordinary `post_asserts`
        # (lowered to read `${output.X}`) and fire through the generic wrapper/eval_node seam
        # against START's coerced output, so START is an ordinary scheduled node.
        coerced = coerce_inputs(self.input_decls, inputs)
        # Enforce each value against its declared type, raising the byte-stable located
        # message. A raise -> NodeFailed at the engine boundary.
        for decl in self.input_decls:
            if decl.type is None or decl.name not in coerced or coerced[decl.name] is None:
                continue
            try:
                build_value_as(decl.type, coerced[decl.name])
            except TypeCheckError as exc:
                # Surface the precise reason (`<value> does not match declared type
                # <scalar>`) and name which input it was; the bare wrapper's
                # "expected Optional, got str" lost the scalar that actually failed.
                # Carry an `input_decl` locator so the CLI can box the failing input's
                # declaration line (node=None: it's the flow boundary, not a body node).
                new = TypeCheckError(f"input `{decl.name}` — {exc}")
                new.locator = SourceSpan(node=None, kind="input_decl", key=decl.name)
                new.range = self.field_ranges.get(("input_decl", decl.name))
                raise new from exc
        final = apply_defaults(self.input_decls, coerced)
        return Output(value=final)
