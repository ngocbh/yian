"""The base descriptor contract — the parse-time analogue of `nodes/base.py`.

A `compose/kinds/<kind>.py` module owns a frozen *descriptor*: the validated, located
shape a `kind:` node body parses into (before it is built into a runtime `Node`). Every one
of those per-kind descriptors inherits `NodeDescriptor`, which holds the fields common to all
nodes so no kind redeclares them — exactly as every runtime node subclasses `nodes.base.Node`.

Beyond the common fields it also carries the two node-signature surfaces every value-shaped
kind shares — `inputs` (the call-arg bindings) and `outputs` (the declared output shape) — plus
the build-time derivations over them (`output_type`/`sink_params`/`sink_wiring`), the descriptor
analogue of the runtime `Node`'s `output_type`/`params`. A kind that has no signature (wait/case)
simply leaves them at their empty defaults, and a kind that derives its output from a child flow
(call/map/loop) ignores `outputs`. The fields live on the base but are NOT auto-allowed as
authoring keys — each kind still lists `inputs`/`outputs` in its own ALLOWED set (they are
subtracted from `NodeDescriptor.common_field_names()`), so wait/case/call/loop/map keep
rejecting them.

Like `nodes/base.py` (which imports `typesys`/`binding` for the `Node` contract), this base
imports the deps its methods need — `typesys` (Type/TypeRegistry), `nodes.binding` (ParamDecl),
`compose.types` (read_type), `events` (SourceRange) — none of which import back into `kinds`, so
it stays a leaf of the kinds package: it never imports `helpers.py`, any sibling kind,
`kinds/__init__`, or `parser.py`. Both `helpers.py` (which derives `_COMMON_FIELDS` from this
base) and every kind module import it at module top level without cycling; `parser.py`
re-exports it as the descriptor base type for its importers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional

from agent_composer.nodes.binding import ParamDecl
from agent_composer.typesys.values import Type
from agent_composer.typesys.types import TypeRegistry
from agent_composer.compose.types import read_type
from agent_composer.events import SourceRange


@dataclass(frozen=True, kw_only=True)
class NodeDescriptor:
    """The base every `compose/kinds/<kind>.py` descriptor inherits — the fields common to
    all nodes, so no kind redeclares them.

    Three field groups:
    - **Author-surface commons** (`id` + `node_name`/`depends_on`/`runs_after`/`env`): `id` is
      the node's keyed-map key (no default); `node_name` the display title; `depends_on`/
      `runs_after` are RUN-ORDERING edges (no data flows — both gate on the source settling,
      but `depends_on` co-skips the dependent if the source skipped while `runs_after` does not);
      `env` is static per-node config, merged under the flow `env:` at build.
    - **Node-signature surface** (`inputs`/`outputs`): `inputs` is the call-arg binding map
      (`param -> ${source}`); `outputs` is the declared output shape (a `read_type`-able spec, or
      `None`). Both default empty, so a signature-less kind (wait/case) or a child-derived one
      (call/map/loop) simply never populates them. They are NOT common authoring keys — each kind
      opts in via its own ALLOWED set (see the module docstring).
    - **Located-parse metadata** (`range`/`field_ranges`): the node body's `SourceRange` and each
      present field's `SourceRange`, populated by each kind's `parse`. `compare=False` keeps them
      out of `==`, so a descriptor still equals an identically-configured one built without ranges.

    `kw_only=True` (inherited by every subclass) makes all fields keyword-only, so a subclass may
    declare REQUIRED kind fields (`code`, `model_id`, `call`, ...) after these defaulted commons
    without the "non-default follows default" error. Every construction site is keyword-only."""

    id: str
    node_name: Optional[str] = None
    depends_on: list[str] = field(default_factory=list)
    runs_after: list[str] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: Any = None
    range: Optional[SourceRange] = field(default=None, compare=False)
    field_ranges: dict[str, SourceRange] = field(default_factory=dict, compare=False)

    def output_type(self, registry: TypeRegistry) -> Optional[Type]:
        """This descriptor's declared `outputs:` as one `Type`; `None` when absent."""
        if self.outputs is None:
            return None
        return read_type(self.outputs, registry)

    def sink_params(self) -> list[ParamDecl]:
        """This descriptor's `inputs:` map -> untyped `ParamDecl`s (the node-side signature).

        The split: `params` carries only the declared NAMES (no source); the flow owns the
        sources in `CompiledFlow.wiring`. Untyped (`type_str`/`type` None) — the source carries
        the type; derived from the SAME map as `sink_wiring` so names and keys are in lockstep."""
        return [ParamDecl(name=k) for k in (self.inputs or {})]

    def sink_wiring(self) -> dict[str, Any]:
        """This descriptor's `inputs:` map -> the flow-owned wiring (`param -> source`).
        The source half of the split; the same dict `sink_params` reads the names from."""
        return dict(self.inputs or {})
