"""VariablePool — the runtime state primitive for a flow run.

Each node id has at most one stored value: an `Output` commits one typed `TypedValue`
(scalar, object, or list), while a CASE `Route` commits none. "Multiple outputs" are
fields of one object. `${input.X}`
resolves to `store[<start_id>].X` — the synthesized START_ID node's committed bound-input
record IS the flow's run-arguments object (the standalone `inputs` namespace is
retired). The one side namespace is `system` (host-injected ambient — run id / clock /
tenant; reserved, run-global). There is no engine-private key-addressed namespace;
live AGENT/HUMAN_INPUT/WAIT effects all return the leaf's ordinary Output.

Two responsibilities:
- `resolve(head, rest)` backs the `${...}` expression evaluator. Because values
  are typed objects (not stringified), `${x.output.ratio}` traverses into an
  object output; `${x.output}` is the node's whole value.
- `dumps()/loads()` serialize the whole pool losslessly (the discriminated
  `AnyValue` union round-trips each value's exact type), matching the pool wire
  representation recorded at run initialization and reconstructed during replay.
"""

from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

import yian.json_codec as json_codec
from yian.typesys.values import (
    AnyValue,
    TypedValue,
    ValueKind,
    Type,
    build_value,
    build_value_as,
)


class VariablePool(BaseModel):
    """
    The runtime variable store for one flow run.

    Each node id has at most one stored typed value (`store[node_id]`); a routing-only
    node can have none. "Multiple outputs" are fields of one object. `${input.X}` reads
    the synthesized START node's committed record; `${system.X}` reads host-injected
    ambients. Backs the `${...}` resolver and serializes losslessly for canonical
    run-record payloads and recovery.

    Attributes:
        store (`dict[str, AnyValue]`):
            Map of node id to that node's single produced value, as a discriminated
            `TypedValue`. Read via `${<id>.output[.path]}`.
        system (`dict[str, AnyValue]`):
            Host-injected ambient namespace (run id / clock / tenant); reserved and
            run-global. Read via `${system.<key>}`.
        start_id (`str`, *optional*, defaults to `"__start__"`):
            The enclosing flow's START node id, so `${input.X}` resolves to
            `store[start_id].X`. The engine sets this per flow; the default is the
            top-level convention for standalone pools.
    """

    model_config = ConfigDict(extra="forbid")

    # node_id -> the node's single produced value (${<id>.output[.path]})
    store: dict[str, AnyValue] = Field(default_factory=dict)
    # ${system.<key>}  — host-injected ambient (run id / clock / tenant); reserved
    system: dict[str, AnyValue] = Field(default_factory=dict)
    # The enclosing flow's START_ID node id: `${input.X}` ≡ `store[start_id].X`. The
    # engine sets this per-flow (and seeds store[start_id] with the bound input record); the
    # default below is the top-level convention used by standalone/no-engine pools. The literal
    # MIRRORS `StartNode.ID` but is NOT imported from it: `typesys` is a leaf below `nodes`, so it
    # cannot import the node class. A consistency test pins `VariablePool().start_id ==
    # StartNode.ID` so the two can never silently diverge.
    start_id: str = "__start__"

    # --- writes ------------------------------------------------------------- #

    def set(
        self,
        node_id: str,
        value: Any,
        declared: Optional[Union[ValueKind, Type]] = None,
    ) -> None:
        """Store a node's single produced value, type-checking it if a type is declared.

        Args:
            node_id (`str`):
                The producing node's id — the key the value is bound under.
            value (`Any`):
                The plain Python value the node returned.
            declared (`ValueKind | Type`, *optional*, defaults to `None`):
                The node's declared output type. When given, the value is validated
                against it (`build_value_as`); when `None`, the type is inferred
                (`build_value`).
        """
        self.store[node_id] = (
            build_value_as(declared, value)
            if declared is not None
            else build_value(value)
        )

    def add_system(self, key: str, value: Any) -> None:
        """Store a host-injected ambient under `key`, read via `${system.<key>}`."""
        self.system[key] = build_value(value)

    # --- reads -------------------------------------------------------------- #

    def get_value(self, node_id: str) -> Optional[TypedValue]:
        """Return a node's stored `TypedValue`, or `None` if it has not produced yet."""
        return self.store.get(node_id)

    def get(self, node_id: str, default: Any = None) -> Any:
        """Return a node's plain (unwrapped) value, or `default` if it is unset.

        Args:
            node_id (`str`):
                The producing node's id.
            default (`Any`, *optional*, defaults to `None`):
                Returned when `node_id` has no stored value.
        """
        seg = self.store.get(node_id)
        return seg.to_object() if seg is not None else default

    def resolve(self, head: str, rest: list[str]) -> Any:
        """
        Resolve a parsed `${head.rest...}` reference to a plain value.

        Recognizes three namespaces: `input.<key>[.<path>]` reads the START record,
        `<node>.output[.<path>]` reads a node's value (`.output` is a syntactic
        discriminator the resolver skips), and `system.<key>` reads a host ambient. An
        unrecognized head resolves to `None` (the missing-ref-is-falsy contract);
        legacy plural surfaces (`${outputs.X}`/`${inputs.X}`) are rejected at load time,
        so the load-time guard — not this method — is the authoritative defense.

        Args:
            head (`str`):
                The first reference segment: `input`, `system`, or a node id.
            rest (`list[str]`):
                The remaining dotted segments after the head (e.g. `["output", "ratio"]`).

        Returns:
            `Any`:
                The resolved plain value, or `None` if any step is absent.
        """
        # Node-first head — `${<node>.output[.path]}` reads `store[<node>]` interior.
        # The literal `output` token is a syntactic discriminator and is SKIPPED.
        if head in self.store and rest and rest[0] == "output":
            return self._traverse(self.store.get(head), rest[1:])
        # Singular input head — `${input.k}` ≡ `store[start_id].k`.
        if head == "input":
            if not rest:
                return None
            key, *path = rest
            return self._traverse(self.store.get(self.start_id), [key, *path])
        if head == "system":
            seg = self.system.get(rest[0]) if rest else None
            return seg.to_object() if seg is not None else None
        return None

    @staticmethod
    def _traverse(seg: Optional[AnyValue], path: list[str]) -> Any:
        """Object-walk a stored TypedValue along `path` (the `${<id>.output.a.b}` /
        `${input.x.y}` dotted read); a missing step resolves to None (propagates falsy)."""
        if seg is None:
            return None
        value = seg.to_object()
        for step in path:
            value = value.get(step) if isinstance(value, dict) else None
            if value is None:
                return None
        return value

    # --- serialization ------------------------------------------------------ #

    def dumps(self) -> str:
        """Serialize the whole pool with its exact typed-value discriminators."""
        return self.model_dump_json()

    @classmethod
    def loads(cls, blob: str) -> "VariablePool":
        """Reconstruct a pool from `dumps` output, decoding each value to its exact type."""
        return cls.model_validate(json_codec.loads(blob))
