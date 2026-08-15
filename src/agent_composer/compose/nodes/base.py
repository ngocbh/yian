"""The base descriptor contract — the parse-time analogue of `nodes/base.py`.

A `compose/nodes/<kind>.py` module owns a frozen *descriptor*: the validated, located
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

Like `nodes/base.py`, this base imports only the dependencies its contract needs —
`typesys` (Type/TypeRegistry), `nodes.contracts` (Parameter),
`compose.helpers.types` (read_type), `compose.helpers.errors` (LoadError) and
`compose.source` (SourceCursor) for parse navigation, `events` (SourceRange) — all
earlier layers on the ladder, so it stays a leaf of the nodes package: it never imports any
sibling kind or `nodes/__init__`. Every kind module imports it at module top level without
cycling; `parse/file.py` re-exports it as the descriptor base type for its importers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

from agent_composer.nodes.base import FieldRanges, Node
from agent_composer.nodes.contracts import Parameter
from agent_composer.typesys.values import Type
from agent_composer.typesys.types import TypeRegistry
from agent_composer.compile.model import Edge
from agent_composer.compose.helpers.types import read_type
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.source import SourceCursor
from agent_composer.events import SourceRange


@dataclass(frozen=True)
class BuildContext:
    """The immutable inputs a node `.build()` reads — the read-only build environment."""

    registry: Any                          # TypeRegistry (flow's read_typedefs map)
    resolver: Any = None                   # ChildResolver | None (None for leaves)
    source_file: str = "<inline-code>"     # flow file label — frames CODE tracebacks
    code_line: int | None = None           # inline CODE content-start override
    code_field_line: int | None = None     # inline CODE field-key override


@dataclass(frozen=True)
class BuildResult:
    """The immutable product a node `.build()` returns — the built node + its wiring/edges."""

    node: Node
    wiring: dict[str, Any]                  # flow-owned {param -> source}
    edges: tuple[Edge, ...] = ()            # () for leaf/call/loop; case returns its own


@dataclass(frozen=True, kw_only=True)
class NodeDescriptor:
    """The base every `compose/nodes/<kind>.py` descriptor inherits — the fields common to
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

    # --- Class-level parse/build contract (per-kind overrides, not dataclass fields) ---
    #
    # These are `ClassVar`s (lowercase names, `ClassVar[...]`), so the dataclass machinery
    # excludes them from `__init__`/`__eq__`; each kind subclass overrides them to declare
    # only what makes that kind different.

    # The kind's surface discriminator (`kind:` in the body). Empty on the abstract base;
    # every concrete kind sets it (e.g. `"agent"`, `"code"`, `"map"`).
    kind: ClassVar[str] = ""

    # The COMPLETE set of legal body keys for this kind (surface spelling). Starts as the
    # keys every node body accepts — the four author-surface commons plus `kind` itself —
    # so `_check_keys` compares against `cls.allowed` directly (no manual COMMON union).
    # Each kind EXTENDS it via `allow(...)` with its own keys; `inputs`/`outputs` are opt-in
    # (a signature-less kind like wait/case never lists them, so it keeps rejecting them).
    allowed: ClassVar[frozenset[str]] = frozenset(
        {"kind", "node_name", "depends_on", "runs_after", "env"}
    )

    # The body keys this kind REQUIRES (surface spelling). Empty on the base — no key is
    # required by every kind (`kind` is resolved by dispatch before `parse`, never checked
    # here). Each kind extends it via `require(...)` (e.g. `code` requires `code`).
    required: ClassVar[frozenset[str]] = frozenset()

    # The descriptor FIELD names carrying `${...}` binding strings, consumed by the
    # kind-blind ref walk in `bindings.py`. Empty on the base; a value-shaped kind sets it
    # (e.g. `("inputs",)`).
    binding_fields: ClassVar[tuple[str, ...]] = ()

    # Surface key -> descriptor field-name overrides the template `parse` applies. Empty on the
    # base: the remaps are kind-specific, not common, so the kind that owns the collision declares
    # it (`case` maps `else:`->`else_`; `loop` maps `while:`/`until:`->`while_`/`until_`). Keeping
    # it empty here is what lets `wait` author a plain `until:` field without a spurious `until_`
    # rename — the collision the base cannot know about until a kind opts in.
    keyword_remap: ClassVar[dict[str, str]] = {}

    id: str
    node_name: Optional[str] = None
    depends_on: list[str] = field(default_factory=list)
    runs_after: list[str] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: Any = None
    range: Optional[SourceRange] = field(default=None, compare=False)
    # `(kind, key) -> SourceRange` for each authored NAMED site, guarded against a bare-key
    # lookup (see `FieldRanges`). Stamped here at parse; copied onto the runtime node by `_finish`.
    field_ranges: FieldRanges = field(default_factory=FieldRanges, compare=False)
    # Positional `asserts:` ranges — one entry per authored assert IN ORDER, aligned 1:1 with the
    # `asserts` list (`None` where no range exists). Stamped in `parse/nodes` via the cursor's
    # `item_range_list`; split into the built node's `pre_assert_ranges`/`post_assert_ranges` (in
    # lockstep with the pre/post classification) by `validate_node_asserts`. Kept OUT of
    # `field_ranges` because an assert's text is not a unique key (two identical asserts must still
    # get distinct carets). `compare=False`, like the other render metadata. `[]` when none authored.
    assert_ranges: list[SourceRange | None] = field(default_factory=list, compare=False)

    def output_type(self, registry: TypeRegistry) -> Optional[Type]:
        """This descriptor's declared `outputs:` as one `Type`; `None` when absent."""
        if self.outputs is None:
            return None
        return read_type(self.outputs, registry)

    def sink_params(self) -> list[Parameter]:
        """This descriptor's `inputs:` map -> untyped Parameters (the node-side signature).

        The split: `params` carries only the declared NAMES (no source); the flow owns the
        sources in `CompiledFlow.wiring`. These parameters have `contract=None`; they are
        derived from the SAME map as `sink_wiring` so names and keys stay in lockstep."""
        return [Parameter(name=k) for k in (self.inputs or {})]

    def sink_wiring(self) -> dict[str, Any]:
        """This descriptor's `inputs:` map -> the flow-owned wiring (`param -> source`).
        The source half of the split; the same dict `sink_params` reads the names from."""
        return dict(self.inputs or {})

    # --- Key-set extension helpers ------------------------------------------------------

    @classmethod
    def allow(cls, *keys: str) -> frozenset[str]:
        """`cls.allowed` unioned with `keys` — the idiom a kind uses to add its OWN body keys.

        Unions THIS class's allowed set (already carrying the commons + `kind`, plus any keys
        added by ancestors) with the extra surface keys, so a subclass inherits its parent's
        keys and lists only its additions (e.g. `CallDescriptor.allow("over", "parallel")`)."""
        return cls.allowed | frozenset(keys)

    @classmethod
    def require(cls, *keys: str) -> frozenset[str]:
        """`cls.required` unioned with `keys` — the idiom a kind uses to add REQUIRED body keys.

        Mirror of `allow` for the required set: unions this class's required keys with the
        extras, so a subclass extends its parent's requirements (e.g. `MapDescriptor` requires
        `call` from `CallDescriptor` plus its own `over`)."""
        return cls.required | frozenset(keys)

    # --- Parse-side machinery (shared key-legality + common-field reads) ------------------

    @classmethod
    def _check_keys(cls, node_id: str, cursor: SourceCursor) -> None:
        """Legality check for a node body's surface keys, run by the template `parse`.

        Rejects any body key not in `cls.allowed` (the COMPLETE legal key set for this kind —
        the commons + `kind` + the kind's own keys, so no manual union is needed), then requires
        every key in `cls.required`. Raises a located `LoadError` for the first offense — an
        unknown key caret-points at that key's value, a missing key at the node body. Reads
        nothing into the descriptor (that is `_read_common_fields` + the template's own-key loop)."""
        allowed_keys = cls.allowed
        body = cursor._value_view()
        for key in body:
            if key not in allowed_keys:
                raise LoadError(
                    f"node {node_id!r} (kind={cls.kind}): field {key!r} is not allowed "
                    f"(allowed: {', '.join(sorted(allowed_keys))})",
                    range=cursor.range_of(key),
                )
        for key in cls.required:
            if key not in body:
                raise LoadError(
                    f"node {node_id!r} (kind={cls.kind}): missing required field {key!r}",
                    range=cursor.range,
                )

    @classmethod
    def _read_common_fields(
        cls,
        node_id: str,
        cursor: SourceCursor,
        kwargs: dict[str, Any],
        ranges: dict[tuple[str, str], SourceRange],
    ) -> None:
        """Lift the four common author-surface fields into the template `parse` accumulators.

        Reads `node_name`/`depends_on`/`runs_after`/`env` from the cursor's read-only body
        into `kwargs`, recording each present field's `SourceRange` into `ranges`; both dicts
        are mutated IN PLACE (the accumulators the template passes to the descriptor
        constructor). `env:` is validated to be a mapping with string keys — a bad shape raises
        a located `LoadError` naming `cls.kind`. The other three are copied through untouched
        (their per-field shape is checked later by the loader edge pass)."""
        body = cursor._value_view()
        for key in ("node_name", "depends_on", "runs_after"):
            if key in body:
                kwargs[key] = cursor.get(key)
                rng = cursor.range_of(key)
                if rng is not None:
                    ranges[("field", key)] = rng
        if "env" in body:
            env = body["env"]
            if not isinstance(env, Mapping) or not all(
                isinstance(k, str) for k in env
            ):
                raise LoadError(
                    f"node {node_id!r} (kind={cls.kind}): `env:` must be a mapping "
                    f"with string keys",
                    range=cursor.range_of("env"),
                )
            kwargs["env"] = cursor.mapping("env")
            rng = cursor.range_of("env")
            if rng is not None:
                ranges[("field", "env")] = rng

    @classmethod
    def parse(
        cls,
        node_id: str,
        cursor: SourceCursor,
    ) -> "NodeDescriptor":
        """The template `parse` — the whole parse skeleton every kind shares.

        Runs, in order: `_normalize_cursor` (pure pre-check projection), `_check_keys` (key
        legality), the OWN-KEY loop (`cls.allowed - NodeDescriptor.allowed` yields exactly
        this kind's own keys — the commons + `kind` live on the base's set, so the loop never
        touches them; each present key is read into `kwargs` under its `keyword_remap`ped name,
        its `SourceRange` recorded), `_validate_fields` (kind-specific legality hook, default:
        the asserts-list check), then `_read_common_fields` (the four commons with `env`
        validation). Finally constructs `cls(**kwargs, range=..., field_ranges=...)`.

        A kind customizes only the two hooks; 8 of 10 kinds inherit this untouched. Returns
        the fully-populated descriptor instance (`cls`, never the abstract base)."""
        cursor = cls._normalize_cursor(cursor)
        cls._check_keys(node_id, cursor)
        body = cursor._value_view()
        own_range = cursor.range
        ranges: FieldRanges = FieldRanges()
        kwargs: dict[str, Any] = {"id": node_id}
        for key in cls.allowed - NodeDescriptor.allowed:  # this kind's OWN keys only
            if key not in body:
                continue
            rng = cursor.range_of(key)
            if rng is not None:
                ranges[("field", key)] = rng
            kwargs[cls.keyword_remap.get(key, key)] = cursor.get(key)
        cls._validate_fields(node_id, kwargs, ranges, own_range)
        cls._read_common_fields(node_id, cursor, kwargs, ranges)
        return cls(**kwargs, range=own_range, field_ranges=ranges)

    @classmethod
    def _normalize_cursor(cls, cursor: SourceCursor) -> SourceCursor:
        """Return the cursor view inspected by `_check_keys`; default unchanged.

        A hook the template `parse` calls first, before any key is inspected. Only `case`
        overrides it to project YAML 1.1's `on:` boolean key back to its schema name.
        """

        return cursor

    @classmethod
    def _validate_fields(
        cls,
        node_id: str,
        kwargs: dict[str, Any],
        ranges: dict[tuple[str, str], SourceRange],
        own_range: Optional[SourceRange],
    ) -> None:
        """Kind-specific field legality, run by the template `parse` after the own-key loop.

        Receives `kwargs` (already remapped/populated by the loop), not the raw body, plus
        `own_range` (the node body's range, for anchoring an error about an ABSENT field, which
        has no per-field range). Default: if an `asserts:` field is present, it must be a `list`
        of `str` — a bad shape raises a located `LoadError` naming `cls.kind`. Presence-guarded:
        absent `asserts` is a no-op, so kinds that never author `asserts` are unaffected. Only
        `human_input` overrides this (running its one-of-three check, then `super()` for asserts)."""
        asserts = kwargs.get("asserts")
        if asserts is not None and not (
            isinstance(asserts, list) and all(isinstance(a, str) for a in asserts)
        ):
            raise LoadError(
                f"node {node_id!r} (kind={cls.kind}): `asserts:` must be a list of strings",
                range=ranges.get(("field", "asserts")),
            )

    # --- Build-side machinery -----------------------------------------------------------

    def build(self, ctx: BuildContext) -> BuildResult:
        """Build this descriptor into a runtime `Node` (+ its flow-owned wiring), returning a
        `BuildResult`.

        Abstract on the base — each concrete kind overrides it, returning a `BuildResult`
        (typically via `self._finish(node, wiring)`; `case` also passes `edges=`). Its one input
        is the immutable `BuildContext`: `ctx.registry` (the `TypeRegistry` for `output_type`),
        `ctx.resolver` (the child-flow resolver for call/map/loop), and the inline-CODE traceback
        anchors `ctx.source_file`/`ctx.code_line`/`ctx.code_field_line` (only the CODE build reads
        them). A leaf kind reads only `ctx.registry`; call/map/loop also read `ctx.resolver`."""
        raise NotImplementedError(f"{type(self).__name__} has no build")

    def _finish(self, node, wiring, edges=()) -> BuildResult:
        """Stamp parse metadata onto a freshly-built `node` and wrap it in a `BuildResult`.

        The tail helper every `build` ends with — the ONE place the two range stamps live, so
        they cannot drift across the per-kind builds: copies this descriptor's `range` and a COPY
        of its `field_ranges` onto the runtime node (so diagnostics after build still caret back
        to the source), then returns the `BuildResult(node, wiring, edges)`. `edges` defaults to
        `()` (leaf/call/loop own no edges); `case` passes its data+control edges.

        The copy is a `FieldRanges`, not a plain `dict` — the runtime node is where the lookups
        actually happen, so copying into a bare dict would drop the bare-key guard precisely
        where it earns its keep."""
        node.range = self.range
        node.field_ranges = FieldRanges(self.field_ranges)
        return BuildResult(node=node, wiring=wiring, edges=tuple(edges))
