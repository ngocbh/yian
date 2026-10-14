"""The base descriptor contract — the parse-time analogue of `nodes/base.py`.

A `compose/nodes/<kind>.py` module owns a frozen *descriptor*: the validated, located
shape a `kind:` node body parses into (before it is built into a runtime `Node`). Every one
of those per-kind descriptors inherits `NodeDescriptor`, which holds the fields common to all
nodes so no kind redeclares them — exactly as every runtime node subclasses `nodes.base.Node`.

Beyond the common fields it carries the normalized node-signature fields every value-shaped
kind shares: `inputs` stores authored `input:` bindings and `outputs` stores the authored
`output:` declaration. Build-time derivations over them (`output_contract`/`sink_params`/
`sink_wiring`) are the descriptor analogue of the runtime `Node`'s `output_contract`/`params`.
A signature-less kind leaves them empty, while a child-derived kind ignores `outputs`. Each
kind opts into these normalized keys through its own `allowed` set, so kinds without the
corresponding authored fields continue to reject them.

Like `nodes/base.py`, this base imports only the dependencies its contract needs —
`typesys` (TypeRegistry), `nodes.contracts` (OutputContract/Parameter/ValueContract),
`compose.helpers.types` (read_type), `compose.helpers.errors` (LoadError),
`compose.helpers.tags` (has_span), `compose.helpers.yaml_words` (yaml_read_as), `pydantic`
(ValidationError), `nodes.human_input.node` (site_label, a runtime leaf that imports no
`yian.compose` module), and `compose.source` (SourceCursor) for parse navigation,
`diagnostics` (SourceRange) — all earlier layers on the ladder, so it stays a leaf of the
nodes package: it never imports any sibling kind or `nodes/__init__`. Every kind module
imports it at module top level without cycling; `parse/file.py` re-exports it as the
descriptor base type for its importers.

It also owns the shared messages for a field whose value YAML read as the wrong type (text
or otherwise), so every kind reports that mistake in the same words.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional

from pydantic import ValidationError

from yian.nodes.base import Node
from yian.nodes.contracts import OutputContract, Parameter, ValueContract
from yian.nodes.human_input.node import site_label
from yian.typesys.types import TypeRegistry
from yian.typesys.values import value_repr
from yian.compile.model import Edge
from yian.compose.helpers.types import read_type
from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.helpers.tags import has_span
from yian.compose.helpers.yaml_words import yaml_read_as
from yian.compose.source import SourceCursor
from yian.diagnostics import DiagnosticFrame, OriginRef, SourceRange

if TYPE_CHECKING:
    from yian.compose.source.catalog import _SourceCatalogBuilder


def reject_container_spans(
    kind: str,
    node_id: str,
    key: str,
    value: Any,
    range: Optional[SourceRange],
    diagnostics: LoadDiagnostics,
) -> None:
    """Reject a list or mapping holding a `${...}` span in a field that binds whole.

    Such a field's container is a literal, so a span inside it would stay text.

    Args:
        kind (`str`): The node kind, for the message.
        node_id (`str`): The node id, for the message.
        key (`str`): The authored field name.
        value (`Any`): The field's authored value.
        range (`SourceRange | None`): Where the error points.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.

    Raises:
        `LoadError`: Raised when `value` is a list or mapping with a span inside it.
    """
    if isinstance(value, (list, dict)) and _holds_span(value):
        raise diagnostics.error(
            f"{kind} node {node_id!r} {key}: a value holding ${{...}} can't be a list or "
            "mapping; write the list as an expression, e.g. `${[a.output, b.output]}`",
            range=range,
        )


def value_type_error(field: str, needed: str, value: Any, fix: str) -> str:
    """Return the load error message for a field whose value YAML read as the wrong type.

    Args:
        field (`str`):
            The field's path as authored, such as `over` or `questions[0].header`.
        needed (`str`):
            What the field takes, with its article, such as `text` or `a list`.
        value (`Any`):
            The field's value as YAML read it; `None` is reported as empty.
        fix (`str`):
            What the author should write instead.

    Returns:
        `str`:
            The message, naming what YAML read the value as and the fix.

    Example:
        `value_type_error("over", "a list", 5, fix="write a list")` is
        ``"`over:` must be a list, but YAML read it as an integer; write a list"``;
        `value_type_error("over", "a list", None, fix="write a list")` is
        ``"`over:` must be a list, but it is empty; write a list"``.
    """
    if value is None:
        return f"`{field}:` must be {needed}, but it is empty; {fix}"
    return f"`{field}:` must be {needed}, but YAML read it as {yaml_read_as(value)}; {fix}"


def not_text_error(field: str, value: Any, fix: str | None = None) -> str:
    """Return the load error message for a text field whose value YAML read as non-text.

    Args:
        field (`str`):
            The field's path as authored, such as `prompt` or `questions[0].header`.
        value (`Any`):
            The field's value as YAML read it.
        fix (`str`, *optional*, defaults to `None`):
            What the author should write instead. `None` uses "write the text" for an empty
            value and "put it in quotes" otherwise.

    Returns:
        `str`:
            The message, naming what YAML read the value as and the fix.

    Example:
        `not_text_error("prompt", 5)` is
        ``"`prompt:` must be text, but YAML read it as an integer; put it in quotes"``;
        `not_text_error("questions[0].header", None)` is
        ``"`questions[0].header:` must be text, but it is empty; write the text"``;
        `not_text_error("kind", [1], fix="write one of a, b")` is
        ``"`kind:` must be text, but YAML read it as a list; write one of a, b"``.
    """
    if fix is None:
        fix = "write the text" if value is None else "put it in quotes"
    return value_type_error(field, "text", value, fix)


def not_text_detail(exc: ValidationError) -> tuple[str, Any] | None:
    """Return the field and value of a pydantic error that is a text field read as non-text.

    Only the first error is read, so the field named agrees with the range the caller locates
    from that same first error.

    Args:
        exc (`ValidationError`):
            The pydantic validation error.

    Returns:
        `tuple[str, Any] | None`:
            The field's path as authored (such as `uses.lib` or `asserts[1]`) and its value,
            for `not_text_error`; `None` when the first error is not a `string_type` failure,
            or is the `str` member of a union field, which takes more than text.
    """
    errors = exc.errors()
    first = errors[0]
    if first["type"] != "string_type":
        return None
    loc = tuple(first["loc"])
    if loc[-1] == "str":
        # A union reports one error per member, the `str` member first; any other error under
        # the same parent (a list member's item error extends it) marks a field that takes more than text.
        parent = loc[:-1]
        if any(
            error is not first and tuple(error["loc"][: len(parent)]) == parent
            for error in errors
        ):
            return None
    return site_label(loc), first["input"]


def reject_non_text(
    kind: str,
    node_id: str,
    key: str,
    value: Any,
    range: Optional[SourceRange],
    diagnostics: LoadDiagnostics,
    *,
    required: bool = False,
    fix: str | None = None,
) -> None:
    """Reject a text or expression field whose value YAML read as non-text.

    Args:
        kind (`str`): The node kind, for the message.
        node_id (`str`): The node id, for the message.
        key (`str`): The authored field name.
        value (`Any`): The field's authored value.
        range (`SourceRange | None`): Where the error points.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.
        required (`bool`, *optional*, defaults to `False`):
            Whether the field needs a value. `False` lets `None` through as an absent field;
            `True` rejects it as empty.
        fix (`str`, *optional*, defaults to `None`):
            What the author should write instead; `None` uses `not_text_error`'s default.

    Raises:
        `LoadError`: Raised when `value` is not a `str` and is not `None`, or is `None` while
            `required` is set.
    """
    if not isinstance(value, str) and (value is not None or required):
        raise diagnostics.error(
            f"node {node_id!r} (kind={kind}): " + not_text_error(key, value, fix),
            range=range,
        )


def _holds_span(value: Any) -> bool:
    """Whether any string inside `value`, or `value` itself, holds a `${...}` span."""
    if isinstance(value, str):
        return has_span(value)
    if isinstance(value, dict):
        return any(_holds_span(item) for item in value.values())
    if isinstance(value, list):
        return any(_holds_span(item) for item in value)
    return False


@dataclass(frozen=True)
class BuildContext:
    """The immutable inputs a node `.build()` reads.

    ``resolver`` resolves child flows for composites. ``output_contract_reader``
    optionally compiles one authored node-output declaration at its established
    build position; ``variable_declaration_reader`` does the same for local declarations.
    ``resolve_origin`` presents an origin registered before file finalization.
    ``binding_contract_reader`` resolves one parent binding's output contract from
    the flow inputs and the producers it references (args: diagnostic label,
    binding, binding range); LOOP reads it to type a carried field the body omits.
    Source-free and direct programmatic builds leave these optional seams unset.
    ``owner`` is the lexical flow path of the flow being assembled; AGENT
    descriptors stamp it into memory declaration sites.
    """

    registry: TypeRegistry  # flow's read_typedefs map
    resolver: Any = None  # ChildResolver | None (None for leaves)
    source_file: str = "<inline-code>"  # opaque filename for Python tracebacks
    resolve_origin: Callable[[OriginRef], DiagnosticFrame | None] | None = None
    diagnostics: LoadDiagnostics = field(default_factory=LoadDiagnostics)
    output_contract_reader: Callable[[str, Any], ValueContract | None] | None = None
    variable_declaration_reader: Callable[[str, Any], list[Parameter] | None] | None = None
    binding_contract_reader: (
        Callable[[str, Any, SourceRange | None], OutputContract | None] | None
    ) = None
    owner: tuple[str, ...] = ("root",)  # lexical flow path that stamps DeclSite.owner


@dataclass(frozen=True)
class BuildResult:
    """A built node plus transient Compose-owned assembly metadata.

    ``node_range``, ``field_ranges``, and ``field_labels`` exist only while Compose
    validates the assembled graph. They are discarded before ``CompiledFlow`` construction.
    ``field_labels`` maps a site label (a field name or an input's ``input 'x'`` label) to
    the full author-facing phrase that names the site in diagnostics.
    """

    node: Node
    wiring: dict[str, Any]  # flow-owned {param -> source}
    edges: tuple[Edge, ...] = ()  # () for leaf/call/loop; case returns its own
    node_range: SourceRange | None = field(default=None, compare=False)
    field_ranges: dict[tuple[str, str], SourceRange] = field(
        default_factory=dict,
        compare=False,
    )
    field_labels: dict[str, str] = field(default_factory=dict, compare=False)

    def site_name(self, node_id: str, site: str) -> str:
        """Return the author-facing name of one site of this node in a diagnostic.

        Args:
            node_id (`str`):
                The id this node is built under.
            site (`str`):
                A site label: a field name (`prompt`) or an input's wiring label
                (`input 'x'`).

        Returns:
            `str`:
                The site's `field_labels` phrase when a desugar set one, else
                `node '<node_id>' <site>`.
        """
        return self.field_labels.get(site) or f"node {node_id!r} {site}"


@dataclass(frozen=True, kw_only=True)
class NodeDescriptor:
    """The base every `compose/nodes/<kind>.py` descriptor inherits — the fields common to
    all nodes, so no kind redeclares them.

    Six field groups:
    - **Author-surface commons** (`id` + `node_name`/`depends_on`/`runs_after`/`env`): `id` is
      the node's keyed-map key (no default); `node_name` the display title; `depends_on`/
      `runs_after` are RUN-ORDERING edges (no data flows — both gate on the source settling,
      but `depends_on` co-skips the dependent if the source skipped while `runs_after` does not);
      `env` is static per-node config, merged under the flow `env:` at build.
    - **Normalized node-signature carriers** (`inputs`/`outputs`): `inputs` stores the authored
      `input:` binding map (`param -> ${source}`); `outputs` stores the authored `output:` shape
      (a `read_type`-able spec, or `None`). Both default empty, and each kind opts into them via
      its own `allowed` set (see the module docstring).
    - **Declared input types** (`input_types`): `param -> ValueContract` for each typed
      `input:` binding (`qty: int = ${size.output.qty}`). Compose fills it at load when it
      splits the type off the binding value; it is not a YAML key, so `parse` never reads it.
    - **Lift host** (`host`): the id of the node a desugar pass lifted this synth node out of
      (the node whose binding held an inline `call(...)`, an adaptive gate, the outer call of a
      nested branch call), or `None`. It is not a YAML key. The host-edge step gives a lifted
      node its host's CASE routes, `depends_on`, and `runs_after`, and gates it on the host's
      other inputs.
    - **Field labels** (`field_labels`): site label (a field name or an input's `input 'x'`
      label) -> the full author-facing phrase that names the site in diagnostics, owner
      included (`"node 'ask' adaptive_questions.prompt"`). A desugar pass sets it on a synth
      node whose field text or input binding was authored under another node. It is not a
      YAML key, and `compare=False` keeps it out of `==`.
    - **Located-parse metadata** (`origin`/`range`/`field_ranges`): the opaque authored node
      identity, node body's `SourceRange`, and each present field's `SourceRange`. `compare=False`
      keeps them out of `==`, so a descriptor still equals an identically-configured one built
      without source metadata.

    `kw_only=True` (inherited by every subclass) makes all fields keyword-only, so a subclass may
    declare REQUIRED kind fields (`code`, `tool_id`, `call`, ...) after these defaulted commons
    without the "non-default follows default" error. Every construction site is keyword-only.
    """

    # --- Class-level parse/build contract (per-kind overrides, not dataclass fields) ---
    #
    # These are `ClassVar`s (lowercase names, `ClassVar[...]`), so the dataclass machinery
    # excludes them from `__init__`/`__eq__`; each kind subclass overrides them to declare
    # only what makes that kind different.

    # The kind's surface discriminator (`kind:` in the body). Empty on the abstract base;
    # every concrete kind sets it (e.g. `"agent"`, `"code"`, `"map"`).
    kind: ClassVar[str] = ""

    # The complete set of normalized body keys for this kind. Starts as the
    # keys every node body accepts — the four author-surface commons plus `kind` itself —
    # so `_check_keys` compares against `cls.allowed` directly (no manual COMMON union).
    # Each kind extends it via `allow(...)` with its own keys; `inputs`/`outputs` are the
    # normalized carriers for authored `input:`/`output:` and remain opt-in
    # (a signature-less kind like wait never lists them, so it keeps rejecting them).
    allowed: ClassVar[frozenset[str]] = frozenset(
        {"kind", "node_name", "depends_on", "runs_after", "env"}
    )

    # The normalized body keys this kind requires. Empty on the base — no key is
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
    input_types: dict[str, ValueContract] = field(default_factory=dict)
    # Set only by desugar on a synth node, so the lifted node runs only when its host can;
    # part of `==` because it changes the compiled graph.
    host: Optional[str] = None
    field_labels: dict[str, str] = field(default_factory=dict, compare=False)
    origin: OriginRef | None = field(default=None, compare=False)
    range: Optional[SourceRange] = field(default=None, compare=False)
    # `(kind, key) -> SourceRange` for each authored named site. Stamped here at parse and
    # copied into the transient Compose-owned `BuildResult` by `_finish`.
    field_ranges: dict[tuple[str, str], SourceRange] = field(
        default_factory=dict,
        compare=False,
    )
    # Positional `asserts:` ranges — one entry per authored assert IN ORDER, aligned 1:1 with the
    # `asserts` list (`None` where no range exists). Stamped in `parse/nodes` via the cursor's
    # `item_range_list` and consumed only by Compose's node-assert validator. Kept OUT of
    # `field_ranges` because an assert's text is not a unique key (two identical asserts must still
    # get distinct carets). `compare=False`, like the other render metadata. `[]` when none authored.
    assert_ranges: list[SourceRange | None] = field(default_factory=list, compare=False)

    def __post_init__(self) -> None:
        """Reject non-opaque provenance before a descriptor reaches build."""
        if self.origin is not None and not isinstance(self.origin, OriginRef):
            raise TypeError("node descriptor origin must be an OriginRef or None")

    def output_contract(
        self,
        ctx: BuildContext,
    ) -> OutputContract | None:
        """Return the declared ``output:`` as one canonical contract.

        The optional Compose reader runs at this existing build position so
        source compilation cannot move ahead of kind-owned validation. Direct
        descriptor builds omit it and retain the source-free type reader. A
        subclass whose output is inferred rather than declared may return a
        partial record.

        Args:
            ctx (`BuildContext`):
                Registry, diagnostics, and optional authored-contract reader.

        Returns:
            `OutputContract | None`:
                Canonical declared contract (an exact ``ValueContract`` here), or
                ``None`` when no output exists.

        Raises:
            `LoadError`:
                Raised when the declaration is not a valid type expression.
            `TypeError`:
                Raised when the injected reader violates its return contract.
        """
        if self.outputs is None:
            return None
        diagnostics = ctx.diagnostics
        try:
            if ctx.output_contract_reader is not None:
                contract = ctx.output_contract_reader(self.id, self.outputs)
                if contract is not None:
                    if type(contract) is not ValueContract:
                        raise TypeError(
                            "output contract reader must return an exact "
                            "ValueContract or None"
                        )
                    return contract
            return ValueContract(
                read_type(self.outputs, ctx.registry, diagnostics=diagnostics)
            )
        except LoadError as exc:
            if exc.diagnostic.frames:
                raise
            raise diagnostics.prepend(
                exc,
                range=self.field_ranges.get(("field", "outputs")) or self.range,
            ) from exc

    def sink_params(self) -> list[Parameter]:
        """This descriptor's authored `input:` map -> untyped Parameters (the node-side signature).

        The split: `params` carries only the declared NAMES (no source); the flow owns the
        sources in `CompiledFlow.wiring`. These parameters have `contract=None`; they are
        derived from the SAME map as `sink_wiring` so names and keys stay in lockstep.
        """
        return [Parameter(name=k) for k in (self.inputs or {})]

    def sink_wiring(self) -> dict[str, Any]:
        """This descriptor's authored `input:` map -> the flow-owned wiring (`param -> source`).
        The source half of the split; the same dict `sink_params` reads the names from.
        """
        return dict(self.inputs or {})

    # --- Key-set extension helpers ------------------------------------------------------

    @classmethod
    def allow(cls, *keys: str) -> frozenset[str]:
        """`cls.allowed` unioned with `keys` — the idiom a kind uses to add its OWN body keys.

        Unions THIS class's allowed set (already carrying the commons + `kind`, plus any keys
        added by ancestors) with the extra surface keys, so a subclass inherits its parent's
        keys and lists only its additions (e.g. `CallDescriptor.allow("over", "parallel")`).
        """
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
        """Check a node body's normalized keys, run by the template `parse`.

        Rejects any body key not in `cls.allowed` (the complete legal key set for this kind —
        the commons + `kind` + the kind's own keys, so no manual union is needed), then requires
        every key in `cls.required`. Raises a located `LoadError` for the first offense — an
        unknown key caret-points at that key's value, a missing key at the node body. Reads
        nothing into the descriptor (that is `_read_common_fields` + the template's own-key loop).
        """
        diagnostics = LoadDiagnostics.from_cursor(cursor)
        allowed_keys = cls.allowed
        body = cursor._value_view()
        for key in body:
            if key not in allowed_keys:
                raise diagnostics.error(
                    f"node {node_id!r} (kind={cls.kind}): field {key!r} is not allowed "
                    f"(allowed: {', '.join(sorted(allowed_keys))})",
                    range=cursor.range_of(key),
                )
        for key in cls.required:
            if key not in body:
                raise diagnostics.error(
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
        constructor). `env:` is validated to be a mapping with string keys, and `depends_on:`/
        `runs_after:` to be lists of strings — a bad shape raises a located `LoadError` naming
        `cls.kind`. A `node_name:` YAML read as non-text is a located `LoadError` (`null` stays
        absent); otherwise it is copied through untouched. Whether each ordering target names
        a real node is checked later by the loader edge pass."""
        diagnostics = LoadDiagnostics.from_cursor(cursor)
        body = cursor._value_view()
        for key in ("node_name", "depends_on", "runs_after"):
            if key in body:
                value = cursor.get(key)
                rng = cursor.range_of(key)
                if key == "node_name":
                    reject_non_text(cls.kind, node_id, key, value, rng, diagnostics)
                else:
                    cls._check_ordering_field(node_id, key, value, cursor, diagnostics)
                kwargs[key] = value
                if rng is not None:
                    ranges[("field", key)] = rng
        if "env" in body:
            env = body["env"]
            if not isinstance(env, Mapping) or not all(isinstance(k, str) for k in env):
                raise diagnostics.error(
                    f"node {node_id!r} (kind={cls.kind}): `env:` must be a mapping "
                    f"with string keys",
                    range=cursor.range_of("env"),
                )
            kwargs["env"] = cursor.mapping("env")
            rng = cursor.range_of("env")
            if rng is not None:
                ranges[("field", "env")] = rng

    @classmethod
    def _check_ordering_field(
        cls,
        node_id: str,
        key: str,
        value: Any,
        cursor: SourceCursor,
        diagnostics: LoadDiagnostics,
    ) -> None:
        """Raise a located `LoadError` unless a `depends_on:`/`runs_after:` value is a list of
        strings. A bare string gets a hint to wrap it in a list, the likely intent."""
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return
        if isinstance(value, list):
            got = "a list with a non-string item"
        elif isinstance(value, dict):
            got = "a mapping"
        elif value is None:
            got = "null"
        else:
            got = f"{type(value).__name__} {value_repr(value)}"
        message = (
            f"node {node_id!r} (kind={cls.kind}): `{key}:` must be a list of node ids "
            f"(got {got})"
        )
        if isinstance(value, str) and value:
            message += f"; write `{key}: [{value}]`"
        raise diagnostics.error(message, range=cursor.range_of(key))

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

        Kinds customize the two hooks and may wrap `parse` itself (agent, human_input, case,
        and map do). Returns the fully-populated descriptor instance (`cls`, never the abstract
        base)."""
        cursor = cls._normalize_cursor(cursor)
        diagnostics = LoadDiagnostics.from_cursor(cursor)
        cls._check_keys(node_id, cursor)
        body = cursor._value_view()
        own_range = cursor.range
        ranges: dict[tuple[str, str], SourceRange] = {}
        kwargs: dict[str, Any] = {"id": node_id}
        for key in cls.allowed - NodeDescriptor.allowed:  # this kind's OWN keys only
            if key not in body:
                continue
            rng = cursor.range_of(key)
            if rng is not None:
                ranges[("field", key)] = rng
            kwargs[cls.keyword_remap.get(key, key)] = cursor.get(key)
        cls._validate_fields(node_id, kwargs, ranges, own_range, diagnostics)
        cls._read_common_fields(node_id, cursor, kwargs, ranges)
        return cls(**kwargs, range=own_range, field_ranges=ranges)

    @classmethod
    def _normalize_cursor(cls, cursor: SourceCursor) -> SourceCursor:
        """Return the cursor view inspected by `_check_keys`; default unchanged.

        A hook the template `parse` calls first, before any key is inspected. Kinds may
        override it for source-aware projection or retired-key checks.
        """

        return cursor

    @classmethod
    def _validate_fields(
        cls,
        node_id: str,
        kwargs: dict[str, Any],
        ranges: dict[tuple[str, str], SourceRange],
        own_range: Optional[SourceRange],
        diagnostics: LoadDiagnostics,
    ) -> None:
        """Kind-specific field legality, run by the template `parse` after the own-key loop.

        Receives `kwargs` (already remapped/populated by the loop), not the raw body, plus
        `own_range` (the node body's range, for anchoring an error about an ABSENT field, which
        has no per-field range). Default: if an `asserts:` field is present, it must be a `list`
        of `str` — a bad shape raises a located `LoadError` naming `cls.kind`. Presence-guarded:
        absent `asserts` is a no-op, so kinds that never author `asserts` are unaffected. Kinds
        override it for their own fields and call `super()`.
        """
        asserts = kwargs.get("asserts")
        if asserts is not None and not (
            isinstance(asserts, list) and all(isinstance(a, str) for a in asserts)
        ):
            raise diagnostics.error(
                f"node {node_id!r} (kind={cls.kind}): `asserts:` must be a list of strings",
                range=ranges.get(("field", "asserts")),
            )

    # --- Source-origin attachment ------------------------------------------------------

    def with_source_origins(
        self,
        cursor: SourceCursor,
        *,
        site_path: tuple[str | int, ...],
        catalog_builder: "_SourceCatalogBuilder",
    ) -> "NodeDescriptor":
        """Return this descriptor with kind-owned authored origins attached.

        The Compose conductor calls this immutable hook once for every authored
        descriptor after it stamps the common node-header origin. Most kinds own no
        additional semantic source sites and return themselves unchanged. A kind with
        ordered or interior operations overrides the hook and registers those sites
        through the file-owned catalog builder.

        Args:
            cursor (`SourceCursor`):
                Normalized authored node occurrence for this descriptor.
            site_path (`tuple[str | int, ...]`):
                Complete semantic path of the node occurrence.
            catalog_builder (`_SourceCatalogBuilder`):
                File-owned registry that mints source-neutral origins.

        Returns:
            `NodeDescriptor`:
                This immutable descriptor, unchanged by the default implementation.
        """
        return self

    # --- Build-side machinery -----------------------------------------------------------

    def build(self, ctx: BuildContext) -> BuildResult:
        """Build this descriptor into a runtime `Node` (+ its flow-owned wiring), returning a
        `BuildResult`.

        Abstract on the base — each concrete kind overrides it, returning a `BuildResult`
        (typically via `self._finish(node, wiring)`; `case` also passes `edges=`). Its one input
        is the immutable `BuildContext`: `ctx.registry` (the `TypeRegistry` for
        `output_contract`),
        `ctx.resolver` (the child-flow resolver for call/map/loop), the inline-CODE
        traceback label `ctx.source_file`, the optional Compose presentation seam
        `ctx.resolve_origin`, and the optional source-aware `ctx.output_contract_reader`.
        CODE alone reads the filename/resolver pair; every output-authoring leaf may use
        the contract reader at its normal build position. Call/map/loop also read
        `ctx.resolver`.
        """
        raise NotImplementedError(f"{type(self).__name__} has no build")

    def _finish(
        self,
        node,
        wiring,
        edges=(),
    ) -> BuildResult:
        """Attach semantic provenance and return the Compose build product.

        The tail helper every `build` ends with — the ONE place source metadata stamps live, so
        they cannot drift across the per-kind builds: copies this descriptor's opaque `origin`
        onto the semantic node and retains its YAML ranges and a copy of its `field_labels` only
        in the transient `BuildResult`. `edges` defaults to `()` (leaf/call/loop own no edges);
        `case` passes its data+control edges.
        """
        if self.origin is not None and not isinstance(self.origin, OriginRef):
            raise TypeError("node descriptor origin must be an OriginRef or None")
        node.origin = self.origin
        return BuildResult(
            node=node,
            wiring=wiring,
            edges=tuple(edges),
            node_range=self.range,
            field_ranges=dict(self.field_ranges),
            field_labels=dict(self.field_labels),
        )
