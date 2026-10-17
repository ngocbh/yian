"""kind=agent — the LLM-reasoner node's descriptor + parse/build handle.

One of the two-axis `compose/nodes/<kind>.py` modules: it owns the `agent` node's frozen
`AgentDescriptor`, wrapping the template `parse` from `NodeDescriptor` (which records each
field's `SourceRange`) to reject the removed `mode:` key, and overriding `build` (which
constructs the runtime `AgentNode` + wiring). `_finish` stamps only the descriptor's opaque
origin onto that node and retains concrete node/field ranges on the Compose-owned
`BuildResult`. Registered into `KINDS`/`BINDING_FIELDS` at import so the shared dispatch
routes here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Optional

from pydantic import ValidationError

from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.helpers.memory import parse_memory_declarations
from yian.compose.helpers.tags import read_tagged_value
from yian.compose.helpers.types import read_variables
from yian.compose.helpers.yaml_words import former_boolean_fix
from yian.compose.nodes.base import (
    BuildContext,
    BuildResult,
    NodeDescriptor,
    not_text_detail,
    not_text_error,
    reject_non_text,
)
from yian.compose.source import SourceCursor
from yian.defaults import current
from yian.diagnostics import SourceRange
from yian.expr import eval_binding
from yian.expr.grammar import whole_span_interior
from yian.llm_clients import LLMConfig
from yian.llm_clients.config import host_only_field_error
from yian.llm_clients.provider_blocks import PROVIDER_BLOCKS, partial_block_problems
from yian.memory.contracts import DeclSite, MemoryDecl
from yian.nodes.agent import AgentNode
from yian.nodes.agent.llm_values import LLM_CONFIG_KEY, is_value, value_leaves
from yian.nodes.agent.node import is_max_tool_iterations, is_text_output
from yian.nodes.agent.structured import type_to_schema
from yian.nodes.contracts import Binding, Parameter, ValueContract, ValueTemplate
from yian.skills import SkillError
from yian.typesys.values import Type, type_levels

if TYPE_CHECKING:
    from yian.compose.source.catalog import _SourceCatalogBuilder
    from yian.typesys.types import TypeRegistry

# Shared by the AGENT and `adaptive_questions:` errors for the removed `mode:` key.
MODE_REMOVED_FIX = (
    "an AGENT with tools runs the tool loop, one without tools makes a single call; "
    "delete this line"
)


def retries_error(value: Any) -> str | None:
    """Return why `value` is not a valid `retries:` count, or `None` when it is."""
    # bool is an int subclass, so `retries: true` would otherwise mean 1.
    if type(value) is int and value >= 0:
        return None
    return f"must be a non-negative integer, got {value!r}"


def _former_boolean_hint(value: Any) -> str:
    """Return `"; "` and the fix for a YAML 1.1 boolean spelling in a boolean field, else `""`.

    Args:
        value (`Any`):
            The field's value as YAML read it.

    Returns:
        `str`:
            Such as ``"; `yes` is text; write `true`"``, or `""` for any other value.
    """
    fix = former_boolean_fix(value)
    return "" if fix is None else f"; {fix[1]}"


def read_llm_config(
    cursor: SourceCursor,
    registry: "TypeRegistry",
    *,
    prefix: str,
    site_path: tuple[str | int, ...],
    catalog_builder: "_SourceCatalogBuilder",
    diagnostics: LoadDiagnostics,
) -> tuple[Any, dict[str, SourceRange]]:
    """Read one authored `llm_config` layer with the `${...}` values and tags in it.

    A layer without values is a plain dict whose strings are `$$`-unescaped; a tagged
    literal in it stands for its value. One holding values is a `ValueTemplate`. A
    whole record written as exactly one `${...}` (optionally tagged) is a `Binding`;
    any other top-level text or scalar is returned as written for the shape check to
    reject.

    Args:
        cursor (`SourceCursor`): The authored `llm_config:` value.
        registry (`TypeRegistry`): The flow's typedefs the tags resolve against.
        prefix (`str`): Message prefix naming the layer, such as
            ``"node 'a': llm_config"``.
        site_path (`tuple[str | int, ...]`): Catalog site prefix for the layer.
        catalog_builder (`_SourceCatalogBuilder`): File-owned origin catalog builder.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.

    Returns:
        `tuple[Any, dict[str, SourceRange]]`: The layer, and for a `ValueTemplate` the
            value range of each key written in it, keyed by its dotted path
            (``"anthropic.thinking"``); empty otherwise.

    Raises:
        `LoadError`: A tag is malformed or names no type, a tag sits on the whole
            mapping, a tag has no value, or the tag reader rejects the value.
    """
    slot = read_tagged_value(
        cursor,
        registry,
        label="llm_config",
        site_path=site_path,
        catalog_builder=catalog_builder,
        diagnostics=diagnostics,
        ledger=catalog_builder.consumed_tags,
        nested_call_fix="compute it in a node and read that node's output with `${...}`, "
        "or reword the text",
    )
    if not slot.has_value:
        raise diagnostics.error(
            f"{prefix}: a tag needs a value: `llm_config: !{cursor.tag} ${{...}}`",
            range=cursor.range,
        )
    value = slot.value
    if isinstance(value, str):
        if whole_span_interior(value) is None:
            return value, {}
        origin = catalog_builder.register((*site_path, "path"), cursor)
        return Binding(source=value, origin=origin, contract=slot.contract), {}
    if slot.contract is not None:
        raise diagnostics.error(
            f"{prefix}: a tag on the whole mapping types nothing; tag a field's value, "
            "or write the whole record as one value: `llm_config: !T ${...}`",
            range=cursor.range,
        )
    if not isinstance(value, ValueTemplate):
        return value, {}
    if not value_leaves(value):
        # Only tagged literals, already checked against their tags: a literal layer.
        return _literal_part(value.value), {}
    ranges: dict[str, SourceRange] = {}
    _collect_key_ranges(cursor, "", ranges)
    return value, ranges


def _collect_key_ranges(
    cursor: SourceCursor, prefix: str, ranges: dict[str, SourceRange]
) -> None:
    """Record the value range of every mapping key under `cursor` by its dotted path."""
    value = cursor.value
    if type(value) is not dict:
        return
    for key in value:
        if not isinstance(key, str):
            continue
        child = cursor.child(key)
        path = f"{prefix}{key}"
        if child.range is not None:
            ranges[path] = child.range
        _collect_key_ranges(child, f"{path}.", ranges)


def _holds_value(tree: Any) -> bool:
    """Whether a layer part is, or holds, a value resolved later; a tagged literal is not."""
    if is_value(tree):
        return True
    if type(tree) is dict:
        return any(_holds_value(item) for item in tree.values())
    if type(tree) is list:
        return any(_holds_value(item) for item in tree)
    return False


def _unescape_literals(tree: Any) -> Any:
    """Return `tree` with its values in place and its literals as the values they bind."""
    if is_value(tree):
        return tree
    if type(tree) is dict:
        return {key: _unescape_literals(item) for key, item in tree.items()}
    if type(tree) is list:
        return [_unescape_literals(item) for item in tree]
    return _literal_part(tree)


def _block_problem_range(
    key: str, problem: str, field_ranges: Mapping[str, SourceRange]
) -> SourceRange | None:
    """Return the range of the deepest written key a `partial_block_problems` message names."""
    # Each message starts with the backticked key path, such as `anthropic.thinking.type`.
    path = key
    if problem.startswith("`"):
        rest = problem[1:].split("`", 1)[0].partition(".")[2]
        if rest:
            path = f"{key}.{rest}"
    while path not in field_ranges and path != key:
        path = path[: max(path.rfind("."), path.rfind("["))]
    return field_ranges.get(path)


def _literal_part(tree: Any) -> Any:
    """Return the value a literal template part binds: tags dropped, strings `$$`-unescaped."""
    if isinstance(tree, Binding):
        return _literal_part(tree.source)
    if isinstance(tree, str):
        return eval_binding(tree, lambda path: None)
    if type(tree) is dict:
        return {key: _literal_part(item) for key, item in tree.items()}
    if type(tree) is list:
        return [_literal_part(item) for item in tree]
    return tree


def _host_only_message(prefix: str, message: str) -> str:
    """Put `prefix` in place of the leading `llm_config` of a host-only field message."""
    # A message that does not start with `llm_config` is kept whole after the prefix.
    rest = message.removeprefix("llm_config")
    return prefix + rest if rest != message else f"{prefix}: {message}"


def _host_only_field(message: str) -> str:
    """Return the field a `host_only_field_error` message names (`llm_config.<field> ...`)."""
    return message.removeprefix("llm_config.").split(" ", 1)[0]


def _declared_fields(typ: Type) -> set[str]:
    """Return the field names a record type, or any member of a record union, declares."""
    names = set(typ.fields or ())
    for member in typ.variants or ():
        names |= set(member.fields or ())
    return names


def check_llm_config_value(
    cfg: ValueTemplate | Binding,
    prefix: str,
    source_range: SourceRange | None,
    diagnostics: LoadDiagnostics,
    field_ranges: Mapping[str, SourceRange],
    *,
    allow_inherit: bool = True,
) -> tuple[ValueTemplate | Binding, bool]:
    """Check an `llm_config` layer holding values, and split off its literal `inherit`.

    Fields that only the host sets may not appear, from a value or not, and a
    whole-record type may not declare them; ``base_url`` beside a literal
    ``provider: azure`` must itself be literal. ``inherit`` must be a literal boolean.
    The literal top-level fields are checked as `LLMConfig` fields, and the literal
    parts of a provider block holding values against that block.

    Args:
        cfg (`ValueTemplate | Binding`): The layer, or the whole-record value.
        prefix (`str`): Message prefix naming the layer.
        source_range (`SourceRange | None`): The layer's location, when retained.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.
        field_ranges (`Mapping[str, SourceRange]`): Value range of each key written
            in the layer, by dotted path.
        allow_inherit (`bool`, *optional*, defaults to `True`): Whether the layer may
            set ``inherit`` (an AGENT's own layer); elsewhere it is an unknown field.

    Returns:
        `tuple[ValueTemplate | Binding, bool]`: The layer without ``inherit``, then
            ``inherit``.

    Raises:
        `LoadError`: Located at the offending field when its range is known.
    """
    if isinstance(cfg, Binding):
        if cfg.contract is not None:
            declared = _declared_fields(cfg.contract.type)
            message = host_only_field_error(dict.fromkeys(declared))
            if message is not None:
                raise diagnostics.error(
                    f"{prefix}: its type declares `{_host_only_field(message)}`; {message}",
                    range=source_range,
                )
        return cfg, True
    tree = cfg.value
    if type(tree) is not dict:
        raise diagnostics.error(f"{prefix} must be a mapping", range=source_range)

    def at(path: str) -> SourceRange | None:
        return field_ranges.get(path) or source_range

    inherit = True
    if allow_inherit and "inherit" in tree:
        if _holds_value(tree["inherit"]):
            raise diagnostics.error(
                f"{prefix} `inherit:` must be written as true or false; it decides which "
                "layers apply before any value is known",
                range=at("inherit"),
            )
        inherit = _literal_part(tree["inherit"])
        if type(inherit) is not bool:
            raise diagnostics.error(
                f"{prefix} `inherit:` must be true or false, got "
                f"{inherit!r}{_former_boolean_hint(inherit)}",
                range=at("inherit"),
            )
        tree = {key: item for key, item in tree.items() if key != "inherit"}
        origins = tuple(
            (path, origin) for path, origin in cfg.origins if path[:1] != ("inherit",)
        )
        cfg = ValueTemplate(tree, origins)
    # Each literal field as the value it binds, so a tagged `provider: !str azure`
    # counts as `azure`; a field holding a value is None here.
    literal = {
        key: None if _holds_value(item) else _literal_part(item)
        for key, item in tree.items()
    }
    host_only = host_only_field_error(literal)
    if host_only is not None:
        raise diagnostics.error(
            _host_only_message(prefix, host_only), range=at(_host_only_field(host_only))
        )
    if "base_url" in tree and _holds_value(tree["base_url"]):
        raise diagnostics.error(
            f"{prefix}.base_url can't come from a value: it decides where the key is "
            "sent, which stays with the host or `yian auth`; write the Azure resource "
            "endpoint as literal text",
            range=at("base_url"),
        )
    _check_literal_fields(literal, prefix, source_range, diagnostics, at)
    for key, block in tree.items():
        if key not in PROVIDER_BLOCKS or not _holds_value(block):
            continue
        problems = partial_block_problems(key, _unescape_literals(block), is_value)
        if problems:
            raise diagnostics.error(
                f"{prefix}: {problems[0]}",
                range=_block_problem_range(key, problems[0], field_ranges) or source_range,
            )
    return cfg, inherit


def _check_literal_fields(
    cfg: dict[str, Any],
    prefix: str,
    source_range: SourceRange | None,
    diagnostics: LoadDiagnostics,
    field_range: Callable[[str], SourceRange | None] | None = None,
) -> None:
    """Check literal `llm_config` fields as `LLMConfig` fields.

    Args:
        cfg (`dict[str, Any]`): The literal fields; a field holding a value is `None`.
        prefix (`str`): Message prefix naming the layer.
        source_range (`SourceRange | None`): The layer's location, when retained.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.
        field_range (`Callable[[str], SourceRange | None] | None`, *optional*, defaults
            to `None`): The location of a top-level field by its key; `None` locates
            every error at the layer.

    Raises:
        `LoadError`: A field violates `LLMConfig`; a text field YAML read as non-text
            is named by `not_text_error`, any other violation keeps pydantic's message.
    """
    try:
        LLMConfig(**cfg)
    except ValidationError as exc:
        detail = not_text_detail(exc)
        where = source_range
        errors = exc.errors(include_input=False, include_url=False)
        loc = errors[0]["loc"] if errors else ()
        if field_range is not None and loc and isinstance(loc[0], str):
            where = field_range(loc[0])
        raise diagnostics.error(
            f"{prefix}: " + (not_text_error(*detail) if detail is not None else str(exc)),
            range=where,
        ) from exc
    except Exception as exc:
        raise diagnostics.error(
            f"{prefix}: {exc}",
            range=source_range,
        ) from exc


def _validate_llm_config(
    cfg: Mapping[str, Any] | ValueTemplate | Binding | None,
    prefix: str,
    source_range: SourceRange | None = None,
    diagnostics: LoadDiagnostics | None = None,
    field_ranges: Mapping[str, SourceRange] | None = None,
) -> tuple[dict[str, Any] | ValueTemplate | Binding, bool]:
    """Validate an authored llm_config layer at LOAD time and split off `inherit`.

    `inherit` (whole-node cascade opt-out, default True) is NOT an `LLMConfig` field, so it
    must be popped before the `extra="forbid"` round-trip that fires on typo'd keys (e.g.
    `temparature`). Returns `(config_without_inherit, inherit)`. Raises `LoadError` (not
    pydantic ValidationError) so the surface speaks one error type. A layer holding
    `${...}` values (a `ValueTemplate`, or a `Binding` for the whole record) is checked by
    `check_llm_config_value`; its values are checked when they resolve.

    Module-local to `agent.py`: agent is the only kind with an `llm_config:` field, so the
    validator lives next to its sole caller rather than in shared machinery.

    Args:
        cfg (`Mapping[str, Any] | ValueTemplate | Binding | None`): Authored config, or
            `None` for defaults.
        prefix (`str`): Message prefix naming the config, such as
            ``"node 'a': llm_config"``.
        source_range (`SourceRange | None`): Authored config location, when retained.
        diagnostics (`LoadDiagnostics | None`): Factory for located load errors.
        field_ranges (`Mapping[str, SourceRange] | None`): For a layer holding values,
            the value range of each key written in it, by dotted path.

    Returns:
        `tuple[dict[str, Any] | ValueTemplate | Binding, bool]`: Config without
            `inherit`, then its boolean value.

    Raises:
        `LoadError`: The config is not a mapping, sets a non-boolean `inherit` or one
            from a value, sets `api_key`, `organization`, `project` or `headers`, sets
            `base_url` other than beside `provider: azure` or from a value, has a
            whole-record type declaring one of those, or violates `LLMConfig`. A text
            field YAML read as non-text (such as `model: 5`) is named by
            `not_text_error`; any other violation keeps pydantic's message.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    if cfg is None:
        return {}, True
    if isinstance(cfg, (ValueTemplate, Binding)):
        return check_llm_config_value(
            cfg, prefix, source_range, diagnostics, field_ranges or {}
        )
    if not isinstance(cfg, Mapping):
        raise diagnostics.error(
            f"{prefix} must be a mapping",
            range=source_range,
        )
    cfg = dict(cfg)
    if not cfg:
        return {}, True
    inherit = cfg.pop("inherit", True)
    if type(inherit) is not bool:
        raise diagnostics.error(
            f"{prefix} `inherit:` must be true or false, got "
            f"{inherit!r}{_former_boolean_hint(inherit)}",
            range=source_range,
        )
    host_only = host_only_field_error(cfg)
    if host_only is not None:
        raise diagnostics.error(_host_only_message(prefix, host_only), range=source_range)
    _check_literal_fields(cfg, prefix, source_range, diagnostics)
    return cfg, inherit


@dataclass(frozen=True, kw_only=True)
class AgentDescriptor(NodeDescriptor):
    """kind=agent — an LLM reasoner (prompt + optional system prompt + optional tools/llm).

    Inherits the common descriptor fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` +
    the Compose-only `range`/`field_ranges` parse metadata) from `NodeDescriptor`, along with the `input:`/
    `output:` node-signature surface and the template `parse`, which it wraps to reject the
    removed `mode:` key. Its cursor hook rejects the retired memory key before the default
    allowed/required key check; only the agent-specific fields are declared here, while
    `inputs`/`outputs` come from the base.

    Class attrs: `kind="agent"`; `allowed` adds normalized `inputs`/`outputs` plus `prompt`/
    `system_prompt`/`tools`/`variables`/`llm_config`/`long_term_memory`/`retries`/`asserts`/
    `steerable` to the commons; `required` is empty (no mandatory flat field);
    `binding_fields=("inputs",)` — the `${...}` refs of `prompt:` and `system_prompt:` are handled
    by template scanning, not the inline-call desugar, so only `inputs` is walked.
    """

    kind = "agent"
    allowed = NodeDescriptor.allow(
        "inputs",
        "outputs",
        "prompt",
        "system_prompt",
        "variables",
        "tools",
        "skills",
        "llm_config",
        "long_term_memory",
        "retries",
        "asserts",
        "steerable",
    )
    required = NodeDescriptor.require()
    binding_fields = ("inputs",)
    keyword_remap = {"long_term_memory": "memories"}

    prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    variables: dict[str, Any] = field(default_factory=dict)
    tools: list[str] = field(default_factory=list)
    skills: list[str] | str = field(default_factory=list)
    # A plain dict, or a `ValueTemplate`/`Binding` once read with the values in it.
    llm_config: dict[str, Any] | ValueTemplate | Binding = field(default_factory=dict)
    # The effective count, so the manifest holds it and a default flow keeps its identity.
    retries: int = field(default_factory=lambda: current().agent_retries)
    asserts: list[str] = field(default_factory=list)
    steerable: bool = False
    memories: tuple[MemoryDecl, ...] = ()
    # INTERNAL — never parsed from YAML; set only by the adaptive_questions desugar pass.
    # When present it overrides the surface-derived contract for a synthesized structured
    # agent whose codomain is code-built and has no surface type-string.
    output_contract_override: Optional[ValueContract] = None

    def __post_init__(self) -> None:
        """Validate the source-neutral declaration tuple."""
        super().__post_init__()
        if type(self.memories) is not tuple or any(
            type(decl) is not MemoryDecl for decl in self.memories
        ):
            raise TypeError("agent descriptor memories must be a tuple of exact MemoryDecl values")

    @classmethod
    def parse(cls, node_id: str, cursor: SourceCursor) -> "AgentDescriptor":
        """Reject the removed `mode:` key, then parse as usual.

        Raises:
            `LoadError`: The node authors `mode:`, or the template `parse` rejects it.
        """
        if cursor.has("mode"):
            message = f"node {node_id!r}: `mode:` was removed: {MODE_REMOVED_FIX}"
            tools = cursor.get("tools")
            if cursor.get("mode") == "plain" and isinstance(tools, list) and tools:
                message += " (its `tools:` will now be used)"
            raise LoadDiagnostics.from_cursor(cursor).error(
                message, range=cursor.key_range_of("mode")
            )
        return super().parse(node_id, cursor)

    @classmethod
    def _normalize_cursor(cls, cursor: SourceCursor) -> SourceCursor:
        """Reject the retired authored memory key before generic key checking."""
        if cursor.has("memory_policy"):
            diagnostics = LoadDiagnostics.from_cursor(cursor)
            raise diagnostics.error(
                "node field `memory_policy:` was renamed to `long_term_memory:`",
                range=cursor.key_range_of("memory_policy"),
            )
        return cursor

    @classmethod
    def _validate_fields(
        cls,
        node_id: str,
        kwargs: dict[str, Any],
        ranges: dict[tuple[str, str], SourceRange],
        own_range: SourceRange | None,
        diagnostics: LoadDiagnostics,
    ) -> None:
        """Check the `prompt:`, `system_prompt:`, `tools:`, `skills:`, `retries:`, and `steerable:` shapes and parse a present named policy."""
        reject_non_text(
            cls.kind,
            node_id,
            "prompt",
            kwargs.get("prompt"),
            ranges.get(("field", "prompt")) or own_range,
            diagnostics,
        )
        if "system_prompt" in kwargs:
            system_prompt = kwargs["system_prompt"]
            if not isinstance(system_prompt, str) or not system_prompt:
                raise diagnostics.error(
                    f"node {node_id!r} (kind={cls.kind}): `system_prompt:` must be a "
                    "non-empty string",
                    range=ranges.get(("field", "system_prompt")) or own_range,
                )
        if "tools" in kwargs:
            tools = kwargs["tools"]
            if not isinstance(tools, list) or not all(
                isinstance(name, str) and name for name in tools
            ):
                example = tools if isinstance(tools, str) and tools else "search"
                raise diagnostics.error(
                    f"node {node_id!r} (kind={cls.kind}): `tools:` must be a list of "
                    f"tool names; write `tools: [{example}]`",
                    range=ranges.get(("field", "tools")) or own_range,
                )
        if "retries" in kwargs:
            problem = retries_error(kwargs["retries"])
            if problem is not None:
                raise diagnostics.error(
                    f"node {node_id!r} (kind={cls.kind}): `retries:` {problem}",
                    range=ranges.get(("field", "retries")) or own_range,
                )
        skills = kwargs.get("skills")
        if skills is not None and skills != "all" and not (
            isinstance(skills, list) and all(isinstance(name, str) for name in skills)
        ):
            raise diagnostics.error(
                f"node {node_id!r} (kind={cls.kind}): `skills:` must be a list of "
                "skill names or `all`",
                range=ranges.get(("field", "skills")) or own_range,
            )
        steerable = kwargs.get("steerable", False)
        if type(steerable) is not bool:
            raise diagnostics.error(
                f"node {node_id!r} (kind={cls.kind}): `steerable:` must be true or false"
                f"{_former_boolean_hint(steerable)}",
                range=ranges.get(("field", "steerable")) or own_range,
            )
        if "memories" in kwargs:
            value = kwargs["memories"]
            kwargs["memories"] = (
                ()
                if value is None
                else parse_memory_declarations(
                    value,
                    diagnostics=diagnostics,
                    source_range=ranges.get(("field", "long_term_memory")),
                    site=DeclSite((), ("root",), node_id),
                )
            )
        super()._validate_fields(
            node_id,
            kwargs,
            ranges,
            own_range,
            diagnostics,
        )

    def with_source_origins(
        self,
        cursor: SourceCursor,
        *,
        site_path: tuple[str | int, ...],
        catalog_builder: "_SourceCatalogBuilder",
    ) -> "AgentDescriptor":
        """Attach one origin per authored ``long_term_memory:`` entry."""
        if not self.memories or not cursor.has("long_term_memory"):
            return self
        value_cursor = cursor.child("long_term_memory")
        stamped = []
        for decl in self.memories:
            path = (*site_path, "field", "memories", decl.name)
            if value_cursor.has(decl.name):
                origin = catalog_builder.register(
                    path, value_cursor.child(decl.name), anchor="key"
                )
            else:  # sugar: the whole scalar value is the one entry
                origin = catalog_builder.register(path, value_cursor)
            stamped.append(
                MemoryDecl(decl.name, decl.type, decl.config, decl.site, origin)
            )
        return replace(self, memories=tuple(stamped))

    def build(self, ctx: BuildContext) -> BuildResult:
        """Construct the runtime `AgentNode` (+ wiring) for this `AgentDescriptor`.

        Builds the agent leaf (LLM-config and local-variable validation, node-side
        `params` + flow-owned `inputs` wiring, `output_contract` from authored `output:`/override).
        `_finish` stamps the descriptor origin on the node and keeps its ranges on the
        Compose-owned `BuildResult`. `ctx.resolver` is unused (leaf kind), and the inline-CODE
        anchors on `ctx` are read only by the CODE build.
        """
        llm_config_range = self.field_ranges.get(("field", "llm_config")) or self.range
        own_cfg, inherit = _validate_llm_config(
            self.llm_config,
            self.field_labels.get("llm_config") or f"node {self.id!r}: llm_config",
            llm_config_range,
            ctx.diagnostics,
            {
                path: rng
                for (kind, path), rng in self.field_ranges.items()
                if kind == "llm_config"
            },
        )
        steerable_range = self.field_ranges.get(("field", "steerable")) or self.range
        if "max_tool_iterations" in self.env and not is_max_tool_iterations(
            self.env["max_tool_iterations"]
        ):
            raise ctx.diagnostics.error(
                f"node {self.id!r}: `env.max_tool_iterations` must be a positive int or -1 "
                f"(no cap), got {self.env['max_tool_iterations']!r}",
                range=self.field_ranges.get(("field", "env")) or self.range,
            )
        variables = self._read_variables(ctx)
        memories = tuple(
            MemoryDecl(decl.name, decl.type, decl.config, DeclSite((), ctx.owner, self.id), decl.origin)
            for decl in self.memories
        )
        try:
            node = AgentNode(
                self.id,
                prompt=self.prompt or "",
                system_prompt=self.system_prompt,
                tools=list(self.tools),
                llm_config=own_cfg,
                llm_inherit=inherit,
                retries=self.retries,
                title=self.node_name,
                memories=memories,
                variables=variables,
                skills=self.skills,
                steerable=self.steerable,
            )
        except ValueError as exc:
            # AgentNode validates skill grants and tool names; surface a
            # rejection as the loader's one error type (loud), not a bare ValueError.
            skills_range = self.field_ranges.get(("field", "skills")) or self.range
            tools_range = self.field_ranges.get(("field", "tools")) or self.range
            raise ctx.diagnostics.error(
                f"node {self.id!r}: {exc}",
                range=skills_range if isinstance(exc, SkillError) else tools_range,
            ) from exc
        node.llm_config_label = self.field_labels.get("llm_config")

        override = self.output_contract_override
        node.output_contract = (
            override
            if override is not None
            else self.output_contract(ctx)
        )
        if type(node.output_contract) is ValueContract:
            outputs_range = self.field_ranges.get(("field", "outputs")) or self.range
            # Schema generation, here and at run, recurses per level; deeper types aren't
            # proven to fit the stack. An engine-built output type is not the author's.
            levels = type_levels(node.output_contract.type)
            limit = current().max_agent_output_nesting
            if override is None and levels > limit:
                raise ctx.diagnostics.error(
                    f"node {self.id!r}: this AGENT's `output:` nests {levels} levels, deeper "
                    f"than {limit}, the limit for a model's structured reply (each record "
                    "and list is a level); flatten it, or build the deeper value in a CODE "
                    "node",
                    range=outputs_range,
                )
            try:
                type_to_schema(node.output_contract.type)
            except ValueError as exc:
                raise ctx.diagnostics.error(
                    f"node {self.id!r}: an AGENT can't produce this `output:`; a model "
                    "generates only str, int, float, bool, date, datetime, Literal, record, "
                    "union, object, Any, and list values, not `file` or `None`. Produce "
                    "the value in a CODE node.",
                    range=outputs_range,
                ) from exc
        if self.steerable and not is_text_output(node.output_contract):
            # Checked here too so the error points at the `steerable:` line.
            raise ctx.diagnostics.error(
                f"node {self.id!r}: steerable needs `str` output",
                range=steerable_range,
            )
        node.params = self.sink_params()
        input_names = {parameter.name for parameter in node.params}
        for variable in variables:
            if variable.name in input_names:
                frame = (
                    ctx.resolve_origin(variable.origin)
                    if ctx.resolve_origin is not None and variable.origin is not None
                    else None
                )
                message = f"node {self.id!r}: variable {variable.name!r} collides with an input name"
                if frame is not None:
                    raise ctx.diagnostics.error(message, frames=(frame,))
                raise ctx.diagnostics.error(
                    message,
                    range=self.field_ranges.get(("field", "variables")) or self.range,
                )
        if LLM_CONFIG_KEY in input_names:
            raise ctx.diagnostics.error(
                f"node {self.id!r}: input {LLM_CONFIG_KEY!r} is reserved for the values "
                "of `llm_config`; rename the input",
                range=self.field_ranges.get(("input", LLM_CONFIG_KEY)) or self.range,
            )
        wiring = self.sink_wiring()
        result = self._finish(node, wiring)
        # Errors about `llm_config` values point at the authored `llm_config:`.
        return replace(
            result,
            field_ranges={**result.field_ranges, ("field", LLM_CONFIG_KEY): llm_config_range},
        )

    def _read_variables(self, ctx: BuildContext) -> list[Parameter]:
        """Compile this node's declarations at their authored or source-free site."""
        try:
            if ctx.variable_declaration_reader is not None:
                variables = ctx.variable_declaration_reader(self.id, self.variables)
                if variables is not None:
                    if type(variables) is not list or any(
                        type(variable) is not Parameter for variable in variables
                    ):
                        raise TypeError(
                            "variable declaration reader must return a list of exact Parameters or None"
                        )
                    return variables
            # Only a node with no authored source gets here; a variable needs a
            # default, which only a tagged YAML declaration carries, so any
            # declaration fails.
            return read_variables(self.variables, ctx.registry, diagnostics=ctx.diagnostics)
        except LoadError as exc:
            if exc.diagnostic.frames:
                raise
            raise ctx.diagnostics.prepend(
                exc,
                range=self.field_ranges.get(("field", "variables")) or self.range,
            ) from exc
