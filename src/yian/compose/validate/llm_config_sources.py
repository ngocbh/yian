"""Graph-stage: the sources of `llm_config` values must be typed and fit their fields.

A value in an AGENT's own `llm_config`, in an adaptive composer's, or in the flow's
`llm_config` layer is checked where it is written, against the flow's contracts as
written (a config sees no CASE narrowing):

- the source needs a proven type or a tag; an unproven or untagged `Any` source is
  rejected with "tag it";
- `provider` and `model` take `str`, and a `Literal` provider needs at least one known
  provider among its members;
- a provider-block field whose annotation is `str`, `int`, `float` or `bool` (with its
  inclusive bounds) or a `Literal` takes that type; any other block field is checked
  when the value resolves;
- a whole record must be a record of `llm_config` fields other than the ones only the
  host sets and `inherit`, or `object`; a whole block a record of that block's fields,
  or `object`;
- no `isinstance` or constructor call.

Only a disjoint type is a load error; a narrower one is checked when the value
resolves. The flow layer is checked once, so a value every agent overrides is still
checked, its references included.

Imports flow down only: `compile.validation` (the type relation and reference paths),
`compose.helpers` (constructor names, tag labels, load diagnostics, output contracts),
`compose.nodes.base` (build results), `compose.validate.typed_bindings` (binding type
inference), `llm_clients`, `nodes`, `typesys`, `expr` and `diagnostics`.
"""

from __future__ import annotations

from typing import Any, Optional

from yian.compile.validation import _classify_path, type_relation
from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.helpers.isinstance_types import (
    collect_constructor_names,
    collect_isinstance_names,
)
from yian.compose.helpers.output_contracts import TypeFn
from yian.compose.helpers.tags import named_type_label
from yian.compose.nodes.base import BuildResult
from yian.compose.validate.typed_bindings import ResolveOrigin, infer_binding_type
from yian.diagnostics import SourceRange
from yian.expr import ExpressionError, expr_refs_of
from yian.llm_clients.config import (
    LLM_CONFIG_KEYS,
    SUPPORTED_LLM_PROVIDERS,
    host_only_field_error,
)
from yian.llm_clients.provider_blocks import PROVIDER_BLOCKS, block_field_annotation
from yian.nodes.agent import AgentNode
from yian.nodes.agent.llm_values import value_leaves
from yian.nodes.contracts import (
    Binding,
    OutputContract,
    ValueContract,
    ValuePath,
    ValueTemplate,
    iter_source_strings,
)
from yian.typesys import Type, TypeRegistry, ValueKind

_TEXT = Type.scalar(ValueKind.STRING)
_KINDS = {
    str: ValueKind.STRING,
    int: ValueKind.INTEGER,
    float: ValueKind.NUMBER,
    bool: ValueKind.BOOLEAN,
}


def check_llm_config_sources(
    build_results: dict[str, BuildResult],
    flow_llm_config: Any,
    flow_label: str,
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
    *,
    type_fn: TypeFn | None = None,
    registry: TypeRegistry | None = None,
    resolve_origin: ResolveOrigin | None = None,
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """Reject an `llm_config` value whose source is untyped or can never fit its field.

    Args:
        build_results (`dict[str, BuildResult]`): Built nodes with their transient
            ranges and labels; each AGENT's own layer is checked.
        flow_llm_config (`Any`): The flow's `llm_config:` layer: a dict, a
            `ValueTemplate` or a `Binding`.
        flow_label (`str`): Names the flow layer in a message, such as
            ``"flow llm_config"``.
        producers (`dict[str, OutputContract]`): Node output contracts by producer id.
        flow_inputs (`dict[str, ValueContract]`): Flow input contracts by name.
        type_fn (`TypeFn | None`, *optional*, defaults to `None`): Resolves a
            constructor name to its type for inference.
        registry (`TypeRegistry | None`, *optional*, defaults to `None`): The flow's
            typedefs, so a message names a declared record by its typedef.
        resolve_origin (`ResolveOrigin | None`, *optional*, defaults to `None`):
            Resolves a value's registered origin to its source frame, the preferred
            location.
        diagnostics (`LoadDiagnostics | None`, *optional*, defaults to `None`):
            Factory for located load errors.

    Raises:
        `LoadError`: At the first value that reads an unknown name, calls `isinstance`
            or a constructor, has no proven type, or has a type disjoint from its field.
    """
    checker = _Checker(
        producers,
        flow_inputs,
        set(build_results),
        type_fn,
        registry or {},
        resolve_origin,
        diagnostics or LoadDiagnostics(),
    )
    for node_id, result in build_results.items():
        node = result.node
        if isinstance(node, AgentNode):
            checker.layer(
                node.own_llm_config,
                result.site_name(node_id, "llm_config"),
                result.field_ranges.get(("field", "llm_config")) or result.node_range,
                check_refs=False,  # the reference pass checked the carried values
            )
    checker.layer(flow_llm_config, flow_label, None, check_refs=True)


class _Checker:
    """Checks the values of one layer at a time against one flow's contracts."""

    def __init__(
        self,
        producers: dict[str, OutputContract],
        flow_inputs: dict[str, ValueContract],
        node_ids: set[str],
        type_fn: TypeFn | None,
        registry: TypeRegistry,
        resolve_origin: ResolveOrigin | None,
        diagnostics: LoadDiagnostics,
    ) -> None:
        self.producers = producers
        self.flow_inputs = flow_inputs
        self.node_ids = node_ids
        self.type_fn = type_fn
        self.registry = registry
        self.resolve_origin = resolve_origin
        self.diagnostics = diagnostics

    def layer(
        self, layer: Any, label: str, fallback: Optional[SourceRange], *, check_refs: bool
    ) -> None:
        """Check every value in ``layer``, naming it ``label`` in messages.

        Raises:
            `LoadError`: As `check_llm_config_sources` describes.
        """
        tree = layer.value if isinstance(layer, ValueTemplate) else {}
        for path, leaf in value_leaves(layer).items():
            where = label + "".join(f".{part}" for part in path)
            origin = layer.origin if isinstance(layer, Binding) else (
                leaf.origin if isinstance(leaf, Binding) and leaf.origin is not None
                else layer.origin_at(path)
            )
            frame = (
                self.resolve_origin(origin)
                if self.resolve_origin is not None and origin is not None
                else None
            )

            def fail(message: str) -> LoadError:
                if frame is not None:
                    return self.diagnostics.error(f"{where}: {message}", frames=(frame,))
                return self.diagnostics.error(f"{where}: {message}", range=fallback)

            self._names(leaf, check_refs, fail)
            self._type(path, leaf, tree, fail)

    def _names(self, leaf: Any, check_refs: bool, fail) -> None:
        """Reject an unknown reference (when asked), `isinstance` or a constructor."""
        try:
            isinstance_names = collect_isinstance_names(leaf)
            constructors = collect_constructor_names(leaf)
            refs = [ref for _, text in iter_source_strings(leaf) for ref in expr_refs_of(text)]
        except ExpressionError as exc:
            raise fail(str(exc)) from exc
        if isinstance_names or constructors:
            used = "isinstance" if isinstance_names else f"{constructors[0]}(...)"
            raise fail(
                f"{used} is not supported in an llm_config value; compute the value in a "
                "CODE node and read its output"
            )
        if not check_refs:
            return
        for ref in refs:
            problem = _classify_path(ref, self.node_ids, set(self.flow_inputs))
            if problem is not None:
                raise fail(problem)

    def _type(self, path: ValuePath, leaf: Any, tree: Any, fail) -> None:
        """Reject a value with no proven type, or one disjoint from its field."""
        inferred = infer_binding_type(
            leaf, self.producers, self.flow_inputs, type_fn=self.type_fn
        )
        tagged = isinstance(leaf, Binding) and leaf.contract is not None
        if inferred is None or (inferred.kind is ValueKind.ANY and not tagged):
            raise fail(
                "the value's type isn't known; tag it with the type it holds, e.g. "
                "`model: !str ${...}`"
            )
        if inferred.kind is ValueKind.NONE:
            return  # null means unset
        label = named_type_label(inferred, self.registry)
        if not path:
            self._record(inferred, LLM_CONFIG_KEYS, "llm_config", label, fail)
        elif path[0] in PROVIDER_BLOCKS and len(path) == 1:
            block = PROVIDER_BLOCKS[path[0]]
            keys = {field.alias or name for name, field in block.model_fields.items()}
            self._record(inferred, keys, f"the {path[0]} block", label, fail)
        elif path[0] in PROVIDER_BLOCKS:
            field = block_field_annotation(path[0], tuple(map(str, path[1:])), tree[path[0]])
            if field is not None:
                self._fits(inferred, _block_field_type(field), label, fail)
        elif path == ("provider",):
            self._fits(inferred, _TEXT, label, fail)
            members = sorted(inferred.tags or ())
            if members and not set(members) & set(SUPPORTED_LLM_PROVIDERS):
                raise fail(
                    f"none of {', '.join(members)} is a known provider; known providers "
                    f"are {', '.join(SUPPORTED_LLM_PROVIDERS)}"
                )
        elif path == ("model",):
            self._fits(inferred, _TEXT, label, fail)

    def _fits(self, inferred: Type, declared: Type, label: str, fail) -> None:
        """Reject ``inferred`` when no non-null value of it fits ``declared``."""
        if inferred.kind is ValueKind.ANY or type_relation(inferred, declared) != "disjoint":
            return
        raise fail(
            f"must be {named_type_label(declared, self.registry)}, but the value is {label}"
        )

    def _record(
        self, inferred: Type, keys: set[str] | frozenset[str], owner: str, label: str, fail
    ) -> None:
        """Reject a whole record or block whose type isn't a record of ``keys`` or `object`."""
        if inferred.kind is ValueKind.ANY:
            return
        if inferred.kind is not ValueKind.OBJECT:
            raise fail(f"must be a record of {owner}'s fields, but the value is {label}")
        declared: list[str] = list(inferred.fields or ())
        for member in inferred.variants or ():
            declared.extend(name for name in member.fields or () if name not in declared)
        host_only = host_only_field_error(dict.fromkeys(declared))
        if host_only is not None and owner == "llm_config":
            field = host_only.removeprefix("llm_config.").split(" ", 1)[0]
            raise fail(f"its type declares `{field}`; {host_only}")
        for name in declared:
            if name == "inherit" and owner == "llm_config":
                raise fail(
                    "its type declares `inherit`, which decides which layers apply before "
                    "any value is known; write it as true or false in the agent's own "
                    "llm_config"
                )
            if name not in keys:
                raise fail(f"its type declares `{name}`, which is not a field of {owner}")


def _block_field_type(field: Any) -> Type:
    """Return the type a value for a simple provider-block field must overlap."""
    kind = _KINDS[field.base]
    if field.literals and field.base is str:
        return Type(kind=kind, tags=frozenset(field.literals))
    bounds = None
    if kind in (ValueKind.INTEGER, ValueKind.NUMBER) and (
        field.ge is not None or field.le is not None
    ):
        cast = float if kind is ValueKind.NUMBER else int
        bounds = tuple(None if end is None else cast(end) for end in (field.ge, field.le))
    return Type(kind=kind, bounds=bounds)
