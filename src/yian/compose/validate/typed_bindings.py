"""Graph-stage: reject typed bindings whose value can never fit the declared type.

A typed binding (`name: !Type value`, on a node input or a flow output) states the
type its value must have; the binder enforces it at runtime. A literal right side is
already checked against its type when the binding is split, where the authored text
is at hand. This load pass rejects the remaining bindings that could never pass the
runtime check:

- a string source with no `${...}` span (an escaped `$${x}`) is literal text, so it
  is checked exactly with the same write-boundary check the binder uses;
- a reference or expression is inferred from its raw source over the producer and
  flow-input contracts, and rejected only when that type is disjoint from the
  declared one. A narrower inferred type (an Optional source, a str into a Literal)
  loads and is left to the runtime check, as is an unproven source;
- a lone `${item}` or `${item.field}` read on a node that binds per item (MAP) is
  inferred from the element type of its `over` source, when that is a proven list;
- a value template is inferred from its structure, and each typed binding nested in
  it is checked the same way at its own path and origin.

`infer_binding_type` returns the type one binding is proven to produce (a typed
binding's declared type, a literal's exact type, or a reference's inferred type), for
passes that record binding types rather than reject them.

Node inputs are checked as soon as producer contracts exist, before the CALL, MAP,
and LOOP checks that compare the declared type with a child's parameters; flow
outputs are checked once their contracts are attached. A node input error locates at
the binding's registered value origin, else its `input:` entry range, else the node
range; a flow output error at its registered origin, else the `output:` range.

Imports flow down only: `compile.validation` (the type relation), `expr` (template
scanning), `typesys`, and `compose.helpers` (output-contract inference and load
diagnostics). The loader, the compose package facade, and the CALL/MAP check
(`call_types`, for its literal and `${item}` helpers) import this back.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import date, datetime
from typing import Any, Callable, Optional

from yian.compile.model import FlowOutput
from yian.compile.validation import type_relation
from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.helpers.output_contracts import (
    NOT_LITERAL,
    TypeFn,
    direct_reference_path,
    literal_type,
    literal_value,
    project_output_contract,
    resolve_output_binding_contract,
    template_contract,
)
from yian.compose.helpers.tags import named_type_label, path_label
from yian.compose.nodes.base import BuildResult
from yian.compose.validate.narrowing import Narrowing
from yian.diagnostics import DiagnosticFrame, OriginRef, SourceRange
from yian.nodes.contracts import (
    Binding,
    OutputContract,
    ValuePath,
    ValueContract,
    ValueTemplate,
    binding_source,
)
from yian.typesys import (
    Type,
    TypeRegistry,
    TypeCheckError,
    build_value_as,
    canonical_type_label,
)
from yian.typesys.values import value_repr

ResolveOrigin = Callable[[OriginRef], DiagnosticFrame | None]


def check_typed_inputs(
    flow_wiring: dict[str, dict[str, Any]],
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
    build_results: dict[str, BuildResult],
    *,
    narrowing: Narrowing | None = None,
    type_fn: TypeFn | None = None,
    registry: TypeRegistry | None = None,
    resolve_origin: ResolveOrigin | None = None,
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """Reject every typed node input that can never fit its declared type.

    Args:
        flow_wiring (`dict[str, dict[str, Any]]`): Final binding maps keyed by node id
            and parameter name; typed entries are `Binding`s with a contract.
        producers (`dict[str, OutputContract]`): Exact or partial node output
            contracts keyed by producer id.
        flow_inputs (`dict[str, ValueContract]`): Exact flow input contracts keyed by
            input name.
        build_results (`dict[str, BuildResult]`): Built nodes with their transient
            value ranges, the fallback location for a node input.
        narrowing (`Narrowing | None`, *optional*, defaults to `None`):
            The union narrowing each param reads against, from the CASE branches
            that route its node; `None` reads every param against the unchanged maps.
        type_fn (`TypeFn | None`, *optional*, defaults to `None`): Resolves a
            constructor name to its type for inference.
        registry (`TypeRegistry | None`, *optional*, defaults to `None`): The flow's
            typedefs, so a message names a declared record or union by its typedef.
        resolve_origin (`ResolveOrigin | None`, *optional*, defaults to `None`):
            Resolves a binding's registered origin to its source frame, the
            preferred location.
        diagnostics (`LoadDiagnostics | None`, *optional*, defaults to `None`):
            Factory for located load errors.

    Raises:
        `LoadError`: A literal-text source does not fit its declared type, or a
            source's inferred type is disjoint from it. The input is named by its
            node's `BuildResult.site_name`, so an input a desugar copied onto a synth
            node is named as the node it was authored on.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    for node_id, wiring in flow_wiring.items():
        result = build_results.get(node_id)
        node_producers, node_inputs = (
            narrowing.for_node(node_id) if narrowing is not None else (producers, flow_inputs)
        )
        item = item_contract(result, wiring, node_producers, node_inputs, type_fn)
        for name, binding in wiring.items():
            param_producers, param_inputs = (
                narrowing.for_param(node_id, name)
                if narrowing is not None
                else (node_producers, node_inputs)
            )
            for path, typed in _typed_sites(binding):
                message = _binding_error(
                    typed,
                    param_producers,
                    param_inputs,
                    item=item,
                    type_fn=type_fn,
                    registry=registry,
                    slot=path_label(name, path),
                )
                if message is None:
                    continue
                fallback, where = None, f"node {node_id!r} input {path_label(name, path)!r}"
                if result is not None:
                    fallback = result.field_ranges.get(("input", name)) or result.node_range
                    where = result.site_name(node_id, f"input {path_label(name, path)!r}")
                raise _located_error(
                    f"{where}: {message}",
                    typed,
                    fallback,
                    resolve_origin,
                    diagnostics,
                )


def check_typed_outputs(
    flow_outputs: list[FlowOutput],
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
    *,
    value_mode: bool = False,
    outputs_range: Optional[SourceRange] = None,
    type_fn: TypeFn | None = None,
    registry: TypeRegistry | None = None,
    resolve_origin: ResolveOrigin | None = None,
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """Reject every typed flow output that can never fit its declared type.

    Args:
        flow_outputs (`list[FlowOutput]`): The flow's outputs, typed ones carrying
            their declared contract on the binding.
        producers (`dict[str, OutputContract]`): Exact or partial node output
            contracts keyed by producer id.
        flow_inputs (`dict[str, ValueContract]`): Exact flow input contracts keyed by
            input name.
        value_mode (`bool`, *optional*, defaults to `False`): Whether the flow has a
            single `output:` value, whose internal name the author never wrote, so
            errors name it `output` instead.
        outputs_range (`SourceRange | None`, *optional*, defaults to `None`): The
            authored `output:` range, the fallback location.
        type_fn (`TypeFn | None`, *optional*, defaults to `None`): Resolves a
            constructor name to its type for inference.
        registry (`TypeRegistry | None`, *optional*, defaults to `None`): The flow's
            typedefs, so a message names a declared record or union by its typedef.
        resolve_origin (`ResolveOrigin | None`, *optional*, defaults to `None`):
            Resolves a binding's registered origin to its source frame, the
            preferred location.
        diagnostics (`LoadDiagnostics | None`, *optional*, defaults to `None`):
            Factory for located load errors.

    Raises:
        `LoadError`: A literal-text source does not fit its declared type, or a
            source's inferred type is disjoint from it.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    for output in flow_outputs:
        for path, typed in _typed_sites(output.binding):
            slot = path_label("output" if value_mode else output.name, path)
            message = _binding_error(
                typed,
                producers,
                flow_inputs,
                type_fn=type_fn,
                registry=registry,
                slot=slot,
            )
            if message is None:
                continue
            label = slot if value_mode else f"output {slot!r}"
            raise _located_error(
                f"{label}: {message}",
                typed,
                outputs_range,
                resolve_origin,
                diagnostics,
            )


def infer_binding_type(
    binding: Any,
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
    *,
    type_fn: TypeFn | None = None,
) -> Type | None:
    """Return the type one binding is proven to produce, else `None`.

    A typed binding has its declared type; a literal has its exact type (a one-tag
    `Literal(...)` for non-empty text, one-point bounds for a number); a reference or
    expression has the exact type inferred over the producer and flow-input contracts;
    a value template has the exact type inferred from its structure.

    Args:
        binding (`Any`): A wiring binding, raw or a `Binding`.
        producers (`dict[str, OutputContract]`): Node output contracts by producer id.
        flow_inputs (`dict[str, ValueContract]`): Flow input contracts by name.
        type_fn (`TypeFn | None`, *optional*, defaults to `None`): Resolves a
            constructor name to its type.

    Returns:
        `Type | None`: The proven type, or `None` when it is unproven or only partial.
    """
    if isinstance(binding, Binding) and binding.contract is not None:
        return binding.contract.type
    literal = literal_value(binding_source(binding))
    if literal is not NOT_LITERAL:
        return literal_type(literal)
    inferred = resolve_output_binding_contract(
        binding, producers, flow_inputs, type_fn=type_fn
    )
    return inferred.type if type(inferred) is ValueContract else None


def _located_error(
    message: str,
    binding: Binding,
    fallback: Optional[SourceRange],
    resolve_origin: ResolveOrigin | None,
    diagnostics: LoadDiagnostics,
) -> LoadError:
    """Build the load error, located at the binding's origin when it resolves.

    Args:
        message (`str`): The error message.
        binding (`Binding`): The rejected binding.
        fallback (`SourceRange | None`): The location used when the origin does not
            resolve.
        resolve_origin (`ResolveOrigin | None`): Resolves a registered origin to its
            source frame.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.

    Returns:
        `LoadError`: The located error.
    """
    frame = (
        resolve_origin(binding.origin)
        if resolve_origin is not None and binding.origin is not None
        else None
    )
    if frame is not None:
        return diagnostics.error(message, frames=(frame,))
    return diagnostics.error(message, range=fallback)


def item_contract(
    result: BuildResult | None,
    wiring: dict[str, Any],
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
    type_fn: TypeFn | None = None,
) -> ValueContract | None:
    """Return the exact element contract `${item}` reads on a per-item node, else `None`.

    Args:
        result (`BuildResult | None`): The built node; only a node with an item
            source key (MAP) has an item scope.
        wiring (`dict[str, Any]`): The node's binding map, holding the item source
            under that key.
        producers (`dict[str, OutputContract]`): Node output contracts by producer id.
        flow_inputs (`dict[str, ValueContract]`): Flow input contracts by name.
        type_fn (`TypeFn | None`): Resolves a constructor name to its type.

    Returns:
        `ValueContract | None`: The element contract when the item source is an
            exact list with a known element type, else `None`.
    """
    key = result.node.item_source_key if result is not None else None
    if key is None or key not in wiring:
        return None
    source = resolve_output_binding_contract(
        wiring[key], producers, flow_inputs, type_fn=type_fn
    )
    if type(source) is not ValueContract:
        return None
    if not source.type.kind.is_list() or source.type.element is None:
        return None
    return ValueContract(source.type.element)


def _binding_error(
    binding: Any,
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
    *,
    item: ValueContract | None = None,
    type_fn: TypeFn | None = None,
    registry: TypeRegistry | None = None,
    slot: str = "",
) -> str | None:
    """Return why one typed binding can never fit its declared type, else `None`.

    Args:
        binding (`Any`): A wiring or output binding; untyped ones are skipped.
        producers (`dict[str, OutputContract]`): Node output contracts by producer id.
        flow_inputs (`dict[str, ValueContract]`): Flow input contracts by name.
        item (`ValueContract | None`, *optional*, defaults to `None`): The element
            contract a lone `${item...}` reference reads, when the node binds per
            item over a proven list.
        type_fn (`TypeFn | None`, *optional*, defaults to `None`): Resolves a
            constructor name to its type.
        registry (`TypeRegistry | None`, *optional*, defaults to `None`): The flow's
            typedefs, so the message names a declared record or union by its typedef.
        slot (`str`, *optional*, defaults to `""`): The binding's slot as the
            message names it (`p`, `p.decision`), for the path of a record nested
            in the value that misses a required field.

    Returns:
        `str | None`: The error message, without the binding's name, or `None` when
            the binding may fit.
    """
    if not isinstance(binding, Binding) or binding.contract is None:
        return None
    declared = binding.contract.type
    label = named_type_label(declared, registry or {})
    literal = literal_value(binding.source)
    if literal is not NOT_LITERAL:
        try:
            build_value_as(declared, literal)
        except TypeCheckError as exc:
            message = f"declared {label}, but the value {shown_literal(literal)} does not fit"
            if exc.path:
                message += f" ({exc})"
            return message
        return None
    # Infer from the raw source: the Binding itself would resolve to its own
    # declared contract and always fit.
    def string_contract(text: Any) -> OutputContract | None:
        contract = _item_reference_contract(text, item)
        if contract is None:
            contract = resolve_output_binding_contract(
                text, producers, flow_inputs, type_fn=type_fn
            )
        return contract

    if isinstance(binding.source, ValueTemplate):
        # Exact literals, so a wrong literal under a tag is disjoint from it.
        inferred = template_contract(
            binding.source, string_contract, exact_literals=True
        )
    else:
        inferred = string_contract(binding.source)
    if type(inferred) is not ValueContract:
        return None  # unproven or partial: the runtime check covers it
    if type_relation(inferred.type, declared) != "disjoint":
        return None
    if isinstance(binding.source, ValueTemplate):
        shown = _template_label(binding.source.value, string_contract)
    else:
        shown = canonical_type_label(inferred.type)
    message = f"declared {label}, but the value is {shown}"
    found = None
    if isinstance(binding.source, ValueTemplate):
        found = _missing_in_tree(binding.source.value, declared, (), string_contract)
    if found is None:
        found = _missing_in_type(inferred.type, declared, ())
    if found is not None:
        path, missing = found
        message += " (missing required " + ", ".join(f"`{name}`" for name in missing)
        if path:
            message += f" at `{path_label(slot, path)}`"
        message += ")"
    return message


def _typed_sites(binding: Any) -> Iterator[tuple[ValuePath, Binding]]:
    """Yield each typed binding nested in a binding's value, then the binding itself.

    Nested bindings come innermost first, each with its path in the value. A nested
    binding over a container holding strings or bindings is yielded with that
    container as a value template, since the binder evaluates it; the binding itself is yielded
    only when it is typed.

    Args:
        binding (`Any`): A wiring or output binding.

    Returns:
        `Iterator[tuple[ValuePath, Binding]]`: `(path, binding)` pairs.
    """
    source = binding_source(binding)
    if isinstance(source, ValueTemplate):
        yield from _nested_typed_sites(source.value, ())
    if isinstance(binding, Binding) and binding.contract is not None:
        yield (), binding


def _nested_typed_sites(node: Any, path: ValuePath) -> Iterator[tuple[ValuePath, Binding]]:
    """Yield the typed bindings in one template tree node, innermost first."""
    if isinstance(node, Binding):
        yield from _nested_typed_sites(node.source, path)
        if node.contract is None:
            return
        source = node.source
        if type(source) in (dict, list) and _has_template_part(source):
            source = ValueTemplate(source)
        yield path, replace(node, source=source)
    elif type(node) is dict:
        for key, item in node.items():
            yield from _nested_typed_sites(item, (*path, key))
    elif type(node) is list:
        for index, item in enumerate(node):
            yield from _nested_typed_sites(item, (*path, index))


def _has_template_part(node: Any) -> bool:
    """Return whether a template tree node holds a string or a typed binding.

    Such a container is evaluated by the binder, so it is inferred as a template
    rather than checked as a literal.
    """
    if isinstance(node, (str, Binding)):
        return True
    if type(node) is dict:
        return any(_has_template_part(item) for item in node.values())
    if type(node) is list:
        return any(_has_template_part(item) for item in node)
    return False


def _template_label(
    node: Any, string_contract: Callable[[Any], OutputContract | None]
) -> str:
    """Spell a template's value for a message: literals as written, references by type.

    `{qty: 1, decision: {conviction: ${score.output}}}` gives
    `{"decision":{"conviction":float},"qty":1}`, with keys in the order type labels
    use.

    Args:
        node (`Any`): One node of the template tree.
        string_contract (`Callable[[Any], OutputContract | None]`): Infers one
            template string's contract.

    Returns:
        `str`: The spelling.
    """
    if isinstance(node, Binding):
        if node.contract is not None:
            return canonical_type_label(node.contract.type)
        return _template_label(node.source, string_contract)
    if type(node) is dict:
        fields = (
            f"{_compact_json(key)}:{_template_label(node[key], string_contract)}"
            for key in sorted(node)
        )
        return "{" + ",".join(fields) + "}"
    if type(node) is list:
        return "[" + ",".join(_template_label(item, string_contract) for item in node) + "]"
    literal = literal_value(node)
    if literal is not NOT_LITERAL:
        return shown_literal(literal)
    contract = string_contract(node)
    if type(contract) is ValueContract:
        return canonical_type_label(contract.type)
    return "object"


def _compact_json(text: str) -> str:
    """Encode one record key as compact JSON, as type labels do."""
    return json.dumps(text, ensure_ascii=False, separators=(",", ":"))


def _missing_in_tree(
    node: Any,
    declared: Type,
    path: ValuePath,
    string_contract: Callable[[Any], OutputContract | None],
) -> tuple[ValuePath, list[str]] | None:
    """Find a record in a template tree that misses required fields of its declared type.

    The walk follows the declared type through record fields and list items, so a
    record nested at any depth is named with its path in the value.

    Args:
        node (`Any`): One node of the binding's template tree.
        declared (`Type`): The type declared for this node.
        path (`ValuePath`): This node's path in the binding's value.
        string_contract (`Callable[[Any], OutputContract | None]`): Infers one
            template string's contract.

    Returns:
        `tuple[ValuePath, list[str]] | None`: The record's path and the required
            fields whose absence alone keeps it from fitting, or `None` when no
            record in the tree has that problem.
    """
    if isinstance(node, Binding):
        if node.contract is not None:
            return None  # a nested typed binding is checked on its own
        return _missing_in_tree(node.source, declared, path, string_contract)
    if type(node) is dict and declared.fields is not None:
        inferred = template_contract(
            ValueTemplate(node), string_contract, exact_literals=True
        )
        if type(inferred) is ValueContract:
            missing = _missing_required_fields(inferred.type, declared)
            if missing:
                return path, missing
        for key, item in node.items():
            if key in declared.fields:
                found = _missing_in_tree(
                    item, declared.fields[key], (*path, key), string_contract
                )
                if found is not None:
                    return found
        return None
    if type(node) is list and declared.element is not None:
        for index, item in enumerate(node):
            found = _missing_in_tree(item, declared.element, (*path, index), string_contract)
            if found is not None:
                return found
        return None
    if isinstance(node, str) and literal_value(node) is NOT_LITERAL:
        inferred = string_contract(node)
        if type(inferred) is ValueContract:
            return _missing_in_type(inferred.type, declared, path)
    return None


def _missing_in_type(
    inferred: Type, declared: Type, path: ValuePath
) -> tuple[ValuePath, list[str]] | None:
    """Find a record in an inferred type that misses required fields of the declared one.

    The walk follows record fields that both types have and that can never fit.

    Returns:
        `tuple[ValuePath, list[str]] | None`: The record's path and its missing
            required fields, or `None` when no record has that problem.
    """
    missing = _missing_required_fields(inferred, declared)
    if missing:
        return path, missing
    if inferred.fields is None or declared.fields is None:
        return None
    for name, field in declared.fields.items():
        found_type = inferred.fields.get(name)
        if found_type is None or type_relation(found_type, field) != "disjoint":
            continue
        found = _missing_in_type(found_type, field, (*path, name))
        if found is not None:
            return found
    return None


def _missing_required_fields(inferred: Type, declared: Type) -> list[str]:
    """Return the required fields that alone keep an inferred record from fitting.

    Returns:
        `list[str]`: The declared record's required fields the inferred record
            lacks, when the records would no longer be disjoint without them;
            otherwise empty, since another field is the real mismatch.
    """
    if inferred.fields is None or declared.fields is None:
        return []
    required = declared.required or frozenset()
    missing = [
        name for name in declared.fields if name in required and name not in inferred.fields
    ]
    if not missing:
        return []
    relaxed = replace(
        declared,
        fields={k: v for k, v in declared.fields.items() if k not in missing},
        required=required - set(missing),
    )
    return missing if type_relation(inferred, relaxed) != "disjoint" else []


def _item_reference_contract(
    source: Any, item: ValueContract | None
) -> OutputContract | None:
    """Return the contract a lone `${item...}` reference reads, else `None`.

    Args:
        source (`Any`): The binding's raw source.
        item (`ValueContract | None`): The node's element contract, or `None` when
            the node has no proven item scope.

    Returns:
        `OutputContract | None`: The element contract projected through the
            reference's fields, or `None` for any other source or an unknown field
            (reported by reference validation).
    """
    if item is None:
        return None
    path = direct_reference_path(source)
    if path is None:
        return None
    head, *fields = path.split(".")
    if head != "item":
        return None
    return project_output_contract(item, fields)


def shown_literal(value: Any) -> str:
    """Render a literal as YAML would write it (`true`, `null`, `"text"`, `2026-01-01`).

    Args:
        value (`Any`): The literal to render.

    Returns:
        `str`: JSON text, with each date or datetime (YAML reads an unquoted one as a
            date) written as its bare ISO text; `repr` (or a placeholder for an int
            too long to render) for anything else JSON can't write.
    """
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, list):
        return "[" + ", ".join(shown_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        fields = (f"{shown_literal(key)}: {shown_literal(item)}" for key, item in value.items())
        return "{" + ", ".join(fields) + "}"
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        # An int past Python's digit limit raises ValueError in json and repr alike.
        return value_repr(value)
