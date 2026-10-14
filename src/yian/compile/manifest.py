"""Canonical representation-neutral manifests for authored flows and built fragments.

The manifest is deliberately explicit: every execution-relevant IR field is named here,
and every known diagnostic/cache field is named as ignored. A new instance field therefore
fails loudly until its identity policy is decided, instead of silently changing program
semantics without changing the fingerprint.

Every authored value, including a literal under any name, is part of the manifest: the
manifest never guesses at secrets, since flow files are kept free of literal secrets when
they load.
"""

from __future__ import annotations

import base64
import hashlib
import struct
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any

import yian.json_codec as json_codec
from yian.compile.model import CompiledFlow, Edge, Flow, FlowOutput, validate_growth_metadata
from yian.diagnostics import OriginRef
from yian.memory.contracts import (
    DeclSite,
    MemoryDecl,
    MemoryOccurrenceSegment,
    MemoryScope,
)
from yian.nodes.agent import AgentNode
from yian.nodes.base import Node, NodeKind
from yian.nodes.call import CallNode
from yian.nodes.case import Case, CaseNode
from yian.nodes.code import CodeNode
from yian.nodes.contracts import (
    Assertion,
    Binding,
    ContractOrigins,
    ExecutionScope,
    Parameter,
    PartialRecordContract,
    ValueContract,
)
from yian.nodes.end import EndNode
from yian.nodes.human_input import HumanInputNode
from yian.nodes.loop import LoopNode
from yian.nodes.map import MapNode
from yian.nodes.start import StartNode
from yian.nodes.tool import ToolNode
from yian.nodes.wait import WaitNode
from yian.typesys.values import Type, canonical_union_order

__all__ = ["expansion_digest", "flow_fingerprint", "flow_manifest"]


# Fields common to every concrete Node. Most are emitted below; title is emitted only for
# HUMAN_INPUT because other node kinds use it as display metadata, and types only when non-empty.
# Keeping both sets explicit is the fail-loud field policy.
_COMMON_NODE_FIELDS = frozenset(
    {
        "id",
        "title",
        "output_contract",
        "params",
        "pre_asserts",
        "post_asserts",
        "commit_as",
        "origin_id",
        "env",
        "types",
    }
)
_IGNORED_NODE_FIELDS = frozenset(
    {
        "origin",
        "origin_context",
        "memory_scope",
        "execution_scope",
        "wiring_labels",
    }
)


# A kind owns both the canonical fields that affect its behavior and known noncanonical fields
# that are derived caches or render-only metadata. Unknown fields are never serialized by guess.
_KIND_FIELDS: dict[NodeKind, frozenset[str]] = {
    NodeKind.START: frozenset(),
    NodeKind.END: frozenset({"_mode", "_names", "_n"}),
    NodeKind.AGENT: frozenset(
        {
            "prompt",
            "system_prompt",
            "tools",
            "own_llm_config",
            "llm_inherit",
            "retries",
            "memories",
            "variables",
            "skills",
            "steerable",
        }
    ),
    NodeKind.CODE: frozenset({"code"}),
    NodeKind.TOOL: frozenset({"tool_id"}),
    NodeKind.CASE: frozenset({"cases"}),
    NodeKind.HUMAN_INPUT: frozenset(
        {"prompt", "answer_schema", "questions", "questions_input"}
    ),
    # WAIT's execution identity is its required flow-owned `until` wiring; the node owns
    # no additional constructor state.
    NodeKind.WAIT: frozenset(),
    NodeKind.CALL: frozenset({"flow_id", "flow_version", "child"}),
    NodeKind.MAP: frozenset({"flow_id", "flow_version", "child"}),
    NodeKind.LOOP: frozenset(
        {
            "flow_id",
            "flow_version",
            "child",
            "predicate_kind",
            "predicate",
            "times",
            "max_iters",
            "iteration",
            "returned_fields",
        }
    ),
}
_IGNORED_KIND_FIELDS: dict[NodeKind, frozenset[str]] = {
    NodeKind.START: frozenset(),
    NodeKind.END: frozenset(),
    # llm_config is always run-resolved; own_llm_config is the authored identity source.
    NodeKind.AGENT: frozenset({"llm_config"}),
    # _mode/ref are derived from code; _code is a process-local compiled-code cache.
    # CODE origins are diagnostic metadata resolved only by the authoring front end.
    NodeKind.CODE: frozenset(
        {
            "_mode",
            "ref",
            "_code",
            "_owned_code_objects",
            "code_origin",
            "code_line_origins",
        }
    ),
    # input_types is derived from wiring and producer contracts; the run-start tool check
    # reads it.
    NodeKind.TOOL: frozenset({"input_types"}),
    NodeKind.CASE: frozenset(),
    NodeKind.HUMAN_INPUT: frozenset(),
    NodeKind.WAIT: frozenset(),
    # child_inputs duplicates the recursive child's START signature.
    NodeKind.CALL: frozenset({"child_inputs"}),
    # parallel is retained by the authoring model but runtime concurrency uses num_workers.
    NodeKind.MAP: frozenset({"child_inputs", "parallel"}),
    NodeKind.LOOP: frozenset({"child_inputs"}),
}
_KIND_CLASSES: dict[NodeKind, type[Node]] = {
    NodeKind.START: StartNode,
    NodeKind.END: EndNode,
    NodeKind.AGENT: AgentNode,
    NodeKind.CODE: CodeNode,
    NodeKind.TOOL: ToolNode,
    NodeKind.CASE: CaseNode,
    NodeKind.HUMAN_INPUT: HumanInputNode,
    NodeKind.WAIT: WaitNode,
    NodeKind.CALL: CallNode,
    NodeKind.MAP: MapNode,
    NodeKind.LOOP: LoopNode,
}

_EDGE_FIELDS = frozenset(
    {"id", "from_", "to", "source_handle", "input_group", "optional", "ordering"}
)
_FLOW_OUTPUT_FIELDS = frozenset({"name", "binding"})
_BINDING_CANONICAL_FIELDS = frozenset({"source", "contract"})
_BINDING_IGNORED_FIELDS = frozenset({"origin"})
_ASSERTION_CANONICAL_FIELDS = frozenset({"expression", "display"})
_ASSERTION_IGNORED_FIELDS = frozenset({"origin"})
_PARAMETER_CANONICAL_FIELDS = frozenset(
    {"name", "contract", "required", "has_default", "default", "coerce_strings"}
)
_PARAMETER_IGNORED_FIELDS = frozenset({"origin", "default_origin"})
_VALUE_CONTRACT_CANONICAL_FIELDS = frozenset({"type"})
_VALUE_CONTRACT_IGNORED_FIELDS = frozenset({"origins"})
_PARTIAL_RECORD_CONTRACT_FIELDS = frozenset({"fields"})
_CONTRACT_ORIGINS_IGNORED_FIELDS = frozenset({"default", "entries"})
_TYPE_FIELDS = frozenset(
    {"kind", "fields", "required", "tags", "element", "nullable", "bounds", "variants"}
)
_CASE_CANONICAL_FIELDS = frozenset({"handle", "when"})
_CASE_IGNORED_FIELDS = frozenset({"origin"})
_MEMORY_DECL_CANONICAL_FIELDS = frozenset({"name", "type", "_config", "site"})
_MEMORY_DECL_IGNORED_FIELDS = frozenset({"origin"})

_MISSING = object()


class _ManifestContext:
    """Recursion stacks used to reject cycles while allowing shared immutable aliases."""

    def __init__(self) -> None:
        self.flows: set[int] = set()
        self.values: set[int] = set()
        self.types: set[int] = set()


def _field_names(value: Any) -> frozenset[str]:
    """Census instance, dataclass, attrs, Pydantic, and slotted fields without values."""
    names: set[str] = set()
    try:
        names.update(vars(value))
    except TypeError:
        pass
    model_fields = getattr(type(value), "model_fields", None)
    if model_fields is not None:
        names.update(model_fields)
        model_extra = getattr(value, "model_extra", None)
        if model_extra:
            names.update(model_extra)
        # BaseModel's slots are framework storage, not declared or extra model fields.
        return frozenset(names)
    if is_dataclass(value):
        names.update(field.name for field in fields(value))
    names.update(
        attribute.name
        for attribute in getattr(type(value), "__attrs_attrs__", ())
    )
    for cls in type(value).__mro__:
        slots = getattr(cls, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        names.update(name for name in slots if name not in {"__dict__", "__weakref__"})
    return frozenset(names)


def _check_fields(value: Any, expected: frozenset[str], path: str) -> None:
    """Require an object's instance fields to match an explicit manifest policy."""
    actual = _field_names(value)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise ValueError(f"{path}: unknown noncanonical field(s): {unknown}")
    if missing:
        raise ValueError(f"{path}: missing canonical field(s): {missing}")


def _binding_manifest_source(value: Any, path: str) -> Any:
    """Project one canonical binding to its identity-bearing source data.

    The declared contract is canonical too, but callers project it separately
    so an untyped binding keeps its source-only encoding.
    """
    if not isinstance(value, Binding):
        raise TypeError(f"{path}: expected Binding, got {type(value).__name__}")
    _check_fields(
        value,
        _BINDING_CANONICAL_FIELDS | _BINDING_IGNORED_FIELDS,
        path,
    )
    return value.source


def _binding_type_manifest(
    value: Binding,
    ctx: "_ManifestContext",
    path: str,
) -> Any:
    """Return a binding's canonical declared type, or ``_MISSING`` when untyped."""
    if value.contract is None:
        return _MISSING
    return _value_contract_manifest(value.contract, ctx, f"{path}.contract")["type"]


def _binding_manifest_mapping(value: dict, path: str) -> dict:
    """Project direct wiring carriers while retaining mapping validation order."""
    bad_key = next((key for key in value if not isinstance(key, str)), _MISSING)
    if bad_key is not _MISSING:
        raise TypeError(
            f"{path}: mapping keys must be strings, got {type(bad_key).__name__}"
        )
    return {
        key: _binding_manifest_source(item, f"{path}.values[{index}]")
        for index, (key, item) in enumerate(value.items())
    }


def _assertion_manifest(
    value: Any,
    ctx: "_ManifestContext",
    path: str,
) -> dict[str, Any]:
    """Project one canonical assertion to behavior and presentation identity."""
    if type(value) is not Assertion:
        raise TypeError(f"{path}: expected Assertion, got {type(value).__name__}")
    _check_fields(
        value,
        _ASSERTION_CANONICAL_FIELDS | _ASSERTION_IGNORED_FIELDS,
        path,
    )
    return {
        "expression": _canonical_value(value.expression, ctx, f"{path}.expression"),
        "display": _canonical_value(value.display, ctx, f"{path}.display"),
    }


def _check_node_fields(node: Node, path: str) -> None:
    """Apply the common plus per-kind strict field policy to one concrete node."""
    if not isinstance(node.kind, NodeKind):
        raise ValueError(f"{path}: node kind must be a NodeKind")
    expected_class = _KIND_CLASSES.get(node.kind)
    if expected_class is None or type(node) is not expected_class:
        expected_name = expected_class.__name__ if expected_class is not None else "<none>"
        raise ValueError(
            f"{path}: node kind {node.kind.value!r} requires {expected_name}, "
            f"got {type(node).__name__}"
        )
    expected = (
        _COMMON_NODE_FIELDS
        | _IGNORED_NODE_FIELDS
        | _KIND_FIELDS[node.kind]
        | _IGNORED_KIND_FIELDS[node.kind]
    )
    actual = _field_names(node)
    unknown = sorted(actual - expected)
    if unknown:
        raise ValueError(
            f"{path}: unknown noncanonical execution field(s) for {node.kind.value}: {unknown}"
        )
    # Conditional fields (END mode and CODE mode) are allowed to be absent. Every other field
    # is constructor-owned and must be present, or the object is not a complete canonical node.
    conditional = {"_names", "_n", "ref", "_code", "_owned_code_objects"}
    missing = sorted((expected - conditional) - actual)
    if missing:
        raise ValueError(f"{path}: missing canonical field(s): {missing}")


def _sorted_mapping(value: dict, ctx: _ManifestContext, path: str) -> dict[str, Any]:
    """Canonicalize a string-keyed mapping as sorted typed key/value entries.

    Args:
        value (`dict`): The authored mapping, such as `env:`, wiring or memory config.
        ctx (`_ManifestContext`): Recursion stacks for cycle detection.
        path (`str`): Location used in error messages.

    Returns:
        `dict[str, Any]`: The `{"entries": [...], "type": "mapping"}` record.

    Raises:
        TypeError: If a key is not a string.
    """
    bad_key = next((key for key in value if not isinstance(key, str)), _MISSING)
    if bad_key is not _MISSING:
        raise TypeError(
            f"{path}: mapping keys must be strings, got {type(bad_key).__name__}"
        )
    return _canonical_mapping(value, ctx, path)


def _canonical_sequence(value: Any, ctx: _ManifestContext, path: str) -> list[Any]:
    """Canonicalize an ordered list/tuple while preserving element order."""
    object_id = id(value)
    if object_id in ctx.values:
        raise ValueError(f"{path}: cyclic value is not manifestable")
    ctx.values.add(object_id)
    try:
        return [
            _canonical_value(item, ctx, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    finally:
        ctx.values.remove(object_id)


def _canonical_set(value: Any, ctx: _ManifestContext, path: str) -> list[Any]:
    """Canonicalize unordered members by each member's canonical JSON spelling."""
    object_id = id(value)
    if object_id in ctx.values:
        raise ValueError(f"{path}: cyclic value is not manifestable")
    ctx.values.add(object_id)
    try:
        items = [_canonical_value(item, ctx, f"{path}[]") for item in value]
        return sorted(items, key=_canonical_json)
    finally:
        ctx.values.remove(object_id)


def _canonical_mapping(value: dict, ctx: _ManifestContext, path: str) -> dict[str, Any]:
    """Canonicalize an arbitrary-key mapping as sorted typed key/value entries."""
    object_id = id(value)
    if object_id in ctx.values:
        raise ValueError(f"{path}: cyclic value is not manifestable")
    ctx.values.add(object_id)
    try:
        entries = [
            {
                "key": _canonical_value(key, ctx, f"{path}.keys[{index}]"),
                "value": _canonical_value(item, ctx, f"{path}.values[{index}]"),
            }
            for index, (key, item) in enumerate(value.items())
        ]
        entries.sort(key=_canonical_json)
        return {"entries": entries, "type": "mapping"}
    finally:
        ctx.values.remove(object_id)


def _canonical_value(value: Any, ctx: _ManifestContext, path: str) -> Any:
    """Encode one value as a deterministic, type-discriminated JSON record."""
    if isinstance(value, Enum):
        enum_type = type(value)
        return {
            "class": f"{enum_type.__module__}.{enum_type.__qualname__}",
            "type": "enum",
            "value": _canonical_value(value.value, ctx, path),
        }
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": json_codec.int_text(value)}
    if isinstance(value, float):
        return {"type": "float", "value": struct.pack("!d", value).hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "value": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, datetime):
        return {"fold": value.fold, "type": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, dict):
        return _canonical_mapping(value, ctx, path)
    if isinstance(value, list):
        return {"items": _canonical_sequence(value, ctx, path), "type": "list"}
    if isinstance(value, tuple):
        return {"items": _canonical_sequence(value, ctx, path), "type": "tuple"}
    if isinstance(value, set):
        return {"items": _canonical_set(value, ctx, path), "type": "set"}
    if isinstance(value, frozenset):
        return {"items": _canonical_set(value, ctx, path), "type": "frozenset"}
    raise TypeError(
        f"{path}: unsupported non-JSON value of type {type(value).__module__}."
        f"{type(value).__qualname__}"
    )


def _type_manifest(value: Type | None, ctx: _ManifestContext, path: str) -> Any:
    """Canonicalize one resolved structural Type, including nested fields/elements/members."""
    if value is None:
        return None
    _check_fields(value, _TYPE_FIELDS, path)
    object_id = id(value)
    if object_id in ctx.types:
        raise ValueError(f"{path}: cyclic Type is not manifestable")
    ctx.types.add(object_id)
    try:
        fields = None
        if value.fields is not None:
            fields = [
                {
                    "name": key,
                    "type": _type_manifest(field_type, ctx, f"{path}.fields.{key}"),
                }
                for key, field_type in value.fields.items()
            ]
        manifest = {
            "element": _type_manifest(value.element, ctx, f"{path}.element"),
            "fields": fields,
            "kind": value.kind.value,
            "nullable": value.nullable,
            "required": sorted(value.required) if value.required is not None else None,
            "tags": sorted(value.tags) if value.tags is not None else None,
        }
        if value.bounds is not None:
            value.validate_shape()
            manifest["bounds"] = list(value.bounds)
        if value.variants is not None:
            # Members are emitted in canonical order so authored order is not identity.
            value.validate_shape()
            manifest["union"] = [
                _type_manifest(member, ctx, f"{path}.union[{index}]")
                for index, member in enumerate(canonical_union_order(value.variants))
            ]
        return manifest
    finally:
        ctx.types.remove(object_id)


def _value_contract_manifest(
    value: ValueContract,
    ctx: _ManifestContext,
    path: str,
) -> dict[str, Any]:
    """Project one value contract while excluding every nested origin."""
    if type(value) is not ValueContract:
        raise TypeError(f"{path}: expected ValueContract, got {type(value).__name__}")
    _check_fields(
        value,
        _VALUE_CONTRACT_CANONICAL_FIELDS | _VALUE_CONTRACT_IGNORED_FIELDS,
        path,
    )
    if type(value.origins) is not ContractOrigins:
        raise TypeError(
            f"{path}.origins: expected ContractOrigins, got {type(value.origins).__name__}"
        )
    _check_fields(value.origins, _CONTRACT_ORIGINS_IGNORED_FIELDS, f"{path}.origins")
    return {"type": _type_manifest(value.type, ctx, f"{path}.type")}


def _output_contract_manifest(
    value: ValueContract | PartialRecordContract,
    ctx: _ManifestContext,
    path: str,
) -> Any:
    """Canonicalize either output-contract arm while excluding provenance."""
    if type(value) is ValueContract:
        return _value_contract_manifest(value, ctx, path)["type"]
    if type(value) is not PartialRecordContract:
        raise TypeError(
            f"{path}: expected ValueContract or PartialRecordContract, "
            f"got {type(value).__name__}"
        )
    _check_fields(value, _PARTIAL_RECORD_CONTRACT_FIELDS, path)
    value.validate_shape()
    return {
        "partial_record": [
            {
                "name": name,
                "type": (
                    None
                    if contract is None
                    else _value_contract_manifest(
                        contract,
                        ctx,
                        f"{path}.fields.{name}",
                    )["type"]
                ),
            }
            for name, contract in value.fields
        ]
    }


def _parameter_manifest(
    value: Parameter,
    ctx: _ManifestContext,
    path: str,
) -> dict[str, Any]:
    """Project one exact canonical node parameter to source-neutral identity."""
    if type(value) is not Parameter:
        raise TypeError(f"{path}: expected Parameter, got {type(value).__name__}")
    _check_fields(
        value,
        _PARAMETER_CANONICAL_FIELDS | _PARAMETER_IGNORED_FIELDS,
        path,
    )
    contract = (
        _value_contract_manifest(value.contract, ctx, f"{path}.contract")
        if value.contract is not None
        else None
    )
    return {
        "coerce_strings": value.coerce_strings,
        "contract": contract,
        "default": _canonical_value(value.default, ctx, f"{path}.default"),
        "has_default": value.has_default,
        "name": value.name,
        "required": value.required,
    }


def _case_manifest(value: Any, ctx: _ManifestContext, path: str) -> dict[str, Any]:
    """Canonicalize one ordered CASE arm."""
    if type(value) is not Case:
        raise TypeError(f"{path}: expected Case, got {type(value).__name__}")
    _check_fields(value, _CASE_CANONICAL_FIELDS | _CASE_IGNORED_FIELDS, path)
    if value.origin is not None and not isinstance(value.origin, OriginRef):
        raise TypeError(f"{path}.origin: expected OriginRef or None")
    return {
        "handle": value.handle,
        "when": _canonical_value(value.when, ctx, f"{path}.when"),
    }


def _memory_decl_manifest(
    value: MemoryDecl,
    ctx: _ManifestContext,
    path: str,
) -> dict[str, Any]:
    """Project one authored memory declaration while excluding its origin."""
    if type(value) is not MemoryDecl:
        raise TypeError(
            f"{path}: expected exact MemoryDecl, got {type(value).__name__}"
        )
    _check_fields(
        value,
        _MEMORY_DECL_CANONICAL_FIELDS | _MEMORY_DECL_IGNORED_FIELDS,
        path,
    )
    site = value.site
    if type(site) is not DeclSite:
        raise TypeError(f"{path}.site: expected exact DeclSite")
    DeclSite(site.scope, site.owner, site.node)
    for label in ("name", "type"):
        item = getattr(value, label)
        if type(item) is not str or not item:
            raise ValueError(f"{path}.{label}: expected a non-empty exact string")
    if value.origin is not None:
        if not isinstance(value.origin, OriginRef):
            raise TypeError(f"{path}.origin: expected OriginRef or None")
        OriginRef(value.origin.key)
    return {
        "config": _sorted_mapping(value.config, ctx, f"{path}.config"),
        "name": value.name,
        "site": {
            "node": site.node,
            "owner": list(site.owner),
            "scope": list(site.scope),
        },
        "type": value.type,
    }


def _memories_manifest(
    values: tuple[MemoryDecl, ...],
    ctx: _ManifestContext,
    path: str,
) -> list[dict[str, Any]]:
    """Project authored declarations in authored order."""
    if type(values) is not tuple:
        raise TypeError(f"{path}: expected a tuple of MemoryDecl values")
    return [
        _memory_decl_manifest(value, ctx, f"{path}[{index}]")
        for index, value in enumerate(values)
    ]


def _memory_occurrence_topology(
    segment: MemoryOccurrenceSegment,
    path: str,
) -> dict[str, Any]:
    """Project one dynamic occurrence segment."""
    if type(segment) is not MemoryOccurrenceSegment:
        raise TypeError(
            f"{path}: expected exact MemoryOccurrenceSegment, "
            f"got {type(segment).__name__}"
        )
    MemoryOccurrenceSegment(
        kind=segment.kind,
        static_path=segment.static_path,
        index=segment.index,
    )
    return {
        "index": segment.index,
        "kind": segment.kind.value,
        "static_path": list(segment.static_path),
    }


def _memory_scope_topology(value: MemoryScope, path: str) -> dict[str, Any]:
    """Project the memory topology needed for deterministic growth identity."""
    if type(value) is not MemoryScope:
        raise TypeError(
            f"{path}: expected exact MemoryScope, got {type(value).__name__}"
        )
    # Replay the frozen carrier's public invariants in case an object was forged
    # or mutated through ``object.__setattr__``.
    MemoryScope(
        static_path=value.static_path,
        memories=value.memories,
        occurrence_path=value.occurrence_path,
    )
    return {
        "memories": list(value.memories),
        "occurrence_path": [
            _memory_occurrence_topology(segment, f"{path}.occurrence_path[{index}]")
            for index, segment in enumerate(value.occurrence_path)
        ],
        "static_path": list(value.static_path),
    }


def _node_manifest(
    node: Node,
    ctx: _ManifestContext,
    path: str,
    *,
    topology_only: bool,
) -> dict[str, Any]:
    """Canonicalize one concrete node under its exhaustive per-kind field policy."""
    _check_node_fields(node, path)
    params = None
    if node.params is not None:
        if type(node.params) is not list:
            raise TypeError(
                f"{path}.params: expected list, got {type(node.params).__name__}"
            )
        params = [
            _parameter_manifest(param, ctx, f"{path}.params[{index}]")
            for index, param in enumerate(node.params)
        ]
    output_type = None
    if node.output_contract is not None:
        output_type = _output_contract_manifest(
            node.output_contract,
            ctx,
            f"{path}.output_contract",
        )
    manifest: dict[str, Any] = {
        "commit_as": _canonical_value(node.commit_as, ctx, f"{path}.commit_as"),
        "env": _sorted_mapping(node.env, ctx, f"{path}.env"),
        "id": node.id,
        "kind": node.kind.value,
        "origin_id": _canonical_value(node.origin_id, ctx, f"{path}.origin_id"),
        # Preserve the pre-contract wire shape: the carrier is canonical in memory,
        # while its diagnostic origins remain outside program identity.
        "output_type": output_type,
        "params": params,
        "post_asserts": [
            _assertion_manifest(assertion, ctx, f"{path}.post_asserts[{index}]")
            for index, assertion in enumerate(node.post_asserts)
        ],
        "pre_asserts": [
            _assertion_manifest(assertion, ctx, f"{path}.pre_asserts[{index}]")
            for index, assertion in enumerate(node.pre_asserts)
        ],
    }

    # Emitted only when non-empty so nodes without `isinstance` keep their fingerprints.
    if type(node.types) is not dict:
        raise TypeError(f"{path}.types: expected dict, got {type(node.types).__name__}")
    for name, typ in node.types.items():
        if not isinstance(name, str):
            raise TypeError(f"{path}.types: keys must be strings, got {type(name).__name__}")
        if type(typ) is not Type:
            raise TypeError(f"{path}.types.{name}: expected Type, got {type(typ).__name__}")
    if node.types:
        manifest["types"] = [
            {"name": name, "type": _type_manifest(node.types[name], ctx, f"{path}.types.{name}")}
            for name in sorted(node.types)
        ]

    if node.kind is NodeKind.START:
        pass
    elif node.kind is NodeKind.END:
        # A VALUE terminal keeps the encoding a one-input (or input-free) record END had, so
        # flows written `output: ${x}` or with no output keep their fingerprints.
        manifest["mode"] = "list" if node._mode == "list" else "record"
        manifest["names"] = (
            _canonical_sequence(node._names, ctx, f"{path}.names")
            if node._mode != "list"
            else None
        )
        if node._mode == "record" and len(node._names) <= 1:
            # A zero- or one-field record would otherwise encode like a value END with the same
            # names, though it returns a dict; `keyed` separates them while value and multi-field
            # ENDs keep their fingerprints.
            manifest["keyed"] = True
        manifest["n"] = node._n if node._mode == "list" else None
    elif node.kind is NodeKind.AGENT:
        if type(node.variables) is not list:
            raise TypeError(f"{path}.variables: expected list")
        if node.variables:
            manifest["variables"] = [
                _parameter_manifest(variable, ctx, f"{path}.variables[{index}]")
                for index, variable in enumerate(node.variables)
            ]
        # Omitted when empty so flows without skills keep their identity. Grant order
        # never changes behavior (the listing is sorted), so it is not identity either.
        if node.skills:
            manifest["skills"] = "all" if node.skills == "all" else sorted(node.skills)
        # Omitted when false so flows that never opt in keep their identity.
        if node.steerable:
            manifest["steerable"] = True
        # Omitted when unset so flows on the default system prompt keep their identity.
        if node.system_prompt is not None:
            manifest["system_prompt"] = node.system_prompt
        memories = _memories_manifest(node.memories, ctx, f"{path}.memories")
        manifest.update(
            {
                "llm_inherit": node.llm_inherit,
                "prompt": node.prompt,
                "retries": node.retries,
                "tools": _canonical_sequence(node.tools, ctx, f"{path}.tools"),
                "own_llm_config": _sorted_mapping(
                    node.own_llm_config, ctx, f"{path}.own_llm_config"
                ),
            }
        )
        # Omitted when empty so memory-free flows keep their identity.
        if not topology_only and memories:
            manifest["memories"] = memories
    elif node.kind is NodeKind.CODE:
        manifest["code"] = node.code
    elif node.kind is NodeKind.TOOL:
        manifest["tool_id"] = node.tool_id
    elif node.kind is NodeKind.CASE:
        manifest["cases"] = [
            _case_manifest(case, ctx, f"{path}.cases[{index}]")
            for index, case in enumerate(node.cases)
        ]
    elif node.kind is NodeKind.HUMAN_INPUT:
        manifest.update(
            {
                "answer_schema": _canonical_value(
                    node.answer_schema, ctx, f"{path}.answer_schema"
                ),
                "prompt": _canonical_value(node.prompt, ctx, f"{path}.prompt"),
                "questions": _canonical_value(node.questions, ctx, f"{path}.questions"),
                "questions_input": _canonical_value(
                    node.questions_input, ctx, f"{path}.questions_input"
                ),
                "title": _canonical_value(node.title, ctx, f"{path}.title"),
            }
        )
    elif node.kind is NodeKind.WAIT:
        # WAIT owns no constructor state; its absolute deadline is canonicalized through
        # the ordinary flow `wiring` manifest.
        pass
    elif node.kind in (NodeKind.CALL, NodeKind.MAP, NodeKind.LOOP):
        manifest.update(
            {
                "child": _flow_manifest(
                    node.child,
                    ctx,
                    f"{path}.child",
                    topology_only=topology_only,
                ),
                "flow_id": node.flow_id,
                "flow_version": _canonical_value(
                    node.flow_version, ctx, f"{path}.flow_version"
                ),
            }
        )
        if node.kind is NodeKind.LOOP:
            manifest.update(
                {
                    "iteration": node.iteration,
                    "max_iters": node.max_iters,
                    "predicate": _canonical_value(node.predicate, ctx, f"{path}.predicate"),
                    "predicate_kind": node.predicate_kind,
                    "times": _canonical_value(node.times, ctx, f"{path}.times"),
                }
            )
            # Omitted for a total loop so its fingerprint and expansion digests stay unchanged.
            if node.returned_fields is not None:
                manifest["returned_fields"] = list(node.returned_fields)
    else:  # pragma: no cover - _check_node_fields exhausts the closed enum first
        raise ValueError(f"{path}: unsupported node kind {node.kind.value!r}")

    # Derived memory state is never authored program identity. Dynamic fragment
    # identity includes the scope's root-relative path, key stack and
    # occurrence path, so replay rejects a structurally different memory
    # occurrence without persisting configs or source origins.
    if topology_only and node.memory_scope is not None:
        manifest["memory_scope"] = _memory_scope_topology(
            node.memory_scope,
            f"{path}.memory_scope",
        )
    if topology_only and node.execution_scope is not None:
        scope = node.execution_scope
        if type(scope) is not ExecutionScope:
            raise TypeError(f"{path}.execution_scope must be an ExecutionScope")
        ExecutionScope(scope.static_path, scope.flow_path, scope.occurrence_path, scope.boundary)
        manifest["execution_scope"] = {
            "static_path": list(scope.static_path),
            "flow_path": list(scope.flow_path),
            "occurrence_path": [
                {"kind": position.kind, "static_path": list(position.static_path), "index": position.index}
                for position in scope.occurrence_path
            ],
            "boundary": scope.boundary,
        }

    return {key: manifest[key] for key in sorted(manifest)}


def _edge_manifest(edge: Edge, ctx: _ManifestContext, path: str) -> dict[str, Any]:
    """Canonicalize one edge; callers preserve the flow's semantic edge order."""
    if type(edge) is not Edge:
        raise TypeError(f"{path}: expected Edge, got {type(edge).__name__}")
    _check_fields(edge, _EDGE_FIELDS, path)
    return {
        "from": edge.from_,
        "id": edge.id,
        "input_group": _canonical_value(edge.input_group, ctx, f"{path}.input_group"),
        "optional": edge.optional,
        "ordering": edge.ordering,
        "source_handle": _canonical_value(edge.source_handle, ctx, f"{path}.source_handle"),
        "to": edge.to,
    }


def _output_manifest(output: FlowOutput, ctx: _ManifestContext, path: str) -> dict[str, Any]:
    """Canonicalize one declared flow output; callers preserve declaration order."""
    if type(output) is not FlowOutput:
        raise TypeError(f"{path}: expected FlowOutput, got {type(output).__name__}")
    _check_fields(output, _FLOW_OUTPUT_FIELDS, path)
    source = _binding_manifest_source(output.binding, f"{path}.binding")
    manifest = {
        "from": _canonical_value(source, ctx, f"{path}.binding.source"),
        "name": output.name,
    }
    # Only a typed output adds a key, so untyped outputs keep their identity.
    output_type = _binding_type_manifest(output.binding, ctx, f"{path}.binding")
    if output_type is not _MISSING:
        manifest["type"] = output_type
    return manifest


def _flow_manifest(
    flow: Flow,
    ctx: _ManifestContext,
    path: str,
    *,
    topology_only: bool,
) -> dict[str, Any]:
    """Canonicalize a Flow recursively, optionally omitting CompiledFlow-only root fields."""
    if not isinstance(flow, Flow):
        raise TypeError(f"{path}: expected Flow, got {type(flow).__name__}")
    object_id = id(flow)
    if object_id in ctx.flows:
        raise ValueError(f"{path}: recursive child-flow cycle is not manifestable")
    ctx.flows.add(object_id)
    try:
        if type(flow) is CompiledFlow:
            expected = frozenset(
                {
                    "nodes",
                    "edges",
                    "wiring",
                    "start_id",
                    "end_id",
                    "input_connections",
                    "child_boundaries",
                    "outputs",
                    "flow_llm_config",
                    "flow_id",
                    "memories",
                    "_incoming",
                    "_outgoing",
                }
            )
        elif type(flow) is Flow:
            expected = frozenset({"nodes", "edges", "wiring", "start_id", "end_id",
                                  "input_connections", "child_boundaries"})
        else:
            raise TypeError(f"{path}: unsupported Flow subclass {type(flow).__name__}")
        _check_fields(flow, expected, path)
        validate_growth_metadata(flow)

        nodes: dict[str, Any] = {}
        bad_node_id = next(
            (node_id for node_id in flow.nodes if not isinstance(node_id, str)), _MISSING
        )
        if bad_node_id is not _MISSING:
            raise TypeError(
                f"{path}.nodes: keys must be strings, got {type(bad_node_id).__name__}"
            )
        for node_id in sorted(flow.nodes):
            nodes[node_id] = _node_manifest(
                flow.nodes[node_id],
                ctx,
                f"{path}.nodes.{node_id}",
                topology_only=topology_only,
            )

        wiring: dict[str, Any] = {}
        binding_types: dict[str, dict[str, Any]] = {}
        bad_wiring_id = next(
            (node_id for node_id in flow.wiring if not isinstance(node_id, str)), _MISSING
        )
        if bad_wiring_id is not _MISSING:
            raise TypeError(
                f"{path}.wiring: keys must be strings, got {type(bad_wiring_id).__name__}"
            )
        for node_id in sorted(flow.wiring):
            node_wiring = flow.wiring[node_id]
            if not isinstance(node_wiring, dict):
                raise TypeError(
                    f"{path}.wiring.{node_id}: expected dict, got {type(node_wiring).__name__}"
                )
            binding_sources = _binding_manifest_mapping(
                node_wiring,
                f"{path}.wiring.{node_id}",
            )
            wiring[node_id] = _sorted_mapping(
                binding_sources, ctx, f"{path}.wiring.{node_id}"
            )
            node_types = {}
            for param in sorted(node_wiring):
                param_type = _binding_type_manifest(
                    node_wiring[param],
                    ctx,
                    f"{path}.wiring.{node_id}.{param}",
                )
                if param_type is not _MISSING:
                    node_types[param] = param_type
            if node_types:
                binding_types[node_id] = node_types

        manifest: dict[str, Any] = {
            "edges": [
                _edge_manifest(edge, ctx, f"{path}.edges[{index}]")
                for index, edge in enumerate(flow.edges)
            ],
            "end_id": flow.end_id,
            "nodes": nodes,
            "start_id": flow.start_id,
            "wiring": wiring,
        }
        # Declared binding types live beside the wiring rather than inside it, so
        # every untyped flow keeps its existing manifest byte for byte.
        if binding_types:
            manifest["binding_types"] = binding_types
        if topology_only and flow.input_connections:
            manifest["input_connections"] = [
                {"child_start_id": connection.child_start_id, "input_name": connection.input_name,
                 "seed_path": list(connection.seed_path)}
                for connection in sorted(flow.input_connections,
                                         key=lambda item: (item.child_start_id, item.input_name))
            ]
        if topology_only and flow.child_boundaries:
            manifest["child_boundaries"] = [
                {"start_id": boundary.start_id, "end_id": boundary.end_id}
                for boundary in sorted(flow.child_boundaries, key=lambda item: (item.start_id, item.end_id))
            ]
        if type(flow) is CompiledFlow and not topology_only:
            if flow.flow_id is not None:
                if type(flow.flow_id) is not str or not flow.flow_id:
                    raise ValueError(
                        f"{path}.flow_id: expected a non-empty exact string or None"
                    )
                manifest["flow_id"] = flow.flow_id
            if flow.memories:
                manifest["memories"] = _memories_manifest(
                    flow.memories, ctx, f"{path}.memories"
                )
            manifest["flow_llm_config"] = _sorted_mapping(
                flow.flow_llm_config, ctx, f"{path}.flow_llm_config"
            )
            manifest["outputs"] = [
                _output_manifest(output, ctx, f"{path}.outputs[{index}]")
                for index, output in enumerate(flow.outputs)
            ]
        elif type(flow) is CompiledFlow:
            if flow.flow_id is not None and (
                type(flow.flow_id) is not str or not flow.flow_id
            ):
                raise ValueError(
                    f"{path}.flow_id: expected a non-empty exact string or None"
                )
            _memories_manifest(flow.memories, ctx, f"{path}.memories")
        return {key: manifest[key] for key in sorted(manifest)}
    finally:
        ctx.flows.remove(object_id)


def _canonical_json(value: Any) -> str:
    """Encode already-canonical primitives in one byte-stable JSON spelling."""
    return json_codec.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(value: Any) -> str:
    """Return lowercase SHA-256 over a canonical JSON value."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def flow_manifest(flow: Flow) -> dict[str, Any]:
    """Return the canonical recursive build manifest for a representation-neutral flow.

    Args:
        flow: A compiled authored flow or shared Flow topology carrier.

    Returns:
        A JSON-safe dictionary with sorted maps and semantic list order preserved.

    Raises:
        TypeError: If the graph contains a noncanonical class or unsupported value.
        ValueError: If a node/flow has an unknown field or recursive object cycle.
    """
    return _flow_manifest(flow, _ManifestContext(), "flow", topology_only=False)


def flow_fingerprint(flow: Flow) -> str:
    """Return lowercase SHA-256 over the canonical recursive authored-flow manifest."""
    return _digest(flow_manifest(flow))


def expansion_digest(flow_fragment: Flow) -> str:
    """Return SHA-256 over a built fragment's canonical topology manifest.

    The topology identity contains the entry and terminal ids, sorted node and wiring maps,
    every build-relevant node field (including recursive baked children), and edges in their
    semantic order. CompiledFlow-only root outputs/LLM layers are intentionally not part of a
    dynamic fragment digest.

    Args:
        flow_fragment: The self-contained Flow returned by a deterministic growth builder.

    Returns:
        A lowercase 64-character SHA-256 hexadecimal digest.
    """
    manifest = _flow_manifest(
        flow_fragment,
        _ManifestContext(),
        "expansion",
        topology_only=True,
    )
    return _digest(manifest)
