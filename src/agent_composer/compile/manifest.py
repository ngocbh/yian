"""Canonical representation-neutral manifests for authored flows and built fragments.

The manifest is deliberately explicit: every execution-relevant IR field is named here,
and every known diagnostic/cache field is named as ignored. A new instance field therefore
fails loudly until its identity policy is decided, instead of silently changing program
semantics without changing the fingerprint.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import struct
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any
from urllib.parse import unquote_plus

from lark import Token, Tree

from agent_composer.compile.model import CompiledFlow, Edge, Flow, FlowOutput
from agent_composer.expr import ExpressionError, expr_refs, scan_template
from agent_composer.expr.template import Literal, Span
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.base import Node, NodeKind
from agent_composer.nodes.call import CallNode
from agent_composer.nodes.case import Case, CaseNode
from agent_composer.nodes.code import CodeNode
from agent_composer.nodes.contracts import (
    Assertion,
    Binding,
    ContractOrigins,
    Parameter,
    ValueContract,
)
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.human_input import HumanInputNode
from agent_composer.nodes.loop import LoopNode
from agent_composer.nodes.map import MapNode
from agent_composer.nodes.model import ModelNode
from agent_composer.nodes.start import StartNode
from agent_composer.nodes.tool import ToolNode
from agent_composer.nodes.wait import WaitNode
from agent_composer.typesys.values import Type

__all__ = ["expansion_digest", "flow_fingerprint", "flow_manifest"]


# Fields common to every concrete Node. Most are emitted below; title is emitted only for
# HUMAN_INPUT because other node kinds use it as display metadata. Keeping both sets explicit is
# the fail-loud field policy.
_COMMON_NODE_FIELDS = frozenset(
    {
        "id",
        "title",
        "output_type",
        "params",
        "pre_asserts",
        "post_asserts",
        "commit_as",
        "origin_id",
        "env",
    }
)
_IGNORED_NODE_FIELDS = frozenset(
    {
        "range",
        "field_ranges",
        "wiring_labels",
        "pre_assert_ranges",
        "post_assert_ranges",
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
            "tools",
            "own_llm_config",
            "llm_inherit",
            "mode",
            "retries",
        }
    ),
    NodeKind.CODE: frozenset({"code"}),
    NodeKind.MODEL: frozenset({"model_id", "weights_uri", "runtime_name"}),
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
        }
    ),
}
_IGNORED_KIND_FIELDS: dict[NodeKind, frozenset[str]] = {
    NodeKind.START: frozenset(),
    NodeKind.END: frozenset(),
    # llm_config is always run-resolved; own_llm_config is the authored identity source.
    NodeKind.AGENT: frozenset({"llm_config"}),
    # _mode/ref are derived from code; _code is a process-local compiled-code cache.
    NodeKind.CODE: frozenset({"_mode", "ref", "_code"}),
    NodeKind.MODEL: frozenset(),
    NodeKind.TOOL: frozenset(),
    NodeKind.CASE: frozenset(),
    NodeKind.HUMAN_INPUT: frozenset(),
    NodeKind.WAIT: frozenset(),
    # child_inputs duplicates the recursive child's START signature; child_source is diagnostic.
    NodeKind.CALL: frozenset({"child_inputs", "child_source"}),
    # parallel is retained by the authoring model but runtime concurrency uses num_workers.
    NodeKind.MAP: frozenset({"child_inputs", "child_source", "parallel"}),
    NodeKind.LOOP: frozenset({"child_inputs", "child_source"}),
}
_KIND_CLASSES: dict[NodeKind, type[Node]] = {
    NodeKind.START: StartNode,
    NodeKind.END: EndNode,
    NodeKind.AGENT: AgentNode,
    NodeKind.CODE: CodeNode,
    NodeKind.MODEL: ModelNode,
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
_BINDING_CANONICAL_FIELDS = frozenset({"source"})
_BINDING_IGNORED_FIELDS = frozenset({"origin"})
_ASSERTION_CANONICAL_FIELDS = frozenset({"expression", "display"})
_ASSERTION_IGNORED_FIELDS = frozenset({"origin"})
_PARAMETER_CANONICAL_FIELDS = frozenset(
    {"name", "contract", "required", "has_default", "default", "coerce_strings"}
)
_PARAMETER_IGNORED_FIELDS = frozenset({"origin", "default_origin"})
_VALUE_CONTRACT_CANONICAL_FIELDS = frozenset({"type"})
_VALUE_CONTRACT_IGNORED_FIELDS = frozenset({"origins"})
_CONTRACT_ORIGINS_IGNORED_FIELDS = frozenset({"default", "entries"})
_TYPE_FIELDS = frozenset({"kind", "fields", "required", "tags", "element", "nullable"})
_CASE_FIELDS = frozenset({"handle", "when"})

# Config secrets are intentionally absent from identity. Exact names and high-confidence
# compound suffixes are credentials in every config spelling. Generic auth/token/signature
# suffixes are credentials in flat config only when authored like environment variables; URI
# parameters use the stricter suffix policy regardless of case.
_CREDENTIAL_NAMES = frozenset(
    {
        "api_key",
        "access_token",
        "auth",
        "auth_token",
        "authorization",
        "basic_auth",
        "bearer_token",
        "client_secret",
        "connection_string",
        "credential",
        "credentials",
        "databricks_token",
        "docker_auth_config",
        "google_application_credentials",
        "github_token",
        "hf_token",
        "id_token",
        "mysql_pwd",
        "npm_token",
        "password",
        "passphrase",
        "pgpassword",
        "private_key",
        "proxy_auth",
        "refresh_token",
        "secret",
        "secret_access_key",
        "secret_key",
        "session_token",
        "shared_access_key",
        "shared_access_signature",
        "sig",
        "signature",
        "slack_bot_token",
        "sslpassword",
        "token",
        "webhook_signature",
        "webhook_url",
        "x_amz_security_token",
        "x_amz_signature",
        "x_goog_signature",
    }
)
_HIGH_CONFIDENCE_CREDENTIAL_SUFFIXES = (
    "_api_key",
    "_api_token",
    "_account_key",
    "_access_token",
    "_access_key",
    "_access_key_id",
    "_auth_token",
    "_authorization",
    "_bearer_token",
    "_credential",
    "_credentials",
    "_application_credentials",
    "_connection_string",
    "_password",
    "_passphrase",
    "_private_key",
    "_refresh_token",
    "_secret",
    "_secret_access_key",
    "_secret_key",
    "_session_cookie",
    "_session_token",
    "_shared_access_key",
    "_shared_access_signature",
    "_subscription_key",
    "_webhook_url",
)
_AMBIGUOUS_CREDENTIAL_SUFFIXES = ("_auth", "_signature", "_token")
_CREDENTIAL_COMPACT_NAMES = frozenset(
    name.replace("_", "") for name in _CREDENTIAL_NAMES
)
_ENVIRONMENT_VARIABLE_KEY = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
_URI_CREDENTIAL_PARAM_NAMES = frozenset({"key", "user", "user_name", "username"})
_CONNECTION_PROPERTY_CREDENTIAL_NAMES = frozenset(
    {
        "pwd",
        "uid",
        "user",
        "user_id",
        "user_name",
        "username",
    }
)
_DSN_IDENTITY_PROPERTY_NAMES = frozenset(
    {
        "account_endpoint",
        "account_name",
        "addr",
        "address",
        "database",
        "database_name",
        "data_source",
        "dbname",
        "default_endpoints_protocol",
        "driver",
        "endpoint",
        "endpoint_suffix",
        "entity_path",
        "host",
        "hostname",
        "initial_catalog",
        "network_address",
        "port",
        "server",
        "service",
        "shared_access_key_name",
        "sid",
        "sslmode",
        "transport_type",
    }
)
_URI_PREFIX = re.compile(r"^\s*(?:[A-Za-z][A-Za-z0-9+.-]*:)+//")
_JDBC_SQLSERVER_PREFIX = re.compile(r"^\s*jdbc:sqlserver://", re.IGNORECASE)
_JDBC_ORACLE_THIN_PREFIX = re.compile(r"^\s*jdbc:oracle:thin:", re.IGNORECASE)
_REDACTED_URI_CREDENTIAL = "__redacted__"
_REDACTED_BINDING_PREFIX = "__redacted_binding__:"
_SAFE_HTTP_AUTH_SCHEMES = frozenset({"basic", "bearer", "digest", "negotiate"})
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
    """Project one canonical binding to identity-bearing source data only."""
    if not isinstance(value, Binding):
        raise TypeError(f"{path}: expected Binding, got {type(value).__name__}")
    _check_fields(
        value,
        _BINDING_CANONICAL_FIELDS | _BINDING_IGNORED_FIELDS,
        path,
    )
    return value.source


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
    conditional = {"_names", "_n", "ref", "_code"}
    missing = sorted((expected - conditional) - actual)
    if missing:
        raise ValueError(f"{path}: missing canonical field(s): {missing}")


def _normalized_key_spellings(key: str) -> tuple[str, str]:
    """Return separator-preserving and compact forms of one authored key."""
    separated = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key.strip())
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", separated)
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", separated).strip("_").lower()
    return normalized, normalized.replace("_", "")


def _matches_suffixes(
    normalized: str,
    suffixes: tuple[str, ...],
) -> bool:
    """Whether a separator-preserving key spelling ends in one named suffix."""
    return any(
        normalized == suffix[1:] or normalized.endswith(suffix)
        for suffix in suffixes
    )


def _matches_known_credential(normalized: str, compact: str) -> bool:
    """Whether normalized spellings match exact or high-confidence credential names."""
    return (
        normalized in _CREDENTIAL_NAMES
        or compact in _CREDENTIAL_COMPACT_NAMES
        or _matches_suffixes(
            normalized,
            _HIGH_CONFIDENCE_CREDENTIAL_SUFFIXES,
        )
    )


def _matches_ambiguous_credential_suffix(normalized: str) -> bool:
    """Whether a normalized spelling ends in generic auth/token/signature vocabulary."""
    return _matches_suffixes(
        normalized,
        _AMBIGUOUS_CREDENTIAL_SUFFIXES,
    )


def _credential_key(key: str) -> bool:
    """Whether a flat config key carries a credential rather than program semantics."""
    normalized, compact = _normalized_key_spellings(key)
    return _matches_known_credential(normalized, compact) or (
        _ENVIRONMENT_VARIABLE_KEY.fullmatch(key.strip()) is not None
        and _matches_ambiguous_credential_suffix(normalized)
    )


def _connection_string_key(key: str) -> bool:
    """Whether a key explicitly identifies a URI or DSN connection string."""
    normalized, compact = _normalized_key_spellings(key)
    return (
        normalized == "connection_string"
        or normalized.endswith("_connection_string")
        or compact == "connectionstring"
    )


def _uri_credential_key(key: str) -> bool:
    """Whether a URI parameter carries a credential, including the URI-specific `key`."""
    normalized, compact = _normalized_key_spellings(key)
    return (
        normalized in _URI_CREDENTIAL_PARAM_NAMES
        or compact in _URI_CREDENTIAL_PARAM_NAMES
        or _matches_known_credential(normalized, compact)
        or _matches_ambiguous_credential_suffix(normalized)
    )


def _connection_property_credential_key(key: str) -> bool:
    """Whether a URI/DSN property identifies connection authentication material."""
    normalized, compact = _normalized_key_spellings(key)
    return (
        normalized in _CONNECTION_PROPERTY_CREDENTIAL_NAMES
        or _matches_known_credential(normalized, compact)
    )


def _redacted_expression_shape(node: Tree | Token) -> dict[str, Any]:
    """Encode expression structure without retaining literal operand values."""
    if isinstance(node, Token):
        token_type = str(node.type)
        shape: dict[str, Any] = {"token": token_type}
        if token_type in {"NAME", "COMP_OP"}:
            shape["value"] = str(node)
        return shape
    return {
        "children": [_redacted_expression_shape(child) for child in node.children],
        "rule": str(node.data),
    }


def _credential_binding_marker(
    value: str,
    *,
    preserve_http_auth_scheme: bool = False,
) -> str | None:
    """Return a secret-free identity marker for a credential template with references.

    Reference paths and expression structure affect execution and therefore identity.
    Literal template runs and literal operands contribute only their shape. The sole
    exception is an exact, known HTTP scheme in an authorization slot: that vocabulary
    is non-secret and changes how the dynamic credential is interpreted.
    """
    try:
        segments = scan_template(value)
    except ExpressionError:
        return None

    references: list[str] = []
    shape: list[dict[str, Any]] = []
    for segment in segments:
        if isinstance(segment, Literal):
            literal_shape = {"segment": "literal"}
            scheme = segment.text.strip().casefold()
            if preserve_http_auth_scheme and scheme in _SAFE_HTTP_AUTH_SCHEMES:
                literal_shape["http_auth_scheme"] = scheme
            shape.append(literal_shape)
            continue
        if not isinstance(segment, Span):  # pragma: no cover - closed template sum
            raise TypeError(f"unsupported template segment {type(segment).__name__}")
        references.extend(expr_refs(segment.tree))
        shape.append(
            {
                "expression": _redacted_expression_shape(segment.tree),
                "segment": "expression",
            }
        )
    if not references:
        return None
    identity = {"references": references, "shape": shape}
    return f"{_REDACTED_BINDING_PREFIX}{_digest(identity)}"


def _redacted_credential_value(
    value: str,
    *,
    preserve_binding_identity: bool = False,
) -> str:
    """Redact a credential, retaining template identity only at the wiring boundary."""
    if preserve_binding_identity:
        marker = _credential_binding_marker(value)
        if marker is not None:
            return marker
    return _REDACTED_URI_CREDENTIAL


def _redact_uri_params(
    value: str,
    *,
    preserve_binding_identity: bool = False,
) -> str:
    """Replace credential parameter values while preserving spelling and ordering."""
    redacted_parts: list[str] = []
    for part in value.split("&"):
        raw_key, separator, raw_value = part.partition("=")
        if separator and _uri_credential_key(unquote_plus(raw_key)):
            replacement = _redacted_credential_value(
                raw_value,
                preserve_binding_identity=preserve_binding_identity,
            )
            part = f"{raw_key}={replacement}"
        redacted_parts.append(part)
    return "&".join(redacted_parts)


def _redact_uri_tail_params(
    value: str,
    *,
    preserve_binding_identity: bool = False,
) -> str:
    """Redact credential query/fragment parameters in a URI-like tail."""
    fragment_start = value.find("#")
    if fragment_start == -1:
        before_fragment = value
        fragment = None
    else:
        before_fragment = value[:fragment_start]
        fragment = value[fragment_start + 1 :]

    query_start = before_fragment.find("?")
    if query_start != -1:
        before_fragment = (
            before_fragment[: query_start + 1]
            + _redact_uri_params(
                before_fragment[query_start + 1 :],
                preserve_binding_identity=preserve_binding_identity,
            )
        )
    redacted_tail = before_fragment
    if fragment is not None:
        redacted_tail += "#" + _redact_uri_params(
            fragment,
            preserve_binding_identity=preserve_binding_identity,
        )
    return redacted_tail


def _redact_hierarchical_uri(
    value: str,
    *,
    preserve_binding_identity: bool = False,
) -> str | None:
    """Sanitize a hierarchical URI while retaining its endpoint and path spelling."""
    prefix = _URI_PREFIX.match(value)
    if prefix is None:
        return None

    authority_start = prefix.end()
    delimiters = (
        index
        for delimiter in "/?#"
        if (index := value.find(delimiter, authority_start)) != -1
    )
    authority_end = min(delimiters, default=len(value))
    authority = value[authority_start:authority_end]
    if "@" in authority:
        userinfo, host = authority.rsplit("@", 1)
        replacement = _redacted_credential_value(
            userinfo,
            preserve_binding_identity=preserve_binding_identity,
        )
        authority = f"{replacement}@{host}"
    tail = _redact_uri_tail_params(
        value[authority_end:],
        preserve_binding_identity=preserve_binding_identity,
    )
    return value[:authority_start] + authority + tail


def _semicolon_property_spans(
    value: str, start: int
) -> list[tuple[int, int, int, str]] | None:
    """Parse semicolon properties, honoring braced and quoted delimiter values."""
    spans: list[tuple[int, int, int, str]] = []
    cursor = start
    while cursor < len(value):
        delimiter = value.find(";", cursor)
        equal = value.find("=", cursor)
        if equal == -1 or (delimiter != -1 and delimiter < equal):
            return None
        key = value[cursor:equal].strip()
        if not key:
            return None

        scan = equal + 1
        while scan < len(value) and value[scan].isspace():
            scan += 1
        if scan < len(value) and value[scan] == "{":
            scan += 1
            while scan < len(value):
                if value[scan] != "}":
                    scan += 1
                    continue
                if scan + 1 < len(value) and value[scan + 1] == "}":
                    scan += 2
                    continue
                scan += 1
                break
            else:
                return None
            while scan < len(value) and value[scan].isspace():
                scan += 1
            if scan < len(value) and value[scan] != ";":
                return None
            end = scan
        elif scan < len(value) and value[scan] in {"'", '"'}:
            quote = value[scan]
            scan += 1
            while scan < len(value):
                if value[scan] != quote:
                    scan += 1
                    continue
                if scan + 1 < len(value) and value[scan + 1] == quote:
                    scan += 2
                    continue
                scan += 1
                break
            else:
                return None
            while scan < len(value) and value[scan].isspace():
                scan += 1
            if scan < len(value) and value[scan] != ";":
                return None
            end = scan
        else:
            end = delimiter if delimiter != -1 else len(value)
            unquoted_value = value[scan:end]
            if re.search(r"\s+[A-Za-z_][A-Za-z0-9_]*\s*=", unquoted_value):
                # Whitespace-separated assignments belong to libpq conninfo. Without
                # this check, its first semicolon is mistaken for a DSN delimiter and
                # credential fields remain buried inside the first property value.
                return None

        spans.append((cursor, end, equal, key))
        if end == len(value):
            break
        cursor = end + 1
        if cursor == len(value):
            break
    return spans


def _redact_semicolon_properties(
    value: str,
    start: int,
    *,
    preserve_binding_identity: bool = False,
) -> tuple[str, frozenset[str]] | None:
    """Redact credential fields in one fully parsed semicolon property sequence."""
    spans = _semicolon_property_spans(value, start)
    if not spans:
        return None

    replacements: list[tuple[int, int, str]] = []
    normalized_keys: set[str] = set()
    for _, end, equal, key in spans:
        normalized, _ = _normalized_key_spellings(key)
        normalized_keys.add(normalized)
        if _connection_property_credential_key(key):
            replacements.append(
                (
                    equal + 1,
                    end,
                    _redacted_credential_value(
                        value[equal + 1 : end],
                        preserve_binding_identity=preserve_binding_identity,
                    ),
                )
            )

    redacted = value
    for value_start, value_end, replacement in reversed(replacements):
        redacted = (
            redacted[:value_start]
            + replacement
            + redacted[value_end:]
        )
    return redacted, frozenset(normalized_keys)


def _redact_jdbc_sqlserver_credentials(
    value: str,
    *,
    preserve_binding_identity: bool = False,
) -> str | None:
    """Sanitize standard SQL Server JDBC semicolon properties."""
    prefix = _JDBC_SQLSERVER_PREFIX.match(value)
    if prefix is None:
        return None
    property_delimiter = value.find(";", prefix.end())
    if property_delimiter == -1:
        return (
            _redact_hierarchical_uri(
                value,
                preserve_binding_identity=preserve_binding_identity,
            )
            or value
        )
    parsed = _redact_semicolon_properties(
        value,
        property_delimiter + 1,
        preserve_binding_identity=preserve_binding_identity,
    )
    if parsed is None:
        return _REDACTED_URI_CREDENTIAL
    redacted, _ = parsed
    return (
        _redact_hierarchical_uri(
            redacted,
            preserve_binding_identity=preserve_binding_identity,
        )
        or redacted
    )


def _redact_jdbc_oracle_thin_credentials(
    value: str,
    *,
    preserve_binding_identity: bool = False,
) -> str | None:
    """Sanitize Oracle thin inline credentials while preserving service/SID identity."""
    prefix = _JDBC_ORACLE_THIN_PREFIX.match(value)
    if prefix is None:
        return None
    rest = value[prefix.end() :]
    at = rest.rfind("@")
    if at == -1:
        return _redact_uri_tail_params(
            value,
            preserve_binding_identity=preserve_binding_identity,
        )
    endpoint = _redact_uri_tail_params(
        rest[at + 1 :],
        preserve_binding_identity=preserve_binding_identity,
    )
    if at == 0:
        return value[: prefix.end()] + "@" + endpoint
    credential = _redacted_credential_value(
        rest[:at],
        preserve_binding_identity=preserve_binding_identity,
    )
    return (
        value[: prefix.end()]
        + f"{credential}/{credential}@"
        + endpoint
    )


def _redact_semicolon_dsn_credentials(
    value: str,
    *,
    preserve_binding_identity: bool = False,
) -> str | None:
    """Sanitize a recognized key/value DSN without normalizing semantic fields."""
    parsed = _redact_semicolon_properties(
        value,
        0,
        preserve_binding_identity=preserve_binding_identity,
    )
    if parsed is None:
        return None
    redacted, keys = parsed
    if keys.isdisjoint(_DSN_IDENTITY_PROPERTY_NAMES):
        return None
    return redacted


def _libpq_property_spans(
    value: str,
) -> list[tuple[int, int, str]] | None:
    """Parse libpq keyword/value fields, including quoted and escaped values."""
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    while cursor < len(value):
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1
        if cursor == len(value):
            break

        key_start = cursor
        while (
            cursor < len(value)
            and not value[cursor].isspace()
            and value[cursor] != "="
        ):
            cursor += 1
        key = value[key_start:cursor]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
            return None
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1
        if cursor == len(value) or value[cursor] != "=":
            return None
        cursor += 1
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1

        value_start = cursor
        if cursor < len(value) and value[cursor] == "'":
            cursor += 1
            while cursor < len(value):
                if value[cursor] == "\\":
                    if cursor + 1 == len(value):
                        return None
                    cursor += 2
                elif value[cursor] == "'":
                    cursor += 1
                    break
                else:
                    cursor += 1
            else:
                return None
            if cursor < len(value) and not value[cursor].isspace():
                return None
        else:
            while cursor < len(value) and not value[cursor].isspace():
                if value[cursor] == "\\":
                    if cursor + 1 == len(value):
                        return None
                    cursor += 2
                else:
                    cursor += 1
        spans.append((value_start, cursor, key))
    return spans


def _redact_libpq_conninfo_credentials(
    value: str,
    *,
    require_credential: bool = False,
    preserve_binding_identity: bool = False,
) -> str | None:
    """Sanitize a recognized libpq keyword/value connection string."""
    spans = _libpq_property_spans(value)
    if not spans:
        return None
    normalized_keys = {
        _normalized_key_spellings(key)[0] for _, _, key in spans
    }
    if normalized_keys.isdisjoint(_DSN_IDENTITY_PROPERTY_NAMES):
        return None
    if require_credential and not any(
        _connection_property_credential_key(key) for _, _, key in spans
    ):
        return None

    redacted = value
    for value_start, value_end, key in reversed(spans):
        if _connection_property_credential_key(key):
            replacement = _redacted_credential_value(
                value[value_start:value_end],
                preserve_binding_identity=preserve_binding_identity,
            )
            redacted = (
                redacted[:value_start]
                + replacement
                + redacted[value_end:]
            )
    return redacted


def _redact_connection_credentials(
    value: str,
    *,
    preserve_binding_identity: bool = False,
) -> tuple[str, bool]:
    """Return a sanitized recognized URI/DSN and whether its structure was recognized."""
    sqlserver = _redact_jdbc_sqlserver_credentials(
        value,
        preserve_binding_identity=preserve_binding_identity,
    )
    if sqlserver is not None:
        return sqlserver, True
    oracle = _redact_jdbc_oracle_thin_credentials(
        value,
        preserve_binding_identity=preserve_binding_identity,
    )
    if oracle is not None:
        return oracle, True
    uri = _redact_hierarchical_uri(
        value,
        preserve_binding_identity=preserve_binding_identity,
    )
    if uri is not None:
        return uri, True
    if ";" in value:
        dsn = _redact_semicolon_dsn_credentials(
            value,
            preserve_binding_identity=preserve_binding_identity,
        )
        if dsn is not None:
            return dsn, True
        # A malformed semicolon DSN can resemble one giant libpq property. Only
        # accept the fallback when libpq parsed a credential as its own field;
        # otherwise exclude the ambiguous value rather than risk retaining a secret.
        conninfo = _redact_libpq_conninfo_credentials(
            value,
            require_credential=True,
            preserve_binding_identity=preserve_binding_identity,
        )
    else:
        conninfo = _redact_libpq_conninfo_credentials(
            value,
            preserve_binding_identity=preserve_binding_identity,
        )
    if conninfo is not None:
        return conninfo, True
    return value, False


def _redact_uri_credentials(
    value: str,
    *,
    preserve_binding_identity: bool = False,
) -> str:
    """Remove recognized URI/JDBC/DSN credentials without normalizing semantics."""
    redacted, _ = _redact_connection_credentials(
        value,
        preserve_binding_identity=preserve_binding_identity,
    )
    return redacted


def _sorted_mapping(
    value: dict,
    ctx: _ManifestContext,
    path: str,
    *,
    redact_credentials: bool = False,
    preserve_binding_identity: bool = False,
) -> dict[str, Any]:
    """Canonicalize a string-keyed mapping, using typed entries when redacting."""
    bad_key = next((key for key in value if not isinstance(key, str)), _MISSING)
    if bad_key is not _MISSING:
        raise TypeError(
            f"{path}: mapping keys must be strings, got {type(bad_key).__name__}"
        )
    if redact_credentials:
        return _canonical_mapping(
            value,
            ctx,
            path,
            redact_credentials=True,
            preserve_binding_identity=preserve_binding_identity,
        )

    object_id = id(value)
    if object_id in ctx.values:
        raise ValueError(f"{path}: cyclic value is not manifestable")
    ctx.values.add(object_id)
    try:
        out: dict[str, Any] = {}
        for key in sorted(value):
            out[key] = _canonical_value(
                value[key],
                ctx,
                f"{path}.{key}",
            )
        return out
    finally:
        ctx.values.remove(object_id)


def _canonical_sequence(
    value: Any,
    ctx: _ManifestContext,
    path: str,
    *,
    redact_credentials: bool = False,
    redact_connection_properties: bool = False,
) -> list[Any]:
    """Canonicalize an ordered list/tuple while preserving element order."""
    object_id = id(value)
    if object_id in ctx.values:
        raise ValueError(f"{path}: cyclic value is not manifestable")
    ctx.values.add(object_id)
    try:
        return [
            _canonical_value(
                item,
                ctx,
                f"{path}[{index}]",
                redact_credentials=redact_credentials,
                redact_connection_properties=redact_connection_properties,
            )
            for index, item in enumerate(value)
        ]
    finally:
        ctx.values.remove(object_id)


def _canonical_set(
    value: Any,
    ctx: _ManifestContext,
    path: str,
    *,
    redact_credentials: bool = False,
    redact_connection_properties: bool = False,
) -> list[Any]:
    """Canonicalize unordered members by each member's canonical JSON spelling."""
    object_id = id(value)
    if object_id in ctx.values:
        raise ValueError(f"{path}: cyclic value is not manifestable")
    ctx.values.add(object_id)
    try:
        items = [
            _canonical_value(
                item,
                ctx,
                f"{path}[]",
                redact_credentials=redact_credentials,
                redact_connection_properties=redact_connection_properties,
            )
            for item in value
        ]
        return sorted(items, key=_canonical_json)
    finally:
        ctx.values.remove(object_id)


def _canonical_connection_value(
    value: Any,
    ctx: _ManifestContext,
    path: str,
    *,
    preserve_binding_identity: bool = False,
) -> Any:
    """Canonicalize only a safely recognized connection string or structured mapping."""
    if isinstance(value, str):
        sanitized, recognized = _redact_connection_credentials(
            value,
            preserve_binding_identity=preserve_binding_identity,
        )
        if recognized:
            return {"type": "str", "value": sanitized}
    elif isinstance(value, dict):
        return _canonical_value(
            value,
            ctx,
            path,
            redact_credentials=True,
            redact_connection_properties=True,
        )
    return _MISSING


def _canonical_mapping(
    value: dict,
    ctx: _ManifestContext,
    path: str,
    *,
    redact_credentials: bool = False,
    redact_connection_properties: bool = False,
    preserve_binding_identity: bool = False,
) -> dict[str, Any]:
    """Canonicalize an arbitrary-key mapping as sorted typed key/value entries."""
    object_id = id(value)
    if object_id in ctx.values:
        raise ValueError(f"{path}: cyclic value is not manifestable")
    ctx.values.add(object_id)
    try:
        entries: list[dict[str, Any]] = []
        for index, (key, item) in enumerate(value.items()):
            # A URI used AS a mapping key is semantic endpoint identity, not a credential
            # field name. Retain it and redact only its userinfo/query values below.
            canonical_item: Any = _MISSING
            redacted_string_key = (
                redact_credentials
                and isinstance(key, str)
                and _URI_PREFIX.match(key) is None
            )
            redacted_credential_key = redacted_string_key and (
                _connection_string_key(key)
                or _credential_key(key)
                or (
                    redact_connection_properties
                    and _connection_property_credential_key(key)
                )
            )
            binding_marker = (
                _credential_binding_marker(
                    item,
                    preserve_http_auth_scheme=(
                        _normalized_key_spellings(key)[0]
                        in {"authorization", "proxy_authorization"}
                    ),
                )
                if (
                    preserve_binding_identity
                    and redacted_credential_key
                    and isinstance(item, str)
                )
                else None
            )
            if redacted_string_key and _connection_string_key(key):
                canonical_item = _canonical_connection_value(
                    item,
                    ctx,
                    f"{path}.values[{index}]",
                    preserve_binding_identity=preserve_binding_identity,
                )
                if canonical_item is _MISSING:
                    if binding_marker is None:
                        continue
                    canonical_item = _canonical_value(
                        binding_marker,
                        ctx,
                        f"{path}.values[{index}]",
                    )
            elif binding_marker is not None:
                # Dynamic source identity is safe to retain even when the target
                # slot is credential-named; the marker contains no literal operand.
                canonical_item = _canonical_value(
                    binding_marker,
                    ctx,
                    f"{path}.values[{index}]",
                )
            elif redacted_string_key and (
                _credential_key(key)
                or (
                    redact_connection_properties
                    and _connection_property_credential_key(key)
                )
            ):
                normalized_key, _ = _normalized_key_spellings(key)
                if isinstance(item, dict) and normalized_key == "auth":
                    canonical_item = _canonical_value(
                        item,
                        ctx,
                        f"{path}.values[{index}]",
                        redact_credentials=True,
                        redact_connection_properties=True,
                    )
                else:
                    continue
            entry = {
                "key": _canonical_value(
                    key,
                    ctx,
                    f"{path}.keys[{index}]",
                    redact_credentials=redact_credentials,
                    redact_connection_properties=redact_connection_properties,
                ),
                "value": (
                    canonical_item
                    if canonical_item is not _MISSING
                    else _canonical_value(
                        item,
                        ctx,
                        f"{path}.values[{index}]",
                        redact_credentials=redact_credentials,
                        redact_connection_properties=redact_connection_properties,
                        preserve_binding_identity=preserve_binding_identity,
                    )
                ),
            }
            entries.append(entry)
        entries.sort(key=_canonical_json)
        return {"entries": entries, "type": "mapping"}
    finally:
        ctx.values.remove(object_id)


def _canonical_value(
    value: Any,
    ctx: _ManifestContext,
    path: str,
    *,
    redact_credentials: bool = False,
    redact_connection_properties: bool = False,
    preserve_binding_identity: bool = False,
) -> Any:
    """Encode one value as a deterministic, type-discriminated JSON record."""
    if isinstance(value, Enum):
        enum_type = type(value)
        return {
            "class": f"{enum_type.__module__}.{enum_type.__qualname__}",
            "type": "enum",
            "value": _canonical_value(
                value.value,
                ctx,
                path,
                redact_credentials=redact_credentials,
                redact_connection_properties=redact_connection_properties,
            ),
        }
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        return {"type": "float", "value": struct.pack("!d", value).hex()}
    if isinstance(value, str):
        if redact_credentials:
            value = _redact_uri_credentials(
                value,
                preserve_binding_identity=preserve_binding_identity,
            )
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
        return _canonical_mapping(
            value,
            ctx,
            path,
            redact_credentials=redact_credentials,
            redact_connection_properties=redact_connection_properties,
        )
    if isinstance(value, list):
        return {
            "items": _canonical_sequence(
                value,
                ctx,
                path,
                redact_credentials=redact_credentials,
                redact_connection_properties=redact_connection_properties,
            ),
            "type": "list",
        }
    if isinstance(value, tuple):
        return {
            "items": _canonical_sequence(
                value,
                ctx,
                path,
                redact_credentials=redact_credentials,
                redact_connection_properties=redact_connection_properties,
            ),
            "type": "tuple",
        }
    if isinstance(value, set):
        return {
            "items": _canonical_set(
                value,
                ctx,
                path,
                redact_credentials=redact_credentials,
                redact_connection_properties=redact_connection_properties,
            ),
            "type": "set",
        }
    if isinstance(value, frozenset):
        return {
            "items": _canonical_set(
                value,
                ctx,
                path,
                redact_credentials=redact_credentials,
                redact_connection_properties=redact_connection_properties,
            ),
            "type": "frozenset",
        }
    raise TypeError(
        f"{path}: unsupported non-JSON value of type {type(value).__module__}."
        f"{type(value).__qualname__}"
    )


def _type_manifest(value: Type | None, ctx: _ManifestContext, path: str) -> Any:
    """Canonicalize one resolved structural Type, including nested fields/elements."""
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
        return {
            "element": _type_manifest(value.element, ctx, f"{path}.element"),
            "fields": fields,
            "kind": value.kind.value,
            "nullable": value.nullable,
            "required": sorted(value.required) if value.required is not None else None,
            "tags": sorted(value.tags) if value.tags is not None else None,
        }
    finally:
        ctx.types.remove(object_id)


def _declaration_default_manifest(
    name: str,
    default: Any,
    ctx: _ManifestContext,
    path: str,
) -> Any:
    """Canonicalize one declaration default under the credential-redaction policy."""
    if _connection_string_key(name):
        default = _canonical_connection_value(
            default,
            ctx,
            path,
        )
        if default is _MISSING:
            default = {"type": "redacted"}
    elif _credential_key(name):
        default = {"type": "redacted"}
    else:
        default = _canonical_value(
            default,
            ctx,
            path,
            redact_credentials=True,
        )
    return default


def _value_contract_manifest(
    value: ValueContract,
    ctx: _ManifestContext,
    path: str,
) -> dict[str, Any]:
    """Project one parameter value contract while excluding every nested origin."""
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
        "default": _declaration_default_manifest(
            value.name,
            value.default,
            ctx,
            f"{path}.default",
        ),
        "has_default": value.has_default,
        "name": value.name,
        "required": value.required,
    }


def _case_manifest(value: Any, ctx: _ManifestContext, path: str) -> dict[str, Any]:
    """Canonicalize one ordered CASE arm."""
    if type(value) is not Case:
        raise TypeError(f"{path}: expected Case, got {type(value).__name__}")
    _check_fields(value, _CASE_FIELDS, path)
    return {"handle": value.handle, "when": _canonical_value(value.when, ctx, f"{path}.when")}


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
    manifest: dict[str, Any] = {
        "commit_as": _canonical_value(node.commit_as, ctx, f"{path}.commit_as"),
        "env": _sorted_mapping(node.env, ctx, f"{path}.env", redact_credentials=True),
        "id": node.id,
        "kind": node.kind.value,
        "origin_id": _canonical_value(node.origin_id, ctx, f"{path}.origin_id"),
        "output_type": _type_manifest(node.output_type, ctx, f"{path}.output_type"),
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

    if node.kind is NodeKind.START:
        pass
    elif node.kind is NodeKind.END:
        manifest["mode"] = node._mode
        manifest["names"] = (
            _canonical_sequence(node._names, ctx, f"{path}.names")
            if node._mode == "record"
            else None
        )
        manifest["n"] = node._n if node._mode == "list" else None
    elif node.kind is NodeKind.AGENT:
        manifest.update(
            {
                "llm_inherit": node.llm_inherit,
                "mode": node.mode,
                "prompt": node.prompt,
                "retries": node.retries,
                "tools": _canonical_sequence(node.tools, ctx, f"{path}.tools"),
                "own_llm_config": _sorted_mapping(
                    node.own_llm_config,
                    ctx,
                    f"{path}.own_llm_config",
                    redact_credentials=True,
                ),
            }
        )
    elif node.kind is NodeKind.CODE:
        manifest["code"] = node.code
    elif node.kind is NodeKind.MODEL:
        manifest.update(
            {
                "model_id": node.model_id,
                "runtime": _canonical_value(node.runtime_name, ctx, f"{path}.runtime"),
                "weights_uri": _canonical_value(
                    node.weights_uri,
                    ctx,
                    f"{path}.weights_uri",
                    redact_credentials=True,
                ),
            }
        )
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
    else:  # pragma: no cover - _check_node_fields exhausts the closed enum first
        raise ValueError(f"{path}: unsupported node kind {node.kind.value!r}")

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
    return {
        "from": _canonical_value(source, ctx, f"{path}.binding.source"),
        "name": output.name,
    }


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
                    "outputs",
                    "flow_llm_config",
                    "_incoming",
                    "_outgoing",
                }
            )
        elif type(flow) is Flow:
            expected = frozenset({"nodes", "edges", "wiring", "start_id", "end_id"})
        else:
            raise TypeError(f"{path}: unsupported Flow subclass {type(flow).__name__}")
        _check_fields(flow, expected, path)

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
                binding_sources,
                ctx,
                f"{path}.wiring.{node_id}",
                redact_credentials=True,
                preserve_binding_identity=True,
            )

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
        if type(flow) is CompiledFlow and not topology_only:
            manifest["flow_llm_config"] = _sorted_mapping(
                flow.flow_llm_config,
                ctx,
                f"{path}.flow_llm_config",
                redact_credentials=True,
            )
            manifest["outputs"] = [
                _output_manifest(output, ctx, f"{path}.outputs[{index}]")
                for index, output in enumerate(flow.outputs)
            ]
        return {key: manifest[key] for key in sorted(manifest)}
    finally:
        ctx.flows.remove(object_id)


def _canonical_json(value: Any) -> str:
    """Encode already-canonical primitives in one byte-stable JSON spelling."""
    return json.dumps(
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
