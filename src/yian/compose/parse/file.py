"""The section parser: one owned source document -> a strict `ComposeFile`.

This is the analogue of `manifest/parser.py`: inspect an existing document
cursor, copy its semantic value into a strict Pydantic model, and raise
`LoadError` on any violation. Four surface rules shape it:

- **Top-level sections only** (`id`/`name`/`description?`/`input`/`nodes`/
  `output`/`asserts`/`long_term_memory`/`typedefs`/`defs`/`uses`/`system`).
  `extra="forbid"` on the
  strict body makes a typo'd top-level key (e.g. `nodez:`) loud rather than silently
  dropped. (`uses:` = external callables; `system: paths:` = the resolution search
  path — NOT the strict `${system.X}` ambient namespace.)
- **`x-*` extension keys** at the top level are stripped before validation —
  custom metadata / YAML-anchor holders the engine ignores.
- **YAML anchors** (`&` / `*` / `<<:` merge) are expanded by the source
  document's single safe construction pass before the strict schema sees it.
- **Every mapping key is text**, anywhere in the file (`x-*` blocks included): a
  key YAML reads as a bool, number, null, date, or bytes fails at the key first.

`nodes` stays a raw `dict` here; `compose.parse.nodes` constructs per-kind descriptors.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.nodes import REMOVED_KINDS
from yian.compose.parse.nodes import IDENTIFIER_RULE, identifier_suggestion, is_identifier
from yian.compose.source import (
    SourceAccessError,
    SourceCursor,
    SourceDocument,
    SourceSyntaxError,
)
from yian.diagnostics import SourceRange

# `AgentDescriptor`/`CodeDescriptor`/`ToolDescriptor`/`WaitDescriptor` now live with
# their parse/build handles in `nodes/<kind>.py`; re-export them here so existing
# importers keep working. `NodeDescriptor` — the base class every kind descriptor
# inherits — lives in `compose/nodes/base.py` (the leaf, mirroring the engine's `nodes/base.py`);
# re-exported here as the descriptor base type for importers that annotate
# `dict[str, NodeDescriptor]` (build/pipeline).
from yian.compose.nodes.base import NodeDescriptor  # noqa: F401
from yian.compose.nodes.agent import AgentDescriptor  # noqa: F401
from yian.compose.nodes.code import CodeDescriptor  # noqa: F401
from yian.compose.nodes.tool import ToolDescriptor  # noqa: F401
from yian.compose.nodes.wait import WaitDescriptor  # noqa: F401
from yian.compose.nodes.human_input import HumanInputDescriptor  # noqa: F401
from yian.compose.nodes.case import CaseDescriptor  # noqa: F401
from yian.compose.nodes.call import CallDescriptor  # noqa: F401
from yian.compose.nodes.map import MapDescriptor  # noqa: F401
from yian.compose.nodes.loop import LoopDescriptor  # noqa: F401

_STRICT = ConfigDict(extra="forbid")


def _normalize_section_keys(
    body: Any,
    *,
    diagnostics: LoadDiagnostics | None = None,
    range_for_key: Callable[[str], SourceRange | None] | None = None,
) -> Any:
    """Reject retired top-level flow-key spellings.

    Post-alias-delete, the normalizer is a REJECTOR. A top-level flow with legacy
    `inputs:` / `outputs:` or retired `memory_policy:` raises a bespoke `LoadError`
    naming the replacement. Only `input:` / `output:` and `long_term_memory:` are
    accepted.

    NOTE: this function does NOT recurse into nested per-node or `defs:` bodies — the
    descriptor dataclasses (AgentDescriptor, CodeDescriptor, ...) keep their internal
    `inputs`/`outputs` field names (Python collection fields stay plural).
    The body-level back-map (`_back_map_to_plural`) translates the new singular
    keys to the dataclass field names. This helper's rejection scope is the top-level
    mapping passed to it; `_reject_nested_legacy_keys` checks node and def bodies.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    if not isinstance(body, dict):
        return body
    for legacy, new in (("inputs", "input"), ("outputs", "output")):
        if legacy in body:
            raise diagnostics.error(
                f"unknown top-level key {legacy!r} — flows use `{new}:` "
                f"(rename the section)",
                range=(range_for_key(legacy) if range_for_key is not None else None),
            )
    if "memory_policy" in body:
        raise diagnostics.error(
            "unknown top-level key 'memory_policy' — `memory_policy:` was renamed "
            "to `long_term_memory:`",
            range=(
                range_for_key("memory_policy")
                if range_for_key is not None
                else None
            ),
        )
    return body


_NESTED_SECTIONS = (("nodes", "node", "nodes"), ("defs", "defs entry", "defs"))


def _reject_nested_legacy_keys(cursor: SourceCursor, diagnostics: LoadDiagnostics) -> None:
    """Reject retired `inputs:`/`outputs:` in every authored node and def body.

    Walks the same `nodes:`/`defs:` nesting that `_back_map_to_plural` projects. The
    check reads the authored source because the projection would merge `inputs:` with
    a renamed `input:` and keep only one of them.

    Args:
        cursor (`SourceCursor`):
            Authored flow, node, or def body mapping whose nested sections are checked.
        diagnostics (`LoadDiagnostics`):
            Builder for the located `LoadError`.

    Raises:
        `LoadError`:
            Raised at the first `inputs:` or `outputs:` key in a nested body.
        `SourceAccessError`:
            Raised when a nested section cannot be traversed.
    """
    for section, noun, plural_noun in _NESTED_SECTIONS:
        if not cursor.has(section):
            continue
        section_cursor = cursor.child(section)
        entries = section_cursor._value_view()
        if not isinstance(entries, Mapping):
            continue
        for name, entry in entries.items():
            if not isinstance(entry, Mapping):
                continue
            entry_cursor = section_cursor.child(name)
            for legacy, new in (("inputs", "input"), ("outputs", "output")):
                if legacy in entry:
                    raise diagnostics.error(
                        f"{noun} {name!r}: unknown key {legacy!r} — {plural_noun} use "
                        f"`{new}:` (rename the key)",
                        range=entry_cursor.key_range_of(legacy),
                    )
            _reject_nested_legacy_keys(entry_cursor, diagnostics)


def _back_map_to_plural(body: Any) -> Any:
    """Map normalized `input`/`output` keys back to `inputs`/`outputs` for the dataclass
    constructors (transitional — the dataclass fields stay plural). Recurses into
    `nodes:`/`defs:`."""
    if not isinstance(body, dict):
        return body
    out = dict(body)
    if "input" in out:
        out["inputs"] = out.pop("input")
    if "output" in out:
        out["outputs"] = out.pop("output")
    if isinstance(out.get("nodes"), dict):
        out["nodes"] = {
            nid: _back_map_to_plural(nb) for nid, nb in out["nodes"].items()
        }
    if isinstance(out.get("defs"), dict):
        out["defs"] = {
            dname: _back_map_to_plural(db) for dname, db in out["defs"].items()
        }
    return out


class ComposeFile(BaseModel):
    """A parsed Compose flow — top-level sections, pre-node-descriptor.

    `inputs` / `nodes` / `typedefs` / `defs` stay raw dicts and `outputs` raw `Any` (a
    `${...}` binding string or a name -> binding map); downstream parsers read each
    section into its typed form. `defs` holds in-file callables (each entry an authored
    child flow `{input?, nodes, output?, asserts?}`), resolved by a `call:` defs-first.

    A `model_validator(mode='before')` maps authored `input:`/`output:` keys to the
    plural Python carrier fields at the top level and recursively in node/definition
    bodies. `_normalize_section_keys` rejects retired top-level spellings.
    """

    model_config = _STRICT

    id: str
    name: str
    description: Optional[str] = None
    version: Optional[str] = (
        None  # opaque version tag (v1, 1.2.0, ...); None = unversioned
    )
    inputs: dict[str, Any] = Field(default_factory=dict)
    nodes: dict[str, Any]
    outputs: Any = None
    asserts: list[str] = Field(default_factory=list)
    typedefs: dict[str, Any] = Field(default_factory=dict)
    defs: dict[str, Any] = Field(default_factory=dict)
    uses: dict[str, str] = Field(default_factory=dict)
    system: dict[str, Any] = Field(default_factory=dict)
    # Flow-level model-selection defaults (the cascade's flow layer). Optional — absent
    # is `{}`. A child agent fills the gaps it leaves unset from this, then from the
    # parent/CLI layers and the explicit process defaults captured at run initialization;
    # `resolve_llm_cascade` bakes each dict and returns the complete static-path map.
    llm_config: dict[str, Any] = Field(default_factory=dict)
    # Flow-level static per-node config defaults (`{str -> literal}`). Optional — absent is
    # `{}`. Each node in THIS flow merges it under its own `env:` (node override wins); the
    # builder bakes the effective map onto `Node.env`. Flow-local: a called child flow does
    # NOT inherit this — it has its own flow `env:`.
    env: dict[str, Any] = Field(default_factory=dict)
    # Full flows retain this raw occurrence for the shared located policy parser.
    # Compact flows project it into their synthesized node because it intentionally
    # stays outside ``_COMPACT_FLOW_KEYS`` below.
    long_term_memory: Any = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_sections(cls, data: Any) -> Any:
        """Map authored `input:`/`output:` to carrier fields before Pydantic validates.

        The Python fields are named `inputs`/`outputs`, so `_back_map_to_plural`
        performs the projection recursively through `nodes:` and `defs:`. The top-level
        legacy-key check happens before this projection.
        """
        if not isinstance(data, dict):
            return data
        normalized = _normalize_section_keys(data)
        return _back_map_to_plural(normalized)


# Compact (single-node) flow: the flow's top-level body carries a node `kind:` and
# NO `nodes:` map — the flow *is* one node. These flow-metadata keys stay at the flow
# level on desugar; every other key (the node `kind:` + its logic fields) becomes the
# single node's body. `input:`/`output:` get special treatment (see `_desugar_compact`).
_COMPACT_FLOW_KEYS = frozenset({"id", "name", "description", "version", "typedefs"})

# Compact form is restricted to value-producing LEAF kinds. case/call/map reference
# other nodes or callables that a one-node flow has none of, so they make no sense
# inline — an author who needs them writes a full `nodes:` map.
_COMPACT_KINDS = frozenset({"agent", "code", "tool", "human_input"})


def _desugar_compact(body: dict, root: SourceCursor) -> dict:
    """Desugar a compact single-node flow into the canonical one-node form.

    A compact flow carries a node `kind:` at the top level and NO `nodes:` map — the
    flow *is* the node. Mirrors `loader._load_single_node_def` (the `defs:` precedent):

    - the flow `id:` names the single node, so it must be an identifier (one id for both
      flow and node, so errors reference a meaningful name);
    - the flow `input:` is the node's SIGNATURE (`name: TYPE`), auto-wired into the node
      by name (`p = ${input.p}`);
    - the flow `output:` is the node's output TYPE (the codomain), re-exported as the
      flow output (`output: ${<id>.output}`), so the author skips the redundant wiring;
    - everything else (the `kind:` + its logic fields, e.g. `prompt:`/`asserts:`/
      `llm_config:`/`long_term_memory:`) is the node body.

    The returned dict is the canonical compose body (with a `nodes:` map) the strict
    `ComposeFile` schema then validates unchanged. Raises `LoadError` on a removed
    kind, a non-leaf kind, or a missing or non-identifier flow `id:`.
    """
    diagnostics = LoadDiagnostics.from_cursor(root)
    kind = body["kind"]
    if kind in REMOVED_KINDS:
        raise diagnostics.error(
            f"compact (single-node) flow: kind {kind!r} was removed; {REMOVED_KINDS[kind]}",
            range=root.range_of("kind"),
        )
    if kind not in _COMPACT_KINDS:
        raise diagnostics.error(
            f"compact (single-node) flow: kind {kind!r} is not allowed inline "
            f"(allowed: {', '.join(sorted(_COMPACT_KINDS))}); case/call/map reference "
            f"other nodes a one-node flow has none of — write a full `nodes:` map instead",
            range=root.range_of("kind"),
        )
    flow_id = body.get("id")
    if not isinstance(flow_id, str) or not flow_id:
        raise diagnostics.error(
            "compact (single-node) flow: a top-level `id:` is required — it names the "
            "single node",
            range=root.range_of("id") if root.has("id") else root.range,
        )
    if not is_identifier(flow_id):
        raise diagnostics.error(
            f"compact flow id {flow_id!r} also names its node and must be an identifier "
            f"({IDENTIFIER_RULE}); use {identifier_suggestion(flow_id)!r}, or the full "
            f"`nodes:` form to keep a hyphenated flow id",
            range=root.range_of("id"),
        )
    params = body.get("input") or {}
    if not isinstance(params, dict):
        raise diagnostics.error(
            "compact (single-node) flow: `input:` must be a param map (name: TYPE)",
            range=root.range_of("input"),
        )

    # The node body = the kind + its logic fields (everything that is not flow metadata
    # or the special input:/output: keys). Auto-wire the flow inputs by name; carry the
    # flow `output:` through as the node's output TYPE.
    node_body = {
        k: v
        for k, v in body.items()
        if k not in _COMPACT_FLOW_KEYS and k not in ("input", "output")
    }
    node_body["input"] = {p: f"${{input.{p}}}" for p in params}
    if "output" in body:
        node_body["output"] = body["output"]

    canonical = {k: body[k] for k in _COMPACT_FLOW_KEYS if k in body}
    canonical["input"] = params
    canonical["nodes"] = {flow_id: node_body}
    canonical["output"] = f"${{{flow_id}.output}}"  # re-export the single node's output
    return canonical


_MODEL_TO_SOURCE_SECTION = {"inputs": "input", "outputs": "output"}


def _validation_error_range(
    exc: ValidationError,
    root: SourceCursor,
) -> SourceRange | None:
    """Locate the first Pydantic error path that replays through ``root``.

    Args:
        exc (`ValidationError`): Schema failure whose ordered locations are tested.
        root (`SourceCursor`): Retained physical root used to replay each location.

    Returns:
        `SourceRange | None`: First replayable range, otherwise the root range.
    """

    for detail in exc.errors():
        if not isinstance(detail, Mapping):
            continue
        loc = detail.get("loc")
        if not isinstance(loc, (list, tuple)) or not loc:
            continue

        cursor = root
        for index, segment in enumerate(loc):
            if segment == "[key]" and index == len(loc) - 1:
                if cursor.key_range is not None:
                    return cursor.key_range
                break
            if index == 0:
                segment = _MODEL_TO_SOURCE_SECTION.get(segment, segment)
            try:
                cursor = cursor.child(segment)
            except SourceAccessError:
                break
        else:
            if cursor.range is not None:
                return cursor.range

    return root.range


def _yaml_read_as(key: Any) -> str:
    """Name what YAML read a non-text key as, for an error message.

    Args:
        key (`Any`): The constructed key.

    Returns:
        `str`: A phrase such as ``a boolean`` or ``a date``.
    """
    # A bool is an int and a datetime is a date, so the subclasses go first.
    if isinstance(key, bool):
        return "a boolean"
    if isinstance(key, int):
        return "an integer"
    if isinstance(key, float):
        return "a float"
    if isinstance(key, datetime):
        return "a date and time"
    if isinstance(key, date):
        return "a date"
    if key is None:
        return "null"
    if isinstance(key, bytes):
        return "binary data"
    return f"a `{type(key).__name__}`"


def _non_text_key_message(key: Any, text: str) -> str:
    """Explain that a key YAML read as non-text can't name anything, and the fix.

    Args:
        key (`Any`): The constructed key.
        text (`str`): The key's scalar text as written.

    Returns:
        `str`: The load error message.
    """
    read_as = _yaml_read_as(key)
    if not text:
        return (
            f"an empty key was read by YAML as {read_as}, but names in a flow must "
            "be text. Write a name."
        )
    # json.dumps gives a valid YAML double-quoted scalar for any text.
    return (
        f"the key `{text}` was read by YAML as {read_as}, but names in a flow must "
        f"be text. Quote it ({json.dumps(text)}) or choose another name."
    )


def parse_source_document(text: str, label: str) -> SourceDocument:
    """Parse one owned source document once.

    Rejects bad syntax, a key YAML read as non-text, and repeated keys.
    """

    try:
        document = SourceDocument.parse(text, label)
    except SourceSyntaxError as exc:
        raise LoadDiagnostics(label=label, text=text).error(
            f"invalid YAML: {exc}",
            range=exc.range,
        ) from exc
    if document.non_text_key is not None:
        key, key_text, key_range = document.non_text_key
        raise LoadDiagnostics(label=label, text=text).error(
            _non_text_key_message(key, key_text),
            range=key_range,
        )
    if document.repeated_key is not None:
        key, first, second = document.repeated_key
        raise LoadDiagnostics(label=label, text=text).error(
            f"duplicate key {key!r} (first at line {first.start_line}); "
            "a YAML mapping's keys must be unique, so remove or rename one",
            range=second,
        )
    return document


def flow_nodes_cursor(
    document: SourceDocument,
    nodes_section: dict[str, Any],
) -> SourceCursor:
    """Return the authored nodes cursor or a source-aligned compact projection.

    A full flow already owns a physical ``nodes:`` mapping. A compact flow has
    one synthesized node instead: its header is aligned with the authored
    ``id:`` key, unchanged logic fields and the authored output contract retain
    their top-level sources, and generated input bindings remain virtual.
    """

    root = document.root
    if root.has("nodes"):
        return root.child("nodes")
    if len(nodes_section) != 1:
        return SourceCursor.virtual(nodes_section)

    node_id, node_body = next(iter(nodes_section.items()))
    if not isinstance(node_body, Mapping):
        return SourceCursor.virtual(nodes_section)

    entries: dict[Any, tuple[Any, SourceCursor | None]] = {}
    for key, value in node_body.items():
        source = None
        if key == "outputs" and root.has("output"):
            source = root.child("output")
        elif key not in ("inputs", "outputs") and root.has(key):
            source = root.child(key)
        entries[key] = (value, source)

    header = root.child("id") if root.has("id") else root
    body_cursor = header.project_mapping(entries)
    return root.project_mapping({node_id: (node_body, body_cursor)})


def parse_file(source: str | SourceDocument) -> ComposeFile:
    """Parse a source document into a strict `ComposeFile`.

    Raises `LoadError` on malformed YAML, a non-mapping document, an unknown
    top-level key (after stripping `x-*`), a retired `inputs:`/`outputs:` key in a
    node or def body, or any schema violation.
    """
    document = (
        parse_source_document(source, "<flow>") if isinstance(source, str) else source
    )
    root = document.root
    diagnostics = LoadDiagnostics.from_document(document)
    raw = root._value_view()
    if not isinstance(raw, Mapping):
        raise diagnostics.error(
            "a flow must be a YAML mapping at the top level",
            range=root.range,
        )

    body = {key: root.get(key) for key in raw if not str(key).startswith("x-")}
    _normalize_section_keys(
        body,
        diagnostics=diagnostics,
        range_for_key=root.key_range_of,
    )
    try:
        _reject_nested_legacy_keys(root, diagnostics)
    except SourceAccessError as exc:
        raise diagnostics.error(str(exc), range=exc.range) from exc

    # Compact (single-node) flow: a top-level node `kind:` with no `nodes:` map. Desugar
    # to the canonical one-node form BEFORE the strict-schema checks below, since `nodes:`
    # is required and `extra="forbid"` would otherwise reject the inline node fields.
    if "kind" in body and "nodes" not in body:
        body = _desugar_compact(body, root)

    # ComposeFile's model validator projects the singular surface keys onto plural
    # Python carrier fields at the top level and in node/definition bodies. Legacy
    # plural top-level spellings were rejected above.
    allowed_top = set(ComposeFile.model_fields) | {"input", "output"}
    for key in body:
        if key not in allowed_top:
            raise diagnostics.error(
                f"unknown top-level key {key!r} "
                f"(allowed: {', '.join(sorted(set(ComposeFile.model_fields)))})",
                range=root.key_range_of(key),
            )

    try:
        return ComposeFile.model_validate(body)
    except LoadError:
        raise
    except ValidationError as exc:
        raise diagnostics.error(
            str(exc),
            range=_validation_error_range(exc, root),
        ) from exc
    except Exception as exc:
        raise diagnostics.error(str(exc)) from exc


# ---------- per-kind node descriptors ----------
#
# A `nodes:` entry is a keyed map (key = node id, no `id:` field) carrying a
# `kind:` plus flat kind fields — the surface places `input:`/`output:`
# onto the node (vs the legacy spec's body wrappers, `spec/nodes.py`). These are
# DESCRIPTORS: the validated parsed shape per kind, each owned by its `nodes/<kind>.py`
# module (the frozen dataclass + its `parse`/`build`) and inheriting the common
# `NodeDescriptor` base (see `nodes/base.py`); this module only re-exports them.
# Building the runtime `Node` (`output_contract` + `params` + flow `wiring`, with
# `read_type` wrapped in a `ValueContract`) happens later in the loader; the descriptors
# keep `inputs`/`outputs`/`args`/`cases` raw (binding strings / type nodes).
#
# `node_name`/`depends_on`/`runs_after`/`env` are common to every kind (the `NodeDescriptor`
# base). `depends_on` and `runs_after` are RUN-ORDERING edges (no data flows): both gate on the
# source settling; `depends_on` co-skips the dependent if the source skipped, `runs_after` does
# not (the dependent still runs). See the loader edge pass.
