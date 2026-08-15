"""The section parser: one owned source document -> a strict `ComposeFile`.

This is the analogue of `manifest/parser.py`: inspect an existing document
cursor, copy its semantic value into a strict Pydantic model, and raise
`LoadError` on any violation. Three surface rules shape it:

- **Top-level sections only** (`id`/`name`/`description?`/`inputs`/`nodes`/
  `outputs`/`asserts`/`typedefs`/`defs`/`uses`/`system`). `extra="forbid"` on the
  strict body makes a typo'd top-level key (e.g. `nodez:`) loud rather than silently
  dropped. (`uses:` = external callables; `system: paths:` = the resolution search
  path — NOT the strict `${system.X}` ambient namespace.)
- **`x-*` extension keys** at the top level are stripped before validation —
  custom metadata / YAML-anchor holders the engine ignores.
- **YAML anchors** (`&` / `*` / `<<:` merge) are expanded by the source
  document's single safe construction pass before the strict schema sees it.

`nodes` stays a raw `dict` here; per-kind node descriptors are a later step.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.helpers.source import _section_lines
from agent_composer.compose.source import (
    SourceCursor,
    SourceDocument,
    SourceSyntaxError,
)

# `AgentDescriptor`/`CodeDescriptor`/`ModelDescriptor`/`ToolDescriptor`/`WaitDescriptor`
# now live with their parse/build handles in `nodes/<kind>.py`; re-export them here so
# existing importers keep working. `NodeDescriptor` — the base class every kind descriptor
# inherits — lives in `compose/nodes/base.py` (the leaf, mirroring the engine's `nodes/base.py`);
# re-exported here as the descriptor base type for importers that annotate
# `dict[str, NodeDescriptor]` (build/pipeline).
from agent_composer.compose.nodes.base import NodeDescriptor  # noqa: F401
from agent_composer.compose.nodes.agent import AgentDescriptor  # noqa: F401
from agent_composer.compose.nodes.code import CodeDescriptor  # noqa: F401
from agent_composer.compose.nodes.model import ModelDescriptor  # noqa: F401
from agent_composer.compose.nodes.tool import ToolDescriptor  # noqa: F401
from agent_composer.compose.nodes.wait import WaitDescriptor  # noqa: F401
from agent_composer.compose.nodes.human_input import HumanInputDescriptor  # noqa: F401
from agent_composer.compose.nodes.case import CaseDescriptor  # noqa: F401
from agent_composer.compose.nodes.call import CallDescriptor  # noqa: F401
from agent_composer.compose.nodes.map import MapDescriptor  # noqa: F401
from agent_composer.compose.nodes.loop import LoopDescriptor  # noqa: F401

_STRICT = ConfigDict(extra="forbid")


def _normalize_section_keys(body: Any) -> Any:
    """Reject legacy `inputs:`/`outputs:` section keywords AT THE TOP LEVEL.

    Post-alias-delete: the dual-arm normalizer is a REJECTOR. A top-level flow with a
    legacy `inputs:` / `outputs:` section keyword raises a bespoke `LoadError` naming the
    retirement. The new singular `input:` / `output:` form is the only accepted spelling.

    NOTE: this function does NOT recurse into nested per-node or `defs:` bodies — the
    descriptor dataclasses (AgentDescriptor, CodeDescriptor, ...) keep their internal
    `inputs`/`outputs` field names (Python collection fields stay plural).
    The body-level back-map (`_back_map_to_plural`) translates the new singular
    keys to the dataclass field names. The rejector fires at the TOP-level only — if a
    flow author writes `inputs:` at any depth, the top-level rejector catches it via
    parse_file before _parse_node sees it.
    """
    if not isinstance(body, dict):
        return body
    for legacy, new in (("inputs", "input"), ("outputs", "output")):
        if legacy in body:
            raise LoadError(
                f"unknown top-level key {legacy!r} — flows use `{new}:` "
                f"(rename the section)"
            )
    return body


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
        out["nodes"] = {nid: _back_map_to_plural(nb) for nid, nb in out["nodes"].items()}
    if isinstance(out.get("defs"), dict):
        out["defs"] = {dname: _back_map_to_plural(db) for dname, db in out["defs"].items()}
    return out


class ComposeFile(BaseModel):
    """A parsed Compose flow — top-level sections, pre-node-descriptor.

    `inputs` / `nodes` / `typedefs` / `defs` stay raw dicts and `outputs` raw `Any` (a
    `${...}` binding string or a name -> binding map); later steps read each
    section into its typed form. `defs` holds in-file callables (each entry a sub-flow
    `{inputs?, nodes, outputs?, asserts?}`), resolved by a `call:` defs-first.

    A `model_validator(mode='before')` normalizes BOTH `inputs:` ↔ `input:` and
    `outputs:` ↔ `output:` (top-level + nested per-node + defs bodies) — see
    `_normalize_section_keys`. The dataclass fields keep the plural names; the
    normalizer rejects the legacy plural section keywords with a bespoke `LoadError`.
    """

    model_config = _STRICT

    id: str
    name: str
    description: Optional[str] = None
    version: Optional[str] = None  # opaque version tag (v1, 1.2.0, ...); None = unversioned
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

    @model_validator(mode="before")
    @classmethod
    def _normalize_sections(cls, data: Any) -> Any:
        """Normalize `inputs:`/`outputs:` ↔ `input:`/`output:` BEFORE Pydantic validates.
        Recurses into nested `nodes:` + `defs:` bodies. Raises on ambiguous (both).

        The dataclass fields are still named `inputs`/`outputs` plural, so after
        normalization we map BACK to the plural names (`_back_map_to_plural`).
        The validator rejects the legacy plural section keywords with a bespoke message."""
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
_COMPACT_KINDS = frozenset({"agent", "code", "model", "tool", "human_input"})


def _desugar_compact(body: dict, lines: dict[str, int]) -> dict:
    """Desugar a compact single-node flow into the canonical one-node form.

    A compact flow carries a node `kind:` at the top level and NO `nodes:` map — the
    flow *is* the node. Mirrors `loader._load_single_node_def` (the `defs:` precedent):

    - the flow `id:` names the single node (one id for both flow and node, so errors
      reference a meaningful name);
    - the flow `input:` is the node's SIGNATURE (`name: TYPE`), auto-wired into the node
      by name (`p = ${input.p}`);
    - the flow `output:` is the node's output TYPE (the codomain), re-exported as the
      flow output (`output: ${<id>.output}`), so the author skips the redundant wiring;
    - everything else (the `kind:` + its logic fields, e.g. `prompt:`/`asserts:`/
      `llm_config:`) is the node body.

    The returned dict is the canonical compose body (with a `nodes:` map) the strict
    `ComposeFile` schema then validates unchanged. Raises `LoadError` on a non-leaf
    kind or a missing flow `id:`.
    """
    kind = body["kind"]
    if kind not in _COMPACT_KINDS:
        raise LoadError(
            f"compact (single-node) flow: kind {kind!r} is not allowed inline "
            f"(allowed: {', '.join(sorted(_COMPACT_KINDS))}); case/call/map reference "
            f"other nodes a one-node flow has none of — write a full `nodes:` map instead",
            line=lines.get("kind"),
        )
    flow_id = body.get("id")
    if not isinstance(flow_id, str) or not flow_id:
        raise LoadError(
            "compact (single-node) flow: a top-level `id:` is required — it names the "
            "single node",
            line=lines.get("id"),
        )
    params = body.get("input") or {}
    if not isinstance(params, dict):
        raise LoadError(
            "compact (single-node) flow: `input:` must be a param map (name: TYPE)",
            line=lines.get("input"),
        )

    # The node body = the kind + its logic fields (everything that is not flow metadata
    # or the special input:/output: keys). Auto-wire the flow inputs by name; carry the
    # flow `output:` through as the node's output TYPE.
    node_body = {
        k: v for k, v in body.items()
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


def parse_source_document(text: str, label: str) -> SourceDocument:
    """Parse one owned source document and translate syntax failure once."""

    try:
        return SourceDocument.parse(text, label)
    except SourceSyntaxError as exc:
        raise LoadError(f"invalid YAML: {exc}", range=exc.range) from exc


def flow_nodes_cursor(
    document: SourceDocument,
    nodes_section: dict[str, Any],
) -> SourceCursor:
    """Return the authored nodes cursor or a source-aligned compact projection.

    A full flow already owns a physical ``nodes:`` mapping. A compact flow has
    one synthesized node instead: its header is aligned with the authored
    ``id:`` key, unchanged logic fields retain their top-level sources, and
    generated input/output fields remain explicitly virtual.
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
        if key not in ("inputs", "outputs") and root.has(key):
            source = root.child(key)
        entries[key] = (value, source)

    header = root.child("id") if root.has("id") else root
    body_cursor = header.project_mapping(entries)
    return root.project_mapping({node_id: (node_body, body_cursor)})


def parse_file(source: str | SourceDocument) -> ComposeFile:
    """Parse a source document into a strict `ComposeFile`.

    Raises `LoadError` on malformed YAML, a non-mapping document, an unknown
    top-level key (after stripping `x-*`), or any schema violation.
    """
    document = (
        parse_source_document(source, "<flow>")
        if isinstance(source, str)
        else source
    )
    root = document.root
    raw = root._value_view()
    if not isinstance(raw, Mapping):
        raise LoadError("a flow must be a YAML mapping at the top level", range=root.range)

    lines = _section_lines(root)
    body = {
        key: root.get(key)
        for key in raw
        if not str(key).startswith("x-")
    }

    # Compact (single-node) flow: a top-level node `kind:` with no `nodes:` map. Desugar
    # to the canonical one-node form BEFORE the strict-schema checks below, since `nodes:`
    # is required and `extra="forbid"` would otherwise reject the inline node fields.
    if "kind" in body and "nodes" not in body:
        body = _desugar_compact(body, lines)

    # ComposeFile's model_validator(mode='before') normalizes section
    # keywords (input:↔inputs:, output:↔outputs:) at top + nested + defs bodies.
    # We pre-check unknown top-level keys here, allowing BOTH the new singular and the
    # legacy plural spellings.
    allowed_top = set(ComposeFile.model_fields) | {"input", "output"}
    for key in body:
        if key not in allowed_top:
            raise LoadError(
                f"unknown top-level key {key!r} "
                f"(allowed: {', '.join(sorted(set(ComposeFile.model_fields)))})",
                line=lines.get(key),
            )

    try:
        return ComposeFile.model_validate(body)
    except LoadError:
        raise
    except Exception as exc:  # pydantic ValidationError -> LoadError
        raise LoadError(str(exc)) from exc


# ---------- per-kind node descriptors ----------
#
# A `nodes:` entry is a keyed map (key = node id, no `id:` field) carrying a
# `kind:` plus FLAT kind-fields — the surface flattens `inputs:`/`outputs:`
# onto the node (vs the legacy spec's body wrappers, `spec/nodes.py`). These are
# DESCRIPTORS: the validated parsed shape per kind, each owned by its `nodes/<kind>.py`
# module (the frozen dataclass + its `parse`/`build`) and inheriting the common
# `NodeDescriptor` base (see `nodes/base.py`); this module only re-exports them.
# Building the runtime `Node` (`output_type` + `params` + flow `wiring`, `read_type` over
# `outputs:`) happens later in the loader; the descriptors keep `inputs`/`outputs`/`args`/
# `cases` raw (binding strings / type nodes).
#
# `node_name`/`depends_on`/`runs_after`/`env` are common to every kind (the `NodeDescriptor`
# base). `depends_on` and `runs_after` are RUN-ORDERING edges (no data flows): both gate on the
# source settling; `depends_on` co-skips the dependent if the source skipped, `runs_after` does
# not (the dependent still runs). See the loader edge pass.
