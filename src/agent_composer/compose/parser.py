"""The section parser: Compose YAML text -> a strict `ComposeFile`.

This is the analogue of `manifest/parser.py`: `yaml.safe_load` -> wrap as a
strict Pydantic model -> raise `LoadError` on any violation. Three surface
rules shape it:

- **Top-level sections only** (`id`/`name`/`description?`/`inputs`/`nodes`/
  `outputs`/`asserts`/`typedefs`/`defs`/`uses`/`system`). `extra="forbid"` on the
  strict body makes a typo'd top-level key (e.g. `nodez:`) loud rather than silently
  dropped. (`uses:` = external callables; `system: paths:` = the resolution search
  path — NOT the strict `${system.X}` ambient namespace.)
- **`x-*` extension keys** at the top level are stripped before validation —
  custom metadata / YAML-anchor holders the engine ignores.
- **YAML anchors** (`&` / `*` / `<<:` merge) are expanded by PyYAML's
  `safe_load` *before* the strict schema sees the document.

`nodes` stays a raw `dict` here; per-kind node descriptors are a later step.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_composer.compose.errors import LoadError
from agent_composer.compile.model import END_ID, START_ID

# `AgentDescriptor`/`CodeDescriptor`/`ModelDescriptor`/`ToolDescriptor`/`WaitDescriptor`
# now live with their parse/build handles in `kinds/<kind>.py` (#70); re-export them here so
# `_KIND_SPECS`, the `NodeDescriptor` union, and existing importers keep working.
from agent_composer.compose.kinds.agent import AgentDescriptor
from agent_composer.compose.kinds.code import CodeDescriptor
from agent_composer.compose.kinds.model import ModelDescriptor
from agent_composer.compose.kinds.tool import ToolDescriptor
from agent_composer.compose.kinds.wait import WaitDescriptor
from agent_composer.compose.kinds.human_input import HumanInputDescriptor
from agent_composer.compose.kinds.case import CaseDescriptor
from agent_composer.compose.kinds.call import CallDescriptor
from agent_composer.compose.kinds.loop import LoopDescriptor

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
    # parent/CLI layers; `resolve_llm_cascade` bakes the effective dict per agent.
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


def _top_level_lines(text: str) -> dict[str, int]:
    """Best-effort map of top-level key -> 1-based source line (for loud errors).

    Returns {} if PyYAML can't compose the document (the strict validate below
    surfaces the real error); full source mapping is a later slice.
    """
    try:
        root = yaml.compose(text)
    except yaml.YAMLError:
        return {}
    if not isinstance(root, yaml.MappingNode):
        return {}
    lines: dict[str, int] = {}
    for key_node, _ in root.value:
        if isinstance(key_node, yaml.ScalarNode):
            lines[key_node.value] = key_node.start_mark.line + 1
    return lines


def section_lines(text: str) -> dict[str, int]:
    """Map each top-level section key -> its 1-based source line (best-effort).

    The public face of `_top_level_lines`, so the loader can locate a section-level
    error (e.g. a dangling flow-output ref at the `outputs:` line). Returns {} when
    PyYAML can't compose the document.
    """
    return _top_level_lines(text)


def parse_file(text: str) -> ComposeFile:
    """Parse Compose YAML text into a strict `ComposeFile`.

    Raises `LoadError` on malformed YAML, a non-mapping document, an unknown
    top-level key (after stripping `x-*`), or any schema violation.
    """
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise LoadError(f"invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise LoadError("a flow must be a YAML mapping at the top level")

    lines = _top_level_lines(text)
    body = {k: v for k, v in raw.items() if not str(k).startswith("x-")}

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
# DESCRIPTORS: the validated parsed shape per kind. Building the runtime `Node`
# (`output_type` + `params` + flow `wiring`, `read_type` over `outputs:`) happens
# later in the loader; here `inputs`/`outputs`/`args`/`cases` stay raw (binding strings / type nodes).
#
# Per-kind allowed fields mirror `spec/nodes.py:_validate_body_shape` in flat form
# (a field illegal for the kind is loud); `node_name`/`depends_on`/`runs_after` are common
# to every kind. `depends_on` and `runs_after` are RUN-ORDERING edges (no data flows): both
# gate on the source settling; `depends_on` co-skips the dependent if the source skipped,
# `runs_after` does not (the dependent still runs). See the loader edge pass.


NodeDescriptor = (
    AgentDescriptor
    | CodeDescriptor
    | ModelDescriptor
    | ToolDescriptor
    | CaseDescriptor
    | CallDescriptor
    | LoopDescriptor
    | HumanInputDescriptor
    | WaitDescriptor
)

# Common keys every kind accepts (besides `kind` itself).
_COMMON_FIELDS = frozenset({"node_name", "depends_on", "runs_after", "env"})

# Per-kind: (descriptor class, required flat fields, all allowed flat fields).
# `else` (a Python keyword) maps onto CaseDescriptor.else_.
_KIND_SPECS: dict[str, tuple[type, frozenset, frozenset]] = {
    "agent": (
        AgentDescriptor,
        frozenset(),
        frozenset({"inputs", "outputs", "prompt", "tools", "controls", "mode", "llm_config", "retries", "asserts"}),
    ),
    "code": (CodeDescriptor, frozenset({"code"}), frozenset({"code", "inputs", "outputs", "asserts"})),
    "model": (
        ModelDescriptor,
        frozenset({"model_id"}),
        frozenset({"model_id", "inputs", "outputs", "weights_uri", "runtime", "asserts"}),
    ),
    "tool": (ToolDescriptor, frozenset({"tool_id"}), frozenset({"tool_id", "args", "asserts"})),
    "case": (CaseDescriptor, frozenset({"cases"}), frozenset({"cases", "else", "on"})),
    # REF: a single application — `over:`/`parallel:` on `kind: call` is a LoadError (use `map`).
    "call": (
        CallDescriptor,
        frozenset({"call"}),
        frozenset({"call", "inputs", "asserts"}),
    ),
    # MAP: `List.map` — `over:` is required (the iteration source); `parallel:` overlaps elements.
    "map": (
        CallDescriptor,
        frozenset({"call", "over"}),
        frozenset({"call", "inputs", "over", "parallel", "asserts"}),
    ),
    # LOOP: a `('a -> 'a) -> 'a -> 'a` driver — `call:` is the body subflow; `inputs:` seed the
    # carried record; `while`/`until`/`times` carry the predicate (remapped to `while_`/`until_`);
    # `max` is the runaway guard.
    "loop": (
        LoopDescriptor,
        frozenset({"call"}),
        frozenset({"call", "inputs", "while", "until", "times", "max", "asserts"}),
    ),
    # `adaptive_questions:` is a nested block; its inner keys (prompt/mode/llm_config/
    # retries) live INSIDE the block, so they need no top-level allow-list entry.
    "human_input": (
        HumanInputDescriptor,
        frozenset(),
        frozenset({"prompt", "questions", "adaptive_questions", "inputs", "outputs", "asserts"}),
    ),
    "wait": (WaitDescriptor, frozenset({"until"}), frozenset({"until"})),
}


# YAML 1.1 (PyYAML `safe_load`) coerces the bare key `on:` to the boolean `True`
# (likewise `off`/`yes`/`no`). The case-node surface key `on:` is the only such
# key in a node body, so we map the coerced key back to its string form.
_YAML_BOOL_KEYS = {True: "on", False: "off"}

# Surface keys that collide with Python keywords map onto a trailing-underscore field on
# the descriptor: `else:`->`else_` (case), `while:`/`until:`->`while_`/`until_` (loop).
_KEYWORD_FIELD_REMAP = {"else": "else_", "while": "while_", "until": "until_"}


def _normalize_keys(body: dict) -> dict:
    """Undo YAML 1.1 boolean-key coercion for the surface key `on:`."""
    if not any(k in _YAML_BOOL_KEYS for k in body):
        return body
    return {_YAML_BOOL_KEYS.get(k, k): v for k, v in body.items()}


def _parse_node(node_id: str, body: Any, line: Optional[int]) -> NodeDescriptor:
    """Read one keyed-map node body into its typed per-kind descriptor."""
    if not isinstance(body, dict):
        raise LoadError(
            f"node {node_id!r}: body must be a mapping, got {type(body).__name__}",
            line=line,
        )
    if "#" in node_id or "/" in node_id:
        raise LoadError(
            f"node id {node_id!r} uses a reserved separator ('#'/'/' are reserved for "
            f"runtime graph expansion)",
            line=line,
        )
    if node_id in (START_ID, END_ID):
        raise LoadError(
            f"node id {node_id!r} is reserved for the synthesized START_ID/END_ID boundary"
            f"; authors write `input:`/`output:`, not a boundary node",
            line=line,
        )
    # Reserve the four singular head literals (input/output/system/item) so the
    # node-first ref `${<node>.output.k}` is unambiguous. Reserve the plural forms
    # (inputs/outputs) too as typo-catchers: combined with the _classify_path typo
    # hint post-migration, this makes migration typos loudly self-correcting.
    if node_id in ("input", "output", "system", "item"):
        raise LoadError(
            f"node id {node_id!r} is reserved (used as `${{{node_id}.X}}` as a "
            f"resolver head — pick a different node id)",
            line=line,
        )
    if node_id in ("inputs", "outputs"):
        raise LoadError(
            f"node id {node_id!r} is reserved (the plural head is retired — "
            f"rename to `{node_id[:-1]}` or pick a different node id)",
            line=line,
        )
    body = _normalize_keys(body)
    # Normalize the singular section keywords (`input:`/`output:` → back-mapped
    # to `inputs:`/`outputs:` plural that the descriptor dataclasses still use).
    # Direct callers of `_parse_node` (`parse_nodes`) get the same treatment as
    # `parse_file` does at the top level. The top-level rejector is in parse_file;
    # per-node we only run the back-map (legacy is already gone by the time we get here).
    body = _back_map_to_plural(body)
    kind = body.get("kind")
    if kind is None:
        raise LoadError(f"node {node_id!r}: missing `kind`", line=line)
    spec = _KIND_SPECS.get(kind)
    if spec is None:
        raise LoadError(
            f"node {node_id!r}: unknown kind {kind!r} "
            f"(allowed: {', '.join(sorted(_KIND_SPECS))})",
            line=line,
        )
    cls, required, allowed = spec
    allowed_keys = allowed | _COMMON_FIELDS | {"kind"}

    for key in body:
        if key not in allowed_keys:
            raise LoadError(
                f"node {node_id!r} (kind={kind}): field {key!r} is not allowed "
                f"(allowed: {', '.join(sorted(allowed_keys))})",
                line=line,
            )
    for key in required:
        if key not in body:
            raise LoadError(
                f"node {node_id!r} (kind={kind}): missing required field {key!r}",
                line=line,
            )
    # human_input legality: exactly one question surface, and an adaptive block needs
    # its own `prompt:` (the LLM brief). `required` is empty for this kind because the
    # choice is one-of-three, not a single mandatory field.
    if kind == "human_input":
        has_q = body.get("questions") is not None
        auto = body.get("adaptive_questions")
        if body.get("prompt") is None and not has_q and auto is None:
            raise LoadError(
                f"node {node_id!r} (kind=human_input): needs `prompt:`, "
                f"`questions:`, or `adaptive_questions:`",
                line=line,
            )
        if has_q and auto is not None:
            raise LoadError(
                f"node {node_id!r} (kind=human_input): `questions:` and "
                f"`adaptive_questions:` are mutually exclusive",
                line=line,
            )
        if auto is not None:
            if not isinstance(auto, dict) or not auto.get("prompt"):
                raise LoadError(
                    f"node {node_id!r} (kind=human_input): `adaptive_questions:` "
                    f"requires a `prompt:` (the LLM brief)",
                    line=line,
                )
    # `asserts:` is a list of boolean-expression strings (the generic field-copy below does
    # no type-check, unlike the Pydantic-validated top-level `asserts`).
    asserts = body.get("asserts")
    if asserts is not None and not (
        isinstance(asserts, list) and all(isinstance(a, str) for a in asserts)
    ):
        raise LoadError(
            f"node {node_id!r} (kind={kind}): `asserts:` must be a list of strings",
            line=line,
        )

    kwargs: dict[str, Any] = {"id": node_id}
    # The REF/MAP discriminator: `kind: call|map` both build a CallDescriptor; `build` branches
    # on `.kind`. Every other kind has a 1:1 descriptor class, so `kind` is informational there.
    if cls is CallDescriptor:
        kwargs["kind"] = kind
    _cls_fields = {f.name for f in fields(cls)}
    for key in allowed:
        if key not in body:
            continue
        # Python-keyword surface keys map onto the descriptor's trailing-underscore field:
        # `else`->`else_` (case), `while`->`while_` / `until`->`until_` (loop). Only remap
        # when the keyword form is NOT itself a field — `wait` keeps its literal `until` field.
        remapped = _KEYWORD_FIELD_REMAP.get(key, key)
        kwargs[remapped if key not in _cls_fields else key] = body[key]
    if "node_name" in body:
        kwargs["node_name"] = body["node_name"]
    if "depends_on" in body:
        kwargs["depends_on"] = body["depends_on"]
    if "runs_after" in body:
        kwargs["runs_after"] = body["runs_after"]
    if "env" in body:
        # `env:` is a static config map (`{str -> literal}`) the node's own run() reads.
        # Validate the shape here (string keys); per-key type/meaning is the node's concern.
        env = body["env"]
        if not isinstance(env, dict) or not all(isinstance(k, str) for k in env):
            raise LoadError(
                f"node {node_id!r} (kind={kind}): `env:` must be a mapping with string keys",
                line=line,
            )
        kwargs["env"] = env
    return cls(**kwargs)


def parse_nodes(
    raw_nodes: dict[str, Any],
    node_lines: Optional[dict[str, int]] = None,
) -> dict[str, NodeDescriptor]:
    """Parse a `nodes:` mapping into typed per-kind descriptors keyed by node id.

    `node_lines` (node id -> 1-based source line) lets errors locate at the
    offending node's `.yaml` line; pass `_node_lines(text)` (or the ComposeFile's
    source map) for located errors.
    """
    lines = node_lines or {}
    out: dict[str, NodeDescriptor] = {}
    for node_id, body in (raw_nodes or {}).items():
        out[node_id] = _parse_node(node_id, body, lines.get(node_id))
    return out


def node_lines(text: str) -> dict[str, int]:
    """Map each `nodes:` entry's id -> 1-based source line (best-effort).

    Mirrors `_top_level_lines`: composes the document, descends into the `nodes:`
    mapping, and records each node-key's line. Returns {} if PyYAML can't compose.
    """
    return _node_lines_of(_nodes_mapping(text))


def _find_mapping_child(mapping, name: str):
    """The MappingNode value of key `name` inside `mapping`, or None.

    Shared scan over a composed MappingNode's `(key, value)` pairs. Returns None when
    `mapping` is None/not a mapping or `name` is absent / not itself a mapping.
    """
    if not isinstance(mapping, yaml.MappingNode):
        return None
    for key_node, value_node in mapping.value:
        if (
            isinstance(key_node, yaml.ScalarNode)
            and key_node.value == name
            and isinstance(value_node, yaml.MappingNode)
        ):
            return value_node
    return None


def _node_lines_of(nodes) -> dict[str, int]:
    """Map node id -> 1-based line for the entries of a `nodes:` MappingNode.

    The shared extraction behind `node_lines` (top-level) and `def_node_lines` (each
    def's inner `nodes:`). Returns {} when `nodes` is None.
    """
    if not isinstance(nodes, yaml.MappingNode):
        return {}
    return {
        k.value: k.start_mark.line + 1
        for k, _ in nodes.value
        if isinstance(k, yaml.ScalarNode)
    }


def _node_field_lines_of(nodes) -> dict[str, dict[str, int]]:
    """Map node id -> {field name -> 1-based line} for a `nodes:` MappingNode.

    The shared extraction behind `node_field_lines` (top-level) and
    `def_node_field_lines` (each def's inner `nodes:`). The legacy plural `outputs:`
    spelling is aliased to `output` so the locator key stays canonical. Returns {}
    when `nodes` is None.
    """
    if not isinstance(nodes, yaml.MappingNode):
        return {}
    out: dict[str, dict[str, int]] = {}
    for nid, body in nodes.value:
        if not (isinstance(nid, yaml.ScalarNode) and isinstance(body, yaml.MappingNode)):
            continue
        fields = {
            fk.value: fk.start_mark.line + 1
            for fk, _ in body.value
            if isinstance(fk, yaml.ScalarNode)
        }
        if "outputs" in fields:
            fields.setdefault("output", fields["outputs"])
        out[nid.value] = fields
    return out


def _nodes_mapping(text: str):
    """The top-level `nodes:` MappingNode of a composed flow, or None (best-effort).

    Shared by the sub-line maps below; returns None when PyYAML can't compose the
    document, the root isn't a mapping, or there is no `nodes:` mapping (a compact
    single-node flow has its node desugared in-memory, not in source).
    """
    try:
        root = yaml.compose(text)
    except yaml.YAMLError:
        return None
    return _find_mapping_child(root, "nodes")


def _defs_mapping(text: str):
    """The top-level `defs:` MappingNode of a composed flow, or None (best-effort).

    Returns None when PyYAML can't compose, the root isn't a mapping, or there is no
    `defs:` mapping (a flow with no in-file callables).
    """
    try:
        root = yaml.compose(text)
    except yaml.YAMLError:
        return None
    return _find_mapping_child(root, "defs")


def def_node_lines(text: str) -> dict[str, dict[str, int]]:
    """Map each `defs:<name>` -> {inner node id -> 1-based line} (best-effort).

    Descends the top-level `defs:` mapping and, for every def entry carrying its own
    `nodes:` mapping, records its inner node ids. A compact single-node def (a
    top-level `kind:` with no `nodes:`) is absent from the result: its sole node is
    synthesized in-memory at load, so there is no authored inner line to point at.
    Defs are syntactically flat (a def referenced only by another def still lives
    under `defs:<name>:nodes:`), so every callable is indexed. Returns {} when
    uncomposable or there is no `defs:` section.
    """
    defs = _defs_mapping(text)
    if defs is None:
        return {}
    out: dict[str, dict[str, int]] = {}
    for name_node, body in defs.value:
        if not (isinstance(name_node, yaml.ScalarNode) and isinstance(body, yaml.MappingNode)):
            continue
        nodes = _find_mapping_child(body, "nodes")
        if nodes is not None:
            out[name_node.value] = _node_lines_of(nodes)
    return out


def def_node_field_lines(text: str) -> dict[str, dict[str, dict[str, int]]]:
    """Map each `defs:<name>` -> {inner node id -> {field -> 1-based line}}.

    The def-internal parallel of `node_field_lines` — used to point the error frame
    at a specific field (e.g. `output:`) of a node inside a def. Same compact-def
    omission and flat-defs coverage as `def_node_lines`. Returns {} when uncomposable
    or there is no `defs:` section.
    """
    defs = _defs_mapping(text)
    if defs is None:
        return {}
    out: dict[str, dict[str, dict[str, int]]] = {}
    for name_node, body in defs.value:
        if not (isinstance(name_node, yaml.ScalarNode) and isinstance(body, yaml.MappingNode)):
            continue
        nodes = _find_mapping_child(body, "nodes")
        if nodes is not None:
            out[name_node.value] = _node_field_lines_of(nodes)
    return out


def _section_mapping(text: str, *names: str):
    """The first top-level `<name>:` MappingNode among `names`, or None (best-effort).

    Used for `input:`/`inputs:` (the singular/plural spelling pair). Returns None when
    the document can't be composed or no listed section is a mapping.
    """
    try:
        root = yaml.compose(text)
    except yaml.YAMLError:
        return None
    if not isinstance(root, yaml.MappingNode):
        return None
    for name in names:
        for key_node, value_node in root.value:
            if (
                isinstance(key_node, yaml.ScalarNode)
                and key_node.value == name
                and isinstance(value_node, yaml.MappingNode)
            ):
                return value_node
    return None


def node_input_lines(text: str) -> dict[str, dict[str, int]]:
    """Map node id -> {input key -> 1-based source line} for each node's `input:` mapping.

    Locates a specific input *binding* line (e.g. a `:?` ref that fired). Reads both
    the `input:` and legacy `inputs:` spelling. Best-effort: {} when uncomposable, and
    a node with no input mapping is simply absent.
    """
    nodes = _nodes_mapping(text)
    if nodes is None:
        return {}
    out: dict[str, dict[str, int]] = {}
    for nid, body in nodes.value:
        if not (isinstance(nid, yaml.ScalarNode) and isinstance(body, yaml.MappingNode)):
            continue
        for fk, fv in body.value:
            if (
                isinstance(fk, yaml.ScalarNode)
                and fk.value in ("input", "inputs")
                and isinstance(fv, yaml.MappingNode)
            ):
                out[nid.value] = {
                    ik.value: ik.start_mark.line + 1
                    for ik, _ in fv.value
                    if isinstance(ik, yaml.ScalarNode)
                }
    return out


def node_field_lines(text: str) -> dict[str, dict[str, int]]:
    """Map node id -> {field name -> 1-based source line} for each node's direct fields.

    The kind-fallback source: when no precise sub-line is determinable the CLI points
    at the node's best field (e.g. `code:` for a CODE node). The `field` locator also
    resolves here (e.g. a wrong-type-output box points at `output:`); the legacy plural
    `outputs:` spelling is aliased to `output` so the locator key stays canonical.
    Best-effort: {} when uncomposable.
    """
    return _node_field_lines_of(_nodes_mapping(text))


def code_content_lines(text: str) -> dict[str, int]:
    """Map each CODE node's id -> the 1-based line where its `code:` block content begins.

    For an inline `code: |` node this is the line AFTER the block indicator (where the
    first source character sits) — the anchor the node uses to frame inline tracebacks at
    absolute YAML lines (leading-newline padding). PyYAML reports a block scalar's
    `start_mark` at the indicator line, and its content always begins on the next line, so
    the value is `start_mark.line + 2` (0-based -> 1-based content line).

    Only block-style scalars (`|` / `>`) are captured — a plain `module:function`
    reference is a plain scalar with no block content to frame and is skipped. Best-effort:
    {} when uncomposable; a node absent from the result frames body-relative (line 1).
    """
    nodes = _nodes_mapping(text)
    if nodes is None:
        return {}
    out: dict[str, int] = {}
    for nid, body in nodes.value:
        if not (isinstance(nid, yaml.ScalarNode) and isinstance(body, yaml.MappingNode)):
            continue
        for fk, fv in body.value:
            if (
                isinstance(fk, yaml.ScalarNode)
                and fk.value == "code"
                and isinstance(fv, yaml.ScalarNode)
                and fv.style in ("|", ">")
            ):
                out[nid.value] = fv.start_mark.line + 2
    return out


def assert_lines(text: str) -> dict[tuple[Optional[str], str], int]:
    """Map (node id | None, assert expr) -> 1-based source line.

    Covers BOTH the flow top-level `asserts:` list (key `(None, expr)`) and each
    node's `asserts:` list (key `(node_id, expr)`). There is no singular `assert:`
    surface; the key is the `(node, expr)` pair so identical assert strings on
    different nodes resolve distinctly. Best-effort: {} when uncomposable.
    """
    out: dict[tuple[Optional[str], str], int] = {}

    def _scan(seq, node_id: Optional[str]) -> None:
        if not isinstance(seq, yaml.SequenceNode):
            return
        for item in seq.value:
            if isinstance(item, yaml.ScalarNode):
                out[(node_id, item.value)] = item.start_mark.line + 1

    try:
        root = yaml.compose(text)
    except yaml.YAMLError:
        return {}
    if not isinstance(root, yaml.MappingNode):
        return {}
    for key_node, value_node in root.value:
        if isinstance(key_node, yaml.ScalarNode) and key_node.value == "asserts":
            _scan(value_node, None)
    nodes = _nodes_mapping(text)
    if nodes is not None:
        for nid, body in nodes.value:
            if not (isinstance(nid, yaml.ScalarNode) and isinstance(body, yaml.MappingNode)):
                continue
            for fk, fv in body.value:
                if isinstance(fk, yaml.ScalarNode) and fk.value == "asserts":
                    _scan(fv, nid.value)
    return out


def input_decl_lines(text: str) -> dict[str, int]:
    """Map flow input name -> 1-based source line for the top-level `input:` mapping.

    Locates the declaration of an input that failed coercion at the run boundary.
    Reads both `input:` and legacy `inputs:`. Best-effort: {} when uncomposable.
    """
    m = _section_mapping(text, "input", "inputs")
    if m is None:
        return {}
    return {
        k.value: k.start_mark.line + 1
        for k, _ in m.value
        if isinstance(k, yaml.ScalarNode)
    }
