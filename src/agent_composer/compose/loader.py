"""The top-level loader: Compose YAML text -> a runnable `LoadedFlow`.

The integration that wires every prior slice into one `CompiledFlow`. The
post-parse pipeline is `_assemble`, shared by the top-level flow AND every in-file
`defs:` callable:

  parse -> typedefs registry -> [build defs] -> flow-input decls -> build
  leaf nodes + desugar cases -> infer data edges -> reconcile case edges
  -> validate references + classify asserts -> synthesize roots/terminals over
  ALL edges -> reject cycles + check if/else handles -> CompiledFlow.from_parts.

`call` resolution is **defs-first**: a `call:` names an in-file `defs:` callable (built
in-loader by the same `_assemble`) or, failing that, an external flow via the
injected `child_resolver` (`(flow_id, version) -> LoadedFlow`). `_make_call_resolver`
composes the two (and rejects recursive/mutual defs). `call` nodes are resolved-and-baked
at load: the loader derives the callable's signature (a plain call re-exports the single
codomain `Type`; a mapped call stamps `list[<codomain>]`; bindings are name/arity- and
type-checked against it) AND bakes the callable's compiled flow onto the built
`CallNode` so `run` drives the embedded child. A flow whose `call`s are all
in-file defs loads resolver-free; a `call` to a non-def callable without a resolver is
loud. `run_flow` stays resolver-free.

The flow `outputs:` section is mapped to `FlowOutput(name, from_)` carriers the way
`runtime.engine.terminal_output` consumes them: a name -> binding map keeps each name; a
bare whole-string binding (`${note.output}`) is one output named `result` (so
`len(outputs) == 1` -> `terminal_output` returns the bare resolved value).

Imports flow DOWN only: this module composes the compose package's own slices plus
`typesys`/`compile.model`; nothing in the engine imports it back.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import yaml

from agent_composer.compile.model import CompiledFlow, Edge, FlowOutput
from agent_composer.events import SourceRange
from agent_composer.nodes.base import Node
from agent_composer.typesys.values import TypeCheckError, Type
from agent_composer.typesys.types import read_typedefs
from agent_composer.compose.asserts import AssertSet, classify_asserts
from agent_composer.compose.build import (
    build_leaf_node,
    infer_data_edges,
    synthesize_boundary_graph,
)
from agent_composer.compose.kinds.resolver import ChildResolver
from agent_composer.compose.stages.inline_calls import _synth_id_gen, desugar_inline_calls
from agent_composer.compose.cases import (
    CaseDesugar,
    desugar_case,
    expand_case_outputs,
    reconcile_case_edges,
)
from agent_composer.compose.errors import LoadError
from agent_composer.compose.stages.adaptive_questions import desugar_adaptive_questions
from agent_composer.compose.stages.case_targets import desugar_case_call_targets
from agent_composer.compose.stages.cycles import reject_cycles
from agent_composer.compose.stages.human_questions_validate import (
    validate_human_questions,
)
from agent_composer.compose.stages.loop_contract import check_loop_type_contract
from agent_composer.compose.stages.ordering import infer_ordering_edges
from agent_composer.compose.stages.ref_map_types import check_ref_map_types
from agent_composer.compose.stages.references import validate_references
from agent_composer.compose.stages.wiring_parity import check_wiring_parity
from agent_composer.compose.uses import parse_uses_ref
from agent_composer.compose.reader import NodeReader
from agent_composer.compose.spanloader import load_with_spans
from agent_composer.compose.stages.parse import parse_nodes
from agent_composer.compose.parser import (
    CallDescriptor,
    CaseDescriptor,
    LoopDescriptor,
    parse_file,
)
from agent_composer.compose.types import InputDecl, read_flow_inputs
from agent_composer.compose.validate import (
    check_case_handles,
    validate_node_asserts,
)

# Name for the single output of a bare whole-string `outputs:` binding. With one
# `FlowOutput`, `terminal_output` returns the bare resolved value, so the name
# only labels the carrier.
_SINGLE_OUTPUT_NAME = "result"


def _nodes_reader(text: Optional[str], nodes_section: dict) -> NodeReader:
    """A `NodeReader` over a flow's `nodes:` mapping for `parse_nodes`.

    Span-aware over the source when `text` carries a `nodes:` mapping (the top-level flow);
    otherwise a spanless reader over `nodes_section` — a compact flow (its sole node is
    desugared in-memory, absent from source), a `defs:` sub-flow (nested, already
    normalized), or an uncomposable document.
    """
    if text:
        try:
            value, spans, root = load_with_spans(text, return_root=True)
        except yaml.YAMLError:
            pass
        else:
            if isinstance(value, dict) and isinstance(value.get("nodes"), dict):
                return NodeReader(value, spans, root).child("nodes")
    return NodeReader.spanless(nodes_section)


def _def_nodes_reader(parent_text: str, name: str) -> NodeReader:
    """A span-aware `NodeReader` over a `defs:<name>:nodes:` mapping in the parent file.

    A def is BUILT from a spanless reader (its nodes have no baked ranges), so the def's
    render-only `SourceFrame` sources its node/field ranges here — a separate span parse of
    the parent file, descended to `defs.<name>.nodes`, where the def's nodes physically live
    at their absolute lines. Spanless (empty ranges -> frame collapses to the call line) when
    `parent_text` does not compose or lacks that sub-tree (a single-node / compact def).
    """
    if parent_text:
        try:
            value, spans, root = load_with_spans(parent_text, return_root=True)
        except yaml.YAMLError:
            pass
        else:
            defs = value.get("defs") if isinstance(value, dict) else None
            body = defs.get(name) if isinstance(defs, dict) else None
            if isinstance(body, dict) and isinstance(body.get("nodes"), dict):
                return (
                    NodeReader(value, spans, root)
                    .child("defs").child(name).child("nodes")
                )
    return NodeReader.spanless({})


def _boundary_field_ranges(text: Optional[str]) -> dict[str, SourceRange]:
    """The flow-boundary source map stamped onto the top-level StartNode.

    Keys: each top-level `input:`/`inputs:` decl name -> its declaration value `SourceRange`,
    plus each flow-level `asserts:` expr string -> its `SourceRange`. Feeds StartNode.field_ranges
    so a boundary input-type failure (`input <name> — ...`) or a false boundary assert resolves a
    precise caret. Empty for a spanless / uncomposable flow (compact / in-memory / a def, whose
    `input:` lives under `defs.<name>` — best-effort, boundary errors there fall back to the node
    range). Mirrors `_frame_ranges`: render metadata, never raises."""
    if not text:
        return {}
    try:
        value, spans, root = load_with_spans(text, return_root=True)
    except yaml.YAMLError:
        return {}
    if not isinstance(value, dict):
        return {}
    reader = NodeReader(value, spans, root)
    out: dict[str, SourceRange] = {}
    for key in ("input", "inputs"):
        if isinstance(value.get(key), dict):
            out.update(reader.mapping_value_ranges(key))
    out.update(reader.item_ranges("asserts"))
    return out


def _section_lines(text: Optional[str]) -> dict[str, int]:
    """Map each top-level section key -> its 1-based header line (best-effort).

    Replaces the retired `parser.section_lines`: locates a section-level error (a bad
    `typedefs:`, an eager `uses:` failure, a dangling `outputs:`/`asserts:` ref) at the
    section's header line. Derived from the span parse (`load_with_spans` + each section
    key's own `SourceRange`), consistent with the range-native front end — raw keys are
    kept (`output`/`outputs` both surface; the loader coalesces). `{}` when the text is
    absent or PyYAML can't compose it. Render metadata, never the error path — never raises.
    """
    if not text:
        return {}
    try:
        value, spans, root = load_with_spans(text, return_root=True)
    except yaml.YAMLError:
        return {}
    if not isinstance(value, dict):
        return {}
    reader = NodeReader(value, spans, root)
    out: dict[str, int] = {}
    for key in value:
        rng = reader.key_range_of(key)
        if rng is not None:
            out[str(key)] = rng.start_line
    return out


def _frame_ranges(
    reader: NodeReader,
) -> tuple[dict[str, SourceRange], dict[str, dict[str, SourceRange]]]:
    """The `(node_ranges, field_ranges)` a `SourceFrame` carries, from a best-effort parse.

    Runs `parse_nodes(reader)` and pulls each descriptor's own `.range`/`.field_ranges` — the
    render-only location maps the CLI's call-traceback boxes. A spanless reader (a compact /
    single-node def) yields empty maps; a malformed `nodes:` (which the build path rejects with
    its own located error) is swallowed here to empty maps — this is render metadata, never the
    error path, so it must not raise.
    """
    try:
        descriptors = parse_nodes(reader)
    except LoadError:
        return {}, {}
    node_ranges: dict[str, SourceRange] = {}
    field_ranges: dict[str, dict[str, SourceRange]] = {}
    for nid, desc in descriptors.items():
        rng = getattr(desc, "range", None)
        if rng is not None:
            node_ranges[nid] = rng
        fr = getattr(desc, "field_ranges", None)
        if fr:
            field_ranges[nid] = dict(fr)
    return node_ranges, field_ranges


def top_frame_ranges(
    text: str,
) -> tuple[dict[str, SourceRange], dict[str, dict[str, SourceRange]]]:
    """Best-effort `(node_ranges, field_ranges)` for a top-level flow from its text alone.

    The `(node_id -> header `SourceRange`, node_id -> {field -> `SourceRange`})` maps a
    `LoadedFlow.source` already carries — recomputed here from `text` for the CLI's error
    renderer when it does NOT hold the loaded IR (e.g. a `RunResult` handed in on its own).
    Empty maps when `text` can't be parsed: this is render metadata, never the error path,
    so it must not raise.
    """
    try:
        cf = parse_file(text)
    except LoadError:
        return {}, {}
    return _frame_ranges(_nodes_reader(text, cf.nodes))


@dataclass(frozen=True)
class SourceFrame:
    """Render-only source-location metadata for one flow (a def or an external file).

    Carried on a `LoadedFlow` and copied onto the built `CallNode`/`MapNode` so the CLI
    can box a `.yaml` frame for the *child* a call descends into — the basis of the
    nested-error call traceback. Never read by the engine's `run()`; purely for error
    rendering.

    Attributes:
        label (`str`):
            The frame title — `"defs:<name>"` for an in-file callable, or the resolved
            filename (e.g. `"24-uses-external.yaml"`) for an external `uses:` flow.
        text (`str`):
            The YAML source whose lines the maps below index. For a def this is the
            **parent** file's text (the def's nodes live at absolute lines there); for an
            external flow it is that file's own text.
        node_ranges (`dict[str, SourceRange]`):
            `node_id -> the node body's `SourceRange` in `text`` for the child's nodes.
            Empty (frame collapses to the call line) for a compact single-node def with no
            authored inner `nodes:` mapping.
        field_ranges (`dict[str, dict[str, SourceRange]]`):
            `node_id -> {field -> `SourceRange`}` — the kind-fallback / `output:` precision
            used when the failing locator is `kind=field`.
    """

    label: str
    text: str
    node_ranges: dict[str, SourceRange]
    field_ranges: dict[str, dict[str, SourceRange]]

    def __deepcopy__(self, memo):
        # Frozen + render-only: every per-callsite `clone_child` deep-copies each child
        # node, which would otherwise copy the full YAML `text` per clone. Sharing the one
        # frozen instance is safe (it is never mutated) and avoids O(elements x filesize).
        return self


@dataclass(frozen=True)
class LoadedFlow:
    """
    A compiled, validated flow ready to run.

    Produced by [`load_flow`][agent_composer.load_flow] and consumed by
    [`run_flow`][agent_composer.run_flow] / the engine. Immutable so it can be reused
    across runs and resumes.

    Attributes:
        compiled (`CompiledFlow`):
            The runnable IR (nodes + edges + outputs + wiring) the `FlowEngine` executes.
        input (`list[InputDecl]`):
            The declared flow inputs `run_flow` coerces and defaults run arguments
            against. Singular for symmetry with the `input:` YAML keyword.
        asserts (`AssertSet`):
            The boundary / post-terminal assert split `run_flow` enforces in two phases
            (boundary before any node runs, post after the flow terminates).
        version (`str`, *optional*, defaults to `None`):
            The flow's declared `version:`, or `None` if unversioned. Used to validate a
            `uses: ref@<version>` pin against the resolved file.
        name (`str`, *optional*, defaults to `None`):
            The flow's declared `name:` — human-facing metadata (e.g. the CLI's
            interactive-prompt header). `None` for a `defs:` sub-flow, which has no name.
        description (`str`, *optional*, defaults to `None`):
            The flow's declared `description:`, or `None` if absent — human-facing
            metadata (e.g. the CLI's interactive-prompt header).
        source (`SourceFrame`, *optional*, defaults to `None`):
            Render-only source-location metadata for THIS flow (its own text + node-line
            maps). Set for the top-level flow, each `defs:` callable, and each external
            `uses:` file; copied onto the built `CallNode`/`MapNode` as `child_source` so
            the CLI can box a frame for the child a call descends into. `None` only when a
            flow was assembled without a known source (defensive).
    """

    compiled: CompiledFlow
    input: list[InputDecl]
    asserts: AssertSet
    version: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    source: Optional[SourceFrame] = None


def _flow_outputs(outputs) -> list[FlowOutput]:
    """Read the flow `outputs:` section into the `FlowOutput` carriers `terminal_output` reads.

    A name -> binding map becomes one `FlowOutput` per entry
    (>=2 -> `terminal_output` returns an object keyed by name); a bare whole-string
    binding (`${note.output}`) is one output named `result` (1 ->
    `terminal_output` returns the bare value). A `None` section -> no declared outputs
    (`terminal_output` returns `None`).

    Args:
        outputs (`dict | str | None`):
            The raw `outputs:` section: a name -> binding map, a single binding string,
            or `None`.
    """
    if outputs is None:
        return []
    if isinstance(outputs, dict):
        return [FlowOutput(name=str(name), from_=value) for name, value in outputs.items()]
    return [FlowOutput(name=_SINGLE_OUTPUT_NAME, from_=outputs)]


# A multi-node `defs:` callable is a sub-flow inline: the same sections as a top-level
# flow MINUS the file-scoped metadata + type registry (`id`/`name`/`typedefs` — a def
# shares the file's). A single-node form (a top-level `kind:`) is deferred.
# `asserts:` ARE allowed: the REF/MAP child seam enforces a def's two-phase asserts
# against the child pool (boundary before the child runs, post after it terminates).
_DEF_ALLOWED = frozenset({"inputs", "nodes", "outputs", "asserts", "llm_config"})


def _validate_system(system: dict) -> None:
    """Validate the top-level `system:` SECTION shape. Only `paths:` (a list of
    strings) is allowed — this is the resolution search path, NOT the strict
    `${system.X}` ambient namespace (a `${system.paths}` ref stays a compile error)."""
    extra = sorted(set(system) - {"paths"})
    if extra:
        raise LoadError(f"system: unknown key(s) {extra} (allowed: paths)")
    paths = system.get("paths", [])
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        raise LoadError("system: paths must be a list of strings")


@dataclass
class _LoadCtx:
    """Shared state threaded across the mutually-recursive `_load_flow` /
    `make_file_resolver` (the whole cross-file import graph): a `loading` set =
    cross-file cycle guard, a `cache` keyed by resolved abs-path = diamond reuse."""

    loading: set = field(default_factory=set)
    cache: dict = field(default_factory=dict)


def _join_path(base, p: str) -> Path:
    """A `system.paths` entry: relative -> joined onto the file's own dir; absolute as-is."""
    pp = Path(p)
    return pp if pp.is_absolute() else Path(base) / pp


def _check_version(child: "LoadedFlow", path: str, version: Optional[str]) -> None:
    """Raise if a `uses:` ref pins a version the resolved file does not declare.

    Args:
        child (`LoadedFlow`):
            The resolved child flow whose declared `version` is compared.
        path (`str`):
            The `uses:` path (for the error message).
        version (`str`, *optional*, defaults to `None`):
            The pinned `@<version>`; `None` means unpinned (no check).
    """
    if version is not None and child.version != version:
        raise LoadError(
            f"uses: ref pinned {path}@{version} but the resolved file declares "
            f"version: {child.version!r} (file-per-version: name a file that declares {version!r})"
        )


def make_file_resolver(search_paths, ctx: "_LoadCtx") -> ChildResolver:
    """
    Build the default local file resolver for `uses:` references.

    The returned `(path, version) -> LoadedFlow` callable resolves a ref to a `.yaml` on
    the search path (`<dir>/<path>.yaml`, first match wins); `@<version>` is never part of
    the filename but acts as a guard — the resolved file is loaded, then its declared
    `version:` must equal the pinned `<v>`. Children load recursively so their own
    `uses:`/`system.paths` re-root at the child's directory. A `hub:`/scheme ref never
    reaches here (it is handled in the composite resolver).

    Args:
        search_paths (`list[str | Path]`):
            Directories to search, in order; the first containing the target wins.
        ctx (`_LoadCtx`):
            Shared load context for the whole import graph — a cross-file cycle guard
            plus a diamond-reuse cache, keyed by resolved absolute path.

    Returns:
        `ChildResolver`:
            A `resolve(path, version=None) -> LoadedFlow` closure over `search_paths`/`ctx`.

    Raises:
        `LoadError`:
            From the returned closure on a missing target, a cross-file reference cycle,
            or a pinned-version mismatch.
    """

    def resolve(path: str, version: Optional[str] = None) -> LoadedFlow:
        rel = path  # version is a guard on the resolved file, never part of the filename
        for d in search_paths:
            cand = (Path(d) / (rel + ".yaml"))
            try:
                cand = cand.resolve()
            except OSError:
                continue
            if not cand.exists():
                continue  # skip a non-existent search dir / miss (sys.path-style)
            if cand in ctx.cache:
                child = ctx.cache[cand]
                _check_version(child, path, version)
                return child
            if cand in ctx.loading:
                raise LoadError(f"flow-reference cycle across files: {cand}")
            ctx.loading.add(cand)
            try:
                child = _load_flow(cand.read_text(), None, [cand.parent], ctx)
            finally:
                ctx.loading.discard(cand)
            # Relabel the external child's frame to its filename BEFORE caching, so every
            # diamond-reuse cache-hit also returns the filename label (not the file's own
            # `name:`). The label is value-stable across sites (same abs-path -> same name).
            if child.source is not None:
                child = replace(child, source=replace(child.source, label=cand.name))
            ctx.cache[cand] = child
            _check_version(child, path, version)
            return child
        raise LoadError(
            f"flow {rel!r} not found on search path {[str(d) for d in search_paths]}; "
            f"marketplace resolution needs a server — not available yet"
        )

    return resolve


def load_flow(
    text: str,
    *,
    child_resolver: Optional[ChildResolver] = None,
    search_paths: Optional[list] = None,
) -> LoadedFlow:
    """
    Load Compose YAML text into a runnable `LoadedFlow`.

    Parses, compiles, and validates the flow in one pass. A `call:` resolves
    **defs-first** (an in-file `defs:` callable), then a **`uses:` alias** (external, with
    the ref's `@<version>`), then a legacy bare name. With neither `search_paths` nor
    `child_resolver`, a `call:`/`uses:` to a non-def callable raises — external resolution
    is additive. The loader derives each callable's signature for the binding checks and
    bakes its compiled flow onto the built `CallNode` so `run` drives it.

    Args:
        text (`str`):
            The Compose-shaped YAML source of the flow.
        child_resolver (`ChildResolver`, *optional*, defaults to `None`):
            An explicit `(flow_id, version) -> LoadedFlow` resolver for external flows
            (a fake, or a future marketplace). Mutually preferred over `search_paths`
            when both are given.
        search_paths (`list[str | Path]`, *optional*, defaults to `None`):
            Directories the default local file resolver searches for `uses:` targets
            (`<dir>/<path>.yaml`). Pass the flow file's own directory so relative
            `call:`/`uses:` references resolve; the list is extended by the flow's
            `system: paths:`.

    Returns:
        `LoadedFlow`:
            The compiled, validated flow ready for `run_flow`.

    Raises:
        `LoadError`:
            On any malformed or inconsistent flow, located at the offending `.yaml`
            line where a slice supplies one.

    Example:
        ```python
        from agent_composer import load_flow

        loaded = load_flow(open("hello.yaml").read(), search_paths=["."])
        ```
    """
    return _load_flow(text, child_resolver, search_paths, _LoadCtx())


def _load_flow(text, child_resolver, search_paths, ctx: "_LoadCtx") -> LoadedFlow:
    f = parse_file(text)
    _validate_system(f.system)

    try:
        registry = read_typedefs(f.typedefs)
    except TypeCheckError as exc:
        # coarse: the typesys layer doesn't track source lines, so anchor at the
        # `typedefs:` section line (the box's context still shows the offending name).
        raise LoadError(
            f"bad typedefs: {exc}", line=_section_lines(text).get("typedefs")
        ) from exc

    # `system.paths` is honored only when the file's OWN dir is known (loaded WITH
    # search_paths, whose first entry is the file's dir). With no search_paths the own dir
    # is unknown -> system.paths is inert -> external stays None -> today's loud behavior.
    if search_paths:
        own = search_paths[0]
        effective = [Path(p) for p in search_paths] + [
            _join_path(own, p) for p in f.system.get("paths", [])
        ]
    else:
        effective = []

    if child_resolver is not None:
        external: Optional[ChildResolver] = child_resolver  # explicit fake / marketplace
    elif effective:
        external = make_file_resolver(effective, ctx)        # default local file resolver
    else:
        external = None                                      # no resolver -> non-def call: is loud

    # The composite call resolver: defs-first, then a `uses:` alias (external, with the
    # ref's version), then a legacy bare name. Building the defs eagerly surfaces a
    # broken/recursive callable at load (loud, not a hang). `text` is threaded so each
    # def's `SourceFrame` indexes its inner nodes at their absolute lines in THIS file.
    resolver = _make_call_resolver(f.defs, registry, external, f.uses, text)

    # Render-only source frame for THIS flow: copied onto child call/map nodes for the
    # nested-error traceback. For an external file (loaded via make_file_resolver) the
    # label is relabeled to the filename in `resolve`.
    top_node_ranges, top_field_ranges = _frame_ranges(_nodes_reader(text, f.nodes))
    source = SourceFrame(
        label=f.name or "<flow>",
        text=text,
        node_ranges=top_node_ranges,
        field_ranges=top_field_ranges,
    )

    return _assemble(
        inputs_section=f.inputs,
        nodes_reader=_nodes_reader(text, f.nodes),
        outputs_section=f.outputs,
        asserts_section=f.asserts,
        registry=registry,
        resolver=resolver,
        n_lines={nid: rng.start_line for nid, rng in top_node_ranges.items()},
        s_lines=_section_lines(text),
        uses_aliases=set(f.uses),
        version=f.version,
        name=f.name,
        description=f.description,
        flow_llm_config=f.llm_config,
        flow_env=f.env,
        source=source,
        boundary_field_ranges=_boundary_field_ranges(text),
    )


def _make_call_resolver(
    defs_section: dict, registry, external: Optional[ChildResolver], uses: dict,
    parent_text: str = "",
) -> ChildResolver:
    """A `(flow_id, version) -> LoadedFlow` resolver: **defs-first** (in-file `defs:`)
    → **`uses:` alias** (external, with the ref's `@<version>`). A bare `call:` that is
    NEITHER a `defs:` callable NOR a `uses:` alias is a `LoadError` — external flows are
    reachable ONLY through a `uses:` alias (no bare-name backdoor to `external`).
    Each callable builds once (memoized); an in-progress set detects recursive/mutual
    `defs:` references (-> `LoadError`, never a load hang). Every def is built eagerly so a
    broken/unused one is still loud at load (eager `uses:` resolution is in `_assemble`,
    after the node-id collision guard).

    `parent_text` is the source of the file declaring these `defs:` — threaded into each
    `_load_def` so a def's `SourceFrame` indexes its inner nodes at their absolute lines
    in that file (the nested-error traceback).

    `uses:` translates an alias to its `UsesRef`: a `hub:` scheme is the (deferred)
    marketplace, any other scheme is unknown, and a local ref is handed to `external` as
    `(path, version)`. A name in both `defs:` and `uses:` is loud (one callable namespace)."""
    both = sorted(set(defs_section) & set(uses))
    if both:
        raise LoadError(
            f"name(s) {both} declared in both defs: and uses: (a callable has one namespace)"
        )

    built: dict[str, LoadedFlow] = {}
    in_progress: set[str] = set()

    def resolve(flow_id: str, version: Optional[str] = None) -> LoadedFlow:
        if flow_id in built:
            return built[flow_id]
        if flow_id in defs_section:
            if flow_id in in_progress:
                raise LoadError(
                    f"recursive defs reference: callable {flow_id!r} is defined in terms "
                    f"of itself (the baked call graph must be finite)"
                )
            in_progress.add(flow_id)
            try:
                loaded = _load_def(flow_id, defs_section[flow_id], registry, resolve, parent_text)
            finally:
                in_progress.discard(flow_id)
            built[flow_id] = loaded
            return loaded
        if flow_id in uses:
            ref = parse_uses_ref(uses[flow_id])
            if ref.scheme == "hub":
                raise LoadError(
                    f"callable {flow_id!r} (uses: {uses[flow_id]!r}): marketplace not "
                    f"supported yet (hub: scheme)"
                )
            if ref.scheme is not None:
                raise LoadError(
                    f"callable {flow_id!r} (uses: {uses[flow_id]!r}): unknown scheme "
                    f"{ref.scheme!r}: (recognized: a local ref, or hub:)"
                )
            if external is None:
                raise LoadError(
                    f"callable {flow_id!r} (uses: {uses[flow_id]!r}) needs a search path "
                    f"or child_resolver to resolve (load with search_paths=[...])"
                )
            built[flow_id] = external(ref.path, ref.version)
            return built[flow_id]
        raise LoadError(
            f"callable {flow_id!r} is neither an in-file defs: callable nor a uses: "
            f"alias — external flows are reachable only through a uses: alias "
            f"(declare `uses: {{{flow_id}: <ref>}}`)"
        )

    for name in defs_section:
        resolve(name)  # eager: a broken/recursive def is loud at load, not at first call
    return resolve


def _load_def(name: str, body, registry, resolver: ChildResolver, parent_text: str = "") -> LoadedFlow:
    """Build one in-file `defs:` entry into a `LoadedFlow` (a callable / sub-flow inline).

    Supports the **multi-node** form only — `{inputs?, nodes, outputs?, asserts?}`,
    sharing the file's `typedefs:` registry + the composite resolver (so a def can call
    other defs / external flows). A **single-node** form (a top-level `kind:`) is deferred
    -> `LoadError`. `parent_text` is the source of the declaring file: the def's nodes
    live at absolute lines there, so its `SourceFrame` indexes them via
    `_frame_ranges(_def_nodes_reader(parent_text, name))` for the nested-error traceback."""
    if not isinstance(body, dict):
        raise LoadError(
            f"defs entry {name!r}: body must be a mapping, got {type(body).__name__}"
        )
    if "nodes" not in body:
        if "kind" in body:
            return _load_single_node_def(name, body, registry, resolver, parent_text)
        raise LoadError(
            f"defs entry {name!r}: a callable needs a `nodes:` sub-flow body "
            f"(inputs?/nodes/outputs?) OR a single-node `kind:` form"
        )
    extra = sorted(set(body) - _DEF_ALLOWED)
    if extra:
        raise LoadError(
            f"defs entry {name!r}: unknown field(s) {extra} "
            f"(allowed: {', '.join(sorted(_DEF_ALLOWED))})"
        )
    # The def's frame indexes its inner nodes at their absolute lines in the parent file.
    def_node_ranges, def_field_ranges = _frame_ranges(_def_nodes_reader(parent_text, name))
    source = SourceFrame(
        label="defs:" + name,
        text=parent_text,
        node_ranges=def_node_ranges,
        field_ranges=def_field_ranges,
    )
    try:
        return _assemble(
            inputs_section=body.get("inputs") or {},
            nodes_reader=NodeReader.spanless(body["nodes"]),
            outputs_section=body.get("outputs"),
            asserts_section=body.get("asserts") or [],  # enforced at the child seam
            registry=registry,
            resolver=resolver,
            n_lines={},
            s_lines={},
            flow_llm_config=body.get("llm_config") or {},
            flow_env=body.get("env") or {},
            source=source,
        )
    except LoadError as exc:
        # Name the def in any internal error. The line stays None (a nested def has
        # no isolated source text — the deferred line-mapping limitation). Don't re-prefix
        # an already-named nested-def error (keep the innermost, most specific def name).
        if str(exc).startswith("defs entry "):
            raise
        raise LoadError(f"defs entry {name!r}: {exc}", line=exc.line) from exc


def _load_single_node_def(name: str, body: dict, registry, resolver: ChildResolver, parent_text: str = "") -> LoadedFlow:
    """Build a SINGLE-node `defs:` callable — `let f params = <one node over params>`.

    The flat body is `{kind, <logic fields>, input?, output?}` (singular keys; the
    parser back-maps to the plural-keyed descriptor space). Convention (auto-wire by
    name): `input:` declares the def's PARAMETERS (name -> TYPE, like a flow signature),
    each auto-bound by name into the single node (`p = ${input.p}`); `output:` is the
    node's output TYPE (the codomain). Everything except `input:`/`output:` is the node
    body (kind + its logic fields + any node-local `asserts:`). It desugars to a one-node
    sub-flow and runs through the same `_assemble` as the multi-node form. Its sole node is
    synthesized in-memory (no authored inner `nodes:` mapping), so its `SourceFrame` maps
    are empty and the traceback frame collapses to the call line."""
    param_types = body.get("inputs") or {}
    if not isinstance(param_types, dict):
        raise LoadError(
            f"defs entry {name!r}: single-node `inputs:` must be a param map (name: TYPE)"
        )
    node_body = {k: v for k, v in body.items() if k not in ("inputs", "outputs")}
    node_body["inputs"] = {p: f"${{input.{p}}}" for p in param_types}  # auto-wire by name
    if "outputs" in body:
        node_body["outputs"] = body["outputs"]  # the node's output type = the def codomain
    source = SourceFrame(label="defs:" + name, text=parent_text, node_ranges={}, field_ranges={})
    try:
        return _assemble(
            inputs_section=param_types,
            nodes_reader=NodeReader.spanless({name: node_body}),
            outputs_section={"result": f"${{{name}.output}}"},  # single value re-export (node-first)
            asserts_section=[],
            registry=registry,
            resolver=resolver,
            n_lines={},
            s_lines={},
            source=source,
        )
    except LoadError as exc:
        if str(exc).startswith("defs entry "):
            raise
        raise LoadError(f"defs entry {name!r}: {exc}", line=exc.line) from exc


def _assemble(
    *,
    inputs_section: dict,
    nodes_reader: NodeReader,
    outputs_section,
    asserts_section: list,
    registry,
    resolver: ChildResolver,
    n_lines: dict,
    s_lines: dict,
    uses_aliases: set = frozenset(),
    version: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    flow_llm_config: Optional[dict] = None,
    flow_env: Optional[dict] = None,
    source: Optional[SourceFrame] = None,
    boundary_field_ranges: Optional[dict] = None,
) -> LoadedFlow:
    """Assemble parsed sections into a `LoadedFlow` — the post-parse pipeline shared by
    the top-level flow and every in-file def:

      flow-input decls -> build leaf + `call` nodes -> cross-flow type check
      -> desugar cases -> infer data edges -> reconcile case edges -> validate
      references + classify asserts -> synthesize roots/terminals -> reject
      cycles + check if/else handles -> CompiledFlow.from_parts.

    `resolver` is the composite call resolver (defs-first); `registry` is the file's
    `read_typedefs(...)`. `n_lines`/`s_lines` are the source maps — each node id / top-level
    section key -> its 1-based HEADER line (`{}` for a nested def, whose errors are unlocated).
    Inline-CODE traceback anchors are derived from each descriptor's own `code:` range."""
    # The flow's file label frames inline-CODE tracebacks; each inline node's content start
    # line is derived from its recorded `code:` range in the CODE build. Inert for reference-mode CODE.
    flow_label = source.label if source is not None else "<inline-code>"
    inputs = read_flow_inputs(inputs_section, registry)
    flow_inputs = {decl.name for decl in inputs}
    descriptors = parse_nodes(nodes_reader)

    # A `uses:` alias shares the callable namespace, so it must not collide with a node
    # id (the resolver only sees a bare name; the author conflates them). Checked here, where
    # node ids exist, and BEFORE eager resolution so the message is the precise collision.
    collide = sorted(uses_aliases & set(descriptors))
    if collide:
        raise LoadError(f"uses: alias {collide} collides with a node id")

    # Eager `uses:` resolution (mirrors eager defs): resolve every alias now so a
    # declared-but-uncalled broken/`hub:`/missing target is loud at load, not at first call.
    for alias in uses_aliases:
        try:
            resolver(alias)
        except LoadError as exc:
            if exc.line is None:
                exc.line = s_lines.get("uses")
            raise

    # Desugar inline `${ f(...) }` call expressions: each becomes a synth `call`
    # node (CallDescriptor, over=None) and its host binding is rewritten to
    # `${<synth>.output}`. Pure sugar — the rest of the pipeline treats the synth
    # nodes as ordinary `call` nodes (build/edges/validate/DAG run unchanged).
    mint = _synth_id_gen()  # ONE synth-id minter shared by both inline-call desugars (no collision)
    descriptors, outputs_section, asserts_section = desugar_inline_calls(
        descriptors, outputs_section, asserts_section=asserts_section,
        outputs_line=s_lines.get("output") or s_lines.get("outputs"), asserts_line=s_lines.get("asserts"), next_id=mint,
    )

    # Lower each `human_input` `adaptive_questions:` block into a synth composer agent
    # + the rewritten gate. A descriptor-level pass (the `desugar_inline_calls` precedent
    # above): runs BEFORE the build loop so the synth agent flows through build + prompt-
    # scope validation + ref wiring like any author-written agent.
    descriptors = desugar_adaptive_questions(descriptors)

    # `then:/else: ${call}`: a `case` branch target that is an inline call becomes a
    # synth `call` node, the then:/else: rewritten to its id. Runs BEFORE expand_case_outputs
    # (so the synth args + the rewritten then-targets are visible to it) and shares `mint`.
    descriptors = desugar_case_call_targets(descriptors, mint)

    # Expand `${<case>.output}` value refs to a coalesce over the case's branch targets.
    # Runs after the inline-call desugar (so synth-node bindings are covered too)
    # and before the build loop — downstream sees only ordinary branch-target refs.
    descriptors, outputs_section = expand_case_outputs(
        descriptors,
        outputs_section,
        asserts_section,
        node_lines=n_lines,
        outputs_line=s_lines.get("output") or s_lines.get("outputs"),
        asserts_line=s_lines.get("asserts"),
    )

    # Build leaf runtime nodes (agent/code/model/tool) + `call` nodes (REF/MAP, via the
    # resolver); cases desugar separately.
    # Flow-owned wiring (the split): `flow_wiring[node_id][param] -> source`. A `{}`
    # entry for EVERY node (params-bearing or not) so a relocated node never falls through and
    # the key-parity check can range over all nodes. Leaf + WAIT are populated here; REF/MAP
    # and CASE stay `{}` until relocated.
    leaf: dict[str, Node] = {}
    flow_wiring: dict[str, dict] = {}

    for nid, desc in descriptors.items():
        if isinstance(desc, CaseDescriptor):
            flow_wiring[nid] = {}  # case wiring relocated below
            continue
        if isinstance(desc, CallDescriptor):
            try:
                # Build through the descriptor's own `build` (call/map) so it stamps
                # `.range`/`.field_ranges` uniformly with the leaf path — the funnel's
                # concrete-range resolution needs a call/map node's `field_ranges` (e.g. an
                # `over:`/call-arg binding failure) just as much as a leaf's.
                node, wiring = desc.build(registry=registry, resolver=resolver)
            except LoadError as exc:  # locate a resolver failure at the call node's line
                if exc.line is None:
                    exc.line = n_lines.get(nid)
                raise
            leaf[nid] = node
            flow_wiring[nid] = wiring
        elif isinstance(desc, LoopDescriptor):
            try:
                node, wiring = desc.build(registry=registry, resolver=resolver)
            except LoadError as exc:  # locate a resolver / name-set / legality failure at the loop node's line
                if exc.line is None:
                    exc.line = n_lines.get(nid)
                raise
            leaf[nid] = node
            flow_wiring[nid] = wiring
        else:
            try:
                # Inline-CODE traceback anchors (content start line + `code:` field line)
                # are derived inside the CODE build from the descriptor's own `code:` range.
                node, wiring = build_leaf_node(
                    desc, registry, source_file=flow_label,
                )
            except LoadError as exc:  # locate a build failure (e.g. a bad type expr) at the node's line
                if exc.line is None:
                    exc.line = n_lines.get(nid)
                raise
            leaf[nid] = node
            flow_wiring[nid] = wiring
    # Stamp each built node's effective static config: the flow-level `env:` default merged
    # under the node's own `env:` (node override wins), flow-local (a called child flow has its
    # own flow env). The engine core never reads `Node.env` — only a node's own `run()` does.
    _flow_env = flow_env or {}
    for nid, node in leaf.items():
        node.env = {**_flow_env, **getattr(descriptors[nid], "env", {})}
    # producers: each built node's declared output_type (drives the case `on:` enum
    # exhaustiveness + the dotted-field walk in ref/assert validation).
    producers: dict[str, Type] = {
        nid: node.output_type
        for nid, node in leaf.items()
        if node.output_type is not None
    }

    # cross-flow type check: each `call` binding's source type vs the callable
    # input type (needs producers, so it runs after all leaf/call nodes are built).
    flow_input_types = {decl.name: decl.type for decl in inputs}
    check_ref_map_types(leaf, producers, flow_input_types, flow_wiring, n_lines)
    # loop `'a -> 'a` type contract (types): each carried field's seed-source Type vs the
    # body output's same-named field Type (needs producers, so it runs alongside the
    # cross-flow call/map type check above).
    check_loop_type_contract(leaf, producers, flow_input_types, flow_wiring, n_lines)

    # Desugar each `case` to an CaseNode + its data/control edges.
    desugars: dict[str, CaseDesugar] = {}
    for nid, desc in descriptors.items():
        if not isinstance(desc, CaseDescriptor):
            continue
        try:
            desugars[nid] = desugar_case(desc, producers)
        except LoadError as exc:  # locate a malformed/non-exhaustive case at the case node's line
            if exc.line is None:
                exc.line = n_lines.get(nid)
            raise

    nodes: dict[str, Node] = dict(leaf)
    for nid, desugar in desugars.items():
        nodes[nid] = desugar.node
        flow_wiring[nid] = desugar.wiring  # case (CaseNode) wiring relocated to the flow

    # flow.wiring/params key parity over EVERY node (catches an orphan/
    # missing source or a node absent from the wiring). Runs once the wiring is complete.
    check_wiring_parity(nodes, flow_wiring, n_lines)

    # Data edges — the projection of flow.wiring (leaf + WAIT) + descriptor sources
    # (REF/MAP/CASE) — reconciled with the case desugars' edges.
    data_edges = infer_data_edges(descriptors, flow_wiring)
    reconciled = reconcile_case_edges(data_edges, desugars)

    # Reference + assert validation over the built flow. Errors locate
    # at the node's line (a bad node-input ref) or the `outputs:` section line (a
    # dangling flow-output ref).
    validate_references(
        nodes,
        flow_inputs,
        producers,
        outputs_section,
        flow_wiring,
        node_lines=n_lines,
        outputs_line=s_lines.get("output") or s_lines.get("outputs"),
    )
    # human_input gates: structurally check a literal `questions:` list at load, and
    # pin a ref-form `questions: ${name}` to a declared input (the prompt-scope rule).
    validate_human_questions(nodes, node_lines=n_lines)
    # Node-local `asserts:` (per-node contract): validate node-locally + classify pre/post +
    # stamp onto each built leaf/call node. Shared by defs (a def's internal nodes too).
    validate_node_asserts(leaf, descriptors, n_lines)
    asserts = classify_asserts(asserts_section, flow_inputs, set(descriptors), producers)

    # Run-ordering edges: depends_on (co-skip) / runs_after (pure order). Folded in
    # BEFORE root synthesis so an ordering-only dependent is demoted from root, and into
    # the cycle check below so an ordering cycle is loud.
    ordering_edges = infer_ordering_edges(descriptors, node_ids := set(descriptors))

    # DAG assembly: synthesize the START_ID/END_ID boundary NODES + their edges over the
    # FULL reconciled body-edge set (the __start__/__end__ sentinels retire; the strings
    # persist as the boundary node ids). synthesize_boundary_graph mints the END_ID producer edges
    # (one per producer per output, keyed by output name, with the coalesce/optional stance)
    # + the START_ID root edges + (0-output) the bare START_ID->END_ID root edge,
    # and returns parity-clean START_ID/END_ID wiring.
    flow_outputs = _flow_outputs(outputs_section)
    structural = reconciled + ordering_edges
    boundary_nodes, boundary_edges, boundary_wiring = synthesize_boundary_graph(
        inputs, flow_outputs, node_ids, structural, boundary_field_ranges
    )
    nodes.update(boundary_nodes)
    flow_wiring.update(boundary_wiring)
    edges: list[Edge] = structural + boundary_edges

    # A cycle spans nodes; anchor the error on the first stuck node's range and highlight every
    # stuck node's line (transitional `n_lines`, retired at Step 8). START_ID/END_ID are acyclic
    # by construction (START_ID has no incoming, END_ID no outgoing).
    node_ranges = {
        nid: d.range for nid, d in descriptors.items() if getattr(d, "range", None)
    }
    reject_cycles(edges, node_ids, n_lines, node_ranges)
    check_case_handles(nodes, edges)

    compiled = CompiledFlow.from_parts(nodes, edges, outputs=flow_outputs, wiring=flow_wiring,
                                       flow_llm_config=flow_llm_config or {})
    return LoadedFlow(
        compiled=compiled, input=inputs, asserts=asserts,
        version=version, name=name, description=description, source=source,
    )
