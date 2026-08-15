"""The top-level loader: Compose YAML text -> a runnable `ComposedFlow`.

The file-I/O front door: parse the YAML, build the typedefs registry, compose the
call resolver, load every `defs:` callable, and hand the parsed sections to the
post-parse conductor (`build.pipeline._assemble`, imported back here) which returns
the `ComposedFlow`. This module owns the span readers (`_nodes_reader`/`_def_nodes_reader`/
`_frame_ranges`/`top_frame_ranges`), the resolver composition (`make_file_resolver`/
`_CallableScope`), and the defs loaders (`_load_def`/`_load_single_node_def`) —
not the assembly pipeline itself.

`call` resolution is **defs-first**: a `call:` names an in-file `defs:` callable (built
in-loader by the same `_assemble`) or a `uses:` alias resolved through the injected
`child_resolver` (`(flow_id, version) -> ComposedFlow`). `_CallableScope`
composes the two (and rejects recursive/mutual defs). `call` nodes are resolved-and-baked
at load: the loader derives the callable's signature (a plain call re-exports the single
codomain `Type`; a mapped call stamps `list[<codomain>]`; bindings are name/arity- and
type-checked against it) AND bakes the callable's compiled flow onto the built
`CallNode` so `run` drives the embedded child. A flow whose `call`s are all
in-file defs loads resolver-free; a `call` to a non-def callable without a resolver is
loud. `run_flow` stays resolver-free.

Imports flow DOWN only: this module composes the compose package's own slices plus
`typesys`/`compile.model`; nothing in the engine imports it back.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

import yaml

from agent_composer.events import SourceRange
from agent_composer.nodes.base import FieldRanges
from agent_composer.typesys.values import TypeCheckError
from agent_composer.typesys.types import read_typedefs
from agent_composer.compose.build.pipeline import _assemble, ComposedFlow
from agent_composer.compose.desugar.inline_bodies import INLINE_FLOW_PREFIX
from agent_composer.compose.nodes.resolver import ChildResolver
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.helpers.uses import parse_uses_ref
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.compose.helpers.spanloader import load_with_spans
from agent_composer.compose.helpers.source import (
    SourceFrame,
    _boundary_assert_ranges,
    _boundary_field_ranges,
    _section_lines,
)
from agent_composer.compose.parse.nodes import parse_nodes
from agent_composer.compose.parse.file import (
    parse_file,
)


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

    Descends a fresh span parse to the raw author surface, where nested node keys remain
    singular and ranges retain their absolute parent-file lines. Spanless when `parent_text`
    does not compose or lacks that subtree (a programmatic or single-node def).
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


def _restore_node_surface(nodes):
    """Restore normalized node signatures to singular author-surface keys.

    The normalized `ComposeFile` representation uses `inputs`/`outputs`. A spanless def
    fallback must look like raw YAML before inline-body extraction, so only each direct
    node body's signature keys are restored. Recursion follows an actual nested `nodes:`
    graph and deliberately does not descend into arbitrary literal mappings or lists.
    """
    if not isinstance(nodes, dict):
        return nodes

    restored: dict = {}
    for node_id, raw_body in nodes.items():
        if not isinstance(raw_body, dict):
            restored[node_id] = raw_body
            continue
        body = dict(raw_body)
        if "inputs" in body:
            body["input"] = body.pop("inputs")
        if "outputs" in body:
            body["output"] = body.pop("outputs")
        if isinstance(body.get("nodes"), dict):
            body["nodes"] = _restore_node_surface(body["nodes"])
        restored[node_id] = body
    return restored


def _def_assembly_reader(
    parent_text: str,
    name: str,
    normalized_nodes,
) -> NodeReader:
    """Return a fresh raw def reader, or a surface-restored spanless fallback."""
    located = _def_nodes_reader(parent_text, name)
    if located.yaml_node is not None:
        return located
    return NodeReader.spanless(_restore_node_surface(normalized_nodes))


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
    # `node_id -> FieldRanges` — the OUTER map is keyed by a bare node id (correct: that is not
    # a locator); each INNER map keeps its `(kind, key)` guard, since the CLI renderer looks up
    # through it and is where a bare-key lookup would silently miss.
    field_ranges: dict[str, FieldRanges] = {}
    for nid, desc in descriptors.items():
        rng = getattr(desc, "range", None)
        if rng is not None:
            node_ranges[nid] = rng
        fr = getattr(desc, "field_ranges", None)
        if fr:
            field_ranges[nid] = FieldRanges(fr)
    return node_ranges, field_ranges


def top_frame_ranges(
    text: str,
) -> tuple[dict[str, SourceRange], dict[str, dict[str, SourceRange]]]:
    """Best-effort `(node_ranges, field_ranges)` for a top-level flow from its text alone.

    The `(node_id -> header `SourceRange`, node_id -> {field -> `SourceRange`})` maps a
    `ComposedFlow.source` already carries — recomputed here from `text` for the CLI's error
    renderer when it does NOT hold the loaded IR (e.g. a `RunResult` handed in on its own).
    Empty maps when `text` can't be parsed: this is render metadata, never the error path,
    so it must not raise.
    """
    try:
        cf = parse_file(text)
    except LoadError:
        return {}, {}
    return _frame_ranges(_nodes_reader(text, cf.nodes))


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


def _reject_authored_private_callable_names(text: str, defs: dict, uses: dict) -> None:
    """Reject file-scoped callable declarations in the inline-flow namespace.

    The inexpensive declaration scan runs before any other loader work. Source
    spans are parsed only after a reserved name is found, so valid files do not
    pay for another document parse at this boundary.
    """
    for section, declarations in (("defs", defs), ("uses", uses)):
        offender = next(
            (
                name
                for name in declarations
                if isinstance(name, str) and name.startswith(INLINE_FLOW_PREFIX)
            ),
            None,
        )
        if offender is None:
            continue

        declaration_range = None
        try:
            value, spans, root = load_with_spans(text, return_root=True)
        except yaml.YAMLError:
            pass
        else:
            raw_declarations = value.get(section) if isinstance(value, dict) else None
            if isinstance(raw_declarations, dict):
                declaration_range = (
                    NodeReader(value, spans, root)
                    .child(section)
                    .key_range_of(offender)
                )

        noun = "callable names" if section == "defs" else "aliases"
        raise LoadError(
            f"{section}: {noun} with prefix {INLINE_FLOW_PREFIX!r} are reserved "
            "for compiler-generated inline child flows",
            range=declaration_range,
        )


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


def _check_version(child: "ComposedFlow", path: str, version: Optional[str]) -> None:
    """Raise if a `uses:` ref pins a version the resolved file does not declare.

    Args:
        child (`ComposedFlow`):
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

    The returned `(path, version) -> ComposedFlow` callable resolves a ref to a `.yaml` on
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
            A `resolve(path, version=None) -> ComposedFlow` closure over `search_paths`/`ctx`.

    Raises:
        `LoadError`:
            From the returned closure on a missing target, a cross-file reference cycle,
            or a pinned-version mismatch.
    """

    def resolve(path: str, version: Optional[str] = None) -> ComposedFlow:
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
) -> ComposedFlow:
    """
    Load Compose YAML text into a runnable `ComposedFlow`.

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
            An explicit `(flow_id, version) -> ComposedFlow` resolver for external flows
            (a fake, or a future marketplace). Mutually preferred over `search_paths`
            when both are given.
        search_paths (`list[str | Path]`, *optional*, defaults to `None`):
            Directories the default local file resolver searches for `uses:` targets
            (`<dir>/<path>.yaml`). Pass the flow file's own directory so relative
            `call:`/`uses:` references resolve; the list is extended by the flow's
            `system: paths:`.

    Returns:
        `ComposedFlow`:
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


def _load_flow(text, child_resolver, search_paths, ctx: "_LoadCtx") -> ComposedFlow:
    f = parse_file(text)
    _reject_authored_private_callable_names(text, f.defs, f.uses)
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
    # ref's version); an undeclared bare name is loud. Building the defs eagerly surfaces a
    # broken/recursive callable at load (loud, not a hang). `text` is threaded so each
    # def's `SourceFrame` indexes its inner nodes at their absolute lines in THIS file.
    callable_scope = _CallableScope(f.defs, registry, external, f.uses, text)
    callable_scope.eager_load_defs()

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
        scope=callable_scope,
        owner=("root",),
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
        boundary_assert_ranges=_boundary_assert_ranges(text),
    )


@dataclass(frozen=True)
class _PrivateCallable:
    """One compiler-owned callable declaration awaiting lazy construction.

    ``owner`` is the complete lexical path that determines generated identity,
    ``label`` is its author-facing diagnostic name, and ``builder`` constructs the
    child flow without arguments when resolution first reaches it.
    """

    owner: tuple[str, ...]
    label: str
    builder: Callable[[], ComposedFlow]


class _CallableScope:
    """The file-local namespace and cache for callable resolution.

    Compiler-owned callables resolve first, then authored ``defs:`` and ``uses:``.
    A bare name in neither authored mapping is rejected instead of reaching the
    external resolver. Private declarations, local defs, and external aliases share
    one memo table, while an in-progress set makes recursive construction loud.

    Args:
        defs_section (`dict`):
            In-file callable name -> raw `defs:` body, retained in insertion order.
        registry:
            The file's parsed typedef registry, shared by every local def.
        external (`ChildResolver`, *optional*):
            Resolver for local refs declared through `uses:`; `None` disables them.
        uses (`dict`):
            Callable alias -> authored external reference.
        parent_text (`str`, *optional*, defaults to `""`):
            Source declaring the defs, used to retain absolute source locations.
    """

    def __init__(
        self,
        defs_section: dict,
        registry,
        external: Optional[ChildResolver],
        uses: dict,
        parent_text: str = "",
    ) -> None:
        both = sorted(set(defs_section) & set(uses))
        if both:
            raise LoadError(
                f"name(s) {both} declared in both defs: and uses: "
                f"(a callable has one namespace)"
            )

        self._defs_section = defs_section
        self._registry = registry
        self._external = external
        self._uses = uses
        self._parent_text = parent_text
        self._private: dict[str, _PrivateCallable] = {}
        self._built: dict[str, ComposedFlow] = {}
        self._in_progress: set[str] = set()

    def register_private(
        self,
        flow_id: str,
        owner: tuple[str, ...],
        label: str,
        builder: Callable[[], ComposedFlow],
    ) -> None:
        """Register one lazy compiler-owned callable in this file scope.

        Re-registering the same generated identity for the same complete lexical
        owner is idempotent: the first label and builder remain authoritative.
        Generated-identity diagnostics deliberately report only the friendly label,
        owner, and reserved prefix; the opaque generated identifier is never exposed.
        """
        if not flow_id.startswith(INLINE_FLOW_PREFIX):
            raise LoadError(
                f"compiler-generated callable {label!r} owned by {owner!r} must use "
                f"the reserved prefix {INLINE_FLOW_PREFIX!r}"
            )

        existing = self._private.get(flow_id)
        if existing is not None:
            if existing.owner == owner:
                return
            raise LoadError(
                "compiler-generated callable collision in reserved prefix "
                f"{INLINE_FLOW_PREFIX!r}: {existing.label!r} owned by "
                f"{existing.owner!r} conflicts with {label!r} owned by {owner!r}"
            )

        authored_section = None
        if flow_id in self._defs_section:
            authored_section = "defs"
        elif flow_id in self._uses:
            authored_section = "uses"
        if authored_section is not None:
            raise LoadError(
                "compiler-generated callable collision in reserved prefix "
                f"{INLINE_FLOW_PREFIX!r}: {label!r} owned by {owner!r} conflicts "
                f"with an authored {authored_section}: name"
            )

        self._private[flow_id] = _PrivateCallable(owner, label, builder)

    def resolve(
        self, flow_id: str, version: Optional[str] = None
    ) -> ComposedFlow:
        """Resolve one callable once; a `uses:` declaration owns its external version."""
        if flow_id in self._built:
            return self._built[flow_id]
        if flow_id in self._private:
            private = self._private[flow_id]
            if flow_id in self._in_progress:
                raise LoadError(
                    "recursive compiler-generated callable in reserved prefix "
                    f"{INLINE_FLOW_PREFIX!r}: {private.label!r} owned by "
                    f"{private.owner!r} is defined in terms of itself "
                    "(the baked call graph must be finite)"
                )
            self._in_progress.add(flow_id)
            try:
                loaded = private.builder()
            finally:
                self._in_progress.discard(flow_id)
            self._built[flow_id] = loaded
            return loaded
        if flow_id in self._defs_section:
            if flow_id in self._in_progress:
                raise LoadError(
                    f"recursive defs reference: callable {flow_id!r} is defined in terms "
                    f"of itself (the baked call graph must be finite)"
                )
            self._in_progress.add(flow_id)
            try:
                loaded = _load_def(
                    flow_id,
                    self._defs_section[flow_id],
                    self._registry,
                    self,
                    self._parent_text,
                )
            finally:
                self._in_progress.discard(flow_id)
            self._built[flow_id] = loaded
            return loaded
        if flow_id in self._uses:
            ref = parse_uses_ref(self._uses[flow_id])
            if ref.scheme == "hub":
                raise LoadError(
                    f"callable {flow_id!r} (uses: {self._uses[flow_id]!r}): marketplace "
                    f"not supported yet (hub: scheme)"
                )
            if ref.scheme is not None:
                raise LoadError(
                    f"callable {flow_id!r} (uses: {self._uses[flow_id]!r}): unknown "
                    f"scheme {ref.scheme!r}: (recognized: a local ref, or hub:)"
                )
            if self._external is None:
                raise LoadError(
                    f"callable {flow_id!r} (uses: {self._uses[flow_id]!r}) needs a search "
                    f"path or child_resolver to resolve (load with search_paths=[...])"
                )
            self._built[flow_id] = self._external(ref.path, ref.version)
            return self._built[flow_id]
        if flow_id.startswith(INLINE_FLOW_PREFIX):
            raise LoadError(
                "compiler-generated callable with reserved prefix "
                f"{INLINE_FLOW_PREFIX!r} was not registered in this file scope"
            )
        raise LoadError(
            f"callable {flow_id!r} is neither an in-file defs: callable nor a uses: "
            f"alias — external flows are reachable only through a uses: alias "
            f"(declare `uses: {{{flow_id}: <ref>}}`)"
        )

    def eager_load_defs(self) -> None:
        """Build every local def in authored mapping order before the parent flow."""
        for name in self._defs_section:
            self.resolve(name)


def _load_def(
    name: str,
    body,
    registry,
    scope: _CallableScope,
    parent_text: str = "",
) -> ComposedFlow:
    """Build one in-file `defs:` entry into a `ComposedFlow` (a callable / sub-flow inline).

    Supports the **multi-node** form only — `{inputs?, nodes, outputs?, asserts?}`,
    sharing the file's `typedefs:` registry + callable scope (so a def can call other defs,
    external flows, or compiler-owned children). A **single-node** form (a top-level
    `kind:`) is deferred -> `LoadError`. `parent_text` is the source of the declaring
    file: the def's nodes live at absolute lines there, so its `SourceFrame` indexes them via
    `_frame_ranges(_def_nodes_reader(parent_text, name))` for the nested-error traceback."""
    if not isinstance(body, dict):
        raise LoadError(
            f"defs entry {name!r}: body must be a mapping, got {type(body).__name__}"
        )
    if "nodes" not in body:
        if "kind" in body:
            return _load_single_node_def(name, body, registry, scope, parent_text)
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
    # Frame extraction parses and mutates singular node keys while back-mapping them to
    # descriptor fields. Assembly therefore receives a second fresh raw reader, preserving
    # the author's singular inline-body fields and their absolute parent-file ranges.
    def_node_ranges, def_field_ranges = _frame_ranges(
        _def_nodes_reader(parent_text, name)
    )
    nodes_reader = _def_assembly_reader(parent_text, name, body["nodes"])
    source = SourceFrame(
        label="defs:" + name,
        text=parent_text,
        node_ranges=def_node_ranges,
        field_ranges=def_field_ranges,
    )
    try:
        return _assemble(
            inputs_section=body.get("inputs") or {},
            nodes_reader=nodes_reader,
            outputs_section=body.get("outputs"),
            asserts_section=body.get("asserts") or [],  # enforced at the child seam
            registry=registry,
            scope=scope,
            owner=("def", name),
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



def _load_single_node_def(
    name: str,
    body: dict,
    registry,
    scope: _CallableScope,
    parent_text: str = "",
) -> ComposedFlow:
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
            scope=scope,
            owner=("def", name),
            n_lines={},
            s_lines={},
            source=source,
        )
    except LoadError as exc:
        if str(exc).startswith("defs entry "):
            raise
        raise LoadError(f"defs entry {name!r}: {exc}", line=exc.line) from exc
