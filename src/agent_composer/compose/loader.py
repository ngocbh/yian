"""The top-level loader: Compose YAML text -> a runnable `ComposedFlow`.

The file-I/O front door: own one source document, build the typedefs registry, compose the
call resolver, load every `defs:` callable, and hand the parsed sections to the
post-parse conductor (`build.pipeline._assemble`, imported back here) which returns
the `ComposedFlow`. This module owns root/def source projections, including routing the
physical flow-assertion cursor for a full root or multi-node def. Compact roots and flat
single-node defs instead project their `asserts:` onto the synthesized node, so they do not
invent a parallel flow-assertion site. It also owns frame ranges, the resolver composition
(`make_file_resolver`/`_CallableScope`), and the defs loaders
(`_load_def`/`_load_single_node_def`) — not the assembly pipeline itself.

`call` resolution is **defs-first**: a `call:` names an in-file `defs:` callable (built
in-loader by the same `_assemble`) or a `uses:` alias resolved through the injected
`child_resolver` (`(flow_id, version) -> ComposedFlow`). `_CallableScope`
composes the two (and rejects recursive/mutual defs). `call` nodes are resolved-and-baked
at load: the loader derives the callable's signature (a plain call re-exports the single
codomain `ValueContract`; a mapped call derives a `list[<codomain>]` contract; bindings
are name/arity- and type-checked against it) AND bakes the callable's compiled flow onto the built
`CallNode` so `run` drives the embedded child. A flow whose `call`s are all
in-file defs loads resolver-free; a `call` to a non-def callable without a resolver is
loud. `run_flow` stays resolver-free.

Imports flow DOWN only: this module composes the compose package's own slices plus
`typesys`/`compile.model`; nothing in the engine imports it back.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

from agent_composer.compile.origins import iter_flow_origins, rebase_flow_origins
from agent_composer.diagnostics import (
    DiagnosticFrame,
    OriginRef,
    OriginResolver,
    ResolverTable,
)
from agent_composer.events import SourceRange
from agent_composer.nodes.base import FieldRanges
from agent_composer.typesys.values import TypeCheckError
from agent_composer.typesys.types import read_typedefs
from agent_composer.compose.build.pipeline import _assemble, ComposedFlow
from agent_composer.compose.desugar.inline_bodies import (
    INLINE_FLOW_PREFIX,
    extract_inline_bodies,
)
from agent_composer.compose.nodes.resolver import ChildResolver
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.helpers.uses import parse_uses_ref
from agent_composer.compose.helpers.source import (
    SourceFrame,
    _boundary_field_ranges,
    _section_lines,
)
from agent_composer.compose.parse.nodes import parse_nodes
from agent_composer.compose.parse.file import (
    flow_nodes_cursor,
    parse_file,
    parse_source_document,
)
from agent_composer.compose.source import (
    SourceAccessError,
    SourceCatalog,
    SourceCursor,
    SourceDocument,
)
from agent_composer.compose.source.catalog import _SourceCatalogBuilder, _scope_origin


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


def _def_assembly_cursor(
    def_cursor: SourceCursor | None,
    normalized_nodes,
) -> SourceCursor:
    """Return the authored def cursor, or a surface-restored virtual fallback."""

    if def_cursor is not None:
        try:
            return def_cursor.child("nodes")
        except SourceAccessError:
            pass
    return SourceCursor.virtual(_restore_node_surface(normalized_nodes))


def _source_child(
    cursor: SourceCursor | None,
    key: Any,
) -> SourceCursor | None:
    """Return one existing child source, or ``None`` for a virtual fallback."""

    if cursor is None:
        return None
    try:
        return cursor.child(key) if cursor.has(key) else None
    except SourceAccessError:
        return None


def _single_node_def_cursor(
    name: str,
    body: dict,
    def_cursor: SourceCursor | None,
) -> SourceCursor:
    """Project a flat def into one source-aligned synthetic node mapping."""

    param_types = body.get("inputs") or {}
    generated_inputs = {
        param: f"${{input.{param}}}"
        for param in param_types
    }
    node_body = {
        key: value
        for key, value in body.items()
        if key not in ("inputs", "outputs")
    }
    node_body["input"] = generated_inputs
    if "outputs" in body:
        node_body["output"] = body["outputs"]

    if def_cursor is None:
        return SourceCursor.virtual({name: node_body})

    body_entries: dict[Any, tuple[Any, SourceCursor | None]] = {
        key: (value, _source_child(def_cursor, key))
        for key, value in node_body.items()
        if key not in ("input", "output")
    }

    input_source = _source_child(def_cursor, "input")
    if input_source is not None and isinstance(input_source._value_view(), Mapping):
        input_cursor = input_source.project_mapping(
            {
                param: (binding, _source_child(input_source, param))
                for param, binding in generated_inputs.items()
            }
        )
    else:
        input_cursor = None
    body_entries["input"] = (generated_inputs, input_cursor)

    if "output" in node_body:
        body_entries["output"] = (
            node_body["output"],
            _source_child(def_cursor, "output"),
        )

    body_cursor = def_cursor.project_mapping(body_entries)
    return def_cursor.project_mapping({name: (node_body, body_cursor)})


def _frame_ranges(
    cursor: SourceCursor,
) -> tuple[dict[str, SourceRange], dict[str, FieldRanges]]:
    """The `(node_ranges, field_ranges)` a `SourceFrame` carries, from a best-effort parse.

    Runs `parse_nodes(cursor)` and pulls each descriptor's own `.range`/`.field_ranges` — the
    render-only location maps the CLI's call-traceback boxes. A virtual compact cursor yields
    empty maps; a malformed `nodes:` (which the build path rejects with
    its own located error) is swallowed here to empty maps — this is render metadata, never the
    error path, so it must not raise.
    """
    try:
        descriptors = parse_nodes(cursor)
    except (LoadError, SourceAccessError):
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
    source: SourceDocument | SourceCursor,
) -> tuple[dict[str, SourceRange], dict[str, FieldRanges]]:
    """Best-effort top-frame ranges from an existing source document or root cursor.

    The `(node_id -> header `SourceRange`, node_id -> {field -> `SourceRange`})` maps a
    `ComposedFlow.source` already carries, derived without reparsing for the CLI's defensive
    error-rendering path. Empty maps for a source-free cursor or invalid semantic shape:
    this is render metadata, never the error path, so it must not raise.
    """
    document = source if isinstance(source, SourceDocument) else source.document
    if document is None:
        return {}, {}
    try:
        cf = parse_file(document)
    except (LoadError, SourceAccessError):
        return {}, {}
    try:
        lowered, _inline_bodies = extract_inline_bodies(
            flow_nodes_cursor(document, cf.nodes),
            owner=("root",),
        )
    except (LoadError, SourceAccessError):
        return {}, {}
    return _frame_ranges(lowered)


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


def _reject_authored_private_callable_names(
    root: SourceCursor,
    defs: dict,
    uses: dict,
) -> None:
    """Reject file-scoped callable declarations in the inline-flow namespace.

    The inexpensive declaration scan reuses the root cursor built at the source
    boundary, including the offending declaration's key range. It never parses
    the document again.
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

        declaration_range = (
            root.child(section).key_range_of(offender)
            if root.has(section)
            else None
        )

        noun = "callable names" if section == "defs" else "aliases"
        raise LoadError(
            f"{section}: {noun} with prefix {INLINE_FLOW_PREFIX!r} are reserved "
            "for compiler-generated inline child flows",
            range=declaration_range,
        )


@dataclass
class _LoadCtx:
    """Cross-file cycle state and successful compiled-flow reuse by resolved path."""

    loading: set[Path] = field(default_factory=set)
    flows: dict[Path, ComposedFlow] = field(default_factory=dict)


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


def _distinct_origins(compiled) -> tuple[OriginRef, ...]:
    """Return canonical reachable origins in first structural visit order.

    Args:
        compiled (`CompiledFlow`): Static graph whose provenance slots are walked.

    Returns:
        `tuple[OriginRef, ...]`: One exact base origin per distinct opaque key.
    """
    by_key: dict[str, OriginRef] = {}
    for origin in iter_flow_origins(compiled):
        canonical = OriginRef(origin.key)
        by_key.setdefault(canonical.key, canonical)
    return tuple(by_key.values())


def _validate_external_child(alias: str, child: object) -> tuple[OriginRef, ...]:
    """Validate one custom or cached child before lexical rebasing.

    Args:
        alias (`str`): Authored ``uses:`` alias selecting the child.
        child (`object`): Value returned by the external child resolver.

    Returns:
        `tuple[OriginRef, ...]`: Canonical reachable child origins.

    Raises:
        `LoadError`:
            Raised when the value is not a `ComposedFlow`, resolver presence does
            not match provenance, or any reachable origin lacks a concrete frame.
    """
    if not isinstance(child, ComposedFlow):
        raise LoadError(
            f"uses: alias {alias!r} resolver returned "
            f"{type(child).__name__}, expected ComposedFlow"
        )
    try:
        origins = _distinct_origins(child.compiled)
    except (TypeError, ValueError) as exc:
        raise LoadError(
            f"uses: alias {alias!r} returned an invalid compiled origin graph: {exc}"
        ) from exc

    resolver = child.origin_resolver
    if not origins:
        if resolver is not None:
            raise LoadError(
                f"uses: alias {alias!r} returned a source-free child with an "
                "origin resolver"
            )
        return origins
    if resolver is None:
        raise LoadError(
            f"uses: alias {alias!r} returned an originful child without an "
            "origin resolver"
        )
    resolve = getattr(resolver, "resolve", None)
    if not callable(resolve):
        raise LoadError(
            f"uses: alias {alias!r} child origin resolver does not implement resolve"
        )

    for origin in origins:
        try:
            frame = resolve(origin)
        except Exception as exc:
            raise LoadError(
                f"uses: alias {alias!r} child origin resolver failed for "
                f"{origin.key!r}: {exc}"
            ) from exc
        if not isinstance(frame, DiagnosticFrame):
            detail = "no frame" if frame is None else type(frame).__name__
            raise LoadError(
                f"uses: alias {alias!r} child origin resolver returned {detail} "
                f"for {origin.key!r}"
            )
    return origins


def _rebase_external_child(alias: str, child: object) -> ComposedFlow:
    """Return one validated child copy scoped under its lexical alias.

    Args:
        alias (`str`): Authored ``uses:`` alias owning the returned view.
        child (`object`): Canonical `ComposedFlow` returned by the host resolver.

    Returns:
        `ComposedFlow`: Independent alias-scoped child with an exact route table,
        or `origin_resolver=None` when its compiled graph is source-free.

    Raises:
        `LoadError`: Raised when child validation or generic rebasing fails.
    """
    origins = _validate_external_child(alias, child)
    assert isinstance(child, ComposedFlow)
    try:
        compiled, table = rebase_flow_origins(
            child.compiled,
            lambda origin: _scope_origin(alias, origin),
            resolver=child.origin_resolver,
        )
    except (TypeError, ValueError) as exc:
        raise LoadError(
            f"uses: alias {alias!r} could not rebase child origins: {exc}"
        ) from exc
    if len(table) != len(origins):
        raise LoadError(
            f"uses: alias {alias!r} produced an incomplete rebased origin table"
        )
    return replace(
        child,
        compiled=compiled,
        origin_resolver=table if origins else None,
    )


def _finalize_composed_flow(
    staged: ComposedFlow,
    local_catalog: SourceCatalog,
    external_tables: tuple[ResolverTable, ...],
) -> ComposedFlow:
    """Install one exact resolver over a fully assembled physical file.

    Args:
        staged (`ComposedFlow`): Root assembly whose graph is complete but whose
            public resolver is still unset.
        local_catalog (`SourceCatalog`): Frozen authority for this document's sites.
        external_tables (`tuple[ResolverTable, ...]`): Alias-scoped child
            authorities, including eagerly loaded aliases that may be unreachable.

    Returns:
        `ComposedFlow`: Copy carrying exactly one route per reachable origin, or
        `origin_resolver=None` when the root graph is source-free.

    Raises:
        `LoadError`: Raised for malformed graphs, missing or overlapping ownership,
            or a selected authority that cannot produce a `DiagnosticFrame`.
    """
    try:
        origins = _distinct_origins(staged.compiled)
    except (TypeError, ValueError) as exc:
        raise LoadError(f"invalid compiled origin graph: {exc}") from exc
    if not origins:
        return replace(staged, origin_resolver=None)

    entries: list[tuple[OriginRef, OriginResolver, OriginRef]] = []
    for origin in origins:
        authorities: list[OriginResolver] = []
        if origin in local_catalog:
            authorities.append(local_catalog)
        authorities.extend(table for table in external_tables if origin in table)
        if not authorities:
            raise LoadError(
                f"reachable origin {origin.key!r} has no resolver route"
            )
        if len(authorities) != 1:
            raise LoadError(
                f"reachable origin {origin.key!r} has multiple resolver routes"
            )
        authority = authorities[0]
        try:
            frame = authority.resolve(origin)
        except Exception as exc:
            raise LoadError(
                f"resolver route failed for origin {origin.key!r}: {exc}"
            ) from exc
        if not isinstance(frame, DiagnosticFrame):
            detail = "no frame" if frame is None else type(frame).__name__
            raise LoadError(
                f"resolver route returned {detail} for origin {origin.key!r}"
            )
        entries.append((origin, authority, origin))
    return replace(
        staged,
        origin_resolver=ResolverTable.from_entries(entries),
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
            if cand in ctx.flows:
                child = ctx.flows[cand]
                _check_version(child, path, version)
                return child
            if cand in ctx.loading:
                raise LoadError(f"flow-reference cycle across files: {cand}")
            ctx.loading.add(cand)
            try:
                child_document = parse_source_document(
                    cand.read_text(),
                    cand.name,
                )
                child = _load_flow(child_document, None, [cand.parent], ctx)
            finally:
                ctx.loading.discard(cand)
            # Relabel the external child's frame to its filename BEFORE caching, so every
            # diamond-reuse cache-hit also returns the filename label (not the file's own
            # `name:`). The label is value-stable across sites (same abs-path -> same name).
            if child.source is not None:
                child = replace(child, source=replace(child.source, label=cand.name))
            ctx.flows[cand] = child
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
    document = parse_source_document(text, "<flow>")
    return _load_flow(document, child_resolver, search_paths, _LoadCtx())


def _load_flow(
    document: SourceDocument,
    child_resolver,
    search_paths,
    ctx: "_LoadCtx",
) -> ComposedFlow:
    root = document.root
    text = document.text
    f = parse_file(document)
    _reject_authored_private_callable_names(root, f.defs, f.uses)
    _validate_system(f.system)

    try:
        registry = read_typedefs(f.typedefs)
    except TypeCheckError as exc:
        # coarse: the typesys layer doesn't track source lines, so anchor at the
        # `typedefs:` section line (the box's context still shows the offending name).
        raise LoadError(
            f"bad typedefs: {exc}", line=_section_lines(root).get("typedefs")
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
    # broken/recursive callable at load (loud, not a hang). The owned document supplies each
    # def's raw source cursor and absolute parent-file locations without another parse.
    defs_cursor = root.child("defs") if root.has("defs") else None
    callable_scope = _CallableScope(
        f.defs,
        registry,
        external,
        f.uses,
        defs_cursor,
    )
    callable_scope.eager_load_defs()

    # Render-only source frame for THIS flow: copied onto child call/map nodes for the
    # nested-error traceback. For an external file (loaded via make_file_resolver) the
    # label is relabeled to the filename in `resolve`.
    source = SourceFrame(
        label=f.name or "<flow>",
        text=text,
        node_ranges={},
        field_ranges={},
    )

    staged = _assemble(
        inputs_section=f.inputs,
        nodes_cursor=flow_nodes_cursor(document, f.nodes),
        outputs_section=f.outputs,
        asserts_section=f.asserts,
        registry=registry,
        scope=callable_scope,
        owner=("root",),
        n_lines={},
        s_lines=_section_lines(root),
        uses_aliases=set(f.uses),
        version=f.version,
        name=f.name,
        description=f.description,
        flow_llm_config=f.llm_config,
        flow_env=f.env,
        source=source,
        boundary_field_ranges=_boundary_field_ranges(root),
        asserts_cursor=(
            root.child("asserts")
            if root.has("nodes") and root.has("asserts")
            else None
        ),
        outputs_cursor=(
            root.child("output")
            if root.has("nodes") and root.has("output")
            else None
        ),
    )
    return callable_scope.finalize(staged)


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
    one memo table. The in-progress set makes recursive private/def construction
    loud; cross-file recursion is outside this scope and belongs to the selected
    external resolver. Failed callable construction atomically drops new private
    declarations, nested cache entries, and source-catalog sites so a later retry
    cannot observe a stale partial attempt.

    Args:
        defs_section (`dict`):
            In-file callable name -> raw `defs:` body, retained in insertion order.
        registry:
            The file's parsed typedef registry, shared by every local def.
        external (`ChildResolver`, *optional*):
            Resolver for local refs declared through `uses:`; `None` disables them.
        uses (`dict`):
            Callable alias -> authored external reference.
        defs_cursor (`SourceCursor | None`, *optional*, defaults to `None`):
            Cursor over the authored ``defs:`` mapping. ``None`` supports
            programmatic scopes with virtual, location-free def cursors.
        catalog_builder (`_SourceCatalogBuilder | None`, *optional*):
            Mutable document-local registry shared by root, definitions, and
            inline bodies. A private builder is created for programmatic scopes.
    """

    def __init__(
        self,
        defs_section: dict,
        registry,
        external: Optional[ChildResolver],
        uses: dict,
        defs_cursor: SourceCursor | None = None,
        catalog_builder: _SourceCatalogBuilder | None = None,
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
        self._defs_cursor = defs_cursor
        self._catalog_builder = (
            catalog_builder
            if catalog_builder is not None
            else _SourceCatalogBuilder()
        )
        self._private: dict[str, _PrivateCallable] = {}
        self._built: dict[str, ComposedFlow] = {}
        self._in_progress: set[str] = set()

    @property
    def catalog_builder(self) -> _SourceCatalogBuilder:
        """Return this physical file's mutable origin registry."""
        return self._catalog_builder

    def finalize(self, staged: ComposedFlow) -> ComposedFlow:
        """Install the exact resolver for one completed physical-file root.

        Args:
            staged (`ComposedFlow`): Root graph assembled with its resolver unset.

        Returns:
            `ComposedFlow`: Final public/cache value with exact reachable routes.

        Raises:
            `LoadError`: Raised when an eager alias is missing its rebased table or
                final reachable-origin ownership is incomplete or ambiguous.
        """
        tables: list[ResolverTable] = []
        for alias in self._uses:
            child = self._built.get(alias)
            if child is None:
                raise LoadError(
                    f"uses: alias {alias!r} was not resolved before finalization"
                )
            resolver = child.origin_resolver
            if resolver is None:
                continue
            if not isinstance(resolver, ResolverTable):
                raise LoadError(
                    f"uses: alias {alias!r} did not retain a rebased resolver table"
                )
            tables.append(resolver)
        return _finalize_composed_flow(
            staged,
            self._catalog_builder.freeze(),
            tuple(tables),
        )

    def _build_transactionally(
        self,
        builder: Callable[[], ComposedFlow],
    ) -> ComposedFlow:
        """Build one callable without caching origins from a failed attempt.

        Nested resolutions may populate `_built`, and assembly may register
        inline builders in `_private`, while `builder` runs. If construction
        fails, discard both sets of new entries together with the catalog sites
        rolled back by the builder transaction.
        """
        cached_before = frozenset(self._built)
        private_before = frozenset(self._private)
        try:
            with self._catalog_builder.transaction():
                return builder()
        except BaseException:
            for flow_id in tuple(self._built):
                if flow_id not in cached_before:
                    del self._built[flow_id]
            for flow_id in tuple(self._private):
                if flow_id not in private_before:
                    del self._private[flow_id]
            raise

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
                loaded = self._build_transactionally(private.builder)
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
                def_cursor = None
                if self._defs_cursor is not None:
                    try:
                        def_cursor = self._defs_cursor.child(flow_id)
                    except SourceAccessError:
                        pass
                loaded = self._build_transactionally(
                    lambda: _load_def(
                        flow_id,
                        self._defs_section[flow_id],
                        self._registry,
                        self,
                        def_cursor,
                    )
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
            child = self._external(ref.path, ref.version)
            loaded = _rebase_external_child(flow_id, child)
            self._built[flow_id] = loaded
            return loaded
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
    def_cursor: SourceCursor | None = None,
) -> ComposedFlow:
    """Build one in-file `defs:` entry into a `ComposedFlow` (a callable / sub-flow inline).

    Supports both a multi-node sub-flow (`{inputs?, nodes, outputs?, asserts?}`)
    and a flat single-node body (`{kind, ...}`). Both share the file's typedef
    registry and callable scope. ``def_cursor`` is the occurrence-local authored
    body: it retains absolute locations, and `_assemble` derives the def's
    `SourceFrame` from its lowered descriptors.
    """
    if not isinstance(body, dict):
        raise LoadError(
            f"defs entry {name!r}: body must be a mapping, got {type(body).__name__}"
        )
    if "nodes" not in body:
        if "kind" in body:
            return _load_single_node_def(
                name,
                body,
                registry,
                scope,
                def_cursor,
            )
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
    nodes_cursor = _def_assembly_cursor(def_cursor, body["nodes"])
    source_text = (
        def_cursor.document.text
        if def_cursor is not None and def_cursor.document is not None
        else ""
    )
    source = SourceFrame(
        label="defs:" + name,
        text=source_text,
        node_ranges={},
        field_ranges={},
    )
    try:
        return _assemble(
            inputs_section=body.get("inputs") or {},
            nodes_cursor=nodes_cursor,
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
            boundary_field_ranges=(
                _boundary_field_ranges(def_cursor)
                if def_cursor is not None
                else None
            ),
            asserts_cursor=_source_child(def_cursor, "asserts"),
            outputs_cursor=(
                _source_child(def_cursor, "output")
                or _source_child(def_cursor, "outputs")
            ),
        )
    except LoadError as exc:
        # Name the def without discarding the absolute parent-file diagnostic metadata.
        # Don't re-prefix an already-named nested-def error; keep the innermost name.
        if str(exc).startswith("defs entry "):
            raise
        raise LoadError(
            f"defs entry {name!r}: {exc}",
            line=exc.line,
            lines=exc.lines,
            notes=exc.notes,
            range=exc.range,
        ) from exc



def _load_single_node_def(
    name: str,
    body: dict,
    registry,
    scope: _CallableScope,
    def_cursor: SourceCursor | None = None,
) -> ComposedFlow:
    """Build a SINGLE-node `defs:` callable — `let f params = <one node over params>`.

    The flat body is `{kind, <logic fields>, input?, output?}` (singular keys; the
    parser back-maps to the plural-keyed descriptor space). Convention (auto-wire by
    name): `input:` declares the def's PARAMETERS (name -> TYPE, like a flow signature),
    each auto-bound by name into the single node (`p = ${input.p}`); `output:` is the
    node's output TYPE (the codomain). Everything except `input:`/`output:` is the node
    body (kind + its logic fields + any node-local `asserts:`). It desugars to a one-node
    sub-flow and runs through the same `_assemble` as the multi-node form. The synthetic
    node projects its header, logic, generated bindings, and output type onto the flat
    authored def cursor without mutating that source."""
    param_types = body.get("inputs") or {}
    if not isinstance(param_types, dict):
        raise LoadError(
            f"defs entry {name!r}: single-node `inputs:` must be a param map (name: TYPE)"
        )
    nodes_cursor = _single_node_def_cursor(name, body, def_cursor)
    output_source = _source_child(def_cursor, "output")
    output_range = output_source.range if output_source is not None else None
    source_text = (
        def_cursor.document.text
        if def_cursor is not None and def_cursor.document is not None
        else ""
    )
    source = SourceFrame(
        label="defs:" + name,
        text=source_text,
        node_ranges={},
        field_ranges={},
    )
    try:
        return _assemble(
            inputs_section=param_types,
            nodes_cursor=nodes_cursor,
            outputs_section={"result": f"${{{name}.output}}"},  # single value re-export (node-first)
            asserts_section=[],
            registry=registry,
            scope=scope,
            owner=("def", name),
            n_lines={},
            s_lines=(
                {"output": output_range.start_line}
                if output_range is not None
                else {}
            ),
            source=source,
            boundary_field_ranges=(
                _boundary_field_ranges(def_cursor)
                if def_cursor is not None
                else None
            ),
            outputs_range=output_range,
        )
    except LoadError as exc:
        if str(exc).startswith("defs entry "):
            raise
        raise LoadError(
            f"defs entry {name!r}: {exc}",
            line=exc.line,
            lines=exc.lines,
            notes=exc.notes,
            range=exc.range,
        ) from exc
