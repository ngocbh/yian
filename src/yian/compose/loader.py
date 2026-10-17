"""The top-level loader: Compose YAML text -> a runnable `ComposedFlow`.

The file-I/O front door: own one source document, build the typedefs registry, compose the
call resolver, load every `defs:` callable, and hand the parsed sections to the
post-parse conductor (`build.pipeline._assemble`, imported back here) which returns
the `ComposedFlow`. This module owns root/def source projections, including routing the
physical flow-input cursor for roots and both definition forms, plus the flow-assertion
cursor for a full root or multi-node def. Compact roots and flat single-node defs instead
project their `asserts:` onto the synthesized node, so they do not invent a parallel
flow-assertion site. It also owns resolver composition
(`make_file_resolver`/`_CallableScope`), and the defs loaders
(`_load_def`/`_load_single_node_def`) — not the assembly pipeline itself.

`call` resolution is **defs-first**: a `call:` names an in-file `defs:` callable (built
in-loader by the same `_assemble`) or a `uses:` alias resolved through the injected
`child_resolver` (`(flow_id, version) -> ComposedFlow`). `_CallableScope`
composes the two (and rejects recursive/mutual defs). `call` nodes are resolved-and-baked
at load: the loader derives the callable's signature (a plain call re-exports the exact
or partial codomain contract; a mapped call derives `list(<codomain>)` only from an exact
contract; bindings
are name/arity- and type-checked against it) AND bakes the callable's compiled flow onto the built
`CallNode` so `run` drives the embedded child. A flow whose `call`s are all
in-file defs loads resolver-free; a `call` to a non-def callable without a resolver is
loud. `run_flow` stays resolver-free.

Source resolution follows the same two cache boundaries. A physical-file cache entry is
canonical and complete. `_CallableScope` validates it, copies/rebases its compiled origins
once per lexical `uses:` alias, and reuses that alias view at every callsite. After the
root is assembled, this module freezes the one document catalog, filters local and alias
authorities to the exact reachable keys, and installs `ComposedFlow.origin_resolver`.
Source-free graphs carry `None`; private `_assemble` products keep the resolver unset.
Final resolver routes include the START signature's parameter, contract, and default origins.

A flow's chain of `uses:` imports is at most `current().max_call_nesting` files deep, the
call nesting limit, counted apart from call depth; each loaded flow records its own
`import_depth`.

Imports flow DOWN only: this module composes the compose package's own components plus
`typesys`/`compile`; nothing in the engine imports it back.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import ValidationError

from yian.compile.memory_layout import (
    resolve_memory_layout,
    scope_memory_declarations,
)
from yian.compile.model import CompiledFlow
from yian.compile.origins import iter_flow_origins, rebase_flow_origins
from yian.defaults import current
from yian.diagnostics import (
    DiagnosticFrame,
    OriginRef,
    OriginResolver,
    ResolverTable,
)
from yian.memory.contracts import DeclSite, MemorySetupError
from yian.typesys.values import TypeCheckError
from yian.typesys.types import read_typedefs
from yian.compose.build.pipeline import _assemble, ComposedFlow
from yian.compose.desugar.inline_bodies import INLINE_FLOW_PREFIX
from yian.compose.nodes.base import not_text_detail, not_text_error
from yian.compose.nodes.resolver import _ACTIVE_CALLS, ChildResolver, call_nesting_scope
from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.helpers.memory import parse_memory_declarations
from yian.compose.helpers.tags import reject_unread_tags
from yian.compose.helpers.uses import parse_uses_ref
from yian.compose.helpers.value_nesting import check_env_values
from yian.compose.helpers.source import _boundary_field_ranges
from yian.compose.parse.file import (
    flow_nodes_cursor,
    parse_file,
    parse_source_document,
)
from yian.compose.source import (
    SourceAccessError,
    SourceCatalog,
    SourceCursor,
    SourceDocument,
)
from yian.compose.source.catalog import _SourceCatalogBuilder, _scope_origin
from yian.llm_clients import LLMConfig
from yian.llm_clients.config import host_only_field_error
from yian.nodes.agent.node import is_max_tool_iterations


_UNLOCATED_DIAGNOSTICS = LoadDiagnostics()


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


# A `defs:` child flow accepts either a multi-node body or a flat single-node `kind:`
# body. Both share the file-scoped metadata and type registry rather than declaring
# `id`/`name`/`typedefs` themselves.
# `asserts:` are allowed: the call/map child seam enforces a def's two-phase asserts
# against the child pool (boundary before the child runs, post after it terminates).
# `env:` is the def's own flow-level default for its nodes; a parent's `env:` does not
# reach a called child.
_DEF_ALLOWED = frozenset(
    {"inputs", "nodes", "outputs", "asserts", "llm_config", "env", "long_term_memory"}
)
# The authored spellings of `_DEF_ALLOWED`, for error messages.
_DEF_SURFACE_KEYS = frozenset(
    {"inputs": "input", "outputs": "output"}.get(key, key) for key in _DEF_ALLOWED
)


def _validate_system(
    system: dict,
    cursor: SourceCursor | None = None,
) -> None:
    """Validate the top-level `system:` SECTION shape. Only `paths:` (a list of
    strings) is allowed — this is the resolution search path, NOT the strict
    `${system.X}` ambient namespace (a `${system.paths}` ref stays a compile error)."""
    extra = sorted(set(system) - {"paths"})
    if extra:
        raise LoadDiagnostics.from_cursor(cursor).error(
            f"system: unknown key(s) {extra} (allowed: paths)",
            range=(cursor.key_range_of(extra[0]) if cursor is not None else None),
        )
    paths = system.get("paths", [])
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        raise LoadDiagnostics.from_cursor(cursor).error(
            "system: paths must be a list of strings",
            range=(cursor.range_of("paths") if cursor is not None else None),
        )


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
        raise LoadDiagnostics.from_cursor(root).error(
            f"{section}: {noun} with prefix {INLINE_FLOW_PREFIX!r} are reserved "
            "for compiler-generated inline child flows",
            range=declaration_range,
        )


# How many file loads the default file resolver nests above the file `_load_callee_first`
# is loading before it hands the next import to the scheduler's stack. Nesting lets a
# file load its imports in one pass, so a file that imports many files is loaded once;
# the bound keeps a long import chain from nesting more file loads than that on the
# Python stack.
_MAX_NESTED_FILE_LOADS = 8


@dataclass
class _PendingFile:
    """One `uses:` file on the callee-first load stack, or one file load nested above it.

    `active_imports` is the import count its load starts from, `outer` is the set of
    files being loaded when it was imported, without itself, and `document` is the
    file's parsed source once read. `failures` maps each import that failed to load
    while this file was on top, keyed by the import and its `outer`, to the exception
    and traceback it failed with.
    """

    path: Path
    active_imports: int
    outer: frozenset[Path]
    document: SourceDocument | None = None
    failures: dict[tuple[Path, frozenset[Path]], tuple[Exception, Any]] = field(
        default_factory=dict
    )


@dataclass
class _LoadCtx:
    """Cross-file cycle state and successful compiled-flow reuse by resolved path.

    `loading` holds the files being loaded, so it is empty while no default file
    resolver of this context is loading a file. While one is, `pending` is the
    callee-first load stack and `nested` the file loads nested above its top.
    """

    loading: set[Path] = field(default_factory=set)
    flows: dict[Path, ComposedFlow] = field(default_factory=dict)
    pending: list[_PendingFile] = field(default_factory=list)
    nested: list[_PendingFile] = field(default_factory=list)


class _LoadFirst(BaseException):
    """Asks the scheduler loading `uses:` files callee-first to load a file before its importers.

    It is a `BaseException` so the `except Exception` sites a file load passes through
    let it reach the scheduler.

    Args:
        ctx (`_LoadCtx`): The load context whose scheduler handles it.
        chain (`list[_PendingFile]`): The file loads nested above the scheduler's top
            when the import was made, outermost first, then the imported file.
    """

    def __init__(self, ctx: _LoadCtx, chain: list[_PendingFile]) -> None:
        super().__init__(str(chain[-1].path))
        self.ctx = ctx
        self.chain = chain


# `uses:` imports being loaded around the current one. It spans files and is never reset
# by `load_flow`, so a custom `child_resolver` that calls `load_flow` is bounded too.
_ACTIVE_IMPORTS: ContextVar[int] = ContextVar("_ACTIVE_IMPORTS", default=0)


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
        raise _UNLOCATED_DIAGNOSTICS.error(
            f"uses: ref pinned {path}@{version} but the resolved file declares "
            f"version: {child.version!r} (file-per-version: name a file that declares {version!r})"
        )


def _distinct_origins(compiled: CompiledFlow) -> tuple[OriginRef, ...]:
    """Return distinct reachable origins in first structural visit order.

    Args:
        compiled (`CompiledFlow`): Static graph whose provenance slots are walked.

    Returns:
        `tuple[OriginRef, ...]`:
            The first exact origin object reached for each distinct opaque key.
    """
    by_key: dict[str, OriginRef] = {}
    for origin in iter_flow_origins(compiled):
        canonical = OriginRef(origin.key)
        by_key.setdefault(canonical.key, origin)
    return tuple(by_key.values())


def _validate_external_child(alias: str, child: object) -> tuple[OriginRef, ...]:
    """Validate one custom or cached child before lexical rebasing.

    Args:
        alias (`str`): Authored ``uses:`` alias selecting the child.
        child (`object`): Value returned by the external child resolver.

    Returns:
        `tuple[OriginRef, ...]`:
            The first exact reachable child origin for each distinct opaque key.

    Raises:
        `LoadError`:
            Raised when the value is not a `ComposedFlow`, resolver presence does
            not match provenance, or any reachable origin lacks a concrete frame.
    """
    if not isinstance(child, ComposedFlow):
        raise _UNLOCATED_DIAGNOSTICS.error(
            f"uses: alias {alias!r} resolver returned "
            f"{type(child).__name__}, expected ComposedFlow"
        )
    try:
        origins = _distinct_origins(child.compiled)
    except Exception as exc:
        raise _UNLOCATED_DIAGNOSTICS.error(
            f"uses: alias {alias!r} returned an invalid compiled origin graph: {exc}"
        ) from exc

    resolver = child.origin_resolver
    if not origins:
        if resolver is not None:
            raise _UNLOCATED_DIAGNOSTICS.error(
                f"uses: alias {alias!r} returned a source-free child with an "
                "origin resolver"
            )
        return origins
    if resolver is None:
        raise _UNLOCATED_DIAGNOSTICS.error(
            f"uses: alias {alias!r} returned an originful child without an "
            "origin resolver"
        )
    try:
        resolve = getattr(resolver, "resolve", None)
    except Exception as exc:
        raise _UNLOCATED_DIAGNOSTICS.error(
            f"uses: alias {alias!r} child origin resolver could not expose "
            f"resolve: {exc}"
        ) from exc
    if not callable(resolve):
        raise _UNLOCATED_DIAGNOSTICS.error(
            f"uses: alias {alias!r} child origin resolver does not implement resolve"
        )

    for origin in origins:
        try:
            frame = resolve(origin)
        except Exception as exc:
            raise _UNLOCATED_DIAGNOSTICS.error(
                f"uses: alias {alias!r} child origin resolver failed for "
                f"{origin.key!r}: {exc}"
            ) from exc
        if not isinstance(frame, DiagnosticFrame):
            detail = "no frame" if frame is None else type(frame).__name__
            raise _UNLOCATED_DIAGNOSTICS.error(
                f"uses: alias {alias!r} child origin resolver returned {detail} "
                f"for {origin.key!r}"
            )
    return origins


def _rebase_external_child(alias: str, child: object) -> ComposedFlow:
    """Return one validated child copy scoped under its lexical alias.

    Every reachable memory declaration in the copy gains ``alias`` as its
    outermost scope segment.

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
        # rebase_flow_origins returned an independent deep copy, so the scope
        # rewrite cannot reach the cached canonical child.
        scope_memory_declarations(compiled, alias)
    except Exception as exc:
        raise _UNLOCATED_DIAGNOSTICS.error(
            f"uses: alias {alias!r} could not rebase child: {exc}"
        ) from exc
    if len(table) != len(origins):
        raise _UNLOCATED_DIAGNOSTICS.error(
            f"uses: alias {alias!r} produced an incomplete rebased origin table"
        )
    return replace(
        child,
        compiled=compiled,
        origin_resolver=table if origins else None,
    )


def _memory_setup_load_error(flow: ComposedFlow, exc: MemorySetupError) -> LoadError:
    """Locate one memory layout failure at its first declaration site.

    Args:
        flow (`ComposedFlow`): Loaded root whose resolver maps origins to source.
        exc (`MemorySetupError`): Resolver failure with its origins and notes.

    Returns:
        `LoadError`: Failure framed at ``exc.origin``. Its notes are the
        exception's own notes, then one located note per related origin whose
        location differs from the primary frame and from every earlier note.
    """
    resolver = flow.origin_resolver

    def frame(origin: OriginRef | None) -> DiagnosticFrame | None:
        return None if origin is None or resolver is None else resolver.resolve(origin)

    primary = frame(exc.origin)
    notes = list(getattr(exc, "__notes__", ()))
    # Aliases of one file map distinct origins to one source location.
    seen: set[tuple[str, int, int]] = set()
    if primary is not None:
        seen.add((primary.label, primary.range.start_line, primary.range.start_col))
    for origin in exc.related:
        other = frame(origin)
        if other is None:
            continue
        location = (other.label, other.range.start_line, other.range.start_col)
        if location in seen:
            continue
        seen.add(location)
        notes.append(f"related declaration at {location[0]}:{location[1]}:{location[2]}")
    return _UNLOCATED_DIAGNOSTICS.error(
        str(exc),
        frames=() if primary is None else (primary,),
        notes=notes,
    )


def _finalize_composed_flow(
    staged: ComposedFlow,
    local_catalog: SourceCatalog,
    external_tables: tuple[ResolverTable, ...],
    diagnostics: LoadDiagnostics = _UNLOCATED_DIAGNOSTICS,
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
        raise diagnostics.error(f"invalid compiled origin graph: {exc}") from exc
    if not origins:
        return replace(staged, origin_resolver=None)

    entries: list[tuple[OriginRef, OriginResolver, OriginRef]] = []
    for origin in origins:
        authorities: list[OriginResolver] = []
        if origin in local_catalog:
            authorities.append(local_catalog)
        authorities.extend(table for table in external_tables if origin in table)
        if not authorities:
            raise diagnostics.error(
                f"reachable origin {origin.key!r} has no resolver route"
            )
        if len(authorities) != 1:
            raise diagnostics.error(
                f"reachable origin {origin.key!r} has multiple resolver routes"
            )
        authority = authorities[0]
        try:
            frame = authority.resolve(origin)
        except Exception as exc:
            raise diagnostics.error(
                f"resolver route failed for origin {origin.key!r}: {exc}"
            ) from exc
        if not isinstance(frame, DiagnosticFrame):
            detail = "no frame" if frame is None else type(frame).__name__
            raise diagnostics.error(
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
    `version:` must equal the pinned `<v>`. Each child loads with its own directory as
    its search path, so its own `uses:`/`system.paths` re-root there. A file's imports
    load nested inside it up to `_MAX_NESTED_FILE_LOADS` file loads deep, and
    callee-first past that (`_load_callee_first`), so a long chain of imports does not
    nest one load per file. A `hub:`/scheme ref never reaches here (it is handled in the
    composite resolver).

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
                raise _UNLOCATED_DIAGNOSTICS.error(
                    f"flow-reference cycle across files: {cand}"
                )
            if not ctx.loading:
                child = _load_callee_first(ctx, cand)
                _check_version(child, path, version)
                return child
            imported = _PendingFile(cand, _ACTIVE_IMPORTS.get(), frozenset(ctx.loading))
            failure = ctx.pending[-1].failures.get((cand, imported.outer))
            if failure is not None:
                exc, traceback = failure
                raise exc.with_traceback(traceback)
            if len(ctx.nested) >= _MAX_NESTED_FILE_LOADS:
                raise _LoadFirst(ctx, [*ctx.nested, imported])
            ctx.nested.append(imported)
            ctx.loading.add(cand)
            try:
                imported.document = parse_source_document(cand.read_text(), cand.name)
                child = _load_flow(imported.document, None, [cand.parent], ctx)
            finally:
                ctx.loading.discard(cand)
                ctx.nested.pop()
            ctx.flows[cand] = child
            _check_version(child, path, version)
            return child
        raise _UNLOCATED_DIAGNOSTICS.error(
            f"flow {rel!r} not found on search path {[str(d) for d in search_paths]}"
        )

    return resolve


def _load_callee_first(ctx: _LoadCtx, first: Path) -> ComposedFlow:
    """Load `uses:` file `first` and the files it imports, nesting at most a few file loads.

    While this runs, a default file resolver of `ctx` loads a file not loaded yet nested
    inside its importer, as outside this, while fewer than `_MAX_NESTED_FILE_LOADS` file
    loads are nested above the file on top of the stack. Past that it raises
    `_LoadFirst`, which aborts the load on top. The nested files and then the imported
    file are pushed on the stack, the imported file is loaded first, and each file below
    it is loaded again once the file above it is done. A pushed file loads with the
    files being loaded and the active import count it was imported with, which are what
    a load nested in its importers sees, so cross-file cycles and the call nesting limit
    are checked as before. Files the aborted load finished loading stay cached in
    `ctx.flows`, so loading again finds them loaded, as a nested load would.

    When a file on the stack fails, its exception is recorded for the file below it,
    keyed by the failed file and the files being loaded when it was imported. While the
    file below loads again, the resolver asked for the failed file with those same files
    being loaded raises it again, so the importer's own error handling wraps it as if
    the file had been loaded nested. The records are dropped when the file below leaves
    the stack. The failure of `first` is raised.

    Each file is read and parsed once; loading it again reuses its document, which a
    load only reads. This relies on a file's load being deterministic given `ctx`, and
    on an aborted load leaving no trace but the files it finished loading: each load
    has its own callable scope and catalog, and only a successful load is cached in
    `ctx.flows`.

    Args:
        ctx (`_LoadCtx`): The load context, with no file loading.
        first (`Path`): The resolved file to load.

    Returns:
        `ComposedFlow`: The loaded `first`, also cached in `ctx.flows`.

    Raises:
        `Exception`: Re-raises the recorded failure of `first` or of a file it imports,
            which may be any `Exception`.
    """
    ctx.pending = [_PendingFile(first, _ACTIVE_IMPORTS.get(), frozenset())]
    try:
        while ctx.pending:
            top = ctx.pending[-1]
            ctx.loading.clear()
            ctx.loading.update({*top.outer, top.path})
            abort: _LoadFirst | None = None
            failure: Exception | None = None
            token = _ACTIVE_IMPORTS.set(top.active_imports)
            try:
                if top.document is None:
                    top.document = parse_source_document(top.path.read_text(), top.path.name)
                ctx.flows[top.path] = _load_flow(top.document, None, [top.path.parent], ctx)
            except _LoadFirst as exc:
                if exc.ctx is not ctx:
                    raise
                abort = exc
            except Exception as exc:
                failure = exc
            finally:
                _ACTIVE_IMPORTS.reset(token)
            # Acted on after the handlers exit, so the next load and a replayed failure
            # don't carry the handled exception as their `__context__`.
            if abort is not None:
                ctx.pending.extend(abort.chain)
                continue
            ctx.pending.pop()
            if failure is not None:
                # The traceback starts at this frame; raising it here or replaying it in
                # the resolver adds the raising frame, so this one would show twice.
                traceback = failure.__traceback__.tb_next
                if not ctx.pending:
                    raise failure.with_traceback(traceback)
                ctx.pending[-1].failures[(top.path, top.outer)] = (failure, traceback)
    finally:
        ctx.pending = []
        ctx.nested = []
        ctx.loading.clear()
    return ctx.flows[first]


def load_flow(
    text: str,
    *,
    child_resolver: Optional[ChildResolver] = None,
    search_paths: Optional[list] = None,
    source_label: str = "<flow>",
) -> ComposedFlow:
    """
    Load Compose YAML text into a runnable `ComposedFlow`.

    Parses, compiles, and validates the flow in one pass. A `call:` resolves
    **defs-first** (an in-file `defs:` child flow), then a **`uses:` alias** (external,
    with the ref's `@<version>`); any other name raises. With neither `search_paths`
    nor `child_resolver`, a `uses:` alias raises — external resolution is additive. The loader derives each child flow's signature for binding
    checks and bakes its compiled graph onto the built `CallNode` so `run` drives it.

    Args:
        text (`str`):
            The Compose-shaped YAML source of the flow.
        child_resolver (`ChildResolver`, *optional*, defaults to `None`):
            An explicit `(flow_id, version) -> ComposedFlow` resolver for external flows
            (a fake, or a future marketplace). When given, it replaces the local file
            resolver: `search_paths` and the flow's `system: paths:` are not searched.
        search_paths (`list[str | Path]`, *optional*, defaults to `None`):
            Directories the default local file resolver searches for `uses:` targets
            (`<dir>/<path>.yaml`). Pass the flow file's own directory so relative
            `call:`/`uses:` references resolve; the list is extended by the flow's
            `system: paths:`.
        source_label (`str`, *optional*, defaults to `"<flow>"`):
            Human-facing name retained by load diagnostics for this source text.

    Returns:
        `ComposedFlow`:
            The compiled, validated flow ready for `run_flow`.

    Raises:
        `LoadError`:
            On any malformed or inconsistent flow, located at the offending `.yaml`
            line when source provenance is available.
        `DefaultsError`:
            A `YIAN_<NAME>` variable holds a value its entry does not accept. This
            is a host environment error, not a flow error, and is a `ValueError`
            rather than a `LoadError`.

    Example:
        ```python
        from pathlib import Path

        from yian import load_flow

        path = Path("hello.yaml")
        loaded = load_flow(
            path.read_text(),
            search_paths=[path.parent],
            source_label=path.name,
        )
        ```
    """
    # A bad YIAN_<NAME> value fails here, before any parsing.
    current()
    if type(source_label) is not str:
        raise TypeError("source_label must be a string")
    document = parse_source_document(text, source_label)
    flow = _load_flow(document, child_resolver, search_paths, _LoadCtx())
    # Only the root load sees the fully assembled graph, imports included;
    # children loaded for uses: are checked as part of their importer.
    try:
        resolve_memory_layout(flow.compiled)
    except MemorySetupError as exc:
        raise _memory_setup_load_error(flow, exc) from exc
    return flow


def _validate_authored_llm_config(
    llm_config: Any, where: str, *, diagnostics: LoadDiagnostics, cursor: SourceCursor | None,
) -> None:
    """Check a flow- or def-level ``llm_config`` the way an AGENT node's own is checked.

    The layer must be a mapping of `LLMConfig` fields without ``api_key``, or
    ``base_url`` other than beside ``provider: azure``, which only the host sets. A text
    field YAML read as non-text (such as ``model: 5``) is named by `not_text_error`; any
    other violation keeps pydantic's message.
    """
    if llm_config is None:
        return
    source_range = (
        cursor.range_of("llm_config")
        if cursor is not None and cursor.has("llm_config")
        else None
    )
    if not isinstance(llm_config, Mapping):
        raise diagnostics.error(f"{where}: llm_config must be a mapping", range=source_range)
    message = host_only_field_error(llm_config)
    if message is not None:
        raise diagnostics.error(f"{where}: {message}", range=source_range)
    try:
        LLMConfig(**llm_config)
    except ValidationError as exc:
        detail = not_text_detail(exc)
        raise diagnostics.error(
            f"{where}: llm_config: "
            + (not_text_error(*detail) if detail is not None else str(exc)),
            range=source_range,
        ) from exc
    except Exception as exc:
        raise diagnostics.error(f"{where}: llm_config: {exc}", range=source_range) from exc


def _validate_flow_env(
    env: dict[str, Any], *, diagnostics: LoadDiagnostics, cursor: SourceCursor,
) -> None:
    """Refuse a flow `env:` value too deep or holding itself, or an invalid `max_tool_iterations`.

    The nesting and recursive alias checks run first, so a deep `max_tool_iterations` is
    not shown in full; a `max_tool_iterations` default must be one every AGENT node would
    accept when it runs.
    """
    check_env_values(
        env, "flow ", cursor.child("env") if cursor.has("env") else None, diagnostics
    )
    if "max_tool_iterations" in env and not is_max_tool_iterations(env["max_tool_iterations"]):
        raise diagnostics.error(
            "flow `env.max_tool_iterations` must be a positive int or -1 (no cap), "
            f"got {env['max_tool_iterations']!r}",
            range=cursor.child("env").range_of("max_tool_iterations"),
        )


def _typedef_error_range(root: SourceCursor, path: tuple) -> Optional[SourceRange]:
    """Return the source range of the `typedefs:` entry or field a typedef error names.

    Args:
        root (`SourceCursor`):
            The flow document's root cursor.
        path (`tuple`):
            The error's `(typedef,)` or `(typedef, field)` path; empty for a block error.

    Returns:
        `SourceRange | None`:
            The field's value range, else the typedef's key range, else the `typedefs:`
            key range; `None` when the flow has no `typedefs:` section.
    """
    if not root.has("typedefs"):
        return None
    typedefs = root.child("typedefs")
    if not path or not typedefs.has(path[0]):
        return root.key_range_of("typedefs")
    entry = typedefs.child(path[0])
    if len(path) > 1 and entry.has(path[1]):
        return entry.range_of(path[1])
    return typedefs.key_range_of(path[0])


def _load_flow(
    document: SourceDocument,
    child_resolver,
    search_paths,
    ctx: "_LoadCtx",
) -> ComposedFlow:
    with call_nesting_scope():
        return _load_flow_body(document, child_resolver, search_paths, ctx)


def _load_flow_body(
    document: SourceDocument,
    child_resolver,
    search_paths,
    ctx: "_LoadCtx",
) -> ComposedFlow:
    root = document.root
    diagnostics = LoadDiagnostics.from_document(document)
    f = parse_file(document)
    _reject_authored_private_callable_names(root, f.defs, f.uses)
    _validate_authored_llm_config(f.llm_config, "flow", diagnostics=diagnostics, cursor=root)
    _validate_flow_env(f.env, diagnostics=diagnostics, cursor=root)
    _validate_system(
        f.system,
        root.child("system") if root.has("system") else None,
    )

    # Typedefs are read as plain values, so a tag in them is rejected before they are.
    reject_unread_tags(document, set(), diagnostics, typedefs_only=True)
    try:
        registry = read_typedefs(f.typedefs)
    except TypeCheckError as exc:
        raise diagnostics.error(
            f"bad typedefs: {exc}", range=_typedef_error_range(root, exc.path)
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
    uses_cursor = root.child("uses") if root.has("uses") else None
    callable_scope = _CallableScope(
        f.defs,
        registry,
        external,
        f.uses,
        defs_cursor,
        uses_cursor=uses_cursor,
        diagnostics=diagnostics,
    )
    root_memories = (
        parse_memory_declarations(
            f.long_term_memory,
            diagnostics=diagnostics,
            source_range=root.range_of("long_term_memory"),
            site=DeclSite((), ("root",), None),
        )
        if root.has("nodes") and f.long_term_memory is not None
        else ()
    )
    # Imports load before defs, from this file's base stack: a def calling an alias then
    # hits the cache instead of loading the imported file at the bottom of its own chain.
    # A collision is reported before a colliding alias's broken target.
    collide = sorted(set(f.uses) & set(f.nodes))
    if collide:
        raise diagnostics.error(f"uses: alias {collide} collides with a node id")
    for alias in f.uses:
        callable_scope.resolve(alias)
    callable_scope.eager_load_defs()

    staged = _assemble(
        inputs_section=f.inputs,
        inputs_cursor=_source_child(root, "input"),
        nodes_cursor=flow_nodes_cursor(document, f.nodes),
        outputs_section=f.outputs,
        asserts_section=f.asserts,
        registry=registry,
        scope=callable_scope,
        owner=("root",),
        uses_aliases=tuple(f.uses),
        version=f.version,
        name=f.name,
        description=f.description,
        flow_llm_config=f.llm_config,
        flow_env=f.env,
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
        flow_id=f.id,
        memories=root_memories,
        memories_cursor=(
            root.child("long_term_memory")
            if root.has("nodes") and root.has("long_term_memory")
            else None
        ),
    )
    # Defs, inline bodies, and the root are assembled now, so every slot has read
    # its tags; a `uses:` child checks its own document in its own load.
    reject_unread_tags(
        document, callable_scope.catalog_builder.consumed_tags, diagnostics
    )
    return replace(
        callable_scope.finalize(staged), import_depth=callable_scope.import_depth
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


# How many def builds `resolve` nests above the def `eager_load_defs` is building before
# it hands the next callee to the scheduler's stack. Nesting lets a def build its callees
# in one pass, so a def that calls many defs written after it is built once when their
# def subtrees fit under the bound. It does not spare every rebuild: each callee whose def
# subtree goes deeper than the bound aborts its caller's build, so a def is built again
# once per such callee, and a def calling k deep chains is built k + 1 times. The bound
# keeps a long def chain to one to two hundred Python frames per build the scheduler runs.
_MAX_NESTED_DEF_BUILDS = 8


@dataclass
class _PendingDef:
    """One def on the callee-first build stack, or one def build nested on the Python stack.

    `active_calls` is the call count its build starts from, and `outer` is the
    in-progress set when it was called, without itself. `failures` maps each callee that
    failed to build while this def was on top, keyed by the callee and its `outer`, to
    the exception and traceback it failed with.
    """

    name: str
    active_calls: int
    outer: frozenset[str]
    failures: dict[tuple[str, frozenset[str]], tuple[Exception, Any]] = field(
        default_factory=dict
    )


class _BuildFirst(BaseException):
    """Asks a scope building defs callee-first to build a def before its callers.

    It is a `BaseException` so the `except Exception` sites a def build passes through
    let it reach the scope's scheduler.

    Args:
        scope (`_CallableScope`): The scope whose scheduler handles it.
        chain (`list[_PendingDef]`): The def builds nested on the Python stack when the
            call was made, outermost first, then the called def.
        built (`list[_PendingDef]`): The defs built nested in the aborted build and still
            built when the call was made (a failed nested build rolls its defs back), in
            the order they were built. Aborting rolls them back, so they are built again
            before the called def, which would find them built.
    """

    def __init__(
        self, scope: _CallableScope, chain: list[_PendingDef], built: list[_PendingDef]
    ) -> None:
        super().__init__(chain[-1].name)
        self.scope = scope
        self.chain = chain
        self.built = built


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
        uses_cursor (`SourceCursor | None`, *optional*, defaults to `None`):
            Cursor over the authored ``uses:`` mapping. It locates alias resolution
            failures without retaining a parallel section-line map.
        catalog_builder (`_SourceCatalogBuilder | None`, *optional*):
            Mutable document-local registry shared by root, definitions, and
            inline bodies. A private builder is created for programmatic scopes.
        diagnostics (`LoadDiagnostics | None`, *optional*):
            Physical document authority for load diagnostics. When omitted, it is
            derived from the first retained declaration cursor, or source-free.
    """

    def __init__(
        self,
        defs_section: dict,
        registry,
        external: Optional[ChildResolver],
        uses: dict,
        defs_cursor: SourceCursor | None = None,
        uses_cursor: SourceCursor | None = None,
        catalog_builder: _SourceCatalogBuilder | None = None,
        diagnostics: LoadDiagnostics | None = None,
    ) -> None:
        source_cursor = defs_cursor if defs_cursor is not None else uses_cursor
        owned_diagnostics = (
            diagnostics
            if diagnostics is not None
            else LoadDiagnostics.from_cursor(source_cursor)
        )
        both = sorted(set(defs_section) & set(uses))
        if both:
            collision_cursor = _source_child(defs_cursor, both[0])
            raise owned_diagnostics.error(
                f"name(s) {both} declared in both defs: and uses: "
                f"(a callable has one namespace)",
                range=(collision_cursor.range if collision_cursor is not None else None),
            )

        self._defs_section = defs_section
        self._registry = registry
        self._external = external
        self._uses = uses
        self._defs_cursor = defs_cursor
        self._uses_cursor = uses_cursor
        self._diagnostics = owned_diagnostics
        self._catalog_builder = (
            catalog_builder
            if catalog_builder is not None
            else _SourceCatalogBuilder()
        )
        self._private: dict[str, _PrivateCallable] = {}
        self._built: dict[str, ComposedFlow] = {}
        self._in_progress: set[str] = set()
        self._import_depth = 0
        # Set while `eager_load_defs` builds defs callee-first; `_pending` is its stack,
        # `_nested` the def builds nested on the Python stack above its top, and
        # `_nested_built` the defs built nested since the top's build started.
        self._scheduling = False
        self._pending: list[_PendingDef] = []
        self._nested: list[_PendingDef] = []
        self._nested_built: dict[str, _PendingDef] = {}

    @property
    def import_depth(self) -> int:
        """Return the longest chain of `uses:` imports among the aliases resolved so far."""
        return self._import_depth

    @property
    def catalog_builder(self) -> _SourceCatalogBuilder:
        """Return this physical file's mutable origin registry."""
        return self._catalog_builder

    @property
    def diagnostics(self) -> LoadDiagnostics:
        """Return this physical file's immutable load-diagnostic authority."""
        return self._diagnostics

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
                alias_cursor = _source_child(self._uses_cursor, alias)
                raise self._diagnostics.error(
                    f"uses: alias {alias!r} was not resolved before finalization",
                    range=(alias_cursor.range if alias_cursor is not None else None),
                )
            resolver = child.origin_resolver
            if resolver is None:
                continue
            if not isinstance(resolver, ResolverTable):
                alias_cursor = _source_child(self._uses_cursor, alias)
                raise self._diagnostics.error(
                    f"uses: alias {alias!r} did not retain a rebased resolver table",
                    range=(alias_cursor.range if alias_cursor is not None else None),
                )
            tables.append(resolver)
        return _finalize_composed_flow(
            staged,
            self._catalog_builder.freeze(),
            tuple(tables),
            self._diagnostics,
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
            raise self._diagnostics.error(
                f"compiler-generated callable {label!r} owned by {owner!r} must use "
                f"the reserved prefix {INLINE_FLOW_PREFIX!r}"
            )

        existing = self._private.get(flow_id)
        if existing is not None:
            if existing.owner == owner:
                return
            raise self._diagnostics.error(
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
            raise self._diagnostics.error(
                "compiler-generated callable collision in reserved prefix "
                f"{INLINE_FLOW_PREFIX!r}: {label!r} owned by {owner!r} conflicts "
                f"with an authored {authored_section}: name"
            )

        self._private[flow_id] = _PrivateCallable(owner, label, builder)

    def resolve(
        self, flow_id: str, version: Optional[str] = None
    ) -> ComposedFlow:
        """Resolve one callable once; a `uses:` declaration owns its external version.

        A `uses:` alias whose import would make the chain of `uses:` files deeper than
        `current().max_call_nesting` is refused: before loading when the files already
        being imported reach the limit, and after loading by the `import_depth` the
        resolved flow carries, so a cached or memoized file is held to the same rule.
        While `eager_load_defs` runs, a def not built yet is built inside its caller up to
        `_MAX_NESTED_DEF_BUILDS` nested def builds, and before its caller past that.

        Args:
            flow_id (`str`):
                The callable name: a compiler-generated callable, a `defs:` entry, or a
                `uses:` alias.
            version (`str`, *optional*, defaults to `None`):
                Unused; a `uses:` alias takes its version from its own ref.

        Returns:
            `ComposedFlow`:
                The built callable, cached for later calls.

        Raises:
            `LoadError`:
                Raised when the callable is unknown, recursive, fails to build, or its
                import chain is past the limit; a `uses:` failure is located at its alias.
            `_BuildFirst`:
                Raised while `eager_load_defs` runs when a def not built yet would nest
                past `_MAX_NESTED_DEF_BUILDS` def builds; the scope's scheduler in
                `_build_callee_first` catches it.
        """
        if flow_id in self._built:
            return self._built[flow_id]
        if flow_id in self._private:
            private = self._private[flow_id]
            if flow_id in self._in_progress:
                raise self._diagnostics.error(
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
                def_cursor = _source_child(self._defs_cursor, flow_id)
                raise self._diagnostics.error(
                    f"recursive defs reference: callable {flow_id!r} is defined in terms "
                    f"of itself (the baked call graph must be finite)",
                    range=(def_cursor.range if def_cursor is not None else None),
                )
            called = None
            if self._scheduling:
                called = _PendingDef(
                    flow_id, _ACTIVE_CALLS.get(), frozenset(self._in_progress)
                )
                failure = self._pending[-1].failures.get((flow_id, called.outer))
                if failure is not None:
                    exc, traceback = failure
                    raise exc.with_traceback(traceback)
                if len(self._nested) >= _MAX_NESTED_DEF_BUILDS:
                    # `_built` keeps build order, and a failed build's defs left it.
                    built = [
                        self._nested_built[name]
                        for name in self._built
                        if name in self._nested_built
                    ]
                    raise _BuildFirst(self, [*self._nested, called], built)
                self._nested.append(called)
            self._in_progress.add(flow_id)
            try:
                loaded = self._build_def(flow_id)
            finally:
                self._in_progress.discard(flow_id)
                if called is not None:
                    self._nested.pop()
            self._built[flow_id] = loaded
            if called is not None:
                self._nested_built[flow_id] = called
            return loaded
        if flow_id in self._uses:
            try:
                ref = parse_uses_ref(
                    self._uses[flow_id],
                    diagnostics=self._diagnostics,
                )
                if ref.scheme == "hub":
                    raise self._diagnostics.error(
                        f"callable {flow_id!r} (uses: {self._uses[flow_id]!r}): marketplace "
                        f"not supported yet (hub: scheme)"
                    )
                if ref.scheme is not None:
                    raise self._diagnostics.error(
                        f"callable {flow_id!r} (uses: {self._uses[flow_id]!r}): unknown "
                        f"scheme {ref.scheme!r}: (recognized: a local ref, or hub:)"
                    )
                if self._external is None:
                    raise self._diagnostics.error(
                        f"callable {flow_id!r} (uses: {self._uses[flow_id]!r}) needs a search "
                        f"path or child_resolver to resolve (load with search_paths=[...])"
                    )
                limit = current().max_call_nesting
                active = _ACTIVE_IMPORTS.get()
                # Checked before loading: loading the file would nest another level.
                if active >= limit:
                    raise self._import_too_deep(flow_id, active + 1, limit)
                token = _ACTIVE_IMPORTS.set(active + 1)
                try:
                    child = self._external(ref.path, ref.version)
                except LoadError:
                    raise
                except Exception as exc:  # a resolver's own failure is a load error
                    raise self._diagnostics.error(
                        f"callable {flow_id!r} (uses: {self._uses[flow_id]!r}): could not "
                        f"resolve: {exc}"
                    ) from exc
                finally:
                    _ACTIVE_IMPORTS.reset(token)
                loaded = _rebase_external_child(flow_id, child)
                depth = loaded.import_depth
                if type(depth) is not int or depth < 0:
                    raise self._diagnostics.error(
                        f"callable {flow_id!r} (uses: {self._uses[flow_id]!r}): resolved "
                        f"flow has import_depth {depth!r}; expected a non-negative int"
                    )
                # A file loaded earlier (cached or memoized) never ran on this count.
                if active + 1 + depth > limit:
                    raise self._import_too_deep(flow_id, active + 1 + depth, limit)
                self._import_depth = max(self._import_depth, 1 + depth)
            except LoadError as exc:
                alias_cursor = _source_child(self._uses_cursor, flow_id)
                raise self._diagnostics.prepend(
                    exc,
                    range=(alias_cursor.range if alias_cursor is not None else None),
                    note=f"uses alias {flow_id!r}",
                ) from exc
            self._built[flow_id] = loaded
            return loaded
        if flow_id.startswith(INLINE_FLOW_PREFIX):
            raise self._diagnostics.error(
                "compiler-generated callable with reserved prefix "
                f"{INLINE_FLOW_PREFIX!r} was not registered in this file scope"
            )
        raise self._diagnostics.error(
            f"callable {flow_id!r} is neither an in-file defs: callable nor a uses: "
            f"alias — external flows are reachable only through a uses: alias "
            f"(declare `uses: {{{flow_id}: <ref>}}`)"
        )

    def _import_too_deep(self, alias: str, levels: int, limit: int) -> LoadError:
        """Return the error for a `uses:` alias whose import makes the chain `levels` deep."""
        return self._diagnostics.error(
            f"callable {alias!r} (uses: {self._uses[alias]!r}): importing it makes a chain "
            f"of uses: files {levels} deep, past the call nesting limit of {limit} (a "
            "chain of uses: imports is held to the same limit as nested call, map, and "
            "loop bodies); flatten the chain"
        )

    def _build_def(self, name: str) -> ComposedFlow:
        """Build `defs:` entry `name` transactionally; the caller tracks it as in progress.

        Args:
            name (`str`): The `defs:` entry to build.

        Returns:
            `ComposedFlow`: The built def.
        """
        def_cursor = None
        if self._defs_cursor is not None:
            try:
                def_cursor = self._defs_cursor.child(name)
            except SourceAccessError:
                pass
        return self._build_transactionally(
            lambda: _load_def(
                name,
                self._defs_section[name],
                self._registry,
                self,
                def_cursor,
            )
        )

    def eager_load_defs(self) -> None:
        """Build every local def in authored mapping order before the parent flow.

        Defs are built callee-first, so a def chain loads without recursing per def.
        """
        self._scheduling = True
        try:
            for name in self._defs_section:
                if name not in self._built:
                    self._build_callee_first(name)
        finally:
            self._scheduling = False

    def _build_callee_first(self, first: str) -> None:
        """Build def `first` and the defs it calls, nesting at most a few def builds.

        While scheduling, `resolve` builds a def not built yet nested inside its caller,
        as outside scheduling, while fewer than `_MAX_NESTED_DEF_BUILDS` def builds are
        nested above the def on top of the stack. Past that it raises `_BuildFirst`,
        which aborts the build on top. The nested defs and then the called def are
        pushed on the stack, the called def is built first, and each def below it is
        built again once the def above it is done. The defs the aborted build had built
        nested are pushed last, so they are built again first, in the order they were
        built: the called def and the defs after it then find built what a nested build
        would. A pushed def builds with the in-progress set and active call count it was
        called with, which are what a build nested in its callers sees, so recursion and
        the call nesting limit are checked as before.

        Nesting does not spare every rebuild. Each callee whose def subtree goes deeper
        than `_MAX_NESTED_DEF_BUILDS` aborts its caller's build, so a def is built again
        once per such callee: a def calling k chains written after it, each deeper than
        the bound, is built k + 1 times. Build counts stay linear in the defs, but each
        rebuild repeats the caller's whole build, so a wide caller of deep chains loads
        slower than building each callee nested would.

        When a def on the stack fails, its exception is recorded for the def below it,
        keyed by the failed def and the in-progress set it was called with. While the
        def below builds again, `resolve` of the failed def with that same set raises
        it again, so the caller's own error handling wraps it as if the def had been
        built nested. The records are dropped when the def below leaves the stack. The
        failure of `first` is raised.

        This relies on def builds being deterministic given the in-progress set, active
        call count, and defs built so far, and on an aborted build leaving no trace but
        the defs it built, which are built again: `_build_transactionally` rolls back
        its cache entries, inline callables, and catalog sites, and no def build changes
        the `defs:` section.

        Args:
            first (`str`): The def to build.

        Raises:
            `LoadError`: Raised when `first`, or a def it calls, fails to build.
        """
        self._pending = [_PendingDef(first, _ACTIVE_CALLS.get(), frozenset())]
        try:
            while self._pending:
                top = self._pending[-1]
                self._in_progress = {*top.outer, top.name}
                self._nested_built = {}
                abort: _BuildFirst | None = None
                failure: Exception | None = None
                token = _ACTIVE_CALLS.set(top.active_calls)
                try:
                    self._built[top.name] = self._build_def(top.name)
                except _BuildFirst as exc:
                    if exc.scope is not self:
                        raise
                    abort = exc
                except Exception as exc:
                    failure = exc
                finally:
                    _ACTIVE_CALLS.reset(token)
                # Acted on after the handlers exit, so the next build and a replayed
                # failure don't carry the handled exception as their `__context__`.
                if abort is not None:
                    self._pending.extend(abort.chain)
                    self._pending.extend(reversed(abort.built))
                    continue
                self._pending.pop()
                if failure is not None:
                    # The traceback starts at this frame; raising it here or replaying it
                    # in `resolve` adds the raising frame, so this one would show twice.
                    traceback = failure.__traceback__.tb_next
                    if not self._pending:
                        raise failure.with_traceback(traceback)
                    self._pending[-1].failures[(top.name, top.outer)] = (failure, traceback)
        finally:
            # `resolve` pushes onto `_nested` before its `try`, so an interrupt there
            # would leave an entry no `finally` pops.
            self._pending = []
            self._nested = []
            self._nested_built = {}
            self._in_progress = set()


def _check_def_asserts(
    name: str,
    asserts: Any,
    def_cursor: SourceCursor | None,
    diagnostics: LoadDiagnostics,
) -> None:
    """Reject a multi-node `defs:` entry's `asserts:` that is not a list of strings.

    A def body is not validated by `ComposeFile`, so its `asserts:` is checked here, in the
    same words as a node's.

    Args:
        name (`str`): The `defs:` entry name, for the message.
        asserts (`Any`): The authored `asserts:` value; `None` means absent.
        def_cursor (`SourceCursor | None`): The authored def body, when retained.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.

    Raises:
        `LoadError`: Raised at the field when `asserts:` is set but not a list, and at the
            item for the first item YAML read as non-text.
    """
    if asserts is None:
        return
    asserts_cursor = _source_child(def_cursor, "asserts")
    field_range = asserts_cursor.range if asserts_cursor is not None else None
    if not isinstance(asserts, list):
        raise diagnostics.error(
            f"defs entry {name!r}: `asserts:` must be a list of strings", range=field_range
        )
    for i, item in enumerate(asserts):
        if isinstance(item, str):
            continue
        item_range = None
        # `_source_child` selects mapping keys only (`has` is False for a sequence index).
        if asserts_cursor is not None:
            try:
                item_range = asserts_cursor.child(i).range
            except SourceAccessError:
                pass
        raise diagnostics.error(
            f"defs entry {name!r}: " + not_text_error(f"asserts[{i}]", item),
            range=item_range or field_range,
        )


def _load_def(
    name: str,
    body,
    registry,
    scope: _CallableScope,
    def_cursor: SourceCursor | None = None,
) -> ComposedFlow:
    """Build one in-file `defs:` entry into a `ComposedFlow` (an inline child flow).

    Supports both a multi-node child flow (`{input?, nodes, output?, asserts?}`)
    and a flat single-node body (`{kind, ...}`). Both share the file's typedef
    registry and callable scope. ``def_cursor`` is the occurrence-local authored
    body: it retains absolute locations used by assembly and source diagnosis. A
    multi-node body's ``asserts:`` must be a list of strings, and its ``env:`` values must
    nest within the value limit, else a located `LoadError`.
    """
    diagnostics = scope.diagnostics
    if not isinstance(body, dict):
        raise diagnostics.error(
            f"defs entry {name!r}: body must be a mapping, got {type(body).__name__}",
            range=(def_cursor.range if def_cursor is not None else None),
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
        raise diagnostics.error(
            f"defs entry {name!r}: a child flow needs a `nodes:` body "
            f"(input?/nodes/output?) OR a single-node `kind:` form",
            range=(def_cursor.range if def_cursor is not None else None),
        )
    if "memory_policy" in body:
        raise diagnostics.error(
            f"defs entry {name!r}: field `memory_policy:` was renamed to "
            "`long_term_memory:`",
            range=(
                def_cursor.key_range_of("memory_policy")
                if def_cursor is not None
                else None
            ),
        )
    extra = sorted(set(body) - _DEF_ALLOWED)
    if extra:
        raise diagnostics.error(
            f"defs entry {name!r}: unknown field(s) {extra} "
            f"(allowed: {', '.join(sorted(_DEF_SURFACE_KEYS))})",
            range=(
                def_cursor.key_range_of(extra[0])
                if def_cursor is not None
                else None
            ),
        )
    _validate_authored_llm_config(
        body.get("llm_config"), f"defs entry {name!r}", diagnostics=diagnostics, cursor=def_cursor
    )
    def_env = body.get("env")
    if def_env is not None and (
        not isinstance(def_env, Mapping) or not all(isinstance(k, str) for k in def_env)
    ):
        raise diagnostics.error(
            f"defs entry {name!r}: `env:` must be a mapping with string keys",
            range=(def_cursor.range_of("env") if def_cursor is not None else None),
        )
    if def_env is not None:
        check_env_values(
            def_env,
            f"defs entry {name!r}: ",
            def_cursor.child("env") if def_cursor is not None else None,
            diagnostics,
        )
    _check_def_asserts(name, body.get("asserts"), def_cursor, diagnostics)
    nodes_cursor = _def_assembly_cursor(def_cursor, body["nodes"])
    try:
        def_memories = (
            parse_memory_declarations(
                body["long_term_memory"],
                diagnostics=diagnostics,
                source_range=(
                    def_cursor.range_of("long_term_memory")
                    if def_cursor is not None
                    else None
                ),
                site=DeclSite((), ("def", name), None),
            )
            if body.get("long_term_memory") is not None
            else ()
        )
        return _assemble(
            inputs_section=body.get("inputs") or {},
            inputs_cursor=(
                _source_child(def_cursor, "input")
                or _source_child(def_cursor, "inputs")
            ),
            nodes_cursor=nodes_cursor,
            outputs_section=body.get("outputs"),
            asserts_section=body.get("asserts") or [],  # enforced at the child seam
            registry=registry,
            scope=scope,
            owner=("def", name),
            flow_llm_config=body.get("llm_config") or {},
            flow_env=def_env or {},
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
            flow_id=name,
            memories=def_memories,
            memories_cursor=_source_child(def_cursor, "long_term_memory"),
        )
    except LoadError as exc:
        raise diagnostics.prepend(
            exc,
            range=(def_cursor.range if def_cursor is not None else None),
            message=f"defs entry {name!r}: {exc}",
            note=f"definition {name!r}",
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
    child flow and runs through the same `_assemble` as the multi-node form. The synthetic
    node projects its header, logic, generated bindings, and output type onto the flat
    authored def cursor without mutating that source."""
    diagnostics = scope.diagnostics
    param_types = body.get("inputs") or {}
    if not isinstance(param_types, dict):
        input_cursor = _source_child(def_cursor, "input") or _source_child(
            def_cursor, "inputs"
        )
        raise diagnostics.error(
            f"defs entry {name!r}: single-node `input:` must be a param map "
            "(`name: !TYPE`, e.g. `topic: !str`)",
            range=(
                input_cursor.range
                if input_cursor is not None
                else (def_cursor.range if def_cursor is not None else None)
            ),
        )
    nodes_cursor = _single_node_def_cursor(name, body, def_cursor)
    output_source = _source_child(def_cursor, "output")
    output_range = output_source.range if output_source is not None else None
    try:
        return _assemble(
            inputs_section=param_types,
            inputs_cursor=(
                _source_child(def_cursor, "input")
                or _source_child(def_cursor, "inputs")
            ),
            nodes_cursor=nodes_cursor,
            outputs_section=f"${{{name}.output}}",  # single value re-export (node-first)
            asserts_section=[],
            registry=registry,
            scope=scope,
            owner=("def", name),
            boundary_field_ranges=(
                _boundary_field_ranges(def_cursor)
                if def_cursor is not None
                else None
            ),
            outputs_range=output_range,
            flow_id=name,
        )
    except LoadError as exc:
        raise diagnostics.prepend(
            exc,
            range=(def_cursor.range if def_cursor is not None else None),
            message=f"defs entry {name!r}: {exc}",
            note=f"definition {name!r}",
        ) from exc
