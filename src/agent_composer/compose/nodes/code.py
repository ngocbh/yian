"""kind=code — the deterministic-code node's descriptor + parse/build handle.

One of the two-axis `compose/nodes/<kind>.py` modules: it owns the `code` node's frozen
`CodeDescriptor`, inheriting the template `parse` from `NodeDescriptor` (which records each
field's `SourceRange`, incl. the `code:` body's own range), and overriding `build` (which
constructs the runtime `CodeNode` + wiring and resolves load-time syntax origins through the
already-registered Compose catalog). Its source-origin hook registers the whole `code:` scalar
and exact literal constructed lines without retaining YAML objects. Registered into
`KINDS`/`BINDING_FIELDS` at import so the shared dispatch routes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from agent_composer.compose.nodes.base import BuildContext, BuildResult, NodeDescriptor
from agent_composer.compose.source import SourceCursor
from agent_composer.diagnostics import OriginRef
from agent_composer.nodes.code import CodeNode, select_inline_code_origin

if TYPE_CHECKING:
    from agent_composer.compose.source.catalog import _SourceCatalogBuilder

@dataclass(frozen=True, kw_only=True)
class CodeDescriptor(NodeDescriptor):
    """kind=code — deterministic code (a `module:function` ref or a source snippet).

    Inherits the common node fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` + the
    `range`/`field_ranges` parse metadata) from `NodeDescriptor`, along with the `inputs:`
    call-arg map and the template `parse` (no custom parse hook — code's field legality is the
    default allowed/required key check + the asserts-list check). The code-specific semantic
    fields (`code`, `asserts`) and diagnostic origin carriers (`code_origin`,
    `code_line_origins`) are declared here; `inputs`/`outputs` come from the base.

    Class attrs: `kind="code"`; `allowed` adds `code`/`inputs`/`outputs`/`asserts` to the
    commons; `required` is `{"code"}` (the one mandatory flat field); `binding_fields=("inputs",)`
    — the `code:` body is source text, not a `${...}` binding surface, so only `inputs` is walked.
    """

    kind = "code"
    allowed = NodeDescriptor.allow("code", "inputs", "outputs", "asserts")
    required = NodeDescriptor.require("code")
    binding_fields = ("inputs",)

    code: str
    asserts: list[str] = field(default_factory=list)
    code_origin: OriginRef | None = field(default=None, compare=False)
    code_line_origins: tuple[OriginRef, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        """Reject mutable or non-opaque CODE provenance."""
        super().__post_init__()
        if self.code_origin is not None and not isinstance(self.code_origin, OriginRef):
            raise TypeError("CODE source origin must be an OriginRef or None")
        if type(self.code_line_origins) is not tuple:
            raise TypeError("CODE line origins must be a tuple")
        if not all(isinstance(origin, OriginRef) for origin in self.code_line_origins):
            raise TypeError("CODE line origins must contain only OriginRef values")

    def with_source_origins(
        self,
        cursor: SourceCursor,
        *,
        site_path: tuple[str | int, ...],
        catalog_builder: "_SourceCatalogBuilder",
    ) -> "CodeDescriptor":
        """Attach the whole ``code:`` scalar and exact literal-line origins.

        Folded, quoted, and projected-with-a-changed-value scalars provide only a
        whole-value Compose site; source-free scalars provide none. Exact literal
        block lines are stored in constructed-value order, independent of physical
        YAML coordinates. A slot for which Compose cannot register a source site
        preserves any explicit programmatic origin already carried by the descriptor.

        Args:
            cursor (`SourceCursor`): Normalized authored CODE node occurrence.
            site_path (`tuple[str | int, ...]`): Complete semantic node path.
            catalog_builder (`_SourceCatalogBuilder`): File-owned origin registry.

        Returns:
            `CodeDescriptor`: Copy carrying every available CODE source origin.
        """
        code_cursor = cursor.child("code")
        field_path = (*site_path, "field", "code")
        registered_code_origin = catalog_builder.register(field_path, code_cursor)
        code_origin = (
            self.code_origin
            if registered_code_origin is None
            else registered_code_origin
        )

        code_line_origins = self.code_line_origins
        scalar_source = code_cursor.scalar_source
        line_ranges = (
            scalar_source.constructed_line_ranges
            if scalar_source is not None
            else None
        )
        if line_ranges is not None:
            registered_line_origins: list[OriginRef] = []
            for index in range(len(line_ranges)):
                origin = catalog_builder.register_constructed_line(
                    (*field_path, "line", index),
                    code_cursor,
                    index,
                )
                if origin is None:
                    break
                registered_line_origins.append(origin)
            else:
                code_line_origins = tuple(registered_line_origins)

        return replace(
            self,
            code_origin=code_origin,
            code_line_origins=code_line_origins,
        )

    def build(self, ctx: BuildContext) -> BuildResult:
        """Construct the runtime `CodeNode` (+ wiring) for this `CodeDescriptor`.

        Builds the code leaf (classify + syntax/return gates, node-side `params` +
        flow-owned `inputs` wiring, `output_contract` from `outputs:`). The
        descriptor's source-neutral whole-code and exact-line origins are transferred
        unchanged. A load-time `SyntaxError.lineno` uses the same wrapped-line selector
        as runtime exceptions, then `ctx.resolve_origin` presents the already-registered
        Compose site without another parse or catalog freeze. Reject and missing-return
        failures select the whole scalar. Legacy ranges remain only as a fallback for
        standalone descriptors built without registered origins.

        `ctx.source_file` is an opaque Python traceback label. `ctx.resolver` is unused
        by this leaf kind; it rides the shared context for call/map/loop.
        """
        code_rng = self.field_ranges.get(("field", "code"))
        code_error_range = code_rng or self.range

        try:
            node = CodeNode(
                self.id,
                code=self.code,
                source_file=ctx.source_file,
                code_origin=self.code_origin,
                code_line_origins=self.code_line_origins,
                title=self.node_name,
            )
        except (ValueError, SyntaxError) as exc:
            lineno = exc.lineno if isinstance(exc, SyntaxError) else None
            error_origin = select_inline_code_origin(
                lineno,
                code_origin=self.code_origin,
                code_line_origins=self.code_line_origins,
                node_origin=self.origin,
            )
            error_frame = (
                ctx.resolve_origin(error_origin)
                if ctx.resolve_origin is not None and error_origin is not None
                else None
            )
            message = exc.msg if isinstance(exc, SyntaxError) else str(exc)
            if error_frame is not None:
                error = ctx.diagnostics.error(
                    f"node {self.id!r}: {message}",
                    frames=(error_frame,),
                )
            else:
                error = ctx.diagnostics.error(
                    f"node {self.id!r}: {message}",
                    range=code_error_range,
                )
            raise error from exc

        node.output_contract = self.output_contract(ctx)
        node.params = self.sink_params()
        wiring = self.sink_wiring()
        return self._finish(node, wiring)
