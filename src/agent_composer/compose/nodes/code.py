"""kind=code — the deterministic-code node's descriptor + parse/build handle.

One of the two-axis `compose/nodes/<kind>.py` modules: it owns the `code` node's frozen
`CodeDescriptor`, inheriting the template `parse` from `NodeDescriptor` (which records each
field's `SourceRange`, incl. the `code:` body's own range), and overriding `build` (which
constructs the runtime `CodeNode` + wiring, reconciling the inline `def main(inputs):` wrapper
offset so tracebacks frame at absolute YAML lines). Its source-origin hook registers the whole
`code:` scalar and exact literal constructed lines without retaining YAML objects. Registered
into `KINDS`/`BINDING_FIELDS` at import so the shared dispatch routes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from agent_composer.compose.nodes.base import BuildContext, BuildResult, NodeDescriptor
from agent_composer.compose.source import SourceCursor
from agent_composer.diagnostics import OriginRef, SourceRange
from agent_composer.nodes.code import CodeNode

if TYPE_CHECKING:
    from agent_composer.compose.source.catalog import _SourceCatalogBuilder


def _line_range(line: int | None) -> SourceRange | None:
    """Return a zero-width source range for one trustworthy absolute line.

    Args:
        line (`int | None`): Positive 1-based line, or `None` when unavailable.

    Returns:
        `SourceRange | None`: A point at column one, or `None` for an invalid line.
    """
    if type(line) is not int or line < 1:
        return None
    return SourceRange(line, 1, line, 1)


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

        Folded, quoted, projected-with-a-changed-value, and source-free scalars
        retain only the whole-value fallback. Exact literal block lines are stored
        in constructed-value order, independent of physical YAML coordinates.

        Args:
            cursor (`SourceCursor`): Normalized authored CODE node occurrence.
            site_path (`tuple[str | int, ...]`): Complete semantic node path.
            catalog_builder (`_SourceCatalogBuilder`): File-owned origin registry.

        Returns:
            `CodeDescriptor`: Copy carrying every available CODE source origin.
        """
        code_cursor = cursor.child("code")
        field_path = (*site_path, "field", "code")
        code_origin = catalog_builder.register(field_path, code_cursor)

        line_origins: list[OriginRef] = []
        scalar_source = code_cursor.scalar_source
        line_ranges = (
            scalar_source.constructed_line_ranges
            if scalar_source is not None
            else None
        )
        if line_ranges is not None:
            for index in range(len(line_ranges)):
                origin = catalog_builder.register_constructed_line(
                    (*field_path, "line", index),
                    code_cursor,
                    index,
                )
                if origin is None:
                    line_origins.clear()
                    break
                line_origins.append(origin)

        return replace(
            self,
            code_origin=code_origin,
            code_line_origins=tuple(line_origins),
        )

    def build(self, ctx: BuildContext) -> BuildResult:
        """Construct the runtime `CodeNode` (+ wiring) for this `CodeDescriptor`.

        Builds the code leaf (classify + syntax/return gates,
        node-side `params` + flow-owned `inputs` wiring, `output_contract` from `outputs:`) and
        anchors inline tracebacks from two line hints on `ctx`, preferring the caller's over the
        desc's:

        - `ctx.code_line` — the body's content start line (leading-newline padding frames
          tracebacks at absolute YAML lines). Derived from the recorded `code:` range as
          `range.start_line + 1` for a multi-line block scalar (algebraically identical to
          `start_mark.line + 2`), else `1` (a single-line `module:function` ref has no body to
          pad); a caller may override it.
        - `ctx.code_field_line` — an explicit `code:` start-line override for a reject /
          no-`return` load error (which carries no interior line). Without an override, the
          descriptor's full `code:` range is retained, then the node range is the fallback.

        The descriptor's source-neutral whole-code and exact-line origins are transferred to
        the runtime node unchanged. `ctx.source_file` (the flow's file label) frames inline
        tracebacks at the flow filename. `ctx.resolver` is unused (leaf kind); it rides the
        shared context for call/map/loop.
        """
        source_file = ctx.source_file
        code_line = ctx.code_line
        code_field_line = ctx.code_field_line
        # Prefer the caller's explicit hints (the loader's source maps today); otherwise derive
        # them from the recorded `code:` range (the standalone range-threaded path).
        code_rng = self.field_ranges.get(("field", "code"))
        if code_field_line is None and code_rng is not None:
            code_field_line = code_rng.start_line
        if code_line is None:
            if code_rng is not None and code_rng.end_line > code_rng.start_line:
                code_line = code_rng.start_line + 1
            else:
                code_line = 1
        code_error_range = (
            _line_range(code_field_line)
            if ctx.code_field_line is not None
            else code_rng
        ) or self.range

        try:
            node = CodeNode(
                self.id,
                code=self.code,
                source_file=source_file,
                code_line=code_line,
                code_origin=self.code_origin,
                code_line_origins=self.code_line_origins,
                title=self.node_name,
            )
        except (ValueError, SyntaxError) as exc:
            # CodeNode classifies `code:` (reference / inline / reject) and syntax- + has-return-
            # gates inline at construction; surface a reject, an inline SyntaxError, or a
            # missing-return as the loader's one error type (loud), anchored precisely:
            #  - an inline SyntaxError's padded compile gives an ABSOLUTE `lineno` (the real YAML
            #    line of the slip) — box it, but only with a real anchor (`code_line > 1`).
            #  - a reject / no-`return` ValueError has no trustworthy interior line, so retain
            #    the `code:` field range (or the node range when source metadata is partial).
            if isinstance(exc, SyntaxError) and code_line > 1:
                error_range = _line_range(exc.lineno) or code_error_range
            else:
                error_range = code_error_range
            raise ctx.diagnostics.error(
                f"node {self.id!r}: {exc}",
                range=error_range,
            ) from exc

        node.output_contract = self.output_contract(ctx.registry, ctx.diagnostics)
        node.params = self.sink_params()
        wiring = self.sink_wiring()
        return self._finish(node, wiring)
