"""kind=code — the deterministic-code node's descriptor + parse/build handle.

One of the two-axis `compose/nodes/<kind>.py` modules: it owns the `code` node's frozen
`CodeDescriptor`, inheriting the template `parse` from `NodeDescriptor` (which records each
field's `SourceRange`, incl. the `code:` body's own range), and overriding `build` (which
constructs the runtime `CodeNode` + wiring, reconciling the inline `def main(inputs):` wrapper
offset so tracebacks frame at absolute YAML lines). Registered into `KINDS`/`BINDING_FIELDS`
at import so the shared dispatch routes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.nodes.base import BuildContext, BuildResult, NodeDescriptor
from agent_composer.nodes.code import CodeNode


@dataclass(frozen=True, kw_only=True)
class CodeDescriptor(NodeDescriptor):
    """kind=code — deterministic code (a `module:function` ref or a source snippet).

    Inherits the common node fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` + the
    `range`/`field_ranges` parse metadata) from `NodeDescriptor`, along with the `inputs:`
    call-arg map and the template `parse` (no custom parse hook — code's field legality is the
    default allowed/required key check + the asserts-list check). Only the code-specific fields
    (`code`, `asserts`) are declared here; `inputs`/`outputs` come from the base.

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

    def build(self, ctx: BuildContext) -> BuildResult:
        """Construct the runtime `CodeNode` (+ wiring) for this `CodeDescriptor`.

        Builds the code leaf (classify + syntax/return gates,
        node-side `params` + flow-owned `inputs` wiring, `output_type` from `outputs:`) and
        anchors inline tracebacks from two line hints on `ctx`, preferring the caller's over the
        desc's:

        - `ctx.code_line` — the body's content start line (leading-newline padding frames
          tracebacks at absolute YAML lines). Derived from the recorded `code:` range as
          `range.start_line + 1` for a multi-line block scalar (algebraically identical to
          `start_mark.line + 2`), else `1` (a single-line `module:function` ref has no body to
          pad); a caller may override it.
        - `ctx.code_field_line` — the `code:` value's own start line, the anchor for a reject /
          no-`return` load error (which carries no interior line); `range.start_line` when the
          caller does not supply it. `None` -> the loader anchors at the node header.

        `ctx.source_file` (the flow's file label) frames inline tracebacks at the flow filename.
        `ctx.resolver` is unused (leaf kind); it rides the shared context for call/map/loop.
        """
        source_file = ctx.source_file
        code_line = ctx.code_line
        code_field_line = ctx.code_field_line
        # Prefer the caller's explicit hints (the loader's source maps today); otherwise derive
        # them from the recorded `code:` range (the standalone range-threaded path).
        code_rng = self.field_ranges.get("code")
        if code_field_line is None and code_rng is not None:
            code_field_line = code_rng.start_line
        if code_line is None:
            if code_rng is not None and code_rng.end_line > code_rng.start_line:
                code_line = code_rng.start_line + 1
            else:
                code_line = 1

        try:
            node = CodeNode(
                self.id,
                code=self.code,
                source_file=source_file,
                code_line=code_line,
                title=self.node_name,
            )
        except (ValueError, SyntaxError) as exc:
            # CodeNode classifies `code:` (reference / inline / reject) and syntax- + has-return-
            # gates inline at construction; surface a reject, an inline SyntaxError, or a
            # missing-return as the loader's one error type (loud), anchored precisely:
            #  - an inline SyntaxError's padded compile gives an ABSOLUTE `lineno` (the real YAML
            #    line of the slip) — box it, but only with a real anchor (`code_line > 1`).
            #  - a reject / no-`return` ValueError has no trustworthy interior line, so fall to
            #    the `code:` field line (None -> the loader anchors at the node header).
            if isinstance(exc, SyntaxError) and code_line > 1:
                line = exc.lineno
            else:
                line = code_field_line
            raise LoadError(f"node {self.id!r}: {exc}", line=line) from exc

        node.output_type = self.output_type(ctx.registry)
        node.params = self.sink_params()
        wiring = self.sink_wiring()
        return self._finish(node, wiring)
