"""kind=code — the deterministic-code node's descriptor + parse/build handle (#70).

One of the two-axis `compose/kinds/<kind>.py` modules: it owns the `code` node's frozen
`CodeDescriptor`, its located `parse(node_id, reader)` (which records each field's
`SourceRange`, incl. the `code:` body's own range), and its `build(desc, *, registry,
resolver, source_file)` (which constructs the runtime `CodeNode` + wiring, reconciling the
inline `def main(inputs):` wrapper offset so tracebacks frame at absolute YAML lines).
Registered into `KINDS`/`BINDING_FIELDS` at import so the shared dispatch routes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agent_composer.compose.errors import LoadError
from agent_composer.compose.reader import NodeReader
from agent_composer.events import SourceRange

# Code's own field surface (was the `code` row of parser._KIND_SPECS): `code` is the one
# required flat field; these flat keys are allowed besides the shared `_COMMON_FIELDS` + `kind`.
CODE_REQUIRED: frozenset[str] = frozenset({"code"})
CODE_ALLOWED: frozenset[str] = frozenset({"code", "inputs", "outputs", "asserts"})
# Descriptor fields carrying inline-call binding strings (the kind-blind walk in bindings.py):
# only `inputs` — the `code:` body is source text, not a `${...}` binding surface.
CODE_BINDING_FIELDS: tuple[str, ...] = ("inputs",)


@dataclass(frozen=True)
class CodeDescriptor:
    """kind=code — deterministic code (a `module:function` ref or a source snippet).

    `range`/`field_ranges` are located-parse metadata (the node body's `SourceRange` and
    per-field `SourceRange`s); `compare=False` keeps them out of `==` so a descriptor still
    equals an identically-configured one built without ranges. Populated by `parse`.
    """

    id: str
    code: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: Any = None
    asserts: list[str] = field(default_factory=list)
    node_name: Optional[str] = None
    depends_on: list[str] = field(default_factory=list)
    runs_after: list[str] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=dict)  # static per-node config; merged under flow env: at build
    # Located-parse metadata (see class docstring); excluded from equality.
    range: Optional[SourceRange] = field(default=None, compare=False)
    field_ranges: dict[str, SourceRange] = field(default_factory=dict, compare=False)


def parse(node_id: str, reader: NodeReader) -> CodeDescriptor:
    """Read one `kind: code` node body (via a located `NodeReader`) into a `CodeDescriptor`.

    Records the node body's `SourceRange` (`desc.range`) and each present field's
    `SourceRange` (`desc.field_ranges[field]`) — crucially the `code:` body's own range,
    which `build` uses to frame inline tracebacks. Field legality mirrors the legacy
    `parser._parse_node` code path (allowed/required flat keys, `asserts:`/`env:` shape).
    """
    from agent_composer.compose.parser import _COMMON_FIELDS

    body = reader.value
    allowed_keys = CODE_ALLOWED | _COMMON_FIELDS | {"kind"}
    for key in body:
        if key not in allowed_keys:
            raise LoadError(
                f"node {node_id!r} (kind=code): field {key!r} is not allowed "
                f"(allowed: {', '.join(sorted(allowed_keys))})",
                range=reader.range_of(key),
            )
    for key in CODE_REQUIRED:
        if key not in body:
            raise LoadError(
                f"node {node_id!r} (kind=code): missing required field {key!r}",
                range=reader._own_range(),
            )

    ranges: dict[str, SourceRange] = {}
    kwargs: dict[str, Any] = {"id": node_id}
    for key in CODE_ALLOWED:
        if key not in body:
            continue
        rng = reader.range_of(key)
        if rng is not None:
            ranges[key] = rng
        kwargs[key] = body[key]

    asserts = kwargs.get("asserts")
    if asserts is not None and not (
        isinstance(asserts, list) and all(isinstance(a, str) for a in asserts)
    ):
        raise LoadError(
            f"node {node_id!r} (kind=code): `asserts:` must be a list of strings",
            range=ranges.get("asserts"),
        )

    for key in ("node_name", "depends_on", "runs_after"):
        if key in body:
            kwargs[key] = body[key]
            rng = reader.range_of(key)
            if rng is not None:
                ranges[key] = rng
    if "env" in body:
        env = body["env"]
        if not isinstance(env, dict) or not all(isinstance(k, str) for k in env):
            raise LoadError(
                f"node {node_id!r} (kind=code): `env:` must be a mapping with string keys",
                range=reader.range_of("env"),
            )
        kwargs["env"] = env
        rng = reader.range_of("env")
        if rng is not None:
            ranges["env"] = rng

    return CodeDescriptor(**kwargs, range=reader._own_range(), field_ranges=ranges)


def build(
    desc: CodeDescriptor,
    *,
    registry,
    resolver=None,
    source_file: str = "<inline-code>",
    code_line: Optional[int] = None,
    code_field_line: Optional[int] = None,
):
    """Construct the runtime `CodeNode` (+ wiring) for one `CodeDescriptor`.

    Mirrors the legacy `build.build_leaf_node` code path (classify + syntax/return gates,
    node-side `params` + flow-owned `inputs` wiring, `output_type` from `outputs:`) and
    anchors inline tracebacks from two line hints, preferring the caller's over the desc's:

    - `code_line` — the body's content start line (leading-newline padding frames tracebacks
      at absolute YAML lines). Derived from the recorded `code:` range as `range.start_line + 1`
      for a multi-line block scalar (algebraically identical to `start_mark.line + 2`), else `1`
      (a single-line `module:function` ref has no body to pad); a caller may override it.
    - `code_field_line` — the `code:` value's own start line, the anchor for a reject /
      no-`return` load error (which carries no interior line); `range.start_line` when the
      caller does not supply it. `None` -> the loader anchors at the node header.

    `source_file` (the flow's file label) frames inline tracebacks at the flow filename.
    All three ride the shared `KindSpec.build` call from `build_leaf_node`. `resolver` is
    unused (leaf kind); it rides the shared signature for call/map/loop.
    """
    from agent_composer.compose.build import _output_type, _sink_params, _sink_wiring
    from agent_composer.nodes.code import CodeNode

    # Prefer the caller's explicit hints (the loader's source maps today); otherwise derive
    # them from the recorded `code:` range (the standalone / Step-7 range-threaded path).
    code_rng = desc.field_ranges.get("code")
    if code_field_line is None and code_rng is not None:
        code_field_line = code_rng.start_line
    if code_line is None:
        if code_rng is not None and code_rng.end_line > code_rng.start_line:
            code_line = code_rng.start_line + 1
        else:
            code_line = 1

    try:
        node = CodeNode(
            desc.id,
            code=desc.code,
            source_file=source_file,
            code_line=code_line,
            title=desc.node_name,
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
        raise LoadError(f"node {desc.id!r}: {exc}", line=line) from exc

    node.output_type = _output_type(desc.outputs, registry)
    node.params = _sink_params(desc.inputs)
    wiring = _sink_wiring(desc.inputs)
    node.range = desc.range
    node.field_ranges = dict(desc.field_ranges)
    return node, wiring


def register() -> None:
    """Register the `code` kind into the shared dispatch tables (idempotent)."""
    from agent_composer.compose.kinds import KindSpec, KINDS, BINDING_FIELDS

    KINDS["code"] = KindSpec(descriptor_cls=CodeDescriptor, parse=parse, build=build)
    BINDING_FIELDS["code"] = CODE_BINDING_FIELDS


register()
