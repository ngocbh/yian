"""Compose provenance contracts for inline CODE source."""

from __future__ import annotations

import pytest

from agent_composer.compile import rewrite_flow_origins
from agent_composer.compile.expand import clone_child
from agent_composer.compile.manifest import flow_fingerprint, flow_manifest
from agent_composer.compose import load_flow
from agent_composer.compose.nodes.code import CodeDescriptor
from agent_composer.compose.source import SourceDocument
from agent_composer.compose.source.catalog import _scope_origin
from agent_composer.diagnostics import OriginRef, SourceRange
from agent_composer.nodes.code import CodeNode


_CODE_ORIGIN = OriginRef("/s:root/s:node/s:worker/s:field/s:code")
_LINE_ORIGINS = (
    OriginRef("/s:root/s:node/s:worker/s:field/s:code/s:line/i:0"),
    OriginRef("/s:root/s:node/s:worker/s:field/s:code/s:line/i:1"),
)

_LITERAL_FLOW = """\
id: code-origins
name: code-origins
nodes:
  worker:
    kind: code
    outputs: int
    code: |-
      value = 1
      return value
output: ${worker.output}
"""


def _worker(text: str):
    """Load ``text`` and return its authored CODE node and resolver."""
    loaded = load_flow(text, source_label="code-origins.yaml")
    return loaded.compiled.nodes["worker"], loaded.origin_resolver


def _assert_resolves_code_cursor(node, resolver, cursor):
    """Assert a CODE node's carriers resolve to one retained scalar occurrence."""
    assert resolver is not None
    assert node.code_origin is not None
    whole_frame = resolver.resolve(node.code_origin)
    assert whole_frame is not None
    assert whole_frame.range == cursor.range
    assert cursor.document is not None
    assert whole_frame.text == cursor.document.text

    scalar_source = cursor.scalar_source
    assert scalar_source is not None
    assert scalar_source.constructed_line_ranges is not None
    line_frames = [resolver.resolve(origin) for origin in node.code_line_origins]
    assert all(frame is not None for frame in line_frames)
    assert all(frame.text == cursor.document.text for frame in line_frames)
    assert tuple(frame.range for frame in line_frames) == (
        scalar_source.constructed_line_ranges
    )


def _direct_code_node(**origins):
    """Construct a source-free CODE node with explicit provenance arguments."""
    return CodeNode("worker", code="return 1", **origins)


def _direct_code_descriptor(**origins):
    """Construct a source-free CODE descriptor with explicit provenance arguments."""
    return CodeDescriptor(id="worker", code="return 1", **origins)


def test_literal_code_carries_whole_scalar_and_exact_constructed_lines():
    node, resolver = _worker(_LITERAL_FLOW)

    assert node.code_origin == _CODE_ORIGIN
    assert node.code_line_origins == _LINE_ORIGINS
    assert resolver is not None
    assert resolver.resolve(node.code_origin).range == SourceRange(7, 11, 10, 1)
    assert tuple(
        resolver.resolve(origin).range for origin in node.code_line_origins
    ) == (
        SourceRange(8, 7, 8, 16),
        SourceRange(9, 7, 9, 19),
    )


def test_compact_root_code_origins_retain_the_projected_authored_scalar():
    text = """\
id: compact_worker
name: compact_worker
output: int
kind: code
code: |-
  value = 1
  return value
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    node = loaded.compiled.nodes["compact_worker"]
    code_cursor = document.root.child("code")

    assert node.code_origin == OriginRef(
        "/s:root/s:node/s:compact_worker/s:field/s:code"
    )
    assert node.code_line_origins == (
        OriginRef(
            "/s:root/s:node/s:compact_worker/s:field/s:code/s:line/i:0"
        ),
        OriginRef(
            "/s:root/s:node/s:compact_worker/s:field/s:code/s:line/i:1"
        ),
    )
    _assert_resolves_code_cursor(node, loaded.origin_resolver, code_cursor)


@pytest.mark.parametrize(
    "code_source",
    [
        '    code: "return 1"\n',
        "    code: >-\n      return 1\n",
    ],
    ids=["quoted", "folded"],
)
def test_nonliteral_code_carries_only_whole_scalar_fallback(code_source):
    text = (
        "id: code-origins\n"
        "name: code-origins\n"
        "nodes:\n"
        "  worker:\n"
        "    kind: code\n"
        "    outputs: int\n"
        f"{code_source}"
        "output: ${worker.output}\n"
    )

    node, resolver = _worker(text)

    assert node.code_origin == _CODE_ORIGIN
    assert node.code_line_origins == ()
    assert resolver is not None
    assert resolver.resolve(node.code_origin) is not None


def test_source_free_code_accepts_empty_origin_carriers():
    node = CodeNode(
        "worker",
        code="return 1",
        code_origin=None,
        code_line_origins=(),
    )

    assert node.code_origin is None
    assert node.code_line_origins == ()
    assert list(node.iter_local_origins()) == []


def test_code_origins_participate_in_local_traversal_and_rewrite():
    node_origin = OriginRef("node")
    code_origin = OriginRef("code")
    line_origins = (OriginRef("line/0"), OriginRef("line/1"))
    node = CodeNode(
        "worker",
        code="value = 1\nreturn value",
        code_origin=code_origin,
        code_line_origins=line_origins,
    )
    node.origin = node_origin

    assert list(node.iter_local_origins()) == [
        node_origin,
        code_origin,
        *line_origins,
    ]

    node.rewrite_local_origins(
        lambda origin: OriginRef(f"rewritten/{origin.key}")
    )

    assert list(node.iter_local_origins()) == [
        OriginRef("rewritten/node"),
        OriginRef("rewritten/code"),
        OriginRef("rewritten/line/0"),
        OriginRef("rewritten/line/1"),
    ]
    assert node.code_origin == OriginRef("rewritten/code")
    assert node.code_line_origins == (
        OriginRef("rewritten/line/0"),
        OriginRef("rewritten/line/1"),
    )


def test_code_origins_are_noncanonical_manifest_and_fingerprint_metadata():
    loaded = load_flow(_LITERAL_FLOW)
    original = loaded.compiled
    rewritten = rewrite_flow_origins(
        original,
        lambda origin: OriginRef(f"rewritten/{origin.key}"),
    )

    original_node = original.nodes["worker"]
    rewritten_node = rewritten.nodes["worker"]
    assert rewritten_node.code_origin != original_node.code_origin
    assert rewritten_node.code_line_origins != original_node.code_line_origins
    assert flow_manifest(rewritten) == flow_manifest(original)
    assert flow_fingerprint(rewritten) == flow_fingerprint(original)


def test_multi_node_def_code_origins_use_the_definition_lexical_path():
    text = """\
id: multi-def-code-origins
name: multi-def-code-origins
defs:
  child:
    nodes:
      worker:
        kind: code
        outputs: int
        code: |-
          value = 1
          return value
    output: ${worker.output}
nodes:
  invoke:
    kind: call
    call: child
output: ${invoke.output}
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    node = loaded.compiled.nodes["invoke"].child.nodes["worker"]
    code_cursor = (
        document.root.child("defs")
        .child("child")
        .child("nodes")
        .child("worker")
        .child("code")
    )

    assert node.code_origin == OriginRef(
        "/s:def/s:child/s:node/s:worker/s:field/s:code"
    )
    assert node.code_line_origins == (
        OriginRef("/s:def/s:child/s:node/s:worker/s:field/s:code/s:line/i:0"),
        OriginRef("/s:def/s:child/s:node/s:worker/s:field/s:code/s:line/i:1"),
    )
    _assert_resolves_code_cursor(node, loaded.origin_resolver, code_cursor)


def test_flat_def_code_origins_retain_the_projected_authored_scalar():
    text = """\
id: flat-def-code-origins
name: flat-def-code-origins
defs:
  child:
    kind: code
    outputs: int
    code: |-
      value = 1
      return value
nodes:
  invoke:
    kind: call
    call: child
output: ${invoke.output}
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    node = loaded.compiled.nodes["invoke"].child.nodes["child"]
    code_cursor = document.root.child("defs").child("child").child("code")

    assert node.code_origin == OriginRef(
        "/s:def/s:child/s:node/s:child/s:field/s:code"
    )
    assert node.code_line_origins == (
        OriginRef("/s:def/s:child/s:node/s:child/s:field/s:code/s:line/i:0"),
        OriginRef("/s:def/s:child/s:node/s:child/s:field/s:code/s:line/i:1"),
    )
    _assert_resolves_code_cursor(node, loaded.origin_resolver, code_cursor)


def test_inline_child_code_origins_include_the_enclosing_node_path():
    text = """\
id: inline-code-origins
name: inline-code-origins
nodes:
  invoke:
    kind: call
    nodes:
      worker:
        kind: code
        outputs: int
        code: |-
          value = 1
          return value
    output: ${worker.output}
output: ${invoke.output}
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    node = loaded.compiled.nodes["invoke"].child.nodes["worker"]
    code_cursor = (
        document.root.child("nodes")
        .child("invoke")
        .child("nodes")
        .child("worker")
        .child("code")
    )

    assert node.code_origin == OriginRef(
        "/s:root/s:node/s:invoke/s:inline/s:node/s:worker/s:field/s:code"
    )
    assert node.code_line_origins == (
        OriginRef(
            "/s:root/s:node/s:invoke/s:inline/s:node/s:worker/"
            "s:field/s:code/s:line/i:0"
        ),
        OriginRef(
            "/s:root/s:node/s:invoke/s:inline/s:node/s:worker/"
            "s:field/s:code/s:line/i:1"
        ),
    )
    _assert_resolves_code_cursor(node, loaded.origin_resolver, code_cursor)


def test_external_alias_rebases_every_code_origin_without_mutating_child():
    child = load_flow(_LITERAL_FLOW, source_label="child.yaml")
    parent = """\
id: parent
name: parent
nodes:
  invoke:
    kind: call
    call: child_view
output: ${invoke.output}
uses:
  child_view: child
"""

    loaded = load_flow(
        parent,
        child_resolver=lambda _path, _version=None: child,
    )
    canonical = child.compiled.nodes["worker"]
    embedded = loaded.compiled.nodes["invoke"].child.nodes["worker"]

    assert canonical.code_origin == _CODE_ORIGIN
    assert canonical.code_line_origins == _LINE_ORIGINS
    assert embedded.code_origin == _scope_origin(
        "child_view", canonical.code_origin
    )
    assert embedded.code_line_origins == tuple(
        _scope_origin("child_view", origin)
        for origin in canonical.code_line_origins
    )
    assert loaded.origin_resolver is not None
    assert child.origin_resolver is not None
    assert loaded.origin_resolver.resolve(embedded.code_origin) == (
        child.origin_resolver.resolve(canonical.code_origin)
    )
    assert tuple(
        loaded.origin_resolver.resolve(origin)
        for origin in embedded.code_line_origins
    ) == tuple(
        child.origin_resolver.resolve(origin)
        for origin in canonical.code_line_origins
    )


def test_generated_child_clone_preserves_code_origins_for_original_resolver():
    loaded = load_flow(_LITERAL_FLOW)
    original = loaded.compiled.nodes["worker"]

    cloned = clone_child(loaded.compiled, callsite="invoke", record={})
    generated = cloned.nodes["invoke/worker"]

    assert generated is not original
    assert generated.code_origin == original.code_origin
    assert generated.code_line_origins == original.code_line_origins
    assert loaded.origin_resolver is not None
    assert loaded.origin_resolver.resolve(generated.code_origin) == (
        loaded.origin_resolver.resolve(original.code_origin)
    )
    assert tuple(
        loaded.origin_resolver.resolve(origin)
        for origin in generated.code_line_origins
    ) == tuple(
        loaded.origin_resolver.resolve(origin)
        for origin in original.code_line_origins
    )


def test_formatting_moves_resolved_ranges_without_changing_semantic_keys():
    moved = """\
id: code-origins
name: code-origins

# A formatting-only comment moves the authored scalar.
nodes:
  worker:
    kind: code
    outputs: int
    # Another formatting-only comment.
    code: |-
      value = 1
      return value
output: ${worker.output}
"""
    baseline_document = SourceDocument.parse(_LITERAL_FLOW)
    moved_document = SourceDocument.parse(moved)
    baseline = load_flow(_LITERAL_FLOW)
    relocated = load_flow(moved)
    baseline_node = baseline.compiled.nodes["worker"]
    relocated_node = relocated.compiled.nodes["worker"]
    baseline_cursor = (
        baseline_document.root.child("nodes").child("worker").child("code")
    )
    moved_cursor = moved_document.root.child("nodes").child("worker").child("code")

    assert relocated_node.code_origin == baseline_node.code_origin
    assert relocated_node.code_line_origins == baseline_node.code_line_origins
    assert flow_fingerprint(relocated.compiled) == flow_fingerprint(baseline.compiled)
    _assert_resolves_code_cursor(
        baseline_node,
        baseline.origin_resolver,
        baseline_cursor,
    )
    _assert_resolves_code_cursor(
        relocated_node,
        relocated.origin_resolver,
        moved_cursor,
    )
    assert baseline_cursor.range != moved_cursor.range
    assert baseline_cursor.scalar_source.constructed_line_ranges != (
        moved_cursor.scalar_source.constructed_line_ranges
    )


@pytest.mark.parametrize(
    "factory",
    [_direct_code_node, _direct_code_descriptor],
    ids=["node", "descriptor"],
)
@pytest.mark.parametrize(
    "origins",
    [
        {"code_origin": "not-an-origin"},
        {"code_line_origins": [OriginRef("line/0")]},
        {"code_line_origins": (OriginRef("line/0"), "not-an-origin")},
    ],
    ids=["whole-not-origin", "lines-not-tuple", "line-item-not-origin"],
)
def test_code_origin_carriers_reject_invalid_shapes(factory, origins):
    with pytest.raises(TypeError):
        factory(**origins)
