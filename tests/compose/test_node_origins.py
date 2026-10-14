"""Integration tests for authored Compose node origins."""

from copy import deepcopy

import pytest

from yian.compile.model import END_ID, START_ID
from yian.compose import load_flow
from yian.compose.build.pipeline import _assemble
from yian.compose.helpers.errors import LoadError
from yian.compose.loader import _CallableScope
from yian.compose.parse.file import flow_nodes_cursor, parse_file
from yian.compose.source import SourceCursor, SourceDocument
from yian.diagnostics import OriginRef, OriginResolver
from yian.nodes.base import Node


_SHARED_NODE_IDS = """id: shared-origins
name: shared-origins
defs:
  named:
    nodes:
      worker:
        kind: code
        code: return 2
        output: int
    output: ${worker.output}
nodes:
  worker:
    kind: code
    code: return 1
    output: int
  named_call:
    kind: call
    call: named
  inline_call:
    kind: call
    nodes:
      worker:
        kind: code
        code: return 3
        output: int
    output: ${worker.output}
output:
  root: ${worker.output}
  named: ${named_call.output}
  inline: ${inline_call.output}
"""


def _assert_header_origin(
    catalog: OriginResolver,
    node: Node,
    expected_cursor: SourceCursor,
) -> None:
    """Assert that one authored node origin selects its mapping key."""
    assert isinstance(node.origin, OriginRef)
    frame = catalog.resolve(node.origin)

    assert frame is not None
    assert frame.text == expected_cursor.document.text
    assert frame.range == expected_cursor.key_range


def test_root_authored_node_origin_resolves_its_header():
    document = SourceDocument.parse(_SHARED_NODE_IDS)
    loaded = load_flow(_SHARED_NODE_IDS)

    assert loaded.origin_resolver is not None
    _assert_header_origin(
        loaded.origin_resolver,
        loaded.compiled.nodes["worker"],
        document.root.child("nodes").child("worker"),
    )


def test_root_def_and_inline_same_node_id_have_distinct_resolvable_origins():
    document = SourceDocument.parse(_SHARED_NODE_IDS)
    loaded = load_flow(_SHARED_NODE_IDS)
    catalog = loaded.origin_resolver
    assert catalog is not None

    root_node = loaded.compiled.nodes["worker"]
    def_node = loaded.compiled.nodes["named_call"].child.nodes["worker"]
    inline_node = loaded.compiled.nodes["inline_call"].child.nodes["worker"]
    origins = {root_node.origin, def_node.origin, inline_node.origin}

    assert None not in origins
    assert len(origins) == 3
    _assert_header_origin(
        catalog,
        root_node,
        document.root.child("nodes").child("worker"),
    )
    _assert_header_origin(
        catalog,
        def_node,
        document.root.child("defs").child("named").child("nodes").child("worker"),
    )
    _assert_header_origin(
        catalog,
        inline_node,
        document.root.child("nodes")
        .child("inline_call")
        .child("nodes")
        .child("worker"),
    )


def test_compact_def_node_origin_projects_to_the_def_header():
    text = """id: compact-origin
name: compact-origin
defs:
  compact_worker:
    kind: code
    code: return 4
    output: int
nodes:
  invoke:
    kind: call
    call: compact_worker
output: ${invoke.output}
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    compact_node = loaded.compiled.nodes["invoke"].child.nodes["compact_worker"]

    _assert_header_origin(
        loaded.origin_resolver,
        compact_node,
        document.root.child("defs").child("compact_worker"),
    )


def test_synthesized_boundary_and_inline_call_nodes_are_originless():
    text = """id: generated-origins
name: generated-origins
input:
  topic: !str
defs:
  echo:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: !str
    output: str
nodes:
  consumer:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: call(echo, topic=${input.topic})
    output: str
output: ${consumer.output}
"""
    nodes = load_flow(text).compiled.nodes

    assert nodes[START_ID].origin is None
    assert nodes[END_ID].origin is None
    assert nodes["__call_0"].origin is None
    assert isinstance(nodes["consumer"].origin, OriginRef)


def test_deepcopy_reuses_immutable_origin_resolver():
    loaded = load_flow(_SHARED_NODE_IDS)
    copied = deepcopy(loaded)

    assert copied.origin_resolver is loaded.origin_resolver
    assert copied.compiled.nodes["worker"].origin == (
        loaded.compiled.nodes["worker"].origin
    )
    assert copied.origin_resolver.resolve(copied.compiled.nodes["worker"].origin) == (
        loaded.origin_resolver.resolve(loaded.compiled.nodes["worker"].origin)
    )


def test_direct_assembly_uses_its_callable_scopes_document_catalog():
    """The file scope, not each assembly call, owns the shared builder."""
    text = """id: direct-origin
name: direct-origin
defs:
  child:
    kind: code
    code: return 1
    output: int
nodes:
  invoke:
    kind: call
    call: child
output: ${invoke.output}
"""
    document = SourceDocument.parse(text)
    parsed = parse_file(document)
    scope = _CallableScope(
        parsed.defs,
        {},
        None,
        parsed.uses,
        document.root.child("defs"),
    )
    scope.eager_load_defs()

    loaded = _assemble(
        inputs_section=parsed.inputs,
        nodes_cursor=flow_nodes_cursor(document, parsed.nodes),
        outputs_section=parsed.outputs,
        asserts_section=parsed.asserts,
        registry={},
        scope=scope,
        owner=("root",),
    )

    root_node = loaded.compiled.nodes["invoke"]
    child_node = root_node.child.nodes["child"]
    catalog = scope.catalog_builder.freeze()
    assert loaded.origin_resolver is None
    assert catalog.resolve(root_node.origin) is not None
    assert catalog.resolve(child_node.origin) is not None


def test_failed_nested_def_build_rolls_back_origins_and_success_cache():
    """Fixing and retrying a failed callable reuses every lexical site safely."""
    text = """id: retry-origin
name: retry-origin
defs:
  outer:
    nodes:
      invoke:
        kind: call
        call: inner
    output: ${missing.output}
  inner:
    nodes:
      worker:
        kind: code
        code: return 1
        output: int
    output: ${worker.output}
nodes: {}
"""
    document = SourceDocument.parse(text)
    parsed = parse_file(document)
    scope = _CallableScope(
        parsed.defs,
        {},
        None,
        parsed.uses,
        document.root.child("defs"),
    )

    with pytest.raises(LoadError):
        scope.resolve("outer")

    parsed.defs["outer"]["outputs"] = "${invoke.output}"
    retried = scope.resolve("outer")
    outer_node = retried.compiled.nodes["invoke"]
    inner_node = outer_node.child.nodes["worker"]

    catalog = scope.catalog_builder.freeze()
    assert retried.origin_resolver is None
    assert catalog.resolve(outer_node.origin) is not None
    assert catalog.resolve(inner_node.origin) is not None


def test_a_mapping_output_replaced_after_parsing_is_kept_as_given():
    """A retried callable reads its current output mapping, not the stale source."""
    text = """id: retry-mapping
name: retry-mapping
defs:
  outer:
    nodes:
      worker:
        kind: code
        code: return 1
        output: int
    output:
      r: {a: '${missing.output}'}
nodes: {}
"""
    document = SourceDocument.parse(text)
    parsed = parse_file(document)
    scope = _CallableScope(
        parsed.defs,
        {},
        None,
        parsed.uses,
        document.root.child("defs"),
    )

    with pytest.raises(LoadError):
        scope.resolve("outer")

    parsed.defs["outer"]["outputs"] = {"r": {"a": "${worker.output}"}}
    retried = scope.resolve("outer")

    (output,) = retried.compiled.outputs
    assert output.binding.source == {"a": "${worker.output}"}


def test_failed_def_build_discards_inline_builder_closures_before_retry():
    """A retry assembles the current inline body, not a failed attempt's snapshot."""
    defs = {
        "outer": {
            "nodes": {
                "broken": {
                    "kind": "code",
                    "code": "return (",
                    "outputs": "int",
                },
                "wrap": {
                    "kind": "call",
                    "nodes": {
                        "worker": {
                            "kind": "code",
                            "code": "return 1",
                            "outputs": "int",
                        }
                    },
                    "outputs": "${worker.output}",
                },
            },
            "outputs": "${wrap.output}",
        }
    }
    scope = _CallableScope(defs, {}, None, {})

    with pytest.raises(LoadError):
        scope.resolve("outer")

    defs["outer"]["nodes"]["broken"]["code"] = "return 0"
    defs["outer"]["nodes"]["wrap"]["nodes"]["worker"]["code"] = "return 2"
    retried = scope.resolve("outer")

    worker = retried.compiled.nodes["wrap"].child.nodes["worker"]
    assert worker.code == "return 2"
