"""Integration tests for authored Compose node origins."""

from copy import deepcopy

from agent_composer.compile.model import END_ID, START_ID
from agent_composer.compose import load_flow
from agent_composer.compose.source import SourceCatalog, SourceCursor, SourceDocument
from agent_composer.diagnostics import OriginRef
from agent_composer.nodes.base import Node


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
    catalog: SourceCatalog,
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

    assert isinstance(loaded._local_catalog, SourceCatalog)
    _assert_header_origin(
        loaded._local_catalog,
        loaded.compiled.nodes["worker"],
        document.root.child("nodes").child("worker"),
    )


def test_root_def_and_inline_same_node_id_have_distinct_resolvable_origins():
    document = SourceDocument.parse(_SHARED_NODE_IDS)
    loaded = load_flow(_SHARED_NODE_IDS)
    catalog = loaded._local_catalog

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
        loaded._local_catalog,
        compact_node,
        document.root.child("defs").child("compact_worker"),
    )


def test_synthesized_boundary_and_inline_call_nodes_are_originless():
    text = """id: generated-origins
name: generated-origins
input:
  topic: str
defs:
  echo:
    kind: code
    code: tests.engine._compose_codefns:echo
    input: {topic: str}
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


def test_deepcopy_reuses_immutable_local_catalog():
    loaded = load_flow(_SHARED_NODE_IDS)
    copied = deepcopy(loaded)

    assert copied._local_catalog is loaded._local_catalog
    assert copied.compiled.nodes["worker"].origin == (
        loaded.compiled.nodes["worker"].origin
    )
    assert copied._local_catalog.resolve(copied.compiled.nodes["worker"].origin) == (
        loaded._local_catalog.resolve(loaded.compiled.nodes["worker"].origin)
    )
