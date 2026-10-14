"""Integration tests for authored Compose node-input origins."""

from __future__ import annotations

import pytest

from yian.compose import load_flow
from yian.compose.build.pipeline import (
    _assemble,
    _stamp_node_wiring_origins,
)
from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.loader import _CallableScope
from yian.compose.parse.nodes import parse_nodes
from yian.compose.source import SourceCursor, SourceDocument
from yian.compose.source.catalog import _SourceCatalogBuilder
from yian.diagnostics import OriginRef, OriginResolver
from yian.nodes.base import Node
from yian.nodes.contracts import Binding, Parameter


def _parameter(node: Node, name: str) -> Parameter:
    """Return one exact compiled parameter by name."""
    matches = [parameter for parameter in node.params or [] if parameter.name == name]
    assert len(matches) == 1
    parameter = matches[0]
    assert type(parameter) is Parameter
    return parameter


def _assert_input_site(
    catalog: OriginResolver,
    node: Node,
    binding: Binding,
    name: str,
    cursor: SourceCursor,
    *,
    source,
) -> tuple[OriginRef, OriginRef]:
    """Assert key/value origins for one authored ordinary ``input:`` entry."""
    parameter = _parameter(node, name)
    assert isinstance(parameter.origin, OriginRef)
    assert isinstance(binding.origin, OriginRef)
    assert binding.source == source

    parameter_frame = catalog.resolve(parameter.origin)
    binding_frame = catalog.resolve(binding.origin)
    assert parameter_frame is not None
    assert binding_frame is not None
    assert parameter_frame.text == cursor.document.text
    assert binding_frame.text == cursor.document.text
    assert parameter_frame.range == cursor.key_range
    assert binding_frame.range == cursor.range
    return parameter.origin, binding.origin


def _assert_binding_site(
    catalog: OriginResolver,
    binding: Binding,
    cursor: SourceCursor,
    *,
    source,
) -> OriginRef:
    """Assert one authored value origin carried only by a flow ``Binding``.

    Args:
        catalog (`OriginResolver`):
            Resolver that must locate the binding origin.
        binding (`Binding`):
            Canonical flow-owned binding under test.
        cursor (`SourceCursor`):
            Expected authored value occurrence.
        source (`object`):
            Expected final semantic binding payload after desugaring.

    Returns:
        `OriginRef`:
            The verified binding origin for identity comparisons.
    """
    assert isinstance(binding.origin, OriginRef)
    assert binding.source == source

    frame = catalog.resolve(binding.origin)
    assert frame is not None
    assert frame.text == cursor.document.text
    assert frame.range == cursor.range
    return binding.origin


def test_root_def_and_inline_inputs_resolve_distinct_key_and_value_sites():
    text = """id: shared-input-origins
name: shared-input-origins
defs:
  named:
    nodes:
      worker:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: named
        output: str
    output: ${worker.output}
nodes:
  worker:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: root
    output: str
  named_call:
    kind: call
    call: named
  inline_call:
    kind: call
    nodes:
      worker:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: inline
        output: str
    output: ${worker.output}
output:
  root: ${worker.output}
  named: ${named_call.output}
  inline: ${inline_call.output}
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    catalog = loaded.origin_resolver
    assert catalog is not None

    root_node = loaded.compiled.nodes["worker"]
    named_node = loaded.compiled.nodes["named_call"].child.nodes["worker"]
    inline_node = loaded.compiled.nodes["inline_call"].child.nodes["worker"]
    sites = (
        (
            root_node,
            loaded.compiled.wiring["worker"]["x"],
            document.root.child("nodes").child("worker").child("input").child("x"),
            "root",
        ),
        (
            named_node,
            loaded.compiled.nodes["named_call"].child.wiring["worker"]["x"],
            document.root.child("defs")
            .child("named")
            .child("nodes")
            .child("worker")
            .child("input")
            .child("x"),
            "named",
        ),
        (
            inline_node,
            loaded.compiled.nodes["inline_call"].child.wiring["worker"]["x"],
            document.root.child("nodes")
            .child("inline_call")
            .child("nodes")
            .child("worker")
            .child("input")
            .child("x"),
            "inline",
        ),
    )

    origins = {
        origin
        for node, binding, cursor, source in sites
        for origin in _assert_input_site(
            catalog,
            node,
            binding,
            "x",
            cursor,
            source=source,
        )
    }
    assert len(origins) == 6


def test_compact_def_generated_binding_retains_projected_physical_site():
    text = """id: compact-input-origin
name: compact-input-origin
defs:
  compact:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: !str
    output: str
nodes:
  invoke:
    kind: call
    call: compact
    input:
      x: hello
output: ${invoke.output}
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    child = loaded.compiled.nodes["invoke"].child
    cursor = (
        document.root.child("defs").child("compact").child("input").child("x")
    )

    _assert_input_site(
        loaded.origin_resolver,
        child.nodes["compact"],
        child.wiring["compact"]["x"],
        "x",
        cursor,
        source="${input.x}",
    )


def test_compact_root_generated_binding_remains_originless():
    text = """id: compact_root_input_origin
name: compact_root_input_origin
input:
  x: !str
output: str
kind: code
code: tests.engine._compose_codefns:echo_x
"""
    loaded = load_flow(text)
    node = loaded.compiled.nodes["compact_root_input_origin"]
    binding = loaded.compiled.wiring["compact_root_input_origin"]["x"]

    assert _parameter(node, "x").origin is None
    assert binding.source == "${input.x}"
    assert binding.origin is None


def test_inline_call_rewrite_keeps_host_site_and_generated_call_originless():
    text = """id: rewritten-input-origin
name: rewritten-input-origin
defs:
  helper:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: !int
    output: int
nodes:
  consumer:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: call(helper, x=3)
    output: int
output: ${consumer.output}
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    host_cursor = (
        document.root.child("nodes")
        .child("consumer")
        .child("input")
        .child("x")
    )

    host = loaded.compiled.nodes["consumer"]
    _assert_input_site(
        loaded.origin_resolver,
        host,
        loaded.compiled.wiring["consumer"]["x"],
        "x",
        host_cursor,
        source="${__call_0.output}",
    )

    generated = loaded.compiled.nodes["__call_0"]
    generated_binding = loaded.compiled.wiring["__call_0"]["x"]
    assert _parameter(generated, "x").origin is None
    assert generated_binding.source == 3
    assert generated_binding.origin is None


def test_virtual_input_entries_remain_originless():
    loaded = _assemble(
        inputs_section={},
        nodes_cursor=SourceCursor.virtual(
            {
                "worker": {
                    "kind": "code",
                    "code": "tests.engine._compose_codefns:echo_x",
                    "inputs": {"x": "virtual"},
                    "outputs": "str",
                }
            }
        ),
        outputs_section="${worker.output}",
        asserts_section=[],
        registry={},
        scope=_CallableScope({}, {}, None, {}),
        owner=("root",),
    )

    node = loaded.compiled.nodes["worker"]
    binding = loaded.compiled.wiring["worker"]["x"]
    assert _parameter(node, "x").origin is None
    assert binding.source == "virtual"
    assert binding.origin is None


def test_adaptive_generated_inputs_remain_originless():
    text = """id: adaptive-input-origins
name: adaptive-input-origins
input:
  topic: !str
nodes:
  ask:
    kind: human_input
    input:
      topic: ${input.topic}
    adaptive_questions:
      prompt: Ask about ${topic}
output: ${ask.output}
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    gate = loaded.compiled.nodes["ask"]
    composer = loaded.compiled.nodes["ask__compose"]

    authored = (
        document.root.child("nodes")
        .child("ask")
        .child("input")
        .child("topic")
    )
    _assert_input_site(
        loaded.origin_resolver,
        gate,
        loaded.compiled.wiring["ask"]["topic"],
        "topic",
        authored,
        source="${input.topic}",
    )

    assert _parameter(gate, "__questions").origin is None
    assert loaded.compiled.wiring["ask"]["__questions"].origin is None
    assert _parameter(composer, "topic").origin is None
    assert loaded.compiled.wiring["ask__compose"]["topic"].origin is None


def test_adaptive_rejects_an_authored_desugar_owned_input():
    text = """id: adaptive-input-collision
name: adaptive-input-collision
nodes:
  ask:
    kind: human_input
    input:
      __questions: authored
    adaptive_questions:
      prompt: Ask something
output: ${ask.output}
"""
    document = SourceDocument.parse(text)
    collision = (
        document.root.child("nodes")
        .child("ask")
        .child("input")
        .child("__questions")
    )

    with pytest.raises(LoadError, match="__questions.*reserved") as exc_info:
        load_flow(text)

    assert exc_info.value.diagnostic.frames[-1].range == collision.range


def test_map_over_and_wait_until_resolve_exact_value_sites_without_parameters():
    text = """id: reserved-input-origins
name: reserved-input-origins
input:
  items: !list(str)
  deadline: !str
defs:
  echo:
    input:
      x: !str
    nodes:
      worker:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.x}
        output: str
    output: ${worker.output}
nodes:
  mapping:
    kind: map
    call: echo
    over: ${input.items}
    input:
      x: ${item}
  waiting:
    kind: wait
    until: ${input.deadline}
output: ${mapping.output}
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    mapping = loaded.compiled.nodes["mapping"]
    mapping_input = (
        document.root.child("nodes")
        .child("mapping")
        .child("input")
        .child("x")
    )

    _assert_input_site(
        loaded.origin_resolver,
        mapping,
        loaded.compiled.wiring["mapping"]["x"],
        "x",
        mapping_input,
        source="${item}",
    )

    mapping_origin = _assert_binding_site(
        loaded.origin_resolver,
        loaded.compiled.wiring["mapping"]["over"],
        document.root.child("nodes").child("mapping").child("over"),
        source="${input.items}",
    )
    waiting_origin = _assert_binding_site(
        loaded.origin_resolver,
        loaded.compiled.wiring["waiting"]["until"],
        document.root.child("nodes").child("waiting").child("until"),
        source="${input.deadline}",
    )
    assert mapping_origin != waiting_origin
    assert {parameter.name for parameter in mapping.params or []} == {"x"}
    assert loaded.compiled.nodes["waiting"].params == []


def test_desugared_map_over_keeps_authored_value_site_with_final_source():
    text = """id: rewritten-reserved-origin
name: rewritten-reserved-origin
input:
  items: !list(str)
defs:
  echo:
    input:
      x: !str
    nodes:
      worker:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.x}
        output: str
    output: ${worker.output}
  produce:
    input:
      x: !list(str)
    nodes:
      worker:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.x}
        output: list(str)
    output: ${worker.output}
nodes:
  mapping:
    kind: map
    call: echo
    over: call(produce, x=${input.items})
    input:
      x: ${item}
output: ${mapping.output}
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    mapping = loaded.compiled.nodes["mapping"]
    wiring = loaded.compiled.wiring["mapping"]

    assert {name: binding.source for name, binding in wiring.items()} == {
        "over": "${__call_0.output}",
        "x": "${item}",
    }
    _assert_binding_site(
        loaded.origin_resolver,
        wiring["over"],
        document.root.child("nodes").child("mapping").child("over"),
        source="${__call_0.output}",
    )
    assert {parameter.name for parameter in mapping.params or []} == {"x"}


def test_root_def_and_inline_wait_origins_are_lexically_distinct():
    text = """id: lexical-reserved-origins
name: lexical-reserved-origins
defs:
  named:
    nodes:
      pause:
        kind: wait
        until: named-deadline
      worker:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        depends_on: [pause]
        input:
          x: named
        output: str
    output: ${worker.output}
nodes:
  pause:
    kind: wait
    until: root-deadline
  named_call:
    kind: call
    call: named
  inline_call:
    kind: call
    nodes:
      pause:
        kind: wait
        until: inline-deadline
      worker:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        depends_on: [pause]
        input:
          x: inline
        output: str
    output: ${worker.output}
output:
  named: ${named_call.output}
  inline: ${inline_call.output}
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    named_child = loaded.compiled.nodes["named_call"].child
    inline_child = loaded.compiled.nodes["inline_call"].child
    sites = (
        (
            loaded.compiled.nodes["pause"],
            loaded.compiled.wiring["pause"]["until"],
            document.root.child("nodes").child("pause").child("until"),
            "root-deadline",
        ),
        (
            named_child.nodes["pause"],
            named_child.wiring["pause"]["until"],
            document.root.child("defs")
            .child("named")
            .child("nodes")
            .child("pause")
            .child("until"),
            "named-deadline",
        ),
        (
            inline_child.nodes["pause"],
            inline_child.wiring["pause"]["until"],
            document.root.child("nodes")
            .child("inline_call")
            .child("nodes")
            .child("pause")
            .child("until"),
            "inline-deadline",
        ),
    )

    origins = set()
    for node, binding, cursor, source in sites:
        assert node.params == []
        origins.add(
            _assert_binding_site(
        loaded.origin_resolver,
                binding,
                cursor,
                source=source,
            )
        )
    assert len(origins) == 3


def test_virtual_reserved_wiring_does_not_mint_or_erase_origins():
    map_origin = OriginRef("programmatic-map-over")
    wait_origin = OriginRef("programmatic-wait-until")
    child = load_flow(
        """id: echo
name: echo
input:
  x: !str
nodes:
  worker:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: ${input.x}
    output: str
output: ${worker.output}
"""
    )

    def resolve_child(flow_id: str, version=None):
        assert flow_id == "echo"
        return child

    nodes_cursor = SourceCursor.virtual(
        {
            "mapping": {
                "kind": "map",
                "call": "echo",
                "over": "${input.items}",
                "inputs": {"x": "${item}"},
            },
            "waiting": {
                "kind": "wait",
                "until": "${input.deadline}",
            },
        }
    )
    loaded = _assemble(
        inputs_section={"items": "list(str)", "deadline": "str"},
        nodes_cursor=nodes_cursor,
        outputs_section="${mapping.output}",
        asserts_section=[],
        registry={},
        scope=_CallableScope({}, {}, resolve_child, {"echo": "echo"}),
        owner=("root",),
    )

    mapping = loaded.compiled.nodes["mapping"]
    waiting = loaded.compiled.nodes["waiting"]
    assert loaded.compiled.wiring["mapping"]["over"].origin is None
    assert loaded.compiled.wiring["waiting"]["until"].origin is None
    assert {parameter.name for parameter in mapping.params or []} == {"x"}
    assert waiting.params == []

    loaded.compiled.wiring["mapping"]["over"] = Binding(
        "${input.items}",
        origin=map_origin,
    )
    loaded.compiled.wiring["waiting"]["until"] = Binding(
        "${input.deadline}",
        origin=wait_origin,
    )
    _stamp_node_wiring_origins(
        parse_nodes(nodes_cursor),
        nodes_cursor,
        loaded.compiled.nodes,
        loaded.compiled.wiring,
        owner=("root",),
        catalog_builder=_SourceCatalogBuilder(),
        diagnostics=LoadDiagnostics.from_cursor(nodes_cursor),
    )
    assert loaded.compiled.wiring["mapping"]["over"].origin == map_origin
    assert loaded.compiled.wiring["waiting"]["until"].origin == wait_origin
