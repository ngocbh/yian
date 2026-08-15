"""Integration tests for authored Compose node-input origins."""

from __future__ import annotations

import pytest

from agent_composer.compose import load_flow
from agent_composer.compose.build.pipeline import _assemble
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.loader import _CallableScope
from agent_composer.compose.source import SourceCatalog, SourceCursor, SourceDocument
from agent_composer.diagnostics import OriginRef
from agent_composer.nodes.base import Node
from agent_composer.nodes.contracts import Binding, Parameter


def _parameter(node: Node, name: str) -> Parameter:
    """Return one exact compiled parameter by name."""
    matches = [parameter for parameter in node.params or [] if parameter.name == name]
    assert len(matches) == 1
    parameter = matches[0]
    assert type(parameter) is Parameter
    return parameter


def _assert_input_site(
    catalog: SourceCatalog,
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
    catalog = loaded._local_catalog

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
      x: str
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
        loaded._local_catalog,
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
  x: str
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
      x: int
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
        loaded._local_catalog,
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
        n_lines={},
        s_lines={},
    )

    node = loaded.compiled.nodes["worker"]
    binding = loaded.compiled.wiring["worker"]["x"]
    assert _parameter(node, "x").origin is None
    assert binding.source == "virtual"
    assert binding.origin is None


def test_input_origin_uses_the_same_normalized_occurrence_as_parsing():
    text = """id: normalized-input-origin
name: normalized-input-origin
nodes:
  worker:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: singular
    inputs:
      x: plural
    output: str
output: ${worker.output}
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    selected = (
        document.root.child("nodes")
        .child("worker")
        .child("inputs")
        .child("x")
    )

    _assert_input_site(
        loaded._local_catalog,
        loaded.compiled.nodes["worker"],
        loaded.compiled.wiring["worker"]["x"],
        "x",
        selected,
        source="plural",
    )


def test_adaptive_generated_inputs_remain_originless():
    text = """id: adaptive-input-origins
name: adaptive-input-origins
input:
  topic: str
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
        loaded._local_catalog,
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

    assert exc_info.value.range == collision.range


def test_map_over_and_wait_until_reserved_wiring_stay_originless():
    text = """id: reserved-input-origins
name: reserved-input-origins
input:
  items: list[str]
  deadline: str
defs:
  echo:
    inputs:
      x: str
    nodes:
      worker:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.x}
        output: str
    outputs: ${worker.output}
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
        loaded._local_catalog,
        mapping,
        loaded.compiled.wiring["mapping"]["x"],
        "x",
        mapping_input,
        source="${item}",
    )
    assert loaded.compiled.wiring["mapping"]["over"].origin is None
    assert loaded.compiled.wiring["waiting"]["until"].origin is None
    assert {parameter.name for parameter in mapping.params or []} == {"x"}
    assert loaded.compiled.nodes["waiting"].params == []
