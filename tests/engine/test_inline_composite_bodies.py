"""Compose lowering tests for inline CALL/MAP/LOOP child-flow bodies."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest
import yaml

from agent_composer.compile.model import END_ID, START_ID
from agent_composer.compose import loader as compose_loader
from agent_composer.compose.build import pipeline as compose_pipeline
from agent_composer.compose.build.pipeline import _register_inline_bodies
from agent_composer.compose.desugar import inline_bodies as inline_bodies_module
from agent_composer.compose.desugar.inline_bodies import (
    INLINE_FLOW_PREFIX,
    _overlay_node_occurrences,
    extract_inline_bodies,
    inline_flow_id,
)
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.compose.helpers.spanloader import load_with_spans
from agent_composer.compose.helpers.types import InputDecl
from agent_composer.compose.loader import load_flow
from agent_composer.compose.nodes.code import CodeDescriptor
from agent_composer.compose.parse.nodes import parse_nodes
from agent_composer.events import SourceRange
from agent_composer.run import run_flow
from agent_composer.typesys.values import Type, ValueKind


def _located_root(text: str) -> tuple[NodeReader, dict]:
    """Parse YAML text into a located root reader and expose its base span table."""
    value, spans, root = load_with_spans(text, return_root=True)
    return NodeReader(value, spans, root), spans


def _yaml_pairs(reader: NodeReader, key: str) -> list[tuple[yaml.Node, yaml.Node]]:
    """Return every composed pair for ``key`` in author/effective merge order."""
    return [
        (key_node, value_node)
        for key_node, value_node in reader.yaml_node.value
        if key_node.value == key
    ]


class _RecordingRegistrar:
    """Test double retaining private registration arguments without resolving them."""

    def __init__(self) -> None:
        self.registrations: list[tuple[str, tuple[str, ...], str, object]] = []

    def register_private(self, flow_id, owner, label, builder) -> None:
        """Record one lazy private builder without invoking it."""
        self.registrations.append((flow_id, owner, label, builder))


def _record_private_scope(monkeypatch: pytest.MonkeyPatch):
    """Record compiler-private registrations and resolutions for one load."""
    registrations = []
    resolutions = []
    original_register = compose_loader._CallableScope.register_private
    original_resolve = compose_loader._CallableScope.resolve

    def record_register(self, flow_id, owner, label, builder):
        registrations.append((self, flow_id, owner, label))
        return original_register(self, flow_id, owner, label, builder)

    def record_resolve(self, flow_id, version=None):
        if flow_id.startswith(INLINE_FLOW_PREFIX):
            resolutions.append((self, flow_id))
        return original_resolve(self, flow_id, version)

    monkeypatch.setattr(
        compose_loader._CallableScope,
        "register_private",
        record_register,
    )
    monkeypatch.setattr(
        compose_loader._CallableScope,
        "resolve",
        record_resolve,
    )
    return registrations, resolutions


def _association_fixture():
    """Return located retained bodies and their final CALL/MAP/LOOP descriptors."""
    root, _spans = _located_root(
        """nodes:
  call_node:
    kind: call
    nodes:
      echo:
        kind: code
        run: tests.engine._compose_codefns:echo
        input:
          value: hello
        output: str
    output: ${echo.output}
  map_node:
    kind: map
    over: ${input.items}
    nodes:
      echo:
        kind: code
        run: tests.engine._compose_codefns:echo
        input:
          value: hello
        output: str
    output: ${echo.output}
  loop_node:
    kind: loop
    times: 1
    input:
      value: 0
    nodes:
      echo:
        kind: code
        run: tests.engine._compose_codefns:echo
        input:
          value: hello
        output: str
    output: ${echo.output}
"""
    )
    masked, bodies = extract_inline_bodies(root.child("nodes"), owner=("root",))
    return parse_nodes(masked), bodies


def test_overlay_alias_wrappers_are_distinct_and_leave_the_base_unchanged():
    root, base_spans = _located_root(
        """
body: &body
  kind: code
  run: tests.engine._compose_codefns:echo
nodes:
  first: *body
  second: *body
"""
    )
    base = root.child("nodes")
    anchor = root.child("body")
    base_first = base.child("first")
    base_second = base.child("second")
    original_span_entries = dict(base_spans)
    original_pairs = tuple(anchor.yaml_node.value)

    overlay = _overlay_node_occurrences(base)
    first = overlay.child("first")
    second = overlay.child("second")

    assert base_first.value is anchor.value is base_second.value
    assert base_first.yaml_node is anchor.yaml_node is base_second.yaml_node
    assert first.value is not second.value
    assert first.value is not anchor.value
    assert second.value is not anchor.value
    assert first.yaml_node is not second.yaml_node
    assert first.yaml_node is not anchor.yaml_node
    assert second.yaml_node is not anchor.yaml_node
    assert first.yaml_node.value is not anchor.yaml_node.value
    assert second.yaml_node.value is not anchor.yaml_node.value
    assert all(
        copied_key is not original_key
        for (copied_key, _), (original_key, _) in zip(
            first.yaml_node.value, original_pairs, strict=True
        )
    )
    assert first._own_range() == anchor._own_range()
    assert second._own_range() == anchor._own_range()
    assert first.key_range_of("kind") == anchor.key_range_of("kind")
    assert second.key_range_of("run") == anchor.key_range_of("run")

    assert base_spans == original_span_entries
    assert id(first.yaml_node) not in base_spans
    assert id(first.yaml_node.value[0][0]) not in base_spans
    assert anchor.value == {
        "kind": "code",
        "run": "tests.engine._compose_codefns:echo",
    }
    assert tuple(anchor.yaml_node.value) == original_pairs


def test_overlay_merge_ranges_retain_expanded_keys_and_delegate_to_base():
    root, base_spans = _located_root(
        """
body: &body
  kind: code
  run: tests.engine._compose_codefns:echo
  output: str
nodes:
  merged:
    <<: *body
    input:
      value: hello
"""
    )
    anchor = root.child("body")
    base = root.child("nodes")
    base_merged = base.child("merged")
    original_span_entries = dict(base_spans)

    overlay = _overlay_node_occurrences(base)
    merged = overlay.child("merged")

    assert list(merged.value) == ["kind", "run", "output", "input"]
    assert merged.value == base_merged.value
    assert isinstance(merged.yaml_node, yaml.MappingNode)
    assert merged.yaml_node is not base_merged.yaml_node
    assert merged.yaml_node.value is not base_merged.yaml_node.value
    assert merged._own_range() == base_merged._own_range()

    for key in merged.value:
        copied_key, reused_value = merged._pair(key)
        original_key, original_value = base_merged._pair(key)
        assert copied_key is not original_key
        assert reused_value is original_value
        assert merged.key_range_of(key) == base_merged.key_range_of(key)
        assert merged.range_of(key) == base_merged.range_of(key)
        assert merged.spans.get(id(reused_value)) == base_spans.get(id(original_value))

    assert merged.key_range_of("kind") == anchor.key_range_of("kind")
    assert merged.range_of("output") == anchor.range_of("output")
    assert base_spans == original_span_entries
    assert id(merged.yaml_node) not in base_spans
    assert all(id(key_node) not in base_spans for key_node, _ in merged.yaml_node.value)


def test_extract_one_inline_body_masks_only_the_occurrence_and_retains_locations():
    root, base_spans = _located_root(
        """nodes:
  wrapper:
    kind: call
    input:
      text: ${input.text}
    nodes:
      echo:
        kind: code
        run: tests.engine._compose_codefns:echo
        input:
          value: ${input.text}
        output: str
    output: ${echo.output}
"""
    )
    original_nodes = root.child("nodes")
    original_body = original_nodes.child("wrapper")
    original_child_nodes = original_body.child("nodes")
    original_body_value = dict(original_body.value)
    original_pairs = tuple(original_body.yaml_node.value)
    original_span_entries = dict(base_spans)
    child_owner = ("root", "node", "wrapper", "inline")

    masked_nodes, bodies = extract_inline_bodies(original_nodes, owner=("root",))

    assert inline_flow_id(child_owner) == (
        INLINE_FLOW_PREFIX
        + "78882e0096fb763300120ec6c6428e700d2dd5094af58139242c79a1023a243a"
    )
    assert list(bodies) == ["wrapper"]
    body = bodies["wrapper"]
    assert body.node_id == "wrapper"
    assert body.owner == child_owner
    assert body.flow_id == inline_flow_id(child_owner)
    assert body.nodes_reader.value is original_child_nodes.value
    assert body.nodes_reader.yaml_node is original_child_nodes.yaml_node
    assert body.output == "${echo.output}"
    assert body.output_range == original_body.range_of("output")
    assert body.input_ranges == original_body.mapping_value_ranges("input")

    masked_body = masked_nodes.child("wrapper")
    assert masked_body.value == {
        "kind": "call",
        "input": {"text": "${input.text}"},
        "call": body.flow_id,
    }
    assert "nodes" not in masked_body.value
    assert "output" not in masked_body.value
    assert masked_body.range_of("call") == original_body.range_of("nodes")
    assert masked_body.child("call").yaml_node is original_child_nodes.yaml_node
    assert [key.value for key, _value in masked_body.yaml_node.value] == [
        "kind",
        "input",
        "call",
    ]

    assert original_body.value == original_body_value
    assert tuple(original_body.yaml_node.value) == original_pairs
    assert base_spans == original_span_entries


def test_extract_alias_occurrences_get_distinct_ids_and_share_retained_children():
    root, base_spans = _located_root(
        """
body: &body
  kind: call
  input:
    text: ${input.text}
  nodes:
    echo:
      kind: code
      run: tests.engine._compose_codefns:echo
      input:
        value: ${input.text}
      output: str
  output: ${echo.output}
nodes:
  first: *body
  second: *body
"""
    )
    original_nodes = root.child("nodes")
    anchor = root.child("body")
    anchor_children = anchor.child("nodes")
    original_anchor_value = dict(anchor.value)
    original_anchor_pairs = tuple(anchor.yaml_node.value)
    original_anchor_keys = [key.value for key, _value in anchor.yaml_node.value]
    original_span_entries = dict(base_spans)

    masked_nodes, bodies = extract_inline_bodies(original_nodes, owner=("root",))

    assert list(bodies) == ["first", "second"]
    assert bodies["first"].owner == ("root", "node", "first", "inline")
    assert bodies["second"].owner == ("root", "node", "second", "inline")
    assert bodies["first"].flow_id != bodies["second"].flow_id
    for body in bodies.values():
        assert body.nodes_reader.value is anchor_children.value
        assert body.nodes_reader.yaml_node is anchor_children.yaml_node
        assert body.output == "${echo.output}"
        assert body.output_range == anchor.range_of("output")

    first = masked_nodes.child("first")
    second = masked_nodes.child("second")
    assert first.value is not second.value
    assert first.yaml_node is not second.yaml_node
    assert first.value["call"] == bodies["first"].flow_id
    assert second.value["call"] == bodies["second"].flow_id
    assert first.range_of("call") == anchor.range_of("nodes")
    assert second.range_of("call") == anchor.range_of("nodes")
    assert [key.value for key, _value in first.yaml_node.value].count("call") == 1
    assert [key.value for key, _value in second.yaml_node.value].count("call") == 1

    assert anchor.value == original_anchor_value
    assert tuple(anchor.yaml_node.value) == original_anchor_pairs
    assert [key.value for key, _value in anchor.yaml_node.value] == original_anchor_keys
    assert base_spans == original_span_entries


def test_extract_merged_inline_body_retains_anchor_values_and_ranges():
    root, base_spans = _located_root(
        """
body: &body
  kind: call
  nodes:
    echo:
      kind: code
      run: tests.engine._compose_codefns:echo
      input:
        value: hello
      output: str
  output: ${echo.output}
nodes:
  merged:
    <<: *body
    input:
      text: hello
"""
    )
    anchor = root.child("body")
    anchor_children = anchor.child("nodes")
    original_nodes = root.child("nodes")
    merged_before = original_nodes.child("merged")
    original_merged_pairs = tuple(merged_before.yaml_node.value)
    original_merged_keys = [key.value for key, _value in merged_before.yaml_node.value]
    original_span_entries = dict(base_spans)

    masked_nodes, bodies = extract_inline_bodies(original_nodes, owner=("root",))

    body = bodies["merged"]
    assert body.nodes_reader.value is anchor_children.value
    assert body.nodes_reader.yaml_node is anchor_children.yaml_node
    assert body.output == "${echo.output}"
    assert body.output_range == anchor.range_of("output")

    masked = masked_nodes.child("merged")
    assert masked.value == {
        "kind": "call",
        "call": body.flow_id,
        "input": {"text": "hello"},
    }
    assert masked.range_of("call") == anchor.range_of("nodes")
    assert [key.value for key, _value in masked.yaml_node.value].count("call") == 1
    assert tuple(merged_before.yaml_node.value) == original_merged_pairs
    assert [key.value for key, _value in merged_before.yaml_node.value] == (
        original_merged_keys
    )
    assert base_spans == original_span_entries


def test_extract_merged_local_body_override_uses_effective_values_and_ranges():
    root, base_spans = _located_root(
        """
body: &body
  kind: call
  nodes:
    anchored:
      kind: code
      run: tests.engine._compose_codefns:echo
      input:
        value: anchored
      output: str
  output: ${anchored.output}
nodes:
  merged:
    <<: *body
    nodes:
      local:
        kind: code
        run: tests.engine._compose_codefns:echo
        input:
          value: local
        output: str
    output: ${local.output}
"""
    )
    anchor = root.child("body")
    original_nodes = root.child("nodes")
    merged_before = original_nodes.child("merged")
    nodes_pairs = _yaml_pairs(merged_before, "nodes")
    output_pairs = _yaml_pairs(merged_before, "output")
    assert len(nodes_pairs) == 2
    assert len(output_pairs) == 2
    local_nodes_node = nodes_pairs[-1][1]
    local_output_node = output_pairs[-1][1]
    local_nodes_range = base_spans[id(local_nodes_node)]
    local_output_range = base_spans[id(local_output_node)]
    original_merged_pairs = tuple(merged_before.yaml_node.value)
    original_anchor_pairs = tuple(anchor.yaml_node.value)
    original_merged_keys = [key.value for key, _value in merged_before.yaml_node.value]
    original_anchor_keys = [key.value for key, _value in anchor.yaml_node.value]
    original_span_entries = dict(base_spans)

    assert list(merged_before.value["nodes"]) == ["local"]
    assert merged_before.value["output"] == "${local.output}"

    masked_nodes, bodies = extract_inline_bodies(original_nodes, owner=("root",))

    body = bodies["merged"]
    assert body.nodes_reader.value is merged_before.value["nodes"]
    assert body.nodes_reader.yaml_node is local_nodes_node
    assert list(body.nodes_reader.value) == ["local"]
    assert body.output == "${local.output}"
    assert body.output_range == local_output_range

    masked = masked_nodes.child("merged")
    call_pairs = _yaml_pairs(masked, "call")
    assert masked.value == {"kind": "call", "call": body.flow_id}
    assert len(call_pairs) == 1
    assert call_pairs[0][1] is local_nodes_node
    assert masked.range_of("call") == local_nodes_range

    assert tuple(merged_before.yaml_node.value) == original_merged_pairs
    assert tuple(anchor.yaml_node.value) == original_anchor_pairs
    assert [key.value for key, _value in merged_before.yaml_node.value] == (
        original_merged_keys
    )
    assert [key.value for key, _value in anchor.yaml_node.value] == original_anchor_keys
    assert base_spans == original_span_entries


def test_extract_rejects_authored_private_call_target_without_mutating_source():
    root, base_spans = _located_root(
        f"""nodes:
  attacker:
    kind: call
    call: {INLINE_FLOW_PREFIX}not-generated
"""
    )
    original_nodes = root.child("nodes")
    attacker = original_nodes.child("attacker")
    original_value = dict(attacker.value)
    original_pairs = tuple(attacker.yaml_node.value)
    original_span_entries = dict(base_spans)

    with pytest.raises(LoadError) as caught:
        extract_inline_bodies(original_nodes, owner=("root",))

    assert "reserved" in str(caught.value).lower()
    assert INLINE_FLOW_PREFIX in str(caught.value)
    assert caught.value.range == attacker.range_of("call")
    assert attacker.value == original_value
    assert tuple(attacker.yaml_node.value) == original_pairs
    assert base_spans == original_span_entries


@pytest.mark.parametrize("attacker_first", [True, False], ids=["before", "after"])
def test_extract_rejects_sibling_private_id_attack_before_masking_any_body(
    attacker_first: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    victim_target = inline_flow_id(("root", "node", "victim", "inline"))
    attacker = f"""  attacker:
    kind: call
    call: {victim_target}
"""
    victim = """  victim:
    kind: call
    nodes:
      echo:
        kind: code
        run: tests.engine._compose_codefns:echo
        input:
          value: hello
        output: str
    output: ${echo.output}
"""
    first, second = (attacker, victim) if attacker_first else (victim, attacker)
    root, base_spans = _located_root("nodes:\n" + first + second)
    original_nodes = root.child("nodes")
    attacker_reader = original_nodes.child("attacker")
    victim_reader = original_nodes.child("victim")
    original_attacker = dict(attacker_reader.value)
    original_victim = dict(victim_reader.value)
    original_attacker_pairs = tuple(attacker_reader.yaml_node.value)
    original_victim_pairs = tuple(victim_reader.yaml_node.value)
    original_span_entries = dict(base_spans)
    mask_calls: list[str] = []
    original_mask = inline_bodies_module._mask_inline_body

    def track_mask(reader: NodeReader, flow_id: str) -> None:
        mask_calls.append(flow_id)
        original_mask(reader, flow_id)

    monkeypatch.setattr(inline_bodies_module, "_mask_inline_body", track_mask)

    with pytest.raises(LoadError) as caught:
        extract_inline_bodies(original_nodes, owner=("root",))

    assert "reserved" in str(caught.value).lower()
    assert INLINE_FLOW_PREFIX in str(caught.value)
    assert victim_target not in str(caught.value)
    assert caught.value.range == attacker_reader.range_of("call")
    assert mask_calls == []
    assert attacker_reader.value == original_attacker
    assert victim_reader.value == original_victim
    assert tuple(attacker_reader.yaml_node.value) == original_attacker_pairs
    assert tuple(victim_reader.yaml_node.value) == original_victim_pairs
    assert base_spans == original_span_entries


def test_extract_allows_private_prefix_in_graph_node_id():
    root, _spans = _located_root(
        f"""nodes:
  {INLINE_FLOW_PREFIX}host:
    kind: call
    call: named
"""
    )

    masked_nodes, bodies = extract_inline_bodies(
        root.child("nodes"), owner=("root",)
    )

    assert bodies == {}
    assert masked_nodes.value[f"{INLINE_FLOW_PREFIX}host"] == {
        "kind": "call",
        "call": "named",
    }


def test_loader_allows_private_callable_prefix_in_graph_node_id():
    loaded = load_flow(
        f"""id: prefix-node
name: prefix_node
input: {{x: str}}
nodes:
  {INLINE_FLOW_PREFIX}host:
    kind: code
    code: tests.seeds.fns:one_line_summary
    input:
      x: ${{input.x}}
output: ${{{INLINE_FLOW_PREFIX}host.output}}
"""
    )

    assert f"{INLINE_FLOW_PREFIX}host" in loaded.compiled.nodes


_INLINE_CHILD = """nodes:
      echo:
        kind: code
        run: tests.engine._compose_codefns:echo
        input:
          value: hello
        output: str
"""


_INLINE_SHAPE_CASES = [
    (
        "map",
        "call: named\n    " + _INLINE_CHILD + "    output: ${echo.output}\n",
        "nodes",
        "exactly one",
    ),
    ("call", "input: {}\n", None, "exactly one"),
    ("loop", _INLINE_CHILD, None, "requires `output:`"),
    ("call", "call: named\n    output: ${named.output}\n", "output", "only valid"),
    ("call", "nodes: {}\n    output: ${missing.output}\n", "nodes", "non-empty"),
    ("map", "nodes: not-a-map\n    output: value\n", "nodes", "mapping"),
    (
        "call",
        "input: not-a-map\n    " + _INLINE_CHILD + "    output: ${echo.output}\n",
        "input",
        "string keys",
    ),
    (
        "loop",
        "input:\n      1: value\n    "
        + _INLINE_CHILD
        + "    output: ${echo.output}\n",
        "input",
        "string keys",
    ),
]

_INLINE_SHAPE_IDS = [
    "both-call-and-nodes",
    "neither-call-nor-nodes",
    "inline-missing-output",
    "named-with-output",
    "empty-nodes",
    "non-mapping-nodes",
    "non-mapping-input",
    "non-string-input-key",
]


@pytest.mark.parametrize(
    ("kind", "body", "range_key", "message"),
    _INLINE_SHAPE_CASES,
    ids=_INLINE_SHAPE_IDS,
)
def test_inline_body_shape_errors_are_located_without_private_ids(
    kind: str,
    body: str,
    range_key: str | None,
    message: str,
):
    root, _spans = _located_root(
        "nodes:\n"
        "  composite:\n"
        f"    kind: {kind}\n"
        f"    {body}"
    )
    nodes_reader = root.child("nodes")
    raw_body = nodes_reader.child("composite")
    expected_range = raw_body.range_of(range_key) if range_key else raw_body._own_range()

    with pytest.raises(LoadError) as caught:
        extract_inline_bodies(nodes_reader, owner=("root",))

    assert message in str(caught.value)
    assert INLINE_FLOW_PREFIX not in str(caught.value)
    assert caught.value.range == expected_range


@pytest.mark.parametrize(
    ("kind", "body", "range_key", "message"),
    _INLINE_SHAPE_CASES,
    ids=_INLINE_SHAPE_IDS,
)
def test_inline_body_shape_matrix_flows_through_loader(
    kind: str,
    body: str,
    range_key: str | None,
    message: str,
):
    source = (
        "id: inline-shape-loader\n"
        "name: inline_shape_loader\n"
        "nodes:\n"
        "  composite:\n"
        f"    kind: {kind}\n"
        f"    {body}"
    )
    root, _spans = _located_root(source)
    raw_body = root.child("nodes").child("composite")
    expected_range = raw_body.range_of(range_key) if range_key else raw_body._own_range()

    with pytest.raises(LoadError) as caught:
        load_flow(source)

    assert message in str(caught.value)
    assert INLINE_FLOW_PREFIX not in str(caught.value)
    assert caught.value.range == expected_range


def test_loader_rejects_actual_sibling_private_digest_without_exposing_it():
    victim_id = inline_flow_id(("root", "node", "victim", "inline"))
    source = f"""id: inline-sibling-digest
name: inline_sibling_digest
nodes:
  victim:
    kind: call
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: safe
        output: str
    output: ${{echo.output}}
  attacker:
    kind: call
    call: {victim_id}
"""
    root, _spans = _located_root(source)
    expected_range = root.child("nodes").child("attacker").range_of("call")

    with pytest.raises(LoadError) as caught:
        load_flow(source)

    message = str(caught.value)
    assert "reserved" in message.lower()
    assert INLINE_FLOW_PREFIX in message
    assert victim_id not in message
    assert caught.value.range == expected_range


def test_register_inline_bodies_associates_call_map_loop_and_stays_lazy():
    descriptors, bodies = _association_fixture()
    originals = dict(descriptors)
    registrar = _RecordingRegistrar()
    build_calls: list[tuple[object, object]] = []

    def build_child(body, descriptor):
        build_calls.append((body, descriptor))
        return body.node_id

    associated = _register_inline_bodies(
        descriptors,
        bodies,
        registrar=registrar,
        build_child=build_child,
    )

    assert associated is not descriptors
    assert list(associated) == ["call_node", "map_node", "loop_node"]
    assert len(registrar.registrations) == 3
    assert build_calls == []

    for body, registration in zip(
        bodies.values(), registrar.registrations, strict=True
    ):
        flow_id, owner, label, builder = registration
        expected_label = f"inline body of node {body.node_id!r}"
        rewritten = associated[body.node_id]

        assert flow_id == body.flow_id
        assert owner == body.owner
        assert label == expected_label
        assert rewritten is not originals[body.node_id]
        assert rewritten == originals[body.node_id]
        assert rewritten.child_label == expected_label
        assert rewritten.call == body.flow_id
        assert originals[body.node_id] is descriptors[body.node_id]
        assert originals[body.node_id].child_label is None

        assert builder() == body.node_id
        assert build_calls[-1] == (body, rewritten)


@pytest.mark.parametrize(
    ("kind", "body"),
    [
        ("call", "call: named"),
        ("map", "call: named\n    over: ${input.items}"),
        ("loop", "call: named\n    times: 1"),
    ],
)
def test_child_label_is_not_authorable(kind: str, body: str):
    root, _spans = _located_root(
        "nodes:\n"
        "  composite:\n"
        f"    kind: {kind}\n"
        f"    {body}\n"
        "    child_label: forged\n"
    )
    node = root.child("nodes").child("composite")

    with pytest.raises(LoadError) as caught:
        parse_nodes(root.child("nodes"))

    assert "child_label" in str(caught.value)
    assert "not allowed" in str(caught.value)
    assert caught.value.range == node.range_of("child_label")


@pytest.mark.parametrize("failure", ["missing", "wrong-kind", "wrong-target"])
def test_register_inline_body_rejects_bad_association_located_and_redacted(
    failure: str,
):
    descriptors, bodies = _association_fixture()
    body = bodies["call_node"]
    candidate = {"call_node": descriptors["call_node"]}
    if failure == "missing":
        candidate = {}
    elif failure == "wrong-kind":
        candidate = {"call_node": object()}
    else:
        candidate = {
            "call_node": replace(descriptors["call_node"], call="named")
        }
    registrar = _RecordingRegistrar()

    with pytest.raises(LoadError) as caught:
        _register_inline_bodies(
            candidate,
            {"call_node": body},
            registrar=registrar,
            build_child=lambda _body, _descriptor: object(),
        )

    message = str(caught.value)
    assert "inline body of node 'call_node'" in message
    assert body.flow_id not in message
    assert INLINE_FLOW_PREFIX not in message
    assert caught.value.range == body.nodes_reader._own_range()
    assert registrar.registrations == []


def test_register_inline_bodies_preflights_all_before_first_registration():
    descriptors, bodies = _association_fixture()
    broken = dict(descriptors)
    del broken["loop_node"]
    registrar = _RecordingRegistrar()

    with pytest.raises(LoadError) as caught:
        _register_inline_bodies(
            broken,
            bodies,
            registrar=registrar,
            build_child=lambda _body, _descriptor: object(),
        )

    assert "inline body of node 'loop_node'" in str(caught.value)
    assert caught.value.range == bodies["loop_node"].nodes_reader._own_range()
    assert registrar.registrations == []


def test_zero_input_inline_call_loads_and_runs_in_one_file_scope(monkeypatch):
    registrations = []
    resolutions = []
    original_register = compose_loader._CallableScope.register_private
    original_resolve = compose_loader._CallableScope.resolve

    def record_register(self, flow_id, owner, label, builder):
        registrations.append((self, flow_id, owner, label))
        return original_register(self, flow_id, owner, label, builder)

    def record_resolve(self, flow_id, version=None):
        if flow_id.startswith(INLINE_FLOW_PREFIX):
            resolutions.append((self, flow_id))
        return original_resolve(self, flow_id, version)

    monkeypatch.setattr(
        compose_loader._CallableScope,
        "register_private",
        record_register,
    )
    monkeypatch.setattr(
        compose_loader._CallableScope,
        "resolve",
        record_resolve,
    )

    loaded = load_flow(
        """id: inline-call-zero-input
name: inline_call_zero_input
nodes:
  wrapper:
    kind: call
    nodes:
      constant:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: inline-ok
        output: str
    output: ${constant.output}
output: ${wrapper.output}
"""
    )

    expected_flow_id = inline_flow_id(
        ("root", "node", "wrapper", "inline")
    )
    wrapper = loaded.compiled.nodes["wrapper"]
    result = run_flow(loaded, {})

    assert wrapper.flow_id == expected_flow_id
    assert wrapper.child_inputs == []
    assert len(registrations) == 1
    assert registrations[0][0] is resolutions[0][0]
    assert registrations[0][1:] == (
        expected_flow_id,
        ("root", "node", "wrapper", "inline"),
        "inline body of node 'wrapper'",
    )
    assert resolutions == [(registrations[0][0], expected_flow_id)]
    assert result.status == "succeeded"
    assert result.output == "inline-ok"


@pytest.mark.parametrize("value", [None, 7])
def test_typed_inline_call_preserves_nullable_parent_scalar(value):
    loaded = load_flow(
        """id: inline-call-nullable-input
name: inline_call_nullable_input
input:
  maybe: Optional[int]
nodes:
  wrapper:
    kind: call
    input:
      value: ${input.maybe}
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.value}
        output: Optional[int]
    output: ${echo.output}
output: ${wrapper.output}
"""
    )

    wrapper = loaded.compiled.nodes["wrapper"]
    assert wrapper.child_inputs == [
        InputDecl(
            name="value",
            type_str="Optional[int]",
            default=None,
            required=False,
            type=Type(kind=ValueKind.INTEGER, nullable=True),
        )
    ]

    result = run_flow(loaded, {"maybe": value})
    assert result.status == "succeeded"
    assert result.output == value


def test_typed_inline_call_canonicalizes_literal_record_and_retains_input_order():
    loaded = load_flow(
        """id: inline-call-record-input
name: inline_call_record_input
nodes:
  wrapper:
    kind: call
    input:
      payload:
        z: tail
        a: 3
      marker: before
    nodes:
      echo:
        kind: code
        code: |
          return {"payload": inputs["payload"], "marker": inputs["marker"]}
        input:
          payload: ${input.payload}
          marker: ${input.marker}
        output:
          payload:
            a: int
            z: str
          marker: str
    output: ${echo.output}
output: ${wrapper.output}
"""
    )

    wrapper = loaded.compiled.nodes["wrapper"]
    assert [decl.name for decl in wrapper.child_inputs] == ["payload", "marker"]
    payload, marker = wrapper.child_inputs
    assert payload.type_str == '{"a":int,"z":str}'
    assert payload.default is None
    assert payload.required is True
    assert list(payload.type.fields or {}) == ["a", "z"]
    assert payload.type == Type(
        kind=ValueKind.OBJECT,
        fields={
            "a": Type.scalar(ValueKind.INTEGER),
            "z": Type.scalar(ValueKind.STRING),
        },
        required=frozenset({"a", "z"}),
    )
    assert marker.type_str == "str"
    assert marker.default is None
    assert marker.required is True
    assert marker.type == Type.scalar(ValueKind.STRING)

    result = run_flow(loaded, {})
    assert result.status == "succeeded"
    assert result.output == {
        "payload": {"z": "tail", "a": 3},
        "marker": "before",
    }


def test_forward_producer_type_builds_once_and_keeps_source_order(monkeypatch):
    builds = []
    original_build = CodeDescriptor.build

    def record_producer_build(self, ctx):
        if self.id == "producer":
            builds.append(self.id)
        return original_build(self, ctx)

    monkeypatch.setattr(CodeDescriptor, "build", record_producer_build)
    loaded = load_flow(
        """id: inline-call-forward-producer
name: inline_call_forward_producer
nodes:
  consumer:
    kind: call
    input:
      value: ${producer.output}
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.value}
        output: str
    output: ${echo.output}
  producer:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: forward-value
    output: str
output: ${consumer.output}
"""
    )

    consumer = loaded.compiled.nodes["consumer"]
    body_order = [
        node_id
        for node_id in loaded.compiled.nodes
        if node_id not in {START_ID, END_ID}
    ]
    result = run_flow(loaded, {})

    assert builds == ["producer"]
    assert body_order == ["consumer", "producer"]
    assert consumer.child_inputs == [
        InputDecl(
            name="value",
            type_str="str",
            default=None,
            required=True,
            type=Type.scalar(ValueKind.STRING),
        )
    ]
    assert result.status == "succeeded"
    assert result.output == "forward-value"


def test_producer_declared_none_type_is_not_treated_as_missing_output():
    loaded = load_flow(
        """id: inline-call-none-producer
name: inline_call_none_producer
nodes:
  consumer:
    kind: call
    input:
      value: ${producer.output}
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.value}
        output: "None"
    output: ${echo.output}
  producer:
    kind: code
    code: |
      return None
    output: "None"
output: ${consumer.output}
"""
    )

    consumer = loaded.compiled.nodes["consumer"]
    result = run_flow(loaded, {})

    assert consumer.child_inputs == [
        InputDecl(
            name="value",
            type_str="None",
            default=None,
            required=True,
            type=Type.scalar(ValueKind.NONE),
        )
    ]
    assert result.status == "succeeded"
    assert result.output is None


def test_inline_signature_self_cycle_uses_binding_path_and_range():
    source = """id: inline-signature-self-cycle
name: inline_signature_self_cycle
nodes:
  first:
    kind: call
    input:
      value: ${first.output}
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.value}
        output: str
    output: ${echo.output}
output: ${first.output}
"""
    root, _spans = _located_root(source)
    expected_range = (
        root.child("nodes").child("first").mapping_value_ranges("input")["value"]
    )

    with pytest.raises(LoadError) as caught:
        load_flow(source)

    message = str(caught.value)
    assert message.endswith("first.input.value -> first")
    assert INLINE_FLOW_PREFIX not in message
    assert caught.value.range == expected_range
    assert caught.value.line == expected_range.start_line
    assert not isinstance(caught.value, RecursionError)


def test_inline_signature_mutual_cycle_uses_closing_binding_range():
    source = """id: inline-signature-mutual-cycle
name: inline_signature_mutual_cycle
nodes:
  left:
    kind: call
    input:
      from_right: ${right.output}
    nodes:
      echo_left:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.from_right}
        output: str
    output: ${echo_left.output}
  right:
    kind: call
    input:
      from_left: ${left.output}
    nodes:
      echo_right:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.from_left}
        output: str
    output: ${echo_right.output}
output: ${left.output}
"""
    root, _spans = _located_root(source)
    expected_range = (
        root.child("nodes").child("right")
        .mapping_value_ranges("input")["from_left"]
    )

    with pytest.raises(LoadError) as caught:
        load_flow(source)

    message = str(caught.value)
    assert message.endswith(
        "left.input.from_right -> right.input.from_left -> left"
    )
    assert INLINE_FLOW_PREFIX not in message
    assert caught.value.range == expected_range
    assert caught.value.line == expected_range.start_line
    assert not isinstance(caught.value, RecursionError)


def test_inline_signature_unknown_producer_is_located_and_strict():
    source = """id: inline-signature-unknown-producer
name: inline_signature_unknown_producer
nodes:
  consumer:
    kind: call
    input:
      value: ${missing.output}
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.value}
        output: str
    output: ${echo.output}
output: ${consumer.output}
"""
    root, _spans = _located_root(source)
    expected_range = (
        root.child("nodes").child("consumer")
        .mapping_value_ranges("input")["value"]
    )

    with pytest.raises(LoadError) as caught:
        load_flow(source)

    message = str(caught.value)
    assert "consumer.input.value" in message
    assert "producer 'missing' has no available static type" in message
    assert "explicitly typed named `defs:` body" in message
    assert INLINE_FLOW_PREFIX not in message
    assert caught.value.range == expected_range


def test_inline_signature_producer_without_output_is_located_and_strict():
    source = """id: inline-signature-untyped-producer
name: inline_signature_untyped_producer
nodes:
  consumer:
    kind: call
    input:
      value: ${producer.output}
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.value}
        output: str
    output: ${echo.output}
  producer:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: no-static-output
output: ${consumer.output}
"""
    root, _spans = _located_root(source)
    expected_range = (
        root.child("nodes").child("consumer")
        .mapping_value_ranges("input")["value"]
    )

    with pytest.raises(LoadError) as caught:
        load_flow(source)

    message = str(caught.value)
    assert "consumer.input.value" in message
    assert "producer 'producer' has no available static type" in message
    assert "explicitly typed named `defs:` body" in message
    assert INLINE_FLOW_PREFIX not in message
    assert caught.value.range == expected_range


def test_inline_alias_and_merge_occurrences_run_as_distinct_private_children():
    loaded = load_flow(
        """id: inline-alias-merge
name: inline_alias_merge
nodes:
  first: &body
    kind: call
    input:
      x: base
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.x}
        output: str
    output: ${echo.output}
  second: *body
  merged:
    <<: *body
    input:
      x: merged
output:
  first: ${first.output}
  second: ${second.output}
  merged: ${merged.output}
"""
    )

    calls = [loaded.compiled.nodes[name] for name in ("first", "second", "merged")]
    assert len({node.flow_id for node in calls}) == 3
    assert len({id(node.child) for node in calls}) == 3

    result = run_flow(loaded, {})
    assert result.status == "succeeded"
    assert result.output == {"first": "base", "second": "base", "merged": "merged"}


def test_inline_literal_containers_are_inert_at_edges_and_runtime():
    loaded = load_flow(
        """id: inline-literal-containers
name: inline_literal_containers
nodes:
  producer:
    kind: code
    code: |
      return "resolved"
    output: str
  wrapper:
    kind: call
    input:
      payload:
        producer: ${producer.output}
        item: ${item.x}
        parent: ${input.x}
        dollars: $$
        deep:
          - ${producer.output}
          - ${item.x}
    nodes:
      echo:
        kind: code
        code: |
          return inputs["payload"]
        input:
          payload: ${input.payload}
        output:
          producer: str
          item: str
          parent: str
          dollars: str
          deep: list[str]
    output: ${echo.output}
output: ${wrapper.output}
"""
    )

    assert not any(
        edge.from_ == "producer" and edge.to == "wrapper"
        for edge in loaded.compiled.edges
    )
    result = run_flow(loaded, {})
    assert result.status == "succeeded"
    assert result.output == {
        "producer": "${producer.output}",
        "item": "${item.x}",
        "parent": "${input.x}",
        "dollars": "$$",
        "deep": ["${producer.output}", "${item.x}"],
    }


def test_inline_optional_record_paths_make_generated_inputs_nullable():
    loaded = load_flow(
        """id: inline-optional-record-paths
name: inline_optional_record_paths
typedefs:
  Inner:
    value: str
  Outer:
    inner: Optional[Inner]
input:
  root: Optional[Inner]
  outer: Outer
nodes:
  wrapper:
    kind: call
    input:
      root_value: ${input.root.value}
      nested_value: ${input.outer.inner.value}
    nodes:
      collect:
        kind: code
        code: |
          return {"root": inputs["root_value"], "nested": inputs["nested_value"]}
        input:
          root_value: ${input.root_value}
          nested_value: ${input.nested_value}
        output:
          root: Optional[str]
          nested: Optional[str]
    output: ${collect.output}
output: ${wrapper.output}
"""
    )

    wrapper = loaded.compiled.nodes["wrapper"]
    assert [
        (decl.name, decl.type_str, decl.required, decl.type.nullable)
        for decl in wrapper.child_inputs
    ] == [
        ("root_value", "Optional[str]", False, True),
        ("nested_value", "Optional[str]", False, True),
    ]

    for inputs, expected in [
        ({"root": None, "outer": {"inner": None}}, {"root": None, "nested": None}),
        (
            {
                "root": {"value": "root"},
                "outer": {"inner": {"value": "nested"}},
            },
            {"root": "root", "nested": "nested"},
        ),
    ]:
        result = run_flow(loaded, inputs)
        assert result.status == "succeeded"
        assert result.output == expected


def test_inline_multinode_body_matches_named_def_signature_and_runtime():
    loaded = load_flow(
        """id: inline-named-equivalence
name: inline_named_equivalence
defs:
  named:
    input:
      x: int
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.x}
        output: int
      pack:
        kind: code
        code: |
          return {"value": inputs["value"]}
        input:
          value: ${echo.output}
        output:
          value: int
    output: ${pack.output}
nodes:
  named_call:
    kind: call
    call: named
    input:
      x: 5
  inline_call:
    kind: call
    input:
      x: 5
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.x}
        output: int
      pack:
        kind: code
        code: |
          return {"value": inputs["value"]}
        input:
          value: ${echo.output}
        output:
          value: int
    output: ${pack.output}
output:
  named: ${named_call.output}
  inline: ${inline_call.output}
"""
    )

    named = loaded.compiled.nodes["named_call"]
    inline = loaded.compiled.nodes["inline_call"]
    assert named.child_inputs == inline.child_inputs
    assert named.output_type == inline.output_type
    assert set(named.child.nodes) - {START_ID, END_ID} == {"echo", "pack"}
    assert set(inline.child.nodes) - {START_ID, END_ID} == {"echo", "pack"}

    result = run_flow(loaded, {})
    assert result.status == "succeeded"
    assert result.output == {"named": {"value": 5}, "inline": {"value": 5}}


@pytest.mark.parametrize(
    ("outer_nodes", "bad_ref", "namespace"),
    [
        (
            "  parent:\n"
            "    kind: code\n"
            "    code: |\n"
            "      return 'parent'\n"
            "    output: str\n",
            "${parent.output}",
            "parent",
        ),
        ("", "${item}", "item"),
    ],
    ids=["parent-node", "item"],
)
def test_inline_child_scope_is_closed_and_redacts_private_identity(
    outer_nodes: str,
    bad_ref: str,
    namespace: str,
):
    source = (
        "id: inline-closed-scope\n"
        "name: inline_closed_scope\n"
        "nodes:\n"
        f"{outer_nodes}"
        "  wrapper:\n"
        "    kind: call\n"
        "    nodes:\n"
        "      leak:\n"
        "        kind: code\n"
        "        code: tests.engine._compose_codefns:echo_x\n"
        "        input:\n"
        f"          x: {bad_ref}\n"
        "        output: str\n"
        "    output: ${leak.output}\n"
        "output: ${wrapper.output}\n"
    )
    root, _spans = _located_root(source)
    expected_range = (
        root.child("nodes").child("wrapper").child("nodes").child("leak")
        .mapping_value_ranges("input")["x"]
    )

    with pytest.raises(LoadError) as caught:
        load_flow(source)

    message = str(caught.value)
    assert f"unknown namespace {namespace!r}" in message
    assert INLINE_FLOW_PREFIX not in message
    assert caught.value.range == expected_range


def test_eager_def_inline_call_uses_raw_ranges_and_shared_scope(monkeypatch):
    registrations = []
    resolutions = []
    original_register = compose_loader._CallableScope.register_private
    original_resolve = compose_loader._CallableScope.resolve

    def record_register(self, flow_id, owner, label, builder):
        registrations.append((self, flow_id, owner, label))
        return original_register(self, flow_id, owner, label, builder)

    def record_resolve(self, flow_id, version=None):
        if flow_id.startswith(INLINE_FLOW_PREFIX):
            resolutions.append((self, flow_id))
        return original_resolve(self, flow_id, version)

    monkeypatch.setattr(
        compose_loader._CallableScope,
        "register_private",
        record_register,
    )
    monkeypatch.setattr(
        compose_loader._CallableScope,
        "resolve",
        record_resolve,
    )

    source = """id: eager-def-inline-call
name: eager_def_inline_call
defs:
  outer:
    nodes:
      wrapper:
        kind: call
        nodes:
          constant:
            kind: code
            code: tests.engine._compose_codefns:echo
            input:
              topic: inline-ok
            output: str
        output: ${constant.output}
    output: ${wrapper.output}
nodes:
  invoke:
    kind: call
    call: outer
output: ${invoke.output}
"""
    loaded = load_flow(source)

    expected_owner = ("def", "outer", "node", "wrapper", "inline")
    expected_flow_id = inline_flow_id(expected_owner)
    wrapper = loaded.compiled.nodes["invoke"].child.nodes["wrapper"]
    wrapper_line = source.splitlines().index("      wrapper:") + 1
    result = run_flow(loaded, {})

    assert wrapper.flow_id == expected_flow_id
    assert wrapper.child_inputs == []
    assert wrapper.range.start_line == wrapper_line
    assert len(registrations) == 1
    assert len(resolutions) == 1
    assert registrations[0] == (
        resolutions[0][0],
        expected_flow_id,
        expected_owner,
        "inline body of node 'wrapper'",
    )
    assert resolutions[0] == (registrations[0][0], expected_flow_id)
    assert result.status == "succeeded"
    assert result.output == "inline-ok"


def test_def_assembly_reader_fallback_restores_only_node_surfaces():
    literal = {"inputs": "literal-input", "outputs": "literal-output"}
    normalized = {
        "wrapper": {
            "kind": "call",
            "inputs": {"payload": literal},
            "nodes": {
                "constant": {
                    "kind": "code",
                    "inputs": {"topic": "inline-ok"},
                    "outputs": "str",
                    "env": {"literal": literal},
                }
            },
            "outputs": "${constant.output}",
        }
    }

    reader = compose_loader._def_assembly_reader("", "outer", normalized)

    wrapper = reader.value["wrapper"]
    constant = wrapper["nodes"]["constant"]
    assert reader.yaml_node is None
    assert "input" in wrapper and "inputs" not in wrapper
    assert "output" in wrapper and "outputs" not in wrapper
    assert "input" in constant and "inputs" not in constant
    assert "output" in constant and "outputs" not in constant
    assert wrapper["input"]["payload"] is literal
    assert constant["env"]["literal"] is literal
    assert literal == {
        "inputs": "literal-input",
        "outputs": "literal-output",
    }
    assert "inputs" in normalized["wrapper"]
    assert "outputs" in normalized["wrapper"]


def test_inline_map_opaque_over_without_direct_item_walk_loads_and_runs():
    loaded = load_flow(
        """id: inline-map-opaque-over-no-item
name: inline_map_opaque_over_no_item
input:
  primary: list[str]
  fallback: list[str]
  prefix: str
nodes:
  each:
    kind: map
    over: ${input.primary | input.fallback}
    input:
      constant: fixed
      parent: ${input.prefix}
      template: item ${item}
      literal:
        nested: ${item.value}
    nodes:
      collect:
        kind: code
        code: |
          return {
            "constant": inputs["constant"],
            "parent": inputs["parent"],
            "template": inputs["template"],
            "literal": inputs["literal"],
          }
        input:
          constant: ${input.constant}
          parent: ${input.parent}
          template: ${input.template}
          literal: ${input.literal}
        output:
          constant: str
          parent: str
          template: str
          literal:
            nested: str
    output: ${collect.output}
output: ${each.output}
"""
    )

    result = run_flow(
        loaded,
        {"primary": ["first", "second"], "fallback": ["unused"], "prefix": "P"},
    )

    assert result.status == "succeeded"
    assert result.output == [
        {
            "constant": "fixed",
            "parent": "P",
            "template": "item first",
            "literal": {"nested": "${item.value}"},
        },
        {
            "constant": "fixed",
            "parent": "P",
            "template": "item second",
            "literal": {"nested": "${item.value}"},
        },
    ]


@pytest.mark.parametrize("item_source", ["${item}", "${item.name}"])
def test_inline_map_direct_item_on_opaque_over_fails_at_requesting_binding(
    item_source: str,
):
    source = f"""id: inline-map-opaque-over-item
name: inline_map_opaque_over_item
input:
  primary: list[str]
  fallback: list[str]
nodes:
  each:
    kind: map
    over: ${{input.primary | input.fallback}}
    input:
      row: {item_source}
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${{input.row}}
        output: str
    output: ${{echo.output}}
output: ${{each.output}}
"""
    root, _spans = _located_root(source)
    each = root.child("nodes").child("each")
    expected_range = each.mapping_value_ranges("input")["row"]
    over_range = each.range_of("over")

    with pytest.raises(LoadError) as caught:
        load_flow(source)

    message = str(caught.value)
    assert "each.input.row" in message
    assert "explicitly typed named `defs:` body" in message
    assert "only typed inside an outer MAP input" not in message
    assert caught.value.range == expected_range
    assert caught.value.range != over_range
    assert INLINE_FLOW_PREFIX not in message


def test_inline_map_item_proof_is_lazy_and_cached(
    monkeypatch: pytest.MonkeyPatch,
):
    original_infer = compose_pipeline.infer_inline_binding_type
    over_requests: list[tuple[str, SourceRange | None]] = []

    def counting_infer(value, **kwargs):
        if value == "${input.items}":
            over_requests.append(
                (kwargs["binding_label"], kwargs.get("binding_range"))
            )
        return original_infer(value, **kwargs)

    monkeypatch.setattr(
        compose_pipeline,
        "infer_inline_binding_type",
        counting_infer,
    )

    load_flow(
        """id: inline-map-lazy-item-proof
name: inline_map_lazy_item_proof
typedefs:
  Row:
    name: str
input:
  items: list[Row]
  prefix: str
nodes:
  each:
    kind: map
    over: ${input.items}
    input:
      constant: fixed
      parent: ${input.prefix}
      template: prefix ${input.prefix}
      literal:
        nested: ${item.name}
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.constant}
        output: str
    output: ${echo.output}
output: ${each.output}
"""
    )
    assert over_requests == []

    direct_source = """id: inline-map-cached-item-proof
name: inline_map_cached_item_proof
typedefs:
  Row:
    name: str
input:
  items: list[Row]
nodes:
  each:
    kind: map
    over: ${input.items}
    input:
      row: ${item}
      name: ${item.name}
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.name}
        output: str
    output: ${echo.output}
output: ${each.output}
    """
    direct_root, _spans = _located_root(direct_source)
    expected_request_range = (
        direct_root.child("nodes").child("each")
        .mapping_value_ranges("input")["row"]
    )

    loaded = load_flow(direct_source)

    assert [decl.name for decl in loaded.compiled.nodes["each"].child_inputs] == [
        "row",
        "name",
    ]
    assert over_requests == [("each.input.row", expected_request_range)]


def test_inline_map_typed_producer_maps_whole_item_and_field_in_parallel_order():
    loaded = load_flow(
        """id: inline-map-typed-records
name: inline_map_typed_records
typedefs:
  Row:
    name: str
    position: int
input:
  rows: list[Row]
nodes:
  prepare:
    kind: code
    code: |
      return inputs["rows"]
    input:
      rows: ${input.rows}
    output: list[Row]
  each:
    kind: map
    parallel: true
    over: ${prepare.output}
    input:
      row: ${item}
      name: ${item.name}
    nodes:
      render:
        kind: code
        code: |
          return {"row": inputs["row"], "name": inputs["name"]}
        input:
          row: ${input.row}
          name: ${input.name}
        output:
          row: Row
          name: str
    output: ${render.output}
output: ${each.output}
"""
    )
    each = loaded.compiled.nodes["each"]

    assert each.parallel is True
    assert [decl.name for decl in each.child_inputs] == ["row", "name"]
    assert each.child_inputs[0].type == loaded.input[0].type.element
    assert each.child_inputs[1].type == Type.scalar(ValueKind.STRING)

    rows = [
        {"name": "third", "position": 3},
        {"name": "first", "position": 1},
        {"name": "second", "position": 2},
    ]
    result = run_flow(loaded, {"rows": rows}, num_workers=4)

    assert result.status == "succeeded"
    assert result.output == [
        {"row": row, "name": row["name"]}
        for row in rows
    ]


def test_inline_map_proves_list_any_date_element_and_runs():
    loaded = load_flow(
        """id: inline-map-date-items
name: inline_map_date_items
input:
  days: list[date]
nodes:
  each:
    kind: map
    over: ${input.days}
    input:
      day: ${item}
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.day}
        output: date
    output: ${echo.output}
output: ${each.output}
"""
    )
    days = [date(2026, 8, 15), date(2026, 8, 16)]

    assert loaded.input[0].type.kind == ValueKind.LIST_ANY
    assert loaded.input[0].type.element == Type.scalar(ValueKind.DATE)
    assert loaded.compiled.nodes["each"].child_inputs[0].type == Type.scalar(
        ValueKind.DATE
    )

    result = run_flow(loaded, {"days": days})
    assert result.status == "succeeded"
    assert result.output == ["2026-08-15", "2026-08-16"]


def test_inline_map_missing_list_element_fails_at_item_binding(
    monkeypatch: pytest.MonkeyPatch,
):
    source = """id: inline-map-missing-element
name: inline_map_missing_element
input:
  items: list[str]
nodes:
  each:
    kind: map
    over: ${input.items}
    input:
      value: ${item}
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.value}
        output: str
    output: ${echo.output}
output: ${each.output}
"""
    root, _spans = _located_root(source)
    expected_range = (
        root.child("nodes").child("each")
        .mapping_value_ranges("input")["value"]
    )

    monkeypatch.setattr(
        compose_pipeline,
        "read_flow_inputs",
        lambda _mapping, _registry: [
            InputDecl(
                name="items",
                type_str="list[Any]",
                default=None,
                required=True,
                type=Type(kind=ValueKind.LIST_ANY),
            )
        ],
    )

    with pytest.raises(LoadError) as caught:
        load_flow(source)

    assert "each.input.value" in str(caught.value)
    assert "no proven element type" in str(caught.value)
    assert caught.value.range == expected_range


def test_inline_map_nullable_root_and_intermediate_infer_optional_and_intermediate_runs_none():
    loaded = load_flow(
        """id: inline-map-nullable-items
name: inline_map_nullable_items
typedefs:
  Detail:
    name: str
  Row:
    detail: Optional[Detail]
input:
  items: list[Optional[Row]]
nodes:
  each:
    kind: map
    over: ${input.items}
    input:
      row: ${item}
      name: ${item.detail.name}
    nodes:
      collect:
        kind: code
        code: |
          return {"row": inputs["row"], "name": inputs["name"]}
        input:
          row: ${input.row}
          name: ${input.name}
        output:
          row: Optional[Row]
          name: Optional[str]
    output: ${collect.output}
output: ${each.output}
"""
    )
    decls = {
        decl.name: decl
        for decl in loaded.compiled.nodes["each"].child_inputs
    }

    assert decls["row"].required is False
    assert decls["row"].type.nullable is True
    assert decls["name"].required is False
    assert decls["name"].type == Type(
        kind=ValueKind.STRING,
        nullable=True,
    )

    items = [
        {"detail": None},
        {"detail": {"name": "ready"}},
    ]
    result = run_flow(loaded, {"items": items})

    assert result.status == "succeeded"
    assert result.output == [
        {"row": {"detail": None}, "name": None},
        {
            "row": {"detail": {"name": "ready"}},
            "name": "ready",
        },
    ]


def test_inline_map_rejects_outer_asserts_without_private_identity():
    with pytest.raises(LoadError) as caught:
        load_flow(
            """id: inline-map-asserts
name: inline_map_asserts
input:
  items: list[str]
nodes:
  each:
    kind: map
    over: ${input.items}
    input:
      value: ${item}
    asserts:
      - ${output} != null
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.value}
        output: str
    output: ${echo.output}
output: ${each.output}
"""
        )

    message = str(caught.value)
    assert "`asserts:` is not allowed on a mapped call" in message
    assert INLINE_FLOW_PREFIX not in message


def test_inline_map_child_keeps_item_namespace_closed():
    source = """id: inline-map-closed-child
name: inline_map_closed_child
input:
  items: list[str]
nodes:
  each:
    kind: map
    over: ${input.items}
    input:
      value: ${item}
    nodes:
      leak:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${item}
        output: str
    output: ${leak.output}
output: ${each.output}
"""
    root, _spans = _located_root(source)
    expected_range = (
        root.child("nodes").child("each").child("nodes").child("leak")
        .mapping_value_ranges("input")["x"]
    )

    with pytest.raises(LoadError) as caught:
        load_flow(source)

    message = str(caught.value)
    assert "unknown namespace 'item'" in message
    assert INLINE_FLOW_PREFIX not in message
    assert caught.value.range == expected_range


def test_basic_inline_loop_times_bakes_private_child_signature_and_runs():
    loaded = load_flow(
        """id: inline-loop-times
name: inline_loop_times
nodes:
  repeat:
    kind: loop
    times: 2
    input:
      n: 0
      exited: false
    nodes:
      bump:
        kind: code
        code: tests.engine._compose_codefns:loop_bump
        input:
          n: ${input.n}
          exited: ${input.exited}
        output:
          n: int
          exited: bool
    output: ${bump.output}
output: ${repeat.output}
"""
    )
    repeat = loaded.compiled.nodes["repeat"]
    expected_owner = ("root", "node", "repeat", "inline")
    expected_output = Type(
        kind=ValueKind.OBJECT,
        fields={
            "n": Type.scalar(ValueKind.INTEGER),
            "exited": Type.scalar(ValueKind.BOOLEAN),
        },
        required=frozenset({"n", "exited"}),
    )

    assert repeat.flow_id == inline_flow_id(expected_owner)
    assert repeat.child_inputs == [
        InputDecl(
            name="n",
            type_str="int",
            default=None,
            required=True,
            type=Type.scalar(ValueKind.INTEGER),
        ),
        InputDecl(
            name="exited",
            type_str="bool",
            default=None,
            required=True,
            type=Type.scalar(ValueKind.BOOLEAN),
        ),
    ]
    assert repeat.output_type == expected_output
    assert repeat.predicate_kind == "times"
    assert repeat.times == repeat.max_iters == 2

    result = run_flow(loaded, {})
    assert result.status == "succeeded"
    assert result.output == {"n": 2, "exited": False}


@pytest.mark.parametrize(
    ("predicate", "predicate_kind"),
    [
        ("while: not ${exited}", "while"),
        ("until: ${exited}", "until"),
    ],
)
def test_inline_loop_predicates_run_to_the_same_carried_result(
    predicate: str,
    predicate_kind: str,
):
    source = """id: inline-loop-predicate
name: inline_loop_predicate
nodes:
  repeat:
    kind: loop
    <predicate>
    max: 10
    input:
      n: 0
      exited: false
    nodes:
      bump:
        kind: code
        code: tests.engine._compose_codefns:loop_bump
        input:
          n: ${input.n}
          exited: ${input.exited}
        output:
          n: int
          exited: bool
    output: ${bump.output}
output: ${repeat.output}
""".replace("<predicate>", predicate)

    loaded = load_flow(source)
    repeat = loaded.compiled.nodes["repeat"]
    result = run_flow(loaded, {})

    assert repeat.predicate_kind == predicate_kind
    assert result.status == "succeeded"
    assert result.output == {"n": 3, "exited": True}


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (
            """      step:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.n}
        output: int
""",
            "body inline body of node 'repeat' must declare a record output",
        ),
        (
            """      step:
        kind: code
        code: |
          return {"value": inputs["n"]}
        input:
          n: ${input.n}
        output:
          value: int
""",
            "body output fields ['value'] must equal the carried record fields ['n']",
        ),
        (
            """      step:
        kind: code
        code: |
          return {"n": str(inputs["n"])}
        input:
          n: ${input.n}
        output:
          n: str
""",
            "carried field 'n' — body output is 'str', carried source is 'int'",
        ),
    ],
    ids=["scalar-output", "record-fields", "field-type"],
)
def test_inline_loop_contract_diagnostics_are_friendly_and_redacted(
    step: str,
    expected: str,
):
    source = (
        """id: inline-loop-bad-contract
name: inline_loop_bad_contract
input:
  seed: int
nodes:
  repeat:
    kind: loop
    times: 1
    input:
      n: ${input.seed}
    nodes:
"""
        + step
        + """    output: ${step.output}
output: ${repeat.output}
"""
    )

    with pytest.raises(LoadError) as caught:
        load_flow(source)

    message = str(caught.value)
    assert "loop node 'repeat'" in message
    assert expected in message
    assert INLINE_FLOW_PREFIX not in message


def test_nested_inline_composites_register_four_lexical_children_once(
    monkeypatch: pytest.MonkeyPatch,
):
    registrations, resolutions = _record_private_scope(monkeypatch)
    loaded = load_flow(
        """id: nested-inline-composites
name: nested_inline_composites
input:
  items: list[int]
nodes:
  outer_call:
    kind: call
    input:
      items: ${input.items}
    nodes:
      inner:
        kind: map
        over: ${input.items}
        input:
          value: ${item}
        nodes:
          echo:
            kind: code
            code: tests.engine._compose_codefns:echo_x
            input:
              x: ${input.value}
            output: int
        output: ${echo.output}
    output: ${inner.output}
  outer_loop:
    kind: loop
    times: 2
    input:
      n: 0
      exited: false
    nodes:
      inner:
        kind: call
        input:
          n: ${input.n}
          exited: ${input.exited}
        nodes:
          bump:
            kind: code
            code: tests.engine._compose_codefns:loop_bump
            input:
              n: ${input.n}
              exited: ${input.exited}
            output:
              n: int
              exited: bool
        output: ${bump.output}
    output: ${inner.output}
output:
  mapped: ${outer_call.output}
  looped: ${outer_loop.output}
"""
    )

    outer_call_owner = ("root", "node", "outer_call", "inline")
    inner_map_owner = (*outer_call_owner, "node", "inner", "inline")
    outer_loop_owner = ("root", "node", "outer_loop", "inline")
    inner_call_owner = (*outer_loop_owner, "node", "inner", "inline")
    expected = {
        inline_flow_id(outer_call_owner): (
            outer_call_owner,
            "inline body of node 'outer_call'",
        ),
        inline_flow_id(inner_map_owner): (
            inner_map_owner,
            "inline body of node 'inner'",
        ),
        inline_flow_id(outer_loop_owner): (
            outer_loop_owner,
            "inline body of node 'outer_loop'",
        ),
        inline_flow_id(inner_call_owner): (
            inner_call_owner,
            "inline body of node 'inner'",
        ),
    }
    expected_ids = set(expected)
    outer_call = loaded.compiled.nodes["outer_call"]
    inner_map = outer_call.child.nodes["inner"]
    outer_loop = loaded.compiled.nodes["outer_loop"]
    inner_call = outer_loop.child.nodes["inner"]

    assert len(expected_ids) == 4
    assert outer_call.flow_id == inline_flow_id(outer_call_owner)
    assert inner_map.flow_id == inline_flow_id(inner_map_owner)
    assert outer_loop.flow_id == inline_flow_id(outer_loop_owner)
    assert inner_call.flow_id == inline_flow_id(inner_call_owner)
    assert {
        outer_call.flow_id,
        inner_map.flow_id,
        outer_loop.flow_id,
        inner_call.flow_id,
    } == expected_ids
    assert len(registrations) == len(resolutions) == 4
    assert {
        flow_id: (owner, label)
        for _scope, flow_id, owner, label in registrations
    } == expected
    assert sorted(flow_id for _scope, flow_id in resolutions) == sorted(expected_ids)
    assert all(
        sum(entry[1] == flow_id for entry in registrations) == 1
        and sum(entry[1] == flow_id for entry in resolutions) == 1
        for flow_id in expected_ids
    )
    assert all(entry[0] is registrations[0][0] for entry in registrations)
    assert all(entry[0] is registrations[0][0] for entry in resolutions)

    registrations_at_load = tuple(registrations)
    resolutions_at_load = tuple(resolutions)
    result = run_flow(loaded, {"items": [3, 1, 2]}, num_workers=4)

    assert result.status == "succeeded"
    assert result.output == {
        "mapped": [3, 1, 2],
        "looped": {"n": 2, "exited": False},
    }
    assert tuple(registrations) == registrations_at_load
    assert tuple(resolutions) == resolutions_at_load


def test_eager_def_nested_inline_calls_use_exact_lexical_owners_once(
    monkeypatch: pytest.MonkeyPatch,
):
    registrations, resolutions = _record_private_scope(monkeypatch)
    loaded = load_flow(
        """id: eager-def-nested-inline-calls
name: eager_def_nested_inline_calls
defs:
  outer:
    nodes:
      outer_call:
        kind: call
        nodes:
          inner_call:
            kind: call
            nodes:
              constant:
                kind: code
                code: tests.engine._compose_codefns:echo
                input:
                  topic: nested-inline-ok
                output: str
            output: ${constant.output}
        output: ${inner_call.output}
    output: ${outer_call.output}
nodes:
  invoke:
    kind: call
    call: outer
output: ${invoke.output}
"""
    )

    outer_owner = ("def", "outer", "node", "outer_call", "inline")
    inner_owner = (*outer_owner, "node", "inner_call", "inline")
    expected = {
        inline_flow_id(outer_owner): (
            outer_owner,
            "inline body of node 'outer_call'",
        ),
        inline_flow_id(inner_owner): (
            inner_owner,
            "inline body of node 'inner_call'",
        ),
    }
    outer_call = loaded.compiled.nodes["invoke"].child.nodes["outer_call"]
    inner_call = outer_call.child.nodes["inner_call"]

    assert outer_call.flow_id == inline_flow_id(outer_owner)
    assert inner_call.flow_id == inline_flow_id(inner_owner)
    assert len(registrations) == len(resolutions) == 2
    assert {
        flow_id: (owner, label)
        for _scope, flow_id, owner, label in registrations
    } == expected
    assert sorted(flow_id for _scope, flow_id in resolutions) == sorted(expected)
    assert all(entry[0] is registrations[0][0] for entry in registrations)
    assert all(entry[0] is registrations[0][0] for entry in resolutions)
    assert all(
        sum(entry[1] == flow_id for entry in registrations) == 1
        and sum(entry[1] == flow_id for entry in resolutions) == 1
        for flow_id in expected
    )

    registrations_at_load = tuple(registrations)
    resolutions_at_load = tuple(resolutions)
    result = run_flow(loaded, {})

    assert result.status == "succeeded"
    assert result.output == "nested-inline-ok"
    assert tuple(registrations) == registrations_at_load
    assert tuple(resolutions) == resolutions_at_load


def test_inline_sibling_preserves_outer_and_top_frame_ranges():
    source = """id: inline-outer-frames
name: inline_outer_frames
nodes:
  before:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: start
    output: str
  wrapper:
    kind: call
    input:
      value: ${before.output}
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.value}
        output: str
    output: ${echo.output}
  after:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: ${wrapper.output}
    output: str
output: ${after.output}
"""
    root, _spans = _located_root(source)
    nodes = root.child("nodes")
    expected_nodes = {
        node_id: nodes.key_range_of(node_id)
        for node_id in ("before", "wrapper", "after")
    }
    expected_after_input = (
        nodes.child("after").mapping_value_ranges("input")["x"]
    )

    loaded = load_flow(source)
    top_nodes, top_fields = compose_loader.top_frame_ranges(source)
    repeated_nodes, repeated_fields = compose_loader.top_frame_ranges(source)

    assert loaded.source is not None
    assert loaded.source.node_ranges == expected_nodes
    assert loaded.source.field_ranges["after"][("input", "x")] == expected_after_input
    assert top_nodes == expected_nodes
    assert top_fields["after"][("input", "x")] == expected_after_input
    assert (repeated_nodes, repeated_fields) == (top_nodes, top_fields)


def test_inline_sibling_keeps_outer_bad_binding_range():
    source = """id: inline-outer-bad-binding
name: inline_outer_bad_binding
nodes:
  wrapper:
    kind: call
    nodes:
      constant:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ready
        output: str
    output: ${constant.output}
  broken:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: ${missing.output}
    output: str
output: ${broken.output}
"""
    root, _spans = _located_root(source)
    expected_range = (
        root.child("nodes").child("broken")
        .mapping_value_ranges("input")["x"]
    )

    with pytest.raises(LoadError) as caught:
        load_flow(source)

    assert "unknown namespace 'missing'" in str(caught.value)
    assert caught.value.range == expected_range


def test_inline_child_start_uses_outer_input_binding_ranges():
    source = """id: inline-child-input-ranges
name: inline_child_input_ranges
input:
  value: str
  count: int
nodes:
  wrapper:
    kind: call
    input:
      value: ${input.value}
      count: ${input.count}
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.value}
        output: str
    output: ${echo.output}
output: ${wrapper.output}
"""
    root, _spans = _located_root(source)
    expected_ranges = (
        root.child("nodes").child("wrapper").mapping_value_ranges("input")
    )

    loaded = load_flow(source)
    child_start = loaded.compiled.nodes["wrapper"].child.nodes[START_ID]

    assert {
        name: child_start.field_ranges[("input_decl", name)]
        for name in ("value", "count")
    } == expected_ranges


def test_inline_child_bad_output_uses_retained_output_range():
    source = """id: inline-child-output-range
name: inline_child_output_range
nodes:
  wrapper:
    kind: call
    nodes:
      constant:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ready
        output: str
    output: ${missing.output}
output: ${wrapper.output}
"""
    root, _spans = _located_root(source)
    expected_range = root.child("nodes").child("wrapper").range_of("output")

    with pytest.raises(LoadError) as caught:
        load_flow(source)

    assert "unknown namespace 'missing'" in str(caught.value)
    assert caught.value.range == expected_range


def test_eager_def_inline_child_bad_output_preserves_absolute_range():
    source = """id: eager-def-inline-output-range
name: eager_def_inline_output_range
defs:
  outer:
    nodes:
      wrapper:
        kind: call
        nodes:
          constant:
            kind: code
            code: tests.engine._compose_codefns:echo
            input:
              topic: ready
            output: str
        output: ${missing.output}
    output: ${wrapper.output}
nodes:
  invoke:
    kind: call
    call: outer
output: ${invoke.output}
"""
    root, _spans = _located_root(source)
    expected_range = (
        root.child("defs").child("outer").child("nodes")
        .child("wrapper").range_of("output")
    )

    with pytest.raises(LoadError) as caught:
        load_flow(source)

    assert str(caught.value).startswith("defs entry 'outer':")
    assert "unknown namespace 'missing'" in str(caught.value)
    assert caught.value.range == expected_range


def test_inline_map_output_error_uses_friendly_child_label():
    with pytest.raises(LoadError) as caught:
        load_flow(
            """id: inline-map-no-output
name: inline_map_no_output
input:
  items: list[str]
nodes:
  each:
    kind: map
    over: ${input.items}
    nodes:
      constant:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: inline-ok
        output: str
    output:
output: ${each.output}
"""
        )

    message = str(caught.value)
    assert "map node 'each': inline body of node 'each'" in message
    assert "must declare exactly one output value" in message
    assert INLINE_FLOW_PREFIX not in message


def test_callable_scope_builds_private_once_and_same_owner_registration_is_idempotent():
    flow_id = INLINE_FLOW_PREFIX + "opaque-once"
    owner = ("root", "node", "wrapper", "inline")
    child = object()
    built: list[str] = []

    def first_builder():
        built.append("first")
        return child

    def ignored_builder():
        built.append("ignored")
        return object()

    scope = compose_loader._CallableScope({}, object(), None, {})
    scope.register_private(flow_id, owner, "inline body `wrapper`", first_builder)
    scope.register_private(flow_id, owner, "ignored label", ignored_builder)

    assert built == []
    assert scope._private[flow_id].label == "inline body `wrapper`"
    assert scope._private[flow_id].builder is first_builder
    assert scope.resolve(flow_id) is child
    assert scope.resolve(flow_id, "ignored-version") is child
    assert built == ["first"]


def test_callable_scope_rejects_non_private_id_without_registering_it():
    flow_id = "ordinary_name"
    built: list[str] = []
    scope = compose_loader._CallableScope({}, object(), None, {})

    with pytest.raises(LoadError) as caught:
        scope.register_private(
            flow_id,
            ("root", "node", "wrapper", "inline"),
            "inline body `wrapper`",
            lambda: built.append("built"),
        )

    message = str(caught.value)
    assert "inline body `wrapper`" in message
    assert "wrapper" in message
    assert INLINE_FLOW_PREFIX in message
    assert flow_id not in message
    assert built == []

    with pytest.raises(LoadError, match="neither an in-file"):
        scope.resolve(flow_id)
    assert built == []


def test_callable_scope_rejects_private_owner_collision_without_exposing_id():
    flow_id = INLINE_FLOW_PREFIX + "opaque-owner-collision"
    scope = compose_loader._CallableScope({}, object(), None, {})
    scope.register_private(
        flow_id,
        ("root", "node", "first", "inline"),
        "inline body `first`",
        lambda: object(),
    )

    with pytest.raises(LoadError) as caught:
        scope.register_private(
            flow_id,
            ("root", "node", "second", "inline"),
            "inline body `second`",
            lambda: object(),
        )

    message = str(caught.value)
    assert "collision" in message
    assert "inline body `first`" in message
    assert "inline body `second`" in message
    assert "first" in message and "second" in message
    assert INLINE_FLOW_PREFIX in message
    assert flow_id not in message


@pytest.mark.parametrize("section", ["defs", "uses"])
def test_callable_scope_rejects_private_authored_collision_without_exposing_id(
    section: str,
):
    flow_id = INLINE_FLOW_PREFIX + "opaque-authored-collision"
    defs = {flow_id: {"nodes": {}}} if section == "defs" else {}
    uses = {flow_id: "library/child"} if section == "uses" else {}
    scope = compose_loader._CallableScope(defs, object(), None, uses)

    with pytest.raises(LoadError) as caught:
        scope.register_private(
            flow_id,
            ("root", "node", "wrapper", "inline"),
            "inline body `wrapper`",
            lambda: object(),
        )

    message = str(caught.value)
    assert "collision" in message
    assert f"{section}:" in message
    assert "inline body `wrapper`" in message
    assert "wrapper" in message
    assert INLINE_FLOW_PREFIX in message
    assert flow_id not in message


def test_callable_scope_reports_private_recursion_without_exposing_id():
    flow_id = INLINE_FLOW_PREFIX + "opaque-recursion"
    owner = ("root", "node", "wrapper", "inline")
    scope = compose_loader._CallableScope({}, object(), None, {})
    calls = 0

    def recursive_builder():
        nonlocal calls
        calls += 1
        return scope.resolve(flow_id)

    scope.register_private(
        flow_id, owner, "inline body `wrapper`", recursive_builder
    )

    with pytest.raises(LoadError) as caught:
        scope.resolve(flow_id)

    message = str(caught.value)
    assert "recursive" in message
    assert "inline body `wrapper`" in message
    assert "wrapper" in message
    assert INLINE_FLOW_PREFIX in message
    assert flow_id not in message
    assert calls == 1


def test_callable_scope_retries_failed_private_builder_and_caches_success():
    flow_id = INLINE_FLOW_PREFIX + "opaque-retry"
    child = object()
    attempts = 0

    def flaky_builder():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise LoadError("temporary child build failure")
        return child

    scope = compose_loader._CallableScope({}, object(), None, {})
    scope.register_private(
        flow_id,
        ("root", "node", "wrapper", "inline"),
        "inline body `wrapper`",
        flaky_builder,
    )

    with pytest.raises(LoadError, match="temporary child build failure"):
        scope.resolve(flow_id)

    assert scope.resolve(flow_id) is child
    assert scope.resolve(flow_id) is child
    assert attempts == 2


def test_callable_scope_reports_unregistered_private_prefix_without_exposing_id():
    flow_id = INLINE_FLOW_PREFIX + "opaque-unregistered"
    scope = compose_loader._CallableScope({}, object(), None, {})

    with pytest.raises(LoadError) as caught:
        scope.resolve(flow_id)

    message = str(caught.value)
    assert "reserved" in message
    assert "not registered" in message
    assert INLINE_FLOW_PREFIX in message
    assert flow_id not in message
    assert "neither an in-file" not in message


def test_callable_scope_eager_def_can_resolve_pre_registered_private(
    monkeypatch: pytest.MonkeyPatch,
):
    flow_id = INLINE_FLOW_PREFIX + "opaque-eager-def"
    body = {"nodes": {}}
    private_child = object()
    def_child = object()
    events: list[str] = []

    def build_private():
        events.append("private")
        return private_child

    def fake_load_def(name, value, registry, scope, parent_text=""):
        events.append(f"def:{name}")
        assert value is body
        assert scope.resolve(flow_id) is private_child
        return def_child

    monkeypatch.setattr(compose_loader, "_load_def", fake_load_def)
    scope = compose_loader._CallableScope({"outer": body}, object(), None, {})
    scope.register_private(
        flow_id,
        ("def", "outer", "node", "wrapper", "inline"),
        "inline body `wrapper` in defs:`outer`",
        build_private,
    )

    scope.eager_load_defs()

    assert events == ["def:outer", "private"]
    assert scope.resolve("outer") is def_child
    assert scope.resolve(flow_id) is private_child
