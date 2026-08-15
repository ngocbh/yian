"""Compose lowering tests for inline CALL/MAP/LOOP child-flow bodies."""

from __future__ import annotations

import pytest
import yaml

from agent_composer.compose.desugar.inline_bodies import (
    INLINE_FLOW_PREFIX,
    _overlay_node_occurrences,
    extract_inline_bodies,
    inline_flow_id,
)
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.compose.helpers.spanloader import load_with_spans


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


_INLINE_CHILD = """nodes:
      echo:
        kind: code
        run: tests.engine._compose_codefns:echo
        input:
          value: hello
        output: str
"""


@pytest.mark.parametrize(
    ("kind", "body", "range_key", "message"),
    [
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
    ],
    ids=[
        "both-call-and-nodes",
        "neither-call-nor-nodes",
        "inline-missing-output",
        "named-with-output",
        "empty-nodes",
        "non-mapping-nodes",
        "non-mapping-input",
        "non-string-input-key",
    ],
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
