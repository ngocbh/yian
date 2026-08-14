"""Compose lowering tests for inline CALL/MAP/LOOP child-flow bodies."""

from __future__ import annotations

import yaml

from agent_composer.compose.desugar.inline_bodies import _overlay_node_occurrences
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.compose.helpers.spanloader import load_with_spans


def _located_root(text: str) -> tuple[NodeReader, dict]:
    """Parse YAML text into a located root reader and expose its base span table."""
    value, spans, root = load_with_spans(text, return_root=True)
    return NodeReader(value, spans, root), spans


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
