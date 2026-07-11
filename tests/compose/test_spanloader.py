import yaml

from agent_composer.compose.spanloader import (
    SpanLoader,
    _mark_to_range,
    load_with_spans,
)


def test_mark_to_range_is_1based():
    node = yaml.compose("key: value")  # a MappingNode
    k, v = node.value[0]
    r = _mark_to_range(v.start_mark, v.end_mark)
    # PyYAML marks are 0-based line/col; range is 1-based.
    assert r.start_line == 1 and r.start_col == v.start_mark.column + 1
    # end mark maps directly (PyYAML end is already one-past-last, half-open) with the
    # same +1 shift: "value" ends at 0-based col 10 -> 1-based exclusive end col 11.
    assert r.end_line == 1 and r.end_col == v.end_mark.column + 1
    assert r.end_col == 11


def test_spanloader_maps_values_and_marks():
    text = "count: 30\nflag: yes\n"
    value, spans = load_with_spans(text)
    assert value == {"count": 30, "flag": True}      # coercion intact
    # a range exists for the top-level mapping node
    assert spans is not None
    assert len(spans) > 0


def test_spanloader_merge_keys_point_at_anchor():
    # S-new-3: a `<<:` merge key inherits its source location from the ANCHOR definition.
    # The anchor's key node is shared by identity between the anchor mapping (`base`) and
    # every mapping that merges it (`derived`), so its recorded range must point at the
    # anchor's own line, never the `<<: *base` line.
    #
    # This test descends the SAME loader's live nodes (yaml.compose would produce
    # different node identities that never appear in _span_marks) and asserts the merged
    # `timeout` key node is recorded with the anchor's line-2 range. It guards that the
    # merged pair is present in node.value at all (flatten_mapping ran) and that its mark
    # is the anchor site.
    #
    # HONEST SCOPE (verified during this change): this test does NOT fail if the whole
    # construct_mapping key/value loop is deleted, nor if it is moved before super().
    # Reason: the `timeout` key/value nodes are the anchor's own nodes, recorded when the
    # `base` mapping is constructed (and every scalar is also recorded by construct_scalar).
    # No span-content assertion can catch the loop's removal because it records nothing
    # that the other overrides don't already record. See the report accompanying this change.
    text = (
        "base: &base\n"
        "  timeout: 10\n"
        "derived:\n"
        "  <<: *base\n"
        "  name: x\n"
    )
    loader = SpanLoader(text)
    try:
        # get_single_node() first so we keep the root node's live identities; then
        # construct_document builds the value against those same nodes (load_with_spans
        # does not return the root yet — Step 3 will add return_root).
        root = loader.get_single_node()
        value = loader.construct_document(root)
        spans = loader._span_marks
    finally:
        loader.dispose()

    # merge worked: derived inherited timeout from the anchor, plus its own key
    assert value["derived"]["timeout"] == 10
    assert value["derived"]["name"] == "x"

    # descend to the `derived` value node (a MappingNode) and find its key nodes
    derived_node = None
    for key_node, value_node in root.value:
        if key_node.value == "derived":
            derived_node = value_node
            break
    assert derived_node is not None

    key_ranges = {}  # constructed key string -> recorded SourceRange of its key node
    for key_node, value_node in derived_node.value:
        assert id(key_node) in spans, f"key node {key_node.value!r} was not recorded"
        key_ranges[key_node.value] = spans[id(key_node)]

    # the merged `timeout` pair is present because flatten_mapping spliced it into
    # node.value; its key node is the anchor's, so the recorded range points at the anchor
    # definition line (`  timeout: 10`, 1-based line 2) — NOT the `<<: *base` line.
    assert "timeout" in key_ranges  # flatten_mapping ran and spliced the anchor pair
    assert key_ranges["timeout"].start_line == 2
    # `derived`'s own key sits on its own line (`  name: x`, line 5)
    assert key_ranges["name"].start_line == 5


def test_spanloader_records_sequence_and_scalar_ranges():
    text = "items:\n  - a\n  - b\nbare: 42\n"
    value, spans = load_with_spans(text)
    # list works as a value; scalar coercion intact
    assert value == {"items": ["a", "b"], "bare": 42}
    # The span tree is keyed by the loader's own node identities, which cannot be
    # reproduced from a separate parse — so assert on the recorded RANGES instead.
    ranges = set(spans.values())
    # sequence element scalars: "- a" at line 2 col 5, "- b" at line 3 col 5
    assert any(r.start_line == 2 and r.start_col == 5 for r in ranges)  # "a"
    assert any(r.start_line == 3 and r.start_col == 5 for r in ranges)  # "b"
    # the sequence node itself spans the two element lines (starts at line 2)
    assert any(r.start_line == 2 and r.end_line >= 3 for r in ranges)
    # bare scalar value "42" at line 4
    assert any(r.start_line == 4 and r.start_col == 7 for r in ranges)  # 42
    # plenty of ranges recorded overall
    assert len(spans) > 4
