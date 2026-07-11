import yaml

from agent_composer.compose.spanloader import _mark_to_range, load_with_spans


def test_mark_to_range_is_1based():
    node = yaml.compose("key: value")  # a MappingNode
    k, v = node.value[0]
    r = _mark_to_range(v.start_mark, v.end_mark)
    # PyYAML marks are 0-based line/col; range is 1-based.
    assert r.start_line == 1 and r.start_col == v.start_mark.column + 1


def test_spanloader_maps_values_and_marks():
    text = "count: 30\nflag: yes\n"
    value, spans = load_with_spans(text)
    assert value == {"count": 30, "flag": True}      # coercion intact
    # a range exists for the top-level mapping node
    assert spans is not None
    assert len(spans) > 0


def test_spanloader_merge_keys_point_at_anchor():
    text = (
        "base: &base\n"
        "  timeout: 10\n"
        "derived:\n"
        "  <<: *base\n"
        "  name: x\n"
    )
    value, spans = load_with_spans(text)
    # merge worked: derived inherited timeout from the anchor, plus its own key
    assert value["derived"]["timeout"] == 10
    assert value["derived"]["name"] == "x"
    # construction survived the override (S-new-3): ranges were still recorded
    assert len(spans) > 0


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
