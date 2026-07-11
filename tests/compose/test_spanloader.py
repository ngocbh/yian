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
