import yaml

from agent_composer.compose.spanloader import _mark_to_range


def test_mark_to_range_is_1based():
    node = yaml.compose("key: value")  # a MappingNode
    k, v = node.value[0]
    r = _mark_to_range(v.start_mark, v.end_mark)
    # PyYAML marks are 0-based line/col; range is 1-based.
    assert r.start_line == 1 and r.start_col == v.start_mark.column + 1
