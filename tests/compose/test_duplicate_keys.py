"""A mapping that repeats one of its own keys is a load error."""

import pytest

from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow
from yian.compose.source import SourceDocument


_NODE_ID = """id: f
name: f
nodes:
  a:
    kind: code
    code: return 1
    output: int
  a:
    kind: code
    code: return 2
    output: int
output: ${a.output}
"""

_NODE_FIELD = """id: f
name: f
nodes:
  a:
    kind: code
    code: return 1
    code: return 2
    output: int
output: ${a.output}
"""

_TOP_LEVEL = """id: f
name: f
nodes:
  a:
    kind: code
    code: return 1
    output: int
output: ${a.output}
output: ${a.output}
"""

_FLOW_STYLE = """id: f
name: f
nodes:
  a:
    kind: code
    input: {x: 1, x: 2}
    code: return inputs["x"]
    output: int
output: ${a.output}
"""


@pytest.mark.parametrize(
    ("text", "key", "first_line", "second_line"),
    [
        (_NODE_ID, "a", 4, 8),
        (_NODE_FIELD, "code", 6, 7),
        (_TOP_LEVEL, "output", 8, 9),
        (_FLOW_STYLE, "x", 6, 6),
    ],
    ids=["node-id", "node-field", "top-level", "flow-style"],
)
def test_repeated_key_is_a_load_error_at_the_second_pair(
    text, key, first_line, second_line
):
    with pytest.raises(LoadError) as caught:
        load_flow(text, source_label="dup.yaml")

    message = str(caught.value)
    assert f"duplicate key {key!r}" in message
    assert f"first at line {first_line}" in message
    assert caught.value.diagnostic.frames[-1].range.start_line == second_line


def test_merged_keys_may_be_overridden():
    text = """id: f
name: f
x-base: &base
  kind: code
  code: return 1
  output: int
x-other: &other
  code: return 3
nodes:
  a:
    <<: [*base, *other]
    code: return 2
output: ${a.output}
"""

    loaded = load_flow(text)

    assert "a" in loaded.compiled.nodes


def test_a_shared_anchor_reports_its_duplicate_once_at_the_definition():
    text = """id: f
name: f
x-base: &base
  kind: code
  code: return 1
  code: return 2
  output: int
nodes:
  a: *base
  b: *base
output: ${a.output}
"""

    with pytest.raises(LoadError, match="duplicate key 'code'") as caught:
        load_flow(text)

    assert caught.value.diagnostic.frames[-1].range.start_line == 6


def test_keys_that_read_as_the_same_integer_fail_as_non_text_first():
    text = """id: f
name: f
x-meta:
  017: a
  17: b
nodes:
  a:
    kind: code
    code: return 1
    output: int
output: ${a.output}
"""

    with pytest.raises(LoadError) as caught:
        load_flow(text, source_label="dup.yaml")

    assert "the key `017` was read by YAML as an integer" in str(caught.value)
    assert caught.value.diagnostic.frames[-1].range.start_line == 4
    repeated = SourceDocument.parse(text).repeated_key
    assert repeated is not None
    assert repeated[0] == 17
    assert (repeated[1].start_line, repeated[2].start_line) == (4, 5)


def test_on_and_true_are_two_keys_and_true_is_not_text():
    text = """id: f
name: f
x-meta:
  on: a
  true: b
nodes:
  a:
    kind: code
    code: return 1
    output: int
output: ${a.output}
"""

    with pytest.raises(LoadError) as caught:
        load_flow(text)

    assert "the key `true` was read by YAML as a boolean" in str(caught.value)
    assert caught.value.diagnostic.frames[-1].range.start_line == 5
    meta = SourceDocument.parse(text).root.child("x-meta").value
    assert meta == {"on": "a", True: "b"}
